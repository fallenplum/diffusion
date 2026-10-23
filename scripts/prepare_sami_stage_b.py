"""Prepare signed external programs and spatial context without model fitting.

All rows follow the already locked Stage A primary analysis exactly. Original
counts remain read-only. Program directions are external definitions, not
learned from this cohort. Tissue categories are not cell-source ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.sami_surv.annotations import read_json, require, sha256
from src.phase22_spatial_bridge.data_contract import read_h5ad_raw

PROGRAM_ORDER = ("iCMS2", "iCMS3", "FAP_fibroblast_associated_top25", "MARCO_macrophage_associated_top25")


def validate_programs(record, require_locked_order=True):
    require(record.get("schema_version") == "sami_surv_stage_b_programs_v1", "unrecognized external program schema")
    programs = record.get("programs")
    require(isinstance(programs, list) and programs, "nonempty external program list required")
    ids = [p.get("id") for p in programs]
    require(all(isinstance(p, str) and p for p in ids) and len(ids) == len(set(ids)), "invalid duplicate program identifiers")
    if require_locked_order:
        require(tuple(ids) == PROGRAM_ORDER, "locked four-program identity/order mismatch")
    for p in programs:
        up, down = p.get("up_genes"), p.get("down_genes")
        require(isinstance(up, list) and len(up) >= 5 and isinstance(down, list)
                and (not down or len(down) >= 5), "invalid fixed signed program directions")
        genes = up + down
        require(all(isinstance(g, str) and g and g == g.strip() for g in genes)
                and len(genes) == len(set(genes)) and p.get("genes") == genes,
                "external genes must equal exact unique up-plus-down symbols")
        for key in ("source_url", "source_table", "source_file_sha256", "source_hypothesis", "limitations"):
            require(key in p and bool(p[key]), "program lacks external provenance or interpretation boundary")
    return programs


def signed_program_scores(counts, gene_names, programs, *, scale=10000., minimum_coverage=.95,
                          minimum_direction_genes=5):
    """Exact measured-name coverage; full-library float64 CP10K log1p means."""
    genes = np.asarray(gene_names).astype(str)
    require(genes.ndim == 1 and len(genes) == len(set(genes)), "duplicate or malformed assay gene names")
    require(counts.ndim == 2 and counts.shape[1] == len(genes), "count/gene dimensions differ")
    raw = counts.data if sparse.issparse(counts) else np.asarray(counts)
    require(np.isfinite(raw).all() and np.all(raw >= 0) and np.all(raw == np.floor(raw)),
            "program targets require finite nonnegative integer counts")
    count64 = counts.astype(np.float64)
    library = np.asarray(count64.sum(axis=1)).ravel()
    require(np.isfinite(library).all() and np.all(library > 0) and np.all(library <= 2**53 - 1),
            "invalid or empty full-gene library size")
    if sparse.issparse(count64):
        detected = np.asarray((count64 > 0).sum(axis=1)).ravel().astype(np.int64)
    else:
        detected = (count64 > 0).sum(axis=1).astype(np.int64)
    lookup = {g: i for i, g in enumerate(genes)}
    targets, coverage = [], []
    for program in programs:
        up, down = program["up_genes"], program["down_genes"]
        source_genes = up + down
        require(source_genes and len(source_genes) == len(set(source_genes)), "invalid signed program gene set")
        measured_up = [g for g in up if g in lookup]
        measured_down = [g for g in down if g in lookup]
        measured = len(measured_up) + len(measured_down)
        fraction = measured / len(source_genes)
        require(fraction >= minimum_coverage, "fixed program measured coverage below locked threshold")
        require(len(measured_up) >= minimum_direction_genes and
                (not down or len(measured_down) >= minimum_direction_genes),
                "fixed program direction has insufficient measured genes")

        def mean_log(names):
            if not names:
                return np.zeros(len(library), dtype=np.float64)
            selected = count64[:, [lookup[g] for g in names]]
            if sparse.issparse(selected):
                selected = selected.toarray()
            return np.log1p(np.asarray(selected) * (scale / library[:, None])).mean(axis=1)

        targets.append(mean_log(measured_up) - mean_log(measured_down))
        coverage.append(dict(id=program["id"], fixed_genes=len(source_genes), measured_genes=measured,
                             measured_coverage=fraction, fixed_up=len(up), measured_up=len(measured_up),
                             fixed_down=len(down), measured_down=len(measured_down),
                             missing_up_genes=[g for g in up if g not in lookup],
                             missing_down_genes=[g for g in down if g not in lookup],
                             zeros_are_measured=True, selection_by_expression=False))
    result = np.column_stack(targets)
    require(np.isfinite(result).all(), "nonfinite program targets")
    return result, library, detected, coverage


def spatial_neighbors(features, coordinates, sections, *, radius=300., maximum=16):
    """Same-section radius neighbors, ordered by distance then original row."""
    X = np.asarray(features)
    coords = np.asarray(coordinates, dtype=np.float64)
    sections = np.asarray(sections).astype(str)
    require(X.ndim == 2 and X.dtype == np.float32 and np.isfinite(X).all(), "invalid frozen feature array")
    require(coords.shape == (len(X), 2) and np.isfinite(coords).all() and sections.shape == (len(X),),
            "invalid coordinates or section vector")
    require(np.isfinite(radius) and radius > 0 and type(maximum) is int and maximum >= 1,
            "invalid neighbor contract")
    result = np.empty_like(X)
    counts = np.zeros(len(X), dtype=np.int64)
    neighbors = np.full((len(X), maximum), -1, dtype=np.int64)
    maximum_distance = 0.
    for section in sorted(set(sections)):
        rows = np.flatnonzero(sections == section)
        local = coords[rows]
        require(len(np.unique(local, axis=0)) == len(local), "duplicate physical coordinates within section")
        tree = cKDTree(local)
        for local_i, row in enumerate(rows):
            candidates = np.asarray(tree.query_ball_point(local[local_i], radius), dtype=np.int64)
            distance_sq = np.sum((local[candidates] - local[local_i]) ** 2, axis=1)
            eligible = distance_sq <= radius**2
            candidates, distance_sq = candidates[eligible], distance_sq[eligible]
            order = np.lexsort((rows[candidates], distance_sq))[:maximum]
            chosen = rows[candidates[order]]
            require(len(chosen) and row in chosen and np.all(sections[chosen] == section),
                    "invalid self/section neighbor identity")
            require(chosen[0] == row, "self must be first under unique physical coordinates")
            result[row] = np.mean(X[chosen], axis=0, dtype=np.float64).astype(np.float32)
            counts[row] = len(chosen)
            neighbors[row, :len(chosen)] = chosen
            maximum_distance = max(maximum_distance, float(np.sqrt(distance_sq[order]).max()))
    require(np.all(counts >= 1), "unfilled neighbor rows")
    audit = dict(radius_um=float(radius), maximum_neighbors_including_self=maximum,
                 actual_neighbor_count_min=int(counts.min()), actual_neighbor_count_median=float(np.median(counts)),
                 actual_neighbor_count_max=int(counts.max()), self_only_rows=int(np.sum(counts == 1)),
                 maximum_retained_distance_um=maximum_distance, cross_section_edges=0,
                 self_included_rows=len(X), stable_tie_break="distance_then_original_StageA_row_index",
                 mean_accumulation="float64_then_float32_output", use_primary_subset_only=True,
                 sampled_random_or_disconnected_sequence=False)
    return result, neighbors, counts, audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "refusing to overwrite prepared Stage B artifact")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(args):
    phase22 = args.phase22_root.resolve()
    a_root, b_root, output = args.stage_a_root.resolve(), args.feasibility_root.resolve(), args.output.resolve()
    require(not output.is_relative_to(REPO) and not output.is_relative_to(phase22 / "raw"),
            "prepared patient-level arrays must stay outside source repository/raw assets")
    require(not output.exists() or not any(output.iterdir()), "refusing to overwrite Stage B prepared data")
    config, program_record = read_json(args.protocol), read_json(args.programs)
    require(config.get("schema_version") == "sami_surv_stage_b_protocol_v1"
            and config.get("status") == "LOCKED_BEFORE_MODEL_FIT", "Stage B protocol is not locked")
    contract = config["data_contract"]
    am_path = a_root / "manifest.json"
    ad_path = a_root / "prepared.npz"
    feasibility_path = b_root / "DATA_FEASIBILITY_PRIVATE.json"
    vocabulary_path = b_root / "FULL_GENE_VOCABULARY_PRIVATE.json"
    expected_contract = dict(stage_a_data_sha256=sha256(ad_path), stage_a_manifest_sha256=sha256(am_path),
                             program_definition_sha256=sha256(args.programs), source_feasibility_sha256=sha256(feasibility_path),
                             raw_gene_count=36601, primary_spots=13011, CP10K_scale=10000.0,
                             neighbor_radius_um=300.0, neighbor_max=16, minimum_coverage=.95,
                             minimum_measured_genes_per_nonempty_direction=5)
    require(all(contract.get(k) == v for k, v in expected_contract.items()), "locked Stage B data contract mismatch")
    programs = validate_programs(program_record)
    a_manifest, feasibility = read_json(am_path), read_json(feasibility_path)
    vocabulary = read_json(vocabulary_path)
    require(a_manifest.get("schema_version") == "sami_stage_a_prepared_v1"
            and a_manifest.get("status") == "PASS_PREPARED_FROZEN_TISSUE3"
            and a_manifest["data_sha256"] == expected_contract["stage_a_data_sha256"], "Stage A preparation failed qualification")
    require(feasibility.get("status") == "PASS_DEVELOPMENT_RAW_DATA"
            and feasibility["stage_a_manifest_sha256"] == sha256(am_path)
            and feasibility["full_gene_vocabulary_sha256"] == sha256(vocabulary_path), "Stage B source feasibility mismatch")
    require(feasibility.get("original_ST_test_arrays_read") is False
            and feasibility.get("TCGA_DSS_read") is False and feasibility.get("actual_model_fits") == 0,
            "unexpected feasibility scope")
    with np.load(ad_path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in ("X", "y", "donor", "section", "barcodes", "coords_um")}
    X, tissue = arrays["X"], arrays["y"]
    require(X.dtype == np.float32 and X.shape == (13011, 1536) and np.isfinite(X).all(), "Stage A feature shape/dtype mismatch")
    require(tissue.shape == (13011,) and set(tissue) == {0, 1, 2}, "Stage A tissue labels are not the locked primary classes")
    sources = feasibility["sections"]
    require(len(sources) == 12 and len({r["section_id_private"] for r in sources}) == 12
            and set(arrays["section"]) == {r["section_id_private"] for r in sources}, "source section set mismatch")
    targets = np.full((len(X), len(programs)), np.nan, dtype=np.float64)
    library = np.full(len(X), np.nan, dtype=np.float64)
    detected = np.full(len(X), -1, dtype=np.int64)
    lineage, coverage_reference = [], None
    for source in sources:
        section = source["section_id_private"]
        require(source["split"] in {"train", "validation"} and source["study_id"] == "Zenodo_7760264",
                "original ST-test or unexpected source study is forbidden")
        source_path = (phase22 / "raw/hest/st" / (section + ".h5ad")).resolve()
        require(source_path.is_relative_to(phase22 / "raw/hest/st")
                and source_path == Path(source["raw_h5ad_path_private"]).resolve(), "raw source path differs from feasibility")
        require(sha256(source_path) == source["raw_h5ad_sha256"], "raw source file SHA changed")
        expression = read_h5ad_raw(source_path)
        require(expression["gene_names"].tolist() == vocabulary["gene_names"]
                and len(expression["gene_names"]) == 36601, "raw gene vocabulary differs from feasibility")
        rows = np.flatnonzero(arrays["section"] == section)
        require(np.all(arrays["donor"][rows] == source["donor_id_private"]), "Stage A donor assignment mismatch")
        barcodes = arrays["barcodes"][rows].astype(str)
        require(len(set(barcodes)) == len(barcodes), "duplicate Stage A barcode within source section")
        lookup = {b: i for i, b in enumerate(expression["barcodes"])}
        require(all(b in lookup for b in barcodes), "primary barcode absent from raw source")
        raw_rows = np.asarray([lookup[b] for b in barcodes], dtype=np.int64)
        primary_counts = expression["matrix"][raw_rows]
        y, lib, n_detected, coverage = signed_program_scores(primary_counts, expression["gene_names"], programs)
        if coverage_reference is None:
            coverage_reference = coverage
        require(coverage == coverage_reference, "program coverage differs across locked source sections")
        targets[rows], library[rows], detected[rows] = y, lib, n_detected
        lineage.append(dict(section_id_private=section, donor_id_private=source["donor_id_private"],
                            raw_count_sha256=source["raw_h5ad_sha256"], feature_sha256=source["features_sha256"],
                            primary_rows=len(rows), exact_primary_barcode_alignment=True))
    require(np.isfinite(targets).all() and np.isfinite(library).all() and np.all(detected >= 0),
            "program target population incomplete")
    neighbor_X, neighbor_indices, neighbor_counts, neighbor_audit = spatial_neighbors(
        X, arrays["coords_um"], arrays["section"])
    prepared = dict(X=X, neighbor_X=neighbor_X, tissue_y=tissue, donor=arrays["donor"], section=arrays["section"],
                    barcodes=arrays["barcodes"], coords_um=arrays["coords_um"], program_y=targets,
                    library_size=library, n_detected_genes=detected)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "prepared.npz"
    with data_path.open("wb") as stream:
        np.savez_compressed(stream, **prepared)
    neighbors_path = output / "NEIGHBORS_PRIVATE.npz"
    with neighbors_path.open("wb") as stream:
        np.savez_compressed(stream, neighbor_indices=neighbor_indices, neighbor_counts=neighbor_counts)
    manifest = dict(schema_version="sami_stage_b_prepared_v1", status="PASS_PREPARED_SIGNED_PROGRAMS",
                    data_file="prepared.npz", prepared_data_sha256=sha256(data_path), data_sha256=sha256(data_path),
                    config_sha256=sha256(args.protocol), program_definition_sha256=sha256(args.programs),
                    source_feasibility_sha256=sha256(feasibility_path), stage_a_manifest_sha256=sha256(am_path),
                    stage_a_data_sha256=sha256(ad_path), source_files=lineage, n_spots=len(X), donors=6, sections=12,
                    feature_dimension=1536, program_order=list(PROGRAM_ORDER), program_coverage=coverage_reference,
                    tissue_classes=a_manifest["classes"], outer_donor_folds=a_manifest["outer_donor_folds"],
                    neighbor_file="NEIGHBORS_PRIVATE.npz", neighbor_file_sha256=sha256(neighbors_path),
                    neighborhood_contract=neighbor_audit,
                    source_code_sha256={p: sha256(REPO / p) for p in ["scripts/prepare_sami_stage_b.py",
                        "src/phase22_spatial_bridge/data_contract.py", "src/sami_surv/annotations.py"]},
                    preprocessing_fitted_on_cohort=False, zero_counts_treated_as_measured=True,
                    primary_StageA_row_order_preserved=True, original_ST_test_read=False, TCGA_DSS_read=False,
                    actual_model_fits=0, raw_sources_modified=False, source_association_is_not_cell_source_truth=True)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    public_coverage = [{k: v for k, v in p.items() if not k.startswith("missing_")} for p in coverage_reference]
    audit = dict(schema_version="sami_stage_b_prepared_audit_public_v1", status=manifest["status"],
                 prepared_manifest_sha256=sha256(manifest_path), prepared_data_sha256=manifest["data_sha256"],
                 config_sha256=manifest["config_sha256"], program_definition_sha256=manifest["program_definition_sha256"],
                 source_feasibility_sha256=manifest["source_feasibility_sha256"], stage_a_manifest_sha256=sha256(am_path),
                 stage_a_data_sha256=sha256(ad_path), n_spots=len(X), donors=6, sections=12,
                 frozen_UNI2_feature_dimension=1536, program_order=list(PROGRAM_ORDER), program_coverage=public_coverage,
                 all_12_raw_source_hashes_verified=True, exact_StageA_primary_order=True,
                 library_size_from_all_36601_raw_genes=True, accumulation_and_program_scores="float64",
                 score_definition="mean(log1p(CP10K(up))) - mean(log1p(CP10K(down))); empty_down=0; measured_genes_only",
                 neighborhood=neighbor_audit, expression_based_program_selection=False,
                 preprocessing_fitted_on_cohort=False, original_ST_test_read=False, TCGA_DSS_read=False,
                 actual_model_fits=0, source_association_is_not_cell_source_truth=True)
    write_json(args.public_report, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase22-root", "stage-a-root", "feasibility-root", "output", "protocol", "programs"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--public-report", type=Path, default=REPO / "reports/sami_surv_stage_b/DATA_AUDIT.json")
    print(json.dumps(prepare(parser.parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

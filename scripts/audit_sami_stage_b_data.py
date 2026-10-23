"""Read-only development-ST feasibility audit for SAMI Stage B; zero fitting.

The original test sections and all TCGA/survival assets are outside the input
contract. Outputs contain new lineage metadata only, never modified sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.sami_surv.annotations import (
    read_json, require, sha256, under, qualified_feature_section,
    validate_development_lock, validate_protocol_source,
)
from src.phase22_spatial_bridge.data_contract import read_h5ad_raw


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "refusing to overwrite an existing Stage B feasibility receipt")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def text_values(dataset):
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]], dtype=str)


def categorical_values(node):
    if isinstance(node, h5py.Dataset):
        return sorted(set(text_values(node)))
    require("categories" in node, "unexpected categorical metadata encoding")
    return sorted(set(text_values(node["categories"])))


def h5ad_schema(path):
    with h5py.File(path, "r") as f:
        matrices = {}
        for name in ["X", "raw/X"] + ["layers/" + k for k in f.get("layers", {})]:
            if name not in f:
                continue
            node = f[name]
            shape = node.shape if isinstance(node, h5py.Dataset) else node.attrs["shape"]
            dtype = node.dtype if isinstance(node, h5py.Dataset) else node["data"].dtype
            matrices[name] = dict(shape=[int(x) for x in shape], dtype=str(dtype),
                                  encoding=str(node.attrs.get("encoding-type", "dense")))
        return dict(matrices=matrices, obs_columns=list(f["obs"]), var_columns=list(f["var"]),
                    obsm_keys=list(f.get("obsm", {})), uns_keys=list(f.get("uns", {})),
                    layers=list(f.get("layers", {})),
                    feature_types=categorical_values(f["var/feature_types"]) if "var/feature_types" in f else [],
                    genome_categories=categorical_values(f["var/genome"]) if "var/genome" in f else [])


def program_coverage(programs, gene_sets):
    require(isinstance(programs, list) and bool(programs), "nonempty fixed program list required")
    seen, result = set(), []
    for program in programs:
        name = program.get("id", program.get("name"))
        genes = program.get("genes")
        require(isinstance(name, str) and name and name not in seen, "program IDs must be nonempty and unique")
        require(isinstance(genes, list) and genes and len(genes) == len(set(genes))
                and all(isinstance(g, str) and g and g == g.strip() for g in genes),
                "program gene symbols must be exact nonempty unique strings")
        seen.add(name)
        available = [len(set(genes) & vocabulary) for vocabulary in gene_sets]
        result.append(dict(id=name, fixed_genes=len(genes), measured_gene_count_min=min(available),
                           measured_gene_count_max=max(available), coverage_min=min(available) / len(genes),
                           coverage_max=max(available) / len(genes),
                           availability_only_no_expression_or_outcome_selection=True))
    return result


def audit(args):
    root = args.phase22_root.resolve()
    private_root = args.private_output.resolve()
    require(not private_root.is_relative_to(REPO) and not private_root.is_relative_to(root / "raw"),
            "private feasibility outputs must be outside repository/source raw tree")
    protocol = read_json(args.source_protocol)
    lock_path = root / "metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json"
    lock_sha = sha256(lock_path)
    validate_protocol_source(protocol, lock_sha)
    marker_path = root / "processed/DATA_QUALIFICATION_PRIVATE.json"
    lock, marker = read_json(lock_path), read_json(marker_path)
    require(marker.get("source_revision") == protocol["source"]["hest_revision"], "qualified HEST revision mismatch")
    rows, evidence = validate_development_lock(lock, marker, lock_sha)
    require(sha256(root / "metadata/hest/HEST_v1_3_0.csv") == lock["metadata_sha256"], "source donor ledger mismatch")
    stage_a = args.stage_a_root.resolve()
    a_manifest = read_json(stage_a / "manifest.json")
    require(a_manifest["original_split_lock_sha256"] == lock_sha
            and a_manifest["protocol_sha256"] == sha256(args.source_protocol), "Stage A source lineage mismatch")
    label_path = stage_a / "annotation_mapping.npz"
    require(sha256(label_path) == a_manifest["annotation_mapping_sha256"], "Stage A independent annotation mapping SHA mismatch")
    with np.load(label_path, allow_pickle=False) as mapping:
        labels = {k: mapping[k].copy() for k in ("section", "donor", "barcodes", "raw_labels", "categories", "y")}
    sources, schemas, gene_sets, raw_gene_sequences = [], [], [], []
    for row in rows:
        section = row["section_id_private"]
        X, coords, barcodes, _, feature_source = qualified_feature_section(
            root, row, evidence[section], lock_sha, REPO)
        qualified_meta = read_json(under(root, evidence[section]["metadata_path"]))
        expression_source = qualified_meta["source_files"]["expression"]
        raw_path = under(root, expression_source["path"])
        raw_sha = sha256(raw_path)
        require(raw_sha == expression_source["sha256"], "raw development expression SHA mismatch")
        schema = h5ad_schema(raw_path)
        expression = read_h5ad_raw(raw_path)
        matrix = expression["matrix"]
        gene_names = expression["gene_names"]
        require(schema["feature_types"] == ["Gene Expression"], "unexpected assay feature type")
        require(set(expression["barcodes"]) == set(barcodes), "raw/feature barcode sets differ")
        index = {b: i for i, b in enumerate(expression["barcodes"])}
        order = np.asarray([index[b] for b in barcodes], dtype=np.int64)
        source_mpp = float(qualified_meta["audit"]["coordinates"]["mpp_x"])
        require(np.array_equal(expression["coords_px"][order] * source_mpp, coords),
                "raw expression to physical feature coordinates mismatch")
        with np.load(root / "processed/sections" / (section + ".npz"), allow_pickle=False) as processed:
            old_genes, old_counts = processed["gene_names"].astype(str), processed["counts"]
            measured = processed["measured_mask"].astype(bool)
        require(measured.all(), "old development vocabulary has unmeasured genes")
        gene_lookup = {g: i for i, g in enumerate(gene_names)}
        require(all(g in gene_lookup for g in old_genes), "old vocabulary not contained in original genes")
        subset = matrix[order][:, [gene_lookup[g] for g in old_genes]]
        if sparse.issparse(subset):
            subset = subset.toarray()
        require(np.array_equal(subset, old_counts), "old 378-gene cache differs from raw integer source")
        selected = labels["section"] == section
        require(selected.sum() == len(barcodes) and np.array_equal(labels["barcodes"][selected], barcodes)
                and np.all(labels["donor"][selected] == row["donor_id_private"]),
                "independent annotation alignment mismatch")
        lib = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
        require(np.isfinite(lib).all() and np.all(lib > 0), "invalid or empty full-gene library")
        gene_sets.append(set(gene_names))
        raw_gene_sequences.append(gene_names.tolist())
        schemas.append(schema)
        sources.append(dict(feature_source, raw_h5ad_path_private=str(raw_path), raw_h5ad_sha256=raw_sha,
                            raw_h5ad_bytes=raw_path.stat().st_size, raw_schema=schema,
                            raw_count_audit=expression["audit"], processed_gene_count=len(old_genes),
                            processed_counts_elementwise_equal_raw_subset=True,
                            raw_barcode_to_feature_exact_join=True, physical_coordinates_elementwise_equal=True,
                            annotation_elementwise_alignment=True, feature_dimension=int(X.shape[1]),
                            annotated_spots=len(barcodes), primary_spots=int(np.sum(labels["y"][selected] >= 0)),
                            original_matrix_has_no_HVG_selection_field="highly_variable" not in schema["var_columns"],
                            full_library_summary=dict(min=float(lib.min()), median=float(np.median(lib)), max=float(lib.max()))))
    require(sum(s["annotated_spots"] for s in sources) == 17392
            and sum(s["primary_spots"] for s in sources) == 13011, "locked development spot counts changed")
    union, intersection = set.union(*gene_sets), set.intersection(*gene_sets)
    all_gene_order_equal = all(sequence == raw_gene_sequences[0] for sequence in raw_gene_sequences)
    require(all_gene_order_equal and len(intersection) == 36601, "full source gene vocabulary/order unexpectedly differs")
    common_schema = {k: schemas[0][k] for k in ("obs_columns", "var_columns", "obsm_keys", "uns_keys", "layers", "feature_types")}
    require(all(all(s[k] == v for k, v in common_schema.items()) for s in schemas), "development assay schema differs")
    potential_source_fields = {k for schema in schemas for k in schema["obs_columns"] + schema["obsm_keys"]
                               if any(t in k.lower() for t in ("celltype", "cell_type", "deconv", "cell2location", "rctd", "cell_fraction"))}
    vocabulary_record = dict(schema_version="sami_stage_b_raw_gene_vocabulary_private_v1",
                             gene_names=raw_gene_sequences[0], gene_count=len(intersection),
                             source_split_lock_sha256=lock_sha, gene_order_equal_across_development=True,
                             gene_metadata_not_expression_selection=True)
    vocabulary_path = private_root / "FULL_GENE_VOCABULARY_PRIVATE.json"
    write_json(vocabulary_path, vocabulary_record)
    coverage = []
    if args.programs is not None:
        program_record = read_json(args.programs)
        coverage = program_coverage(program_record["programs"], gene_sets)
    private_record = dict(schema_version="sami_stage_b_data_feasibility_private_v1", status="PASS_DEVELOPMENT_RAW_DATA",
                          source_protocol_sha256=sha256(args.source_protocol), source_split_lock_sha256=lock_sha,
                          source_qualification_sha256=sha256(marker_path), stage_a_manifest_sha256=sha256(stage_a / "manifest.json"),
                          annotation_mapping_sha256=sha256(label_path), full_gene_vocabulary_sha256=sha256(vocabulary_path),
                          source_code_sha256=sha256(Path(__file__)), sections=sources,
                          program_definition_sha256=sha256(args.programs) if args.programs is not None else None,
                          program_coverage=coverage, original_ST_test_arrays_read=False, TCGA_DSS_read=False,
                          raw_sources_modified=False, actual_model_fits=0)
    private_path = private_root / "DATA_FEASIBILITY_PRIVATE.json"
    write_json(private_path, private_record)
    public = dict(schema_version="sami_stage_b_data_feasibility_public_v1", status="PASS_DEVELOPMENT_RAW_DATA",
                  private_manifest_sha256=sha256(private_path), source_protocol_sha256=sha256(args.source_protocol),
                  source_code_sha256=sha256(Path(__file__)), development_sections=12, development_donors=6,
                  source_studies=1, annotated_spots=17392, primary_labeled_spots=13011,
                  raw_h5ad_total_bytes=sum(s["raw_h5ad_bytes"] for s in sources),
                  full_raw_gene_count_min=min(len(s) for s in gene_sets), full_raw_gene_count_max=max(len(s) for s in gene_sets),
                  full_raw_gene_union=len(union), full_raw_gene_intersection=len(intersection),
                  full_gene_order_equal=all_gene_order_equal, processed_phase22_gene_count=378,
                  raw_matrix_path="X", raw_matrix_encoding="CSR", raw_matrix_dtype="float32",
                  nonnegative_integer_source_count_checks_passed=12, source_has_only_1000_HVG=False,
                  source_gene_vocabulary_scope="stored_reference_feature_vocabulary_not_independent_assay_inventory",
                  all_gene_names_present_does_not_mean_all_genes_detected_in_each_spot=True,
                  raw_feature_processed_annotation_alignment_passed_sections=12,
                  processed_378_gene_cache_raw_subset_elementwise_equal_sections=12,
                  source_schema=common_schema,
                  independent_label_availability=dict(pathologist_spot_anatomical_categories=True,
                      same_spot_cell_type_proportions=False, same_spot_true_cell_source=False,
                      deconvolution_or_cell_type_fields_in_inspected_H5AD=bool(potential_source_fields),
                      annotations_are_cell_fractions=False, new_single_cell_reference_required_for_cell_source_supervision=True),
                  fixed_programs_coverage_status="COMPLETE" if coverage else "PENDING_EXTERNALLY_LOCKED_PROGRAMS",
                  fixed_program_coverage=coverage, program_definition_sha256=private_record["program_definition_sha256"],
                  safe_next_step="Use exact full-gene raw counts and fixed external program definitions with donor-held-out preprocessing; qualify tissue-associated programs without claiming identified cell sources.",
                  original_ST_test_arrays_read=False, TCGA_DSS_read=False, raw_sources_modified=False, actual_model_fits=0)
    write_json(args.public_report, public)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase22-root", type=Path, required=True)
    parser.add_argument("--stage-a-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, default=REPO / "configs/sami_surv_stage_a_v1.json")
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--programs", type=Path)
    parser.add_argument("--public-report", type=Path, default=REPO / "reports/sami_surv_stage_b/DATA_FEASIBILITY.json")
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

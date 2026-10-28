"""Read-only source program/depth diagnostics; no TCGA, model or optimizer.

True-target correlations are computed from the SHA-locked spatial-source cache.
Prediction/depth correlations, when available, are copied only from completed,
replayed LODO metadata. The two types of correlation are never conflated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.bgb_en.r6_source import SourceContract, load_private_cache, sha256, PROGRAM_ORDER


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def average_ranks(values):
    """One-based midranks, including exact ties, without p-value computation."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranked[order[start:end]] = (start + 1 + end) / 2.
        start = end
    return ranked


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.ndim == y.ndim == 1 and x.shape == y.shape, "DEPTH_CORRELATION_SHAPE_INVALID")
    keep = np.isfinite(x) & np.isfinite(y)
    a, b = x[keep], y[keep]
    support = len(a)
    if support < 3:
        return dict(support=support, Spearman=None, status="INSUFFICIENT_SUPPORT")
    constant_a, constant_b = np.ptp(a) == 0, np.ptp(b) == 0
    if constant_a or constant_b:
        status = "BOTH_CONSTANT" if constant_a and constant_b else "CONSTANT_PROGRAM_VALUE" if constant_a else "CONSTANT_TECHNICAL_VALUE"
        return dict(support=support, Spearman=None, status=status)
    a, b = average_ranks(a), average_ranks(b)
    a, b = a - a.mean(), b - b.mean()
    rho = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return dict(support=support, Spearman=min(1., max(-1., rho)), status="AVAILABLE")


def target_correlations(dataset):
    target = np.asarray(dataset.program_targets, float)
    donor, section, tissue = (np.asarray(getattr(dataset, name)) for name in ("donor_codes", "section_codes", "tissue_y"))
    library, detected = np.asarray(dataset.library_size, float), np.asarray(dataset.n_detected_genes, float)
    n = len(target)
    require(target.shape == (n, 4) and all(values.shape == (n,) for values in (donor, section, tissue, library, detected)), "SOURCE_DEPTH_INPUT_SHAPES_CHANGED")
    require(all(values.dtype.kind in "iu" for values in (donor, section, tissue)) and np.isin(tissue, (0, 1, 2)).all()
            and np.isfinite(library).all() and (library > 0).all() and np.isfinite(detected).all() and (detected >= 0).all(), "SOURCE_DEPTH_COVARIATES_INVALID")
    groups = [("ALL", -1, np.ones(n, bool))]
    groups += [("DONOR", int(code), donor == code) for code in np.unique(donor)]
    groups += [("SECTION", int(code), section == code) for code in np.unique(section)]
    result = []
    for scope, code, selection in groups:
        for tissue_code in (-1, 0, 1, 2):
            selected = selection if tissue_code == -1 else selection & (tissue == tissue_code)
            for k, name in enumerate(PROGRAM_ORDER):
                depth = spearman(target[selected, k], np.log1p(library[selected]))
                genes = spearman(target[selected, k], detected[selected])
                result.append(dict(measure="TRUE_MEASURED_PROGRAM", scope=scope, group_code=code, tissue_class=tissue_code,
                    program=name, rows=int(selected.sum()), support=depth["support"], log_library_Spearman=depth["Spearman"],
                    log_library_status=depth["status"], n_detected_genes_Spearman=genes["Spearman"],
                    n_detected_genes_status=genes["status"]))
    return result


def cell_name(cell):
    return f"{cell['arm']}_{cell['control']}_h{cell['heldout_donor']}_seed{cell['seed']}"


def saved_prediction_correlations(private_root, config, config_sha, source_report_path):
    """No model artifact or saved output-array deserialization; JSON metadata only."""
    if not Path(source_report_path).is_file():
        return [], dict(status="BLOCKED_SOURCE_REPORT_NOT_AVAILABLE", expected_LODO_program_rows=576,
                        source_report_sha256=None, prediction_n_detected_and_within_tissue_depth="NOT_SAVED_IN_ORIGINAL_METRICS")
    report = read(source_report_path)
    require(report.get("status") == "COMPLETE_147_SOURCE_UNITS_AND_REPLAY"
            and report.get("source_config_sha256") == config_sha
            and report.get("source_code_sha256") == config["source_code_sha256"]
            and report.get("source_cache_manifest_sha256") == config["source_cache_manifest_sha256"]
            and report.get("fresh_certificate_sha256") == config["fresh_certificate_sha256"]
            and report.get("counts", {}).get("PASS_COMPLETE_AND_REPLAYED") == 147,
            "SOURCE_DEPTH_REPORT_BINDING_OR_COMPLETION_INVALID")
    ledger = {cell_name(row): row for row in report.get("ledger", [])}
    require(len(ledger) == 147 and set(ledger) == {cell_name(cell) for cell in config["cells"]}, "SOURCE_DEPTH_147_LEDGER_IDENTITIES_CHANGED")
    rows, certificates = [], []
    for cell in config["cells"]:
        name = cell_name(cell)
        row = ledger[name]
        require(row.get("status") == "PASS_COMPLETE_AND_REPLAYED", "SOURCE_DEPTH_REPLAYED_CELL_REQUIRED")
        directory = Path(private_root) / "source_cells" / name
        complete_path, replay_path = directory / "COMPLETE.json", directory / "REPLAY.json"
        require(sha256(complete_path) == row.get("receipt_sha256") and sha256(replay_path) == row.get("replay_sha256"), "SOURCE_DEPTH_METADATA_RECEIPT_SHA_CHANGED")
        complete, replay = read(complete_path), read(replay_path)
        require(complete.get("cell") == replay.get("cell") == cell
                and complete.get("config_sha256") == replay.get("config_sha256") == config_sha
                and complete.get("source_code_sha256") == replay.get("source_code_sha256") == config["source_code_sha256"]
                and replay.get("status") == "PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY"
                and replay.get("completion_sha256") == row["receipt_sha256"]
                and replay.get("new_optimizer_fits") == replay.get("optimizer_updates") == 0
                and replay.get("image_only_heldout_program_prediction") is True, "SOURCE_DEPTH_REPLAY_METADATA_LINEAGE_CHANGED")
        certificates.append(dict(arm=cell["arm"], control=cell["control"], heldout_donor=cell["heldout_donor"], seed=cell["seed"],
            receipt_sha256=row["receipt_sha256"], replay_sha256=row["replay_sha256"]))
        if cell["heldout_donor"] < 0:
            continue
        metrics = complete["metrics"]["heldout"]["per_program"]
        replay_metrics = replay["metrics"]["heldout"]["per_program"]
        require(len(metrics) == len(replay_metrics) == 4
                and [program.get("program") for program in metrics] == [program.get("program") for program in replay_metrics] == list(PROGRAM_ORDER), "SOURCE_DEPTH_METRIC_PROGRAMS_CHANGED")
        correlation_rows = complete["metrics"]["heldout"].get("spots")
        require(type(correlation_rows) is int and correlation_rows >= 0
                and correlation_rows == replay["metrics"]["heldout"].get("spots"), "SOURCE_DEPTH_PREDICTION_ROWS_CHANGED")
        for original, repeated in zip(metrics, replay_metrics):
            value, value_again = original.get("log_library_Spearman"), repeated.get("log_library_Spearman")
            require((value is None and value_again is None) or finite(value) and finite(value_again)
                    and -1 <= value <= 1 and abs(value - value_again) <= 2e-6, "SOURCE_DEPTH_PREDICTION_CORRELATION_REPLAY_MISMATCH")
            require(type(original.get("support")) is int and original["support"] >= 0
                    and original["support"] == repeated.get("support"), "SOURCE_DEPTH_PREDICTION_SUPPORT_CHANGED")
            rows.append(dict(measure="SAVED_IMAGE_ONLY_PREDICTION", scope="HELDOUT_DONOR", group_code=cell["heldout_donor"],
                tissue_class=-1, program=original["program"], arm=cell["arm"], control=cell["control"], seed=cell["seed"],
                rows=correlation_rows, support=correlation_rows, original_program_scoring_support=original["support"], log_library_Spearman=value,
                log_library_status="SAVED_AVAILABLE" if value is not None else "SAVED_UNDEFINED_CONSTANT_OR_INSUFFICIENT",
                n_detected_genes_Spearman=None, n_detected_genes_status="NOT_SAVED_IN_ORIGINAL_METRICS"))
    require(len(rows) == 576, "SOURCE_DEPTH_REQUIRES_144_TIMES_FOUR_LODO_PROGRAMS")
    return rows, dict(status="COMPLETE_576_SAVED_LODO_PROGRAM_DEPTH_CORRELATIONS", source_report_sha256=sha256(source_report_path),
        expected_LODO_program_rows=576, metadata_certificates=certificates,
        prediction_n_detected_and_within_tissue_depth="NOT_SAVED_IN_ORIGINAL_METRICS",
        NN_recomputed=False, model_artifact_or_prediction_arrays_deserialized=False)


def audit(args):
    config = read(args.source_config)
    require(config.get("schema") == "bgb_en_r6_source_pretrain_lock_v1" and config.get("status") == "LOCKED_BEFORE_SOURCE_FITS"
            and len(config.get("cells", [])) == 147, "SOURCE_DEPTH_FORMAL_LOCK_REQUIRED")
    require(sha256(args.fresh_certificate) == config.get("fresh_certificate_sha256"), "SOURCE_DEPTH_FRESH_CERTIFICATE_CHANGED")
    for relative, expected in config.get("source_code_sha256", {}).items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha256(path) == expected, "SOURCE_DEPTH_LOCKED_SOURCE_CODE_CHANGED")
    require(len(config.get("source_code_sha256", {})) == 6, "SOURCE_DEPTH_SIX_SCIENTIFIC_SOURCE_PINS_REQUIRED")
    dataset = load_private_cache(args.source_cache, SourceContract(fresh_certificate_sha256=config["fresh_certificate_sha256"]),
                                 expected_manifest_sha256=config["source_cache_manifest_sha256"])
    true_rows = target_correlations(dataset)
    require(len(np.unique(dataset.donor_codes)) == 6 and len(np.unique(dataset.section_codes)) == 12
            and len(true_rows) == 304, "SOURCE_DEPTH_PRIMARY_SIX_DONOR_TWELVE_SECTION_SCOPE_CHANGED")
    source_report = args.source_report or Path(args.output) / "R6_1_SOURCE_SUPERVISION.json"
    prediction_rows, prediction_audit = saved_prediction_correlations(args.private_root, config, sha256(args.source_config), source_report)
    return dict(schema="bgb_en_r6_source_depth_audit_v1", status="COMPLETE_TRUE_MEASUREMENT_DEPTH_AUDIT",
        script_sha256=sha256(__file__), source_config_sha256=sha256(args.source_config),
        source_cache_manifest_sha256=config["source_cache_manifest_sha256"], fresh_certificate_sha256=config["fresh_certificate_sha256"],
        source_code_sha256=config["source_code_sha256"], programs=list(PROGRAM_ORDER), source_spots=len(dataset.program_targets),
        source_donors=6, source_sections=12, true_target_rows=true_rows, prediction_metadata=prediction_audit,
        saved_prediction_rows=prediction_rows, targets="raw_signed_program_CP10K_not_cohort_standardized",
        library="full_measured_36601_gene_library_log1p", detected="full_measured_detected_gene_count",
        correlation="Spearman_average_ranks_exact_ties_pairwise_finite_support_ge3_nonconstant_both",
        within_tissue_is_descriptive_not_causal_adjustment=True, donor_section_and_tissue_codes_are_anonymous=True,
        spots_are_not_independent_cohorts=True, no_CI_pvalues_selection_or_causal_conclusion=True,
        TCGA_data_or_labels_read=False, models_loaded=False, model_forward_calls=0, new_optimizer_updates=0,
        identifiers_target_arrays_coordinates_or_private_paths_exported=False)


def write_results(result, output):
    output = Path(output).resolve()
    require(not any(output == path.resolve() or output.is_relative_to(path.resolve()) for path in
                    (ROOT / "reports/bgb_en", ROOT / "reports/bgb_en_r5", ROOT / "reports/sami_surv_stage_b")), "SOURCE_DEPTH_OLD_NAMESPACES_PRESERVED")
    output.mkdir(parents=True, exist_ok=True)
    (output / "R6_SOURCE_DEPTH_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ("measure", "scope", "group_code", "tissue_class", "program", "arm", "control", "seed", "rows", "support",
        "log_library_Spearman", "log_library_status", "n_detected_genes_Spearman", "n_detected_genes_status", "original_program_scoring_support")
    with (output / "R6_SOURCE_DEPTH_AUDIT.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore"); writer.writeheader()
        writer.writerows(result["true_target_rows"] + result["saved_prediction_rows"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--fresh-certificate", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    args = parser.parse_args()
    result = audit(args)
    write_results(result, args.output)
    print(json.dumps(dict(status=result["status"], true_target_rows=len(result["true_target_rows"]),
        saved_prediction_rows=len(result["saved_prediction_rows"]), new_optimizer_updates=0)))


if __name__ == "__main__":
    main()

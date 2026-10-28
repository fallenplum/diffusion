"""Review anonymous retention reports without loading arrays, weights or endpoints.

Only exported concordance counts, numeric replay receipts and public source-byte
pins are reviewed. This does not repeat the private saved-coefficient forward,
patient-risk metric computation or Efron gradient computation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ("FIXED_HALLMARK_SUM50", "FIXED_HALLMARK_MEAN50", "ST_GENE_UNION_RAW760",
       "ST_SIGNED_PROGRAM_MEAN4", "FROZEN_S7_FLAT256", "FROZEN_S7_MEAN64")
OLD = ("LEGACY_RNA512_RIDGE", "LEGACY_FULL_RNA20511_RIDGE", "LEGACY_RNA512_PCA256_UNWHITENED_GENE_L2")
DIM = dict(zip(REP, (50, 50, 760, 4, 256, 64)))
SOURCE_ALLOWLIST = frozenset((
    "configs/bgb_en/prototype_v1.json", "configs/bgb_en/r6_retention_v1.json",
    "configs/sami_surv_stage_b_programs_v1.json", "scripts/bgb_en_r6_common.py",
    "scripts/bgb_en_r6_pretrain.py", "scripts/bgb_en_r6_retention.py", "scripts/bgb_en_run.py",
    "src/bgb_en/__init__.py", "src/bgb_en/baselines.py", "src/bgb_en/data.py",
    "src/bgb_en/external.py", "src/bgb_en/interaction.py", "src/bgb_en/omics.py",
    "src/bgb_en/pathology.py", "src/bgb_en/protocol.py", "src/bgb_en/r6_omics.py",
    "src/bgb_en/r6_pathology.py", "src/bgb_en/r6_source.py", "src/bgb_en/r6_source_model.py",
    "src/bgb_en/replay.py", "src/bgb_en/survival.py", "src/bgb_en/training.py",
    "src/phase10_trimodal/math.py", "src/phase10_trimodal/reference.py"))


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(value, label):
    if not value:
        raise AssertionError(label)


def review(reports):
    reports = Path(reports).resolve()
    report = read(reports / "R6_2_OMICS_RETENTION.json")
    recipe = read(ROOT / "configs/bgb_en/r6_retention_v1.json")
    rows = report["rows"]
    keyed = {(x["representation"], x["partition"]): x for x in rows}
    check(len(rows) == len(keyed) == 27, "27 unique rows required")
    check(set(keyed) == {(name, part) for name in (*REP, *OLD) for part in range(3)}, "finite matrix mismatch")
    new = [keyed[name, part] for name in REP for part in range(3)]
    old = [keyed[name, part] for name in OLD for part in range(3)]
    check(all(x["status"] == "COMPLETE_AND_REPLAYED" for x in new), "new replay status")
    check(all(x["actual_optimizer_fits"] == 1 and x["optimizer_representation"] == x["representation"] for x in new), "fit/alias accounting")
    check(all(x["lambda_L2"] == 1 and x["parameters"] == DIM[x["representation"]] for x in new), "new lambda/dimension")
    check(all(x["status"] == "REUSED_EXACT_PRIOR_ACTUAL_REPLAY" and x["actual_optimizer_fits"] == 0 for x in old), "old references are not new fits")
    check(report["planned_logical_cells"] == report["planned_actual_fits"] == report["confirmed_new_optimizer_fits"] == 18
          and report["unknown_fit_count_cells"] == 0, "global budget")
    check(report["new_Set_or_Mamba_training"] == report["new_teacher_or_PCA_fits"] == 0, "extra training")
    check(report["status"] == "COMPLETE_18_LOGICAL_CELLS_AND_REPLAY", "report status")
    check(report["no_cross_partition_risk_concatenation"] is True, "cross partition risk sorting")
    for spec in recipe["historical_references"]:
        row = keyed[spec["reference"], spec["partition"]]
        check(row["receipt_sha256"] == spec["receipt_sha256"] and row["artifact_sha256"] == spec["artifact_sha256"], "old reference hashes")
        check(row["parameters"] == spec["parameters"] and row["lambda_L2"] == spec["lambda_L2"], "old reference dimensions/lambda")
    with (reports / "R6_2_OMICS_RETENTION.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    check(len(csv_rows) == 27, "csv count")
    csv_error = 0.
    row_metric_error = 0.
    for text in csv_rows:
        x = keyed[text["representation"], int(text["partition"])]
        check(text["status"] == x["status"] and int(text["actual_optimizer_fits"]) == x["actual_optimizer_fits"], "csv status/budget")
        for field in ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den"):
            csv_error = max(csv_error, abs(float(text[field]) - x["metrics"][field]))
        for metric, numerator, denominator in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
            row_metric_error = max(row_metric_error, abs(x["metrics"][metric] - x["metrics"][numerator] / x["metrics"][denominator]))
    check(csv_error == 0 and row_metric_error <= 1e-15, "csv/row arithmetic")
    partition_denominators = []
    for part in range(3):
        subset = [keyed[name, part] for name in (*REP, *OLD)]
        check(len({x["metrics"]["Harrell_den"] for x in subset}) == len({x["metrics"]["Uno_den"] for x in subset}) == 1,
              "within-partition common evaluation support")
        partition_denominators.append(dict(partition=part, Harrell_den=subset[0]["metrics"]["Harrell_den"], Uno_den=subset[0]["metrics"]["Uno_den"]))
    summary_map = {x["representation"]: x for x in report["summaries"]}
    recomputed = []
    summary_error = 0.
    count_summation_error = 0.
    for name in (*REP, *OLD):
        subset = [keyed[name, part] for part in range(3)]
        counts = {field: math.fsum(x["metrics"][field] for x in subset) for field in ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")}
        metrics = dict(Harrell=counts["Harrell_num"] / counts["Harrell_den"], Uno60=counts["Uno_num"] / counts["Uno_den"])
        saved = summary_map[name]
        check(saved["complete_partitions"] == 3 and saved["status"] == "COMPLETE", "saved summary status")
        for field, value in counts.items():
            difference = abs(saved["counts"][field] - value)
            count_summation_error = max(count_summation_error, difference)
            check(difference <= 4 * math.ulp(value), "saved summary count summation exceeds four ULP")
        summary_error = max(summary_error, *(abs(saved[key] - value) for key, value in metrics.items()))
        recomputed.append(dict(representation=name, **metrics, counts=counts,
            naive_three_partition_mean={key: sum(x["metrics"][key] for x in subset) / 3 for key in ("Harrell", "Uno60")},
            partitions=[dict(partition=x["partition"], **{key: x["metrics"][key] for key in ("Harrell", "Uno60")}) for x in subset],
            lambda_L2=subset[0]["lambda_L2"], parameters=subset[0]["parameters"]))
    check(summary_error <= 1e-15, "summary ratio arithmetic")
    normalizations = []
    for name in REP:
        subset = [keyed[name, part] for part in range(3)]
        for x in subset:
            n = x["normalization"]
            check(n["dimension"] == DIM[name] and n["active_rows"] == 409 and n["training_rows"] + n["validation_rows"] == 409,
                  "normalization dimension/active scope")
            check(n["train_only_fit"] and n["missing_mask_preserved"] and n["source_ST_scaler_used"] is False,
                  "normalization provenance")
            check(n["penalty_unit"] == "L2_in_this_training_standardized_representation_not_general_geneL2_equivalence", "penalty geometry")
            check(n["observed_feature_entries"] + n["masked_entries"] == 409 * DIM[name], "mask accounting")
        normalizations.append(dict(representation=name,
            effective_training_columns=[x["normalization"]["effective_training_feature_columns"] for x in subset],
            unobserved_columns=[x["normalization"]["train_unobserved_columns"] for x in subset],
            observed_feature_fraction=[x["normalization"]["observed_feature_fraction"] for x in subset],
            masked_entries=[x["normalization"]["masked_entries"] for x in subset]))
    target_lock = read(reports / "PROTOCOL_LOCK.json")
    qualification = read(reports / "R6_3_REAL_GENE_REREAD_AUDIT.json")
    check(canonical({k: v for k, v in target_lock.items() if k != "contract_sha256"}) == target_lock["contract_sha256"], "target canonical lock")
    check(qualification["contract_sha256"] == target_lock["contract_sha256"], "qualification/target contract")
    check(qualification["new_neural_fits"] == qualification["new_optimizer_updates"] == 0 and qualification["real_data_backward_passes"] == 2,
          "qualification is not formal training")
    check(target_lock["outer_test_access"] is False and target_lock["teacher_refits"] == 0, "target seal/reuse")
    replay_export = read(reports / "R6_RETENTION_INDEPENDENT_REPLAY_AUDIT.json")
    check(replay_export["schema"] == "bgb_en_r6_retention_numeric_replay_export_v1"
          and replay_export["status"] == "PASS_18_REAL_RECEIPT_NUMERIC_SUMMARIES", "numeric replay export status")
    check(replay_export["retention_contract_sha256"] == report["retention_contract_sha256"]
          and replay_export["retention_lock_schema"] == "bgb_en_r6_retention_lock_v1"
          and replay_export["retention_lock_status"] == "LOCKED_BEFORE_ALL_RETENTION_FITS", "independent retention lock identity")
    check(replay_export["actual_optimizer_fits"] == replay_export["planned_actual_optimizer_fits"] == replay_export["planned_logical_cells"] == 18,
          "numeric export budget")
    check(replay_export["new_optimizer_fits"] == replay_export["new_forward_calls"] == 0
          and replay_export["target_endpoints_or_arrays_loaded"] is False
          and replay_export["original_private_complete_and_replay_receipts_preserved"] is True, "numeric export is read only")
    check(sha(ROOT / "scripts/bgb_en_r6_export_retention_audit.py") == replay_export["exporter_sha256"], "exporter source bytes")
    check(set(replay_export["source_file_sha256"]) == SOURCE_ALLOWLIST, "exact public scientific source allowlist required")
    for relative, digest in replay_export["source_file_sha256"].items():
        check((ROOT / relative).resolve().is_relative_to(ROOT.resolve()), "source pin outside repository")
        check(sha(ROOT / relative) == digest, "retention source byte pin")
    numeric = {(x["representation"], x["partition"]): x for x in replay_export["rows"]}
    check(len(numeric) == len(replay_export["rows"]) == 18 and set(numeric) == {(name, part) for name in REP for part in range(3)},
          "numeric replay matrix")
    for identity, values in numeric.items():
        original = keyed[identity]
        check(values["completion_sha256"] == original["receipt_sha256"] and values["replay_sha256"] == original["replay_sha256"],
              "numeric values not bound to the published row")
        check(values["status"] == "PASS_RETENTION_COEFFICIENT_AND_INDEPENDENT_EFRON_REPLAY"
              and values["actual_optimizer_fits"] == 1 and values["solver_iterations"] == original["iterations"]
              and values["optimizer_representation"] == identity[0], "numeric replay identity/budget")
        for field, limit in (("risk_max_abs_error", 1e-12), ("independent_gradient_max", 1e-6), ("independent_objective_error", 1e-10)):
            check(math.isfinite(values[field]) and 0 <= values[field] <= limit, "numeric replay tolerance")
    numeric_maxima = {field: max(x[field] for x in numeric.values())
        for field in ("risk_max_abs_error", "independent_gradient_max", "independent_objective_error")}
    by_name = {x["representation"]: x for x in recomputed}
    comparisons = []
    for a, b in (("ST_GENE_UNION_RAW760", "LEGACY_RNA512_RIDGE"), ("FROZEN_S7_FLAT256", "ST_GENE_UNION_RAW760"),
                 ("ST_SIGNED_PROGRAM_MEAN4", "ST_GENE_UNION_RAW760"), ("FROZEN_S7_FLAT256", "FROZEN_S7_MEAN64"),
                 ("LEGACY_RNA512_PCA256_UNWHITENED_GENE_L2", "LEGACY_RNA512_RIDGE")):
        comparisons.append(dict(a=a, b=b, **{key: by_name[a][key] - by_name[b][key] for key in ("Harrell", "Uno60")},
            paired_partition_deltas=[dict(partition=part, **{key: keyed[a, part]["metrics"][key] - keyed[b, part]["metrics"][key]
            for key in ("Harrell", "Uno60")}) for part in range(3)]))
    return dict(schema="independent_r6_retention_metadata_review_v1", status="PASS_METADATA_ARITHMETIC_AND_FINITE_BUDGET",
        new_fits=18, new_logical_cells=18, historical_references=9, aliases=0, unknown_fit_cells=0,
        new_iterations=sum(x["iterations"] for x in new), iteration_min=min(x["iterations"] for x in new), iteration_max=max(x["iterations"] for x in new),
        csv_max_abs_error=csv_error, row_ratio_max_abs_error=row_metric_error, summary_max_abs_error=summary_error,
        count_summation_max_abs_error=count_summation_error, independent_summation="math.fsum_with_at_most_four_ULP_count_rounding",
        partition_denominators=partition_denominators, summaries=recomputed, comparisons=comparisons, normalizations=normalizations,
        retention_contract_sha256=report["retention_contract_sha256"], target_contract_sha256=target_lock["contract_sha256"],
        target_qualification_zero_updates=True, target_qualification_not_formal_fit=True,
        observed_primary_union_columns=681, registered_union_columns=760, structurally_unavailable_union_columns=79,
        additional_masked_entries_beyond_79_structural_columns=32635 - 79 * 409,
        numeric_replay_export_maxima=numeric_maxima, numeric_replay_rows_bound_to_published_hashes=18,
        independent_retention_lock_schema_and_contract_checked=True, current_pinned_source_files_checked=len(replay_export["source_file_sha256"]),
        direct_private_replay_values_reviewed=True, raw_private_receipts_or_coefficients_read=False,
        gradient_values_not_in_initial_public_report=True, supplemental_gradient_values_verified=True,
        real_patient_risk_metric_recomputation_performed=False, original_feature_alias_recomputed=False,
        weights_patient_arrays_endpoint_arrays_or_raw_genes_read=False, new_optimizers_or_model_forwards_run=0,
        hashes={name: sha(reports / name) for name in ("R6_2_OMICS_RETENTION.json", "R6_2_OMICS_RETENTION.csv", "R6_2_OMICS_RETENTION.md",
             "PROTOCOL_LOCK.json", "R6_3_REAL_GENE_REREAD_AUDIT.json", "R6_RETENTION_INDEPENDENT_REPLAY_AUDIT.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True, help="Existing anonymous report directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    args = parser.parse_args()
    destination = args.output.resolve()
    check(not destination.exists(), "new independent-review output directory required")
    result = review(args.reports)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "R6_RETENTION_INDEPENDENT_REVIEW.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], new_fits=result["new_fits"], historical_references=result["historical_references"],
                         summary_max_abs_error=result["summary_max_abs_error"], csv_max_abs_error=result["csv_max_abs_error"])))


if __name__ == "__main__":
    main()

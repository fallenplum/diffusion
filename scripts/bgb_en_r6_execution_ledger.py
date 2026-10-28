"""R6-only terminal accounting from JSON receipts and anonymous reports.

No torch/numpy import, checkpoint/array/label read, fit, forward or backward.
Existing actual replay evidence is bound by JSON SHA; weights are not replayed
again by this accountant. All formal matrices must be complete before output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B13")
PARTITIONS, SEEDS = (0, 1, 2), (31, 47, 83)
FAMILIES = (("S0", "MATCHED"), ("S1", "MATCHED"), ("S2", "MATCHED"),
    ("S3", "MATCHED"), ("S4", "MATCHED"), ("S5", "WRONG_DONOR"),
    ("S4", "WITHIN_SECTION_TISSUE_DEPTH"), ("S4", "RANDOM_GATE"),
    ("S6", "MATCHED"), ("S6", "WRONG_NEIGHBOR"), ("S7", "MATCHED"),
    ("S7", "WRONG_SPOT_ALIGNMENT"))
REPRESENTATIONS = ("FIXED_HALLMARK_SUM50", "FIXED_HALLMARK_MEAN50", "ST_GENE_UNION_RAW760",
    "ST_SIGNED_PROGRAM_MEAN4", "FROZEN_S7_FLAT256", "FROZEN_S7_MEAN64")
OLD_RETENTION = ("LEGACY_RNA512_RIDGE", "LEGACY_FULL_RNA20511_RIDGE",
    "LEGACY_RNA512_PCA256_UNWHITENED_GENE_L2")
RESOURCE_FIELDS = ("elapsed_seconds", "GPU_util_percent", "GPU_memory_MiB", "GPU_total_MiB",
    "CPU_util_percent", "RAM_available_MiB", "disk_available_MiB")
NAMES = ("R6_EXECUTION_LEDGER.json", "R6_RESOURCE_SUMMARY.json", "R6_ARM_EXECUTION_STATUS.csv")


def require(value, code):
    if not value:
        raise ValueError(code)


def sha(path):
    # Used exclusively for JSON, CSV and this accountant's Python source.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value, *, unicode=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=not unicode, allow_nan=False).encode()).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def count(value):
    return type(value) is int and value >= 0


def number(value, code):
    require(finite(value) and value >= 0, code)
    return value


def read(path, evidence=None, label=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.suffix in (".json", ".py", ".csv"),
        "REQUIRED_METADATA_FILE_ABSENT_OR_UNSAFE")
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "METADATA_CHANGED_DURING_READ")
    result = json.loads(content.decode("utf-8-sig"))
    require(isinstance(result, dict), "METADATA_JSON_OBJECT_REQUIRED")
    if evidence is not None:
        key = label or path.name
        value = hashlib.sha256(content).hexdigest()
        require(key not in evidence or evidence[key] == value, "DUPLICATE_EVIDENCE_LABEL_CHANGED")
        evidence[key] = value
    return result


def exact_rows(rows, fields, expected, code):
    require(isinstance(rows, list), code)
    keys = [tuple(row.get(k) for k in fields) for row in rows]
    require(len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == set(expected), code)
    return dict(zip(keys, rows))


def source_keys():
    keys = {(arm, control, donor, seed) for arm, control in FAMILIES for donor in range(6) for seed in (31, 47)}
    return keys | {("S7", control, -1, 31) for control in ("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH")}


def target_keys(arms=ARMS):
    return {(arm, part, seed) for arm in arms for part in PARTITIONS for seed in SEEDS}


def unit_name(key):
    return f"{key[0]}_p{key[1]}_seed{key[2]}"


def check_zero(value, keys, code):
    require(all(type(value.get(k)) is int and value[k] == 0 for k in keys), code)


def check_lock(lock, schema, status):
    require(lock.get("schema") == schema and lock.get("status") == status, "PROTOCOL_LOCK_SCHEMA_OR_STATE_CHANGED")
    contents = {k: v for k, v in lock.items() if k not in ("contract_sha256", "created_UTC")}
    require(digest(lock.get("contract_sha256")) and canonical(contents) == lock["contract_sha256"],
        "PROTOCOL_LOCK_CANONICAL_HASH_CHANGED")


def _check_artifact_manifest(manifest):
    require(isinstance(manifest, dict) and manifest and all(isinstance(k, str) and Path(k).name == k
        and "/" not in k and "\\" not in k and digest(v) for k, v in manifest.items()), "ARTIFACT_HASH_MANIFEST_INVALID")


def source_account(private, reports, evidence):
    report = read(reports / "R6_1_SOURCE_SUPERVISION.json", evidence, "report_source")
    require(report.get("schema") == "bgb_en_r6_source_supervision_report_v1"
        and report.get("status") == "COMPLETE_147_SOURCE_UNITS_AND_REPLAY"
        and report.get("counts") == {"PASS_COMPLETE_AND_REPLAYED": 147}, "SOURCE_MATRIX_NOT_COMPLETE")
    fields = ("arm", "control", "heldout_donor", "seed")
    rows = exact_rows(report.get("ledger"), fields, source_keys(), "SOURCE_EXACT_147_UNITS_REQUIRED")
    require(digest(report.get("source_config_sha256")) and report.get("source_code_sha256")
        and all(digest(v) for v in report["source_code_sha256"].values()), "SOURCE_LINEAGE_MISSING")
    output = []
    for key, row in sorted(rows.items()):
        arm, control, donor, seed = key
        label = f"source_{arm}_{control}_d{donor}_s{seed}"
        directory = private / "source_cells" / f"{arm}_{control}_h{donor}_seed{seed}"
        require(not (directory / "ERROR.json").exists(), "SOURCE_ERROR_AND_COMPLETE_CONFLICT")
        done = read(directory / "COMPLETE.json", evidence, label + "_complete")
        replay = read(directory / "REPLAY.json", evidence, label + "_replay")
        require(tuple(done.get("cell", {}).get(k) for k in fields) == key and replay.get("cell") == done["cell"],
            "SOURCE_RECEIPT_IDENTITY_CHANGED")
        fits, updates = (0, 0) if arm == "S0" else (1, 80)
        status = "COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER" if arm == "S0" else "COMPLETE_R6_SOURCE_FIT"
        require(done.get("status") == status and done.get("epochs") == 80
            and done.get("optimizer_updates") == updates and done.get("actual_new_optimizer_fits") == fits,
            "SOURCE_RECEIPT_TRAINING_COST_CHANGED")
        require(row.get("status") == "PASS_COMPLETE_AND_REPLAYED" and row.get("metadata_validated") is True
            and row.get("independent_replay_passed") is True and row.get("actual_optimizer_fits") == fits
            and row.get("actual_optimizer_updates") == updates and row.get("receipt_sha256") == evidence[label + "_complete"]
            and row.get("replay_sha256") == evidence[label + "_replay"], "SOURCE_REPORT_RECEIPT_COST_OR_SHA_CHANGED")
        for item in (done, replay):
            require(item.get("config_sha256") == report["source_config_sha256"]
                and item.get("source_code_sha256") == report["source_code_sha256"], "SOURCE_RECEIPT_LINEAGE_CHANGED")
        require(replay.get("status") == "PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY"
            and replay.get("completion_sha256") == evidence[label + "_complete"]
            and replay.get("model_frozen") is True and replay.get("image_only_heldout_program_prediction") is True
            and replay.get("original_fit_artifacts_modified") is False, "SOURCE_ACTUAL_REPLAY_CHAIN_FAILED")
        check_zero(replay, ("optimizer_updates", "new_optimizer_fits"), "SOURCE_REPLAY_HAS_OPTIMIZER_COST")
        errors = replay.get("forward_max_abs_errors", {})
        require(set(errors) == ({"train"} if donor < 0 else {"train", "heldout"})
            and all(isinstance(v, dict) and v and all(finite(e) and 0 <= e <= 2e-6 for e in v.values())
                for v in errors.values()), "SOURCE_REPLAY_NUMERIC_GUARD_FAILED")
        manifest = done.get("artifact_sha256")
        _check_artifact_manifest(manifest)
        require(row.get("artifact_manifest_sha256") == canonical(manifest)
            and row.get("verified_artifact_files") == len(manifest), "SOURCE_REPORT_ARTIFACT_MANIFEST_CHANGED")
        elapsed = number(done.get("elapsed_seconds"), "SOURCE_ELAPSED_INVALID")
        require(row.get("elapsed_seconds") == elapsed, "SOURCE_REPORT_ELAPSED_CHANGED")
        output.append(dict(arm=arm, control=control, heldout_donor=donor, seed=seed,
            formal_optimizer_fits=fits, formal_optimizer_updates=updates, elapsed_seconds=elapsed,
            complete_sha256=evidence[label + "_complete"], replay_sha256=evidence[label + "_replay"],
            artifact_manifest_sha256=canonical(manifest), actual_replay_verified=True))
    for key, expected in (("planned_logical_units", 147), ("planned_formal_optimizer_fits", 135),
        ("planned_formal_optimizer_updates", 10800), ("planned_random_references", 12),
        ("confirmed_formal_optimizer_fits", 135), ("confirmed_formal_optimizer_updates", 10800),
        ("unknown_fit_cost_units", 0), ("unknown_update_cost_units", 0)):
        require(type(report.get(key)) is int and report[key] == expected, "SOURCE_REPORTED_TOTAL_CHANGED")
    require(sum(r["formal_optimizer_fits"] for r in output) == 135
        and sum(r["formal_optimizer_updates"] for r in output) == 10800, "SOURCE_ACTUAL_TOTAL_CHANGED")
    schedule = read(private / "SOURCE_SCHEDULE_COMPLETE.private.json", evidence, "source_schedule_complete")
    require(not (private / "SOURCE_SCHEDULE_ERROR.private.json").exists()
        and schedule.get("status") == "COMPLETE_147_SOURCE_UNITS_AND_ACTUAL_REPLAY"
        and schedule.get("source_config_sha256") == report["source_config_sha256"]
        and schedule.get("planned_logical_units") == 147 and schedule.get("planned_optimized_units") == 135,
        "SOURCE_SCHEDULER_NOT_CLOSED")
    exact_rows(schedule.get("records"), ("cell_index",), {(i,) for i in range(147)}, "SOURCE_SCHEDULER_UNIT_SET_CHANGED")
    require(all(r.get("status") == "FIT_AND_ACTUAL_REPLAY_COMPLETE" for r in schedule["records"]), "SOURCE_SCHEDULER_HAS_FAILURE")
    qual = read(private / "REAL_SOURCE_COMPUTE_QUALIFICATION.private.json", evidence, "source_qualification")
    require(qual.get("schema") == "bgb_en_r6_real_source_compute_qualification_v1"
        and qual.get("status") == "PASS_TWO_REAL_SOURCE_GPU_UPDATES"
        and qual.get("config_sha256") == report["source_config_sha256"]
        and qual.get("qualification_updates") == 2 and qual.get("separate_qualification_optimizer_instances") == 1
        and qual.get("formal_source_fits") == 0 and qual.get("saved_qualification_weights") is False
        and qual.get("TCGA_DSS_or_PFI_read") is False
        and [r.get("qualification_update") for r in qual.get("steps", [])] == [1, 2], "SOURCE_QUALIFICATION_COST_CHANGED")
    q = report.get("qualification_cost_separate_from_formal_matrix", {})
    require(q.get("provided") is True and q.get("receipt_sha256") == evidence["source_qualification"]
        and (q.get("optimizer_instances"), q.get("optimizer_updates"), q.get("formal_source_fits")) == (1, 2, 0),
        "SOURCE_REPORT_QUALIFICATION_CHANGED")
    return report, output, schedule, qual


def target_account(private, reports, source, evidence):
    report = read(reports / "R6_4_INNER_MATRIX.json", evidence, "report_target")
    lock = read(private / "targetLOCK.json", evidence, "target_lock")
    check_lock(lock, "bgb_en_r6_target_protocol_lock_v1", "LOCKED_BEFORE_TARGET_FITS")
    contract = lock["contract_sha256"]
    require(report.get("schema") == "bgb_en_r6_target_inner_report_v1"
        and report.get("contract_sha256") == contract and report.get("target_lock_sha256") == evidence["target_lock"]
        and report.get("config_sha256") == lock.get("config_sha256")
        and report.get("target_sources_sha256") == lock.get("sources")
        and lock.get("source_config_sha256") == source["source_config_sha256"]
        and lock.get("teacher_refits") == 0 and lock.get("planned_neural_fits") == 90
        and lock.get("outer_test_access") is False, "TARGET_REPORT_OR_LOCK_LINEAGE_CHANGED")
    require(report.get("source", {}).get("complete_source_and_full_transfer") is True
        and report["source"].get("report_sha256") == evidence["report_source"], "TARGET_REPORT_SOURCE_BINDING_CHANGED")
    expected = target_keys() | target_keys(("B0", "B1"))
    rows = exact_rows(report.get("ledger"), ("arm", "partition", "seed"), expected, "TARGET_EXACT_108_LOGICAL_ROWS_REQUIRED")
    qual = read(private / "QUALIFICATION.json", evidence, "target_qualification")
    upstream = read(private / "UPSTREAM_REPLAY_AUDIT.json", evidence, "upstream_replay")
    require(qual.get("status") == "PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER"
        and qual.get("contract_sha256") == contract and qual.get("real_data_backward_passes") == 2
        and qual.get("upstream_replay_audit_sha256") == evidence["upstream_replay"], "TARGET_QUALIFICATION_INVALID")
    check_zero(qual, ("new_optimizer_updates", "new_neural_fits", "source_teacher_or_scalar_fits"), "TARGET_QUALIFICATION_OPTIMIZER_COST_CHANGED")
    require(all(qual.get(k) is True for k in ("query_only_main_and_offset_exact", "missing_gene_gradients_exact_zero",
        "source_parameters_frozen", "original_gene_values_in_dynamic_reread")), "TARGET_QUALIFICATION_MECHANISM_GUARD_FAILED")
    require(upstream.get("status") == "PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY"
        and upstream.get("contract_sha256") == contract and upstream.get("prepared_receipt_sha256")
        == {str(p): lock["prepared_receipt_sha256"][f"outer1_inner{p}"] for p in PARTITIONS}, "TARGET_UPSTREAM_REPLAY_LINEAGE_CHANGED")
    uparts = exact_rows(upstream.get("partitions"), ("partition",), {(p,) for p in PARTITIONS}, "UPSTREAM_EXACT_THREE_PARTITIONS_REQUIRED")
    require(all(r.get("status") == "PASS" and r.get("coefficient_models") == 13
        and finite(r.get("beta_gradient_abs")) and 0 <= r["beta_gradient_abs"] <= 1e-6 for r in uparts.values()), "UPSTREAM_REPLAY_GUARD_FAILED")
    check_zero(upstream, ("new_optimizer_fits", "new_preprocessing_fits"), "UPSTREAM_REPLAY_HAS_REFIT")
    q = report.get("qualification", {})
    require(q.get("passed") is True and q.get("status") == "PASS" and q.get("sha256") == evidence["target_qualification"]
        and q.get("real_data_backward_passes") == 2 and q.get("optimizer_updates") == q.get("neural_fits") == 0
        and q.get("upstream_replay_audit_sha256") == evidence["upstream_replay"], "TARGET_REPORT_QUALIFICATION_BINDING_CHANGED")
    replay = read(private / "REPLAY_AUDIT.json", evidence, "target_replay")
    require(replay.get("status") == "PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY"
        and replay.get("contract_sha256") == contract and replay.get("qualification_sha256") == evidence["target_qualification"]
        and replay.get("upstream_replay_audit_sha256") == evidence["upstream_replay"]
        and all(replay.get(k) == 90 for k in ("expected_units", "available_units", "passed_units"))
        and replay.get("missing_or_unreplayed_units") == [] and replay.get("numerical_risk_tolerance") == 2e-6
        and replay.get("maximum_allowed_changed_pair_credits") == 0, "TARGET_REPLAY_NOT_COMPLETE")
    check_zero(replay, ("failed_units", "new_optimizer_fits", "new_preprocessing_fits"), "TARGET_REPLAY_FAILED_OR_REFIT")
    runits = exact_rows(replay.get("units"), ("arm", "partition", "seed"), target_keys(), "TARGET_REPLAY_EXACT_90_REQUIRED")
    rrefs = exact_rows(replay.get("zero_fit_baseline_reference_replays"), ("arm", "partition", "seed"),
        target_keys(("B0", "B1")), "TARGET_REFERENCE_REPLAY_EXACT_18_REQUIRED")
    require(report.get("actual_replay") == dict(status=replay["status"], complete=True, sha256=evidence["target_replay"]),
        "TARGET_REPORT_ACTUAL_REPLAY_BINDING_CHANGED")
    output = []
    for key, row in sorted(rows.items()):
        arm, part, seed = key
        name = unit_name(key)
        baseline = arm in ("B0", "B1")
        path = private / "references" / (name + ".json") if baseline else private / "cells" / name / "COMPLETE.json"
        label = "target_" + name
        done = read(path, evidence, label + "_complete")
        require(tuple(done.get(k) for k in ("arm", "partition", "seed")) == key
            and row.get("receipt_sha256") == evidence[label + "_complete"] and row.get("actual_replay_passed") is True,
            "TARGET_REPORT_RECEIPT_IDENTITY_OR_SHA_CHANGED")
        if baseline:
            r = rrefs[key]
            require(done.get("status") == "R6_BASELINE_REFERENCE" and done.get("group") == "REFERENCE"
                and done.get("contract_sha256") == contract and done.get("actual_new_optimizer_fits") == done.get("teacher_refits") == 0
                and done.get("deterministic_seed_reference") is (seed != 31)
                and done.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][f"outer1_inner{part}"]
                and row.get("status") == "PASS_ZERO_FIT_REFERENCE" and row.get("group") == "REFERENCE"
                and row.get("actual_new_optimizer_fits") == row.get("actual_optimizer_updates") == 0
                and r.get("status") == "PASS" and r.get("receipt_sha256") == evidence[label + "_complete"]
                and r.get("independent_fixed_anchor_metrics_recomputed") is True and r.get("new_optimizer_fits") == 0,
                "TARGET_ZERO_FIT_REFERENCE_CHANGED")
            output.append(dict(arm=arm, partition=part, seed=seed, group="REFERENCE", formal_optimizer_fits=0,
                formal_optimizer_updates=0, complete_sha256=evidence[label + "_complete"], actual_replay_verified=True))
            continue
        require(not any((path.parent / n).exists() for n in ("ERROR_RECEIPT.json", "FAILED.json")), "TARGET_FAILURE_AND_COMPLETE_CONFLICT")
        c = done.get("contract", {})
        require(done.get("status") == "COMPLETE_R6_TARGET_FIT" and c.get("contract_sha256") == contract
            and tuple(c.get(k) for k in ("arm", "partition", "seed")) == key
            and c.get("config_sha256") == lock["config_sha256"] and c.get("source_config_sha256") == lock["source_config_sha256"]
            and c.get("qualification_sha256") == evidence["target_qualification"]
            and c.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][f"outer1_inner{part}"]
            and c.get("fixed_deterministic_runtime") == lock.get("fixed_deterministic_runtime"), "TARGET_CELL_CONTRACT_CHANGED")
        require(done.get("epochs") == done.get("optimizer_updates") == c.get("epochs") == 80
            and done.get("actual_new_optimizer_fits") == 1 and done.get("primary_epoch_locked_before_training") == 80
            and done.get("outer_test_used") is c.get("outer_test_used") is False
            and row.get("group") == "NEURAL" and row.get("status") == "PASS_FIT_AND_ACTUAL_REPLAY"
            and row.get("actual_new_optimizer_fits") == 1 and row.get("actual_optimizer_updates") == 80,
            "TARGET_FORMAL_TRAINING_COST_CHANGED")
        for value in (done, c):
            check_zero(value, ("teacher_refits", "source_training_updates"), "TARGET_UNPLANNED_UPSTREAM_OPTIMIZER")
        require(digest(done.get("artifact_sha256")) and row.get("artifact_sha256") == done["artifact_sha256"], "TARGET_ARTIFACT_METADATA_CHANGED")
        r = runits[key]
        terminal = read(private / "REPLAY_PRIVATE" / (name + ".PASS.json"), evidence, label + "_replay")
        require(not (private / "REPLAY_PRIVATE" / (name + ".FAIL.json")).exists() and terminal == r
            and r.get("unit") == name and r.get("status") == "PASS" and r.get("contract_sha256") == contract
            and r.get("completion_sha256") == evidence[label + "_complete"]
            and r.get("artifact_sha256") == done["artifact_sha256"]
            and r.get("qualification_sha256") == evidence["target_qualification"]
            and r.get("upstream_replay_audit_sha256") == evidence["upstream_replay"]
            and all(r.get(k) is True for k in ("actual_state_dict_forward_replayed", "source_frozen_weight_hash_replayed", "saved_query_mapping_recomputed"))
            and r.get("changed_pair_credits") == 0 and finite(r.get("maximum_risk_error")) and 0 <= r["maximum_risk_error"] <= 2e-6
            and r.get("new_optimizer_updates") == 0, "TARGET_ACTUAL_REPLAY_TERMINAL_OR_NUMERIC_FAILED")
        elapsed = number(done.get("elapsed_seconds"), "TARGET_ELAPSED_INVALID")
        require(row.get("elapsed_seconds") == elapsed and count(done.get("peak_GPU_bytes"))
            and row.get("peak_GPU_bytes") == done["peak_GPU_bytes"], "TARGET_REPORT_COST_CHANGED")
        output.append(dict(arm=arm, partition=part, seed=seed, group="NEURAL", formal_optimizer_fits=1,
            formal_optimizer_updates=80, elapsed_seconds=elapsed, peak_GPU_allocated_bytes=done["peak_GPU_bytes"],
            complete_sha256=evidence[label + "_complete"], replay_sha256=evidence[label + "_replay"],
            artifact_sha256=done["artifact_sha256"], actual_replay_verified=True))
    for key, expected in (("planned_new_target_fits", 90), ("planned_baseline_reference_rows", 18),
        ("planned_target_updates", 7200), ("confirmed_new_target_fits", 90), ("confirmed_target_updates", 7200), ("unknown_cost_units", 0)):
        require(type(report.get(key)) is int and report[key] == expected, "TARGET_REPORTED_TOTAL_CHANGED")
    require(report.get("counts") == {"PASS_FIT_AND_ACTUAL_REPLAY": 90, "PASS_ZERO_FIT_REFERENCE": 18}, "TARGET_TERMINAL_COUNTS_CHANGED")
    decision = report.get("decision", {})
    require(decision.get("complete_source_and_target_evidence") is True
        and decision.get("formal_target_matrix_denominator") == 90 and decision.get("optional_not_counted_as_failure") is True,
        "TARGET_DECISION_FORMAL_DENOMINATOR_CHANGED")
    schedule = read(private / "TARGET_SCHEDULE_COMPLETE.json", evidence, "target_schedule_complete")
    require(not (private / "TARGET_SCHEDULE_ERROR.json").exists()
        and schedule.get("status") == "COMPLETE_90_R6_TARGET_FITS_AND_REPLAY" and schedule.get("contract_sha256") == contract
        and schedule.get("new_unplanned_candidates") == 0 and schedule.get("optional_controls_excluded_from_this_90_unit_denominator") is True
        and type(schedule.get("worker_ceiling")) is int and 1 <= schedule["worker_ceiling"] <= 4, "TARGET_SCHEDULER_NOT_CLOSED")
    expected_labels = {(unit_name(k),) for k in target_keys()} | {(f"references_p{p}",) for p in PARTITIONS} | {("actual_checkpoint_replay",)}
    srows = exact_rows(schedule.get("records"), ("unit",), expected_labels, "TARGET_SCHEDULER_EXACT_94_TASKS_REQUIRED")
    require(all(r.get("returncode") == 0 for r in srows.values()), "TARGET_SCHEDULER_NONZERO_RETURN")
    return report, output, schedule, lock, qual


def retention_account(private, reports, evidence):
    report = read(reports / "R6_2_OMICS_RETENTION.json", evidence, "report_retention")
    lock = read(private / "retentionLOCK.json", evidence, "retention_lock")
    check_lock(lock, "bgb_en_r6_retention_lock_v1", "LOCKED_BEFORE_ALL_RETENTION_FITS")
    require(report.get("schema") == "bgb_en_r6_retention_report_v1"
        and report.get("status") == "COMPLETE_18_LOGICAL_CELLS_AND_REPLAY"
        and report.get("retention_contract_sha256") == lock["contract_sha256"], "RETENTION_NOT_COMPLETE")
    keys = {(r, p) for r in REPRESENTATIONS for p in PARTITIONS}
    entries = exact_rows(lock.get("representations"), ("representation", "partition"), keys, "RETENTION_LOCK_EXACT_18_REQUIRED")
    all_keys = keys | {(r, p) for r in OLD_RETENTION for p in PARTITIONS}
    rows = exact_rows(report.get("rows"), ("representation", "partition"), all_keys, "RETENTION_REPORT_EXACT_27_REQUIRED")
    historical = exact_rows(lock.get("historical_references"), ("representation", "partition"), all_keys - keys,
        "RETENTION_EXACT_NINE_HISTORICAL_REFERENCES_REQUIRED")
    for key, row in historical.items():
        require(row == rows[key] and row.get("status") == "REUSED_EXACT_PRIOR_ACTUAL_REPLAY"
            and row.get("actual_optimizer_fits") == 0 and row.get("historical_optimizer_fits_referenced_not_new") == 1,
            "RETENTION_OLD_REFERENCE_CHANGED_OR_COUNTED_NEW")
    output = []
    for key, entry in sorted(entries.items()):
        name, part = key
        row = rows[key]
        require(entry.get("optimizer_representation") == name and row.get("optimizer_representation") == name,
            "RETENTION_ALIAS_FORBIDDEN_ALL_18_ACTUAL_FITS")
        directory = private / "cells" / f"{name}_p{part}"
        require(not (directory / "ERROR.json").exists(), "RETENTION_ERROR_AND_COMPLETE_CONFLICT")
        label = f"retention_{name}_p{part}"
        done = read(directory / "COMPLETE.json", evidence, label + "_complete")
        replay = read(directory / "REPLAY.json", evidence, label + "_replay")
        expected_contract = dict(retention_contract_sha256=lock["contract_sha256"], representation=name,
            partition=part, feature_values_sha256=entry["feature_values_sha256"], measured_mask_sha256=entry["measured_mask_sha256"],
            optimizer_representation=name)
        require(done.get("contract") == replay.get("contract") == expected_contract
            and done.get("status") == "COMPLETE_RETENTION_FIT" and done.get("actual_optimizer_fits") == 1
            and done.get("lambda_L2") == 1. and digest(done.get("artifact_sha256"))
            and row.get("status") == "COMPLETE_AND_REPLAYED" and row.get("actual_optimizer_fits") == 1
            and row.get("receipt_sha256") == evidence[label + "_complete"]
            and row.get("replay_sha256") == evidence[label + "_replay"], "RETENTION_RECEIPT_OR_REPORTED_COST_CHANGED")
        require(replay.get("status") == "PASS_RETENTION_COEFFICIENT_AND_INDEPENDENT_EFRON_REPLAY"
            and replay.get("complete_sha256") == evidence[label + "_complete"] and replay.get("new_optimizer_fits") == 0
            and replay.get("lambda_L2") == 1. and replay.get("parameters") == done.get("parameters")
            and finite(replay.get("risk_max_abs_error")) and 0 <= replay["risk_max_abs_error"] <= 1e-12
            and finite(replay.get("independent_gradient_max")) and 0 <= replay["independent_gradient_max"] <= 1e-6,
            "RETENTION_REPLAY_GUARD_FAILED")
        it = done.get("solver_iterations")
        require(count(it) and done.get("solver", {}).get("iterations") == it and row.get("iterations") == it,
            "RETENTION_SOLVER_ITERATIONS_CHANGED")
        output.append(dict(representation=name, partition=part, actual_convex_optimizer_fits=1,
            LBFGS_iterations=it, solver_elapsed_seconds=number(done["solver"].get("elapsed_seconds"), "RETENTION_SOLVER_TIME_INVALID"),
            complete_sha256=evidence[label + "_complete"], replay_sha256=evidence[label + "_replay"],
            artifact_sha256=done["artifact_sha256"], actual_coefficient_replay_verified=True))
    for obj, key, expected in ((report, "planned_logical_cells", 18), (report, "planned_actual_fits", 18),
        (report, "confirmed_new_optimizer_fits", 18), (report, "unknown_fit_count_cells", 0),
        (report, "new_teacher_or_PCA_fits", 0), (report, "new_Set_or_Mamba_training", 0),
        (lock, "planned_logical_cells", 18), (lock, "planned_actual_optimizer_fits", 18), (lock, "new_teacher_or_PCA_fits", 0)):
        require(type(obj.get(key)) is int and obj[key] == expected, "RETENTION_REPORTED_TOTAL_CHANGED")
    return report, output, lock


def read_attention(path, evidence):
    """Equivalent JSON-only read_diagnostics validation, without numpy import."""
    index = read(path, evidence, "report_attention_index")
    require(index.get("storage_schema") == "bgb_en_r6_attention_diagnostics_sharded_index_v1"
        and "rows" not in index and index.get("rows_count") == 90, "ATTENTION_SHARDED_INDEX_REQUIRED")
    desc = exact_rows(index.get("rows_shards"), ("arm",), {(arm,) for arm in ARMS}, "ATTENTION_TEN_SHARDS_REQUIRED")
    rows = [None] * 90
    for (arm,), d in desc.items():
        require(d.get("file") == f"R6_ATTENTION_DIAGNOSTICS_{arm}.json" and d.get("row_count") == 9,
            "ATTENTION_SHARD_PATH_OR_COUNT_CHANGED")
        p = path.parent / d["file"]
        part = read(p, evidence, "report_attention_" + arm)
        require(d.get("sha256") == evidence["report_attention_" + arm] and d.get("bytes") == p.stat().st_size
            and part.get("schema") == "bgb_en_r6_attention_diagnostics_arm_shard_v1" and part.get("arm") == arm
            and part.get("contract_sha256") == index.get("contract_sha256")
            and part.get("logical_object_sha256") == index.get("logical_object_sha256"), "ATTENTION_SHARD_HASH_OR_CONTRACT_CHANGED")
        exact_rows(part.get("rows"), ("arm", "partition", "seed"), target_keys((arm,)), "ATTENTION_SHARD_EXACT_NINE_REQUIRED")
        positions = d.get("row_positions", [])
        require(len(positions) == 9 and len(set(positions)) == 9 and all(type(i) is int and 0 <= i < 90 for i in positions),
            "ATTENTION_SHARD_POSITIONS_INVALID")
        for i, row in zip(positions, part["rows"]):
            require(rows[i] is None, "ATTENTION_DUPLICATE_ROW_POSITION")
            rows[i] = row
    require(all(r is not None for r in rows), "ATTENTION_MISSING_ROW_POSITION")
    value = {k: v for k, v in index.items() if k not in ("storage_schema", "logical_object_sha256", "rows_shards", "rows_count")}
    value["rows"] = rows
    require(canonical(value) == index.get("logical_object_sha256"), "ATTENTION_LOGICAL_OBJECT_HASH_CHANGED")
    return value


def read_risk(path, evidence):
    value = read(path, evidence, "report_risk_index")
    if value.get("schema") == "bgb_en_r6_risk_diagnostics_v1":
        return value
    require(value.get("schema") == "bgb_en_r6_risk_diagnostic_index_v1" and value.get("unit_count") == 90,
        "RISK_DIAGNOSTIC_INDEX_INVALID")
    parts = exact_rows(value.get("parts"), ("arm",), {(arm,) for arm in ARMS}, "RISK_DIAGNOSTIC_TEN_SHARDS_REQUIRED")
    units = []
    for (arm,), d in parts.items():
        require(d.get("file") == f"R6_RISK_DIAGNOSTICS_{arm}.json" and d.get("count") == 9, "RISK_SHARD_PATH_OR_COUNT_CHANGED")
        part = read(path.parent / d["file"], evidence, "report_risk_" + arm)
        require(d.get("sha256") == evidence["report_risk_" + arm]
            and part.get("schema") == "bgb_en_r6_risk_diagnostic_arm_v1" and part.get("arm") == arm
            and part.get("contract_sha256") == value["logical_metadata"]["contract_sha256"], "RISK_SHARD_HASH_OR_CONTRACT_CHANGED")
        exact_rows(part.get("units"), ("arm", "partition", "seed"), target_keys((arm,)), "RISK_SHARD_EXACT_NINE_REQUIRED")
        units.extend(part["units"])
    units.sort(key=lambda row: (ARMS.index(row["arm"]), row["partition"], SEEDS.index(row["seed"])))
    result = dict(value["logical_metadata"], units=units)
    require(canonical(result) == value.get("logical_sha256"), "RISK_LOGICAL_OBJECT_HASH_CHANGED")
    return result


def diagnostic_account(reports, lock, evidence):
    attention = read_attention(reports / "R6_ATTENTION_DIAGNOSTICS.json", evidence)
    require(attention.get("schema") == "bgb_en_r6_attention_diagnostics_v1"
        and attention.get("status") == "COMPLETE_90_FIRST8_INFERENCE_DIAGNOSTICS"
        and attention.get("contract_sha256") == lock["contract_sha256"]
        and attention.get("replay_audit_sha256") == evidence["target_replay"]
        and attention.get("completed_model_cells") == 90 and attention.get("controls_per_cell") == 5
        and attention.get("diagnostic_forward_calls") == 450 and attention.get("fixed_first_valid_rows") == 8,
        "ATTENTION_DIAGNOSTICS_NOT_COMPLETE_OR_LINEAGE_CHANGED")
    exact_rows(attention.get("rows"), ("arm", "partition", "seed"), target_keys(), "ATTENTION_EXACT_90_REQUIRED")
    check_zero(attention, ("new_optimizer_fits", "new_backward_calls"), "ATTENTION_DIAGNOSTICS_HAS_TRAINING")
    require(attention.get("cases_selected_using_outcomes") is False
        and attention.get("independent_patient_count_is_not_720") is True, "ATTENTION_PATIENT_UNIT_SCOPE_CHANGED")
    repeated = sum(r.get("fixed_validation_rows", -1000) for r in attention["rows"])
    unique = attention.get("unique_probed_patients")
    require(repeated == attention.get("repeated_patient_evaluations") == 720
        and count(unique) and 0 < unique <= 24, "ATTENTION_REPEATED_VS_UNIQUE_COUNT_CHANGED")
    risk = read_risk(reports / "R6_RISK_DIAGNOSTICS.json", evidence)
    require(risk.get("schema") == "bgb_en_r6_risk_diagnostics_v1"
        and risk.get("status") == "COMPLETE_SAVED_METADATA_DIAGNOSTICS"
        and risk.get("contract_sha256") == lock["contract_sha256"] and risk.get("report_sha256") == evidence["report_target"]
        and risk.get("model_or_array_or_label_files_read") is False, "RISK_DIAGNOSTICS_NOT_COMPLETE_OR_LINEAGE_CHANGED")
    exact_rows(risk.get("units"), ("arm", "partition", "seed"), target_keys(), "RISK_DIAGNOSTICS_EXACT_90_REQUIRED")
    check_zero(risk, ("new_optimizer_fits", "new_optimizer_updates", "new_model_forwards", "new_backward_passes"),
        "RISK_DIAGNOSTICS_COMPUTE_COST_CHANGED")
    return dict(attention=dict(complete_models=90, actual_forward_calls=450, new_backward_calls=0, new_optimizer_fits=0,
        repeated_model_patient_evaluations=repeated, unique_probed_patients=unique, independent_patients_not_720=True,
        scope="fixed_first_eight_validation_rows_per_partition_reused_across_arms_and_seeds"),
        saved_risk_metadata=dict(complete_models=90, model_forward_calls=0, backward_calls=0, optimizer_fits=0,
            model_or_array_or_label_files_read=False))


def resource_window(path, name, evidence):
    require(path.is_file() and not path.is_symlink(), "RESOURCE_CSV_REQUIRED")
    before = sha(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(set(RESOURCE_FIELDS) <= set(reader.fieldnames or []), "RESOURCE_CSV_SCHEMA_CHANGED")
        rows = [{k: float(r[k]) for k in RESOURCE_FIELDS} for r in reader]
    require(sha(path) == before, "RESOURCE_CSV_STILL_CHANGING")
    evidence[name + "_resource_csv"] = before
    require(len(rows) >= 2 and all(finite(v) and v >= 0 for r in rows for v in r.values()), "RESOURCE_SAMPLES_INVALID")
    require(all(r["GPU_util_percent"] <= 100 and r["CPU_util_percent"] <= 100
        and r["GPU_memory_MiB"] <= r["GPU_total_MiB"] + 1 for r in rows), "RESOURCE_SAMPLE_RANGE_INVALID")
    dt = [b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(rows, rows[1:])]
    require(all(x > 0 for x in dt), "RESOURCE_TIMESTAMPS_NOT_STRICTLY_INCREASING")
    nominal = statistics.median(dt)
    supported = [i for i, value in enumerate(dt) if value <= 3 * nominal]
    duration = sum(dt[i] for i in supported)
    def weighted(key):
        return sum(.5 * (rows[i][key] + rows[i + 1][key]) * dt[i] for i in supported) / duration if duration else None
    return dict(window=name, status="OBSERVED_MONITOR_WINDOW_ONLY", resource_csv_sha256=before, samples=len(rows),
        elapsed_first_sample=rows[0]["elapsed_seconds"], elapsed_last_sample=rows[-1]["elapsed_seconds"],
        observed_window_seconds=rows[-1]["elapsed_seconds"] - rows[0]["elapsed_seconds"],
        supported_interpolation_seconds=duration, excluded_long_gap_seconds=sum(dt) - duration,
        nominal_sample_interval_seconds=nominal, time_weighting="trapezoidal_omit_gaps_over_three_median_intervals",
        visible_device_GPU_util_time_weighted_mean_percent=weighted("GPU_util_percent"),
        visible_device_GPU_util_sample_max_percent=max(r["GPU_util_percent"] for r in rows),
        visible_device_GPU_memory_peak_MiB=max(r["GPU_memory_MiB"] for r in rows),
        visible_device_GPU_total_MiB=max(r["GPU_total_MiB"] for r in rows),
        hostwide_CPU_util_time_weighted_mean_percent=weighted("CPU_util_percent"),
        hostwide_RAM_available_min_MiB=min(r["RAM_available_MiB"] for r in rows),
        observed_filesystem_available_min_MiB=min(r["disk_available_MiB"] for r in rows),
        host_procfs_values_are_not_cgroup_or_process_available_resources=True,
        device_samples_are_not_individual_process_peaks=True, unsampled_periods_not_inferred=True,
        may_include_queue_wait_or_idle=True)


def cgroup_capacity(launch):
    limits = launch.get("cgroup_limits", {})
    memory, cpu = limits.get("memory.max"), limits.get("cpu.max")
    require(isinstance(memory, str) and (memory == "max" or memory.isdecimal()), "CGROUP_MEMORY_LIMIT_MISSING")
    require(isinstance(cpu, str) and len(cpu.split()) == 2, "CGROUP_CPU_LIMIT_MISSING")
    quota, period = cpu.split()
    require(period.isdecimal() and int(period) > 0 and (quota == "max" or quota.isdecimal() and int(quota) > 0), "CGROUP_CPU_LIMIT_INVALID")
    return dict(source_launch_memory_max_raw=memory, source_launch_cpu_max_raw=cpu,
        memory_limit_bytes=None if memory == "max" else int(memory),
        memory_limit_GiB=None if memory == "max" else int(memory) / 1024**3,
        CPU_quota_equivalent_cores=None if quota == "max" else int(quota) / int(period),
        scope="recorded_source_launch_cgroup_capacity_not_current_available_memory",
        current_cgroup_available_memory_not_measured=True)


def arm_status_rows(decision):
    optional = decision.get("optional_followup", {})
    require(set(optional) == {"B10", "B12", "B14", "B15"}, "OPTIONAL_ARM_STATES_MISSING")
    output = []
    for i in range(16):
        arm = "B" + str(i)
        if arm in ARMS:
            status, role, logical, fits, updates, replayed = "COMPLETE_AND_ACTUALLY_REPLAYED", "FORMAL_NEURAL", 9, 9, 720, 9
        elif arm in ("B0", "B1"):
            status, role, logical, fits, updates, replayed = "COMPLETE_ZERO_NEW_FIT_REFERENCE", "ANCHOR_REFERENCE", 9, 0, 0, 9
        else:
            status = optional[arm]
            require(status in ("NOT_ACTIVATED", "PENDING", "TEXT_BLOCKED", "BLOCKED"), "UNKNOWN_OPTIONAL_DECISION_STATUS")
            if arm == "B15":
                require(status in ("TEXT_BLOCKED", "BLOCKED"), "TEXT_ARM_MUST_RETAIN_BLOCKED_STATE")
            role, logical, fits, updates, replayed = "OPTIONAL_OUTSIDE_FORMAL_MATRIX", 0, 0, 0, 0
        output.append(dict(arm=arm, status=status, role=role, formal_logical_rows=logical,
            actual_formal_neural_fits=fits, actual_formal_optimizer_updates=updates, actual_replay_rows=replayed,
            counted_in_90_fit_denominator=arm in ARMS, counted_as_formal_failure=False,
            optional_decision_status=optional.get(arm)))
    return output


def shared_lineage(source_private, source, target_lock, retention_lock, evidence):
    """Different task recipes must share evidence provenance, not recipe SHA."""
    require(digest(target_lock.get("config_sha256")) and digest(retention_lock.get("config_sha256"))
        and retention_lock.get("source_config_sha256") == target_lock.get("source_config_sha256")
        == source["source_config_sha256"], "RETENTION_SHARED_SOURCE_CONFIG_CHANGED")
    controls = {"MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH"}
    rstates, tstates = retention_lock.get("source_full_states", {}), target_lock.get("source_full_states", {})
    require(set(rstates) == controls and controls <= set(tstates), "RETENTION_SHARED_FULL_SOURCE_STATES_MISSING")
    for control in sorted(controls):
        origin = rstates[control]
        label = f"source_S7_{control}_d-1_s31"
        directory = Path(source_private) / "source_cells" / f"S7_{control}_h-1_seed31"
        done = read(directory / "COMPLETE.json", evidence, label + "_complete")
        require(origin == tstates[control] and origin.get("source_config_sha256") == source["source_config_sha256"]
            and origin.get("source_control") == control and origin.get("external_full_source_seed") == 31
            and origin.get("completion_sha256") == evidence[label + "_complete"]
            and origin.get("replay_sha256") == evidence[label + "_replay"]
            and origin.get("checkpoint_sha256") == done["artifact_sha256"].get("state_epoch080.private.pt")
            and digest(origin.get("complete_source_state_sha256")), "RETENTION_SHARED_FULL_SOURCE_PROVENANCE_CHANGED")
    expected_fingerprints = {"cohort_sha256", "endpoints_sha256"}
    fingerprints = retention_lock.get("data_fingerprints", {})
    require(set(fingerprints) == expected_fingerprints and fingerprints == target_lock.get("data_fingerprints")
        and all(digest(v) for v in fingerprints.values()), "RETENTION_SHARED_COHORT_ENDPOINT_FINGERPRINT_CHANGED")
    require(digest(retention_lock.get("launch_gate_sha256"))
        and retention_lock["launch_gate_sha256"] == target_lock.get("launch_gate_sha256"), "RETENTION_SHARED_LAUNCH_GATE_CHANGED")
    stems = {f"outer1_inner{p}" for p in PARTITIONS}
    origins, prepared = retention_lock.get("prepared_origins", {}), target_lock.get("prepared_receipt_sha256", {})
    require(set(origins) == set(prepared) == stems and all(
        origins[stem].get("prepared_receipt_sha256") == prepared[stem] and digest(prepared[stem])
        and digest(origins[stem].get("prepared_artifact_sha256")) for stem in stems),
        "RETENTION_SHARED_PREPARED_ORIGIN_CHANGED")
    return dict(target_recipe_sha256=target_lock["config_sha256"], retention_recipe_sha256=retention_lock["config_sha256"],
        task_recipes_are_separately_bound_not_required_equal=True,
        shared_source_config_sha256=source["source_config_sha256"], verified_full_source_states=3,
        verified_cohort_endpoint_fingerprints=2, verified_prepared_receipt_origins=3,
        shared_launch_gate_sha256=retention_lock["launch_gate_sha256"],
        full_source_completion_replay_and_checkpoint_metadata_SHA_bound=True,
        no_model_or_array_or_label_files_read=True)


def build(source_private, target_private, retention_private, reports):
    source_private, target_private, retention_private, reports = map(Path,
        (source_private, target_private, retention_private, reports))
    evidence = {}
    source, srows, sschedule, squal = source_account(source_private, reports, evidence)
    target, trows, tschedule, lock, tqual = target_account(target_private, reports, source, evidence)
    retention, rrows, rlock = retention_account(retention_private, reports, evidence)
    shared = shared_lineage(source_private, source, lock, rlock, evidence)
    qualifier_code = "scripts/bgb_en_r6_qualify_target.py"
    require(lock.get("sources", {}).get(qualifier_code) == sha(ROOT / qualifier_code),
        "QUALIFICATION_CODE_DERIVED_PROBE_SOURCE_CHANGED")
    diag = diagnostic_account(reports, lock, evidence)
    launch = read(source_private / "SOURCE_SCHEDULE_LAUNCH.private.json", evidence, "source_schedule_launch")
    resource = dict(schema="bgb_en_r6_resource_summary_v1", status="COMPLETE_RECORDED_WINDOWS",
        cgroup_capacity_from_source_launch=cgroup_capacity(launch),
        windows=[resource_window(source_private / "resource_samples.csv", "source", evidence),
            resource_window(target_private / "resource_samples.csv", "target", evidence)],
        scheduler_scopes=dict(source_elapsed_seconds=number(sschedule.get("elapsed_seconds"), "SOURCE_SCHEDULER_ELAPSED_INVALID"),
            target_elapsed_seconds=number(tschedule.get("elapsed_seconds"), "TARGET_SCHEDULER_ELAPSED_INVALID")),
        summed_individual_durations=dict(source_fit_seconds=sum(r["elapsed_seconds"] for r in srows if r["formal_optimizer_fits"]),
            source_random_reference_seconds=sum(r["elapsed_seconds"] for r in srows if not r["formal_optimizer_fits"]),
            target_fit_seconds=sum(r["elapsed_seconds"] for r in trows if r["group"] == "NEURAL"),
            retention_solver_seconds=sum(r["solver_elapsed_seconds"] for r in rrows)),
        summed_fit_seconds_are_not_walltime=True, scheduler_elapsed_not_assumed_sum_of_jobs=True,
        global_end_to_end_makespan_seconds=None, unsampled_or_historical_time_not_invented=True)
    formal_fits = sum(r["formal_optimizer_fits"] for r in srows + trows)
    formal_updates = sum(r["formal_optimizer_updates"] for r in srows + trows)
    require((formal_fits, formal_updates) == (225, 18000), "R6_FORMAL_TOTAL_MISMATCH")
    iterations = sum(r["LBFGS_iterations"] for r in rrows)
    # Retention iterations are actual receipt sums, not a neural-update count.
    arm_rows = arm_status_rows(target["decision"])
    ledger = dict(schema="bgb_en_r6_execution_ledger_v1", status="COMPLETE_FORMAL_R6_ACCOUNTING",
        scope="R6_ONLY_SOURCE_TARGET_RETENTION", ledger_code_sha256=sha(__file__),
        scientific_contracts=dict(source_config_sha256=source["source_config_sha256"],
            target_contract_sha256=lock["contract_sha256"], retention_contract_sha256=rlock["contract_sha256"]),
        shared_evidence_lineage=shared,
        formal_neural=dict(source_logical_units=147, source_actual_fits=135, source_optimizer_updates=10800,
            source_random_zero_fit_references=12, target_actual_fits=90, target_optimizer_updates=7200,
            target_zero_fit_anchor_references=18, combined_actual_fits=formal_fits, combined_optimizer_updates=formal_updates,
            completed_actual_source_replay_units=147, completed_actual_target_replay_units=90,
            teacher_or_beta_refits=0, epochs_per_optimized_cell=80, incomplete_or_unknown_formal_units=0),
        separate_convex_retention=dict(logical_units=18, actual_optimizer_fits=18, actual_coefficient_replays=18,
            exact_representation_fit_aliases=0, historical_zero_new_fit_references=9, LBFGS_iterations=iterations,
            teacher_PCA_Set_Mamba_refits=0, iterations_are_not_neural_optimizer_updates=True),
        separate_qualification=dict(source=dict(optimizer_instances=1, optimizer_updates=2, formal_source_fits=0,
            recorded_elapsed_seconds=number(squal.get("elapsed_seconds"), "SOURCE_QUALIFICATION_ELAPSED_INVALID")),
            target=dict(optimizer_fits=0, optimizer_updates=0, recorded_explicit_backward_passes=2,
                additional_RNA_input_autograd_grad_probes_code_derived=1,
                probe_count_is_not_a_receipt_execution_counter=True,
                code_evidence="scripts/bgb_en_r6_qualify_target.py",
                code_sha256=sha(ROOT / "scripts/bgb_en_r6_qualify_target.py"))),
        separate_diagnostics=diag,
        excluded_cost_scopes=dict(R5_entirely_excluded=True, previous_phases_and_historical_unknown_costs_excluded=True,
            optional_B10_B12_B14_and_blocked_text_B15_not_failures_or_formal_denominator=True,
            actual_checkpoint_replay_forward_call_total="NOT_RECORDED_NOT_INFERRED_FROM_UNIT_COUNT"),
        source_units=srows, target_units=trows, retention_units=rrows,
        arm_execution_status=arm_rows, evidence_sha256=evidence,
        accountant_runtime=dict(optimizer_fits=0, optimizer_updates=0, model_forwards=0, backwards=0,
            model_array_label_files_read=False, new_risk_or_performance_analysis=False,
            existing_actual_replays_bound_by_metadata_SHA=True, model_artifact_bytes_rehashed_by_this_tool=False),
        limits=dict(no_new_model_selection=True, no_outer_metrics_read=True,
            source_and_target_success_do_not_establish_DSS_or_biological_mechanism=True,
            resource_windows_do_not_prove_peak_or_full_run_utilization=True))
    resource["evidence_sha256"] = {k: v for k, v in evidence.items() if "resource" in k or "schedule" in k}
    return ledger, resource, arm_rows


def write_outputs(output, ledger, resource, arms):
    output = Path(output).resolve()
    require(not any(output == p.resolve() or output.is_relative_to(p.resolve())
        for p in (ROOT / "reports/bgb_en", ROOT / "reports/bgb_en_r5")), "HISTORICAL_REPORT_NAMESPACES_PROTECTED")
    require(not any((output / n).exists() or (output / (n + ".writing")).exists() for n in NAMES), "LEDGER_EXISTING_OUTPUT_PRESERVED")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in zip(NAMES[:2], (ledger, resource)):
        tmp = output / (name + ".writing")
        with tmp.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temp = output / (NAMES[2] + ".writing")
    with temp.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(arms[0]))
        writer.writeheader(); writer.writerows(arms)
    for name in NAMES:
        (output / (name + ".writing")).replace(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-private", "target-private", "retention-private", "reports", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    # build must succeed before any output file or directory is created.
    ledger, resource, arms = build(args.source_private, args.target_private, args.retention_private, args.reports)
    write_outputs(args.output, ledger, resource, arms)
    print(json.dumps(dict(status=ledger["status"], formal_neural_fits=225, formal_neural_updates=18000,
        convex_retention_fits=18, LBFGS_iterations=ledger["separate_convex_retention"]["LBFGS_iterations"],
        accountant_optimizer_fits=0)))


if __name__ == "__main__":
    main()

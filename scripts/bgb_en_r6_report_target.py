"""R6 target inner-protocol report from anonymous metadata and artifact SHA.

No model, patient arrays, risk vectors, cohort labels or previous repair
modules are imported. Direct pair repair/damage is accepted only from the
independent saved-state replay's actual paired-risk records.
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
BASELINES = ("B0", "B1")
SEEDS, PARTITIONS = (31, 47, 83), (0, 1, 2)
PAIR_REFERENCES = ("B1", "B3", "B4", "B5", "B6", "B8", "B9", "B11", "B13")
COMPONENTS = ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
FULL_CONTROLS = ("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH")


def require(value, code):
    if not value:
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temp = Path(path).with_name(Path(path).name + ".writing")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def mean(values):
    return statistics.mean(values) if values and all(finite(v) for v in values) else None


def unit_name(arm, partition, seed):
    return f"{arm}_p{partition}_seed{seed}"


def validate_metrics(metrics):
    require(isinstance(metrics, dict) and all(finite(metrics.get(k)) and metrics[k] >= 0 for k in COMPONENTS),
            "TARGET_CONCORDANCE_COMPONENT_MISSING_OR_INVALID")
    for label, numerator, denominator in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
        require(metrics[numerator] <= metrics[denominator] + 1e-10, "TARGET_CONCORDANCE_NUMERATOR_EXCEEDS_DENOMINATOR")
        expected = metrics[numerator] / metrics[denominator] if metrics[denominator] else None
        require(metrics.get(label) is None if expected is None else finite(metrics.get(label)) and abs(metrics[label] - expected) <= 1e-10,
                "TARGET_CONCORDANCE_RATIO_MISMATCH")
    return metrics


def combine_components(rows):
    require(len(rows) == 3, "ALL_THREE_PARTITIONS_REQUIRED")
    for row in rows:
        validate_metrics(row)
    result = {key: sum(row[key] for row in rows) for key in COMPONENTS}
    result.update(Harrell=result["Harrell_num"] / result["Harrell_den"] if result["Harrell_den"] else None,
                  Uno60=result["Uno_num"] / result["Uno_den"] if result["Uno_den"] else None)
    return result


def aggregate_arm(arm, metrics):
    available = [(p, s) for p in PARTITIONS for s in SEEDS if (arm, p, s) in metrics]
    answer = dict(arm=arm, expected_cells=9, complete_cells=len(available),
        status="COMPLETE_NINE_CELLS" if len(available) == 9 else "INCOMPLETE_NO_OVERALL_MEAN",
        Harrell=None, Uno60=None, per_seed=[])
    if len(available) != 9:
        return answer
    for seed in SEEDS:
        answer["per_seed"].append(dict(seed=seed, **combine_components([metrics[(arm, p, seed)] for p in PARTITIONS])))
    answer.update(Harrell=mean([r["Harrell"] for r in answer["per_seed"]]),
                  Uno60=mean([r["Uno60"] for r in answer["per_seed"]]),
                  aggregation="within_partition_components_sum_per_seed_then_equal_three_seed_mean",
                  cross_partition_risk_ranking=False)
    return answer


def validate_inputs(config_path, lock_path):
    config, lock = read(config_path), read(lock_path)
    require(config.get("schema") == "bgb_en_r6_inner_v1" and tuple(config.get("trainable_arms", [])) == ARMS
            and config.get("formal_target_neural_fits") == 90 and config["prototype"]["inner_partitions"] == list(PARTITIONS)
            and config["prototype"]["neural_seeds"] == list(SEEDS) and config["prototype"]["outer_test_access"] is False,
            "TARGET_FINITE_PROTOCOL_CHANGED")
    require(lock.get("schema") == "bgb_en_r6_target_protocol_lock_v1" and lock.get("status") == "LOCKED_BEFORE_TARGET_FITS",
            "TARGET_IMMUTABLE_LOCK_MISSING")
    contents = {k: v for k, v in lock.items() if k not in ("contract_sha256", "created_UTC")}
    require(canonical_hash(contents) == lock.get("contract_sha256") and lock.get("config_sha256") == sha256(config_path)
            and lock.get("planned_neural_fits") == 90 and lock.get("outer_test_access") is False,
            "TARGET_LOCK_DIGEST_OR_CONFIG_CHANGED")
    for relative, expected in lock.get("sources", {}).items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha256(path) == expected, "TARGET_LOCKED_SOURCE_SHA_CHANGED")
    require(lock.get("sources") and lock.get("teacher_refits") == 0, "TARGET_SOURCE_OR_NO_REFIT_CONTRACT_MISSING")
    return config, lock


def read_source(source_decision_path, source_report_path, lock):
    result = dict(local_support=None, complete_source_and_full_transfer=False,
        decision_sha256=None, report_sha256=None, status="MISSING_SOURCE_METADATA", costs={})
    if not Path(source_decision_path).exists() or not Path(source_report_path).exists():
        return result
    decision, report = read(source_decision_path), read(source_report_path)
    result.update(decision_sha256=sha256(source_decision_path), report_sha256=sha256(source_report_path))
    require(decision.get("status") in ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE")
            and decision.get("source_mechanism_only") is True and report.get("decision") == decision,
            "SOURCE_DECISION_SCOPE_INVALID")
    result["local_support"] = True if decision["status"] == "SUPPORTED" else False if decision["status"] == "NOT_SUPPORTED" else None
    full = report.get("full_source_transfer_units", [])
    by_control = {r.get("control"): r for r in full}
    complete = (report.get("status") == "COMPLETE_147_SOURCE_UNITS_AND_REPLAY"
        and report.get("source_config_sha256") == lock["source_config_sha256"]
        and report.get("counts", {}).get("PASS_COMPLETE_AND_REPLAYED") == 147
        and report.get("confirmed_formal_optimizer_fits") == 135
        and report.get("confirmed_formal_optimizer_updates") == 10800 and report.get("unknown_update_cost_units") == 0)
    for control in FULL_CONTROLS:
        origin = lock.get("source_full_states", {}).get(control, {})
        row = by_control.get(control, {})
        complete &= bool(origin and row.get("arm") == "S7" and row.get("heldout_donor") == -1 and row.get("seed") == 31
            and row.get("status") == "PASS_COMPLETE_AND_REPLAYED"
            and row.get("receipt_sha256") == origin.get("completion_sha256")
            and row.get("replay_sha256") == origin.get("replay_sha256")
            and origin.get("source_config_sha256") == lock["source_config_sha256"])
    costs = {key: report.get(key) for key in ("planned_formal_optimizer_fits", "confirmed_formal_optimizer_fits",
        "confirmed_formal_optimizer_updates", "unknown_update_cost_units")}
    costs = {key: value if finite(value) or value is None else None for key, value in costs.items()}
    costs["qualification_cost_separate_from_formal_matrix"] = {key: report.get("qualification_cost_separate_from_formal_matrix", {}).get(key)
        for key in ("provided", "optimizer_updates", "optimizer_instances", "formal_source_fits", "receipt_sha256")}
    result.update(status=decision["status"], complete_source_and_full_transfer=bool(complete), costs=costs)
    return result


def read_qualification(private, lock):
    path = private / "QUALIFICATION.json"
    if not path.exists():
        return dict(status="MISSING", passed=False, sha256=None, optimizer_updates=None, neural_fits=None)
    value = read(path)
    passed = value.get("status") == "PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER" and value.get("contract_sha256") == lock["contract_sha256"]
    passed &= all(value.get(k) is True for k in ("query_only_main_and_offset_exact", "missing_gene_gradients_exact_zero",
                                                "source_parameters_frozen", "original_gene_values_in_dynamic_reread"))
    passed &= value.get("new_optimizer_updates") == value.get("new_neural_fits") == value.get("source_teacher_or_scalar_fits") == 0
    up_path = private / "UPSTREAM_REPLAY_AUDIT.json"
    passed &= up_path.exists() and value.get("upstream_replay_audit_sha256") == sha256(up_path)
    if up_path.exists():
        up = read(up_path)
        parts = up.get("partitions", [])
        passed &= (up.get("status") == "PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY"
            and up.get("contract_sha256") == lock["contract_sha256"] and len(parts) == 3
            and {r.get("partition") for r in parts} == set(PARTITIONS)
            and all(r.get("status") == "PASS" and r.get("coefficient_models") == 13
                and finite(r.get("beta_gradient_abs")) and r["beta_gradient_abs"] <= 1e-6 for r in parts)
            and up.get("new_optimizer_fits") == up.get("new_preprocessing_fits") == 0
            and up.get("prepared_receipt_sha256") == {str(p): lock["prepared_receipt_sha256"][f"outer1_inner{p}"] for p in PARTITIONS})
    return dict(status="PASS" if passed else "INVALID", passed=bool(passed), sha256=sha256(path),
        optimizer_updates=value.get("new_optimizer_updates"), neural_fits=value.get("new_neural_fits"),
        real_data_backward_passes=value.get("real_data_backward_passes"),
        upstream_replay_audit_sha256=value.get("upstream_replay_audit_sha256"),
        shared_target_initialization_sha256=value.get("shared_target_initialization_sha256"))


def read_replay(private, lock, qualification):
    path = private / "REPLAY_AUDIT.json"
    if not path.exists():
        return dict(status="MISSING", complete=False, sha256=None, units={}, baseline_units={}, direct_rows=[], raw_gene_rows=[])
    evidence = read(path)
    require(evidence.get("contract_sha256") == lock["contract_sha256"]
            and evidence.get("qualification_sha256") == qualification["sha256"], "TARGET_REPLAY_LINEAGE_CHANGED")
    units = evidence.get("units", [])
    require(len({r.get("unit") for r in units}) == len(units), "TARGET_REPLAY_DUPLICATE_UNITS")
    complete = evidence.get("status") == "PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY" and evidence.get("expected_units") == evidence.get("available_units") == evidence.get("passed_units") == 90
    complete &= evidence.get("failed_units") == evidence.get("new_optimizer_fits") == evidence.get("new_preprocessing_fits") == 0
    complete &= evidence.get("numerical_risk_tolerance") == 2e-6 and evidence.get("maximum_allowed_changed_pair_credits") == 0
    complete &= len(units) == 90 and all(r.get("status") == "PASS" for r in units)
    complete &= evidence.get("upstream_replay_audit_sha256") == qualification.get("upstream_replay_audit_sha256")
    references = evidence.get("zero_fit_baseline_reference_replays", [])
    ref_keys = [(r.get("arm"), r.get("partition"), r.get("seed")) for r in references]
    require(len(set(ref_keys)) == len(ref_keys), "TARGET_REPLAY_DUPLICATE_BASELINE_REFERENCES")
    complete &= len(references) == 18 and all(r.get("status") == "PASS" for r in references)
    return dict(status=evidence.get("status", "INVALID"), complete=bool(complete), sha256=sha256(path),
        units={r["unit"]: r for r in units}, baseline_units={k: r for k, r in zip(ref_keys, references)},
        direct_rows=evidence.get("paired_retrained_controls", []),
        raw_gene_rows=evidence.get("paired_raw_gene_controls", []),
        upstream_replay_audit_sha256=evidence.get("upstream_replay_audit_sha256"))


def _contract(contract, arm, partition, seed, lock, config_sha, qualification):
    return (contract.get("contract_sha256") == lock["contract_sha256"] and contract.get("arm") == arm
        and contract.get("partition") == partition and contract.get("seed") == seed
        and contract.get("config_sha256") == config_sha and contract.get("source_config_sha256") == lock["source_config_sha256"]
        and contract.get("qualification_sha256") == qualification["sha256"]
        and contract.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][f"outer1_inner{partition}"]
        and contract.get("epochs") == 80 and contract.get("teacher_refits") == 0
        and contract.get("source_training_updates") == 0 and contract.get("outer_test_used") is False
        and contract.get("fixed_deterministic_runtime") == lock.get("fixed_deterministic_runtime"))


def scalar_diagnostics(value):
    """Explicit scalar allowlist; attention is not histological ground truth."""
    result = dict(frozen_interventions={}, risk_magnitude={}, attention_sensitivity={})
    for control in ("matched_query", "wrong_query", "mean_query", "interaction_off", "value_off", "attention_permutation", "whole_omics"):
        record = value.get("frozen_controls", {}).get(control)
        if record is not None:
            ratio = record.get("risk_SD_ratio")
            require(finite(ratio) and ratio >= 0 and record.get("frozen_not_retrained") is True,
                    "TARGET_FROZEN_RESPONSE_METADATA_INVALID")
            result["frozen_interventions"][control] = dict(risk_SD_ratio=ratio, frozen_not_retrained=True,
                whole_modality_replaced=record.get("whole_modality_replaced") is True)
    for split in ("train", "validation"):
        record = value.get("risk_diagnostics", {}).get(split, {})
        result["risk_magnitude"][split] = {key: record[key] for key in (
            "SD_total_delta_over_offset", "SD_cross_delta_over_offset") if key in record and finite(record[key])}
    probe = value.get("attention_probe", {})
    for key in ("patients", "valid_WSI_program_rows", "patch_attention_TV", "risk_difference_SD", "delta_E_change_L2",
                "delta_G_change_L2", "query_source_mask_changed_rows", "reverse_weight_absolute_change_sum"):
        if key in probe:
            require(probe[key] is None or finite(probe[key]), "TARGET_ATTENTION_SCALAR_METADATA_INVALID")
            result["attention_sensitivity"][key] = probe[key]
    for key in ("original_values_and_main_effects_preserved", "frozen_not_retrained", "histology_or_gene_causality_claimed"):
        if key in probe:
            require(isinstance(probe[key], bool), "TARGET_ATTENTION_BOOLEAN_METADATA_INVALID")
            result["attention_sensitivity"][key] = probe[key]
    result["frozen_is_dependency_not_retraining"] = True
    result["attention_is_not_histological_truth"] = True
    return result


def read_cell(private, arm, partition, seed, lock, config_sha, qualification, replay):
    name = unit_name(arm, partition, seed)
    directory = private / "cells" / name
    row = dict(arm=arm, partition=partition, seed=seed, group="NEURAL", status="MISSING", expected_updates=80,
        actual_optimizer_updates=None, actual_new_optimizer_fits=None, artifact_sha256=None, receipt_sha256=None,
        actual_replay_passed=False, elapsed_seconds=None, peak_GPU_bytes=None)
    if not (directory / "COMPLETE.json").exists():
        error_path = directory / "ERROR_RECEIPT.json"
        if error_path.exists():
            try:
                error = read(error_path)
                valid = _contract(error.get("contract", {}), arm, partition, seed, lock, config_sha, qualification)
                row["status"] = "ERROR" if valid else "INVALID_ERROR_RECEIPT"
                if valid:
                    updates = error.get("optimizer_updates_completed")
                    allowed_types = {"RuntimeError", "ValueError", "TypeError", "KeyError", "FileNotFoundError", "OSError",
                        "OutOfMemoryError", "AssertionError", "Exception", "KeyboardInterrupt", "MemoryError"}
                    row.update(actual_optimizer_updates=updates if type(updates) is int and 0 <= updates <= 80 else None,
                        actual_new_optimizer_fits=1 if type(updates) is int and updates > 0 else 0 if updates == 0 else None,
                        error_type=error.get("error_type") if error.get("error_type") in allowed_types else "OtherError")
            except (ValueError, KeyError, TypeError, OSError):
                row["status"] = "INVALID_ERROR_RECEIPT"
        elif directory.exists():
            row["status"] = "INCOMPLETE_PRESERVED"
        return row, None, None
    try:
        path, artifact = directory / "COMPLETE.json", directory / "model.pt"
        value = read(path)
        require(not (directory / "ERROR_RECEIPT.json").exists() and value.get("status") == "COMPLETE_R6_TARGET_FIT"
                and (value.get("arm"), value.get("partition"), value.get("seed")) == (arm, partition, seed)
                and _contract(value.get("contract", {}), arm, partition, seed, lock, config_sha, qualification), "TARGET_CELL_LINEAGE_CHANGED")
        require(value.get("epochs") == value.get("optimizer_updates") == 80 and value.get("actual_new_optimizer_fits") == 1
                and value.get("teacher_refits") == value.get("source_training_updates") == 0
                and value.get("outer_test_used") is False and value.get("primary_epoch_locked_before_training") == 80,
                "TARGET_CELL_TRAINING_BUDGET_CHANGED")
        require(artifact.is_file() and value.get("artifact_sha256") == sha256(artifact), "TARGET_CELL_ARTIFACT_SHA_CHANGED")
        source_key = "RANDOM_NO_ST" if arm in ("B2", "B13") else "WRONG_DONOR" if arm == "B8" else "MATCHED"
        origin = value.get("source_origin", {})
        require(origin.get("source_key") == source_key and origin.get("source_origin") == lock["source_full_states"].get(source_key),
                "TARGET_FROZEN_SOURCE_ORIGIN_CHANGED")
        metrics = validate_metrics(value.get("metrics"))
        require(value.get("metrics_vs", {}).get("CP_SCALAR") == metrics, "TARGET_CP_SCALAR_METRICS_MISMATCH")
        for tag in ("CP_STRONG", "PFI_frozen_DSS_risk"):
            validate_metrics(value.get("metrics_vs", {}).get(tag))
        curves = value.get("curves", [])
        require([r.get("epoch") for r in curves] == [20, 40, 80]
                and [r.get("optimizer_updates_completed") for r in curves] == [20, 40, 80], "TARGET_BUDGET_CURVE_CHANGED")
        budget = value.get("parameters", {})
        require(finite(value.get("elapsed_seconds")) and value["elapsed_seconds"] >= 0
                and type(value.get("peak_GPU_bytes")) is int and value["peak_GPU_bytes"] >= 0
                and all(type(budget.get(k)) is int and budget[k] >= 0 for k in ("registered", "requires_grad"))
                and budget["requires_grad"] <= budget["registered"]
                and digest(value.get("initialization", {}).get("target_state_sha256")), "TARGET_COST_OR_INITIALIZATION_METADATA_INVALID")
        row.update(actual_optimizer_updates=80, actual_new_optimizer_fits=1, artifact_sha256=value["artifact_sha256"],
            receipt_sha256=sha256(path), elapsed_seconds=value.get("elapsed_seconds"), peak_GPU_bytes=value.get("peak_GPU_bytes"),
            parameters={k: value.get("parameters", {}).get(k) for k in ("registered", "requires_grad")},
            target_initialization_sha256=value.get("initialization", {}).get("target_state_sha256"),
            budget_snapshots=[{key: record.get(key) for key in ("epoch", "optimizer_updates_completed", "cox_loss",
                "gradient_norm", "risk_cache_replay_max", "cross_cache_replay_max", "train_SD_delta_over_offset", "valid_SD_delta_over_offset")}
                for record in curves])
        require(all(v is None or finite(v) for record in row["budget_snapshots"] for v in record.values()),
                "TARGET_BUDGET_DIAGNOSTIC_NONSCALAR_OR_NONFINITE")
        row["mechanism_diagnostics"] = scalar_diagnostics(value)
        current = replay["units"].get(name)
        if current is None:
            row["status"] = "FIT_COMPLETE_REPLAY_PENDING"
        elif current.get("status") != "PASS":
            row["status"] = "ACTUAL_REPLAY_FAILED"
        else:
            require(current.get("arm") == arm and current.get("partition") == partition and current.get("seed") == seed
                and current.get("contract_sha256") == lock["contract_sha256"] and current.get("completion_sha256") == sha256(path)
                and current.get("artifact_sha256") == value["artifact_sha256"] and current.get("qualification_sha256") == qualification["sha256"]
                and current.get("upstream_replay_audit_sha256") == qualification["upstream_replay_audit_sha256"]
                and current.get("actual_state_dict_forward_replayed") is True and current.get("source_frozen_weight_hash_replayed") is True
                and current.get("saved_query_mapping_recomputed") is True and current.get("changed_pair_credits") == 0
                and finite(current.get("maximum_risk_error")) and current["maximum_risk_error"] <= 2e-6
                and current.get("new_optimizer_updates") == 0, "TARGET_ACTUAL_REPLAY_IDENTITY_OR_NUMERICS_CHANGED")
            row.update(status="PASS_FIT_AND_ACTUAL_REPLAY", actual_replay_passed=True)
        return row, metrics, value
    except (ValueError, KeyError, TypeError, OSError) as error:
        row["status"] = "INVALID"
        code = str(error) if isinstance(error, ValueError) else type(error).__name__
        row["validation_failure_code"] = code if code.replace("_", "").isalnum() else type(error).__name__
        return row, None, None


def read_reference(private, arm, partition, seed, lock, replay):
    row = dict(arm=arm, partition=partition, seed=seed, group="REFERENCE", status="MISSING", expected_updates=0,
        actual_optimizer_updates=None, actual_new_optimizer_fits=None, actual_replay_passed=False,
        elapsed_seconds=None, peak_GPU_bytes=None)
    path = private / "references" / (unit_name(arm, partition, seed) + ".json")
    if not path.exists():
        return row, None, None
    try:
        value = read(path)
        require(value.get("status") == "R6_BASELINE_REFERENCE" and value.get("group") == "REFERENCE"
                and (value.get("arm"), value.get("partition"), value.get("seed")) == (arm, partition, seed)
                and value.get("contract_sha256") == lock["contract_sha256"] and value.get("actual_new_optimizer_fits") == 0
                and value.get("teacher_refits") == 0 and value.get("deterministic_seed_reference") is (seed != 31)
                and value.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][f"outer1_inner{partition}"],
                "TARGET_ZERO_FIT_REFERENCE_LINEAGE_CHANGED")
        metrics = validate_metrics(value.get("metrics"))
        for label in ("CP_STRONG", "CP_SCALAR", "PFI_frozen_DSS_risk"):
            validate_metrics(value.get("metrics_vs", {}).get(label))
        require(metrics == value["metrics_vs"]["CP_SCALAR"], "TARGET_REFERENCE_CP_SCALAR_METRICS_CHANGED")
        row.update(actual_optimizer_updates=0, actual_new_optimizer_fits=0, receipt_sha256=sha256(path))
        replayed = replay["baseline_units"].get((arm, partition, seed))
        if replayed is None:
            row["status"] = "REFERENCE_REPLAY_PENDING"
        else:
            require(replayed.get("status") == "PASS" and replayed.get("receipt_sha256") == sha256(path)
                    and replayed.get("independent_fixed_anchor_metrics_recomputed") is True
                    and replayed.get("new_optimizer_fits") == 0, "TARGET_REFERENCE_ACTUAL_REPLAY_CHANGED")
            row.update(status="PASS_ZERO_FIT_REFERENCE", actual_replay_passed=True)
        return row, metrics, value
    except (ValueError, KeyError, TypeError, OSError):
        row.update(status="INVALID", validation_failure_code="TARGET_REFERENCE_METADATA_INVALID")
        return row, None, None


def direct_pairs(replay, valid_metrics):
    if not replay["complete"]:
        return {}, [], dict(expected_rows=90, valid_rows=0, missing_rows=90, validation_failures=[], complete=False,
                            reason="COMPLETE_ACTUAL_REPLAY_REQUIRED_FOR_DIRECT_CREDIT_EXPORT")
    rows = replay["direct_rows"] + replay["raw_gene_rows"]
    expected = {("B7", reference, p, s) for reference in PAIR_REFERENCES for p in PARTITIONS for s in SEEDS}
    expected |= {("B6", "B5", p, s) for p in PARTITIONS for s in SEEDS}
    indexed, public, failures = {}, [], []
    for row in rows:
        try:
            key = (row["candidate"], row["reference"], row["partition"], row["seed"])
            require(key in expected and key not in indexed and row.get("direct_risk_pair_credits") is True
                    and row.get("bootstrap_or_CI_computed") is False, "TARGET_DIRECT_PAIR_IDENTITY_CHANGED")
            candidate, reference, part, seed = key
            actual = validate_metrics(row.get("candidate_vs_reference"))
            a, b = valid_metrics[(candidate, part, seed)], valid_metrics[(reference, part, seed)]
            require(all(abs(actual[k] - a[k]) <= 1e-10 for k in COMPONENTS), "TARGET_DIRECT_PAIR_CANDIDATE_COMPONENT_CHANGED")
            require(all(abs(a[k] - b[k]) <= 1e-10 for k in ("Harrell_den", "Uno_den"))
                    and finite(row.get("risk_SD_ratio")) and row["risk_SD_ratio"] >= 0, "TARGET_DIRECT_PAIR_SCOPE_OR_RISK_SCALE_CHANGED")
            for credit_name, damage_name, metric in (("repair", "damage", "Harrell"), ("Uno_repair", "Uno_damage", "Uno60")):
                require(finite(actual.get(credit_name)) and finite(actual.get(damage_name))
                        and 0 <= actual[credit_name] <= 1 and 0 <= actual[damage_name] <= 1,
                        "TARGET_DIRECT_REPAIR_DAMAGE_MISSING")
                if a[metric] is not None and b[metric] is not None:
                    require(abs(actual[credit_name] - actual[damage_name] - (a[metric] - b[metric])) <= 1e-10,
                            "TARGET_DIRECT_REPAIR_DAMAGE_IDENTITY_FAILED")
            indexed[key] = actual
            public.append(dict(candidate=candidate, reference=reference, partition=part, seed=seed,
                direct_repair=actual["repair"], direct_damage=actual["damage"], direct_Uno_repair=actual["Uno_repair"],
                direct_Uno_damage=actual["Uno_damage"], Harrell_delta=a["Harrell"] - b["Harrell"] if a["Harrell"] is not None and b["Harrell"] is not None else None,
                Uno60_delta=a["Uno60"] - b["Uno60"] if a["Uno60"] is not None and b["Uno60"] is not None else None,
                risk_SD_ratio=row.get("risk_SD_ratio"), pair_credits_from_actual_replay=True))
        except (ValueError, KeyError, TypeError) as error:
            code = str(error) if isinstance(error, ValueError) else type(error).__name__
            failures.append(code if code.replace("_", "").isalnum() else type(error).__name__)
    return indexed, public, dict(expected_rows=90, valid_rows=len(indexed), missing_rows=len(expected - set(indexed)),
                               validation_failures=failures, complete=len(indexed) == 90 and not failures)


def compare(candidate, reference, summaries, pairs):
    a, b = summaries[candidate], summaries[reference]
    out = dict(candidate=candidate, reference=reference, complete=False, Harrell_delta=None, Uno60_delta=None,
        direct_repair=None, direct_damage=None, direct_Uno_repair=None, direct_Uno_damage=None, per_seed=[])
    if a["complete_cells"] != 9 or b["complete_cells"] != 9:
        return out
    out.update(complete=True, Harrell_delta=a["Harrell"] - b["Harrell"] if finite(a["Harrell"]) and finite(b["Harrell"]) else None,
               Uno60_delta=a["Uno60"] - b["Uno60"] if finite(a["Uno60"]) and finite(b["Uno60"]) else None)
    for seed, ma, mb in zip(SEEDS, a["per_seed"], b["per_seed"]):
        keys = [(candidate, reference, p, seed) for p in PARTITIONS]
        credits = [pairs[k] for k in keys] if all(k in pairs for k in keys) else None
        record = dict(seed=seed, Harrell_delta=ma["Harrell"] - mb["Harrell"] if finite(ma["Harrell"]) and finite(mb["Harrell"]) else None,
            Uno60_delta=ma["Uno60"] - mb["Uno60"] if finite(ma["Uno60"]) and finite(mb["Uno60"]) else None)
        for key, denominator in (("repair", "Harrell_den"), ("damage", "Harrell_den"), ("Uno_repair", "Uno_den"), ("Uno_damage", "Uno_den")):
            record["direct_" + key] = sum(r[key] * r[denominator] for r in credits) / sum(r[denominator] for r in credits) if credits and sum(r[denominator] for r in credits) > 0 else None
        out["per_seed"].append(record)
    for name, credit_key in (("direct_repair", "repair"), ("direct_damage", "damage"),
                            ("direct_Uno_repair", "Uno_repair"), ("direct_Uno_damage", "Uno_damage")):
        out[name] = mean([r["direct_" + credit_key] for r in out["per_seed"]])
    out["direct_credit_complete"] = all(out[k] is not None for k in ("direct_repair", "direct_damage", "direct_Uno_repair", "direct_Uno_damage"))
    out["small_positive_delta_below_0_003"] = any(finite(out[k]) and 0 < out[k] < .003 for k in ("Harrell_delta", "Uno60_delta"))
    return out


def two_positive(comparison):
    if not comparison.get("complete") or not finite(comparison.get("Harrell_delta")) or not finite(comparison.get("Uno60_delta")):
        return None
    return comparison["Harrell_delta"] > 0 and comparison["Uno60_delta"] > 0


def one_positive_other_nonnegative(comparison):
    if not comparison.get("complete") or not finite(comparison.get("Harrell_delta")) or not finite(comparison.get("Uno60_delta")):
        return None
    a, b = comparison["Harrell_delta"], comparison["Uno60_delta"]
    return a >= 0 and b >= 0 and (a > 0 or b > 0)


def decide(config, source, qualification, replay, ledger, comparisons, paired_audit):
    primary = {ref: two_positive(comparisons[f"B7_minus_{ref}"]) for ref in config["outer_gate"]["B7_both_metrics_exceed"]}
    reverse = one_positive_other_nonnegative(comparisons["B7_minus_B11"])
    raw = one_positive_other_nonnegative(comparisons["B6_minus_B5"])
    prior_plus_raw = one_positive_other_nonnegative(comparisons["B7_minus_B5"])
    raw_flag = None if reverse is None or raw is None or prior_plus_raw is None else (raw or prior_plus_raw) and reverse
    flags = dict(ST_LOCAL_EVIDENCE_SUPPORTED=source["local_support"], TRUE_GENE_REREAD_WORKS=raw_flag,
        CORRECT_QUERY_BEATS_WRONG=two_positive(comparisons["B7_minus_B9"]),
        DSS_BEATS_CP_SCALAR=two_positive(comparisons["B7_minus_B1"]))
    complete = (qualification["passed"] and replay["complete"] and paired_audit["complete"]
        and source["complete_source_and_full_transfer"] and len(ledger) == 108
        and sum(r["status"] == "PASS_FIT_AND_ACTUAL_REPLAY" for r in ledger) == 90
        and sum(r["status"] == "PASS_ZERO_FIT_REFERENCE" for r in ledger) == 18)
    primary_known = all(value is not None for value in primary.values()) and reverse is not None and source["local_support"] is not None
    supported = complete and primary_known and source["local_support"] is True and all(primary.values()) and reverse
    trigger_values = [two_positive(comparisons[f"B7_minus_{ref}"]) for ref in ("B3", "B4", "B5", "B6")]
    trigger = None if any(v is None for v in trigger_values) else all(trigger_values)
    optional = {arm: "PENDING" if trigger is None or trigger else "NOT_ACTIVATED" for arm in ("B10", "B12", "B14")}
    optional["B15"] = "TEXT_BLOCKED"
    status = "INCONCLUSIVE" if not complete or not primary_known else "INNER_GATE_PASSED_OPTIONAL_PENDING" if supported else "NOT_SUPPORTED"
    return dict(decision=status, core_flags=flags, complete_source_and_target_evidence=bool(complete),
        preliminary_outer_criteria_met=bool(supported), outer_start_allowed=False,
        optional_activation_trigger=trigger, optional_followup=optional,
        actual_optional_followup_required_before_external_stage=bool(supported),
        formal_target_matrix_denominator=90, optional_not_counted_as_failure=True,
        strict_primary_comparisons=primary, reverse_reread_condition=reverse,
        raw_reread_attribution=dict(strict_same_source_no_prior_B6_vs_B5=raw,
            B7_vs_B5_also_changes_source_prior=prior_plus_raw, reverse_B7_vs_B11=reverse),
        wrong_ST_is_separate_from_wrong_patient_query=True,
        PFI_used_for_selection=False, CI_or_pvalues_computed=False, biological_or_causal_mechanism_proven=False,
        all_medical_ideas_rejected=False, source_or_numeric_failure_is_not_performance_negative=True,
        historical_outer_is_new_independent_validation=False)


def build_report(config_path, private_root, source_decision_path, source_report_path, *, target_lock=None):
    private = Path(private_root)
    lock_path = Path(target_lock) if target_lock is not None else private / "targetLOCK.json"
    config, lock = validate_inputs(config_path, lock_path)
    config_sha = sha256(config_path)
    source = read_source(source_decision_path, source_report_path, lock)
    qualification = read_qualification(private, lock)
    replay = read_replay(private, lock, qualification)
    ledger, metrics, pfi, values = [], {}, {}, {}
    for arm in BASELINES + ARMS:
        for partition in PARTITIONS:
            for seed in SEEDS:
                row, score, value = read_reference(private, arm, partition, seed, lock, replay) if arm in BASELINES else read_cell(
                    private, arm, partition, seed, lock, config_sha, qualification, replay)
                ledger.append(row)
                if row["status"] in ("PASS_ZERO_FIT_REFERENCE", "PASS_FIT_AND_ACTUAL_REPLAY"):
                    metrics[(arm, partition, seed)] = score
                    pfi[(arm, partition, seed)] = value["metrics_vs"]["PFI_frozen_DSS_risk"]
                    values[(arm, partition, seed)] = value
    # The two references reuse one risk vector per partition, not 18 new fits.
    for arm in BASELINES:
        for partition in PARTITIONS:
            group = [values[(arm, partition, s)]["metrics_vs"] for s in SEEDS if (arm, partition, s) in values]
            require(not group or all(v == group[0] for v in group), "TARGET_BASELINE_REUSED_SEED_RISKS_CHANGED")
    initialization_audit = []
    for partition in PARTITIONS:
        for seed in SEEDS:
            rows = [r for r in ledger if r["arm"] in ARMS and r["partition"] == partition and r["seed"] == seed and r["status"] == "PASS_FIT_AND_ACTUAL_REPLAY"]
            if len(rows) == 10:
                hashes = {r["target_initialization_sha256"] for r in rows}
                require(len(hashes) == 1 and None not in hashes, "TARGET_MATCHED_INITIALIZATION_CHANGED")
                if partition == 0 and seed == 31 and qualification["passed"]:
                    require(next(iter(hashes)) == qualification["shared_target_initialization_sha256"], "TARGET_REAL_QUALIFICATION_INITIALIZATION_CHANGED")
            initialization_audit.append(dict(partition=partition, seed=seed, validated_arms=len(rows),
                complete_same_initialization=len(rows) == 10))
    summaries = {arm: aggregate_arm(arm, metrics) for arm in BASELINES + ARMS}
    pfi_summaries = {arm: aggregate_arm(arm, pfi) for arm in BASELINES + ARMS}
    paired, pair_rows, paired_audit = direct_pairs(replay, metrics)
    comparisons = {f"B7_minus_{ref}": compare("B7", ref, summaries, paired) for ref in PAIR_REFERENCES}
    comparisons["B6_minus_B5"] = compare("B6", "B5", summaries, paired)
    decision = decide(config, source, qualification, replay, ledger, comparisons, paired_audit)
    counts = {status: sum(r["status"] == status for r in ledger) for status in sorted({r["status"] for r in ledger})}
    report = dict(schema="bgb_en_r6_target_inner_report_v1", config_sha256=config_sha,
        target_lock_sha256=sha256(lock_path), contract_sha256=lock["contract_sha256"], report_code_sha256=sha256(__file__),
        target_sources_sha256=lock["sources"], source=source, qualification=qualification,
        actual_replay=dict(status=replay["status"], complete=replay["complete"], sha256=replay["sha256"]),
        planned_new_target_fits=90, planned_baseline_reference_rows=18, planned_target_updates=7200,
        confirmed_new_target_fits=sum(r["actual_new_optimizer_fits"] for r in ledger if r["actual_new_optimizer_fits"] is not None),
        confirmed_target_updates=sum(r["actual_optimizer_updates"] for r in ledger if r["actual_optimizer_updates"] is not None),
        unknown_cost_units=sum(r["actual_optimizer_updates"] is None for r in ledger), ledger=ledger, counts=counts,
        primary_DSS=list(summaries.values()), secondary_PFI_frozen_DSS_risk=list(pfi_summaries.values()),
        paired_comparisons=comparisons, direct_pair_audit=paired_audit, initialization_audit=initialization_audit,
        decision=decision, outer_results=dict(status="NOT_RUN", Harrell=None, Uno60=None,
            reason="CURRENT_REPORT_IS_INNER_ONLY_REQUIRES_ACTUAL_OPTIONAL_FOLLOWUP_AND_NEW_OUTER_AUTHORIZATION"),
        no_cross_partition_risk_ranking=True, seed_means_are_not_independent_cohorts=True,
        no_model_or_array_deserialization=True, no_raw_patient_rows_read=True,
        repair_damage_never_invented_from_Cindex_differences=True, no_epoch_selection=True)
    fold_rows = []
    for endpoint, mapping in (("DSS", metrics), ("PFI_frozen_DSS_risk", pfi)):
        fold_rows.extend(dict(endpoint=endpoint, arm=arm, partition=partition, seed=seed,
            **{key: score.get(key) for key in ("Harrell", "Uno60") + COMPONENTS}) for (arm, partition, seed), score in mapping.items())
    return report, fold_rows, pair_rows


def write_report(report, fold_rows, pair_rows, output):
    output = Path(output).resolve()
    require(not any(output == p.resolve() or output.is_relative_to(p.resolve()) for p in (ROOT / "reports/bgb_en", ROOT / "reports/bgb_en_r5")), "HISTORICAL_NAMESPACES_PRESERVED")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "R6_4_INNER_MATRIX.json", report)
    write_json(output / "decision.json", report["decision"])
    fields = ["arm", "status", "expected_cells", "complete_cells", "Harrell", "Uno60"]
    write_csv(output / "R6_4_INNER_MATRIX.csv", report["primary_DSS"], fields)
    write_csv(output / "R6_SEED_FOLD.csv", fold_rows, ["endpoint", "arm", "partition", "seed", "Harrell", "Uno60", *COMPONENTS])
    write_csv(output / "R6_MECHANISM_ABLATIONS.csv", pair_rows, ["candidate", "reference", "partition", "seed", "Harrell_delta", "Uno60_delta",
        "direct_repair", "direct_damage", "direct_Uno_repair", "direct_Uno_damage", "risk_SD_ratio", "pair_credits_from_actual_replay"])
    write_csv(output / "R6_COSTS.csv", report["ledger"], ["group", "arm", "partition", "seed", "status", "expected_updates",
        "actual_optimizer_updates", "actual_new_optimizer_fits", "elapsed_seconds", "peak_GPU_bytes", "actual_replay_passed"])
    write_csv(output / "R6_5_OUTER_RESULTS.csv", [report["outer_results"]], ["status", "Harrell", "Uno60", "reason"])
    audit_path = output / "SOURCE_AUDIT.json"
    audit = read(audit_path) if audit_path.exists() else {}
    audit["r6_target_survival"] = {key: report[key] for key in ("config_sha256", "target_lock_sha256", "contract_sha256",
        "report_code_sha256", "target_sources_sha256", "no_model_or_array_deserialization", "no_raw_patient_rows_read")}
    audit["r6_target_survival"].update(source_decision_sha256=report["source"]["decision_sha256"],
        source_report_sha256=report["source"]["report_sha256"], qualification_sha256=report["qualification"]["sha256"], replay_sha256=report["actual_replay"]["sha256"])
    write_json(audit_path, audit)
    def fmt(value):
        return "—" if value is None else f"{value:.6f}"
    lines = ["# R6 内层原型矩阵", "", f"决策状态：**{report['decision']['decision']}**。", "",
        f"预锁90次目标拟合及18行零拟合参考；确认新拟合{report['confirmed_new_target_fits']}、更新{report['confirmed_target_updates']}，成本未知单元{report['unknown_cost_units']}。", "",
        "每seed先汇总三个分区内的concordance分子/分母，再等权三个seed；没有跨分区风险排序、置信区间或p值。", "",
        "| 模型 | 完成/9 | DSS Harrell | DSS Uno60 |", "|---|---:|---:|---:|"]
    for row in report["primary_DSS"]:
        lines.append(f"| {row['arm']} | {row['complete_cells']} | {fmt(row['Harrell'])} | {fmt(row['Uno60'])} |")
    lines += ["", "## 成对机制比较", "", "修复/损伤仅来自独立重放的实际配对风险信用；缺失保持空值。B7−B8检验错误ST来源，B7−B9检验错误患者查询，两者分别解释。", "",
        "| 比较 | ΔHarrell | ΔUno60 | 修复 | 损伤 |", "|---|---:|---:|---:|---:|"]
    for name, comparison in report["paired_comparisons"].items():
        lines.append(f"| {name} | {fmt(comparison['Harrell_delta'])} | {fmt(comparison['Uno60_delta'])} | {fmt(comparison['direct_repair'])} | {fmt(comparison['direct_damage'])} |")
    lines += ["", "## 解释边界", "", "B6−B5是同源、无ST先验的原始基因与token重读对照；B7−B5还改变了先验，不能把差值全部归因于原始基因。小于0.003的正向差仍属弱点估计证据。", "",
        "20/40/80轮预算诊断仅描述固定80轮训练；不选epoch。PFI使用冻结DSS风险，独立列在JSON和逐折表，不参与选择。", "",
        "优势：保留强锚点、逐单元谱系与实际重放；风险：小事件数与内层重复使用限制独立验证强度，当前结果不能证明细胞来源或因果机制。", ""]
    (output / "R6_4_INNER_MATRIX.md").write_text("\n".join(lines), encoding="utf-8")
    decision = report["decision"]
    text = ["# R6 决策", "", f"**{decision['decision']}**；外层状态：**NOT_RUN**。", "",
            "| 核心问题 | 方向性状态 |", "|---|---|"]
    for key, value in decision["core_flags"].items():
        text.append(f"| {key} | {'SUPPORTED' if value is True else 'NOT_SUPPORTED' if value is False else 'INCONCLUSIVE'} |")
    text += ["", "可选真实重训：" + "；".join(f"{arm}={status}" for arm, status in decision["optional_followup"].items()) + "。", "",
        "满足内层方向门槛后仍须按新锁完成必要可选对照，再由独立外层协议启动。未完成和数值重放失败都不作为性能负结果；负结果也不等于否定全部医学研究思路。", "",
        "## 拓展思考", "", "区分源对应证据、原始基因重读、患者查询与强锚点条件增量四个层次。", "",
        "## 你可能想知道", "", "三个seed来自同一cohort，不是三个独立队列；旧外折结果不是本阶段新独立验证。", ""]
    (output / "R6_DECISION.md").write_text("\n".join(text), encoding="utf-8")
    (output / "R6_5_OUTER_RESULTS.md").write_text("# R6 外层结果\n\n**NOT_RUN**。本报告仅读取内层原型元数据；没有新增外层指标。需完成可选对照并另行锁定与执行外层协议。\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bgb_en/r6_inner_v1.json")
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path)
    parser.add_argument("--source-decision", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bgb_en_r6")
    args = parser.parse_args()
    source_report = args.source_report or args.source_decision.parent / "R6_1_SOURCE_SUPERVISION.json"
    report, folds, pairs = build_report(args.config, args.private_root, args.source_decision, source_report, target_lock=args.target_lock)
    write_report(report, folds, pairs, args.output)
    print(json.dumps(dict(decision=report["decision"]["decision"], counts=report["counts"], outer="NOT_RUN")))


if __name__ == "__main__":
    main()

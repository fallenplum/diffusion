"""R7 metadata-only qualification export and fail-closed completion accounting.

No scientific module, checkpoint, array or endpoint is loaded. Existing actual
replay evidence is SHA-bound, never relabelled as a new model replay. The raw
anonymous replay is copied byte-for-byte only after a strict public preflight.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ("ADD", "RAW", "NORM", "INDEP", "SLOT", "GATE1", "GATE2", "SLOT_WRONG_RNA",
    "GATE1_WRONG_RNA", "GATE1_WRONG_PATH", "GATE2_WRONG_RNA")
PARTITIONS, SEEDS, EPOCHS = (0, 1, 2), (31, 47, 83), (20, 40, 80)
SCALE_STATS = {"context_rms": ("raw", "raw_G0"), "patch_rms": ("raw", "raw_adapted_patch"),
    "query_rms": ("raw", "raw_base_query"), "gene_key_rms": ("raw", "raw_gene_key"),
    "evidence_rms": ("base", "scaled_P0"), "reverse_query_rms": ("reverse", "projected_reverse_query")}
MAX_PUBLIC_BYTES = 2 * 1024**2


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def count(value):
    return type(value) is int and value >= 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path, evidence=None, label=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.suffix in (".json", ".jsonl"), "REQUIRED_JSON_METADATA_MISSING_OR_UNSAFE")
    before = path.stat(); data = path.read_bytes(); after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "METADATA_CHANGED_DURING_READ")
    def invalid(_):
        raise ValueError("NONFINITE_JSON_METADATA")
    value = json.loads(data.decode("utf-8-sig"), parse_constant=invalid)
    require(isinstance(value, dict), "JSON_METADATA_OBJECT_REQUIRED")
    if evidence is not None:
        label = label or path.name; actual = hashlib.sha256(data).hexdigest()
        require(label not in evidence or evidence[label] == actual, "EVIDENCE_CHANGED_BETWEEN_READS")
        evidence[label] = actual
    return value


def exact(rows, fields, expected, code):
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), code)
    keys = [tuple(r.get(k) for k in fields) for r in rows]
    require(len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == set(expected), code)
    return dict(zip(keys, rows))


def zero(value, keys, code):
    require(all(type(value.get(k)) is int and value[k] == 0 for k in keys), code)


def units(arms=ARMS):
    return {(a, p, s) for a in arms for p in PARTITIONS for s in SEEDS}


def name(key):
    return f"{key[0]}_p{key[1]}_seed{key[2]}"


def controls(arm):
    return {"matched_query", "wrong_query", "mean_query", "interaction_off", "main_off"} | (
        set() if arm in ("ADD", "RAW") else {"wrong_path"})


def locked_inputs(private, evidence):
    lock = read(private / "targetLOCK.json", evidence, "target_lock")
    require(lock.get("schema") == "bgb_en_r7_target_protocol_lock_v1"
        and lock.get("status") == "LOCKED_BEFORE_R7_FITS" and digest(lock.get("contract_sha256"))
        and canonical({k: v for k, v in lock.items() if k != "contract_sha256"}) == lock["contract_sha256"], "R7_IMMUTABLE_LOCK_CHANGED")
    require(lock.get("planned_neural_fits") == 99 and lock.get("planned_target_optimizer_updates") == 7920
        and lock.get("outer_test_access") is False, "R7_LOCKED_FORMAL_MATRIX_CHANGED")
    zero(lock, ("teacher_refits", "source_refits"), "R7_LOCK_HAS_UPSTREAM_REFITS")
    sources = lock.get("sources", {})
    require(isinstance(sources, dict) and len(sources) == 40 and all(digest(v) for v in sources.values()), "R7_EXACT_40_SCIENCE_SOURCES_REQUIRED")
    for relative, expected in sources.items():
        parsed = PurePosixPath(relative)
        require(isinstance(relative, str) and not parsed.is_absolute() and ".." not in parsed.parts
            and "\\" not in relative and parsed.parts[0] in ("src", "scripts", "configs", "reports"), "R7_LOCKED_SOURCE_PATH_UNSAFE")
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file() and not path.is_symlink()
            and path.suffix in (".py", ".json", ".md", ".yaml", ".yml") and sha(path) == expected, "R7_LOCKED_SOURCE_BYTES_CHANGED")
    require(digest(lock.get("config_sha256")) and digest(lock.get("source_config_sha256")), "R7_CONFIG_HASH_MISSING")
    recipe = "configs/bgb_en/r7_inner_v1.json"
    require(sources.get(recipe) == lock["config_sha256"], "R7_OWN_RECIPE_SOURCE_HASH_CHANGED")
    config = read(ROOT / recipe)
    require(config.get("schema") == "bgb_en_r7_inner_v1" and tuple(config.get("trainable_arms", [])) == ARMS
        and config.get("formal_target_neural_fits") == 99, "R7_OWN_RECIPE_MATRIX_CHANGED")
    require(set(lock.get("prepared_receipt_sha256", {})) == {str(p) for p in PARTITIONS}
        and all(digest(v) for v in lock["prepared_receipt_sha256"].values()), "R7_PREPARED_RECEIPT_BINDINGS_MISSING")
    q = read(private / "QUALIFICATION.json", evidence, "qualification")
    require(q.get("schema") == "bgb_en_r7_real_data_qualification_v1"
        and q.get("status") == "PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER"
        and q.get("contract_sha256") == lock["contract_sha256"] and q.get("qualification_weights_saved") is False
        and q.get("outer_test_access") is False, "R7_REAL_QUALIFICATION_INVALID")
    zero(q, ("source_refits", "teacher_refits", "scalar_beta_refits", "optimizer_instances", "optimizer_updates"), "R7_QUALIFICATION_HAS_OPTIMIZER")
    records = exact(q.get("records"), ("arm",), {(a,) for a in ARMS}, "R7_QUALIFICATION_EXACT_11_ARMS_REQUIRED")
    forwards = 2  # Explicit RAW-versus-R6 equality pair, outside checks().
    for (arm,), row in records.items():
        expected = len(controls(arm)) + 4
        require(row.get("explicit_backward_passes") == 1 and row.get("optimizer_updates") == 0
            and row.get("complete_model_forward_calls") == expected
            and row.get("repeated_forward_max_error") == 0, "R7_QUALIFICATION_COMPUTE_COUNTS_CHANGED")
        forwards += expected
    require(q.get("explicit_backward_passes") == 11 and q.get("complete_model_forward_calls") == forwards == 110
        and q.get("distinct_train_RMS_calibrations") == 9 and q.get("scalar_scales_per_calibration") == 6
        and q.get("calibration_fits_are_unsupervised_not_neural_fits") is True, "R7_QUALIFICATION_TOTAL_COUNTS_CHANGED")
    return lock, config, q


def calibration_audit(private, evidence=None):
    private = Path(private); evidence = {} if evidence is None else evidence
    lock, config, q = locked_inputs(private, evidence)
    expected = {(p, s) for p in PARTITIONS for s in SEEDS}
    qcal = exact(q.get("calibrations"), ("partition", "seed"), expected, "R7_NINE_QUALIFIED_CALIBRATIONS_REQUIRED")
    require(set(lock.get("calibration_receipt_sha256", {})) == {f"p{p}_seed{s}" for p, s in expected}, "R7_CALIBRATION_LOCK_INVENTORY_CHANGED")
    rows, scopes = [], {}
    for part, seed in sorted(expected):
        label = f"calibration_p{part}_seed{seed}"
        cache = read(private / "calibration" / f"p{part}_seed{seed}.json", evidence, label)
        origin, cal = cache.get("origin", {}), cache.get("calibration", {})
        require(evidence[label] == lock["calibration_receipt_sha256"][f"p{part}_seed{seed}"]
            and cache == qcal[(part, seed)].get("calibration") and cache.get("status") == "PASS_R7_TRAIN_RMS_CALIBRATION",
            "R7_CALIBRATION_CACHE_QUALIFICATION_OR_LOCK_SHA_CHANGED")
        zero(cache, ("new_optimizer_updates",), "R7_CALIBRATION_OPTIMIZER_COST_CHANGED")
        require((origin.get("partition"), origin.get("seed")) == (part, seed)
            and origin.get("calibration_model") == "untrained_NORM" and origin.get("validation_used") is False
            and origin.get("outcomes_used") is False and origin.get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][str(part)]
            and origin.get("core_source_sha256") == lock["sources"].get("src/bgb_en/r7_survival.py")
            and origin.get("common_source_sha256") == lock["sources"].get("scripts/bgb_en_r7_common.py")
            and origin.get("config_sha256") == lock["config_sha256"]
            and origin.get("source_state_sha256") == lock["source_full_states"]["MATCHED"]["complete_source_state_sha256"]
            and all(digest(origin.get(k)) for k in ("train_hash", "patch_cache_receipt_sha256", "initial_shared_state_sha256")),
            "R7_CALIBRATION_SOURCE_OR_TRAIN_SCOPE_CHANGED")
        require(cal.get("train_hash") == origin["train_hash"] and type(cal.get("training_patients")) is int
            and 8 <= cal["training_patients"] <= 512 and cal.get("train_only") is True
            and cal.get("outcomes_accessed") is False and cal.get("fixed_after_initialization") is True,
            "R7_CALIBRATION_NOT_FIXED_FULL_CURRENT_TRAIN")
        zero(cal, ("new_optimizer_updates",), "R7_CALIBRATION_INTERNAL_OPTIMIZER_COST_CHANGED")
        scope = (origin["train_hash"], cal["training_patients"], origin["prepared_receipt_sha256"])
        require(part not in scopes or scopes[part] == scope, "R7_SEED_CALIBRATIONS_CHANGED_TRAIN_PATIENT_SCOPE")
        scopes[part] = scope
        scales, statistics_ = cal.get("scales", {}), cal.get("statistics", {})
        require(set(scales) == set(SCALE_STATS) and set(statistics_) == {"raw", "base", "reverse"}, "R7_SIX_STAGED_RMS_INTERFACES_REQUIRED")
        require(set(statistics_["raw"]) == {"raw_G0", "raw_adapted_patch", "raw_base_query", "raw_gene_key", "raw_gene_value"}
            and set(statistics_["base"]) == {"scaled_P0"} and set(statistics_["reverse"]) == {"projected_reverse_query"}, "R7_CALIBRATION_STATISTIC_INVENTORY_CHANGED")
        for stage in statistics_.values():
            for statistic in stage.values():
                require(set(statistic) == {"sum_squares", "count"} and finite(statistic["sum_squares"])
                    and statistic["sum_squares"] >= 0 and type(statistic["count"]) is int and statistic["count"] > 0,
                    "R7_CALIBRATION_STATISTIC_NONFINITE_OR_EMPTY")
        counts = {}
        for key, (stage, statistic) in SCALE_STATS.items():
            value, measured = scales[key], statistics_[stage][statistic]
            expected_scale = max(math.sqrt(measured["sum_squares"] / measured["count"]), 1e-6)
            require(finite(value) and value >= 1e-6 and math.isclose(value, expected_scale, rel_tol=2e-12, abs_tol=1e-12),
                "R7_CALIBRATION_RMS_FORMULA_OR_MINIMUM_CHANGED")
            counts[key] = measured["count"]
        rows.append(dict(partition=part, seed=seed, scales=scales, statistic_counts=counts,
            training_patients=cal["training_patients"], calibration_cache_sha256=evidence[label]))
    result = dict(schema="bgb_en_r7_scale_calibration_audit_v1", status="PASS_NINE_FULL_TRAIN_CALIBRATIONS_54_SCALARS",
        contract_sha256=lock["contract_sha256"], target_lock_sha256=evidence["target_lock"],
        qualification_sha256=evidence["qualification"], delivery_code_sha256=sha(__file__),
        calibration_sets=9, scalar_scale_count=54, scales_per_set=6, minimum_RMS=1e-6, rows=rows,
        verification=dict(shared_untrained_NORM_origin=True, all_40_locked_science_bytes_verified=True,
            full_current_train_receipt_and_train_hash_scope_verified=True, raw_training_indices_read=False,
            independent_train_validation_disjointness_recomputed=False, validation_used=False, outcomes_used=False,
            scales_recomputed_from_full_train_sum_squares_and_counts=True, gene_value_statistic_is_not_a_seventh_scale=True),
        new_calibration_computations=0, optimizer_instances=0, optimizer_updates=0, model_forwards=0, backwards=0,
        raw_measurements_or_origins_or_indices_exported=False)
    return result, lock, config, q, evidence


def public_preflight(data, *, denylist=None):
    """Generic raw-key/pattern check; actual registry scan only when supplied."""
    from scripts import bgb_en_publication as inherited
    require(len(data) <= MAX_PUBLIC_BYTES, "PUBLIC_JSON_EXCEEDS_2MIB")
    value = json.loads(data.decode("utf-8-sig"))
    inherited.validate_public_json(value)
    text = data.decode("utf-8-sig")
    require(not any(rule.search(text) for rule in inherited.PATTERNS.values()), "PUBLIC_PAYLOAD_PRIVATE_PATTERN_DETECTED")
    full_scan = False
    if denylist is not None:
        registry = inherited.validate_denylist(read(denylist))
        require(not inherited.identifier_pattern(registry).search(text), "PUBLIC_PAYLOAD_ACTUAL_PRIVATE_IDENTIFIER_DETECTED")
        full_scan = True
    return dict(schema_and_generic_pattern_preflight_passed=True, actual_private_identifier_denylist_scanned=full_scan,
        final_publication_registry_scan_still_required=not full_scan)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def exclusive_write(path, data):
    require(not path.exists(), "EXISTING_PUBLIC_DELIVERY_PRESERVED")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def export_qualification(private, reports, *, denylist=None):
    audit, _, _, _, _ = calibration_audit(private)
    audit["public_preflight"] = public_preflight(json_bytes(audit), denylist=denylist)
    data = json_bytes(audit); public_preflight(data, denylist=denylist)
    exclusive_write(Path(reports) / "R7_SCALE_CALIBRATION_AUDIT.json", data)
    return audit


def scalar_map(value, allowed, code, *, nonnegative=False):
    require(isinstance(value, dict) and set(value) <= set(allowed), code)
    require(all(v is None or finite(v) and (not nonnegative or v >= 0) for v in value.values()), code)


def scalar_leaves(value, complex_keys, code):
    """Known containers only; an arbitrary short array is still private data."""
    require(all(k in complex_keys or v is None or type(v) in (str, bool, int, float)
        and (not isinstance(v, float) or math.isfinite(v)) for k, v in value.items()), code)


def replay_public_structure(replay):
    """Explicit producer structure: dictionaries of named scalars, not arrays."""
    root_keys = {"status", "contract_sha256", "qualification_sha256", "upstream_replay_audit_sha256", "expected_units",
        "available_units", "passed_units", "failed_units", "expected_checkpoints", "passed_checkpoints", "missing_or_unreplayed_units",
        "units", "new_actual_replays", "reused_terminal_receipts", "upstream_reference_replay", "zero_fit_baseline_reference_replays",
        "paired_retrained_controls", "new_optimizer_fits", "new_preprocessing_fits", "primary_checkpoint_epoch",
        "diagnostic_checkpoints_not_selected", "patient_arrays_exported"}
    require(set(replay) == root_keys and replay.get("missing_or_unreplayed_units") == [], "REPLAY_PUBLIC_ROOT_STRUCTURE_CHANGED")
    scalar_leaves(replay, {"missing_or_unreplayed_units", "units", "upstream_reference_replay",
        "zero_fit_baseline_reference_replays", "paired_retrained_controls"}, "REPLAY_PUBLIC_ROOT_NONSCALAR_LEAF")
    unit_keys = {"unit", "arm", "partition", "seed", "contract_sha256", "completion_sha256", "artifact_sha256", "qualification_sha256",
        "upstream_replay_audit_sha256", "replay_script_sha256", "status", "checkpoints", "actual_checkpoint_count", "maximum_risk_error",
        "changed_pair_credits", "all_checkpoints_actual_forward_replayed", "actual_state_dict_forward_replayed",
        "source_frozen_weight_hash_replayed", "saved_query_mapping_recomputed", "new_optimizer_updates"}
    checkpoint_keys = {"epoch", "status", "checkpoint_sha256", "maximum_risk_error", "risk_errors_by_split", "changed_pair_credits",
        "frozen_control_risk_errors", "maximum_metric_error", "mean_query_error", "actual_state_dict_forward_replayed", "new_optimizer_updates"}
    for row in replay.get("units", []):
        require(set(row) == unit_keys, "REPLAY_PUBLIC_UNIT_STRUCTURE_CHANGED")
        scalar_leaves(row, {"checkpoints"}, "REPLAY_PUBLIC_UNIT_NONSCALAR_LEAF")
        for cp in row.get("checkpoints", []):
            require(set(cp) == checkpoint_keys and set(cp["risk_errors_by_split"]) == {"train", "validation"}, "REPLAY_PUBLIC_CHECKPOINT_STRUCTURE_CHANGED")
            scalar_leaves(cp, {"risk_errors_by_split", "frozen_control_risk_errors"}, "REPLAY_PUBLIC_CHECKPOINT_NONSCALAR_LEAF")
            require(all(finite(cp.get(k)) and cp[k] >= 0 for k in ("maximum_risk_error", "maximum_metric_error", "mean_query_error")),
                "REPLAY_PUBLIC_CHECKPOINT_ERRORS_NOT_NUMERIC_SCALARS")
            scalar_map(cp["risk_errors_by_split"], ("train", "validation"), "REPLAY_PUBLIC_SPLIT_ERRORS_NOT_SCALARS", nonnegative=True)
            require(set(cp["frozen_control_risk_errors"]) == controls(row["arm"]), "REPLAY_PUBLIC_CONTROL_INVENTORY_CHANGED")
            scalar_map(cp["frozen_control_risk_errors"], controls(row["arm"]), "REPLAY_PUBLIC_CONTROL_ERRORS_NOT_SCALARS", nonnegative=True)
    ref_keys = {"arm", "partition", "seed", "status", "receipt_sha256", "independent_fixed_anchor_metrics_recomputed", "new_optimizer_fits"}
    require(all(set(r) == ref_keys for r in replay.get("zero_fit_baseline_reference_replays", [])), "REPLAY_PUBLIC_REFERENCE_STRUCTURE_CHANGED")
    for row in replay.get("zero_fit_baseline_reference_replays", []):
        scalar_leaves(row, set(), "REPLAY_PUBLIC_REFERENCE_NONSCALAR_LEAF")
    up = replay.get("upstream_reference_replay", {})
    require(set(up) == {"status", "rows", "teacher_refits", "source_refits", "new_optimizer_fits", "inherited_R6_contract_sha256"}, "REPLAY_PUBLIC_UPSTREAM_STRUCTURE_CHANGED")
    scalar_leaves(up, {"rows"}, "REPLAY_PUBLIC_UPSTREAM_NONSCALAR_LEAF")
    up_keys = {"partition", "status", "coefficient_models", "beta_gradient_abs", "maximum_risk_error", "new_optimizer_fits", "new_preprocessing_fits",
        "teacher_heldout_labels_excluded_from_teacher_fits", "validation_labels_excluded"}
    require(all(set(r) == up_keys for r in up.get("rows", [])), "REPLAY_PUBLIC_UPSTREAM_ROW_STRUCTURE_CHANGED")
    for row in up.get("rows", []):
        scalar_leaves(row, set(), "REPLAY_PUBLIC_UPSTREAM_ROW_NONSCALAR_LEAF")
    pair_keys = {"candidate", "reference", "partition", "seed", "candidate_vs_reference", "PFI_frozen_DSS_risk", "risk_SD_ratio",
        "direct_risk_pair_credits", "bootstrap_or_CI_computed", "primary_checkpoint_epoch"}
    metric_keys = {"Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den", "repair", "damage", "Uno_repair", "Uno_damage", "risk_shift_sd_ratio"}
    for row in replay.get("paired_retrained_controls", []):
        require(set(row) == pair_keys and row.get("candidate") in ARMS and row.get("reference") in (*ARMS, "B0", "B1"), "REPLAY_PUBLIC_PAIR_STRUCTURE_CHANGED")
        scalar_leaves(row, {"candidate_vs_reference", "PFI_frozen_DSS_risk"}, "REPLAY_PUBLIC_PAIR_NONSCALAR_LEAF")
        for key in ("candidate_vs_reference", "PFI_frozen_DSS_risk"):
            scalar_map(row.get(key), metric_keys, "REPLAY_PUBLIC_METRICS_NOT_NAMED_SCALARS", nonnegative=True)


def complete_account(private, reports, *, denylist=None):
    private, reports = Path(private), Path(reports)
    audit, lock, config, q, evidence = calibration_audit(private)
    result = read(reports / "R7_RESULTS.json", evidence, "results")
    require(result.get("schema") == "bgb_en_r7_complete_report_v1" and result.get("status") == "COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS"
        and result.get("contract_sha256") == lock["contract_sha256"] and result.get("config_sha256") == lock["config_sha256"]
        and result.get("qualification_sha256") == evidence["qualification"], "R7_RESULTS_NOT_COMPLETE_OR_LINEAGE_CHANGED")
    expected_budget = dict(formal_neural_fits=99, formal_optimizer_updates=7920, source_refits=0, teacher_or_beta_refits=0,
        unsupervised_train_RMS_calibrations=9, zero_fit_baseline_reference_rows=18, diagnostics_optimizer_updates=0)
    require(result.get("budget") == expected_budget, "R7_RESULTS_BUDGET_CHANGED")
    replay_path = private / "REPLAY_AUDIT.json"
    replay = read(replay_path, evidence, "actual_replay")
    require(replay.get("status") == "PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY" and replay.get("contract_sha256") == lock["contract_sha256"]
        and replay.get("qualification_sha256") == evidence["qualification"] and result.get("replay_sha256") == evidence["actual_replay"]
        and all(replay.get(k) == 99 for k in ("expected_units", "available_units", "passed_units"))
        and replay.get("expected_checkpoints") == replay.get("passed_checkpoints") == 297
        and replay.get("missing_or_unreplayed_units") == [] and replay.get("patient_arrays_exported") is False, "R7_ALL_99_UNITS_297_STATES_NOT_ACTUALLY_REPLAYED")
    zero(replay, ("failed_units", "new_optimizer_fits", "new_preprocessing_fits"), "R7_REPLAY_FAILED_OR_REFIT")
    rrows = exact(replay.get("units"), ("arm", "partition", "seed"), units(), "R7_EXACT_99_ACTUAL_REPLAY_UNITS_REQUIRED")
    found = {p.parent.name for p in (private / "cells").glob("*/COMPLETE.json")}
    require(found == {name(k) for k in units()}, "R7_EXACT_99_COMPLETION_RECEIPTS_REQUIRED")
    upstream = read(private / "UPSTREAM_REPLAY_AUDIT.json", evidence, "upstream_replay")
    require(upstream == replay.get("upstream_reference_replay") and q.get("upstream_replay_audit_sha256")
        == replay.get("upstream_replay_audit_sha256") == evidence["upstream_replay"], "R7_UPSTREAM_REPLAY_SHA_CHANGED")
    check_upstream(upstream, lock)
    rows, diagnostic_counts = [], dict(saved_diagnostic_receipts=0, recorded_gradient_probe_complete_model_forward_calls=0,
        recorded_query_probe_complete_model_forward_calls=0, scheduled_six_gradient_path_slots=0,
        recorded_independent_parameter_L2_gradient_calls=0, complete_riskset_leaf_gradient_calls_code_derived=0)
    for key, rp in sorted(rrows.items()):
        arm, part, seed = key; unit = name(key); directory = private / "cells" / unit
        require(not (directory / "ERROR_RECEIPT.json").exists(), "R7_FAILED_AND_COMPLETE_CONFLICT")
        done = read(directory / "COMPLETE.json", evidence, unit + "_complete")
        c = done.get("contract", {})
        require(done.get("status") == "COMPLETE_R7_TARGET_FIT" and tuple(done.get(k) for k in ("arm", "partition", "seed")) == key
            and c.get("contract_sha256") == lock["contract_sha256"] and tuple(c.get(k) for k in ("arm", "partition", "seed")) == key
            and c.get("qualification_sha256") == evidence["qualification"] and c.get("config_sha256") == lock["config_sha256"]
            and c.get("source_config_sha256") == lock["source_config_sha256"] and c.get("upstream_r6") == lock["upstream_r6"]
            and c.get("fixed_deterministic_runtime") == lock["fixed_deterministic_runtime"]
            and c.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][str(part)]
            and done.get("epochs") == done.get("optimizer_updates") == c.get("epochs") == 80
            and done.get("actual_new_optimizer_fits") == 1 and done.get("outer_test_used") is c.get("outer_test_used") is False
            and done.get("primary_epoch_locked_before_training") == 80, "R7_COMPLETION_CONTRACT_OR_BUDGET_CHANGED")
        for value in (c, done):
            zero(value, ("teacher_refits", "source_training_updates"), "R7_COMPLETION_HAS_UPSTREAM_REFITS")
        origin = done.get("source_origin", {})
        require(origin.get("source_key") == "MATCHED" and origin.get("source_origin") == lock["source_full_states"]["MATCHED"]
            and origin.get("calibration_receipt_sha256") == lock["calibration_receipt_sha256"][f"p{part}_seed{seed}"]
            and digest(done.get("artifact_sha256")), "R7_COMPLETION_SOURCE_OR_CALIBRATION_CHANGED")
        terminal = read(private / "REPLAY_PRIVATE" / (unit + ".PASS.json"), evidence, unit + "_actual_replay")
        require(not (private / "REPLAY_PRIVATE" / (unit + ".FAIL.json")).exists() and terminal == rp
            and rp.get("unit") == unit and rp.get("status") == "PASS" and rp.get("contract_sha256") == lock["contract_sha256"]
            and rp.get("completion_sha256") == evidence[unit + "_complete"] and rp.get("artifact_sha256") == done["artifact_sha256"]
            and rp.get("qualification_sha256") == evidence["qualification"] and rp.get("upstream_replay_audit_sha256") == evidence["upstream_replay"]
            and rp.get("replay_script_sha256") == lock["sources"]["scripts/bgb_en_r7_replay.py"]
            and rp.get("actual_checkpoint_count") == 3 and rp.get("changed_pair_credits") == 0
            and all(rp.get(k) is True for k in ("all_checkpoints_actual_forward_replayed", "actual_state_dict_forward_replayed",
                "source_frozen_weight_hash_replayed", "saved_query_mapping_recomputed"))
            and rp.get("new_optimizer_updates") == 0 and finite(rp.get("maximum_risk_error"))
            and 0 <= rp["maximum_risk_error"] <= 2e-6, "R7_ACTUAL_REPLAY_TERMINAL_OR_NUMERIC_FAILED")
        cps = rp.get("checkpoints", [])
        require([v.get("epoch") for v in cps] == list(EPOCHS) and set(done.get("checkpoint_artifact_sha256", {})) == {str(e) for e in EPOCHS}, "R7_THREE_SAVED_EPOCHS_REQUIRED")
        for cp in cps:
            require(cp.get("status") == "PASS" and cp.get("actual_state_dict_forward_replayed") is True
                and cp.get("changed_pair_credits") == cp.get("new_optimizer_updates") == 0
                and cp.get("checkpoint_sha256") == done["checkpoint_artifact_sha256"][str(cp["epoch"])]
                and digest(cp.get("checkpoint_sha256")) and finite(cp.get("maximum_risk_error")) and 0 <= cp["maximum_risk_error"] <= 2e-6,
                "R7_CHECKPOINT_REPLAY_SHA_OR_NUMERIC_FAILED")
        require(set(done.get("checkpoint_diagnostics", {})) == {"20", "40", "60", "80"}, "R7_FOUR_DIAGNOSTIC_EPOCHS_REQUIRED")
        for epoch in (20, 40, 60, 80):
            diag = read(directory / f"DIAGNOSTIC_epoch{epoch}.json", evidence, unit + f"_diagnostic_{epoch}")
            require(diag == done["checkpoint_diagnostics"][str(epoch)] and diag.get("epoch") == diag.get("optimizer_updates_completed") == epoch
                and diag.get("validation_ranking_used_for_selection") is False, "R7_DIAGNOSTIC_RECEIPT_CHANGED")
            add_diagnostic_counts(diag, diagnostic_counts, arm)
        require(finite(done.get("elapsed_seconds")) and done["elapsed_seconds"] >= 0 and count(done.get("peak_GPU_bytes")), "R7_COST_METADATA_INVALID")
        rows.append(dict(arm=arm, partition=part, seed=seed, formal_fits=1, optimizer_updates=80,
            actual_replayed_checkpoint_states=3, elapsed_seconds=done["elapsed_seconds"], peak_allocated_GPU_bytes=done["peak_GPU_bytes"],
            completion_sha256=evidence[unit + "_complete"], actual_replay_sha256=evidence[unit + "_actual_replay"]))
    reference_rows = exact(replay.get("zero_fit_baseline_reference_replays"), ("arm", "partition", "seed"), units(("B0", "B1")), "R7_18_REFERENCE_REPLAYS_REQUIRED")
    require({p.name for p in (private / "references").glob("*.json")} == {name(k) + ".json" for k in reference_rows}, "R7_REFERENCE_FILE_INVENTORY_CHANGED")
    for key, rp in reference_rows.items():
        reference = read(private / "references" / (name(key) + ".json"), evidence, name(key) + "_reference")
        require(reference.get("status") == "R7_BASELINE_REFERENCE" and reference.get("group") == "REFERENCE"
            and tuple(reference.get(k) for k in ("arm", "partition", "seed")) == key and reference.get("contract_sha256") == lock["contract_sha256"]
            and reference.get("actual_new_optimizer_fits") == reference.get("teacher_refits") == 0
            and reference.get("deterministic_seed_reference") is (key[2] != 31)
            and reference.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][str(key[1])]
            and rp.get("status") == "PASS" and rp.get("receipt_sha256") == evidence[name(key) + "_reference"]
            and rp.get("independent_fixed_anchor_metrics_recomputed") is True and rp.get("new_optimizer_fits") == 0, "R7_REFERENCE_RECEIPT_CHAIN_CHANGED")
    schedule = read(private / "SCHEDULE_COMPLETE.json", evidence, "schedule_complete")
    require(not (private / "SCHEDULE_ERROR.json").exists() and schedule.get("status") == "COMPLETE_99_R7_TARGET_FITS_AND_297_CHECKPOINT_REPLAYS"
        and schedule.get("contract_sha256") == lock["contract_sha256"] and schedule.get("formal_fit_count") == 99
        and schedule.get("formal_optimizer_updates") == 7920 and schedule.get("unplanned_candidates") == 0
        and schedule.get("performance_adaptive_budget") is False and finite(schedule.get("elapsed_seconds")) and schedule["elapsed_seconds"] >= 0,
        "R7_SCHEDULER_NOT_COMPLETE")
    scheduled = exact(schedule.get("records"), ("unit",), {(name(k),) for k in units()} | {(f"references_p{p}",) for p in PARTITIONS}
        | {("actual_all_checkpoint_replay",)}, "R7_EXACT_103_SCHEDULED_TASKS_REQUIRED")
    require(all(r.get("returncode") == 0 for r in scheduled.values()), "R7_SCHEDULE_HAS_FAILED_PROCESS")
    replay_public_structure(replay)
    replay_bytes = replay_path.read_bytes()
    require(hashlib.sha256(replay_bytes).hexdigest() == evidence["actual_replay"], "R7_REPLAY_CHANGED_BEFORE_COPY_PREFLIGHT")
    copy_preflight = public_preflight(replay_bytes, denylist=denylist)
    old = old_interface_cost(reports, evidence)
    resource_qual = resource_qualification(private, lock, evidence, schedule["worker_ceiling"])
    resources = resource_summary(private, lock, schedule, resource_qual, evidence)
    ledger = dict(schema="bgb_en_r7_execution_ledger_v1", status="COMPLETE_99_FORMAL_FITS_297_ACTUAL_STATE_REPLAYS",
        contract_sha256=lock["contract_sha256"], delivery_code_sha256=sha(__file__),
        formal=dict(neural_fits=99, optimizer_updates=7920, epochs_per_fit=80, source_teacher_beta_refits=0,
            independent_checkpoint_states_actually_replayed=297, replayed_model_units=99, zero_new_fit_anchor_references=18),
        unsupervised_calibration=dict(fixed_full_train_sets=9, scalar_scales=54, optimizer_updates=0,
            already_completed_before_formal_lock=True, recomputed_by_delivery=False),
        separate_real_qualification=dict(optimizer_instances=0, optimizer_updates=0, explicit_backward_passes=11,
            recorded_complete_model_forward_calls=q["complete_model_forward_calls"], weights_saved=False),
        separate_old_R6_interface_qualification=old, separate_concurrent_resource_qualification=resource_qual,
        separate_training_diagnostics=diagnostic_counts, units=rows, evidence_sha256=evidence,
        accounting=dict(diagnostic_records_are_not_checkpoint_states_or_optimizer_updates=True,
            gradient_path_slots_are_not_actual_backward_call_count=True, unrecorded_training_forward_total=None,
            sum_individual_fit_seconds_is_not_walltime=True, schedule_makespan_seconds=schedule["elapsed_seconds"],
            R6_formal_fits_and_R5_not_recounted=True, source_teacher_beta_refits=0),
        replay_copy=dict(raw_sha256=evidence["actual_replay"], R7_RESULTS_replay_sha256=result["replay_sha256"],
            original_bytes_preserved=True, public_preflight=copy_preflight),
        delivery_runtime=dict(model_forward_calls=0, backward_calls=0, optimizer_updates=0,
            models_arrays_labels_loaded=False, model_artifact_bytes_rehashed=False,
            existing_actual_replays_verified_through_terminal_JSON_SHA_only=True))
    resources["sum_individual_fit_seconds"] = sum(r["elapsed_seconds"] for r in rows)
    return ledger, resources, replay_bytes


def check_upstream(upstream, lock):
    require(upstream.get("status") == "PASS_R7_REUSED_UPSTREAM_39_COX_AND_3_BETA"
        and upstream.get("inherited_R6_contract_sha256") == lock["upstream_r6"]["contract_sha256"], "R7_UPSTREAM_CONTRACT_CHANGED")
    zero(upstream, ("teacher_refits", "source_refits", "new_optimizer_fits"), "R7_UPSTREAM_HAS_REFIT")
    rows = exact(upstream.get("rows"), ("partition",), {(p,) for p in PARTITIONS}, "R7_THREE_UPSTREAM_PARTITIONS_REQUIRED")
    require(all(r.get("status") == "PASS" and r.get("coefficient_models") == 13
        and finite(r.get("beta_gradient_abs")) and 0 <= r["beta_gradient_abs"] <= 1e-6
        and r.get("new_optimizer_fits") == r.get("new_preprocessing_fits") == 0 for r in rows.values()), "R7_UPSTREAM_REPLAY_FAILED")


def add_diagnostic_counts(diag, total, arm):
    gradient, query = diag.get("gradient_probe", {}), diag.get("query_probe", {})
    mb = gradient.get("microbatches")
    require(gradient.get("scope") == "complete_train_riskset_Cox_vector_accumulated_all_microbatches" and type(mb) is int and mb > 0
        and gradient.get("diagnostic_forward_calls") == 2 * mb and gradient.get("diagnostic_gradient_calls_per_microbatch") == 6
        and gradient.get("independent_parameter_L2_gradient_calls") == 1 and gradient.get("new_optimizer_updates") == 0,
        "R7_GRADIENT_DIAGNOSTIC_COUNT_CONTRACT_CHANGED")
    forwards = 2 if arm in ("ADD", "RAW") else 3
    require(query.get("diagnostic_forward_calls") == forwards and query.get("new_backward_calls") == query.get("new_optimizer_updates") == 0,
        "R7_QUERY_DIAGNOSTIC_COUNT_CONTRACT_CHANGED")
    total["saved_diagnostic_receipts"] += 1
    total["recorded_gradient_probe_complete_model_forward_calls"] += 2 * mb
    total["recorded_query_probe_complete_model_forward_calls"] += forwards
    total["scheduled_six_gradient_path_slots"] += 6 * mb
    total["recorded_independent_parameter_L2_gradient_calls"] += 1
    total["complete_riskset_leaf_gradient_calls_code_derived"] += 1


def old_interface_cost(reports, evidence):
    old = read(reports / "R7_0_R6_INTERFACE_AUDIT.json", evidence, "old_R6_interface_qualification")
    require(old.get("status") == "PASS_9_R6_CELLS_36_STATE_FORWARDS" and old.get("actual_model_forwards") == old.get("rows_count") == 36
        and old.get("new_backward_calls") == old.get("new_optimizer_fits") == 0, "R7_OLD_INTERFACE_COST_NOT_COMPLETE")
    exact(old.get("rows"), ("partition", "seed", "epoch"), {(p, s, e) for p in PARTITIONS for s in SEEDS for e in (0, 20, 40, 80)}, "R7_OLD_INTERFACE_EXACT_36_STATES_REQUIRED")
    return dict(old_R6_cells=9, actual_model_forwards=36, new_backward_calls=0, optimizer_fits=0,
        optimizer_updates=0, scope="R6_initial_20_40_80_interface_probe_not_R7_formal_states")


def resource_qualification(private, lock, evidence, worker_ceiling):
    path = private / "RESOURCE_QUALIFICATION.json"
    if not path.exists():
        require(worker_ceiling <= 4, "R7_MORE_THAN_FOUR_WORKERS_WITHOUT_RESOURCE_QUALIFICATION")
        return dict(status="NOT_RECORDED_OPTIONAL_AT_FOUR_OR_LESS", qualified=False, optimizer_updates=0)
    value = read(path, evidence, "resource_qualification")
    require(value.get("status") == "PASS_R7_CONCURRENT_RESOURCE_QUALIFICATION" and value.get("contract_sha256") == lock["contract_sha256"]
        and value.get("qualification_sha256") == evidence["qualification"] and value.get("observed_real_forward_backward") is True
        and value.get("observed_concurrent_workers") == 4 and value.get("qualified_max_workers") == 6
        and value.get("model_forward_calls") == value.get("backward_calls") == 12
        and value.get("optimizer_updates") == 0 and value.get("weights_saved") is False, "R7_RESOURCE_QUALIFICATION_LINEAGE_OR_COUNTS_CHANGED")
    rows = exact(value.get("workers"), ("slot",), {(i,) for i in range(4)}, "R7_FOUR_RESOURCE_PROBE_WORKERS_REQUIRED")
    require(all(r.get("status") == "PASS" and r.get("arm") == "GATE2"
        and r.get("model_forward_calls") == r.get("backward_calls") == 3 and r.get("optimizer_updates") == 0
        and r.get("weights_saved") is False for r in rows.values()), "R7_RESOURCE_WORKER_COUNTS_CHANGED")
    return dict(status=value["status"], qualified=True, actual_model_forwards=12, actual_backward_calls=12,
        optimizer_updates=0, weights_saved=False, observed_concurrent_workers=4, qualified_max_workers=6,
        cgroup_memory_limit_bytes=value.get("resource_measurements", {}).get("RAM_total_bytes"),
        measurement_scope="separate_real_four_worker_GATED2_engineering_qualification_not_formal_fit")


def jsonl(path, evidence, label):
    require(path.is_file() and not path.is_symlink(), "RESOURCE_JSONL_MISSING_OR_UNSAFE")
    before = sha(path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    require(before == sha(path) and all(isinstance(r, dict) for r in lines), "RESOURCE_JSONL_STILL_CHANGING_OR_INVALID")
    evidence[label] = before
    return lines


def summarize_window(rows, time_key, fields, code):
    require(len(rows) >= 2, code + "_AT_LEAST_TWO_SAMPLES_REQUIRED")
    require(all(finite(r.get(time_key)) and all(finite(r.get(k)) and r[k] >= 0 for k in fields) for r in rows), code + "_NONFINITE_SAMPLE")
    dt = [b[time_key] - a[time_key] for a, b in zip(rows, rows[1:])]
    require(all(v > 0 for v in dt), code + "_TIMESTAMPS_NOT_STRICTLY_INCREASING")
    nominal = statistics.median(dt); valid = [i for i, v in enumerate(dt) if v <= 3 * nominal]
    duration = sum(dt[i] for i in valid)
    return dict(samples=len(rows), observed_window_seconds=rows[-1][time_key] - rows[0][time_key],
        supported_interpolation_seconds=duration, excluded_long_gap_seconds=sum(dt) - duration,
        nominal_interval_seconds=nominal, statistics={k: dict(sample_min=min(r[k] for r in rows), sample_max=max(r[k] for r in rows),
            time_weighted_mean=sum(.5 * (rows[i][k] + rows[i + 1][k]) * dt[i] for i in valid) / duration if duration else None) for k in fields},
        time_weighting="trapezoidal_excluding_gaps_over_three_median_intervals", unsampled_intervals_not_backfilled=True)


def resource_summary(private, lock, schedule, qualification, evidence):
    driver_path = private / "RESOURCE_MONITOR.jsonl"
    driver = jsonl(driver_path, evidence, "driver_resource_monitor")
    gpu_counts = {len(r.get("GPU", [])) for r in driver}
    require(len(gpu_counts) == 1 and next(iter(gpu_counts)) > 0, "R7_DRIVER_VISIBLE_GPU_INVENTORY_CHANGED_OR_ABSENT")
    parsed = []
    for row in driver:
        stamp = datetime.fromisoformat(row["UTC"].replace("Z", "+00:00")).timestamp()
        current = dict(unix_time=stamp, host_RAM_available_bytes=row.get("RAM_available_bytes"),
            host_RAM_total_bytes=row.get("RAM_total_bytes"), disk_free_bytes=row.get("disk_free_bytes"))
        for index, gpu in enumerate(row["GPU"]):
            require(finite(gpu.get("utilization_percent")) and 0 <= gpu["utilization_percent"] <= 100, "R7_GPU_UTILIZATION_INVALID")
            for key in ("total_MiB", "used_MiB", "free_MiB", "utilization_percent"):
                current[f"visible_GPU{index}_" + key] = gpu.get(key)
        parsed.append(current)
    fields = list(parsed[0].keys() - {"unix_time"})
    result = dict(schema="bgb_en_r7_resource_summary_v1", status="RECORDED_RESOURCE_WINDOWS_ONLY",
        contract_sha256=lock["contract_sha256"], delivery_code_sha256=sha(__file__),
        driver_monitor=summarize_window(parsed, "unix_time", fields, "R7_DRIVER_RESOURCE"),
        driver_monitor_file="RESOURCE_MONITOR.jsonl", visible_GPU_samples_are_not_per_process_peaks=True,
        host_psutil_RAM_is_not_container_available_memory=True, schedule_makespan_seconds=schedule["elapsed_seconds"],
        sum_fit_seconds_is_not_walltime=True, cgroup_monitor=dict(status="NOT_RECORDED"),
        cgroup_capacity=dict(memory_limit_bytes=qualification.get("cgroup_memory_limit_bytes"),
            CPU_quota_equivalent_cores=16, CPU_quota_source="protocol_declared_not_current_available_CPU_measurement",
            scope="capacity_not_available_or_headroom"))
    path = private / "cgroup_resource_samples.jsonl"
    if path.exists():
        rows = jsonl(path, evidence, "cgroup_resource_monitor")
        fields = ("cgroup_memory_current_bytes", "cgroup_memory_limit_bytes", "data_disk_free_bytes")
        window = summarize_window(rows, "unix_time", fields, "R7_CGROUP_RESOURCE")
        limits = {r["cgroup_memory_limit_bytes"] for r in rows}
        starts = {r.get("monitor_started_formal_elapsed_seconds") for r in rows}
        require(len(limits) == len(starts) == 1 and all(finite(v) and v >= 0 for v in starts)
            and (qualification.get("cgroup_memory_limit_bytes") is None or next(iter(limits)) == qualification["cgroup_memory_limit_bytes"]),
            "R7_CGROUP_CAPACITY_OR_MONITOR_START_CHANGED")
        window.update(status="PARTIAL_MONITOR_WINDOW_ONLY", monitor_started_formal_elapsed_seconds=next(iter(starts)),
            earlier_training_window_not_observed_or_backfilled=True)
        result["cgroup_monitor"] = window
        result["cgroup_capacity"]["memory_limit_bytes"] = next(iter(limits))
    cpu_path = private.parent / "CPU_CAPACITY_OBSERVATION.private.json"
    if cpu_path.exists():
        observed = read(cpu_path, evidence, "CPU_capacity_observation")
        raw = observed.get("cpu_max_raw", "").split()
        require(len(raw) == 2 and all(v.isdecimal() and int(v) > 0 for v in raw)
            and finite(observed.get("quota_equivalent_cores"))
            and math.isclose(observed["quota_equivalent_cores"], int(raw[0]) / int(raw[1]), rel_tol=0., abs_tol=1e-12)
            and finite(observed.get("unix_time")) and observed.get("scope")
            == "current_cgroup_capacity_snapshot_not_current_available_CPU_or_training_utilization",
            "R7_CPU_CAPACITY_OBSERVATION_INVALID")
        result["cgroup_capacity"].update(CPU_quota_equivalent_cores=observed["quota_equivalent_cores"],
            CPU_quota_source=observed["scope"], CPU_capacity_observation_unix_time=observed["unix_time"],
            CPU_capacity_observation_sha256=evidence["CPU_capacity_observation"], CPU_whole_run_utilization_not_measured=True)
    result["resource_metadata_sha256"] = {k: v for k, v in evidence.items() if "resource" in k}
    return result


def export_completion(private, reports, *, denylist=None):
    reports = Path(reports)
    names = ("R7_EXECUTION_LEDGER.json", "R7_RESOURCE_SUMMARY.json", "R7_CHECKPOINT_REPLAY_AUDIT.json")
    require(not any((reports / n).exists() for n in names), "EXISTING_COMPLETION_DELIVERY_PRESERVED")
    ledger, resource, raw_replay = complete_account(private, reports, denylist=denylist)
    payloads = (json_bytes(ledger), json_bytes(resource), raw_replay)
    for data in payloads:
        public_preflight(data, denylist=denylist)
    for filename, data in zip(names, payloads):
        exclusive_write(reports / filename, data)
    require(sha(reports / names[2]) == ledger["replay_copy"]["R7_RESULTS_replay_sha256"], "R7_EXACT_REPLAY_COPY_SHA_CHANGED")
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualification", "completion"), required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--denylist", type=Path, help="Optional real private registry; absent means generic preflight only")
    args = parser.parse_args()
    require(args.reports.resolve() == (ROOT / "reports/bgb_en_r7").resolve()
        or not args.reports.resolve().is_relative_to((ROOT / "reports").resolve()), "ONLY_NEW_R7_PUBLIC_NAMESPACE_ALLOWED")
    result = export_qualification(args.private_root, args.reports, denylist=args.denylist) if args.mode == "qualification" else export_completion(args.private_root, args.reports, denylist=args.denylist)
    print(json.dumps(dict(status=result["status"], model_forwards=0, optimizer_updates=0)))


if __name__ == "__main__":
    main()

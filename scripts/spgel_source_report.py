"""Metadata-only, fail-closed reporting for the locked SPGEL source pilot.

No tensors, model checkpoints, raw arrays or survival data are loaded here.
The report is assembled only after the complete registered pilot matrix and
its actual saved-state replay receipts have been validated.
"""
from __future__ import annotations

import hashlib
import argparse
import csv
import json
import math
from pathlib import Path
import re


class ReportContractError(ValueError):
    """The metadata cannot support the requested complete source report."""


def require(condition, code):
    if not bool(condition):
        raise ReportContractError(code)


def _reject_constant(value):
    raise ReportContractError("NONFINITE_JSON_CONSTANT:" + value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY:" + key)
        result[key] = value
    return result


def read_metadata(path):
    """Read a small JSON receipt; never unpickle a model or an array."""
    path = Path(path)
    require(path.suffix.lower() == ".json", "REPORT_ACCEPTS_JSON_METADATA_ONLY")
    require(path.is_file() and path.stat().st_size <= 32 * 1024**2,
            "METADATA_FILE_MISSING_OR_TOO_LARGE")
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    require(isinstance(value, dict), "METADATA_ROOT_MUST_BE_OBJECT")
    return value


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path):
    """Hash a JSON receipt/config only, without opening checkpoint bytes."""
    path = Path(path)
    require(path.suffix.lower() == ".json", "REPORT_HASHES_JSON_METADATA_ONLY")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value, code):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, code)
    return value


def finite_number(value, code, *, minimum=None, maximum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), code)
    if minimum is not None:
        require(value >= minimum, code)
    if maximum is not None:
        require(value <= maximum, code)
    return float(value)


def mean_defined(values):
    """Return a mean with explicit undefined support, never convert null to zero."""
    values = list(values)
    selected = []
    for value in values:
        if value is None:
            continue
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value), "INVALID_SCALAR_METRIC")
        selected.append(float(value))
    return dict(mean=sum(selected) / len(selected) if selected else None,
                defined_count=len(selected), expected_count=len(values),
                undefined_count=len(values)-len(selected), undefined_is_not_zero=True)


def validate_config(config):
    """Bind the pre-registered source budget, rather than infer it from files."""
    require(config.get("schema") == "spgel_source_pilot_recipe_v1", "WRONG_PILOT_CONFIG_SCHEMA")
    require(config.get("trainable_arms") == ["S_A" + str(k) for k in range(6)]
            and config.get("heldout_donor_codes") == list(range(6))
            and config.get("seeds") == [31, 47] and config.get("primary_candidate") == "S_A5",
            "REGISTERED_SOURCE_MATRIX_CHANGED")
    require(config.get("training", {}).get("save_epochs") == [10, 20, 40]
            and config["training"].get("epochs") == config["training"].get("primary_epoch") == 40,
            "REGISTERED_FIXED_EPOCH_CHANGED")
    expected = dict(pilot_logical_cells=72, pilot_convex_optimizer_fits=12,
        pilot_neural_optimizer_fits=60, neural_optimizer_updates=52000,
        neural_updates_per_fit_by_heldout_donor=[880, 720, 960, 880, 960, 800],
        neural_saved_checkpoint_states=180,
        convex_solver_final_states=12, pilot_actual_replay_states=192,
        unplanned_or_automatic_retry_fits=0)
    require(config.get("budget") == expected, "REGISTERED_SOURCE_BUDGET_CHANGED")
    require(len(config.get("program_order", [])) == 4
            and len(set(config["program_order"])) == 4, "FOUR_FIXED_SOURCE_PROGRAMS_REQUIRED")
    require(config.get("evaluation", {}).get("official_HVG_task") ==
            "NOT_RUN_SEPARATE_OFFICIAL_HVG_LOG1P_ONLY_CONTRACT",
            "OFFICIAL_HVG_CONTRACT_MUST_STAY_SEPARATE")
    scope = config.get("scope", {})
    for key in ("DSS_or_PFI_read", "TCGA_loaded", "R8_results_influence_source",
                "source_test_for_model_selection", "independent_source_confirmation_claimed",
                "histological_or_cell_source_truth_claimed"):
        require(scope.get(key) is False, "SOURCE_SCOPE_INVALID:" + key)
    return config


def expected_units(config, *, stage="pilot"):
    validate_config(config)
    if stage == "pilot":
        return {(arm, donor, seed) for arm in config["trainable_arms"]
                for donor in config["heldout_donor_codes"] for seed in config["seeds"]}
    require(stage == "s4", "UNKNOWN_SOURCE_REPORT_STAGE")
    return {("S_A1", -1, 31), ("S_A5", -1, 31)}


def cell_name(key, *, stage="pilot"):
    arm, donor, seed = key
    return f"{arm}_d{donor}_seed{seed}" if stage == "pilot" else f"{arm}_full_seed{seed}"


def _key(record):
    key = (record.get("arm"), record.get("heldout_donor"), record.get("seed"))
    require(isinstance(key[0], str) and type(key[1]) is int and type(key[2]) is int,
            "CELL_KEY_REQUIRED")
    return key


def _state_labels(arm):
    return {"solver_final": None} if arm == "S_A0" else {"epoch10": 10, "epoch20": 20, "epoch40": 40}


def _snapshot_map(snapshots, key):
    require(isinstance(snapshots, list), "SNAPSHOT_LIST_REQUIRED")
    result = {}
    for state in snapshots:
        label = state.get("label")
        require(label not in result, "DUPLICATE_SAVED_STATE")
        result[label] = state
    require(set(result) == set(_state_labels(key[0])), "INCOMPLETE_OR_UNREGISTERED_SAVED_STATES")
    return result


def validate_cell(cell, key, *, config_sha256, contract_sha256, stage):
    require(cell.get("schema") == "spgel_source_cell_v1"
            and cell.get("status") == "COMPLETE_SPGEL_SOURCE_FIT", "SOURCE_CELL_NOT_COMPLETE")
    require(_key(cell) == key and cell.get("stage") == stage, "CELL_IDENTITY_OR_STAGE_CHANGED")
    require(cell.get("config_sha256") == config_sha256
            and cell.get("contract_sha256") == contract_sha256, "CELL_LINEAGE_CHANGED")
    digest(cell.get("artifact_sha256"), "CELL_ARTIFACT_SHA_REQUIRED")
    convex = key[0] == "S_A0"
    require(cell.get("fit_kind") == ("convex" if convex else "neural")
            and cell.get("convex_optimizer_fits") == int(convex)
            and cell.get("neural_optimizer_fits") == int(not convex), "CELL_FIT_COUNT_CHANGED")
    train_rows = cell.get("train_rows")
    require(type(train_rows) is int and train_rows > 1, "CELL_TRAIN_SUPPORT_INVALID")
    require(type(cell.get("heldout_rows")) is int and
            (cell["heldout_rows"] > 1 if stage == "pilot" else cell["heldout_rows"] == 0),
            "CELL_HELDOUT_SUPPORT_INVALID")
    updates = 0 if convex else 40 * math.ceil(train_rows / 512)
    require(cell.get("optimizer_updates") == updates
            and cell.get("epochs") == (None if convex else 40), "CELL_TRAINING_BUDGET_CHANGED")
    if not convex:
        locked_updates = [880, 720, 960, 880, 960, 800][key[1]] if stage == "pilot" else 1040
        require(updates == locked_updates, "CELL_REGISTERED_DONOR_UPDATE_BUDGET_CHANGED")
    parameters = cell.get("parameters", {})
    require(all(type(parameters.get(k)) is int and parameters[k] >= 0 for k in ("optimized", "registered"))
            and parameters["optimized"] <= parameters["registered"], "CELL_PARAMETER_COUNTS_INVALID")
    if convex:
        require(parameters.get("analytic_program_table_coefficients") == 12,
                "TISSUE_TABLE_IS_NOT_EXTRA_OPTIMIZER_FIT")
    require(train_rows + cell["heldout_rows"] == 13011, "SOURCE_ROW_SUPPORT_CHANGED")
    finite_number(cell.get("elapsed_seconds"), "CELL_RUNTIME_INVALID", minimum=0)
    finite_number(cell.get("peak_GPU_bytes"), "CELL_GPU_COUNTER_INVALID", minimum=0)
    states = _snapshot_map(cell.get("snapshots"), key)
    for label, state in states.items():
        require(state.get("epoch") == _state_labels(key[0])[label], "SAVED_STATE_EPOCH_CHANGED")
        digest(state.get("checkpoint_sha256"), "SAVED_STATE_SHA_REQUIRED")
    return states


def validate_replay(replay, cells, completion_sha256, *, config_sha256, contract_sha256, stage):
    expected_count, expected_states = (72, 192) if stage == "pilot" else (2, 6)
    expected_status = ("PASS_COMPLETE_SPGEL_PILOT_72_FITS_192_REPLAYS" if stage == "pilot" else
                       "PASS_COMPLETE_SPGEL_S4_2_FITS_6_REPLAYS")
    require(replay.get("schema") == "spgel_source_replay_v1"
            and replay.get("status") == expected_status, "ACTUAL_SAVED_STATE_REPLAY_NOT_COMPLETE")
    require(replay.get("config_sha256") == config_sha256
            and replay.get("contract_sha256") == contract_sha256, "REPLAY_LINEAGE_CHANGED")
    require(replay.get("expected_units") == replay.get("passed_units") == expected_count
            and replay.get("failed_units") == 0
            and replay.get("expected_states") == replay.get("passed_states") == expected_states,
            "REPLAY_COVERAGE_COUNTS_CHANGED")
    units = replay.get("units")
    require(isinstance(units, list) and len(units) == expected_count, "REPLAY_UNIT_LIST_INCOMPLETE")
    found, state_count = set(), 0
    max_prediction, max_metric = 0., 0.
    for unit in units:
        key = _key(unit)
        require(key in cells and key not in found, "REPLAY_UNIT_MISSING_DUPLICATED_OR_UNREGISTERED")
        found.add(key)
        cell = cells[key]
        require(unit.get("status") == "PASS" and unit.get("new_optimizer_updates") == 0
                and unit.get("new_optimizer_fits") == 0
                and unit.get("actual_saved_state_loaded") is True, "REPLAY_MUST_NOT_REFIT")
        require(unit.get("completion_sha256") == completion_sha256[key]
                and unit.get("artifact_sha256") == cell["artifact_sha256"], "REPLAY_NOT_BOUND_TO_COMPLETED_FIT")
        saved = _snapshot_map(cell["snapshots"], key)
        replayed = _snapshot_map(unit.get("snapshots"), key)
        for label, state in replayed.items():
            require(state.get("checkpoint_sha256") == saved[label]["checkpoint_sha256"]
                    and state.get("actual_state_forward_replayed") is True,
                    "REPLAY_MUST_LOAD_ACTUAL_REGISTERED_SAVED_STATE")
            prediction = finite_number(state.get("maximum_prediction_error"), "REPLAY_PREDICTION_ERROR_INVALID",
                                       minimum=0, maximum=2e-6)
            metric = finite_number(state.get("maximum_metric_error"), "REPLAY_METRIC_ERROR_INVALID",
                                  minimum=0, maximum=1e-8)
            max_prediction, max_metric = max(max_prediction, prediction), max(max_metric, metric)
            state_count += 1
    require(found == set(cells) and state_count == expected_states, "REPLAY_ACTUAL_STATE_COVERAGE_INCOMPLETE")
    return dict(passed_units=len(found), passed_states=state_count, new_optimizer_fits=0,
                maximum_prediction_error=max_prediction, maximum_metric_error=max_metric)


def load_validated_matrix(private, config, config_sha256, lock, *, stage="pilot"):
    require(lock.get("schema") == "spgel_source_run_lock_v1"
            and lock.get("status") == "LOCKED_BEFORE_SPGEL_FITS"
            and lock.get("config_sha256") == config_sha256, "SOURCE_RUN_LOCK_INVALID")
    contract = digest(lock.get("contract_sha256"), "SOURCE_CONTRACT_SHA_REQUIRED")
    require(contract == canonical_sha256({k:v for k,v in lock.items() if k != "contract_sha256"}),
            "SOURCE_RUN_LOCK_CONTENT_CHANGED")
    required = dict(logical_cells=72, neural_fits=60, convex_fits=12, actual_saved_states=192)
    require(all(lock.get("pilot", {}).get(k) == v for k,v in required.items())
            and all(lock.get("s4", {}).get(k) == v for k,v in dict(neural_fits=2, actual_saved_states=6).items()),
            "SOURCE_LOCKED_BUDGET_CHANGED")
    require(isinstance(lock.get("sources"), dict) and len(lock["sources"]) > 0, "SOURCE_CODE_BINDING_REQUIRED")
    for value in lock["sources"].values():
        digest(value, "SOURCE_CODE_SHA_INVALID")
    directory = Path(private) / stage
    expected = expected_units(config, stage=stage)
    actual_paths = list((directory / "cells").glob("*/COMPLETE.json"))
    expected_names = {cell_name(k, stage=stage) for k in expected}
    require(len(actual_paths) == len(expected) and {p.parent.name for p in actual_paths} == expected_names,
            "COMPLETE_MATRIX_REQUIRED_NO_PARTIAL_MEANS")
    cells, hashes = {}, {}
    for key in sorted(expected):
        path = directory / "cells" / cell_name(key, stage=stage) / "COMPLETE.json"
        cell = read_metadata(path)
        validate_cell(cell, key, config_sha256=config_sha256, contract_sha256=contract, stage=stage)
        cells[key], hashes[key] = cell, file_sha256(path)
    replay = read_metadata(directory / "REPLAY_AUDIT.json")
    audit = validate_replay(replay, cells, hashes, config_sha256=config_sha256,
                            contract_sha256=contract, stage=stage)
    return cells, audit


def regression_summary(metrics, *, rows):
    """Rebuild donor-level four-program macros from replayed JSON scalars."""
    per = metrics.get("per_output")
    require(isinstance(per, list) and len(per) == 4
            and [r.get("column") for r in per] == list(range(4)), "FOUR_PROGRAM_METRICS_REQUIRED")
    result = []
    for item in per:
        count = item.get("observed_count")
        require(type(count) is int and 0 <= count <= rows, "PROGRAM_METRIC_SUPPORT_INVALID")
        row = dict(column=item["column"], observed_count=count)
        for name in ("Pearson", "Spearman", "MAE", "MSE", "Huber_delta1"):
            require(name in item, "PROGRAM_METRIC_FIELD_MISSING")
            value = item.get(name)
            if value is not None:
                finite_number(value, "NONFINITE_PROGRAM_METRIC", minimum=-1 if name in ("Pearson", "Spearman") else 0,
                              maximum=1 if name in ("Pearson", "Spearman") else None)
                require(count > 0, "NONNULL_METRIC_WITHOUT_OBSERVATIONS")
            if name in ("MAE", "MSE", "Huber_delta1"):
                require((value is not None) == (count > 0), "PROGRAM_ERROR_SUPPORT_CHANGED")
            row[name] = value
        result.append(row)
    pcc = mean_defined(r["Pearson"] for r in result)
    strict = pcc["mean"] if pcc["defined_count"] == 4 else None
    require(metrics.get("defined_Pearson_count") == pcc["defined_count"], "PROGRAM_DEFINED_COUNT_CHANGED")
    for field, name in (("Mean_Pearson", "Pearson"), ("Mean_MAE", "MAE"),
                        ("Mean_MSE", "MSE"), ("Mean_Huber_delta1", "Huber_delta1")):
        expected = mean_defined(r[name] for r in result)["mean"]
        supplied = metrics.get(field)
        require((expected is None and supplied is None) or
                (expected is not None and supplied is not None and
                 abs(finite_number(supplied, "INVALID_METRIC_MACRO")-expected) <= 1e-10),
                "PROGRAM_MACRO_DOES_NOT_REPLAY_FROM_METADATA")
    return dict(per_output=result, strict_four_program_Pearson=strict,
                defined_programs=pcc["defined_count"], registered_programs=4,
                Mean_MAE=mean_defined(r["MAE"] for r in result),
                Mean_MSE=mean_defined(r["MSE"] for r in result),
                Mean_Huber_delta1=mean_defined(r["Huber_delta1"] for r in result))


def task_a_summary(task, *, rows):
    require(task.get("task") == "IMAGE_ONLY_ST_PREDICTION" and task.get("program_count") == 4
            and task.get("scale") == "raw_signed_log1pCP10K_program_and_current_train_Z"
            and task.get("primary_requires_all_four_defined") is True
            and task.get("true_RNA_or_tissue_not_prediction_input") is True,
            "TASK_A_INTERFACE_OR_SCALE_CHANGED")
    native = regression_summary(task.get("native", {}), rows=rows)
    z = regression_summary(task.get("train_Z", {}), rows=rows)
    primary = native["strict_four_program_Pearson"]
    require(task.get("defined_program_count") == native["defined_programs"], "TASK_A_PROGRAM_SUPPORT_CHANGED")
    supplied = task.get("primary_donor_four_program_Pearson")
    require((primary is None and supplied is None) or
            (primary is not None and supplied is not None and abs(primary-supplied) <= 1e-10),
            "TASK_A_PRIMARY_DONOR_MACRO_CHANGED")
    return dict(task="IMAGE_ONLY_ST_PREDICTION", native=native, train_Z=z,
                primary_donor_four_program_Pearson=primary,
                scale="raw_signed_log1pCP10K_program_and_current_train_Z")


def task_c_summary(task, *, rows):
    require(task.get("task") == "PROGRAM_TO_REGION_LOCALIZATION"
            and task.get("threshold_fit_scope") == "source_training_rows_only"
            and task.get("lesion_segmentation_truth_claimed") is False,
            "TASK_C_THRESHOLDS_MUST_BE_TRAIN_ONLY")
    per = task.get("per_output")
    require(isinstance(per, list) and len(per) == 4
            and [r.get("column") for r in per] == list(range(4)), "TASK_C_FOUR_PROGRAMS_REQUIRED")
    selected = []
    for item in per:
        count = item.get("observed_count");positives = item.get("high_activity_count")
        require(type(count) is int and 0 <= count <= rows and type(positives) is int
                and 0 <= positives <= count, "TASK_C_SUPPORT_INVALID")
        row = dict(column=item["column"], observed_count=count, high_activity_count=positives)
        for name in ("Pearson", "Spearman", "AUPRC", "prevalence", "high_activity_threshold_from_train"):
            value = item.get(name)
            if value is not None:
                finite_number(value, "TASK_C_METRIC_INVALID",
                    minimum=-1 if name in ("Pearson", "Spearman") else (0 if name in ("AUPRC", "prevalence") else None),
                    maximum=1 if name in ("Pearson", "Spearman", "AUPRC", "prevalence") else None)
            if name == "AUPRC":
                require((value is not None) == (0 < positives < count), "TASK_C_UNDEFINED_AUPRC_HANDLING")
            row[name] = value
        selected.append(row)
    pcc = mean_defined(r["Pearson"] for r in selected)
    auprc = mean_defined(r["AUPRC"] for r in selected)
    strict_auprc = auprc["mean"] if auprc["defined_count"] == 4 else None
    supplied = task.get("primary_four_program_AUPRC")
    require(task.get("defined_AUPRC_count") == auprc["defined_count"]
            and task.get("primary_requires_all_four_defined") is True,
            "TASK_C_DEFINED_AUPRC_COUNT_CHANGED")
    require((strict_auprc is None and supplied is None) or
            (strict_auprc is not None and supplied is not None and abs(strict_auprc-supplied) <= 1e-10),
            "TASK_C_PRIMARY_AUPRC_MACRO_CHANGED")
    return dict(task="PROGRAM_TO_REGION_LOCALIZATION", per_output=selected,
        strict_four_program_Pearson=pcc["mean"] if pcc["defined_count"] == 4 else None,
        defined_Pearson_programs=pcc["defined_count"], AUPRC=auprc,
        primary_four_program_AUPRC=strict_auprc,
        threshold_fit_scope="source_training_rows_only", lesion_segmentation_truth_claimed=False)


def task_b_summary(task, *, native_applicable):
    if not native_applicable:
        require(task.get("status") == "NOT_APPLICABLE_NO_TRAINED_GENE_ENCODER",
                "NATIVE_RETRIEVAL_REQUIRES_TRAINED_GENE_ENCODER")
        return dict(status="NOT_APPLICABLE_NO_TRAINED_GENE_ENCODER")
    require(task.get("status") == "COMPLETE" and task.get("candidate_scope") ==
            "entire_same_heldout_slide_no_tissue_depth_filter", "TASK_B_CANDIDATE_PROTOCOL_CHANGED")
    require(task.get("query_anchor_cap_per_slide") == 128, "TASK_B_FIXED_QUERY_BUDGET_CHANGED")
    result = dict(status="COMPLETE", candidate_scope=task["candidate_scope"], query_anchor_cap_per_slide=128)
    for direction in ("image_to_gene", "gene_to_image"):
        raw = task.get(direction, {});queries = raw.get("queries_count")
        require(type(queries) is int and queries > 0, "TASK_B_QUERY_SUPPORT_MISSING")
        row = dict(queries_count=queries)
        for name in ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC"):
            value = raw.get(name)
            if value is not None:
                finite_number(value, "TASK_B_METRIC_INVALID", minimum=1 if name == "median_rank" else 0,
                              maximum=None if name == "median_rank" else 1)
            row[name] = value
        result[direction] = row
    sections = task.get("per_section")
    require(isinstance(sections, list) and len(sections) == 2, "TASK_B_TWO_SLIDE_CANDIDATE_LIBRARIES_REQUIRED")
    selected, seen = [], set()
    for item in sections:
        code, anchors, count = item.get("section_code"), item.get("anchors_count"), item.get("candidates_count")
        require(type(code) is int and code not in seen and type(count) is int and count > 1
                and type(anchors) is int and anchors == min(128, count), "TASK_B_SECTION_CANDIDATE_SUPPORT_INVALID")
        seen.add(code)
        row = dict(section_code=code, anchors_count=anchors, candidates_count=count)
        for direction in ("image_to_gene", "gene_to_image"):
            raw = item.get(direction, {})
            require(raw.get("queries_count") == anchors and raw.get("candidates_count") == count
                    and raw.get("candidate_negative_count") == count-1, "TASK_B_DIRECTION_SUPPORT_CHANGED")
            metrics = {name:raw.get(name) for name in ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC")}
            for name, value in metrics.items():
                if value is not None:
                    finite_number(value, "TASK_B_SECTION_METRIC_INVALID", minimum=1 if name == "median_rank" else 0,
                                  maximum=None if name == "median_rank" else 1)
            row[direction] = dict(**metrics, queries_count=anchors)
        selected.append(row)
    for direction in ("image_to_gene", "gene_to_image"):
        require(result[direction]["queries_count"] == sum(s["anchors_count"] for s in selected),
                "TASK_B_TOTAL_QUERIES_CHANGED")
        for name in ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC"):
            values = [r[direction][name] for r in selected]
            expected = sum(values)/len(values) if all(v is not None for v in values) else None
            value = result[direction][name]
            require((expected is None and value is None) or
                    (expected is not None and value is not None and abs(expected-value) <= 1e-10),
                    "TASK_B_EQUAL_SLIDE_MACRO_CHANGED")
    result["per_section"] = selected
    result["within_donor_aggregation"] = "equal_slide_macro_not_independent_spots"
    return result


def stratified_summary(heldout, *, rows):
    section = heldout.get("section_metrics", {})
    require(section.get("task") == "within_slide_program_prediction"
            and isinstance(section.get("sections"), list), "WITHIN_SLIDE_REPORT_REQUIRED")
    sections, seen = [], set()
    for item in section["sections"]:
        code, n = item.get("section_code"), item.get("spots_count")
        require(type(code) is int and code not in seen and type(n) is int and 0 < n <= rows,
                "SECTION_METRIC_SUPPORT_INVALID")
        seen.add(code)
        sections.append(dict(section_code=code, spots_count=n, task_A=task_a_summary(item.get("task_A", {}), rows=n)))
    require(len(sections) == 2 and sum(s["spots_count"] for s in sections) == rows,
            "TWO_SOURCE_SECTIONS_PER_HELDOUT_DONOR_REQUIRED")
    tissue = heldout.get("tissue_metrics", {})
    prediction = tissue.get("image_predicted_tissue", {})
    safe_prediction = dict(status=prediction.get("status"))
    if prediction.get("status") == "COMPLETE":
        ba = prediction.get("balanced_accuracy")
        recall, counts = prediction.get("per_class_recall"), prediction.get("true_class_counts")
        require(isinstance(recall, list) and len(recall) == 3 and isinstance(counts, list) and len(counts) == 3
                and all(type(x) is int and x >= 0 for x in counts) and sum(counts) == rows,
                "TISSUE_CLASS_SUPPORT_INVALID")
        for score, count in zip(recall, counts):
            require((score is not None) == (count > 0), "UNDEFINED_TISSUE_RECALL_SUPPORT_CHANGED")
            if score is not None:
                finite_number(score, "TISSUE_RECALL_INVALID", minimum=0, maximum=1)
        if all(score is not None for score in recall):
            finite_number(ba, "TISSUE_BALANCED_ACCURACY_INVALID", minimum=0, maximum=1)
            require(abs(ba-sum(recall)/3) <= 1e-10, "TISSUE_BALANCED_ACCURACY_MACRO_CHANGED")
        else:
            require(ba is None, "TISSUE_MISSING_CLASS_MUST_KEEP_BALANCED_ACCURACY_UNDEFINED")
        safe_prediction.update(balanced_accuracy=ba, per_class_recall=recall, true_class_counts=counts)
    else:
        require(prediction.get("status") == "NOT_APPLICABLE_NO_TISSUE_HEAD", "TISSUE_HEAD_STATUS_REQUIRED")
    within = tissue.get("within_slide_tissue", {})
    require(within.get("task") == "within_slide_tissue_stratified_prediction"
            and within.get("tissue_truth_used_for_evaluation_groups_only") is True
            and within.get("labels_used_as_model_inputs") is False
            and isinstance(within.get("groups"), list), "EVALUATION_ONLY_TISSUE_GROUPS_REQUIRED")
    groups, group_keys = [], set()
    for item in within["groups"]:
        code, label, n = item.get("section_code"), item.get("tissue_code"), item.get("spots_count")
        key = (code, label)
        require(code in seen and type(label) is int and label in (0, 1, 2) and key not in group_keys
                and type(n) is int and 0 < n <= rows, "WITHIN_SLIDE_TISSUE_SUPPORT_INVALID")
        group_keys.add(key)
        groups.append(dict(section_code=code, tissue_code=label, spots_count=n,
            metrics=regression_summary(item.get("metrics", {}), rows=n)))
    require(sum(r["spots_count"] for r in groups) == rows
            and all(sum(r["spots_count"] for r in groups if r["section_code"] == s["section_code"]) == s["spots_count"]
                    for s in sections), "WITHIN_SLIDE_TISSUE_GROUPS_DO_NOT_PARTITION_ROWS")
    return dict(sections=sections, image_predicted_tissue=safe_prediction, within_slide_tissue=groups,
                true_tissue_is_evaluation_metadata_only=True)


def loss_diagnostics(state, *, convex):
    losses = state.get("losses", {})
    if convex:
        require(type(losses.get("solver_iterations")) is int and losses["solver_iterations"] >= 0,
                "CONVEX_SOLVER_COUNTER_MISSING")
        gradient = finite_number(losses.get("solver_gradient_infinity_norm"), "CONVEX_FINAL_GRADIENT_INVALID",
                                 minimum=0, maximum=1e-6)
        return dict(solver_iterations=losses["solver_iterations"], solver_gradient_infinity_norm=gradient)
    names = ["epoch_minibatch_preupdate_mean_"+name for name in ("total", "prediction_Huber", "tissue_CE", "ITC")]
    return {name:finite_number(losses.get(name), "TRAIN_LOSS_DIAGNOSTIC_MISSING_OR_INVALID", minimum=0) for name in names}


def build_report(config, cells, replay_audit, *, config_sha256, contract_sha256, s4=None):
    """Create an allowlisted public summary, with no original donor/spot IDs."""
    require(set(cells) == expected_units(config), "COMPLETE_MATRIX_REQUIRED_NO_PARTIAL_MEANS")
    require(replay_audit.get("passed_units") == 72 and replay_audit.get("passed_states") == 192,
            "ACTUAL_COMPLETE_REPLAY_REQUIRED_BEFORE_METRICS")
    donor_rows, diagnostic_rows = [], []
    for key, cell in sorted(cells.items()):
        arm, donor, seed = key
        states = validate_cell(cell, key, config_sha256=config_sha256,
            contract_sha256=contract_sha256, stage="pilot")
        final = "solver_final" if arm == "S_A0" else "epoch40"
        for label, state in states.items():
            train = task_a_summary(state.get("train", {}).get("task_A", {}), rows=cell["train_rows"])
            held = state.get("heldout", {})
            a = task_a_summary(held.get("task_A", {}), rows=cell["heldout_rows"])
            c = task_c_summary(held.get("task_C", {}), rows=cell["heldout_rows"])
            diagnostic_rows.append(dict(arm=arm, heldout_donor_code=donor, seed=seed, snapshot=label,
                fixed_primary_snapshot=label == final, train_task_A=train, heldout_task_A=a,
                losses=loss_diagnostics(state, convex=arm == "S_A0"),
                checkpoint_selection_performed=False))
            if label != final:
                for name in ("task_B_common", "task_B_native"):
                    require(held.get(name, {}).get("status") == "NOT_EVALUATED_FOR_TASKB",
                            "DIAGNOSTIC_CHECKPOINT_RETRIEVAL_SCOPE_CHANGED")
                continue
            common = task_b_summary(held.get("task_B_common", {}), native_applicable=True)
            native = task_b_summary(held.get("task_B_native", {}), native_applicable=arm in ("S_A3", "S_A4", "S_A5"))
            groups = stratified_summary(held, rows=cell["heldout_rows"])
            section_codes = {s["section_code"] for s in groups["sections"]}
            require({s["section_code"] for s in common["per_section"]} == section_codes
                    and (native["status"] != "COMPLETE" or
                         {s["section_code"] for s in native["per_section"]} == section_codes),
                    "TASK_B_AND_TASK_A_SECTION_IDENTITIES_CHANGED")
            donor_rows.append(dict(arm=arm, heldout_donor_code=donor, seed=seed, heldout_spots=cell["heldout_rows"],
                primary_snapshot=final, task_A=a, task_C=c, task_B_common=common, task_B_native=native,
                grouped_diagnostics=groups,
                optimized_parameters=cell["parameters"]["optimized"],
                registered_parameters=cell["parameters"]["registered"], optimizer_updates=cell["optimizer_updates"]))
    summary = []
    for arm in config["trainable_arms"]:
        for seed in config["seeds"]:
            rows = [r for r in donor_rows if r["arm"] == arm and r["seed"] == seed]
            require(len(rows) == 6, "SIX_DONORS_REQUIRED_FOR_ARM_SEED_SUMMARY")
            require(sum(r["heldout_spots"] for r in rows) == 13011
                    and len({s["section_code"] for r in rows for s in r["grouped_diagnostics"]["sections"]}) == 12,
                    "DONOR_FOLDS_MUST_COVER_13011_SPOTS_AND_12_DISTINCT_SECTIONS")
            item = dict(arm=arm, seed=seed, donors=6, heldout_spots=sum(r["heldout_spots"] for r in rows),
                task_A_primary_Pearson=mean_defined(r["task_A"]["primary_donor_four_program_Pearson"] for r in rows),
                task_A_native_MSE=mean_defined(r["task_A"]["native"]["Mean_MSE"]["mean"] for r in rows),
                task_A_train_Z_MSE=mean_defined(r["task_A"]["train_Z"]["Mean_MSE"]["mean"] for r in rows),
                task_A_train_Z_Huber_delta1=mean_defined(r["task_A"]["train_Z"]["Mean_Huber_delta1"]["mean"] for r in rows),
                task_C_primary_Pearson=mean_defined(r["task_C"]["strict_four_program_Pearson"] for r in rows),
                task_C_AUPRC=mean_defined(r["task_C"]["primary_four_program_AUPRC"] for r in rows))
            for mode in ("common", "native"):
                applicable = [r["task_B_"+mode] for r in rows if r["task_B_"+mode]["status"] == "COMPLETE"]
                item["task_B_"+mode] = dict(status="COMPLETE" if applicable else "NOT_APPLICABLE_NO_TRAINED_GENE_ENCODER")
                for direction in ("image_to_gene", "gene_to_image"):
                    if applicable:
                        item["task_B_"+mode][direction] = {name:mean_defined(r[direction][name] for r in applicable)
                            for name in ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC")}
            summary.append(item)
    counters = dict(logical_cells=72, convex_optimizer_fits=sum(c["convex_optimizer_fits"] for c in cells.values()),
        neural_optimizer_fits=sum(c["neural_optimizer_fits"] for c in cells.values()),
        actual_saved_states=sum(len(c["snapshots"]) for c in cells.values()),
        actual_saved_state_replays=replay_audit["passed_states"],
        optimizer_updates=sum(c["optimizer_updates"] for c in cells.values()),
        sum_fit_elapsed_seconds=sum(c["elapsed_seconds"] for c in cells.values()),
        maximum_individual_fit_peak_GPU_bytes=max(c["peak_GPU_bytes"] for c in cells.values()),
        sum_fit_elapsed_is_not_parallel_wall_time=True)
    require(counters["optimizer_updates"] == config["budget"]["neural_optimizer_updates"] == 52000,
            "ACTUAL_PILOT_UPDATE_TOTAL_CHANGED")
    return dict(schema="spgel_source_metadata_report_v1", status="COMPLETE_REGISTERED_SOURCE_PILOT_AND_REPLAY",
        config_sha256=config_sha256, contract_sha256=contract_sha256, primary_candidate="S_A5",
        program_order=config["program_order"], counters=counters, replay=replay_audit,
        aggregation="strict_four_program_macro_within_heldout_donor_then_equal_donor_mean_per_seed",
        task_A_scale="native_signed_log1pCP10K_program_and_current_source_train_Z_separate",
        task_B_is_paired_not_image_only=True, train_metrics_are_diagnostic_not_independent=True,
        undefined_metrics_are_not_zero=True, source_development_donors_historically_reused=True,
        independent_source_confirmation="NOT_RUN", official_50HVG="NOT_RUN_SEPARATE_LOG1P_ONLY_CONTRACT",
        full_requested_research_program_complete=False, source_to_survival_claimed=False,
        donor_rows=donor_rows, arm_seed_summary=summary, fixed_checkpoint_diagnostics=diagnostic_rows,
        s4_export=s4 if s4 is not None else dict(status="NOT_REPORTED_SEPARATE_EXPORT_STAGE"))


def summarize_s4(config, cells, audit, *, config_sha256, contract_sha256):
    require(set(cells) == expected_units(config, stage="s4")
            and audit.get("passed_units") == 2 and audit.get("passed_states") == 6,
            "COMPLETE_S4_MATRIX_AND_REPLAY_REQUIRED")
    result = []
    for key, cell in sorted(cells.items()):
        states = validate_cell(cell, key, config_sha256=config_sha256,
                               contract_sha256=contract_sha256, stage="s4")
        train = task_a_summary(states["epoch40"].get("train", {}).get("task_A", {}), rows=13011)
        result.append(dict(arm=key[0], seed=key[2], train_rows=13011, heldout_rows=0,
            primary_epoch=40, artifact_sha256=cell["artifact_sha256"],
            actual_saved_states=len(states), optimizer_updates=cell["optimizer_updates"],
            train_task_A_diagnostic_only=train, independent_performance_evidence=False))
    return dict(status="COMPLETE_FIXED_S4_EXPORT_2_FITS_6_STATE_REPLAYS",
        neural_optimizer_fits=2, actual_saved_state_replays=6, replay=audit, models=result,
        source_checkpoint_selected_from_pilot_or_survival_scores=False,
        new_R8_version_required="R8_SOURCE_SPGEL_V1", performance_is_training_only=True)


def paired_comparisons(result):
    rows = {(r["arm"], r["heldout_donor_code"], r["seed"]):r for r in result["donor_rows"]}
    comparisons = []
    for reference in ("S_A0", "S_A1", "S_A2", "S_A3", "S_A4"):
        for seed in (31, 47):
            for task, field in (("task_A", "primary_donor_four_program_Pearson"),
                                 ("task_C", "primary_four_program_AUPRC")):
                per = []
                for donor in range(6):
                    candidate, control = rows[("S_A5", donor, seed)][task][field], rows[(reference, donor, seed)][task][field]
                    delta = None if candidate is None or control is None else candidate-control
                    per.append(dict(heldout_donor_code=donor, delta=delta))
                differences = [r["delta"] for r in per]
                comparisons.append(dict(candidate="S_A5", reference=reference, seed=seed, task=task, metric=field,
                    equal_donor_delta=mean_defined(differences), per_donor=per,
                    positive_donors=sum(v is not None and v > 0 for v in differences),
                    negative_donors=sum(v is not None and v < 0 for v in differences),
                    tied_donors=sum(v == 0 for v in differences if v is not None),
                    tests_or_independent_cohort_claimed=False))
    return comparisons


def _display(value):
    return "NA" if value is None else f"{value:.6f}"


def render_markdown(result):
    lines = ["# SPGEL：固定源开发 pilot 与实际检查点回放", "",
        "状态：72 个注册拟合及 192 份实际保存状态回放完整。六位源供体具有历史开发使用记录；本表不构成独立来源确认。",
        "", "主候选 S_A5、40 epoch 与程序定义预先固定；10/20 epoch 只作诊断，不进行留出成绩选模。",
        "", "官方 HEST 50 HVG：**NOT_RUN**。本表四程序 native log1pCP10K 与训练分区 Z 尺度分别报告，不替代官方 log1p-only 任务。",
        "", "## 每 seed 的等供体汇总", "",
        "TASK-A Pearson 先要求每供体四程序全部定义，再对供体等权。NA 保持缺失；定义数列防止把部分支持冒充六供体完整性能。",
        "", "| Arm | Seed | A Pearson | 定义供体 | A Z-MSE | A Z-Huber δ=1 | C AUPRC | 定义供体 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["arm_seed_summary"]:
        p, c = row["task_A_primary_Pearson"], row["task_C_AUPRC"]
        lines.append(f"| {row['arm']} | {row['seed']} | {_display(p['mean'])} | {p['defined_count']}/6 | "
            f"{_display(row['task_A_train_Z_MSE']['mean'])} | {_display(row['task_A_train_Z_Huber_delta1']['mean'])} | "
            f"{_display(c['mean'])} | {c['defined_count']}/6 |")
    lines += ["", "TASK-B 为同时输入图像与实测表达的配对检索；双向 Recall@1/5、median rank、pair AUC 另存于 JSON 与逐供体表，不能解释为纯图像预测。",
        "", "## 实际预算和回放", "", f"```json\n{json.dumps(result['counters'], ensure_ascii=False, indent=2)}\n```",
        "", "训练耗时求和不是并行墙钟时间；GPU 字节数为单拟合峰值最大值。S_A0 的解析程序均值表与凸优化拟合分开计数。",
        "", "## 机制解释边界", "",
        "优势是统一供体划分、固定训练预算、严格图像单支接口及实际保存状态回放，能够检查局部程序可预测性与有限结构差异。",
        "风险是供体数少且已复用、bulk/spot 语义不可自动互换，程序值并非细胞来源真值。当前 pilot 不含完整错配重训消融、官方强方法复现及新研究独立确认，不能据此宣称细胞机制、因果对应或 SOTA。",
        "", "S4 状态：`"+result["s4_export"]["status"]+"`。全源导出的训练指标不混入 LODO 主表。", ""]
    return "\n".join(lines)


def _write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def write_report(output, result):
    """Write allowlisted public artifacts only after all validation succeeded."""
    require(result.get("status") == "COMPLETE_REGISTERED_SOURCE_PILOT_AND_REPLAY"
            and result.get("counters", {}).get("logical_cells") == 72,
            "NO_PARTIAL_REPORT_PUBLICATION")
    output = Path(output)
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "REFUSE_OVERWRITE_EXISTING_REPORT_NAMESPACE")
    result = dict(result, paired_comparisons=paired_comparisons(result))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    markdown = render_markdown(result)
    donor_csv, program_csv = [], []
    for row in result["donor_rows"]:
        base = dict(arm=row["arm"], seed=row["seed"], heldout_donor_code=row["heldout_donor_code"],
                    heldout_spots=row["heldout_spots"])
        donor_csv.append(dict(**base,
            task_A_Pearson=row["task_A"]["primary_donor_four_program_Pearson"],
            task_A_defined_programs=row["task_A"]["native"]["defined_programs"],
            task_A_native_MSE=row["task_A"]["native"]["Mean_MSE"]["mean"],
            task_A_train_Z_MSE=row["task_A"]["train_Z"]["Mean_MSE"]["mean"],
            task_A_train_Z_Huber_delta1=row["task_A"]["train_Z"]["Mean_Huber_delta1"]["mean"],
            task_C_Pearson=row["task_C"]["strict_four_program_Pearson"],
            task_C_AUPRC=row["task_C"]["primary_four_program_AUPRC"],
            task_C_defined_AUPRC_programs=row["task_C"]["AUPRC"]["defined_count"],
            optimized_parameters=row["optimized_parameters"], registered_parameters=row["registered_parameters"],
            optimizer_updates=row["optimizer_updates"]))
        for mode in ("common", "native"):
            task = row["task_B_"+mode]
            donor_csv[-1]["task_B_"+mode+"_status"] = task["status"]
            for direction in ("image_to_gene", "gene_to_image"):
                for name in ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC", "queries_count"):
                    donor_csv[-1]["task_B_"+mode+"_"+direction+"_"+name] = task.get(direction, {}).get(name)
        for scale in ("native", "train_Z"):
            for program in row["task_A"][scale]["per_output"]:
                program_csv.append(dict(**base, task="TASK_A", scale=scale,
                    program=result["program_order"][program["column"]], **program))
    output.mkdir(parents=True, exist_ok=True)
    with (output/"S3_PILOT_SUMMARY.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded+"\n")
    with (output/"S3_RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown)
    _write_csv(output/"S3_PER_DONOR.csv", list(donor_csv[0]), donor_csv)
    _write_csv(output/"S3_PROGRAM_METRICS.csv", list(program_csv[0]), program_csv)
    decision = dict(schema="spgel_source_development_decision_v1",
        status="PILOT_COMPLETE_OFFICIAL_AND_INDEPENDENT_CONFIRMATION_PENDING",
        primary_candidate="S_A5", selected_using_heldout_or_survival=False,
        pilot_optimizer_fits=72, actual_pilot_state_replays=192,
        S4_status=result["s4_export"]["status"],
        independent_source_confirmation="NOT_RUN", official_HVG_benchmark="NOT_RUN",
        model_expansion_or_new_test_access_authorized_by_report=False,
        paired_comparisons=result["paired_comparisons"])
    with (output/"SPGEL_DECISION.json").open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, ensure_ascii=False, indent=2, allow_nan=False);stream.write("\n")
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-s4", action="store_true")
    args = parser.parse_args(argv)
    config = validate_config(read_metadata(args.config));config_sha = file_sha256(args.config)
    lock = read_metadata(args.private/"SOURCE_RUN_LOCK.json")
    cells, audit = load_validated_matrix(args.private, config, config_sha, lock)
    s4 = None
    if args.include_s4:
        s4_cells, s4_audit = load_validated_matrix(args.private, config, config_sha, lock, stage="s4")
        s4 = summarize_s4(config, s4_cells, s4_audit, config_sha256=config_sha,
                           contract_sha256=lock["contract_sha256"])
    result = build_report(config, cells, audit, config_sha256=config_sha,
                           contract_sha256=lock["contract_sha256"], s4=s4)
    decision = write_report(args.output, result)
    print(json.dumps(dict(status=decision["status"], pilot_optimizer_fits=72, actual_pilot_state_replays=192),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

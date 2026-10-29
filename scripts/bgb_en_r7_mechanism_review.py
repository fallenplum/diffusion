"""Postfit R7 anonymous metadata review; no arrays, models, labels or fitting.

The authoritative R7 decision is preserved. Gradients are already-computed
full-train vector gradients; routing/state probes cover only fixed first-eight
training rows. Norm arithmetic and frozen-output decompositions are descriptive,
not causal attribution, new validation, or histological truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ARMS = ("ADD", "RAW", "NORM", "INDEP", "SLOT", "GATE1", "GATE2", "SLOT_WRONG_RNA",
        "GATE1_WRONG_RNA", "GATE1_WRONG_PATH", "GATE2_WRONG_RNA")
PARTITIONS, SEEDS, EPOCHS = (0, 1, 2), (31, 47, 83), (20, 40, 60, 80)
GRAD_PATHS = ("cross_delta", "risk", "path_cross_delta", "gene_cross_delta", "total_delta_L2", "cross_delta_L2", "parameter_L2")
GROUPS = frozenset("core." + name for name in ("base_queries", "query_condition", "spatial_adapter", "path_main", "gene_main",
    "hallmark_main", "path_cross", "gene_cross", "gene_rereader", "slot_path_weight", "slot_gene_weight", "path_state_gate", "gene_state_gate")) | {"hallmark.rna"}
INTERFACES = {"raw_G0", "RNA_context", "base_query", "condition_query", "query_delta", "P0", "P_update",
              "reverse_query", "G_reverse_base", "G_update", "raw_patch_mean_L2", "null"}
ROUND_FIELDS = {"P_before", "P_after", "G_before", "G_after", "raw_delta_P", "raw_delta_G", "RNA_route", "PATH_route", "path_gate", "gene_gate"}
QUANTITIES = {"delta", "main_delta", "cross_delta", "path_main_delta", "gene_main_delta", "hallmark_main_delta", "path_cross_delta", "gene_cross_delta",
    "delta_E_L2", "delta_G_L2", "gate_path_mean", "gate_gene_mean", "gate_path_sd", "gate_gene_sd", "gate_path_round2_mean", "gate_gene_round2_mean",
    "gate_path_valid_programs", "gate_gene_valid_programs"}
METRICS = ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
CREDITS = ("repair", "damage", "Uno_repair", "Uno_damage")
MAX_BYTES = 2 * 1024**2
SCOPES = dict(FT_GRAD="complete current-training risk set; vector gradients accumulated over all microbatches",
    FT_RISK="all current training rows; saved per-patient risk summaries", IV_RISK="all current inner-validation rows; saved risk summaries",
    FT8_QUERY="fixed first at most eight current-training patients; NOT the full training cohort or independent cases",
    CALIBRATION="fixed untrained full-training RMS constants; no outcomes", IV_PAIR="epoch80 inner-validation direct paired-risk credits or frozen metrics")
LONG_COLUMNS = ("arm", "partition", "seed", "epoch", "scope", "section", "split", "control", "round", "path", "group", "metric", "program",
                "value", "sd", "n", "min", "max", "tensors", "numel", "nonzero", "valid_items")


def require(value, code):
    if not value:
        raise ValueError(code)


def read(path):
    path = Path(path)
    require(path.suffix.lower() == ".json", "R7_MECHANISM_JSON_METADATA_ONLY")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value, *, optional=False, nonnegative=False):
    if value is None and optional:
        return None
    require(type(value) in (int, float) and math.isfinite(value), "R7_MECHANISM_NONFINITE_OR_NONNUMERIC")
    require(not nonnegative or value >= 0, "R7_MECHANISM_NEGATIVE_NORM_OR_COUNT")
    return float(value)


def optional_mean(values):
    return statistics.mean(values) if values and all(value is not None for value in values) else None


def difference(a, b):
    return None if a is None or b is None else a - b


def metrics(value):
    result = {key: finite(value[key], optional=key in ("Harrell", "Uno60"), nonnegative=True) for key in METRICS}
    for metric, prefix in (("Harrell", "Harrell"), ("Uno60", "Uno")):
        n, d, score = result[prefix + "_num"], result[prefix + "_den"], result[metric]
        require(0 <= n <= d + 1e-8 and (score is None if d == 0 else score is not None and abs(score - n / d) <= 1e-10),
                "R7_MECHANISM_INVALID_CONCORDANCE_COMPONENTS")
    return result


def combine(rows):
    require(len(rows) == 3, "R7_MECHANISM_THREE_PARTITIONS_REQUIRED")
    answer = {key: math.fsum(row[key] for row in rows) for key in METRICS[2:]}
    answer.update(Harrell=answer["Harrell_num"] / answer["Harrell_den"] if answer["Harrell_den"] else None,
                  Uno60=answer["Uno_num"] / answer["Uno_den"] if answer["Uno_den"] else None)
    return answer


def credit_fields(value):
    answer = {}
    for key in CREDITS:
        number = finite(value[key], optional=True, nonnegative=True)
        require(number is None or number <= 1 + 1e-10, "R7_MECHANISM_INVALID_DIRECT_CREDIT")
        denominator = value["Uno_den" if key.startswith("Uno") else "Harrell_den"]
        require((number is None) == (denominator == 0), "R7_MECHANISM_CREDIT_DENOMINATOR_CHANGED")
        answer[key] = number
    return answer


def validate_replay(result, replay, replay_path):
    require(sha(replay_path) == result["replay_sha256"], "R7_MECHANISM_REPLAY_SHA_CHANGED")
    require(replay.get("status") == "PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY" and replay["contract_sha256"] == result["contract_sha256"]
        and replay["qualification_sha256"] == result["qualification_sha256"] and replay["expected_units"] == replay["available_units"] == replay["passed_units"] == 99
        and replay["failed_units"] == 0 and replay["expected_checkpoints"] == replay["passed_checkpoints"] == 297
        and not replay["missing_or_unreplayed_units"] and replay["new_optimizer_fits"] == replay["new_preprocessing_fits"] == 0,
        "R7_MECHANISM_COMPLETE_ACTUAL_REPLAY_REQUIRED")
    expected = {(a, p, s) for a in ARMS for p in PARTITIONS for s in SEEDS}
    require(len(replay["units"]) == 99 and {(r["arm"], r["partition"], r["seed"]) for r in replay["units"]} == expected,
        "R7_MECHANISM_REPLAY_MATRIX_CHANGED")
    for row in replay["units"]:
        require(row["status"] == "PASS" and row["actual_checkpoint_count"] == 3 and row["all_checkpoints_actual_forward_replayed"] is True
            and row["source_frozen_weight_hash_replayed"] is True and row["saved_query_mapping_recomputed"] is True
            and row["changed_pair_credits"] == row["new_optimizer_updates"] == 0, "R7_MECHANISM_UNIT_NOT_ACTUALLY_REPLAYED")
        require([r["epoch"] for r in row["checkpoints"]] == [20, 40, 80], "R7_MECHANISM_CHECKPOINT_INVENTORY_CHANGED")
        for checkpoint in row["checkpoints"]:
            error = finite(checkpoint["maximum_risk_error"], nonnegative=True)
            require(checkpoint["status"] == "PASS" and checkpoint["actual_state_dict_forward_replayed"] is True and error <= 2e-6
                    and checkpoint["new_optimizer_updates"] == checkpoint["changed_pair_credits"] == 0, "R7_MECHANISM_CHECKPOINT_REPLAY_FAILED")
    refs = replay["zero_fit_baseline_reference_replays"]
    require(len(refs) == 18 and {(r["arm"], r["partition"], r["seed"]) for r in refs} ==
            {(a, p, s) for a in ("B0", "B1") for p in PARTITIONS for s in SEEDS}
            and all(r["status"] == "PASS" and r["new_optimizer_fits"] == 0 for r in refs), "R7_MECHANISM_18_REFERENCE_REPLAYS_REQUIRED")


def load_cells(reports, result):
    index = result["diagnostic_index"]
    require(index["logical_cells"] == 99 and index["epochs_per_cell"] == list(EPOCHS), "R7_MECHANISM_DIAGNOSTIC_SCHEDULE_CHANGED")
    expected = {f"R7_DIAGNOSTICS_{arm}_p{part}.json": (arm, part) for arm in ARMS for part in PARTITIONS}
    require(len(index["parts"]) == 33 and {x["file"] for x in index["parts"]} == set(expected), "R7_MECHANISM_33_SHARDS_REQUIRED")
    cells, origins = {}, {}
    for part in index["parts"]:
        require(part["file"] in expected and part["rows"] == 3, "R7_MECHANISM_SHARD_NAME_OR_COUNT_CHANGED")
        path = (reports / part["file"]).resolve()
        require(path.parent == reports and path.stat().st_size == part["bytes"] <= MAX_BYTES and sha(path) == part["sha256"],
                "R7_MECHANISM_SHARD_SHA_SIZE_OR_PATH_CHANGED")
        shard = read(path); arm, partition = expected[part["file"]]
        require(shard["schema"] == "bgb_en_r7_diagnostic_shard_v1" and len(shard["rows"]) == 3
                and {(x["arm"], x["partition"], x["seed"]) for x in shard["rows"]} == {(arm, partition, seed) for seed in SEEDS},
                "R7_MECHANISM_SHARD_CELL_IDENTITY_CHANGED")
        for row in shard["rows"]:
            key = (row["arm"], row["partition"], row["seed"])
            require(key not in cells and set(row["checkpoint_diagnostics"]) == {str(e) for e in EPOCHS}, "R7_MECHANISM_MISSING_OR_DUPLICATE_DIAGNOSTIC")
            cells[key] = row
            origins[key] = dict(file=part["file"], shard_sha256=part["sha256"], anonymous_cell_canonical_sha256=canonical(row))
    require(len(cells) == 99, "R7_MECHANISM_INCOMPLETE_MATRIX")
    return cells, origins


def load_fold_metrics(path, result):
    require(path.suffix.lower() == ".csv", "R7_MECHANISM_FOLD_CSV_REQUIRED")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    table = {}
    for row in rows:
        key = (row["arm"], int(row["partition"]), int(row["seed"]))
        require(key not in table, "R7_MECHANISM_DUPLICATE_FOLD_METRIC")
        table[key] = metrics({name: None if row[name] == "" else float(row[name]) for name in METRICS})
    require(set(table) == {(a, p, s) for a in ("B0", "B1", *ARMS) for p in PARTITIONS for s in SEEDS}, "R7_MECHANISM_ALL_117_FOLD_ROWS_REQUIRED")
    for arm in ("B0", "B1", *ARMS):
        reference = result["primary_DSS"][arm]
        require(reference["complete_cells"] == 9, "R7_MECHANISM_INCOMPLETE_PRIMARY_SUMMARY")
        per_seed = []
        for seed in SEEDS:
            current = combine([table[arm, part, seed] for part in PARTITIONS]); per_seed.append(current)
            saved = next(x for x in reference["per_seed"] if x["seed"] == seed)
            for metric in METRICS:
                a, b = current[metric], saved[metric]
                require(a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-9,
                        "R7_MECHANISM_FOLD_CSV_DOES_NOT_MATCH_RESULTS")
        for metric in ("Harrell", "Uno60"):
            a, b = optional_mean([x[metric] for x in per_seed]), reference[metric]
            require(a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-10,
                    "R7_MECHANISM_PRIMARY_AGGREGATION_CHANGED")
    return table


def build(reports, output, replay_path=None):
    reports, output = Path(reports).resolve(), Path(output).resolve()
    require(not output.exists(), "R7_MECHANISM_NEW_OUTPUT_DIRECTORY_REQUIRED")
    source_path = reports / "R7_RESULTS.json"
    result = read(source_path)
    require(result.get("schema") == "bgb_en_r7_complete_report_v1" and result.get("status") == "COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS"
        and result["budget"]["formal_neural_fits"] == 99 and result["budget"]["formal_optimizer_updates"] == 7920
        and result["budget"]["source_refits"] == result["budget"]["teacher_or_beta_refits"] == 0,
        "R7_MECHANISM_COMPLETE_RESULTS_REQUIRED")
    replay_path = Path(replay_path).resolve() if replay_path is not None else reports / "R7_CHECKPOINT_REPLAY_AUDIT.json"
    require(replay_path.suffix.lower() == ".json", "R7_MECHANISM_REPLAY_JSON_ONLY")
    replay = read(replay_path); validate_replay(result, replay, replay_path)
    cells, origins = load_cells(reports, result)
    metric_path = reports / "R7_FOLD_METRICS.csv"
    metric_table = load_fold_metrics(metric_path, result)
    records, master, cell_index = {arm: [] for arm in ARMS}, [], []
    initial_scales = {}

    def emit(identity, *, scope, section, metric, value, **extra):
        value = finite(value, optional=True)
        row = dict(zip(("arm", "partition", "seed", "epoch"), identity)) | dict(scope=scope, section=section, metric=metric, value=value) | extra
        for key in ("sd", "n", "min", "max", "numel", "nonzero", "tensors", "valid_items"):
            if key in row:
                finite(row[key], optional=True, nonnegative=key != "min" and key != "max")
        records[identity[0]].append(row)

    def emit_stat(identity, value, **labels):
        require(isinstance(value, dict), "R7_MECHANISM_SAVED_SCALAR_SUMMARY_REQUIRED")
        n = value.get("n", value.get("patients_with_valid_items", value.get("patients")))
        require(type(n) is int and n >= 0, "R7_MECHANISM_SUMMARY_SUPPORT_MISSING")
        mean, sd = value.get("mean"), value.get("sd")
        require((n == 0) == (mean is None) and (n == 0) == (sd is None), "R7_MECHANISM_EMPTY_SUMMARY_MUST_BE_NONE")
        emit(identity, value=mean, sd=sd, n=n, min=value.get("min"), max=value.get("max"),
             valid_items=value.get("valid_items"), **labels)

    def risk_block(identity, block, split):
        require(isinstance(block, dict) and isinstance(block.get("quantities"), dict), "R7_MECHANISM_RISK_SUMMARIES_REQUIRED")
        scope = "FT_RISK" if split == "train" else "IV_RISK"
        require(set(block["quantities"]) <= QUANTITIES, "R7_MECHANISM_UNKNOWN_RISK_QUANTITY")
        for name, value in block["quantities"].items():
            emit_stat(identity, value, scope=scope, section="risk_quantity", split=split, metric=name)
        for name in ("SD_total_delta_over_offset", "SD_cross_delta_over_offset"):
            emit(identity, scope=scope, section="risk_ratio", split=split, metric=name, value=finite(block[name], nonnegative=True))
        for direction, value in block.get("slot_contributions", {}).items():
            require(direction in ("slot_path_contributions", "slot_gene_contributions") and [r["program"] for r in value["per_program"]] == list(range(4)),
                    "R7_MECHANISM_FOUR_PROGRAM_CONTRIBUTIONS_REQUIRED")
            for item in value["per_program"]:
                emit_stat(identity, item, scope=scope, section="program_risk", split=split, metric=direction, program=item["program"])
            for name in ("sum", "sum_absolute"):
                emit_stat(identity, value[name], scope=scope, section="program_risk", split=split, metric=direction + "." + name)
            retention = finite(value["absolute_sum_over_sum_absolute_mean"], optional=True, nonnegative=True)
            count = value["nonzero_absolute_sum_patients"]
            require(type(count) is int and count >= 0 and ((retention is None) == (count == 0))
                    and (retention is None or retention <= 1 + 1e-10), "R7_MECHANISM_CANCELLATION_NOT_DEFINED_FOR_ZERO_OUTPUT")
            emit(identity, scope=scope, section="program_cancellation", split=split, metric=direction + ".retained_absolute_fraction",
                 value=retention, n=count)
            emit(identity, scope=scope, section="program_cancellation", split=split, metric=direction + ".cancelled_absolute_fraction",
                 value=None if retention is None else 1 - retention, n=count)

    for key in sorted(cells):
        arm, part, seed = key
        for epoch in EPOCHS:
            identity = (*key, epoch); diagnostic = cells[key]["checkpoint_diagnostics"][str(epoch)]
            require(diagnostic["epoch"] == diagnostic["optimizer_updates_completed"] == epoch
                and diagnostic["validation_ranking_used_for_selection"] is False, "R7_MECHANISM_DIAGNOSTIC_EPOCH_OR_SELECTION_CHANGED")
            grad, probe = diagnostic["gradient_probe"], diagnostic["query_probe"]
            require(grad["scope"] == "complete_train_riskset_Cox_vector_accumulated_all_microbatches"
                and grad["new_optimizer_updates"] == 0 and grad["risk_penalties_excluded_from_pure_Cox_paths"] is True
                and grad["parameter_L2_uses_actual_slot_one_quarter_geometry"] is True
                and set(grad["groups_by_path"]) == set(GRAD_PATHS), "R7_MECHANISM_FULL_TRAIN_GRADIENT_SCOPE_REQUIRED")
            train_n = grad["patients"]
            require(type(train_n) is int and train_n > 0, "R7_MECHANISM_TRAIN_SCOPE_COUNT_INVALID")
            require(diagnostic["train"]["quantities"]["delta"]["n"] == train_n,
                    "R7_MECHANISM_FULL_TRAIN_SUPPORT_DOES_NOT_MATCH_RISK")
            for path, groups in grad["groups_by_path"].items():
                require(set(groups) <= GROUPS, "R7_MECHANISM_UNKNOWN_PARAMETER_GROUP")
                for group, value in groups.items():
                    norm = math.sqrt(finite(value["L2_squared"], nonnegative=True))
                    require(0 <= value["nonzero"] <= value["tensors"] and value["numel"] >= 0, "R7_MECHANISM_GRADIENT_SUPPORT_INVALID")
                    emit(identity, scope="FT_GRAD", section="gradient", path=path, group=group, metric="vector_gradient_L2", value=norm,
                         n=train_n, tensors=value["tensors"], numel=value["numel"], nonzero=value["nonzero"])
            emit(identity, scope="FT_GRAD", section="objective", metric="full_train_Cox_loss", value=grad["cox_loss"], n=train_n)
            emit(identity, scope="FT_GRAD", section="objective", metric="parameter_L2_value", value=grad["parameter_regularizer"], n=train_n)
            risk_block(identity, diagnostic["train"], "train")
            if epoch == 60:
                require(diagnostic["validation"] == "NOT_EVALUATED_AT_DIAGNOSTIC_ONLY_EPOCH", "R7_MECHANISM_EPOCH60_VALIDATION_NOT_ALLOWED")
            else:
                risk_block(identity, diagnostic["validation"], "validation")
            require(probe["scope"] == "fixed_first_eight_current_train_rows_no_validation_no_outcome_selection"
                and probe["patients"] == min(8, train_n) and probe["new_backward_calls"] == probe["new_optimizer_updates"] == 0,
                "R7_MECHANISM_FIRST8_QUERY_SCOPE_REQUIRED")
            expected_controls = {"matched_query", "wrong_query"} | (set() if arm in ("RAW", "ADD") else {"wrong_path"})
            require(set(probe["controls"]) == expected_controls and set(probe["matched_vs_controls"]) == expected_controls - {"matched_query"},
                    "R7_MECHANISM_QUERY_CONTROL_INVENTORY_CHANGED")
            for control, value in probe["controls"].items():
                require(set(value["interfaces"]) <= INTERFACES, "R7_MECHANISM_UNKNOWN_INTERFACE")
                for name, summary in value["interfaces"].items():
                    emit_stat(identity, summary, scope="FT8_QUERY", section="interface", control=control, metric=name)
                expected_rounds = 0 if arm in ("RAW", "ADD") else 2 if arm.startswith("GATE2") else 1
                require([r["round"] for r in value["rounds"]] == list(range(1, expected_rounds + 1)), "R7_MECHANISM_STATE_ROUND_INVENTORY_CHANGED")
                for step in value["rounds"]:
                    require(set(step) == ROUND_FIELDS | {"round"}, "R7_MECHANISM_UNKNOWN_STATE_TRACE_FIELD")
                    for name in ROUND_FIELDS:
                        emit_stat(identity, step[name], scope="FT8_QUERY", section="state", control=control, round=step["round"], metric=name)
            for control, value in probe["matched_vs_controls"].items():
                require(value["own_values_main_offset_exact"] is True and value["frozen_not_retrained"] is True,
                        "R7_MECHANISM_QUERY_INTERVENTION_REPLACED_OWN_VALUES")
                emit_stat(identity, value["patch_plus_null_TV"], scope="FT8_QUERY", section="query_TV", control=control, metric="patch_plus_null_TV")
                emit(identity, scope="FT8_QUERY", section="query_risk_difference", control=control, metric="matched_minus_control_risk_SD",
                     value=finite(value["risk_difference_SD"], nonnegative=True), n=probe["patients"])
            scales = probe["fixed_six_RMS"]
            require(set(scales) == {"context_rms", "patch_rms", "query_rms", "gene_key_rms", "evidence_rms", "reverse_query_rms"}
                and all(finite(v, nonnegative=True) >= 1e-6 for v in scales.values()), "R7_MECHANISM_INVALID_FIXED_RMS")
            shared_key = part, seed
            require(shared_key not in initial_scales or initial_scales[shared_key] == scales, "R7_MECHANISM_RMS_CHANGED_BETWEEN_EPOCHS_OR_ARMS")
            initial_scales[shared_key] = scales
            for name, value in scales.items():
                emit(identity, scope="CALIBRATION", section="calibration", metric=name, value=value)
            cell_index.append(dict(arm=arm, partition=part, seed=seed, epoch=epoch, full_train_gradient_patients=train_n,
                query_probe_patients=probe["patients"], epoch60_has_no_saved_model_replay=epoch == 60, **origins[key]))
            master.append(dict(arm=arm, partition=part, seed=seed, epoch=epoch, full_train_gradient_patients=train_n,
                query_probe_patients=probe["patients"], train_total_SD_ratio=diagnostic["train"]["SD_total_delta_over_offset"],
                train_cross_SD_ratio=diagnostic["train"]["SD_cross_delta_over_offset"],
                validation_total_SD_ratio=None if epoch == 60 else diagnostic["validation"]["SD_total_delta_over_offset"],
                validation_cross_SD_ratio=None if epoch == 60 else diagnostic["validation"]["SD_cross_delta_over_offset"]))

    paired_rows, frozen_rows = [], []
    expected_pairs = {(r["candidate"], r["reference"]) for r in result["paired_comparisons"]}
    raw_pairs = replay["paired_retrained_controls"]
    require(len(raw_pairs) == len(expected_pairs) * 9 and {(r["candidate"], r["reference"], r["partition"], r["seed"]) for r in raw_pairs} ==
            {(a, b, p, s) for a, b in expected_pairs for p in PARTITIONS for s in SEEDS}, "R7_MECHANISM_DIRECT_PAIRED_CREDIT_MATRIX_CHANGED")
    for row in raw_pairs:
        require(row["direct_risk_pair_credits"] is True and row["primary_checkpoint_epoch"] == 80, "R7_MECHANISM_DIRECT_RISK_CREDIT_REQUIRED")
        m = metrics(row["candidate_vs_reference"]); credits = credit_fields(row["candidate_vs_reference"])
        clean = {key: row[key] for key in ("candidate", "reference", "partition", "seed")}
        clean.update(epoch=80, source="actual_replay_direct_risk_pair_credits", **m, **credits)
        paired_rows.append(clean)
        for name in CREDITS:
            emit((row["candidate"], row["partition"], row["seed"], 80), scope="IV_PAIR", section="paired_retrained", metric=name,
                 control=row["reference"], value=credits[name])
    per_cell, per_seed, per_arm = [], [], []
    off_table = {}
    for key in sorted(cells):
        arm, part, seed = key; frozen = cells[key]["frozen_controls"]
        require("interaction_off" in frozen, "R7_MECHANISM_FROZEN_OFF_REQUIRED")
        for control, value in frozen.items():
            require(control in {"matched_query", "wrong_query", "wrong_path", "mean_query", "interaction_off", "main_off"}, "R7_MECHANISM_UNKNOWN_FROZEN_CONTROL")
            require(value.get("frozen_not_retrained") is True, "R7_MECHANISM_FROZEN_NOT_RETRAINED_FLAG_REQUIRED")
            direct = metrics(value["relative_to_trained"]); credits = credit_fields(value["relative_to_trained"])
            frozen_rows.append(dict(arm=arm, partition=part, seed=seed, epoch=80, control=control,
                source="saved_frozen_direct_risk_credits_authenticated_by_actual_replay", **direct, **credits))
            for name in CREDITS:
                emit((*key, 80), scope="IV_PAIR", section="frozen_credit", control=control, metric=name, value=credits[name])
        off = metrics(frozen["interaction_off"]["metrics_vs_base"]); off_table[key] = off
        full, add = metric_table[key], metric_table["ADD", part, seed]
        require(all(full[d] == off[d] == add[d] for d in ("Harrell_den", "Uno_den")), "R7_MECHANISM_DECOMPOSITION_SUPPORT_CHANGED")
        direct_off = frozen["interaction_off"]["relative_to_trained"]
        direct_credit = credit_fields(direct_off)
        for field in METRICS:
            a, b = full[field], direct_off[field]
            require(a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-9,
                    "R7_MECHANISM_FROZEN_CREDIT_DIRECTION_IS_NOT_TRAINED_MINUS_CONTROL")
        row = dict(arm=arm, partition=part, seed=seed, epoch=80, frozen_not_retrained=True,
                   direct_full_off_credit=direct_credit, off_add_direct_pair_credit=None)
        for name in ("Harrell", "Uno60"):
            total, cross, main = difference(full[name], add[name]), difference(full[name], off[name]), difference(off[name], add[name])
            require(total is None or abs(total - cross - main) <= 1e-10, "R7_MECHANISM_OUTPUT_MAIN_IDENTITY_FAILED")
            repair, damage = (direct_credit[k] for k in (("repair", "damage") if name == "Harrell" else ("Uno_repair", "Uno_damage")))
            require(cross is None and repair is None and damage is None or cross is not None and repair is not None
                    and damage is not None and abs(repair - damage - cross) <= 1e-10, "R7_MECHANISM_FROZEN_DIRECT_CREDITS_DISAGREE")
            row[name] = dict(Cfull=full[name], Coff=off[name], Cadd=add[name], delta_full_add=total, delta_full_off=cross, delta_off_add=main)
            for metric_name, value in row[name].items():
                emit((*key, 80), scope="IV_PAIR", section="main_output_decomposition", split="validation",
                     group=name, metric=metric_name, value=value)
        per_cell.append(row)
    for arm in ARMS:
        seeds = []
        for seed in SEEDS:
            full = combine([metric_table[arm, p, seed] for p in PARTITIONS])
            off = combine([off_table[arm, p, seed] for p in PARTITIONS])
            add = combine([metric_table["ADD", p, seed] for p in PARTITIONS])
            row = dict(arm=arm, seed=seed)
            row["direct_full_off_credit"] = {}
            for credit in CREDITS:
                denom = "Uno_den" if credit.startswith("Uno") else "Harrell_den"
                source = [cells[arm, p, seed]["frozen_controls"]["interaction_off"]["relative_to_trained"] for p in PARTITIONS]
                total_den = math.fsum(r[denom] for r in source)
                row["direct_full_off_credit"][credit] = math.fsum(r[credit] * r[denom] for r in source if r[denom] > 0) / total_den if total_den else None
            for name in ("Harrell", "Uno60"):
                row[name] = dict(Cfull=full[name], Coff=off[name], Cadd=add[name], delta_full_add=difference(full[name], add[name]),
                                 delta_full_off=difference(full[name], off[name]), delta_off_add=difference(off[name], add[name]))
                v = row[name]
                require(v["delta_full_add"] is None or abs(v["delta_full_add"] - v["delta_full_off"] - v["delta_off_add"]) <= 1e-10,
                        "R7_MECHANISM_SEED_OUTPUT_MAIN_IDENTITY_FAILED")
            per_seed.append(row); seeds.append(row)
        row = dict(arm=arm)
        row["direct_full_off_credit"] = {key: optional_mean([r["direct_full_off_credit"][key] for r in seeds]) for key in CREDITS}
        for name in ("Harrell", "Uno60"):
            row[name] = {key: optional_mean([r[name][key] for r in seeds]) for key in seeds[0][name]}
            v = row[name]
            require(v["delta_full_add"] is None or abs(v["delta_full_add"] - v["delta_full_off"] - v["delta_off_add"]) <= 1e-10,
                    "R7_MECHANISM_ARM_OUTPUT_MAIN_IDENTITY_FAILED")
        per_arm.append(row)
    summary_sections = {"gradient", "objective", "risk_ratio", "program_cancellation", "query_TV", "query_risk_difference", "state"}
    summaries = []
    for arm in ARMS:
        for epoch in EPOCHS:
            groups = {}
            for row in records[arm]:
                if row["epoch"] != epoch or row["section"] not in summary_sections:
                    continue
                dimensions = tuple(row.get(k) for k in ("scope", "section", "split", "control", "round", "path", "group", "metric", "program"))
                groups.setdefault(dimensions, []).append(row["value"])
            measurements = []
            for dimensions, values in sorted(groups.items(), key=lambda item: repr(item[0])):
                require(len(values) == 9, "R7_MECHANISM_ALL_NINE_CELLS_REQUIRED_FOR_SUMMARY")
                valid = [v for v in values if v is not None]
                item = dict(zip(("scope", "section", "split", "control", "round", "path", "group", "metric", "program"), dimensions))
                item = {k: v for k, v in item.items() if v is not None}
                item.update(available_cells=len(valid), mean=statistics.mean(valid) if valid else None,
                            median=statistics.median(valid) if valid else None, min=min(valid) if valid else None, max=max(valid) if valid else None)
                measurements.append(item)
            summaries.append(dict(arm=arm, epoch=epoch, cells=9, aggregation="descriptive_equal_nine_cell_summary_not_pooled_patients", measurements=measurements))
    input_hashes = dict(R7_RESULTS=sha(source_path), replay_metadata=sha(replay_path), R7_FOLD_METRICS=sha(metric_path))
    decomposition_by_cell = {(row["arm"], row["partition"], row["seed"]): row for row in per_cell}
    for row in master:
        decomposition = decomposition_by_cell[row["arm"], row["partition"], row["seed"]]
        for name in ("Harrell", "Uno60"):
            for field in ("Cfull", "Coff", "Cadd", "delta_full_add", "delta_full_off", "delta_off_add"):
                row[name + "_" + field] = decomposition[name][field] if row["epoch"] == 80 else None
        for field in CREDITS:
            row["direct_full_off_" + field] = decomposition["direct_full_off_credit"][field] if row["epoch"] == 80 else None
    review = dict(schema="bgb_en_r7_mechanism_review_v1", status="COMPLETE_METADATA_ONLY_99_CELLS_396_DIAGNOSTICS",
        script_sha256=sha(__file__), contract_sha256=result["contract_sha256"], input_hashes=input_hashes,
        primary_decision_canonical_sha256=canonical(result["decision"]), primary_decision_unchanged=True,
        model_cells=99, diagnostic_cell_epochs=396, diagnostic_shards=33, expected_measurement_cells=9,
        actual_saved_checkpoint_replays_referenced=297,
        epoch60_is_metadata_authenticated_diagnostic_not_saved_checkpoint_replay=True,
        new_model_forwards=0, new_backward_calls=0, new_optimizer_fits=0, arrays_weights_or_outcomes_read=False,
        scope_definitions=SCOPES, query_probe_unique_patients_unknown_without_private_indices=True,
        query_probe_unique_patient_upper_bound=24, query_probe_upper_bound_applies_to_target_rows_not_foreign_context_donors=True,
        repeated_cell_epoch_probes_are_not_independent_patients=True,
        arm_epoch_summaries=summaries, cell_epoch_index=cell_index,
        paired_retrained_credit_rows=paired_rows, frozen_credit_rows=frozen_rows,
        main_output_decomposition=dict(per_cell=per_cell, per_seed=per_seed, per_arm=per_arm, identity_verified=True,
            aggregation="sum_within_partition_components_per_seed_then_equal_three_seed_mean",
            full_off_direct_credits_source="frozen_relative_to_trained_not_C_difference",
            off_add_direct_pair_credits="NOT_SAVED_NOT_RECONSTRUCTED", frozen_decomposition_is_not_retrained_causal_necessity=True),
        interpretations=dict(norms_cannot_be_subtracted_to_recover_gradient_vectors=True,
            Cox_path_gradients_share_full_model_risk_derivative_not_separate_component_Cox_models=True,
            full_train_Cox_penalty_and_L2_gradients_are_separate_paths=True, first8_query_probe_is_not_full_train=True,
            program_cancellation_is_mean_abs_sum_over_sum_abs_not_variance_or_performance=True,
            zero_program_output_cancellation_is_undefined_not_one=True, magnitude_and_L2_signs_do_not_prove_optimization_cause=True,
            independent_reverse_is_not_statistically_pure_interaction=True, gates_null_and_queries_are_not_histological_truth=True,
            frozen_output_dependence_does_not_establish_retraining_necessity=True,
            repair_and_damage_are_normalized_concordance_credits_not_unique_patient_pair_counts=True,
            full_gain_explained_by_frozen_off_does_not_show_prediction_depends_on_cross_output=True,
            current_implementation_only_not_all_biological_ideas=True, no_new_independent_validation=True),
        diagnostic_source_index=result["diagnostic_index"])
    payloads, csv_artifacts = [], []

    def add_csv(name, rows, columns):
        stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        data = stream.getvalue().encode("utf-8-sig")
        require(len(data) <= MAX_BYTES, "R7_MECHANISM_CSV_EXCEEDS_2MIB")
        payloads.append((name, data)); csv_artifacts.append(dict(file=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=len(rows)))
    add_csv("R7_MECHANISM_REVIEW.csv", master, list(master[0]))
    for arm in ARMS:
        add_csv(f"R7_MECHANISM_REVIEW_{arm}.csv", records[arm], LONG_COLUMNS)
    review["csv_artifacts"] = csv_artifacts
    data = json.dumps(review, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    require(len(data) <= MAX_BYTES, "R7_MECHANISM_JSON_EXCEEDS_2MIB")
    payloads.append(("R7_MECHANISM_REVIEW.json", data))
    lines = ["# R7 训练后机制诊断复核", "", "状态：**COMPLETE_METADATA_ONLY_99_CELLS_396_DIAGNOSTICS**。仅使用已认证匿名元数据，0新前向、0反向、0拟合。", "",
        "99模型的20/40/80共297份保存检查点已实际回放；60轮为完整训练中保存的诊断元数据，未单独保存并回放该轮模型。33分片均通过文件大小与SHA校验。", "",
        "## 作用域", "", "- 梯度：完整当前训练风险集、全部微批的向量累计；四个Cox路径共享完整模型的风险导数，并非分别对各风险分量拟合Cox。风险惩罚两路径与真实参数L2另列，不能把范数相减当向量差或因果归因。",
        "- Query/状态：每单元固定首8训练患者。跨模型、seed、epoch是重复评估；匿名数据只能给最多24名目标探针患者上界，不能报告精确独立人数。此上界不统计额外的外来查询上下文供者。",
        "- 风险和四程序抵消：区分全训练与内层验证。保留逐fold/seed/epoch长表；绝对保留比例为mean(|sum_k r_k|/sum_k|r_k|)，全零时未定义。",
        "- epoch60不使用内层验证；不选择epoch、模型、缩放或新超参数。", "", "## 风险输出与共同主效应优化", "",
        "先在同seed累加三个分区的concordance分子/分母，再平均三个seed，分别核对 Δfull-add = Δfull-off + Δoff-add。", "",
        "| 模型 | Harrell full−ADD | full−冻结off | 冻结off−ADD | Uno60 full−ADD | full−冻结off | 冻结off−ADD |", "|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "NA" if value is None else f"{value:+.9f}"
    for row in per_arm:
        values = [row[name][key] for name in ("Harrell", "Uno60") for key in ("delta_full_add", "delta_full_off", "delta_off_add")]
        lines.append("| " + row["arm"] + " | " + " | ".join(fmt(x) for x in values) + " |")
    lines += ["", "full/off的repair与damage来自保存并实际回放认证的直接风险配对信用；重训模型配对信用来自原回放元数据，不从C差重建。它们是归一化concordance信用，不是独立患者对计数。off−ADD没有直接风险配对收据，明确不补造。",
        "如果优势主要保留在冻结off中，不能说最终预测依赖有效cross风险输出；这仍不等于证明交互在训练中没有作用。冻结干预与真实重训的因果问题不同。", "",
        "## 限制", "", "幅度、参数L2与程序抵消均为当前实现的描述证据，不单独证明优化原因、数据不足或医学机制失败。null/gate/注意力不是组织学真值。此附录不更改R7主decision、不启动外层，也不是新的独立验证。",
        "原回放元数据以输入SHA绑定；若其未获准公开，复核直接配对需要数据持有人提供同SHA的原始JSON收据。", ""]
    payloads.append(("R7_MECHANISM_REVIEW.md", "\n".join(lines).encode("utf-8")))
    require(sha(source_path) == input_hashes["R7_RESULTS"] and sha(replay_path) == input_hashes["replay_metadata"]
            and sha(metric_path) == input_hashes["R7_FOLD_METRICS"]
            and all(sha(reports / part["file"]) == part["sha256"] for part in result["diagnostic_index"]["parts"]),
            "R7_MECHANISM_INPUT_CHANGED_DURING_REVIEW")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in payloads:
        with (output / name).open("xb") as stream:
            stream.write(content)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    review = build(args.reports, args.output, args.replay)
    print(json.dumps(dict(status=review["status"], model_cells=99, diagnostic_cell_epochs=396,
                          new_model_forwards=0, new_backward_calls=0, new_optimizer_fits=0)))


if __name__ == "__main__":
    main()

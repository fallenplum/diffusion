"""Anonymous R5 receipt reports: no model deserialization or patient arrays.

All metric means require the complete locked grid. Artifact bytes are hashed,
but model.pt, prepared teachers and raw measurements are never deserialized.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bgb_en.r5_protocol import (
    ALL_MAIN_ARMS, MAIN_ARMS, MECHANISM_CONTROLS, PARTITIONS, SEEDS,
    aggregate_metrics, canonical_sha256, decision_gate, paired_comparison,
    require, select_finalist, sha256, verify_protocol_lock,
)

METRICS = ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den",
           "repair", "damage", "Uno_repair", "Uno_damage", "risk_shift_sd_ratio")
IDENTITY = ("group", "arm", "report_arm", "training_control", "partition", "seed")
CONTROL_SET = set(MECHANISM_CONTROLS) | {"MATCHED"}
QUANTITY_NAMES = ("delta", "main_delta", "cross_delta", "sham_delta", "z_o", "z_p", "z",
                  "readout_derivative_z", "readout_derivative_zo", "readout_derivative_zp",
                  "summary_jacobian_L2")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def csv_write(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value, label):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value), "NONFINITE_OR_NONNUMERIC_" + label)
    return float(value)


def validate_metrics(metric):
    for field in ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den"):
        require(finite(metric[field], field) >= 0, "NEGATIVE_CONCORDANCE_COMPONENT")
    for field, num, den in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
        if metric[den] > 0:
            require(metric[field] is not None and abs(finite(metric[field], field) - metric[num] / metric[den]) <= 1e-10,
                    "METRIC_NUMERATOR_DENOMINATOR_MISMATCH")
            require(0 <= metric[field] <= 1, "CONCORDANCE_OUTSIDE_UNIT_INTERVAL")
        else:
            require(metric.get(field) is None, "ZERO_DENOMINATOR_METRIC_NOT_NULL")
    for name in ("repair", "damage", "Uno_repair", "Uno_damage", "risk_shift_sd_ratio"):
        if metric.get(name) is not None:
            require(finite(metric[name], name) >= 0, "NEGATIVE_RISK_DIAGNOSTIC")


def parameter_counts(value):
    if isinstance(value, dict):
        return value.get("registered"), value.get("requires_grad")
    return value, value


def old_report_snapshot(root):
    parent = root / "reports/bgb_en"
    return {str(path.relative_to(parent)).replace("\\", "/"): sha256(path)
            for path in parent.rglob("*") if path.is_file()}


def receipt_contract(value, contract_sha):
    declared = value.get("contract", {}).get("contract_sha256", value.get("contract_sha256"))
    require(declared == contract_sha, "RECEIPT_R5_CONTRACT_CHANGED")


def prepared_origin_guard(value, lock, partition):
    origin = value.get("contract", {}).get("prepared_origin", {})
    expected = lock["prepared_receipt_sha256"][f"outer1_inner{partition}"]
    require(origin.get("prepared_receipt_sha256") == expected, "PREPARED_REFERENCE_PARTITION_ORIGIN_CHANGED")


def identity(arm, control, partition, seed):
    return ("MAIN" if control == "MATCHED" else "MECHANISM", arm,
            arm if control == "MATCHED" else control, control, partition, seed)


def item_key(value):
    return tuple(value[field] for field in IDENTITY)


def load_receipts(private, lock):
    """Return only allowlisted receipt fields; never export exception messages."""
    private = Path(private)
    records, fits, references = [], [], []
    seen = set()
    for path in sorted((private / "cells").glob("*/COMPLETE.json")):
        value = read(path)
        require(value.get("status") == "COMPLETE_R5_FIT", "INVALID_R5_COMPLETE_STATUS")
        receipt_contract(value, lock["contract_sha256"])
        arm, control = value["arm"], value["training_control"]
        part, seed = int(value["partition"]), int(value["seed"])
        require(arm in MAIN_ARMS and control in CONTROL_SET and part in PARTITIONS and seed in SEEDS,
                "FIT_OUTSIDE_PRELOCKED_MATRIX")
        key = identity(arm, control, part, seed)
        require(value["group"] == key[0] and value["report_arm"] == key[2], "FIT_GROUP_OR_REPORT_ARM_MISMATCH")
        prepared_origin_guard(value, lock, part)
        for field, expected in (("arm", arm), ("partition", part), ("seed", seed), ("training_control", control)):
            require(value["contract"].get(field, expected) == expected, "FIT_CONTRACT_IDENTITY_CHANGED")
        require(path.parent.name == f"{arm}_{control}_p{part}_seed{seed}", "FIT_DIRECTORY_IDENTITY_MISMATCH")
        require(key not in seen, "DUPLICATE_R5_FIT_IDENTITY")
        require(not any((path.parent / name).exists() for name in ("FAILED.json", "ERROR_RECEIPT.json")),
                "CONFLICTING_R5_TERMINAL_MARKERS")
        require(value["epochs"] == value["optimizer_updates"] == 80
                and value["actual_new_optimizer_fits"] == 1, "R5_FIXED_BUDGET_CHANGED")
        require(sha256(path.parent / "model.pt") == value["artifact_sha256"], "SAVED_MODEL_PAYLOAD_SHA_CHANGED")
        validate_metrics(value["metrics"])
        for metric in value.get("metrics_vs", {}).values():
            validate_metrics(metric)
        if "CP_SCALAR" in value.get("metrics_vs", {}):
            require(all(value["metrics"].get(name) == value["metrics_vs"]["CP_SCALAR"].get(name) for name in METRICS),
                    "PRIMARY_METRIC_NOT_CP_SCALAR_REFERENCE")
        registered, active = parameter_counts(value.get("parameters"))
        require(registered is not None and active is not None and registered >= active >= 0,
                "PARAMETER_COUNT_CONTRACT_MISSING")
        for name in ("backbone_sha256", "full_state_sha256"):
            require(re.fullmatch("[0-9a-f]{64}", value["initialization"][name]) is not None,
                    "INITIALIZATION_FINGERPRINT_MISSING")
        seen.add(key)
        row = dict(zip(IDENTITY, key))
        row.update(status="COMPLETE", metrics=value["metrics"], active_parameters=active,
                   registered_parameters=registered, artifact_sha256=value["artifact_sha256"],
                   receipt_sha256=sha256(path), epochs=80, optimizer_updates=80,
                   actual_new_optimizer_fits=1, elapsed_seconds=finite(value["elapsed_seconds"], "elapsed_seconds"),
                   _receipt=value)
        fits.append(row)
    for path in sorted((private / "references").glob("*.json")):
        value = read(path)
        require(value.get("status") == "R5_BASELINE_REFERENCE", "INVALID_BASELINE_REFERENCE_STATUS")
        receipt_contract(value, lock["contract_sha256"])
        arm, part, seed = value.get("report_arm", value.get("arm")), int(value["partition"]), int(value["seed"])
        require(arm in ("A0", "A1") and part in PARTITIONS and seed in SEEDS,
                "BASELINE_REFERENCE_OUTSIDE_MATRIX")
        require(value["actual_new_optimizer_fits"] == 0 and bool(value["deterministic_seed_reference"]) == (seed != 31),
                "BASELINE_SEED_REFERENCE_MISCOUNTED_AS_NEW_FIT")
        key = identity(arm, "MATCHED", part, seed)
        prepared_origin_guard(value, lock, part)
        require(key not in seen, "DUPLICATE_BASELINE_REFERENCE")
        validate_metrics(value["metrics"])
        for metric in value.get("metrics_vs", {}).values():
            validate_metrics(metric)
        seen.add(key)
        registered, active = parameter_counts(value.get("parameters"))
        references.append(dict(zip(IDENTITY, key), status="COMPLETE", metrics=value["metrics"],
            active_parameters=active, registered_parameters=registered,
            actual_new_optimizer_fits=0, optimizer_updates=0, epochs=0, receipt_sha256=sha256(path),
            deterministic_seed_reference=bool(value["deterministic_seed_reference"]), _receipt=value))
    by_key = {item_key(row): row for row in fits + references}
    for arm in ALL_MAIN_ARMS:
        for part in PARTITIONS:
            for seed in SEEDS:
                key = identity(arm, "MATCHED", part, seed)
                if key not in by_key:
                    by_key[key] = missing_record(private, key, reference=arm in ("A0", "A1"))
    records = list(by_key.values())
    return records, fits, references


def missing_record(private, key, *, reference=False):
    row = dict(zip(IDENTITY, key), status="NOT_STARTED", epochs=None, optimizer_updates=None,
               actual_new_optimizer_fits=0, known_failed_optimizer_updates=None)
    if reference:
        row["status"] = "MISSING_REFERENCE"
        return row
    _, arm, _, control, part, seed = key
    cell = private / "cells" / f"{arm}_{control}_p{part}_seed{seed}"
    failures = [cell / name for name in ("FAILED.json", "ERROR_RECEIPT.json") if (cell / name).is_file()]
    require(len(failures) <= 1, "MULTIPLE_FAILURE_MARKERS_REQUIRE_PRIVATE_REVIEW")
    if failures:
        failure = read(failures[0])
        row.update(status="FAILED", actual_new_optimizer_fits=None, failure_receipt_sha256=sha256(failures[0]),
                   failure_type=str(failure.get("exception_type", failure.get("type", "UNSPECIFIED"))))
        require(re.fullmatch("[A-Za-z0-9_]+", row["failure_type"]) is not None, "UNSAFE_PUBLIC_FAILURE_TYPE")
        updates = failure.get("actual_optimizer_updates_completed")
        if isinstance(updates, int) and not isinstance(updates, bool) and updates >= 0:
            row["known_failed_optimizer_updates"] = updates
    elif (cell / "START.json").is_file():
        row.update(status="STARTED_NO_TERMINAL_UPDATE_COUNT_UNKNOWN", actual_new_optimizer_fits=None,
                   start_receipt_sha256=sha256(cell / "START.json"))
    return row


def metric_rows(records, endpoint="DSS"):
    result = []
    for row in records:
        metric = row.get("metrics") if endpoint == "DSS" else row.get("_receipt", {}).get("metrics_vs", {}).get("PFI_frozen_DSS_risk")
        value = {key: row.get(key) for key in IDENTITY}
        value.update(arm=row["report_arm"], status=row["status"], active_parameters=row.get("active_parameters"))
        if metric is not None:
            validate_metrics(metric)
            value["metrics"] = metric
        else:
            value["status"] = "METRICS_NOT_AVAILABLE"
        result.append(value)
    return result


def initialization_audit(fits):
    groups, controls = [], []
    main = {(r["arm"], r["partition"], r["seed"]): r for r in fits if r["group"] == "MAIN"}
    main_fair, mechanism_fair = True, True
    for part in PARTITIONS:
        for seed in SEEDS:
            rows = [main[(arm, part, seed)] for arm in MAIN_ARMS if (arm, part, seed) in main]
            hashes = {r["_receipt"]["initialization"]["backbone_sha256"] for r in rows}
            same = len(hashes) <= 1
            main_fair = main_fair and same
            groups.append(dict(partition=part, seed=seed, available_arms=len(rows), expected_arms=7,
                               shared_backbone_equal=same, complete_family=len(rows) == 7))
    for row in fits:
        if row["group"] != "MECHANISM":
            continue
        original = main.get((row["arm"], row["partition"], row["seed"]))
        same = original is not None and row["_receipt"]["initialization"]["full_state_sha256"] == original["_receipt"]["initialization"]["full_state_sha256"]
        mechanism_fair = mechanism_fair and same
        controls.append(dict(control=row["training_control"], partition=row["partition"], seed=row["seed"],
                             same_initial_full_state_as_matched=same))
    fair = main_fair and mechanism_fair
    return dict(status="PASS_AVAILABLE_SHARED_INITIALIZATION" if fair else "FAIL_SHARED_INITIALIZATION",
                fair=fair, main_fair=main_fair, mechanism_fair=mechanism_fair,
                main_families=groups, retrained_controls=controls,
                new_heads_not_required_to_share_full_state_across_different_modes=True)


def references_audit(references):
    by_key = {(r["arm"], r["partition"], r["seed"]): r for r in references}
    for arm in ("A0", "A1"):
        for part in PARTITIONS:
            available = [by_key[(arm, part, seed)] for seed in SEEDS if (arm, part, seed) in by_key]
            if len(available) == 3:
                for row in available[1:]:
                    require(row["metrics"] == available[0]["metrics"], "DETERMINISTIC_REFERENCE_CHANGED_ACROSS_SEEDS")
    return dict(logical_reference_rows=len(references), expected_logical_reference_rows=18,
                actual_new_optimizer_fits=0, unique_upstream_prepared_bundles=3,
                repeated_reference_rows_are_not_new_fits=True)


def summary_rows(summary, endpoint, *, group, source_arm=None):
    output = []
    for arm, entry in summary["arms"].items():
        row = dict(group=group, source_arm=source_arm, arm=arm, endpoint=endpoint,
                   status=entry["status"], completed_cells=entry["completed_cells"],
                   expected_cells=entry["expected_cells"], Harrell=entry["Harrell"], Uno60=entry["Uno60"],
                   active_parameters=entry.get("active_parameters"))
        if group == "MAIN":
            for reference in ("A1", "A2", "A3", "A8"):
                comparison = paired_comparison(summary, arm, reference)
                row["delta_H_vs_" + reference] = comparison["delta_Harrell"]
                row["delta_U_vs_" + reference] = comparison["delta_Uno60"]
                row["positive_H_seeds_vs_" + reference] = comparison.get("positive_Harrell_seeds")
                row["positive_U_seeds_vs_" + reference] = comparison.get("positive_Uno60_seeds")
        output.append(row)
    return output


def describe_mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.mean(values) if values else None


def diagnostic_tables(output, fits, replay, selected_arm=None):
    readout, attention, gradients, subgroup, curves, frozen = [], [], [], [], [], []
    for row in fits:
        base = {key: row[key] for key in IDENTITY}
        value = row["_receipt"]
        for population, diagnostic in value.get("risk_diagnostics", {}).items():
            if population not in ("train", "validation", "valid"):
                continue
            out = dict(base, population=population)
            for name in ("SD_total_delta_over_offset", "SD_cross_delta_over_offset",
                         "derivative_below_one_percent_of_initial_max", "scalar_z_derivative_not_defined_for_rank4",
                         "derivative_saturation_not_applicable_when_route_disabled"):
                out[name] = diagnostic.get(name)
            coefficients = value.get("readout_coefficients") or {}
            out.update(a_o=coefficients.get("a_o"), a_p=coefficients.get("a_p"))
            for quantity in QUANTITY_NAMES:
                for stat in ("mean", "sd", "min", "q10", "median", "q90", "max"):
                    out[quantity + "_" + stat] = diagnostic.get(quantity, {}).get(stat)
            readout.append(out)
        probe = value.get("attention_probe", {})
        for direction, stats in probe.get("directions", {}).items():
            attention.append(dict(base, direction=direction, probe_patients=probe.get("probe_patients"),
                scope="FIXED_FIRST_EIGHT_NOT_HISTOLOGY_TRUTH", **{k: stats.get(k) for k in (
                    "rows", "attention_TV", "top32_positive_support_Jaccard")}))
        probe = value.get("gradient_probe", {})
        for path, groups in probe.get("groups_by_path", {}).items():
            for module, stats in groups.items():
                gradients.append(dict(base, risk_path=path, parameter_group=module,
                    scope="FIRST_TRAIN_MICROBATCH_FULL_RISKSET_COX_CONTRIBUTION_NO_L2", **{k: stats.get(k) for k in (
                        "tensors", "nonzero", "L2_squared")}))
        for stats in value.get("subgroups", []):
            out = dict(base, subgroup=stats.get("group"))
            for name in ("patients", "events", "lower", "upper", "comparable_pairs", "Uno_pair_weight", "events_at_pair_left"):
                out[name] = stats.get(name)
            metric = stats.get("metrics", stats)
            out.update({name: metric.get(name) for name in METRICS})
            subgroup.append(out)
        for curve in value.get("curves", []):
            require(curve["epoch"] in (20, 40, 80), "UNLOCKED_LEARNING_CURVE_EPOCH")
            out = dict(base, epoch=curve["epoch"])
            for name in ("cox_loss", "parameter_regularizer", "total_risk_penalty", "cross_risk_penalty",
                         "gradient_norm", "risk_cache_replay_max", "cross_cache_replay_max"):
                out[name] = curve.get(name)
            for population, aliases in (("train", ("train",)), ("validation", ("validation", "valid"))):
                stats = next((curve[k] for k in aliases if k in curve), {})
                stats = stats.get("CP_SCALAR", stats)
                out.update({population + "_" + name: stats.get(name) for name in ("Harrell", "Uno60", "repair", "damage")})
            curves.append(out)
        for name, stats in value.get("frozen_controls", {}).items():
            for reference, field in (("CP_SCALAR", "metrics_vs_base"), ("TRAINED_SAME_MODEL", "relative_to_trained")):
                metric = stats.get(field, {})
                frozen.append(dict(base, frozen_control=name, reference=reference, frozen_not_retrained=True,
                    risk_SD_ratio=stats.get("risk_SD_ratio"), **{key: metric.get(key) for key in METRICS}))
    tables = (
        ("R5_READOUT_DIAGNOSTICS.csv", readout, ["population", "SD_total_delta_over_offset", "SD_cross_delta_over_offset", "derivative_below_one_percent_of_initial_max", "scalar_z_derivative_not_defined_for_rank4", "derivative_saturation_not_applicable_when_route_disabled", "a_o", "a_p"]
         + [q + "_" + s for q in QUANTITY_NAMES for s in ("mean", "sd", "min", "q10", "median", "q90", "max")]),
        ("R5_ATTENTION_DIAGNOSTICS.csv", attention, ["direction", "probe_patients", "scope", "rows", "attention_TV", "top32_positive_support_Jaccard"]),
        ("R5_GRADIENT_DIAGNOSTICS.csv", gradients, ["risk_path", "parameter_group", "scope", "tensors", "nonzero", "L2_squared"]),
        ("R5_SUBGROUPS.csv", subgroup, ["subgroup", "patients", "events", "lower", "upper", "comparable_pairs", "Uno_pair_weight", "events_at_pair_left", *METRICS]),
        ("R5_LEARNING_CURVES.csv", curves, ["epoch", "cox_loss", "parameter_regularizer", "total_risk_penalty", "cross_risk_penalty", "gradient_norm", "risk_cache_replay_max", "cross_cache_replay_max"]
         + [population + "_" + name for population in ("train", "validation") for name in ("Harrell", "Uno60", "repair", "damage")]),
        ("R5_FROZEN_CONTROLS.csv", frozen, ["frozen_control", "reference", "frozen_not_retrained", "risk_SD_ratio", *METRICS]),
    )
    for filename, rows, columns in tables:
        csv_write(output / filename, rows, [*IDENTITY, *columns])
    direct, main_direct = [], []
    direct_keys = set()
    for stats in replay.get("paired_retrained_controls", []):
        control = stats.get("control", stats.get("training_control"))
        require(stats.get("selected_arm", selected_arm) == selected_arm, "DIRECT_PAIR_SELECTED_ARM_CHANGED")
        require(control in MECHANISM_CONTROLS and int(stats["partition"]) in PARTITIONS and int(stats["seed"]) in SEEDS,
                "UNRECOGNIZED_DIRECT_RETRAINED_PAIR_IDENTITY")
        metric = stats.get("metrics", stats.get("matched_vs_control", stats))
        validate_metrics(metric)
        key = (control, int(stats["partition"]), int(stats["seed"]))
        require(key not in direct_keys, "DUPLICATE_DIRECT_RETRAINED_PAIR")
        direct_keys.add(key)
        direct.append(dict(control=control, partition=int(stats["partition"]), seed=int(stats["seed"]),
                           risk_SD_ratio=stats.get("risk_SD_ratio"), **{name: metric.get(name) for name in METRICS}))
    csv_write(output / "R5_DIRECT_RETRAINED_PAIRS.csv", direct, ["control", "partition", "seed", "risk_SD_ratio", *METRICS])
    direct_keys = set()
    for stats in replay.get("paired_main_controls", []):
        reference = stats["reference"]
        require(stats.get("selected_arm", selected_arm) == selected_arm, "DIRECT_MAIN_SELECTED_ARM_CHANGED")
        require(reference in ("A1", "A2", "A3", "A8") and int(stats["partition"]) in PARTITIONS
                and int(stats["seed"]) in SEEDS, "UNRECOGNIZED_DIRECT_MAIN_PAIR_IDENTITY")
        key = (reference, int(stats["partition"]), int(stats["seed"]))
        require(key not in direct_keys, "DUPLICATE_DIRECT_MAIN_PAIR")
        direct_keys.add(key)
        metric = stats.get("metrics", stats.get("matched_vs_control", stats))
        validate_metrics(metric)
        main_direct.append(dict(reference=reference, partition=int(stats["partition"]), seed=int(stats["seed"]),
            risk_SD_ratio=stats.get("risk_SD_ratio"), **{name: metric.get(name) for name in METRICS}))
    csv_write(output / "R5_DIRECT_MAIN_PAIRS.csv", main_direct, ["reference", "partition", "seed", "risk_SD_ratio", *METRICS])
    return dict(readout_rows=len(readout), attention_rows=len(attention), gradient_rows=len(gradients),
                subgroup_rows=len(subgroup), curve_rows=len(curves), frozen_rows=len(frozen),
                direct_retrained_pair_rows=len(direct),
                direct_main_pair_rows=len(main_direct),
                direct_retrained_pair_credit_status="AVAILABLE_FROM_INDEPENDENT_REPLAY" if len(direct) == 45
                else "NOT_AVAILABLE_COMPLETE_DIRECT_RETRAINED_PAIR_CREDIT")


def run(args, *, root=ROOT):
    root, private, output = Path(root).resolve(), args.private.resolve(), args.output.resolve()
    require(not output.is_relative_to(root / "reports/bgb_en"), "R5_CANNOT_WRITE_OLD_FORMAL_REPORTS")
    for folder in ("src", "scripts", "configs", "tests"):
        require(not output.is_relative_to(root / folder), "R5_REPORT_OUTPUT_IS_A_SOURCE_DIRECTORY")
    before = old_report_snapshot(root)
    lock = read(private / "LOCK.json")
    verify_protocol_lock(lock, root)
    records, fits, references = load_receipts(private, lock)
    ref_guard, init_guard = references_audit(references), initialization_audit(fits)
    main_records = [row for row in records if row["group"] == "MAIN"]
    main = aggregate_metrics(metric_rows(main_records))
    main_pfi = aggregate_metrics(metric_rows(main_records, "PFI"))
    proposed = select_finalist(main) if init_guard["main_fair"] and all(main["arms"][arm]["status"] == "COMPLETE" for arm in MAIN_ARMS) else None
    selection_path = private / "SELECTION.json"
    selection = read(selection_path) if selection_path.exists() else None
    selection_valid = False
    selected_arm = selection.get("selected_arm") if selection else None
    if selection:
        require(proposed is not None and selected_arm == proposed["selected_arm"], "SELECTION_NOT_FROM_COMPLETE_LOCKED_MAIN")
        if "contract_sha256" in selection:
            require(selection["contract_sha256"] == lock["contract_sha256"], "SELECTION_R5_CONTRACT_CHANGED")
        if "selection_summary_sha256" in selection:
            require(selection["selection_summary_sha256"] == proposed["selection_summary_sha256"], "SELECTION_SUMMARY_CHANGED")
        selection_valid = True
    mechanism_records = [row for row in records if row["group"] == "MECHANISM"]
    require(not mechanism_records or selection_valid, "MECHANISM_FITS_PRECEDE_SELECTION_LOCK")
    require(all(row["arm"] == selected_arm for row in mechanism_records), "MECHANISM_ARM_DIFFERS_FROM_SELECTION")
    mechanism_by_key = {item_key(row): row for row in mechanism_records}
    for control in MECHANISM_CONTROLS:
        for part in PARTITIONS:
            for seed in SEEDS:
                key = identity(selected_arm or "UNSELECTED", control, part, seed)
                if key not in mechanism_by_key:
                    mechanism_by_key[key] = missing_record(private, key)
    mechanism_records = list(mechanism_by_key.values())
    mechanism = aggregate_metrics(metric_rows(mechanism_records), arms=MECHANISM_CONTROLS)
    mechanism_pfi = aggregate_metrics(metric_rows(mechanism_records, "PFI"), arms=MECHANISM_CONTROLS)
    qualification = read(private / "QUALIFICATION.json") if (private / "QUALIFICATION.json").is_file() else {}
    qual_pass = qualification.get("status") == "PASS_R5_REAL_DATA_AND_LEGACY_PARITY" and qualification.get("contract_sha256") == lock["contract_sha256"]
    replay = read(private / "REPLAY_AUDIT.json") if (private / "REPLAY_AUDIT.json").is_file() else {}
    replay_pass = (replay.get("status") == "PASS_COMPLETE_R5_CHECKPOINT_REPLAY" and replay.get("contract_sha256") == lock["contract_sha256"]
                   and replay.get("passed_units") == 108 and replay.get("failed_units") == 0
                   and replay.get("main_passed") == 63 and replay.get("mechanism_passed") == 45)
    if args.stage == "final" and selection_valid:
        decision = decision_gate(main, mechanism, selected_arm,
                                 qualification_passed=qual_pass and init_guard["fair"], replay_passed=replay_pass)
    else:
        decision = dict(decision="INCONCLUSIVE", R6_formal_start_eligible=False,
                        selected_arm=selected_arm, hypotheses={h: "INCONCLUSIVE" for h in ("H5.1", "H5.2", "H5.3")},
                        reason="Main-stage report or immutable selection is not yet available")
    output.mkdir(parents=True, exist_ok=True)
    records = main_records + mechanism_records
    ledger = [{key: value for key, value in row.items() if not key.startswith("_") and key != "metrics"} for row in records]
    ledger_fields = [*IDENTITY, "status", "epochs", "optimizer_updates", "actual_new_optimizer_fits", "known_failed_optimizer_updates",
                     "elapsed_seconds", "registered_parameters", "active_parameters", "deterministic_seed_reference", "failure_type",
                     "artifact_sha256", "receipt_sha256", "failure_receipt_sha256", "start_receipt_sha256"]
    csv_write(output / "RUN_LEDGER.csv", ledger, ledger_fields)
    ledger_report = dict(schema="bgb_en_r5_run_ledger_v1", status="COMPLETE_108_NEURAL_FITS" if len(fits) == 108 else "INCOMPLETE",
        planned_main_neural_fits=63, completed_main_neural_fits=sum(row["group"] == "MAIN" for row in fits),
        planned_mechanism_neural_fits=45, completed_mechanism_neural_fits=sum(row["group"] == "MECHANISM" for row in fits),
        complete_new_supervised_optimizer_fits=len(fits), complete_optimizer_updates=80 * len(fits),
        failed_current_units=sum(row["status"] == "FAILED" for row in records),
        started_without_terminal=sum(row["status"] == "STARTED_NO_TERMINAL_UPDATE_COUNT_UNKNOWN" for row in records),
        known_failed_updates=sum(row.get("known_failed_optimizer_updates") or 0 for row in records),
        unknown_failed_or_interrupted_update_counts=sum(row["status"] in ("FAILED", "STARTED_NO_TERMINAL_UPDATE_COUNT_UNKNOWN")
            and row.get("known_failed_optimizer_updates") is None for row in records),
        sum_complete_unit_elapsed_seconds=sum(row.get("elapsed_seconds", 0) for row in fits),
        parallel_makespan_not_inferred_from_unit_time_sum=True, reference_accounting=ref_guard,
        teacher_refits_requested_by_report=0, history_outside_current_cells_not_enumerated=True, units=ledger)
    write(output / "RUN_LEDGER.json", ledger_report)
    columns = ["group", "source_arm", "arm", "endpoint", "status", "completed_cells", "expected_cells", "Harrell", "Uno60", "active_parameters"]
    columns += [name + ref for ref in ("A1", "A2", "A3", "A8") for name in (
        "delta_H_vs_", "delta_U_vs_", "positive_H_seeds_vs_", "positive_U_seeds_vs_")]
    csv_write(output / "R5_MAIN.csv", summary_rows(main, "DSS", group="MAIN") + summary_rows(main_pfi, "PFI_FROZEN_DSS_RISK", group="MAIN"), columns)
    mech_rows = summary_rows(mechanism, "DSS", group="MECHANISM", source_arm=selected_arm)
    mech_rows += summary_rows(mechanism_pfi, "PFI_FROZEN_DSS_RISK", group="MECHANISM", source_arm=selected_arm)
    for row in mech_rows:
        source = main if row["endpoint"] == "DSS" else main_pfi
        if selected_arm and source["arms"][selected_arm]["status"] == row["status"] == "COMPLETE":
            row["matched_minus_control_H"] = source["arms"][selected_arm]["Harrell"] - row["Harrell"]
            row["matched_minus_control_U"] = source["arms"][selected_arm]["Uno60"] - row["Uno60"]
    csv_write(output / "R5_MECHANISM.csv", mech_rows, columns[:10] + ["matched_minus_control_H", "matched_minus_control_U"])
    fold_rows = []
    for row in records:
        for endpoint in ("DSS", "PFI"):
            metric = row.get("metrics", {}) if endpoint == "DSS" else row.get("_receipt", {}).get("metrics_vs", {}).get("PFI_frozen_DSS_risk", {})
            fold_rows.append(dict({key: row.get(key) for key in IDENTITY}, endpoint=endpoint,
                status=row["status"] if metric else "METRICS_NOT_AVAILABLE", frozen_DSS_risk=(endpoint == "PFI"),
                actual_new_optimizer_fits=row.get("actual_new_optimizer_fits"),
                **{key: metric.get(key) for key in METRICS}))
    csv_write(output / "R5_SEED_FOLD.csv", fold_rows, [*IDENTITY, "endpoint", "status", "frozen_DSS_risk", "actual_new_optimizer_fits", *METRICS])
    diagnostic = diagnostic_tables(output, fits, replay, selected_arm)
    after = old_report_snapshot(root)
    require(before == after, "OLD_FORMAL_REPORTS_CHANGED_DURING_R5_REPORT")
    audit = dict(schema="bgb_en_r5_report_audit_v1", stage=args.stage,
        status="COMPLETE_REQUIRED_R5_TRAINING_AND_REPLAY" if decision["decision"] != "INCONCLUSIVE" else "INCOMPLETE_OR_MAIN_ONLY",
        contract_sha256=lock["contract_sha256"], main_summary=main, main_PFI_summary=main_pfi,
        mechanism_summary=mechanism, mechanism_PFI_summary=mechanism_pfi, proposed_selection=proposed,
        selection_qualified=selection_valid, qualification_passed=qual_pass, replay_passed=replay_pass,
        initializations=init_guard, diagnostics=diagnostic, model_deserializations=0,
        optimizer_updates_by_report=0, private_arrays_or_identifiers_exported=False)
    write(output / "R5_REPORT_AUDIT.json", audit)
    write(output / "decision.json", decision)
    write(output / "SOURCE_AUDIT.json", dict(schema="bgb_en_r5_report_source_audit_v1",
        status="PASS_FROZEN_SOURCES_RECEIPTS_AND_AVAILABLE_INITIALIZATIONS" if init_guard["fair"] else "FAIL_INITIALIZATION_FAIRNESS",
        contract_sha256=lock["contract_sha256"], private_lock_file_sha256=sha256(private / "LOCK.json"), sources=lock["sources"],
        complete_fit_payloads_SHA_verified=len(fits), baseline_references_qualified=len(references),
        old_report_snapshot_sha256=canonical_sha256(before), old_formal_reports_unchanged_by_report=True,
        qualification_passed=qual_pass, replay_passed=replay_pass,
        model_deserializations=0, new_optimizer_fits=0, initialization_audit=init_guard))
    write_markdown(output, audit, decision, ledger_report)
    return audit, decision


def fmt(value):
    return "—" if value is None else f"{value:.6f}"


def write_markdown(output, audit, decision, ledger):
    lines = ["# R5 非饱和低容量读出实验", "", f"**状态：{audit['status']}；决策：{decision['decision']}。**", "",
             f"实际完成主矩阵 {ledger['completed_main_neural_fits']}/63、机制重训 {ledger['completed_mechanism_neural_fits']}/45。主终点DSS；PFI仅使用冻结DSS风险，不参与模型选择。", "",
             "每seed先汇总三个分区内的concordant/eligible分子分母，再平均三个seed；不拼不同分区风险排序。确定性A0/A1跨seed引用不是额外拟合或独立队列。", "",
             "| 模型 | 完整单元 | DSS Harrell | DSS Uno60 |", "|---|---:|---:|---:|"]
    for arm, entry in audit["main_summary"]["arms"].items():
        lines.append(f"| {arm} | {entry['completed_cells']}/{entry['expected_cells']} | {fmt(entry['Harrell'])} | {fmt(entry['Uno60'])} |")
    lines += ["", f"锁定候选：{decision.get('selected_arm') or '尚未锁定'}；R6正式训练启动资格：{decision['R6_formal_start_eligible']}。", "",
              "## 机制比较", "", "| 对照 | 完整单元 | DSS Harrell | DSS Uno60 |", "|---|---:|---:|---:|"]
    for arm, entry in audit["mechanism_summary"]["arms"].items():
        lines.append(f"| {arm} | {entry['completed_cells']}/{entry['expected_cells']} | {fmt(entry['Harrell'])} | {fmt(entry['Uno60'])} |")
    for name, value in decision.get("hypotheses", {}).items():
        lines.append(f"\n- {name}：{value}。")
    lines += ["", "## 解释与未完成边界", "",
              "- A3是R5共同惩罚目标下的legacy读出函数。所有正式臂含原0.1 total-delta L2及新0.1 cross-delta L2；不能宣称其优化轨迹等于历史F5。旧F5保真资格单独验证。",
              "- A7为纯rank4替换，初始化与线性A4匹配；只在真实初始batch检验与legacy接近，不宣称全域等价tanh。A8为解析预算匹配的单模态sham，active参数与A7差7，未假定同FLOPs。",
              "- 训练使用cross-fit教师风险，验证使用当前训练分区full-fit风险；两者分布差异另列，不能仅凭训练C低于验证C判断欠拟合。",
              "- 冻结控制描述已训模型依赖，不能替代45个真实重训控制。真实重训pair-credit只从独立回放直接字段报告，不由两组C或相对CP的修复/损伤推断。",
              f"- 实际重训直接配对字段状态：{audit['diagnostics']['direct_retrained_pair_credit_status']}。缺失时留空，不能拿冻结探针代替。",
              "- 单轮F5仅替换RNA query时，反向PtoO没有直接受到该干预；零TV是结构性可能性，不证明该方向不用组学。",
              "- RNA完整309/块缺失203是全cohort规模，分组表报告实际当前验证分区人数/事件/可比较对。小组和margin仅用于解释，不选择模型。",
              "- 小于0.003的差值只列弱方向性；不计算bootstrap、置信区间或p值。历史TCGA反复开发，不是新的外部确认。未证明饱和为唯一原因，也不因R5负结果否定全部空间监督或原基因重读想法。", ""]
    (output / "R5_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    text = ["# R5 读出与条件路径诊断", "", f"状态：{audit['status']}。仅使用匿名COMPLETE统计；没有反序列化模型、患者风险或原始数组。", "",
            "R5_READOUT_DIAGNOSTICS.csv分别记录训练/验证的风险SD比、原始方向量、导数及分位统计；统计量跨模型仅作描述，不是方差贡献分解。rank4没有唯一标量z导数，使用差值摘要Jacobian范数，不能拿它与标量导数比值当同量纲贡献。", "",
            "R5_ATTENTION_DIAGNOSTICS与GRADIENT_DIAGNOSTICS是固定首8人的有限探针；梯度来自全训练风险集Cox的首微批贡献，不混入参数L2，不是完整患者群的组织学真值。", "",
            "R5_LEARNING_CURVES保留20/40/80轮；正式成绩只取80轮，不从曲线选epoch。R5_SUBGROUPS保存实际验证人数/事件及分组可比较对；跨组与跨fold风险不直接拼接。", "",
            "R5_FROZEN_CONTROLS分别保留相对CP_SCALAR和相对同一已训模型的直接字段；R5_DIRECT_RETRAINED_PAIRS只来自独立replay导出的真实重训配对。", "",
            "## 证据覆盖", ""]
    text += [f"- {key}：{value}" for key, value in audit["diagnostics"].items()]
    text += ["", "优势：读出、幅度、查询、双向消融和容量分开记录；限制：COMPLETE均值不恢复未保存的逐患者交集，注意力或梯度变化本身不证明医学对应。", ""]
    (output / "R5_RISK_READOUT_DIAGNOSTICS.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--stage", choices=("main", "final"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bgb_en_r5")
    args = parser.parse_args()
    audit, decision = run(args)
    print(json.dumps(dict(status=audit["status"], stage=args.stage, decision=decision["decision"],
                          R6_formal_start_eligible=decision["R6_formal_start_eligible"],
                          model_deserializations=0, new_optimizer_fits=0)), flush=True)
    if not audit["initializations"]["fair"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

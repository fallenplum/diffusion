"""Post-completion P2 aggregation. This program never trains or forwards a model.

Private prediction vectors are read only after explicit completion authorization;
only constructed anonymous aggregates, byte hashes and source-relative names are
published. No bootstrap, confidence interval, p-value or cross-fold risk pairing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_p2_schedule import checked_phase, chosen_epoch
from src.semantic_b0b7.p2_survival import array_sha, evaluate_safe, risk_repair_damage, select_lambda

SEEDS = (31, 47, 83)
FOLDS = (1, 2, 3, 4, 5)
MAX_BYTES = 2 * 1024**2
MAX_CSV_ROWS = 20_000
MAX_JSON_ROWS = 1_000  # unchanged established publication scanner's list bound
METRICS = ("Harrell", "Uno60")
PRIMARY = ("B1", "B2_MATCHED", "B3", "B4", "B5", "B6a", "B7")
CORE = (("B5", "B0"), ("B5", "B1"), ("B5", "B2_MATCHED"),
        ("B5", "B2_NATIVE"), ("B5", "B4"),
        ("B5", "B5_CLINICAL_CAPACITY_CONTROL"), ("B5", "B6a"),
        ("B5", "B7"), ("B4", "B3"), ("B4", "B3_TEXTLESS"),
        ("B4", "B4_MAP_SHUFFLE"), ("B1", "B0"))


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(root, relative, expected_sha=None):
    base = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "UNSAFE_ARTIFACT_NAME")
    path = base / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base),
            "MISSING_OR_ESCAPING_PRIVATE_ARTIFACT")
    if expected_sha is not None:
        require(sha(path) == expected_sha, "PRIVATE_BYTES_CHANGED")
    return path


def load_verified_pickle(root, relative, expected_sha):
    # Only own-research artifacts whose receipt hash has already been checked.
    with safe_path(root, relative, expected_sha).open("rb") as f:
        return pickle.load(f)


def clean(value):
    """No ndarray escape hatch: publication objects must be constructed scalars."""
    if isinstance(value, np.ndarray):
        raise ValueError("RAW_ARRAY_CANNOT_ENTER_PUBLIC_REPORT")
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        require(math.isfinite(value), "NONFINITE_PUBLIC_NUMBER")
    require(value is None or isinstance(value, (str, bool, int, float)), "UNSUPPORTED_PUBLIC_OBJECT")
    return value


def public_bytes(value):
    data = (json.dumps(clean(value), ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")
    text = data.decode("utf-8")
    require(not re.search(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}", text), "PUBLIC_PATIENT_IDENTIFIER")
    require(not re.search(r"(?:[A-Za-z]:[\\/]|/root/|/home/|/mnt/|hf_[A-Za-z0-9]{12})", text),
            "PRIVATE_PATH_OR_CREDENTIAL_IN_PUBLIC_REPORT")
    require(len(data) <= MAX_BYTES, "PUBLIC_FILE_EXCEEDS_2MIB")
    return data


def write_json(path, value):
    payload = public_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(payload)
    return {"file": path.name, "sha256": sha(path), "bytes": len(payload)}


def write_markdown(path, text):
    public_bytes({"markdown": text})
    payload = text.encode("utf-8")
    require(len(payload) <= MAX_BYTES, "PUBLIC_FILE_EXCEEDS_2MIB")
    with Path(path).open("xb") as f:
        f.write(payload)
    return {"file": Path(path).name, "sha256": sha(path), "bytes": len(payload)}


def write_records(out, stem, records, *, csv_output=True):
    """Deterministic chunks, each below both publication limits."""
    records = [clean(x) for x in records]
    pieces, current, indented_bytes = [], [], 0
    for row in records:
        encoded = public_bytes(row)
        # indent=2 adds two spaces to every object line inside a list. Exact
        # byte counting avoids reserializing the growing ledger quadratically.
        row_bytes = len(encoded) - 1 + 2 * encoded.count(b"\n")
        candidate_bytes = indented_bytes + row_bytes + 2 * (len(current) + 1) + 3
        fits = candidate_bytes <= MAX_BYTES and len(current) + 1 <= min(MAX_CSV_ROWS - 1, MAX_JSON_ROWS)
        if not fits:
            require(bool(current), "SINGLE_PUBLIC_ROW_TOO_LARGE")
            pieces.append(current)
            current = [row]
            indented_bytes = row_bytes
            require(row_bytes + 5 <= MAX_BYTES, "SINGLE_PUBLIC_ROW_TOO_LARGE")
        else:
            current.append(row)
            indented_bytes += row_bytes
    pieces.append(current)
    outputs = []
    for index, rows in enumerate(pieces):
        suffix = "" if len(pieces) == 1 else f"_{index + 1:02d}"
        outputs.append(write_json(Path(out) / (stem + suffix + ".json"), rows))
        if csv_output and rows:
            keys = sorted({k for row in rows for k in row})
            buf = io.StringIO(newline="")
            writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()} for row in rows)
            payload = buf.getvalue().encode("utf-8")
            # Nested JSON can grow in CSV quoting, so CSV has its own bound.
            if len(payload) <= MAX_BYTES:
                path = Path(out) / (stem + suffix + ".csv")
                with path.open("xb") as f:
                    f.write(payload)
                outputs.append({"file": path.name, "sha256": sha(path), "bytes": len(payload), "data_rows": len(rows)})
    return outputs


def same_number(a, b, *, tolerance=1e-11):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def checked_metrics(actual, expected):
    for key in ("Harrell", "Harrell_num", "Harrell_den", "Uno60", "Uno_num", "Uno_den"):
        require(same_number(actual.get(key), expected.get(key)), "RECOMPUTED_FOLD_METRICS_CHANGED:" + key)
    if "Uno60_valid" in expected:
        require(actual["Uno60_valid"] == expected["Uno60_valid"], "RECOMPUTED_UNO_SUPPORT_CHANGED")


def fold_micro(rows, *, required_folds=FOLDS):
    """A seed's five within-fold components; no concatenated OOF risk ordering."""
    required = tuple(required_folds)
    seen = [r["fold"] for r in rows]
    require(len(seen) == len(set(seen)), "DUPLICATE_FOLD")
    available = sorted(seen)
    complete = available == sorted(required)
    output = {"status": "COMPLETE" if complete else "NOT_RUN_OR_INCOMPLETE", "fold_count": len(rows),
              "required_fold_count": len(required), "available_folds": available,
              "aggregation": "sum_within_fold_numerators_divided_by_sum_within_fold_denominators"}
    for metric, nk, dk in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
        support = complete and all(r.get(nk) is not None and r.get(dk) is not None and r[dk] > 0 for r in rows)
        n = math.fsum(float(r[nk]) for r in rows) if support else None
        d = math.fsum(float(r[dk]) for r in rows) if support else None
        output.update({metric: n/d if support else None, nk: n, dk: d,
                       metric + "_complete_support": bool(support)})
    return output


def aggregate_arm(arm, fold_rows, *, seeds=SEEDS, required_folds=FOLDS):
    by_seed = []
    for seed in seeds:
        rows = [r for r in fold_rows if r["arm"] == arm and r["seed"] == seed and r.get("status") == "COMPLETE"]
        by_seed.append({"seed": seed, **fold_micro(rows, required_folds=required_folds)})
    out = {"arm": arm, "seed_results": by_seed, "required_seeds": list(seeds),
           "primary_aggregation": "equal_mean_of_seed_level_five_fold_micro_scores", "cross_fold_risk_pairs": False}
    for metric in METRICS:
        values = [r[metric] for r in by_seed]
        ok = all(x is not None for x in values)
        out[metric] = float(np.mean(values)) if ok else None
        out[metric + "_seed_sd"] = float(np.std(values, ddof=0)) if ok else None
        out[metric + "_seed_min"] = min(values) if ok else None
        out[metric + "_seed_max"] = max(values) if ok else None
    out["status"] = "COMPLETE" if all(r["status"] == "COMPLETE" for r in by_seed) else "NOT_RUN_OR_INCOMPLETE"
    return out


def paired_fold_seed_rows(left, right, fold_rows):
    """All fifteen matched cells, including explicit NA cells, without refits."""
    indexed = {(r["arm"], r["fold"], r["seed"]): r for r in fold_rows}
    require(len(indexed) == len(fold_rows), "DUPLICATE_ARM_FOLD_SEED_CELL")
    output = []
    for fold in FOLDS:
        for seed in SEEDS:
            reference_seed = 31 if right == "B0" else seed
            a, b = indexed.get((left, fold, seed)), indexed.get((right, fold, reference_seed))
            valid = a is not None and b is not None and a.get("status") == b.get("status") == "COMPLETE"
            row = {"comparison": left + "-" + right, "left": left, "reference": right,
                   "fold": fold, "seed": seed, "reference_seed": reference_seed,
                   "status": "COMPLETE" if valid else "NOT_RUN_OR_INCOMPLETE",
                   "reference_B0_is_shared_deterministic_fit": right == "B0"}
            for metric, nk, dk in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
                supported = valid and a.get(metric) is not None and b.get(metric) is not None
                row[metric + "_delta"] = float(a[metric] - b[metric]) if supported else None
                row[metric + "_left"] = a.get(metric) if valid else None
                row[metric + "_reference"] = b.get(metric) if valid else None
                for key in (nk, dk):
                    row[key + "_left"] = a.get(key) if valid else None
                    row[key + "_reference"] = b.get(key) if valid else None
                row[metric + "_direction"] = ("NA" if not supported else "POSITIVE" if row[metric + "_delta"] > 0
                                                else "NEGATIVE" if row[metric + "_delta"] < 0 else "ZERO")
            output.append(row)
    return output


def paired_fold_directions(left, right, paired):
    """Fold diagnostics; these equal-seed means do not replace primary micro C."""
    output = []
    for fold in FOLDS:
        group = [r for r in paired if r["fold"] == fold]
        require(len(group) == len(SEEDS) and {r["seed"] for r in group} == set(SEEDS),
                "PAIRED_FOLD_SEED_SUPPORT_CHANGED")
        row = {"comparison": left + "-" + right, "left": left, "reference": right, "fold": fold,
               "aggregation": "equal_seed_mean_of_this_fold_deltas_diagnostic_only_not_primary_micro_score"}
        for metric in METRICS:
            available = [r[metric + "_delta"] for r in group if r[metric + "_delta"] is not None]
            delta = math.fsum(available) / len(SEEDS) if len(available) == len(SEEDS) else None
            row.update({metric + "_fold_delta": delta, metric + "_available_seeds": len(available),
                        metric + "_positive_seeds": sum(x > 0 for x in available),
                        metric + "_negative_seeds": sum(x < 0 for x in available),
                        metric + "_zero_seeds": sum(x == 0 for x in available),
                        metric + "_direction": "NA" if delta is None else "POSITIVE" if delta > 0
                                                else "NEGATIVE" if delta < 0 else "ZERO"})
        output.append(row)
    return output


def comparison(left, right, aggregates, fold_rows):
    out = {"comparison": left + "-" + right, "left": left, "reference": right,
           "confidence_intervals_computed": False, "bootstrap_replicates": 0, "p_value_computed": False}
    a, b = aggregates.get(left), aggregates.get(right)
    if a is None or b is None or a["status"] != "COMPLETE" or b["status"] != "COMPLETE":
        return {**out, "status": "NOT_RUN_OR_INCOMPLETE", "Harrell_delta": None, "Uno60_delta": None}
    out["status"] = "COMPLETE"
    for metric in METRICS:
        delta = a[metric] - b[metric] if a[metric] is not None and b[metric] is not None else None
        out[metric + "_delta"] = delta
        out[metric + "_interpretation"] = ("UNSUPPORTED" if delta is None else
            "NEGATIVE" if delta <= 0 else "WEAK_POSITIVE_DIRECTION_LT_0.005" if delta < .005 else "POSITIVE_POINT_ESTIMATE_NO_UNCERTAINTY_INTERVAL")
        seed_delta = []
        for seed in SEEDS:
            ar = next((r for r in a["seed_results"] if r["seed"] == seed), None)
            br = b["seed_results"][0] if right == "B0" else next((r for r in b["seed_results"] if r["seed"] == seed), None)
            if ar is not None and br is not None:
                seed_delta.append({"seed": seed, "delta": ar[metric] - br[metric] if ar[metric] is not None and br[metric] is not None else None})
        out[metric + "_seed_deltas"] = seed_delta
        cells = paired_fold_seed_rows(left, right, fold_rows)
        paired = [row for row in cells if row[metric + "_delta"] is not None]
        directions = paired_fold_directions(left, right, cells)
        supported_folds = [row for row in directions if row[metric + "_fold_delta"] is not None]
        out[metric + "_fold_seed_pairs"] = len(paired)
        out[metric + "_positive_fold_seed_pairs"] = sum(x[metric + "_delta"] > 0 for x in paired)
        out[metric + "_negative_fold_seed_pairs"] = sum(x[metric + "_delta"] < 0 for x in paired)
        out[metric + "_zero_fold_seed_pairs"] = sum(x[metric + "_delta"] == 0 for x in paired)
        worst_cell = min(paired, key=lambda r: (r[metric + "_delta"], r["fold"], r["seed"])) if paired else None
        worst_fold = min(supported_folds, key=lambda r: (r[metric + "_fold_delta"], r["fold"])) if supported_folds else None
        out[metric + "_worst_fold_seed_cell"] = (
            {"fold": worst_cell["fold"], "seed": worst_cell["seed"], "delta": worst_cell[metric + "_delta"]}
            if worst_cell else None)
        out[metric + "_worst_fold_equal_seed_mean_diagnostic"] = (
            {"fold": worst_fold["fold"], "delta": worst_fold[metric + "_fold_delta"]} if worst_fold else None)
    return out


def risk_scale_row(receipt, cell, compact, scope):
    """Expose already fitted risk scales and aggregate metrics, not risk arrays."""
    train = evaluate_safe(compact["train_time"], compact["train_event"], compact["train_time"],
                          compact["train_event"], cell.training_risk)
    checked_metrics(train, receipt["train_metrics"])
    train_anchor = evaluate_safe(compact["train_time"], compact["train_event"], compact["train_time"],
                                 compact["train_event"], compact["training_offset"])
    train_sd = float(np.std(cell.training_risk - compact["training_offset"]))
    train_anchor_sd = float(np.std(compact["training_offset"]))
    require(same_number(train_sd, receipt["train_delta_sd"]) and
            same_number(train_anchor_sd, receipt["train_anchor_sd"]), "TRAIN_RISK_SCALE_RECEIPT_CHANGED")
    evaluation_sd = float(np.std(cell.evaluation_risk - compact["evaluation_offset"]))
    evaluation_anchor_sd = float(np.std(compact["evaluation_offset"]))
    require(same_number(evaluation_sd, receipt["repair_damage"]["risk_delta_sd"]) and
            same_number(evaluation_anchor_sd, receipt["repair_damage"]["anchor_sd"]), "EVALUATION_RISK_SCALE_RECEIPT_CHANGED")
    output = {**scope, "status": "COMPLETE", "role": receipt["role"],
              "reference": "same_partition_shared_r0_offset",
              "training_metrics_scope": "in_sample_head_fit_over_training_crossfit_teacher_offsets_not_end_to_end_OOF",
              "train_delta_sd": train_sd, "train_anchor_sd": train_anchor_sd,
              "train_delta_sd_over_anchor_sd": train_sd/train_anchor_sd if train_anchor_sd > 1e-12 else None,
              "evaluation_delta_sd": evaluation_sd, "evaluation_anchor_sd": evaluation_anchor_sd,
              "evaluation_delta_sd_over_anchor_sd": evaluation_sd/evaluation_anchor_sd if evaluation_anchor_sd > 1e-12 else None,
              "new_model_forwards": 0, "new_optimizer_steps": 0}
    for label, metrics, anchor_metrics in (("train", train, train_anchor),
                                          ("evaluation", receipt["evaluation_metrics"], receipt["anchor_evaluation_metrics"])):
        output[label + "_Uno60_valid"] = metrics.get("Uno60_valid")
        output[label + "_Uno60_NA_reason"] = metrics.get("Uno60_NA_reason")
        output[label + "_r0_Uno60_valid"] = anchor_metrics.get("Uno60_valid")
        output[label + "_r0_Uno60_NA_reason"] = anchor_metrics.get("Uno60_NA_reason")
        for metric, nk, dk in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
            for key in (metric, nk, dk):
                output[label + "_" + key] = metrics.get(key)
                output[label + "_r0_" + key] = anchor_metrics.get(key)
            a, b = metrics.get(metric), anchor_metrics.get(metric)
            output[label + "_" + metric + "_delta_vs_r0"] = a-b if a is not None and b is not None else None
    return output


def checked_risk(private, lock_sha, split, arm, seed, lam, family, phase):
    folder = Path(private) / "risk" / split / (arm + "_seed" + str(seed))
    path = safe_path(folder, "lambda" + str(float(lam)) + ".private.json")
    row = read(path)
    binding = row["source_and_scope_bindings"]
    require(row["status"] == "FITTED_AND_EVALUATED" and row["DSS_coefficients"] == 26 and
            row["role"] == ("outer_final" if split.endswith("final") else "inner") and
            row["risk_optimizer_fits"] == 1 and row["ridge_lambda"] == lam and
            binding["lock_sha256"] == lock_sha and binding["split"] == split and binding["arm"] == arm and
            binding["seed"] == seed and binding["phase_a_source_family"] == family and
            binding["phase_a_selected_state_sha256"] == phase["source_checkpoint_sha256"] and
            binding["input_training_sha256"] == phase["input_training_sha256"], "RISK_CELL_SCOPE_CHANGED")
    optimizer, replay = row["risk_optimizer"], row["saved_linear_head_replay"]
    require(optimizer["parameters"] == optimizer["active_design_columns"] == 26 and
            optimizer["gradient_max"] <= optimizer["gradient_tolerance"] and optimizer["oracle_loss_abs_error"] <= 1e-10,
            "RISK_OPTIMIZER_QUALIFICATION_FAILED")
    require(replay["actual_linear_head_replay"] and not replay["neural_forward"] and
            replay["max_abs_risk_error"] <= 1e-12 and replay["fold_metric_components_equal"], "MISSING_ACTUAL_LINEAR_HEAD_REPLAY")
    cell = load_verified_pickle(folder, path.with_suffix(".pkl").name, row["artifact_sha256"])
    for key, value in (("evaluation_risk_sha256", cell.evaluation_risk), ("training_risk_sha256", cell.training_risk)):
        require(array_sha(np.asarray(value)) == row[key], "PERSISTED_RISK_ARRAY_HASH_CHANGED")
    compact_path = safe_path(folder, "lambda" + str(float(lam)) + ".compact.private.pkl")
    with compact_path.open("rb") as f:
        compact = pickle.load(f)
    for key, receipt_key in (("evaluation_time", "evaluation_time_sha256"), ("evaluation_event", "evaluation_event_sha256"),
                             ("evaluation_offset", "evaluation_offset_sha256")):
        require(array_sha(np.asarray(compact[key])) == row[receipt_key], "COMPACT_REPLAY_INPUT_HASH_CHANGED")
    for key, receipt_key in (("train_time", "fit_time_sha256"), ("train_event", "fit_event_sha256"),
                             ("training_offset", "fit_offset_sha256"), ("train_design", "fit_design_sha256")):
        require(array_sha(np.asarray(compact[key])) == optimizer[receipt_key], "COMPACT_TRAINING_INPUT_HASH_CHANGED")
    predicted = cell.model.predict(compact["evaluation_design"], compact["evaluation_offset"])
    require(np.max(np.abs(predicted - cell.evaluation_risk)) <= 1e-12, "INDEPENDENT_SAVED_HEAD_ARITHMETIC_CHANGED")
    metric = evaluate_safe(compact["train_time"], compact["train_event"], compact["evaluation_time"],
                           compact["evaluation_event"], cell.evaluation_risk)
    checked_metrics(metric, row["evaluation_metrics"])
    checked_metrics(metric, cell.receipt["evaluation_metrics"])
    return row, cell, compact, {"receipt_sha256": sha(path), "artifact_sha256": row["artifact_sha256"],
                              "compact_artifact_sha256": sha(compact_path), "independent_head_arithmetic_only": True,
                              "new_neural_forward": False, "new_optimizer_steps": 0}


def metric_row(arm, fold, seed, metric, **extra):
    return {"arm": arm, "reported_arm": "B3_TEXTLESS_INCREMENT" if arm == "B3_TEXTLESS" else arm,
            "fold": fold, "seed": seed, "status": "COMPLETE", **{k: metric.get(k) for k in
            ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den", "Uno60_valid", "Uno60_NA_reason")}, **extra}


def public_failure(scope, exc):
    # Exception strings may include private file paths. Only classified codes escape.
    code = type(exc).__name__
    if isinstance(exc, ValueError) and re.fullmatch(r"[A-Z0-9_:]+", str(exc)):
        code = str(exc)
    return {**scope, "status": "FAILED" if not isinstance(exc, FileNotFoundError) else "NOT_RUN", "reason_code": code}


def phase_summary(private, args, lock, split, family, seed):
    phase = checked_phase(args, lock, split, family, seed)
    receipt_path = Path(private) / "phase_a" / split / (family + "_seed" + str(seed)) / "cell.private.json"
    receipt = read(receipt_path)
    row = {"split": split, "family": family, "seed": seed, "status": "COMPLETE", **phase,
           "actual_neural_fits": 1, "elapsed_seconds": receipt["elapsed_seconds"],
           "peak_cuda_allocated_bytes": receipt["peak_cuda_allocated_bytes"],
           "parameters": receipt["parameter_counts"], "evaluate_valid": receipt["evaluate_valid"],
           "DSS_OS_PFI_accessed": receipt["DSS_OS_PFI_accessed"]}
    # Histories/gradients contain many arrays-by-proxy. Only explicit finite summaries are public.
    if receipt["selected_validation"] is not None:
        val = receipt["selected_validation"]
        row["selected_program_huber"] = val["program"]["patient_equal_masked_huber"]
        row["selected_hard_gap_image_to_gene"] = val["retrieval_hard"]["image_to_gene"]["paired_gap_mean"]
        row["selected_hard_gap_gene_to_image"] = val["retrieval_hard"]["gene_to_image"]["paired_gap_mean"]
    return phase, row


def collect_run(private, lock, lock_sha):
    """Read actual completed receipts only. Caller enforces the outer-read gate."""
    private = Path(private)
    args = SimpleNamespace(private=str(private), expected_lock_sha256=lock_sha)
    selection_path = safe_path(private, "ALL_OUTER_SELECTIONS_FROZEN.private.json")
    selection = read(selection_path)
    require(selection["p2_lock_sha256"] == lock_sha and not selection["outer_results_read"] and
            selection["status"] == "ALL_FIVE_OUTERS_SELECTED_BEFORE_FIRST_OUTER_SCORE", "OUTER_SELECTION_BARRIER_MISSING")
    barrier = read(safe_path(private, "ALL_OUTER_SELECTION_BARRIER_BINDING.private.json"))
    require(barrier["p2_lock_sha256"] == lock_sha and barrier["selection_file_sha256"] == sha(selection_path)
            and barrier["status"] == "VERIFIED_ALL_FIVE_OUTER_SELECTIONS_BEFORE_FINAL_STAGE",
            "IMMUTABLE_GLOBAL_SELECTION_BARRIER_CHANGED")
    plan = read(safe_path(private, "SPLITS.private.json"))
    require(plan["p2_lock_sha256"] == lock_sha, "SPLIT_MASTER_HASH_CHANGED")
    arms = {arm: family for family, values in lock["family_readouts"].items() for arm in values}
    require(len(arms) == sum(map(len, lock["family_readouts"].values())), "DUPLICATE_READOUT_FAMILY")
    inner_rows, fold_rows, phase_rows, head_rows, anchor_rows, failures, vectors, risk_scales = [], [], [], [], [], [], {}, []
    phase_cache, verified_inner = {}, {}
    for outer in FOLDS:
        for index in (*range(3), None):
            split = f"outer{outer}_" + ("final" if index is None else f"inner{index}")
            for family in lock["family_readouts"]:
                for seed in SEEDS:
                    scope = {"split": split, "family": family, "seed": seed}
                    try:
                        phase, row = phase_summary(private, args, lock, split, family, seed)
                        phase_cache[(split, family, seed)] = phase
                        phase_rows.append(row)
                        if index is not None:
                            inner_rows.append(row)
                    except (ValueError, OSError, KeyError, StopIteration) as exc:
                        failures.append(public_failure(scope, exc))
            anchor_path = private / "anchors" / (split + ".private.json")
            try:
                anchor = read(safe_path(private, anchor_path.relative_to(private)))
                require(anchor["p2_lock_sha256"] == lock_sha and anchor["split"] == split and
                        anchor["Cox_optimizer_fits"] == 52 and anchor["scalar_pair_optimizer_fits"] == 4 and
                        anchor["full_pipeline_crossfit_for_training_offset"] and
                        anchor["beta_supervision_excluded_for_each_downstream_training_patient"], "R0_FULL_PIPELINE_CROSSFIT_FAILED")
                safe_path(private, anchor_path.with_suffix(".pkl").relative_to(private), anchor["artifact_sha256"])
                anchor_rows.append({"split": split, "status": "COMPLETE", "Cox_optimizer_fits": 52,
                    "scalar_pair_optimizer_fits": 4, "preprocessing_fits": anchor["preprocessing_fits"],
                    "elapsed_seconds": anchor["elapsed_seconds"], "receipt_sha256": sha(anchor_path),
                    "artifact_sha256": anchor["artifact_sha256"], "training_offset_is_complete_pipeline_crossfit": True,
                    "B0_full_teacher_shared_not_double_counted": True})
            except (ValueError, OSError, KeyError) as exc:
                failures.append(public_failure({"split": split, "component": "r0_anchor"}, exc))
            for arm, family in arms.items():
                for seed in SEEDS:
                    phase = phase_cache.get((split, family, seed))
                    lambdas = (.1, 1., 10.) if index is not None else (selection["entries"][str(outer)]["risk"][arm][str(seed)]["ridge_lambda"],)
                    for lam in lambdas:
                        scope = {"split": split, "arm": arm, "seed": seed, "ridge_lambda": lam}
                        try:
                            require(phase is not None, "SOURCE_PHASE_A_NOT_QUALIFIED")
                            row, cell, compact, audit = checked_risk(private, lock_sha, split, arm, seed, lam, family, phase)
                            bg = compact.get("background_audit")
                            if bg is not None:
                                require(bg["DSS_read"] is False and bg["encoder_coordinates"] == "one_frozen_current_partition_model",
                                        "BACKGROUND_OUTCOME_OR_COORDINATE_SCOPE_CHANGED")
                            risk_scales.append(risk_scale_row(row, cell, compact, scope))
                            head_rows.append({**scope, "status": "COMPLETE", **audit,
                                "risk_optimizer_fits": row["risk_optimizer_fits"],
                                "PCA_fits": row["actual_feature_map_refits"]["PCA"],
                                "background_Ridge_fits_for_this_view": row["background_models_in_current_frozen_views"],
                                "background_counted_once_per_view_not_per_lambda": True,
                                "background_used_DSS": False if bg is None else bg["DSS_read"],
                                "DSS_coefficients": row["DSS_coefficients"],
                                "nonzero_estimated_coefficients": row["risk_optimizer"]["nonzero_estimated_coefficients"],
                                "parameter_activity_scope": "nonconstant_design_and_fitted_coefficients_not_per_parameter_causal_interventions",
                                "optimizer_iterations": row["risk_optimizer"]["iterations"],
                                "optimizer_gradient_max": row["risk_optimizer"]["gradient_max"]})
                            if index is None:
                                fold_rows.append(metric_row(arm, outer, seed, row["evaluation_metrics"],
                                    ridge_lambda=lam, selected_phase_a_epoch=phase["selected_epoch"],
                                    state_sha256=phase["source_checkpoint_sha256"], **audit))
                                vectors[(arm, outer, seed)] = (cell, compact)
                            else:
                                verified_inner.setdefault((outer, arm, seed), {}).setdefault(float(lam), []).append(row["evaluation_metrics"])
                        except (ValueError, OSError, KeyError, AttributeError) as exc:
                            failures.append(public_failure(scope, exc))
    # Independently recalculate all inner choices before interpreting final scores.
    selection_audit = []
    for (outer, arm, seed), candidates in verified_inner.items():
        try:
            choice = select_lambda(candidates)
            frozen = selection["entries"][str(outer)]["risk"][arm][str(seed)]
            require(choice["ridge_lambda"] == frozen["ridge_lambda"] and
                    choice["candidate_summaries"] == frozen["candidate_summaries"] and not frozen["uses_outer_test"],
                    "FROZEN_INNER_RIDGE_SELECTION_CHANGED")
            actual_bindings = []
            for lam in (.1, 1., 10.):
                for inner in range(3):
                    candidates_for_binding = [h for h in head_rows if h["split"] == f"outer{outer}_inner{inner}"
                        and h["arm"] == arm and h["seed"] == seed and h["ridge_lambda"] == lam]
                    require(len(candidates_for_binding) == 1, "INNER_RISK_BINDING_MISSING_OR_DUPLICATE")
                    head = candidates_for_binding[0]
                    actual_bindings.append({"inner": inner, "lambda": lam, "receipt_sha256": head["receipt_sha256"],
                                            "artifact_sha256": head["artifact_sha256"]})
            require(actual_bindings == frozen["bindings"], "FROZEN_INNER_RISK_RECEIPT_HASH_CHANGED")
            selection_audit.append({"outer": outer, "arm": arm, "seed": seed, **choice})
        except (ValueError, KeyError) as exc:
            failures.append(public_failure({"outer": outer, "arm": arm, "seed": seed, "component": "lambda_selection"}, exc))
    # Every outer neural fit must have the median of three independently verified inner epochs.
    for outer in FOLDS:
        for family in lock["family_readouts"]:
            for seed in SEEDS:
                inner = [phase_cache.get((f"outer{outer}_inner{i}", family, seed)) for i in range(3)]
                final = phase_cache.get((f"outer{outer}_final", family, seed))
                if final is not None and all(x is not None for x in inner):
                    median = sorted(x["selected_epoch"] for x in inner)[1]
                    chosen = selection["entries"][str(outer)]["pretraining"][family][str(seed)]
                    require(final["selected_epoch"] == median == chosen["epoch"] and
                            chosen["inner_selected_epochs"] == [x["selected_epoch"] for x in inner] and
                            chosen["bindings"] == [{"inner": i, "receipt_sha256": x["receipt_sha256"],
                                "selected_state_sha256": x["source_checkpoint_sha256"]} for i, x in enumerate(inner)],
                            "OUTER_FIXED_EPOCH_CHANGED")
    for outer in FOLDS:
        try:
            path = safe_path(private, f"B0/outer{outer}_final.private.json")
            row = read(path)
            anchor_row = read(safe_path(private, f"anchors/outer{outer}_final.private.json"))
            require(row["source_anchor_sha256"] == anchor_row["artifact_sha256"] and row["deterministic_single_refit"] and row["seed"] == 31,
                    "B0_SHARED_TEACHER_IDENTITY_CHANGED")
            anchor = load_verified_pickle(private, f"anchors/outer{outer}_final.private.pkl", row["source_anchor_sha256"])
            sample = next((vectors[k][1] for k in vectors if k[1] == outer), None)
            require(sample is not None, "NO_QUALIFIED_OUTER_ENDPOINT_REFERENCE")
            require(np.array_equal(anchor.b0_exact_evaluation, anchor.evaluation_offset), "B0_EVAL_OFFSET_CHANGED")
            metric = evaluate_safe(sample["train_time"], sample["train_event"], sample["evaluation_time"], sample["evaluation_event"], anchor.b0_exact_evaluation)
            checked_metrics(metric, row["DSS_metrics"])
            for key, (_, compact) in vectors.items():
                if key[1] == outer:
                    require(np.array_equal(compact["evaluation_offset"], anchor.evaluation_offset) and
                            np.array_equal(compact["training_offset"], anchor.training_offset), "COMMON_R0_NOT_ELEMENTWISE_IDENTICAL")
                    for field in ("train_time", "train_event", "evaluation_time", "evaluation_event"):
                        require(np.array_equal(compact[field], sample[field]), "PAIRED_ENDPOINT_ORDER_CHANGED")
            fold_rows.append(metric_row("B0", outer, 31, metric, source_anchor_sha256=row["source_anchor_sha256"],
                                        deterministic_single_refit=True, historical_three_seed_mean_reproduced=False))
            vectors[("B0", outer, 31)] = (SimpleNamespace(evaluation_risk=anchor.b0_exact_evaluation), sample)
        except (ValueError, OSError, KeyError, AttributeError) as exc:
            failures.append(public_failure({"fold": outer, "arm": "B0", "component": "B0_identity"}, exc))
    return {"arms": arms, "inner": inner_rows, "fold": fold_rows, "phase": phase_rows, "heads": head_rows,
            "anchors": anchor_rows, "failures": failures, "vectors": vectors, "selection": selection_audit,
            "selection_file_sha256": sha(selection_path), "risk_scales": risk_scales}


def repair_reports(collected):
    vectors = collected["vectors"]
    pairs = set(CORE) | {(arm, "B0") for arm in collected["arms"]}
    rows = []
    for left, right in sorted(pairs):
        for outer in FOLDS:
            for seed in SEEDS:
                a = vectors.get((left, outer, seed))
                b = vectors.get((right, outer, 31 if right == "B0" else seed))
                if a is None or b is None:
                    continue
                cell, c = a
                reference, other = b
                for key in ("train_time", "train_event", "evaluation_time", "evaluation_event"):
                    require(np.array_equal(c[key], other[key]), "REPAIR_PAIRED_ENDPOINTS_CHANGED")
                report = risk_repair_damage(c["evaluation_time"], c["evaluation_event"], cell.evaluation_risk,
                    reference.evaluation_risk, c["train_time"], c["train_event"])
                metric = evaluate_safe(c["train_time"], c["train_event"], c["evaluation_time"], c["evaluation_event"], cell.evaluation_risk)
                refmetric = evaluate_safe(c["train_time"], c["train_event"], c["evaluation_time"], c["evaluation_event"], reference.evaluation_risk)
                require(same_number(report["net_Harrell_change"], metric["Harrell"] - refmetric["Harrell"]), "PAIR_REPAIR_MINUS_DAMAGE_IDENTITY_FAILED")
                rows.append({"comparison": left + "-" + right, "fold": outer, "seed": seed, **report})
    return rows


def cost_report(collected, lock):
    phase, heads, anchors = collected["phase"], collected["heads"], collected["anchors"]
    backgrounds = {}
    for row in heads:
        key = (row["split"], row["arm"], row["seed"])
        count = row["background_Ridge_fits_for_this_view"]
        require(key not in backgrounds or backgrounds[key] == count, "SAME_VIEW_BACKGROUND_COUNT_CHANGED")
        backgrounds[key] = count
    expected_phase = 20 * len(lock["family_readouts"]) * len(SEEDS)
    expected_heads = len(collected["arms"]) * len(SEEDS) * (15 * 3 + 5)
    return {"status": "COMPLETE" if len(phase) == expected_phase and len(heads) == expected_heads and len(anchors) == 20 else "PARTIAL",
        "actual_neural_fits": len(phase), "expected_neural_fits": expected_phase,
        "actual_phase_A_optimizer_steps": sum(x["optimizer_steps"] for x in phase),
        "actual_saved_checkpoint_replays": sum(x["actual_checkpoint_replays"] for x in phase),
        "actual_neural_fit_elapsed_seconds_sum_not_wallclock": math.fsum(x["elapsed_seconds"] for x in phase),
        "max_per_fit_peak_cuda_allocated_bytes_not_concurrent_sum": max((x["peak_cuda_allocated_bytes"] or 0 for x in phase), default=0),
        "actual_risk_Cox_fits": len(heads), "expected_risk_Cox_fits": expected_heads,
        "actual_risk_map_PCA_fits_including_lambda_repeats": sum(x["PCA_fits"] for x in heads),
        "actual_outcome_free_background_Ridge_fits": sum(backgrounds.values()),
        "background_count_rule": "once_per_split_arm_seed_frozen_view_reused_across_its_lambdas_not_shared_across_arms",
        "actual_r0_anchor_partitions": len(anchors), "actual_r0_Cox_fits": sum(x["Cox_optimizer_fits"] for x in anchors),
        "actual_r0_scalar_fits": sum(x["scalar_pair_optimizer_fits"] for x in anchors),
        "actual_r0_preprocessing_fits": sum(x["preprocessing_fits"] for x in anchors),
        "actual_r0_elapsed_seconds_sum_not_wallclock": math.fsum(x["elapsed_seconds"] for x in anchors),
        "B0_full_teacher_shared_with_r0_not_counted_twice": True,
        "excluded_separate_scopes": ["P0_old95_prediction_rescores", "P0_zero_optimizer_probes", "P1", "P1R", "E0_independent_anchor_profile"],
        "report_new_optimizer_steps": 0, "report_new_neural_forwards": 0,
        "report_head_arithmetic_checks_are_not_second_neural_replays": True,
        "confidence_intervals_computed": False}


def resource_profile(path, expected_sha256):
    require(expected_sha256 is not None and sha(path) == expected_sha256, "RESOURCE_PROFILE_SHA_REQUIRED_OR_CHANGED")
    value = read(path)
    require(value["endpoints_read"] is False and value["optimizer_updates"] == value["new_neural_fits"] == 0,
            "RESOURCE_PROFILE_HAS_OUTCOME_OR_OPTIMIZER_ACCESS")
    require(value["actual_saved_state_replays"] == 6 and len(value["profiles"]) == 6, "RESOURCE_PROFILE_SIX_CELLS_REQUIRED")
    for relative, digest in value["sources"].items():
        safe_path(ROOT, relative, digest)
    fields = ("role", "arm", "train_patients", "parameters", "forward_profile_seconds",
              "forward_backward_profile_seconds", "peak_cuda_allocated_bytes", "forward_supported_matrix_FLOPs",
              "forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs", "checkpoint_sha256",
              "FLOPs_exclude_unsupported_elementwise_operators", "actual_saved_reload_forward_replay",
              "optimizer_steps", "new_neural_fits", "profiler_event_collection")
    rows = []
    cells = {}
    for profile in value["profiles"]:
        key = (profile["role"], profile["arm"])
        require(key not in cells and key[0] in ("inner", "outer_final") and key[1] in ("B4_DUAL", "B2_MATCHED", "B2_NATIVE"),
                "RESOURCE_PROFILE_DUPLICATE_OR_UNEXPECTED_CELL")
        gradients = list(profile["gradient_norms"].values())
        require(gradients and all(g is not None and math.isfinite(g) and g > 0 for g in gradients), "RESOURCE_PROFILE_DEAD_GRADIENT")
        require(profile["optimizer_steps"] == profile["new_neural_fits"] == 0 and profile["actual_saved_reload_forward_replay"]
                and profile["FLOPs_exclude_unsupported_elementwise_operators"], "RESOURCE_PROFILE_SCOPE_OR_REPLAY_INVALID")
        cells[key] = profile
        rows.append({k: profile[k] for k in fields})
    ratios = []
    for role in ("inner", "outer_final"):
        reference = cells[(role, "B4_DUAL")]
        for arm in ("B2_MATCHED", "B2_NATIVE"):
            row = {"role": role, "arm": arm, "reference": "B4_DUAL", "matching_tolerance": .1}
            for field in ("parameters", "forward_supported_matrix_FLOPs",
                          "forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs"):
                require(reference[field] > 0 and cells[(role, arm)][field] > 0, "RESOURCE_PROFILE_ZERO_DENOMINATOR")
                error = cells[(role, arm)][field]/reference[field] - 1.
                row[field + "_ratio_minus_one"] = error
                row[field + "_within_10_percent"] = abs(error) <= .1
            row["matched_qualification"] = ("PASS_PARAMETERS_AND_SUPPORTED_MATRIX_FLOPS" if all(
                row[k] for k in row if k.endswith("_within_10_percent")) else "NOT_QUALIFIED_AS_COMPUTE_MATCHED")
            ratios.append(row)
    return {"status": "VERIFIED_REAL_ZERO_UPDATE_RESOURCE_PROFILE", "receipt_sha256": expected_sha256,
        "profiles": rows, "comparisons": ratios, "new_neural_fits": 0, "optimizer_updates": 0,
        "scope": "actual_CUDA_execution_CPU_dispatcher_supported_matrix_FLOPs_not_all_operator_FLOPs",
        "forward_and_forward_backward_are_separate_statistics": True,
        "summed_profile_timings_are_not_concurrent_training_wallclock": True}


def results_markdown(status, aggregates, comparisons, costs, failure_count):
    def number(value):
        return "NA" if value is None else f"{value:.6f}"
    lines = ["# P2 B0–B7：统一协议 DSS 生存验证", "", f"**状态：{status}**。失败或未完成记录：{failure_count}。", "",
        "每个 seed 先汇总五折内的 concordance numerator/denominator，再对三个 seed 等权平均；不跨折构建风险比较对。B0 只执行一次确定性配方的五折评估。", "",
        "| 模型 | 状态 | Harrell | Uno60 |", "|---|---|---:|---:|"]
    for arm in sorted(aggregates):
        row = aggregates[arm]
        label = "B3_TEXTLESS_INCREMENT" if arm == "B3_TEXTLESS" else arm
        lines.append(f"| {label} | {row['status']} | {number(row['Harrell'])} | {number(row['Uno60'])} |")
    lines += ["", "| 预锁配对比较 | ΔHarrell | ΔUno60 |", "|---|---:|---:|"]
    for row in comparisons:
        label = row["comparison"].replace("B3_TEXTLESS", "B3_TEXTLESS_INCREMENT")
        lines.append(f"| {label} | {number(row.get('Harrell_delta'))} | {number(row.get('Uno60_delta'))} |")
    lines += ["", "0 < Δ < 0.005 只记为弱正向点估计。未计算置信区间、bootstrap 或 p 值，不能据此宣称机制已被证实。", "",
        f"实际神经拟合 {costs['actual_neural_fits']} 次，Phase-A 优化 {costs['actual_phase_A_optimizer_steps']} 步；"
        f"下游 Cox {costs['actual_risk_Cox_fits']} 次；完整交叉拟合 r0 的 Cox {costs['actual_r0_Cox_fits']} 次、标量拟合 {costs['actual_r0_scalar_fits']} 次。",
        "PCA 与无结局背景 Ridge 的实际重复次数见 09_cost_ledger.json；B0 与 r0 共用的完整教师没有重复计数。", "",
        "优势：保留强共同风险主干，统一 26 个下游系数，独立验证保存头的算术、逐折评测与配对修复/损伤。", "",
        "风险与限制：这是历史开发过的内部队列，DSS 事件有限；TITAN 代理分数不等于真实组织学标签；种子/折方向一致性不代替外部验证。", "",
        "TEXTLESS_INCREMENT 仅移除增量编码器的显式文本；共同 r0 仍包含 TITAN S32，完整风险不是无文本模型。"
        "B2 为 8 个无语义读取位置 × 50 个程序的普通注意力适配；B2_NATIVE 不是论文官方原生复现，计算匹配须依据真实 profile。", "",
        "P0 的 95 份旧风险重评分属于单独历史桥接，未记为本轮新拟合或同输入架构证据。保存头重算属于算术复核，没有新增神经前向或训练。", ""]
    return "\n".join(lines)


def generate_reports(private, lock_path, expected_lock_sha256, output, *, allow_completed_outer_results=False,
                     resource_profile_path=None, expected_resource_profile_sha256=None):
    require(allow_completed_outer_results, "OUTER_READ_REQUIRES_POSTCOMPLETION_AUTHORIZATION")
    require(sha(lock_path) == expected_lock_sha256, "P2_LOCK_HASH_CHANGED")
    lock = read(lock_path)
    require(lock["status"] == "LOCKED_BEFORE_OUTER_EVALUATION", "P2_NOT_PRELOCKED")
    # Report code is independent of frozen training sources, whose current bytes still must match.
    for relative, digest in lock["frozen_sources"].items():
        safe_path(ROOT, relative, digest)
    out = Path(output)
    require(not out.exists(), "APPEND_ONLY_NEW_REPORT_DIRECTORY_REQUIRED")
    collected = collect_run(private, lock, expected_lock_sha256)
    aggregates = {arm: aggregate_arm(arm, collected["fold"]) for arm in collected["arms"]}
    aggregates["B0"] = aggregate_arm("B0", collected["fold"], seeds=(31,))
    comparisons = [comparison(a, b, aggregates, collected["fold"]) for a, b in CORE]
    costs = cost_report(collected, lock)
    resources = (resource_profile(resource_profile_path, expected_resource_profile_sha256) if resource_profile_path is not None
                 else {"status": "NOT_PROVIDED_COMPUTE_MATCH_CLAIM_NOT_QUALIFIED", "new_neural_fits": 0, "optimizer_updates": 0})
    complete = not collected["failures"] and costs["status"] == "COMPLETE" and all(x["status"] == "COMPLETE" for x in aggregates.values())
    status = "COMPLETE" if complete else "FAILED_OR_PARTIAL_NO_CONFIRMATORY_CONCLUSION"
    files = []
    files += write_records(out, "04_inner_phase_selection", collected["inner"])
    files += write_records(out, "04_inner_ridge_selection", collected["selection"])
    files += write_records(out, "05_fold_seed_metrics", collected["fold"])
    files.append(write_json(out / "05_seed_aggregated_metrics.json", list(aggregates.values())))
    files += write_records(out, "06_paired_comparisons", comparisons)
    paired_cells, paired_directions = [], []
    for left, right in CORE:
        cells = paired_fold_seed_rows(left, right, collected["fold"])
        paired_cells.extend(cells)
        paired_directions.extend(paired_fold_directions(left, right, cells))
    files += write_records(out, "06_fold_seed_paired_deltas", paired_cells)
    files += write_records(out, "06_fold_direction_summaries", paired_directions)
    reference_pairs = sorted({(arm, baseline) for arm in collected["arms"]
                              for baseline in ("B0", "B1") if arm != baseline} - set(CORE))
    reference_pairs += [("B5", "B3")]  # enum-required exclusion, derived from the already locked arms
    files += write_records(out, "06_supplementary_reference_comparisons", [
        {**comparison(a, b, aggregates, collected["fold"]), "scope": "derived_reference_diagnostics_not_new_primary_model_selection"}
        for a, b in reference_pairs])
    extra_cells, extra_directions = [], []
    for left, right in reference_pairs:
        cells = paired_fold_seed_rows(left, right, collected["fold"])
        extra_cells.extend(cells)
        extra_directions.extend(paired_fold_directions(left, right, cells))
    files += write_records(out, "06_supplementary_reference_fold_seed_deltas", extra_cells)
    files += write_records(out, "06_supplementary_reference_fold_directions", extra_directions)
    controls = {"scope": "same_frozen_feature_and_strong_r0_controls", "results": [x for x in comparisons if x["reference"] != "B0"],
        "TEXTLESS": {"reported_as": "B3_TEXTLESS_INCREMENT", "adapter_explicit_text": False,
                     "common_r0_contains_TITAN_S32": True, "whole_risk_text_free": False},
        "B2": {"comparison_domain": "eight_nonsemantic_readers_by_fifty_program_tokens",
               "B2_NATIVE_is_official_paper_reproduction": False, "matched_compute_claim_requires_actual_profile": True}}
    files.append(write_json(out / "07_controls.json", controls))
    files += write_records(out, "08_pair_repair_damage", repair_reports(collected))
    files += write_records(out, "08_risk_scale_and_fit_metrics", collected.get("risk_scales", []))
    files.append(write_json(out / "09_cost_ledger.json", costs))
    files.append(write_json(out / "09_resource_profile.json", resources))
    files += write_records(out, "09_phase_a_actual_fits", collected["phase"])
    files += write_records(out, "09_risk_actual_fits", collected["heads"])
    files += write_records(out, "09_r0_actual_fits", collected["anchors"])
    files += write_records(out, "10_failures", collected["failures"])
    core_deltas = {x["comparison"]: {m: x.get(m + "_delta") for m in METRICS} for x in comparisons}
    decision = {"status": status, "DSS_primary_outcome": True, "decision_basis": "all_predeclared_comparisons_preserved_no_single_metric_winner_selection",
        "core_point_estimates": core_deltas, "positive_delta_below_0_005": "weak_direction_only",
        "scientific_mechanism_proven": False, "requires_independent_validation": True,
        "no_further_model_expansion_automatically_authorized": True, "confidence_intervals_computed": False,
        "bootstrap_replicates": 0, "p_values_computed": False}
    files.append(write_json(out / "11_decision.json", decision))
    limits = {"cohort": "historically_developed_internal_patient_level_crossvalidation_not_external_confirmation",
        "events": "few_DSS_events_limit_conditional_increment_precision",
        "inference": "paired_point_estimates_and_seed_fold_stability_without_CI_or_p_values",
        "semantics": "frozen_FM_concept_scores_are_proxies_not_independent_morphology_truth",
        "textless": "incremental_adapter_only_shared_r0_has_TITAN_S32",
        "B2": "shared_token_attention_adaptation_not_official_native_method_reproduction",
        "repair": "ranking_pair_repair_and_damage_are_predictive_diagnostics_not_biological_causation",
        "E0": "independent_profile_is_excluded_from_main_fit_counts_unless_exact_reuse_receipt_explicitly_recorded",
        "P0_history": "95_old_vectors_recomputed_separately_not_new_training_or_same_input_architecture_evidence",
        "no_selection_from_outer_results": True}
    files.append(write_json(out / "12_limitations.json", limits))
    reproduce = {"schema": "semantic_p2_anonymous_reproducibility_v1", "master_lock_sha256": expected_lock_sha256,
        "all_outer_selections_sha256": collected["selection_file_sha256"], "frozen_scientific_sources": lock["frozen_sources"],
        "report_source_sha256": sha(__file__), "primary_aggregation": "within_fold_micro_per_seed_then_equal_three_seed_mean",
        "B0_aggregation": "five_folds_one_deterministic_seed31_recipe_no_pretend_three_seed_refits",
        "reports_contain_patient_arrays_or_weights": False, "report_new_optimizer_steps": 0, "report_new_neural_forwards": 0,
        "verification": "receipt_hash_and_saved_head_arithmetic_not_a_second_actual_neural_forward_replay"}
    files.append(write_json(out / "13_reproduce.json", reproduce))
    files.append(write_markdown(out / "RESULTS.md", results_markdown(status, aggregates, comparisons, costs, len(collected["failures"]))))
    files.append(write_json(out / "00_report_index.json", {"status": status, "files": files,
        "failures": len(collected["failures"]), "expected_arms_from_lock": list(collected["arms"]),
        "all_metrics_come_from_verified_actual_receipts": True, "no_private_arrays_published": True}))
    return {"status": status, "public_files": len(files), "failures": len(collected["failures"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resource-profile")
    parser.add_argument("--expected-resource-profile-sha256")
    parser.add_argument("--allow-completed-outer-results", action="store_true",
                        help="Use only after the complete locked training run is finished and authorized for aggregation")
    args = parser.parse_args()
    print(json.dumps(generate_reports(args.private, args.lock, args.expected_lock_sha256, args.output,
        allow_completed_outer_results=args.allow_completed_outer_results, resource_profile_path=args.resource_profile,
        expected_resource_profile_sha256=args.expected_resource_profile_sha256), ensure_ascii=False))


if __name__ == "__main__":
    main()

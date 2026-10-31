"""Independent CPU-only P2 saved-risk, scorer and hash audit after completion.

This verifier does not import the producer scorer, scheduler or reporter. Its
NumPy pair loops, reverse KM and Efron derivatives are independent arithmetic.
Only own hash-qualified private pickles are decoded into passive records; no
neural model class, checkpoint weights, H5 features or optimizer is executed.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (31, 47, 83)
FOLDS = (1, 2, 3, 4, 5)
CHECKPOINTS = (50, 100, 200, 400)
CORE_COMPARISONS = (("B5", "B0"), ("B5", "B1"), ("B5", "B2_MATCHED"),
    ("B5", "B2_NATIVE"), ("B5", "B4"), ("B5", "B5_CLINICAL_CAPACITY_CONTROL"),
    ("B5", "B6a"), ("B5", "B7"), ("B4", "B3"), ("B4", "B3_TEXTLESS"),
    ("B4", "B4_MAP_SHUFFLE"), ("B1", "B0"))
COHORT_SHA = "4aaeb4cc22374f32615148605d824abed679fcb957c64286b6a8e44f12a294a1"
ENDPOINT_SHA = "91b5d15aa32489c825a8341aed8d9733281dc62386151c8c6246b6312975fd4e"
OFFICIAL_SOURCES = [
    "https://scikit-survival.readthedocs.io/en/v0.23.1/api/generated/sksurv.metrics.concordance_index_ipcw.html",
    "https://github.com/sebp/scikit-survival/blob/v0.23.1/sksurv/nonparametric.py",
    "https://github.com/sebp/scikit-survival/blob/v0.23.1/sksurv/metrics.py"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_sha(value):
    x = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(x.dtype), "shape": list(x.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + x.tobytes()).hexdigest()


def metadata_tensor_sha(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str(x.shape).encode() + str(x.dtype).encode() + x.tobytes()).hexdigest()


def checked(root, name, expected=None):
    require(isinstance(name, str) and "\\" not in name and ":" not in name, "UNSAFE_RELATIVE_FILE")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name, "UNSAFE_RELATIVE_FILE")
    base = Path(root).resolve()
    path = base.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base), "MISSING_OR_ESCAPING_ARTIFACT")
    require(not any(p.is_symlink() for p in path.parents if p.is_relative_to(base)), "SYMLINK_ARTIFACT_PARENT")
    if expected is not None:
        require(sha(path) == expected, "ARTIFACT_SHA_CHANGED")
    return path


class PassiveRecord:
    def __setstate__(self, value):
        if isinstance(value, dict):
            self.__dict__.update(value)
        elif isinstance(value, tuple) and len(value) == 2:
            for part in value:
                if part:
                    self.__dict__.update(part)
        else:
            raise ValueError("UNSUPPORTED_PASSIVE_RECORD_STATE")


class PassiveUnpickler(pickle.Unpickler):
    """Decode arrays/records without importing any study model implementation."""
    def find_class(self, module, name):
        if module == "builtins" and name in ("set", "frozenset", "slice", "complex"):
            return getattr(builtins, name)
        allowed_numpy = {
            ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer")}
        normal = module.replace("numpy._core", "numpy.core")
        if (normal, name) in allowed_numpy:
            return getattr(importlib.import_module(normal), name)
        if (module in ("src.semantic_b0b7.p2_survival", "src.semantic_b0b7.p2_anchor") or
                module.startswith(("_semantic_r4_legacy_", "src.phase10_trimodal.", "sklearn."))):
            return PassiveRecord
        raise ValueError("UNAPPROVED_PRIVATE_PICKLE_GLOBAL")


def passive_pickle(path):
    with Path(path).open("rb") as f:
        return PassiveUnpickler(f).load()


def endpoints(time, event):
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event)
    require(t.ndim == 1 and len(t) and t.shape == e.shape and np.isfinite(t).all() and
            (t >= 0).all() and np.isin(e, (0, 1)).all(), "INVALID_SURVIVAL_ENDPOINTS")
    return t, e.astype(bool)


def reverse_km_table(train_time, train_event):
    """Patient counts, then remove each time group once; no producer KM used."""
    t, e = endpoints(train_time, train_event)
    counts = {}
    for at, observed in zip(t, e):
        group = counts.setdefault(float(at), [0, 0])
        group[int(observed)] += 1
    probability, remaining, times, values = 1., len(t), [], []
    for at in sorted(counts):
        censor, death = counts[at]
        risk = remaining - death
        if censor:
            require(risk >= censor > 0, "INVALID_REVERSE_KM_RISKSET")
            probability *= (risk - censor) / risk
        times.append(at)
        values.append(probability)
        remaining -= censor + death
    return np.asarray(times), np.asarray(values)


def km_query(table, query):
    require(math.isfinite(query) and query >= 0, "INVALID_KM_QUERY")
    times, probabilities = table
    if query > times[-1] and probabilities[-1] > 0:
        raise ValueError("CENSOR_QUERY_EXCEEDS_TRAIN_SUPPORT")
    index = int(np.searchsorted(times, query, side="right"))-1
    return float(probabilities[index]) if index >= 0 else 1.


def censor_km_value(train_time, train_event, query):
    """Right-continuous reverse KM: deaths precede tied censor jumps."""
    return km_query(reverse_km_table(train_time, train_event), query)


def pair_statistic(time, event, risk, weights=None):
    """Naive directed patient-pair loop; no sorting/producer helper reused."""
    t, e = endpoints(time, event)
    r = np.asarray(risk, np.float64)
    require(r.shape == t.shape and np.isfinite(r).all(), "INVALID_RISK_VECTOR")
    w = np.ones(len(t)) if weights is None else np.asarray(weights, np.float64)
    require(w.shape == t.shape and np.isfinite(w).all() and (w >= 0).all(), "INVALID_PAIR_WEIGHTS")
    numerator = denominator = 0.
    pairs = event_censor_ties = risk_ties = 0
    for i in range(len(t)):
        if not e[i]:
            continue
        for j in range(len(t)):
            if t[i] < t[j] or (t[i] == t[j] and not e[j]):
                difference = r[i] - r[j]
                credit = .5 if abs(difference) <= 1e-8 else float(difference > 0)
                numerator += float(w[i]) * credit
                denominator += float(w[i])
                pairs += 1
                event_censor_ties += int(t[i] == t[j])
                risk_ties += int(abs(difference) <= 1e-8)
    return {"num": numerator, "den": denominator, "score": numerator/denominator if denominator > 0 else None,
            "pairs": pairs, "event_censor_tied_time_pairs": event_censor_ties, "risk_ties": risk_ties}


def independent_metrics(train_time, train_event, time, event, risk):
    t, e = endpoints(time, event)
    h = pair_statistic(t, e, risk)
    table = reverse_km_table(train_time, train_event)
    try:
        g60 = km_query(table, 60.)
        reason = None if g60 > 0 else "IPCW_ZERO_CENSOR_SURVIVAL_AT_TAU"
    except ValueError as exc:
        g60, reason = None, str(exc)
    result = {"Harrell": h["score"], "Harrell_num": h["num"], "Harrell_den": h["den"],
              "Uno60": None, "Uno_num": None, "Uno_den": None, "Uno60_valid": False,
              "Uno60_NA_reason": reason, "tau_support": {"supported": reason is None, "probability": g60, "reason": reason}}
    if reason is None:
        weights = np.zeros(len(t))
        try:
            for i in range(len(t)):
                if e[i] and t[i] < 60.:
                    g = km_query(table, float(t[i]))
                    if g <= 0:
                        raise ValueError("IPCW_ZERO_CENSOR_SURVIVAL")
                    weights[i] = 1. / (g*g)
        except ValueError as exc:
            result["Uno60_NA_reason"] = str(exc)
            return result
        u = pair_statistic(t, e, risk, weights)
        result.update(Uno60=u["score"], Uno_num=u["num"], Uno_den=u["den"], Uno60_valid=True)
    return result


def efron_gradient(risk, time, event):
    """Direct risk-set sums with a separate event-tie loop and exp stabilization."""
    t, e = endpoints(time, event)
    r = np.asarray(risk, np.float64)
    require(r.shape == t.shape and np.isfinite(r).all() and e.any(), "INVALID_EFRON_RISK_OR_NO_EVENTS")
    loss = -math.fsum(float(r[i]) for i in range(len(t)) if e[i])
    gradient = -e.astype(float)
    for at in sorted(set(float(t[i]) for i in range(len(t)) if e[i])):
        dead = [i for i in range(len(t)) if e[i] and t[i] == at]
        alive = [i for i in range(len(t)) if t[i] >= at]
        maximum = max(float(r[j]) for j in alive)
        for tie in range(len(dead)):
            weighted = np.array([math.exp(float(r[j])-maximum) *
                                 (1.-tie/len(dead) if j in dead else 1.) for j in alive])
            total = math.fsum(weighted.tolist())
            loss += maximum + math.log(total)
            gradient[alive] += weighted / total
    return float(loss/e.sum()), gradient/e.sum()


def cox_objective(design, coefficient, offset, time, event, ridge_lambda):
    x, w, base = np.asarray(design, float), np.asarray(coefficient, float), np.asarray(offset, float)
    require(x.ndim == 2 and x.shape[1] == w.size and w.ndim == 1 and base.shape == (len(x),) and
            np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(base).all(), "INVALID_COX_DESIGN")
    loss, dr = efron_gradient(base + x @ w, time, event)
    return loss + ridge_lambda/2*float(w @ w), x.T @ dr + ridge_lambda*w


def close(a, b, *, tolerance=1e-10):
    return a is None and b is None if a is None or b is None else math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def check_metrics(actual, expected):
    errors = []
    for key in ("Harrell", "Harrell_num", "Harrell_den", "Uno60", "Uno_num", "Uno_den"):
        require(close(actual[key], expected[key]), "INDEPENDENT_METRIC_DISAGREEMENT:" + key)
        if actual[key] is not None:
            errors.append(abs(float(actual[key])-float(expected[key])))
    if "Uno60_valid" in expected:
        require(actual["Uno60_valid"] == expected["Uno60_valid"] and actual["Uno60_NA_reason"] == expected["Uno60_NA_reason"],
                "INDEPENDENT_UNO_SUPPORT_DISAGREEMENT")
    return max(errors, default=0.)


def canonical_labels(cohort_path, endpoint_path):
    require(sha(cohort_path) == COHORT_SHA and sha(endpoint_path) == ENDPOINT_SHA, "CANONICAL_CONTAINER_BYTES_CHANGED")
    with np.load(cohort_path, allow_pickle=False) as f:
        patient = np.array(f["patient_ids"], copy=True)
        folds = np.array(f["folds"], copy=True)
    require(patient.shape == folds.shape == (512,) and len(np.unique(patient)) == 512 and set(folds) == set(FOLDS),
            "CANONICAL_PATIENT_OR_FOLD_SHAPE_CHANGED")
    with np.load(endpoint_path, allow_pickle=False) as f:
        require(str(f["patient_hash"]) == hashlib.sha256(np.ascontiguousarray(patient).tobytes()).hexdigest(), "CANONICAL_ENDPOINT_ORDER_CHANGED")
        time = np.array(f["DSS_time"], dtype=np.float64)
        event = np.array(f["DSS_event"], dtype=np.float64)
        valid = np.array(f["DSS_valid"], dtype=bool)
    endpoints(time, event)
    require(time.shape == (512,) and valid.shape == (512,) and valid.all() and event.sum() == 65, "CANONICAL_DSS_ELIGIBILITY_CHANGED")
    return patient, folds, time, event


def verify_splits(plan, lock, patient, folds, time, event):
    from sklearn.model_selection import StratifiedKFold
    expected = {f"outer{o}_" + ("final" if i is None else f"inner{i}") for o in FOLDS for i in (*range(3), None)}
    require(set(plan["partitions"]) == expected, "EXACT_TWENTY_PARTITIONS_REQUIRED")
    for outer in FOLDS:
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        splitter = StratifiedKFold(3, shuffle=True, random_state=lock["inner_split_seed"]+outer)
        inner = list(splitter.split(train, event[train]))
        for index in (*range(3), None):
            name = f"outer{outer}_" + ("final" if index is None else f"inner{index}")
            row = plan["partitions"][name]
            a, b = (train, test) if index is None else (train[inner[index][0]], train[inner[index][1]])
            require(np.array_equal(row["fit"], a) and np.array_equal(row["evaluation"], b) and
                    row["fit_sha256"] == array_sha(a) and row["evaluation_sha256"] == array_sha(b), "CANONICAL_NESTED_ROW_ORDER_CHANGED")
            require(row["teacher_seed"] == 27011+100*outer+(31 if index is None else index), "R4_TEACHER_SEED_CHANGED")


def selected_epoch(receipt):
    rows = [r for r in receipt["history"] if "validation" in r]
    best = min(r["validation"]["program"]["patient_equal_masked_huber"] for r in rows)
    eligible = [r for r in rows if r["validation"]["program"]["patient_equal_masked_huber"] <= best+receipt["program_huber_tolerance"]]
    def gap(row):
        values = [row["validation"]["retrieval_hard"][direction]["paired_gap_mean"] for direction in ("image_to_gene", "gene_to_image")]
        values = [float(v) for v in values if v is not None and math.isfinite(v)]
        return math.fsum(values)/len(values) if values else -math.inf
    return min(eligible, key=lambda r: (-gap(r), r["epoch"]))["epoch"]


def verify_phase(private, lock, lock_sha, split_name, family, seed, patient, split):
    folder = Path(private) / "phase_a" / split_name / (family+"_seed"+str(seed))
    path = checked(folder, "cell.private.json")
    row = read(path)
    binding = read(checked(folder, "MATRIX_BINDING.private.json"))
    require(row["status"] == "COMPLETE" and row["identity"] == "NEW_FIT" and row["seed"] == seed and
            row["frozen_sources"] == lock["phase_a_lock"]["frozen_sources"] and row["master_lock_sha256"] == canonical(lock["phase_a_lock"]) and
            row["DSS_OS_PFI_accessed"] is False and row["P2_main_pretraining_eligible"] is True, "PHASE_A_SOURCE_OR_SCOPE_INVALID")
    definition = lock["phase_a_families"][family]
    require(row["variant"] == definition["name"] and all(row["spec"][k] == v for k, v in definition.items() if k != "name"),
            "PHASE_A_ARCHITECTURE_DEFINITION_CHANGED")
    require(binding["p2_lock_sha256"] == lock_sha and binding["split"] == split_name and binding["family"] == family and
            binding["seed"] == seed and binding["cell_receipt_sha256"] == sha(path) and
            binding["training_only_input_sha256"] == row["input_sha256"], "PHASE_A_MATRIX_BINDING_INVALID")
    # P1R partition metadata uses private canonical integer rows, not string IDs.
    require(row["train_rows_sha256"] == metadata_tensor_sha(np.asarray(split["fit"], np.int64)), "PHASE_A_CANONICAL_TRAIN_ORDER_CHANGED")
    prep = binding["shared_preprocessing"]
    checked(Path(private)/"preprocessing", prep["file"], prep["sha256"])
    steps = row["optimizer_steps"]
    require(row["train_patients"] == len(split["fit"]) and steps in CHECKPOINTS and len(row["history"]) == steps and
            [r["epoch"] for r in row["history"]] == list(range(1, steps+1)) and
            [r["optimizer_step"] for r in row["history"]] == list(range(1, steps+1)), "PHASE_A_OPTIMIZER_STEPS_INVALID")
    for epoch in row["history"]:
        gradients = list(epoch["gradient_norms"].values())
        require(gradients and all(v is not None and math.isfinite(v) and v > 0 for v in gradients), "PHASE_A_NONFINITE_OR_INACTIVE_GRADIENT")
        require(all(math.isfinite(epoch[key]) for key in ("train_program_loss", "train_pair_loss", "train_auxiliary_loss")),
                "PHASE_A_NONFINITE_TRAINING_OBJECTIVE")
    inner = split["role"] == "inner"
    if inner:
        require(steps == 400 and row["evaluate_valid"] is True and row["valid_patients"] == len(split["evaluation"]) and
                row["valid_rows_sha256"] == metadata_tensor_sha(np.asarray(split["evaluation"], np.int64)) and
                row["selected_epoch"] == selected_epoch(row), "INNER_PRETRAIN_SELECTION_INVALID")
    else:
        require(row["evaluate_valid"] is False and row["valid_patients"] == 0 and row["valid_rows_sha256"] is None and
                row["selected_epoch"] == row["fixed_epoch"] == steps and not any("validation" in r for r in row["history"]),
                "OUTER_PRETRAIN_EVALUATION_LEAK_OR_EPOCH_CHANGED")
    states = row["states"]
    require({r["epoch"] for r in states} == {e for e in CHECKPOINTS if e <= steps} and
            len(states) == len({r["epoch"] for r in states}), "CHECKPOINT_SET_INVALID")
    for state in states:
        checked(folder, state["file"], state["sha256"])
        require(state["actual_replay"] is True and state["replay_max_error"] == 0., "PRODUCER_ACTUAL_REPLAY_MARKER_INVALID")
    chosen = next(r for r in states if r["epoch"] == row["selected_epoch"])
    require(chosen["file"] == row["source_state_file"] and chosen["sha256"] == row["selected_checkpoint_sha256"], "SELECTED_CHECKPOINT_CHANGED")
    return {"split": split_name, "family": family, "seed": seed, "receipt_sha256": sha(path),
        "selected_epoch": row["selected_epoch"], "optimizer_steps": steps, "input_sha256": row["input_sha256"],
        "selected_state_sha256": chosen["sha256"],
        "checkpoint_hashes": [{"epoch": r["epoch"], "sha256": r["sha256"]} for r in states],
        "producer_actual_saved_replays_verified_as_receipt_markers": len(states), "new_model_forwards": 0}


def verify_head(folder, expected, phase, split, canonical_time, canonical_event, anchor):
    lam = expected["ridge_lambda"]
    path = checked(folder, "lambda"+str(float(lam))+".private.json")
    receipt = read(path)
    b = receipt["source_and_scope_bindings"]
    require(receipt["role"] == "outer_final" and receipt["status"] == "FITTED_AND_EVALUATED" and
            receipt["DSS_coefficients"] == 26 and receipt["risk_optimizer_fits"] == 1 and receipt["ridge_lambda"] == lam and
            all(b[k] == v for k, v in expected.items() if k != "ridge_lambda") and
            b["phase_a_selected_state_sha256"] == phase["selected_state_sha256"] and b["input_training_sha256"] == phase["input_sha256"],
            "OUTER_RISK_SCOPE_OR_SOURCE_INVALID")
    cell_path = checked(folder, path.with_suffix(".pkl").name, receipt["artifact_sha256"])
    cell = passive_pickle(cell_path)
    compact_path = checked(folder, "lambda"+str(float(lam))+".compact.private.pkl")
    c = passive_pickle(compact_path)
    for role, indices in (("train", split["fit"]), ("evaluation", split["evaluation"])):
        rows = np.asarray(indices, int)
        require(np.array_equal(c[role+"_time"], canonical_time[rows]) and np.array_equal(c[role+"_event"], canonical_event[rows]),
                "COMPACT_LABELS_NOT_CANONICAL_IN_SAME_ROW_ORDER")
    require(np.array_equal(c["training_offset"], anchor.training_offset) and np.array_equal(c["evaluation_offset"], anchor.evaluation_offset),
            "COMMON_ANCHOR_NOT_ELEMENTWISE_IDENTICAL")
    require(b["anchor_training_sha256"] == array_sha(anchor.training_offset), "RISK_ANCHOR_BINDING_CHANGED")
    optimizer = receipt["risk_optimizer"]
    for key, field in (("train_time", "fit_time_sha256"), ("train_event", "fit_event_sha256"),
                       ("train_design", "fit_design_sha256"), ("training_offset", "fit_offset_sha256")):
        require(array_sha(c[key]) == optimizer[field], "COMPACT_FIT_INPUT_HASH_CHANGED")
    for key, field in (("evaluation_time", "evaluation_time_sha256"), ("evaluation_event", "evaluation_event_sha256"),
                       ("evaluation_offset", "evaluation_offset_sha256")):
        require(array_sha(c[key]) == receipt[field], "COMPACT_EVALUATION_INPUT_HASH_CHANGED")
    x, z = np.asarray(c["train_design"], float), np.asarray(c["evaluation_design"], float)
    w = np.asarray(cell.model.coefficient, float)
    require(w.shape == (26,) and x.shape == (len(c["train_time"]), 26) and z.shape == (len(c["evaluation_time"]), 26) and
            np.isfinite(w).all() and np.isfinite(x).all() and np.isfinite(z).all() and (x.std(0) > 1e-10).all(), "DSS_26_ACTIVE_FEATURE_CONTRACT_INVALID")
    require(set(cell.design.blocks) == {"pathology","omics","clinical"}, "PERSISTED_MAIN_EFFECT_BLOCK_SET_CHANGED")
    for block in cell.design.blocks.values():
        values = np.asarray(block.receipt["retained_singular_values"])
        require(values.shape == (8,) and (values > 1e-5*max(values)).all() and
                block.receipt["effective_retained_rank"] == block.receipt["output_active_columns"] == 8,
                "PERSISTED_FEATURE_MAP_USES_RANK_DEFICIENT_NOISE_DIRECTIONS")
    fit_risk = np.asarray(c["training_offset"]) + x @ w
    eval_risk = np.asarray(c["evaluation_offset"]) + z @ w
    train_error = float(np.max(np.abs(fit_risk-cell.training_risk)))
    error = float(np.max(np.abs(eval_risk-cell.evaluation_risk)))
    require(max(train_error, error) <= 1e-12 and array_sha(cell.training_risk) == receipt["training_risk_sha256"] and
            array_sha(cell.evaluation_risk) == receipt["evaluation_risk_sha256"], "SAVED_LINEAR_RISK_ARITHMETIC_OR_HASH_CHANGED")
    require(close(cell.model.ridge_lambda, lam), "PERSISTED_MODEL_LAMBDA_CHANGED")
    objective, grad = cox_objective(x, w, c["training_offset"], c["train_time"], c["train_event"], lam)
    maximum_gradient = float(np.abs(grad).max())
    require(optimizer["parameters"] == optimizer["active_design_columns"] == 26 and optimizer["gradient_tolerance"] == 2e-5 and
            maximum_gradient <= 2e-5 and close(objective, optimizer["objective"]) and
            close(maximum_gradient, optimizer["gradient_max"]), "INDEPENDENT_EFRON_RIDGE_OBJECTIVE_OR_GRADIENT_CHANGED")
    metrics = independent_metrics(c["train_time"], c["train_event"], c["evaluation_time"], c["evaluation_event"], eval_risk)
    metric_error = check_metrics(metrics, receipt["evaluation_metrics"])
    check_metrics(metrics, cell.receipt["evaluation_metrics"])
    check_metrics(independent_metrics(c["train_time"], c["train_event"], c["train_time"], c["train_event"], fit_risk), receipt["train_metrics"])
    check_metrics(independent_metrics(c["train_time"], c["train_event"], c["evaluation_time"], c["evaluation_event"], c["evaluation_offset"]),
                  receipt["anchor_evaluation_metrics"])
    replay = receipt["saved_linear_head_replay"]
    require(replay["actual_linear_head_replay"] and not replay["neural_forward"] and replay["max_abs_risk_error"] <= 1e-12 and
            replay["fold_metric_components_equal"], "ORIGINAL_LINEAR_REPLAY_RECEIPT_INVALID")
    return {"receipt_sha256": sha(path), "risk_artifact_sha256": receipt["artifact_sha256"], "compact_artifact_sha256": sha(compact_path),
        "max_abs_evaluation_risk_error": error, "max_abs_training_risk_error": train_error,
        "independent_objective_abs_error": abs(objective-optimizer["objective"]),
        "independent_ridge_gradient_max": maximum_gradient, "metric_component_max_abs_error": metric_error,
        "metrics": metrics}, {"train_time": c["train_time"], "train_event": c["train_event"],
                             "time": c["evaluation_time"], "event": c["evaluation_event"], "risk": eval_risk, "expected": metrics}


def verify_inner_choice(private, outer, arm, family, seed, frozen, phases, lock_sha):
    summaries, bindings = [], []
    for lam in (.1, 1., 10.):
        metrics = []
        for inner in range(3):
            split_name = f"outer{outer}_inner{inner}"
            folder = Path(private)/"risk"/split_name/(arm+"_seed"+str(seed))
            path = checked(folder, "lambda"+str(lam)+".private.json")
            row = read(path)
            source = row["source_and_scope_bindings"]
            phase = phases[(split_name, family, seed)]
            require(source["lock_sha256"] == lock_sha and source["arm"] == arm and source["seed"] == seed and
                    source["split"] == split_name and source["phase_a_source_family"] == family and
                    source["phase_a_selected_state_sha256"] == phase["selected_state_sha256"] and
                    source["input_training_sha256"] == phase["input_sha256"] and row["ridge_lambda"] == lam and
                    row["role"] == "inner" and row["saved_linear_head_replay"]["fold_metric_components_equal"], "INNER_SELECTION_RECEIPT_SCOPE_CHANGED")
            checked(folder, path.with_suffix(".pkl").name, row["artifact_sha256"])
            metrics.append(row["evaluation_metrics"])
            bindings.append({"inner": inner, "lambda": lam, "receipt_sha256": sha(path), "artifact_sha256": row["artifact_sha256"]})
        hn, hd = sum(r["Harrell_num"] for r in metrics), sum(r["Harrell_den"] for r in metrics)
        supported = all(r["Uno60_valid"] and r["Uno_den"] is not None and r["Uno_den"] > 0 for r in metrics)
        un = sum(r["Uno_num"] for r in metrics) if supported else None
        ud = sum(r["Uno_den"] for r in metrics) if supported else None
        summaries.append({"ridge_lambda": lam, "Harrell": hn/hd, "Harrell_num": hn, "Harrell_den": hd,
            "Uno60": un/ud if supported else None, "Uno_num": un, "Uno_den": ud, "Uno60_complete_support": supported})
    selected = max(summaries, key=lambda r: (r["Harrell"], r["Uno60"] if r["Uno60"] is not None else -math.inf, r["ridge_lambda"]))
    require(frozen["bindings"] == bindings and frozen["ridge_lambda"] == selected["ridge_lambda"] and not frozen["uses_outer_test"] and
            frozen["candidate_summaries"] == summaries, "FROZEN_RIDGE_INNER_SELECTION_CHANGED")
    return len(bindings)


def aggregate(rows):
    result = {}
    for arm in sorted({r["arm"] for r in rows}):
        seeds = (31,) if arm == "B0" else SEEDS
        summaries = []
        for seed in seeds:
            group = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            require(len(group) == 5 and {r["fold"] for r in group} == set(FOLDS), "FIVE_FOLD_SEED_AGGREGATION_INCOMPLETE")
            value = {"seed": seed}
            for score, nk, dk in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
                supported = all(r["metrics"][nk] is not None and r["metrics"][dk] is not None and r["metrics"][dk] > 0 for r in group)
                num = math.fsum(r["metrics"][nk] for r in group) if supported else None
                den = math.fsum(r["metrics"][dk] for r in group) if supported else None
                value.update({score: num/den if supported else None, nk: num, dk: den})
            summaries.append(value)
        result[arm] = {"arm": arm, "seed_results": summaries}
        for metric in ("Harrell", "Uno60"):
            values = [r[metric] for r in summaries]
            result[arm][metric] = math.fsum(values)/len(values) if all(v is not None for v in values) else None
    return result


OFFICIAL_CHILD = r'''
import json,sys,importlib.metadata,numpy as np
from pathlib import Path
value=json.load(sys.stdin);sys.path.insert(0,value['target'])
import sksurv
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
if not Path(sksurv.__file__).resolve().is_relative_to(Path(value['target']).resolve()):raise RuntimeError('ISOLATED_PACKAGE_TARGET_NOT_USED')
version=importlib.metadata.version('scikit-survival')
if version!='0.23.1':raise RuntimeError('OFFICIAL_PACKAGE_VERSION_MISMATCH')
herror=uerror=0.;calls=na=0
for c in value['cases']:
    event=np.asarray(c['event'],bool);time=np.asarray(c['time'],float);risk=np.asarray(c['risk'],float)
    h=float(concordance_index_censored(event,time,risk,tied_tol=1e-8)[0]);calls+=1
    herror=max(herror,abs(h-c['expected']['Harrell']))
    if c['expected']['Uno60_valid'] and c['expected']['Uno60'] is not None:
        train=Surv.from_arrays(np.asarray(c['train_event'],bool),np.asarray(c['train_time'],float))
        test=Surv.from_arrays(event,time)
        u=float(concordance_index_ipcw(train,test,risk,tau=60.,tied_tol=1e-8)[0]);calls+=1
        uerror=max(uerror,abs(u-c['expected']['Uno60']))
    else:na+=1
if max(herror,uerror)>1e-10:raise RuntimeError('OFFICIAL_API_DISAGREES')
print(json.dumps({'status':'PASS_REAL_OFFICIAL_PACKAGE_METRICS_API','package_version':version,
    'cases':len(value['cases']),'actual_API_calls':calls,'Uno_NA_by_prelocked_support':na,
    'Harrell_max_abs_error':herror,'Uno60_max_abs_error':uerror,'model_forwards':0,'optimizer_updates':0}))
'''


def official_oracle(cases, target):
    serial = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in cases]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    child = subprocess.run([sys.executable, "-I", "-c", OFFICIAL_CHILD],
        input=json.dumps({"target": str(Path(target).resolve()), "cases": serial}, allow_nan=False),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=300)
    require(child.returncode == 0, "OFFICIAL_PACKAGE_SUBPROCESS_FAILED_NO_PRIVATE_STDERR_EXPORTED")
    value = json.loads(child.stdout)
    require(value["status"] == "PASS_REAL_OFFICIAL_PACKAGE_METRICS_API" and value["package_version"] == "0.23.1"
            and value["cases"] == len(cases), "OFFICIAL_API_CASE_COUNT_OR_PACKAGE_CHANGED")
    return value


def compare_report(report_dir, rows, summaries):
    published = []
    for path in sorted(Path(report_dir).glob("05_fold_seed_metrics*.json")):
        value = read(path)
        require(isinstance(value, list), "REPORT_FOLD_TABLE_FORMAT_INVALID")
        published.extend(value)
    index = {(r["arm"], r["fold"], r["seed"]): r for r in published}
    require(len(index) == len(published) == len(rows) and set(index) == {(r["arm"], r["fold"], r["seed"]) for r in rows}, "REPORT_FOLD_CELL_SET_CHANGED")
    maximum = 0.
    for row in rows:
        maximum = max(maximum, check_metrics(row["metrics"], index[(row["arm"], row["fold"], row["seed"])]))
    values = read(Path(report_dir)/"05_seed_aggregated_metrics.json")
    mapped = {r["arm"]: r for r in values}
    require(len(mapped) == len(values) and set(mapped) == set(summaries), "REPORT_AGGREGATE_ARM_SET_CHANGED")
    for arm, expected in summaries.items():
        actual = mapped[arm]
        for metric in ("Harrell", "Uno60"):
            require(close(expected[metric], actual[metric]), "REPORT_EQUAL_SEED_AGGREGATION_CHANGED")
        require(len(expected["seed_results"]) == len(actual["seed_results"]), "REPORT_SEED_COUNT_CHANGED")
        for expected_seed in expected["seed_results"]:
            actual_seed = next(r for r in actual["seed_results"] if r["seed"] == expected_seed["seed"])
            check_metrics(expected_seed, actual_seed)
    paired = []
    for path in sorted(Path(report_dir).glob("06_paired_comparisons*.json")):
        value = read(path)
        require(isinstance(value, list), "REPORT_COMPARISON_TABLE_FORMAT_INVALID")
        paired.extend(value)
    # A numerical check of only the rows that happen to exist would silently
    # accept a deleted comparison table. Independently bind the complete set.
    expected_pairs = {p for p in CORE_COMPARISONS if p[0] in summaries and p[1] in summaries}
    actual_pairs = [(r["left"], r["reference"]) for r in paired]
    require(len(actual_pairs) == len(expected_pairs) and set(actual_pairs) == expected_pairs,
            "REPORT_REQUIRED_COMPARISON_SET_CHANGED")
    for row in paired:
        require(not row["confidence_intervals_computed"] and row["bootstrap_replicates"] == 0 and not row["p_value_computed"], "UNREQUESTED_INTERVAL_OR_PVALUE")
        for metric in ("Harrell", "Uno60"):
            a, b = summaries[row["left"]][metric], summaries[row["reference"]][metric]
            require(close(None if a is None or b is None else a-b, row[metric+"_delta"]), "REPORT_PAIRED_POINT_ESTIMATE_CHANGED")
    return {"status": "PASS_INDEPENDENT_COMPONENTS_AND_MICRO_SEED_AGGREGATION", "fold_rows": len(rows),
            "arms": len(summaries), "required_comparisons_verified": len(expected_pairs),
            "missing_or_duplicate_comparisons_allowed": False,
            "maximum_fold_component_error": maximum, "reporter_imported_or_called": False}


def public_write(path, value):
    def validate(x):
        require(not isinstance(x, np.ndarray), "PRIVATE_VECTOR_IN_PUBLIC_AUDIT")
        if isinstance(x, dict):
            for k, v in x.items():
                require(k not in {"patient_ids", "rowids", "train_indices", "test_indices", "risk_vector", "predictions", "weights", "coefficients"}, "PRIVATE_PUBLIC_KEY")
                validate(v)
        elif isinstance(x, list):
            require(len(x) <= 1000 and not (len(x) > 128 and all(isinstance(v, (int, float)) for v in x)), "UNBOUNDED_PUBLIC_LIST")
            for v in x:
                validate(v)
        elif isinstance(x, float):
            require(math.isfinite(x), "NONFINITE_PUBLIC_SCALAR")
    validate(value)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    text = raw.decode()
    require(len(raw) <= 2*1024**2 and not re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", text) and
            not re.search(r"(?:/root/|/home/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]|hf_[A-Za-z0-9]{20})", text), "PUBLIC_AUDIT_PRIVACY_OR_SIZE_FAILED")
    with Path(path).open("xb") as f:
        f.write(raw)


def verify_run(private, lock_path, expected_lock_sha256, output, *, cohort_path, endpoint_path, report_dir,
               sksurv_target=None, allow_completed_outer_results=False):
    require(allow_completed_outer_results, "POSTCOMPLETION_EXPLICIT_AUTHORIZATION_REQUIRED_BEFORE_ANY_INPUT_READ")
    require(sha(lock_path) == expected_lock_sha256, "MASTER_LOCK_BYTES_CHANGED")
    lock = read(lock_path)
    require(lock["status"] == "LOCKED_BEFORE_OUTER_EVALUATION", "MASTER_LOCK_NOT_PREFIT")
    require(lock["assets"]["cohort_sha256"] == COHORT_SHA and lock["assets"]["endpoints_sha256"] == ENDPOINT_SHA,
            "MASTER_LOCK_CANONICAL_ASSET_BINDING_CHANGED")
    for name, digest in lock["frozen_sources"].items():
        checked(ROOT, name, digest)
    require(not Path(output).exists(), "NEW_APPEND_ONLY_AUDIT_DIRECTORY_REQUIRED")
    patient, folds, time, event = canonical_labels(cohort_path, endpoint_path)
    plan_path = checked(private, "SPLITS.private.json")
    plan = read(plan_path)
    require(plan["p2_lock_sha256"] == expected_lock_sha256, "SPLITS_LOCK_BINDING_CHANGED")
    verify_splits(plan, lock, patient, folds, time, event)
    selection_path = checked(private, "ALL_OUTER_SELECTIONS_FROZEN.private.json")
    selection = read(selection_path)
    barrier = read(checked(private, "ALL_OUTER_SELECTION_BARRIER_BINDING.private.json"))
    require(selection["p2_lock_sha256"] == expected_lock_sha256 and selection["outer_results_read"] is False and
            selection["status"] == "ALL_FIVE_OUTERS_SELECTED_BEFORE_FIRST_OUTER_SCORE" and
            barrier["p2_lock_sha256"] == expected_lock_sha256 and barrier["selection_file_sha256"] == sha(selection_path),
            "GLOBAL_SELECTION_BARRIER_CHANGED")
    arm_family = {arm: family for family, arms in lock["family_readouts"].items() for arm in arms}
    require(len(arm_family) == sum(map(len, lock["family_readouts"].values())), "DUPLICATE_ARM_FAMILY")
    phases, all_rows, official_cases, outers, inner_receipts = {}, [], [], {}, 0
    for outer in FOLDS:
        phase_rows = []
        for index in (*range(3), None):
            split_name = f"outer{outer}_" + ("final" if index is None else f"inner{index}")
            split = plan["partitions"][split_name]
            for family in lock["family_readouts"]:
                for seed in SEEDS:
                    phase = verify_phase(private, lock, expected_lock_sha256, split_name, family, seed, patient, split)
                    phases[(split_name, family, seed)] = phase
                    phase_rows.append(phase)
        split_name = f"outer{outer}_final"
        split = plan["partitions"][split_name]
        anchor_path = checked(private, f"anchors/{split_name}.private.json")
        ar = read(anchor_path)
        anchor = passive_pickle(checked(private, f"anchors/{split_name}.private.pkl", ar["artifact_sha256"]))
        require(ar["p2_lock_sha256"] == expected_lock_sha256 and ar["split"] == split_name and ar["old_scores_reused"] is False and
                ar["Cox_optimizer_fits"] == 52 and ar["scalar_pair_optimizer_fits"] == 4 and
                ar["full_pipeline_crossfit_for_training_offset"] is True and ar["beta_supervision_excluded_for_each_downstream_training_patient"] is True and
                ar["shared_full_pipeline_is_B0_exact_source"] is True and ar["evaluation_is_original_full_fit_R4_recipe"] is True and
                ar["evaluation_labels_available_to_teachers"] is False, "ANCHOR_FULL_PIPELINE_SOURCE_INVALID")
        require(ar["teacher_seed"] == split["teacher_seed"] and ar["pipeline_crossfit_seed"] == split["teacher_seed"] and
                ar["legacy_source_sha256"] == lock["B0_R4_EXACT"]["legacy_source_sha256"] and
                ar["full_CONCH_mean_sha256"] == lock["B0_R4_EXACT"]["full_CONCH_mean_sha256"] and
                ar["legacy_eight_program_union_sha256"] == lock["B0_R4_EXACT"]["legacy_eight_program_union_sha256"],
                "ORIGINAL_R4_FEATURE_OR_SEED_LINEAGE_CHANGED")
        require(np.array_equal(anchor.train_rows, split["fit"]) and np.array_equal(anchor.evaluation_rows, split["evaluation"]) and
                np.array_equal(anchor.b0_exact_evaluation, anchor.evaluation_offset) and
                array_sha(anchor.training_offset) == ar["train_offset_sha256"] and array_sha(anchor.evaluation_offset) == ar["evaluation_offset_sha256"],
                "ANCHOR_ORDER_OR_RISK_HASH_CHANGED")
        require(ar["fit_hash"] == ar["calls"][0]["fit_hash"] == array_sha(np.asarray(split["fit"],np.int64)) and
                ar["evaluation_hash"] == ar["calls"][0]["held_hash"] == array_sha(np.asarray(split["evaluation"],np.int64)),
                "FULL_R4_TEACHER_FIT_EVAL_ROW_BINDING_CHANGED")
        require(len(anchor.state["crossfit"]) == 3 and len(ar["calls"]) == 4 and
                close(ar["full_beta"], anchor.state["full"]["beta"]), "FULL_PIPELINE_BETA_OR_CROSSFIT_STATES_CHANGED")
        held_all = []
        expected_blocks = np.array_split(np.random.default_rng(split["teacher_seed"]).permutation(np.asarray(split["fit"],np.int64)),3)
        for group, state in enumerate(anchor.state["crossfit"]):
            fit, held = np.asarray(state["fit_rows"], int), np.asarray(state["held_rows"], int)
            require(np.array_equal(held,expected_blocks[group]) and not np.intersect1d(fit, held).size and np.array_equal(fit, np.setdiff1d(split["fit"], held)) and
                    ar["calls"][group+1]["fit_hash"] == array_sha(fit) and ar["calls"][group+1]["held_hash"] == array_sha(held),
                    "DOWNSTREAM_TEACHER_OWN_LABEL_EXCLUSION_INVALID")
            held_all.extend(held.tolist())
        require(np.array_equal(np.sort(held_all), split["fit"]), "FULL_PIPELINE_CROSSFIT_TRAIN_COVERAGE_CHANGED")
        fold_audit = {"outer": outer, "phase_a": phase_rows, "risk_cells": [], "anchor_receipt_sha256": sha(anchor_path),
                      "anchor_artifact_sha256": ar["artifact_sha256"], "common_anchor_verified_elementwise": True}
        teacher_gradients, scalar_gradients = [], []
        for pipeline in [anchor.state["full"]]+[s["state"] for s in anchor.state["crossfit"]]:
            model_receipts = [m["optimizer"] for m in pipeline["full_models"].values()]
            model_receipts += [m["optimizer"] for teacher in pipeline["teachers"] for m in teacher["models"].values()]
            require(len(model_receipts) == 13 and all(math.isfinite(m["loss"]) and 0 <= m["gradient_max"] <= 2e-5 for m in model_receipts),
                    "RECORDED_R4_COX_OPTIMIZER_CONVERGENCE_OR_COUNT_INVALID")
            teacher_gradients.extend(float(m["gradient_max"]) for m in model_receipts)
            scalar_gradients.append(float(pipeline["scalar_audit"]["gradient_max"]))
        require(len(teacher_gradients) == 52 and len(scalar_gradients) == 4 and all(0 <= g <= 1e-6 for g in scalar_gradients),
                "RECORDED_R4_SCALAR_OPTIMIZER_CONVERGENCE_INVALID")
        fold_audit["recorded_teacher_optimizer_gradients_checked"] = {"Cox_fits":52,"scalar_fits":4,
            "Cox_gradient_max":max(teacher_gradients),"scalar_gradient_max":max(scalar_gradients),
            "old_teacher_objectives_not_independently_refit_or_replayed_from_raw_features":True}
        for arm, family in arm_family.items():
            for seed in SEEDS:
                phase = phases[(split_name, family, seed)]
                inner_epochs = [phases[(f"outer{outer}_inner{i}", family, seed)]["selected_epoch"] for i in range(3)]
                chosen = selection["entries"][str(outer)]["pretraining"][family][str(seed)]
                require(chosen["epoch"] == phase["selected_epoch"] == sorted(inner_epochs)[1] and
                        chosen["inner_selected_epochs"] == inner_epochs and chosen["bindings"] == [
                            {"inner": i, "receipt_sha256": phases[(f"outer{outer}_inner{i}", family, seed)]["receipt_sha256"],
                             "selected_state_sha256": phases[(f"outer{outer}_inner{i}", family, seed)]["selected_state_sha256"]} for i in range(3)],
                        "FIXED_OUTER_EPOCH_NOT_MEDIAN_OF_INNER_SELECTIONS")
                risk_choice = selection["entries"][str(outer)]["risk"][arm][str(seed)]
                require(risk_choice["uses_outer_test"] is False, "OUTER_TEST_USED_FOR_RIDGE_SELECTION")
                inner_receipts += verify_inner_choice(private, outer, arm, family, seed, risk_choice, phases, expected_lock_sha256)
                expected = {"lock_sha256": expected_lock_sha256, "split": split_name, "arm": arm, "seed": seed,
                            "phase_a_source_family": family, "ridge_lambda": risk_choice["ridge_lambda"]}
                folder = Path(private)/"risk"/split_name/(arm+"_seed"+str(seed))
                audit, case = verify_head(folder, expected, phase, split, time, event, anchor)
                record = {"arm": arm, "fold": outer, "seed": seed, **audit}
                fold_audit["risk_cells"].append(record)
                all_rows.append(record)
                official_cases.append(case)
        b0_path = checked(private, f"B0/{split_name}.private.json")
        b0 = read(b0_path)
        require(b0["source_anchor_sha256"] == ar["artifact_sha256"] and b0["deterministic_single_refit"] is True and
                b0["historical_three_seed_average_reproduced"] is False and b0["teacher_seed"] == split["teacher_seed"] and b0["seed"] == 31,
                "B0_NOT_SHARED_TRUE_FULL_TEACHER_SOURCE")
        tr, ev = np.asarray(split["fit"], int), np.asarray(split["evaluation"], int)
        metric = independent_metrics(time[tr], event[tr], time[ev], event[ev], anchor.b0_exact_evaluation)
        check_metrics(metric, b0["DSS_metrics"])
        record = {"arm": "B0", "fold": outer, "seed": 31, "metrics": metric, "receipt_sha256": sha(b0_path)}
        all_rows.append(record)
        fold_audit["risk_cells"].append(record)
        official_cases.append({"train_time": time[tr], "train_event": event[tr], "time": time[ev], "event": event[ev],
                               "risk": anchor.b0_exact_evaluation, "expected": metric})
        outers[outer] = fold_audit
    summaries = aggregate(all_rows)
    report_audit = compare_report(report_dir, all_rows, summaries)
    oracle = official_oracle(official_cases, sksurv_target) if sksurv_target is not None else {"status": "NOT_RUN_NO_ISOLATED_PACKAGE_TARGET_SUPPLIED"}
    output = Path(output)
    output.mkdir(parents=True)
    expected_phase = len(lock["family_readouts"])*20*len(SEEDS)
    require(len(phases) == expected_phase and len(all_rows) == len(arm_family)*15+5, "FINAL_MATRIX_CARDINALITY_CHANGED")
    global_audit = {"phase_a_receipts_verified": len(phases), "expected_phase_a_receipts": expected_phase,
        "phase_a_optimizer_steps_verified": sum(p["optimizer_steps"] for p in phases.values()),
        "actual_checkpoint_files_byte_hash_verified": sum(len(p["checkpoint_hashes"]) for p in phases.values()),
        "checkpoint_policy": "four_inner_states_outer_states_only_up_to_the_inner_selected_fixed_epoch",
        "risk_vectors_independently_recomputed": len(all_rows), "new_neural_model_forwards": 0, "new_optimizer_updates": 0,
        "inner_risk_selection_receipt_hashes_and_component_arithmetic_verified": inner_receipts,
        "inner_risk_vectors_independently_rescored": False,
        "H5_features_or_encoder_reconstruction_performed": False, "neural_checkpoints_deserialized": False,
        "evidence_type": "independent_saved_linear_risk_math_and_report_arithmetic_and_producer_checkpoint_byte_hashes",
        "producer_replay_flags_are_not_second_independent_neural_replays": True,
        "DSS_only_labels_read": True, "PFI_OS_arrays_read": False, "official_package_API": oracle,
        "report_comparison": report_audit, "seed_aggregates": list(summaries.values())}
    files = []
    for outer, audit in outers.items():
        value = {"schema": "semantic_b0b7_p2_independent_audit_v1", "status": "PASS_INDEPENDENT_SAVED_RISK_SCORER_AND_HASH_AUDIT",
            "master_lock_sha256": expected_lock_sha256, "verifier_source_sha256": sha(__file__),
            "frozen_scientific_sources_verified": len(lock["frozen_sources"]), "split_plan_sha256": sha(plan_path),
            "all_outer_selection_sha256": sha(selection_path), "cohort_sha256": COHORT_SHA, "endpoint_sha256": ENDPOINT_SHA,
            "confidence_intervals_computed": False, "p_values_computed": False, "bootstrap_replicates": 0,
            "official_definition_sources": OFFICIAL_SOURCES, **audit}
        if outer == 1:
            value["global_summary"] = global_audit
        path = output/f"INDEPENDENT_P2_OUTER{outer}_AUDIT.json"
        public_write(path, value)
        files.append({"file": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
    return {"status": "PASS_INDEPENDENT_SAVED_RISK_SCORER_AND_HASH_AUDIT", "files": files,
            "independent_risk_vectors": len(all_rows), "new_neural_forwards": 0, "new_optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("private", "lock", "expected-lock-sha256", "output", "cohort", "dss-endpoints", "report-dir"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--sksurv-target")
    parser.add_argument("--allow-completed-outer-results", action="store_true")
    args = parser.parse_args()
    # Restrict the separate verifier process; never changes the running scheduler.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(verify_run(args.private, args.lock, args.expected_lock_sha256, args.output,
        cohort_path=args.cohort, endpoint_path=args.dss_endpoints, report_dir=args.report_dir,
        sksurv_target=args.sksurv_target, allow_completed_outer_results=args.allow_completed_outer_results), sort_keys=True))


if __name__ == "__main__":
    main()

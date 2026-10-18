"""Read-only audit and evaluation of locked frozen-kernel CDEI readouts.

Patient arrays stay in private checkpoints or an explicit private OOF archive.
Every expected receipt, byte hash and readout replay is checked before the
endpoint loader is called. This module never fits a predictor or preprocessor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.phase10_trimodal.reference import metrics, uno_weights
from src.phase21_relative.evaluation import (STATISTICS, TRANSITIONS,
                                            pair_transition_summary,
                                            patient_shared_transition_bootstrap)


SEEDS = (31, 47, 83)
FOLDS = (1, 2, 3, 4, 5)
ARMS = ("REL", "DUAL", "ADD", "NO_C", "WRONG_O", "WRONG_C", "MEAN", "CENTERED")
SENSITIVITIES = ("DUAL_ABS_OFF", "DUAL_WRONG_O", "DUAL_WRONG_C", "DUAL_NO_C")
PRIMARY = ("REL", "ADD", "NO_C", "WRONG_O", "WRONG_C")
METRICS = ("Harrell", "Uno60")
REPLAY_ATOL = 1e-12
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 2300181


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluator_identity():
    """Evaluation bytes are bound separately from the training identity."""
    root = Path(__file__).resolve().parents[2]
    names = ("src/phase23_cdei/evaluation.py", "scripts/evaluate_phase23_cdei.py",
             "src/phase10_trimodal/reference.py", "src/phase21_relative/evaluation.py",
             "src/phase13_rna/data.py")
    files = {name: sha256_file(root / name) for name in names}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _array(value, name, *, shape=None, integer=False):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    x = np.asarray(value)
    require(not np.iscomplexobj(x) and np.issubdtype(x.dtype, np.number), name + " must be real numeric")
    require(np.isfinite(x).all(), name + " contains nonfinite values")
    if shape is not None:
        require(x.shape == shape, name + " shape mismatch")
    if integer:
        require(np.issubdtype(x.dtype, np.integer), name + " must retain integer indices")
        return np.array(x, np.int64, copy=True)
    return np.array(x, np.float64, copy=True)


def _replay(actual, expected, name, audit):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + " replay shape mismatch")
    error = float(np.max(np.abs(actual - expected), initial=0.))
    require(np.isfinite(error) and error <= REPLAY_ATOL, name + " replay changed")
    audit["maximum_elementwise_error"] = max(audit["maximum_elementwise_error"], error)
    audit["elementwise_exact"] = audit["elementwise_exact"] and np.array_equal(actual, expected)
    audit["vectors_replayed"] += 1


def _validate_readouts(checkpoint, *, role, seed, fold, source, cohort_size):
    for key, expected in (("source", source), ("role", role), ("seed", seed), ("fold", fold)):
        require(checkpoint.get(key) == expected, "readout metadata mismatch: " + key)
    fit = _array(checkpoint["fit"], "fit", integer=True)
    valid = _array(checkpoint["valid"], "valid", integer=True)
    for name, x in (("fit", fit), ("valid", valid)):
        require(x.ndim == 1 and len(x) > 0 and len(np.unique(x)) == len(x), name + " must be nonempty unique patients")
        require((x >= 0).all() and (x < cohort_size).all(), name + " patient index out of range")
    require(not np.intersect1d(fit, valid).size, "fit/heldout patient leakage")
    if role == "outer":
        require(len(np.union1d(fit, valid)) == cohort_size, "outer fit/test does not partition the locked cohort")
    anchor_fit = _array(checkpoint["anchor_fit"], "anchor_fit", shape=(len(fit),))
    anchor_valid = _array(checkpoint["anchor_valid"], "anchor_valid", shape=(len(valid),))
    cs_valid = _array(checkpoint["J_CS_valid"], "J_CS_valid", shape=(len(valid),))
    require(set(checkpoint["arms"]) == set(ARMS), "eight locked readout arms are required")
    require(set(checkpoint["sensitivities"]) == set(SENSITIVITIES), "four same-head interventions are required")
    audit = {"vectors_replayed": 0, "maximum_elementwise_error": 0., "elementwise_exact": True,
             "fit_preprocessing_replayed": True, "relative_columns_identical": True}
    arms = {}
    for arm in ARMS:
        record = checkpoint["arms"][arm]
        dimensions = 64 if arm == "REL" else 66
        fit_x = _array(record["fit_features"], arm + " fit_features", shape=(len(fit), dimensions))
        valid_x = _array(record["valid_features"], arm + " valid_features", shape=(len(valid), dimensions))
        weights = _array(record["weights"], arm + " weights", shape=(dimensions,))
        scaler = record["scaler"]
        mean = _array(scaler["mean"], arm + " mean", shape=(dimensions,))
        scale = _array(scaler["scale"], arm + " scale", shape=(dimensions,))
        require((scale > 0).all(), arm + " nonpositive scale")
        require(int(scaler.get("fit_n", len(fit))) == len(fit), arm + " preprocessing fit count changed")
        raw_sd = fit_x.std(0, ddof=0)
        constant = raw_sd < 1e-8
        expected_scale = np.where(constant, 1., raw_sd)
        _replay(mean, fit_x.mean(0), arm + " fit-only mean", audit)
        _replay(scale, expected_scale, arm + " fit-only scale", audit)
        if "constant" in scaler:
            require(np.array_equal(np.asarray(scaler["constant"], bool), constant), arm + " constant-column state changed")
        fit_scaled, valid_scaled = (fit_x - mean) / scale, (valid_x - mean) / scale
        fit_risk = _array(record["fit_risk"], arm + " fit_risk", shape=(len(fit),))
        risk = _array(record["risk"], arm + " risk", shape=(len(valid),))
        _replay(fit_risk, anchor_fit + fit_scaled @ weights, arm + " fit risk", audit)
        _replay(risk, anchor_valid + valid_scaled @ weights, arm + " heldout risk", audit)
        if arm != "REL":
            require(np.array_equal(fit_x[:, :64], arms["REL"]["fit_features"]), arm + " fit relative path changed")
            require(np.array_equal(valid_x[:, :64], arms["REL"]["valid_features"]), arm + " heldout relative path changed")
        diagnostic = record.get("diagnostic", {})
        require(diagnostic.get("status") == "CONVERGED", arm + " convex fit not converged")
        require(float(diagnostic.get("gradient_max", np.inf)) <= 1e-7, arm + " gradient gate failed")
        require(float(diagnostic.get("penalty", np.nan)) == 10., arm + " penalty drift")
        require(int(diagnostic.get("parameters", -1)) == dimensions, arm + " readout capacity drift")
        require(diagnostic.get("offset_trainable") is False and diagnostic.get("intercept") is False,
                arm + " offset/intercept contract changed")
        arms[arm] = {"risk": risk, "fit_risk": fit_risk, "fit_features": fit_x,
                     "valid_features": valid_x, "weights": weights, "mean": mean,
                     "scale": scale, "feature_rank": int(np.linalg.matrix_rank(fit_scaled)),
                     "gradient_max": float(diagnostic["gradient_max"])}
    dual = arms["DUAL"]
    relative_only = (dual["valid_features"] - dual["mean"]) / dual["scale"]
    relative_only[:, -2:] = 0.
    sensitivity = {}
    for name in SENSITIVITIES:
        actual = _array(checkpoint["sensitivities"][name], name, shape=(len(valid),))
        if name == "DUAL_ABS_OFF":
            expected = anchor_valid + relative_only @ dual["weights"]
        else:
            context_arm = name[len("DUAL_"):]
            context = (arms[context_arm]["valid_features"] - dual["mean"]) / dual["scale"]
            expected = anchor_valid + context @ dual["weights"]
        _replay(actual, expected, name + " same-head risk", audit)
        sensitivity[name] = actual
    return {"fit": fit, "valid": valid, "anchor_fit": anchor_fit, "anchor_valid": anchor_valid,
            "J_CS_valid": cs_valid, "arms": arms, "sensitivities": sensitivity, "replay": audit,
            "role": role, "seed": seed, "fold": fold}


def verify_stage(private_root, role, source, *, cohort_size=512):
    """Verify all 15 cells before exposing any endpoint-loading callback.

    Hash names in DONE refer to artifact bytes, not an unpersisted state_dict.
    This audits saved head arithmetic; it does not claim encoder feature replay.
    """
    require(role in ("inner", "outer"), "stage must be inner or outer")
    require(isinstance(source, str) and len(source) == 64 and all(c in "0123456789abcdef" for c in source),
            "expected source must be a SHA256")
    root = Path(private_root)
    receipts = []
    # Preflight every expected file and byte hash first. No partial-stage result.
    for seed in SEEDS:
        for fold in FOLDS:
            cell_dir = root / role / f"seed{seed}_fold{fold}"
            receipt_path = cell_dir / "DONE_PRIVATE.json"
            require(receipt_path.is_file(), f"missing complete {role} cell: seed{seed}/fold{fold}")
            receipt = json.loads(receipt_path.read_text(encoding="utf8"))
            for key, expected in (("status", "COMPLETE"), ("source", source), ("role", role),
                                  ("seed", seed), ("fold", fold), ("kernel_epochs", 80), ("readout_fits", 8)):
                require(receipt.get(key) == expected, f"receipt mismatch {key}: seed{seed}/fold{fold}")
            for key, name in (("kernel_sha256", "kernel.pt"), ("readouts_sha256", "readouts.pt")):
                path = cell_dir / name
                nested = receipt.get("artifacts", {}).get(name)
                direct = receipt.get(key)
                require(direct is not None or nested is not None, f"{name} SHA256 receipt is missing")
                if direct is not None and nested is not None:
                    require(direct == nested, f"{name} conflicting SHA256 receipt fields")
                require(path.is_file() and sha256_file(path) == (direct if direct is not None else nested),
                        f"{name} hash mismatch")
            receipts.append((seed, fold, cell_dir, sha256_file(receipt_path)))
    cells, rows = {}, []
    for seed, fold, cell_dir, receipt_hash in receipts:
        # Our own verified private artifact may contain numpy objects.
        checkpoint = torch.load(cell_dir / "readouts.pt", map_location="cpu", weights_only=False)
        cell = _validate_readouts(checkpoint, role=role, seed=seed, fold=fold, source=source,
                                  cohort_size=cohort_size)
        cells[seed, fold] = cell
        rows.append({"seed": seed, "fold": fold, "role": role, "receipt_sha256": receipt_hash,
                     **cell["replay"], "readout_coefficients": {arm: len(cell["arms"][arm]["weights"]) for arm in ARMS},
                     "fit_feature_ranks": {arm: cell["arms"][arm]["feature_rank"] for arm in ARMS}})
    # Patient partitions and J0/J_CS are fixed across algorithm seeds.
    for fold in FOLDS:
        first = cells[SEEDS[0], fold]
        for seed in SEEDS[1:]:
            cell = cells[seed, fold]
            for key in ("fit", "valid", "anchor_fit", "anchor_valid", "J_CS_valid"):
                require(np.array_equal(first[key], cell[key]), "partition/frozen anchor differs across seeds: " + key)
    if role == "outer":
        indices = np.concatenate([cells[SEEDS[0], fold]["valid"] for fold in FOLDS])
        require(np.array_equal(np.sort(indices), np.arange(cohort_size)), "outer tests must cover every patient exactly once")
    fingerprint = hashlib.sha256("".join(q[3] for q in receipts).encode()).hexdigest()
    audit = {"status": "PASS_SAVED_READOUT_REPLAY", "source": source, "role": role,
             "complete_cells": 15, "kernel_fits": 15, "kernel_epochs_each": 80, "readout_fits": 120,
             "receipt_fingerprint": fingerprint, "rows": rows,
             "maximum_elementwise_error": max(q["maximum_elementwise_error"] for q in rows),
             "all_replayed_values_elementwise_exact": all(q["elementwise_exact"] for q in rows),
             "replay_absolute_tolerance": REPLAY_ATOL, "fit_only_preprocessing_replayed": True,
             "encoder_feature_replay_claimed": False, "endpoint_loaded_during_artifact_checks": False,
             "private_patient_arrays_public": False}
    return cells, audit


def _endpoint(value, cohort_size, *, require_complete):
    time, event = (np.asarray(value[name], np.float64) for name in ("time", "event"))
    require(time.shape == event.shape == (cohort_size,), "invalid endpoint shape")
    valid = np.asarray(value["valid"])
    require(valid.shape == (cohort_size,) and valid.dtype == np.dtype(bool), "invalid endpoint availability")
    require(np.isfinite(time[valid]).all() and np.isfinite(event[valid]).all()
            and np.isin(event[valid], [0, 1]).all() and (time[valid] >= 0).all(), "invalid endpoint labels")
    if require_complete:
        require(valid.all(), "primary DSS endpoint cannot silently exclude a locked patient")
    return time, event, valid


def point_results(cells, endpoint, *, cohort_size=512, require_complete=True):
    """Pool five within-fold numerators/denominators before seed comparison.

    Inner validation membership can overlap between outer folds. Its repeated
    patients are never called a complete OOF cohort or independent replicates.
    """
    t, e, available = _endpoint(endpoint, cohort_size, require_complete=require_complete)
    fold_rows, seed_rows = [], []
    for seed in SEEDS:
        totals = {}
        for fold in FOLDS:
            cell = cells[seed, fold]
            fit, ix = cell["fit"], cell["valid"]
            fit = fit[available[fit]]
            keep = available[ix]
            require(len(fit) and keep.any(), "endpoint has no fit or heldout patients in a locked fold")
            q = ix[keep]
            w = uno_weights(t[fit], e[fit], t[q], e[q], tau=60.)
            risks = {arm: cell["arms"][arm]["risk"] for arm in ARMS}
            risks.update(J0=cell["anchor_valid"], J_CS=cell["J_CS_valid"], **cell["sensitivities"])
            for arm, risk in risks.items():
                value = metrics(t[q], e[q], risk[keep], np.full(len(q), fold), w)
                require(all(np.isfinite(v) for v in value.values()), "undefined heldout concordance")
                fold_rows.append({"seed": seed, "fold": fold, "arm": arm,
                                  "heldout_patients": len(q), "heldout_events": int(e[q].sum()), **value})
                total = totals.setdefault(arm, {key: 0. for key in ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")})
                for key in total:
                    total[key] += value[key]
        for arm, total in totals.items():
            seed_rows.append({"seed": seed, "arm": arm, **total,
                              "Harrell": total["Harrell_num"] / total["Harrell_den"],
                              "Uno60": total["Uno_num"] / total["Uno_den"]})
    means = [{"arm": arm, **{metric: float(np.mean([r[metric] for r in seed_rows if r["arm"] == arm]))
                                for metric in METRICS}} for arm in [*ARMS, "J0", "J_CS", *SENSITIVITIES]]
    return {"fold_rows": fold_rows, "seed_rows": seed_rows, "three_seed_means": means,
            "aggregation": "each seed sums five within-fold numerator/denominator pairs, then three complete seed statistics are averaged",
            "risk_ensembling": False, "confidence_intervals": "not_provided_in_this_point_summary"}


def inner_gate(seed_rows, *, source, receipt_fingerprint, evaluator_sha256=None):
    by = {(r["seed"], r["arm"]): r for r in seed_rows}
    rows, anchor_rows = [], []
    for control in PRIMARY:
        for metric in METRICS:
            delta = [by[seed, "DUAL"][metric] - by[seed, control][metric] for seed in SEEDS]
            rows.append({"comparison": "DUAL-" + control, "metric": metric,
                         "mean_seed_delta": float(np.mean(delta)), "positive_seeds": int(np.sum(np.asarray(delta) > 0)),
                         "seeds": 3, "pass": bool(np.mean(delta) > 0 and np.sum(np.asarray(delta) > 0) >= 2)})
    for metric in METRICS:
        delta = [by[seed, "DUAL"][metric] - by[seed, "J0"][metric] for seed in SEEDS]
        anchor_rows.append({"comparison": "DUAL-J0", "metric": metric,
                            "mean_seed_delta": float(np.mean(delta)), "pass": bool(np.mean(delta) >= 0)})
    passed = all(r["pass"] for r in rows + anchor_rows)
    return {"status": "INNER_GATE_PASS" if passed else "NOT_SUPPORTED_IN_THIS_FIXED_KERNEL_READOUT",
            "source": source, "role": "inner", "receipt_fingerprint": receipt_fingerprint,
            "evaluator_sha256": evaluator_sha256 or evaluator_identity(),
            "mechanism_tests": rows, "anchor_noninferiority_screen": anchor_rows,
            "criterion": "all five DUAL-control comparisons: both metrics mean>0 and positive in at least 2/3 seeds; DUAL-J0 both metric means>=0",
            "outer_admitted": passed, "confirmatory_confidence_intervals": False,
            "inner_validation_overlap_allowed": True, "scope": "diagnostic fixed-kernel screen; not proof about all end-to-end CDEI designs",
            "endpoint": "DSS", "automatic_architecture_or_hyperparameter_search": False}


def inner_pair_transitions(cells, endpoint, point, *, cohort_size=512):
    """Descriptive paired credit only; no bootstrap, CI or new decision gate.

    Inner validation patients may recur across outer folds. Compare within each
    cell, add its weighted numerators/denominators, then form each complete seed
    statistic. Averaging independent fold C values would be a different object.
    """
    t, e, _ = _endpoint(endpoint, cohort_size, require_complete=True)
    controls = [*PRIMARY, "MEAN", "CENTERED", "J0", "J_CS", *SENSITIVITIES]
    by_seed = {(row["seed"], row["arm"]): row for row in point["seed_rows"]}
    seed_rows, mean_rows = [], []

    def public_row(values, denominator):
        row = {name: float(value) for name, value in zip(STATISTICS, values)}
        row.update(denominator=float(denominator),
                   C0_numerator=float(row["C0"] * denominator),
                   C1_numerator=float(row["C1"] * denominator),
                   repair_credit_mass=float(row["repair"] * denominator),
                   damage_credit_mass=float(row["damage"] * denominator))
        for name in TRANSITIONS:
            row[name + "_mass"] = float(row[name] * denominator)
        require(abs(row["repair"] - row["damage"] - row["net"]) <= 1e-12,
                "inner repair minus damage differs from net credit")
        require(abs(row["C1"] - row["C0"] - row["net"]) <= 1e-12,
                "inner transition net differs from concordance difference")
        require(abs(sum(row[name] for name in TRANSITIONS) - 1.) <= 1e-12,
                "inner transition fractions do not sum to one")
        return row

    for control in controls:
        comparison = "DUAL-" + control
        for seed in SEEDS:
            numerator = {metric: np.zeros(len(STATISTICS), np.float64) for metric in METRICS}
            denominator = {metric: 0. for metric in METRICS}
            for fold in FOLDS:
                cell = cells[seed, fold]
                fit, ix = cell["fit"], cell["valid"]
                weights = uno_weights(t[fit], e[fit], t[ix], e[ix], tau=60.)
                risk = {arm: cell["arms"][arm]["risk"] for arm in ARMS}
                risk.update(J0=cell["anchor_valid"], J_CS=cell["J_CS_valid"], **cell["sensitivities"])
                transition = pair_transition_summary(t[ix], e[ix], risk[control], risk["DUAL"],
                    folds=np.full(len(ix), fold), ipcw=weights)
                for row in transition["rows"]:
                    metric, mass = row["metric"], row["denominator"]
                    numerator[metric] += np.asarray([row[name] for name in STATISTICS]) * mass
                    denominator[metric] += mass
            for metric in METRICS:
                require(denominator[metric] > 0, "zero inner transition denominator")
                row = public_row(numerator[metric] / denominator[metric], denominator[metric])
                expected = by_seed[seed, "DUAL"][metric] - by_seed[seed, control][metric]
                require(abs(row["net"] - expected) <= 1e-12,
                        "inner transition credit differs from existing seed metric contrast")
                seed_rows.append({"comparison": comparison, "seed": seed, "metric": metric, **row})
        for metric in METRICS:
            selected = [row for row in seed_rows if row["comparison"] == comparison and row["metric"] == metric]
            require(len(selected) == 3, "three complete inner transition seeds required")
            masses = np.asarray([row["denominator"] for row in selected])
            require(np.max(masses) - np.min(masses) <= 1e-10,
                    "inner pair denominator differs across fixed-partition seeds")
            values = np.mean([[row[name] for name in STATISTICS] for row in selected], axis=0)
            mean_rows.append({"comparison": comparison, "metric": metric,
                              **public_row(values, float(masses.mean())),
                              "scope": "descriptive_inner_only_no_bootstrap_or_CI"})
    return {"repair_damage": mean_rows, "seed_repair_damage": seed_rows}


def assemble_outer(cells, endpoint, *, cohort_size=512):
    t, e, valid = _endpoint(endpoint, cohort_size, require_complete=True)
    folds, ipcw = np.zeros(cohort_size, np.int64), np.zeros(cohort_size)
    risks = {(arm, seed): np.full(cohort_size, np.nan) for seed in SEEDS
             for arm in [*ARMS, "J0", "J_CS", *SENSITIVITIES]}
    for fold in FOLDS:
        first = cells[SEEDS[0], fold]
        ix, fit = first["valid"], first["fit"]
        require(not folds[ix].any(), "duplicate outer patient")
        folds[ix] = fold
        ipcw[ix] = uno_weights(t[fit], e[fit], t[ix], e[ix], tau=60.)
        for seed in SEEDS:
            cell = cells[seed, fold]
            available = {arm: cell["arms"][arm]["risk"] for arm in ARMS}
            available.update(J0=cell["anchor_valid"], J_CS=cell["J_CS_valid"], **cell["sensitivities"])
            for arm, values in available.items():
                risks[arm, seed][ix] = values
    require((folds > 0).all() and all(np.isfinite(x).all() for x in risks.values()), "incomplete outer OOF")
    return t, e, valid, folds, ipcw, risks


def outer_comparisons(cells, endpoint, point, *, cohort_size=512):
    t, e, valid, folds, ipcw, risks = assemble_outer(cells, endpoint, cohort_size=cohort_size)
    controls = [*PRIMARY, "MEAN", "CENTERED", "J0", "J_CS", *SENSITIVITIES]
    comparisons = {"DUAL-" + old: {seed: (risks[old, seed], risks["DUAL", seed]) for seed in SEEDS}
                   for old in controls}
    result = patient_shared_transition_bootstrap(t, e, comparisons, folds=folds, valid=valid, ipcw=ipcw,
                                                draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, family_tests=10)
    seed_metrics = {(row["seed"], row["arm"]): row for row in point["seed_rows"]}
    intervals = {(r["comparison"], r["metric"], r["statistic"]): r for r in result["interval_rows"]}
    paired = []
    for row in result["point_rows"]:
        name, metric = row["comparison"], row["metric"]
        control = name[len("DUAL-"):]
        deltas = np.asarray([seed_metrics[seed, "DUAL"][metric] - seed_metrics[seed, control][metric] for seed in SEEDS])
        require(abs(deltas.mean() - row["net"]) <= 1e-12, "bootstrap statistic differs from reported seed metric contrast")
        ci = intervals[name, metric, "net"]
        primary = control in PRIMARY
        paired.append({"comparison": name, "metric": metric, "family": "primary" if primary else "diagnostic",
                       "delta": row["net"], "CI95_low": ci["CI_low"], "CI95_high": ci["CI_high"],
                       "Bonferroni_low": ci["corrected_low"] if primary else None,
                       "Bonferroni_high": ci["corrected_high"] if primary else None,
                       "family_tests": 10 if primary else None, "positive_seeds": int(np.sum(deltas > 0)), "seeds": 3})
    # Do not publish an unregistered family correction on diagnostic contrasts.
    selected_intervals = []
    for row in result["interval_rows"]:
        if row["statistic"] not in ("net", "repair", "damage"):
            continue
        output = dict(row)
        if row["comparison"][len("DUAL-"):] not in PRIMARY or row["statistic"] != "net":
            for key in ("family_tests", "corrected_low", "corrected_high"):
                output.pop(key, None)
        selected_intervals.append(output)
    return {"paired_comparisons": paired, "repair_damage": result["point_rows"],
            "seed_repair_damage": result["seed_rows"], "intervals": selected_intervals,
            "bootstrap_audit": {**result["audit"], "seed": BOOTSTRAP_SEED, "primary_family_tests": 10,
                                "primary_metric_contrasts": 10},
            "all_ten_primary_corrected_lower_bounds_positive": all(r["Bonferroni_low"] > 0 for r in paired if r["family"] == "primary")}, risks, folds, ipcw


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf8")
    temporary.replace(path)


def evaluate_stage(private_root, report_dir, role, source, endpoint_loader, *, cohort_size=512, expected_dss_events=65,
                   partition_loader=None, anchor_loader=None):
    """Call endpoint_loader only after complete artifact/replay qualification.

    endpoint_loader returns {'DSS': {'time','event','valid'}, optionally 'PFI'}.
    partition_loader, if supplied, returns {'fit','valid'} for a role/fold.
    anchor_loader optionally verifies inherited artifacts and returns exact
    {'anchor_fit','anchor_valid','J_CS_valid'} arrays for the declared role/fold.
    PFI is a secondary frozen-risk point evaluation and never changes a gate.
    """
    report_dir, private_root = Path(report_dir), Path(private_root)
    evaluator_sha = evaluator_identity()
    gate_path = report_dir / "INNER_GATE.json"
    admitted_gate = None
    if role == "outer":
        require(gate_path.is_file(), "outer evaluation requires a locked inner gate")
        admitted_gate = json.loads(gate_path.read_text(encoding="utf8"))
        require(admitted_gate.get("status") == "INNER_GATE_PASS" and admitted_gate.get("source") == source
                and admitted_gate.get("outer_admitted") is True, "outer evaluation is not admitted by this source's inner gate")
        require(admitted_gate.get("evaluator_sha256") == evaluator_sha, "inner gate requires reevaluation with the current evaluator")
        # Recheck all receipts and head replays behind that gate, without labels.
        inner_cells, inner_audit = verify_stage(private_root, "inner", source, cohort_size=cohort_size)
        require(inner_audit["receipt_fingerprint"] == admitted_gate.get("receipt_fingerprint"), "locked inner artifacts changed")
    cells, artifact_audit = verify_stage(private_root, role, source, cohort_size=cohort_size)
    # The first permitted endpoint read occurs here, after all fifteen cells.
    endpoints = endpoint_loader()
    t, e, available = _endpoint(endpoints["DSS"], cohort_size, require_complete=True)
    if expected_dss_events is not None:
        require(int(e.sum()) == expected_dss_events, "DSS event count differs from locked cohort")
    if role == "outer":
        inner_points = point_results(inner_cells, endpoints["DSS"], cohort_size=cohort_size)
        independently_rebuilt = inner_gate(inner_points["seed_rows"], source=source,
                                           receipt_fingerprint=inner_audit["receipt_fingerprint"], evaluator_sha256=evaluator_sha)
        require(independently_rebuilt == admitted_gate and independently_rebuilt["status"] == "INNER_GATE_PASS",
                "inner gate does not replay from its locked artifacts")
    if partition_loader is not None:
        for fold in FOLDS:
            expected = partition_loader(role, fold)
            for key in ("fit", "valid"):
                require(np.array_equal(cells[SEEDS[0], fold][key], np.asarray(expected[key], np.int64)),
                        "cell partition differs from frozen inherited split: " + key)
    if anchor_loader is not None:
        for fold in FOLDS:
            expected = anchor_loader(role, fold)
            for key in ("anchor_fit", "anchor_valid", "J_CS_valid"):
                require(np.array_equal(cells[SEEDS[0], fold][key], np.asarray(expected[key], np.float64)),
                        "frozen inherited anchor differs elementwise: " + key)
    point = point_results(cells, endpoints["DSS"], cohort_size=cohort_size)
    result = {"source": source, "role": role, "endpoint": "DSS", "cohort_patients": cohort_size,
              "cohort_events": int(e.sum()), "artifact_audit": artifact_audit, **point,
              "scope": "fixed-kernel low-capacity readout audit; no density/PMI or formal end-to-end architecture claim",
              "new_training_by_evaluator": 0, "patient_arrays_public": False,
              "evaluator_sha256": evaluator_sha,
              "inherited_anchor_elementwise_replay": "PASS" if anchor_loader is not None else "NOT_REQUESTED"}
    if role == "inner":
        result.update(status="COMPLETE_DIAGNOSTIC_INNER_SCREEN", validation_patients_can_repeat_between_folds=True,
                      complete_outer_OOF=False, confirmatory_CI=False)
        result.update(inner_pair_transitions(cells, endpoints["DSS"], point, cohort_size=cohort_size))
        gate = inner_gate(point["seed_rows"], source=source, receipt_fingerprint=artifact_audit["receipt_fingerprint"],
                          evaluator_sha256=evaluator_sha)
        require(evaluator_identity() == evaluator_sha, "evaluation source changed during evaluation")
        write_json(report_dir / "INNER_RESULTS.json", result)
        write_json(gate_path, gate)
        write_json(private_root / "INNER_GATE.json", {
            "decision": "GO" if gate["outer_admitted"] else "STOP", "source": source,
            "status": gate["status"], "evaluator_sha256": evaluator_sha,
            "receipt_fingerprint": gate["receipt_fingerprint"],
            "public_gate_sha256": sha256_file(gate_path),
            "scope": "locked fixed-kernel inner diagnostic screen; not architecture confirmation"})
        return result, gate
    paired, risks, folds, ipcw = outer_comparisons(cells, endpoints["DSS"], point, cohort_size=cohort_size)
    result.update(status="COMPLETE_CONDITIONAL_OUTER_EVALUATION", complete_outer_OOF=True,
                  inner_gate_sha256=sha256_file(gate_path), **paired)
    if "PFI" in endpoints:
        result["secondary_PFI"] = {"status": "FROZEN_DSS_RISK_TRANSFER_POINT_DIAGNOSTIC",
                                   "selection_or_training_on_PFI": False,
                                   **point_results(cells, endpoints["PFI"], cohort_size=cohort_size, require_complete=False)}
    archive = private_root / "evaluation" / "OUTER_OOF_PRIVATE.npz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, folds=folds, frozen_DSS_IPCW=ipcw,
                        **{arm + "__seed" + str(seed): risk for (arm, seed), risk in risks.items()})
    result["private_OOF_archive_sha256"] = sha256_file(archive)
    require(evaluator_identity() == evaluator_sha, "evaluation source changed during evaluation")
    write_json(report_dir / "OUTER_RESULTS.json", result)
    return result, {"status": "OUTER_EVALUATED", "source": source,
                    "all_ten_primary_corrected_lower_bounds_positive": paired["all_ten_primary_corrected_lower_bounds_positive"],
                    "scope": "conditional on fixed fitted models; not independent external confirmation"}

"""Independent, anonymous inner evaluation of four end-to-end CDEI arms.

All fifteen receipts, checkpoint hashes and prediction/training contracts are
checked before endpoints are requested. This evaluator never fits a model and
does not claim encoder replay from finite prediction vectors alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.phase10_trimodal.reference import metrics, uno_weights
from src.phase21_relative.evaluation import STATISTICS, TRANSITIONS, pair_transition_summary


SEEDS = (31, 47, 83)
FOLDS = (1, 2, 3, 4, 5)
ARMS = ("REL", "POOL", "LOCAL", "CTRL")
SENSITIVITIES = ("OFF", "NO_C", "WRONG_C", "WRONG_O")
PRIMARY = ("REL", "POOL", "CTRL")
METRICS = ("Harrell", "Uno60")
RISK_NAMES = (*ARMS, "J0", "J_CS", *("LOCAL_" + name for name in SENSITIVITIES))


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluator_identity():
    """Bind evaluator bytes separately from the immutable training identity."""
    root = Path(__file__).resolve().parents[2]
    names = ("src/phase24_cdei_g1/evaluation.py", "scripts/evaluate_phase24_cdei_g1.py",
             "src/phase10_trimodal/reference.py", "src/phase21_relative/evaluation.py",
             "src/phase13_rna/data.py", "src/phase19_attention/data.py",
             "src/phase20_selective/data.py")
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


def _history(record, arm):
    history = record.get("history")
    require(isinstance(history, list) and len(history) == 80, arm + " must have 80 recorded epochs")
    for epoch, row in enumerate(history, 1):
        require(row.get("epoch") == epoch, arm + " epoch sequence changed")
        for key in ("training_loss", "fit_loss", "grad_norm", "fit_correction_SD", "epoch_seconds"):
            value = row.get(key, np.nan)
            require(np.isscalar(value) and np.isfinite(value), arm + " invalid epoch diagnostic: " + key)
            if key in ("grad_norm", "fit_correction_SD", "epoch_seconds"):
                require(value >= 0, arm + " negative epoch diagnostic: " + key)
        gradients = row.get("gradients", {})
        require(isinstance(gradients, dict) and all(np.isscalar(value) and np.isfinite(value) and value >= 0
                                                  for value in gradients.values()),
                arm + " invalid per-module gradient diagnostic")
    audit = record.get("training_audit", {})
    for key, expected in (("actual_epochs", 80), ("optimizer_steps", 80),
                          ("fixed_final_epoch_no_validation_selection", True),
                          ("heldout_endpoint_reads", 0), ("anchor_unchanged", True)):
        require(audit.get(key) == expected, arm + " training audit mismatch: " + key)
    reference = record.get("fit_only_reference_audit", {})
    for key in ("patient_equal", "fit_query_leave_self_out", "current_encoder_current_kernel", "no_heldout_reference"):
        require(reference.get(key) is True, arm + " reference audit mismatch: " + key)
    return {"actual_epochs": 80, "optimizer_steps": 80,
            "initial_training_loss": float(history[0]["training_loss"]),
            "final_fit_loss": float(history[-1]["fit_loss"]),
            "final_fit_correction_SD": float(history[-1]["fit_correction_SD"])}


def _validate_checkpoint(checkpoint, *, source, role, seed, fold, cohort_size):
    for key, expected in (("source", source), ("role", role), ("seed", seed), ("fold", fold)):
        require(checkpoint.get(key) == expected, "checkpoint metadata mismatch: " + key)
    require(not set(checkpoint).intersection({"time", "event", "endpoint", "endpoints", "DSS", "PFI"}),
            "checkpoint must not contain endpoint arrays")
    fit = _array(checkpoint["fit"], "fit", integer=True)
    valid = _array(checkpoint["valid"], "valid", integer=True)
    for name, index in (("fit", fit), ("valid", valid)):
        require(index.ndim == 1 and len(index) > 0 and len(np.unique(index)) == len(index),
                name + " must be nonempty unique patients")
        require((index >= 0).all() and (index < cohort_size).all(), name + " patient index out of range")
    require(not np.intersect1d(fit, valid).size, "fit/heldout patient leakage")
    anchors = {key: _array(checkpoint[key], key, shape=(len(fit) if key == "anchor_fit" else len(valid),))
               for key in ("anchor_fit", "anchor_valid", "J_CS_valid")}
    require(set(checkpoint.get("arms", {})) == set(ARMS), "four locked end-to-end arms are required")
    arms, summaries = {}, {}
    for arm in ARMS:
        record = checkpoint["arms"][arm]
        risk = _array(record["risk"], arm + " risk", shape=(len(valid),))
        fit_risk = _array(record["fit_risk"], arm + " fit_risk", shape=(len(fit),))
        require(isinstance(record.get("clinical_dim"), int) and record["clinical_dim"] > 0,
                arm + " clinical dimension is invalid")
        raw_membership = record["membership"]
        if isinstance(raw_membership, torch.Tensor):
            raw_membership = raw_membership.detach().cpu().numpy()
        membership = _array(np.asarray(raw_membership).astype(float), arm + " membership")
        require(membership.ndim == 2 and min(membership.shape) > 0 and np.isin(membership, (0., 1.)).all(),
                arm + " membership must be a nonempty binary matrix")
        state = record.get("model_state")
        require(isinstance(state, dict) and state, arm + " model state is absent")
        for name, value in state.items():
            require(isinstance(value, torch.Tensor), arm + " model state must contain tensors")
            require(torch.isfinite(value).all().item(), arm + " model state contains nonfinite values: " + name)
        counts = record.get("parameter_counts", {})
        require(all(isinstance(counts.get(k), int) and counts[k] > 0 for k in ("total", "backbone"))
                and counts["total"] >= counts["backbone"], arm + " parameter count audit is invalid")
        require(record.get("fit_only_reference_audit", {}).get("raw_fit_patient_count") == len(fit),
                arm + " reference fit-patient count changed")
        summaries[arm] = {**_history(record, arm), "parameter_counts": dict(counts),
                          "fit_correction_SD_recomputed": float(np.std(fit_risk - anchors["anchor_fit"]))}
        arms[arm] = {"risk": risk, "fit_risk": fit_risk}
    require(checkpoint["arms"]["REL"]["clinical_dim"] == checkpoint["arms"]["LOCAL"]["clinical_dim"],
            "clinical input dimension differs across arms")
    require(checkpoint["arms"]["LOCAL"]["parameter_counts"]["total"]
            == checkpoint["arms"]["CTRL"]["parameter_counts"]["total"],
            "LOCAL and CTRL must have identical total trainable parameter counts")
    for arm in ARMS[1:]:
        require(np.array_equal(np.asarray(checkpoint["arms"][arm]["membership"]),
                               np.asarray(checkpoint["arms"]["REL"]["membership"])),
                "molecular membership differs across arms")
    sensitivity = checkpoint["arms"]["LOCAL"].get("sensitivities", {})
    require(set(sensitivity) == set(SENSITIVITIES), "LOCAL requires all four frozen-head interventions")
    sensitivities = {name: _array(sensitivity[name], "LOCAL_" + name, shape=(len(valid),))
                     for name in SENSITIVITIES}
    return {"fit": fit, "valid": valid, **anchors, "arms": arms, "sensitivities": sensitivities,
            "role": role, "seed": seed, "fold": fold}, summaries


def verify_stage(private_root, role, source, *, cohort_size=512):
    """Check every expected hash before loading any verified checkpoint."""
    require(role == "inner", "only the locked inner screen is implemented; outer requires a separate admitted evaluation")
    require(isinstance(source, str) and len(source) == 64 and all(c in "0123456789abcdef" for c in source),
            "expected source must be a SHA256")
    root = Path(private_root)
    receipts = []
    for seed in SEEDS:
        for fold in FOLDS:
            folder = root / role / f"seed{seed}_fold{fold}"
            receipt_path = folder / "DONE_PRIVATE.json"
            require(receipt_path.is_file(), f"missing complete {role} cell: seed{seed}/fold{fold}")
            receipt = json.loads(receipt_path.read_text(encoding="utf8"))
            for key, expected in (("status", "COMPLETE"), ("source", source), ("role", role),
                                  ("seed", seed), ("fold", fold), ("arms", 4), ("epochs", 80)):
                require(receipt.get(key) == expected, f"receipt mismatch {key}: seed{seed}/fold{fold}")
            path = folder / "checkpoint.pt"
            require(path.is_file() and sha256_file(path) == receipt.get("checkpoint_sha256"), "checkpoint.pt hash mismatch")
            receipts.append((seed, fold, folder, sha256_file(receipt_path)))
    cells, rows = {}, []
    for seed, fold, folder, receipt_hash in receipts:
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        cell, summaries = _validate_checkpoint(checkpoint, source=source, role=role, seed=seed,
                                               fold=fold, cohort_size=cohort_size)
        cells[seed, fold] = cell
        rows.append({"seed": seed, "fold": fold, "role": role, "receipt_sha256": receipt_hash,
                     "fit_patients": len(cell["fit"]), "validation_patients": len(cell["valid"]), "arms": summaries})
    for fold in FOLDS:
        first = cells[SEEDS[0], fold]
        for seed in SEEDS[1:]:
            for key in ("fit", "valid", "anchor_fit", "anchor_valid", "J_CS_valid"):
                require(np.array_equal(first[key], cells[seed, fold][key]), "fixed partition/anchor differs across seeds: " + key)
    fingerprint = hashlib.sha256("".join(row[3] for row in receipts).encode()).hexdigest()
    audit = {"status": "PASS_SAVED_ARTIFACT_CONTRACT", "source": source, "role": role,
             "complete_cells": 15, "end_to_end_model_fits": 60, "epochs_each": 80,
             "receipt_fingerprint": fingerprint, "rows": rows,
             "encoder_feature_replay_claimed": False, "endpoint_loaded_during_artifact_checks": False,
             "private_patient_arrays_public": False}
    return cells, audit


def _endpoint(value, cohort_size):
    time, event = (np.asarray(value[name], np.float64) for name in ("time", "event"))
    valid = np.asarray(value["valid"])
    require(time.shape == event.shape == valid.shape == (cohort_size,), "invalid endpoint shape")
    require(valid.dtype == np.dtype(bool) and valid.all(), "primary DSS cannot exclude a locked patient")
    require(np.isfinite(time).all() and np.isfinite(event).all() and np.isin(event, (0, 1)).all()
            and (time >= 0).all(), "invalid DSS labels")
    return time, event


def _risks(cell):
    risks = {arm: cell["arms"][arm]["risk"] for arm in ARMS}
    risks.update(J0=cell["anchor_valid"], J_CS=cell["J_CS_valid"])
    risks.update({"LOCAL_" + name: risk for name, risk in cell["sensitivities"].items()})
    return risks


def point_results(cells, endpoint, *, cohort_size=512):
    """Sum five within-fold numerators/denominators, then average three seeds."""
    time, event = _endpoint(endpoint, cohort_size)
    fold_rows, seed_rows = [], []
    for seed in SEEDS:
        totals = {}
        for fold in FOLDS:
            cell = cells[seed, fold]
            fit, valid = cell["fit"], cell["valid"]
            weights = uno_weights(time[fit], event[fit], time[valid], event[valid], tau=60.)
            for arm, risk in _risks(cell).items():
                value = metrics(time[valid], event[valid], risk, np.full(len(valid), fold), weights)
                require(all(np.isfinite(x) for x in value.values()), "undefined heldout concordance")
                fold_rows.append({"seed": seed, "fold": fold, "arm": arm,
                                  "heldout_patients": len(valid), "heldout_events": int(event[valid].sum()), **value})
                total = totals.setdefault(arm, {key: 0. for key in ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")})
                for key in total:
                    total[key] += value[key]
        for arm, total in totals.items():
            seed_rows.append({"seed": seed, "arm": arm, **total,
                              "Harrell": total["Harrell_num"] / total["Harrell_den"],
                              "Uno60": total["Uno_num"] / total["Uno_den"]})
    means = [{"arm": arm, **{metric: float(np.mean([r[metric] for r in seed_rows if r["arm"] == arm]))
                               for metric in METRICS}} for arm in RISK_NAMES]
    return {"fold_rows": fold_rows, "seed_rows": seed_rows, "three_seed_means": means,
            "aggregation": "each seed sums five within-fold numerator/denominator pairs before division; then three complete seed statistics are averaged",
            "risk_ensembling": False, "confidence_intervals": "not_estimated_for_overlapping_inner_validation"}


def inner_gate(seed_rows, *, source, receipt_fingerprint, evaluator_sha256=None):
    """Prelocked prototype screen, not a confirmatory architecture test."""
    by = {(row["seed"], row["arm"]): row for row in seed_rows}
    require(all((seed, arm) in by for seed in SEEDS for arm in RISK_NAMES), "gate needs all locked seed/model statistics")
    primary, anchors, interventions = [], [], []
    for control in PRIMARY:
        for metric in METRICS:
            delta = np.asarray([by[seed, "LOCAL"][metric] - by[seed, control][metric] for seed in SEEDS])
            primary.append({"comparison": "LOCAL-" + control, "metric": metric,
                            "mean_seed_delta": float(delta.mean()), "positive_seeds": int((delta > 0).sum()),
                            "seeds": 3, "pass": bool(delta.mean() > 0 and (delta > 0).sum() >= 2)})
    for control in ("J0", "J_CS"):
        for metric in METRICS:
            delta = [by[seed, "LOCAL"][metric] - by[seed, control][metric] for seed in SEEDS]
            anchors.append({"comparison": "LOCAL-" + control, "metric": metric,
                            "mean_seed_delta": float(np.mean(delta)), "pass": bool(np.mean(delta) >= 0)})
    for intervention in SENSITIVITIES:
        changes = {metric: float(np.mean([by[seed, "LOCAL"][metric] - by[seed, "LOCAL_" + intervention][metric]
                                         for seed in SEEDS])) for metric in METRICS}
        passed = all(value >= 0 for value in changes.values()) and any(value > 0 for value in changes.values())
        interventions.append({"comparison": "LOCAL-LOCAL_" + intervention,
                              "mean_seed_deltas": changes, "pass": passed})
    passed = all(row["pass"] for row in primary + anchors + interventions)
    return {"status": "INNER_GATE_PASS" if passed else "NOT_SUPPORTED_IN_THIS_END_TO_END_LOCAL_OPERATOR",
            "source": source, "role": "inner", "receipt_fingerprint": receipt_fingerprint,
            "evaluator_sha256": evaluator_sha256 or evaluator_identity(),
            "primary_tests": primary, "anchor_noninferiority_screen": anchors,
            "same_head_intervention_screen": interventions,
            "criterion": "LOCAL-control (REL/POOL/CTRL): both metrics mean>0 and >=2/3 positive seeds; LOCAL-J0/J_CS: both means>=0; LOCAL minus each frozen OFF/NO_C/WRONG_C/WRONG_O intervention: both means>=0 and at least one>0",
            "outer_admitted": passed, "outer_automatically_started": False,
            "confirmatory_confidence_intervals": False, "inner_validation_overlap_allowed": True,
            "scope": "diagnostic end-to-end local-operator screen; no universal rejection or confirmation of CDEI",
            "endpoint": "DSS", "automatic_architecture_or_hyperparameter_search": False}


def inner_pair_transitions(cells, endpoint, point, *, cohort_size=512):
    """Anonymous repair/damage credit; no error-group teacher or new gate."""
    time, event = _endpoint(endpoint, cohort_size)
    by = {(row["seed"], row["arm"]): row for row in point["seed_rows"]}
    seed_rows, mean_rows = [], []
    for control in (arm for arm in RISK_NAMES if arm != "LOCAL"):
        for seed in SEEDS:
            numerator = {metric: np.zeros(len(STATISTICS)) for metric in METRICS}
            denominator = {metric: 0. for metric in METRICS}
            for fold in FOLDS:
                cell = cells[seed, fold]
                fit, valid = cell["fit"], cell["valid"]
                weights = uno_weights(time[fit], event[fit], time[valid], event[valid], tau=60.)
                risk = _risks(cell)
                result = pair_transition_summary(time[valid], event[valid], risk[control], risk["LOCAL"],
                    folds=np.full(len(valid), fold), ipcw=weights)
                for row in result["rows"]:
                    metric, mass = row["metric"], row["denominator"]
                    numerator[metric] += np.asarray([row[name] for name in STATISTICS]) * mass
                    denominator[metric] += mass
            for metric in METRICS:
                mass = denominator[metric]
                require(mass > 0, "zero transition denominator")
                values = {name: float(value) for name, value in zip(STATISTICS, numerator[metric] / mass)}
                expected = by[seed, "LOCAL"][metric] - by[seed, control][metric]
                require(abs(values["net"] - expected) <= 1e-12, "transition credit differs from seed metric contrast")
                require(abs(values["repair"] - values["damage"] - values["net"]) <= 1e-12,
                        "repair minus damage differs from net credit")
                require(abs(sum(values[name] for name in TRANSITIONS) - 1) <= 1e-12,
                        "transition fractions do not sum to one")
                seed_rows.append({"comparison": "LOCAL-" + control, "seed": seed, "metric": metric,
                                  "denominator": mass, **values})
        for metric in METRICS:
            selected = [row for row in seed_rows if row["comparison"] == "LOCAL-" + control and row["metric"] == metric]
            mean_rows.append({"comparison": "LOCAL-" + control, "metric": metric,
                              **{name: float(np.mean([row[name] for row in selected])) for name in STATISTICS},
                              "scope": "descriptive_inner_only_no_bootstrap_or_CI"})
    return {"repair_damage": mean_rows, "seed_repair_damage": seed_rows}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def evaluate_stage(private_root, report_dir, role, source, endpoint_loader, *, cohort_size=512,
                   expected_dss_events=65, partition_loader=None, anchor_loader=None):
    """Audit complete private artifacts first, then load DSS exactly once.

An independent anchor_loader can inspect old feature/anchor receipts without
loading labels. It may return fit/valid as well as the three exact risks.
partition_loader runs after endpoint loading for inherited canonical splits.
"""
    private_root, report_dir = Path(private_root), Path(report_dir)
    evaluator_sha = evaluator_identity()
    cells, audit = verify_stage(private_root, role, source, cohort_size=cohort_size)
    if anchor_loader is not None:
        for fold in FOLDS:
            expected = anchor_loader(role, fold)
            for key in ("fit", "valid", "anchor_fit", "anchor_valid", "J_CS_valid"):
                if key in expected:
                    require(np.array_equal(cells[SEEDS[0], fold][key], np.asarray(expected[key])),
                            "frozen inherited anchor/partition differs elementwise: " + key)
    # First endpoint request. Partial cells never reach this line.
    endpoints = endpoint_loader()
    _, event = _endpoint(endpoints["DSS"], cohort_size)
    if expected_dss_events is not None:
        require(int(event.sum()) == expected_dss_events, "DSS event count differs from locked cohort")
    if partition_loader is not None:
        for fold in FOLDS:
            expected = partition_loader(role, fold)
            for key in ("fit", "valid"):
                require(np.array_equal(cells[SEEDS[0], fold][key], np.asarray(expected[key], np.int64)),
                        "cell partition differs from frozen inherited split: " + key)
    point = point_results(cells, endpoints["DSS"], cohort_size=cohort_size)
    result = {"status": "COMPLETE_DIAGNOSTIC_INNER_SCREEN", "source": source, "role": role,
              "endpoint": "DSS", "cohort_patients": cohort_size, "cohort_events": int(event.sum()),
              "artifact_audit": audit, **point,
              **inner_pair_transitions(cells, endpoints["DSS"], point, cohort_size=cohort_size),
              "scope": "end-to-end local dual-evidence operator diagnostic; not architecture confirmation",
              "new_training_by_evaluator": 0, "patient_arrays_public": False,
              "validation_patients_can_repeat_between_folds": True, "complete_outer_OOF": False,
              "confirmatory_CI": False, "evaluator_sha256": evaluator_sha,
              "inherited_anchor_elementwise_replay": "PASS" if anchor_loader is not None else "NOT_REQUESTED"}
    gate = inner_gate(point["seed_rows"], source=source, receipt_fingerprint=audit["receipt_fingerprint"],
                      evaluator_sha256=evaluator_sha)
    require(evaluator_identity() == evaluator_sha, "evaluation source changed during evaluation")
    gate_path = report_dir / "INNER_GATE.json"
    write_json(report_dir / "INNER_RESULTS.json", result)
    write_json(gate_path, gate)
    write_json(private_root / "INNER_GATE.json", {
        "decision": "GO" if gate["outer_admitted"] else "STOP", "source": source,
        "status": gate["status"], "evaluator_sha256": evaluator_sha,
        "receipt_fingerprint": audit["receipt_fingerprint"], "public_gate_sha256": sha256_file(gate_path),
        "outer_automatically_started": False,
        "scope": "locked end-to-end local-operator inner diagnostic; not universal CDEI conclusion"})
    return result, gate

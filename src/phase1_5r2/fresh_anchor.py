"""Fresh Phase1R B1 anchor reproduction.

The wrapper deliberately calls the original ``phase1_neural`` survival
implementation (sampled risk-set Cox training).  It only adapts the frozen
Phase1R patient manifest to the legacy column names and writes checkpoints and
OOF predictions under the R2 output directory, so a historical OOF file is
never used as a training input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import phase1_neural as original_phase1
from phase1_utils import censoring_km, harrell_c, uno_c


FRESH_SEED = 20260723
FOLDS = (1, 2, 3, 4, 5)
MAX_TOKENS = 1024
HISTORICAL_HARRELL = 0.6487327701200534
HISTORICAL_UNO = 0.605460419831024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _legacy_frame(patients: pd.DataFrame) -> pd.DataFrame:
    frame = patients.rename(columns={"dss_event": "dss_status", "dss_time": "dss_months", "fold": "outer_fold"}).copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["dss_status"] = frame["dss_status"].astype(int)
    frame["dss_months"] = frame["dss_months"].astype(float)
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    return frame


def _inner_split(frame: pd.DataFrame, fold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_train = frame.loc[frame.outer_fold != fold].copy().reset_index(drop=True)
    outer_test = frame.loc[frame.outer_fold == fold].copy().reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold)
    train_idx, valid_idx = next(splitter.split(np.zeros(len(outer_train)), outer_train.dss_status.to_numpy(int)))
    return outer_train.iloc[train_idx].copy(), outer_train.iloc[valid_idx].copy(), outer_test


def _fold_paths(out: Path, fold: int) -> dict[str, Path]:
    root = out / "anchor"
    return {
        "checkpoint": root / "checkpoints" / f"fold_{fold}_seed_{FRESH_SEED}.pt",
        "metric": root / "fold_metrics" / f"fold_{fold}_seed_{FRESH_SEED}.json",
        "history": root / "fold_metrics" / f"fold_{fold}_seed_{FRESH_SEED}_training.csv",
        "prediction": root / "oof_predictions" / f"fold_{fold}_seed_{FRESH_SEED}.csv",
    }


def _run_fold(root: Path, out: Path, patients: pd.DataFrame, slides: pd.DataFrame, fold: int, device: torch.device, force: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = _fold_paths(out, fold)
    if not force and all(path.exists() for path in paths.values()):
        return json.loads(paths["metric"].read_text(encoding="utf-8")), pd.read_csv(paths["prediction"])

    # Reuse the original implementation's classes, stochastic sampler,
    # sampled risk-set loss and deterministic evaluation exactly as written.
    original_phase1.DEVICE = device
    # ``train_survival`` in the original Phase1 implementation calls
    # ``set_seed(seed + fold)`` before constructing the model.  Repeating that
    # call is essential for a genuine anchor reproduction rather than merely
    # reusing the same architecture with an arbitrary process RNG state.
    original_phase1.set_seed(FRESH_SEED + fold)
    legacy = _legacy_frame(patients)
    inner_train, inner_val, outer_test = _inner_split(legacy, fold, FRESH_SEED)
    store = original_phase1.BagStore(root, slides, max_tokens=MAX_TOKENS)
    config = original_phase1.EncoderConfig(kind="abmil", use_coords=False, max_tokens=MAX_TOKENS)
    base = original_phase1.SurvivalModel(config)
    initial = copy.deepcopy(base.state_dict())
    candidates = []
    histories = []
    started = time.perf_counter()
    for lr in (1e-4, 3e-4):
        model = original_phase1.SurvivalModel(config).to(device)
        model.load_state_dict(initial)
        best_c, best_epoch, best_state, history = original_phase1.train_survival_candidate(
            model, store, inner_train, inner_val, "cox", lr, fold, FRESH_SEED
        )
        candidates.append((best_c, -lr, lr, best_epoch, best_state))
        histories.extend([{**row, "candidate_lr": lr} for row in history])
    candidates.sort(reverse=True, key=lambda row: (row[0], row[1]))
    best_c, _, best_lr, best_epoch, best_state = candidates[0]
    model = original_phase1.SurvivalModel(config).to(device)
    model.load_state_dict(best_state)
    _, test_risk = original_phase1.evaluate_head(model, store, outer_test)
    outer_train = legacy.loc[legacy.outer_fold != fold].copy()
    censor = censoring_km(outer_train.dss_months.to_numpy(float), outer_train.dss_status.to_numpy(int))
    g_test = censor(outer_test.dss_months.to_numpy(float), left_limit=True)
    h = float(harrell_c(outer_test.dss_months, outer_test.dss_status, test_risk))
    u = float(uno_c(outer_test.dss_months, outer_test.dss_status, test_risk, g_test))
    result = {
        "method": "B1_Phase1R_Fresh",
        "seed": FRESH_SEED,
        "fold": fold,
        "n_train": int(len(outer_train)),
        "n_test": int(len(outer_test)),
        "events_train": int(outer_train.dss_status.sum()),
        "events_test": int(outer_test.dss_status.sum()),
        "loss_type": "cox",
        "best_lr": float(best_lr),
        "best_epoch": int(best_epoch),
        "inner_val_harrell_c": float(best_c),
        "test_harrell_c": h,
        "test_uno_c": u,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "training_time_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated() if device.type == "cuda" else 0),
        "survival_training_implementation": "phase1_neural.py original sampled risk-set survival training",
        "historical_oof_read_for_training": False,
    }
    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "encoder_config": original_phase1.asdict(config),
        "result": result,
        "inner_train_patients": inner_train.patient_id.astype(str).tolist(),
        "inner_val_patients": inner_val.patient_id.astype(str).tolist(),
        "outer_test_patients": outer_test.patient_id.astype(str).tolist(),
        "fresh_training_started_at": _utc_now(),
    }
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, paths["checkpoint"])
    _atomic_csv(pd.DataFrame(histories), paths["history"])
    pred = outer_test[["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    pred = pred.rename(columns={"outer_fold": "fold", "dss_months": "dss_time", "dss_status": "dss_event"})
    pred["seed"] = FRESH_SEED
    pred["risk_score"] = test_risk
    pred["censor_survival_outer_train"] = g_test
    pred["method"] = "B1_Phase1R_Fresh"
    _atomic_csv(pred, paths["prediction"])
    _atomic_json(paths["metric"], result)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return result, pred


def run_fresh_anchor(
    root: Path,
    out: Path,
    patients: pd.DataFrame,
    slides: pd.DataFrame,
    device: torch.device,
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run or resume the five-fold fresh B1 anchor and enforce its gate."""

    aggregate_path = out / "oof_predictions" / "B1_Phase1R_Fresh.csv"
    manifest_path = out / "anchor" / "fresh_anchor_manifest.json"
    gate_path = out / "anchor" / "fresh_anchor_gate.json"
    table_path = out / "anchor" / "fresh_anchor_table.csv"
    if resume and aggregate_path.exists() and manifest_path.exists() and gate_path.exists() and table_path.exists():
        table = pd.read_csv(table_path)
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") == "PASS":
            return table, json.loads(manifest_path.read_text(encoding="utf-8"))
        # Preserve a failed attempt as evidence, then rerun only after the
        # protocol correction.  A failed gate is never silently promoted.
        attempt_dir = out / "anchor" / "failed_attempts" / f"attempt_{len(list((out / 'anchor' / 'failed_attempts').glob('attempt_*'))) + 1}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        for source in (gate_path, table_path, aggregate_path):
            if source.exists():
                target = attempt_dir / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())

    started_at = _utc_now()
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    force = bool(gate_path.exists() and json.loads(gate_path.read_text(encoding="utf-8")).get("status") != "PASS")
    for fold in FOLDS:
        result, prediction = _run_fold(root, out, patients, slides, fold, device, force=force)
        rows.append(result)
        predictions.append(prediction)
    oof = pd.concat(predictions, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    _atomic_csv(oof, aggregate_path)
    fresh_h = float(harrell_c(oof.dss_time, oof.dss_event, oof.risk_score))
    fresh_u = float(uno_c(oof.dss_time, oof.dss_event, oof.risk_score, oof.censor_survival_outer_train))
    table = pd.DataFrame(
        [
            {
                "method": "B1_Phase1R_Fresh",
                "historical_harrell": HISTORICAL_HARRELL,
                "fresh_harrell": fresh_h,
                "abs_delta_harrell": abs(fresh_h - HISTORICAL_HARRELL),
                "historical_uno": HISTORICAL_UNO,
                "fresh_uno": fresh_u,
                "abs_delta_uno": abs(fresh_u - HISTORICAL_UNO),
                "harrell_gate": bool(abs(fresh_h - HISTORICAL_HARRELL) <= 0.02),
                "uno_secondary_check": bool(abs(fresh_u - HISTORICAL_UNO) <= 0.02),
            }
        ]
    )
    _atomic_csv(table, table_path)
    historical_oof = root / "outputs" / "phase1r_confirmatory_20260828" / "runtime_root" / "aggregate_oof_predictions.csv"
    checkpoint_paths = [_fold_paths(out, fold)["checkpoint"] for fold in FOLDS]
    manifest = {
        "training_started_at": started_at,
        "training_finished_at": _utc_now(),
        "fresh_checkpoint_paths": [str(path.relative_to(out).as_posix()) for path in checkpoint_paths],
        "fresh_checkpoint_mtime_ns": [path.stat().st_mtime_ns for path in checkpoint_paths],
        "fresh_checkpoint_sha256": [_sha256(path) for path in checkpoint_paths],
        "fresh_oof_path": str(aggregate_path.relative_to(out).as_posix()),
        "fresh_oof_sha256": _sha256(aggregate_path),
        "historical_oof_read_for_training": False,
        "historical_oof_used_only_for_metric_reference": True,
        "historical_oof_path_for_reference": str(historical_oof.relative_to(root).as_posix()),
        "historical_oof_sha256": _sha256(historical_oof) if historical_oof.exists() else None,
        "fresh_oof_sha_differs_from_historical": bool(not historical_oof.exists() or _sha256(aggregate_path) != _sha256(historical_oof)),
        "implementation_module": "src/phase1_neural.py",
        "implementation_functions": ["SurvivalModel", "BagStore", "train_survival_candidate", "evaluate_head"],
        "seed": FRESH_SEED,
        "folds": list(FOLDS),
        "raw_feature_dim": 1536,
        "projection_dim": 256,
        "max_tokens": MAX_TOKENS,
        "training_sampling": "stochastic sampled risk-set survival training",
        "evaluation_sampling": "original deterministic four-seed embedding average",
    }
    _atomic_json(manifest_path, manifest)
    gate = {
        "status": "PASS" if bool(table.harrell_gate.all()) else "STOP_PHASE1_5R2 ANCHOR_REPRODUCTION_FAILED",
        "primary_metric": "Harrell C",
        "tolerance": 0.02,
        "fresh_harrell": fresh_h,
        "historical_harrell": HISTORICAL_HARRELL,
        "abs_delta_harrell": abs(fresh_h - HISTORICAL_HARRELL),
        "fresh_uno": fresh_u,
        "historical_uno": HISTORICAL_UNO,
        "checked_at": _utc_now(),
    }
    _atomic_json(gate_path, gate)
    if gate["status"] != "PASS":
        raise RuntimeError("STOP_PHASE1_5R2: fresh Phase1R B1 anchor failed the ±0.02 Harrell gate")
    return table, manifest

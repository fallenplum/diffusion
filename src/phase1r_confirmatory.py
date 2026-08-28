"""Local, resumable Phase1R confirmatory split-replication.

The runner deliberately keeps all patient-level intermediates below the run's
``runtime_root``.  The historical Phase P1 output tree is read-only input and
is never passed as the root to the training functions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

import phase1_b0
import phase1_neural
from phase1_utils import censoring_km, fit_cox, harrell_c, scaler_pca, uno_c


EXPERIMENT_ID = "phase1r_confirmatory_20260828"
METHODS = ("B0_mean", "B1_ABMIL", "P0_nocoords", "P0_coords")
METHOD_TO_PHASE1 = {
    "B1_ABMIL": "B1",
    "P0_nocoords": "P0_nocoords",
    "P0_coords": "P0",
}
CONFIG_TO_METHOD = {
    "spatial040_nocoords": "P0_nocoords",
    "spatial040": "P0_coords",
}


class Phase1RBlocked(RuntimeError):
    """A recoverable environment or execution interruption."""


class Phase1RInvalid(RuntimeError):
    """A scientific-integrity failure that must stop conclusions."""


class Tee:
    def __init__(self, stream, log_handle):
        self.stream = stream
        self.log_handle = log_handle

    def write(self, value):
        self.stream.write(value)
        self.stream.flush()
        self.log_handle.write(value)
        self.log_handle.flush()

    def flush(self):
        self.stream.flush()
        self.log_handle.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_from_timestamps(started_at: str, ended_at: str) -> float | None:
    """Return wall-clock seconds between persisted ISO-8601 timestamps."""
    try:
        value = (datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds()
    except (TypeError, ValueError):
        return None
    return max(0.0, float(value))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Phase1RBlocked(f"Missing config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase1RInvalid("Phase1R config root must be a mapping")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise Phase1RBlocked(f"Missing {label}: {path}")
    return path


def resolve_artifact(value: object, data_root: Path, repo_root: Path, label: str) -> str:
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [data_root / raw, repo_root / raw]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return str(resolved)
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise Phase1RBlocked(f"Missing {label}; checked {attempted}")


def canonical_split_bytes(frame: pd.DataFrame) -> bytes:
    rows = []
    ordered = frame.sort_values("patient_id").reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        rows.append(
            f"{str(row.patient_id)}|{int(row.outer_fold)}|{int(row.dss_status)}|{float(row.dss_months):.12g}\n"
        )
    return "".join(rows).encode("utf-8")


def split_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_split_bytes(frame)).hexdigest()


def state_path(run_dir: Path) -> Path:
    return run_dir / "run_state.json"


def read_state(run_dir: Path) -> dict[str, Any]:
    path = state_path(run_dir)
    if not path.exists():
        raise Phase1RBlocked(f"Missing run state: {path}; run preflight first")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Phase1RInvalid(f"Invalid run_state.json: {path}") from exc
    if not isinstance(value, dict):
        raise Phase1RInvalid("run_state.json must contain a mapping")
    return value


def write_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(state_path(run_dir), state)


def update_state(run_dir: Path, **changes: Any) -> dict[str, Any]:
    state = read_state(run_dir)
    state.update(changes)
    write_state(run_dir, state)
    return state


def ensure_state(
    run_dir: Path,
    outer_seed: int,
    model_seed: int,
    resume: bool,
    preflight_only: bool,
) -> dict[str, Any]:
    path = state_path(run_dir)
    if path.exists():
        state = read_state(run_dir)
        expected = {
            "experiment_id": EXPERIMENT_ID,
            "outer_seed": outer_seed,
            "model_seed": model_seed,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise Phase1RInvalid(f"run_state {key} mismatch: {state.get(key)!r} != {value!r}")
        if not resume and not preflight_only and state.get("status") != "COMPLETE":
            raise Phase1RBlocked("Existing run state requires --resume; split/artifacts will not be overwritten")
        return state
    state = {
        "experiment_id": EXPERIMENT_ID,
        "outer_seed": outer_seed,
        "model_seed": model_seed,
        "status": "PREFLIGHT",
        "completed_jobs": [],
        "failed_jobs": [],
        "current_job": None,
        "start_time": utc_now(),
        "updated_at": utc_now(),
    }
    write_state(run_dir, state)
    return state


def build_jobs(n_folds: int = 5) -> list[str]:
    jobs = [f"B0/fold_{fold}" for fold in range(1, n_folds + 1)]
    jobs.extend(f"B1/fold_{fold}" for fold in range(1, n_folds + 1))
    for label in ("nocoords", "coords"):
        jobs.extend(f"SSL_{label}/fold_{fold}" for fold in range(1, n_folds + 1))
        jobs.extend(f"Probe_{label}/fold_{fold}" for fold in range(1, n_folds + 1))
    jobs.extend(["aggregate", "bootstrap", "decision"])
    return jobs


def mark_job_started(run_dir: Path, job: str) -> None:
    state = read_state(run_dir)
    state["status"] = "RUNNING"
    state["current_job"] = job
    write_state(run_dir, state)


def mark_job_complete(run_dir: Path, job: str) -> None:
    state = read_state(run_dir)
    completed = list(state.get("completed_jobs", []))
    if job not in completed:
        completed.append(job)
    state["completed_jobs"] = completed
    state["failed_jobs"] = [item for item in state.get("failed_jobs", []) if item.get("job") != job]
    state["current_job"] = None
    state["status"] = "RUNNING"
    write_state(run_dir, state)


def mark_job_failed(run_dir: Path, job: str, exc: BaseException, status: str) -> None:
    state = read_state(run_dir)
    failed = [item for item in state.get("failed_jobs", []) if item.get("job") != job]
    failed.append({"job": job, "error": str(exc), "time": utc_now()})
    state["failed_jobs"] = failed
    state["current_job"] = None
    state["status"] = status
    write_state(run_dir, state)


def complete_or_run(
    run_dir: Path,
    job: str,
    validator: Callable[[], bool],
    invalidator: Callable[[], None],
    action: Callable[[], None],
) -> None:
    state = read_state(run_dir)
    if job in state.get("completed_jobs", []):
        if validator():
            print(f"[resume] skip completed job: {job}", flush=True)
            return
        print(f"[resume] invalid artifact; rerun job: {job}", flush=True)
        invalidator()
        state["completed_jobs"] = [item for item in state["completed_jobs"] if item != job]
        write_state(run_dir, state)
    elif validator():
        print(f"[resume] artifact integrity passed; adopt job: {job}", flush=True)
        mark_job_complete(run_dir, job)
        return
    else:
        invalidator()
    mark_job_started(run_dir, job)
    print(f"[job start] {job}", flush=True)
    try:
        action()
        if not validator():
            raise Phase1RInvalid(f"Integrity check failed after job: {job}")
    except (Phase1RInvalid, Phase1RBlocked):
        raise
    except BaseException as exc:
        raise Phase1RBlocked(f"Job {job} failed: {exc}") from exc
    mark_job_complete(run_dir, job)
    print(f"[job end] {job}", flush=True)


def source_manifests(repo_root: Path, data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        repo_root / "outputs" / "pathology_context_phase1" / "audit",
        data_root / "outputs" / "pathology_context_phase1" / "audit",
    ]
    audit = next((path for path in candidates if path.exists()), None)
    if audit is None:
        raise Phase1RBlocked("Phase P1 audit directory is unavailable")
    patients = pd.read_csv(require_file(audit / "patient_manifest_used.csv", "P1 patient manifest"))
    slides = pd.read_csv(require_file(audit / "slide_manifest_used.csv", "P1 slide manifest"))
    required_patient = {"patient_id", "dss_status", "dss_months"}
    required_slide = {"patient_id", "slide_id", "bag_path", "coords_path", "n_patches"}
    if not required_patient.issubset(patients.columns):
        raise Phase1RInvalid(f"P1 patient manifest missing {sorted(required_patient - set(patients.columns))}")
    if not required_slide.issubset(slides.columns):
        raise Phase1RInvalid(f"P1 slide manifest missing {sorted(required_slide - set(slides.columns))}")
    patients = patients.copy()
    slides = slides.copy()
    patients["patient_id"] = patients["patient_id"].astype(str)
    slides["patient_id"] = slides["patient_id"].astype(str)
    patients["dss_status"] = pd.to_numeric(patients["dss_status"], errors="raise").astype(int)
    patients["dss_months"] = pd.to_numeric(patients["dss_months"], errors="raise").astype(float)
    if patients["patient_id"].duplicated().any():
        raise Phase1RInvalid("P1 patient manifest contains duplicate patients")
    if set(patients["patient_id"]) != set(slides["patient_id"]):
        raise Phase1RInvalid("P1 patient and slide manifests have different patient sets")
    if not patients["dss_status"].isin([0, 1]).all():
        raise Phase1RInvalid("DSS status is not binary")
    if not np.isfinite(patients["dss_months"]).all() or not (patients["dss_months"] > 0).all():
        raise Phase1RInvalid("DSS durations are not finite and positive")
    if slides["slide_id"].duplicated().any():
        raise Phase1RInvalid("P1 slide manifest contains duplicate slide IDs")
    slides["bag_path"] = [resolve_artifact(value, data_root, repo_root, "UNI2-h bag") for value in slides["bag_path"]]
    slides["coords_path"] = [resolve_artifact(value, data_root, repo_root, "real coordinate array") for value in slides["coords_path"]]
    if "h5_path" in slides.columns:
        slides["h5_path"] = [resolve_artifact(value, data_root, repo_root, "source H5") for value in slides["h5_path"]]
    return patients.sort_values("patient_id").reset_index(drop=True), slides.reset_index(drop=True)


def assign_or_verify_split(
    split_dir: Path,
    source_patients: pd.DataFrame,
    n_folds: int,
    outer_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "confirmatory_patient_manifest.csv"
    summary_path = split_dir / "fold_summary.csv"
    lock_path = split_dir / "split_manifest.json"
    if manifest_path.exists() or summary_path.exists() or lock_path.exists():
        if not (manifest_path.exists() and summary_path.exists() and lock_path.exists()):
            raise Phase1RInvalid("Phase1R split lock is incomplete; refusing to reshuffle")
        locked = pd.read_csv(manifest_path)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        required = {"patient_id", "dss_status", "dss_months", "outer_fold"}
        if not required.issubset(locked.columns):
            raise Phase1RInvalid("Locked split manifest misses required columns")
        locked["patient_id"] = locked["patient_id"].astype(str)
        locked["dss_status"] = pd.to_numeric(locked["dss_status"], errors="raise").astype(int)
        locked["dss_months"] = pd.to_numeric(locked["dss_months"], errors="raise").astype(float)
        locked["outer_fold"] = pd.to_numeric(locked["outer_fold"], errors="raise").astype(int)
        if set(locked["patient_id"]) != set(source_patients["patient_id"]):
            raise Phase1RInvalid("Locked split patients differ from P1 audit patients")
        source_index = source_patients.set_index("patient_id")
        for row in locked.itertuples(index=False):
            source = source_index.loc[row.patient_id]
            if int(row.dss_status) != int(source.dss_status) or not np.isclose(float(row.dss_months), float(source.dss_months), rtol=0, atol=1e-10):
                raise Phase1RInvalid("Locked split labels/durations differ from P1 audit")
        observed_hash = split_hash(locked)
        if lock.get("split_hash") != observed_hash:
            raise Phase1RInvalid("Locked split hash mismatch; refusing to reshuffle")
        if int(lock.get("outer_seed", -1)) != outer_seed or int(lock.get("n_folds", -1)) != n_folds:
            raise Phase1RInvalid("Locked split seed/fold count mismatch")
        return locked.sort_values("patient_id").reset_index(drop=True), lock

    frame = source_patients.copy()
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=outer_seed)
    folds = np.zeros(len(frame), dtype=int)
    events = frame["dss_status"].to_numpy(dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(frame)), events), start=1):
        folds[test_idx] = fold
    if not np.all(folds > 0):
        raise Phase1RInvalid("Confirmatory fold assignment is incomplete")
    frame["outer_fold"] = folds
    frame = frame.sort_values("patient_id").reset_index(drop=True)
    value = {
        "experiment_id": EXPERIMENT_ID,
        "n_folds": n_folds,
        "outer_seed": outer_seed,
        "stratify_on": "dss_status",
        "shuffle": True,
        "n_patients": int(len(frame)),
        "n_events": int(frame["dss_status"].sum()),
        "split_hash_algorithm": "sha256",
        "split_hash_definition": "SHA256 of sorted patient_id|outer_fold|dss_status|dss_months(.12g) rows",
        "split_hash": split_hash(frame),
    }
    atomic_csv(frame[["patient_id", "dss_status", "dss_months", "outer_fold"]], manifest_path)
    rows = []
    for fold in range(1, n_folds + 1):
        test = frame["outer_fold"].to_numpy() == fold
        rows.append({
            "outer_fold": fold,
            "n_train": int((~test).sum()),
            "n_test": int(test.sum()),
            "events_train": int(frame.loc[~test, "dss_status"].sum()),
            "events_test": int(frame.loc[test, "dss_status"].sum()),
        })
    atomic_csv(pd.DataFrame(rows), summary_path)
    atomic_json(lock_path, value)
    return frame, value


def prepare_runtime(
    run_dir: Path,
    runtime_root: Path,
    patients: pd.DataFrame,
    slides: pd.DataFrame,
    model_seed: int,
    n_folds: int,
) -> None:
    if runtime_root.resolve() == run_dir.parent.resolve():
        raise Phase1RInvalid("runtime_root must be isolated below the Phase1R run directory")
    phase = runtime_root / "outputs" / "pathology_context_phase1"
    audit = phase / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    atomic_csv(patients, audit / "patient_manifest_used.csv")
    atomic_csv(slides, audit / "slide_manifest_used.csv")
    internal_state = phase / "run_state.json"
    if not internal_state.exists():
        internal_jobs = []
        internal_jobs.extend(f"B1/fold_{fold}/seed_{model_seed}" for fold in range(1, n_folds + 1))
        internal_jobs.extend(f"SSL_{config}/fold_{fold}/seed_{model_seed}" for config in ("spatial040_nocoords", "spatial040") for fold in range(1, n_folds + 1))
        internal_jobs.extend(f"{method}/fold_{fold}/seed_{model_seed}" for method in ("B1", "P0_nocoords", "P0") for fold in range(1, n_folds + 1))
        atomic_json(internal_state, {
            "stage": "phase1r_runtime",
            "status": "RUNNING",
            "completed_jobs": [],
            "pending_jobs": internal_jobs,
            "updated_at": utc_now(),
        })


def expected_test_ids(patients: pd.DataFrame, fold: int) -> list[str]:
    return sorted(patients.loc[patients["outer_fold"] == fold, "patient_id"].astype(str).tolist())


def read_prediction(path: Path, patients: pd.DataFrame, fold: int, seed: int, method: str) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path, f"{method} prediction"))
    required = {"patient_id", "outer_fold", "dss_months", "dss_status", "seed", "oof_risk", "censor_survival_outer_train"}
    if not required.issubset(frame.columns):
        raise Phase1RInvalid(f"{method} prediction missing columns")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].duplicated().any():
        raise Phase1RInvalid(f"{method} prediction has duplicated patient IDs")
    expected = expected_test_ids(patients, fold)
    if sorted(frame["patient_id"].tolist()) != expected:
        raise Phase1RInvalid(f"{method} fold {fold} test patients mismatch")
    for column in ("outer_fold", "dss_status", "seed"):
        values = pd.to_numeric(frame[column], errors="raise").astype(int)
        if column == "outer_fold" and not (values == fold).all():
            raise Phase1RInvalid(f"{method} fold field mismatch")
        if column == "seed" and not (values == seed).all():
            raise Phase1RInvalid(f"{method} seed field mismatch")
    source = patients.set_index("patient_id").loc[frame["patient_id"]]
    if not np.array_equal(frame["dss_status"].to_numpy(int), source["dss_status"].to_numpy(int)):
        raise Phase1RInvalid(f"{method} DSS labels mismatch")
    if not np.allclose(frame["dss_months"].to_numpy(float), source["dss_months"].to_numpy(float), rtol=0, atol=1e-10):
        raise Phase1RInvalid(f"{method} DSS durations mismatch")
    if not np.isfinite(frame[["oof_risk", "censor_survival_outer_train"]].to_numpy(float)).all():
        raise Phase1RInvalid(f"{method} prediction contains NaN/Inf")
    return frame


def b0_paths(runtime_root: Path, fold: int, seed: int) -> tuple[Path, Path]:
    phase = runtime_root / "outputs" / "pathology_context_phase1"
    return (
        phase / "fold_metrics" / "B0" / f"fold_{fold}_seed_{seed}.json",
        phase / "oof_predictions" / "fold_parts" / "B0" / f"fold_{fold}_seed_{seed}.csv",
    )


def run_b0_fold(
    runtime_root: Path,
    patients: pd.DataFrame,
    slides: pd.DataFrame,
    features: np.ndarray,
    fold: int,
    model_seed: int,
) -> None:
    times = patients["dss_months"].to_numpy(float)
    events = patients["dss_status"].to_numpy(int)
    fold_ids = patients["outer_fold"].to_numpy(int)
    test = fold_ids == fold
    train = ~test
    n_components, alpha, inner_c = phase1_b0.tune(features[train], times[train], events[train], model_seed + fold)
    x_train, x_test, _, pca = scaler_pca(features[train], features[test], n_components, model_seed + fold)
    beta = fit_cox(times[train], events[train], x_train, alpha)
    risk = x_test @ beta
    censor = censoring_km(times[train], events[train])
    g = censor(times[test], left_limit=True)
    row = {
        "method": "B0_mean", "fold": fold, "seed": model_seed,
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "events_train": int(events[train].sum()), "events_test": int(events[test].sum()),
        "pca_components": int(n_components), "alpha": float(alpha), "inner_harrell_c": float(inner_c),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "test_harrell_c": float(harrell_c(times[test], events[test], risk)),
        "test_uno_c": float(uno_c(times[test], events[test], risk, g)),
    }
    metric_path, pred_path = b0_paths(runtime_root, fold, model_seed)
    atomic_json(metric_path, row)
    prediction = patients.loc[test, ["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    prediction["seed"] = model_seed
    prediction["oof_risk"] = risk
    prediction["censor_survival_outer_train"] = g
    atomic_csv(prediction, pred_path)


def validate_b0(runtime_root: Path, patients: pd.DataFrame, fold: int, seed: int) -> bool:
    metric_path, pred_path = b0_paths(runtime_root, fold, seed)
    try:
        metric = json.loads(require_file(metric_path, "B0 fold metric").read_text(encoding="utf-8"))
        if metric.get("method") != "B0_mean" or int(metric.get("fold")) != fold or int(metric.get("seed")) != seed:
            return False
        prediction = read_prediction(pred_path, patients, fold, seed, "B0_mean")
        return bool(np.isfinite([metric["test_harrell_c"], metric["test_uno_c"]]).all() and len(prediction) == len(expected_test_ids(patients, fold)))
    except (OSError, KeyError, TypeError, ValueError, Phase1RBlocked, Phase1RInvalid, json.JSONDecodeError):
        return False


def invalidate_b0(runtime_root: Path, fold: int, seed: int) -> None:
    for path in b0_paths(runtime_root, fold, seed):
        path.unlink(missing_ok=True)


def ssl_checkpoint_path(runtime_root: Path, config: str, fold: int, seed: int) -> Path:
    return runtime_root / "outputs" / "pathology_context_phase1" / "checkpoints" / "ssl" / config / f"fold_{fold}" / f"seed_{seed}.pt"


def validate_ssl(runtime_root: Path, patients: pd.DataFrame, config: str, fold: int, seed: int) -> bool:
    path = ssl_checkpoint_path(runtime_root, config, fold, seed)
    try:
        import torch

        bundle = torch.load(require_file(path, "SSL checkpoint"), map_location="cpu", weights_only=False)
        encoder = bundle["encoder_config"]
        ssl_config = bundle["ssl_config"]
        expected_coords = config == "spatial040"
        required_encoder = {"input_dim": 1536, "dim": 256, "heads": 8, "layers": 2, "ffn_dim": 512, "dropout": 0.1, "max_tokens": 1024}
        if any(encoder.get(key) != value for key, value in required_encoder.items()) or bool(encoder.get("use_coords")) != expected_coords:
            return False
        if ssl_config.get("mask_type") != "spatial" or float(ssl_config.get("mask_ratio")) != 0.4 or bool(ssl_config.get("use_coords")) != expected_coords:
            return False
        if int(bundle.get("fold")) != fold or int(bundle.get("seed")) != seed:
            return False
        if sorted(map(str, bundle.get("outer_test_patients", []))) != expected_test_ids(patients, fold):
            return False
        return bool(np.isfinite(float(bundle["best_val_loss"])))
    except (OSError, KeyError, TypeError, ValueError, Phase1RBlocked, Phase1RInvalid, RuntimeError):
        return False


def invalidate_ssl(runtime_root: Path, config: str, fold: int, seed: int) -> None:
    path = ssl_checkpoint_path(runtime_root, config, fold, seed)
    path.unlink(missing_ok=True)
    for relative in [
        Path("ssl_metrics") / config / f"fold_{fold}",
    ]:
        directory = runtime_root / "outputs" / "pathology_context_phase1" / relative
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


def survival_paths(runtime_root: Path, fold: int, seed: int) -> tuple[Path, Path, Path]:
    phase = runtime_root / "outputs" / "pathology_context_phase1"
    return (
        phase / "checkpoints" / "survival" / "B1" / f"fold_{fold}" / f"seed_{seed}.pt",
        phase / "fold_metrics" / "B1" / f"fold_{fold}_seed_{seed}.json",
        phase / "oof_predictions" / "fold_parts" / "B1" / f"fold_{fold}_seed_{seed}.csv",
    )


def validate_b1(runtime_root: Path, patients: pd.DataFrame, fold: int, seed: int) -> bool:
    ckpt, metric_path, pred_path = survival_paths(runtime_root, fold, seed)
    try:
        import torch

        bundle = torch.load(require_file(ckpt, "B1 checkpoint"), map_location="cpu", weights_only=False)
        config = bundle["encoder_config"]
        result = json.loads(require_file(metric_path, "B1 fold metric").read_text(encoding="utf-8"))
        if config.get("kind") != "abmil" or bool(config.get("use_coords")):
            return False
        if result.get("model") != "B1" or int(result.get("outer_fold")) != fold or int(result.get("seed")) != seed:
            return False
        if sorted(map(str, bundle.get("outer_test_patients", []))) != expected_test_ids(patients, fold):
            return False
        prediction = read_prediction(pred_path, patients, fold, seed, "B1_ABMIL")
        return bool(np.isfinite([result["test_harrell_c"], result["test_uno_c"]]).all() and len(prediction) == len(expected_test_ids(patients, fold)))
    except (OSError, KeyError, TypeError, ValueError, Phase1RBlocked, Phase1RInvalid, RuntimeError, json.JSONDecodeError):
        return False


def invalidate_b1(runtime_root: Path, fold: int, seed: int) -> None:
    for path in survival_paths(runtime_root, fold, seed):
        path.unlink(missing_ok=True)


def probe_paths(runtime_root: Path, method: str, fold: int, seed: int) -> tuple[Path, Path]:
    phase = runtime_root / "outputs" / "pathology_context_phase1"
    return (
        phase / "fold_metrics" / method / f"fold_{fold}_seed_{seed}.json",
        phase / "oof_predictions" / "fold_parts" / method / f"fold_{fold}_seed_{seed}.csv",
    )


def validate_probe(runtime_root: Path, patients: pd.DataFrame, method: str, fold: int, seed: int) -> bool:
    metric_path, pred_path = probe_paths(runtime_root, method, fold, seed)
    try:
        result = json.loads(require_file(metric_path, "probe fold metric").read_text(encoding="utf-8"))
        if result.get("model") != method or int(result.get("outer_fold")) != fold or int(result.get("seed")) != seed:
            return False
        prediction = read_prediction(pred_path, patients, fold, seed, method)
        return bool(np.isfinite([result["test_harrell_c"], result["test_uno_c"]]).all() and len(prediction) == len(expected_test_ids(patients, fold)))
    except (OSError, KeyError, TypeError, ValueError, Phase1RBlocked, Phase1RInvalid, json.JSONDecodeError):
        return False


def invalidate_probe(runtime_root: Path, method: str, fold: int, seed: int) -> None:
    for path in probe_paths(runtime_root, method, fold, seed):
        path.unlink(missing_ok=True)


def run_neural_job(runtime_root: Path, action: str, fold: int, seed: int) -> None:
    if action == "B1":
        phase1_neural.train_survival(runtime_root, "B1", fold, seed)
    elif action == "SSL_nocoords":
        phase1_neural.train_ssl(runtime_root, "spatial040_nocoords", fold, seed)
    elif action == "SSL_coords":
        phase1_neural.train_ssl(runtime_root, "spatial040", fold, seed)
    elif action == "Probe_nocoords":
        phase1_neural.run_probe(runtime_root, "spatial040_nocoords", fold, seed)
    elif action == "Probe_coords":
        phase1_neural.run_probe(runtime_root, "spatial040", fold, seed)
    else:
        raise ValueError(action)


def compute_runtime_features(runtime_root: Path, patients: pd.DataFrame, slides: pd.DataFrame) -> np.ndarray:
    features = phase1_b0.compute_patient_means(runtime_root, patients, slides)
    if features.shape != (len(patients), 1536) or not np.isfinite(features).all():
        raise Phase1RInvalid(f"B0 patient features have invalid shape/values: {features.shape}")
    return features


def aggregate_oof(
    run_dir: Path,
    runtime_root: Path,
    patients: pd.DataFrame,
    model_seed: int,
    n_folds: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame, pd.DataFrame]:
    phase = runtime_root / "outputs" / "pathology_context_phase1"
    rows: list[dict[str, Any]] = []
    by_method: dict[str, list[pd.DataFrame]] = {method: [] for method in METHODS}
    for fold in range(1, n_folds + 1):
        b0_metric, b0_pred = b0_paths(runtime_root, fold, model_seed)
        metric = json.loads(require_file(b0_metric, "B0 metric").read_text(encoding="utf-8"))
        rows.append(metric)
        by_method["B0_mean"].append(read_prediction(b0_pred, patients, fold, model_seed, "B0_mean"))
        b1_ckpt, b1_metric, b1_pred = survival_paths(runtime_root, fold, model_seed)
        _ = b1_ckpt
        metric = json.loads(require_file(b1_metric, "B1 metric").read_text(encoding="utf-8"))
        metric = {**metric, "method": "B1_ABMIL", "fold": int(metric["outer_fold"]), "seed": int(metric["seed"]), "test_harrell_c": metric["test_harrell_c"], "test_uno_c": metric["test_uno_c"]}
        rows.append(metric)
        by_method["B1_ABMIL"].append(read_prediction(b1_pred, patients, fold, model_seed, "B1_ABMIL"))
        for method in ("P0_nocoords", "P0_coords"):
            phase_method = METHOD_TO_PHASE1[method]
            metric_path, pred_path = probe_paths(runtime_root, phase_method, fold, model_seed)
            metric = json.loads(require_file(metric_path, f"{method} metric").read_text(encoding="utf-8"))
            metric = {**metric, "method": method, "fold": int(metric["outer_fold"]), "seed": int(metric["seed"]), "test_harrell_c": metric["test_harrell_c"], "test_uno_c": metric["test_uno_c"]}
            rows.append(metric)
            by_method[method].append(read_prediction(pred_path, patients, fold, model_seed, method))
    fold_frame = pd.DataFrame(rows)
    if len(fold_frame) != 20 or set(fold_frame["method"]) != set(METHODS) or fold_frame.duplicated(["method", "fold"]).any():
        raise Phase1RInvalid("Expected exactly 4 methods x 5 fold metric rows")
    risks: dict[str, np.ndarray] = {}
    g_values: np.ndarray | None = None
    all_oof = []
    expected_ids = set(patients["patient_id"].astype(str))
    for method, parts in by_method.items():
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != len(patients) or set(frame["patient_id"]) != expected_ids or frame["patient_id"].duplicated().any():
            raise Phase1RInvalid(f"{method} does not have exactly one OOF risk per patient")
        frame = frame.set_index("patient_id").loc[patients["patient_id"].astype(str)].reset_index()
        if not np.array_equal(frame["outer_fold"].to_numpy(int), patients["outer_fold"].to_numpy(int)):
            raise Phase1RInvalid(f"{method} outer folds mismatch")
        risk = frame["oof_risk"].to_numpy(float)
        g = frame["censor_survival_outer_train"].to_numpy(float)
        if not np.isfinite(risk).all() or not np.isfinite(g).all():
            raise Phase1RInvalid(f"{method} OOF contains NaN/Inf")
        risks[method] = risk
        if g_values is None:
            g_values = g
        elif not np.allclose(g_values, g, rtol=0, atol=1e-12):
            raise Phase1RInvalid(f"{method} censoring weights disagree with B0")
        all_oof.append(frame.assign(method=method))
    assert g_values is not None
    oof_frame = pd.concat(all_oof, ignore_index=True)
    atomic_csv(oof_frame, runtime_root / "aggregate_oof_predictions.csv")
    final = run_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    atomic_csv(fold_frame, final / "fold_metrics.csv")
    split_summary = pd.read_csv(run_dir / "split" / "fold_summary.csv")
    atomic_csv(split_summary, final / "fold_summary.csv")
    return risks, g_values, fold_frame, oof_frame


def paired_bootstrap(
    times: np.ndarray,
    events: np.ndarray,
    baseline: np.ndarray,
    method: np.ndarray,
    g_values: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    delta_h: list[float] = []
    delta_u: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(times), size=len(times))
        h0 = harrell_c(times[indices], events[indices], baseline[indices])
        h1 = harrell_c(times[indices], events[indices], method[indices])
        u0 = uno_c(times[indices], events[indices], baseline[indices], g_values[indices])
        u1 = uno_c(times[indices], events[indices], method[indices], g_values[indices])
        if np.isfinite(h0) and np.isfinite(h1):
            delta_h.append(float(h1 - h0))
        if np.isfinite(u0) and np.isfinite(u1):
            delta_u.append(float(u1 - u0))
    if not delta_h or not delta_u:
        raise Phase1RInvalid("No valid paired bootstrap replicate")
    return {
        "delta_harrell": float(harrell_c(times, events, method) - harrell_c(times, events, baseline)),
        "delta_harrell_ci95_low": float(np.quantile(delta_h, 0.025)),
        "delta_harrell_ci95_high": float(np.quantile(delta_h, 0.975)),
        "delta_uno": float(uno_c(times, events, method, g_values) - uno_c(times, events, baseline, g_values)),
        "delta_uno_ci95_low": float(np.quantile(delta_u, 0.025)),
        "delta_uno_ci95_high": float(np.quantile(delta_u, 0.975)),
        "valid_harrell_bootstraps": len(delta_h),
        "valid_uno_bootstraps": len(delta_u),
        "bootstrap_replicates": replicates,
        "seed": seed,
    }


def summarize(
    patients: pd.DataFrame,
    risks: dict[str, np.ndarray],
    g_values: np.ndarray,
    fold_frame: pd.DataFrame,
    bootstrap_replicates: int,
    outer_seed: int,
    min_positive_folds: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    times = patients["dss_months"].to_numpy(float)
    events = patients["dss_status"].to_numpy(int)
    methods: dict[str, Any] = {}
    baseline_rows = fold_frame.loc[fold_frame["method"] == "B0_mean"].set_index("fold")
    for method in METHODS:
        rows = fold_frame.loc[fold_frame["method"] == method].set_index("fold").sort_index()
        methods[method] = {
            "oof_harrell_c": float(harrell_c(times, events, risks[method])),
            "oof_uno_c": float(uno_c(times, events, risks[method], g_values)),
            "fold_mean_harrell_c": float(rows["test_harrell_c"].mean()),
            "fold_std_harrell_c": float(rows["test_harrell_c"].std(ddof=1)),
            "fold_mean_uno_c": float(rows["test_uno_c"].mean()),
            "fold_std_uno_c": float(rows["test_uno_c"].std(ddof=1)),
            "per_fold_harrell_c": [float(value) for value in rows["test_harrell_c"]],
            "per_fold_uno_c": [float(value) for value in rows["test_uno_c"]],
        }
    bootstrap_rows = []
    for offset, method in enumerate(("B1_ABMIL", "P0_nocoords", "P0_coords"), start=1):
        rows = fold_frame.loc[fold_frame["method"] == method].set_index("fold")
        methods[method]["delta_harrell_vs_b0"] = methods[method]["oof_harrell_c"] - methods["B0_mean"]["oof_harrell_c"]
        methods[method]["delta_uno_vs_b0"] = methods[method]["oof_uno_c"] - methods["B0_mean"]["oof_uno_c"]
        methods[method]["positive_harrell_folds_vs_b0"] = int((rows["test_harrell_c"] > baseline_rows["test_harrell_c"]).sum())
        methods[method]["positive_uno_folds_vs_b0"] = int((rows["test_uno_c"] > baseline_rows["test_uno_c"]).sum())
        boot = paired_bootstrap(times, events, risks["B0_mean"], risks[method], g_values, bootstrap_replicates, outer_seed + offset * 101)
        methods[method].update(boot)
        bootstrap_rows.append({"comparison": f"{method} vs B0_mean", **boot})
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    def passes(method: str) -> bool:
        result = methods[method]
        return bool(
            result["delta_harrell_vs_b0"] > 0
            and result["delta_uno_vs_b0"] > 0
            and result["positive_harrell_folds_vs_b0"] >= min_positive_folds
            and result["positive_uno_folds_vs_b0"] >= min_positive_folds
        )
    if passes("P0_nocoords"):
        decision_code = "PASS_P0_NOCOORDS"
        selected = "P0_nocoords frozen pathology representation"
    elif passes("B1_ABMIL"):
        decision_code = "FALLBACK_B1"
        selected = "B1_ABMIL"
    else:
        decision_code = "FALLBACK_B0"
        selected = "B0_mean"
    decision = {
        "decision": decision_code,
        "selected_pathology_branch_for_phase2": selected,
        "p0_nocoords_gate": "PASS" if passes("P0_nocoords") else "FAIL",
        "b1_gate": "PASS" if passes("B1_ABMIL") else "FAIL",
        "min_positive_folds_per_metric": min_positive_folds,
        "p0_nocoords_beats_p0_coords_on_both_oof_metrics": bool(
            methods["P0_nocoords"]["oof_harrell_c"] > methods["P0_coords"]["oof_harrell_c"]
            and methods["P0_nocoords"]["oof_uno_c"] > methods["P0_coords"]["oof_uno_c"]
        ),
        "scope_note": "This is a same-cohort split-replication, not an independent external validation.",
    }
    return methods, decision, bootstrap_frame


def git_value(repo_root: Path, *args: str) -> str:
    for git_dir in (repo_root / ".git_local", repo_root / ".git"):
        if not git_dir.exists():
            continue
        result = subprocess.run(["git", f"--git-dir={git_dir}", f"--work-tree={repo_root}", *args], cwd=repo_root, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    return "unknown"


def report_safe(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if re.search(r"[A-Za-z]:[\\/]", text) or re.search(r"TCGA-[0-9A-Z]{2}-[A-Z0-9]{4}", text):
        raise Phase1RInvalid(f"Sensitive absolute path or patient ID found in report: {path}")


def write_reports(
    repo_root: Path,
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    split_lock: dict[str, Any],
    patients: pd.DataFrame,
    fold_frame: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    methods: dict[str, Any],
    decision: dict[str, Any],
    started_at: str,
    elapsed_seconds: float,
    completed_count: int,
    ended_at: str | None = None,
) -> None:
    final = run_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    report_end_time = ended_at or utc_now()
    atomic_json(final / "summary.json", {
        "experiment_id": EXPERIMENT_ID,
        "phase": "phase1r-confirmatory",
        "n_patients": int(len(patients)),
        "n_events": int(patients["dss_status"].sum()),
        "n_slides": 494,
        "outer_seed": int(config["split"]["outer_seed"]),
        "model_seed": int(config["reproducibility"]["model_seed"]),
        "split_hash": split_lock["split_hash"],
        "methods": methods,
        "decision": decision,
        "start_time": started_at,
        "end_time": report_end_time,
        "elapsed_seconds": float(elapsed_seconds),
    })
    atomic_json(final / "decision.json", decision)
    atomic_csv(bootstrap_frame, final / "bootstrap_summary.csv")
    fold_rows = fold_frame.sort_values(["method", "fold"]).copy()
    atomic_csv(fold_rows, final / "fold_metrics.csv")
    split_summary = pd.read_csv(run_dir / "split" / "fold_summary.csv")
    atomic_csv(split_summary, final / "fold_summary.csv")
    run_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "branch": git_value(repo_root, "symbolic-ref", "--short", "-q", "HEAD"),
        "python_version": sys.version.split()[0],
        "pytorch_version": getattr(phase1_neural.torch, "__version__", "unknown"),
        "cuda_version": getattr(phase1_neural.torch.version, "cuda", None),
        "gpu": phase1_neural.torch.cuda.get_device_name(0) if phase1_neural.torch.cuda.is_available() else None,
        "outer_seed": int(config["split"]["outer_seed"]),
        "model_seed": int(config["reproducibility"]["model_seed"]),
        "n_patients": int(len(patients)),
        "n_events": int(patients["dss_status"].sum()),
        "fold_sizes": split_summary.to_dict("records"),
        "config_path": "configs/phase1r/confirmatory.yaml",
        "config_sha256": sha256_file(config_path),
        "split_hash": split_lock["split_hash"],
        "start_time": started_at,
        "end_time": report_end_time,
        "elapsed_seconds": float(elapsed_seconds),
        "completed_job_count": int(completed_count),
    }
    atomic_json(final / "run_manifest.json", run_manifest)
    table_lines = ["| Method | OOF Harrell C | OOF Uno C | Fold Harrell mean ± SD | Fold Uno mean ± SD |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        result = methods[method]
        table_lines.append(f"| {method} | {result['oof_harrell_c']:.6f} | {result['oof_uno_c']:.6f} | {result['fold_mean_harrell_c']:.6f} ± {result['fold_std_harrell_c']:.6f} | {result['fold_mean_uno_c']:.6f} ± {result['fold_std_uno_c']:.6f} |")
    fold_lines = ["| Method | Fold 1 Harrell | Fold 2 Harrell | Fold 3 Harrell | Fold 4 Harrell | Fold 5 Harrell |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        values = methods[method]["per_fold_harrell_c"]
        fold_lines.append("| " + method + " | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    bootstrap_lines = ["| Comparison | ΔHarrell | 95% CI | ΔUno | 95% CI | Valid Harrell / Uno |", "|---|---:|---|---:|---|---:|"]
    for row in bootstrap_frame.to_dict("records"):
        bootstrap_lines.append(f"| {row['comparison']} | {row['delta_harrell']:.6f} | [{row['delta_harrell_ci95_low']:.6f}, {row['delta_harrell_ci95_high']:.6f}] | {row['delta_uno']:.6f} | [{row['delta_uno_ci95_low']:.6f}, {row['delta_uno_ci95_high']:.6f}] | {int(row['valid_harrell_bootstraps'])} / {int(row['valid_uno_bootstraps'])} |")
    results_text = "\n".join([
        "# Phase1R confirmatory split-replication",
        "",
        "## Scope",
        "",
        "This is a same-cohort split-replication, not an independent external validation.",
        "The purpose was to test the pre-registered Phase P0_nocoords signal on a new patient-level outer split; no post-result tuning or seed hunting was performed.",
        "",
        "## Data and split audit",
        "",
        f"- Cohort: TCGA only; {len(patients)} patients, {int(patients['dss_status'].sum())} DSS events, 494 slides, feature dimension 1536.",
        "- All selected bags and coordinate arrays were resolved to existing local artifacts; coordinates were taken from the audited real H5-derived arrays.",
        f"- Outer split: StratifiedKFold(n_splits=5, shuffle=True, random_state={config['split']['outer_seed']}), patient-level, locked hash `{split_lock['split_hash']}`.",
        "- All PCA/scaler fitting, SSL pretraining, and survival training were restricted to outer-train patients for each fold.",
        "",
        "## Four-method primary results",
        "",
        *table_lines,
        "",
        "## Per-fold Harrell C",
        "",
        *fold_lines,
        "",
        "Per-fold Uno C is available in `fold_metrics.csv`.",
        "",
        "## Paired bootstrap (1000 replicates)",
        "",
        *bootstrap_lines,
        "",
        "## Pre-registered gate",
        "",
        f"- P0_nocoords gate: **{decision['p0_nocoords_gate']}**; B1 gate: **{decision['b1_gate']}**.",
        f"- Decision: **{decision['decision']}**; selected pathology branch: **{decision['selected_pathology_branch_for_phase2']}**.",
        f"- P0_nocoords exceeds P0_coords on both OOF metrics: **{decision['p0_nocoords_beats_p0_coords_on_both_oof_metrics']}**.",
        "- P0_coords was diagnostic only and did not participate in fallback selection.",
        "",
        "## Limitations",
        "",
        "This is not external validation; cohort-specific bias and limited DSS events remain. The neural survival implementation uses the locked nested risk-set approximation documented in Phase P1. Bootstrap intervals are uncertainty evidence, not a tuning criterion.",
    ])
    atomic_text(final / "RESULTS.md", results_text + "\n")
    reproducibility = "\n".join([
        "# Reproducibility",
        "",
        "Use the validated local CUDA environment (PyTorch 2.4.1+cu121) and keep protected data outside Git.",
        "",
        "```powershell",
        "$env:DIFFUSION_DATA_ROOT = '<local data root containing TCGA>'",
        "python src/phase1r_confirmatory.py --config configs/phase1r/confirmatory.yaml --run-dir outputs/phase1r_confirmatory_20260828 --preflight-only",
        "python -u src/phase1r_confirmatory.py --config configs/phase1r/confirmatory.yaml --run-dir outputs/phase1r_confirmatory_20260828 --resume",
        "```",
        "",
        "The split is generated once and then verified by SHA256 on every resume. Intermediate checkpoints, patient-level predictions, and absolute local paths remain below the ignored runtime directory.",
    ])
    atomic_text(final / "REPRODUCIBILITY.md", reproducibility + "\n")
    report_root = repo_root / "reports" / "phase1r_confirmatory_20260828"
    report_root.mkdir(parents=True, exist_ok=True)
    for name in ("RESULTS.md", "summary.json", "decision.json", "fold_metrics.csv", "bootstrap_summary.csv", "fold_summary.csv", "run_manifest.json", "REPRODUCIBILITY.md"):
        shutil.copy2(final / name, report_root / name)
        report_safe(report_root / name)


def cleanup_runtime(run_dir: Path) -> None:
    state = read_state(run_dir)
    final = run_dir / "final"
    required = [final / name for name in ("RESULTS.md", "summary.json", "decision.json", "fold_metrics.csv", "bootstrap_summary.csv", "fold_summary.csv", "run_manifest.json", "REPRODUCIBILITY.md")]
    if state.get("status") != "COMPLETE" or not all(path.exists() for path in required):
        raise Phase1RBlocked("--cleanup-runtime requires COMPLETE status and all final reports")
    runtime = run_dir / "runtime_root"
    if runtime.exists():
        shutil.rmtree(runtime)
    print(f"[cleanup] removed isolated runtime: {runtime}", flush=True)


def preflight(
    repo_root: Path,
    run_dir: Path,
    runtime_root: Path,
    config_path: Path,
    config: dict[str, Any],
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not data_root.exists() or not data_root.is_dir():
        raise Phase1RBlocked(f"DIFFUSION_DATA_ROOT is not a directory: {data_root}")
    if not (data_root / "TCGA").exists():
        raise Phase1RBlocked(f"Required data directory is missing: {data_root / 'TCGA'}")
    patients, slides = source_manifests(repo_root, data_root)
    if len(patients) != 487 or int(patients["dss_status"].sum()) != 65 or len(slides) != 494:
        raise Phase1RInvalid(f"P1 audit counts differ from locked protocol: patients={len(patients)}, events={int(patients['dss_status'].sum())}, slides={len(slides)}")
    if "feature_dim" in slides.columns and set(pd.to_numeric(slides["feature_dim"], errors="raise").astype(int)) != {1536}:
        raise Phase1RInvalid("Slide feature dimension is not exactly 1536")
    if "coords_available" in slides.columns and not slides["coords_available"].astype(str).str.lower().isin(["true", "1"]).all():
        raise Phase1RInvalid("Not all selected slides have audited real coordinates")
    if not phase1_neural.torch.cuda.is_available():
        raise Phase1RBlocked("CUDA is unavailable; formal Phase1R requires the validated local GPU")
    patients, split_lock = assign_or_verify_split(run_dir / "split", patients, int(config["split"]["n_folds"]), int(config["split"]["outer_seed"]))
    prepare_runtime(run_dir, runtime_root, patients, slides, int(config["reproducibility"]["model_seed"]), int(config["split"]["n_folds"]))
    probe = run_dir / ".write_probe"
    atomic_text(probe, "ok\n")
    probe.unlink(missing_ok=True)
    atomic_json(run_dir / "preflight.json", {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "n_patients": len(patients),
        "n_events": int(patients["dss_status"].sum()),
        "n_slides": len(slides),
        "outer_seed": int(config["split"]["outer_seed"]),
        "model_seed": int(config["reproducibility"]["model_seed"]),
        "device": str(phase1_neural.DEVICE),
        "gpu": phase1_neural.torch.cuda.get_device_name(0),
        "split_hash": split_lock["split_hash"],
        "config_sha256": sha256_file(config_path),
    })
    return patients, slides, split_lock


def run(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_config(config_path)
    run_dir = args.run_dir.resolve()
    runtime_root = run_dir / "runtime_root"
    run_dir.mkdir(parents=True, exist_ok=True)
    outer_seed = int(config["split"]["outer_seed"])
    model_seed = int(config["reproducibility"]["model_seed"])
    state = ensure_state(run_dir, outer_seed, model_seed, args.resume, args.preflight_only)
    prior_status = state.get("status")
    prior_elapsed = state.get("elapsed_seconds")
    log_path = run_dir / "logs" / "phase1r.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("a", encoding="utf-8") as log_handle:
        sys.stdout, sys.stderr = Tee(original_stdout, log_handle), Tee(original_stderr, log_handle)
        try:
            data_root_value = os.environ.get("DIFFUSION_DATA_ROOT")
            if not data_root_value:
                raise Phase1RBlocked("DIFFUSION_DATA_ROOT is not set")
            data_root = Path(data_root_value).resolve()
            patients, slides, split_lock = preflight(repo_root, run_dir, runtime_root, config_path, config, data_root)
            print(json.dumps({"phase1r": "PREFLIGHT PASS", "n_patients": len(patients), "n_events": int(patients["dss_status"].sum()), "n_slides": len(slides), "split_hash": split_lock["split_hash"]}, ensure_ascii=False), flush=True)
            if args.preflight_only:
                update_state(run_dir, status="PREFLIGHT_PASS", current_job=None)
                return 0
            update_state(run_dir, status="RUNNING", current_job=None)
            started_at = read_state(run_dir).get("start_time", utc_now())
            n_folds = int(config["split"]["n_folds"])
            features = compute_runtime_features(runtime_root, patients, slides)
            for fold in range(1, n_folds + 1):
                complete_or_run(
                    run_dir, f"B0/fold_{fold}",
                    lambda fold=fold: validate_b0(runtime_root, patients, fold, model_seed),
                    lambda fold=fold: invalidate_b0(runtime_root, fold, model_seed),
                    lambda fold=fold: run_b0_fold(runtime_root, patients, slides, features, fold, model_seed),
                )
            for fold in range(1, n_folds + 1):
                complete_or_run(
                    run_dir, f"B1/fold_{fold}",
                    lambda fold=fold: validate_b1(runtime_root, patients, fold, model_seed),
                    lambda fold=fold: invalidate_b1(runtime_root, fold, model_seed),
                    lambda fold=fold: run_neural_job(runtime_root, "B1", fold, model_seed),
                )
            for action, label, config_name, method in (
                ("SSL_nocoords", "nocoords", "spatial040_nocoords", None),
                ("Probe_nocoords", "nocoords", None, "P0_nocoords"),
                ("SSL_coords", "coords", "spatial040", None),
                ("Probe_coords", "coords", None, "P0_coords"),
            ):
                for fold in range(1, n_folds + 1):
                    job = f"{action}/fold_{fold}"
                    if action.startswith("SSL"):
                        validator = lambda config_name=config_name, fold=fold: validate_ssl(runtime_root, patients, config_name, fold, model_seed)
                        invalidator = lambda config_name=config_name, fold=fold: invalidate_ssl(runtime_root, config_name, fold, model_seed)
                    else:
                        phase_method = METHOD_TO_PHASE1[method]
                        validator = lambda phase_method=phase_method, fold=fold: validate_probe(runtime_root, patients, phase_method, fold, model_seed)
                        invalidator = lambda phase_method=phase_method, fold=fold: invalidate_probe(runtime_root, phase_method, fold, model_seed)
                    complete_or_run(
                        run_dir, job, validator, invalidator,
                        lambda action=action, fold=fold: run_neural_job(runtime_root, action, fold, model_seed),
                    )
            aggregate_box: dict[str, Any] = {}
            def aggregate_action() -> None:
                aggregate_box["value"] = aggregate_oof(run_dir, runtime_root, patients, model_seed, n_folds)
            complete_or_run(
                run_dir, "aggregate",
                lambda: (run_dir / "final" / "fold_metrics.csv").exists() and (run_dir / "final" / "fold_summary.csv").exists(),
                lambda: None,
                aggregate_action,
            )
            if "value" not in aggregate_box:
                aggregate_box["value"] = aggregate_oof(run_dir, runtime_root, patients, model_seed, n_folds)
            risks, g_values, fold_frame, _ = aggregate_box["value"]
            summary_box: dict[str, Any] = {}
            def bootstrap_action() -> None:
                summary_box["value"] = summarize(
                    patients, risks, g_values, fold_frame,
                    int(config["statistics"]["bootstrap_replicates"]), outer_seed,
                    int(config["decision"]["min_positive_folds_per_metric"]),
                )
                atomic_csv(summary_box["value"][2], run_dir / "final" / "bootstrap_summary.csv")
            complete_or_run(
                run_dir, "bootstrap",
                lambda: (run_dir / "final" / "bootstrap_summary.csv").exists() and len(pd.read_csv(run_dir / "final" / "bootstrap_summary.csv")) == 3,
                lambda: (run_dir / "final" / "bootstrap_summary.csv").unlink(missing_ok=True),
                bootstrap_action,
            )
            if "value" not in summary_box:
                summary_box["value"] = summarize(
                    patients, risks, g_values, fold_frame,
                    int(config["statistics"]["bootstrap_replicates"]), outer_seed,
                    int(config["decision"]["min_positive_folds_per_metric"]),
                )
            methods, decision, bootstrap_frame = summary_box["value"]
            def decision_action() -> None:
                write_reports(repo_root, run_dir, config_path, config, split_lock, patients, fold_frame, bootstrap_frame, methods, decision, started_at, time.perf_counter() - getattr(run, "_started_perf", time.perf_counter()), len(read_state(run_dir).get("completed_jobs", [])))
            complete_or_run(
                run_dir, "decision",
                lambda: all((run_dir / "final" / name).exists() for name in ("RESULTS.md", "summary.json", "decision.json", "fold_metrics.csv", "bootstrap_summary.csv", "fold_summary.csv", "run_manifest.json", "REPRODUCIBILITY.md")),
                lambda: None,
                decision_action,
            )
            completed_at = utc_now()
            if prior_status == "COMPLETE" and isinstance(prior_elapsed, (int, float)):
                elapsed = float(prior_elapsed)
                # A completed run is immutable on resume.  Derive its end time
                # from the persisted start and duration so an older inconsistent
                # state cannot produce a misleading report timestamp.
                try:
                    completed_at = (datetime.fromisoformat(started_at) + timedelta(seconds=elapsed)).isoformat()
                except (TypeError, ValueError, OverflowError):
                    completed_at = utc_now()
            else:
                elapsed = elapsed_from_timestamps(started_at, completed_at)
                if elapsed is None:
                    elapsed = time.perf_counter() - getattr(run, "_started_perf", time.perf_counter())
            # Rewrite reports with one persisted end time and final job count.
            write_reports(repo_root, run_dir, config_path, config, split_lock, patients, fold_frame, bootstrap_frame, methods, decision, started_at, elapsed, len(read_state(run_dir).get("completed_jobs", [])), ended_at=completed_at)
            update_state(run_dir, status="COMPLETE", current_job=None, end_time=completed_at, elapsed_seconds=float(elapsed))
            print(json.dumps({"phase1r": "COMPLETE", "decision": decision, "reports": "reports/phase1r_confirmatory_20260828"}, ensure_ascii=False, indent=2), flush=True)
            if args.cleanup_runtime:
                cleanup_runtime(run_dir)
            return 0
        except Phase1RInvalid as exc:
            current = read_state(run_dir).get("current_job")
            mark_job_failed(run_dir, current or "preflight", exc, "INVALID")
            atomic_text(run_dir / "failure_traceback.txt", traceback.format_exc())
            print(f"Phase1R INVALID: {exc}", file=sys.stderr, flush=True)
            return 2
        except Phase1RBlocked as exc:
            current = read_state(run_dir).get("current_job")
            mark_job_failed(run_dir, current or "preflight", exc, "BLOCKED")
            atomic_text(run_dir / "failure_traceback.txt", traceback.format_exc())
            print(f"Phase1R BLOCKED: {exc}", file=sys.stderr, flush=True)
            return 3
        except KeyboardInterrupt:
            current = read_state(run_dir).get("current_job")
            exc = Phase1RBlocked("Interrupted by user; completed jobs and runtime artifacts were retained")
            mark_job_failed(run_dir, current or "unknown", exc, "BLOCKED")
            atomic_text(run_dir / "failure_traceback.txt", traceback.format_exc())
            print("Phase1R BLOCKED: interrupted", file=sys.stderr, flush=True)
            return 130
        except BaseException as exc:
            current = read_state(run_dir).get("current_job")
            blocked = Phase1RBlocked(f"Unhandled execution error: {exc}")
            mark_job_failed(run_dir, current or "unknown", blocked, "BLOCKED")
            atomic_text(run_dir / "failure_traceback.txt", traceback.format_exc())
            print(f"Phase1R BLOCKED: {exc}", file=sys.stderr, flush=True)
            return 3
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase1R local confirmatory runner")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1r/confirmatory.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/phase1r_confirmatory_20260828"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cleanup-runtime", action="store_true")
    args = parser.parse_args()
    run._started_perf = time.perf_counter()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())


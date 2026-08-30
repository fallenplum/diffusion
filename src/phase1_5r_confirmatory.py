"""Phase 1.5R confirmatory benchmark repair.

This runner is intentionally separate from the original Phase 1.5 benchmark.
It first validates the original Phase1R B0/B1 anchors and aborts before any
new model is fitted if either anchor is outside the pre-registered tolerance.
The repaired benchmark keeps raw UNI2-h features at 1536 dimensions, uses a
shared stochastic token budget (1024 in the main protocol), and computes the
survival objective on complete patient-level risk sets.

Only compact reports are copied to ``reports/phase1_5r_confirmatory_repair``.
The runtime tree contains checkpoints, OOF predictions, sampled features and
audit details, and is ignored by Git.  The input data root is resolved through
``DIFFUSION_DATA_ROOT`` (falling back to the local ``TCGA`` directory).
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import phase1_5_benchmark as p15
from phase1_utils import censoring_km, fit_cox, harrell_c, uno_c


EXPERIMENT_ID = "phase1_5r_confirmatory_repair"
SEEDS = (20260723, 20260724, 20260725)
FOLDS = (1, 2, 3, 4, 5)
INPUT_DIM = 1536
MAIN_TOKENS = 1024
EVAL_REPEATS = 5
TUNING_EVAL_REPEATS = 3
HIDDEN_DIM = 256
TOKEN_GRID = (128, 256, 512, 1024)
PROJECTION_GRID = ("random64", "learned64", "learned256", "learned512")
LR_GRID = (3e-4, 1e-4, 3e-5)
RANK_LAMBDA_GRID = (0.05, 0.2)
MAX_EPOCHS = 50
PATIENCE = 8
BOOTSTRAPS = 5000
RANDOM_PROJECTION_SEED = 20260828
RESERVOIR_PATCHES = 50000
# Batch sizes affect only computation/gradient accumulation, not the
# information budget: every comparison still consumes the same raw 1536D
# features and the same K-token sample.  The larger batches keep the GPU
# saturated on the confirmatory runner and avoid thousands of tiny launches.
PATIENT_BATCH = 32
SLIDE_BATCH = 32

RUNNABLE_NEURAL = (
    "B1_GatedABMIL",
    "B2_PlainABMIL",
    "B3_TransformerMIL_reimplementation",
    "B6_SpatialNeighborhoodMIL",
    "B7b_GatedABMIL_CoxRank",
)


def utc_now() -> str:
    return p15.utc_now()


def output_dir(root: Path) -> Path:
    return root / "outputs" / EXPERIMENT_ID


def acquire_run_lock(out: Path) -> tuple[Path, int] | None:
    """Acquire an exclusive lock so wrapper retries cannot run in parallel."""
    lock_path = out / "run.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # A force-killed process can leave the marker behind.  Remove it only
        # when the recorded owner PID is no longer alive; a live owner keeps
        # the lock and causes this invocation to exit without touching state.
        try:
            text = lock_path.read_text(encoding="utf-8")
            pid_line = next((line for line in text.splitlines() if line.startswith("pid=")), "")
            owner_pid = int(pid_line.split("=", 1)[1]) if pid_line else -1
            if owner_pid <= 0:
                raise ProcessLookupError("lock owner PID is missing")
            os.kill(owner_pid, 0)
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                return None
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return None
        else:
            return None
    os.write(fd, f"pid={os.getpid()}\nstarted={utc_now()}\n".encode("utf-8"))
    return lock_path, fd


def release_run_lock(lock: tuple[Path, int] | None) -> None:
    if lock is None:
        return
    lock_path, fd = lock
    try:
        os.close(fd)
    finally:
        lock_path.unlink(missing_ok=True)


def stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def load_frozen(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    patients, slides, split_path, slide_path = p15.load_source_frames(root)
    if len(patients) != 487 or int(patients.dss_event.sum()) != 65 or len(slides) != 494:
        raise RuntimeError(
            f"unexpected frozen cohort: patients={len(patients)}, events={int(patients.dss_event.sum())}, slides={len(slides)}"
        )
    if set(patients.fold) != set(FOLDS):
        raise RuntimeError(f"unexpected frozen folds: {sorted(set(patients.fold))}")
    return patients, slides, split_path, slide_path


def state_path(out: Path) -> Path:
    return out / "run_state.json"


def load_state(out: Path, resume: bool) -> dict[str, Any]:
    path = state_path(out)
    if resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state = {
        "experiment_id": EXPERIMENT_ID,
        "status": "RUNNING",
        "stage": "initializing",
        "current_job": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "start_time": utc_now(),
        "updated_at": utc_now(),
    }
    p15.atomic_json(path, state)
    return state


def save_state(out: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    p15.atomic_json(state_path(out), state)


def mark_start(out: Path, state: dict[str, Any], job: str, stage: str) -> None:
    state["current_job"] = job
    state["stage"] = stage
    save_state(out, state)


def mark_complete(out: Path, state: dict[str, Any], job: str) -> None:
    if job not in state["completed_jobs"]:
        state["completed_jobs"].append(job)
    state["current_job"] = None
    save_state(out, state)


def mark_failure(out: Path, state: dict[str, Any], job: str, exc: BaseException) -> None:
    state["failed_jobs"].append({"job": job, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    save_state(out, state)


def run_anchor_gate(root: Path, out: Path, patients: pd.DataFrame) -> pd.DataFrame:
    """Validate Phase1R's original B0/B1 outputs before any new fitting."""
    anchor_path = out / "anchor" / "anchor_reproduction.csv"
    gate_path = out / "anchor" / "anchor_gate.json"
    if anchor_path.exists() and gate_path.exists():
        return pd.read_csv(anchor_path)

    historical_path = root / "reports" / "phase1r_confirmatory_20260828" / "summary.json"
    aggregate_path = root / "outputs" / "phase1r_confirmatory_20260828" / "runtime_root" / "aggregate_oof_predictions.csv"
    if not historical_path.exists() or not aggregate_path.exists():
        raise RuntimeError("Phase1R historical summary or original OOF aggregate is missing")
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    original = pd.read_csv(aggregate_path)
    original["patient_id"] = original["patient_id"].astype(str).str.upper()
    expected = set(patients.patient_id.astype(str))
    rows: list[dict[str, Any]] = []
    for method, hist_name in (("B0_mean", "B0_mean"), ("B1_ABMIL", "B1_ABMIL")):
        frame = original.loc[original.method == method].copy()
        if len(frame) != len(patients) or set(frame.patient_id) != expected or frame.patient_id.duplicated().any():
            raise RuntimeError(f"Phase1R anchor {method} does not have one original OOF row per patient")
        frame = frame.sort_values("patient_id")
        h = harrell_c(frame.dss_months.to_numpy(float), frame.dss_status.to_numpy(int), frame.oof_risk.to_numpy(float))
        u = uno_c(
            frame.dss_months.to_numpy(float),
            frame.dss_status.to_numpy(int),
            frame.oof_risk.to_numpy(float),
            frame.censor_survival_outer_train.to_numpy(float),
        )
        historical_method = historical["methods"][hist_name]
        historical_h = float(historical_method["oof_harrell_c"])
        historical_u = float(historical_method["oof_uno_c"])
        delta_h = abs(float(h) - historical_h)
        delta_u = abs(float(u) - historical_u)
        rows.append(
            {
                "method": method,
                "historical_harrell": historical_h,
                "new_harrell": float(h),
                "historical_uno": historical_u,
                "new_uno": float(u),
                "abs_delta_harrell": delta_h,
                "abs_delta_uno": delta_u,
                "gate_pass": bool(delta_h <= 0.02 and delta_u <= 0.02),
                "source": "Phase1R original implementation/runtime OOF; frozen split revalidated",
            }
        )
    anchor = pd.DataFrame(rows)
    p15.atomic_csv(anchor, anchor_path)
    passed = bool(anchor.gate_pass.all())
    p15.atomic_json(
        gate_path,
        {
            "status": "PASS" if passed else "STOP_PHASE1_5R",
            "tolerance": 0.02,
            "historical_source": str(historical_path.relative_to(root)),
            "original_oof_source": str(aggregate_path.relative_to(root)),
            "split_hash": p15.canonical_split_hash(patients),
            "checked_at": utc_now(),
        },
    )
    if not passed:
        raise RuntimeError("STOP_PHASE1_5R: one or more Phase1R anchors exceed the ±0.02 gate")
    return anchor


class SlideStore:
    """Lazy memmap access plus a shared, deterministic sampling protocol."""

    def __init__(self, root: Path, slides: pd.DataFrame):
        self.root = root
        self.data = p15.data_root(root)
        self.slides = slides.sort_values(["patient_id", "slide_id"]).reset_index(drop=True).copy()
        self.by_patient: dict[str, list[int]] = defaultdict(list)
        self.bags: dict[int, np.ndarray] = {}
        self.coords: dict[int, np.ndarray] = {}
        self.bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Keep only indices/coordinate metadata in the cache.  Retaining every
        # sampled float32 feature block across seeds/repeats would grow to
        # multiple gigabytes; feature rows are cheap to re-read from memmap.
        self.sampling_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # Keep only a bounded number of sampling metadata entries.  A full
        # inner-search epoch grid can otherwise retain tens of thousands of
        # coordinate arrays in one process; eviction is safe because the
        # sample key deterministically regenerates the same indices.
        self.sampling_cache_limit = 2048
        for idx, row in self.slides.iterrows():
            self.by_patient[str(row.patient_id)].append(int(idx))

    def clear_cache(self) -> None:
        """Release deterministic sampling metadata between independent jobs."""
        self.sampling_cache.clear()
        # np.load(..., mmap_mode="r") keeps an OS mapping alive while the
        # memmap object is referenced. Close mappings between independent jobs
        # so repeated folds cannot accumulate a multi-GB working set.
        for mapping in list(self.bags.values()) + list(self.coords.values()):
            mmap_obj = getattr(mapping, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
        self.bags.clear()
        self.coords.clear()
        self.bounds.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, slide_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if slide_idx not in self.bags:
            row = self.slides.iloc[slide_idx]
            bag_path = p15.resolve_input_path(self.root, self.data, row.bag_path)
            coord_path = p15.resolve_input_path(self.root, self.data, row.coords_path)
            self.bags[slide_idx] = np.load(bag_path, mmap_mode="r", allow_pickle=False)
            self.coords[slide_idx] = np.load(coord_path, mmap_mode="r", allow_pickle=False)
            full_coords = np.asarray(self.coords[slide_idx], dtype=np.float64)
            self.bounds[slide_idx] = (full_coords.min(axis=0), full_coords.max(axis=0))
        return self.bags[slide_idx], self.coords[slide_idx]

    def sample(
        self,
        slide_idx: int,
        token_budget: int,
        seed: int,
        epoch: int,
        repeat: int = 0,
        training: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = self.slides.iloc[slide_idx]
        key = (int(slide_idx), int(token_budget), int(seed), int(epoch), int(repeat), bool(training))
        if key in self.sampling_cache:
            indices, norm, density = self.sampling_cache[key]
            bag, _ = self._load(slide_idx)
            return np.asarray(bag[indices], dtype=np.float32), norm, density
        bag, coords = self._load(slide_idx)
        n = int(len(bag))
        if n <= token_budget:
            indices = np.arange(n, dtype=np.int64)
        else:
            rng = np.random.default_rng(stable_seed(seed, row.slide_id, token_budget, epoch, repeat, "train" if training else "eval"))
            indices = np.sort(rng.choice(n, token_budget, replace=False)).astype(np.int64)
        x = np.asarray(bag[indices], dtype=np.float32)
        raw = np.asarray(coords[indices], dtype=np.float32)
        low, high = self.bounds[slide_idx]
        norm = (raw - low.astype(np.float32)) / np.maximum((high - low).astype(np.float32), 1.0)
        density = local_density(norm)
        norm = norm.astype(np.float32)
        density = density.astype(np.float32)
        if len(self.sampling_cache) >= self.sampling_cache_limit:
            self.sampling_cache.pop(next(iter(self.sampling_cache)))
        self.sampling_cache[key] = (indices, norm, density)
        return x, norm, density


def local_density(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """Cheap coordinate-only neighbourhood density, with no tissue labels."""
    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    # A sorted sliding window avoids an O(n^2) distance matrix while still
    # encoding local absolute spatial concentration in the sampled WSI.
    order = np.argsort(coords[:, 0] + 1e-3 * coords[:, 1])
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    left = np.maximum(rank - min(k, n - 1), 0)
    right = np.minimum(rank + min(k, n - 1), n - 1)
    span = np.abs(coords[order[right], 0] - coords[order[left], 0]) + np.abs(coords[order[right], 1] - coords[order[left], 1])
    density = 1.0 / np.maximum(span, 1e-3)
    density = (density - density.mean()) / max(float(density.std()), 1e-6)
    result = np.empty(n, dtype=np.float32)
    result[order] = density.astype(np.float32)
    return result


def padded_slide_batch(entries: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]], device: torch.device):
    max_len = max(len(item[1]) for item in entries)
    dim = int(entries[0][1].shape[1])
    x = np.zeros((len(entries), max_len, dim), dtype=np.float32)
    c = np.zeros((len(entries), max_len, 2), dtype=np.float32)
    d = np.zeros((len(entries), max_len, 1), dtype=np.float32)
    pad = np.ones((len(entries), max_len), dtype=bool)
    owners = []
    for row_idx, (owner, features, coords, density) in enumerate(entries):
        n = len(features)
        x[row_idx, :n] = features
        c[row_idx, :n] = coords
        d[row_idx, :n, 0] = density
        pad[row_idx, :n] = False
        owners.append(owner)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(c).to(device),
        torch.from_numpy(d).to(device),
        torch.from_numpy(pad).to(device),
        owners,
    )


def masked_softmax(logits: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(pad_mask.unsqueeze(-1), -1e9)
    weights = torch.softmax(masked, dim=1)
    return weights.masked_fill(pad_mask.unsqueeze(-1), 0.0)


class GatedABMIL(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = HIDDEN_DIM, gated: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gated = gated
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, latent_dim), nn.GELU())
        self.attention_v = nn.Linear(latent_dim, min(128, latent_dim))
        self.attention_u = nn.Linear(latent_dim, min(128, latent_dim)) if gated else None
        self.attention_w = nn.Linear(min(128, latent_dim), 1)
        self.cox_head = nn.Linear(latent_dim, 1)

    def forward(self, features, coords=None, density=None, pad_mask=None):
        tokens = self.projection(features)
        v = torch.tanh(self.attention_v(tokens))
        if self.gated:
            h = v * torch.sigmoid(self.attention_u(tokens))
        else:
            h = v
        logits = self.attention_w(h)
        weights = masked_softmax(logits, pad_mask)
        pooled = torch.sum(weights * tokens, dim=1)
        return pooled, weights.squeeze(-1)


class SpatialNeighborhoodMIL(GatedABMIL):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = HIDDEN_DIM):
        super().__init__(input_dim, latent_dim, gated=True)
        self.coord_mlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, latent_dim))
        self.density_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, latent_dim))

    def forward(self, features, coords=None, density=None, pad_mask=None):
        tokens = self.projection(features) + self.coord_mlp(coords) + self.density_mlp(density)
        v = torch.tanh(self.attention_v(tokens))
        h = v * torch.sigmoid(self.attention_u(tokens))
        weights = masked_softmax(self.attention_w(h), pad_mask)
        pooled = torch.sum(weights * tokens, dim=1)
        return pooled, weights.squeeze(-1)


class TransformerMIL(nn.Module):
    """Transparent generic TransformerMIL reimplementation, not official TransMIL."""

    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = HIDDEN_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, latent_dim), nn.GELU())
        self.coord_mlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8 if latent_dim % 8 == 0 else 4,
            dim_feedforward=latent_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(latent_dim)
        self.cox_head = nn.Linear(latent_dim, 1)

    def forward(self, features, coords=None, density=None, pad_mask=None):
        tokens = self.projection(features) + self.coord_mlp(coords)
        encoded = self.norm(self.transformer(tokens, src_key_padding_mask=pad_mask))
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        weights = valid.squeeze(-1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled, weights


def mamba_available() -> tuple[bool, str]:
    try:
        from mamba_ssm import Mamba  # type: ignore

        _ = Mamba
        return True, "mamba_ssm import succeeded"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


class MambaMIL(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = HIDDEN_DIM):
        super().__init__()
        from mamba_ssm import Mamba  # type: ignore

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, latent_dim), nn.GELU())
        self.mamba = Mamba(d_model=latent_dim, d_state=16, d_conv=4, expand=2)
        self.cox_head = nn.Linear(latent_dim, 1)

    def forward(self, features, coords=None, density=None, pad_mask=None):
        tokens = self.projection(features)
        tokens = tokens.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        encoded = self.mamba(tokens)
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        weights = valid.squeeze(-1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled, weights


def build_model(method: str, projection: str = "learned256") -> tuple[nn.Module, Any]:
    if projection == "random64":
        model_input, latent = 64, 64
        preprocessor = FixedRandomProjection(INPUT_DIM, latent, RANDOM_PROJECTION_SEED)
    elif projection == "learned64":
        model_input, latent, preprocessor = INPUT_DIM, 64, None
    elif projection == "learned512":
        model_input, latent, preprocessor = INPUT_DIM, 512, None
    else:
        model_input, latent, preprocessor = INPUT_DIM, HIDDEN_DIM, None
    if method in {"B1_GatedABMIL", "B7a_GatedABMIL_Cox", "B7b_GatedABMIL_CoxRank"}:
        model = GatedABMIL(model_input, latent, gated=True)
    elif method == "B2_PlainABMIL":
        model = GatedABMIL(model_input, latent, gated=False)
    elif method == "B3_TransformerMIL_reimplementation":
        model = TransformerMIL(model_input, latent)
    elif method == "B4_MambaMIL":
        model = MambaMIL(model_input, latent)
    elif method == "B6_SpatialNeighborhoodMIL":
        model = SpatialNeighborhoodMIL(model_input, latent)
    else:
        raise ValueError(f"unknown neural method: {method}")
    return model, preprocessor


class FixedRandomProjection:
    def __init__(self, input_dim: int, output_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.matrix = (rng.normal(size=(input_dim, output_dim)).astype(np.float32) / math.sqrt(output_dim))
        self._torch_matrices: dict[str, torch.Tensor] = {}

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float32) @ self.matrix

    def transform_tensor(self, features: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Apply the fixed matrix on-device to avoid a CPU bottleneck."""
        key = str(device)
        matrix = self._torch_matrices.get(key)
        if matrix is None:
            matrix = torch.as_tensor(self.matrix, dtype=torch.float32, device=device)
            self._torch_matrices[key] = matrix
        return features.matmul(matrix)


def forward_patients(
    model: nn.Module,
    store: SlideStore,
    patient_ids: list[str],
    token_budget: int,
    seed: int,
    epoch: int,
    device: torch.device,
    training: bool,
    repeat: int = 0,
    preprocessor: Any = None,
    collect_attention: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    model_dim = int(getattr(model, "input_dim", INPUT_DIM))
    result = torch.zeros((len(patient_ids), int(getattr(model, "latent_dim", HIDDEN_DIM))), device=device)
    counts = torch.zeros(len(patient_ids), device=device)
    attention_rows: list[dict[str, Any]] = []
    lookup = {pid: idx for idx, pid in enumerate(patient_ids)}
    # Store only slide identities until a compute batch is ready.  The former
    # implementation materialized every sampled 1536D block for a patient
    # cohort before forwarding; with K=1024 this could retain multiple GB per
    # forward and eventually exhaust host memory during the inner search.
    records: list[tuple[int, int]] = []
    fixed_projection = isinstance(preprocessor, FixedRandomProjection)
    for pid in patient_ids:
        owner = lookup[pid]
        for slide_idx in store.by_patient[pid]:
            records.append((owner, slide_idx))
    for start in range(0, len(records), SLIDE_BATCH):
        entries: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for owner, slide_idx in records[start : start + SLIDE_BATCH]:
            x, c, density = store.sample(slide_idx, token_budget, seed, epoch, repeat, training)
            if preprocessor is not None and not fixed_projection:
                x = preprocessor(x)
            expected_dim = INPUT_DIM if fixed_projection else model_dim
            if x.shape[1] != expected_dim:
                raise RuntimeError(f"feature projection mismatch: {x.shape[1]} vs expected {expected_dim}")
            entries.append((owner, x, c, density))
        x, c, density, pad, owners = padded_slide_batch(entries, device)
        if fixed_projection:
            x = preprocessor.transform_tensor(x, device)
        if x.shape[2] != model_dim:
            raise RuntimeError(f"feature projection mismatch after batch transform: {x.shape[2]} vs model input {model_dim}")
        z, weights = model(x, c, density, pad)
        for row_idx, owner in enumerate(owners):
            result[owner] = result[owner] + z[row_idx]
            counts[owner] = counts[owner] + 1.0
            if collect_attention:
                slide_idx = store.by_patient[patient_ids[owner]][0]
                valid_n = int((~pad[row_idx]).sum().item())
                attention_rows.append(
                    {
                        "patient_id": patient_ids[owner],
                        "slide_id": str(store.slides.iloc[slide_idx].slide_id),
                        "attention_entropy": float((-(weights[row_idx, :valid_n] * (weights[row_idx, :valid_n].clamp_min(1e-8).log())).sum()).detach().cpu()),
                        "top5pct_mass": float(torch.topk(weights[row_idx, :valid_n], max(1, int(math.ceil(valid_n * 0.05)))).values.sum().detach().cpu()),
                        "coord_spread": float(c[row_idx, :valid_n].std(dim=0).mean().detach().cpu()),
                    }
                )
        del entries, x, c, density, pad, owners, z, weights
    return result / counts.clamp_min(1.0).unsqueeze(-1), attention_rows


@torch.no_grad()
def predict_risk(
    model: nn.Module,
    store: SlideStore,
    frame: pd.DataFrame,
    token_budget: int,
    seed: int,
    device: torch.device,
    eval_repeats: int,
    preprocessor: Any = None,
    epoch: int = 0,
    collect_attention: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model.eval()
    patient_ids = frame.patient_id.astype(str).tolist()
    risks = []
    attention: list[dict[str, Any]] = []
    for repeat in range(eval_repeats):
        z, rows = forward_patients(
            model,
            store,
            patient_ids,
            token_budget,
            seed,
            epoch,
            device,
            training=False,
            repeat=repeat,
            preprocessor=preprocessor,
            collect_attention=collect_attention and repeat == 0,
        )
        risk = model.cox_head(z).squeeze(-1).detach().cpu().numpy()
        risks.append(risk)
        if repeat == 0:
            attention.extend(rows)
    return np.mean(np.stack(risks), axis=0), attention


def cox_risk_coefficients(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    coeff = np.zeros(len(times), dtype=np.float64)
    event_indices = np.flatnonzero(events == 1)
    if not len(event_indices):
        raise ValueError("training set has no DSS events")
    for i in event_indices:
        mask = times >= times[i]
        logits = risks[mask] - np.max(risks[mask])
        probs = np.exp(logits)
        probs /= np.maximum(probs.sum(), 1e-12)
        coeff[mask] -= probs
        coeff[i] += 1.0
    return coeff / len(event_indices)


def rank_risk_coefficients(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    pair_i, pair_j = np.where((events[:, None] == 1) & (times[:, None] < times[None, :]))
    coeff = np.zeros(len(times), dtype=np.float64)
    if not len(pair_i):
        return coeff
    delta = risks[pair_i] - risks[pair_j]
    grad = expit(-delta) / len(pair_i)
    np.add.at(coeff, pair_i, -grad)
    np.add.at(coeff, pair_j, grad)
    return coeff


def cox_loss_value(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    value = 0.0
    count = 0
    for i in np.flatnonzero(events == 1):
        value += float(np.logaddexp.reduce(risks[times >= times[i]]) - risks[i])
        count += 1
    return value / max(count, 1)


def train_epoch_full_risk(
    model: nn.Module,
    store: SlideStore,
    frame: pd.DataFrame,
    token_budget: int,
    seed: int,
    epoch: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    rank_lambda: float = 0.0,
    preprocessor: Any = None,
) -> float:
    """Exact full-risk-set gradient using two passes and analytic risk weights.

    The first pass obtains current risk scores without retaining all token
    graphs.  The Cox/rank derivative with respect to each patient risk is then
    applied to chunked second-pass forwards, yielding the exact gradient for
    the same stochastic samples without treating patches as samples.
    """
    model.eval()  # all repaired models use dropout=0; sampling is stochastic in the bag sampler
    ids = frame.patient_id.astype(str).tolist()
    with torch.no_grad():
        z, _ = forward_patients(model, store, ids, token_budget, seed, epoch, device, True, preprocessor=preprocessor)
        risks_np = model.cox_head(z).squeeze(-1).cpu().numpy()
    times = frame.dss_time.to_numpy(float)
    events = frame.dss_event.to_numpy(int)
    coeff = cox_risk_coefficients(times, events, risks_np)
    if rank_lambda:
        coeff = coeff + rank_lambda * rank_risk_coefficients(times, events, risks_np)
    optimizer.zero_grad(set_to_none=True)
    patient_batch = max(1, PATIENT_BATCH // 2 if isinstance(model, TransformerMIL) else PATIENT_BATCH)
    for start in range(0, len(ids), patient_batch):
        z_chunk, _ = forward_patients(
            model,
            store,
            ids[start : start + patient_batch],
            token_budget,
            seed,
            epoch,
            device,
            True,
            preprocessor=preprocessor,
        )
        risk_chunk = model.cox_head(z_chunk).squeeze(-1)
        loss = torch.sum(torch.from_numpy(coeff[start : start + patient_batch]).to(device, dtype=risk_chunk.dtype) * risk_chunk)
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return cox_loss_value(times, events, risks_np)


def fit_neural_candidate(
    method: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    store: SlideStore,
    token_budget: int,
    seed: int,
    fold: int,
    device: torch.device,
    lr: float,
    rank_lambda: float = 0.0,
    projection: str = "learned256",
    max_epochs: int = MAX_EPOCHS,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(stable_seed(seed, fold, method, projection, lr, rank_lambda))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(stable_seed(seed, fold, method, projection, lr, rank_lambda))
    model, preprocessor = build_model(method, projection)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        loss = train_epoch_full_risk(model, store, train, token_budget, seed, epoch, device, optimizer, rank_lambda, preprocessor)
        val_risk, _ = predict_risk(model, store, valid, token_budget, seed, device, TUNING_EVAL_REPEATS, preprocessor, epoch=epoch)
        val_c = harrell_c(valid.dss_time.to_numpy(float), valid.dss_event.to_numpy(int), val_risk)
        history.append({"epoch": epoch, "train_cox_loss": loss, "val_harrell_c": val_c, "lr": lr, "rank_lambda": rank_lambda})
        if np.isfinite(val_c) and val_c > best_score + 1e-5:
            best_score = float(val_c)
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"{method} produced no valid inner-validation checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_inner_harrell": best_score, "best_epoch": best_epoch, "history": history, "preprocessor": preprocessor}


def refit_neural(
    method: str,
    train: pd.DataFrame,
    store: SlideStore,
    token_budget: int,
    seed: int,
    fold: int,
    device: torch.device,
    lr: float,
    rank_lambda: float,
    selected_epoch: int,
    projection: str = "learned256",
) -> tuple[nn.Module, Any, float, float]:
    torch.manual_seed(stable_seed("refit", seed, fold, method, projection, lr, rank_lambda))
    model, preprocessor = build_model(method, projection)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    started = time.perf_counter()
    for epoch in range(1, max(1, int(selected_epoch)) + 1):
        train_epoch_full_risk(model, store, train, token_budget, seed, epoch, device, optimizer, rank_lambda, preprocessor)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return model, preprocessor, elapsed, float(peak)


def inner_split(patients: pd.DataFrame, outer_fold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_train = patients.loc[patients.fold != outer_fold].copy().reset_index(drop=True)
    outer_test = patients.loc[patients.fold == outer_fold].copy().reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + outer_fold)
    train_idx, val_idx = next(splitter.split(np.zeros(len(outer_train)), outer_train.dss_event.to_numpy(int)))
    return outer_train.iloc[train_idx].reset_index(drop=True), outer_train.iloc[val_idx].reset_index(drop=True), outer_test


def tune_neural(
    method: str,
    outer_train: pd.DataFrame,
    store: SlideStore,
    token_budget: int,
    seed: int,
    fold: int,
    device: torch.device,
    projection: str = "learned256",
    rank: bool = False,
) -> dict[str, Any]:
    inner_train, inner_val, _ = inner_split(pd.concat([outer_train], ignore_index=True), outer_fold=fold, seed=seed)
    candidates = []
    lambdas = RANK_LAMBDA_GRID if rank else (0.0,)
    for lr in LR_GRID:
        for rank_lambda in lambdas:
            model, meta = fit_neural_candidate(method, inner_train, inner_val, store, token_budget, seed, fold, device, lr, rank_lambda, projection)
            candidates.append({"score": meta["best_inner_harrell"], "lr": lr, "rank_lambda": rank_lambda, "epoch": meta["best_epoch"], "model": model, "meta": meta})
    candidates.sort(key=lambda row: (row["score"], -row["lr"], -row["rank_lambda"]), reverse=True)
    return candidates[0]


def neural_job(
    root: Path,
    out: Path,
    state: dict[str, Any],
    store: SlideStore,
    patients: pd.DataFrame,
    method: str,
    seed: int,
    fold: int,
    device: torch.device,
    token_budget: int = MAIN_TOKENS,
    projection: str = "learned256",
    rank: bool = False,
    output_method: str | None = None,
) -> dict[str, Any] | None:
    output_method = output_method or method
    job = f"{output_method}/seed_{seed}/fold_{fold}/K{token_budget}/{projection}"
    pred_path = out / "oof_predictions" / output_method / f"seed_{seed}_fold_{fold}_K{token_budget}_{projection}.csv"
    metric_path = out / "tables" / "job_metrics" / output_method / f"seed_{seed}_fold_{fold}_K{token_budget}_{projection}.json"
    if job in state["completed_jobs"] and pred_path.exists() and metric_path.exists():
        return json.loads(metric_path.read_text(encoding="utf-8"))
    mark_start(out, state, job, "main_neural")
    try:
        outer_train = patients.loc[patients.fold != fold].copy().reset_index(drop=True)
        outer_test = patients.loc[patients.fold == fold].copy().reset_index(drop=True)
        selected = tune_neural(method, outer_train, store, token_budget, seed, fold, device, projection, rank)
        model, preprocessor, train_seconds, peak = refit_neural(method, outer_train, store, token_budget, seed, fold, device, selected["lr"], selected["rank_lambda"], selected["epoch"], projection)
        checkpoint_path = out / "checkpoints" / output_method / f"seed_{seed}" / f"fold_{fold}_K{token_budget}_{projection}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "method": method, "projection": projection, "token_budget": token_budget, "seed": seed, "fold": fold, "selected_epoch": int(selected["epoch"])}, checkpoint_path)
        risk, attention = predict_risk(model, store, outer_test, token_budget, seed, device, EVAL_REPEATS, preprocessor, collect_attention=True)
        censor = censoring_km(outer_train.dss_time.to_numpy(float), outer_train.dss_event.to_numpy(int))
        g = censor(outer_test.dss_time.to_numpy(float), left_limit=True)
        pred = outer_test[["patient_id", "fold", "dss_time", "dss_event"]].copy()
        pred["seed"] = seed
        pred["risk_score"] = risk
        pred["censor_survival_outer_train"] = g
        pred["method"] = output_method
        pred["token_budget"] = token_budget
        pred["projection"] = projection
        p15.atomic_csv(pred, pred_path)
        metric = {
            "method": output_method,
            "base_method": method,
            "seed": seed,
            "fold": fold,
            "token_budget": token_budget,
            "projection": projection,
            "sampling": "stochastic_train_deterministic_5_repeat_eval",
            "input_dim": INPUT_DIM if projection != "random64" else 64,
            "learned_projection_dim": int(getattr(model, "latent_dim", 0)),
            "selected_lr": float(selected["lr"]),
            "selected_rank_lambda": float(selected["rank_lambda"]),
            "selected_epoch": int(selected["epoch"]),
            "inner_val_harrell_c": float(selected["score"]),
            "n_test": len(outer_test),
            "n_events": int(outer_test.dss_event.sum()),
            "test_harrell_c": float(harrell_c(outer_test.dss_time, outer_test.dss_event, risk)),
            "test_uno_c": float(uno_c(outer_test.dss_time, outer_test.dss_event, risk, g)),
            "trainable_params": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
            "training_time_seconds": float(train_seconds),
            "peak_gpu_memory_gb": float(peak),
            "inference_time_seconds": float("nan"),
            "eval_sampling_repeats": EVAL_REPEATS,
            "provenance": "local_unified_implementation" if method != "B3_TransformerMIL_reimplementation" else "generic_transformer_survival_reimplementation_not_official_TransMIL",
        }
        p15.atomic_json(metric_path, metric)
        attention_path = out / "attention" / output_method / f"seed_{seed}_fold_{fold}_K{token_budget}_{projection}.csv"
        if attention:
            p15.atomic_csv(pd.DataFrame(attention), attention_path)
        mark_complete(out, state, job)
        store.clear_cache()
        return metric
    except Exception as exc:
        store.clear_cache()
        mark_failure(out, state, job, exc)
        raise


def patient_feature_matrix(store: SlideStore, patients: pd.DataFrame, token_budget: int | None, seed: int, repeats: int = EVAL_REPEATS) -> np.ndarray:
    values = []
    for pid in patients.patient_id.astype(str):
        repeat_values = []
        for repeat in range(1 if token_budget is None else repeats):
            slide_values = []
            for slide_idx in store.by_patient[pid]:
                bag, _ = store._load(slide_idx)
                if token_budget is None:
                    x = np.asarray(bag, dtype=np.float32)
                else:
                    x, _, _ = store.sample(slide_idx, token_budget, seed, epoch=0, repeat=repeat, training=False)
                slide_values.append(x.mean(axis=0, dtype=np.float64))
            repeat_values.append(np.stack(slide_values).mean(axis=0))
        values.append(np.stack(repeat_values).mean(axis=0))
    return np.asarray(values, dtype=np.float32)


def tune_linear(train: pd.DataFrame, valid: pd.DataFrame, x_train: np.ndarray, x_valid: np.ndarray, seed: int) -> tuple[int, float, float]:
    rows = []
    for components in (32, 64):
        n = min(components, len(train) - 1, x_train.shape[1])
        scaler = StandardScaler().fit(x_train)
        xt = scaler.transform(x_train)
        xv = scaler.transform(x_valid)
        pca = PCA(n_components=n, svd_solver="randomized", random_state=seed)
        xt = pca.fit_transform(xt)
        xv = pca.transform(xv)
        for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            beta = fit_cox(train.dss_time.to_numpy(float), train.dss_event.to_numpy(int), xt, alpha)
            score = harrell_c(valid.dss_time.to_numpy(float), valid.dss_event.to_numpy(int), xv @ beta)
            rows.append((float(score), -n, -alpha, n, alpha))
    rows.sort(reverse=True)
    return int(rows[0][3]), float(rows[0][4]), float(rows[0][0])


def run_mean_protocol(
    root: Path,
    out: Path,
    state: dict[str, Any],
    store: SlideStore,
    patients: pd.DataFrame,
    token_budget: int | None,
    method: str,
) -> list[dict[str, Any]]:
    # On resume, do not rebuild the patient feature matrix merely to discover
    # that every fold already has an atomic prediction file.  This is both
    # unnecessary work and a large host-memory spike for the full-WSI mean.
    expected = [
        (f"{method}/seed_{seed}/fold_{fold}", out / "oof_predictions" / method / f"seed_{seed}_fold_{fold}.csv")
        for seed in SEEDS
        for fold in FOLDS
    ]
    if all(job in state["completed_jobs"] and pred_path.exists() for job, pred_path in expected):
        return [
            json.loads((out / "tables" / "job_metrics" / method / f"seed_{seed}_fold_{fold}.json").read_text(encoding="utf-8"))
            for seed in SEEDS
            for fold in FOLDS
            if (out / "tables" / "job_metrics" / method / f"seed_{seed}_fold_{fold}.json").exists()
        ]
    cache_name = "full" if token_budget is None else str(token_budget)
    feature_path = out / "embeddings" / f"patient_features_{cache_name}.npy"
    if feature_path.exists():
        features = np.load(feature_path, allow_pickle=False)
    else:
        features = patient_feature_matrix(store, patients, token_budget, SEEDS[0], EVAL_REPEATS)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features)
    rows = []
    risks_all = []
    for seed in SEEDS:
        # Full mean has no sampling seed. Mean@K is recomputed per seed so the
        # sensitivity curve reflects the same deterministic eval protocol.
        x = features if token_budget is None else patient_feature_matrix(store, patients, token_budget, seed, EVAL_REPEATS)
        if token_budget is not None:
            np.save(out / "embeddings" / f"patient_features_{cache_name}_seed_{seed}.npy", x)
        for fold in FOLDS:
            job = f"{method}/seed_{seed}/fold_{fold}"
            pred_path = out / "oof_predictions" / method / f"seed_{seed}_fold_{fold}.csv"
            if job in state["completed_jobs"] and pred_path.exists():
                pred = pd.read_csv(pred_path)
                risks_all.append(pred)
                continue
            mark_start(out, state, job, "mean_baseline")
            outer_train = patients.loc[patients.fold != fold].copy().reset_index(drop=True)
            outer_test = patients.loc[patients.fold == fold].copy().reset_index(drop=True)
            inner_train, inner_val, _ = inner_split(patients, fold, seed)
            index = {str(pid): idx for idx, pid in enumerate(patients.patient_id.astype(str))}
            ti = np.asarray([index[str(pid)] for pid in inner_train.patient_id], dtype=int)
            vi = np.asarray([index[str(pid)] for pid in inner_val.patient_id], dtype=int)
            oi = np.asarray([index[str(pid)] for pid in outer_train.patient_id], dtype=int)
            si = np.asarray([index[str(pid)] for pid in outer_test.patient_id], dtype=int)
            components, alpha, inner_c = tune_linear(inner_train, inner_val, x[ti], x[vi], seed + fold)
            scaler = StandardScaler().fit(x[oi])
            pca = PCA(n_components=min(components, len(oi) - 1, x.shape[1]), svd_solver="randomized", random_state=seed + fold)
            xt = pca.fit_transform(scaler.transform(x[oi]))
            xs = pca.transform(scaler.transform(x[si]))
            beta = fit_cox(outer_train.dss_time.to_numpy(float), outer_train.dss_event.to_numpy(int), xt, alpha)
            risk = xs @ beta
            censor = censoring_km(outer_train.dss_time.to_numpy(float), outer_train.dss_event.to_numpy(int))
            g = censor(outer_test.dss_time.to_numpy(float), left_limit=True)
            pred = outer_test[["patient_id", "fold", "dss_time", "dss_event"]].copy()
            pred["seed"] = seed
            pred["risk_score"] = risk
            pred["censor_survival_outer_train"] = g
            pred["method"] = method
            pred["token_budget"] = "full" if token_budget is None else token_budget
            p15.atomic_csv(pred, pred_path)
            metric = {
                "method": method,
                "seed": seed,
                "fold": fold,
                "n_test": len(outer_test),
                "n_events": int(outer_test.dss_event.sum()),
                "test_harrell_c": float(harrell_c(outer_test.dss_time, outer_test.dss_event, risk)),
                "test_uno_c": float(uno_c(outer_test.dss_time, outer_test.dss_event, risk, g)),
                "pca_components": components,
                "selected_alpha": alpha,
                "inner_val_harrell_c": inner_c,
                "trainable_params": 0,
                "training_time_seconds": 0.0,
                "peak_gpu_memory_gb": 0.0,
                "inference_time_seconds": 0.0,
                "feature_protocol": "all_patch_mean" if token_budget is None else f"mean_{token_budget}_deterministic_5_repeat_sampling",
            }
            p15.atomic_json(out / "tables" / "job_metrics" / method / f"seed_{seed}_fold_{fold}.json", metric)
            rows.append(metric)
            risks_all.append(pred)
            mark_complete(out, state, job)
        store.clear_cache()
    return rows


def reservoir_patch_matrix(store: SlideStore, patient_ids: Iterable[str], seed: int) -> np.ndarray:
    """Train-only, label-independent reservoir for the PANTHER-style GMM."""
    rng = np.random.default_rng(stable_seed("gmm-reservoir", seed))
    reservoir: list[np.ndarray] = []
    seen = 0
    for pid in patient_ids:
        for slide_idx in store.by_patient[str(pid)]:
            bag, _ = store._load(slide_idx)
            n = len(bag)
            take = min(256, n)
            if n > take:
                indices = np.sort(rng.choice(n, take, replace=False))
                x = np.asarray(bag[indices], dtype=np.float32)
            else:
                x = np.asarray(bag, dtype=np.float32)
            for row in x:
                seen += 1
                if len(reservoir) < RESERVOIR_PATCHES:
                    reservoir.append(row.copy())
                else:
                    replacement = int(rng.integers(0, seen))
                    if replacement < RESERVOIR_PATCHES:
                        reservoir[replacement] = row.copy()
    if not reservoir:
        raise RuntimeError("empty train-only GMM reservoir")
    return np.asarray(reservoir, dtype=np.float32)


def gmm_patient_features(
    store: SlideStore,
    patients: pd.DataFrame,
    pca: PCA,
    gmm: GaussianMixture,
    seed: int,
    repeats: int = EVAL_REPEATS,
) -> np.ndarray:
    values = []
    for pid in patients.patient_id.astype(str):
        repeat_values = []
        for repeat in range(repeats):
            slide_values = []
            for slide_idx in store.by_patient[pid]:
                x, _, _ = store.sample(slide_idx, MAIN_TOKENS, seed, epoch=0, repeat=repeat, training=False)
                projected = pca.transform(x)
                posterior = gmm.predict_proba(projected)
                slide_values.append(np.concatenate([posterior.mean(axis=0), np.asarray([gmm.score(projected)])]))
            repeat_values.append(np.stack(slide_values).mean(axis=0))
        values.append(np.stack(repeat_values).mean(axis=0))
    return np.asarray(values, dtype=np.float32)


def tune_cox_features(train: pd.DataFrame, valid: pd.DataFrame, x_train: np.ndarray, x_valid: np.ndarray) -> tuple[float, float]:
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xv = scaler.transform(x_valid)
    rows = []
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        beta = fit_cox(train.dss_time.to_numpy(float), train.dss_event.to_numpy(int), xt, alpha)
        score = harrell_c(valid.dss_time.to_numpy(float), valid.dss_event.to_numpy(int), xv @ beta)
        rows.append((float(score), -alpha, float(alpha)))
    rows.sort(reverse=True)
    return float(rows[0][2]), float(rows[0][0])


def run_panther_protocol(root: Path, out: Path, state: dict[str, Any], store: SlideStore, patients: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            method = "B5_PANTHER_style_GMM"
            job = f"{method}/seed_{seed}/fold_{fold}"
            pred_path = out / "oof_predictions" / method / f"seed_{seed}_fold_{fold}.csv"
            if job in state["completed_jobs"] and pred_path.exists():
                continue
            mark_start(out, state, job, "panther_style_gmm")
            outer_train = patients.loc[patients.fold != fold].copy().reset_index(drop=True)
            outer_test = patients.loc[patients.fold == fold].copy().reset_index(drop=True)
            inner_train, inner_val, _ = inner_split(patients, fold, seed)
            inner_reservoir = reservoir_patch_matrix(store, inner_train.patient_id.astype(str), stable_seed(seed, fold, "inner"))
            inner_pca = PCA(n_components=64, svd_solver="randomized", random_state=stable_seed(seed, fold, "pca-inner"))
            inner_pca.fit(inner_reservoir)
            best = None
            for k in (4, 8):
                inner_gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3, max_iter=100, n_init=1, random_state=stable_seed(seed, fold, k))
                inner_gmm.fit(inner_pca.transform(inner_reservoir))
                x_it = gmm_patient_features(store, inner_train, inner_pca, inner_gmm, seed)
                x_iv = gmm_patient_features(store, inner_val, inner_pca, inner_gmm, seed + 7919)
                alpha, score = tune_cox_features(inner_train, inner_val, x_it, x_iv)
                candidate = (score, -k, k, alpha)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            assert best is not None
            _, _, selected_k, selected_alpha = best
            outer_reservoir = reservoir_patch_matrix(store, outer_train.patient_id.astype(str), stable_seed(seed, fold, "outer"))
            outer_pca = PCA(n_components=64, svd_solver="randomized", random_state=stable_seed(seed, fold, "pca-outer"))
            outer_pca.fit(outer_reservoir)
            outer_gmm = GaussianMixture(n_components=selected_k, covariance_type="diag", reg_covar=1e-3, max_iter=100, n_init=1, random_state=stable_seed(seed, fold, "outer-gmm"))
            outer_gmm.fit(outer_pca.transform(outer_reservoir))
            x_ot = gmm_patient_features(store, outer_train, outer_pca, outer_gmm, seed)
            # Use the same deterministic evaluation seed for train and test
            # features; the repeat offset is already encoded by repeat in the
            # sampler and changing the patient split seed would create an
            # avoidable protocol mismatch.
            x_te = gmm_patient_features(store, outer_test, outer_pca, outer_gmm, seed)
            scaler = StandardScaler().fit(x_ot)
            beta = fit_cox(outer_train.dss_time.to_numpy(float), outer_train.dss_event.to_numpy(int), scaler.transform(x_ot), selected_alpha)
            risk = scaler.transform(x_te) @ beta
            censor = censoring_km(outer_train.dss_time.to_numpy(float), outer_train.dss_event.to_numpy(int))
            g = censor(outer_test.dss_time.to_numpy(float), left_limit=True)
            pred = outer_test[["patient_id", "fold", "dss_time", "dss_event"]].copy()
            pred["seed"] = seed
            pred["risk_score"] = risk
            pred["censor_survival_outer_train"] = g
            pred["method"] = method
            pred["token_budget"] = MAIN_TOKENS
            pred["projection"] = "train_only_PCA64"
            p15.atomic_csv(pred, pred_path)
            metric = {
                "method": method,
                "seed": seed,
                "fold": fold,
                "token_budget": MAIN_TOKENS,
                "feature_space": "train_only_PCA64_from_raw_1536D",
                "prototype_count": int(selected_k),
                "selected_alpha": float(selected_alpha),
                "inner_val_harrell_c": float(best[0]),
                "n_test": len(outer_test),
                "n_events": int(outer_test.dss_event.sum()),
                "test_harrell_c": float(harrell_c(outer_test.dss_time, outer_test.dss_event, risk)),
                "test_uno_c": float(uno_c(outer_test.dss_time, outer_test.dss_event, risk, g)),
                "trainable_params": 0,
                "training_time_seconds": 0.0,
                "peak_gpu_memory_gb": 0.0,
                "inference_time_seconds": 0.0,
                "provenance": "PANTHER_style_GMM_not_official_PANTHER",
            }
            p15.atomic_json(out / "tables" / "job_metrics" / method / f"seed_{seed}_fold_{fold}.json", metric)
            rows.append(metric)
            mark_complete(out, state, job)
            store.clear_cache()
    return rows


def attempt_mamba_dependencies(out: Path) -> tuple[bool, dict[str, Any]]:
    """Try the official dependency routes before recording a Mamba failure."""
    log_dir = out / "failures" / "mamba_attempts"
    log_dir.mkdir(parents=True, exist_ok=True)
    cached_path = out / "failures" / "mamba_dependency_attempts.json"
    if cached_path.exists():
        # Dependency installation is an audit step, not a per-resume
        # experiment.  Reusing the recorded outcome avoids launching fresh
        # pip/conda subprocesses (and network retries) every time a long run
        # is resumed.  A completed audit remains the source of truth unless
        # the file is explicitly removed for a new environment check.
        payload = json.loads(cached_path.read_text(encoding="utf-8"))
        return bool(payload.get("available", False)), payload
    available, reason = mamba_available()
    attempts: list[dict[str, Any]] = [{"route": "import_preflight", "command": f"{sys.executable} -c import mamba_ssm, causal_conv1d", "returncode": 0 if available else 1, "output": reason}]
    commands = [
        ("official_pip_compatible_stack", [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "causal-conv1d", "mamba-ssm", "--no-build-isolation"]),
        ("conda_or_prebuilt_route", ["conda", "install", "-y", "-c", "conda-forge", "mamba-ssm", "causal-conv1d"]),
        ("official_source_install", [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "git+https://github.com/state-spaces/mamba.git"]),
    ]
    if not available:
        for name, command in commands:
            started = time.perf_counter()
            try:
                result = subprocess.run(command, cwd=out.parent.parent, capture_output=True, text=True, timeout=180, check=False)
                output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
                attempts.append({"route": name, "command": " ".join(command), "returncode": int(result.returncode), "elapsed_seconds": time.perf_counter() - started, "output": output[-12000:]})
            except Exception as exc:
                attempts.append({"route": name, "command": " ".join(command), "returncode": None, "elapsed_seconds": time.perf_counter() - started, "output": f"{type(exc).__name__}: {exc}"})
            available, reason = mamba_available()
            if available:
                attempts.append({"route": "post_install_import", "command": f"{sys.executable} -c import mamba_ssm, causal_conv1d", "returncode": 0, "output": reason})
                break
    payload = {"available": bool(available), "reason": reason, "attempt_count": len(attempts), "attempts": attempts, "checked_at": utc_now()}
    p15.atomic_json(out / "failures" / "mamba_dependency_attempts.json", payload)
    return bool(available), payload


def write_initial_protocol(out: Path, patients: pd.DataFrame, split_path: Path, anchor: pd.DataFrame, mamba_payload: dict[str, Any]) -> None:
    config = {
        "experiment_id": EXPERIMENT_ID,
        "cohort": {"patients": len(patients), "events": int(patients.dss_event.sum()), "folds": 5},
        "split_source": str(split_path),
        "split_hash": p15.canonical_split_hash(patients),
        "raw_feature_dim": INPUT_DIM,
        "main_token_budget": MAIN_TOKENS,
        "token_grid": list(TOKEN_GRID),
        "eval_sampling_repeats": EVAL_REPEATS,
        "sampling": "if N<=K all patches; otherwise sorted stochastic choice without replacement; deterministic five-repeat evaluation",
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "lr_grid": list(LR_GRID),
        "rank_lambda_grid": list(RANK_LAMBDA_GRID),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "patient_batch": PATIENT_BATCH,
        "slide_batch": SLIDE_BATCH,
        "batch_change_note": "compute-only batching; raw feature and token information budget unchanged",
        "bootstrap_replicates": BOOTSTRAPS,
        "uno_censoring": "outer-train censoring KM per fold; no test KM",
        "anchor_gate": anchor.to_dict("records"),
        "mamba_dependency": mamba_payload,
    }
    p15.atomic_json(out / "configs" / "resolved_config.json", config)
    registry = """# Phase 1.5R pre-registered registry\n# All trainable models consume raw 1536D UNI2-h features and K=1024 in the main protocol.\nB0_full:\n  feature_dim: 1536\n  token_budget: full\n  sampling: all_patches\n  transform: train_only_standard_scaler_plus_PCA32_or_64\nB0_1024:\n  feature_dim: 1536\n  token_budget: 1024\n  sampling: deterministic_five_repeat_without_replacement\nB1_GatedABMIL:\n  feature_dim: 1536\n  learned_projection: 256\n  gated_attention: true\n  token_budget: 1024\n  train_sampling: stochastic_without_replacement\n  eval_sampling_repeats: 5\n  loss: full_patient_level_cox\nB2_PlainABMIL:\n  feature_dim: 1536\n  learned_projection: 256\n  gated_attention: false\n  token_budget: 1024\n  loss: full_patient_level_cox\nB3_TransformerMIL_reimplementation:\n  feature_dim: 1536\n  learned_projection: 256\n  token_budget: 1024\n  official_status: generic_reimplementation_not_official_TransMIL\nB4_MambaMIL:\n  feature_dim: 1536\n  learned_projection: 256\n  token_budget: 1024\n  official_status: dependency_gate_recorded\nB5_PANTHER_style_GMM:\n  feature_dim: 1536\n  feature_space: train_only_PCA64\n  token_budget: 1024\n  prototype_fit: outer_train_only\n  official_status: PANTHER_style_not_official_PANTHER\nB6_SpatialNeighborhoodMIL:\n  feature_dim: 1536\n  learned_projection: 256\n  token_budget: 1024\n  neighborhood: coordinate_only_local_density\nB7a_GatedABMIL_Cox:\n  alias_of: B1_GatedABMIL\nB7b_GatedABMIL_CoxRank:\n  alias_base: B1_GatedABMIL\n  rank_pairs: event_i_and_Ti_less_than_Tj_only\nsensitivity_token_budget:\n  values: [128, 256, 512, 1024]\nsensitivity_projection:\n  values: [random64, learned64, learned256, learned512]\n"""
    p15.atomic_text(out / "configs" / "hyperparameter_registry.yaml", registry)
    p15.atomic_json(out / "configs" / "protocol.json", {"raw_feature_dim": INPUT_DIM, "main_tokens": MAIN_TOKENS, "token_grid": list(TOKEN_GRID), "eval_sampling_repeats": EVAL_REPEATS, "seeds": list(SEEDS), "folds": list(FOLDS), "bootstrap": BOOTSTRAPS, "sampling": "sorted choice without replacement", "uno": "outer-train KM"})


def alias_b1_cox(out: Path, state: dict[str, Any], patients: pd.DataFrame) -> None:
    """B7a is the pre-registered Cox-only comparator using identical B1 OOF."""
    for seed in SEEDS:
        for fold in FOLDS:
            source = out / "oof_predictions" / "B1_GatedABMIL" / f"seed_{seed}_fold_{fold}_K{MAIN_TOKENS}_learned256.csv"
            target = out / "oof_predictions" / "B7a_GatedABMIL_Cox" / f"seed_{seed}_fold_{fold}.csv"
            job = f"B7a_GatedABMIL_Cox/seed_{seed}/fold_{fold}"
            if not source.exists() or (job in state["completed_jobs"] and target.exists()):
                continue
            frame = pd.read_csv(source)
            frame["method"] = "B7a_GatedABMIL_Cox"
            p15.atomic_csv(frame, target)
            p15.atomic_json(
                out / "tables" / "job_metrics" / "B7a_GatedABMIL_Cox" / f"seed_{seed}_fold_{fold}.json",
                {
                    "method": "B7a_GatedABMIL_Cox",
                    "seed": seed,
                    "fold": fold,
                    "base_method": "B1_GatedABMIL",
                    "provenance": "exact_OOF_alias_of_B1_GatedABMIL_Cox_only",
                },
            )
            mark_complete(out, state, job)


def run_neural_suite(root: Path, out: Path, state: dict[str, Any], store: SlideStore, patients: pd.DataFrame, device: torch.device, mamba_ok: bool) -> None:
    jobs = [
        ("B1_GatedABMIL", "learned256", False, MAIN_TOKENS, "B1_GatedABMIL"),
        ("B2_PlainABMIL", "learned256", False, MAIN_TOKENS, "B2_PlainABMIL"),
        ("B3_TransformerMIL_reimplementation", "learned256", False, MAIN_TOKENS, "B3_TransformerMIL_reimplementation"),
        ("B6_SpatialNeighborhoodMIL", "learned256", False, MAIN_TOKENS, "B6_SpatialNeighborhoodMIL"),
        ("B7b_GatedABMIL_CoxRank", "learned256", True, MAIN_TOKENS, "B7b_GatedABMIL_CoxRank"),
    ]
    if mamba_ok:
        jobs.append(("B4_MambaMIL", "learned256", False, MAIN_TOKENS, "B4_MambaMIL"))
    for method, projection, rank, budget, output_method in jobs:
        for seed in SEEDS:
            for fold in FOLDS:
                neural_job(root, out, state, store, patients, method, seed, fold, device, budget, projection, rank, output_method)
    alias_b1_cox(out, state, patients)


def pooled_method_frame(out: Path, method: str) -> pd.DataFrame:
    directory = out / "oof_predictions" / method
    frames = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = pd.read_csv(path)
        if {"patient_id", "risk_score", "dss_time", "dss_event", "censor_survival_outer_train"}.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pooled_risk_frame(out: Path, method: str) -> pd.DataFrame:
    frame = pooled_method_frame(out, method)
    if frame.empty:
        return frame
    grouped = frame.groupby("patient_id", as_index=False).agg(
        dss_time=("dss_time", "first"),
        dss_event=("dss_event", "first"),
        risk_score=("risk_score", "mean"),
        censor_survival_outer_train=("censor_survival_outer_train", "first"),
    )
    return grouped.sort_values("patient_id").reset_index(drop=True)


def summarize_methods(out: Path, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    main_rows = []
    fold_rows = []
    pooled: dict[str, pd.DataFrame] = {}
    for method in methods:
        frame = pooled_method_frame(out, method)
        if frame.empty:
            continue
        pooled_frame = pooled_risk_frame(out, method)
        pooled[method] = pooled_frame
        pooled_h = harrell_c(pooled_frame.dss_time, pooled_frame.dss_event, pooled_frame.risk_score)
        pooled_u = uno_c(pooled_frame.dss_time, pooled_frame.dss_event, pooled_frame.risk_score, pooled_frame.censor_survival_outer_train)
        seed_h, seed_u = [], []
        for seed, seed_frame in frame.groupby("seed"):
            seed_risk = seed_frame.groupby("patient_id", as_index=False).agg(dss_time=("dss_time", "first"), dss_event=("dss_event", "first"), risk_score=("risk_score", "mean"), censor_survival_outer_train=("censor_survival_outer_train", "first"))
            seed_h.append(harrell_c(seed_risk.dss_time, seed_risk.dss_event, seed_risk.risk_score))
            seed_u.append(uno_c(seed_risk.dss_time, seed_risk.dss_event, seed_risk.risk_score, seed_risk.censor_survival_outer_train))
        for (seed, fold), fold_frame in frame.groupby(["seed", "fold"]):
            fold_frame = fold_frame.groupby("patient_id", as_index=False).agg(dss_time=("dss_time", "first"), dss_event=("dss_event", "first"), risk_score=("risk_score", "mean"), censor_survival_outer_train=("censor_survival_outer_train", "first"))
            fold_rows.append({"method": method, "seed": int(seed), "fold": int(fold), "n_test": len(fold_frame), "n_events": int(fold_frame.dss_event.sum()), "harrell_c": float(harrell_c(fold_frame.dss_time, fold_frame.dss_event, fold_frame.risk_score)), "uno_c": float(uno_c(fold_frame.dss_time, fold_frame.dss_event, fold_frame.risk_score, fold_frame.censor_survival_outer_train))})
        method_metrics = pd.DataFrame([row for row in fold_rows if row["method"] == method])
        main_rows.append({
            "method": method,
            "n_seeds": int(frame.seed.nunique()),
            "oof_harrell_c": float(pooled_h),
            "oof_uno_c": float(pooled_u),
            "seed_harrell_mean": float(np.mean(seed_h)),
            "seed_harrell_std": float(np.std(seed_h, ddof=1)) if len(seed_h) > 1 else 0.0,
            "seed_uno_mean": float(np.mean(seed_u)),
            "seed_uno_std": float(np.std(seed_u, ddof=1)) if len(seed_u) > 1 else 0.0,
            "fold_harrell_mean": float(method_metrics.harrell_c.mean()),
            "fold_harrell_std": float(method_metrics.harrell_c.std(ddof=1)) if len(method_metrics) > 1 else 0.0,
            "fold_uno_mean": float(method_metrics.uno_c.mean()),
            "fold_uno_std": float(method_metrics.uno_c.std(ddof=1)) if len(method_metrics) > 1 else 0.0,
            "worst_fold_harrell": float(method_metrics.harrell_c.min()),
        })
    main = pd.DataFrame(main_rows)
    folds = pd.DataFrame(fold_rows)
    if not main.empty:
        b1 = float(main.loc[main.method == "B1_GatedABMIL", "oof_harrell_c"].iloc[0]) if (main.method == "B1_GatedABMIL").any() else np.nan
        mean = float(main.loc[main.method == "B0_full", "oof_harrell_c"].iloc[0]) if (main.method == "B0_full").any() else np.nan
        main["delta_harrell_vs_B1"] = main.oof_harrell_c - b1
        main["delta_harrell_vs_B0_full"] = main.oof_harrell_c - mean
    p15.atomic_csv(main, out / "tables" / "main_metrics.csv")
    p15.atomic_csv(folds, out / "tables" / "fold_metrics.csv")
    return main, folds, pooled


def paired_bootstrap(base: pd.DataFrame, other: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    merged = base[["patient_id", "dss_time", "dss_event", "risk_score", "censor_survival_outer_train"]].merge(other[["patient_id", "risk_score"]], on="patient_id", suffixes=("_base", "_other"), validate="one_to_one")
    times = merged.dss_time.to_numpy(float)
    events = merged.dss_event.to_numpy(int)
    g = merged.censor_survival_outer_train.to_numpy(float)
    base_risk = merged.risk_score_base.to_numpy(float)
    other_risk = merged.risk_score_other.to_numpy(float)
    observed_h = harrell_c(times, events, other_risk) - harrell_c(times, events, base_risk)
    observed_u = uno_c(times, events, other_risk, g) - uno_c(times, events, base_risk, g)
    pair_i, pair_j = np.where((events[:, None] == 1) & (times[:, None] < times[None, :]))
    if len(pair_i) == 0:
        raise ValueError("no comparable pairs for bootstrap")
    other_h = (other_risk[pair_i] > other_risk[pair_j]).astype(float) + 0.5 * (other_risk[pair_i] == other_risk[pair_j])
    base_h = (base_risk[pair_i] > base_risk[pair_j]).astype(float) + 0.5 * (base_risk[pair_i] == base_risk[pair_j])
    weights = 1.0 / np.maximum(g[pair_i], 1e-8) ** 2
    rng = np.random.default_rng(seed)
    h_values, u_values = [], []
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        multiplicity = np.zeros((count, len(times)), dtype=np.float64)
        draws = rng.integers(0, len(times), size=(count, len(times)))
        for row_idx in range(count):
            multiplicity[row_idx] = np.bincount(draws[row_idx], minlength=len(times))
        pair_mult = multiplicity[:, pair_i] * multiplicity[:, pair_j]
        denominator = np.maximum(pair_mult.sum(axis=1), 1e-12)
        h_values.extend(((pair_mult @ (other_h - base_h)) / denominator).tolist())
        u_den = np.maximum(pair_mult @ weights, 1e-12)
        u_values.extend(((pair_mult @ ((other_h - base_h) * weights)) / u_den).tolist())
    delta_h = np.asarray(h_values[:replicates], dtype=float)
    delta_u = np.asarray(u_values[:replicates], dtype=float)
    return {
        "n_patients": len(merged),
        "replicates": replicates,
        "delta_harrell": float(observed_h),
        "delta_harrell_ci_low": float(np.quantile(delta_h, 0.025)),
        "delta_harrell_ci_high": float(np.quantile(delta_h, 0.975)),
        "p_two_sided_harrell": float(min(1.0, 2 * min(np.mean(delta_h <= 0), np.mean(delta_h >= 0)))),
        "prob_improvement_harrell": float(np.mean(delta_h > 0)),
        "delta_uno": float(observed_u),
        "delta_uno_ci_low": float(np.quantile(delta_u, 0.025)),
        "delta_uno_ci_high": float(np.quantile(delta_u, 0.975)),
        "p_two_sided_uno": float(min(1.0, 2 * min(np.mean(delta_u <= 0), np.mean(delta_u >= 0)))),
        "prob_improvement_uno": float(np.mean(delta_u > 0)),
        "bootstrap_censoring": "outer-train KM values held fixed per patient/fold",
    }


def run_bootstrap_suite(out: Path, main: pd.DataFrame, pooled: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if main.empty or "B0_full" not in pooled or "B1_GatedABMIL" not in pooled:
        return pd.DataFrame()
    comparisons = []
    baselines = ["B0_full", "B0_1024", "B1_GatedABMIL"]
    targets = [method for method in main.method if method not in {"B0_full", "B0_1024"}]
    for target in targets:
        for baseline in baselines:
            if baseline not in pooled or target == baseline:
                continue
            result = paired_bootstrap(pooled[baseline], pooled[target], BOOTSTRAPS, stable_seed("bootstrap", baseline, target))
            comparisons.append({"method": target, "baseline": baseline, **result})
    strongest = main.loc[~main.method.isin(["B0_full", "B0_1024"])]
    if not strongest.empty:
        target = str(strongest.sort_values("oof_harrell_c", ascending=False).iloc[0].method)
        if target in pooled:
            result = paired_bootstrap(pooled["B0_full"], pooled[target], BOOTSTRAPS, stable_seed("bootstrap", "strongest", target))
            comparisons.append({"method": target, "baseline": "B0_full_strongest_target", **result})
    frame = pd.DataFrame(comparisons)
    p15.atomic_csv(frame, out / "bootstrap" / "paired_bootstrap.csv")
    return frame


def run_token_sensitivity(root: Path, out: Path, state: dict[str, Any], store: SlideStore, patients: pd.DataFrame, device: torch.device) -> None:
    for budget in TOKEN_GRID:
        run_mean_protocol(root, out, state, store, patients, budget, f"Mean@{budget}")
        if budget == MAIN_TOKENS:
            continue
        for method, output_method in (("B1_GatedABMIL", f"Token_B1_K{budget}"), ("B3_TransformerMIL_reimplementation", f"Token_Transformer_K{budget}")):
            for seed in SEEDS:
                for fold in FOLDS:
                    neural_job(root, out, state, store, patients, method, seed, fold, device, budget, "learned256", False, output_method)


def run_projection_sensitivity(root: Path, out: Path, state: dict[str, Any], store: SlideStore, patients: pd.DataFrame, device: torch.device) -> None:
    for projection in PROJECTION_GRID:
        if projection == "learned256":
            continue  # The repaired B1 mainline is the learned-256 row.
        output_method = f"Projection_{projection}"
        for seed in SEEDS:
            for fold in FOLDS:
                neural_job(root, out, state, store, patients, "B1_GatedABMIL", seed, fold, device, MAIN_TOKENS, projection, False, output_method)


def write_failure_ledger(out: Path, mamba_payload: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "method": "CLAM_survival",
            "status": "FAILED_TO_REPRODUCE",
            "attempts": 2,
            "reason": "Official CLAM repository and a separate survival-adaptation search were checked; the available official training path is classification-focused and no validated survival checkpoint/adapter is present. Generic gated attention plus clustering is not reported as CLAM.",
            "references": "https://github.com/mahmoodlab/CLAM; survival-adaptation search recorded in this run",
        },
        {
            "method": "B4_MambaMIL",
            "status": "RUNNABLE" if mamba_payload.get("available") else "FAILED_TO_REPRODUCE",
            "attempts": int(mamba_payload.get("attempt_count", 0)),
            "reason": "Official dependency preflight/install routes are recorded in failures/mamba_dependency_attempts.json; no result is promoted unless mamba_ssm and causal-conv1d import successfully." if not mamba_payload.get("available") else "Official mamba_ssm and causal-conv1d imports succeeded; benchmark was run with the recorded environment.",
            "references": "https://github.com/isyangshu/MambaMIL",
        },
        {
            "method": "TransMIL",
            "status": "RENAMED_REIMPLEMENTATION",
            "attempts": 1,
            "reason": "The local method is named TransformerMIL_reimplementation because the official TransMIL structure/checkpoint was not vendored; no generic Transformer result is labeled official TransMIL.",
            "references": "https://github.com/szc19990412/TransMIL",
        },
        {
            "method": "EAGLE",
            "status": "RENAMED_REIMPLEMENTATION",
            "attempts": 1,
            "reason": "SpatialNeighborhoodMIL is a paper-faithful coordinate-only neighbourhood equivalent; official EAGLE/MASE implementation was not available in the current assets.",
            "references": "paper-faithful equivalent; official implementation unavailable",
        },
        {
            "method": "PANTHER",
            "status": "RENAMED_REIMPLEMENTATION",
            "attempts": 1,
            "reason": "PANTHER_style_GMM uses train-only PCA64 and GMM prototypes from raw 1536D features; it is not called official PANTHER.",
            "references": "https://github.com/mahmoodlab/PANTHER",
        },
    ]
    frame = pd.DataFrame(rows)
    p15.atomic_csv(frame, out / "failures" / "failure_ledger.csv")
    text = "# Phase 1.5R failure log\n\n"
    for row in rows:
        text += f"## {row['method']}\n\n- status: `{row['status']}`\n- attempts: {row['attempts']}\n- reason: {row['reason']}\n- references: {row['references']}\n\n"
    p15.atomic_text(out / "PHASE1_5R_FAILURE_LOG.md", text.rstrip() + "\n")
    return frame


def remeasure_compute_profile(out: Path, main_methods: list[str], device: torch.device) -> pd.DataFrame:
    rows = []
    for method in main_methods:
        metric_files = list((out / "tables" / "job_metrics" / method).glob("*.json"))
        metrics = []
        for path in metric_files:
            try:
                metrics.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        if metrics:
            times = [float(row.get("training_time_seconds", np.nan)) for row in metrics]
            finite_times = [value for value in times if np.isfinite(value)]
            train_minutes = float(np.mean(finite_times) / 60.0) if finite_times else 0.0
            params = int(max(int(row.get("trainable_params", 0)) for row in metrics))
            budget = metrics[0].get("token_budget", 1024 if method == "B0_1024" else "full")
            projection = metrics[0].get("projection", "none")
        else:
            train_minutes, params, budget, projection = 0.0, 0, "full", "none"
        inference_seconds = np.nan
        peak = np.nan
        if method in {"B1_GatedABMIL", "B2_PlainABMIL", "B3_TransformerMIL_reimplementation", "B4_MambaMIL", "B6_SpatialNeighborhoodMIL", "B7b_GatedABMIL_CoxRank"}:
            candidate = sorted((out / "checkpoints" / method).rglob("*.pt"))
            if candidate:
                ckpt = candidate[0]
                try:
                    bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
                    model, preprocessor = build_model(str(bundle["method"]), str(bundle["projection"]))
                    model.load_state_dict(bundle["model_state"])
                    model = model.to(device).eval()
                    fold = int(bundle["fold"])
                    seed = int(bundle["seed"])
                    root = out.parents[1]
                    patients, _, _, _ = p15.load_source_frames(root)
                    test = patients.loc[patients.fold == fold].copy().reset_index(drop=True)
                    store = SlideStore(root, p15.load_source_frames(root)[1])
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    predict_risk(model, store, test, int(bundle["token_budget"]), seed, device, 1, preprocessor)
                    inference_seconds = time.perf_counter() - started
                    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                except Exception:
                    inference_seconds, peak = np.nan, np.nan
        rows.append({"method": method, "feature_dim": INPUT_DIM if projection != "random64" else 64, "projection": projection, "token_budget": budget, "train_time_minutes": train_minutes, "inference_time_seconds": inference_seconds, "peak_gpu_memory_gb": peak, "trainable_params": params, "measurement_status": "recorded_from_job_metrics_and_saved_checkpoint" if np.isfinite(inference_seconds) else "training_recorded_inference_unavailable"})
    frame = pd.DataFrame(rows)
    p15.atomic_csv(frame, out / "tables" / "compute_profile.csv")
    return frame


def calculate_sensitivity_tables(out: Path, pooled: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_rows = []
    for method, frame in pooled.items():
        if method.startswith("Mean@"):
            token_rows.append({"method": "Mean", "token_budget": int(method.split("@")[1]), "oof_harrell_c": float(harrell_c(frame.dss_time, frame.dss_event, frame.risk_score)), "oof_uno_c": float(uno_c(frame.dss_time, frame.dss_event, frame.risk_score, frame.censor_survival_outer_train)), "n_seeds": int(pooled_method_seed_count(out, method))})
        elif method.startswith("Token_B1_K"):
            token_rows.append({"method": "B1_GatedABMIL", "token_budget": int(method.split("K")[1]), "oof_harrell_c": float(harrell_c(frame.dss_time, frame.dss_event, frame.risk_score)), "oof_uno_c": float(uno_c(frame.dss_time, frame.dss_event, frame.risk_score, frame.censor_survival_outer_train)), "n_seeds": int(pooled_method_seed_count(out, method))})
        elif method.startswith("Token_Transformer_K"):
            token_rows.append({"method": "TransformerMIL_reimplementation", "token_budget": int(method.split("K")[1]), "oof_harrell_c": float(harrell_c(frame.dss_time, frame.dss_event, frame.risk_score)), "oof_uno_c": float(uno_c(frame.dss_time, frame.dss_event, frame.risk_score, frame.censor_survival_outer_train)), "n_seeds": int(pooled_method_seed_count(out, method))})
    # K=1024 is the repaired main-budget result, so include it explicitly in
    # the sensitivity table even though the expensive sensitivity runner does
    # not duplicate the main B1/Transformer jobs.
    for method, label in (("B1_GatedABMIL", "B1_GatedABMIL"), ("B3_TransformerMIL_reimplementation", "TransformerMIL_reimplementation")):
        if method in pooled:
            frame = pooled[method]
            token_rows.append({"method": label, "token_budget": MAIN_TOKENS, "oof_harrell_c": float(harrell_c(frame.dss_time, frame.dss_event, frame.risk_score)), "oof_uno_c": float(uno_c(frame.dss_time, frame.dss_event, frame.risk_score, frame.censor_survival_outer_train)), "n_seeds": int(pooled_method_seed_count(out, method))})
    token = pd.DataFrame(token_rows)
    projection_rows = []
    for projection, method in (("random64", "Projection_random64"), ("learned64", "Projection_learned64"), ("learned256", "B1_GatedABMIL"), ("learned512", "Projection_learned512")):
        if method not in pooled:
            continue
        frame = pooled[method]
        projection_rows.append({"projection": projection, "feature_dim": 64 if projection != "learned256" and projection != "learned512" else (256 if projection == "learned256" else 512), "token_budget": MAIN_TOKENS, "oof_harrell_c": float(harrell_c(frame.dss_time, frame.dss_event, frame.risk_score)), "oof_uno_c": float(uno_c(frame.dss_time, frame.dss_event, frame.risk_score, frame.censor_survival_outer_train)), "n_seeds": int(pooled_method_seed_count(out, method)), "protocol": "repaired K=1024; P1 random64 is a new K=1024 sensitivity, not the old K=128 report"})
    old_path = out.parents[1] / "reports" / "phase1_5_benchmark_audit" / "main_metrics.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        old_row = old.loc[old.method == "B1_ABMIL"]
        if not old_row.empty:
            projection_rows.append({"projection": "random64_old_Phase1.5", "feature_dim": 64, "token_budget": 128, "oof_harrell_c": float(old_row.oof_harrell_c.iloc[0]), "oof_uno_c": float(old_row.oof_uno_c.iloc[0]), "n_seeds": int(old_row.n_seeds.iloc[0]), "protocol": "historical Phase1.5 fixed random64/K128 comparator"})
    projection = pd.DataFrame(projection_rows)
    p15.atomic_csv(token, out / "sensitivity" / "token_budget_sensitivity.csv")
    p15.atomic_csv(projection, out / "sensitivity" / "projection_sensitivity.csv")
    return token, projection


def pooled_method_seed_count(out: Path, method: str) -> int:
    frame = pooled_method_frame(out, method)
    return int(frame.seed.nunique()) if not frame.empty else 0


def main_method_names(mamba_ok: bool) -> list[str]:
    names = [
        "B0_full",
        "B0_1024",
        "B1_GatedABMIL",
        "B2_PlainABMIL",
        "B3_TransformerMIL_reimplementation",
        "B5_PANTHER_style_GMM",
        "B6_SpatialNeighborhoodMIL",
        "B7a_GatedABMIL_Cox",
        "B7b_GatedABMIL_CoxRank",
    ]
    if mamba_ok:
        names.insert(5, "B4_MambaMIL")
    return names


def make_figures(out: Path, anchor: pd.DataFrame, main: pd.DataFrame, token: pd.DataFrame, projection: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    labels = ["B0 Mean", "B1 Gated ABMIL"]
    x = np.arange(len(labels))
    width = 0.35
    hist = [float(anchor.loc[anchor.method == "B0_mean", "historical_harrell"].iloc[0]), float(anchor.loc[anchor.method == "B1_ABMIL", "historical_harrell"].iloc[0])]
    new = [float(anchor.loc[anchor.method == "B0_mean", "new_harrell"].iloc[0]), float(anchor.loc[anchor.method == "B1_ABMIL", "new_harrell"].iloc[0])]
    plt.bar(x - width / 2, hist, width, label="Phase1R historical")
    plt.bar(x + width / 2, new, width, label="Phase1.5R anchor")
    plt.xticks(x, labels)
    plt.ylabel("Harrell C")
    plt.ylim(0.45, 0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig1_anchor_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    if not token.empty:
        for method, frame in token.groupby("method"):
            frame = frame.sort_values("token_budget")
            plt.plot(frame.token_budget, frame.oof_harrell_c, marker="o", label=f"{method} Harrell")
            plt.plot(frame.token_budget, frame.oof_uno_c, marker="x", linestyle="--", label=f"{method} Uno")
    plt.xlabel("Token budget K")
    plt.ylabel("OOF C-index")
    plt.xticks(list(TOKEN_GRID))
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig2_token_budget_sensitivity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    if not projection.empty:
        frame = projection.loc[~projection.projection.str.contains("old_Phase1.5")].copy()
        pos = np.arange(len(frame))
        plt.bar(pos - 0.18, frame.oof_harrell_c, 0.36, label="Harrell")
        plt.bar(pos + 0.18, frame.oof_uno_c, 0.36, label="Uno")
        plt.xticks(pos, frame.projection, rotation=20)
    plt.ylabel("OOF C-index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig3_projection_sensitivity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    if not main.empty:
        frame = main.sort_values("oof_harrell_c")
        pos = np.arange(len(frame))
        plt.barh(pos - 0.18, frame.oof_harrell_c, 0.36, label="Harrell")
        plt.barh(pos + 0.18, frame.oof_uno_c, 0.36, label="Uno")
        plt.yticks(pos, frame.method, fontsize=8)
    plt.xlabel("OOF C-index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig4_repaired_benchmark.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    if not bootstrap.empty:
        frame = bootstrap.loc[bootstrap.baseline.isin(["B0_full", "B1_GatedABMIL"])].copy()
        frame["label"] = frame.method + " vs " + frame.baseline
        frame = frame.sort_values("delta_harrell")
        pos = np.arange(len(frame))
        plt.errorbar(frame.delta_harrell, pos, xerr=[frame.delta_harrell - frame.delta_harrell_ci_low, frame.delta_harrell_ci_high - frame.delta_harrell], fmt="o")
        plt.axvline(0.0, color="black", linewidth=0.8)
        plt.yticks(pos, frame.label, fontsize=7)
        plt.xlabel("Δ Harrell (target - baseline), 95% bootstrap CI")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig5_bootstrap_delta.png", dpi=180)
    plt.close()


def _metric_value(main: pd.DataFrame, method: str, column: str) -> float:
    row = main.loc[main.method == method, column]
    return float(row.iloc[0]) if not row.empty else float("nan")


def choose_decision(anchor: pd.DataFrame, main: pd.DataFrame, token: pd.DataFrame, projection: pd.DataFrame, bootstrap: pd.DataFrame) -> tuple[str, str]:
    if not bool(anchor.gate_pass.all()):
        return "STOP_PHASE1_5R", "Phase1R anchor gate failed; no repaired benchmark is admissible."
    full = _metric_value(main, "B0_full", "oof_harrell_c")
    mean_k = _metric_value(main, "B0_1024", "oof_harrell_c")
    b1 = _metric_value(main, "B1_GatedABMIL", "oof_harrell_c")
    advanced_methods = [method for method in main.method if method not in {"B0_full", "B0_1024", "B1_GatedABMIL", "B7a_GatedABMIL_Cox"}]
    full_boot = bootstrap.loc[bootstrap.baseline == "B0_full"] if not bootstrap.empty else pd.DataFrame()
    strongest_stable = False
    if not full_boot.empty:
        strongest = full_boot.sort_values("delta_harrell", ascending=False).iloc[0]
        strongest_stable = bool(strongest.delta_harrell_ci_low > 0 and strongest.delta_uno_ci_low > 0)
    token_rows = token.loc[token.method == "B1_GatedABMIL"].sort_values("token_budget") if not token.empty else pd.DataFrame()
    token_gain = float("nan")
    if len(token_rows) >= 2 and 128 in set(token_rows.token_budget) and 1024 in set(token_rows.token_budget):
        token_gain = float(token_rows.loc[token_rows.token_budget == 1024, "oof_harrell_c"].iloc[0] - token_rows.loc[token_rows.token_budget == 128, "oof_harrell_c"].iloc[0])
    proj_gain = float("nan")
    if not projection.empty and {"random64", "learned256"}.issubset(set(projection.projection)):
        proj_gain = float(projection.loc[projection.projection == "learned256", "oof_harrell_c"].iloc[0] - projection.loc[projection.projection == "random64", "oof_harrell_c"].iloc[0])
    rank_row = bootstrap.loc[(bootstrap.method == "B7b_GatedABMIL_CoxRank") & (bootstrap.baseline == "B1_GatedABMIL")] if not bootstrap.empty else pd.DataFrame()
    rank_stable = bool(not rank_row.empty and rank_row.delta_harrell_ci_low.iloc[0] > 0 and rank_row.delta_uno_ci_low.iloc[0] > 0)
    if advanced_methods and np.isfinite(full) and all(_metric_value(main, method, "oof_harrell_c") <= full + 1e-12 for method in advanced_methods) and np.isfinite(mean_k) and mean_k >= b1 and not strongest_stable:
        return "Decision A — SIMPLE_POOLING_SUFFICIENT", "Anchor passed; Full Mean is not weaker than the repaired advanced methods, Mean@1024 is not weaker than B1, and the strongest-vs-Full-Mean paired bootstrap has no two-metric stable gain."
    if np.isfinite(token_gain) and token_gain > 0.02:
        return "Decision C — TOKEN_BUDGET_BOTTLENECK", f"B1 Harrell changes by {token_gain:.4f} from K=128 to K=1024; Phase1.5's truncation is a material bottleneck candidate."
    if np.isfinite(proj_gain) and proj_gain > 0.02:
        return "Decision D — PROJECTION_BOTTLENECK", f"Learned256 changes Harrell by {proj_gain:.4f} relative to random64 under K=1024; fixed random projection is a material bottleneck candidate."
    if rank_stable:
        return "Decision E — SURVIVAL_SUPERVISION_BOTTLENECK", "Cox+Rank is stably better than the same Gated ABMIL Cox-only comparator; morphology aggregation alone is not the first next intervention."
    if strongest_stable:
        return "Decision B — MIL_AGGREGATION_SUPPORTED", "At least one repaired advanced method has a two-metric bootstrap gain over Full Mean; continue only with the named reproducible method."
    return "Decision E — EVIDENCE_INSUFFICIENT_SURVIVAL_SUPERVISION", "No repaired aggregator has a stable two-metric advantage and no clean pooling gate is satisfied; the limiting evidence is sparse DSS supervision rather than justification for a new architecture."


def write_reports(
    root: Path,
    out: Path,
    patients: pd.DataFrame,
    split_path: Path,
    anchor: pd.DataFrame,
    main: pd.DataFrame,
    folds: pd.DataFrame,
    token: pd.DataFrame,
    projection: pd.DataFrame,
    bootstrap: pd.DataFrame,
    profile: pd.DataFrame,
    failures: pd.DataFrame,
) -> tuple[str, str]:
    label, explanation = choose_decision(anchor, main, token, projection, bootstrap)
    main_lines = ["| Method | Seeds | Harrell | Uno | ΔHarrell vs B1 |", "|---|---:|---:|---:|---:|"]
    b1 = _metric_value(main, "B1_GatedABMIL", "oof_harrell_c")
    for row in main.sort_values("oof_harrell_c").itertuples(index=False):
        main_lines.append(f"| {row.method} | {int(row.n_seeds)} | {row.oof_harrell_c:.4f} | {row.oof_uno_c:.4f} | {row.oof_harrell_c - b1:.4f} |")
    anchor_lines = ["| Method | Historical Harrell | New Harrell | Δ | Historical Uno | New Uno | Gate |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in anchor.itertuples(index=False):
        anchor_lines.append(f"| {row.method} | {row.historical_harrell:.6f} | {row.new_harrell:.6f} | {row.abs_delta_harrell:.6f} | {row.historical_uno:.6f} | {row.new_uno:.6f} | {row.gate_pass} |")
    token_lines = ["| Method | K | Harrell | Uno | Seeds |", "|---|---:|---:|---:|---:|"]
    if not token.empty:
        for row in token.sort_values(["method", "token_budget"]).itertuples(index=False):
            token_lines.append(f"| {row.method} | {int(row.token_budget)} | {row.oof_harrell_c:.4f} | {row.oof_uno_c:.4f} | {int(row.n_seeds)} |")
    projection_lines = ["| Projection | Feature dim | K | Harrell | Uno | Seeds |", "|---|---:|---:|---:|---:|---:|"]
    if not projection.empty:
        for row in projection.itertuples(index=False):
            projection_lines.append(f"| {row.projection} | {int(row.feature_dim)} | {int(row.token_budget)} | {row.oof_harrell_c:.4f} | {row.oof_uno_c:.4f} | {int(row.n_seeds)} |")
    result_text = f"""# Phase 1.5R confirmatory benchmark repair results

## Decision boundary

**{label}**

{explanation}

This report follows the attached Phase1.5R protocol. It is a pathology-only, same-cohort confirmation on frozen UNI2-h features and the frozen Phase1R patient split; it does not start Phase2, tri-modal modeling, or morphology-level credit assignment.

## 1. Phase1R anchor gate

{chr(10).join(anchor_lines)}

Both anchors are re-read from the original Phase1R implementation/runtime OOF and agree with the historical report within ±0.02. The gate was completed before repaired baseline fitting.

## 2. Repaired input protocol

- Cohort: {len(patients)} patients, {int(patients.dss_event.sum())} DSS events, 494 slides, raw feature dimension 1536.
- Split: frozen Phase1R patient-level five-fold manifest `{split_path.name}`; no re-splitting.
- Main neural protocol: learned projection from 1536D, K=1024 stochastic train sampling, five deterministic evaluation repeats, slide-first then mean across slides.
- Mean baselines: Full Mean and Mean@1024; PCA/standardization/Cox fitting are train-only.
- Survival loss: exact full patient-level risk sets; no patch-level survival samples.
- Uno: censoring KM is fitted on outer-training patients and evaluated on the held-out fold; test-fold KM is not used.

## 3. Main repaired benchmark

{chr(10).join(main_lines)}

TransformerMIL is explicitly a generic reimplementation and is not labeled official TransMIL. SpatialNeighborhoodMIL is the paper-faithful coordinate-only equivalent and is not labeled EAGLE. B5_PANTHER_style_GMM uses train-only PCA64/GMM and is not labeled official PANTHER.

## 4. Token budget sensitivity

{chr(10).join(token_lines)}

## 5. Projection sensitivity

{chr(10).join(projection_lines)}

## 6. Required answers

1. **Anchor:** both Phase1R anchors pass.
2. **Phase1.5 performance loss:** the repaired tables quantify token and projection effects; no architecture conclusion is drawn from the old unequal budget.
3. **128→1024:** see token table and `sensitivity/token_budget_sensitivity.csv`.
4. **random64→learned256:** see projection table; the historical K=128 random64 row is kept separately.
5. **Repaired ABMIL vs Mean:** compare B1 with B0_full and B0_1024 using 5000 paired bootstrap replicates.
6. **Gated vs plain:** compare B1 and B2 in the main table and paired bootstrap.
7. **TransMIL:** only `TransformerMIL_reimplementation` is reported because official TransMIL was not vendored.
8. **MambaMIL:** dependency attempts and outcome are in `failures/mamba_dependency_attempts.json`.
9. **PANTHER:** the result is explicitly PANTHER-style and uses train-only PCA/GMM.
10. **Spatial:** the method is explicitly SpatialNeighborhoodMIL; no official EAGLE claim is made.
11. **Cox+Rank:** B7b uses only event_i with T_i<T_j comparable pairs.
12. **Bottleneck:** {explanation}
13. **New pathology model:** no new architecture is authorized by this report.
14. **Pan-cancer benchmark:** remains a possible follow-up only after independent protocol review.
15. **Tri-modal:** remains a possible follow-up; this stage does not execute it.

## 7. Integrity and limitations

- Failed/unavailable implementations remain in `PHASE1_5R_FAILURE_LOG.md`; they are not replaced by paper numbers.
- Only outer-training data are used for scaling, PCA, prototype fitting, hyperparameter selection and censoring KM.
- The cohort has 65 events; conflicting Harrell/Uno directions are reported as mixed evidence.
- Official MambaMIL/CLAM assets may remain unavailable; a queued dependency install is not treated as a successful reproduction.
- The long run resumed after a compute-only batch adjustment (patient_batch=32, slide_batch=32; earlier interrupted work used 8). Raw 1536D features, K values, split, seeds, folds and epoch grid were unchanged, but bitwise identity across batch settings is not claimed.
- Training/inference timing and model memory are recorded in `tables/compute_profile.csv` when measurable.

## 8. Data boundary

Raw TCGA files, H5/NPY/NPZ features, checkpoints, patient-level OOF/attention and caches remain local/ignored. GitHub contains only code, protocol, compact aggregate tables, audit summaries and figures.
"""
    p15.atomic_text(out / "PHASE1_5R_RESULTS.md", result_text)
    decision_text = f"""# Phase 1.5R decision

## {label}

{explanation}

## Scope boundary

本阶段只修复信息预算、baseline、Uno(train censoring KM) 和确认性 benchmark；不执行 Phase2、三模态或 morphology-level risk attribution。
"""
    p15.atomic_text(out / "PHASE1_5R_DECISION.md", decision_text)
    repro = {
        "experiment_id": EXPERIMENT_ID,
        "git_revision": p15.repo_git_revision(root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "raw_feature_dim": INPUT_DIM,
        "main_token_budget": MAIN_TOKENS,
        "token_grid": list(TOKEN_GRID),
        "projection_grid": list(PROJECTION_GRID),
        "eval_sampling_repeats": EVAL_REPEATS,
        "patient_batch": PATIENT_BATCH,
        "slide_batch": SLIDE_BATCH,
        "batch_change_note": "compute-only adjustment from 8 to 32 after an interrupted resume; raw feature/token budgets and split/seeds/folds unchanged; bitwise identity across batch settings not claimed",
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "bootstrap_replicates": BOOTSTRAPS,
        "split_hash": p15.canonical_split_hash(patients),
        "uno_censoring": "outer-train censoring KM per fold",
        "cox": "full patient-level risk sets with T_j >= T_i",
        "rank_pairs": "event i and T_i < T_j only",
        "attention_status": "aggregate audit only; patient-level attention remains local",
    }
    p15.atomic_json(out / "PHASE1_5R_REPRODUCIBILITY.json", repro)
    p15.atomic_text(out / "PHASE1_5R_REPRODUCIBILITY.md", "# Phase 1.5R reproducibility\n\nThe JSON beside this file records the frozen split hash, raw 1536D input, K=1024 main budget, token/projection sensitivity grids, five-repeat evaluation, outer-train censoring KM, full patient-level Cox risk sets, seeds, folds and 5000 paired bootstrap replicates.\n")
    p15.atomic_json(out / "PHASE1_5R_MANIFEST.json", {"experiment_id": EXPERIMENT_ID, "created_at": utc_now(), "git_revision": p15.repo_git_revision(root), "decision": label, "files": sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file() and path.stat().st_size < 5_000_000), "failed_methods": failures.to_dict("records")})

    report_root = root / "reports" / EXPERIMENT_ID
    report_root.mkdir(parents=True, exist_ok=True)
    for name in ["PHASE1_5R_RESULTS.md", "PHASE1_5R_DECISION.md", "PHASE1_5R_REPRODUCIBILITY.md", "PHASE1_5R_REPRODUCIBILITY.json", "PHASE1_5R_FAILURE_LOG.md"]:
        source = out / name
        if source.exists():
            shutil.copy2(source, report_root / name)
    for name, source in [
        ("anchor_reproduction.csv", out / "anchor" / "anchor_reproduction.csv"),
        ("main_metrics.csv", out / "tables" / "main_metrics.csv"),
        ("fold_metrics.csv", out / "tables" / "fold_metrics.csv"),
        ("token_budget_sensitivity.csv", out / "sensitivity" / "token_budget_sensitivity.csv"),
        ("projection_sensitivity.csv", out / "sensitivity" / "projection_sensitivity.csv"),
        ("paired_bootstrap.csv", out / "bootstrap" / "paired_bootstrap.csv"),
        ("compute_profile.csv", out / "tables" / "compute_profile.csv"),
        ("failure_ledger.csv", out / "failures" / "failure_ledger.csv"),
        ("protocol.json", out / "configs" / "protocol.json"),
        ("resolved_config.json", out / "configs" / "resolved_config.json"),
        ("hyperparameter_registry.yaml", out / "configs" / "hyperparameter_registry.yaml"),
        ("split_manifest.json", out / "splits" / "split_manifest.json"),
    ]:
        if source.exists():
            shutil.copy2(source, report_root / name)
    figure_root = report_root / "figures"
    figure_root.mkdir(exist_ok=True)
    for path in (out / "figures").glob("*.png"):
        shutil.copy2(path, figure_root / path.name)
    return label, explanation


def _run_unlocked(root: Path, resume: bool = False, force_audit: bool = False, anchor_only: bool = False) -> None:
    out = output_dir(root)
    for name in ["anchor", "data_audit", "configs", "logs", "checkpoints", "oof_predictions", "tables", "embeddings", "attention", "figures", "bootstrap", "sensitivity", "failures"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    state = load_state(out, resume)
    patients, slides, split_path, slide_path = load_frozen(root)
    audit = p15.audit_and_freeze(root, out, force=force_audit)
    p15.atomic_json(out / "configs" / "cohort_sources.json", {"split_source": str(split_path), "slide_source": str(slide_path), "data_root_env": "DIFFUSION_DATA_ROOT", "patients": len(patients), "slides": len(slides), "events": int(patients.dss_event.sum()), "split_hash": p15.canonical_split_hash(patients)})
    state["stage"] = "anchor_gate"
    save_state(out, state)
    anchor = run_anchor_gate(root, out, patients)
    if anchor_only:
        state["status"] = "ANCHOR_PASS"
        state["stage"] = "anchor_gate_complete"
        save_state(out, state)
        print(json.dumps({"status": state["status"], "anchor": anchor.to_dict("records"), "split_hash": p15.canonical_split_hash(patients)}, ensure_ascii=False, indent=2), flush=True)
        return
    mamba_ok, mamba_payload = attempt_mamba_dependencies(out)
    write_initial_protocol(out, patients, split_path, anchor, mamba_payload)
    store = SlideStore(root, slides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["stage"] = "mean_baselines"
    save_state(out, state)
    run_mean_protocol(root, out, state, store, patients, None, "B0_full")
    run_mean_protocol(root, out, state, store, patients, MAIN_TOKENS, "B0_1024")
    state["stage"] = "token_sensitivity"
    save_state(out, state)
    run_token_sensitivity(root, out, state, store, patients, device)
    state["stage"] = "projection_sensitivity"
    save_state(out, state)
    run_projection_sensitivity(root, out, state, store, patients, device)
    state["stage"] = "main_neural"
    save_state(out, state)
    run_neural_suite(root, out, state, store, patients, device, mamba_ok)
    state["stage"] = "panther_style_gmm"
    save_state(out, state)
    run_panther_protocol(root, out, state, store, patients)
    methods = main_method_names(mamba_ok)
    state["stage"] = "summaries"
    save_state(out, state)
    main, folds, pooled = summarize_methods(out, methods)
    # The main table intentionally contains only the pre-registered core
    # methods.  Sensitivity jobs (Mean@K, Token_* and Projection_*) are
    # separate output methods, so explicitly add their pooled OOF frames
    # before constructing the sensitivity tables.  Without this bridge the
    # jobs were complete on disk but the aggregate token/projection tables
    # silently contained only the K=1024 core rows.
    sensitivity_methods = [
        *(f"Mean@{budget}" for budget in TOKEN_GRID),
        *(f"Token_B1_K{budget}" for budget in TOKEN_GRID if budget != MAIN_TOKENS),
        *(f"Token_Transformer_K{budget}" for budget in TOKEN_GRID if budget != MAIN_TOKENS),
        *(f"Projection_{projection}" for projection in PROJECTION_GRID if projection != "learned256"),
    ]
    for sensitivity_method in sensitivity_methods:
        if sensitivity_method not in pooled:
            frame = pooled_risk_frame(out, sensitivity_method)
            if not frame.empty:
                pooled[sensitivity_method] = frame
    token, projection = calculate_sensitivity_tables(out, pooled)
    bootstrap = run_bootstrap_suite(out, main, pooled)
    profile = remeasure_compute_profile(out, methods, device)
    failures = write_failure_ledger(out, mamba_payload)
    make_figures(out, anchor, main, token, projection, bootstrap)
    label, explanation = write_reports(root, out, patients, split_path, anchor, main, folds, token, projection, bootstrap, profile, failures)
    state["status"] = "COMPLETE"
    state["stage"] = "complete"
    state["current_job"] = None
    state["end_time"] = utc_now()
    state["final_artifacts"] = {
        "anchor": True,
        "data_audit": True,
        "fixed_split": True,
        "main_metrics": True,
        "fold_metrics": True,
        "token_sensitivity": True,
        "projection_sensitivity": True,
        "bootstrap": True,
        "compute_profile": True,
        "figures": True,
        "reports": True,
        "failure_log": True,
    }
    state["decision"] = label
    state["decision_explanation"] = explanation
    save_state(out, state)
    print(json.dumps({"status": state["status"], "decision": label, "completed_jobs": len(state["completed_jobs"]), "failed_jobs": len(state["failed_jobs"]), "methods": methods}, ensure_ascii=False, indent=2), flush=True)


def run(root: Path, resume: bool = False, force_audit: bool = False, anchor_only: bool = False) -> None:
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    lock = acquire_run_lock(out)
    if lock is None:
        print(json.dumps({"status": "LOCKED", "experiment_id": EXPERIMENT_ID, "lock": str(out / "run.lock")}, ensure_ascii=False), flush=True)
        return
    try:
        _run_unlocked(root, resume=resume, force_audit=force_audit, anchor_only=anchor_only)
    finally:
        release_run_lock(lock)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5R confirmatory benchmark repair")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-audit", action="store_true")
    parser.add_argument("--anchor-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), resume=args.resume, force_audit=args.force_audit, anchor_only=args.anchor_only)


if __name__ == "__main__":
    main()

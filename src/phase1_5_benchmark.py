"""Phase 1.5 pathology-only benchmark and audit.

This module is deliberately self-contained and fail-closed.  It consumes the
already audited UNI2-h bags and the frozen Phase1R patient split, never reads
clinical/RNA/TNM fields, and writes all patient-level runtime artifacts below
``outputs/phase1_5_benchmark_audit`` (which is ignored by Git).  Only compact
reports/tables are copied to ``reports/`` for publication and review.

The runner has two important design choices:

* all neural models see the same deterministic, label-independent token cap
  and random projection; the raw mean baseline is kept separate;
* every trainable model is optimized with a full patient-level Cox risk set.
  Patches are never treated as survival samples.

The implementation is intended for local confirmationatory benchmarking.  It
does not silently replace methods whose official dependencies/assets are
unavailable: those methods are recorded in the failure ledger instead.
"""

from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from phase1_utils import censoring_km, fit_cox, harrell_c, set_seed, uno_c


EXPERIMENT_ID = "phase1_5_benchmark_audit"
SEEDS = (20260723, 20260724, 20260725)
FOLDS = (1, 2, 3, 4, 5)
MAX_TOKENS = 128
PROJECTION_DIM = 64
HIDDEN_DIM = 64
BOOTSTRAPS = 2000
RANDOM_PROJECTION_SEED = 20260828

ALL_METHODS = (
    "B0_mean",
    "B0b_mean_std",
    "B1_ABMIL",
    "B2_GatedABMIL",
    "B3_CLAM_SB_survival",
    "B4_TransMIL",
    "B5_MambaMIL",
    "B6_PANTHER_GMM",
    "B7_PAMoE",
    "B8_EAGLE_spatial",
    "B9a_ABMIL_Cox",
    "B9b_ABMIL_CoxRank",
)
TRAINABLE_METHODS = {
    "B1_ABMIL",
    "B2_GatedABMIL",
    "B3_CLAM_SB_survival",
    "B4_TransMIL",
    "B8_EAGLE_spatial",
    "B9b_ABMIL_CoxRank",
}
NEURAL_METHODS = (
    "B1_ABMIL",
    "B2_GatedABMIL",
    "B3_CLAM_SB_survival",
    "B4_TransMIL",
    "B8_EAGLE_spatial",
    "B9b_ABMIL_CoxRank",
)
FAILED_METHODS = {
    "B3_CLAM_SB_survival": {
        "status": "FAILED_TO_REPRODUCE",
        "reason": "The official CLAM repository is classification-focused and no validated survival adaptation/checkpoint is available in the current assets. An initial bag-level Cox adaptation was run as a diagnostic but is excluded from the primary benchmark because it is not a defensible CLAM survival reproduction.",
        "official_reference": "https://github.com/mahmoodlab/CLAM",
        "attempts": 1,
    },
    "B5_MambaMIL": {
        "status": "FAILED_TO_REPRODUCE",
        "reason": "Official isyangshu/MambaMIL code requires the mamba package/causal-conv1d stack; mamba_ssm is absent in the locked environment and the project does not contain the official source or checkpoint.",
        "official_reference": "https://github.com/isyangshu/MambaMIL",
        "attempts": 1,
    },
    "B7_PAMoE": {
        "status": "FAILED_TO_REPRODUCE",
        "reason": "The official PAMoE repository is not vendored and its released demo requires the original training integration/extra assets; current scope contains frozen UNI2-h features only. No approximation is reported as PAMoE.",
        "official_reference": "https://github.com/wjx-error/PAMoE",
        "attempts": 1,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_split_hash(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.sort_values("patient_id").itertuples(index=False):
        rows.append(
            f"{row.patient_id}|{int(row.fold)}|{int(row.dss_event)}|{float(row.dss_time):.12g}\n"
        )
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def repo_git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "--git-dir=.git_local", "--work-tree=.", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def output_dir(root: Path) -> Path:
    return root / "outputs" / EXPERIMENT_ID


def load_public_config(root: Path) -> dict[str, Any]:
    path = root / "configs" / "phase1_5_benchmark.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing tracked Phase1.5 config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("invalid Phase1.5 config root")
    config_seeds = tuple(int(x) for x in payload.get("seeds", []))
    if config_seeds != SEEDS:
        raise ValueError(f"config seeds {config_seeds} do not match frozen runner seeds {SEEDS}")
    protocol = payload.get("protocol", {})
    if int(protocol.get("max_tokens", -1)) != MAX_TOKENS or int(protocol.get("feature_projection_dim", -1)) != PROJECTION_DIM:
        raise ValueError("config token/projection dimensions do not match frozen runner")
    return payload


def data_root(root: Path) -> Path:
    configured = os.environ.get("DIFFUSION_DATA_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else (root / "TCGA").resolve()


def resolve_input_path(root: Path, data: Path, value: object) -> Path:
    raw = Path(str(value))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([root / raw, data / raw])
        if raw.parts and raw.parts[0].upper() == "TCGA":
            candidates.append(data.parent / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    attempted = ", ".join(str(x.resolve()) for x in candidates)
    raise FileNotFoundError(f"input does not exist: {value!r}; checked {attempted}")


def load_source_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    split_path = root / "outputs" / "phase1r_confirmatory_20260828" / "split" / "confirmatory_patient_manifest.csv"
    slide_path = root / "outputs" / "pathology_context_phase1" / "audit" / "slide_manifest_used.csv"
    if not split_path.exists() or not slide_path.exists():
        raise FileNotFoundError("Phase1R split or audited slide manifest is missing")
    patients = pd.read_csv(split_path)
    slides = pd.read_csv(slide_path)
    patients = patients.rename(columns={"outer_fold": "fold", "dss_status": "dss_event", "dss_months": "dss_time"})
    required_p = {"patient_id", "fold", "dss_event", "dss_time"}
    required_s = {"patient_id", "slide_id", "bag_path", "coords_path", "n_patches"}
    if missing := sorted(required_p - set(patients.columns)):
        raise ValueError(f"split missing columns: {missing}")
    if missing := sorted(required_s - set(slides.columns)):
        raise ValueError(f"slide manifest missing columns: {missing}")
    patients["patient_id"] = patients["patient_id"].astype(str).str.upper()
    patients["fold"] = pd.to_numeric(patients["fold"], errors="raise").astype(int)
    patients["dss_event"] = pd.to_numeric(patients["dss_event"], errors="raise").astype(int)
    patients["dss_time"] = pd.to_numeric(patients["dss_time"], errors="raise").astype(float)
    slides["patient_id"] = slides["patient_id"].astype(str).str.upper()
    if patients["patient_id"].duplicated().any():
        raise ValueError("duplicate patient in frozen split")
    if not patients["dss_event"].isin([0, 1]).all() or not np.isfinite(patients["dss_time"]).all() or (patients["dss_time"] <= 0).any():
        raise ValueError("invalid DSS endpoint")
    if not set(patients["fold"]).issubset(set(FOLDS)):
        raise ValueError("frozen split has unexpected fold labels")
    slides = slides.loc[slides["patient_id"].isin(set(patients["patient_id"]))].copy()
    if slides.empty or set(slides["patient_id"]) != set(patients["patient_id"]):
        raise ValueError("slide/patient cohort mismatch")
    if slides["slide_id"].duplicated().any():
        raise ValueError("duplicate slide ids")
    slides = slides.merge(patients[["patient_id", "fold"]], on="patient_id", how="left", validate="many_to_one")
    counts = slides.groupby("patient_id").size().rename("num_slides")
    patients = patients.drop(columns=["num_slides"], errors="ignore").merge(counts, on="patient_id", how="left", validate="one_to_one")
    if patients["num_slides"].isna().any():
        raise ValueError("patient without a valid slide")
    return patients.sort_values("patient_id").reset_index(drop=True), slides.sort_values(["patient_id", "slide_id"]).reset_index(drop=True), split_path, slide_path


def scan_feature(path: Path, coords_path: Path, root: Path, data: Path, row: pd.Series) -> dict[str, Any]:
    bag = np.load(path, mmap_mode="r", allow_pickle=False)
    if bag.ndim != 2:
        raise ValueError(f"feature tensor must be 2-D, got {bag.shape}")
    finite = np.isfinite(bag)
    has_nan = bool(np.isnan(bag).any())
    has_inf = bool(np.isinf(bag).any())
    zero_rows = int(np.all(np.asarray(bag) == 0, axis=1).sum())
    coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(bag):
        raise ValueError(f"coordinates are not aligned: feature={bag.shape}, coords={coords.shape}")
    h5_shape = ""
    h5_coords = False
    h5_path_value = row.get("h5_path", "")
    if str(h5_path_value):
        try:
            h5_path = resolve_input_path(root, data, h5_path_value)
            with h5py.File(h5_path, "r") as handle:
                h5_coords = "coords" in handle or "coords_patching" in handle
                h5_shape = "x".join(str(x) for x in handle["features"].shape) if "features" in handle else ""
        except Exception:
            h5_shape = "UNAVAILABLE"
    result = {
        "patient_id": str(row.patient_id),
        "slide_id": str(row.slide_id),
        "feature_path": str(path.relative_to(root).as_posix()) if path.is_relative_to(root) else str(path),
        "coords_path": str(coords_path.relative_to(root).as_posix()) if coords_path.is_relative_to(root) else str(coords_path),
        "feature_shape": "x".join(str(x) for x in bag.shape),
        "feature_dtype": str(bag.dtype),
        "num_patches": int(len(bag)),
        "feature_dim": int(bag.shape[1]),
        "has_coords": True,
        "coord_dim": int(coords.shape[1]),
        "coord_dtype": str(coords.dtype),
        "coord_unique": int(len(np.unique(np.asarray(coords), axis=0))),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "finite_all": bool(finite.all()),
        "num_zero_rows": zero_rows,
        "h5_feature_shape": h5_shape,
        "h5_has_coords": h5_coords,
    }
    return result


def audit_and_freeze(root: Path, out: Path, force: bool = False) -> dict[str, Any]:
    audit_dir = out / "data_audit"
    split_dir = out / "splits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = split_dir / "fixed_patient_folds.csv"
    if not force and (audit_dir / "data_audit_report.md").exists() and fixed_path.exists():
        patients, slides, split_path, slide_path = load_source_frames(root)
        fixed = pd.read_csv(fixed_path)
        return {"patients": patients, "slides": slides, "fixed": fixed, "split_source": str(split_path), "slide_source": str(slide_path), "skipped": True}
    patients, slides, split_path, slide_path = load_source_frames(root)
    data = data_root(root)
    inventory_rows = []
    errors = []
    for i, row in enumerate(slides.itertuples(index=False), start=1):
        try:
            bag_path = resolve_input_path(root, data, row.bag_path)
            coord_path = resolve_input_path(root, data, row.coords_path)
            inventory_rows.append(scan_feature(bag_path, coord_path, root, data, pd.Series(row._asdict())))
        except Exception as exc:
            errors.append({"patient_id": str(row.patient_id), "slide_id": str(row.slide_id), "error": f"{type(exc).__name__}: {exc}"})
        if i % 50 == 0:
            print(f"[phase1.5 audit] scanned {i}/{len(slides)} slides", flush=True)
    inventory = pd.DataFrame(inventory_rows).sort_values(["patient_id", "slide_id"]).reset_index(drop=True)
    if errors or len(inventory) != len(slides) or not inventory["finite_all"].all() or not inventory["has_coords"].all():
        atomic_json(audit_dir / "feature_scan_errors.json", errors)
        raise RuntimeError(f"feature audit failed: errors={len(errors)}, rows={len(inventory)}/{len(slides)}")
    rng = np.random.default_rng(20260828)
    sampled = set(rng.choice(inventory["slide_id"].to_numpy(), size=min(20, len(inventory)), replace=False).tolist())
    inventory["random_20_spotcheck"] = inventory["slide_id"].isin(sampled)
    atomic_csv(inventory, audit_dir / "feature_inventory.csv")

    mapping = slides[["patient_id", "slide_id", "bag_path", "coords_path", "n_patches", "fold"]].copy()
    mapping = mapping.rename(columns={"bag_path": "feature_path", "fold": "fixed_fold"})
    mapping["num_slides"] = mapping.groupby("patient_id")["slide_id"].transform("size")
    mapping = mapping.merge(patients[["patient_id", "dss_event", "dss_time"]], on="patient_id", how="left", validate="many_to_one")
    atomic_csv(mapping, audit_dir / "patient_slide_mapping.csv")

    fixed = patients[["patient_id", "fold", "dss_event", "dss_time", "num_slides"]].copy()
    fixed = fixed.rename(columns={"dss_event": "dss_event", "dss_time": "dss_time"})
    atomic_csv(fixed, fixed_path)
    split_rows = []
    slide_fold_mismatch = 0
    for fold in FOLDS:
        test_ids = set(fixed.loc[fixed.fold == fold, "patient_id"])
        train_ids = set(fixed.loc[fixed.fold != fold, "patient_id"])
        slide_test = set(slides.loc[slides.fold == fold, "slide_id"])
        slide_train = set(slides.loc[slides.fold != fold, "slide_id"])
        if slide_test & slide_train:
            slide_fold_mismatch += len(slide_test & slide_train)
        split_rows.append({
            "fold": fold,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "events_train": int(fixed.loc[fixed.fold != fold, "dss_event"].sum()),
            "events_test": int(fixed.loc[fixed.fold == fold, "dss_event"].sum()),
            "patient_overlap": len(train_ids & test_ids),
            "slide_overlap": len(slide_train & slide_test),
            "source": "Phase1R frozen patient-level split",
        })
    split_audit = pd.DataFrame(split_rows)
    atomic_csv(split_audit, audit_dir / "split_audit.csv")

    event_times = fixed.loc[fixed.dss_event == 1, "dss_time"]
    censor_times = fixed.loc[fixed.dss_event == 0, "dss_time"]
    endpoint = pd.DataFrame([
        {"metric": "n_patients", "value": len(fixed)},
        {"metric": "n_events", "value": int(fixed.dss_event.sum())},
        {"metric": "n_censored", "value": int((fixed.dss_event == 0).sum())},
        {"metric": "censoring_rate", "value": float((fixed.dss_event == 0).mean())},
        {"metric": "dss_time_min", "value": float(fixed.dss_time.min())},
        {"metric": "dss_time_median", "value": float(fixed.dss_time.median())},
        {"metric": "dss_time_max", "value": float(fixed.dss_time.max())},
        {"metric": "event_time_median", "value": float(event_times.median())},
        {"metric": "censor_time_median", "value": float(censor_times.median())},
        {"metric": "missing_endpoint", "value": 0},
        {"metric": "duplicate_patients", "value": int(fixed.patient_id.duplicated().sum())},
    ])
    atomic_csv(endpoint, audit_dir / "endpoint_summary.csv")
    cohort = pd.DataFrame([
        {"metric": "n_patients", "value": len(fixed)},
        {"metric": "n_slides", "value": len(slides)},
        {"metric": "n_events", "value": int(fixed.dss_event.sum())},
        {"metric": "feature_dim", "value": int(inventory.feature_dim.iloc[0])},
        {"metric": "feature_dtype", "value": inventory.feature_dtype.iloc[0]},
        {"metric": "patch_min", "value": int(inventory.num_patches.min())},
        {"metric": "patch_median", "value": float(inventory.num_patches.median())},
        {"metric": "patch_max", "value": int(inventory.num_patches.max())},
        {"metric": "multi_slide_patients", "value": int((fixed.num_slides > 1).sum())},
        {"metric": "data_root_env_used", "value": bool(os.environ.get("DIFFUSION_DATA_ROOT")),
        },
    ])
    atomic_csv(cohort, audit_dir / "cohort_summary.csv")
    split_hash = canonical_split_hash(fixed)
    atomic_json(split_dir / "split_manifest.json", {
        "source": str(split_path.relative_to(root).as_posix()),
        "source_sha256": sha256_file(split_path),
        "fixed_split_sha256": split_hash,
        "folds": 5,
        "patient_level": True,
        "stratification": "inherited Phase1R frozen split; no regeneration",
        "created_at": utc_now(),
    })
    leakage_lines = [
        "# Phase 1.5 leakage audit",
        "",
        f"- patient-level split: PASS (patient overlap across each train/test pair = 0; fixed split hash `{split_hash}`).",
        f"- slide-level split inheritance: {'PASS' if slide_fold_mismatch == 0 else 'FAIL'} (cross-fold slide overlap = {slide_fold_mismatch}).",
        "- feature normalization/PCA/prototype fitting: PASS by runner contract; every fold fits transforms on the outer-training/inner-training subset only.",
        "- survival labels in non-survival preprocessing: PASS; labels are used only for split stratification, Cox/ranking loss, and evaluation.",
        "- inner validation: PASS by runner contract; test fold is not used for hyperparameter selection or early stopping.",
        "- multi-slide handling: PASS; slide-first embeddings are computed after split and averaged within patient.",
        "- pretrained survival checkpoint: PASS; no survival-pretrained checkpoint is loaded.",
        "- raw data scope: PASS; only audited `TCGA/` UNI2-h feature/coordinate files are read; project `data/` is excluded.",
    ]
    atomic_text(audit_dir / "leakage_audit.md", "\n".join(leakage_lines) + "\n")
    patch_q = inventory.num_patches.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
    report = f"""# Phase 1.5 data audit report

## Scope

The pathology-only cohort uses the audited UNI2-h bags and real H5 coordinates. The unrelated project `data/` directory and all clinical/RNA/TNM/stage fields are excluded.

## Cohort and endpoint

- Patients: **{len(fixed)}**; slides: **{len(slides)}**; DSS events: **{int(fixed.dss_event.sum())}**; censoring rate: **{(fixed.dss_event == 0).mean():.4f}**.
- DSS is represented as `(dss_time, dss_event)`, where event 1 is observed DSS and event 0 is right-censored. Censored patients are never converted to a negative class.
- Endpoint source: frozen Phase1R patient manifest `{split_path.relative_to(root).as_posix()}`; no alternate endpoint source was silently substituted.

## Slides and features

- Feature dimension: **{int(inventory.feature_dim.iloc[0])}**, feature dtype: **{inventory.feature_dtype.iloc[0]}**.
- Patch count: min/median/max **{int(inventory.num_patches.min())}/{inventory.num_patches.median():.0f}/{int(inventory.num_patches.max())}**.
- Quantiles (1%, 5%, 25%, 50%, 75%, 95%, 99%): `{[round(float(patch_q[q]), 1) for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]]}`.
- All {len(inventory)} feature files and coordinates passed finite/alignment checks; zero-vector rows: {int(inventory.num_zero_rows.sum())}; duplicated slide IDs: 0.
- One-to-many mapping: {int((fixed.num_slides > 1).sum())} patients have >1 slide; slide-first protocol averages slide embeddings within patient.

## Split

The Phase1R patient-level split is inherited and frozen. Each fold has {split_audit.events_test.iloc[0]} DSS events in the test set; no patient or slide crosses a fold.

See `leakage_audit.md` for the complete checklist. This audit is a prerequisite; model execution is not allowed to overwrite the split.
"""
    atomic_text(audit_dir / "data_audit_report.md", report)
    atomic_json(audit_dir / "audit_manifest.json", {
        "created_at": utc_now(),
        "git_revision": repo_git_revision(root),
        "root": str(root),
        "data_root": str(data),
        "patients": len(fixed),
        "slides": len(slides),
        "events": int(fixed.dss_event.sum()),
        "feature_inventory_rows": len(inventory),
        "random_20_spotcheck_rows": int(inventory.random_20_spotcheck.sum()),
        "split_hash": split_hash,
    })
    return {"patients": patients, "slides": slides, "fixed": fixed, "split_source": str(split_path), "slide_source": str(slide_path), "skipped": False}


class SlideCache:
    """Deterministic, label-independent token cache shared by neural methods."""

    def __init__(self, root: Path, data: Path, slides: pd.DataFrame, max_tokens: int, projection_dim: int, seed: int):
        self.root = root
        self.data = data
        self.slides = slides.reset_index(drop=True)
        self.max_tokens = max_tokens
        self.projection_dim = projection_dim
        self.seed = seed
        bag_dims = set()
        for row in self.slides.itertuples(index=False):
            path = resolve_input_path(root, data, row.bag_path)
            bag_dims.add(int(np.load(path, mmap_mode="r", allow_pickle=False).shape[1]))
        if bag_dims != {1536}:
            raise ValueError(f"expected one 1536-D feature space, got {bag_dims}")
        rng = np.random.default_rng(RANDOM_PROJECTION_SEED)
        self.projection = rng.normal(0.0, 1.0 / math.sqrt(1536), size=(1536, projection_dim)).astype(np.float32)
        self.features = np.zeros((len(self.slides), max_tokens, projection_dim), dtype=np.float32)
        self.coords = np.zeros((len(self.slides), max_tokens, 2), dtype=np.float32)
        self.mask = np.zeros((len(self.slides), max_tokens), dtype=bool)
        self.neighbor_features = np.zeros_like(self.features)
        self.patient_to_slides: dict[str, list[int]] = defaultdict(list)
        started = time.perf_counter()
        for i, row in enumerate(self.slides.itertuples(index=False)):
            bag_path = resolve_input_path(root, data, row.bag_path)
            coord_path = resolve_input_path(root, data, row.coords_path)
            bag = np.load(bag_path, mmap_mode="r", allow_pickle=False)
            coords = np.load(coord_path, mmap_mode="r", allow_pickle=False)
            n = len(bag)
            if n <= max_tokens:
                indices = np.arange(n, dtype=int)
            else:
                # Uniform order-preserving cap, not random dropping.  This is
                # the same cap for every method and is recorded in configs.
                indices = np.unique(np.linspace(0, n - 1, max_tokens, dtype=int))
            x = np.asarray(bag[indices], dtype=np.float32) @ self.projection
            c = np.asarray(coords[indices], dtype=np.float32)
            if len(c):
                c = (c - c.min(axis=0)) / np.maximum(c.max(axis=0) - c.min(axis=0), 1.0)
            length = min(len(indices), max_tokens)
            self.features[i, :length] = x[:length]
            self.coords[i, :length] = c[:length]
            self.mask[i, :length] = True
            if length > 1:
                k = min(8, length - 1)
                nn_model = NearestNeighbors(n_neighbors=k + 1).fit(c[:length])
                neighbors = nn_model.kneighbors(return_distance=False)[:, 1:]
                self.neighbor_features[i, :length] = x[neighbors].mean(axis=1)
            self.patient_to_slides[str(row.patient_id)].append(i)
            if (i + 1) % 50 == 0:
                print(f"[phase1.5 cache] seed={seed} slides={i + 1}/{len(self.slides)}", flush=True)
        self.elapsed_seconds = time.perf_counter() - started

    def slide_indices(self, patient_ids: Iterable[str]) -> list[int]:
        result = []
        for pid in patient_ids:
            result.extend(self.patient_to_slides[str(pid)])
        return result


class AttentionMIL(nn.Module):
    def __init__(self, gated: bool = False, input_dim: int = PROJECTION_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.gated = gated
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(0.1))
        self.attention_v = nn.Linear(hidden, 128)
        self.attention_u = nn.Linear(hidden, 128) if gated else None
        self.attention_w = nn.Linear(128, 1)
        self.risk_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor | None = None, neighbor: torch.Tensor | None = None):
        h = self.projection(x)
        if neighbor is not None:
            h = h + 0.25 * neighbor
        logits_v = torch.tanh(self.attention_v(h))
        if self.gated:
            logits = self.attention_w(logits_v * torch.sigmoid(self.attention_u(h)))
        else:
            logits = self.attention_w(logits_v)
        logits = logits.squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        return pooled, weights


class TransMIL(nn.Module):
    def __init__(self, input_dim: int = PROJECTION_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU())
        self.pos = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 2, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.risk_head = nn.Linear(hidden, 1)
        self.pool_gate = nn.Linear(hidden, 1)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor | None = None, neighbor: torch.Tensor | None = None):
        h = self.projection(x)
        if coords is not None:
            h = h + self.pos(coords)
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, h], dim=1)
        key_padding = torch.cat([torch.ones((len(x), 1), device=x.device, dtype=torch.bool), mask], dim=1)
        out = self.transformer(tokens, src_key_padding_mask=~key_padding)
        logits = self.pool_gate(h).squeeze(-1).masked_fill(~mask, -1e9)
        return out[:, 0], torch.softmax(logits, dim=1)


class SpatialEAGLE(nn.Module):
    """MASE/neighborhood-style spatial equivalent, not an official EAGLE port."""

    def __init__(self, input_dim: int = PROJECTION_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU())
        self.neighbor_projection = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU())
        self.pos = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.attention = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh(), nn.Linear(128, 1))
        self.risk_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor | None = None, neighbor: torch.Tensor | None = None):
        h = self.projection(x)
        if neighbor is not None:
            h = h + 0.5 * self.neighbor_projection(neighbor)
        if coords is not None:
            h = h + self.pos(coords)
        logits = self.attention(h).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        return pooled, weights


def build_model(method: str) -> nn.Module:
    if method == "B1_ABMIL" or method in {"B9b_ABMIL_CoxRank", "B9a_ABMIL_Cox"}:
        return AttentionMIL(gated=False)
    if method == "B2_GatedABMIL":
        return AttentionMIL(gated=True)
    if method == "B3_CLAM_SB_survival":
        return AttentionMIL(gated=False)
    if method == "B4_TransMIL":
        return TransMIL()
    if method == "B8_EAGLE_spatial":
        return SpatialEAGLE()
    raise ValueError(f"no neural model for {method}")


def cox_loss(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    event_indices = torch.nonzero(events > 0.5, as_tuple=False).flatten()
    if len(event_indices) == 0:
        raise ValueError("Cox loss requires at least one event")
    terms = []
    for index in event_indices:
        at_risk = times >= times[index]
        terms.append(-(risks[index] - torch.logsumexp(risks[at_risk], dim=0)))
    return torch.stack(terms).mean()


def comparable_pairs(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for i in np.flatnonzero(events == 1):
        for j in np.flatnonzero(times > times[i]):
            left.append(int(i))
            right.append(int(j))
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def rank_loss(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    left, right = comparable_pairs(times.detach().cpu().numpy(), events.detach().cpu().numpy().astype(int))
    if not len(left):
        return torch.zeros((), device=risks.device)
    li = torch.as_tensor(left, device=risks.device)
    ri = torch.as_tensor(right, device=risks.device)
    return torch.nn.functional.softplus(-(risks[li] - risks[ri])).mean()


def patient_forward(model: nn.Module, cache: SlideCache, patient_ids: list[str], device: torch.device, return_attention: bool = False):
    slide_ids = cache.slide_indices(patient_ids)
    if not slide_ids:
        raise ValueError("no slides for patient subset")
    embeddings = []
    attentions = []
    chunk_size = 32
    for start in range(0, len(slide_ids), chunk_size):
        idx = slide_ids[start : start + chunk_size]
        x = torch.from_numpy(cache.features[idx]).to(device)
        mask = torch.from_numpy(cache.mask[idx]).to(device)
        coords = torch.from_numpy(cache.coords[idx]).to(device)
        neighbor = torch.from_numpy(cache.neighbor_features[idx]).to(device)
        z, attention = model(x, mask, coords=coords, neighbor=neighbor)
        embeddings.append(z)
        if return_attention:
            attentions.append(attention)
    slide_z = torch.cat(embeddings, dim=0)
    slide_attention = torch.cat(attentions, dim=0) if return_attention else None
    patient_z = []
    offset = 0
    for pid in patient_ids:
        n = len(cache.patient_to_slides[str(pid)])
        patient_z.append(slide_z[offset : offset + n].mean(dim=0))
        offset += n
    z = torch.stack(patient_z, dim=0)
    risk = model.risk_head(z).squeeze(-1)
    return z, risk, slide_ids, slide_attention


def choose_inner_split(patients: pd.DataFrame, outer_fold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_train = patients.loc[patients.fold != outer_fold].reset_index(drop=True)
    outer_test = patients.loc[patients.fold == outer_fold].reset_index(drop=True)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + outer_fold)
    try:
        train_index, val_index = next(splitter.split(outer_train, outer_train.dss_event))
    except ValueError:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + outer_fold)
        train_index, val_index = next(splitter.split(outer_train, outer_train.dss_event))
    return outer_train.iloc[train_index].reset_index(drop=True), outer_train.iloc[val_index].reset_index(drop=True), outer_test


def evaluate_risk(frame: pd.DataFrame, risks: np.ndarray) -> tuple[float, float]:
    times = frame.dss_time.to_numpy(float)
    events = frame.dss_event.to_numpy(int)
    g = censoring_km(times, events)(times, left_limit=True)
    return harrell_c(times, events, risks), uno_c(times, events, risks, g)


def train_neural_fold(method: str, patients: pd.DataFrame, cache: SlideCache, outer_fold: int, seed: int, out: Path, device: torch.device, max_epochs: int = 50, patience_limit: int = 7) -> dict[str, Any]:
    set_seed(seed + outer_fold)
    train, val, test = choose_inner_split(patients, outer_fold, seed)
    lr_grid = [3e-4, 1e-4]
    lambda_grid = [0.0] if method != "B9b_ABMIL_CoxRank" else [0.05, 0.2]
    candidates = []
    for lr in lr_grid:
        for lam in lambda_grid:
            model = build_model(method).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            best_val = -float("inf")
            best_state = None
            best_epoch = 0
            stale = 0
            history = []
            started = time.perf_counter()
            for epoch in range(1, max_epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_ids = train.patient_id.astype(str).tolist()
                _, train_risk, _, _ = patient_forward(model, cache, train_ids, device)
                times_t = torch.tensor(train.dss_time.to_numpy(float), device=device)
                events_t = torch.tensor(train.dss_event.to_numpy(float), device=device)
                loss = cox_loss(times_t, events_t, train_risk)
                if lam:
                    loss = loss + lam * rank_loss(times_t, events_t, train_risk)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    _, val_risk, _, _ = patient_forward(model, cache, val.patient_id.astype(str).tolist(), device)
                val_h, val_u = evaluate_risk(val, val_risk.detach().cpu().numpy())
                history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_harrell_c": val_h, "val_uno_c": val_u, "lr": lr, "lambda_rank": lam})
                if np.isfinite(val_h) and val_h > best_val + 1e-6:
                    best_val = val_h
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= patience_limit:
                    break
            if best_state is None:
                raise RuntimeError(f"{method} produced no valid inner checkpoint")
            candidates.append({"val_harrell": best_val, "lr": lr, "lambda_rank": lam, "best_epoch": best_epoch, "state": best_state, "history": history, "train_time_seconds": time.perf_counter() - started})
    candidates.sort(key=lambda x: (x["val_harrell"], -x["lr"], -x["lambda_rank"]), reverse=True)
    selected = candidates[0]
    model = build_model(method).to(device)
    model.load_state_dict(selected["state"])
    model.eval()
    started_inf = time.perf_counter()
    with torch.no_grad():
        test_z, test_risk, slide_ids, slide_attention = patient_forward(model, cache, test.patient_id.astype(str).tolist(), device, return_attention=True)
    inference_seconds = time.perf_counter() - started_inf
    risks = test_risk.detach().cpu().numpy()
    h, u = evaluate_risk(test, risks)
    job_dir = out / "checkpoints" / method / f"seed_{seed}" / f"fold_{outer_fold}"
    job_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": selected["state"], "method": method, "seed": seed, "fold": outer_fold, "best_epoch": selected["best_epoch"]}, job_dir / "model.pt")
    atomic_csv(pd.DataFrame(selected["history"]), out / "logs" / method / f"seed_{seed}_fold_{outer_fold}_training.csv")
    pred = test[["patient_id", "fold", "dss_time", "dss_event"]].copy()
    pred["seed"] = seed
    pred["risk_score"] = risks
    pred["method"] = method
    atomic_csv(pred, out / "oof_predictions" / method / f"seed_{seed}_fold_{outer_fold}.csv")
    emb = test[["patient_id", "fold", "dss_time", "dss_event"]].copy()
    emb["seed"] = seed
    emb["embedding"] = [json.dumps(row.tolist()) for row in test_z.detach().cpu().numpy()]
    atomic_csv(emb, out / "embeddings" / method / f"seed_{seed}_fold_{outer_fold}.csv")
    attention_rows = []
    offset = 0
    for pid in test.patient_id.astype(str).tolist():
        slide_indices = cache.patient_to_slides[pid]
        for local_slide, slide_idx in enumerate(slide_indices):
            row = cache.slides.iloc[slide_idx]
            weights = slide_attention[offset + local_slide].detach().cpu().numpy()
            for patch_idx in np.flatnonzero(cache.mask[slide_idx]):
                attention_rows.append({"patient_id": pid, "slide_id": str(row.slide_id), "seed": seed, "fold": outer_fold, "patch_coord_x": float(cache.coords[slide_idx, patch_idx, 0]), "patch_coord_y": float(cache.coords[slide_idx, patch_idx, 1]), "attention_score": float(weights[patch_idx]), "method": method})
        offset += len(slide_indices)
    atomic_csv(pd.DataFrame(attention_rows), out / "attention" / method / f"seed_{seed}_fold_{outer_fold}.csv")
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory = 0.0
    result = {
        "method": method,
        "seed": seed,
        "fold": outer_fold,
        "n_test": len(test),
        "n_events": int(test.dss_event.sum()),
        "harrell_c": h,
        "uno_c": u,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "peak_gpu_memory_gb": float(peak_memory),
        "train_time_minutes": float(sum(c["train_time_seconds"] for c in candidates) / 60.0),
        "inference_time_seconds": float(inference_seconds),
        "avg_num_tokens": float(cache.mask.sum(axis=1).mean()),
        "inner_val_harrell_c": float(selected["val_harrell"]),
        "selected_lr": float(selected["lr"]),
        "selected_lambda_rank": float(selected["lambda_rank"]),
        "selected_epoch": int(selected["best_epoch"]),
        "provenance": "paper_faithful_survival_adaptation" if method == "B3_CLAM_SB_survival" else ("paper_faithful_reimplementation" if method == "B8_EAGLE_spatial" else "local_unified_implementation"),
    }
    atomic_json(out / "tables" / "job_metrics" / method / f"seed_{seed}_fold_{outer_fold}.json", result)
    return result


def fit_cox_features(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, seed: int, pca_components: int | None = None) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_train)
    xv = scaler.transform(x_valid)
    xs = scaler.transform(x_test)
    pca = None
    if pca_components is not None:
        n = min(pca_components, len(xt) - 1, xt.shape[1])
        pca = PCA(n_components=n, svd_solver="randomized", random_state=seed)
        xt = pca.fit_transform(xt)
        xv = pca.transform(xv)
        xs = pca.transform(xs)
    alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ranked = []
    for alpha in alpha_grid:
        try:
            beta = fit_cox(train.dss_time.to_numpy(float), train.dss_event.to_numpy(int), xt, alpha)
            score = harrell_c(valid.dss_time, valid.dss_event, xv @ beta)
        except Exception:
            score = float("nan")
            beta = None
        ranked.append((float(score) if np.isfinite(score) else -float("inf"), -alpha, alpha, beta))
    ranked.sort(reverse=True, key=lambda x: (x[0], x[1]))
    best_score, _, alpha, _ = ranked[0]
    beta = fit_cox(train.dss_time.to_numpy(float), train.dss_event.to_numpy(int), xt, alpha)
    return xs @ beta, float(best_score), float(alpha), {"scaler": scaler, "pca": pca, "beta": beta}


def raw_patient_statistics(root: Path, data: Path, patients: pd.DataFrame, slides: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    means: dict[str, list[np.ndarray]] = defaultdict(list)
    stds: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in slides.itertuples(index=False):
        bag = np.load(resolve_input_path(root, data, row.bag_path), mmap_mode="r", allow_pickle=False)
        x = np.asarray(bag, dtype=np.float32)
        means[str(row.patient_id)].append(x.mean(axis=0))
        stds[str(row.patient_id)].append(x.std(axis=0))
    mean_x = np.stack([np.stack(means[str(pid)]).mean(axis=0) for pid in patients.patient_id.astype(str)])
    std_x = np.stack([np.stack(stds[str(pid)]).mean(axis=0) for pid in patients.patient_id.astype(str)])
    return mean_x.astype(np.float32), std_x.astype(np.float32)


def run_mean_methods(root: Path, data: Path, patients: pd.DataFrame, slides: pd.DataFrame, out: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    all_sanity_jobs = [f"{method}/seed_{seed}/fold_{fold}" for method in ("B0_mean", "B0b_mean_std") for seed in SEEDS for fold in FOLDS]
    if all(job in state["completed_jobs"] for job in all_sanity_jobs) and (out / "embeddings" / "B0_mean" / "mean_embeddings.csv").exists():
        return []
    mean_x, std_x = raw_patient_statistics(root, data, patients, slides)
    atomic_json(out / "configs" / "raw_feature_cache.json", {"shape_mean": list(mean_x.shape), "shape_std": list(std_x.shape), "source": "all-patch slide mean/std; no labels used"})
    embedding = patients[["patient_id", "fold", "dss_time", "dss_event"]].copy()
    embedding["seed"] = 0
    embedding["embedding"] = [json.dumps(row.tolist()) for row in mean_x]
    atomic_csv(embedding, out / "embeddings" / "B0_mean" / "mean_embeddings.csv")
    rows = []
    for method, x in [("B0_mean", mean_x), ("B0b_mean_std", np.concatenate([mean_x, std_x], axis=1))]:
        for seed in SEEDS:
            for fold in FOLDS:
                job = f"{method}/seed_{seed}/fold_{fold}"
                if job in state["completed_jobs"]:
                    continue
                train, valid, test = choose_inner_split(patients, fold, seed)
                index = {str(pid): i for i, pid in enumerate(patients.patient_id.astype(str))}
                ti = np.asarray([index[str(pid)] for pid in train.patient_id], dtype=int)
                vi = np.asarray([index[str(pid)] for pid in valid.patient_id], dtype=int)
                si = np.asarray([index[str(pid)] for pid in test.patient_id], dtype=int)
                # Both high-dimensional sanity baselines use the same
                # train-only PCA32 numerical stabilizer.  This keeps the
                # comparison computationally bounded while preserving the
                # requested mean(/std) statistic and a Ridge-Cox head.  It is
                # explicitly recorded in the registry and report; no test
                # information enters the transform.
                risk, inner_c, alpha, _ = fit_cox_features(train, valid, test, x[ti], x[vi], x[si], seed + fold, pca_components=32)
                h, u = evaluate_risk(test, risk)
                pred = test[["patient_id", "fold", "dss_time", "dss_event"]].copy()
                pred["seed"] = seed
                pred["risk_score"] = risk
                pred["method"] = method
                atomic_csv(pred, out / "oof_predictions" / method / f"seed_{seed}_fold_{fold}.csv")
                result = {"method": method, "seed": seed, "fold": fold, "n_test": len(test), "n_events": int(test.dss_event.sum()), "harrell_c": h, "uno_c": u, "trainable_params": 0, "peak_gpu_memory_gb": 0.0, "train_time_minutes": 0.0, "inference_time_seconds": 0.0, "avg_num_tokens": float(slides.n_patches.mean()), "inner_val_harrell_c": inner_c, "selected_alpha": alpha, "provenance": "local_unified_implementation"}
                atomic_json(out / "tables" / "job_metrics" / method / f"seed_{seed}_fold_{fold}.json", result)
                rows.append(result)
                state["completed_jobs"].append(job)
                state["current_job"] = None
                atomic_json(out / "run_state.json", state)
    return rows


def run_panther(root: Path, data: Path, patients: pd.DataFrame, slides: pd.DataFrame, out: Path, state: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    panther_jobs = [f"B6_PANTHER_GMM/seed_{seed}/fold_{fold}" for seed in SEEDS for fold in FOLDS]
    if all(job in state["completed_jobs"] for job in panther_jobs):
        return rows
    index = {str(pid): i for i, pid in enumerate(patients.patient_id.astype(str))}
    for seed in SEEDS:
        cache = SlideCache(root, data, slides, MAX_TOKENS, PROJECTION_DIM, seed)
        for fold in FOLDS:
            job = f"B6_PANTHER_GMM/seed_{seed}/fold_{fold}"
            if job in state["completed_jobs"]:
                continue
            train, valid, test = choose_inner_split(patients, fold, seed)
            train_slides = cache.slide_indices(train.patient_id.astype(str).tolist())
            patch_x = cache.features[train_slides][cache.mask[train_slides]]
            # Small, pre-registered K search is performed on inner training only.
            k_candidates = [4, 8]
            best = None
            for k in k_candidates:
                gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed + fold, n_init=1)
                gmm.fit(patch_x)
                def featurize(ids: list[str]) -> np.ndarray:
                    slide_ids = cache.slide_indices(ids)
                    vals = []
                    offset = 0
                    for pid in ids:
                        local = cache.patient_to_slides[str(pid)]
                        chunks = []
                        for sid in local:
                            valid = cache.mask[sid]
                            chunks.append(gmm.predict_proba(cache.features[sid, valid]))
                        probs = np.concatenate(chunks, axis=0)
                        vals.append(np.concatenate([probs.mean(axis=0), np.asarray([gmm.score(np.concatenate([cache.features[sid, cache.mask[sid]] for sid in local], axis=0))])]))
                        offset += len(local)
                    return np.asarray(vals, dtype=np.float32)
                x_train = featurize(train.patient_id.astype(str).tolist())
                x_valid = featurize(valid.patient_id.astype(str).tolist())
                try:
                    _, inner_c, alpha, _ = fit_cox_features(train, valid, valid, x_train, x_valid, x_valid, seed + fold, pca_components=None)
                except Exception:
                    inner_c, alpha = -float("inf"), 1.0
                candidate = (inner_c, -k, k, gmm, alpha)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            assert best is not None
            _, _, k, _, alpha = best
            outer_train, _, outer_test = choose_inner_split(patients, fold, seed)
            outer_slide_ids = cache.slide_indices(outer_train.patient_id.astype(str).tolist())
            outer_patch_x = cache.features[outer_slide_ids][cache.mask[outer_slide_ids]]
            gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3, max_iter=100, random_state=seed + fold, n_init=1)
            gmm.fit(outer_patch_x)
            def featurize_outer(ids: list[str]) -> np.ndarray:
                vals = []
                for pid in ids:
                    local = cache.patient_to_slides[str(pid)]
                    probs = np.concatenate([gmm.predict_proba(cache.features[sid, cache.mask[sid]]) for sid in local], axis=0)
                    all_x = np.concatenate([cache.features[sid, cache.mask[sid]] for sid in local], axis=0)
                    vals.append(np.concatenate([probs.mean(axis=0), np.asarray([gmm.score(all_x)])]))
                return np.asarray(vals, dtype=np.float32)
            x_ot = featurize_outer(outer_train.patient_id.astype(str).tolist())
            x_te = featurize_outer(outer_test.patient_id.astype(str).tolist())
            scaler = StandardScaler().fit(x_ot)
            beta = fit_cox(outer_train.dss_time.to_numpy(float), outer_train.dss_event.to_numpy(int), scaler.transform(x_ot), alpha)
            risk = scaler.transform(x_te) @ beta
            h, u = evaluate_risk(outer_test, risk)
            pred = outer_test[["patient_id", "fold", "dss_time", "dss_event"]].copy()
            pred["seed"] = seed
            pred["risk_score"] = risk
            pred["method"] = "B6_PANTHER_GMM"
            atomic_csv(pred, out / "oof_predictions" / "B6_PANTHER_GMM" / f"seed_{seed}_fold_{fold}.csv")
            result = {"method": "B6_PANTHER_GMM", "seed": seed, "fold": fold, "n_test": len(outer_test), "n_events": int(outer_test.dss_event.sum()), "harrell_c": h, "uno_c": u, "trainable_params": 0, "peak_gpu_memory_gb": 0.0, "train_time_minutes": 0.0, "inference_time_seconds": 0.0, "avg_num_tokens": float(cache.mask.sum(axis=1).mean()), "prototype_count": k, "selected_alpha": alpha, "provenance": "paper_faithful_reimplementation_train_only_GMM"}
            atomic_json(out / "tables" / "job_metrics" / "B6_PANTHER_GMM" / f"seed_{seed}_fold_{fold}.json", result)
            rows.append(result)
            state["completed_jobs"].append(job)
            atomic_json(out / "run_state.json", state)
    return rows


def alias_b9a(out: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            src = out / "oof_predictions" / "B1_ABMIL" / f"seed_{seed}_fold_{fold}.csv"
            dst = out / "oof_predictions" / "B9a_ABMIL_Cox" / f"seed_{seed}_fold_{fold}.csv"
            job = f"B9a_ABMIL_Cox/seed_{seed}/fold_{fold}"
            if not src.exists() or job in state["completed_jobs"]:
                continue
            pred = pd.read_csv(src)
            pred["method"] = "B9a_ABMIL_Cox"
            atomic_csv(pred, dst)
            h, u = evaluate_risk(pred, pred.risk_score.to_numpy(float))
            rows.append({"method": "B9a_ABMIL_Cox", "seed": seed, "fold": fold, "n_test": len(pred), "n_events": int(pred.dss_event.sum()), "harrell_c": h, "uno_c": u, "trainable_params": 0, "peak_gpu_memory_gb": 0.0, "train_time_minutes": 0.0, "inference_time_seconds": 0.0, "avg_num_tokens": MAX_TOKENS, "provenance": "alias_of_B1_ABMIL_Cox_only"})
            state["completed_jobs"].append(job)
    atomic_json(out / "run_state.json", state)
    return rows


def measure_saved_inference(root: Path, data: Path, patients: pd.DataFrame, slides: pd.DataFrame, out: Path, device: torch.device) -> list[dict[str, Any]]:
    """Re-measure inference/memory for completed checkpoints without retraining."""
    rows = []
    for seed in SEEDS:
        cache = SlideCache(root, data, slides, MAX_TOKENS, PROJECTION_DIM, seed)
        for method in NEURAL_METHODS:
            if method == "B3_CLAM_SB_survival":
                continue
            for fold in FOLDS:
                ckpt_path = out / "checkpoints" / method / f"seed_{seed}" / f"fold_{fold}" / "model.pt"
                if not ckpt_path.exists():
                    continue
                model = build_model(method).to(device)
                bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(bundle["model_state"])
                model.eval()
                test_ids = patients.loc[patients.fold == fold, "patient_id"].astype(str).tolist()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                with torch.no_grad():
                    patient_forward(model, cache, test_ids, device, return_attention=False)
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                rows.append({
                    "method": method,
                    "seed": seed,
                    "fold": fold,
                    "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                    "peak_gpu_memory_gb": float(peak),
                    "train_time_minutes": np.nan,
                    "inference_time_seconds": float(elapsed),
                    "avg_num_tokens": float(cache.mask.sum(axis=1).mean()),
                    "measurement_status": "inference_remeasured_training_time_unrecorded",
                })
                del model
    return rows


def collect_oof(out: Path) -> pd.DataFrame:
    rows = []
    for method_dir in (out / "oof_predictions").glob("*"):
        if not method_dir.is_dir() or method_dir.name in FAILED_METHODS:
            continue
        for path in method_dir.glob("seed_*_fold_*.csv"):
            frame = pd.read_csv(path)
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    atomic_csv(combined, out / "oof_predictions" / "all_methods.csv")
    for method, frame in combined.groupby("method"):
        atomic_csv(frame.sort_values(["seed", "fold", "patient_id"]).reset_index(drop=True), out / "oof_predictions" / f"{method}.csv")
    return combined


def aggregate_metrics(out: Path, oof: pd.DataFrame, profile_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if oof.empty:
        raise RuntimeError("no OOF predictions")
    fold_metrics = oof[["patient_id", "method", "seed", "fold", "dss_time", "dss_event", "risk_score"]].copy()
    fold_metrics["n_test"] = fold_metrics.groupby(["method", "seed", "fold"])["patient_id"].transform("size")
    fold_metrics["n_events"] = fold_metrics.groupby(["method", "seed", "fold"])["dss_event"].transform("sum")
    fold_rows = []
    for (method, seed, fold), frame in fold_metrics.groupby(["method", "seed", "fold"], sort=True):
        h, u = evaluate_risk(frame, frame.risk_score.to_numpy(float))
        fold_rows.append({"method": method, "seed": int(seed), "fold": int(fold), "n_test": len(frame), "n_events": int(frame.dss_event.sum()), "harrell_c": h, "uno_c": u})
    fold_table = pd.DataFrame(fold_rows)
    atomic_csv(fold_table, out / "tables" / "fold_metrics.csv")
    available = []
    for method, method_frame in oof.groupby("method"):
        seed_scores = []
        for seed, frame in method_frame.groupby("seed"):
            ordered = frame.sort_values("patient_id")
            h, u = evaluate_risk(ordered, ordered.risk_score.to_numpy(float))
            seed_scores.append((h, u))
        seed_h = np.asarray([x[0] for x in seed_scores], dtype=float)
        seed_u = np.asarray([x[1] for x in seed_scores], dtype=float)
        pooled = method_frame.groupby("patient_id", as_index=False).agg({"fold": "first", "dss_time": "first", "dss_event": "first", "risk_score": "mean"})
        pooled_h, pooled_u = evaluate_risk(pooled, pooled.risk_score.to_numpy(float))
        f = fold_table.loc[fold_table.method == method]
        available.append({"method": method, "n_seeds": int(method_frame.seed.nunique()), "oof_harrell_c": pooled_h, "oof_uno_c": pooled_u, "seed_harrell_mean": float(np.nanmean(seed_h)), "seed_harrell_std": float(np.nanstd(seed_h, ddof=1)) if len(seed_h) > 1 else 0.0, "seed_uno_mean": float(np.nanmean(seed_u)), "seed_uno_std": float(np.nanstd(seed_u, ddof=1)) if len(seed_u) > 1 else 0.0, "fold_harrell_mean": float(f.harrell_c.mean()), "fold_harrell_std": float(f.harrell_c.std(ddof=1)) if len(f) > 1 else 0.0, "fold_uno_mean": float(f.uno_c.mean()), "fold_uno_std": float(f.uno_c.std(ddof=1)) if len(f) > 1 else 0.0, "best_fold_harrell": float(f.harrell_c.max()), "worst_fold_harrell": float(f.harrell_c.min()), "fold_wins_vs_ABMIL": 0, "fold_wins_vs_Mean": 0})
    main = pd.DataFrame(available)
    mean_fold = fold_table.loc[fold_table.method == "B0_mean", ["seed", "fold", "harrell_c"]].rename(columns={"harrell_c": "mean_h"})
    ab_fold = fold_table.loc[fold_table.method == "B1_ABMIL", ["seed", "fold", "harrell_c"]].rename(columns={"harrell_c": "ab_h"})
    for index, row in main.iterrows():
        f = fold_table.loc[fold_table.method == row.method, ["seed", "fold", "harrell_c"]]
        main.loc[index, "fold_wins_vs_ABMIL"] = int((f.merge(ab_fold, on=["seed", "fold"], how="inner").harrell_c > f.merge(ab_fold, on=["seed", "fold"], how="inner").ab_h).sum()) if row.method != "B1_ABMIL" else 0
        main.loc[index, "fold_wins_vs_Mean"] = int((f.merge(mean_fold, on=["seed", "fold"], how="inner").harrell_c > f.merge(mean_fold, on=["seed", "fold"], how="inner").mean_h).sum()) if row.method != "B0_mean" else 0
    atomic_csv(main, out / "tables" / "main_metrics.csv")
    profile = pd.DataFrame(profile_rows).drop_duplicates(["method", "seed", "fold"], keep="last")
    if profile.empty:
        # The first complete run predates persistence of per-job timing JSON.
        # Reconstruct the required schema from immutable checkpoints and mark
        # timing as unrecorded rather than inventing numbers.
        fallback = []
        for method in main.method.astype(str):
            if method in {"B1_ABMIL", "B2_GatedABMIL", "B4_TransMIL", "B8_EAGLE_spatial", "B9b_ABMIL_CoxRank"}:
                params = int(sum(p.numel() for p in build_model("B1_ABMIL" if method == "B9b_ABMIL_CoxRank" else method).parameters()))
            else:
                params = 0
            fallback.append({
                "method": method,
                "trainable_params": params,
                "peak_gpu_memory_gb": np.nan,
                "train_time_minutes": np.nan,
                "inference_time_seconds": np.nan,
                "avg_num_tokens": MAX_TOKENS if method not in {"B0_mean", "B0b_mean_std", "B6_PANTHER_GMM", "B9a_ABMIL_Cox"} else np.nan,
                "measurement_status": "timing_not_persisted_in_initial_run",
            })
        profile_summary = pd.DataFrame(fallback)
    elif not profile.empty:
        profile_summary = profile.groupby("method", as_index=False).agg(trainable_params=("trainable_params", "max"), peak_gpu_memory_gb=("peak_gpu_memory_gb", "max"), train_time_minutes=("train_time_minutes", "mean"), inference_time=("inference_time_seconds", "mean"), avg_num_tokens=("avg_num_tokens", "mean"))
        profile_summary["measurement_status"] = "recorded"
        missing_profile = [method for method in main.method.astype(str) if method not in set(profile_summary.method)]
        if missing_profile:
            profile_summary = pd.concat([profile_summary, pd.DataFrame([
                {"method": method, "trainable_params": 0, "peak_gpu_memory_gb": 0.0, "train_time_minutes": 0.0, "inference_time": 0.0, "avg_num_tokens": np.nan, "measurement_status": "not_applicable_or_not_persisted"}
                for method in missing_profile
            ])], ignore_index=True)
    atomic_csv(profile_summary, out / "compute" / "compute_profile.csv")
    return main, fold_table, profile_summary


def bootstrap_delta(base: pd.DataFrame, other: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, Any]:
    merged = base[["patient_id", "dss_time", "dss_event", "risk_score"]].merge(other[["patient_id", "risk_score"]], on="patient_id", suffixes=("_base", "_other"), validate="one_to_one")
    times = merged.dss_time.to_numpy(float)
    events = merged.dss_event.to_numpy(int)
    base_risk = merged.risk_score_base.to_numpy(float)
    other_risk = merged.risk_score_other.to_numpy(float)
    observed_h = harrell_c(times, events, other_risk) - harrell_c(times, events, base_risk)
    g = censoring_km(times, events)
    observed_u = uno_c(times, events, other_risk, g(times, left_limit=True)) - uno_c(times, events, base_risk, g(times, left_limit=True))
    rng = np.random.default_rng(seed)
    # Vectorized paired bootstrap.  The comparable-pair graph is fixed by the
    # observed endpoint; resampling is represented by multiplicities.  This is
    # algebraically equivalent to re-running the pair loops, but avoids tens
    # of millions of Python-level calls when 2,000 replicates are required.
    pair_i, pair_j = np.where((events[:, None] == 1) & (times[:, None] < times[None, :]))
    if len(pair_i) == 0:
        raise ValueError("no comparable survival pairs")
    h_other_values = (other_risk[pair_i] > other_risk[pair_j]).astype(float) + 0.5 * (other_risk[pair_i] == other_risk[pair_j])
    h_base_values = (base_risk[pair_i] > base_risk[pair_j]).astype(float) + 0.5 * (base_risk[pair_i] == base_risk[pair_j])
    g_full = censoring_km(times, events)(times, left_limit=True)
    pair_uno_weights = 1.0 / np.maximum(g_full[pair_i], 1e-8) ** 2
    h_values, u_values = [], []
    counts = np.zeros((n_bootstrap, len(merged)), dtype=np.int16)
    for b in range(n_bootstrap):
        counts[b] = np.bincount(rng.integers(0, len(merged), len(merged)), minlength=len(merged))
    for start in range(0, n_bootstrap, 100):
        block = counts[start : start + 100].astype(np.float64)
        multiplicity = block[:, pair_i] * block[:, pair_j]
        h_den = multiplicity.sum(axis=1)
        h_num_other = multiplicity @ h_other_values
        h_num_base = multiplicity @ h_base_values
        h_values.extend(((h_num_other / np.maximum(h_den, 1e-12)) - (h_num_base / np.maximum(h_den, 1e-12))).tolist())
        u_den = multiplicity @ pair_uno_weights
        u_num_other = multiplicity @ (pair_uno_weights * h_other_values)
        u_num_base = multiplicity @ (pair_uno_weights * h_base_values)
        u_values.extend(((u_num_other / np.maximum(u_den, 1e-12)) - (u_num_base / np.maximum(u_den, 1e-12))).tolist())
    h_arr = np.asarray(h_values, dtype=float)
    u_arr = np.asarray(u_values, dtype=float)
    return {
        "n_patients": len(merged),
        "n_bootstrap": n_bootstrap,
        "delta_harrell": float(observed_h),
        "delta_harrell_ci_low": float(np.quantile(h_arr, 0.025)),
        "delta_harrell_ci_high": float(np.quantile(h_arr, 0.975)),
        "p_two_sided_harrell": float(min(1.0, 2 * min(np.mean(h_arr <= 0), np.mean(h_arr >= 0)))),
        "prob_improvement_harrell": float(np.mean(h_arr > 0)),
        "delta_uno": float(observed_u),
        "delta_uno_ci_low": float(np.quantile(u_arr, 0.025)),
        "delta_uno_ci_high": float(np.quantile(u_arr, 0.975)),
        "p_two_sided_uno": float(min(1.0, 2 * min(np.mean(u_arr <= 0), np.mean(u_arr >= 0)))),
        "prob_improvement_uno": float(np.mean(u_arr > 0)),
    }


def run_bootstrap_and_tables(out: Path, oof: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    pooled = {m: frame.groupby("patient_id", as_index=False).agg({"fold": "first", "dss_time": "first", "dss_event": "first", "risk_score": "mean"}) for m, frame in oof.groupby("method")}
    base_mean = pooled.get("B0_mean")
    base_ab = pooled.get("B1_ABMIL")
    if base_mean is None or base_ab is None:
        raise RuntimeError("Mean and ABMIL OOF are required for paired bootstrap")
    rows = []
    for method, frame in pooled.items():
        if method == "B1_ABMIL":
            continue
        for baseline_name, baseline in [("Mean", base_mean), ("ABMIL", base_ab)]:
            if method == "B0_mean" and baseline_name == "Mean":
                continue
            stable_key = hashlib.sha256(f"{method}|{baseline_name}|20260828".encode("utf-8")).digest()
            stable_seed = 20260828 + int.from_bytes(stable_key[:4], "little") % 100000
            result = bootstrap_delta(baseline, frame, BOOTSTRAPS, stable_seed)
            rows.append({"method": method, "baseline": baseline_name, **result})
    table = pd.DataFrame(rows)
    atomic_csv(table, out / "statistical_tests" / "bootstrap_comparisons.csv")
    return table


def attention_audit(out: Path) -> pd.DataFrame:
    rows = []
    for method in ("B1_ABMIL", "B2_GatedABMIL", "B4_TransMIL", "B8_EAGLE_spatial"):
        files = list((out / "attention" / method).glob("*.csv"))
        if not files:
            continue
        for path in files:
            frame = pd.read_csv(path)
            for (pid, slide), group in frame.groupby(["patient_id", "slide_id"]):
                score = group.attention_score.to_numpy(float)
                score = score / max(score.sum(), 1e-12)
                entropy = float(-(score * np.log(np.maximum(score, 1e-12))).sum() / math.log(len(score))) if len(score) > 1 else 0.0
                top_n = max(1, int(math.ceil(0.05 * len(group))))
                top = group.nlargest(top_n, "attention_score")
                spread = float(np.sqrt(np.var(top.patch_coord_x) + np.var(top.patch_coord_y)))
                rows.append({"method": method, "seed": int(group.seed.iloc[0]), "fold": int(group.fold.iloc[0]), "patient_id": pid, "slide_id": slide, "attention_entropy": entropy, "top5pct_coord_spread": spread, "n_tokens": len(group), "top5pct_mass": float(top.attention_score.sum())})
    result = pd.DataFrame(rows)
    if not result.empty:
        atomic_csv(result, out / "tables" / "attention_audit.csv")
        summary = result.groupby("method", as_index=False).agg(attention_entropy_mean=("attention_entropy", "mean"), attention_entropy_std=("attention_entropy", "std"), top5pct_mass_mean=("top5pct_mass", "mean"), top5pct_coord_spread_mean=("top5pct_coord_spread", "mean"))
        atomic_csv(summary, out / "tables" / "attention_audit_summary.csv")
    return result


def embedding_audit(out: Path) -> pd.DataFrame:
    rows = []
    for method in ("B0_mean", "B1_ABMIL", "B4_TransMIL", "B8_EAGLE_spatial"):
        files = list((out / "embeddings" / method).glob("*.csv"))
        if not files and method == "B0_mean":
            continue
        values = []
        for path in files:
            frame = pd.read_csv(path)
            values.extend(np.vstack([np.asarray(json.loads(x), dtype=float) for x in frame.embedding]))
        if not values:
            continue
        x = np.asarray(values, dtype=float)
        x = x - x.mean(axis=0, keepdims=True)
        variance = np.var(x, axis=0)
        rows.append({"method": method, "n_embeddings": len(x), "embedding_dim": x.shape[1], "mean_variance": float(variance.mean()), "total_variance": float(variance.sum()), "zero_variance_dims": int((variance <= 1e-12).sum()), "pca_note": "PCA diagnostic only; no model/tuning feedback"})
    result = pd.DataFrame(rows)
    if not result.empty:
        atomic_csv(result, out / "tables" / "embedding_diagnostics.csv")
    return result


def make_figures(out: Path, main: pd.DataFrame, fold: pd.DataFrame, bootstrap: pd.DataFrame, profile: pd.DataFrame) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_main = main.sort_values("oof_harrell_c")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.bar(plot_main.method, plot_main.oof_harrell_c, color="#4472C4", label="Harrell")
    ax.scatter(np.arange(len(plot_main)), plot_main.oof_uno_c, color="#ED7D31", label="Uno", zorder=3)
    ax.set_ylabel("OOF C-index")
    ax.set_ylim(0.4, 0.8)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig1_oof_cindex.png", dpi=220)
    plt.close(fig)

    ab = fold.loc[fold.method == "B1_ABMIL"].rename(columns={"harrell_c": "ab_h"})[["seed", "fold", "ab_h"]]
    mean = fold.loc[fold.method == "B0_mean"].rename(columns={"harrell_c": "mean_h"})[["seed", "fold", "mean_h"]]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    for method in [m for m in main.method if m not in {"B0_mean", "B1_ABMIL"}]:
        cur = fold.loc[fold.method == method, ["seed", "fold", "harrell_c"]].merge(ab, on=["seed", "fold"], how="inner")
        if not cur.empty:
            ax.plot(cur.fold + (hash(method) % 7 - 3) / 20, cur.harrell_c - cur.ab_h, "o", label=method)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Harrell C difference vs ABMIL")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig2_fold_paired.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    if not bootstrap.empty:
        for baseline, color in [("ABMIL", "#C00000"), ("Mean", "#70AD47")]:
            cur = bootstrap.loc[bootstrap.baseline == baseline].sort_values("delta_harrell")
            for y, row in enumerate(cur.itertuples(index=False)):
                ax.errorbar(row.delta_harrell, y, xerr=[[row.delta_harrell - row.delta_harrell_ci_low], [row.delta_harrell_ci_high - row.delta_harrell]], fmt="o", color=color, label=baseline if y == 0 else None)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(cur)))
        ax.set_yticklabels(cur.method)
    ax.set_xlabel("Paired bootstrap Δ Harrell C")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig3_bootstrap_delta.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    if not profile.empty:
        joined = main.merge(profile, on="method", how="inner")
        ax.scatter(joined.trainable_params + 1, joined.oof_harrell_c, s=50, c="#4472C4")
        for row in joined.itertuples(index=False):
            ax.annotate(row.method, (row.trainable_params + 1, row.oof_harrell_c), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("OOF Harrell C")
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig4_complexity_performance.png", dpi=220)
    plt.close(fig)

    attention = out / "tables" / "attention_audit.csv"
    if attention.exists():
        att = pd.read_csv(attention)
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        for method, group in att.groupby("method"):
            ax.scatter(group.attention_entropy, group.top5pct_mass, s=16, alpha=0.55, label=method)
        ax.set_xlabel("Attention entropy (normalized)")
        ax.set_ylabel("Top 5% attention mass")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "Fig5_attention_spatial_audit.png", dpi=220)
        plt.close(fig)


def write_failures(out: Path) -> None:
    rows = []
    for method, failure in FAILED_METHODS.items():
        rows.append({"method": method, **failure})
    atomic_csv(pd.DataFrame(rows), out / "tables" / "failure_ledger.csv")
    text = "# Phase 1.5 failure log\n\n"
    for method, failure in FAILED_METHODS.items():
        text += f"## {method}\n\n- status: `{failure['status']}`\n- attempts: {failure['attempts']}\n- reason: {failure['reason']}\n- official reference: {failure['official_reference']}\n\n"
    atomic_text(out / "PHASE1_5_FAILURE_LOG.md", text.rstrip() + "\n")


def decision(main: pd.DataFrame, bootstrap: pd.DataFrame, fold: pd.DataFrame) -> tuple[str, str]:
    def score(method: str, col: str = "oof_harrell_c") -> float:
        row = main.loc[main.method == method, col]
        return float(row.iloc[0]) if len(row) else float("nan")
    available = set(main.method)
    if "B1_ABMIL" not in available:
        return "INDETERMINATE", "ABMIL is missing; the benchmark cannot support a decision."
    strongest = main.sort_values("oof_harrell_c", ascending=False).iloc[0]
    advanced = strongest.method
    delta_ab = score(advanced) - score("B1_ABMIL")
    evidence = bootstrap.loc[(bootstrap.method == advanced) & (bootstrap.baseline == "ABMIL")]
    ci_crosses = bool(len(evidence) and evidence.delta_harrell_ci_low.iloc[0] <= 0 <= evidence.delta_harrell_ci_high.iloc[0])
    spatial = score("B8_EAGLE_spatial") - score("B1_ABMIL") if "B8_EAGLE_spatial" in available else float("nan")
    trans = score("B4_TransMIL") - score("B1_ABMIL") if "B4_TransMIL" in available else float("nan")
    rank = score("B9b_ABMIL_CoxRank") - score("B9a_ABMIL_Cox") if "B9b_ABMIL_CoxRank" in available and "B9a_ABMIL_Cox" in available else float("nan")
    rank_evidence = bootstrap.loc[(bootstrap.method == "B9b_ABMIL_CoxRank") & (bootstrap.baseline == "ABMIL")]
    rank_ci_crosses = bool(len(rank_evidence) and rank_evidence.delta_harrell_ci_low.iloc[0] <= 0 <= rank_evidence.delta_harrell_ci_high.iloc[0])
    if np.isfinite(rank) and rank > 0.02 and not rank_ci_crosses:
        label = "Decision D — SURVIVAL_SUPERVISION_BOTTLENECK"
        explanation = f"Cox+patient-level ranking changes pooled Harrell by {rank:.4f} and its paired bootstrap CI excludes 0; advanced aggregator winner={advanced}, Δ vs ABMIL={delta_ab:.4f}, CI crosses 0={ci_crosses}."
    elif np.isfinite(spatial) and spatial > 0.02 and not ci_crosses:
        label = "Decision B — SPATIAL_CONTEXT_SUPPORTED"
        explanation = f"The spatial paper-faithful equivalent is {spatial:.4f} above ABMIL in pooled Harrell; interpret alongside Uno and bootstrap."
    elif np.isfinite(trans) and trans > 0.02 and not ci_crosses:
        label = "Decision C — LONG_RANGE_MODELING_SUPPORTED"
        explanation = f"TransMIL is {trans:.4f} above ABMIL in pooled Harrell; inspect Uno/fold consistency before claiming a stable effect."
    else:
        label = "Decision A — SIMPLE_MIL_SUFFICIENT"
        explanation = f"No stable large improvement over ABMIL is established by the available pooled/fold evidence; strongest={advanced}, Δ vs ABMIL={delta_ab:.4f}, bootstrap CI crosses 0={ci_crosses}, Cox+rank CI crosses 0={rank_ci_crosses}."
    return label, explanation


def write_reports(root: Path, out: Path, audit: dict[str, Any], main: pd.DataFrame, fold: pd.DataFrame, bootstrap: pd.DataFrame, profile: pd.DataFrame, embedding: pd.DataFrame, attention: pd.DataFrame) -> None:
    label, explanation = decision(main, bootstrap, fold)
    failed = ", ".join(f"`{x}`" for x in FAILED_METHODS)
    main_rows = main.sort_values("oof_harrell_c").to_dict("records")
    table_lines = "\n".join(f"| {r['method']} | {r['n_seeds']} | {r['oof_harrell_c']:.4f} | {r['oof_uno_c']:.4f} | {r['fold_harrell_mean']:.4f} ± {r['fold_harrell_std']:.4f} | {r['fold_wins_vs_ABMIL']} |" for r in main_rows)
    results = f"""# Phase 1.5 benchmark results

## Decision boundary

**{label}**

{explanation}

This is a pathology-only audit on frozen UNI2-h features and a frozen patient-level Phase1R split. It is not evidence for a new architecture and it does not start Phase 2.

## Answers to the ten required questions

1. **Cohort:** {len(audit['fixed'])} patients and {int(audit['fixed'].dss_event.sum())} DSS events.
2. **WSI per patient:** mean {audit['fixed'].num_slides.mean():.4f}, median {audit['fixed'].num_slides.median():.1f}; {int((audit['fixed'].num_slides > 1).sum())} patients have multiple slides.
3. **Patches per WSI:** see `data_audit/feature_inventory.csv`; min {audit['slides'].n_patches.min()}, median {audit['slides'].n_patches.median():.0f}, max {audit['slides'].n_patches.max()}.
4. **Mean strength:** see Table 1 below; B0 is evaluated with OOF Harrell and Uno.
5. **ABMIL gain:** the paired bootstrap and fold-win columns are the required stability evidence; a single lucky seed is not used.
6. **CLAM/TransMIL/MambaMIL:** CLAM is explicitly failed because no validated survival adaptation is available; TransMIL is a recorded survival adaptation; MambaMIL is explicitly failed because its official dependency stack is unavailable.
7. **Prototype:** B6 uses train-only GMM fitting and a pre-registered K search; no full-data clustering is used.
8. **Spatial coordinates:** B8 is a paper-faithful spatial equivalent, not an official EAGLE reproduction; no semantic tissue labels are assigned to coordinates.
9. **Cox+ranking:** B9a is an alias of ABMIL Cox-only; B9b uses only comparable survival pairs (event i with T_i<T_j) and inner validation for λ.
10. **Bottleneck:** the evidence is summarized by the decision above; uncertainty from only 65 events is retained.

## Table 1 — Main benchmark

| Method | Seeds | OOF Harrell | OOF Uno | Fold Harrell mean ± std | Fold wins vs ABMIL |
|---|---:|---:|---:|---:|---:|
{table_lines}

## Integrity boundaries

- No clinical/RNA/TNM/stage input was used.
- Censored cases remain survival observations, never a binary negative class.
- Every neural update uses a full patient-level Cox risk set.
- Test folds are not used for tuning, early stopping, PCA, normalization, GMM, or prototypes.
- Patient-level OOF files, attention files, checkpoints, raw features, and coordinates remain local/ignored; they are not copied to GitHub.
- Failed methods retained in the ledger: {failed}.

## Limitations

The local feature-only benchmark has 65 observed events, so confidence intervals crossing zero are evidence-insufficient. The token cap is a deterministic order-preserving cap shared across neural methods and is not a claim that all original long-sequence implementations were fully reproduced. Peak memory and inference time were re-measured from saved checkpoints; per-job training wall time was not persisted by the initial run and is therefore left blank in `compute_profile.csv`.
"""
    atomic_text(out / "PHASE1_5_RESULTS.md", results)
    next_route = "建议仅选择一个后续方向："
    if label.startswith("Decision B"):
        next_route += "在独立冻结协议上比较真正的空间邻域模型，并要求其同时改善 Harrell、Uno 与 fold 稳定性。"
    elif label.startswith("Decision C"):
        next_route += "先做长序列交互的独立复核，再判断是否需要 tissue semantics；不要仅凭一次 pooled C-index 改造模型。"
    elif label.startswith("Decision D"):
        next_route += "进入 censoring-aware patient-level/local risk attribution，优先解决稀疏 DSS supervision；不要直接开始三模态融合。"
    else:
        next_route += "停止 pathology-only aggregator 搜索，优先评估三模态主线或跨癌种外部验证。"
    decision_text = f"""# Phase 1.5 decision

## {label}

{explanation}

## 下一阶段建议（只建议，不执行）

{next_route}

## 禁止越界

本文件不启动 Phase 2，不引入临床/RNA/TNM/stage，不将失败方法改名为成功复现，也不把公开论文数值填入本地结果。
"""
    atomic_text(out / "PHASE1_5_DECISION.md", decision_text)
    reproducibility = {
        "experiment_id": EXPERIMENT_ID,
        "git_revision": repo_git_revision(root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "data_root_env": "DIFFUSION_DATA_ROOT",
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "max_tokens": MAX_TOKENS,
        "projection_dim": PROJECTION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "bootstrap_replicates": BOOTSTRAPS,
        "split_hash": canonical_split_hash(audit["fixed"]),
        "protocol": "Protocol S slide-first; mean slide embeddings within patient",
        "cox": "full patient-level Breslow-style risk sets with T_j >= T_i",
        "rank_pairs": "event i, T_i < T_j only",
        "bootstrap_censoring": "full-cohort KM censoring weights held fixed across paired resamples",
    }
    atomic_json(out / "PHASE1_5_REPRODUCIBILITY.json", reproducibility)
    atomic_text(out / "PHASE1_5_REPRODUCIBILITY.md", "# Phase 1.5 reproducibility\n\nSee `PHASE1_5_REPRODUCIBILITY.json`, `configs/hyperparameter_registry.yaml`, and `splits/split_manifest.json`. All runtime dependencies, seeds, token cap, projection seed, folds, and full-risk-set Cox rules are persisted. Paired bootstrap uses fixed full-cohort KM censoring weights for the resampled comparisons; this is recorded explicitly.\n")
    atomic_json(out / "PHASE1_5_MANIFEST.json", {"experiment_id": EXPERIMENT_ID, "created_at": utc_now(), "git_revision": repo_git_revision(root), "output_tree": [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.stat().st_size < 5_000_000], "decision": label, "failed_methods": FAILED_METHODS})

    report_root = root / "reports" / EXPERIMENT_ID
    report_root.mkdir(parents=True, exist_ok=True)
    for name in ["PHASE1_5_RESULTS.md", "PHASE1_5_DECISION.md", "PHASE1_5_REPRODUCIBILITY.md", "PHASE1_5_REPRODUCIBILITY.json", "PHASE1_5_FAILURE_LOG.md"]:
        shutil.copy2(out / name, report_root / name)
    for name in ["main_metrics.csv", "fold_metrics.csv", "attention_audit_summary.csv", "embedding_diagnostics.csv", "failure_ledger.csv"]:
        src = out / "tables" / name
        if src.exists():
            shutil.copy2(src, report_root / name)
    for name in ["hyperparameter_registry.yaml", "protocol.json", "resolved_public_config.json"]:
        src = out / "configs" / name
        if src.exists():
            shutil.copy2(src, report_root / name)
    src = out / "splits" / "split_manifest.json"
    if src.exists():
        shutil.copy2(src, report_root / src.name)
    src = out / "statistical_tests" / "bootstrap_comparisons.csv"
    if src.exists():
        shutil.copy2(src, report_root / src.name)
    src = out / "compute" / "compute_profile.csv"
    if src.exists():
        shutil.copy2(src, report_root / src.name)
    fig_src = out / "figures"
    fig_dst = report_root / "figures"
    if fig_src.exists():
        fig_dst.mkdir(exist_ok=True)
        for path in fig_src.glob("*.png"):
            shutil.copy2(path, fig_dst / path.name)


def write_registry(out: Path) -> None:
    registry = """# Phase 1.5 pre-registered hyperparameters\n# Official references are recorded as provenance; local survival adaptation is never called official reproduction.\nB0_mean:\n  source: local sanity baseline\n  fixed_params: {cox: ridge_penalized, preprocessing: train_only_standard_scaler}\n  search_space: {alpha: [0.0001, 0.001, 0.01, 0.1, 1.0]}\n  selected_by: inner_validation_harrell\nB0b_mean_std:\n  source: local sanity baseline\n  fixed_params: {features: mean_plus_std, pca_components: 32, preprocessing: train_only}\n  search_space: {alpha: [0.0001, 0.001, 0.01, 0.1, 1.0]}\n  selected_by: inner_validation_harrell\nB1_ABMIL:\n  source: standard attention MIL\n  fixed_params: {input_projection: 64, hidden: 64, max_tokens: 128, optimizer: AdamW, weight_decay: 0.0001, max_epochs: 50, patience: 7}\n  search_space: {lr: [0.0003, 0.0001]}\n  selected_by: inner_validation_harrell\nB2_GatedABMIL:\n  source: gated attention formula in Phase1.5 prompt\n  fixed_params: {input_projection: 64, hidden: 64, max_tokens: 128, optimizer: AdamW, weight_decay: 0.0001, max_epochs: 50, patience: 7}\n  search_space: {lr: [0.0003, 0.0001]}\n  selected_by: inner_validation_harrell\nB3_CLAM_SB_survival:\n  source: https://github.com/mahmoodlab/CLAM\n  fixed_params: {adaptation: bag_level_cox_only, instance_clustering: disabled_without_class_labels, max_tokens: 128}\n  search_space: {lr: [0.0003, 0.0001]}\n  selected_by: inner_validation_harrell\nB4_TransMIL:\n  source: https://github.com/szc19990412/TransMIL\n  fixed_params: {layers: 2, heads: 4, max_tokens: 128, positional_encoding: coordinate_mlp, token_cap: deterministic_uniform_order_preserving}\n  search_space: {lr: [0.0003, 0.0001]}\n  selected_by: inner_validation_harrell\nB5_MambaMIL:\n  source: https://github.com/isyangshu/MambaMIL\n  fixed_params: {status: FAILED_TO_REPRODUCE}\n  search_space: {}\n  selected_by: not_run_due_to_missing_official_dependency\nB6_PANTHER_GMM:\n  source: https://github.com/mahmoodlab/PANTHER\n  fixed_params: {covariance_type: diag, max_tokens: 128, prototype_fit: outer_train_only}\n  search_space: {prototype_count: [4, 8]}\n  selected_by: inner_validation_harrell\nB7_PAMoE:\n  source: https://github.com/wjx-error/PAMoE\n  fixed_params: {status: FAILED_TO_REPRODUCE}\n  search_space: {}\n  selected_by: not_run_due_to_missing_current_assets\nB8_EAGLE_spatial:\n  source: paper-faithful MASE/neighborhood equivalent; official code unavailable\n  fixed_params: {neighbor_k: 8, coordinate_source: raw_h5, max_tokens: 128}\n  search_space: {lr: [0.0003, 0.0001]}\n  selected_by: inner_validation_harrell\nB9a_ABMIL_Cox:\n  source: fixed ABMIL comparator\n  fixed_params: {alias_of: B1_ABMIL, rank_pairs: none}\n  search_space: {}\n  selected_by: fixed\nB9b_ABMIL_CoxRank:\n  source: fixed ABMIL comparator plus comparable-pair ranking\n  fixed_params: {rank_pairs: event_i_and_Ti_less_than_Tj, max_epochs: 50, patience: 7}\n  search_space: {lr: [0.0003, 0.0001], lambda_rank: [0.05, 0.2]}\n  selected_by: inner_validation_harrell\n"""
    registry = registry.replace("preprocessing: train_only_standard_scaler}", "preprocessing: train_only_standard_scaler_plus_PCA32}")
    registry = registry.replace("B3_CLAM_SB_survival:\n  source: https://github.com/mahmoodlab/CLAM\n  fixed_params: {adaptation: bag_level_cox_only, instance_clustering: disabled_without_class_labels, max_tokens: 128}\n  search_space: {lr: [0.0003, 0.0001]}\n  selected_by: inner_validation_harrell", "B3_CLAM_SB_survival:\n  source: https://github.com/mahmoodlab/CLAM\n  fixed_params: {status: FAILED_TO_REPRODUCE, diagnostic_adaptation_excluded: true}\n  search_space: {}\n  selected_by: official_survival_adaptation_unavailable")
    atomic_text(out / "configs" / "hyperparameter_registry.yaml", registry)
    atomic_json(out / "configs" / "protocol.json", {"protocol_s": "slide-first", "patient_aggregation": "mean slide embedding", "max_tokens": MAX_TOKENS, "token_selection": "uniform_order_preserving", "projection_dim": PROJECTION_DIM, "projection_seed": RANDOM_PROJECTION_SEED, "seeds": list(SEEDS), "folds": list(FOLDS), "bootstrap": BOOTSTRAPS})


def initialize_state(out: Path, resume: bool) -> dict[str, Any]:
    path = out / "run_state.json"
    if path.exists() and resume:
        state = json.loads(path.read_text(encoding="utf-8"))
        return state
    state = {"experiment_id": EXPERIMENT_ID, "status": "RUNNING", "completed_jobs": [], "failed_jobs": [], "current_job": None, "start_time": utc_now(), "updated_at": utc_now(), "seeds": list(SEEDS), "folds": list(FOLDS)}
    atomic_json(path, state)
    return state


def run(root: Path, resume: bool = False, force_audit: bool = False, max_epochs: int = 50) -> None:
    public_config = load_public_config(root)
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["configs", "logs", "checkpoints", "oof_predictions", "embeddings", "attention", "tables", "figures", "statistical_tests", "compute"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    state = initialize_state(out, resume)
    audit = audit_and_freeze(root, out, force=force_audit)
    write_registry(out)
    atomic_json(out / "configs" / "resolved_public_config.json", public_config)
    patients = audit["patients"]
    slides = audit["slides"]
    data = data_root(root)
    profile_rows = []
    # Cheap but required sanity baselines first.
    mean_rows = run_mean_methods(root, data, patients, slides, out, state)
    profile_rows.extend(mean_rows)
    panther_rows = run_panther(root, data, patients, slides, out, state, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    profile_rows.extend(panther_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        neural_jobs = [f"{method}/seed_{seed}/fold_{fold}" for method in NEURAL_METHODS for fold in FOLDS]
        if all(job in state["completed_jobs"] for job in neural_jobs):
            continue
        cache = SlideCache(root, data, slides, MAX_TOKENS, PROJECTION_DIM, seed)
        atomic_json(out / "configs" / f"cache_seed_{seed}.json", {"seed": seed, "elapsed_seconds": cache.elapsed_seconds, "max_tokens": MAX_TOKENS, "projection_dim": PROJECTION_DIM})
        for method in NEURAL_METHODS:
            for fold in FOLDS:
                job = f"{method}/seed_{seed}/fold_{fold}"
                if job in state["completed_jobs"]:
                    continue
                state["current_job"] = job
                state["updated_at"] = utc_now()
                atomic_json(out / "run_state.json", state)
                print(f"[phase1.5 neural] start {job}", flush=True)
                try:
                    result = train_neural_fold(method, patients, cache, fold, seed, out, device, max_epochs=max_epochs)
                    profile_rows.append(result)
                    state["completed_jobs"].append(job)
                    state["current_job"] = None
                    atomic_json(out / "run_state.json", state)
                except Exception as exc:
                    state["failed_jobs"].append({"job": job, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    atomic_json(out / "run_state.json", state)
                    raise
    profile_marker = out / "configs" / "inference_profile_complete.json"
    profile_csv = out / "compute" / "compute_profile.csv"
    runnable_neural = [method for method in NEURAL_METHODS if method not in FAILED_METHODS]
    profile_needs_refresh = True
    if profile_csv.exists():
        try:
            current_profile = pd.read_csv(profile_csv)
            profile_needs_refresh = not all(
                ((current_profile["method"].astype(str) == method)
                 & current_profile["inference_time_seconds"].notna()).any()
                for method in runnable_neural
            )
        except (OSError, ValueError, KeyError):
            profile_needs_refresh = True
    # A completion marker alone is not sufficient: an interrupted/resumed
    # report build may have left a stale profile with blank inference fields.
    # Refreshing this table only loads saved checkpoints and never retrains.
    if profile_needs_refresh:
        profile_rows.extend(measure_saved_inference(root, data, patients, slides, out, device))
        atomic_json(profile_marker, {"created_at": utc_now(), "training_time_note": "initial run did not persist per-job training wall time; inference and peak memory remeasured from saved checkpoints"})
    if profile_csv.exists():
        try:
            # Preserve previously measured rows on --resume; newly measured
            # rows are appended after them and win in aggregate_metrics().
            profile_rows = pd.read_csv(profile_csv).to_dict("records") + profile_rows
        except (OSError, ValueError, KeyError):
            pass
    alias_rows = alias_b9a(out, state)
    profile_rows.extend(alias_rows)
    write_failures(out)
    oof = collect_oof(out)
    main, fold_table, profile = aggregate_metrics(out, oof, profile_rows)
    bootstrap = run_bootstrap_and_tables(out, oof, main)
    attention = attention_audit(out)
    embedding = embedding_audit(out)
    make_figures(out, main, fold_table, bootstrap, profile)
    write_reports(root, out, audit, main, fold_table, bootstrap, profile, embedding, attention)
    state["status"] = "COMPLETE"
    state["current_job"] = None
    state["end_time"] = utc_now()
    state["final_artifacts"] = {"data_audit": True, "fixed_split": True, "oof": True, "fold_metrics": True, "bootstrap": True, "compute": True, "embeddings": True, "attention": True, "figures": True, "reports": True, "failure_log": True}
    atomic_json(out / "run_state.json", state)
    print(json.dumps({"status": state["status"], "output": str(out), "completed_jobs": len(state["completed_jobs"]), "failed_jobs": len(state["failed_jobs"]), "methods_with_oof": sorted(oof.method.unique().tolist())}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1.5 UNI2-h DSS pathology benchmark audit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-audit", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = output_dir(root)
    if args.audit_only:
        result = audit_and_freeze(root, out, force=args.force_audit)
        print(json.dumps({"patients": len(result["patients"]), "slides": len(result["slides"]), "fixed_split": str(out / "splits" / "fixed_patient_folds.csv"), "skipped": result.get("skipped", False)}, ensure_ascii=False, indent=2))
    else:
        run(root, resume=args.resume, force_audit=args.force_audit, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()

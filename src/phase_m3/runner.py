from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from src.phase_m2.camr import CAMRModel
from src.phase_m2.common import (
    NEURAL_SEEDS,
    Timer,
    aggregate_uno,
    atomic_csv,
    atomic_json,
    count_parameters,
    cox_loss,
    gpu_memory_gb,
    harrell_c,
    set_seed,
    stable_int,
    uno_c_index,
)
from src.phase_m2.data import FoldArrays, RawInputs, load_raw_inputs, prepare_path_cache, preprocess_fold
from src.phase_m2.mcat import MCATModel
from src.phase_m2.runner import PathTokenProvider


MCAT_VARIANTS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
CAMR_VARIANTS = ("B0", "B1", "B2", "B3")
VARIANTS = MCAT_VARIANTS + CAMR_VARIANTS
LABELS = {
    "A0": "MCAT-Full",
    "A1": "MCAT-NoSemanticGroups-EqualWidth",
    "A2": "MCAT-NoCoAttention",
    "A3": "MCAT-ReverseCoAttention",
    "A4": "MCAT-BidirectionalCoAttention",
    "A5": "MCAT-NoPathTransformer",
    "A6": "MCAT-NoOmicTransformer",
    "A7": "MCAT-MeanPooling",
    "B0": "CAMR-Full",
    "B1": "CAMR-NoAdversarialAlignment",
    "B2": "CAMR-NoModalitySpecificBranch",
    "B3": "CAMR-NoCMFM-CommonConcat",
}
MECHANISMS = {
    "A0": "official semantic six-token grouping + omic-query path co-attention + dual transformers + gated pooling",
    "A1": "equal-width chunks of the same fold-local top-512 RNA/top-256 CNA arrays; six tokens retained",
    "A2": "no co-attention; independent path/omic streams followed by the locked transformers and pooling",
    "A3": "reverse pathology-query to omic-key/value attention",
    "A4": "bidirectional omic-to-path and path-to-omic attention with residual stream updates",
    "A5": "path transformer removed; other MCAT blocks retained",
    "A6": "omic transformer removed; other MCAT blocks retained",
    "A7": "gated attention pooling replaced by arithmetic mean pooling",
    "B0": "CAMR invariant/specific branches + adversarial alignment + CMFM",
    "B1": "CAMR adversarial alignment loss removed",
    "B2": "CAMR modality-specific branch and its reconstruction/orthogonality losses removed",
    "B3": "CAMR CMFM removed and common representations concatenated",
}


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _batch_indices(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    values = np.asarray(indices, dtype=int)
    if rng is not None:
        values = values[rng.permutation(len(values))]
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def _tensor_batch(arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, deterministic: bool, epoch: int, variant: str) -> tuple[torch.Tensor, ...]:
    path = torch.from_numpy(provider.batch(indices, epoch=epoch, deterministic=deterministic)).to(device=device, non_blocking=True)
    rna = torch.from_numpy(arrays.rna[indices]).to(device=device, non_blocking=True)
    cna = torch.from_numpy(arrays.cna[indices]).to(device=device, non_blocking=True)
    clinical = torch.from_numpy(arrays.clinical[indices]).to(device=device, non_blocking=True)
    if variant.startswith("A"):
        group_array = arrays.mcat_equal_groups if variant == "A1" else arrays.mcat_groups
        groups = torch.from_numpy(group_array[indices]).to(device=device, non_blocking=True)
        return path, rna, cna, clinical, groups
    return path, rna, cna, clinical


def _make_model(variant: str, clinical_dim: int, arrays: FoldArrays) -> torch.nn.Module:
    if variant.startswith("A"):
        widths = arrays.mcat_equal_widths if variant == "A1" else arrays.mcat_widths
        return MCATModel(clinical_dim, signature_widths=widths, variant=variant)
    return CAMRModel(clinical_dim, variant=variant)


def _forward_epoch(model: torch.nn.Module, arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, times: np.ndarray, events: np.ndarray, device: torch.device, batch_size: int, train: bool, rng: np.random.Generator | None, epoch: int, variant: str) -> tuple[float, torch.Tensor | np.ndarray, torch.Tensor | None]:
    model.train(train)
    risks: list[torch.Tensor] = []
    aux_values: list[torch.Tensor] = []
    batches = _batch_indices(indices, batch_size, rng if train else None)
    ordered = np.concatenate(batches) if batches else np.empty(0, dtype=int)
    for batch in batches:
        risk, aux = model(*_tensor_batch(arrays, provider, batch, device, not train, epoch, variant))
        risks.append(risk)
        aux_values.append(aux)
    all_risk = torch.cat(risks, dim=0)
    target_times = torch.tensor(times[ordered], dtype=torch.float32, device=device)
    target_events = torch.tensor(events[ordered], dtype=torch.float32, device=device)
    loss = cox_loss(target_times, target_events, all_risk)
    if aux_values:
        loss = loss + torch.stack(aux_values).mean()
    if train:
        return float(loss.detach().cpu()), all_risk, loss
    return float(loss.detach().cpu()), all_risk.detach().cpu().numpy(), None


def _predict(model: torch.nn.Module, arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, batch_size: int, variant: str) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batch_indices(indices, batch_size):
            values.append(model(*_tensor_batch(arrays, provider, batch, device, True, 0, variant))[0].detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _inner_split(raw: RawInputs, outer_fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    outer = np.flatnonzero(raw.outer_folds != outer_fold)
    strata = np.char.add(np.char.add(raw.cancer_type[outer].astype(str), "_"), raw.dss_event[outer].astype(str))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + outer_fold)
        tr, va = next(splitter.split(outer, strata))
        return outer[tr], outer[va]
    except ValueError:
        rng = np.random.default_rng(seed + outer_fold)
        order = rng.permutation(outer)
        n_val = max(1, int(round(0.15 * len(order))))
        return order[n_val:], order[:n_val]


def _fit_fold(variant: str, raw: RawInputs, arrays: FoldArrays, provider: PathTokenProvider, fold: int, seed: int, device: torch.device, max_epochs: int, patience: int, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    provider.train_seed = seed
    set_seed(seed + fold)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inner_train, inner_val = _inner_split(raw, fold, seed)
    model = _make_model(variant, arrays.clinical.shape[1], arrays).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_metric = -float("inf")
    best_epoch = 1
    wait = 0
    timer = Timer()
    rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, _, loss = _forward_epoch(model, arrays, provider, inner_train, raw.dss_time, raw.dss_event, device, batch_size, True, rng, epoch, variant)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_risk = _predict(model, arrays, provider, inner_val, device, batch_size, variant)
        val_c = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val_risk)
        score = val_c if np.isfinite(val_c) else -train_loss
        if score > best_metric + 1e-5:
            best_metric, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if wait >= patience:
            break
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(seed + fold)
    provider.train_seed = seed
    model = _make_model(variant, arrays.clinical.shape[1], arrays).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    outer_train = np.flatnonzero(raw.outer_folds != fold)
    refit_rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max(1, best_epoch) + 1):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = _forward_epoch(model, arrays, provider, outer_train, raw.dss_time, raw.dss_event, device, batch_size, True, refit_rng, epoch, variant)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    test_idx = np.flatnonzero(raw.outer_folds == fold)
    prediction = _predict(model, arrays, provider, test_idx, device, batch_size, variant)
    profile = {"method": variant, "seed": seed, "outer_fold": fold, "params": parameters, "best_epoch": best_epoch, "inner_best_harrell_c": float(best_metric), "training_seconds": timer.seconds, "peak_gpu_memory_gb": gpu_memory_gb(device), "outer_refit": "reinitialized_same_seed_full_outer_train", "epochwise_sampling": True, "status": "completed"}
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return prediction, profile


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    data_value = os.environ.get(args.data_root_env)
    if not data_value:
        raise RuntimeError(f"{args.data_root_env} is not set")
    data_root = Path(data_value).resolve()
    runtime = Path(args.runtime_root).resolve()
    reports = Path(args.reports_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    raw = load_raw_inputs(data_root, repo_root)
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    train_pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    cache_meta = prepare_path_cache(raw, runtime, k=1024, pool_k=2048)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    requested = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown M3 variants: {unknown}")
    seeds = [int(x) for x in args.seeds.split(",")]
    state_path = runtime / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": [], "failed": []}
    cells = runtime / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []
    seed_rows: list[dict] = []
    profiles: list[dict] = []
    started = time.perf_counter()
    arrays_cache: dict[int, FoldArrays] = {}
    for variant in requested:
        for seed in seeds:
            oof = np.full(len(raw.patient_ids), np.nan, dtype=float)
            seed_folds: list[dict] = []
            for fold in range(1, 6):
                cell = f"{variant}|{seed}|{fold}"
                stem = cell.replace("|", "_")
                test_idx = np.flatnonzero(raw.outer_folds == fold)
                if cell in state.get("completed", []) and (cells / f"{stem}.json").exists() and (cells / f"{stem}.npy").exists():
                    payload = json.loads((cells / f"{stem}.json").read_text(encoding="utf-8"))
                    row, profile = payload["row"], payload["profile"]
                    pred = np.load(cells / f"{stem}.npy")
                else:
                    try:
                        if fold not in arrays_cache:
                            arrays_cache[fold] = preprocess_fold(raw, train_pool, fold)
                        arrays = arrays_cache[fold]
                        provider = PathTokenProvider(train_pool=train_pool, deterministic=deterministic, train_seed=seed, k=1024)
                        pred, profile = _fit_fold(variant, raw, arrays, provider, fold, seed, device, args.max_epochs, args.patience, args.batch_size)
                        train_idx = np.flatnonzero(raw.outer_folds != fold)
                        uno, num, den = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], pred, raw.dss_time[train_idx], raw.dss_event[train_idx])
                        row = {"method": variant, "label": LABELS[variant], "seed": seed, "outer_fold": fold, "n_test": len(test_idx), "events_test": int(raw.dss_event[test_idx].sum()), "harrell_c": harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred), "uno_c": uno, "uno_numerator": num, "uno_denominator": den, "params": profile["params"], "status": "completed"}
                        np.save(cells / f"{stem}.npy", pred)
                        atomic_json(cells / f"{stem}.json", {"row": row, "profile": profile})
                        state.setdefault("completed", []).append(cell)
                        atomic_json(state_path, state)
                    except Exception as exc:
                        state.setdefault("failed", []).append({"cell": cell, "error": repr(exc)})
                        atomic_json(state_path, state)
                        raise
                fold_rows.append(row)
                seed_folds.append(row)
                profiles.append(profile)
                oof[test_idx] = pred
            seed_rows.append({"method": variant, "label": LABELS[variant], "seed": seed, "n_patients": len(raw.patient_ids), "dss_events": int(raw.dss_event.sum()), "harrell_c": harrell_c(raw.dss_time, raw.dss_event, oof), "uno_c": aggregate_uno(seed_folds), "status": "completed"})
    atomic_csv(reports / "fold_metrics.csv", fold_rows)
    atomic_csv(reports / "seed_metrics.csv", seed_rows)
    atomic_csv(reports / "compute_profile.csv", profiles)
    atomic_json(reports / "run_manifest.json", {"phase": "M3", "status": "worker_completed", "execution_commit": _git_revision(repo_root), "variants": requested, "seeds": seeds, "cells_completed": len(fold_rows), "elapsed_seconds": time.perf_counter() - started, "device": str(device), "cache": cache_meta, "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv", "no_preprocessing_change": True})
    print(json.dumps({"status": "worker_completed", "variants": requested, "seeds": seeds, "cells": len(fold_rows), "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2))
    return {"status": "worker_completed", "cells": len(fold_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase M3 MCAT/CAMR mechanism ablation runner")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root-env", default="DIFFUSION_DATA_ROOT")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

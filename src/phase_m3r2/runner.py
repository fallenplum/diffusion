from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from src.phase_m2.common import (
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
from src.phase_m3r.runner import PathTokenProvider

from .models import build_model


VARIANTS = ("S0", "S1", "S2", "S3", "S4", "S5")
LABELS = {
    "S0": "ClinicalOnly",
    "S1": "SimpleCAMR-FullReg",
    "S2": "SimpleCAMR-NoAdv",
    "S3": "SimpleCAMR-CommonOnly",
    "S4": "RawEncodedConcat",
    "S5": "SimpleCAMR-NoClinical",
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


def _tensor_batch(arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, deterministic: bool, epoch: int) -> tuple[torch.Tensor, ...]:
    path = torch.from_numpy(provider.batch(indices, epoch=epoch, deterministic=deterministic)).to(device=device, non_blocking=True)
    rna = torch.from_numpy(arrays.rna[indices]).to(device=device, non_blocking=True)
    cna = torch.from_numpy(arrays.cna[indices]).to(device=device, non_blocking=True)
    clinical = torch.from_numpy(arrays.clinical[indices]).to(device=device, non_blocking=True)
    return path, rna, cna, clinical


def _variant_seed(variant: str, seed: int, fold: int, kind: str) -> int:
    return 100000 + stable_int(f"m3r2:{variant}:{kind}:{seed}:{fold}", modulo=2_000_000_000)


def _make_model(variant: str, clinical_dim: int, seed: int, fold: int) -> torch.nn.Module:
    return build_model(
        variant,
        clinical_dim,
        variant_seed=_variant_seed(variant, seed, fold, "modules"),
        concat_seed=_variant_seed(variant, seed, fold, "fusion"),
    )


def _forward_epoch(model: torch.nn.Module, arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, times: np.ndarray, events: np.ndarray, device: torch.device, batch_size: int, train: bool, rng: np.random.Generator | None, epoch: int) -> tuple[float, np.ndarray | None, torch.Tensor | None]:
    model.train(train)
    risks: list[torch.Tensor] = []
    aux_values: list[torch.Tensor] = []
    batches = _batch_indices(indices, batch_size, rng if train else None)
    ordered = np.concatenate(batches) if batches else np.empty(0, dtype=int)
    for batch in batches:
        risk, aux = model(*_tensor_batch(arrays, provider, batch, device, not train, epoch))
        risks.append(risk)
        aux_values.append(aux)
    all_risk = torch.cat(risks, dim=0)
    target_times = torch.tensor(times[ordered], dtype=torch.float32, device=device)
    target_events = torch.tensor(events[ordered], dtype=torch.float32, device=device)
    loss = cox_loss(target_times, target_events, all_risk)
    if aux_values:
        loss = loss + torch.stack(aux_values).mean()
    if train:
        return float(loss.detach().cpu()), None, loss
    return float(loss.detach().cpu()), all_risk.detach().cpu().numpy(), None


def _predict(model: torch.nn.Module, arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batch_indices(indices, batch_size):
            values.append(model(*_tensor_batch(arrays, provider, batch, device, True, 0))[0].detach().cpu().numpy())
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
    model = _make_model(variant, arrays.clinical.shape[1], seed, fold).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_metric = -float("inf")
    best_epoch = 1
    wait = 0
    timer = Timer()
    rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, _, loss = _forward_epoch(model, arrays, provider, inner_train, raw.dss_time, raw.dss_event, device, batch_size, True, rng, epoch)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_risk = _predict(model, arrays, provider, inner_val, device, batch_size)
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
    # The outer refit starts from the same seed and trains only on outer train.
    set_seed(seed + fold)
    model = _make_model(variant, arrays.clinical.shape[1], seed, fold).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    outer_train = np.flatnonzero(raw.outer_folds != fold)
    refit_rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max(1, best_epoch) + 1):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = _forward_epoch(model, arrays, provider, outer_train, raw.dss_time, raw.dss_event, device, batch_size, True, refit_rng, epoch)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    test_idx = np.flatnonzero(raw.outer_folds == fold)
    prediction = _predict(model, arrays, provider, test_idx, device, batch_size)
    profile = {
        "method": variant,
        "seed": seed,
        "outer_fold": fold,
        "params": parameters,
        "best_epoch": best_epoch,
        "inner_best_harrell_c": float(best_metric),
        "training_seconds": timer.seconds,
        "peak_gpu_memory_gb": gpu_memory_gb(device),
        "outer_refit": "reinitialized_same_seed_full_outer_train",
        "epochwise_sampling": True,
        "status": "completed",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return prediction, profile


def _initialization_audit(clinical_dim: int, seed: int = 20260907, fold: int = 1) -> dict[str, Any]:
    models = {}
    for variant in ("S1", "S2", "S3", "S4"):
        set_seed(seed + fold)
        models[variant] = _make_model(variant, clinical_dim, seed, fold)
    reference = models["S1"].state_dict()
    results: dict[str, Any] = {}
    for variant in ("S2", "S3", "S4"):
        state = models[variant].state_dict()
        mismatches = []
        for key in sorted(reference):
            if not key.startswith("backbone."):
                continue
            if key not in state or not torch.equal(reference[key].cpu(), state[key].cpu()):
                mismatches.append(key)
        results[variant] = {"shared_backbone_byte_identical": not mismatches, "mismatches": mismatches}
    return {
        "status": "PASS" if all(item["shared_backbone_byte_identical"] for item in results.values()) else "FAIL",
        "seed": seed,
        "fold": fold,
        "variants_checked": ["S1", "S2", "S3", "S4"],
        "comparisons_to_S1": results,
        "shared_scope": "backbone.pathology_encoder, backbone.omics_encoder, backbone.clinical_encoder",
        "variant_only_seed_policy": "stable hash of variant, seed, fold and module kind, isolated with torch.random.fork_rng",
        "s1_to_s4_parameter_counts": {key: count_parameters(value) for key, value in models.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    data_value = os.environ.get(args.data_root_env)
    if not data_value:
        raise RuntimeError(f"{args.data_root_env} is not set")
    data_root = Path(data_value).resolve()
    runtime = Path(args.runtime_root).resolve()
    reports = Path(args.reports_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    deterministic_audit = {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "known_cuda_warnings": [
            "memory-efficient attention backward may be nondeterministic",
            "adaptive_avg_pool2d_backward_cuda may be nondeterministic",
        ],
    }
    raw = load_raw_inputs(data_root, repo_root)
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    train_pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    cache_meta = prepare_path_cache(raw, runtime, k=1024, pool_k=2048)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    audit_arrays = preprocess_fold(raw, train_pool, 1)
    initialization_audit = _initialization_audit(audit_arrays.clinical.shape[1])
    atomic_json(reports / "initialization_audit.json", {**initialization_audit, "determinism": deterministic_audit, "execution_commit": _git_revision(repo_root)})
    if initialization_audit["status"] != "PASS":
        raise RuntimeError("M3R2 shared-backbone initialization audit failed")
    requested = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown M3R2 variants: {unknown}")
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) != 1:
        raise ValueError("each worker must receive exactly one seed")
    state_path = runtime / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": [], "failed": []}
    cells = runtime / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    started = time.perf_counter()
    arrays_cache: dict[int, FoldArrays] = {}
    seed = seeds[0]
    for variant in requested:
        oof = np.full(len(raw.patient_ids), np.nan, dtype=float)
        seed_folds: list[dict[str, Any]] = []
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
                    row = {
                        "method": variant,
                        "label": LABELS[variant],
                        "seed": seed,
                        "outer_fold": fold,
                        "n_test": len(test_idx),
                        "events_test": int(raw.dss_event[test_idx].sum()),
                        "harrell_c": harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred),
                        "uno_c": uno,
                        "uno_numerator": num,
                        "uno_denominator": den,
                        "params": profile["params"],
                        "status": "completed",
                    }
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
        h = np.asarray([float(item["harrell_c"]) for item in seed_folds], dtype=float)
        u = np.asarray([float(item["uno_c"]) for item in seed_folds], dtype=float)
        seed_rows.append({
            "method": variant,
            "label": LABELS[variant],
            "seed": seed,
            "n_patients": len(raw.patient_ids),
            "dss_events": int(raw.dss_event.sum()),
            "fold_mean_harrell": float(np.nanmean(h)),
            "fold_std_harrell": float(np.nanstd(h, ddof=1)),
            "worst_fold_harrell": float(np.nanmin(h)),
            "fold_mean_uno": float(np.nanmean(u)),
            "fold_std_uno": float(np.nanstd(u, ddof=1)),
            "worst_fold_uno": float(np.nanmin(u)),
            "global_oof_harrell_diagnostic": harrell_c(raw.dss_time, raw.dss_event, oof),
            "global_oof_uno_diagnostic": aggregate_uno(seed_folds),
            "status": "completed",
        })
    atomic_csv(reports / "fold_metrics.csv", fold_rows)
    atomic_csv(reports / "seed_metrics.csv", seed_rows)
    atomic_csv(reports / "compute_profile.csv", profiles)
    atomic_json(reports / "run_manifest.json", {
        "phase": "M3R2",
        "status": "worker_completed",
        "execution_commit": _git_revision(repo_root),
        "variants": requested,
        "seeds": seeds,
        "cells_completed": len(fold_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "cache": cache_meta,
        "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv",
        "no_preprocessing_change": True,
        "global_oof_status": "diagnostic_only",
        "primary_metric": "fold-level Harrell C; aggregate mean/std over seed×fold cells",
        "secondary_metric": "fold-level Uno C; aggregate mean/std over seed×fold cells",
        "n_patients": int(len(raw.patient_ids)),
        "dss_events": int(raw.dss_event.sum()),
        "pathology_K": 1024,
        "determinism": deterministic_audit,
        "initialization_audit_status": initialization_audit["status"],
    })
    print(json.dumps({"status": "worker_completed", "variants": requested, "seeds": seeds, "cells": len(fold_rows), "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2))
    return {"status": "worker_completed", "cells": len(fold_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase M3R2 simple-fusion and clinical control runner")
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

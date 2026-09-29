from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from .camr import CAMRModel
from .common import (
    BASELINE_SEED,
    NEURAL_SEEDS,
    Timer,
    aggregate_uno,
    atomic_csv,
    atomic_json,
    count_parameters,
    cox_loss,
    gpu_memory_gb,
    harrell_c,
    run_functional_checks,
    set_seed,
    uno_c_index,
)
from .data import FoldArrays, RawInputs, load_raw_inputs, prepare_path_cache, preprocess_fold
from .early_fusion import fit_ridge_cox
from .mgct import MGCTModel
from .mmp import MMPModel
from .motcat import MOTCatModel
from .transsurv import TransSurvModel


MODEL_ORDER = ("B0", "B1", "B2", "B3", "B4", "B5")
MODEL_LABELS = {
    "B0": "EarlyFusion / Concat + Cox",
    "B1": "TransSurv-style Transformer",
    "B2": "MGCT",
    "B3": "CAMR",
    "B4": "MMP",
    "B5": "MOTCat / MCAT-style co-attention",
}
MECHANISMS = {
    "B0": "Concat -> no explicit interaction",
    "B1": "TransSurv -> transformer cross-modal interaction",
    "B2": "MGCT -> mutual-guided cross-modal interaction",
    "B3": "CAMR -> shared/specific + alignment",
    "B4": "MMP -> multimodal prototypes",
    "B5": "MOTCat/MCAT -> co-attention / transport-guided interaction",
}
REPRODUCTION_SOURCES = {
    "B0": "fixed protocol baseline",
    "B1": "TransSurv-style architecture-faithful adaptation",
    "B2": "https://github.com/lmxmercy/MGCT",
    "B3": "https://github.com/wxq-ustc/CAMR",
    "B4": "https://github.com/mahmoodlab/MMP",
    "B5": "https://github.com/mahmoodlab/MCAT",
}


@dataclass
class PathTokenProvider:
    train_random: np.ndarray
    deterministic: np.ndarray
    outer_folds: np.ndarray
    fold: int

    def batch(self, indices: np.ndarray, deterministic: bool = False) -> np.ndarray:
        source = self.deterministic if deterministic else self.train_random
        values = np.asarray(source[indices], dtype=np.float16)
        return values

    def mean(self, indices: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.batch(indices, deterministic=deterministic).astype(np.float32).mean(axis=1)


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _batch_indices(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if rng is not None:
        indices = indices[rng.permutation(len(indices))]
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _tensor_batch(arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, deterministic: bool) -> tuple[torch.Tensor, ...]:
    path = torch.from_numpy(provider.batch(indices, deterministic=deterministic)).to(device=device, non_blocking=True)
    rna = torch.from_numpy(arrays.rna[indices]).to(device=device, non_blocking=True)
    cna = torch.from_numpy(arrays.cna[indices]).to(device=device, non_blocking=True)
    clinical = torch.from_numpy(arrays.clinical[indices]).to(device=device, non_blocking=True)
    return path, rna, cna, clinical


def _make_model(name: str, clinical_dim: int) -> torch.nn.Module:
    if name == "B1":
        return TransSurvModel(clinical_dim)
    if name == "B2":
        return MGCTModel(clinical_dim)
    if name == "B3":
        return CAMRModel(clinical_dim)
    if name == "B4":
        return MMPModel(clinical_dim)
    if name == "B5":
        return MOTCatModel(clinical_dim)
    raise ValueError(name)


def _forward_epoch(
    model: torch.nn.Module,
    arrays: FoldArrays,
    provider: PathTokenProvider,
    indices: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    device: torch.device,
    batch_size: int,
    train: bool,
    rng: np.random.Generator | None,
) -> tuple[float, torch.Tensor | np.ndarray, torch.Tensor | None]:
    model.train(train)
    risks: list[torch.Tensor] = []
    aux_values: list[torch.Tensor] = []
    batches = _batch_indices(indices, batch_size, rng if train else None)
    ordered_indices = np.concatenate(batches) if batches else np.empty(0, dtype=int)
    for batch in batches:
        inputs = _tensor_batch(arrays, provider, batch, device, deterministic=False)
        risk, aux = model(*inputs)
        risks.append(risk)
        aux_values.append(aux)
    all_risk = torch.cat(risks, dim=0)
    target_times = torch.tensor(times[ordered_indices], dtype=torch.float32, device=device)
    target_events = torch.tensor(events[ordered_indices], dtype=torch.float32, device=device)
    loss = cox_loss(target_times, target_events, all_risk)
    if aux_values:
        loss = loss + torch.stack(aux_values).mean()
    if train:
        return float(loss.detach().cpu()), all_risk, loss
    return float(loss.detach().cpu()), all_risk.detach().cpu().numpy(), None


def _predict(
    model: torch.nn.Module,
    arrays: FoldArrays,
    provider: PathTokenProvider,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    deterministic: bool,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batch_indices(indices, batch_size):
            inputs = _tensor_batch(arrays, provider, batch, device, deterministic=deterministic)
            values.append(model(*inputs)[0].detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _inner_split(raw: RawInputs, outer_fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    outer = np.flatnonzero(raw.outer_folds != outer_fold)
    strata = np.char.add(np.char.add(raw.cancer_type[outer].astype(str), "_"), raw.dss_event[outer].astype(str))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + outer_fold)
        tr_local, va_local = next(splitter.split(outer, strata))
        return outer[tr_local], outer[va_local]
    except ValueError:
        rng = np.random.default_rng(seed + outer_fold)
        order = rng.permutation(outer)
        n_val = max(1, int(round(0.15 * len(order))))
        return order[n_val:], order[:n_val]


def _fit_neural_fold(
    name: str,
    raw: RawInputs,
    arrays: FoldArrays,
    provider: PathTokenProvider,
    fold: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed + fold)
    inner_train, inner_val = _inner_split(raw, fold, seed)
    model = _make_model(name, arrays.clinical.shape[1]).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -float("inf")
    best_epoch = 1
    wait = 0
    history: list[dict[str, float]] = []
    timer = Timer()
    rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, _train_risk, loss = _forward_epoch(model, arrays, provider, inner_train, raw.dss_time, raw.dss_event, device, batch_size, True, rng)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_risk = _predict(model, arrays, provider, inner_val, device, batch_size, deterministic=False)
        val_c = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val_risk)
        score = val_c if np.isfinite(val_c) else -train_loss
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_harrell_c": float(val_c)})
        if score > best_metric + 1e-5:
            best_metric = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    # Outer refit uses the selected epoch count and never observes the outer test fold.
    model.load_state_dict(best_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    outer_train = np.flatnonzero(raw.outer_folds != fold)
    for _ in range(max(1, best_epoch)):
        optimizer.zero_grad(set_to_none=True)
        _, _risk, loss = _forward_epoch(model, arrays, provider, outer_train, raw.dss_time, raw.dss_event, device, batch_size, True, rng)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    test_idx = np.flatnonzero(raw.outer_folds == fold)
    prediction = _predict(model, arrays, provider, test_idx, device, batch_size, deterministic=True)
    profile = {
        "method": name,
        "seed": seed,
        "outer_fold": fold,
        "params": parameters,
        "best_epoch": best_epoch,
        "inner_best_harrell_c": float(best_metric),
        "training_seconds": timer.seconds,
        "peak_gpu_memory_gb": gpu_memory_gb(device),
        "status": "completed",
    }
    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return prediction, profile


def _b0_fold(raw: RawInputs, arrays: FoldArrays, provider: PathTokenProvider, fold: int, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    timer = Timer()
    train = np.flatnonzero(raw.outer_folds != fold)
    test = np.flatnonzero(raw.outer_folds == fold)
    train_path = provider.mean(train, deterministic=False)
    test_path = provider.mean(test, deterministic=True)
    train_x = np.concatenate([train_path, arrays.rna[train], arrays.cna[train], arrays.clinical[train]], axis=1)
    test_x = np.concatenate([test_path, arrays.rna[test], arrays.cna[test], arrays.clinical[test]], axis=1)
    beta, loss = fit_ridge_cox(train_x, raw.dss_time[train], raw.dss_event[train], alpha=1e-2, device=device)
    profile = {
        "method": "B0",
        "seed": BASELINE_SEED,
        "outer_fold": fold,
        "params": int(len(beta)),
        "best_epoch": 0,
        "inner_best_harrell_c": float("nan"),
        "training_seconds": timer.seconds,
        "peak_gpu_memory_gb": gpu_memory_gb(device),
        "cox_train_loss": loss,
        "status": "completed",
    }
    return test_x @ beta, profile


def _write_results(reports: Path, raw: RawInputs, fold_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], profiles: list[dict[str, Any]], functional: dict[str, Any], statuses: list[dict[str, Any]], elapsed: float) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    fold_path = reports / "fold_metrics.csv"
    seed_path = reports / "seed_metrics.csv"
    compute_path = reports / "compute_profile.csv"
    status_path = reports / "reproduction_status.csv"
    main_rows: list[dict[str, Any]] = []
    for method in MODEL_ORDER:
        method_seeds = [row for row in seed_rows if row["method"] == method and row["status"] == "completed"]
        method_folds = [row for row in fold_rows if row["method"] == method and row["status"] == "completed"]
        if not method_seeds:
            main_rows.append({"method": method, "label": MODEL_LABELS[method], "status": "failed"})
            continue
        harrell_seed = np.asarray([float(row["harrell_c"]) for row in method_seeds], dtype=float)
        uno_seed = np.asarray([float(row["uno_c"]) for row in method_seeds], dtype=float)
        harrell_fold = np.asarray([float(row["harrell_c"]) for row in method_folds], dtype=float)
        uno_fold = np.asarray([float(row["uno_c"]) for row in method_folds], dtype=float)
        params = [float(row["params"]) for row in profiles if row["method"] == method and row["status"] == "completed"]
        memories = [float(row["peak_gpu_memory_gb"]) for row in profiles if row["method"] == method and row["status"] == "completed"]
        seconds = [float(row["training_seconds"]) for row in profiles if row["method"] == method and row["status"] == "completed"]
        main_rows.append(
            {
                "method": method,
                "label": MODEL_LABELS[method],
                "mechanism": MECHANISMS[method],
                "harrell_c_seed_mean": float(np.nanmean(harrell_seed)),
                "harrell_c_seed_std": float(np.nanstd(harrell_seed, ddof=1)) if len(harrell_seed) > 1 else 0.0,
                "uno_c_seed_mean": float(np.nanmean(uno_seed)),
                "uno_c_seed_std": float(np.nanstd(uno_seed, ddof=1)) if len(uno_seed) > 1 else 0.0,
                "harrell_c_fold_mean": float(np.nanmean(harrell_fold)),
                "harrell_c_fold_std": float(np.nanstd(harrell_fold, ddof=1)) if len(harrell_fold) > 1 else 0.0,
                "uno_c_fold_mean": float(np.nanmean(uno_fold)),
                "uno_c_fold_std": float(np.nanstd(uno_fold, ddof=1)) if len(uno_fold) > 1 else 0.0,
                "harrell_c_worst_fold": float(np.nanmin(harrell_fold)),
                "uno_c_worst_fold": float(np.nanmin(uno_fold)),
                "params": int(round(np.nanmedian(params))) if params else 0,
                "peak_gpu_memory_gb": float(np.nanmax(memories)) if memories else 0.0,
                "training_seconds": float(np.nansum(seconds)) if seconds else 0.0,
                "reproduction_status": next((row["status"] for row in statuses if row["method"] == method), "unknown"),
                "status": "completed",
            }
        )
    atomic_csv(reports / "main_metrics.csv", main_rows)
    atomic_csv(fold_path, fold_rows)
    atomic_csv(seed_path, seed_rows)
    atomic_csv(compute_path, profiles)
    atomic_csv(status_path, statuses)
    protocol = {
        "phase": "M2",
        "endpoint": "DSS",
        "n_patients": int(len(raw.patient_ids)),
        "dss_events": int(raw.dss_event.sum()),
        "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv",
        "folds": 5,
        "neural_seeds": list(NEURAL_SEEDS),
        "pathology": {"source": "TCGA COAD/READ UNI2-h H5", "K": 1024, "train_sampling": "random", "test_sampling": "deterministic"},
        "omics": {"RNA": "outer-train imputation/standardization/top-512 variance", "CNA": "outer-train imputation/standardization/top-256 variance"},
        "clinical": "outer-train median imputation and categorical encoding",
        "objective": "negative Breslow Cox partial likelihood with T_j >= T_i",
        "functional_checks": functional,
        "no_patient_level_outputs_in_reports": True,
        "data_root_env": "DIFFUSION_DATA_ROOT",
    }
    atomic_json(reports / "protocol.json", protocol)
    completed = {row["method"] for row in main_rows if row.get("status") == "completed"}
    complete = len(completed) >= 5 and {"B2", "B3", "B4"}.issubset(completed) and all(row.get("status") == "completed" for row in statuses)
    status_text = "Phase M2 COMPLETE" if complete else "Phase M2 BLOCKED"
    sorted_rows = sorted([row for row in main_rows if row.get("status") == "completed"], key=lambda row: row.get("harrell_c_seed_mean", -float("inf")), reverse=True)
    lines = [f"# Phase M2 SOTA Baselines", "", f"## Decision", "", f"**{status_text}**", "", f"- Cohort: {len(raw.patient_ids)} patients, {int(raw.dss_event.sum())} DSS events.", "- The committed Phase M1 fixed patient split was reused; no split was regenerated.", "- Reports contain aggregate fold/seed/compute summaries only; patient IDs, H5/NPY/NPZ, checkpoints and patient-level predictions are excluded.", "", "## Baseline ranking", "", "| Rank | Method | Harrell C seed mean ± std | Uno C seed mean ± std | Fold mean ± std | Worst fold | Params | GPU GB | Time s | Reproduction status |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for rank, row in enumerate(sorted_rows, 1):
        lines.append(f"| {rank} | {row['label']} | {row['harrell_c_seed_mean']:.4f} ± {row['harrell_c_seed_std']:.4f} | {row['uno_c_seed_mean']:.4f} ± {row['uno_c_seed_std']:.4f} | {row['harrell_c_fold_mean']:.4f} ± {row['harrell_c_fold_std']:.4f} | {row['harrell_c_worst_fold']:.4f} | {row['params']} | {row['peak_gpu_memory_gb']:.2f} | {row['training_seconds']:.1f} | {row['reproduction_status']} |")
    lines.extend(["", "## Core mechanisms", "", "- B0: early concatenation with a Cox head.", "- B1: Transformer self-attention across pathology, omics and clinical tokens.", "- B2: bidirectional mutual-guided cross-attention between pathology and omics.", "- B3: shared/specific representations with alignment and orthogonality penalties.", "- B4: learned pathology and omics prototype tokens with prototype-level cross-attention.", "- B5: co-attention with entropic transport coupling between pathology and omics tokens.", "", "## Failure signatures for the next phase", "", "- Compare whether explicit cross-modal interactions improve both Harrell and Uno C consistently across folds and seeds.", "- Inspect methods whose worst fold collapses while seed means remain competitive; this indicates fold sensitivity rather than a stable mechanism gain.", "- Treat any `FAILED_TO_REPRODUCE` row as a boundary condition and do not describe it as a successful official reproduction."])
    (reports / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "phase": "M2",
        "status": status_text,
        "n_patients": int(len(raw.patient_ids)),
        "dss_events": int(raw.dss_event.sum()),
        "methods_completed": sorted(completed),
        "elapsed_seconds": elapsed,
        "platform": platform.platform(),
        "functional_checks": functional,
        "data_root_env": "DIFFUSION_DATA_ROOT",
        "git_revision": _git_revision(reports.parents[1]),
    }
    atomic_json(reports / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    env_name = args.data_root_env
    data_root_value = os.environ.get(env_name)
    if not data_root_value:
        raise RuntimeError(f"{env_name} is not set")
    data_root = Path(data_root_value).resolve()
    runtime_root = Path(args.runtime_root).resolve() if args.runtime_root else repo_root / "runtime" / "phase_m2"
    reports = Path(args.reports_root).resolve() if args.reports_root else repo_root / "reports" / "phase_m2_sota_baselines"
    runtime_root.mkdir(parents=True, exist_ok=True)
    functional = run_functional_checks()
    atomic_json(runtime_root / "functional_checks.json", functional)
    if not functional["status"]:
        raise RuntimeError("Cox functional gates failed")
    raw = load_raw_inputs(data_root, repo_root)
    atomic_json(runtime_root / "data_audit.json", raw.audit)
    cache_meta = prepare_path_cache(raw, runtime_root, k=1024)
    if args.prepare_cache:
        print(json.dumps({"status": "cache_prepared", "data_root_env_set": True, "cache": cache_meta}, ensure_ascii=False))
        return {"status": "cache_prepared"}
    deterministic = np.load(runtime_root / "path_tokens_deterministic.npy", mmap_mode="r")
    train_random = np.load(runtime_root / "path_tokens_train_random.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    neural_seeds = list(NEURAL_SEEDS if args.seeds == "all" else [int(x) for x in args.seeds.split(",")])
    state_path = runtime_root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": [], "failed": []}
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    started = time.perf_counter()
    for method in requested:
        method_seeds = [BASELINE_SEED] if method == "B0" else neural_seeds
        method_failed = False
        for seed in method_seeds:
            seed_oof = np.full(len(raw.patient_ids), np.nan, dtype=float)
            seed_fold_rows: list[dict[str, Any]] = []
            seed_failed = False
            for fold in range(1, 6):
                cell = f"{method}|{seed}|{fold}"
                if cell in state.get("completed", []):
                    # A completed cell is not recomputed; a fresh aggregate can be
                    # generated by rerunning from a clean runtime if needed.
                    continue
                try:
                    arrays = preprocess_fold(raw, train_random, fold)
                    provider = PathTokenProvider(train_random=train_random, deterministic=deterministic, outer_folds=raw.outer_folds, fold=fold)
                    if method == "B0":
                        prediction, profile = _b0_fold(raw, arrays, provider, fold, device)
                    else:
                        prediction, profile = _fit_neural_fold(method, raw, arrays, provider, fold, seed, device, args.max_epochs, args.patience, args.batch_size)
                    test_idx = np.flatnonzero(raw.outer_folds == fold)
                    uno, numerator, denominator = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], prediction, raw.dss_time[np.flatnonzero(raw.outer_folds != fold)], raw.dss_event[np.flatnonzero(raw.outer_folds != fold)])
                    row = {"method": method, "seed": seed, "outer_fold": fold, "n_test": len(test_idx), "events_test": int(raw.dss_event[test_idx].sum()), "harrell_c": harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], prediction), "uno_c": uno, "uno_numerator": numerator, "uno_denominator": denominator, "params": profile["params"], "status": "completed"}
                    fold_rows.append(row)
                    seed_fold_rows.append(row)
                    seed_oof[test_idx] = prediction
                    profiles.append(profile)
                    state.setdefault("completed", []).append(cell)
                    atomic_json(state_path, state)
                except Exception as exc:
                    method_failed = True
                    seed_failed = True
                    state.setdefault("failed", []).append({"cell": cell, "error": repr(exc)})
                    atomic_json(state_path, state)
                    fold_rows.append({"method": method, "seed": seed, "outer_fold": fold, "status": "failed", "error": repr(exc)})
            if not seed_failed and len(seed_fold_rows) == 5:
                # Predictions are only retained in memory; no patient-level file is written.
                seed_rows.append({"method": method, "seed": seed, "n_patients": len(raw.patient_ids), "dss_events": int(raw.dss_event.sum()), "harrell_c": harrell_c(raw.dss_time, raw.dss_event, seed_oof), "uno_c": aggregate_uno(seed_fold_rows), "status": "completed"})
            else:
                seed_rows.append({"method": method, "seed": seed, "status": "failed"})
        statuses.append({"method": method, "label": MODEL_LABELS[method], "mechanism": MECHANISMS[method], "source": REPRODUCTION_SOURCES[method], "status": "FAILED_TO_REPRODUCE" if method_failed else "ADAPTED_ARCHITECTURE_FAITHFUL"})
    elapsed = time.perf_counter() - started
    _write_results(reports, raw, fold_rows, seed_rows, profiles, functional, statuses, elapsed)
    result = {"status": "completed", "reports": str(reports), "device": str(device), "elapsed_seconds": elapsed}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase M2 unified multimodal survival baselines")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root-env", default="DIFFUSION_DATA_ROOT")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--seeds", default="all", help="all or comma-separated neural seeds")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

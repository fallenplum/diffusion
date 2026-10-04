from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn

from src.phase_m2.common import (
    NEURAL_SEEDS,
    censoring_survival,
    harrell_c,
    lookup_censor_survival,
    set_seed,
    uno_c_index,
    atomic_csv,
    atomic_json,
)
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.po_phase0.pathway_probe import (
    PathwayEncoder,
    PathwayCNA,
    SignaturePO,
    make,
    split,
    inps,
)
from src.po_phase0.runner import cox_loss_grouped


METHODS = ("Pathway-RNA", "Pathway-RNA+CNA", "Signature-RNA+CNA")
SEEDS = tuple(int(x) for x in NEURAL_SEEDS)
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0)


def cox_loss_fast(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    """Vectorized Breslow loss for the mechanism controls.

    ``cox_loss_grouped`` is retained for the official pathway replay so that
    its exact implementation is unchanged.  The controls use this algebraic
    equivalent because the event loop dominates hundreds of small optimizer
    steps and needlessly serializes GPU kernels.
    """
    event_times = torch.unique(times[events > 0], sorted=True)
    if event_times.numel() == 0:
        return risks.sum() * 0.0
    at_risk = times.unsqueeze(0) >= event_times.unsqueeze(1)
    event_at_time = (events.unsqueeze(0) > 0) & (times.unsqueeze(0) == event_times.unsqueeze(1))
    masked_risk = risks.unsqueeze(0).expand(event_times.shape[0], -1).masked_fill(~at_risk, -torch.inf)
    log_risk = torch.logsumexp(masked_risk, dim=1)
    event_count = event_at_time.sum(dim=1).to(risks.dtype)
    event_sum = (event_at_time.to(risks.dtype) * risks.unsqueeze(0)).sum(dim=1)
    return (event_count * log_risk - event_sum).sum() / (events > 0).sum().to(risks.dtype)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nparams(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_pathway_cell(
    name: str,
    arrays: Any,
    raw: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    seed: int,
    device: torch.device,
    selected_epoch: int | None = None,
    select_epoch: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run one exact pathway cell and return test and outer-train risks.

    The selection/refit order intentionally mirrors the archived probe: one
    seeded inner model, then a fresh model with the same seed and the selected
    number of full outer-train epochs.  Recovery cells set ``select_epoch`` to
    false and replay the archived epoch only.
    """
    inner_best = float("nan")
    if select_epoch:
        set_seed(seed)
        inner_train, inner_val = split(train_idx, raw.dss_event, seed + 101)
        model = make(name, arrays).to(device)
        params = nparams(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best = -float("inf")
        selected_epoch = 1
        wait = 0
        for epoch in range(1, 26):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tt = torch.as_tensor(raw.dss_time[inner_train], dtype=torch.float32, device=device)
            ee = torch.as_tensor(raw.dss_event[inner_train], dtype=torch.float32, device=device)
            risk = model(*inps(name, arrays, inner_train, device))
            loss = cox_loss_grouped(tt, ee, risk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val = model(*inps(name, arrays, inner_val, device)).detach().cpu().numpy()
            score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val)
            if score > best + 1e-5:
                best = score
                selected_epoch = epoch
                wait = 0
            else:
                wait += 1
            if wait >= 5:
                break
        inner_best = float(best)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if selected_epoch is None or int(selected_epoch) < 1:
        raise ValueError(f"invalid selected epoch for {name} seed={seed} fold={fold}: {selected_epoch}")

    set_seed(seed)
    model = make(name, arrays).to(device)
    params = nparams(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tt = torch.as_tensor(raw.dss_time[train_idx], dtype=torch.float32, device=device)
    ee = torch.as_tensor(raw.dss_event[train_idx], dtype=torch.float32, device=device)
    for _ in range(int(selected_epoch)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(*inps(name, arrays, train_idx, device))
        loss = cox_loss_grouped(tt, ee, risk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_risk = model(*inps(name, arrays, test_idx, device)).detach().cpu().numpy().astype(np.float64)
        train_risk = model(*inps(name, arrays, train_idx, device)).detach().cpu().numpy().astype(np.float64)
    uno, numerator, denominator = uno_c_index(
        raw.dss_time[test_idx], raw.dss_event[test_idx], test_risk,
        raw.dss_time[train_idx], raw.dss_event[train_idx],
    )
    row = {
        "method": name,
        "model_field": name,
        "model_class": type(model).__name__,
        "seed": int(seed),
        "outer_fold": int(fold),
        "harrell_c": float(harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], test_risk)),
        "uno_c": float(uno),
        "uno_numerator": float(numerator),
        "uno_denominator": float(denominator),
        "best_epoch": int(selected_epoch),
        "inner_best_harrell": inner_best,
        "params": int(params),
        "n_test": int(len(test_idx)),
        "events_test": int(raw.dss_event[test_idx].sum()),
        "status": "completed",
        "recovery_mode": "full_selection_refit" if select_epoch else "archived_epoch_replay",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, test_risk, train_risk


def save_cell(cache: Path, model: str, seed: int, fold: int, test_idx: np.ndarray, test_risk: np.ndarray, train_idx: np.ndarray, train_risk: np.ndarray) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / f"{model.replace('+', '_').replace('-', '_')}_s{seed}_f{fold}.npz", test_idx=test_idx.astype(np.int32), test_risk=test_risk, train_idx=train_idx.astype(np.int32), train_risk=train_risk)


def load_old_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"method", "seed", "outer_fold", "best_epoch", "harrell_c", "uno_c", "params"}
    if not required.issubset(frame.columns):
        raise ValueError(f"archived fold file lacks {required - set(frame.columns)}: {path}")
    return frame


def _all_km(raw: Any, train_indices: dict[int, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for fold, idx in train_indices.items():
        result[fold] = censoring_survival(raw.dss_time[idx], raw.dss_event[idx])
    return result


def _uno_from_arrays(t: np.ndarray, e: np.ndarray, r: np.ndarray, fold_for_patient: np.ndarray, km_by_fold: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    numerator = 0.0
    denominator = 0.0
    for i in np.flatnonzero(e == 1):
        km_t, km_g = km_by_fold[int(fold_for_patient[i])]
        g = float(lookup_censor_survival(np.asarray([t[i]]), km_t, km_g)[0])
        if g <= 1e-8:
            continue
        mask = t > t[i]
        weight = 1.0 / (g * g)
        denominator += weight * float(mask.sum())
        numerator += weight * float((r[i] > r[mask]).sum() + 0.5 * (r[i] == r[mask]).sum())
    return (float(numerator / denominator) if denominator else float("nan"), numerator, denominator)


def oof_uno_local(raw: Any, risks: np.ndarray, fold_for_patient: np.ndarray, km_by_fold: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    return _uno_from_arrays(np.asarray(raw.dss_time, dtype=float), np.asarray(raw.dss_event, dtype=int), np.asarray(risks, dtype=float), fold_for_patient, km_by_fold)


def collect_oof(rows: list[dict[str, Any]], cache: Path, raw: Any, out_dir: Path, prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    summaries = []
    train_indices = {fold: np.flatnonzero(raw.outer_folds != fold) for fold in range(1, 6)}
    km_by_fold = _all_km(raw, train_indices)
    for model in sorted({str(r["method"]) for r in rows}):
        for seed in sorted({int(r["seed"]) for r in rows if r["method"] == model}):
            selected = [r for r in rows if r["method"] == model and int(r["seed"]) == seed]
            if len(selected) != 5:
                raise ValueError(f"OOF requires exactly five folds: {model} seed={seed} got {len(selected)}")
            risk = np.full(len(raw.patient_ids), np.nan, dtype=float)
            for row in selected:
                fold = int(row["outer_fold"])
                p = cache / f"{model.replace('+', '_').replace('-', '_')}_s{seed}_f{fold}.npz"
                with np.load(p) as z:
                    idx = np.asarray(z["test_idx"], dtype=int)
                    values = np.asarray(z["test_risk"], dtype=float)
                if np.any(np.isfinite(risk[idx])) or len(idx) != len(np.unique(idx)):
                    raise ValueError(f"duplicate OOF assignment: {model} seed={seed} fold={fold}")
                risk[idx] = values
            if not np.all(np.isfinite(risk)):
                raise ValueError(f"incomplete OOF coverage: {model} seed={seed}")
            fold_for_patient = raw.outer_folds.astype(int)
            u, un, ud = oof_uno_local(raw, risk, fold_for_patient, km_by_fold)
            records.append({
                "method": model,
                "seed": seed,
                "oof_harrell_c": float(harrell_c(raw.dss_time, raw.dss_event, risk)),
                "oof_uno_c_fold_local_ipcw": float(u),
                "oof_uno_numerator": float(un),
                "oof_uno_denominator": float(ud),
                "n_patients": int(len(risk)),
                "events": int(raw.dss_event.sum()),
                "folds_concatenated": 5,
                "status": "completed",
            })
        z = [r for r in records if r["method"] == model]
        summaries.append({
            "method": model,
            "seed_oof_harrell_mean": float(np.mean([r["oof_harrell_c"] for r in z])),
            "seed_oof_harrell_std": float(np.std([r["oof_harrell_c"] for r in z], ddof=1)),
            "seed_oof_uno_mean": float(np.mean([r["oof_uno_c_fold_local_ipcw"] for r in z])),
            "seed_oof_uno_std": float(np.std([r["oof_uno_c_fold_local_ipcw"] for r in z], ddof=1)),
            "n_seeds": len(z),
            "aggregation": "mean/std of three per-seed five-fold OOF metrics",
            "status": "completed",
        })
    oof = pd.DataFrame(records)
    summary = pd.DataFrame(summaries)
    atomic_csv(out_dir / f"{prefix}seed_oof_metrics.csv", records)
    atomic_csv(out_dir / f"{prefix}seed_oof_summary.csv", summaries)
    return oof, summary


def fast_harrell(t: np.ndarray, e: np.ndarray, risk: np.ndarray) -> float:
    comparable = (e[:, None] == 1) & (t[None, :] > t[:, None])
    denominator = int(comparable.sum())
    if denominator == 0:
        return float("nan")
    better = (risk[:, None] > risk[None, :])
    ties = (risk[:, None] == risk[None, :])
    return float((better[comparable].sum() + 0.5 * ties[comparable].sum()) / denominator)


def paired_bootstrap(raw: Any, cache: Path, model_a: str, model_b: str, seeds: tuple[int, ...], out_dir: Path, n_boot: int = 1000) -> dict[str, Any]:
    risks_a = {}
    risks_b = {}
    for seed in seeds:
        risks_a[seed] = np.full(len(raw.patient_ids), np.nan)
        risks_b[seed] = np.full(len(raw.patient_ids), np.nan)
        for fold in range(1, 6):
            for model, dest in ((model_a, risks_a[seed]), (model_b, risks_b[seed])):
                p = cache / f"{model.replace('+', '_').replace('-', '_')}_s{seed}_f{fold}.npz"
                with np.load(p) as z:
                    dest[np.asarray(z["test_idx"], dtype=int)] = np.asarray(z["test_risk"], dtype=float)
    rng = np.random.default_rng(20261004)
    dh = np.empty(n_boot, dtype=float)
    du = np.empty(n_boot, dtype=float)
    train_indices = {fold: np.flatnonzero(raw.outer_folds != fold) for fold in range(1, 6)}
    km_by_fold = _all_km(raw, train_indices)
    for b in range(n_boot):
        sample = rng.integers(0, len(raw.patient_ids), size=len(raw.patient_ids))
        hs = []
        us = []
        for seed in seeds:
            hs.append(fast_harrell(raw.dss_time[sample], raw.dss_event[sample], risks_b[seed][sample]) - fast_harrell(raw.dss_time[sample], raw.dss_event[sample], risks_a[seed][sample]))
            ub, _, _ = _uno_from_arrays(raw.dss_time[sample], raw.dss_event[sample], risks_b[seed][sample], raw.outer_folds[sample], km_by_fold)
            ua, _, _ = _uno_from_arrays(raw.dss_time[sample], raw.dss_event[sample], risks_a[seed][sample], raw.outer_folds[sample], km_by_fold)
            us.append(ub - ua)
        dh[b] = float(np.mean(hs))
        du[b] = float(np.mean(us))
    result = {
        "model_a": model_a,
        "model_b": model_b,
        "delta_definition": f"{model_b} - {model_a}",
        "seeds_pooled_per_replicate": list(seeds),
        "bootstrap_replicates": int(n_boot),
        "paired_patient_resampling": True,
        "delta_harrell_mean": float(np.mean(dh)),
        "delta_harrell_ci95_low": float(np.quantile(dh, 0.025)),
        "delta_harrell_ci95_high": float(np.quantile(dh, 0.975)),
        "delta_uno_mean": float(np.mean(du)),
        "delta_uno_ci95_low": float(np.quantile(du, 0.025)),
        "delta_uno_ci95_high": float(np.quantile(du, 0.975)),
        "status": "completed",
    }
    atomic_json(out_dir / "paired_patient_bootstrap.json", result)
    return result


def _cox_linear(x_train: np.ndarray, t_train: np.ndarray, e_train: np.ndarray, alpha: float, seed: int, layer_norm: bool = False, steps: int = 220) -> np.ndarray:
    set_seed(seed)
    device = _device()
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    t = torch.as_tensor(t_train, dtype=torch.float32, device=device)
    e = torch.as_tensor(e_train, dtype=torch.float32, device=device)
    w = nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float32, device=device))
    ln = nn.LayerNorm(x.shape[1], elementwise_affine=False).to(device) if layer_norm else None
    opt = torch.optim.Adam([w], lr=0.03)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        z = ln(x) if ln is not None else x
        loss = cox_loss_fast(t, e, z @ w) + 0.5 * float(alpha) * (w * w).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 5.0)
        opt.step()
    out = w.detach().cpu().numpy().astype(np.float64)
    del w, opt, ln
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _linear_predict(x: np.ndarray, w: np.ndarray, layer_norm: bool = False) -> np.ndarray:
    if layer_norm:
        x = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-5)
    return np.asarray(x, dtype=np.float64) @ w


def _mlp_cell(x: np.ndarray, raw: Any, train_idx: np.ndarray, test_idx: np.ndarray, seed: int, no_layer_norm: bool = False) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    set_seed(seed)
    device = _device()
    inner_train, inner_val = split(train_idx, raw.dss_event, seed + 101)
    n = x.shape[1]

    def new_model() -> nn.Module:
        first = [] if no_layer_norm else [nn.LayerNorm(n)]
        return nn.Sequential(*first, nn.Linear(n, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 1)).to(device)

    model = new_model()
    params = nparams(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = -float("inf")
    best_epoch = 1
    wait = 0
    xx = torch.as_tensor(x, dtype=torch.float32, device=device)
    for epoch in range(1, 26):
        model.train(); opt.zero_grad(set_to_none=True)
        tt = torch.as_tensor(raw.dss_time[inner_train], dtype=torch.float32, device=device)
        ee = torch.as_tensor(raw.dss_event[inner_train], dtype=torch.float32, device=device)
        loss = cox_loss_fast(tt, ee, model(xx[inner_train]).squeeze(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad(): val = model(xx[inner_val]).squeeze(-1).cpu().numpy()
        score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val)
        if score > best + 1e-5:
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if wait >= 5:
            break
    del model, opt
    if device.type == "cuda": torch.cuda.empty_cache()
    set_seed(seed)
    model = new_model(); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tt = torch.as_tensor(raw.dss_time[train_idx], dtype=torch.float32, device=device)
    ee = torch.as_tensor(raw.dss_event[train_idx], dtype=torch.float32, device=device)
    for _ in range(best_epoch):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = cox_loss_fast(tt, ee, model(xx[train_idx]).squeeze(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()
    with torch.no_grad():
        pred_test = model(xx[test_idx]).squeeze(-1).cpu().numpy().astype(np.float64)
        pred_train = model(xx[train_idx]).squeeze(-1).cpu().numpy().astype(np.float64)
    del model, opt
    if device.type == "cuda": torch.cuda.empty_cache()
    return pred_test, pred_train, int(params), int(best_epoch), float(best)


def run_mechanism_controls(raw: Any, fold_arrays: dict[int, Any], cache: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = ("RNA-Linear", "RNA-LN-Linear", "RNA-PCA32-Linear", "RNA-PathwayMean-Linear", "RNA-MLP", "RNA-MLP-noLN")
    rows = []
    for fold in range(1, 6):
        arrays = fold_arrays[fold]
        train_idx = np.flatnonzero(raw.outer_folds != fold)
        test_idx = np.flatnonzero(raw.outer_folds == fold)
        pathway = np.stack([arrays.mmp_groups[:, i, :int(w)].mean(axis=1) for i, w in enumerate(arrays.mmp_widths)], axis=1).astype(np.float32)
        pca = PCA(n_components=32, svd_solver="full", random_state=0).fit(arrays.rna[train_idx])
        pca_x = pca.transform(arrays.rna).astype(np.float32)
        x_by_method = {
            "RNA-Linear": arrays.rna,
            "RNA-LN-Linear": arrays.rna,
            "RNA-PCA32-Linear": pca_x,
            "RNA-PathwayMean-Linear": pathway,
            "RNA-MLP": arrays.rna,
            "RNA-MLP-noLN": arrays.rna,
        }
        for seed in SEEDS:
            for method in methods:
                x = x_by_method[method]
                if method.startswith("RNA-MLP"):
                    pred_test, pred_train, params, best_epoch, inner_best = _mlp_cell(x, raw, train_idx, test_idx, seed, no_layer_norm=method.endswith("noLN"))
                    alpha = "NA"
                else:
                    inner_train, inner_val = split(train_idx, raw.dss_event, seed + 101)
                    layer_norm = method == "RNA-LN-Linear"
                    scores = []
                    for alpha_try in ALPHAS:
                        w = _cox_linear(x[inner_train], raw.dss_time[inner_train], raw.dss_event[inner_train], alpha_try, seed, layer_norm=layer_norm, steps=180)
                        scores.append((harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], _linear_predict(x[inner_val], w, layer_norm)), alpha_try))
                    scores.sort(key=lambda z: (-z[0], z[1]))
                    alpha = float(scores[0][1])
                    w = _cox_linear(x[train_idx], raw.dss_time[train_idx], raw.dss_event[train_idx], alpha, seed, layer_norm=layer_norm)
                    pred_test = _linear_predict(x[test_idx], w, layer_norm)
                    pred_train = _linear_predict(x[train_idx], w, layer_norm)
                    params = int(len(w)); best_epoch = 220; inner_best = float(scores[0][0])
                u, un, ud = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], pred_test, raw.dss_time[train_idx], raw.dss_event[train_idx])
                rows.append({"method": method, "model_field": method, "seed": int(seed), "outer_fold": fold, "harrell_c": float(harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred_test)), "uno_c": float(u), "uno_numerator": float(un), "uno_denominator": float(ud), "alpha": alpha, "best_epoch": int(best_epoch), "inner_best_harrell": inner_best, "params": params, "status": "completed"})
                save_cell(cache, method, seed, fold, test_idx, pred_test, train_idx, pred_train)
    atomic_csv(out_dir / "mechanism_fold_metrics.csv", rows)
    oof, summary = collect_oof(rows, cache, raw, out_dir, prefix="mechanism_")
    # Add parameter and training diagnostics to the summary without changing
    # the OOF aggregation definition.
    extra = []
    for method in methods:
        z = [r for r in rows if r["method"] == method]
        extra.append({"method": method, "median_params": int(np.median([r["params"] for r in z])), "median_selected_epoch": float(np.median([r["best_epoch"] for r in z])), "median_inner_harrell": float(np.median([r["inner_best_harrell"] for r in z]))})
    summary = summary.merge(pd.DataFrame(extra), on="method", how="left")
    atomic_csv(out_dir / "mechanism_summary.csv", summary.to_dict("records"))
    return pd.DataFrame(rows), summary


def main(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root)
    runtime = Path(args.runtime_root)
    out = Path(args.reports_root)
    out.mkdir(parents=True, exist_ok=True)
    cache = runtime / "oof_cache"
    mechanism_cache = runtime / "mechanism_oof_cache"
    data_root = Path(os.environ["DIFFUSION_DATA_ROOT"])
    started = time.time()
    device = _device()
    raw = load_raw_inputs(data_root, repo)
    pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    fold_arrays = {fold: preprocess_fold(raw, pool, fold) for fold in range(1, 6)}
    train_indices = {fold: np.flatnonzero(raw.outer_folds != fold) for fold in range(1, 6)}
    test_indices = {fold: np.flatnonzero(raw.outer_folds == fold) for fold in range(1, 6)}

    rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    # The only full selection rerun is Pathway-RNA, as requested.
    for fold in range(1, 6):
        for seed in SEEDS:
            row, test_risk, train_risk = _train_pathway_cell("Pathway-RNA", fold_arrays[fold], raw, train_indices[fold], test_indices[fold], fold, seed, device, select_epoch=True)
            rows.append(row)
            save_cell(cache, "Pathway-RNA", seed, fold, test_indices[fold], test_risk, train_indices[fold], train_risk)

    # Existing RNA+CNA and Signature raw cells are preserved.  We only replay
    # the recorded final epoch to create private OOF predictions and validate
    # that the replay metrics agree with the archived cell.
    old_sources = {
        "Pathway-RNA+CNA": Path(args.old_cna_rows),
        "Signature-RNA+CNA": Path(args.old_signature_rows),
    }
    for model, path in old_sources.items():
        old = load_old_rows(path)
        for _, archived in old.iterrows():
            fold = int(archived.outer_fold); seed = int(archived.seed)
            row, test_risk, train_risk = _train_pathway_cell(model, fold_arrays[fold], raw, train_indices[fold], test_indices[fold], fold, seed, device, selected_epoch=int(archived.best_epoch), select_epoch=False)
            row["archived_harrell_c"] = float(archived.harrell_c)
            row["archived_uno_c"] = float(archived.uno_c)
            row["archived_params"] = int(archived.params)
            row["replay_harrell_abs_error"] = abs(row["harrell_c"] - float(archived.harrell_c))
            row["replay_uno_abs_error"] = abs(row["uno_c"] - float(archived.uno_c))
            row["replay_params_match"] = bool(row["params"] == int(archived.params))
            row["replay_metric_match"] = bool(row["replay_harrell_abs_error"] <= 1e-8 and row["replay_uno_abs_error"] <= 1e-8 and row["replay_params_match"])
            replay_rows.append(row)
            rows.append(row)
            save_cell(cache, model, seed, fold, test_indices[fold], test_risk, train_indices[fold], train_risk)
    if not all(bool(r.get("replay_metric_match", True)) for r in replay_rows):
        raise RuntimeError("archived replay did not reproduce one or more existing raw cells; refusing to label the recovered OOF as exact")

    fold_metrics = pd.DataFrame(rows)
    atomic_csv(out / "fold_cell_metrics.csv", rows)
    atomic_csv(out / "replay_validation.csv", replay_rows)
    oof, oof_summary = collect_oof(rows, cache, raw, out)
    paired_rows = []
    for seed in SEEDS:
        for fold in range(1, 6):
            a = next(r for r in rows if r["method"] == "Pathway-RNA" and int(r["seed"]) == seed and int(r["outer_fold"]) == fold)
            b = next(r for r in rows if r["method"] == "Pathway-RNA+CNA" and int(r["seed"]) == seed and int(r["outer_fold"]) == fold)
            paired_rows.append({"seed": seed, "outer_fold": fold, "rna_harrell_c": a["harrell_c"], "rna_cna_harrell_c": b["harrell_c"], "delta_harrell_c": b["harrell_c"] - a["harrell_c"], "rna_uno_c": a["uno_c"], "rna_cna_uno_c": b["uno_c"], "delta_uno_c": b["uno_c"] - a["uno_c"], "replay_validated": bool(b.get("replay_metric_match", False))})
    atomic_csv(out / "paired_fold_comparison.csv", paired_rows)
    paired_seed = []
    for seed in SEEDS:
        a = oof[(oof.method == "Pathway-RNA") & (oof.seed == seed)].iloc[0]
        b = oof[(oof.method == "Pathway-RNA+CNA") & (oof.seed == seed)].iloc[0]
        paired_seed.append({"seed": seed, "rna_oof_harrell_c": a.oof_harrell_c, "rna_cna_oof_harrell_c": b.oof_harrell_c, "delta_oof_harrell_c": b.oof_harrell_c - a.oof_harrell_c, "rna_oof_uno_c": a.oof_uno_c_fold_local_ipcw, "rna_cna_oof_uno_c": b.oof_uno_c_fold_local_ipcw, "delta_oof_uno_c": b.oof_uno_c_fold_local_ipcw - a.oof_uno_c_fold_local_ipcw})
    atomic_csv(out / "paired_seed_oof_comparison.csv", paired_seed)
    bootstrap = paired_bootstrap(raw, cache, "Pathway-RNA", "Pathway-RNA+CNA", SEEDS, out, n_boot=1000)

    mechanism_rows, mechanism_summary = run_mechanism_controls(raw, fold_arrays, mechanism_cache, out)
    profiles = []
    for name in METHODS:
        obj = make(name, fold_arrays[1])
        profiles.append({"model_field": name, "model_class": type(obj).__name__, "params": nparams(obj), "trainable_parameter_count": nparams(obj), "uses_clinical": False, "input": "MMP RNA groups" if name == "Pathway-RNA" else ("MMP RNA groups + top256 CNA" if name == "Pathway-RNA+CNA" else "six MCAT signature RNA+CNA groups")})
    source_files = [repo / "src/po_phase0/pathway_probe.py", repo / "src/phase_m2/data.py", repo / "src/phase_m2/common.py", repo / "src/po_phase0/runner.py"]
    manifest = {
        "phase": "PO-pathway-representation-repair",
        "cohort": int(len(raw.patient_ids)),
        "events": int(raw.dss_event.sum()),
        "folds": 5,
        "seeds": list(SEEDS),
        "models": list(METHODS),
        "full_selection_rerun": ["Pathway-RNA"],
        "existing_raw_preserved": ["Pathway-RNA+CNA", "Signature-RNA+CNA"],
        "oof_definition": "each seed concatenates five disjoint outer-test predictions; Harrell is computed over all 512 OOF predictions; Uno uses each patient's outer-train censoring KM while comparing against the complete OOF cohort",
        "paired_comparison": "same seed and same outer fold; CNA minus RNA",
        "linear_controls": ["RNA-Linear", "RNA-LN-Linear", "RNA-PCA32-Linear", "RNA-PathwayMean-Linear", "RNA-MLP", "RNA-MLP-noLN"],
        "source_sha256": {str(p.relative_to(repo)): sha256_file(p) for p in source_files},
        "patient_order_sha256": hashlib.sha256("\n".join(raw.patient_ids).encode()).hexdigest(),
        "device": str(device),
        "runtime_seconds": time.time() - started,
        "status": "completed",
    }
    atomic_json(out / "repair_manifest.json", manifest)
    atomic_json(out / "model_profiles.json", {"profiles": profiles})
    atomic_json(out / "paired_comparison.json", {"comparison": "Pathway-RNA+CNA - Pathway-RNA", "seed_oof": paired_seed, "bootstrap": bootstrap, "status": "completed"})
    lines = ["# Pathway representation repair", "", "- Pathway-RNA was fully rerun with 3 seeds x 5 outer folds.", "- Pathway-RNA+CNA and Signature raw cells were preserved; only archived-epoch replay was used to recover private OOF predictions.", "- `seed_oof_*` is computed after concatenating five outer-test predictions per seed. It is not a mean over 15 cells.", "", "## Genuine per-seed OOF metrics", "", "|Method|OOF Harrell mean +/- std|OOF Uno mean +/- std|", "|---|---:|---:|"]
    for _, r in oof_summary.iterrows(): lines.append(f"|{r['method']}|{r['seed_oof_harrell_mean']:.4f} +/- {r['seed_oof_harrell_std']:.4f}|{r['seed_oof_uno_mean']:.4f} +/- {r['seed_oof_uno_std']:.4f}|")
    lines += ["", "## Same-seed paired comparison", "", f"- Delta definition: Pathway-RNA+CNA minus Pathway-RNA.", f"- Bootstrap paired-patient Harrell delta: {bootstrap['delta_harrell_mean']:.4f} (95% CI {bootstrap['delta_harrell_ci95_low']:.4f} to {bootstrap['delta_harrell_ci95_high']:.4f}).", f"- Bootstrap paired-patient Uno delta: {bootstrap['delta_uno_mean']:.4f} (95% CI {bootstrap['delta_uno_ci95_low']:.4f} to {bootstrap['delta_uno_ci95_high']:.4f}).", "- The comparison is conditional on the two fitted architectures and does not establish a universal biological claim about CNA.", "", "## Linear-layer mechanism controls", "", "The controls hold the outer split and preprocessing fixed; alpha and epoch are selected only on inner data.", "", "|Control|OOF Harrell mean +/- std|OOF Uno mean +/- std|Median parameters|", "|---|---:|---:|---:|"]
    for _, r in mechanism_summary.iterrows():
        lines.append(f"|{r['method']}|{r['seed_oof_harrell_mean']:.4f} +/- {r['seed_oof_harrell_std']:.4f}|{r['seed_oof_uno_mean']:.4f} +/- {r['seed_oof_uno_std']:.4f}|{int(r['median_params'])}|")
    lines += ["", "- These controls explain the direct RNA linear-vs-MLP result only; they do not claim that every hidden pathway head should be linear.", "", f"- Total wall time: {time.time() - started:.1f} seconds."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--reports-root", required=True)
    p.add_argument("--old-cna-rows", required=True)
    p.add_argument("--old-signature-rows", required=True)
    main(p.parse_args())

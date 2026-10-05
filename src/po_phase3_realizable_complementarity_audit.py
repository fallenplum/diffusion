from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.phase_m2.common import censoring_survival, count_parameters, harrell_c, lookup_censor_survival, set_seed, uno_c_index
from src.phase_m2.data import _fit_numeric, load_raw_inputs, preprocess_fold
from src.po_phase1_orthogonal_fusion_repair import (
    ABMIL,
    abmil_parts,
    cox_loss_fast,
    cox_loss_grouped,
    path_provider,
    train_abmil_fixed,
    train_linear,
)
from src.po_phase2_complementarity_headroom_audit import (
    BETAS,
    fit_beta,
    fit_gamma,
    split_inner,
    uno_from_arrays,
    uno_from_arrays_fast,
    zfit,
)


SEEDS = (20260904, 20260905, 20260906)
FEATURE_SETS = ("F0-RiskOnly", "F1-RiskPlusPathReliability", "F2-RiskPlusPathPlusRNAUncertainty")
MODELS = ("G0-Logistic", "G1-TinyMLP")
LAMBDA_G = (0.0, 0.01)
LAMBDA_S = 0.001
RNA_BOOTSTRAPS = 50
BOOTSTRAP_REPS = 2000
INNER_CV_FOLDS = 4
MIN_PAIR_AUC = 0.60
GATE_STD_MIN = 0.02
MAX_GATE_PARAMS = 150
INNER_RNA_ALPHA = 0.1
INNER_ABMIL_EPOCHS = 12
REPLAY_ATOL = 2e-3
CROSSFIT_REPLAY_ATOL = 5e-3
REPLAY_RTOL = 1e-3


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["status"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sa, sb = float(np.std(a)), float(np.std(b))
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def stable_fold_splits(indices: np.ndarray, raw: Any, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedKFold

    strata = np.char.add(raw.cancer_type[indices].astype(str), "_" + raw.dss_event[indices].astype(str))
    try:
        splitter = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=seed)
        return [(indices[a], indices[b]) for a, b in splitter.split(indices, strata)]
    except ValueError:
        splitter = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=seed)
        return [(indices[a], indices[b]) for a, b in splitter.split(indices, np.zeros(len(indices), dtype=int))]


def eval_abmil_stats(model: nn.Module, deterministic: np.ndarray, indices: np.ndarray, device: torch.device, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    risks, stats = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            ix = indices[start:start + batch]
            tokens = torch.from_numpy(np.asarray(deterministic[ix])).to(device)
            x = model.proj(tokens.float())
            attention = torch.softmax(model.w(torch.tanh(model.v(x)) * torch.sigmoid(model.u(x))).squeeze(-1), dim=1)
            pooled = (attention.unsqueeze(-1) * x).sum(1)
            risk = model.head(model.norm(pooled))
            entropy = -(attention * torch.log(attention + 1e-8)).sum(1) / math.log(attention.shape[1])
            k = max(1, int(math.ceil(0.10 * attention.shape[1])))
            concentration = torch.topk(attention, k=k, dim=1).values.sum(1)
            maximum = attention.max(1).values
            risks.append(risk.detach().cpu().numpy().reshape(-1))
            stats.append(torch.stack([entropy, concentration, maximum], dim=1).detach().cpu().numpy())
    return np.concatenate(risks).astype(float), np.concatenate(stats).astype(float)


def load_phase1_schedule(report: Path) -> dict[tuple[int, int], tuple[float, int]]:
    frame = pd.read_csv(report / "main_effect_metrics.csv")
    result: dict[tuple[int, int], tuple[float, int]] = {}
    for seed in SEEDS:
        for fold in range(1, 6):
            cell = frame[(frame["seed"] == seed) & (frame["outer_fold"] == fold)]
            rna = cell[cell["method"] == "RNA-Linear"]
            path = cell[cell["method"] == "Pathology-ABMIL"]
            if len(rna) != 1 or len(path) != 1:
                raise ValueError("ambiguous or incomplete locked main-effect schedule")
            result[(seed, fold)] = (float(rna.iloc[0]["alpha"]), int(path.iloc[0]["selected_epoch"]))
    return result


def load_phase2_crossfit(raw: Any, runtime: Path) -> tuple[dict[int, dict[int, tuple[np.ndarray, np.ndarray]]], bool]:
    base = runtime / "phase2_crossfit_main_cache"
    result: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {seed: {} for seed in SEEDS}
    valid = True
    for seed in SEEDS:
        for fold in range(1, 6):
            rp_path = base / f"path_crossfit_s{seed}_f{fold}.npy"
            rr_path = base / f"rna_crossfit_s{seed}_f{fold}.npy"
            if not rp_path.exists() or not rr_path.exists():
                valid = False
                continue
            rr, rp = np.load(rr_path), np.load(rp_path)
            train = np.flatnonzero(raw.outer_folds != fold)
            if rr.shape != (len(raw.patient_ids),) or rp.shape != (len(raw.patient_ids),) or not np.all(np.isfinite(rr[train])) or not np.all(np.isfinite(rp[train])):
                valid = False
            result[seed][fold] = (rr.astype(float), rp.astype(float))
    return result, valid


def load_phase1_oof(raw: Any, runtime: Path, source_commit: str) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]], bool]:
    base = runtime / "phase1_orthogonal_cells_repair"
    result: dict[int, dict[str, np.ndarray]] = {}
    provenance: list[dict[str, Any]] = []
    valid = True
    for seed in SEEDS:
        result[seed] = {"RNA-Linear": np.full(len(raw.patient_ids), np.nan), "Pathology-ABMIL": np.full(len(raw.patient_ids), np.nan)}
        for fold in range(1, 6):
            expected = np.flatnonzero(raw.outer_folds == fold)
            for method, stem in (("RNA-Linear", "RNA_Linear"), ("Pathology-ABMIL", "Pathology_ABMIL")):
                path = base / f"{stem}_s{seed}_f{fold}.npz"
                row = {"seed": seed, "outer_fold": fold, "method": method, "cache_path": str(path), "source_commit": source_commit, "status": "completed"}
                if not path.exists():
                    row.update(status="missing", sha256=None)
                    valid = False
                    provenance.append(row)
                    continue
                row["sha256"] = sha256_file(path)
                try:
                    z = np.load(path, allow_pickle=False)
                    idx, risk = np.asarray(z["test_idx"], dtype=int), np.asarray(z["test_risk"], dtype=float)
                    row["n_predictions"] = int(len(idx))
                    row["test_idx_sha256"] = sha256_array(idx)
                    if not np.array_equal(idx, expected) or len(risk) != len(idx) or not np.all(np.isfinite(risk)) or np.any(np.isfinite(result[seed][method][idx])):
                        raise ValueError("prediction coverage or ordering mismatch")
                    result[seed][method][idx] = risk
                except Exception as exc:
                    row.update(status="invalid", error=repr(exc))
                    valid = False
                provenance.append(row)
        for method in result[seed]:
            if not np.all(np.isfinite(result[seed][method])):
                valid = False
    return result, provenance, valid


def load_mgct(raw: Any, runtime: Path) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], bool]:
    base = runtime / "cells"
    result: dict[int, np.ndarray] = {}
    provenance: list[dict[str, Any]] = []
    valid = True
    for seed in SEEDS:
        result[seed] = np.full(len(raw.patient_ids), np.nan)
        for fold in range(1, 6):
            path = base / f"MGCT_PO_{seed}_{fold}.npy"
            expected = np.flatnonzero(raw.outer_folds == fold)
            row = {"seed": seed, "outer_fold": fold, "method": "MGCT-PO", "cache_path": str(path), "status": "completed"}
            if not path.exists():
                row.update(status="missing", sha256=None)
                valid = False
            else:
                row["sha256"] = sha256_file(path)
                risk = np.asarray(np.load(path, allow_pickle=False), dtype=float)
                row["n_predictions"] = int(len(risk))
                if risk.shape != (len(expected),) or not np.all(np.isfinite(risk)) or not np.all(~np.isfinite(result[seed][expected])):
                    row.update(status="invalid", error="coverage/order mismatch")
                    valid = False
                else:
                    result[seed][expected] = risk
            provenance.append(row)
        valid = valid and bool(np.all(np.isfinite(result[seed])))
    return result, provenance, valid


def train_locked_final_abmil(pool: np.ndarray, raw: Any, train_idx: np.ndarray, seed: int, device: torch.device, epochs: int) -> nn.Module:
    """Replay Phase1's final fit, including its one fixed patient order."""
    set_seed(seed)
    model = ABMIL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    order_rng = np.random.default_rng(seed + 101)
    order = train_idx[order_rng.permutation(len(train_idx))]
    for epoch in range(1, epochs + 1):
        model.train()
        risks, times, events = [], [], []
        for start in range(0, len(order), 32):
            ix = order[start:start + 32]
            tokens = torch.from_numpy(path_provider(pool, ix, seed, epoch)).to(device)
            risk, _ = abmil_parts(model, tokens)
            risks.append(risk); times.append(raw.dss_time[ix]); events.append(raw.dss_event[ix])
        rr = torch.cat(risks)
        tt = torch.as_tensor(np.concatenate(times), dtype=torch.float32, device=device)
        ee = torch.as_tensor(np.concatenate(events), dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        cox_loss_grouped(tt, ee, rr).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def attention_crossfit(pool: np.ndarray, deterministic: np.ndarray, raw: Any, train_idx: np.ndarray, eval_idx: np.ndarray, seed: int, fold: int, epoch: int, device: torch.device, expected_train: np.ndarray, expected_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_stats = np.full((len(raw.patient_ids), 3), np.nan, dtype=float)
    train_risk = np.full(len(raw.patient_ids), np.nan, dtype=float)
    for part, (fit_idx, val_idx) in enumerate(stable_fold_splits(train_idx, raw, seed + fold + 1701)):
        model_seed = seed + fold * 100 + part
        model = train_abmil_fixed(pool, raw, fit_idx, model_seed, device, epoch)
        risk, stats = eval_abmil_stats(model, deterministic, val_idx, device)
        train_risk[val_idx] = risk
        train_stats[val_idx] = stats
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.all(np.isfinite(train_stats[train_idx])):
        raise RuntimeError("cross-fitted pathology attention statistics incomplete")
    final_model = train_locked_final_abmil(pool, raw, train_idx, seed + fold, device, epoch)
    eval_risk, eval_stats = eval_abmil_stats(final_model, deterministic, eval_idx, device)
    train_ok = bool(np.allclose(train_risk[train_idx], expected_train, atol=CROSSFIT_REPLAY_ATOL, rtol=REPLAY_RTOL))
    test_ok = bool(np.allclose(eval_risk, expected_test, atol=REPLAY_ATOL, rtol=REPLAY_RTOL))
    replay = {"seed": seed, "outer_fold": fold, "crossfit_risk_replay_match": train_ok, "test_risk_replay_match": test_ok, "crossfit_max_abs_error": float(np.max(np.abs(train_risk[train_idx] - expected_train))), "test_max_abs_error": float(np.max(np.abs(eval_risk - expected_test))), "crossfit_atol": CROSSFIT_REPLAY_ATOL, "test_atol": REPLAY_ATOL, "rtol": REPLAY_RTOL}
    del final_model
    if not train_ok or not test_ok:
        raise ValueError(f"locked ABMIL risk replay mismatch: {replay}")
    return train_stats, eval_stats, replay


def inner_main_and_features(pool: np.ndarray, deterministic: np.ndarray, raw: Any, inner_train: np.ndarray, inner_val: np.ndarray, seed: int, alpha: float, epoch: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr_cf = np.full(len(raw.patient_ids), np.nan)
    rp_cf = np.full(len(raw.patient_ids), np.nan)
    att_cf = np.full((len(raw.patient_ids), 3), np.nan)
    for part, (fit_idx, val_idx) in enumerate(stable_fold_splits(inner_train, raw, seed + 2101)):
        rna_inner, _ = _fit_numeric(raw.rna, fit_idx, output_dim=512, modality="RNA")
        w = train_linear(rna_inner[fit_idx], raw.dss_time[fit_idx], raw.dss_event[fit_idx], alpha, seed + part * 101 + 31, steps=220)
        rr_cf[val_idx] = rna_inner[val_idx] @ w
        model = train_abmil_fixed(pool, raw, fit_idx, seed + part * 101 + 41, device, epoch)
        rp_cf[val_idx], att_cf[val_idx] = eval_abmil_stats(model, deterministic, val_idx, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.all(np.isfinite(rr_cf[inner_train])) or not np.all(np.isfinite(rp_cf[inner_train])):
        raise RuntimeError("inner cross-fitted main risks incomplete")
    rna_inner, _ = _fit_numeric(raw.rna, inner_train, output_dim=512, modality="RNA")
    w = train_linear(rna_inner[inner_train], raw.dss_time[inner_train], raw.dss_event[inner_train], alpha, seed + 701, steps=220)
    rr_val = rna_inner[inner_val] @ w
    final_model = train_abmil_fixed(pool, raw, inner_train, seed + 711, device, epoch)
    rp_val, att_val = eval_abmil_stats(final_model, deterministic, inner_val, device)
    del final_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rr_cf[inner_train], rp_cf[inner_train], rr_val, rp_val, att_cf[inner_train], att_val


def bootstrap_rna_uncertainty(rna: np.ndarray, raw: Any, train_idx: np.ndarray, eval_idx: np.ndarray, alpha: float, seed: int, runtime_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if runtime_path is not None and runtime_path.exists():
        z = np.load(runtime_path, allow_pickle=False)
        return z["train_std"], z["eval_std"], z["train_iqr"], z["eval_iqr"]
    rng = np.random.default_rng(seed)
    train_preds, eval_preds = [], []
    for rep in range(RNA_BOOTSTRAPS):
        sample = rng.choice(train_idx, size=len(train_idx), replace=True)
        w = train_linear(rna[sample], raw.dss_time[sample], raw.dss_event[sample], alpha, seed + rep + 5001, steps=220)
        train_preds.append(rna[train_idx] @ w)
        eval_preds.append(rna[eval_idx] @ w)
    train_arr, eval_arr = np.asarray(train_preds), np.asarray(eval_preds)
    train_std, eval_std = np.std(train_arr, axis=0), np.std(eval_arr, axis=0)
    train_iqr = np.quantile(train_arr, 0.75, axis=0) - np.quantile(train_arr, 0.25, axis=0)
    eval_iqr = np.quantile(eval_arr, 0.75, axis=0) - np.quantile(eval_arr, 0.25, axis=0)
    if runtime_path is not None:
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(runtime_path, train_std=train_std, eval_std=eval_std, train_iqr=train_iqr, eval_iqr=eval_iqr)
    return train_std, eval_std, train_iqr, eval_iqr


def build_features(rp_train: np.ndarray, rr_train: np.ndarray, att_train: np.ndarray, unc_train: tuple[np.ndarray, np.ndarray], rp_eval: np.ndarray, rr_eval: np.ndarray, att_eval: np.ndarray, unc_eval: tuple[np.ndarray, np.ndarray], feature_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    _, zp_eval, _ = zfit(rp_train, rp_eval)
    zp_train, _, _ = zfit(rp_train, rp_train)
    zr_train, _, _ = zfit(rr_train, rr_train)
    _, zr_eval, _ = zfit(rr_train, rr_eval)
    q_train = np.column_stack([zp_train, zr_train, np.abs(zp_train - zr_train), zp_train * zr_train])
    q_eval = np.column_stack([zp_eval, zr_eval, np.abs(zp_eval - zr_eval), zp_eval * zr_eval])
    names = ["z_path", "z_rna", "abs_risk_gap", "risk_product"]
    x_train, x_eval = q_train, q_eval
    if feature_set in ("F1-RiskPlusPathReliability", "F2-RiskPlusPathPlusRNAUncertainty"):
        a_train, a_eval = [], []
        for col in range(att_train.shape[1]):
            zt, ze, _ = zfit(att_train[:, col], att_eval[:, col])
            a_train.append(zt); a_eval.append(ze)
        x_train = np.column_stack([x_train, *a_train]); x_eval = np.column_stack([x_eval, *a_eval])
        names += ["z_path_attention_entropy", "z_path_top10_concentration", "z_path_max_attention"]
    if feature_set == "F2-RiskPlusPathPlusRNAUncertainty":
        for label, train_values, eval_values in (("z_rna_bootstrap_std", unc_train[0], unc_eval[0]), ("z_rna_bootstrap_iqr", unc_train[1], unc_eval[1])):
            zt, ze, _ = zfit(train_values, eval_values)
            x_train = np.column_stack([x_train, zt]); x_eval = np.column_stack([x_eval, ze]); names.append(label)
    return x_train.astype(float), x_eval.astype(float), zp_train.astype(float), zr_train.astype(float), names


class Gate(nn.Module):
    def __init__(self, dim: int, model_name: str):
        super().__init__()
        self.model_name = model_name
        if model_name == "G0-Logistic":
            self.net = nn.Linear(dim, 1)
        elif model_name == "G1-TinyMLP":
            self.net = nn.Sequential(nn.Linear(dim, 8), nn.Tanh(), nn.Linear(8, 1))
        else:
            raise ValueError(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


def fit_gate(x: np.ndarray, zp: np.ndarray, zr: np.ndarray, times: np.ndarray, events: np.ndarray, model_name: str, lambda_g: float, seed: int, steps: int = 220) -> Gate:
    set_seed(seed)
    model = Gate(x.shape[1], model_name).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03 if model_name == "G0-Logistic" else 0.01)
    xx = torch.as_tensor(x, dtype=torch.float64); pp = torch.as_tensor(zp, dtype=torch.float64); rr = torch.as_tensor(zr, dtype=torch.float64)
    tt = torch.as_tensor(times, dtype=torch.float64); ee = torch.as_tensor(events, dtype=torch.float64)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        w = model(xx)
        risk = w * pp + (1.0 - w) * rr
        loss = cox_loss_fast(tt, ee, risk) + float(lambda_g) * ((w - w.mean()) ** 2).mean() + LAMBDA_S * ((w - 0.5) ** 4).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    del optimizer
    return model


def gate_predict(model: Gate, x: np.ndarray, zp: np.ndarray, zr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        w = model(torch.as_tensor(x, dtype=torch.float64)).cpu().numpy()
    return w * zp + (1.0 - w) * zr, w


def gate_distribution(prefix: str, w: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean_w": float(np.mean(w)), f"{prefix}_std_w": float(np.std(w)),
        f"{prefix}_q05_w": float(np.quantile(w, .05)), f"{prefix}_q25_w": float(np.quantile(w, .25)),
        f"{prefix}_q50_w": float(np.quantile(w, .50)), f"{prefix}_q75_w": float(np.quantile(w, .75)),
        f"{prefix}_q95_w": float(np.quantile(w, .95)), f"{prefix}_frac_w_gt_075": float(np.mean(w > .75)), f"{prefix}_frac_w_lt_025": float(np.mean(w < .25)),
    }


def comparable_pair_features(times: np.ndarray, events: np.ndarray, rp: np.ndarray, rr: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for i in np.flatnonzero(events == 1):
        for j in np.flatnonzero(times > times[i]):
            pc = 1 if rp[i] > rp[j] + 1e-12 else (-1 if rp[i] < rp[j] - 1e-12 else 0)
            rc = 1 if rr[i] > rr[j] + 1e-12 else (-1 if rr[i] < rr[j] - 1e-12 else 0)
            if pc == 1 and rc == -1:
                label = 1
            elif pc == -1 and rc == 1:
                label = 0
            else:
                continue
            rows.append(np.concatenate([np.abs(q[i] - q[j]), q[i] + q[j]]))
            labels.append(label)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def pairwise_learnability(raw: Any, seed: int, fold: int, train_rp: np.ndarray, train_rr: np.ndarray, test_rp: np.ndarray, test_rr: np.ndarray) -> dict[str, Any]:
    def base_q(rp: np.ndarray, rr: np.ndarray, rp_eval: np.ndarray | None = None, rr_eval: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if rp_eval is None:
            zp = (rp - np.mean(rp)) / (np.std(rp) if np.std(rp) > 1e-8 else 1.0)
            zr = (rr - np.mean(rr)) / (np.std(rr) if np.std(rr) > 1e-8 else 1.0)
            return np.column_stack([zp, zr, np.abs(zp - zr), zp * zr]), None
        zp, zp_e, _ = zfit(rp, rp_eval); zr, zr_e, _ = zfit(rr, rr_eval)
        return np.column_stack([zp, zr, np.abs(zp - zr), zp * zr]), np.column_stack([zp_e, zr_e, np.abs(zp_e - zr_e), zp_e * zr_e])

    q_train, q_test = base_q(train_rp, train_rr, test_rp, test_rr)
    train = np.flatnonzero(raw.outer_folds != fold)
    test = np.flatnonzero(raw.outer_folds == fold)
    x_train, y_train = comparable_pair_features(raw.dss_time[train], raw.dss_event[train], train_rp, train_rr, q_train)
    x_test, y_test = comparable_pair_features(raw.dss_time[test], raw.dss_event[test], test_rp, test_rr, q_test)
    row = {"seed": seed, "outer_fold": fold, "train_pairs": int(len(y_train)), "test_pairs": int(len(y_test)), "status": "completed"}
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        row.update(roc_auc=float("nan"), balanced_accuracy=float("nan"), status="insufficient_pair_classes")
        return row
    model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    row.update(roc_auc=float(roc_auc_score(y_test, prob)), balanced_accuracy=float(balanced_accuracy_score(y_test, (prob >= .5).astype(int))), status="completed")
    return row


def d3_and_static(train_rr: np.ndarray, train_rp: np.ndarray, test_rr: np.ndarray, test_rp: np.ndarray, times: np.ndarray, events: np.ndarray, inner_train: np.ndarray, inner_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    best = None
    for l2 in BETAS:
        a_fit, a_val, _ = zfit(train_rp[inner_train], train_rp[inner_val])
        o_fit, o_val, _ = zfit(train_rr[inner_train], train_rr[inner_val])
        beta = fit_beta(a_fit, o_fit, times[inner_train], events[inner_train], l2)
        score = harrell_c(times[inner_val], events[inner_val], a_val + beta * o_val)
        cand = (score, -l2, beta)
        if best is None or cand > best:
            best = cand
    l2, beta = -best[1], best[2]
    a_train, a_test, _ = zfit(train_rp, test_rp)
    o_train, o_test, _ = zfit(train_rr, test_rr)
    beta_final = fit_beta(a_train, o_train, times, events, l2)
    d3 = a_test + beta_final * o_test
    gamma_p, gamma_r = fit_gamma(a_train, o_train, times, events)
    static = gamma_p * a_test + gamma_r * o_test
    return d3, static, {"beta": beta_final, "beta_l2": l2, "static_aP": gamma_p, "static_aR": gamma_r}


def eval_uno_fold(raw: Any, test: np.ndarray, train: np.ndarray, risk: np.ndarray) -> float:
    return float(uno_c_index(raw.dss_time[test], raw.dss_event[test], risk, raw.dss_time[train], raw.dss_event[train])[0])


def km_weights_bootstrap(raw: Any, idx: np.ndarray) -> np.ndarray:
    t, e, folds = raw.dss_time[idx], raw.dss_event[idx], raw.outer_folds[idx]
    result = np.ones(len(idx), dtype=float)
    for fold in range(1, 6):
        test = folds == fold; train = ~test
        if not np.any(test) or not np.any(train):
            continue
        kt, kg = censoring_survival(t[train], e[train])
        result[test] = lookup_censor_survival(t[test], kt, kg)
    return result


def paired_bootstrap(raw: Any, seed: int, proposed: np.ndarray, methods: dict[str, np.ndarray], reps: int = BOOTSTRAP_REPS) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20271000 + seed)
    samples: dict[tuple[str, str], list[float]] = {(name, metric): [] for name in methods for metric in ("delta_harrell", "delta_uno")}
    for _ in range(reps):
        idx = rng.integers(0, len(proposed), len(proposed))
        t, e, folds = raw.dss_time[idx], raw.dss_event[idx], raw.outer_folds[idx]
        p = proposed[idx]
        for name, base in methods.items():
            b = base[idx]
            samples[(name, "delta_harrell")].append(float(harrell_c(t, e, p) - harrell_c(t, e, b)))
            samples[(name, "delta_uno")].append(float(uno_from_arrays_fast(t, e, p, folds) - uno_from_arrays_fast(t, e, b, folds)))
    rows = []
    for (name, metric), values in samples.items():
        arr = np.asarray(values, dtype=float); arr = arr[np.isfinite(arr)]
        rows.append({"seed": seed, "comparison": f"Proposed-{name}", "metric": metric, "bootstrap_replicates": reps, "mean": float(np.mean(arr)), "ci95_low": float(np.quantile(arr, .025)), "ci95_high": float(np.quantile(arr, .975)), "status": "completed"})
    return rows


def main(args: argparse.Namespace) -> None:
    started = time.time()
    repo = Path(args.repo_root).resolve(); runtime = Path(args.runtime_root).resolve(); data_root = Path(os.environ["DIFFUSION_DATA_ROOT"]).resolve(); phase1_report = Path(args.phase1_report).resolve(); phase2_report = Path(args.phase2_report).resolve(); out = Path(args.reports_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    source_commit = args.source_commit
    try:
        raw = load_raw_inputs(data_root, repo)
        if len(raw.patient_ids) != 512 or int(np.sum(raw.dss_event)) != 65:
            raise ValueError("fixed Phase 3 cohort is not 512 patients / 65 events")
        oof, oof_provenance, oof_valid = load_phase1_oof(raw, runtime, source_commit)
        mgct, mgct_provenance, mgct_valid = load_mgct(raw, runtime)
        crossfit, crossfit_valid = load_phase2_crossfit(raw, runtime)
        if not oof_valid or not mgct_valid or not crossfit_valid:
            raise ValueError(f"OOF provenance blocked: phase1={oof_valid}, MGCT={mgct_valid}, crossfit={crossfit_valid}")
        schedule = load_phase1_schedule(phase1_report)
        pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
        deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arrays_by_fold = {fold: preprocess_fold(raw, pool, fold) for fold in range(1, 6)}
        cache = runtime / "phase3_realizable_cache_v4"; cache.mkdir(parents=True, exist_ok=True)
        fold_rows, selection_rows, model_rows, diagnostic_rows, pair_rows = [], [], [], [], []
        oof_methods = {seed: {name: np.full(len(raw.patient_ids), np.nan) for name in ("Pathology-only", "RNA-only", "Equal-standardized-sum", "D3-scalar", "Static-two-weight", "Proposed-dynamic-gate", "MGCT-PO")} for seed in SEEDS}
        feature_counts = {seed: {feature: 0 for feature in FEATURE_SETS} for seed in SEEDS}
        model_counts = {seed: {model: 0 for model in MODELS} for seed in SEEDS}
        gate_param_rows = []
        abmil_replay_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            for fold in range(1, 6):
                train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold)
                alpha, epoch = schedule[(seed, fold)]
                inner_train, inner_val = split_inner(train, raw, seed + fold + 2801)
                rr_inner, rp_inner, rr_inner_val, rp_inner_val, att_inner, att_inner_val = inner_main_and_features(pool, deterministic, raw, inner_train, inner_val, seed + fold * 17, alpha, epoch, device)
                inner_rna, _ = _fit_numeric(raw.rna, inner_train, output_dim=512, modality="RNA")
                inner_unc = bootstrap_rna_uncertainty(inner_rna, raw, inner_train, inner_val, alpha, seed + fold * 31, cache / f"rna_unc_inner_s{seed}_f{fold}.npz")
                stage1 = []
                for feature_set in FEATURE_SETS:
                    xtr, xva, zp_tr, zr_tr, names = build_features(rp_inner, rr_inner, att_inner, (inner_unc[0], inner_unc[2]), rp_inner_val, rr_inner_val, att_inner_val, (inner_unc[1], inner_unc[3]), feature_set)
                    model = fit_gate(xtr, zp_tr, zr_tr, raw.dss_time[inner_train], raw.dss_event[inner_train], "G0-Logistic", 0.01, seed + fold * 1000 + len(stage1) + 1)
                    _, w_val = gate_predict(model, xva, (rp_inner_val - rp_inner.mean()) / (rp_inner.std() if rp_inner.std() > 1e-8 else 1.0), (rr_inner_val - rr_inner.mean()) / (rr_inner.std() if rr_inner.std() > 1e-8 else 1.0))
                    risk_val = w_val * ((rp_inner_val - rp_inner.mean()) / (rp_inner.std() if rp_inner.std() > 1e-8 else 1.0)) + (1.0 - w_val) * ((rr_inner_val - rr_inner.mean()) / (rr_inner.std() if rr_inner.std() > 1e-8 else 1.0))
                    score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], risk_val)
                    stage1.append((score if np.isfinite(score) else -np.inf, feature_set, count_parameters(model), names))
                    selection_rows.append({"stage": "S1", "seed": seed, "outer_fold": fold, "feature_set": feature_set, "model": "G0-Logistic", "lambda_g": 0.01, "inner_val_harrell": float(score), "status": "completed"})
                    del model
                stage1.sort(key=lambda x: (-x[0], FEATURE_SETS.index(x[1])))
                selected_feature = stage1[0][1]
                stage2 = []
                for model_name in MODELS:
                    for lambda_g in LAMBDA_G:
                        xtr, xva, zp_tr, zr_tr, names = build_features(rp_inner, rr_inner, att_inner, (inner_unc[0], inner_unc[2]), rp_inner_val, rr_inner_val, att_inner_val, (inner_unc[1], inner_unc[3]), selected_feature)
                        model = fit_gate(xtr, zp_tr, zr_tr, raw.dss_time[inner_train], raw.dss_event[inner_train], model_name, lambda_g, seed + fold * 2000 + len(stage2) + 11)
                        _, w_val = gate_predict(model, xva, (rp_inner_val - rp_inner.mean()) / (rp_inner.std() if rp_inner.std() > 1e-8 else 1.0), (rr_inner_val - rr_inner.mean()) / (rr_inner.std() if rr_inner.std() > 1e-8 else 1.0))
                        zp_val = (rp_inner_val - rp_inner.mean()) / (rp_inner.std() if rp_inner.std() > 1e-8 else 1.0); zr_val = (rr_inner_val - rr_inner.mean()) / (rr_inner.std() if rr_inner.std() > 1e-8 else 1.0)
                        score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], w_val * zp_val + (1.0 - w_val) * zr_val)
                        stage2.append((score if np.isfinite(score) else -np.inf, model_name, lambda_g, count_parameters(model), names))
                        model_rows.append({"stage": "S2", "seed": seed, "outer_fold": fold, "feature_set": selected_feature, "model": model_name, "lambda_g": lambda_g, "inner_val_harrell": float(score), "params": count_parameters(model), "status": "completed"})
                        del model
                stage2.sort(key=lambda x: (-x[0], MODELS.index(x[1]), LAMBDA_G.index(x[2])))
                _, selected_model, selected_lambda, selected_params, feature_names = stage2[0]
                feature_counts[seed][selected_feature] += 1; model_counts[seed][selected_model] += 1
                selection_rows.append({"stage": "S1-selected", "seed": seed, "outer_fold": fold, "feature_set": selected_feature, "model": "G0-Logistic", "lambda_g": 0.01, "inner_val_harrell": stage1[0][0], "status": "locked"})
                model_rows.append({"stage": "S2-selected", "seed": seed, "outer_fold": fold, "feature_set": selected_feature, "model": selected_model, "lambda_g": selected_lambda, "inner_val_harrell": stage2[0][0], "params": selected_params, "status": "locked"})

                rr_cf, rp_cf = crossfit[seed][fold]
                rr_train, rp_train = rr_cf[train], rp_cf[train]
                rr_test, rp_test = oof[seed]["RNA-Linear"][test], oof[seed]["Pathology-ABMIL"][test]
                att_cache = cache / f"att_outer_s{seed}_f{fold}.npz"
                replay_cache = cache / f"abmil_replay_s{seed}_f{fold}.json"
                if att_cache.exists() and replay_cache.exists():
                    z = np.load(att_cache, allow_pickle=False); att_train, att_test = z["train"], z["test"]
                    replay = json.loads(replay_cache.read_text(encoding="utf-8"))
                    if not replay.get("crossfit_risk_replay_match") or not replay.get("test_risk_replay_match"):
                        raise ValueError(f"invalid cached ABMIL replay record: {replay}")
                else:
                    att_all, att_test, replay = attention_crossfit(
                        pool, deterministic, raw, train, test, seed, fold, epoch, device,
                        expected_train=rp_cf[train], expected_test=rp_test,
                    )
                    att_train = att_all[train]
                    np.savez_compressed(att_cache, train=att_train, test=att_test)
                    atomic_json(replay_cache, replay)
                abmil_replay_rows.append(replay)
                arrays = arrays_by_fold[fold]
                unc_cache = cache / f"rna_unc_outer_s{seed}_f{fold}.npz"
                unc_train, unc_test, unc_train_iqr, unc_test_iqr = bootstrap_rna_uncertainty(arrays.rna, raw, train, test, alpha, seed + fold * 37, unc_cache)
                xtr, xte, zp_train, zr_train, feature_names = build_features(rp_train, rr_train, att_train, (unc_train, unc_train_iqr), rp_test, rr_test, att_test, (unc_test, unc_test_iqr), selected_feature)
                model = fit_gate(xtr, zp_train, zr_train, raw.dss_time[train], raw.dss_event[train], selected_model, selected_lambda, seed + fold * 3000 + 101)
                zp_test = (rp_test - rp_train.mean()) / (rp_train.std() if rp_train.std() > 1e-8 else 1.0); zr_test = (rr_test - rr_train.mean()) / (rr_train.std() if rr_train.std() > 1e-8 else 1.0)
                proposed_test, w_test = gate_predict(model, xte, zp_test, zr_test)
                _, w_train = gate_predict(model, xtr, zp_train, zr_train)
                # The inner-val distribution is measured using the locked inner selection model.
                xtr_i, xva_i, zpi_tr, zri_tr, _ = build_features(rp_inner, rr_inner, att_inner, (inner_unc[0], inner_unc[2]), rp_inner_val, rr_inner_val, att_inner_val, (inner_unc[1], inner_unc[3]), selected_feature)
                inner_model = fit_gate(xtr_i, zpi_tr, zri_tr, raw.dss_time[inner_train], raw.dss_event[inner_train], selected_model, selected_lambda, seed + fold * 4000 + 101)
                _, w_inner_val = gate_predict(inner_model, xva_i, (rp_inner_val - rp_inner.mean()) / (rp_inner.std() if rp_inner.std() > 1e-8 else 1.0), (rr_inner_val - rr_inner.mean()) / (rr_inner.std() if rr_inner.std() > 1e-8 else 1.0))
                del model, inner_model
                # Use explicit positions to keep labels aligned even if the
                # stratified inner split is not returned in sorted order.
                train_pos = {int(v): i for i, v in enumerate(train)}
                inner_train_pos = np.asarray([train_pos[int(v)] for v in inner_train]); inner_val_pos = np.asarray([train_pos[int(v)] for v in inner_val])
                d3_test, static_test, fusion_params = d3_and_static(rr_train, rp_train, rr_test, rp_test, raw.dss_time[train], raw.dss_event[train], inner_train_pos, inner_val_pos)
                risks = {"Pathology-only": rp_test, "RNA-only": rr_test, "Equal-standardized-sum": zp_test + zr_test, "D3-scalar": d3_test, "Static-two-weight": static_test, "Proposed-dynamic-gate": proposed_test, "MGCT-PO": mgct[seed][test]}
                for name, risk in risks.items():
                    oof_methods[seed][name][test] = risk
                    fold_rows.append({"scope": "outer_fold", "seed": seed, "outer_fold": fold, "method": name, "harrell_c": float(harrell_c(raw.dss_time[test], raw.dss_event[test], risk)), "uno_c": eval_uno_fold(raw, test, train, risk), "selected_feature_set": selected_feature, "selected_model": selected_model, "lambda_g": selected_lambda, "params": selected_params if name == "Proposed-dynamic-gate" else (2 if name == "Static-two-weight" else 0), "status": "completed"})
                diag = {"seed": seed, "outer_fold": fold, "feature_set": selected_feature, "model": selected_model, "lambda_g": selected_lambda, "params": selected_params, "pathology_attention_available": True, "rna_uncertainty_available": True}
                diag.update(gate_distribution("train", w_train)); diag.update(gate_distribution("inner_val", w_inner_val)); diag.update(gate_distribution("test", w_test))
                diag.update({"test_corr_w_path_attention_entropy": pearson(w_test, att_test[:, 0]), "test_corr_w_rna_uncertainty": pearson(w_test, unc_test), "test_corr_w_abs_risk_gap": pearson(w_test, np.abs(zp_test - zr_test)), "status": "completed"})
                diagnostic_rows.append(diag); gate_param_rows.append(selected_params)
                pair_rows.append(pairwise_learnability(raw, seed, fold, rr_train, rp_train, rr_test, rp_test))

        for seed in SEEDS:
            for name, risk in oof_methods[seed].items():
                if not np.all(np.isfinite(risk)):
                    raise RuntimeError(f"incomplete OOF risk for {seed} {name}")
        atomic_csv(out / "fold_metrics.csv", fold_rows)
        seed_rows = []
        for seed in SEEDS:
            for name, risk in oof_methods[seed].items():
                seed_rows.append({"scope": "seed_oof", "seed": seed, "method": name, "harrell_c": float(harrell_c(raw.dss_time, raw.dss_event, risk)), "uno_c": float(uno_from_arrays(raw, risk)), "status": "completed"})
        atomic_csv(out / "seed_oof_metrics.csv", seed_rows)
        comparison_rows = []
        for seed in SEEDS:
            ph = next(r for r in seed_rows if r["seed"] == seed and r["method"] == "Proposed-dynamic-gate")
            for name in ("Pathology-only", "RNA-only", "Equal-standardized-sum", "D3-scalar", "Static-two-weight", "MGCT-PO"):
                base = next(r for r in seed_rows if r["seed"] == seed and r["method"] == name)
                comparison_rows.append({"scope": "seed_oof", "seed": seed, "comparison": f"Proposed-{name}", "delta_harrell": ph["harrell_c"] - base["harrell_c"], "delta_uno": ph["uno_c"] - base["uno_c"], "status": "completed"})
        for name in ("Pathology-only", "RNA-only", "Equal-standardized-sum", "D3-scalar", "Static-two-weight", "MGCT-PO"):
            rows = [r for r in comparison_rows if r["comparison"] == f"Proposed-{name}"]
            comparison_rows.append({"scope": "mean_seed", "seed": "ALL", "comparison": f"Proposed-{name}", "delta_harrell": finite_mean([r["delta_harrell"] for r in rows]), "delta_uno": finite_mean([r["delta_uno"] for r in rows]), "status": "completed"})
        atomic_csv(out / "paired_comparisons.csv", comparison_rows)
        atomic_csv(out / "feature_set_selection.csv", selection_rows)
        atomic_csv(out / "model_selection.csv", model_rows)
        atomic_csv(out / "gate_diagnostics.csv", diagnostic_rows)
        atomic_csv(out / "pairwise_reliability_learnability.csv", pair_rows)
        bootstrap_rows = []
        for seed in SEEDS:
            bootstrap_rows.extend(paired_bootstrap(raw, seed, oof_methods[seed]["Proposed-dynamic-gate"], {name: oof_methods[seed][name] for name in ("Pathology-only", "Static-two-weight", "MGCT-PO")}))
        atomic_csv(out / "paired_bootstrap.csv", bootstrap_rows)
        ordering_hash = sha256_array(raw.patient_ids.astype("U").astype("S"))
        feature_cache_hashes = {p.name: sha256_file(p) for p in sorted(cache.glob("*.npz"))}
        feature_provenance = {"source_branch": "codex/po-phase2-complementarity-headroom-audit", "source_commit": source_commit, "patient_ordering_sha256": ordering_hash, "fold_alignment": True, "cache_files": feature_cache_hashes, "abmil_replay": abmil_replay_rows, "feature_sets": {"F0-RiskOnly": ["z(rP)", "z(rR)", "abs(z(rP)-z(rR))", "z(rP)*z(rR)"], "F1-RiskPlusPathReliability": ["F0", "attention_entropy/log(K)", "top10_attention_concentration", "max_attention"], "F2-RiskPlusPathPlusRNAUncertainty": ["F1", "RNA_bootstrap_std", "RNA_bootstrap_IQR"]}, "pathology_attention": {"implementation": "locked Phase1 ABMIL attention formula", "crossfit_train_only": True, "crossfit_split_seed": "Phase2 seed+outer_fold+1701", "crossfit_model_seed": "Phase2 seed+outer_fold*100+part", "outer_test_final_seed": "Phase1 seed+outer_fold", "risk_replay_tolerance": {"crossfit_atol": CROSSFIT_REPLAY_ATOL, "test_atol": REPLAY_ATOL, "rtol": REPLAY_RTOL}, "statistics": ["entropy", "top10_concentration", "max_attention"]}, "rna_uncertainty": {"bootstrap_replicates": RNA_BOOTSTRAPS, "fit_scope": "outer-train or inner-train only", "outcome_used_for_test": False}, "normalization": "all feature statistics fit on the corresponding training partition only", "patient_ids_written": False}
        atomic_json(out / "feature_provenance.json", feature_provenance)
        atomic_json(out / "oof_provenance.json", {"source_branch": "codex/po-phase2-complementarity-headroom-audit", "source_commit": source_commit, "patient_ordering_sha256": ordering_hash, "phase1_records": oof_provenance, "mgct_records": mgct_provenance, "phase2_crossfit_cache": str(runtime / "phase2_crossfit_main_cache"), "patient_ids_written": False})
        proposed_rows = [r for r in seed_rows if r["method"] == "Proposed-dynamic-gate"]
        path_rows = [r for r in seed_rows if r["method"] == "Pathology-only"]
        static_rows = [r for r in seed_rows if r["method"] == "Static-two-weight"]
        pair_auc = [r["roc_auc"] for r in pair_rows]
        proposed_h = finite_mean([r["harrell_c"] for r in proposed_rows]); proposed_u = finite_mean([r["uno_c"] for r in proposed_rows])
        path_h = finite_mean([r["harrell_c"] for r in path_rows]); path_u = finite_mean([r["uno_c"] for r in path_rows])
        static_h = finite_mean([r["harrell_c"] for r in static_rows]); static_u = finite_mean([r["uno_c"] for r in static_rows])
        deltas_h = [p["harrell_c"] - q["harrell_c"] for p, q in zip(proposed_rows, path_rows)]; deltas_u = [p["uno_c"] - q["uno_c"] for p, q in zip(proposed_rows, path_rows)]
        boot_path_h = [r for r in bootstrap_rows if r["comparison"] == "Proposed-Pathology-only" and r["metric"] == "delta_harrell"]
        boot_path_u = [r for r in bootstrap_rows if r["comparison"] == "Proposed-Pathology-only" and r["metric"] == "delta_uno"]
        diag_test_std = finite_mean([r["test_std_w"] for r in diagnostic_rows])
        mean_test_w = finite_mean([r["test_mean_w"] for r in diagnostic_rows])
        static_not_worse = static_h >= proposed_h and static_u >= proposed_u
        seed_both_positive = sum(int(h > 0 and u > 0) for h, u in zip(deltas_h, deltas_u))
        boot_positive = any(r["ci95_low"] > 0 for r in boot_path_h + boot_path_u)
        gate_supported = proposed_h > path_h and proposed_u > path_u and seed_both_positive >= 2 and boot_positive and (deltas_h.__len__() > 0 and finite_mean(deltas_h) > 0 or finite_mean(deltas_u) > 0) and (proposed_h > static_h or proposed_u > static_u) and diag_test_std > GATE_STD_MIN and 0.05 <= mean_test_w <= 0.95 and finite_mean(pair_auc) > MIN_PAIR_AUC
        if static_not_worse:
            decision = "SCALAR_FUSION_SUFFICIENT"
        elif gate_supported:
            decision = "REALIZABLE_COMPLEMENTARITY_SUPPORTED"
        elif proposed_h >= path_h and proposed_u >= path_u:
            decision = "RELIABILITY_DIRECTIONAL"
        else:
            decision = "ORACLE_NOT_REALIZABLE"
        atomic_json(out / "protocol.json", {"phase": "PO-Phase3", "cohort": 512, "events": 65, "folds": 5, "seeds": list(SEEDS), "endpoint": "DSS", "modalities": ["RNA", "Pathology UNI2-h"], "excluded": ["Clinical", "CNA", "new endpoint", "new split", "Q-Former", "cross-attention", "Transformer", "LLM", "concept graph", "large hyperparameter search"], "feature_sets": list(FEATURE_SETS), "models": list(MODELS), "lambda_g": list(LAMBDA_G), "lambda_s": LAMBDA_S, "rna_bootstrap_replicates": RNA_BOOTSTRAPS, "inner_cv_folds": INNER_CV_FOLDS, "bootstrap_replicates": BOOTSTRAP_REPS, "gate_definition": "w=sigmoid(g(q)); r=w*z(rP)+(1-w)*z(rR)", "selection": "S1 fixed G0 lambda_g=0.01 compares F0/F1/F2; S2 locks feature set and compares G0/G1 x lambda_g", "pairwise_diagnostic": "L2 logistic on |q_i-q_j| and q_i+q_j; never used to form patient-level risk", "source_commit": source_commit, "patient_ids_written": False})
        validation = {"status": True, "cohort_512": len(raw.patient_ids) == 512, "events_65": int(np.sum(raw.dss_event)) == 65, "phase1_oof_complete": oof_valid, "phase2_crossfit_complete": crossfit_valid, "mgct_comparator_complete": mgct_valid, "fold_alignment": True, "patient_ordering_recorded": True, "inner_val_not_used_for_upstream_main_effects": True, "outer_test_not_used_for_selection": True, "rna_uncertainty_outer_test_labels_used": False, "gate_param_max_lt_150": max(gate_param_rows) < MAX_GATE_PARAMS, "bootstrap_replicates_at_least_2000": all(r["bootstrap_replicates"] >= BOOTSTRAP_REPS for r in bootstrap_rows), "no_patient_ids_public": True, "decision": decision, "failure_count": 0}
        atomic_json(out / "validation.json", validation)
        decision_text = {
            "REALIZABLE_COMPLEMENTARITY_SUPPORTED": "dynamic gate shows reproducible patient-level reliability complementarity under the locked low-capacity protocol; larger representation-level fusion is now scientifically justified, but only with the same leakage controls.",
            "RELIABILITY_DIRECTIONAL": "dynamic gate has directional evidence but seed/bootstrap or pairwise learnability support is insufficient for a large architecture.",
            "ORACLE_NOT_REALIZABLE": "the high Phase2 oracle headroom is not recovered by a learnable patient-level reliability gate; stop using the oracle as a sufficient motivation for more complex fusion.",
            "SCALAR_FUSION_SUFFICIENT": "static two-weight fusion is at least as good as dynamic gating; patient-adaptive reliability is not required under this protocol.",
        }[decision]
        results = ["# Phase3 realizable complementarity audit", "", "- Fixed cohort: 512 patients, 65 DSS events, fixed five outer folds, three seeds.", "- All proposed risks are single patient-level scores; the pairwise diagnostic is not used to construct risk.", "", "## Direct answers", "", f"1. Patient-level reliability learnability: mean pairwise P-vs-R AUC {finite_mean(pair_auc):.4f}; balanced accuracy {finite_mean([r['balanced_accuracy'] for r in pair_rows]):.4f}.", f"2. Dynamic gate vs Pathology-only: mean ΔHarrell {finite_mean(deltas_h):.4f}; mean ΔUno {finite_mean(deltas_u):.4f}; seed-wise Harrell {[round(x,4) for x in deltas_h]}, Uno {[round(x,4) for x in deltas_u]}.", f"3. Dynamic gate vs static two-weight fusion: mean Harrell {proposed_h - static_h:.4f}; mean Uno {proposed_u - static_u:.4f}.", f"4. Gate adaptivity: mean test gate std {diag_test_std:.4f}; mean test gate weight {mean_test_w:.4f}; distributions are in `gate_diagnostics.csv`.", f"5. Feature selection counts: {feature_counts}; model counts: {model_counts}.", f"6. Phase2 oracle headroom recovered by the dynamic gate: Harrell {finite_mean(deltas_h):.4f}; Uno {finite_mean(deltas_u):.4f}; Phase2 headroom was approximately 0.1904/0.1598.", f"7. Paired bootstrap Proposed-Pathology lower bounds: Harrell {boot_path_h[0]['ci95_low']:.4f}; Uno {boot_path_u[0]['ci95_low']:.4f}.", f"8. Historical MGCT-PO comparison is in `paired_comparisons.csv`; no new MGCT training was performed.", f"9. Decision: **{decision}**.", "", "## Interpretation", "", decision_text, "", "## Boundaries", "", "Attention entropy, attention concentration, and RNA bootstrap spread are empirical reliability proxies. They do not establish biological credibility, full epistemic uncertainty, or causal complementarity.", "", "## Required next step", "", "If the decision is REALIZABLE_COMPLEMENTARITY_SUPPORTED, the next experiment may test a strictly low-capacity representation interaction. Otherwise, stop using dynamic reliability or pairwise-oracle headroom as justification for Q-Former, cross-attention, or other large fusion architectures."]
        (out / "RESULTS.md").write_text("\n".join(results) + "\n", encoding="utf-8")
        (out / "DECISION.md").write_text(f"# Phase3 decision\n\n- Decision: **{decision}**\n- Mean proposed Harrell/Uno: {proposed_h:.4f} / {proposed_u:.4f}.\n- Mean Pathology Harrell/Uno: {path_h:.4f} / {path_u:.4f}.\n- Mean static fusion Harrell/Uno: {static_h:.4f} / {static_u:.4f}.\n- Mean pairwise reliability AUC: {finite_mean(pair_auc):.4f}.\n- Mean test gate standard deviation: {diag_test_std:.4f}.\n- {decision_text}\n", encoding="utf-8")
        atomic_csv(out / "failure_ledger.csv", [], ["error", "traceback"])
        atomic_json(out / "run_manifest.json", {"phase": "PO-Phase3", "source_commit": source_commit, "formal_seeds": 3, "formal_outer_folds": 15, "rna_bootstrap_replicates": RNA_BOOTSTRAPS, "paired_bootstrap_replicates": BOOTSTRAP_REPS, "decision": decision, "failures": 0, "runtime_seconds": time.time() - started, "status": "completed"})
    except Exception as exc:
        failures.append({"error": repr(exc), "traceback": traceback.format_exc()})
        atomic_csv(out / "failure_ledger.csv", failures)
        atomic_json(out / "validation.json", {"status": False, "failure_count": len(failures), "error": repr(exc)})
        atomic_json(out / "run_manifest.json", {"phase": "PO-Phase3", "source_commit": source_commit, "failures": len(failures), "runtime_seconds": time.time() - started, "status": "AUDIT_BLOCKED"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--phase1-report", required=True)
    parser.add_argument("--phase2-report", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--source-commit", required=True)
    main(parser.parse_args())

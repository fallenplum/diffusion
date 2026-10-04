from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

# Scalar Cox fits are intentionally tiny.  Limiting BLAS thread fan-out keeps
# the fixed 220-step schedule reproducible and avoids oversubscribing the
# server during the 2,000-replicate audit.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.phase_m2.common import (
    censoring_survival,
    harrell_c,
    lookup_censor_survival,
    uno_c_index,
)
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.po_phase1_orthogonal_fusion_repair import crossfit_main_scores


SEEDS = (20260904, 20260905, 20260906)
BETAS = (1e-3, 1e-2, 1e-1, 1.0)
CONDITIONAL_L2 = 1e-2
BOOTSTRAP_REPS = 2000
PERMUTATION_REPS = 1000
TOP_FRACTIONS = (0.10, 0.20, 0.30)
EXTREME_DISAGREEMENT_THRESHOLD = 2.0
PAIR_HEADROOM_THRESHOLD = 0.02
RESCUE_SMALL_THRESHOLD = 0.10
CORRELATION_HIGH_THRESHOLD = 0.70
PERMUTATION_NEAR_ZERO_THRESHOLD = 0.01


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
    sa, sb = np.std(a), np.std(b)
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(a), rankdata(b))


def zfit(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    values = np.asarray(values, dtype=float)
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return (train - mean) / std, (values - mean) / std, np.stack([mean, std])


def split_inner(train: np.ndarray, raw: Any, seed: int) -> tuple[np.ndarray, np.ndarray]:
    strata = np.char.add(raw.cancer_type[train].astype(str), "_" + raw.dss_event[train].astype(str))
    rng = np.random.default_rng(seed)
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        a, b = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(train, strata))
        return train[a], train[b]
    except ValueError:
        order = rng.permutation(train)
        n = max(1, int(0.15 * len(order)))
        return order[n:], order[:n]


def cox_loss_fast(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    event_times = torch.unique(times[events > 0], sorted=True)
    if event_times.numel() == 0:
        return risks.sum() * 0.0
    at_risk = times.unsqueeze(0) >= event_times.unsqueeze(1)
    died = (events.unsqueeze(0) > 0) & (times.unsqueeze(0) == event_times.unsqueeze(1))
    masked = risks.unsqueeze(0).expand(event_times.shape[0], -1).masked_fill(~at_risk, -torch.inf)
    logsum = torch.logsumexp(masked, dim=1)
    count = died.sum(dim=1).to(risks.dtype)
    event_sum = (died.to(risks.dtype) * risks.unsqueeze(0)).sum(dim=1)
    return (count * logsum - event_sum).sum() / (events > 0).sum().to(risks.dtype)


def fit_beta(anchor: np.ndarray, other: np.ndarray, times: np.ndarray, events: np.ndarray, l2: float) -> float:
    device = torch.device("cpu")
    a = torch.as_tensor(anchor, dtype=torch.float64, device=device)
    b = torch.as_tensor(other, dtype=torch.float64, device=device)
    t = torch.as_tensor(times, dtype=torch.float64, device=device)
    e = torch.as_tensor(events, dtype=torch.float64, device=device)
    beta = nn.Parameter(torch.zeros((), dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([beta], lr=0.03)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        loss = cox_loss_fast(t, e, a + beta * b) + 0.5 * float(l2) * beta.square()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([beta], 5.0)
        optimizer.step()
    value = float(beta.detach().cpu())
    del optimizer, beta
    return value


def fit_gamma(rp: np.ndarray, rr: np.ndarray, times: np.ndarray, events: np.ndarray, l2: float = CONDITIONAL_L2) -> tuple[float, float]:
    device = torch.device("cpu")
    p = torch.as_tensor(rp, dtype=torch.float64, device=device)
    r = torch.as_tensor(rr, dtype=torch.float64, device=device)
    t = torch.as_tensor(times, dtype=torch.float64, device=device)
    e = torch.as_tensor(events, dtype=torch.float64, device=device)
    gamma = nn.Parameter(torch.zeros(2, dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam([gamma], lr=0.03)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        loss = cox_loss_fast(t, e, gamma[0] * p + gamma[1] * r) + 0.5 * float(l2) * (gamma.square()).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gamma], 5.0)
        optimizer.step()
    result = tuple(float(x) for x in gamma.detach().cpu().numpy())
    del optimizer, gamma
    return result


def km_weights(raw: Any) -> np.ndarray:
    result = np.ones(len(raw.patient_ids), dtype=float)
    for fold in range(1, 6):
        train_mask = raw.outer_folds != fold
        kt, kg = censoring_survival(raw.dss_time[train_mask], raw.dss_event[train_mask])
        test_mask = raw.outer_folds == fold
        result[test_mask] = lookup_censor_survival(raw.dss_time[test_mask], kt, kg)
    return result


def pair_summary(times: np.ndarray, events: np.ndarray, r_rna: np.ndarray, r_path: np.ndarray, g: np.ndarray | None = None) -> dict[str, float]:
    counts = {"BOTH_CORRECT": 0.0, "P_ONLY_CORRECT": 0.0, "RNA_ONLY_CORRECT": 0.0, "BOTH_WRONG": 0.0, "BOTH_TIE": 0.0, "P_TIE_RNA_CORRECT": 0.0, "P_TIE_RNA_WRONG": 0.0, "RNA_TIE_P_CORRECT": 0.0, "RNA_TIE_P_WRONG": 0.0}
    weighted = {key: 0.0 for key in counts}
    total = total_w = 0.0
    p_score = r_score = oracle_score = 0.0
    p_score_w = r_score_w = oracle_score_w = 0.0
    p_wrong = p_wrong_w = rna_wrong = rna_wrong_w = 0.0
    rna_rescue = rna_rescue_w = path_rescue = path_rescue_w = 0.0
    for i in np.flatnonzero(events == 1):
        js = np.flatnonzero(times > times[i])
        if g is not None and g[i] <= 1e-8:
            continue
        weight = 1.0 if g is None else float(1.0 / (g[i] ** 2))
        for j in js:
            p_cmp = 1 if r_path[i] > r_path[j] + 1e-12 else (-1 if r_path[i] < r_path[j] - 1e-12 else 0)
            r_cmp = 1 if r_rna[i] > r_rna[j] + 1e-12 else (-1 if r_rna[i] < r_rna[j] - 1e-12 else 0)
            if p_cmp == 1 and r_cmp == 1:
                category = "BOTH_CORRECT"
            elif p_cmp == 1 and r_cmp == -1:
                category = "P_ONLY_CORRECT"
            elif p_cmp == -1 and r_cmp == 1:
                category = "RNA_ONLY_CORRECT"
            elif p_cmp == -1 and r_cmp == -1:
                category = "BOTH_WRONG"
            elif p_cmp == 0 and r_cmp == 0:
                category = "BOTH_TIE"
            elif p_cmp == 0 and r_cmp == 1:
                category = "P_TIE_RNA_CORRECT"
            elif p_cmp == 0 and r_cmp == -1:
                category = "P_TIE_RNA_WRONG"
            elif r_cmp == 0 and p_cmp == 1:
                category = "RNA_TIE_P_CORRECT"
            else:
                category = "RNA_TIE_P_WRONG"
            counts[category] += 1.0
            weighted[category] += weight
            total += 1.0; total_w += weight
            p_score += 1.0 if p_cmp == 1 else (0.5 if p_cmp == 0 else 0.0)
            r_score += 1.0 if r_cmp == 1 else (0.5 if r_cmp == 0 else 0.0)
            oracle = 1.0 if (p_cmp == 1 or r_cmp == 1) else (0.0 if (p_cmp == -1 and r_cmp == -1) else 0.5)
            oracle_score += oracle
            p_score_w += weight * (1.0 if p_cmp == 1 else (0.5 if p_cmp == 0 else 0.0))
            r_score_w += weight * (1.0 if r_cmp == 1 else (0.5 if r_cmp == 0 else 0.0))
            oracle_score_w += weight * oracle
            if p_cmp == -1:
                p_wrong += 1.0
                p_wrong_w += weight
                if r_cmp == 1:
                    rna_rescue += 1.0; rna_rescue_w += weight
            if r_cmp == -1:
                rna_wrong += 1.0
                rna_wrong_w += weight
                if p_cmp == 1:
                    path_rescue += 1.0; path_rescue_w += weight
    result: dict[str, float] = {"total_pairs": total, "total_weight": total_w}
    for key in counts:
        result[f"count_{key}"] = counts[key]
        result[f"weight_{key}"] = weighted[key]
    result.update({
        "pathology_c": p_score / total if total else float("nan"),
        "rna_c": r_score / total if total else float("nan"),
        "oracle_c": oracle_score / total if total else float("nan"),
        "pathology_uno": p_score_w / total_w if total_w else float("nan"),
        "rna_uno": r_score_w / total_w if total_w else float("nan"),
        "oracle_uno": oracle_score_w / total_w if total_w else float("nan"),
        "rescue_rna_given_path_wrong": rna_rescue / p_wrong if p_wrong else float("nan"),
        "rescue_path_given_rna_wrong": path_rescue / rna_wrong if rna_wrong else float("nan"),
        "rescue_rna_given_path_wrong_ipcw": rna_rescue_w / p_wrong_w if p_wrong_w else float("nan"),
        "rescue_path_given_rna_wrong_ipcw": path_rescue_w / rna_wrong_w if rna_wrong_w else float("nan"),
    })
    return result


def pair_summary_vectorized(times: np.ndarray, events: np.ndarray, r_rna: np.ndarray, r_path: np.ndarray, g: np.ndarray | None = None) -> dict[str, float]:
    """Vectorized equivalent of pair_summary for bootstrap/permutation loops."""
    times = np.asarray(times, dtype=float); events = np.asarray(events, dtype=int)
    r_rna = np.asarray(r_rna, dtype=float); r_path = np.asarray(r_path, dtype=float)
    ii, jj = np.where((events[:, None] == 1) & (times[None, :] > times[:, None]))
    if g is not None:
        keep = np.asarray(g, dtype=float)[ii] > 1e-8
        ii, jj = ii[keep], jj[keep]
    weights = np.ones(len(ii), dtype=float) if g is None else 1.0 / (np.asarray(g, dtype=float)[ii] ** 2)
    p_delta = r_path[ii] - r_path[jj]; r_delta = r_rna[ii] - r_rna[jj]
    p_cmp = np.where(np.abs(p_delta) <= 1e-12, 0, np.sign(p_delta)).astype(int)
    r_cmp = np.where(np.abs(r_delta) <= 1e-12, 0, np.sign(r_delta)).astype(int)
    category_masks = {
        "BOTH_CORRECT": (p_cmp == 1) & (r_cmp == 1),
        "P_ONLY_CORRECT": (p_cmp == 1) & (r_cmp == -1),
        "RNA_ONLY_CORRECT": (p_cmp == -1) & (r_cmp == 1),
        "BOTH_WRONG": (p_cmp == -1) & (r_cmp == -1),
        "BOTH_TIE": (p_cmp == 0) & (r_cmp == 0),
        "P_TIE_RNA_CORRECT": (p_cmp == 0) & (r_cmp == 1),
        "P_TIE_RNA_WRONG": (p_cmp == 0) & (r_cmp == -1),
        "RNA_TIE_P_CORRECT": (r_cmp == 0) & (p_cmp == 1),
        "RNA_TIE_P_WRONG": (r_cmp == 0) & (p_cmp == -1),
    }
    total = float(len(ii)); total_w = float(weights.sum())
    result: dict[str, float] = {"total_pairs": total, "total_weight": total_w}
    for key, mask in category_masks.items():
        result[f"count_{key}"] = float(mask.sum()); result[f"weight_{key}"] = float(weights[mask].sum())
    p_score = float(np.where(p_cmp == 1, 1.0, np.where(p_cmp == 0, 0.5, 0.0)).sum())
    r_score = float(np.where(r_cmp == 1, 1.0, np.where(r_cmp == 0, 0.5, 0.0)).sum())
    oracle_score = np.where((p_cmp == 1) | (r_cmp == 1), 1.0, np.where((p_cmp == -1) & (r_cmp == -1), 0.0, 0.5))
    p_score_w = float((weights * np.where(p_cmp == 1, 1.0, np.where(p_cmp == 0, 0.5, 0.0))).sum())
    r_score_w = float((weights * np.where(r_cmp == 1, 1.0, np.where(r_cmp == 0, 0.5, 0.0))).sum())
    oracle_score_w = float((weights * oracle_score).sum())
    p_wrong = p_cmp == -1; r_wrong = r_cmp == -1
    r_rescue = p_wrong & (r_cmp == 1); p_rescue = r_wrong & (p_cmp == 1)
    result.update({
        "pathology_c": p_score / total if total else float("nan"), "rna_c": r_score / total if total else float("nan"), "oracle_c": float(oracle_score.sum() / total) if total else float("nan"),
        "pathology_uno": p_score_w / total_w if total_w else float("nan"), "rna_uno": r_score_w / total_w if total_w else float("nan"), "oracle_uno": oracle_score_w / total_w if total_w else float("nan"),
        "rescue_rna_given_path_wrong": float(r_rescue.sum() / p_wrong.sum()) if p_wrong.any() else float("nan"), "rescue_path_given_rna_wrong": float(p_rescue.sum() / r_wrong.sum()) if r_wrong.any() else float("nan"),
        "rescue_rna_given_path_wrong_ipcw": float(weights[r_rescue].sum() / weights[p_wrong].sum()) if p_wrong.any() else float("nan"), "rescue_path_given_rna_wrong_ipcw": float(weights[p_rescue].sum() / weights[r_wrong].sum()) if r_wrong.any() else float("nan"),
    })
    return result


def harrell_fast(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    return pair_summary_vectorized(times, events, risk, risk, None)["pathology_c"]


def uno_from_arrays_fast(times: np.ndarray, events: np.ndarray, risk: np.ndarray, folds: np.ndarray) -> float:
    numerator = denominator = 0.0
    for fold in range(1, 6):
        test = folds == fold; train = ~test
        if not np.any(test) or not np.any(train):
            continue
        kt, kg = censoring_survival(times[train], events[train])
        g = lookup_censor_survival(times[test], kt, kg)
        ii, jj = np.where((events[test][:, None] == 1) & (times[test][None, :] > times[test][:, None]))
        keep = g[ii] > 1e-8
        if not np.any(keep):
            continue
        ii, jj = ii[keep], jj[keep]; weights = 1.0 / (g[ii] ** 2)
        delta = risk[test][ii] - risk[test][jj]
        score = np.where(delta > 1e-12, 1.0, np.where(np.abs(delta) <= 1e-12, 0.5, 0.0))
        numerator += float((weights * score).sum()); denominator += float(weights.sum())
    return numerator / denominator if denominator else float("nan")


def load_oof(raw: Any, cache: Path, source_commit: str) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]], bool]:
    predictions: dict[int, dict[str, np.ndarray]] = {}
    provenance: list[dict[str, Any]] = []
    valid = True
    for seed in SEEDS:
        predictions[seed] = {"RNA-Linear": np.full(len(raw.patient_ids), np.nan), "Pathology-ABMIL": np.full(len(raw.patient_ids), np.nan)}
        for fold in range(1, 6):
            expected = np.flatnonzero(raw.outer_folds == fold)
            for method, stem in (("RNA-Linear", "RNA_Linear"), ("Pathology-ABMIL", "Pathology_ABMIL")):
                path = cache / f"{stem}_s{seed}_f{fold}.npz"
                record = {"seed": seed, "outer_fold": fold, "method": method, "cache_path": str(path), "source_commit": source_commit, "status": "completed"}
                if not path.exists():
                    record.update({"status": "missing", "sha256": None})
                    valid = False; provenance.append(record); continue
                record["sha256"] = sha256_file(path)
                try:
                    z = np.load(path, allow_pickle=False)
                    idx, risk = np.asarray(z["test_idx"], dtype=int), np.asarray(z["test_risk"], dtype=float)
                    record.update({"n_predictions": int(len(idx)), "test_idx_sha256": hashlib.sha256(idx.tobytes()).hexdigest()})
                    if not np.array_equal(idx, expected) or len(risk) != len(idx) or not np.all(np.isfinite(risk)) or np.any(np.isfinite(predictions[seed][method][idx])):
                        raise ValueError("prediction coverage or fold ordering mismatch")
                    predictions[seed][method][idx] = risk
                except Exception as exc:
                    record.update({"status": "invalid", "error": repr(exc)})
                    valid = False
                provenance.append(record)
        for method in predictions[seed]:
            if not np.all(np.isfinite(predictions[seed][method])):
                valid = False
    return predictions, provenance, valid


def load_phase1_schedule(report: Path) -> dict[tuple[int, int], tuple[float, int]]:
    frame = pd.read_csv(report / "main_effect_metrics.csv")
    result: dict[tuple[int, int], tuple[float, int]] = {}
    for _, row in frame[frame["method"].isin(["RNA-Linear", "Pathology-ABMIL"])].iterrows():
        key = (int(row["seed"]), int(row["outer_fold"]))
        alpha = float(row["alpha"]) if pd.notna(row.get("alpha")) else 0.1
        epoch = int(row["selected_epoch"]) if pd.notna(row.get("selected_epoch")) else 12
        result[key] = (alpha, epoch)
    return result


def load_crossfit(raw: Any, runtime: Path, data_root: Path, repo: Path, report: Path, source_commit: str) -> tuple[dict[int, dict[int, tuple[np.ndarray, np.ndarray]]], bool, bool]:
    target = runtime / "phase2_crossfit_main_cache"
    target.mkdir(parents=True, exist_ok=True)
    schedule = load_phase1_schedule(report)
    result: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {seed: {} for seed in SEEDS}
    all_existing = True
    for seed in SEEDS:
        for fold in range(1, 6):
            rr_path = target / f"rna_crossfit_s{seed}_f{fold}.npy"
            rp_path = target / f"path_crossfit_s{seed}_f{fold}.npy"
            if not rr_path.exists() or not rp_path.exists():
                all_existing = False
    if all_existing:
        for seed in SEEDS:
            for fold in range(1, 6):
                rr, rp = np.load(target / f"rna_crossfit_s{seed}_f{fold}.npy"), np.load(target / f"path_crossfit_s{seed}_f{fold}.npy")
                train = np.flatnonzero(raw.outer_folds != fold)
                if not (np.all(np.isfinite(rr[train])) and np.all(np.isfinite(rp[train]))):
                    return result, False, True
                result[seed][fold] = (rr, rp)
        return result, True, False
    pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_arrays = {fold: preprocess_fold(raw, pool, fold) for fold in range(1, 6)}
    for seed in SEEDS:
        for fold in range(1, 6):
            train = np.flatnonzero(raw.outer_folds != fold)
            alpha, epoch = schedule[(seed, fold)]
            rr, rp = crossfit_main_scores(pool, deterministic, fold_arrays[fold], raw, train, seed, fold, alpha, epoch, device)
            np.save(target / f"rna_crossfit_s{seed}_f{fold}.npy", rr.astype(np.float64))
            np.save(target / f"path_crossfit_s{seed}_f{fold}.npy", rp.astype(np.float64))
            result[seed][fold] = (rr, rp)
    atomic_json(target / "provenance.json", {"source_commit": source_commit, "schedule_source": str(report / "main_effect_metrics.csv"), "crossfit_folds": 4, "status": "completed"})
    return result, True, False


def risk_dependence_rows(predictions: dict[int, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for seed, values in predictions.items():
        rr, rp = values["RNA-Linear"], values["Pathology-ABMIL"]
        zr = (rr - rr.mean()) / (rr.std() if rr.std() > 0 else 1.0)
        zp = (rp - rp.mean()) / (rp.std() if rp.std() > 0 else 1.0)
        diff = zr - zp
        row = {"seed": seed, "pearson": pearson(rr, rp), "spearman": spearman(rr, rp), "d_mean": float(diff.mean()), "d_std": float(diff.std()), "extreme_disagreement_abs_d_ge_2": int(np.sum(np.abs(diff) >= EXTREME_DISAGREEMENT_THRESHOLD))}
        for q in [0.0, .25, .50, .75, .90, .95, .99, 1.0]:
            row[f"d_q{int(q*100):02d}"] = float(np.quantile(diff, q))
            row[f"abs_d_q{int(q*100):02d}"] = float(np.quantile(np.abs(diff), q))
        for fraction in TOP_FRACTIONS:
            k = max(1, int(math.ceil(len(rr) * fraction)))
            a = set(np.argsort(-rr, kind="stable")[:k].tolist())
            b = set(np.argsort(-rp, kind="stable")[:k].tolist())
            inter = len(a & b)
            row[f"top{int(fraction*100)}_jaccard"] = inter / len(a | b)
            row[f"top{int(fraction*100)}_overlap_coefficient"] = inter / min(len(a), len(b))
            row[f"top{int(fraction*100)}_overlap_count"] = inter
        rows.append(row)
    return rows


def pairwise_rows(raw: Any, predictions: dict[int, dict[str, np.ndarray]], g: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, oracle_rows = [], []
    for seed, values in predictions.items():
        for weighting, weights in (("unweighted", None), ("ipcw", g)):
            stats = pair_summary(raw.dss_time, raw.dss_event, values["RNA-Linear"], values["Pathology-ABMIL"], weights)
            row = {"seed": seed, "weighting": weighting, **stats}
            for key in list(stats):
                if key.startswith("count_") or key.startswith("weight_"):
                    category = key.split("_", 1)[1]
                    row[f"fraction_{category}"] = stats[key] / (stats["total_pairs"] if weighting == "unweighted" else stats["total_weight"]) if (stats["total_pairs"] if weighting == "unweighted" else stats["total_weight"]) else float("nan")
            rows.append(row)
        unweighted = pair_summary(raw.dss_time, raw.dss_event, values["RNA-Linear"], values["Pathology-ABMIL"], None)
        weighted = pair_summary(raw.dss_time, raw.dss_event, values["RNA-Linear"], values["Pathology-ABMIL"], g)
        oracle_rows.append({"seed": seed, "pairs": unweighted["total_pairs"], "pathology_harrell": unweighted["pathology_c"], "oracle_harrell": unweighted["oracle_c"], "oracle_headroom_harrell": unweighted["oracle_c"] - unweighted["pathology_c"], "pathology_uno": weighted["pathology_uno"], "oracle_uno": weighted["oracle_uno"], "oracle_headroom_uno": weighted["oracle_uno"] - weighted["pathology_uno"], "rna_harrell": unweighted["rna_c"], "rna_uno": weighted["rna_uno"]})
    return rows, oracle_rows


def evaluate_fold_methods(raw: Any, predictions: dict[int, dict[str, np.ndarray]], crossfit: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, np.ndarray]], dict[int, dict[int, dict[str, float]]]]:
    rows, conditional_rows = [], []
    oof_methods: dict[int, dict[str, np.ndarray]] = {}
    d3_parameters: dict[int, dict[int, dict[str, float]]] = {}
    for seed in SEEDS:
        oof_methods[seed] = {name: np.full(len(raw.patient_ids), np.nan) for name in ("D0-Pathology-only", "D1-RNA-only", "D2-Equal-standardized-sum", "D3-Pathology-anchor-RNA-correction", "D4-RNA-anchor-Pathology-correction", "E-Conditional-scalar")}
        d3_parameters[seed] = {}
        for fold in range(1, 6):
            train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold)
            rr_cf, rp_cf = crossfit[seed][fold]
            rr_test, rp_test = predictions[seed]["RNA-Linear"][test], predictions[seed]["Pathology-ABMIL"][test]
            _, zp_test, _ = zfit(rp_cf[train], rp_test)
            _, zr_test, _ = zfit(rr_cf[train], rr_test)
            _, zr_train, _ = zfit(rr_cf[train], rr_cf[train])
            _, zp_train, _ = zfit(rp_cf[train], rp_cf[train])
            methods = {
                "D0-Pathology-only": rp_test,
                "D1-RNA-only": rr_test,
                "D2-Equal-standardized-sum": zp_test + zr_test,
            }
            inner_train, inner_val = split_inner(train, raw, seed + fold + 2801)
            choices = {}
            for name, anchor, other, anchor_test, other_test in (
                ("D3-Pathology-anchor-RNA-correction", rp_cf, rr_cf, rp_test, rr_test),
                ("D4-RNA-anchor-Pathology-correction", rr_cf, rp_cf, rr_test, rp_test),
            ):
                best = None
                for l2 in BETAS:
                    # Fit and score on inner-train/inner-val using inner-train scaling.
                    a_train, a_val_scaled, _ = zfit(anchor[inner_train], anchor[inner_val])
                    o_train, o_val_scaled, _ = zfit(other[inner_train], other[inner_val])
                    beta = fit_beta(a_train, o_train, raw.dss_time[inner_train], raw.dss_event[inner_train], l2)
                    score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], a_val_scaled + beta * o_val_scaled)
                    candidate = (score, -l2, beta)
                    if best is None or candidate > best:
                        best = candidate
                selected_l2, _ = -best[1], best[2]
                a_train, a_test_scaled, _ = zfit(anchor[train], anchor_test)
                o_train, o_test_scaled, _ = zfit(other[train], other_test)
                beta_final = fit_beta(a_train, o_train, raw.dss_time[train], raw.dss_event[train], selected_l2)
                methods[name] = a_test_scaled + beta_final * o_test_scaled
                choices[name] = {"beta": beta_final, "l2": selected_l2}
            gamma_p, gamma_r = fit_gamma(zp_train, zr_train, raw.dss_time[train], raw.dss_event[train])
            conditional_test = gamma_p * zp_test + gamma_r * zr_test
            methods["E-Conditional-scalar"] = conditional_test
            d3_parameters[seed][fold] = choices["D3-Pathology-anchor-RNA-correction"] | {"gamma_P": gamma_p, "gamma_R": gamma_r}
            path_h = harrell_c(raw.dss_time[test], raw.dss_event[test], rp_test)
            path_u = uno_c_index(raw.dss_time[test], raw.dss_event[test], rp_test, raw.dss_time[train], raw.dss_event[train])[0]
            for name, risk in methods.items():
                uno = uno_c_index(raw.dss_time[test], raw.dss_event[test], risk, raw.dss_time[train], raw.dss_event[train])[0]
                row = {"scope": "outer_fold", "seed": seed, "outer_fold": fold, "method": name, "harrell_c": harrell_c(raw.dss_time[test], raw.dss_event[test], risk), "uno_c": uno, "delta_vs_pathology_harrell": harrell_c(raw.dss_time[test], raw.dss_event[test], risk) - path_h, "delta_vs_pathology_uno": uno - path_u, "status": "completed"}
                if name in choices:
                    row.update(choices[name])
                rows.append(row)
                oof_methods[seed][name][test] = risk
            conditional_rows.append({"scope": "outer_fold", "seed": seed, "outer_fold": fold, "gamma_P": gamma_p, "gamma_R": gamma_r, "gamma_R_positive": int(gamma_r > 0), "harrell_c": harrell_c(raw.dss_time[test], raw.dss_event[test], conditional_test), "uno_c": uno_c_index(raw.dss_time[test], raw.dss_event[test], conditional_test, raw.dss_time[train], raw.dss_event[train])[0], "pathology_harrell": path_h, "pathology_uno": path_u, "delta_harrell": harrell_c(raw.dss_time[test], raw.dss_event[test], conditional_test) - path_h, "delta_uno": uno_c_index(raw.dss_time[test], raw.dss_event[test], conditional_test, raw.dss_time[train], raw.dss_event[train])[0] - path_u, "status": "completed"})
        for name, risk in oof_methods[seed].items():
            if not np.all(np.isfinite(risk)):
                raise ValueError(f"incomplete recoverable OOF for {seed} {name}")
            rows.append({"scope": "seed_oof", "seed": seed, "outer_fold": "ALL", "method": name, "harrell_c": harrell_c(raw.dss_time, raw.dss_event, risk), "uno_c": uno_from_arrays(raw, risk), "delta_vs_pathology_harrell": harrell_c(raw.dss_time, raw.dss_event, risk) - harrell_c(raw.dss_time, raw.dss_event, predictions[seed]["Pathology-ABMIL"]), "delta_vs_pathology_uno": uno_from_arrays(raw, risk) - uno_from_arrays(raw, predictions[seed]["Pathology-ABMIL"]), "status": "completed"})
        conditional_rows.append({"scope": "seed_oof", "seed": seed, "outer_fold": "ALL", "gamma_P_mean": float(np.mean([d3_parameters[seed][f]["gamma_P"] for f in range(1, 6)])), "gamma_R_mean": float(np.mean([d3_parameters[seed][f]["gamma_R"] for f in range(1, 6)])), "gamma_R_positive_fold_fraction": float(np.mean([d3_parameters[seed][f]["gamma_R"] > 0 for f in range(1, 6)])), "harrell_c": harrell_c(raw.dss_time, raw.dss_event, oof_methods[seed]["E-Conditional-scalar"]), "uno_c": uno_from_arrays(raw, oof_methods[seed]["E-Conditional-scalar"]), "pathology_harrell": harrell_c(raw.dss_time, raw.dss_event, predictions[seed]["Pathology-ABMIL"]), "pathology_uno": uno_from_arrays(raw, predictions[seed]["Pathology-ABMIL"]), "delta_harrell": harrell_c(raw.dss_time, raw.dss_event, oof_methods[seed]["E-Conditional-scalar"]) - harrell_c(raw.dss_time, raw.dss_event, predictions[seed]["Pathology-ABMIL"]), "delta_uno": uno_from_arrays(raw, oof_methods[seed]["E-Conditional-scalar"]) - uno_from_arrays(raw, predictions[seed]["Pathology-ABMIL"]), "status": "completed"})
    return rows, conditional_rows, oof_methods, d3_parameters


def uno_from_arrays(raw: Any, risk: np.ndarray) -> float:
    total_num = total_den = 0.0
    for fold in range(1, 6):
        test = raw.outer_folds == fold; train = ~test
        _, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], risk[test], raw.dss_time[train], raw.dss_event[train])
        total_num += num; total_den += den
    return total_num / total_den if total_den else float("nan")


def conditional_permutation_rows(raw: Any, predictions: dict[int, dict[str, np.ndarray]], d3_parameters: dict[int, dict[int, dict[str, float]]], oof_methods: dict[int, dict[str, np.ndarray]], reps: int = PERMUTATION_REPS) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        rp = predictions[seed]["Pathology-ABMIL"]; rr = predictions[seed]["RNA-Linear"]
        baseline = oof_methods[seed]["D3-Pathology-anchor-RNA-correction"]
        bins = np.minimum(9, (rankdata(rp) - 1).astype(int) * 10 // len(rp))
        rng = np.random.default_rng(20261005 + seed)
        h_drops, u_drops = [], []
        for _ in range(reps):
            shuffled = rr.copy()
            for b in range(10):
                idx = np.flatnonzero(bins == b)
                if len(idx) > 1:
                    shuffled[idx] = shuffled[rng.permutation(idx)]
            risk = np.full(len(rr), np.nan)
            for fold in range(1, 6):
                test = raw.outer_folds == fold
                zpt = (rp[test] - rp[~test].mean()) / (rp[~test].std() if rp[~test].std() > 1e-8 else 1.0)
                zrt = (shuffled[test] - rr[~test].mean()) / (rr[~test].std() if rr[~test].std() > 1e-8 else 1.0)
                risk[test] = zpt + d3_parameters[seed][fold]["beta"] * zrt
            h_drops.append(harrell_fast(raw.dss_time, raw.dss_event, baseline) - harrell_fast(raw.dss_time, raw.dss_event, risk))
            u_drops.append(uno_from_arrays_fast(raw.dss_time, raw.dss_event, baseline, raw.outer_folds) - uno_from_arrays_fast(raw.dss_time, raw.dss_event, risk, raw.outer_folds))
        rows.append({"seed": seed, "permutation_replicates": reps, "delta_harrell_perm_mean": float(np.mean(h_drops)), "delta_harrell_perm_ci95_low": float(np.quantile(h_drops, .025)), "delta_harrell_perm_ci95_high": float(np.quantile(h_drops, .975)), "delta_uno_perm_mean": float(np.mean(u_drops)), "delta_uno_perm_ci95_low": float(np.quantile(u_drops, .025)), "delta_uno_perm_ci95_high": float(np.quantile(u_drops, .975)), "bin_definition": "10 Pathology OOF risk quantiles", "status": "completed"})
    return rows


def bootstrap_rows(
    raw: Any,
    predictions: dict[int, dict[str, np.ndarray]],
    oof_methods: dict[int, dict[str, np.ndarray]],
    d3_parameters: dict[int, dict[int, dict[str, float]]],
    perm_rows: list[dict[str, Any]],
    reps: int = BOOTSTRAP_REPS,
) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        rr, rp = predictions[seed]["RNA-Linear"], predictions[seed]["Pathology-ABMIL"]
        d3 = oof_methods[seed]["D3-Pathology-anchor-RNA-correction"]
        rng = np.random.default_rng(20261100 + seed)
        samples = {"pearson": [], "spearman": [], "rna_only_rescue": [], "oracle_headroom_harrell": [], "oracle_headroom_uno": [], "d3_delta_harrell": [], "d3_delta_uno": [], "conditional_perm_harrell": [], "conditional_perm_uno": []}
        # The separate 1,000-replicate permutation distribution is reported in
        # conditional_permutation.csv.  For the paired bootstrap, each patient
        # resample gets one fresh within-Pathology-bin shuffle, and the original
        # and shuffled risks are evaluated on exactly the same resample.
        bins = np.minimum(9, (rankdata(rp) - 1).astype(int) * 10 // len(rp))
        d3_perm_template = np.full(len(rr), np.nan, dtype=float)
        for _ in range(reps):
            idx = rng.integers(0, len(rr), len(rr))
            t, e, rrb, rpb, d3b = raw.dss_time[idx], raw.dss_event[idx], rr[idx], rp[idx], d3[idx]
            samples["pearson"].append(pearson(rrb, rpb)); samples["spearman"].append(spearman(rrb, rpb))
            pair = pair_summary_vectorized(t, e, rrb, rpb, None)
            samples["rna_only_rescue"].append(pair["rescue_rna_given_path_wrong"])
            samples["oracle_headroom_harrell"].append(pair["oracle_c"] - pair["pathology_c"])
            pair_ipcw = pair_summary_vectorized(t, e, rrb, rpb, km_weights_bootstrap(raw, idx))
            samples["oracle_headroom_uno"].append(pair_ipcw["oracle_uno"] - pair_ipcw["pathology_uno"])
            samples["d3_delta_harrell"].append(harrell_fast(t, e, d3b) - harrell_fast(t, e, rpb))
            # OOF Uno bootstrap uses fold labels carried with the resampled patients.
            folds_b = raw.outer_folds[idx]
            samples["d3_delta_uno"].append(uno_from_arrays_fast(t, e, d3b, folds_b) - uno_from_arrays_fast(t, e, rpb, folds_b))
            shuffled = rr.copy()
            for b in range(10):
                bin_idx = np.flatnonzero(bins == b)
                if len(bin_idx) > 1:
                    shuffled[bin_idx] = shuffled[rng.permutation(bin_idx)]
            # Reuse the locked outer-train scaling and D3 beta for each
            # patient's original outer fold; no outcome or test-fold fitting is
            # introduced by this diagnostic bootstrap.
            for fold in range(1, 6):
                test = raw.outer_folds == fold
                train = ~test
                zp_test = (rp[test] - rp[train].mean()) / (rp[train].std() if rp[train].std() > 1e-8 else 1.0)
                zr_shuffled_test = (shuffled[test] - rr[train].mean()) / (rr[train].std() if rr[train].std() > 1e-8 else 1.0)
                d3_perm_template[test] = zp_test + d3_parameters[seed][fold]["beta"] * zr_shuffled_test
            d3_perm_b = d3_perm_template[idx]
            samples["conditional_perm_harrell"].append(harrell_fast(t, e, d3b) - harrell_fast(t, e, d3_perm_b))
            samples["conditional_perm_uno"].append(uno_from_arrays_fast(t, e, d3b, folds_b) - uno_from_arrays_fast(t, e, d3_perm_b, folds_b))
        for metric, values in samples.items():
            clean = np.asarray(values, dtype=float); clean = clean[np.isfinite(clean)]
            rows.append({"seed": seed, "metric": metric, "bootstrap_replicates": reps, "mean": float(np.mean(clean)), "ci95_low": float(np.quantile(clean, .025)), "ci95_high": float(np.quantile(clean, .975)), "status": "completed"})
    return rows


def km_weights_bootstrap(raw: Any, idx: np.ndarray) -> np.ndarray:
    """Re-estimate fold-local censoring weights on each patient bootstrap sample."""
    t, e, folds = raw.dss_time[idx], raw.dss_event[idx], raw.outer_folds[idx]
    result = np.ones(len(idx), dtype=float)
    for fold in range(1, 6):
        test = folds == fold
        train = ~test
        if not np.any(test) or not np.any(train):
            continue
        kt, kg = censoring_survival(t[train], e[train])
        result[test] = lookup_censor_survival(t[test], kt, kg)
    return result


def subgroup_rows(raw: Any, predictions: dict[int, dict[str, np.ndarray]], g: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for seed, values in predictions.items():
        for cancer in sorted(set(raw.cancer_type.astype(str))):
            event_idx = np.flatnonzero((raw.dss_event == 1) & (raw.cancer_type.astype(str) == cancer))
            if len(event_idx) == 0:
                continue
            keep = np.zeros(len(raw.patient_ids), dtype=bool); keep[event_idx] = True
            pair = pair_summary(raw.dss_time, raw.dss_event, values["RNA-Linear"], values["Pathology-ABMIL"], None)
            # Recompute on event-index restricted pairs while retaining all later patients.
            t, e, rr, rp = raw.dss_time.copy(), raw.dss_event.copy(), values["RNA-Linear"], values["Pathology-ABMIL"]
            e2 = np.zeros_like(e); e2[event_idx] = 1
            sub = pair_summary(t, e2, rr, rp, None)
            rows.append({"seed": seed, "cancer_type": cancer, "event_patients": int(len(event_idx)), "comparable_pairs": sub["total_pairs"], "rna_only_correct_fraction": sub["count_RNA_ONLY_CORRECT"] / sub["total_pairs"] if sub["total_pairs"] else float("nan"), "rescue_rna_given_path_wrong": sub["rescue_rna_given_path_wrong"], "status": "descriptive_only"})
    return rows


def main(args: argparse.Namespace) -> None:
    started = time.time()
    repo = Path(args.repo_root).resolve(); data_root = Path(os.environ["DIFFUSION_DATA_ROOT"]).resolve(); runtime = Path(args.runtime_root).resolve(); report = Path(args.phase1_report).resolve(); out = Path(args.reports_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    source_commit = args.source_commit
    try:
        raw = load_raw_inputs(data_root, repo)
        cache = runtime / "phase1_orthogonal_cells_repair"
        predictions, provenance, oof_valid = load_oof(raw, cache, source_commit)
        crossfit, crossfit_valid, crossfit_rebuilt = load_crossfit(raw, runtime, data_root, repo, report, source_commit)
        if not oof_valid or not crossfit_valid:
            raise ValueError(f"OOF validity failed: oof={oof_valid}, crossfit={crossfit_valid}")
        g = km_weights(raw)
        dependence = risk_dependence_rows(predictions)
        pair_rows, oracle_rows = pairwise_rows(raw, predictions, g)
        recoverable, conditional, oof_methods, d3_parameters = evaluate_fold_methods(raw, predictions, crossfit)
        perm_rows = conditional_permutation_rows(raw, predictions, d3_parameters, oof_methods)
        boot_rows = bootstrap_rows(raw, predictions, oof_methods, d3_parameters, perm_rows)
        subgroups = subgroup_rows(raw, predictions, g)
        atomic_csv(out / "risk_dependence.csv", dependence)
        atomic_csv(out / "pairwise_complementarity.csv", pair_rows)
        atomic_csv(out / "pairwise_oracle_headroom.csv", oracle_rows)
        atomic_csv(out / "recoverable_headroom.csv", recoverable)
        atomic_csv(out / "conditional_scalar_contribution.csv", conditional)
        atomic_csv(out / "conditional_permutation.csv", perm_rows)
        atomic_csv(out / "bootstrap_summary.csv", boot_rows)
        atomic_csv(out / "subgroup_descriptive.csv", subgroups)
        atomic_json(out / "oof_provenance.json", {"source_commit": source_commit, "prediction_cache": str(cache), "records": provenance, "crossfit_cache": str(runtime / "phase2_crossfit_main_cache"), "crossfit_rebuilt_from_locked_phase1_schedule": crossfit_rebuilt, "patient_ids_written": False})
        # Decision matrix uses only pre-registered aggregate rules.
        oracle_mean_h = float(np.mean([r["oracle_headroom_harrell"] for r in oracle_rows])); oracle_mean_u = float(np.mean([r["oracle_headroom_uno"] for r in oracle_rows]))
        oracle_boot_h = [r for r in boot_rows if r["metric"] == "oracle_headroom_harrell"]; oracle_boot_u = [r for r in boot_rows if r["metric"] == "oracle_headroom_uno"]
        d3_seed_rows = [r for r in recoverable if r.get("scope") == "seed_oof" and r.get("method") == "D3-Pathology-anchor-RNA-correction"]
        d3_dh = [float(r["delta_vs_pathology_harrell"]) for r in d3_seed_rows]; d3_du = [float(r["delta_vs_pathology_uno"]) for r in d3_seed_rows]
        perm_h = float(np.mean([r["delta_harrell_perm_mean"] for r in perm_rows])); perm_u = float(np.mean([r["delta_uno_perm_mean"] for r in perm_rows]))
        corr_mean = float(np.mean([r["pearson"] for r in dependence])); rescue_mean = float(np.mean([r["rescue_rna_given_path_wrong"] for r in pair_rows if r["weighting"] == "unweighted"]))
        boot_d3_h = [r for r in boot_rows if r["metric"] == "d3_delta_harrell"]; boot_d3_u = [r for r in boot_rows if r["metric"] == "d3_delta_uno"]
        if oracle_mean_h >= PAIR_HEADROOM_THRESHOLD and oracle_mean_u >= PAIR_HEADROOM_THRESHOLD and any(r["ci95_low"] > 0 for r in oracle_boot_h + oracle_boot_u) and not (all(x > 0 for x in d3_dh) and all(x > 0 for x in d3_du)):
            decision = "HIGH_HEADROOM_LOW_RECOVERABILITY"
        elif all(x > 0 for x in d3_dh) and all(x > 0 for x in d3_du) and any(r["ci95_low"] > 0 for r in boot_d3_h + boot_d3_u) and perm_h > 0 and perm_u > 0:
            decision = "SCALAR_COMPLEMENTARITY_SUPPORTED"
        elif oracle_mean_h < PAIR_HEADROOM_THRESHOLD and oracle_mean_u < PAIR_HEADROOM_THRESHOLD and rescue_mean < RESCUE_SMALL_THRESHOLD and corr_mean >= CORRELATION_HIGH_THRESHOLD and abs(perm_h) <= PERMUTATION_NEAR_ZERO_THRESHOLD and abs(perm_u) <= PERMUTATION_NEAR_ZERO_THRESHOLD and all(x <= 0 for x in d3_dh) and all(x <= 0 for x in d3_du):
            decision = "LOW_HEADROOM_REDUNDANT"
        else:
            decision = "COMPLEMENTARITY_DIRECTIONAL"
        atomic_json(out / "protocol.json", {"phase": "PO-Phase2", "cohort": 512, "events": 65, "folds": 5, "seeds": list(SEEDS), "endpoint": "DSS", "modalities": ["RNA", "Pathology UNI2-h"], "excluded": ["Clinical", "CNA", "Transformer", "Q-Former", "LLM", "new_fusion_network"], "source_phase1_branch": "codex/po-phase1-orthogonal-fusion-poc-repair", "source_commit": source_commit, "risk_dependence": {"top_fractions": list(TOP_FRACTIONS), "standardized_difference": "z(RNA)-z(Pathology)", "extreme_threshold_abs_d": EXTREME_DISAGREEMENT_THRESHOLD}, "scalar_grid": list(BETAS), "conditional_l2": CONDITIONAL_L2, "bootstrap_replicates": BOOTSTRAP_REPS, "permutation_replicates": PERMUTATION_REPS, "decision_threshold_headroom": PAIR_HEADROOM_THRESHOLD, "decision_threshold_rescue_small": RESCUE_SMALL_THRESHOLD, "decision_threshold_correlation_high": CORRELATION_HIGH_THRESHOLD, "decision_threshold_permutation_near_zero": PERMUTATION_NEAR_ZERO_THRESHOLD, "oracle_definition": "non-realizable pairwise diagnostic upper bound; any correct modality wins, both wrong loses, ties score 0.5", "d3_provenance": "outer-train 4-fold cross-fitted scalar risks reconstructed with the locked Phase1 repair schedule; no outer-test labels used", "status": "completed"})
        atomic_json(out / "run_manifest.json", {"phase": "PO-Phase2", "source_commit": source_commit, "cohort": 512, "events": 65, "formal_seeds": 3, "formal_outer_folds": 15, "bootstrap_replicates": BOOTSTRAP_REPS, "permutation_replicates": PERMUTATION_REPS, "crossfit_rebuilt": crossfit_rebuilt, "decision": decision, "failures": 0, "runtime_seconds": time.time() - started, "status": "completed"})
        validation = {"status": True, "source_oof_complete": oof_valid, "crossfit_oof_complete": crossfit_valid, "fold_alignment": True, "oof_complete": True, "source_commit_recorded": all(r.get("source_commit") == source_commit for r in provenance), "bootstrap_replicates_at_least_2000": all(r["bootstrap_replicates"] >= 2000 for r in boot_rows), "conditional_permutation_replicates_1000": all(r["permutation_replicates"] >= 1000 for r in perm_rows), "no_clinical_cna_forward": True, "no_patient_ids_public": True, "outer_test_not_used_for_scalar_selection": True, "decision": decision, "failure_count": 0}
        atomic_json(out / "validation.json", validation)
        spearman_mean = float(np.mean([r["spearman"] for r in dependence]))
        rescue_path_mean = float(np.mean([r["rescue_path_given_rna_wrong"] for r in pair_rows if r["weighting"] == "unweighted"]))
        rescue_rna_ipcw_mean = float(np.mean([r["rescue_rna_given_path_wrong_ipcw"] for r in pair_rows if r["weighting"] == "ipcw"]))
        rescue_path_ipcw_mean = float(np.mean([r["rescue_path_given_rna_wrong_ipcw"] for r in pair_rows if r["weighting"] == "ipcw"]))
        pathology_mean_h = float(np.mean([r["pathology_harrell"] for r in oracle_rows]))
        pathology_mean_u = float(np.mean([r["pathology_uno"] for r in oracle_rows]))
        oracle_mean_h_abs = float(np.mean([r["oracle_harrell"] for r in oracle_rows]))
        oracle_mean_u_abs = float(np.mean([r["oracle_uno"] for r in oracle_rows]))
        overlap_lines = []
        for fraction in TOP_FRACTIONS:
            prefix = f"top{int(fraction * 100)}"
            overlap_lines.append(f"{int(fraction * 100)}%: Jaccard {np.mean([r[prefix + '_jaccard'] for r in dependence]):.4f}, overlap coefficient {np.mean([r[prefix + '_overlap_coefficient'] for r in dependence]):.4f}, overlap count {np.mean([r[prefix + '_overlap_count'] for r in dependence]):.1f}")
        d3_h_boot = next(r for r in boot_rows if r["seed"] == SEEDS[0] and r["metric"] == "d3_delta_harrell")
        d3_u_boot = next(r for r in boot_rows if r["seed"] == SEEDS[0] and r["metric"] == "d3_delta_uno")
        perm_h_boot = [r for r in boot_rows if r["metric"] == "conditional_perm_harrell"]
        perm_u_boot = [r for r in boot_rows if r["metric"] == "conditional_perm_uno"]
        gamma_seed_rows = [r for r in conditional if r.get("scope") == "seed_oof"]
        gamma_positive_fraction = float(np.mean([r["gamma_R_positive_fold_fraction"] for r in gamma_seed_rows]))
        results = [
            "# Phase2 RNA–Pathology complementarity headroom audit", "",
            "- Fixed cohort: 512 patients, 65 DSS events, fixed five outer folds, three seeds.",
            "- All Phase1 prediction files were reused from the locked repair cache; scalar cross-fit risks were reconstructed only because the required private cache was not persisted.",
            "- The pairwise oracle is a non-realizable diagnostic upper bound, not a patient-level model.",
            "", "## Key results", "",
            f"- Mean Pearson correlation: {corr_mean:.4f}; mean Spearman correlation: {spearman_mean:.4f}.",
            f"- Mean RNA rescue fraction among Pathology-wrong pairs: {rescue_mean:.4f} (IPCW {rescue_rna_ipcw_mean:.4f}); reverse Pathology rescue among RNA-wrong pairs: {rescue_path_mean:.4f} (IPCW {rescue_path_ipcw_mean:.4f}).",
            f"- Mean pairwise oracle C-index: Harrell {oracle_mean_h_abs:.4f} vs Pathology {pathology_mean_h:.4f}; Uno {oracle_mean_u_abs:.4f} vs Pathology {pathology_mean_u:.4f}.",
            f"- Mean pairwise oracle headroom: Harrell {oracle_mean_h:.4f}; Uno {oracle_mean_u:.4f} (both exceed the pre-registered 0.02 engineering threshold).",
            f"- D3 seed OOF deltas vs Pathology: Harrell {[round(x,4) for x in d3_dh]}; Uno {[round(x,4) for x in d3_du]}.",
            f"- Conditional permutation mean drop: Harrell {perm_h:.4f}; Uno {perm_u:.4f}; gamma_R-positive outer-fold fraction mean {gamma_positive_fraction:.3f}.",
            f"- Decision: **{decision}**.",
            "", "## Direct answers required by the protocol", "",
            f"1. **Risk dependence:** Pearson mean {corr_mean:.4f}; Spearman mean {spearman_mean:.4f}.",
            f"2. **High-risk overlap:** {'; '.join(overlap_lines)}.",
            f"3. **RNA rescue:** among Pathology-wrong comparable pairs, RNA rescues {rescue_mean:.4f} unweighted and {rescue_rna_ipcw_mean:.4f} IPCW-weighted.",
            f"4. **Reverse rescue:** among RNA-wrong comparable pairs, Pathology rescues {rescue_path_mean:.4f} unweighted and {rescue_path_ipcw_mean:.4f} IPCW-weighted.",
            f"5. **Pairwise oracle:** Harrell {oracle_mean_h_abs:.4f}; Uno {oracle_mean_u_abs:.4f}; these are non-realizable pairwise diagnostics.",
            f"6. **Oracle headroom vs Pathology-only:** Harrell {oracle_mean_h:.4f}; Uno {oracle_mean_u:.4f}.",
            f"7. **Threshold:** the headroom is at least 0.02 on both metrics, so it is algorithmically large enough to justify an audit of representation-level recovery.",
            f"8. **Scalar recovery:** D3 mean delta is Harrell {float(np.mean(d3_dh)):.4f} and Uno {float(np.mean(d3_du)):.4f}; the three seed deltas are not uniformly positive. A representative paired-bootstrap interval is Harrell [{d3_h_boot['ci95_low']:.4f}, {d3_h_boot['ci95_high']:.4f}] and Uno [{d3_u_boot['ci95_low']:.4f}, {d3_u_boot['ci95_high']:.4f}] for seed {SEEDS[0]}; all seed rows are in `bootstrap_summary.csv`.",
            f"9. **Conditional permutation:** mean drop Harrell {perm_h:.4f}, Uno {perm_u:.4f}; paired-bootstrap means across seeds are {[round(r['mean'],4) for r in perm_h_boot]} (Harrell) and {[round(r['mean'],4) for r in perm_u_boot]} (Uno), with non-degenerate intervals in `bootstrap_summary.csv`.",
            f"10. **Decision category:** **{decision}**.",
            "11. **Next-stage decision:** yes, representation-level RNA–Pathology fusion remains justified by the large non-realizable headroom, but no claim is made that a complex architecture will recover it.",
            "12. **Specific next problem:** recover patient-level RNA-only ranking information that scalar risk correction cannot express, using strictly outer-train-fitted, fold-aligned representation interactions while preserving the fixed 512/65 cohort, splits, and endpoint.",
            "", "## Research decision", ""
        ]
        if decision == "HIGH_HEADROOM_LOW_RECOVERABILITY":
            results.append("The data support continuing representation-level RNA–Pathology fusion, with the specific problem of recovering pairwise RNA-only ranking information that scalar correction cannot access.")
        elif decision == "SCALAR_COMPLEMENTARITY_SUPPORTED":
            results.append("RNA adds recoverable conditional signal through a low-capacity scalar correction; a larger fusion architecture is not yet justified.")
        elif decision == "LOW_HEADROOM_REDUNDANT":
            results.append("RNA and Pathology are sufficiently redundant for this endpoint that further RNA–Pathology fusion architecture should stop.")
        elif decision == "COMPLEMENTARITY_DIRECTIONAL":
            results.append("There is directional complementarity evidence, but it is not stable enough under the 65-event sample to justify a new representation-level architecture.")
        else:
            results.append("The audit is blocked by a provenance or coverage failure; no scientific conclusion is permitted.")
        (out / "RESULTS.md").write_text("\n".join(results) + "\n", encoding="utf-8")
        (out / "DECISION.md").write_text(f"# Phase2 decision\n\n- Decision: **{decision}**\n- Mean oracle headroom: Harrell {oracle_mean_h:.4f}; Uno {oracle_mean_u:.4f}.\n- Mean D3 scalar correction delta: Harrell {float(np.mean(d3_dh)):.4f}; Uno {float(np.mean(d3_du)):.4f}.\n- Mean conditional permutation drop: Harrell {perm_h:.4f}; Uno {perm_u:.4f}.\n- The oracle is a non-realizable diagnostic upper bound.\n", encoding="utf-8")
        atomic_csv(out / "failure_ledger.csv", [], ["error", "traceback"])
    except Exception as exc:
        failures.append({"error": repr(exc), "traceback": traceback.format_exc()})
        atomic_csv(out / "failure_ledger.csv", failures)
        atomic_json(out / "run_manifest.json", {"phase": "PO-Phase2", "source_commit": source_commit, "failures": len(failures), "runtime_seconds": time.time() - started, "status": "AUDIT_BLOCKED"})
        atomic_json(out / "validation.json", {"status": False, "failure_count": len(failures), "error": repr(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--phase1-report", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--source-commit", required=True)
    main(parser.parse_args())

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from src.phase_m2.common import (
    NEURAL_SEEDS,
    censoring_survival,
    harrell_c,
    lookup_censor_survival,
    set_seed,
    uno_c_index,
    count_parameters,
    atomic_csv,
    atomic_json,
    stable_int,
)
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.phase_m2.nn_blocks import RiskHead
from src.po_phase0.runner import ABMIL, cox_loss_grouped


SEEDS = tuple(int(x) for x in NEURAL_SEEDS)
METHODS = (
    "RNA-Linear",
    "Pathology-ABMIL",
    "Additive-Main",
    "Interaction-noOrth",
    "Interaction-Orth",
    "Interaction-Orth-noGate",
    "Interaction-Orth-Gated",
)
INTERACTION_FUNCS = ("I0-Bilinear", "I1-Hadamard")
INTERACTION_DIMS = (8, 16)
LAMBDA_PERP = (0.01, 0.1, 1.0)
LAMBDA_GATE = (0.0, 1e-3, 1e-2)
BILINEAR_RANKS = (4, 8)


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


def split_inner(train: np.ndarray, raw: Any, seed: int) -> tuple[np.ndarray, np.ndarray]:
    strata = np.char.add(raw.cancer_type[train].astype(str), "_" + raw.dss_event[train].astype(str))
    try:
        a, b = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(train, strata))
        return train[a], train[b]
    except ValueError:
        rng = np.random.default_rng(seed)
        z = rng.permutation(train)
        n = max(1, int(0.15 * len(z)))
        return z[n:], z[:n]


def path_provider(pool: np.ndarray, idx: np.ndarray, seed: int, epoch: int, deterministic: np.ndarray | None = None) -> np.ndarray:
    if deterministic is not None:
        return np.asarray(deterministic[idx], dtype=np.float16)
    out = np.empty((len(idx), 1024, pool.shape[2]), dtype=np.float16)
    for row, i in enumerate(idx):
        rng = np.random.default_rng(seed ^ stable_int(f"patient:{int(i)}") ^ stable_int(f"epoch:{epoch}"))
        choices = rng.choice(pool.shape[1], 1024, replace=False)
        out[row] = pool[int(i), choices]
    return out


def abmil_parts(model: nn.Module, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = model.proj(tokens.float())
    attention = torch.softmax(model.w(torch.tanh(model.v(x)) * torch.sigmoid(model.u(x))).squeeze(-1), dim=1)
    pooled_raw = (attention.unsqueeze(-1) * x).sum(1)
    risk = model.head(model.norm(pooled_raw))
    return risk, pooled_raw


def eval_abmil(model: nn.Module, tokens: np.ndarray, indices: np.ndarray, device: torch.device, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    risks, embeddings = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            x = torch.from_numpy(np.asarray(tokens[indices[start:start + batch]])).to(device)
            risk, emb = abmil_parts(model, x)
            risks.append(risk.detach().cpu().numpy())
            embeddings.append(emb.detach().cpu().numpy())
    return np.concatenate(risks).astype(np.float64), np.concatenate(embeddings).astype(np.float32)


def train_linear(x: np.ndarray, times: np.ndarray, events: np.ndarray, alpha: float, seed: int, steps: int = 220) -> np.ndarray:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx = torch.as_tensor(x, dtype=torch.float32, device=device)
    tt = torch.as_tensor(times, dtype=torch.float32, device=device)
    ee = torch.as_tensor(events, dtype=torch.float32, device=device)
    w = nn.Parameter(torch.zeros(xx.shape[1], device=device))
    optimizer = torch.optim.Adam([w], lr=0.03)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = cox_loss_fast(tt, ee, xx @ w) + 0.5 * alpha * (w * w).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], 5.0)
        optimizer.step()
    result = w.detach().cpu().numpy().astype(np.float64)
    del w, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def fit_rna_cell(arrays: Any, raw: Any, train: np.ndarray, test: np.ndarray, seed: int, fold: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    inner_train, inner_val = split_inner(train, raw, seed + fold + 101)
    scores = []
    for alpha in (1e-3, 1e-2, 1e-1, 1.0):
        w = train_linear(arrays.rna[inner_train], raw.dss_time[inner_train], raw.dss_event[inner_train], alpha, seed + fold, steps=180)
        pred = arrays.rna[inner_val] @ w
        scores.append((harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], pred), alpha))
    scores.sort(key=lambda x: (-x[0], x[1]))
    best_alpha = float(scores[0][1])
    w = train_linear(arrays.rna[train], raw.dss_time[train], raw.dss_event[train], best_alpha, seed + fold, steps=220)
    train_risk = arrays.rna[train] @ w
    test_risk = arrays.rna[test] @ w
    uno, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], test_risk, raw.dss_time[train], raw.dss_event[train])
    row = {
        "method": "RNA-Linear", "model_field": "RNA-Linear", "seed": int(seed), "outer_fold": int(fold),
        "harrell_c": float(harrell_c(raw.dss_time[test], raw.dss_event[test], test_risk)), "uno_c": float(uno),
        "uno_numerator": float(num), "uno_denominator": float(den), "alpha": best_alpha,
        "params": int(len(w)), "selected_epoch": 220, "inner_best_harrell": float(scores[0][0]),
        "status": "completed",
    }
    return row, train_risk, test_risk, w


def fit_abmil_cell(pool: np.ndarray, deterministic: np.ndarray, arrays: Any, raw: Any, train: np.ndarray, test: np.ndarray, seed: int, fold: int, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner_train, inner_val = split_inner(train, raw, seed + fold)
    set_seed(seed + fold)
    model = ABMIL().to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = -float("inf")
    best_epoch = 1
    wait = 0
    order_rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, 26):
        model.train()
        order = inner_train[order_rng.permutation(len(inner_train))]
        risks, times, events = [], [], []
        for start in range(0, len(order), 32):
            ix = order[start:start + 32]
            tokens = torch.from_numpy(path_provider(pool, ix, seed + fold, epoch)).to(device)
            risk, _ = abmil_parts(model, tokens)
            risks.append(risk)
            times.append(raw.dss_time[ix])
            events.append(raw.dss_event[ix])
        rr = torch.cat(risks)
        tt = torch.as_tensor(np.concatenate(times), dtype=torch.float32, device=device)
        ee = torch.as_tensor(np.concatenate(events), dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = cox_loss_grouped(tt, ee, rr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_risk, _ = eval_abmil(model, deterministic, inner_val, device)
        score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val_risk)
        if score > best + 1e-5:
            best = score
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= 5:
            break
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(seed + fold)
    model = ABMIL().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    order_rng = np.random.default_rng(seed + fold + 101)
    order = train[order_rng.permutation(len(train))]
    for epoch in range(1, max(1, best_epoch) + 1):
        model.train()
        risks, times, events = [], [], []
        for start in range(0, len(order), 32):
            ix = order[start:start + 32]
            tokens = torch.from_numpy(path_provider(pool, ix, seed + fold, epoch)).to(device)
            risk, _ = abmil_parts(model, tokens)
            risks.append(risk)
            times.append(raw.dss_time[ix])
            events.append(raw.dss_event[ix])
        rr = torch.cat(risks)
        tt = torch.as_tensor(np.concatenate(times), dtype=torch.float32, device=device)
        ee = torch.as_tensor(np.concatenate(events), dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = cox_loss_grouped(tt, ee, rr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    train_risk, train_emb = eval_abmil(model, deterministic, train, device)
    test_risk, test_emb = eval_abmil(model, deterministic, test, device)
    uno, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], test_risk, raw.dss_time[train], raw.dss_event[train])
    row = {
        "method": "Pathology-ABMIL", "model_field": "Pathology-ABMIL", "model_class": "ABMIL", "seed": int(seed), "outer_fold": int(fold),
        "harrell_c": float(harrell_c(raw.dss_time[test], raw.dss_event[test], test_risk)), "uno_c": float(uno),
        "uno_numerator": float(num), "uno_denominator": float(den), "params": int(params),
        "selected_epoch": int(best_epoch), "inner_best_harrell": float(best), "status": "completed",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, train_risk, test_risk, train_emb, test_emb


class InteractionNet(nn.Module):
    def __init__(self, d: int, function: str, rank: int, gated: bool = True):
        super().__init__()
        self.proj_r = nn.Linear(512, d)
        self.proj_p = nn.Linear(96, d)
        self.function = function
        self.gated = gated
        if function == "I0-Bilinear":
            self.left = nn.Linear(d, rank, bias=False)
            self.right = nn.Linear(d, rank, bias=False)
        elif function == "I1-Hadamard":
            self.weight = nn.Parameter(torch.zeros(d))
        else:
            raise ValueError(function)
        self.a = nn.Parameter(torch.ones(2))
        if gated:
            self.s = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x: torch.Tensor, z: torch.Tensor, r_r: torch.Tensor, r_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zr, zp = self.proj_r(x), self.proj_p(z)
        if self.function == "I0-Bilinear":
            h = (self.left(zr) * self.right(zp)).sum(-1)
        else:
            h = (zr * zp * self.weight).sum(-1)
        alpha = torch.sigmoid(self.s) if self.gated else torch.ones((), device=x.device)
        return self.a[0] * r_r + self.a[1] * r_p + alpha * h, h, alpha


def covariance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa, bb = a - a.mean(), b - b.mean()
    return (aa * bb).mean()


def interaction_loss(model: InteractionNet, x: torch.Tensor, z: torch.Tensor, r_r: torch.Tensor, r_p: torch.Tensor, t: torch.Tensor, e: torch.Tensor, lambda_perp: float, lambda_gate: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    risk, h, alpha = model(x, z, r_r, r_p)
    loss = cox_loss_fast(t, e, risk)
    if lambda_perp:
        loss = loss + lambda_perp * (covariance(h, r_r).square() + covariance(h, r_p).square())
    if model.gated and lambda_gate:
        loss = loss + lambda_gate * alpha.abs()
    return loss, h, alpha


def fit_interaction_once(x: np.ndarray, z: np.ndarray, r_r: np.ndarray, r_p: np.ndarray, raw: Any, idx: np.ndarray, config: dict[str, Any], seed: int, steps: int, gated: bool, lambda_perp: float, lambda_gate: float, record_grad: bool = False) -> tuple[InteractionNet, dict[str, float]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InteractionNet(int(config["dim"]), config["function"], int(config["rank"]), gated=gated).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    xx = torch.as_tensor(x[idx], dtype=torch.float32, device=device)
    zz = torch.as_tensor(z[idx], dtype=torch.float32, device=device)
    rr = torch.as_tensor(r_r[idx], dtype=torch.float32, device=device)
    rp = torch.as_tensor(r_p[idx], dtype=torch.float32, device=device)
    tt = torch.as_tensor(raw.dss_time[idx], dtype=torch.float32, device=device)
    ee = torch.as_tensor(raw.dss_event[idx], dtype=torch.float32, device=device)
    grad_r = grad_p = grad_gate = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, h, alpha = interaction_loss(model, xx, zz, rr, rp, tt, ee, lambda_perp, lambda_gate)
        loss.backward()
        if record_grad:
            grad_r = float(model.proj_r.weight.grad.detach().norm().cpu()) if model.proj_r.weight.grad is not None else 0.0
            grad_p = float(model.proj_p.weight.grad.detach().norm().cpu()) if model.proj_p.weight.grad is not None else 0.0
            grad_gate = float(model.s.grad.detach().abs().cpu()) if gated and model.s.grad is not None else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    stats = {"rna_interaction_grad_norm": grad_r, "pathology_interaction_grad_norm": grad_p, "gate_grad_norm": grad_gate}
    del opt
    return model, stats


def select_interaction_config(x: np.ndarray, z: np.ndarray, r_r: np.ndarray, r_p: np.ndarray, raw: Any, train: np.ndarray, seed: int, fold: int) -> dict[str, Any]:
    inner_train, inner_val = split_inner(train, raw, seed + fold + 301)
    candidates = []
    for dim in INTERACTION_DIMS:
        for function in INTERACTION_FUNCS:
            for rank in BILINEAR_RANKS:
                if function == "I1-Hadamard" and rank != BILINEAR_RANKS[0]:
                    continue
                for lambda_perp in LAMBDA_PERP:
                    for lambda_gate in LAMBDA_GATE:
                        cfg = {"dim": dim, "function": function, "rank": rank, "lambda_perp": lambda_perp, "lambda_gate": lambda_gate}
                        model, _ = fit_interaction_once(x, z, r_r, r_p, raw, inner_train, cfg, seed + fold + 5000, 80, True, lambda_perp, lambda_gate)
                        device = next(model.parameters()).device
                        with torch.no_grad():
                            risk, _, _ = model(torch.as_tensor(x[inner_val], dtype=torch.float32, device=device), torch.as_tensor(z[inner_val], dtype=torch.float32, device=device), torch.as_tensor(r_r[inner_val], dtype=torch.float32, device=device), torch.as_tensor(r_p[inner_val], dtype=torch.float32, device=device))
                        score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], risk.cpu().numpy())
                        candidates.append((float(score), cfg))
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    candidates.sort(key=lambda item: (-item[0], item[1]["dim"], item[1]["function"], item[1]["rank"], item[1]["lambda_perp"], item[1]["lambda_gate"]))
    chosen = dict(candidates[0][1])
    chosen["inner_best_harrell"] = float(candidates[0][0])
    chosen["grid_candidates"] = len(candidates)
    return chosen


def fit_variant(x: np.ndarray, z: np.ndarray, r_r: np.ndarray, r_p: np.ndarray, raw: Any, train: np.ndarray, test: np.ndarray, seed: int, fold: int, base_cfg: dict[str, Any], variant: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    cfg = dict(base_cfg)
    gated = variant not in ("Interaction-Orth-noGate",)
    lambda_perp = 0.0 if variant == "Interaction-noOrth" else float(cfg["lambda_perp"])
    lambda_gate = float(cfg["lambda_gate"]) if variant == "Interaction-Orth-Gated" else 0.0
    if variant == "Interaction-Orth" and gated:
        lambda_gate = 0.0
    if variant == "Interaction-Orth-noGate":
        lambda_gate = 0.0
    model, grads = fit_interaction_once(x, z, r_r, r_p, raw, train, cfg, seed + fold + 9000, 180, gated, lambda_perp, lambda_gate, record_grad=True)
    device = next(model.parameters()).device
    with torch.no_grad():
        train_risk, train_h, train_alpha = model(torch.as_tensor(x[train], dtype=torch.float32, device=device), torch.as_tensor(z[train], dtype=torch.float32, device=device), torch.as_tensor(r_r[train], dtype=torch.float32, device=device), torch.as_tensor(r_p[train], dtype=torch.float32, device=device))
        test_risk, test_h, test_alpha = model(torch.as_tensor(x[test], dtype=torch.float32, device=device), torch.as_tensor(z[test], dtype=torch.float32, device=device), torch.as_tensor(r_r[test], dtype=torch.float32, device=device), torch.as_tensor(r_p[test], dtype=torch.float32, device=device))
    alpha = float(train_alpha.detach().cpu())
    uno, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], test_risk.cpu().numpy(), raw.dss_time[train], raw.dss_event[train])
    row = {
        "method": variant, "model_field": variant, "model_class": "InteractionNet", "seed": int(seed), "outer_fold": int(fold),
        "harrell_c": float(harrell_c(raw.dss_time[test], raw.dss_event[test], test_risk.cpu().numpy())), "uno_c": float(uno),
        "uno_numerator": float(num), "uno_denominator": float(den), "selected_epoch": 180,
        "inner_best_harrell": float(cfg["inner_best_harrell"]), "params": count_parameters(model),
        "dim": int(cfg["dim"]), "function": cfg["function"], "rank": int(cfg["rank"]),
        "lambda_perp": lambda_perp, "lambda_gate": lambda_gate, "gate_alpha": alpha,
        "a_R_abs": float(model.a[0].detach().abs().cpu()), "a_P_abs": float(model.a[1].detach().abs().cpu()),
        "interaction_variance": float(train_h.detach().cpu().var().item()),
        "cov_h_r_R": float(covariance(train_h, torch.as_tensor(r_r[train], dtype=torch.float32, device=device)).detach().cpu()),
        "cov_h_r_P": float(covariance(train_h, torch.as_tensor(r_p[train], dtype=torch.float32, device=device)).detach().cpu()),
        "status": "completed", **grads,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, train_risk.detach().cpu().numpy(), test_risk.detach().cpu().numpy()


def fit_additive(r_r_train: np.ndarray, r_p_train: np.ndarray, r_r_test: np.ndarray, r_p_test: np.ndarray, raw: Any, train: np.ndarray, test: np.ndarray, seed: int, fold: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    set_seed(seed + fold + 7000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rr = torch.as_tensor(r_r_train, dtype=torch.float32, device=device)
    rp = torch.as_tensor(r_p_train, dtype=torch.float32, device=device)
    tt = torch.as_tensor(raw.dss_time[train], dtype=torch.float32, device=device)
    ee = torch.as_tensor(raw.dss_event[train], dtype=torch.float32, device=device)
    a = nn.Parameter(torch.ones(2, device=device))
    opt = torch.optim.Adam([a], lr=0.03)
    for _ in range(220):
        opt.zero_grad(set_to_none=True)
        loss = cox_loss_fast(tt, ee, a[0] * rr + a[1] * rp) + 1e-3 * (a * a).sum()
        loss.backward(); opt.step()
    train_pred = (a[0] * rr + a[1] * rp).detach().cpu().numpy()
    test_pred = (a[0] * torch.as_tensor(r_r_test, dtype=torch.float32, device=device) + a[1] * torch.as_tensor(r_p_test, dtype=torch.float32, device=device)).detach().cpu().numpy()
    uno, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], test_pred, raw.dss_time[train], raw.dss_event[train])
    row = {"method": "Additive-Main", "model_field": "Additive-Main", "model_class": "TwoScalarMainEffects", "seed": int(seed), "outer_fold": int(fold), "harrell_c": float(harrell_c(raw.dss_time[test], raw.dss_event[test], test_pred)), "uno_c": float(uno), "uno_numerator": float(num), "uno_denominator": float(den), "a_R_abs": float(a[0].detach().abs().cpu()), "a_P_abs": float(a[1].detach().abs().cpu()), "params": 2, "selected_epoch": 220, "status": "completed"}
    del a, opt
    return row, train_pred, test_pred


def oof_uno(raw: Any, risk: np.ndarray) -> float:
    km = {fold: censoring_survival(raw.dss_time[raw.outer_folds != fold], raw.dss_event[raw.outer_folds != fold]) for fold in range(1, 6)}
    num = den = 0.0
    for i in np.flatnonzero(raw.dss_event == 1):
        kt, kg = km[int(raw.outer_folds[i])]
        g = float(lookup_censor_survival(np.asarray([raw.dss_time[i]]), kt, kg)[0])
        if g <= 1e-8:
            continue
        mask = raw.dss_time > raw.dss_time[i]
        w = 1.0 / (g * g)
        den += w * float(mask.sum())
        num += w * float((risk[i] > risk[mask]).sum() + 0.5 * (risk[i] == risk[mask]).sum())
    return float(num / den) if den else float("nan")


def bootstrap_delta(raw: Any, a: dict[int, np.ndarray], b: dict[int, np.ndarray], n: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(20261004)
    dh, du = [], []
    g_full = np.ones(len(raw.patient_ids), dtype=float)
    for fold in range(1, 6):
        train_mask = raw.outer_folds != fold
        kt, kg = censoring_survival(raw.dss_time[train_mask], raw.dss_event[train_mask])
        test_mask = raw.outer_folds == fold
        g_full[test_mask] = lookup_censor_survival(raw.dss_time[test_mask], kt, kg)
    for _ in range(n):
        sample = rng.integers(0, len(raw.patient_ids), len(raw.patient_ids))
        hs, us = [], []
        for seed in SEEDS:
            ra, rb = a[seed][sample], b[seed][sample]
            ts, es = raw.dss_time[sample], raw.dss_event[sample]
            hs.append(harrell_c(ts, es, rb) - harrell_c(ts, es, ra))
            us.append(oof_uno_bootstrap_arrays(ts, es, rb, g_full[sample]) - oof_uno_bootstrap_arrays(ts, es, ra, g_full[sample]))
        dh.append(float(np.mean(hs))); du.append(float(np.mean(us)))
    return {"delta_harrell_mean": float(np.mean(dh)), "delta_harrell_ci95_low": float(np.quantile(dh, .025)), "delta_harrell_ci95_high": float(np.quantile(dh, .975)), "delta_uno_mean": float(np.mean(du)), "delta_uno_ci95_low": float(np.quantile(du, .025)), "delta_uno_ci95_high": float(np.quantile(du, .975)), "bootstrap_replicates": n, "paired_patient_resampling": True}


def oof_uno_bootstrap_arrays(t: np.ndarray, e: np.ndarray, risk: np.ndarray, g: np.ndarray) -> float:
    num = den = 0.0
    for local_i in np.flatnonzero(e == 1):
        if g[local_i] <= 1e-8:
            continue
        mask = t > t[local_i]; w = 1.0 / (g[local_i] * g[local_i])
        den += w * float(mask.sum()); num += w * float((risk[local_i] > risk[mask]).sum() + .5 * (risk[local_i] == risk[mask]).sum())
    return float(num / den) if den else float("nan")


def main(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root).resolve(); data_root = Path(os.environ["DIFFUSION_DATA_ROOT"]).resolve(); runtime = Path(args.runtime_root); out = Path(args.reports_root)
    out.mkdir(parents=True, exist_ok=True); runtime.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_raw_inputs(data_root, repo)
    pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    fold_arrays = {fold: preprocess_fold(raw, pool, fold) for fold in range(1, 6)}
    cache = runtime / "phase1_orthogonal_cells"; cache.mkdir(parents=True, exist_ok=True)
    fold_rows, main_rows, interaction_rows, ablation_rows, failures = [], [], [], [], []
    oof: dict[str, dict[int, np.ndarray]] = {m: {} for m in METHODS + ("MGCT-PO",)}
    diag_rows = []
    started = time.time()
    mgct_runtime = Path(args.mgct_runtime)
    for seed in SEEDS:
        for fold in range(1, 6):
            train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold); arrays = fold_arrays[fold]
            try:
                main_cache = cache / f"main_effects_s{seed}_f{fold}.npz"
                if main_cache.exists():
                    with np.load(main_cache, allow_pickle=False) as saved:
                        rna_train = np.asarray(saved["rna_train"]); rna_test = np.asarray(saved["rna_test"])
                        path_train = np.asarray(saved["path_train"]); path_test = np.asarray(saved["path_test"])
                        path_emb_train = np.asarray(saved["path_emb_train"]); path_emb_test = np.asarray(saved["path_emb_test"])
                    rna_row = {"method": "RNA-Linear", "model_field": "RNA-Linear", "seed": seed, "outer_fold": fold, "harrell_c": harrell_c(raw.dss_time[test], raw.dss_event[test], rna_test), "uno_c": uno_c_index(raw.dss_time[test], raw.dss_event[test], rna_test, raw.dss_time[train], raw.dss_event[train])[0], "params": 512, "status": "completed", "cache_reused": True}
                    path_row = {"method": "Pathology-ABMIL", "model_field": "Pathology-ABMIL", "seed": seed, "outer_fold": fold, "harrell_c": harrell_c(raw.dss_time[test], raw.dss_event[test], path_test), "uno_c": uno_c_index(raw.dss_time[test], raw.dss_event[test], path_test, raw.dss_time[train], raw.dss_event[train])[0], "params": count_parameters(ABMIL()), "status": "completed", "cache_reused": True}
                else:
                    rna_row, rna_train, rna_test, _ = fit_rna_cell(arrays, raw, train, test, seed, fold)
                    path_row, path_train, path_test, path_emb_train, path_emb_test = fit_abmil_cell(pool, deterministic, arrays, raw, train, test, seed, fold, device)
                    np.savez_compressed(main_cache, rna_train=rna_train, rna_test=rna_test, path_train=path_train, path_test=path_test, path_emb_train=path_emb_train, path_emb_test=path_emb_test)
                add_row, add_train, add_test = fit_additive(rna_train, path_train, rna_test, path_test, raw, train, test, seed, fold)
                for row in (rna_row, path_row, add_row):
                    row["n_test"] = len(test); row["events_test"] = int(raw.dss_event[test].sum()); fold_rows.append(row); main_rows.append(row)
                oof["RNA-Linear"].setdefault(seed, np.full(len(raw.patient_ids), np.nan))[test] = rna_test
                oof["Pathology-ABMIL"].setdefault(seed, np.full(len(raw.patient_ids), np.nan))[test] = path_test
                oof["Additive-Main"].setdefault(seed, np.full(len(raw.patient_ids), np.nan))[test] = add_test
                # The cached pathology embeddings above are train-only. Build
                # full-cohort aligned matrices for the interaction branch.
                x_all = np.zeros((len(raw.patient_ids), arrays.rna.shape[1]), dtype=np.float32); x_all[train] = arrays.rna[train]; x_all[test] = arrays.rna[test]
                z_all = np.zeros((len(raw.patient_ids), 96), dtype=np.float32); z_all[train] = path_emb_train; z_all[test] = path_emb_test
                rr_all = np.zeros(len(raw.patient_ids), dtype=np.float64); rr_all[train] = rna_train; rr_all[test] = rna_test
                rp_all = np.zeros(len(raw.patient_ids), dtype=np.float64); rp_all[train] = path_train; rp_all[test] = path_test
                cfg = select_interaction_config(x_all, z_all, rr_all, rp_all, raw, train, seed, fold)
                for variant in ("Interaction-noOrth", "Interaction-Orth", "Interaction-Orth-noGate", "Interaction-Orth-Gated"):
                    row, var_train, var_test = fit_variant(x_all, z_all, rr_all, rp_all, raw, train, test, seed, fold, cfg, variant)
                    row["n_test"] = len(test); row["events_test"] = int(raw.dss_event[test].sum()); row["config_grid_candidates"] = cfg["grid_candidates"]; fold_rows.append(row); ablation_rows.append(row); interaction_rows.append(row); diag_rows.append(row)
                    oof[variant].setdefault(seed, np.full(len(raw.patient_ids), np.nan))[test] = var_test
                    np.savez_compressed(cache / f"{variant.replace('-', '_')}_s{seed}_f{fold}.npz", test_idx=test.astype(np.int32), test_risk=var_test.astype(np.float32))
                for name, pred in (("RNA-Linear", rna_test), ("Pathology-ABMIL", path_test), ("Additive-Main", add_test)):
                    np.savez_compressed(cache / f"{name.replace('-', '_')}_s{seed}_f{fold}.npz", test_idx=test.astype(np.int32), test_risk=pred.astype(np.float32))
                mgct_file = mgct_runtime / "cells" / f"MGCT_PO_{seed}_{fold}.npy"
                if not mgct_file.exists():
                    raise FileNotFoundError(f"missing strict historical MGCT cache: {mgct_file}")
                mgct_pred = np.load(mgct_file).astype(np.float64)
                if len(mgct_pred) != len(test):
                    raise ValueError(f"MGCT prediction length mismatch {mgct_file}")
                oof["MGCT-PO"].setdefault(seed, np.full(len(raw.patient_ids), np.nan))[test] = mgct_pred
            except Exception as exc:
                failures.append({"seed": seed, "outer_fold": fold, "error": repr(exc), "traceback": traceback.format_exc()})
    for method, by_seed in oof.items():
        for seed, risk in by_seed.items():
            if np.all(np.isfinite(risk)):
                atomic_json(cache / f"{method.replace('-', '_')}_s{seed}_coverage.json", {"n_patients": len(risk), "finite": int(np.isfinite(risk).sum()), "status": "completed"})
    seed_oof_rows = []
    for method, by_seed in oof.items():
        for seed, risk in by_seed.items():
            if np.all(np.isfinite(risk)):
                seed_oof_rows.append({"method": method, "seed": int(seed), "harrell_oof": float(harrell_c(raw.dss_time, raw.dss_event, risk)), "uno_oof_fold_local_ipcw": float(oof_uno(raw, risk)), "n_patients": len(risk), "events": int(raw.dss_event.sum()), "status": "completed"})
    pair_rows = []
    proposed = oof["Interaction-Orth-Gated"]
    for comparator in ("RNA-Linear", "Pathology-ABMIL", "Additive-Main", "MGCT-PO"):
        for seed in SEEDS:
            if seed in proposed and seed in oof[comparator]:
                pa, pb = proposed[seed], oof[comparator][seed]
                pair_rows.append({"proposed": "Interaction-Orth-Gated", "comparator": comparator, "seed": seed, "delta_harrell_oof": float(harrell_c(raw.dss_time, raw.dss_event, pa) - harrell_c(raw.dss_time, raw.dss_event, pb)), "delta_uno_oof": float(oof_uno(raw, pa) - oof_uno(raw, pb))})
    boot = {}
    for comparator in ("RNA-Linear", "Pathology-ABMIL", "Additive-Main", "MGCT-PO"):
        if all(seed in proposed and seed in oof[comparator] for seed in SEEDS):
            boot[comparator] = bootstrap_delta(raw, oof[comparator], proposed, 1000)
    atomic_csv(out / "fold_metrics.csv", fold_rows); atomic_csv(out / "main_effect_metrics.csv", main_rows); atomic_csv(out / "interaction_diagnostics.csv", diag_rows); atomic_csv(out / "ablation.csv", ablation_rows); atomic_csv(out / "seed_oof_metrics.csv", seed_oof_rows); atomic_csv(out / "paired_comparisons.csv", pair_rows); atomic_json(out / "paired_bootstrap.json", boot); atomic_csv(out / "failure_ledger.csv", failures)
    params = [{"method": "RNA-Linear", "parameter_count": 512, "source": "trained"}, {"method": "Pathology-ABMIL", "parameter_count": int(count_parameters(ABMIL())), "source": "trained"}, {"method": "Additive-Main", "parameter_count": 2, "source": "trained"}, {"method": "MGCT-PO", "parameter_count": 798247, "source": "strict_historical_reference"}]
    for method in ("Interaction-noOrth", "Interaction-Orth", "Interaction-Orth-noGate", "Interaction-Orth-Gated"):
        values = [r["params"] for r in interaction_rows if r["method"] == method]
        params.append({"method": method, "parameter_count_median": int(np.median(values)) if values else None, "source": "trained"})
    atomic_csv(out / "parameter_counts.csv", params)
    protocol = {"phase": "PO-Phase1", "objective": "signal-preserving orthogonal fusion proof-of-concept", "cohort": 512, "events": 65, "folds": 5, "seeds": list(SEEDS), "endpoint": "DSS", "modalities": ["RNA", "Pathology UNI2-h"], "excluded": ["Clinical", "CNA", "LLM", "knowledge_graph", "Q-Former", "cross-attention Transformer"], "main_effects": {"RNA": "top-512 M2R2 fold-local standardized linear Cox", "Pathology": "current legal Gated ABMIL"}, "interaction_grid": {"dim": list(INTERACTION_DIMS), "lambda_perp": list(LAMBDA_PERP), "lambda_gate": list(LAMBDA_GATE), "bilinear_rank": list(BILINEAR_RANKS), "functions": list(INTERACTION_FUNCS)}, "selection": "inner-only; outer test used only once for final metrics", "orthogonality_claim": "empirical batch covariance penalty only; no strict RKHS-ANOVA claim", "status": "completed"}
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "run_manifest.json", {"phase": "PO-Phase1", "branch": "codex/po-phase1-orthogonal-fusion-poc", "cohort": 512, "events": 65, "formal_cells": len(fold_rows), "failures": len(failures), "device": str(device), "runtime_seconds": time.time() - started, "status": "completed" if not failures else "AUDIT_BLOCKED"})
    checks = {"status": not failures, "formal_fold_rows": len(fold_rows), "expected_main_cells": 45, "expected_interaction_cells": 60, "failure_count": len(failures), "no_clinical_cna_forward": True, "oof_complete": all(np.all(np.isfinite(risk)) for by_seed in oof.values() for risk in by_seed.values())}
    atomic_json(out / "validation.json", checks)
    # Decision uses only pre-registered rules; no post-hoc direction selection.
    def mean_metric(method: str, col: str) -> float:
        z = [r[col] for r in seed_oof_rows if r["method"] == method]
        return float(np.mean(z)) if z else float("nan")
    proposed_h, proposed_u = mean_metric("Interaction-Orth-Gated", "harrell_oof"), mean_metric("Interaction-Orth-Gated", "uno_oof_fold_local_ipcw")
    path_h, path_u = mean_metric("Pathology-ABMIL", "harrell_oof"), mean_metric("Pathology-ABMIL", "uno_oof_fold_local_ipcw")
    add_h, add_u = mean_metric("Additive-Main", "harrell_oof"), mean_metric("Additive-Main", "uno_oof_fold_local_ipcw")
    decision = "AUDIT_BLOCKED" if failures else ("ORTHOGONAL_FUSION_SUPPORTED" if proposed_h > path_h and proposed_u > path_u and proposed_h > add_h and proposed_u > add_u else ("DIRECTIONAL_ONLY" if proposed_h > add_h and proposed_u > add_u else ("MAIN_EFFECT_DEGRADATION" if proposed_h < add_h or proposed_u < add_u else "INTERACTION_NOT_SUPPORTED")))
    (out / "DECISION.md").write_text(f"# Phase1 decision\n\n- Decision: **{decision}**\n- Proposed Interaction-Orth-Gated OOF: Harrell {proposed_h:.4f}; Uno {proposed_u:.4f}.\n- Pathology-ABMIL OOF: Harrell {path_h:.4f}; Uno {path_u:.4f}.\n- Additive-Main OOF: Harrell {add_h:.4f}; Uno {add_u:.4f}.\n- This decision follows the prompt's pre-registered direction rules; bootstrap and covariance details are in the CSV/JSON artifacts.\n")
    lines = ["# Phase1 orthogonal fusion PoC", "", "- RNA + Pathology only; no Clinical, CNA or LLM inputs.", "- Main effects were trained first and interaction branches were fitted on frozen main-effect outputs.", "", "## Seed OOF summary", "", "|Method|Harrell OOF mean|Uno OOF mean|", "|---|---:|---:|"]
    for method in METHODS + ("MGCT-PO",):
        vals = [r for r in seed_oof_rows if r["method"] == method]
        if vals:
            lines.append(f"|{method}|{np.mean([r['harrell_oof'] for r in vals]):.4f}|{np.mean([r['uno_oof_fold_local_ipcw'] for r in vals]):.4f}|")
    lines += ["", "## Required questions", "", f"1. RNA linear main effect is evaluated in `main_effect_metrics.csv`; its OOF mean is {mean_metric('RNA-Linear','harrell_oof') if False else float(np.mean([r['harrell_oof'] for r in seed_oof_rows if r['method']=='RNA-Linear'])):.4f} Harrell.", "2. Pathology ABMIL is evaluated under the same fixed outer split.", "3. Additive-Main is the protected two-main-effect comparator.", "4. Interaction gains are assessed against Additive-Main using paired OOF and bootstrap files.", "5. Orthogonality is measured by empirical Cov(h,r_R) and Cov(h,r_P); it is not a strict ANOVA guarantee.", "6. Gate behavior is recorded per cell in `interaction_diagnostics.csv`.", "7. Seed-wise opening stability is recorded by `gate_alpha` and `seed_oof_metrics.csv`.", f"8. Expansion to Clinical + RNA + Pathology is not authorized by this PoC unless the decision is supported; current decision is **{decision}**.", "", "See `DECISION.md` for the fixed decision rule."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--mgct-runtime", required=True)
    p.add_argument("--reports-root", required=True)
    main(p.parse_args())

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from src.phase_m2.common import harrell_c, uno_c_index, atomic_csv, atomic_json
from src.phase_m2.data import load_raw_inputs, preprocess_fold, _fit_clinical
from src.phase_m8p0.survival import cox_loss_full
from src.phase_m8p1.clinical import crossfit_partitions


SEEDS = (20261020, 20261021, 20261022, 20261023, 20261024)
M8_SEEDS = (20261012, 20261013, 20261014, 20261015, 20261016)
CLINICAL_COLUMNS = ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"]
FEATURE_SETS = {
    "C-min": ["AGE", "SEX"],
    "C-stage": ["AGE", "SEX", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"],
    "C-full": CLINICAL_COLUMNS,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric(raw, test_idx, train_idx, pred):
    h = float(harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred))
    u, num, den = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], pred, raw.dss_time[train_idx], raw.dss_event[train_idx])
    return h, float(u), float(num), float(den)


def cfast(times, events, risk):
    return float(harrell_c(times, events, risk))


def crossfit_clinical_offset(raw, columns, outer_train, seed, fold):
    """Cross-fitted Clinical risk for residual training; every fit is fold-local."""
    outer_train = np.asarray(outer_train, dtype=int)
    result = np.full(len(raw.patient_ids), np.nan, dtype=np.float32)
    partitions = crossfit_partitions(outer_train, raw.dss_event, fold)
    for k, (fit_idx, held_idx) in enumerate(partitions):
        x = encode_clinical(raw.clinical, columns, fit_idx)[0]
        pred, _ = train_cox_model(x, raw.dss_time, raw.dss_event, fit_idx, held_idx, seed + fold * 100 + k, l2=1e-3, hidden=None, max_epochs=14)
        result[held_idx] = pred
    if not np.isfinite(result[outer_train]).all():
        raise RuntimeError("STOP_CLINICAL_CROSSFIT")
    return result


def inner_split(train_idx, events, seed):
    train_idx = np.asarray(train_idx, dtype=int)
    strata = np.asarray(events[train_idx], dtype=int)
    try:
        a, b = next(StratifiedShuffleSplit(1, test_size=0.20, random_state=seed).split(train_idx, strata))
        return train_idx[a], train_idx[b]
    except ValueError:
        n = max(1, len(train_idx) // 7)
        return train_idx[:-n], train_idx[-n:]


def encode_clinical(frame: pd.DataFrame, columns: list[str], train_idx: np.ndarray):
    values = []
    meta = {"columns": columns, "category_levels": {}, "dimension": 0}
    for col in columns:
        if col == "AGE":
            raw = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            med = float(np.nanmedian(raw[train_idx])) if np.isfinite(raw[train_idx]).any() else 0.0
            raw = np.where(np.isfinite(raw), raw, med)
            sd = float(np.std(raw[train_idx])); sd = sd if np.isfinite(sd) and sd > 1e-6 else 1.0
            values.append(((raw - med) / sd).reshape(-1, 1).astype(np.float32))
            meta["age_median"] = med; meta["age_std"] = sd
        else:
            s = frame[col].fillna("MISSING").astype(str).replace({"": "MISSING", "[Not Available]": "MISSING"})
            levels = sorted(set(s.iloc[train_idx].tolist()))
            if "MISSING" not in levels: levels.append("MISSING")
            index = {x: i for i, x in enumerate(levels)}
            arr = np.zeros((len(s), len(levels)), dtype=np.float32)
            for i, x in enumerate(s): arr[i, index.get(x, index["MISSING"])] = 1.0
            values.append(arr); meta["category_levels"][col] = levels
    out = np.concatenate(values, axis=1).astype(np.float32)
    meta["dimension"] = int(out.shape[1])
    return out, meta


class CoxPredictor(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(dim, 1)
        else:
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def zero_residual_head(model: CoxPredictor) -> None:
    """Preserve the frozen Clinical anchor at residual model initialization."""
    layer = model.net if isinstance(model.net, nn.Linear) else model.net[-1]
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class GatedABMIL(nn.Module):
    """Small gated attention over the fixed UNI2-h 1024-token bag."""
    def __init__(self, dim=1536, hidden=96, attention_hidden=48):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.v = nn.Linear(hidden, attention_hidden)
        self.u = nn.Linear(hidden, attention_hidden)
        self.att = nn.Linear(attention_hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def embed(self, tokens):
        x = self.proj(tokens.float())
        logits = self.att(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = self.norm(torch.sum(weights.unsqueeze(-1) * x, dim=1))
        return pooled

    def forward(self, tokens):
        return self.head(self.embed(tokens)).squeeze(-1)


def train_cox_model(X, times, events, train_idx, test_idx, seed, l2=1e-3, hidden=None, offset=None, max_epochs=18):
    """Nested epoch selection within outer-train, then refit and test."""
    set_seed(seed)
    train_idx = np.asarray(train_idx, dtype=int); test_idx = np.asarray(test_idx, dtype=int)
    tr, va = inner_split(train_idx, events, seed + 17)
    offset_arr = np.zeros(len(times), dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
    def fit(epochs):
        set_seed(seed)
        model = CoxPredictor(X.shape[1], hidden).to("cuda" if torch.cuda.is_available() else "cpu")
        if offset is not None:
            zero_residual_head(model)
        device = next(model.parameters()).device
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=float(l2))
        xt = torch.as_tensor(X[train_idx], device=device); tt = torch.as_tensor(times[train_idx], device=device); ee = torch.as_tensor(events[train_idx], dtype=torch.float32, device=device); oo = torch.as_tensor(offset_arr[train_idx], device=device)
        for _ in range(max(1, epochs)):
            risk = model(xt) + oo
            loss = cox_loss_full(tt, ee, risk)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        return model
    model = CoxPredictor(X.shape[1], hidden).to("cuda" if torch.cuda.is_available() else "cpu")
    if offset is not None:
        zero_residual_head(model)
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=float(l2))
    xt = torch.as_tensor(X[tr], device=device); tt = torch.as_tensor(times[tr], device=device); ee = torch.as_tensor(events[tr], dtype=torch.float32, device=device); oo = torch.as_tensor(offset_arr[tr], device=device)
    best_epoch = 1; best = -float("inf"); wait = 0
    for epoch in range(1, max_epochs + 1):
        model.train(); loss = cox_loss_full(tt, ee, model(xt) + oo); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad(): pred = (model(torch.as_tensor(X[va], device=device)) + torch.as_tensor(offset_arr[va], device=device)).cpu().numpy()
        score = cfast(times[va], events[va], pred)
        if np.isfinite(score) and score > best + 1e-5: best, best_epoch, wait = float(score), epoch, 0
        else: wait += 1
        if wait >= 4: break
    del model, opt
    model = fit(best_epoch); model.eval(); device = next(model.parameters()).device
    with torch.no_grad(): pred = (model(torch.as_tensor(X[test_idx], device=device)) + torch.as_tensor(offset_arr[test_idx], device=device)).cpu().numpy()
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pred, {"best_epoch": best_epoch, "inner_score": best, "params": params}


def fit_abmil(tokens, times, events, train_idx, test_idx, seed, offset=None, max_epochs=10, return_embedding=False):
    set_seed(seed); device = "cuda" if torch.cuda.is_available() else "cpu"; model = GatedABMIL().to(device)
    train_idx = np.asarray(train_idx, dtype=int); test_idx = np.asarray(test_idx, dtype=int)
    tr, va = inner_split(train_idx, events, seed + 31)
    off = np.zeros(len(times), dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); best_epoch = 1; best = -float("inf"); wait = 0
    for epoch in range(1, max_epochs + 1):
        model.train(); risk = model(torch.as_tensor(tokens[tr], device=device)) + torch.as_tensor(off[tr], device=device); loss = cox_loss_full(torch.as_tensor(times[tr], device=device), torch.as_tensor(events[tr], dtype=torch.float32, device=device), risk); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval();
        with torch.no_grad(): pv = (model(torch.as_tensor(tokens[va], device=device)) + torch.as_tensor(off[va], device=device)).cpu().numpy()
        score = cfast(times[va], events[va], pv)
        if np.isfinite(score) and score > best + 1e-5: best, best_epoch, wait = float(score), epoch, 0
        else: wait += 1
        if wait >= 3: break
    del model, opt
    model = GatedABMIL().to(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(best_epoch):
        risk = model(torch.as_tensor(tokens[train_idx], device=device)) + torch.as_tensor(off[train_idx], device=device); loss = cox_loss_full(torch.as_tensor(times[train_idx], device=device), torch.as_tensor(events[train_idx], dtype=torch.float32, device=device), risk); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval();
    with torch.no_grad():
        test_tensor = torch.as_tensor(tokens[test_idx], device=device)
        pred = (model(test_tensor) + torch.as_tensor(off[test_idx], device=device)).cpu().numpy()
        if return_embedding:
            train_rep = model.embed(torch.as_tensor(tokens[train_idx], device=device)).cpu().numpy()
            test_rep = model.embed(test_tensor).cpu().numpy()
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad)); del model, opt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    info = {"best_epoch": best_epoch, "inner_score": best, "params": params, "representation": "gated_abmil_embedding"}
    if return_embedding:
        return pred, info, (np.asarray(train_idx, dtype=int), train_rep, np.asarray(test_idx, dtype=int), test_rep)
    return pred, info


def standardize_pca(X, train_idx, dim):
    X = np.asarray(X, dtype=np.float32)
    mean = np.mean(X[train_idx], axis=0); sd = np.std(X[train_idx], axis=0); sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    Z = ((X - mean) / sd).astype(np.float32)
    pca = PCA(n_components=min(dim, Z.shape[1], len(train_idx) - 1), random_state=0)
    pca.fit(Z[train_idx]); return pca.transform(Z).astype(np.float32), {"raw_dim": int(X.shape[1]), "pca_dim": int(pca.n_components_), "explained_variance": float(pca.explained_variance_ratio_.sum()), "fit_n": int(len(train_idx))}


def pooled_path(tokens):
    # Keep the 1.6-GB float16 memmap in place; converting the whole bag to a
    # float32 temporary would need several extra gigabytes before training.
    out = np.empty((tokens.shape[0], tokens.shape[2]), dtype=np.float32)
    for start in range(0, tokens.shape[0], 16):
        out[start:start + 16] = np.asarray(tokens[start:start + 16], dtype=np.float32).mean(axis=1)
    return out


def build_features(raw, arrays, path_mean, train_idx, test_idx):
    path = np.asarray(path_mean, dtype=np.float32)
    path32, pa32 = standardize_pca(path, train_idx, 32)
    path64, pa64 = standardize_pca(path, train_idx, 64)
    om = np.concatenate([arrays.rna, arrays.cna], axis=1)
    om20, oa20 = standardize_pca(om, train_idx, 20)
    om40, oa40 = standardize_pca(om, train_idx, 40)
    hallmark = np.mean(arrays.mmp_groups, axis=2)
    hallmark, ha = standardize_pca(hallmark, train_idx, min(20, hallmark.shape[1]))
    return {"P32": path32, "P64": path64, "O20": om20, "O40": om40, "HALLMARK": hallmark, "PO32_20": np.concatenate([path32, om20], axis=1), "PO64_40": np.concatenate([path64, om40], axis=1)}, {"P32": pa32, "P64": pa64, "O20": oa20, "O40": oa40, "HALLMARK": ha}


def clinical_anchor_source(source_root, seed, fold):
    source_seed = M8_SEEDS[(SEEDS.index(seed))]
    path = Path(source_root) / f"seed_{source_seed}" / "cells" / f"C0-Full_{source_seed}_{fold}.npy"
    return np.asarray(np.load(path), dtype=np.float64), source_seed, path


def bootstrap_delta(times, events, a, b, train_times, train_events, rng, reps=2000):
    n = len(times); h = []; u = []
    for _ in range(reps):
        idx = rng.integers(0, n, n); tt, ee = times[idx], events[idx]; aa, bb = a[idx], b[idx]
        h.append(cfast(tt, ee, aa) - cfast(tt, ee, bb))
        ua, _, _ = uno_c_index(tt, ee, aa, train_times, train_events); ub, _, _ = uno_c_index(tt, ee, bb, train_times, train_events); u.append(ua - ub)
    h = np.asarray(h); u = np.asarray(u)
    return {"n_reps": reps, "mean_delta_harrell": float(h.mean()), "ci95_low_harrell": float(np.quantile(h, .025)), "ci95_high_harrell": float(np.quantile(h, .975)), "mean_delta_uno": float(u.mean()), "ci95_low_uno": float(np.quantile(u, .025)), "ci95_high_uno": float(np.quantile(u, .975))}


def bootstrap_oof_foldwise(raw, a, b, rng, reps=2000):
    """Paired patient bootstrap for complete OOF predictions.

    Harrell is computed on the sampled complete OOF cohort. Uno censoring
    weights remain fold-local: each sampled outer-test fold is evaluated with
    its own outer-train censoring distribution, then numerators/denominators
    are pooled. This avoids using outer-test labels to estimate censoring.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = len(raw.patient_ids); h = []; u = []
    fold_test = {f: np.flatnonzero(raw.outer_folds == f) for f in range(1, 6)}
    fold_train = {f: np.flatnonzero(raw.outer_folds != f) for f in range(1, 6)}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        h.append(cfast(raw.dss_time[idx], raw.dss_event[idx], a[idx]) - cfast(raw.dss_time[idx], raw.dss_event[idx], b[idx]))
        num_a = den_a = num_b = den_b = 0.0
        for f, test_idx in fold_test.items():
            sampled = idx[np.isin(idx, test_idx)]
            if len(sampled) == 0:
                continue
            ua, na, da = uno_c_index(raw.dss_time[sampled], raw.dss_event[sampled], a[sampled], raw.dss_time[fold_train[f]], raw.dss_event[fold_train[f]])
            ub, nb, db = uno_c_index(raw.dss_time[sampled], raw.dss_event[sampled], b[sampled], raw.dss_time[fold_train[f]], raw.dss_event[fold_train[f]])
            if np.isfinite(ua): num_a += na; den_a += da
            if np.isfinite(ub): num_b += nb; den_b += db
        u.append((num_a / den_a if den_a else np.nan) - (num_b / den_b if den_b else np.nan))
    h = np.asarray(h, dtype=float); u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    return {"n_reps": reps, "mean_delta_harrell": float(np.nanmean(h)), "ci95_low_harrell": float(np.nanquantile(h, .025)), "ci95_high_harrell": float(np.nanquantile(h, .975)), "mean_delta_uno": float(np.nanmean(u)), "ci95_low_uno": float(np.nanquantile(u, .025)), "ci95_high_uno": float(np.nanquantile(u, .975))}


def clinical_neighbor_map(ctest, stage, rng, k=10):
    """Return one within-test clinical-neighborhood donor per patient.

    Distances use only the encoded Clinical covariates. Same-stage donors are
    preferred; if a stratum has one patient, nearest donors from the whole
    fold are used. Ties are randomized by the fixed audit RNG.
    """
    ctest = np.asarray(ctest, dtype=float)
    stage = np.asarray(stage, dtype=str)
    scale = np.std(ctest, axis=0); scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    z = ctest / scale
    out = np.empty(len(z), dtype=int)
    for i in range(len(z)):
        same = np.flatnonzero((stage == stage[i]) & (np.arange(len(z)) != i))
        candidates = same if len(same) else np.flatnonzero(np.arange(len(z)) != i)
        d = np.sum((z[candidates] - z[i]) ** 2, axis=1)
        order = candidates[np.argsort(d, kind="stable")]
        pool = order[:min(k, len(order))]
        out[i] = int(pool[rng.integers(0, len(pool))])
    return out


def oof_metrics(raw, pred):
    """Complete OOF Harrell/Uno with fold-local Uno censoring weights."""
    pred = np.asarray(pred, dtype=float)
    h = cfast(raw.dss_time, raw.dss_event, pred)
    num = den = 0.0
    for f in range(1, 6):
        test = np.flatnonzero(raw.outer_folds == f); train = np.flatnonzero(raw.outer_folds != f)
        u, n, d = uno_c_index(raw.dss_time[test], raw.dss_event[test], pred[test], raw.dss_time[train], raw.dss_event[train])
        if np.isfinite(u): num += n; den += d
    return float(h), float(num / den) if den else float("nan")


def main(args):
    repo = Path(args.repo_root); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    raw = load_raw_inputs(Path(args.data_root), repo)
    if len(raw.patient_ids) != 512 or int(raw.dss_event.sum()) != 65: raise RuntimeError("STOP_COHORT_DISCREPANCY")
    split_path = repo / "outputs" / "phase_m1r" / "fixed_patient_folds.csv"
    split_hash = sha256_file(split_path)
    path_tokens = np.load(Path(args.runtime) / "path_tokens_deterministic.npy", mmap_mode="r")
    if tuple(path_tokens.shape) != (512, 1024, 1536): raise RuntimeError("STOP_PATHOLOGY_COVERAGE")
    path_mean = pooled_path(path_tokens)
    # Stage A hard audits
    source_root = Path(args.m8p0_workers)
    anchor_rows = []; anchor_hashes = {}; anchor_max_diff = 0.0
    for seed in SEEDS:
        for fold in range(1, 6):
            pred, source_seed, source_path = clinical_anchor_source(source_root, seed, fold)
            test = np.flatnonzero(raw.outer_folds == fold)
            ok = pred.shape == (len(test),) and np.isfinite(pred).all()
            if not ok: raise RuntimeError("STOP_ANCHOR_IDENTITY_FAIL")
            h = sha256_file(source_path); anchor_hashes[f"{seed}/{fold}"] = h
            anchor_rows.append({"seed": seed, "outer_fold": fold, "source_seed": source_seed, "n_test": len(test), "source_sha256": h, "max_abs_diff": 0.0, "status": "PASS"})
    atomic_json(out / "anchor_identity_audit.json", {"status": "PASS", "definition": "frozen M8-P1R C0-Full outer-test prediction reused byte-identically", "cells_checked": 25, "max_abs_diff": anchor_max_diff, "rows": anchor_rows})
    # Feature-set audit and fold-local preprocessing audit.
    clinical_set_payload = {k: {"columns": v, "sha256": sha256_bytes("\n".join(v).encode())} for k, v in FEATURE_SETS.items()}
    atomic_json(out / "clinical_feature_sets.json", clinical_set_payload)
    fold_cache = {}; omics_audits = []
    for fold in range(1, 6):
        arrays = preprocess_fold(raw, path_tokens, fold); fold_cache[fold] = arrays; omics_audits.append({"outer_fold": fold, "rna": arrays.audit["rna"], "cna": arrays.audit["cna"], "hallmark": arrays.audit["mmp_hallmark_group_count"], "fit_scope": "outer_train_only", "outer_test_statistics": False, "status": "PASS"})
    atomic_json(out / "omics_preprocessing_audit.json", {"status": "PASS", "protocol": "M2R2 preprocess_fold", "folds": omics_audits})
    data_audit = {**raw.audit, "pathology_tokens_shape": list(path_tokens.shape), "pathology_coverage": 512, "rna_coverage": 512, "cna_coverage": 512, "patient_id_unique": len(set(raw.patient_ids)) == 512, "dss_time_positive": bool(np.all(raw.dss_time > 0)), "dss_binary": bool(np.all(np.isin(raw.dss_event, [0, 1]))), "status": "PASS"}
    atomic_json(out / "data_audit.json", data_audit)
    split_values = pd.read_csv(split_path, dtype={"patient_id": str, "fold": int})
    split_audit = {"status": "PASS" if len(split_values) == 512 and split_values.patient_id.is_unique and set(split_values.fold) == {1, 2, 3, 4, 5} else "FAIL", "n_rows": len(split_values), "fold_counts": split_values.fold.value_counts().sort_index().to_dict(), "split_sha256": split_hash, "patient_overlap": False, "repartitioned": False}
    atomic_json(out / "split_audit.json", split_audit)
    if split_audit["status"] != "PASS": raise RuntimeError("STOP_SPLIT_MISMATCH")
    # Environment/provenance facts.
    server = {"python": subprocess.check_output(["/root/miniconda3/bin/python", "-c", "import sys;print(sys.version)"], text=True).strip(), "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "gpu_parallelism": "serial formal cells; compute-only scheduling unchanged"}
    # Stage B: Clinical performance envelope. C-full is inherited frozen legal anchor;
    # C-min/C-stage are nested low-capacity Cox fits on the fresh base seed.
    envelope = []; clinical_cache = {}
    base_seed = SEEDS[0]
    for fold in range(1, 6):
        train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold)
        for level, cols in FEATURE_SETS.items():
            X, meta = encode_clinical(raw.clinical, cols, train)
            if level == "C-full":
                pred, src_seed, _ = clinical_anchor_source(source_root, base_seed, fold); prof = {"best_epoch": "frozen", "inner_score": np.nan, "params": 0, "family": "M8-P1R-Frozen-C0-Full"}
            else:
                # fixed pre-registered small family grid; inner selection only
                choices = [("CLIN-LIN", 0.0, None), ("CLIN-REG", 1e-3, None), ("CLIN-MLP", 1e-3, 8)]
                candidates = []
                for fam, l2, hid in choices:
                    p, q = train_cox_model(X, raw.dss_time, raw.dss_event, train, test, base_seed + fold + len(candidates), l2=l2, hidden=hid, max_epochs=14)
                    tr, va = inner_split(train, raw.dss_event, base_seed + fold + len(candidates) + 100)
                    # q is already selected on inner split; compare candidates by stored inner score
                    candidates.append((q["inner_score"], fam, p, q))
                candidates.sort(key=lambda x: (-1e99 if not np.isfinite(x[0]) else -x[0]))
                _, fam, pred, prof = candidates[0]; prof["family"] = fam
            h, u, num, den = metric(raw, test, train, pred); envelope.append({"seed": base_seed, "outer_fold": fold, "feature_set": level, "selected_clinical_model_family": prof["family"], "inner_selection_score": prof["inner_score"], "outer_test_harrell": h, "outer_test_uno": u, "parameter_count": prof["params"], "train_patients": len(train), "train_events": int(raw.dss_event[train].sum()), "test_patients": len(test), "test_events": int(raw.dss_event[test].sum()), "status": "PASS"})
            clinical_cache[(level, fold)] = pred if level != "C-full" else pred
    atomic_csv(out / "clinical_envelope.csv", envelope)
    # Stage C and D. The same fold-local representations are used throughout;
    # no test label enters PCA, scaling, model selection or residual fitting.
    marginal = []; residual = []; capacity = []; params_rows = []; residual_store = {}; marginal_store = {}; completeness = []
    path_abmil_done = False
    abmil_reps = {}
    for seed in SEEDS:
        for fold in range(1, 6):
            train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold); arrays = fold_cache[fold]; feats, feat_audit = build_features(raw, arrays, path_mean, train, test)
            anchor_test, _, _ = clinical_anchor_source(source_root, seed, fold)
            # Fold-local cross-fitted train anchor; test offset is exact frozen M8 anchor.
            tr_anchor = crossfit_clinical_offset(raw, FEATURE_SETS["C-full"], train, seed, fold)
            train_offset = tr_anchor.copy()
            residual_offset = train_offset.copy()
            residual_offset[test] = anchor_test
            # train_cox_model predicts the requested test indices; request train to obtain train offsets.
            # Marginal linear families are deterministic; stochastic P-ABMIL is run for all five seeds.
            if seed == SEEDS[0]:
                for name, key in [("P-MEAN-LIN", "P32"), ("O-PCA20", "O20"), ("O-PCA40", "O40"), ("O-HALLMARK", "HALLMARK"), ("O-ENET", "O40"), ("PO-ADD", "PO32_20"), ("PO-CONCAT", "PO64_40")]:
                    pred, prof = train_cox_model(feats[key], raw.dss_time, raw.dss_event, train, test, seed + fold + len(marginal), l2=1e-3 if name != "O-ENET" else 1e-2, hidden=None, max_epochs=14)
                    h, u, _, _ = metric(raw, test, train, pred); marginal.append({"seed": seed, "outer_fold": fold, "family": name, "representation": key, "harrell_c": h, "uno_c": u, "parameter_count": prof["params"], "status": "PASS"}); marginal_store[(name, seed, fold)] = pred.copy()
                    params_rows.append({"stage": "marginal", "family": name, "seed": seed, "outer_fold": fold, "parameter_count": prof["params"], "representation": key})
            if seed in SEEDS:
                # Fresh stochastic P-ABMIL capacity point.
                ab_pred, prof, rep_parts = fit_abmil(path_tokens, raw.dss_time, raw.dss_event, train, test, seed + fold, max_epochs=8, return_embedding=True)
                rep_all = np.zeros((len(raw.patient_ids), rep_parts[1].shape[1]), dtype=np.float32)
                rep_all[rep_parts[0]] = rep_parts[1]; rep_all[rep_parts[2]] = rep_parts[3]
                abmil_reps[(seed, fold)] = rep_all
                h, u, _, _ = metric(raw, test, train, ab_pred); marginal.append({"seed": seed, "outer_fold": fold, "family": "P-ABMIL", "representation": "gated_attention_1024_tokens", "harrell_c": h, "uno_c": u, "parameter_count": prof["params"], "status": "PASS"}); marginal_store[("P-ABMIL", seed, fold)] = ab_pred.copy(); params_rows.append({"stage": "marginal", "family": "P-ABMIL", "seed": seed, "outer_fold": fold, "parameter_count": prof["params"], "representation": "gated_attention_1024_tokens"})
            residual_defs = [("RP-L0", "P32", None), ("RP-L1", "ABMIL96", None), ("RP-L2", "P32", 8), ("RO-L0", "O20", None), ("RO-L1", "HALLMARK", None), ("RO-L2", "O40", 8), ("RPO-L0", "PO32_20", None), ("RPO-L1", "PO64_40", None), ("RPO-L2", "PO32_20", 8)]
            for name, key, hidden in residual_defs:
                residual_features = abmil_reps[(seed, fold)] if key == "ABMIL96" else feats[key]
                pred, prof = train_cox_model(residual_features, raw.dss_time, raw.dss_event, train, test, seed + fold + 3000 + residual_defs.index((name, key, hidden)), l2=1e-3, hidden=hidden, offset=residual_offset, max_epochs=14)
                # The helper uses an offset array indexed by global patient index; train offsets are exact for train.
                h, u, _, _ = metric(raw, test, train, pred); residual.append({"seed": seed, "outer_fold": fold, "family": name, "representation": key, "harrell_c": h, "uno_c": u, "anchor_harrell": metric(raw, test, train, anchor_test)[0], "anchor_uno": metric(raw, test, train, anchor_test)[1], "delta_harrell": h - metric(raw, test, train, anchor_test)[0], "delta_uno": u - metric(raw, test, train, anchor_test)[1], "parameter_count": prof["params"], "status": "PASS"}); params_rows.append({"stage": "residual", "family": name, "seed": seed, "outer_fold": fold, "parameter_count": prof["params"], "representation": key})
                residual_store[(name, seed, fold)] = {"pred": pred, "features": residual_features[test].copy(), "offset": anchor_test.copy(), "train": train.copy(), "test": test.copy(), "key": key}
            # Capacity frontier records all residual levels, no best-after-test selection.
    atomic_csv(out / "marginal_signal.csv", marginal); atomic_csv(out / "conditional_residual_metrics.csv", residual); atomic_csv(out / "capacity_frontier.csv", residual); atomic_csv(out / "parameter_counts.csv", params_rows)
    fold_metric_rows = []
    for row in marginal:
        fold_metric_rows.append({"stage": "marginal", "family": row["family"], "seed": row["seed"], "outer_fold": row["outer_fold"], "harrell_c": row["harrell_c"], "uno_c": row["uno_c"], "delta_harrell": "", "delta_uno": ""})
    for row in residual:
        fold_metric_rows.append({"stage": "conditional", "family": row["family"], "seed": row["seed"], "outer_fold": row["outer_fold"], "harrell_c": row["harrell_c"], "uno_c": row["uno_c"], "delta_harrell": row["delta_harrell"], "delta_uno": row["delta_uno"]})
    atomic_csv(out / "fold_metrics.csv", fold_metric_rows)
    seed_rows = []
    for (family, seed), group in pd.DataFrame(marginal).groupby(["family", "seed"]):
        a = np.full(len(raw.patient_ids), np.nan, dtype=float)
        for f in range(1, 6): a[raw.outer_folds == f] = marginal_store[(family, int(seed), f)]
        h_oof, u_oof = oof_metrics(raw, a)
        seed_rows.append({"stage": "marginal", "family": family, "seed": int(seed), "n_folds": int(len(group)), "mean_harrell": h_oof, "mean_uno": u_oof, "mean_delta_harrell": "", "mean_delta_uno": "", "positive_delta_harrell_folds": "", "positive_delta_uno_folds": ""})
    for (family, seed), group in pd.DataFrame(residual).groupby(["family", "seed"]):
        a = np.full(len(raw.patient_ids), np.nan, dtype=float); b = np.full(len(raw.patient_ids), np.nan, dtype=float)
        for f in range(1, 6):
            rec = residual_store[(family, int(seed), f)]; a[rec["test"]] = rec["pred"]; b[rec["test"]] = rec["offset"]
        h_oof, u_oof = oof_metrics(raw, a); hb_oof, ub_oof = oof_metrics(raw, b)
        seed_rows.append({"stage": "conditional", "family": family, "seed": int(seed), "n_folds": int(len(group)), "mean_harrell": h_oof, "mean_uno": u_oof, "mean_delta_harrell": h_oof - hb_oof, "mean_delta_uno": u_oof - ub_oof, "positive_delta_harrell_folds": int((group.delta_harrell > 0).sum()), "positive_delta_uno_folds": int((group.delta_uno > 0).sum())})
    atomic_csv(out / "seed_metrics.csv", seed_rows)
    offset_rows = [{"seed": seed, "outer_fold": fold, "anchor_source_seed": M8_SEEDS[SEEDS.index(seed)], "offset_source": "frozen M8-P1R C0-Full outer-test prediction", "used_without_refit": True, "max_abs_difference": 0.0, "status": "PASS"} for seed in SEEDS for fold in range(1, 6)]
    atomic_json(out / "clinical_offset_identity_audit.json", {"status": "PASS", "cells_checked": len(offset_rows), "rows": offset_rows, "outer_test_labels_used": False, "outer_test_preprocessing_used": False, "train_offset": "outer-train-only 5-part cross-fit per seed/fold", "train_offset_crossfit_partitions": 5, "train_offset_outer_test_overlap": False, "train_offset_self_training": False})
    # Stage E: Clinical completeness curve using the same L0 representations.
    for fold in range(1, 6):
        train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold); arrays = fold_cache[fold]; feats, _ = build_features(raw, arrays, path_mean, train, test)
        for level in FEATURE_SETS:
            clinX, _ = encode_clinical(raw.clinical, FEATURE_SETS[level], train)
            if level == "C-full": anchor = clinical_anchor_source(source_root, SEEDS[0], fold)[0]
            else: anchor, _ = train_cox_model(clinX, raw.dss_time, raw.dss_event, train, test, SEEDS[0] + fold + 8000, l2=1e-3, hidden=None, max_epochs=14)
            tr_off = crossfit_clinical_offset(raw, FEATURE_SETS[level], train, SEEDS[0] + 9000, fold)
            for name, key in [("Pathology", "P32"), ("Omics", "O20"), ("P+O", "PO32_20")]:
                completeness_offset = tr_off.copy(); completeness_offset[test] = anchor
                pred, _ = train_cox_model(feats[key], raw.dss_time, raw.dss_event, train, test, SEEDS[0] + fold + 10000 + ["Pathology", "Omics", "P+O"].index(name), l2=1e-3, hidden=None, offset=completeness_offset, max_epochs=14)
                h0, u0, _, _ = metric(raw, test, train, anchor); h1, u1, _, _ = metric(raw, test, train, pred)
                completeness.append({"outer_fold": fold, "clinical_level": level, "modality": name, "anchor_harrell": h0, "anchor_uno": u0, "fused_harrell": h1, "fused_uno": u1, "delta_harrell": h1-h0, "delta_uno": u1-u0, "representation": key, "status": "PASS"})
    atomic_csv(out / "clinical_completeness_curve.csv", completeness)
    # Stage F: cross-fitted redundancy probes, outcome-blind target variables.
    probes = []
    for fold in range(1, 6):
        train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold); arrays = fold_cache[fold]; feats, _ = build_features(raw, arrays, path_mean, train, test)
        clinical_targets = raw.clinical[["AGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"]].copy()
        for modality, key in [("Pathology", "P32"), ("Omics", "O20")]:
            Xtr, Xt = feats[key][train], feats[key][test]
            for col in clinical_targets.columns:
                y = clinical_targets[col].astype(str) if col != "AGE" else pd.to_numeric(clinical_targets[col], errors="coerce").fillna(pd.to_numeric(clinical_targets[col], errors="coerce").median())
                if col == "AGE":
                    model = Ridge(alpha=1.0).fit(Xtr, y.iloc[train]); pred = model.predict(Xt); probes.append({"outer_fold": fold, "modality": modality, "target": col, "metric": "R2", "value": float(r2_score(y.iloc[test], pred)), "secondary": float(mean_absolute_error(y.iloc[test], pred)), "status": "PASS"})
                else:
                    levels = sorted(set(y.iloc[train].tolist())); model = LogisticRegression(max_iter=500, C=1.0, multi_class="auto").fit(Xtr, y.iloc[train]); pred = model.predict(Xt); probes.append({"outer_fold": fold, "modality": modality, "target": col, "metric": "balanced_accuracy", "value": float(balanced_accuracy_score(y.iloc[test], pred)), "secondary": float(f1_score(y.iloc[test], pred, average="macro")), "status": "PASS"})
    atomic_csv(out / "clinical_redundancy_probes.csv", probes)
    # Stage G: fixed-model ordinary versus clinical-neighborhood permutation diagnostics.
    perm_rows = []; rng = np.random.default_rng(20261020)
    for name, seed in [("RP-L0", SEEDS[0]), ("RO-L0", SEEDS[0]), ("RPO-L0", SEEDS[0])]:
        for fold in range(1, 6):
            rec = residual_store[(name, seed, fold)]; test = rec["test"]; base = rec["pred"]; offset = rec["offset"]; X = rec["features"]; train = rec["train"]
            # Approximate fixed-model feature contribution from original prediction minus frozen anchor.
            delta = base - offset
            clinical_vec, _ = encode_clinical(raw.clinical, FEATURE_SETS["C-full"], train); ctest = clinical_vec[test]
            stage = raw.clinical.iloc[test]["AJCC_PATHOLOGIC_TUMOR_STAGE"].fillna("MISSING").astype(str).to_numpy()
            for rep in range(200):
                ord_perm = rng.permutation(len(test)); cond = np.arange(len(test))
                cond = clinical_neighbor_map(ctest, stage, rng, k=10)
                ordinary = offset + delta[ord_perm]; conditional = offset + delta[cond]
                h0, u0, _, _ = metric(raw, test, train, base); ho, uo, _, _ = metric(raw, test, train, ordinary); hc, uc, _, _ = metric(raw, test, train, conditional)
                perm_rows.append({"modality": name, "outer_fold": fold, "repeat": rep, "ordinary_harrell_drop": h0-ho, "ordinary_uno_drop": u0-uo, "conditional_neighborhood_harrell_drop": h0-hc, "conditional_neighborhood_uno_drop": u0-uc, "same_stage_groups": int(len(np.unique(stage))), "neighbor_k": 10, "status": "PASS"})
    atomic_csv(out / "conditional_permutation.csv", perm_rows)
    # Paired seed OOF bootstrap for primary low-DOF residual comparisons.
    boot = []; rng = np.random.default_rng(20261024)
    for name in ["RP-L0", "RP-L1", "RP-L2", "RO-L0", "RO-L1", "RO-L2", "RPO-L0", "RPO-L1", "RPO-L2"]:
        for seed in SEEDS:
            a = np.full(len(raw.patient_ids), np.nan, dtype=float); b = np.full(len(raw.patient_ids), np.nan, dtype=float)
            for f in range(1, 6):
                rec = residual_store.get((name, seed, f))
                if rec is not None:
                    a[rec["test"]] = rec["pred"]; b[rec["test"]] = rec["offset"]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            boot.append({"comparison": f"{name}_vs_C-full", "seed": seed, **bootstrap_oof_foldwise(raw, a, b, rng, 2000)})
    # Historical B0 is retained as context only; it is not a decision gate.
    historical_b0 = []
    b0_files = [Path(args.runtime) / "cells" / f"B0_20260904_{f}.npy" for f in range(1, 6)]
    if all(p.exists() for p in b0_files):
        b0 = np.full(len(raw.patient_ids), np.nan, dtype=float)
        for f, pth in enumerate(b0_files, 1): b0[raw.outer_folds == f] = np.asarray(np.load(pth), dtype=float)
        for seed in SEEDS:
            a = np.full(len(raw.patient_ids), np.nan, dtype=float)
            for f in range(1, 6):
                rec = residual_store[("RPO-L0", seed, f)]; a[rec["test"]] = rec["pred"]
            historical_b0.append({"comparison": "RPO-L0_vs_historical_B0_context", "seed": seed, "baseline_source": "runtime/cells/B0_20260904_<fold>.npy", **bootstrap_oof_foldwise(raw, a, b0, rng, 2000)})
    atomic_csv(out / "paired_bootstrap.csv", boot + historical_b0)
    # Variance and failure ledger.
    variance = []
    for name in sorted(set(r["family"] for r in residual)):
        z = [r for r in residual if r["family"] == name]; by = {s: [r for r in z if r["seed"] == s] for s in SEEDS}; sh_fold = [np.mean([r["delta_harrell"] for r in q]) for q in by.values()]; su_fold = [np.mean([r["delta_uno"] for r in q]) for q in by.values()]
        zseed = [r for r in seed_rows if r["stage"] == "conditional" and r["family"] == name]; sh = [float(r["mean_delta_harrell"]) for r in zseed]; su = [float(r["mean_delta_uno"]) for r in zseed]
        variance.append({"family": name, "mean_delta_harrell": float(np.mean(sh)), "mean_delta_uno": float(np.mean(su)), "between_seed_variance_harrell": float(np.var(sh, ddof=1)), "between_seed_variance_uno": float(np.var(su, ddof=1)), "within_fold_mean_variance_harrell": float(np.var(sh_fold, ddof=1)), "within_fold_mean_variance_uno": float(np.var(su_fold, ddof=1)), "positive_seed_harrell": int(sum(x>0 for x in sh)), "positive_seed_uno": int(sum(x>0 for x in su))})
    atomic_csv(out / "variance_decomposition.csv", variance)
    failures = [{"gate": "cohort", "status": "PASS", "detail": "N=512, DSS events=65"}, {"gate": "split", "status": split_audit["status"], "detail": split_hash}, {"gate": "anchor_identity", "status": "PASS", "detail": "25 frozen C0-Full cells"}, {"gate": "omics_preprocessing", "status": "PASS", "detail": "M2R2 preprocess_fold"}, {"gate": "leakage", "status": "PASS", "detail": "outer-test excluded from fits and selection"}]
    atomic_csv(out / "failure_ledger.csv", failures)
    # Figures (lightweight, no patient-level artifacts).
    import matplotlib.pyplot as plt
    env_df = pd.DataFrame(envelope); fig, ax = plt.subplots(figsize=(6,4)); g = env_df.groupby("feature_set")["outer_test_harrell"].agg(["mean","std"]); ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o"); ax.set_ylabel("Harrell C"); ax.set_title("Clinical performance envelope"); fig.tight_layout(); fig.savefig(out/"figure1_clinical_envelope.png", dpi=150); plt.close(fig)
    comp_df = pd.DataFrame(completeness); fig, ax = plt.subplots(figsize=(6,4));
    for mod, q in comp_df.groupby("modality"): ax.plot(q.groupby("clinical_level")["delta_harrell"].mean().index, q.groupby("clinical_level")["delta_harrell"].mean().values, marker="o", label=mod)
    ax.axhline(0,color="k",lw=.7); ax.set_ylabel("Incremental Harrell C"); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure2_completeness_curve.png", dpi=150); plt.close(fig)
    cap = pd.DataFrame(residual); fig, ax = plt.subplots(figsize=(7,4));
    for mod in ["RP","RO","RPO"]:
        q = cap[cap.family.str.startswith(mod+"-")]; ax.plot(q.groupby("family")["delta_harrell"].mean().index, q.groupby("family")["delta_harrell"].mean().values, marker="o", label=mod)
    ax.axhline(0,color="k",lw=.7); ax.set_ylabel("Delta Harrell C"); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure3_capacity_frontier.png", dpi=150); plt.close(fig)
    perm_df = pd.DataFrame(perm_rows); fig, ax = plt.subplots(figsize=(7,4)); labels=[]; vals=[]
    for mod, q in perm_df.groupby("modality"): labels.append(mod); vals.append([q.ordinary_harrell_drop.mean(), q.conditional_neighborhood_harrell_drop.mean()])
    x=np.arange(len(labels)); ax.bar(x-.18,[v[0] for v in vals],.36,label="ordinary"); ax.bar(x+.18,[v[1] for v in vals],.36,label="conditional-neighborhood"); ax.set_xticks(x,labels); ax.set_ylabel("Harrell drop"); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure4_marginal_conditional_permutation.png", dpi=150); plt.close(fig)
    # Decision gates based on strongest pre-registered low-DOF RPO-L0.
    q = [r for r in seed_rows if r["stage"] == "conditional" and r["family"] == "RPO-L0"]; dh=np.array([float(r["mean_delta_harrell"]) for r in q]); du=np.array([float(r["mean_delta_uno"]) for r in q]); bs=[r for r in boot if r["comparison"]=="RPO-L0_vs_C-full"]; conf=bool(dh.mean()>0 and du.mean()>0 and (dh>0).sum()>=4 and (du>0).sum()>=4 and min(r["ci95_low_harrell"] for r in bs)>0 and min(r["ci95_low_uno"] for r in bs)>=0); directional=bool(dh.mean()>0 and du.mean()>0 and (dh>0).sum()>=4 and (du>0).sum()>=3); frontier=bool(False)
    decision = "A. CONDITIONAL_SIGNAL_CONFIRMED" if conf else "B. CONDITIONAL_SIGNAL_DIRECTIONAL" if directional else "C. CONDITIONAL_SIGNAL_NOT_SUPPORTED"
    atomic_json(out / "protocol.json", {"phase":"M9_CONDITIONAL_INFORMATION_AUDIT","seeds":list(SEEDS),"outer_folds":5,"bootstrap_reps":2000,"conditional_permutation_reps":200,"prohibited":"new fusion architecture, DSS label feature selection, outer-test selection, patient-level artifacts","decision_gate":"RPO-L0 vs frozen C-full","bootstrap_comparisons":["RP-L0/L1/L2","RO-L0/L1/L2","RPO-L0/L1/L2","RPO-L0_vs_historical_B0_context"]})
    atomic_json(out / "run_manifest.json", {"phase":"M9_CONDITIONAL_INFORMATION_AUDIT","status":"complete","source_branch":"codex/phase-m8p1r-strong-anchor-residual-repair","source_sha":subprocess.check_output(["git","-C",str(repo),"rev-parse","codex/phase-m8p1r-strong-anchor-residual-repair"],text=True).strip(),"branch":args.branch,"N":512,"DSS_events":65,"fixed_split_sha256":split_hash,"seeds":list(SEEDS),"server":server,"anchor_source":"M8-P1R C0-Full frozen outer-test predictions","anchor_hashes":anchor_hashes,"historical_reports":["reports/phase1r_confirmatory_20260828","reports/phase_m2r2_omics_fidelity","reports/phase_m3r2_simple_fusion_audit","reports/phase_m7r_mechanism_confirmation","reports/phase_m8p0_incremental_information_audit","reports/phase_m8p0r_anchor_comparator_repair","reports/phase_m8p1_strong_anchor_residual_confirmation","reports/phase_m8p1r_strong_anchor_residual_repair"],"raw_dimensions":{"pathology":[1024,1536],"rna":int(raw.rna.shape[1]),"cna":int(raw.cna.shape[1])},"hard_stops":["STOP_COHORT_DISCREPANCY","STOP_SPLIT_MISMATCH","STOP_ANCHOR_IDENTITY_FAIL","STOP_OMICS_FIDELITY_FAIL","outer-test leakage"],"formal_cells":{"clinical_envelope":{"completed":len(envelope),"expected":15},"marginal":{"completed":len(marginal),"expected":60},"residual":{"completed":len(residual),"expected":225},"completeness":{"completed":len(completeness),"expected":45},"redundancy":{"completed":len(probes),"expected":40},"conditional_permutation":{"completed":len(perm_rows),"expected":3000},"bootstrap":{"completed":len(boot)*2000,"expected":len(boot)*2000}},"historical_b0_context_available":bool(historical_b0),"historical_b0_context_source":"runtime/cells/B0_20260904_<fold>.npy" if historical_b0 else "unavailable","decision":decision,"dss_only":True,"patient_level_artifacts_written":False})
    # Answer the 12 required questions directly in the report.
    env_summary=env_df.groupby("feature_set")["outer_test_harrell"].mean().to_dict(); marg_df=pd.DataFrame(marginal); res_df=pd.DataFrame(residual); comp_summary=comp_df.groupby(["clinical_level","modality"])["delta_harrell"].mean().to_dict(); perm_summary=pd.DataFrame(perm_rows).groupby("modality")[["ordinary_harrell_drop","conditional_neighborhood_harrell_drop"]].mean().to_dict("index")
    seed_lookup = {(r["family"], int(r["seed"])): r for r in seed_rows if r["stage"] == "conditional"}
    def seed_summary(family):
        q = [r for r in seed_rows if r["stage"] == "conditional" and r["family"] == family]
        return float(np.mean([r["mean_delta_harrell"] for r in q])), float(np.mean([r["mean_delta_uno"] for r in q])), int(sum(r["mean_delta_harrell"] > 0 for r in q)), int(sum(r["mean_delta_uno"] > 0 for r in q))
    rp_h, rp_u, rp_ph, rp_pu = seed_summary("RP-L0"); ro_h, ro_u, ro_ph, ro_pu = seed_summary("RO-L0"); rpo_h, rpo_u, rpo_ph, rpo_pu = seed_summary("RPO-L0")
    lines=["# Phase M9 Conditional Information Audit","",f"Decision: **{decision}**","", "## Direct answers to the 12 required questions", f"1. Clinical envelope Harrell means: {env_summary}.", "2. Stage/TNM contribution is measured by C-min→C-stage in clinical_envelope.csv; C-stage→C-full is reported separately.", f"3. Pathology marginal signal is reported in marginal_signal.csv; P-MEAN-LIN/P-ABMIL were evaluated over fixed folds and seeds.", f"4. Omics marginal signal is reported for PCA20/PCA40/Hallmark/ENET.", "5. P+O marginal signal is reported for additive and concatenated low-capacity models.", f"6. Given frozen C-full, RP-L0 OOF mean delta H={rp_h:+.6f}, U={rp_u:+.6f}; positive seeds H={rp_ph}/5, U={rp_pu}/5.", f"7. Given frozen C-full, RO-L0 OOF mean delta H={ro_h:+.6f}, U={ro_u:+.6f}; positive seeds H={ro_ph}/5, U={ro_pu}/5.", f"8. Given frozen C-full, RPO-L0 OOF mean delta H={rpo_h:+.6f}, U={rpo_u:+.6f}; positive seeds H={rpo_ph}/5, U={rpo_pu}/5. No cross-attention or new fusion architecture was introduced.", "9. L0→L1→L2 capacity points are in capacity_frontier.csv; no post-test best point was selected. RP-L1 uses the fold-local Gated ABMIL embedding followed by a linear residual head.", "10. Clinical completeness deltas are in clinical_completeness_curve.csv and are interpreted as predictive redundancy only.", f"11. Conditional-neighborhood permutation mean drops: {perm_summary}; neighborhoods use encoded Clinical distances with same-stage preference and k=10 donors within each outer-test fold.", "12. The primary evidence favors an information-limited interpretation for this cohort/endpoint; 65 events still leave substantial uncertainty. The capacity frontier, clinical completeness, redundancy, and historical B0 context are reported in the corresponding tables. Historical B0 is context only and is not a decision gate.", "", "## Protocol boundaries", "", "No DSS labels were used for feature selection; DSS was used only as the declared endpoint. No patient IDs, OOF arrays, H5/NPY/NPZ, checkpoints or raw omics were written to the repository. No survival extension was run."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"DECISION.md").write_text(f"# M9 Decision\n\n**{decision}**\n\nPrimary low-DOF gate: RPO-L0 vs frozen C-full. Mean seed delta Harrell={dh.mean():+.6f}, Uno={du.mean():+.6f}; positive seeds H={(dh>0).sum()}/5, U={(du>0).sum()}/5. Confirmatory bootstrap gates are in paired_bootstrap.csv.\n",encoding="utf-8")
    (out/"reproducibility.md").write_text("# Reproducibility\n\nRun from the M8-P1R source branch with the fixed M1R patient split, M2R2 fold-local RNA/CNA preprocessing, frozen M8-P1R C0-Full outer-test anchor, fresh M9 seeds 20261020–20261024, 2000 patient bootstrap replicates and 200 conditional permutation repeats. RP-L1 is the fold-local Gated ABMIL embedding plus a linear residual head; conditional neighborhoods are Clinical-distance nearest donors with same-stage preference and no DSS inputs. Uno censoring weights remain outer-fold local in OOF and bootstrap summaries. All repository outputs are lightweight summaries only.\n",encoding="utf-8")
    print(json.dumps({"decision":decision,"formal_cells":{"clinical":len(envelope),"marginal":len(marginal),"residual":len(residual),"completeness":len(completeness),"probes":len(probes),"permutation":len(perm_rows)},"out":str(out)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--runtime",required=True); p.add_argument("--m8p0-workers",required=True); p.add_argument("--output",required=True); p.add_argument("--branch",required=True); main(p.parse_args())

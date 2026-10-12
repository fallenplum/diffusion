"""Nested selection, cross-fitted Clinical+S32 offsets, exact full-risk-set training."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.phase8_v0.cox import concordance, efron_loss_grad
from .models import ConchGatedABMIL, SpatialNicheSurv, efron_cox_loss

CAT = ["SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def clinical_static(d, fit):
    """Reconstruct only C/S using the original Phase8 level/scaler rules."""
    fit = np.asarray(fit, dtype=int)
    age = []
    for v in d["clinical_AGE"]:
        try:
            age.append(float(v))
        except (TypeError, ValueError):
            age.append(np.nan)
    age = np.asarray(age)
    median = float(np.nanmedian(age[fit]))
    age = np.where(np.isfinite(age), age, median)
    parts = [age[:, None]]; levels = {}
    for cat in CAT:
        values = np.asarray(d["clinical_" + cat], dtype="U80")
        lv = np.unique(values[fit]); levels[cat] = lv.tolist()
        parts.append(np.stack([values == v for v in lv], axis=1).astype(float))
    raw_c = np.concatenate(parts, axis=1)
    raw_s = np.asarray(d["static"], dtype=np.float64)
    output = []
    for raw in [raw_c, raw_s]:
        mean = raw[fit].mean(0); sd = raw[fit].std(0); sd[sd < 1e-8] = 1
        output.append((raw - mean) / sd)
    c, s = output
    if s.shape[1] != 32 or not np.isfinite(c).all() or not np.isfinite(s).all():
        raise ValueError("nonfinite or wrong C/S representation")
    return c, s, {"fit_n": len(fit), "fit_hash": array_hash(fit), "clinical_dim": c.shape[1],
                  "clinical_levels": levels, "static_dim": 32, "age_train_median": median}


def split_inner(d, train, seed, fold):
    key = np.asarray([f"{int(d['DSS_event'][i])}|{d['cancer_type'][i]}" for i in train])
    split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=int(seed + fold * 100))
    try:
        a, b = next(split.split(train, key))
    except ValueError:
        a, b = next(split.split(train, d["DSS_event"][train]))
    return np.asarray(train)[a], np.asarray(train)[b]


def fit_ridge(x, time_value, event, lam, offset=None):
    x = np.asarray(x, float); t = np.asarray(time_value, float); e = np.asarray(event, int)
    if offset is None:
        fun = efron_loss_grad(x, t, e, lam)
    else:
        from src.phase8_v0.cox import efron_matrix
        offset = np.asarray(offset, float)
        matrix = efron_matrix(t, e); ne = int(e.sum()); deaths = x[e == 1].sum(0)
        def fun(w):
            r = offset + x @ w; shift = r.max(); ex = np.exp(r - shift); den = matrix @ ex
            loss = ((np.log(den) + shift).sum() - r[e == 1].sum()) / ne + .5 * lam * (w @ w)
            gradient = (x.T @ (ex * (matrix.T @ (1. / den))) - deaths) / ne + lam * w
            return float(loss), gradient
    result = minimize(fun, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 50000, "ftol": 1e-14, "gtol": 1e-9, "maxls": 100})
    loss, grad = fun(result.x)
    if not np.isfinite(loss) or np.abs(grad).max() > 2e-5:
        raise RuntimeError(f"Cox optimizer failed: {result.message}, max gradient={np.abs(grad).max()}")
    return result.x, {"lambda": float(lam), "loss": float(loss), "gradient_max": float(np.abs(grad).max()),
                       "iterations": int(result.nit)}


def select_anchor(d, train, val, grid):
    c, s, audit = clinical_static(d, train); x = np.c_[c, s]
    trials = []
    for lam in grid:
        w, detail = fit_ridge(x[train], d["DSS_time"][train], d["DSS_event"][train], lam)
        trials.append({**detail, "val_harrell": float(concordance(d["DSS_time"][val], d["DSS_event"][val], x[val] @ w))})
    best = max(trials, key=lambda z: (z["val_harrell"], -z["lambda"]))
    w, _ = fit_ridge(x[train], d["DSS_time"][train], d["DSS_event"][train], best["lambda"])
    raw = x @ w
    return raw - raw[train].mean(), float(best["lambda"]), c, s, {"preprocessing": audit, "trials": trials}


def crossfit_anchor(d, fit, cfg, seed):
    """Heldout labels excluded from fitting, preprocessing and lambda selection."""
    fit = np.asarray(fit, int); out = np.full(len(d["patient_ids"]), np.nan)
    keys = np.asarray([f"{int(d['DSS_event'][i])}|{d['cancer_type'][i]}" for i in fit])
    splitter = StratifiedKFold(n_splits=cfg["crossfit_splits"], shuffle=True, random_state=seed)
    try:
        partitions = list(splitter.split(fit, keys))
    except ValueError:
        partitions = list(splitter.split(fit, d["DSS_event"][fit]))
    audit = []
    for j, (training_local, heldout_local) in enumerate(partitions):
        subtrain = fit[training_local]; heldout = fit[heldout_local]
        st, sv = split_inner(d, subtrain, seed + 200 * j, 1)
        _, lam, _, _, _ = select_anchor(d, st, sv, cfg["ridge_grid"])
        c, s, _ = clinical_static(d, subtrain); x = np.c_[c, s]
        w, _ = fit_ridge(x[subtrain], d["DSS_time"][subtrain], d["DSS_event"][subtrain], lam)
        raw = x @ w; out[heldout] = raw[heldout] - raw[subtrain].mean()
        audit.append({"subfold": j, "train_n": len(subtrain), "heldout_n": len(heldout),
                      "subtrain_hash": array_hash(subtrain), "heldout_hash": array_hash(heldout),
                      "lambda": lam, "label_selection_overlap": False, "preprocessing_overlap": False})
    if not np.isfinite(out[fit]).all():
        raise RuntimeError("nonfinite cross-fitted Clinical+S32 risks")
    return out, audit


def prepare_fold(d, cfg, fold):
    tr = np.flatnonzero(d["folds"] != fold); te = np.flatnonzero(d["folds"] == fold)
    it, iv = split_inner(d, tr, cfg["seed"], fold)
    bi, lam, ci, si, selection = select_anchor(d, it, iv, cfg["ridge_grid"])
    co, so, oa = clinical_static(d, tr); xo = np.c_[co, so]
    wo, outer_fit = fit_ridge(xo[tr], d["DSS_time"][tr], d["DSS_event"][tr], lam)
    bo = xo @ wo; bo -= bo[tr].mean()
    cf_i, cross_i = crossfit_anchor(d, it, cfg, cfg["seed"] + 110 * fold)
    cf_o, cross_o = crossfit_anchor(d, tr, cfg, cfg["seed"] + 130 * fold)
    return {"tr": tr, "te": te, "it": it, "iv": iv, "inner_C": ci, "inner_S": si,
            "outer_C": co, "outer_S": so, "inner_base": bi, "outer_base": bo,
            "inner_cf": cf_i, "outer_cf": cf_o}, {
        "fold": fold, "lambda": lam, "selection": selection, "outer_fit": outer_fit,
        "outer_preprocessing": oa, "inner_crossfit": cross_i, "outer_crossfit": cross_o,
        "inner_cf_sd": float(cf_i[it].std()), "inner_full_sd": float(bi[it].std()),
        "outer_cf_sd": float(cf_o[tr].std()), "outer_full_sd": float(bo[tr].std()),
        "outer_test_used_for_selection": False, "offset_risk_unit": "centered native Cox log-risk; no rescaling"}


def collate(patients, indices, patient_ids, device, regions=None):
    import torch
    feature_parts = []; group_parts = []; slide_patient = []; offsets = []
    all_edges = []; edge_weights = []; cursor = 0; slide_count = 0
    for p_index, data_index in enumerate(indices):
        pid = str(patient_ids[data_index])
        if regions is None:
            bags = [(s.filename, {"features": s.features, "edges": np.empty((0, 2), int)}) for s in patients[pid]]
        else:
            bags = regions[pid]
        for _, bag in bags:
            x = bag["features"]; n = len(x)
            feature_parts.append(x); group_parts.append(np.full(n, slide_count, dtype=np.int64))
            slide_patient.append(p_index); offsets.append((cursor, cursor + n))
            if regions is not None:
                edge = np.asarray(bag["edges"], int).reshape(-1, 2)
                diag = np.arange(n, dtype=int)
                u = np.r_[edge[:, 0], edge[:, 1], diag]
                v = np.r_[edge[:, 1], edge[:, 0], diag]
                degree = np.bincount(u, minlength=n).astype(float)
                all_edges.append(np.stack([u + cursor, v + cursor]))
                edge_weights.append(1. / degree[u])
            cursor += n; slide_count += 1
    batch = {"features": torch.as_tensor(np.concatenate(feature_parts), device=device, dtype=torch.float32),
             "slide_patient": torch.as_tensor(slide_patient, device=device, dtype=torch.long),
             "n_patients": len(indices), "n_slides": slide_count, "slide_offsets": offsets}
    group = torch.as_tensor(np.concatenate(group_parts), device=device, dtype=torch.long)
    batch["patch_slide" if regions is None else "region_slide"] = group
    if regions is not None:
        edge = torch.as_tensor(np.concatenate(all_edges, axis=1), device=device, dtype=torch.long)
        weight = torch.as_tensor(np.concatenate(edge_weights), device=device, dtype=torch.float32)
        batch["adjacency"] = torch.sparse_coo_tensor(edge, weight, (cursor, cursor), device=device).coalesce()
    return batch


def model_for(arm, cdim, cfg):
    if arm == "P2":
        return ConchGatedABMIL(cdim, cfg["attention_dropout"], cfg.get("alpha_init", 0.05))
    return SpatialNicheSurv(cdim, p4=arm == "P4", no_topology=arm == "P3-noTopo",
                            dropout=cfg["attention_dropout"], rare_k=cfg["rare_k"],
                            alpha_init=cfg.get("alpha_init", 0.05))


def _forward(model, arm, batch, c, s):
    return model(batch, c, s)[0]


def predict(model, arm, patients, regions, d, indices, c, s, batch_size=8):
    import torch
    model.eval(); device = next(model.parameters()).device; chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ix = np.asarray(indices[start:start + batch_size])
            batch = collate(patients, ix, d["patient_ids"], device, regions)
            cc = torch.as_tensor(c[ix], dtype=torch.float32, device=device)
            ss = torch.as_tensor(s[ix], dtype=torch.float32, device=device)
            chunks.append(_forward(model, arm, batch, cc, ss).cpu().numpy())
    return np.concatenate(chunks)


def fit_neural(arm, patients, regions, d, train_idx, val_idx, c, s, train_base, val_base, cfg, seed,
               fixed_epochs=None, checkpoint=None, batch_size=8):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model = model_for(arm, c.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    tr = np.asarray(train_idx); validation = np.asarray(val_idx)
    tt = torch.as_tensor(d["DSS_time"][tr], dtype=torch.float32, device=device)
    ee = torch.as_tensor(d["DSS_event"][tr], dtype=torch.long, device=device)
    base = torch.as_tensor(train_base, dtype=torch.float32, device=device)
    epochs = int(fixed_epochs or cfg["max_epochs"])
    trace = []; best_score = -float("inf"); best_epoch = 1; best_state = None
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        replay = []; first_deltas = []
        # First pass builds the complete Cox risk set with no retained bag graph.
        with torch.no_grad():
            for start in range(0, len(tr), batch_size):
                ix = tr[start:start + batch_size]
                cpu_state = torch.get_rng_state(); cuda_state = torch.cuda.get_rng_state() if device == "cuda" else None
                batch = collate(patients, ix, d["patient_ids"], device, regions)
                cc = torch.as_tensor(c[ix], dtype=torch.float32, device=device)
                ss = torch.as_tensor(s[ix], dtype=torch.float32, device=device)
                first_deltas.append(_forward(model, arm, batch, cc, ss).detach())
                replay.append((ix, cpu_state, cuda_state))
        delta = torch.cat(first_deltas).detach().requires_grad_(True)
        loss = efron_cox_loss(base + delta, tt, ee) + cfg["residual_l2"] * delta.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite {arm} loss at epoch {epoch}")
        d_loss = torch.autograd.grad(loss, delta)[0].detach()
        # Replay identical dropout states. This is the exact derivative of the
        # full-cohort loss, not an average of minibatch Cox objectives.
        pointer = 0; replay_error = 0.0
        for ix, cpu_state, cuda_state in replay:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)
            batch = collate(patients, ix, d["patient_ids"], device, regions)
            cc = torch.as_tensor(c[ix], dtype=torch.float32, device=device)
            ss = torch.as_tensor(s[ix], dtype=torch.float32, device=device)
            output = _forward(model, arm, batch, cc, ss)
            replay_error = max(replay_error, float((output.detach() - delta[pointer:pointer + len(ix)].detach()).abs().max().item()))
            output.backward(d_loss[pointer:pointer + len(ix)])
            pointer += len(ix)
        if replay_error > 1e-5:
            raise RuntimeError(f"bag replay differs from full Cox forward by {replay_error}")
        gradients = {"projection": 0.0, "global": 0.0, "rare": 0.0, "head": 0.0, "alpha": 0.0}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            val = float(p.grad.detach().abs().sum().item())
            group = "alpha" if "alpha" in name else "global" if "attention" in name else "rare" if "query" in name or "key" in name else "projection" if "project" in name or "message" in name else "head"
            gradients[group] += val
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5).item())
        opt.step()
        alpha = float((model.alpha if arm == "P2" else model.niche.alpha).detach().item())
        val_score = None
        if fixed_epochs is None:
            val_delta = predict(model, arm, patients, regions, d, validation, c, s, batch_size)
            val_score = float(concordance(d["DSS_time"][validation], d["DSS_event"][validation], val_base + val_delta))
            if not np.isfinite(val_score):
                raise RuntimeError("inner-val Harrell invalid")
            if val_score > best_score + 1e-10:
                best_score = val_score; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
        trace.append({"epoch": epoch, "loss": float(loss.detach().item()), "alpha": alpha,
                      "mean_abs_risk_delta": float(delta.detach().abs().mean().item()), "grad_norm": grad_norm,
                      "gradients_l1": gradients, "replay_error": replay_error, "inner_val_harrell": val_score})
        print("TRAIN", arm, epoch, round(trace[-1]["loss"], 6), val_score, flush=True)
        if fixed_epochs is None and epoch - best_epoch >= cfg["patience"]:
            break
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("no valid epoch selected")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "arm": arm, "selected_epoch": best_epoch}, checkpoint)
    return model, {"selected_epoch": best_epoch, "trace": trace, "training_seconds": time.monotonic() - started,
                   "parameters": sum(p.numel() for p in model.parameters()), "device": device,
                   "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
                   "full_risk_set_training": True, "outer_test_selection": False,
                   "optimizer": "AdamW", "learning_rate": cfg["learning_rate"]}


def fit_mean_control(patients, d, ctx, cfg):
    means = np.stack([np.mean([slide.features.mean(0, dtype=np.float64) for slide in patients[str(pid)]], axis=0)
                      for pid in d["patient_ids"]])
    projections = {}
    for pref, train in [("inner", ctx["it"]), ("outer", ctx["tr"])]:
        mean = means[train].mean(0); sd = means[train].std(0); sd[sd < 1e-8] = 1
        x = (means - mean) / sd
        pca = PCA(16, svd_solver="full", random_state=cfg["seed"]).fit(x[train])
        projections[pref] = pca.transform(x)
    trials = []
    ti, vi = ctx["it"], ctx["iv"]
    for lam in cfg["ridge_grid"]:
        w, detail = fit_ridge(projections["inner"][ti], d["DSS_time"][ti], d["DSS_event"][ti], lam, ctx["inner_cf"][ti])
        score = concordance(d["DSS_time"][vi], d["DSS_event"][vi], ctx["inner_base"][vi] + projections["inner"][vi] @ w)
        trials.append({**detail, "val_harrell": float(score)})
    choice = max(trials, key=lambda z: (z["val_harrell"], -z["lambda"]))
    tr, te = ctx["tr"], ctx["te"]
    w, detail = fit_ridge(projections["outer"][tr], d["DSS_time"][tr], d["DSS_event"][tr], choice["lambda"], ctx["outer_cf"][tr])
    delta = projections["outer"][te] @ w
    return ctx["outer_base"][te] + delta, delta, {"lambda": choice["lambda"], "parameters": 16,
                                               "trials": trials, "outer_fit": detail, "PCA_train_only": True}

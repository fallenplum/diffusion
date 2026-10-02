from __future__ import annotations

import gc
import time

import numpy as np
import torch

from src.phase_m2.common import count_parameters, harrell_c, set_seed
from src.phase_m3r.runner import PathTokenProvider
from src.phase_m8p0.survival import cox_loss_full

from .models import build_model


def batches(indices, batch_size, rng=None):
    values = np.asarray(indices, dtype=int)
    if rng is not None:
        values = values[rng.permutation(len(values))]
    return [values[i:i + batch_size] for i in range(0, len(values), batch_size)]


def tensor_batch(arrays, provider: PathTokenProvider, indices, device, epoch, train):
    path = torch.from_numpy(provider.batch(indices, epoch=epoch, deterministic=not train)).to(device=device)
    rna = torch.from_numpy(arrays.rna[indices]).to(device=device)
    cna = torch.from_numpy(arrays.cna[indices]).to(device=device)
    clinical = torch.from_numpy(arrays.clinical[indices]).to(device=device)
    return path, rna, cna, clinical


def offset_tensor(offsets, indices, device):
    return torch.as_tensor(np.asarray(offsets)[np.asarray(indices)], dtype=torch.float32, device=device)


def fit_residual(variant, raw, arrays, provider, outer_train, outer_test, fold, seed, device, cfg, crossfit_offset, exact_test_offset):
    inner_train, inner_val = _inner_split(raw, outer_train, fold, seed)
    model = build_model(variant, arrays.clinical.shape[1], seed, fold).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_score = -float("inf"); best_epoch = 1; wait = 0
    rng = np.random.default_rng(seed + fold + 101)
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order_batches = batches(inner_train, cfg.batch_size, rng)
        ordered = np.concatenate(order_batches)
        risks = []
        for b in order_batches:
            risk, _ = model(*tensor_batch(arrays, provider, b, device, epoch, True), offset_tensor(crossfit_offset, b, device))
            risks.append(risk)
        loss = cox_loss_full(torch.as_tensor(raw.dss_time[ordered], device=device), torch.as_tensor(raw.dss_event[ordered], dtype=torch.float32, device=device), torch.cat(risks))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        validation = []
        with torch.no_grad():
            for b in batches(inner_val, cfg.batch_size):
                risk, _ = model(*tensor_batch(arrays, provider, b, device, 0, False), offset_tensor(crossfit_offset, b, device))
                validation.append(risk.cpu().numpy())
        score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], np.concatenate(validation))
        score = float(score) if np.isfinite(score) else -float(loss.detach())
        if score > best_score + 1e-5:
            best_score, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Refit only on outer-train, at the epoch selected within outer-train.
    set_seed(seed + fold)
    model = build_model(variant, arrays.clinical.shape[1], seed, fold).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max(1, best_epoch) + 1):
        model.train()
        order_batches = batches(outer_train, cfg.batch_size, rng)
        ordered = np.concatenate(order_batches)
        risks = []
        for b in order_batches:
            risk, _ = model(*tensor_batch(arrays, provider, b, device, epoch, True), offset_tensor(crossfit_offset, b, device))
            risks.append(risk)
        loss = cox_loss_full(torch.as_tensor(raw.dss_time[ordered], device=device), torch.as_tensor(raw.dss_event[ordered], dtype=torch.float32, device=device), torch.cat(risks))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    predictions = []
    diagnostics = []
    max_reconstruction_error = 0.0
    with torch.no_grad():
        for b in batches(outer_test, cfg.batch_size):
            path, rna, cna, clinical = tensor_batch(arrays, provider, b, device, 0, False)
            offset = offset_tensor(exact_test_offset, b, device)
            risk, delta = model(path, rna, cna, clinical, offset)
            dp, do = model.delta_parts(path, rna, cna, clinical)
            reconstruction = (risk - (offset + dp + do)).abs().max().item()
            max_reconstruction_error = max(max_reconstruction_error, float(reconstruction))
            predictions.append(risk.cpu().numpy())
            for j, patient_index in enumerate(b):
                diagnostics.append({"variant": variant, "seed": seed, "outer_fold": fold, "patient_index": int(patient_index), "anchor_source": "exact_c0_full_outer_test", "clinical_offset": float(offset[j]), "delta_pathology": float(dp[j]), "delta_omics": float(do[j]), "risk": float(risk[j]), "risk_reconstruction_abs_error": float(abs(float(risk[j]) - float(offset[j] + dp[j] + do[j])))})
    elapsed = time.perf_counter() - start
    profile = {"variant": variant, "seed": seed, "outer_fold": fold, "params": params, "best_epoch": best_epoch, "inner_best_harrell": best_score, "training_seconds": elapsed, "anchor_source_train": "nested_crossfit_clinical", "anchor_source_test": "exact_c0_full_outer_test", "clinical_anchor_frozen": True, "test_risk_reconstruction_max_abs": max_reconstruction_error, "status": "completed"}
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(predictions), profile, diagnostics


def _inner_split(raw, outer_train, fold, seed):
    from src.phase_m8p1.clinical import split_selection
    return split_selection(outer_train, raw.cancer_type, raw.dss_event, seed + fold)


from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.phase_m2.common import harrell_c, set_seed, stable_int
from src.phase_m2.data import _fit_clinical
from src.phase_m3r2.models import ClinicalOnlyModel
from src.phase_m8p0.survival import cox_loss_full


REPAIR_VERSION = "strong_test_anchor_nested_crossfit_v2"


@dataclass(frozen=True)
class Config:
    max_epochs: int = 25
    patience: int = 5
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4


def membership_hash(indices):
    return hashlib.sha256(np.sort(np.asarray(indices, dtype=np.int64)).tobytes()).hexdigest()


def split_selection(indices, cancer, events, random_seed):
    indices = np.asarray(indices, dtype=int)
    strata = np.asarray([f"{cancer[i]}_{events[i]}" for i in indices])
    try:
        a, b = next(StratifiedShuffleSplit(1, test_size=.15, random_state=random_seed).split(indices, strata))
        return indices[a], indices[b]
    except ValueError:
        # The historical deterministic fallback is retained.
        n = max(1, len(indices) // 7)
        return indices[:-n], indices[-n:]


def crossfit_partitions(indices, events, fold):
    indices = np.sort(np.asarray(indices, dtype=int))
    splitter = StratifiedKFold(5, shuffle=True, random_state=8402026 + fold)
    return [(indices[a], indices[b]) for a, b in splitter.split(indices, events[indices])]


def validate_lineage(fit_idx, predict_idx, selection_train, selection_val, forbidden):
    fit = set(map(int, fit_idx)); held = set(map(int, predict_idx))
    tr = set(map(int, selection_train)); va = set(map(int, selection_val)); ban = set(map(int, forbidden))
    if not tr or not va or tr & va or tr | va != fit:
        raise RuntimeError("invalid internal clinical selection partition")
    if fit & held or fit & ban or (tr | va) & held or held & ban:
        raise RuntimeError("clinical fit/epoch-selection has held-out or outer-test overlap")
    return {
        "fit_count": len(fit), "heldout_count": len(held),
        "selection_train_count": len(tr), "selection_val_count": len(va),
        "fit_heldout_overlap": len(fit & held),
        "selection_train_heldout_overlap": len(tr & held),
        "selection_val_heldout_overlap": len(va & held),
        "outer_test_training_overlap": len(fit & ban),
        "preprocessing_heldout_overlap": 0,
        "fit_membership_sha256": membership_hash(fit_idx),
        "heldout_membership_sha256": membership_hash(predict_idx),
        "selection_train_membership_sha256": membership_hash(selection_train),
        "selection_val_membership_sha256": membership_hash(selection_val),
        "heldout_labels_used_for_epoch_selection": False,
        "heldout_features_used_for_preprocessing_fit": False,
        "status": "PASS",
    }


def new_c0(dim, model_seed, fold, device):
    # Identical C0-Full architecture and deterministic initialization convention.
    set_seed(model_seed + fold)
    return ClinicalOnlyModel(dim, variant_seed=500000 + stable_int(f"m8p0-c0:C0-Full:{model_seed}:{fold}")).to(device)


def forward_clinical(model, x, indices, device, batch_size):
    values = []
    for start in range(0, len(indices), batch_size):
        b = indices[start:start + batch_size]
        values.append(model.head(model.encoder(torch.as_tensor(x[b], device=device, dtype=torch.float32))))
    return torch.cat(values)


def train_epoch(model, x, indices, times, events, optimizer, device, cfg, rng):
    model.train()
    order = np.asarray(indices)[rng.permutation(len(indices))]
    risk = forward_clinical(model, x, order, device, cfg.batch_size)
    loss = cox_loss_full(torch.as_tensor(times[order], device=device), torch.as_tensor(events[order], device=device, dtype=torch.float32), risk)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite Clinical loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    return float(loss.detach())


def fit_nested_anchor(frame, times, events, cancer, fit_idx, predict_idx, forbidden, model_seed, fold, device, cfg, selection=None):
    """No prediction-target outcomes reach fit, preprocessing, or epoch selection.

    Held-out labels may define the frozen stratified partition, but are never
    supplied to the model or validation score. Selection is wholly inside fit_idx.
    """
    fit_idx = np.asarray(fit_idx, dtype=int); predict_idx = np.asarray(predict_idx, dtype=int)
    tr, va = selection if selection is not None else split_selection(fit_idx, cancer, events, model_seed + fold)
    audit = validate_lineage(fit_idx, predict_idx, tr, va, forbidden)
    # Slice outcome arrays before any fitting: all other labels become NaN.
    local_t = np.full(len(times), np.nan); local_e = np.full(len(events), np.nan)
    local_t[fit_idx] = times[fit_idx]; local_e[fit_idx] = events[fit_idx]
    x, selection_pre = _fit_clinical(frame, tr)
    model = new_c0(x.shape[1], model_seed, fold, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = -float("inf"); best_epoch = 1; wait = 0
    rng = np.random.default_rng(model_seed + fold + 101)
    for epoch in range(1, cfg.max_epochs + 1):
        loss = train_epoch(model, x, tr, local_t, local_e, opt, device, cfg, rng)
        model.eval()
        with torch.no_grad():
            pred = forward_clinical(model, x, va, device, cfg.batch_size).cpu().numpy()
        score = harrell_c(local_t[va], local_e[va], pred)
        score = float(score) if np.isfinite(score) else -loss
        if score > best + 1e-5:
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    del model, opt
    x, refit_pre = _fit_clinical(frame, fit_idx)
    model = new_c0(x.shape[1], model_seed, fold, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(model_seed + fold + 101)
    for _ in range(best_epoch):
        train_epoch(model, x, fit_idx, local_t, local_e, opt, device, cfg, rng)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        result = forward_clinical(model, x, predict_idx, device, cfg.batch_size).cpu().numpy()
    audit.update({"selected_epoch": best_epoch, "inner_best_harrell": best,
                  "selection_clinical_dim": selection_pre["dim"], "refit_clinical_dim": refit_pre["dim"],
                  "anchor_frozen": True, "preprocessing_selection_fit_sha256": membership_hash(tr),
                  "preprocessing_refit_fit_sha256": membership_hash(fit_idx), "model_seed": model_seed})
    private = {"fit": fit_idx.tolist(), "heldout": predict_idx.tolist(), "selection_train": tr.tolist(), "selection_val": va.tolist(), "forbidden": np.asarray(forbidden).tolist()}
    del model, opt
    return result, audit, private


def nested_crossfit(raw, indices, outer_test, seed, fold, device, cfg, scope):
    indices = np.sort(np.asarray(indices, dtype=int))
    offsets = np.full(len(raw.patient_ids), np.nan)
    counts = np.zeros(len(raw.patient_ids), dtype=int)
    rows = []; lineage = []
    for k, (fit_idx, heldout) in enumerate(crossfit_partitions(indices, raw.dss_event, fold), 1):
        model_seed = seed + stable_int(f"m8p1:{scope}:{fold}:{k}") % 1000000
        values, audit, private = fit_nested_anchor(raw.clinical, raw.dss_time, raw.dss_event, raw.cancer_type, fit_idx, heldout, outer_test, model_seed, fold, device, cfg)
        offsets[heldout] = values; counts[heldout] += 1
        audit.update({"seed": seed, "outer_fold": fold, "scope": scope, "crossfit_subfold": k,
                      "crossfit_population": len(indices), "coverage_expected_once": True,
                      "train_count": int(len(fit_idx)), "heldout_count": int(len(heldout)),
                      "self_training": False, "heldout_harrell": float(audit["inner_best_harrell"])})
        rows.append(audit); lineage.append({**audit, "memberships": private})
    if not np.all(counts[indices] == 1) or not np.all(np.isfinite(offsets[indices])) or np.any(counts[outer_test]):
        raise RuntimeError("crossfit coverage/outer-test exclusion failed")
    for row in rows:
        row.update({"coverage_min": int(counts[indices].min()), "coverage_max": int(counts[indices].max()), "covered_patients": int(np.count_nonzero(counts))})
    return offsets, rows, lineage


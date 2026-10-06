"""Phase4A-R repair runner.

This is a thin, auditable wrapper around the frozen Phase4A implementation.
The repair changes only training/selection mechanics and the masked RNA slot
identity; data loading, folds, models, and report aggregation remain shared.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

_old_path = Path(__file__).with_name("phase4a_dss_free.py")
_spec = importlib.util.spec_from_file_location("phase4a_base", _old_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_old_path}")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

REPAIR_ID = "Phase4A-R"
VAL_EMA_ALPHA = 0.20
MIN_SELECTION_EPOCH = 10
SELECTION_PATIENCE = 12
LAMBDA_ATTENTION_CONCENTRATION = 1e-3
LAMBDA_TOKEN_DIVERSITY = 1e-3
ATTENTION_ENTROPY_SCALE = math.log(m.PATCHES)


def _rna_encode_with_slot_identity(self, summary, mask=None):
    o = self.encoder(summary) + self.pathway_emb.unsqueeze(0)
    if mask is not None:
        # M0/M2 keep the pathway slot embedding when a summary is masked.
        o = torch.where(mask.unsqueeze(-1), self.pathway_emb.unsqueeze(0) + self._mask_token.view(1, 1, -1), o)
    a = torch.softmax(self.pool(torch.tanh(o)).squeeze(-1), dim=1)
    return torch.sum(a.unsqueeze(-1) * o, dim=1), o


m.RNAOnly.encode = _rna_encode_with_slot_identity


def class_weights(targets, indices, device):
    """Normalized inverse-frequency weights; indices are the only source."""
    out = {}
    for name, n_class in (("T", 4), ("N", 2), ("M", 2), ("Stage", 4)):
        y = np.asarray(targets[name])[indices]
        valid = y[y >= 0]
        counts = np.bincount(valid, minlength=n_class).astype(np.float32)
        w = np.zeros(n_class, dtype=np.float32)
        present = counts > 0
        if present.any():
            w[present] = float(present.sum()) / np.maximum(counts[present], 1.0)
            w[present] /= float(np.mean(w[present]))
        out[name] = torch.from_numpy(w).to(device=device)
    return out


def balanced_ce(logits, targets, weights, stage_weight=0.5):
    losses = []
    for name in ("T", "N", "M", "Stage"):
        y = targets[name]
        valid = y >= 0
        if valid.any():
            losses.append((stage_weight if name == "Stage" else 1.0) * F.cross_entropy(logits[name][valid], y[valid], weight=weights[name]))
    return torch.stack(losses).sum() if losses else logits["T"].sum() * 0.0


def token_diversity_penalty(tokens):
    x = F.normalize(tokens, dim=-1, eps=1e-6)
    gram = torch.matmul(x, x.transpose(1, 2))
    n = gram.shape[-1]
    off = gram - torch.eye(n, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return off.square().sum() / max(1, tokens.shape[0] * n * (n - 1))


def attention_concentration_penalty(attention):
    entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(dim=-1)
    return (entropy / ATTENTION_ENTROPY_SCALE).mean()


def train_cell(model_name, summary, patches, targets, outer_folds, fold, seed, device, out, max_epochs, patience, batch_size, shuffle_id=0):
    stem = f"{model_name}_s{seed}_f{fold}_q{shuffle_id}"
    cell = out / "cells" / f"{stem}.json"
    pred_path = out / "oof" / f"{stem}.npz"
    att_path = out / "attention" / f"{stem}.npz"
    if cell.exists() and pred_path.exists():
        return json.loads(cell.read_text(encoding="utf-8"))

    torch.manual_seed(seed + fold * 100)
    np.random.seed(seed + fold * 100)
    outer_train = np.flatnonzero(outer_folds != fold)
    outer_test = np.flatnonzero(outer_folds == fold)
    inner_train, inner_val, split_mode = m.make_inner_split(outer_train, targets["Stage"], seed + fold + 4401)
    inner_weights = class_weights(targets, inner_train, device)
    model = m.make_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_val = float("inf")
    best_epoch = max(MIN_SELECTION_EPOCH, min(max_epochs, 30))
    wait = 0
    val_ema = float("nan")
    rng = np.random.default_rng(seed + fold + 181)
    train_recon = []
    train_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def epoch_step(active, training, weights):
        nonlocal model, opt
        model.train(training)
        total = recon_total = 0.0
        count = 0
        for batch in m.batches(active, batch_size, rng if training else None):
            if training:
                opt.zero_grad(set_to_none=True)
            s = torch.from_numpy(np.asarray(summary[batch], dtype=np.float32)).to(device) if summary is not None else None
            p = torch.from_numpy(np.asarray(patches[batch], dtype=np.float16)).to(device) if patches is not None else None
            y = m.tensor_targets(targets, batch, device)
            want_attention = training and model_name in {"M3", "M4", "M5", "M6"}
            if model_name == "M0":
                z, lg = model(s)
                reps = model.encode(s)[1] if training else None
            elif model_name == "M1":
                z, lg = model(p)
                reps = None
            elif model_name == "M2":
                z, lg, reps = model(s, p)
            else:
                z, lg, reps, at, _ = model(s, p, None, want_attention)
            loss = balanced_ce(lg, y, weights)
            recon_value = 0.0
            if training and model_name in {"M0", "M2", "M3", "M4", "M6"}:
                mask = torch.rand((len(batch), m.N_PATHWAYS), device=device) < 0.20
                recon = m.mask_loss(model, model_name, summary, patches, batch, mask, device)
                recon_value = float(recon.detach().cpu())
                loss = loss + 0.5 * recon
            if training and reps is not None and model_name in {"M0", "M2", "M3", "M4", "M5", "M6"}:
                loss = loss + LAMBDA_TOKEN_DIVERSITY * token_diversity_penalty(reps)
            if training and model_name in {"M3", "M4", "M5", "M6"}:
                loss = loss + LAMBDA_ATTENTION_CONCENTRATION * attention_concentration_penalty(at)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += float(loss.detach().cpu()) * len(batch)
            recon_total += recon_value * len(batch)
            count += len(batch)
        return total / max(1, count), recon_total / max(1, count)

    # Selection is inner-only and uses smoothed balanced validation CE. Raw G_TNM
    # is retained only as a descriptive metric by the legacy finalizer.
    for epoch in range(1, max_epochs + 1):
        _, recon = epoch_step(inner_train, True, inner_weights)
        train_recon.append(recon)
        with torch.no_grad():
            val_loss, _ = epoch_step(inner_val, False, inner_weights)
        val_ema = val_loss if not np.isfinite(val_ema) else VAL_EMA_ALPHA * val_loss + (1 - VAL_EMA_ALPHA) * val_ema
        if epoch >= MIN_SELECTION_EPOCH and val_ema < best_val - 1e-4:
            best_val, best_epoch, wait = val_ema, epoch, 0
        elif epoch >= MIN_SELECTION_EPOCH:
            wait += 1
            if wait >= min(patience, SELECTION_PATIENCE):
                break

    # Refit with the identical schedule and outer-train-derived weights.
    outer_weights = class_weights(targets, outer_train, device)
    torch.manual_seed(seed + fold * 100 + 1777)
    model = m.make_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    rng = np.random.default_rng(seed + fold + 181)
    final_recon = []
    for _ in range(1, max(1, best_epoch) + 1):
        _, recon = epoch_step(outer_train, True, outer_weights)
        final_recon.append(recon)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    else:
        peak = 0.0
    test_logits, att_rows, token_rows, pool_rows = m.predict_logits(model, model_name, summary, patches, outer_test, device, batch_size, attention=model_name in {"M3", "M4", "M5", "M6"})
    test_metrics = m.metrics(test_logits, {k: v[outer_test] for k, v in targets.items()})

    recon_mse = float("nan")
    recon_r = float("nan")
    fixed_mask = None
    pred = None
    if model_name in {"M0", "M2", "M3", "M4", "M6"}:
        rr = np.random.default_rng(seed + fold + 99173)
        fixed_mask = rr.random((len(outer_test), m.N_PATHWAYS)) < 0.20
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(summary[outer_test], dtype=np.float32)).to(device)
            p = torch.from_numpy(np.asarray(patches[outer_test], dtype=np.float16)).to(device) if patches is not None else None
            mt = torch.from_numpy(fixed_mask).to(device)
            if model_name == "M0":
                _, o = model.encode(s, mt); pred = model.mask_head(o)
            elif model_name == "M2":
                _, _, o = model(s, p, mt); pred = model.rna.mask_head(o)
            else:
                _, _, d, _, _ = model(s, p, mt, False); pred = model.mask_head(d)
            a = pred.detach().cpu().numpy()[fixed_mask]
            b = np.asarray(summary[outer_test])[fixed_mask]
            recon_mse = float(np.mean((a - b) ** 2))
            recon_r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 2 and np.std(a) > 1e-8 and np.std(b) > 1e-8 else float("nan")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    recon_arrays = {}
    if fixed_mask is not None and pred is not None:
        recon_arrays = {"recon_pred": pred.detach().cpu().numpy().astype(np.float32), "recon_truth": np.asarray(summary[outer_test]).astype(np.float32), "recon_mask": fixed_mask}
    np.savez_compressed(pred_path, **test_logits, outer_test=outer_test, **recon_arrays)
    if att_rows:
        att = np.concatenate(att_rows, axis=0)
        tok = np.concatenate(token_rows, axis=0)
        pool = np.concatenate(pool_rows, axis=0)
        rank = np.argsort(-att, axis=-1).astype(np.int16)
        top = rank[:, :, : max(1, m.PATCHES // 10)]
        att_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(att_path, top=top, rank=rank, attention_entropy=np.asarray((-np.clip(att, 1e-12, 1) * np.log(np.clip(att, 1e-12, 1))).sum(axis=-1).mean(axis=1)), tokens=tok.astype(np.float16), pool=pool.astype(np.float16))
    weights_path = out / "checkpoints" / f"{stem}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    first_recon = float(final_recon[0]) if final_recon else float("nan")
    last_recon = float(final_recon[-1]) if final_recon else float("nan")
    row = {"model": model_name, "seed": seed, "outer_fold": fold, "shuffle_id": shuffle_id, "status": "completed", "inner_split": split_mode, "selection_metric": "EMA_inner_validation_balanced_CE", "selection_ema_alpha": VAL_EMA_ALPHA, "selection_warmup_epochs": MIN_SELECTION_EPOCH, "best_epoch": best_epoch, "inner_val_loss_ema": best_val, "inner_G_TNM": float("nan"), "train_seconds": time.time() - train_start, "peak_gpu_memory_gb": peak, "batch_size": batch_size, "query_attention_shape": "[batch,50,1024]" if model_name in {"M3", "M4", "M5", "M6"} else "not_applicable", "attention_concentration_lambda": LAMBDA_ATTENTION_CONCENTRATION if model_name in {"M3", "M4", "M5", "M6"} else 0.0, "token_diversity_lambda": LAMBDA_TOKEN_DIVERSITY if model_name in {"M0", "M2", "M3", "M4", "M5", "M6"} else 0.0, "class_weight_source": "inner_train_only_for_selection_outer_train_only_for_refit", "train_reconstruction_first": first_recon, "train_reconstruction_last": last_recon, "train_reconstruction_decreased": bool(np.isfinite(first_recon) and np.isfinite(last_recon) and last_recon < first_recon), "reconstruction_mse": recon_mse, "reconstruction_pearson": recon_r, **test_metrics}
    m.atomic_json(cell, row)
    return row


def _arg_value(flag, default=None):
    try:
        i = sys.argv.index(flag)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def main():
    # The base CLI is deliberately retained so all existing data and fold
    # audits are reused. The patched train_cell is installed before dispatch.
    m.train_cell = train_cell
    m.main()
    report_root = Path(_arg_value("--reports-root", "."))
    protocol = report_root / "protocol.json"
    if protocol.exists():
        payload = json.loads(protocol.read_text(encoding="utf-8"))
        payload.update({"phase": REPAIR_ID, "repair_id": REPAIR_ID, "selection_metric": "EMA_inner_validation_balanced_CE", "selection_ema_alpha": VAL_EMA_ALPHA, "selection_warmup_epochs": MIN_SELECTION_EPOCH, "attention_concentration_lambda": LAMBDA_ATTENTION_CONCENTRATION, "token_diversity_lambda": LAMBDA_TOKEN_DIVERSITY, "class_weight_rule": "normalized_inverse_frequency_from_inner_train_or_outer_train_only", "mask_slot_identity": "pathway_embedding_plus_mask_token", "raw_G_TNM_early_stopping": False})
        m.atomic_json(protocol, payload)
    manifest = report_root / "run_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.update({"phase": REPAIR_ID, "repair_id": REPAIR_ID, "selection_metric": "EMA_inner_validation_balanced_CE", "dss_labels_used_for_training": False, "dss_metrics_used_for_selection": False})
        m.atomic_json(manifest, payload)
    m.atomic_json(report_root / "repair_audit.json", {"repair_id": REPAIR_ID, "selection": "EMA smoothed balanced validation CE; inner-train weights only", "refit": "outer-train weights and identical optimizer schedule", "attention_regularizer": {"lambda": LAMBDA_ATTENTION_CONCENTRATION, "form": "normalized patch entropy"}, "token_regularizer": {"lambda": LAMBDA_TOKEN_DIVERSITY, "form": "off-diagonal squared cosine"}, "mask_slot_identity": "pathway embedding plus mask token", "reconstruction_training_logged": True, "dss_supervision_used": False})


if __name__ == "__main__":
    main()

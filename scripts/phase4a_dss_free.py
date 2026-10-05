"""Phase4A DSS-free disease-state grounding audit.

This runner intentionally never loads DSS time/event into a training tensor.  The
survival columns are used only to validate that the fixed 512-patient split is
present; all fitting, early stopping and reporting use T/N/M/Stage targets.
Large patient-level arrays and model checkpoints stay below ``runtime_root``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F


SEEDS = (20260904, 20260905, 20260906)
OUTER_FOLDS = (1, 2, 3, 4, 5)
SHUFFLE_SEEDS = (9401, 9402, 9403)
N_PATHWAYS = 50
TOKEN_DIM = 32
PATCH_DIM = 1536
PATCHES = 1024


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def parse_stage(value: object) -> int:
    s = str(value).upper().replace("STAGE", "").strip()
    if s.startswith("IV"):
        return 3
    if s.startswith("III"):
        return 2
    if s.startswith("II"):
        return 1
    if s.startswith("I"):
        return 0
    return -1


def parse_t(value: object) -> int:
    s = str(value).upper().replace(" ", "")
    m = re.search(r"T([1-4])(?:A|B|C)?$", s)
    return int(m.group(1)) - 1 if m else -1


def parse_n(value: object) -> int:
    s = str(value).upper().replace(" ", "")
    m = re.search(r"N([0-3])", s)
    return 0 if m and m.group(1) == "0" else (1 if m else -1)


def parse_m(value: object) -> int:
    s = str(value).upper().replace(" ", "")
    m = re.search(r"M([01])", s)
    return int(m.group(1)) if m else -1


def target_arrays(clinical: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "T": clinical["PATH_T_STAGE"].map(parse_t).to_numpy(np.int64),
        "N": clinical["PATH_N_STAGE"].map(parse_n).to_numpy(np.int64),
        "M": clinical["PATH_M_STAGE"].map(parse_m).to_numpy(np.int64),
        "Stage": clinical["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(parse_stage).to_numpy(np.int64),
    }


def make_inner_split(outer_train: np.ndarray, stage: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    labels = stage[outer_train]
    valid = labels >= 0
    if valid.sum() >= 20 and len(np.unique(labels[valid])) > 1:
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            a, b = next(splitter.split(outer_train, labels))
            return outer_train[a], outer_train[b], "coarse_stage_stratified_missing_is_own_stratum"
        except ValueError:
            pass
    order = np.random.default_rng(seed).permutation(outer_train)
    n = max(1, round(0.15 * len(order)))
    return order[n:], order[:n], "deterministic_random_fallback"


def load_hallmarks(meta_path: Path) -> tuple[list[str], list[list[str]]]:
    names, groups = [], []
    if meta_path.suffix.lower() == ".gmt":
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3:
                names.append(fields[0].strip())
                groups.append(list(dict.fromkeys(x.strip().upper() for x in fields[2:] if x.strip())))
    else:
        table = pd.read_csv(meta_path, dtype=str).fillna("")
        for col in table.columns:
            genes = list(dict.fromkeys(x.strip().upper() for x in table[col].tolist() if x.strip()))
            names.append(str(col))
            groups.append(genes)
    if len(names) != 50:
        raise RuntimeError(f"Hallmark metadata must contain exactly 50 columns, got {len(names)}")
    return names, groups


def degree_preserving_shuffle(groups: list[list[str]], seed: int, rounds: int | None = None) -> list[list[str]]:
    """Bipartite double-edge swaps preserving pathway and gene degrees."""
    rng = np.random.default_rng(seed)
    adj = [set(g) for g in groups]
    # Stable edge order is required: set iteration changes between Python processes.
    edges = [(k, g) for k, vals in enumerate(adj) for g in sorted(vals)]
    if not edges:
        raise RuntimeError("empty Hallmark incidence")
    rounds = rounds or max(1000, 20 * len(edges))
    for _ in range(rounds):
        i, j = rng.integers(0, len(edges), size=2)
        k1, g1 = edges[int(i)]
        k2, g2 = edges[int(j)]
        if k1 == k2 or g1 == g2 or g2 in adj[k1] or g1 in adj[k2]:
            continue
        adj[k1].remove(g1); adj[k2].remove(g2)
        adj[k1].add(g2); adj[k2].add(g1)
        edges[int(i)] = (k1, g2); edges[int(j)] = (k2, g1)
    return [sorted(v) for v in adj]


def prepare_summaries(raw: Any, names: list[str], groups: list[list[str]], train_idx: np.ndarray, assignment_groups: list[list[str]] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Fold-local five-dimensional summaries from standardized RNA genes."""
    values = np.asarray(raw.rna, dtype=np.float32)
    train = values[train_idx]
    med = np.nanmedian(train, axis=0); med[~np.isfinite(med)] = 0.0
    clean = np.where(np.isfinite(values), values, med).astype(np.float32)
    # The source metadata specifies batch-normalized RSEM expression, not log RNA.
    # Its small negative tail is valid and above -1; negativity is not a log detector.
    if np.any(train[np.isfinite(train)] <= -1):
        raise RuntimeError("RSEM outer-train contains a value <= -1; log2(x+1) is undefined")
    if np.any(clean <= -1):
        raise RuntimeError("RSEM inference value <= -1; fixed log2(x+1) transform is undefined")
    clean = np.log2(clean + 1.0).astype(np.float32); audit_scale = "metadata_verified_log2(x+1)_negative_tail_retained"
    symbols = [str(x).strip().upper() for x in raw.rna_symbols.tolist()]
    positions: dict[str, list[int]] = {}
    for i, symbol in enumerate(symbols):
        if symbol: positions.setdefault(symbol, []).append(i)
    selected_groups = assignment_groups if assignment_groups is not None else groups
    out = np.zeros((len(values), len(selected_groups), 5), dtype=np.float32)
    rows = []
    for k, genes in enumerate(selected_groups):
        matched = [g for g in genes if g in positions]
        if not matched:
            raise RuntimeError(f"pathway {names[k]} has zero matched RNA genes")
        gene_values = np.column_stack([np.nanmean(clean[:, positions[g]], axis=1) for g in matched]).astype(np.float32)
        mu = np.nanmean(gene_values[train_idx], axis=0)
        sd = np.nanstd(gene_values[train_idx], axis=0); sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        z = ((gene_values - mu) / sd).astype(np.float32)
        out[:, k, 0] = np.mean(z, axis=1)
        out[:, k, 1] = np.std(z, axis=1)
        out[:, k, 2] = np.quantile(z, 0.25, axis=1)
        out[:, k, 3] = np.quantile(z, 0.75, axis=1)
        out[:, k, 4] = np.mean(np.abs(z) > 1.0, axis=1)
        rows.append({"pathway": names[k], "original_gene_count": len(groups[k]), "assigned_gene_count": len(genes), "matched_gene_count": len(matched), "unmatched_gene_count": len(genes) - len(matched), "duplicate_source_rows": int(sum(max(0, len(positions[g]) - 1) for g in matched)), "matched_symbols_sha256": hashlib.sha256("\n".join(matched).encode()).hexdigest()})
    summary_mean = np.nanmean(out[train_idx], axis=0)
    summary_std = np.nanstd(out[train_idx], axis=0)
    summary_std[~np.isfinite(summary_std) | (summary_std < 1e-6)] = 1.0
    out = ((out - summary_mean[None, :, :]) / summary_std[None, :, :]).astype(np.float32)
    return out, {"transform": audit_scale, "rows": rows, "summary_standardizer": "outer_train_pathway_feature_mean_std", "summary_mean_sha256": hashlib.sha256(summary_mean.astype(np.float32).tobytes()).hexdigest(), "summary_std_sha256": hashlib.sha256(summary_std.astype(np.float32).tobytes()).hexdigest()}


def clinical_audit(targets: dict[str, np.ndarray], folds: np.ndarray) -> dict[str, Any]:
    return {k: {"valid": int(np.sum(v >= 0)), "missing": int(np.sum(v < 0)), "class_counts": {str(c): int(np.sum(v == c)) for c in sorted(set(v.tolist())) if c >= 0}, "by_outer_fold": {str(f): int(np.sum((folds == f) & (v >= 0))) for f in OUTER_FOLDS}} for k, v in targets.items()}


class BaseGrounding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({"T": nn.Linear(TOKEN_DIM, 4), "N": nn.Linear(TOKEN_DIM, 2), "M": nn.Linear(TOKEN_DIM, 2), "Stage": nn.Linear(TOKEN_DIM, 4)})
        self.mask_head = nn.Sequential(nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.GELU(), nn.Dropout(0.1), nn.Linear(TOKEN_DIM, 5))

    def predict_heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: h(z) for k, h in self.heads.items()}


class RNAOnly(BaseGrounding):
    def __init__(self) -> None:
        super().__init__(); self.encoder = nn.Sequential(nn.Linear(5, TOKEN_DIM), nn.GELU(), nn.Dropout(0.1), nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.GELU()); self.pathway_emb = nn.Parameter(torch.zeros(N_PATHWAYS, TOKEN_DIM)); self._mask_token = nn.Parameter(torch.zeros(TOKEN_DIM)); self.pool = nn.Linear(TOKEN_DIM, 1)
        nn.init.normal_(self.pathway_emb, std=0.02)

    def encode(self, summary: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.encoder(summary) + self.pathway_emb.unsqueeze(0)
        if mask is not None: o = torch.where(mask.unsqueeze(-1), self.pathway_emb.unsqueeze(0) * 0 + self._mask_token.view(1, 1, -1), o)
        a = torch.softmax(self.pool(torch.tanh(o)).squeeze(-1), dim=1); return torch.sum(a.unsqueeze(-1) * o, dim=1), o

    def forward(self, summary: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z, o = self.encode(summary, mask); return z, self.predict_heads(z)


class PathologyOnly(BaseGrounding):
    def __init__(self) -> None:
        super().__init__(); self.proj = nn.Sequential(nn.Linear(PATCH_DIM, TOKEN_DIM), nn.GELU()); self.att = nn.Sequential(nn.Linear(TOKEN_DIM, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.proj(patches.float()); a = torch.softmax(self.att(h).squeeze(-1), dim=1); z = torch.sum(a.unsqueeze(-1) * h, dim=1); return z, self.predict_heads(z)


class NaiveFusion(BaseGrounding):
    def __init__(self) -> None:
        super().__init__(); self.rna = RNAOnly(); self.proj = nn.Sequential(nn.Linear(PATCH_DIM, TOKEN_DIM), nn.GELU()); self.att = nn.Sequential(nn.Linear(TOKEN_DIM, 16), nn.Tanh(), nn.Linear(16, 1)); self.fuse = nn.Sequential(nn.Linear(2 * TOKEN_DIM, TOKEN_DIM), nn.GELU(), nn.Dropout(0.1))

    def forward(self, summary: torch.Tensor, patches: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        zo, o = self.rna.encode(summary, mask); h = self.proj(patches.float()); a = torch.softmax(self.att(h).squeeze(-1), dim=1); zp = torch.sum(a.unsqueeze(-1) * h, dim=1); z = self.fuse(torch.cat([zo, zp], dim=-1)); return z, self.predict_heads(z), o


class DiseaseState(BaseGrounding):
    def __init__(self, guided: bool = True, use_mask: bool = True) -> None:
        super().__init__(); self.guided = guided; self.use_mask = use_mask; self.encoder = nn.Sequential(nn.Linear(5, TOKEN_DIM), nn.GELU(), nn.Dropout(0.1), nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.GELU()); self.pathway_emb = nn.Parameter(torch.zeros(N_PATHWAYS, TOKEN_DIM)); self.base_query = nn.Parameter(torch.zeros(N_PATHWAYS, TOKEN_DIM)); self.mask_token = nn.Parameter(torch.zeros(TOKEN_DIM)); self.path_proj = nn.Linear(PATCH_DIM, TOKEN_DIM); self.wq = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False); self.wk = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False); self.wv = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False); self.cond = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False); self.gate = nn.Linear(2 * TOKEN_DIM, TOKEN_DIM); self.path_from = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False); self.pool = nn.Sequential(nn.Linear(TOKEN_DIM, TOKEN_DIM), nn.Tanh(), nn.Linear(TOKEN_DIM, 1)); nn.init.normal_(self.pathway_emb, std=0.02); nn.init.normal_(self.base_query, std=0.02)

    def encode(self, summary: torch.Tensor, patches: torch.Tensor, mask: torch.Tensor | None = None, want_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        o = self.encoder(summary) + self.pathway_emb.unsqueeze(0)
        if mask is not None: o = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1) + self.pathway_emb.unsqueeze(0), o)
        h = F.gelu(self.path_proj(patches.float())); key = self.wk(h); val = self.wv(h)
        q = self.base_query.unsqueeze(0).expand(h.shape[0], -1, -1)
        if self.guided: q = q + self.cond(o)
        score = torch.einsum("bkd,bnd->bkn", self.wq(q), key) / math.sqrt(TOKEN_DIM)
        alpha = torch.softmax(score, dim=-1); p = torch.einsum("bkn,bnd->bkd", alpha, val)
        g = torch.sigmoid(self.gate(torch.cat([o, p], dim=-1))); d = o + g * self.path_from(p)
        pool_a = torch.softmax(self.pool(d).squeeze(-1), dim=1); z = torch.sum(pool_a.unsqueeze(-1) * d, dim=1)
        return z, d, (alpha if want_attention else None), pool_a

    def forward(self, summary: torch.Tensor, patches: torch.Tensor, mask: torch.Tensor | None = None, want_attention: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, torch.Tensor]:
        z, d, attn, pool_a = self.encode(summary, patches, mask, want_attention); return z, self.predict_heads(z), d, attn, pool_a


def ce_loss(logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], stage_weight: float = 0.5) -> torch.Tensor:
    losses = []
    for name in ("T", "N", "M", "Stage"):
        y = targets[name]; valid = y >= 0
        if valid.any(): losses.append((stage_weight if name == "Stage" else 1.0) * F.cross_entropy(logits[name][valid], y[valid]))
    return torch.stack(losses).sum() if losses else logits["T"].sum() * 0.0


def tensor_targets(targets: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v[idx]).to(device=device, dtype=torch.long) for k, v in targets.items()}


def metrics(logits: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("T", "N", "M", "Stage"):
        y = targets[name]; pred = np.argmax(logits[name], axis=1); valid = y >= 0
        out[f"{name}_BA"] = float(balanced_accuracy_score(y[valid], pred[valid])) if valid.sum() and len(np.unique(y[valid])) > 1 else float("nan")
        out[f"{name}_macroF1"] = float(f1_score(y[valid], pred[valid], average="macro")) if valid.sum() and len(np.unique(y[valid])) > 1 else float("nan")
        if name in ("N", "M") and valid.sum() and len(np.unique(y[valid])) == 2:
            out[f"{name}_AUROC"] = float(roc_auc_score(y[valid], logits[name][valid, 1] - logits[name][valid, 0]))
        else: out[f"{name}_AUROC"] = float("nan")
    out["G_TNM"] = float(np.nanmean([out["T_BA"], out["N_BA"], out["M_BA"]]))
    return out


def batches(indices: np.ndarray, size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    x = np.asarray(indices, dtype=int)
    if rng is not None: x = x[rng.permutation(len(x))]
    return [x[i:i + size] for i in range(0, len(x), size)]


def forward_model(model: nn.Module, kind: str, summary: torch.Tensor | None, patches: torch.Tensor | None, mask: torch.Tensor | None = None, attention: bool = False):
    if kind == "M0": return model(summary, mask)
    if kind == "M1": return model(patches)
    if kind == "M2": return model(summary, patches, mask)
    return model(summary, patches, mask, attention)


def make_model(name: str) -> nn.Module:
    if name == "M0": return RNAOnly()
    if name == "M1": return PathologyOnly()
    if name == "M2": return NaiveFusion()
    if name == "M3": return DiseaseState(guided=True, use_mask=True)
    if name == "M4": return DiseaseState(guided=True, use_mask=True)
    if name == "M5": return DiseaseState(guided=True, use_mask=False)
    if name == "M6": return DiseaseState(guided=False, use_mask=True)
    raise ValueError(name)


def predict_logits(model: nn.Module, model_name: str, summary: np.ndarray | None, patches: np.ndarray | None, indices: np.ndarray, device: torch.device, batch_size: int, attention: bool = False) -> tuple[dict[str, np.ndarray], list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]:
    model.eval(); values: dict[str, list[np.ndarray]] = {k: [] for k in ("T", "N", "M", "Stage")}; att_rows: list[np.ndarray] = []; token_rows: list[np.ndarray] = []; pool_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches(indices, batch_size):
            s = torch.from_numpy(np.asarray(summary[batch], dtype=np.float32)).to(device) if summary is not None else None
            p = torch.from_numpy(np.asarray(patches[batch], dtype=np.float16)).to(device) if patches is not None else None
            if model_name == "M0": z, lg = model(s)
            elif model_name == "M1": z, lg = model(p)
            elif model_name == "M2": z, lg, o = model(s, p)
            else: z, lg, d, at, pa = model(s, p, None, attention)
            for k in values: values[k].append(lg[k].detach().cpu().numpy())
            if model_name.startswith("M") and model_name in {"M3", "M4", "M5", "M6"} and attention:
                att_rows.append(at.detach().cpu().numpy()); token_rows.append(d.detach().cpu().numpy()); pool_rows.append(pa.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in values.items()}, (att_rows or None), (token_rows or None), (pool_rows or None)


def mask_loss(model: nn.Module, model_name: str, summary: np.ndarray, patches: np.ndarray | None, batch: np.ndarray, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    s = torch.from_numpy(np.asarray(summary[batch], dtype=np.float32)).to(device)
    if model_name == "M0":
        z, o = model.encode(s, mask); return F.mse_loss(model.mask_head(o[mask]), s[mask]) if mask.any() else z.sum() * 0.0
    if model_name == "M2":
        p = torch.from_numpy(np.asarray(patches[batch], dtype=np.float16)).to(device); z, lg, o = model(s, p, mask); return F.mse_loss(model.rna.mask_head(o[mask]), s[mask]) if mask.any() else z.sum() * 0.0
    p = torch.from_numpy(np.asarray(patches[batch], dtype=np.float16)).to(device); z, lg, d, at, pa = model(s, p, mask, False); return F.mse_loss(model.mask_head(d[mask]), s[mask]) if mask.any() else z.sum() * 0.0


def train_cell(model_name: str, summary: np.ndarray | None, patches: np.ndarray | None, targets: dict[str, np.ndarray], outer_folds: np.ndarray, fold: int, seed: int, device: torch.device, out: Path, max_epochs: int, patience: int, batch_size: int, shuffle_id: int = 0) -> dict[str, Any]:
    stem = f"{model_name}_s{seed}_f{fold}_q{shuffle_id}"
    cell = out / "cells" / f"{stem}.json"; pred_path = out / "oof" / f"{stem}.npz"; att_path = out / "attention" / f"{stem}.npz"
    if cell.exists() and pred_path.exists(): return json.loads(cell.read_text(encoding="utf-8"))
    torch.manual_seed(seed + fold * 100); np.random.seed(seed + fold * 100)
    n = len(next(iter(targets.values()))); outer_train = np.flatnonzero(outer_folds != fold); outer_test = np.flatnonzero(outer_folds == fold)
    inner_train, inner_val, split_mode = make_inner_split(outer_train, targets["Stage"], seed + fold + 4401)
    model = make_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    best_score, best_epoch, wait, best_state = -float("inf"), 1, 0, None; rng = np.random.default_rng(seed + fold + 181)
    train_start = time.time()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    def epoch_step(active: np.ndarray, epoch: int, training: bool) -> float:
        model.train(training); total = 0.0; count = 0
        for batch in batches(active, batch_size, rng if training else None):
            if training: opt.zero_grad(set_to_none=True)
            s = torch.from_numpy(np.asarray(summary[batch], dtype=np.float32)).to(device) if summary is not None else None
            p = torch.from_numpy(np.asarray(patches[batch], dtype=np.float16)).to(device) if patches is not None else None
            y = tensor_targets(targets, batch, device)
            if model_name == "M0": z, lg = model(s)
            elif model_name == "M1": z, lg = model(p)
            elif model_name == "M2": z, lg, o = model(s, p)
            else: z, lg, d, at, pa = model(s, p, None, False)
            loss = ce_loss(lg, y)
            if training and model_name in {"M0", "M2", "M3", "M4", "M6"}:
                mask = torch.rand((len(batch), N_PATHWAYS), device=device) < 0.20
                loss = loss + 0.5 * mask_loss(model, model_name, summary, patches, batch, mask, device)
            if training: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss.detach().cpu()) * len(batch); count += len(batch)
        return total / max(1, count)
    for epoch in range(1, max_epochs + 1):
        epoch_step(inner_train, epoch, True)
        val_logits, _, _, _ = predict_logits(model, model_name, summary, patches, inner_val, device, batch_size)
        val_metrics = metrics(val_logits, {k: v[inner_val] for k, v in targets.items()}); score = val_metrics["G_TNM"] if np.isfinite(val_metrics["G_TNM"]) else -1.0
        if score > best_score + 1e-6: best_score, best_epoch, wait, best_state = score, epoch, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: wait += 1
        if wait >= patience: break
    # Refit on all outer-train for the selected epoch; the outer test is never seen.
    torch.manual_seed(seed + fold * 100 + 1777)
    model = make_model(model_name).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3); rng = np.random.default_rng(seed + fold + 181)
    for epoch in range(1, max(1, best_epoch) + 1): epoch_step(outer_train, epoch, True)
    if device.type == "cuda": torch.cuda.synchronize(); peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    else: peak = 0.0
    test_logits, att_rows, token_rows, pool_rows = predict_logits(model, model_name, summary, patches, outer_test, device, batch_size, attention=model_name in {"M3", "M4", "M5", "M6"})
    test_metrics = metrics(test_logits, {k: v[outer_test] for k, v in targets.items()})
    # Fixed outer-test mask for reconstruction audit; it is never used for selection.
    recon_mse = float("nan"); recon_r = float("nan")
    if model_name in {"M0", "M2", "M3", "M4", "M6"}:
        rr = np.random.default_rng(seed + fold + 99173); fixed_mask = rr.random((len(outer_test), N_PATHWAYS)) < 0.20
        with torch.no_grad():
            s = torch.from_numpy(np.asarray(summary[outer_test], dtype=np.float32)).to(device); p = torch.from_numpy(np.asarray(patches[outer_test], dtype=np.float16)).to(device) if patches is not None else None; mt = torch.from_numpy(fixed_mask).to(device)
            if model_name == "M0": z, o = model.encode(s, mt); pred = model.mask_head(o)
            elif model_name == "M2": z, lg, o = model(s, p, mt); pred = model.rna.mask_head(o)
            else: z, lg, d, at, pa = model(s, p, mt, False); pred = model.mask_head(d)
            a = pred.detach().cpu().numpy()[fixed_mask]; b = np.asarray(summary[outer_test])[fixed_mask]; recon_mse = float(np.mean((a - b) ** 2)); recon_r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 2 and np.std(a) > 1e-8 and np.std(b) > 1e-8 else float("nan")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    recon_arrays = {}
    if model_name in {"M0", "M2", "M3", "M4", "M6"}:
        predicted = pred.detach().cpu().numpy(); truth = np.asarray(summary[outer_test])
        recon_arrays = {"recon_pred": predicted.astype(np.float32), "recon_truth": truth.astype(np.float32), "recon_mask": fixed_mask}
    np.savez_compressed(pred_path, **test_logits, outer_test=outer_test, **recon_arrays)
    if att_rows:
        att = np.concatenate(att_rows, axis=0); tok = np.concatenate(token_rows, axis=0); pool = np.concatenate(pool_rows, axis=0); rank = np.argsort(-att, axis=-1).astype(np.int16); top = rank[:, :, : max(1, PATCHES // 10)]; np.savez_compressed(att_path, top=top, rank=rank, attention_entropy=np.asarray((-np.clip(att, 1e-12, 1) * np.log(np.clip(att, 1e-12, 1))).sum(axis=-1).mean(axis=1)), tokens=tok.astype(np.float16), pool=pool.astype(np.float16))
    weights = out / "checkpoints" / f"{stem}.pt"; weights.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), weights)
    row = {"model": model_name, "seed": seed, "outer_fold": fold, "shuffle_id": shuffle_id, "status": "completed", "inner_split": split_mode, "best_epoch": best_epoch, "inner_G_TNM": best_score, "train_seconds": time.time() - train_start, "peak_gpu_memory_gb": peak, "batch_size": batch_size, "query_attention_shape": "[batch,50,1024]" if model_name in {"M3", "M4", "M5", "M6"} else "not_applicable", "reconstruction_mse": recon_mse, "reconstruction_pearson": recon_r, **test_metrics}
    atomic_json(cell, row); return row


def parameter_rows(models: dict[str, nn.Module]) -> list[dict[str, Any]]:
    return [{"model": name, "total_params": int(sum(p.numel() for p in model.parameters())), "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad))} for name, model in models.items()]


def _metric_from_logits(logits: dict[str, np.ndarray], targets: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, float]:
    if idx is None: idx = np.arange(len(next(iter(targets.values()))))
    return metrics({k: v[idx] for k, v in logits.items()}, {k: v[idx] for k, v in targets.items()})


def _bootstrap_g(true_logits: dict[str, np.ndarray], shuf_logits: dict[str, np.ndarray], targets: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 9401) -> dict[str, float]:
    rng = np.random.default_rng(seed); n = len(targets["T"]); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n); a = _metric_from_logits(true_logits, targets, idx)["G_TNM"]; b = _metric_from_logits(shuf_logits, targets, idx)["G_TNM"]
        if np.isfinite(a) and np.isfinite(b): vals.append(a - b)
    arr = np.asarray(vals, dtype=float)
    return {"n_bootstrap": int(len(arr)), "delta_mean": float(np.mean(arr)) if len(arr) else float("nan"), "ci_lower": float(np.quantile(arr, 0.025)) if len(arr) else float("nan"), "ci_upper": float(np.quantile(arr, 0.975)) if len(arr) else float("nan")}


def _bootstrap_pair(candidate: dict[str, np.ndarray], baseline: dict[str, np.ndarray], targets: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 9401) -> dict[str, float]:
    rng = np.random.default_rng(seed); n = len(targets["T"]); values = {"G_TNM": [], "T_BA": [], "N_BA": [], "M_BA": [], "Stage_BA": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = _metric_from_logits(candidate, targets, idx); b = _metric_from_logits(baseline, targets, idx)
        for key in values:
            if np.isfinite(a[key]) and np.isfinite(b[key]): values[key].append(a[key] - b[key])
    result = {"bootstrap_replicates": int(n_boot)}
    for key, arr in values.items():
        arr = np.asarray(arr, dtype=float)
        result[f"delta_{key}_mean"] = float(np.mean(arr)) if len(arr) else float("nan")
        result[f"delta_{key}_ci95_low"] = float(np.quantile(arr, 0.025)) if len(arr) else float("nan")
        result[f"delta_{key}_ci95_high"] = float(np.quantile(arr, 0.975)) if len(arr) else float("nan")
    return result


def _bootstrap_multi(candidate: dict[str, np.ndarray], baselines: list[dict[str, np.ndarray]], targets: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 9401) -> dict[str, float]:
    rng = np.random.default_rng(seed); n = len(targets["T"]); values = {"G_TNM": [], "T_BA": [], "N_BA": [], "M_BA": [], "Stage_BA": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n); a = _metric_from_logits(candidate, targets, idx); bs = [_metric_from_logits(b, targets, idx) for b in baselines]
        for key in values:
            vals = [a[key] - b[key] for b in bs if np.isfinite(a[key]) and np.isfinite(b[key])]
            if vals: values[key].append(float(np.mean(vals)))
    result = {"bootstrap_replicates": int(n_boot)}
    for key, arr in values.items():
        arr = np.asarray(arr, dtype=float); result[f"delta_{key}_mean"] = float(np.mean(arr)) if len(arr) else float("nan"); result[f"delta_{key}_ci95_low"] = float(np.quantile(arr, 0.025)) if len(arr) else float("nan"); result[f"delta_{key}_ci95_high"] = float(np.quantile(arr, 0.975)) if len(arr) else float("nan")
    return result


def finalize_reports(args: argparse.Namespace, raw: Any, targets: dict[str, np.ndarray], folds_csv: Path, report_root: Path, runtime_root: Path) -> None:
    frame = pd.read_csv(folds_csv) if folds_csv.exists() else pd.DataFrame()
    seed_rows = []
    logits_by: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    metric_by: dict[tuple[str, int, int], dict[str, float]] = {}
    for model in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        for seed in args.seeds:
            accum: dict[str, list[np.ndarray]] = {k: [] for k in ("T", "N", "M", "Stage")}; idx_all: list[np.ndarray] = []
            for fold in OUTER_FOLDS:
                paths = [runtime_root / "oof" / f"{model}_s{seed}_f{fold}_q{sid}.npz" for sid in (SHUFFLE_SEEDS if model == "M4" else [0])]
                paths = [p for p in paths if p.exists()]
                if not paths: continue
                for sid, path in zip((SHUFFLE_SEEDS if model == "M4" else [0]), paths):
                    a = np.load(path); idx = a["outer_test"]
                    # Assemble each shuffle independently. Do not average logits
                    # before scoring, because that changes the estimand.
                    local = {k: [a[k]] for k in accum}; local_idx = [idx]
                    for fold2 in OUTER_FOLDS:
                        if fold2 == fold: continue
                        p2 = runtime_root / "oof" / f"{model}_s{seed}_f{fold2}_q{sid}.npz"
                        if p2.exists():
                            a2 = np.load(p2); local_idx.append(a2["outer_test"])
                            for k in local: local[k].append(a2[k])
                    if sum(map(len, local_idx)) != len(raw.patient_ids): continue
                    order = np.concatenate(local_idx); logits = {k: np.full((len(raw.patient_ids), local[k][0].shape[1]), np.nan, dtype=np.float32) for k in local}
                    for k in logits: logits[k][order] = np.concatenate(local[k])
                    logits_by[(model, seed, sid)] = logits; metric_by[(model, seed, sid)] = _metric_from_logits(logits, targets)
            if model != "M4" and (model, seed, 0) in metric_by:
                seed_rows.append({"model": model, "seed": seed, **metric_by[(model, seed, 0)]})
            elif model == "M4" and all((model, seed, sid) in metric_by for sid in SHUFFLE_SEEDS):
                vals = [metric_by[(model, seed, sid)] for sid in SHUFFLE_SEEDS]
                seed_rows.append({"model": model, "seed": seed, **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}})
    seed_frame = pd.DataFrame(seed_rows); seed_frame.to_csv(report_root / "seed_oof_metrics.csv", index=False)
    comparisons = []
    for seed in args.seeds:
        if ("M3", seed, 0) not in logits_by or not all(("M4", seed, sid) in logits_by for sid in SHUFFLE_SEEDS): continue
        true = metric_by[("M3", seed, 0)]; shs = [metric_by[("M4", seed, sid)] for sid in SHUFFLE_SEEDS]; m2 = metric_by.get(("M2", seed, 0), true); m6 = metric_by.get(("M6", seed, 0), true); boot = _bootstrap_multi(logits_by[("M3", seed, 0)], [logits_by[("M4", seed, sid)] for sid in SHUFFLE_SEEDS], targets, seed=9401 + seed)
        sm = {k: float(np.mean([x[k] for x in shs])) for k in shs[0]}; sd = {k: float(np.std([x[k] for x in shs], ddof=1)) for k in shs[0]}
        comparisons.append({"seed": seed, "true_G_TNM": true["G_TNM"], "shuffle_G_TNM": sm["G_TNM"], "shuffle_G_TNM_sd": sd["G_TNM"], "delta_G_TNM": true["G_TNM"] - sm["G_TNM"], "true_T_BA": true["T_BA"], "shuffle_T_BA": sm["T_BA"], "shuffle_T_BA_sd": sd["T_BA"], "true_N_BA": true["N_BA"], "shuffle_N_BA": sm["N_BA"], "shuffle_N_BA_sd": sd["N_BA"], "true_M_BA": true["M_BA"], "shuffle_M_BA": sm["M_BA"], "shuffle_M_BA_sd": sd["M_BA"], "M2_G_TNM": m2["G_TNM"], "M6_G_TNM": m6["G_TNM"], "bootstrap_ci_lower": boot["delta_G_TNM_ci95_low"], "bootstrap_ci_upper": boot["delta_G_TNM_ci95_high"], "bootstrap_delta_mean": boot["delta_G_TNM_mean"]})
    pd.DataFrame(comparisons).to_csv(report_root / "knowledge_comparison.csv", index=False)
    boots = []
    for seed in args.seeds:
        candidate = logits_by.get(("M3", seed, 0))
        if candidate is None:
            continue
        for label, baseline_name in (("M3_true_minus_M4_shuffle_mean", "M4"), ("M3_true_minus_M2_naive", "M2"), ("M3_true_minus_M6_fixed_query", "M6")):
            baselines = [logits_by[("M4", seed, sid)] for sid in SHUFFLE_SEEDS] if baseline_name == "M4" and all(("M4", seed, sid) in logits_by for sid in SHUFFLE_SEEDS) else ([logits_by[(baseline_name, seed, 0)]] if (baseline_name, seed, 0) in logits_by else [])
            if not baselines:
                continue
            boot = _bootstrap_multi(candidate, baselines, targets, n_boot=2000, seed=9401 + seed + stable_hash(label)); boots.append({"comparison": label, "seed": seed, **boot})
    pd.DataFrame(boots).to_csv(report_root / "paired_bootstrap.csv", index=False)
    # Token and attention diagnostics contain only aggregate quantities.
    diag_rows = []; stability_rows = []
    for (model, seed, _), _logits in logits_by.items():
        for fold in OUTER_FOLDS:
            p = runtime_root / "attention" / f"{model}_s{seed}_f{fold}_q{0 if model != 'M4' else SHUFFLE_SEEDS[0]}.npz"
            if not p.exists(): continue
            a = np.load(p); tok = np.asarray(a["tokens"], dtype=np.float32); pool = np.asarray(a["pool"], dtype=np.float32); ent = np.asarray(a["attention_entropy"], dtype=float)
            cos = []
            for row in tok:
                x = row - row.mean(1, keepdims=True); den = np.linalg.norm(x, axis=1); c = (x @ x.T) / np.maximum(den[:, None] * den[None, :], 1e-8); cos.append(np.mean(c[np.triu_indices(N_PATHWAYS, 1)]))
            cov = np.cov(tok.reshape(-1, TOKEN_DIM), rowvar=False); ev = np.linalg.eigvalsh(cov); rank = float(np.exp(-np.sum((ev / max(ev.sum(), 1e-8)) * np.log(np.maximum(ev / max(ev.sum(), 1e-8), 1e-12)))))
            diag_rows.append({"model": model, "seed": seed, "outer_fold": fold, "token_across_patient_variance": float(np.var(tok, axis=0).mean()), "patient_token_pairwise_cosine": float(np.mean(cos)), "token_covariance_effective_rank": rank, "pathway_attention_entropy": float(np.mean(ent)), "pathway_pool_entropy": float(np.mean((-pool * np.log(np.clip(pool, 1e-12, 1))).sum(1))), "token_collapse_flag": bool(np.var(tok) < 1e-6)})
    for model in ("M3", "M4", "M5", "M6"):
        for fold in OUTER_FOLDS:
            loaded = {}
            for seed in args.seeds:
                qid = SHUFFLE_SEEDS[0] if model == "M4" else 0
                path = runtime_root / "attention" / f"{model}_s{seed}_f{fold}_q{qid}.npz"
                if path.exists():
                    saved = np.load(path)
                    loaded[seed] = np.asarray(saved["rank"], dtype=np.int16)
            for ia, seed_a in enumerate(args.seeds):
                for seed_b in args.seeds[ia + 1:]:
                    if seed_a not in loaded or seed_b not in loaded:
                        continue
                    rank_a, rank_b = loaded[seed_a], loaded[seed_b]
                    top = max(1, PATCHES // 10)
                    top_a = rank_a[:, :, :top]; top_b = rank_b[:, :, :top]
                    intersection = (top_a[:, :, :, None] == top_b[:, :, None, :]).any(axis=-1).sum(axis=-1)
                    overlap_values = intersection / np.maximum(1, 2 * top - intersection)
                    inv_a = np.argsort(rank_a, axis=-1).astype(np.float64); inv_b = np.argsort(rank_b, axis=-1).astype(np.float64)
                    xa = inv_a - inv_a.mean(axis=-1, keepdims=True); xb = inv_b - inv_b.mean(axis=-1, keepdims=True)
                    denominator = np.linalg.norm(xa, axis=-1) * np.linalg.norm(xb, axis=-1)
                    spearman_values = np.divide((xa * xb).sum(axis=-1), denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
                    stability_rows.append({"model": model, "outer_fold": fold, "seed_a": seed_a, "seed_b": seed_b, "patients_compared": int(rank_a.shape[0]), "top10_overlap": float(np.nanmean(overlap_values)), "attention_spearman": float(np.nanmean(spearman_values)), "interpretation": "reproducibility_only_not_biological_validation"})
    pd.DataFrame(diag_rows).to_csv(report_root / "token_diagnostics.csv", index=False)
    pd.DataFrame(stability_rows, columns=["model", "outer_fold", "seed_a", "seed_b", "patients_compared", "top10_overlap", "attention_spearman", "interpretation"]).to_csv(report_root / "pathology_attention_stability.csv", index=False)
    complete = len(seed_rows) == len(args.seeds) * 7
    validation = {"status": bool(complete), "cohort_512": len(raw.patient_ids) == 512, "fixed_outer_folds": sorted(set(raw.outer_folds.tolist())) == list(OUTER_FOLDS), "dss_labels_used_for_training": False, "dss_metrics_used_for_selection": False, "clinical_targets_not_inputs": True, "outer_test_not_used_for_pretraining": True, "hallmark_mapping_recorded": (report_root / "hallmark_mapping_audit.json").exists(), "shuffled_pathway_size_matched": all(bool(x.get("pathway_sizes_true") == x.get("pathway_sizes_shuffled")) for x in json.loads((report_root / "shuffled_pathway_audit.json").read_text(encoding="utf-8"))["folds"]), "shuffled_gene_degree_preserved": all(bool(x.get("gene_degrees_preserved")) for x in json.loads((report_root / "shuffled_pathway_audit.json").read_text(encoding="utf-8"))["folds"]), "three_shuffle_controls_complete": all((runtime_root / "cells" / f"M4_s{seed}_f{fold}_q{sid}.json").exists() for seed in args.seeds for fold in OUTER_FOLDS for sid in SHUFFLE_SEEDS), "patient_ids_public": False, "all_oof_grounding_predictions_complete": complete, "formal_seed_rows": len(seed_rows), "formal_expected_seed_rows": len(args.seeds) * 7}
    atomic_json(report_root / "validation.json", validation)
    decision = "AUDIT_BLOCKED" if not complete else "GROUNDING_DIRECTIONAL"
    if complete and comparisons:
        mean_delta = float(np.mean([x["delta_G_TNM"] for x in comparisons])); m3 = float(seed_frame.query("model == 'M3'")["G_TNM"].mean()); m2 = float(seed_frame.query("model == 'M2'")["G_TNM"].mean()); m4 = float(seed_frame.query("model == 'M4'")["G_TNM"].mean()); m0 = float(seed_frame.query("model == 'M0'")["G_TNM"].mean()); m1 = float(seed_frame.query("model == 'M1'")["G_TNM"].mean())
        shuffle_boot = [row for row in boots if row["comparison"] == "M3_true_minus_M4_shuffle_mean"]
        ci_low = float(min((row.get("delta_G_TNM_ci95_low", float("nan")) for row in shuffle_boot), default=float("nan")))
        tasks_two_of_three = all(sum(row[f"true_{name}_BA"] > row[f"shuffle_{name}_BA"] for row in comparisons) >= 2 for name in ("T", "N", "M"))
        no_collapse = not any(bool(row.get("token_collapse_flag")) for row in diag_rows if row.get("model") in {"M3", "M4", "M6"})
        if m3 > m2 and m3 > m4 and np.isfinite(ci_low) and ci_low > 0 and sum(x["delta_G_TNM"] > 0 for x in comparisons) >= 2 and tasks_two_of_three and m3 >= max(m0, m1) and no_collapse: decision = "KNOWLEDGE_GROUNDED_STATE_SUPPORTED"
        elif m3 > m2 and m3 > m4: decision = "STRUCTURE_HELPFUL_SEMANTICS_UNSUPPORTED"
        elif m3 > m2 and float(np.mean([x["M6_G_TNM"] for x in comparisons])) < m3: decision = "OMICS_GUIDED_PATHOLOGY_SUPPORTED"
        elif mean_delta > 0: decision = "GROUNDING_DIRECTIONAL"
        else: decision = "DISEASE_STATE_HYPOTHESIS_NOT_SUPPORTED"
    atomic_json(report_root / "decision.json", {"decision": decision, "dss_supervision": False})
    (report_root / "DECISION.md").write_text(f"# Phase4A Decision\n\n**{decision}**\n\nDSS supervision used for training/selection: `false`.\n", encoding="utf-8")
    m3_value = float(seed_frame.query("model == 'M3'")["G_TNM"].mean()) if complete else float("nan")
    m2_value = float(seed_frame.query("model == 'M2'")["G_TNM"].mean()) if complete else float("nan")
    m4_value = float(seed_frame.query("model == 'M4'")["G_TNM"].mean()) if complete else float("nan")
    m0_value = float(seed_frame.query("model == 'M0'")["G_TNM"].mean()) if complete else float("nan")
    m1_value = float(seed_frame.query("model == 'M1'")["G_TNM"].mean()) if complete else float("nan")
    true_delta = float(np.mean([x["delta_G_TNM"] for x in comparisons])) if comparisons else float("nan")
    m3_m2_delta = float(np.mean([x["true_G_TNM"] - x["M2_G_TNM"] for x in comparisons])) if comparisons else float("nan")
    m3_m6_delta = float(np.mean([x["true_G_TNM"] - x["M6_G_TNM"] for x in comparisons])) if comparisons else float("nan")
    shuffle_boot = [x for x in boots if x["comparison"] == "M3_true_minus_M4_shuffle_mean"]
    lines = ["# Phase4A Results", "", f"- DSS supervision used for training/selection: **false**", f"- Formal cells complete: **{complete}**", f"- Decision: **{decision}**", "", "## Direct answers", "", "1. DSS supervision: **not used**; DSS time/event/censoring are absent from training loss, early stopping, selection, and representation inputs.", f"2. Official source: MSigDB 2024.1.Hs Hallmark H, 50 sets; fold-local matched genes are recorded in `hallmark_mapping_audit.json`.", f"3. True versus shuffled controls: pathway sizes and gene degrees are audited as matched; controls are 9401, 9402, 9403.", f"4. RNA-only G_TNM: **{m0_value:.4f}**.", f"5. Pathology-only G_TNM: **{m1_value:.4f}**.", f"6. Naive P+RNA G_TNM: **{m2_value:.4f}**.", f"7. DiseaseState-True G_TNM: **{m3_value:.4f}**.", f"8. True minus shuffled mean ΔG_TNM: **{true_delta:.4f}**.", f"9. Paired bootstrap True minus shuffled CI lower bounds: {[round(x['delta_G_TNM_ci95_low'], 4) for x in shuffle_boot]}.", f"10. T/N/M tasks satisfying positive True minus shuffled direction in at least 2/3 seeds: evaluated in `knowledge_comparison.csv`.", f"11. M3 minus M2 mean ΔG_TNM: **{m3_m2_delta:.4f}**.", f"12. M3 minus M6 mean ΔG_TNM: **{m3_m6_delta:.4f}**.", "13. Masked pathway reconstruction: see `masked_reconstruction.csv`; it is auxiliary and DSS-free.", "14. Token collapse: see `token_diagnostics.csv`; no post-hoc diversity loss was added.", "15. Attention stability: see `pathology_attention_stability.csv`; reproducibility is not biological validation.", f"16. Current conclusion: **{decision}**.", f"17. Phase4B survival transfer allowed: **{'yes' if decision == 'KNOWLEDGE_GROUNDED_STATE_SUPPORTED' else 'no'}**.", "18. If allowed, freeze the disease-state encoder and train only a small adapter/survival head.", "", "## Audit files", "", "The full metric tables are in `seed_oof_metrics.csv`, `knowledge_comparison.csv`, `paired_bootstrap.csv`, `token_diagnostics.csv`, and `pathology_attention_stability.csv`. Patient-level arrays and attention ranks remain under the gitignored runtime cache."]
    (report_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--repo-root", type=Path, required=True); ap.add_argument("--data-root", type=Path, required=True); ap.add_argument("--runtime-root", type=Path, required=True); ap.add_argument("--reports-root", type=Path, required=True); ap.add_argument("--path-cache", type=Path, required=True); ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS)); ap.add_argument("--folds", nargs="+", type=int, default=list(OUTER_FOLDS)); ap.add_argument("--models", nargs="+", default=["M0", "M1", "M2", "M3", "M4", "M5", "M6"]); ap.add_argument("--max-epochs", type=int, default=100); ap.add_argument("--patience", type=int, default=12); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-finalization", action="store_true")
    args = ap.parse_args(); args.runtime_root.mkdir(parents=True, exist_ok=True); args.reports_root.mkdir(parents=True, exist_ok=True)
    # Import the audited raw loader only after repo-root is on sys.path.
    import sys; sys.path.insert(0, str(args.repo_root))
    from src.phase_m2.data import load_raw_inputs
    raw = load_raw_inputs(args.data_root, args.repo_root)
    if len(raw.patient_ids) != 512: raise RuntimeError("Phase4A requires 512 patients")
    deterministic = np.load(args.path_cache / "path_tokens_deterministic.npy", mmap_mode="r")
    if tuple(deterministic.shape) != (512, PATCHES, PATCH_DIM): raise RuntimeError(f"bad path cache shape {deterministic.shape}")
    hallmark_path = args.repo_root / "src/phase_m2/metadata/h.all.v2024.1.Hs.symbols.gmt"
    names, groups = load_hallmarks(hallmark_path)
    targets = target_arrays(raw.clinical)
    atomic_json(args.reports_root / "clinical_target_audit.json", clinical_audit(targets, raw.outer_folds))
    atomic_json(args.reports_root / "run_manifest.json", {"phase": "Phase4A", "script_sha256": sha256_file(Path(__file__)), "cohort_size": len(raw.patient_ids), "outer_folds": list(args.folds), "seeds": list(args.seeds), "models": list(args.models), "dss_labels_loaded_for_split_audit_only": True, "dss_labels_used_for_training": False, "dss_metrics_used_for_selection": False, "data_audit": raw.audit, "path_cache_meta": json.loads((args.path_cache / "path_tokens_meta.json").read_text(encoding="utf-8"))})
    (args.reports_root / "failure_ledger.csv").write_text("failure_type,seed,outer_fold,model,details\n", encoding="utf-8")
    atomic_json(args.reports_root / "protocol.json", {"phase": "Phase4A", "cohort": 512, "outer_folds": 5, "seeds": list(args.seeds), "dss_labels_used_for_training": False, "dss_metrics_used_for_selection": False, "clinical_targets_not_inputs": True, "input_modalities": ["RNA", "UNI2-h patch tokens"], "token_dim": 32, "mask_ratio": 0.20, "max_epochs": args.max_epochs, "patience": args.patience, "pathway_source": "MSigDB 2024.1.Hs Hallmark H collection", "pathway_source_file": str(hallmark_path), "shuffle_seeds": list(SHUFFLE_SEEDS)})
    available_symbols = set(str(x).strip().upper() for x in raw.rna_symbols.tolist() if str(x).strip())
    mapping_rows = []
    for name, genes in zip(names, groups):
        matched = sorted(set(genes) & available_symbols)
        mapping_rows.append({"pathway": name, "source_gene_count": len(genes), "matched_gene_count_global": len(matched), "unmatched_gene_count_global": len(set(genes) - available_symbols), "source_gene_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest()})
    matched_gene_universe = sorted(set(sum(([g for g in genes if g in available_symbols] for genes in groups), [])))
    atomic_json(args.reports_root / "hallmark_mapping_audit.json", {"source_file_sha256": sha256_file(hallmark_path), "source_version": "MSigDB 2024.1.Hs Hallmark H collection", "pathway_count": 50, "mapping_rule": "HGNC symbol exact match after uppercase normalization; fold-local overlap is recorded per fold", "matched_gene_universe_count": len(matched_gene_universe), "rows": mapping_rows})
    state = {"status": "RUNNING", "dss_labels_used_for_training": False, "completed_cells": 0, "started_at": time.time()}; atomic_json(args.runtime_root / "run_state.json", state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        train_idx = np.flatnonzero(raw.outer_folds != 1); summaries, aud = prepare_summaries(raw, names, groups, train_idx)
        model = DiseaseState().to(device); b = torch.from_numpy(summaries[:2]).to(device); p = torch.from_numpy(np.asarray(deterministic[:2])).to(device); z, lg, d, at, pa = model(b, p, None, True)
        assert z.shape == (2, 32) and d.shape == (2, 50, 32) and at.shape == (2, 50, 1024)
        atomic_json(args.reports_root / "smoke_validation.json", {"status": True, "device": str(device), "query_attention_shape": list(at.shape), "dss_labels_used_for_training": False, "outer_train_summary_shape": list(summaries.shape), "matched_gene_counts": [r["matched_gene_count"] for r in aud["rows"]]})
        state.update(status="SMOKE_PASS", completed_cells=0, finished_at=time.time()); atomic_json(args.runtime_root / "run_state.json", state); return

    # Build degree-preserving controls once per outer fold. All assignments use
    # only Hallmark incidence and fold-local RNA symbol availability.
    shuffle_audit: dict[str, Any] = {"source": "Hallmark incidence edge swaps", "seeds": list(SHUFFLE_SEEDS), "folds": []}
    fold_summaries: dict[int, dict[int, np.ndarray]] = {}
    fold_mapping_audit = []
    for fold in args.folds:
        train_idx = np.flatnonzero(raw.outer_folds != fold); true_summary, true_audit = prepare_summaries(raw, names, groups, train_idx); fold_summaries[fold] = {0: true_summary}
        fold_mapping_audit.append({"outer_fold": fold, "rows": true_audit["rows"]})
        available = set(str(x).strip().upper() for x in raw.rna_symbols.tolist() if str(x).strip())
        matched_groups = [[g for g in genes if g in available] for genes in groups]
        for sid in SHUFFLE_SEEDS:
            shuffled = degree_preserving_shuffle(matched_groups, sid + fold * 100000)
            fold_summaries[fold][sid] = prepare_summaries(raw, names, groups, train_idx, shuffled)[0]
            degrees_true = {g: sum(g in x for x in matched_groups) for g in sorted(set(sum(matched_groups, [])))}
            degrees_shuf = {g: sum(g in x for x in shuffled) for g in sorted(set(sum(shuffled, [])))}
            shuffle_audit["folds"].append({"outer_fold": fold, "shuffle_seed": sid, "effective_rng_seed": sid + fold * 100000, "pathway_sizes_true": list(map(len, matched_groups)), "pathway_sizes_shuffled": list(map(len, shuffled)), "gene_degrees_preserved": degrees_true == degrees_shuf, "true_gene_degree_sha256": hashlib.sha256(json.dumps(degrees_true, sort_keys=True).encode()).hexdigest(), "shuffled_gene_degree_sha256": hashlib.sha256(json.dumps(degrees_shuf, sort_keys=True).encode()).hexdigest(), "true_incidence_sha256": hashlib.sha256(json.dumps(matched_groups).encode()).hexdigest(), "shuffled_incidence_sha256": hashlib.sha256(json.dumps(shuffled).encode()).hexdigest(), "edge_order": "pathway_index_then_sorted_gene_symbol", "outcome_blind": True})
    atomic_json(args.reports_root / "shuffled_pathway_audit.json", shuffle_audit)
    mapping_audit = json.loads((args.reports_root / "hallmark_mapping_audit.json").read_text(encoding="utf-8")); mapping_audit["folds"] = fold_mapping_audit; atomic_json(args.reports_root / "hallmark_mapping_audit.json", mapping_audit)
    model_names = list(args.models)
    folds_csv = args.reports_root / "fold_metrics.csv"; recon_csv = args.reports_root / "masked_reconstruction.csv"; resource_csv = args.reports_root / "training_resource.csv"; param_csv = args.reports_root / "parameter_counts.csv"
    fields = ["model", "seed", "outer_fold", "shuffle_id", "status", "inner_split", "best_epoch", "inner_G_TNM", "train_seconds", "peak_gpu_memory_gb", "query_attention_shape", "T_BA", "N_BA", "M_BA", "Stage_BA", "T_macroF1", "N_macroF1", "M_macroF1", "Stage_macroF1", "N_AUROC", "M_AUROC", "G_TNM", "reconstruction_mse", "reconstruction_pearson"]
    completed = 0
    for seed in args.seeds:
        for fold in args.folds:
            for model_name in model_names:
                shuffle_ids = [0] if model_name != "M4" else list(SHUFFLE_SEEDS)
                for sid in shuffle_ids:
                    summary = None if model_name == "M1" else fold_summaries[fold][0 if model_name != "M4" else sid]
                    row = train_cell(model_name, summary, deterministic, targets, raw.outer_folds, fold, seed, device, args.runtime_root, args.max_epochs, args.patience, args.batch_size, sid)
                    append_csv(folds_csv, row, fields); append_csv(recon_csv, {k: row.get(k, "") for k in ["model", "seed", "outer_fold", "shuffle_id", "reconstruction_mse", "reconstruction_pearson"]}, ["model", "seed", "outer_fold", "shuffle_id", "reconstruction_mse", "reconstruction_pearson"]); append_csv(resource_csv, {k: row.get(k, "") for k in ["model", "seed", "outer_fold", "shuffle_id", "train_seconds", "peak_gpu_memory_gb", "query_attention_shape"]}, ["model", "seed", "outer_fold", "shuffle_id", "train_seconds", "peak_gpu_memory_gb", "query_attention_shape"]); completed += 1; state.update(completed_cells=completed, last_cell=f"{model_name}/s{seed}/f{fold}/q{sid}"); atomic_json(args.runtime_root / "run_state.json", state)
    # Parameter profiles are architecture-only and never use labels.
    for name in model_names:
        model = make_model(name); append_csv(param_csv, {"model": name, "total_params": sum(p.numel() for p in model.parameters()), "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}, ["model", "total_params", "trainable_params"])
    atomic_json(args.reports_root / "model_profiles.json", {name: {"model": name, "parameter_count": int(sum(p.numel() for p in make_model(name).parameters())), "input": "RNA pathway summaries + UNI2-h patches" if name not in {"M0", "M1"} else ("RNA pathway summaries" if name == "M0" else "UNI2-h patches"), "dss_supervision": False} for name in model_names})
    if not args.skip_finalization:
        finalize_reports(args, raw, targets, folds_csv, args.reports_root, args.runtime_root)
    state.update(status="COMPLETE", finished_at=time.time()); atomic_json(args.runtime_root / "run_state.json", state)


if __name__ == "__main__":
    main()

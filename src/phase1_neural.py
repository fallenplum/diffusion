from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

from phase1_utils import censoring_km, harrell_c, set_seed, uno_c, update_run_state


SEED = 20260723
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class EncoderConfig:
    kind: str = "context"
    input_dim: int = 1536
    dim: int = 256
    heads: int = 8
    layers: int = 2
    ffn_dim: int = 512
    dropout: float = 0.1
    use_coords: bool = True
    max_tokens: int = 1024


@dataclass(frozen=True)
class SSLConfig:
    name: str
    mask_type: str
    mask_ratio: float
    use_coords: bool
    max_epochs: int = 50
    patience: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4


SSL_CONFIGS = {
    "spatial040": SSLConfig("spatial040", "spatial", 0.40, True),
    "random040": SSLConfig("random040", "random", 0.40, True),
    "spatial025": SSLConfig("spatial025", "spatial", 0.25, True),
    "spatial040_nocoords": SSLConfig("spatial040_nocoords", "spatial", 0.40, False),
}


class ContextEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, config.dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.output_norm = nn.LayerNorm(config.dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, features, coords, mask=None):
        tokens = self.projection(features)
        if self.config.use_coords:
            tokens = tokens + self.coord_mlp(coords)
        if mask is not None:
            replacement = self.mask_token.expand(features.shape[0], features.shape[1], -1)
            if self.config.use_coords:
                replacement = replacement + self.coord_mlp(coords)
            tokens = torch.where(mask.unsqueeze(-1), replacement, tokens)
        cls = self.cls_token.expand(features.shape[0], -1, -1)
        encoded = self.output_norm(self.transformer(torch.cat([cls, tokens], dim=1)))
        return encoded[:, 0], encoded[:, 1:]


class ABMIL(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.attention_v = nn.Linear(config.dim, 128)
        self.attention_u = nn.Linear(config.dim, 128)
        self.attention_w = nn.Linear(128, 1)

    def forward(self, features, coords=None, mask=None):
        tokens = self.projection(features)
        logits = self.attention_w(torch.tanh(self.attention_v(tokens)) * torch.sigmoid(self.attention_u(tokens)))
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights * tokens, dim=1)
        return pooled, weights.squeeze(-1)


class SSLModel(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.encoder = ContextEncoder(config)
        self.decoder = nn.Sequential(nn.Linear(config.dim, 512), nn.GELU(), nn.Linear(512, config.input_dim))

    def forward(self, features, coords, mask):
        _, tokens = self.encoder(features, coords, mask)
        return self.decoder(tokens[mask])


class SurvivalModel(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = ABMIL(config) if config.kind == "abmil" else ContextEncoder(config)
        self.cox_head = nn.Linear(config.dim, 1)

    def forward(self, features, coords):
        z, importance = self.encoder(features, coords)
        return z, self.cox_head(z).squeeze(-1), importance


class BagStore:
    def __init__(self, root: Path, slides: pd.DataFrame, max_tokens: int = 1024):
        self.root = root
        self.max_tokens = max_tokens
        self.slides = slides.copy()
        self.by_patient = {
            str(pid): group.to_dict("records") for pid, group in self.slides.groupby("patient_id", sort=False)
        }
        self.bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.coverage: dict[str, set[int]] = defaultdict(set)

    def sample_slide(self, row: dict, rng: np.random.Generator, track: bool = False):
        bag = np.load(self.root / str(row["bag_path"]), mmap_mode="r", allow_pickle=False)
        coords = np.load(self.root / str(row["coords_path"]), mmap_mode="r", allow_pickle=False)
        n = len(bag)
        if n <= self.max_tokens:
            indices = np.arange(n)
        else:
            indices = np.sort(rng.choice(n, self.max_tokens, replace=False))
        if track:
            self.coverage[str(row["slide_id"])].update(indices.tolist())
        x = np.asarray(bag[indices], dtype=np.float32)
        raw_coords = np.asarray(coords[indices], dtype=np.float32)
        key = str(row["coords_path"])
        if key not in self.bounds:
            all_coords = np.asarray(coords, dtype=np.float64)
            self.bounds[key] = (all_coords.min(axis=0), all_coords.max(axis=0))
        low, high = self.bounds[key]
        norm_coords = (raw_coords - low) / np.maximum(high - low, 1.0)
        return x, norm_coords.astype(np.float32), indices

    def patient_tensor(self, pid: str, rng: np.random.Generator, track: bool = False):
        return [self.sample_slide(row, rng, track) for row in self.by_patient[str(pid)]]

    def save_coverage(self, path: Path):
        rows = []
        for row in self.slides.to_dict("records"):
            slide = str(row["slide_id"])
            unique = len(self.coverage.get(slide, set()))
            rows.append({
                "patient_id": row["patient_id"], "slide_id": slide,
                "n_patches": int(row["n_patches"]), "unique_sampled": unique,
                "coverage": unique / int(row["n_patches"]),
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)


def tensor_batch(x: np.ndarray, coords: np.ndarray):
    return (
        torch.from_numpy(x).unsqueeze(0).to(DEVICE, non_blocking=True),
        torch.from_numpy(coords).unsqueeze(0).to(DEVICE, non_blocking=True),
    )


def make_mask(coords: np.ndarray, ratio: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_tokens, 2)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords contains NaN or Inf")
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("mask ratio must be between 0 and 1")
    n = len(coords)
    if n < 2:
        raise ValueError("at least two tokens are required for masking")
    count = min(n - 1, max(1, int(round(n * ratio))))
    if kind == "random":
        selected = rng.choice(n, count, replace=False)
    elif kind == "spatial":
        selected_set: set[int] = set()
        block_count = min(4, max(1, count // 64))
        per_block = math.ceil(count / block_count)
        anchors = rng.choice(n, block_count, replace=False)
        for anchor in anchors:
            distances = np.square(coords - coords[anchor]).sum(axis=1)
            selected_set.update(np.argpartition(distances, min(per_block, n - 1))[:per_block].tolist())
        if len(selected_set) < count:
            remaining = np.setdiff1d(np.arange(n), np.fromiter(selected_set, dtype=int), assume_unique=False)
            selected_set.update(rng.choice(remaining, count - len(selected_set), replace=False).tolist())
        selected = np.asarray(sorted(selected_set)[:count], dtype=int)
    else:
        raise ValueError(kind)
    mask = np.zeros(n, dtype=bool)
    mask[selected] = True
    return mask


def patient_split(frame: pd.DataFrame, outer_fold: int, seed: int):
    outer_train = frame.loc[frame["outer_fold"] != outer_fold].copy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + outer_fold)
    train_idx, val_idx = next(splitter.split(outer_train, outer_train["dss_status"]))
    return outer_train.iloc[train_idx].copy(), outer_train.iloc[val_idx].copy(), frame.loc[frame["outer_fold"] == outer_fold].copy()


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda")


def ssl_epoch(model, store, slide_rows, config, epoch, optimizer=None):
    training = optimizer is not None
    model.train(training)
    rng = np.random.default_rng(SEED + epoch * 1009 + (0 if training else 900000))
    order = rng.permutation(len(slide_rows))
    totals = defaultdict(float)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda") if training else None
    for index in order:
        row = slide_rows.iloc[int(index)].to_dict()
        x, coords, _ = store.sample_slide(row, rng, track=training)
        mask_np = make_mask(coords, config.mask_ratio, config.mask_type, rng)
        features, coord_tensor = tensor_batch(x, coords)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(DEVICE)
        target = F.normalize(features[mask].float(), dim=-1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context():
            prediction = model(features, coord_tensor, mask)
            cosine = F.cosine_similarity(prediction.float(), target, dim=-1)
            loss = (1.0 - cosine).mean()
            mse = F.mse_loss(F.normalize(prediction.float(), dim=-1), target)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        totals["loss"] += float(loss.detach())
        totals["cosine"] += float(cosine.mean().detach())
        totals["mse"] += float(mse.detach())
        totals["mask_ratio"] += float(mask_np.mean())
        totals["n"] += 1
    return {key: value / totals["n"] for key, value in totals.items() if key != "n"}


def train_ssl(root: Path, config_name: str, fold: int, seed: int = SEED):
    config = SSL_CONFIGS[config_name]
    set_seed(seed + fold)
    phase = root / "outputs" / "pathology_context_phase1"
    ckpt = phase / "checkpoints" / "ssl" / config_name / f"fold_{fold}" / f"seed_{seed}.pt"
    if ckpt.exists():
        print(f"[skip] SSL checkpoint exists: {ckpt}", flush=True)
        return
    patients = pd.read_csv(phase / "audit" / "patient_manifest_used.csv")
    slides = pd.read_csv(phase / "audit" / "slide_manifest_used.csv")
    inner_train, inner_val, outer_test = patient_split(patients, fold, seed)
    assert not set(inner_train.patient_id) & set(outer_test.patient_id)
    assert not set(inner_val.patient_id) & set(outer_test.patient_id)
    ssl_train_slides = slides.loc[slides.patient_id.isin(inner_train.patient_id)].reset_index(drop=True)
    ssl_val_slides = slides.loc[slides.patient_id.isin(inner_val.patient_id)].reset_index(drop=True)
    store = BagStore(root, slides)
    encoder_config = EncoderConfig(use_coords=config.use_coords)
    model = SSLModel(encoder_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_metrics = ssl_epoch(model, store, ssl_train_slides, config, epoch, optimizer)
        val_metrics = ssl_epoch(model, store, ssl_val_slides, config, epoch, None)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"[SSL {config_name} f{fold}] {json.dumps(row)}", flush=True)
        if val_metrics["loss"] < best_loss - 1e-5:
            best_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("SSL produced no checkpoint")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": {k.removeprefix("encoder."): v.cpu() for k, v in best_state.items() if k.startswith("encoder.")},
        "model_state": {k: v.cpu() for k, v in best_state.items()},
        "encoder_config": asdict(encoder_config), "ssl_config": asdict(config),
        "fold": fold, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "ssl_train_patients": sorted(inner_train.patient_id.astype(str).tolist()),
        "ssl_val_patients": sorted(inner_val.patient_id.astype(str).tolist()),
        "outer_test_patients": sorted(outer_test.patient_id.astype(str).tolist()),
        "training_time_seconds": time.perf_counter() - started,
    }, ckpt)
    metric_dir = phase / "ssl_metrics" / config_name / f"fold_{fold}" / f"seed_{seed}"
    metric_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    history_frame[["epoch", "train_loss"]].to_csv(metric_dir / "ssl_train_loss.csv", index=False)
    history_frame[["epoch", "val_loss"]].to_csv(metric_dir / "ssl_val_loss.csv", index=False)
    history_frame[["epoch", "train_cosine", "val_cosine"]].to_csv(metric_dir / "masked_cosine_similarity.csv", index=False)
    history_frame[["epoch", "train_mse", "val_mse"]].to_csv(metric_dir / "masked_mse.csv", index=False)
    history_frame.to_csv(metric_dir / "history.csv", index=False)
    (metric_dir / "ssl_patient_ids.json").write_text(json.dumps({
        "train": sorted(inner_train.patient_id.astype(str).tolist()),
        "val": sorted(inner_val.patient_id.astype(str).tolist()),
        "outer_test": sorted(outer_test.patient_id.astype(str).tolist()),
        "overlap": 0,
    }, indent=2), encoding="utf-8")
    store.save_coverage(metric_dir / "sampling_coverage.csv")
    update_run_state(root, f"SSL_{config_name}/fold_{fold}/seed_{seed}", "stage_2_ssl")


def encode_patient(encoder, store: BagStore, pid: str, rng: np.random.Generator, training: bool):
    slide_z = []
    slide_importance = []
    for x, coords, _ in store.patient_tensor(pid, rng, track=training):
        features, coord_tensor = tensor_batch(x, coords)
        with autocast_context():
            z, importance = encoder(features, coord_tensor)
        slide_z.append(z.squeeze(0))
        slide_importance.append(importance)
    return torch.stack(slide_z).mean(dim=0), slide_importance


@torch.no_grad()
def deterministic_embeddings(encoder, store: BagStore, patient_ids: Iterable[str], seeds=(11, 23, 47, 89)):
    encoder.eval()
    output = []
    for pid in patient_ids:
        samples = []
        for sampling_seed in seeds:
            rng = np.random.default_rng(SEED + sampling_seed + sum(map(ord, str(pid))))
            z, _ = encode_patient(encoder, store, str(pid), rng, False)
            samples.append(z.float().cpu().numpy())
        output.append(np.stack(samples).mean(axis=0))
    return np.asarray(output, dtype=np.float32)


def sampled_survival_loss(model, store, patients, event_row, rng, loss_type, censor_predict):
    event_time = float(event_row.dss_months)
    risk_set = patients.loc[(patients.dss_months >= event_time) & (patients.patient_id != event_row.patient_id)]
    if risk_set.empty:
        return None
    count = min(7, len(risk_set))
    sampled = risk_set.iloc[rng.choice(len(risk_set), count, replace=False)]
    group = [str(event_row.patient_id)] + sampled.patient_id.astype(str).tolist()
    z_values = []
    for pid in group:
        z, _ = encode_patient(model.encoder, store, pid, rng, True)
        z_values.append(z)
    z_group = torch.stack(z_values)
    risks = model.cox_head(z_group).squeeze(-1)
    cox = torch.logsumexp(risks, dim=0) - risks[0]
    rank = F.softplus(-(risks[0] - risks[1:])).mean()
    if loss_type == "cox":
        return cox
    if loss_type == "cox_rank":
        return cox + 0.2 * rank
    if loss_type == "cox_ipcw":
        g = float(censor_predict(np.asarray([event_time]), left_limit=True)[0])
        weight = min(10.0, 1.0 / max(g * g, 1e-4))
        return cox + 0.2 * weight * rank
    raise ValueError(loss_type)


def evaluate_head(model, store, patients):
    z = deterministic_embeddings(model.encoder, store, patients.patient_id.astype(str))
    with torch.no_grad():
        risk = model.cox_head(torch.from_numpy(z).to(DEVICE)).squeeze(-1).float().cpu().numpy()
    return z, risk


def train_survival_candidate(model, store, inner_train, inner_val, loss_type, lr, fold, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    event_rows = inner_train.loc[inner_train.dss_status == 1]
    censor = censoring_km(inner_train.dss_months.to_numpy(), inner_train.dss_status.to_numpy())
    best_c = -float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, 21):
        model.train()
        rng = np.random.default_rng(seed + fold * 10000 + epoch * 101)
        order = rng.permutation(len(event_rows))
        losses = []
        for index in order:
            row = event_rows.iloc[int(index)]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                loss = sampled_survival_loss(model, store, inner_train, row, rng, loss_type, censor)
            if loss is None:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        _, val_risk = evaluate_head(model, store, inner_val)
        val_c = harrell_c(inner_val.dss_months, inner_val.dss_status, val_risk)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_harrell_c": val_c, "lr": lr})
        print(f"[SURV f{fold} {loss_type} lr={lr}] epoch={epoch} loss={np.mean(losses):.5f} valC={val_c:.5f}", flush=True)
        if np.isfinite(val_c) and val_c > best_c + 1e-5:
            best_c = val_c
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 5:
            break
    if best_state is None:
        raise RuntimeError("Survival training produced no checkpoint")
    return best_c, best_epoch, best_state, history


def load_ssl_encoder(root: Path, config_name: str, fold: int, seed: int):
    path = root / "outputs" / "pathology_context_phase1" / "checkpoints" / "ssl" / config_name / f"fold_{fold}" / f"seed_{seed}.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    config = EncoderConfig(**bundle["encoder_config"])
    encoder = ContextEncoder(config)
    encoder.load_state_dict(bundle["encoder_state"])
    return encoder, config


def train_survival(root: Path, model_name: str, fold: int, seed: int = SEED):
    set_seed(seed + fold)
    phase = root / "outputs" / "pathology_context_phase1"
    ckpt = phase / "checkpoints" / "survival" / model_name / f"fold_{fold}" / f"seed_{seed}.pt"
    if ckpt.exists():
        print(f"[skip] survival checkpoint exists: {ckpt}", flush=True)
        return
    patients = pd.read_csv(phase / "audit" / "patient_manifest_used.csv")
    slides = pd.read_csv(phase / "audit" / "slide_manifest_used.csv")
    inner_train, inner_val, outer_test = patient_split(patients, fold, seed)
    store = BagStore(root, slides)
    if model_name == "B1":
        config = EncoderConfig(kind="abmil", use_coords=False)
        base_model = SurvivalModel(config)
        loss_type = "cox"
        ssl_source = None
    elif model_name == "B2":
        config = EncoderConfig(kind="context", use_coords=True)
        base_model = SurvivalModel(config)
        loss_type = "cox"
        ssl_source = None
    elif model_name in {"P1", "P2", "P3"}:
        pretrained, config = load_ssl_encoder(root, "spatial040", fold, seed)
        base_model = SurvivalModel(config)
        base_model.encoder.load_state_dict(pretrained.state_dict())
        loss_type = {"P1": "cox", "P2": "cox_rank", "P3": "cox_ipcw"}[model_name]
        ssl_source = "spatial040"
    else:
        raise ValueError(model_name)

    initial = copy.deepcopy(base_model.state_dict())
    candidates = []
    all_history = []
    started = time.perf_counter()
    for lr in [1e-4, 3e-4]:
        model = SurvivalModel(config).to(DEVICE)
        model.load_state_dict(initial)
        best_c, best_epoch, state, history = train_survival_candidate(
            model, store, inner_train, inner_val, loss_type, lr, fold, seed
        )
        candidates.append((best_c, -lr, lr, best_epoch, state))
        all_history.extend(history)
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    best_c, _, best_lr, best_epoch, best_state = candidates[0]
    model = SurvivalModel(config).to(DEVICE)
    model.load_state_dict(best_state)
    train_outer = patients.loc[patients.outer_fold != fold].copy()
    train_z, train_risk = evaluate_head(model, store, train_outer)
    test_z, test_risk = evaluate_head(model, store, outer_test)
    censor = censoring_km(train_outer.dss_months.to_numpy(), train_outer.dss_status.to_numpy())
    g_test = censor(outer_test.dss_months.to_numpy(), left_limit=True)
    result = {
        "model": model_name, "seed": seed, "outer_fold": fold,
        "loss_type": loss_type, "ssl_source": ssl_source,
        "best_lr": best_lr, "best_epoch": best_epoch, "inner_val_harrell_c": best_c,
        "test_harrell_c": harrell_c(outer_test.dss_months, outer_test.dss_status, test_risk),
        "test_uno_c": uno_c(outer_test.dss_months, outer_test.dss_status, test_risk, g_test),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "training_time_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if DEVICE.type == "cuda" else 0,
        "survival_training_approximation": "nested risk-set sampling: one event plus up to seven outer-train at-risk comparators per update",
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {k: v.cpu() for k, v in best_state.items()},
        "encoder_config": asdict(config), "result": result,
        "inner_train_patients": inner_train.patient_id.astype(str).tolist(),
        "inner_val_patients": inner_val.patient_id.astype(str).tolist(),
        "outer_test_patients": outer_test.patient_id.astype(str).tolist(),
    }, ckpt)
    metric_dir = phase / "fold_metrics" / model_name
    metric_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_history).to_csv(metric_dir / f"fold_{fold}_seed_{seed}_training.csv", index=False)
    (metric_dir / f"fold_{fold}_seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    prediction_dir = phase / "oof_predictions" / "fold_parts" / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pred = outer_test[["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    pred["seed"] = seed
    pred["oof_risk"] = test_risk
    pred["censor_survival_outer_train"] = g_test
    pred.to_csv(prediction_dir / f"fold_{fold}_seed_{seed}.csv", index=False)
    emb_dir = phase / "embeddings" / model_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb = pred.copy()
    emb["embedding"] = [json.dumps(row.tolist()) for row in test_z]
    emb.to_csv(emb_dir / f"fold_{fold}_seed_{seed}.csv", index=False)
    update_run_state(root, f"{model_name}/fold_{fold}/seed_{seed}", "stage_3_survival")
    print(json.dumps(result, indent=2), flush=True)


def run_probe(root: Path, config_name: str, fold: int, seed: int = SEED):
    from phase1_utils import fit_cox, scaler_pca
    from sklearn.model_selection import StratifiedKFold

    set_seed(seed + fold)
    phase = root / "outputs" / "pathology_context_phase1"
    method = {
        "spatial040": "P0",
        "random040": "P0_random040",
        "spatial025": "P0_spatial025",
        "spatial040_nocoords": "P0_nocoords",
    }[config_name]
    part = phase / "oof_predictions" / "fold_parts" / method / f"fold_{fold}_seed_{seed}.csv"
    if part.exists():
        print(f"[skip] probe exists: {part}")
        return
    patients = pd.read_csv(phase / "audit" / "patient_manifest_used.csv")
    slides = pd.read_csv(phase / "audit" / "slide_manifest_used.csv")
    train = patients.loc[patients.outer_fold != fold].copy().reset_index(drop=True)
    test = patients.loc[patients.outer_fold == fold].copy().reset_index(drop=True)
    encoder, config = load_ssl_encoder(root, config_name, fold, seed)
    encoder = encoder.to(DEVICE)
    store = BagStore(root, slides)
    x_train = deterministic_embeddings(encoder, store, train.patient_id.astype(str))
    x_test = deterministic_embeddings(encoder, store, test.patient_id.astype(str))
    times = train.dss_months.to_numpy(float)
    events = train.dss_status.to_numpy(int)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold)
    ranked = []
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        scores = []
        for ti, vi in inner.split(x_train, events):
            scaler = torch.tensor(0)  # local name guard; no role in fitting
            xt, xv, _, _ = scaler_pca(x_train[ti], x_train[vi], min(32, x_train.shape[1]), seed + fold)
            beta = fit_cox(times[ti], events[ti], xt, alpha)
            scores.append(harrell_c(times[vi], events[vi], xv @ beta))
        ranked.append((float(np.nanmean(scores)), -alpha, alpha))
    ranked.sort(reverse=True)
    inner_c, _, alpha = ranked[0]
    xt, xv, _, _ = scaler_pca(x_train, x_test, min(32, x_train.shape[1]), seed + fold)
    beta = fit_cox(times, events, xt, alpha)
    risk = xv @ beta
    censor = censoring_km(times, events)
    g = censor(test.dss_months.to_numpy(float), left_limit=True)
    result = {
        "model": method, "ssl_config": config_name, "outer_fold": fold, "seed": seed,
        "alpha": alpha, "inner_harrell_c": inner_c,
        "test_harrell_c": harrell_c(test.dss_months, test.dss_status, risk),
        "test_uno_c": uno_c(test.dss_months, test.dss_status, risk, g),
        "n_parameters": sum(p.numel() for p in encoder.parameters()),
    }
    part.parent.mkdir(parents=True, exist_ok=True)
    pred = test[["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    pred["seed"] = seed
    pred["oof_risk"] = risk
    pred["censor_survival_outer_train"] = g
    pred.to_csv(part, index=False)
    emb_dir = phase / "embeddings" / method
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb = pred.copy()
    emb["embedding"] = [json.dumps(row.tolist()) for row in x_test]
    emb.to_csv(emb_dir / f"fold_{fold}_seed_{seed}.csv", index=False)
    metric_dir = phase / "fold_metrics" / method
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / f"fold_{fold}_seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    update_run_state(root, f"{method}/fold_{fold}/seed_{seed}", "stage_3_probe")
    print(json.dumps(result, indent=2), flush=True)


def init_jobs(root: Path):
    path = root / "outputs" / "pathology_context_phase1" / "run_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    jobs = []
    for config in SSL_CONFIGS:
        jobs.extend(f"SSL_{config}/fold_{fold}/seed_{SEED}" for fold in range(1, 6))
    for model in ["P0", "P0_random040", "P0_spatial025", "P0_nocoords", "B1", "B2", "P1", "P2", "P3"]:
        jobs.extend(f"{model}/fold_{fold}/seed_{SEED}" for fold in range(1, 6))
    # Pre-registered stability extension after every method completed seed 20260723:
    # repeat the strongest context-SSL representation plus B1/B2 at two further seeds.
    for stability_seed in [20260724, 20260725]:
        jobs.extend(f"SSL_spatial040_nocoords/fold_{fold}/seed_{stability_seed}" for fold in range(1, 6))
        jobs.extend(f"P0_nocoords/fold_{fold}/seed_{stability_seed}" for fold in range(1, 6))
        for model in ["B1", "B2"]:
            jobs.extend(f"{model}/fold_{fold}/seed_{stability_seed}" for fold in range(1, 6))
    jobs.extend(f"P2_permuted/fold_{fold}/seed_{SEED}" for fold in range(1, 6))
    for model in ["B1", "B2", "P1", "P2"]:
        jobs.extend(f"{model}_probe/fold_{fold}/seed_{SEED}" for fold in range(1, 6))
    completed = set(state["completed_jobs"])
    state["pending_jobs"] = [job for job in jobs if job not in completed]
    state["stage"] = "stage_2_ssl"
    state["status"] = "RUNNING"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    config_dir = root / "outputs" / "pathology_context_phase1" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "encoder.json").write_text(json.dumps(asdict(EncoderConfig()), indent=2), encoding="utf-8")
    (config_dir / "ssl_configs.json").write_text(json.dumps({k: asdict(v) for k, v in SSL_CONFIGS.items()}, indent=2), encoding="utf-8")
    (config_dir / "survival.json").write_text(json.dumps({
        "optimizer": "AdamW", "weight_decay": 1e-4, "lr_candidates": [1e-4, 3e-4],
        "max_epochs": 20, "patience": 5, "risk_set_comparators": 7,
        "cox_approximation": "nested risk-set sampling", "lambda_rank": 0.2, "tau": 1.0,
        "evaluation_sampling_seeds": [11, 23, 47, 89],
    }, indent=2), encoding="utf-8")
    print(json.dumps({"pending": len(state["pending_jobs"]), "completed": len(completed)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "ssl", "survival", "probe"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", choices=list(SSL_CONFIGS))
    parser.add_argument("--model", choices=["B1", "B2", "P1", "P2", "P3"])
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "init":
        init_jobs(root)
    elif args.action == "ssl":
        train_ssl(root, args.config, args.fold, args.seed)
    elif args.action == "survival":
        train_survival(root, args.model, args.fold, args.seed)
    elif args.action == "probe":
        run_probe(root, args.config, args.fold, args.seed)


if __name__ == "__main__":
    main()

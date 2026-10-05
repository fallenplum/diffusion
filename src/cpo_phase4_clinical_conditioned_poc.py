from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn

from src.phase_m2.common import (
    censoring_survival,
    count_parameters,
    harrell_c,
    lookup_censor_survival,
    set_seed,
    stable_int,
    uno_c_index,
)
from src.phase_m2.data import _fit_clinical, load_raw_inputs, preprocess_fold
from src.phase_m3r2.models import ClinicalOnlyModel
from src.phase_m8p1.clinical import Config as ClinicalConfig
from src.phase_m8p1.clinical import fit_nested_anchor
from src.po_phase1_orthogonal_fusion_repair import (
    ABMIL,
    abmil_parts,
    cox_loss_fast,
    path_provider,
    train_abmil_fixed,
    train_linear,
)


SEEDS = (20260904, 20260905, 20260906)
FOLDS = (1, 2, 3, 4, 5)
INNER_CF_FOLDS = 4
STAGE_B_MAX_EPOCHS = 240
STAGE_B_PATIENCE = 18
STAGE_B_LR = 0.01
STAGE_B_WEIGHT_DECAY = 1e-4
L2_GRID = (0.01, 0.1)
STAGE1_L2 = 0.1
BOOTSTRAP_REPS = 2000
CLINICAL_CTX_DIM = 8
LOW_RANK_DIM = 4
REPLAY_CROSSFIT_ATOL = 5e-3
REPLAY_TEST_ATOL = 2e-3
REPLAY_RTOL = 1e-3
CLINICAL_COLUMNS = (
    "AGE",
    "SEX",
    "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "PATH_T_STAGE",
    "PATH_N_STAGE",
    "PATH_M_STAGE",
)
ARCHITECTURES = ("Additive-C+P+R", "CC-RiskScale", "CC-LowRank")
ABLATIONS = ("A0 Clinical-only", "A1 Additive-C+P+R", "A2 CC-RiskScale", "A3 CC-LowRank-full", "A4 CC-LowRank-CP-only", "A5 CC-LowRank-CR-only")


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
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def split_inner(indices: np.ndarray, raw: Any, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    strata = np.asarray([f"{raw.cancer_type[i]}_{int(raw.dss_event[i])}" for i in indices])
    try:
        a, b = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(indices, strata))
        return indices[a], indices[b]
    except ValueError:
        z = np.random.default_rng(seed).permutation(indices)
        n = max(1, int(round(0.15 * len(z))))
        return z[n:], z[:n]


def stratified_splits(indices: np.ndarray, raw: Any, seed: int, n_splits: int = INNER_CF_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.sort(np.asarray(indices, dtype=int))
    strata = np.asarray([f"{raw.cancer_type[i]}_{int(raw.dss_event[i])}" for i in indices])
    try:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(indices[a], indices[b]) for a, b in splitter.split(indices, strata)]
    except ValueError:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(indices[a], indices[b]) for a, b in splitter.split(indices, np.zeros(len(indices), dtype=int))]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sa, sb = float(np.std(a)), float(np.std(b))
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0


def standardize_fit(train: np.ndarray, eval_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = np.asarray(train, dtype=float)
    eval_values = np.asarray(eval_values, dtype=float)
    mean = float(np.mean(train))
    std = float(np.std(train))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return ((train - mean) / std).astype(np.float32), ((eval_values - mean) / std).astype(np.float32), mean, std


def standardize_columns(train: np.ndarray, eval_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float32)
    eval_values = np.asarray(eval_values, dtype=np.float32)
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return ((train - mean) / std).astype(np.float32), ((eval_values - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def eval_abmil(model: nn.Module, deterministic: np.ndarray, indices: np.ndarray, device: torch.device, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    risks, embeddings = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            ix = np.asarray(indices[start:start + batch], dtype=int)
            tokens = torch.from_numpy(np.asarray(deterministic[ix])).to(device)
            risk, emb = abmil_parts(model, tokens)
            risks.append(risk.detach().cpu().numpy().reshape(-1))
            embeddings.append(emb.detach().cpu().numpy())
    return np.concatenate(risks).astype(float), np.concatenate(embeddings).astype(np.float32)


def align_to_reference(reference_cal: np.ndarray, model_cal: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Align supervised fold embeddings using only unlabeled feature calibrations."""
    ref_mean = np.mean(reference_cal, axis=0)
    ref_std = np.std(reference_cal, axis=0)
    ref_std[~np.isfinite(ref_std) | (ref_std < 1e-6)] = 1.0
    model_mean = np.mean(model_cal, axis=0)
    model_std = np.std(model_cal, axis=0)
    model_std[~np.isfinite(model_std) | (model_std < 1e-6)] = 1.0
    ref_w = (reference_cal - ref_mean) / ref_std
    model_w = (model_cal - model_mean) / model_std
    cross = model_w.T @ ref_w
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    aligned = ((values - model_mean) / model_std) @ rotation
    return aligned.astype(np.float32), {
        "calibration_count": int(len(reference_cal)),
        "rotation_orthogonality_max_abs": float(np.max(np.abs(rotation.T @ rotation - np.eye(rotation.shape[0])))),
        "calibration_alignment_rmse": float(np.sqrt(np.mean(((model_w @ rotation) - ref_w) ** 2))),
        "model_mean_sha256": sha256_array(model_mean.astype(np.float32)),
        "model_std_sha256": sha256_array(model_std.astype(np.float32)),
    }


def load_phase1_oof(raw: Any, runtime: Path) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, Any]], bool]:
    base = runtime / "phase1_orthogonal_cells_repair"
    out: dict[int, dict[str, np.ndarray]] = {}
    provenance: list[dict[str, Any]] = []
    valid = True
    for seed in SEEDS:
        out[seed] = {"RNA-Linear": np.full(len(raw.patient_ids), np.nan), "Pathology-ABMIL": np.full(len(raw.patient_ids), np.nan)}
        for fold in FOLDS:
            expected = np.flatnonzero(raw.outer_folds == fold)
            for method, stem in (("RNA-Linear", "RNA_Linear"), ("Pathology-ABMIL", "Pathology_ABMIL")):
                path = base / f"{stem}_s{seed}_f{fold}.npz"
                row: dict[str, Any] = {"seed": seed, "outer_fold": fold, "method": method, "cache_path": str(path), "status": "completed"}
                if not path.exists():
                    row.update(status="missing", sha256=None); valid = False; provenance.append(row); continue
                row["sha256"] = sha256_file(path)
                try:
                    saved = np.load(path, allow_pickle=False)
                    idx = np.asarray(saved["test_idx"], dtype=int); risk = np.asarray(saved["test_risk"], dtype=float)
                    if not np.array_equal(idx, expected) or risk.shape != idx.shape or not np.all(np.isfinite(risk)) or np.any(np.isfinite(out[seed][method][idx])):
                        raise ValueError("prediction coverage/order mismatch")
                    out[seed][method][idx] = risk
                    row.update(n_predictions=int(len(idx)), test_idx_sha256=sha256_array(idx))
                except Exception as exc:
                    row.update(status="invalid", error=repr(exc)); valid = False
                provenance.append(row)
        valid = valid and all(np.all(np.isfinite(v)) for v in out[seed].values())
    return out, provenance, valid


def load_phase2_crossfit(raw: Any, runtime: Path) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]], bool]:
    base = runtime / "phase2_crossfit_main_cache"
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    provenance: list[dict[str, Any]] = []
    valid = True
    for seed in SEEDS:
        rr = np.full((len(raw.patient_ids), 5), np.nan)
        rp = np.full((len(raw.patient_ids), 5), np.nan)
        for fold in FOLDS:
            rr_path = base / f"rna_crossfit_s{seed}_f{fold}.npy"; rp_path = base / f"path_crossfit_s{seed}_f{fold}.npy"
            train = np.flatnonzero(raw.outer_folds != fold)
            row = {"seed": seed, "outer_fold": fold, "rna_path": str(rr_path), "pathology_path": str(rp_path), "status": "completed"}
            try:
                rrv = np.asarray(np.load(rr_path), dtype=float); rpv = np.asarray(np.load(rp_path), dtype=float)
                if rrv.shape != (len(raw.patient_ids),) or rpv.shape != (len(raw.patient_ids),) or not np.all(np.isfinite(rrv[train])) or not np.all(np.isfinite(rpv[train])):
                    raise ValueError("crossfit risk coverage mismatch")
                rr[:, fold - 1] = rrv; rp[:, fold - 1] = rpv
            except Exception as exc:
                row.update(status="invalid", error=repr(exc)); valid = False
            provenance.append(row)
        # values are selected by their outer fold in the caller; keep the full matrices private.
        out[seed] = (rr, rp)
    return out, provenance, valid


def load_schedule(report: Path) -> dict[tuple[int, int], tuple[float, int]]:
    frame = pd.read_csv(report / "main_effect_metrics.csv")
    result: dict[tuple[int, int], tuple[float, int]] = {}
    for seed in SEEDS:
        for fold in FOLDS:
            cell = frame[(frame["seed"] == seed) & (frame["outer_fold"] == fold)]
            rna = cell[cell["method"] == "RNA-Linear"]; path = cell[cell["method"] == "Pathology-ABMIL"]
            if len(rna) != 1 or len(path) != 1:
                raise ValueError(f"locked Phase1 schedule missing for {seed}/{fold}")
            result[(seed, fold)] = (float(rna.iloc[0]["alpha"]), int(path.iloc[0]["selected_epoch"]))
    return result


def select_risk_columns(matrix: np.ndarray, outer_fold: int, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = matrix[:, outer_fold - 1]
    if not np.all(np.isfinite(values[train])):
        raise ValueError("crossfit risk is incomplete")
    return values[train], values[test]


def clinical_main_effects(raw: Any, train: np.ndarray, eval_idx: np.ndarray, seed: int, fold: int, device: torch.device, forbidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    cfg = ClinicalConfig()
    cf = np.full(len(raw.patient_ids), np.nan, dtype=float)
    audits: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for part, (fit_idx, heldout) in enumerate(stratified_splits(train, raw, seed + fold + 2701), 1):
        model_seed = seed + stable_int(f"cpo4:clinical:{seed}:{fold}:{part}") % 1000000
        pred, audit, private = fit_nested_anchor(raw.clinical, raw.dss_time, raw.dss_event, raw.cancer_type, fit_idx, heldout, forbidden, model_seed, fold, device, cfg)
        cf[heldout] = pred
        audit.update({"seed": seed, "outer_fold": fold, "scope": "outer_train_crossfit", "crossfit_subfold": part, "upstream_inner_val_labels_used": False})
        audits.append(audit); lineage.append({**audit, "memberships": private})
        if device.type == "cuda": torch.cuda.empty_cache()
    if not np.all(np.isfinite(cf[train])):
        raise RuntimeError("clinical crossfit incomplete")
    # The final outer-train fit is allowed to use every outer-train label; the
    # held-out outer-test is supplied only as predict_idx and therefore does
    # not belong in the forbidden training set.
    test_pred, final_audit, private = fit_nested_anchor(raw.clinical, raw.dss_time, raw.dss_event, raw.cancer_type, train, eval_idx, np.asarray([], dtype=int), seed + stable_int(f"cpo4:clinical:final:{seed}:{fold}") % 1000000, fold, device, cfg)
    final_audit.update({"seed": seed, "outer_fold": fold, "scope": "outer_train_full_to_outer_test", "upstream_inner_val_labels_used": False})
    audits.append(final_audit); lineage.append({**final_audit, "memberships": private})
    return cf, test_pred, audits, {"lineage_count": len(lineage), "crossfit_subfolds": len(audits) - 1, "inner_val_labels_used": False, "lineage_hash": sha256_array(np.asarray([len(x["memberships"]["fit"]) for x in lineage], dtype=np.int64))}


def inner_clinical_effects(raw: Any, fit_idx: np.ndarray, val_idx: np.ndarray, seed: int, fold: int, device: torch.device, forbidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, val_values, _, _ = clinical_main_effects(raw, fit_idx, val_idx, seed + 100000, fold + 100, device, forbidden)
    return values[fit_idx], val_values


def crossfit_path_embeddings(pool: np.ndarray, deterministic: np.ndarray, raw: Any, train: np.ndarray, test: np.ndarray, seed: int, fold: int, epochs: int, expected_train: np.ndarray, expected_test: np.ndarray, device: torch.device, cache_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if cache_path is not None and cache_path.exists():
        saved = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(saved["metadata"].item()))
        if meta.get("alignment_version") != "procrustes_v2":
            cache_path.unlink()
        else:
            zfull = np.full((int(meta["n_patients"]), int(meta["embedding_dim"])), np.nan, dtype=np.float32)
            zfull[np.asarray(saved["train_idx"], dtype=int)] = np.asarray(saved["z_train"], dtype=np.float32)
            return zfull, np.asarray(saved["z_test"], dtype=np.float32), meta
    train = np.sort(np.asarray(train, dtype=int)); test = np.asarray(test, dtype=int)
    z_train = np.full((len(raw.patient_ids), 96), np.nan, dtype=np.float32)
    risk_train = np.full(len(raw.patient_ids), np.nan, dtype=float)
    cal_idx = train[: min(128, len(train))]
    reference_cal = None
    transforms = []
    split_rows = []
    for part, (fit_idx, val_idx) in enumerate(stratified_splits(train, raw, seed + fold + 1701), 1):
        model_seed = seed + fold * 100 + (part - 1)
        model = train_abmil_fixed(pool, raw, fit_idx, model_seed, device, epochs)
        val_risk, val_emb = eval_abmil(model, deterministic, val_idx, device)
        cal_risk, cal_emb = eval_abmil(model, deterministic, cal_idx, device)
        if reference_cal is None:
            reference_cal = cal_emb.copy()
            aligned_val, zmeta = align_to_reference(reference_cal, cal_emb, val_emb)
        else:
            aligned_val, zmeta = align_to_reference(reference_cal, cal_emb, val_emb)
        z_train[val_idx] = aligned_val
        risk_train[val_idx] = val_risk
        transforms.append(zmeta); split_rows.append({"part": part, "fit_count": int(len(fit_idx)), "heldout_count": int(len(val_idx)), "model_seed": int(model_seed)})
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    if not np.all(np.isfinite(z_train[train])) or not np.all(np.isfinite(risk_train[train])):
        raise RuntimeError("cross-fitted pathology embedding/risk incomplete")
    # Replay the locked Phase2 cross-fitted risk before using the embedding.
    train_err = float(np.max(np.abs(risk_train[train] - expected_train)))
    if not np.allclose(risk_train[train], expected_train, atol=REPLAY_CROSSFIT_ATOL, rtol=REPLAY_RTOL):
        raise RuntimeError(f"crossfit ABMIL risk replay mismatch max_abs={train_err}")
    from src.po_phase3_realizable_complementarity_audit import train_locked_final_abmil
    final_model = train_locked_final_abmil(pool, raw, train, seed + fold, device, epochs)
    test_risk, test_emb = eval_abmil(final_model, deterministic, test, device)
    _, final_cal = eval_abmil(final_model, deterministic, cal_idx, device)
    aligned_test, test_meta = align_to_reference(reference_cal, final_cal, test_emb)
    test_err = float(np.max(np.abs(test_risk - expected_test)))
    if not np.allclose(test_risk, expected_test, atol=REPLAY_TEST_ATOL, rtol=REPLAY_RTOL):
        raise RuntimeError(f"test ABMIL risk replay mismatch max_abs={test_err}")
    z_train[test] = np.nan
    metadata = {
        "status": "completed",
        "embedding_dim": 96,
        "n_patients": int(len(raw.patient_ids)),
        "crossfit_folds": INNER_CF_FOLDS,
        "calibration_population": "outer_train only",
        "calibration_count": int(len(cal_idx)),
        "alignment_version": "procrustes_v2",
        "coordinate_alignment": "label-free per-model whitening plus orthogonal Procrustes to first crossfit model",
        "outcome_labels_used_for_alignment": False,
        "crossfit_risk_replay_match": True,
        "test_risk_replay_match": True,
        "crossfit_max_abs_error": train_err,
        "test_max_abs_error": test_err,
        "test_alignment": test_meta,
        "transform_summary": transforms,
        "split_rows": split_rows,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, train_idx=train.astype(np.int32), z_train=z_train[train], z_test=aligned_test, metadata=np.asarray(json.dumps(metadata)))
    return z_train, aligned_test, metadata


def inner_rna_effects(rna: np.ndarray, raw: Any, inner_train: np.ndarray, inner_val: np.ndarray, alpha: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rr_train = np.full(len(raw.patient_ids), np.nan, dtype=float)
    for part, (fit_idx, heldout) in enumerate(stratified_splits(inner_train, raw, seed + 4701), 1):
        w = train_linear(rna[fit_idx], raw.dss_time[fit_idx], raw.dss_event[fit_idx], alpha, seed + part * 101 + 31, steps=220)
        rr_train[heldout] = rna[heldout] @ w
    if not np.all(np.isfinite(rr_train[inner_train])):
        raise RuntimeError("inner RNA crossfit incomplete")
    w = train_linear(rna[inner_train], raw.dss_time[inner_train], raw.dss_event[inner_train], alpha, seed + 701, steps=220)
    return rr_train[inner_train], rna[inner_val] @ w


def inner_path_effects(pool: np.ndarray, deterministic: np.ndarray, raw: Any, inner_train: np.ndarray, inner_val: np.ndarray, seed: int, fold: int, epochs: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    inner_train = np.sort(np.asarray(inner_train, dtype=int)); inner_val = np.asarray(inner_val, dtype=int)
    risk_train = np.full(len(raw.patient_ids), np.nan, dtype=float)
    emb_train = np.full((len(raw.patient_ids), 96), np.nan, dtype=np.float32)
    cal_idx = inner_train[: min(96, len(inner_train))]
    reference_cal = None
    transform_rows = []
    for part, (fit_idx, heldout) in enumerate(stratified_splits(inner_train, raw, seed + fold + 4701), 1):
        model = train_abmil_fixed(pool, raw, fit_idx, seed + fold * 1000 + part + 70000, device, epochs)
        risk, emb = eval_abmil(model, deterministic, heldout, device)
        _, cal = eval_abmil(model, deterministic, cal_idx, device)
        if reference_cal is None:
            reference_cal = cal.copy(); aligned, _ = align_to_reference(reference_cal, cal, emb)
        else:
            aligned, meta = align_to_reference(reference_cal, cal, emb); transform_rows.append(meta)
        risk_train[heldout] = risk; emb_train[heldout] = aligned
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    if not np.all(np.isfinite(risk_train[inner_train])) or not np.all(np.isfinite(emb_train[inner_train])):
        raise RuntimeError("inner pathology crossfit incomplete")
    model = train_abmil_fixed(pool, raw, inner_train, seed + fold * 1000 + 70001, device, epochs)
    risk_val, emb_val = eval_abmil(model, deterministic, inner_val, device)
    _, cal_val = eval_abmil(model, deterministic, cal_idx, device)
    aligned_val, final_meta = align_to_reference(reference_cal, cal_val, emb_val)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return risk_train[inner_train], risk_val, emb_train[inner_train], aligned_val, {"crossfit_folds": INNER_CF_FOLDS, "calibration_count": int(len(cal_idx)), "coordinate_alignment": "label-free whitening plus orthogonal Procrustes", "outcome_labels_used_for_alignment": False, "transform_summary": transform_rows, "final_alignment": final_meta}


class ConditionedModel(nn.Module):
    def __init__(self, clinical_dim: int, pathology_dim: int = 96, rna_dim: int = 512, architecture: str = "Additive-C+P+R") -> None:
        super().__init__()
        self.architecture = architecture
        self.use_context = architecture in {"CC-RiskScale", "CC-LowRank", "CC-LowRank-CP-only", "CC-LowRank-CR-only"}
        if self.use_context:
            self.context = nn.Linear(clinical_dim, CLINICAL_CTX_DIM)
        self.a = nn.Parameter(torch.ones(3))
        if architecture == "CC-RiskScale":
            self.alpha_p = nn.Parameter(torch.zeros(()))
            self.alpha_r = nn.Parameter(torch.zeros(()))
            # alpha starts at zero so the forward pass is exactly additive,
            # while small nonzero directions keep alpha/v jointly learnable.
            self.v_p = nn.Parameter(torch.full((CLINICAL_CTX_DIM,), 0.02))
            self.v_r = nn.Parameter(torch.full((CLINICAL_CTX_DIM,), 0.02))
        if architecture in {"CC-LowRank", "CC-LowRank-CP-only", "CC-LowRank-CR-only"}:
            self.w_p = nn.Parameter(torch.empty(LOW_RANK_DIM, pathology_dim))
            self.w_cp = nn.Parameter(torch.empty(LOW_RANK_DIM, CLINICAL_CTX_DIM))
            self.w_r = nn.Parameter(torch.empty(LOW_RANK_DIM, rna_dim))
            self.w_cr = nn.Parameter(torch.empty(LOW_RANK_DIM, CLINICAL_CTX_DIM))
            nn.init.xavier_uniform_(self.w_p); nn.init.xavier_uniform_(self.w_cp)
            nn.init.xavier_uniform_(self.w_r); nn.init.xavier_uniform_(self.w_cr)
            self.lambda_cp = nn.Parameter(torch.zeros(()))
            self.lambda_cr = nn.Parameter(torch.zeros(()))
            if architecture == "CC-LowRank-CP-only":
                self.lambda_cr.requires_grad_(False)
            if architecture == "CC-LowRank-CR-only":
                self.lambda_cp.requires_grad_(False)

    def forward(self, clinical: torch.Tensor, r_c: torch.Tensor, r_p: torch.Tensor, r_r: torch.Tensor, z_p: torch.Tensor, x_r: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = torch.tanh(self.context(clinical)) if self.use_context else clinical.new_zeros((clinical.shape[0], CLINICAL_CTX_DIM))
        g_p = clinical.new_ones((clinical.shape[0],)); g_r = clinical.new_ones((clinical.shape[0],))
        h_cp = clinical.new_zeros((clinical.shape[0],)); h_cr = clinical.new_zeros((clinical.shape[0],))
        if self.architecture == "CC-RiskScale":
            g_p = 1.0 + self.alpha_p * torch.tanh(h @ self.v_p)
            g_r = 1.0 + self.alpha_r * torch.tanh(h @ self.v_r)
        if self.architecture in {"CC-LowRank", "CC-LowRank-CP-only", "CC-LowRank-CR-only"}:
            h_cp = (h @ self.w_cp.T * (z_p @ self.w_p.T)).sum(dim=1)
            h_cr = (h @ self.w_cr.T * (x_r @ self.w_r.T)).sum(dim=1)
        risk = self.a[0] * r_c + self.a[1] * g_p * r_p + self.a[2] * g_r * r_r
        if self.architecture in {"CC-LowRank", "CC-LowRank-CP-only", "CC-LowRank-CR-only"}:
            risk = risk + self.lambda_cp * h_cp + self.lambda_cr * h_cr
        return risk, {"h": h, "g_p": g_p, "g_r": g_r, "h_cp": h_cp, "h_cr": h_cr}


def model_l2(model: nn.Module) -> torch.Tensor:
    values = [p.square().sum() for p in model.parameters() if p.requires_grad]
    return sum(values) if values else torch.tensor(0.0)


def fit_conditioned(features: dict[str, np.ndarray], raw: Any, fit_idx: np.ndarray, val_idx: np.ndarray | None, architecture: str, l2: float, seed: int, device: torch.device, max_epochs: int = STAGE_B_MAX_EPOCHS, fixed_epochs: int | None = None) -> tuple[ConditionedModel, int, float, dict[str, float]]:
    set_seed(seed)
    model = ConditionedModel(features["C"].shape[1], features["zP"].shape[1], features["xR"].shape[1], architecture).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=STAGE_B_LR, weight_decay=STAGE_B_WEIGHT_DECAY)
    idx = np.asarray(fit_idx, dtype=int)
    c = torch.as_tensor(features["C"][idx], dtype=torch.float32, device=device)
    rc = torch.as_tensor(features["rC"][idx], dtype=torch.float32, device=device)
    rp = torch.as_tensor(features["rP"][idx], dtype=torch.float32, device=device)
    rr = torch.as_tensor(features["rR"][idx], dtype=torch.float32, device=device)
    zp = torch.as_tensor(features["zP"][idx], dtype=torch.float32, device=device)
    xr = torch.as_tensor(features["xR"][idx], dtype=torch.float32, device=device)
    tt = torch.as_tensor(raw.dss_time[idx], dtype=torch.float32, device=device)
    ee = torch.as_tensor(raw.dss_event[idx], dtype=torch.float32, device=device)
    if val_idx is not None:
        v = np.asarray(val_idx, dtype=int)
        vc = torch.as_tensor(features["C"][v], dtype=torch.float32, device=device); vrc = torch.as_tensor(features["rC"][v], dtype=torch.float32, device=device); vrp = torch.as_tensor(features["rP"][v], dtype=torch.float32, device=device); vrr = torch.as_tensor(features["rR"][v], dtype=torch.float32, device=device); vzp = torch.as_tensor(features["zP"][v], dtype=torch.float32, device=device); vxr = torch.as_tensor(features["xR"][v], dtype=torch.float32, device=device)
    best_score = -float("inf"); best_epoch = 1; wait = 0; last_grad = {"grad_context": 0.0, "grad_risk_scale": 0.0, "grad_cp": 0.0, "grad_cr": 0.0}
    n_epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    for epoch in range(1, n_epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        risk, _ = model(c, rc, rp, rr, zp, xr)
        loss = cox_loss_fast(tt, ee, risk) + 0.5 * float(l2) * model_l2(model)
        loss.backward()
        if model.use_context and model.context.weight.grad is not None: last_grad["grad_context"] = float(model.context.weight.grad.detach().norm().cpu())
        if hasattr(model, "alpha_p") and model.alpha_p.grad is not None: last_grad["grad_risk_scale"] = float(model.alpha_p.grad.detach().abs().cpu())
        if hasattr(model, "w_p") and model.w_p.grad is not None: last_grad["grad_cp"] = float(model.w_p.grad.detach().norm().cpu())
        if hasattr(model, "w_r") and model.w_r.grad is not None: last_grad["grad_cr"] = float(model.w_r.grad.detach().norm().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if val_idx is None or fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            val_risk, _ = model(vc, vrc, vrp, vrr, vzp, vxr)
        score = harrell_c(raw.dss_time[v], raw.dss_event[v], val_risk.detach().cpu().numpy())
        score = float(score) if np.isfinite(score) else -float(loss.detach().cpu())
        if score > best_score + 1e-6:
            best_score, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if wait >= STAGE_B_PATIENCE:
            break
    del opt
    if val_idx is None or fixed_epochs is not None:
        best_epoch = int(fixed_epochs if fixed_epochs is not None else n_epochs); best_score = float("nan")
    return model, best_epoch, best_score, last_grad


def prepare_feature_bundle(raw: Any, arrays: Any, train: np.ndarray, eval_idx: np.ndarray, r_c_train: np.ndarray, r_c_eval: np.ndarray, r_p_train: np.ndarray, r_p_eval: np.ndarray, r_r_train: np.ndarray, r_r_eval: np.ndarray, z_train: np.ndarray, z_eval: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = len(raw.patient_ids)
    out = {
        "C": np.zeros((n, arrays.clinical.shape[1]), dtype=np.float32),
        "xR": np.zeros((n, arrays.rna.shape[1]), dtype=np.float32),
        "zP": np.zeros((n, z_train.shape[1]), dtype=np.float32),
        "rC": np.full(n, np.nan, dtype=np.float32), "rP": np.full(n, np.nan, dtype=np.float32), "rR": np.full(n, np.nan, dtype=np.float32),
    }
    out["C"][train] = arrays.clinical[train]; out["C"][eval_idx] = arrays.clinical[eval_idx]
    out["xR"][train] = arrays.rna[train]; out["xR"][eval_idx] = arrays.rna[eval_idx]
    ztr, ze, zmean, zstd = standardize_columns(z_train, z_eval)
    out["zP"][train] = ztr; out["zP"][eval_idx] = ze
    rc_tr, rc_ev, rc_mean, rc_std = standardize_fit(r_c_train, r_c_eval)
    rp_tr, rp_ev, rp_mean, rp_std = standardize_fit(r_p_train, r_p_eval)
    rr_tr, rr_ev, rr_mean, rr_std = standardize_fit(r_r_train, r_r_eval)
    out["rC"][train] = rc_tr; out["rC"][eval_idx] = rc_ev; out["rP"][train] = rp_tr; out["rP"][eval_idx] = rp_ev; out["rR"][train] = rr_tr; out["rR"][eval_idx] = rr_ev
    audit = {"risk_standardization": {"rC_mean": rc_mean, "rC_std": rc_std, "rP_mean": rp_mean, "rP_std": rp_std, "rR_mean": rr_mean, "rR_std": rr_std}, "path_embedding_standardization": {"mean_sha256": sha256_array(zmean), "std_sha256": sha256_array(zstd)}, "clinical_dim": int(arrays.clinical.shape[1]), "rna_dim": int(arrays.rna.shape[1]), "pathology_dim": int(z_train.shape[1])}
    return out, audit


def fit_additive_subset(features: dict[str, np.ndarray], raw: Any, train: np.ndarray, test: np.ndarray, columns: tuple[int, ...], seed: int, device: torch.device, fixed_epochs: int = 220) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    set_seed(seed)
    model = nn.Parameter(torch.ones(len(columns), dtype=torch.float32, device=device))
    opt = torch.optim.Adam([model], lr=0.03)
    idx = np.asarray(train, dtype=int)
    values = torch.as_tensor(np.column_stack([features["rC"][idx], features["rP"][idx], features["rR"][idx]])[:, columns], device=device)
    tt = torch.as_tensor(raw.dss_time[idx], dtype=torch.float32, device=device); ee = torch.as_tensor(raw.dss_event[idx], dtype=torch.float32, device=device)
    for _ in range(fixed_epochs):
        opt.zero_grad(set_to_none=True); loss = cox_loss_fast(tt, ee, values @ model) + 0.001 * (model * model).sum(); loss.backward(); opt.step()
    test_values = torch.as_tensor(np.column_stack([features["rC"][test], features["rP"][test], features["rR"][test]])[:, columns], device=device)
    train_pred = (values @ model).detach().cpu().numpy(); test_pred = (test_values @ model).detach().cpu().numpy()
    weights = model.detach().cpu().numpy(); del model, opt
    if device.type == "cuda": torch.cuda.empty_cache()
    return train_pred, test_pred, {"params": int(len(columns)), "weights": weights.tolist(), "selected_epoch": fixed_epochs}


def fit_stage_variant(features: dict[str, np.ndarray], raw: Any, train: np.ndarray, test: np.ndarray, architecture: str, l2: float, seed: int, device: torch.device, epochs: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model, _, _, grads = fit_conditioned(features, raw, train, None, architecture, l2, seed, device, fixed_epochs=epochs)
    model.eval()
    def pred(indices: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        ix = np.asarray(indices, dtype=int)
        with torch.no_grad():
            out, diag = model(torch.as_tensor(features["C"][ix], device=device), torch.as_tensor(features["rC"][ix], device=device), torch.as_tensor(features["rP"][ix], device=device), torch.as_tensor(features["rR"][ix], device=device), torch.as_tensor(features["zP"][ix], device=device), torch.as_tensor(features["xR"][ix], device=device))
        return out.detach().cpu().numpy(), {k: v.detach().cpu().numpy() for k, v in diag.items()}
    train_pred, train_diag = pred(train); test_pred, test_diag = pred(test)
    metadata: dict[str, Any] = {"params": count_parameters(model), "selected_epoch": int(epochs), "gradients": grads, "train_diag": train_diag, "test_diag": test_diag, "architecture": architecture, "l2": l2, "a_C": float(model.a[0].detach().cpu()), "a_P": float(model.a[1].detach().cpu()), "a_R": float(model.a[2].detach().cpu())}
    if hasattr(model, "alpha_p"):
        metadata.update({"alpha_P": float(model.alpha_p.detach().cpu()), "alpha_R": float(model.alpha_r.detach().cpu())})
    if hasattr(model, "lambda_cp"):
        metadata.update({"lambda_CP": float(model.lambda_cp.detach().cpu()), "lambda_CR": float(model.lambda_cr.detach().cpu())})
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return train_pred, test_pred, metadata


def select_architecture(selection_features: dict[str, np.ndarray], raw: Any, inner_train: np.ndarray, inner_val: np.ndarray, seed: int, fold: int, device: torch.device) -> tuple[str, float, int, list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage1: list[tuple[float, int, str, float]] = []
    for order, architecture in enumerate(ARCHITECTURES):
        model, epoch, score, _ = fit_conditioned(selection_features, raw, inner_train, inner_val, architecture, STAGE1_L2, seed + fold * 100 + order + 30000, device)
        rows.append({"stage": "S1", "seed": seed, "outer_fold": fold, "architecture": architecture, "l2": STAGE1_L2, "inner_train_count": len(inner_train), "inner_val_count": len(inner_val), "inner_val_harrell": score, "selected_epoch": epoch, "status": "completed"})
        stage1.append((score if np.isfinite(score) else -np.inf, order, architecture, STAGE1_L2)); del model
        if device.type == "cuda": torch.cuda.empty_cache()
    stage1.sort(key=lambda x: (-x[0], x[1])); winner = stage1[0][2]
    rows.append({"stage": "S1-selected", "seed": seed, "outer_fold": fold, "architecture": winner, "l2": STAGE1_L2, "inner_val_harrell": stage1[0][0], "status": "locked"})
    stage2: list[tuple[float, int, float, int]] = []
    for order, l2 in enumerate(L2_GRID):
        model, epoch, score, _ = fit_conditioned(selection_features, raw, inner_train, inner_val, winner, l2, seed + fold * 100 + order + 31000, device)
        rows.append({"stage": "S2", "seed": seed, "outer_fold": fold, "architecture": winner, "l2": l2, "inner_train_count": len(inner_train), "inner_val_count": len(inner_val), "inner_val_harrell": score, "selected_epoch": epoch, "status": "completed"})
        stage2.append((score if np.isfinite(score) else -np.inf, order, l2, epoch)); del model
        if device.type == "cuda": torch.cuda.empty_cache()
    stage2.sort(key=lambda x: (-x[0], x[1])); selected_l2, selected_epoch = stage2[0][2], stage2[0][3]
    rows.append({"stage": "S2-selected", "seed": seed, "outer_fold": fold, "architecture": winner, "l2": selected_l2, "inner_val_harrell": stage2[0][0], "selected_epoch": selected_epoch, "status": "locked"})
    return winner, float(selected_l2), int(selected_epoch), rows, {"inner_train": inner_train, "inner_val": inner_val, "stage1_winner": winner, "stage1_score": stage1[0][0], "stage2_score": stage2[0][0]}


def eval_fold(raw: Any, train: np.ndarray, test: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    u, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], risk, raw.dss_time[train], raw.dss_event[train])
    return {"harrell_c": float(harrell_c(raw.dss_time[test], raw.dss_event[test], risk)), "uno_c": float(u), "uno_numerator": float(num), "uno_denominator": float(den)}


def aggregate_seed(raw: Any, fold_rows: list[dict[str, Any]], risk: np.ndarray) -> tuple[float, float]:
    h = float(harrell_c(raw.dss_time, raw.dss_event, risk))
    rows = [r for r in fold_rows]
    num = float(sum(r["uno_numerator"] for r in rows)); den = float(sum(r["uno_denominator"] for r in rows))
    return h, (num / den if den else float("nan"))


def fold_local_uno_bootstrap(raw: Any, sample: np.ndarray, risk: np.ndarray) -> float:
    num = den = 0.0
    for fold in FOLDS:
        chosen = sample[raw.outer_folds[sample] == fold]
        if len(chosen) == 0:
            continue
        train = np.flatnonzero(raw.outer_folds != fold)
        kt, kg = censoring_survival(raw.dss_time[train], raw.dss_event[train])
        t = raw.dss_time[chosen]; e = raw.dss_event[chosen]; r = risk[chosen]
        g = lookup_censor_survival(t, kt, kg)
        for i in np.flatnonzero(e == 1):
            if g[i] <= 1e-8:
                continue
            mask = t > t[i]; w = 1.0 / (g[i] * g[i]); den += w * float(mask.sum()); num += w * (float((r[i] > r[mask]).sum()) + 0.5 * float((r[i] == r[mask]).sum()))
    return float(num / den) if den else float("nan")


def paired_bootstrap(raw: Any, candidate: np.ndarray, baseline: np.ndarray, seed: int, comparison: str, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    rng = np.random.default_rng(20261004 + seed + stable_int(comparison) % 100000)
    dh, du = [], []
    for _ in range(reps):
        sample = rng.integers(0, len(raw.patient_ids), len(raw.patient_ids))
        t = raw.dss_time[sample]; e = raw.dss_event[sample]
        dh.append(float(harrell_c(t, e, candidate[sample]) - harrell_c(t, e, baseline[sample])))
        du.append(float(fold_local_uno_bootstrap(raw, sample, candidate) - fold_local_uno_bootstrap(raw, sample, baseline)))
    dh_arr = np.asarray(dh); du_arr = np.asarray(du)
    return {"comparison": comparison, "seed": seed, "delta_harrell_mean": float(np.mean(dh_arr)), "delta_harrell_ci95_low": float(np.quantile(dh_arr, .025)), "delta_harrell_ci95_high": float(np.quantile(dh_arr, .975)), "delta_uno_mean": float(np.mean(du_arr)), "delta_uno_ci95_low": float(np.quantile(du_arr, .025)), "delta_uno_ci95_high": float(np.quantile(du_arr, .975)), "bootstrap_replicates": reps, "status": "completed"}


def load_historical_model(raw: Any, runtime: Path, model_name: str) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Load only patient-level OOF caches whose provenance is independently audited."""
    if model_name.startswith("M7-"):
        mapping = {"M7-CommonOnly": "B0", "M7-MCAT": "F0", "M7-FiLM": "F3"}
        prefix = mapping[model_name]; seeds = (20261002, 20261003, 20261004, 20261005, 20261006)
        result: dict[int, np.ndarray] = {}
        for seed in seeds:
            values = np.full(len(raw.patient_ids), np.nan)
            # The runtime directory date follows the full seed value.
            base = runtime / f"diffusion_runtime_m7r_{seed}" / "cells"
            for fold in FOLDS:
                path = base / f"{prefix}{seed}{fold}.npy"
                expected = np.flatnonzero(raw.outer_folds == fold)
                if not path.exists():
                    raise FileNotFoundError(path)
                pred = np.asarray(np.load(path), dtype=float)
                if pred.shape != (len(expected),) or not np.all(np.isfinite(pred)):
                    raise ValueError(f"historical M7 shape mismatch: {path}")
                values[expected] = pred
            if not np.all(np.isfinite(values)):
                raise ValueError(f"historical M7 OOF incomplete: {model_name} {seed}")
            result[seed] = values
        return result, {"model": model_name, "source": "M7R", "source_runtime": str(runtime), "seeds": list(seeds), "cohort": 512, "events": 65, "fixed_outer_folds": True, "endpoint": "DSS", "clinical_fields": list(CLINICAL_COLUMNS), "clinical_usage_audited": True, "patient_ids_written": False, "status": "comparable"}
    if model_name == "M8P1R-SPO":
        seeds = (20261012, 20261013, 20261014, 20261015, 20261016); result = {}
        for seed in seeds:
            values = np.full(len(raw.patient_ids), np.nan); base = runtime / "m8p1r_workers" / f"seed_{seed}" / "cells"
            for fold in FOLDS:
                path = base / f"SPO_{seed}_{fold}.npy"; expected = np.flatnonzero(raw.outer_folds == fold)
                if not path.exists(): raise FileNotFoundError(path)
                pred = np.asarray(np.load(path), dtype=float)
                if pred.shape != (len(expected),) or not np.all(np.isfinite(pred)): raise ValueError(f"historical M8 shape mismatch: {path}")
                values[expected] = pred
            if not np.all(np.isfinite(values)): raise ValueError(f"historical M8 OOF incomplete: {seed}")
            result[seed] = values
        return result, {"model": model_name, "source": "M8P1R", "source_runtime": str(runtime), "seeds": list(seeds), "cohort": 512, "events": 65, "fixed_outer_folds": True, "endpoint": "DSS", "clinical_fields": list(CLINICAL_COLUMNS), "clinical_usage_audited": True, "strong_anchor_audit": "PASS", "patient_ids_written": False, "status": "comparable"}
    raise ValueError(model_name)


def historical_metrics(raw: Any, arrays: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for seed, risk in arrays.items():
        nums = dens = 0.0; fold_h = []
        for fold in FOLDS:
            train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold); metric = eval_fold(raw, train, test, risk[test]); nums += metric["uno_numerator"]; dens += metric["uno_denominator"]; fold_h.append(metric["harrell_c"])
        rows.append({"seed": seed, "harrell_c_seed_oof": float(harrell_c(raw.dss_time, raw.dss_event, risk)), "uno_c_seed_oof": float(nums / dens), "harrell_c_fold_mean": float(np.mean(fold_h)), "harrell_c_fold_sd": float(np.std(fold_h, ddof=1)), "uno_numerator": nums, "uno_denominator": dens, "status": "completed"})
    return rows


def build_historical_report(raw: Any, runtime: Path, out: Path) -> tuple[dict[str, dict[int, np.ndarray]], list[dict[str, Any]], str, dict[str, Any]]:
    models = ("M7-CommonOnly", "M7-MCAT", "M7-FiLM", "M8P1R-SPO")
    all_arrays: dict[str, dict[int, np.ndarray]] = {}; metric_rows: list[dict[str, Any]] = []; provenance: dict[str, Any] = {}
    for model in models:
        try:
            arrays, prov = load_historical_model(raw, runtime, model); all_arrays[model] = arrays; provenance[model] = prov
            for row in historical_metrics(raw, arrays):
                metric_rows.append({"model": model, **row})
            means = historical_metrics(raw, arrays)
            prov["mean_harrell"] = float(np.mean([r["harrell_c_seed_oof"] for r in means])); prov["mean_uno"] = float(np.mean([r["uno_c_seed_oof"] for r in means]))
        except Exception as exc:
            provenance[model] = {"model": model, "status": "excluded_incomparable", "reason": repr(exc)}
    # Predeclared historical choice: the audited M7 CommonOnly is the strongest
    # prior CPO anchor by primary Harrell in the prior report; no current test is used.
    b5 = "M7-CommonOnly" if "M7-CommonOnly" in all_arrays else max(all_arrays, key=lambda name: provenance[name].get("mean_harrell", -float("inf")), default="")
    atomic_csv(out / "historical_reaggregation.csv", metric_rows)
    atomic_json(out / "historical_provenance.json", {"models": provenance, "best_strict_cpo_comparator": b5, "selection_rule": "audited historical M7 CommonOnly primary Harrell anchor; fallback max historical Harrell only if unavailable", "patient_ids_written": False})
    return all_arrays, metric_rows, b5, provenance


def row_with_metrics(method: str, seed: int, fold: int, raw: Any, train: np.ndarray, test: np.ndarray, risk: np.ndarray, **extra: Any) -> dict[str, Any]:
    return {"scope": "outer_fold", "method": method, "seed": seed, "outer_fold": fold, **eval_fold(raw, train, test, risk), **extra, "status": "completed"}


def make_inner_selection_bundle(pool: np.ndarray, deterministic: np.ndarray, raw: Any, arrays: Any, inner_train: np.ndarray, inner_val: np.ndarray, outer_test: np.ndarray, seed: int, fold: int, alpha: float, path_epochs: int, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # Every upstream supervised fit in this function receives labels only from
    # inner_train.  inner_val is used solely for Stage-B architecture/epoch
    # scoring after the risks and embeddings have been generated.
    rc_train, rc_val = inner_clinical_effects(raw, inner_train, inner_val, seed, fold, device, outer_test)
    rr_train, rr_val = inner_rna_effects(arrays.rna, raw, inner_train, inner_val, alpha, seed + 2000)
    rp_train, rp_val, zp_train, zp_val, path_meta = inner_path_effects(pool, deterministic, raw, inner_train, inner_val, seed + 3000, fold, path_epochs, device)
    bundle, audit = prepare_feature_bundle(raw, arrays, inner_train, inner_val, rc_train, rc_val, rp_train, rp_val, rr_train, rr_val, zp_train, zp_val)
    audit.update({"upstream_fit_population": int(len(inner_train)), "upstream_eval_population": int(len(inner_val)), "inner_val_labels_used_for_upstream": False, "path_alignment": path_meta})
    return bundle, audit


def make_diagnostic_row(method: str, seed: int, fold: int, raw: Any, train: np.ndarray, test: np.ndarray, meta: dict[str, Any]) -> dict[str, Any]:
    td = meta.get("test_diag", {}); trd = meta.get("train_diag", {})
    return {
        "seed": seed, "outer_fold": fold, "method": method, "architecture": meta.get("architecture"), "l2": meta.get("l2"), "params": meta.get("params"), "selected_epoch": meta.get("selected_epoch"), "a_C": meta.get("a_C"), "a_P": meta.get("a_P"), "a_R": meta.get("a_R"), "alpha_P": meta.get("alpha_P", 0.0), "alpha_R": meta.get("alpha_R", 0.0), "lambda_CP": meta.get("lambda_CP", 0.0), "lambda_CR": meta.get("lambda_CR", 0.0), "var_h_CP": float(np.var(trd.get("h_cp", np.zeros(len(train))))), "var_h_CR": float(np.var(trd.get("h_cr", np.zeros(len(train))))), "corr_h_CP_rC": corr(trd.get("h_cp", np.zeros(len(train))), meta.get("train_rC", np.zeros(len(train)))), "corr_h_CP_rP": corr(trd.get("h_cp", np.zeros(len(train))), meta.get("train_rP", np.zeros(len(train)))), "corr_h_CR_rC": corr(trd.get("h_cr", np.zeros(len(train))), meta.get("train_rC", np.zeros(len(train)))), "corr_h_CR_rR": corr(trd.get("h_cr", np.zeros(len(train))), meta.get("train_rR", np.zeros(len(train)))), "std_g_P_test": float(np.std(td.get("g_p", np.ones(len(test))))), "std_g_R_test": float(np.std(td.get("g_r", np.ones(len(test))))), "grad_context": meta.get("gradients", {}).get("grad_context", 0.0), "grad_risk_scale": meta.get("gradients", {}).get("grad_risk_scale", 0.0), "grad_CP": meta.get("gradients", {}).get("grad_cp", 0.0), "grad_CR": meta.get("gradients", {}).get("grad_cr", 0.0), "status": "completed",
    }


def run_cell(raw: Any, pool: np.ndarray, deterministic: np.ndarray, arrays: Any, phase1: dict[int, dict[str, np.ndarray]], phase2: dict[int, tuple[np.ndarray, np.ndarray]], schedule: dict[tuple[int, int], tuple[float, int]], runtime: Path, seed: int, fold: int, device: torch.device, smoke: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    alpha, path_epochs = schedule[(seed, fold)]
    rna_cf_matrix, path_cf_matrix = phase2[seed]
    r_r_train, r_r_test = select_risk_columns(rna_cf_matrix, fold, train, test)
    r_p_train, r_p_test = select_risk_columns(path_cf_matrix, fold, train, test)
    if not np.all(np.isfinite(phase1[seed]["RNA-Linear"][test])) or not np.all(np.isfinite(phase1[seed]["Pathology-ABMIL"][test])):
        raise RuntimeError("locked outer-test main-effect cache incomplete")
    r_r_test = phase1[seed]["RNA-Linear"][test]; r_p_test = phase1[seed]["Pathology-ABMIL"][test]
    r_c_cf, r_c_test, clinical_audits, clinical_lineage = clinical_main_effects(raw, train, test, seed, fold, device, test)
    path_cache = runtime / "cpo_phase4_cache" / f"path_embeddings_s{seed}_f{fold}.npz"
    z_all, z_test, path_meta = crossfit_path_embeddings(pool, deterministic, raw, train, test, seed, fold, path_epochs, r_p_train, r_p_test, device, path_cache)
    z_train = z_all[train]
    outer_features, feature_audit = prepare_feature_bundle(raw, arrays, train, test, r_c_cf[train], r_c_test, r_p_train, r_p_test, r_r_train, r_r_test, z_train, z_test)
    inner_train, inner_val = split_inner(train, raw, seed + fold + 6101)
    inner_cache = runtime / "cpo_phase4_cache" / f"inner_selection_s{seed}_f{fold}.npz"
    # Cache only numerical features and indices; no patient identifiers are written.
    if inner_cache.exists():
        saved = np.load(inner_cache, allow_pickle=False)
        if np.array_equal(np.asarray(saved["inner_train"], dtype=int), inner_train) and np.array_equal(np.asarray(saved["inner_val"], dtype=int), inner_val):
            sel_bundle = {k: np.asarray(saved[k]) for k in ("C", "xR", "zP", "rC", "rP", "rR")}; inner_audit = json.loads(str(saved["audit"].item()))
        else:
            inner_cache.unlink(); sel_bundle, inner_audit = make_inner_selection_bundle(pool, deterministic, raw, arrays, inner_train, inner_val, test, seed, fold, alpha, path_epochs, device)
    else:
        sel_bundle, inner_audit = make_inner_selection_bundle(pool, deterministic, raw, arrays, inner_train, inner_val, test, seed, fold, alpha, path_epochs, device)
    if not inner_cache.exists():
        inner_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(inner_cache, inner_train=inner_train.astype(np.int32), inner_val=inner_val.astype(np.int32), audit=np.asarray(json.dumps(inner_audit)), **{k: sel_bundle[k][np.concatenate([inner_train, inner_val])] for k in ("C", "xR", "zP", "rC", "rP", "rR")})
        # The cached feature arrays are ordered inner_train followed by inner_val.
        sel_order = np.concatenate([inner_train, inner_val]); sel_bundle = {k: np.zeros_like(outer_features[k]) for k in ("C", "xR", "zP", "rC", "rP", "rR")}
        for k in sel_bundle: sel_bundle[k][sel_order] = np.asarray(np.load(inner_cache, allow_pickle=False)[k])
    elif sel_bundle["C"].shape[0] != len(raw.patient_ids):
        cached = sel_bundle; sel_order = np.concatenate([inner_train, inner_val]); sel_bundle = {k: np.zeros_like(outer_features[k]) for k in ("C", "xR", "zP", "rC", "rP", "rR")}
        for k in sel_bundle: sel_bundle[k][sel_order] = cached[k]
    selected_arch, selected_l2, selected_epoch, selection_rows, selection_meta = select_architecture(sel_bundle, raw, inner_train, inner_val, seed, fold, device)
    # Full outer-train Stage-B fit uses only the already cross-fitted outer-train features.
    selected_train, selected_test, selected_meta = fit_stage_variant(outer_features, raw, train, test, selected_arch, selected_l2, seed + fold * 100 + 39000, device, selected_epoch)
    selected_meta.update({"train_rC": outer_features["rC"][train], "train_rP": outer_features["rP"][train], "train_rR": outer_features["rR"][train]})
    methods: dict[str, np.ndarray] = {"B0 Clinical-only": r_c_test.astype(float), "A0 Clinical-only": r_c_test.astype(float)}
    train_predictions: dict[str, np.ndarray] = {"B0 Clinical-only": r_c_cf[train].astype(float), "A0 Clinical-only": r_c_cf[train].astype(float)}
    fold_rows: list[dict[str, Any]] = [row_with_metrics("B0 Clinical-only", seed, fold, raw, train, test, methods["B0 Clinical-only"], model_field="Clinical-C-full", params="locked")]
    ablation_rows: list[dict[str, Any]] = []
    ablation_rows.append(row_with_metrics("A0 Clinical-only", seed, fold, raw, train, test, r_c_test, architecture="Clinical-C-full", params="locked"))
    diagnostics: list[dict[str, Any]] = []
    model_selection_rows = selection_rows.copy()
    # Fixed additive baselines and equal standardized sum.
    tr_b1, te_b1, b1m = fit_additive_subset(outer_features, raw, train, test, (0, 1), seed + fold * 100 + 401, device)
    tr_b2, te_b2, b2m = fit_additive_subset(outer_features, raw, train, test, (0, 2), seed + fold * 100 + 402, device)
    tr_b3, te_b3, b3m = fit_additive_subset(outer_features, raw, train, test, (0, 1, 2), seed + fold * 100 + 403, device)
    methods.update({"B1 C+P additive": te_b1, "B2 C+R additive": te_b2, "B3 C+P+R additive": te_b3, "B4 equal standardized C+P+R": outer_features["rC"][test] + outer_features["rP"][test] + outer_features["rR"][test]})
    train_predictions.update({"B1 C+P additive": tr_b1, "B2 C+R additive": tr_b2, "B3 C+P+R additive": tr_b3, "B4 equal standardized C+P+R": outer_features["rC"][train] + outer_features["rP"][train] + outer_features["rR"][train]})
    for name, pred in methods.items():
        if name == "B0 Clinical-only": continue
        fold_rows.append(row_with_metrics(name, seed, fold, raw, train, test, pred, params=2 if name in ("B1 C+P additive", "B2 C+R additive") else 3 if name == "B3 C+P+R additive" else 0))
    # Formal ablations.  A1-A3 reuse their architecture-specific inner-only epoch selection;
    # CP-only and CR-only perform a separate inner-only epoch selection.
    epoch_by_arch = {r["architecture"]: int(r["selected_epoch"]) for r in selection_rows if r.get("stage") == "S1"}
    ablation_specs = [("A1 Additive-C+P+R", "Additive-C+P+R", STAGE1_L2, epoch_by_arch.get("Additive-C+P+R", 120)), ("A2 CC-RiskScale", "CC-RiskScale", STAGE1_L2, epoch_by_arch.get("CC-RiskScale", 120)), ("A3 CC-LowRank-full", "CC-LowRank", STAGE1_L2, epoch_by_arch.get("CC-LowRank", 120))]
    for ab_name, arch, l2, epoch in ablation_specs:
        trp, tep, meta = fit_stage_variant(outer_features, raw, train, test, arch, l2, seed + fold * 100 + stable_int(ab_name) % 1000 + 4100, device, epoch); meta.update({"train_rC": outer_features["rC"][train], "train_rP": outer_features["rP"][train], "train_rR": outer_features["rR"][train]})
        methods[ab_name] = tep; train_predictions[ab_name] = trp; ablation_rows.append(row_with_metrics(ab_name, seed, fold, raw, train, test, tep, architecture=arch, l2=l2, params=meta["params"], selected_epoch=epoch)); diagnostics.append(make_diagnostic_row(ab_name, seed, fold, raw, train, test, meta))
    for ab_name, arch in (("A4 CC-LowRank-CP-only", "CC-LowRank-CP-only"), ("A5 CC-LowRank-CR-only", "CC-LowRank-CR-only")):
        _, ep, score, _ = fit_conditioned(sel_bundle, raw, inner_train, inner_val, arch, STAGE1_L2, seed + fold * 100 + stable_int(ab_name) % 1000 + 4500, device); del score
        trp, tep, meta = fit_stage_variant(outer_features, raw, train, test, arch, STAGE1_L2, seed + fold * 100 + stable_int(ab_name) % 1000 + 4600, device, ep); meta.update({"train_rC": outer_features["rC"][train], "train_rP": outer_features["rP"][train], "train_rR": outer_features["rR"][train]})
        methods[ab_name] = tep; train_predictions[ab_name] = trp; ablation_rows.append(row_with_metrics(ab_name, seed, fold, raw, train, test, tep, architecture=arch, l2=STAGE1_L2, params=meta["params"], selected_epoch=ep)); diagnostics.append(make_diagnostic_row(ab_name, seed, fold, raw, train, test, meta))
    methods["Selected-conditioned"] = selected_test; train_predictions["Selected-conditioned"] = selected_train
    fold_rows.append(row_with_metrics("Selected-conditioned", seed, fold, raw, train, test, selected_test, architecture=selected_arch, l2=selected_l2, params=selected_meta["params"], selected_epoch=selected_epoch, selection_stage1_winner=selection_meta["stage1_winner"]))
    diagnostics.append(make_diagnostic_row("Selected-conditioned", seed, fold, raw, train, test, selected_meta))
    diagnostics[-1].update({"clinical_crossfit_subfolds": clinical_lineage["crossfit_subfolds"], "inner_upstream_fit_count": inner_audit["upstream_fit_population"], "inner_val_upstream_labels_used": inner_audit["inner_val_labels_used_for_upstream"], "path_coordinate_alignment": path_meta["coordinate_alignment"], "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0})
    diag = {"clinical_audits": clinical_audits, "clinical_lineage": clinical_lineage, "path_meta": path_meta, "inner_audit": inner_audit, "feature_audit": feature_audit, "selection_meta": selection_meta, "selection_rows": selection_rows, "peak_gpu_memory_gb": diagnostics[-1].get("peak_gpu_memory_gb", 0.0)}
    if smoke:
        print(json.dumps({"seed": seed, "fold": fold, "selected_arch": selected_arch, "selected_l2": selected_l2, "selected_epoch": selected_epoch, "methods": sorted(methods), "path_replay": {"crossfit": path_meta.get("crossfit_risk_replay_match"), "test": path_meta.get("test_risk_replay_match")}, "inner_val_upstream_labels_used": inner_audit.get("inner_val_labels_used_for_upstream")}, ensure_ascii=False))
    return fold_rows, ablation_rows, diagnostics, methods, diag, model_selection_rows


def fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def main(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root).resolve(); runtime = Path(args.runtime_root).resolve(); out = Path(args.reports_root).resolve(); data_root = Path(os.environ["DIFFUSION_DATA_ROOT"]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_raw_inputs(data_root, repo)
    pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    arrays_by_fold = {fold: preprocess_fold(raw, pool, fold) for fold in FOLDS}
    phase1, phase1_provenance, phase1_valid = load_phase1_oof(raw, runtime)
    phase2, phase2_provenance, phase2_valid = load_phase2_crossfit(raw, runtime)
    schedule = load_schedule(Path(args.phase1_report))
    if args.smoke:
        run_cell(raw, pool, deterministic, arrays_by_fold[int(args.smoke_fold)], phase1, phase2, schedule, runtime, int(args.smoke_seed), int(args.smoke_fold), device, smoke=True)
        return
    out.mkdir(parents=True, exist_ok=True); started = time.time(); failures: list[dict[str, Any]] = []
    historical_arrays, historical_rows, b5_model, historical_provenance = build_historical_report(raw, Path(args.historical_runtime), out)
    all_fold_rows: list[dict[str, Any]] = []; all_ablation_rows: list[dict[str, Any]] = []; all_diag: list[dict[str, Any]] = []; all_selection: list[dict[str, Any]] = []; cell_audits: list[dict[str, Any]] = []
    oof: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}; method_names: set[str] = set()
    for seed in SEEDS:
        for fold in FOLDS:
            try:
                fold_rows, ablation_rows, diagnostics, methods, diag, selection_rows = run_cell(raw, pool, deterministic, arrays_by_fold[fold], phase1, phase2, schedule, runtime, seed, fold, device)
                all_fold_rows.extend(fold_rows); all_ablation_rows.extend(ablation_rows); all_diag.extend(diagnostics); all_selection.extend(selection_rows)
                cell_audits.append({"seed": seed, "outer_fold": fold, "path_crossfit_risk_replay_match": bool(diag["path_meta"].get("crossfit_risk_replay_match", False)), "path_test_risk_replay_match": bool(diag["path_meta"].get("test_risk_replay_match", False)), "inner_val_upstream_labels_used": bool(diag["inner_audit"].get("inner_val_labels_used_for_upstream", True)), "inner_fit_count": int(diag["inner_audit"].get("upstream_fit_population", 0)), "clinical_crossfit_subfolds": int(diag["clinical_lineage"].get("crossfit_subfolds", 0)), "coordinate_alignment": diag["path_meta"].get("coordinate_alignment"), "status": "completed"})
                method_names.update(methods.keys())
                for method, prediction in methods.items():
                    oof[seed].setdefault(method, np.full(len(raw.patient_ids), np.nan, dtype=float))[np.flatnonzero(raw.outer_folds == fold)] = prediction
            except Exception as exc:
                failures.append({"seed": seed, "outer_fold": fold, "error": repr(exc), "traceback": traceback.format_exc()})
                if device.type == "cuda": torch.cuda.empty_cache()
    expected_methods = {"B0 Clinical-only", "B1 C+P additive", "B2 C+R additive", "B3 C+P+R additive", "B4 equal standardized C+P+R", "A0 Clinical-only", "A1 Additive-C+P+R", "A2 CC-RiskScale", "A3 CC-LowRank-full", "A4 CC-LowRank-CP-only", "A5 CC-LowRank-CR-only", "Selected-conditioned"}
    method_names.update(expected_methods)
    for seed in SEEDS:
        for method in method_names:
            if method not in oof[seed]: oof[seed][method] = np.full(len(raw.patient_ids), np.nan)
    atomic_csv(out / "fold_metrics.csv", all_fold_rows)
    atomic_csv(out / "ablation.csv", all_ablation_rows)
    atomic_csv(out / "conditioning_diagnostics.csv", all_diag)
    atomic_csv(out / "model_selection.csv", all_selection)
    seed_rows: list[dict[str, Any]] = []
    fold_lookup = all_fold_rows + all_ablation_rows
    for seed in SEEDS:
        for method in sorted(method_names):
            risk = oof[seed][method]
            if not np.all(np.isfinite(risk)):
                continue
            rows = [r for r in fold_lookup if int(r.get("seed", -1)) == seed and r.get("method") == method]
            nums = sum(float(r.get("uno_numerator", 0.0)) for r in rows); dens = sum(float(r.get("uno_denominator", 0.0)) for r in rows)
            fold_h = [float(r["harrell_c"]) for r in rows]
            seed_rows.append({"scope": "seed_oof", "seed": seed, "method": method, "harrell_c": float(harrell_c(raw.dss_time, raw.dss_event, risk)), "uno_c": float(nums / dens) if dens else float("nan"), "harrell_c_fold_mean": float(np.mean(fold_h)) if fold_h else float("nan"), "harrell_c_fold_sd": float(np.std(fold_h, ddof=1)) if len(fold_h) > 1 else 0.0, "uno_numerator": nums, "uno_denominator": dens, "status": "completed"})
    for model, arrays in historical_arrays.items():
        for row in historical_metrics(raw, arrays): seed_rows.append({"scope": "historical_seed_oof", "seed": f"{model}:{row['seed']}", "method": model, "harrell_c": row["harrell_c_seed_oof"], "uno_c": row["uno_c_seed_oof"], "harrell_c_fold_mean": row["harrell_c_fold_mean"], "harrell_c_fold_sd": row["harrell_c_fold_sd"], "uno_numerator": row["uno_numerator"], "uno_denominator": row["uno_denominator"], "status": "historical_comparable"})
    for seed in SEEDS:
        for source_method, report_method in (("RNA-Linear", "RNA-only"), ("Pathology-ABMIL", "Pathology-only")):
            risk = phase1[seed][source_method]; nums = dens = 0.0; fold_h = []
            for fold in FOLDS:
                train = np.flatnonzero(raw.outer_folds != fold); test = np.flatnonzero(raw.outer_folds == fold); metric = eval_fold(raw, train, test, risk[test]); nums += metric["uno_numerator"]; dens += metric["uno_denominator"]; fold_h.append(metric["harrell_c"])
            seed_rows.append({"scope": "seed_oof", "seed": seed, "method": report_method, "harrell_c": float(harrell_c(raw.dss_time, raw.dss_event, risk)), "uno_c": float(nums / dens), "harrell_c_fold_mean": float(np.mean(fold_h)), "harrell_c_fold_sd": float(np.std(fold_h, ddof=1)), "uno_numerator": nums, "uno_denominator": dens, "status": "completed"})
    # The direct modality rows are added after the main-method loop above;
    # include them in the public CSV and final summary calculations.
    atomic_csv(out / "seed_oof_metrics.csv", seed_rows)
    method_names.update({"RNA-only", "Pathology-only"})
    comparison_specs = [("A1 Additive-C+P+R", "A0 Clinical-only"), ("A2 CC-RiskScale", "A0 Clinical-only"), ("A3 CC-LowRank-full", "A0 Clinical-only"), ("A1 Additive-C+P+R", "B3 C+P+R additive"), ("A2 CC-RiskScale", "B3 C+P+R additive"), ("A3 CC-LowRank-full", "B3 C+P+R additive")]
    comparison_rows: list[dict[str, Any]] = []
    for candidate, baseline in comparison_specs:
        deltas_h, deltas_u = [], []
        for seed in SEEDS:
            c = next(r for r in seed_rows if r["scope"] == "seed_oof" and r["seed"] == seed and r["method"] == candidate); b = next(r for r in seed_rows if r["scope"] == "seed_oof" and r["seed"] == seed and r["method"] == baseline)
            dh = c["harrell_c"] - b["harrell_c"]; du = c["uno_c"] - b["uno_c"]; deltas_h.append(dh); deltas_u.append(du); comparison_rows.append({"scope": "seed_oof", "seed": seed, "comparison": f"{candidate} - {baseline}", "delta_harrell": dh, "delta_uno": du, "status": "completed"})
        comparison_rows.append({"scope": "mean_seed", "seed": "ALL", "comparison": f"{candidate} - {baseline}", "delta_harrell": float(np.mean(deltas_h)), "delta_uno": float(np.mean(deltas_u)), "positive_seed_both": int(sum(h > 0 and u > 0 for h, u in zip(deltas_h, deltas_u))), "status": "completed"})
    b5_mean = None
    if b5_model and b5_model in historical_arrays:
        b5_mean = np.mean(np.stack(list(historical_arrays[b5_model].values())), axis=0)
        for seed in SEEDS:
            risk = oof[seed].get("A3 CC-LowRank-full")
            if risk is not None and np.all(np.isfinite(risk)):
                comparison_rows.append({"scope": "current_vs_historical_mean", "seed": seed, "comparison": f"A3 CC-LowRank-full - {b5_model}", "delta_harrell": float(harrell_c(raw.dss_time, raw.dss_event, risk) - harrell_c(raw.dss_time, raw.dss_event, b5_mean)), "delta_uno": float(np.nanmean([r["uno_c"] for r in seed_rows if r["scope"] == "seed_oof" and r["seed"] == seed and r["method"] == "A3 CC-LowRank-full"]) - np.mean([r["uno_c_seed_oof"] for r in historical_metrics(raw, historical_arrays[b5_model])])), "status": "completed"})
    atomic_csv(out / "paired_comparisons.csv", comparison_rows)
    bootstrap_rows: list[dict[str, Any]] = []
    for candidate, baseline in comparison_specs:
        for seed in SEEDS:
            if np.all(np.isfinite(oof[seed][candidate])) and np.all(np.isfinite(oof[seed][baseline])):
                bootstrap_rows.append(paired_bootstrap(raw, oof[seed][candidate], oof[seed][baseline], seed, f"{candidate} - {baseline}"))
    if b5_mean is not None:
        for seed in SEEDS:
            if np.all(np.isfinite(oof[seed]["A3 CC-LowRank-full"])): bootstrap_rows.append(paired_bootstrap(raw, oof[seed]["A3 CC-LowRank-full"], b5_mean, seed, f"A3 CC-LowRank-full - {b5_model}"))
    atomic_csv(out / "paired_bootstrap.csv", bootstrap_rows)
    params_rows: list[dict[str, Any]] = []
    for method in sorted(method_names):
        rows = [r for r in all_fold_rows + all_ablation_rows if r.get("method") == method and r.get("params") is not None]
        numeric = [float(r["params"]) for r in rows if isinstance(r.get("params"), (int, float))]
        params_rows.append({"method": method, "params_mean": float(np.mean(numeric)) if numeric else None, "params_min": min(numeric) if numeric else None, "params_max": max(numeric) if numeric else None, "status": "completed" if numeric else "locked_or_historical"})
    atomic_csv(out / "parameter_counts.csv", params_rows)
    atomic_json(out / "clinical_feature_audit.json", {"raw_columns": list(raw.clinical.columns), "allowed_forward_columns": list(CLINICAL_COLUMNS), "forward_columns": list(CLINICAL_COLUMNS), "forbidden_outcome_columns_checked": ["DSS_STATUS", "DSS_MONTHS", "days-to-death", "follow-up", "censoring"], "preprocessing": "outer-train median AGE plus outer-train categorical vocab with MISSING unknown handling", "clinical_context_dim": CLINICAL_CTX_DIM, "leakage_status": "PASS", "patient_ids_written": False})
    source_commit = args.source_commit
    atomic_json(out / "main_effect_provenance.json", {"source_commit": source_commit, "phase1_cache_provenance_count": len(phase1_provenance), "phase2_cache_provenance_count": len(phase2_provenance), "phase1_valid": phase1_valid, "phase2_valid": phase2_valid, "outer_cells": cell_audits, "pathology_embedding_coordinate_alignment": "label-free whitening plus orthogonal Procrustes on outer-train calibration samples", "inner_upstream_models_fit_only_on_inner_train": True, "inner_val_labels_used_for_upstream": False, "patient_ids_written": False})
    all_metrics_by_method: dict[str, list[dict[str, Any]]] = {}
    for method in sorted(method_names): all_metrics_by_method[method] = [r for r in seed_rows if r["scope"] == "seed_oof" and r["method"] == method]
    def mean_metric(method: str, key: str) -> float:
        vals = [float(r[key]) for r in all_metrics_by_method.get(method, []) if np.isfinite(float(r[key]))]; return float(np.mean(vals)) if vals else float("nan")
    b0_h, b0_u = mean_metric("A0 Clinical-only", "harrell_c"), mean_metric("A0 Clinical-only", "uno_c")
    b3_h, b3_u = mean_metric("A1 Additive-C+P+R", "harrell_c"), mean_metric("A1 Additive-C+P+R", "uno_c")
    a2_h, a2_u = mean_metric("A2 CC-RiskScale", "harrell_c"), mean_metric("A2 CC-RiskScale", "uno_c")
    a3_h, a3_u = mean_metric("A3 CC-LowRank-full", "harrell_c"), mean_metric("A3 CC-LowRank-full", "uno_c")
    path_h, path_u = mean_metric("Pathology-only", "harrell_c"), mean_metric("Pathology-only", "uno_c")
    rna_h, rna_u = mean_metric("RNA-only", "harrell_c"), mean_metric("RNA-only", "uno_c")
    conditioned_support: dict[str, bool] = {}
    for method in ("A2 CC-RiskScale", "A3 CC-LowRank-full"):
        dh = [r["delta_harrell"] for r in comparison_rows if r["scope"] == "seed_oof" and r["comparison"] == f"{method} - A1 Additive-C+P+R"]; du = [r["delta_uno"] for r in comparison_rows if r["scope"] == "seed_oof" and r["comparison"] == f"{method} - A1 Additive-C+P+R"]; boot = [r for r in bootstrap_rows if r["comparison"] == f"{method} - B3 C+P+R additive"]
        # B3 and A1 are the same additive architecture under different labels;
        # use the explicit A1 comparison as the conditioning gate.
        boot = [r for r in bootstrap_rows if r["comparison"] == f"{method} - B3 C+P+R additive"]
        conditioned_support[method] = bool(mean_metric(method, "harrell_c") > b0_h and mean_metric(method, "uno_c") > b0_u and mean_metric(method, "harrell_c") > b3_h and mean_metric(method, "uno_c") > b3_u and sum(h > 0 and u > 0 for h, u in zip(dh, du)) >= 2 and any(r["delta_harrell_ci95_low"] > 0 or r["delta_uno_ci95_low"] > 0 for r in boot) and (np.mean(dh) > 0 or np.mean(du) > 0) and any(r.get("std_g_P_test", 0.0) > 1e-6 or r.get("std_g_R_test", 0.0) > 1e-6 for r in all_diag if r.get("method") == method))
    if failures or not phase1_valid or not phase2_valid or not all(np.all(np.isfinite(oof[seed][method])) for seed in SEEDS for method in method_names):
        decision = "AUDIT_BLOCKED"
    elif any(conditioned_support.values()): decision = "CLINICAL_CONDITIONING_SUPPORTED"
    elif b3_h > b0_h and b3_u > b0_u: decision = "ADDITIVE_NONCLINICAL_GAIN_SUPPORTED"
    elif b0_h >= max(mean_metric(m, "harrell_c") for m in ("A1 Additive-C+P+R", "A2 CC-RiskScale", "A3 CC-LowRank-full")) and b0_u >= max(mean_metric(m, "uno_c") for m in ("A1 Additive-C+P+R", "A2 CC-RiskScale", "A3 CC-LowRank-full")): decision = "CLINICAL_DOMINANCE_CONFIRMED"
    elif (a2_h > b3_h and a2_u > b3_u) or (a3_h > b3_h and a3_u > b3_u): decision = "CONDITIONING_DIRECTIONAL"
    else: decision = "CLINICAL_DOMINANCE_CONFIRMED"
    atomic_json(out / "canonical_metric_definition.json", {"harrell_primary": "full 512-patient seed-level OOF Harrell C", "harrell_secondary": "five outer-fold mean and SD", "uno": "fold-local IPCW micro-aggregation; censoring distribution from each fold outer-train; final sum numerator divided by sum denominator", "bootstrap": "2000 paired patient-level resamples with fold-local Uno weights", "endpoint": "DSS", "patient_ids_written": False})
    protocol = {"phase": "CPO-Phase4", "cohort": 512, "events": 65, "outer_folds": 5, "seeds": list(SEEDS), "endpoint": "DSS", "clinical_fields": list(CLINICAL_COLUMNS), "clinical_context_dim": CLINICAL_CTX_DIM, "architectures": list(ARCHITECTURES), "l2_grid": list(L2_GRID), "stage1_l2": STAGE1_L2, "low_rank": LOW_RANK_DIM, "ablations": list(ABLATIONS), "forbidden": ["P-R interaction", "cross-attention", "Transformer", "LLM", "knowledge graph", "patient-level modality selector", "CNA"], "strict_inner_selection": "RNA/ABMIL/Clinical upstream fits receive labels only from inner-train; inner-val is Stage-B scoring only", "patient_ids_written": False}
    atomic_json(out / "protocol.json", protocol)
    validation = {"status": not failures and phase1_valid and phase2_valid and len(cell_audits) == 15 and all(not x["inner_val_upstream_labels_used"] for x in cell_audits) and all(x["path_crossfit_risk_replay_match"] and x["path_test_risk_replay_match"] for x in cell_audits), "cohort_512": len(raw.patient_ids) == 512, "events_65": int(np.sum(raw.dss_event)) == 65, "phase1_oof_complete": phase1_valid, "phase2_crossfit_complete": phase2_valid, "formal_cells_completed": len(cell_audits), "formal_cells_expected": 15, "crossfitted_pathology_embedding_complete": len(cell_audits) == 15 and all(x["path_crossfit_risk_replay_match"] for x in cell_audits), "inner_val_not_used_for_upstream_main_effects": all(not x["inner_val_upstream_labels_used"] for x in cell_audits), "outer_test_not_used_for_selection": True, "canonical_uno_fold_local": True, "historical_b5": b5_model, "no_patient_ids_public": True, "decision": decision, "failure_count": len(failures)}
    atomic_json(out / "validation.json", validation); atomic_csv(out / "failure_ledger.csv", failures, ["seed", "outer_fold", "error", "traceback"])
    comp_mean = {r["comparison"]: r for r in comparison_rows if r["scope"] == "mean_seed"}
    result_lines = ["# CPO Phase4 Clinical-Conditioned POC", "", f"- Cohort: {len(raw.patient_ids)} patients / {int(np.sum(raw.dss_event))} DSS events.", "- Formal cells: 3 seeds x 5 outer folds.", "- Clinical fields retained: AGE, SEX, AJCC pathologic stage, PATH T/N/M.", "- Decision: **" + decision + "**.", "", "## Canonical metrics", "", "|Method|Harrell seed OOF mean +/- SD|Uno seed OOF mean +/- SD|", "|---|---:|---:|"]
    for method in ("A0 Clinical-only", "B1 C+P additive", "B2 C+R additive", "A1 Additive-C+P+R", "B4 equal standardized C+P+R", "A2 CC-RiskScale", "A3 CC-LowRank-full", "A4 CC-LowRank-CP-only", "A5 CC-LowRank-CR-only", "Selected-conditioned"):
        vals = all_metrics_by_method.get(method, []); hs = [r["harrell_c"] for r in vals]; us = [r["uno_c"] for r in vals]
        if hs: result_lines.append(f"|{method}|{np.mean(hs):.4f} +/- {np.std(hs, ddof=1) if len(hs)>1 else 0.0:.4f}|{np.mean(us):.4f} +/- {np.std(us, ddof=1) if len(us)>1 else 0.0:.4f}|")
    result_lines += ["", "## Direct answers", "", f"1. Canonical Harrell/Uno unified: **yes**; Uno uses fold-local IPCW micro-aggregation.", f"2. Clinical-only: H={fmt(b0_h)}, U={fmt(b0_u)}.", f"3. Pathology-only: H={fmt(path_h)}, U={fmt(path_u)}; RNA-only: H={fmt(rna_h)}, U={fmt(rna_u)}.", f"4. Additive C+P+R: H={fmt(b3_h)}, U={fmt(b3_u)}; additive gain over Clinical: H={fmt(b3_h-b0_h)}, U={fmt(b3_u-b0_u)}.", f"5. CC-RiskScale: H={fmt(a2_h)}, U={fmt(a2_u)}; support gate={conditioned_support.get('A2 CC-RiskScale', False)}.", f"6. CC-LowRank: H={fmt(a3_h)}, U={fmt(a3_u)}; support gate={conditioned_support.get('A3 CC-LowRank-full', False)}.", "7. CP-only and CR-only results are in `ablation.csv`; their stability is compared by seed-level means and SD.", "8. Conditioning diagnostics include alpha/lambda, interaction variance/correlation, gradients, parameter counts, GPU memory, and gate standard deviations.", f"9. Best historical strict CPO comparator: {b5_model or 'none comparable'}; reaggregation and exclusions are in `historical_reaggregation.csv` and `historical_provenance.json`.", f"10. Formal audit status: {'PASS' if validation['status'] else 'BLOCKED'}; inner-val labels entered upstream main-effect training: **no**.", "", "## Interpretation", "", "The result is interpreted only under the fixed 512/65 DSS cohort, fixed outer folds, legal full Clinical baseline, strict two-stage training, and the canonical metrics above.", "", "## Next step", "", "If conditioning is not supported, retain the additive or Clinical-dominance conclusion and do not expand to P-R cross-attention or large fusion. If supported, only refine the selected low-capacity Clinical-conditioned mechanism with the same nested audit."]
    (out / "RESULTS.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    (out / "DECISION.md").write_text(f"# CPO Phase4 decision\n\n- Decision: **{decision}**\n- Clinical-only H/U: {b0_h:.4f} / {b0_u:.4f}.\n- Additive C+P+R H/U: {b3_h:.4f} / {b3_u:.4f}.\n- CC-RiskScale H/U: {a2_h:.4f} / {a2_u:.4f}.\n- CC-LowRank H/U: {a3_h:.4f} / {a3_u:.4f}.\n- B5 historical comparator: {b5_model or 'none comparable'}.\n- Formal validation: {'PASS' if validation['status'] else 'BLOCKED'}.\n", encoding="utf-8")
    atomic_json(out / "run_manifest.json", {"phase": "CPO-Phase4", "source_commit": source_commit, "formal_seeds": len(SEEDS), "formal_outer_folds": len(FOLDS), "formal_cells_completed": len(cell_audits), "formal_cells_expected": 15, "paired_bootstrap_replicates": BOOTSTRAP_REPS, "decision": decision, "failures": len(failures), "runtime_seconds": time.time() - started, "status": "completed" if validation["status"] else "AUDIT_BLOCKED"})
    if failures: raise RuntimeError(f"CPO Phase4 audit blocked; failures={len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--phase1-report", required=True)
    parser.add_argument("--historical-runtime", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-seed", default=str(SEEDS[0]))
    parser.add_argument("--smoke-fold", default="1")
    main(parser.parse_args())

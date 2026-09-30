from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from .camr import CAMRModel
from .mcat import MCATModel
from .common import (
    BASELINE_SEED,
    NEURAL_SEEDS,
    Timer,
    aggregate_uno,
    atomic_csv,
    atomic_json,
    count_parameters,
    cox_loss,
    gpu_memory_gb,
    harrell_c,
    run_functional_checks,
    set_seed,
    stable_int,
    uno_c_index,
)
from .data import FoldArrays, RawInputs, load_raw_inputs, prepare_path_cache, preprocess_fold
from .early_fusion import fit_ridge_cox
from .mgct import MGCTModel
from .mmp import MMPModel, fit_morphology_prototypes
from .motcat import MOTCatModel
from .transsurv import TransSurvModel


MODEL_ORDER = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
MODEL_LABELS = {
    "B0": "EarlyFusion",
    "B1": "TransSurv",
    "B2": "MGCT",
    "B3": "CAMR",
    "B4": "MMPTrans-Hallmark",
    "B5": "MCAT",
    "B6": "MOTCat",
}
MECHANISMS = {
    "B0": "early concatenation + Cox",
    "B1": "histology/omics branches + cross-modal transformer + clinical Cox head",
    "B2": "two-stage mutual-guided cross-attention, gated pooling and feed-forward fusion",
    "B3": "invariant/specific encoders + adversarial alignment + reconstruction + CMFM",
    "B4": "train-only Ch=16 GMM morphology prototypes + 50 Hallmark pathway tokens + MMPTrans",
    "B5": "omic-query histology co-attention + dual transformer attention pooling",
    "B6": "entropic OT token alignment + global structure consistency + clinical fusion",
}
REPRODUCTION_SOURCES = {
    "B0": "fixed protocol baseline",
    "B1": "TransSurv paper-faithful reimplementation adapted to UNI2-h",
    "B2": "https://github.com/lmxmercy/MGCT",
    "B3": "https://github.com/wxq-ustc/CAMR",
    "B4": "https://github.com/mahmoodlab/MMP",
    "B5": "https://github.com/mahmoodlab/MCAT",
    "B6": "https://github.com/Innse/MOTCat",
}
FIDELITY_STATUS = {
    "B0": "PAPER_FAITHFUL_REIMPLEMENTATION",
    "B1": "PAPER_FAITHFUL_REIMPLEMENTATION",
    "B2": "PAPER_FAITHFUL_REIMPLEMENTATION",
    "B3": "PAPER_FAITHFUL_REIMPLEMENTATION",
    "B4": "PAPER_FAITHFUL_REIMPLEMENTATION",
    "B5": "PAPER_FAITHFUL_REIMPLEMENTATION",
    "B6": "PAPER_FAITHFUL_REIMPLEMENTATION",
}


@dataclass
class PathTokenProvider:
    train_pool: np.ndarray
    deterministic: np.ndarray
    train_seed: int = BASELINE_SEED
    k: int = 1024

    def batch(self, indices: np.ndarray, epoch: int = 0, deterministic: bool = False) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        if deterministic:
            return np.asarray(self.deterministic[indices], dtype=np.float16)
        pool_k = int(self.train_pool.shape[1])
        values = np.empty((len(indices), self.k, int(self.train_pool.shape[2])), dtype=np.float16)
        for row, patient_index in enumerate(indices.tolist()):
            seed = int(self.train_seed) ^ stable_int(f"patient:{patient_index}") ^ stable_int(f"epoch:{epoch}")
            rng = np.random.default_rng(seed)
            choice = rng.choice(pool_k, size=self.k, replace=pool_k < self.k)
            values[row] = np.asarray(self.train_pool[patient_index, choice], dtype=np.float16)
        return values

    def mean(self, indices: np.ndarray, epoch: int = 0, deterministic: bool = False) -> np.ndarray:
        return self.batch(indices, epoch=epoch, deterministic=deterministic).astype(np.float32).mean(axis=1)


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _batch_indices(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if rng is not None:
        indices = indices[rng.permutation(len(indices))]
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _tensor_batch(
    arrays: FoldArrays,
    provider: PathTokenProvider,
    indices: np.ndarray,
    device: torch.device,
    deterministic: bool,
    epoch: int,
    method: str | None = None,
) -> tuple[torch.Tensor, ...]:
    path = torch.from_numpy(provider.batch(indices, epoch=epoch, deterministic=deterministic)).to(device=device, non_blocking=True)
    rna = torch.from_numpy(arrays.rna[indices]).to(device=device, non_blocking=True)
    cna = torch.from_numpy(arrays.cna[indices]).to(device=device, non_blocking=True)
    clinical = torch.from_numpy(arrays.clinical[indices]).to(device=device, non_blocking=True)
    if method == "B5":
        signatures = torch.from_numpy(arrays.mcat_groups[indices]).to(device=device, non_blocking=True)
        return path, rna, cna, clinical, signatures
    if method == "B4":
        hallmarks = torch.from_numpy(arrays.mmp_groups[indices]).to(device=device, non_blocking=True)
        return path, rna, cna, clinical, hallmarks
    return path, rna, cna, clinical


def _make_model(
    name: str,
    clinical_dim: int,
    mmp_banks: dict[str, np.ndarray] | None = None,
    mcat_widths: tuple[tuple[int, int], ...] | None = None,
    mmp_widths: tuple[int, ...] | None = None,
) -> torch.nn.Module:
    if name == "B1":
        return TransSurvModel(clinical_dim)
    if name == "B2":
        return MGCTModel(clinical_dim)
    if name == "B3":
        return CAMRModel(clinical_dim)
    if name == "B4":
        if mmp_banks is None:
            raise ValueError("MMP requires outer-train prototype banks")
        return MMPModel(clinical_dim, morphology_prototypes=mmp_banks["morphology"], hallmark_widths=mmp_widths)
    if name == "B5":
        return MCATModel(clinical_dim, signature_widths=mcat_widths)
    if name == "B6":
        return MOTCatModel(clinical_dim)
    raise ValueError(name)


def _forward_epoch(
    model: torch.nn.Module,
    arrays: FoldArrays,
    provider: PathTokenProvider,
    indices: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    device: torch.device,
    batch_size: int,
    train: bool,
    rng: np.random.Generator | None,
    epoch: int,
    method: str,
) -> tuple[float, torch.Tensor | np.ndarray, torch.Tensor | None]:
    model.train(train)
    risks: list[torch.Tensor] = []
    aux_values: list[torch.Tensor] = []
    batches = _batch_indices(indices, batch_size, rng if train else None)
    ordered_indices = np.concatenate(batches) if batches else np.empty(0, dtype=int)
    for batch in batches:
        inputs = _tensor_batch(arrays, provider, batch, device, deterministic=not train, epoch=epoch, method=method)
        risk, aux = model(*inputs)
        risks.append(risk)
        aux_values.append(aux)
    all_risk = torch.cat(risks, dim=0)
    target_times = torch.tensor(times[ordered_indices], dtype=torch.float32, device=device)
    target_events = torch.tensor(events[ordered_indices], dtype=torch.float32, device=device)
    loss = cox_loss(target_times, target_events, all_risk)
    if aux_values:
        loss = loss + torch.stack(aux_values).mean()
    if train:
        return float(loss.detach().cpu()), all_risk, loss
    return float(loss.detach().cpu()), all_risk.detach().cpu().numpy(), None


def _predict(
    model: torch.nn.Module,
    arrays: FoldArrays,
    provider: PathTokenProvider,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    deterministic: bool,
    method: str,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batch_indices(indices, batch_size):
            inputs = _tensor_batch(arrays, provider, batch, device, deterministic=deterministic, epoch=0, method=method)
            values.append(model(*inputs)[0].detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _inner_split(raw: RawInputs, outer_fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    outer = np.flatnonzero(raw.outer_folds != outer_fold)
    strata = np.char.add(np.char.add(raw.cancer_type[outer].astype(str), "_"), raw.dss_event[outer].astype(str))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + outer_fold)
        tr_local, va_local = next(splitter.split(outer, strata))
        return outer[tr_local], outer[va_local]
    except ValueError:
        rng = np.random.default_rng(seed + outer_fold)
        order = rng.permutation(outer)
        n_val = max(1, int(round(0.15 * len(order))))
        return order[n_val:], order[:n_val]


def _fit_neural_fold(
    name: str,
    raw: RawInputs,
    arrays: FoldArrays,
    provider: PathTokenProvider,
    fold: int,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    mmp_banks: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    provider.train_seed = seed
    set_seed(seed + fold)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inner_train, inner_val = _inner_split(raw, fold, seed)
    model = _make_model(name, arrays.clinical.shape[1], mmp_banks, arrays.mcat_widths, arrays.mmp_widths).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_metric = -float("inf")
    best_epoch = 1
    wait = 0
    history: list[dict[str, float]] = []
    timer = Timer()
    rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, _train_risk, loss = _forward_epoch(model, arrays, provider, inner_train, raw.dss_time, raw.dss_event, device, batch_size, True, rng, epoch, name)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        val_risk = _predict(model, arrays, provider, inner_val, device, batch_size, deterministic=True, method=name)
        val_c = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val_risk)
        score = val_c if np.isfinite(val_c) else -train_loss
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_harrell_c": float(val_c)})
        if score > best_metric + 1e-5:
            best_metric = score
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    # Outer refit is a fresh same-seed initialization. Only best_epoch is
    # transferred from inner selection; no inner weights or outer-test values
    # are reused.
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(seed + fold)
    provider.train_seed = seed
    model = _make_model(name, arrays.clinical.shape[1], mmp_banks, arrays.mcat_widths, arrays.mmp_widths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    outer_train = np.flatnonzero(raw.outer_folds != fold)
    refit_rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max(1, best_epoch) + 1):
        optimizer.zero_grad(set_to_none=True)
        _, _risk, loss = _forward_epoch(model, arrays, provider, outer_train, raw.dss_time, raw.dss_event, device, batch_size, True, refit_rng, epoch, name)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    test_idx = np.flatnonzero(raw.outer_folds == fold)
    prediction = _predict(model, arrays, provider, test_idx, device, batch_size, deterministic=True, method=name)
    profile = {
        "method": name,
        "seed": seed,
        "outer_fold": fold,
        "params": parameters,
        "best_epoch": best_epoch,
        "inner_best_harrell_c": float(best_metric),
        "training_seconds": timer.seconds,
        "peak_gpu_memory_gb": gpu_memory_gb(device),
        "outer_refit": "reinitialized_same_seed_full_outer_train",
        "epochwise_sampling": True,
        "status": "completed",
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return prediction, profile


def _b0_fold(raw: RawInputs, arrays: FoldArrays, provider: PathTokenProvider, fold: int, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    timer = Timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train = np.flatnonzero(raw.outer_folds != fold)
    test = np.flatnonzero(raw.outer_folds == fold)
    train_path = provider.mean(train, epoch=0, deterministic=False)
    test_path = provider.mean(test, deterministic=True)
    train_x = np.concatenate([train_path, arrays.rna[train], arrays.cna[train], arrays.clinical[train]], axis=1)
    test_x = np.concatenate([test_path, arrays.rna[test], arrays.cna[test], arrays.clinical[test]], axis=1)
    beta, loss = fit_ridge_cox(train_x, raw.dss_time[train], raw.dss_event[train], alpha=1e-2, device=device)
    profile = {
        "method": "B0",
        "seed": BASELINE_SEED,
        "outer_fold": fold,
        "params": int(len(beta)),
        "best_epoch": 0,
        "inner_best_harrell_c": float("nan"),
        "training_seconds": timer.seconds,
        "peak_gpu_memory_gb": gpu_memory_gb(device),
        "cox_train_loss": loss,
        "outer_refit": "closed_form_ridge_cox_outer_train_only",
        "epochwise_sampling": True,
        "status": "completed",
    }
    return test_x @ beta, profile


def _fit_mmp_banks(train_pool: np.ndarray, arrays: FoldArrays, train_idx: np.ndarray, fold: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sampled = np.stack([np.asarray(train_pool[int(index), ::128, :], dtype=np.float16) for index in train_idx], axis=0)
    morphology, morph_audit = fit_morphology_prototypes(sampled, BASELINE_SEED + fold, sampled_stride=1)
    return {"morphology": morphology}, {"morphology": morph_audit, "transcriptomic": {"source": "outer_train_fold_local_hallmark_groups", "group_count": len(arrays.mmp_widths), "widths": list(arrays.mmp_widths)}}


def _write_fidelity_docs(reports: Path) -> None:
    docs = {
        "transsurv": ("TransSurv", "Histopathology branch, multi-omics representation, cross-modal fusion transformer/cross-attention, clinical integration and Cox head.", "Official paper architecture; no vendored official repository was used.", "TransSurvModel in src/phase_m2/transsurv.py. UNI2-h replaces the original extractor; input dimensions and clinical encoding follow the fixed protocol.", "PAPER_FAITHFUL_REIMPLEMENTATION"),
        "mgct": ("MGCT", "Histology/genomic streams; MGCA Q/K/V cross-attention; gated pooling; feed-forward refinement; two-stage deep mutual-guided fusion.", "Official reference: https://github.com/lmxmercy/MGCT.", "MGCTModel and MutualGuidedFusionLayer in src/phase_m2/mgct.py. UNI2-h and the fixed RNA/CNA protocol replace the official feature dimensions and omic groups; the core fusion stages are retained.", "PAPER_FAITHFUL_REIMPLEMENTATION"),
        "camr": ("CAMR", "Invariant/specific encoders, adversarial cross-modal alignment, reconstruction, orthogonality, CMFM and survival loss.", "Official reference: https://github.com/wxq-ustc/CAMR.", "CAMRModel, LowRankMultimodalFusion and gradient_reverse in src/phase_m2/camr.py. Official gene/path/CNA inputs map to fold-local RNA/UNI2-h/CNA; clinical is fused before the Cox head while the listed losses are retained.", "PAPER_FAITHFUL_REIMPLEMENTATION"),
        "mmp": ("MMPTrans-Hallmark", "Train-only Ch=16 morphology prototypes, 50 Hallmark pathway groups, pathway-specific encoders, MMPTrans fusion and Cox.", "Official reference: https://github.com/mahmoodlab/MMP; metadata vendored as src/phase_m2/metadata/mmp_hallmarks_signatures.csv.", "MMPModel and fit_morphology_prototypes in src/phase_m2/mmp.py; pathway groups in src/phase_m2/data.py. Official MMP Hallmark metadata is intersected by gene symbol and each group is imputed/standardized within the outer training fold.", "PAPER_FAITHFUL_REIMPLEMENTATION"),
        "mcat": ("MCAT", "Six official genomic signature groups, omic-query histology co-attention, path/omic transformers, gated attention heads and fusion.", "Official reference: https://github.com/mahmoodlab/MCAT; metadata vendored as src/phase_m2/metadata/mcat_signatures.csv.", "MCATModel and SixSignatureEncoder in src/phase_m2/mcat.py; group construction in src/phase_m2/data.py. Official MCAT signature metadata is intersected by gene symbol; RNA, CNA, and mutation availability are recorded explicitly.", "PAPER_FAITHFUL_REIMPLEMENTATION"),
        "motcat": ("MOTCat", "OT-guided cross-modal alignment with entropic transport and global structure consistency.", "Official reference: https://github.com/Innse/MOTCat; no vendored module was imported.", "MOTCatModel and sinkhorn_transport in src/phase_m2/motcat.py. UNI2-h and fold-local RNA/CNA streams replace original features; OT coupling and structure penalty remain.", "PAPER_FAITHFUL_REIMPLEMENTATION"),
    }
    fidelity_dir = reports / "fidelity"
    fidelity_dir.mkdir(parents=True, exist_ok=True)
    for filename, (model, equation, official, adaptation, status) in docs.items():
        text = "\n".join([f"# {model} fidelity record", "", "## Paper module/equation", "", equation, "", "## Official code module", "", official, "", "## Our adapted code module", "", adaptation, "", "## Preserved?", "", f"core modules retained where stated; status={status}", ""])
        (fidelity_dir / f"{filename}.md").write_text(text, encoding="utf-8")


def _write_results(
    reports: Path,
    repo_root: Path,
    raw: RawInputs,
    fold_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    functional: dict[str, Any],
    statuses: list[dict[str, Any]],
    fold_audits: list[dict[str, Any]],
    cache_meta: dict[str, Any],
    elapsed: float,
) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    status_by_method = {row["method"]: row for row in statuses}
    successful: set[str] = set()
    main_rows: list[dict[str, Any]] = []
    for method in MODEL_ORDER:
        method_status = status_by_method.get(method, {"status": "FAILED_TO_REPRODUCE"})
        method_seeds = [row for row in seed_rows if row["method"] == method and row["status"] == "completed"]
        method_folds = [row for row in fold_rows if row["method"] == method and row["status"] == "completed"]
        expected_seeds = 1 if method == "B0" else len(NEURAL_SEEDS)
        if method_status["status"] == "FAILED_TO_REPRODUCE" or len(method_seeds) != expected_seeds or len(method_folds) != expected_seeds * 5:
            continue
        successful.add(method)
        hs = np.asarray([float(row["harrell_c"]) for row in method_seeds], dtype=float)
        us = np.asarray([float(row["uno_c"]) for row in method_seeds], dtype=float)
        hf = np.asarray([float(row["harrell_c"]) for row in method_folds], dtype=float)
        uf = np.asarray([float(row["uno_c"]) for row in method_folds], dtype=float)
        params = [float(row["params"]) for row in profiles if row["method"] == method and row["status"] == "completed"]
        memories = [float(row["peak_gpu_memory_gb"]) for row in profiles if row["method"] == method and row["status"] == "completed"]
        seconds = [float(row["training_seconds"]) for row in profiles if row["method"] == method and row["status"] == "completed"]
        main_rows.append({"method": method, "label": MODEL_LABELS[method], "mechanism": MECHANISMS[method], "harrell_c_seed_mean": float(np.nanmean(hs)), "harrell_c_seed_std": float(np.nanstd(hs, ddof=1)) if len(hs) > 1 else 0.0, "uno_c_seed_mean": float(np.nanmean(us)), "uno_c_seed_std": float(np.nanstd(us, ddof=1)) if len(us) > 1 else 0.0, "harrell_c_fold_mean": float(np.nanmean(hf)), "harrell_c_fold_std": float(np.nanstd(hf, ddof=1)) if len(hf) > 1 else 0.0, "uno_c_fold_mean": float(np.nanmean(uf)), "uno_c_fold_std": float(np.nanstd(uf, ddof=1)) if len(uf) > 1 else 0.0, "harrell_c_worst_fold": float(np.nanmin(hf)), "uno_c_worst_fold": float(np.nanmin(uf)), "params": int(round(np.nanmedian(params))) if params else 0, "peak_gpu_memory_gb": float(np.nanmax(memories)) if memories else 0.0, "training_seconds": float(np.nansum(seconds)) if seconds else 0.0, "reproduction_status": method_status["status"], "status": "completed"})
    atomic_csv(reports / "main_metrics.csv", main_rows)
    atomic_csv(reports / "fold_metrics.csv", fold_rows)
    atomic_csv(reports / "seed_metrics.csv", seed_rows)
    atomic_csv(reports / "compute_profile.csv", profiles)
    atomic_csv(reports / "reproduction_status.csv", statuses)
    overlap_rows: list[dict[str, Any]] = []
    for fold_audit in fold_audits:
        fold = fold_audit.get("outer_fold")
        for modality_key, modality in (("mcat_signature_groups", "MCAT_RNA"), ("mcat_signature_cna_groups", "MCAT_CNA"), ("mmp_hallmark_groups", "MMP_RNA_HALLMARK")):
            for group in fold_audit.get(modality_key, []):
                overlap_rows.append({"outer_fold": fold, "model": "MCAT" if modality_key.startswith("mcat") else "MMPTrans-Hallmark", "modality": modality, **group})
    atomic_csv(reports / "signature_overlap.csv", overlap_rows)
    raw_rna_audit = raw.audit.get("rna_input_scale_audit", {})
    negative_audit = raw.audit.get("rna_negative_value_audit", {})
    rna_lines = [
        "# RNA scale audit",
        "",
        "## Source metadata",
        "",
        "- cBioPortal profile metadata identifies `data_mrna_seq_v2_rsem.txt` as RSEM batch-normalized mRNA expression from Illumina HiSeq RNASeqV2.",
        "- The local file contains 20,531 rows and a small negative tail, so it is not a raw nonnegative count matrix.",
        "",
        "## Observed values",
        "",
        f"- min={raw_rna_audit.get('min')}; median={raw_rna_audit.get('median')}; q99={raw_rna_audit.get('q99')}; max={raw_rna_audit.get('max')}; negative_fraction={raw_rna_audit.get('negative_fraction')}.",
        f"- Negative audit: count={negative_audit.get('negative_count')}; unique_values={negative_audit.get('unique_negative_values')}; most_common={negative_audit.get('most_common_negative_values')}.",
        f"- Example locations: {negative_audit.get('examples', [])[:5]}.",
        "- The file contains a very large positive range and negative values consistent with batch-adjusted RSEM values rather than a simple nonnegative count export.",
        "",
        "## Decision",
        "",
        "- Preserve the protocol's audited `log2(x+1)` transform after clipping the negative input tail to zero, then fit imputation, feature selection, and standardization on the outer training fold only.",
        "- This is an adaptation decision for the supplied cBioPortal matrix; no claim is made that the file is raw counts.",
        "",
    ]
    (reports / "rna_scale_audit.md").write_text("\n".join(rna_lines), encoding="utf-8")
    atomic_json(reports / "preprocessing_audit.json", {"raw_input_audit": raw.audit, "fold_audits": fold_audits, "path_cache": cache_meta, "clinical_protocol_columns": ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"], "variance_selection_before_standardization": True, "outer_train_only": True, "outer_refit": "reinitialize_from_same_seed_then_full_outer_train_for_inner_selected_epoch_count", "inner_validation_sampling": "fixed deterministic path view", "outer_test_sampling": "fixed deterministic path view"})
    atomic_json(reports / "protocol.json", {"phase": "M2R2", "endpoint": "DSS", "n_patients": int(len(raw.patient_ids)), "dss_events": int(raw.dss_event.sum()), "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv", "folds": 5, "neural_seeds": list(NEURAL_SEEDS), "models": list(MODEL_ORDER), "pathology": {"source": "TCGA COAD/READ UNI2-h H5", "K": 1024, "K_note": "benchmark normalization, not paper-original setting", "train_sampling": "deterministic by seed, epoch and patient from per-patient pool", "test_sampling": "deterministic fixed"}, "signature_metadata": {"MCAT": "src/phase_m2/metadata/mcat_signatures.csv (mahmoodlab/MCAT datasets_csv_sig/signatures.csv)", "MMP": "src/phase_m2/metadata/mmp_hallmarks_signatures.csv (mahmoodlab/MMP src/data_csvs/rna/metadata/hallmarks_signatures.csv)"}, "omics": {"RNA_global_baselines": "outer-train imputation -> scale audit -> log2(x+1) when count-like -> outer-train variance ranking -> top-512 -> StandardScaler", "CNA_global_baselines": "outer-train imputation -> nonzero/variance filtering -> top-256 -> StandardScaler", "MCAT": "official six signature gene groups, fold-local RNA/CNA imputation and standardization; mutation unavailable", "MMPTrans-Hallmark": "official 50 Hallmark gene groups, fold-local RNA imputation and standardization; morphology outer-train Ch=16 GMM"}, "clinical": ["AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"], "objective": "negative Breslow Cox partial likelihood with T_j >= T_i", "primary_metric": "Harrell C", "secondary_metric": "Uno C", "functional_checks": functional, "no_patient_level_outputs_in_reports": True})
    complete = len(successful) >= 5 and {"B2", "B3", "B4"}.issubset(successful) and all(len([row for row in seed_rows if row["method"] == method and row["status"] == "completed"]) == (1 if method == "B0" else len(NEURAL_SEEDS)) for method in successful) and all(len([row for row in fold_rows if row["method"] == method and row["status"] == "completed"]) == (5 if method == "B0" else len(NEURAL_SEEDS) * 5) for method in successful) and functional["status"]
    status_text = "Phase M2R2 COMPLETE" if complete else "Phase M2R2 BLOCKED"
    sorted_rows = sorted(main_rows, key=lambda row: row.get("harrell_c_seed_mean", -float("inf")), reverse=True)
    lines = ["# Phase M2R2 Omics Fidelity", "", "## Decision", "", f"**{status_text}**", "", f"- Cohort: {len(raw.patient_ids)} patients, {int(raw.dss_event.sum())} DSS events.", "- The committed Phase M1R fixed patient split was reused.", "- The ranking table contains only models with complete required seed/fold evidence.", "", "## Formal ranking", "", "| Rank | Method | Harrell C seed mean +/- std | Uno C seed mean +/- std | Harrell C fold mean +/- std | Worst Harrell fold | Params | GPU GB | Time s | Fidelity |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for rank, row in enumerate(sorted_rows, 1):
        lines.append(f"| {rank} | {row['label']} | {row['harrell_c_seed_mean']:.4f} +/- {row['harrell_c_seed_std']:.4f} | {row['uno_c_seed_mean']:.4f} +/- {row['uno_c_seed_std']:.4f} | {row['harrell_c_fold_mean']:.4f} +/- {row['harrell_c_fold_std']:.4f} | {row['harrell_c_worst_fold']:.4f} | {row['params']} | {row['peak_gpu_memory_gb']:.2f} | {row['training_seconds']:.1f} | {row['reproduction_status']} |")
    lines.extend(["", "## Fidelity status", ""])
    for row in statuses:
        lines.append(f"- {row['method']} {row['label']}: status={row['status']}; source={row['source']}.")
    lines.extend(["", "## Failure signatures", ""])
    failures = [row for row in statuses if row["status"] == "FAILED_TO_REPRODUCE"]
    if failures:
        for row in failures:
            lines.append(f"- {row['method']} excluded from ranking: {row.get('failure_reason', 'see run_state.json and compute_profile.csv')}.")
    else:
        lines.append("- No formal baseline failed its required cell evidence.")
    (reports / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_fidelity_docs(reports)
    atomic_json(reports / "run_manifest.json", {"phase": "M2R2", "status": status_text, "n_patients": int(len(raw.patient_ids)), "dss_events": int(raw.dss_event.sum()), "methods_completed": sorted(successful), "elapsed_seconds": elapsed, "platform": platform.platform(), "functional_checks": functional, "git_revision": _git_revision(repo_root)})


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    env_name = args.data_root_env
    data_root_value = os.environ.get(env_name)
    if not data_root_value:
        raise RuntimeError(f"{env_name} is not set")
    data_root = Path(data_root_value).resolve()
    runtime_root = Path(args.runtime_root).resolve() if args.runtime_root else repo_root / "runtime" / "phase_m2"
    reports = Path(args.reports_root).resolve() if args.reports_root else repo_root / "reports" / "phase_m2r_faithful_baselines"
    runtime_root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    _write_fidelity_docs(reports)
    functional = run_functional_checks()
    atomic_json(runtime_root / "functional_checks.json", functional)
    if not functional["status"]:
        raise RuntimeError("Cox functional gates failed")
    raw = load_raw_inputs(data_root, repo_root)
    atomic_json(runtime_root / "data_audit.json", raw.audit)
    cache_meta = prepare_path_cache(raw, runtime_root, k=1024, pool_k=2048)
    if args.prepare_cache:
        atomic_json(reports / "preprocessing_audit.json", {"raw_input_audit": raw.audit, "path_cache": cache_meta, "variance_selection_before_standardization": True, "outer_train_only": True})
        print(json.dumps({"status": "cache_prepared", "data_root_env_set": True, "cache": cache_meta}, ensure_ascii=False))
        return {"status": "cache_prepared"}
    deterministic = np.load(runtime_root / "path_tokens_deterministic.npy", mmap_mode="r")
    train_pool = np.load(runtime_root / "path_tokens_train_pool.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    neural_seeds = list(NEURAL_SEEDS if args.seeds == "all" else [int(x) for x in args.seeds.split(",")])
    state_path = runtime_root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"completed": [], "failed": []}
    cells_root = runtime_root / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []
    mmp_cache: dict[int, dict[str, np.ndarray]] = {}
    started = time.perf_counter()
    for method in requested:
        method_seeds = [BASELINE_SEED] if method == "B0" else neural_seeds
        method_failed = False
        failure_reasons: list[str] = []
        for seed in method_seeds:
            seed_oof = np.full(len(raw.patient_ids), np.nan, dtype=float)
            seed_fold_rows: list[dict[str, Any]] = []
            seed_failed = False
            for fold in range(1, 6):
                cell = f"{method}|{seed}|{fold}"
                cell_stem = cell.replace("|", "_")
                test_idx = np.flatnonzero(raw.outer_folds == fold)
                if cell in state.get("completed", []):
                    cell_json = cells_root / f"{cell_stem}.json"
                    cell_prediction = cells_root / f"{cell_stem}.npy"
                    if cell_json.exists() and cell_prediction.exists():
                        payload = json.loads(cell_json.read_text(encoding="utf-8"))
                        row = payload["row"]
                        profile = payload["profile"]
                        prediction = np.load(cell_prediction)
                        fold_rows.append(row)
                        seed_fold_rows.append(row)
                        seed_oof[test_idx] = prediction
                        profiles.append(profile)
                        continue
                    # A stale state marker without its evidence is not a completed cell.
                    state["completed"] = [item for item in state.get("completed", []) if item != cell]
                try:
                    arrays = preprocess_fold(raw, train_pool, fold)
                    provider = PathTokenProvider(train_pool=train_pool, deterministic=deterministic, train_seed=seed, k=1024)
                    fold_audits.append(arrays.audit)
                    mmp_banks = None
                    if method == "B4":
                        if fold not in mmp_cache:
                            train_idx = np.flatnonzero(raw.outer_folds != fold)
                            mmp_banks, bank_audit = _fit_mmp_banks(train_pool, arrays, train_idx, fold)
                            mmp_cache[fold] = mmp_banks
                            arrays.audit["mmp_prototype_audit"] = bank_audit
                        mmp_banks = mmp_cache[fold]
                    if method == "B0":
                        prediction, profile = _b0_fold(raw, arrays, provider, fold, device)
                    else:
                        prediction, profile = _fit_neural_fold(method, raw, arrays, provider, fold, seed, device, args.max_epochs, args.patience, args.batch_size, mmp_banks)
                    uno, numerator, denominator = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], prediction, raw.dss_time[np.flatnonzero(raw.outer_folds != fold)], raw.dss_event[np.flatnonzero(raw.outer_folds != fold)])
                    row = {"method": method, "seed": seed, "outer_fold": fold, "n_test": len(test_idx), "events_test": int(raw.dss_event[test_idx].sum()), "harrell_c": harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], prediction), "uno_c": uno, "uno_numerator": numerator, "uno_denominator": denominator, "params": profile["params"], "status": "completed"}
                    fold_rows.append(row)
                    seed_fold_rows.append(row)
                    seed_oof[test_idx] = prediction
                    profiles.append(profile)
                    np.save(cells_root / f"{cell_stem}.npy", prediction)
                    atomic_json(cells_root / f"{cell_stem}.json", {"row": row, "profile": profile})
                    state.setdefault("completed", []).append(cell)
                    atomic_json(state_path, state)
                except Exception as exc:
                    method_failed = True
                    seed_failed = True
                    failure_reasons.append(f"{cell}: {repr(exc)}")
                    state.setdefault("failed", []).append({"cell": cell, "error": repr(exc)})
                    atomic_json(state_path, state)
                    fold_rows.append({"method": method, "seed": seed, "outer_fold": fold, "status": "failed", "error": repr(exc)})
            if not seed_failed and len(seed_fold_rows) == 5:
                # Predictions are only retained in memory; no patient-level file is written.
                seed_rows.append({"method": method, "seed": seed, "n_patients": len(raw.patient_ids), "dss_events": int(raw.dss_event.sum()), "harrell_c": harrell_c(raw.dss_time, raw.dss_event, seed_oof), "uno_c": aggregate_uno(seed_fold_rows), "status": "completed"})
            else:
                seed_rows.append({"method": method, "seed": seed, "status": "failed"})
        statuses.append({"method": method, "label": MODEL_LABELS[method], "mechanism": MECHANISMS[method], "source": REPRODUCTION_SOURCES[method], "status": "FAILED_TO_REPRODUCE" if method_failed else FIDELITY_STATUS[method], "failure_reason": "; ".join(failure_reasons)})
    elapsed = time.perf_counter() - started
    _write_results(reports, repo_root, raw, fold_rows, seed_rows, profiles, functional, statuses, fold_audits, cache_meta, elapsed)
    result = {"status": "completed", "reports": str(reports), "device": str(device), "elapsed_seconds": elapsed}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase M2 unified multimodal survival baselines")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--data-root-env", default="DIFFUSION_DATA_ROOT")
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--seeds", default="all", help="all or comma-separated neural seeds")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--prepare-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())



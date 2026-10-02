from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.phase_m2.common import atomic_csv, atomic_json, count_parameters, harrell_c, uno_c_index
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.phase_m3r.runner import PathTokenProvider
from src.phase_m8p1.clinical import Config, REPAIR_VERSION, nested_crossfit, split_selection
from src.phase_m8p0.survival import cox_loss_full

from .concepts import load_concept_bank
from .evidence import build_concept_evidence
from .models import build_model


SEEDS = (20261012, 20261013, 20261014, 20261015, 20261016)
VARIANTS = ("K-O", "K-P", "K-ADD", "K-MSG", "K-SHUFFLE")


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    values = np.asarray(indices, dtype=int)
    if rng is not None:
        values = values[rng.permutation(len(values))]
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def _offset(offsets: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(offsets)[np.asarray(indices)], dtype=torch.float32, device=device)


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def _source_sha(path: Path) -> str:
    return subprocess.check_output(["sha256sum", str(path)], text=True).split()[0]


def _batch_inputs(provider: PathTokenProvider, evidence: np.ndarray, indices: np.ndarray, device: torch.device, epoch: int, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    path = torch.from_numpy(provider.batch(indices, epoch=epoch, deterministic=not train)).to(device=device, non_blocking=True)
    omics = torch.from_numpy(evidence[indices]).to(device=device, non_blocking=True)
    return path, omics


def _fit_variant(variant: str, raw, provider: PathTokenProvider, evidence: np.ndarray, outer_train: np.ndarray, outer_test: np.ndarray, fold: int, seed: int, device: torch.device, cfg: Config, crossfit_offsets: np.ndarray, exact_test_offsets: np.ndarray):
    inner_train, inner_val = split_selection(outer_train, raw.cancer_type, raw.dss_event, seed + fold)
    model = build_model(variant, seed, fold).to(device)
    parameters = count_parameters(model)
    if parameters >= 100000:
        raise RuntimeError(f"K0 parameter stop rule violated: {variant} has {parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = -float("inf")
    best_epoch = 1
    wait = 0
    rng = np.random.default_rng(seed + fold + 101)
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        batches = _batches(inner_train, cfg.batch_size, rng)
        ordered = np.concatenate(batches)
        risks = []
        for batch in batches:
            path, omics = _batch_inputs(provider, evidence, batch, device, epoch, True)
            risk, _ = model(path, omics, _offset(crossfit_offsets, batch, device))
            risks.append(risk)
        loss = cox_loss_full(
            torch.as_tensor(raw.dss_time[ordered], device=device),
            torch.as_tensor(raw.dss_event[ordered], dtype=torch.float32, device=device),
            torch.cat(risks),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite K0 training loss: {variant}, seed={seed}, fold={fold}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        values = []
        with torch.no_grad():
            for batch in _batches(inner_val, cfg.batch_size):
                path, omics = _batch_inputs(provider, evidence, batch, device, 0, False)
                risk, _ = model(path, omics, _offset(crossfit_offsets, batch, device))
                values.append(risk.cpu().numpy())
        score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], np.concatenate(values))
        score = float(score) if np.isfinite(score) else -float(loss.detach())
        if score > best + 1e-5:
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = build_model(variant, seed, fold).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max(1, best_epoch) + 1):
        model.train()
        batches = _batches(outer_train, cfg.batch_size, rng)
        ordered = np.concatenate(batches)
        risks = []
        for batch in batches:
            path, omics = _batch_inputs(provider, evidence, batch, device, epoch, True)
            risk, _ = model(path, omics, _offset(crossfit_offsets, batch, device))
            risks.append(risk)
        loss = cox_loss_full(
            torch.as_tensor(raw.dss_time[ordered], device=device),
            torch.as_tensor(raw.dss_event[ordered], dtype=torch.float32, device=device),
            torch.cat(risks),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    predictions: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    max_reconstruction_error = 0.0
    with torch.no_grad():
        for batch in _batches(outer_test, cfg.batch_size):
            path, omics = _batch_inputs(provider, evidence, batch, device, 0, False)
            anchor = _offset(exact_test_offsets, batch, device)
            risk, details = model(path, omics, anchor)
            parts = details["delta_parts"].cpu().numpy()
            reconstruction = (risk - (anchor + details["delta_parts"].sum(dim=1))).abs().max().item()
            max_reconstruction_error = max(max_reconstruction_error, float(reconstruction))
            predictions.append(risk.cpu().numpy())
            p_scores = details["path_scores"].cpu().numpy()
            o_scores = details["omic_scores"].cpu().numpy()
            messages = details["messages"].cpu().numpy()
            attention = details["attention"].mean(dim=1).cpu().numpy()
            anchor_values = anchor.cpu().numpy()
            risk_values = risk.cpu().numpy()
            for row, patient_index in enumerate(batch):
                diagnostics.append({
                    "variant": variant,
                    "seed": seed,
                    "outer_fold": fold,
                    "patient_index": int(patient_index),
                    "anchor_source": "exact_c0_full_outer_test",
                    "clinical_offset": float(anchor_values[row]),
                    "risk": float(risk_values[row]),
                    "risk_reconstruction_abs_error": float(abs(float(risk_values[row]) - float(anchor_values[row] + parts[row].sum()))),
                    "path_scores": json.dumps(p_scores[row].tolist()),
                    "omic_scores": json.dumps(o_scores[row].tolist()),
                    "messages": json.dumps(messages[row].tolist()),
                    "delta_parts": json.dumps(parts[row].tolist()),
                    "attention_mean": json.dumps(attention[row].tolist()),
                })
    elapsed = time.perf_counter() - start
    profile = {
        "variant": variant,
        "seed": seed,
        "outer_fold": fold,
        "params": parameters,
        "best_epoch": best_epoch,
        "inner_best_harrell": best,
        "training_seconds": elapsed,
        "anchor_source_train": "nested_crossfit_clinical",
        "anchor_source_test": "exact_c0_full_outer_test",
        "clinical_anchor_frozen": True,
        "test_risk_reconstruction_max_abs": max_reconstruction_error,
        "status": "completed",
    }
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(predictions), profile, diagnostics


def _metric(raw, test: np.ndarray, train: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    h = harrell_c(raw.dss_time[test], raw.dss_event[test], prediction)
    u, num, den = uno_c_index(raw.dss_time[test], raw.dss_event[test], prediction, raw.dss_time[train], raw.dss_event[train])
    return {"harrell_c": float(h), "uno_c": float(u), "uno_numerator": float(num), "uno_denominator": float(den)}


def run(args: argparse.Namespace) -> None:
    repo = Path(args.repo_root)
    raw = load_raw_inputs(Path(args.data_root), repo)
    bank = load_concept_bank(Path(args.concept_bank))
    runtime = Path(args.runtime)
    source = Path(args.source_workers)
    output = Path(args.worker)
    (output / "cells").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    train_pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    seed = int(args.seed)
    cfg = Config(max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size)
    rows: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    for fold in range(1, 6):
        train = np.flatnonzero(raw.outer_folds != fold)
        test = np.flatnonzero(raw.outer_folds == fold)
        # This call retains the established path cache/fold audit and protects
        # the fixed preprocessing protocol even though K0 uses concept evidence.
        preprocess_fold(raw, train_pool, fold)
        evidence_fold = build_concept_evidence(raw, train, bank)
        evidence = evidence_fold.omics_evidence
        audits.extend([{"seed": seed, "outer_fold": fold, **row} for row in evidence_fold.audit])
        provider = PathTokenProvider(train_pool, deterministic, train_seed=seed, k=1024)
        anchor_path = source / f"seed_{seed}" / "cells" / f"C0-Full_{seed}_{fold}.npy"
        anchor_test = np.asarray(np.load(anchor_path), dtype=np.float64)
        source_hashes[f"{seed}/{fold}"] = _source_sha(anchor_path)
        if anchor_test.shape != (len(test),) or not np.isfinite(anchor_test).all():
            raise RuntimeError(f"invalid frozen C0-Full anchor: {anchor_path}")
        exact_offsets = np.full(len(raw.patient_ids), np.nan, dtype=np.float64)
        exact_offsets[test] = anchor_test
        s0 = {"variant": "S0", "seed": seed, "outer_fold": fold, "n_test": len(test), "events_test": int(raw.dss_event[test].sum()), **_metric(raw, test, train, anchor_test), "params": 0, "status": "frozen_exact_c0"}
        rows.append(s0)
        np.save(output / "cells" / f"S0_{seed}_{fold}.npy", anchor_test)
        crossfit_offsets, crossfit_rows, _lineage = nested_crossfit(raw, train, test, seed, fold, device, cfg, REPAIR_VERSION)
        for audit_row in crossfit_rows:
            audits.append({"seed": seed, "outer_fold": fold, "audit_type": "clinical_crossfit", **audit_row})
        for variant in VARIANTS:
            variant_evidence = evidence
            if variant == "K-SHUFFLE":
                permutation = np.asarray(bank["shuffle_permutation"], dtype=int)
                variant_evidence = evidence[:, permutation, :]
            prediction, profile, diag = _fit_variant(variant, raw, provider, variant_evidence, train, test, fold, seed, device, cfg, crossfit_offsets, exact_offsets)
            profile.update({"concept_bank_sha256": bank["content_sha256"], "crossfit_version": REPAIR_VERSION, "shuffle_mapping": variant == "K-SHUFFLE"})
            row = {"variant": variant, "seed": seed, "outer_fold": fold, "n_test": len(test), "events_test": int(raw.dss_event[test].sum()), **_metric(raw, test, train, prediction), "params": profile["params"], "best_epoch": profile["best_epoch"], "status": "completed"}
            rows.append(row); profiles.append(profile); diagnostics.extend(diag)
            np.save(output / "cells" / f"{variant}_{seed}_{fold}.npy", prediction)
            atomic_json(output / "cells" / f"{variant}_{seed}_{fold}.json", row)
        del evidence_fold, evidence, provider, crossfit_offsets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_csv(output / "fold_metrics.csv", rows)
    atomic_csv(output / "compute_profile.csv", profiles)
    atomic_csv(output / "concept_evidence_audit.csv", audits)
    atomic_csv(output / "concept_diagnostics.csv", diagnostics)
    atomic_json(output / "worker_manifest.json", {
        "phase": "K0",
        "seed": seed,
        "variants": ["S0", *VARIANTS],
        "cells_completed": len(rows),
        "cells_expected": 30,
        "formal_concept_cells": 25,
        "concept_bank_sha256": bank["content_sha256"],
        "execution_sha": _git_sha(repo),
        "source_c0_sha256": source_hashes,
        "clinical_crossfit_version": REPAIR_VERSION,
        "status": "complete",
        "device": str(device),
    })
    print(json.dumps({"seed": seed, "cells": len(rows), "device": str(device)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--source-workers", required=True)
    parser.add_argument("--concept-bank", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


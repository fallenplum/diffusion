from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from src.phase_m2.common import Timer, aggregate_uno, atomic_csv, atomic_json, count_parameters, cox_loss, gpu_memory_gb, harrell_c, set_seed, stable_int, uno_c_index
from src.phase_m2.runner import PathTokenProvider
from src.phase_m5a.prototypes import fit_outer_train_prototypes
from .data import M6AFoldArrays, RawInputs, load_raw_inputs, prepare_path_cache, preprocess_fold_m6a
from .models import M6AModel, build_model


VARIANTS = ("G0", "G1", "G2", "G3", "G4", "G5", "B0", "C0")
MCAT_VARIANTS = ("G0", "G1", "G2", "G3", "G4", "G5")
PERM_VARIANTS = ("G0", "G1", "G3")
DIAG_VARIANTS = ("G2", "G3", "G4")
LABELS = {
    "G0": "MCAT6-AdaptiveDense", "G1": "MCAT6-ProtoDense", "G2": "Hallmark50RNA-ProtoDense",
    "G3": "Hallmark50RC-ProtoDense", "G4": "Shuffled50RC-ProtoDense", "G5": "Hallmark50RC-AdaptiveDense",
    "B0": "CommonOnly-OriginalFusion", "C0": "ClinicalOnly",
}


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    values = np.asarray(indices, dtype=int)
    if rng is not None:
        values = values[rng.permutation(len(values))]
    return [values[i : i + batch_size] for i in range(0, len(values), batch_size)]


def _groups(arrays: M6AFoldArrays, variant: str) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    if variant in {"G0", "G1"}:
        return arrays.mcat_groups, arrays.mcat_widths
    if variant == "G2":
        return arrays.hallmark_rna_groups, tuple((x, 0) for x in arrays.hallmark_rna_widths)
    if variant in {"G3", "G5"}:
        return arrays.hallmark_rc_groups, arrays.hallmark_rc_widths
    if variant == "G4":
        return arrays.shuffled_rc_groups, arrays.shuffled_rc_widths
    raise ValueError(variant)


def _tensor_batch(arrays: M6AFoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, deterministic: bool, epoch: int, variant: str) -> tuple[torch.Tensor, ...]:
    path = torch.from_numpy(provider.batch(indices, epoch=epoch, deterministic=deterministic)).to(device=device, non_blocking=True)
    rna = torch.from_numpy(arrays.rna[indices]).to(device=device, non_blocking=True)
    cna = torch.from_numpy(arrays.cna[indices]).to(device=device, non_blocking=True)
    clinical = torch.from_numpy(arrays.clinical[indices]).to(device=device, non_blocking=True)
    if variant in MCAT_VARIANTS:
        group_values, _ = _groups(arrays, variant)
        return path, rna, cna, clinical, torch.from_numpy(group_values[indices]).to(device=device, non_blocking=True)
    return path, rna, cna, clinical


def _make_model(variant: str, arrays: M6AFoldArrays, centers: np.ndarray, seed: int, fold: int) -> torch.nn.Module:
    set_seed(seed + fold)
    widths = arrays.mcat_widths if variant in {"G0", "G1", "B0", "C0"} else _groups(arrays, variant)[1]
    return build_model(variant, arrays.clinical.shape[1], tuple(widths), centers, seed, fold)


def _forward_epoch(model: torch.nn.Module, arrays: M6AFoldArrays, provider: PathTokenProvider, indices: np.ndarray, times: np.ndarray, events: np.ndarray, device: torch.device, batch_size: int, train: bool, rng: np.random.Generator | None, epoch: int, variant: str) -> tuple[float, np.ndarray | None, torch.Tensor | None]:
    model.train(train)
    risks: list[torch.Tensor] = []
    aux: list[torch.Tensor] = []
    batches = _batches(indices, batch_size, rng if train else None)
    ordered = np.concatenate(batches) if batches else np.empty(0, dtype=int)
    for batch in batches:
        risk, value = model(*_tensor_batch(arrays, provider, batch, device, not train, epoch, variant))
        risks.append(risk)
        aux.append(value)
    if not risks:
        raise RuntimeError("empty fold")
    all_risk = torch.cat(risks)
    loss = cox_loss(torch.tensor(times[ordered], dtype=torch.float32, device=device), torch.tensor(events[ordered], dtype=torch.float32, device=device), all_risk)
    if aux:
        loss = loss + torch.stack(aux).mean()
    return (float(loss.detach().cpu()), None, loss) if train else (float(loss.detach().cpu()), all_risk.detach().cpu().numpy(), None)


def _predict(model: torch.nn.Module, arrays: M6AFoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, batch_size: int, variant: str) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in _batches(indices, batch_size):
            values.append(model(*_tensor_batch(arrays, provider, batch, device, True, 0, variant))[0].detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _predict_paths(model: torch.nn.Module, arrays: M6AFoldArrays, path_values: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int, variant: str) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for pos in _batches(np.arange(len(indices)), batch_size):
            batch = indices[pos]
            path = torch.from_numpy(np.asarray(path_values[pos], dtype=np.float16)).to(device=device)
            rna = torch.from_numpy(arrays.rna[batch]).to(device=device)
            cna = torch.from_numpy(arrays.cna[batch]).to(device=device)
            clinical = torch.from_numpy(arrays.clinical[batch]).to(device=device)
            if variant in MCAT_VARIANTS:
                group_values, _ = _groups(arrays, variant)
                values.append(model(path, rna, cna, clinical, torch.from_numpy(group_values[batch]).to(device=device))[0].detach().cpu().numpy())
            else:
                values.append(model(path, rna, cna, clinical)[0].detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _inner_split(raw: RawInputs, fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    outer = np.flatnonzero(raw.outer_folds != fold)
    strata = np.char.add(np.char.add(raw.cancer_type[outer].astype(str), "_"), raw.dss_event[outer].astype(str))
    try:
        a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold).split(outer, strata))
        return outer[a], outer[b]
    except ValueError:
        order = np.random.default_rng(seed + fold).permutation(outer)
        n = max(1, round(0.15 * len(order)))
        return order[n:], order[:n]


def _fit_fold(variant: str, raw: RawInputs, arrays: M6AFoldArrays, provider: PathTokenProvider, centers: np.ndarray, fold: int, seed: int, device: torch.device, max_epochs: int, patience: int, batch_size: int):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    inner_train, inner_val = _inner_split(raw, fold, seed)
    timer = Timer()
    model = _make_model(variant, arrays, centers, seed, fold).to(device)
    breakdown = {"total_params": count_parameters(model), "representation_params": count_parameters(model), "projector_params": 0, "fusion_head_params": 0}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, best_epoch, wait = -float("inf"), 1, 0
    rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad(set_to_none=True)
        train_loss, _, loss = _forward_epoch(model, arrays, provider, inner_train, raw.dss_time, raw.dss_event, device, batch_size, True, rng, epoch, variant)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        val = _predict(model, arrays, provider, inner_val, device, batch_size, variant)
        score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val)
        score = score if np.isfinite(score) else -train_loss
        if score > best + 1e-5:
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        if wait >= patience:
            break
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = _make_model(variant, arrays, centers, seed, fold).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    outer_train = np.flatnonzero(raw.outer_folds != fold)
    refit_rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max(1, best_epoch) + 1):
        opt.zero_grad(set_to_none=True)
        _, _, loss = _forward_epoch(model, arrays, provider, outer_train, raw.dss_time, raw.dss_event, device, batch_size, True, refit_rng, epoch, variant)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    test_idx = np.flatnonzero(raw.outer_folds == fold)
    pred = _predict(model, arrays, provider, test_idx, device, batch_size, variant)
    profile = {"method": variant, "seed": seed, "outer_fold": fold, **breakdown, "best_epoch": int(best_epoch), "inner_best_harrell_c": float(best), "training_seconds": timer.seconds, "peak_gpu_memory_gb": gpu_memory_gb(device), "outer_refit": "fresh reinitialize same seed full outer train", "epochwise_sampling": True, "status": "completed"}
    del opt
    return pred, profile, model


def _initialization_audit(arrays: M6AFoldArrays, seed: int = 20260925, fold: int = 1) -> dict[str, Any]:
    centers = np.zeros((16, 1536), dtype=np.float32)
    states: dict[str, dict[str, torch.Tensor]] = {}
    widths = {
        "G0": arrays.mcat_widths, "G1": arrays.mcat_widths,
        "G2": tuple((x, 0) for x in arrays.hallmark_rna_widths),
        "G3": arrays.hallmark_rc_widths, "G4": arrays.shuffled_rc_widths, "G5": arrays.hallmark_rc_widths,
    }
    for variant in MCAT_VARIANTS:
        set_seed(seed + fold)
        states[variant] = build_model(variant, arrays.clinical.shape[1], tuple(widths[variant]), centers, seed, fold).state_dict()

    def mismatches(a: str, b: str, excluded: tuple[str, ...] = ()) -> list[str]:
        keys = [k for k in states[a] if not any(k.startswith(prefix) for prefix in excluded)]
        return [k for k in keys if k not in states[b] or not torch.equal(states[a][k].cpu(), states[b][k].cpu())]

    g01 = mismatches("G0", "G1", ("representation.omic_encoder", "representation.prototype_centers"))
    g34 = mismatches("G3", "G4", ("representation.prototype_centers",))
    g35 = mismatches("G3", "G5", ("representation.prototype_centers",))
    g34_params = count_parameters(build_model("G3", arrays.clinical.shape[1], tuple(widths["G3"]), centers, seed, fold)) == count_parameters(build_model("G4", arrays.clinical.shape[1], tuple(widths["G4"]), centers, seed, fold))
    status = not g01 and not g34 and not g35 and g34_params
    return {
        "status": "PASS" if status else "FAIL",
        "same_seed_fold": int(seed),
        "G0_G1_non_omics_mismatches": g01,
        "G3_G4_all_trainable_mismatches": g34,
        "G3_G5_omic_and_downstream_mismatches": g35,
        "G3_G4_parameter_count_identical": bool(g34_params),
        "G3_parameter_count": count_parameters(build_model("G3", arrays.clinical.shape[1], tuple(widths["G3"]), centers, seed, fold)),
        "G4_parameter_count": count_parameters(build_model("G4", arrays.clinical.shape[1], tuple(widths["G4"]), centers, seed, fold)),
        "variant_only_trainable_parameters": [],
    }


def _control_initialization_audit(clinical_dim: int, widths: tuple[tuple[int, int], ...], fold: int = 1) -> dict[str, Any]:
    centers = np.zeros((16, 1536), dtype=np.float32)
    same: dict[str, bool] = {}
    different: dict[str, bool] = {}
    for method in ("B0", "C0"):
        set_seed(991 + fold); a = build_model(method, clinical_dim, widths, centers, 20260925, fold).state_dict()
        set_seed(991 + fold); b = build_model(method, clinical_dim, widths, centers, 20260925, fold).state_dict()
        set_seed(991 + fold); c = build_model(method, clinical_dim, widths, centers, 20260926, fold).state_dict()
        same[method] = all(k in b and torch.equal(v.cpu(), b[k].cpu()) for k, v in a.items())
        different[method] = any(k in c and not torch.equal(v.cpu(), c[k].cpu()) for k, v in a.items())
    return {"status": "PASS" if all(same.values()) and all(different.values()) else "FAIL", "same_seed_fold_reproducible": same, "different_seeds_change_initialization": different, "seed_rule": "deterministic_hash(method, seed, fold)"}


def _pathway_diagnostics(model: M6AModel, arrays: M6AFoldArrays, provider: PathTokenProvider, test_idx: np.ndarray, device: torch.device, batch_size: int, variant: str, seed: int, fold: int) -> list[dict[str, Any]]:
    model.eval()
    names = arrays.shuffled_group_names if variant == "G4" else arrays.group_names
    norm_sum = np.zeros(len(names), dtype=float)
    norm_sq = np.zeros(len(names), dtype=float)
    pool_sum = np.zeros(len(names), dtype=float)
    pool_sq = np.zeros(len(names), dtype=float)
    entropies: list[float] = []
    n = 0
    with torch.no_grad():
        for batch in _batches(test_idx, batch_size):
            path = torch.from_numpy(provider.batch(batch, epoch=0, deterministic=True)).to(device=device)
            clinical = torch.from_numpy(arrays.clinical[batch]).to(device=device)
            group_values, _ = _groups(arrays, variant)
            groups = torch.from_numpy(group_values[batch]).to(device=device)
            diag = model.pathway_diagnostics(path, clinical, groups)
            norms = diag["omic_token_norm"].detach().cpu().numpy()
            pools = diag["pool_weights"].detach().cpu().numpy()
            entropies.extend(diag["attention_entropy"].detach().cpu().numpy().tolist())
            norm_sum += norms.sum(axis=0); norm_sq += (norms ** 2).sum(axis=0)
            pool_sum += pools.sum(axis=0); pool_sq += (pools ** 2).sum(axis=0)
            n += norms.shape[0]
    norm_mean = norm_sum / max(1, n); pool_mean = pool_sum / max(1, n)
    norm_std = np.sqrt(np.maximum(0, norm_sq / max(1, n) - norm_mean ** 2)); pool_std = np.sqrt(np.maximum(0, pool_sq / max(1, n) - pool_mean ** 2))
    top5 = set(np.argsort(-pool_mean)[:5].tolist())
    entropy = float(np.mean(entropies)) if entropies else float("nan")
    return [{"method": variant, "seed": seed, "outer_fold": fold, "token_index": i, "token_name": names[i], "token_norm_mean": float(norm_mean[i]), "token_norm_std": float(norm_std[i]), "pooling_weight_mean": float(pool_mean[i]), "pooling_weight_std": float(pool_std[i]), "top5_pooled": bool(i in top5), "attention_entropy": entropy} for i in range(len(names))]


def _permutation_audit(model: torch.nn.Module, arrays: M6AFoldArrays, provider: PathTokenProvider, raw: RawInputs, test_idx: np.ndarray, device: torch.device, batch_size: int, variant: str, seed: int, fold: int):
    original = np.asarray(provider.deterministic[test_idx], dtype=np.float16)
    preds = [_predict_paths(model, arrays, original, test_idx, device, batch_size, variant)]
    for p in range(1, 6):
        perm = original.copy()
        for row, patient in enumerate(test_idx.tolist()):
            perm[row] = perm[row, np.random.default_rng(stable_int(f"m6a-permutation:{seed}:{fold}:{patient}:{p}")).permutation(perm.shape[1])]
        preds.append(_predict_paths(model, arrays, perm, test_idx, device, batch_size, variant))
    mat = np.stack(preds, axis=1)
    std = np.std(mat, axis=1); diff = np.max(np.abs(mat - mat[:, :1]), axis=1)
    rows = [{"row_type": "patient", "method": variant, "seed": seed, "outer_fold": fold, "patient_index": int(patient), "permutation_count": 5, "prediction_std": float(s), "max_abs_prediction_difference": float(d)} for patient, s, d in zip(test_idx, std, diff)]
    cvals = [harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], x) for x in preds]
    row: dict[str, Any] = {"row_type": "summary", "method": variant, "seed": seed, "outer_fold": fold, "patient_index": "", "permutation_count": 5, "prediction_std_mean": float(std.mean()), "prediction_std_p95": float(np.quantile(std, .95)), "prediction_std_max": float(std.max()), "max_abs_prediction_difference": float(diff.max()), "invariance_pass": bool(diff.max() <= 1e-5) if variant in {"G1", "G3"} else "diagnostic", "harrell_c_original": float(cvals[0]), "max_c_index_change": float(np.max(np.abs(np.asarray(cvals[1:]) - cvals[0])))}
    row.update({f"harrell_c_perm{i}": float(cvals[i]) for i in range(1, 6)})
    rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve(); data_value = os.environ.get(args.data_root_env)
    if not data_value:
        raise RuntimeError(f"{args.data_root_env} is not set")
    data_root = Path(data_value).resolve(); runtime = Path(args.runtime_root).resolve(); reports = Path(args.reports_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True, warn_only=True)
    raw = load_raw_inputs(data_root, repo_root)
    deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r"); train_pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r")
    cache_meta = prepare_path_cache(raw, runtime, k=1024, pool_k=2048)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    audit_arrays = preprocess_fold_m6a(raw, train_pool, 1)
    init_audit = _initialization_audit(audit_arrays)
    control_audit = _control_initialization_audit(audit_arrays.clinical.shape[1], audit_arrays.mcat_widths)
    atomic_json(reports / "initialization_audit.json", {**init_audit, "execution_commit": _git_revision(repo_root)})
    atomic_json(reports / "control_initialization_audit.json", control_audit)
    if init_audit["status"] != "PASS" or control_audit["status"] != "PASS":
        raise RuntimeError("initialization audit failed")
    requested = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(VARIANTS))
    if unknown:
        raise ValueError(unknown)
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) != 1:
        raise ValueError("one seed per worker")
    seed = seeds[0]; state_path = runtime / "run_state.json"; state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": [], "failed": []}; cells = runtime / "cells"; cells.mkdir(exist_ok=True)
    fold_rows: list[dict[str, Any]] = []; seed_rows: list[dict[str, Any]] = []; profiles: list[dict[str, Any]] = []; prototype_rows: list[dict[str, Any]] = []; group_rows: list[dict[str, Any]] = []; shuffle_audits: list[dict[str, Any]] = []; pathway_rows: list[dict[str, Any]] = []; perm_rows: list[dict[str, Any]] = []; arrays_cache: dict[int, M6AFoldArrays] = {}; proto_cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}; started = time.perf_counter()
    for variant in requested:
        oof = np.full(len(raw.patient_ids), np.nan); variant_rows: list[dict[str, Any]] = []
        for fold in range(1, 6):
            cell = f"{variant}|{seed}|{fold}"; stem = cell.replace("|", "_"); test_idx = np.flatnonzero(raw.outer_folds == fold)
            if fold not in arrays_cache:
                arrays_cache[fold] = preprocess_fold_m6a(raw, train_pool, fold)
                group_rows.extend(arrays_cache[fold].audit["hallmark_group_audit_rows"])
                shuffle_audits.append(arrays_cache[fold].audit["shuffled_pathway_audit"])
            if fold not in proto_cache:
                outer_train = np.flatnonzero(raw.outer_folds != fold); proto_cache[fold] = fit_outer_train_prototypes(train_pool, outer_train, fold); prototype_rows.append({"outer_fold": fold, **proto_cache[fold][1]})
            arrays = arrays_cache[fold]; centers = proto_cache[fold][0]; provider = PathTokenProvider(train_pool=train_pool, deterministic=deterministic, train_seed=seed, k=1024)
            if cell in state.get("completed", []) and (cells / f"{stem}.json").exists() and (cells / f"{stem}.npy").exists():
                payload = json.loads((cells / f"{stem}.json").read_text()); row, profile, pred = payload["row"], payload["profile"], np.load(cells / f"{stem}.npy")
                pathway_rows.extend(payload.get("pathway_rows", [])); perm_rows.extend(payload.get("perm_rows", []))
            else:
                try:
                    pred, profile, model = _fit_fold(variant, raw, arrays, provider, centers, fold, seed, device, args.max_epochs, args.patience, args.batch_size)
                    train_idx = np.flatnonzero(raw.outer_folds != fold); uno, num, den = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], pred, raw.dss_time[train_idx], raw.dss_event[train_idx])
                    row = {"method": variant, "label": LABELS[variant], "seed": seed, "outer_fold": fold, "n_test": len(test_idx), "events_test": int(raw.dss_event[test_idx].sum()), "harrell_c": harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred), "uno_c": uno, "uno_numerator": num, "uno_denominator": den, "params": profile["total_params"], "status": "completed"}
                    np.save(cells / f"{stem}.npy", pred)
                    cell_pathway_rows = _pathway_diagnostics(model, arrays, provider, test_idx, device, args.batch_size, variant, seed, fold) if variant in DIAG_VARIANTS else []
                    cell_perm_rows = _permutation_audit(model, arrays, provider, raw, test_idx, device, args.batch_size, variant, seed, fold) if variant in PERM_VARIANTS else []
                    pathway_rows.extend(cell_pathway_rows); perm_rows.extend(cell_perm_rows)
                    atomic_json(cells / f"{stem}.json", {"row": row, "profile": profile, "pathway_rows": cell_pathway_rows, "perm_rows": cell_perm_rows}); state.setdefault("completed", []).append(cell); atomic_json(state_path, state); del model; gc.collect()
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as exc:
                    state.setdefault("failed", []).append({"cell": cell, "error": repr(exc)}); atomic_json(state_path, state); raise
            fold_rows.append(row); variant_rows.append(row); profiles.append(profile); oof[test_idx] = pred
        h = np.asarray([float(x["harrell_c"]) for x in variant_rows]); u = np.asarray([float(x["uno_c"]) for x in variant_rows])
        seed_rows.append({"method": variant, "label": LABELS[variant], "seed": seed, "n_patients": len(raw.patient_ids), "dss_events": int(raw.dss_event.sum()), "fold_mean_harrell": float(np.mean(h)), "fold_std_harrell": float(np.std(h, ddof=1)), "worst_fold_harrell": float(np.min(h)), "fold_mean_uno": float(np.mean(u)), "fold_std_uno": float(np.std(u, ddof=1)), "worst_fold_uno": float(np.min(u)), "global_oof_harrell_diagnostic": harrell_c(raw.dss_time, raw.dss_event, oof) if np.isfinite(oof).all() else float("nan"), "global_oof_uno_diagnostic": aggregate_uno(variant_rows), "status": "completed"})
    atomic_csv(reports / "fold_metrics.csv", fold_rows); atomic_csv(reports / "seed_metrics.csv", seed_rows); atomic_csv(reports / "compute_profile.csv", profiles); atomic_csv(reports / "prototype_audit.csv", prototype_rows); atomic_csv(reports / "hallmark_group_audit.csv", group_rows); atomic_csv(reports / "pathway_attention_diagnostics.csv", pathway_rows); atomic_csv(reports / "permutation_invariance_audit.csv", perm_rows); atomic_json(reports / "shuffled_pathway_audit.json", {"phase": "M6A", "folds": shuffle_audits, "outcome_blind": True, "size_matched": True}); atomic_json(reports / "run_manifest.json", {"phase": "M6A", "status": "worker_completed", "execution_commit": _git_revision(repo_root), "variants": requested, "seeds": seeds, "cells_completed": len(fold_rows), "device": str(device), "cache": cache_meta, "n_patients": int(len(raw.patient_ids)), "dss_events": int(raw.dss_event.sum()), "pathology_K": 1024, "prototype_fit": "outer-train-only train_pool patches; M5A/M5R unchanged", "no_leakage": True, "no_seed_hunting": True, "initialization_audit_status": init_audit["status"], "control_initialization_audit_status": control_audit["status"]}); print(json.dumps({"status": "worker_completed", "seed": seed, "cells": len(fold_rows), "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False)); return {"status": "worker_completed", "cells": len(fold_rows)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--data-root-env", default="DIFFUSION_DATA_ROOT"); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--reports-root", type=Path, required=True); p.add_argument("--variants", default=",".join(VARIANTS)); p.add_argument("--seeds", required=True); p.add_argument("--max-epochs", type=int, default=25); p.add_argument("--patience", type=int, default=5); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--cpu", action="store_true"); return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

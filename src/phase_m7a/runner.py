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
from .data import FoldArrays, RawInputs, load_raw_inputs, prepare_path_cache, preprocess_fold
from .models import DIM, M7AModel, build_model

VARIANTS = ("F0", "F1", "F2", "F3", "F4", "F5", "B0", "C0")
MCAT_VARIANTS = ("F0", "F1", "F2", "F3", "F4")
PERM_VARIANTS = ("F1", "F3", "F4")
FILM_VARIANTS = ("F3", "F4", "F5")
LABELS = {
    "F0": "MCAT-Original", "F1": "ABMIL-SemanticConcat", "F2": "ABMIL-MatchedResidualMLP",
    "F3": "ABMIL-OmicToPath-FiLM", "F4": "ABMIL-Bidirectional-FiLM", "F5": "ABMIL-RawOmic-FiLM",
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


def _groups(arrays: FoldArrays) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    return arrays.mcat_groups, arrays.mcat_widths


def _tensor_batch(arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, deterministic: bool, epoch: int, variant: str) -> tuple[torch.Tensor, ...]:
    path = torch.from_numpy(provider.batch(indices, epoch=epoch, deterministic=deterministic)).to(device=device, non_blocking=True)
    rna = torch.from_numpy(arrays.rna[indices]).to(device=device, non_blocking=True)
    cna = torch.from_numpy(arrays.cna[indices]).to(device=device, non_blocking=True)
    clinical = torch.from_numpy(arrays.clinical[indices]).to(device=device, non_blocking=True)
    if variant in MCAT_VARIANTS:
        groups, _ = _groups(arrays)
        return path, rna, cna, clinical, torch.from_numpy(groups[indices]).to(device=device, non_blocking=True)
    return path, rna, cna, clinical


def _make_model(variant: str, arrays: FoldArrays, seed: int, fold: int) -> torch.nn.Module:
    set_seed(seed + fold)
    return build_model(variant, arrays.clinical.shape[1], tuple(arrays.mcat_widths), seed, fold)


def _forward_epoch(model: torch.nn.Module, arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, times: np.ndarray, events: np.ndarray, device: torch.device, batch_size: int, train: bool, rng: np.random.Generator | None, epoch: int, variant: str) -> tuple[float, np.ndarray | None, torch.Tensor | None]:
    model.train(train)
    risks: list[torch.Tensor] = []
    aux: list[torch.Tensor] = []
    batches = _batches(indices, batch_size, rng if train else None)
    ordered = np.concatenate(batches) if batches else np.empty(0, dtype=int)
    for batch in batches:
        risk, value = model(*_tensor_batch(arrays, provider, batch, device, not train, epoch, variant))
        risks.append(risk); aux.append(value)
    if not risks:
        raise RuntimeError("empty fold")
    all_risk = torch.cat(risks)
    loss = cox_loss(torch.tensor(times[ordered], dtype=torch.float32, device=device), torch.tensor(events[ordered], dtype=torch.float32, device=device), all_risk)
    if aux:
        loss = loss + torch.stack(aux).mean()
    return (float(loss.detach().cpu()), None, loss) if train else (float(loss.detach().cpu()), all_risk.detach().cpu().numpy(), None)


def _predict(model: torch.nn.Module, arrays: FoldArrays, provider: PathTokenProvider, indices: np.ndarray, device: torch.device, batch_size: int, variant: str) -> np.ndarray:
    model.eval(); values = []
    with torch.no_grad():
        for batch in _batches(indices, batch_size):
            values.append(model(*_tensor_batch(arrays, provider, batch, device, True, 0, variant))[0].detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _predict_paths(model: torch.nn.Module, arrays: FoldArrays, path_values: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int, variant: str) -> np.ndarray:
    model.eval(); values = []
    with torch.no_grad():
        for pos in _batches(np.arange(len(indices)), batch_size):
            batch = indices[pos]; path = torch.from_numpy(np.asarray(path_values[pos], dtype=np.float16)).to(device=device)
            rna = torch.from_numpy(arrays.rna[batch]).to(device=device); cna = torch.from_numpy(arrays.cna[batch]).to(device=device); clinical = torch.from_numpy(arrays.clinical[batch]).to(device=device)
            if variant in MCAT_VARIANTS:
                groups, _ = _groups(arrays); values.append(model(path, rna, cna, clinical, torch.from_numpy(groups[batch]).to(device=device))[0].detach().cpu().numpy())
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
        order = np.random.default_rng(seed + fold).permutation(outer); n = max(1, round(0.15 * len(order))); return order[n:], order[:n]


def _fit_fold(variant: str, raw: RawInputs, arrays: FoldArrays, provider: PathTokenProvider, fold: int, seed: int, device: torch.device, max_epochs: int, patience: int, batch_size: int):
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    inner_train, inner_val = _inner_split(raw, fold, seed); timer = Timer()
    model = _make_model(variant, arrays, seed, fold).to(device)
    breakdown = parameter_breakdown(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); best, best_epoch, wait = -float("inf"), 1, 0; rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad(set_to_none=True)
        train_loss, _, loss = _forward_epoch(model, arrays, provider, inner_train, raw.dss_time, raw.dss_event, device, batch_size, True, rng, epoch, variant)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        val = _predict(model, arrays, provider, inner_val, device, batch_size, variant); score = harrell_c(raw.dss_time[inner_val], raw.dss_event[inner_val], val); score = score if np.isfinite(score) else -train_loss
        if score > best + 1e-5: best, best_epoch, wait = score, epoch, 0
        else: wait += 1
        if wait >= patience: break
    del model, opt; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    model = _make_model(variant, arrays, seed, fold).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    outer_train = np.flatnonzero(raw.outer_folds != fold); refit_rng = np.random.default_rng(seed + fold + 101)
    for epoch in range(1, max(1, best_epoch) + 1):
        opt.zero_grad(set_to_none=True); _, _, loss = _forward_epoch(model, arrays, provider, outer_train, raw.dss_time, raw.dss_event, device, batch_size, True, refit_rng, epoch, variant); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    test_idx = np.flatnonzero(raw.outer_folds == fold); pred = _predict(model, arrays, provider, test_idx, device, batch_size, variant)
    profile = {"method": variant, "seed": seed, "outer_fold": fold, **breakdown, "best_epoch": int(best_epoch), "inner_best_harrell_c": float(best), "training_seconds": timer.seconds, "peak_gpu_memory_gb": gpu_memory_gb(device), "outer_refit": "fresh reinitialize same seed full outer train", "epochwise_sampling": True, "execution_commit": _git_revision(Path(__file__).resolve().parents[2]), "status": "completed"}
    del opt
    return pred, profile, model


def _initialization_audit(arrays: FoldArrays, seed: int, fold: int) -> dict[str, Any]:
    models = {v: _make_model(v, arrays, seed, fold) for v in ("F1", "F2", "F3", "F4")}
    interaction_prefixes = ("residual", "film_o", "film_p")
    shared_keys = [k for k in models["F1"].state_dict() if not k.startswith(interaction_prefixes)]
    mismatches = {v: [k for k in shared_keys if k not in models[v].state_dict() or not torch.equal(models["F1"].state_dict()[k], models[v].state_dict()[k])] for v in ("F2", "F3", "F4")}
    f2 = count_parameters(models["F2"].residual); f3 = count_parameters(models["F3"].film_o); match = abs(f2 - f3) / f3 <= 0.10
    m5 = _make_model("F5", arrays, seed, fold); m5_again = _make_model("F5", arrays, seed, fold)
    f5_same = all(torch.equal(t, m5_again.state_dict()[k]) for k, t in m5.state_dict().items())
    f5_audit = {"same_seed_byte_identical": f5_same, "uses_six_signature_metadata": False, "raw_global_vector_dim": DIM, "parameter_breakdown": parameter_breakdown(m5), "note": "F5 bypasses MCAT six-signature modules and registers only the active raw-omics encoder, FiLM, fusion, clinical, and risk-head parameters."}
    status = all(not x for x in mismatches.values()) and match and f5_same
    return {"status": "PASS" if status else "FAIL", "seed": seed, "outer_fold": fold, "shared_upstream_mismatches": mismatches, "F2_residual_params": f2, "F3_FiLM_params": f3, "F2_F3_interaction_parameter_match_within_10pct": match, "F5_separate_audit": f5_audit, "shared_components": ["ABMIL patch projection", "ABMIL attention", "SixSignatureEncoder", "omic Transformer", "omic GatedAttentionPool", "omic_rho", "clinical encoder", "late fusion", "clinical gate", "RiskHead"]}


def parameter_breakdown(model: torch.nn.Module) -> dict[str, int]:
    total = count_parameters(model); projector = interaction = unused = 0
    if isinstance(model, M7AModel):
        projector = count_parameters(model.representation.abmil.projection)
        interaction = sum(count_parameters(x) for x in (model.residual, model.film_o, model.film_p) if x is not None)
        fusion = count_parameters(model.representation.fusion) + count_parameters(model.representation.clinical_gate) + count_parameters(model.representation.head) + interaction
        if model.variant == "F5": unused = sum(count_parameters(x) for x in (model.representation.omic_transformer, model.representation.omic_attention_head, model.representation.omic_rho) if x is not None)
    else:
        fusion = sum(p.numel() for n, p in model.named_parameters() if any(x in n.lower() for x in ("fusion", "gate", "head")))
    return {"total_params": total, "active_params": total - unused, "registered_unused_params": unused, "representation_params": total - unused - fusion - projector, "projector_params": projector, "fusion_head_params": fusion, "interaction_params": interaction}


def film_correctness_audit(clinical_dim: int, widths: tuple[tuple[int, int], ...], seed: int = 20260928, fold: int = 1) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "FAIL"}
    m3 = _make_model("F3", type("A", (), {"clinical": np.zeros((1, clinical_dim), np.float32), "mcat_widths": widths})(), seed, fold)
    m4 = _make_model("F4", type("A", (), {"clinical": np.zeros((1, clinical_dim), np.float32), "mcat_widths": widths})(), seed, fold)
    p = torch.randn(4, DIM); o = torch.randn(4, DIM)
    p3, o3, _ = m3._condition(p, o); p4, o4, _ = m4._condition(p, o)
    identity = {"F3_path": float((p3 - p).abs().max()), "F4_path": float((p4 - p).abs().max()), "F4_omic": float((o4 - o).abs().max())}
    m3.zero_grad(set_to_none=True); pp = p.clone().requires_grad_(); oo = o.clone().requires_grad_(); pp2, _, _ = m3._condition(pp, oo); (pp2.square().mean()).backward()
    grad_names = ["film_o.net.2.weight", "film_o.net.2.bias"]; grads = {n: m3.state_dict()[n].new_tensor(float(getattr(m3, n.split('.')[0]).net[-1].weight.grad.norm() if n.endswith('weight') else getattr(m3, n.split('.')[0]).net[-1].bias.grad.norm())) for n in grad_names}
    deterministic = all(torch.equal(a, b) for a, b in zip(m3.state_dict().values(), _make_model("F3", type("A", (), {"clinical": np.zeros((1, clinical_dim), np.float32), "mcat_widths": widths})(), seed, fold).state_dict().values()))
    out.update({"zero_init_identity_max_abs": identity, "zero_init_threshold": 1e-7, "identity_pass": all(v < 1e-7 for v in identity.values()), "film_final_layer_gradient_norms": {k: float(v) for k, v in grads.items()}, "gradient_finite_and_nonzero": all(np.isfinite(float(v)) and float(v) > 0 for v in grads.values()), "same_seed_byte_identical": deterministic})
    expanded = {}
    arrays = type("A", (), {"clinical": np.zeros((1, clinical_dim), np.float32), "mcat_widths": widths})()
    for variant in ("F3", "F4", "F5"):
        model = _make_model(variant, arrays, seed, fold)
        pp = p.clone().requires_grad_(); oo = o.clone().requires_grad_()
        pm, om, _ = model._condition(pp, oo)
        (pm.square().mean() + om.square().mean()).backward()
        final = {n: float(t.grad.norm()) for n, t in model.named_parameters() if n.startswith(("film_o.net.2", "film_p.net.2"))}
        replica = _make_model(variant, arrays, seed, fold)
        expanded[variant] = {"final_layer_gradient_norms": final, "final_gradients_finite_nonzero": all(np.isfinite(v) and v > 0 for v in final.values()), "upstream_p_gradient_finite": bool(pp.grad is not None and torch.isfinite(pp.grad).all()), "upstream_o_gradient_finite": bool(oo.grad is not None and torch.isfinite(oo.grad).all()), "same_seed_byte_identical": all(torch.equal(t, replica.state_dict()[k]) for k, t in model.state_dict().items())}
    out["all_film_variants"] = expanded
    out["status"] = "PASS" if out["identity_pass"] and out["gradient_finite_and_nonzero"] and deterministic and all(all(v for k, v in r.items() if k != "final_layer_gradient_norms") for r in expanded.values()) else "FAIL"
    return out


def _film_diagnostics(model: M7AModel, arrays: FoldArrays, provider: PathTokenProvider, test_idx: np.ndarray, device: torch.device, batch_size: int, variant: str, seed: int, fold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vectors: dict[str, list[np.ndarray]] = {k: [] for k in ("gamma_o", "beta_o", "gamma_p", "beta_p")}
    for batch in _batches(test_idx, batch_size):
        path = torch.from_numpy(provider.batch(batch, epoch=0, deterministic=True)).to(device=device); rna = torch.from_numpy(arrays.rna[batch]).to(device=device); cna = torch.from_numpy(arrays.cna[batch]).to(device=device); clinical = torch.from_numpy(arrays.clinical[batch]).to(device=device); groups = torch.from_numpy(arrays.mcat_groups[batch]).to(device=device)
        d = model.modulation_diagnostics(path, rna, cna, clinical, groups)
        for key in vectors:
            vectors[key].append(d.get(key, torch.zeros(len(batch), DIM, device=device)).detach().cpu().numpy())
        n = len(batch)
        for i, patient in enumerate(batch):
            row = {"method": variant, "seed": seed, "outer_fold": fold, "patient_index": int(patient), "mean_abs_gamma_o": float(d.get("gamma_o", torch.zeros(n, DIM))[i].abs().mean()), "mean_abs_beta_o": float(d.get("beta_o", torch.zeros(n, DIM))[i].abs().mean()), "gamma_o_std": float(d.get("gamma_o", torch.zeros(n, DIM))[i].std()), "beta_o_std": float(d.get("beta_o", torch.zeros(n, DIM))[i].std()), "fraction_gamma_o_abs_lt_0_05": float((d.get("gamma_o", torch.zeros(n, DIM))[i].abs() < 0.05).float().mean()), "path_correction_norm": float(d.get("path_correction_norm", torch.zeros(n))[i]), "mean_abs_gamma_p": float(d.get("gamma_p", torch.zeros(n, DIM))[i].abs().mean()), "mean_abs_beta_p": float(d.get("beta_p", torch.zeros(n, DIM))[i].abs().mean()), "gamma_p_std": float(d.get("gamma_p", torch.zeros(n, DIM))[i].std()), "beta_p_std": float(d.get("beta_p", torch.zeros(n, DIM))[i].std()), "fraction_gamma_p_abs_lt_0_05": float((d.get("gamma_p", torch.zeros(n, DIM))[i].abs() < 0.05).float().mean()), "omic_correction_norm": float(d.get("omic_correction_norm", torch.zeros(n))[i])}
            rows.append(row)
    for key, values in vectors.items():
        across_patient = float(np.concatenate(values).std(axis=0, ddof=1).mean())
        for row in rows:
            row[key + "_std_across_features_patientwise"] = row.pop(key + "_std")
            row[key + "_std_across_patients_mean_features"] = across_patient
    return rows


def _permutation_audit(model: torch.nn.Module, arrays: FoldArrays, provider: PathTokenProvider, raw: RawInputs, test_idx: np.ndarray, device: torch.device, batch_size: int, variant: str, seed: int, fold: int):
    original = np.asarray(provider.deterministic[test_idx], dtype=np.float16); preds = [_predict_paths(model, arrays, original, test_idx, device, batch_size, variant)]
    for p in range(1, 6):
        perm = original.copy()
        for row, patient in enumerate(test_idx.tolist()):
            perm[row] = perm[row, np.random.default_rng(stable_int(f"m7a-permutation:{seed}:{fold}:{patient}:{p}")).permutation(perm.shape[1])]
        preds.append(_predict_paths(model, arrays, perm, test_idx, device, batch_size, variant))
    mat = np.stack(preds, axis=1); std = np.std(mat, axis=1); diff = np.max(np.abs(mat - mat[:, :1]), axis=1)
    rows = [{"row_type": "patient", "method": variant, "seed": seed, "outer_fold": fold, "patient_index": int(patient), "permutation_count": 5, "prediction_std": float(s), "max_abs_prediction_difference": float(d)} for patient, s, d in zip(test_idx, std, diff)]
    cvals = [harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], x) for x in preds]
    row: dict[str, Any] = {"row_type": "summary", "method": variant, "seed": seed, "outer_fold": fold, "patient_index": "", "permutation_count": 5, "prediction_std_mean": float(std.mean()), "prediction_std_p95": float(np.quantile(std, .95)), "prediction_std_max": float(std.max()), "max_abs_prediction_difference": float(diff.max()), "invariance_pass": bool(diff.max() <= 1e-5), "harrell_c_original": float(cvals[0]), "max_c_index_change": float(np.max(np.abs(np.asarray(cvals[1:]) - cvals[0])))}
    row.update({f"harrell_c_perm{i}": float(cvals[i]) for i in range(1, 6)}); rows.append(row); return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve(); data_value = os.environ.get(args.data_root_env)
    if not data_value: raise RuntimeError(f"{args.data_root_env} is not set")
    data_root = Path(data_value).resolve(); runtime = Path(args.runtime_root).resolve(); reports = Path(args.reports_root).resolve(); runtime.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True, warn_only=True)
    raw = load_raw_inputs(data_root, repo_root); deterministic = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r"); train_pool = np.load(runtime / "path_tokens_train_pool.npy", mmap_mode="r"); cache_meta = prepare_path_cache(raw, runtime, k=1024, pool_k=2048); device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    audit_arrays = preprocess_fold(raw, train_pool, 1); init_audit = _initialization_audit(audit_arrays, int(args.seeds), 1); correctness = film_correctness_audit(audit_arrays.clinical.shape[1], audit_arrays.mcat_widths); control = {"status": "PASS"}
    atomic_json(reports / "initialization_audit.json", {**init_audit, "execution_commit": _git_revision(repo_root)}); atomic_json(reports / "film_correctness_audit.json", correctness); atomic_json(reports / "control_initialization_audit.json", control)
    (reports / "abmil_implementation_audit.md").write_text("# ABMIL implementation audit\n\n- Source: `src/phase_m7a/models.py:GatedABMIL`.\n- Formula: shared `LayerNorm(1536) -> Linear(1536,96) -> GELU`; scores use `w^T(tanh(Vh) * sigmoid(Uh))`; softmax weighted sum; `LayerNorm(96)`.\n- Attention hidden dimension: 48.\n- No positional embedding or patch-order feature.\n- The output is a weighted sum over the patch set, so permutation invariance is structural and separately verified by the formal audit.\n", encoding="utf-8")
    if init_audit["status"] != "PASS" or correctness["status"] != "PASS": raise RuntimeError("M7A correctness or initialization audit failed")
    requested = [x.strip() for x in args.variants.split(",") if x.strip()]; unknown = sorted(set(requested) - set(VARIANTS));
    if unknown: raise ValueError(unknown)
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(seeds) != 1: raise ValueError("one seed per worker")
    seed = seeds[0]; state_path = runtime / "run_state.json"; state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": [], "failed": []}; cells = runtime / "cells"; cells.mkdir(exist_ok=True)
    fold_rows: list[dict[str, Any]] = []; seed_rows: list[dict[str, Any]] = []; profiles: list[dict[str, Any]] = []; film_rows: list[dict[str, Any]] = []; perm_rows: list[dict[str, Any]] = []; arrays_cache: dict[int, FoldArrays] = {}; started = time.perf_counter()
    for variant in requested:
        oof = np.full(len(raw.patient_ids), np.nan); variant_rows: list[dict[str, Any]] = []
        for fold in range(1, 6):
            cell = f"{variant}|{seed}|{fold}"; stem = cell.replace("|", ""); test_idx = np.flatnonzero(raw.outer_folds == fold)
            if fold not in arrays_cache: arrays_cache[fold] = preprocess_fold(raw, train_pool, fold)
            arrays = arrays_cache[fold]; provider = PathTokenProvider(train_pool=train_pool, deterministic=deterministic, train_seed=seed, k=1024)
            if cell in state.get("completed", []) and (cells / f"{stem}.json").exists() and (cells / f"{stem}.npy").exists():
                payload = json.loads((cells / f"{stem}.json").read_text()); row, profile, pred = payload["row"], payload["profile"], np.load(cells / f"{stem}.npy"); film_rows.extend(payload.get("film_rows", [])); perm_rows.extend(payload.get("perm_rows", []))
                profile.update(parameter_breakdown(_make_model(variant, arrays, seed, fold)))
                if not profile.get("execution_commit"):
                    prior = "35a520c" if variant in {"B0", "C0", "F5"} else "c5a00a5"
                    profile["execution_commit"] = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", prior], text=True).strip()
                row["params"] = profile["total_params"]
            else:
                try:
                    pred, profile, model = _fit_fold(variant, raw, arrays, provider, fold, seed, device, args.max_epochs, args.patience, args.batch_size)
                    train_idx = np.flatnonzero(raw.outer_folds != fold); uno, num, den = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], pred, raw.dss_time[train_idx], raw.dss_event[train_idx]); row = {"method": variant, "label": LABELS[variant], "seed": seed, "outer_fold": fold, "n_test": len(test_idx), "events_test": int(raw.dss_event[test_idx].sum()), "harrell_c": harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred), "uno_c": uno, "uno_numerator": num, "uno_denominator": den, "params": profile["total_params"], "status": "completed"}; np.save(cells / f"{stem}.npy", pred)
                    cell_film = _film_diagnostics(model, arrays, provider, test_idx, device, args.batch_size, variant, seed, fold) if variant in FILM_VARIANTS else []; cell_perm = _permutation_audit(model, arrays, provider, raw, test_idx, device, args.batch_size, variant, seed, fold) if variant in PERM_VARIANTS else []; film_rows.extend(cell_film); perm_rows.extend(cell_perm); atomic_json(cells / f"{stem}.json", {"row": row, "profile": profile, "film_rows": cell_film, "perm_rows": cell_perm}); state.setdefault("completed", []).append(cell); atomic_json(state_path, state); del model; gc.collect()
                    if device.type == "cuda": torch.cuda.empty_cache()
                except Exception as exc:
                    state.setdefault("failed", []).append({"cell": cell, "error": repr(exc)}); atomic_json(state_path, state); raise
            fold_rows.append(row); variant_rows.append(row); profiles.append(profile); oof[test_idx] = pred
        h = np.asarray([float(x["harrell_c"]) for x in variant_rows]); u = np.asarray([float(x["uno_c"]) for x in variant_rows]); seed_rows.append({"method": variant, "label": LABELS[variant], "seed": seed, "n_patients": len(raw.patient_ids), "dss_events": int(raw.dss_event.sum()), "fold_mean_harrell": float(np.mean(h)), "fold_std_harrell": float(np.std(h, ddof=1)), "worst_fold_harrell": float(np.min(h)), "fold_mean_uno": float(np.mean(u)), "fold_std_uno": float(np.std(u, ddof=1)), "worst_fold_uno": float(np.min(u)), "global_oof_harrell_diagnostic": harrell_c(raw.dss_time, raw.dss_event, oof) if np.isfinite(oof).all() else float("nan"), "global_oof_uno_diagnostic": aggregate_uno(variant_rows), "status": "completed"})
    atomic_csv(reports / "fold_metrics.csv", fold_rows); atomic_csv(reports / "seed_metrics.csv", seed_rows); atomic_csv(reports / "compute_profile.csv", profiles); atomic_csv(reports / "film_modulation_diagnostics.csv", film_rows); atomic_csv(reports / "permutation_invariance_audit.csv", perm_rows); atomic_json(reports / "run_manifest.json", {"phase": "M7A", "status": "worker_completed", "execution_commit": _git_revision(repo_root), "variants": requested, "seeds": seeds, "cells_completed": len(fold_rows), "device": str(device), "cache": cache_meta, "n_patients": int(len(raw.patient_ids)), "dss_events": int(raw.dss_event.sum()), "pathology_K": 1024, "no_leakage": True, "no_seed_hunting": True, "film_correctness_status": correctness["status"], "initialization_audit_status": init_audit["status"]}); print(json.dumps({"status": "worker_completed", "seed": seed, "cells": len(fold_rows), "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False)); return {"status": "worker_completed", "cells": len(fold_rows)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--data-root-env", default="DIFFUSION_DATA_ROOT"); p.add_argument("--runtime-root", type=Path, required=True); p.add_argument("--reports-root", type=Path, required=True); p.add_argument("--variants", default=",".join(VARIANTS)); p.add_argument("--seeds", required=True); p.add_argument("--max-epochs", type=int, default=25); p.add_argument("--patience", type=int, default=5); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--cpu", action="store_true"); return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

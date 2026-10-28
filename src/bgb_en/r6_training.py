"""R6 deterministic full-risk Efron cache and private scalar diagnostics.

No source fits, teacher fits, checkpoint selection or scheduling occurs here.
Only scalar risks/deltas survive the no-grad cache; patch/gene traces are freed
per microbatch. Frozen interventions are dependencies, not retrained controls.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from src.phase10_trimodal.math import torch_cox, log_weights
from .survival import evaluate


def saved_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_hash(state, *, target_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if target_only and (name.startswith("core.foundation.") or
                            name.startswith("core.gene_rereader.gene_encoder.")):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def full_risk_step(model, batches, time, event, optimizer, *, total_weight=.1, cross_weight=.1):
    """One optimizer update exactly representing the entire supplied risk set."""
    if total_weight != .1 or cross_weight != .1:
        raise ValueError("R6_LOCKED_RESIDUAL_PENALTIES_CHANGED")
    if any(isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm))
           for module in model.modules()):
        raise ValueError("R6_CACHE_REQUIRES_DETERMINISTIC_MODULES")
    indices = torch.cat(batches)
    if len(indices) != len(time) or len(time) != len(event) or len(torch.unique(indices)) != len(indices):
        raise ValueError("R6_FULL_RISKSET_SCOPE_CHANGED")
    if hasattr(model, "fit_indices") and not np.array_equal(indices.detach().cpu().numpy(), model.fit_indices):
        raise ValueError("R6_FULL_RISKSET_PATIENT_ORDER_CHANGED")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if getattr(model, "intervention", None) == "mean_query":
        model.refresh_training_query_mean()
    with torch.no_grad():
        pieces = {key: [] for key in ("risk", "delta", "cross_delta")}
        for idx in batches:
            row = model(idx)
            for key in pieces:
                pieces[key].append(row[key].detach())
            del row
        cached = {key: torch.cat(value) for key, value in pieces.items()}
        del pieces
    if any(not torch.isfinite(value).all() for value in cached.values()):
        raise RuntimeError("R6_NONFINITE_CACHED_RISK")
    leaf = cached["risk"].detach().requires_grad_(True)
    cox = torch_cox(leaf, log_weights(time, event, model.device), torch.as_tensor(event, device=model.device))
    if not torch.isfinite(cox):
        raise RuntimeError("R6_NONFINITE_FULL_RISK_COX")
    gradient = torch.autograd.grad(cox, leaf)[0].detach()
    position, maximum, cross_maximum = 0, 0., 0.
    for idx in batches:
        row = model(idx)
        n = len(idx)
        maximum = max(maximum, float((row["risk"].detach() - cached["risk"][position:position+n]).abs().max()))
        cross_maximum = max(cross_maximum, float((row["cross_delta"].detach() - cached["cross_delta"][position:position+n]).abs().max()))
        if maximum > 2e-5 or cross_maximum > 2e-5:
            raise RuntimeError("R6_DETERMINISTIC_CACHE_REPLAY_FAILED")
        penalty = (.1 * row["delta"].square().sum() + .1 * row["cross_delta"].square().sum()) / len(indices)
        (torch.dot(row["risk"], gradient[position:position+n]) + penalty).backward()
        position += n
        del row, penalty
    regularizer = model.regularization()
    if not torch.isfinite(regularizer):
        raise RuntimeError("R6_NONFINITE_PARAMETER_REGULARIZER")
    regularizer.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return dict(cox_loss=float(cox.detach()), parameter_regularizer=float(regularizer.detach()),
        total_risk_penalty=float(.1 * cached["delta"].square().mean()),
        cross_risk_penalty=float(.1 * cached["cross_delta"].square().mean()),
        gradient_norm=float(norm), risk_cache_replay_max=maximum, cross_cache_replay_max=cross_maximum,
        full_riskset_patients=len(indices), optimizer_updates=1)


SCALARS = ("risk", "delta", "main_delta", "cross_delta", "path_main_delta", "gene_main_delta",
           "hallmark_main_delta", "path_cross_delta", "gene_cross_delta")


def predict(model, indices, batch_size=8, control=None):
    model.eval()
    if control == "mean_query":
        model.refresh_training_query_mean()
    pieces = {key: [] for key in SCALARS}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = torch.as_tensor(indices[start:start+batch_size], device=model.device)
            row = model(idx, control=control)
            for key in SCALARS:
                pieces[key].append(row[key].detach().cpu().numpy())
            for key in ("delta_E", "delta_G"):
                pieces.setdefault(key + "_L2", []).append(row[key].detach().flatten(1).norm(dim=-1).cpu().numpy())
            del row
    result = {key: np.concatenate(value).astype(np.float64) for key, value in pieces.items()}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("R6_NONFINITE_PREDICTION")
    return result


def metrics_for(raw, prepared, indices, prediction, *, include_pfi=True):
    train, risk = prepared["train"], prediction["risk"]
    result = {name: evaluate(raw["t"][train], raw["e"][train], raw["t"][indices], raw["e"][indices], risk, ref[indices])
              for name, ref in (("CP_STRONG", prepared["CP"]), ("CP_SCALAR", prepared["base"]))}
    if include_pfi:
        usable = raw["pfi_valid"][indices]
        pfitrain = train[raw["pfi_valid"][train]]
        result["PFI_frozen_DSS_risk"] = evaluate(raw["pfi_t"][pfitrain], raw["pfi_e"][pfitrain],
            raw["pfi_t"][indices][usable], raw["pfi_e"][indices][usable], risk[usable])
    return result


def describe(values):
    x = np.asarray(values, float)
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std()), min=float(x.min()),
                q10=float(np.quantile(x, .1)), median=float(np.median(x)),
                q90=float(np.quantile(x, .9)), max=float(x.max()))


def risk_diagnostics(prediction, offset):
    scale = max(float(np.std(offset)), 1e-12)
    return dict(quantities={key: describe(value) for key, value in prediction.items() if key != "risk"},
                SD_total_delta_over_offset=float(np.std(prediction["delta"]) / scale),
                SD_cross_delta_over_offset=float(np.std(prediction["cross_delta"]) / scale),
                linear_readout_has_no_tanh_saturation_bound=True)


def pure_survival_gradient_probe(model, prepared, raw, batch_size=8):
    """First-microbatch contribution to full-risk Cox, no regularizers or steps."""
    model.eval()
    train = np.asarray(prepared["train"])
    batches = [torch.as_tensor(train[j:j+batch_size], device=model.device) for j in range(0, len(train), batch_size)]
    with torch.no_grad():
        pieces = []
        for idx in batches:
            row = model(idx)
            pieces.append(row["risk"].detach())
            del row
        risk = torch.cat(pieces)
    leaf = risk.detach().requires_grad_(True)
    loss = torch_cox(leaf, log_weights(raw["t"][train], raw["e"][train], model.device),
                     torch.as_tensor(raw["e"][train], device=model.device))
    gradient = torch.autograd.grad(loss, leaf)[0].detach()
    row = model(batches[0])
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    paths = {}
    for path in ("cross_delta", "risk"):
        target = (row[path] * gradient[:len(batches[0])]).sum()
        grads = torch.autograd.grad(target, parameters, allow_unused=True, retain_graph=True) if target.requires_grad else [None] * len(parameters)
        groups = {}
        for (name, _), grad in zip(named, grads):
            group = ".".join(name.split(".")[:2])
            cell = groups.setdefault(group, dict(tensors=0, nonzero=0, L2_squared=0.))
            cell["tensors"] += 1
            if grad is not None:
                cell["nonzero"] += int(bool(torch.any(grad != 0)))
                cell["L2_squared"] += float(grad.square().sum())
        paths[path] = groups
    return dict(scope="first_train_microbatch_contribution_to_full_riskset_Cox_no_regularizers",
                patients=len(batches[0]), groups_by_path=paths, new_optimizer_updates=0)


def frozen_attention_probe(model, indices):
    """Fixed first 8 held-out patients, no diagnosis or tissue truth claim."""
    selected = np.asarray(indices)[:8]
    idx = torch.as_tensor(selected, device=model.device)
    model.eval()
    with torch.no_grad():
        matched = model(idx, control="matched_query")
        wrong = model(idx, control="wrong_query")
        a, b = matched["patch_cond_read"], wrong["patch_cond_read"]
        valid = a["valid_WSI"] & b["valid_WSI"]
        tv = .5 * ((a["weights"] - b["weights"]).abs().sum(-1) + (a["null"] - b["null"]).abs())
        answer = dict(patients=len(selected), valid_WSI_program_rows=int(valid.sum()),
            patch_attention_TV=float(tv[valid].mean()) if valid.any() else None,
            risk_difference_SD=float((matched["risk"] - wrong["risk"]).std(unbiased=False)),
            delta_E_change_L2=float((matched["delta_E"] - wrong["delta_E"]).norm()),
            delta_G_change_L2=float((matched["delta_G"] - wrong["delta_G"]).norm()),
            query_source_mask_changed_rows=int((matched["query_source_mask"] != wrong["query_source_mask"]).sum()),
            original_values_and_main_effects_preserved=all(torch.equal(matched[key], wrong[key])
                for key in ("patch_values", "gene_values", "gene_measured_mask", "main_delta", "offset")),
            frozen_not_retrained=True, histology_or_gene_causality_claimed=False)
        if matched["reverse_cond_read"] is not None:
            gene_a = matched["reverse_cond_read"]["weights"]
            gene_b = wrong["reverse_cond_read"]["weights"]
            answer["reverse_weight_absolute_change_sum"] = float((gene_a - gene_b).abs().sum())
    return answer

"""Fit-only, fixed-budget relative-attention kernel warmup for CDEI-G0.

No validation endpoint access or epoch selection is implemented.  The caller
must enforce 80 epochs for a formal run; shorter calls are numerical
qualification only.  Checkpoints/arrays are never persisted by this module.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.phase17_ssd.losses import full_risk_efron
from .model import CompatibilityBackbone


FORMAL_EPOCHS = 80
MICROBATCH_PATIENTS = 32


def _seed(seed, fold):
    if (isinstance(seed, bool) or isinstance(fold, bool)
            or int(seed) != seed or int(fold) != fold or seed < 0 or fold < 1):
        raise ValueError("seed must be a nonnegative integer and fold a positive integer")
    value = int(seed) + 100 * int(fold)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return value


def _fit_contract(ds):
    """Read shapes, FIT features/offsets and one fit_labels() result only."""
    fit = np.asarray(ds.fit)
    owner = np.asarray(ds.owner)
    if (fit.ndim != 1 or not len(fit) or not np.issubdtype(fit.dtype, np.integer)
            or len(np.unique(fit)) != len(fit)):
        raise ValueError("fit must contain unique integer patient indices")
    if (owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer)
            or len(owner) != len(ds.visual)):
        raise ValueError("WSI owner indices and visual rows differ")
    ft = ds.ft
    required = ("membership", "C", "R", "G", "L", "RM", "CM", "ANCHOR")
    if any(key not in ft for key in required):
        raise ValueError("warmup is missing qualified input tensors")
    if any(not isinstance(ft[key], torch.Tensor) for key in required):
        raise TypeError("qualified warmup features must be tensors")
    patients = len(ft["C"])
    if fit.min() < 0 or fit.max() >= patients:
        raise ValueError("fit patient index is outside feature rows")
    if ft["C"].ndim != 2 or ft["C"].shape[1] < 1:
        raise ValueError("clinical features must have a positive fit-local encoded dimension")
    member = ft["membership"]
    if member.dtype != torch.bool or member.shape != (8, 512) or not bool(member.any(-1).all()):
        raise ValueError("CDEI-G0 requires eight nonempty Boolean programs over RNA512")
    for key in ("R", "G", "L", "RM", "CM"):
        if ft[key].shape != (patients, 512):
            raise ValueError("RNA/CNA features and availability must be patient by 512")
    if ft["RM"].dtype != torch.bool or ft["CM"].dtype != torch.bool:
        raise TypeError("RNA/CNA availability tensors must be Boolean")
    if (ds.visual.ndim != 3 or ds.visual.shape[-1] != 1536
            or ds.mask.shape != ds.visual.shape[:2] or ds.mask.dtype != torch.bool):
        raise ValueError("invalid qualified pathology feature/mask contract")
    device = ds.visual.device
    if any(ft[key].device != device for key in required if key != "membership") or ds.mask.device != device:
        raise ValueError("features, masks and frozen offsets must share the visual device")
    if any(not ft[key].is_floating_point() for key in ("C", "R", "G", "L", "ANCHOR")):
        raise TypeError("observed features and frozen offsets must be floating point")
    if ft["ANCHOR"].shape != (patients,):
        raise ValueError("frozen anchor must be one scalar per patient")
    anchor = ft["ANCHOR"][fit].detach().double().clone()
    anchor_snapshot = ft["ANCHOR"][fit].detach().clone()
    if not bool(torch.isfinite(anchor).all()):
        raise ValueError("nonfinite frozen fit anchor")
    scale = float(ds.anchor_sd_fit)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("anchor_sd_fit must be finite and positive")
    selected_slides = np.flatnonzero(np.isin(owner, fit))
    if set(owner[selected_slides].tolist()) != set(fit.tolist()):
        raise ValueError("fit patient without a qualified WSI")
    # The ONLY endpoint interface. It returns exactly the current fit order.
    labels = ds.fit_labels()
    if isinstance(labels, dict):
        times, events = labels["time"], labels["event"]
    else:
        times, events = labels
    times = np.asarray(times.detach().cpu() if isinstance(times, torch.Tensor) else times, dtype=np.float64)
    events = np.asarray(events.detach().cpu() if isinstance(events, torch.Tensor) else events, dtype=np.float64)
    if (times.shape != fit.shape or events.shape != fit.shape
            or not np.isfinite(times).all() or not np.isfinite(events).all()
            or (times < 0).any() or not np.isin(events, [0., 1.]).all()
            or events.sum() < 1):
        raise ValueError("fit_labels must provide finite fit-ordered times and binary events")
    return (fit.astype(np.int64, copy=False), owner, anchor, anchor_snapshot,
            scale, times, events, len(selected_slides))


def _fit_forward(backbone, head, ds, fit, owner, anchor, scale, *, training):
    """Keep all patient graphs until ONE full-risk loss is formed."""
    corrections = []
    for start in range(0, len(fit), MICROBATCH_PATIENTS):
        current = fit[start:start + MICROBATCH_PATIENTS]
        slides = np.flatnonzero(np.isin(owner, current))
        owners = owner[slides]
        ft = ds.ft
        args = (ds.visual[slides], ds.mask[slides], ft["C"][owners],
                *(ft[key][owners] for key in ("R", "G", "L", "RM", "CM")))

        def call(visual, mask, clinical, rna, gain, loss, rm, cm):
            relative, _ = backbone(visual, mask, clinical, rna, gain, loss, rm, cm)
            return torch.tanh(head(relative).squeeze(-1))

        wsi = (checkpoint(call, *args, use_reentrant=False, preserve_rng_state=True)
               if training else call(*args))
        corrections.append(torch.stack([
            wsi[torch.as_tensor(owners == patient, device=wsi.device)].double().mean()
            for patient in current]))
    correction = scale * torch.cat(corrections)
    # Add anchor ONCE after equal-WSI correction pooling. Repeated averaging of
    # identical float32 offsets must not alter the frozen anchor by one ULP.
    return anchor + correction, correction


def _gradient_audit(backbone, head):
    groups = {key: [] for key in ("kernel", "q", "k", "path_factor", "omics_factor",
                                  "clinical_factor", "clinical_nonintercept", "molecular",
                                  "RNA_filters", "CNA_gain_filters", "CNA_loss_filters",
                                  "visual", "values", "readout")}
    active = 0
    for name, parameter in backbone.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("nonfinite kernel parameter gradient")
        active += parameter.numel()
        norm = float(gradient.double().norm())
        groups["kernel"].append(norm * norm)
        if name.startswith("visual."):
            group = "visual"
        elif name.startswith(("p_value.", "o_value.")):
            group = "values"
        elif name.startswith("molecular."):
            group = "molecular"
        else:
            group = {"q": "q", "k": "k", "a": "path_factor", "b": "omics_factor",
                     "c": "clinical_factor"}.get(name.split(".")[1])
        if group is not None:
            groups[group].append(norm * norm)
        if name == "compatibility.c.weight":
            groups["clinical_nonintercept"].append(float(gradient[:, 1:].double().norm()) ** 2)
        if name.startswith("molecular.filters."):
            index = int(name.split(".")[2])
            groups[("RNA_filters", "CNA_gain_filters", "CNA_loss_filters")[index // 8]].append(norm * norm)
    for parameter in head.parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("nonfinite relative readout gradient")
            active += parameter.numel()
            groups["readout"].append(float(parameter.grad.detach().double().norm()) ** 2)
    return {key: float(math.sqrt(sum(value))) for key, value in groups.items()}, active


def train_kernel(ds, seed, fold, epochs=FORMAL_EPOCHS, step_callback=None):
    """Return (frozen eval backbone, CPU head state, anonymous training audit).

    Exactly ``epochs`` full-fit Efron optimizer steps; no early stopping,
    validation scoring, best-state selection or persistent output. A non-80
    budget is labeled QUALIFICATION_ONLY_NOT_FORMAL in the returned audit.
    """
    if isinstance(epochs, bool) or int(epochs) != epochs or not 1 <= epochs <= FORMAL_EPOCHS:
        raise ValueError("epochs must be an integer between 1 and the locked 80-step budget")
    epochs = int(epochs)
    seeded = _seed(seed, fold)
    (fit, owner, anchor, anchor_snapshot, scale, times, events, fit_wsi) = _fit_contract(ds)
    backbone = CompatibilityBackbone(ds.ft["C"].shape[1], ds.ft["membership"].detach().cpu()).to(ds.visual.device)
    head = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Dropout(.1),
                         nn.Linear(16, 1, bias=False)).to(ds.visual.device)
    parameters = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    backbone.eval(); head.eval()
    with torch.no_grad():
        baseline_loss = float(full_risk_efron(anchor, times, events))
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        backbone.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        risk, correction = _fit_forward(backbone, head, ds, fit, owner, anchor, scale, training=True)
        objective = full_risk_efron(risk, times, events)
        if not bool(torch.isfinite(objective)):
            raise ValueError("nonfinite full-fit Efron warmup objective")
        objective.backward()
        gradients, active = _gradient_audit(backbone, head)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise ValueError("nonfinite kernel/readout parameter after optimizer step")
        if not torch.equal(ds.ft["ANCHOR"][fit].detach(), anchor_snapshot):
            raise ValueError("frozen anchor changed during kernel warmup")
        backbone.eval(); head.eval()
        with torch.no_grad():
            eval_risk, eval_correction = _fit_forward(backbone, head, ds, fit, owner, anchor, scale, training=False)
            fit_loss = full_risk_efron(eval_risk, times, events)
        if not bool(torch.isfinite(fit_loss)):
            raise ValueError("nonfinite updated full-fit evaluation loss")
        history.append({"epoch": epoch, "training_loss": float(objective.detach()),
                        "fit_loss": float(fit_loss), "grad_norm": float(norm),
                        "gradients": gradients, "gradient_active_parameters": active,
                        "fit_correction_population_SD": float(eval_correction.std(unbiased=False)),
                        "epoch_seconds": float(time.monotonic() - started)})
        if step_callback is not None:
            step_callback("kernel_warmup", epoch)
    for parameter in backbone.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    backbone.eval(); head.eval()
    head_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    audit = {"status": "COMPLETE_FIXED_BUDGET_KERNEL_WARMUP",
             "budget_mode": "FORMAL_FIXED_80" if epochs == FORMAL_EPOCHS else "QUALIFICATION_ONLY_NOT_FORMAL",
             "seed": int(seed), "fold": int(fold), "effective_seed": seeded,
             "requested_epochs": epochs, "actual_epochs": len(history), "optimizer_steps": len(history),
             "kernel_epoch_selection": "fixed_final_epoch_no_heldout_endpoint_access",
             "fit_patients": len(fit), "fit_WSIs": fit_wsi, "fit_events": int(events.sum()),
             "full_risk_calls": 1 + 2 * epochs, "fit_labels_calls": 1,
             "microbatch_patients": MICROBATCH_PATIENTS, "checkpoint_preserve_rng_state": True,
             "checkpoint_use_reentrant": False, "complete_patient_risk_set": True,
             "equal_WSI_patient_correction": True, "frozen_anchor_unchanged": True,
             "main_effect_trainable_parameters": 0, "anchor_sd_fit": scale,
             "kernel_parameters": sum(p.numel() for p in backbone.parameters()),
             "readout_parameters": sum(p.numel() for p in head.parameters()),
             "learning_rate": .001, "weight_decay": .01, "gradient_clip": 5.,
             "anchor_only_fit_loss": baseline_loss, "history": history,
             "backbone_frozen_eval": not backbone.training and not any(p.requires_grad for p in backbone.parameters()),
             "persistent_checkpoint_writes": 0, "heldout_endpoint_reads": 0}
    return backbone, head_state, audit

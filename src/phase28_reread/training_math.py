"""Full-risk Efron arithmetic and deterministic microbatch gradient replay.

This module never fetches labels, chooses patients, zeroes gradients, or steps
an optimizer. The caller supplies the ordered fit-only endpoints and a callback
``forward_batch(patient_indices)``. The callback returns a one-dimensional total
risk tensor, ``(risk, main_residual, interaction)``, or a mapping with ``risk``
and optional ``main_residual``/``interaction`` tensors.

Every patient must be computationally independent of other microbatch patients;
all WSIs of one patient must already be combined inside that callback. Batch
normalization is deliberately unsupported. Training-mode dropout is rejected;
functional randomness or cross-patient operations remain the caller's contract.
On any replay error the caller must discard accumulated gradients, not step.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Integral

import numpy as np
import torch
from torch import nn

from src.phase17_ssd.losses import full_risk_efron


REPLAY_ATOL = 1e-6


def _numpy_metadata(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _endpoints(time, event, count):
    try:
        t = np.asarray(_numpy_metadata(time), dtype=np.float64)
        e = np.asarray(_numpy_metadata(event), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("fit endpoints must be numeric vectors") from exc
    if t.shape != (count,) or e.shape != (count,):
        raise ValueError("fit endpoint order/length must match the patient vector")
    if not np.isfinite(t).all() or not np.isfinite(e).all() or np.any(t < 0):
        raise ValueError("fit endpoints must be finite with nonnegative times")
    if not np.isin(e, (0., 1.)).all():
        raise ValueError("fit event indicators must be exactly zero or one")
    if not np.any(e == 1):
        raise ValueError("full-risk Efron loss requires at least one fit event")
    return t.copy(), e.copy()


def efron_loss(risk, time, event):
    """Mean negative Efron partial log-likelihood over the supplied full fit set.

    Endpoint order is exactly the risk order; censored rows at an event time
    remain in its risk set. No finite-difference or tie approximation is used.
    """
    if (not isinstance(risk, torch.Tensor) or risk.ndim != 1 or risk.numel() == 0
            or risk.dtype not in (torch.float32, torch.float64)):
        raise ValueError("risk must be a nonempty float32/float64 vector")
    if not torch.isfinite(risk).all():
        raise ValueError("nonfinite full-risk prediction")
    t, e = _endpoints(time, event, len(risk))
    loss = full_risk_efron(risk, t, e)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("nonfinite full-risk Efron loss")
    return loss


def _guard_deterministic_modules(model):
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    for name, module in model.named_modules():
        label = name or "<root>"
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            raise ValueError(f"gradient replay rejects BatchNorm: {label}")
        if (isinstance(module, nn.modules.dropout._DropoutNd)
                and module.training and float(module.p) != 0.):
            raise ValueError(f"gradient replay rejects active dropout: {label}")
        # These modules implement dropout functionally rather than exposing a
        # child Dropout module. Reject a nonzero training setting explicitly.
        if (isinstance(module, (nn.MultiheadAttention, nn.modules.rnn.RNNBase))
                and module.training and float(module.dropout) != 0.):
            raise ValueError(f"gradient replay rejects active dropout: {label}")


def _risk_parts(output, count):
    if isinstance(output, torch.Tensor):
        result = {"risk": output}
    elif isinstance(output, Mapping):
        if "risk" not in output:
            raise ValueError("forward mapping must contain total risk")
        result = {key: output[key] for key in ("risk", "main_residual", "interaction") if key in output}
    elif isinstance(output, (tuple, list)) and len(output) == 3:
        result = dict(zip(("risk", "main_residual", "interaction"), output))
    else:
        raise ValueError("forward must return total risk or three risk components")
    risk = result["risk"]
    for name, value in result.items():
        if (not isinstance(value, torch.Tensor) or value.shape != (count,)
                or value.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid forward risk component: {name}")
        if value.device != risk.device or value.dtype != risk.dtype:
            raise ValueError("risk components must share one dtype/device")
    return result


def _gradient_summary(model, initial):
    """Summarize gradients introduced by this call, excluding entry gradients."""
    groups = {}
    nonzero_elements = nonzero_tensors = active_tensor_elements = 0
    for name, parameter in model.named_parameters():
        group = name.split('.')[0]
        groups.setdefault(group, 0.)
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if gradient.is_sparse:
            raise ValueError("gradient audit requires dense parameter gradients")
        if name in initial:
            gradient = gradient-initial[name]
        gradient = gradient.double()
        if not torch.isfinite(gradient).all():
            raise ValueError("nonfinite accumulated parameter gradient")
        groups[group] += float(gradient.square().sum())
        count = int(torch.count_nonzero(gradient))
        nonzero_elements += count
        if count:
            nonzero_tensors += 1
            active_tensor_elements += parameter.numel()
    return {"L2_by_top_level_module": {key: value**.5 for key, value in groups.items()},
            "nonzero_elements": nonzero_elements, "nonzero_tensors": nonzero_tensors,
            "elements_in_active_tensors": active_tensor_elements}


def risk_gradient_cache(model, forward_batch: Callable, patient_indices, time, event,
                        microbatch=1, regularizer: Callable | None = None):
    """Accumulate exact full-risk gradients through deterministic recomputation.

    The first no-grad pass caches ordered total risks. A full-risk leaf then
    yields the Efron derivative for every patient. Each identical microbatch is
    recomputed, checked to absolute tolerance 1e-6, and backpropagated against
    that slice of the full-set derivative. This is not minibatch Cox training.

    ``regularizer`` is a zero-argument closure returning a scalar tensor with
    its own live graph (normally parameter L2). It is called/backpropagated
    once after all risk slices. An output-dependent regularizer must build its
    own intended graph; no-grad cached components cannot supply its gradient.
    Existing parameter/input gradients are accumulated, never cleared here.
    The returned audit contains aggregate scalars only, not patient risks.
    """
    _guard_deterministic_modules(model)
    if not torch.is_grad_enabled():
        raise ValueError("gradient replay requires enabled autograd")
    if isinstance(microbatch, bool) or not isinstance(microbatch, Integral) or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    initial_gradients = {name: parameter.grad.detach().clone()
                         for name, parameter in model.named_parameters() if parameter.grad is not None}
    indices = _numpy_metadata(patient_indices)
    if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or len(np.unique(indices)) != len(indices)):
        raise ValueError("patient indices must be a nonempty unique nonnegative integer vector")
    indices = np.array(indices, dtype=np.int64, copy=True)
    t, e = _endpoints(time, event, len(indices))
    chunks = [indices[start:start+int(microbatch)].copy()
              for start in range(0, len(indices), int(microbatch))]

    cached_parts = None
    schema = None
    with torch.no_grad():
        for current in chunks:
            parts = _risk_parts(forward_batch(current.copy()), len(current))
            if schema is None:
                schema = tuple(parts)
                cached_parts = {name: [] for name in schema}
            if tuple(parts) != schema:
                raise ValueError("forward component schema changed between microbatches")
            for name, value in parts.items():
                cached_parts[name].append(value.detach().clone())
        cache = {name: torch.cat(values) for name, values in cached_parts.items()}
    leaf = cache["risk"].clone().requires_grad_(True)
    cox = efron_loss(leaf, t, e)
    risk_gradient, = torch.autograd.grad(cox, leaf)
    if not torch.isfinite(risk_gradient).all():
        raise ValueError("nonfinite full-risk derivative")

    maximum_error = 0.
    maximum_component_error = 0.
    replay_backwards = 0
    start = 0
    for current in chunks:
        _guard_deterministic_modules(model)
        parts = _risk_parts(forward_batch(current.copy()), len(current))
        if tuple(parts) != schema:
            raise ValueError("forward component schema changed during gradient replay")
        stop = start+len(current)
        for name, value in parts.items():
            expected = cache[name][start:stop]
            if value.dtype != expected.dtype or value.device != expected.device:
                raise ValueError("forward dtype/device changed during gradient replay")
            error = float((value.detach()-expected).abs().max())
            maximum_component_error = max(maximum_component_error, error)
            if name == "risk":
                maximum_error = max(maximum_error, error)
            if error > REPLAY_ATOL:
                raise ValueError(f"nondeterministic gradient replay for {name}: error {error:.9g}")
        surrogate = torch.dot(parts["risk"], risk_gradient[start:stop])
        if surrogate.requires_grad:
            surrogate.backward()
            replay_backwards += 1
        start = stop

    data_gradient = _gradient_summary(model, initial_gradients)
    penalty_value = 0.
    penalty_calls = 0
    penalty_requires_grad = False
    if regularizer is not None:
        penalty = regularizer()
        penalty_calls = 1
        if (not isinstance(penalty, torch.Tensor) or penalty.ndim != 0
                or not penalty.is_floating_point() or not torch.isfinite(penalty)):
            raise ValueError("regularizer must return one finite scalar tensor")
        penalty_value = float(penalty.detach())
        penalty_requires_grad = bool(penalty.requires_grad)
        if penalty_requires_grad:
            penalty.backward()
    total_gradient = _gradient_summary(model, initial_gradients)

    audit = {
        "patients": len(indices), "events": int(e.sum()),
        "microbatch": int(microbatch), "microbatches": len(chunks),
        "forward_calls": 2*len(chunks), "risk_backward_calls": replay_backwards,
        "regularizer_calls": penalty_calls,
        "regularizer_requires_grad": penalty_requires_grad,
        "cox_loss": float(cox.detach()), "regularizer": penalty_value,
        "objective": float(cox.detach())+penalty_value,
        "maximum_absolute_risk_replay_error": maximum_error,
        "maximum_absolute_component_replay_error": maximum_component_error,
        "replay_absolute_tolerance": REPLAY_ATOL,
        "risk_gradient_maximum_absolute": float(risk_gradient.abs().max()),
        "risk_gradient_sum": float(risk_gradient.sum()),
        "data_gradient_L2_by_top_level_module": data_gradient["L2_by_top_level_module"],
        "parameters_with_nonzero_data_gradient": data_gradient["nonzero_elements"],
        "parameter_tensors_with_nonzero_data_gradient": data_gradient["nonzero_tensors"],
        "parameter_elements_in_tensors_with_nonzero_data_gradient": data_gradient["elements_in_active_tensors"],
        "total_gradient_L2_by_top_level_module": total_gradient["L2_by_top_level_module"],
        "parameters_with_nonzero_total_gradient": total_gradient["nonzero_elements"],
        "parameter_tensors_with_nonzero_total_gradient": total_gradient["nonzero_tensors"],
        "parameter_elements_in_tensors_with_nonzero_total_gradient": total_gradient["elements_in_active_tensors"],
        "gradient_audit_excludes_preexisting_gradients": True,
        "optimizer_steps": 0,
    }
    for name, value in cache.items():
        audit[name+"_RMS"] = float(value.double().square().mean().sqrt())
    return audit

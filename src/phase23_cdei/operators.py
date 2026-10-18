"""Outcome-blind arithmetic for the frozen-kernel CDEI-G0 audit.

Relative attention uses ORIGINAL scores; only the evidence branch subtracts
null. Empty rows have zero outputs AND has_valid=False: they are missing,
not observed neutral biological evidence. No fitting or reference estimation.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains a nonfinite active value or overflow")


def _temperature(tau: float) -> float:
    if isinstance(tau, bool) or not isinstance(tau, Real):
        raise TypeError("tau must be a finite positive real scalar")
    result = float(tau)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("tau must be finite and strictly positive")
    return result


def _mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("valid must be a Boolean tensor")
    if mask.device != x.device:
        raise ValueError("valid and values must be on the same device")
    try:
        return torch.broadcast_to(mask, x.shape)
    except RuntimeError as exc:
        raise ValueError("valid must be broadcastable to the values shape") from exc


def _stable_row_logmeanexp(x, valid, count, has_valid, tau):
    """FP64 masked row LME; x already has finite, zeroed padding."""
    peak = x.masked_fill(~valid, -torch.inf).amax(-1)
    peak = torch.where(has_valid, peak, torch.zeros_like(peak))
    difference = torch.where(valid, x - peak[..., None], torch.zeros_like(x))
    _finite(difference, "centered LogMeanExp scores")
    scaled = difference / tau
    _finite(scaled, "temperature-scaled LogMeanExp scores")
    total = (torch.exp(scaled) * valid).sum(-1)
    safe_total = torch.where(has_valid, total, torch.ones_like(total))
    safe_count = count.clamp_min(1).double()
    ordinary_logmean = torch.log(safe_total) - torch.log(safe_count)
    # log(sum(exp(x)))-log(n) loses tiny deviations from a constant.  The
    # expm1/log1p form preserves them, including scores tiny relative to tau.
    # Use it only away from -1; the unchosen log1p arm must remain finite for
    # backward, too (torch.where alone does not neutralize 0*infinite gradients).
    mean_expm1 = (torch.expm1(scaled) * valid / safe_count[..., None]).sum(-1)
    near_constant = mean_expm1 > -0.5
    safe_expm1 = torch.where(near_constant, mean_expm1, torch.zeros_like(mean_expm1))
    logmean = torch.where(near_constant, torch.log1p(safe_expm1), ordinary_logmean)
    result = peak + tau * logmean
    result = torch.where(has_valid, result, torch.zeros_like(result))
    _finite(result, "LogMeanExp result")
    return result


def dual_evidence_core(scores, null, valid, tau=1.0):
    """Return original-score relative attention and calibrated row statistics.

    Last axis is keys. scores/null must have identical shapes/devices and be
    floating point. Null MUST already be detached: a live graph is rejected,
    never silently detached. Valid is Boolean and broadcastable to scores.

    po returns in scores.dtype; evidence and decomposition use float64. Inputs
    are promoted BEFORE subtraction, allowing finite FP32 extreme values.
    Nonfinite active values or unrepresentable FP64 arithmetic fail closed.
    NaN/Inf padding is excluded before arithmetic and has zero gradient.
    Empty rows have zero attention/statistics and has_valid=False.
    """
    tau = _temperature(tau)
    if not isinstance(scores, torch.Tensor) or not scores.is_floating_point():
        raise TypeError("scores must be a floating-point tensor")
    if not isinstance(null, torch.Tensor) or not null.is_floating_point():
        raise TypeError("null must be a floating-point tensor")
    if scores.ndim < 1 or scores.numel() == 0 or scores.shape[-1] == 0:
        raise ValueError("scores must have a nonempty key axis and nonempty rows")
    if scores.shape != null.shape or scores.device != null.device:
        raise ValueError("scores and null must have identical shapes and devices")
    if null.requires_grad or null.grad_fn is not None:
        raise ValueError("null must be detached and have no autograd graph")
    valid = _mask(valid, scores)
    # Sanitize operands BEFORE subtraction, not only the final residual.
    s = torch.where(valid, scores.double(), torch.zeros_like(scores, dtype=torch.float64))
    b = torch.where(valid, null.double(), torch.zeros_like(null, dtype=torch.float64))
    _finite(s, "scores")
    _finite(b, "null")
    residual = s - b
    _finite(residual, "scores minus null")
    count = valid.sum(-1)
    has_valid = count > 0
    safe_count = count.clamp_min(1).double()

    peak = s.masked_fill(~valid, -torch.inf).amax(-1)
    peak = torch.where(has_valid, peak, torch.zeros_like(peak))
    score_difference = torch.where(valid, s - peak[..., None], torch.zeros_like(s))
    _finite(score_difference, "relative softmax centered scores")
    numerator = score_difference.exp() * valid
    denominator = numerator.sum(-1, keepdim=True)
    denominator = torch.where(has_valid[..., None], denominator, torch.ones_like(denominator))
    po = (numerator / denominator).to(scores.dtype)
    _finite(po, "relative attention")

    evidence = _stable_row_logmeanexp(residual, valid, count, has_valid, tau)
    # Divide before summation to avoid avoidable sum/count overflow.
    mean = (residual / safe_count[..., None]).sum(-1)
    _finite(mean, "residual mean")
    centered = torch.where(valid, residual - mean[..., None], torch.zeros_like(residual))
    _finite(centered, "mean-centered residual")
    centered_lme = _stable_row_logmeanexp(centered, valid, count, has_valid, tau)
    reconstructed = mean + centered_lme
    _finite(reconstructed, "reconstructed evidence")
    error = (evidence - reconstructed).abs()
    magnitude = torch.stack([evidence.abs(), mean.abs(), centered_lme.abs()], -1).amax(-1)
    tolerance = (128.0 * torch.finfo(torch.float64).eps) * magnitude.clamp_min(1.0)
    if not bool((error <= tolerance).all()):
        raise ValueError("evidence = residual_mean + centered_lme qualification failed")
    return {"po": po, "evidence": evidence, "residual_mean": mean,
            "centered_lme": centered_lme, "valid_count": count,
            "has_valid": has_valid, "decomposition_error": error,
            "decomposition_tolerance": tolerance}


def masked_mean_std_pool(values, valid):
    """Within-WSI patch evidence -> [...,2] mean and population SD.

    Preserve valid.any(-1) for availability. Empty WSIs return [0,0]. The
    later multi-WSI patient aggregation must remain explicit. Zero variance
    rows, including single-patch rows, have finite zero SD gradient.
    """
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError("values must be a floating-point tensor")
    if values.ndim < 1 or values.numel() == 0 or values.shape[-1] == 0:
        raise ValueError("values must have a nonempty patch axis")
    valid = _mask(valid, values)
    x = torch.where(valid, values.double(), torch.zeros_like(values, dtype=torch.float64))
    _finite(x, "pool values")
    n = valid.sum(-1).clamp_min(1).double()
    mean = (x / n[..., None]).sum(-1)
    _finite(mean, "pooled mean")
    centered = torch.where(valid, x - mean[..., None], torch.zeros_like(x))
    _finite(centered, "pooled centered values")
    # Scaled squares avoid overflow for finite evidence such as +/-1e200.
    scale = centered.abs().amax(-1)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = centered / safe_scale[..., None]
    variance = (normalized.square() / n[..., None]).sum(-1)
    positive = variance > 0
    safe_variance = torch.where(positive, variance, torch.ones_like(variance))
    root = torch.where(positive, safe_variance.sqrt(), torch.zeros_like(variance))
    result = torch.stack([mean, scale * root], -1)
    _finite(result, "pooled mean and population SD")
    return result


core = dual_evidence_core

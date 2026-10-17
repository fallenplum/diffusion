"""Count-conditioned composition and within-tissue listwise spatial losses.

STRank's fixed official ``STRankLossList`` source uses the spot axis, raw count
weights and a log library-exposure offset.  This adaptation explicitly masks
unmeasured genes, normalizes each tissue by observed count mass, averages
tissues within a donor and finally averages donors.  It is not an exact
reproduction of the official batch-element mean.  Only true raw counts enter
this objective; standardized expression and survival labels are not inputs.

Source: https://github.com/naivete5656/STRank/blob/
e1917b4b311a72d9150c03bc4c546d1a6550888e/strank/loss_functions.py
"""
from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpatialLoss:
    loss: Tensor
    composition: Tensor
    relative: Tensor
    tissue_count: int
    donor_count: int
    measured_entries: int
    positive_count_entries: int


def _ids(values: Sequence | Tensor, size: int, name: str) -> list:
    if isinstance(values, Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        values = values.detach().cpu().tolist()
    values = list(values)
    if len(values) != size:
        raise ValueError(f"{name} length must equal spot count")
    for value in values:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"{name} must not contain missing IDs")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} IDs must be scalar and hashable") from exc
    return values


def _prepare(log_mu: Tensor, raw_counts: Tensor, tissue_ids, donor_ids,
             measured_mask: Tensor | None, spot_mask: Tensor | None,
             exposure: Tensor | None):
    if not isinstance(log_mu, Tensor) or not log_mu.is_floating_point():
        raise ValueError("log_mu must be a floating-point torch tensor")
    if not isinstance(raw_counts, Tensor) or log_mu.ndim != 2 or raw_counts.shape != log_mu.shape:
        raise ValueError("log_mu and raw_counts must share [spot, gene] shape")
    if min(log_mu.shape) < 1:
        raise ValueError("empty spot or gene dimensions are not supported")
    if raw_counts.device != log_mu.device:
        raise ValueError("counts and predictions must share device")
    n, g = log_mu.shape
    tissue = _ids(tissue_ids, n, "tissue_ids")
    donor = _ids(donor_ids, n, "donor_ids")
    measured = torch.ones_like(log_mu, dtype=torch.bool) if measured_mask is None else measured_mask
    spots = torch.ones(n, dtype=torch.bool, device=log_mu.device) if spot_mask is None else spot_mask
    if measured.shape != log_mu.shape or measured.dtype != torch.bool or measured.device != log_mu.device:
        raise ValueError("measured_mask must be boolean [spot, gene] on prediction device")
    if spots.shape != (n,) or spots.dtype != torch.bool or spots.device != log_mu.device:
        raise ValueError("spot_mask must be boolean [spot] on prediction device")
    measured = measured & spots[:, None]
    observed_counts = raw_counts[measured].double()
    if not torch.isfinite(observed_counts).all() or (observed_counts < 0).any():
        raise ValueError("measured raw counts must be finite and nonnegative")
    if not torch.allclose(observed_counts, observed_counts.round(), rtol=0, atol=1e-6):
        raise ValueError("raw counts must be integer-valued; normalized expression is not raw counts")
    if not torch.isfinite(log_mu[measured]).all():
        raise ValueError("predicted measured composition must have finite log probabilities")
    # Missing/padding payloads may be NaN: sanitize before differentiable math.
    y = torch.where(measured, raw_counts.double(), 0.)
    lm = torch.where(measured, log_mu.double(), 0.)
    library = y.sum(1)
    if exposure is not None:
        if exposure.shape != (n,) or exposure.device != log_mu.device:
            raise ValueError("exposure must be [spot] on prediction device")
        library = torch.where(spots, exposure.double(), 0.)
    if not torch.isfinite(library).all() or (library < 0).any():
        raise ValueError("exposure must be finite and nonnegative")
    if ((y.sum(1) > 0) & (library <= 0)).any():
        raise ValueError("spots with positive count mass require positive exposure")
    tissue_donor = {}
    for index in torch.where(spots)[0].tolist():
        if tissue[index] in tissue_donor and tissue_donor[tissue[index]] != donor[index]:
            raise ValueError("one tissue ID cannot belong to multiple donors")
        tissue_donor[tissue[index]] = donor[index]
    return lm, y, measured, spots, library, tissue, tissue_donor


def spatial_objective(log_mu: Tensor, raw_counts: Tensor, tissue_ids, donor_ids,
                      *, measured_mask: Tensor | None = None,
                      spot_mask: Tensor | None = None, exposure: Tensor | None = None,
                      lambda_relative: float = 1.) -> SpatialLoss:
    """Return L_composition + lambda_relative L_relative with donor equal weight.

    ``log_mu`` is the logarithm of predicted gene composition, not z-scores.
    It is normalized over each spot's observed gene set for composition loss.
    Relative probabilities are proportional to exposure * mu within each
    tissue/gene, over measured spots with positive exposure.  A measured zero
    is a valid denominator candidate. An unmeasured entry is never a candidate.
    Zero-count tissues contribute neither numerator nor an artificial zero to
    the donor mean; an entirely zero-count batch raises instead of training.
    IDs are used only for grouping; this module never returns their values.
    """
    if not math.isfinite(lambda_relative) or lambda_relative < 0:
        raise ValueError("lambda_relative must be finite and nonnegative")
    lm, y, measured, spots, library, tissue, tissue_donor = _prepare(
        log_mu, raw_counts, tissue_ids, donor_ids, measured_mask, spot_mask, exposure)
    by_donor: dict[object, list[tuple[Tensor, Tensor]]] = {}
    used_tissues = 0
    for tissue_id, donor_id in tissue_donor.items():
        idx = torch.tensor([i for i, t in enumerate(tissue) if t == tissue_id and bool(spots[i])],
                           device=log_mu.device, dtype=torch.long)
        yy, mm, ll = y[idx], measured[idx], lm[idx]
        count_mass = yy.sum()
        if count_mass.item() == 0:
            continue
        # Condition composition on the genes actually measured at each spot.
        active_spots = yy.sum(1) > 0
        point_logp = ll[active_spots].masked_fill(~mm[active_spots], -torch.inf)
        point_logp = point_logp - torch.logsumexp(point_logp, dim=1, keepdim=True)
        pp = yy[active_spots] > 0
        composition = -(yy[active_spots][pp] * point_logp[pp]).sum() / count_mass
        # For each gene, normalize over spots, not over genes or other tissues.
        lib = library[idx]
        valid = mm & (lib[:, None] > 0)
        has_candidate = valid.any(0)
        if not has_candidate.any():
            raise ValueError("positive counts without a valid relative candidate")
        logits = ll[:, has_candidate] + lib.clamp_min(torch.finfo(torch.float64).tiny).log()[:, None]
        logits = logits.masked_fill(~valid[:, has_candidate], -torch.inf)
        logp = logits - torch.logsumexp(logits, dim=0, keepdim=True)
        yy_rel = yy[:, has_candidate]
        positive = yy_rel > 0
        relative = -(yy_rel[positive] * logp[positive]).sum() / count_mass
        by_donor.setdefault(donor_id, []).append((composition, relative))
        used_tissues += 1
    if not by_donor:
        raise ValueError("batch has no measured positive count mass")
    composition = torch.stack([torch.stack([v[0] for v in values]).mean()
                               for values in by_donor.values()]).mean()
    relative = torch.stack([torch.stack([v[1] for v in values]).mean()
                            for values in by_donor.values()]).mean()
    return SpatialLoss(composition + lambda_relative * relative, composition, relative,
                       used_tissues, len(by_donor), int(measured.sum().item()),
                       int((y > 0).sum().item()))


def composition_loss(*args, **kwargs) -> Tensor:
    """Count-conditioned composition component (same validation and grouping)."""
    return spatial_objective(*args, **kwargs).composition


def strank_listwise_loss(*args, **kwargs) -> Tensor:
    """Project-adapted listwise STRank component, not the official mean scale."""
    return spatial_objective(*args, **kwargs).relative

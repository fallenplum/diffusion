"""Low-rank clinical-conditioned pathology/omics attention arithmetic.

This is the supplied CP-factorized trilinear score, not an originality claim:

    score_ik = <q(p_i), k(o_k)> / sqrt(d)
               + sum_r a_r(p_i) b_r(o_k) c_r(clinical).

Both attention directions share the same score matrix.  This module performs
neither value/readout pooling nor survival prediction, normalization, dropout,
token compression, clinical main effects, or patient matching.
"""

import math

import torch
from torch import nn


def masked_softmax(scores, mask, dim=-1):
    """Softmax on valid entries; an entirely masked row is exactly zero.

    ``mask`` must be Boolean and broadcastable to ``scores``.  The temporary
    maximum is set to zero on empty rows before subtracting, so the computation
    never forms ``-inf - -inf`` and its backward remains finite.
    """
    if not scores.is_floating_point():
        raise TypeError("scores must have a floating-point dtype")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be Boolean")
    if mask.device != scores.device:
        raise ValueError("scores and mask must be on the same device")
    try:
        valid = torch.broadcast_to(mask, scores.shape)
    except RuntimeError as exc:
        raise ValueError("mask must be broadcastable to scores") from exc
    masked = scores.masked_fill(~valid, -torch.inf)
    has_valid = valid.any(dim=dim, keepdim=True)
    peak = masked.amax(dim=dim, keepdim=True)
    peak = torch.where(has_valid, peak, torch.zeros_like(peak))
    numerator = torch.exp(masked - peak)
    denominator = numerator.sum(dim=dim, keepdim=True)
    return numerator / denominator.clamp_min(torch.finfo(scores.dtype).tiny)


class RankConditionedAttention(nn.Module):
    """One shared bidirectional score kernel with optional rank-R context.

    Inputs are pathology ``p[B,N,Dp]``, omics ``o[B,K,Do]``, clinical context
    ``clinical[B,Dc]``, and Boolean availability masks.  ``p_to_o`` is normalized
    over K, ``o_to_p`` over N.  Invalid source rows and destinations have zero
    probability, including patients with no valid tokens in one modality.

    All five projections have no bias.  A zero clinical vector therefore
    removes the trilinear score exactly; rank=0 removes those parameters
    altogether.  No additional 1/sqrt(rank) or learned scaling is introduced.
    """

    def __init__(self, p_dim, o_dim, c_dim, d=32, rank=4):
        super().__init__()
        if any(int(x) != x or x <= 0 for x in (p_dim, o_dim, c_dim, d)):
            raise ValueError("feature dimensions and d must be positive integers")
        if int(rank) != rank or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.p_dim, self.o_dim, self.c_dim = int(p_dim), int(o_dim), int(c_dim)
        self.d, self.rank = int(d), int(rank)
        self.q = nn.Linear(self.p_dim, self.d, bias=False)
        self.k = nn.Linear(self.o_dim, self.d, bias=False)
        # Avoid zero-width Linear initialization and its warning in rank=0.
        self.a = nn.Linear(self.p_dim, self.rank, bias=False) if self.rank else None
        self.b = nn.Linear(self.o_dim, self.rank, bias=False) if self.rank else None
        self.c = nn.Linear(self.c_dim, self.rank, bias=False) if self.rank else None

    def forward(self, p, o, clinical, patch_mask=None, concept_mask=None, clinical_override=None):
        if p.ndim != 3 or o.ndim != 3 or clinical.ndim != 2:
            raise ValueError("p/o must be rank 3 and clinical rank 2")
        batch, n, dp = p.shape
        if o.shape[0] != batch or clinical.shape != (batch, self.c_dim):
            raise ValueError("all inputs must share batch size and configured clinical dimension")
        if dp != self.p_dim or o.shape[2] != self.o_dim:
            raise ValueError("pathology/omics feature dimensions differ from configuration")
        if n < 1 or o.shape[1] < 1:
            raise ValueError("token axes must be nonempty; represent missing modalities by masks")
        if p.device != o.device or p.device != clinical.device:
            raise ValueError("all inputs must be on the same device")
        if clinical_override is not None:
            if clinical_override.shape != clinical.shape or clinical_override.device != clinical.device:
                raise ValueError("clinical_override must match clinical shape and device")
            clinical = clinical_override
        if patch_mask is None:
            patch_mask = torch.ones((batch, n), dtype=torch.bool, device=p.device)
        if concept_mask is None:
            concept_mask = torch.ones(o.shape[:2], dtype=torch.bool, device=o.device)
        for name, mask, shape in (("patch_mask", patch_mask, (batch, n)),
                                  ("concept_mask", concept_mask, o.shape[:2])):
            if mask.shape != shape or mask.dtype != torch.bool or mask.device != p.device:
                raise ValueError(f"{name} must be Boolean, have shape {tuple(shape)}, and share input device")
        # Invalid token values cannot influence valid scores or leave NaNs in
        # raw diagnostic tensors.  No observed feature is changed.
        pp = p.masked_fill(~patch_mask[:, :, None], 0)
        oo = o.masked_fill(~concept_mask[:, :, None], 0)
        base_scores = (self.q(pp) @ self.k(oo).transpose(-1, -2)) / math.sqrt(self.d)
        if self.rank:
            trilinear = (self.a(pp) * self.c(clinical)[:, None, :]) @ self.b(oo).transpose(-1, -2)
        else:
            trilinear = torch.zeros_like(base_scores)
        scores = base_scores + trilinear
        valid_pairs = patch_mask[:, :, None] & concept_mask[:, None, :]
        return {
            "scores": scores,
            "base_scores": base_scores,
            "trilinear": trilinear,
            "p_to_o": masked_softmax(scores, valid_pairs, dim=-1),
            "o_to_p": masked_softmax(scores.transpose(-1, -2), valid_pairs.transpose(-1, -2), dim=-1),
            "valid_pairs": valid_pairs,
        }

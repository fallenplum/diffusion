"""Bounded multi-RNA-slot readout with an isolated attention-axis control.

The operator uses fixed random position groups, not biological pathway labels.
The competitive weights are a one-step Slot-Attention-inspired normalization,
not the official iterative GRU module or a doubly stochastic transport plan.
All three arms register the identical parameters. QUERY deliberately leaves
Patch/Clinical-score parameters inactive in prediction; the caller must report
that effective-capacity difference rather than imply identical active models.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn


SLOTS, HIDDEN, RANK = 8, 32, 4


def attention_weights(scores, mask, operator):
    """Return weights, slot assignments and pre-row-normalization slot mass.

    COMP is exactly B=softmax_slot(S), A=B/sum_patch(B), implemented as
    softmax_patch(log_softmax_slot(S)) to avoid an all-zero row when exp(logB)
    underflows. No epsilon is added to B, scores, or the row normalizer. The
    diagnostic assignment/mass may underflow; the predictive A stays stable.
    ATTN is ordinary per-slot softmax over the same valid Patch candidates.
    """
    if (scores.ndim != 3 or scores.shape[1] != SLOTS or not scores.is_floating_point()
            or mask.shape != (len(scores), scores.shape[2]) or mask.dtype != torch.bool
            or mask.device != scores.device or not mask.any(-1).all()):
        raise ValueError("attention needs eight slots and a nonempty Boolean Patch mask")
    valid = mask[:, None, :].expand_as(scores)
    if not torch.isfinite(scores[valid]).all():
        raise ValueError("nonfinite observed compatibility score")
    safe = torch.where(valid, scores, torch.zeros_like(scores))
    if operator == "ATTN":
        weights = safe.masked_fill(~valid, -torch.inf).softmax(-1)
        # Ordinary row weights do not define a per-Patch competitive assignment.
        assignment = torch.zeros_like(weights)
        mass = torch.zeros_like(weights.sum(-1))
    elif operator == "COMP":
        log_assignment = safe.log_softmax(dim=1)
        assignment = torch.where(valid, log_assignment.exp(), torch.zeros_like(scores))
        mass = assignment.sum(-1)
        weights = log_assignment.masked_fill(~valid, -torch.inf).softmax(-1)
    else:
        raise ValueError("unknown attention operator")
    return weights, assignment, mass


class C4StyleSelfAttention(nn.Module):
    """C4-style residual attention/FF arithmetic with locked dropout zero."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN)
        self.qkv = nn.Linear(HIDDEN, HIDDEN*3, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN*2),
                                nn.GELU(), nn.Linear(HIDDEN*2, HIDDEN))

    def forward(self, x):
        q, k, v = self.qkv(self.norm(x)).chunk(3, -1)
        a = (q @ k.transpose(-1, -2)/math.sqrt(HIDDEN)).softmax(-1)
        h = x+a@v
        return h+self.ff(h)


class RNATokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        groups = np.random.default_rng(117301).permutation(512).reshape(SLOTS, 64)
        self.register_buffer("groups", torch.as_tensor(groups, dtype=torch.long))
        self.weight = nn.Parameter(torch.randn(SLOTS, 64, HIDDEN)/8.)
        self.bias = nn.Parameter(torch.zeros(SLOTS, HIDDEN))
        self.norm = nn.LayerNorm(HIDDEN)
        self.inter = C4StyleSelfAttention()

    def forward(self, rna):
        grouped = rna[:, self.groups]
        raw = torch.einsum("bkg,kgd->bkd", grouped, self.weight)/8.+self.bias
        raw = self.norm(torch.nn.functional.elu(raw))
        return self.inter(raw)


class MultiQuery(nn.Module):
    def __init__(self, clinical_dim, arm):
        super().__init__()
        if arm not in ("QUERY", "ATTN", "COMP"):
            raise ValueError("MultiQuery implements QUERY/ATTN/COMP only")
        if (isinstance(clinical_dim, bool) or int(clinical_dim) != clinical_dim or clinical_dim < 1):
            raise ValueError("clinical dimension must be a positive integer")
        self.clinical_dim, self.arm = int(clinical_dim), arm
        # No arm-dependent initialization draw or module construction occurs.
        self.rna = RNATokenizer()
        self.patch = nn.Sequential(nn.Linear(768, HIDDEN), nn.LayerNorm(HIDDEN))
        self.attention_norm = nn.LayerNorm(HIDDEN)
        self.query = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.key = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.value = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.slot_down = nn.Linear(HIDDEN, RANK, bias=False)
        self.clinical_down = nn.Linear(self.clinical_dim, RANK, bias=False)
        self.clinical_up = nn.Linear(RANK, HIDDEN, bias=False)
        nn.init.zeros_(self.clinical_up.weight)
        self.update_ff = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN*2),
                                       nn.GELU(), nn.Linear(HIDDEN*2, HIDDEN))
        self.head = nn.Sequential(nn.Linear(SLOTS*HIDDEN, 16), nn.GELU(), nn.Linear(16, 1, bias=False))
        nn.init.zeros_(self.head[-1].weight)

    def _query_vector(self, slot_source, clinical):
        normalized = self.attention_norm(slot_source)
        conditioned = self.clinical_up(self.slot_down(normalized)*self.clinical_down(clinical)[:, None, :])
        return self.query(normalized)+conditioned

    def forward(self, patch, mask, RNA, Clinical, query_rna=None, query_clinical=None,
                value_off=False, return_details=False):
        if (not isinstance(patch, torch.Tensor) or patch.ndim != 3 or patch.shape[-1] != 768
                or min(patch.shape[:2]) < 1 or not patch.is_floating_point()
                or mask.shape != patch.shape[:2] or mask.dtype != torch.bool
                or mask.device != patch.device or not mask.any(-1).all()):
            raise ValueError("Patch input requires a nonempty observed set for each WSI")
        if not torch.isfinite(patch[mask]).all():
            raise ValueError("nonfinite observed Patch feature")
        for name, x, width in (("RNA", RNA, 512), ("Clinical", Clinical, self.clinical_dim)):
            if (not isinstance(x, torch.Tensor) or x.shape != (len(patch), width)
                    or x.device != patch.device or x.dtype != patch.dtype or not torch.isfinite(x).all()):
                raise ValueError("invalid " + name + " input")
        for name, altered, original in (("RNA", query_rna, RNA), ("Clinical", query_clinical, Clinical)):
            if altered is not None and (not isinstance(altered, torch.Tensor) or altered.shape != original.shape
                    or altered.device != original.device or altered.dtype != original.dtype or not torch.isfinite(altered).all()):
                raise ValueError("invalid " + name + " query override")
        if not isinstance(value_off, bool):
            raise ValueError("value_off must be Boolean")
        initial = self.rna(RNA)
        altered_initial = None if query_rna is None else self.rna(query_rna)
        delta = None if altered_initial is None else altered_initial-initial
        clinical = Clinical if query_clinical is None else query_clinical
        enabled = self.arm != "QUERY" and not value_off
        h = initial
        # In QUERY/VALUE_OFF the actual prediction does not evaluate Patch or
        # Clinical-score modules, and neither receives a risk gradient.
        if enabled or return_details:
            safe = torch.where(mask[..., None], patch, torch.zeros_like(patch))*math.sqrt(768)
            p = self.patch(safe)
            p = torch.where(mask[..., None], p, torch.zeros_like(p))
            normalized_patch = self.attention_norm(p)
            keys, values = self.key(normalized_patch), self.value(normalized_patch)
            keys = torch.where(mask[..., None], keys, torch.zeros_like(keys))
            values = torch.where(mask[..., None], values, torch.zeros_like(values))
        history = []
        for _ in range(2):
            source = h if delta is None else h+delta
            if enabled:
                q = self._query_vector(source, clinical)
                scores = q @ keys.transpose(-1, -2)/math.sqrt(HIDDEN)
                attention, assignment, slot_mass = attention_weights(scores, mask, self.arm)
                message = attention@values
            else:
                message = torch.zeros_like(h)
                if return_details:
                    q = self._query_vector(source, clinical)
                    scores = q @ keys.transpose(-1, -2)/math.sqrt(HIDDEN)
                    attention = scores.new_zeros(scores.shape)
                    assignment = torch.zeros_like(attention)
                    slot_mass = scores.new_zeros(scores.shape[:2])
            previous = h
            with_message = h+message
            h = with_message+self.update_ff(with_message)
            if return_details:
                history.append({"input_slots": previous, "score_source": source, "query": q,
                                "scores": scores, "attention": attention, "assignment": assignment,
                                "slot_mass": slot_mass, "message": message, "output_slots": h})
        logit = self.head(h.flatten(1)).squeeze(-1)
        unit = torch.tanh(logit)
        if not torch.isfinite(unit).all():
            raise ValueError("nonfinite bounded residual output")
        result = {"unit": unit}
        if return_details:
            result.update(slots=h, initial_slots=initial, attention=attention, query=q, message=message,
                          assignment=assignment, slot_mass=slot_mass, logit=logit, key=keys, value=values,
                          query_initial_shift=torch.zeros_like(initial) if delta is None else delta,
                          rounds=history, patch_message_enabled=enabled)
        return result

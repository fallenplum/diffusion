"""Low-capacity local routing outside a shared frozen strong main effect.

Patch inputs are row-normalized frozen CONCH features prepared by the data
pipeline. This module multiplies them by sqrt(768), without refitting or
normalizing the resulting local message. Main risks and patient/WSI pooling
belong to the caller. This operator has no final tanh and no direct Omics or
Clinical path into its local prediction head.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class LocalRouting(nn.Module):
    def __init__(self, clinical_dim, arm, hidden=16):
        super().__init__()
        if arm not in ("R1", "R2", "R3"):
            raise ValueError("local model only implements R1/R2/R3")
        if isinstance(clinical_dim, bool) or int(clinical_dim) != clinical_dim or clinical_dim < 1:
            raise ValueError("clinical dimension must be a positive integer")
        if hidden != 16:
            raise ValueError("locked local routing width is sixteen")
        self.clinical_dim, self.arm, self.hidden = int(clinical_dim), arm, int(hidden)
        # Common parameter draws precede any arm-dependent parameter creation.
        self.keys = nn.Linear(768, hidden, bias=False)
        self.values = nn.Linear(768, hidden, bias=False)
        self.head = nn.Linear(hidden, 1, bias=False)
        self.q0 = nn.Parameter(torch.randn(hidden)*0.1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.1)
        with torch.random.fork_rng(devices=[]):
            self.rna = nn.Linear(512, hidden, bias=False) if arm in ("R2", "R3") else None
            self.clinical = nn.Linear(self.clinical_dim, hidden, bias=False) if arm == "R3" else None
            if self.rna is not None:
                nn.init.zeros_(self.rna.weight)
            if self.clinical is not None:
                nn.init.zeros_(self.clinical.weight)

    def forward(self, patch, mask, RNA, Clinical, *, query_rna=None, query_clinical=None,
                return_details=False):
        if (not isinstance(patch, torch.Tensor) or patch.ndim != 3 or patch.shape[-1] != 768
                or not patch.is_floating_point() or patch.shape[1] < 1):
            raise ValueError("patch must be a nonempty floating B by N by 768 tensor")
        batch = len(patch)
        if (batch < 1 or mask.shape != patch.shape[:2] or mask.dtype != torch.bool
                or mask.device != patch.device or not mask.any(-1).all()):
            raise ValueError("each WSI needs at least one real Patch in its Boolean mask")
        if not torch.isfinite(patch[mask]).all():
            raise ValueError("nonfinite observed Patch feature")
        for name, x, width in (("RNA", RNA, 512), ("Clinical", Clinical, self.clinical_dim)):
            if (not isinstance(x, torch.Tensor) or x.shape != (batch, width)
                    or x.device != patch.device or not x.is_floating_point() or not torch.isfinite(x).all()):
                raise ValueError("invalid " + name + " query feature contract")
        if query_rna is not None:
            if self.rna is None:
                raise ValueError("R1 has no RNA query branch to override")
            if (not isinstance(query_rna, torch.Tensor) or query_rna.shape != RNA.shape
                    or query_rna.device != RNA.device or query_rna.dtype != RNA.dtype
                    or not torch.isfinite(query_rna).all()):
                raise ValueError("invalid RNA query override")
        if query_clinical is not None:
            if self.clinical is None:
                raise ValueError("this arm has no Clinical query branch to override")
            if (not isinstance(query_clinical, torch.Tensor) or query_clinical.shape != Clinical.shape
                    or query_clinical.device != Clinical.device or query_clinical.dtype != Clinical.dtype
                    or not torch.isfinite(query_clinical).all()):
                raise ValueError("invalid Clinical query override")
        safe = torch.where(mask[..., None], patch, torch.zeros_like(patch))*math.sqrt(768)
        key, value = torch.tanh(self.keys(safe)), self.values(safe)
        # Invalid entries are zero already because both maps have no bias.
        prequery = self.q0[None].expand(batch, -1)
        if self.rna is not None:
            prequery = prequery+self.rna(RNA if query_rna is None else query_rna)
        if self.clinical is not None:
            prequery = prequery+self.clinical(Clinical if query_clinical is None else query_clinical)
        query = torch.tanh(prequery)
        logits = torch.einsum("bd,bnd->bn", query, key)/math.sqrt(self.hidden)
        attention = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        mean_value = value.sum(1)/mask.sum(1, keepdim=True)
        weighted_value = torch.einsum("bn,bnd->bd", attention, value)
        feature = weighted_value-mean_value
        unit = self.head(feature).squeeze(-1)
        if not torch.isfinite(unit).all():
            raise ValueError("nonfinite local output")
        result = {"unit": unit, "attention": attention}
        if return_details:
            entropy = -(attention*attention.clamp_min(torch.finfo(attention.dtype).tiny).log()).sum(-1)
            count = mask.sum(-1)
            denominator = count.to(attention.dtype).clamp_min(2).log()
            normalized_entropy = torch.where(count > 1, entropy/denominator, torch.zeros_like(entropy))
            result.update(query=query, prequery=prequery, key=key, value=value, logits=logits,
                          feature=feature, weighted_value=weighted_value, mean_value=mean_value,
                          valid_count=count, entropy=entropy, normalized_entropy=normalized_entropy,
                          effective_patch_count=entropy.exp())
        return result

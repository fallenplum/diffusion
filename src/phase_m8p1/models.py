from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from src.phase_m2.nn_blocks import MultimodalBackbone


DIM = 96


@contextmanager
def isolated_seed(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


class StrongAnchorResidualModel(nn.Module):
    """C0 offset plus additive pathology/omics corrections only."""

    def __init__(self, clinical_dim: int, variant: str, seed: int, fold: int):
        super().__init__()
        if variant not in {"SP", "SO", "SPO"}:
            raise ValueError(variant)
        self.variant = variant
        self.backbone = MultimodalBackbone(clinical_dim, dim=DIM, path_tokens=32)
        with isolated_seed(700000 + seed % 1000000 * 17 + fold):
            self.path_head = nn.Sequential(nn.Linear(DIM, 32), nn.GELU(), nn.Linear(32, 1)) if variant in {"SP", "SPO"} else None
            self.omic_head = nn.Sequential(nn.Linear(DIM, 32), nn.GELU(), nn.Linear(32, 1)) if variant in {"SO", "SPO"} else None
        for head in (self.path_head, self.omic_head):
            if head is not None:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, pathology, rna, cna, clinical, anchor_offset):
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        delta = anchor_offset.detach() * 0.0
        if self.path_head is not None:
            delta = delta + self.path_head(encoded.pathology.mean(1)).squeeze(-1)
        if self.omic_head is not None:
            delta = delta + self.omic_head(encoded.omics.mean(1)).squeeze(-1)
        return anchor_offset.detach() + delta, delta

    def delta_parts(self, pathology, rna, cna, clinical):
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        p = encoded.pathology.mean(1)
        o = encoded.omics.mean(1)
        dp = self.path_head(p).squeeze(-1) if self.path_head is not None else torch.zeros(len(p), device=p.device)
        do = self.omic_head(o).squeeze(-1) if self.omic_head is not None else torch.zeros(len(o), device=o.device)
        return dp, do


def build_model(variant: str, clinical_dim: int, seed: int, fold: int):
    return StrongAnchorResidualModel(clinical_dim, variant, seed, fold)


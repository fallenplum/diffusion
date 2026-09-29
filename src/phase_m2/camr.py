from __future__ import annotations

import torch
from torch import nn

from .nn_blocks import MultimodalBackbone, RiskHead, mean_pool


class CAMRModel(nn.Module):
    """B3: shared/specific representations with cross-modal alignment."""

    def __init__(self, clinical_dim: int, dim: int = 96) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.shared = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, dim), nn.Tanh()) for name in ("path", "omics", "clinical")})
        self.private = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for name in ("path", "omics", "clinical")})
        self.head = RiskHead(dim * 6, hidden=dim * 2)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        base = {
            "path": mean_pool(encoded.pathology),
            "omics": mean_pool(encoded.omics),
            "clinical": mean_pool(encoded.clinical),
        }
        shared = {name: self.shared[name](value) for name, value in base.items()}
        private = {name: self.private[name](value) for name, value in base.items()}
        fused = torch.cat([shared["path"], shared["omics"], shared["clinical"], private["path"], private["omics"], private["clinical"]], dim=-1)
        align = (shared["path"] - shared["omics"]).square().mean() + (shared["path"] - shared["clinical"]).square().mean() + (shared["omics"] - shared["clinical"]).square().mean()
        orth = sum((shared[name] * private[name]).mean().square() for name in base)
        return self.head(fused), 0.1 * align + 0.01 * orth

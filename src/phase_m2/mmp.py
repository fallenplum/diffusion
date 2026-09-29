from __future__ import annotations

import torch
from torch import nn

from .nn_blocks import MultimodalBackbone, RiskHead


class MMPModel(nn.Module):
    """B4: prototype tokens followed by prototype-level cross-attention."""

    def __init__(self, clinical_dim: int, dim: int = 96, prototypes: int = 8) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.path_prototypes = nn.Parameter(torch.randn(1, prototypes, dim) * 0.02)
        self.omics_prototypes = nn.Parameter(torch.randn(1, prototypes, dim) * 0.02)
        self.path_assign = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.omics_assign = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.prototype_cross = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.clinical_cross = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        batch = pathology.shape[0]
        p_proto, _ = self.path_assign(self.path_prototypes.expand(batch, -1, -1), encoded.pathology, encoded.pathology)
        o_proto, _ = self.omics_assign(self.omics_prototypes.expand(batch, -1, -1), encoded.omics, encoded.omics)
        p_cross, _ = self.prototype_cross(p_proto, o_proto, o_proto)
        c, _ = self.clinical_cross(encoded.clinical, torch.cat([p_cross, o_proto], dim=1), torch.cat([p_cross, o_proto], dim=1))
        return self.head(c[:, 0]), pathology.new_zeros(())

from __future__ import annotations

import torch
from torch import nn

from .nn_blocks import MultimodalBackbone, RiskHead


class MGCTModel(nn.Module):
    """B2: mutual-guided bidirectional cross-modality transformer."""

    def __init__(self, clinical_dim: int, dim: int = 96) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.path_to_omics = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.omics_to_path = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.clinical_query = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.path_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.omics_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.path_norm = nn.LayerNorm(dim)
        self.omics_norm = nn.LayerNorm(dim)
        self.clinical_norm = nn.LayerNorm(dim)
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        p_guided, _ = self.omics_to_path(encoded.pathology, encoded.omics, encoded.omics)
        o_guided, _ = self.path_to_omics(encoded.omics, encoded.pathology, encoded.pathology)
        p_gate = self.path_gate(torch.cat([encoded.pathology, p_guided], dim=-1))
        o_gate = self.omics_gate(torch.cat([encoded.omics, o_guided], dim=-1))
        p = self.path_norm(encoded.pathology + p_gate * p_guided)
        o = self.omics_norm(encoded.omics + o_gate * o_guided)
        clinical, _ = self.clinical_query(encoded.clinical, torch.cat([p, o], dim=1), torch.cat([p, o], dim=1))
        return self.head(self.clinical_norm(clinical[:, 0])), pathology.new_zeros(())

from __future__ import annotations

import torch
from torch import nn

from .nn_blocks import MultimodalBackbone, RiskHead


class TransSurvModel(nn.Module):
    """B1: TransSurv-style cross-modal self-attention over modality tokens."""

    def __init__(self, clinical_dim: int, dim: int = 96, layers: int = 2) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embeddings = nn.Parameter(torch.zeros(4, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        batch = pathology.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        sequence = torch.cat([cls, encoded.pathology, encoded.omics, encoded.clinical], dim=1)
        offsets = torch.cat(
            [
                self.modality_embeddings[0:1],
                self.modality_embeddings[1].expand(encoded.pathology.shape[1], -1),
                self.modality_embeddings[2].expand(encoded.omics.shape[1], -1),
                self.modality_embeddings[3:4],
            ],
            dim=0,
        ).unsqueeze(0)
        transformed = self.transformer(sequence + offsets)
        return self.head(transformed[:, 0]), pathology.new_zeros(())

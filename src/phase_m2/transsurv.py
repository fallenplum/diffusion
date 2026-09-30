from __future__ import annotations

import torch
from torch import nn

from .nn_blocks import MultimodalBackbone, RiskHead


class TransSurvModel(nn.Module):
    """TransSurv fusion transformer with explicit cross-modal interactions.

    The original feature extractor is replaced by UNI2-h, while the
    histopathology branch, multi-omics branch, cross-attention fusion and
    clinical integration remain explicit.
    """

    def __init__(self, clinical_dim: int, dim: int = 96) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        path_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        omic_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.path_branch = nn.TransformerEncoder(path_layer, num_layers=2)
        self.omic_branch = nn.TransformerEncoder(omic_layer, num_layers=2)
        self.path_to_omic = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.omic_to_path = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.path_norm = nn.LayerNorm(dim)
        self.omic_norm = nn.LayerNorm(dim)
        self.clinical_fusion = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.fusion_norm = nn.LayerNorm(dim)
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        path = self.path_branch(encoded.pathology)
        omic = self.omic_branch(encoded.omics)
        path_update, _ = self.path_to_omic(path, omic, omic)
        omic_update, _ = self.omic_to_path(omic, path, path)
        path = self.path_norm(path + path_update)
        omic = self.omic_norm(omic + omic_update)
        clinical_context, _ = self.clinical_fusion(encoded.clinical, torch.cat([path, omic], dim=1), torch.cat([path, omic], dim=1))
        return self.head(self.fusion_norm(clinical_context[:, 0])), pathology.new_zeros(())


from __future__ import annotations

import torch
from torch import nn

from .nn_blocks import MultimodalBackbone, RiskHead


class GatedAttentionPool(nn.Module):
    """The gated attention pooling block used by the MGCT implementation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.tanh = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.sigmoid = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.score = nn.Linear(hidden, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score(self.tanh(tokens) * self.sigmoid(tokens)).squeeze(-1)
        attention = torch.softmax(score, dim=1)
        return torch.sum(attention.unsqueeze(-1) * tokens, dim=1), attention


class MutualGuidedFusionLayer(nn.Module):
    """One official MGCA-style bidirectional fusion layer.

    Queries are formed from one modality while keys and values come from the
    other modality. Each direction is followed by gated attention pooling,
    feed-forward refinement and a fused representation.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.path_mgca = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.genomic_mgca = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.path_norm1 = nn.LayerNorm(dim)
        self.genomic_norm1 = nn.LayerNorm(dim)
        self.path_norm2 = nn.LayerNorm(dim)
        self.genomic_norm2 = nn.LayerNorm(dim)
        hidden = dim * 2
        self.path_ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, dim))
        self.genomic_ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, dim))
        self.path_pool = GatedAttentionPool(dim)
        self.genomic_pool = GatedAttentionPool(dim)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, pathology: torch.Tensor, genomic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path_update, _ = self.path_mgca(pathology, genomic, genomic)
        genomic_update, _ = self.genomic_mgca(genomic, pathology, pathology)
        pathology = self.path_norm1(pathology + path_update)
        genomic = self.genomic_norm1(genomic + genomic_update)
        pathology = self.path_norm2(pathology + self.path_ffn(pathology))
        genomic = self.genomic_norm2(genomic + self.genomic_ffn(genomic))
        path_repr, _ = self.path_pool(pathology)
        genomic_repr, _ = self.genomic_pool(genomic)
        fused = self.fusion(torch.cat([path_repr, genomic_repr], dim=-1)).unsqueeze(1)
        return pathology, genomic, fused


class MGCTModel(nn.Module):
    """MGCT adapted to UNI2-h tokens and RNA+CNA+clinical inputs.

    The two-stage deep mutual-guided fusion is retained. UNI2-h replaces the
    original histology feature extractor; RNA/CNA chunk tokens replace the
    original fixed omic groups.
    """

    def __init__(self, clinical_dim: int, dim: int = 96) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.stage1 = MutualGuidedFusionLayer(dim)
        self.stage2 = nn.ModuleList([MutualGuidedFusionLayer(dim) for _ in range(2)])
        self.clinical_query = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.final_norm = nn.LayerNorm(dim)
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        p, g, stage1 = self.stage1(encoded.pathology, encoded.omics)
        fused = stage1
        for layer in self.stage2:
            p, g, stage = layer(torch.cat([p, fused], dim=1), torch.cat([g, fused], dim=1))
            fused = stage
        clinical_context, _ = self.clinical_query(encoded.clinical, torch.cat([fused, p, g], dim=1), torch.cat([fused, p, g], dim=1))
        return self.head(self.final_norm(clinical_context[:, 0])), pathology.new_zeros(())


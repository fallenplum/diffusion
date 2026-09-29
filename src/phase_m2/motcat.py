from __future__ import annotations

import math

import torch
from torch import nn

from .nn_blocks import MultimodalBackbone, RiskHead


def sinkhorn_transport(scores: torch.Tensor, iterations: int = 5, epsilon: float = 0.2) -> torch.Tensor:
    """Entropic OT coupling between pathology and omics tokens."""
    log_kernel = scores / epsilon
    n, m = scores.shape[-2:]
    log_mu = scores.new_full((n,), -math.log(max(1, n)))
    log_nu = scores.new_full((m,), -math.log(max(1, m)))
    u = scores.new_zeros(n)
    v = scores.new_zeros(m)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=-1)
        v = log_nu - torch.logsumexp(log_kernel + u.unsqueeze(-1), dim=-2)
    return torch.exp(log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2))


class MOTCatModel(nn.Module):
    """B5: MCAT/MOTCat-style co-attention with transport-guided alignment."""

    def __init__(self, clinical_dim: int, dim: int = 96) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.path_update = nn.Linear(dim, dim)
        self.omics_update = nn.Linear(dim, dim)
        self.clinical_attention = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.path_norm = nn.LayerNorm(dim)
        self.omics_norm = nn.LayerNorm(dim)
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        scale = math.sqrt(encoded.pathology.shape[-1])
        scores = torch.matmul(encoded.pathology, encoded.omics.transpose(1, 2)) / scale
        transport = torch.stack([sinkhorn_transport(item) for item in scores], dim=0)
        p_aligned = torch.matmul(transport, encoded.omics)
        o_aligned = torch.matmul(transport.transpose(1, 2), encoded.pathology)
        p = self.path_norm(encoded.pathology + self.path_update(p_aligned))
        o = self.omics_norm(encoded.omics + self.omics_update(o_aligned))
        clinical, _ = self.clinical_attention(encoded.clinical, torch.cat([p, o], dim=1), torch.cat([p, o], dim=1))
        return self.head(clinical[:, 0]), pathology.new_zeros(())

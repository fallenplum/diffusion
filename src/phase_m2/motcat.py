from __future__ import annotations

import math

import torch
from torch import nn

from .mgct import GatedAttentionPool
from .nn_blocks import MultimodalBackbone, RiskHead


def sinkhorn_transport(scores: torch.Tensor, iterations: int = 7, epsilon: float = 0.2) -> torch.Tensor:
    """Compute an entropic OT coupling for each batch item."""

    log_kernel = scores / epsilon
    n, m = scores.shape[-2:]
    log_mu = scores.new_full((*scores.shape[:-2], n), -math.log(max(1, n)))
    log_nu = scores.new_full((*scores.shape[:-2], m), -math.log(max(1, m)))
    u = scores.new_zeros(log_mu.shape)
    v = scores.new_zeros(log_nu.shape)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(log_kernel + v.unsqueeze(-2), dim=-1)
        v = log_nu - torch.logsumexp(log_kernel + u.unsqueeze(-1), dim=-2)
    return torch.exp(log_kernel + u.unsqueeze(-1) + v.unsqueeze(-2))


class MOTCatModel(nn.Module):
    """MOTCat with OT-guided cross-modal alignment and clinical fusion.

    MCAT uses ordinary co-attention. MOTCat first obtains a balanced
    transport plan between patch and genomic tokens, then updates both token
    streams with the transported values and retains a global-structure loss.
    """

    def __init__(self, clinical_dim: int, dim: int = 96) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.path_projection = nn.Linear(dim, dim)
        self.omic_projection = nn.Linear(dim, dim)
        self.path_update = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.omic_update = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.path_norm = nn.LayerNorm(dim)
        self.omic_norm = nn.LayerNorm(dim)
        self.path_pool = GatedAttentionPool(dim)
        self.omic_pool = GatedAttentionPool(dim)
        self.global_fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.clinical_attention = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        path = self.path_projection(encoded.pathology)
        omic = self.omic_projection(encoded.omics)
        similarity = torch.matmul(path, omic.transpose(1, 2)) / math.sqrt(path.shape[-1])
        coupling = sinkhorn_transport(similarity)
        path_aligned = torch.matmul(coupling, omic)
        omic_aligned = torch.matmul(coupling.transpose(1, 2), path)
        path = self.path_norm(path + self.path_update(path_aligned))
        omic = self.omic_norm(omic + self.omic_update(omic_aligned))
        path_repr, _ = self.path_pool(path)
        omic_repr, _ = self.omic_pool(omic)
        fused = self.global_fusion(torch.cat([path_repr, omic_repr], dim=-1)).unsqueeze(1)
        clinical_context, _ = self.clinical_attention(encoded.clinical, torch.cat([fused, path, omic], dim=1), torch.cat([fused, path, omic], dim=1))
        # Penalise departures from the uniform OT marginals, as in the global
        # structure consistency term in the official MOTCat formulation.
        marginal_path = coupling.sum(dim=-1)
        marginal_omic = coupling.sum(dim=-2)
        target_path = marginal_path.new_full(marginal_path.shape, 1.0 / path.shape[1])
        target_omic = marginal_omic.new_full(marginal_omic.shape, 1.0 / omic.shape[1])
        structure = (marginal_path - target_path).square().mean() + (marginal_omic - target_omic).square().mean()
        return self.head(clinical_context[:, 0]), 0.01 * structure


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EncodedModalities:
    pathology: torch.Tensor
    omics: torch.Tensor
    clinical: torch.Tensor


class PathologyEncoder(nn.Module):
    """Project 1536-D UNI2-h patch tokens and pool to a fixed token budget."""

    def __init__(self, input_dim: int = 1536, dim: int = 96, output_tokens: int = 32) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU())
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.projection(tokens.float())
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.output_tokens).transpose(1, 2)
        return self.norm(x)


class ChunkTokenEncoder(nn.Module):
    def __init__(self, input_dim: int, n_tokens: int, dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_tokens = min(n_tokens, max(1, input_dim))
        boundaries = torch.linspace(0, input_dim, self.n_tokens + 1).round().to(dtype=torch.long).tolist()
        self.slices = [(boundaries[i], boundaries[i + 1]) for i in range(self.n_tokens)]
        self.projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(max(1, end - start)), nn.Linear(max(1, end - start), dim), nn.GELU())
            for start, end in self.slices
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        tokens = []
        for (start, end), projection in zip(self.slices, self.projections):
            chunk = values[:, start:end]
            if chunk.shape[1] == 0:
                chunk = values[:, -1:].contiguous()
            tokens.append(projection(chunk).unsqueeze(1))
        return self.norm(torch.cat(tokens, dim=1))


class OmicsEncoder(nn.Module):
    def __init__(self, rna_dim: int = 512, cna_dim: int = 256, dim: int = 96) -> None:
        super().__init__()
        self.rna = ChunkTokenEncoder(rna_dim, 8, dim)
        self.cna = ChunkTokenEncoder(cna_dim, 4, dim)

    def forward(self, rna: torch.Tensor, cna: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.rna(rna), self.cna(cna)], dim=1)


class ClinicalEncoder(nn.Module):
    def __init__(self, clinical_dim: int, dim: int = 96) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(clinical_dim), nn.Linear(clinical_dim, dim), nn.GELU())

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.projection(clinical.float()).unsqueeze(1)


class RiskHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 96) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class MultimodalBackbone(nn.Module):
    def __init__(self, clinical_dim: int, dim: int = 96, path_tokens: int = 32) -> None:
        super().__init__()
        self.pathology_encoder = PathologyEncoder(dim=dim, output_tokens=path_tokens)
        self.omics_encoder = OmicsEncoder(dim=dim)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, dim=dim)

    def encode(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> EncodedModalities:
        return EncodedModalities(
            pathology=self.pathology_encoder(pathology),
            omics=self.omics_encoder(rna, cna),
            clinical=self.clinical_encoder(clinical),
        )


def mean_pool(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=1)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.phase_m2.mgct import GatedAttentionPool
from src.phase_m2.mcat import SixSignatureEncoder
from src.phase_m2.nn_blocks import ClinicalEncoder, RiskHead
from src.phase_m3r2.models import ClinicalOnlyModel, SimpleFusionModel


DIM = 96
HEADS = 4


class GroupedOmicEncoder(nn.Module):
    """Official-MCAT style independent encoder for an arbitrary group count."""

    def __init__(self, widths: tuple[tuple[int, int], ...], mode: str, dim: int = DIM) -> None:
        super().__init__()
        if mode not in {"six", "rna", "rc"}:
            raise ValueError(mode)
        if not widths or any(int(r) <= 0 or (mode == "rc" and int(c) <= 0) for r, c in widths):
            raise ValueError(f"invalid grouped widths for {mode}: {widths}")
        self.widths = tuple((int(r), int(c)) for r, c in widths)
        self.mode = mode
        # The six-signature control is required to reuse the unchanged
        # official MCAT encoder.  The generic encoder below is only used for
        # the new 50-group RNA/RC token families.
        self.six_encoder = SixSignatureEncoder(self.widths, dim) if mode == "six" else None
        self.group_networks = nn.ModuleList()
        for rna_width, cna_width in (self.widths if mode != "six" else ()):
            input_width = rna_width if mode == "rna" else rna_width + cna_width
            self.group_networks.append(
                nn.Sequential(
                    nn.Linear(input_width, dim),
                    nn.ELU(),
                    nn.AlphaDropout(p=0.25),
                    nn.Linear(dim, dim),
                    nn.ELU(),
                    nn.AlphaDropout(p=0.25),
                )
            )

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        if self.mode == "six":
            assert self.six_encoder is not None
            return self.six_encoder(groups)
        values = []
        for index, (rna_width, cna_width) in enumerate(self.widths):
            rna = groups[:, index, :rna_width, 0].float()
            if self.mode == "rna":
                value = rna
            else:
                cna = groups[:, index, :cna_width, 1].float()
                value = torch.cat([rna, cna], dim=-1)
            values.append(self.group_networks[index](value))
        return torch.stack(values, dim=1)


@dataclass
class MCATStreams:
    path: torch.Tensor
    omic: torch.Tensor
    clinical: torch.Tensor


class MCATRepresentationM6A(nn.Module):
    """M5R's dense MCAT representation with only token-family changes."""

    def __init__(
        self,
        clinical_dim: int,
        group_widths: tuple[tuple[int, int], ...],
        omic_mode: str,
        tokenizer: str,
        prototype_centers: np.ndarray | None = None,
        dim: int = DIM,
    ) -> None:
        super().__init__()
        if tokenizer not in {"adaptive", "prototype"}:
            raise ValueError(tokenizer)
        self.tokenizer = tokenizer
        self.omic_mode = omic_mode
        self.patch_projection = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, dim), nn.GELU())
        self.path_norm = nn.LayerNorm(dim)
        centers = np.zeros((16, 1536), dtype=np.float32) if prototype_centers is None else np.asarray(prototype_centers, dtype=np.float32)
        if centers.shape != (16, 1536):
            raise ValueError(f"prototype centers must have shape (16,1536), got {centers.shape}")
        self.register_buffer("prototype_centers", torch.from_numpy(centers), persistent=False)
        self.omic_encoder = GroupedOmicEncoder(group_widths, omic_mode, dim)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, dim=dim)
        self.coattn = nn.MultiheadAttention(dim, HEADS, dropout=0.1, batch_first=True)
        path_layer = nn.TransformerEncoderLayer(dim, HEADS, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        omic_layer = nn.TransformerEncoderLayer(dim, HEADS, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        self.path_transformer = nn.TransformerEncoder(path_layer, num_layers=2)
        self.omic_transformer = nn.TransformerEncoder(omic_layer, num_layers=2)
        self.path_attention_head = GatedAttentionPool(dim)
        self.omic_attention_head = GatedAttentionPool(dim)
        self.path_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.omic_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))

    def path_tokens(self, pathology: torch.Tensor) -> torch.Tensor:
        h = self.patch_projection(pathology.float())
        if self.tokenizer == "adaptive":
            return self.path_norm(F.adaptive_avg_pool1d(h.transpose(1, 2), 32).transpose(1, 2))
        centers = self.patch_projection(self.prototype_centers.to(device=h.device, dtype=h.dtype))
        scores = torch.matmul(h, centers.transpose(0, 1)) / math.sqrt(h.shape[-1])
        assignment = torch.softmax(scores.transpose(1, 2), dim=-1)
        return self.path_norm(torch.matmul(assignment, h) + centers.unsqueeze(0))

    def encoded_inputs(self, pathology: torch.Tensor, clinical: torch.Tensor, groups: torch.Tensor):
        return self.path_tokens(pathology), self.omic_encoder(groups), self.clinical_encoder(clinical)


class M6AModel(nn.Module):
    """G0-G5 use only dense omic-query -> pathology co-attention."""

    def __init__(self, variant: str, clinical_dim: int, group_widths: tuple[tuple[int, int], ...], prototype_centers: np.ndarray, dim: int = DIM) -> None:
        super().__init__()
        specs = {
            "G0": ("six", "adaptive"),
            "G1": ("six", "prototype"),
            "G2": ("rna", "prototype"),
            "G3": ("rc", "prototype"),
            "G4": ("rc", "prototype"),
            "G5": ("rc", "adaptive"),
        }
        if variant not in specs:
            raise ValueError(variant)
        mode, tokenizer = specs[variant]
        self.variant = variant
        self.representation = MCATRepresentationM6A(clinical_dim, group_widths, mode, tokenizer, prototype_centers, dim)
        self.mm = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.clinical_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = RiskHead(dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor, groups: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del rna, cna
        if groups is None:
            raise ValueError("M6A MCAT requires grouped omic tokens")
        h_path, h_omic, h_clinical = self.representation.encoded_inputs(pathology, clinical, groups)
        path_tokens, _ = self.representation.coattn(h_omic, h_path, h_path)
        path_tokens = self.representation.path_transformer(path_tokens)
        omic_tokens = self.representation.omic_transformer(h_omic)
        h_path_repr, _ = self.representation.path_attention_head(path_tokens)
        h_omic_repr, _ = self.representation.omic_attention_head(omic_tokens)
        streams = MCATStreams(self.representation.path_rho(h_path_repr), self.representation.omic_rho(h_omic_repr), h_clinical[:, 0])
        fused = self.mm(torch.cat([streams.path, streams.omic], dim=-1))
        return self.head(self.clinical_gate(torch.cat([fused, streams.clinical], dim=-1))), pathology.new_zeros(())

    @torch.no_grad()
    def pathway_diagnostics(self, pathology: torch.Tensor, clinical: torch.Tensor, groups: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        h_path, h_omic, _ = self.representation.encoded_inputs(pathology, clinical, groups)
        path_tokens, weights = self.representation.coattn(h_omic, h_path, h_path, need_weights=True, average_attn_weights=False)
        path_tokens = self.representation.path_transformer(path_tokens)
        omic_tokens = self.representation.omic_transformer(h_omic)
        _, pool_weights = self.representation.omic_attention_head(omic_tokens)
        weights = weights.clamp_min(1e-12)
        entropy = (-weights * weights.log()).sum(dim=-1).mean(dim=(1, 2))
        return {
            "attention_entropy": entropy,
            "omic_token_norm": omic_tokens.norm(dim=-1),
            "pool_weights": pool_weights,
        }


def control_seed(method: str, seed: int, fold: int) -> int:
    from src.phase_m2.common import stable_int
    return 800000 + stable_int(f"m6a-control:{method}:{seed}:{fold}", modulo=1_900_000_000)


def build_model(variant: str, clinical_dim: int, group_widths: tuple[tuple[int, int], ...], prototype_centers: np.ndarray, seed: int, fold: int) -> nn.Module:
    if variant in {"G0", "G1", "G2", "G3", "G4", "G5"}:
        return M6AModel(variant, clinical_dim, group_widths, prototype_centers)
    if variant == "B0":
        cs = control_seed(variant, seed, fold)
        return SimpleFusionModel(clinical_dim, variant="S3", variant_seed=cs, concat_seed=cs + 1)
    if variant == "C0":
        return ClinicalOnlyModel(clinical_dim, variant_seed=control_seed(variant, seed, fold))
    raise ValueError(variant)

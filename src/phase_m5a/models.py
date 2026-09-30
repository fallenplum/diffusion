from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.phase_m2.mcat import SixSignatureEncoder
from src.phase_m2.mgct import GatedAttentionPool
from src.phase_m2.nn_blocks import ClinicalEncoder, RiskHead
from src.phase_m3r2.models import ClinicalOnlyModel, SimpleFusionModel


DIM = 96


@dataclass
class MCATStreams:
    path: torch.Tensor
    omic: torch.Tensor
    clinical: torch.Tensor


class MCATRepresentationM5A(nn.Module):
    """MCAT semantic omics representation with a selectable pathology tokenizer."""

    def __init__(self, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...],
                 tokenizer: str = "adaptive", prototype_centers: np.ndarray | None = None,
                 dim: int = DIM) -> None:
        super().__init__()
        if tokenizer not in {"adaptive", "prototype"}:
            raise ValueError(tokenizer)
        self.tokenizer = tokenizer
        self.output_tokens = 32 if tokenizer == "adaptive" else 16
        # This module ordering is shared by M0-M4.  Only the fixed center
        # buffer and the forward tokenization/mask policy differ.
        self.patch_projection = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, dim), nn.GELU())
        self.path_norm = nn.LayerNorm(dim)
        centers = np.zeros((16, 1536), dtype=np.float32) if prototype_centers is None else np.asarray(prototype_centers, dtype=np.float32)
        if centers.shape != (16, 1536):
            raise ValueError(f"prototype centers must have shape (16,1536), got {centers.shape}")
        self.register_buffer("prototype_centers", torch.from_numpy(centers), persistent=False)
        self.sig_networks = SixSignatureEncoder(signature_widths, dim)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, dim=dim)
        self.coattn = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        path_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        omic_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        self.path_transformer = nn.TransformerEncoder(path_layer, num_layers=2)
        self.omic_transformer = nn.TransformerEncoder(omic_layer, num_layers=2)
        self.path_attention_head = GatedAttentionPool(dim)
        self.omic_attention_head = GatedAttentionPool(dim)
        self.path_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.omic_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))

    def path_tokens(self, pathology: torch.Tensor) -> torch.Tensor:
        values = pathology.float()
        h = self.patch_projection(values)
        if self.tokenizer == "adaptive":
            pooled = F.adaptive_avg_pool1d(h.transpose(1, 2), 32).transpose(1, 2)
            return self.path_norm(pooled)
        centers = self.patch_projection(self.prototype_centers.to(device=h.device, dtype=h.dtype))
        scores = torch.matmul(h, centers.transpose(0, 1)) / math.sqrt(h.shape[-1])
        # Normalize over patches for each prototype, giving a weighted sum of
        # projected patches.  Both weighted sum and the fixed center addition
        # are invariant to a permutation of the patch bag.
        assignment = torch.softmax(scores.transpose(1, 2), dim=-1)
        tokens = torch.matmul(assignment, h)
        return self.path_norm(tokens + centers.unsqueeze(0))

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor,
                clinical: torch.Tensor, signature_groups: torch.Tensor) -> MCATStreams:
        h_path = self.path_tokens(pathology)
        h_omic = self.sig_networks(signature_groups)
        h_clinical = self.clinical_encoder(clinical)
        path_tokens, _ = self.coattn(h_omic, h_path, h_path)
        path_tokens = self.path_transformer(path_tokens)
        omic_tokens = self.omic_transformer(h_omic)
        h_path_repr, _ = self.path_attention_head(path_tokens)
        h_omic_repr, _ = self.omic_attention_head(omic_tokens)
        return MCATStreams(
            path=self.path_rho(h_path_repr),
            omic=self.omic_rho(h_omic_repr),
            clinical=h_clinical[:, 0],
        )


class M5AModel(nn.Module):
    """M0-M4: fixed MCAT semantic omics and original MCAT fusion."""

    def __init__(self, variant: str, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...],
                 prototype_centers: np.ndarray, dim: int = DIM) -> None:
        super().__init__()
        allowed = {"M0", "M1", "M2", "M3", "M4"}
        if variant not in allowed:
            raise ValueError(variant)
        self.variant = variant
        tokenizer = "adaptive" if variant in {"M0", "M2"} else "prototype"
        self.representation = MCATRepresentationM5A(clinical_dim, signature_widths, tokenizer, prototype_centers, dim)
        self.mm = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.clinical_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = RiskHead(dim)

    def _selective_mask(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # Per-sample, per-query boolean mask for dense MHA.  True means blocked.
        q = F.normalize(query.float(), dim=-1, eps=1e-8)
        k = F.normalize(key.float(), dim=-1, eps=1e-8)
        similarity = torch.matmul(q, k.transpose(1, 2))
        keep_count = 8 if key.shape[1] == 32 else 4
        keep = torch.zeros_like(similarity, dtype=torch.bool)
        if self.variant in {"M2", "M3"}:
            selected = torch.topk(similarity, k=keep_count, dim=-1, largest=True).indices
        elif self.variant == "M4":
            selected = torch.topk(similarity, k=keep_count, dim=-1, largest=False).indices
        else:
            raise RuntimeError("selective mask requested for dense variant")
        keep.scatter_(dim=-1, index=selected, value=True)
        blocked = ~keep
        # MultiheadAttention expects [batch*num_heads, target, source].
        return blocked.repeat_interleave(self.representation.coattn.num_heads, dim=0)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor,
                clinical: torch.Tensor, signature_groups: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if signature_groups is None:
            raise ValueError("M5A MCAT requires official six-signature groups")
        # Representation is split here so the selective mask is applied before
        # the unchanged MCAT transformer/pooling/fusion path.
        h_path = self.representation.path_tokens(pathology)
        h_omic = self.representation.sig_networks(signature_groups)
        h_clinical = self.representation.clinical_encoder(clinical)
        if self.variant in {"M0", "M1"}:
            path_tokens, _ = self.representation.coattn(h_omic, h_path, h_path)
        else:
            mask = self._selective_mask(h_omic, h_path)
            path_tokens, _ = self.representation.coattn(h_omic, h_path, h_path, attn_mask=mask)
        path_tokens = self.representation.path_transformer(path_tokens)
        omic_tokens = self.representation.omic_transformer(h_omic)
        h_path_repr, _ = self.representation.path_attention_head(path_tokens)
        h_omic_repr, _ = self.representation.omic_attention_head(omic_tokens)
        streams = MCATStreams(
            path=self.representation.path_rho(h_path_repr),
            omic=self.representation.omic_rho(h_omic_repr),
            clinical=h_clinical[:, 0],
        )
        fused = self.mm(torch.cat([streams.path, streams.omic], dim=-1))
        return self.head(self.clinical_gate(torch.cat([fused, streams.clinical], dim=-1))), pathology.new_zeros(())


def build_model(variant: str, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...],
                prototype_centers: np.ndarray) -> nn.Module:
    if variant in {"M0", "M1", "M2", "M3", "M4"}:
        return M5AModel(variant, clinical_dim, signature_widths, prototype_centers)
    if variant == "B0":
        return SimpleFusionModel(clinical_dim, variant="S3", variant_seed=420001, concat_seed=430001)
    if variant == "C0":
        return ClinicalOnlyModel(clinical_dim, variant_seed=410001)
    raise ValueError(variant)


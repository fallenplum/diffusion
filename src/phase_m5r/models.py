from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.phase_m2.mcat import SixSignatureEncoder
from src.phase_m2.mgct import GatedAttentionPool
from src.phase_m2.nn_blocks import ClinicalEncoder, RiskHead
from src.phase_m3r2.models import ClinicalOnlyModel, SimpleFusionModel


DIM = 96
HEADS = 4
HEAD_DIM = DIM // HEADS


@dataclass
class MCATStreams:
    path: torch.Tensor
    omic: torch.Tensor
    clinical: torch.Tensor


class MCATRepresentationM5R(nn.Module):
    """M5A representation with the original MCAT co-attention module."""

    def __init__(self, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...],
                 tokenizer: str, prototype_centers: np.ndarray | None = None, dim: int = DIM) -> None:
        super().__init__()
        if tokenizer not in {"adaptive", "prototype"}:
            raise ValueError(tokenizer)
        self.tokenizer = tokenizer
        self.patch_projection = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, dim), nn.GELU())
        self.path_norm = nn.LayerNorm(dim)
        centers = np.zeros((16, 1536), dtype=np.float32) if prototype_centers is None else np.asarray(prototype_centers, dtype=np.float32)
        if centers.shape != (16, 1536):
            raise ValueError(f"prototype centers must have shape (16,1536), got {centers.shape}")
        self.register_buffer("prototype_centers", torch.from_numpy(centers), persistent=False)
        self.sig_networks = SixSignatureEncoder(signature_widths, dim)
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
        return self.path_tokens(pathology), self.sig_networks(groups), self.clinical_encoder(clinical)


class M5RModel(nn.Module):
    """R0-R5; only the Q/K-aligned selection policy changes across R variants."""

    def __init__(self, variant: str, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...],
                 prototype_centers: np.ndarray, dim: int = DIM) -> None:
        super().__init__()
        if variant not in {"R0", "R1", "R2", "R3", "R4", "R5"}:
            raise ValueError(variant)
        self.variant = variant
        tokenizer = "adaptive" if variant in {"R0", "R1"} else "prototype"
        self.representation = MCATRepresentationM5R(clinical_dim, signature_widths, tokenizer, prototype_centers, dim)
        self.mm = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.clinical_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = RiskHead(dim)
        self.last_selection: dict[str, Any] = {}

    def _qk_score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        mha = self.representation.coattn
        if mha.in_proj_weight is None:
            raise RuntimeError("M5R requires packed in_proj_weight")
        wq, wk, _ = mha.in_proj_weight.chunk(3, dim=0)
        bq, bk, _ = mha.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, wq, bq).view(query.shape[0], query.shape[1], HEADS, HEAD_DIM).transpose(1, 2)
        k = F.linear(key, wk, bk).view(key.shape[0], key.shape[1], HEADS, HEAD_DIM).transpose(1, 2)
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)

    def _raw_cosine_score(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return torch.matmul(F.normalize(query.float(), dim=-1, eps=1e-8), F.normalize(key.float(), dim=-1, eps=1e-8).transpose(1, 2)).unsqueeze(1).expand(-1, HEADS, -1, -1)

    def _mask(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.variant in {"R0", "R2"}:
            return None, None
        score = self._raw_cosine_score(query, key) if self.variant == "R3" else self._qk_score(query, key)
        keep_n = 8 if key.shape[1] == 32 else 4
        largest = self.variant not in {"R5"}
        selected = torch.topk(score.detach(), k=keep_n, dim=-1, largest=largest).indices.detach()
        keep = torch.zeros_like(score, dtype=torch.bool)
        keep.scatter_(-1, selected, True)
        blocked = ~keep
        return blocked.reshape(query.shape[0] * HEADS, query.shape[1], key.shape[1]), keep

    def _attention_diagnostics(self, query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor | None) -> dict[str, Any]:
        self.representation.coattn.eval()
        with torch.no_grad():
            _, weights = self.representation.coattn(query, key, key, attn_mask=mask, need_weights=True, average_attn_weights=False)
        weights = weights.clamp_min(1e-12)
        entropy = float((-weights * weights.log()).sum(dim=-1).mean().item())
        return {"attention_entropy": entropy, "attention_weight_shape": list(weights.shape)}

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor,
                clinical: torch.Tensor, signature_groups: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del rna, cna
        if signature_groups is None:
            raise ValueError("M5R requires six official MCAT signature groups")
        h_path, h_omic, h_clinical = self.representation.encoded_inputs(pathology, clinical, signature_groups)
        mask, keep = self._mask(h_omic, h_path)
        path_tokens, _ = self.representation.coattn(h_omic, h_path, h_path, attn_mask=mask)
        if keep is not None:
            self.last_selection = {"keep": keep.detach()}
        else:
            self.last_selection = {}
        path_tokens = self.representation.path_transformer(path_tokens)
        omic_tokens = self.representation.omic_transformer(h_omic)
        h_path_repr, _ = self.representation.path_attention_head(path_tokens)
        h_omic_repr, _ = self.representation.omic_attention_head(omic_tokens)
        streams = MCATStreams(self.representation.path_rho(h_path_repr), self.representation.omic_rho(h_omic_repr), h_clinical[:, 0])
        fused = self.mm(torch.cat([streams.path, streams.omic], dim=-1))
        return self.head(self.clinical_gate(torch.cat([fused, streams.clinical], dim=-1))), pathology.new_zeros(())


def control_seed(method: str, seed: int, fold: int) -> int:
    from src.phase_m2.common import stable_int
    return 700000 + stable_int(f"m5r-control:{method}:{seed}:{fold}", modulo=1_900_000_000)


def build_model(variant: str, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...],
                prototype_centers: np.ndarray, seed: int, fold: int) -> nn.Module:
    if variant in {"R0", "R1", "R2", "R3", "R4", "R5"}:
        return M5RModel(variant, clinical_dim, signature_widths, prototype_centers)
    if variant == "B0":
        cs = control_seed(variant, seed, fold)
        return SimpleFusionModel(clinical_dim, variant="S3", variant_seed=cs, concat_seed=cs + 1)
    if variant == "C0":
        return ClinicalOnlyModel(clinical_dim, variant_seed=control_seed(variant, seed, fold))
    raise ValueError(variant)


def qk_attention_audit(seed: int = 20260922) -> dict[str, Any]:
    torch.manual_seed(seed)
    checks = []
    for nk in (16, 32):
        mha = nn.MultiheadAttention(DIM, HEADS, dropout=0.1, batch_first=True).eval()
        q = torch.randn(3, 6, DIM)
        k = torch.randn(3, nk, DIM)
        v = torch.randn(3, nk, DIM)
        with torch.no_grad():
            _, weights = mha(q, k, v, need_weights=True, average_attn_weights=False)
            wq, wk, _ = mha.in_proj_weight.chunk(3, dim=0)
            bq, bk, _ = mha.in_proj_bias.chunk(3, dim=0)
            qq = F.linear(q, wq, bq).view(3, 6, HEADS, HEAD_DIM).transpose(1, 2)
            kk = F.linear(k, wk, bk).view(3, nk, HEADS, HEAD_DIM).transpose(1, 2)
            manual = torch.softmax(torch.matmul(qq, kk.transpose(-2, -1)) / math.sqrt(HEAD_DIM), dim=-1)
        err = float((manual - weights).abs().max().item())
        checks.append({"nk": nk, "batch": 3, "nq": 6, "heads": HEADS, "head_dim": HEAD_DIM, "shape_exact": list(weights.shape) == [3, HEADS, 6, nk], "max_abs_attention_weight_error": err, "finite": bool(torch.isfinite(weights).all() and torch.isfinite(manual).all())})
    passed = all(x["shape_exact"] and x["finite"] and x["max_abs_attention_weight_error"] < 1e-5 for x in checks)
    return {"status": "PASS" if passed else "FAIL", "checks": checks, "split": "in_proj_weight/bias chunks are Wq,Wk,Wv and bq,bk,bv", "score": "QK^T/sqrt(24) before softmax"}


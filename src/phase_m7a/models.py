from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.phase_m2.mcat import MCATModel, SixSignatureEncoder
from src.phase_m2.mgct import GatedAttentionPool
from src.phase_m2.nn_blocks import ClinicalEncoder, RiskHead
from src.phase_m3r2.models import ClinicalOnlyModel, SimpleFusionModel

DIM = 96


@contextmanager
def _isolated_seed(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


class GatedABMIL(nn.Module):
    """Permutation-invariant gated ABMIL over an unordered patch bag."""

    def __init__(self, dim: int = DIM, hidden: int = 48):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, dim), nn.GELU())
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.projection(patches.float())
        scores = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = self.norm(torch.sum(weights.unsqueeze(-1) * h, dim=1))
        return pooled, weights


class FiLMModule(nn.Module):
    def __init__(self, in_dim: int = DIM, hidden: int = 64, out_dim: int = DIM * 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.net(value).chunk(2, dim=-1)
        return gamma, beta


class RawOmicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rna = nn.Sequential(nn.Linear(512, DIM), nn.GELU())
        self.cna = nn.Sequential(nn.Linear(256, DIM), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(DIM * 2, DIM), nn.GELU())

    def forward(self, rna: torch.Tensor, cna: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.rna(rna.float()), self.cna(cna.float())], dim=-1))


@dataclass
class M7AStreams:
    path: torch.Tensor
    omic: torch.Tensor
    clinical: torch.Tensor


class M7ARepresentation(nn.Module):
    """Shared ABMIL, MCAT six-token semantics, transformers, and pools."""

    def __init__(self, clinical_dim: int, widths: tuple[tuple[int, int], ...], seed: int, fold: int, raw_omics: bool = False):
        super().__init__()
        common_seed = 170000000 + (seed % 1000000) * 11 + fold
        with _isolated_seed(common_seed):
            self.abmil = GatedABMIL()
            if raw_omics:
                self.raw_omic = RawOmicEncoder()
                self.six_encoder = None
            else:
                self.six_encoder = SixSignatureEncoder(widths, DIM)
                self.raw_omic = None
            layer = nn.TransformerEncoderLayer(DIM, 4, dim_feedforward=DIM * 2, dropout=0.25, activation="relu", batch_first=True)
            self.omic_transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.omic_attention_head = GatedAttentionPool(DIM)
            self.omic_rho = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(), nn.Dropout(0.25))
            self.clinical_encoder = ClinicalEncoder(clinical_dim, dim=DIM)
            self.fusion = nn.Sequential(nn.Linear(DIM * 2, DIM), nn.ReLU(), nn.Linear(DIM, DIM), nn.ReLU())
            self.clinical_gate = nn.Sequential(nn.Linear(DIM * 2, DIM), nn.GELU(), nn.LayerNorm(DIM))
            self.head = RiskHead(DIM)
        self.raw_omics = raw_omics

    def encode(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor, groups: torch.Tensor) -> tuple[M7AStreams, torch.Tensor, torch.Tensor]:
        path, path_weights = self.abmil(pathology)
        if self.raw_omics:
            assert self.raw_omic is not None
            omic = self.raw_omic(rna, cna).unsqueeze(1)
        else:
            assert self.six_encoder is not None
            omic = self.six_encoder(groups)
            omic = self.omic_transformer(omic)
            omic, _ = self.omic_attention_head(omic)
            omic = self.omic_rho(omic)
        clinical_repr = self.clinical_encoder(clinical)[:, 0]
        return M7AStreams(path, omic, clinical_repr), path_weights, omic


class M7AModel(nn.Module):
    def __init__(self, variant: str, clinical_dim: int, widths: tuple[tuple[int, int], ...], seed: int, fold: int):
        super().__init__()
        if variant not in {"F1", "F2", "F3", "F4", "F5"}:
            raise ValueError(variant)
        self.variant = variant
        self.representation = M7ARepresentation(clinical_dim, widths, seed, fold, raw_omics=variant == "F5")
        interaction_seed = 270000000 + (seed % 1000000) * 13 + fold
        with _isolated_seed(interaction_seed):
            if variant == "F2":
                self.residual = nn.Sequential(nn.Linear(DIM * 2, 64), nn.GELU(), nn.Linear(64, DIM))
            else:
                self.residual = None
            self.film_o = FiLMModule() if variant in {"F3", "F4", "F5"} else None
            self.film_p = FiLMModule() if variant == "F4" else None

    def _condition(self, path: torch.Tensor, omic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        diag: dict[str, torch.Tensor] = {}
        if self.variant == "F1":
            return path, omic, diag
        if self.variant == "F2":
            assert self.residual is not None
            return path + self.residual(torch.cat([path, omic], dim=-1)), omic, diag
        assert self.film_o is not None
        gamma_o, beta_o = self.film_o(omic)
        path_mod = path + torch.tanh(gamma_o) * F.layer_norm(path, (DIM,)) + beta_o
        diag.update({"gamma_o": gamma_o, "beta_o": beta_o, "path_correction_norm": (path_mod - path).norm(dim=-1)})
        if self.variant == "F4":
            assert self.film_p is not None
            gamma_p, beta_p = self.film_p(path)
            omic_mod = omic + torch.tanh(gamma_p) * F.layer_norm(omic, (DIM,)) + beta_p
            diag.update({"gamma_p": gamma_p, "beta_p": beta_p, "omic_correction_norm": (omic_mod - omic).norm(dim=-1)})
            return path_mod, omic_mod, diag
        return path_mod, omic, diag

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor, groups: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if groups is None:
            raise ValueError("M7A F-family requires six-signature groups")
        streams, _, _ = self.representation.encode(pathology, rna, cna, clinical, groups)
        path, omic, _ = self._condition(streams.path, streams.omic)
        fused = self.representation.fusion(torch.cat([path, omic], dim=-1))
        context = self.representation.clinical_gate(torch.cat([fused, streams.clinical], dim=-1))
        return self.representation.head(context), pathology.new_zeros(())

    @torch.no_grad()
    def modulation_diagnostics(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor, groups: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        streams, _, _ = self.representation.encode(pathology, rna, cna, clinical, groups)
        _, _, diag = self._condition(streams.path, streams.omic)
        return diag


def control_seed(method: str, seed: int, fold: int) -> int:
    from src.phase_m2.common import stable_int
    return 800000 + stable_int(f"m7a-control:{method}:{seed}:{fold}", modulo=1_900_000_000)


def build_model(variant: str, clinical_dim: int, widths: tuple[tuple[int, int], ...], seed: int, fold: int) -> nn.Module:
    if variant == "F0":
        return MCATModel(clinical_dim, signature_widths=widths, variant="A0")
    if variant in {"F1", "F2", "F3", "F4", "F5"}:
        return M7AModel(variant, clinical_dim, widths, seed, fold)
    if variant == "B0":
        cs = control_seed(variant, seed, fold)
        return SimpleFusionModel(clinical_dim, variant="S3", variant_seed=cs, concat_seed=cs + 1)
    if variant == "C0":
        return ClinicalOnlyModel(clinical_dim, variant_seed=control_seed(variant, seed, fold))
    raise ValueError(variant)


def apply_film_identity(model: M7AModel, path: torch.Tensor, omic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    return model._condition(path, omic)

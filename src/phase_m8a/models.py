from __future__ import annotations
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from src.phase_m3r2.models import SimpleFusionModel, ClinicalOnlyModel
from src.phase_m7r.models import M7RModel, build_model as build_m7r
from src.phase_m2.common import stable_int

DIM = 96

@contextmanager
def isolated_seed(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield

class CommonOnlyModel(SimpleFusionModel):
    """M3R2 S3 with an isolated interaction module for C1/C2."""
    def __init__(self, clinical_dim: int, variant: str, variant_seed: int, concat_seed: int, interaction_seed: int):
        super().__init__(clinical_dim, variant="S3", variant_seed=variant_seed, concat_seed=concat_seed)
        self.m8a_variant = variant
        if variant == "C1":
            with isolated_seed(interaction_seed):
                self.film = nn.Sequential(nn.Linear(192, 32), nn.GELU(), nn.Linear(32, 192))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        elif variant == "C2":
            with isolated_seed(interaction_seed):
                self.residual = nn.Sequential(nn.Linear(288, 32), nn.GELU(), nn.Linear(32, 96))
        elif variant != "C0":
            raise ValueError(variant)

    def forward(self, pathology, rna, cna, clinical):
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        base = {"gene": encoded.omics[:, :8].mean(dim=1), "path": encoded.pathology.mean(dim=1), "cna": encoded.omics[:, 8:].mean(dim=1)}
        common = {name: self.common[name](value) for name, value in base.items()}
        if self.m8a_variant == "C1":
            gamma, beta = self.film(torch.cat([common["gene"], common["cna"]], dim=-1)).chunk(2, dim=-1)
            path = common["path"] + torch.tanh(gamma) * F.layer_norm(common["path"], (DIM,)) + beta
        elif self.m8a_variant == "C2":
            path = common["path"] + self.residual(torch.cat([common["path"], common["gene"], common["cna"]], dim=-1))
        else:
            path = common["path"]
        fused = F.relu(self.concat_fusion(torch.cat([common["gene"], path, common["cna"]], dim=-1)))
        return self.head(torch.cat([fused, encoded.clinical[:, 0]], dim=-1)), fused.new_zeros(())

    def film_condition(self, p, g, n):
        gamma, beta = self.film(torch.cat([g, n], dim=-1)).chunk(2, dim=-1)
        return p + torch.tanh(gamma) * F.layer_norm(p, (DIM,)) + beta, gamma, beta

def common_seeds(seed: int, fold: int):
    return (800000 + stable_int(f"m8a-common:{seed}:{fold}", modulo=1900000000),
            900000 + stable_int(f"m8a-concat:{seed}:{fold}", modulo=1900000000),
            1000000 + stable_int(f"m8a-interaction:{seed}:{fold}", modulo=1900000000))

def build_model(variant: str, clinical_dim: int, widths, seed: int, fold: int):
    if variant in {"C0", "C1", "C2"}:
        a, b, c = common_seeds(seed, fold)
        return CommonOnlyModel(clinical_dim, variant, a, b, c)
    if variant == "A0": return build_m7r("F1", clinical_dim, widths, seed, fold)
    if variant == "A1": return build_m7r("F3", clinical_dim, widths, seed, fold)
    if variant == "A2": return build_m7r("F2", clinical_dim, widths, seed, fold)
    if variant == "M0": return build_m7r("F0", clinical_dim, widths, seed, fold)
    if variant == "K0": return ClinicalOnlyModel(clinical_dim, variant_seed=500000 + stable_int(f"m8a-clinical:{seed}:{fold}"))
    raise ValueError(variant)

def interaction_parameters(variant: str) -> int:
    return {"C1": sum(p.numel() for p in [torch.empty(32,192), torch.empty(32), torch.empty(192,32), torch.empty(192)]), "C2": sum(p.numel() for p in [torch.empty(32,288), torch.empty(32), torch.empty(96,32), torch.empty(96)])}.get(variant, 0)

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.phase_m2.camr import gradient_reverse
from src.phase_m2.nn_blocks import MultimodalBackbone, RiskHead, mean_pool


DIM = 96


@contextmanager
def _isolated_seed(seed: int | None):
    """Isolate variant-only initialization from shared upstream RNG state."""
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


class ClinicalOnlyModel(nn.Module):
    """Clinical control using exactly the protocol clinical vector and Cox head."""

    def __init__(self, clinical_dim: int, variant_seed: int = 410001) -> None:
        super().__init__()
        with _isolated_seed(variant_seed):
            # Reuse the protocol clinical encoder, with no pathology or omics path.
            self.encoder = nn.Sequential(nn.LayerNorm(clinical_dim), nn.Linear(clinical_dim, DIM), nn.GELU())
            self.head = RiskHead(DIM, hidden=DIM)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del pathology, rna, cna
        value = self.encoder(clinical.float())
        return self.head(value), value.new_zeros(())


class SimpleFusionModel(nn.Module):
    """M3R2 simple-fusion family with a shared multimodal upstream backbone."""

    def __init__(self, clinical_dim: int, variant: str, variant_seed: int = 420001, concat_seed: int = 430001, dim: int = DIM) -> None:
        super().__init__()
        allowed = {"S1", "S2", "S3", "S4", "S5"}
        if variant not in allowed:
            raise ValueError(f"unknown M3R2 variant: {variant}")
        self.variant = variant
        self.dim = dim
        # This is deliberately the first module for S1-S4 so its byte-level
        # initialization is identical at a fixed seed/fold.
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)

        # All modules after the backbone are variant-only.  The common branch
        # uses the same isolated seed in S1-S3; S4 has no decomposition branch.
        with _isolated_seed(variant_seed):
            if variant in {"S1", "S2", "S3", "S5"}:
                self.common = nn.ModuleDict({
                    name: nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim), nn.ReLU())
                    for name in ("gene", "path", "cna")
                })
            if variant in {"S1", "S2", "S5"}:
                self.specific = nn.ModuleDict({
                    name: nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1))
                    for name in ("gene", "path", "cna")
                })
                self.reconstruct = nn.ModuleDict({
                    name: nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
                    for name in ("gene", "path", "cna")
                })
                self.discriminator = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3))

        # Fusion and survival head have their own deterministic variant seed.
        with _isolated_seed(concat_seed):
            self.concat_fusion = nn.Linear(dim * 3, dim)
            self.head = RiskHead(dim if variant == "S5" else dim * 2, hidden=dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        base = {
            "gene": mean_pool(encoded.omics[:, :8]),
            "path": mean_pool(encoded.pathology),
            "cna": mean_pool(encoded.omics[:, 8:]),
        }
        if self.variant == "S4":
            fused = F.relu(self.concat_fusion(torch.cat([base["gene"], base["path"], base["cna"]], dim=-1)))
            risk = self.head(torch.cat([fused, encoded.clinical[:, 0]], dim=-1))
            return risk, fused.new_zeros(())

        common = {name: self.common[name](value) for name, value in base.items()}
        fused = F.relu(self.concat_fusion(torch.cat([common["gene"], common["path"], common["cna"]], dim=-1)))
        if self.variant == "S3":
            risk = self.head(torch.cat([fused, encoded.clinical[:, 0]], dim=-1))
            return risk, fused.new_zeros(())

        specific = {name: self.specific[name](value) for name, value in base.items()}
        reconstruction = sum(
            (self.reconstruct[name](torch.cat([common[name], specific[name]], dim=-1)) - base[name]).square().mean()
            for name in base
        ) / 3.0
        orthogonality = sum(
            (common[name] * specific[name]).sum(dim=-1).square().mean() for name in base
        ) / 3.0
        domain_input = gradient_reverse(torch.cat([common[name] for name in ("gene", "path", "cna")], dim=0))
        domain_labels = torch.arange(3, device=domain_input.device).repeat_interleave(common["gene"].shape[0])
        adversarial = F.cross_entropy(self.discriminator(domain_input), domain_labels)
        clinical_context = encoded.clinical[:, 0]
        if self.variant == "S5":
            risk = self.head(fused)
        else:
            risk = self.head(torch.cat([fused, clinical_context], dim=-1))
        adv_weight = 0.0 if self.variant == "S2" else 0.05
        aux = 0.10 * reconstruction + 0.01 * orthogonality + adv_weight * adversarial
        return risk, aux


def build_model(variant: str, clinical_dim: int, variant_seed: int, concat_seed: int) -> nn.Module:
    if variant == "S0":
        return ClinicalOnlyModel(clinical_dim, variant_seed=variant_seed)
    return SimpleFusionModel(clinical_dim, variant=variant, variant_seed=variant_seed, concat_seed=concat_seed)

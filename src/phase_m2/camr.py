from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from .nn_blocks import MultimodalBackbone, RiskHead, mean_pool


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(value, coefficient)


class LowRankMultimodalFusion(nn.Module):
    """Compact LMF block corresponding to CAMR's common fusion module."""

    def __init__(self, dim: int, rank: int = 16) -> None:
        super().__init__()
        self.rank = rank
        self.audio_factor = nn.Parameter(torch.randn(rank, dim + 1, dim) * 0.02)
        self.video_factor = nn.Parameter(torch.randn(rank, dim + 1, dim) * 0.02)
        self.text_factor = nn.Parameter(torch.randn(rank, dim + 1, dim) * 0.02)
        self.fusion_weights = nn.Parameter(torch.randn(1, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1, dim))

    def forward(self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
        ones = first.new_ones((first.shape[0], 1))
        first = torch.cat([ones, first], dim=-1)
        second = torch.cat([ones, second], dim=-1)
        third = torch.cat([ones, third], dim=-1)
        a = torch.matmul(first, self.audio_factor)
        b = torch.matmul(second, self.video_factor)
        c = torch.matmul(third, self.text_factor)
        product = a * b * c
        return torch.matmul(self.fusion_weights, product.permute(1, 0, 2)).squeeze(1) + self.bias


class CAMRModel(nn.Module):
    """CAMR's invariant/specific encoders, CMFM, reconstruction and adversary.

    The official CAMR repository uses three modalities. Here they are mapped to
    RNA (gene), UNI2-h pathology and CNA (copy-number); the protocol clinical
    vector is integrated by the survival head after CMFM.
    """

    def __init__(self, clinical_dim: int, dim: int = 96, variant: str = "B0", concat_seed: int | None = None) -> None:
        super().__init__()
        allowed = {"B0", "B1", "B2", "B3", "D0", "D1"}
        if variant not in allowed:
            raise ValueError(f"unknown CAMR ablation variant: {variant}")
        self.variant = variant
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.common = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim), nn.ReLU()) for name in ("gene", "path", "cna")})
        self.specific = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1)) for name in ("gene", "path", "cna")})
        self.reconstruct = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)) for name in ("gene", "path", "cna")})
        self.lmf = LowRankMultimodalFusion(dim)
        self.cmfm = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.cmfm_norm = nn.LayerNorm(dim)
        # The confirmatory D0/D1 pair is constructed so that D0 never
        # instantiates the variant-only concat module.  D1 initializes that
        # module under an isolated deterministic RNG after all shared modules
        # have been created, preserving byte-identical shared parameters.
        if variant in {"B3", "D1"}:
            if variant == "D1" and concat_seed is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(concat_seed))
                    self.concat_fusion = nn.Linear(dim * 3, dim)
            else:
                self.concat_fusion = nn.Linear(dim * 3, dim)
        self.discriminator = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3))
        self.head = RiskHead(dim * 2, hidden=dim)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        base = {
            "gene": mean_pool(encoded.omics[:, :8]),
            "path": mean_pool(encoded.pathology),
            "cna": mean_pool(encoded.omics[:, 8:]),
        }
        common = {name: self.common[name](value) for name, value in base.items()}
        specific = {name: self.specific[name](value) for name, value in base.items()}
        if self.variant == "B2":
            # Remove the modality-specific branch from the computation graph.
            specific = {name: torch.zeros_like(value) for name, value in base.items()}

        # CMFM retains CAMR's low-rank multiplicative common fusion and adds
        # cross-modal attention over the common/specific token set.
        if self.variant in {"B3", "D1"}:
            # Remove CMFM and replace it with a common-representation concat.
            cmfm = torch.cat([common["gene"], common["path"], common["cna"]], dim=-1)
            cmfm = nn.functional.relu(self.concat_fusion(cmfm))
        else:
            lmf = self.lmf(common["gene"], common["path"], common["cna"])
            tokens = torch.stack([common["gene"], common["path"], common["cna"]], dim=1)
            attended, _ = self.cmfm(tokens, tokens, tokens)
            cmfm = self.cmfm_norm(attended + tokens).mean(dim=1) + lmf

        if self.variant == "B2":
            reconstruction = cmfm.new_zeros(())
            orthogonality = cmfm.new_zeros(())
        else:
            reconstruction = sum((self.reconstruct[name](torch.cat([common[name], specific[name]], dim=-1)) - base[name]).square().mean() for name in base) / 3.0
            orthogonality = sum((common[name] * specific[name]).sum(dim=-1).square().mean() for name in base) / 3.0
        domain_input = gradient_reverse(torch.cat([common[name] for name in ("gene", "path", "cna")], dim=0))
        domain_labels = torch.arange(3, device=domain_input.device).repeat_interleave(common["gene"].shape[0])
        adversarial = nn.functional.cross_entropy(self.discriminator(domain_input), domain_labels)
        clinical_context = encoded.clinical[:, 0]
        risk = self.head(torch.cat([cmfm, clinical_context], dim=-1))
        aux = 0.10 * reconstruction + 0.01 * orthogonality + (0.0 if self.variant == "B1" else 0.05) * adversarial
        return risk, aux


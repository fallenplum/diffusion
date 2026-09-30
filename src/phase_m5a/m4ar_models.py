from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from src.phase_m2.mcat import SixSignatureEncoder
from src.phase_m2.mgct import GatedAttentionPool
from src.phase_m2.nn_blocks import MultimodalBackbone, RiskHead, mean_pool


DIM = 96
PROJECTED_DIM = 32
KAN_REQUESTED_HIDDEN = 48
# Parameter matching is frozen before formal training.  With grid_size=5 and
# order=3, hidden=48 exceeds the matched MLP head by more than 20%; hidden=8
# is the largest tested width within the 0.8x--1.2x window for both families.
KAN_HIDDEN = 8
KAN_WM_HIDDEN = 48
KAN_GRID_SIZE = 5
KAN_ORDER = 3
KAN_RANGE = (-3.0, 3.0)
LOWRANK_RANK = 8


@contextmanager
def isolated_seed(seed: int | None):
    if seed is None:
        yield
        return
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        yield


@dataclass
class MCATStreams:
    path: torch.Tensor
    omic: torch.Tensor
    clinical: torch.Tensor


@dataclass
class CommonStreams:
    path: torch.Tensor
    rna: torch.Tensor
    cna: torch.Tensor
    clinical: torch.Tensor


class MCATRepresentation(nn.Module):
    """M2R2/M3 MCAT representation, ending before the final fusion head."""

    def __init__(self, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...], dim: int = DIM) -> None:
        super().__init__()
        # Keep module creation order identical across A0-A3.
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.sig_networks = SixSignatureEncoder(signature_widths, dim)
        self.coattn = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        path_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        omic_layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=dim * 2, dropout=0.25, activation="relu", batch_first=True)
        self.path_transformer = nn.TransformerEncoder(path_layer, num_layers=2)
        self.omic_transformer = nn.TransformerEncoder(omic_layer, num_layers=2)
        self.path_attention_head = GatedAttentionPool(dim)
        self.omic_attention_head = GatedAttentionPool(dim)
        self.path_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))
        self.omic_rho = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.25))

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor, signature_groups: torch.Tensor) -> MCATStreams:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        h_path = encoded.pathology
        h_omic = self.sig_networks(signature_groups)
        # Fixed official direction: omic queries attend to pathology K/V.
        path_tokens, _ = self.coattn(h_omic, h_path, h_path)
        omic_tokens = h_omic
        path_tokens = self.path_transformer(path_tokens)
        omic_tokens = self.omic_transformer(omic_tokens)
        h_path, _ = self.path_attention_head(path_tokens)
        h_omic, _ = self.omic_attention_head(omic_tokens)
        return MCATStreams(
            path=self.path_rho(h_path),
            omic=self.omic_rho(h_omic),
            clinical=encoded.clinical[:, 0],
        )


class CommonOnlyRepresentation(nn.Module):
    """M3R2 S3 representation before its fusion head."""

    def __init__(self, clinical_dim: int, dim: int = DIM) -> None:
        super().__init__()
        # Keep module creation order identical across B0-B3.
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        self.common = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim), nn.ReLU())
            for name in ("gene", "path", "cna")
        })

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> CommonStreams:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        base = {
            "gene": mean_pool(encoded.omics[:, :8]),
            "path": mean_pool(encoded.pathology),
            "cna": mean_pool(encoded.omics[:, 8:]),
        }
        return CommonStreams(
            path=self.common["path"](base["path"]),
            rna=self.common["gene"](base["gene"]),
            cna=self.common["cna"](base["cna"]),
            clinical=encoded.clinical[:, 0],
        )


class StreamProjector(nn.Module):
    def __init__(self, dim: int = DIM, output_dim: int = PROJECTED_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim), nn.GELU())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class CompactSplineKAN(nn.Module):
    """Fixed-grid compact B-spline KAN layer.

    The knot grid never adapts and inputs are clamped to [-3, 3].  The basis
    is evaluated with the Cox--de Boor recursion, matching the explicit
    base-plus-spline formula in the M4A protocol.
    """

    def __init__(self, input_dim: int, output_dim: int, grid_size: int = KAN_GRID_SIZE, spline_order: int = KAN_ORDER) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.register_buffer("knots", self._make_clamped_knots(), persistent=True)
        n_basis = self.grid_size + self.spline_order
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.spline_coeff = nn.Parameter(torch.empty(output_dim, input_dim, n_basis))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.register_buffer("clip_total", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("clip_low", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("clip_high", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("clip_abs", torch.zeros((), dtype=torch.float64), persistent=False)
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_coeff, mean=0.0, std=0.01)

    def _make_clamped_knots(self) -> torch.Tensor:
        n_basis = self.grid_size + self.spline_order
        interior_count = n_basis - self.spline_order - 1
        interior = torch.linspace(KAN_RANGE[0], KAN_RANGE[1], interior_count + 2)[1:-1] if interior_count > 0 else torch.empty(0)
        left = torch.full((self.spline_order + 1,), KAN_RANGE[0])
        right = torch.full((self.spline_order + 1,), KAN_RANGE[1])
        return torch.cat([left, interior, right])

    def _basis(self, value: torch.Tensor) -> torch.Tensor:
        x = value.clamp(KAN_RANGE[0], KAN_RANGE[1])
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        # Degree-zero basis has one entry per knot interval.
        basis = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).to(x.dtype)
        for degree in range(1, self.spline_order + 1):
            terms = []
            for index in range(knots.numel() - degree - 1):
                left_den = knots[index + degree] - knots[index]
                right_den = knots[index + degree + 1] - knots[index + 1]
                left = ((x - knots[index]) / left_den * basis[..., index]) if left_den > 0 else torch.zeros_like(basis[..., index])
                right = ((knots[index + degree + 1] - x) / right_den * basis[..., index + 1]) if right_den > 0 else torch.zeros_like(basis[..., index])
                terms.append(left + right)
            basis = torch.stack(terms, dim=-1)
        # At the right clamped boundary only the final degree-p basis is 1.
        # Setting every previous degree-zero interval to 1 breaks partition of
        # unity after recursion, which was the M4A implementation bug.
        at_right = (x == KAN_RANGE[1]).unsqueeze(-1)
        last = torch.zeros_like(basis)
        last[..., -1] = 1.0
        basis = torch.where(at_right, last, basis)
        return basis

    def reset_clip_stats(self) -> None:
        for name in ("clip_total", "clip_low", "clip_high", "clip_abs"):
            getattr(self, name).zero_()

    def clip_stats(self) -> dict[str, float]:
        total = float(self.clip_total.item())
        return {
            "n_values": total,
            "fraction_low": float(self.clip_low.item() / total) if total else 0.0,
            "fraction_high": float(self.clip_high.item() / total) if total else 0.0,
            "fraction_abs_ge3": float(self.clip_abs.item() / total) if total else 0.0,
        }

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(value, (self.input_dim,))
        with torch.no_grad():
            self.clip_total += normalized.numel()
            self.clip_low += (normalized < KAN_RANGE[0]).sum()
            self.clip_high += (normalized > KAN_RANGE[1]).sum()
            self.clip_abs += (normalized.abs() >= KAN_RANGE[1]).sum()
        base = F.silu(normalized) @ self.base_weight.t()
        basis = self._basis(normalized)
        spline = torch.einsum("bik,oik->bo", basis, self.spline_coeff)
        return base + spline + self.bias


class CompactKANFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = KAN_HIDDEN) -> None:
        super().__init__()
        self.kan1 = CompactSplineKAN(input_dim, hidden_dim)
        self.kan2 = CompactSplineKAN(hidden_dim, PROJECTED_DIM)
        self.output_norm = nn.LayerNorm(PROJECTED_DIM)
        self.hidden_dim = hidden_dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.silu(self.kan1(value))
        value = F.silu(self.kan2(value))
        return self.output_norm(value)


class MatchedMLPFusion(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, PROJECTED_DIM),
            nn.GELU(),
            nn.LayerNorm(PROJECTED_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ReferenceLMF(nn.Module):
    """Reference low-rank multimodal fusion with standard Xavier initialization."""
    def __init__(self, n_streams: int, rank: int = LOWRANK_RANK, dim: int = PROJECTED_DIM) -> None:
        super().__init__()
        self.n_streams = n_streams
        self.rank = rank
        self.dim = dim
        self.factors = nn.ParameterList([nn.Parameter(torch.empty(rank, dim + 1, dim)) for _ in range(n_streams)])
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, streams: Iterable[torch.Tensor]) -> torch.Tensor:
        values = list(streams)
        factors = []
        for value, factor in zip(values, self.factors):
            augmented = torch.cat([value.new_ones(value.shape[0], 1), value], dim=-1)
            factors.append(torch.einsum("bi,rij->brj", augmented, factor))
        product = factors[0]
        for factor_value in factors[1:]:
            product = product * factor_value
        return torch.einsum("qr,brd->bqd", self.fusion_weights, product).squeeze(1) + self.bias


class MCATOriginalFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mm = nn.Sequential(nn.Linear(DIM * 2, DIM), nn.ReLU(), nn.Linear(DIM, DIM), nn.ReLU())
        self.clinical_gate = nn.Sequential(nn.Linear(DIM * 2, DIM), nn.GELU(), nn.LayerNorm(DIM))
        self.head = RiskHead(DIM)

    def forward(self, streams: MCATStreams) -> torch.Tensor:
        fused = self.mm(torch.cat([streams.path, streams.omic], dim=-1))
        return self.head(self.clinical_gate(torch.cat([fused, streams.clinical], dim=-1)))


class CommonOriginalFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.concat_fusion = nn.Linear(DIM * 3, DIM)
        self.head = RiskHead(DIM * 2)

    def forward(self, streams: CommonStreams) -> torch.Tensor:
        fused = F.relu(self.concat_fusion(torch.cat([streams.rna, streams.path, streams.cna], dim=-1)))
        return self.head(torch.cat([fused, streams.clinical], dim=-1))


class FusionProbeModel(nn.Module):
    def __init__(self, variant: str, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...], projector_seed: int, variant_seed: int, risk_head_seed: int) -> None:
        super().__init__()
        if variant not in {"A0", "A1", "A2", "A2W", "A3", "B0", "B1", "B2", "B2W", "B3"}:
            raise ValueError(f"unknown fusion probe variant: {variant}")
        self.variant = variant
        self.family = variant[0]
        if self.family == "A":
            self.representation = MCATRepresentation(clinical_dim, signature_widths)
            if variant == "A0":
                with isolated_seed(variant_seed):
                    self.fusion = MCATOriginalFusion()
            else:
                with isolated_seed(projector_seed):
                    self.projectors = nn.ModuleList([StreamProjector() for _ in range(3)])
                with isolated_seed(variant_seed):
                    self.fusion = self._make_projected_fusion(variant, 3)
        else:
            self.representation = CommonOnlyRepresentation(clinical_dim)
            if variant == "B0":
                with isolated_seed(variant_seed):
                    self.fusion = CommonOriginalFusion()
            else:
                with isolated_seed(projector_seed):
                    self.projectors = nn.ModuleList([StreamProjector() for _ in range(4)])
                with isolated_seed(variant_seed):
                    self.fusion = self._make_projected_fusion(variant, 4)
        if variant in {"A0", "B0"}:
            self.risk_head = None
        else:
            # Same family/seed/fold risk-head initialization for all projected
            # heads whose input shape is 32, independent of KAN/LMF choice.
            with isolated_seed(risk_head_seed):
                self.risk_head = RiskHead(PROJECTED_DIM)

    def _make_projected_fusion(self, variant: str, n_streams: int) -> nn.Module:
        input_dim = PROJECTED_DIM * n_streams
        if variant in {"A1", "B1"}:
            return MatchedMLPFusion(input_dim)
        if variant in {"A2", "B2"}:
            return CompactKANFusion(input_dim, hidden_dim=KAN_HIDDEN)
        if variant in {"A2W", "B2W"}:
            return CompactKANFusion(input_dim, hidden_dim=KAN_WM_HIDDEN)
        return ReferenceLMF(n_streams)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor, signature_groups: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.family == "A":
            if signature_groups is None:
                raise ValueError("MCAT family requires six signature groups")
            streams = self.representation(pathology, rna, cna, clinical, signature_groups)
            if self.variant == "A0":
                risk = self.fusion(streams)
            else:
                projected = [projector(value) for projector, value in zip(self.projectors, (streams.path, streams.omic, streams.clinical))]
                fused = self.fusion(projected) if self.variant == "A3" else self.fusion(torch.cat(projected, dim=-1))
                if self.variant == "A3":
                    fused = F.gelu(fused)
                risk = self.risk_head(fused)
        else:
            streams = self.representation(pathology, rna, cna, clinical)
            if self.variant == "B0":
                risk = self.fusion(streams)
            else:
                projected = [projector(value) for projector, value in zip(self.projectors, (streams.path, streams.rna, streams.cna, streams.clinical))]
                fused = self.fusion(projected) if self.variant == "B3" else self.fusion(torch.cat(projected, dim=-1))
                if self.variant == "B3":
                    fused = F.gelu(fused)
                risk = self.risk_head(fused)
        return risk, pathology.new_zeros(())

    def parameter_breakdown(self) -> dict[str, int]:
        representation = sum(parameter.numel() for parameter in self.representation.parameters() if parameter.requires_grad)
        projector = sum(parameter.numel() for parameter in getattr(self, "projectors", nn.ModuleList()).parameters() if parameter.requires_grad)
        fusion = sum(parameter.numel() for parameter in self.fusion.parameters() if parameter.requires_grad)
        head = sum(parameter.numel() for parameter in self.risk_head.parameters() if parameter.requires_grad) if self.risk_head is not None else 0
        return {"representation_params": int(representation), "projector_params": int(projector), "fusion_head_params": int(fusion + head), "total_params": int(representation + projector + fusion + head)}

    def kan_layers(self) -> list[CompactSplineKAN]:
        return [module for module in self.modules() if isinstance(module, CompactSplineKAN)]

    def reset_kan_clip_stats(self) -> None:
        for layer in self.kan_layers():
            layer.reset_clip_stats()

    def kan_clip_stats(self) -> list[dict[str, float]]:
        return [layer.clip_stats() for layer in self.kan_layers()]


def build_model(variant: str, clinical_dim: int, signature_widths: tuple[tuple[int, int], ...], projector_seed: int, variant_seed: int, risk_head_seed: int) -> FusionProbeModel:
    return FusionProbeModel(variant, clinical_dim, signature_widths, projector_seed, variant_seed, risk_head_seed)


class ClinicalOnlyModel(nn.Module):
    def __init__(self, clinical_dim: int, seed: int) -> None:
        super().__init__()
        with isolated_seed(seed):
            self.encoder = nn.Sequential(nn.LayerNorm(clinical_dim), nn.Linear(clinical_dim, DIM), nn.GELU())
            self.head = RiskHead(DIM)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor, signature_groups: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del pathology, rna, cna, signature_groups
        return self.head(self.encoder(clinical.float())), clinical.new_zeros(())


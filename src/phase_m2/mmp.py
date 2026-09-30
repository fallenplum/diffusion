from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch import nn

from .mgct import GatedAttentionPool
from .nn_blocks import MultimodalBackbone, RiskHead


def _gmm_prototypes(values: np.ndarray, components: int, seed: int, max_samples: int = 20000) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim > 2:
        values = values.reshape(-1, values.shape[-1])
    if len(values) == 0:
        return np.zeros((components, values.shape[-1] if values.ndim == 2 else 1), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(values) > max_samples:
        selected = rng.choice(len(values), size=max_samples, replace=False)
        values = values[np.sort(selected)]
    n_components = min(32, values.shape[0], values.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    latent = pca.fit_transform(values)
    n_gmm = min(components, max(2, latent.shape[0]))
    gmm = GaussianMixture(n_components=n_gmm, covariance_type="diag", reg_covar=1e-3, max_iter=40, n_init=1, random_state=seed)
    gmm.fit(latent)
    prototypes = pca.inverse_transform(gmm.means_).astype(np.float32)
    if len(prototypes) < components:
        prototypes = np.concatenate([prototypes, np.repeat(prototypes[-1:], components - len(prototypes), axis=0)], axis=0)
    return prototypes[:components]


def fit_morphology_prototypes(train_path_pool: np.ndarray, seed: int, components: int = 16, sampled_stride: int = 128) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit Ch=16 morphology prototypes from outer-training patches only."""
    values = np.asarray(train_path_pool[:, ::sampled_stride, :], dtype=np.float32)
    prototypes = _gmm_prototypes(values, components, seed)
    return prototypes, {"method": "GaussianMixture", "components": components, "source": "outer_train_pathology_pool", "sampled_stride": sampled_stride}


class PrototypeSpecificFFN(nn.Module):
    def __init__(self, dim: int, count: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)) for _ in range(count)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([block(values[:, i]) for i, block in enumerate(self.blocks)], dim=1)


class MMPModel(nn.Module):
    """MMPTrans adaptation with morphology prototypes and Hallmark tokens."""

    def __init__(
        self,
        clinical_dim: int,
        morphology_prototypes: np.ndarray | None = None,
        hallmark_widths: tuple[int, ...] | None = None,
        dim: int = 96,
        prototype_count: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        morphology = np.zeros((prototype_count, 1536), dtype=np.float32) if morphology_prototypes is None else np.asarray(morphology_prototypes, dtype=np.float32)
        self.register_buffer("morphology_prototypes", torch.from_numpy(morphology), persistent=False)
        self.morphology_projection = nn.Linear(1536, dim)
        self.morphology_ffn = PrototypeSpecificFFN(dim, prototype_count)
        if hallmark_widths is None:
            hallmark_widths = tuple(1 for _ in range(50))
        self.hallmark_widths = tuple(max(1, int(width)) for width in hallmark_widths)
        self.hallmark_encoders = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim), nn.GELU()) for width in self.hallmark_widths]
        )
        self.mmp_cross = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.mmp_reverse = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.path_pool = GatedAttentionPool(dim)
        self.omic_pool = GatedAttentionPool(dim)
        self.fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.clinical_query = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.head = RiskHead(dim)

    def _morphology_tokens(self, pathology: torch.Tensor) -> torch.Tensor:
        values = pathology.float()
        prototypes = self.morphology_prototypes.to(device=values.device, dtype=values.dtype)
        path_proto = self.morphology_projection(prototypes)
        values_proj = self.morphology_projection(values)
        scores = torch.matmul(values_proj, path_proto.transpose(0, 1)) / (values_proj.shape[-1] ** 0.5)
        assignment = torch.softmax(scores, dim=1)
        return self.morphology_ffn(torch.matmul(assignment.transpose(1, 2), values_proj))

    def _hallmark_tokens(self, groups: torch.Tensor) -> torch.Tensor:
        values = []
        for index, (width, encoder) in enumerate(zip(self.hallmark_widths, self.hallmark_encoders)):
            values.append(encoder(groups[:, index, :width].float()).unsqueeze(1))
        return torch.cat(values, dim=1)

    def forward(
        self,
        pathology: torch.Tensor,
        rna: torch.Tensor,
        cna: torch.Tensor,
        clinical: torch.Tensor,
        hallmark_groups: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        morph = self._morphology_tokens(pathology)
        if hallmark_groups is None:
            raise ValueError("MMPTrans requires the 50 official Hallmark pathway groups")
        trans = self._hallmark_tokens(hallmark_groups)
        morph_update, _ = self.mmp_cross(morph, trans, trans)
        trans_update, _ = self.mmp_reverse(trans, morph, morph)
        morph = morph + morph_update
        trans = trans + trans_update
        morph_repr, _ = self.path_pool(morph)
        trans_repr, _ = self.omic_pool(trans)
        fused = self.fusion(torch.cat([morph_repr, trans_repr], dim=-1)).unsqueeze(1)
        clinical_context, _ = self.clinical_query(encoded.clinical, torch.cat([fused, morph, trans], dim=1), torch.cat([fused, morph, trans], dim=1))
        return self.head(clinical_context[:, 0]), pathology.new_zeros(())


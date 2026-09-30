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


def fit_transcriptomic_prototypes(train_rna: np.ndarray, train_cna: np.ndarray, seed: int, components: int = 16) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit transcriptomic prototype bank from outer-training RNA+CNA values."""
    values = np.concatenate([np.asarray(train_rna, dtype=np.float32), np.asarray(train_cna, dtype=np.float32)], axis=1)
    prototypes = _gmm_prototypes(values, components, seed)
    return prototypes, {"method": "GaussianMixture", "components": components, "source": "outer_train_RNA_CNA", "pathway_mapping": "six fixed RNA/CNA signature groups used by the adapted protocol"}


class PrototypeSpecificFFN(nn.Module):
    def __init__(self, dim: int, count: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)) for _ in range(count)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([block(values[:, i]) for i, block in enumerate(self.blocks)], dim=1)


class MMPModel(nn.Module):
    """MMPTrans adaptation with train-only GMM morphology and omic banks."""

    def __init__(
        self,
        clinical_dim: int,
        morphology_prototypes: np.ndarray | None = None,
        transcriptomic_prototypes: np.ndarray | None = None,
        dim: int = 96,
        prototype_count: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = MultimodalBackbone(clinical_dim, dim=dim, path_tokens=32)
        morphology = np.zeros((prototype_count, 1536), dtype=np.float32) if morphology_prototypes is None else np.asarray(morphology_prototypes, dtype=np.float32)
        transcriptomic = np.zeros((prototype_count, 768), dtype=np.float32) if transcriptomic_prototypes is None else np.asarray(transcriptomic_prototypes, dtype=np.float32)
        self.register_buffer("morphology_prototypes", torch.from_numpy(morphology), persistent=False)
        self.register_buffer("transcriptomic_prototypes", torch.from_numpy(transcriptomic), persistent=False)
        self.morphology_projection = nn.Linear(1536, dim)
        self.transcriptomic_projection = nn.Linear(768, dim)
        self.morphology_ffn = PrototypeSpecificFFN(dim, prototype_count)
        self.transcriptomic_ffn = PrototypeSpecificFFN(dim, prototype_count)
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

    def _transcriptomic_tokens(self, rna: torch.Tensor, cna: torch.Tensor) -> torch.Tensor:
        values = torch.cat([rna.float(), cna.float()], dim=1)
        prototypes = self.transcriptomic_prototypes.to(device=values.device, dtype=values.dtype)
        proto = self.transcriptomic_projection(prototypes).unsqueeze(0).expand(values.shape[0], -1, -1)
        global_value = self.transcriptomic_projection(values).unsqueeze(1)
        tokens = proto + global_value
        return self.transcriptomic_ffn(tokens)

    def forward(self, pathology: torch.Tensor, rna: torch.Tensor, cna: torch.Tensor, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.backbone.encode(pathology, rna, cna, clinical)
        morph = self._morphology_tokens(pathology)
        trans = self._transcriptomic_tokens(rna, cna)
        morph_update, _ = self.mmp_cross(morph, trans, trans)
        trans_update, _ = self.mmp_reverse(trans, morph, morph)
        morph = morph + morph_update
        trans = trans + trans_update
        morph_repr, _ = self.path_pool(morph)
        trans_repr, _ = self.omic_pool(trans)
        fused = self.fusion(torch.cat([morph_repr, trans_repr], dim=-1)).unsqueeze(1)
        clinical_context, _ = self.clinical_query(encoded.clinical, torch.cat([fused, morph, trans], dim=1), torch.cat([fused, morph, trans], dim=1))
        return self.head(clinical_context[:, 0]), pathology.new_zeros(())


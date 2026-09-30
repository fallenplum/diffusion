from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


COMPONENTS = 16
PATCHES_PER_PATIENT = 32


def fit_outer_train_prototypes(train_pool: np.ndarray, outer_train: np.ndarray, fold: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit PCA + diagonal GMM morphology centers using outer-train patients only."""
    outer_train = np.asarray(outer_train, dtype=np.int64)
    if outer_train.size == 0:
        raise RuntimeError("prototype fit has no outer-train patients")
    stride = int(train_pool.shape[1] // PATCHES_PER_PATIENT)
    sample_idx = np.arange(0, train_pool.shape[1], max(1, stride), dtype=np.int64)[:PATCHES_PER_PATIENT]
    values = np.asarray(train_pool[outer_train][:, sample_idx, :], dtype=np.float32).reshape(-1, train_pool.shape[-1])
    if values.shape[0] < COMPONENTS or not np.isfinite(values).all():
        raise RuntimeError("prototype fit input is too small or non-finite")
    seed = 910000 + int(fold) * 1009
    pca_dim = min(32, values.shape[0] - 1, values.shape[1])
    if pca_dim < 2:
        raise RuntimeError("PCA dimension is insufficient")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
    latent = pca.fit_transform(values)
    gmm = GaussianMixture(
        n_components=COMPONENTS,
        covariance_type="diag",
        reg_covar=1e-3,
        max_iter=100,
        n_init=1,
        random_state=seed,
    )
    gmm.fit(latent)
    if not bool(gmm.converged_):
        raise RuntimeError(f"GMM did not converge for fold {fold}")
    centers = pca.inverse_transform(gmm.means_).astype(np.float32)
    if centers.shape != (COMPONENTS, values.shape[1]) or not np.isfinite(centers).all():
        raise RuntimeError("prototype centers are invalid")
    norms = np.linalg.norm(centers, axis=1)
    if np.any(norms <= 1e-8):
        raise RuntimeError("prototype center collapsed to zero")
    cosine = (centers @ centers.T) / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    off_diag = cosine[~np.eye(COMPONENTS, dtype=bool)]
    if np.max(np.abs(cosine - np.eye(COMPONENTS))) >= 0.999999:
        raise RuntimeError("duplicate collapsed prototypes")
    audit = {
        "outer_train_patient_count": int(outer_train.size),
        "patch_sample_count": int(values.shape[0]),
        "patches_per_patient": PATCHES_PER_PATIENT,
        "pca_dimension": int(pca_dim),
        "gmm_components": COMPONENTS,
        "gmm_covariance": "diag",
        "gmm_converged": bool(gmm.converged_),
        "prototype_pairwise_cosine_mean": float(np.mean(off_diag)),
        "prototype_pairwise_cosine_std": float(np.std(off_diag)),
        "prototype_norm_mean": float(np.mean(norms)),
        "prototype_norm_std": float(np.std(norms)),
        "outer_test_excluded": True,
        "no_nan": True,
        "no_collapsed_duplicate": True,
        "fit_seed": seed,
        "fit_source": "outer_train patients only; train_pool pathology patches only",
    }
    return centers, audit


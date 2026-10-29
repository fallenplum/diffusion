"""Train-only masked PCA and explicit low-capacity feature construction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def checked(x, mask):
    x = np.asarray(x, np.float64)
    m = np.isfinite(x) if mask is None else np.asarray(mask, bool)
    require(x.ndim == 2 and m.shape == x.shape and len(x) > 0
            and np.isfinite(x[m]).all(), "R8_FEATURE_MATRIX_OR_MEASURED_MASK_INVALID")
    return x, m


@dataclass
class MaskedPCA:
    mean: np.ndarray
    sd: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    observed_count: np.ndarray
    constant_columns: int
    fit_rows: int
    fit_identity_sha256: str

    @classmethod
    def fit(cls, train_values, *, components, measured_mask=None, fit_identity_sha256):
        x, m = checked(train_values, measured_mask)
        require(isinstance(components, int) and 1 <= components <= min(x.shape)
                and isinstance(fit_identity_sha256, str) and len(fit_identity_sha256) == 64,
                "R8_PCA_BUDGET_OR_TRAIN_IDENTITY_INVALID")
        count = m.sum(0)
        require((count > 0).all(), "R8_PCA_FEATURE_HAS_NO_TRAIN_MEASUREMENTS")
        safe = np.where(m, x, 0.)
        mean = safe.sum(0) / count
        variance = np.where(m, (np.where(m, x, mean)-mean)**2, 0.).sum(0) / count
        sd = np.where(variance > 1e-24, np.sqrt(variance), 1.)
        z = np.where(m, (np.where(m, x, mean)-mean) / sd, 0.)
        _, singular, vh = np.linalg.svd(z, full_matrices=False)
        axes = vh[:components].copy()
        # Deterministic sign convention; no patient outcomes or heldout moments.
        for row in axes:
            pivot = np.argmax(np.abs(row))
            if row[pivot] < 0:
                row *= -1
        return cls(mean, sd, axes, singular[:components].copy(), count,
                   int(np.sum(variance <= 1e-24)), len(x), fit_identity_sha256)

    def transform(self, values, measured_mask=None):
        x, m = checked(values, measured_mask)
        require(x.shape[1] == len(self.mean), "R8_PCA_INPUT_COLUMNS_CHANGED")
        z = np.where(m, (np.where(m, x, self.mean)-self.mean) / self.sd, 0.)
        return z @ self.components.T

    def audit(self):
        return dict(components=len(self.components), input_columns=len(self.mean), fit_rows=self.fit_rows,
                    fit_identity_sha256=self.fit_identity_sha256, outcomes_used=False,
                    missing_imputation="training_mean_in_derived_feature_space_only",
                    constant_train_columns=self.constant_columns,
                    retained_effective_rank=int(np.sum(self.singular_values >
                        1e-10*max(float(self.singular_values[0]), 1.))),
                    retained_singular_values=self.singular_values.tolist(),
                    jitter_added=False, deterministic_full_SVD=True,
                    source_gene_measurements_modified=False)


@dataclass
class BlockPCA:
    blocks: tuple[MaskedPCA, ...]

    @classmethod
    def fit(cls, train_values, *, components_per_program, program_mask=None, fit_identity_sha256):
        x = np.asarray(train_values, np.float64)
        require(x.ndim == 3 and x.shape[1] == 4, "R8_BLOCK_PCA_REQUIRES_FOUR_PROGRAM_POSITIONS")
        m = np.ones(x.shape[:2], bool) if program_mask is None else np.asarray(program_mask, bool)
        require(m.shape == x.shape[:2], "R8_BLOCK_PCA_PROGRAM_MASK_CHANGED")
        return cls(tuple(MaskedPCA.fit(x[:, k], components=components_per_program,
                     measured_mask=np.broadcast_to(m[:, k, None], x[:, k].shape),
                     fit_identity_sha256=fit_identity_sha256) for k in range(4)))

    def transform(self, values, program_mask=None):
        x = np.asarray(values, np.float64)
        require(x.ndim == 3 and x.shape[1] == 4, "R8_BLOCK_PCA_PROGRAM_POSITIONS_CHANGED")
        m = np.ones(x.shape[:2], bool) if program_mask is None else np.asarray(program_mask, bool)
        require(m.shape == x.shape[:2], "R8_BLOCK_PCA_PROGRAM_MASK_CHANGED")
        return np.stack([p.transform(x[:, k], np.broadcast_to(m[:, k, None], x[:, k].shape))
                         for k, p in enumerate(self.blocks)], axis=1)


def build_low_capacity_features(image2, gene2, *, mode, query_gene2=None, program_permutation=None):
    """32-coefficient bilinear rank<=2 vs 32-coefficient additive quadratic.

    Per-program 2-by-2 bilinear coefficients are optimized directly: every
    such matrix has rank at most two, without redundant U/V factorization.
    Query-only controls never replace recipient additive molecular inputs.
    """
    h, g = np.asarray(image2, np.float64), np.asarray(gene2, np.float64)
    require(h.shape == g.shape and h.ndim == 3 and h.shape[1:] == (4, 2)
            and np.isfinite(h).all() and np.isfinite(g).all(), "R8_FIXED_BILINEAR_INPUT_SHAPE")
    main = np.concatenate((h.reshape(len(h), -1), g.reshape(len(g), -1)), axis=1)
    require(mode in ("additive", "bilinear_rank2", "additive_quadratic"), "R8_UNKNOWN_READOUT_FEATURES")
    require(not (query_gene2 is not None and program_permutation is not None), "R8_CONTROL_CHANGES_MULTIPLE_FACTORS")
    if mode == "additive":
        require(query_gene2 is None and program_permutation is None, "R8_ADDITIVE_HAS_NO_CONTEXT_TO_REPLACE")
        return main
    if mode == "additive_quadratic":
        require(query_gene2 is None and program_permutation is None, "R8_NO_INTERACTION_HAS_NO_CROSS_QUERY")
        return np.concatenate((main, main**2), axis=1)
    query = g if query_gene2 is None else np.asarray(query_gene2, np.float64)
    require(query.shape == g.shape and np.isfinite(query).all(), "R8_QUERY_ONLY_BULK_SHAPE_CHANGED")
    if program_permutation is not None:
        p = np.asarray(program_permutation)
        require(p.shape == (4,) and p.dtype.kind in "iu" and np.array_equal(np.sort(p), np.arange(4))
                and np.all(p != np.arange(4)), "R8_WRONG_PROGRAM_MUST_BE_FOUR_WAY_DERANGEMENT")
        query = query[:, p]
    cross = np.einsum("nki,nkj->nkij", h, query).reshape(len(h), 16)
    return np.concatenate((main, cross), axis=1)

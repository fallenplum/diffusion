"""Fit-only equal-dimensional additive control and explicit patient pooling."""
from __future__ import annotations
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from .readout import FitOnlyStandardizer


def equal_slide_patient_mean(values, owners, patients):
    values, owners, patients = np.asarray(values), np.asarray(owners), np.asarray(patients)
    if (values.ndim != 2 or owners.shape != (len(values),) or patients.ndim != 1
            or len(np.unique(patients)) != len(patients) or len(patients) < 1
            or not np.isfinite(values).all()):
        raise ValueError("invalid slide/patient aggregation")
    rows = []
    for patient in patients:
        keep = owners == patient
        if not keep.any():
            raise ValueError("patient has no qualified slide")
        rows.append(values[keep].mean(0))
    return np.stack(rows)


def unimodal_feature_matrix(clinical, pathology_pooled, omics_tokens, omics_mask):
    c, p, o, mask = (np.asarray(v) for v in (clinical, pathology_pooled, omics_tokens, omics_mask))
    if (c.ndim != 2 or p.shape != (len(c), 32) or o.shape != (len(c), 24, 32)
            or mask.shape != (len(c), 24) or mask.dtype != np.dtype(bool)
            or not np.isfinite(c).all() or not np.isfinite(p).all()
            or not np.isfinite(o[mask]).all()):
        raise ValueError("invalid separate-modality feature contract")
    # No clinical/pathology/omics products: a subsequent linear PCA/readout is
    # an additive function of these frozen modality features. Availability is
    # retained, rather than treating absent token values as measured zeros.
    clean = np.where(mask[..., None], o, 0.)
    return np.c_[c, p, clean.reshape(len(c), -1), mask.astype(np.float64)]


class AdditiveTwoDimensionalControl:
    """A specified two-direction main-effect control, not all nonlinear mains.

    Only fit_indices are inspected when estimating preprocessing. Matching
    two extra readout coefficients is not a claim of equal effective model
    complexity or equal prognostic information in the two feature maps.
    """
    def fit(self, matrix, fit_indices):
        x, index = np.asarray(matrix), np.asarray(fit_indices)
        if (x.ndim != 2 or min(x.shape) < 2 or index.ndim != 1 or len(index) < 3
                or not np.issubdtype(index.dtype, np.integer)
                or len(np.unique(index)) != len(index) or (index < 0).any() or (index >= len(x)).any()):
            raise ValueError("invalid fit-only additive control")
        selected = x[index].astype(np.float64, copy=True)
        self.before = FitOnlyStandardizer().fit(selected)
        normalized = self.before.transform(selected)
        if np.count_nonzero(~self.before.constant) < 2:
            raise ValueError("additive control has fewer than two varying fit directions")
        self.pca = PCA(n_components=2, svd_solver='full', whiten=False).fit(normalized)
        singular = self.pca.singular_values_
        if not np.isfinite(singular).all() or singular[1] <= 1e-10 * max(1., singular[0]):
            raise ValueError("additive control has fewer than two independent fit directions")
        self.after = FitOnlyStandardizer().fit(self.pca.transform(normalized))
        self.fit_indices_sha256 = hashlib.sha256(np.ascontiguousarray(index, dtype=np.int64).tobytes()).hexdigest()
        self.columns = x.shape[1]
        return self

    def transform(self, matrix):
        return self.after.transform(self.pca.transform(self.before.transform(matrix)))

    def public_summary(self):
        h = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        return {'status': 'FIT_ONLY_CONTROL', 'dimensions': 2, 'input_columns': self.columns,
                'fit_indices_sha256': self.fit_indices_sha256,
                'components_sha256': h(self.pca.components_),
                'explained_variance_fraction': float(self.pca.explained_variance_ratio_.sum()),
                'before': self.before.public_summary(), 'after': self.after.public_summary(),
                'outcome_selection': False,
                'scope': 'two_additive_latent_directions_not_exhaustive_nonlinear_main_effects'}

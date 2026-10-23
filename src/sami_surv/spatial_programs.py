"""Frozen-image, tissue-conditioned molecular-program regression for SAMI B.

No survival API. All learned transforms fit the current training partition.
True training targets define the shared target scale before any negative-control
remapping. Tissue classes are region labels, not inferred cell-source fractions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.utils.extmath import randomized_svd

from .morphology import Standardizer, hierarchical_weights


PCA_SEED = 7032001
RANDOM_GATE_SEED = 7032002
WRONG_DONOR_SEED = 7032003
SPATIAL_SHUFFLE_SEED = 7032004
ARMS = ('COMPOSITION', 'ADDITIVE', 'TISSUE_COND', 'RANDOM_GATE',
        'WRONG_DONOR', 'SPATIAL_SHUFFLE')
DESIGN_DIMENSIONS = dict(COMPOSITION=6, ADDITIVE=70, TISSUE_COND=166,
                         RANDOM_GATE=166, WRONG_DONOR=166, SPATIAL_SHUFFLE=166)


def _finite_matrix(x, name, columns=None):
    array = np.asarray(x, np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a nonempty finite matrix; no missing-value imputation')
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f'{name} must have {columns} columns')
    return array


@dataclass
class PCATransform:
    standardizer: Standardizer
    components: np.ndarray
    singular_values: np.ndarray
    component_scale: np.ndarray
    fit_audit: dict

    @classmethod
    def fit(cls, X, donor, section):
        X = _finite_matrix(X, 'frozen UNI2 X', 1536)
        if len(X) < 32:
            raise ValueError('PCA32 requires at least 32 training observations')
        standardizer = Standardizer.fit(X, donor, section)
        z = standardizer.transform(X)
        weight = hierarchical_weights(donor, section)
        weighted = np.sqrt(weight)[:, None] * z
        _, singular, components = randomized_svd(weighted, n_components=32,
            n_iter=7, n_oversamples=10, random_state=PCA_SEED, flip_sign=False,
            power_iteration_normalizer='auto', transpose='auto')
        # Lock signs to feature loading coordinates, not sample ordering.
        pivot = np.argmax(np.abs(components), axis=1)
        sign = np.sign(components[np.arange(32), pivot])
        sign[sign == 0.] = 1.
        components *= sign[:, None]
        if not np.isfinite(components).all() or not np.isfinite(singular).all():
            raise ValueError('PCA produced nonfinite components')
        if not np.allclose(components @ components.T, np.eye(32), atol=1e-10, rtol=1e-10):
            raise ValueError('PCA components failed orthogonality qualification')
        train_scores = z @ components.T
        score_mean = weight @ train_scores
        component_scale = np.sqrt(np.sum(weight[:, None]*(train_scores-score_mean)**2, axis=0))
        if not np.isfinite(component_scale).all() or np.any(component_scale <= 1e-12):
            raise ValueError('PCA32 whitening requires all train component scales >1e-12')
        variance = float(np.square(weighted).sum())
        audit = dict(method='weighted_randomized_SVD_of_train_standardized_local_features',
                     components=32, seed=PCA_SEED, n_iter=7, n_oversamples=10,
                     loading_sign='maximum_absolute_feature_loading_positive_first_index_tie',
                     training_rows=len(X), training_donors=len(np.unique(donor)),
                     training_sections=len(np.unique(section)), weight_sum=float(weight.sum()),
                     weighted_total_variance=variance,
                     retained_variance_fraction=float(np.square(component_scale).sum()/variance) if variance else 0.,
                     randomized_SVD_retained_variance_fraction=float(np.square(singular).sum()/variance) if variance else 0.,
                     whitening='actual_train_projected_scores_weighted_population_std_not_approximate_S',
                     maximum_relative_S_vs_actual_scale_difference=float(np.max(np.abs(singular/component_scale-1.))),
                     neighbor_features_used_to_fit_PCA=False)
        return cls(standardizer, components, singular, component_scale, audit)

    def transform(self, X):
        return (self.standardizer.transform(X) @ self.components.T)/self.component_scale

    def arrays(self):
        return dict(mean=self.standardizer.mean, scale=self.standardizer.scale,
                    components=self.components, singular_values=self.singular_values,
                    component_scale=self.component_scale)

    @classmethod
    def from_arrays(cls, arrays, fit_audit=None):
        return cls(Standardizer(np.asarray(arrays['mean']), np.asarray(arrays['scale'])),
                   np.asarray(arrays['components']), np.asarray(arrays['singular_values']),
                   np.asarray(arrays['component_scale']), fit_audit or {})


def _probability_matrix(p, n, name):
    p = _finite_matrix(p, name, 3)
    if len(p) != n or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1., atol=1e-6, rtol=0):
        raise ValueError(f'{name} must be aligned three-class probabilities')
    return p


def random_gates(neighbor_pc):
    neighbor_pc = _finite_matrix(neighbor_pc, 'neighbor PCs', 32)
    matrix = np.random.default_rng(RANDOM_GATE_SEED).normal(size=(32, 3))
    return softmax(neighbor_pc @ matrix / math.sqrt(32), axis=1)


def build_design(local_pc, neighbor_pc, p_local, p_neighbor, arm):
    if arm not in ARMS:
        raise ValueError('Unknown locked Stage B arm')
    local_pc = _finite_matrix(local_pc, 'local PCs', 32)
    neighbor_pc = _finite_matrix(neighbor_pc, 'neighbor PCs', 32)
    n = len(local_pc)
    if len(neighbor_pc) != n:
        raise ValueError('Local and neighbor PC rows must align')
    p_local = _probability_matrix(p_local, n, 'local tissue probabilities')
    p_neighbor = _probability_matrix(p_neighbor, n, 'neighbor tissue probabilities')
    if arm == 'COMPOSITION':
        design = np.concatenate([p_local, p_neighbor], axis=1)
    else:
        base = np.concatenate([local_pc, neighbor_pc, p_local, p_neighbor], axis=1)
        if arm == 'ADDITIVE':
            design = base
        else:
            gate = random_gates(neighbor_pc) if arm == 'RANDOM_GATE' else p_neighbor
            interaction = np.concatenate([gate[:, c:c+1] * local_pc for c in range(3)], axis=1)
            design = np.concatenate([base, interaction], axis=1)
    if design.shape != (n, DESIGN_DIMENSIONS[arm]):
        raise AssertionError('Locked design dimension mismatch')
    return design


def huber_objective_gradient(theta, x, target, weight, ridge=.1, delta=1.):
    """sum_i w_i mean_k Huber_delta(yhat-y) + ridge/2 ||W||^2."""
    n, d = x.shape
    k = target.shape[1]
    W = theta[:d*k].reshape(d, k)
    b = theta[d*k:]
    residual = x @ W + b - target
    absolute = np.abs(residual)
    point = np.where(absolute <= delta, .5*residual**2, delta*(absolute-.5*delta))
    data = float(np.dot(weight, point.mean(1)))
    regularizer = .5*ridge*float(np.sum(W*W))
    derivative = np.clip(residual, -delta, delta)*weight[:, None]/k
    gradient = np.concatenate([(x.T @ derivative + ridge*W).ravel(), derivative.sum(0)])
    return data+regularizer, gradient


def _validate_target_map(index_map, n):
    if index_map is None:
        return np.arange(n, dtype=np.int64)
    index = np.asarray(index_map)
    if index.dtype.kind not in 'iu' or index.shape != (n,) or (index < 0).any() or (index >= n).any():
        raise ValueError('Target remapping must be one in-bounds training-row index per recipient')
    return index.astype(np.int64, copy=False)


@dataclass
class ProgramRegressor:
    W: np.ndarray
    b: np.ndarray
    x_standardizer: Standardizer
    y_standardizer: Standardizer
    fit_audit: dict

    @classmethod
    def fit(cls, design, program_y, donor, section, target_index_map=None):
        x = _finite_matrix(design, 'regression design')
        y = _finite_matrix(program_y, 'true program targets', 4)
        if len(x) != len(y):
            raise ValueError('Design and target rows must align')
        target_index = _validate_target_map(target_index_map, len(y))
        # Always fit both scalers using true training rows in their own order.
        x_scaler = Standardizer.fit(x, donor, section)
        y_scaler = Standardizer.fit(y, donor, section)
        z = x_scaler.transform(x)
        true_target = y_scaler.transform(y)
        target = true_target[target_index]
        weight = hierarchical_weights(donor, section)
        theta0 = np.zeros((z.shape[1]+1)*4, np.float64)
        result = minimize(huber_objective_gradient, theta0,
                          args=(z, target, weight, .1, 1.), method='L-BFGS-B', jac=True,
                          options=dict(maxiter=2000, ftol=1e-12, gtol=1e-6, maxls=50))
        objective, grad = huber_objective_gradient(result.x, z, target, weight, .1, 1.)
        W, b = result.x[:z.shape[1]*4].reshape(z.shape[1],4), result.x[z.shape[1]*4:]
        reg = .05*float(np.sum(W*W))
        gradient_inf = float(np.max(np.abs(grad)))
        audit = dict(solver='scipy_L-BFGS-B', success=bool(result.success),
                     status=int(result.status), message=str(result.message), iterations=int(result.nit),
                     function_evaluations=int(result.nfev), maxiter=2000, ftol=1e-12, gtol=1e-6,
                     maxls=50, accepted_gradient_inf=1e-5, gradient_inf=gradient_inf,
                     objective=float(objective), data_loss=float(objective-reg), regularizer=reg,
                     loss='natural_weighted_mean_over_four_Huber_delta1_targets',
                     ridge=.1, delta=1., intercept_penalized=False,
                     initialization='all_zero', training_rows=len(y), targets=4,
                     design_dimension=z.shape[1], allocated_parameters=int(result.x.size),
                     regularized_parameters=int(W.size), weight_sum=float(weight.sum()),
                     target_scaler_source='unmodified_true_training_programs_before_remapping',
                     target_rows_remapped=int(np.sum(target_index != np.arange(len(y)))),
                     target_unique_source_rows=int(len(np.unique(target_index))))
        if (not result.success or not np.isfinite(objective) or not np.isfinite(result.x).all()
                or not np.isfinite(gradient_inf) or gradient_inf > 1e-5):
            raise RuntimeError(f'OPTIMIZATION_FAILED: {audit}')
        return cls(W, b, x_scaler, y_scaler, audit)

    def predict_standardized(self, design):
        return self.x_standardizer.transform(design) @ self.W + self.b

    def predict(self, design):
        return self.predict_standardized(design)*self.y_standardizer.scale + self.y_standardizer.mean

    def arrays(self):
        return dict(W=self.W, b=self.b, xmean=self.x_standardizer.mean, xscale=self.x_standardizer.scale,
                    ymean=self.y_standardizer.mean, yscale=self.y_standardizer.scale)

    @classmethod
    def from_arrays(cls, arrays, fit_audit=None):
        return cls(np.asarray(arrays['W']), np.asarray(arrays['b']),
                   Standardizer(np.asarray(arrays['xmean']),np.asarray(arrays['xscale'])),
                   Standardizer(np.asarray(arrays['ymean']),np.asarray(arrays['yscale'])),fit_audit or {})


def _group_columns(donor, section, tissue_y):
    donor, section, tissue = np.asarray(donor), np.asarray(section), np.asarray(tissue_y)
    if donor.ndim != 1 or not len(donor) or section.shape != donor.shape or tissue.shape != donor.shape:
        raise ValueError('Aligned nonempty donor, section, tissue vectors required')
    if tissue.dtype.kind not in 'iu' or not np.isin(tissue, [0,1,2]).all():
        raise ValueError('Three locked tissue classes required')
    for s in np.unique(section):
        if len(np.unique(donor[section == s])) != 1:
            raise ValueError('A section cannot cross donors')
    return donor, section, tissue


def wrong_donor_target_map(donor, section, tissue_y):
    """Cycle donors, permute same-class source rows, then repeat as necessary.

    This is a remapping with possible repeats, NOT a global permutation.
    Source/recipient donor identities stay private with the runner.
    """
    donor, section, tissue = _group_columns(donor, section, tissue_y)
    donors = np.unique(donor)
    if len(donors) < 2:
        raise ValueError('Wrong-donor control requires at least two training donors')
    rng = np.random.default_rng(WRONG_DONOR_SEED)
    index = np.full(len(donor), -1, np.int64)
    for j, d in enumerate(donors):
        source = donors[(j+1)%len(donors)]
        for c in np.unique(tissue[donor == d]):
            recipient_rows = np.flatnonzero((donor == d)&(tissue == c))
            source_rows = np.flatnonzero((donor == source)&(tissue == c))
            if not len(source_rows):
                raise ValueError('Cyclic source donor lacks a recipient tissue class')
            index[recipient_rows] = np.resize(rng.permutation(source_rows), len(recipient_rows))
    if (index < 0).any() or np.any(donor[index] == donor) or not np.array_equal(tissue[index], tissue):
        raise AssertionError('Wrong-donor target pairing invariant failed')
    audit = dict(kind='wrong_donor_same_tissue_target_remap', seed=WRONG_DONOR_SEED,
                 donor_rule='lexicographic_training_donor_cyclic_successor',
                 row_rule='source_class_shuffle_then_repeat_to_recipient_count',
                 donor_count=int(len(donors)), recipient_rows=len(index),
                 source_unique_rows=int(len(np.unique(index))),
                 repeated_source_row_uses=int(len(index)-len(np.unique(index))),
                 same_donor_pairs=0, mismatched_tissue_pairs=0,
                 global_bijection_claim=False, all_four_targets_share_same_map=True)
    return index, audit


def spatial_shuffle_target_map(donor, section, tissue_y):
    """Within each section/class: shuffle rows then cyclically rotate by one."""
    donor, section, tissue = _group_columns(donor, section, tissue_y)
    rng = np.random.default_rng(SPATIAL_SHUFFLE_SEED)
    index = np.arange(len(donor), dtype=np.int64)
    singleton_groups = 0
    groups = 0
    for s in np.unique(section):
        for c in np.unique(tissue[section == s]):
            rows = np.flatnonzero((section == s)&(tissue == c))
            order = rng.permutation(rows)
            index[order] = np.roll(order, 1)
            singleton_groups += int(len(rows) == 1)
            groups += 1
    fixed = int(np.sum(index == np.arange(len(index))))
    if (len(np.unique(index)) != len(index) or not np.array_equal(section[index],section)
            or not np.array_equal(tissue[index],tissue) or fixed != singleton_groups):
        raise AssertionError('Spatial shuffle target pairing invariant failed')
    return index, dict(kind='same_section_same_tissue_spatial_target_shuffle',seed=SPATIAL_SHUFFLE_SEED,
                       group_count=groups,singleton_groups=singleton_groups,fixed_rows=fixed,
                       recipient_rows=len(index),bijective=True,all_four_targets_share_same_map=True)

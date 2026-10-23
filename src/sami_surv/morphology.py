"""SAMI Stage A: outcome-free, donor-held-out frozen-feature qualification.

The probabilities are tissue-class model outputs, not calibrated cell fractions.
No survival endpoint is accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def hierarchical_weights(donor, section, y=None) -> np.ndarray:
    """Sum-one donor/section/spot weights, optionally donor/class/section/spot."""
    donor, section = np.asarray(donor), np.asarray(section)
    if donor.ndim != 1 or section.shape != donor.shape or not len(donor):
        raise ValueError("Nonempty aligned donor/section vectors required")
    labels = None if y is None else np.asarray(y)
    if labels is not None and labels.shape != donor.shape:
        raise ValueError("Labels must align with rows")
    weights = np.zeros(len(donor), np.float64)
    donors = np.unique(donor)
    for d in donors:
        di = np.flatnonzero(donor == d)
        groups = [di] if labels is None else [di[labels[di] == c] for c in np.unique(labels[di])]
        for gi in groups:
            slides = np.unique(section[gi])
            for s in slides:
                si = gi[section[gi] == s]
                weights[si] = 1.0 / (len(donors) * len(groups) * len(slides) * len(si))
    if not np.isclose(weights.sum(), 1., atol=1e-13):
        raise AssertionError("Hierarchical weights do not sum to one")
    return weights


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x, donor, section):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Finite feature matrix required")
        w = hierarchical_weights(donor, section)
        if len(w) != len(x):
            raise ValueError("Feature/metadata length mismatch")
        mean = w @ x
        var = np.sum(w[:, None] * (x - mean) ** 2, axis=0)
        scale = np.sqrt(np.maximum(var, 0.))
        scale[scale <= 1e-12] = 1.
        return cls(mean=mean, scale=scale)

    def transform(self, x):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Invalid features for fitted standardizer")
        return (x - self.mean) / self.scale


def objective_gradient(theta, x, y, weight, n_classes, C):
    """Normalized weighted CE + ||W||^2/(2C); intercept is unpenalized."""
    n, d = x.shape
    W = theta[:d*n_classes].reshape(d, n_classes)
    b = theta[d*n_classes:]
    logits = x @ W + b
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    objective = -np.dot(weight, logp[np.arange(n), y]) + np.sum(W*W)/(2.*C)
    residual = np.exp(logp)
    residual[np.arange(n), y] -= 1.
    residual *= weight[:, None]
    gradient = np.concatenate([(x.T @ residual + W/C).ravel(), residual.sum(axis=0)])
    return float(objective), gradient


@dataclass
class MultinomialModel:
    W: np.ndarray
    b: np.ndarray
    standardizer: Standardizer
    fit_audit: dict[str, Any]

    def predict_proba(self, x):
        logits = self.standardizer.transform(x) @ self.W + self.b
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

    def arrays(self):
        return dict(W=self.W, b=self.b, mean=self.standardizer.mean, scale=self.standardizer.scale)

    @classmethod
    def from_arrays(cls, arrays, fit_audit=None):
        return cls(np.asarray(arrays['W']), np.asarray(arrays['b']),
                   Standardizer(np.asarray(arrays['mean']), np.asarray(arrays['scale'])), fit_audit or {})


def fit_model(x, y, donor, section, *, n_classes, C, maxiter=2000,
              ftol=1e-12, gtol=1e-6, maxls=50, accepted_gradient_inf=1e-5):
    x, y = np.asarray(x), np.asarray(y)
    if C <= 0 or len(y) != len(x) or not np.array_equal(np.unique(y), np.arange(n_classes)):
        raise ValueError("Every class must occur in this training partition; C must be positive")
    standardizer = Standardizer.fit(x, donor, section)
    z = standardizer.transform(x)
    w = hierarchical_weights(donor, section, y)
    theta = np.zeros((z.shape[1] + 1)*n_classes, np.float64)
    prior = np.bincount(y, weights=w, minlength=n_classes)
    theta[z.shape[1]*n_classes:] = np.log(prior) - np.log(prior).mean()
    result = minimize(objective_gradient, theta, args=(z, y, w, n_classes, C),
                      method='L-BFGS-B', jac=True,
                      options=dict(maxiter=maxiter, ftol=ftol, gtol=gtol, maxls=maxls))
    value, gradient = objective_gradient(result.x, z, y, w, n_classes, C)
    grad_inf = float(np.max(np.abs(gradient)))
    audit = dict(C=float(C), solver='scipy_L-BFGS-B', success=bool(result.success),
                 status=int(result.status), message=str(result.message), iterations=int(result.nit),
                 function_evaluations=int(result.nfev), objective=float(value),
                 gradient_inf=grad_inf, accepted_gradient_inf=float(accepted_gradient_inf),
                 weight_sum=float(w.sum()), n_train=int(len(y)),
                 n_train_donors=int(len(np.unique(donor))), n_train_sections=int(len(np.unique(section))),
                 class_counts=np.bincount(y, minlength=n_classes).tolist(),
                 allocated_parameters=int(result.x.size), regularized_parameters=int(z.shape[1]*n_classes),
                 scale_equal_one_dimensions=int(np.sum(standardizer.scale == 1.)))
    if not result.success or not np.isfinite(value) or not np.isfinite(result.x).all() or grad_inf > accepted_gradient_inf:
        raise RuntimeError(f"OPTIMIZATION_FAILED: {audit}")
    d = z.shape[1]
    return MultinomialModel(result.x[:d*n_classes].reshape(d, n_classes), result.x[d*n_classes:], standardizer, audit)


def classification_metrics(y, probabilities, donor, section, n_classes):
    y, p = np.asarray(y), np.asarray(probabilities, np.float64)
    if p.shape != (len(y), n_classes) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-8):
        raise ValueError("Invalid probability matrix")
    prediction = np.argmax(p, axis=1)
    natural_w = hierarchical_weights(donor, section)
    balanced_w = hierarchical_weights(donor, section, y)
    cm = np.zeros((n_classes, n_classes), np.int64)
    weighted_cm = np.zeros((n_classes, n_classes), np.float64)
    np.add.at(cm, (y, prediction), 1)
    np.add.at(weighted_cm, (y, prediction), balanced_w)
    recalls = [float(weighted_cm[c, c]/weighted_cm[c].sum()) if weighted_cm[c].sum() else None
               for c in range(n_classes)]
    if any(v is None for v in recalls):
        raise ValueError("Locked balanced-accuracy evaluation requires all classes")
    onehot = np.eye(n_classes)[y]
    confidence, correct = p.max(axis=1), prediction == y
    ece = 0.
    edges = np.linspace(0., 1., 11)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= low) & (confidence < high if index < 9 else confidence <= high)
        mass = natural_w[mask].sum()
        if mass:
            ece += abs(float(np.dot(natural_w[mask], correct[mask]-confidence[mask])))
    return dict(balanced_accuracy=float(np.mean(recalls)),
                accuracy=float(np.dot(natural_w, prediction == y)),
                negative_log_likelihood=float(-np.dot(natural_w, np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)))),
                brier_multiclass_sum=float(np.dot(natural_w, np.sum((p-onehot)**2, axis=1))),
                ece_10_equal_width_bins=float(ece),
                class_recall=recalls, confusion_counts=cm.tolist(),
                balanced_weighted_confusion=weighted_cm.tolist(), n_spots=int(len(y)))


def train_prior(y, donor, section, n_classes):
    w = hierarchical_weights(donor, section)
    return np.bincount(np.asarray(y), weights=w, minlength=n_classes)


def select_c(inner_scores, candidates, tie_tolerance=1e-12):
    """Only completed inner scores enter selection; outer labels have no API here."""
    candidates = sorted(float(c) for c in candidates)
    if set(inner_scores) != set(candidates) or not candidates:
        raise ValueError("All locked candidates must have scores")
    lengths = {len(inner_scores[c]) for c in candidates}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("Every candidate must contain every inner donor")
    means = {c: float(np.mean(inner_scores[c])) for c in candidates}
    if not all(np.isfinite(v) for v in means.values()):
        raise ValueError("Nonfinite inner result")
    best = max(means.values())
    return min(c for c in candidates if means[c] >= best-tie_tolerance), means


def outer_inner_splits(donor):
    donor = np.asarray(donor)
    donors = np.unique(donor)
    for outer_index, held in enumerate(donors):
        outer_train, outer_test = np.flatnonzero(donor != held), np.flatnonzero(donor == held)
        inners = []
        for inner_held in donors[donors != held]:
            train = outer_train[donor[outer_train] != inner_held]
            valid = outer_train[donor[outer_train] == inner_held]
            inners.append((train, valid))
        yield outer_index, outer_train, outer_test, inners

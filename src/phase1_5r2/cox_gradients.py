"""Analytic negative Cox partial-likelihood derivatives.

The Phase 1.5R implementation used the opposite sign.  These functions are
deliberately small and NumPy-only so they can be independently checked
against PyTorch autograd and finite differences before any TCGA job starts.
"""

from __future__ import annotations

import numpy as np


def _validate_inputs(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    risks = np.asarray(risks, dtype=np.float64)
    if times.ndim != 1 or events.ndim != 1 or risks.ndim != 1:
        raise ValueError("times, events and risks must be one-dimensional")
    if not (len(times) == len(events) == len(risks)):
        raise ValueError("times, events and risks must have equal length")
    if not np.isfinite(times).all() or not np.isfinite(risks).all():
        raise ValueError("times and risks must be finite")
    if not np.isin(events, [0, 1]).all():
        raise ValueError("events must be binary")
    return times, events, risks


def cox_loss_numpy(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    """Return the negative Breslow-style partial log likelihood."""

    times, events, risks = _validate_inputs(times, events, risks)
    event_indices = np.flatnonzero(events == 1)
    if not len(event_indices):
        raise ValueError("at least one event is required")
    value = 0.0
    for event_index in event_indices:
        value += float(np.logaddexp.reduce(risks[times >= times[event_index]]) - risks[event_index])
    return value / len(event_indices)


def cox_risk_coefficients(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Return the exact derivative ``d negative_Cox_loss / d risk``.

    For an event ``i`` and a patient ``k`` in its risk set, the contribution is
    ``p_ik``; the event patient's own contribution additionally subtracts one.
    This is the corrected sign required by the Phase 1.5R2 prompt.
    """

    times, events, risks = _validate_inputs(times, events, risks)
    coefficients = np.zeros(len(times), dtype=np.float64)
    event_indices = np.flatnonzero(events == 1)
    if not len(event_indices):
        raise ValueError("at least one event is required")
    for event_index in event_indices:
        mask = times >= times[event_index]
        logits = risks[mask] - np.max(risks[mask])
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(), 1e-300)
        coefficients[mask] += probabilities
        coefficients[event_index] -= 1.0
    return coefficients / len(event_indices)


def legacy_wrong_cox_risk_coefficients(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Reproduce the pre-repair sign for the audit table only."""

    return -cox_risk_coefficients(times, events, risks)


def finite_difference_gradient(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Central finite-difference derivative of :func:`cox_loss_numpy`."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    risks = np.asarray(risks, dtype=np.float64)
    result = np.zeros_like(risks)
    for index in range(len(risks)):
        plus = risks.copy()
        minus = risks.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (cox_loss_numpy(times, events, plus) - cox_loss_numpy(times, events, minus)) / (2.0 * epsilon)
    return result


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denominator)

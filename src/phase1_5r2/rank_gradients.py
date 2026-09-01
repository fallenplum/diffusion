"""Comparable-pair ranking objective and its analytic derivative."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def comparable_pairs(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    return np.where((events[:, None] == 1) & (times[:, None] < times[None, :]))


def rank_loss_numpy(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    risks = np.asarray(risks, dtype=np.float64)
    pair_i, pair_j = comparable_pairs(times, events)
    if not len(pair_i):
        return 0.0
    delta = risks[pair_i] - risks[pair_j]
    return float(np.logaddexp(0.0, -delta).mean())


def rank_risk_coefficients(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Return ``d mean softplus(-(risk_i-risk_j)) / d risks``."""

    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    risks = np.asarray(risks, dtype=np.float64)
    if not (times.ndim == events.ndim == risks.ndim == 1) or not (len(times) == len(events) == len(risks)):
        raise ValueError("times, events and risks must be aligned one-dimensional arrays")
    pair_i, pair_j = comparable_pairs(times, events)
    coefficients = np.zeros(len(times), dtype=np.float64)
    if not len(pair_i):
        return coefficients
    delta = risks[pair_i] - risks[pair_j]
    gradient = expit(-delta) / len(pair_i)
    np.add.at(coefficients, pair_i, -gradient)
    np.add.at(coefficients, pair_j, gradient)
    return coefficients


def finite_difference_rank_gradient(
    times: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    risks = np.asarray(risks, dtype=np.float64)
    result = np.zeros_like(risks)
    for index in range(len(risks)):
        plus = risks.copy()
        minus = risks.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (rank_loss_numpy(times, events, plus) - rank_loss_numpy(times, events, minus)) / (2.0 * epsilon)
    return result

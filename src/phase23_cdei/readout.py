"""Low-capacity conditional Efron Cox with an immutable main-effect offset.

No scalar stacking coefficient, endpoint selection or feature selection occurs
here. Callers must supply fit-only data and persist any fitted states privately.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from src.phase10_trimodal.math import efron_weights


class FitOnlyStandardizer:
    def fit(self, x):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError("finite nonempty fit-only matrix required")
        self.mean = x.mean(0)
        raw_sd = x.std(0, ddof=0)
        self.constant = raw_sd < 1e-8
        self.scale = np.where(self.constant, 1., raw_sd)
        self.fit_n = len(x)
        return self

    def transform(self, x):
        x = np.asarray(x, np.float64)
        if not hasattr(self, "mean") or x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("invalid heldout feature transform")
        return (x - self.mean) / self.scale

    def public_summary(self):
        h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        return {"fit_patients": self.fit_n, "feature_columns": len(self.mean),
                "constant_fit_columns": int(self.constant.sum()),
                "mean_sha256": h(self.mean), "scale_sha256": h(self.scale),
                "fit_only": True}


def conditional_objective(x, time, event, offset, penalty=10.0):
    x, time, event, offset = (np.asarray(a) for a in (x, time, event, offset))
    if (x.ndim != 2 or min(x.shape) < 1 or time.shape != (len(x),)
            or event.shape != time.shape or offset.shape != time.shape
            or not np.isin(event, [0, 1]).all() or event.sum() < 1
            or not all(np.isfinite(a).all() for a in (x, time, event, offset))
            or (time < 0).any() or not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("invalid conditional Cox fit inputs")
    x, offset = np.array(x, np.float64, copy=True), np.array(offset, np.float64, copy=True)
    event = np.array(event, np.float64, copy=True)
    weights = efron_weights(time, event)
    with np.errstate(divide="ignore"):
        loga = np.log(weights)
    count = event.sum()

    def evaluate(w, hessian=False):
        w = np.asarray(w, np.float64)
        if w.shape != (x.shape[1],) or not np.isfinite(w).all():
            raise ValueError("invalid conditional coefficients")
        risk = offset + x @ w
        terms = loga + risk
        denom = logsumexp(terms, axis=1)
        p = np.exp(terms - denom[:, None])
        loss = float((denom.sum() - risk[event == 1].sum()) / count + .5 * penalty * (w @ w))
        gradient = x.T @ ((p.sum(0) - event) / count) + penalty * w
        if not hessian:
            return loss, gradient
        means = p @ x
        h = (x.T @ (x * p.sum(0)[:, None]) - means.T @ means) / count + penalty * np.eye(x.shape[1])
        return loss, gradient, h
    return evaluate


def fit_conditional_cox(x, time, event, offset, penalty=10.0):
    objective = conditional_objective(x, time, event, offset, penalty)
    result = minimize(objective, np.zeros(np.asarray(x).shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12, "maxls": 50})
    w = result.x.copy()
    polish = 0
    loss, gradient, h = objective(w, hessian=True)
    while np.abs(gradient).max() > 1e-7 and polish < 30:
        direction = np.linalg.solve(h, gradient)
        step = 1.
        accepted = False
        for _ in range(40):
            proposal = w - step * direction
            candidate, _ = objective(proposal)
            if candidate <= loss - 1e-4 * step * (gradient @ direction) + 1e-14:
                accepted = True
                break
            step *= .5
        if not accepted:
            raise RuntimeError("conditional Cox line search failed")
        w = proposal
        loss, gradient, h = objective(w, hessian=True)
        polish += 1
    if not np.isfinite(loss) or np.abs(gradient).max() > 1e-7:
        raise RuntimeError("conditional Cox failed its locked gradient gate")
    return w, {"status": "CONVERGED", "parameters": len(w), "penalty": penalty,
               "iterations": int(result.nit), "newton_polish_steps": polish,
               "objective": loss, "gradient_max": float(np.abs(gradient).max()),
               "offset_trainable": False, "intercept": False}


def risk_with_offset(x, weights, offset):
    x, weights, offset = (np.asarray(a, np.float64) for a in (x, weights, offset))
    if x.ndim != 2 or weights.shape != (x.shape[1],) or offset.shape != (len(x),):
        raise ValueError("risk/offset shapes disagree")
    if not all(np.isfinite(a).all() for a in (x, weights, offset)):
        raise ValueError("nonfinite risk input")
    delta = x @ weights
    return offset + delta, delta

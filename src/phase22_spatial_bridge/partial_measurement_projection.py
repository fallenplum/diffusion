"""Strictly convex, soft bulk measurement projection with an image KL prior.

This module consumes no outcomes, clinical labels, or patient identifiers.  One
call represents one patient and one bulk measurement, including all selected
positions from all that patient's WSIs.  ``measurement_weights`` are fixed
measurement approximations, not attention weights or survival pooling weights.

The frozen dictionary has shape [states, genes] and simplex rows.  On valid
positions the prior is strictly positive.  ``omega`` is either a diagonal scale
vector [genes] or an arbitrary *fixed* linear measurement operator [q, genes].
Unmeasured genes are removed before applying this operator; they never become
observed zero expression.  Nothing here estimates rho or the assay transform.

The implementation uses float64 log-domain KL-prox mirror descent, objective
backtracking, and a separately checked simplex-constrained KKT residual.  A
small change in objective alone is never reported as convergence.  The default
100-iteration budget is a limit, not a claim that every problem is solved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectionConfig:
    """Numerical controls; these do not select biological hyperparameters."""

    max_iterations: int = 100
    objective_relative_tolerance: float = 1e-8
    kkt_tolerance: float = 1e-7
    simplex_tolerance: float = 1e-12
    initial_step_size: float = 1.0
    backtracking_factor: float = 0.5
    max_backtracking_steps: int = 60

    def validate(self) -> None:
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        if not isinstance(self.max_backtracking_steps, int) or self.max_backtracking_steps < 1:
            raise ValueError("max_backtracking_steps must be a positive integer")
        for name in ("objective_relative_tolerance", "kkt_tolerance", "simplex_tolerance", "initial_step_size"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.backtracking_factor < 1:
            raise ValueError("backtracking_factor must lie strictly between zero and one")


@dataclass
class ProjectionResult:
    pi: np.ndarray
    u: np.ndarray
    predicted_bulk: np.ndarray
    converged: bool
    status: str
    iterations: int
    objective: float
    kkt_projected_residual: float
    simplex_residual: float
    objective_relative_change: float
    objective_history: list[float]
    kkt_history: list[float]
    step_size_history: list[float]
    backtracking_rejections: int
    active_positions: int
    measured_genes: int
    joint_bulk_constraints: int
    problem_sha256: str

    def aggregate_diagnostics(self) -> dict[str, Any]:
        """Return solver diagnostics without positions, genes, or state arrays."""
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"pi", "u", "predicted_bulk"}
        }


def _softmax_log(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return np.exp(logp), logp


def _simplex(name: str, x: np.ndarray, *, positive: bool, tolerance: float) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")
    invalid_sign = np.any(x <= 0) if positive else np.any(x < 0)
    if invalid_sign:
        raise ValueError(f"{name} must be {'strictly positive' if positive else 'nonnegative'}")
    if not np.all(np.abs(x.sum(axis=-1) - 1.0) <= tolerance):
        raise ValueError(f"{name} must have simplex rows; no implicit renormalization is performed")


class _Problem:
    def __init__(
        self, pi0: np.ndarray, dictionary: np.ndarray, bulk: np.ndarray, *,
        u0: np.ndarray, measurement_weights: np.ndarray | None,
        row_mask: np.ndarray | None, measured_gene_mask: np.ndarray | None,
        omega: np.ndarray | None, rho: float, tau: float, tau_u: float,
        lambda_b: float, config: ProjectionConfig,
    ) -> None:
        config.validate()
        self.config = config
        prior = np.array(pi0, dtype=np.float64, copy=True)
        D = np.array(dictionary, dtype=np.float64, copy=True)
        b = np.array(bulk, dtype=np.float64, copy=True)
        unobserved = np.array(u0, dtype=np.float64, copy=True)
        if prior.ndim != 2 or not prior.shape[0] or not prior.shape[1]:
            raise ValueError("pi0 must have nonempty shape [positions, states]")
        n, k = prior.shape
        if D.ndim != 2 or D.shape[0] != k or not D.shape[1]:
            raise ValueError("dictionary must have nonempty shape [states, genes]")
        g = D.shape[1]
        if b.shape != (g,) or unobserved.shape != (k,):
            raise ValueError("bulk or u0 shape does not match dictionary")
        for name, value in (("rho", rho), ("tau", tau), ("tau_u", tau_u), ("lambda_b", lambda_b)):
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
        if not 0 <= rho <= 1 or tau <= 0 or tau_u <= 0 or lambda_b < 0:
            raise ValueError("require rho in [0,1], positive tau/tau_u, and nonnegative lambda_b")
        valid = self._mask(row_mask, n, "row_mask")
        measured = self._mask(measured_gene_mask, g, "measured_gene_mask")
        if not valid.any():
            raise ValueError("at least one valid position is required")
        _simplex("pi0 on valid positions", prior[valid], positive=True, tolerance=config.simplex_tolerance)
        _simplex("u0", unobserved, positive=True, tolerance=config.simplex_tolerance)
        _simplex("dictionary", D, positive=False, tolerance=config.simplex_tolerance)
        # Padded values are never evaluated or normalized; preserve them verbatim.
        if not np.all(np.isfinite(prior[~valid])):
            raise ValueError("padded priors must be finite, although they need not be simplex")
        if np.any(~np.isfinite(b[measured])) or np.any(b[measured] <= 0):
            raise ValueError("measured bulk composition must be finite and strictly positive")
        if measured.all() and abs(b.sum() - 1) > config.simplex_tolerance:
            raise ValueError("fully measured bulk must be a composition summing to one")
        if measurement_weights is None:
            a = np.zeros(n, dtype=np.float64)
            a[valid] = 1.0 / valid.sum()
        else:
            a = np.array(measurement_weights, dtype=np.float64, copy=True)
            if a.shape != (n,) or not np.all(np.isfinite(a)):
                raise ValueError("measurement_weights must be finite with shape [positions]")
            if np.any(a[valid] <= 0) or np.any(a[~valid] != 0):
                raise ValueError("valid weights must be positive; padded weights must be zero")
            if abs(a.sum() - 1) > config.simplex_tolerance:
                raise ValueError("measurement_weights must sum to one; no silent normalization")
        if omega is None:
            scales = np.ones(g, dtype=np.float64)
            M = D[:, measured]
            target = b[measured]
            omega_for_hash = scales[measured]
        else:
            op = np.asarray(omega, dtype=np.float64)
            if op.ndim == 1 and op.shape == (g,):
                if np.any(~np.isfinite(op[measured])) or np.any(op[measured] < 0):
                    raise ValueError("measured diagonal omega scales must be finite and nonnegative")
                M = D[:, measured] * op[measured][None, :]
                target = b[measured] * op[measured]
                omega_for_hash = op[measured]
            elif op.ndim == 2 and op.shape[1] == g:
                if np.any(~np.isfinite(op[:, measured])):
                    raise ValueError("omega columns for measured genes must be finite")
                M = D[:, measured] @ op[:, measured].T
                target = op[:, measured] @ b[measured]
                omega_for_hash = op[:, measured]
            else:
                raise ValueError("omega must have shape [genes] or [measurements, genes]")
        self.prior_all = prior
        self.valid = valid
        self.measured = measured
        self.p0 = prior[valid].copy()
        self.logp0 = np.log(self.p0)
        self.u0 = unobserved
        self.logu0 = np.log(unobserved)
        self.D = D
        self.a = a[valid]
        self.M = M
        self.target = target
        self.rho, self.tau, self.tau_u, self.lambda_b = float(rho), float(tau), float(tau_u), float(lambda_b)
        digest = hashlib.sha256()
        scalar = {"rho": self.rho, "tau": self.tau, "tau_u": self.tau_u, "lambda_b": self.lambda_b, "config": asdict(config)}
        digest.update(json.dumps(scalar, sort_keys=True, allow_nan=False).encode())
        for arr in (self.p0, self.u0, D, self.a, measured, M, target, omega_for_hash):
            arr = np.ascontiguousarray(arr)
            digest.update(str((arr.dtype.str, arr.shape)).encode())
            digest.update(arr.tobytes())
        self.sha256 = digest.hexdigest()

    @staticmethod
    def _mask(mask: np.ndarray | None, size: int, name: str) -> np.ndarray:
        if mask is None:
            return np.ones(size, dtype=bool)
        arr = np.asarray(mask)
        if arr.shape != (size,) or arr.dtype != np.dtype(bool):
            raise ValueError(f"{name} must be an explicit boolean mask of shape [{size}]")
        return arr.copy()

    def mean_state(self, pi: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.rho * (self.a @ pi) + (1 - self.rho) * u

    def measurement_gradient(self, pi: np.ndarray, u: np.ndarray) -> np.ndarray:
        residual = self.mean_state(pi, u) @ self.M - self.target
        return self.lambda_b * (self.M @ residual)

    def objective(self, pi: np.ndarray, logpi: np.ndarray, u: np.ndarray, logu: np.ndarray) -> float:
        klp = np.sum(pi * (logpi - self.logp0), axis=1)
        klu = float(np.sum(u * (logu - self.logu0)))
        residual = self.mean_state(pi, u) @ self.M - self.target
        return float(self.tau * (self.a @ klp) + self.tau_u * klu + 0.5 * self.lambda_b * (residual @ residual))

    def kkt(self, pi: np.ndarray, logpi: np.ndarray, u: np.ndarray, logu: np.ndarray) -> tuple[float, float]:
        shared = self.measurement_gradient(pi, u)
        # Divide pi gradients by a_i, as required for the weighted KL metric.
        # The additive +tau terms are row constants and cancel in this projection.
        gp = self.tau * (logpi - self.logp0) + self.rho * shared[None, :]
        gu = self.tau_u * (logu - self.logu0) + (1 - self.rho) * shared
        rp = gp - np.sum(pi * gp, axis=1, keepdims=True)
        ru = gu - u @ gu
        stationarity = max(float(np.max(np.abs(rp))), float(np.max(np.abs(ru))))
        simplex = max(float(np.max(np.abs(pi.sum(axis=1) - 1))), float(abs(u.sum() - 1)))
        return stationarity, simplex


def solve_partial_measurement_projection(
    pi0: np.ndarray, dictionary: np.ndarray, bulk: np.ndarray, *,
    u0: np.ndarray, measurement_weights: np.ndarray | None = None,
    row_mask: np.ndarray | None = None, measured_gene_mask: np.ndarray | None = None,
    omega: np.ndarray | None = None, rho: float = 0.5, tau: float = 1.0,
    tau_u: float = 1.0, lambda_b: float = 1.0,
    config: ProjectionConfig | None = None,
) -> ProjectionResult:
    """Solve the locked convex projection, reporting numerical failure explicitly.

    ``objective_relative_change`` uses |J_new-J_old| / max(1, |J_old|).
    The KKT residual is the maximum absolute simplex-tangent stationarity
    residual after dividing position gradients by a_i; it is not merely an
    objective change or a gradient of the measurement term alone.  Both this
    residual and objective-change gates must pass before ``converged=True``.

    To handle multiple WSIs, concatenate the patient's positions and supply one
    set of a_i summing to one and one bulk vector.  This is one joint constraint;
    the solver never replicates bulk constraints by WSI or infers WSI identity.
    Padded rows have zero a_i and are returned unchanged, outside the problem.
    """
    cfg = ProjectionConfig() if config is None else config
    problem = _Problem(pi0, dictionary, bulk, u0=u0, measurement_weights=measurement_weights,
                       row_mask=row_mask, measured_gene_mask=measured_gene_mask, omega=omega,
                       rho=rho, tau=tau, tau_u=tau_u, lambda_b=lambda_b, config=cfg)
    pi, u = problem.p0.copy(), problem.u0.copy()
    logpi, logu = problem.logp0.copy(), problem.logu0.copy()
    objective = problem.objective(pi, logpi, u, logu)
    kkt, simplex = problem.kkt(pi, logpi, u, logu)
    objectives, kkts, steps = [objective], [kkt], []
    relative_change = 0.0
    rejected = 0
    converged = kkt <= cfg.kkt_tolerance and simplex <= cfg.simplex_tolerance
    status = "CONVERGED" if converged else "NOT_CONVERGED"
    for _ in range(cfg.max_iterations):
        if converged:
            break
        shared = problem.measurement_gradient(pi, u)
        step = cfg.initial_step_size
        accepted = False
        for _backtrack in range(cfg.max_backtracking_steps):
            if problem.rho == 0:
                next_pi, next_logpi = problem.p0.copy(), problem.logp0.copy()
            else:
                proposal = (logpi + step * problem.tau * problem.logp0 - step * problem.rho * shared[None, :]) / (1 + step * problem.tau)
                next_pi, next_logpi = _softmax_log(proposal)
            if problem.rho == 1:
                next_u, next_logu = problem.u0.copy(), problem.logu0.copy()
            else:
                proposal_u = (logu + step * problem.tau_u * problem.logu0 - step * (1 - problem.rho) * shared) / (1 + step * problem.tau_u)
                next_u, next_logu = _softmax_log(proposal_u)
            next_objective = problem.objective(next_pi, next_logpi, next_u, next_logu)
            # Permit only floating-point rounding at the objective's scale.
            rounding = 16 * np.finfo(np.float64).eps * max(1.0, abs(objective))
            if np.isfinite(next_objective) and next_objective <= objective + rounding:
                accepted = True
                break
            rejected += 1
            step *= cfg.backtracking_factor
        if not accepted:
            status = "LINE_SEARCH_FAILED"
            break
        relative_change = abs(next_objective - objective) / max(1.0, abs(objective))
        pi, logpi, u, logu = next_pi, next_logpi, next_u, next_logu
        objective = next_objective
        kkt, simplex = problem.kkt(pi, logpi, u, logu)
        objectives.append(objective)
        kkts.append(kkt)
        steps.append(step)
        converged = (relative_change <= cfg.objective_relative_tolerance and
                     kkt <= cfg.kkt_tolerance and simplex <= cfg.simplex_tolerance)
        if converged:
            status = "CONVERGED"
    all_pi = problem.prior_all.copy()
    all_pi[problem.valid] = pi
    return ProjectionResult(
        pi=all_pi, u=u, predicted_bulk=problem.mean_state(pi, u) @ problem.D,
        converged=converged, status=status, iterations=len(steps), objective=objective,
        kkt_projected_residual=kkt, simplex_residual=simplex,
        objective_relative_change=relative_change, objective_history=objectives,
        kkt_history=kkts, step_size_history=steps, backtracking_rejections=rejected,
        active_positions=int(problem.valid.sum()), measured_genes=int(problem.measured.sum()),
        joint_bulk_constraints=1, problem_sha256=problem.sha256,
    )


def evaluate_projection_objective(
    pi: np.ndarray, u: np.ndarray, pi0: np.ndarray, dictionary: np.ndarray,
    bulk: np.ndarray, **problem_kwargs: Any,
) -> float:
    """Independently evaluate the same objective for audit / reference solvers."""
    cfg = problem_kwargs.pop("config", ProjectionConfig())
    defaults = dict(measurement_weights=None, row_mask=None, measured_gene_mask=None,
                    omega=None, rho=0.5, tau=1.0, tau_u=1.0, lambda_b=1.0)
    defaults.update(problem_kwargs)
    problem = _Problem(pi0, dictionary, bulk, config=cfg, **defaults)
    p = np.asarray(pi, dtype=np.float64)[problem.valid]
    unobserved = np.asarray(u, dtype=np.float64)
    _simplex("pi", p, positive=True, tolerance=cfg.simplex_tolerance)
    _simplex("u", unobserved, positive=True, tolerance=cfg.simplex_tolerance)
    return problem.objective(p, np.log(p), unobserved, np.log(unobserved))

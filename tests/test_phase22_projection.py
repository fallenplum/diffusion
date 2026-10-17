"""Numerical qualification of the outcome-blind Phase22 convex projection.

The SciPy comparator below implements the objective and analytic gradient
independently of the production solver.  It is not used by production code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.phase22_spatial_bridge.partial_measurement_projection import (
    ProjectionConfig,
    evaluate_projection_objective,
    solve_partial_measurement_projection,
)


def _case(seed=31, n=3, k=3, g=5):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k) * 2, n)
    D = rng.dirichlet(np.ones(g) * 2, k)
    b = rng.dirichlet(np.ones(g) * 2)
    u0 = rng.dirichlet(np.ones(k) * 2)
    a = rng.dirichlet(np.ones(n) * 2)
    return p, D, b, dict(u0=u0, measurement_weights=a)


def _independent_slsqp(p0, D, b, *, u0, measurement_weights, rho, lambda_b, tau=1.0, tau_u=1.0, measured_gene_mask=None, omega=None):
    """Constrained primal solve with independent direct KL algebra/Jacobian."""
    n, k = p0.shape
    measured = np.ones(len(b), dtype=bool) if measured_gene_mask is None else measured_gene_mask
    Dm, bm = D[:, measured], b[measured]
    if omega is not None:
        if omega.ndim == 1:
            Dm, bm = Dm * omega[measured], bm * omega[measured]
        else:
            Dm, bm = Dm @ omega[:, measured].T, omega[:, measured] @ bm
    a = measurement_weights

    def fun(x):
        p, u = x[:n*k].reshape(n, k), x[n*k:]
        state = rho * np.sum(a[:, None] * p, axis=0) + (1-rho) * u
        residual = state @ Dm - bm
        f = tau*np.sum(a[:, None]*p*np.log(p/p0)) + tau_u*np.sum(u*np.log(u/u0)) + lambda_b/2*np.sum(residual**2)
        shared = lambda_b * (Dm @ residual)
        gp = a[:, None]*(tau*(np.log(p/p0)+1) + rho*shared)
        gu = tau_u*(np.log(u/u0)+1) + (1-rho)*shared
        return float(f), np.concatenate([gp.ravel(), gu])

    constraint = np.zeros((n+1, (n+1)*k))
    for row in range(n+1):
        constraint[row, row*k:(row+1)*k] = 1
    return minimize(fun, np.concatenate([p0.ravel(), u0]), method="SLSQP", jac=True,
                    constraints=[LinearConstraint(constraint, np.ones(n+1), np.ones(n+1))],
                    bounds=Bounds(np.full((n+1)*k, 1e-12), np.ones((n+1)*k)),
                    options={"ftol": 1e-13, "maxiter": 1000})


def numerical_parity_cases():
    rows = []
    for seed, rho, lb, operator in ((31, .25, .1, "identity"), (47, .5, 1., "diagonal"),
                                     (83, .75, 10., "identity"), (97, 1., 10., "matrix"),
                                     (101, 0., 10., "identity"), (109, .5, 10., "masked")):
        p0, D, b, kwargs = _case(seed)
        if operator == "diagonal":
            kwargs["omega"] = np.array([1., .2, 2., 1.5, .7])
        elif operator == "matrix":
            kwargs["omega"] = np.array([[1., .2, 0., 0., 0.], [0., .3, .4, .1, .2], [.1, 0., .2, .5, .4]])
        elif operator == "masked":
            kwargs["measured_gene_mask"] = np.array([True, False, True, True, False])
            b[~kwargs["measured_gene_mask"]] = np.nan
        common = dict(kwargs, rho=rho, lambda_b=lb)
        ours = solve_partial_measurement_projection(p0, D, b, **common)
        independent = _independent_slsqp(p0, D, b, **common)
        n, k = p0.shape
        refp, refu = independent.x[:n*k].reshape(n, k), independent.x[n*k:]
        rows.append(dict(seed=seed, rho=rho, lambda_b=lb, measurement_operator=operator,
                         independent_success=bool(independent.success), independent_message=str(independent.message),
                         independent_objective=float(independent.fun), projection_objective=ours.objective,
                         objective_absolute_difference=abs(ours.objective-independent.fun),
                         primal_max_absolute_difference=max(float(np.max(np.abs(ours.pi-refp))), float(np.max(np.abs(ours.u-refu)))),
                         status=ours.status, iterations=ours.iterations,
                         kkt_projected_residual=ours.kkt_projected_residual,
                         simplex_residual=ours.simplex_residual,
                         maximum_objective_increase=float(max(0., np.max(np.diff(ours.objective_history)))) if ours.iterations else 0.))
    return rows


@pytest.mark.parametrize("seed", [31, 47, 83])
def test_lambda_zero_returns_priors_bitwise(seed):
    p, D, b, kwargs = _case(seed)
    result = solve_partial_measurement_projection(p, D, b, **kwargs, lambda_b=0)
    np.testing.assert_array_equal(result.pi, p)
    np.testing.assert_array_equal(result.u, kwargs["u0"])
    assert result.converged and result.iterations == 0 and result.objective == 0


def test_rho_zero_preserves_image_prior_bitwise():
    p, D, b, kwargs = _case()
    result = solve_partial_measurement_projection(p, D, b, **kwargs, rho=0, lambda_b=10)
    np.testing.assert_array_equal(result.pi, p)
    assert result.converged
    assert np.max(np.abs(result.u-kwargs["u0"])) > 1e-5


def test_rho_one_preserves_unobserved_prior_bitwise():
    p, D, b, kwargs = _case()
    result = solve_partial_measurement_projection(p, D, b, **kwargs, rho=1, lambda_b=10)
    np.testing.assert_array_equal(result.u, kwargs["u0"])
    assert result.converged


def test_bulk_equal_complete_prior_is_optimal_without_update():
    p, D, _, kwargs = _case()
    rho = .75
    b = (rho*(kwargs["measurement_weights"] @ p)+(1-rho)*kwargs["u0"]) @ D
    result = solve_partial_measurement_projection(p, D, b, **kwargs, rho=rho, lambda_b=10)
    np.testing.assert_array_equal(result.pi, p)
    np.testing.assert_array_equal(result.u, kwargs["u0"])
    assert result.objective < 1e-28 and result.converged and result.iterations == 0


def test_padding_is_excluded_not_observed_zero():
    p, D, b, kwargs = _case()
    base = solve_partial_measurement_projection(p, D, b, **kwargs, lambda_b=10)
    padded = np.concatenate([p, [[0., 123., 7.]]])
    result = solve_partial_measurement_projection(padded, D, b, u0=kwargs["u0"],
        measurement_weights=np.r_[kwargs["measurement_weights"], 0.], row_mask=np.array([True, True, True, False]), lambda_b=10)
    np.testing.assert_array_equal(result.pi[:3], base.pi)
    np.testing.assert_array_equal(result.pi[3], padded[3])
    np.testing.assert_array_equal(result.u, base.u)
    assert result.problem_sha256 == base.problem_sha256


def test_unmeasured_values_and_dictionary_distribution_cannot_influence_fit():
    p, D, b, kwargs = _case()
    mask = np.array([True, True, True, False, False])
    bad_bulk = b.copy()
    bad_bulk[~mask] = [np.nan, -99.]
    D2 = D.copy()
    D2[:, 3], D2[:, 4] = D[:, 4], D[:, 3]
    scales = np.array([1., .5, 2., np.nan, -99.])
    one = solve_partial_measurement_projection(p, D, b, **kwargs, measured_gene_mask=mask, omega=scales)
    two = solve_partial_measurement_projection(p, D2, bad_bulk, **kwargs, measured_gene_mask=mask, omega=scales)
    np.testing.assert_array_equal(one.pi, two.pi)
    np.testing.assert_array_equal(one.u, two.u)


def test_no_measured_genes_returns_priors():
    p, D, b, kwargs = _case()
    b[:] = np.nan
    result = solve_partial_measurement_projection(p, D, b, **kwargs, measured_gene_mask=np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(result.pi, p)
    np.testing.assert_array_equal(result.u, kwargs["u0"])
    assert result.converged


def test_positive_priors_and_declared_simplexes_are_enforced():
    p, D, b, kwargs = _case()
    broken = p.copy()
    broken[0] = [0., .5, .5]
    with pytest.raises(ValueError, match="strictly positive"):
        solve_partial_measurement_projection(broken, D, b, **kwargs)
    with pytest.raises(ValueError, match="simplex"):
        solve_partial_measurement_projection(p, D*2, b, **kwargs)
    with pytest.raises(ValueError, match="sum to one"):
        solve_partial_measurement_projection(p, D, b, u0=kwargs["u0"], measurement_weights=kwargs["measurement_weights"]*2)
    with pytest.raises(ValueError, match="strictly positive"):
        solve_partial_measurement_projection(p, D, b, u0=np.array([0., .5, .5]))


@pytest.mark.parametrize("parameters", [dict(rho=-.1), dict(rho=1.1), dict(tau=0), dict(tau_u=-1), dict(lambda_b=-1)])
def test_illegal_hyperparameters_fail(parameters):
    p, D, b, kwargs = _case()
    with pytest.raises(ValueError):
        solve_partial_measurement_projection(p, D, b, **kwargs, **parameters)


def test_objective_small_change_does_not_replace_kkt():
    p, D, b, kwargs = _case()
    config = ProjectionConfig(max_iterations=1, objective_relative_tolerance=1., kkt_tolerance=1e-14)
    result = solve_partial_measurement_projection(p, D, b, **kwargs, lambda_b=10, config=config)
    assert result.objective_relative_change < config.objective_relative_tolerance
    assert result.kkt_projected_residual > config.kkt_tolerance
    assert not result.converged and result.status == "NOT_CONVERGED" and result.iterations == 1


def test_zero_iteration_budget_reports_not_converged():
    p, D, b, kwargs = _case()
    result = solve_partial_measurement_projection(p, D, b, **kwargs, config=ProjectionConfig(max_iterations=0))
    assert not result.converged and result.status == "NOT_CONVERGED" and result.iterations == 0


def test_default_hundred_step_failure_is_retained_despite_flat_objective():
    p, D, b, kwargs = _case()
    result = solve_partial_measurement_projection(p, D, b, **kwargs, omega=np.full(5, 10000.), lambda_b=10.)
    assert result.iterations == 100 and result.status == "NOT_CONVERGED" and not result.converged
    assert result.objective_relative_change < ProjectionConfig().objective_relative_tolerance
    assert result.kkt_projected_residual > ProjectionConfig().kkt_tolerance


def test_permutation_equivariance_and_single_joint_multi_wsi_constraint():
    p, D, b, kwargs = _case(n=6)
    # Treat first/last three positions as two WSIs: they are solved together.
    result = solve_partial_measurement_projection(p, D, b, **kwargs, lambda_b=10)
    permutation = np.array([5, 2, 0, 4, 1, 3])
    shuffled = solve_partial_measurement_projection(p[permutation], D, b, u0=kwargs["u0"],
        measurement_weights=kwargs["measurement_weights"][permutation], lambda_b=10)
    np.testing.assert_allclose(shuffled.pi, result.pi[permutation], rtol=0, atol=2e-15)
    np.testing.assert_allclose(shuffled.u, result.u, rtol=0, atol=2e-15)
    joint = (.5*(kwargs["measurement_weights"] @ result.pi)+.5*result.u) @ D
    np.testing.assert_allclose(result.predicted_bulk, joint, rtol=0, atol=2e-15)
    assert result.joint_bulk_constraints == 1 and result.active_positions == 6


def test_shared_state_tilt_uses_gradient_divided_by_measurement_weights():
    p, D, b, kwargs = _case()
    result = solve_partial_measurement_projection(p, D, b, **kwargs, lambda_b=10)
    logratio = np.log(result.pi/p)
    logodds_shift = logratio-logratio[:, :1]
    np.testing.assert_allclose(logodds_shift, np.broadcast_to(logodds_shift[:1], logodds_shift.shape), rtol=0, atol=2e-14)


def test_objective_convexity_along_feasible_segment():
    p0, D, b, kwargs = _case()
    p1, _, _, extra = _case(47)
    p2, _, _, extra2 = _case(83)
    common = dict(kwargs, rho=.75, lambda_b=10.)
    j1 = evaluate_projection_objective(p1, extra["u0"], p0, D, b, **common)
    j2 = evaluate_projection_objective(p2, extra2["u0"], p0, D, b, **common)
    pm, um = .3*p1+.7*p2, .3*extra["u0"]+.7*extra2["u0"]
    jm = evaluate_projection_objective(pm, um, p0, D, b, **common)
    assert jm < .3*j1+.7*j2


def test_float64_no_input_mutation_and_outcome_free_interface():
    p, D, b, kwargs = _case()
    snapshots = [x.copy() for x in (p, D, b, kwargs["u0"], kwargs["measurement_weights"])]
    result = solve_partial_measurement_projection(p, D, b, **kwargs)
    assert result.pi.dtype == np.float64 and result.u.dtype == np.float64
    for actual, old in zip((p, D, b, kwargs["u0"], kwargs["measurement_weights"]), snapshots):
        np.testing.assert_array_equal(actual, old)
    with pytest.raises(TypeError):
        solve_partial_measurement_projection(p, D, b, **kwargs, survival_labels=np.zeros(3))
    aggregate = result.aggregate_diagnostics()
    assert "pi" not in aggregate and "u" not in aggregate and "predicted_bulk" not in aggregate


def test_independent_constrained_solver_parity():
    for row in numerical_parity_cases():
        assert row["independent_success"], row
        assert row["status"] == "CONVERGED", row
        assert row["objective_absolute_difference"] < 2e-10, row
        assert row["primal_max_absolute_difference"] < 1e-5, row
        assert row["kkt_projected_residual"] <= 1e-7, row
        assert row["simplex_residual"] <= 1e-12, row
        assert row["maximum_objective_increase"] <= 4e-15, row


def write_numerical_qualification(destination):
    """Write only synthetic numerical evidence, never cohort measurements."""
    import scipy
    rows = numerical_parity_cases()
    passed = all(r["independent_success"] and r["status"] == "CONVERGED" and
                 r["objective_absolute_difference"] < 2e-10 and r["primal_max_absolute_difference"] < 1e-5 and
                 r["kkt_projected_residual"] <= 1e-7 and r["simplex_residual"] <= 1e-12 and
                 r["maximum_objective_increase"] <= 4e-15 for r in rows)
    root = Path(__file__).resolve().parents[1]
    source = root / "src/phase22_spatial_bridge/partial_measurement_projection.py"
    p, D, b, kwargs = _case()
    stress = solve_partial_measurement_projection(p, D, b, **kwargs, omega=np.full(5, 10000.), lambda_b=10.)
    stress_preserved = not stress.converged and stress.status == "NOT_CONVERGED" and stress.iterations == 100
    passed = passed and stress_preserved
    report = dict(status="PASS_SYNTHETIC_NUMERICAL_QUALIFICATION" if passed else "FAIL",
                  scope="Synthetic convex problems only; no ST, TCGA, DSS, or survival fit read or performed.",
                  config=ProjectionConfig().__dict__, independent_solver="SciPy SLSQP, primal simplex equality constraints, analytic independent objective and gradient",
                  locked_parity_gates=dict(objective_absolute_difference=2e-10, primal_max_absolute_difference=1e-5,
                                          kkt_projected_residual=1e-7, simplex_residual=1e-12,
                                          maximum_objective_increase=4e-15),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=rows,
                  default_budget_stress_case=dict(omega_diagonal_scale=10000., lambda_b=10.,
                     expected_status="NOT_CONVERGED", retained_failure_pass=stress_preserved,
                     diagnostics=stress.aggregate_diagnostics()),
                  limitations=["The numerical gate is not a biological or assay qualification.",
                               "Default 100 iterations may fail on other qualified real inputs; NOT_CONVERGED is retained.",
                               "Strict mathematical uniqueness is not identifiability of biological spatial truth.",
                               "Padding is excluded; measured-zero is rejected until an explicit positive-composition transform is qualified."])
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "reports/phase22_spatial_bridge/NUMERICAL_QUALIFICATION.json"
    output = write_numerical_qualification(target)
    print(json.dumps({"status": output["status"], "cases": len(output["cases"])}))
    raise SystemExit(0 if output["status"] == "PASS_SYNTHETIC_NUMERICAL_QUALIFICATION" else 1)

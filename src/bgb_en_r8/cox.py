"""Convex full-risk Efron Cox with a training-only residual-risk SD bound.

No metric, validation label, line search over hyperparameters or checkpoint
selection enters fitting. The explicit coefficient vector has no intercept.
The global spectral scaling ensures SD(Xw)<=||w|| in the training partition.
An L2 ball then limits SD(delta)/SD(frozen offset) by fixed gamma.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def efron_log_weights(time, event):
    t, e = np.asarray(time, np.float64), np.asarray(event)
    require(t.ndim == 1 and e.shape == t.shape and len(t) > 1
            and np.isfinite(t).all() and (t >= 0).all()
            and np.isin(e, (0, 1)).all() and e.sum() > 0,
            "R8_EFRON_TRAIN_ENDPOINT_INVALID")
    rows = []
    for at in np.unique(t[e == 1]):
        tied = (t == at) & (e == 1)
        d = int(tied.sum())
        for ell in range(d):
            weights = (t >= at).astype(np.float64) - ell/d * tied
            with np.errstate(divide="ignore"):
                rows.append(np.log(weights))
    return np.stack(rows), e.astype(np.float64)


def efron_loss_gradient(risk, log_weights, event):
    r = np.asarray(risk, np.float64)
    require(r.shape == event.shape and np.isfinite(r).all(), "R8_EFRON_RISK_INVALID")
    scores = log_weights + r[None]
    den = logsumexp(scores, axis=1)
    count = float(event.sum())
    value = (den.sum() - np.dot(event, r)) / count
    grad = (np.exp(scores-den[:, None]).sum(0)-event) / count
    return float(value), grad


@dataclass
class ProtectedOffsetCox:
    mean: np.ndarray
    sd: np.ndarray
    spectral_scale: float
    coefficient: np.ndarray
    bound: float
    gamma: float
    ridge_lambda: float
    fit_identity_sha256: str
    receipt: dict

    def transformed(self, features):
        z = np.asarray(features, np.float64)
        require(z.ndim == 2 and z.shape[1] == len(self.mean) and np.isfinite(z).all(),
                "R8_PREDICT_FEATURES_INVALID")
        return (z-self.mean)/self.sd/self.spectral_scale

    def delta(self, features):
        return self.transformed(features) @ self.coefficient

    def predict(self, features, frozen_offset):
        base = np.asarray(frozen_offset, np.float64)
        delta = self.delta(features)
        require(base.shape == delta.shape and np.isfinite(base).all(), "R8_PREDICT_OFFSET_INVALID")
        return base+delta


def fit_protected_offset_cox(features, time, event, frozen_offset, *, gamma, ridge_lambda,
                             fit_identity_sha256, maxiter=2000, kkt_tolerance=5e-6):
    x = np.asarray(features, np.float64);base = np.asarray(frozen_offset, np.float64)
    require(x.ndim == 2 and len(x) > 1 and x.shape[1] > 0 and np.isfinite(x).all()
            and base.shape == (len(x),) and np.isfinite(base).all(), "R8_COX_INPUT_SHAPE_OR_VALUES")
    require(np.isfinite(gamma) and gamma > 0 and np.isfinite(ridge_lambda) and ridge_lambda > 0
            and isinstance(fit_identity_sha256, str) and len(fit_identity_sha256) == 64,
            "R8_COX_PRELOCKED_BUDGET_OR_FIT_IDENTITY_REQUIRED")
    require(type(maxiter) is int and maxiter > 0 and kkt_tolerance > 0, "R8_COX_SOLVER_BUDGET_INVALID")
    la, events = efron_log_weights(time, event)
    require(len(events) == len(x), "R8_COX_ENDPOINT_ROW_COUNT")
    mean = x.mean(0);sd = x.std(0);active = sd > 1e-12
    sd = np.where(active, sd, 1.)
    z = (x-mean)/sd
    covariance = z.T@z/len(z)
    largest = max(float(np.linalg.eigvalsh(covariance)[-1]), 0.)
    spectral = max(np.sqrt(largest), 1.)
    z = z/spectral
    offset_sd = float(base.std())
    require(offset_sd > 1e-10, "R8_FROZEN_OFFSET_HAS_NO_TRAIN_VARIANCE")
    bound = float(gamma*offset_sd)

    def fun(u):
        w = bound*u
        loss, dr = efron_loss_gradient(base+z@w, la, events)
        value = loss + ridge_lambda/2 * (w@w)
        gradient = bound*(z.T@dr + ridge_lambda*w)
        return float(value), gradient

    trace = []
    def callback(u):
        val, grad = fun(u)
        trace.append(dict(iteration=len(trace)+1, objective=val,
                          unconstrained_gradient_max=float(np.abs(grad).max()),
                          normalized_coefficient_norm=float(np.linalg.norm(u)),
                          validation_metrics_evaluated=False))
    zero = np.zeros(x.shape[1])
    result = minimize(fun, zero, jac=True, method="SLSQP",
        constraints=[dict(type="ineq", fun=lambda u: 1.-u@u, jac=lambda u: -2*u)],
        callback=callback,
        options=dict(maxiter=maxiter, ftol=1e-12, disp=False))
    value, gradient = fun(result.x)
    norm = float(np.linalg.norm(result.x));slack = float(1.-result.x@result.x)
    multiplier = max(0., -float(gradient@result.x)/(2*max(norm*norm, 1e-30))) if norm >= 1-1e-5 else 0.
    stationarity = gradient+2*multiplier*result.x
    residual = float(np.max(np.abs(stationarity)))
    complementarity = float(abs(multiplier*slack))
    coef = bound*result.x;delta = z@coef
    ratio = float(delta.std()/offset_sd)
    require(np.isfinite(value) and np.isfinite(coef).all() and slack >= -1e-8
            and residual <= kkt_tolerance and complementarity <= kkt_tolerance
            and ratio <= gamma+1e-8,
            "R8_CONSTRAINED_COX_DID_NOT_MEET_KKT_OR_SD_BOUND")
    receipt = dict(schema="r8_protected_offset_cox_fit_v1", solver="SLSQP_convex_Efron_ridge_L2_ball",
        status="CONVERGED_KKT_AND_TRAIN_SD_PROTECTION", solver_success=bool(result.success),
        iterations=int(result.nit), function_evaluations=int(result.nfev),
        diagnostic_objective_evaluations=len(trace), iteration_trace=trace,
        optimizer_fits=1, neural_updates=0, target_label_fit_rows=len(x),
        registered_coefficients=x.shape[1], active_variance_columns=int(active.sum()),
        train_design_rank=int(np.linalg.matrix_rank(z)), learned_intercepts=0,
        risk_residual_scale_learned=False, gamma=float(gamma), ridge_lambda=float(ridge_lambda),
        coefficient_L2=float(np.linalg.norm(coef)), coefficient_L2_bound=bound,
        residual_SD=float(delta.std()), frozen_offset_train_SD=offset_sd, residual_SD_ratio=ratio,
        covariance_largest_eigenvalue_before_spectral_scaling=largest,
        spectral_scale=spectral, objective=value, kkt_stationarity_max=residual,
        kkt_stationarity_coordinate="unit_ball_u_where_coefficient_equals_bound_times_u",
        constraint_slack=slack, kkt_multiplier=multiplier, kkt_complementarity=complementarity,
        training_riskset_complete=True, riskset_microbatching=False,
        train_identity_sha256=fit_identity_sha256,
        validation_or_test_labels_used=False, hyperparameters_selected_from_metrics=False,
        prediction_uses_train_transform_only=True)
    return ProtectedOffsetCox(mean, sd, spectral, coef, bound, float(gamma),
                              float(ridge_lambda), fit_identity_sha256, receipt)


def replay_cox_certificate(model, features, time, event, frozen_offset, *, tolerance=5e-6):
    """Independently recompute the saved optimum's KKT/SD without optimizing."""
    require(isinstance(model, ProtectedOffsetCox), "R8_SAVED_CONVEX_MODEL_REQUIRED")
    x=np.asarray(features,np.float64);base=np.asarray(frozen_offset,np.float64)
    require(x.ndim==2 and base.shape==(len(x),) and np.isfinite(base).all(), "R8_CERTIFICATE_INPUT_SHAPE")
    expected_mean=x.mean(0);expected_sd=x.std(0);expected_sd=np.where(expected_sd>1e-12,expected_sd,1.)
    centered=(x-expected_mean)/expected_sd
    expected_spectral=max(np.sqrt(max(float(np.linalg.eigvalsh(centered.T@centered/len(x))[-1]),0.)),1.)
    require(np.array_equal(model.mean,expected_mean) and np.array_equal(model.sd,expected_sd)
            and abs(model.spectral_scale-expected_spectral)<=2e-12,
            "R8_SAVED_TRAIN_ONLY_FEATURE_TRANSFORM_CHANGED")
    z=model.transformed(x);la,e=efron_log_weights(time,event)
    loss,dr=efron_loss_gradient(base+z@model.coefficient,la,e)
    require(model.bound>0 and model.gamma>0 and model.ridge_lambda>0,
            "R8_SAVED_RISK_BUDGET_INVALID")
    u=model.coefficient/model.bound
    grad=model.bound*(z.T@dr+model.ridge_lambda*model.coefficient)
    norm=float(np.linalg.norm(u));slack=float(1-u@u)
    dual=max(0.,-float(grad@u)/(2*max(norm*norm,1e-30))) if norm>=1-1e-5 else 0.
    stationarity=float(np.max(np.abs(grad+2*dual*u)))
    complementary=float(abs(dual*slack));ratio=float(np.std(z@model.coefficient)/max(np.std(base),1e-30))
    require(abs(model.bound-model.gamma*np.std(base))<=1e-12 and slack>=-1e-8
            and stationarity<=tolerance and complementary<=tolerance and ratio<=model.gamma+1e-8,
            "R8_SAVED_MODEL_KKT_OR_TRAIN_SD_CERTIFICATE_FAILED")
    require(abs(loss+model.ridge_lambda/2*(model.coefficient@model.coefficient)-model.receipt["objective"])<=1e-10,
            "R8_SAVED_MODEL_OBJECTIVE_CHANGED")
    return dict(status="PASS_ACTUAL_SAVED_COEFFICIENT_KKT_AND_SD_REPLAY", stationarity_max=stationarity,
        complementarity=complementary, constraint_slack=slack, train_residual_SD_ratio=ratio,
        gamma=model.gamma, ridge_lambda=model.ridge_lambda, objective=loss+model.ridge_lambda/2*(model.coefficient@model.coefficient),
        training_preprocessing_moments_replayed_without_fitting_or_selection=True,
        actual_saved_coefficient_forward_replayed=True, new_optimizer_fits=0)

"""Locked fit-only PCA and convex linear Cox readout with an exact offset.

Only fit arrays supplied by callers are consumed. No endpoint registry, file,
model-selection or neural training API is accessed by this module.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA

RANK = 16
INPUT_DIM = 256


def _finite(value, name, *, ndim=None):
    value = np.asarray(value)
    if (not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value)
            or value.size == 0 or (ndim is not None and value.ndim != ndim)):
        raise ValueError(name + ' must be a nonempty real numeric array')
    value = np.array(value, np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(name + ' contains nonfinite values')
    return value


def fit_projection(features):
    """raw256 -> fit-only full-SVD PCA16 -> fit-score population scaling."""
    features = _finite(features, 'fit slot features', ndim=2)
    if features.shape[1] != INPUT_DIM or len(features) <= RANK:
        raise ValueError('projection requires 256 columns and more than 16 fit patients')
    # SVD is applied to the original, centered slot coordinates. There is no
    # feature standardization before PCA or rank/regularization search.
    model = PCA(RANK, svd_solver='full', whiten=False).fit(features)
    centered = features - model.mean_
    singular = np.linalg.svd(centered, compute_uv=False)
    tolerance = float(singular[0] * max(centered.shape) * np.finfo(np.float64).eps) if len(singular) else 0.
    effective_rank = int((singular > tolerance).sum())
    if effective_rank < RANK:
        raise ValueError('fit input rank is below the locked PCA16 dimension')
    scores = centered @ model.components_.T
    mean, sd = scores.mean(0), scores.std(0, ddof=0)
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError('a locked PCA component has zero fit variance')
    return {'pca_mean': model.mean_.copy(), 'components': model.components_.copy(),
            'pca_score_mean': mean, 'pca_score_sd': sd,
            'explained_variance_ratio_sum': float(model.explained_variance_ratio_.sum()),
            'input_fit_rank': effective_rank, 'input_fit_rank_tolerance': tolerance,
            'fit_n': len(features), 'input_dim': INPUT_DIM, 'rank': RANK,
            'input_standardized_before_PCA': False, 'whiten': False, 'svd_solver': 'full'}


def transform_projection(features, state):
    features = _finite(features, 'slot features', ndim=2)
    if features.shape[1] != INPUT_DIM or state.get('rank') != RANK or state.get('input_dim') != INPUT_DIM:
        raise ValueError('projection dimension/state mismatch')
    mean = _finite(state['pca_mean'], 'PCA mean', ndim=1)
    components = _finite(state['components'], 'PCA components', ndim=2)
    score_mean = _finite(state['pca_score_mean'], 'PCA score mean', ndim=1)
    sd = _finite(state['pca_score_sd'], 'PCA score SD', ndim=1)
    if (mean.shape != (INPUT_DIM,) or components.shape != (RANK, INPUT_DIM)
            or score_mean.shape != (RANK,) or sd.shape != (RANK,) or (sd <= 0).any()):
        raise ValueError('projection parameter shape or scale mismatch')
    transformed = (((features - mean) @ components.T) - score_mean) / sd
    if not np.isfinite(transformed).all():
        raise ValueError('nonfinite projected feature')
    return transformed


def _supervision(time, event, n):
    time, event = _finite(time, 'fit duration', ndim=1), _finite(event, 'fit indicator', ndim=1)
    if time.shape != (n,) or event.shape != (n,) or (time < 0).any() or not np.isin(event, (0, 1)).all() or event.sum() < 1:
        raise ValueError('invalid fit survival supervision or no events')
    rows, owners = [], []
    for value in np.unique(time[event == 1]):
        deaths = np.flatnonzero((time == value) & (event == 1))
        riskset = (time >= value).astype(float)
        deathmask = ((time == value) & (event == 1)).astype(float)
        for number, owner in enumerate(deaths):
            rows.append(riskset - number / len(deaths) * deathmask)
            owners.append(owner)
    weights = np.stack(rows)
    with np.errstate(divide='ignore'):
        log_weights = np.log(weights)
    return time, event, log_weights, np.asarray(owners, int)


def efron_nll(time, event, prediction):
    """Stable event-normalized Efron NLL, independent of any fitted head."""
    prediction = _finite(prediction, 'log-risk', ndim=1)
    _, _, log_weights, owners = _supervision(time, event, len(prediction))
    shifted = prediction - np.max(prediction)
    terms = logsumexp(log_weights + shifted, axis=1) - shifted[owners]
    answer = float(terms.mean())
    if not np.isfinite(answer):
        raise ValueError('nonfinite Efron NLL')
    return answer


def offset_objective(features, time, event, base, lam=10.):
    """Return same-objective loss/gradient/optional Hessian; offset coefficient=1."""
    x = _finite(features, 'fit readout features', ndim=2)
    if x.shape[1] != RANK:
        raise ValueError('locked Cox head must have exactly 16 coefficients')
    offset = _finite(base, 'fit main offset', ndim=1)
    if offset.shape != (len(x),) or not np.isscalar(lam) or not np.isfinite(lam) or lam <= 0:
        raise ValueError('offset shape or ridge penalty invalid')
    _, indicator, log_weights, owners = _supervision(time, event, len(x))
    # Remove an irrelevant COMMON constant before adding Xw. This preserves a
    # small learned correction even when every saved offset is a huge constant.
    centered_offset = offset - np.max(offset)
    if not np.isfinite(centered_offset).all():
        raise ValueError('offset range is not representable as finite float64')
    count = float(indicator.sum())
    penalty = float(lam)

    def evaluate(coefficient, hessian=False):
        coefficient = _finite(coefficient, 'readout coefficient', ndim=1)
        if coefficient.shape != (RANK,):
            raise ValueError('coefficient must contain exactly 16 values')
        prediction = centered_offset + x @ coefficient
        scores = log_weights + prediction
        denominator = logsumexp(scores, axis=1)
        probability = np.exp(scores - denominator[:, None])
        cox = float(np.mean(denominator - prediction[owners]))
        loss = cox + .5 * penalty * float(coefficient @ coefficient)
        gradient = x.T @ ((probability.sum(0) - indicator) / count) + penalty * coefficient
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError('nonfinite conditional Cox objective/gradient')
        if not hessian:
            return loss, gradient
        expected = probability @ x
        matrix = (x.T @ (x * probability.sum(0)[:, None]) - expected.T @ expected) / count
        matrix = (matrix + matrix.T) * .5 + penalty * np.eye(RANK)
        if not np.isfinite(matrix).all():
            raise ValueError('nonfinite conditional Cox Hessian')
        return loss, gradient, matrix
    return evaluate


def fit_offset_ridge(features, time, event, base, lam=10.):
    """Single convex Efron fit with optional same-objective Newton polishing."""
    if float(lam) != 10.:
        raise ValueError('Phase27 regularization is locked at lambda10')
    objective = offset_objective(features, time, event, base, lam)
    offset_snapshot = np.array(base, copy=True)
    history = []

    def record(coefficient, phase, iteration):
        loss, gradient = objective(coefficient)
        history.append({'phase': phase, 'iteration': int(iteration), 'objective': float(loss),
                        'cox': float(loss - .5 * lam * np.dot(coefficient, coefficient)),
                        'gradient_norm': float(np.linalg.norm(gradient)), 'gradient_max': float(np.abs(gradient).max())})
    record(np.zeros(RANK), 'initial', 0)
    iteration = 0

    def callback(coefficient):
        nonlocal iteration
        iteration += 1
        record(coefficient, 'L_BFGS_B', iteration)
    result = minimize(objective, np.zeros(RANK), jac=True, method='L-BFGS-B', callback=callback,
                      options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50})
    coefficient = np.array(result.x, np.float64, copy=True)
    loss, gradient, matrix = objective(coefficient, hessian=True)
    polish = 0
    while np.abs(gradient).max() > 1e-7 and polish < 30:
        direction = np.linalg.solve(matrix, gradient)
        accepted, step = False, 1.
        for _ in range(40):
            candidate = coefficient - step * direction
            value, _ = objective(candidate)
            if value <= loss - 1e-4 * step * np.dot(gradient, direction) + 1e-14:
                accepted = True
                break
            step *= .5
        if not accepted:
            raise RuntimeError('same-objective Newton line search failed')
        coefficient = candidate
        loss, gradient, matrix = objective(coefficient, hessian=True)
        polish += 1
        record(coefficient, 'Newton_polish', polish)
    maximum = float(np.abs(gradient).max())
    if not np.isfinite(loss) or maximum > 1e-7:
        raise RuntimeError('conditional Ridge Cox failed the locked 1e-7 gradient gate')
    if not np.array_equal(np.asarray(base), offset_snapshot):
        raise RuntimeError('caller offset changed during convex optimization')
    record(coefficient, 'final', iteration + polish)
    return coefficient, {'status': 'CONVERGED', 'parameters': RANK, 'penalty': float(lam),
        'objective': float(loss), 'cox': float(loss - .5 * lam * np.dot(coefficient, coefficient)),
        'gradient_max': maximum, 'gradient_norm': float(np.linalg.norm(gradient)),
        'iterations': int(result.nit), 'newton_polish_steps': polish, 'history': history,
        'coefficient_L2': float(np.linalg.norm(coefficient)), 'intercept': False, 'offset_trainable': False,
        'offset_unchanged': True, 'labels_normalization': 'number_of_observed_events',
        'selection': 'none_fixed_objective_no_validation_callback'}


def pair_diagnostic(time, event, prediction):
    """All comparable training pairs, also split by comparison-partner status."""
    prediction = _finite(prediction, 'fit log-risk', ndim=1)
    time, event, _, _ = _supervision(time, event, len(prediction))
    first, second = np.where((event[:, None] == 1) & ((time[:, None] < time[None, :])
        | ((time[:, None] == time[None, :]) & (event[None, :] == 0))))
    margin = prediction[first] - prediction[second]
    if not np.isfinite(margin).all():
        raise ValueError('pair margin is not finite')
    credit = np.where(np.abs(margin) <= 1e-8, .5, margin > 0)
    logistic = np.logaddexp(0., -margin)
    answer = {}
    for name, keep in (('ALL', np.ones(len(first), bool)), ('EVENT_EVENT', event[second] == 1),
                       ('EVENT_CENSOR', event[second] == 0)):
        count = int(keep.sum())
        answer[name] = {'count': count, 'credit_num': float(credit[keep].sum()), 'credit_den': count,
            'Harrell': float(credit[keep].mean()) if count else None,
            'logistic_loss': float(logistic[keep].mean()) if count else None, 'defined': bool(count)}
    if (answer['EVENT_EVENT']['count'] + answer['EVENT_CENSOR']['count'] != answer['ALL']['count']
            or abs(answer['EVENT_EVENT']['credit_num'] + answer['EVENT_CENSOR']['credit_num']
                   - answer['ALL']['credit_num']) > 1e-12):
        raise RuntimeError('event pair partition does not close')
    answer['scope'] = 'descriptive_comparable_pairs_not_a_new_training_objective'
    answer['risk_tie_tolerance'] = 1e-8
    return answer

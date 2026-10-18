"""Small independent numerical oracles; no cohort or representation files."""
import numpy as np
import pytest

from src.phase27_readout.math import (efron_nll, fit_offset_ridge, fit_projection,
                                      offset_objective, pair_diagnostic, transform_projection)


def literal_efron(time, event, prediction):
    total = 0.
    for value in sorted(set(time[event == 1])):
        deaths = np.flatnonzero((time == value) & (event == 1))
        at_risk = np.flatnonzero(time >= value)
        shift = max(prediction[at_risk])
        for fraction in np.arange(len(deaths)) / len(deaths):
            denominator = sum(np.exp(prediction[j]-shift) * (1-fraction if j in deaths else 1) for j in at_risk)
            total += shift + np.log(denominator)
        total -= sum(prediction[deaths])
    return total / event.sum()


def toy():
    generator = np.random.default_rng(271)
    x = generator.normal(size=(24, 16))
    time = np.repeat(np.arange(1., 9.), 3)
    event = np.tile([1., 1., 0.], 8)
    base = generator.normal(size=len(x))
    coefficient = generator.normal(scale=.05, size=16)
    return x, time, event, base, coefficient


def test_raw_PCA_matches_independent_SVD_and_fit_PC_covariance_is_identity():
    rng = np.random.default_rng(2701)
    features = rng.normal(size=(40, 256)) * np.linspace(.1, 5, 256)
    state = fit_projection(features)
    transformed = transform_projection(features, state)
    _, singular, right = np.linalg.svd(features-features.mean(0), full_matrices=False)
    projector = right[:16].T @ right[:16]
    assert np.allclose(state['components'].T @ state['components'], projector, atol=1e-12)
    assert np.max(np.abs(transformed.mean(0))) < 1e-12
    assert np.allclose(transformed.T @ transformed / len(features), np.eye(16), atol=1e-12)
    assert state['input_fit_rank'] == 39
    assert not state['input_standardized_before_PCA'] and not state['whiten']


def test_control_transform_uses_same_saved_projection_without_refitting():
    rng = np.random.default_rng(2702)
    fit = rng.normal(size=(35, 256)); control = fit + 20
    state = fit_projection(fit)
    snapshot = state['components'].copy()
    actual = transform_projection(control, state)
    expected = (((control-state['pca_mean']) @ state['components'].T)-state['pca_score_mean']) / state['pca_score_sd']
    assert np.array_equal(actual, expected) and np.array_equal(state['components'], snapshot)
    with pytest.raises(ValueError, match='rank is below'):
        fit_projection(np.ones((35, 256)))


def test_offset_Efron_and_penalty_match_literal_tied_event_oracle():
    x, time, event, base, coefficient = toy()
    objective = offset_objective(x, time, event, base)
    value, gradient = objective(coefficient)
    expected = literal_efron(time, event, base+x@coefficient) + 5*np.dot(coefficient, coefficient)
    assert value == pytest.approx(expected, abs=1e-12)
    assert efron_nll(time, event, base+x@coefficient) == pytest.approx(expected-5*np.dot(coefficient, coefficient), abs=1e-12)


def test_independent_finite_difference_gradient_Hessian_and_strict_convexity():
    x, time, event, base, coefficient = toy()
    objective = offset_objective(x, time, event, base)
    _, gradient, hessian = objective(coefficient, hessian=True)
    step = 1e-5; direction = np.eye(16)
    finite_gradient = np.array([(objective(coefficient+step*d)[0]-objective(coefficient-step*d)[0])/(2*step) for d in direction])
    finite_hessian = np.column_stack([(objective(coefficient+step*d)[1]-objective(coefficient-step*d)[1])/(2*step) for d in direction])
    assert np.allclose(gradient, finite_gradient, atol=1e-8)
    assert np.allclose(hessian, finite_hessian, atol=1e-8)
    assert np.linalg.eigvalsh(hessian).min() >= 10-1e-10


def test_common_huge_offset_shift_is_invariant_and_extreme_offsets_are_finite():
    x, time, event, _, coefficient = toy()
    first = offset_objective(x, time, event, np.zeros(len(x)))
    shifted = offset_objective(x, time, event, np.full(len(x), 1e150))
    assert first(coefficient)[0] == shifted(coefficient)[0]
    assert np.array_equal(first(coefficient)[1], shifted(coefficient)[1])
    extreme = offset_objective(x, time, event, np.linspace(-1e6, 1e6, len(x)))
    value, gradient, hessian = extreme(coefficient, hessian=True)
    assert np.isfinite(value) and np.isfinite(gradient).all() and np.isfinite(hessian).all()


def test_fixed_ridge_solver_converges_preserves_offset_and_records_fit_only_history():
    x, time, event, base, _ = toy(); snapshot = base.copy()
    coefficient, audit = fit_offset_ridge(x, time, event, base)
    assert audit['status'] == 'CONVERGED' and audit['gradient_max'] <= 1e-7
    assert audit['parameters'] == 16 and not audit['intercept'] and not audit['offset_trainable']
    assert np.array_equal(base, snapshot)
    assert audit['history'][0]['phase'] == 'initial' and audit['history'][-1]['phase'] == 'final'
    assert audit['objective'] <= audit['history'][0]['objective']
    assert all(set(row) == {'phase','iteration','objective','cox','gradient_norm','gradient_max'} for row in audit['history'])
    assert coefficient.shape == (16,)


def test_pair_partition_event_event_event_censor_and_equal_time_rules():
    time = np.array([1., 1., 1., 2.]); event = np.array([1., 1., 0., 1.]); prediction = np.array([3., 2., 1., 0.])
    result = pair_diagnostic(time, event, prediction)
    assert result['ALL']['count'] == 4
    assert result['EVENT_EVENT']['count'] == 2
    assert result['EVENT_CENSOR']['count'] == 2
    assert result['ALL']['Harrell'] == 1
    assert result['EVENT_EVENT']['credit_num'] + result['EVENT_CENSOR']['credit_num'] == result['ALL']['credit_num']
    weighted = sum(result[key]['logistic_loss']*result[key]['count'] for key in ('EVENT_EVENT','EVENT_CENSOR')) / result['ALL']['count']
    assert weighted == pytest.approx(result['ALL']['logistic_loss'])
    ties = pair_diagnostic([1., 2.], [1., 0.], [0., 1e-9])
    assert ties['ALL']['Harrell'] == .5
    assert not ties['EVENT_EVENT']['defined'] and ties['EVENT_EVENT']['logistic_loss'] is None


@pytest.mark.parametrize('invalid', ['width', 'no_event', 'nonfinite', 'lambda'])
def test_invalid_inputs_fail_closed_without_alternate_fit(invalid):
    x, time, event, base, _ = toy()
    if invalid == 'width': x = x[:, :15]
    if invalid == 'no_event': event[:] = 0
    if invalid == 'nonfinite': base[0] = np.nan
    with pytest.raises(ValueError):
        fit_offset_ridge(x, time, event, base, lam=1 if invalid == 'lambda' else 10)

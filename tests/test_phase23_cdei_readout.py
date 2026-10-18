import numpy as np
import pytest
import torch

from src.phase17_ssd.losses import full_risk_efron
from src.phase23_cdei.readout import (FitOnlyStandardizer, conditional_objective,
                                     fit_conditional_cox, risk_with_offset)


def case():
    rng = np.random.default_rng(230018)
    return (rng.normal(size=(12, 4)), np.array([1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8.]),
            np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]), rng.normal(size=12))


def test_objective_gradient_and_hessian_against_independent_torch_efron():
    x, t, e, off = case(); w = np.array([.2, -.1, .4, .3]); lam = 2.
    actual, grad, hessian = conditional_objective(x, t, e, off, lam)(w, hessian=True)
    xx, oo, ww = torch.tensor(x), torch.tensor(off), torch.tensor(w, requires_grad=True)
    objective = lambda q: full_risk_efron(oo + xx @ q, t, e) + .5 * lam * q.square().sum()
    expected = objective(ww)
    eg = torch.autograd.grad(expected, ww)[0]
    eh = torch.autograd.functional.hessian(objective, ww)
    assert actual == pytest.approx(expected.item(), abs=1e-12)
    np.testing.assert_allclose(grad, eg.detach().numpy(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(hessian, eh.numpy(), atol=1e-12, rtol=0)


def test_convex_fit_converges_and_preserves_exact_offset():
    x, t, e, off = case(); copied = off.copy()
    w, report = fit_conditional_cox(x, t, e, off)
    assert report['status'] == 'CONVERGED' and report['gradient_max'] <= 1e-7
    np.testing.assert_array_equal(off, copied)
    risk, delta = risk_with_offset(x, w, off)
    np.testing.assert_array_equal(risk, off + delta)
    assert conditional_objective(x, t, e, off)(w)[0] <= conditional_objective(x, t, e, off)(np.zeros(4))[0]


def test_constant_shift_of_all_offsets_does_not_change_fitted_correction():
    x, t, e, off = case()
    w, _ = fit_conditional_cox(x, t, e, off)
    shifted, _ = fit_conditional_cox(x, t, e, off + 1000.)
    np.testing.assert_allclose(w, shifted, atol=1e-7, rtol=0)


def test_fit_only_scaling_and_zero_effect_when_head_off():
    x, _, _, off = case(); x[:, 0] = 3.
    fitted = FitOnlyStandardizer().fit(x[:8]); before = fitted.public_summary()
    heldout = x[8:] * 1e6
    _ = fitted.transform(heldout)
    assert before == fitted.public_summary() and before['constant_fit_columns'] == 1
    np.testing.assert_array_equal(fitted.transform(x[:8])[:, 0], 0.)
    risk, delta = risk_with_offset(x, np.zeros(4), off)
    np.testing.assert_array_equal(risk, off); np.testing.assert_array_equal(delta, 0.)


@pytest.mark.parametrize('field', ['no_event', 'negative_time', 'nan_x', 'bad_offset', 'nonbinary'])
def test_bad_endpoints_or_features_fail_closed(field):
    x, t, e, off = case()
    if field == 'no_event': e[:] = 0
    if field == 'negative_time': t[0] = -1
    if field == 'nan_x': x[0, 0] = np.nan
    if field == 'bad_offset': off = off[:-1]
    if field == 'nonbinary': e[0] = 2
    with pytest.raises(ValueError): conditional_objective(x, t, e, off)


@pytest.mark.parametrize('penalty', [0, -1, float('nan'), float('inf')])
def test_invalid_penalty(penalty):
    with pytest.raises(ValueError): conditional_objective(*case(), penalty)

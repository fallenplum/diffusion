"""Independent synthetic Efron and full-risk gradient-cache qualification."""
import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from src.phase28_reread.training_math import efron_loss, risk_gradient_cache


def scalar_efron(risk, time, event):
    """Independent scalar tie loop; no production weight-matrix helper."""
    risk, time, event = map(lambda x: np.asarray(x, dtype=float), (risk, time, event))
    value = 0.
    for instant in sorted(set(time[event == 1])):
        deaths = np.flatnonzero((time == instant) & (event == 1))
        at_risk = np.flatnonzero(time >= instant)
        denominator = sum(math.exp(float(risk[i])) for i in at_risk)
        tied = sum(math.exp(float(risk[i])) for i in deaths)
        value -= sum(float(risk[i]) for i in deaths)
        for step in range(len(deaths)):
            value += math.log(denominator-step*tied/len(deaths))
    return value/event.sum()


def test_full_efron_matches_independent_tied_scalar_and_finite_difference():
    risk = torch.tensor([.4, -.2, .1, .7, -.3, .2], dtype=torch.float64, requires_grad=True)
    time = np.array([2., 2., 2., 4., 5., 5.])
    event = np.array([1., 1., 0., 1., 1., 0.])
    loss = efron_loss(risk, time, event)
    assert float(loss) == pytest.approx(scalar_efron(risk.detach(), time, event), abs=1e-13)
    gradient, = torch.autograd.grad(loss, risk)
    finite = []
    for position in range(len(risk)):
        plus = risk.detach().numpy().copy(); minus = plus.copy()
        plus[position] += 1e-6; minus[position] -= 1e-6
        finite.append((scalar_efron(plus, time, event)-scalar_efron(minus, time, event))/2e-6)
    np.testing.assert_allclose(gradient.numpy(), finite, atol=3e-10, rtol=1e-8)


def test_efron_is_shift_and_joint_order_invariant_and_allows_zero_time():
    r = torch.tensor([.1, -.3, .6, -.2], dtype=torch.float64, requires_grad=True)
    t, e = np.array([0., 0., 2., 3.]), np.array([1., 0., 1., 0.])
    order = np.array([2, 0, 3, 1])
    a = efron_loss(r, t, e)
    torch.testing.assert_close(efron_loss(r+123., t, e), a, atol=3e-14, rtol=0)
    torch.testing.assert_close(efron_loss(r[order], t[order], e[order]), a, atol=1e-14, rtol=0)
    assert abs(float(torch.autograd.grad(a, r)[0].sum())) < 1e-14


def make_example():
    torch.manual_seed(281003)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 1, bias=False)).double()
    x = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    indices = np.array([8, 2, 9, 1, 7, 5, 3], dtype=np.int64)
    time = np.array([1., 1., 2., 3., 3., 4., 5.])
    event = np.array([1., 1., 0., 1., 0., 1., 0.])
    offset = torch.linspace(-.3, .5, len(x), dtype=torch.float64)+.37
    return model, x, indices, time, event, offset


@pytest.mark.parametrize("microbatch", [1, 2, 3, 7, 20])
@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "mapping"])
def test_cache_matches_full_parameters_inputs_offset_and_one_regularizer(microbatch, output_kind):
    full, x_full, indices, time, event, offset = make_example()
    cached = copy.deepcopy(full)
    x_cached = x_full.detach().clone().requires_grad_(True)
    # Existing gradients must be accumulated, not erased by the utility.
    for model in (full, cached):
        for p in model.parameters():
            p.grad = torch.full_like(p, .031)
    x_full.grad = torch.full_like(x_full, -.02)
    x_cached.grad = torch.full_like(x_cached, -.02)
    original_parameters = [p.detach().clone() for p in cached.parameters()]
    main = full(x_full[indices]).squeeze(-1)
    risk = offset[indices]+main
    penalty = .07*sum(p.square().sum() for p in full.parameters())
    objective = efron_loss(risk, time, event)+penalty
    objective.backward()
    calls = []; regularizer_calls = []

    def forward(current):
        calls.append(tuple(current.tolist()))
        m = cached(x_cached[current]).squeeze(-1)
        total = offset[current]+m
        if output_kind == "tensor": return total
        if output_kind == "tuple": return total, m, m*0.
        return {"risk": total, "main_residual": m, "interaction": m*0.}

    def regularizer():
        regularizer_calls.append(1)
        return .07*sum(p.square().sum() for p in cached.parameters())

    audit = risk_gradient_cache(cached, forward, indices, time, event, microbatch, regularizer)
    for left, right, before in zip(full.parameters(), cached.parameters(), original_parameters):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-13, rtol=2e-12)
        torch.testing.assert_close(right, before, atol=0, rtol=0)
    torch.testing.assert_close(x_full.grad, x_cached.grad, atol=2e-13, rtol=2e-12)
    assert audit["objective"] == pytest.approx(float(objective.detach()), abs=2e-13)
    assert audit["regularizer"] == pytest.approx(float(penalty.detach()), abs=2e-13)
    assert audit["regularizer_calls"] == len(regularizer_calls) == 1
    assert audit["optimizer_steps"] == 0
    assert audit["patients"] == len(indices) and audit["events"] == 4
    assert audit["maximum_absolute_risk_replay_error"] == 0.
    chunks = math.ceil(len(indices)/microbatch)
    assert audit["forward_calls"] == len(calls) == chunks*2
    assert calls[:chunks] == calls[chunks:]
    assert set(i for chunk in calls for i in chunk) == set(indices)
    assert abs(audit["risk_gradient_sum"]) < 2e-15
    assert all(not isinstance(value, (list, np.ndarray, torch.Tensor)) for value in audit.values())
    if output_kind != "tensor": assert audit["interaction_RMS"] == 0.


def test_no_event_fails_before_forward_or_regularizer():
    model, x, indices, time, event, offset = make_example()
    calls = []
    def forbidden(*_): calls.append(1); raise AssertionError("must not be called")
    with pytest.raises(ValueError, match="at least one fit event"):
        risk_gradient_cache(model, forbidden, indices, time, np.zeros_like(event), regularizer=forbidden)
    assert calls == []


@pytest.mark.parametrize("module", [nn.Dropout(.2), nn.AlphaDropout(.1), nn.MultiheadAttention(4, 1, dropout=.2)])
def test_training_dropout_rejected_before_callback(module):
    module.train(); calls = []
    with pytest.raises(ValueError, match="active dropout"):
        risk_gradient_cache(module, lambda _: calls.append(1), [0], [1.], [1.])
    assert calls == []


@pytest.mark.parametrize("training", [False, True])
def test_batchnorm_rejected_even_in_eval(training):
    module = nn.BatchNorm1d(3); module.train(training)
    with pytest.raises(ValueError, match="BatchNorm"):
        risk_gradient_cache(module, lambda _: None, [0], [1.], [1.])


@pytest.mark.parametrize("probability,training", [(0., True), (.4, False)])
def test_zero_dropout_or_disabled_eval_dropout_is_deterministic(probability, training):
    model, x, indices, time, event, offset = make_example()
    model = nn.Sequential(nn.Dropout(probability), model); model.train(training)
    audit = risk_gradient_cache(model, lambda i: model(x[i]).squeeze(-1)+offset[i], indices, time, event, 2)
    assert audit["maximum_absolute_risk_replay_error"] == 0.
    assert audit["regularizer_calls"] == 0


def test_deliberately_nondeterministic_callback_fails_replay():
    model, x, indices, time, event, _ = make_example(); calls = []
    def forward(current):
        calls.append(1)
        return model(x[current]).squeeze(-1)+len(calls)*.01
    with pytest.raises(ValueError, match="nondeterministic gradient replay"):
        risk_gradient_cache(model, forward, indices, time, event, 2)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("indices", [[], [1, 1], [True], [-1], [1.5]])
def test_invalid_or_repeated_patient_indices_fail_before_forward(indices):
    model = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="patient indices"):
        risk_gradient_cache(model, lambda _: None, indices, [1.]*len(indices), [1.]*len(indices))


@pytest.mark.parametrize("time,event", [([1., float("nan")], [1., 0.]),
                                        ([-1., 2.], [1., 0.]),
                                        ([1., 2.], [1., .5]),
                                        ([1.], [1., 0.])])
def test_invalid_fit_endpoint_contract(time, event):
    with pytest.raises(ValueError, match="fit endpoint|event indicators"):
        efron_loss(torch.zeros(2, dtype=torch.float64), time, event)


def test_nonfinite_prediction_fails_before_backward():
    model = nn.Linear(1, 1).double()
    with pytest.raises(ValueError, match="invalid forward risk"):
        risk_gradient_cache(model, lambda _: torch.tensor([float("nan")], dtype=torch.float64),
                            [0], [1.], [1.])
    assert all(p.grad is None for p in model.parameters())


def test_disabled_autograd_is_rejected():
    with torch.no_grad(), pytest.raises(ValueError, match="enabled autograd"):
        risk_gradient_cache(nn.Linear(1, 1), lambda _: None, [0], [1.], [1.])


def test_data_gradient_audit_does_not_mislabel_regularizer_only_branch():
    class TwoBranches(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Linear(2, 1, bias=False)
            self.dormant = nn.Linear(2, 1, bias=False)

    torch.manual_seed(280041)
    model = TwoBranches().double()
    x = torch.tensor([[1., .4], [-.2, .9], [.7, -.5]], dtype=torch.float64)
    audit = risk_gradient_cache(
        model, lambda i: model.main(x[i]).squeeze(-1), [0, 1, 2], [1., 2., 3.], [1., 0., 1.],
        regularizer=lambda: .1*sum(p.square().sum() for p in model.parameters()))
    assert audit['data_gradient_L2_by_top_level_module']['main'] > 0
    assert audit['data_gradient_L2_by_top_level_module']['dormant'] == 0
    assert audit['total_gradient_L2_by_top_level_module']['dormant'] > 0
    assert audit['parameters_with_nonzero_data_gradient'] == 2
    assert audit['parameters_with_nonzero_total_gradient'] == 4
    assert audit['parameter_tensors_with_nonzero_data_gradient'] == 1
    assert audit['parameter_tensors_with_nonzero_total_gradient'] == 2

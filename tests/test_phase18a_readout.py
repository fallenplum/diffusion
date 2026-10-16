"""Meaningful mathematical checks for the Phase18A readout-only change."""

import copy

import pytest
import torch
import torch.nn.functional as F

from src.phase17_ssd.ssm import cuda_final_scan, reference_scan
from src.phase18a_readout.scan import (
    explicit_local_readout, local_readout_scan, operator_parity,
)


def _inputs(length=13, dtype=torch.float64, device="cpu"):
    torch.manual_seed(1818)
    u = torch.randn(2, length, 4, dtype=dtype, device=device)
    m = torch.sigmoid(torch.randn(2, length, 1, dtype=dtype, device=device))
    dt = F.softplus(torch.randn_like(u)) * m
    B = torch.randn(2, length, 3, dtype=dtype, device=device)
    C = torch.randn_like(B)
    A = -torch.exp(torch.randn(4, 3, dtype=dtype, device=device))
    h0 = torch.randn(2, 4, 3, dtype=dtype, device=device)
    return u, dt, B, C, A, m, h0


def test_cpu_double_all_inputs_and_boundaries():
    assert operator_parity("cpu", torch.float64)["passed"]


def test_cpu_float_all_inputs_and_boundaries():
    assert operator_parity("cpu", torch.float32)["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_all_inputs_and_boundaries():
    assert operator_parity("cuda", torch.float64)["passed"]
    assert operator_parity("cuda", torch.float32)["passed"]


def test_n1_is_parent_final_readout():
    u, dt, B, C, A, m, h0 = _inputs(length=1)
    actual, h = local_readout_scan(u, dt, B, C, A, m, h0=h0, return_final=True)
    h_final = cuda_final_scan(u, dt, B, A, h0)
    c_pool = (C * m).sum(1) / m.sum(1).clamp_min(1e-8)
    torch.testing.assert_close(actual, (h_final * c_pool[:, None, :]).sum(-1))
    torch.testing.assert_close(h, h_final)


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_chunk_size_changes_neither_states_nor_gradients(checkpoint_chunks):
    original = _inputs(length=37)
    output, gradients = [], []
    for chunk in [1, 7, 16, 32, 64, 128]:
        args = [x.detach().clone().requires_grad_() for x in original]
        y, h = local_readout_scan(*args[:6], h0=args[6], chunk_size=chunk,
                                 checkpoint_chunks=checkpoint_chunks, return_final=True)
        output.append((y, h))
        gradients.append(torch.autograd.grad(y.square().sum() + h.square().sum(), args))
    for out, grad in zip(output[1:], gradients[1:]):
        for x, y in zip(out, output[0]):
            torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)
        for x, y in zip(grad, gradients[0]):
            torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-10)


def test_zero_write_gate_preserves_nonzero_state_and_empty_readout():
    u, dt, B, C, A, m, h0 = _inputs(length=9)
    m = torch.zeros_like(m)
    dt = torch.zeros_like(dt)
    y, h = local_readout_scan(u, dt, B, C, A, m, h0=h0, return_final=True)
    torch.testing.assert_close(h, h0, atol=0, rtol=0)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)
    assert torch.isfinite(y).all()


def test_padding_does_not_add_states_or_weight_to_readout():
    u, dt, B, C, A, m, h0 = _inputs(length=7)
    expected = local_readout_scan(u, dt, B, C, A, m, h0=h0, return_final=True)
    # Nonzero finite padded u/B/C are allowed: dt=0 keeps the state and m=0
    # excludes C_t h_t.  No extra input sanitation or model change is inserted.
    pad = 6
    up = torch.cat([u, torch.full((2, pad, 4), 100., dtype=u.dtype)], 1)
    dp = torch.cat([dt, torch.zeros(2, pad, 4, dtype=u.dtype)], 1)
    bp = torch.cat([B, torch.full((2, pad, 3), -300., dtype=u.dtype)], 1)
    cp = torch.cat([C, torch.full((2, pad, 3), 400., dtype=u.dtype)], 1)
    mp = torch.cat([m, torch.zeros(2, pad, 1, dtype=u.dtype)], 1)
    actual = local_readout_scan(up, dp, bp, cp, A, mp, h0=h0, chunk_size=4, return_final=True)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-11, rtol=1e-11)
        assert torch.isfinite(x).all()


def test_gate_weight_applied_once_in_transition():
    u, dt, B, C, A, m, h0 = _inputs(length=13)
    actual = local_readout_scan(u, dt, B, C, A, m, h0=h0)
    expected, _ = explicit_local_readout(u, dt, B, C, A, m, h0=h0)
    incorrect, _ = explicit_local_readout(u, dt * m, B, C, A, m, h0=h0)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert not torch.allclose(actual, incorrect, atol=1e-5, rtol=1e-5)


class _ParameterProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.u = torch.nn.Linear(5, 4)
        self.dt = torch.nn.Linear(5, 4)
        self.B = torch.nn.Linear(5, 3)
        self.C = torch.nn.Linear(5, 3)
        self.write = torch.nn.Linear(5, 1)
        self.log_A = torch.nn.Parameter(torch.zeros(4, 3))
        self.readout = torch.nn.Linear(4, 1, bias=False)

    def forward(self, x, mask, h0, oracle=False):
        m = torch.sigmoid(self.write(x)) * mask[:, :, None]
        args = (self.u(x), m * F.softplus(self.dt(x)), self.B(x), self.C(x), -self.log_A.exp(), m)
        if oracle:
            y, h = explicit_local_readout(*args, h0=h0)
        else:
            y, h = local_readout_scan(*args, h0=h0, chunk_size=5, return_final=True)
        return self.readout(y).squeeze(-1), h


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"))])
def test_trainable_parameter_and_upstream_input_gradients(device):
    torch.manual_seed(155)
    model = _ParameterProbe().to(device=device, dtype=torch.float64)
    oracle = copy.deepcopy(model)
    x = torch.randn(2, 17, 5, device=device, dtype=torch.float64, requires_grad=True)
    xo = x.detach().clone().requires_grad_()
    h0 = torch.randn(2, 4, 3, device=device, dtype=torch.float64, requires_grad=True)
    ho = h0.detach().clone().requires_grad_()
    mask = torch.ones(2, 17, dtype=torch.bool, device=device)
    mask[0, 11:] = False
    risk, h = model(x, mask, h0)
    risk_o, h_o = oracle(xo, mask, ho, oracle=True)
    (risk.square().sum() + .03 * h.square().sum()).backward()
    (risk_o.square().sum() + .03 * h_o.square().sum()).backward()
    torch.testing.assert_close(risk, risk_o, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(x.grad, xo.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(h0.grad, ho.grad, atol=1e-10, rtol=1e-10)
    for (name, p), (other, q) in zip(model.named_parameters(), oracle.named_parameters()):
        assert name == other
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-10, msg=name)


def test_sequence_order_is_not_silently_pooled_before_recurrence():
    u, dt, B, C, A, m, h0 = _inputs(length=13)
    actual = local_readout_scan(u, dt, B, C, A, m, h0=h0)
    reversed_output = local_readout_scan(u.flip(1), dt.flip(1), B.flip(1), C.flip(1), A,
                                         m.flip(1), h0=h0)
    assert not torch.allclose(actual, reversed_output)


def test_local_pairs_each_C_with_its_own_state():
    u, dt, B, C, A, m, h0 = _inputs(length=13)
    actual = local_readout_scan(u, dt, B, C, A, m, h0=h0)
    hs = reference_scan(u, dt, B, A, h0=h0, states=True)
    explicit = torch.zeros_like(actual)
    for t in range(u.shape[1]):
        explicit += (C[:, t, None, :] * hs[:, t]).sum(-1) * m[:, t]
    explicit /= m.sum(1).clamp_min(1e-8)
    torch.testing.assert_close(actual, explicit, atol=1e-11, rtol=1e-11)
    final = (hs[:, -1] * ((C * m).sum(1) / m.sum(1))[:, None, :]).sum(-1)
    assert not torch.allclose(actual, final)


def test_nested_checkpoint_preserves_dropout_rng_and_full_sequence_gradients():
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(775)
    model = _ParameterProbe().double()
    oracle = copy.deepcopy(model)
    x = torch.randn(2, 17, 5, dtype=torch.float64, requires_grad=True)
    xo = x.detach().clone().requires_grad_()
    h0 = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    ho = h0.detach().clone().requires_grad_()
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[0, 11:] = False
    # Parent Data.forward uses an outer nonreentrant checkpoint.  The local
    # scan's internal checkpoints must remain exact under that same nesting.
    def call(xx):
        return model(F.dropout(xx, p=.25, training=True), mask, h0)[0]
    torch.manual_seed(923)
    risk = checkpoint(call, x, use_reentrant=False)
    risk.square().sum().backward()
    torch.manual_seed(923)
    risk_o = oracle(F.dropout(xo, p=.25, training=True), mask, ho, oracle=True)[0]
    risk_o.square().sum().backward()
    torch.testing.assert_close(risk, risk_o, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(x.grad, xo.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(h0.grad, ho.grad, atol=1e-10, rtol=1e-10)
    for (name, p), (_, q) in zip(model.named_parameters(), oracle.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-10, msg=name)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.2])
def test_invalid_chunk_size_fails_closed(chunk_size):
    args = _inputs(length=1)
    with pytest.raises(ValueError):
        local_readout_scan(*args[:6], h0=args[6], chunk_size=chunk_size)

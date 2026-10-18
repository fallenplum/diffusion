"""Independent arithmetic qualification; no fitting, outcomes or saved arrays."""

import math

import numpy as np
import pytest
import torch

from src.phase23_cdei.operators import core, dual_evidence_core, masked_mean_std_pool


def test_alias_and_original_score_attention_not_residual_attention():
    scores = torch.tensor([[0., 1., 2.]], dtype=torch.float64)
    null = torch.tensor([[0., 3., -2.]], dtype=torch.float64)
    result = core(scores, null, torch.ones_like(scores, dtype=torch.bool))
    assert core is dual_evidence_core
    torch.testing.assert_close(result["po"], torch.softmax(scores, -1), rtol=1e-14, atol=1e-14)
    assert not torch.allclose(result["po"], torch.softmax(scores - null, -1))


def test_lme_matches_independent_direct_numpy_and_decomposition():
    scores = torch.tensor([[[.4, -.7, 1.2], [2., 4., 1.]]], dtype=torch.float64)
    null = torch.tensor([[[.1, .2, -.3], [.5, -1., .7]]], dtype=torch.float64)
    tau = .7
    result = core(scores, null, torch.ones_like(scores, dtype=torch.bool), tau)
    residual = scores.numpy() - null.numpy()
    expected = tau * np.log(np.exp(residual / tau).mean(-1))
    np.testing.assert_allclose(result["evidence"].numpy(), expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(result["evidence"], result["residual_mean"] + result["centered_lme"],
                               rtol=1e-14, atol=1e-14)
    assert bool((result["decomposition_error"] <= result["decomposition_tolerance"]).all())


def test_row_shift_keeps_relative_changes_evidence_and_mean():
    scores = torch.tensor([[1., -.3, .4], [-.2, .1, 1.]], dtype=torch.float64)
    null = torch.tensor([[.2, .2, -.1], [-.1, .5, .2]], dtype=torch.float64)
    valid = torch.ones_like(scores, dtype=torch.bool)
    shift = torch.tensor([[2.], [-3.]], dtype=torch.float64)
    before = core(scores, null, valid)
    after = core(scores + shift, null, valid)
    torch.testing.assert_close(after["po"], before["po"], rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(after["evidence"], before["evidence"] + shift.squeeze(-1))
    torch.testing.assert_close(after["residual_mean"], before["residual_mean"] + shift.squeeze(-1))
    torch.testing.assert_close(after["centered_lme"], before["centered_lme"])


def test_joint_score_null_shift_cancels_for_absolute_evidence():
    scores = torch.tensor([[.1, .7, -.2]], dtype=torch.float64)
    null = torch.tensor([[.5, .2, -.1]], dtype=torch.float64)
    shift = torch.tensor([[.7, -1., 3.]], dtype=torch.float64)
    mask = torch.ones_like(scores, dtype=torch.bool)
    before, after = core(scores, null, mask), core(scores + shift, null + shift, mask)
    for key in ("evidence", "residual_mean", "centered_lme"):
        torch.testing.assert_close(after[key], before[key])
    assert not torch.allclose(after["po"], before["po"])


def test_scaled_scores_change_relative_attention():
    scores = torch.tensor([[-1., 0., 1.]], dtype=torch.float64)
    mask = torch.ones_like(scores, dtype=torch.bool)
    before = core(scores, torch.zeros_like(scores), mask)
    after = core(scores * 3, torch.zeros_like(scores), mask)
    assert not torch.allclose(after["po"], before["po"])
    assert after["po"][0, -1] > before["po"][0, -1]


def test_bidirectional_row_shift_counterexample():
    before = torch.zeros((2, 2), dtype=torch.float64)
    after = before + torch.tensor([[0.], [1.]], dtype=torch.float64)
    valid = torch.ones_like(before, dtype=torch.bool)
    torch.testing.assert_close(core(before, before, valid)["po"], core(after, before, valid)["po"])
    reverse_before = core(before.T, before.T, valid.T)["po"]
    reverse_after = core(after.T, before.T, valid.T)["po"]
    assert not torch.allclose(reverse_before, reverse_after)
    torch.testing.assert_close(reverse_after[0], torch.tensor([1 / (1 + math.e), math.e / (1 + math.e)],
                                                             dtype=torch.float64))


def test_equal_mean_dispersion_is_not_public_level_recovery():
    scores = torch.tensor([[0., 0.], [-1., 1.]], dtype=torch.float64)
    result = core(scores, torch.zeros_like(scores), torch.ones_like(scores, dtype=torch.bool))
    torch.testing.assert_close(result["residual_mean"], torch.zeros(2, dtype=torch.float64))
    assert result["evidence"][0].item() == 0
    assert result["evidence"][1].item() == pytest.approx(math.log(math.cosh(1.)))
    torch.testing.assert_close(result["evidence"], result["centered_lme"])


def test_uniformly_replicated_keys_keep_mean_evidence_not_raw_key_count():
    scores = torch.tensor([[.1, .8, -.6]], dtype=torch.float64)
    null = torch.tensor([[.3, .1, -.2]], dtype=torch.float64)
    valid = torch.tensor([[True, False, True]])
    original = core(scores, null, valid)
    replicated = core(scores.repeat(1, 2), null.repeat(1, 2), valid.repeat(1, 2))
    for key in ("evidence", "residual_mean", "centered_lme"):
        torch.testing.assert_close(original[key], replicated[key])
    assert original["valid_count"].item() == 2
    assert replicated["valid_count"].item() == 4


def test_nonuniform_replication_can_change_empirical_distribution():
    scores = torch.tensor([[0., 1.]], dtype=torch.float64)
    result = core(scores, torch.zeros_like(scores), torch.ones_like(scores, dtype=torch.bool))
    repeated = torch.tensor([[0., 1., 1.]], dtype=torch.float64)
    other = core(repeated, torch.zeros_like(repeated), torch.ones_like(repeated, dtype=torch.bool))
    assert other["evidence"].item() > result["evidence"].item()


def test_padding_nan_inf_excluded_and_gradients_zero():
    scores = torch.tensor([[1., float("nan"), -2., float("inf")]], dtype=torch.float64, requires_grad=True)
    null = torch.tensor([[.3, float("inf"), .7, float("nan")]], dtype=torch.float64)
    valid = torch.tensor([[True, False, True, False]])
    result = core(scores, null, valid)
    active = core(scores[:, [0, 2]], null[:, [0, 2]], torch.ones((1, 2), dtype=torch.bool))
    for key in ("evidence", "residual_mean", "centered_lme"):
        torch.testing.assert_close(result[key], active[key])
    assert result["po"][0, [1, 3]].eq(0).all()
    (result["evidence"].sum() + result["po"].square().sum()).backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, [1, 3]].eq(0).all()


def test_all_masked_rows_have_explicit_missing_evidence_and_finite_zero_gradient():
    scores = torch.tensor([[float("nan"), float("inf")]], dtype=torch.float64, requires_grad=True)
    null = torch.full_like(scores, float("nan"), requires_grad=False)
    result = core(scores, null, torch.zeros_like(scores, dtype=torch.bool))
    for key in ("po", "evidence", "residual_mean", "centered_lme"):
        assert result[key].eq(0).all()
    assert not result["has_valid"].item()
    assert result["valid_count"].item() == 0
    (result["evidence"].sum() + result["centered_lme"].sum() + result["po"].sum()).backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.eq(0).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_input_dtypes_and_output_contract(dtype):
    scores = torch.tensor([[.1, -.2, .3]], dtype=dtype, requires_grad=True)
    result = core(scores, torch.zeros_like(scores, requires_grad=False), torch.ones_like(scores, dtype=torch.bool))
    assert result["po"].dtype == dtype
    assert result["evidence"].dtype == torch.float64
    assert result["has_valid"].dtype == torch.bool
    assert result["valid_count"].dtype == torch.int64
    result["evidence"].sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_broadcast_mask_and_valid_count():
    scores = torch.zeros((2, 3, 4), dtype=torch.float64)
    result = core(scores, scores, torch.tensor([True, False, True, False]))
    assert result["valid_count"].shape == (2, 3)
    assert result["valid_count"].eq(2).all()
    assert result["po"][..., [1, 3]].eq(0).all()


def test_large_common_level_is_stable_and_has_finite_gradient():
    scores = torch.tensor([[10000., 10001., 9999.]], dtype=torch.float64, requires_grad=True)
    null = torch.tensor([[1., 1., 1.]], dtype=torch.float64)
    result = core(scores, null, torch.ones_like(scores, dtype=torch.bool))
    expected = 9999. + math.log((1 + math.exp(1) + math.exp(-1)) / 3)
    assert result["evidence"].item() == pytest.approx(expected)
    result["evidence"].backward()
    torch.testing.assert_close(scores.grad, torch.softmax((scores - null).detach(), -1))


def test_promote_before_fp32_subtraction():
    limit = torch.finfo(torch.float32).max
    scores = torch.full((1, 2), limit, dtype=torch.float32)
    null = torch.full((1, 2), -limit, dtype=torch.float32)
    result = core(scores, null, torch.ones_like(scores, dtype=torch.bool))
    assert torch.isfinite(result["evidence"]).all()
    assert result["evidence"].item() == 2 * float(limit)


def test_tiny_residual_variation_is_not_rounded_up_to_peak():
    scores = torch.tensor([[0., 2e-20]], dtype=torch.float64, requires_grad=True)
    result = core(scores, torch.zeros_like(scores), torch.ones_like(scores, dtype=torch.bool))
    # At this scale the second-order term is below representable resolution.
    assert result["evidence"].item() == pytest.approx(1e-20, rel=1e-14, abs=0.)
    result["evidence"].backward()
    torch.testing.assert_close(scores.grad, torch.full_like(scores, .5))


def test_very_large_finite_tau_preserves_mean_limit():
    scores = torch.tensor([[0., 1.]], dtype=torch.float64, requires_grad=True)
    result = core(scores, torch.zeros_like(scores), torch.ones_like(scores, dtype=torch.bool), tau=1e200)
    assert result["evidence"].item() == pytest.approx(.5, abs=1e-14)
    result["evidence"].backward()
    assert torch.isfinite(scores.grad).all()
    torch.testing.assert_close(scores.grad, torch.full_like(scores, .5))


def test_genuinely_unrepresentable_fp64_residual_fails_closed():
    limit = torch.finfo(torch.float64).max
    scores = torch.full((1, 2), limit, dtype=torch.float64)
    null = torch.full((1, 2), -limit, dtype=torch.float64)
    with pytest.raises(ValueError, match="scores minus null"):
        core(scores, null, torch.ones_like(scores, dtype=torch.bool))


def test_unrepresentable_temperature_arithmetic_fails_closed():
    scores = torch.tensor([[0., 1.]], dtype=torch.float64)
    with pytest.raises(ValueError, match="temperature-scaled"):
        core(scores, torch.zeros_like(scores), torch.ones_like(scores, dtype=torch.bool), tau=1e-320)


@pytest.mark.parametrize("tau", [0., -1., float("nan"), float("inf"), -float("inf")])
def test_invalid_real_temperature_rejected(tau):
    scores = torch.ones((1, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="tau"):
        core(scores, scores, torch.ones_like(scores, dtype=torch.bool), tau)


@pytest.mark.parametrize("tau", [True, "1", torch.tensor(1.)])
def test_non_scalar_real_temperature_rejected(tau):
    scores = torch.ones((1, 2), dtype=torch.float64)
    with pytest.raises(TypeError, match="tau"):
        core(scores, scores, torch.ones_like(scores, dtype=torch.bool), tau)


def test_live_null_graph_is_rejected_instead_of_detached():
    scores = torch.ones((1, 2), dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        core(scores, scores * .5, torch.ones_like(scores, dtype=torch.bool))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_active_inputs_rejected(bad):
    x = torch.tensor([[1., bad]], dtype=torch.float64)
    mask = torch.ones_like(x, dtype=torch.bool)
    with pytest.raises(ValueError, match="scores"):
        core(x, torch.zeros_like(x), mask)
    with pytest.raises(ValueError, match="null"):
        core(torch.zeros_like(x), x, mask)


def test_shape_empty_dtype_validation():
    x = torch.ones((1, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="identical"):
        core(x, x[:, :1], torch.ones_like(x, dtype=torch.bool))
    with pytest.raises(TypeError, match="Boolean"):
        core(x, x, torch.ones_like(x))
    with pytest.raises(ValueError, match="broadcastable"):
        core(x, x, torch.ones((3, 4), dtype=torch.bool))
    with pytest.raises(ValueError, match="nonempty"):
        core(x[:, :0], x[:, :0], torch.ones((1, 0), dtype=torch.bool))
    with pytest.raises(TypeError, match="floating"):
        core(torch.ones((1, 2), dtype=torch.int64), x, torch.ones_like(x, dtype=torch.bool))


def test_lme_gradient_is_residual_softmax_and_row_gradient_sum_one():
    scores = torch.tensor([[.2, -.3, .4], [.9, -.5, .1]], dtype=torch.float64, requires_grad=True)
    null = torch.tensor([[.3, .1, -.2], [-.1, .4, .2]], dtype=torch.float64)
    result = core(scores, null, torch.ones_like(scores, dtype=torch.bool), tau=.6)
    result["evidence"].sum().backward()
    torch.testing.assert_close(scores.grad, torch.softmax((scores.detach() - null) / .6, -1),
                               rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(scores.grad.sum(-1), torch.ones(2, dtype=torch.float64))


def test_gradcheck_evidence_decomposition_and_relative_branch():
    x = torch.tensor([[.7, -.3, .2]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[.2, .1, -.1]], dtype=torch.float64)
    mask = torch.ones_like(x, dtype=torch.bool)
    def operation(z):
        out = core(z, b, mask, tau=.8)
        return out["evidence"], out["residual_mean"], out["centered_lme"], out["po"]
    assert torch.autograd.gradcheck(operation, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_pool_is_population_mean_std_with_padding_and_missing_rows():
    values = torch.tensor([[1., 3., float("nan")], [4., float("inf"), 9.],
                           [float("nan"), float("inf"), float("nan")]], dtype=torch.float64,
                          requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    result = masked_mean_std_pool(values, valid)
    torch.testing.assert_close(result, torch.tensor([[2., 1.], [4., 0.], [0., 0.]], dtype=torch.float64))
    result.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[~valid].eq(0).all()


def test_pool_zero_variance_gradient_is_finite():
    values = torch.tensor([[2., 2., 2.]], dtype=torch.float64, requires_grad=True)
    result = masked_mean_std_pool(values, torch.ones_like(values, dtype=torch.bool))
    torch.testing.assert_close(result, torch.tensor([[2., 0.]], dtype=torch.float64))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 1 / 3))


def test_pool_large_finite_variance_uses_stable_scaled_squares():
    values = torch.tensor([[-1e200, 1e200]], dtype=torch.float64, requires_grad=True)
    result = masked_mean_std_pool(values, torch.ones_like(values, dtype=torch.bool))
    assert result[0, 0].item() == 0
    assert result[0, 1].item() == 1e200
    result.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_pool_gradcheck_and_wsi_axes_preserved():
    values = torch.tensor([[[.1, .7, -.2], [1., .4, -.9]]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(values, dtype=torch.bool)
    result = masked_mean_std_pool(values, mask)
    assert result.shape == (1, 2, 2)
    assert torch.autograd.gradcheck(lambda x: masked_mean_std_pool(x, mask), (values,))


def test_pool_active_nonfinite_and_invalid_inputs_fail_closed():
    values = torch.tensor([[1., float("nan")]], dtype=torch.float64)
    with pytest.raises(ValueError, match="pool values"):
        masked_mean_std_pool(values, torch.ones_like(values, dtype=torch.bool))
    with pytest.raises(TypeError, match="Boolean"):
        masked_mean_std_pool(values, torch.ones_like(values))
    with pytest.raises(ValueError, match="nonempty"):
        masked_mean_std_pool(values[:, :0], torch.ones((1, 0), dtype=torch.bool))

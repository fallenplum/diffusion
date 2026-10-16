"""Numerical and gradient oracles for the specified low-rank score only."""

import copy
import math

import pytest
import torch

from src.phase19_attention.attention import RankConditionedAttention, masked_softmax


def sample(device="cpu"):
    torch.manual_seed(1901)
    model = RankConditionedAttention(3, 4, 2, d=5, rank=4).to(device=device, dtype=torch.float64)
    p = torch.randn(2, 5, 3, dtype=torch.float64, device=device)
    o = torch.randn(2, 3, 4, dtype=torch.float64, device=device)
    c = torch.randn(2, 2, dtype=torch.float64, device=device)
    pm = torch.tensor([[True, True, True, False, False], [True] * 5], device=device)
    om = torch.tensor([[True, True, False], [True] * 3], device=device)
    return model, p, o, c, pm, om


def explicit_score(model, p, o, clinical, pm, om):
    """Literal loops independent of the optimized CP matrix multiplication."""
    pp = p.masked_fill(~pm[:, :, None], 0)
    oo = o.masked_fill(~om[:, :, None], 0)
    rows = []
    for patient in range(p.shape[0]):
        pr = []
        for i in range(p.shape[1]):
            rr = []
            for k in range(o.shape[1]):
                q = torch.stack([(w * pp[patient, i]).sum() for w in model.q.weight])
                key = torch.stack([(w * oo[patient, k]).sum() for w in model.k.weight])
                v = (q * key).sum() / math.sqrt(model.d)
                for r in range(model.rank):
                    v = v + ((model.a.weight[r] * pp[patient, i]).sum()
                             * (model.b.weight[r] * oo[patient, k]).sum()
                             * (model.c.weight[r] * clinical[patient]).sum())
                rr.append(v)
            pr.append(torch.stack(rr))
        rows.append(torch.stack(pr))
    return torch.stack(rows)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_scores_and_all_parameter_input_gradients_match_literal_oracle(device):
    model, p, o, clinical, pm, om = sample(device)
    oracle = copy.deepcopy(model)
    p, o, clinical = [x.requires_grad_() for x in (p, o, clinical)]
    po, oo, co = [x.detach().clone().requires_grad_() for x in (p, o, clinical)]
    actual = model(p, o, clinical, pm, om)
    expected = explicit_score(oracle, po, oo, co, pm, om)
    torch.testing.assert_close(actual["scores"], expected, atol=1e-12, rtol=1e-12)
    valid = pm[:, :, None] & om[:, None, :]
    # Oracle softmax uses ordinary torch on each nonempty active submatrix.
    aw = torch.zeros_like(expected)
    bw = torch.zeros_like(expected.transpose(-1, -2))
    for patient in range(p.shape[0]):
        for i in range(p.shape[1]):
            if pm[patient, i] and om[patient].any():
                aw[patient, i, om[patient]] = expected[patient, i, om[patient]].softmax(-1)
        for k in range(o.shape[1]):
            if om[patient, k] and pm[patient].any():
                bw[patient, k, pm[patient]] = expected[patient, pm[patient], k].softmax(-1)
    torch.testing.assert_close(actual["p_to_o"], aw, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual["o_to_p"], bw, atol=1e-12, rtol=1e-12)
    torch.manual_seed(155)
    ws, wa, wb = torch.randn_like(expected), torch.randn_like(aw), torch.randn_like(bw)
    loss = (actual["scores"] * ws * valid).sum() + (actual["p_to_o"] * wa).sum() + (actual["o_to_p"] * wb).sum()
    oracle_loss = (expected * ws * valid).sum() + (aw * wa).sum() + (bw * wb).sum()
    loss.backward()
    oracle_loss.backward()
    for name, x, y in zip(("p", "o", "clinical"), (p, o, clinical), (po, oo, co)):
        torch.testing.assert_close(x.grad, y.grad, atol=2e-12, rtol=2e-12, msg=name)
    for (name, parameter), (other, expected_parameter) in zip(model.named_parameters(), oracle.named_parameters()):
        assert name == other
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0, name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=3e-12, rtol=3e-12, msg=name)


def test_zero_clinical_removes_cp_term_exactly_but_preserves_clinical_gradient():
    model, p, o, c, pm, om = sample()
    zero = torch.zeros_like(c).requires_grad_()
    out = model(p, o, zero, pm, om)
    torch.testing.assert_close(out["trilinear"], torch.zeros_like(out["trilinear"]), atol=0, rtol=0)
    torch.testing.assert_close(out["scores"], out["base_scores"], atol=0, rtol=0)
    out["scores"].square().sum().backward()
    assert zero.grad.norm() > 0


def test_rank_zero_has_no_empty_parameter_or_context_dependence():
    model, p, o, c, pm, om = sample()
    baseline = RankConditionedAttention(3, 4, 2, d=5, rank=0).double()
    baseline.q.load_state_dict(model.q.state_dict())
    baseline.k.load_state_dict(model.k.state_dict())
    a = baseline(p, o, c, pm, om)
    b = baseline(p, o, -c, pm, om)
    assert set(dict(baseline.named_parameters())) == {"q.weight", "k.weight"}
    for key in ("scores", "p_to_o", "o_to_p"):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    torch.testing.assert_close(a["scores"], model(p, o, torch.zeros_like(c), pm, om)["scores"], atol=0, rtol=0)


def test_wrong_clinical_changes_scores_and_both_directions_without_changing_base():
    model, p, o, c, pm, om = sample()
    correct, wrong = model(p, o, c, pm, om), model(p, o, c.flip(0), pm, om)
    torch.testing.assert_close(correct["base_scores"], wrong["base_scores"], atol=0, rtol=0)
    for key in ("scores", "p_to_o", "o_to_p"):
        assert not torch.allclose(correct[key], wrong[key]), key


def test_shared_context_can_change_relative_association_among_tokens():
    # A patient-global clinical vector is identical across tokens but multiplies
    # token-dependent a(p_i)b(o_k), so it need not be a softmax-cancelled constant.
    model = RankConditionedAttention(2, 2, 1, d=2, rank=1).double()
    with torch.no_grad():
        model.q.weight.zero_(); model.k.weight.zero_()
        model.a.weight.copy_(torch.tensor([[1., 0.]]))
        model.b.weight.copy_(torch.tensor([[0., 1.]]))
        model.c.weight.fill_(1.)
    p = torch.tensor([[[1., 0.], [2., 0.]]], dtype=torch.float64)
    o = torch.tensor([[[0., 1.], [0., -1.]]], dtype=torch.float64)
    plus = model(p, o, torch.ones(1, 1, dtype=torch.float64))
    minus = model(p, o, -torch.ones(1, 1, dtype=torch.float64))
    assert plus["p_to_o"][0, 0, 0] > plus["p_to_o"][0, 0, 1]
    assert minus["p_to_o"][0, 0, 0] < minus["p_to_o"][0, 0, 1]
    assert plus["o_to_p"][0, 0, 1] > plus["o_to_p"][0, 0, 0]
    assert minus["o_to_p"][0, 0, 1] < minus["o_to_p"][0, 0, 0]


def test_identity_context_override_and_patient_token_permutation_equivariance():
    model, p, o, c, pm, om = sample()
    original = model(p, o, c, pm, om)
    same = model(p, o, c, pm, om, clinical_override=c)
    bp = torch.tensor([1, 0]); pp = torch.tensor([3, 0, 4, 1, 2]); op = torch.tensor([2, 0, 1])
    permuted = model(p[bp][:, pp], o[bp][:, op], c[bp], pm[bp][:, pp], om[bp][:, op])
    for key in ("scores", "base_scores", "trilinear", "p_to_o", "valid_pairs"):
        torch.testing.assert_close(original[key], same[key], atol=0, rtol=0)
        torch.testing.assert_close(permuted[key], original[key][bp][:, pp][:, :, op], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(permuted["o_to_p"], original["o_to_p"][bp][:, op][:, :, pp], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("missing", ["p", "o", "both"])
def test_all_masked_rows_have_zero_weights_and_finite_gradients(missing):
    model, p, o, c, pm, om = sample()
    if missing in ("p", "both"):
        pm[0] = False
    if missing in ("o", "both"):
        om[0] = False
    p, o, c = [v.requires_grad_() for v in (p, o, c)]
    out = model(p, o, c, pm, om)
    for key in ("scores", "base_scores", "trilinear", "p_to_o", "o_to_p"):
        assert torch.isfinite(out[key]).all(), key
    assert out["p_to_o"][0].count_nonzero() == 0
    assert out["o_to_p"][0].count_nonzero() == 0
    (out["p_to_o"].square().sum() + out["o_to_p"].square().sum()).backward()
    for v in [p, o, c, *model.parameters()]:
        assert torch.isfinite(v.grad).all()
    assert p.grad[0].count_nonzero() == 0
    assert o.grad[0].count_nonzero() == 0
    assert c.grad[0].count_nonzero() == 0


def test_padding_ignores_arbitrary_values_including_nan():
    model, p, o, c, pm, om = sample()
    expected = model(p, o, c, pm, om)
    pp, oo = p.clone(), o.clone()
    pp[~pm] = torch.nan
    oo[~om] = torch.nan
    actual = model(pp, oo, c, pm, om)
    for key in ("scores", "p_to_o", "o_to_p"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        assert torch.isfinite(actual[key]).all()


def test_probability_sums_and_softmax_direction():
    model, p, o, c, pm, om = sample()
    out = model(p, o, c, pm, om)
    torch.testing.assert_close(out["p_to_o"].sum(-1), pm.double())
    torch.testing.assert_close(out["o_to_p"].sum(-1), om.double())
    assert out["p_to_o"].shape == (2, 5, 3)
    assert out["o_to_p"].shape == (2, 3, 5)
    assert not torch.allclose(out["o_to_p"], out["p_to_o"].transpose(-1, -2))


@pytest.mark.parametrize("rank", [0, 4])
def test_score_third_mixed_derivative_matches_cp_tensor(rank):
    torch.manual_seed(99)
    model = RankConditionedAttention(2, 3, 2, d=3, rank=rank).double()
    p = torch.randn(1, 1, 2, dtype=torch.float64, requires_grad=True)
    o = torch.randn(1, 1, 3, dtype=torch.float64, requires_grad=True)
    c = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
    score = model(p, o, c)["scores"].sum()
    gp = torch.autograd.grad(score, p, create_graph=True)[0]
    for i in range(2):
        go = torch.autograd.grad(gp[0, 0, i], o, create_graph=True, retain_graph=True)[0]
        for j in range(3):
            gc = torch.autograd.grad(go[0, 0, j], c, retain_graph=True, allow_unused=True)[0]
            actual = torch.zeros(2, dtype=torch.float64) if gc is None else gc[0]
            expected = (model.a.weight[:, i, None] * model.b.weight[:, j, None] * model.c.weight).sum(0) if rank else torch.zeros(2, dtype=torch.float64)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    # This algebraic derivative is not evidence of identifiable biological or
    # causal three-way synergy; it checks the supplied score's implementation.


def test_masked_softmax_extreme_scores_and_empty_rows():
    scores = torch.tensor([[10000., -10000., 0.], [3., 4., 5.]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    weights = masked_softmax(scores, mask)
    torch.testing.assert_close(weights, torch.tensor([[1., 0., 0.], [0., 0., 0.]], dtype=torch.float64))
    weights.square().sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_masked_softmax_matches_torch_for_fully_available_rows():
    scores = torch.randn(2, 3, 5, dtype=torch.float64)
    for dim in (-1, -2):
        torch.testing.assert_close(masked_softmax(scores, torch.ones_like(scores, dtype=torch.bool), dim), scores.softmax(dim))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_augmented_query_key_equivalence_includes_all_gradients(device):
    # CP conditioning is exactly an augmented Q/K bilinear kernel with a
    # clinical multiplicative modulation of the appended query coordinates.
    # This representation is an algebraic oracle, not a distinct mechanism
    # baseline that could independently confirm clinical-conditioned synergy.
    model, p, o, clinical, pm, om = sample(device)
    augmented = copy.deepcopy(model)
    p, o, clinical = [x.requires_grad_() for x in (p, o, clinical)]
    pa, oa, ca = [x.detach().clone().requires_grad_() for x in (p, o, clinical)]
    actual = model(p, o, clinical, pm, om)
    pp = pa.masked_fill(~pm[:, :, None], 0)
    oo = oa.masked_fill(~om[:, :, None], 0)
    q_aug = torch.cat([augmented.q(pp) / math.sqrt(augmented.d),
                       augmented.a(pp) * augmented.c(ca)[:, None, :]], dim=-1)
    k_aug = torch.cat([augmented.k(oo), augmented.b(oo)], dim=-1)
    expected = q_aug @ k_aug.transpose(-1, -2)
    valid = pm[:, :, None] & om[:, None, :]
    a = masked_softmax(expected, valid)
    b = masked_softmax(expected.transpose(-1, -2), valid.transpose(-1, -2))
    torch.testing.assert_close(actual["scores"], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual["p_to_o"], a, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual["o_to_p"], b, atol=1e-12, rtol=1e-12)
    ws, wa, wb = torch.randn_like(expected), torch.randn_like(a), torch.randn_like(b)
    loss = (actual["scores"] * ws).sum() + (actual["p_to_o"] * wa).sum() + (actual["o_to_p"] * wb).sum()
    loss_a = (expected * ws).sum() + (a * wa).sum() + (b * wb).sum()
    loss.backward()
    loss_a.backward()
    for name, x, y in zip(("p", "o", "clinical"), (p, o, clinical), (pa, oa, ca)):
        torch.testing.assert_close(x.grad, y.grad, atol=3e-12, rtol=3e-12, msg=name)
    for (name, par), (_, ref) in zip(model.named_parameters(), augmented.named_parameters()):
        torch.testing.assert_close(par.grad, ref.grad, atol=3e-12, rtol=3e-12, msg=name)


@pytest.mark.parametrize("kwargs", [{"rank": -1}, {"rank": 1.5}, {"d": 0}, {"p_dim": 0}])
def test_invalid_configuration_fails_closed(kwargs):
    config = {"p_dim": 3, "o_dim": 4, "c_dim": 2, "d": 32, "rank": 4}
    config.update(kwargs)
    with pytest.raises(ValueError):
        RankConditionedAttention(**config)


def test_bad_masks_and_misaligned_context_fail_closed():
    model, p, o, c, pm, om = sample()
    with pytest.raises(ValueError):
        model(p, o, c[:1], pm, om)
    with pytest.raises(ValueError):
        model(p, o, c, pm.float(), om)
    with pytest.raises(ValueError):
        model(p, o, c, pm[:, :2], om)
    with pytest.raises(TypeError):
        masked_softmax(torch.randn(2, 3), torch.ones(2, 3))

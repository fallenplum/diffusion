"""Numerical operator/gradient tests only; no cohort data or real training."""
import copy
import math

import pytest
import torch

from src.phase24_cdei_g1.model import G1Model


@pytest.fixture
def example():
    torch.set_num_threads(2)
    torch.manual_seed(240018)
    member = torch.zeros(8, 16, dtype=torch.bool)
    for k in range(8):
        member[k, 2*k:2*k+2] = True
    model = G1Model(4, member, arm="LOCAL").eval()
    r, g, l = torch.randn(7, 16), torch.rand(7, 16), torch.rand(7, 16)
    rm, cm = torch.ones_like(r, dtype=torch.bool), torch.ones_like(r, dtype=torch.bool)
    clinical = torch.randn(7, 4)
    visual, mask = torch.randn(2, 5, 1536), torch.ones(2, 5, dtype=torch.bool)
    mask[:, -1] = False
    visual[:, -1] = float("nan")
    raw = {k: v[:4].clone() for k, v in zip(("R", "G", "L", "RM", "CM"), (r, g, l, rm, cm))}
    raw["indices"] = torch.arange(4)
    return model, (visual, mask, clinical, r, g, l, rm, cm), raw


def call(model, f, raw, ids=(4, 5), **kw):
    v, m, c, r, g, l, rm, cm = f
    ix = torch.as_tensor(ids)
    return model(v, m, c[ix], r[ix], g[ix], l[ix], rm[ix], cm[ix], ix, raw,
                 return_details=True, **kw)


@pytest.mark.parametrize("arm,extra_dim", [("REL", 0), ("POOL", 2), ("LOCAL", 32), ("CTRL", 32)])
def test_four_arms_shapes_and_finite_backward(example, arm, extra_dim):
    original, f, raw = example
    torch.manual_seed(240018)
    model = G1Model(4, original.backbone.molecular.membership, arm=arm)
    result = call(model, f, raw)
    assert result["risk_delta_unit"].shape == (2,)
    assert result["relative"].shape == (2, 64)
    assert result["extra"].shape == (2, extra_dim)
    assert torch.isfinite(result["risk_delta_unit"]).all()
    assert (result["risk_delta_unit"].abs() <= 1).all()
    result["risk_delta_unit"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for name in ("q", "k", "a", "b", "c"):
        grad = getattr(model.backbone.compatibility, name).weight.grad
        assert grad is not None and grad.norm() > 0, name
    for channel in range(3):
        assert sum(float(model.backbone.molecular.filters[i].grad.norm())
                   for i in range(channel*8, channel*8+8)) > 0
    if extra_dim:
        assert model.extra_linear.weight.grad.norm() > 0


def test_local_is_signed_value_message_without_second_normalization(example):
    model, f, raw = example
    result = call(model, f, raw)
    d = result["details"]
    expected = torch.where(d["absolute_valid_pairs"], torch.tanh(d["D"]/model.tau),
                           torch.zeros_like(d["D"]))
    weight = d["A"] * expected
    local = weight @ d["context_omics_values"]
    pool = torch.where(f[1][..., None], local, torch.zeros_like(local)).sum(1)/f[1].sum(1, keepdim=True)
    torch.testing.assert_close(d["weighted_weights"], weight, atol=0, rtol=0)
    torch.testing.assert_close(d["weightedmessages"], local, atol=0, rtol=0)
    torch.testing.assert_close(result["extra"], pool, atol=0, rtol=0)
    assert (weight < 0).any(), "signed evidence must retain the negative contributions"
    assert not torch.allclose(weight.sum(-1)[f[1]], torch.ones_like(weight.sum(-1)[f[1]]))
    assert not torch.allclose(result["extra"].norm(dim=-1), torch.ones(2))


def test_ctrl_exact_formula_and_parameter_initialization_match_local(example):
    local, f, raw = example
    torch.manual_seed(240018)
    ctrl = G1Model(4, local.backbone.molecular.membership, "CTRL").eval()
    assert sum(p.numel() for p in ctrl.parameters()) == sum(p.numel() for p in local.parameters())
    for key, x in local.state_dict().items():
        torch.testing.assert_close(x, ctrl.state_dict()[key], atol=0, rtol=0)
    result = call(ctrl, f, raw)
    d = result["details"]
    weights = d["A"] * torch.where(d["absolute_valid_pairs"],
                                   torch.sigmoid(d["context_scores"]/ctrl.tau),
                                   torch.zeros_like(d["context_scores"]))
    torch.testing.assert_close(d["weighted_weights"], weights, atol=0, rtol=0)
    torch.testing.assert_close(d["weightedmessages"], weights @ d["context_omics_values"], atol=0, rtol=0)


def test_shared_parameters_and_rng_state_are_identical_across_arms(example):
    initial, _, _ = example
    models, states = [], []
    for arm in ("REL", "POOL", "LOCAL", "CTRL"):
        torch.manual_seed(51)
        models.append(G1Model(4, initial.backbone.molecular.membership, arm))
        states.append(torch.random.get_rng_state().clone())
    for other in models[1:]:
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
    assert all(torch.equal(states[0], s) for s in states[1:])


def test_pool_is_logmeanexp_then_patch_mean_and_population_sd(example):
    initial, f, raw = example
    torch.manual_seed(240018)
    model = G1Model(4, initial.backbone.molecular.membership, "POOL").eval()
    result = call(model, f, raw)
    d = result["details"]
    expected = []
    for patient in range(2):
        rows = []
        for patch in range(f[1].shape[1]):
            valid = d["absolute_valid_pairs"][patient, patch]
            if valid.any():
                x = d["D"][patient, patch, valid].double()/model.tau
                rows.append(model.tau*(torch.logsumexp(x, 0)-math.log(len(x))))
        rows = torch.stack(rows)
        expected.append(torch.stack((rows.mean(), rows.std(unbiased=False))))
    torch.testing.assert_close(result["extra"], torch.stack(expected).float(), atol=1e-7, rtol=1e-6)


def test_reference_is_patient_equal_leave_self_out_and_has_no_heldout_rows(example):
    model, _, raw = example
    query = torch.tensor([0, 1, 5])
    ref = model.current_reference(raw, query)
    encoded, masks = model.backbone.encode_omics(*(raw[k] for k in ("R", "G", "L", "RM", "CM")))
    for i, keep in enumerate(([1, 2, 3], [0, 2, 3], [0, 1, 2, 3])):
        torch.testing.assert_close(ref["tokens"][i], encoded[keep].mean(0), atol=1e-7, rtol=1e-6)
        assert torch.equal(ref["counts"][i], masks[keep].sum(0))
    extra = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 5, 0]])
    joint = model.current_reference(raw, query, additional_exclusions=extra)
    torch.testing.assert_close(joint["tokens"][0], encoded[[2, 3]].mean(0))
    torch.testing.assert_close(joint["tokens"][1], encoded[[2, 3]].mean(0))
    torch.testing.assert_close(joint["tokens"][2], encoded[[1, 2, 3]].mean(0))


def test_reference_mean_has_slot_specific_denominator_not_zero_imputation(example):
    model, _, raw = example
    raw["RM"][0, :2] = False
    raw["R"][0, :2] = float("nan")
    raw["CM"][:, 2:4] = False
    raw["G"][:, 2:4] = float("nan")
    raw["L"][:, 2:4] = float("nan")
    ref = model.current_reference(raw, torch.tensor([5]))
    assert int(ref["counts"][0, 0]) == 3
    assert int(ref["counts"][0, 9]) == 0
    assert int(ref["counts"][0, 17]) == 0
    assert not ref["mask"][0, [9, 17]].any()
    assert torch.equal(ref["tokens"][0, [9, 17]], torch.zeros(2, 32))
    assert torch.isfinite(ref["tokens"]).all()


def test_reference_gradient_is_not_detached_and_refreshes_after_step(example):
    model, f, raw = example
    raw["R"].requires_grad_()
    before = call(model, f, raw)
    assert before["details"]["reference_tokens"].requires_grad
    before["extra"].square().sum().backward()
    assert raw["R"].grad is not None and raw["R"].grad.norm() > 0
    assert model.backbone.molecular.filters[0].grad.norm() > 0
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    saved = before["details"]["reference_tokens"].detach().clone()
    opt.step()
    after = call(model, f, raw)
    assert not torch.equal(saved, after["details"]["reference_tokens"])
    current = model.current_reference(raw, [4, 5])
    torch.testing.assert_close(after["details"]["reference_tokens"], current["tokens"], atol=0, rtol=0)
    p = model.backbone.encode_pathology(f[0], f[1])
    recomputed = model.backbone.score_encoded(p, current["tokens"], f[2][[4, 5]], f[1], current["mask"])
    torch.testing.assert_close(after["details"]["null"], recomputed["scores"], atol=0, rtol=0)


def test_clinical_intervention_keeps_matched_attention_relative_and_inputs(example):
    model, f, raw = example
    copies = [v.clone() for v in f]
    matched = call(model, f, raw)
    wrong = call(model, f, raw, absolute_clinical=f[2][[5, 4]])
    for key in ("relative",):
        torch.testing.assert_close(wrong[key], matched[key], atol=0, rtol=0)
    torch.testing.assert_close(wrong["details"]["A"], matched["details"]["A"], atol=0, rtol=0)
    assert not torch.equal(wrong["extra"], matched["extra"])
    for value, previous in zip(f, copies):
        torch.testing.assert_close(value, previous, atol=0, rtol=0, equal_nan=True)


def test_wrong_omics_uses_same_reference_union_mask_and_matched_A(example):
    model, f, raw = example
    own, donor = torch.tensor([0, 1]), torch.tensor([1, 0])
    exclusions = torch.stack((own, donor), -1)
    shared = torch.ones(2, 24, dtype=torch.bool)
    args = dict(reference_exclusions=exclusions, absolute_token_mask=shared)
    matched = call(model, f, raw, ids=own, **args)
    wrong = call(model, f, raw, ids=own, absolute_omics=tuple(x[donor] for x in f[3:]),
                 omics_source_indices=donor, **args)
    for key in ("relative",):
        torch.testing.assert_close(matched[key], wrong[key], atol=0, rtol=0)
    for key in ("A", "reference_counts", "reference_tokens"):
        torch.testing.assert_close(matched["details"][key], wrong["details"][key], atol=0, rtol=0)
    assert not torch.equal(matched["extra"], wrong["extra"])


@pytest.mark.parametrize("missing", ["source", "plan", "mask"])
def test_wrong_omics_requires_explicit_pair_and_support_plan(example, missing):
    model, f, raw = example
    args = dict(absolute_omics=tuple(x[[5, 4]] for x in f[3:]), omics_source_indices=[5, 4],
                reference_exclusions=[[4, 5], [5, 4]], absolute_token_mask=torch.ones(2, 24, dtype=torch.bool))
    args.pop({"source": "omics_source_indices", "plan": "reference_exclusions", "mask": "absolute_token_mask"}[missing])
    with pytest.raises(ValueError):
        call(model, f, raw, **args)


def test_absolute_off_keeps_relative_head_and_zeroes_only_extra(example):
    model, f, raw = example
    matched = call(model, f, raw)
    off = call(model, f, raw, absolute_off=True)
    torch.testing.assert_close(matched["relative"], off["relative"], atol=0, rtol=0)
    assert torch.equal(off["extra"], torch.zeros(2, 32))
    expected = torch.tanh(model.output(model.activation(model.relative_linear(off["relative"]))).squeeze(-1))
    torch.testing.assert_close(off["risk_delta_unit"], expected, atol=0, rtol=0)


@pytest.mark.parametrize("arm", ["POOL", "LOCAL", "CTRL"])
def test_empty_modalities_and_padding_nan_are_finite_and_empty_extra_zero(example, arm):
    initial, f, raw = example
    model = G1Model(4, initial.backbone.molecular.membership, arm).eval()
    f = list(f)
    f[1][:] = False
    f[0][:] = float("nan")
    result = call(model, f, raw)
    assert torch.isfinite(result["risk_delta_unit"]).all()
    assert not result["extra"].any()
    result["risk_delta_unit"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("arm", ["POOL", "LOCAL", "CTRL"])
def test_no_observed_omics_returns_zero_extra_with_finite_gradient(example, arm):
    initial, f, raw = example
    model = G1Model(4, initial.backbone.molecular.membership, arm).eval()
    f = list(f)
    f[-2][:] = False
    f[-1][:] = False
    for x in f[3:6]:
        x[:] = float("nan")
    result = call(model, f, raw)
    assert torch.isfinite(result["risk_delta_unit"]).all()
    assert not result["extra"].any()
    result["risk_delta_unit"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_zero_reference_difference_has_zero_pool_sd_and_finite_backward(example):
    initial, f, raw = example
    model = G1Model(4, initial.backbone.molecular.membership, "POOL").eval()
    f = list(f)
    for x in f[3:]:
        x[1] = x[0].clone()
    raw = {k: x[:2] for k, x in zip(("R", "G", "L", "RM", "CM"), f[3:])}
    raw["indices"] = [0, 1]
    result = call(model, f, raw, ids=(0, 1))
    torch.testing.assert_close(result["extra"], torch.zeros(2, 2), atol=0, rtol=0)
    result["risk_delta_unit"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_rejected(example, tau):
    model, _, _ = example
    with pytest.raises(ValueError):
        G1Model(4, model.backbone.molecular.membership, tau=tau)


@pytest.mark.parametrize("failure", ["duplicate", "label", "float_indices", "bad_width", "nonfinite"])
def test_reference_contract_fails_closed(example, failure):
    model, _, raw = example
    if failure == "duplicate": raw["indices"] = [0, 0, 1, 2]
    elif failure == "label": raw["event"] = torch.ones(4)
    elif failure == "float_indices": raw["indices"] = torch.arange(4).float()
    elif failure == "bad_width": raw["R"] = raw["R"][:, :-1]
    elif failure == "nonfinite": raw["R"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        model.current_reference(raw, [4, 5])


@pytest.mark.parametrize("field", [0, 2, 3])
def test_nonfinite_observed_inputs_rejected(example, field):
    model, f, raw = example
    f[field].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        call(model, f, raw, ids=(0, 1))

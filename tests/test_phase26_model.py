"""Synthetic arithmetic, information-path and initialization qualification."""
import copy
import math
import numpy as np
import pytest
import torch

from src.phase26_multiquery.model import MultiQuery, attention_weights


@pytest.fixture
def example():
    torch.set_num_threads(2); torch.manual_seed(260018)
    p = torch.nn.functional.normalize(torch.randn(3, 6, 768), dim=-1)
    m = torch.ones(3, 6, dtype=torch.bool); m[:, -2:] = False
    p[:, -2:] = float("nan")
    return p, m, torch.randn(3, 512), torch.randn(3, 5)


def make(arm, active_head=False):
    torch.manual_seed(260201)
    model = MultiQuery(5, arm)
    if active_head:
        with torch.no_grad(): torch.nn.init.normal_(model.head[-1].weight, std=.1)
    return model


def test_all_arms_have_identical_initial_states_and_exact_zero_initial_risk(example):
    models = [make(arm) for arm in ("QUERY", "ATTN", "COMP")]
    for model in models:
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, models[0].state_dict()[key], atol=0, rtol=0)
        result = model(*example, return_details=True)
        assert torch.equal(result["unit"], torch.zeros(3))
        assert result["slots"].shape == (3, 8, 32) and result["attention"].shape == (3, 8, 6)
        assert not any(isinstance(x, torch.nn.Dropout) for x in model.modules())
    assert not models[1].clinical_up.weight.any()
    assert models[1].slot_down.weight.norm() > 0 and models[1].clinical_down.weight.norm() > 0


def test_fixed_balanced_groups_use_all_genes_once_without_changing_numpy_rng():
    before = copy.deepcopy(np.random.get_state())
    model = make("ATTN")
    groups = model.rna.groups
    assert groups.shape == (8, 64)
    assert torch.equal(groups.flatten().sort().values, torch.arange(512))
    assert np.array_equal(groups.numpy(), np.random.default_rng(117301).permutation(512).reshape(8, 64))
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize("arm", ["QUERY", "ATTN", "COMP"])
def test_final_zero_head_first_gradient_gate_is_expected_and_finite(example, arm):
    model = make(arm)
    result = model(*example)
    assert set(result) == {"unit"}
    (result["unit"]-torch.tensor([.3, -.1, .2])).square().sum().backward()
    assert model.head[-1].weight.grad.norm() > 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            if name != "head.2.weight": assert not parameter.grad.any(), name


@pytest.mark.parametrize("arm", ["ATTN", "COMP"])
def test_risk_gradients_activate_after_head_step_and_clinical_U_has_gradient(example, arm):
    model = make(arm)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    target = torch.tensor([.3, -.1, .2])
    (model(*example)["unit"]-target).square().sum().backward(); optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    (model(*example)["unit"]-target).square().sum().backward()
    for name in ("rna.weight", "patch.0.weight", "query.weight", "key.weight", "value.weight", "clinical_up.weight"):
        g = dict(model.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, name
    # V/T are behind U=0 until its first effective update; not a detached path.
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    (model(*example)["unit"]-target).square().sum().backward()
    assert model.slot_down.weight.grad.norm() > 0 and model.clinical_down.weight.grad.norm() > 0


def test_query_arm_keeps_all_slots_and_has_no_patch_or_score_branch_risk_path(example):
    model = make("QUERY", active_head=True)
    result = model(*example, return_details=True)
    assert not result["attention"].any() and not result["message"].any()
    expected = result["initial_slots"]
    for _ in range(2): expected = expected+model.update_ff(expected)
    torch.testing.assert_close(result["slots"], expected, atol=0, rtol=0)
    torch.testing.assert_close(result["unit"], torch.tanh(model.head(expected.flatten(1))).squeeze(-1), atol=0, rtol=0)
    result["unit"].sum().backward()
    assert model.rna.weight.grad.norm() > 0
    for name in ("patch.0.weight", "query.weight", "key.weight", "value.weight", "clinical_up.weight"):
        assert dict(model.named_parameters())[name].grad is None, name


@pytest.mark.parametrize("arm", ["ATTN", "COMP"])
def test_value_off_equals_query_path_with_same_weights_and_keeps_own_RNA_residual(example, arm):
    model = make(arm, active_head=True)
    query_model = copy.deepcopy(model); query_model.arm = "QUERY"
    off = model(*example, value_off=True, return_details=True)
    direct = query_model(*example, return_details=True)
    for name in ("unit", "initial_slots", "slots", "attention", "message"):
        torch.testing.assert_close(off[name], direct[name], atol=0, rtol=0)
    wrong = model(*example, query_rna=example[2].roll(1, 0), query_clinical=example[3].roll(1, 0),
                  value_off=True, return_details=True)
    torch.testing.assert_close(wrong["unit"], off["unit"], atol=0, rtol=0)


@pytest.mark.parametrize("arm", ["ATTN", "COMP"])
def test_two_rounds_match_manual_fixed_projection_FF_arithmetic(example, arm):
    model = make(arm, active_head=True)
    result = model(*example, return_details=True)
    h = result["initial_slots"]
    for row in result["rounds"]:
        q = model._query_vector(h, example[3])
        scores = q @ result["key"].transpose(-1, -2)/math.sqrt(32)
        a, _, _ = attention_weights(scores, example[1], arm)
        message = a @ result["value"]
        h = h+message; h = h+model.update_ff(h)
        torch.testing.assert_close(row["message"], message, atol=0, rtol=0)
        torch.testing.assert_close(row["output_slots"], h, atol=0, rtol=0)
    torch.testing.assert_close(result["slots"], h, atol=0, rtol=0)


@pytest.mark.parametrize("arm", ["ATTN", "COMP"])
def test_RNA_query_intervention_only_shifts_score_source_each_round(example, arm):
    model = make(arm, active_head=True)
    matched = model(*example, return_details=True)
    altered = model(*example, query_rna=example[2].roll(1, 0), return_details=True)
    torch.testing.assert_close(matched["initial_slots"], altered["initial_slots"], atol=0, rtol=0)
    for name in ("key", "value"):
        torch.testing.assert_close(matched[name], altered[name], atol=0, rtol=0)
    for row in altered["rounds"]:
        torch.testing.assert_close(row["score_source"], row["input_slots"]+altered["query_initial_shift"], atol=0, rtol=0)
    assert not torch.equal(matched["attention"], altered["attention"])
    assert not torch.equal(matched["unit"], altered["unit"])


@pytest.mark.parametrize("arm", ["ATTN", "COMP"])
def test_slot_dependent_Clinical_term_can_change_competition_with_nonzero_U(example, arm):
    model = make(arm, active_head=True)
    with torch.no_grad(): torch.nn.init.normal_(model.clinical_up.weight, std=.3)
    matched = model(*example, return_details=True)
    altered = model(*example, query_clinical=example[3].roll(1, 0), return_details=True)
    torch.testing.assert_close(matched["initial_slots"], altered["initial_slots"], atol=0, rtol=0)
    assert not torch.equal(matched["attention"], altered["attention"])
    assert not torch.equal(matched["unit"], altered["unit"])


def test_competitive_operator_matches_direct_double_normalization_without_epsilon():
    torch.manual_seed(26)
    score = torch.randn(2, 8, 5, dtype=torch.float64)
    mask = torch.tensor([[True, True, False, True, False], [True, True, True, False, False]])
    a, b, mass = attention_weights(score, mask, "COMP")
    direct = score.softmax(1)*mask[:, None]
    expected = direct/direct.sum(-1, keepdim=True)
    torch.testing.assert_close(b, direct, atol=2e-16, rtol=1e-15)
    torch.testing.assert_close(mass, direct.sum(-1), atol=2e-16, rtol=1e-15)
    torch.testing.assert_close(a, expected, atol=2e-16, rtol=1e-15)
    torch.testing.assert_close(a.sum(-1), torch.ones(2, 8, dtype=torch.float64), atol=3e-16, rtol=1e-15)


def test_common_per_patch_score_shift_cancels_in_competition_but_not_row_attention():
    torch.manual_seed(261)
    score = torch.randn(2, 8, 6, dtype=torch.float64)
    shift = torch.randn(2, 1, 6, dtype=torch.float64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    a = attention_weights(score, mask, "COMP")[0]
    b = attention_weights(score+shift, mask, "COMP")[0]
    torch.testing.assert_close(a, b, atol=2e-16, rtol=1e-15)
    assert not torch.allclose(attention_weights(score, mask, "ATTN")[0],
                              attention_weights(score+shift, mask, "ATTN")[0])


def test_log_domain_competition_keeps_rows_normalized_under_assignment_underflow():
    score = torch.zeros(1, 8, 4); score[:, 0] = 10000
    mask = torch.tensor([[True, True, False, True]])
    a, assignment, mass = attention_weights(score, mask, "COMP")
    assert (mass[:, 1:] == 0).all()  # Honest diagnostic underflow, no epsilon.
    assert torch.isfinite(a).all() and not a[:, :, 2].any()
    torch.testing.assert_close(a.sum(-1), torch.ones(1, 8), atol=1e-7, rtol=0)


@pytest.mark.parametrize("arm", ["QUERY", "ATTN", "COMP"])
def test_padded_nan_rows_are_ignored_and_additional_padding_is_invariant(example, arm):
    model = make(arm, active_head=True)
    p, mask, r, c = example
    original = model(*example, return_details=True)
    extra = torch.cat((p, torch.full((3, 2, 768), float("nan"))), 1)
    extra_mask = torch.cat((mask, torch.zeros(3, 2, dtype=torch.bool)), 1)
    padded = model(extra, extra_mask, r, c, return_details=True)
    torch.testing.assert_close(original["unit"], padded["unit"], atol=2e-7, rtol=1e-6)
    assert not padded["attention"].masked_select(~extra_mask[:, None].expand_as(padded["attention"])).any()
    assert torch.isfinite(padded["unit"]).all() and torch.isfinite(padded["slots"]).all()


@pytest.mark.parametrize("arm", ["QUERY", "ATTN", "COMP"])
def test_unit_is_hard_bounded_before_callers_fixed_point1_coefficient(example, arm):
    model = make(arm, active_head=True)
    with torch.no_grad(): model.head[-1].weight.mul_(10000)
    out = model(*example, return_details=True)
    assert torch.isfinite(out["unit"]).all() and (out["unit"].abs() <= 1).all()
    assert (.1*out["unit"].abs() <= .1).all()


@pytest.mark.parametrize("failure", ["all_masked", "observed_nan", "RNA_width", "Clinical_width", "override_nan", "mask_float"])
def test_invalid_contract_rejected(example, failure):
    p, m, r, c = example; kwargs = {}
    if failure == "all_masked": m[0] = False
    elif failure == "observed_nan": p[0, 0, 0] = float("nan")
    elif failure == "RNA_width": r = r[:, :-1]
    elif failure == "Clinical_width": c = c[:, :-1]
    elif failure == "override_nan": kwargs["query_rna"] = r.clone(); kwargs["query_rna"][0, 0] = float("nan")
    elif failure == "mask_float": m = m.float()
    with pytest.raises(ValueError): make("COMP")(p, m, r, c, **kwargs)

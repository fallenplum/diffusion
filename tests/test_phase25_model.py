"""Synthetic locked local-routing arithmetic and information-path contracts."""
import math

import pytest
import torch

from src.phase25_local.model import LocalRouting


@pytest.fixture
def example():
    torch.set_num_threads(2); torch.manual_seed(250125)
    patch = torch.nn.functional.normalize(torch.randn(3, 7, 768), dim=-1)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[:, -2:] = False
    patch[:, -2:] = float("nan")
    r, c = torch.randn(3, 512), torch.randn(3, 5)
    return patch, mask, r, c


def models():
    result, rng = [], []
    for arm in ("R1", "R2", "R3"):
        torch.manual_seed(250201)
        result.append(LocalRouting(5, arm))
        rng.append(torch.random.get_rng_state().clone())
    return result, rng


def test_all_shared_parameters_initial_outputs_and_post_init_rng_are_identical(example):
    networks, rng = models()
    base = networks[0](*example, return_details=True)
    for model in networks[1:]:
        state = model.state_dict()
        for name, value in networks[0].state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)
        result = model(*example, return_details=True)
        for key, value in base.items():
            torch.testing.assert_close(value, result[key], atol=0, rtol=0)
    assert all(torch.equal(rng[0], other) for other in rng[1:])
    assert not networks[1].rna.weight.any()
    assert not networks[2].rna.weight.any() and not networks[2].clinical.weight.any()


@pytest.mark.parametrize("arm", ["R1", "R2", "R3"])
def test_entire_operator_matches_literal_formula(example, arm):
    model = LocalRouting(5, arm)
    p, m, r, c = example
    result = model(*example, return_details=True)
    x = torch.where(m[..., None], p, torch.zeros_like(p))*math.sqrt(768)
    k, v = torch.tanh(model.keys(x)), model.values(x)
    pre = model.q0[None].expand(len(p), -1)
    if model.rna is not None: pre = pre+model.rna(r)
    if model.clinical is not None: pre = pre+model.clinical(c)
    q = torch.tanh(pre)
    score = (q[:, None]*k).sum(-1)/4
    a = score.masked_fill(~m, -torch.inf).softmax(-1)
    z = (a[..., None]*v).sum(1)-v.sum(1)/m.sum(1, keepdim=True)
    torch.testing.assert_close(result["attention"], a, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(result["feature"], z, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(result["unit"], model.head(z).squeeze(-1), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("arm", ["R1", "R2", "R3"])
def test_mask_padding_nan_and_extra_padding_do_not_affect_outputs(example, arm):
    model = LocalRouting(5, arm)
    p, m, r, c = example
    original = model(*example, return_details=True)
    padded = torch.cat((p, torch.full((3, 4, 768), float("nan"))), 1)
    mask = torch.cat((m, torch.zeros(3, 4, dtype=torch.bool)), 1)
    changed = model(padded, mask, r, c, return_details=True)
    torch.testing.assert_close(original["unit"], changed["unit"], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(original["attention"], changed["attention"][:, :7], atol=1e-7, rtol=1e-6)
    assert not changed["attention"].masked_select(~mask).any()
    for key in changed:
        assert torch.isfinite(changed[key]).all(), key


@pytest.mark.parametrize("arm", ["R1", "R2", "R3"])
def test_zero_conditional_init_still_receives_nonzero_finite_gradients(example, arm):
    model = LocalRouting(5, arm)
    output = model(*example)["unit"]
    loss = (output-torch.tensor([.5, -.3, .2])).square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name


def test_context_can_change_routing_after_one_synthetic_step(example):
    model = LocalRouting(5, "R3")
    initial = model(*example, return_details=True)
    initial["unit"].square().sum().backward()
    torch.optim.SGD(model.parameters(), lr=.1).step()
    own = model(*example, return_details=True)
    wrong = model(*example, query_rna=example[2].roll(1, 0),
                  query_clinical=example[3].roll(1, 0), return_details=True)
    assert not torch.equal(own["query"], wrong["query"])
    assert not torch.equal(own["attention"], wrong["attention"])
    assert not torch.equal(own["unit"], wrong["unit"])
    torch.testing.assert_close(own["key"], wrong["key"], atol=0, rtol=0)
    torch.testing.assert_close(own["value"], wrong["value"], atol=0, rtol=0)


def test_query_override_equivalent_to_changing_inputs_with_no_direct_head_path(example):
    model = LocalRouting(5, "R3")
    with torch.no_grad():
        torch.nn.init.normal_(model.rna.weight, std=.03)
        torch.nn.init.normal_(model.clinical.weight, std=.1)
    p, m, r, c = example
    other_r, other_c = r.roll(1, 0), c.roll(1, 0)
    intervened = model(*example, query_rna=other_r, query_clinical=other_c, return_details=True)
    replaced = model(p, m, other_r, other_c, return_details=True)
    for key in intervened:
        torch.testing.assert_close(intervened[key], replaced[key], atol=0, rtol=0)
    # With context projected out, altered R/C cannot alter the output.
    with torch.no_grad(): model.rna.weight.zero_(); model.clinical.weight.zero_()
    torch.testing.assert_close(model(*example)["unit"], model(p, m, other_r, other_c)["unit"], atol=0, rtol=0)


def test_uniform_attention_exactly_removes_mean_value(example):
    model = LocalRouting(5, "R3")
    with torch.no_grad(): model.q0.zero_()
    result = model(*example, return_details=True)
    torch.testing.assert_close(result["feature"], torch.zeros_like(result["feature"]), atol=1e-7, rtol=0)
    torch.testing.assert_close(result["unit"], torch.zeros_like(result["unit"]), atol=1e-7, rtol=0)
    torch.testing.assert_close(result["normalized_entropy"], torch.ones(3), atol=1e-7, rtol=0)
    torch.testing.assert_close(result["effective_patch_count"], torch.full((3,), 5.), atol=1e-6, rtol=0)


def test_single_real_patch_has_zero_centered_message_and_finite_gradients(example):
    p, m, r, c = example
    m[:] = False; m[:, 0] = True
    model = LocalRouting(5, "R3")
    output = model(p, m, r, c, return_details=True)
    torch.testing.assert_close(output["feature"], torch.zeros(3, 16), atol=0, rtol=0)
    assert not output["normalized_entropy"].any()
    assert torch.equal(output["effective_patch_count"], torch.ones(3))
    output["unit"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_local_output_has_no_final_tanh_or_other_bound(example):
    model = LocalRouting(5, "R1")
    before = model(*example)["unit"].detach().clone()
    assert before.abs().max() > 0
    with torch.no_grad(): model.head.weight.mul_(1000)
    after = model(*example)["unit"]
    torch.testing.assert_close(after, before*1000, atol=1e-5, rtol=1e-5)
    assert after.abs().max() > 1


@pytest.mark.parametrize("bad", ["empty", "mask_float", "observed_nan", "RNA_width", "C_width", "RNA_nan", "C_nan"])
def test_invalid_observation_contract_fails_closed(example, bad):
    p, m, r, c = example
    if bad == "empty": m[0] = False
    elif bad == "mask_float": m = m.float()
    elif bad == "observed_nan": p[0, 0, 0] = float("nan")
    elif bad == "RNA_width": r = r[:, :-1]
    elif bad == "C_width": c = c[:, :-1]
    elif bad == "RNA_nan": r[0, 0] = float("nan")
    elif bad == "C_nan": c[0, 0] = float("nan")
    with pytest.raises(ValueError): LocalRouting(5, "R3")(p, m, r, c)


@pytest.mark.parametrize("arm,key", [("R1", "query_rna"), ("R1", "query_clinical"), ("R2", "query_clinical")])
def test_inactive_context_override_is_rejected(example, arm, key):
    extra = example[2] if key == "query_rna" else example[3]
    with pytest.raises(ValueError): LocalRouting(5, arm)(*example, **{key: extra})


@pytest.mark.parametrize("kwargs", [{"clinical_dim": 0}, {"clinical_dim": True}, {"clinical_dim": 2.5},
                                    {"arm": "R0"}, {"hidden": 32}])
def test_locked_constructor_rejects_other_architectures(kwargs):
    with pytest.raises(ValueError): LocalRouting(**{"clinical_dim": 5, "arm": "R1", **kwargs})

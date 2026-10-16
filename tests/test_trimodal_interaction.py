"""Synthetic invariants for the four-path interaction, without cohort access.

These checks prove operator/gradient properties only.  They are not model fits,
biological validation, or a replacement for private data/provenance audits.
"""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from src.phase17_ssd.losses import full_risk_efron
from src.trimodal_interaction.model import Interaction, Model, masked_softmax


@pytest.fixture(autouse=True)
def deterministic_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def example(arm="A5", batch=3, patches=7, dtype=torch.float64):
    torch.manual_seed(1917)
    nc = 5
    c0 = torch.randn(nc) * .2
    r0 = torch.randn(512) * .1
    rm0 = torch.linspace(.1, .9, 512)
    operator = Interaction(nc, c0, r0, rm0, arm).to(dtype=dtype)
    c = torch.randn(batch, nc, dtype=dtype)
    r = torch.randn(batch, 512, dtype=dtype)
    rm = torch.randint(0, 2, (batch, 512)).to(dtype=dtype)
    p = torch.randn(batch, patches, 64, dtype=dtype)
    mask = torch.ones(batch, patches, dtype=torch.bool)
    return operator, (c, r, rm, p, mask)


def four_maps(operator, inputs):
    c, r, rm, p, mask = inputs
    c0 = operator.c0.expand_as(c)
    r0 = operator.r0.expand_as(r)
    rm0 = operator.rm0.expand_as(rm)
    k = operator.k(p)
    out = []
    for cc, rr, mm in [(c, r, rm), (c, r0, rm0),
                       (c0, r, rm), (c0, r0, rm0)]:
        q = operator.queries(cc, rr, mm)[0]
        out.append(operator.attention(q, k, mask))
    a11, a10, a01, a00 = out
    return (a11 - a10) - (a01 - a00)


@pytest.mark.parametrize("arm", ["A4", "A5", "A6", "A7"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_clinical_and_rna_references_are_exact_nulls(arm, dtype):
    operator, inputs = example(arm, dtype=dtype)
    c, r, rm, p, mask = inputs
    # The RNA null includes missingness; replacing only expression is not R0.
    for null_inputs in [
        (operator.c0.expand_as(c), r, rm, p, mask),
        (c, operator.r0.expand_as(r), operator.rm0.expand_as(rm), p, mask),
    ]:
        delta = four_maps(operator, null_inputs)
        assert torch.count_nonzero(delta) == 0
        risk, diagnostics = operator(*null_inputs, diagnostic=True)
        assert torch.count_nonzero(risk) == 0
        assert torch.count_nonzero(diagnostics["delta_norm"]) == 0


def test_rna_missing_indicator_is_part_of_conditioning():
    operator, inputs = example()
    c, r, rm, p, mask = inputs
    # Deliberately hold the values at R0 but retain each patient's missingness.
    risk = operator(c, operator.r0.expand_as(r), rm, p, mask)
    assert risk.abs().max() > 1e-10


def test_signed_attention_rows_sum_to_zero_and_masked_columns_are_zero():
    operator, inputs = example()
    c, r, rm, p, mask = inputs
    mask[0, -2:] = False
    mask[1, -1] = False
    delta = four_maps(operator, (c, r, rm, p, mask))
    assert delta.abs().max() > 1e-8  # Do not pass through an identically dead map.
    torch.testing.assert_close(delta.sum(-1), torch.zeros_like(delta.sum(-1)),
                               rtol=0, atol=1e-14)
    invalid = (~mask)[:, None, None, :].expand_as(delta)
    assert torch.count_nonzero(delta[invalid]) == 0


def test_fixed_attention_message_is_invariant_to_constant_value_shift():
    operator, inputs = example()
    delta = four_maps(operator, inputs)
    v = operator.v(inputs[3])
    # Hold Q/K and all maps fixed: no claim of raw-patch translation invariance.
    shift = torch.randn(len(v), 1, v.shape[-1], dtype=v.dtype)
    torch.testing.assert_close(operator.message(delta, v + shift),
                               operator.message(delta, v), rtol=0, atol=1e-13)


@pytest.mark.parametrize("arm", ["A4", "A5", "A7"])
def test_single_patch_and_constant_values_cannot_produce_mixed_message(arm):
    single, inputs = example(arm, patches=1)
    assert torch.count_nonzero(single(*inputs)) == 0
    operator, inputs = example(arm)
    delta = four_maps(operator, inputs)
    constant_v = torch.randn(len(inputs[0]), 1, 64, dtype=inputs[0].dtype)
    constant_v = constant_v.expand(-1, inputs[3].shape[1], -1)
    message = operator.message(delta, constant_v)
    torch.testing.assert_close(message, torch.zeros_like(message), rtol=0, atol=1e-13)


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4", "A5", "A7", "F0", "PR"])
def test_padding_and_masked_payload_do_not_change_predictions(arm):
    operator, inputs = example(arm)
    c, r, rm, p, mask = inputs
    baseline = operator(*inputs)
    garbage = torch.randn(len(c), 4, 64, dtype=p.dtype) * 1000
    padded = torch.cat([p, garbage], 1)
    padded_mask = torch.cat([mask, torch.zeros(len(c), 4, dtype=torch.bool)], 1)
    torch.testing.assert_close(operator(c, r, rm, padded, padded_mask), baseline,
                               rtol=0, atol=2e-13)


def test_empty_patch_patient_is_rejected():
    operator, inputs = example()
    c, r, rm, p, mask = inputs
    mask[0] = False
    with pytest.raises(ValueError, match="no valid patch"):
        operator(c, r, rm, p, mask)
    with pytest.raises(ValueError, match="no valid patch"):
        masked_softmax(torch.zeros(3, 4, 512, 7), mask)


@pytest.mark.parametrize("arm", ["A1", "A2", "A3", "A4", "A5", "A7", "F0", "PR"])
def test_patient_permutation_is_equivariant(arm):
    operator, inputs = example(arm)
    permutation = torch.tensor([2, 0, 1])
    permuted = tuple(value[permutation] for value in inputs)
    torch.testing.assert_close(operator(*permuted), operator(*inputs)[permutation],
                               rtol=0, atol=1e-14)


def test_all_pairwise_branches_cannot_read_the_third_modality():
    operator, inputs = example("A1")
    seen = {}
    branch_outputs = {}

    def capture(name):
        def hook(_module, arguments):
            seen[name] = arguments[0].detach().clone()
        return hook

    def capture_output(name):
        def hook(_module, _arguments, output):
            branch_outputs[name] = output.detach().clone().squeeze(-1)
        return hook

    handles = [operator.pair_heads[0].register_forward_pre_hook(capture("CP")),
               operator.pair_heads[1].register_forward_pre_hook(capture("RP")),
               operator.cr.register_forward_pre_hook(capture("CR")),
               operator.pair_heads[0].register_forward_hook(capture_output("CP")),
               operator.pair_heads[1].register_forward_hook(capture_output("RP")),
               operator.cr.register_forward_hook(capture_output("CR"))]
    try:
        result = operator(*inputs)
        expected = operator.bound / 3 * sum(torch.tanh(value) for value in branch_outputs.values())
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-14)
        # A nonlinear transform AFTER summing all branches would introduce a
        # forbidden all-modal mixer even though each branch is individually pure.
        forbidden = operator.bound * torch.tanh(sum(branch_outputs.values()))
        assert (expected - forbidden).abs().max() > 1e-10
        original = {key: value.clone() for key, value in seen.items()}
        c, r, rm, p, mask = inputs
        operator(c, r + 5, 1 - rm, p, mask)
        assert torch.equal(original["CP"], seen["CP"])
        operator(c + 5, r, rm, p, mask)
        assert torch.equal(original["RP"], seen["RP"])
        operator(c, r, rm, p + 5, mask)
        assert torch.equal(original["CR"], seen["CR"])
    finally:
        for handle in handles:
            handle.remove()


def test_wrong_rna_changes_only_interaction_not_own_main_effect():
    operator, inputs = example()
    c, r, rm, p, mask = inputs
    data = SimpleNamespace(f={"C": np.zeros((len(c), c.shape[1]))},
                           C0=operator.c0, R0=operator.r0, RM0=operator.rm0)
    config = {"training": {"hidden": 64, "heads": 4, "interaction_risk_bound": .25}}
    x = torch.cat([c, r], -1)
    w = torch.randn(x.shape[1], dtype=torch.float64) * .01
    model = Model(data, "A5", w, config).double()
    permutation = torch.tensor([2, 0, 1])
    true_risk = model(x, c, r, rm, p, mask)
    wrong_risk = model(x, c, r[permutation], rm[permutation], p, mask)
    own_main = x @ model.main
    true_interaction = model.interaction(c, r, rm, p, mask)
    wrong_interaction = model.interaction(c, r[permutation], rm[permutation], p, mask)
    torch.testing.assert_close(true_risk - true_interaction, own_main, rtol=0, atol=1e-14)
    torch.testing.assert_close(wrong_risk - wrong_interaction, own_main, rtol=0, atol=1e-14)
    assert (true_interaction - wrong_interaction).abs().max() > 1e-10


def test_query_clinical_rna_gate_and_value_paths_receive_finite_nonzero_gradient():
    operator, inputs = example()
    c, r, rm, p, mask = inputs
    c = c.clone().requires_grad_()
    r = r.clone().requires_grad_()
    risk = operator(c, r, rm, p, mask)
    # Asymmetric weights avoid an accidental patient-sum cancellation.
    (risk * torch.tensor([.5, -1.1, 1.7], dtype=risk.dtype)).sum().backward()
    for name in ["rna.weight", "c.0.weight", "c.1.weight", "q.weight",
                 "k.weight", "v.weight", "gate.weight", "out.weight"]:
        grad = dict(operator.named_parameters())[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().max() > 1e-12, name
    for grad in [c.grad, r.grad]:
        assert torch.isfinite(grad).all()
        assert grad.abs().max() > 1e-12


def test_full_efron_with_ties_matches_independent_manual_denominators():
    time = np.array([1., 1., 1., 2., 3., 3., 5.])
    event = np.array([1, 1, 0, 1, 1, 0, 0])
    risk = torch.tensor([.8, -.1, .5, -.4, .2, .6, -.7], dtype=torch.float64)
    expected = -risk[torch.as_tensor(event).bool()].sum()
    for when in np.unique(time[event == 1]):
        deaths = (time == when) & (event == 1)
        n_deaths = int(deaths.sum())
        risk_sum = risk[torch.as_tensor(time >= when)].exp().sum()
        death_sum = risk[torch.as_tensor(deaths)].exp().sum()
        for j in range(n_deaths):
            expected = expected + (risk_sum - j / n_deaths * death_sum).log()
    expected = expected / event.sum()
    torch.testing.assert_close(full_risk_efron(risk, time, event), expected,
                               rtol=0, atol=1e-14)


def test_exact_two_pass_gradient_matches_full_risk_efron_ties_and_one_l2():
    torch.manual_seed(71)
    network = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(),
                                  torch.nn.Linear(7, 1, bias=False)).double()
    x = torch.randn(11, 5, dtype=torch.float64)
    # Include both tied deaths and event/censor ties; chunks cut across ties.
    time = np.array([1., 1., 1., 2., 3., 3., 4., 5., 5., 6., 7.])
    event = np.array([1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    parameters = tuple(network.parameters())
    lam = .17

    def regularizer():
        return lam / 2 * sum(parameter.square().sum() for parameter in parameters)

    full_risk = network(x).squeeze(-1)
    objective = full_risk_efron(full_risk, time, event) + regularizer()
    full_grad = torch.autograd.grad(objective, parameters)
    with torch.no_grad():
        all_risk = torch.cat([network(chunk).squeeze(-1) for chunk in x.split(3)])
    eta = all_risk.detach().requires_grad_()
    coefficient = torch.autograd.grad(full_risk_efron(eta, time, event), eta)[0].detach()
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    for start in range(0, len(x), 3):
        prediction = network(x[start:start + 3]).squeeze(-1)
        term = (prediction * coefficient[start:start + 3]).sum()
        grads = torch.autograd.grad(term, parameters)
        accumulated = [acc + grad for acc, grad in zip(accumulated, grads)]
    reg_grad = torch.autograd.grad(regularizer(), parameters)
    for expected, actual, penalty in zip(full_grad, accumulated, reg_grad):
        torch.testing.assert_close(actual + penalty, expected, rtol=2e-13, atol=1e-14)


def test_data_rejects_invalid_role_and_predictions_outside_active_partition():
    from src.trimodal_interaction.data import Data

    # Invalid roles/folds must fail before trying to open private cohort files.
    with pytest.raises(ValueError, match="invalid preprocessing role/fold"):
        Data(1, "unknown", 31)
    with pytest.raises(ValueError, match="invalid preprocessing role/fold"):
        Data(0, "inner", 31)
    wrapper = Data.__new__(Data)
    wrapper.active = np.array([1, 3, 7], dtype=np.int64)
    np.testing.assert_array_equal(wrapper.check_indices([7, 1]), [7, 1])
    with pytest.raises(ValueError, match="outside active role partition"):
        wrapper.check_indices([1, 2])


def test_source_guard_rejects_changed_parent_training_hash(monkeypatch):
    from src.trimodal_interaction import data

    monkeypatch.setattr(data, "cfg", lambda: {"parent_training_source": "pinned-parent"})
    monkeypatch.setattr(data.old, "source", lambda: "changed-parent")
    with pytest.raises(ValueError, match="parent Phase17 training source changed"):
        data.source()


def test_visual_registry_rejects_changed_visual_hash(monkeypatch):
    from src.trimodal_interaction import data

    receipt = {"status": "PASS", "patients": 512,
               "hashes": {"conch": "expected-conch-hash"}}
    class SyntheticRegistryPath:
        def __truediv__(self, _name):
            return self

        def read_text(self):
            return json.dumps(receipt)

    monkeypatch.setattr(data, "OLD_VISUAL", SyntheticRegistryPath())
    monkeypatch.setattr(data, "digest", lambda _path: "modified-conch-hash")
    with pytest.raises(ValueError, match="historical visual hash changed: conch"):
        data._visual_registry()


# Source/partition tests above exercise fail-closed branches with synthetic
# registries.  They do NOT claim the real private files or patient splits pass;
# that separate audit must be run against the server's actual source registry.

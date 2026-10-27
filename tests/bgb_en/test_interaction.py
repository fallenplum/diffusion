"""Synthetic R3 mechanism and intervention isolation tests."""
import pytest
import torch

from src.bgb_en.interaction import ARMS, BidirectionalInteraction, masked_softmax


def fixture(n=11, k=5):
    torch.manual_seed(9033001)
    p = torch.randn(3, 2, n, 12)
    g = torch.randn(3, k, 7)
    pm = torch.ones(p.shape[:-1], dtype=torch.bool)
    gm = torch.ones(g.shape[:-1], dtype=torch.bool)
    pm[1, 1] = False
    pm[0, 1, -2:] = False
    gm[2, -1] = False
    return p, pm, g, gm


def model(arm):
    return BidirectionalInteraction(arm, patch_dim=12, omics_dim=7, width=8, rank=4, topk=4)


@pytest.mark.parametrize("arm", ARMS)
def test_forward_gradient_empty_and_padded_values(arm):
    p, pm, g, gm = fixture()
    p.requires_grad_(True); g.requires_grad_(True)
    net = model(arm)
    output = net(p, pm, g, gm)
    assert output["delta_scalar"].shape == (3,)
    assert torch.isfinite(output["delta_scalar"]).all()
    output["delta_scalar"].square().sum().backward()
    assert p.grad[pm].abs().sum() > 0 and g.grad[gm].abs().sum() > 0
    assert p.grad[~pm].abs().sum() == 0 and g.grad[~gm].abs().sum() == 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in net.parameters())
    corrupted_p = p.detach().clone(); corrupted_p[~pm] = float("nan")
    corrupted_g = g.detach().clone(); corrupted_g[~gm] = float("nan")
    torch.testing.assert_close(output["delta_scalar"], net(corrupted_p, pm, corrupted_g, gm)["delta_scalar"])
    empty = net(p.detach(), torch.zeros_like(pm), g.detach(), torch.zeros_like(gm))
    assert torch.equal(empty["delta_scalar"], torch.zeros(3))


@pytest.mark.parametrize("arm", ARMS)
def test_interaction_off_exactly_recovers_same_main_effects(arm):
    p, pm, g, gm = fixture()
    net = model(arm)
    active = net(p, pm, g, gm)
    off = net(p, pm, g, gm, interaction_off=True)
    assert torch.equal(active["pooled_path_risk"], off["pooled_path_risk"])
    assert torch.equal(active["pooled_omics_risk"], off["pooled_omics_risk"])
    assert torch.equal(off["interaction_risk"], torch.zeros(3))
    assert torch.equal(off["delta_scalar"], off["pooled_path_risk"]+off["pooled_omics_risk"])


@pytest.mark.parametrize("arm", ["F3", "F5", "F6", "F7", "F10"])
def test_query_only_swap_preserves_own_keys_values_and_main_effects(arm):
    p, pm, g, gm = fixture()
    net = model(arm)
    original = net(p, pm, g, gm)
    swap = torch.tensor([1, 2, 0])
    changed = net(p, pm, g, gm, query_omics=g[swap], query_omics_mask=gm[swap])
    assert torch.equal(original["pooled_path_risk"], changed["pooled_path_risk"])
    assert torch.equal(original["pooled_omics_risk"], changed["pooled_omics_risk"])
    assert torch.equal(original["path_initial"], changed["path_initial"])
    assert torch.equal(original["omics_initial"], changed["omics_initial"])
    assert not torch.allclose(original["OtoP_risk"], changed["OtoP_risk"], atol=1e-8, rtol=1e-6)
    whole = net(p, pm, g, gm, whole_omics=g[swap], whole_omics_mask=gm[swap])
    assert not torch.equal(original["pooled_omics_risk"], whole["pooled_omics_risk"])
    assert torch.equal(original["pooled_path_risk"], whole["pooled_path_risk"])


def test_direction_isolation_and_shared_two_rounds():
    p, pm, g, gm = fixture()
    f3 = model("F3")(p, pm, g, gm)
    f4 = model("F4")(p, pm, g, gm)
    assert torch.equal(f3["PtoO_risk"], torch.zeros(3))
    assert torch.equal(f3["path_initial"], f3["path_updated"])
    assert torch.equal(f4["OtoP_risk"], torch.zeros(3))
    assert torch.equal(f4["omics_initial"], f4["omics_updated"])
    f5, f6 = model("F5"), model("F6")
    assert f5.parameter_count() == f6.parameter_count()
    f6.load_state_dict(f5.state_dict())
    one, two = f5(p, pm, g, gm), f6(p, pm, g, gm)
    assert len(one["trace"]) == 1 and len(two["trace"]) == 2
    assert not torch.equal(one["interaction_risk"], two["interaction_risk"])
    loss = two["OtoP_risk"].square().sum()+two["PtoO_risk"].square().sum()
    loss.backward()
    assert f6.path_value.weight.grad.abs().sum() > 0
    assert f6.omics_value.weight.grad.abs().sum() > 0
    assert f6.path_match.weight.grad.abs().sum() > 0
    assert f6.omics_match.weight.grad.abs().sum() > 0


def test_value_fixed_attention_assignment_permutation_preserves_weight_spectrum():
    p, pm, g, gm = fixture()
    net = model("F5")
    base = net(p, pm, g, gm)
    permutation = torch.arange(pm.shape[-1]).expand_as(pm).clone()
    for batch in range(len(pm)):
        for slide in range(pm.shape[1]):
            real = torch.where(pm[batch, slide])[0]
            permutation[batch, slide, real] = real.roll(1)
    changed = net(p, pm, g, gm, attention_patch_permutation=permutation)
    original_attention = base["trace"][0]["omics_to_patch"]["attention"]
    changed_attention = changed["trace"][0]["omics_to_patch"]["attention"]
    torch.testing.assert_close(original_attention.sort(-1).values, changed_attention.sort(-1).values)
    assert not torch.equal(base["OtoP_risk"], changed["OtoP_risk"])
    assert torch.equal(base["PtoO_risk"], changed["PtoO_risk"])
    assert torch.equal(base["pooled_path_risk"], changed["pooled_path_risk"])
    assert torch.equal(base["pooled_omics_risk"], changed["pooled_omics_risk"])


def test_slot_competition_fixed_topk_keeps_only_actual_patch_values():
    p, pm, g, gm = fixture(n=19)
    output = model("F7")(p, pm, g, gm)
    weight = output["trace"][0]["omics_to_patch"]["attention"]
    assert ((weight > 0).sum(-1) <= 4).all()
    assert weight.masked_select(~pm[:, :, None, :].expand_as(weight)).abs().sum() == 0
    sums = weight.sum(-1)
    valid = pm.any(-1)[:, :, None] & gm[:, None, :]
    torch.testing.assert_close(sums[valid], torch.ones_like(sums[valid]))
    assert sums[~valid].abs().sum() == 0


def test_null_gate_zero_value_and_no_evidence_gives_no_directional_correction():
    p, pm, g, gm = fixture()
    net = model("F10")
    output = net(p, pm, g, gm)
    for direction in ("omics_to_patch", "path_to_omics"):
        row = output["trace"][0][direction]
        total = row["attention"].sum(-1)+row["null"]
        assert ((total == 0) | torch.isclose(total, torch.ones_like(total))).all()
        assert ((row["gate"] > 0) & (row["gate"] < 1)).all()
    empty_path = net(p, torch.zeros_like(pm), g, gm)
    empty_omics = net(p, pm, g, torch.zeros_like(gm))
    assert torch.equal(empty_path["interaction_risk"], torch.zeros(3))
    assert torch.equal(empty_omics["interaction_risk"], torch.zeros(3))


def test_equal_slide_aggregation_no_cross_slide_state_and_clinical_unused():
    p, pm, g, gm = fixture()
    net = model("F5")
    base = net(p, pm, g, gm)
    clinical = net(p, pm, g, gm, clinical=torch.randn(3, 20))
    assert torch.equal(base["delta_scalar"], clinical["delta_scalar"])
    permutation = torch.tensor([1, 0])
    moved = net(p[:, permutation], pm[:, permutation], g, gm)
    torch.testing.assert_close(base["delta_scalar"], moved["delta_scalar"], atol=1e-7, rtol=1e-6)


def test_pooled_main_initialization_identical_across_arm_construction():
    p, pm, g, gm = fixture()
    results = []
    for arm in ARMS:
        torch.manual_seed(31)
        results.append(model(arm)(p, pm, g, gm))
    for row in results[1:]:
        assert torch.equal(row["pooled_path_risk"], results[0]["pooled_path_risk"])
        assert torch.equal(row["pooled_omics_risk"], results[0]["pooled_omics_risk"])


def test_interventions_fail_closed_for_padding_exchange_and_mixed_control():
    p, pm, g, gm = fixture()
    net = model("F5")
    invalid = torch.arange(pm.shape[-1]).expand_as(pm).clone()
    invalid[0, 1, [0, -1]] = invalid[0, 1, [-1, 0]]
    with pytest.raises(ValueError, match="real values and padding"):
        net(p, pm, g, gm, attention_patch_permutation=invalid)
    with pytest.raises(ValueError, match="provided together"):
        net(p, pm, g, gm, whole_omics=g)
    with pytest.raises(ValueError, match="must be separate"):
        net(p, pm, g, gm, whole_omics=g, whole_omics_mask=gm, query_omics=g, query_omics_mask=gm)
    with pytest.raises(ValueError, match="own mask"):
        net(p, pm, g, gm, query_omics=g)


def test_empty_softmax_has_zero_gradient_without_nan():
    score = torch.randn(3, 4, requires_grad=True)
    probability = masked_softmax(score, torch.zeros_like(score, dtype=torch.bool))
    assert torch.equal(probability, torch.zeros_like(probability))
    probability.sum().backward()
    assert torch.isfinite(score.grad).all() and score.grad.abs().sum() == 0


def test_root_trainer_context_and_risk_aliases():
    p, pm, g, gm = fixture()
    net = model("F5")
    swap = torch.tensor([1, 2, 0])
    explicit = net(p, pm, g, gm, query_omics=g[swap], query_omics_mask=gm[swap])
    alias = net(p, pm, g, gm, context_omics=(g[swap], gm[swap]))
    assert torch.equal(explicit["risk"], alias["risk"])
    assert torch.equal(alias["risk"], alias["delta_scalar"])
    base = net(p, pm, g, gm)
    permuted = net(p, pm, g, gm, permute_attention=True)
    assert not torch.equal(base["OtoP_risk"], permuted["OtoP_risk"])
    assert torch.equal(base["PtoO_risk"], permuted["PtoO_risk"])

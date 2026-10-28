"""R5 readout identities and patient/value/mask isolation; no survival fits."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.bgb_en.r5_readout import (MODES, RiskReadout, R5MechanismModel,
                                  evidence_summaries)
from src.bgb_en.training import MechanismModel


def interaction_fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(34031)
    pm = torch.ones(4, 2, 5, dtype=torch.bool)
    gm = torch.ones(4, 3, dtype=torch.bool)
    pm[0, 1, -2:] = False
    pm[1, 1] = False
    gm[2, -1] = False
    p = torch.randn(4, 2, 5, 32, generator=generator, dtype=dtype)*.1
    g = torch.randn(4, 3, 32, generator=generator, dtype=dtype)*.1
    pd = torch.randn(p.shape, generator=generator, dtype=dtype)*.01
    gd = torch.randn(g.shape, generator=generator, dtype=dtype)*.01
    output = dict(path_initial=p, path_updated=p+pd, omics_initial=g, omics_updated=g+gd,
                  pooled_path_risk=torch.randn(4, generator=generator, dtype=dtype),
                  pooled_omics_risk=torch.randn(4, generator=generator, dtype=dtype))
    heads = (nn.Linear(32, 1, bias=False).to(dtype), nn.Linear(32, 1, bias=False).to(dtype))
    summaries = evidence_summaries(output, pm, gm)
    output['OtoP_risk'] = heads[0](summaries['summary_omics_update']).squeeze(-1)
    output['PtoO_risk'] = heads[1](summaries['summary_path_update']).squeeze(-1)
    return output, pm, gm, heads


def model_fixture():
    generator = np.random.default_rng(201031)
    patients, genes, slots, patches, dim = 6, 9, 2, 5, 12
    rna = generator.normal(size=(patients, genes)).astype('float32')
    rna_mask = np.ones_like(rna, dtype=bool)
    rna_mask[:3, 6:] = False
    rna_mask[4, :3] = False
    rna[~rna_mask] = np.nan
    membership = np.zeros((3, genes), bool)
    membership[0, :3] = True; membership[1, 3:6] = True; membership[2, 6:] = True
    features = dict(rna=rna, rna_mask=rna_mask,
                    cna_gain=np.abs(generator.normal(size=rna.shape)).astype('float32'),
                    cna_loss=np.abs(generator.normal(size=rna.shape)).astype('float32'),
                    cna_mask=np.ones_like(rna_mask), C=generator.normal(size=(patients, 2)).astype('float32'))
    prepared = dict(features=features, train=np.arange(3), valid=np.arange(3, 6),
                    RNA=generator.normal(size=patients).astype('float32'), beta=.2,
                    base=generator.normal(size=patients).astype('float32'))
    x = torch.from_numpy(generator.normal(size=(patients, slots, patches, dim)).astype('float32'))
    mask = torch.ones(patients, slots, patches, dtype=torch.bool)
    mask[0, 1, -2:] = False; mask[5, 1] = False
    xy = torch.zeros(patients, slots, patches, 2, dtype=torch.long)
    xy[..., 0] = torch.arange(patches)*1024
    bank = SimpleNamespace(x=x, xy=xy, mask=mask,
                           region_ids=torch.zeros_like(mask, dtype=torch.long), step=1024., input_dim=dim)
    return prepared, membership, bank


def build(arm='A3', **kwargs):
    prepared, membership, bank = model_fixture()
    torch.manual_seed(31)
    return R5MechanismModel(arm, prepared, membership, bank, 'cpu', **kwargs)


def test_legacy_preserves_original_float32_operation_order():
    output, pm, gm, _ = interaction_fixture(torch.float32)
    output['OtoP_risk'] = torch.tensor([-20., -.04, .04, 20.])
    output['PtoO_risk'] = torch.tensor([0., .01, -.02, 0.])
    readout = RiskReadout('legacy')
    expected = .1*(.25*torch.tanh(output['pooled_path_risk'])
                  +.25*torch.tanh(output['pooled_omics_risk'])
                  +.5*torch.tanh(output['OtoP_risk']+output['PtoO_risk']))
    assert torch.equal(readout(output, pm, gm)['delta'], expected)
    assert not readout.state_dict()


@pytest.mark.parametrize('mode', ('legacy', 'linear', 'temp4', 'additive'))
def test_scalar_derivative_is_the_actual_autograd_derivative(mode):
    model = RiskReadout(mode)
    z = torch.linspace(-20, 20, 81, dtype=torch.float64, requires_grad=True)
    response = model.scalar_response(z)
    if mode == 'additive':
        assert torch.equal(response, torch.zeros_like(z))
    else:
        actual = torch.autograd.grad(response.sum(), z)[0]
        torch.testing.assert_close(actual, model.scalar_derivative(z), atol=1e-14, rtol=1e-12)
    assert model.scalar_derivative(torch.zeros(1)).item() == pytest.approx(0 if mode == 'additive' else .05)


def test_temperature_moves_saturation_without_changing_zero_slope():
    legacy, temp, linear = (RiskReadout(name) for name in ('legacy', 'temp4', 'linear'))
    z = torch.tensor([0., 8., 20.], dtype=torch.float64)
    assert temp.scalar_derivative(z)[1] > legacy.scalar_derivative(z)[1]
    assert torch.equal(temp.scalar_derivative(z[:1]), legacy.scalar_derivative(z[:1]))
    assert linear.scalar_response(z)[-1] > temp.scalar_response(z)[-1] > legacy.scalar_response(z)[-1]


def test_directional_prior_is_centered_at_one_and_both_directions_receive_gradient():
    output, pm, gm, _ = interaction_fixture()
    model = RiskReadout('directional').double()
    assert model.coefficient_regularization().item() == 0
    result = model(output, pm, gm)
    result['cross_delta'].square().sum().backward()
    assert model.a_o.grad.abs() > 0 and model.a_p.grad.abs() > 0
    with torch.no_grad(): model.a_o.fill_(1.5)
    assert model.coefficient_regularization().item() == pytest.approx(1e-4*.25)
    d_o, d_p = model.directional_derivatives(output['OtoP_risk'])
    torch.testing.assert_close(d_o, torch.full_like(d_o, .075))
    torch.testing.assert_close(d_p, torch.full_like(d_p, .05))


def test_rank4_initialization_is_linear_not_an_asserted_global_tanh_identity():
    output, pm, gm, heads = interaction_fixture()
    rank = RiskReadout('rank4').double()
    rank.initialize_rank4_from_directional(*heads)
    actual = rank(output, pm, gm)['cross_delta']
    expected = .05*(output['OtoP_risk']+output['PtoO_risk'])
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-12)
    assert sum(p.numel() for p in rank.parameters()) == 260
    with pytest.raises(ValueError, match='no unique'):
        rank.scalar_response(torch.ones(1))
    with pytest.raises(ValueError, match='summary Jacobian'):
        rank.directional_derivatives(torch.ones(1))


def test_rank4_summary_jacobian_matches_autograd_after_weights_change():
    rank = RiskReadout('rank4').double()
    with torch.no_grad(): rank.up.weight.normal_(std=.2)
    x = torch.randn(4, 64, dtype=torch.float64, requires_grad=True)
    y = .05*rank.up(torch.nn.functional.gelu(rank.down(x))).sum()
    actual = torch.autograd.grad(y, x)[0]
    torch.testing.assert_close(rank.summary_jacobian(x), actual, atol=1e-14, rtol=1e-12)


def test_rank4_uses_only_update_differences_and_never_padded_values():
    output, pm, gm, heads = interaction_fixture()
    rank = RiskReadout('rank4').double(); rank.initialize_rank4_from_directional(*heads)
    baseline = rank(output, pm, gm)['cross_delta']
    changed = {k:v.clone() for k,v in output.items()}
    for key, mask in (('path_initial', pm), ('path_updated', pm), ('omics_initial', gm), ('omics_updated', gm)):
        changed[key][~mask] = float('nan')
    assert torch.equal(rank(changed, pm, gm)['cross_delta'], baseline)
    # Changing both initial and updated by the same vector does not change evidence deltas.
    for initial, updated in (('path_initial', 'path_updated'), ('omics_initial', 'omics_updated')):
        noise = torch.randn_like(output[initial])*.01
        changed[initial] = output[initial]+noise; changed[updated] = output[updated]+noise
    torch.testing.assert_close(rank(changed, pm, gm)['cross_delta'], baseline, atol=1e-15, rtol=1e-12)


def test_sham_derived_whole_model_budget_and_forward_additivity():
    output, pm, gm, _ = interaction_fixture()
    sham = RiskReadout('sham').double()
    assert sum(p.numel() for p in sham.parameters()) == 4875
    with torch.no_grad(): sham.up.weight.normal_(std=.2)
    # A mixed second derivative cannot flow from a path input into the omics branch.
    output['path_initial'].requires_grad_(True); output['omics_initial'].requires_grad_(True)
    result = sham(output, pm, gm)
    path_gradient = torch.autograd.grad(result['sham_delta'].sum(), output['path_initial'], create_graph=True)[0]
    cross_gradient = torch.autograd.grad(path_gradient.sum(), output['omics_initial'], allow_unused=True)[0]
    assert cross_gradient is None
    assert torch.equal(result['cross_delta'], torch.zeros_like(result['cross_delta']))


def test_all_shared_backbone_tensors_identical_across_arm_construction_and_legacy_replay():
    prepared, membership, bank = model_fixture()
    torch.manual_seed(31)
    old = MechanismModel('R3', 'F5', prepared, membership, bank, 'cpu')
    reference = old.state_dict()
    for arm in ('A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'):
        torch.manual_seed(31)
        model = R5MechanismModel(arm, prepared, membership, bank, 'cpu')
        shared = {k:v for k,v in model.state_dict().items() if not k.startswith('readout.')}
        assert set(shared) == set(reference)
        assert all(torch.equal(shared[k], reference[k]) for k in reference)
        if arm == 'A3': assert torch.equal(model([0, 3, 4])['risk'], old([0, 3, 4])['risk'])
        if arm == 'A7':
            assert not model.interaction.otop_head.weight.requires_grad
            assert not model.interaction.ptoo_head.weight.requires_grad
        if arm in ('A2', 'A8'):
            assert not model.interaction.path_match.weight.requires_grad


def test_sham_whole_model_active_budget_is_within_seven_parameters_of_rank4():
    additive, rank4, sham = build('A2'), build('A7'), build('A8')
    base = additive.parameter_budget()['requires_grad']
    assert rank4.parameter_budget()['requires_grad']-base == 4868
    assert sham.parameter_budget()['requires_grad']-base == 4875
    assert sham.parameter_budget()['requires_grad']-rank4.parameter_budget()['requires_grad'] == 7


def test_query_only_and_explicit_matched_override_preserve_own_evidence_and_offset():
    model = build('A4', intervention='wrong_query')
    model.query_map[:6] = np.array([1, 2, 0, 4, 5, 3])
    indices = [3, 4, 5]
    correct = model(indices, control='matched_query')
    wrong = model(indices)
    for key in ('main_delta', 'pool_p', 'pool_o'):
        assert torch.equal(correct[key], wrong[key])
    for key in ('path_initial', 'omics_initial'):
        assert torch.equal(correct['interaction_details'][key], wrong['interaction_details'][key])
    assert not torch.equal(correct['z_o'], wrong['z_o'])
    torch.testing.assert_close(wrong['risk']-wrong['delta'], model.offset[indices], atol=2e-7, rtol=0)
    source = model.omics(**{k:v[torch.tensor([4, 5, 3])] for k,v in model.inputs.items()})
    weight = wrong['interaction_details']['trace'][0]['omics_to_patch']['attention']
    unsupported = ~source['token_mask'][:, None, :, None].expand_as(weight)
    assert weight[unsupported].abs().sum() == 0


def test_mean_query_and_mask_are_train_local_and_fixed_for_cache_replays():
    model = build('A4', intervention='query_mean')
    mean, mask = model.refresh_training_query_mean(batch=2)
    assert not mean.requires_grad and not mask[:, -1].any()
    expected = model.omics(**{k:v[torch.tensor([0, 1, 2])] for k,v in model.inputs.items()})
    counts = expected['token_mask'].sum(0, keepdim=True)
    value = torch.where(expected['token_mask'][..., None], expected['tokens'], 0.).sum(0, keepdim=True)
    torch.testing.assert_close(mean, value/counts.clamp_min(1)[..., None])
    before = model([3, 4, 5])
    replay = model([3, 4, 5])
    assert torch.equal(before['risk'], replay['risk'])
    with torch.no_grad():
        model.inputs['rna'][3:] = 99.
        model.inputs['rna_mask'][3:] = True
    later, later_mask = model.refresh_training_query_mean(batch=1)
    torch.testing.assert_close(mean, later)
    assert torch.equal(mask, later_mask)
    assert before['interaction_details']['trace'][0]['omics_to_patch']['attention'][:, :, -1].abs().sum() == 0


@pytest.mark.parametrize('arm', ('A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'))
def test_frozen_interaction_off_preserves_main_and_is_strictly_zero(arm):
    model = build(arm)
    if arm == 'A8':
        with torch.no_grad(): model.readout.up.weight.fill_(.2)
    on = model([0, 3, 4]); off = model([0, 3, 4], control='interaction_off')
    assert torch.equal(on['main_delta'], off['main_delta'])
    assert torch.equal(on['sham_delta'], off['sham_delta'])
    assert torch.equal(off['cross_delta'], torch.zeros_like(off['cross_delta']))
    if off['scalar_derivative'] is not None:
        assert torch.equal(off['scalar_derivative'], torch.zeros_like(off['scalar_derivative']))


def test_oneway_direction_and_missing_query_effect_are_not_confused():
    model = build('A4', direction='P2O')
    model.query_map[:6] = np.array([1, 2, 0, 4, 5, 3])
    baseline = model([3, 4, 5]); wrong = model([3, 4, 5], control='wrong_query')
    assert torch.equal(baseline['z_o'], torch.zeros_like(baseline['z_o']))
    assert torch.equal(baseline['cross_delta'], wrong['cross_delta'])
    assert not model.interaction.path_value.weight.requires_grad
    assert model.interaction.omics_value.weight.requires_grad


def test_legacy_checkpoint_loading_and_control_validation_fail_closed():
    prepared, membership, bank = model_fixture()
    torch.manual_seed(31); old = MechanismModel('R3', 'F5', prepared, membership, bank, 'cpu')
    rank = build('A7'); result = rank.load_legacy_backbone(old.state_dict())
    assert result['new_readout_tensors'] == 2
    with pytest.raises(ValueError, match='checkpoint keys'):
        rank.load_legacy_backbone({})
    with pytest.raises(ValueError, match='unknown R5'):
        rank([0], control='typo_query')

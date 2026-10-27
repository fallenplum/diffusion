import numpy as np
import torch
from src.bgb_en.mechanism_audit import (attention_change, attention_summary, bounded_components,
                                      independent_efron_gradient, loss_chain_probe)


def test_two_direction_readout_shares_sum_to_actual_bounded_risk():
    output = {name: torch.tensor([-.7, .1, 2.]) for name in ('OtoP_risk', 'pooled_path_risk', 'pooled_omics_risk')}
    output['PtoO_risk'] = torch.tensor([.4, -.8, 3.])
    components = bounded_components(output, .1)
    torch.testing.assert_close(components['OtoP_SharedReadout']+components['PtoO_SharedReadout'], components['interaction'])


def test_independent_cox_gradient_matches_finite_difference_with_ties():
    t, e, r = np.array([2., 2., 3., 4.]), np.array([1, 1, 0, 1]), np.array([.1, -.2, .8, .3])
    audit = independent_efron_gradient(t, e, r)
    epsilon = 1e-5
    finite = []
    for i in range(len(r)):
        up, down = r.copy(), r.copy(); up[i] += epsilon; down[i] -= epsilon
        finite.append((independent_efron_gradient(t, e, up)['loss']-independent_efron_gradient(t, e, down)['loss'])/(2*epsilon))
    np.testing.assert_allclose(audit['gradient'], finite, atol=1e-8, rtol=0)
    assert abs(audit['gradient_sum']) < 1e-12


def test_attention_tv_effective_rank_and_null_are_defined_separately():
    row = dict(attention=torch.tensor([[[[.8, 0.], [0., .8]]]]), null=torch.full((1, 1, 2), .2),
               gate=torch.full((1, 2, 3), .5), message=torch.zeros(1, 2, 3))
    summary = attention_summary(row)
    assert abs(summary['relative_effective_keys']['mean']-1) < 1e-6
    assert abs(summary['query_key_effective_rank']['mean']-2) < 1e-6
    assert abs(summary['null_mass']['mean']-.2) < 1e-6
    other = {**row, 'attention': row['attention'].flip(-1)}
    assert abs(attention_change(row, other)['TV']['mean']-.8) < 1e-6


def test_chain_gradient_excludes_l2_only_and_pooled_parameters():
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.route = torch.nn.Parameter(torch.tensor(.2)); self.unused = torch.nn.Parameter(torch.tensor(100.))
    model = Probe()
    rna = torch.tensor([.3, .5], requires_grad=True)
    patch = torch.tensor([.4, .8], requires_grad=True)
    actual = .1*torch.tanh(model.route*rna*patch)
    result = loss_chain_probe(actual, np.array([.6, -.4]), model, rna, patch)
    by_name = {row['name']: row for row in result['parameter_gradients']}
    assert by_name['route']['connected'] and by_name['route']['gradient_L2'] > 0
    assert not by_name['unused']['connected'] and by_name['unused']['gradient_L2'] == 0
    assert result['input_gradients']['RNA_input']['gradient_L2'] > 0
    assert result['input_gradients']['patch_input']['gradient_L2'] > 0


def test_complete_cpu_replay_on_actual_f5_instance_without_optimization():
    from types import SimpleNamespace
    from scripts.bgb_en_mechanism_replay import replay_cell, grouped_mechanisms
    from src.bgb_en.training import MechanismModel
    torch.manual_seed(719)
    rng = np.random.default_rng(719)
    features = dict(rna=rng.normal(size=(8, 12)).astype(np.float32), rna_mask=np.ones((8, 12), bool),
                    cna_gain=np.zeros((8, 12), np.float32), cna_loss=np.zeros((8, 12), np.float32),
                    cna_mask=np.ones((8, 12), bool), C=np.zeros((8, 2), np.float32))
    prepared = dict(features=features, train=np.arange(4), valid=np.arange(4, 8), base=np.zeros(8),
                    RNA=np.zeros(8), CS=np.zeros(8), beta=1.)
    bank = SimpleNamespace(x=torch.randn(8, 1, 6, 768), mask=torch.ones(8, 1, 6, dtype=torch.bool),
        xy=torch.zeros(8, 1, 6, 2), region_ids=torch.zeros(8, 1, 6, dtype=torch.long), step=1., input_dim=768)
    bank.xy[..., 0] = torch.arange(6)
    membership = torch.zeros(3, 12, dtype=torch.bool)
    for index in range(3):
        membership[index, index*4:(index+1)*4] = True
    model = MechanismModel('R3', 'F5', prepared, membership, bank, 'cpu').eval()
    model.query_map[:8] = np.r_[np.roll(np.arange(4), 1), np.roll(np.arange(4, 8), 1)]
    with torch.no_grad():
        risk = model(torch.arange(4, 8))['risk'].numpy()
    raw = dict(t=np.array([1., 4., 8., 12., 2., 5., 9., 11.]), e=np.array([1, 0, 1, 1, 1, 0, 1, 1]))
    checkpoint = dict(snapshots={80: dict(risk=risk)}, frozen_controls={})
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = replay_cell(model, prepared, raw, checkpoint, 8)
    assert result['status'] == 'PASS_FINAL80_MECHANISM_REPLAY'
    assert result['optimizer_updates'] == 0 and result['diagnostic_validation_patients'] == 4
    assert result['sampled_final80_risk_replay_max_abs'] < 1e-6
    assert result['loss_chain_gradients']['interaction']['input_gradients']['RNA_input']['gradient_L2'] > 0
    assert result['loss_chain_gradients']['interaction']['input_gradients']['patch_input']['gradient_L2'] > 0
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    result['identity'] = dict(arm='F5', seed=31, partition=0)
    grouped = grouped_mechanisms([result], ('arm',))[0]
    gradients = grouped['pure_interaction_loss_gradients']['interaction']
    assert gradients['input_gradients']['RNA_input']['nonzero_cells'] == 1
    route_norm = gradients['parameter_groups']['routing_kernel']['gradient_L2_across_cells']['mean']
    assert route_norm > 0
    assert gradients['excluded_pooled_head_check']['gradient_L2_across_cells']['max_abs'] == 0
    assert grouped['last_layer_attention']['omics_to_patch']['cells_with_direction'] == 1
    assert grouped['last_layer_attention']['omics_to_patch']['query_key_effective_rank_cell_mean']['n'] == 1
    assert grouped['readout_derivative_and_amplitude']['bounded_interaction_readout_derivative_cell_mean']['n'] == 1
    # A leaked pooled gradient must be visible but never inflate route evidence.
    result['loss_chain_gradients']['interaction']['parameter_gradients'].append(
        dict(name='interaction.pooled_path_head.fake', parameters=1, connected=True, gradient_L2=1e8, gradient_max_abs=1e8))
    altered = grouped_mechanisms([result], ('arm',))[0]['pure_interaction_loss_gradients']['interaction']
    assert altered['parameter_groups']['routing_kernel']['gradient_L2_across_cells']['mean'] == route_norm
    assert altered['excluded_pooled_head_check']['gradient_L2_across_cells']['max_abs'] == 1e8

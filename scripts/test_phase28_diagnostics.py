"""Synthetic-only qualification of the optional post-fit diagnostics.

This is outside the locked qualification suite and does not revise its receipt.
No patient input, file-backed risk vector, fitted weight or optimizer is used.
"""
import copy

import numpy as np
import pytest
import torch

from scripts.diagnose_phase28 import (aggregate_patient_diagnostics, detached_route_context,
    diagnose_encoded, history_summary, moments, route_summary, trace_summary, weight_change)
from tests.test_phase28_model import fixture


torch.set_num_threads(1)


def fake_record():
    rows = []
    for epoch in range(1, 81):
        rows.append({'epoch': epoch, 'cox_loss': 5-epoch/100, 'regularizer': .3,
            'objective': 5.3-epoch/100, 'risk_RMS': .2+epoch/100,
            'main_residual_RMS': .1, 'interaction_RMS': epoch/10000,
            'clipped_from_gradient_L2': float(epoch > 40)*6.,
            'maximum_absolute_component_replay_error': 0.,
            'data_gradient_L2_by_top_level_module': {'up': .3, 'delta_gene': .2},
            'total_gradient_L2_by_top_level_module': {'up': .4, 'delta_gene': .5}})
    base = np.asarray([.1, -.2, .4, -.3])
    interaction = np.asarray([.01, .02, -.01, -.02])
    return {'history': rows, 'fit_components': {'risk': base+interaction, 'base': base,
        'interaction': interaction, 'path_main': np.zeros(4), 'gene_main': base},
        'training_audit': {'final_fit_Efron_NLL': 4.19}}


def test_history_tracks_only_training_and_retains_pre_step_boundary():
    record = fake_record()
    class Forbidden:
        def __array__(self, *args): raise AssertionError('heldout risk must not be accessed')
    record['valid_risks'] = Forbidden()
    result = history_summary(record)
    assert len(result['curves']['objective']) == 80
    assert result['last10_descriptive_trends']['cox_loss']['last10_OLS_slope_per_epoch'] == pytest.approx(-.01)
    assert result['gradient_clip_fraction'] == .5
    assert result['learning_curve_records_are_pre_step']
    assert not result['checkpoint_or_epoch_selected_by_this_analysis']


def test_history_rejects_partial_training():
    record = fake_record(); record['history'].pop()
    with pytest.raises(ValueError, match='eighty'):
        history_summary(record)


def test_history_rejects_inconsistent_fit_decomposition():
    record = fake_record(); record['fit_components']['risk'][0] += .1
    with pytest.raises(ValueError, match='decomposition'):
        history_summary(record)


def test_weight_changes_account_for_singletons_and_empty_queries():
    support = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    base = torch.tensor([[1., 0, 0], [.5, .5, 0], [0, 0, 0]])
    cond = torch.tensor([[1., 0, 0], [.7, .3, 0], [0, 0, 0]])
    out = weight_change(base, cond, support)
    assert out['TV_mean_all_valid_queries'] == pytest.approx(.1)
    assert out['TV_mean_queries_ge2_candidates'] == pytest.approx(.2)
    assert out['eligible_query_fraction_ge2_candidates'] == .5
    assert out['singleton_TV_maximum'] == 0


def test_weight_all_singletons_have_no_eligible_effect_statistic():
    out = weight_change(torch.eye(2), torch.eye(2), torch.eye(2).bool())
    assert out['TV_mean_queries_ge2_candidates'] is None
    assert out['singleton_TV_maximum'] == 0


@pytest.mark.parametrize('arm', ['C00', 'C10', 'C01', 'C11', 'DIRECT'])
def test_diagnostic_graph_works_for_all_direction_switches_without_weight_change(arm):
    model, raw, mask, ids, slides = fixture(arm)
    original = copy.deepcopy(model.state_dict())
    encoded = model.encode(raw, mask, ids, slides)
    row, output = diagnose_encoded(model, raw, encoded, .7)
    assert len(row['layers']) == 2
    assert row['gradients']['isolated_gene_context_to_path_risk_L2'] >= 0
    assert row['gradients']['isolated_path_context_to_gene_risk_L2'] >= 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert not any(p.grad is not None for p in model.parameters())
    if arm == 'C00':
        assert row['risk']['interaction'] == 0
        assert row['gradients']['isolated_gene_context_to_path_risk_L2'] == 0
        assert row['gradients']['isolated_path_context_to_gene_risk_L2'] == 0
    if arm == 'C10':
        assert row['risk']['gene_interaction'] == 0
    if arm == 'C01':
        assert row['risk']['path_interaction'] == 0


def test_route_common_score_energy_and_null_measurements_have_defined_bounds():
    model, raw, mask, ids, slides = fixture()
    encoded = model.encode(raw, mask, ids, slides)
    output = model.read_encoded(encoded, 0., details=True)
    route = output['trace'][0]['own_routing_reference'][0]
    result = route_summary(route, encoded['path_valid'][0], encoded['gene_valid'])
    assert 0 <= result['score_row_centered_energy_fraction'] <= 1+1e-12
    assert result['path_target_null_probability_mean']+result['path_target_real_probability_mass_mean'] == pytest.approx(1.)
    assert result['gene_target_null_probability_mean']+result['gene_target_real_probability_mass_mean'] == pytest.approx(1.)


def test_two_round_risk_decomposition_uses_final_state_for_second_round():
    model, raw, mask, ids, slides = fixture()
    encoded = model.encode(raw, mask, ids, slides)
    output = model.read_encoded(encoded, 0., details=True)
    trace = trace_summary(model, encoded, output)
    assert trace[-1]['path_risk_correction'] == pytest.approx(float(output['path_interaction']))
    assert trace[-1]['gene_risk_correction'] == pytest.approx(float(output['gene_interaction']))


def test_three_patient_aggregation_contains_no_per_patient_vectors():
    model, raw, mask, ids, slides = fixture()
    row, _ = diagnose_encoded(model, raw, model.encode(raw, mask, ids, slides), .7)
    out = aggregate_patient_diagnostics([copy.deepcopy(row) for _ in range(3)])
    assert out['patients'] == 3 and len(out['layers']) == 2
    assert out['risk_component_distributions']['interaction']['SD'] == pytest.approx(0.)
    assert 'risk' not in out and 'patient_ids' not in out


def test_scalar_summaries_fail_closed_on_nonfinite_input():
    with pytest.raises(ValueError, match='nonfinite'):
        moments([1., np.nan])

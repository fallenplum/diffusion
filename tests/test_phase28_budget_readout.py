"""Checkpoint-only diagnostics on algebraic arrays; no patient/model input."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('phase28_budget_readout',
    Path(__file__).resolve().parents[1]/'scripts/audit_phase28_budget_readout.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def synthetic(arm='C11'):
    fit_cp, fit_rna = np.array([-.2, .1, .3]), np.array([-.1, .2, 0.])
    valid_cp, valid_rna = np.array([-.1, .3]), np.array([.2, -.2])
    fit_offset, valid_offset = fit_cp+fit_rna, valid_cp+valid_rna
    fp, fg = np.array([.01, -.02, .03], np.float32), np.array([.02, .01, -.01], np.float32)
    vp, vg = np.array([.02, -.01], np.float32), np.array([-.01, .03], np.float32)
    fi = np.array([.001, -.002, .003]) if arm != 'C00' else np.zeros(3)
    vpi = np.array([.001, -.002], np.float32) if arm != 'C00' else np.zeros(2, np.float32)
    vgi = np.array([.002, .001], np.float32) if arm != 'C00' else np.zeros(2, np.float32)
    fit_base, valid_base = fit_offset+(fp+fg).astype(np.float64), valid_offset+(vp+vg).astype(np.float64)
    fit_risk, valid_risk = fit_base+fi, valid_base+(vpi+vgi).astype(np.float64)
    history = [{'epoch': epoch, 'cox_loss': 3.-epoch*.001, 'regularizer': .1,
        'objective': 3.1-epoch*.001, 'interaction_RMS': epoch*1e-5, 'main_residual_RMS': .02,
        'risk_RMS': .4, 'clipped_from_gradient_L2': 6. if epoch % 2 else 2.,
        'data_gradient_L2_by_top_level_module': {'dormant': 0., 'active': .03},
        'total_gradient_L2_by_top_level_module': {'dormant': .01, 'active': .04},
        'parameters_with_nonzero_data_gradient': 100, 'parameters_with_nonzero_total_gradient': 120,
        'maximum_absolute_component_replay_error': 0.} for epoch in range(1, 81)]
    controls = {name: valid_risk.copy() for name in ('MATCHED', 'OFF', 'GENE_WRONG')}
    controls['OFF'] = valid_base.copy()
    reference = {'fit': np.array([0, 1, 2]), 'valid': np.array([3, 4]),
        'CP_cf_fit': fit_cp, 'RNA_cf_fit': fit_rna, 'CP_valid': valid_cp,
        'RNA_valid': valid_rna, 'M0_valid': valid_offset}
    record = {'seed': 31, 'fold': 1, 'arm': arm, 'fit': reference['fit'], 'valid': reference['valid'],
        'fit_risk': fit_risk, 'fit_components': {'base': fit_base, 'interaction': fi,
            'path_main': fp, 'gene_main': fg}, 'M0_valid': valid_offset, 'CP_valid': valid_cp,
        'RNA_valid': valid_rna, 'valid_base_risk': valid_base, 'valid_components': np.column_stack((vp, vg, vpi, vgi)),
        'valid_risks': controls, 'history': history, 'model_parameters': {'allocated_trainable': 120},
        'training_audit': {'final_fit_Efron_NLL': 2.8}}
    return record, reference


def test_zero_interaction_is_reported_without_invented_correlation_or_scale():
    record, reference = synthetic('C00')
    result = audit.summarize_record(record, reference)
    assert result['risk_scales']['fit']['interaction']['RMS'] == 0.
    assert result['risk_scales']['fit']['interaction_correlation_with_external'] is None
    assert result['fit_to_valid_SD_ratios']['interaction_valid_over_fit'] is None
    assert result['risk_scales']['valid']['total_vs_own_main_all_pair_rank_changes']['strict_flip_fraction'] == 0.


def test_budget_and_data_gradient_are_separate_from_l2_gradient():
    record, _ = synthetic()
    result = audit.history_summary(record['history'])
    assert result['actual_epochs'] == 80
    assert result['gradient_clipping']['fraction_preclip_above_threshold'] == .5
    assert result['data_vs_total_gradient_by_module']['dormant']['epochs_nonzero_data_gradient'] == 0
    assert result['data_vs_total_gradient_by_module']['dormant']['last_total_L2'] == .01
    assert result['cox_loss']['last10_linear_slope_per_epoch'] == pytest.approx(-.001)
    assert result['convergence_or_budget_sufficiency_claimed'] is False


def test_saved_fit_cf_vs_valid_fullfit_scale_is_unpaired_diagnostic():
    record, reference = synthetic()
    result = audit.summarize_record(record, reference)
    expected = np.std(reference['M0_valid'])/np.std(reference['CP_cf_fit']+reference['RNA_cf_fit'])
    assert result['fit_to_valid_SD_ratios']['external_CP_plus_RNA_valid_over_fit'] == pytest.approx(expected)
    assert result['fit_CF_offset_and_valid_fullfit_offset_use_different_patients'] is True
    assert result['fit_valid_scale_difference_is_not_a_paired_calibration_error'] is True
    assert result['maximum_risk_decomposition_error'] < 1e-15


def test_pair_rank_changes_are_not_called_survival_repair():
    result = audit.all_pair_rank_change(np.array([1., 2., 3.]), np.array([3., 2., 1.]))
    assert result['strict_flip_fraction'] == 1.
    assert result['unordered_pairs'] == 3
    assert result['uses_survival_outcomes'] is False
    assert result['is_survival_repair_or_damage'] is False


@pytest.mark.parametrize('tamper', ['main', 'interaction', 'partition', 'reference'])
def test_component_and_reference_drift_fail_closed(tamper):
    record, reference = synthetic()
    if tamper == 'main': record['valid_components'][0, 0] += .2
    if tamper == 'interaction': record['fit_components']['interaction'][0] += .2
    if tamper == 'partition': reference['valid'] = reference['valid'][::-1]
    if tamper == 'reference': reference['M0_valid'] = reference['M0_valid']+.1
    with pytest.raises(ValueError): audit.summarize_record(record, reference)


def test_cross_arm_total_delta_closes_with_joint_main_effect_drift():
    records = {arm: synthetic(arm)[0] for arm in ('C00', 'C10', 'C01', 'C11', 'DIRECT')}
    records['C11']['valid_base_risk'] += .02
    records['C11']['valid_risks']['MATCHED'] += .02
    result = audit.compare_arms(records)
    row = next(row for row in result if row['comparison'] == 'C11-C00')
    assert row['main_effect_change']['RMS'] == pytest.approx(.02)
    assert row['maximum_decomposition_error'] < 1e-15
    assert row['joint_training_allows_main_effect_drift'] is True
    assert row['is_survival_performance_comparison'] is False


def test_history_missing_epoch_or_nonfinite_fails():
    record, _ = synthetic()
    with pytest.raises(ValueError): audit.history_summary(record['history'][:-1])
    bad = copy.deepcopy(record['history']); bad[4]['cox_loss'] = float('nan')
    with pytest.raises(ValueError): audit.history_summary(bad)


def test_saved_float32_heads_keep_original_sum_order_at_large_offsets():
    record, reference = synthetic()
    p = np.array([64., 32., 8.], np.float32)
    g = np.full(3, 2.**-18, np.float32)
    expected = (p+g).astype(np.float64)
    naive = p.astype(np.float64)+g.astype(np.float64)
    assert np.max(np.abs(expected-naive)) > 1e-6
    record['fit_components']['path_main'], record['fit_components']['gene_main'] = p, g
    record['fit_components']['base'] = reference['CP_cf_fit']+reference['RNA_cf_fit']+expected
    record['fit_risk'] = record['fit_components']['base']+record['fit_components']['interaction']
    result = audit.summarize_record(record, reference)
    assert result['maximum_risk_decomposition_error'] < 1e-15
    np.testing.assert_array_equal(audit.model_head_sum(p, g), expected)


def test_head_sum_rejects_unrepresentable_serialized_components():
    with pytest.raises(ValueError, match='original Float32'):
        audit.model_head_sum(np.array([0.1], np.float64), np.array([0.2], np.float64))

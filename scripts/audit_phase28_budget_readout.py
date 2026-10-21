"""Outcome-blind, checkpoint-only Phase28 budget and risk-scale diagnostics.

This optional post-fit audit performs no forward pass, parameter update,
endpoint request, risk rescaling search, or additional admission decision.
All thirty immutable fits and their independent inference replay must pass
before this script can deserialize diagnostic components. It writes only
anonymous aggregates; patient risks, indices and model tensors stay private.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, name, count=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == 1 and result.size > 0 and np.isfinite(result).all(),
            'finite nonempty one-dimensional diagnostic required: '+name)
    require(count is None or len(result) == count, 'diagnostic component count differs: '+name)
    return result


def moments(value):
    x = vector(value, 'moments')
    return {'count': len(x), 'mean': float(x.mean()), 'SD': float(x.std()),
            'RMS': float(np.sqrt(np.mean(x*x))), 'abs_q95': float(np.quantile(np.abs(x), .95)),
            'min': float(x.min()), 'max': float(x.max())}


def correlation(left, right):
    a = vector(left, 'correlation left')
    b = vector(right, 'correlation right', len(a))
    a, b = a-a.mean(), b-b.mean()
    denominator = float(np.sqrt(np.dot(a, a)*np.dot(b, b)))
    return None if denominator == 0 else float(np.dot(a, b)/denominator)


def sd_ratio(numerator, denominator):
    a, b = float(np.std(vector(numerator, 'scale numerator'))), float(np.std(vector(denominator, 'scale denominator')))
    return None if b == 0 else a/b


def close(left, right, name):
    a, b = vector(left, name+' left'), vector(right, name+' right')
    require(a.shape == b.shape and np.max(np.abs(a-b)) <= 1e-6, 'risk decomposition failed: '+name)
    return float(np.max(np.abs(a-b)))


def model_head_sum(left, right):
    """Reproduce model.py: add Float32 heads first, then promote to Float64.

    Checkpoints serialize individual Float32 scalar heads as Python floats.
    Adding those serialized numbers directly in Float64 changes the operation.
    """
    a, b = vector(left, 'left saved head'), vector(right, 'right saved head')
    require(a.shape == b.shape, 'head shapes differ')
    require(np.array_equal(a, a.astype(np.float32).astype(np.float64))
            and np.array_equal(b, b.astype(np.float32).astype(np.float64)),
            'saved model head is not an exact original Float32 scalar')
    return (a.astype(np.float32)+b.astype(np.float32)).astype(np.float64)


def all_pair_rank_change(baseline, candidate):
    """Endpoint-free rank perturbation, explicitly not survival repair/damage."""
    a = vector(baseline, 'pair baseline'); b = vector(candidate, 'pair candidate', len(a))
    i, j = np.triu_indices(len(a), 1)
    require(len(i) > 0, 'two patients required for rank diagnostic')
    d0, d1 = a[i]-a[j], b[i]-b[j]
    sign0 = np.where(np.abs(d0) <= 1e-8, 0, np.sign(d0))
    sign1 = np.where(np.abs(d1) <= 1e-8, 0, np.sign(d1))
    return {'unordered_pairs': len(i), 'strict_flip_fraction': float(np.mean(sign0*sign1 == -1)),
            'any_rank_state_change_fraction': float(np.mean(sign0 != sign1)),
            'baseline_tied_fraction': float(np.mean(sign0 == 0)),
            'candidate_tied_fraction': float(np.mean(sign1 == 0)),
            'uses_survival_outcomes': False, 'is_survival_repair_or_damage': False}


def history_summary(history):
    require(isinstance(history, list) and len(history) == 80
            and [row.get('epoch') for row in history] == list(range(1, 81)), 'exact eighty-epoch history required')
    result = {'actual_epochs': 80, 'fixed_final_epoch': True,
              'validation_based_epoch_selection': False, 'convergence_or_budget_sufficiency_claimed': False}
    for key in ('cox_loss', 'regularizer', 'objective', 'interaction_RMS', 'main_residual_RMS', 'risk_RMS'):
        x = vector([row[key] for row in history], 'history '+key, 80)
        result[key] = {'first_preupdate': float(x[0]), 'last_preupdate': float(x[-1]),
                       'minimum_preupdate': float(x.min()), 'minimum_epoch_diagnostic_only': int(x.argmin()+1),
                       'last10_preupdate_change': float(x[-1]-x[-11]),
                       'last10_linear_slope_per_epoch': float(np.polyfit(np.arange(11), x[-11:], 1)[0])}
    norms = vector([row['clipped_from_gradient_L2'] for row in history], 'unclipped gradient norms', 80)
    result['gradient_clipping'] = {'fixed_threshold': 5., 'fraction_preclip_above_threshold': float(np.mean(norms > 5.)),
                                   'unclipped_gradient_L2': moments(norms)}
    modules = sorted(set().union(*(row['data_gradient_L2_by_top_level_module'] for row in history)))
    result['data_vs_total_gradient_by_module'] = {}
    for module in modules:
        data = vector([row['data_gradient_L2_by_top_level_module'].get(module, 0.) for row in history], 'data gradient')
        total = vector([row['total_gradient_L2_by_top_level_module'].get(module, 0.) for row in history], 'total gradient')
        result['data_vs_total_gradient_by_module'][module] = {
            'first_data_L2': float(data[0]), 'last_data_L2': float(data[-1]), 'max_data_L2': float(data.max()),
            'epochs_nonzero_data_gradient': int(np.count_nonzero(data)), 'last_total_L2': float(total[-1])}
    for key in ('parameters_with_nonzero_data_gradient', 'parameters_with_nonzero_total_gradient'):
        values = vector([row[key] for row in history], key, 80)
        result[key] = {'first': int(values[0]), 'last': int(values[-1]), 'min': int(values.min()), 'max': int(values.max())}
    result['maximum_gradient_cache_component_error'] = max(float(row['maximum_absolute_component_replay_error']) for row in history)
    require(result['maximum_gradient_cache_component_error'] <= 1e-6, 'gradient cache error exceeds lock')
    return result


def summarize_record(record, reference):
    """Pure aggregation; synthetic harnesses need no real weights or outcomes."""
    fit, valid = np.asarray(record['fit']), np.asarray(record['valid'])
    require(np.array_equal(fit, reference['fit']) and np.array_equal(valid, reference['valid']), 'reference partition differs')
    fit_n, valid_n = len(fit), len(valid)
    fit_offset = vector(reference['CP_cf_fit'], 'fit CF CP', fit_n)+vector(reference['RNA_cf_fit'], 'fit CF RNA', fit_n)
    valid_offset = vector(record['M0_valid'], 'valid fullfit offset', valid_n)
    require(np.array_equal(valid_offset, vector(reference['M0_valid'], 'reference M0', valid_n)), 'external M0 changed')
    fit_components = record['fit_components']
    fit_total = vector(record['fit_risk'], 'fit risk', fit_n)
    fit_base = vector(fit_components['base'], 'fit own learned main', fit_n)
    fit_interaction = vector(fit_components['interaction'], 'fit interaction', fit_n)
    valid_total = vector(record['valid_risks']['MATCHED'], 'valid risk', valid_n)
    valid_base = vector(record['valid_base_risk'], 'valid own learned main', valid_n)
    valid_interaction = valid_total-valid_base
    components = np.asarray(record['valid_components'], dtype=np.float64)
    require(components.shape == (valid_n, 4) and np.isfinite(components).all(), 'four valid branch components required')
    reconstruction_error = max(close(fit_total, fit_base+fit_interaction, 'fit total'),
        close(valid_total, valid_base+model_head_sum(components[:, 2], components[:, 3]), 'valid interaction'),
        close(fit_base, fit_offset+model_head_sum(fit_components['path_main'], fit_components['gene_main']), 'fit main'),
        close(valid_base, valid_offset+model_head_sum(components[:, 0], components[:, 1]), 'valid main'))
    summary = {'seed': record['seed'], 'fold': record['fold'], 'arm': record['arm'],
        'allocated_parameter_counts': record['model_parameters'], 'training': history_summary(record['history']),
        'final_postupdate_fit_Efron_NLL_saved_by_training': float(record['training_audit']['final_fit_Efron_NLL']),
        'maximum_risk_decomposition_error': reconstruction_error,
        'decomposition_arithmetic': 'Float32_head_pair_sum_then_Float64_offset_addition_as_locked_model',
        'risk_scales': {}, 'frozen_control_risk_response': {},
        'fit_CF_offset_and_valid_fullfit_offset_use_different_patients': True,
        'fit_valid_scale_difference_is_not_a_paired_calibration_error': True}
    for scope, total, base, offset, interaction, pmain, gmain in (
            ('fit', fit_total, fit_base, fit_offset, fit_interaction, fit_components['path_main'], fit_components['gene_main']),
            ('valid', valid_total, valid_base, valid_offset, valid_interaction, components[:, 0], components[:, 1])):
        residual = base-offset
        summary['risk_scales'][scope] = {name: moments(value) for name, value in (
            ('external_CP_plus_RNA', offset), ('learned_path_main', pmain), ('learned_gene_main', gmain),
            ('learned_main_residual', residual), ('own_learned_main', base), ('interaction', interaction), ('total', total))}
        summary['risk_scales'][scope].update(interaction_SD_over_external_SD=sd_ratio(interaction, offset),
            interaction_SD_over_learned_main_SD=sd_ratio(interaction, base),
            interaction_correlation_with_external=correlation(interaction, offset),
            interaction_correlation_with_learned_main_residual=correlation(interaction, residual),
            total_vs_own_main_all_pair_rank_changes=all_pair_rank_change(base, total))
    summary['risk_scales']['valid']['path_interaction'] = moments(components[:, 2])
    summary['risk_scales']['valid']['gene_interaction'] = moments(components[:, 3])
    for component in ('CP', 'RNA'):
        train = vector(reference[component+'_cf_fit'], 'fit '+component, fit_n)
        validation = vector(reference[component+'_valid'], 'valid '+component, valid_n)
        require(np.array_equal(validation, vector(record[component+'_valid'], 'saved '+component, valid_n)), 'saved main reference differs')
        summary['risk_scales']['fit']['external_'+component] = moments(train)
        summary['risk_scales']['valid']['external_'+component] = moments(validation)
    summary['fit_to_valid_SD_ratios'] = {
        'external_CP_plus_RNA_valid_over_fit': sd_ratio(valid_offset, fit_offset),
        'interaction_valid_over_fit': sd_ratio(valid_interaction, fit_interaction),
        'learned_main_residual_valid_over_fit': sd_ratio(valid_base-valid_offset, fit_base-fit_offset)}
    for name, risk in record['valid_risks'].items():
        values = vector(risk, name, valid_n)
        summary['frozen_control_risk_response'][name] = {
            'matched_minus_control': moments(valid_total-values),
            'risk_correlation': correlation(valid_total, values),
            'all_pair_rank_changes': all_pair_rank_change(values, valid_total),
            'condition_only': name not in ('OFF', 'WHOLE_OMICS_WRONG')}
    return summary


def compare_arms(records):
    """Decompose trained-risk differences; do not call them pure interactions."""
    answer = []
    for candidate, baseline in (('C11', 'C00'), ('C11', 'DIRECT'), ('C11', 'C10'), ('C11', 'C01')):
        a, b = records[candidate], records[baseline]
        require(np.array_equal(a['valid'], b['valid']), 'cross-arm patient order differs')
        total_a, total_b = (vector(row['valid_risks']['MATCHED'], 'cross-arm risk') for row in (a, b))
        main_a, main_b = (vector(row['valid_base_risk'], 'cross-arm own main') for row in (a, b))
        main_delta = main_a-main_b
        interaction_delta = (total_a-main_a)-(total_b-main_b)
        total_delta = total_a-total_b
        error = close(total_delta, main_delta+interaction_delta, 'cross-arm total delta')
        answer.append({'comparison': candidate+'-'+baseline, 'seed': a['seed'], 'fold': a['fold'],
            'main_effect_change': moments(main_delta), 'interaction_change': moments(interaction_delta),
            'total_change': moments(total_delta), 'maximum_decomposition_error': error,
            'main_and_interaction_change_correlation': correlation(main_delta, interaction_delta),
            'all_pair_rank_changes': all_pair_rank_change(total_b, total_a),
            'joint_training_allows_main_effect_drift': True,
            'is_survival_performance_comparison': False})
    return answer


def _reference_main_only(path, expected_sha, sha):
    """No label key is requested; legacy pickle includes unused fit-label bytes."""
    require(sha(path) == expected_sha, 'frozen prepared SHA changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    keys = ('fit', 'valid', 'CP_cf_fit', 'RNA_cf_fit', 'CP_valid', 'RNA_valid', 'M0_valid')
    answer = {key: np.array(saved[key]) for key in keys}
    del saved
    require(sha(path) == expected_sha, 'frozen prepared changed during risk reference read')
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--execute-audit', action='store_true')
    args = parser.parse_args()
    require(args.execute_audit, 'explicit --execute-audit required after independent replay')
    from src.phase28_reread import runtime, data
    from src.phase28_reread.evaluation import verify_stage, verify_qualifications, verify_replay
    source = args.source
    require(runtime.source() == source, 'requested training source differs')
    runtime.verify_qualifications(source)
    source_files = runtime.source_files()
    qualification = verify_qualifications(runtime.REPORT, source, source_files, project_root=ROOT, private_root=runtime.PRIVATE)
    _, artifact = verify_stage(runtime.PRIVATE, 'prototype', source)
    replay = verify_replay(runtime.REPORT/'REPLAY_AUDIT.json', source, artifact, qualification, source_files, project_root=ROOT)
    destination = runtime.REPORT/'BUDGET_READOUT_AUDIT.json'
    require(not destination.exists(), 'existing anonymous budget report must not be overwritten')
    script_sha = runtime.sha(Path(__file__))
    counters = {'np_load_calls': 0, 'optimizer_constructions': 0, 'backward_calls': 0, 'autograd_grad_calls': 0}
    def deny(label):
        def fail(*a, **k):
            counters[label] += 1
            raise RuntimeError('checkpoint-only audit forbids '+label)
        return fail
    rows, cross_arm = [], []
    with ExitStack() as stack:
        for owner, name, label in ((np, 'load', 'np_load_calls'),
                (torch.optim.Optimizer, '__init__', 'optimizer_constructions'),
                (torch.Tensor, 'backward', 'backward_calls'), (torch.autograd, 'grad', 'autograd_grad_calls')):
            stack.enter_context(patch.object(owner, name, deny(label)))
        for seed in (31, 47):
            for fold in (1, 2, 3):
                records = {}
                for arm in ('C00', 'C10', 'C01', 'C11', 'DIRECT'):
                    directory = runtime.cell_dir(seed, fold, arm)
                    receipt = runtime.verify_done(directory, source, seed, fold, arm)
                    record = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
                    reference = _reference_main_only(data.PREPARED_ROOT/f'prototype_fold{fold}.pt', record['prepared_sha256'], runtime.sha)
                    rows.append(summarize_record(record, reference)); records[arm] = record
                    require(runtime.sha(directory/'checkpoint.pt') == receipt['checkpoint_sha256'], 'checkpoint changed during audit')
                cross_arm.extend(compare_arms(records))
                del records
    require(not any(counters.values()) and len(rows) == 30, 'diagnostic coverage/isolation failed')
    require(runtime.source() == source and runtime.source_files() == source_files, 'training closure changed during audit')
    require(runtime.sha(Path(__file__)) == script_sha, 'diagnostic executable changed during audit')
    require(runtime.sha(runtime.REPORT/'REPLAY_AUDIT.json') == replay['report_sha256'], 'replay receipt changed during audit')
    for name, row in qualification.items():
        require(runtime.sha(runtime.REPORT/(name+'.json')) == row['report_sha256'], 'qualification receipt changed during audit')
    for binding in artifact['model_bindings']:
        directory = runtime.cell_dir(binding['seed'], binding['fold'], binding['arm'])
        require(runtime.sha(directory/'checkpoint.pt') == binding['checkpoint_sha256']
                and runtime.sha(directory/'DONE_PRIVATE.json') == binding['DONE_sha256'], 'saved model/receipt changed during audit')
    payload = {'schema': 'phase28_checkpoint_budget_readout_v1', 'status': 'PASS_OUTCOME_BLIND_CHECKPOINT_AGGREGATION',
        'source': source, 'script_sha256': script_sha, 'replay_report_sha256': replay['report_sha256'],
        'receipt_fingerprint': artifact['receipt_fingerprint'], 'neural_models': 30, 'rows': rows,
        'cross_arm_risk_decompositions': cross_arm, 'forbidden_API_call_counts': counters,
        'new_predictor_fits': 0, 'new_optimizer_steps': 0, 'new_forward_calls': 0,
        'endpoint_array_requests': 0, 'fit_label_bytes_deserialized_in_legacy_prepared': True,
        'label_keys_inspected_or_used': False, 'rescaling_coefficients_searched': False,
        'admission_gate_changed': False, 'global_convergence_or_budget_sufficiency_claimed': False,
        'patient_risks_indices_or_model_tensors_public': False,
        'risk_rank_changes_without_outcomes_are_not_survival_repairs': True}
    runtime.atomic_json(destination, payload)
    print('PHASE28_BUDGET_READOUT_AUDIT', payload['status'], len(rows), flush=True)


if __name__ == '__main__':
    main()

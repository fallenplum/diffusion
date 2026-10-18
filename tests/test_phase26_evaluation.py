"""Synthetic receipts and known-pair oracles; no patient data or training."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.phase26_multiquery import evaluation as ev

SOURCE = 'synthetic-phase26-source'


def moment(count, value=0.):
    return {'count': count, 'mean': value, 'population_SD': 0., 'RMS': abs(value),
            'minimum': value, 'maximum': value}


def mechanism(nfit, nvalid):
    return {'fit_unit_residual': moment(nfit), 'valid_unit_residual': moment(nvalid),
            'attention': {k: moment(nvalid*8) for k in
                          ('normalized_entropy', 'effective_patch_fraction', 'TV_from_uniform')},
            'slot': {k: moment(nvalid) for k in
                     ('pairwise_cosine', 'effective_rank', 'message_over_query_norm', 'assignment_fraction')},
            'frozen_query_changes': {control: {
                'patient_risk_difference_over_fit_offset_SD': moment(nvalid),
                **({k: moment(nvalid*8) for k in ('attention_TV', 'top32_Jaccard')}
                   if control not in ('OFF', 'VALUE_OFF') else {})} for control in ev.CONTROLS}}


def record(seed, fold):
    valid = np.arange((fold-1)*4, fold*4)
    fit = np.setdiff1d(np.arange(12), valid)
    main = np.arange(4, dtype=float)*.02
    out = {'source': SOURCE, 'stage': 'prototype', 'seed': seed, 'fold': fold,
           'fit': fit, 'valid': valid, 'M0_valid': main, 'CP_valid': np.zeros(4), 'RNA_valid': main,
           'data_audit': {'prepared_sha256': 'prepared-'+str(fold)}, 'arms': {}}
    shifts = {'QUERY': np.array([.01, .002, -.002, -.01]),
              'ATTN': np.array([.04, .02, -.02, -.04]), 'COMP': np.array([.09, .03, -.03, -.09])}
    for arm in ev.NEURAL_ARMS:
        risk = main+shifts[arm]
        control = {k: risk.copy() if arm == 'QUERY' else main.copy() for k in ev.CONTROLS}
        control['OFF'] = main.copy()
        out['arms'][arm] = {
            'model_state': {'weight': torch.tensor([1.])}, 'initial_state': {'weight': torch.tensor([0.])},
            'clinical_dim': 33, 'fit_risk': np.linspace(-.1, .1, len(fit))+.001, 'risk': risk,
            'history': [{'epoch': i, 'training_loss': 2.-i/1000, 'fit_loss': 2.-i/1000,
                         'fit_cox': 1.9-i/1000, 'unit_residual_RMS': .01, 'correction_RMS': .001,
                         'saturation_fraction': 0., 'grad_norm': .1,
                         'gradients': {k: .1 if k == 'head' else 0. for k in ev.GRADIENT_GROUPS},
                         'epoch_seconds': .01} for i in range(1, 81)],
            'training_audit': {'actual_epochs': 80, 'optimizer_steps': 80, 'heldout_endpoint_reads': 0,
                               'fixed_final_epoch': True, 'main_offset_unchanged': True,
                               'fit_offset_kind': 'strict_threefold_crossfit_M0',
                               'heldout_offset_kind': 'full_fit_M0', 'bound': .1, 'query_residual_preserved': True},
            'parameter_counts': {'trainable_local': 1, 'common_frozen_main': 2},
            'sensitivities': control, 'mechanism': mechanism(len(fit), len(valid))}
    return out


def put_cell(root, saved):
    path = root/'prototype'/f"seed{saved['seed']}_fold{saved['fold']}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path/'checkpoint.pt')
    ev.write_json(path/'DONE_PRIVATE.json', {
        'status': 'COMPLETE', 'source': SOURCE, 'stage': 'prototype', 'seed': saved['seed'], 'fold': saved['fold'],
        'model_slots': 4, 'neural_models': 3, 'epochs': 80, 'optimizer_steps': 240,
        'checkpoint_sha256': ev.sha256_file(path/'checkpoint.pt')})
    return path


def put_replay_evidence(report, scripts, monkeypatch):
    scripts.mkdir(parents=True, exist_ok=True)
    neural_script, main_script = scripts/'replay_phase26.py', scripts/'replay_phase26_main.py'
    neural_script.write_text('synthetic neural replay executable\n')
    main_script.write_text('synthetic main replay executable\n')
    monkeypatch.setattr(ev, 'REPLAY_SCRIPT', neural_script)
    monkeypatch.setattr(ev, 'MAIN_REPLAY_SCRIPT', main_script)
    prepared = {str(f): 'prepared-'+str(f) for f in ev.FOLDS}
    main = {'status': 'PASS_NO_FIT_MAIN_REPLAY', 'source': SOURCE,
            'replay_script_sha256': ev.sha256_file(main_script), 'prepared_cells': 3,
            'prepared_sha256_by_fold': prepared, 'new_fit_calls': 0, 'new_predictor_fits': 0,
            'new_PCA_fits': 0, 'endpoint_API_reads': 0, 'risk_vectors_replayed': 48,
            'legacy_preprocessing_states_reused': 12, 'maximum_absolute_risk_error': 0.,
            'maximum_feature_replay_error': 0., 'absolute_tolerance': 1e-10,
            'valid_labels_loaded_or_used': False, 'fitter_prepare_PCA_and_endpoint_APIs_hard_blocked': True,
            'fold_rows': [{'fold': f, 'prepared_sha256': prepared[str(f)], 'CP_exact_legacy_anchor': True}
                          for f in ev.FOLDS]}
    ev.write_json(report/'MAIN_REPLAY_AUDIT.json', main)
    evidence = {'report_path': 'reports/phase26_multiquery/MAIN_REPLAY_AUDIT.json',
                'report_sha256': ev.sha256_file(report/'MAIN_REPLAY_AUDIT.json'),
                'script_path': 'scripts/replay_phase26_main.py', 'script_sha256': ev.sha256_file(main_script),
                'source': SOURCE, 'status': main['status'], 'prepared_sha256_by_fold': prepared,
                'maximum_absolute_risk_error': 0., 'raw_main_replay_calls_repeated_here': 0}
    ev.write_json(report/'REPLAY_AUDIT.json', {
        'status': 'PASS', 'source': SOURCE, 'stage': 'prototype', 'cells': 6, 'neural_models': 18,
        'neural_risk_vectors': 144, 'reference_risk_vectors': 18, 'new_predictor_fits': 0,
        'new_optimizer_steps': 0, 'endpoint_API_reads': 0, 'maximum_absolute_risk_error': 0.,
        'absolute_tolerance': 1e-7, 'replay_script_sha256': ev.sha256_file(neural_script),
        'neural_weights_unchanged': True, 'all_initialized_module_states_bitwise_shared': True,
        'OFF_exact_own_M0': True, 'QUERY_five_inactive_controls_exact_matched': True,
        'independent_main_replay_reused_no_raw_main_replay_here': True,
        'forbidden_API_call_counts': {'synthetic.fit': 0, 'synthetic.endpoint': 0},
        'main_replay_evidence': evidence})


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    private, report = tmp_path/'private', tmp_path/'report'
    for seed in ev.SEEDS:
        for fold in ev.FOLDS: put_cell(private, record(seed, fold))
    put_replay_evidence(report, tmp_path/'replay_scripts', monkeypatch)
    monkeypatch.setattr(ev, 'evaluator_identity', lambda: 'synthetic-evaluator')
    calls = []
    def endpoints():
        calls.append('endpoint')
        return {'DSS': {'time': np.arange(1., 13.), 'event': np.ones(12), 'valid': np.ones(12, bool)}}
    def baseline(fold):
        saved = record(31, fold)
        return {k: saved[k] for k in ('fit', 'valid', 'M0_valid', 'CP_valid', 'RNA_valid')} | {
            '_prepared_sha256': 'prepared-'+str(fold), '_fit_offset': np.linspace(-.1, .1, len(saved['fit']))}
    return private, report, calls, endpoints, baseline


def run(fixture, **kwargs):
    private, report, calls, endpoints, baseline = fixture
    result = ev.evaluate_stage(private, report, 'prototype', SOURCE, endpoints,
                              baseline_loader=baseline, cohort_size=12, expected_dss_events=12, **kwargs)
    return result


def seed_rows():
    rows = []
    for seed in ev.SEEDS:
        for name in ev.RISK_NAMES:
            value = {'M0': .6, 'CP': .6, 'RNA': .5, 'QUERY': .61, 'ATTN': .65, 'COMP': .7}.get(name, .62)
            rows.append({'seed': seed, 'model': name, 'Harrell': value, 'Uno60': value})
    return rows


def test_gate_go_still_cannot_admit_outer():
    gate = ev.prototype_gate(seed_rows(), source=SOURCE, receipt_fingerprint='fake')
    assert gate['decision'] == 'GO'
    assert gate['outer_admitted'] is False and gate['outer_automatically_started'] is False
    assert gate['separate_full_protocol_required_even_after_GO'] is True


@pytest.mark.parametrize('comparison', [('ATTN', 'QUERY'), ('COMP', 'ATTN')])
@pytest.mark.parametrize('metric', ev.METRICS)
def test_gate_requires_two_positive_seeds(comparison, metric):
    rows = seed_rows(); candidate, baseline = comparison
    by = {(r['seed'], r['model']): r for r in rows}
    by[31, candidate][metric] = by[31, baseline][metric]
    gate = ev.prototype_gate(rows, source=SOURCE, receipt_fingerprint='fake')
    assert gate['decision'] == 'STOP'


@pytest.mark.parametrize('control', ev.GATE_CONTROLS)
def test_gate_pairing_cannot_pass_by_one_metric_only(control):
    rows = seed_rows()
    for row in rows:
        if row['model'] == 'COMP_'+control: row['Harrell'] = .71
    assert ev.prototype_gate(rows, source=SOURCE, receipt_fingerprint='fake')['decision'] == 'STOP'


def test_known_pairs_complete_evaluation(fixture):
    (result, gate) = run(fixture)
    assert fixture[2] == ['endpoint']
    means = {r['model']: r for r in result['two_seed_means']}
    assert means['M0']['Harrell'] == 0 and means['COMP']['Harrell'] == 1
    assert len(result['fold_rows']) == 6*len(ev.RISK_NAMES)
    assert result['unique_patient_OOF'] is False and result['bootstrap_draws'] == 0
    for row in result['repair_damage']:
        assert row['repair']-row['damage'] == pytest.approx(row['net'], abs=1e-12)
    assert gate['decision'] == 'GO' and gate['outer_admitted'] is False
    public = json.dumps(result)
    for key in ('model_state', 'initial_state', 'fit_risk', 'prepared_sha256'):
        assert '"'+key+'"' not in public
    assert json.loads((fixture[0]/'INNER_GATE.json').read_text())['outer_admitted'] is False


def test_missing_last_cell_prevents_any_endpoint_read(fixture):
    (fixture[0]/'prototype/seed47_fold3/DONE_PRIVATE.json').unlink()
    with pytest.raises(ValueError, match='all six'): run(fixture)
    assert fixture[2] == []


def test_tampered_last_checkpoint_prevents_any_deserialization(fixture, monkeypatch):
    path = fixture[0]/'prototype/seed47_fold3/checkpoint.pt'
    with path.open('ab') as stream: stream.write(b'tamper')
    def forbidden(*args, **kwargs): raise AssertionError('must check all SHA before deserialize')
    monkeypatch.setattr(ev.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='SHA'): run(fixture)
    assert fixture[2] == []


@pytest.mark.parametrize('change', ['source', 'status', 'cells', 'neural_models'])
def test_replay_missing_or_wrong_prevents_endpoints(fixture, change):
    path = fixture[1]/'REPLAY_AUDIT.json'
    report = json.loads(path.read_text()); report[change] = 'wrong'
    ev.write_json(path, report)
    with pytest.raises(ValueError, match='replay'): run(fixture)
    assert fixture[2] == []


@pytest.mark.parametrize('field', ['maximum_absolute_risk_error', 'replay_script_sha256',
                                 'new_predictor_fits', 'neural_risk_vectors',
                                 'OFF_exact_own_M0', 'main_replay_evidence'])
def test_replay_critical_fields_cannot_be_omitted(fixture, field):
    path = fixture[1]/'REPLAY_AUDIT.json'
    saved = json.loads(path.read_text()); saved.pop(field); ev.write_json(path, saved)
    with pytest.raises(ValueError): run(fixture)
    assert fixture[2] == []


@pytest.mark.parametrize('script', ['REPLAY_SCRIPT', 'MAIN_REPLAY_SCRIPT'])
def test_replay_script_sha_drift_blocks_endpoints(fixture, script):
    path = getattr(ev, script)
    path.write_text(path.read_text()+'changed executable\n')
    with pytest.raises(ValueError, match='SHA'): run(fixture)
    assert fixture[2] == []


def test_replay_main_report_hash_drift_blocks_endpoints(fixture):
    path = fixture[1]/'MAIN_REPLAY_AUDIT.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='SHA'): run(fixture)
    assert fixture[2] == []


def test_replay_main_prepared_map_is_bound_to_cells(fixture):
    path = fixture[1]/'REPLAY_AUDIT.json'
    saved = json.loads(path.read_text())
    saved['main_replay_evidence']['prepared_sha256_by_fold']['3'] = 'different-prepared'
    ev.write_json(path, saved)
    with pytest.raises(ValueError, match='prepared bindings'): run(fixture)
    assert fixture[2] == []


def test_replay_forbidden_api_call_blocks_endpoints(fixture):
    path = fixture[1]/'REPLAY_AUDIT.json'
    saved = json.loads(path.read_text()); saved['forbidden_API_call_counts']['synthetic.fit'] = 1
    ev.write_json(path, saved)
    with pytest.raises(ValueError, match='forbidden API'): run(fixture)
    assert fixture[2] == []


def test_independent_prepared_risk_mismatch_prevents_endpoints(fixture):
    private, report, calls, endpoints, baseline = fixture
    def mismatch(fold):
        out = baseline(fold); out['CP_valid'] = np.ones(4)
        return out
    with pytest.raises(ValueError, match='independent main risk'):
        ev.evaluate_stage(private, report, 'prototype', SOURCE, endpoints, baseline_loader=mismatch, cohort_size=12)
    assert calls == []


def test_patient_leakage_blocks_endpoints(fixture):
    saved = record(31, 1); saved['fit'][0] = saved['valid'][0]; put_cell(fixture[0], saved)
    with pytest.raises(ValueError, match='leakage'): run(fixture)
    assert fixture[2] == []


@pytest.mark.parametrize('change', ['OFF', 'QUERY_VALUE', 'bound', 'history', 'initial', 'private_mechanism'])
def test_invalid_model_contract_blocks_endpoints(fixture, change):
    saved = record(31, 1)
    if change == 'OFF': saved['arms']['COMP']['sensitivities']['OFF'][0] += .01
    elif change == 'QUERY_VALUE': saved['arms']['QUERY']['sensitivities']['VALUE_OFF'][0] += .01
    elif change == 'bound': saved['arms']['COMP']['risk'][0] += 1
    elif change == 'history': saved['arms']['COMP']['history'].pop()
    elif change == 'initial': saved['arms']['COMP']['initial_state']['weight'] += 1
    else: saved['arms']['COMP']['mechanism']['patient_ids'] = ['private']
    put_cell(fixture[0], saved)
    with pytest.raises(ValueError): run(fixture)
    assert fixture[2] == []


def test_existing_result_refuses_before_endpoints(fixture):
    ev.write_json(fixture[1]/'INNER_RESULTS.json', {'historical': True})
    with pytest.raises(ValueError, match='overwrite'): run(fixture)
    assert fixture[2] == []


def test_outer_always_refused(fixture):
    with pytest.raises(ValueError, match='prototype only'):
        ev.evaluate_stage(fixture[0], fixture[1], 'outer', SOURCE, fixture[3], baseline_loader=fixture[4])
    assert fixture[2] == []


def test_mechanism_moments_reject_arrays_and_identifying_fields():
    value = mechanism(8, 4)
    value['slot']['pairwise_cosine']['mean'] = np.arange(4)
    with pytest.raises(ValueError, match='scalar'): ev.mechanism_summary(value, 8, 4)
    value = mechanism(8, 4); value['frozen_query_changes']['OFF']['patient_ids'] = [1, 2]
    with pytest.raises(ValueError, match='unreviewed'): ev.mechanism_summary(value, 8, 4)

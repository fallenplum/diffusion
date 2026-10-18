"""Synthetic saved-head contracts and pair oracles; no real endpoints or fit."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.phase27_readout import evaluation as ev

SOURCE = 'synthetic-phase27'
COHORT = 24


def optimizer():
    return {'status': 'CONVERGED', 'penalty': 10., 'parameters': 16, 'cox': 2., 'objective': 2.,
            'gradient_max': 1e-9, 'intercept': False, 'offset_trainable': False,
            'history': [{'phase': 'final', 'iteration': 3, 'objective': 2., 'cox': 2.,
                         'gradient_max': 1e-9, 'gradient_norm': 2e-9}]}


def projection(n):
    return {'rank': 16, 'input_dim': 256, 'svd_solver': 'full', 'whiten': False,
            'input_standardized_before_PCA': False, 'fit_n': n, 'input_fit_rank': 19,
            'pca_mean': np.zeros(256), 'components': np.eye(256)[:16],
            'pca_score_mean': np.zeros(16), 'pca_score_sd': np.ones(16)}


def record(seed, fold):
    valid = np.arange((fold-1)*4, fold*4); fit = np.setdiff1d(np.arange(COHORT), valid)
    main = np.arange(4, dtype=float)*.02
    output = {'source': SOURCE, 'stage': 'prototype', 'seed': seed, 'fold': fold, 'fit': fit, 'valid': valid,
              'legacy_checkpoint_sha256': f'legacy-{seed}-{fold}', 'prepared_sha256': f'prepared-{fold}',
              'feature_receipt_sha256': f'feature-{seed}-{fold}',
              'M0_valid': main, 'CP_valid': np.zeros(4), 'RNA_valid': main,
              'old_risks': {arm: main+.001 for arm in ev.ARMS}, 'heads': {}}
    shifts = {'QUERY': [.01, .002, -.002, -.01], 'ATTN': [.04, .02, -.02, -.04],
              'COMP': [.09, .03, -.03, -.09]}
    for arm in ev.ARMS:
        risk = main+np.array(shifts[arm])
        output['heads'][arm] = {'coef': np.zeros(16), 'projection': projection(len(fit)),
            'fit_risk': np.zeros(len(fit)), 'valid_risks': {control: risk.copy() if control == 'MATCHED' or arm == 'QUERY'
                else main.copy() for control in ev.CONTROLS}, 'optimizer': optimizer(),
            'fit_diagnostics': {'anonymous': {'n': len(fit), 'mean': 0.}}, 'parameter_count': 16}
    return output


def put_head(root, saved):
    path = root/'prototype'/f"seed{saved['seed']}_fold{saved['fold']}"
    path.mkdir(parents=True, exist_ok=True); torch.save(saved, path/'head.pt')
    ev.write_json(path/'DONE_PRIVATE.json', {'status': 'COMPLETE', 'source': SOURCE,
        'stage': 'prototype', 'seed': saved['seed'], 'fold': saved['fold'], 'heads': 3,
        'new_convex_fits': 3, 'new_neural_steps': 0, 'checkpoint_sha256': ev.sha256_file(path/'head.pt'),
        'feature_receipt_sha256': saved['feature_receipt_sha256']})


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    private, report = tmp_path/'private', tmp_path/'report'
    for seed in ev.SEEDS:
        for fold in ev.FOLDS: put_head(private, record(seed, fold))
    project = tmp_path/'project'; scripts = project/'scripts'; scripts.mkdir(parents=True)
    script = scripts/'replay_phase27.py'; script.write_text('synthetic replay\n')
    for name in ('replay_phase26_main.py', 'replay_phase26.py'): (scripts/name).write_text('synthetic old replay\n')
    monkeypatch.setattr(ev, 'PROJECT_ROOT', project)
    monkeypatch.setattr(ev, 'REPLAY_SCRIPT', script)
    monkeypatch.setattr(ev, 'evaluator_identity', lambda: 'synthetic-evaluator')
    olddir = project/'reports/phase26_multiquery'
    ev.write_json(olddir/'MAIN_REPLAY_AUDIT.json', {'status': 'PASS_NO_FIT_MAIN_REPLAY',
        'source': ev.LEGACY_SOURCE, 'prepared_cells': 3, 'risk_vectors_replayed': 48,
        'legacy_preprocessing_states_reused': 12,
        'replay_script_sha256': ev.sha256_file(scripts/'replay_phase26_main.py'),
        'prepared_sha256_by_fold': {str(f): f'prepared-{f}' for f in ev.FOLDS}})
    ev.write_json(olddir/'REPLAY_AUDIT.json', {'status': 'PASS', 'source': ev.LEGACY_SOURCE,
        'neural_risk_vectors': 144, 'reference_risk_vectors': 18, 'cells': 6, 'neural_models': 18,
        'replay_script_sha256': ev.sha256_file(scripts/'replay_phase26.py'),
        'main_replay_evidence': {'report_sha256': ev.sha256_file(olddir/'MAIN_REPLAY_AUDIT.json')}})
    evidence = {'source': ev.LEGACY_SOURCE, 'raw_main_or_neural_replay_repeated': False,
        'main_report_sha256': ev.sha256_file(olddir/'MAIN_REPLAY_AUDIT.json'),
        'main_replay_script_sha256': ev.sha256_file(scripts/'replay_phase26_main.py'),
        'neural_report_sha256': ev.sha256_file(olddir/'REPLAY_AUDIT.json'),
        'neural_replay_script_sha256': ev.sha256_file(scripts/'replay_phase26.py')}
    replay_rows = []
    for seed in ev.SEEDS:
        for fold in ev.FOLDS:
            saved = record(seed, fold)
            replay_rows.append({'seed': seed, 'fold': fold,
                'head_checkpoint_sha256': ev.sha256_file(private/'prototype'/f'seed{seed}_fold{fold}'/'head.pt'),
                **{k: saved[k] for k in ('feature_receipt_sha256', 'legacy_checkpoint_sha256', 'prepared_sha256')},
                'feature_sha256': f'featurebytes-{seed}-{fold}', 'risk_vectors_replayed': 18,
                'maximum_absolute_risk_error': 0.})
    ev.write_json(report/'REPLAY_AUDIT.json', {'status': 'PASS', 'source': SOURCE, 'stage': 'prototype',
        'cells': 6, 'new_convex_heads': 18, 'no_new_predictor_fits': 0, 'no_forward': 0,
        'old_main_reused': True, 'maximum_absolute_risk_error': 0.,
        'new_risk_vectors_replayed': 108, 'main_reference_vectors_checked': 18,
        'original_bounded_head_references_checked': 18, 'new_predictor_fits': 0,
        'neural_forward_calls': 0, 'optimizer_steps': 0, 'endpoint_API_reads': 0,
        'absolute_tolerance': 1e-10, 'private_artifacts_bytes_unchanged': True,
        'new_and_legacy_training_source_bytes_unchanged': True,
        'projection_and_risk_arithmetic_independent_of_new_math_helpers': True,
        'heldout_labels_read': False, 'gradient_recomputed_from_labels': False,
        'legacy_replay_evidence': evidence, 'rows': replay_rows,
        'replay_script_sha256': ev.sha256_file(script), 'forbidden_API_call_counts': {'synthetic.fit': 0}})
    calls = []
    def endpoints():
        calls.append('endpoint')
        return {'DSS': {'time': np.arange(1., COHORT+1), 'event': np.ones(COHORT), 'valid': np.ones(COHORT, bool)}}
    def provenance(seed, fold):
        saved = record(seed, fold)
        return {key: saved[key] for key in ('legacy_checkpoint_sha256', 'prepared_sha256',
                                             'feature_receipt_sha256', 'old_risks')} | {
            'legacy_source': ev.LEGACY_SOURCE, 'feature_sha256': f'featurebytes-{seed}-{fold}'}
    def baseline(fold):
        saved = record(31, fold)
        return {key: saved[key] for key in ('fit', 'valid', 'M0_valid', 'CP_valid', 'RNA_valid', 'prepared_sha256')}
    def partition(fold):
        saved = record(31, fold); return {k: saved[k] for k in ('fit', 'valid')}
    return private, report, calls, endpoints, provenance, baseline, partition


def run(fixture, **kwargs):
    private, report, _, endpoints, provenance, baseline, partition = fixture
    return ev.evaluate_stage(private, report, 'prototype', SOURCE, endpoints,
        provenance_loader=provenance, baseline_loader=baseline, partition_loader=partition,
        cohort_size=COHORT, expected_dss_events=COHORT, **kwargs)


def gate_rows():
    scores = {'M0': .6, 'CP': .6, 'RNA': .5, 'QUERY': .61, 'ATTN': .65, 'COMP': .7}
    return [{'seed': seed, 'model': name, 'Harrell': scores.get(name, .62), 'Uno60': scores.get(name, .62)}
            for seed in ev.SEEDS for name in ev.RISK_NAMES]


def test_gate_go_cannot_launch_outer_and_old_head_only_diagnostic():
    rows = gate_rows()
    for row in rows:
        if row['model'].startswith('OLD_'): row.update(Harrell=.99, Uno60=.99)
    gate = ev.prototype_gate(rows, source=SOURCE, receipt_fingerprint='fake')
    assert gate['decision'] == 'GO' and gate['outer_admitted'] is False
    assert gate['old_head_comparisons_are_diagnostic_only'] is True


@pytest.mark.parametrize('candidate,baseline', [('ATTN', 'QUERY'), ('COMP', 'ATTN')])
@pytest.mark.parametrize('metric', ev.METRICS)
def test_primary_requires_both_positive_seeds(candidate, baseline, metric):
    rows = gate_rows(); by = {(r['seed'], r['model']): r for r in rows}
    by[31, candidate][metric] = by[31, baseline][metric]
    assert ev.prototype_gate(rows, source=SOURCE, receipt_fingerprint='fake')['decision'] == 'STOP'


@pytest.mark.parametrize('control', ev.GATE_CONTROLS)
def test_every_locked_frozen_control_requires_both_metric_nonnegative(control):
    rows = gate_rows()
    for row in rows:
        if row['model'] == 'COMP_'+control: row['Harrell'] = .71
    assert ev.prototype_gate(rows, source=SOURCE, receipt_fingerprint='fake')['decision'] == 'STOP'


def test_known_pairs_saved_readout_and_public_privacy(fixture):
    result, gate = run(fixture)
    by = {r['model']: r for r in result['two_seed_means']}
    assert by['M0']['Harrell'] == 0 and by['COMP']['Harrell'] == 1
    assert gate['decision'] == 'GO' and gate['outer_admitted'] is False
    assert fixture[2] == ['endpoint']
    for row in result['repair_damage']:
        assert row['repair']-row['damage'] == pytest.approx(row['net'], abs=1e-12)
    public = json.dumps(result)
    for key in ('coef', 'projection', 'fit_risk', 'fit_diagnostics', 'old_risks'):
        assert '"'+key+'"' not in public
    assert result['new_predictor_fits_during_evaluation'] == result['bootstrap_draws'] == 0


def test_all_receipts_checked_before_deserializing(fixture, monkeypatch):
    path = fixture[0]/'prototype/seed47_fold3/head.pt'
    with path.open('ab') as stream: stream.write(b'changed')
    def forbidden(*a, **k): raise AssertionError('deserialize before all receipt checks')
    monkeypatch.setattr(ev.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='SHA'): run(fixture)
    assert fixture[2] == []


def test_missing_last_head_blocks_endpoint(fixture):
    (fixture[0]/'prototype/seed47_fold3/DONE_PRIVATE.json').unlink()
    with pytest.raises(ValueError, match='six'): run(fixture)
    assert fixture[2] == []


@pytest.mark.parametrize('field', ['maximum_absolute_risk_error', 'replay_script_sha256', 'new_convex_heads',
                                 'no_new_predictor_fits', 'forbidden_API_call_counts'])
def test_replay_critical_fields_fail_closed(fixture, field):
    path = fixture[1]/'REPLAY_AUDIT.json'; saved = json.loads(path.read_text()); saved.pop(field)
    ev.write_json(path, saved)
    with pytest.raises(ValueError): run(fixture)
    assert fixture[2] == []


def test_replay_script_drift_blocks_endpoint(fixture):
    ev.REPLAY_SCRIPT.write_text('changed executable')
    with pytest.raises(ValueError, match='SHA'): run(fixture)
    assert fixture[2] == []


@pytest.mark.parametrize('field', ['head_checkpoint_sha256', 'feature_receipt_sha256', 'legacy_checkpoint_sha256',
                                  'prepared_sha256', 'feature_sha256'])
def test_replay_row_must_bind_current_artifact_bytes(fixture, field):
    path = fixture[1]/'REPLAY_AUDIT.json'; saved = json.loads(path.read_text())
    saved['rows'][-1][field] = 'changed-artifact'
    ev.write_json(path, saved)
    with pytest.raises(ValueError): run(fixture)
    assert fixture[2] == []


def test_bound_legacy_report_sha_drift_blocks_endpoint(fixture):
    path = ev.PROJECT_ROOT/'reports/phase26_multiquery/MAIN_REPLAY_AUDIT.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='legacy replay SHA'): run(fixture)
    assert fixture[2] == []


@pytest.mark.parametrize('bad', ['source', 'feature', 'old_risk'])
def test_independent_provenance_checked_before_endpoint(fixture, bad):
    private, report, calls, endpoints, provenance, baseline, partition = fixture
    def invalid(seed, fold):
        saved = provenance(seed, fold)
        if bad == 'source': saved['legacy_source'] = 'wrong'
        elif bad == 'feature': saved['feature_receipt_sha256'] = 'wrong'
        else: saved['old_risks']['COMP'][0] += 1
        return saved
    with pytest.raises(ValueError):
        ev.evaluate_stage(private, report, 'prototype', SOURCE, endpoints, provenance_loader=invalid,
                          baseline_loader=baseline, partition_loader=partition, cohort_size=COHORT)
    assert calls == []


@pytest.mark.parametrize('bad', ['coef', 'control', 'query_control', 'gradient', 'projection', 'leakage'])
def test_invalid_readout_record_blocks_endpoint(fixture, bad):
    saved = record(31, 1)
    if bad == 'coef': saved['heads']['COMP']['coef'] = np.ones(17)
    elif bad == 'control': saved['heads']['COMP']['valid_risks'].pop('PATCH_PERMUTE')
    elif bad == 'query_control': saved['heads']['QUERY']['valid_risks']['PATCH_PERMUTE'][0] += .1
    elif bad == 'gradient': saved['heads']['COMP']['optimizer']['gradient_max'] = .01
    elif bad == 'projection': saved['heads']['COMP']['projection']['pca_score_sd'][0] = 0
    else: saved['fit'][0] = saved['valid'][0]
    put_head(fixture[0], saved)
    with pytest.raises(ValueError): run(fixture)
    assert fixture[2] == []


def test_existing_result_and_outer_fail_closed(fixture):
    ev.write_json(fixture[1]/'INNER_RESULTS.json', {'old': True})
    with pytest.raises(ValueError, match='overwrite'): run(fixture)
    with pytest.raises(ValueError, match='prototype only'):
        ev.evaluate_stage(fixture[0], fixture[1], 'outer', SOURCE, fixture[3],
                          provenance_loader=fixture[4], baseline_loader=fixture[5], partition_loader=fixture[6])
    assert fixture[2] == []

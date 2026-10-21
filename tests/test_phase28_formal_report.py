"""Formal rendering uses the real pure evaluator's synthetic JSON schema."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.phase28_reread import evaluation as ev


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/report_phase28_formal.py'
spec = importlib.util.spec_from_file_location('phase28_formal_report', SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
SOURCE = 'a'*64
FINGERPRINT = 'c'*64
SCRIPT_SHA = 'b'*64


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def history():
    return {'actual_epochs': 80,
        'cox_loss': {'last10_preupdate_change': -.01, 'last_preupdate': 2., 'minimum_preupdate': 2.},
        'convergence_or_budget_sufficiency_claimed': False}


def fixture_reports(directory, *, go=False):
    """All risks/outcomes below are synthetic in-memory arithmetic only."""
    time, event = np.arange(1., 513), np.zeros(512)
    event[:65] = 1
    selected = (np.r_[0:24, 100:140], np.r_[24:45, 140:180], np.r_[45:65, 180:220])
    cells = {}
    for seed in ev.SEEDS:
        for fold, valid in zip(ev.FOLDS, selected):
            risks = {name: -time[valid].copy() for name in ev.RISK_NAMES}
            risks['RNA'] = np.zeros(len(valid))
            if go:
                risks['C00'], risks['DIRECT'] = time[valid].copy(), time[valid].copy()
                for control in ev.GATE_CONTROLS: risks['C11_'+control] = time[valid].copy()
            cells[seed, fold] = {'fit': np.setdiff1d(np.arange(512), valid), 'valid': valid, 'risks': risks}
    endpoint = {'time': time, 'event': event, 'valid': np.ones(512, bool)}
    points = ev.point_results(cells, endpoint)
    pairs = ev.pair_results(cells, endpoint, points)
    gate = ev.prototype_gate(points['seed_rows'], source=SOURCE, receipt_fingerprint=FINGERPRINT,
                             evaluator_sha256='d'*64)
    assert gate['decision'] == ('GO' if go else 'STOP')
    replay = {'status': 'PASS', 'source': SOURCE, 'neural_models': 30, 'risk_vectors_replayed': 300,
              'reference_vectors_checked': 90, 'maximum_absolute_risk_error': 0.}
    write(directory/'REPLAY_AUDIT.json', replay)
    replay_sha = report.sha(directory/'REPLAY_AUDIT.json')
    rows = [{'seed': seed, 'fold': fold, 'arm': arm, 'training': history()}
            for seed in ev.SEEDS for fold in ev.FOLDS for arm in ev.ARMS]
    artifact = {'status': 'PASS_COMPLETE_CHECKPOINT_CONTRACT', 'neural_models': 30,
        'seed_partition_cells': 6, 'epochs_each': 80, 'optimizer_steps': 2400, 'new_main_anchor_fits': 0,
        'receipt_fingerprint': FINGERPRINT, 'rows': rows}
    result = {'status': 'COMPLETE_BIDIRECTIONAL_REREAD_INNER_PROTOTYPE', 'source': SOURCE,
        'stage': 'prototype', 'endpoint': 'DSS', 'cohort_patients': 512, 'DSS_events': 65,
        'artifact_audit': artifact, 'independent_replay': {**replay, 'report_sha256': replay_sha},
        'valid_endpoint_callback_calls': 1, 'new_predictor_fits_during_evaluation': 0,
        'bootstrap_draws': 0, 'patient_arrays_weights_or_IDs_public': False,
        'Clinical_is_not_a_local_query_input': True, 'learned_unimodal_heads_can_drift_with_joint_training': True,
        **points, **pairs}
    budget_rows = copy.deepcopy(rows)
    for row in budget_rows:
        value = 0. if row['arm'] == 'C00' else .001
        row['risk_scales'] = {'fit': {'interaction': {'RMS': value}}, 'valid': {'interaction': {'RMS': value*.8}}}
    budget = {'schema': 'phase28_checkpoint_budget_readout_v1', 'status': 'PASS_OUTCOME_BLIND_CHECKPOINT_AGGREGATION',
        'source': SOURCE, 'neural_models': 30, 'receipt_fingerprint': FINGERPRINT, 'rows': budget_rows,
        'replay_report_sha256': replay_sha, 'new_predictor_fits': 0, 'new_optimizer_steps': 0,
        'new_forward_calls': 0, 'endpoint_array_requests': 0, 'label_keys_inspected_or_used': False,
        'rescaling_coefficients_searched': False, 'admission_gate_changed': False,
        'global_convergence_or_budget_sufficiency_claimed': False, 'patient_risks_indices_or_model_tensors_public': False}
    decision = {'source': SOURCE, 'stage': 'prototype', 'status': gate['classification'], 'decision': gate['decision'],
        'candidate': 'C11', 'outer_admitted': False, 'outer_started': False, 'confirmation_claimed': False}
    inputs = {'INNER_RESULTS.json': result, 'INNER_GATE.json': gate, 'decision.json': decision,
              'BUDGET_READOUT_AUDIT.json': budget}
    for name, value in inputs.items(): write(directory/name, value)
    return inputs


@pytest.mark.parametrize('go', [False, True])
def test_full_anonymous_synthetic_schema_renders_stop_or_go_without_new_decision(tmp_path, go):
    fixture_reports(tmp_path, go=go)
    sentinel = b'P0 byte preservation sentinel'
    (tmp_path/'RESULTS.md').write_bytes(sentinel)
    delivery = report.generate(tmp_path, SOURCE, script_sha256=SCRIPT_SHA)
    assert delivery['scientific_decision'] == ('GO' if go else 'STOP')
    assert delivery['schema'] == 'phase28_formal_delivery_v1'
    assert delivery['status'] == 'COMPLETE_INNER_FORMAL_REPORTS'
    assert delivery['endpoint_loaders_called'] == 0 and delivery['new_predictor_fits'] == 0
    assert (tmp_path/'RESULTS.md').read_bytes() == sentinel
    assert set(delivery['report_sha256']) == {*report.INPUT_NAMES, 'RESULTS_FORMAL.md'}
    for name, digest in delivery['report_sha256'].items(): assert report.sha(tmp_path/name) == digest
    text = (tmp_path/'RESULTS_FORMAL.md').read_text(encoding='utf-8')
    assert '不是完整5外折×3seed确认' in text
    assert '未计算确认性置信区间' in text
    assert '不能把全部收益或损失直接归给重读差分' in text
    assert '不等于证明双向重读' in text
    assert report.generate(tmp_path, SOURCE, script_sha256=SCRIPT_SHA) == delivery


@pytest.mark.parametrize('tamper', ['source', 'P0', 'missing_model', 'missing_replay', 'endpoint_twice',
    'point_aggregation', 'fold_weighting', 'primary_gate', 'main_guard', 'frozen_guard', 'decision', 'pair_credit', 'seed_pair_credit',
    'budget_source', 'budget_endpoints', 'budget_fingerprint', 'budget_epochs', 'nonfinite'])
def test_incomplete_or_tampered_results_refuse_to_render(tmp_path, tamper):
    inputs = fixture_reports(tmp_path, go=True)
    name = 'INNER_RESULTS.json'; result = inputs[name]
    if tamper == 'source': result['source'] = 'f'*64
    elif tamper == 'P0': result['status'] = 'QUALIFIED_NO_SEMANTIC_CORE_NOT_PATIENT_TRAINED'
    elif tamper == 'missing_model': result['artifact_audit']['rows'].pop()
    elif tamper == 'missing_replay': result['independent_replay']['risk_vectors_replayed'] = 299
    elif tamper == 'endpoint_twice': result['valid_endpoint_callback_calls'] = 2
    elif tamper == 'point_aggregation': result['seed_rows'][0]['Harrell'] -= .1
    elif tamper == 'fold_weighting': result['fold_rows'][0]['Uno_den'] *= 1.5
    elif tamper == 'primary_gate':
        name = 'INNER_GATE.json'; inputs[name]['primary_tests'][0]['pass'] = False
    elif tamper == 'main_guard':
        name = 'INNER_GATE.json'; inputs[name]['main_effect_guards'][0]['mean_seed_delta'] = .1
    elif tamper == 'frozen_guard':
        name = 'INNER_GATE.json'; inputs[name]['frozen_pairing_guards'][0]['mean_seed_deltas']['Harrell'] = -.1
    elif tamper == 'decision':
        name = 'decision.json'; inputs[name]['decision'] = 'STOP'
    elif tamper == 'pair_credit': result['repair_damage'][0]['net'] = .1
    elif tamper == 'seed_pair_credit': result['seed_repair_damage'][0]['repair'] += .1
    elif tamper == 'budget_source':
        name = 'BUDGET_READOUT_AUDIT.json'; inputs[name]['source'] = 'f'*64
    elif tamper == 'budget_endpoints':
        name = 'BUDGET_READOUT_AUDIT.json'; inputs[name]['endpoint_array_requests'] = 1
    elif tamper == 'budget_fingerprint':
        name = 'BUDGET_READOUT_AUDIT.json'; inputs[name]['receipt_fingerprint'] = 'f'*64
    elif tamper == 'budget_epochs':
        name = 'BUDGET_READOUT_AUDIT.json'; inputs[name]['rows'][0]['training']['actual_epochs'] = 79
    elif tamper == 'nonfinite':
        (tmp_path/name).write_text('{"x":NaN}', encoding='utf-8')
    if tamper != 'nonfinite': write(tmp_path/name, inputs[name])
    with pytest.raises((ValueError, KeyError)):
        report.generate(tmp_path, SOURCE, script_sha256=SCRIPT_SHA)
    assert not (tmp_path/'RESULTS_FORMAL.md').exists()
    assert not (tmp_path/'FORMAL_DELIVERY.json').exists()


def test_live_replay_hash_and_conflicting_existing_report_fail(tmp_path):
    fixture_reports(tmp_path, go=False)
    replay_path = tmp_path/'REPLAY_AUDIT.json'
    replay_path.write_bytes(replay_path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='live independent replay'):
        report.generate(tmp_path, SOURCE, script_sha256=SCRIPT_SHA)
    fixture_reports(tmp_path, go=False)
    (tmp_path/'RESULTS_FORMAL.md').write_text('different existing report', encoding='utf-8')
    with pytest.raises(ValueError, match='refusing overwrite'):
        report.generate(tmp_path, SOURCE, script_sha256=SCRIPT_SHA)


def test_missing_real_evaluation_refuses_before_any_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.generate(tmp_path, SOURCE, script_sha256=SCRIPT_SHA)
    assert list(tmp_path.iterdir()) == []

"""Independent inference oracle and synthetic-only artifact tamper checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.phase28_reread import evaluation as ev
from src.phase28_reread import replay as rp


SOURCE = 'SYNTHETIC_PHASE28_ONLY'
COHORT = 24


def synthetic_record(seed, fold, arm):
    valid = np.arange((fold-1)*4, fold*4)
    fit = np.setdiff1d(np.arange(COHORT), valid)
    external = np.arange(4, dtype=float)*.02
    own_main = external+.01
    matched = -external if arm == 'C11' else own_main
    controls = {key: (matched.copy() if key == 'MATCHED' else own_main.copy()) for key in ev.CONTROLS}
    if arm == 'C00': controls = {key: own_main.copy() for key in ev.CONTROLS}
    history = [{'epoch': epoch, 'patients': len(fit), 'events': 2, 'microbatch': 1, 'optimizer_steps': 0,
        'cox_loss': 2., 'regularizer': .1, 'objective': 2.1, 'maximum_absolute_component_replay_error': 0.,
        'seconds': .01, 'data_gradient_L2_by_top_level_module': {'main': .1, 'dormant': 0.},
        'total_gradient_L2_by_top_level_module': {'main': .1, 'dormant': .01}} for epoch in range(1, 81)]
    audit = {'actual_epochs': 80, 'optimizer_steps': 80, 'new_main_anchor_fits': 0,
        'complete_fit_risk_sets': True, 'fixed_final_epoch': True, 'valid_endpoint_reads': 0,
        'external_offset_unchanged': True}
    return {'source': SOURCE, 'stage': 'prototype', 'seed': seed, 'fold': fold, 'arm': arm,
        'fit': fit, 'valid': valid, 'prepared_sha256': str(fold)*64, 'initial_state_sha256': 'b'*64,
        'model_state': {'SYNTHETIC_weight': torch.tensor([.2])},
        'model_init': {'genes': 26708, 'path_dim': 1536, 'rank': 4},
        'CP_valid': external*0, 'RNA_valid': external, 'M0_valid': external,
        'fit_risk': np.zeros(len(fit)), 'valid_risks': controls, 'valid_base_risk': own_main,
        'history': history, 'training_audit': audit, 'data_audit': {'synthetic_only': True}}


def put(directory, record):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(record, directory/'checkpoint.pt')
    ev.write_json(directory/'DONE_PRIVATE.json', {'status': 'COMPLETE', 'source': SOURCE,
        'stage': 'prototype', 'seed': record['seed'], 'fold': record['fold'], 'arm': record['arm'],
        'epochs': 80, 'optimizer_steps': 80, 'checkpoint_sha256': ev.sha256_file(directory/'checkpoint.pt'),
        'initial_state_sha256': record['initial_state_sha256'], 'prepared_sha256': record['prepared_sha256']})


@pytest.fixture
def harness(tmp_path, monkeypatch):
    private, report, root = tmp_path/'private', tmp_path/'report', tmp_path/'project'
    for seed in ev.SEEDS:
        for fold in ev.FOLDS:
            for arm in ev.ARMS:
                put(private/'prototype'/f'seed{seed}_fold{fold}'/arm, synthetic_record(seed, fold, arm))
    paths = ('src/phase28_reread/data.py', 'src/phase28_reread/spatial.py',
             'src/phase28_reread/replay.py', 'scripts/replay_phase28.py',
             'scripts/audit_phase28_inputs.py', 'scripts/qualify_phase28.py')
    for name in paths:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic '+name)
    files = {name: ev.sha256_file(root/name) for name in paths[:2]}
    report.mkdir()
    for name in ('DEPENDENCY_QUALIFICATION', 'NUMERICAL_QUALIFICATION'):
        ev.write_json(report/(name+'.json'), {'status': 'PASS_SYNTHETIC_ONLY', 'source': SOURCE,
            'source_files': files, 'script_path': 'scripts/qualify_phase28.py',
            'script_sha256': ev.sha256_file(root/'scripts/qualify_phase28.py')})
    ev.write_json(private/'H5_SOURCES_PRIVATE.json', {'synthetic_only': True})
    ev.write_json(report/'INPUT_AUDIT.json', {'status': 'PASS', 'stage': 'prototype_input_qualification_only',
        'input_code_sha256': {'data': ev.sha256_file(root/paths[0]), 'spatial': ev.sha256_file(root/paths[1]),
                             'audit_script': ev.sha256_file(root/'scripts/audit_phase28_inputs.py')},
        'audit_script_sha256': ev.sha256_file(root/'scripts/audit_phase28_inputs.py'),
        'private_registry_sha256': ev.sha256_file(private/'H5_SOURCES_PRIVATE.json'),
        'folds': [{'fold': fold, 'prepared_sha256': str(fold)*64} for fold in ev.FOLDS],
        'new_neural_training_steps': 0, 'new_predictor_fits': 0, 'optimizer_constructions': 0,
        'neural_forward_calls': 0, 'valid_endpoint_array_reads': 0, 'fit_label_getter_calls': 0,
        'forbidden_API_call_counts': {'synthetic_forbidden': 0},
        'fit_label_bytes_deserialized_in_prepared': True})
    _, audit = ev.verify_stage(private, 'prototype', SOURCE, cohort_size=COHORT)
    qualification = ev.verify_qualifications(report, SOURCE, files, project_root=root, private_root=private)
    rows = [{**row, 'risk_vectors_replayed': 10, 'reference_vectors_checked': 3,
             'maximum_absolute_risk_error': 0.} for row in audit['model_bindings']]
    ev.write_json(report/'REPLAY_AUDIT.json', {'schema': ev.REPLAY_SCHEMA, 'status': 'PASS', 'source': SOURCE,
        'stage': 'prototype', 'source_files': files, 'neural_models': 30, 'seed_partition_cells': 6,
        'risk_vectors_replayed': 300, 'reference_vectors_checked': 90, 'new_predictor_fits': 0,
        'optimizer_steps': 0, 'endpoint_API_reads': 0, 'absolute_tolerance': 1e-6,
        'maximum_absolute_risk_error': 0., 'rows': rows,
        'replay_script_sha256': ev.sha256_file(root/'scripts/replay_phase28.py'),
        'replay_module_sha256': ev.sha256_file(root/'src/phase28_reread/replay.py'),
        'private_checkpoints_and_receipts_unchanged': True, 'training_source_bytes_unchanged': True,
        'model_state_unchanged': True, 'external_references_elementwise_exact': True, 'no_valid_endpoints_loaded': True,
        'forbidden_API_call_counts': dict.fromkeys(ev.REPLAY_FORBIDDEN, 0),
        'qualification_report_hashes': {k: v['report_sha256'] for k, v in qualification.items()}})
    monkeypatch.setattr(ev, 'evaluator_identity', lambda: 'synthetic-evaluator')
    calls = []
    def endpoint():
        calls.append('endpoint')
        return {'DSS': {'time': np.arange(1., COHORT+1), 'event': np.ones(COHORT), 'valid': np.ones(COHORT, bool)}}
    def baseline(fold):
        saved = synthetic_record(31, fold, 'C00')
        return {k: saved[k] for k in ('fit', 'valid', 'CP_valid', 'RNA_valid', 'M0_valid', 'prepared_sha256')}
    def partition(fold):
        saved = synthetic_record(31, fold, 'C00'); return {k: saved[k] for k in ('fit', 'valid')}
    return SimpleNamespace(private=private, report=report, root=root, files=files,
                           endpoint=endpoint, baseline=baseline, partition=partition, calls=calls)


def run(harness, **options):
    return ev.evaluate_stage(harness.private, harness.report, 'prototype', SOURCE, harness.endpoint,
        source_files=harness.files, baseline_loader=harness.baseline, partition_loader=harness.partition,
        source_guard=lambda: None, project_root=harness.root, cohort_size=COHORT,
        expected_dss_events=options.get('expected_dss_events', COHORT))


def test_synthetic_thirty_artifact_chain_calls_endpoint_once_and_cannot_launch_outer(harness):
    result, gate = run(harness)
    assert harness.calls == ['endpoint']
    assert result['artifact_audit']['neural_models'] == 30
    assert result['independent_replay']['risk_vectors_replayed'] == 300
    assert result['only_external_anchor_identical_between_arms'] is True
    assert gate['outer_admitted'] is False
    with pytest.raises(ValueError, match='existing evaluation'):
        run(harness)
    assert harness.calls == ['endpoint']


@pytest.mark.parametrize('tamper', ['missing_DONE', 'epochs', 'checkpoint_bytes', 'rebound_checkpoint',
                                  'replay_count', 'replay_rows', 'replay_script', 'qualification_source',
                                  'prepared', 'OFF', 'missing_forbidden_key'])
def test_tamper_fails_before_endpoint(harness, tamper):
    directory = harness.private/'prototype/seed31_fold1/C11'
    done = directory/'DONE_PRIVATE.json'; checkpoint = directory/'checkpoint.pt'
    replay = harness.report/'REPLAY_AUDIT.json'
    if tamper == 'missing_DONE': done.unlink()
    elif tamper == 'epochs':
        row = ev._json(done); row['epochs'] = 79; ev.write_json(done, row)
    elif tamper == 'checkpoint_bytes':
        with checkpoint.open('ab') as stream: stream.write(b'tamper')
    elif tamper == 'rebound_checkpoint':
        row = torch.load(checkpoint, weights_only=False); row['model_state']['SYNTHETIC_weight'] += 1
        put(directory, row)
    elif tamper in ('replay_count', 'replay_rows', 'missing_forbidden_key'):
        row = ev._json(replay)
        if tamper == 'replay_count': row['risk_vectors_replayed'] = 299
        elif tamper == 'replay_rows': row['rows'] = row['rows'][:-1]
        else: row['forbidden_API_call_counts'].pop(ev.REPLAY_FORBIDDEN[0])
        ev.write_json(replay, row)
    elif tamper == 'replay_script':
        (harness.root/'scripts/replay_phase28.py').write_text('changed')
    elif tamper == 'qualification_source':
        path = harness.report/'NUMERICAL_QUALIFICATION.json'; row = ev._json(path)
        row['source'] = 'wrong'; ev.write_json(path, row)
    elif tamper in ('prepared', 'OFF'):
        row = torch.load(checkpoint, weights_only=False)
        if tamper == 'prepared': row['prepared_sha256'] = 'f'*64
        else: row['valid_risks']['OFF'] = row['M0_valid'].copy()
        put(directory, row)
    with pytest.raises(ValueError): run(harness)
    assert harness.calls == []


def test_endpoint_failure_marker_prevents_silent_second_read(harness):
    with pytest.raises(ValueError, match='event count'):
        run(harness, expected_dss_events=0)
    assert harness.calls == ['endpoint']
    with pytest.raises(ValueError, match='previous evaluation attempt'):
        run(harness)
    assert harness.calls == ['endpoint']


class TinyReplayModel(nn.Module):
    """Explicit algebraic model for the independent prediction-loop harness."""
    def __init__(self):
        super().__init__(); self.coefficient = nn.Parameter(torch.tensor(.4, dtype=torch.float64))

    def encode(self, values, mask, gene_ids, slides):
        value = values.mean()
        path = [slide['features'].mean().expand(4, 128) for slide in slides]
        return {'own_gene': value, 'path0': path, 'path_valid': [torch.ones(len(p), dtype=torch.bool) for p in path],
                'gene_valid': torch.ones(108, dtype=torch.bool)}

    def read_encoded(self, encoded, offset, *, context=None, interaction_off=False,
                     constant_path_value=False, constant_gene_value=False):
        gene = encoded['own_gene']; path = torch.stack([p.mean() for p in encoded['path0']]).mean()
        source_gene = gene if context is None or 'gene' not in context[0] else context[0]['gene'].mean()
        source_path = path if context is None or 'path' not in context[0] else torch.stack([p.mean() for p in context[0]['path']]).mean()
        base = torch.as_tensor(offset, dtype=torch.float64)+self.coefficient*(gene+path)
        interaction = self.coefficient*.01*(path*source_gene+gene*source_path)
        if interaction_off or (constant_path_value and constant_gene_value): interaction = interaction*0
        states = [{'gene': gene.expand(108, 128), 'path': encoded['path0']} for _ in range(2)]
        return {'risk': base+interaction, 'base': base, 'states': states}


def tiny_oracle():
    ds = SimpleNamespace(fit=np.array([0, 1]), valid=np.array([2, 3]), prepared_sha256='c'*64,
        molecular=torch.stack([torch.full((2, 3), p+2., dtype=torch.float64) for p in range(4)]),
        molecular_mask=torch.ones(4, 2, 3, dtype=torch.bool), gene_ids=torch.arange(2),
        CP_valid=np.array([.02, .03]), RNA_valid=np.array([.04, .06]),
        base=torch.arange(4, dtype=torch.float64)*.03)
    ds.M0_valid = ds.CP_valid+ds.RNA_valid
    ds.load_patient_slides = lambda p, device=None: [{'features': torch.full((2, 1), p+1., dtype=torch.float64, device=device)}]
    wrong = np.array([3, 2]); theta = .4
    def score(p, *, donor_gene=None, donor_path=None, offset=None, off=False):
        gene, path = p+2., p+1.
        g = gene if donor_gene is None else donor_gene
        q = path if donor_path is None else donor_path
        result = float(ds.base[p]) if offset is None else offset
        result += theta*(gene+path)
        return result if off else result+.01*theta*(path*g+gene*q)
    risks = {key: [] for key in ev.CONTROLS}
    for p, j in zip(ds.valid, wrong):
        values = {'MATCHED': score(p), 'OFF': score(p, off=True), 'CONSTANT_VALUES': score(p, off=True),
            'GENE_MEAN': score(p, donor_gene=2.5), 'PATH_MEAN': score(p, donor_path=1.5),
            'GENE_WRONG': score(p, donor_gene=j+2.), 'PATH_WRONG': score(p, donor_path=j+1.),
            'BOTH_WRONG': score(p, donor_gene=j+2., donor_path=j+1.)}
        # Whole omics also changes the own molecular main effect, unlike the
        # query-only controls. Derive its oracle separately.
        own_path, foreign_gene = p+1., j+2.
        offset = ds.CP_valid[p-2]+ds.RNA_valid[j-2]
        values['WHOLE_OMICS_WRONG'] = offset+theta*(own_path+foreign_gene)+.02*theta*own_path*foreign_gene
        for key in ev.CONTROLS: risks[key].append(values[key])
    record = {'fit': ds.fit.copy(), 'valid': ds.valid.copy(), 'prepared_sha256': ds.prepared_sha256,
        'CP_valid': ds.CP_valid.copy(), 'RNA_valid': ds.RNA_valid.copy(), 'M0_valid': ds.M0_valid.copy(),
        'fit_risk': np.array([score(p) for p in ds.fit]), 'valid_risks': {k: np.array(v) for k, v in risks.items()},
        'wrong_context_mapping': wrong}
    return record, ds, TinyReplayModel(), wrong


def test_independent_prediction_loop_matches_scalar_oracle_all_controls():
    record, ds, model, wrong = tiny_oracle()
    result = rp.replay_record(record, ds, model, wrong_map=wrong)
    assert result['risk_vectors_replayed'] == 10 and result['reference_vectors_checked'] == 3
    assert result['maximum_absolute_risk_error'] < 1e-12
    assert result['model_state_unchanged'] is True
    record['valid_risks']['PATH_WRONG'][0] += .01
    with pytest.raises(ValueError, match='replay differs'):
        rp.replay_record(record, ds, model, wrong_map=wrong)


def test_replay_refuses_supervision_in_view_and_requires_exact_main():
    record, ds, model, wrong = tiny_oracle(); ds._fit_event = np.ones(2)
    with pytest.raises(ValueError, match='supervision'):
        rp.replay_record(record, ds, model, wrong_map=wrong)
    del ds._fit_event; record['CP_valid'][0] += 1e-10
    with pytest.raises(ValueError, match='replay differs'):
        rp.replay_record(record, ds, model, wrong_map=wrong)

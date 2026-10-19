"""Outcome statistics for the locked Phase28 prototype, without model fitting.

These pure statistics accept already verified, patient-aligned private risks.
Artifact/replay integration is deliberately separate and must precede endpoint
access. OFF is the same checkpoint's external anchor plus learned main heads;
it is never substituted with M0. No outer admission or bootstrap is implemented.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.phase10_trimodal.reference import metrics, uno_weights
from src.phase21_relative.evaluation import STATISTICS, TRANSITIONS, pair_transition_summary
from src.phase25_local.evaluation import _array, _endpoint, _finite_tree, require, sha256_file, write_json


SEEDS = (31, 47)
FOLDS = (1, 2, 3)
ARMS = ('C00', 'C10', 'C01', 'C11', 'DIRECT')
METRICS = ('Harrell', 'Uno60')
CONTROLS = ('MATCHED', 'OFF', 'GENE_MEAN', 'GENE_WRONG', 'PATH_MEAN',
            'PATH_WRONG', 'BOTH_WRONG', 'CONSTANT_VALUES', 'WHOLE_OMICS_WRONG')
GATE_CONTROLS = ('GENE_MEAN', 'GENE_WRONG', 'PATH_MEAN', 'PATH_WRONG',
                 'BOTH_WRONG', 'CONSTANT_VALUES')
RISK_NAMES = ('M0', 'CP', 'RNA', *ARMS,
              *(arm+'_'+control for arm in ARMS for control in CONTROLS if control != 'MATCHED'))
PRIMARY_COMPARISONS = (('C11', 'C00'), ('C11', 'DIRECT'))
DIRECTION_COMPARISONS = (('C11', 'C10'), ('C11', 'C01'))
PAIR_COMPARISONS = (*PRIMARY_COMPARISONS, *DIRECTION_COMPARISONS,
                    ('C11', 'M0'), ('C11', 'CP'), ('C00', 'M0'),
                    *(('C11', 'C11_'+control) for control in CONTROLS if control != 'MATCHED'))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPLAY_SCHEMA = 'phase28_independent_replay_v1'
REPLAY_FORBIDDEN = ('Data.fit_labels', 'train.train_cell', 'train.predict', 'train.forward_patients',
                    'train.risk_gradient_cache', 'train.efron_loss', 'Optimizer.__init__',
                    'Tensor.backward', 'autograd.grad', 'parent.endpoints',
                    'np.load_endpoint', 'central.forbidden_key')


def evaluator_identity():
    paths = ('src/phase28_reread/evaluation.py', 'scripts/evaluate_phase28.py',
             'src/phase28_reread/replay.py', 'scripts/replay_phase28.py',
             'src/phase10_trimodal/reference.py', 'src/phase21_relative/evaluation.py',
             'src/phase25_local/evaluation.py')
    hashes = {name: sha256_file(PROJECT_ROOT/name) for name in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _json(path):
    require(Path(path).is_file(), 'required evidence file missing: '+Path(path).name)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _relative_script(root, value):
    require(isinstance(value, str) and value and not Path(value).is_absolute()
            and '..' not in Path(value).parts and ':' not in value and '\\' not in value,
            'qualification script must be a repository-relative path')
    path = (Path(root)/value).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), 'qualification executable missing/outside repo')
    return path


def verify_qualifications(report_dir, source, source_files, *, project_root=None, private_root=None):
    """Bind input-specific evidence and both full training-closure certificates.

    Training source excludes evaluator/replay/qualification reports to avoid a
    hash cycle. Their executable byte hashes are bound independently here.
    """
    root, report_dir = Path(project_root or PROJECT_ROOT), Path(report_dir)
    require(isinstance(source_files, dict) and bool(source_files), 'training closure inventory missing')
    require(all(sha256_file(_relative_script(root, name)) == digest for name, digest in source_files.items()),
            'current training closure file SHA drift')
    rows = {}
    for name in ('DEPENDENCY_QUALIFICATION', 'NUMERICAL_QUALIFICATION'):
        path = report_dir/(name+'.json'); report = _json(path)
        require(str(report.get('status', '')).startswith('PASS') and report.get('source') == source
                and report.get('source_files') == source_files, name+' source/closure/status failed')
        script = _relative_script(root, report.get('script_path'))
        require(report.get('script_sha256') == sha256_file(script), name+' executable SHA drift')
        rows[name] = {'report_sha256': sha256_file(path), 'script_path': report['script_path'],
                      'script_sha256': report['script_sha256'], 'source': source, 'status': report['status']}
    path = report_dir/'INPUT_AUDIT.json'; report = _json(path)
    require(report.get('status') == 'PASS' and report.get('stage') == 'prototype_input_qualification_only',
            'input qualification status/stage failed')
    expected = {'data': 'src/phase28_reread/data.py', 'spatial': 'src/phase28_reread/spatial.py',
                'audit_script': 'scripts/audit_phase28_inputs.py'}
    require(set(report.get('input_code_sha256', {})) == set(expected), 'input implementation inventory differs')
    for key, relative in expected.items():
        require(report['input_code_sha256'][key] == sha256_file(root/relative), 'input implementation SHA drift: '+key)
    require(report.get('audit_script_sha256') == report['input_code_sha256']['audit_script'], 'input script SHA mismatch')
    for key in ('new_neural_training_steps', 'new_predictor_fits', 'optimizer_constructions',
                'neural_forward_calls', 'valid_endpoint_array_reads', 'fit_label_getter_calls'):
        require(type(report.get(key)) is int and report[key] == 0, 'input isolation evidence failed: '+key)
    calls = report.get('forbidden_API_call_counts')
    require(isinstance(calls, dict) and bool(calls) and all(type(v) is int and v == 0 for v in calls.values()),
            'input forbidden-call evidence missing/failed')
    folds = report.get('folds')
    require(isinstance(folds, list) and len(folds) == 3 and {r.get('fold') for r in folds} == set(FOLDS),
            'input fold coverage differs')
    prepared = {str(row['fold']): row['prepared_sha256'] for row in folds}
    registry_path = Path(private_root)/'H5_SOURCES_PRIVATE.json' if private_root is not None else None
    require(registry_path is not None and registry_path.is_file()
            and report.get('private_registry_sha256') == sha256_file(registry_path), 'private input registry SHA differs')
    supplement = report.get('geometry_risks')
    if supplement is not None:
        require(supplement.get('supplement_script_sha256') == sha256_file(root/'scripts/supplement_phase28_geometry.py'),
                'geometry supplement executable SHA differs')
    rows['INPUT_AUDIT'] = {'status': 'PASS', 'report_sha256': sha256_file(path),
        'script_path': expected['audit_script'], 'script_sha256': report['audit_script_sha256'],
        'private_registry_sha256': report['private_registry_sha256'], 'prepared_sha256_by_fold': prepared,
        'fit_label_bytes_deserialized_in_prepared': report.get('fit_label_bytes_deserialized_in_prepared') is True}
    return rows


def _history_summary(history, fit_n):
    require(isinstance(history, list) and len(history) == 80, 'all 80 actual epoch records required')
    for epoch, row in enumerate(history, 1):
        require(isinstance(row, dict) and row.get('epoch') == epoch, 'epoch history order differs')
        _finite_tree(row, 'private epoch')
        require(row.get('patients') == fit_n and row.get('microbatch') == 1 and row.get('optimizer_steps') == 0,
                'history full-fit cache scope differs')
        require(type(row.get('events')) is int and row['events'] > 0, 'fit event count invalid')
        for key in ('cox_loss', 'regularizer', 'objective', 'maximum_absolute_component_replay_error', 'seconds'):
            require(type(row.get(key)) in (int, float) and np.isfinite(row[key]), 'missing finite epoch scalar: '+key)
        require(0 <= row['maximum_absolute_component_replay_error'] <= 1e-6, 'gradient cache replay error exceeds lock')
        require(abs(row['objective']-row['cox_loss']-row['regularizer']) <= 1e-7
                and row['regularizer'] >= 0 and row['seconds'] >= 0, 'epoch objective/penalty accounting differs')
        for key in ('data_gradient_L2_by_top_level_module', 'total_gradient_L2_by_top_level_module'):
            require(isinstance(row.get(key), dict) and bool(row[key]) and
                    all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in row[key].values()),
                    'separate data/total gradient diagnostics missing')
    values = np.array([row['objective'] for row in history], float)
    return {'actual_epochs': 80, 'first_preupdate_objective': float(values[0]),
            'last_preupdate_objective': float(values[-1]), 'last_preupdate_cox': history[-1]['cox_loss'],
            'last10_preupdate_objective_improvement': float(values[-11]-values[-1]),
            'training_seconds': float(sum(row['seconds'] for row in history)),
            'maximum_gradient_cache_error': max(row['maximum_absolute_component_replay_error'] for row in history),
            'last_data_gradient_L2_by_module': dict(history[-1]['data_gradient_L2_by_top_level_module']),
            'last_total_gradient_L2_by_module': dict(history[-1]['total_gradient_L2_by_top_level_module']),
            'global_convergence_claimed': False}


def _checkpoint(record, source, seed, fold, arm, cohort_size):
    require(isinstance(record, dict), 'checkpoint mapping required')
    for key, wanted in (('source', source), ('stage', 'prototype'), ('seed', seed), ('fold', fold), ('arm', arm)):
        require(record.get(key) == wanted, 'checkpoint identity differs: '+key)
    require(not set(record) & {'time', 'event', 'DSS', 'endpoint', 'endpoints', 'valid_time', 'valid_event'},
            'checkpoint contains heldout endpoints')
    fit, valid = (_array(record[key], key, integer=True) for key in ('fit', 'valid'))
    for name, indices in (('fit', fit), ('valid', valid)):
        require(indices.ndim == 1 and len(indices) and len(np.unique(indices)) == len(indices)
                and ((indices >= 0) & (indices < cohort_size)).all(), 'invalid '+name+' partition')
    require(not np.intersect1d(fit, valid).size, 'checkpoint fit/valid overlap')
    state = record.get('model_state')
    require(isinstance(state, dict) and bool(state) and all(isinstance(k, str) and isinstance(v, torch.Tensor)
            and torch.isfinite(v).all() for k, v in state.items()), 'finite saved model state required')
    require(record.get('model_init') == {'genes': 26708, 'path_dim': 1536, 'rank': 4}, 'saved model dimensions differ')
    require(set(record.get('valid_risks', {})) == set(CONTROLS), 'all nine saved controls required')
    risks = {key: _array(record[key+'_valid'], key, (len(valid),)) for key in ('M0', 'CP', 'RNA')}
    require(np.array_equal(risks['M0'], risks['CP']+risks['RNA']), 'external anchor is not exact saved CP+RNA')
    controls = {key: _array(record['valid_risks'][key], key, (len(valid),)) for key in CONTROLS}
    base = _array(record['valid_base_risk'], 'own learned main', (len(valid),))
    require(np.array_equal(controls['OFF'], base), 'OFF must equal own learned main, not external M0')
    if arm == 'C00':
        require(all(np.array_equal(controls['MATCHED'], controls[key]) for key in CONTROLS
                    if key != 'WHOLE_OMICS_WRONG'), 'C00 condition-only control changed risk')
    audit = record.get('training_audit', {})
    for key, wanted in (('actual_epochs', 80), ('optimizer_steps', 80), ('new_main_anchor_fits', 0),
                        ('complete_fit_risk_sets', True), ('fixed_final_epoch', True),
                        ('valid_endpoint_reads', 0), ('external_offset_unchanged', True)):
        require(audit.get(key) == wanted, 'training audit differs: '+key)
    fit_risk = _array(record['fit_risk'], 'fit risk', (len(fit),))
    summary = _history_summary(record['history'], len(fit))
    _finite_tree(record.get('data_audit'), 'private data audit')
    for key in ('prepared_sha256', 'initial_state_sha256'):
        require(isinstance(record.get(key), str) and len(record[key]) == 64, 'checkpoint lineage SHA missing: '+key)
    return {'fit': fit, 'valid': valid, 'risks': risks, 'controls': controls, 'fit_risk': fit_risk,
            'prepared_sha256': record['prepared_sha256'], 'initial_state_sha256': record['initial_state_sha256']}, summary


def verify_stage(private_root, stage, source, *, cohort_size=512):
    """Check every DONE/SHA before deserializing any of the thirty models."""
    require(stage == 'prototype', 'Phase28 outer execution is not implemented/admitted')
    entries, cells, rows = [], {}, []
    for seed in SEEDS:
        for fold in FOLDS:
            for arm in ARMS:
                directory = Path(private_root)/stage/f'seed{seed}_fold{fold}'/arm
                path, done = directory/'checkpoint.pt', directory/'DONE_PRIVATE.json'
                receipt = _json(done)
                for key, wanted in (('status', 'COMPLETE'), ('source', source), ('stage', stage), ('seed', seed),
                                    ('fold', fold), ('arm', arm), ('epochs', 80), ('optimizer_steps', 80)):
                    require(receipt.get(key) == wanted, 'all thirty DONE contracts required: '+key)
                require(path.is_file() and receipt.get('checkpoint_sha256') == sha256_file(path), 'checkpoint SHA differs')
                entries.append((seed, fold, arm, path, done, receipt))
    bindings = []
    for seed, fold, arm, path, done, receipt in entries:
        record = torch.load(path, map_location='cpu', weights_only=False)
        actual, summary = _checkpoint(record, source, seed, fold, arm, cohort_size)
        require(all(actual[key] == receipt.get(key) for key in ('prepared_sha256', 'initial_state_sha256')),
                'DONE/checkpoint source lineage differs')
        cell = cells.setdefault((seed, fold), {'fit': actual['fit'], 'valid': actual['valid'],
            'risks': actual['risks'], 'prepared_sha256': actual['prepared_sha256'],
            'initial_state_sha256': actual['initial_state_sha256']})
        require(all(np.array_equal(cell[key], actual[key]) for key in ('fit', 'valid')) and
                all(np.array_equal(cell['risks'][key], actual['risks'][key]) for key in ('M0', 'CP', 'RNA'))
                and cell['prepared_sha256'] == actual['prepared_sha256']
                and cell['initial_state_sha256'] == actual['initial_state_sha256'],
                'shared partition/external anchor/initialization differs across arms')
        cell['risks'][arm] = actual['controls']['MATCHED']
        for control in CONTROLS[1:]: cell['risks'][arm+'_'+control] = actual['controls'][control]
        binding = {'seed': seed, 'fold': fold, 'arm': arm, 'checkpoint_sha256': sha256_file(path),
                   'DONE_sha256': sha256_file(done), 'prepared_sha256': actual['prepared_sha256'],
                   'initial_state_sha256': actual['initial_state_sha256']}
        bindings.append(binding)
        rows.append({'seed': seed, 'fold': fold, 'arm': arm, 'fit_patients': len(actual['fit']),
                     'valid_patients': len(actual['valid']), 'training': summary})
        del record
    _validated_cells(cells, cohort_size)
    fingerprint = hashlib.sha256(json.dumps(bindings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return cells, {'status': 'PASS_COMPLETE_CHECKPOINT_CONTRACT', 'neural_models': 30, 'seed_partition_cells': 6,
                   'epochs_each': 80, 'optimizer_steps': 2400, 'new_main_anchor_fits': 0,
                   'receipt_fingerprint': fingerprint, 'model_bindings': bindings, 'rows': rows}


def verify_replay(path, source, artifact_audit, qualification, source_files, *, project_root=None):
    root = Path(project_root or PROJECT_ROOT); report = _json(path)
    for key, wanted in (('schema', REPLAY_SCHEMA), ('status', 'PASS'), ('source', source), ('stage', 'prototype'),
                        ('neural_models', 30), ('seed_partition_cells', 6), ('risk_vectors_replayed', 300),
                        ('reference_vectors_checked', 90), ('new_predictor_fits', 0), ('optimizer_steps', 0),
                        ('endpoint_API_reads', 0), ('absolute_tolerance', 1e-6)):
        require(report.get(key) == wanted, 'independent replay missing/changed: '+key)
    require(report.get('source_files') == source_files, 'replay training closure inventory differs')
    require(report.get('replay_script_sha256') == sha256_file(root/'scripts/replay_phase28.py')
            and report.get('replay_module_sha256') == sha256_file(root/'src/phase28_reread/replay.py'),
            'independent replay executable SHA drift')
    for key in ('maximum_absolute_risk_error',):
        value = report.get(key)
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1e-6, 'replay risk error exceeds lock')
    for key in ('private_checkpoints_and_receipts_unchanged', 'training_source_bytes_unchanged',
                'model_state_unchanged', 'external_references_elementwise_exact', 'no_valid_endpoints_loaded'):
        require(report.get(key) is True, 'replay invariant failed: '+key)
    calls = report.get('forbidden_API_call_counts')
    require(isinstance(calls, dict) and set(calls) == set(REPLAY_FORBIDDEN)
            and all(type(v) is int and v == 0 for v in calls.values()), 'replay forbidden API isolation failed')
    require(report.get('qualification_report_hashes') == {key: value['report_sha256'] for key, value in qualification.items()},
            'replay qualification binding drift')
    rows = report.get('rows')
    require(isinstance(rows, list) and len(rows) == 30, 'replay model binding rows incomplete')
    by = {(row.get('seed'), row.get('fold'), row.get('arm')): row for row in rows}
    require(len(by) == 30, 'duplicate replay model rows')
    for binding in artifact_audit['model_bindings']:
        key = binding['seed'], binding['fold'], binding['arm']
        require(key in by and all(by[key].get(field) == value for field, value in binding.items()),
                'replay/current checkpoint or prepared binding differs')
        require(by[key].get('risk_vectors_replayed') == 10 and by[key].get('reference_vectors_checked') == 3,
                'replay per-model vector coverage differs')
        value = by[key].get('maximum_absolute_risk_error')
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1e-6, 'invalid model replay error')
    return {'status': 'PASS', 'report_sha256': sha256_file(path), 'replay_script_sha256': report['replay_script_sha256'],
            'replay_module_sha256': report['replay_module_sha256'], 'neural_models': 30,
            'risk_vectors_replayed': 300, 'reference_vectors_checked': 90,
            'maximum_absolute_risk_error': report['maximum_absolute_risk_error'], 'source': source}


def _seed_table(seed_rows):
    require(isinstance(seed_rows, list), 'seed/model rows must be a list')
    require(all(isinstance(row, dict) and 'seed' in row and 'model' in row for row in seed_rows),
            'invalid seed/model row')
    by = {(row['seed'], row['model']): row for row in seed_rows}
    required = {(seed, model) for seed in SEEDS for model in RISK_NAMES}
    require(len(by) == len(seed_rows) and set(by) == required,
            'complete unique seed/model coverage required')
    for row in by.values():
        for metric in METRICS:
            value = row.get(metric)
            require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1,
                    'seed concordance must be finite in [0,1]')
    return by


def prototype_gate(seed_rows, *, source, receipt_fingerprint, evaluator_sha256=None):
    """Apply only the predeclared C11 tests; diagnostic arms cannot be promoted."""
    by = _seed_table(seed_rows)

    def compare(candidate, baseline, strict):
        rows = []
        for metric in METRICS:
            deltas = np.array([by[seed, candidate][metric]-by[seed, baseline][metric] for seed in SEEDS])
            passed = (deltas.mean() > 0 and np.all(deltas > 0)) if strict else deltas.mean() >= 0
            rows.append({'comparison': candidate+'-'+baseline, 'metric': metric,
                         'mean_seed_delta': float(deltas.mean()),
                         'seed_deltas': [{'seed': seed, 'delta': float(delta)} for seed, delta in zip(SEEDS, deltas)],
                         'positive_seeds': int((deltas > 0).sum()), 'seeds': len(SEEDS), 'pass': bool(passed)})
        return rows

    primary = [row for pair in PRIMARY_COMPARISONS for row in compare(*pair, strict=True)]
    guards = compare('C11', 'M0', False)+compare('C11', 'CP', False)
    pairing = []
    for control in GATE_CONTROLS:
        deltas = {metric: float(np.mean([by[seed, 'C11'][metric]-by[seed, 'C11_'+control][metric]
                                       for seed in SEEDS])) for metric in METRICS}
        pairing.append({'comparison': 'C11-C11_'+control, 'mean_seed_deltas': deltas,
                        'pass': all(delta >= 0 for delta in deltas.values())
                                and any(delta > 0 for delta in deltas.values())})
    passed = all(row['pass'] for row in primary+guards+pairing)
    directions = [row for pair in DIRECTION_COMPARISONS for row in compare(*pair, strict=True)]
    factorial = []
    for metric in METRICS:
        rows = [{'seed': seed, 'value': float(by[seed, 'C11'][metric]-by[seed, 'C10'][metric]
                                             -by[seed, 'C01'][metric]+by[seed, 'C00'][metric])}
                for seed in SEEDS]
        factorial.append({'metric': metric, 'formula': 'C11-C10-C01+C00', 'seed_contrasts': rows,
                          'mean_seed_contrast': float(np.mean([row['value'] for row in rows])),
                          'diagnostic_only': True, 'biological_synergy_claimed': False})
    return {'status': 'PROTOTYPE_GO_REQUIRES_NEW_PROTOCOL' if passed else 'PROTOTYPE_NOT_ADMITTED',
            'source': source, 'stage': 'prototype', 'decision': 'GO' if passed else 'STOP',
            'classification': 'GO' if passed else 'STOP', 'candidate': 'C11',
            'primary_tests': primary, 'main_effect_guards': guards, 'frozen_pairing_guards': pairing,
            'direction_comparisons_diagnostic_only': directions, 'factorial_contrasts_diagnostic_only': factorial,
            'OFF_and_WHOLE_OMICS_WRONG_are_diagnostic_only': True,
            'outer_admitted': False, 'outer_automatically_started': False,
            'separate_new_protocol_required': True, 'automatic_alternate_arm_promotion': False,
            'automatic_hyperparameter_or_architecture_search': False,
            'confirmatory_confidence_intervals': False,
            'receipt_fingerprint': receipt_fingerprint, 'evaluator_sha256': evaluator_sha256,
            'scope': 'repeated_inner_DSS_architecture_screen_not_independent_confirmation_or_medical_causality'}


def _validated_cells(cells, cohort_size):
    require(isinstance(cells, dict) and set(cells) == {(seed, fold) for seed in SEEDS for fold in FOLDS},
            'complete six seed/partition cells required')
    clean = {}
    for (seed, fold), cell in cells.items():
        require(isinstance(cell, dict), 'cell mapping required')
        fit, valid = (_array(cell[key], key, integer=True) for key in ('fit', 'valid'))
        for name, indices in (('fit', fit), ('valid', valid)):
            require(indices.ndim == 1 and len(indices) > 0 and len(np.unique(indices)) == len(indices),
                    name+' must be unique and nonempty')
            require(((indices >= 0) & (indices < cohort_size)).all(), name+' outside cohort')
        require(not np.intersect1d(fit, valid).size, 'fit/valid overlap')
        require(isinstance(cell.get('risks'), dict) and set(cell['risks']) == set(RISK_NAMES),
                'all matched and frozen-control risks required')
        risks = {name: _array(cell['risks'][name], name, (len(valid),)) for name in RISK_NAMES}
        clean[seed, fold] = {'fit': fit, 'valid': valid, 'risks': risks}
    for fold in FOLDS:
        first, second = (clean[seed, fold] for seed in SEEDS)
        require(all(np.array_equal(first[key], second[key]) for key in ('fit', 'valid')),
                'patient partition changed across seeds')
        require(all(np.array_equal(first['risks'][name], second['risks'][name]) for name in ('M0', 'CP', 'RNA')),
                'external main risk changed across seeds')
    return clean


def point_results(cells, endpoint, *, cohort_size=512):
    """Sum three within-fold num/den per seed, then average seed concordances."""
    cells = _validated_cells(cells, cohort_size)
    time, event = _endpoint(endpoint, cohort_size)
    fold_rows, seed_rows = [], []
    for seed in SEEDS:
        totals = {}
        for fold in FOLDS:
            cell = cells[seed, fold]; fit, valid = cell['fit'], cell['valid']
            weights = uno_weights(time[fit], event[fit], time[valid], event[valid], tau=60.)
            for name in RISK_NAMES:
                values = metrics(time[valid], event[valid], cell['risks'][name],
                                 np.full(len(valid), fold), weights)
                require(all(np.isfinite(value) for value in values.values()), 'undefined within-fold concordance')
                fold_rows.append({'seed': seed, 'fold': fold, 'model': name, 'valid_patients': len(valid),
                                  'valid_events': int(event[valid].sum()), **values})
                total = totals.setdefault(name, dict.fromkeys(('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den'), 0.))
                for key in total: total[key] += values[key]
        for name, values in totals.items():
            require(values['Harrell_den'] > 0 and values['Uno_den'] > 0, 'zero seed concordance denominator')
            seed_rows.append({'seed': seed, 'model': name, **values,
                              'Harrell': values['Harrell_num']/values['Harrell_den'],
                              'Uno60': values['Uno_num']/values['Uno_den']})
    _seed_table(seed_rows)
    return {'fold_rows': fold_rows, 'seed_rows': seed_rows,
            'two_seed_means': [{'model': name, **{metric: float(np.mean([row[metric] for row in seed_rows
                                                                       if row['model'] == name]))
                                                for metric in METRICS}} for name in RISK_NAMES],
            'aggregation': 'sum_three_within_fold_num_den_per_seed_then_mean_two_seeds',
            'unique_patient_OOF': False, 'risk_ensemble': False, 'confirmatory_CI': False,
            'Uno_tau': 60., 'risk_tie_tolerance': 1e-8, 'IPCW_fit_scope': 'corresponding_fit_partition_only'}


def pair_results(cells, endpoint, point, *, cohort_size=512):
    """Same-fold repair/damage on all comparable-pair credit, not error subsets."""
    cells = _validated_cells(cells, cohort_size)
    time, event = _endpoint(endpoint, cohort_size)
    by = _seed_table(point['seed_rows'])
    fold_rows, seed_rows = [], []
    for candidate, baseline in PAIR_COMPARISONS:
        for seed in SEEDS:
            numerators = {metric: np.zeros(len(STATISTICS)) for metric in METRICS}
            denominators = dict.fromkeys(METRICS, 0.)
            for fold in FOLDS:
                cell = cells[seed, fold]; fit, valid = cell['fit'], cell['valid']
                weights = uno_weights(time[fit], event[fit], time[valid], event[valid], tau=60.)
                summary = pair_transition_summary(time[valid], event[valid], cell['risks'][baseline],
                    cell['risks'][candidate], folds=np.full(len(valid), fold), ipcw=weights)
                for row in summary['rows']:
                    metric = row['metric']
                    numerators[metric] += np.array([row[key] for key in STATISTICS])*row['denominator']
                    denominators[metric] += row['denominator']
                    fold_rows.append({'comparison': candidate+'-'+baseline, 'seed': seed, 'fold': fold, **row})
            for metric in METRICS:
                require(denominators[metric] > 0, 'zero paired denominator')
                values = dict(zip(STATISTICS, (numerators[metric]/denominators[metric]).tolist()))
                delta = by[seed, candidate][metric]-by[seed, baseline][metric]
                require(abs(values['net']-delta) < 1e-12, 'paired credit does not close to seed metric delta')
                require(abs(values['repair']-values['damage']-values['net']) < 1e-12,
                        'repair minus damage identity failed')
                require(abs(sum(values[key] for key in TRANSITIONS)-1.) < 1e-12,
                        'pair transition fractions do not sum to one')
                denominator = denominators[metric]
                seed_rows.append({'comparison': candidate+'-'+baseline, 'seed': seed, 'metric': metric,
                                  'denominator': denominator, **values,
                                  'repair_credit_mass': values['repair']*denominator,
                                  'damage_credit_mass': values['damage']*denominator,
                                  'net_credit_mass': values['net']*denominator})
    means = []
    for candidate, baseline in PAIR_COMPARISONS:
        for metric in METRICS:
            selected = [row for row in seed_rows if row['comparison'] == candidate+'-'+baseline
                        and row['metric'] == metric]
            means.append({'comparison': candidate+'-'+baseline, 'metric': metric,
                          **{key: float(np.mean([row[key] for row in selected])) for key in STATISTICS}})
    return {'fold_repair_damage': fold_rows, 'seed_repair_damage': seed_rows, 'repair_damage': means,
            'pair_analysis_scope': 'all_comparable_pair_credit_within_fold_only',
            'OFF_is_same_arm_external_plus_learned_main_not_M0': True,
            'wrong_patient_controls_are_frozen_not_retrained': True,
            'WHOLE_OMICS_WRONG_is_not_condition_only_or_an_admission_gate': True,
            'bootstrap_draws': 0}


def evaluate_stage(private_root, report_dir, stage, source, endpoint_loader, *, source_files,
                   baseline_loader, partition_loader, source_guard,
                   project_root=None, cohort_size=512, expected_dss_events=65):
    """Read DSS once only after all receipts, qualifications and replay bind.

    The caller's strict runtime qualification guard must already have passed;
    this function independently binds its report/closure executable evidence.
    A private exclusive attempt marker prevents silent repeat after any later
    failure, including a failure following the sole endpoint callback.
    """
    require(stage == 'prototype', 'outer requires a separate protocol; not implemented')
    root = Path(project_root or PROJECT_ROOT)
    private_root, report_dir = Path(private_root), Path(report_dir)
    require(not any((report_dir/name).exists() for name in ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json')),
            'existing evaluation requires explicit review; refusing overwrite')
    attempt = private_root/'EVALUATION_STARTED_PRIVATE.json'
    require(not attempt.exists(), 'previous evaluation attempt requires explicit review')
    source_guard()
    identity = evaluator_identity()
    qualification = verify_qualifications(report_dir, source, source_files, project_root=root, private_root=private_root)
    cells, artifact = verify_stage(private_root, stage, source, cohort_size=cohort_size)
    replay = verify_replay(report_dir/'REPLAY_AUDIT.json', source, artifact, qualification, source_files, project_root=root)
    provenance = []
    for fold in FOLDS:
        baseline = baseline_loader(fold)
        canonical = partition_loader(fold)
        prepared = qualification['INPUT_AUDIT']['prepared_sha256_by_fold'][str(fold)]
        require(baseline['prepared_sha256'] == prepared, 'input audit/current prepared SHA differs')
        for seed in SEEDS:
            cell = cells[seed, fold]
            require(cell['prepared_sha256'] == prepared, 'checkpoint/prepared provenance differs')
            for key in ('fit', 'valid'):
                require(np.array_equal(cell[key], np.asarray(baseline[key]))
                        and np.array_equal(cell[key], np.asarray(canonical[key])), 'canonical/prepared partition differs')
            for name in ('M0', 'CP', 'RNA'):
                require(np.array_equal(cell['risks'][name], np.asarray(baseline[name+'_valid'])),
                        'independent unchanged external reference differs')
        provenance.append({'fold': fold, 'prepared_sha256': prepared,
                           'status': 'PASS_exact_prepared_and_canonical_partition_and_external_main'})
    source_guard()
    attempt.parent.mkdir(parents=True, exist_ok=True)
    with attempt.open('x', encoding='utf-8') as stream:
        json.dump({'source': source, 'stage': stage, 'evaluator_sha256': identity,
                   'replay_report_sha256': replay['report_sha256'], 'receipt_fingerprint': artifact['receipt_fingerprint']}, stream)
    # This is the first and only callback allowed to request valid outcomes.
    endpoint = endpoint_loader()['DSS']
    _, event = _endpoint(endpoint, cohort_size)
    if expected_dss_events is not None:
        require(int(event.sum()) == expected_dss_events, 'DSS event count differs from locked cohort')
    point = point_results(cells, endpoint, cohort_size=cohort_size)
    paired = pair_results(cells, endpoint, point, cohort_size=cohort_size)
    gate = prototype_gate(point['seed_rows'], source=source, receipt_fingerprint=artifact['receipt_fingerprint'],
                          evaluator_sha256=identity)
    result = {'status': 'COMPLETE_BIDIRECTIONAL_REREAD_INNER_PROTOTYPE', 'source': source, 'stage': stage,
        'endpoint': 'DSS', 'cohort_patients': cohort_size, 'DSS_events': int(event.sum()),
        'artifact_audit': artifact, 'qualifications': qualification, 'independent_replay': replay,
        'external_main_and_partition_audit': provenance, **point, **paired,
        'evaluator_sha256': identity, 'new_predictor_fits_during_evaluation': 0,
        'valid_endpoint_callback_calls': 1, 'bootstrap_draws': 0,
        'only_external_anchor_identical_between_arms': True,
        'learned_unimodal_heads_can_drift_with_joint_training': True,
        'condition_only_controls_keep_own_original_values_learned_main_and_external_anchor': True,
        'Clinical_is_not_a_local_query_input': True,
        'patient_arrays_weights_or_IDs_public': False}
    source_guard()
    require(evaluator_identity() == identity, 'evaluation executable changed during run')
    write_json(report_dir/'INNER_RESULTS.json', result)
    write_json(report_dir/'INNER_GATE.json', gate)
    write_json(report_dir/'decision.json', {'status': gate['classification'], 'decision': gate['decision'],
        'source': source, 'stage': stage, 'candidate': 'C11', 'outer_admitted': False,
        'outer_started': False, 'confirmation_claimed': False, 'separate_new_protocol_required': True})
    write_json(private_root/'INNER_GATE.json', {'decision': gate['decision'], 'source': source, 'stage': stage,
        'outer_admitted': False, 'outer_automatically_started': False, 'evaluator_sha256': identity,
        'receipt_fingerprint': artifact['receipt_fingerprint'],
        'public_gate_sha256': sha256_file(report_dir/'INNER_GATE.json')})
    return result, gate

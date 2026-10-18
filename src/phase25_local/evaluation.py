"""Independent, fail-closed Phase25 prototype evaluation; never fits models.

Six hashed cells and all 24 model slots must pass before an endpoint callback
is requested. Repeated inner partitions do not form a unique patient OOF.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.phase10_trimodal.reference import metrics, uno_weights
from src.phase21_relative.evaluation import STATISTICS, TRANSITIONS, pair_transition_summary

SEEDS = (31, 47)
FOLDS = (1, 2, 3)
NEURAL_ARMS = ('R1', 'R2', 'R3')
METRICS = ('Harrell', 'Uno60')
SENSITIVITIES = {
    'R1': ('OFF',),
    'R2': ('OFF', 'RNA_MEAN', 'WRONG_RNA'),
    'R3': ('OFF', 'RNA_MEAN', 'WRONG_RNA', 'C_MEAN', 'WRONG_C'),
}
RISK_NAMES = ('R0', 'CP', *NEURAL_ARMS,
              *(arm+'_'+key for arm in NEURAL_ARMS for key in SENSITIVITIES[arm]))
MOMENT_FIELDS = {'count', 'mean', 'population_SD', 'RMS', 'minimum', 'maximum'}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluator_identity():
    root = Path(__file__).resolve().parents[2]
    names = ('src/phase25_local/evaluation.py', 'scripts/evaluate_phase25.py',
             'src/phase25_local/data.py', 'src/phase13_rna/data.py',
             'src/phase10_trimodal/reference.py', 'src/phase21_relative/evaluation.py')
    hashes = {name: sha256_file(root/name) for name in names}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _array(value, name, shape=None, *, integer=False):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    require(np.issubdtype(result.dtype, np.number) and not np.iscomplexobj(result), name+' must be real numeric')
    require(np.isfinite(result).all(), name+' contains nonfinite values')
    if shape is not None:
        require(result.shape == shape, name+' shape mismatch')
    if integer:
        require(np.issubdtype(result.dtype, np.integer), name+' must contain integer indices')
    return np.array(result, dtype=np.int64 if integer else np.float64, copy=True)


def _finite_tree(value, name):
    """Validate anonymous saved training/mechanism diagnostics without exporting."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str), name+' diagnostic keys must be text')
            _finite_tree(child, name+'.'+key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_tree(child, name)
    elif isinstance(value, (float, np.floating)):
        require(np.isfinite(value), name+' contains nonfinite diagnostic')
    else:
        require(value is None or isinstance(value, (str, int, bool, np.integer, np.bool_)),
                name+' contains non-scalar diagnostic payload')


def _public_moments(value, name):
    require(isinstance(value, dict) and set(value) == MOMENT_FIELDS, name+' must contain reviewed aggregate moments only')
    require(isinstance(value['count'], int) and not isinstance(value['count'], bool) and value['count'] > 0,
            name+' aggregate count invalid')
    for key in MOMENT_FIELDS-{'count'}:
        require(isinstance(value[key], (int, float)) and not isinstance(value[key], bool) and np.isfinite(value[key]),
                name+' aggregate scalar invalid')
    require(value['population_SD'] >= 0 and value['RMS'] >= 0 and value['maximum'] >= value['minimum'],
            name+' aggregate range invalid')
    return dict(value)


def _mechanism_summary(value, arm, fit_n, valid_n):
    expected = {'fit_unit_residual', 'valid_unit_residual', 'attention', 'frozen_query_changes'}
    require(isinstance(value, dict) and set(value) == expected, arm+' mechanism fields are not reviewed aggregates')
    result = {key: _public_moments(value[key], arm+' '+key) for key in ('fit_unit_residual', 'valid_unit_residual')}
    require(result['fit_unit_residual']['count'] == fit_n and result['valid_unit_residual']['count'] == valid_n,
            arm+' residual moments must use one row per patient')
    attention = value['attention']
    require(isinstance(attention, dict) and set(attention) == {'normalized_entropy', 'effective_patch_fraction', 'TV_from_uniform'},
            arm+' attention fields are not reviewed aggregates')
    result['attention'] = {key: _public_moments(v, arm+' '+key) for key, v in attention.items()}
    changes = value['frozen_query_changes']
    require(isinstance(changes, dict) and set(changes) == set(SENSITIVITIES[arm]), arm+' query diagnostics incomplete')
    result['frozen_query_changes'] = {}
    for condition, changed in changes.items():
        wanted = {'patient_risk_difference_over_fit_offset_SD'} if condition == 'OFF' else {
            'patient_risk_difference_over_fit_offset_SD', 'attention_TV', 'top32_Jaccard', 'scope'}
        require(isinstance(changed, dict) and set(changed) == wanted, arm+' query diagnostic fields are not reviewed')
        out = {key: _public_moments(v, arm+' '+condition+' '+key) for key, v in changed.items() if key != 'scope'}
        require(out['patient_risk_difference_over_fit_offset_SD']['count'] == valid_n,
                arm+' query-risk moments must be patient level')
        if 'scope' in changed:
            require(changed['scope'] == 'WSI_weighted_same_valid_patches_not_histological_ground_truth',
                    arm+' unreviewed query diagnostic scope')
            out['scope'] = changed['scope']
        result['frozen_query_changes'][condition] = out
    result['attention_weighting'] = 'WSI_equal_as_stored;not_patient_equal_or_histological_ground_truth'
    return result


def _history_summary(history):
    groups = tuple(history[0]['gradients'])
    require(all(tuple(row['gradients']) == groups for row in history), 'training gradient groups changed')
    gradients = {}
    for group in groups:
        values = np.array([r['gradients'][group] for r in history], float)
        require(np.isfinite(values).all() and (values >= 0).all(), 'invalid gradient norm history')
        gradients[group] = {'epochs': 80, 'zero_epochs': int((values == 0).sum()),
                            'mean_norm': float(values.mean()), 'first_norm': float(values[0]),
                            'last_norm': float(values[-1]), 'maximum_norm': float(values.max())}
    loss = np.array([r['fit_loss'] for r in history], float)
    return {'actual_epochs': 80, 'first_training_loss': float(history[0]['training_loss']),
            'first_updated_fit_loss': float(loss[0]), 'final_fit_loss': float(loss[-1]),
            'final_fit_cox': float(history[-1]['fit_cox']),
            'final_unit_residual_RMS': float(history[-1]['unit_residual_RMS']),
            'last10_fit_loss_improvement': float(loss[-11]-loss[-1]),
            'last10_relative_fit_loss_improvement': float((loss[-11]-loss[-1])/max(abs(loss[-11]), 1e-12)),
            'last10_fit_loss_slope': float(np.polyfit(np.arange(10), loss[-10:], 1)[0]),
            'gradient_groups': gradients, 'training_seconds': float(sum(r['epoch_seconds'] for r in history)),
            'global_convergence_or_generalization_claimed': False}


def _validate_checkpoint(record, *, source, stage, seed, fold, cohort_size):
    for key, expected in (('source', source), ('stage', stage), ('seed', seed), ('fold', fold)):
        require(record.get(key) == expected, 'checkpoint identity mismatch: '+key)
    require(not set(record).intersection({'time', 'event', 'DSS', 'endpoint', 'endpoints'}),
            'checkpoint cannot contain endpoint arrays')
    fit, valid = [_array(record[key], key, integer=True) for key in ('fit', 'valid')]
    for name, index in (('fit', fit), ('valid', valid)):
        require(index.ndim == 1 and len(index) and len(np.unique(index)) == len(index), name+' requires unique patients')
        require((index >= 0).all() and (index < cohort_size).all(), name+' indices outside cohort')
    require(not np.intersect1d(fit, valid).size, 'fit/valid patient leakage')
    risks = {arm: _array(record[arm+'_valid'], arm+' valid', (len(valid),)) for arm in ('R0', 'CP')}
    require(set(record.get('arms', {})) == set(NEURAL_ARMS), 'all three neural arms required')
    training = {}
    for arm in NEURAL_ARMS:
        cell = record['arms'][arm]
        risks[arm] = _array(cell['risk'], arm+' risk', (len(valid),))
        _array(cell['fit_risk'], arm+' fit risk', (len(fit),))
        state = cell.get('model_state')
        require(isinstance(state, dict) and state, arm+' model state missing')
        require(all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in state.values()),
                arm+' model state must contain finite tensors')
        history = cell.get('history')
        require(isinstance(history, list) and len(history) == 80, arm+' requires 80 history rows')
        for epoch, row in enumerate(history, 1):
            require(isinstance(row, dict) and row.get('epoch') == epoch, arm+' epoch sequence differs')
            _finite_tree(row, arm+' history')
            require(all(key in row for key in ('training_loss', 'fit_loss', 'fit_cox', 'unit_residual_RMS',
                                               'grad_norm', 'gradients', 'epoch_seconds')),
                    arm+' history objective/gradient fields missing')
        audit = cell.get('training_audit', {})
        for key, expected in (('actual_epochs', 80), ('optimizer_steps', 80), ('heldout_endpoint_reads', 0),
                              ('fixed_final_epoch', True), ('main_offset_unchanged', True),
                              ('fit_offset_kind', 'strict_threefold_crossfit_R0')):
            require(audit.get(key) == expected, arm+' training isolation audit differs: '+key)
        counts = cell.get('parameter_counts')
        require(isinstance(counts, dict) and counts and all(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and v >= 0 for v in counts.values()),
            arm+' parameter counts invalid')
        require(any(v > 0 for v in counts.values()), arm+' must have active parameters')
        saved = cell.get('sensitivities', {})
        require(set(saved) == set(SENSITIVITIES[arm]), arm+' frozen query sensitivity set differs')
        for name in SENSITIVITIES[arm]:
            risks[arm+'_'+name] = _array(saved[name], arm+'_'+name, (len(valid),))
        require(np.array_equal(risks[arm+'_OFF'], risks['R0']), arm+' OFF must equal its own frozen R0')
        mechanism = _mechanism_summary(cell.get('mechanism', {}), arm, len(fit), len(valid))
        training[arm] = {**_history_summary(history), 'parameter_counts': dict(counts), 'mechanism': mechanism}
    require(set(risks) == set(RISK_NAMES), 'missing risk slot')
    return {'source': source, 'stage': stage, 'seed': seed, 'fold': fold,
            'fit': fit, 'valid': valid, 'risks': risks}, training


def verify_stage(private_root, stage, source, *, cohort_size=512):
    require(stage == 'prototype', 'outer requires separately locked admitted evaluation; prototype only')
    private_root = Path(private_root)
    entries = []
    for seed in SEEDS:
        for fold in FOLDS:
            folder = private_root/stage/f'seed{seed}_fold{fold}'
            receipt, checkpoint = folder/'DONE_PRIVATE.json', folder/'checkpoint.pt'
            require(receipt.is_file() and checkpoint.is_file(), 'missing complete prototype cell')
            saved = json.loads(receipt.read_text(encoding='utf-8-sig'))
            for key, expected in (('status', 'COMPLETE'), ('source', source), ('stage', stage),
                                  ('seed', seed), ('fold', fold), ('model_slots', 4), ('epochs', 80),
                                  ('neural_models', 3), ('optimizer_steps', 240),
                                  ('convex_main_shared_reference', True)):
                require(saved.get(key) == expected, 'cell receipt mismatch: '+key)
            require(saved.get('checkpoint_sha256') == sha256_file(checkpoint), 'checkpoint hash mismatch')
            entries.append((seed, fold, folder, sha256_file(receipt)))
    # No checkpoint is deserialized until every expected receipt/hash passed.
    cells, summaries = {}, []
    for seed, fold, folder, receipt_hash in entries:
        record = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
        cell, training = _validate_checkpoint(record, source=source, stage=stage, seed=seed,
                                              fold=fold, cohort_size=cohort_size)
        cells[seed, fold] = cell
        summaries.append({'seed': seed, 'fold': fold, 'receipt_sha256': receipt_hash,
                          'fit_patients': len(cell['fit']), 'valid_patients': len(cell['valid']),
                          'neural_training': training})
    for fold in FOLDS:
        left, right = cells[SEEDS[0], fold], cells[SEEDS[1], fold]
        for key in ('fit', 'valid'):
            require(np.array_equal(left[key], right[key]), 'partition differs across neural seeds')
        for key in ('R0', 'CP'):
            require(np.array_equal(left['risks'][key], right['risks'][key]), 'shared convex baseline differs across seeds')
    return cells, {'status': 'PASS_SAVED_ARTIFACT_CONTRACT', 'complete_cells': 6,
                   'evaluation_model_slots': 24, 'neural_models': 18, 'epochs_each': 80,
                   'unique_fullfit_R0_models': 3, 'R0_crossfit_teacher_models': 9, 'unique_CP_guard_models': 3,
                   'receipt_fingerprint': hashlib.sha256(''.join(x[3] for x in entries).encode()).hexdigest(),
                   'endpoint_callback_called_during_checks': False, 'encoder_replay_claimed': False,
                   'rows': summaries}


def _endpoint(value, cohort_size):
    t, e = (_array(value[key], 'DSS '+key, (cohort_size,)) for key in ('time', 'event'))
    valid = np.asarray(value['valid'])
    require(valid.dtype == np.dtype(bool) and valid.shape == (cohort_size,) and valid.all(),
            'DSS cannot exclude a locked patient')
    require((t >= 0).all() and np.isin(e, [0, 1]).all(), 'DSS endpoint invalid')
    return t, e


def point_results(cells, endpoint, *, cohort_size=512):
    t, e = _endpoint(endpoint, cohort_size)
    fold_rows, seed_rows = [], []
    for seed in SEEDS:
        totals = {}
        for fold in FOLDS:
            cell = cells[seed, fold]
            fit, valid = cell['fit'], cell['valid']
            weights = uno_weights(t[fit], e[fit], t[valid], e[valid], tau=60.)
            for name in RISK_NAMES:
                result = metrics(t[valid], e[valid], cell['risks'][name], np.full(len(valid), fold), weights)
                require(all(np.isfinite(v) for v in result.values()), 'undefined prototype concordance')
                fold_rows.append({'seed': seed, 'fold': fold, 'model': name,
                                  'valid_patients': len(valid), 'valid_events': int(e[valid].sum()), **result})
                total = totals.setdefault(name, {k: 0. for k in ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den')})
                for key in total:
                    total[key] += result[key]
        for name, total in totals.items():
            seed_rows.append({'seed': seed, 'model': name, **total,
                              'Harrell': total['Harrell_num']/total['Harrell_den'],
                              'Uno60': total['Uno_num']/total['Uno_den']})
    return {'fold_rows': fold_rows, 'seed_rows': seed_rows,
            'two_seed_means': [{'model': name, **{metric: float(np.mean([r[metric] for r in seed_rows if r['model'] == name]))
                                                 for metric in METRICS}} for name in RISK_NAMES],
            'aggregation': 'sum_three_within_fold_num_den_per_seed_then_mean_two_seeds',
            'unique_patient_OOF': False, 'risk_ensemble': False, 'confirmatory_CI': False}


def prototype_gate(seed_rows, *, source, receipt_fingerprint, evaluator_sha256=None):
    require(len(seed_rows) == len(SEEDS)*len(RISK_NAMES), 'gate needs complete unique seed/model rows')
    by = {(r['seed'], r['model']): r for r in seed_rows}
    require(len(by) == len(seed_rows) and all((s, n) in by for s in SEEDS for n in RISK_NAMES),
            'gate seed/model coverage differs')
    require(all(np.isfinite(by[s, n][m]) for s in SEEDS for n in RISK_NAMES for m in METRICS),
            'gate requires finite metrics')

    def compare(candidate, control, positive):
        output = []
        for metric in METRICS:
            delta = np.array([by[s, candidate][metric]-by[s, control][metric] for s in SEEDS])
            passed = delta.mean() > 0 and (delta > 0).sum() == 2 if positive else delta.mean() >= 0
            output.append({'comparison': candidate+'-'+control, 'metric': metric,
                           'mean_seed_delta': float(delta.mean()), 'positive_seeds': int((delta > 0).sum()),
                           'seeds': 2, 'pass': bool(passed)})
        return output

    def pairing(arm, control):
        values = {m: float(np.mean([by[s, arm][m]-by[s, arm+'_'+control][m] for s in SEEDS])) for m in METRICS}
        return {'comparison': arm+'-'+arm+'_'+control, 'mean_seed_deltas': values,
                'pass': all(x >= 0 for x in values.values()) and any(x > 0 for x in values.values())}

    primary = compare('R2', 'R1', True)+compare('R3', 'R2', True)
    guards = compare('R3', 'R0', False)+compare('R3', 'CP', False)
    paired = [pairing('R3', c) for c in ('WRONG_RNA', 'WRONG_C', 'C_MEAN')]
    r1 = compare('R1', 'R0', True)+compare('R1', 'CP', False)
    r2 = compare('R2', 'R1', True)+compare('R2', 'R0', False)+compare('R2', 'CP', False)
    r2paired = pairing('R2', 'WRONG_RNA')
    r1effective, r2effective = all(x['pass'] for x in r1), all(x['pass'] for x in r2) and r2paired['pass']
    admitted = all(x['pass'] for x in primary+guards+paired)
    classification = 'GO' if admitted else 'R2_ONLY_DIAGNOSTIC' if r2effective else 'R1_ONLY_DIAGNOSTIC' if r1effective else 'STOP'
    return {'status': 'PROTOTYPE_GO' if admitted else 'PROTOTYPE_NOT_ADMITTED', 'source': source,
            'stage': 'prototype', 'classification': classification, 'decision': 'GO' if admitted else 'STOP',
            'primary_tests': primary, 'R3_anchor_guards': guards, 'R3_query_pairing_guards': paired,
            'R1_effective_tests': r1, 'R1_effective': r1effective, 'R2_effective_tests': r2,
            'R2_pairing_guard': r2paired, 'R2_effective': r2effective,
            'outer_admitted': admitted, 'outer_automatically_started': False,
            'automatic_hyperparameter_or_architecture_search': False, 'confirmatory_confidence_intervals': False,
            'receipt_fingerprint': receipt_fingerprint, 'evaluator_sha256': evaluator_sha256 or evaluator_identity(),
            'scope': 'overlapping_inner_partition_diagnostic;not_confirmed_generalization_or_medical_causality',
            'R1_R2_only_action': 'query_and_region_diagnostics_only;not_alternate_outer_candidate'}


def pair_results(cells, endpoint, point, *, cohort_size=512):
    t, e = _endpoint(endpoint, cohort_size)
    by = {(r['seed'], r['model']): r for r in point['seed_rows']}
    comparisons = [('R1', 'R0'), ('R1', 'CP'), ('R2', 'R1'), ('R2', 'R0'), ('R2', 'CP'),
                   ('R3', 'R2'), ('R3', 'R0'), ('R3', 'CP')]
    comparisons += [(a, a+'_'+c) for a in NEURAL_ARMS for c in SENSITIVITIES[a]]
    rows = []
    for candidate, baseline in comparisons:
        for seed in SEEDS:
            numerator = {m: np.zeros(len(STATISTICS)) for m in METRICS}
            denominator = {m: 0. for m in METRICS}
            for fold in FOLDS:
                cell = cells[seed, fold]
                fit, valid = cell['fit'], cell['valid']
                weights = uno_weights(t[fit], e[fit], t[valid], e[valid], tau=60.)
                value = pair_transition_summary(t[valid], e[valid], cell['risks'][baseline], cell['risks'][candidate],
                                               folds=np.full(len(valid), fold), ipcw=weights)
                for one in value['rows']:
                    m = one['metric']
                    numerator[m] += np.array([one[k] for k in STATISTICS])*one['denominator']
                    denominator[m] += one['denominator']
            for metric in METRICS:
                require(denominator[metric] > 0, 'undefined paired denominator')
                values = dict(zip(STATISTICS, (numerator[metric]/denominator[metric]).tolist()))
                require(abs(values['net']-(by[seed, candidate][metric]-by[seed, baseline][metric])) < 1e-12,
                        'paired credit differs from complete-seed metric delta')
                require(abs(values['repair']-values['damage']-values['net']) < 1e-12, 'repair minus damage differs')
                require(abs(sum(values[k] for k in TRANSITIONS)-1) < 1e-12, 'transition masses differ')
                rows.append({'comparison': candidate+'-'+baseline, 'seed': seed, 'metric': metric,
                             'denominator': denominator[metric], **values})
    means = []
    for candidate, baseline in comparisons:
        for metric in METRICS:
            part = [r for r in rows if r['comparison'] == candidate+'-'+baseline and r['metric'] == metric]
            means.append({'comparison': candidate+'-'+baseline, 'metric': metric,
                          **{k: float(np.mean([r[k] for r in part])) for k in STATISTICS}})
    return {'seed_repair_damage': rows, 'repair_damage': means,
            'pair_analysis_scope': 'descriptive_all_comparable_within_fold_credit;no_error_group_teacher'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def evaluate_stage(private_root, report_dir, stage, source, endpoint_loader, *, cohort_size=512,
                   expected_dss_events=65, baseline_loader=None, partition_loader=None):
    require(stage == 'prototype', 'outer requires separately locked admitted evaluation; prototype only')
    identity = evaluator_identity()
    cells, audit = verify_stage(private_root, stage, source, cohort_size=cohort_size)
    if baseline_loader is not None:
        for fold in FOLDS:
            expected = baseline_loader(fold)
            actual = cells[SEEDS[0], fold]
            for key in ('fit', 'valid'):
                require(np.array_equal(actual[key], np.asarray(expected[key])), 'independent baseline partition differs')
            for key in ('R0', 'CP'):
                require(np.array_equal(actual['risks'][key], np.asarray(expected[key+'_valid'])),
                        'independent convex baseline differs elementwise')
    # This is the first endpoint callback. Partial/tampered cells never reach it.
    endpoint = endpoint_loader()['DSS']
    _, event = _endpoint(endpoint, cohort_size)
    if expected_dss_events is not None:
        require(int(event.sum()) == expected_dss_events, 'DSS event count differs from locked cohort')
    if partition_loader is not None:
        for fold in FOLDS:
            expected = partition_loader(fold)
            for key in ('fit', 'valid'):
                require(np.array_equal(cells[SEEDS[0], fold][key], np.asarray(expected[key], int)),
                        'canonical prototype partition differs')
    point = point_results(cells, endpoint, cohort_size=cohort_size)
    paired = pair_results(cells, endpoint, point, cohort_size=cohort_size)
    gate = prototype_gate(point['seed_rows'], source=source, receipt_fingerprint=audit['receipt_fingerprint'],
                          evaluator_sha256=identity)
    result = {'status': 'COMPLETE_PROTOTYPE_DIAGNOSTIC', 'source': source, 'stage': stage,
              'endpoint': 'DSS', 'cohort_patients': cohort_size, 'DSS_events': int(event.sum()),
              'artifact_audit': audit, **point, **paired, 'new_training_fits': 0, 'bootstrap_draws': 0,
              'patient_arrays_or_weights_public': False, 'repeated_inner_patients_allowed': True,
              'evaluator_sha256': identity,
              'mechanism_rows': [{'seed': row['seed'], 'fold': row['fold'], 'arm': arm,
                                 'training': {k: v for k, v in summary.items() if k != 'mechanism'},
                                 'mechanism': summary['mechanism']}
                                for row in audit['rows'] for arm, summary in row['neural_training'].items()],
              'independent_convex_baseline_replay': 'PASS' if baseline_loader is not None else 'NOT_REQUESTED',
              'convex_replay_scope': 'cached_full_X_times_coefficients_and_teacher_state_hashes;not_raw_teacher_feature_replay'
                  if baseline_loader is not None else 'not_requested'}
    require(identity == evaluator_identity(), 'evaluation source drift')
    private_root, report_dir = Path(private_root), Path(report_dir)
    require(not any((report_dir/n).exists() for n in ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json')),
            'existing evaluation requires explicit review; refusing overwrite')
    write_json(report_dir/'INNER_RESULTS.json', result)
    write_json(report_dir/'INNER_GATE.json', gate)
    write_json(report_dir/'decision.json', {'status': gate['classification'], 'decision': gate['decision'],
               'source': source, 'stage': stage, 'outer_admitted': gate['outer_admitted'],
               'outer_started': False, 'confirmation_claimed': False,
               'scope': 'locked_prototype_only_no_automatic_repair_or_alternate_outer_candidate'})
    write_json(private_root/'INNER_GATE.json', {'decision': gate['decision'], 'classification': gate['classification'],
               'source': source, 'stage': stage, 'outer_admitted': gate['outer_admitted'],
               'outer_automatically_started': False, 'evaluator_sha256': identity,
               'receipt_fingerprint': audit['receipt_fingerprint'],
               'public_gate_sha256': sha256_file(report_dir/'INNER_GATE.json')})
    return result, gate

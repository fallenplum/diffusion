"""Frozen-protocol Phase26 evaluation: no fitting and no automatic outer run."""
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
NEURAL_ARMS = ('QUERY', 'ATTN', 'COMP')
METRICS = ('Harrell', 'Uno60')
CONTROLS = ('OFF', 'VALUE_OFF', 'WRONG_RNA_QUERY', 'RNA_MEAN_QUERY', 'WRONG_C_QUERY', 'C_MEAN_QUERY')
GATE_CONTROLS = ('VALUE_OFF', 'WRONG_RNA_QUERY', 'WRONG_C_QUERY', 'C_MEAN_QUERY')
RISK_NAMES = ('M0', 'CP', 'RNA', *NEURAL_ARMS,
              *(arm+'_'+control for arm in NEURAL_ARMS for control in CONTROLS))
MOMENT_FIELDS = {'count', 'mean', 'population_SD', 'RMS', 'minimum', 'maximum'}
GRADIENT_GROUPS = {'rna', 'patch', 'attention_norm', 'query', 'key', 'value', 'slot_down',
                   'clinical_down', 'clinical_up', 'update_ff', 'head'}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = PROJECT_ROOT/'scripts/replay_phase26.py'
MAIN_REPLAY_SCRIPT = PROJECT_ROOT/'scripts/replay_phase26_main.py'


def evaluator_identity():
    root = Path(__file__).resolve().parents[2]
    names = ('src/phase26_multiquery/evaluation.py', 'scripts/evaluate_phase26.py',
             'src/phase26_multiquery/data.py', 'configs/phase26_multiquery.json',
             'src/phase25_local/evaluation.py', 'src/phase10_trimodal/reference.py',
             'src/phase21_relative/evaluation.py')
    hashes = {name: sha256_file(root/name) for name in names}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _moments(value, name):
    require(isinstance(value, dict) and set(value) == MOMENT_FIELDS,
            name+' must contain reviewed aggregate moments only')
    require(type(value['count']) is int and value['count'] > 0, name+' aggregate count invalid')
    for key in MOMENT_FIELDS-{'count'}:
        require(type(value[key]) in (int, float) and np.isfinite(value[key]), name+' aggregate scalar invalid')
    require(value['population_SD'] >= 0 and value['RMS'] >= 0 and value['maximum'] >= value['minimum'],
            name+' invalid aggregate range')
    return dict(value)


def mechanism_summary(value, fit_n=None, valid_n=None):
    """Publish only explicitly reviewed numerical aggregate paths.

    Unknown keys fail closed. Raw patient/slot vectors and arbitrary strings
    are never passed through merely because they were called diagnostics.
    """
    require(isinstance(value, dict), 'mechanism must be a reviewed mapping')
    allowed = {'fit_unit_residual', 'valid_unit_residual', 'attention', 'slot', 'frozen_query_changes'}
    require(set(value) == allowed, 'unreviewed or incomplete mechanism field')
    out = {}
    for key, n in (('fit_unit_residual', fit_n), ('valid_unit_residual', valid_n)):
        out[key] = _moments(value[key], key)
        if n is not None: require(out[key]['count'] == n, key+' must be patient-level')
    for group, fields in {
        'attention': {'normalized_entropy', 'effective_patch_fraction', 'TV_from_uniform'},
        'slot': {'message_over_query_norm', 'pairwise_cosine', 'effective_rank', 'assignment_fraction'},
    }.items():
        require(isinstance(value[group], dict) and set(value[group]) == fields,
                'unreviewed '+group+' aggregate')
        out[group] = {k: _moments(v, group+'.'+k) for k, v in value[group].items()}
    changes = value['frozen_query_changes']
    require(isinstance(changes, dict) and set(changes) == set(CONTROLS), 'unreviewed or missing query control')
    out['frozen_query_changes'] = {}
    for control, row in changes.items():
        fields = {'patient_risk_difference_over_fit_offset_SD'}
        if control not in ('OFF', 'VALUE_OFF'): fields |= {'attention_TV', 'top32_Jaccard'}
        require(isinstance(row, dict) and set(row) == fields, 'unreviewed query change')
        out['frozen_query_changes'][control] = {k: _moments(v, control+'.'+k) for k, v in row.items()}
        if valid_n is not None:
            require(out['frozen_query_changes'][control]['patient_risk_difference_over_fit_offset_SD']['count'] == valid_n,
                    'risk difference must be patient-level')
    out['scope'] = 'patient_risk_moments;WSI_slot_attention_moments;not_histological_ground_truth'
    return out


def history_summary(history):
    require(isinstance(history, list) and len(history) == 80, 'complete 80 epoch history required')
    required = {'epoch', 'training_loss', 'fit_loss', 'fit_cox', 'grad_norm', 'gradients', 'epoch_seconds',
                'unit_residual_RMS', 'correction_RMS', 'saturation_fraction'}
    for i, row in enumerate(history, 1):
        require(isinstance(row, dict) and required == set(row) and row['epoch'] == i,
                'epoch history order/budget changed')
        _finite_tree(row, 'history')
        require(isinstance(row['gradients'], dict) and bool(row['gradients']), 'missing gradient groups')
        for key in required-{'epoch', 'gradients'}:
            require(type(row[key]) in (int, float) and np.isfinite(row[key]), 'invalid scalar history')
        require(row['epoch_seconds'] >= 0 and row['grad_norm'] >= 0, 'negative training diagnostic')
        require(row['unit_residual_RMS'] >= 0 and 0 <= row['correction_RMS'] <= .10000002
                and 0 <= row['saturation_fraction'] <= 1, 'invalid bounded correction diagnostic')
    groups = set(history[0]['gradients'])
    require(groups == GRADIENT_GROUPS, 'missing or unreviewed gradient group')
    require(all(set(row['gradients']) == groups for row in history), 'gradient groups changed')
    gradients = {}
    for key in sorted(groups):
        values = np.asarray([row['gradients'][key] for row in history], float)
        require(values.shape == (80,) and np.isfinite(values).all() and (values >= 0).all(),
                'invalid gradient norm')
        gradients[key] = {'epochs': 80, 'zero_epochs': int((values == 0).sum()),
                          'first_norm': float(values[0]), 'last_norm': float(values[-1]),
                          'mean_norm': float(values.mean()), 'maximum_norm': float(values.max())}
    loss = np.asarray([row['fit_loss'] for row in history], float)
    return {'actual_epochs': 80, 'first_training_loss': history[0]['training_loss'],
            'first_updated_fit_loss': float(loss[0]), 'final_fit_loss': float(loss[-1]),
            'final_fit_cox': history[-1]['fit_cox'],
            'final_unit_residual_RMS': history[-1]['unit_residual_RMS'],
            'final_correction_RMS': history[-1]['correction_RMS'],
            'final_saturation_fraction': history[-1]['saturation_fraction'],
            'last10_fit_loss_improvement': float(loss[-11]-loss[-1]),
            'last10_relative_fit_loss_improvement': float((loss[-11]-loss[-1])/max(abs(loss[-11]), 1e-12)),
            'training_seconds': float(sum(row['epoch_seconds'] for row in history)),
            'gradient_groups': gradients, 'global_convergence_or_generalization_claimed': False}


def _state(value, name):
    require(isinstance(value, dict) and bool(value), name+' missing')
    for key, tensor in value.items():
        require(isinstance(key, str) and isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all(),
                name+' must contain finite saved tensors')


def _validate_checkpoint(record, source, stage, seed, fold, cohort_size):
    require(isinstance(record, dict), 'checkpoint mapping required')
    for key, wanted in (('source', source), ('stage', stage), ('seed', seed), ('fold', fold)):
        require(record.get(key) == wanted, 'checkpoint identity mismatch: '+key)
    require(not set(record) & {'time', 'event', 'DSS', 'endpoint', 'endpoints'}, 'checkpoint contains endpoints')
    fit, valid = (_array(record[key], key, integer=True) for key in ('fit', 'valid'))
    for key, ix in (('fit', fit), ('valid', valid)):
        require(ix.ndim == 1 and len(ix) > 0 and len(np.unique(ix)) == len(ix), key+' not unique/nonempty')
        require(((ix >= 0) & (ix < cohort_size)).all(), key+' outside locked cohort')
    require(not np.intersect1d(fit, valid).size, 'fit/valid leakage')
    risks = {key: _array(record[key+'_valid'], key, (len(valid),)) for key in ('M0', 'CP', 'RNA')}
    require(np.array_equal(risks['M0'], risks['CP']+risks['RNA']), 'M0 is not fixed unit CP+RNA')
    require(set(record.get('arms', {})) == set(NEURAL_ARMS), 'all three neural arms required')
    summary, fit_risks = {}, {}
    for arm in NEURAL_ARMS:
        saved = record['arms'][arm]
        risks[arm] = _array(saved['risk'], arm+' risk', (len(valid),))
        fit_risks[arm] = _array(saved['fit_risk'], arm+' fit risk', (len(fit),))
        require(np.max(np.abs(risks[arm]-risks['M0'])) <= .10000002, arm+' correction exceeds locked bound')
        _state(saved['model_state'], arm+' state'); _state(saved['initial_state'], arm+' initial state')
        require(type(saved['clinical_dim']) is int and saved['clinical_dim'] > 0, 'invalid Clinical dimension')
        audit = saved['training_audit']
        for key, wanted in (('actual_epochs', 80), ('optimizer_steps', 80), ('heldout_endpoint_reads', 0),
                            ('fixed_final_epoch', True), ('main_offset_unchanged', True)):
            require(audit.get(key) == wanted, arm+' training audit mismatch: '+key)
        require(audit.get('fit_offset_kind') == 'strict_threefold_crossfit_M0', 'fit offset is not CF M0')
        require(audit.get('heldout_offset_kind') == 'full_fit_M0', 'heldout offset is not full-fit M0')
        require(audit.get('bound') == .1 and audit.get('query_residual_preserved') is True,
                'bounded query-residual contract differs')
        counts = saved['parameter_counts']
        require(isinstance(counts, dict) and bool(counts), 'missing parameter counts')
        require(all(type(v) is int and v >= 0 for v in counts.values()), 'invalid parameter counts')
        require(set(saved['sensitivities']) == set(CONTROLS), 'incomplete frozen controls')
        for control in CONTROLS:
            risks[arm+'_'+control] = _array(saved['sensitivities'][control], arm+' '+control, (len(valid),))
        require(np.array_equal(risks[arm+'_OFF'], risks['M0']), 'OFF changed own M0')
        if arm == 'QUERY':
            for control in CONTROLS[1:]:
                require(np.array_equal(risks[arm+'_'+control], risks[arm]), 'QUERY changed without Patch message')
        summary[arm] = {**history_summary(saved['history']), 'parameter_counts': dict(counts),
                        'mechanism': mechanism_summary(saved.get('mechanism'), len(fit), len(valid))}
    states = [record['arms'][arm]['initial_state'] for arm in NEURAL_ARMS]
    require(all(set(state) == set(states[0]) for state in states), 'initial module sets differ')
    require(all(torch.equal(state[k], states[0][k]) for state in states[1:] for k in states[0]),
            'common initial states differ')
    return {'seed': seed, 'fold': fold, 'fit': fit, 'valid': valid, 'risks': risks,
            'fit_risks': fit_risks, 'prepared_sha256': record.get('data_audit', {}).get('prepared_sha256')}, summary


def verify_stage(private_root, stage, source, *, cohort_size=512):
    require(stage == 'prototype', 'outer requires a separately locked protocol; prototype only')
    root = Path(private_root); entries = []
    for seed in SEEDS:
        for fold in FOLDS:
            path = root/stage/f'seed{seed}_fold{fold}'
            done, checkpoint = path/'DONE_PRIVATE.json', path/'checkpoint.pt'
            require(done.is_file() and checkpoint.is_file(), 'all six complete cells required')
            receipt = json.loads(done.read_text(encoding='utf-8-sig'))
            for key, wanted in (('status', 'COMPLETE'), ('source', source), ('stage', stage), ('seed', seed),
                                ('fold', fold), ('model_slots', 4), ('neural_models', 3), ('epochs', 80),
                                ('optimizer_steps', 240)):
                require(receipt.get(key) == wanted, 'DONE receipt mismatch: '+key)
            require(receipt.get('checkpoint_sha256') == sha256_file(checkpoint), 'checkpoint SHA differs')
            entries.append((seed, fold, checkpoint, sha256_file(done)))
    cells, rows = {}, []
    for seed, fold, checkpoint, receipt_hash in entries:
        cell, summary = _validate_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False),
                                            source, stage, seed, fold, cohort_size)
        cells[seed, fold] = cell
        rows.append({'seed': seed, 'fold': fold, 'receipt_sha256': receipt_hash,
                     'fit_patients': len(cell['fit']), 'valid_patients': len(cell['valid']), 'neural_training': summary})
    for fold in FOLDS:
        first, second = (cells[seed, fold] for seed in SEEDS)
        for key in ('fit', 'valid'):
            require(np.array_equal(first[key], second[key]), 'partitions changed across neural seeds')
        for key in ('M0', 'CP', 'RNA'):
            require(np.array_equal(first['risks'][key], second['risks'][key]), 'shared main risk differs across seeds')
    return cells, {'status': 'PASS_SAVED_ARTIFACT_CONTRACT', 'complete_cells': 6, 'model_slots': 24,
                   'neural_models': 18, 'epochs_each': 80, 'new_convex_fits_referenced': 21,
                   'reused_full_CP_fits': 3, 'receipt_fingerprint': hashlib.sha256(
                       ''.join(row[3] for row in entries).encode()).hexdigest(), 'rows': rows}


def verify_replay(path, source):
    path = Path(path)
    require(path.is_file(), 'independent replay report missing')
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    require(report.get('status') == 'PASS' and report.get('source') == source,
            'independent replay status/source failed')
    require(report.get('stage') == 'prototype', 'independent replay stage missing/changed')
    for key, wanted in (('cells', 6), ('neural_models', 18), ('neural_risk_vectors', 144),
                        ('reference_risk_vectors', 18), ('new_predictor_fits', 0),
                        ('new_optimizer_steps', 0), ('endpoint_API_reads', 0)):
        require(type(report.get(key)) is int and report[key] == wanted, 'independent replay count differs: '+key)
    require(REPLAY_SCRIPT.is_file() and report.get('replay_script_sha256') == sha256_file(REPLAY_SCRIPT),
            'independent replay executable SHA differs')
    for key in ('neural_weights_unchanged', 'all_initialized_module_states_bitwise_shared',
                'OFF_exact_own_M0', 'QUERY_five_inactive_controls_exact_matched',
                'independent_main_replay_reused_no_raw_main_replay_here'):
        require(report.get(key) is True, 'independent replay invariant missing/failed: '+key)
    error = report.get('maximum_absolute_risk_error')
    require(type(error) in (int, float) and np.isfinite(error) and 0 <= error <= 1e-7,
            'replay error missing or outside tolerance')
    require(report.get('absolute_tolerance') == 1e-7, 'independent replay tolerance differs')
    forbidden = report.get('forbidden_API_call_counts')
    require(isinstance(forbidden, dict) and bool(forbidden) and all(
        isinstance(k, str) and type(v) is int and v == 0 for k, v in forbidden.items()),
        'independent replay forbidden API evidence missing/failed')

    evidence = report.get('main_replay_evidence')
    require(isinstance(evidence, dict), 'independent main replay binding missing')
    main_path = path.parent/'MAIN_REPLAY_AUDIT.json'
    require(main_path.is_file() and MAIN_REPLAY_SCRIPT.is_file(), 'independent main replay report/script missing')
    require(evidence.get('report_path') == 'reports/phase26_multiquery/MAIN_REPLAY_AUDIT.json'
            and evidence.get('script_path') == 'scripts/replay_phase26_main.py', 'main replay bound paths differ')
    require(evidence.get('report_sha256') == sha256_file(main_path)
            and evidence.get('script_sha256') == sha256_file(MAIN_REPLAY_SCRIPT), 'main replay bound SHA differs')
    main = json.loads(main_path.read_text(encoding='utf-8-sig'))
    for value in (main, evidence):
        require(value.get('source') == source and value.get('status') == 'PASS_NO_FIT_MAIN_REPLAY',
                'independent main replay source/status differs')
    require(main.get('replay_script_sha256') == evidence['script_sha256'], 'main replay executable changed')
    for key, wanted in (('prepared_cells', 3), ('new_fit_calls', 0), ('new_predictor_fits', 0),
                        ('new_PCA_fits', 0), ('endpoint_API_reads', 0), ('risk_vectors_replayed', 48),
                        ('legacy_preprocessing_states_reused', 12)):
        require(type(main.get(key)) is int and main[key] == wanted, 'main replay count differs: '+key)
    for key in ('maximum_absolute_risk_error', 'maximum_feature_replay_error'):
        value = main.get(key)
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1e-10,
                'main replay error missing/outside tolerance: '+key)
    require(main.get('absolute_tolerance') == 1e-10 and main.get('valid_labels_loaded_or_used') is False
            and main.get('fitter_prepare_PCA_and_endpoint_APIs_hard_blocked') is True,
            'main replay isolation/tolerance evidence differs')
    require(evidence.get('maximum_absolute_risk_error') == main['maximum_absolute_risk_error']
            and evidence.get('raw_main_replay_calls_repeated_here') == 0, 'main replay evidence scope differs')
    prepared = main.get('prepared_sha256_by_fold')
    require(isinstance(prepared, dict) and set(prepared) == {'1', '2', '3'}
            and all(isinstance(v, str) and bool(v) for v in prepared.values()), 'main replay prepared map incomplete')
    require(evidence.get('prepared_sha256_by_fold') == prepared, 'neural/main replay prepared bindings differ')
    rows = main.get('fold_rows')
    require(isinstance(rows, list) and len(rows) == 3 and all(isinstance(r, dict) for r in rows),
            'main replay fold evidence incomplete')
    by_fold = {r.get('fold'): r for r in rows}
    require(set(by_fold) == set(FOLDS) and all(
        by_fold[f].get('prepared_sha256') == prepared[str(f)] and by_fold[f].get('CP_exact_legacy_anchor') is True
        for f in FOLDS), 'main replay prepared rows/CP identities differ')
    return {'status': 'PASS', 'report_sha256': sha256_file(path), 'source': source,
            'replay_script_sha256': sha256_file(REPLAY_SCRIPT),
            'main_report_sha256': sha256_file(main_path), 'main_script_sha256': sha256_file(MAIN_REPLAY_SCRIPT),
            'prepared_sha256_by_fold': prepared, 'maximum_absolute_risk_error': error,
            'scope': 'independent_saved_checkpoint_and_main_replay;not_new_training'}


def point_results(cells, endpoint, *, cohort_size=512):
    time, event = _endpoint(endpoint, cohort_size)
    folds, seeds = [], []
    for seed in SEEDS:
        totals = {}
        for fold in FOLDS:
            cell = cells[seed, fold]; fit, valid = cell['fit'], cell['valid']
            weights = uno_weights(time[fit], event[fit], time[valid], event[valid], tau=60.)
            for name in RISK_NAMES:
                result = metrics(time[valid], event[valid], cell['risks'][name], np.full(len(valid), fold), weights)
                require(all(np.isfinite(v) for v in result.values()), 'undefined within-fold metric')
                folds.append({'seed': seed, 'fold': fold, 'model': name, 'valid_patients': len(valid),
                              'valid_events': int(event[valid].sum()), **result})
                total = totals.setdefault(name, dict.fromkeys(('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den'), 0.))
                for key in total: total[key] += result[key]
        for name, total in totals.items():
            seeds.append({'seed': seed, 'model': name, **total,
                          'Harrell': total['Harrell_num']/total['Harrell_den'], 'Uno60': total['Uno_num']/total['Uno_den']})
    return {'fold_rows': folds, 'seed_rows': seeds,
            'two_seed_means': [{'model': name, **{m: float(np.mean([r[m] for r in seeds if r['model'] == name]))
                                                 for m in METRICS}} for name in RISK_NAMES],
            'aggregation': 'sum_three_within_fold_num_den_per_seed_then_mean_two_seeds',
            'unique_patient_OOF': False, 'risk_ensemble': False, 'confirmatory_CI': False}


def prototype_gate(seed_rows, *, source, receipt_fingerprint, evaluator_sha256=None):
    by = {(row['seed'], row['model']): row for row in seed_rows}
    require(len(by) == len(seed_rows) == len(SEEDS)*len(RISK_NAMES), 'gate requires complete unique seed/model rows')
    require(all((seed, name) in by for seed in SEEDS for name in RISK_NAMES), 'gate missing seed/model')
    require(all(np.isfinite(by[seed, name][m]) for seed in SEEDS for name in RISK_NAMES for m in METRICS),
            'gate requires finite metrics')
    def compare(candidate, baseline, strict):
        output = []
        for metric in METRICS:
            delta = np.asarray([by[seed, candidate][metric]-by[seed, baseline][metric] for seed in SEEDS])
            passed = delta.mean() > 0 and np.all(delta > 0) if strict else delta.mean() >= 0
            output.append({'comparison': candidate+'-'+baseline, 'metric': metric,
                           'mean_seed_delta': float(delta.mean()), 'positive_seeds': int((delta > 0).sum()),
                           'seeds': 2, 'pass': bool(passed)})
        return output
    primary = compare('ATTN', 'QUERY', True)+compare('COMP', 'ATTN', True)
    guards = compare('COMP', 'M0', False)+compare('COMP', 'CP', False)
    paired = []
    for control in GATE_CONTROLS:
        delta = {m: float(np.mean([by[s, 'COMP'][m]-by[s, 'COMP_'+control][m] for s in SEEDS])) for m in METRICS}
        paired.append({'comparison': 'COMP-COMP_'+control, 'mean_seed_deltas': delta,
                       'pass': all(v >= 0 for v in delta.values()) and any(v > 0 for v in delta.values())})
    passed = all(row['pass'] for row in primary+guards+paired)
    return {'status': 'PROTOTYPE_GO_REQUIRES_NEW_PROTOCOL' if passed else 'PROTOTYPE_NOT_ADMITTED',
            'decision': 'GO' if passed else 'STOP', 'classification': 'GO' if passed else 'STOP',
            'source': source, 'stage': 'prototype', 'primary_tests': primary,
            'main_effect_guards': guards, 'frozen_query_guards': paired,
            'outer_admitted': False, 'outer_automatically_started': False,
            'separate_full_protocol_required_even_after_GO': True,
            'automatic_hyperparameter_or_architecture_search': False, 'alternate_outer_candidate_allowed': False,
            'confirmatory_confidence_intervals': False, 'receipt_fingerprint': receipt_fingerprint,
            'evaluator_sha256': evaluator_sha256,
            'scope': 'overlapping_inner_partition_diagnostic;not_confirmed_generalization_or_medical_causality'}


def pair_results(cells, endpoint, point, *, cohort_size=512):
    time, event = _endpoint(endpoint, cohort_size)
    by = {(r['seed'], r['model']): r for r in point['seed_rows']}
    comparisons = [('QUERY', 'M0'), ('ATTN', 'QUERY'), ('ATTN', 'M0'), ('COMP', 'ATTN'), ('COMP', 'M0'), ('COMP', 'CP')]
    comparisons += [(arm, arm+'_'+control) for arm in NEURAL_ARMS for control in CONTROLS]
    rows = []
    for candidate, baseline in comparisons:
        for seed in SEEDS:
            num = {m: np.zeros(len(STATISTICS)) for m in METRICS}; den = dict.fromkeys(METRICS, 0.)
            for fold in FOLDS:
                cell = cells[seed, fold]; fit, valid = cell['fit'], cell['valid']
                weights = uno_weights(time[fit], event[fit], time[valid], event[valid], tau=60.)
                pair = pair_transition_summary(time[valid], event[valid], cell['risks'][baseline], cell['risks'][candidate],
                                               folds=np.full(len(valid), fold), ipcw=weights)
                for row in pair['rows']:
                    m = row['metric']; num[m] += np.asarray([row[k] for k in STATISTICS])*row['denominator']
                    den[m] += row['denominator']
            for m in METRICS:
                require(den[m] > 0, 'undefined pair denominator')
                values = dict(zip(STATISTICS, (num[m]/den[m]).tolist()))
                require(abs(values['net']-(by[seed, candidate][m]-by[seed, baseline][m])) < 1e-12,
                        'pair credit differs from complete seed delta')
                require(abs(values['repair']-values['damage']-values['net']) < 1e-12, 'repair-damage identity failed')
                require(abs(sum(values[k] for k in TRANSITIONS)-1) < 1e-12, 'transition credit mass differs')
                rows.append({'comparison': candidate+'-'+baseline, 'seed': seed, 'metric': m, 'denominator': den[m], **values})
    means = []
    for candidate, baseline in comparisons:
        for m in METRICS:
            part = [r for r in rows if r['comparison'] == candidate+'-'+baseline and r['metric'] == m]
            means.append({'comparison': candidate+'-'+baseline, 'metric': m,
                          **{k: float(np.mean([r[k] for r in part])) for k in STATISTICS}})
    return {'seed_repair_damage': rows, 'repair_damage': means,
            'pair_analysis_scope': 'all_comparable_within_fold_credit;no_error_group_supervision'}


def evaluate_stage(private_root, report_dir, stage, source, endpoint_loader, *, baseline_loader,
                   partition_loader=None, replay_path=None, cohort_size=512, expected_dss_events=65):
    require(stage == 'prototype', 'outer requires a separately locked protocol; prototype only')
    private_root, report_dir = Path(private_root), Path(report_dir)
    require(not any((report_dir/n).exists() for n in ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json')),
            'existing evaluation requires explicit review; refusing overwrite')
    identity = evaluator_identity()
    cells, audit = verify_stage(private_root, stage, source, cohort_size=cohort_size)
    replay = verify_replay(replay_path or report_dir/'REPLAY_AUDIT.json', source)
    for fold in FOLDS:
        expected = baseline_loader(fold)
        cell = cells[SEEDS[0], fold]
        require(all(cells[s, fold]['prepared_sha256'] == replay['prepared_sha256_by_fold'][str(fold)] for s in SEEDS),
                'replay/checkpoint prepared SHA lineage differs')
        for key in ('fit', 'valid'):
            require(np.array_equal(cell[key], np.asarray(expected[key])), 'independent baseline partition differs')
        m0_key = 'M0_valid' if 'M0_valid' in expected else 'R0_valid'
        for key, saved_key in (('M0', m0_key), ('CP', 'CP_valid'), ('RNA', 'RNA_valid')):
            require(np.array_equal(cell['risks'][key], np.asarray(expected[saved_key])), 'independent main risk differs')
        if '_prepared_sha256' in expected:
            require(all(cells[s, fold]['prepared_sha256'] == expected['_prepared_sha256'] for s in SEEDS),
                    'checkpoint/prepared SHA lineage differs')
        if '_fit_offset' in expected:
            base_fit = _array(expected['_fit_offset'], 'fit CF offset', (len(cell['fit']),))
            require(all(np.max(np.abs(cells[s, fold]['fit_risks'][arm]-base_fit)) <= .10000002
                        for s in SEEDS for arm in NEURAL_ARMS), 'fit correction exceeds locked bound')
        if partition_loader:
            partition = partition_loader(fold)
            for key in ('fit', 'valid'):
                require(np.array_equal(cell[key], np.asarray(partition[key])), 'canonical partition differs')
    # No endpoint callback can be reached by partial, tampered, or unreplayed cells.
    endpoint = endpoint_loader()['DSS']
    _, event = _endpoint(endpoint, cohort_size)
    if expected_dss_events is not None:
        require(int(event.sum()) == expected_dss_events, 'locked DSS event count differs')
    point = point_results(cells, endpoint, cohort_size=cohort_size)
    paired = pair_results(cells, endpoint, point, cohort_size=cohort_size)
    gate = prototype_gate(point['seed_rows'], source=source, receipt_fingerprint=audit['receipt_fingerprint'],
                          evaluator_sha256=identity)
    result = {'status': 'COMPLETE_PROTOTYPE_DIAGNOSTIC', 'source': source, 'stage': stage,
              'endpoint': 'DSS', 'cohort_patients': cohort_size, 'DSS_events': int(event.sum()),
              'artifact_audit': audit, 'independent_replay': replay, **point, **paired,
              'new_training_fits': 0, 'bootstrap_draws': 0, 'patient_arrays_or_weights_public': False,
              'evaluator_sha256': identity,
              'mechanism_rows': [{'seed': row['seed'], 'fold': row['fold'], 'arm': arm,
                                  'training': {k: v for k, v in summary.items() if k != 'mechanism'},
                                  'mechanism': summary['mechanism']}
                                 for row in audit['rows'] for arm, summary in row['neural_training'].items()],
              'independent_main_reference_comparison': 'PASS_elementwise_cached_prepared_reference'}
    require(identity == evaluator_identity(), 'evaluator source drift')
    write_json(report_dir/'INNER_RESULTS.json', result); write_json(report_dir/'INNER_GATE.json', gate)
    write_json(report_dir/'decision.json', {'status': gate['classification'], 'decision': gate['decision'],
               'source': source, 'stage': stage, 'outer_admitted': False, 'outer_started': False,
               'confirmation_claimed': False, 'separate_full_protocol_required_even_after_GO': True})
    write_json(private_root/'INNER_GATE.json', {'decision': gate['decision'], 'source': source, 'stage': stage,
               'outer_admitted': False, 'outer_automatically_started': False,
               'evaluator_sha256': identity, 'receipt_fingerprint': audit['receipt_fingerprint'],
               'public_gate_sha256': sha256_file(report_dir/'INNER_GATE.json')})
    return result, gate

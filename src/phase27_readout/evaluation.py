"""Fail-closed evaluation of fixed Phase27 PCA16 offset-Cox readouts.

No encoder, projection or predictor is fitted here. All six receipts, feature
lineage, independent saved-head replay and main risks precede endpoint access.
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
ARMS = ('QUERY', 'ATTN', 'COMP')
METRICS = ('Harrell', 'Uno60')
CONTROLS = ('MATCHED', 'RNA_MEAN_QUERY', 'WRONG_RNA_QUERY', 'VALUE_OFF', 'PATCH_PERMUTE')
GATE_CONTROLS = CONTROLS[1:]
LEGACY_SOURCE = 'c4c29b105b3780a11ca8edc32c3a78bde8302ffd59f3c7eefd50a0fa4c13096c'
RISK_NAMES = ('M0', 'CP', 'RNA', *ARMS, *('OLD_'+arm for arm in ARMS),
              *(arm+'_'+control for arm in ARMS for control in GATE_CONTROLS))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = PROJECT_ROOT/'scripts/replay_phase27.py'


def evaluator_identity():
    names = ('src/phase27_readout/evaluation.py', 'scripts/evaluate_phase27.py',
             'configs/phase27_separated_readout.json', 'src/phase25_local/evaluation.py',
             'src/phase10_trimodal/reference.py', 'src/phase21_relative/evaluation.py')
    hashes = {name: sha256_file(PROJECT_ROOT/name) for name in names}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def optimizer_summary(saved):
    require(isinstance(saved, dict), 'saved optimizer diagnostic required')
    required = {'status', 'penalty', 'parameters', 'cox', 'objective', 'gradient_max', 'history',
                'intercept', 'offset_trainable'}
    require(required <= set(saved), 'optimizer convergence diagnostic incomplete')
    require(saved['status'] in ('CONVERGED', 'PASS', 'COMPLETE'), 'saved convex optimizer not converged')
    require(saved['penalty'] == 10 and saved['parameters'] == 16 and saved['intercept'] is False
            and saved['offset_trainable'] is False, 'offset Cox locked objective differs')
    for key in ('cox', 'objective', 'gradient_max'):
        require(type(saved[key]) in (int, float) and np.isfinite(saved[key]), 'nonfinite optimizer diagnostic')
    require(0 <= saved['gradient_max'] <= 1e-7, 'saved convex head did not converge')
    history = saved['history']
    require(isinstance(history, list) and bool(history), 'optimizer history missing')
    for row in history:
        require(isinstance(row, dict) and set(row) == {'phase', 'iteration', 'objective', 'cox', 'gradient_max', 'gradient_norm'},
                'unreviewed optimizer history fields')
        require(isinstance(row['phase'], str) and type(row['iteration']) is int and row['iteration'] >= 0,
                'invalid optimizer history iteration')
        require(all(type(row[k]) in (int, float) and np.isfinite(row[k]) for k in
                    ('objective', 'cox', 'gradient_max', 'gradient_norm')), 'nonfinite optimizer history')
    return {**{key: saved[key] for key in ('status', 'penalty', 'parameters', 'cox', 'objective', 'gradient_max',
                                           'intercept', 'offset_trainable')},
            'recorded_history_rows': len(history), 'maximum_recorded_iteration': max(r['iteration'] for r in history)}


def _projection(saved):
    """Inspect private projection state without publishing its arrays/keys."""
    require(isinstance(saved, dict) and bool(saved), 'saved projection state missing')
    require(saved.get('rank') == 16 and saved.get('input_dim') == 256 and saved.get('svd_solver') == 'full'
            and saved.get('whiten') is False and saved.get('input_standardized_before_PCA') is False,
            'saved PCA protocol differs')
    for key, shape in (('pca_mean', (256,)), ('components', (16, 256)),
                       ('pca_score_mean', (16,)), ('pca_score_sd', (16,))):
        array = _array(saved[key], 'private projection '+key, shape)
        if key == 'pca_score_sd': require((array > 0).all(), 'saved PCA scale is not positive')
    require(np.allclose(np.asarray(saved['components']) @ np.asarray(saved['components']).T,
                        np.eye(16), rtol=0, atol=1e-8), 'saved PCA axes are not orthonormal')
    require(type(saved.get('fit_n')) is int and saved['fit_n'] > 16
            and type(saved.get('input_fit_rank')) is int and saved['input_fit_rank'] >= 16,
            'saved PCA fit sample/rank invalid')


def _validate_head_record(record, source, seed, fold, cohort_size):
    require(isinstance(record, dict), 'head record mapping required')
    for key, wanted in (('source', source), ('stage', 'prototype'), ('seed', seed), ('fold', fold)):
        require(record.get(key) == wanted, 'head record identity differs: '+key)
    require(not set(record) & {'time', 'event', 'DSS', 'endpoint', 'endpoints'}, 'head record contains endpoints')
    fit, valid = (_array(record[key], key, integer=True) for key in ('fit', 'valid'))
    for key, ix in (('fit', fit), ('valid', valid)):
        require(ix.ndim == 1 and len(ix) > 0 and len(np.unique(ix)) == len(ix), key+' not unique/nonempty')
        require(((ix >= 0) & (ix < cohort_size)).all(), key+' outside locked cohort')
    require(not np.intersect1d(fit, valid).size, 'fit/valid leakage')
    risks = {key: _array(record[key+'_valid'], key, (len(valid),)) for key in ('M0', 'CP', 'RNA')}
    require(np.array_equal(risks['M0'], risks['CP']+risks['RNA']), 'M0 is not unchanged unit CP+RNA sum')
    require(set(record.get('heads', {})) == set(ARMS) and set(record.get('old_risks', {})) == set(ARMS),
            'all new and historical arms required')
    summaries, fit_risks = {}, {}
    for arm in ARMS:
        head = record['heads'][arm]
        coefficient = _array(head['coef'], arm+' coefficient', (16,))
        require(type(head['parameter_count']) is int and head['parameter_count'] == 16, 'readout must have 16 coefficients')
        _projection(head['projection'])
        require(head['projection']['fit_n'] == len(fit), 'PCA fit population differs')
        fit_risks[arm] = _array(head['fit_risk'], arm+' fit risk', (len(fit),))
        require(set(head['valid_risks']) == set(CONTROLS), 'all five fixed readout controls required')
        risks[arm] = _array(head['valid_risks']['MATCHED'], arm+' matched risk', (len(valid),))
        for control in GATE_CONTROLS:
            risks[arm+'_'+control] = _array(head['valid_risks'][control], arm+' '+control, (len(valid),))
            if arm == 'QUERY':
                require(np.array_equal(risks[arm+'_'+control], risks[arm]), 'QUERY inactive control changed risk')
        risks['OLD_'+arm] = _array(record['old_risks'][arm], 'old '+arm, (len(valid),))
        require(np.max(np.abs(risks['OLD_'+arm]-risks['M0'])) <= .10000002, 'historical bounded risk differs')
        # fit_diagnostics is never copied to public output. Training-label and
        # pair diagnostic publication has its own separately reviewed schema.
        _finite_tree(head.get('fit_diagnostics', {}), 'private fit diagnostic')
        optimizer = optimizer_summary(head['optimizer'])
        require(abs(optimizer['objective']-optimizer['cox']-5.*float(coefficient@coefficient)) < 1e-10,
                'saved objective/cox/coefficient penalty disagree')
        summaries[arm] = {'parameter_count': 16, 'coefficient_L2_norm': float(np.linalg.norm(coefficient)),
                          'optimizer': optimizer}
    lineage = {}
    for key in ('legacy_checkpoint_sha256', 'prepared_sha256', 'feature_receipt_sha256'):
        require(isinstance(record.get(key), str) and bool(record[key]), 'missing head lineage: '+key)
        lineage[key] = record[key]
    return {'seed': seed, 'fold': fold, 'fit': fit, 'valid': valid, 'risks': risks,
            'fit_risks': fit_risks, **lineage}, summaries


def verify_stage(private_root, stage, source, *, cohort_size=512):
    require(stage == 'prototype', 'prototype only; outer requires a new independent protocol')
    private_root = Path(private_root); entries = []
    for seed in SEEDS:
        for fold in FOLDS:
            path = private_root/'prototype'/f'seed{seed}_fold{fold}'
            receipt_path, head_path = path/'DONE_PRIVATE.json', path/'head.pt'
            require(receipt_path.is_file() and head_path.is_file(), 'all six complete head cells required')
            receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
            for key, wanted in (('status', 'COMPLETE'), ('source', source), ('seed', seed), ('fold', fold),
                                ('stage', 'prototype'), ('heads', 3), ('new_convex_fits', 3), ('new_neural_steps', 0)):
                require(receipt.get(key) == wanted, 'head DONE mismatch: '+key)
            require(receipt.get('checkpoint_sha256') == sha256_file(head_path), 'head checkpoint SHA differs')
            entries.append((seed, fold, head_path, sha256_file(receipt_path)))
    cells, rows = {}, []
    for seed, fold, path, receipt_hash in entries:
        record = torch.load(path, map_location='cpu', weights_only=False)
        cell, summaries = _validate_head_record(record, source, seed, fold, cohort_size)
        cell['head_checkpoint_sha256'] = sha256_file(path)
        require(cell['feature_receipt_sha256'] == json.loads(path.with_name('DONE_PRIVATE.json').read_text()).get(
            'feature_receipt_sha256'), 'head DONE feature receipt SHA differs')
        cells[seed, fold] = cell
        rows.append({'seed': seed, 'fold': fold, 'receipt_sha256': receipt_hash, 'heads': summaries,
                     'fit_patients': len(cell['fit']), 'valid_patients': len(cell['valid'])})
    for fold in FOLDS:
        one, two = (cells[seed, fold] for seed in SEEDS)
        for key in ('fit', 'valid'):
            require(np.array_equal(one[key], two[key]), 'patient partition differs across seeds')
        for name in ('M0', 'CP', 'RNA'):
            require(np.array_equal(one['risks'][name], two['risks'][name]), 'frozen main differs across seeds')
    return cells, {'status': 'PASS_COMPLETE_SAVED_HEAD_CONTRACT', 'complete_cells': 6, 'new_heads': 18,
                   'new_neural_steps': 0, 'model_slots': 24, 'CP_RNA_references_are_not_neural_fits': True,
                   'receipt_fingerprint': hashlib.sha256(''.join(row[3] for row in entries).encode()).hexdigest(),
                   'rows': rows}


def verify_replay(path, source):
    path = Path(path)
    require(path.is_file() and REPLAY_SCRIPT.is_file(), 'saved-head replay report/script missing')
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    for key, wanted in (('status', 'PASS'), ('source', source), ('stage', 'prototype'), ('cells', 6),
                        ('new_convex_heads', 18), ('no_new_predictor_fits', 0), ('no_forward', 0),
                        ('old_main_reused', True), ('new_risk_vectors_replayed', 108),
                        ('main_reference_vectors_checked', 18), ('original_bounded_head_references_checked', 18),
                        ('new_predictor_fits', 0), ('neural_forward_calls', 0), ('optimizer_steps', 0),
                        ('endpoint_API_reads', 0)):
        require(report.get(key) == wanted, 'saved-head replay missing/changed: '+key)
    error = report.get('maximum_absolute_risk_error')
    require(type(error) in (int, float) and np.isfinite(error) and 0 <= error <= 1e-10, 'head replay error invalid')
    require(report.get('replay_script_sha256') == sha256_file(REPLAY_SCRIPT), 'head replay executable SHA differs')
    require(report.get('absolute_tolerance') == 1e-10, 'head replay tolerance differs')
    for key in ('private_artifacts_bytes_unchanged', 'new_and_legacy_training_source_bytes_unchanged',
                'projection_and_risk_arithmetic_independent_of_new_math_helpers'):
        require(report.get(key) is True, 'head replay invariant missing/failed: '+key)
    require(report.get('heldout_labels_read') is False and report.get('gradient_recomputed_from_labels') is False,
            'head replay supervision boundary differs')
    blocked = report.get('forbidden_API_call_counts')
    require(isinstance(blocked, dict) and bool(blocked) and all(type(v) is int and v == 0 for v in blocked.values()),
            'head replay forbidden API evidence failed/missing')
    evidence = report.get('legacy_replay_evidence')
    require(isinstance(evidence, dict) and evidence.get('source') == LEGACY_SOURCE
            and evidence.get('raw_main_or_neural_replay_repeated') is False, 'legacy replay binding missing/changed')
    old_dir = PROJECT_ROOT/'reports/phase26_multiquery'
    old = {}
    for kind, name, script, status in (
        ('main', 'MAIN_REPLAY_AUDIT.json', 'replay_phase26_main.py', 'PASS_NO_FIT_MAIN_REPLAY'),
        ('neural', 'REPLAY_AUDIT.json', 'replay_phase26.py', 'PASS')):
        report_path, script_path = old_dir/name, PROJECT_ROOT/'scripts'/script
        require(report_path.is_file() and script_path.is_file(), 'bound legacy replay artifact missing')
        require(evidence.get(kind+'_report_sha256') == sha256_file(report_path)
                and evidence.get(kind+'_replay_script_sha256') == sha256_file(script_path), 'bound legacy replay SHA differs')
        saved = json.loads(report_path.read_text())
        require(saved.get('status') == status and saved.get('source') == LEGACY_SOURCE
                and saved.get('replay_script_sha256') == sha256_file(script_path), 'bound legacy replay source/status differs')
        old[kind] = saved
    require(old['main'].get('prepared_cells') == 3 and old['main'].get('risk_vectors_replayed') == 48
            and old['main'].get('legacy_preprocessing_states_reused') == 12
            and old['neural'].get('neural_risk_vectors') == 144 and old['neural'].get('cells') == 6
            and old['neural'].get('neural_models') == 18 and old['neural'].get('reference_risk_vectors') == 18,
            'bound legacy replay coverage differs')
    require(old['neural'].get('main_replay_evidence', {}).get('report_sha256') == evidence['main_report_sha256'],
            'legacy neural/main replay chain differs')
    prepared = old['main'].get('prepared_sha256_by_fold')
    require(isinstance(prepared, dict) and set(prepared) == {'1', '2', '3'}, 'legacy prepared map incomplete')
    rows = report.get('rows')
    require(isinstance(rows, list) and len(rows) == 6 and all(isinstance(row, dict) for row in rows),
            'head replay cell binding inventory missing')
    by_cell = {(row.get('seed'), row.get('fold')): row for row in rows}
    require(len(by_cell) == 6 and set(by_cell) == {(seed, fold) for seed in SEEDS for fold in FOLDS},
            'head replay cell coverage differs')
    bindings = []
    for seed in SEEDS:
        for fold in FOLDS:
            row = by_cell[seed, fold]
            require(row.get('risk_vectors_replayed') == 18 and type(row.get('maximum_absolute_risk_error')) in (int, float)
                    and np.isfinite(row['maximum_absolute_risk_error']) and 0 <= row['maximum_absolute_risk_error'] <= 1e-10,
                    'head replay cell risk coverage/error differs')
            fields = ('head_checkpoint_sha256', 'feature_receipt_sha256', 'feature_sha256',
                      'legacy_checkpoint_sha256', 'prepared_sha256')
            require(all(isinstance(row.get(key), str) and bool(row[key]) for key in fields),
                    'head replay cell hashes missing')
            require(row['prepared_sha256'] == prepared[str(fold)], 'head replay prepared map differs')
            bindings.append({'seed': seed, 'fold': fold, **{key: row[key] for key in fields}})
    return {'status': 'PASS', 'report_sha256': sha256_file(path), 'script_sha256': sha256_file(REPLAY_SCRIPT),
            'maximum_absolute_risk_error': error, 'source': source, 'cell_bindings': bindings,
            'legacy_main_report_sha256': evidence['main_report_sha256'],
            'legacy_neural_report_sha256': evidence['neural_report_sha256']}


def point_results(cells, endpoint, *, cohort_size=512):
    time, event = _endpoint(endpoint, cohort_size); fold_rows, seed_rows = [], []
    for seed in SEEDS:
        totals = {}
        for fold in FOLDS:
            cell = cells[seed, fold]; fit, valid = cell['fit'], cell['valid']
            weights = uno_weights(time[fit], event[fit], time[valid], event[valid], tau=60.)
            for name in RISK_NAMES:
                result = metrics(time[valid], event[valid], cell['risks'][name], np.full(len(valid), fold), weights)
                require(all(np.isfinite(v) for v in result.values()), 'undefined frozen-readout concordance')
                fold_rows.append({'seed': seed, 'fold': fold, 'model': name, 'valid_patients': len(valid),
                                  'valid_events': int(event[valid].sum()), **result})
                total = totals.setdefault(name, dict.fromkeys(('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den'), 0.))
                for key in total: total[key] += result[key]
        for name, total in totals.items():
            seed_rows.append({'seed': seed, 'model': name, **total,
                              'Harrell': total['Harrell_num']/total['Harrell_den'], 'Uno60': total['Uno_num']/total['Uno_den']})
    return {'fold_rows': fold_rows, 'seed_rows': seed_rows,
            'two_seed_means': [{'model': name, **{m: float(np.mean([r[m] for r in seed_rows if r['model'] == name]))
                                                 for m in METRICS}} for name in RISK_NAMES],
            'aggregation': 'sum_three_within_fold_num_den_per_seed_then_mean_two_seeds',
            'unique_patient_OOF': False, 'risk_ensemble': False, 'confirmatory_CI': False}


def prototype_gate(seed_rows, *, source, receipt_fingerprint, evaluator_sha256=None):
    by = {(r['seed'], r['model']): r for r in seed_rows}
    require(len(by) == len(seed_rows) == len(SEEDS)*len(RISK_NAMES), 'complete unique seed/model rows required')
    require(all((seed, arm) in by for seed in SEEDS for arm in RISK_NAMES), 'gate risk coverage differs')
    require(all(np.isfinite(by[s, arm][m]) for s in SEEDS for arm in RISK_NAMES for m in METRICS), 'nonfinite gate statistic')
    def compare(candidate, baseline, strict):
        out = []
        for m in METRICS:
            delta = np.array([by[s, candidate][m]-by[s, baseline][m] for s in SEEDS])
            passed = delta.mean() > 0 and np.all(delta > 0) if strict else delta.mean() >= 0
            out.append({'comparison': candidate+'-'+baseline, 'metric': m, 'mean_seed_delta': float(delta.mean()),
                        'positive_seeds': int((delta > 0).sum()), 'seeds': 2, 'pass': bool(passed)})
        return out
    primary = compare('ATTN', 'QUERY', True)+compare('COMP', 'ATTN', True)
    guards = compare('COMP', 'M0', False)+compare('COMP', 'CP', False)
    pairing = []
    for control in GATE_CONTROLS:
        delta = {m: float(np.mean([by[s, 'COMP'][m]-by[s, 'COMP_'+control][m] for s in SEEDS])) for m in METRICS}
        pairing.append({'comparison': 'COMP-COMP_'+control, 'mean_seed_deltas': delta,
                        'pass': all(v >= 0 for v in delta.values()) and any(v > 0 for v in delta.values())})
    passed = all(r['pass'] for r in primary+guards+pairing)
    return {'status': 'PROTOTYPE_GO_REQUIRES_NEW_PROTOCOL' if passed else 'PROTOTYPE_NOT_ADMITTED',
            'source': source, 'stage': 'prototype', 'classification': 'GO' if passed else 'STOP',
            'decision': 'GO' if passed else 'STOP', 'primary_tests': primary, 'main_effect_guards': guards,
            'frozen_pairing_guards': pairing, 'old_head_comparisons_are_diagnostic_only': True,
            'outer_admitted': False, 'outer_automatically_started': False, 'separate_new_protocol_required': True,
            'automatic_hyperparameter_or_architecture_search': False, 'confirmatory_confidence_intervals': False,
            'receipt_fingerprint': receipt_fingerprint, 'evaluator_sha256': evaluator_sha256,
            'scope': 'posthoc_repeated_inner_frozen_supervised_encoder_readout;not_confirmatory_or_medical_causality'}


def pair_results(cells, endpoint, point, *, cohort_size=512):
    time, event = _endpoint(endpoint, cohort_size)
    by = {(r['seed'], r['model']): r for r in point['seed_rows']}
    comparisons = [(arm, 'OLD_'+arm) for arm in ARMS]
    comparisons += [('ATTN', 'QUERY'), ('COMP', 'ATTN'), ('COMP', 'QUERY'), ('COMP', 'M0'), ('COMP', 'CP')]
    comparisons += [(arm, arm+'_'+control) for arm in ARMS for control in GATE_CONTROLS]
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
                    m = row['metric']; num[m] += np.array([row[k] for k in STATISTICS])*row['denominator']
                    den[m] += row['denominator']
            for m in METRICS:
                require(den[m] > 0, 'undefined paired denominator')
                values = dict(zip(STATISTICS, (num[m]/den[m]).tolist()))
                require(abs(values['net']-(by[seed, candidate][m]-by[seed, baseline][m])) < 1e-12,
                        'paired credit differs from complete-seed delta')
                require(abs(values['repair']-values['damage']-values['net']) < 1e-12, 'repair-damage identity failed')
                require(abs(sum(values[k] for k in TRANSITIONS)-1) < 1e-12, 'transition credit differs')
                rows.append({'comparison': candidate+'-'+baseline, 'seed': seed, 'metric': m, 'denominator': den[m], **values})
    means = []
    for candidate, baseline in comparisons:
        for m in METRICS:
            selected = [r for r in rows if r['comparison'] == candidate+'-'+baseline and r['metric'] == m]
            means.append({'comparison': candidate+'-'+baseline, 'metric': m,
                          **{k: float(np.mean([r[k] for r in selected])) for k in STATISTICS}})
    return {'seed_repair_damage': rows, 'repair_damage': means,
            'pair_analysis_scope': 'same_fold_all_comparable_credit;old_head_comparisons_diagnostic_only'}


def evaluate_stage(private_root, report_dir, stage, source, endpoint_loader, *, provenance_loader,
                   baseline_loader, partition_loader, replay_path=None, cohort_size=512, expected_dss_events=65):
    require(stage == 'prototype', 'prototype only; outer requires a new independent protocol')
    private_root, report_dir = Path(private_root), Path(report_dir)
    require(not any((report_dir/n).exists() for n in ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json')),
            'existing evaluation requires explicit review; refusing overwrite')
    identity = evaluator_identity()
    cells, audit = verify_stage(private_root, stage, source, cohort_size=cohort_size)
    replay = verify_replay(replay_path or report_dir/'REPLAY_AUDIT.json', source)
    replay_cells = {(row['seed'], row['fold']): row for row in replay['cell_bindings']}
    provenance_rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            actual, expected = cells[seed, fold], provenance_loader(seed, fold)
            require(expected.get('legacy_source') == LEGACY_SOURCE, 'frozen Phase26 source changed')
            for key in ('legacy_checkpoint_sha256', 'prepared_sha256', 'feature_receipt_sha256'):
                require(actual[key] == expected[key], 'independent saved feature lineage differs: '+key)
            for key in ('head_checkpoint_sha256', 'legacy_checkpoint_sha256', 'prepared_sha256', 'feature_receipt_sha256'):
                require(replay_cells[seed, fold][key] == actual[key], 'replay/head artifact SHA lineage differs: '+key)
            require(replay_cells[seed, fold]['feature_sha256'] == expected['feature_sha256'],
                    'replay/current saved feature SHA differs')
            for arm in ARMS:
                require(np.array_equal(actual['risks']['OLD_'+arm], np.asarray(expected['old_risks'][arm])),
                        'historical head risk changed')
            provenance_rows.append({'seed': seed, 'fold': fold, 'status': 'PASS_existing_feature_and_encoder_lineage'})
    for fold in FOLDS:
        expected = baseline_loader(fold); actual = cells[SEEDS[0], fold]
        for key in ('fit', 'valid'):
            require(np.array_equal(actual[key], np.asarray(expected[key])), 'independent prepared partition differs')
        for key in ('M0', 'CP', 'RNA'):
            require(np.array_equal(actual['risks'][key], np.asarray(expected[key+'_valid'])), 'independent main risk differs')
        require(all(cells[s, fold]['prepared_sha256'] == expected['prepared_sha256'] for s in SEEDS),
                'independent prepared SHA differs')
        canonical = partition_loader(fold)
        for key in ('fit', 'valid'):
            require(np.array_equal(actual[key], np.asarray(canonical[key])), 'canonical partition differs')
    # Only after every independent artifact gate is satisfied may DSS be read.
    endpoint = endpoint_loader()['DSS']; _, event = _endpoint(endpoint, cohort_size)
    if expected_dss_events is not None:
        require(int(event.sum()) == expected_dss_events, 'DSS event count differs from locked cohort')
    point = point_results(cells, endpoint, cohort_size=cohort_size)
    paired = pair_results(cells, endpoint, point, cohort_size=cohort_size)
    gate = prototype_gate(point['seed_rows'], source=source, receipt_fingerprint=audit['receipt_fingerprint'],
                          evaluator_sha256=identity)
    result = {'status': 'COMPLETE_FROZEN_READOUT_PROTOTYPE', 'source': source, 'stage': stage,
              'endpoint': 'DSS', 'cohort_patients': cohort_size, 'DSS_events': int(event.sum()),
              'artifact_audit': audit, 'independent_replay': replay, 'provenance_audit': provenance_rows,
              **point, **paired, 'new_predictor_fits_during_evaluation': 0, 'bootstrap_draws': 0,
              'evaluator_sha256': identity, 'patient_arrays_weights_coefficients_or_projections_public': False,
              'fit_encoder_representations_are_not_crossfitted': True,
              'historical_head_comparison_is_not_a_tanh_only_ablation': True}
    require(identity == evaluator_identity(), 'evaluator source drift')
    write_json(report_dir/'INNER_RESULTS.json', result); write_json(report_dir/'INNER_GATE.json', gate)
    write_json(report_dir/'decision.json', {'status': gate['classification'], 'decision': gate['decision'],
               'source': source, 'stage': stage, 'outer_admitted': False, 'outer_started': False,
               'confirmation_claimed': False, 'separate_new_protocol_required': True})
    write_json(private_root/'INNER_GATE.json', {'decision': gate['decision'], 'source': source, 'stage': stage,
               'outer_admitted': False, 'outer_automatically_started': False,
               'evaluator_sha256': identity, 'receipt_fingerprint': audit['receipt_fingerprint'],
               'public_gate_sha256': sha256_file(report_dir/'INNER_GATE.json')})
    return result, gate

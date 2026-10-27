"""Replay completed, locked outer checkpoints without fitting any model.

Private arrays stay in the audit process. Only anonymous fold/seed labels,
coverage counts, hashes and numerical errors are written to the public JSON.
The Phase10 reference supplies independent survival-pair/IPCW counting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.bgb_en_checkpoint_audit import (
    AuditError, require, sha256, index_hash, valid_indices,
    verify_preprocessing_state, verify_current_source,
    metrics_from_risk, verify_metric_fields, verify_replayed_risk,
)
from src.bgb_en.replay import apply_saved_preprocessing, replay_full_risks
from src.phase10_trimodal.reference import components


def independent_beta_check(cp, rna, time, event, groups, beta, saved, lam=1., tolerance=1e-6):
    """Differentiate the same-teacher pair objective directly, without a solver."""
    cp, rna, time = map(lambda x: np.asarray(x, float), (cp, rna, time))
    event, groups = np.asarray(event), np.asarray(groups)
    require(cp.shape == rna.shape == time.shape == event.shape == groups.shape,
            'BETA_INPUT_SHAPE_MISMATCH')
    require(np.isfinite(cp).all() and np.isfinite(rna).all() and np.isfinite(beta), 'BETA_NONFINITE_INPUT')
    require((groups >= 0).all() and np.isin(event, [0, 1]).all(), 'BETA_UNCROSSFITTED_ROW')
    # Independently enumerate comparable pairs; do not call fit_scalar_same_teacher.
    margins, differences = [], []
    for left in range(len(time)):
        if event[left] != 1:
            continue
        for right in range(len(time)):
            if groups[left] != groups[right]:
                continue
            if time[right] > time[left] or (time[right] == time[left] and event[right] == 0):
                margins.append(cp[left] - cp[right])
                differences.append(rna[left] - rna[right])
    require(bool(margins), 'BETA_NO_SAME_TEACHER_PAIRS')
    margins, differences = np.asarray(margins), np.asarray(differences)
    score = margins + float(beta) * differences
    loss = float(np.logaddexp(0., -score).mean() + lam * float(beta)**2 / 2.)
    gradient = float(np.mean(-np.exp(-np.logaddexp(0., score)) * differences) + lam * float(beta))
    require(saved.get('pairs') == len(score), 'BETA_PAIR_COUNT_MISMATCH')
    require(saved.get('lambda_beta') == lam, 'BETA_REGULARIZATION_MISMATCH')
    require(abs(loss - saved['loss']) <= 1e-10, 'BETA_SAVED_OBJECTIVE_MISMATCH')
    require(abs(gradient) <= tolerance, 'BETA_INDEPENDENT_GRADIENT_NOT_STATIONARY')
    return dict(same_teacher_pairs=len(score), independent_gradient_abs=abs(gradient),
                saved_loss_error=abs(loss - saved['loss']), both_pair_members_excluded=True,
                optimizer_steps=0)


def _replay_models(features, models, fit, saved_risks=None, tolerance=2e-6):
    replay = {}; maximum = 0.; scale_error = 0.
    for name, model in models.items():
        x, coef = np.asarray(features[name]), np.asarray(model['coef'])
        require(x.ndim == 2 and coef.shape == (x.shape[1],) and np.isfinite(coef).all(),
                'COX_COEFFICIENT_SHAPE_OR_FINITE_MISMATCH')
        pred = x @ coef; mean, sd = model['scale']['mean'], model['scale']['sd']
        require(np.isfinite(mean) and np.isfinite(sd) and sd > 1e-8, 'COX_SAVED_SCALE_INVALID')
        this_scale = max(abs(float(pred[fit].mean()) - mean), abs(float(pred[fit].std()) - sd))
        require(this_scale <= tolerance, 'COX_SAVED_TRAIN_MOMENTS_MISMATCH')
        scale_error = max(scale_error, this_scale)
        replay[name] = (pred - mean) / sd
        if saved_risks is not None:
            maximum = max(maximum, verify_replayed_risk(saved_risks[name], replay[name], len(pred), tolerance))
    return replay, dict(coefficient_models_replayed=len(models), maximum_risk_error=maximum,
                       maximum_saved_train_moment_error=scale_error)


def replay_teachers(raw, common, seed, tolerance=2e-6):
    """Replay actual held-out teacher predictions, including every saved PCA."""
    n = len(raw['t']); train, test = np.asarray(common['train']), np.asarray(common['test'])
    calibrated = common['teachers'][seed]
    groups = np.asarray(calibrated['teacher_groups'])
    require(groups.shape == (n,) and (groups[test] == -1).all(), 'OUTER_TEST_IN_TEACHER_GROUPS')
    require((groups[train] >= 0).all(), 'TRAIN_ROW_NOT_CROSSFITTED')
    teachers = calibrated['teachers']; require(len(teachers) == 3, 'TEACHER_COUNT_MISMATCH')
    expected_models = {'LOCAL', 'CS', 'R'} if seed == 31 else {'LOCAL', 'R'}
    cf = {name: np.zeros(n) for name in expected_models}; held_blocks = []
    maximum = 0.; scale_error = 0.
    for group, teacher in enumerate(teachers):
        fit, held = valid_indices(teacher['train'], n), valid_indices(teacher['held'], n)
        require(not np.intersect1d(fit, held).size, 'TEACHER_TRAIN_HELD_OVERLAP')
        require(not np.intersect1d(fit, test).size and not np.intersect1d(held, test).size,
                'TEACHER_OUTER_TEST_LEAKAGE')
        require(set(fit) | set(held) == set(train), 'TEACHER_PARENT_TRAIN_MISMATCH')
        require((groups[held] == group).all(), 'TEACHER_ASSIGNMENT_MISMATCH')
        verify_preprocessing_state(teacher['preprocessing'], fit)
        require(set(teacher['models']) == expected_models, 'TEACHER_MODEL_SET_MISMATCH')
        features = apply_saved_preprocessing(raw, teacher['preprocessing'])
        risks, audit = _replay_models(features, teacher['models'], fit, tolerance=tolerance)
        scale_error = max(scale_error, audit['maximum_saved_train_moment_error'])
        for name, values in risks.items():
            cf[name][held] = values[held]
        held_blocks.extend(held.tolist())
        del features, risks
    require(np.array_equal(np.sort(held_blocks), np.sort(train)), 'TEACHER_HELD_COVERAGE_MISMATCH')
    for name, saved_name in [('LOCAL', 'CP_cf'), ('R', 'RNA_cf')] + ([('CS', 'CS_cf')] if seed == 31 else []):
        saved = np.asarray(calibrated[saved_name])
        require(saved.shape == (n,) and np.array_equal(saved[test], np.zeros(len(test))), 'CROSSFIT_VALUES_OUTSIDE_TRAIN')
        maximum = max(maximum, verify_replayed_risk(saved[train], cf[name][train], len(train), tolerance))
    for name, full_key in [('CP', 'LOCAL'), ('RNA', 'R')]:
        expected = np.asarray(common['full_risks'][full_key]).copy(); expected[train] = cf[full_key][train]
        maximum = max(maximum, verify_replayed_risk(calibrated[name], expected, n, tolerance))
    expected_base = np.asarray(calibrated['CP']) + calibrated['beta'] * np.asarray(calibrated['RNA'])
    maximum = max(maximum, verify_replayed_risk(calibrated['base'], expected_base, n, tolerance))
    if seed == 31:
        cs = np.asarray(common['full_risks']['CS']).copy(); cs[train] = cf['CS'][train]
        maximum = max(maximum, verify_replayed_risk(common['seed31_CS_risk'], cs, n, tolerance))
        maximum = max(maximum, verify_replayed_risk(calibrated['CS'], cs, n, tolerance))
    beta = independent_beta_check(cf['LOCAL'][train], cf['R'][train], raw['t'][train], raw['e'][train],
                                  groups[train], calibrated['beta'], calibrated['scalar_audit'])
    return dict(teachers_replayed=3, teacher_Cox_models_replayed=3 * len(expected_models),
                CS_teacher_scope='seed31_saved_CS_teachers' if seed == 31 else 'NOT_FITTED_FOR_THIS_SEED_SHARED_SEED31_CS',
                maximum_crossfit_risk_error=maximum, maximum_teacher_train_moment_error=scale_error,
                train_held_test_pairwise_disjoint=True, all_preprocessing_fit_hashes_verified=True,
                actual_saved_teacher_coefficients_and_PCA_replayed=True, beta=beta)


def _paired_ranking_error(a, b):
    return max(abs(a[name] - b[name]) for name in ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den'))


def audit_baseline(raw, common, receipt, tolerance=2e-6):
    train, test = np.asarray(common['train']), np.asarray(common['test'])
    verify_preprocessing_state(common['full_preprocessing'], train)
    features = apply_saved_preprocessing(raw, common['full_preprocessing'])
    require(set(common['full_models']) == {'LOCAL', 'GLOBAL', 'CS', 'R'}, 'FULL_MODEL_SET_MISMATCH')
    full, full_audit = _replay_models(features, common['full_models'], train, common['full_risks'], tolerance)
    teachers = {}; metric_error = 0.; ranking_error = 0.; fold_metrics = []
    require(set(common['teachers']) == {31, 47, 83}, 'OUTER_TEACHER_SEED_SET_MISMATCH')
    require(common['results'] == receipt['results'], 'BASELINE_ARTIFACT_RECEIPT_RESULTS_MISMATCH')
    for seed in (31, 47, 83):
        teachers[str(seed)] = replay_teachers(raw, common, seed, tolerance)
        calibrated = common['teachers'][seed]
        cp = np.asarray(common['full_risks']['LOCAL'])[test]
        scalar = cp + calibrated['beta'] * np.asarray(common['full_risks']['R'])[test]
        for name, pred, rebuilt in [('CP_STRONG', calibrated['test_CP_risk'], full['LOCAL'][test]),
                                   ('CP_SCALAR', calibrated['test_CP_SCALAR_risk'], full['LOCAL'][test] + calibrated['beta'] * full['R'][test])]:
            verify_replayed_risk(pred, cp if name == 'CP_STRONG' else scalar, len(test), tolerance)
            metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], pred, cp)
            expected = receipt['results'][str(seed)]['models'][name]
            metric_error = max(metric_error, verify_metric_fields(metric, expected))
            rebuilt_metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], rebuilt, full['LOCAL'][test])
            ranking_error = max(ranking_error, _paired_ranking_error(metric, rebuilt_metric))
            ptrain, ptest = raw['pfi_valid'][train], raw['pfi_valid'][test]
            pfi = metrics_from_risk(raw['pfi_t'][train][ptrain], raw['pfi_e'][train][ptrain],
                                    raw['pfi_t'][test][ptest], raw['pfi_e'][test][ptest], np.asarray(pred)[ptest])
            metric_error = max(metric_error, verify_metric_fields(pfi, expected['PFI_frozen_DSS_risk']))
            fold_metrics.extend([dict(model=name, seed=seed, endpoint='DSS', **metric),
                                 dict(model=name, seed=seed, endpoint='PFI', **pfi)])
    return features, dict(full_anchor_replay=full_audit, seed_teacher_replays=teachers,
                          independent_metric_max_error=metric_error, replay_ranking_numerator_max_error=ranking_error,
                          replay_ranking_status='EXACT_CONCORDANCE' if ranking_error <= 1e-10 else 'ROUNDING_SENSITIVE_WITHIN_RISK_TOLERANCE',
                          independent_fold_metrics=fold_metrics,
                          fold_local_only=True, current_outer_train_IPCW_only=True)


def rebuild_neural(artifact, receipt, common, features, raw, membership, bank, device):
    """Load the actual saved weights and execute the final model, without a solver."""
    from src.bgb_en.training import MechanismModel, predict
    from src.bgb_en.protocol import patient_mismatch
    train, test = np.asarray(common['train']), np.asarray(common['test'])
    require(np.array_equal(artifact['train'], train) and np.array_equal(artifact['test'], test), 'NEURAL_SPLIT_MISMATCH')
    require(artifact.get('contract') == receipt['contract'], 'NEURAL_ARTIFACT_CONTRACT_MISMATCH')
    require(artifact.get('epochs') == artifact.get('optimizer_updates') == receipt.get('epochs') == 80,
            'NEURAL_FIXED_EPOCH_BUDGET_MISMATCH')
    require(artifact.get('outer_evaluations_during_training') == 0, 'NEURAL_OUTER_READ_DURING_TRAINING')
    require(all(row.get('outer_test_read') is False for row in artifact['curves']), 'NEURAL_TRAIN_CURVE_OUTER_READ')
    calibrated = common['teachers'][receipt['seed']]
    prepared = dict(features=features, train=train, valid=test, CP=calibrated['CP'], RNA=calibrated['RNA'],
                    base=calibrated['base'], CS=common['seed31_CS_risk'], beta=calibrated['beta'], full_risks=common['full_risks'])
    torch.manual_seed(receipt['seed']); np.random.seed(receipt['seed'])
    model = MechanismModel('R3', receipt['selected_arm'], prepared, membership, bank, device, gamma=.1)
    strata = np.array([str(raw['cancer'][i]) + '|' + str(raw['clinical_raw']['AJCC_PATHOLOGIC_TUMOR_STAGE'][i])
                       + '|' + str(raw['platform'][i]) for i in range(len(raw['t']))])
    for name, part in [('train', train), ('validation', test)]:
        mapping, _ = patient_mismatch(part, strata, 270077 + receipt['fold'] * 100 + (name == 'validation'))
        model.query_map[part] = mapping[part]
    model.load_state_dict(artifact['final_state_dict'], strict=True)
    risk, delta = predict(model, test, 8)
    del model
    return risk, delta


def audit_neural(raw, artifact, receipt, common, features, membership, bank, device, tolerance=2e-6):
    train, test = np.asarray(common['train']), np.asarray(common['test'])
    rebuilt, delta = rebuild_neural(artifact, receipt, common, features, raw, membership, bank, device)
    maximum = verify_replayed_risk(artifact['risk'], rebuilt, len(test), tolerance)
    delta_error = verify_replayed_risk(artifact['delta'], delta, len(test), tolerance)
    calibrated = common['teachers'][receipt['seed']]
    refs = dict(CP_STRONG=np.asarray(common['full_risks']['LOCAL'])[test], CP_SCALAR=np.asarray(calibrated['base'])[test])
    metric_error = 0.; ranking_error = 0.; reference_metrics = []
    for name, ref in refs.items():
        metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], artifact['risk'], ref)
        metric_error = max(metric_error, verify_metric_fields(metric, artifact['metrics'][name]), verify_metric_fields(metric, receipt['metrics'][name]))
        actual = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], rebuilt, ref)
        ranking_error = max(ranking_error, _paired_ranking_error(metric, actual))
        reference_metrics.append(dict(reference=name, **metric))
    ptrain, ptest = raw['pfi_valid'][train], raw['pfi_valid'][test]
    pfi = metrics_from_risk(raw['pfi_t'][train][ptrain], raw['pfi_e'][train][ptrain], raw['pfi_t'][test][ptest], raw['pfi_e'][test][ptest], np.asarray(artifact['risk'])[ptest])
    metric_error = max(metric_error, verify_metric_fields(pfi, artifact['PFI_frozen_DSS_risk']), verify_metric_fields(pfi, receipt['PFI_frozen_DSS_risk']))
    controls_error = 0.
    require(set(artifact['frozen_controls']) == {'wrong_query', 'mean_query', 'whole_omics', 'attention_permutation', 'interaction_off', 'direct_off'}, 'NEURAL_CONTROL_SET_MISMATCH')
    for name, control in artifact['frozen_controls'].items():
        metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], artifact['risk'], control['risk'])
        controls_error = max(controls_error, verify_metric_fields(metric, control['metrics']), verify_metric_fields(metric, receipt['frozen_controls'][name]))
    return dict(actual_state_dict_forward=True, maximum_risk_error=maximum, maximum_delta_error=delta_error,
                independent_metric_max_error=metric_error, frozen_controls_checked=6,
                independent_fold_metrics=[dict(model='NESTED_INTERACTION', seed=receipt['seed'], endpoint='DSS', **{k:v for k,v in reference_metrics[0].items() if k!='reference'}),
                                          dict(model='NESTED_INTERACTION', seed=receipt['seed'], endpoint='PFI', **pfi)],
                independent_DSS_comparison_metrics=reference_metrics,
                frozen_control_scope='independent_metrics_from_actual_saved_control_vectors_not_new_control_forwards',
                independent_control_metric_max_error=controls_error, replay_ranking_numerator_max_error=ranking_error,
                replay_ranking_status='EXACT_CONCORDANCE' if ranking_error <= 1e-10 else 'ROUNDING_SENSITIVE_WITHIN_RISK_TOLERANCE')


def coverage_report(units, folds, pfi_valid, finalists, n=512, pfi_expected=510):
    """Count OOF membership only; never compare risk values across folds."""
    folds, pfi_valid = np.asarray(folds), np.asarray(pfi_valid, bool)
    require(folds.shape == pfi_valid.shape == (n,), 'OOF_COHORT_SHAPE_MISMATCH')
    require(int(pfi_valid.sum()) == pfi_expected, 'PFI_ELIGIBILITY_COUNT_MISMATCH')
    output = []; all_complete = True
    for model in finalists:
        for seed in (31, 47, 83):
            selected = [u for u in units if u['status'] == 'PASS' and
                        (u['kind'] == 'BASELINE' if model in ('CP_STRONG', 'CP_SCALAR') else u['kind'] == 'NEURAL' and u['seed'] == seed)]
            counts = np.zeros(n, int); completed = []
            for unit in selected:
                fold = unit['fold']; require(fold in range(1, 6), 'OOF_FOLD_OUTSIDE_CONTRACT')
                require(fold not in completed, 'OOF_DUPLICATE_FOLD')
                completed.append(fold); counts[folds == fold] += 1
            require((counts <= 1).all(), 'OOF_PATIENT_REPEATED')
            full = len(completed) == 5 and np.array_equal(counts, np.ones(n, int))
            all_complete &= full
            output.append(dict(model=model, seed=seed, completed_folds=sorted(completed),
                               DSS_patients_covered=int((counts == 1).sum()), PFI_patients_covered=int(((counts == 1) & pfi_valid).sum()),
                               expected_DSS_patients=n, expected_PFI_patients=pfi_expected,
                               every_patient_exactly_once=bool(full), coverage_status='COMPLETE' if full else 'PARTIAL',
                               CP_deterministic_reference_reused=(model == 'CP_STRONG' and seed != 31)))
    return output, bool(all_complete)


def summarize_fold_metrics(units, finalists):
    """Independent fold-local numerator aggregation, with no pooled risk vector."""
    rows = []
    for unit in units:
        if unit['status'] == 'PASS':
            rows.extend(dict(fold=unit['fold'], **metric) for metric in unit.get('independent_fold_metrics', []))
    summaries = []
    for model in finalists:
        for endpoint in ('DSS', 'PFI'):
            seeds = []
            for seed in (31, 47, 83):
                selected = [row for row in rows if (row['model'], row['endpoint'], row['seed']) == (model, endpoint, seed)]
                require(len({r['fold'] for r in selected}) == len(selected), 'DUPLICATE_INDEPENDENT_METRIC_FOLD')
                if {r['fold'] for r in selected} != {1, 2, 3, 4, 5}:
                    continue
                counts = {name: float(sum(r[name] for r in selected)) for name in ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den')}
                counts.update(seed=seed, Harrell=counts['Harrell_num'] / counts['Harrell_den'] if counts['Harrell_den'] else None,
                              Uno60=counts['Uno_num'] / counts['Uno_den'] if counts['Uno_den'] else None)
                seeds.append(counts)
            complete = len(seeds) == 3
            row = dict(model=model, endpoint=endpoint, complete_all_five_folds_three_seeds=complete, seed_counts=seeds)
            if complete:
                for name in ('Harrell', 'Uno60'):
                    row[name] = float(np.mean([s[name] for s in seeds])) if all(s[name] is not None for s in seeds) else None
            summaries.append(row)
    return summaries


def _sources():
    paths = [Path(__file__).resolve(), ROOT / 'scripts/bgb_en_checkpoint_audit.py', ROOT / 'scripts/bgb_en_nested.py',
             ROOT / 'scripts/bgb_en_final_neural.py', ROOT / 'scripts/bgb_en_outer.py',
             ROOT / 'src/bgb_en/replay.py', ROOT / 'src/phase10_trimodal/reference.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha256(path) for path in paths}


def _validate_baseline_contract(receipt, common, args, raw, lineage, train, test, lock_sha, inner_sha):
    require(receipt.get('status') == 'COMPLETE_OUTER_BASELINES' and receipt.get('selection_used_outer_test') is False,
            'BASELINE_RECEIPT_NOT_OUTER_BLIND')
    contract = receipt['contract']
    wanted = dict(selection_lock_sha256=lock_sha, inner_report_sha256=inner_sha,
                  config_sha256=sha256(args.config), assets_configuration_sha256=sha256(args.assets),
                  source_lineage=lineage, source_lineage_sha256=hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest(),
                  train_sha256=index_hash(train), test_sha256=index_hash(test), source_sha256=sha256(ROOT / 'scripts/bgb_en_outer.py'),
                  cohort_sha256=raw['audit']['cohort_sha256'], endpoints_sha256=raw['audit']['endpoints_sha256'],
                  visual_sources_sha256=raw['audit'].get('visual_sources_sha256'), full_CONCH_mean_sha256=args._audit_full_mean_sha256)
    require(all(contract.get(k) == value for k, value in wanted.items()), 'BASELINE_CURRENT_CONTRACT_MISMATCH')
    for name, digest in contract['module_sha256'].items():
        require(sha256(ROOT / 'src/bgb_en' / name) == digest, 'BASELINE_MODULE_SOURCE_CHANGED')
    require(common['contract'] == contract, 'BASELINE_ARTIFACT_CONTRACT_MISMATCH')
    require(np.array_equal(common['train'], train) and np.array_equal(common['test'], test), 'BASELINE_CURRENT_OUTER_SPLIT_MISMATCH')
    require(set(receipt['results']) == {'31', '47', '83'}, 'BASELINE_RECEIPT_SEEDS_MISMATCH')


def run(args):
    from scripts.bgb_en_run import load_inputs
    from scripts.bgb_en_outer import validate_selection_lock, source_lineage_hash
    from scripts.bgb_en_nested import select_receipts
    from src.bgb_en.data import outer_partition, read_patch_cache
    from src.bgb_en.training import PatchBank

    torch.set_num_threads(2)
    assets, config, raw, membership, lineage = load_inputs(args)
    verify_current_source(lineage, config_path=args.config)
    lock = json.loads(args.selection_lock.read_text()); inner = json.loads(args.inner_report.read_text())
    lock_sha, inner_sha = sha256(args.selection_lock), sha256(args.inner_report)
    validate_selection_lock(lock, inner, lineage, config, sha256(args.config), inner_sha, 1)
    require(len(raw['t']) == 512 and int(np.asarray(raw['pfi_valid']).sum()) == 510, 'OUTER_COHORT_SIZE_MISMATCH')
    native_cache = Path(assets['patch_grid'])
    native_ready = json.loads((native_cache / 'READY.json').read_text(encoding='utf-8-sig'))
    args._audit_full_mean_sha256 = sha256(native_cache / 'full_mean.npy')
    require(args._audit_full_mean_sha256 == native_ready['array_sha256']['full_mean'], 'CURRENT_FULL_CONCH_MEAN_RECEIPT_MISMATCH')
    assets_sha = sha256(args.assets)
    sources = _sources(); canonical_lineage = source_lineage_hash(lineage)
    previous = {}
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text())
        if (old.get('audit_sources') == sources and old.get('source_lineage_sha256') == canonical_lineage
                and old.get('selection_lock_sha256') == lock_sha and old.get('inner_report_sha256') == inner_sha
                and old.get('assets_configuration_sha256') == assets_sha and old.get('risk_tolerance') == args.tolerance):
            previous = {(row['kind'], row['fold'], row.get('seed')): row for row in old.get('units', []) if row.get('status') == 'PASS'}
    available = []; seen = set()
    for kind, pattern in [('BASELINE', 'outer/fold*/COMPLETE.json'), ('NEURAL', 'neural/*/COMPLETE.json')]:
        for path in sorted(args.private.glob(pattern)):
            receipt = json.loads(path.read_text()); fold = receipt.get('fold'); seed = receipt.get('seed') if kind == 'NEURAL' else None
            require(fold in range(1, 6) and (kind == 'BASELINE' or seed in (31, 47, 83)), 'OUTER_UNIT_IDENTITY_INVALID')
            identity = kind, fold, seed; require(identity not in seen, 'DUPLICATE_OUTER_UNIT'); seen.add(identity)
            require(kind != 'NEURAL' or 'NESTED_INTERACTION' in lock['finalists'], 'UNAUTHORIZED_NEURAL_OUTER_UNIT')
            available.append((fold, kind != 'BASELINE', seed or 0, path, receipt))
    units = []; new = reused = 0; bank = None; common_fold = features = common = None
    for fold, is_neural, _, path, receipt in sorted(available):
        kind = 'NEURAL' if is_neural else 'BASELINE'; seed = receipt.get('seed') if is_neural else None
        artifact_path = path.parent / ('model.pt' if is_neural else 'models.pt')
        digest = sha256(artifact_path); require(digest == receipt['artifact_sha256'], 'OUTER_ARTIFACT_SHA256_MISMATCH')
        identity = kind, fold, seed
        source = dict(artifact_sha256=digest, receipt_sha256=sha256(path))
        row = dict(kind=kind, fold=fold, status='FAILED', optimizer_steps=0, **source)
        if is_neural: row['seed'] = seed
        prior = previous.get(identity)
        if prior and all(prior.get(k) == v for k, v in source.items()):
            units.append(dict(prior, replayed_this_invocation=False, previous_actual_replay_reused=True)); reused += 1; continue
        if args.limit and new >= args.limit: continue
        row['replayed_this_invocation'] = True
        try:
            train, test = outer_partition(raw['folds'], fold)
            baseline_dir = args.private / 'outer' / f'fold{fold}'
            baseline_receipt = json.loads((baseline_dir / 'COMPLETE.json').read_text())
            baseline_path = baseline_dir / 'models.pt'
            require(sha256(baseline_path) == baseline_receipt['artifact_sha256'], 'SHARED_BASELINE_ARTIFACT_CHANGED')
            if common_fold != fold:
                common = torch.load(baseline_path, map_location='cpu', weights_only=False)
                features = None; common_fold = fold
            _validate_baseline_contract(baseline_receipt, common, args, raw, lineage, train, test, lock_sha, inner_sha)
            if not is_neural:
                features, audit = audit_baseline(raw, common, receipt, args.tolerance)
                row.update(status='PASS', **audit)
            else:
                require(any(u['kind'] == 'BASELINE' and u['fold'] == fold and u['status'] == 'PASS' for u in units),
                        'NEURAL_UPSTREAM_BASELINE_NOT_AUDITED')
                require(receipt.get('status') == 'COMPLETE_OUTER_NEURAL' and receipt.get('outer_test_read_for_selection') is False,
                        'NEURAL_RECEIPT_OUTER_SELECTION_LEAKAGE')
                selected_path = args.private / 'selection' / f'outer{fold}.json'
                selected = json.loads(selected_path.read_text())
                wanted = dict(status='LOCKED_WITHIN_OUTER_TRAIN_BEFORE_TEST', outer_fold=fold, outer_test_evaluated=False,
                              selection_lock_sha256=lock_sha, inner_report_sha256=inner_sha,
                              source_lineage_sha256=canonical_lineage, source_sha256=sha256(ROOT / 'scripts/bgb_en_nested.py'),
                              selected_epochs=80, selected_gamma=.1)
                require(all(selected.get(k) == v for k, v in wanted.items()), 'NEURAL_PER_OUTER_SELECTION_LOCK_CHANGED')
                require(receipt['selected_arm'] == selected['selected_arm'], 'NEURAL_SELECTED_ARM_MISMATCH')
                inner_fits = args.primary_private if fold == 1 else args.private / 'nested_inner'
                chosen, candidates, receipts = select_receipts(inner_fits, fold, config, lineage)
                require(chosen['arm'] == selected['selected_arm'] and candidates == selected['candidates'] and receipts == selected['receipts'],
                        'NEURAL_INNER_SELECTION_ARTIFACTS_CHANGED')
                wanted_contract = dict(selection_lock_sha256=lock_sha, fold_lock_sha256=sha256(selected_path),
                    baseline_artifact_sha256=sha256(baseline_path), core_lineage=lineage,
                    train_hash=index_hash(train), test_hash=index_hash(test), seed=seed, arm=selected['selected_arm'])
                require(all(receipt['contract'].get(k) == v for k, v in wanted_contract.items()), 'NEURAL_CURRENT_CONTRACT_MISMATCH')
                for name, digest in receipt['contract']['additional_sources'].items():
                    require(sha256(ROOT / name) == digest, 'NEURAL_ADDITIONAL_SOURCE_CHANGED')
                if features is None: features = apply_saved_preprocessing(raw, common['full_preprocessing'])
                if bank is None: bank = PatchBank(read_patch_cache(assets['patch_regions']), args.device)
                artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
                row.update(status='PASS', **audit_neural(raw, artifact, receipt, common, features, membership, bank, args.device, args.tolerance))
                del artifact
        except AuditError as exc:
            row.update(failure_code=str(exc), failure_type=type(exc).__name__)
        except Exception as exc:
            row.update(failure_code='UNEXPECTED_RUNTIME_FAILURE', failure_type=type(exc).__name__)
            directory = args.private / 'outer_audit_private'; directory.mkdir(parents=True, exist_ok=True)
            (directory / f'{kind}_fold{fold}_seed{seed}.log').write_text(traceback.format_exc(), encoding='utf-8')
            row['private_runtime_log_created'] = True
        units.append(row); new += 1
        print(json.dumps({k: row.get(k) for k in ('kind', 'fold', 'seed', 'status', 'maximum_risk_error', 'failure_code')}), flush=True)
    coverage, complete = coverage_report(units, raw['folds'], raw['pfi_valid'], lock['finalists'])
    failed = sum(row['status'] != 'PASS' for row in units); passed = len(units) - failed
    result = dict(schema='bgb_en_outer_checkpoint_audit_v1',
        status='FAILED_AVAILABLE_OUTER_CHECKPOINTS' if failed else 'PASS_COMPLETE_LOCKED_FINALISTS' if complete else 'PASS_AVAILABLE_SUBSET_NOT_WHOLE_FINALISTS',
        date_utc=datetime.now(timezone.utc).isoformat(), audit_sources=sources, source_lineage_sha256=canonical_lineage,
        selection_lock_sha256=lock_sha, inner_report_sha256=inner_sha,
        assets_configuration_sha256=assets_sha,
        available_complete_receipts=len(available), new_actual_replays=new, previous_actual_replays_reused=reused,
        passed_units=passed, failed_units=failed, planned_baseline_fold_units=5,
        planned_neural_fold_seed_units=15 if 'NESTED_INTERACTION' in lock['finalists'] else 0,
        actual_neural_state_dict_replay_units=sum(u['status'] == 'PASS' and u['kind'] == 'NEURAL' for u in units),
        unique_full_Cox_coefficients_replayed=sum(u.get('full_anchor_replay', {}).get('coefficient_models_replayed', 0) for u in units if u['status'] == 'PASS'),
        unique_teacher_Cox_coefficients_replayed=sum(sum(t['teacher_Cox_models_replayed'] for t in u.get('seed_teacher_replays', {}).values()) for u in units if u['status'] == 'PASS'),
        unique_same_teacher_beta_gradient_checks=sum(len(u.get('seed_teacher_replays', {})) for u in units if u['status'] == 'PASS'),
        whole_finalist_matrix_complete_and_replayed=complete and not failed, optimizer_steps=0,
        new_preprocessing_fits=0, risk_tolerance=args.tolerance, requested_limit_new_replays=args.limit,
        independent_metrics='Phase10_reference_with_current_outer_training_IPCW_only',
        cross_fold_risk_comparisons=False, private_arrays_or_identifiers_exported=False, OOF_coverage=coverage,
        independent_within_fold_seed_aggregates=summarize_fold_metrics(units, lock['finalists']), units=units)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8'); temporary.replace(args.output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True); parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--primary-private', type=Path, required=True)
    parser.add_argument('--selection-lock', type=Path, required=True); parser.add_argument('--inner-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en/OUTER_CHECKPOINT_AUDIT.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--device', default='cuda'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=0); parser.add_argument('--tolerance', type=float, default=2e-6)
    args = parser.parse_args()
    if args.limit < 0 or not 0 < args.tolerance <= 2e-6: raise ValueError('Invalid audit limit/tolerance')
    try:
        result = run(args)
    except AuditError as exc:
        result = dict(schema='bgb_en_outer_checkpoint_audit_v1', status='FAILED_PRE_REPLAY_GUARD', failure_code=str(exc),
                      optimizer_steps=0, new_actual_replays=0, whole_finalist_matrix_complete_and_replayed=False,
                      private_arrays_or_identifiers_exported=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result)); raise SystemExit(1)
    print(json.dumps({key: result[key] for key in ('status', 'available_complete_receipts', 'passed_units', 'failed_units', 'new_actual_replays')}))
    if result['failed_units']: raise SystemExit(1)


if __name__ == '__main__':
    main()

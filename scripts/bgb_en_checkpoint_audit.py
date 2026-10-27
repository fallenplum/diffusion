"""Independently replay saved BG-BEN checkpoints without optimization.

Reads private model/preparation arrays locally, but exports only anonymous unit
labels, aggregate counts, source hashes and numerical errors. The replay uses
the frozen model code; survival counting comes directly from the independent
Phase10 reference module rather than the training evaluation wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

F7_REPAIR_BACKEND = None
if '--runtime-repair' in sys.argv:
    from scripts.bgb_en_f7_determinism_diagnostic import configure_backend
    F7_REPAIR_BACKEND = configure_backend('deterministic')

import numpy as np
import torch

from src.phase10_trimodal.reference import components, pairs, uno_weights


class AuditError(ValueError):
    """Fixed public error code; exception messages never expose private paths."""


def require(condition, code):
    if not bool(condition):
        raise AuditError(code)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def index_hash(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def verify_current_source(lineage, *, config_path, root=ROOT, expected_lineage=None):
    """Reject source or data/config drift before loading any trained weights."""
    require(lineage.get('protocol_sha256') == sha256(config_path), 'CONFIG_SHA256_MISMATCH')
    require(isinstance(lineage.get('source'), dict) and bool(lineage['source']), 'MISSING_SOURCE_CLOSURE')
    for name, expected in lineage['source'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'UNSAFE_SOURCE_REFERENCE')
        path = Path(root) / relative
        require(path.is_file(), 'SOURCE_FILE_MISSING')
        require(sha256(path) == expected, 'SOURCE_SHA256_MISMATCH')
    if expected_lineage is not None:
        require(lineage == expected_lineage, 'SOURCE_DATA_LINEAGE_MISMATCH')


def valid_indices(value, n):
    x = np.asarray(value)
    require(x.ndim == 1 and np.issubdtype(x.dtype, np.integer) and len(x) > 0, 'INVALID_INDEX_VECTOR')
    require(len(np.unique(x)) == len(x) and (x >= 0).all() and (x < n).all(), 'INVALID_INDEX_SET')
    return x


def verify_preprocessing_state(state, fit):
    expected_hash = index_hash(fit)
    require(np.array_equal(state.get('fit'), fit), 'PREPROCESSING_FIT_INDICES_MISMATCH')
    require(state.get('fit_hash') == expected_hash, 'PREPROCESSING_FIT_HASH_MISMATCH')
    for scale in [state['clinical']['scale'], state['S32'], state['molecular']['scale'],
                  state['molecular']['selected_scale']]:
        require(scale.get('fit_hash') == expected_hash, 'SCALER_FIT_HASH_MISMATCH')
    require(state['molecular'].get('fit_hash') == expected_hash, 'MOLECULAR_FIT_HASH_MISMATCH')
    for visual in state['visual'].values():
        require(visual['before'].get('fit_hash') == expected_hash, 'VISUAL_BEFORE_FIT_HASH_MISMATCH')
        require(visual['after'].get('fit_hash') == expected_hash, 'VISUAL_AFTER_FIT_HASH_MISMATCH')
    selected = np.asarray(state['molecular']['selected'])
    require(selected.ndim == 1 and len(np.unique(selected)) == len(selected), 'DUPLICATED_RNA_SELECTED_INDEX')


def verify_teacher_lineage(prepared, n):
    train, valid = valid_indices(prepared['train'], n), valid_indices(prepared['valid'], n)
    require(not np.intersect1d(train, valid).size, 'TRAIN_VALIDATION_OVERLAP')
    require(prepared.get('fit_hash') == index_hash(train), 'ANCHOR_FIT_HASH_MISMATCH')
    require(prepared.get('valid_hash') == index_hash(valid), 'ANCHOR_VALID_HASH_MISMATCH')
    verify_preprocessing_state(prepared['features']['state'], train)
    groups = np.asarray(prepared['teacher_groups'])
    require(groups.shape == (n,) and (groups[valid] == -1).all(), 'TEACHER_GROUP_VALIDATION_LEAKAGE')
    teachers = prepared['teachers']
    require(len(teachers) == 3, 'TEACHER_COUNT_MISMATCH')
    all_held = []
    for group, teacher in enumerate(teachers):
        fit, held = valid_indices(teacher['train'], n), valid_indices(teacher['held'], n)
        require(not np.intersect1d(fit, held).size, 'TEACHER_HELDOUT_OVERLAP')
        require(not np.intersect1d(fit, valid).size and not np.intersect1d(held, valid).size,
                'TEACHER_OUTER_VALIDATION_LEAKAGE')
        require(set(fit) | set(held) == set(train), 'TEACHER_TRAIN_PARTITION_MISMATCH')
        require((groups[held] == group).all(), 'TEACHER_ASSIGNMENT_MISMATCH')
        verify_preprocessing_state(teacher['preprocessing'], fit)
        require(set(teacher['models']) == {'LOCAL', 'CS', 'R'}, 'TEACHER_MODEL_SET_MISMATCH')
        for model in teacher['models'].values():
            coef = np.asarray(model['coef'])
            require(coef.ndim == 1 and np.isfinite(coef).all(), 'INVALID_TEACHER_COEFFICIENT')
            require(np.isfinite(model['scale']['mean']) and np.isfinite(model['scale']['sd'])
                    and model['scale']['sd'] > 0, 'INVALID_TEACHER_RISK_SCALE')
        all_held.extend(held.tolist())
    require(np.array_equal(np.sort(all_held), np.sort(train)), 'TEACHER_HELDOUT_COVERAGE_MISMATCH')
    # Every admitted scalar pair is wholly inside one held block. Checking the
    # whole block against its teacher's fit set checks both pair members at once.
    require((groups[train] >= 0).all(), 'UNCROSSFITTED_TRAINING_PATIENT')
    return {'teachers_checked': len(teachers), 'teacher_Cox_models_checked': 9,
            'full_Cox_models_expected': 4, 'same_teacher_beta_models_expected': 1,
            'training_rows': len(train), 'validation_rows': len(valid),
            'pair_both_members_excluded': True, 'preprocessing_fit_hashes_verified': True}


def verify_full_anchor_predictions(prepared, tolerance=2e-6):
    train, valid = np.asarray(prepared['train']), np.asarray(prepared['valid'])
    maximum = 0.
    require(set(prepared['full_models']) == {'LOCAL', 'GLOBAL', 'CS', 'R'}, 'FULL_ANCHOR_MODEL_SET_MISMATCH')
    for key, model in prepared['full_models'].items():
        values, coef = np.asarray(prepared['features'][key]), np.asarray(model['coef'])
        require(values.ndim == 2 and coef.shape == (values.shape[1],), 'FULL_ANCHOR_COEFFICIENT_SHAPE_MISMATCH')
        raw_risk = values @ coef
        scale = model['scale']
        require(abs(raw_risk[train].mean() - scale['mean']) <= tolerance, 'FULL_ANCHOR_RISK_CENTER_MISMATCH')
        require(abs(raw_risk[train].std() - scale['sd']) <= tolerance, 'FULL_ANCHOR_RISK_SCALE_MISMATCH')
        rebuilt = (raw_risk - scale['mean']) / scale['sd']
        error = float(np.max(np.abs(rebuilt - prepared['full_risks'][key])))
        maximum = max(maximum, error)
        require(error <= tolerance, 'FULL_ANCHOR_COEFFICIENT_REPLAY_MISMATCH')
    for risk_key, model_key in [('CP', 'LOCAL'), ('CS', 'CS'), ('RNA', 'R')]:
        require(np.allclose(prepared[risk_key][valid], prepared['full_risks'][model_key][valid],
                            atol=tolerance, rtol=0), 'VALIDATION_ANCHOR_NOT_FULL_TRAIN_FIT')
    require(np.allclose(prepared['base'], prepared['CP'] + prepared['beta'] * prepared['RNA'],
                        atol=tolerance, rtol=0), 'BASE_CP_SCALAR_IDENTITY_MISMATCH')
    return {'full_Cox_coefficients_replayed': 4, 'maximum_risk_error': maximum,
            'crossfit_vs_fullfit_training_risks_not_conflated': True}


def metrics_from_risk(train_time, train_event, time, event, risk, reference=None):
    risk = np.asarray(risk, float)
    require(risk.shape == np.asarray(time).shape and np.isfinite(risk).all(), 'INVALID_SAVED_RISK')
    weights = uno_weights(train_time, train_event, time, event, tau=60.)
    hn, hd = components(time, event, risk)
    un, ud = components(time, event, risk, weights)
    result = {'Harrell': hn / hd if hd else None, 'Uno60': un / ud if ud else None,
              'Harrell_num': hn, 'Harrell_den': hd, 'Uno_num': un, 'Uno_den': ud}
    if reference is not None:
        left, right = pairs(time, event)
        def credit(values):
            diff = np.asarray(values)[left] - np.asarray(values)[right]
            return np.where(np.abs(diff) <= 1e-8, .5, (diff > 0).astype(float))
        change = credit(risk) - credit(reference)
        result.update(repair=float(np.maximum(change, 0).sum() / hd) if hd else None,
                      damage=float(np.maximum(-change, 0).sum() / hd) if hd else None,
                      Uno_repair=float(np.dot(weights[left], np.maximum(change, 0)) / ud) if ud else None,
                      Uno_damage=float(np.dot(weights[left], np.maximum(-change, 0)) / ud) if ud else None)
    return result


def verify_metric_fields(actual, expected, *, tolerance=1e-10):
    maximum = 0.
    for key in ('Harrell', 'Uno60', 'Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den',
                'repair', 'damage', 'Uno_repair', 'Uno_damage'):
        if key not in actual or key not in expected:
            continue
        a, b = actual[key], expected[key]
        require((a is None) == (b is None), 'METRIC_DEFINEDNESS_MISMATCH')
        if a is not None:
            error = abs(float(a) - float(b)); maximum = max(maximum, error)
            require(np.isfinite(a) and np.isfinite(b) and error <= tolerance, 'INDEPENDENT_METRIC_MISMATCH')
    return maximum


def verify_replayed_risk(saved, rebuilt, n, tolerance=2e-6):
    saved, rebuilt = np.asarray(saved, float), np.asarray(rebuilt, float)
    require(saved.shape == rebuilt.shape == (n,), 'REPLAY_RISK_SHAPE_MISMATCH')
    error = float(np.max(np.abs(saved - rebuilt)))
    require(np.isfinite(error) and error <= tolerance, 'CHECKPOINT_FORWARD_RISK_MISMATCH')
    return error


def rebuild_convex_risk(artifact, receipt, prepared):
    arm, valid = receipt['arm'], np.asarray(prepared['valid'])
    if arm in ('P0', 'F0', 'F1', 'O0'):
        value = {'P0': prepared['full_risks']['GLOBAL'], 'F0': prepared['CP'],
                 'F1': prepared['base'], 'O0': prepared['RNA']}[arm]
        return np.asarray(value)[valid]
    if arm == 'O1':
        x = np.asarray(prepared['features']['rna'], float)
    elif arm.startswith('O2_PCA'):
        state = artifact['feature_state']
        x = np.asarray(prepared['features']['R'], float)
        x = (x - state['mean']) @ np.asarray(state['components']).T
        x = (x - state['post_mean']) / state['post_scale']
    else:
        raise AuditError('UNKNOWN_CONVEX_ARM')
    coef = np.asarray(artifact['coefficient'])
    require(coef.shape == (x.shape[1],), 'CONVEX_COEFFICIENT_SHAPE_MISMATCH')
    return x[valid] @ coef


def is_reference_or_convex(arm):
    return arm in ('P0', 'F0', 'F1', 'O0', 'O1') or arm.startswith('O2_')


def rebuild_query_map(model, raw, prepared, partition):
    from src.bgb_en.protocol import patient_mismatch
    strata = np.array([str(raw['cancer'][i]) + '|' + str(raw['clinical_raw']['AJCC_PATHOLOGIC_TUMOR_STAGE'][i])
                       + '|' + str(raw['platform'][i]) for i in range(len(raw['t']))])
    audits = {}
    for name, indices in [('train', prepared['train']), ('validation', prepared['valid'])]:
        mapping, audit = patient_mismatch(indices, strata, 270077 + partition + (name == 'validation'))
        model.query_map[indices] = mapping[indices]
        require(np.all(model.query_map[indices] != indices), 'QUERY_MAP_HAS_FIXED_POINT')
        require(set(model.query_map[indices]) == set(indices), 'QUERY_MAP_CROSSES_PARTITION')
        audits[name] = audit
    return {'train_fixed_points': 0, 'validation_fixed_points': 0,
            'training_fallbacks': audits['train']['singleton_stratum_fallbacks'],
            'validation_fallbacks': audits['validation']['singleton_stratum_fallbacks']}


def expected_plan(config, stage, arm_filter=None):
    arms = config[stage]['arms'][:]
    if stage == 'R3':
        arms += config[stage].get('additional_real_retraining', [])
    if arm_filter is not None:
        arms = [arm for arm in arms if arm in arm_filter]
    return sum(config['prototype']['inner_splits'] * (1 if is_reference_or_convex(arm) else len(config['prototype']['neural_seeds'])) for arm in arms)


def imported_f7_units(report, *, lineage_sha, audit_sha, reference_sha, recovery_audit_sha):
    """Only admit an actual separate F7 replay under the explicit backend."""
    require(report.get('schema') == 'bgb_en_checkpoint_replay_audit_v1'
            and report.get('F7_runtime_repair') is True and report.get('arm_filter') == ['F7']
            and str(report.get('device')).startswith('cuda')
            and report.get('passed_units') == 6 and report.get('failed_units') == 0
            and report.get('whole_requested_matrix_complete_and_replayed') is True, 'IMPORTED_F7_AUDIT_NOT_COMPLETE')
    require(report.get('audit_script_sha256') == audit_sha
            and report.get('independent_reference_sha256') == reference_sha
            and report.get('recovery_audit_sha256') == recovery_audit_sha
            and report.get('source_lineage_sha256') == lineage_sha, 'IMPORTED_F7_AUDIT_SOURCE_DIFFERS')
    expected = {(part, seed) for part in (0, 1, 2) for seed in (31, 47)}
    units = report.get('units', [])
    require(len(units) == 6 and {(row['inner_partition'], row['seed']) for row in units} == expected
            and all(row['stage'] == 'R3' and row['arm'] == 'F7' and row['status'] == 'PASS'
                    and row.get('replay_type') == 'ACTUAL_FINAL80_STATE_DICT_FORWARD'
                    and row.get('F7_runtime_repair') is True and row.get('runtime_receipt_sha256') for row in units),
            'IMPORTED_F7_AUDIT_CELL_SET_DIFFERS')
    return {(row['stage'], row['arm'], row['inner_partition'], row['seed']): row for row in units}


def run(args):
    require(not args.runtime_repair or F7_REPAIR_BACKEND is not None, 'F7_RUNTIME_BOOTSTRAP_MISSING')
    from scripts.bgb_en_run import load_inputs
    from src.bgb_en.data import read_patch_cache
    from src.bgb_en.training import PatchBank, MechanismModel, predict
    from src.bgb_en.protocol import fixed_inner_splits
    from scripts.bgb_en_recovery_audit import verify_runtime_receipt

    torch.set_num_threads(2)
    assets, config, raw, membership, current_lineage = load_inputs(args)
    verify_current_source(current_lineage, config_path=args.config)
    lineage_sha = hashlib.sha256(json.dumps(current_lineage, sort_keys=True).encode()).hexdigest()
    recovery_audit_sha = sha256(ROOT/'scripts/bgb_en_recovery_audit.py')
    independent_reference_sha = sha256(sys.modules[components.__module__].__file__)
    imported_f7, imported_f7_sha = {}, None
    if args.f7_audit is not None:
        require(not args.runtime_repair, 'F7_AUDIT_IMPORT_AND_RUNTIME_FLAGS_CONFLICT')
        imported_f7 = imported_f7_units(json.loads(args.f7_audit.read_text()), lineage_sha=lineage_sha,
            audit_sha=sha256(__file__), reference_sha=independent_reference_sha, recovery_audit_sha=recovery_audit_sha)
        imported_f7_sha = sha256(args.f7_audit)
    stages = ('R1', 'R2', 'R3') if args.stage == 'all' else (args.stage,)
    available = []
    seen_units = set()
    for path in sorted((args.private / 'cells').glob('**/COMPLETE.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('stage') not in stages:
            continue
        if args.arm is not None and receipt.get('arm') not in args.arm:
            continue
        require(receipt.get('status') == 'COMPLETE', 'NONCOMPLETE_RECEIPT')
        arms = config[receipt['stage']]['arms'][:]
        if receipt['stage'] == 'R3':
            arms += config['R3'].get('additional_real_retraining', [])
        require(receipt.get('arm') in arms, 'ARM_OUTSIDE_LOCKED_MATRIX')
        require(receipt.get('outer_fold') == config['prototype']['outer_fold'], 'OUTER_FOLD_OUTSIDE_PROTOTYPE')
        require(receipt.get('inner_partition') in range(config['prototype']['inner_splits']), 'INNER_PARTITION_OUTSIDE_LOCK')
        allowed_seeds = [31] if is_reference_or_convex(receipt['arm']) else config['prototype']['neural_seeds']
        require(receipt.get('seed') in allowed_seeds, 'SEED_OUTSIDE_LOCK')
        identity = tuple(receipt[k] for k in ('stage', 'arm', 'outer_fold', 'inner_partition', 'seed'))
        require(identity not in seen_units, 'DUPLICATE_COMPLETE_UNIT')
        seen_units.add(identity)
        verify_current_source(receipt['lineage'], config_path=args.config, expected_lineage=current_lineage)
        require(receipt.get('outer_test_evaluated') is False, 'UNEXPECTED_OUTER_EVALUATION')
        if receipt['arm'] == 'F7' and (path.parent/'RUNTIME_REPAIR_COMPLETE.json').exists():
            require(args.runtime_repair or bool(imported_f7), 'REPAIRED_F7_REQUIRES_SEPARATE_RUNTIME_AUDIT')
        available.append((path, receipt))
    planned = sum(expected_plan(config, stage, args.arm) for stage in stages)
    require(planned > 0, 'EMPTY_REQUESTED_ARM_FILTER')
    previous = {}
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text())
        if (old.get('audit_script_sha256') == sha256(__file__)
                and old.get('independent_reference_sha256') == sha256(sys.modules[components.__module__].__file__)):
            for row in old.get('units', []):
                if row.get('status') == 'PASS':
                    previous[row['stage'], row['arm'], row['inner_partition'], row['seed']] = row
    rows, preprocessing_cache = [], {}
    bank_key, bank = None, None
    new_replays, reused, imported_f7_reused = 0, 0, 0
    for path, receipt in available:
        label = {key: receipt[key] for key in ('stage', 'arm', 'outer_fold', 'inner_partition', 'seed')}
        artifact_path = path.parent / 'model.pt'
        artifact_sha = sha256(artifact_path)
        require(artifact_sha == receipt['artifact_sha256'], 'ARTIFACT_SHA256_MISMATCH')
        runtime_guard = None
        if label['arm'] == 'F7':
            require((path.parent/'RUNTIME_REPAIR_COMPLETE.json').exists(), 'F7_RECOVERY_RUNTIME_RECEIPT_REQUIRED')
            try:
                runtime_guard = verify_runtime_receipt(path.parent, receipt,
                    expected_backend=F7_REPAIR_BACKEND if args.runtime_repair else None)
            except ValueError as exc:
                raise AuditError('F7_RUNTIME_RECEIPT_GUARD_FAILED') from exc
            if not args.runtime_repair:
                separate = imported_f7.get((label['stage'], label['arm'], label['inner_partition'], label['seed']))
                require(separate is not None and separate['artifact_sha256'] == artifact_sha
                        and separate['receipt_sha256'] == sha256(path) and separate['lineage'] == lineage_sha
                        and separate['runtime_receipt_sha256'] == runtime_guard['runtime_receipt_sha256']
                        and separate['runtime_backend'] == runtime_guard['backend'], 'IMPORTED_F7_CHECKPOINT_RUNTIME_DIFFERS')
                rows.append({**separate, 'replayed_this_invocation': False, 'previous_actual_replay_reused': True,
                             'separate_F7_audit_sha256': imported_f7_sha})
                reused += 1; imported_f7_reused += 1
                continue
        cached = previous.get((label['stage'], label['arm'], label['inner_partition'], label['seed']))
        if (cached and cached.get('artifact_sha256') == artifact_sha and cached.get('receipt_sha256') == sha256(path)
            and cached.get('lineage') == hashlib.sha256(json.dumps(current_lineage, sort_keys=True).encode()).hexdigest()
            and (runtime_guard is None or (cached.get('runtime_receipt_sha256') == runtime_guard['runtime_receipt_sha256']
                                          and cached.get('runtime_backend') == runtime_guard['backend']))):
            rows.append({**cached, 'replayed_this_invocation': False, 'previous_actual_replay_reused': True}); reused += 1
            continue
        if args.limit and new_replays >= args.limit:
            continue
        row = {**label, 'status': 'FAILED', 'artifact_sha256': artifact_sha,
               'receipt_sha256': sha256(path), 'optimizer_steps': 0, 'replayed_this_invocation': True,
               'lineage': hashlib.sha256(json.dumps(current_lineage, sort_keys=True).encode()).hexdigest()}
        if runtime_guard is not None:
            row.update(F7_runtime_repair=True, runtime_receipt_sha256=runtime_guard['runtime_receipt_sha256'],
                       runtime_backend=runtime_guard['backend'])
        try:
            outer, partition = int(receipt['outer_fold']), int(receipt['inner_partition'])
            prep_key = outer, partition
            if prep_key not in preprocessing_cache:
                stem = f'outer{outer}_inner{partition}'
                prep_path, prep_receipt_path = args.private / 'prepared' / (stem + '.pt'), args.private / 'prepared' / (stem + '.json')
                prep_receipt = json.loads(prep_receipt_path.read_text())
                require(prep_receipt['lineage'] == current_lineage, 'PREPARED_LINEAGE_MISMATCH')
                require(prep_receipt['sha256'] == sha256(prep_path), 'PREPARED_ARTIFACT_SHA256_MISMATCH')
                prepared = torch.load(prep_path, map_location='cpu', weights_only=False)
                require(prepared['lineage'] == current_lineage, 'PREPARED_MODEL_LINEAGE_MISMATCH')
                teacher_audit = verify_teacher_lineage(prepared, len(raw['t']))
                wanted_train, wanted_valid = fixed_inner_splits(raw['folds'], raw['e'], outer=outer)[partition]
                require(np.array_equal(wanted_train, prepared['train']) and np.array_equal(wanted_valid, prepared['valid']),
                        'PREPARED_SPLIT_DIFFERS_FROM_LOCK')
                anchor_audit = verify_full_anchor_predictions(prepared)
                preprocessing_cache[prep_key] = prepared, teacher_audit, anchor_audit
            prepared, teacher_audit, anchor_audit = preprocessing_cache[prep_key]
            artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
            require(artifact['lineage'] == current_lineage, 'MODEL_LINEAGE_MISMATCH')
            for key in ('stage', 'arm', 'seed'):
                require(artifact[key] == receipt[key], 'MODEL_UNIT_IDENTITY_MISMATCH')
            require(np.array_equal(artifact['train'], prepared['train']) and np.array_equal(artifact['valid'], prepared['valid']),
                    'MODEL_TRAIN_VALID_INDICES_MISMATCH')
            train, valid = np.asarray(prepared['train']), np.asarray(prepared['valid'])
            if is_reference_or_convex(receipt['arm']):
                saved = np.asarray(artifact['risk'], float)
                rebuilt = rebuild_convex_risk(artifact, receipt, prepared)
                replay_type = 'SAVED_COEFFICIENT_OR_UPSTREAM_REFERENCE'
                reference = np.asarray(prepared['full_risks']['LOCAL'])[valid]
                query_audit = None
                official_mamba_loaded = False
            else:
                epoch = config['training']['epochs']
                require(receipt['epochs'] == epoch and epoch in artifact['snapshots'], 'FINAL_CHECKPOINT_MISSING')
                snapshot = artifact['snapshots'][epoch]
                saved = np.asarray(snapshot['risk'], float)
                key = ('patch_rand' if receipt['arm'] == 'P7_RAND' else 'patch_diverse' if receipt['arm'] == 'P7_DIVERSE'
                       else 'patch_grid' if receipt['arm'] == 'P7_GRID' else 'patch_uni2' if receipt['arm'].endswith('_UNI2') else 'patch_regions')
                if bank_key != key:
                    if bank is not None:
                        del bank
                    bank = PatchBank(read_patch_cache(assets[key]), args.device); bank_key = key
                torch.manual_seed(receipt['seed']); np.random.seed(receipt['seed'])
                model = MechanismModel(receipt['stage'], receipt['arm'], prepared, membership, bank, args.device,
                                       gamma=config[receipt['stage']].get('gamma', .1),
                                       intervention='wrong_query' if receipt['arm'] == 'F5_WRONG_QUERY' else None)
                model.load_state_dict(snapshot['state_dict'], strict=True)
                official_mamba_loaded = any(type(module).__module__.startswith('mamba_ssm')
                                            for module in model.modules())
                query_audit = rebuild_query_map(model, raw, prepared, partition)
                rebuilt, _ = predict(model, valid, batch_size=8)
                reference = model.offset[valid].detach().cpu().numpy()
                replay_type = 'ACTUAL_FINAL80_STATE_DICT_FORWARD'
                del model
            error = verify_replayed_risk(saved, rebuilt, len(valid), args.tolerance)
            saved_metrics = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], saved, reference)
            metric_error = verify_metric_fields(saved_metrics, receipt['metrics'])
            rebuilt_metrics = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], rebuilt)
            # Tiny permitted forward errors can change 1e-8 tie classification;
            # expose that sensitivity rather than silently changing saved scores.
            replay_rank_error = max(abs(float(rebuilt_metrics[k]) - float(saved_metrics[k]))
                                    for k in ('Harrell_num', 'Uno_num') if rebuilt_metrics[k] is not None)
            controls_checked, controls_error = 0, 0.
            for control, value in artifact.get('frozen_controls', {}).items():
                require(control in receipt.get('frozen_controls', {}), 'MISSING_CONTROL_METRIC_RECEIPT')
                control_metrics = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid],
                                                    value['risk'], np.asarray(prepared['base'])[valid])
                controls_error = max(controls_error, verify_metric_fields(control_metrics, receipt['frozen_controls'][control]))
                controls_error = max(controls_error, verify_metric_fields(control_metrics, value['metrics']))
                controls_checked += 1
            row.update(status='PASS', replay_type=replay_type, maximum_risk_error=error,
                       independent_saved_metric_max_error=metric_error, frozen_controls_checked=controls_checked,
                       independent_control_metric_max_error=controls_error,
                       replay_ranking_numerator_max_error=replay_rank_error,
                       replay_ranking_status='EXACT_CONCORDANCE' if replay_rank_error <= 1e-10 else 'ROUNDING_SENSITIVE_WITHIN_RISK_TOLERANCE',
                       teacher_lineage=teacher_audit, full_anchor_replay=anchor_audit,
                       wrong_query_map=query_audit,
                       official_mamba_weights_actually_loaded=official_mamba_loaded)
        except AuditError as exc:
            row.update(failure_code=str(exc), failure_type=type(exc).__name__)
        except Exception as exc:
            # Raw runtime messages may contain private file paths; keep them
            # out of the public JSON. Operators can rerun that anonymous unit.
            row.update(failure_code='UNEXPECTED_RUNTIME_FAILURE', failure_type=type(exc).__name__)
            error_root = args.private / 'checkpoint_audit_private'
            error_root.mkdir(parents=True, exist_ok=True)
            error_label = f"{label['stage']}_{label['arm']}_p{label['inner_partition']}_s{label['seed']}.log"
            (error_root / error_label).write_text(traceback.format_exc(), encoding='utf-8')
            row['private_runtime_log_created'] = True
            print(json.dumps({**label, 'private_runtime_error_type': type(exc).__name__}), flush=True)
        rows.append(row); new_replays += 1
        print(json.dumps({**label, 'status': row['status'], 'replay': row.get('replay_type'),
                          'risk_error': row.get('maximum_risk_error'), 'failure_code': row.get('failure_code')}), flush=True)
    passed = sum(row['status'] == 'PASS' for row in rows)
    failed = sum(row['status'] != 'PASS' for row in rows)
    status = ('FAILED_SOME_AVAILABLE_CHECKPOINTS' if failed else 'PASS_COMPLETE_LOCKED_MATRIX' if passed == planned
              else 'PASS_AVAILABLE_SUBSET_NOT_WHOLE_MATRIX')
    result = {'schema': 'bgb_en_checkpoint_replay_audit_v1', 'status': status,
              'date_utc': datetime.now(timezone.utc).isoformat(), 'stage_filter': args.stage,
              'arm_filter': args.arm, 'F7_runtime_repair': args.runtime_repair,
              'device': args.device,
              'explicit_runtime_backend': F7_REPAIR_BACKEND,
              'imported_separate_F7_actual_replays': imported_f7_reused, 'separate_F7_audit_sha256': imported_f7_sha,
              'requested_limit_new_replays': args.limit, 'planned_unique_cells_in_stage_filter': planned,
              'available_complete_receipts': len(available), 'new_actual_replays': new_replays,
              'previous_actual_replays_reused': reused, 'passed_units': passed, 'failed_units': failed,
              'whole_requested_matrix_complete_and_replayed': passed == planned and not failed,
              'optimizer_steps': 0, 'risk_tolerance': args.tolerance,
              'new_unique_preparations_checked': len(preprocessing_cache),
              'new_unique_teacher_Cox_lineage_guards': 9 * len(preprocessing_cache),
              'new_unique_full_Cox_coefficient_replays': 4 * len(preprocessing_cache),
              'same_teacher_beta_guard': 'both_pair_members_excluded_by_teacher_assignment;no_claim_of_reoptimizing_beta',
              'private_arrays_or_identifiers_exported': False,
              'upstream_teacher_guard_scope': 'index_sets_and_every_preprocessing_fit_hash;no_claim_of_refitting_each_teacher_PCA',
              'full_anchor_guard_scope': 'actual_saved_Cox_coefficient_replay_and_teacher_own_training_risk_scaling',
              'replay_metric_scope': 'independent_Phase10_reference_on_saved_risks;actual_state_dict_forward_bound_checked_separately',
              'source_lineage_sha256': hashlib.sha256(json.dumps(current_lineage, sort_keys=True).encode()).hexdigest(),
              'independent_reference_sha256': sha256(sys.modules[components.__module__].__file__),
              'audit_script_sha256': sha256(__file__), 'units': rows}
    result['recovery_audit_sha256'] = recovery_audit_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n'); temporary.replace(args.output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--stage', choices=['all', 'R1', 'R2', 'R3'], default='all')
    parser.add_argument('--arm', action='append', help='optional exact arm filter; repeat for multiple arms')
    parser.add_argument('--runtime-repair', action='store_true', help='separate deterministic F7 process; requires --arm F7')
    parser.add_argument('--f7-audit', type=Path, help='import six verified separate F7 replays; do not forward F7 under inherited backend')
    parser.add_argument('--limit', type=int, default=0, help='Maximum new actual replays; zero means every available unit')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tolerance', type=float, default=2e-6)
    parser.add_argument('--resume', action='store_true', help='Reuse previously executed passing replays with unchanged source/artifact hashes')
    args = parser.parse_args()
    if args.runtime_repair and args.arm != ['F7']:
        raise ValueError('--runtime-repair requires an isolated --arm F7 process')
    if args.runtime_repair and not str(args.device).startswith('cuda'):
        raise ValueError('Runtime-repaired F7 requires the original CUDA backend; CPU cannot qualify this replay')
    if args.limit < 0 or not 0 < args.tolerance <= 2e-6:
        raise ValueError('Replay limit/tolerance outside locked audit contract')
    try:
        result = run(args)
    except AuditError as exc:
        result = {'schema': 'bgb_en_checkpoint_replay_audit_v1', 'status': 'FAILED_PRE_REPLAY_GUARD',
                  'failure_code': str(exc), 'passed_units': 0, 'failed_units': 0,
                  'new_actual_replays': 0, 'previous_actual_replays_reused': 0,
                  'available_complete_receipts': None, 'whole_requested_matrix_complete_and_replayed': False,
                  'optimizer_steps': 0, 'private_arrays_or_identifiers_exported': False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        raise SystemExit(1)
    print(json.dumps({key: result[key] for key in ('status', 'available_complete_receipts', 'passed_units',
                                                  'failed_units', 'new_actual_replays', 'previous_actual_replays_reused')}), flush=True)
    if result['failed_units']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

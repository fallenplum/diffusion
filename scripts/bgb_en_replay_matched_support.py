"""Actual final80 replay for 12 matched-support units; no optimization.

GPU forward execution is explicit and intended only after the training queue.
The --qualify-cpu path performs synthetic dispatch/provenance guards without
reading cohort inputs or executing any model. Rebuilt risks remain private.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import bgb_en_run as driver
from scripts.bgb_en_checkpoint_audit import (
    AuditError, require, sha256, index_hash, verify_current_source, verify_teacher_lineage,
    verify_full_anchor_predictions, rebuild_query_map, verify_replayed_risk,
    metrics_from_risk, verify_metric_fields,
)
from scripts.bgb_en_matched_support import SPEC_PATH, base_model
from scripts.bgb_en_matched_support_plan import verify_constructor_and_gamma
from scripts.bgb_en_report_matched_support import verify_receipt, inspect_saved_state, expected_null_lineage
from src.phase10_trimodal.reference import pairs, uno_weights

RISK_TOLERANCE = 2e-6
METRIC_TOLERANCE = 1e-10


def risk_diagnostics(train_t, train_e, t, e, saved, rebuilt, reference):
    """Expose tie/rank sensitivity separately from the fixed forward guard."""
    saved_metric = metrics_from_risk(train_t, train_e, t, e, saved, reference)
    rebuilt_metric = metrics_from_risk(train_t, train_e, t, e, rebuilt, reference)
    left, right = pairs(t, e)
    weights = uno_weights(train_t, train_e, t, e, tau=60.)
    def credit(risk):
        difference = np.asarray(risk, float)[left] - np.asarray(risk, float)[right]
        return np.where(np.abs(difference) <= 1e-8, .5, (difference > 0).astype(float))
    old, new = credit(saved), credit(rebuilt)
    change = new - old
    result = {
        'saved_independent_metrics': saved_metric, 'rebuilt_independent_metrics': rebuilt_metric,
        'delta_Harrell_rebuilt_minus_saved': rebuilt_metric['Harrell'] - saved_metric['Harrell'],
        'delta_Uno60_rebuilt_minus_saved': rebuilt_metric['Uno60'] - saved_metric['Uno60'],
        'delta_Harrell_numerator': float(change.sum()),
        'delta_Uno60_numerator': float(np.dot(weights[left], change)),
        'eligible_pair_count': int(len(left)), 'changed_pair_credit_count': int(np.count_nonzero(change)),
        'pair_credit_gain_count': float(np.maximum(change, 0).sum()),
        'pair_credit_loss_count': float(np.maximum(-change, 0).sum()),
        'Uno_pair_credit_gain_weight': float(np.dot(weights[left], np.maximum(change, 0))),
        'Uno_pair_credit_loss_weight': float(np.dot(weights[left], np.maximum(-change, 0))),
        'saved_tied_pair_count': int(np.count_nonzero(old == .5)),
        'rebuilt_tied_pair_count': int(np.count_nonzero(new == .5)),
        'rank_sensitivity_present': bool(np.any(change)), 'pair_tolerance': 1e-8,
    }
    return result


def verify_risk_with_fixed_guard(saved, rebuilt, n):
    # The caller cannot raise the threshold through a CLI or saved receipt.
    return verify_replayed_risk(saved, rebuilt, n, tolerance=RISK_TOLERANCE)


def synthetic_qualification():
    config = json.loads((ROOT / 'configs/bgb_en/prototype_v1.json').read_text())
    dispatch = verify_constructor_and_gamma(config)
    checks = []
    for stage, variant, gamma, arm in [('R2', 'O4_MATCHED_SUPPORT', .3, 'O4'),
                                      ('R3', 'F5_MATCHED_SUPPORT', .1, 'F5')]:
        called = []
        def constructor(*args, **kwargs):
            called.append((args, kwargs)); return None
        member = torch.tensor([[True, False], [False, True]])
        base_model(constructor, stage, variant, {}, member, None, 'cpu', gamma=gamma)
        require(called[0][0][1] == arm and called[0][0][3] is member
                and called[0][1]['gamma'] == gamma, 'SYNTHETIC_CONSTRUCTOR_GUARD_FAILED')
        checks.append({'variant': variant, 'base_arm': arm, 'gamma': gamma,
                       'same_membership_object': True, 'second_shuffle_applied': False})
    return {'schema': 'bgb_en_matched_support_replay_qualification_v1',
        'status': 'PASS_SYNTHETIC_CPU_ONLY', 'dispatch': dispatch, 'checks': checks,
        'actual_model_forwards': 0, 'optimizer_calls': 0, 'GPU_calls': 0,
        'cohort_inputs_read': False, 'risk_tolerance': RISK_TOLERANCE,
        'replay_script_sha256': sha256(__file__)}


def run(args):
    from src.bgb_en.training import PatchBank, MechanismModel, predict
    from src.bgb_en.data import read_patch_cache
    from src.bgb_en.protocol import fixed_inner_splits

    initial_cgroup = None
    if args.device == 'cpu':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_REPLAY_REQUIRES_GPU_HIDDEN')
        cg = Path('/sys/fs/cgroup')
        initial_cgroup = {name: (cg / name).read_text().strip() for name in ('memory.max', 'memory.current')}
        require(initial_cgroup['memory.max'] != 'max' and
                int(initial_cgroup['memory.max']) - int(initial_cgroup['memory.current']) >= 8 * 1024**3,
                'CPU_REPLAY_MEMORY_HEADROOM_BELOW_8GIB')
    assets, config, raw, member, core = driver.load_inputs(args)
    verify_current_source(core, config_path=args.config)
    verify_constructor_and_gamma(config)
    source_lock_path = args.private / 'LOCK.json'
    lock = json.loads(source_lock_path.read_text())
    launch_path = args.private / 'LAUNCH_LOCK.json'
    launch = json.loads(launch_path.read_text())
    require(lock['lineage']['core_lineage'] == core, 'CURRENT_CORE_LINEAGE_MISMATCH')
    require(lock['lineage']['specification'] == json.loads(SPEC_PATH.read_text()), 'MATCHED_SUPPORT_SPEC_CHANGED')
    require(launch['source_lock_sha256'] == sha256(source_lock_path)
            and launch['private_job_plan_sha256'] == sha256(args.private / 'JOB_PLAN.json'), 'AUXILIARY_LOCK_CHANGED')
    for relative, expected in lock['lineage']['auxiliary_source'].items():
        require(sha256(ROOT / relative) == expected, 'FROZEN_AUXILIARY_SOURCE_CHANGED')
    require(launch['planner_sha256'] == sha256(ROOT / 'scripts/bgb_en_matched_support_plan.py'), 'PLANNER_SOURCE_CHANGED')
    selection_paths = [args.inner_report, args.selection_lock]
    selection_before = {p.name: sha256(p) for p in selection_paths}
    require(selection_before == launch['selection_sha256'], 'SELECTION_LOCK_CHANGED')
    source_closure = {name: sha256(ROOT / name) for name in (
        'scripts/bgb_en_replay_matched_support.py', 'scripts/bgb_en_checkpoint_audit.py',
        'scripts/bgb_en_report_matched_support.py', 'scripts/bgb_en_matched_support.py',
        'scripts/bgb_en_matched_support_plan.py', 'src/phase10_trimodal/reference.py')}
    spec = lock['lineage']['specification']
    locked_parts = {r['inner_partition']: r for r in lock['partitions']}
    expected = {(d['stage'], v, p, s) for v, d in spec['variants'].items()
                for p in spec['partitions'] for s in spec['seeds']}
    available = {}
    for path in sorted((args.private / 'cells').glob('**/COMPLETE.json')):
        row = json.loads(path.read_text()); key = tuple(row[k] for k in ('stage', 'arm', 'inner_partition', 'seed'))
        require(key in expected and key not in available, 'DUPLICATE_OR_UNPLANNED_REPLAY_UNIT')
        stage, variant, part, seed = key
        definition = spec['variants'][variant]
        lineage = expected_null_lineage(core, lock, source_lock_path, locked_parts[part], variant, definition['base_arm'])
        verify_receipt(row, stage=stage, arm=variant, partition=part, seed=seed, lineage=lineage)
        require(sha256(path.parent / 'model.pt') == row['artifact_sha256'], 'MODEL_ARTIFACT_HASH_MISMATCH')
        available[key] = path, row
    previous = {}
    if args.resume and args.output.exists():
        old = json.loads(args.output.read_text())
        require(old.get('source_closure') == source_closure
                and old.get('source_lock_sha256') == sha256(source_lock_path)
                and old.get('launch_lock_sha256') == sha256(launch_path)
                and old.get('device') == args.device and old.get('risk_tolerance') == RISK_TOLERANCE,
                'REPLAY_RESUME_CONTRACT_CHANGED')
        previous = {(r['stage'], r['arm'], r['inner_partition'], r['seed']): r
                    for r in old.get('units', []) if r.get('status') == 'PASS'}
    preparation, bank, rows = {}, None, []
    actual_forward_units, reused = 0, 0

    def save_report():
        passed = sum(r['status'] == 'PASS' for r in rows)
        failed = sum(r['status'] == 'FAILED' for r in rows)
        status = 'PASS_COMPLETE_12_ACTUAL_FORWARDS' if passed == len(expected) and not failed else (
            'PARTIAL_FAILED_UNITS' if failed else 'PARTIAL_OR_NOT_YET_AVAILABLE')
        result = {'schema': 'bgb_en_matched_support_risk_replay_v1', 'status': status,
            'planned_units': len(expected), 'available_complete_units': len(available),
            'passed_units': passed, 'failed_units': failed,
            'whole_requested_matrix_complete_and_replayed': passed == len(expected) and not failed,
            'new_actual_forward_units': actual_forward_units, 'previous_actual_forward_units_reused': reused,
            'units': rows, 'source_closure': source_closure, 'device': args.device,
            'risk_tolerance': RISK_TOLERANCE, 'metric_tolerance': METRIC_TOLERANCE,
            'source_lock_sha256': sha256(source_lock_path), 'launch_lock_sha256': sha256(launch_path),
            'actual_backend': 'CPU_vs_GPU_saved' if args.device == 'cpu' else 'GPU_vs_GPU_saved',
            'CPU_initial_cgroup_memory': initial_cgroup,
            'torch_threads': torch.get_num_threads(), 'BLAS_threads_requested': 2,
            'selection_sha256_before': selection_before,
            'selection_sha256_after': {p.name: sha256(p) for p in selection_paths},
            'optimizer_calls': 0, 'teacher_refits': 0, 'preprocessing_refits': 0,
            'forward_batch_size': 8, 'patient_or_gene_indices_exported': False,
            'raw_risks_public': False, 'outer_performance_read': False, 'R4_selection_modified': False,
            'rank_sensitivity_does_not_overwrite_saved_metrics': True}
        require(result['selection_sha256_after'] == selection_before, 'SELECTION_CHANGED_DURING_REPLAY')
        driver.write(args.output, result)
        return result

    for key in sorted(expected):
        stage, variant, part, seed = key
        label = {'stage': stage, 'arm': variant, 'inner_partition': part, 'seed': seed}
        if key not in available:
            rows.append({**label, 'status': 'NOT_COMPLETE', 'actual_model_forward': False})
            continue
        path, receipt = available[key]
        old = previous.get(key)
        private_risk_path = args.private / 'risk_replay' / f'{variant}_p{part}_seed{seed}.npz'
        if (old is not None and old.get('receipt_sha256') == sha256(path)
                and old.get('artifact_sha256') == receipt['artifact_sha256']
                and private_risk_path.is_file() and old.get('private_replayed_risk_sha256') == sha256(private_risk_path)):
            rows.append({**old, 'replayed_this_invocation': False, 'previous_actual_forward_reused': True})
            reused += 1
            continue
        if args.limit and actual_forward_units >= args.limit:
            rows.append({**label, 'status': 'NOT_REPLAYED_LIMIT', 'actual_model_forward': False})
            continue
        row = {**label, 'status': 'FAILED', 'actual_model_forward': False, 'optimizer_calls': 0,
            'receipt_sha256': sha256(path), 'artifact_sha256': receipt['artifact_sha256'],
            'actual_backend': 'CPU_vs_GPU_saved' if args.device == 'cpu' else 'GPU_vs_GPU_saved',
            'saved_training_GPU_peak_bytes': receipt['peak_GPU_bytes'],
            'replayed_this_invocation': True, 'previous_actual_forward_reused': False}
        try:
            require(receipt['peak_GPU_bytes'] > 0, 'SAVED_TRAINING_GPU_BACKEND_NOT_EVIDENCED')
            if part not in preparation:
                locked_part = locked_parts[part]
                prep_path = args.primary_private / 'prepared' / f'outer1_inner{part}.pt'
                require(sha256(prep_path) == locked_part['prepared_sha256'], 'PREPARED_HASH_MISMATCH')
                prepared = torch.load(prep_path, map_location='cpu', weights_only=False)
                require(prepared['lineage'] == core, 'PREPARED_LINEAGE_MISMATCH')
                teacher_audit = verify_teacher_lineage(prepared, len(raw['t']))
                anchor_audit = verify_full_anchor_predictions(prepared)
                wanted = fixed_inner_splits(raw['folds'], raw['e'], outer=1)[part]
                require(np.array_equal(wanted[0], prepared['train']) and np.array_equal(wanted[1], prepared['valid']),
                        'PREPARED_SPLIT_DIFFERS_FROM_LOCK')
                null_path = args.private / 'nulls' / f'partition_{part}.pt'
                require(sha256(null_path) == locked_part['null_artifact_sha256'], 'NULL_ARTIFACT_HASH_MISMATCH')
                null = torch.load(null_path, map_location='cpu', weights_only=False)
                require(index_hash(null['membership'].numpy()) == locked_part['actual_membership_sha256']
                        and index_hash(null['permutation'].numpy()) == locked_part['actual_permutation_sha256'],
                        'NULL_MEMBERSHIP_OR_PERMUTATION_HASH_MISMATCH')
                preparation[part] = prepared, null['membership'], teacher_audit, anchor_audit
            prepared, null, teacher_audit, anchor_audit = preparation[part]
            artifact = torch.load(path.parent / 'model.pt', map_location='cpu', weights_only=False)
            require(artifact['lineage'] == receipt['lineage'] and artifact['stage'] == stage
                    and artifact['arm'] == variant and artifact['seed'] == seed, 'ARTIFACT_IDENTITY_OR_LINEAGE_MISMATCH')
            require(np.array_equal(artifact['train'], prepared['train']) and np.array_equal(artifact['valid'], prepared['valid']),
                    'ARTIFACT_TRAIN_VALID_MISMATCH')
            snapshot = artifact['snapshots'][80]
            inspect_saved_state(snapshot['state_dict'], null)
            require(snapshot['metrics'] == receipt['metrics'], 'SAVED_SNAPSHOT_METRICS_MISMATCH')
            if stage == 'R3' and bank is None:
                bank = PatchBank(read_patch_cache(assets['patch_regions']), args.device)
            torch.manual_seed(seed); np.random.seed(seed)
            model = base_model(MechanismModel, stage, variant, prepared, null,
                               bank if stage == 'R3' else None, args.device, gamma=config[stage]['gamma'])
            require(model.arm == spec['variants'][variant]['base_arm']
                    and model.gamma == (.3 if stage == 'R2' else .1), 'ACTUAL_MODEL_BASE_ARM_OR_GAMMA_MISMATCH')
            model.load_state_dict(snapshot['state_dict'], strict=True)
            query_audit = rebuild_query_map(model, raw, prepared, part)
            train, valid = np.asarray(prepared['train']), np.asarray(prepared['valid'])
            saved = np.asarray(snapshot['risk'], float)
            rebuilt, delta = predict(model, valid, batch_size=8)
            actual_forward_units += 1
            row['actual_model_forward'] = True
            row['risk_max_abs_error'] = float(np.max(np.abs(saved - rebuilt)))
            reference = model.offset[valid].detach().cpu().numpy()
            diagnostic = risk_diagnostics(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid],
                                          saved, rebuilt, reference)
            private_risk_path.parent.mkdir(parents=True, exist_ok=True)
            # The private file is an audit artifact; original model risks stay unchanged.
            np.savez_compressed(private_risk_path, validation_indices=valid, rebuilt_risk=rebuilt,
                                rebuilt_delta=delta, saved_risk=saved)
            row.update(**diagnostic,
                replay_type='ACTUAL_FINAL80_STATE_DICT_FORWARD', base_arm=model.arm, gamma=model.gamma,
                actual_membership_sha256=locked_parts[part]['actual_membership_sha256'],
                prepared_sha256=locked_parts[part]['prepared_sha256'],
                private_replayed_risk_sha256=sha256(private_risk_path),
                teacher_lineage_audit=teacher_audit, full_anchor_audit=anchor_audit, query_map_audit=query_audit)
            # Retain scalars/ranking sensitivity and the private risk witness
            # even if the unchanged fixed risk gate fails on another backend.
            metric_error = verify_metric_fields(diagnostic['saved_independent_metrics'], receipt['metrics'],
                                                 tolerance=METRIC_TOLERANCE)
            row['saved_independent_metric_max_error'] = metric_error
            verify_risk_with_fixed_guard(saved, rebuilt, len(valid))
            row['status'] = 'PASS'
            del model, artifact
        except Exception as exc:
            # Fixed public errors omit private paths, identifiers and data values.
            row.update(error_type=type(exc).__name__, error_code=str(exc) if isinstance(exc, AuditError) else 'REPLAY_EXCEPTION')
        rows.append(row)
        save_report()
    return save_report()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualify-cpu', action='store_true')
    p.add_argument('--assets', type=Path)
    p.add_argument('--private', type=Path)
    p.add_argument('--primary-private', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en/MATCHED_SUPPORT_RISK_REPLAY.json')
    p.add_argument('--inner-report', type=Path, default=ROOT / 'reports/bgb_en/INNER_REPORT_FOR_SELECTION.json')
    p.add_argument('--selection-lock', type=Path, default=ROOT / 'reports/bgb_en/OUTER_SELECTION_LOCK.json')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); torch.set_num_threads(2)
    require(args.limit >= 0, 'NEGATIVE_REPLAY_LIMIT')
    if args.qualify_cpu:
        print(json.dumps(synthetic_qualification()))
        return
    require(args.assets is not None and args.private is not None and args.primary_private is not None,
            'EXPLICIT_PRIVATE_REPLAY_INPUTS_REQUIRED')
    result = run(args)
    print(json.dumps({key: result[key] for key in ('status', 'planned_units', 'passed_units', 'failed_units', 'optimizer_calls')}))


if __name__ == '__main__':
    main()

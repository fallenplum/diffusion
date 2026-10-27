"""CPU-only F7 recovery certificate: hashes/receipts only, no tensor loading.

The old partial schedules are immutable inputs, never rewritten. An independent
recovered gate may consume the generated certificate after all six replacement
F7 cells and the complete original matrix have passed these guards.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNTIME_SOURCES = ('scripts/bgb_en_f7_runtime_repair.py', 'scripts/bgb_en_f7_determinism_diagnostic.py',
                   'configs/bgb_en/F7_deterministic_runtime_v1.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, code):
    if not bool(condition):
        raise ValueError(code)


def verify_runtime_receipt(cell, complete, *, expected_backend=None, root=ROOT):
    cell = Path(cell)
    path = cell/'RUNTIME_REPAIR_COMPLETE.json'
    runtime = json.loads(path.read_text())
    require(runtime['schema'] == 'bgb_en_F7_runtime_repair_receipt_v1'
            and runtime['status'] == 'COMPLETE_F7_DETERMINISTIC_FRESH_FIT', 'F7_RUNTIME_NOT_COMPLETE')
    require(complete['stage'] == 'R3' and complete['arm'] == 'F7' and complete['epochs'] == 80,
            'F7_RUNTIME_CELL_SCOPE_DIFFERS')
    require(runtime['partition'] == complete['inner_partition'] and runtime['seed'] == complete['seed']
            and runtime['core_lineage'] == complete['lineage'], 'F7_RUNTIME_IDENTITY_LINEAGE_DIFFERS')
    require(runtime['artifact_sha256'] == complete['artifact_sha256'] == sha256(cell/'model.pt')
            and runtime['core_receipt_sha256'] == sha256(cell/'COMPLETE.json'), 'F7_RUNTIME_ARTIFACT_RECEIPT_SHA_DIFFERS')
    require(runtime['actual_new_fits'] == 1 and runtime['actual_new_updates'] == 80
            and runtime['teacher_refits'] == 0 and runtime['optimizer_resumed'] is False
            and runtime['gradient_cache_threshold'] == 2e-5 and runtime['threshold_unchanged'] is True,
            'F7_RUNTIME_TRAINING_CONTRACT_DIFFERS')
    require(set(runtime['sources']) == set(RUNTIME_SOURCES), 'F7_RUNTIME_SOURCE_CLOSURE_DIFFERS')
    require(all(sha256(Path(root)/name) == expected for name, expected in runtime['sources'].items()),
            'F7_RUNTIME_SOURCE_SHA_DIFFERS')
    backend = runtime['backend']
    require(backend['mode'] == 'deterministic' and backend['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            and backend['deterministic_algorithms'] is True and backend['deterministic_warn_only'] is False
            and backend['matmul_allow_tf32'] is False and backend['cudnn_allow_tf32'] is False
            and backend['cudnn_benchmark'] is False and backend['cudnn_deterministic'] is True
            and backend['matmul_precision'] == 'highest', 'F7_RUNTIME_BACKEND_DIFFERS')
    if expected_backend is not None:
        require(backend == expected_backend, 'F7_REPLAY_PROCESS_BACKEND_DIFFERS')
    return dict(runtime_receipt_sha256=sha256(path), runtime_sources=runtime['sources'], backend=backend,
                origin_manifest_sha256=runtime['original_inventory_sha256'],
                prepared_origin_sha256=runtime['prepared_origin_sha256'],
                prepared_receipt_sha256=runtime['prepared_receipt_sha256'])


def matrix_inventory(private, config):
    expected = set()
    for stage in ('R1', 'R2', 'R3'):
        arms = config[stage]['arms']+config[stage].get('additional_real_retraining', [])
        for arm in arms:
            convex = arm in ('P0', 'F0', 'F1', 'O0', 'O1') or arm.startswith('O2_')
            for partition in range(config['prototype']['inner_splits']):
                for seed in ([31] if convex else config['prototype']['neural_seeds']):
                    expected.add((stage, arm, config['prototype']['outer_fold'], partition, seed))
    found = {}
    for path in sorted((Path(private)/'cells').glob('*/COMPLETE.json')):
        complete = json.loads(path.read_text())
        identity = tuple(complete[key] for key in ('stage', 'arm', 'outer_fold', 'inner_partition', 'seed'))
        require(identity in expected and identity not in found, 'CORE_MATRIX_UNEXPECTED_OR_DUPLICATE_CELL')
        require(complete['status'] == 'COMPLETE' and complete['outer_test_evaluated'] is False,
                'CORE_MATRIX_BAD_COMPLETION_SCOPE')
        require(complete['artifact_sha256'] == sha256(path.parent/'model.pt'), 'CORE_MATRIX_ARTIFACT_SHA_DIFFERS')
        found[identity] = (path, complete)
    require(set(found) == expected, 'CORE_MATRIX_INCOMPLETE_AFTER_RECOVERY')
    return found


def certify_recovery(private, origin_manifest, archive_root, *, config_path, root=ROOT):
    from scripts.bgb_en_f7_runtime_repair import original_inventory
    private, origin_manifest, archive_root = Path(private), Path(origin_manifest), Path(archive_root)
    config = json.loads(Path(config_path).read_text())
    lock_path = Path(root)/RUNTIME_SOURCES[-1]
    lock = json.loads(lock_path.read_text())
    origin = json.loads(origin_manifest.read_text())
    require(origin['schema'] == 'bgb_en_F7_original_inventory_v1', 'ORIGINAL_F7_INVENTORY_SCHEMA_DIFFERS')
    require(origin['sources'] == {name: sha256(Path(root)/name) for name in RUNTIME_SOURCES},
            'ORIGINAL_F7_RUNTIME_SOURCE_DIFFERS')
    require(original_inventory(archive_root, lock) == origin['original_cells'], 'ORIGINAL_F7_ARCHIVE_SHA_DIFFERS')
    schedule_paths = {name: private/name for name in ('R3_SCHEDULE.json', 'SCHEDULER_RECEIPT.json')}
    schedule = json.loads(schedule_paths['R3_SCHEDULE.json'].read_text())
    require(schedule['status'] == 'PARTIAL_FAILED_UNITS_CONTINUE_INDEPENDENT_MECHANISMS', 'ORIGINAL_R3_SCHEDULE_NOT_PRESERVED_PARTIAL')
    schedules = {name: dict(sha256=sha256(path), status=json.loads(path.read_text())['status']) for name, path in schedule_paths.items()}
    matrix = matrix_inventory(private, config)
    lineages = [row['core_lineage'] for row in origin['original_cells']]
    lineage = lineages[0]
    require(all(value == lineage for value in lineages)
            and all(complete['lineage'] == lineage for _, complete in matrix.values()), 'RECOVERED_CORE_LINEAGE_DIFFERS')
    require(lineage['protocol_sha256'] == sha256(config_path), 'RECOVERY_CONFIG_SHA_DIFFERS')
    for name, digest in lineage['source'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and sha256(Path(root)/relative) == digest,
                'RECOVERY_CORE_SOURCE_SHA_DIFFERS')
    repaired, runtime_backend = [], None
    for (stage, arm, outer, partition, seed), (path, complete) in sorted(matrix.items()):
        if stage != 'R3' or arm != 'F7':
            continue
        guard = verify_runtime_receipt(path.parent, complete, root=root)
        if runtime_backend is None:
            runtime_backend = guard['backend']
        require(guard['backend'] == runtime_backend, 'F7_RECOVERY_MIXED_RUNTIME_BACKENDS')
        require(guard['origin_manifest_sha256'] == sha256(origin_manifest), 'F7_RUNTIME_ORIGINAL_INVENTORY_DIFFERS')
        prepared = private/'prepared'/f'outer{outer}_inner{partition}.pt'
        require(guard['prepared_origin_sha256'] == sha256(prepared)
                and guard['prepared_receipt_sha256'] == sha256(prepared.with_suffix('.json')), 'F7_RUNTIME_PREPARED_SHA_DIFFERS')
        repaired.append(dict(cell=path.parent.name, inner_partition=partition, seed=seed,
            artifact_sha256=complete['artifact_sha256'], complete_sha256=sha256(path),
            runtime_receipt_sha256=guard['runtime_receipt_sha256'], runtime_sources=guard['runtime_sources']))
    require(len(repaired) == 6 and {(row['inner_partition'], row['seed']) for row in repaired}
            == {(p, s) for p in (0, 1, 2) for s in (31, 47)}, 'RECOVERY_REQUIRES_ALL_SIX_NEW_F7')
    return dict(schema='bgb_en_F7_deterministic_recovery_v1', status='COMPLETE_VERIFIED_F7_RUNTIME_RECOVERY',
        origin_manifest_path=str(origin_manifest.resolve()), origin_manifest_sha256=sha256(origin_manifest),
        archive_root=str(archive_root.resolve()), original_schedule_receipts=schedules,
        sources={**origin['sources'], 'scripts/bgb_en_recovery_audit.py': sha256(Path(root)/'scripts/bgb_en_recovery_audit.py')},
        original_successful_fits=3, original_failed_attempts=3, original_failed_updates=None,
        actual_new_fits=6, actual_new_updates=480, actual_teacher_refits=0, optimizer_resumed=False,
        original_schedules_unchanged=True, originals_archived_and_SHA_verified=True,
        replaced_all_six_F7=True, old_and_new_F7_mixed=False, core_matrix_complete=True,
        core_unique_cells=len(matrix), per_stage_unique_cells={stage: sum(key[0] == stage for key in matrix) for stage in ('R1', 'R2', 'R3')},
        repaired_cells=repaired, core_lineage_sha256=hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest(),
        runtime_backend=runtime_backend,
        original_R3_schedule_still_partial=True, audit_optimizer_steps=0, original_failed_updates_not_fabricated=True)


def verify_recovery_receipt(path, private, *, config_path, root=ROOT):
    receipt = json.loads(Path(path).read_text())
    require(receipt['schema'] == 'bgb_en_F7_deterministic_recovery_v1'
            and receipt['status'] == 'COMPLETE_VERIFIED_F7_RUNTIME_RECOVERY', 'RECOVERY_RECEIPT_NOT_COMPLETE')
    rebuilt = certify_recovery(private, receipt['origin_manifest_path'], receipt['archive_root'], config_path=config_path, root=root)
    require(receipt == rebuilt, 'RECOVERY_RECEIPT_REPLAY_DIFFERS')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('certify', 'verify'), default='verify')
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--output', type=Path, required=True, help='recovery certificate path; existing verify input')
    parser.add_argument('--origin-manifest', type=Path)
    parser.add_argument('--archive-root', type=Path)
    args = parser.parse_args()
    if args.stage == 'certify':
        require(not args.output.exists() and args.origin_manifest is not None and args.archive_root is not None,
                'CERTIFY_REQUIRES_NEW_OUTPUT_AND_ORIGINAL_INPUTS')
        result = certify_recovery(args.private, args.origin_manifest, args.archive_root, config_path=args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    else:
        result = verify_recovery_receipt(args.output, args.private, config_path=args.config)
    print(json.dumps({key: result[key] for key in ('status', 'core_unique_cells', 'actual_new_fits', 'actual_new_updates', 'audit_optimizer_steps')}))


if __name__ == '__main__':
    main()

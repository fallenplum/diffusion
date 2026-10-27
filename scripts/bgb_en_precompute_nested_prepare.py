"""Precompute only the 12 already-authorized outer2--5 inner preparations.

Uses the unchanged bgb_en_run.py prepare CLI and original selection validator.
Two CPU processes at most, two BLAS threads each, GPU hidden. No neural model,
new selection, or outer-risk artifact is read or written by this helper.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bgb_en_run as driver
from scripts.bgb_en_checkpoint_audit import require, sha256, verify_current_source
from scripts.bgb_en_outer import validate_selection_lock

UNITS = [(fold, part) for fold in (2, 3, 4, 5) for part in (0, 1, 2)]
RESERVE_BYTES = 1024**3


def resources():
    root = Path('/sys/fs/cgroup')
    result = {}
    for name in ('memory.max', 'memory.current', 'memory.high', 'cpu.max'):
        path = root / name
        result[name] = path.read_text().strip() if path.is_file() else None
    result['host_loadavg_diagnostic_only'] = list(os.getloadavg())
    result['host_procfs_is_not_container_limit'] = True
    return result


def prep_paths(private, fold, part):
    root = Path(private) / 'prepared' / f'outer{fold}_inner{part}'
    return root.with_suffix('.pt'), root.with_suffix('.json')


def validate_existing(path, receipt_path, lineage):
    require(path.exists() == receipt_path.exists(), 'INTERRUPTED_PREPARE_REQUIRES_REVIEW')
    if not path.exists():
        return None
    row = json.loads(receipt_path.read_text())
    require(row.get('status') == 'COMPLETE' and row.get('lineage') == lineage,
            'EXISTING_PREPARED_LINEAGE_OR_STATUS_MISMATCH')
    require(row.get('sha256') == sha256(path), 'EXISTING_PREPARED_HASH_MISMATCH')
    require(row.get('upstream_optimizer_fits') == 13 and row.get('upstream_preprocessing_fits') == 4,
            'UNEXPECTED_UPSTREAM_PREPARE_ACCOUNTING')
    return row


def remaining_storage_bytes(private, estimated_bytes):
    remaining = 0
    for fold, part in UNITS:
        path, receipt = prep_paths(private, fold, part)
        if path.exists() and receipt.exists():
            continue
        temporary = path.with_suffix(path.suffix + '.tmp')
        already_written = temporary.stat().st_size if temporary.is_file() else 0
        remaining += max(0, estimated_bytes - already_written)
    return remaining


def resource_guard(private, estimated_bytes):
    cg = resources()
    current = int(cg['memory.current']) if cg['memory.current'] is not None else None
    limit = cg['memory.max']
    require(limit not in (None, 'max') and current is not None, 'UNKNOWN_CONTAINER_MEMORY_LIMIT')
    require(int(limit) - current >= 8 * 1024**3, 'CONTAINER_MEMORY_HEADROOM_BELOW_8GIB')
    quota, period = cg['cpu.max'].split()
    require(quota == 'max' or int(quota) / int(period) >= 8, 'CPU_QUOTA_TOO_SMALL_FOR_PARALLEL_PREFETCH')
    free = shutil.disk_usage(Path(private).parent).free
    need = remaining_storage_bytes(private, estimated_bytes)
    require(free - need >= RESERVE_BYTES, 'OVERLAY_PROJECTED_RESERVE_BELOW_1GIB')
    return {'cgroup': cg, 'overlay_free_bytes': free,
            'remaining_estimated_prepare_bytes': need, 'reserve_bytes': RESERVE_BYTES}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--primary-private', type=Path, required=True)
    p.add_argument('--selection-lock', type=Path, required=True)
    p.add_argument('--inner-report', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    p.add_argument('--receipt-directory', type=Path, required=True)
    p.add_argument('--max-workers', type=int, default=2)
    args = p.parse_args()
    require(1 <= args.max_workers <= 2, 'CPU_WORKER_CEILING_IS_TWO')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU_MUST_BE_HIDDEN')
    _, config, raw, member, lineage = driver.load_inputs(args)
    verify_current_source(lineage, config_path=args.config)
    lock = json.loads(args.selection_lock.read_text())
    inner = json.loads(args.inner_report.read_text())
    for fold in (2, 3, 4, 5):
        validate_selection_lock(lock, inner, lineage, config, sha256(args.config), sha256(args.inner_report), fold)
    require('NESTED_INTERACTION' in lock['finalists'], 'NESTED_FINALIST_NOT_AUTHORIZED')
    del raw, member
    source_hashes = {'helper': sha256(__file__), 'original_driver': sha256(ROOT / 'scripts/bgb_en_run.py'),
                     'selection_validator': sha256(ROOT / 'scripts/bgb_en_outer.py'),
                     'selection_lock': sha256(args.selection_lock), 'frozen_inner_report': sha256(args.inner_report),
                     'config': sha256(args.config)}
    reference_sizes = [p.stat().st_size for p in (args.primary_private / 'prepared').glob('outer1_inner*.pt')]
    require(len(reference_sizes) == 3, 'THREE_REFERENCE_PREPARE_BUNDLES_REQUIRED')
    estimate = int(max(reference_sizes) * 1.05)
    args.receipt_directory.mkdir(parents=True, exist_ok=True)
    args.private.mkdir(parents=True, exist_ok=True)
    initial = resource_guard(args.private, estimate)
    existing = {}
    for fold, part in UNITS:
        path, receipt_path = prep_paths(args.private, fold, part)
        existing[fold, part] = validate_existing(path, receipt_path, lineage) is not None
    started = time.monotonic(); rows = []; mutex = threading.Lock()
    receipt_path = args.receipt_directory / 'CPU_PREPARE_RECEIPT.json'
    require(not receipt_path.exists(), 'EXISTING_PREFETCH_RECEIPT_REQUIRES_REVIEW')
    lock_path = args.receipt_directory / 'CPU_PREPARE_LOCK.json'
    require(not lock_path.exists(), 'EXISTING_PREFETCH_LOCK_REQUIRES_REVIEW')
    contract = {'schema': 'bgb_en_nested_prepare_cpu_prefetch_v1', 'source_hashes': source_hashes,
        'scientific_source_lineage': lineage, 'planned_bundles': 12, 'max_CPU_processes': args.max_workers,
        'BLAS_threads_per_process': 2, 'device': 'cpu', 'GPU_hidden': True,
        'planned_upstream_Cox_fits': 156, 'planned_same_teacher_beta_fits': 12,
        'planned_upstream_supervised_optimizations': 168, 'existing_core_plan_not_additional_scientific_models': True,
        'estimated_bundle_bytes_including_5pct_margin': estimate,
        'preexisting_verified_bundles': sum(existing.values()), 'initial_resources': initial,
        'neural_fits': 0, 'outer_risk_files_read': False, 'selection_modified': False,
        'started_utc': datetime.now(timezone.utc).isoformat()}
    driver.write(lock_path, contract)

    def unit(fold, part):
        begin = time.monotonic()
        label = f'outer{fold}_inner{part}'
        if existing[fold, part]:
            return {'outer_fold': fold, 'inner_partition': part, 'status': 'REUSED_VERIFIED_PREEXISTING',
                    'new_bundle': False, 'actual_new_Cox_fits': 0, 'actual_new_beta_fits': 0,
                    'elapsed_seconds': 0., 'prepared_sha256': sha256(prep_paths(args.private, fold, part)[0])}
        with mutex:
            before = resource_guard(args.private, estimate)
            verify_current_source(lineage, config_path=args.config)
            require(sha256(args.selection_lock) == source_hashes['selection_lock']
                    and sha256(args.inner_report) == source_hashes['frozen_inner_report'], 'SELECTION_CHANGED_BEFORE_UNIT')
        command = [sys.executable, str(ROOT / 'scripts/bgb_en_run.py'), '--assets', str(args.assets),
            '--private', str(args.private), '--config', str(args.config), '--stage', 'prepare',
            '--outer-fold', str(fold), '--partition', str(part), '--seed', '31', '--resume', '--device', 'cpu']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        with (args.receipt_directory / (label + '.log')).open('xb') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, 'ORIGINAL_PREPARE_COMMAND_FAILED')
        path, receipt = prep_paths(args.private, fold, part)
        verified = validate_existing(path, receipt, lineage)
        return {'outer_fold': fold, 'inner_partition': part, 'status': 'NEW_PRECOMPUTED_VERIFIED',
                'new_bundle': True, 'actual_new_Cox_fits': 13, 'actual_new_beta_fits': 1,
                'actual_new_preprocessing_fits': verified['upstream_preprocessing_fits'],
                'elapsed_seconds': time.monotonic() - begin, 'prepared_sha256': sha256(path),
                'prepared_bytes': path.stat().st_size, 'receipt_sha256': sha256(receipt),
                'resource_guard_before': before, 'private_exact_command': command}

    def write_receipt(final=False):
        failed = sum(r['status'] == 'FAILED' for r in rows)
        complete = len(rows) == 12 and failed == 0
        value = {'schema': contract['schema'], 'status': 'COMPLETE_PRECOMPUTED_LOCKED_PREPARES' if complete else
            ('PARTIAL_FAILED_NO_AUTORETRY' if final and failed else 'RUNNING'),
            'contract_sha256': sha256(lock_path), 'source_hashes': source_hashes,
            'completed_bundles': sum(r['status'] != 'FAILED' for r in rows), 'failed_bundles': failed,
            'planned_bundles': 12, 'preexisting_bundles': sum(existing.values()),
            'new_completed_bundles': sum(r.get('new_bundle', False) for r in rows),
            'actual_new_Cox_fits': sum(r.get('actual_new_Cox_fits', 0) for r in rows),
            'actual_new_beta_fits': sum(r.get('actual_new_beta_fits', 0) for r in rows),
            'completed_bundle_optimizer_counts_only': True,
            'failed_unit_partial_optimizer_attempts_unknown': failed > 0,
            'elapsed_seconds': time.monotonic() - started, 'units': rows,
            'neural_fits': 0, 'GPU_calls': 0, 'outer_risk_files_read': False,
            'selection_modified': False,
            'selection_sha256_after': sha256(args.selection_lock), 'frozen_inner_sha256_after': sha256(args.inner_report)}
        require(value['selection_sha256_after'] == source_hashes['selection_lock']
                and value['frozen_inner_sha256_after'] == source_hashes['frozen_inner_report'], 'SELECTION_CHANGED_DURING_PREFETCH')
        driver.write(receipt_path, value)
        return value

    write_receipt()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(unit, *key): key for key in UNITS}
        for future in as_completed(futures):
            fold, part = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {'outer_fold': fold, 'inner_partition': part, 'status': 'FAILED',
                    'error_type': type(exc).__name__, 'public_error_code': str(exc) if isinstance(exc, ValueError) else 'PREPARE_EXCEPTION',
                    'new_bundle': False}
            rows.append(row); write_receipt()
            print(json.dumps({key: row[key] for key in ('outer_fold', 'inner_partition', 'status')}), flush=True)
    final = write_receipt(final=True)
    print(json.dumps({key: final[key] for key in ('status', 'completed_bundles', 'failed_bundles', 'actual_new_Cox_fits', 'actual_new_beta_fits')}), flush=True)


if __name__ == '__main__':
    main()

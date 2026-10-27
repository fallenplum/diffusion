"""Publish anonymous metadata for the completed, original nested preparations.

Zero fits/forwards; validates 12 prepared hashes and teacher-local member/fit
metadata. Does not inspect full_risks, CP/CS/RNA/base or outcome predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts import bgb_en_run as driver
from scripts.bgb_en_checkpoint_audit import require, sha256, verify_current_source, verify_teacher_lineage
from scripts.bgb_en_precompute_nested_prepare import UNITS, prep_paths, validate_existing, resources


def run(args):
    private_receipt = args.receipt_directory / 'CPU_PREPARE_RECEIPT.json'
    contract_path = args.receipt_directory / 'CPU_PREPARE_LOCK.json'
    row = json.loads(private_receipt.read_text()); contract = json.loads(contract_path.read_text())
    require(row['status'] == 'COMPLETE_PRECOMPUTED_LOCKED_PREPARES' and row['completed_bundles'] == 12
            and row['failed_bundles'] == 0, 'PRECOMPUTATION_NOT_COMPLETE')
    require(row['contract_sha256'] == sha256(contract_path), 'PREFETCH_CONTRACT_HASH_MISMATCH')
    lineage = contract['scientific_source_lineage']
    verify_current_source(lineage, config_path=ROOT / 'configs/bgb_en/prototype_v1.json')
    require(contract['source_hashes']['helper'] == sha256(ROOT / 'scripts/bgb_en_precompute_nested_prepare.py')
            and contract['source_hashes']['selection_validator'] == sha256(ROOT / 'scripts/bgb_en_outer.py'),
            'PREFETCH_SOURCE_CHANGED')
    require(row['selection_sha256_after'] == contract['source_hashes']['selection_lock']
            and row['frozen_inner_sha256_after'] == contract['source_hashes']['frozen_inner_report'],
            'PREFETCH_SELECTION_HASH_MISMATCH')
    selection = ROOT / 'reports/bgb_en/OUTER_SELECTION_LOCK.json'
    inner = ROOT / 'reports/bgb_en/INNER_REPORT_FOR_SELECTION.json'
    require(sha256(selection) == contract['source_hashes']['selection_lock']
            and sha256(inner) == contract['source_hashes']['frozen_inner_report'], 'FROZEN_SELECTION_CHANGED')
    found = {(r['outer_fold'], r['inner_partition']): r for r in row['units']}
    require(len(found) == len(row['units']) == 12 and set(found) == set(UNITS), 'PREFETCH_UNIT_SET_MISMATCH')
    units = []
    for fold, part in UNITS:
        item = found[fold, part]
        path, receipt = prep_paths(args.private, fold, part)
        public = validate_existing(path, receipt, lineage)
        require(item['prepared_sha256'] == public['sha256'], 'SCHEDULER_PREPARED_HASH_MISMATCH')
        prepared = torch.load(path, map_location='cpu', weights_only=False)
        require(prepared['lineage'] == lineage, 'PREPARED_LINEAGE_MISMATCH')
        teacher = verify_teacher_lineage(prepared, 512)
        # No saved risk/output/clinical prediction array is accessed here.
        units.append({key: item[key] for key in (
            'outer_fold', 'inner_partition', 'status', 'new_bundle', 'actual_new_Cox_fits',
            'actual_new_beta_fits', 'elapsed_seconds', 'prepared_sha256')})
        units[-1].update(prepared_bytes=path.stat().st_size, receipt_sha256=sha256(receipt),
                         teacher_lineage_audit=teacher)
        del prepared
    output = {'schema': 'bgb_en_nested_prepare_cpu_prefetch_audit_v1',
        'status': 'PASS_12_LOCKED_PREPARED_BUNDLES', 'prepared_bundles_verified': 12,
        'preexisting_bundles': row['preexisting_bundles'], 'new_bundles': row['new_completed_bundles'],
        'original_planned_upstream_Cox_optimizations': row['actual_new_Cox_fits'],
        'original_planned_same_teacher_beta_optimizations': row['actual_new_beta_fits'],
        'original_planned_total_supervised_optimizations': row['actual_new_Cox_fits'] + row['actual_new_beta_fits'],
        'same_fits_as_later_nested_schedule_account_once': True,
        'CPU_scheduler_wall_seconds': row['elapsed_seconds'],
        'sum_unit_wall_seconds_not_CPU_time': sum(r['elapsed_seconds'] for r in units),
        'prepared_total_bytes': sum(r['prepared_bytes'] for r in units), 'units': units,
        'source_hashes': contract['source_hashes'], 'audit_source_sha256': sha256(__file__),
        'private_scheduler_receipt_sha256': sha256(private_receipt), 'private_contract_sha256': sha256(contract_path),
        'initial_resources': contract['initial_resources'],
        'final_resources': {'cgroup': resources(), 'overlay_free_bytes': shutil.disk_usage(args.private).free},
        'max_CPU_processes': contract['max_CPU_processes'], 'BLAS_threads_per_process': 2,
        'actual_audit_optimizer_calls': 0, 'actual_audit_model_forwards': 0,
        'neural_fits': 0, 'GPU_calls': 0, 'outer_risk_arrays_consumed': False,
        'patient_or_gene_indices_exported': False,
        'selection_sha256_before': contract['source_hashes']['selection_lock'], 'selection_sha256_after': sha256(selection),
        'frozen_inner_sha256_before': contract['source_hashes']['frozen_inner_report'], 'frozen_inner_sha256_after': sha256(inner),
        'reproducible_unit_command': 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 python scripts/bgb_en_run.py --assets "$ASSETS" --private "$NESTED_PRIVATE" --config configs/bgb_en/prototype_v1.json --stage prepare --outer-fold "$FOLD" --partition "$PARTITION" --seed 31 --resume --device cpu'}
    driver.write(args.output, output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private', required=True, type=Path)
    p.add_argument('--receipt-directory', required=True, type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en/R4_CPU_PREPARE_PREFETCH_AUDIT.json')
    args = p.parse_args(); torch.set_num_threads(2); result = run(args)
    print(json.dumps({key: result[key] for key in ('status', 'prepared_bundles_verified', 'CPU_scheduler_wall_seconds',
        'prepared_total_bytes', 'original_planned_total_supervised_optimizations')}))


if __name__ == '__main__':
    main()

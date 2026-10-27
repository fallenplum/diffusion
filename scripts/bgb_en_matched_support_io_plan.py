"""Derive an I/O-only job plan with one public-lock view directory per unit.

No adapter, monkey patch, subprocess, model forward or optimizer is used. The
original scientific plan, null/source locks and immutable launch lock remain
byte-for-byte unchanged. Only an existing --output CLI destination is added.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bgb_en_run as driver
from scripts.bgb_en_batch import validate_plan
from scripts.bgb_en_checkpoint_audit import require, sha256


def derive_plan(scientific, private, scientific_sha256):
    original = validate_plan(scientific)
    require(len(original) == 12, 'EXPECTED_TWELVE_SCIENTIFIC_JOBS')
    base = Path(private).resolve() / 'unit_public_views'
    runtime = copy.deepcopy(scientific)
    outputs = set(); audit = []
    for old, task in zip(original, runtime['tasks']):
        require(old['script'] == 'scripts/bgb_en_matched_support.py', 'UNEXPECTED_SCIENTIFIC_DRIVER')
        require('--output' not in old['args'] and not any(x.startswith('--output=') for x in old['args']),
                'SCIENTIFIC_OUTPUT_ALREADY_EXPLICIT')
        destination = (base / old['label']).resolve()
        require(destination.parent == base and destination != base, 'UNSAFE_PER_UNIT_OUTPUT')
        require(destination not in outputs, 'DUPLICATED_PER_UNIT_OUTPUT')
        outputs.add(destination)
        task['args'] += ['--output', str(destination)]
        require(task['args'][:-2] == old['args'] and task['args'][-2] == '--output'
                and task['label'] == old['label'] and task['script'] == old['script'], 'NON_IO_ARGUMENT_CHANGED')
        audit.append({'label': old['label'], 'scientific_argv_unchanged_after_removing_output': True,
            'output_directory_unique_and_private': True,
            'output_directory_sha256': hashlib.sha256(str(destination).encode()).hexdigest()})
    runtime['io_runtime_parent_plan_sha256'] = scientific_sha256
    runtime['io_runtime_derivation'] = 'append_existing_output_flag_with_unique_private_unit_public_view_only'
    validate_plan(runtime)
    return runtime, audit


def write_once(path, value):
    if path.exists():
        require(json.loads(path.read_text()) == value, 'EXISTING_IO_RUNTIME_LOCK_CHANGED')
    else:
        driver.write(path, value)


def build(args):
    paths = {name: args.private / name for name in ('JOB_PLAN.json', 'LOCK.json', 'LAUNCH_LOCK.json')}
    before = {name: sha256(path) for name, path in paths.items()}
    source_lock = json.loads(paths['LOCK.json'].read_text())
    launch = json.loads(paths['LAUNCH_LOCK.json'].read_text())
    require(before['JOB_PLAN.json'] == launch['private_job_plan_sha256']
            and before['LOCK.json'] == launch['source_lock_sha256'], 'SCIENTIFIC_PLAN_OR_LOCK_CHANGED')
    for relative, expected in source_lock['lineage']['auxiliary_source'].items():
        require(sha256(ROOT / relative) == expected, 'FROZEN_SCIENTIFIC_SOURCE_CHANGED')
    require(not list((args.private / 'cells').glob('**/START.json')), 'TRAINING_ALREADY_STARTED_BEFORE_IO_LOCK')
    scientific = json.loads(paths['JOB_PLAN.json'].read_text())
    runtime, unit_audit = derive_plan(scientific, args.private, before['JOB_PLAN.json'])
    runtime_path = args.private / 'RUNTIME_IO_PLAN.json'
    write_once(runtime_path, runtime)
    selection_paths = [ROOT / 'reports/bgb_en' / name for name in launch['selection_sha256']]
    selection = {path.name: sha256(path) for path in selection_paths}
    require(selection == launch['selection_sha256'], 'SELECTION_LOCK_CHANGED')
    lock = {'schema': 'bgb_en_matched_support_io_runtime_v1', 'status': 'LOCKED_BEFORE_ADDITIONAL_FITS',
        'scientific_plan_sha256': before['JOB_PLAN.json'], 'runtime_io_plan_sha256': sha256(runtime_path),
        'scientific_source_lock_sha256': before['LOCK.json'], 'immutable_launch_lock_sha256': before['LAUNCH_LOCK.json'],
        'runtime_plan_builder_sha256': sha256(__file__), 'scientific_driver_sha256': sha256(ROOT / 'scripts/bgb_en_matched_support.py'),
        'selection_sha256': selection, 'planned_units': len(unit_audit), 'unit_argument_audits': unit_audit,
        'only_existing_output_destination_added': True, 'same_labels_and_scientific_arguments': True,
        'canonical_public_null_lock_will_not_be_rewritten_by_jobs': True,
        'one_output_directory_per_unit': True, 'adapter_used': False, 'monkey_patch_used': False,
        'scientific_original_files_unchanged': before == {name: sha256(path) for name, path in paths.items()},
        'actual_new_optimizer_fits': 0, 'actual_model_forwards': 0, 'GPU_calls': 0,
        'R4_selection_modified': False, 'outer_performance_read': False}
    require(lock['scientific_original_files_unchanged'], 'SCIENTIFIC_LOCK_CHANGED_DURING_DERIVATION')
    write_once(args.private / 'IO_RUNTIME_LOCK.json', lock)
    args.output.mkdir(parents=True, exist_ok=True)
    write_once(args.output / 'MATCHED_SUPPORT_IO_RUNTIME_LOCK.json', lock)
    return lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private', required=True, type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en')
    result = build(p.parse_args())
    print(json.dumps({key: result[key] for key in ('status', 'planned_units', 'runtime_io_plan_sha256',
                                                 'actual_new_optimizer_fits', 'GPU_calls')}))


if __name__ == '__main__':
    main()

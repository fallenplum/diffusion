"""Create an explicit 12-job batch plan; never launch a subprocess or optimizer.

The already-frozen null-builder/source lock is preserved. This companion launch
lock binds its hash, the existing R4 selection hashes, this planner and exact
batch commands. Private paths stay in JOB_PLAN.json, not the anonymous plan.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import bgb_en_run as driver
from scripts.bgb_en_batch import validate_plan
from scripts.bgb_en_checkpoint_audit import require, sha256, verify_current_source
from scripts.bgb_en_matched_support import SPEC_PATH, base_model


def verify_constructor_and_gamma(config):
    evidence = []
    for stage, variant, arm, gamma in [('R2', 'O4_MATCHED_SUPPORT', 'O4', .3),
                                       ('R3', 'F5_MATCHED_SUPPORT', 'F5', .1)]:
        require(config[stage]['gamma'] == gamma, 'CORE_GAMMA_DRIFT')
        calls = []
        def spy(*args, **kwargs):
            calls.append((args, kwargs)); return None
        member = object()
        base_model(spy, stage, variant, {}, member, None, 'cpu', gamma=gamma)
        require(len(calls) == 1 and calls[0][0][1] == arm and calls[0][0][3] is member
                and calls[0][1]['gamma'] == gamma, 'MODEL_DISPATCH_OR_GAMMA_MISMATCH')
        evidence.append({'variant': variant, 'stage': stage, 'actual_base_arm': arm,
                         'gamma': gamma, 'same_membership_object_forwarded': True,
                         'original_O9_or_SHUFFLED_constructor_invoked': False})
    # The reused driver chooses stage gamma at the sole model-construction call.
    tree = ast.parse(inspect.getsource(driver.run_unit))
    constructors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'MechanismModel']
    expected_gamma = ast.parse("config[args.stage].get('gamma', .1)", mode='eval').body
    require(len(constructors) == 1 and any(keyword.arg == 'gamma' and
            ast.dump(keyword.value) == ast.dump(expected_gamma) for keyword in constructors[0].keywords),
            'TRAINER_GAMMA_PATH_CHANGED')
    return evidence


def write_once(path, value):
    path = Path(path)
    if path.exists():
        require(json.loads(path.read_text()) == value, 'EXISTING_EXECUTION_PLAN_CHANGED')
    else:
        driver.write(path, value)


def build(args):
    source_lock_path = args.private / 'LOCK.json'
    lock = json.loads(source_lock_path.read_text())
    spec = json.loads(SPEC_PATH.read_text())
    require(lock.get('status') == 'LOCKED_BEFORE_AUXILIARY_FITS'
            and lock['lineage']['specification'] == spec, 'NULL_SOURCE_LOCK_NOT_CURRENT')
    require(lock['actual_optimizer_fits_at_lock'] == 0 and lock['actual_teacher_refits'] == 0,
            'SOURCE_LOCK_ALREADY_FITTED')
    require(not list((args.private / 'cells').glob('**/START.json')), 'ADDITIONAL_FITS_ALREADY_STARTED')
    for relative, expected in lock['lineage']['auxiliary_source'].items():
        require(sha256(ROOT / relative) == expected, 'MATCHED_SUPPORT_SOURCE_CHANGED')
    core_config = ROOT / 'configs/bgb_en/prototype_v1.json'
    verify_current_source(lock['lineage']['core_lineage'], config_path=core_config)
    config = json.loads(core_config.read_text())
    dispatch = verify_constructor_and_gamma(config)
    selection_files = [args.inner_report, args.selection_lock]
    selection = {path.name: sha256(path) for path in selection_files}
    require(json.loads(args.selection_lock.read_text()).get('outer_test_evaluated') is False,
            'SELECTION_LOCK_NOT_SEALED')
    rows, tasks = [], []
    for variant, definition in spec['variants'].items():
        for partition in spec['partitions']:
            for seed in spec['seeds']:
                label = f'{variant}_p{partition}_seed{seed}'
                tasks.append({'label': label, 'script': 'scripts/bgb_en_matched_support.py', 'args': [
                    '--stage', 'train', '--assets', str(args.assets), '--primary-private', str(args.primary_private),
                    '--private', str(args.private), '--variant', variant,
                    '--partition', str(partition), '--seed', str(seed), '--device', 'cuda']})
                rows.append({'label': label, 'variant': variant, 'stage': definition['stage'],
                             'base_arm': definition['base_arm'], 'inner_partition': partition,
                             'seed': seed, 'epochs': 80, 'teacher_refits': 0})
    require(len(tasks) == 12, 'EXPECTED_TWELVE_UNITS')
    common = {'status': 'LOCKED_BEFORE_ADDITIONAL_FITS', 'outer_test_evaluated': False,
              'post_hoc_exploratory': True, 'R4_selection_modified': False,
              'source_lock_sha256': sha256(source_lock_path), 'selection_sha256': selection,
              'planner_sha256': sha256(__file__), 'planned_neural_fits': 12, 'planned_teacher_refits': 0,
              'prepared_reused_readonly': True, 'outer_performance_read_by_this_audit': False,
              'schedule': {name: config['training'][name] for name in
                           ('epochs', 'evaluation_epochs', 'batch_size', 'learning_rate', 'optimizer',
                            'full_riskset_gradient_cache', 'parameter_L2', 'risk_magnitude_penalty')},
              'constructor_gamma_audit': dispatch}
    plan = {**common, 'tasks': tasks}
    validate_plan(plan)
    private_plan = args.private / 'JOB_PLAN.json'
    write_once(private_plan, plan)
    public = {**common, 'schema': 'bgb_en_matched_support_public_plan_v1', 'units': rows,
              'private_job_plan_sha256': sha256(private_plan),
              'raw_gene_permutation_indices_exported': False, 'GPU_training_launched': False}
    args.output.mkdir(parents=True, exist_ok=True)
    public_path = args.output / 'MATCHED_SUPPORT_PUBLIC_PLAN.json'
    write_once(public_path, public)
    launch = {**common, 'schema': 'bgb_en_matched_support_launch_lock_v1',
              'private_job_plan_sha256': sha256(private_plan), 'public_plan_sha256': sha256(public_path),
              'source_lock_preserved_unchanged': True,
              'additional_optimizer_fits_at_launch_lock': 0, 'GPU_calls': 0}
    write_once(args.private / 'LAUNCH_LOCK.json', launch)
    write_once(args.output / 'MATCHED_SUPPORT_LAUNCH_LOCK.json', launch)
    require({path.name: sha256(path) for path in selection_files} == selection, 'SELECTION_FILE_CHANGED_DURING_PLAN')
    return public


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--primary-private', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--inner-report', type=Path, default=ROOT / 'reports/bgb_en/INNER_REPORT_FOR_SELECTION.json')
    p.add_argument('--selection-lock', type=Path, default=ROOT / 'reports/bgb_en/OUTER_SELECTION_LOCK.json')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en')
    args = p.parse_args(); result = build(args)
    print(json.dumps({'status': result['status'], 'planned_neural_fits': 12,
                      'actual_optimizer_fits': 0, 'GPU_training_launched': False}))


if __name__ == '__main__':
    main()

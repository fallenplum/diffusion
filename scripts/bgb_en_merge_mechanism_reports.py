"""Merge verified private R3 mechanism JSONs; no model/tensor loading or fit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_mechanism_replay import grouped_mechanisms
from scripts.bgb_en_recovery_audit import sha256, require, verify_runtime_receipt

ARMS = ('F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F5_WRONG_QUERY', 'F5_SHUFFLED_MEMBERSHIP')
AUDIT_SOURCES = ('src/bgb_en/mechanism_audit.py', 'scripts/bgb_en_mechanism_replay.py',
                 'scripts/bgb_en_anchor_cast_audit.py')


def identity(row):
    return tuple(row[key] for key in ('arm', 'outer_fold', 'partition', 'seed'))


def label(key):
    return f'R3_outer{key[1]}_inner{key[2]}_{key[0]}_seed{key[3]}'


def merge_reports(input_roots, private, *, config_path, root=ROOT):
    expected = {(arm, 1, part, seed) for arm in ARMS for part in (0, 1, 2) for seed in (31, 47)}
    sources = {name: sha256(Path(root)/name) for name in AUDIT_SOURCES}
    config_sha = sha256(config_path)
    results, inputs, inventories, backend_by_arm, seen = [], [], [], {}, set()
    lineage = None
    for ordinal, input_root in enumerate(map(Path, input_roots), 1):
        inventory_path = input_root/'INVENTORY.json'
        inventory = json.loads(inventory_path.read_text())
        require(inventory['audit_sources'] == sources and inventory['optimizer_updates'] == 0
                and inventory['actual_weights_changed'] is False, 'MECHANISM_INVENTORY_SOURCE_OR_ZERO_FIT_DIFFERS')
        require(inventory['status'] == 'COMPLETE_REQUESTED_MECHANISM_REPLAYS', 'MECHANISM_ROOT_REPLAY_NOT_COMPLETE')
        wanted = {identity(row) for row in inventory['inventory']}
        files = sorted(input_root.glob('R3_*.json'))
        require(len(files) == inventory['completed_mechanism_cells'] == len(wanted)
                and inventory['inventoried_cells'] == len(wanted), 'MECHANISM_ROOT_INVENTORY_COUNT_DIFFERS')
        root_seen = set()
        for path in files:
            result = json.loads(path.read_text())
            key = identity(result['identity'])
            require(key in expected and key in wanted and path.stem == label(key), 'MECHANISM_IDENTITY_OR_FILENAME_DIFFERS')
            require(key not in seen, 'DUPLICATE_MECHANISM_IDENTITY')
            seen.add(key); root_seen.add(key)
            require(result['status'] == 'PASS_FINAL80_MECHANISM_REPLAY' and result['optimizer_updates'] == 0,
                    'MECHANISM_REPLAY_NOT_PASS_OR_ZERO_FIT')
            contract = result['contract']
            require(contract['audit_sources'] == sources, 'MECHANISM_CELL_AUDIT_SOURCE_DIFFERS')
            limit, count, full = contract['patient_limit'], result['diagnostic_validation_patients'], result['full_validation_patients']
            require(isinstance(limit, int) and 1 <= limit <= 8 and count == min(limit, full)
                    and isinstance(full, int) and full > 0, 'MECHANISM_FIXED_SUBSET_SCOPE_DIFFERS')
            require(0 <= result['sampled_final80_risk_replay_max_abs'] <= 2e-5, 'MECHANISM_REPLAY_RISK_GATE_FAILED')
            for probe in ('interaction', 'OtoP_drop_one', 'PtoO_drop_one'):
                gradient = result['loss_chain_gradients'][probe]
                require(gradient['regularization_included'] is False and gradient['pooled_main_effects_included'] is False,
                        'MECHANISM_PURE_INTERACTION_GRADIENT_SCOPE_DIFFERS')
            cell = Path(private)/'cells'/label(key)
            complete_path = cell/'COMPLETE.json'
            complete = json.loads(complete_path.read_text())
            require((complete['arm'], complete['outer_fold'], complete['inner_partition'], complete['seed']) == key
                    and complete['stage'] == 'R3' and complete['epochs'] == 80 and complete['status'] == 'COMPLETE',
                    'MECHANISM_TRAINED_CELL_IDENTITY_DIFFERS')
            require(contract['artifact_sha256'] == complete['artifact_sha256'] == sha256(cell/'model.pt'),
                    'MECHANISM_TRAINED_ARTIFACT_SHA_DIFFERS')
            if lineage is None:
                lineage = complete['lineage']
                require(lineage['protocol_sha256'] == config_sha, 'MECHANISM_CORE_PROTOCOL_SHA_DIFFERS')
                for name, digest in lineage['source'].items():
                    relative = Path(name)
                    require(not relative.is_absolute() and '..' not in relative.parts
                            and sha256(Path(root)/relative) == digest, 'MECHANISM_CORE_SOURCE_SHA_DIFFERS')
            require(complete['lineage'] == lineage, 'MECHANISM_CORE_LINEAGE_MIXED')
            prepared = Path(private)/'prepared'/f'outer{key[1]}_inner{key[2]}.pt'
            prepared_receipt = json.loads(prepared.with_suffix('.json').read_text())
            require(contract['prepared_sha256'] == prepared_receipt['sha256'] == sha256(prepared)
                    and prepared_receipt['lineage'] == lineage, 'MECHANISM_PREPARED_SHA_LINEAGE_DIFFERS')
            if key[0] == 'F7':
                runtime = verify_runtime_receipt(cell, complete, root=root)
                require(inventory['F7_runtime_repair'] is True
                        and inventory['explicit_runtime_backend'] == runtime['backend']
                        and contract['runtime_repair_receipt_sha256'] == runtime['runtime_receipt_sha256']
                        and contract['runtime_backend'] == runtime['backend']
                        and str(contract['device']).startswith('cuda'), 'MECHANISM_F7_RUNTIME_BACKEND_DIFFERS')
                backend = dict(group='F7_deterministic_runtime_repair', device=contract['device'], settings=runtime['backend'])
            else:
                require(inventory['F7_runtime_repair'] is False and inventory['explicit_runtime_backend'] is None
                        and 'runtime_repair_receipt_sha256' not in contract and 'runtime_backend' not in contract,
                        'NON_F7_MECHANISM_BACKEND_WAS_CHANGED')
                backend = dict(group='inherited_non_F7_runtime', device=contract['device'], settings=None,
                    limitation='full backend flags were not recorded by the original inherited mechanism runner')
            if key[0] in backend_by_arm:
                require(backend_by_arm[key[0]] == backend, 'SAME_ARM_DIFFERENT_RUNTIME_MUST_NOT_BE_POOLED')
            backend_by_arm[key[0]] = backend
            results.append(result)
            inputs.append(dict(identity=result['identity'], source_root_ordinal=ordinal,
                private_replay_basename=path.name, replay_json_sha256=sha256(path),
                artifact_sha256=contract['artifact_sha256'], complete_sha256=sha256(complete_path),
                prepared_sha256=contract['prepared_sha256'], runtime_group=backend['group']))
        require(root_seen == wanted, 'MECHANISM_ROOT_INVENTORY_IDENTITY_DIFFERS')
        inventories.append(dict(source_root_ordinal=ordinal, inventory_sha256=sha256(inventory_path),
                                cells=len(files), runtime_repair=inventory['F7_runtime_repair']))
    require(lineage is not None, 'EMPTY_MECHANISM_INPUTS')
    results.sort(key=lambda row: identity(row['identity']))
    inputs.sort(key=lambda row: identity(row['identity']))
    def grouped(fields):
        values = grouped_mechanisms(results, fields)
        for value in values:
            value['runtime'] = backend_by_arm[value['arm']]
        return values
    missing = [dict(arm=key[0], outer_fold=key[1], partition=key[2], seed=key[3]) for key in sorted(expected-seen)]
    report = dict(schema='bgb_en_merged_R3_mechanism_replay_summary_v1',
        status='COMPLETE_VERIFIED_66_MECHANISM_REPLAYS' if not missing else 'PARTIAL_MISSING_MECHANISM_REPLAYS',
        expected_cells=66, completed_mechanism_cells=len(results), missing_cells=missing,
        reporter_forward_calls=0, reporter_tensor_loads=0, optimizer_updates=0, actual_weights_changed=False,
        merge_script_sha256=sha256(Path(root)/'scripts/bgb_en_merge_mechanism_reports.py'), audit_sources=sources,
        core_lineage_sha256=hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest(),
        input_inventories=inventories, verified_private_input_sha256=inputs,
        runtime_groups=[dict(group=group, cells=sum(value['runtime_group'] == group for value in inputs))
                        for group in sorted({value['runtime_group'] for value in inputs})],
        per_arm=grouped(('arm',)), per_arm_seed=grouped(('arm', 'seed')),
        per_arm_partition=grouped(('arm', 'partition')),
        aggregation_boundary='descriptive cell summaries of each fixed first <=8 validation-row mechanism subset, not independent patient cohorts or full-cohort gradients',
        gradient_scope='frozen full-validation Efron dLoss/dr, network backpropagation through only fixed first <=8 validation rows; regularization and pooled-main-effect objectives excluded',
        backend_boundary='F7 deterministic and non-F7 inherited runtimes are explicitly separate; no cross-backend overall mean and no pooling the same arm across runtimes',
        full_validation_boundary='saved full-validation performance remains in private per-cell source reports; subset gradients are not extrapolated to it',
        patient_rows_or_identifiers_exported=False, private_paths_exported=False)
    return report, [(Path(input_roots[row['source_root_ordinal']-1])/row['private_replay_basename'], row) for row in inputs]


def copy_private_reports(combined_root, sources, *, input_roots, public_output, script_sha):
    target = Path(combined_root).resolve()
    require(not target.is_relative_to((ROOT/'reports').resolve())
            and not target.is_relative_to(Path(public_output).resolve().parent), 'COMBINED_MECHANISM_ROOT_MUST_BE_PRIVATE')
    for source in map(lambda value: Path(value).resolve(), input_roots):
        require(not target.is_relative_to(source) and not source.is_relative_to(target), 'COMBINED_ROOT_CANNOT_MUTATE_AN_INPUT_ROOT')
    wanted = {path.name: row['replay_json_sha256'] for path, row in sources}
    manifest = dict(schema='bgb_en_private_mechanism_copy_manifest_v1', merge_script_sha256=script_sha, source_json_sha256=wanted)
    manifest_path = target/'COMBINED_SOURCE_MANIFEST.json'
    if manifest_path.exists():
        require(json.loads(manifest_path.read_text()) == manifest, 'EXISTING_COMBINED_MANIFEST_DIFFERS')
    require(all(path.name in wanted for path in target.glob('R3_*.json')), 'EXTRANEOUS_PRIVATE_COMBINED_REPLAY')
    target.mkdir(parents=True, exist_ok=True)
    for source, row in sources:
        destination = target/source.name
        if destination.exists():
            require(sha256(destination) == row['replay_json_sha256'], 'EXISTING_PRIVATE_COPY_SHA_DIFFERS')
        else:
            shutil.copyfile(source, destination)
        require(sha256(destination) == row['replay_json_sha256'], 'PRIVATE_MECHANISM_COPY_SHA_DIFFERS')
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', action='append', type=Path, required=True)
    parser.add_argument('--private', type=Path, help='formal root; inferred only when all input roots share one parent')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/bgb_en/R3_MECHANISM_REPLAY_SUMMARY.json')
    parser.add_argument('--combined-root', type=Path, help='optional private byte-identical copies for downstream single-root budget audit')
    args = parser.parse_args()
    if args.private is None:
        parents = {path.resolve().parent for path in args.input_root}
        require(len(parents) == 1, 'SUPPLY_EXPLICIT_PRIVATE_ROOT_FOR_DIFFERENT_PARENTS')
        args.private = parents.pop()
    report, sources = merge_reports(args.input_root, args.private, config_path=args.config)
    if args.combined_root is not None:
        copy_private_reports(args.combined_root, sources, input_roots=args.input_root,
                             public_output=args.output, script_sha=report['merge_script_sha256'])
        report['private_combined_copy_verified'] = True
    else:
        report['private_combined_copy_verified'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'completed_mechanism_cells', 'reporter_forward_calls', 'optimizer_updates')}))


if __name__ == '__main__':
    main()

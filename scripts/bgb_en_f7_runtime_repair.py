"""Explicit F7-only runtime lock; default qualification performs no fitting.

Archive migration is owned by the caller and is never performed here. Train
requires an immutable original inventory and a matching deterministic CUDA
forward qualification. Frozen core sources and the 2e-5 guard are unmodified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_f7_determinism_diagnostic import configure_backend, sha


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def original_inventory(root, lock):
    rows = []
    for partition in lock['inner_partitions']:
        for seed in lock['seeds']:
            name = f'R3_outer1_inner{partition}_F7_seed{seed}'
            folder = root/name
            require(folder.is_dir(), 'Original F7 folder absent: '+name)
            failed, complete = folder/'FAILED.json', folder/'COMPLETE.json'
            expected_failure = name in lock['expected_original_failed']
            require(failed.exists() == expected_failure and complete.exists() != expected_failure,
                    'Original F7 success/failure inventory differs: '+name)
            receipt = json.loads((failed if expected_failure else complete).read_text())
            if expected_failure:
                require(receipt['type'] == 'RuntimeError' and receipt['message'] == 'Gradient-cache forward was not deterministic',
                        'Unexpected original failure type/message')
            else:
                require(receipt['arm'] == 'F7' and receipt['epochs'] == 80 and receipt['artifact_sha256'] == sha(folder/'model.pt'),
                        'Original successful model receipt/artifact differs')
            rows.append(dict(cell=name, original_status='FAILED' if expected_failure else 'COMPLETE',
                core_lineage=receipt['lineage'], files={str(path.relative_to(folder)).replace('\\', '/'): sha(path)
                    for path in sorted(folder.rglob('*')) if path.is_file()}))
    require(len(rows) == 6 and sum(row['original_status'] == 'FAILED' for row in rows) == 3, 'Expected all original six F7 cells')
    return rows


def validate_forward_qualification(value, partition, seed, lineage, source_sha):
    require(value['status'] == 'PASS_SAMPLED_READONLY_FORWARD' and value['optimizer_updates'] == 0,
            'No successful zero-optimizer forward qualification')
    require(value['arm'] == 'F7' and value['partition'] == partition and value['seed'] == seed,
            'Forward qualification cell differs')
    require(str(value['device']).startswith('cuda') and value['state'] == 'initial',
            'Fresh-seed CUDA forward qualification is required; CPU does not qualify CUDA')
    require(value['core_lineage'] == lineage and value['threshold'] == 2e-5
            and value['all_comparisons_within_threshold'] is True, 'Forward qualification source/threshold differs')
    require(value['diagnostic_sources']['scripts/bgb_en_f7_determinism_diagnostic.py'] == source_sha,
            'Forward qualification diagnostic source differs')
    settings = value['backend']
    require(settings['mode'] == 'deterministic' and settings['deterministic_algorithms'] is True
            and settings['deterministic_warn_only'] is False and settings['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            and settings['matmul_allow_tf32'] is False and settings['cudnn_allow_tf32'] is False,
            'Forward qualification backend differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('qualify', 'train'), default='qualify')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True, help='existing prepared root; new cell must be absent before train')
    parser.add_argument('--original-root', type=Path, required=True, help='all six original folders before or after caller archives them')
    parser.add_argument('--origin-manifest', type=Path, required=True)
    parser.add_argument('--lock', type=Path, default=ROOT/'configs/bgb_en/F7_deterministic_runtime_v1.json')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--partition', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--seed', type=int, choices=(31, 47), default=31)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Runtime output exists; no implicit overwrite or resume')
    lock = json.loads(args.lock.read_text())
    require(lock['arm'] == 'F7' and lock['required_new_fits'] == 6 and lock['gradient_cache_threshold'] == 2e-5,
            'Unexpected runtime repair scope')
    # No module imported above imports torch. cuBLAS workspace is therefore set
    # before torch import, not merely after creation of a CUDA handle.
    backend = configure_backend('deterministic')
    import torch
    from scripts import bgb_en_run as driver
    torch.set_num_threads(2)
    assets, config, raw, member, lineage = driver.load_inputs(args)
    require(config['training']['epochs'] == 80 and config['training']['evaluation_epochs'] == [20, 40, 80]
            and config['training']['batch_size'] == 8 and config['training']['learning_rate'] == .001
            and config['training']['deterministic_replay_max_error'] == 2e-5, 'Frozen training protocol differs')
    inventory = original_inventory(args.original_root, lock)
    require(all(row['core_lineage'] == lineage for row in inventory), 'Original core source/data differs')
    sources = {name: sha(ROOT/name) for name in ('scripts/bgb_en_f7_runtime_repair.py',
        'scripts/bgb_en_f7_determinism_diagnostic.py', 'configs/bgb_en/F7_deterministic_runtime_v1.json')}
    origin = dict(schema='bgb_en_F7_original_inventory_v1', sources=sources, original_cells=inventory)
    if args.origin_manifest.exists():
        require(json.loads(args.origin_manifest.read_text()) == origin, 'Immutable original F7 inventory differs')
    else:
        require(args.stage == 'qualify', 'Train requires the previously persisted original inventory')
        driver.write(args.origin_manifest, origin)
    prepared = args.private/'prepared'/f'outer1_inner{args.partition}.pt'
    prepared_receipt = prepared.with_suffix('.json')
    receipt = json.loads(prepared_receipt.read_text())
    require(receipt['lineage'] == lineage and receipt['sha256'] == sha(prepared), 'Existing prepared teacher source differs')
    certificate = dict(schema='bgb_en_F7_runtime_repair_receipt_v1', status='PASS_RUNTIME_LOCK_NO_FIT',
        arm='F7', partition=args.partition, seed=args.seed, backend=backend, sources=sources,
        core_lineage=lineage, original_inventory_sha256=sha(args.origin_manifest),
        origin_manifest=args.origin_manifest.name, original_failed_attempts=3, original_successful_fits=3,
        original_failed_updates=None, original_files_modified=False, actual_new_fits=0, actual_new_updates=0,
        teacher_refits=0, prepared_origin_sha256=sha(prepared), prepared_receipt_sha256=sha(prepared_receipt),
        gradient_cache_threshold=2e-5, threshold_unchanged=True, optimizer_resumed=False,
        all_six_same_runtime_required=True, old_and_new_F7_mixing_forbidden=True)
    if args.stage == 'train':
        require(str(args.device).startswith('cuda') and args.qualification is not None,
                'Train requires explicit CUDA and matching zero-optimizer qualification')
        qualification = json.loads(args.qualification.read_text())
        validate_forward_qualification(qualification, args.partition, args.seed, lineage,
                                       sources['scripts/bgb_en_f7_determinism_diagnostic.py'])
        certificate['qualification_sha256'] = sha(args.qualification)
        cell = args.private/'cells'/f'R3_outer1_inner{args.partition}_F7_seed{args.seed}'
        require(not cell.exists(), 'New F7 cell exists; caller must preserve/archive originals before a fresh fit')
        driver.write(cell/'RUNTIME_REPAIR_START.json', certificate)
        run_args = SimpleNamespace(**vars(args), outer_fold=1, arm='F7', resume=False)
        run_args.stage = 'R3'
        try:
            driver.run_unit(run_args, assets, config, raw, member, lineage)
            complete = json.loads((cell/'COMPLETE.json').read_text())
            certificate.update(status='COMPLETE_F7_DETERMINISTIC_FRESH_FIT', actual_new_fits=1, actual_new_updates=80,
                               artifact_sha256=complete['artifact_sha256'], core_receipt_sha256=sha(cell/'COMPLETE.json'))
            require(certificate['prepared_origin_sha256'] == sha(prepared), 'Prepared teacher changed during recovery')
            driver.write(cell/'RUNTIME_REPAIR_COMPLETE.json', certificate)
        except Exception as exc:
            certificate.update(status='FAILED_F7_DETERMINISTIC_FRESH_ATTEMPT', actual_new_fits=1, actual_new_updates=None,
                               error_type=type(exc).__name__, error=str(exc))
            driver.write(cell/'RUNTIME_REPAIR_FAILED.json', certificate)
            driver.write(args.output, certificate)
            raise
    require(original_inventory(args.original_root, lock) == inventory, 'Archived originals changed during runtime operation')
    driver.write(args.output, certificate)
    print(json.dumps({key: certificate[key] for key in ('status', 'actual_new_fits', 'actual_new_updates', 'teacher_refits')}))


if __name__ == '__main__':
    main()

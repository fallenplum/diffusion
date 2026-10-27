"""Independent same-initialization O3/O4 fixed160/320 sensitivity fits.

No Adam state is imported. --resume verifies a completed independent result; it
never continues a partially trained optimizer. Default --stage qualify performs
no training. The original R4/core protocol remains fixed at eighty epochs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import bgb_en_run as driver
from scripts.bgb_en_anchor_cast_audit import comparable, censor_weights
from src.bgb_en.protocol import array_hash
from src.bgb_en.training import MechanismModel, train_neural


def initialize_new_model(arm, prepared, membership, device, seed, gamma):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # R2 never reads patch evidence. No patch bank allocation or construction
    # consumes randomness in the original trainer; None leaves its math intact.
    return MechanismModel('R2', arm, prepared, membership, None, device, gamma=gamma)


def compare_parent80(time, event, train_time, train_event, original, replayed, tolerance=2e-6):
    original, replayed = np.asarray(original, float), np.asarray(replayed, float)
    difference = replayed-original
    left, right = comparable(time, event)
    weights = censor_weights(train_time, train_event, time, event)[left]
    def credit(risk):
        margin = risk[left]-risk[right]
        return np.where(np.abs(margin) <= 1e-8, .5, margin > 0)
    before, after = credit(original), credit(replayed)
    shift = after-before
    return dict(max_abs_risk_error=float(np.abs(difference).max()), RMS_risk_error=float(np.sqrt(np.mean(difference**2))),
        within_locked_tolerance=bool(np.abs(difference).max() <= tolerance), risk_tolerance=tolerance,
        changed_pair_credits=int(np.count_nonzero(shift)),
        Harrell_change=float(shift.sum()/len(left)), Uno60_change=float(np.dot(weights, shift)/weights.sum()),
        repaired_pair_credit=float(np.maximum(shift, 0).sum()), damaged_pair_credit=float(np.maximum(-shift, 0).sum()),
        parent_Harrell=float(before.mean()), new_fit80_Harrell=float(after.mean()),
        parent_Uno60=float(np.dot(weights, before)/weights.sum()), new_fit80_Uno60=float(np.dot(weights, after)/weights.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--primary-private', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/budget_sensitivity_v1.json')
    parser.add_argument('--core-config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--stage', choices=('qualify', 'train'), default='qualify')
    parser.add_argument('--arm', choices=('O3', 'O4'), required=True)
    parser.add_argument('--partition', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--seed', type=int, choices=(31, 47), required=True)
    parser.add_argument('--epochs', type=int, choices=(160, 320), required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    lock = json.loads(args.config.read_text())
    if lock['arms'] != ['O3', 'O4'] or lock['new_fit_epochs'] != [160, 320] or lock['required_new_neural_fits'] != 24:
        raise ValueError('Independent budget sensitivity is restricted to the locked O3/O4 matrix')
    if args.private.resolve().is_relative_to(args.primary_private.resolve()):
        raise ValueError('Independent budget fits require a separate private directory from original80')
    _, core_config, raw, member, core_lineage = driver.load_inputs(SimpleNamespace(assets=args.assets, config=args.core_config))
    if core_config['training']['epochs'] != 80 or lock['R4_training_epochs'] != 80:
        raise ValueError('This sensitivity must not change the core/R4 eighty-epoch protocol')
    prepared_path = args.primary_private/'prepared'/f'outer1_inner{args.partition}.pt'
    prepared_receipt_path = prepared_path.with_suffix('.json')
    teacher_receipt = json.loads(prepared_receipt_path.read_text())
    if teacher_receipt['sha256'] != driver.sha256(prepared_path) or teacher_receipt['lineage'] != core_lineage:
        raise ValueError('Original prepared teacher source does not match; never refit teachers here')
    prepared = torch.load(prepared_path, map_location='cpu', weights_only=False)
    parent_folder = args.primary_private/'cells'/f'R2_outer1_inner{args.partition}_{args.arm}_seed{args.seed}'
    parent_receipt = json.loads((parent_folder/'COMPLETE.json').read_text())
    parent_path = parent_folder/'model.pt'
    if parent_receipt['artifact_sha256'] != driver.sha256(parent_path) or parent_receipt['lineage'] != core_lineage:
        raise ValueError('Original80 neural artifact source differs')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    if parent['epochs'] != 80 or not np.array_equal(parent['valid'], prepared['valid']):
        raise ValueError('Original80 validation partition differs')
    lineage = dict(schema='bgb_en_independent_budget_lineage_v1', core_origin=core_lineage,
        budget_lock_sha256=driver.sha256(args.config), independent_script_sha256=driver.sha256(__file__),
        evaluation_script_sha256=driver.sha256(ROOT/'scripts/bgb_en_anchor_cast_audit.py'),
        parent80_artifact_sha256=parent_receipt['artifact_sha256'],
        parent80_receipt_sha256=driver.sha256(parent_folder/'COMPLETE.json'),
        prepared_origin_sha256=driver.sha256(prepared_path), prepared_receipt_sha256=driver.sha256(prepared_receipt_path),
        arm=args.arm, seed=args.seed, inner_partition=args.partition, epochs=args.epochs,
        initialization='original seed; fresh model and fresh Adam; no checkpoint state loaded',
        teacher_refits=0, preprocessing_refits=0, outer_test_evaluated=False, R4_epochs=80)
    if args.stage == 'qualify':
        print(json.dumps(dict(status='PASS_INDEPENDENT_BUDGET_SOURCE_QUALIFICATION', lineage=lineage,
                              actual_new_neural_fits=0, optimizer_updates=0)), flush=True)
        return
    cell = args.private/'cells'/f'R2_newfit{args.epochs}_outer1_inner{args.partition}_{args.arm}_seed{args.seed}'
    complete, artifact = cell/'COMPLETE.json', cell/'model.pt'
    if complete.exists():
        existing = json.loads(complete.read_text())
        if not args.resume or existing['lineage'] != lineage or existing['artifact_sha256'] != driver.sha256(artifact):
            raise ValueError('Completed independent fit differs or requires explicit --resume')
        print(json.dumps(dict(status='REUSED_VERIFIED_COMPLETE_FIT', optimizer_resumed=False, actual_new_fits=0)), flush=True)
        return
    if cell.exists():
        raise ValueError('Incomplete independent fit exists; missing Adam state prohibits an exact resume')
    cell.mkdir(parents=True)
    driver.write(cell/'START.json', dict(lineage=lineage))
    result = None
    fit_started = False
    try:
        model = initialize_new_model(args.arm, prepared, member, args.device, args.seed, core_config['R2']['gamma'])
        initial_hashes = {name: array_hash(value.detach().cpu().numpy()) for name, value in model.state_dict().items()}
        start = time.monotonic()
        fit_started = True
        result = train_neural(model, raw, prepared, epochs=args.epochs,
            batch_size=core_config['training']['batch_size'], lr=core_config['training']['learning_rate'],
            evaluate_epochs=tuple(epoch for epoch in lock['evaluation_epochs'] if epoch <= args.epochs))
        train, valid = prepared['train'], prepared['valid']
        comparison = compare_parent80(raw['t'][valid], raw['e'][valid], raw['t'][train], raw['e'][train],
            parent['snapshots'][80]['risk'], result['snapshots'][80]['risk'], lock['parent80_validation_risk_max_abs_tolerance'])
        result.update(lineage=lineage, train=train, valid=valid, stage='R2', arm=args.arm, seed=args.seed,
            initial_state_sha256=initial_hashes, parent80_replay=comparison, actual_neural_optimizer_fits=1,
            actual_teacher_refits=0, actual_preprocessing_refits=0, optimizer_resumed=False,
            retention=driver.representation_retention(model, prepared), final_retention_decoder_fits=2,
            outer_test_evaluated=False, R4_training_budget_changed=False,
            budget_sensitivity=True, select_best_epoch=False)
        driver.torch_write(artifact, result)
        if not comparison['within_locked_tolerance']:
            raise ValueError('Same-initialization epoch80 risks exceeded the locked original80 replay tolerance')
        driver.write(complete, dict(status='COMPLETE_INDEPENDENT_BUDGET_FIT', lineage=lineage,
            artifact_sha256=driver.sha256(artifact), parent80_replay=comparison,
            epochs=args.epochs, optimizer_updates=args.epochs, actual_neural_optimizer_fits=1,
            actual_teacher_refits=0, optimizer_resumed=False, curves=result['curves'],
            final_epoch_metrics=result['snapshots'][args.epochs]['metrics'], retention=result['retention'],
            parameters=result['parameters'], peak_GPU_bytes=result['peak_GPU_bytes'], elapsed_seconds=time.monotonic()-start))
        print(json.dumps(dict(status='COMPLETE_INDEPENDENT_BUDGET_FIT', arm=args.arm, seed=args.seed,
                              partition=args.partition, epochs=args.epochs, parent80_replay=comparison)), flush=True)
    except BaseException as error:
        driver.write(cell/'FAILED.json', dict(status='FAILED', exception_type=type(error).__name__, lineage=lineage,
            actual_optimizer_updates_completed=(None if fit_started else 0) if result is None else result['optimizer_updates'],
            actual_neural_fits_started=int(fit_started), optimizer_resumed=False))
        raise


if __name__ == '__main__':
    main()

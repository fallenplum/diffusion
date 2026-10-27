"""Fixed three O9 cells x five forwards; no fitting, retry-to-pass or threshold change."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELLS = ((2, 31, 'failed_original_audit'), (2, 47, 'failed_original_audit'), (1, 31, 'passing_control'))


def pair_changes(time, event, original, current):
    import numpy as np
    from scripts.bgb_en_anchor_cast_audit import comparable
    left, right = comparable(time, event)
    def credits(value):
        difference = np.asarray(value, float)[left]-np.asarray(value, float)[right]
        return np.where(np.abs(difference) <= 1e-8, .5, difference > 0)
    old, new = credits(original), credits(current)
    change = new-old
    return dict(comparable_pairs=len(left), changed_pair_credits=int((old != new).sum()),
                improved_credit_sum=float(np.maximum(change, 0).sum()), damaged_credit_sum=float(np.maximum(-change, 0).sum()),
                net_credit_change=float(change.sum()))


def numerical_error(original, current):
    import numpy as np
    change = np.asarray(current, float)-np.asarray(original, float)
    return dict(max_abs_error=float(np.max(np.abs(change))), RMS_error=float(np.sqrt(np.mean(change**2))),
                bitwise_equal=bool(np.array_equal(original, current)), within_original_2e_minus6_gate=bool(np.max(np.abs(change)) <= 2e-6))


def state_hash(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode()); value.update(str(array.dtype).encode()); value.update(str(array.shape).encode()); value.update(array.tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--backend', choices=('inherited', 'deterministic'), required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prior-audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('New diagnostic output required; the original 184/186 audit must remain untouched')
    # This helper imports torch only after setting CUBLAS_WORKSPACE_CONFIG.
    from scripts.bgb_en_f7_determinism_diagnostic import configure_backend
    backend = configure_backend(args.backend)
    import numpy as np
    import torch
    from scripts import bgb_en_run as driver
    from scripts.bgb_en_checkpoint_audit import metrics_from_risk
    from src.bgb_en.data import read_patch_cache
    from src.bgb_en.training import MechanismModel, PatchBank, predict
    torch.set_num_threads(2)
    assets, config, raw, membership, lineage = driver.load_inputs(args)
    if config['training']['batch_size'] != 8 or config['training']['epochs'] != 80:
        raise ValueError('Frozen O9 batch/epoch protocol differs')
    # Original audit/trainer allocate the same shared patch bank even for R2.
    bank = PatchBank(read_patch_cache(assets['patch_regions']), args.device)
    results = []
    for part, seed, role in CELLS:
        folder = args.private/'cells'/f'R2_outer1_inner{part}_O9_seed{seed}'
        receipt_path, artifact_path = folder/'COMPLETE.json', folder/'model.pt'
        receipt = json.loads(receipt_path.read_text())
        if receipt['artifact_sha256'] != driver.sha256(artifact_path) or receipt['lineage'] != lineage:
            raise ValueError('O9 artifact/core source differs')
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
        prep_path = args.private/'prepared'/f'outer1_inner{part}.pt'
        prep_receipt = json.loads(prep_path.with_suffix('.json').read_text())
        prepared = torch.load(prep_path, map_location='cpu', weights_only=False)
        if (prep_receipt['sha256'] != driver.sha256(prep_path) or prepared['lineage'] != lineage
            or not np.array_equal(artifact['train'], prepared['train']) or not np.array_equal(artifact['valid'], prepared['valid'])):
            raise ValueError('O9 prepared/row partition differs')
        torch.manual_seed(seed); np.random.seed(seed)
        model = MechanismModel('R2', 'O9', prepared, membership, bank, args.device,
                               gamma=config['R2']['gamma'])
        if model.gamma != config['R2']['gamma']:
            raise ValueError('The diagnostic risk correction scale differs from the frozen R2 configuration')
        model.load_state_dict(artifact['snapshots'][80]['state_dict'], strict=True); model.eval()
        original_state = state_hash(model)
        train, valid = prepared['train'], prepared['valid']
        saved = np.asarray(artifact['snapshots'][80]['risk'], float)
        original_metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], saved)
        repeats, vectors = [], []
        for repeat in range(1, 6):
            try:
                risk, _ = predict(model, valid, batch_size=8)
                metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], risk)
                repeats.append(dict(repeat=repeat, status='FORWARD_COMPLETE', **numerical_error(saved, risk),
                    independent_Harrell=metric['Harrell'], independent_Uno60=metric['Uno60'],
                    delta_Harrell_vs_saved=metric['Harrell']-original_metric['Harrell'],
                    delta_Uno60_vs_saved=metric['Uno60']-original_metric['Uno60'],
                    pair_credit_changes_vs_saved=pair_changes(raw['t'][valid], raw['e'][valid], saved, risk)))
                vectors.append((repeat, risk))
            except Exception as exc:
                repeats.append(dict(repeat=repeat, status='FORWARD_FAILED', error_type=type(exc).__name__))
        cross = [dict(left_repeat=i, right_repeat=j, **numerical_error(left, right),
                      pair_credit_changes=pair_changes(raw['t'][valid], raw['e'][valid], left, right))
                 for position, (i, left) in enumerate(vectors) for j, right in vectors[position+1:]]
        unchanged = state_hash(model) == original_state
        if not unchanged or any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError('Zero-fit diagnostic mutated weights or accumulated gradients')
        results.append(dict(arm='O9', inner_partition=part, seed=seed, locked_role=role,
            artifact_sha256=receipt['artifact_sha256'], complete_sha256=driver.sha256(receipt_path),
            prepared_sha256=driver.sha256(prep_path), strict_state_dict_loaded=True,
            state_dict_sha256=original_state, weights_bitwise_unchanged=unchanged, parameter_gradients_absent=True,
            saved_Harrell=original_metric['Harrell'], saved_Uno60=original_metric['Uno60'],
            gamma=model.gamma, gamma_matches_frozen_config=True,
            repeats=repeats, cross_repeat_comparisons=cross,
            omics_mode=model.omics.mode, shuffled_membership_seed=270099, module_buffer_edges_restored_from_checkpoint=True))
        print(json.dumps(dict(partition=part, seed=seed, repeats=len(repeats), successful_forwards=len(vectors),
            max_risk_error=max((row.get('max_abs_error', 0.) for row in repeats), default=None))), flush=True)
        del model
    result = dict(schema='bgb_en_O9_fixed_five_forward_drift_v1',
        status='COMPLETE_FIXED_FIVE_FORWARD_DIAGNOSTIC' if all(all(row['status'] == 'FORWARD_COMPLETE' for row in cell['repeats']) for cell in results) else 'COMPLETED_WITH_FAILED_FORWARDS',
        backend=backend, device=args.device, cells=results, fixed_repeats_per_cell=5, batch_size=8,
        optimizer_fits=0, optimizer_updates=0, backward_calls=0, original_risk_tolerance=2e-6, threshold_relaxed=False,
        repeat_to_pass=False, best_repeat_selected=False, original_audit_overwritten=False,
        prior_audit_sha256=driver.sha256(args.prior_audit) if args.prior_audit else None,
        source_sha256=driver.sha256(__file__), core_lineage_sha256=hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest(),
        metric_reference_sha256=driver.sha256(ROOT/'scripts/bgb_en_checkpoint_audit.py'),
        patient_rows_or_risk_vectors_exported=False,
        interpretation='all five fixed repeats are reported; numerical drift is distinguished from rank-credit changes; a deterministic process is a separate diagnostic, not permission to replace a failed replay or retrain',
        selection_lock_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

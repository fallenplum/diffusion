"""Two source-only target-rematching fits using the frozen v2 trainer.

Every control is a new complete-six-donor fit, with original true-target
standardization, unchanged tissue CE, twenty fixed epochs and no heldout claim.
Source/target identities and weights stay in the supplied private output root.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bgb_en import external as source


def target_remapping(data, control, seed):
    donor, section, tissue = data['donor'], data['section'], data['tissue_y']
    rng = np.random.default_rng(seed)
    mapping = np.arange(len(tissue), dtype=np.int64)
    singleton = 0
    if control == 'WRONG_DONOR':
        donors = np.unique(donor)
        for position, identity in enumerate(donors):
            other = donors[(position+1) % len(donors)]
            for label in np.unique(tissue[donor == identity]):
                destination = np.flatnonzero((donor == identity) & (tissue == label))
                candidates = rng.permutation(np.flatnonzero((donor == other) & (tissue == label)))
                source.require(len(candidates) > 0, 'wrong-donor source lacks the same tissue class')
                mapping[destination] = np.resize(candidates, len(destination))
        source.require(np.all(donor[mapping] != donor), 'wrong-donor target map did not exclude recipient donors')
    elif control == 'SPATIAL_SHUFFLE':
        for identity in np.unique(donor):
            for slide in np.unique(section[donor == identity]):
                for label in np.unique(tissue[(donor == identity) & (section == slide)]):
                    rows = np.flatnonzero((donor == identity) & (section == slide) & (tissue == label))
                    if len(rows) == 1:
                        singleton += 1
                    order = rng.permutation(rows)
                    mapping[order] = np.roll(order, 1)
        source.require(np.array_equal(np.sort(mapping), np.arange(len(mapping))), 'spatial target map must be bijective')
        source.require(np.array_equal(donor[mapping], donor) and np.array_equal(section[mapping], section),
                       'spatial remap crossed donor or section')
    else:
        raise ValueError('Only the two locked negative source controls are supported')
    source.require(np.array_equal(tissue[mapping], tissue), 'source target rematching changed tissue class')
    counts = np.bincount(mapping, minlength=len(mapping))
    return mapping, dict(control=control, seed=int(seed), same_tissue_all_rows=True,
        cross_donor_rows=int(np.count_nonzero(donor[mapping] != donor)),
        self_target_rows=int(np.count_nonzero(mapping == np.arange(len(mapping)))), singleton_groups=singleton,
        unique_source_rows=int(np.count_nonzero(counts)), maximum_source_reuse=int(counts.max()),
        target_reuse_is_allowed=control == 'WRONG_DONOR', programs_share_same_target_map=True,
        target_map_sha256=source.array_hash(mapping))


@contextmanager
def frozen_true_scaler(scaler):
    """Supply the preregistered original targets' statistics to frozen trainer.

    The existing trainer calls this module-level dependency once. The temporary
    explicit substitution changes only which source-fit target statistics are
    supplied; the optimizer, model, losses and twenty-step schedule are intact.
    """
    original = source.fit_target_scaler
    def locked_scaler(target, weight):
        source.require(target.shape[1] == 4 and np.isclose(weight.sum(), 1), 'invalid rematched source-fit scaler request')
        return {key: np.array(value, copy=True) for key, value in scaler.items()}
    source.fit_target_scaler = locked_scaler
    try:
        yield
    finally:
        source.fit_target_scaler = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--control', choices=('WRONG_DONOR', 'SPATIAL_SHUFFLE'), required=True)
    parser.add_argument('--stage', choices=('audit', 'train'), default='audit')
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--seed', type=int, default=9033017)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    source.require(not args.output_root.resolve().is_relative_to((ROOT/'reports').resolve()), 'source artifacts require private output')
    data, source_audit = source.load_qualified_source(args.source_data, args.source_manifest)
    map_seed = 9033021 if args.control == 'WRONG_DONOR' else 9033022
    mapping, map_audit = target_remapping(data, args.control, map_seed)
    weight = source.natural_weights(data['donor'], data['section'])
    true_scaler = source.fit_target_scaler(data['program_y'], weight)
    lineage = dict(schema='bgb_en_external_negative_source_v1', source_protocol=source.PROTOCOL,
        sources={name: source.sha256(ROOT/name) for name in ('src/bgb_en/external.py', 'scripts/bgb_en_external_controls.py')},
        source_data_sha256=source_audit['source_data_sha256'], source_manifest_sha256=source_audit['source_manifest_sha256'],
        source_control=args.control, model_seed=args.seed, target_map=map_audit,
        target_scaler_source='original_unrematched_full_source_train_targets',
        tissue_CE_targets_rematched=False, source_epochs=20, independent_heldout_evaluation=False,
        TCGA_DSS_read=False, original_ST_test_read=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    source.atomic_json(args.output_root/'SOURCE_CONTROL_QUALIFICATION.json', lineage)
    print(json.dumps(dict(status='PASS_SOURCE_CONTROL_QUALIFICATION', **map_audit, actual_new_fits=0)), flush=True)
    if args.stage == 'audit':
        return
    directory = args.output_root/'fit_06'
    complete = directory/'COMPLETE.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        source.require(args.resume and record['lineage'] == lineage, 'existing source control receipt differs')
        source.require(all(source.sha256(directory/name) == digest for name, digest in record['artifacts'].items()),
                       'existing source control artifact SHA differs')
        print(json.dumps(dict(status='REUSED_VERIFIED_COMPLETE', actual_new_fits=0)), flush=True)
        return
    source.require(not directory.exists(), 'incomplete source control exists; refuse silent retraining')
    directory.mkdir()
    source.atomic_json(directory/'STARTED.json', lineage)
    negative = copy.copy(data)
    negative['program_y'] = np.array(data['program_y'][mapping], copy=True)
    try:
        with frozen_true_scaler(true_scaler):
            result = source.fit_source_partition(negative, np.arange(len(mapping)), np.empty(0, np.int64),
                                                 device='cpu', seed=args.seed, epochs=20)
        source.require(all(np.array_equal(result['target_scaler'][key], true_scaler[key]) for key in true_scaler),
                       'negative source fit changed the true target scaler')
        evaluator = source.ExternalProgramModel()
        evaluator.load_state_dict(result['state_dict']); evaluator.eval()
        with torch.no_grad():
            program, tissue = evaluator(torch.as_tensor(np.array(data['X'], copy=True)))
        true_training_metrics = source._metrics(data, np.arange(len(mapping)), program.numpy(), tissue.numpy(), true_scaler)
        torch.save(dict(schema='bgb_en_external_projection_transfer_v2_unitnorm', role='full_source_transfer', epochs=20,
                        source_donors=6, input_normalization=source.INPUT_NORMALIZATION,
                        source_control=args.control, project_state_dict=result['project_state_dict'],
                        source_data_sha256=source_audit['source_data_sha256'],
                        source_manifest_sha256=source_audit['source_manifest_sha256'],
                        control_lineage=lineage, TCGA_DSS_read=False, original_ST_test_read=False), directory/'UNI2_PROJECT_TRANSFER.pt')
        torch.save(dict(schema='bgb_en_external_control_checkpoint_v1', state_dict=result['state_dict'],
                        target_mean=torch.from_numpy(true_scaler['mean']), target_scale=torch.from_numpy(true_scaler['scale']),
                        class_weights=torch.from_numpy(result['class_weights']), lineage=lineage), directory/'checkpoint.pt')
        np.savez_compressed(directory/'PRIVATE_TARGET_MAP.npz', target_map=mapping)
        artifacts = {name: source.sha256(directory/name) for name in ('UNI2_PROJECT_TRANSFER.pt', 'checkpoint.pt', 'PRIVATE_TARGET_MAP.npz')}
        record = dict(status='COMPLETE_NEGATIVE_SOURCE_FIT', lineage=lineage, artifacts=artifacts,
                      actual_new_supervised_fits=1, epochs=20, optimizer_updates=20,
                      curve=result['curve'], rematched_training_target_metrics=result['train_metrics'],
                      original_true_target_training_metrics=true_training_metrics,
                      true_target_evaluation_is_heldout=False, elapsed_seconds=result['elapsed_seconds'])
        source.atomic_json(complete, record)
        print(json.dumps(dict(status=record['status'], control=args.control, actual_new_supervised_fits=1,
                              epochs=20, true_target_metrics=true_training_metrics), sort_keys=True), flush=True)
    except BaseException as error:
        source.atomic_json(directory/'FAILED.json', dict(status='FAILED', exception_type=type(error).__name__))
        raise


if __name__ == '__main__':
    main()

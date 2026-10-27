"""Exploratory matched-support membership null; never launched automatically.

Lock/qualify runs use CPU metadata only. Train requires an explicit variant,
partition and seed and reuses core teachers. The source-null changes annotation
columns only, within exact training measurement-mask strata and active union.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import bgb_en_run as driver
from scripts.bgb_en_checkpoint_audit import require, sha256, index_hash, verify_preprocessing_state
from scripts.bgb_en_input_coverage import canonical_hash, support_counts

SPEC_PATH = ROOT / 'configs/bgb_en/matched_support_v1.json'


def matched_support_null(membership, rna_mask, cna_mask, train, *, seed):
    """Preserve mask-specific pathway coverage exactly; no outcomes consumed."""
    member = np.asarray(torch.as_tensor(membership, dtype=torch.bool).cpu())
    rna, cna, fit = np.asarray(rna_mask, bool), np.asarray(cna_mask, bool), np.asarray(train)
    require(member.ndim == 2 and rna.shape == cna.shape and rna.shape[1] == member.shape[1], 'MASK_GENE_SHAPE_MISMATCH')
    require(fit.ndim == 1 and np.issubdtype(fit.dtype, np.integer) and len(np.unique(fit)) == len(fit), 'INVALID_FIT_INDICES')
    require(len(fit) > 0 and (fit >= 0).all() and (fit < len(rna)).all(), 'INVALID_FIT_RANGE')
    active = member.any(axis=0)
    permutation = np.arange(member.shape[1], dtype=np.int64)
    # Only training mask columns enter the stratum definition. Validation
    # measurement patterns and every outcome are excluded by construction.
    signature = np.packbits(np.concatenate([rna[fit], cna[fit]], axis=0).T, axis=1)
    strata = {}
    for gene in np.flatnonzero(active):
        strata.setdefault(signature[gene].tobytes(), []).append(int(gene))
    generator = np.random.default_rng(int(seed))
    movable, semantically_movable = 0, 0
    for columns in strata.values():
        chosen = np.asarray(columns, np.int64)
        if len(chosen) < 2:
            continue
        movable += len(chosen)
        mixed = np.unique(member[:, chosen].T, axis=0).shape[0] > 1
        semantically_movable += len(chosen) if mixed else 0
        order = generator.permutation(chosen)
        if np.array_equal(order, chosen):
            order = np.roll(order, 1)
        if mixed and np.array_equal(member[:, order], member[:, chosen]):
            # One fixed fallback, not outcome-driven resampling. A rotation of
            # distinct patterns necessarily changes at least one membership.
            order = np.roll(chosen, 1)
        permutation[chosen] = order
    null = member[:, permutation]
    require(np.array_equal(np.sort(permutation), np.arange(len(permutation))), 'NOT_A_COLUMN_PERMUTATION')
    require(np.array_equal(null.any(axis=0), active), 'ACTIVE_GENE_SUPPORT_CHANGED')
    require(np.array_equal(null.sum(axis=1), member.sum(axis=1)), 'PATHWAY_SIZE_CHANGED')
    require(np.array_equal(null.astype(np.int64) @ null.astype(np.int64).T,
                           member.astype(np.int64) @ member.astype(np.int64).T), 'PATHWAY_OVERLAP_CHANGED')
    for mask in (rna[fit], cna[fit]):
        require(np.array_equal(mask.astype(np.int64) @ null.astype(np.int64).T,
                               mask.astype(np.int64) @ member.astype(np.int64).T), 'TRAIN_PATHWAY_MEASUREMENT_COVERAGE_CHANGED')
    changed = np.any(null != member, axis=0)
    audit = {
        'seed': int(seed), 'pathway_count': int(member.shape[0]), 'active_union_gene_count': int(active.sum()),
        'stratum_count': len(strata), 'singleton_stratum_count': sum(len(x) == 1 for x in strata.values()),
        'permutable_gene_count': movable, 'semantically_permutable_gene_count': semantically_movable,
        'changed_gene_annotation_count': int(changed.sum()), 'changed_annotation_edge_count': int(np.sum(null != member)),
        'true_membership_sha256': index_hash(member), 'actual_membership_sha256': index_hash(null),
        'actual_permutation_sha256': index_hash(permutation), 'training_indices_sha256': index_hash(fit),
        'pathway_sizes_equal': True, 'pathway_overlaps_equal': True, 'edge_count_equal': True,
        'active_gene_union_equal': True, 'train_RNA_pathway_mask_counts_equal': True,
        'train_CNA_pathway_mask_counts_equal': True, 'values_permuted': False, 'gene_IDs_permuted': False,
        'validation_masks_used': False, 'outcomes_used': False,
    }
    require(audit['changed_gene_annotation_count'] > 0, 'NO_SEMANTIC_MEMBERSHIP_CHANGE_POSSIBLE')
    return torch.from_numpy(null.copy()), torch.from_numpy(permutation), audit


def base_model(model_class, stage, variant, prepared, member, bank, device, **kwargs):
    spec = json.loads(SPEC_PATH.read_text())
    definition = spec['variants'][variant]
    require(stage == definition['stage'], 'VARIANT_STAGE_MISMATCH')
    require(definition['base_arm'] not in ('O9', 'F5_SHUFFLED_MEMBERSHIP'), 'DOUBLE_SHUFFLE_FORBIDDEN')
    # Explicit true base arm avoids the original constructor's O9 shuffle.
    return model_class(stage, definition['base_arm'], prepared, member, bank, device, **kwargs)


def source(spec, core_lineage):
    return {
        'core_lineage': core_lineage, 'specification': spec,
        'auxiliary_source': {name: sha256(ROOT / name) for name in (
            'scripts/bgb_en_matched_support.py', 'scripts/bgb_en_input_coverage.py',
            'scripts/bgb_en_checkpoint_audit.py', 'configs/bgb_en/matched_support_v1.json')},
    }


def prepared_inputs(args, core_lineage, partition):
    stem = args.primary_private / 'prepared' / f'outer1_inner{partition}'
    path, receipt_path = stem.with_suffix('.pt'), stem.with_suffix('.json')
    require(path.is_file() and receipt_path.is_file(), 'NO_EXISTING_PREPARED_TEACHERS')
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get('status') == 'COMPLETE' and receipt.get('lineage') == core_lineage,
            'PREPARED_RECEIPT_LINEAGE_MISMATCH')
    require(receipt.get('sha256') == sha256(path), 'PREPARED_SHA256_MISMATCH')
    prepared = torch.load(path, map_location='cpu', weights_only=False)
    require(prepared.get('lineage') == core_lineage, 'PREPARED_LINEAGE_MISMATCH')
    verify_preprocessing_state(prepared['features']['state'], np.asarray(prepared['train']))
    return prepared, receipt['sha256']


def null_from_prepared(prepared, membership, seed):
    feature = prepared['features']
    return matched_support_null(membership, feature['rna_mask'], feature['cna_mask'], prepared['train'], seed=seed)


def make_lock(args, assets, config, raw, member, core_lineage):
    spec = json.loads(SPEC_PATH.read_text())
    require(config['training']['epochs'] == spec['epochs'] and config['training']['batch_size'] == spec['batch_size']
            and config['training']['learning_rate'] == spec['learning_rate'], 'CORE_BUDGET_DRIFT')
    rows = []
    for part in spec['partitions']:
        prepared, prep_sha = prepared_inputs(args, core_lineage, part)
        null, perm, audit = null_from_prepared(prepared, member, spec['permutation_seed'])
        selected = prepared['features']['state']['molecular']['selected']
        old = np.asarray(raw['_direct_gene_universe'], bool)
        audit.update(inner_partition=part, prepared_sha256=prep_sha,
                     true_direct512_support=support_counts(selected, member.numpy(), old),
                     null_direct512_support=support_counts(selected, null.numpy(), old))
        require(audit['true_direct512_support']['token_supported_selected_gene_count'] ==
                audit['null_direct512_support']['token_supported_selected_gene_count'], 'DIRECT512_SUPPORT_CHANGED')
        path = args.private / 'nulls' / f'partition_{part}.pt'
        artifact = {'membership': null, 'permutation': perm, 'audit': audit}
        if path.exists():
            previous = torch.load(path, map_location='cpu', weights_only=False)
            require(previous['audit'] == audit and torch.equal(previous['membership'], null)
                    and torch.equal(previous['permutation'], perm), 'NULL_ARTIFACT_CHANGED')
        else:
            driver.torch_write(path, artifact)
        audit['null_artifact_sha256'] = sha256(path)
        rows.append(audit)
        del prepared
    lock = {'schema': 'bgb_en_matched_support_lock_v1', 'status': 'LOCKED_BEFORE_AUXILIARY_FITS',
            'lineage': source(spec, core_lineage), 'partitions': rows,
            'actual_optimizer_fits_at_lock': 0, 'actual_teacher_refits': 0, 'GPU_calls_at_lock': 0,
            'post_hoc_exploratory': True, 'R4_selection_modified': False, 'outer_test_evaluated': False}
    path = args.private / 'LOCK.json'
    if path.exists():
        require(json.loads(path.read_text()) == lock, 'EXISTING_MATCHED_SUPPORT_LOCK_CHANGED')
    else:
        driver.write(path, lock)
    args.output.mkdir(parents=True, exist_ok=True)
    driver.write(args.output / 'MATCHED_SUPPORT_NULL_LOCK.json', lock)
    return lock


def train(args, assets, config, raw, member, core_lineage, lock):
    spec = lock['lineage']['specification']
    require(args.variant in spec['variants'] and args.partition in spec['partitions']
            and args.seed in spec['seeds'], 'UNIT_OUTSIDE_MATCHED_SUPPORT_LOCK')
    definition = spec['variants'][args.variant]
    prepared, prep_sha = prepared_inputs(args, core_lineage, args.partition)
    null, perm, audit = null_from_prepared(prepared, member, spec['permutation_seed'])
    locked = next(row for row in lock['partitions'] if row['inner_partition'] == args.partition)
    require(audit['actual_membership_sha256'] == locked['actual_membership_sha256']
            and audit['actual_permutation_sha256'] == locked['actual_permutation_sha256']
            and prep_sha == locked['prepared_sha256'], 'LOCKED_NULL_OR_PREPARED_CHANGED')
    expected = copy.deepcopy(core_lineage)
    expected['matched_support_auxiliary'] = {'source': lock['lineage'], 'null': locked,
                                            'lock_sha256': sha256(args.private / 'LOCK.json'),
                                            'variant': args.variant, 'base_arm': definition['base_arm']}
    original_prepare, original_model = driver.prepare, driver.MechanismModel
    def reuse_prepared(inner_args, raw, supplied_member, lineage):
        require(inner_args.partition == args.partition and lineage == expected, 'UNEXPECTED_AUXILIARY_PREPARE')
        return prepared
    def build(stage, arm, value, supplied_member, bank, device, **kwargs):
        require(arm == args.variant and torch.equal(supplied_member, null), 'UNEXPECTED_NULL_MEMBERSHIP')
        return base_model(original_model, stage, arm, value, supplied_member, bank, device, **kwargs)
    driver.prepare, driver.MechanismModel = reuse_prepared, build
    args.stage, args.arm, args.outer_fold = definition['stage'], args.variant, spec['outer_fold']
    try:
        driver.run_unit(args, assets, config, raw, null, expected)
    finally:
        driver.prepare, driver.MechanismModel = original_prepare, original_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['lock', 'train'], required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--primary-private', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en')
    p.add_argument('--variant', choices=['O4_MATCHED_SUPPORT', 'F5_MATCHED_SUPPORT'])
    p.add_argument('--partition', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    original_stage = args.stage
    require(args.private.resolve() != args.primary_private.resolve()
            and not args.private.resolve().is_relative_to(args.primary_private.resolve())
            and not args.primary_private.resolve().is_relative_to(args.private.resolve())
            and not args.private.resolve().is_relative_to((ROOT / 'reports').resolve()), 'AUX_PRIVATE_DIRECTORY_COLLISION')
    torch.set_num_threads(2)
    assets, config, raw, member, lineage = driver.load_inputs(args)
    lock = make_lock(args, assets, config, raw, member, lineage)
    if original_stage == 'train':
        train(args, assets, config, raw, member, lineage, lock)
    else:
        print(json.dumps({'status': lock['status'], 'planned_neural_fits': 12,
                          'actual_optimizer_fits': 0, 'GPU_calls': 0}))


if __name__ == '__main__':
    main()

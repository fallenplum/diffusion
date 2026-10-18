"""Fit-only association of saved bounded corrections and known DSS status.

No prediction, preparation, fitting, optimization, endpoint API or heldout-label
analysis. Serialized checkpoints also contain heldout predictions; those arrays
are never selected for a statistic. Patient corrections stay in memory only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'phase26_saved_fit_boundary_association_v1'
SEEDS, FOLDS, ARMS = (31, 47), (1, 2, 3), ('QUERY', 'ATTN', 'COMP')
BOUNDARY = .095


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_vector(values, n, name):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    require(values.shape == (n,) and np.issubdtype(values.dtype, np.number)
            and not np.iscomplexobj(values) and np.isfinite(values).all(), name + ' must be a matching finite vector')
    return np.array(values, np.float64, copy=True)


def correction_summary(correction):
    require(len(correction) > 0, 'a known-DSS training group is empty')
    return {'count': len(correction), 'mean_correction': float(correction.mean()),
        'SD_correction': float(correction.std(ddof=0)),
        'positive_fraction': float((correction > 0).mean()),
        'negative_fraction': float((correction < 0).mean()), 'zero_fraction': float((correction == 0).mean()),
        'absolute_boundary_fraction': float((np.abs(correction) > BOUNDARY).mean()),
        'positive_boundary_fraction': float((correction > BOUNDARY).mean()),
        'negative_boundary_fraction': float((correction < -BOUNDARY).mean())}


def pearson(correction, indicator):
    a = correction - correction.mean(); b = indicator - indicator.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.:
        return None, 'UNDEFINED_ZERO_SD'
    return float(np.clip(np.dot(a, b) / denominator, -1., 1.)), 'DEFINED'


def mapping_hash(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def blocked(*args, **kwargs):
    raise RuntimeError('forward, fitting, optimization, preparation and endpoint APIs are forbidden')


def audit(identity, private_root, report_dir):
    from src.phase26_multiquery import runtime, data
    old = data.old
    private_root, report_dir = Path(private_root), Path(report_dir)
    require(runtime.source() == identity, 'current training source mismatch')
    require(old._runtime().source() == runtime.LEGACY_SOURCE, 'immutable legacy source mismatch')
    source_before = runtime.source_files(); legacy_source_before = old._runtime().source_files()
    prepared, checkpoints, private_paths = {}, [], {}
    # Every expected completion and byte hash is checked before deserialization.
    for fold in FOLDS:
        path = private_root / 'prepared' / f'prototype_fold{fold}.pt'
        receipt_path = path.with_suffix('.json')
        require(path.is_file() and receipt_path.is_file(), 'prepared main receipt missing')
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('status') == 'COMPLETE' and receipt.get('source') == identity
                and receipt.get('fold') == fold and receipt.get('legacy_source') == runtime.LEGACY_SOURCE
                and receipt.get('sha256') == sha(path), 'prepared receipt/source/SHA mismatch')
        legacy_path = old.prepared_path(fold, 'prototype')
        legacy_receipt = legacy_path.with_suffix('.json')
        require(sha(legacy_path) == receipt['legacy_prepared_sha256'], 'legacy prepared payload changed')
        prepared[fold] = (path, receipt)
        for label, artifact in ((f'new_prepared_{fold}', path), (f'new_prepared_receipt_{fold}', receipt_path),
                                 (f'legacy_prepared_{fold}', legacy_path), (f'legacy_prepared_receipt_{fold}', legacy_receipt)):
            private_paths[label] = artifact
    for seed in SEEDS:
        for fold in FOLDS:
            folder = private_root / 'prototype' / f'seed{seed}_fold{fold}'
            done = runtime.verify_done(folder, identity, seed, fold)
            checkpoints.append((seed, fold, folder / 'checkpoint.pt', done))
            private_paths[f'checkpoint_{seed}_{fold}'] = folder / 'checkpoint.pt'
            private_paths[f'completion_{seed}_{fold}'] = folder / 'DONE_PRIVATE.json'
    private_before = {label: sha(path) for label, path in private_paths.items()}
    import sklearn.decomposition
    import src.phase10_trimodal.math as cox
    from src.phase26_multiquery import train, model
    rows, loaded = [], {}
    with contextlib.ExitStack() as stack:
        for obj, name in ((old.parent, 'endpoints'), (old.parent, 'splits'),
                          (data, 'prepare'), (data, 'build_prepared'), (data, 'fit_ridge'),
                          (old, 'prepare'), (old, 'build_prepared'), (old, 'fit_preprocessor'), (old, 'fit_ridge'),
                          (cox, 'fit_ridge'), (train, '_forward'), (train, 'train_arm'),
                          (train, 'predict_arm'), (train, 'run_cell'), (model.MultiQuery, 'forward')):
            stack.enter_context(patch.object(obj, name, blocked))
        stack.enter_context(patch.object(sklearn.decomposition.PCA, 'fit', blocked))
        stack.enter_context(patch.object(torch.optim.AdamW, 'step', blocked))
        for fold, (path, receipt) in prepared.items():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            require(saved.get('source') == identity and saved.get('fold') == fold and saved.get('stage') == 'prototype'
                    and saved.get('legacy_prepared_sha256') == receipt['legacy_prepared_sha256'], 'prepared metadata drift')
            fit = np.asarray(saved['fit'])
            require(fit.ndim == 1 and np.issubdtype(fit.dtype, np.integer) and len(np.unique(fit)) == len(fit),
                    'fit patients are not unique integer indices')
            indicator = finite_vector(saved['fit_event'], len(fit), 'stored training DSS indicator')
            duration = finite_vector(saved['fit_time'], len(fit), 'stored training DSS duration')
            require(np.isin(indicator, (0, 1)).all() and (duration >= 0).all(), 'invalid stored fit supervision')
            # Durations are validated only, never used for ranking, grouping,
            # selection or any new target. No heldout supervision exists here.
            base = finite_vector(np.asarray(saved['base'])[fit], len(fit), 'saved crossfit main offset')
            require(np.array_equal(base, np.asarray(saved['CP_cf_fit']) + np.asarray(saved['RNA_cf_fit'])),
                    'fit baseline is not the exact saved crossfit branch sum')
            loaded[fold] = (fit, indicator, base)
        for seed, fold, path, done in checkpoints:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            require(checkpoint.get('source') == identity and checkpoint.get('seed') == seed
                    and checkpoint.get('fold') == fold and checkpoint.get('stage') == 'prototype', 'checkpoint metadata drift')
            fit, indicator, base = loaded[fold]
            require(np.array_equal(checkpoint['fit'], fit), 'checkpoint and prepared fit patient order differ')
            require(checkpoint['data_audit']['prepared_sha256'] == prepared[fold][1]['sha256'],
                    'checkpoint refers to another prepared main')
            require(set(checkpoint['arms']) == set(ARMS), 'three trained arms are required')
            for arm in ARMS:
                fitted = finite_vector(checkpoint['arms'][arm]['fit_risk'], len(fit), 'saved trained fit prediction')
                correction = fitted - base
                require(np.isfinite(correction).all() and (np.abs(correction) <= .10000002).all(), 'saved correction violates fixed bound')
                association, status = pearson(correction, indicator)
                rows.append({'seed': seed, 'fold': fold, 'arm': arm, 'fit_patients': len(fit),
                    'DSS_event_group': correction_summary(correction[indicator == 1]),
                    'DSS_censored_group': correction_summary(correction[indicator == 0]),
                    'all_fit_corrections': correction_summary(correction),
                    'correction_DSS_indicator_Pearson': association, 'Pearson_status': status,
                    'scope': 'known_fit_labels_only_postfit_descriptive_association'})
    private_after = {label: sha(path) for label, path in private_paths.items()}
    require(private_before == private_after, 'a read private artifact changed during the audit')
    require(source_before == runtime.source_files() and legacy_source_before == old._runtime().source_files()
            and runtime.source() == identity and old._runtime().source() == runtime.LEGACY_SOURCE,
            'new or legacy source bytes changed during audit')
    means = []
    group_fields = ('mean_correction', 'SD_correction', 'positive_fraction', 'negative_fraction', 'zero_fraction',
                    'absolute_boundary_fraction', 'positive_boundary_fraction', 'negative_boundary_fraction')
    for arm in ARMS:
        selected = [row for row in rows if row['arm'] == arm]
        require(len(selected) == 6, 'each arm needs six complete saved models')
        summary = {'arm': arm, 'equally_weighted_model_cells': 6, 'aggregation': 'arithmetic mean of six cell summaries, not pooled patient rows'}
        for group in ('DSS_event_group', 'DSS_censored_group', 'all_fit_corrections'):
            summary[group] = {field: float(np.mean([row[group][field] for row in selected])) for field in group_fields}
            summary[group]['mean_recorded_patients_per_cell'] = float(np.mean([row[group]['count'] for row in selected]))
        correlations = [row['correction_DSS_indicator_Pearson'] for row in selected]
        summary['mean_cell_correction_DSS_indicator_Pearson'] = (
            float(np.mean(correlations)) if all(value is not None for value in correlations) else None)
        summary['Pearson_defined_cells'] = sum(value is not None for value in correlations)
        means.append(summary)
    output = {'status': 'COMPLETE_FIT_BOUNDARY_ASSOCIATION_AUDIT', 'version': VERSION, 'source': identity,
        'audit_script_sha256': sha(__file__), 'complete_checkpoints': 6, 'prepared_partitions': 3,
        'arm_model_cells': 18, 'absolute_boundary_threshold': BOUNDARY,
        'positive_boundary_rule': 'correction > 0.095', 'negative_boundary_rule': 'correction < -0.095',
        'cell_rows': rows, 'arm_equal_cell_means': means,
        'scope': 'association with labels already used to train each saved model, not heldout prediction or causal attribution',
        'patient_repeats_across_partitions_and_seeds': True, 'unique_independent_patient_count_claimed': False,
        'no_p_values_or_confidence_intervals': True, 'DSS_duration_used_for_statistics': False,
        'saved_fit_label_bytes_read': True, 'heldout_labels_read': False,
        'serialized_heldout_predictions_loaded_but_not_analyzed': True,
        'statistic_scope': 'saved fit correction = saved fit prediction minus saved crossfitted main on the identical patient order',
        'new_model_fits': 0, 'optimizer_steps': 0, 'neural_forward_calls': 0, 'endpoint_API_reads': 0,
        'new_or_legacy_training_source_bytes_unchanged': True,
        'new_source_file_count': len(source_before), 'legacy_source_file_count': len(legacy_source_before),
        'new_source_bytes_fingerprint': mapping_hash(source_before), 'legacy_source_bytes_fingerprint': mapping_hash(legacy_source_before),
        'private_artifacts_checked_unchanged': len(private_before),
        'private_artifacts_bytes_fingerprint': mapping_hash(private_before),
        'immutability_scope': 'all source files registered by both runtimes and all 24 private prepared/checkpoint/completion files read here',
        'patient_arrays_weights_identifiers_or_per_patient_labels_public': False}
    report_dir.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(report_dir / 'BOUNDARY_AUDIT.json', output)
    print(json.dumps({'status': output['status'], 'source': identity, 'arm_equal_cell_means': means,
                      'private_artifacts_checked_unchanged': len(private_before),
                      'new_model_fits': 0, 'neural_forward_calls': 0, 'endpoint_API_reads': 0}, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--private-root', type=Path, default=Path('/root/autodl-tmp/phase26_multiquery_private'))
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/phase26_multiquery')
    args = parser.parse_args()
    audit(args.source, args.private_root, args.report_dir)


if __name__ == '__main__':
    main()

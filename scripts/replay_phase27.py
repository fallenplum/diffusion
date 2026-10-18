"""Independent saved-slot PCA/linear-risk replay without labels or fitting.

The original Phase26 main and neural replay certificates are bound by source,
current script bytes and report hashes. This script does not rerun extraction,
predict an encoder, fit a projection, optimize a head or read endpoint APIs.
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
SEEDS, FOLDS, ARMS = (31, 47), (1, 2, 3), ('QUERY', 'ATTN', 'COMP')
CONTROLS = ('MATCHED', 'RNA_MEAN_QUERY', 'WRONG_RNA_QUERY', 'VALUE_OFF', 'PATCH_PERMUTE')
ATOL = 1e-10
VERSION = 'phase27_saved_projection_linear_risk_replay_v1'


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array(values, name, shape=None):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values)
    require(np.issubdtype(result.dtype, np.number) and not np.iscomplexobj(result)
            and np.isfinite(result).all() and (shape is None or result.shape == shape), name + ' finite/shape mismatch')
    return np.asarray(result, np.float64)


def compare(actual, expected, name, errors):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + ' shape mismatch')
    error = float(np.max(np.abs(actual - expected), initial=0.))
    require(np.isfinite(error) and error <= ATOL, name + ' differs from saved arithmetic')
    errors.append(error)
    return error


def legacy_evidence(legacy_source):
    report_dir = ROOT / 'reports/phase26_multiquery'
    main_path, neural_path = report_dir / 'MAIN_REPLAY_AUDIT.json', report_dir / 'REPLAY_AUDIT.json'
    main, neural = json.loads(main_path.read_text()), json.loads(neural_path.read_text())
    for report, path, status in ((main, ROOT / 'scripts/replay_phase26_main.py', 'PASS_NO_FIT_MAIN_REPLAY'),
                                  (neural, ROOT / 'scripts/replay_phase26.py', 'PASS')):
        require(report.get('status') == status and report.get('source') == legacy_source
                and report.get('replay_script_sha256') == sha(path), 'legacy replay source/status/current script bytes changed')
        require(report.get('maximum_absolute_risk_error', np.inf) <= (ATOL if report is main else 1e-7),
                'legacy replay numerical gate failed')
        require(report.get('new_predictor_fits') == 0 and report.get('endpoint_API_reads') == 0,
                'legacy replay isolation gate failed')
    require(main.get('risk_vectors_replayed') == 48 and main.get('legacy_preprocessing_states_reused') == 12
            and main.get('prepared_cells') == 3, 'legacy main replay coverage incomplete')
    require(neural.get('cells') == 6 and neural.get('neural_models') == 18
            and neural.get('neural_risk_vectors') == 144 and neural.get('reference_risk_vectors') == 18,
            'legacy neural replay coverage incomplete')
    require(neural['main_replay_evidence']['report_sha256'] == sha(main_path), 'neural replay refers to another main certificate')
    by_fold = {int(key): value for key, value in main['prepared_sha256_by_fold'].items()}
    require(set(by_fold) == set(FOLDS), 'legacy prepared map incomplete')
    return by_fold, {'source': legacy_source, 'main_report_sha256': sha(main_path),
        'main_replay_script_sha256': sha(ROOT / 'scripts/replay_phase26_main.py'),
        'neural_report_sha256': sha(neural_path), 'neural_replay_script_sha256': sha(ROOT / 'scripts/replay_phase26.py'),
        'raw_main_or_neural_replay_repeated': False}


def manual_projection(raw, state, *, fit_raw=None, validation_errors=None):
    raw = array(raw, 'saved 256D slot features')
    require(raw.ndim == 2 and raw.shape[1] == 256, 'slot feature width must remain 256')
    mean = array(state['pca_mean'], 'saved PCA mean', (256,))
    components = array(state['components'], 'saved PCA components', (16, 256))
    pc_mean = array(state['pca_score_mean'], 'saved PC mean', (16,))
    pc_sd = array(state['pca_score_sd'], 'saved PC SD', (16,))
    require((pc_sd > 0).all() and state.get('rank') == 16 and state.get('input_dim') == 256
            and state.get('whiten') is False and state.get('input_standardized_before_PCA') is False
            and state.get('svd_solver') == 'full', 'locked projection protocol changed')
    if fit_raw is not None:
        errors = [] if validation_errors is None else validation_errors
        fit_raw = array(fit_raw, 'matched fit slot features')
        require(state.get('fit_n') == len(fit_raw), 'projection fit count changed')
        compare(mean, fit_raw.mean(0), 'PCA mean must use matched fit patients only', errors)
        compare(components @ components.T, np.eye(16), 'PCA orthonormal basis', errors)
        scores = (fit_raw - mean) @ components.T
        compare(pc_mean, scores.mean(0), 'PC score matched-fit mean', errors)
        compare(pc_sd, scores.std(0, ddof=0), 'PC score matched-fit population SD', errors)
        rank = int(np.linalg.matrix_rank(fit_raw - mean))
        require(state.get('input_fit_rank') == rank and rank >= 16, 'recorded fit input rank changed')
        # This checks a saved representation's numerical rank. It does not
        # refit a PCA, choose a rank or change saved basis coefficients.
        standardized = (scores - pc_mean) / pc_sd
        compare(standardized.T @ standardized / len(standardized), np.eye(16), 'fit PC population covariance', errors)
    transformed = (((raw - mean) @ components.T) - pc_mean) / pc_sd
    require(np.isfinite(transformed).all(), 'manual projected values are nonfinite')
    return transformed


def replay(identity, private_root, report_dir):
    from src.phase27_readout import runtime, math as head_math
    from src.phase26_multiquery import runtime as prior_runtime, data as prior_data
    require(runtime.source() == identity and prior_runtime.source() == runtime.LEGACY_SOURCE,
            'new or immutable legacy training identity mismatch')
    old_prepared_map, evidence = legacy_evidence(runtime.LEGACY_SOURCE)
    private_root, report_dir = Path(private_root), Path(report_dir)
    registered_before = runtime.source_files(); legacy_before = prior_runtime.source_files()
    prepared_paths, checks, tracked = {}, [], {}
    for fold in FOLDS:
        path = prior_runtime.PRIVATE / 'prepared' / f'prototype_fold{fold}.pt'
        receipt_path = path.with_suffix('.json')
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('status') == 'COMPLETE' and receipt.get('source') == runtime.LEGACY_SOURCE
                and receipt.get('fold') == fold and receipt.get('sha256') == old_prepared_map[fold] == sha(path),
                'prepared main certificate/hash mismatch')
        prepared_paths[fold] = path
        tracked[f'prepared_{fold}'] = path; tracked[f'prepared_receipt_{fold}'] = receipt_path
    # Preflight all eighteen frozen encoders and every new feature/head hash
    # before selecting any saved feature/prediction arrays for calculations.
    for seed in SEEDS:
        for fold in FOLDS:
            folder = private_root / 'prototype' / f'seed{seed}_fold{fold}'
            done = runtime.verify_done(folder, identity, seed, fold)
            feature_path = private_root / 'features' / f'seed{seed}_fold{fold}.pt'
            feature_receipt_path = feature_path.with_suffix('.json')
            feature_receipt = json.loads(feature_receipt_path.read_text())
            for key, value in (('status', 'COMPLETE'), ('source', identity), ('stage', 'prototype'), ('seed', seed), ('fold', fold)):
                require(feature_receipt.get(key) == value, 'feature receipt metadata mismatch: ' + key)
            require(feature_receipt.get('sha256') == sha(feature_path)
                    and done.get('feature_receipt_sha256') == sha(feature_receipt_path), 'new feature receipt/artifact SHA mismatch')
            legacy_folder = prior_runtime.cell_dir(seed, fold)
            legacy_done = prior_runtime.verify_done(legacy_folder, runtime.LEGACY_SOURCE, seed, fold)
            legacy_checkpoint = legacy_folder / 'checkpoint.pt'
            require(feature_receipt.get('legacy_checkpoint_sha256') == legacy_done['checkpoint_sha256'] == sha(legacy_checkpoint)
                    and feature_receipt.get('prepared_sha256') == old_prepared_map[fold], 'feature source differs from verified frozen encoder/main')
            checks.append((seed, fold, folder, done, feature_path, feature_receipt_path, feature_receipt, legacy_checkpoint))
            for name, path in (('head', folder / 'head.pt'), ('head_done', folder / 'DONE_PRIVATE.json'),
                               ('features', feature_path), ('feature_receipt', feature_receipt_path),
                               ('legacy_encoder', legacy_checkpoint), ('legacy_done', legacy_folder / 'DONE_PRIVATE.json')):
                tracked[f'{name}_{seed}_{fold}'] = path
    before = {name: sha(path) for name, path in tracked.items()}
    forbidden_counts = {}

    def forbid(name):
        forbidden_counts[name] = 0
        def call(*args, **kwargs):
            forbidden_counts[name] += 1
            raise RuntimeError('forbidden API during saved-head replay: ' + name)
        return call
    import sklearn.decomposition
    from src.phase27_readout import pipeline
    from src.phase26_multiquery import train as prior_train, model as prior_model
    from src.phase10_trimodal import math as old_cox
    rows, all_risk_errors, all_projection_errors, reference_count, old_count = [], [], [], 0, 0
    with contextlib.ExitStack() as stack:
        for obj, attr, label in (
            (head_math, 'fit_offset_ridge', 'head.fit_offset_ridge'), (head_math, 'fit_projection', 'head.fit_projection'),
            (pipeline, 'fit_offset_ridge', 'pipeline.fit_offset_ridge'), (pipeline, 'fit_projection', 'pipeline.fit_projection'),
            (pipeline, 'fit_cell', 'pipeline.fit_cell'), (pipeline, 'extract', 'pipeline.extract'),
            (pipeline, 'extract_cell', 'pipeline.extract_cell'),
            (prior_model.MultiQuery, 'forward', 'legacy_model.forward'), (prior_train, '_forward', 'legacy_training.forward'),
            (prior_data, 'prepare', 'legacy.prepare'), (prior_data, 'build_prepared', 'legacy.build_prepared'),
            (prior_data, 'fit_ridge', 'legacy.fit_ridge'), (prior_data.old, 'fit_preprocessor', 'old.fit_preprocessor'),
            (prior_data.old.parent, 'endpoints', 'parent.endpoints'), (prior_data.old.parent, 'splits', 'parent.splits'),
            (old_cox, 'fit_ridge', 'old_cox.fit_ridge'),
            (sklearn.decomposition.PCA, 'fit', 'PCA.fit'), (torch.optim.Optimizer, '__init__', 'Optimizer.__init__')):
            stack.enter_context(patch.object(obj, attr, forbid(label)))
        # The serialized main artifact includes saved fit labels. They are
        # deserialized but no key/value from its supervision is requested.
        main = {fold: torch.load(path, map_location='cpu', weights_only=False) for fold, path in prepared_paths.items()}
        for seed, fold, folder, done, feature_path, feature_receipt_path, fr, legacy_checkpoint in checks:
            feature = torch.load(feature_path, map_location='cpu', weights_only=False)
            head = torch.load(folder / 'head.pt', map_location='cpu', weights_only=False)
            previous = torch.load(legacy_checkpoint, map_location='cpu', weights_only=False)
            saved_main = main[fold]
            for record, name in ((feature, 'features'), (head, 'head')):
                require(record.get('source') == identity and record.get('seed') == seed and record.get('fold') == fold
                        and record.get('stage') == 'prototype', name + ' payload identity mismatch')
                require(record.get('legacy_checkpoint_sha256') == fr['legacy_checkpoint_sha256']
                        and record.get('prepared_sha256') == old_prepared_map[fold], name + ' payload source lineage mismatch')
            require(head['feature_receipt_sha256'] == sha(feature_receipt_path), 'head refers to a different feature receipt')
            for key in ('fit', 'valid'):
                require(np.array_equal(feature[key], head[key]) and np.array_equal(feature[key], previous[key])
                        and np.array_equal(feature[key], saved_main[key]), 'new and original patient order differs: ' + key)
            fit, valid = head['fit'], head['valid']; nfit, nvalid = len(fit), len(valid)
            base_fit = array(feature['base_fit'], 'frozen fit main', (nfit,))
            require(np.array_equal(base_fit, saved_main['base'][fit]), 'fit offset is not immutable CF M0')
            for key, prepared_key in (('M0_valid', 'M0_valid'), ('CP_valid', 'CP_valid'), ('RNA_valid', 'RNA_valid')):
                require(np.array_equal(feature[key], saved_main[prepared_key]) and np.array_equal(head[key], saved_main[prepared_key]),
                        'heldout frozen reference differs: ' + key)
                reference_count += 1
            require(set(head['heads']) == set(ARMS) and set(feature['fit_features']) == set(ARMS)
                    and set(feature['valid_features']) == set(ARMS), 'complete three-arm feature/head inventory required')
            errors, projection_errors, arms = [], [], {}
            for arm in ARMS:
                record = head['heads'][arm]
                require(record.get('parameter_count') == 16, 'head must retain fixed sixteen parameters')
                optimizer = record['optimizer']
                require(optimizer.get('status') == 'CONVERGED' and optimizer.get('parameters') == 16
                        and optimizer.get('penalty') == 10. and optimizer.get('offset_trainable') is False
                        and optimizer.get('intercept') is False and optimizer.get('gradient_max', np.inf) <= 1e-7,
                        'saved convex optimizer gate/protocol changed')
                raw_fit = array(feature['fit_features'][arm], arm + ' matched fit slots', (nfit, 256))
                projection = record['projection']
                transformed = manual_projection(raw_fit, projection, fit_raw=raw_fit, validation_errors=projection_errors)
                coefficient = array(record['coef'], arm + ' fixed coefficient', (16,))
                fitted = base_fit + transformed @ coefficient
                error = compare(fitted, record['fit_risk'], arm + ' fit prediction', errors)
                require(set(feature['valid_features'][arm]) == set(CONTROLS) and set(record['valid_risks']) == set(CONTROLS),
                        'all five heldout feature conditions must share one head')
                for control in CONTROLS:
                    raw_control = array(feature['valid_features'][arm][control], arm + ' ' + control + ' slots', (nvalid, 256))
                    prediction = np.asarray(feature['M0_valid'], float) + manual_projection(raw_control, projection) @ coefficient
                    compare(prediction, record['valid_risks'][control], arm + ' ' + control + ' prediction', errors)
                require(np.array_equal(feature['old_risks'][arm], previous['arms'][arm]['risk'])
                        and np.array_equal(head['old_risks'][arm], previous['arms'][arm]['risk']),
                        'original bounded-head heldout reference changed')
                require(np.array_equal(feature['old_fit_risks'][arm], previous['arms'][arm]['fit_risk']),
                        'original bounded-head fit reference changed')
                old_count += 1
                arms[arm] = {'risk_vectors_replayed': 6, 'fit_projection_parameters_checked': True,
                             'input_fit_rank': projection['input_fit_rank'], 'maximum_absolute_risk_error': max(errors[-6:]),
                             'saved_optimizer_gradient_gate_checked_not_recomputed': True}
            all_risk_errors.extend(errors); all_projection_errors.extend(projection_errors)
            rows.append({'seed': seed, 'fold': fold, 'head_checkpoint_sha256': done['checkpoint_sha256'],
                'feature_receipt_sha256': sha(feature_receipt_path), 'feature_sha256': fr['sha256'],
                'legacy_checkpoint_sha256': fr['legacy_checkpoint_sha256'], 'prepared_sha256': fr['prepared_sha256'],
                'risk_vectors_replayed': len(errors), 'maximum_absolute_risk_error': max(errors), 'arms': arms})
    after = {name: sha(path) for name, path in tracked.items()}
    require(before == after and registered_before == runtime.source_files() and legacy_before == prior_runtime.source_files()
            and runtime.source() == identity and prior_runtime.source() == runtime.LEGACY_SOURCE,
            'a source or private input artifact changed during replay')
    require(len(all_risk_errors) == 108 and reference_count == 18 and old_count == 18 and not any(forbidden_counts.values()),
            'replay coverage or forbidden API accounting mismatch')
    output = {'status': 'PASS', 'version': VERSION, 'source': identity, 'stage': 'prototype', 'cells': 6, 'new_convex_heads': 18,
        'replay_script_sha256': sha(__file__), 'maximum_absolute_risk_error': max(all_risk_errors),
        'maximum_projection_statistic_error': max(all_projection_errors), 'absolute_tolerance': ATOL,
        'new_risk_vectors_replayed': len(all_risk_errors), 'main_reference_vectors_checked': reference_count,
        'original_bounded_head_references_checked': old_count, 'no_new_predictor_fits': 0, 'no_forward': 0,
        'new_predictor_fits': 0, 'neural_forward_calls': 0, 'optimizer_steps': 0, 'endpoint_API_reads': 0,
        'old_main_reused': True, 'forbidden_API_call_counts': forbidden_counts,
        'legacy_replay_evidence': evidence, 'private_artifacts_bytes_unchanged': True,
        'new_and_legacy_training_source_bytes_unchanged': True,
        'prepared_fit_supervision_bytes_deserialized_but_not_accessed_or_used': True,
        'heldout_labels_read': False, 'gradient_recomputed_from_labels': False,
        'projection_and_risk_arithmetic_independent_of_new_math_helpers': True,
        'patient_arrays_projection_states_coefficients_or_identifiers_public': False, 'rows': rows}
    report_dir.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(report_dir / 'REPLAY_AUDIT.json', output)
    print(json.dumps({key: value for key, value in output.items() if key not in ('rows', 'forbidden_API_call_counts')}, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--private-root', type=Path, default=Path('/root/autodl-tmp/phase27_readout_private'))
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/phase27_separated_readout')
    args = parser.parse_args()
    replay(args.source, args.private_root, args.report_dir)


if __name__ == '__main__':
    main()

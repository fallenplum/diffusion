"""No-fit, no-endpoint replay of the three prepared Phase25 main effects.

All expected receipt hashes are checked first. Saved train-local states are
transformed from raw covariates; teacher and full-fit risks are independently
reconstructed by coefficient multiplication. No array is written publicly.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_SOURCE = '97b6ab2c56eeece704a311218745e685790d87c9c5870d82f3a2eaa6dc2ca2d4'
VERSION = 'phase25_main_raw_covariate_saved_state_replay_v1'
ATOL = 1e-10


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def blocked(*args, **kwargs):
    raise RuntimeError('endpoint access or estimator fit is forbidden during main replay')


def moments(values):
    values = np.asarray(values, np.float64).reshape(-1)
    require(len(values) > 0 and np.isfinite(values).all(), 'invalid anonymous summary')
    return {'n': len(values), 'minimum': float(values.min()), 'median': float(np.median(values)),
            'maximum': float(values.max()), 'mean': float(values.mean()), 'SD': float(values.std())}


def compare(actual, expected, name, errors):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + ' shape drift')
    error = float(np.max(np.abs(actual - expected), initial=0.))
    require(np.isfinite(error) and error <= ATOL, name + ' replay drift')
    errors.append(error)
    return error


def check_scale(values, fit, saved, expected_fit_hash, label, errors):
    require(saved.get('fit_hash') == expected_fit_hash, label + ' fit hash drift')
    mean, sd = values[fit].mean(0), values[fit].std(0)
    compare(saved['mean'], mean, label + ' mean', errors)
    compare(saved['sd'], np.where(sd < 1e-8, 1., sd), label + ' SD', errors)


def state_scope_and_raw_statistics(raw, state, fit, groups, data, errors):
    """Recompute raw sufficient statistics; never refit a PCA or predictor."""
    fit_hash = data.array_hash(fit)
    require(np.array_equal(state['fit'], fit) and state['fit_hash'] == fit_hash, 'state patient scope drift')
    require(state['state_hash'] == data.state_hash({key: value for key, value in state.items() if key != 'state_hash'}),
            'saved complete preprocessing state hash drift')
    clinical = []
    for field in data.CLINICAL:
        values = np.asarray(raw['clinical_' + field]).astype(str)
        coding = state['clinical_coding'][field]
        if field == 'AGE':
            numeric = np.array([float(value) if value.replace('.', '', 1).isdigit() else np.nan for value in values])
            median = float(np.nanmedian(numeric[fit]))
            compare(coding['median'], median, 'clinical age median', errors)
            clinical.append(np.where(np.isfinite(numeric), numeric, median)[:, None])
        else:
            levels = np.unique(values[fit]).tolist()
            require(coding['levels'] == levels, 'clinical category vocabulary contains nonfit level')
            clinical.append(np.column_stack([values == level for level in levels]))
    clinical = np.concatenate(clinical, 1).astype(np.float64)
    check_scale(clinical, fit, state['scales']['C'], fit_hash, 'clinical', errors)
    check_scale(np.asarray(raw['static'], float), fit, state['scales']['S'], fit_hash, 'S32', errors)
    global_values = np.asarray(raw['global_matched_conch'], float)
    before = state['global']['before']
    check_scale(global_values, fit, before, fit_hash, 'global CONCH before PCA', errors)
    standardized = (global_values - before['mean']) / before['sd']
    compare(state['global']['mean'], standardized[fit].mean(0), 'saved PCA train mean', errors)
    components = np.asarray(state['global']['components'])
    compare(components @ components.T, np.eye(32), 'saved PCA orthonormal basis', errors)
    scores = (standardized - state['global']['mean']) @ components.T
    check_scale(scores, fit, state['scales']['T'], fit_hash, 'global CONCH PCA scores', errors)
    selected = state['RNA']
    require(selected['fit_hash'] == fit_hash, 'RNA selection fit hash drift')
    rna = np.asarray(raw['rna_genes'], float)
    train = rna[fit]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(train, 0)
    median = np.where(np.isfinite(median), median, 0.)
    filled = np.where(np.isfinite(train), train, median)
    mean, sd = filled.mean(0), filled.std(0)
    eligible = np.flatnonzero((np.isfinite(train).mean(0) >= .9) & (sd > 1e-8))
    symbols = np.asarray(raw['rna_gene_symbols']).astype(str)
    union = set().union(*(set(values) for values in groups.values()))
    annotation = np.array([symbol in union for symbol in symbols])
    pool = eligible[annotation[eligible]]
    indices = pool[np.argsort(-sd[pool], kind='stable')[:512]]
    if len(indices) < 512:
        remaining = eligible[~annotation[eligible]]
        indices = np.r_[indices, remaining[np.argsort(-sd[remaining], kind='stable')[:512-len(indices)]]]
    require(np.array_equal(selected['indices'], indices), 'RNA512 selection not derived from teacher train statistics')
    compare(selected['median'], median[indices], 'selected RNA train median', errors)
    require(selected['scale']['fit_hash'] == fit_hash, 'RNA scale fit hash drift')
    compare(selected['scale']['mean'], mean[indices], 'selected RNA train mean', errors)
    compare(selected['scale']['sd'], np.where(sd[indices] < 1e-8, 1., sd[indices]), 'selected RNA train SD', errors)
    membership = np.array([[symbol in members for symbol in symbols[indices]] for members in groups.values()], bool)
    require(np.array_equal(membership, selected['membership']) and membership.any(1).all(), 'true molecular membership changed')
    cna_symbols = np.asarray(raw['cna_gene_symbols']).astype(str)
    lookup = {symbol: index for index, symbol in enumerate(cna_symbols)}
    cna_indices = np.array([lookup.get(symbol, -1) for symbol in symbols[indices]], int)
    require(np.array_equal(cna_indices, state['CNA_indices']), 'CNA gene-symbol matching drift')
    cna = np.full((len(rna), 512), np.nan)
    available = cna_indices >= 0
    cna[:, available] = np.asarray(raw['cna_genes'], float)[:, cna_indices[available]]
    observed = np.isfinite(cna)
    denominator = observed.astype(float) @ membership.T.astype(float)
    safe = np.where(observed, cna, 0.)
    gain = np.maximum(safe, 0.) @ membership.T.astype(float) / np.maximum(denominator, 1.)
    loss = np.maximum(-safe, 0.) @ membership.T.astype(float) / np.maximum(denominator, 1.)
    program_size = membership.sum(1)
    rcov = np.isfinite(rna[:, indices]).astype(float) @ membership.T.astype(float) / program_size
    ccov = denominator / program_size
    for key, values in (('GAIN', gain), ('LOSS', loss), ('RCOV', rcov), ('CCOV', ccov)):
        check_scale(values, fit, state['scales'][key], fit_hash, key, errors)
    return {'fit_patients': len(fit), 'complete_fit_state_sha256': state['state_hash'],
            'selected_RNA_genes': len(indices), 'selected_RNA_observed_fraction_by_gene': moments(np.isfinite(rna[fit][:, indices]).mean(0)),
            'selected_RNA_observed_fraction_by_patient': moments(np.isfinite(rna[fit][:, indices]).mean(1)),
            'RNA_program_coverage_train': moments(rcov[fit]), 'CNA_program_coverage_train': moments(ccov[fit]),
            'RNA_fallback_nonHall_genes': selected['fallback_nonHall_genes'],
            'PCA_refitted': False, 'raw_statistics_and_scope_checked': True}


def replay(expected_source, private_root, report_dir):
    from src.phase25_local import data, runtime
    require(runtime.source() == expected_source, 'current training bytes differ from expected source')
    private_root, report_dir = Path(private_root), Path(report_dir)
    receipts = []
    for fold in (1, 2, 3):
        path = private_root / 'prepared' / f'prototype_fold{fold}.pt'
        receipt_path = path.with_suffix('.json')
        require(path.is_file() and receipt_path.is_file(), 'all three prepared receipts must exist before replay')
        receipt = json.loads(receipt_path.read_text())
        require(receipt.get('status') == 'COMPLETE' and receipt.get('source') == expected_source
                and receipt.get('stage') == 'prototype' and receipt.get('fold') == fold
                and receipt.get('convex_fits') == 5 and receipt.get('sha256') == sha(path),
                'prepared receipt, source or artifact hash mismatch')
        receipts.append((fold, path, receipt_path, receipt))
    # Guard every predictor/preprocessor fitting API that this module imports.
    # Raw covariate loading remains permitted; no survival endpoint is loaded.
    import src.phase10_trimodal.math as cox
    import src.phase13_rna.math as legacy_cox
    import sklearn.decomposition
    with contextlib.ExitStack() as stack:
        for obj, attr in ((data.parent, 'endpoints'), (data.parent, 'splits'),
                          (data, 'fit_ridge'), (data, 'fit_preprocessor'), (data, 'build_prepared'), (data, 'prepare'),
                          (cox, 'fit_ridge'), (legacy_cox, 'fit_ridge'), (legacy_cox, 'block_fit')):
            stack.enter_context(patch.object(obj, attr, blocked))
        stack.enter_context(patch.object(sklearn.decomposition.PCA, 'fit', blocked))
        raw = data.load_raw()
        groups = json.loads(data.GROUPS.read_text(encoding='utf8'))
        rows, teacher_rows, risk_errors, feature_errors, statistics_errors = [], [], [], [], []
        for fold, path, receipt_path, receipt in receipts:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            data.validate_prepared(saved, expected_source)
            # Canonical split metadata is feature-only. Do not call legacy
            # split/endpoints helpers merely to reconstruct those boundaries.
            split_path = data.parent.PARENT / 'formal/manifests' / f'R0_128_seed31_fold{fold}.json'
            canonical = json.loads(split_path.read_text())
            fit, valid, active = saved['fit'], saved['valid'], saved['active']
            require(np.array_equal(fit, canonical['inner_train']) and np.array_equal(valid, canonical['inner_val']),
                    'prepared partition differs from original frozen patient split')
            crossfit = np.zeros(len(saved['base']))
            seen = np.zeros(len(saved['base']), int)
            for teacher in saved['mainstate']['teachers']:
                train, held = teacher['train'], teacher['held']
                summary = state_scope_and_raw_statistics(raw, teacher['state'], train, groups, data, statistics_errors)
                features = data.transform(raw, teacher['state'], active)
                require(data.array_hash(features['X'][train]) == teacher['fit_feature_hash'], 'teacher fit feature bytes differ')
                require(data.array_hash(features['X'][held]) == teacher['held_feature_hash'], 'teacher held feature bytes differ')
                risk = features['X'][held] @ teacher['coefficients']
                compare(risk, teacher['risk_held'], 'teacher held risk', risk_errors)
                seen[held] += 1; crossfit[held] = risk
                teacher_rows.append({'fold': fold, 'teacher': teacher['teacher'], **summary,
                    'held_patients': len(held), 'nominal_parameters': len(teacher['coefficients']),
                    'constant_fit_columns': int((features['X'][train].std(0) < 1e-8).sum())})
            require((seen[fit] == 1).all() and not seen[valid].any(), 'replayed teacher OOF cover not exact fit-only')
            compare(crossfit[fit], saved['base'][fit], 'replayed crossfit offset', risk_errors)
            full = saved['mainstate']['full_state']
            summary = state_scope_and_raw_statistics(raw, full, fit, groups, data, statistics_errors)
            features = data.transform(raw, full, active)
            compare(features['X'], saved['mainstate']['full_X'], 'full-fit main feature matrix', feature_errors)
            compare(features['CP'], saved['mainstate']['CP_X'], 'full-fit CP feature matrix', feature_errors)
            compare(features['R'], saved['R'], 'full-fit RNA query coordinates', feature_errors)
            compare(features['C'], saved['C'], 'full-fit Clinical query coordinates', feature_errors)
            full_coef = saved['mainstate']['full_R0_coefficients']
            cp_coef = saved['mainstate']['CP_coefficients']
            valid_risk, fit_risk = features['X'][valid] @ full_coef, features['X'][fit] @ full_coef
            cp_risk = features['CP'][valid] @ cp_coef
            compare(valid_risk, saved['R0_valid'], 'full-fit heldout R0', risk_errors)
            compare(cp_risk, saved['CP_valid'], 'full-fit heldout CP', risk_errors)
            compare(fit_risk, saved['mainstate']['fullfit_risk_fit'], 'full-fit training risk diagnostic', risk_errors)
            compare(saved['base'][valid], valid_risk, 'heldout main offset', risk_errors)
            constant = features['X'][fit].std(0) < 1e-8
            constant_blocks = {name: int(constant[start:end].sum()) for name, (start, end) in features['blocks'].items()}
            cf_sd, full_sd = float(crossfit[fit].std()), float(fit_risk.std())
            compare(cf_sd, saved['scale'], 'fit-only offset scale', statistics_errors)
            rows.append({'fold': fold, **summary, 'valid_patients': len(valid),
                'R0_nominal_parameters': len(full_coef), 'CP_nominal_parameters': len(cp_coef),
                'constant_fit_columns': int(constant.sum()), 'nonconstant_fit_columns': int((~constant).sum()),
                'constant_fit_columns_by_block': constant_blocks,
                'nominal_columns_by_block': {name: end-start for name, (start, end) in features['blocks'].items()},
                'fit_RNA_all512_retained': features['R'].shape[1] == 512,
                'crossfit_risk_SD': cf_sd, 'fullfit_in_sample_risk_SD': full_sd,
                'crossfit_to_fullfit_SD_ratio': cf_sd / full_sd,
                'crossfit_fullfit_risk_correlation': float(np.corrcoef(crossfit[fit], fit_risk)[0, 1]),
                'crossfit_fullfit_RMS_difference': float(np.sqrt(np.mean((crossfit[fit]-fit_risk)**2))),
                'prepared_sha256': receipt['sha256'], 'receipt_sha256': sha(receipt_path),
                'canonical_split_sha256': sha(split_path)})
    require(runtime.source() == expected_source, 'training source changed during no-fit replay')
    output = {'status': 'PASS_NO_FIT_MAIN_REPLAY', 'version': VERSION, 'source': expected_source,
        'replay_script_sha256': sha(__file__), 'prepared_cells': 3, 'stored_convex_fits_referenced': 15,
        'new_predictor_fits': 0, 'new_PCA_fits': 0, 'endpoint_API_reads': 0,
        'heldout_labels_read': False, 'endpoint_and_fitter_APIs_hard_blocked': True,
        'teacher_held_risk_vectors_replayed': 9, 'full_R0_valid_vectors_replayed': 3,
        'full_CP_valid_vectors_replayed': 3, 'fit_crossfit_offset_vectors_replayed': 3,
        'fullfit_fit_diagnostic_vectors_replayed': 3, 'heldout_main_offset_vectors_checked': 3,
        'complete_teacher_or_full_preprocessing_states_checked': 12,
        'maximum_risk_replay_error': max(risk_errors, default=0.),
        'maximum_fullfit_feature_replay_error': max(feature_errors, default=0.),
        'maximum_raw_statistic_check_error': max(statistics_errors, default=0.),
        'absolute_tolerance': ATOL, 'PCA_bases_reused_with_hash_scope_and_orthogonality_checks_not_refitted': True,
        'patient_features_weights_predictions_indices_and_gene_lists_public': False,
        'fullfit_rows': rows, 'teacher_rows': teacher_rows,
        'crossfit_fullfit_scale_shift_reported_not_assumed_solved': True}
    report_dir.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(report_dir / 'MAIN_REPLAY_AUDIT.json', output)
    print(json.dumps({key: value for key, value in output.items() if key not in ('fullfit_rows', 'teacher_rows')}, indent=2))
    print('MAIN_REPLAY_PARAMETER_SCALE_SUMMARY', json.dumps([{key: row[key] for key in
        ('fold', 'R0_nominal_parameters', 'CP_nominal_parameters', 'constant_fit_columns', 'crossfit_risk_SD',
         'fullfit_in_sample_risk_SD', 'crossfit_to_fullfit_SD_ratio', 'crossfit_fullfit_risk_correlation')}
        for row in rows]), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=EXPECTED_SOURCE)
    parser.add_argument('--private-root', type=Path, default=Path('/root/autodl-tmp/phase25_private/b'))
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/phase25_b')
    args = parser.parse_args()
    replay(args.source, args.private_root, args.report_dir)


if __name__ == '__main__':
    main()

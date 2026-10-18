"""No-fit raw-transform replay of independent Phase26 CP/RNA main heads.

Reuses the immutable Phase25 raw preprocessing replay certificate. This audit
does not refit any preprocessing or predictor, load a survival endpoint API,
compute predictive performance, or publish risk/feature/patient arrays.
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
VERSION = 'phase26_independent_heads_saved_transform_replay_v1'
ATOL = 1e-10


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError('fitting, preparation and endpoint APIs are forbidden during main replay')


def compare(actual, expected, name, errors):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + ' shape mismatch')
    error = float(np.max(np.abs(actual - expected), initial=0.))
    require(np.isfinite(error) and error <= ATOL, name + ' mismatch')
    errors.append(error)
    return error


def old_main_certificate(old_source):
    path = ROOT / 'reports/phase25_b/MAIN_REPLAY_AUDIT.json'
    report = json.loads(path.read_text())
    script = ROOT / 'scripts/replay_phase25_main.py'
    require(report.get('status') == 'PASS_NO_FIT_MAIN_REPLAY' and report.get('source') == old_source,
            'immutable Phase25 main replay is unavailable')
    require(report.get('replay_script_sha256') == sha(script), 'Phase25 replay script bytes changed')
    require(report.get('prepared_cells') == 3 and report.get('complete_teacher_or_full_preprocessing_states_checked') == 12
            and report.get('new_predictor_fits') == 0 and report.get('new_PCA_fits') == 0
            and report.get('endpoint_API_reads') == 0, 'Phase25 preprocessing replay scope incomplete')
    require(report.get('maximum_risk_replay_error', np.inf) <= ATOL
            and report.get('maximum_fullfit_feature_replay_error', np.inf) <= ATOL
            and report.get('maximum_raw_statistic_check_error', np.inf) <= ATOL,
            'Phase25 replay numerical gate not satisfied')
    return report, {'source': old_source, 'report_sha256': sha(path), 'replay_script_sha256': sha(script),
                    'raw_preprocessing_statistics_refitted_or_retraversed': False}


def replay(identity, private_root, report_dir):
    from src.phase26_multiquery import data, runtime
    old = data.old
    require(runtime.source() == identity, 'current Phase26 training identity differs')
    require(old._runtime().source() == runtime.LEGACY_SOURCE == data.LEGACY_SOURCE, 'immutable Phase25 source changed')
    prior, prior_binding = old_main_certificate(runtime.LEGACY_SOURCE)
    old_by_fold = {row['fold']: row for row in prior['fullfit_rows']}
    old_teachers = {(row['fold'], row['teacher']): row for row in prior['teacher_rows']}
    require(set(old_by_fold) == {1, 2, 3} and len(old_teachers) == 9, 'Phase25 replay fold/state inventory incomplete')
    private_root, report_dir = Path(private_root), Path(report_dir)
    receipts = []
    # Verify every expected hash before loading any new prepared tensor file.
    for fold in (1, 2, 3):
        path = private_root / 'prepared' / f'prototype_fold{fold}.pt'
        receipt_path = path.with_suffix('.json')
        require(path.is_file() and receipt_path.is_file(), 'three prepared partitions required before main replay')
        receipt = json.loads(receipt_path.read_text())
        for key, value in (('status', 'COMPLETE'), ('source', identity), ('fold', fold), ('stage', 'prototype'),
                           ('legacy_source', runtime.LEGACY_SOURCE), ('new_convex_fits', 7), ('reused_full_CP_fits', 1)):
            require(receipt.get(key) == value, 'new prepared receipt mismatch: ' + key)
        require(receipt.get('sha256') == sha(path), 'new prepared artifact SHA mismatch')
        require(receipt.get('legacy_prepared_sha256') == old_by_fold[fold]['prepared_sha256'],
                'new prepared main refers to a different Phase25 raw preprocessing artifact')
        receipts.append((fold, path, receipt_path, receipt))
    import sklearn.decomposition
    import src.phase10_trimodal.math as cox
    import src.phase13_rna.math as legacy_cox
    risk_errors, feature_errors, rows, state_checks = [], [], [], 0
    with contextlib.ExitStack() as stack:
        for obj, name in ((old.parent, 'endpoints'), (old.parent, 'splits'),
                          (data, 'fit_ridge'), (data, 'prepare'), (data, 'build_prepared'),
                          (old, 'fit_ridge'), (old, 'prepare'), (old, 'build_prepared'), (old, 'fit_preprocessor'),
                          (cox, 'fit_ridge'), (legacy_cox, 'fit_ridge'), (legacy_cox, 'block_fit')):
            stack.enter_context(patch.object(obj, name, forbidden))
        stack.enter_context(patch.object(sklearn.decomposition.PCA, 'fit', forbidden))
        raw = old.load_raw()
        for fold, path, receipt_path, receipt in receipts:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            legacy, legacy_receipt = old.read_prepared(fold, 'prototype')
            data.validate_prepared(saved, legacy, legacy_receipt, identity)
            require(saved['legacy_receipt_sha256'] == sha(old.prepared_path(fold, 'prototype').with_suffix('.json')),
                    'immutable legacy receipt SHA changed')
            fit, valid, active = saved['fit'], saved['valid'], saved['active']
            require(np.array_equal(saved['fit_time'], legacy['fit_time'])
                    and np.array_equal(saved['fit_event'], legacy['fit_event']), 'saved fit-label lineage differs')
            position = np.full(len(saved['base']), -1, int); position[fit] = np.arange(len(fit))
            cp_cf, rna_cf, count = np.zeros(len(position)), np.zeros(len(position)), np.zeros(len(position), int)
            fold_errors = []
            teacher_rows = []
            for teacher, previous in zip(saved['mainstate']['teachers'], legacy['mainstate']['teachers']):
                train, held, state = teacher['train'], teacher['held'], teacher['state']
                require(not np.intersect1d(train, held).size and np.array_equal(np.union1d(train, held), np.sort(fit))
                        and not np.intersect1d(train, valid).size, 'new teacher patient isolation failed')
                state_hash = old.state_hash({key: value for key, value in state.items() if key != 'state_hash'})
                require(state['state_hash'] == previous['state']['state_hash'] == teacher['legacy_state_sha256']
                        == old_teachers[fold, teacher['teacher']]['complete_fit_state_sha256'] == state_hash,
                        'teacher saved preprocessing state lost its independent raw replay lineage')
                require(np.array_equal(state['fit'], train) and state['fit_hash'] == old.array_hash(train),
                        'teacher preprocessing fit mismatch')
                state_checks += 1
                p = position[train]
                require((p >= 0).all() and teacher['fit_endpoint_sha256']
                        == old.array_hash(np.c_[saved['fit_time'][p], saved['fit_event'][p]]),
                        'teacher fitted-label subset hash mismatch')
                features = old.transform(raw, state, active)
                for branch, key in (('CP', 'CP'), ('RNA', 'R')):
                    x = features[key]
                    require(teacher[branch + '_fit_feature_sha256'] == old.array_hash(x[train])
                            and teacher[branch + '_held_feature_sha256'] == old.array_hash(x[held]),
                            branch + ' saved teacher feature hash mismatch')
                    coefficient = teacher[branch + '_coefficients']
                    require(teacher[branch + '_coefficient_sha256'] == old.array_hash(coefficient),
                            branch + ' saved coefficient hash mismatch')
                    risk = x[held] @ coefficient
                    error = compare(risk, teacher[branch + '_risk_held'], branch + ' teacher held risk', risk_errors)
                    fold_errors.append(error)
                    if branch == 'CP':
                        cp_cf[held] = risk
                    else:
                        rna_cf[held] = risk
                count[held] += 1
                teacher_rows.append({'teacher': int(teacher['teacher']), 'fit_patients': len(train),
                    'held_patients': len(held), 'CP_parameters': len(teacher['CP_coefficients']),
                    'RNA_parameters': len(teacher['RNA_coefficients']), 'preprocessing_state_sha256': state_hash,
                    'fit_label_subset_and_feature_hashes_checked': True})
            require((count[fit] == 1).all() and not count[valid].any(), 'new teacher risk exact cover mismatch')
            for actual, expected, label in ((cp_cf[fit], saved['CP_cf_fit'], 'CF CP'),
                    (rna_cf[fit], saved['RNA_cf_fit'], 'CF RNA'),
                    (cp_cf[fit] + rna_cf[fit], saved['base'][fit], 'CF main sum')):
                fold_errors.append(compare(actual, expected, label, risk_errors))
            full_state = saved['mainstate']['full_state']
            full_hash = old.state_hash({key: value for key, value in full_state.items() if key != 'state_hash'})
            require(full_state['state_hash'] == legacy['mainstate']['full_state']['state_hash']
                    == old_by_fold[fold]['complete_fit_state_sha256'] == full_hash
                    and np.array_equal(full_state['fit'], fit), 'full-fit preprocessing lineage mismatch')
            state_checks += 1
            full = old.transform(raw, full_state, active)
            compare(full['CP'], legacy['mainstate']['CP_X'], 'legacy exact full CP features', feature_errors)
            compare(full['R'], legacy['R'], 'legacy exact full RNA features', feature_errors)
            cp_coef, rna_coef = saved['mainstate']['full_CP_coefficients'], saved['mainstate']['full_RNA_coefficients']
            require(np.array_equal(cp_coef, legacy['mainstate']['CP_coefficients'])
                    and np.array_equal(saved['CP_valid'], legacy['CP_valid']), 'independent full CP anchor changed')
            cp_valid, rna_valid = full['CP'][valid] @ cp_coef, full['R'][valid] @ rna_coef
            cp_fit, rna_fit = full['CP'][fit] @ cp_coef, full['R'][fit] @ rna_coef
            for actual, expected, name in ((cp_valid, saved['CP_valid'], 'full CP valid'),
                    (rna_valid, saved['RNA_valid'], 'full RNA valid'),
                    (cp_valid + rna_valid, saved['M0_valid'], 'full main valid sum'),
                    (cp_valid + rna_valid, saved['base'][valid], 'full main valid offset'),
                    (cp_fit, saved['mainstate']['full_CP_fit_risk'], 'full CP fit diagnostic'),
                    (rna_fit, saved['mainstate']['full_RNA_fit_risk'], 'full RNA fit diagnostic'),
                    (cp_fit + rna_fit, saved['mainstate']['full_main_fit_risk'], 'full main fit diagnostic')):
                fold_errors.append(compare(actual, expected, name, risk_errors))
            rows.append({'fold': fold, 'prepared_sha256': receipt['sha256'], 'receipt_sha256': sha(receipt_path),
                'legacy_prepared_sha256': legacy_receipt['sha256'], 'CP_exact_legacy_anchor': True,
                'CP_parameters': len(cp_coef), 'RNA_parameters': len(rna_coef),
                'teacher_rows': teacher_rows, 'teacher_risk_vectors_replayed': 6,
                'CF_branch_or_sum_vectors_replayed': 3, 'fullfit_valid_branch_or_sum_vectors_replayed': 4,
                'fullfit_fit_diagnostic_vectors_replayed': 3, 'risk_vectors_replayed': len(fold_errors),
                'maximum_absolute_risk_error': max(fold_errors),
                'fullfit_preprocessing_state_sha256': full_hash})
    require(runtime.source() == identity, 'training source changed during no-fit replay')
    output = {'status': 'PASS_NO_FIT_MAIN_REPLAY', 'version': VERSION, 'source': identity,
        'replay_script_sha256': sha(__file__), 'prepared_cells': 3,
        'prepared_sha256_by_fold': {str(row['fold']): row['prepared_sha256'] for row in rows},
        'legacy_main_replay_binding': prior_binding, 'legacy_preprocessing_states_reused': state_checks,
        'new_fit_calls': 0, 'new_predictor_fits': 0, 'new_PCA_fits': 0, 'endpoint_API_reads': 0,
        'saved_prepared_fit_labels_loaded_for_lineage_check': True,
        'valid_labels_loaded_or_used': False, 'fitter_prepare_PCA_and_endpoint_APIs_hard_blocked': True,
        'raw_transform_replayed_for_new_heads': True, 'old_raw_preprocessing_statistics_traversal_repeated': False,
        'maximum_absolute_risk_error': max(risk_errors, default=0.),
        'maximum_risk_replay_error': max(risk_errors, default=0.),
        'maximum_feature_replay_error': max(feature_errors, default=0.), 'absolute_tolerance': ATOL,
        'risk_vectors_replayed': len(risk_errors), 'patient_or_gene_arrays_or_model_weights_public': False,
        'fold_rows': rows}
    report_dir.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(report_dir / 'MAIN_REPLAY_AUDIT.json', output)
    print(json.dumps({key: value for key, value in output.items() if key != 'fold_rows'}, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--private-root', type=Path, default=Path('/root/autodl-tmp/phase26_multiquery_private'))
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/phase26_multiquery')
    args = parser.parse_args()
    replay(args.source, args.private_root, args.report_dir)


if __name__ == '__main__':
    main()

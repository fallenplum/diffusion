"""Independently reconstruct completed OOF metrics and training-budget gates."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def main(out):
    config = json.loads((out / 'protocol.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    assert digest(out / 'protocol.json') == manifest['protocol_sha256']
    assert digest(out / 'private_dataset.npz') == manifest['private_dataset_sha256']
    assert json.loads((out / 'encoder_audit.json').read_text())['status'] == 'PASS'
    assert all(json.loads((out / f'worker_{i}_DONE.json').read_text())['status'] == 'PASS' for i in range(3))
    data = dict(np.load(out / 'private_dataset.npz', allow_pickle=False))
    ids, folds = data['patient_ids'].astype(str), data['folds']
    assert len(ids) == len(set(ids)) == 512
    reported = pd.read_csv(out / 'metrics.csv').set_index(['seed', 'endpoint', 'method'])
    methods = config['methods'] + ['S4_wrong_patient_rna_token_prompt']
    maximum_error, oof_count, cell_count = 0.0, 0, 0
    saturated, inner_plateau, final_plateau, cap_bound = 0, 0, 0, 0
    for seed in config['seeds']:
        for endpoint in config['endpoints']:
            y = data['y_' + endpoint]
            valid = y >= 0
            for method in methods:
                all_p = np.empty((valid.sum(), int(y.max()) + 1))
                all_y, all_ids, predictions = [], [], []
                for fold in range(1, 6):
                    tag = f'{endpoint}__{method}__seed{seed}__fold{fold}'
                    test = np.flatnonzero(valid & (folds == fold))
                    with np.load(out / 'oof' / (tag + '.npz'), allow_pickle=False) as f:
                        assert np.array_equal(f['patient_ids'].astype(str), ids[test])
                        assert np.array_equal(f['y_true'], y[test])
                        p = f['probabilities'].copy()
                        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                        assert np.allclose(p.sum(1), 1, atol=1e-5)
                        all_y.extend(f['y_true'].tolist())
                        all_ids.extend(f['patient_ids'].astype(str).tolist())
                        predictions.append(p)
                    oof_count += 1
                    if method.startswith('S4_'):
                        continue
                    record = json.loads((out / 'cells' / (tag + '.json')).read_text())
                    assert record['status'] == 'PASS'
                    assert not record['outer_test_used_for_selection']
                    assert not record['inner_val_used_for_upstream_parameter_updates']
                    assert not record['selection']['validation_labels_used_for_parameter_updates']
                    assert record['protocol_sha256'] == manifest['protocol_sha256']
                    assert digest(out / 'oof' / (tag + '.npz')) == record['oof_sha256']
                    tr = np.flatnonzero(valid & (folds != fold))
                    rt, rv = next(StratifiedShuffleSplit(n_splits=1, test_size=.2,
                                       random_state=record['inner_split_seed']).split(tr, y[tr]))
                    it, iv = tr[rt], tr[rv]
                    assert set(it).isdisjoint(iv) and set(tr).isdisjoint(test)
                    assert record['selection']['inner_fit']['fit_n'] == len(it)
                    assert record['selection']['inner_fit']['fit_indices_sha256'] == indices_digest(it)
                    assert record['inner_val_indices_sha256'] == indices_digest(iv)
                    assert record['refit']['outer_fit']['fit_indices_sha256'] == indices_digest(tr)
                    assert record['outer_test_indices_sha256'] == indices_digest(test)
                    histories = record['selection']['history']
                    assert len(histories) == record['selection']['inner_epochs_run']
                    assert 30 <= len(histories) <= config['budget']['maximum_ceiling']
                    vals, best_ce, best_epoch = [], float('inf'), None
                    anchor_ce, anchor_ba, last_ce, last_ba = float('inf'), -1., 0, 0
                    for k, h in enumerate(histories):
                        assert h['epoch'] == k + 1
                        assert np.isfinite([h['train_ce'], h['inner_val_ce'], h['train_ba'], h['inner_val_ba']]).all()
                        window = histories[max(0, k - 4):k + 1]
                        ce = float(np.mean([x['inner_val_ce'] for x in window]))
                        ba = float(np.mean([x['inner_val_ba'] for x in window]))
                        assert abs(ce - h['smooth_val_ce']) < 1e-10
                        assert abs(ba - h['smooth_val_ba']) < 1e-10
                        if ce < best_ce - 1e-10:
                            best_ce, best_epoch = ce, k + 1
                        if ce < anchor_ce - config['budget']['meaningful_ce_improvement']:
                            anchor_ce, last_ce = ce, k + 1
                        if ba > anchor_ba + config['budget']['meaningful_ba_improvement']:
                            anchor_ba, last_ba = ba, k + 1
                    is_saturated = min(len(histories) - last_ce, len(histories) - last_ba) >= 20
                    assert is_saturated == record['selection']['validation_saturated']
                    assert best_epoch == record['selection']['selected_epoch']
                    assert best_epoch == len(record['refit']['history'])
                    nc = p.shape[1]
                    expected = {'S0_static_concepts': 33 * nc, 'S1_static_token_prompt': 12292 + 33 * nc,
                                'S2_rna_token_prompt': 20564 + 33 * nc, 'S3_static_concepts_plus_rna': 8208 + 49 * nc}
                    assert record['parameters'] == expected[method]
                    saturated += int(is_saturated)
                    inner_plateau += int(record['selection']['numerical_training_plateau'])
                    final_plateau += int(record['refit']['numerical_training_plateau'])
                    cap_bound += int(not is_saturated)
                    cell_count += 1
                assert len(all_ids) == len(set(all_ids)) == int(valid.sum())
                assert set(all_ids) == set(ids[valid])
                yy, pp = np.asarray(all_y), np.concatenate(predictions)
                pred = pp.argmax(1)
                nc = pp.shape[1]
                recomputed = {'balanced_accuracy': balanced_accuracy_score(yy, pred),
                              'macro_f1': f1_score(yy, pred, average='macro'), 'accuracy': accuracy_score(yy, pred),
                              'macro_ovr_auroc': roc_auc_score(yy, pp[:, 1]) if nc == 2 else roc_auc_score(yy, pp, multi_class='ovr')}
                for metric, value in recomputed.items():
                    error = abs(float(reported.loc[(seed, endpoint, method), metric]) - value)
                    maximum_error = max(maximum_error, error)
                    assert error < 1e-10
    assert cell_count == 240 and oof_count == 300 and len(reported) == 60
    result = {'status': 'PASS', 'patients': 512, 'fitted_cells': cell_count, 'complete_oof_files': oof_count,
              'true_seed_oof_metric_rows': len(reported), 'max_independent_metric_error': maximum_error,
              'inner_preprocessing_disjoint_split_and_epoch_checks': cell_count,
              'inner_validation_saturated_cells': saturated, 'inner_training_numerical_plateau_cells': inner_plateau,
              'refit_training_numerical_plateau_cells': final_plateau, 'budget_cap_bound_cells': cap_bound,
              'survival_columns_read': [], 'scope': 'computational/data/protocol validation; not a positive mechanism claim'}
    (out / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('/root/phase6b_token_budget_audit'))
    main(parser.parse_args().out)

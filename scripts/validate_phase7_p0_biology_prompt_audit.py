"""Independent OOF, transform-boundary and counterfactual checks; no refitting."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main(out):
    p = json.loads((out / 'protocol.json').read_text())
    m = json.loads((out / 'manifest.json').read_text())
    assert digest(out / 'protocol.json') == m['protocol_sha256']
    repo = Path('/root/diffusion_phase5_20261007')
    for s in m['sources']: assert digest(repo / s['path']) == s['sha256']
    for item in m['inputs']: assert digest(item['path']) == item['sha256']
    assert digest(out / 'private_dataset.npz') == m['private_dataset_sha256']
    data = dict(np.load(out / 'private_dataset.npz', allow_pickle=False))
    assert len(data['patient_ids']) == len(set(data['patient_ids'])) == 512
    methods = p['representation_methods'] + p['bridge_methods']
    expected = len(p['seeds']) * len(p['endpoints']) * 5 * len(methods)
    assert len(list((out / 'cells').glob('*.json'))) == len(list((out / 'oof').glob('*.npz'))) == expected == 600
    metrics = pd.read_csv(out / 'metrics.csv')
    assert len(metrics) == 120 and not metrics.duplicated(['seed', 'endpoint', 'method']).any()
    maxerr = 0.; maxcounter = 0.; checked_blocks = 0; maxiter = 0
    mask = np.asarray(m['concept_mask']); sh = np.asarray(m['shuffled_mapping'])
    bmask = mask > 0; bsh = sh > 0
    assert np.array_equal(bmask.sum(0), bsh.sum(0)) and np.array_equal(bmask.sum(1), bsh.sum(1))
    assert not np.array_equal(mask, sh)
    states = p['molecular_states']
    loc = {s: i for i, s in enumerate(data['rna_gene_symbols'])}
    for endpoint in p['endpoints']:
        y = data['y_' + endpoint]; valid = y >= 0
        nclass = len(m['targets'][endpoint]['class_counts'])
        for seed in p['seeds']:
            for method in methods:
                pred = np.full((512, nclass), np.nan); coverage = np.zeros(512, int)
                for fold in range(1, 6):
                    tag = f'{endpoint}__seed{seed}__fold{fold}__{method}'
                    cell = json.loads((out / 'cells' / (tag + '.json')).read_text())
                    assert cell['status'] == 'PASS' and cell['protocol_sha256'] == m['protocol_sha256']
                    assert not cell['outer_test_used_for_selection']
                    path = out / 'oof' / (tag + '.npz')
                    assert digest(path) == cell['oof_sha256']
                    z = dict(np.load(path, allow_pickle=False)); test = np.flatnonzero(valid & (data['folds'] == fold))
                    assert np.array_equal(z['test_indices'], test) and np.array_equal(z['y_true'], y[test])
                    assert str(z['endpoint']) == endpoint and int(z['seed']) == seed and int(z['fold']) == fold and str(z['method']) == method
                    assert str(z['protocol_sha256']) == m['protocol_sha256'] and np.array_equal(z['classes'], np.arange(nclass))
                    prob = z['probabilities']
                    assert prob.shape == (len(test), nclass) and np.isfinite(prob).all()
                    assert (prob >= 0).all() and (prob <= 1).all() and np.allclose(prob.sum(1), 1, atol=1e-10)
                    pred[test] = prob; coverage[test] += 1
                    if method != p['bridge_methods'][-1]:
                        assert cell['converged'] and cell['final_n_iter'] < 5000
                        assert len(cell['inner_trials']) == 3 and all(t['converged'] and t['n_iter'] < 5000 for t in cell['inner_trials'])
                        best = max(cell['inner_trials'], key=lambda t: (t['inner_ba'], -t['C']))
                        assert cell['selected_C'] == best['C']
                        maxiter = max(maxiter, cell['final_n_iter'], *[t['n_iter'] for t in cell['inner_trials']])
                    else:
                        assert cell['additional_fits'] == 0 and cell['pathology_unchanged'] and cell['additive_omics_unchanged']
                assert (coverage[valid] == 1).all() and (coverage[~valid] == 0).all()
                yp = pred[valid].argmax(1)
                values = {'balanced_accuracy': balanced_accuracy_score(y[valid], yp), 'accuracy': accuracy_score(y[valid], yp),
                          'macro_f1': f1_score(y[valid], yp, average='macro'),
                          'macro_ovr_auroc': roc_auc_score(y[valid], pred[valid, 1]) if nclass == 2 else roc_auc_score(y[valid], pred[valid], multi_class='ovr')}
                row = metrics[(metrics.endpoint == endpoint) & (metrics.seed == seed) & (metrics.method == method)].iloc[0]
                for k, v in values.items(): maxerr = max(maxerr, abs(float(row[k]) - v))
            for fold in range(1, 6):
                prefix = f'{endpoint}__seed{seed}__fold{fold}'
                q = dict(np.load(out / 'private_preprocessing' / (prefix + '.npz'), allow_pickle=False))
                tr = np.flatnonzero(valid & (data['folds'] != fold)); te = np.flatnonzero(valid & (data['folds'] == fold))
                split_seed = seed + fold * 10000 + int.from_bytes(hashlib.sha256(endpoint.encode()).digest()[:4], 'little') % 100000
                ii, vv = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=split_seed).split(tr, y[tr]))
                assert np.array_equal(q['tr'], tr) and np.array_equal(q['te'], te)
                assert np.array_equal(q['it'], tr[ii]) and np.array_equal(q['iv'], tr[vv])
                for suffix, idx in [('inner', q['it']), ('outer', tr)]:
                    assert np.array_equal(q['fit_idx_' + suffix], idx) and not (set(idx) & set(te))
                    # Recompute Top512 from raw train log expression, independently of runner helpers.
                    log = np.log2(data['raw_rna'].astype(float) + 1.)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        median = np.nanmedian(log[idx], axis=0)
                    median[~np.isfinite(median)] = 0
                    train = np.where(np.isfinite(log[idx]), log[idx], median)
                    variance = train.var(0)
                    usable = (np.isfinite(log[idx]).mean(0) >= .9) & (variance > 1e-8)
                    eligible = np.flatnonzero(usable)
                    keep = eligible[np.argsort(-variance[eligible], kind='stable')[:512]]
                    assert np.array_equal(q['raw_keep_' + suffix], keep)
                    expected_x = (train[:, keep] - train[:, keep].mean(0)) / train[:, keep].std(0)
                    assert np.allclose(q['raw_' + suffix][idx], expected_x, atol=1e-9)
                    for name in ['s', 'rna', 'random', 'cna']:
                        x = q[name + '_' + suffix][idx]
                        assert np.allclose(x.mean(0), 0, atol=1e-9) and np.allclose(x.std(0), 1, atol=1e-9)
                    perm = q['gene_permutation_' + suffix]; allowed = q['rna_usable_' + suffix]
                    assert len(set(perm[allowed])) == int(allowed.sum()) and allowed[perm[allowed]].all()
                    sets = [set(loc[g] for g in m['groups'][state] if g in loc and allowed[loc[g]]) for state in states]
                    shuffled = [set(perm[list(s)]) for s in sets]
                    assert all(len(a) == len(b) for a, b in zip(sets, shuffled))
                    assert all(len(sets[i] & sets[j]) == len(shuffled[i] & shuffled[j]) for i in range(8) for j in range(8))
                # Reconstruct counterfactuals with EXACT B2 coefficients; no classifier refit.
                S = q['s_outer'][te]; R = q['rna_outer']; C = q['cna_outer']
                base = np.concatenate([S, R[te], C[te]], 1)
                shifted = np.roll(te, 1)
                def probabilities(context):
                    raw_s = q['static_raw_outer'][te]
                    dr = .1 * raw_s * np.tanh(R[context] @ mask.T)
                    dc = .1 * raw_s * np.tanh(C[context] @ mask.T)
                    assert (np.abs(dr) <= .1 * np.abs(raw_s) + 1e-12).all()
                    assert (np.abs(dc) <= .1 * np.abs(raw_s) + 1e-12).all()
                    x = np.concatenate([base, dr / q['static_sd_outer'], dc / q['static_sd_outer']], 1)
                    logits = x @ q['b2_coef'].T + q['b2_intercept']
                    if nclass == 2:
                        pos = expit(logits[:, 0]); return np.column_stack([1 - pos, pos])
                    return softmax(logits, axis=1)
                for method, context in [(p['bridge_methods'][3], te), (p['bridge_methods'][-1], shifted)]:
                    z = dict(np.load(out / 'oof' / (prefix + '__' + method + '.npz'), allow_pickle=False))
                    maxcounter = max(maxcounter, float(np.max(np.abs(probabilities(context) - z['probabilities']))))
                    if method == p['bridge_methods'][-1]: assert np.array_equal(z['conditioning_indices'], shifted) and np.all(shifted != te)
                checked_blocks += 1
    assert maxerr < 1e-10 and maxcounter < 1e-10 and checked_blocks == 60
    result = {'status': 'PASS', 'patients': 512, 'fitted_outer_cells': 540, 'outer_oof_files': expected,
              'true_seed_oof_rows': 120, 'independently_checked_preprocessing_partitions': 120,
              'checked_counterfactual_blocks': checked_blocks, 'max_independent_metric_error': maxerr,
              'max_counterfactual_probability_error': maxcounter, 'max_classifier_iterations': maxiter,
              'all_classifier_fits_converged': True, 'source_and_input_hashes_match': True,
              'all_oof_patients_covered_once_per_method_seed': True, 'mapping_row_column_degree_preservation': True,
              'signature_size_overlap_preservation': True, 'survival_columns_read': [],
              'scope': 'implementation and OOF correctness; not a positive mechanism claim'}
    (out / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    main(ap.parse_args().out)

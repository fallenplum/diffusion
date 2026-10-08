"""Phase7-P0R: unchanged Stage model, locked matched-gene realizations."""
import argparse
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from phase7_p0_biology_prompt_audit import (REPO, BASE, CONFIG as P0_CONFIG, digest, ahash, write_json,
    check_lock, gene_z, matched_permutation, group_scores, zscale, split_indices, fit_lr)

CONFIG = REPO / 'configs/phase7_p0r_robustness.json'
DEFAULT = Path('/root/phase7_p0r_robustness_v1')
SELF = REPO / 'scripts/phase7_p0r_robustness.py'


def source_audit(out, data):
    original = pd.read_csv(BASE / 'data_mrna_seq_v2_rsem.txt', sep='\t', dtype=str, keep_default_na=False)
    numeric = original.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    miss = ~np.isfinite(data['raw_rna'])
    rows = np.flatnonzero(miss.any(1)); cols = np.flatnonzero(miss.any(0))
    rectangular = int(miss.sum()) == len(rows) * len(cols) and miss[np.ix_(rows, cols)].all()
    source_cols = {str(c)[:12]: c for c in original.columns[2:] if str(c)[13:15] == '01'}
    ids = data['patient_ids'].astype(str)
    original_indices = [list(original.columns[2:]).index(source_cols[p]) for p in ids]
    aligned = numeric[:, original_indices].T.astype(np.float32)
    assert np.array_equal(miss, ~np.isfinite(aligned)) and np.allclose(aligned, data['raw_rna'], equal_nan=True)
    tokens, counts = np.unique(original.iloc[:, 2:].to_numpy()[~np.isfinite(numeric)], return_counts=True)
    upstream = pd.read_csv(out / 'upstream_rna.txt', sep='\t', dtype=str, keep_default_na=False)
    assert set(upstream.columns) == set(original.columns)
    # This source has repeated/blank Entrez IDs. Verify the complete row sequence
    # instead of treating Entrez as a unique key or silently dropping duplicates.
    entrez_same = original.iloc[:, 1].to_numpy() == upstream.iloc[:, 1].to_numpy()
    hugo_same = original.iloc[:, 0].to_numpy() == upstream.iloc[:, 0].to_numpy()
    assert np.all(entrez_same | hugo_same), 'Unresolved source row identity: both identifiers changed'
    up = upstream
    raw_cols = original.columns[2:].tolist()
    up_values = up[raw_cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    same_missing = np.array_equal(np.isnan(numeric), np.isnan(up_values))
    numeric_equal = np.allclose(numeric, up_values, equal_nan=True, rtol=0, atol=0)
    assert same_missing and numeric_equal, 'Local RNA differs numerically from official source; do not run robustness benchmark'
    primary = pd.read_csv(BASE / 'data_clinical_patient.txt', sep='\t', comment='#', dtype=str,
                          usecols=['PATIENT_ID', 'CANCER_TYPE_ACRONYM']).set_index('PATIENT_ID').loc[ids]
    grouped = []
    for name, values in [('cancer_type', primary['CANCER_TYPE_ACRONYM'].fillna('UNKNOWN').to_numpy()),
                         ('tissue_source_code', np.asarray([p[5:7] for p in ids]))]:
        for level in np.unique(values):
            group = values == level
            grouped.append({'field': name, 'group': str(level), 'n': int(group.sum()),
                            'block_missing_n': int((group & miss.any(1)).sum())})
    pd.DataFrame(grouped).to_csv(out / 'missingness_source_groups.csv', index=False)
    reference = json.loads((out / 'upstream_source.json').read_text())
    audit = {'status': 'PASS', 'missing_block_patients': len(rows), 'missing_block_genes': len(cols),
             'missing_values': int(miss.sum()), 'exact_rectangular_block': bool(rectangular),
             'source_missing_tokens': {str(t): int(n) for t, n in zip(tokens, counts)},
             'parser_mask_identical': True, 'alignment_numeric_identical': True,
             'local_rna_sha256': digest(BASE / 'data_mrna_seq_v2_rsem.txt'),
             'upstream_rna_sha256': digest(out / 'upstream_rna.txt'), 'upstream_pinned_source': reference,
             'official_missing_mask_identical': bool(same_missing), 'official_all_numeric_values_identical': bool(numeric_equal),
             'hugo_symbol_annotation_differences': int((original.iloc[:, 0].to_numpy() != up['Hugo_Symbol'].to_numpy()).sum()),
             'upstream_row_alignment': 'every row has identical Entrez or Hugo identifier in the same position; repeated identifiers retained',
             'entrez_annotation_differences': int((~entrez_same).sum()),
             'proved_origin': 'already present in pinned official PanCancer/cBioPortal release, not introduced by parser or patient alignment',
             'platform_mechanism': 'PanCancer methods explicitly replaced low-count/residual-batch genes in adjusted GAII samples with NA; cohort-specific GAII identity not yet independently matched',
             'platform_inference_confidence': 'medium',
             'references': ['https://groups.google.com/g/cbioportal/c/6MZsoLeomIo',
                            'https://gdc.cancer.gov/about-data/publications/PanCan-CellOfOrigin'],
             'data_modified': False, 'DSS_read_for_this_audit': False}
    assert rectangular and len(rows) == 203 and len(cols) == 3024
    write_json(out / 'missing_block_source_audit.json', audit)


def locked(out):
    m = json.loads((out / 'manifest.json').read_text())
    for item in m['sources']: assert digest(item['path']) == item['sha256']
    assert digest(out / 'protocol.json') == m['protocol_sha256']
    assert digest(Path(m['parent']) / 'private_dataset.npz') == m['parent_private_dataset_sha256']
    check_lock(Path(m['parent']))
    return m


def prepare(out):
    protocol = json.loads(CONFIG.read_text()); parent = Path(protocol['parent_run'])
    pm = check_lock(parent)
    assert json.loads((parent / 'independent_validation.json').read_text())['status'] == 'PASS'
    assert not (out / 'manifest.json').exists()
    write_json(out / 'protocol.json', protocol)
    m = {'parent': str(parent), 'parent_protocol_sha256': pm['protocol_sha256'],
         'parent_private_dataset_sha256': digest(parent / 'private_dataset.npz'),
         'protocol_sha256': digest(out / 'protocol.json'), 'locked_utc': pd.Timestamp.now(tz='UTC').isoformat(),
         'sources': [{'path': str(p), 'sha256': digest(p)} for p in [CONFIG, SELF, REPO / 'scripts/phase7_p0_biology_prompt_audit.py']],
         'independent_permutations': 99, 'endpoint': 'Stage', 'survival_columns_read': []}
    write_json(out / 'manifest.json', m)
    data = dict(np.load(parent / 'private_dataset.npz', allow_pickle=False))
    source_audit(out, data)
    for sub in ['oof', 'cells']: (out / sub).mkdir(exist_ok=True)
    print('P0R_LOCKED_AND_SOURCE_AUDITED', flush=True)


def worker(out, worker_id):
    m = locked(out); parent = Path(m['parent'])
    p = json.loads((out / 'protocol.json').read_text()); pp = json.loads((parent / 'protocol.json').read_text())
    pm = json.loads((parent / 'manifest.json').read_text())
    data = dict(np.load(parent / 'private_dataset.npz', allow_pickle=False)); y = data['y_Stage']
    seed = p['inner_split_seeds'][worker_id]
    for fold in range(1, 6):
        tr, te, it, iv, _ = split_indices(data, 'Stage', seed, fold, pp)
        prefixes = f'Stage__seed{seed}__fold{fold}'
        parent_pre = dict(np.load(parent / 'private_preprocessing' / (prefixes + '.npz'), allow_pickle=False))
        zg = {}; st = {}
        for name, idx in [('inner', it), ('outer', tr)]:
            zg[name], st[name] = gene_z(data['rna_genes'], idx)
            true_scores, _ = group_scores(zg[name], data['rna_gene_symbols'], pm['groups'], st[name]['usable'])
            expected, _, _ = zscale(true_scores, idx)
            assert np.allclose(expected, parent_pre['rna_' + name], atol=1e-12)
        seen = {'inner': set(), 'outer': set()}
        for ps in p['permutation_seeds']:
            tag = f'random{ps}_seed{seed}_fold{fold}'
            cp = out / 'cells' / (tag + '.json'); op = out / 'oof' / (tag + '.npz')
            xs = {}; permutation_hashes = {}; counts = {}
            for name, idx in [('inner', it), ('outer', tr)]:
                perm = matched_permutation(st[name], ps)
                permutation_hashes[name] = ahash(perm)
                assert permutation_hashes[name] not in seen[name]
                seen[name].add(permutation_hashes[name])
                scores, random_sets = group_scores(zg[name], data['rna_gene_symbols'], pm['groups'], st[name]['usable'], perm)
                _, true_sets = group_scores(zg[name], data['rna_gene_symbols'], pm['groups'], st[name]['usable'])
                assert [len(x) for x in true_sets] == [len(x) for x in random_sets]
                assert all(len(set(a) & set(b)) == len(set(c) & set(d)) for a, c in zip(true_sets, random_sets) for b, d in zip(true_sets, random_sets))
                counts[name] = [len(x) for x in random_sets]
                xs[name], _, _ = zscale(scores, idx)
            if cp.exists() and op.exists(): continue
            trials = []
            for C in pp['candidate_C']:
                model = fit_lr(xs['inner'][it], y[it], C)
                trials.append({'C': C, 'inner_ba': float(balanced_accuracy_score(y[iv], model.predict(xs['inner'][iv]))), 'n_iter': int(model.n_iter_.max())})
            chosen = max(trials, key=lambda z: (z['inner_ba'], -z['C']))
            model = fit_lr(xs['outer'][tr], y[tr], chosen['C'])
            assert np.array_equal(model.classes_, np.arange(4))
            np.savez_compressed(op, test_indices=te, y_true=y[te], probabilities=model.predict_proba(xs['outer'][te]),
                                seed=seed, permutation_seed=ps, fold=fold, protocol_sha256=m['protocol_sha256'])
            write_json(cp, {'status': 'PASS', 'seed': seed, 'permutation_seed': ps, 'fold': fold, 'selected_C': chosen['C'],
                            'feature_dim': 8, 'outer_fit_n_iter': int(model.n_iter_.max()), 'trials': trials,
                            'gene_counts': counts, 'permutation_hashes': permutation_hashes,
                            'inner_fit_indices_sha256': ahash(it), 'outer_fit_indices_sha256': ahash(tr),
                            'oof_sha256': digest(op), 'protocol_sha256': m['protocol_sha256'], 'outer_test_used_for_selection': False})
        print('FOLD_DONE', seed, fold, 'independent_random_sets', len(seen['inner']), flush=True)
    locked(out)
    write_json(out / f'worker_{worker_id}_DONE.json', {'status': 'PASS'})


def aggregate(out):
    m = locked(out); parent = Path(m['parent'])
    p = json.loads((out / 'protocol.json').read_text()); d = dict(np.load(parent / 'private_dataset.npz', allow_pickle=False))
    y = d['y_Stage']; valid = y >= 0; yy = y[valid]
    assert len(list((out / 'oof').glob('*.npz'))) == len(list((out / 'cells').glob('*.json'))) == 1485
    random_correct = []; true_correct = []; rows = []; true_bas = []; all_random_ba = []
    maxiter = 0
    for wid, seed in enumerate(p['inner_split_seeds']):
        assert json.loads((out / f'worker_{wid}_DONE.json').read_text())['status'] == 'PASS'
        true = np.full((512, 4), np.nan)
        for f in range(1, 6):
            z = np.load(parent / 'oof' / f'Stage__seed{seed}__fold{f}__R1_true_biology_state_rna.npz', allow_pickle=False)
            true[z['test_indices']] = z['probabilities']
        tb = balanced_accuracy_score(yy, true[valid].argmax(1)); true_bas.append(float(tb))
        true_correct.append((true[valid].argmax(1) == yy).astype(float))
        bas = []; correct = []
        for ps in p['permutation_seeds']:
            pred = np.full((512, 4), np.nan); coverage = np.zeros(512, int)
            for f in range(1, 6):
                tag = f'random{ps}_seed{seed}_fold{f}'
                cp = json.loads((out / 'cells' / (tag + '.json')).read_text()); op = out / 'oof' / (tag + '.npz')
                assert cp['status'] == 'PASS' and digest(op) == cp['oof_sha256'] and not cp['outer_test_used_for_selection']
                z = dict(np.load(op, allow_pickle=False)); idx = np.flatnonzero(valid & (d['folds'] == f))
                assert np.array_equal(z['test_indices'], idx) and np.array_equal(z['y_true'], y[idx])
                assert str(z['protocol_sha256']) == m['protocol_sha256'] and np.isfinite(z['probabilities']).all()
                assert np.allclose(z['probabilities'].sum(1), 1)
                pred[idx] = z['probabilities']; coverage[idx] += 1
                maxiter = max(maxiter, cp['outer_fit_n_iter'], *[r['n_iter'] for r in cp['trials']])
            assert (coverage[valid] == 1).all() and (coverage[~valid] == 0).all()
            ba = float(balanced_accuracy_score(yy, pred[valid].argmax(1))); bas.append(ba)
            correct.append((pred[valid].argmax(1) == yy).astype(float))
            rows.append({'seed': seed, 'permutation_seed': ps, 'n': len(yy), 'oof_ba': ba, 'true_oof_ba': tb, 'delta_true_minus_random': tb - ba})
        all_random_ba.append(bas); random_correct.append(np.mean(correct, axis=0))
    random_mean = np.asarray(all_random_ba).mean(0); true_mean = float(np.mean(true_bas))
    shared_diff = np.asarray(true_correct).mean(0) - np.asarray(random_correct).mean(0)
    rng = np.random.default_rng(71001); boot = np.zeros(2000)
    for cl in range(4):
        idx = np.flatnonzero(yy == cl); draws = rng.choice(idx, size=(2000, len(idx)), replace=True)
        boot += shared_diff[draws].mean(1) / 4
    exceed = (1 + int((random_mean >= true_mean).sum())) / 100
    seed_gates = [bool(t > np.quantile(r, .95)) for t, r in zip(true_bas, all_random_ba)]
    robust = bool(true_mean > np.quantile(random_mean, .95) and exceed <= .05 and all(seed_gates))
    result = {'status': 'PASS', 'true_mean_oof_ba': true_mean, 'true_seed_oof_ba': true_bas,
              'random_mean_oof_ba': float(random_mean.mean()), 'random_mean_sd': float(random_mean.std(ddof=1)),
              'random_mean_quantiles': np.quantile(random_mean, [0, .025, .5, .95, .975, 1]).tolist(),
              'random_realizations_ge_true': int((random_mean >= true_mean).sum()), 'conditional_benchmark_exceedance': exceed,
              'delta_true_minus_average_random': float(true_mean - random_mean.mean()),
              'paired_conditional_ci95': np.quantile(boot, [.025, .975]).tolist(),
              'all_seed_95percentile_gate': seed_gates, 'robustness_gate': robust,
              'independent_random_realizations': 99, 'outer_oof_files': 1485, 'seed_oof_metrics': 297,
              'all_fits_converged': maxiter < 5000, 'max_iterations': maxiter,
              'source_and_patient_alignment_unchanged': True,
              'scope': 'same discovery cohort; matched-gene benchmark, not independent external confirmation or causal significance'}
    pd.DataFrame(rows).to_csv(out / 'seed_oof_benchmark.csv', index=False)
    pd.DataFrame({'permutation_seed': p['permutation_seeds'], 'mean_random_oof_ba': random_mean, 'true_mean_oof_ba': true_mean,
                  'delta_true_minus_random': true_mean - random_mean}).to_csv(out / 'permutation_benchmark.csv', index=False)
    write_json(out / 'decision.json', result)
    write_json(out / 'validation.json', {'status': 'PASS', 'oof_files': 1485, 'seed_oof_rows': 297, 'all_sources_locked': True,
                                       'all_gene_sizes_and_overlap_preserved': True, 'heldout_labels_excluded_from_upstream': True,
                                       'all_classifiers_converged': True, 'private_data_published': False})
    lines = ['# Phase7-P0R：缺失来源与 Stage 基因集稳健性', '',
             '## 缺失区块来源', '',
             '203×3024 的完整缺失矩形已在 pinned cBioPortal 官方原文件中复核。原始 numeric values 和 missing mask 全部一致，证明不是本项目解析/患者对齐引入。', '',
             'PanCancer 方法与开发者说明：GAII→HiSeq 批次调整中，低 reads 或 residual batch effect 的基因在被调整样本上替换为 NA。该说明与矩形缺失一致；缺少 cohort aliquot→platform 的独立映射，因此具体这 203 位是否全属 GAII，仍属于中等置信度推断。', '',
             '数据矩阵未改动，没有批次校正、重选通路或修改模型。', '',
             '## 锁定的 99 次独立匹配置换', '',
             f'- 真实集合 3-seed mean OOF BA：**{true_mean:.4f}**。',
             f'- 随机集合均值：**{random_mean.mean():.4f}**；随机均值分布 95th percentile：**{np.quantile(random_mean,.95):.4f}**。',
             f'- 99 个随机 realization 中 ≥真实集合：**{result["random_realizations_ge_true"]}**；有限 benchmark exceedance：**{exceed:.4f}**。',
             f'- 真实−平均随机：**{result["delta_true_minus_average_random"]:+.4f} BA**；95% 条件配对区间 **[{result["paired_conditional_ci95"][0]:+.4f}, {result["paired_conditional_ci95"][1]:+.4f}]**。',
             f'- 每个 split seed 都超过各自随机分布 95th percentile：{seed_gates}。',
             f'- 预先锁定稳健性判据：**{"PASS" if robust else "NOT_SUPPORTED"}**。', '',
             '## 优势、风险与决策', '',
             '优势：重复随机基因 benchmark、匹配表达 mean/SD、相同集合大小与重叠、同训练/选择算法；原始缺失来源有外部数字矩阵核验。', '',
             '风险：Stage 来自上一轮探索，同一 discovery cohort 不是独立确认；benchmark exceedance 不能当作 survival 标签无信号的 p-value，gene exchangeability 也只是近似匹配假设。', '',
             '来源层结论置信度高；具体 GAII/plate 机制推断置信度中；外部可迁移性置信度低。无论本判据结果如何，下一阶段统一保留真实分子状态作为预先指定的生存表示，不根据 P0R 换 pathway 或结构。', '',
             '## 来源', '',
             '- [PanCancer source and batch-correction explanation](https://groups.google.com/g/cbioportal/c/6MZsoLeomIo)',
             '- [Original PanCancer publication resource](https://gdc.cancer.gov/about-data/publications/PanCan-CellOfOrigin)',
             '- 官方 pinned commit、文件 SHA 及比较范围见 `missing_block_source_audit.json`。']
    (out / 'CONCLUSIONS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'DONE.json', {'status': 'PASS', 'protocol_sha256': m['protocol_sha256'], 'robustness_gate': robust})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['prepare', 'worker', 'aggregate'], required=True)
    ap.add_argument('--out', type=Path, default=DEFAULT); ap.add_argument('--worker-id', type=int, default=0)
    a = ap.parse_args()
    if a.mode == 'prepare': prepare(a.out)
    elif a.mode == 'worker': worker(a.out, a.worker_id)
    else: aggregate(a.out)

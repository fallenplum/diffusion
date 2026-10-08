"""Post-validation descriptive coverage and mechanism summary; fits no models."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(out):
    check = json.loads((out / 'independent_validation.json').read_text()); assert check['status'] == 'PASS'
    manifest = json.loads((out / 'manifest.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    coverage = {}
    with np.load(out / 'private_dataset.npz', allow_pickle=False) as data:
        for key in ['raw_rna', 'raw_cna']:
            finite = np.isfinite(data[key]); row = finite.mean(1); col = finite.mean(0)
            coverage[key] = {'patient_finite_fraction_quantiles': np.quantile(row, [0, .01, .05, .5, .95, 1]).tolist(),
                             'gene_finite_fraction_quantiles': np.quantile(col, [0, .01, .05, .5, .95, 1]).tolist(),
                             'patients_below_90pct_coverage': int((row < .9).sum()),
                             'genes_below_90pct_coverage': int((col < .9).sum())}
    cells = [json.loads(p.read_text()) for p in (out / 'cells').glob('*.json')]
    selected = pd.read_csv(out / 'selection_summary.csv')
    first = [c for c in cells if c['method'] == protocol['representation_methods'][0]]
    counts = []
    for state in protocol['molecular_states']:
        row = {'state': state, 'nominal_fixed_gene_count': len(manifest['groups'][state])}
        for branch in ['rna', 'shuffled_rna', 'cna']:
            values = [c[part]['counts'][state][branch] for c in first for part in ['inner_preprocessing', 'outer_preprocessing']]
            row[branch + '_effective_min'] = min(values); row[branch + '_effective_max'] = max(values)
        counts.append(row)
    pd.DataFrame(counts).to_csv(out / 'effective_gene_set_coverage.csv', index=False)
    budget = {'outer_fits': 540, 'inner_fits': 1620, 'counterfactual_additional_fits': 0,
              'all_2160_classifier_fits_converged': check['all_classifier_fits_converged'],
              'maximum_iterations': check['max_classifier_iterations'], 'iteration_limit': 5000,
              'rna_genes_require_train_coverage': .9, 'true_random_set_size_overlap_preserved': True,
              'top512_pre_standardization_variance': True, 'correction_lambda': .1,
              'random_gene_replicates': 1, 'random_mapping_replicates': 1,
              'post_validation_summary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    budget['selected_C_counts'] = {method: {str(C): int(n) for C, n in frame.selected_C.value_counts().items()}
                                  for method, frame in selected.groupby('method')}
    budget['selection_grid_boundary'] = 'Fixed before OOF; many lower-bound selections may indicate need for stronger regularization in a future separately locked protocol. No retuning was performed after outer results.'
    for name, value in [('coverage_audit.json', coverage), ('budget_audit.json', budget)]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    agg = pd.read_csv(out / 'aggregate_metrics.csv'); pair = pd.read_csv(out / 'paired_comparisons.csv')
    lines = ['# Phase7-P0 数据与机制补充解读', '',
             '此文件在独立验证 PASS 后生成；只汇总已有正式结果，不重新拟合模型或更改超参数。', '',
             '## 有效覆盖', '',
             f'RNA 中 {coverage["raw_rna"]["patients_below_90pct_coverage"]}/512 位患者覆盖低于 90%，最低覆盖 {coverage["raw_rna"]["patient_finite_fraction_quantiles"][0]:.2%}；{coverage["raw_rna"]["genes_below_90pct_coverage"]} 个基因总体覆盖低于 90%。当前评分只纳入各训练分区覆盖 ≥90% 且非恒定的基因，真实/随机组有效大小严格一致。', '',
             '这会改变固定 gene set 的有效成员，不能只引用原始 gene-set 大小；逐状态覆盖范围见 `effective_gene_set_coverage.csv`。CNA 是 signed log2 dosage，基因组缺失和扩增可能在均值评分中抵消；R3 的结果仅针对该明确定义的 8D CNA 表示。', '',
             '## 每个终点的证据', '']
    for endpoint in protocol['endpoints']:
        section = pair[pair.endpoint == endpoint].set_index('comparison')
        methods = agg[agg.endpoint == endpoint].sort_values('mean_oof_ba', ascending=False)
        winner = methods.iloc[0]
        lines += [f'### {endpoint}', '', f'- 点估计最高：{winner.method}，3-seed mean OOF BA={winner.mean_oof_ba:.4f}。这是当前方法集内的排序，不是外部泛化证据。']
        for name, meaning in [('R1-R0', '真 RNA 状态相对 Raw RNA'), ('R1-R2', '真基因集相对匹配随机基因集'), ('R3-R1', 'CNA 额外增益'),
                              ('B2-B1raw', 'Bridge 相对 Raw RNA additive'), ('B2-B1mol', 'Bridge 相对同源分子 additive'),
                              ('B2-B3', '真实关系相对打乱 mapping'), ('B2-B4', '正确配对相对 conditioning-only 错配')]:
            r = section.loc[name]
            lines.append(f'- {meaning}：ΔBA={r.mean_delta_ba:+.4f}，98.75% 条件区间 [{r.ci98_75_low:+.4f}, {r.ci98_75_high:+.4f}]。')
        lines += ['']
    lines += ['## 解释边界', '',
              '更低维表示的收益可来自正则化与平均降噪；只有真实集合优于匹配随机集合，才进一步支持本轮 gene-set 组织结构的价值。RNA/CNA 存在疾病信息和 mapping 真正有效是两个独立命题。', '',
              '如果桥不优于同源 additive，当前结果没有证明需要概念交互。如果 matched 不优于 wrong-patient conditioning，缺少患者特异语义解释的关键对照。零附近差异和区间跨零不证明所有生物学先验无效。', '',
              'T/N/M/Stage 标签不是病例级形态概念标注；即使辅助分类提高，仍需独立形态或外部队列验证，不能自动外推 DSS。']
    (out / 'MECHANISM_SUMMARY_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    final = ['## 本轮主要发现', '',
             '1. **Stage 的基因集语义有局部提示：**R1−R2=+0.1009 BA，98.75% 条件区间 [0.0224, 0.1712]；R1−R0=+0.0346，但区间跨零。该证据只针对一套固定 gene-set 评分和一个匹配随机 realization，不能称为跨队列可迁移性已成立。',
             '2. **CNA 增益尚未得到支持：**R3−R1 在 T 为负，在 N/Stage 为正，在 M 接近零；四个终点的条件区间均跨零。不能据此断言 CNA 无信息，尤其当前是 gene-set signed dosage 均值，未测试 arm burden。',
             '3. **概念桥机制没有通过必要对照：**N 的 B2 点估计最好（0.4773），但 B3 也为 0.4740，B2−B3=+0.0033 且区间跨零；真实 mapping 没有稳定超过匹配打乱 mapping。T/M/Stage 的静态 B0 点估计仍高于 B2。',
             '4. **本轮不能归因于线性优化预算不足：**2160 次 classifier fits 全部收敛，最大 531/5000 iterations。这个结论仅针对本轮凸线性拟合，不评价尚未测试的可学习 prompt。',
             '5. **下一步边界：**保留 Stage 的局部分子状态线索，当前不扩大 molecular prompt/桥模型；优先明确分子测量覆盖和外部形态证据。BA 改善不能自动转化为 DSS C-index 改善。', '',
             '固定 C 网格及每种方法的边界选择次数见 `budget_audit.json`。若大量选择 C=0.1，则更强正则化没有在本轮被覆盖；这是搜索范围的限制，不能在看过 outer OOF 后回调网格并将结果算作同一轮确认性实验。', '',
             '详细有效基因数与每个终点的配对解释见 `effective_gene_set_coverage.csv` 和 `MECHANISM_SUMMARY_ZH.md`。']
    report = out / 'CONCLUSIONS_ZH.md'
    original = report.read_text(encoding='utf-8').split('\n## 本轮主要发现\n')[0]
    report.write_text(original.rstrip() + '\n\n' + '\n'.join(final) + '\n', encoding='utf-8')
    print(json.dumps({'coverage_audit': coverage, 'maximum_iterations': check['max_classifier_iterations']}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    main(ap.parse_args().out)

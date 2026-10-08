"""Generate a scoped Chinese report only after independent artifact validation."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main(out):
    check = json.loads((out / 'independent_validation.json').read_text()); assert check['status'] == 'PASS'
    p = json.loads((out / 'protocol.json').read_text()); m = json.loads((out / 'manifest.json').read_text())
    a = pd.read_csv(out / 'aggregate_metrics.csv'); c = pd.read_csv(out / 'paired_comparisons.csv')
    selections = pd.read_csv(out / 'selection_summary.csv')
    audit = json.loads((out / 'encoder_audit.json').read_text())
    lines = ['# Phase7-P0：Biology-Grounded Prompt Feasibility Audit', '',
             '**正式完成：512 患者，3 个 inner-split seeds × 5 固定 outer folds，600 份 held-out 预测，120 个真正 seed-level OOF 指标。只使用 T/N/M/Stage，不读取 DSS。**', '',
             '## 1. 本轮检验什么', '',
             '本轮把两个问题分开：固定生物学基因集是否带来优于同容量随机基因集的疾病状态表示；保留静态 TITAN 与分子加性主效应后，候选分子—形态关系是否再提供可泛化交互。', '',
             '**这是固定有界证据交互的可行性审计，不是 RNA-conditioned TITAN token prompt，也没有训练 TITAN/text encoder。没有验证完整可学习 molecular prompt 架构。**', '',
             'RNA 状态为固定 8 组 Hallmark 的训练集基因 z-score 均值；CNA 为相同基因集的 signed log2 dosage 均值，两者保留为 16 个独立维度。CNA 值不代表 pathway activation，也没有声称已计算 chromosome-arm burden。', '',
             'RNA Top512 按训练集 log2 表达插补后的方差选择，再标准化。随机组通过训练集表达均值/SD 的 5×5 分箱内全局基因置换生成，保留基因集大小、重叠结构与测量覆盖；不使用标签。', '',
             '## 2. 数据与拟合审计', '',
             f'- RNA 原始维度 {audit["rna"]["gene_rows"]}；CNA 原始维度 {audit["cna"]["gene_rows"]}；各 512 位患者均有 primary tumor 测量。',
             f'- RNA 非有限比例 {1-audit["rna"]["finite_fraction"]:.4%}，全缺失 gene rows {audit["rna"]["all_missing_gene_rows"]}；CNA 非有限比例 {1-audit["cna"]["finite_fraction"]:.4%}，全缺失 gene rows {audit["cna"]["all_missing_gene_rows"]}。',
             f'- Batch-normalized RNA 含 {audit["rna"]["negative_finite_count"]} 个小负值；有限值均 > −1，因此原样保留并 log2(x+1)，没有截断到零。',
             f'- RNA/CNA 相同选中 aliquot 列表：{audit["same_selected_aliquot_lists_rna_cna"]}/512；重复基因符号在状态评分中先按 gene symbol 合并。',
             '- 所有插补、Top512、基因 z-score、状态 scaling 和随机集合匹配只用 inner-train；锁定 C 后在完整 outer-train 用相同算法重拟合。C 仅比较 0.1/1/10；不存在 epoch 选择或 in-sample stacking。',
             f'- 全部分类器收敛，最大 {check["max_classifier_iterations"]}/5000 iterations；验证/测试没有参与 upstream 表示拟合。',
             f'- 独立重算 {check["independently_checked_preprocessing_partitions"]} 个预处理分区；OOF 指标最大误差 {check["max_independent_metric_error"]:.3g}；B2/B4 counterfactual 误差 {check["max_counterfactual_probability_error"]:.3g}。', '',
             '## 3. Molecular Representation Audit', '',
             '| Endpoint | R0 Raw RNA 512D | R1 真基因集 RNA 8D | R2 匹配随机 8D | R3 RNA+CNA 16D |',
             '|---|---:|---:|---:|---:|']
    def val(e, method):
        row = a[(a.endpoint == e) & (a.method == method)].iloc[0]
        return f'{row.mean_oof_ba:.4f} ± {row.sd_oof_ba:.4f}'
    for e in p['endpoints']:
        lines.append('| ' + e + ' | ' + ' | '.join(val(e, method) for method in p['representation_methods']) + ' |')
    lines += ['', '以上为三个 seed 各自拼接五折 OOF 后的 BA 均值 ± seed SD；seed 只改变 inner split，三者来自同一队列。', '',
              'RNA 表示的语义证据看 R1−R2；信息保留看 R1−R0；CNA 是否额外有效看 R3−R1，独立判断，不把 CNA 增益作为 RNA 生物语义成立的前提。内部 CV 不能证明跨队列迁移。', '',
              '## 4. Concept Bridge Audit', '',
              '| Method | 定义 |', '|---|---|',
              '| B0 | 静态 TITAN 32 concepts |',
              '| B1raw | 静态 concepts + Raw RNA Top512 |',
              '| B1mol | 静态 concepts + RNA/CNA 16 molecular states |',
              '| B2 | 完整 B1mol 主效应 + 有界 RNA/CNA concept-specific 交互，λ=0.1 固定 |',
              '| B3 | B2 相同容量，用行/列度数均相同的打乱 prior mapping |',
              '| B4 | 原封不动复用 B2 权重，仅错配交互 conditioning；病理和分子加性主效应均匹配 |', '',
              '交互为 λ·static cosine·tanh(molecular state × allowed mask)。RNA/CNA 交互分开，不把 dosage 与 expression 平均成“激活”。每条证据修正绝对值 ≤ 0.1×|静态 cosine|；这约束输入证据，不保证最终 head 的 logit 修正同样有界。交互使用静态分数同一 scaler，不单独标准化，避免取消 λ。', '',
              f'固定 prior 只含 {m["mask_edge_count"]} 个候选关联，{m["unmapped_concepts"]} 个概念无交互边；unknown 不强行映射。它们有外部来源和置信度，但没有本队列形态真值验证。', '',
              '| Endpoint | B0 | B1raw | B1mol | B2 | B3 | B4 |', '|---|---:|---:|---:|---:|---:|---:|']
    for e in p['endpoints']:
        lines.append('| ' + e + ' | ' + ' | '.join(val(e, method) for method in p['bridge_methods']) + ' |')
    lines += ['', '## 5. 同 seed 配对差值与不确定性', '',
              'Class-stratified patient bootstrap 2000 次，对三个 seed 使用同一批 resampled patients，再平均 seed 差值。98.75% 区间是固定已训练模型条件下的描述性区间；未重训练整条 pipeline，不是 familywise 显著性保证。', '',
              '| Endpoint | Comparison | 平均 ΔBA | 98.75% CI | 三个 seed 点差均正 |', '|---|---|---:|---:|---|']
    for r in c.itertuples(index=False):
        lines.append(f'| {r.endpoint} | {r.comparison} | {r.mean_delta_ba:+.4f} | [{r.ci98_75_low:+.4f}, {r.ci98_75_high:+.4f}] | {r.all_seed_delta_positive} |')
    def passed(e, names):
        rows = c[(c.endpoint == e) & c.comparison.isin(names)]
        return len(rows) == len(names) and bool((rows.ci98_75_low > 0).all()) and bool(rows.all_seed_delta_positive.all())
    rep_pass = [e for e in p['endpoints'] if passed(e, ['R1-R0', 'R1-R2'])]
    cna_pass = [e for e in p['endpoints'] if passed(e, ['R3-R1'])]
    bridge_pass = [e for e in p['endpoints'] if passed(e, ['B2-B0', 'B2-B1raw', 'B2-B1mol', 'B2-B3', 'B2-B4'])]
    lines += ['', '## 6. 机制判断', '',
              f'- 真实 RNA 状态同时通过 Raw RNA 与匹配随机组比较的 endpoint：**{", ".join(rep_pass) or "无"}**。',
              f'- CNA 额外增益通过保守区间判断的 endpoint：**{", ".join(cna_pass) or "无"}**。',
              f'- Bridge 五项必要对照同时通过的 endpoint：**{", ".join(bridge_pass) or "无"}**。', '',
              '**区间跨零表示证据不足，不等于证明没有疾病信号。B2 高于 B0 不能排除 additive 信息或额外参数容量；B2 高于 B4 也必须结合 B1mol/B3，才能进一步讨论 pairing 与 mapping。任何这些辅助任务结果均不能外推 DSS C-index。**', '',
              '若五项对照未同时支持，当前结果不足以扩大 molecular prompt 模型，也不能因此否定所有医学先验或完整可学习桥。若 true RNA 不优于匹配随机组，不能把低维压缩收益归因于 Hallmark 语义。', '',
              '## 7. 优势与劣势（风险）', '',
              '| 优势 | 劣势 / 风险 |', '|---|---|',
              '| 明确保留静态预训练病理证据和分子主效应 | 小型固定桥只覆盖部分候选联系，不代表完整可学习 prompt |',
              '| 真/随机基因集、degree-matched mapping、conditioning-only mismatch 相互区分 | 仅一个随机 gene permutation 和一个随机 mapping realization，不能概括所有随机先验 |',
              '| 同算法 inner/final fit，所有分类器收敛，120 份真正 OOF | 单队列、一次 inner split、三个相关 seed；训练方差未被 bootstrap 完整覆盖 |',
              '| 关系来源和 uncertainty 可审计 | bulk EMT 混有 CAF、gene-dose 均值可抵消扩增/缺失；病例级 morphology 未标注 |', '',
              '## 8. 置信度及后续边界', '',
              '- **高：**本轮计算、患者对齐、OOF 指标、源文件哈希与错配控制的实现完整性。',
              '- **中：**当前队列、指定评分方法与 prior 下的性能比较。',
              '- **低：**外部泛化、分子—形态因果联系、DSS 改善。', '',
              'Clinical Text 为可选补充，本轮暂缓：尚未建立不包含 T/N/M/Stage、且通过字段审计的文本输入。', '',
              '## 拓展思考', '',
              '将“压缩有效”“真实基因集语义有效”“跨模态配对有效”作为不同结论；只给得到必要对照支持的层级命名。', '',
              '## 你可能想知道', '',
              '这轮只有线性分类器需要训练，在服务器并行用 CPU 更合适；没有为提高显存占用而训练额外网络。TITAN 权重保留在服务器，所有患者预测和大文件不进入 GitHub。', '',
              '## 文献依据', '',
              '- [MSigDB Hallmark collections](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)：状态定义来源；评分方法为本轮明确规定的 gene-z 均值，并非 GSEA/GSVA。',
              '- [CRC consensus molecular subtypes](https://www.nature.com/articles/nm.3967)：免疫、mesenchymal/TGF-β、stromal context 的动机，不能直接作为局部形态标签。',
              '- [Microdissected budding RNA profiles](https://www.nature.com/articles/bjc2016382)：budding/EMT 候选联系。',
              '- [EMT and stromal expression confounding](https://www.nature.com/articles/s41467-021-22800-1)：bulk EMT 的解释限制。',
              '- [cBioPortal file formats](https://github.com/cBioPortal/cbioportal/blob/master/docs/File-Formats.md)：log2 CNA 编码解释。', '',
              '- [cBioPortal data engineer response](https://groups.google.com/g/cbioportal/c/prm-A08br9Y)：PanCancer batch-normalized RSEM 中的负值也存在于其发布来源。', '',
              '协议和 prior 修正在查看有效 OOF 之前锁定。旧 v3 因 Top512 和错配控制错误作废，其不完整结果没有用于本轮调参或机制判断。']
    text = '\n'.join(lines) + '\n'
    (out / 'CONCLUSIONS_ZH.md').write_text(text, encoding='utf-8')
    (out / 'RESULTS.md').write_text(text, encoding='utf-8')
    print(json.dumps({'representation_gate': rep_pass, 'cna_gate': cna_pass, 'bridge_gate': bridge_pass}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    main(ap.parse_args().out)

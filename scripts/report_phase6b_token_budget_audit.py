"""Write evidence-limited Chinese conclusions after the complete independent audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(out):
    check = json.loads((out / 'independent_validation.json').read_text())
    assert check['status'] == 'PASS' and check['fitted_cells'] == 240 and check['complete_oof_files'] == 300
    methods = pd.read_csv(out / 'aggregate_metrics.csv')
    paired = pd.read_csv(out / 'aggregate_comparisons.csv')
    seed_pairs = pd.read_csv(out / 'paired_comparisons_by_seed.csv')
    selections = pd.read_csv(out / 'selection_summary.csv')
    budget = pd.read_csv(out / 'budget_summary.csv')
    diagnostics = pd.read_csv(out / 'mechanism_diagnostics.csv')
    manifest = json.loads((out / 'manifest.json').read_text())
    endpoints = ['T', 'N', 'M', 'Stage']
    labels = {'S0_static_concepts': 'S0 静态概念', 'S1_static_token_prompt': 'S1 静态 token prompt',
              'S2_rna_token_prompt': 'S2 RNA 动态 token prompt', 'S3_static_concepts_plus_rna': 'S3 概念 + RNA 加性',
              'S4_wrong_patient_rna_token_prompt': 'S4 错误患者 RNA'}
    curve_rows = []
    for path in sorted((out / 'cells').glob('*.json')):
        cell = json.loads(path.read_text())
        h = cell['selection']['history']
        select = cell['selection']['selected_epoch']
        curve_rows.append({'endpoint': cell['endpoint'], 'method': cell['method'], 'seed': cell['seed'], 'fold': cell['fold'],
                           'train_ce_epoch3': h[2]['train_ce'], 'terminal_train_ce': h[-1]['train_ce'],
                           'train_ce_decline_after3': h[2]['train_ce'] - h[-1]['train_ce'],
                           'terminal_val_ce_minus_selected': h[-1]['inner_val_ce'] - h[select - 1]['inner_val_ce'],
                           'selected_epoch': select, 'inner_observed_epochs': len(h)})
    pd.DataFrame(curve_rows).to_csv(out / 'convergence_diagnostics.csv', index=False)
    convergence = pd.DataFrame(curve_rows).groupby(['endpoint', 'method'], as_index=False).agg(
        mean_train_ce3=('train_ce_epoch3', 'mean'), mean_terminal_train_ce=('terminal_train_ce', 'mean'),
        mean_terminal_val_ce_minus_selected=('terminal_val_ce_minus_selected', 'mean'))
    convergence.to_csv(out / 'aggregate_convergence.csv', index=False)
    mechanism_rows, budget_rows = [], []
    for endpoint in endpoints:
        records = {}
        for comp in ['S2-S1', 'S2-S3', 'S2-S4']:
            r = paired.loc[(paired.endpoint == endpoint) & (paired.comparison == comp)].iloc[0]
            sp = seed_pairs.loc[(seed_pairs.endpoint == endpoint) & (seed_pairs.comparison == comp)]
            records[comp] = {'mean_delta_ba': float(r.mean_delta_ba), 'positive_seeds': int((sp.delta_balanced_accuracy > 0).sum()),
                             'all_seed_ci_positive': bool((sp.ci98_75_low > 0).all())}
        all_positive = all(x['mean_delta_ba'] > 0 for x in records.values())
        all_cis_positive = all(x['all_seed_ci_positive'] for x in records.values())
        mechanism_rows.append({'endpoint': endpoint, 'comparisons': records, 'all_three_mean_deltas_positive': all_positive,
                               'all_three_all_seed_ci_positive': all_cis_positive})
        for method in labels:
            comp = 'selected_budget-minus-three_epochs/' + method
            r = paired.loc[(paired.endpoint == endpoint) & (paired.comparison == comp)].iloc[0]
            sp = seed_pairs.loc[(seed_pairs.endpoint == endpoint) & (seed_pairs.comparison == comp)]
            selection_method = 'S2_rna_token_prompt' if method.startswith('S4_') else method
            ep = selections.loc[(selections.endpoint == endpoint) & (selections.method == selection_method), 'selected_epoch']
            budget_rows.append({'endpoint': endpoint, 'method': method, 'mean_delta_ba': float(r.mean_delta_ba),
                                'selected_epoch_median': float(ep.median()), 'selected_more_than3_cells': int((ep > 3).sum()),
                                'positive_seeds': int((sp.delta_balanced_accuracy > 0).sum()),
                                'all_seed_ci_positive': bool((sp.ci98_75_low > 0).all())})
    decision = {'status': 'COMPLETE', 'training_budget': budget_rows, 'mechanism': mechanism_rows,
                'inner_validation_saturated_cells': check['inner_validation_saturated_cells'],
                'budget_cap_bound_cells': check['budget_cap_bound_cells'],
                'limitations': ['same cohort previously examined', 'one inner validation split',
                                'three seeds are not three independent cohorts', 'bootstrap conditional on fitted pipelines',
                                'no morphology ground-truth annotations', 'no DSS training or evaluation']}
    (out / 'decision_audit.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# Phase6B：完整训练预算与动态概念机制审计', '',
             '**范围：512 位患者、3 seeds × 5 folds × 4 endpoints；240 个拟合 cell、300 份完整 OOF 文件。目标仅为 T/N/M/Stage，没有使用 DSS。**', '',
             '## 核心结论', '',
             f'- 内部验证达到预置饱和条件：**{check["inner_validation_saturated_cells"]}/240**；到 960 epoch 仍受上限约束：**{check["budget_cap_bound_cells"]}/240**。',
             f'- 内部训练损失达到预置数值平台：**{check["inner_training_numerical_plateau_cells"]}/240**。验证收益饱和与训练损失数值收敛是不同的判断。',
             '- 每个 seed 先拼接五折 OOF，再计算指标；以下均是三个完整 seed OOF 的均值，没有把 15 个 fold cell 叫 seed_mean。', '',
             '## 1. 泛化表现', '', '| Endpoint | S0 | S1 | S2 | S3 | S4 |', '|---|---:|---:|---:|---:|---:|']
    for endpoint in endpoints:
        vals = [methods.loc[(methods.endpoint == endpoint) & (methods.method == method), 'mean_oof_ba'].iloc[0] for method in labels]
        lines.append('| ' + endpoint + ' | ' + ' | '.join(f'{x:.4f}' for x in vals) + ' |')
    lines += ['', '指标为 balanced accuracy。S1 是普通静态 token tuning，S2 才包含患者 RNA 配对；S3 保留直接 RNA 加性通路，S4 使用相同 S2 参数，仅替换为错误患者 RNA。', '',
              '## 2. 训练预算是否充分', '',
              '每个 outer-train 内固定一次分层 80/20 split，RNA 插补/Top512/标准化与所有可训练参数仅在 inner-train 拟合。最多五个 epoch 的 trailing mean 验证 CE 选 epoch；epoch 1–4 同样可选。至少观察 30 epoch，内部验证 CE 与 BA 都连续 20 epoch 无实质改善才停止；未饱和依次扩展 120 → 240 → 960。', '',
              '配置锁定后按所选 epoch，从同一 final seed 在完整 outer-train 重新训练，学习率、batch、优化器和模型容量不变。最早的 warmup selection 限制在任何本轮 OOF 指标汇总前被修正；未使用该版本的测试结果。', '',
              '| Endpoint | Model | Selected epoch median (range) | Observed median | Val saturated /15 | Train numerical plateau /15 |',
              '|---|---|---:|---:|---:|---:|']
    for r in budget.itertuples(index=False):
        lines.append(f'| {r.endpoint} | {labels[r.method]} | {r.selected_epoch_median:g} ({r.selected_epoch_min:g}–{r.selected_epoch_max:g}) | {r.inner_epochs_median:g} | {r.validation_saturated_cells} | {r.inner_numerical_plateau_cells} |')
    lines += ['', '### 相对已修复的固定 3 epoch：同 seed OOF 配对', '',
              '| Endpoint | Model | Delta BA | Positive seeds /3 | Every seed CI >0 |', '|---|---|---:|---:|---:|']
    for r in budget_rows:
        lines.append(f'| {r["endpoint"]} | {labels[r["method"]]} | {r["mean_delta_ba"]:+.4f} | {r["positive_seeds"]} | {r["all_seed_ci_positive"]} |')
    lines += ['', '**判读规则：**所选 epoch >3 只能说明内部选择偏好更长训练；还需要 OOF 配对改善才能说旧预算限制了泛化。所选 epoch <3 则指向更早停止，而非继续加训练。这里同时改变了可用训练预算和 epoch selection，不能把全部差异归因于纯粹增加 epoch。', '',
              '## 3. patient-specific 机制的必要控制', '',
              '| Endpoint | S2−S1 | S2−S3 | S2−S4 | All three means >0 | All three: every seed CI >0 |', '|---|---:|---:|---:|---|---|']
    for r in mechanism_rows:
        delta = [r['comparisons'][name]['mean_delta_ba'] for name in ['S2-S1', 'S2-S3', 'S2-S4']]
        lines.append('| ' + r['endpoint'] + ' | ' + ' | '.join(f'{x:+.4f}' for x in delta) + f' | {r["all_three_mean_deltas_positive"]} | {r["all_three_all_seed_ci_positive"]} |')
    passing = [r['endpoint'] for r in mechanism_rows if r['all_three_mean_deltas_positive']]
    strong = [r['endpoint'] for r in mechanism_rows if r['all_three_all_seed_ci_positive']]
    lines += ['', f'- 三项均值控制同时通过的 endpoint：**{", ".join(passing) if passing else "无"}**。',
              f'- 三项控制且每个 seed 的区间下界均 >0：**{", ".join(strong) if strong else "无"}**。',
              '- S2>S0 单独只能说明整个可训练 prompt pipeline 有收益；S2>S1 才排除普通 prompt tuning 的解释；S2>S3 才超出直接增加 RNA 的解释；S2>S4 才表明患者配对有用。配对有用仍不自动等价于医学概念语义正确。', '',
              '## 4. 结构层面的诊断', '',
              '| Endpoint | Gate normalized entropy | Patient gate mean SD | Dynamic-to-static concept cosine | Score effective rank |', '|---|---:|---:|---:|---:|']
    for endpoint in endpoints:
        d = diagnostics.loc[diagnostics.endpoint == endpoint]
        vals = [d[x].mean() for x in ['gate_normalized_entropy', 'gate_patient_mean_std', 'mean_concept_cosine_to_static', 'concept_score_effective_rank']]
        lines.append('| ' + endpoint + ' | ' + ' | '.join(f'{x:.4f}' for x in vals) + ' |')
    lines += ['',
              '这些诊断不使用标签，限每个测试折的前 64 人。低患者 gate 变化提示弱条件化；低概念余弦提示明显语义漂移；它们均不是机制成立或医学语义正确的独立证明。', '',
              '**结构推断的边界：**S2 的 RNA 只能经四个 prompt basis 的混合与 32 个 pathology cosine 传给分类器，S3 有直接 RNA 通路。如果 S2 持续弱于 S3，这与该中介通路限制信息传递相容；本轮没有专门隔离该瓶颈的消融，不能把它当作已经证明的因果原因。', '',
              '### 与 CTF_2 原文的范围差异', '',
              '原文 GCSP 联合使用 concept-specific global prompt、互补模态的 context prompt 和双模态 shared prompt，并在两个 VLM 分支对称运行。本轮按照已指定的低容量方案，只检验 RNA 驱动的 context prompt → 冻结 TITAN text encoder → pathology concept scores；没有原文全部 GCSP 组件、对称双 VLM 或 Clinical conditioner。这个范围差异由 [原论文 §3.2–3.3](https://openreview.net/pdf/4b3fab8d88280ab054b8f5acb353e783df7c6d2d.pdf) 与本轮源码直接核对。', '',
              '因此，负结果可以否定当前受检配置的性能优势，不能直接否定完整 CTF、医学先验或全部 tri-modal co-adaptation。反过来，这个范围差异也不能用来忽略当前配置的负结果。', '',
              '## 优势与风险', '',
              '| 优势 | 劣势 / 风险 |', '|---|---|',
              '| 官方 TITAN 冻结编码及 prompt 梯度、mask、template aggregation 已校验 | cosine/gate 的数值有效不保证医学语义保持 |',
              '| inner-only preprocessing / training / epoch selection 后重新训练 | 单个 inner split 的小类别验证噪声仍存在 |',
              '| 完整五折 OOF、同 seed 配对与错误患者控制 | 三个 seeds 使用同一 cohort，不能当作三次独立人群验证 |',
              '| 延长预算并区分 train plateau 与 validation saturation | 到上限未饱和的 cell 必须标记预算受限；有限训练不能证明全局最优 |', '',
              '## 置信度与后续判断', '',
              '- **高：**完整矩阵、OOF 指标重算、数据边界和 epoch selection 的计算审计结果；对应 `independent_validation.json`。',
              '- **中：**当前 cohort、当前五个控制及固定超参数下的性能排序；跨 cohort 和其他架构尚未确认。',
              '- **低：**将失败归因于某一个结构瓶颈、将成功解释为医学语义成立、或外推到 DSS。当前实验未直接检验这些命题。', '',
              '## 拓展思考', '',
              '优先根据 S2−S1/S3/S4 的完整证据判断是否值得保留动态概念中介，而不是只看 S2−S0。若仅静态 tuning 受益，应将它与 patient-specific co-adaptation 分开叙述。', '',
              '## 你可能想知道', '',
              '后续若要证明医学解释，仍需独立形态标注或外部队列验证；本轮没有看 DSS，也不能回答 DSS 监督事件是否足够。', '',
              '## 审计与文件', '',
              f'- 独立重算最大指标误差：{check["max_independent_metric_error"]:.3g}。',
              '- TITAN 权重、私有 OOF 和可恢复 checkpoints 留在服务器；仓库只包含源码、protocol、无患者身份的 aggregate / learning curves / 审计报告。',
              '- 98.75% paired bootstrap 区间为逐比较描述性区间，条件于已拟合的 pipelines，没有全家族错误率保证，也未完整包含模型重训的不确定性。',
              '- 当前 cohort 已被历史研究查看，本轮是内部机制与预算审计，不是未触碰的外部确认。',
              '- model selection 的偏差与方差风险参见 [Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)。', '']
    (out / 'CONCLUSIONS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'mean_mechanism_endpoints': passing, 'all_seed_ci_mechanism_endpoints': strong}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('/root/phase6b_token_budget_audit'))
    main(parser.parse_args().out)

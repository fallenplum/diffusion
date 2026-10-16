"""Generate the final Chinese report only after every real audit has passed.

Reads aggregate audit tables and private completion/resource receipts, never
patient predictions, gene identities, or model tensors.  The original cap40,
cap160 sensitivity and A3F bug-fix outputs remain separate and immutable.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE = Path('/root/autodl-tmp/trimodal_interaction_v1_private')
ARM_NAMES = {
    'A0': '共同全输入凸 Ridge Cox', 'A1': 'CP/CR/RP 全两两分支',
    'A2': '常规 Clinical-conditioned attention',
    'A3': '原 Diff 适配（RMSNorm affine 缺失）',
    'A3F': '修复 affine 的官方 Diff 核心适配',
    'A4': 'mixed，无 gate', 'A5': 'mixed + gate',
    'A6': '仅交互通道错患者 RNA',
    'A7': '含 mixed 幅度的中心化方向控制',
    'F0': 'pre-attention FiLM', 'PR': 'PR attention + Clinical late',
}


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('必需的真实审计文件尚未生成：' + str(path))
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_table(path, columns):
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    require(not missing, f'{path}: 缺列 {sorted(missing)}')
    require(len(frame) > 0, f'{path}: 空表')
    return frame


def finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def number(value, digits=6):
    if isinstance(value, (bool, np.bool_)):
        return '是' if value else '否'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError('不能在报告中静默格式化非有限值')
        if abs(float(value)) < .5*10**(-digits):return '0'
        return f'{float(value):.{digits}f}'.rstrip('0').rstrip('.')
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows, digits=6):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join(['---'] * len(headers)) + '|']
    lines += ['| ' + ' | '.join(number(v, digits) for v in row) + ' |' for row in rows]
    return '\n'.join(lines)


def safe_write(path, value):
    path = Path(path)
    text = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            if isinstance(value, dict) else value.rstrip() + '\n')
    # Public reporting is aggregate-only.  A report correction cannot silently
    # introduce a patient barcode, token, or private key into the export.
    protected = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|'
                           r'-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    require(not protected.search(text), '输出出现受保护的患者标识/凭据')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf8')
    temporary.replace(path)


def load_stage(path, name):
    arm_columns = [
        'arm', 'Harrell_mean_over_seed_algorithms', 'Uno60_mean_over_seed_algorithms',
        'Harrell_seed_min', 'Harrell_seed_max', 'Uno60_seed_min', 'Uno60_seed_max',
        'selected_epoch_median', 'actual_epoch_median', 'hit_maximum_fraction',
        'parameters_registered_mean', 'parameters_gradient_active_mean',
        'seconds_mean_concurrent_run', 'peak_GPU_MiB_max',
        'real_training_cells_this_stage', 'selected_model_cells',
    ]
    pair_columns = [
        'comparison', 'family', 'metric', 'mean_seed_pair_weighted_delta',
        'positive_seed_algorithms', 'positive_inner_cells', 'inner_cells',
        'seed_31_pair_weighted_delta', 'seed_47_pair_weighted_delta',
        'seed_83_pair_weighted_delta',
    ]
    stage = {'name': name, 'path': path,
             'validation': read_json(path / 'VALIDATION.json'),
             'gate': read_json(path / 'STAGE_DECISION.json'),
             'counts': read_json(path / 'EXECUTION_COUNTS.json'),
             'arms': read_table(path / 'ARM_SUMMARY.csv', arm_columns),
             'pairs': read_table(path / 'PAIRED_INNER_DIAGNOSTICS.csv', pair_columns)}
    require(stage['validation'].get('status') == 'PASS', name + '技术验证没有 PASS')
    require(not stage['arms'].arm.duplicated().any(), name + ' arm 汇总重复')
    require(not stage['pairs'][['comparison', 'metric']].duplicated().any(),
            name + '配对汇总重复')
    for key in arm_columns[1:]:
        values = stage['arms'][key].to_numpy(float)
        require(np.isfinite(values).all(), name + ' arm 汇总非有限：' + key)
    values = stage['pairs']['mean_seed_pair_weighted_delta'].to_numpy(float)
    require(np.isfinite(values).all(), name + ' 配对差非有限')
    return stage


def query_audit(path):
    value = read_json(path / 'QUERY_REPRESENTATION_AUDIT.json')
    frame = read_table(path / 'QUERY_REPRESENTATION_AUDIT.csv', [
        'arm', 'seed', 'fold', 'probe_patients',
        'query_matched_wrong_cosine_mean', 'query_matched_wrong_relative_L2_mean',
        'query_RNA_scale1p1_relative_L2_mean',
        'pre_LayerNorm_RNA_scale1p1_relative_L2_mean',
        'mixed_to_ordinary_message_norm_mean', 'interaction_to_main_SD',
        'matched_wrong_total_risk_abs_mean'])
    require(value.get('status') == 'PASS', '查询/真实预处理审计没有 PASS')
    require(value.get('risk_arrays_persisted') is False and value.get('outer_test_probes') == 0,
            '查询审计越过未授权的数据边界')
    require(value.get('all_mixed_reference_risks_exact_zero') is True,
            'mixed 完整参考零点检查未通过')
    require(len(frame) == value['cells'], 'query 审计计数不一致')
    return value, frame


def telemetry_summary(rows, description):
    """Unweighted sample statistics, not full-run time-weighted utilization."""
    require(len(rows) > 0, '缺少真实资源样本：' + description)
    memory = np.asarray([r['memory'] for r in rows], float)
    util = np.asarray([r['util'] for r in rows], float)
    times = np.asarray([r['time'] for r in rows], float)
    require(np.isfinite(memory).all() and np.isfinite(util).all() and
            np.isfinite(times).all(), '资源记录存在非有限值')
    require((memory >= 0).all() and ((util >= 0) & (util <= 100)).all(),
            '资源记录超出有效范围')
    return {'scope': description, 'samples': len(rows),
            'first_sample_unix': float(times.min()), 'last_sample_unix': float(times.max()),
            'sample_span_seconds': float(times.max() - times.min()),
            'memory_MiB_mean': float(memory.mean()), 'memory_MiB_median': float(np.median(memory)),
            'memory_MiB_max_sampled': float(memory.max()),
            'GPU_utilization_percent_mean': float(util.mean()),
            'GPU_utilization_percent_median': float(np.median(util)),
            'GPU_utilization_percent_p10': float(np.quantile(util, .1)),
            'GPU_utilization_percent_p90': float(np.quantile(util, .9)),
            'GPU_utilization_percent_max_sampled': float(util.max()),
            'aggregation': 'unweighted arithmetic statistics over observed device samples',
            'not_model_allocator_peak': True,
            'not_complete_wall_clock_time_weighted_utilization': True}


def resources(private):
    original = read_json(private / 'GPU_TELEMETRY.json')
    parsed = []
    for row in original:
        raw = row['GPU'].split(',')
        require(len(raw) == 2, '原调度器 GPU 字符串不是单 GPU memory/util 两列')
        parsed.append({'time': float(row['time']), 'memory': float(raw[0]), 'util': float(raw[1])})
    result = {'original40_scheduler': telemetry_summary(
        parsed, 'original40 scheduler samples; device total while queue active')}
    monitor = read_json(private / 'RESOURCE_TELEMETRY.json')
    for started, key in [(False, 'monitor_remaining40'), (True, 'monitor_budget160')]:
        selected = [{'time': r['time'], 'memory': r['GPU_memory_MiB'],
                     'util': r['GPU_utilization_percent']}
                    for r in monitor if bool(r['budget160_started']) == started]
        result[key] = telemetry_summary(selected, (
            'monitor started during stage40: only remaining40 portion' if not started else
            'monitor samples after budget160 lock; not an isolated per-model speed experiment'))
    result['limitations'] = [
        'Periodic samples may miss true device peaks and include other GPU processes.',
        'The second monitor began during original40; it does not cover the entire original run.',
        'Training per-cell allocator peaks come from ARM_SUMMARY, not these device samples.',
        'The monitor stops at budget160 DONE; it does not measure a later A3F repair run.',
        'Low allocated memory is reported honestly; capacity occupancy is not itself useful computation.',
    ]
    return result


def budget_rows(stage):
    rows = stage['gate'].get('maximum_epoch_fraction_per_arm')
    if stage['name'] == 'A3F_affine_bugfix':
        # The real A3F evaluator is a one-arm append-only repair and exposes
        # its cap fraction in ARM_SUMMARY plus the stage-level gate.
        arm = stage['arms'].iloc[0]
        cells = int(arm.selected_model_cells)
        fraction = float(arm.hit_maximum_fraction)
        hits = round(fraction * cells)
        require(abs(hits / cells - fraction) < 1e-10,
                'A3F触顶比例不能还原真实单元计数')
        require(abs(fraction - stage['gate']['maximum_epoch_fraction']) < 1e-10,
                'A3F触顶门与模型汇总不一致')
        return [{'arm': 'A3F', 'neural_cells': cells, 'hit_maximum_cells': hits,
                 'hit_maximum_fraction': fraction, 'maximum_epochs': 160,
                 'budget_gate_pass': bool(stage['gate']['budget_gate_pass'])}]
    require(isinstance(rows, list) and rows, stage['name'] + '缺少逐臂预算门')
    return rows


def contrast_gate(frame, comparison, config):
    rows = frame[(frame.comparison == comparison) & (frame.metric == 'Harrell')]
    require(len(rows) == 1, '必须恰好一条主比较 Harrell：' + comparison)
    row = rows.iloc[0]
    delta = [float(row[f'seed_{seed}_pair_weighted_delta']) for seed in config['seeds']]
    each = all(d > config['gate']['required_each_seed_mean_Harrell_gain'] for d in delta)
    fraction = float(row['positive_inner_cells']) / int(row['inner_cells'])
    ok = each and fraction >= config['gate']['required_primary_positive_cell_fraction']
    return {'comparison': comparison, 'each_seed_gain_pass': each,
            'seed_deltas': {str(seed): d for seed, d in zip(config['seeds'], delta)},
            'positive_inner_cells': int(row['positive_inner_cells']),
            'inner_cells': int(row['inner_cells']), 'positive_inner_cell_fraction': fraction,
            'pass': ok}


def stage_arm_table(stage):
    rows = []
    for _, row in stage['arms'].iterrows():
        rows.append([row.arm, row.Harrell_mean_over_seed_algorithms,
                     row.Uno60_mean_over_seed_algorithms,
                     f'{row.Harrell_seed_min:.6f}–{row.Harrell_seed_max:.6f}',
                     f'{row.Uno60_seed_min:.6f}–{row.Uno60_seed_max:.6f}',
                     row.selected_epoch_median, row.actual_epoch_median,
                     row.hit_maximum_fraction, int(row.real_training_cells_this_stage)])
    return table(['arm', 'Harrell', 'Uno60', 'seed Harrell范围', 'seed Uno60范围',
                  '选中epoch中位数', '实际epoch中位数', '触顶比例', '本阶段真实拟合'], rows)


def check_paired_points(stage, reference_arms=None):
    """A point difference and its table must describe the same estimand."""
    arms = stage['arms'].set_index('arm')
    if reference_arms is not None:
        arms = pd.concat([reference_arms.set_index('arm'), arms])
    for _, row in stage['pairs'].iterrows():
        a, b = row.comparison.split('-')
        require(a in arms.index and b in arms.index, '配对比较的模型缺失')
        require(row.metric in ['Harrell', 'Uno60'], '未锁定的指标出现在配对表')
        key = row.metric + '_mean_over_seed_algorithms'
        delta = float(arms.loc[a, key]) - float(arms.loc[b, key])
        require(abs(delta - row.mean_seed_pair_weighted_delta) < 2e-10,
                '配对点估计与arm汇总不是同一统计对象：' + row.comparison)
        require(0 <= row.positive_seed_algorithms <= 3 and
                0 <= row.positive_inner_cells <= row.inner_cells == 9,
                '配对方向计数错误')


def pairs_table(frame):
    return table(['比较', '指标', 'seed配对加权Δ均值', 'seed31 Δ', 'seed47 Δ', 'seed83 Δ',
                  '正向seed', '正向内层单元/总数'], [
        [r.comparison, r.metric, r.mean_seed_pair_weighted_delta,
         r.seed_31_pair_weighted_delta, r.seed_47_pair_weighted_delta,
         r.seed_83_pair_weighted_delta, int(r.positive_seed_algorithms),
         f'{int(r.positive_inner_cells)}/{int(r.inner_cells)}']
        for _, r in frame.iterrows()])


def query_table(frame):
    keys = ['query_matched_wrong_cosine_mean', 'query_matched_wrong_relative_L2_mean',
            'query_RNA_scale1p1_relative_L2_mean',
            'pre_LayerNorm_RNA_scale1p1_relative_L2_mean',
            'mixed_to_ordinary_message_norm_mean', 'interaction_to_main_SD',
            'matched_wrong_total_risk_abs_mean']
    return table(['arm', '真/错query cosine', '真/错query相对L2', 'RNA×1.1后query相对L2',
                  'LayerNorm前相对L2', 'mixed/ordinary范数', '交互/主效应风险SD', '真/错总风险MAE'],
                 [[arm] + [float(group[key].median()) for key in keys]
                  for arm, group in frame.groupby('arm', sort=True)], digits=7)


def compute_audit(stages, telemetry, counts):
    # Counts include the initial A11 computed unconditionally by the actual
    # forward implementation, even when only diagnostic use survives.
    score_heads = {'A0': 0, 'A1': 12, 'A2': 8, 'A3': 8, 'A3F': 8,
                   'A4': 16, 'A5': 16, 'A6': 16, 'A7': 16, 'F0': 8, 'PR': 8}
    av_equivalents = {'A0': 0, 'A1': 2, 'A2': 1, 'A3': 1, 'A3F': 1,
                      'A4': 1, 'A5': 1, 'A6': 1, 'A7': 2, 'F0': 1, 'PR': 1}
    operators = []
    for arm, heads in score_heads.items():
        elements = heads * 512 * 128
        operators.append({'arm': arm, 'G': 512, 'N': 128, 'score_head_map_count': heads,
                          'computed_score_elements_per_patient_forward': elements,
                          'QK_dot_product_FLOPs_approx': 2 * elements * 16,
                          'AV_FLOPs_approx': av_equivalents[arm] * 2 * 512 * 128 * 64,
                          'scope': 'QK/AV multiply-adds only; not end-to-end or training FLOPs'})
    records = []
    for stage in stages:
        for _, row in stage['arms'].iterrows():
            records.append({'stage': stage['name'], 'arm': row.arm,
                            'registered_parameters_mean': float(row.parameters_registered_mean),
                            'gradient_active_parameters_mean': float(row.parameters_gradient_active_mean),
                            'seconds_mean_concurrent_run': float(row.seconds_mean_concurrent_run),
                            'model_allocator_peak_MiB_max': float(row.peak_GPU_MiB_max),
                            'real_training_cells_this_stage': int(row.real_training_cells_this_stage),
                            'selected_model_cells': int(row.selected_model_cells)})
    return {'status': 'PASS', 'scope': 'actual architecture counts and observed resource accounting; not compute-matched superiority',
            'operator_counts': operators, 'actual_arm_summary_records': records,
            'resource_telemetry': telemetry, 'fit_accounting': counts,
            'redundant_initial_A11_in_actual_forward_included': True,
            'official_diff_crossattention_heads': 2,
            'ordinary_attention_heads': 4,
            'full_model_FLOPs_computed': False,
            'all_arms_compute_matched': False,
            'training_memory_method': 'exact two-pass full-risk Efron gradient with fixed parameters and dropout0; microbatch activations discarded between backward calls',
            'limitations': [
                'All operators register unused comparison-specific parameters; active count is gradient-defined for the saved summary and is not proof of predictive use.',
                'Runtime is measured during concurrent jobs; reused original40 references retain their original runtime, not new budget160 compute.',
                'Ridge initialization and preprocessing may lie outside the per-cell neural timer; do not interpret its sum as total wall time.',
                'Attention score counts are architectural execution counts, not empirical efficacy or histological evidence.',
                'Exact two-pass saves activation memory but repeats forward work; no padding or dummy allocations were used to raise memory occupancy.',
            ]}


def reproduction(config, source_id, budget_id, repair_validation):
    return f'''# 复现说明

## 边界与依赖

本交付是 DSS 内层探索，不含患者级数据、所选基因、错配索引、OOF、风险或权重。
需要获授权的原始私有数据与历史特征缓存、已有患者折、NVIDIA GPU，以及与锁定训练环境兼容的 Python/PyTorch/numpy/scipy/pandas/scikit-learn/scikit-survival/pytest。
文件来源与真实预处理状态由 hash/分区审计核验；不可用其他 PCA、病例顺序或基因列表静默替代。

- 原训练 source：`{source_id}`
- 160 敏感性 source：`{budget_id}`
- A3F 实现来源：`{repair_validation['implementation_source']}`
- 共同配置：`configs/trimodal_interaction_v1.json`
- 追加预算：`configs/trimodal_interaction_budget160.json`
- seeds={config['seeds']}，既有外折内部分区={config['folds']}；没有新增 outer-test 评估。

## 执行顺序

下列命令从仓库根目录执行。私有路径由现有配置指向；已有 receipt 应先核验，禁止覆盖历史训练产物。

```bash
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q -p no:cacheprovider tests/test_trimodal_interaction.py
python scripts/audit_trimodal_environment.py
python scripts/schedule_trimodal_interaction.py
python scripts/validate_trimodal_interaction.py --device cuda
python scripts/audit_trimodal_query.py --device cuda
# 仅在已记录的逐臂触顶门触发后执行。原40结果不覆盖。
python scripts/extend_trimodal_budget.py
python scripts/validate_trimodal_interaction.py --device cuda --budget160
python scripts/audit_trimodal_query.py --device cuda --budget160
# A3F 是独立来源的 affine bugfix；按其脚本入口完成9个真实拟合及独立审计。
python scripts/repair_trimodal_diff_norm.py
# 以下报告命令会拒绝任何阶段未完成/审计未通过的状态。
python scripts/report_trimodal_interaction.py
python scripts/package_trimodal_interaction.py
```

资源监控使用 `scripts/monitor_trimodal_resources.py`；其实际开始于原40运行中，报告不把它冒充完整40资源轨迹。
A3F 上述无参数入口依次完成其9个真实拟合、独立验证与DONE，不支持 `--help`；不能跳过其 VALIDATION。

## 计数与统计复现

原40的81个神经单元各有一次真实 ridge 初始化，另有3次 A0 凸基线拟合：84个模型单元及84次 ridge solver调用。
160仅重训原40触顶的 M 个神经单元，其余作为原checkpoint引用；A3F再加9个神经拟合及9次 ridge初始化。
最终数量从实际 DONE/EXECUTION_COUNTS交叉核对；同一凸基线在3seed中的引用、checkpoint replay、query probe不计新训练。

指标先在每seed的内层分区累加可比较对分子/分母，再平均seed算法指标；不跨折直接排列不同尺度风险，不平均风险。
这些内层分区含重叠患者且已参与选择；只报告配对描述，不做置信区间/p值，不与历史完整外折OOF榜单比较。
160追加发生于观察到内层40触顶后，A3F发生于发现 affine实现缺陷后，二者均不属于原先独立确认。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root', type=Path, default=ROOT / 'reports/trimodal_interaction_v1')
    parser.add_argument('--private-root', type=Path, default=DEFAULT_PRIVATE)
    args = parser.parse_args()
    report, private = args.report_root.resolve(), args.private_root
    config = read_json(ROOT / 'configs/trimodal_interaction_v1.json')
    original = load_stage(report, 'original40')
    extended = load_stage(report / 'budget160', 'budget160_sensitivity')
    repair = load_stage(report / 'a3f_norm_repair', 'A3F_affine_bugfix')
    oq, oq_frame = query_audit(report)
    eq, eq_frame = query_audit(report / 'budget160')
    original_done = read_json(private / 'DONE.json')
    extended_done = read_json(private / 'budget160/DONE.json')
    repair_done = read_json(private / 'a3f_norm_repair/DONE.json')
    require(original_done.get('status') == 'ALL_LOCKED_INNER_CELLS_FINISHED', '原40未完成')
    require(extended_done.get('status') == 'ALL_BUDGET_CELLS_VERIFIED', '160未完成')
    require(repair_done.get('status') == 'PASS', 'A3F未完成')
    source_id = original['validation']['source']
    require(source_id == extended['validation']['source'] == original_done['source'] ==
            extended_done['source'] == oq['training_source'] == eq['training_source'],
            '原40/160/source/query receipt不一致')
    budget_id = extended_done['budget_source']
    require(budget_id == extended['validation']['budget_source'] == eq['budget_source'],
            '160来源不一致')
    repair_lock = read_json(private / 'a3f_norm_repair/LOCK.json')
    repair_identity = repair['validation']['implementation_source']
    require(repair['validation']['source'] == source_id == repair_done['source'] ==
            repair_lock['original_source'], 'A3F原始source不一致')
    require(repair_identity == repair_done['implementation_source'] ==
            repair_lock['implementation_source'], 'A3F独立实现来源不一致')
    expected_repair_identity = hashlib.sha256((
        source_id + sha(ROOT / 'configs/trimodal_diff_norm_repair.json') +
        sha(ROOT / 'scripts/repair_trimodal_diff_norm.py')).encode()).hexdigest()
    require(repair_identity == expected_repair_identity,
            'A3F审计后脚本或配置变化')
    require(repair['validation']['official_RMSNorm_affine_parameters'] == 32 and
            repair['validation']['actual_RMSNorm_affine_scale_updated'] is True,
            'A3F没有验证32个affine参数实际训练更新')
    require(repair_done['real_neural_fits'] == 9 and repair_done['ridge_initializations'] == 9 and
            repair['counts']['total_ridge_solver_calls'] == 9 and
            repair['counts']['real_neural_training_cells'] == 9,
            'A3F9次真实神经/solver计数未闭合')
    # PASS is only the technical gate, not a GO mechanism decision.
    for stage in [original, extended, repair]:
        require(stage['validation'].get('status') == 'PASS', '所有真实阶段必须完成验证')
        require(stage['counts'].get('outer_fits', 0) == 0 and
                stage['counts'].get('patient_OOF_files', 0) == 0, '发现本轮未授权的外层推断')
    require(set(original['arms'].arm) == set(config['arms']) == set(extended['arms'].arm),
            '原40/160臂集合与配置不一致')
    require(set(repair['arms'].arm) == {'A3F'}, '修复目录必须只含A3F模型汇总')
    check_paired_points(original)
    check_paired_points(extended)
    check_paired_points(repair, extended['arms'])
    neural = (len(config['arms']) - 1) * len(config['seeds']) * len(config['folds'])
    baseline = len(config['folds'])
    total = neural + baseline
    require((neural, baseline, total) == (81, 3, 84), '锁定矩阵计数变化')
    require(original_done['cells'] == total and original_done['real_neural_fits'] == neural and
            original_done['real_convex_baseline_fits'] == baseline, '原完成计数变化')
    M = int(extended_done['new_real_neural_fits'])
    refs = int(extended_done['evaluation_references'])
    require(M + refs == total == extended_done['cells'], '160真实拟合与引用计数不闭合')
    require(int(extended['counts']['real_neural_training_cells_this_stage']) == M and
            int(extended['counts']['original40_references_without_refit']) == refs,
            '160验证计数与DONE不一致')
    repaired = int(repair['arms'].real_training_cells_this_stage.sum())
    require(repaired == 9 and int(repair['arms'].selected_model_cells.sum()) == 9,
            'A3F必须是9个真实模型单元')
    require(int(original['counts']['total_ridge_solver_calls_this_stage']) == total and
            int(extended['counts']['total_ridge_solver_calls_this_stage']) == M,
            'ridge调用次数不一致')
    prefix = []
    for path in sorted((private / 'budget160/cells').glob('*.json')):
        receipt = read_json(path)
        if not receipt['reused_reference']:
            prefix.append(float(receipt['original_prefix_max_error']))
    require(len(prefix) == M and all(finite(v) and v <= 1e-7 for v in prefix),
            '160与原40前缀审计缺失或不通过')
    trigger = read_json(report / 'budget160/TRIGGER.json')
    require(trigger.get('status') == 'TRIGGERED' and trigger.get('outer_results_seen') is False,
            '160追加预算触发边界不明确')
    counts = {'original40_real_neural_cells': neural, 'baseline_only_convex_cells': baseline,
              'original40_real_model_cells': total, 'original40_neural_ridge_initializations': neural,
              'original40_total_ridge_solver_calls': total,
              'budget160_new_real_neural_cells': M, 'budget160_original40_references': refs,
              'A3F_new_real_neural_cells': repaired,
              'total_real_model_cells_all_stages': total + M + repaired,
              'total_real_neural_cells_all_stages': neural + M + repaired,
              'total_ridge_solver_calls_all_stages': total + M + repaired,
              'convex_baseline_metric_references_per_stage': 9,
              'extra_convex_references_without_new_fit_per_stage': 6,
              'outer_fits': 0, 'patient_OOF_files': 0,
              'original40_prefix_max_error_in160': max(prefix, default=0.),
              'replay_and_query_probes_are_not_new_fits': True}
    effective_budget = [r for r in budget_rows(extended) if r['arm'] != 'A3']
    repair_budget = budget_rows(repair)
    require({r['arm'] for r in repair_budget} == {'A3F'}, 'A3F逐臂预算审计缺失')
    effective_budget += repair_budget
    budget_ok = all(bool(r['budget_gate_pass']) for r in effective_budget)
    effective_contrasts = []
    for a, b in config['primary_contrasts']:
        if (a, b) == ('A5', 'A3'):
            effective_contrasts.append(contrast_gate(repair['pairs'], 'A5-A3F', config))
        else:
            effective_contrasts.append(contrast_gate(extended['pairs'], a + '-' + b, config))
    gradient_ok = extended['gate']['nonzero_A5_query_and_gate_gradients_pass'] is True
    comparison_ok = all(r['pass'] for r in effective_contrasts)
    final = ('BUDGET_UNRESOLVED' if not budget_ok else
             'GO_TO_SEPARATELY_PREREGISTERED_FOLLOWUP' if gradient_ok and comparison_ok else
             'STOP_CURRENT_INTERACTION_PILOT')
    telemetry = resources(private)
    compute = compute_audit([original, extended, repair], telemetry, counts)
    public_inputs = {}
    for stage in [original, extended, repair]:
        names = ['ARM_SUMMARY.csv', 'PAIRED_INNER_DIAGNOSTICS.csv', 'STAGE_DECISION.json',
                 'EXECUTION_COUNTS.json', 'VALIDATION.json']
        if stage is not repair:
            names += ['QUERY_REPRESENTATION_AUDIT.json', 'QUERY_REPRESENTATION_AUDIT.csv']
        for name in names:
            path = stage['path'] / name
            public_inputs[path.relative_to(report).as_posix()] = sha(path)
    decision = {'decision': final, 'scope': 'exploratory overlapping inner validation; not OOF, independent confirmation, or efficacy inference',
                'original40_decision': original['gate']['decision'],
                'budget160_original_A3_decision': extended['gate']['decision'],
                'final_basis': 'budget160 with A5-A3 replaced by A5-A3F affine bugfix; all original comparisons retained',
                'budget_gate_pass': budget_ok, 'effective_budget_per_arm': effective_budget,
                'A5_query_and_gate_gradient_pass': gradient_ok,
                'all_effective_primary_comparisons_pass': comparison_ok,
                'effective_primary_comparisons': effective_contrasts,
                'budget_amendment_after_observing_inner_cap40': True,
                'A3F_bugfix_not_best_of_original_and_repair': True,
                'original_training_source': source_id, 'budget160_source': budget_id,
                'A3F_implementation_source': repair_identity,
                'aggregate_report_input_sha256': public_inputs,
                'report_script_sha256': sha(__file__),
                'counts': counts, 'outer_evaluation': 'NOT_STARTED',
                'no_confidence_intervals_or_p_values': True,
                'does_not_establish_RNA_redundancy_or_absence_of_biological_interaction': True,
                'confidence': {'completed_artifact_and_operator_checks': '高（以真实审计为限）',
                               'engineering_followup_decision': '中（内层重复患者、有限预算及选择暴露）',
                               'external_prognostic_or_biological_mechanism': '低（未做独立外部/形态确认）'}}
    primary40 = original['pairs'][original['pairs'].family == 'primary']
    primary160 = extended['pairs'][extended['pairs'].family == 'primary']
    repaired_pair = repair['pairs'][repair['pairs'].comparison == 'A5-A3F']
    require(set(repaired_pair.metric) == {'Harrell', 'Uno60'}, 'A3F必须同时有Harrell及Uno60配对')
    result = [
        '# 三模态条件差分交互：内层可行性及预算/基线修复审计',
        '## 判定与范围',
        f'**{final}**。最终工程门采用160预算及A3F affine修复；原40判定为 '
        f'`{original["gate"]["decision"]}`。技术验证PASS不代表机制有效。',
        '**本轮没有完整外折OOF、outer-test预测、置信区间或p值。** 这是既有外折1/3/5内部的'
        '重叠inner-validation × seeds31/47/83。每seed先累加相应可比较对分子/分母，再平均seed算法指标，'
        '没有跨折直接排列风险或平均风险。以下数字不能与历史完整OOF榜单排名。',
        '## 设计与信息保留',
        '共同线性主效应为Clinical+S32+LOCAL-CONCH-PCA32+RNA512，L2=1凸初始化后共同训练；'
        'RNA值/gene identity/missing分别进入查询。冻结CONCH为唯一局部视觉主干，原512 geometry候选均匀取128；'
        'patch PCA64/LOCAL PCA32/标准化/参考仅由inner-train拟合。无教师、stacking、重建、对比或排序辅助任务。'
        'rawRNA旁路证明直接信息路径存在，不证明RNA→Query保留了原始预后排序。',
        '## 原40预算：保留原结果', stage_arm_table(original),
        '逐臂触顶门检查训练是否有足够机会按固定patience停止；不能由已触顶且仍在改善的模型否定机制。'
        '原40曲线、checkpoint及指标不被160结果覆盖。',
        '### 原40全部主比较（仅描述）', pairs_table(primary40),
        '## 160预算敏感性：明确追加时点',
        '这是看到一项内层单元达到40轮上限后追加的探索性训练预算敏感性，后续按逐臂触顶门记录触发；'
        '不是最初就锁定的独立确认，也未使用outer-test表现。只把原来actual_epochs==40的M个神经单元'
        '从相同初始化重训，其他完整early-stop单元引用原checkpoint。',
        f'实际 M={M}，原checkpoint引用={refs}；重训前40轮确定性字段与原历史最大绝对误差为 '
        f'{counts["original40_prefix_max_error_in160"]:.3g}。独立验证重新检查所选epoch、EMA、patience、'
        'checkpoint风险重放及Efron/Harrell/Uno60；引用不是新训练。',
        stage_arm_table(extended),
        '### 160全部原主比较（保留A3缺陷标记）', pairs_table(primary160),
        '## A3F：修复官方Diff对照的RMSNorm affine',
        '原A3遗漏官方headwise RMSNorm的可学习affine，使用elementwise_affine=False。'
        '它作为缺陷版本完整保留，不用其点估计作最终合法基线。A3F仅将affine修复为True，'
        '保留2个diff heads、lambda、其余输入/损失/选择规则，独立来源真实重训9单元。'
        '这属于bugfix，最终A5−A3F替代A5−A3；不在两者中择优。仍是cross-attention/Cox适配，'
        '不是原生语言模型，也未实现严格算量匹配。',
        stage_arm_table(repair), pairs_table(repaired_pair),
        '## 共同最终门',
        table(['比较', '逐seed正向门', '正向内层单元/总数', '通过'],
              [[r['comparison'], r['each_seed_gain_pass'],
                f'{r["positive_inner_cells"]}/{r["inner_cells"]}', r['pass']]
               for r in effective_contrasts]),
        f'最终逐臂预算门通过={budget_ok}；A5查询与gate真实训练梯度路径门通过={gradient_ok}。'
        '主方向门以Harrell定义；Uno60的全部配对数值同时披露，不事后交换主指标。'
        '若预算仍不足，判定必须为BUDGET_UNRESOLVED，不能写成idea no-go。',
        table(['arm', '预算上限', '触顶单元', '模型单元', '触顶比例', '预算门'],
              [[r['arm'], r.get('maximum_epochs', 160), r['hit_maximum_cells'],
                r['neural_cells'], r['hit_maximum_fraction'], r['budget_gate_pass']]
               for r in effective_budget]),
        '## 查询表征、参考与真实预处理审计',
        '统计均为各单元按固定manifest顺序前32名inner-train患者的无标签探针，表值为单元均值的跨单元中位数；'
        '不是按事件或风险挑样本，风险仅在内存做汇总。A2/A5/A7各9单元。',
        '### 原40查询', query_table(oq_frame), '### 160查询', query_table(eq_frame),
        '逐token LayerNorm可能压缩RNA幅度；RNA×1.1和gene-identity置零只测数值敏感性，'
        '不能单凭query变化断言预后可解码性。真/错query、消息或风险发生变化也是算子响应，'
        '不自动表示患者特异性预测收益。A7范数匹配发生在gate之前且用自身参数检查，'
        '不要求它与独立学习A5的post-gate范数相等；A7仍含mixed幅度，不能说完全排除了所有条件信息。',
        f'完整参考零点两阶段均通过；原40/160最大mixed行和误差分别 '
        f'{oq["mixed_row_sum_error_global_max"]:.3g}/{eq["mixed_row_sum_error_global_max"]:.3g}，'
        f'A7门前范数最大误差分别 {oq["A7_pregate_norm_match_error_global_max"]:.3g}/'
        f'{eq["A7_pregate_norm_match_error_global_max"]:.3g}。实际源patch行重建PCA、每患者采样限额、'
        'fit患者归属、参考均值及特征hash由审计验证；合成测试不替代真实源审计。',
        '## 实际训练数量',
        table(['项目', '真实次数/引用数'], [[k, v] for k, v in counts.items()
              if isinstance(v, int) and not isinstance(v, bool)]),
        '原81个神经模型各做一次ridge初始化，加3个A0凸基线，原阶段共84个模型单元和84次ridge solver调用。'
        '160追加M个神经模型；A3F追加9个。checkpoint replay、query probe及同一凸基线的seed引用不计新训练。',
        '## 参数、真实耗时与GPU',
        table(['阶段', 'arm', '注册参数均值', '梯度活跃参数均值', '耗时均值秒', 'allocator峰值MiB', '本阶段真实拟合'],
              [[r['stage'], r['arm'], r['registered_parameters_mean'],
                r['gradient_active_parameters_mean'], r['seconds_mean_concurrent_run'],
                r['model_allocator_peak_MiB_max'], r['real_training_cells_this_stage']]
               for r in compute['actual_arm_summary_records']], digits=3),
        '代码为比较注册了部分当前arm不使用的参数，所以注册数相同不意味着有效容量相同；'
        'gradient-active来自实际训练统计，也不证明该路径有预后作用。耗时采集于并行环境，'
        '不是隔离速度排名；160引用行的耗时保留原40值，不是新增160耗时。',
        table(['采样范围', '样本数', '显存均值MiB', '显存采样最大MiB', 'GPU利用率均值%', '中位数%', 'P10/P90%'],
              [[k, r['samples'], r['memory_MiB_mean'], r['memory_MiB_max_sampled'],
                r['GPU_utilization_percent_mean'], r['GPU_utilization_percent_median'],
                f'{r["GPU_utilization_percent_p10"]:.1f}/{r["GPU_utilization_percent_p90"]:.1f}']
               for k, r in telemetry.items() if isinstance(r, dict)], digits=2),
        '原调度器样本覆盖原40队列，额外monitor开始于40运行中，只覆盖其剩余部分和160阶段；'
        '它在160 DONE后停止，没有测量之后的A3F独立修复。样本为整卡占用，可包含其他进程，'
        '不等于单模型峰值或完整时长加权利用率。显存占用低时如实披露；exact two-pass先求完整风险集'
        '风险梯度，再逐microbatch反传，在参数不变/dropout0下减少激活保留但增加forward计算，'
        '不为占显存制造无效张量。',
        '### 实际forward的attention计算差异',
        table(['arm', 'score head-map数', '每患者score元素', 'QK近似FLOPs', 'AV近似FLOPs'],
              [[r['arm'], r['score_head_map_count'], r['computed_score_elements_per_patient_forward'],
                r['QK_dot_product_FLOPs_approx'], r['AV_FLOPs_approx']]
               for r in compute['operator_counts']]),
        '实际实现先计算A11，因此A1为3×4=12个score head-map；A2/F0/PR为初始+最终2×4=8；'
        'A3/A3F为初始4加2diff-head×2图=8；A4–A7为四图×4=16。每图G512×N128，'
        '以上含实际冗余，不是理想化删掉诊断计算的数字。QK/AV只计乘加，不含投影、norm、gate、'
        '主效应、反向和数据处理，不能作为完整模型FLOPs。共同hidden/epoch不构成compute-matched优势。',
        '## 次要配对结果（同样仅描述）',
        pairs_table(extended['pairs'][extended['pairs'].family == 'secondary']),
        '## 优势、风险、置信度与下一步',
        '优势：所有臂共同保留原始RNA和强CP输入；完整Efron风险集、训练分区参考、确定性预算前缀、'
        '独立指标与实际源PCA都可审计；错误官方对照通过独立A3F修复，旧结果仍保留。',
        '风险：65个DSS事件、重复研究的单队列、重叠inner患者与同一验证集选epoch，使该轮不是独立效能检验；'
        '128patch/PCA64有限覆盖、query幅度压缩、计算未匹配及没有WSI形态真值限制机制解释。'
        '没有CI不能被曲解为无不确定性。技术检查置信度高，内部工程决策置信度中，'
        '外部预后与具体病理机制置信度低。',
        ('当前预算门仍失败；保留未解决状态，不能否定本idea。后续若追加预算应另锁来源和边界。'
         if final == 'BUDGET_UNRESOLVED' else
         '通过内部工程门只支持另外预注册后续验证，不支持直接宣布新机制成立或在本队列重复挑选赢家。'
         if final == 'GO_TO_SEPARATELY_PREREGISTERED_FOLLOWUP' else
         '逐臂预算触顶门已通过，但A1仍有1/9单元触顶160，不能称所有单元或全局优化已收敛。在这一限定下，当前交互实现未通过全部机制门，停止扩大该分支；'
         '保留主效应收益及负结果，不否定RNA独立排序或一般生物学互补。'),
        '### 拓展思考 / 你可能想知道',
        '直接RNA主效应可用与RNA→Query预后信息保真仍是两个问题。若后续继续，应先使用独立锁定的'
        '可解码性/条件增量问题定位瓶颈；本轮无标签尺度探针、非零梯度或四图代数性质不能代替该验证。',
    ]
    boundary = '''# 证据边界

- 所有新数值来自已完成真实训练和独立审计；没有outer-test/完整OOF，不与历史OOF榜单比较。
- 原40、160追加预算、A3F affine修复是三个分别登记来源的阶段。160在观察触顶后追加，A3F为发现实现缺陷后的修复，均非原先独立确认。
- 最终主比较采用160结果，A5−A3由A5−A3F替换；不在缺陷与修复版本之间择优。两阶段旧数值完整保留。
- inner患者重叠，seed不是新患者。配对结果仅描述，没有置信区间、p值或独立重复的统计解释。
- 指标以各seed分区可比较对分子/分母汇总；Uno60 censor KM仅来自该inner-train；不跨折平均风险。
- rawRNA旁路只保证直接输入通路，不保证query无损。LayerNorm探针只测表征响应，不测生存可解码性。
- query变化、消息非零、gate梯度和四参考零点仅支持其各自检查，不证明患者特异预后收益或医学语义。
- A7含mixed幅度，门前范数匹配不代表与A5门后范数相同；不能声称已经排除所有收缩/正则解释。
- A3F保留官方Diff的affine规范，但仍是cross-attention/Cox适配；score计算与mixed不同，未完成严格compute匹配。
- Exact two-pass降低激活显存但重复forward；总设备周期采样与单模型allocator峰值严格区分，不粉饰低显存占用。
- 没有原图/独立形态标签，不宣称识别特定细胞生态位、因果干预或PID纯三阶协同。
- STOP只限当前实现及预算；BUDGET_UNRESOLVED不可写成no-go；GO仅允许另行预注册后续。
- 代码和匿名聚合可公开；患者数据、基因名单、风险、错配索引及模型权重继续私有。
'''
    # No output is written before every prerequisite above has been validated.
    safe_write(report / 'RESULTS.md', '\n\n'.join(result))
    safe_write(report / 'decision.json', decision)
    safe_write(report / 'COMPUTE_BUDGET_AUDIT.json', compute)
    safe_write(report / 'EVIDENCE_BOUNDARIES.md', boundary)
    safe_write(report / 'REPRODUCE.md', reproduction(config, source_id, budget_id,
                                                   repair['validation']))
    print('REPORT_PASS', final, json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()

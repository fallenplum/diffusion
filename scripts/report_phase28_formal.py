"""Render Chinese Phase28 formal reports from verified anonymous JSON only.

No checkpoint, patient array, endpoint loader, model or training import is used.
This reporter applies no new scientific gate or model-selection rule. The
predeclared internal-screen decision is checked against its saved aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (31, 47)
FOLDS = (1, 2, 3)
ARMS = ('C00', 'C10', 'C01', 'C11', 'DIRECT')
CONTROLS = ('MATCHED', 'OFF', 'GENE_MEAN', 'GENE_WRONG', 'PATH_MEAN',
            'PATH_WRONG', 'BOTH_WRONG', 'CONSTANT_VALUES', 'WHOLE_OMICS_WRONG')
GATE_CONTROLS = CONTROLS[2:8]
METRICS = ('Harrell', 'Uno60')
NAMES = ('M0', 'CP', 'RNA', *ARMS,
         *(arm+'_'+control for arm in ARMS for control in CONTROLS if control != 'MATCHED'))
INPUT_NAMES = ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json', 'BUDGET_READOUT_AUDIT.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'finite scalar required: '+name)
    return float(value)


def equal(left, right, name):
    require(abs(number(left, name)-number(right, name)) <= 1e-12, 'aggregate identity failed: '+name)


def _reject_constant(value):
    raise ValueError('nonfinite JSON constant: '+value)


def read_json(path):
    raw = Path(path).read_bytes()
    result = json.loads(raw.decode('utf-8-sig'), parse_constant=_reject_constant)
    require(isinstance(result, dict), 'anonymous JSON object required: '+Path(path).name)
    return result, sha_bytes(raw)


def _finite_tree(value):
    if type(value) in (float, int):
        require(math.isfinite(value), 'nonfinite anonymous diagnostic')
    elif isinstance(value, dict):
        for child in value.values(): _finite_tree(child)
    elif isinstance(value, list):
        for child in value: _finite_tree(child)


def validate(inputs, replay, source):
    require(re.fullmatch('[0-9a-f]{64}', source) is not None, 'locked source must be a 64-digit SHA256')
    result, gate, decision, budget = (inputs[name] for name in INPUT_NAMES)
    for name, item in inputs.items():
        require(item.get('source') == source, 'source differs: '+name)
        _finite_tree(item)
    require(result.get('status') == 'COMPLETE_BIDIRECTIONAL_REREAD_INNER_PROTOTYPE'
            and result.get('stage') == 'prototype' and result.get('endpoint') == 'DSS'
            and result.get('cohort_patients') == 512 and result.get('DSS_events') == 65,
            'no complete Phase28 DSS internal evaluation is available')
    for key, expected in (('valid_endpoint_callback_calls', 1), ('new_predictor_fits_during_evaluation', 0),
                          ('bootstrap_draws', 0), ('unique_patient_OOF', False), ('risk_ensemble', False),
                          ('confirmatory_CI', False), ('patient_arrays_weights_or_IDs_public', False)):
        require(result.get(key) == expected, 'evaluation boundary differs: '+key)
    require(result.get('aggregation') == 'sum_three_within_fold_num_den_per_seed_then_mean_two_seeds',
            'unknown concordance aggregation')
    require(result.get('Clinical_is_not_a_local_query_input') is True
            and result.get('learned_unimodal_heads_can_drift_with_joint_training') is True,
            'model attribution boundary missing')
    artifact = result.get('artifact_audit', {})
    for key, expected in (('status', 'PASS_COMPLETE_CHECKPOINT_CONTRACT'), ('neural_models', 30),
                          ('seed_partition_cells', 6), ('epochs_each', 80), ('optimizer_steps', 2400),
                          ('new_main_anchor_fits', 0)):
        require(artifact.get(key) == expected, 'complete training evidence required: '+key)
    actual_replay = result.get('independent_replay', {})
    for item in (actual_replay, replay):
        for key, expected in (('status', 'PASS'), ('source', source), ('neural_models', 30),
                              ('risk_vectors_replayed', 300), ('reference_vectors_checked', 90)):
            require(item.get(key) == expected, 'independent replay evidence failed: '+key)
        error = number(item.get('maximum_absolute_risk_error'), 'replay error')
        require(0 <= error <= 1e-6, 'replay error exceeds lock')
    require(budget.get('schema') == 'phase28_checkpoint_budget_readout_v1'
            and budget.get('status') == 'PASS_OUTCOME_BLIND_CHECKPOINT_AGGREGATION'
            and budget.get('neural_models') == 30, 'complete checkpoint-only budget audit required')
    for key in ('new_predictor_fits', 'new_optimizer_steps', 'new_forward_calls', 'endpoint_array_requests'):
        require(budget.get(key) == 0, 'budget audit exceeded outcome-blind scope: '+key)
    for key in ('label_keys_inspected_or_used', 'rescaling_coefficients_searched', 'admission_gate_changed',
                'global_convergence_or_budget_sufficiency_claimed', 'patient_risks_indices_or_model_tensors_public'):
        require(budget.get(key) is False, 'budget audit claim boundary differs: '+key)
    require(budget.get('receipt_fingerprint') == artifact.get('receipt_fingerprint')
            == gate.get('receipt_fingerprint'), 'checkpoint receipt fingerprints differ')
    require(isinstance(artifact.get('receipt_fingerprint'), str)
            and re.fullmatch('[0-9a-f]{64}', artifact['receipt_fingerprint']), 'receipt fingerprint missing')
    require(budget.get('replay_report_sha256') == actual_replay.get('report_sha256'), 'budget/replay lineage differs')
    expected_cells = {(seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in ARMS}
    for title, rows in (('training', artifact.get('rows')), ('budget', budget.get('rows'))):
        require(isinstance(rows, list) and len(rows) == 30, 'all thirty '+title+' rows required')
        require({(row.get('seed'), row.get('fold'), row.get('arm')) for row in rows} == expected_cells,
                'complete unique '+title+' cell identities required')
        require(all(row.get('training', {}).get('actual_epochs') == 80 for row in rows), 'actual training budget differs')
    seed_rows, mean_rows = result.get('seed_rows'), result.get('two_seed_means')
    require(isinstance(seed_rows, list) and len(seed_rows) == 2*len(NAMES), 'all seed/model rows required')
    by = {(row.get('seed'), row.get('model')): row for row in seed_rows}
    require(set(by) == {(seed, name) for seed in SEEDS for name in NAMES}, 'seed/model coverage differs')
    require(isinstance(mean_rows, list) and len(mean_rows) == len(NAMES), 'all model means required')
    means = {row.get('model'): row for row in mean_rows}
    require(set(means) == set(NAMES), 'duplicate/missing mean models')
    for name in NAMES:
        for metric in METRICS:
            values = [number(by[seed, name].get(metric), 'concordance') for seed in SEEDS]
            require(all(0 <= value <= 1 for value in values), 'C-index outside [0,1]')
            equal(means[name].get(metric), sum(values)/2, name+' '+metric+' seed mean')
    folds = result.get('fold_rows')
    require(isinstance(folds, list) and len(folds) == 6*len(NAMES), 'all fold/model rows required')
    require({(row.get('seed'), row.get('fold'), row.get('model')) for row in folds}
            == {(seed, fold, name) for seed in SEEDS for fold in FOLDS for name in NAMES}, 'fold coverage differs')
    for seed in SEEDS:
        for name in NAMES:
            selected = [row for row in folds if row['seed'] == seed and row['model'] == name]
            for metric, prefix in (('Harrell', 'Harrell'), ('Uno60', 'Uno')):
                numerator = sum(number(row.get(prefix+'_num'), 'within-fold numerator') for row in selected)
                denominator = sum(number(row.get(prefix+'_den'), 'within-fold denominator') for row in selected)
                require(denominator > 0, 'zero comparable-pair denominator')
                equal(by[seed, name][metric], numerator/denominator, 'sum-fold concordance')

    def check_comparisons(rows, pairs, strict):
        require(isinstance(rows, list) and len(rows) == 2*len(pairs), 'complete locked comparison rows required')
        table = {(row.get('comparison'), row.get('metric')): row for row in rows}
        require(set(table) == {(a+'-'+b, metric) for a, b in pairs for metric in METRICS}, 'comparison identity differs')
        for a, b in pairs:
            for metric in METRICS:
                row = table[a+'-'+b, metric]
                deltas = [by[seed, a][metric]-by[seed, b][metric] for seed in SEEDS]
                equal(row.get('mean_seed_delta'), sum(deltas)/2, 'paired seed mean')
                saved = row.get('seed_deltas')
                require(isinstance(saved, list) and len(saved) == 2
                        and {item.get('seed') for item in saved} == set(SEEDS), 'complete paired seed deltas required')
                for item in saved:
                    equal(item.get('delta'), deltas[SEEDS.index(item['seed'])], 'paired seed delta')
                passed = all(value > 0 for value in deltas) if strict else sum(deltas)/2 >= 0
                require(row.get('pass') is passed, 'saved comparison decision contradicts points')
        return all(row['pass'] for row in rows)

    primary = check_comparisons(gate.get('primary_tests'), (('C11', 'C00'), ('C11', 'DIRECT')), True)
    guards = check_comparisons(gate.get('main_effect_guards'), (('C11', 'M0'), ('C11', 'CP')), False)
    check_comparisons(gate.get('direction_comparisons_diagnostic_only'), (('C11', 'C10'), ('C11', 'C01')), True)
    pairing = gate.get('frozen_pairing_guards')
    require(isinstance(pairing, list) and len(pairing) == 6
            and {row.get('comparison') for row in pairing} == {'C11-C11_'+name for name in GATE_CONTROLS},
            'complete frozen-pairing guards required')
    for row in pairing:
        name = row['comparison'][4:]
        deltas = {metric: means['C11'][metric]-means[name][metric] for metric in METRICS}
        for metric in METRICS: equal(row.get('mean_seed_deltas', {}).get(metric), deltas[metric], 'frozen guard delta')
        require(row.get('pass') is (all(value >= 0 for value in deltas.values()) and any(value > 0 for value in deltas.values())),
                'frozen-pairing guard contradicts points')
    expected_decision = 'GO' if primary and guards and all(row['pass'] for row in pairing) else 'STOP'
    for item in (gate, decision):
        require(item.get('stage') == 'prototype' and item.get('candidate') == 'C11'
                and item.get('decision') == expected_decision and item.get('outer_admitted') is False,
                'formal decision differs from locked screen')
    require(gate.get('status') == ('PROTOTYPE_GO_REQUIRES_NEW_PROTOCOL' if expected_decision == 'GO' else 'PROTOTYPE_NOT_ADMITTED')
            and gate.get('classification') == decision.get('status') == expected_decision,
            'formal status must not be a qualification-only or alternate-arm decision')
    require(gate.get('outer_automatically_started') is False and gate.get('separate_new_protocol_required') is True
            and decision.get('outer_started') is False and decision.get('confirmation_claimed') is False,
            'outer/confirmation boundary missing')
    paired = result.get('repair_damage')
    require(isinstance(paired, list) and len(paired) > 0, 'paired survival repair/damage required')
    paired_by = {(row.get('comparison'), row.get('metric')): row for row in paired}
    require(len(paired_by) == len(paired), 'duplicate repair/damage rows')
    for (comparison, metric), row in paired_by.items():
        require(metric in METRICS, 'unknown repair/damage metric')
        a, b = comparison.split('-', 1)
        require(a in means and b in means, 'repair/damage model unknown')
        equal(row.get('C0'), means[b][metric], 'paired baseline credit')
        equal(row.get('C1'), means[a][metric], 'paired candidate credit')
        equal(row.get('net'), means[a][metric]-means[b][metric], 'paired net concordance')
        equal(number(row.get('repair'), 'repair')-number(row.get('damage'), 'damage'), row['net'], 'repair-damage closure')
    for a, b in (('C11', 'C00'), ('C11', 'DIRECT'), ('C11', 'C10'), ('C11', 'C01'), ('C11', 'CP'), ('C11', 'M0'), ('C11', 'C11_OFF')):
        require(all((a+'-'+b, metric) in paired_by for metric in METRICS), 'key paired repair/damage comparison missing')
    seed_pairs = result.get('seed_repair_damage')
    require(isinstance(seed_pairs, list) and len(seed_pairs) == 2*len(paired_by), 'complete seed repair/damage rows required')
    seed_paired_by = {(row.get('comparison'), row.get('metric'), row.get('seed')): row for row in seed_pairs}
    require(set(seed_paired_by) == {(comparison, metric, seed) for comparison, metric in paired_by for seed in SEEDS},
            'unique seed repair/damage coverage differs')
    for (comparison, metric, seed), row in seed_paired_by.items():
        a, b = comparison.split('-', 1)
        equal(row.get('C0'), by[seed, b][metric], 'seed paired baseline')
        equal(row.get('C1'), by[seed, a][metric], 'seed paired candidate')
        equal(row.get('net'), by[seed, a][metric]-by[seed, b][metric], 'seed paired concordance')
        equal(number(row.get('repair'), 'seed repair')-number(row.get('damage'), 'seed damage'), row['net'], 'seed repair-damage closure')
    return result, gate, decision, budget, by, means, paired_by, seed_paired_by


def fmt(value, signed=False):
    if value is None: return '未定义（分母或方差为零）'
    value = number(value, 'formatted scalar')
    pattern = ('+.3e' if signed else '.3e') if 0 < abs(value) < 1e-5 else ('+.6f' if signed else '.6f')
    return format(value, pattern)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *('| '+' | '.join(map(str, row))+' |' for row in rows)])


def render(validated, source):
    result, gate, decision, budget, by, means, paired, seed_paired = validated
    passed = gate['decision'] == 'GO'
    lines = ['# Phase28：双向原始证据重读正式内部矩阵', '',
        '## 实际完成与判断', '',
        '**30个拟合已完成：3个既有内部训练/验证划分 × 2个seed（31、47）× 5种结构，每个固定80轮，共2,400次参数更新。** '
        '独立回放已核对300个风险向量及90个外部风险参照；之后仅调用一次内部DSS结局入口。', '',
        '**锁定门结论：'+('GO，仅支持为后续确认另锁协议。' if passed else 'STOP，本轮C11未满足全部预设继续条件。')+'** '
        '该判定约束当前算子、表示、正则和固定预算，不等于证明双向重读、多模态互补信息或整个研究方向无效。', '',
        '本轮是重复内部架构筛选，**不是完整5外折×3seed确认，也不是独立患者OOF**。Clinical只进入冻结主风险，未直接条件化局部query。'
        '无文本桥、无辅助恢复或grounding监督、无组织学真值验证。', '',
        '## 统一评测', '',
        '主终点DSS：512人、65个events。每个seed先合并三分区within-fold分子/分母，再平均两个seed。Uno60的IPCW仅由相应fit分区估计。'
        'Harrell与Uno60分开判断；不平均患者风险、不混合跨分区风险尺度。**未计算确认性置信区间或bootstrap，以下不提供虚构区间。**', '']
    rows = []
    for name in ('CP', 'M0', 'RNA', *ARMS):
        rows.append([name, *[fmt(by[seed, name][metric]) for seed in SEEDS for metric in METRICS],
                     fmt(means[name]['Harrell']), fmt(means[name]['Uno60'])])
    lines += [table(['模型', 'H seed31', 'U60 seed31', 'H seed47', 'U60 seed47', 'H 两seed均值', 'U60 两seed均值'], rows), '',
        'CP为冻结Clinical＋S32＋CONCH32强锚点；RNA为复用的独立RNA风险；M0逐元素为冻结CP＋RNA。'
        'C00及其他四臂在相同offset之外另有可学习单模态头。自己的OFF是其已学习主效应，**不是M0，也不是独立重训C00**。', '',
        '## 两项主要比较及固定锚点防护', '']
    rows = []
    for a, b in (('C11', 'C00'), ('C11', 'DIRECT'), ('C11', 'M0'), ('C11', 'CP')):
        for metric in METRICS:
            deltas = [by[seed, a][metric]-by[seed, b][metric] for seed in SEEDS]
            rows.append([a+'−'+b, metric, fmt(deltas[0], True), fmt(deltas[1], True), fmt(sum(deltas)/2, True)])
    lines += [table(['比较', '指标', 'Δ seed31', 'Δ seed47', 'Δ 两seed均值'], rows), '',
        'C11−C00与C11−DIRECT分别要求两指标均值为正、且两个seed均正向；C11相对固定CP/M0的双指标均值不能下降。'
        '不选择最佳seed、缩放系数或表现最好的替代臂。', '', '## 方向消融与患者对应关系', '']
    rows = []
    for name in ('C10', 'C01'):
        rows.append(['C11−'+name, *[fmt(means['C11'][metric]-means[name][metric], True) for metric in METRICS], '仅诊断'])
    rows.append(['C11−C10−C01＋C00', *[fmt(means['C11'][metric]-means['C10'][metric]-means['C01'][metric]+means['C00'][metric], True)
                                      for metric in METRICS], '非生物学因果协同'])
    lines += [table(['方向比较', 'Δ Harrell', 'Δ Uno60', '解释范围'], rows), '']
    rows = []
    frozen = {row['comparison']: row for row in gate['frozen_pairing_guards']}
    for control in CONTROLS[1:]:
        name = 'C11_'+control
        status = ('通过' if frozen['C11-'+name]['pass'] else '未通过') if control in GATE_CONTROLS else '不进入继续门'
        rows.append(['C11−'+control, *[fmt(means['C11'][metric]-means[name][metric], True) for metric in METRICS], status])
    lines += [table(['冻结干预比较', 'Δ Harrell', 'Δ Uno60', '预设对应关系门'], rows), '',
        '路由错配仅替换另一患者的上下文，保留本人原始value、mask、已学习主效应及offset。均值上下文来自fit。'
        'WHOLE_OMICS_WRONG还交换组学测量及外部RNA风险，不能独占解释为路由效应。'
        '这些是冻结模型依赖性检验，不替代独立重训的必要性消融。', '',
        '## 配对排序：修复与损伤', '',
        '下表以各分区所有可比较生存患者对为分母，计入风险tie的半credit，分别按Harrell和Uno60权重统计。'
        'repair−damage＝net＝ΔC已由正式评测核对；它们不是只在基线错误组中的条件成功率。', '']
    rows = []
    for comparison in ('C11-C00', 'C11-DIRECT', 'C11-C10', 'C11-C01', 'C11-CP', 'C11-M0', 'C11-C11_OFF'):
        for metric in METRICS:
            row = paired[comparison, metric]
            rows.append([comparison, metric, fmt(row['repair']), fmt(row['damage']), fmt(row['net'], True)])
    lines += [table(['比较', '指标', 'repair credit比例', 'damage credit比例', 'net'], rows), '',
        '两项主要比较及方向消融的逐seed配对结果如下，避免只查看两个seed均值：', '']
    rows = []
    for comparison in ('C11-C00', 'C11-DIRECT', 'C11-C10', 'C11-C01'):
        for seed in SEEDS:
            for metric in METRICS:
                row = seed_paired[comparison, metric, seed]
                rows.append([comparison, seed, metric, fmt(row['repair']), fmt(row['damage']), fmt(row['net'], True)])
    lines += [table(['比较', 'seed', '指标', 'repair', 'damage', 'net'], rows), '',
        '## 训练预算与风险尺度', '',
        '**80轮是预锁预算，不是充分收敛证明。** 未用验证结局选择epoch，也没有在本轮补训或自动扩大搜索。', '']
    rows = []
    decreasing = not_minimum = 0
    for arm in ARMS:
        cells = [row for row in budget['rows'] if row['arm'] == arm]
        cox = [row['training']['cox_loss'] for row in cells]
        changes = [number(row['last10_preupdate_change'], 'last10 loss change') for row in cox]
        still_down = sum(value < 0 for value in changes)
        higher_than_min = sum(number(row['last_preupdate'], 'last loss') > number(row['minimum_preupdate'], 'minimum loss') for row in cox)
        decreasing += still_down; not_minimum += higher_than_min
        rms_fit = [number(row['risk_scales']['fit']['interaction']['RMS'], 'fit interaction RMS') for row in cells]
        rms_valid = [number(row['risk_scales']['valid']['interaction']['RMS'], 'valid interaction RMS') for row in cells]
        rows.append([arm, str(len(cells))+'×80', str(still_down)+'/6', fmt(min(changes), True)+' 至 '+fmt(max(changes), True),
                     fmt(sum(rms_fit)/6), fmt(sum(rms_valid)/6)])
    lines += [table(['结构', '完成拟合×轮数', '末10步loss下降', '末10步loss变化范围', 'fit交互RMS均值', 'valid交互RMS均值'], rows), '',
        f'{decreasing}/30个拟合的末10步训练Cox loss仍下降；{not_minimum}/30个拟合的末轮pre-update loss高于已观测最低值。'
        '末10步仍下降时，存在训练目标继续变化的观测；未下降也不能单独证明收敛。'
        '末轮高于最低值表示没有停留在已观测最低时点。这些量不能单独判断追加预算会提高内部或外部泛化。'
        '没有预注册收敛阈值，因此本报告不宣称预算充分或不足；固定预算下的性能和机制观察应共同解读。', '',
        'BUDGET_READOUT_AUDIT另列clip频率、数据梯度与含L2的总梯度、fit交叉拟合offset及valid完整fit offset分布、'
        '交互相对主效应尺度和跨臂主效应漂移。fit/valid来自不同患者，不能把两组分布差异当成同一患者的配对校准误差。'
        '无结局全pair排名扰动只表示风险敏感性，不称为生存修复/损伤。', '',
        '各臂共享冻结外部offset，但单模态编码与读出独立联合优化；C11−C00或C11−DIRECT的变化可能同时包含主效应漂移。'
        '不能把全部收益或损失直接归给重读差分。潜变量0.1修正边界不等于风险修正上界。', '',
        '## 可支持与不能支持的结论', '',
        ('本轮C11通过全部预锁原型门，支持另行设计确认实验；尚不能声称独立确认、组织学语义正确或临床可用。' if passed else
         '本轮C11未通过全部预锁原型门，应按上述具体比较报告限制，不自动推广为组学无独立信息、语义先验无价值或全部交互模型无效。'), '',
        '**优势：** 完整风险集Efron、固定共同offset、同预算方向消融、独立风险回放与配对排序闭合，能够区分实现故障、主效应漂移和已训练交互依赖。', '',
        '**风险与限制：** 65个DSS事件、重复内部探索、固定非凸训练预算、空间区域抽样和基因token压缩仍限制结论。'
        '没有独立组织学标注，注意力、null权重或教师分数均不构成医学真值。冻结敏感性不等同重新训练的必要性证据。', '',
        '**置信度：** 对回执、逐元素外部参照、回放及配对算术一致性的实现结论为高；对未来独立患者泛化、组织学含义及生物学机制尚无充分证据。'
        '此评级不对应未经估计的概率。', '',
        '## 后续边界', '',
        '不自动开启外折、增加模型深度、改变门控或搜索最优风险缩放。是否继续以及如何补训，需要结合本轮具体失败或正向机制，另锁协议。'
        '本报告没有新增科学判定门，也没有覆盖原P0资格报告。', '',
        '## 来源与隐私', '',
        f'- 冻结训练source：`{source}`。',
        f'- 30模型回执指纹：`{result["artifact_audit"]["receipt_fingerprint"]}`。',
        '- 精确输入及本报告SHA见 `FORMAL_DELIVERY.json`；不由source反推未核验的Git提交。',
        '- 仅公开匿名聚合；患者索引、风险向量、基因测量、坐标、检查点和凭据不进入本报告。', '']
    return '\n'.join(lines)


def generate(report_dir, source, *, script_sha256):
    directory = Path(report_dir)
    require(re.fullmatch('[0-9a-f]{64}', script_sha256) is not None, 'reporter executable SHA required')
    inputs, hashes = {}, {}
    for name in INPUT_NAMES: inputs[name], hashes[name] = read_json(directory/name)
    replay, replay_sha = read_json(directory/'REPLAY_AUDIT.json')
    validated = validate(inputs, replay, source)
    result, gate, _, budget, *_ = validated
    require(result['independent_replay']['report_sha256'] == replay_sha
            == budget['replay_report_sha256'], 'live independent replay receipt SHA differs')
    content = render(validated, source).encode('utf-8')
    hashes['RESULTS_FORMAL.md'] = sha_bytes(content)
    payload = {'schema': 'phase28_formal_delivery_v1', 'status': 'COMPLETE_INNER_FORMAL_REPORTS',
        'source': source, 'stage': 'prototype', 'neural_models': 30, 'seed_partition_cells': 6,
        'epochs_each': 80, 'optimizer_steps': 2400, 'script_path': 'scripts/report_phase28_formal.py',
        'script_sha256': script_sha256, 'source_commit_not_assumed': True,
        'report_sha256': hashes, 'receipt_fingerprint': result['artifact_audit']['receipt_fingerprint'],
        'independent_replay_report_sha256': replay_sha, 'scientific_decision': gate['decision'],
        'outer_started': False, 'confirmation_claimed': False, 'patient_level_artifacts_published': False,
        'bootstrap_draws': 0, 'new_predictor_fits': 0, 'endpoint_loaders_called': 0,
        'existing_P0_reports_overwritten': False, 'new_gate_or_rescaling_search': False}
    delivery = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    for name, raw in (('RESULTS_FORMAL.md', content), ('FORMAL_DELIVERY.json', delivery)):
        path = directory/name
        require(not path.exists() or path.read_bytes() == raw, 'existing formal report differs; refusing overwrite: '+name)
    require(all(sha(directory/name) == digest for name, digest in hashes.items() if name != 'RESULTS_FORMAL.md')
            and sha(directory/'REPLAY_AUDIT.json') == replay_sha, 'anonymous input changed while reporting')
    for name, raw in (('RESULTS_FORMAL.md', content), ('FORMAL_DELIVERY.json', delivery)):
        path = directory/name
        if not path.exists():
            temporary = path.with_name(path.name+'.partial')
            temporary.write_bytes(raw); temporary.replace(path)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--execute-report', action='store_true')
    args = parser.parse_args()
    require(args.execute_report, 'explicit --execute-report required after completed formal internal evaluation')
    script_hash = sha(Path(__file__))
    payload = generate(ROOT/'reports/phase28_bidir_reread', args.source, script_sha256=script_hash)
    require(sha(Path(__file__)) == script_hash, 'reporter executable changed during rendering')
    print('PHASE28_FORMAL_REPORT', payload['status'], payload['scientific_decision'], flush=True)


if __name__ == '__main__':
    main()

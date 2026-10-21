"""Summarize sixty anonymous Phase28 diagnostic JSONs without model access.

All histories/mechanisms bind to the thirty checkpoint SHAs already recorded
in anonymous INNER_RESULTS. No training, tensor, endpoint or checkpoint loader
is imported. Disabled reads and DIRECT's unused shifted reads remain distinct
from the evidence rereads that actually generated a state update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SEEDS, FOLDS = (31, 47), (1, 2, 3)
ARMS = ('C00', 'C10', 'C01', 'C11', 'DIRECT')
GRADIENTS = ('original_RNA_to_path_risk_L2', 'original_Ep_to_gene_risk_L2',
             'isolated_gene_context_to_path_risk_L2', 'isolated_path_context_to_gene_risk_L2')
CONTROLS = ('OFF', 'CONSTANT_VALUES', 'GENE_WRONG', 'PATH_WRONG', 'BOTH_WRONG')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'finite scalar required: '+name)
    return float(value)


def fingerprint(value, name):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'SHA256 required: '+name)
    return value


def reject_constant(value):
    raise ValueError('nonfinite JSON constant: '+value)


def load(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode('utf-8-sig'), parse_constant=reject_constant)
    require(isinstance(value, dict), 'anonymous JSON object required')
    return value, hashlib.sha256(raw).hexdigest()


def cell_stats(values):
    valid = [scalar(value, 'cell statistic') for value in values if value is not None]
    if not valid: return None
    mean = sum(valid)/len(valid)
    return {'mean_equal_cells': mean, 'SD_between_cells': math.sqrt(sum((value-mean)**2 for value in valid)/len(valid)),
            'minimum_cell': min(valid), 'maximum_cell': max(valid), 'available_cells': len(valid), 'expected_cells': 6}


def moment_mean(value, name, *, partial=False):
    if value is None: return None
    count = value.get('count') if isinstance(value, dict) else None
    require(type(count) is int and (1 <= count <= 3 if partial else count == 3),
            'geometry-example moment count differs: '+name)
    return scalar(value.get('mean'), name)


def moment_rms(value, name):
    require(isinstance(value, dict), 'three-example RMS moments required: '+name)
    moment_mean(value, name)
    result = scalar(value.get('RMS'), name+' RMS')
    require(result >= 0, 'negative RMS: '+name)
    return result


def association(left, right):
    pairs = [(scalar(a, 'association x'), scalar(b, 'association y')) for a, b in zip(left, right)
             if a is not None and b is not None]
    if len(pairs) < 2:
        return {'cells': len(pairs), 'Pearson_r': None, 'reason': 'fewer_than_two_defined_cells'}
    x, y = zip(*pairs); mx, my = sum(x)/len(x), sum(y)/len(y)
    denominator = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    value = None if denominator == 0 else sum((a-mx)*(b-my) for a, b in pairs)/denominator
    return {'cells': len(pairs), 'Pearson_r': value, 'reason': 'constant_cell_statistic' if value is None else None,
            'descriptive_only_nonindependent_cells': True, 'causal_or_performance_evidence': False}


def validate_reports(results, reports, source, diagnostic_script_sha):
    fingerprint(source, 'training source'); fingerprint(diagnostic_script_sha, 'diagnostic executable')
    require(results.get('source') == source and results.get('stage') == 'prototype'
            and results.get('status') == 'COMPLETE_BIDIRECTIONAL_REREAD_INNER_PROTOTYPE'
            and results.get('valid_endpoint_callback_calls') == 1, 'completed anonymous internal evaluation required')
    artifact = results.get('artifact_audit', {})
    require(artifact.get('status') == 'PASS_COMPLETE_CHECKPOINT_CONTRACT' and artifact.get('neural_models') == 30
            and artifact.get('optimizer_steps') == 2400, 'complete thirty-checkpoint audit required')
    replay = results.get('independent_replay', {})
    require(replay.get('status') == 'PASS' and replay.get('source') == source
            and replay.get('neural_models') == 30 and replay.get('risk_vectors_replayed') == 300,
            'independent model replay required')
    expected = {(seed, fold, arm) for seed in SEEDS for fold in FOLDS for arm in ARMS}
    bindings = artifact.get('model_bindings')
    require(isinstance(bindings, list) and len(bindings) == 30, 'thirty anonymous checkpoint bindings required')
    by = {(row.get('seed'), row.get('fold'), row.get('arm')): row for row in bindings}
    require(set(by) == expected, 'unique complete anonymous checkpoint bindings required')
    require(isinstance(reports, dict) and set(reports) == {(seed, fold, arm, mode) for seed, fold, arm in expected
                                                        for mode in ('history', 'mechanism')},
            'exact thirty history plus thirty mechanism JSONs required')
    for key, report in reports.items():
        seed, fold, arm, mode = key
        require(report.get('version') == 'phase28_postfit_diagnostics_v1'
                and report.get('status') == 'PASS_POSTFIT_'+mode.upper()+'_DIAGNOSTICS'
                and report.get('mode') == mode and report.get('source') == source,
                'diagnostic version/status/source differs')
        require((report.get('seed'), report.get('fold'), report.get('arm')) == (seed, fold, arm), 'diagnostic identity differs')
        require(report.get('script_path') == 'scripts/diagnose_phase28.py'
                and report.get('script_sha256') == diagnostic_script_sha, 'diagnostic executable SHA differs')
        expected_sha = fingerprint(by[seed, fold, arm].get('checkpoint_sha256'), 'reference checkpoint')
        require(report.get('checkpoint_sha256') == expected_sha, 'diagnostic checkpoint differs from independently replayed checkpoint')
        require(report.get('valid_endpoint_reads') == 0 and report.get('new_optimizer_steps') == 0
                and report.get('hyperparameters_or_checkpoint_changed') is False, 'diagnostic exceeded frozen outcome-blind scope')
        history = report.get('history', {})
        require(history.get('epochs') == 80 and history.get('learning_curve_records_are_pre_step') is True
                and history.get('checkpoint_or_epoch_selected_by_this_analysis') is False
                and history.get('budget_sufficient_or_global_convergence_claimed') is False, 'history diagnostic scope differs')
        interaction = history.get('saved_fit_risk_components', {}).get('interaction', {})
        require(scalar(interaction.get('RMS'), 'all-fit interaction RMS') >= 0, 'negative risk RMS')
        if mode == 'mechanism':
            mechanism = report.get('mechanism', {})
            require(mechanism.get('patients') == 3 and mechanism.get('selection') ==
                'fit_only_original_selected_Patch_count_min_lower_median_max;canonical_index_tiebreak',
                'only the predeclared three fit geometry examples are allowed')
            for field, wanted in (('valid_endpoint_reads', 0), ('valid_patient_forward_calls', 0),
                    ('fit_endpoint_getter_calls', 0), ('new_optimizer_steps', 0), ('model_weights_unchanged', True),
                    ('external_offset_unchanged', True), ('fit_geometry_examples_are_not_independent_validation', True),
                    ('gradient_nonzero_is_not_DSS_increment_evidence', True),
                    ('frozen_intervention_is_not_retrained_necessity_evidence', True)):
                require(mechanism.get(field) == wanted, 'mechanism diagnostic isolation failed: '+field)
            counts = mechanism.get('forbidden_API_call_counts')
            require(isinstance(counts, dict) and all(type(value) is int and value == 0 for value in counts.values()),
                    'forbidden diagnostic API called')
            require(mechanism.get('read_or_control_aggregation') == 'first_equal_WSI_within_patient_then_equal_three_geometry_examples',
                    'within-cell example/WSI aggregation differs')
            layers = mechanism.get('layers')
            require(isinstance(layers, list) and len(layers) == 2 and [row.get('round') for row in layers] == [1, 2],
                    'exact two diagnostic update rounds required')
    for key in expected:
        require(reports[*key, 'history']['history'] == reports[*key, 'mechanism']['history'],
                'history differs between CPU and mechanism receipts for the same checkpoint')
    return by


def summarize_arm(arm, reports):
    cells = [reports[seed, fold, arm, 'mechanism'] for seed in SEEDS for fold in FOLDS]
    mechanisms = [row['mechanism'] for row in cells]
    histories = [row['history'] for row in cells]
    require(all(all(name in row.get('gradient_sensitivities', {})
                    and row['gradient_sensitivities'][name] is not None for name in GRADIENTS) for row in mechanisms),
            'all four required cross-path gradients must be present')
    enabled_path, enabled_gene = arm in ('C10', 'C11', 'DIRECT'), arm in ('C01', 'C11', 'DIRECT')
    raw_reread_path, raw_reread_gene = enabled_path and arm != 'DIRECT', enabled_gene and arm != 'DIRECT'
    result = {'arm': arm, 'cells_equal_weighted': 6, 'rounds': [],
        'path_update_enabled': enabled_path, 'gene_update_enabled': enabled_gene,
        'all_fit_interaction_RMS': cell_stats([row['saved_fit_risk_components']['interaction']['RMS'] for row in histories]),
        'gradient_sensitivities': {name: cell_stats([moment_mean(row['gradient_sensitivities'].get(name), name)
                                                  for row in mechanisms]) for name in GRADIENTS},
        'frozen_fit_example_risk_response_RMS': {name: cell_stats([moment_rms(row['frozen_control_sensitivities'].get(name+'_minus_MATCHED'), name)
                                                                 for row in mechanisms]) for name in CONTROLS},
        'last10_parameter_gradient_norms': {}, 'read_risk_associations': {}}
    modules = sorted(set().union(*(row['data_versus_total_gradient_by_module'] for row in histories)))
    for module in modules:
        require(re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,63}', module) is not None, 'invalid module name')
        rows = [row['data_versus_total_gradient_by_module'].get(module) for row in histories]
        require(all(isinstance(row, dict) for row in rows), 'module gradient history must cover all six cells')
        result['last10_parameter_gradient_norms'][module] = {
            'data_gradient_L2': cell_stats([row['data_L2_last10_mean'] for row in rows]),
            'total_including_L2_gradient_L2': cell_stats([row['total_including_L2_last10_mean'] for row in rows]),
            'norm_difference_is_not_regularizer_gradient_norm': True}
    for index in range(2):
        layers = [row['layers'][index] for row in mechanisms]
        summary = {'round': index+1, 'routing': {}}
        for target, field, enabled, reread in (('path', 'path_equal_WSI', enabled_path, raw_reread_path),
                                                ('gene', 'gene', enabled_gene, raw_reread_gene)):
            rows = [row[field] for row in layers]
            required = ('TV_mean_all_valid_queries', 'TV_mean_queries_ge2_candidates',
                'eligible_query_fraction_ge2_candidates', 'evidence_read_difference_RMS',
                'anchored_token_change_RMS', 'anchored_token_change_maximum_absolute',
                'gate_mean', 'gate_fraction_lt_point01', 'gate_fraction_gt_point99')
            require(all(all(name in row and (row[name] is not None or name == 'TV_mean_queries_ge2_candidates')
                            for name in required) for row in rows), 'complete read/state/gate diagnostics required')
            computed = {name: cell_stats([moment_mean(row.get(name), name,
                partial=name == 'TV_mean_queries_ge2_candidates') for row in rows]) for name in (
                'TV_mean_all_valid_queries', 'TV_mean_queries_ge2_candidates', 'eligible_query_fraction_ge2_candidates')}
            computed['at_least_two_candidate_example_count_per_cell'] = cell_stats([
                0 if row.get('TV_mean_queries_ge2_candidates') is None else row['TV_mean_queries_ge2_candidates']['count']
                for row in rows])
            summary[target] = {'update_enabled': enabled, 'raw_evidence_reread_used_as_delta_input': reread,
                'operative_raw_read_TV_all_valid': computed['TV_mean_all_valid_queries'] if reread else None,
                'operative_raw_read_TV_at_least_two_candidates': computed['TV_mean_queries_ge2_candidates'] if reread else None,
                'computed_read_diagnostics_not_necessarily_used': computed,
                'delta_input_type': ('projected_cross_modal_message' if arm == 'DIRECT' else 'conditional_minus_base_evidence_read'),
                'computed_delta_input_RMS': cell_stats([moment_mean(row.get('evidence_read_difference_RMS'), 'legacy named delta input RMS') for row in rows]),
                'anchored_token_change_RMS': cell_stats([moment_mean(row.get('anchored_token_change_RMS'), 'token RMS') for row in rows]),
                'anchored_token_change_max_abs': cell_stats([moment_mean(row.get('anchored_token_change_maximum_absolute'), 'token max') for row in rows]),
                'gate': {name: cell_stats([moment_mean(row.get('gate_'+name), 'gate '+name) for row in rows])
                         for name in ('mean', 'fraction_lt_point01', 'fraction_gt_point99')}}
        for name in ('path_target_null_probability_mean', 'gene_target_null_probability_mean'):
            values = [moment_mean(row['routing_equal_WSI'].get(name), name) for row in layers]
            require(all(value is not None and 0 <= value <= 1 for value in values), 'null probability outside [0,1]')
            summary['routing'][name] = cell_stats(values)
        result['rounds'].append(summary)
    for target, field, use, wrong in (('path', 'path_equal_WSI', raw_reread_path, 'GENE_WRONG'),
                                       ('gene', 'gene', raw_reread_gene, 'PATH_WRONG')):
        if use:
            tv = [moment_mean(row['layers'][1][field].get('TV_mean_all_valid_queries'), 'round2 TV') for row in mechanisms]
            token = [moment_mean(row['layers'][1][field].get('anchored_token_change_RMS'), 'round2 token change') for row in mechanisms]
            risk = [moment_rms(row['frozen_control_sensitivities'].get(wrong+'_minus_MATCHED'), 'wrong risk') for row in mechanisms]
            fit = [row['saved_fit_risk_components']['interaction']['RMS'] for row in histories]
            result['read_risk_associations'][target] = {
                'round2_TV_vs_wrong_context_risk_RMS': association(tv, risk),
                'round2_token_change_vs_all_fit_interaction_RMS': association(token, fit)}
        else:
            result['read_risk_associations'][target] = None
    return result


def fmt(value):
    if value is None: return 'N/A'
    x = scalar(value, 'formatted value')
    return format(x, '.3e' if 0 < abs(x) < 1e-5 else '.6f')


def mean(stat):
    return None if stat is None else stat['mean_equal_cells']


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *('| '+' | '.join(map(str, row))+' |' for row in rows)])


def render(arms, source):
    lines = ['# Phase28：双向重读机制诊断汇总', '',
        '## 证据范围', '',
        '本报告只汇总30份history和30份mechanism匿名JSON，并与正式内部评测记录的30个检查点SHA绑定。'
        '每臂为3个内部划分×2个seed，共6个cell等权；每个mechanism cell仅检查按Patch数量预定的3例fit患者，先等WSI、再等例聚合。'
        '**这3例不是独立验证集，不代表全部患者的组织形态分布。**', '',
        '以下先取每cell既有统计量，再对6个cell求算术均值，不混成全患者池化统计。特定指标不可计算时列出可用cell数量，缺失不置零。'
        '本报告不设新门、不读取生存结局、不重新拟合，也不改变正式GO/STOP。', '',
        '## 哪些读取真正进入了更新', '',
        table(['臂', '组学→病理', '病理→基因', '原始证据重读TV的解释'], [
            ['C00', '关闭', '关闭', 'shifted read虽计算，但未启用；operative TV记N/A'],
            ['C10', '启用', '关闭', '仅病理重读TV参与解释'],
            ['C01', '关闭', '启用', '仅基因重读TV参与解释'],
            ['C11', '启用', '启用', '两个方向的原始证据重读均使用'],
            ['DIRECT', '普通消息', '普通消息', 'wc虽计算，但delta输入为投影消息；原始重读TV全部N/A']]), '',
        'DIRECT的既有字段 `evidence_read_difference_RMS` 实际对应投影跨模态消息，不能称为原始证据读取差分。'
        '关闭方向的null、门和候选读取也可能有数值，这些计算不表示模型已经使用它们更新状态。', '',
        '## 两轮实际读取、状态与null/门', '']
    rows = []
    for arm in arms:
        for layer in arm['rounds']:
            for target, label in (('path', '病理'), ('gene', '基因')):
                row = layer[target]
                tv = row['operative_raw_read_TV_all_valid']
                tv2 = row['operative_raw_read_TV_at_least_two_candidates']
                null = layer['routing']['path_target_null_probability_mean' if target == 'path' else 'gene_target_null_probability_mean']
                rows.append([arm['arm'], layer['round'], label, '开' if row['update_enabled'] else '关',
                    fmt(mean(tv)), fmt(mean(tv2)), 0 if tv2 is None else tv2['available_cells'],
                    fmt(mean(row['anchored_token_change_RMS'])), fmt(mean(row['gate']['mean'])), fmt(mean(null))])
    lines += [table(['臂', '轮', '目标', '更新', 'operative TV全有效query', 'operative TV≥2候选', 'TV≥2可用cells',
                     'token修正RMS', '门均值', 'null均值'], rows), '',
        'TV是同一原始证据支持内条件读取与基础读取的总变差距离。单候选区域无法改变读取分配，TV必为零；'
        '因此同时列出至少两个候选的统计。token修正为锚定初始表示后的实际状态变化。'
        'null是可学习的拒绝交互权重，受候选数与score尺度影响，**不是校准医学置信度**。', '',
        '≥2候选TV在每个cell内只对可计算的几何样例取均值；摘要同时保留可用例数，不能假定3例都可定义。', '',
        '门的极端比例、实际delta输入RMS、token最大幅度及未使用但计算出的TV另存机器可读摘要；不能把计算存在当成路径启用。', '',
        '## 交叉梯度与最终风险响应', '']
    rows = []
    for arm in arms:
        gradients = arm['gradient_sensitivities']; controls = arm['frozen_fit_example_risk_response_RMS']
        rows.append([arm['arm'], *[fmt(mean(gradients[name])) for name in GRADIENTS],
                     fmt(mean(arm['all_fit_interaction_RMS'])), *[fmt(mean(controls[name])) for name in ('GENE_WRONG', 'PATH_WRONG', 'BOTH_WRONG')]])
    lines += [table(['臂', 'rawRNA→病理风险∥grad∥₂', 'Ep→基因风险∥grad∥₂', 'isolated G context→病理风险',
        'isolated P context→基因风险', '全fit交互RMS', '错G风险差RMS（3例）', '错P风险差RMS（3例）', '双错风险差RMS（3例）'], rows), '',
        '**非零梯度说明局部敏感性，不等于预测收益。** rawRNA/Ep路径范数受输入单位、证据数量及其他可微路径影响；'
        'rawRNA在这里指模型的逐基因输入，已作训练分区标准化，并非未经处理的测序counts。'
        'isolated context梯度先固定本人原始证据、分离上下文bank，检查路由条件的局部响应。不同梯度的数值不能直接当作模态重要性排名。', '',
        '全fit交互RMS来自已保存的整个fit风险向量；错患者风险差RMS仅来自3例fit冻结干预。'
        '两个分母不同，不能混成同一患者总体或当作验证增量。冻结干预保留本人value与主效应，不等于真实重训的必要性检验。', '',
        '## 读取变化与风险变化的描述性联系', '']
    rows = []
    for arm in arms:
        for target, label in (('path', '组学条件病理重读'), ('gene', '病理条件基因重读')):
            association_row = arm['read_risk_associations'][target]
            if association_row is None:
                rows.append([arm['arm'], label, 'N/A', 'N/A', '未启用原始证据重读'])
            else:
                a = association_row['round2_TV_vs_wrong_context_risk_RMS']
                b = association_row['round2_token_change_vs_all_fit_interaction_RMS']
                rows.append([arm['arm'], label, fmt(a['Pearson_r']), fmt(b['Pearson_r']), str(a['cells'])+' cells'])
    lines += [table(['臂', '路径', '轮2 TV与错配风险RMS相关r', '轮2 token变化与全fit交互RMS相关r', '范围'], rows), '',
        '这些相关只描述6个cell的聚合量，cell共享患者且不是独立样本；常量统计的相关记N/A。'
        '不报告未经估计的置信区间或p值，不将相关、注意力变化或状态变化称为组织学对应、因果机制或DSS收益。', '',
        '## 末10轮数据梯度与总梯度', '']
    rows = []
    for arm in arms:
        for module, row in arm['last10_parameter_gradient_norms'].items():
            rows.append([arm['arm'], module, fmt(mean(row['data_gradient_L2'])), fmt(mean(row['total_including_L2_gradient_L2']))])
    lines += [table(['臂', '模块', 'data梯度L2范数末10轮均值', '含L2正则的总梯度范数末10轮均值'], rows), '',
        '**两个梯度范数的差不是L2正则梯度的范数。** 向量可以相互抵消或增强，因此没有用norm差构造所谓“正则贡献”。'
        '休眠方向也可能因参数L2出现总梯度；只有数据梯度非零仍不足以证明存在可泛化的条件增量。', '',
        '## 判断边界', '',
        '**优势：** 同一已保存模型可串联检查读取TV、实际token变化、路径隔离梯度与风险响应，避免只打印总梯度或注意力图。', '',
        '**限制：** 仅3例fit几何样例，区域子采样、基因集合压缩、缺测与尺度均影响机制统计；没有原图组织学QA或独立局部标签。'
        '是否改善生存排序仍由正式内部评测的预锁比较回答。风险主效应可随联合训练漂移，也不能用这些机制统计单独归因全部ΔC。', '',
        '**置信度：** 对来源绑定和这些聚合算术的可追溯性为高；对患者外泛化、医学语义及因果解释没有独立验证证据。', '',
        f'冻结训练source：`{source}`。输入60份JSON与正式内部结果的SHA见 `MECHANISM_SUMMARY.json`。'
        '不公开患者索引、风险向量、基因数组、坐标或检查点。', '']
    return '\n'.join(lines)


def generate(report_dir, source, *, diagnostic_script_sha256, summarizer_script_sha256):
    directory = Path(report_dir)
    fingerprint(summarizer_script_sha256, 'summarizer executable')
    results, results_sha = load(directory/'INNER_RESULTS.json')
    reports, input_hashes = {}, {'INNER_RESULTS.json': results_sha}
    for seed in SEEDS:
        for fold in FOLDS:
            for arm in ARMS:
                for mode in ('history', 'mechanism'):
                    relative = f'diagnostics/seed{seed}_fold{fold}_{arm}_{mode}.json'
                    reports[seed, fold, arm, mode], input_hashes[relative] = load(directory/relative)
    require({path.name for path in (directory/'diagnostics').glob('*.json')}
            == {Path(name).name for name in input_hashes if name.startswith('diagnostics/')},
            'unexpected diagnostic JSON outside the exact thirty-plus-thirty matrix')
    bindings = validate_reports(results, reports, source, diagnostic_script_sha256)
    arms = [summarize_arm(arm, reports) for arm in ARMS]
    markdown = render(arms, source).encode('utf-8')
    payload = {'schema': 'phase28_anonymous_mechanism_summary_v1', 'status': 'COMPLETE_SIXTY_POSTFIT_DIAGNOSTICS',
        'source': source, 'stage': 'prototype', 'diagnostic_script_path': 'scripts/diagnose_phase28.py',
        'diagnostic_script_sha256': diagnostic_script_sha256, 'script_path': 'scripts/report_phase28_mechanism.py',
        'script_sha256': summarizer_script_sha256, 'history_reports': 30, 'mechanism_reports': 30,
        'models': 30, 'input_report_sha256': input_hashes,
        'checkpoint_bindings': [{'seed': key[0], 'fold': key[1], 'arm': key[2],
            'checkpoint_sha256': bindings[key]['checkpoint_sha256']} for key in sorted(bindings)],
        'MECHANISM_RESULTS_md_sha256': hashlib.sha256(markdown).hexdigest(), 'arms': arms,
        'aggregation': 'within_cell_existing_equal_WSI_then_three_fit_examples;equal_six_cells_per_arm;available_cell_denominator_explicit',
        'new_forward_calls': 0, 'new_training_steps': 0, 'endpoint_loaders_called': 0,
        'new_scientific_gate': False, 'effectiveness_or_histology_or_causality_claimed': False,
        'norm_difference_used_as_regularizer_gradient_norm': False,
        'original_FORMAL_DELIVERY_contract_modified': False, 'patient_arrays_or_checkpoints_published': False}
    summary = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    for relative, digest in input_hashes.items(): require(sha(directory/relative) == digest, 'anonymous diagnostic changed during summary')
    for name, raw in (('MECHANISM_RESULTS.md', markdown), ('MECHANISM_SUMMARY.json', summary)):
        path = directory/name
        require(not path.exists() or path.read_bytes() == raw, 'existing mechanism summary differs; refusing overwrite')
    for name, raw in (('MECHANISM_RESULTS.md', markdown), ('MECHANISM_SUMMARY.json', summary)):
        path = directory/name
        if not path.exists():
            temporary = path.with_name(path.name+'.partial'); temporary.write_bytes(raw); temporary.replace(path)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--execute-report', action='store_true')
    args = parser.parse_args()
    require(args.execute_report, 'explicit --execute-report required after sixty diagnostics and formal internal evaluation')
    diagnostic_script = ROOT/'scripts/diagnose_phase28.py'
    diagnostic_sha, own_sha = sha(diagnostic_script), sha(Path(__file__))
    payload = generate(ROOT/'reports/phase28_bidir_reread', args.source,
        diagnostic_script_sha256=diagnostic_sha, summarizer_script_sha256=own_sha)
    require(sha(diagnostic_script) == diagnostic_sha and sha(Path(__file__)) == own_sha, 'summary or diagnostic executable changed')
    print('PHASE28_MECHANISM_REPORT', payload['status'], payload['models'], flush=True)


if __name__ == '__main__':
    main()

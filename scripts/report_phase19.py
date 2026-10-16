"""Report verified, completed Phase19 experiments; never synthesize results.

Evaluation is independent in validate_phase19.py. This script changes neither
the protocol nor training identity, and cannot run before completion gates.
"""
import json
import importlib.metadata
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase19_attention.data import P, R, cfg, source, write
from src.phase10_trimodal.data import digest

METRICS = ['Harrell', 'Uno60']
PRIMARY = ['A3-' + arm for arm in ['A0', 'A1', 'A2', 'A4', 'A5', 'A6']]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def text(path, value):
    Path(path).write_text(value.rstrip() + '\n', encoding='utf-8')


def table(frame, columns=None):
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if columns is not None:
        frame = frame[columns]
    if frame.empty:
        return '无对应记录。\n'
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return 'NA' if not np.isfinite(value) else f'{value:.6g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join('---' for _ in frame.columns) + ' |']
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines) + '\n'


def gate_inputs():
    identity, config = source(), cfg()
    lock = read(R/'PROTOCOL_LOCK.json')
    done = read(P/'TRAINING_DONE.json')
    require(lock['source'] == identity == done['source'], 'source lock/completion mismatch')
    require(done['status'] == 'PASS', 'formal training not complete')
    require(len(config['arms']) == 10 and len(config['seeds']) == 3 and len(config['folds']) == 5,
            'locked finite experiment matrix changed')
    gates = {}
    for name in ['DATA_AUDIT.json', 'DATA_LEAKAGE_AUDIT.json', 'OPERATOR_AND_PIPELINE_TESTS.json',
                 'NATIVE_SOURCE_AUDIT.json', 'NATIVE_QUALIFICATION.json',
                 'INDEPENDENT_VALIDATION.json', 'CHECKPOINT_REPLAY.json',
                 'BOOTSTRAP_AUDIT.json', 'CONTRAST_POINT_AUDIT.json', 'FIT_AND_REFERENCE_COUNTS.json',
                 'BUDGET_VERIFICATION.json', 'PROBE_COVERAGE_AUDIT.json', 'MMP_TOKEN_DIAGNOSTICS.json']:
        value = read(R/name)
        require(value.get('status') == 'PASS', 'qualification gate not PASS: ' + name)
        if 'source' in value:
            require(value['source'] == identity, 'stale gate source: ' + name)
        gates[name] = value
    require(gates['OPERATOR_AND_PIPELINE_TESTS.json']['training_source'] == identity,
            'stale operator/pipeline test source')
    replay = gates['CHECKPOINT_REPLAY.json']
    require(replay['new_neural_checkpoints'] == 300 and replay['modes_per_checkpoint'] == 5,
            'checkpoint replay matrix incomplete')
    require(replay['maximum_error'] < 2e-6 and replay['own_main_off_maximum_error'] < 5e-6,
            'checkpoint/main-effect replay failed')
    boot = gates['BOOTSTRAP_AUDIT.json']
    require(boot['draws'] == 20000 and boot['primary_family_tests'] == 12 and boot['valid'] >= 19800,
            'paired bootstrap qualification differs from locked family')
    mmp = gates['MMP_TOKEN_DIAGNOSTICS.json']
    require(mmp['all_ten_role_fold_caches_present'] and mmp['current_source_successful_private_caches_audited'] == 10
            and not mmp['outcomes_read'] and not mmp['parameters_or_training_source_changed']
            and mmp['models_retrained'] == 0 and mmp['NaN_count'] == mmp['Inf_count'] == 0,
            'complete frozen label-blind MMP audit required')
    require(mmp['CSV_sha256'] == digest(R/'MMP_TOKEN_DIAGNOSTICS.csv')
            and mmp['diagnostic_script_sha256'] == digest(ROOT/'scripts/diagnose_phase19.py'),
            'MMP diagnostic data/code identity changed')
    coverage = gates['PROBE_COVERAGE_AUDIT.json']
    require(len(coverage['rows']) == 20 and coverage['additional_forward_or_gradient_calls'] == 0,
            'checkpoint metadata-only probe coverage audit incomplete')
    return identity, config, done, gates


def capture_environment(identity):
    """Only explicitly named, non-secret runtime/version fields are collected."""
    import torch
    packages = {}
    for name in ['numpy','scipy','pandas','scikit-learn','scikit-survival','einops']:
        packages[name] = importlib.metadata.version(name)
    result = {'status':'PASS', 'source':identity,
        'python_version':platform.python_version(), 'python_executable':sys.executable,
        'operating_system':platform.system(), 'machine':platform.machine(),
        'torch_version':str(torch.__version__), 'torch_compiled_cuda_version':torch.version.cuda,
        'packages':packages,
        'GPU':[],
        'locked_training_settings':{'provenance':'hash-locked train.py::seed_all and configured GPU launch',
            'deterministic_algorithms':True, 'cuda_matmul_allow_tf32':False,
            'cudnn_allow_tf32':False, 'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        'report_process_settings':{'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        'scope':'report-time actual package/GPU information; locked training settings and current report-process flags separately identified; no environment dump'}
    query = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                     '--format=csv,noheader,nounits'], text=True)
    for line in query.strip().splitlines():
        name, driver, memory = [part.strip() for part in line.split(',')]
        result['GPU'].append({'name':name,'driver_version':driver,'memory_total_MiB':float(memory)})
    require(result['GPU'], 'actual GPU metadata unavailable')
    write(R/'ENVIRONMENT.json', result)
    return result


def scientific_decision(seed, paired, budget, config):
    primary = paired[paired.family == 'primary'].copy()
    require(len(primary) == 12, 'primary family must have six comparisons by two metrics')
    require(set(zip(primary.comparison, primary.metric)) == {(p, m) for p in PRIMARY for m in METRICS},
            'primary family identities differ')
    require((primary.family_tests == 12).all(), 'primary Bonferroni family differs')
    require(not seed.duplicated(['arm', 'seed']).any(), 'duplicate seed metric rows')
    means = seed.groupby('arm')[METRICS].mean()
    checks = []
    for row in primary.to_dict('records'):
        a, b = row['comparison'].split('-', 1)
        expected = float(means.loc[a, row['metric']] - means.loc[b, row['metric']])
        require(abs(expected - row['delta']) < 1e-11, 'pair point not mean of within-seed C difference')
        checks.append({'comparison': row['comparison'], 'metric': row['metric'], 'delta': row['delta'],
                       'positive_seeds': int(row['positive_seeds']),
                       'direction_and_seed_pass': row['delta'] > 0 and row['positive_seeds'] >= 2,
                       'Bonferroni_positive': row['Bonferroni_low'] > 0})
    strong = []
    for reference in config['engineering_gate']['no_mean_harm_against']:
        require(reference in means.index, 'missing locked strong comparator')
        for metric in METRICS:
            delta = float(means.loc['A3', metric] - means.loc[reference, metric])
            strong.append({'reference': reference, 'metric': metric, 'delta_A3_minus_reference': delta,
                           'no_mean_harm': delta >= 0})
    direction = all(q['direction_and_seed_pass'] for q in checks)
    no_harm = all(q['no_mean_harm'] for q in strong)
    mechanism_budget = bool(budget['mechanism_budget_sufficient'])
    all_budget = bool(budget['all_method_budgets_sufficient'])
    # Whole-study status is conservative when a locked comparator remains
    # budget-censored. Separate mechanism-only status preserves that distinction.
    mechanism_status = ('BUDGET_UNRESOLVED' if not mechanism_budget else
        'DIRECTIONAL_CANDIDATE' if direction and no_harm else
        'MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE')
    status = mechanism_status if all_budget else 'BUDGET_UNRESOLVED'
    return {'status': status, 'mechanism_only_status': mechanism_status,
            'mechanism_budget_sufficient': mechanism_budget, 'all_method_budgets_sufficient': all_budget,
            'all_primary_direction_and_seed_gate': direction, 'strong_reference_no_mean_harm': no_harm,
            'primary_checks': checks, 'strong_reference_checks': strong,
            'corrected_positive_primary_statistics': [q['comparison'] + ':' + q['metric'] for q in checks if q['Bonferroni_positive']],
            'primary_family_tests': 12, 'corrected_interval_coverage': 1-.05/12,
            'independent_external_confirmation': False,
            'inference': 'exploratory, repeatedly developed TCGA cohort; CP equals a clinical-modulated low-rank bilinear/FiLM kernel',
            'next_action': 'stop automatic expansion; report observed mechanism, budget and representation limitations'}


def execution_counts(config, done, manifest, compute, fit_counts):
    neural = manifest[manifest.fit_classification == 'new_neural_fit']
    convex = manifest[manifest.fit_classification == 'new_unique_convex_fit']
    require(len(neural) == 300 and len(convex) == 20, 'formal manifest incomplete')
    require(not neural.duplicated(['arm', 'seed', 'fold', 'role']).any(), 'duplicate formal neural fits')
    for role in ['inner', 'outer']:
        require((neural.role == role).sum() == 150, 'formal role matrix incomplete')
    require(len(compute) == 300 and not compute.duplicated(['arm','seed','fold','role']).any(),
            'formal compute rows incomplete')
    require(fit_counts['new_inner_neural_fits'] == 150 and fit_counts['new_outer_neural_fits'] == 150
            and fit_counts['new_unique_convex_models'] == 20, 'independent fit counts differ')
    ledger = read(P/'SMOKE_LEDGER.json')
    smoke = read(P/'SMOKE_COUNTS.json')
    attempts = ledger['attempts']
    positive = sum(int(q.get('generator_steps', 0) > 0) for q in attempts)
    generator = sum(int(q.get('generator_steps', 0)) for q in attempts)
    discriminator = sum(int(q.get('discriminator_steps', 0)) for q in attempts)
    returned = sum(q['status'] == 'RETURNED_SUCCESSFULLY' for q in attempts)
    require(smoke['real_short_fits'] == positive and smoke['completed_short_fits'] == returned
            and smoke['generator_optimizer_steps'] == generator
            and smoke['discriminator_optimizer_steps'] == discriminator, 'smoke incremental ledger disagrees')
    require(done['new_formal_fits'] == 300 and done['unique_convex_reference_fits'] == 20,
            'training completion counts differ')
    prototypes = []
    for receipt in sorted((P/'baseline_features').glob('MMP_*.json')):
        value = read(receipt)
        path = receipt.with_suffix('.pt')
        require(path.is_file() and digest(path) == value['sha256'], 'MMP prototype cache checksum mismatch')
        provenance = value['provenance']
        parent_source = provenance['preprocess_audit'].get('source')
        prototypes.append({'role': provenance['role'], 'fold': provenance['fold'],
                           'current_source': parent_source == done['source'],
                           'source': parent_source, 'cache_sha256': value['sha256'],
                           'n_init': provenance['parameters']['kmeans_n_init'],
                           'prototype_count': provenance['parameters']['prototypes']})
    current = [q for q in prototypes if q['current_source']]
    require(len(current) == 10 and {(q['role'], q['fold']) for q in current} ==
            {(r, f) for r in ['inner','outer'] for f in config['folds']},
            'MMP must have ten unique current-source fit-local prototype caches')
    formal_steps = int(compute.actual_epochs.sum())
    formal_disc = int(compute.discriminator_optimizer_steps.sum())
    return {'status': 'PASS', 'source': done['source'],
        'new_formal_inner_neural_fits': 150, 'new_formal_outer_neural_fits': 150,
        'new_formal_neural_fits': 300, 'formal_generator_optimizer_steps': formal_steps,
        'formal_discriminator_optimizer_steps': formal_disc,
        'unique_current_source_convex_fits': 20,
        'convex_seed_role_metric_references': int(fit_counts['convex_seed_role_metric_references']),
        'historical_LOCAL_outer_metric_references': int(fit_counts['historical_LOCAL_outer_references']),
        'old_125_inner_pilot_new_fits_or_OOF': 0,
        'smoke_attempts_recorded': len(attempts), 'smoke_real_short_fits_with_generator_steps': positive,
        'smoke_returned_successfully': returned,
        'smoke_failed_after_generator_steps': sum(q['status'] != 'RETURNED_SUCCESSFULLY' and q.get('generator_steps',0)>0 for q in attempts),
        'smoke_generator_optimizer_steps': generator, 'smoke_discriminator_optimizer_steps': discriminator,
        'all_recorded_real_neural_fits': 300+positive,
        'all_recorded_generator_optimizer_steps': formal_steps+generator,
        'all_recorded_discriminator_optimizer_steps': formal_disc+discriminator,
        'verified_completed_MMP_prototype_cache_producers_current_source': len(current),
        'verified_completed_MMP_prototype_cache_producers_other_source': len(prototypes)-len(current),
        'MMP_internal_kmeans_initializations_current_source': sum(q['n_init'] for q in current),
        'MMP_fit_attempt_ledger_available': False,
        'MMP_failed_or_aborted_attempt_count': 'NOT_RECORDED',
        'MMP_prototype_cache_audit': prototypes,
        'clustering_scope': 'verified completed fit-only KMeans cache producers only, not all attempts; three seed consumers reuse them; frozen per-WSI PANTHER EM is not another cross-patient KMeans fit',
        'fit_definition': 'neural fit counted once if generator optimizer stepped; D optimizer steps separate; deterministic convex and unsupervised prototype fits separate',
        'prior_stage_training_recounted_as_new': False,
        'formal_failed_task_receipt_present': (P/'FAILED.json').exists(),
        'formal_fit_count_scope': '300 complete checkpoint+receipt fits; partial failed formal attempts, if any, require their own execution evidence and are not inferred as zero'}


def gpu_profile():
    rows, telemetry = [], []
    for role in ['inner', 'outer']:
        samples = read(P/(role+'_GPU.json'))
        require(len(samples) > 1, 'insufficient GPU telemetry for ' + role)
        x = pd.DataFrame(samples).sort_values('unix_time')
        t = x.unix_time.to_numpy(float)
        duration = np.diff(t)
        require((duration >= 0).all() and duration.sum() > 0, 'invalid GPU sampling timestamps')
        row = {'role': role, 'samples': len(x), 'sampled_interval_seconds': float(duration.sum()),
               'maximum_running_workers': int(x.running.max()),
               'maximum_configured_workers_limit': int(x.workers_limit.max())}
        for key, label in [('GPU_percent', 'GPU_utilization_percent'), ('memory_MiB', 'whole_GPU_memory_MiB')]:
            values = x[key].to_numpy(float)
            require(np.isfinite(values).all(), 'nonfinite GPU sample')
            row[label+'_time_weighted_mean'] = float(np.average(values[:-1], weights=duration))
            row[label+'_sample_maximum'] = float(values.max())
            row[label+'_sample_median'] = float(np.median(values))
        rows.append(row)
        telemetry.extend(samples)
    pd.DataFrame(rows).to_csv(R/'GPU_PROFILE.csv', index=False)
    write(R/'GPU_PROFILE.json', {'status': 'PASS', 'rows': rows,
        'weighting': 'left-held observed value over adjacent scheduler timestamps; final sample has no assumed duration',
        'scope': 'whole card; sampled inner/outer intervals, not all preparation/replay or end-to-end wall time; can include concurrent processes',
        'memory_unit': 'MiB=bytes/2^20; occupancy is not scientific validity',
        'no_dummy_allocation_to_inflate_occupancy': True})
    return rows


def report():
    identity, config, done, gates = gate_inputs()
    seed = pd.read_csv(R/'SEED_METRICS.csv')
    fold = pd.read_csv(R/'FOLD_METRICS.csv')
    paired = pd.read_csv(R/'PAIRED_COMPARISONS.csv')
    per_seed = pd.read_csv(R/'SEED_PAIRED_CONTRASTS.csv')
    compute = pd.read_csv(R/'COMPUTE_PROFILE.csv')
    manifest = pd.read_csv(R/'CELL_MANIFEST.csv')
    mechanism = pd.read_csv(R/'ATTENTION_MECHANISM.csv')
    budget = read(R/'BUDGET_VERIFICATION.json')
    require(budget['status'] == 'PASS', 'budget audit itself failed')
    preouter_budget = read(R/'BUDGET_AUDIT.json')
    require(preouter_budget['source'] == identity and preouter_budget['no_outer_results_used'],
            'preouter budget evidence missing/stale')
    decision = scientific_decision(seed, paired, budget, config)
    decision['source'] = identity
    counts = execution_counts(config, done, manifest, compute, gates['FIT_AND_REFERENCE_COUNTS.json'])
    gpu = gpu_profile()
    environment = capture_environment(identity)
    summary = seed.groupby('arm')[METRICS].agg(['mean','std','min','max'])
    display = pd.DataFrame({'arm': summary.index,
        **{m:summary[m]['mean'].to_numpy() for m in METRICS},
        **{m+'_seed_SD':summary[m]['std'].to_numpy() for m in METRICS},
        **{m+'_seed_min':summary[m]['min'].to_numpy() for m in METRICS},
        **{m+'_seed_max':summary[m]['max'].to_numpy() for m in METRICS}}).set_index('arm')
    order = config['arms'] + config['ridge_references'] + ['HIST_CS_S32_CONCH32']
    require(set(order) <= set(display.index), 'main model table incomplete')
    primary = paired[paired.family == 'primary']
    secondary = paired[paired.family == 'secondary']
    resource = compute.groupby(['role','arm']).agg(cells=('fold','count'),
        registered_parameters=('registered_parameters','max'),
        last_step_gradient_parameters_median=('last_step_gradient_parameters','median'),
        discriminator_parameters=('discriminator_parameters','max'),
        actual_epochs_median=('actual_epochs','median'), selected_epoch_median=('selected_epoch','median'),
        seconds_sum=('seconds','sum'), peak_GPU_MiB_max=('peak_GPU_MiB','max')).reset_index()
    selected = pd.read_csv(R/'TRAINING_CURVES.csv')
    selected = selected[(selected.role == 'inner') & (selected.epoch == selected.selected_epoch)]
    gradient_columns = [c for c in selected.columns if c.endswith('_gradient')]
    selected_grad = selected.groupby('arm')[gradient_columns].median().reset_index()
    mechanism_keys = ['main_risk_SD','interaction_risk_SD','matched_wrongC_risk_MAE','matched_wrongO_risk_MAE',
        'clinical_attention_relative_L2','base_scores_SD','trilinear_scores_SD',
        'actual_combined_scores_SD','actual_score_correction_SD',
        'p_to_o_entropy_mean','p_to_o_maximum_weight_mean','o_to_p_entropy_mean','o_to_p_maximum_weight_mean',
        'isolated_interaction_clinical_gradient_L2','isolated_interaction_RNA_gradient_L2']
    mechanism_keys = [k for k in mechanism_keys if k in mechanism.columns]
    mech = mechanism.groupby(['role','arm'])[mechanism_keys].median().reset_index()
    pfi = pd.read_csv(R/'PFI_SEED_METRICS.csv').groupby('arm')[METRICS].mean()
    pfi_fold = pd.read_csv(R/'PFI_FOLD_METRICS.csv')
    single = pfi_fold[(pfi_fold.arm == 'A3') & (pfi_fold.seed == config['seeds'][0])]
    require(len(single) == 5, 'PFI transfer coverage incomplete')
    cohort = fold[(fold.arm == 'A3') & (fold.seed == config['seeds'][0])]
    require(int(cohort.patients.sum()) == 512 and int(cohort.events.sum()) == 65, 'DSS cohort changed')
    boot = gates['BOOTSTRAP_AUDIT.json']
    body = '# Phase19 — Clinical-conditioned Low-rank Patch–Pathway Attention\n\n'
    body += f"## 判定与证据范围\n\n**{decision['status']}**。机制子矩阵判定为 `{decision['mechanism_only_status']}`；完整方法预算与机制预算分别披露，不能把技术PASS误当优化充分。\n\n"
    body += f"父提交 `{config['parent_commit']}`，训练来源 `{identity}`。本轮完成 **150 inner + 150 fresh outer = 300 次正式神经网络拟合**；另有{counts['smoke_real_short_fits_with_generator_steps']}次实际发生generator更新的短拟合、20个唯一凸控制拟合。旧125次内层Attention pilot没有新OOF，不计本轮300次，也不与完整外折表混排。\n\n"
    body += '## DSS：完整患者级外折结果\n\n' + table(display.loc[order].reset_index())
    body += '\n512患者、65 DSS events；每seed先加总五折同折可比较对分子/分母，再平均3个seed的C值；不是风险ensemble，也不是将15 cells叫seed mean。seed SD/范围描述算法随机性，3 seeds不是三个独立患者队列。历史Clinical+S32+CONCH32的视觉表征和训练协议不同，只是强历史参照。\n\n'
    body += '## 六个预锁定主要比较\n\n' + table(primary, ['comparison','metric','delta','CI95_low','CI95_high','Bonferroni_low','Bonferroni_high','positive_seeds','positive_fold_cells','fold_cells'])
    body += f"\n实际有效bootstrap {boot['valid']}/{boot['draws']}，零分母排除{boot['zero_denominator_draws']}。20,000次固定外折内患者重采样，所有模型及seed共享计数，pair权重count_i×count_j；Uno60使用对应outer-train反向KM，tau=60。12项家族校正区间覆盖率99.5833%，普通95%区间同列。点估计与每次bootstrap都计算同一个mean-seed-C配对统计量；不先平均风险。区间条件于已拟合模型，不包括完整重训练和反复研究者选择方差。\n\n"
    body += '### 逐seed方向\n\n' + table(per_seed[per_seed.family == 'primary'], ['comparison','metric','seed','delta'])
    body += '\n### 工程推进门\n\n' + table(pd.DataFrame(decision['primary_checks']))
    body += '\n' + table(pd.DataFrame(decision['strong_reference_checks']))
    body += f"\n机制预算充分={decision['mechanism_budget_sufficient']}；全部方法预算充分={decision['all_method_budgets_sufficient']}；六比较×两指标方向/seed门={decision['all_primary_direction_and_seed_gate']}；J0/J_CS无均值损害门={decision['strong_reference_no_mean_harm']}。这些是预锁定工程门，不是生物学机制或独立效能确认。校正后下限>0的主要统计量：{decision['corrected_positive_primary_statistics']}。未检出不等于无信息或统计等效。\n\n"
    body += '## 强简单控制、官方核心适配与冻结干预\n\n' + table(secondary, ['comparison','metric','delta','CI95_low','CI95_high','Bonferroni_low','Bonferroni_high','family_tests'])
    body += '\nJ0为共同全输入线性Cox，J_CS为同源Clinical+S32；均固定ridge10，并不预称J0为最强基线。神经模型的共同线性主效应从J0初始化、联合优化，不以训练内风险再学习自由stacking系数。\n\n'
    body += 'A4/A5是分别用错误Clinical/Omics交互条件真实重新训练的模型；A3__WRONG_C/WRONG_O是冻结A3只换交互输入的推理干预。两者不能互称重训消融。本人Clinical/RNA/CNA主效应保持匹配。Clinical错配仅按癌种分区内置换，不按Stage/TNM/Age/Sex保留患者严重度；Omics错配沿用癌种/Stage规则，回退与平台/缺失差异见数据审计，不能把错配优势当因果语义证明。\n\n'
    body += 'MGCT、CAMR_REPAIRED、MMP已运行固定官方核心，但这是共同特征/Cox/Clinical适配，非原生论文训练协议严格复现。CAMR明确保留辅助目标及独立判别器，MMP训练分区拟合原型；不能宣称它们与Attention完全同算量或同优化任务。源码/许可/必要修复见BASELINE_REGISTRY与NATIVE_SOURCE_AUDIT，资格测试只证明登记的实现边界。\n\n'
    body += '## 机制解释与表征边界\n\n'
    body += 'CP三元分数严格等价于Clinical调制的低秩双线性兼容度，以及增广Q/K；也可视作潜在维度上的Clinical-FiLM。等价实现的输出/梯度parity已测试，但它不是独立算法控制，也不是首次三元乘积或FiLM创新。既有工作已经存在Clinical前置交互，不能概括为此前Clinical都仅末端拼接。\n\n'
    body += 'A1是两两logit相加后softmax，其非线性本身可产生三模态依赖；A6才是PO、PC、OC独立标量风险线性相加的严格两两函数。A2保留增广常数1与可学习全局rank4项，因此固定Clinical均值不会把整个CP旁路置零。A3非截距Clinical系数初始0，但其真实参数梯度可学习；不能将零初始化响应误当实现断梯度。\n\n'
    body += '24个Token分为RNA/CNA gain/CNA loss各8个；每个来自通路内一个可学习加权标量、覆盖率及slot embedding，并非完整512维RNA的无损32维通路表征。原始RNA512保留于共同线性主效应，只证明信息旁路可用，不能证明RNA→交互Token保留了全部预后排序。负结果不能推广为未压缩RNA或全部患者特异交互无效。\n\n'
    body += 'J0加入RNA/CNA各8个固定测量覆盖率主效应，降低把缺失模式增益误判为交互的风险；Token仍有逐基因测量mask，J0不是与其完全等表达的模型。平台来源/大块RNA缺失仍可能混杂，coverage控制不等于已识别并消除真实测序平台影响。UNI2-h只做冻结物理足迹/坐标审计，无原图形态真值，不能将attention分布当组织学标签。\n\n'
    body += '### 前32名训练患者的冻结、无标签探针\n\n' + table(mech)
    body += '\n每个checkpoint取固定顺序前32名训练患者，inner/outer分列；患者内WSI等权风险，与WSI级attention统计不混称独立病例。不同seed/fold会重复同一患者，表格是checkpoint中位数，不是独立队列均值。更尖锐attention、更大真/错响应或非零梯度只能证明算子敏感，不自动证明可泛化预测增益。\n\n'
    coverage_rows = pd.DataFrame(gates['PROBE_COVERAGE_AUDIT.json']['rows'])
    coverage_columns = ['arm','role','checkpoint_count','isolated_input_gradient_available',
        'risk_attention_probe_unique_query_patients','risk_attention_probe_unique_query_WSIs',
        'isolated_input_gradient_probe_unique_query_patients','isolated_input_gradient_probe_unique_query_WSIs']
    body += '**隔离输入梯度仅使用fit[:1]的单个训练患者风险**；风险SD/attention/entropy使用fit[:32]。同fold不同seed首患者会重复，15 checkpoints不能当15名独立梯度探针患者。下表由检查点和owner元数据独立去重，无额外forward/gradient/优化；wrong donor身份不冒充query患者。\n\n' + table(coverage_rows, coverage_columns)
    body += '\ntrilinear_scores_SD也可能来自某臂仅为诊断而计算的未使用CP值；实际进入attention的差分尺度看actual_score_correction_SD。A0/A6的CP分数不进入PO attention，A1实际修正是两两logit和；不能按未使用CP诊断宣称这些臂也训练了完整三元机制。原生适配baseline未实现同义的隔离输入梯度时保留NA，不冒充零梯度证据。\n\n'
    probe_a3 = mech[(mech.role == 'outer') & (mech.arm == 'A3')].iloc[0]
    coverage_a3 = coverage_rows[(coverage_rows.role == 'outer') & (coverage_rows.arm == 'A3')].iloc[0]
    pair_lookup = paired.set_index(['comparison','metric'])
    wc_h = pair_lookup.loc[('A3-A3__WRONG_C','Harrell')]
    wo_h = pair_lookup.loc[('A3-A3__WRONG_O','Harrell')]
    wo_u = pair_lookup.loc[('A3-A3__WRONG_O','Uno60')]
    off_h = pair_lookup.loc[('A3-A3__INTERACTION_OFF','Harrell')]
    off_u = pair_lookup.loc[('A3-A3__INTERACTION_OFF','Uno60')]
    a4_h = pair_lookup.loc[('A3-A4','Harrell')]
    body += '### 实际失败发生在哪一层\n\n'
    body += f"1. **已观察到条件分支连通和兼容度变化。** outer模型训练探针中，A3实际CP修正SD的checkpoint中位数为{probe_a3.actual_score_correction_SD:.6g}，真/错Clinical的attention相对L2为{probe_a3.clinical_attention_relative_L2:.6g}。这些不是断路或全零旁路；但真/错Clinical及Omics的患者风险MAE只有{probe_a3.matched_wrongC_risk_MAE:.6g}/{probe_a3.matched_wrongO_risk_MAE:.6g}，相应交互/主效应风险SD为{probe_a3.interaction_risk_SD:.6g}/{probe_a3.main_risk_SD:.6g}。这是固定训练探针统计，跨15 checkpoints实际仅{int(coverage_a3.risk_attention_probe_unique_query_patients)}名去重query患者；隔离输入梯度仅{int(coverage_a3.isolated_input_gradient_probe_unique_query_patients)}名不同患者，不是全队列独立敏感性证据。\n\n"
    body += f"2. **条件响应未转成可靠held-out配对增量。** 冻结A3错Omics后，A3−wrongO的Harrell/Uno60为{wo_h.delta:+.6g}/{wo_u.delta:+.6g}，排序几乎不变。冻结错Clinical的Harrell名义95%区间[{wc_h.CI95_low:.6g},{wc_h.CI95_high:.6g}]为正，但次要20项校正区间[{wc_h.Bonferroni_low:.6g},{wc_h.Bonferroni_high:.6g}]跨0；真实错Clinical重训对照A4的主要Harrell差仅{a4_h.delta:+.6g}、95%区间[{a4_h.CI95_low:.6g},{a4_h.CI95_high:.6g}]跨0。不能把单个冻结干预的名义区间当患者特异机制已确认。\n\n"
    body += f"3. **完整交互未超过保留主效应的简单控制。** 冻结关闭A3自身交互后Harrell为{display.loc['A3__INTERACTION_OFF','Harrell']:.6g}，A3−off的Harrell/Uno60为{off_h.delta:+.6g}/{off_u.delta:+.6g}；Harrell95%区间[{off_h.CI95_low:.6g},{off_h.CI95_high:.6g}]仍跨0，不能宣称交互显著有害。J0的Harrell为{display.loc['J0','Harrell']:.6g}，点估计仍高于该冻结off版本。\n\n"
    body += '本轮支持的有限回答是：实现确实产生了Clinical条件变化，但该RNA/CNA联合配对路径没有形成可靠的held-out条件预后增量。通路标量压缩、joint优化中的主效应变化、事件数量及条件信息重叠仍是未被分别隔离的解释；当前证据不能指定唯一失败原因，也不授权据此盲目增加模块。\n\n'
    mmp = gates['MMP_TOKEN_DIAGNOSTICS.json']
    observations = mmp['observed_current_snapshot']
    body += '### 冻结MMP分布Token的无标签数值审计\n\n' + table([
        {'statistic':key,'range_or_value':value} for key,value in observations.items() if key != 'interpretation'])
    body += f"\n最终覆盖10/10同源role/fold缓存，NaN/Inf计数均0；这是snapshot {mmp['snapshot_count']}，先前较少缓存的可用性快照保留在JSON中，不是多次模型训练。概率熵以log(16)归一，uniform参考为熵1、最大分配概率0.0625；应依据上表实际概率和原型均值差异判断，不能仅因native prior .1相对单位范数UNI2逐维方差较大，就宣称分配均匀或整体collapse。\n\n"
    body += f"活跃患者combined Token的有效秩范围为{mmp['active_combined_token_effective_rank_range']}；均值/协方差各自有效秩及原型余弦见上表。协方差向量近parallel或原始拼接方差由其尺度主导，是当前固定UNI2适配的限制；不等于均值/概率也无差异，更不是native MMP无效的证明。审计未读终点/OOF/风险，不改eps、模型、source或训练参数，也不据此追加调参。\n\n"
    body += '## 训练预算与梯度\n\n' + table(pd.DataFrame(budget['rows']))
    body += '\n固定max160、minimum8、patience40、EMA0.2、dropout0.1、AdamW lr0.001/WD0.01、主效应L2=10；没有依据外折成绩追加epoch/rank/scale搜索。触顶不超过20%只是预锁定预算门；零触顶不等于全局收敛。fresh outer逐单元使用本臂同seed/fold内层选择的准确epoch。\n\n'
    body += f"预外折预算检查原状态为 `{preouter_budget['status']}`；BUDGET_AUDIT.json原件保留，独立重算写BUDGET_VERIFICATION.json，未用事后覆盖掩盖原预算证据。\n\n"
    body += '所选inner epoch、训练目标反传后保留的梯度组中位数（含主效应L2；CAMR另见下述辅助目标及D步边界）：\n\n' + table(selected_grad)
    body += '\n所选checkpoint的隔离输入梯度、训练损失梯度与不同训练长度下梯度是不同对象；不能仅据梯度较大归因唯一的竞争/机制原因。完整逐epoch loss、EMA、Harrell和梯度保存在TRAINING_CURVES.csv。\n\n'
    body += '## 参数与计算\n\n' + table(resource)
    body += '\nregistered_parameters为注册数；last_step_gradient_parameters为最后一次汇总时具有梯度tensor的参数数量，即使数值为0也会计入，不能直接称全部有效预测容量。CAMR独立D步之后保留D.grad，该计数及native_gradient混合了本epoch遗留D步与G步梯度，不能解释为单一Efron/G目标的活跃参数或与Attention作同目标梯度大小排名。实际G optimizer明确排除D参数，资格测试核验G/D更新隔离，因此这一诊断口径不改变风险或参数更新。A0–A6没有判别器，不受此项混合影响；共同main梯度也包含显式L2，CAMR另有辅助损失。各臂可有注册但未用路径；判别器参数及优化步骤另列。并行cell seconds之和不是墙钟时间或隔离速度排名，微批只管理显存，所有患者风险组成一次完整Efron风险集再反传，checkpoint保留dropout RNG。\n\n'
    body += '### 整卡采样\n\n' + table(gpu)
    body += '\n均值按相邻采样时段左值保持加权，仅覆盖记录到的正式inner/outer区间；最大值与中位数为采样统计，不是真实瞬时峰值或完整端到端负载。整卡可能包含并发进程，不能用显存占用率证明模型质量。\n\n'
    body += '运行时private WORKERS可调整并发；上表记录实际worker上限/在跑数，不能把源码默认值当完整运行事实，也不把忙时99%称为始终99%。模型/优化/随机种子/数据及训练source未因调度改变。\n\n'
    body += '### 实际软件与硬件环境\n\n' + table([
        {'field':'Python','value':environment['python_version']},
        {'field':'PyTorch','value':environment['torch_version']},
        {'field':'compiled CUDA','value':environment['torch_compiled_cuda_version']},
        *[{'field':name,'value':value} for name,value in environment['packages'].items()],
        *[{'field':'GPU','value':gpu_row} for gpu_row in environment['GPU']]])
    body += '\nENVIRONMENT.json只采集版本/GPU等白名单，训练固定deterministic=True、TF32=False、CUBLAS_WORKSPACE_CONFIG=:4096:8；另列报告进程当前开关，避免把独立报告进程默认值误称训练实际设置。未导出环境变量集合或凭据。\n\n'
    flat_counts = {k:v for k,v in counts.items() if not isinstance(v, (list,dict))}
    body += '## 实际执行次数\n\n' + table([{'item':k,'value':v} for k,v in flat_counts.items()])
    body += '\n300个正式神经模型与诊断短拟合分计；20个唯一ridge fit跨seed引用，不乘成60次拟合。MMP十个当前source的训练分区原型缓存对应十个已完成KMeans producer，n_init为内部初始化次数，三个seed及冻结回放复用；每WSI PANTHER EM不另冒充跨患者聚类模型。原型生产没有尝试级账本，失败/中断聚类尝试记为NOT_RECORDED，不能从十个完成缓存推断总尝试数或失败数为0。诊断短拟合按实际optimizer step增量账本计数，包括失败但已更新的尝试，不保存其checkpoint或患者风险。完整患者OOF/正式checkpoint/PCA/原型权重仅在私有目录保留。\n\n'
    body += f"## PFI：冻结DSS风险迁移\n\n{int(single.patients.sum())}患者、{int(single.events.sum())} events；只迁移DSS风险，无PFI训练/选配置，不用次要终点改写DSS判断。PFI来自同一研究队列，不是独立外部验证。\n\n" + table(pfi.loc[order].reset_index())
    body += '\n## 实现资格、优势与风险\n\n' + table([{'gate':name,'status':value['status']} for name,value in gates.items()])
    body += f"\n全部300 checkpoints的五种模式回放最大绝对误差={gates['CHECKPOINT_REPLAY.json']['maximum_error']:.6g}，独立主效应重算最大误差={gates['CHECKPOINT_REPLAY.json']['own_main_off_maximum_error']:.6g}。独立指标逐折与scikit-survival一致，配对区间同统计对象，外折标签突变在重建Data及完整内层短拟合期间保持隔离。技术资格不代替预后/生物机制效能。\n\n"
    body += '优势：保留Clinical及原始RNA主效应；分离RNA/CNA证据与缺失；明确非零固定条件、严格两两及错配重训控制；患者级完整风险集、fresh outer、独立指标及配对统计可审计。\n\n'
    body += '风险：65个DSS事件、反复开发的单队列、通路标量压缩、缺失/平台混杂、条件增量估计不稳定、预算门不证明全局最优，以及不同方法辅助任务/计算差异；本次10臂触顶均0，不将潜在截断写成本次已发生。无组织学原图核验和未触碰外部队列。算子/数值/执行计数置信度高，唯一生物学因果解释与外部效能置信度低。\n\n'
    body += f"**最终执行判断：{decision['status']}。** 完成本轮即停止自动扩展，不追加深度/rank/归一化/endpoint搜索；未证明一般Clinical前置或三模态互补不可能，也不把CP公式本身当新理论。\n\n"
    body += '**拓展思考／你可能想知道：** Clinical改变兼容度、兼容度改变预测、预测增加Clinical+病理条件下的泛化排序，是三道不同证据门；任何一道单独通过都不能替代其余两道。\n'
    text(R/'RESULTS.md', body)
    text(R/'DECISION.md', '# Phase19 决策\n\n**' + decision['status'] + '**\n\n' +
         '机制子矩阵：`' + mechanism_status_label(decision) + '`。\n\n' +
         table(pd.DataFrame(decision['primary_checks'])) + '\n' +
         table(pd.DataFrame(decision['strong_reference_checks'])) + '\n' +
         '预算技术检查与优化充分性分开；方向工程门不等于统计确认，Bonferroni12区间仍仅适用于已拟合模型和本反复研究队列。CP等价Clinical-FiLM/增广QK，非首创主张。完整数字及边界见RESULTS.md；不自动扩大模型。\n')
    attempts = read(P/'SMOKE_LEDGER.json')['attempts']
    failed_task = read(P/'FAILED.json') if (P/'FAILED.json').exists() else None
    failure_rows = [{key:row.get(key, '') for key in ['id','arm','source','status','generator_steps','discriminator_steps','error_type']}
                    for row in attempts]
    failure_body = '# Phase19 执行与修复记录\n\n## 实际短拟合账本\n\n' + table(failure_rows)
    failure_body += '\n真实generator/discriminator更新逐step记录；RETURNED_SUCCESSFULLY才表示完整返回，RUNNING或FAILED不能由此推断所有工作已完成。失败但发生更新的记录仍计真实短拟合，不从成功统计中删除。无短拟合checkpoint或患者风险公开/持久化。\n\n'
    failure_body += '## 正式队列\n\n300份完整checkpoint和hash receipt、同源inner epoch选择及五条件回放均已独立核验。'
    if failed_task is None:
        failure_body += '当前私有目录未发现FAILED.json调度失败凭证；这描述记录状态，不以它证明不存在任何进程外部故障。\n\n'
    else:
        failure_body += '\n\n已存在正式调度失败记录：\n\n' + table([{k:failed_task.get(k,'') for k in ['source','role','cell','exit']}])
        failure_body += '\n该凭证不包含完整部分训练step数，300仅为完成拟合数，不代表全部尝试数或其总成本；不得将未跟踪的部分运行隐写为0。\n\n'
    failure_body += '## 已固定的实现资格\n\nsimple cache的嵌套NumPy审计先规范为JSON值再比较，并有SHA receipt；非重入checkpoint保留dropout RNG，逐微批native rows/WSI索引闭包显式绑定；外折标签poisoning在新Data构造及短拟合期间持续生效。上述实现已在最终训练source锁之前通过资格测试，正式300模型使用同一锁定source，不覆盖旧实验。\n\n'
    failure_body += 'MMP没有KMeans尝试级账本，只能核验已完成缓存producer；失败/中断尝试数量NOT_RECORDED。所有优化、模型引用、凸求解、聚类缓存、冻结干预分别计数。\n'
    repair = mmp['prepublication_auxiliary_diagnostic_repair']
    failure_body += '\n## MMP只读辅助分析helper修复\n\n' + table([repair])
    failure_body += '\n首次Gram helper对float64患者数组view原位中心化，导致后续概率量出现NaN，严格JSON拒绝输出；改为copy后从SHA核验的磁盘缓存重算全部诊断。磁盘缓存、优化器、训练source与模型均未改变；属于辅助分析失败/修复，不计神经训练失败或新拟合。\n'
    text(R/'FAILURE_LOG.md', failure_body)
    text(R/'REPRODUCE.md', f'''# Phase19 复现与数据边界

训练来源：`{identity}`。父提交：`{config['parent_commit']}`。正式模型来自本轮fresh fit，不复用旧神经检查点。

## 前置条件

使用本项目既有、合法授权的私有512患者DSS/PFI矩阵、固定患者折、Phase12训练分区预处理缓存、Phase17物理足迹UNI2-h视觉缓存及经验证的历史LOCAL参照。患者数据、真实标识、PCA/原型状态、OOF和权重不包含在公开仓库，不能只下载代码就假定数据齐全。原图形态核验仍缺失。

环境为既有GPU环境及scikit-survival评测依赖；官方核心按`configs/phase19_baseline_sources.json`固定commit/文件内容校验。模型/输入/训练和适配边界见PROTOCOL.md与BASELINE_REGISTRY，不把本轮Cox适配称原生论文训练复现。

## 执行顺序

在仓库根目录设置`PYTHONPATH=.:/root/phase10_trimodal_eval_deps`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`及OMP/MKL/OPENBLAS线程数2。

```sh
python scripts/prepare_phase19.py
python scripts/schedule_phase19.py
python scripts/validate_phase19.py --replay --shard 0 --shards 4
python scripts/validate_phase19.py --replay --shard 1 --shards 4
python scripts/validate_phase19.py --replay --shard 2 --shards 4
python scripts/validate_phase19.py --replay --shard 3 --shards 4
python scripts/validate_phase19.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/diagnose_phase19.py
python scripts/report_phase19.py
```

prepare检查固定官方源、资格测试、10个role/fold预处理、20个唯一凸控制、真实短拟合及外折标签隔离；正式训练前锁最终source。scheduler依次执行150 inner与150 fresh outer，各outer精确使用对应inner所选epoch。不以PFI选择模型。本次使用上述四个互斥回放分片，每片75份检查点，共300份；可顺序执行，也可并行，但必须收齐四片再独立评估。不要混用分片总数，也不要混入其他分片方案留下的文件；collect会汇总全部CHECKPOINT_REPLAY_*.csv，重复/缺失/源hash不一致均拒绝完成。公开报告保留本次0/1/2/3四份，因此原目录复现须按相同四片完整覆盖，不能只把0改跑为shards=1后留下旧1/2/3。

缓存只在source、fit/heldout、特征、官方core与hash receipt一致时复用；不得通过删除receipt或覆盖旧结果绕过检查。GPU worker数只调度并发，不改变参数、样本、风险集或选择协议。完整患者风险形成一次Efron loss，微批不是微批Cox。

## 结果核对

逐折表先sum分子/分母生成真正3-seed指标；六主比较×两指标为Bonferroni12家族。bootstrap为固定折内共享患者计数，20,000 draws；只反映固定已训练模型的不确定性。PFI为同队列冻结风险迁移。报告程序仅在完整训练、回放、数据/指标/配对区间检查通过后生成RESULTS及decision，不生成缺失OOF或预填成绩。

复现新增拟合与历史指标引用分开计数，smoke增量账本保留失败已更新尝试。MMP只有完成缓存producer证据，没有尝试级KMeans账本，因此不能声称知道全部失败尝试数。最终只发布匿名聚合报告/代码，不发布私有患者产物。

实际Python、torch/CUDA、NumPy/SciPy/pandas/scikit-learn/scikit-survival/einops、GPU驱动与显存见本轮ENVIRONMENT.json，不直接继承父实验环境版本。只读diagnose不访问生存标签或改变训练；重复运行增加快照并改变诊断文件hash，需要重新生成report/package以保持引用一致。
''')
    write(R/'decision.json', decision)
    write(R/'EXECUTION_COUNTS.json', counts)
    inputs = ['PROTOCOL_LOCK.json','INDEPENDENT_VALIDATION.json','BUDGET_AUDIT.json','BUDGET_VERIFICATION.json','CHECKPOINT_REPLAY.json',
              'BOOTSTRAP_AUDIT.json','CONTRAST_POINT_AUDIT.json','FIT_AND_REFERENCE_COUNTS.json',
              'SEED_METRICS.csv','FOLD_METRICS.csv','PAIRED_COMPARISONS.csv','COMPUTE_PROFILE.csv',
              'ATTENTION_MECHANISM.csv','TRAINING_CURVES.csv','CELL_MANIFEST.csv','PFI_SEED_METRICS.csv']
    inputs += ['PROBE_COVERAGE_AUDIT.json','MMP_TOKEN_DIAGNOSTICS.csv','MMP_TOKEN_DIAGNOSTICS.json','ENVIRONMENT.json']
    write(R/'REPORT_INPUT_AUDIT.json', {'status':'PASS','source':identity,
        'report_script_sha256':digest(Path(__file__)), 'input_hashes':{name:digest(R/name) for name in inputs},
        'private_ledgers_sha256':{name:digest(P/name) for name in ['TRAINING_DONE.json','SMOKE_LEDGER.json','SMOKE_COUNTS.json']},
        'no_patient_level_output':True, 'no_new_training_or_selection':True})
    print('PHASE19_REPORT_COMPLETE', decision['status'], counts['all_recorded_real_neural_fits'], flush=True)


def mechanism_status_label(decision):
    return decision['mechanism_only_status']


if __name__ == '__main__':
    report()

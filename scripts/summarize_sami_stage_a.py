"""Summarize completed, independently replayed tissue fits; never optimize."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    if path.exists():
        raise RuntimeError('Refusing to overwrite existing report: '+path.name)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def describe(values):
    return dict(min=float(min(values)), median=float(statistics.median(values)), max=float(max(values)))


def main(args):
    out = args.public_output
    result, audit = read(out/'RESULTS.json'), read(out/'INDEPENDENT_AUDIT.json')
    run = read(args.run_root/'run_manifest.json')
    if (audit['status'] != 'PASS' or audit['original_results_sha256'] != sha(out/'RESULTS.json')
            or audit['source_lineage'] != result['lineage'] or run['lineage'] != result['lineage']):
        raise RuntimeError('Independent audit and result lineage must match')
    receipt_paths = sorted(args.run_root.glob('cells/*/*/COMPLETE.json'))
    cells = [read(p) for p in receipt_paths]
    if len(cells) != 96 or list(args.run_root.glob('cells/*/*/FAILED.json')):
        raise RuntimeError('Expected exactly 96 complete fits and no failed cells')
    suites = list(ET.parse(args.junit).getroot().iter('testsuite'))
    test_counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
                   for k in ['tests', 'failures', 'errors', 'skipped']}
    if test_counts['tests'] != 71 or any(test_counts[k] for k in ['failures', 'errors', 'skipped']):
        raise RuntimeError('Full 71-test server qualification is required')
    start = min((p.parent/'START.json').stat().st_mtime for p in receipt_paths)
    end = max(p.stat().st_mtime for p in receipt_paths)
    execution = dict(schema_version='sami_stage_a_execution_v1', status='COMPLETE_INDEPENDENTLY_AUDITED',
        lineage=result['lineage'], formal_fit_count=96, inner_selection_fits=90, final_outer_fits=6,
        formal_retry_count=0, failed_cells=0, formal_device='CPU', threads_per_worker=2,
        schedule='outer_00_first_then_five_remaining_outer_folds_concurrently',
        gpu_used_for_formal_fits=False, python_version=platform.python_version(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        full_server_synthetic_tests=test_counts,
        junit_sha256=sha(args.junit),
        fit_elapsed_seconds=describe([c['elapsed_seconds'] for c in cells]),
        total_cell_fit_elapsed_seconds=sum(c['elapsed_seconds'] for c in cells),
        wall_span_first_start_to_last_cell_complete_seconds=end-start,
        wall_span_includes_launch_and_monitoring_gaps=True,
        optimizer_iterations=describe([c['fit_audit']['iterations'] for c in cells]),
        final_gradient_inf=describe([c['fit_audit']['gradient_inf'] for c in cells]),
        accepted_gradient_inf=1e-5,
        regularization_candidates=[.01, .1, 1.], selected_C=[f['selected_C'] for f in result['outer_folds']],
        allocated_parameters=4611, regularized_parameters=4608,
        original_results_sha256=sha(out/'RESULTS.json'), independent_audit_sha256=sha(out/'INDEPENDENT_AUDIT.json'),
        summarizer_sha256=sha(__file__), DSS_accessed=False,
        old_ST_test_accessed=False, raw_arrays_and_models_not_published=True)
    write(out/'EXECUTION_AUDIT.json', execution)
    sections = []
    with np.load(args.run_root/'oof_predictions.npz', allow_pickle=False) as z:
        y, donor, section, prob = [z[k] for k in ['y', 'donor', 'section', 'probabilities']]
    for oi, d in enumerate(np.unique(donor)):
        for si, s in enumerate(np.unique(section[donor == d])):
            mask = (donor == d) & (section == s)
            target, pred = y[mask], prob[mask].argmax(1)
            cm = np.zeros((3, 3), dtype=int)
            np.add.at(cm, (target, pred), 1)
            counts = cm.sum(1)
            recalls = [float(cm[k,k]/counts[k]) if counts[k] else None for k in range(3)]
            sections.append(dict(outer_fold=oi, section_within_fold=si, n_spots=int(mask.sum()),
                class_counts=counts.tolist(), class_recall=recalls,
                accuracy=float(np.mean(target == pred)), confusion_counts=cm.tolist()))
    if len(sections) != 12:
        raise RuntimeError('Expected 12 anonymous sections')
    write(out/'ANONYMOUS_SECTION_METRICS.json', dict(schema_version='sami_stage_a_sections_v1',
        class_order=result['class_order'], section_count=12, sections=sections,
        missing_class_recall=None, identity_keys_published=False,
        note='Ordinal groups only. Within-section spot summaries, not independent confidence intervals.'))

    model = result['donor_equal_aggregate']['metrics']
    prior = result['donor_equal_aggregate']['prior_baseline']
    ci = audit['descriptive_donor_bootstrap']['estimates']['balanced_accuracy']
    ba = [f['metrics']['balanced_accuracy'] for f in result['outer_folds']]
    decision = dict(schema_version='sami_stage_a_decision_v1',
        status='COMPLETE_LIMITED_TISSUE_DECODABILITY_EVIDENCE',
        numerical_and_leakage_audit='PASS', all_96_fits_complete=True,
        tissue_decodability_in_current_development_cohort='SUPPORTED_WITH_DONOR_HETEROGENEITY',
        balanced_accuracy=model['balanced_accuracy'], fixed_OOF_donor_bootstrap=ci['metrics'],
        balanced_accuracy_minus_train_prior=ci['model_minus_prior_baseline'],
        source_assignment_or_cell_fraction_validation='NOT_TESTED',
        all_tissue_composition_estimation='NOT_QUALIFIED_MISSING_OTHER_AND_UNKNOWN_CLASSES',
        TCGA_domain_transfer='NOT_TESTED', DSS_increment='NOT_TESTED',
        automatic_downstream_unlock=False, expand_C_after_outer_results=False,
        constraints=['six_development_donors_one_study', 'foundation_pretraining_overlap_unknown',
            'all_selected_C_at_locked_grid_upper_boundary', 'class_balanced_uncalibrated_probabilities',
            '4381_of_17392_source_spots_excluded', 'mixed_class_is_weaker',
            'fixed_OOF_intervals_omit_pipeline_refit_uncertainty'],
        next_required_contract='Independent source supervision and domain eligibility; no direct TCGA forced classification or DSS claim.',
        original_results_sha256=sha(out/'RESULTS.json'), independent_audit_sha256=sha(out/'INDEPENDENT_AUDIT.json'))
    write(out/'decision.json', decision)
    lines = [
        '# SAMI-Surv Stage A 实验结果', '',
        '**96/96 次正式拟合完成，独立回放审计通过。当前证据支持三类组织信息在该开发集合内可跨供体解码，存在明显供体差异。**', '',
        '本轮不含 RNA、TCGA、DSS、细胞来源模型或跨模态训练。原始报告 `RESULTS.json` 保持原始汇总状态，最终审计见 `INDEPENDENT_AUDIT.json`；两者由 SHA 绑定。', '',
        '## 数据与协议', '',
        '6位供体、12张切片；原始17392个spot中13011个进入三类任务，4381个排除。输入为冻结UNI2-h的1536维FP32特征。每个外折内部再留一供体选择C，然后只在外层训练供体重拟合。90次内部选型＋6次外层拟合，没有补训或外层选参。', '',
        '连续切片按供体整体隔离；每次拟合独立训练标准化；C候选固定为0.01/0.1/1。主指标依次按供体、类别、类别内切片等权。13,011个spot不是13,011个独立泛化单位。', '',
        '## 主要结果', '',
        '| 指标 | 冻结UNI2-h＋多项Logistic | 训练供体先验参照 |',
        '|---|---:|---:|']
    for label, key in [('Balanced accuracy','balanced_accuracy'), ('Accuracy','accuracy'),
                       ('NLL（低优）','negative_log_likelihood'), ('Brier（低优）','brier_multiclass_sum'),
                       ('ECE（低优）','ece_10_equal_width_bins')]:
        lines.append(f'| {label} | {model[key]:.5f} | {prior[key]:.5f} |')
    l, u = ci['metrics']['percentile_95']
    dl, du = ci['model_minus_prior_baseline']['percentile_95']
    lines += ['', f'BA的固定OOF供体bootstrap描述区间：**[{l:.5f}, {u:.5f}]**；相对先验的配对差值为{model["balanced_accuracy"]-prior["balanced_accuracy"]:+.5f}，区间[{dl:+.5f}, {du:+.5f}]。20,000次抽样、seed7031001，比较共享抽样。该区间条件于已有模型，不含完整流程重训不确定性，也不是跨研究确认。', '',
        '| 匿名外折 | BA | 肿瘤召回 | 间质召回 | 混合类召回 | selected C |',
        '|---|---:|---:|---:|---:|---:|']
    for f in result['outer_folds']:
        r = f['metrics']['class_recall']
        lines.append(f'| {f["outer_fold"]} | {f["metrics"]["balanced_accuracy"]:.5f} | {r[0]:.5f} | {r[1]:.5f} | {r[2]:.5f} | {f["selected_C"]:g} |')
    lines += ['', f'供体BA范围{min(ba):.5f}–{max(ba):.5f}，样本标准差{np.std(ba,ddof=1):.5f}。三类加权召回为肿瘤{model["class_recall"][0]:.5f}、间质{model["class_recall"][1]:.5f}、混合{model["class_recall"][2]:.5f}。汇总混淆矩阵按spot计数，因此不能直接用其行比例替代这些供体／切片等权召回。12张切片的匿名统计另见 `ANONYMOUS_SECTION_METRICS.json`。', '',
        '## 优势、风险与可作出的结论', '',
        '- **优势：** 独立病理医师标签提供了真实的组织类别监督；严格供体留出、训练分区标准化与独立回放形成可核查的证据链。',
        '- **供体偏移：** 匿名外折1的NLL为1.36944，高于训练先验的1.13861，ECE为0.32511；平均改善不能解释为所有供体都校准良好。',
        '- **类别覆盖：** 排除了25.19%的原始spot，模型没有其他组织、淋巴组织或unknown类别。禁止直接强制分类所有TCGA patch，再将类别均值解释为完整组织组成。',
        '- **概率含义：** 类别平衡训练输出未经后校准，不能作为细胞比例、CAF活化、RNA来源贡献或医学真实来源概率。112µm图像视野与55µm表达捕获直径也不是同一区域。',
        '- **正则边界：** 六个外折均选C=1，这是预锁候选中的最弱正则边界；仅说明该候选集合的选择结果，本轮不依据外层表现扩网格，也不宣称全局最优。',
        '- **证据独立性：** 仅6位已用于历史开发的供体、同一研究；UNI2-h预训练是否重叠未知。不能升级成全新外部确认。',
        '- **置信度：** 执行及供体隔离审计为高；当前开发集合的组织可解码性为中；TCGA迁移、分子来源正确性及DSS互补增量尚无本轮证据。', '',
        '## 数值与复现', '',
        f'71项服务器合成测试通过。96次优化均成功，迭代次数范围{execution["optimizer_iterations"]["min"]:g}–{execution["optimizer_iterations"]["max"]:g}，最大最终梯度无穷范数{execution["final_gradient_inf"]["max"]:.3g}，低于预锁1e-5。独立审计逐cell复算训练标准化、嵌套选择、检查点预测、OOF和指标。', '',
        f'CPU每worker2线程，先运行外折0，再并行其余5折；单cell中位用时{execution["fit_elapsed_seconds"]["median"]:.2f}秒。未使用GPU、未改变正式训练协议。Python {platform.python_version()} / NumPy {np.__version__} / SciPy {scipy.__version__}。', '',
        '复现入口：`prepare_sami_stage_a.py` → `run_sami_stage_a.py --mode plan` → `--mode run --outer-fold 0..5` → `--mode finalize` → `audit_sami_stage_a.py` → `summarize_sami_stage_a.py`。先单独完成plan，再启动并行。精确参数见配置，私有资产路径由CLI传入；不公开OOF、权重、坐标、barcode或供体标识。', '',
        '独立审计前修正了审计器的ECE浮点bin边界，使其与已冻结实现的np.linspace边界一致，并增强元数据与梯度阈值验证；未修改训练源码、配置、模型或正式RESULTS。回归测试包含0.6及其nextafter边界反例。', '',
        '## 下一阶段边界', '',
        '**不自动解锁完整SAMI或生存训练。** 下一步需要独立锁定来源相关分子程序、可验证的来源监督目标、组织组成参照、错供体和空间打乱控制。现有三类组织可解码性仅为这项研究提供受限的形态起点；缺少来源真值时不能将预测分配命名为真实细胞来源。', '',
        '拓展思考：首先区分“混合组织难判”与“跨供体域偏移”，再决定需要什么新增独立标注。',
        '你可能想知道：本轮没有需要人工审批的新下载；已有数据和模型都保留在私有位置，匿名交付不包含逐spot信息。', '']
    target = out/'RESULTS.md'
    if target.exists():
        raise RuntimeError('Refusing to overwrite RESULTS.md')
    target.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status='SUMMARY_COMPLETE', formal_fits=96, balanced_accuracy=model['balanced_accuracy'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--public-output', type=Path, required=True)
    parser.add_argument('--junit', type=Path, required=True)
    main(parser.parse_args())

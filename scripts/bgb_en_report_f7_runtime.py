"""Anonymous F7 runtime recovery report: JSON/receipt/hash reads only.

No torch import, tensor loading, forward, backward or optimizer is performed.
An old successful final80 diagnostic is not a replay of an unpersisted failed
training state. Backend recovery and candidate causal explanation stay distinct.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_recovery_audit import verify_recovery_receipt, verify_runtime_receipt, sha256, require
from scripts.bgb_en_f7_runtime_repair import validate_forward_qualification


def minimum(values):
    values = [float(value) for value in values if value is not None]
    return min(values) if values else None


def maximum(values):
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def diagnostic_summary(diagnostic, *, include_comparisons=False):
    """Whitelist numerical fields; never export input rows or private paths."""
    require(diagnostic['status'] in ('PASS_SAMPLED_READONLY_FORWARD', 'FAIL_SAMPLED_READONLY_FORWARD',
                                    'FAILED_DIAGNOSTIC_BACKEND_QUALIFICATION'), 'UNKNOWN_DIAGNOSTIC_STATUS')
    comparisons, boundaries = [], {'no_grad': [], 'grad': []}
    for batch in diagnostic.get('batches', []):
        for mode in boundaries:
            value = batch['boundary_'+mode]
            boundaries[mode].append({key: value[key] for key in (
                'rows_with_more_than_topk_keys', 'minimum_margin', 'mean_margin', 'exact_boundary_ties')})
        for mode, records in batch['comparisons'].items():
            for record in records:
                by_name = {value['name']: value for value in record['intermediate']}
                first = next((value['name'] for value in record['intermediate'] if not value['bitwise_equal']), None)
                require(first is None or re.fullmatch(r'(path\.project|omics\.rna|omics|interaction\.(patch_project|omics_project|path_match|omics_match|path_value|omics_value)):\d+|risk|delta|competition_score|selected_attention|selected_message', first),
                        'UNKNOWN_DIAGNOSTIC_MODULE_NAME')
                require(first == record['first_non_bitwise_intermediate'], 'DIAGNOSTIC_FIRST_DIVERGENCE_FIELD_DIFFERS')
                risk_error = float(by_name['risk']['max_abs_error'])
                require(record['risk_within_unchanged_gradient_cache_threshold'] == (risk_error <= 2e-5),
                        'DIAGNOSTIC_RISK_THRESHOLD_FIELD_DIFFERS')
                topk = record['topk']
                comparisons.append(dict(comparison_number=len(comparisons)+1, comparison_mode=mode,
                    first_non_bitwise_intermediate=first, risk_max_abs_error=risk_error,
                    score_max_abs_error=float(topk['score_max_abs_error']),
                    local_message_max_abs_error=float(by_name['selected_message']['max_abs_error']),
                    changed_support_query_rows=int(topk['changed_support_query_rows']),
                    symmetric_difference_selected_keys=int(topk['symmetric_difference_selected_keys']),
                    eligible_query_rows=int(topk['eligible_query_rows']),
                    attention_TV_mean=topk['attention_TV_mean'], attention_TV_max=topk['attention_TV_max'],
                    risk_within_unchanged_threshold=record['risk_within_unchanged_gradient_cache_threshold'],
                    boundary_no_grad={**boundaries['no_grad'][-1]}, boundary_grad={**boundaries['grad'][-1]}))
    n = len(comparisons)
    result = dict(status=diagnostic['status'], compared_batches=len(diagnostic.get('batches', [])), comparisons=n,
        all_compared_tensors_bitwise_equal=all(value['first_non_bitwise_intermediate'] is None for value in comparisons) if n else None,
        all_risks_within_original_threshold=all(value['risk_within_unchanged_threshold'] for value in comparisons) if n else None,
        maximum_risk_error=maximum([value['risk_max_abs_error'] for value in comparisons]),
        maximum_score_error=maximum([value['score_max_abs_error'] for value in comparisons]),
        maximum_local_message_error=maximum([value['local_message_max_abs_error'] for value in comparisons]),
        first_non_bitwise_counts=dict(Counter(value['first_non_bitwise_intermediate'] or 'BITWISE_IDENTICAL' for value in comparisons)),
        comparisons_with_support_flip=sum(value['changed_support_query_rows'] > 0 for value in comparisons),
        query_support_flip_observations=sum(value['changed_support_query_rows'] for value in comparisons),
        selected_key_symmetric_difference_observations=sum(value['symmetric_difference_selected_keys'] for value in comparisons),
        maximum_attention_TV=maximum([value['attention_TV_max'] for value in comparisons]),
        boundary_margins={},
        boundary_margin_unit='query-axis competition probability rank64/rank65 patch gap before patch normalization; not raw score gap',
        count_boundary='repeated-forward comparisons/query observations, not distinct patients or independent events')
    denominator = sum(value['eligible_query_rows'] for value in comparisons if value['attention_TV_mean'] is not None)
    result['query_row_weighted_attention_TV_mean'] = (sum(value['attention_TV_mean']*value['eligible_query_rows']
        for value in comparisons if value['attention_TV_mean'] is not None)/denominator) if denominator else None
    for mode, values in boundaries.items():
        count = sum(value['rows_with_more_than_topk_keys'] for value in values)
        result['boundary_margins'][mode] = dict(query_rows=count,
            minimum_margin=minimum([value['minimum_margin'] for value in values]),
            query_row_weighted_mean_margin=(sum(value['mean_margin']*value['rows_with_more_than_topk_keys']
                for value in values if value['mean_margin'] is not None)/count) if count else None,
            exact_boundary_ties=sum(value['exact_boundary_ties'] for value in values))
    if n and 'all_comparisons_within_threshold' in diagnostic:
        require(result['all_risks_within_original_threshold'] == diagnostic['all_comparisons_within_threshold']
                and result['all_compared_tensors_bitwise_equal'] == diagnostic['all_comparisons_bitwise_equal'],
                'DIAGNOSTIC_AGGREGATE_FIELDS_DIFFER')
    if include_comparisons:
        result['anonymous_comparison_rows'] = comparisons
    return result


def verify_diagnostic_origin(value, recovery, private, original_cells, *, root=ROOT):
    import hashlib
    require(value['arm'] == 'F7' and value['optimizer_updates'] == 0 and value['neural_fits'] == 0
            and value['backpropagations'] == 0 and value['threshold'] == 2e-5, 'DIAGNOSTIC_NOT_READONLY_OR_THRESHOLD_CHANGED')
    require(value['diagnostic_sources'] == {'scripts/bgb_en_f7_determinism_diagnostic.py':
            sha256(Path(root)/'scripts/bgb_en_f7_determinism_diagnostic.py')}, 'DIAGNOSTIC_SOURCE_SHA_DIFFERS')
    require(hashlib.sha256(json.dumps(value['core_lineage'], sort_keys=True).encode()).hexdigest()
            == recovery['core_lineage_sha256'], 'DIAGNOSTIC_CORE_LINEAGE_DIFFERS')
    name = f"R3_outer1_inner{value['partition']}_F7_seed{value['seed']}"
    original = next(row for row in original_cells if row['cell'] == name)
    require(value['original_receipt_sha256'] and all(original['files'].get(key) == digest
            for key, digest in value['original_receipt_sha256'].items()), 'DIAGNOSTIC_ORIGINAL_RECEIPT_SHA_DIFFERS')
    prepared = Path(private)/'prepared'/f"outer1_inner{value['partition']}.pt"
    require(value['prepared_sha256'] == sha256(prepared), 'DIAGNOSTIC_PREPARED_SHA_DIFFERS')


def build_report(work, private, recovery_path, *, config_path, root=ROOT):
    work, private = Path(work), Path(private)
    recovery = verify_recovery_receipt(recovery_path, private, config_path=config_path, root=root)
    origin = json.loads(Path(recovery['origin_manifest_path']).read_text())
    original = json.loads((work/'ORIGINAL_FORWARD.json').read_text())
    verify_diagnostic_origin(original, recovery, private, origin['original_cells'], root=root)
    require(original['partition'] == 0 and original['seed'] == 31 and original['state'] == 'final80'
            and original['backend']['mode'] == 'inherited' and str(original['device']).startswith('cuda'),
            'ORIGINAL_DIAGNOSTIC_FIXED_CHECKPOINT_DIFFERS')
    old_model = next(row for row in origin['original_cells'] if row['cell'] == 'R3_outer1_inner0_F7_seed31')
    if 'checkpoint_artifact_sha256' in original:
        require(original['checkpoint_artifact_sha256'] == old_model['files']['model.pt'], 'ORIGINAL_DIAGNOSTIC_MODEL_SHA_DIFFERS')
    quals, training, input_hashes = [], [], {'ORIGINAL_FORWARD.json': sha256(work/'ORIGINAL_FORWARD.json'),
        'RECOVERY_RECEIPT.json': sha256(recovery_path)}
    for part in (0, 1, 2):
        for seed in (31, 47):
            qual_path, runtime_path = work/f'CUDA_QUAL_p{part}_s{seed}.json', work/f'RUNTIME_TRAIN_p{part}_s{seed}.json'
            value = json.loads(qual_path.read_text())
            verify_diagnostic_origin(value, recovery, private, origin['original_cells'], root=root)
            validate_forward_qualification(value, part, seed, value['core_lineage'],
                sha256(Path(root)/'scripts/bgb_en_f7_determinism_diagnostic.py'))
            require(value['backend'] == recovery['runtime_backend'], 'QUALIFICATION_TRAIN_RUNTIME_DIFFERS')
            summary = diagnostic_summary(value)
            require(summary['comparisons'] > 0 and summary['all_risks_within_original_threshold'] is True,
                    'EMPTY_OR_FAILED_DETERMINISTIC_QUALIFICATION')
            quals.append(dict(inner_partition=part, seed=seed, **summary))
            cell = private/'cells'/f'R3_outer1_inner{part}_F7_seed{seed}'
            complete = json.loads((cell/'COMPLETE.json').read_text())
            runtime = json.loads(runtime_path.read_text())
            guard = verify_runtime_receipt(cell, complete, expected_backend=value['backend'], root=root)
            require(runtime == json.loads((cell/'RUNTIME_REPAIR_COMPLETE.json').read_text())
                    and runtime['qualification_sha256'] == sha256(qual_path), 'TRAINING_QUALIFICATION_RECEIPT_DIFFERS')
            curves = complete['curves']
            require([row['epoch'] for row in curves] == [20, 40, 80], 'RECOVERY_LOGGED_EPOCHS_DIFFER')
            points = [dict(epoch=row['epoch'], gradient_cache_replay_max=float(row['replay_max']),
                           within_original_2e_minus5_guard=float(row['replay_max']) <= 2e-5) for row in curves]
            require(all(row['within_original_2e_minus5_guard'] for row in points), 'SAVED_TRAINING_REPLAY_GUARD_FAILED')
            training.append(dict(inner_partition=part, seed=seed, actual_new_fits=1, optimizer_updates=80,
                teacher_refits=0, runtime_receipt_sha256=guard['runtime_receipt_sha256'], logged_guard_points=points,
                per_epoch_guard_enforced_by_unchanged_frozen_source=True,
                all_80_numeric_replay_max_values_persisted=False))
            input_hashes[qual_path.name] = sha256(qual_path); input_hashes[runtime_path.name] = sha256(runtime_path)
    queue_path = work/'QUEUE_RECEIPT.json'
    queue = json.loads(queue_path.read_text())
    require(queue['status'] == 'SIX_FITS_COMPLETE_PENDING_INDEPENDENT_CERTIFICATION'
            and queue['actual_new_fits'] == 6 and queue['actual_new_updates'] == 480 and queue['teacher_refits'] == 0
            and queue['original_schedule_modified'] is False, 'RECOVERY_QUEUE_NOT_COMPLETE')
    input_hashes[queue_path.name] = sha256(queue_path)
    source_names = ('scripts/bgb_en_report_f7_runtime.py', 'scripts/bgb_en_recovery_audit.py',
                    'scripts/bgb_en_f7_determinism_diagnostic.py', 'scripts/bgb_en_f7_runtime_repair.py',
                    'configs/bgb_en/F7_deterministic_runtime_v1.json')
    all_logged = [point for row in training for point in row['logged_guard_points']]
    return dict(schema='bgb_en_F7_runtime_repair_public_report_v1', status='COMPLETE_VERIFIED_RECOVERY_REPORT',
        reporter_forward_calls=0, reporter_backward_calls=0, reporter_optimizer_updates=0,
        reporter_tensor_loads=0, patient_rows_or_identifiers_exported=False, private_paths_exported=False,
        sources={name: sha256(Path(root)/name) for name in source_names}, input_sha256=input_hashes,
        core_lineage_sha256=recovery['core_lineage_sha256'],
        original_successful_final80_inherited_diagnostic=diagnostic_summary(original, include_comparisons=True),
        original_diagnostic_scope='fixed original successful inner0/seed31 final80 checkpoint; not an unpersisted failed-epoch state',
        deterministic_initial_CUDA_qualifications=quals,
        all_six_initial_qualifications_bitwise_equal=all(row['all_compared_tensors_bitwise_equal'] for row in quals),
        all_six_initial_qualifications_within_original_threshold=all(row['all_risks_within_original_threshold'] for row in quals),
        recovery=dict(certificate_independently_reverified=True, core_unique_cells=recovery['core_unique_cells'],
            original_successful_fits=3, original_failed_attempts=3, original_failed_optimizer_updates=None,
            actual_new_fits=6, actual_new_optimizer_updates=480, actual_new_teacher_refits=0,
            all_six_F7_replaced=True, old_and_new_models_mixed=False, original_partial_schedules_preserved=True,
            backend=recovery['runtime_backend']),
        training_guard=dict(original_threshold=2e-5, threshold_relaxed=False,
            source_enforced_epoch_checks=480, observed_logged_points=len(all_logged),
            maximum_observed_logged_replay_error=maximum([row['gradient_cache_replay_max'] for row in all_logged]),
            all_observed_logged_points_within_original_threshold=True,
            all_epoch_numerical_errors_available=False,
            boundary='completion under the frozen per-epoch rejection code supports all80 checks per fit; only epochs20/40/80 numerical maxima were persisted, not all480 maxima'),
        per_fit_training=training,
        causal_interpretation=dict(failed_training_state_replayed=False, exact_original_CUDA_cause_localized=False,
            candidate='small upstream floating-point differences amplified by a discontinuous competition-Top64 boundary',
            confidence='medium candidate, not a localized causal finding',
            boundary='successful final80 inherited replay can show numerical sensitivity; it cannot identify the unpersisted failed-epoch state or isolate index_add from every other kernel'),
        caveat='deterministic runtime qualification/recovery does not establish survival benefit, biological mechanism, or an architecture change')


def markdown(report):
    old = report['original_successful_final80_inherited_diagnostic']; guard = report['training_guard']
    def number(value):
        return 'NA' if value is None else f'{value:.8g}'
    lines = ['# F7 数值运行恢复报告', '', '**状态：恢复证书已独立复核；本报告器前向、反传、优化及tensor加载均为0。**', '',
        '原6个F7单元全部按文件SHA保留，其中3成功、3失败；旧失败更新数未知。六个F7在同一确定性后端、原seed下从头拟合，共480次更新，新增教师拟合0。原partial调度收据保持原状。', '',
        '## 原成功80轮模型的继承后端诊断', '',
        f"固定inner0/seed31成功检查点：{old['compared_batches']}个批次，{old['comparisons']}次比较；最大风险误差{number(old['maximum_risk_error'])}，最大score误差{number(old['maximum_score_error'])}；发生Top64支持改变的比较数{old['comparisons_with_support_flip']}，最大attention TV {number(old['maximum_attention_TV'])}。", '',
        '这不是失败时状态的重放。原错误发生时的epoch/权重/Adam状态未持久化，当前证据不能把原始失败的精确原因定位到某一个CUDA算子。', '',
        '|比较模式|首次非逐位一致位置|次数|', '|---|---|---:|']
    counts = Counter((row['comparison_mode'], row['first_non_bitwise_intermediate'] or 'BITWISE_IDENTICAL')
                     for row in old.get('anonymous_comparison_rows', []))
    for (mode, name), count in sorted(counts.items()):
        lines.append(f'|{mode}|{name}|{count}|')
    lines += ['', '逐比较风险/score误差、Top64支持翻转、TV及对应边界margin见JSON；只公开匿名比较编号，不含患者行号或私有路径。', '',
        '## 六个初始状态CUDA资格', '', '|inner partition|seed|比较数|所有比较逐位一致|风险均在2e-5内|最大风险误差|支持改变比较数|', '|---:|---:|---:|---|---|---:|---:|']
    for row in report['deterministic_initial_CUDA_qualifications']:
        lines.append(f"|{row['inner_partition']}|{row['seed']}|{row['comparisons']}|{row['all_compared_tensors_bitwise_equal']}|{row['all_risks_within_original_threshold']}|{number(row['maximum_risk_error'])}|{row['comparisons_with_support_flip']}|")
    lines += ['', '## 训练一致性门槛', '',
        f"六个80轮新拟合保留每epoch原2e-5拒绝检查，未放宽阈值。实际持久化的{guard['observed_logged_points']}个epoch20/40/80记录全部通过，已记录最大误差为{number(guard['maximum_observed_logged_replay_error'])}。完整480个epoch数值误差并未全部保存；不能把18个日志点的最大值称为全部训练步最大值。", '',
        '## 解释边界', '',
        '**可确认：** 六份初始CUDA资格及同后端恢复训练按原合同完成，原失败谱系保留。**尚不能确认：** 原未保存失败状态的精确数值根因。微小上游数值差经Top-k边界放大的解释仍是候选链条，不是已定位的因果结论。', '',
        '优势是恢复不改变架构、Top64、折、损失或预算，并对新旧文件及source SHA实核；限制是确定性后端可能改变计算成本，初始资格和最终成功模型都不能替代原失败状态。该恢复不证明性能或医学机制改善。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--recovery-receipt', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/bgb_en')
    args = parser.parse_args()
    report = build_report(args.work, args.private, args.recovery_receipt, config_path=args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'F7_RUNTIME_REPAIR_REPORT'
    path.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    path.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], reporter_forward_calls=0, reporter_optimizer_updates=0)))


if __name__ == '__main__':
    main()

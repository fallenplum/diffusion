"""Append a numerical-drift diagnosis; never promote the original failed audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {(2, 31), (2, 47), (1, 31)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def aggregate_documents(original, documents):
    require(original['passed_units'] == 184 and original['failed_units'] == 2
            and original['status'] == 'FAILED_SOME_AVAILABLE_CHECKPOINTS', 'Original 184/186 audit must remain failed')
    failures = [row for row in original['units'] if row['status'] != 'PASS']
    require({(row['inner_partition'], row['seed']) for row in failures} == {(2, 31), (2, 47)}
            and all(row['arm'] == 'O9' and row['failure_code'] == 'CHECKPOINT_FORWARD_RISK_MISMATCH' for row in failures),
            'Original audit failures differ')
    rows, signatures = [], {}
    for document in documents:
        mode = document['backend']['mode']
        require(document['status'] == 'COMPLETE_FIXED_FIVE_FORWARD_DIAGNOSTIC'
                and document['fixed_repeats_per_cell'] == 5 and document['batch_size'] == 8
                and document['optimizer_fits'] == document['optimizer_updates'] == document['backward_calls'] == 0
                and document['original_risk_tolerance'] == 2e-6 and document['threshold_relaxed'] is False
                and document['best_repeat_selected'] is False, 'Diagnostic scope differs')
        require({(cell['inner_partition'], cell['seed']) for cell in document['cells']} == EXPECTED,
                'Fixed three diagnostic cells differ')
        for cell in document['cells']:
            key = cell['inner_partition'], cell['seed']
            prior = next(row for row in original['units'] if row['stage'] == 'R2' and row['arm'] == 'O9'
                         and row['inner_partition'] == key[0] and row['seed'] == key[1])
            require(cell['artifact_sha256'] == prior['artifact_sha256'], 'Diagnostic checkpoint differs from the original audit')
            require(cell['gamma'] == .3 and cell['gamma_matches_frozen_config'] is True,
                    'Invalid diagnostic gamma: the earlier gamma=.1 version cannot be aggregated')
            require(cell['weights_bitwise_unchanged'] and cell['parameter_gradients_absent'], 'Diagnostic changed weights/gradients')
            signature = tuple(cell[name] for name in ('artifact_sha256', 'complete_sha256', 'prepared_sha256', 'state_dict_sha256'))
            require(key not in signatures or signatures[key] == signature, 'Backend diagnostics changed the fixed checkpoint')
            signatures[key] = signature
            require([value['repeat'] for value in cell['repeats']] == [1, 2, 3, 4, 5]
                    and all(value['status'] == 'FORWARD_COMPLETE' for value in cell['repeats'])
                    and len(cell['cross_repeat_comparisons']) == 10, 'Do not omit any fixed repeat')
            maximum = max(value['max_abs_error'] for value in cell['repeats'])
            unchanged = all(value['pair_credit_changes_vs_saved']['changed_pair_credits'] == 0
                and value['delta_Harrell_vs_saved'] == value['delta_Uno60_vs_saved'] == 0 for value in cell['repeats'])
            rows.append(dict(backend=mode, inner_partition=key[0], seed=key[1], role=cell['locked_role'],
                artifact_sha256=cell['artifact_sha256'], max_risk_error_vs_saved=maximum,
                max_RMS_error_vs_saved=max(value['RMS_error'] for value in cell['repeats']),
                repetitions_within_original_gate=sum(value['within_original_2e_minus6_gate'] for value in cell['repeats']),
                fixed_repetitions=5, maximum_cross_repeat_error=max(value['max_abs_error'] for value in cell['cross_repeat_comparisons']),
                cross_repeats_bitwise_equal=all(value['bitwise_equal'] for value in cell['cross_repeat_comparisons']),
                all_saved_reference_rank_credits_and_H_U_unchanged=unchanged,
                all_cross_repeat_pair_credits_unchanged=all(value['pair_credit_changes']['changed_pair_credits'] == 0
                    for value in cell['cross_repeat_comparisons']),
                weights_unchanged=True, gamma=.3,
                all_five_repetitions=cell['repeats'], all_ten_cross_repeat_comparisons=cell['cross_repeat_comparisons']))
    require({row['backend'] for row in rows} == {'inherited', 'deterministic'} and len(rows) == 6, 'Two backend diagnostics required')
    unchanged = all(row['all_saved_reference_rank_credits_and_H_U_unchanged'] and row['all_cross_repeat_pair_credits_unchanged'] for row in rows)
    return dict(schema='bgb_en_O9_float32_drift_note_v1',
        status='FLOAT32_DRIFT_WITH_UNCHANGED_RANK_DIAGNOSIS' if unchanged else 'FLOAT32_DRIFT_WITH_RANK_CHANGES_DIAGNOSIS',
        original_audit=dict(status=original['status'], passed_units=184, failed_units=2,
            failed_cells=[{key: value[key] for key in ('stage', 'arm', 'inner_partition', 'seed', 'failure_code')} for value in failures],
            overwritten=False, promoted_to_PASS=False, original_tolerance=2e-6),
        fixed_complete_validation_replays=30, cross_repeat_comparisons=60,
        replay_count_boundary='30 complete validation risk-vector predictions; each uses multiple microbatch8 network forwards, whose total was not persisted',
        optimizer_fits=0, optimizer_updates=0, original_threshold_relaxed=False, original_audit_regraded=False,
        selected_a_favorable_repeat=False, model_selection_lock_changed=False, all_rank_credits_and_H_U_unchanged=unchanged,
        cells=rows,
        invalid_v1=dict(status='INVALIDATED_GAMMA_CONSTRUCTION_DIAGNOSTICS', supplied_gamma=.1, required_gamma=.3,
            invalid_default_and_deterministic_outputs_excluded=True, privately_archived_by_primary_agent=True,
            training_or_original_audit_affected=False, boundary='The reporting script does not read or interpret the invalid v1 risk values'),
        causal_boundary='restricted to these three saved checkpoints and this recorded CUDA/PyTorch environment; nondeterministic reduction is a candidate, not a uniquely localized kernel cause',
        calibration_boundary='unchanged Harrell/Uno and comparable-pair credit do not establish identical raw risks, calibration, every possible ranking, or cross-hardware reproducibility')


def markdown(report):
    require(report['status'] == 'FLOAT32_DRIFT_WITH_UNCHANGED_RANK_DIAGNOSIS',
            'An unchanged-rank note cannot be produced when ranking changed')
    lines = ['# O9 float32 数值漂移追加说明', '',
        '**分类：FLOAT32_DRIFT_WITH_UNCHANGED_RANK_DIAGNOSIS。原检查点审计仍为184/186 PASS、2条 FAILED，不改判PASS。**', '',
        '## 固定诊断合同', '',
        '固定原失败inner2/seed31、inner2/seed47及通过控制inner1/seed31；分别在继承后端和确定性后端各回放5次，使用同一权重、完整验证分区、batch8及锁定R2 γ=0.3。共30次完整验证风险向量预测；每次由多个microbatch网络前向构成。所有5次及跨重复比较完整保留，未择一次PASS。', '',
        '|后端|partition|seed|相对saved最大风险误差|最大RMS误差|5次中满足原2e-6门槛|跨重复最大误差|pair credit / ΔH / ΔUno|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in report['cells']:
        lines.append(f"|{row['backend']}|{row['inner_partition']}|{row['seed']}|{row['max_risk_error_vs_saved']:.9g}|{row['max_RMS_error_vs_saved']:.9g}|{row['repetitions_within_original_gate']}/5|{row['maximum_cross_repeat_error']:.9g}|全部0变化|")
    lines += ['', '30次回放相对原saved风险的可比较患者对credit变化、Harrell变化和Uno60变化全部为0；60组跨重复比较也无credit变化。所有权重和buffer哈希保持不变，无反传、无新拟合。重复数不是独立患者或外部验证样本量。', '',
        '## 应怎样解释', '',
        '继承后端可出现重复间差异；确定性后端五次彼此逐位一致，但不保证与原继承后端保存的风险逐位相同。inner2/seed47在确定性后端仍有3.3378601e-6的固定偏差，因此“可确定性重复”不等于“满足原2e-6回放门槛”。', '',
        '**保留原两条数值回放失败。** 本追加诊断只确认这3个检查点在当前记录环境中存在很小的原始风险值漂移，且本次评测所用可比较患者对排序及Harrell/Uno60未改变。不放宽原门槛，不为取得PASS重训，也不将原184/186改成186/186。未改变候选选择锁。', '',
        '本轮记录为PyTorch 2.5.1+cu124、CUDA build12.4；诊断JSON没有GPU型号，不声称跨GPU或其他软件版本复现。CUDA归约等数值差是候选解释，当前诊断未逐算子定位唯一因果。排序不变也不意味着原始风险、校准或所有可能的排序完全相同。', '',
        '## 作废的v1诊断', '',
        '最初新诊断遗漏构造参数，误用了MechanismModel默认γ=0.1，而原R2锁定γ=0.3。其默认/确定性两份输出及源码已由主代理私有归档并标为INVALID_GAMMA；这些值完全排除，不用于解释原检查点失败。该错误只影响额外的零拟合诊断，未改变原训练、权重或184/186审计。V2显式传入并记录γ=0.3，另有实际构造路径测试验证。', '',
        '## 证据', '',
        f"原审计SHA256：`{report['input_sha256']['CHECKPOINT_AUDIT.json']}`。", '',
        '全部30次数值、60组跨重复比较及输入SHA见 `O9_FLOAT32_DRIFT_AGGREGATE.json`；原两份V2 JSON及CHECKPOINT_AUDIT.json保持原样。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--default', type=Path, required=True)
    parser.add_argument('--deterministic', type=Path, required=True)
    parser.add_argument('--checkpoint-audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/bgb_en')
    args = parser.parse_args()
    inputs = {'CHECKPOINT_AUDIT.json': args.checkpoint_audit, 'O9_FLOAT32_DRIFT_DEFAULT.json': args.default,
              'O9_FLOAT32_DRIFT_DETERMINISTIC.json': args.deterministic}
    hashes = {name: sha(path) for name, path in inputs.items()}
    documents = [json.loads(path.read_text()) for path in (args.default, args.deterministic)]
    require(documents[0]['backend']['mode'] == 'inherited' and documents[1]['backend']['mode'] == 'deterministic', 'Backend file roles differ')
    require(all(value['prior_audit_sha256'] == hashes['CHECKPOINT_AUDIT.json'] for value in documents), 'Diagnostics reference a different original audit')
    require(all(value['source_sha256'] == sha(ROOT/'scripts/bgb_en_o9_drift_diagnostic.py') for value in documents), 'Diagnostic source differs')
    report = aggregate_documents(json.loads(args.checkpoint_audit.read_text()), documents)
    report.update(input_sha256=hashes, report_source_sha256=sha(__file__),
                  diagnostic_source_sha256=documents[0]['source_sha256'], backends=[value['backend'] for value in documents])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'O9_FLOAT32_DRIFT_AGGREGATE.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.output_dir/'O9_FLOAT32_DRIFT_NOTE.md').write_text(markdown(report), encoding='utf-8')
    require(hashes == {name: sha(path) for name, path in inputs.items()}, 'A source audit was modified')
    print(json.dumps(dict(status=report['status'], complete_validation_replays=30, optimizer_fits=0, original_audit_promoted_to_PASS=False)))


if __name__ == '__main__':
    main()

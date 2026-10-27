"""Read fixed20/40/80 logs; never select an epoch or launch budget retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_anchor_cast_audit import comparable, censor_weights


LOCK = dict(epochs=[20, 40, 80], loss_tolerance=1e-8, Harrell_tolerance=1e-8,
            recorded_gradient_norm_threshold=1e-8, gradient_nonzero_ratio_threshold=0.,
            required_complete_cells_per_arm=6, minimum_follow_up_direction_cells=4,
            epoch_selection=False, budget_retraining_authorized_by_this_script=False,
            recorded_gradient_includes_regularization=True,
            pure_interaction_gradient_requires_separate_mechanism_replay=True,
            frozen_gamma_values=[0., .03, .1, .3], frozen_delta_multipliers=[0., .3, 1., 3.],
            frozen_scaling_is_not_gamma_retraining=True,
            train_Cox_log_phase='before the named epoch optimizer update',
            validation_metric_log_phase='after the named epoch optimizer update')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def classify(curves):
    epochs = {int(row['epoch']): row for row in curves}
    if any(epoch not in epochs for epoch in LOCK['epochs']):
        raise ValueError('Actual fixed20/40/80 log missing; do not infer a learning curve')
    first, last = epochs[40], epochs[80]
    train_decreased = float(last['cox_loss']) < float(first['cox_loss'])-LOCK['loss_tolerance']
    valid_improved = float(last['valid']['Harrell']) > float(first['valid']['Harrell'])+LOCK['Harrell_tolerance']
    gradient_active = float(last['gradient_norm']) > LOCK['recorded_gradient_norm_threshold'] and \
                      float(last['gradient_tensor_nonzero_ratio']) > LOCK['gradient_nonzero_ratio_threshold']
    records = []
    for epoch in LOCK['epochs']:
        row = epochs[epoch]
        train_delta_h = row['train']['repair']-row['train']['damage']
        valid_delta_h = row['valid']['repair']-row['valid']['damage']
        train_delta_u = row['train']['Uno_repair']-row['train']['Uno_damage']
        valid_delta_u = row['valid']['Uno_repair']-row['valid']['Uno_damage']
        records.append(dict(epoch=epoch, actual_train_Efron_Cox=float(row['cox_loss']),
            train_H=float(row['train']['Harrell']), train_Uno60=float(row['train']['Uno60']),
            validation_H=float(row['valid']['Harrell']), validation_Uno60=float(row['valid']['Uno60']),
            train_minus_validation_H=float(row['train']['Harrell']-row['valid']['Harrell']),
            train_minus_validation_Uno60=float(row['train']['Uno60']-row['valid']['Uno60']),
            train_delta_H_vs_own_offset=float(train_delta_h), validation_delta_H_vs_own_offset=float(valid_delta_h),
            train_delta_Uno60_vs_own_offset=float(train_delta_u), validation_delta_Uno60_vs_own_offset=float(valid_delta_u),
            train_minus_validation_increment_H=float(train_delta_h-valid_delta_h),
            train_minus_validation_increment_Uno60=float(train_delta_u-valid_delta_u),
            recorded_optimizer_gradient_norm=float(row['gradient_norm']),
            recorded_gradient_tensor_nonzero_ratio=float(row['gradient_tensor_nonzero_ratio']),
            delta_train_SD=float(row['delta_train_SD']), delta_validation_SD=float(row['delta_valid_SD']),
            delta_train_bound_fraction=row.get('delta_train_bound_fraction')))
    return dict(fixed_epoch_log=records, train_Cox_40to80_decreased=train_decreased,
                validation_H_40to80_increased=valid_improved,
                validation_Uno60_40to80_increased=float(last['valid']['Uno60']) > float(first['valid']['Uno60'])+1e-8,
                recorded_optimizer_gradient_active_at80=gradient_active,
                strict_recorded_follow_up_direction=bool(train_decreased and valid_improved and gradient_active),
                train_Cox_change_40to80=float(last['cox_loss']-first['cox_loss']),
                validation_H_change_40to80=float(last['valid']['Harrell']-first['valid']['Harrell']),
                validation_Uno60_change_40to80=float(last['valid']['Uno60']-first['valid']['Uno60']))


def frozen_scaling(base, delta, saved_risk, train_time, train_event, time, event):
    base32 = np.asarray(base, np.float32)
    delta32 = np.asarray(delta, np.float32)
    replay = (base32+delta32).astype(np.float64)
    saved = np.asarray(saved_risk, np.float64)
    if not np.array_equal(replay, saved):
        raise ValueError('Gamma0.1 float32 base+delta does not reproduce saved risk elementwise')
    left, right = comparable(time, event)
    weight = censor_weights(train_time, train_event, time, event)[left]
    reference = base32.astype(np.float64)
    def credits(risk):
        margin = risk[left]-risk[right]
        return np.where(np.abs(margin) <= 1e-8, .5, margin > 0)
    baseline = credits(reference)
    hd, ud = len(left), float(weight.sum())
    result = []
    for gamma, multiplier in zip(LOCK['frozen_gamma_values'], LOCK['frozen_delta_multipliers']):
        # Both multiplication and addition round in float32, in the original
        # eager operation order. Only then does the independent evaluator cast.
        risk = (base32+delta32*np.float32(multiplier)).astype(np.float64)
        credit = credits(risk)
        change = credit-baseline
        result.append(dict(gamma=gamma, fixed_delta_multiplier=multiplier,
            Harrell=float(credit.sum()/hd), Harrell_num=float(credit.sum()), Harrell_den=hd,
            Uno60=float(np.dot(weight, credit)/ud), Uno_num=float(np.dot(weight, credit)), Uno_den=ud,
            delta_Harrell_vs_F1=float(change.sum()/hd), delta_Uno60_vs_F1=float(np.dot(weight, change)/ud),
            repair=float(np.maximum(change, 0).sum()/hd), damage=float(np.maximum(-change, 0).sum()/hd),
            Uno_repair=float(np.dot(weight, np.maximum(change, 0))/ud),
            Uno_damage=float(np.dot(weight, np.maximum(-change, 0))/ud),
            risk_shift_SD=float((risk-reference).std()),
            risk_shift_SD_over_F1_SD=float((risk-reference).std()/max(reference.std(), 1e-12)),
            gamma01_saved_risk_elementwise_equal=bool(np.array_equal(risk, saved)) if gamma == .1 else None,
            retrained=False, reference='same-patient frozen F1 offset cast to float32 before risk addition'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mechanism-root', type=Path)
    parser.add_argument('--prepared-private', type=Path)
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepared_cache = {}
    cells, skipped = [], []
    for path in sorted((args.private/'cells').glob('*/COMPLETE.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('stage') not in ('R1', 'R2', 'R3'):
            continue
        identity = dict(stage=receipt['stage'], arm=receipt['arm'], seed=receipt['seed'],
                        partition=receipt['inner_partition'], outer_fold=receipt['outer_fold'])
        if 'curves' not in receipt:
            skipped.append(dict(**identity, reason='convex/no neural training curve; no epoch inference'))
            continue
        artifact = path.parent/'model.pt'
        if sha(artifact) != receipt['artifact_sha256']:
            raise ValueError('Budget checkpoint SHA differs')
        checkpoint = torch.load(artifact, map_location='cpu', weights_only=False)
        if checkpoint['curves'] != receipt['curves'] or checkpoint.get('epochs') != 80:
            raise ValueError('Public log and actual final80 checkpoint curves disagree')
        result = classify(checkpoint['curves'])
        pure_active = None
        if args.mechanism_root is not None:
            mechanism_path = args.mechanism_root/(path.parent.name+'.json')
            if mechanism_path.exists():
                replay = json.loads(mechanism_path.read_text())
                if replay['contract']['artifact_sha256'] != receipt['artifact_sha256']:
                    raise ValueError('Mechanism replay references a different checkpoint')
                gradients = replay['loss_chain_gradients']['interaction']['input_gradients']
                pure_active = bool(gradients['RNA_input']['gradient_L2'] > 1e-12 and gradients['patch_input']['gradient_L2'] > 1e-12)
        scaling = None
        if identity['stage'] == 'R3':
            prepared_root = args.private if args.prepared_private is None else args.prepared_private
            prepared_path = prepared_root/'prepared'/f"outer{identity['outer_fold']}_inner{identity['partition']}.pt"
            if prepared_path not in prepared_cache:
                prepared_cache[prepared_path] = torch.load(prepared_path, map_location='cpu', weights_only=False)
            prepared = prepared_cache[prepared_path]
            if not np.array_equal(checkpoint['valid'], prepared['valid']):
                raise ValueError('Scaling diagnostic checkpoint/prepared validation rows differ')
            # Prepared stores training endpoints only. Recover validation DSS
            # from the endpoint source explicitly supplied by the run lineage.
            # The sibling assets configuration is the source of its path.
            assets_path = args.private.parent/'assets_config.json' if args.assets is None else args.assets
            if not assets_path.is_file():
                raise ValueError('Scaling diagnostic requires --assets when private root has no sibling assets_config.json')
            assets = json.loads(assets_path.read_text())
            if sha(assets['endpoints']) != checkpoint['lineage']['endpoints_sha256']:
                raise ValueError('Scaling endpoint source SHA differs from checkpoint')
            with np.load(assets['endpoints'], allow_pickle=False) as endpoint:
                time, event = np.array(endpoint['DSS_time']), np.array(endpoint['DSS_event'])
            train, valid = prepared['train'], prepared['valid']
            scaling = frozen_scaling(prepared['base'][valid], checkpoint['snapshots'][80]['delta'],
                checkpoint['snapshots'][80]['risk'], time[train], event[train], time[valid], event[valid])
        cells.append(dict(**identity, **result, checkpoint_sha256=receipt['artifact_sha256'],
                          pure_interaction_loss_chain_inputs_active=pure_active,
                          frozen_gamma_output_scaling=scaling))
    groups = {}
    for row in cells:
        groups.setdefault((row['stage'], row['arm']), []).append(row)
    summaries = []
    for (stage, arm), rows in sorted(groups.items()):
        count = sum(row['strict_recorded_follow_up_direction'] for row in rows)
        complete = len(rows) == LOCK['required_complete_cells_per_arm']
        summary = dict(stage=stage, arm=arm, completed_cells=len(rows), expected_cells=6,
            strict_40to80_follow_up_direction_cells=count,
            strict_40to80_with_Uno_also_up_cells=sum(row['strict_recorded_follow_up_direction'] and row['validation_Uno60_40to80_increased'] for row in rows),
            candidate_for_root_budget_review=bool(complete and count >= 4),
            pure_interaction_inputs_active_available_cells=sum(row['pure_interaction_loss_chain_inputs_active'] is not None for row in rows),
            pure_interaction_inputs_active_cells=sum(row['pure_interaction_loss_chain_inputs_active'] is True for row in rows),
            final80_mean_train_validation_H_gap=float(np.mean([row['fixed_epoch_log'][-1]['train_minus_validation_H'] for row in rows])),
            final80_mean_train_validation_Uno_gap=float(np.mean([row['fixed_epoch_log'][-1]['train_minus_validation_Uno60'] for row in rows])),
            mean_validation_H_change_40to80=float(np.mean([row['validation_H_change_40to80'] for row in rows])),
            mean_validation_Uno60_change_40to80=float(np.mean([row['validation_Uno60_change_40to80'] for row in rows])))
        for split in ('train', 'validation'):
            for metric in ('H', 'Uno60'):
                name = split+'_delta_'+metric+'_vs_own_offset'
                summary['final80_mean_'+name] = float(np.mean([row['fixed_epoch_log'][-1][name] for row in rows]))
        summary['final80_train_positive_valid_negative_increment_H_cells'] = sum(
            row['fixed_epoch_log'][-1]['train_delta_H_vs_own_offset'] > 0 and
            row['fixed_epoch_log'][-1]['validation_delta_H_vs_own_offset'] < 0 for row in rows)
        summaries.append(summary)
    result = dict(status='READ_ONLY_FIXED_BUDGET_DIAGNOSTICS', lock=LOCK, script_sha256=sha(__file__),
        actual_audit_optimizer_updates=0, completed_neural_cells=len(cells), skipped_no_curve_cells=skipped,
        independent_evaluation_source_sha256=sha(ROOT/'scripts/bgb_en_anchor_cast_audit.py'),
        arms=summaries, cells=cells,
        frozen_gamma_diagnostic_boundary='Fixed scaling of an already trained output, not a re-trained gamma model; no best gamma is selected',
        actual_gamma_retraining_completed_by_this_audit=False,
        interpretation='Direction flag motivates root review only; not proof of insufficient budget or new independent validation performance',
        gradient_boundary='Recorded optimizer gradients include regularization; pure interaction gradients require the separate zero-L2 mechanism replay',
        train_validation_anchor_boundary='Training offsets are crossfitted teachers; validation offsets use full-fit teachers. Absolute train<validation is not underfitting evidence. Increment deltas use each split own offset.',
        resume_boundary='Any authorized160/320 follow-up needs new same-initialization fits; checkpoint weights without Adam state cannot be described as exact resume')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ['# 固定训练预算诊断', '', '仅读取实际20/40/80日志与完整checkpoint，不训练、不选最好epoch。', '',
        '严格方向：40→80训练Efron Cox下降超过1e-8、验证Harrell上升超过1e-8，且记录的optimizer梯度norm>1e-8、非零tensor比例>0。只有完整6 cells中至少4 cells满足时，提示root复核预算。', '',
        '**记录的梯度包含L2。** 它不单独证明交互路线收到生存梯度；纯交互loss-chain结果另列。记录的Cox损失取该epoch更新前，训练/验证指标取更新后；不把它们错误称为完全相同瞬间的测量。', '',
        '|阶段|arm|完成cells|严格方向cells|Uno也上升|80轮H训练−验证差|40→80平均ΔH|预算复核候选|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for row in summaries:
        lines.append(f"|{row['stage']}|{row['arm']}|{row['completed_cells']}/6|{row['strict_40to80_follow_up_direction_cells']}|{row['strict_40to80_with_Uno_also_up_cells']}|{row['final80_mean_train_validation_H_gap']:.6f}|{row['mean_validation_H_change_40to80']:.6f}|{row['candidate_for_root_budget_review']}|")
    lines += ['', '## 相对各自锚点的增量', '',
        '训练offset来自crossfit教师，验证offset来自full-fit教师。绝对train<valid不能据此判欠拟合；以下增量分别相对本人partition/split的offset，以repair−damage计算。', '',
        '|阶段|arm|80轮train ΔH|80轮valid ΔH|80轮train ΔUno|80轮valid ΔUno|train正/valid负H cells|',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"|{row['stage']}|{row['arm']}|{row['final80_mean_train_delta_H_vs_own_offset']:.6f}|{row['final80_mean_validation_delta_H_vs_own_offset']:.6f}|{row['final80_mean_train_delta_Uno60_vs_own_offset']:.6f}|{row['final80_mean_validation_delta_Uno60_vs_own_offset']:.6f}|{row['final80_train_positive_valid_negative_increment_H_cells']}|")
    lines += ['', '这些是同一开发队列的学习曲线诊断，不是独立性能证据。训练损失单独下降不构成预算不足证据。本诊断程序不执行160/320轮训练；另行锁定的预算实验及其实际状态见R2_BUDGET_SENSITIVITY报告。追加预算采用同初始化的新模型与新Adam，不能把缺少Adam状态的权重加载称作精确续训。', '',
        'R3另固定γ=0/.03/.1/.3，对已训练delta作0/.3/1/3倍缩放。乘法与本人base相加均保持原float32运算，γ=.1必须逐元素回放原saved risk。该诊断不训练、不选最优γ，不构成四个真实重训模型；真实γ重训尚未由此完成。逐cell数值见JSON。', '']
    args.output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(completed_neural_cells=len(cells), review_candidates=[row['arm'] for row in summaries if row['candidate_for_root_budget_review']], optimizer_updates=0)))


if __name__ == '__main__':
    main()

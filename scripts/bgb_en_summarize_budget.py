"""Anonymous, independent aggregation of the locked24 new budget fits.

Eighty-epoch main rows use the original parent models. Each160/320 main row
uses that independent fit's final snapshot. Intermediate snapshots are retained
as trajectories/replay diagnostics and never counted as extra fitted models.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.bgb_en_anchor_cast_audit import comparable, censor_weights
from scripts.bgb_en_budget_retrain import compare_parent80


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def independent_metrics(time, event, train_time, train_event, risk, reference):
    left, right = comparable(time, event)
    weight = censor_weights(train_time, train_event, time, event)[left]
    def credit(value):
        margin = np.asarray(value, np.float64)[left]-np.asarray(value, np.float64)[right]
        return np.where(np.abs(margin) <= 1e-8, .5, margin > 0)
    value, original = credit(risk), credit(reference)
    shift = value-original
    hd, ud = len(left), float(weight.sum())
    return dict(Harrell_num=float(value.sum()), Harrell_den=hd, Uno_num=float(np.dot(weight, value)), Uno_den=ud,
        Harrell=float(value.mean()), Uno60=float(np.dot(weight, value)/ud),
        repair=float(np.maximum(shift, 0).sum()/hd), damage=float(np.maximum(-shift, 0).sum()/hd),
        Uno_repair=float(np.dot(weight, np.maximum(shift, 0))/ud), Uno_damage=float(np.dot(weight, np.maximum(-shift, 0))/ud))


def aggregate_metrics(rows):
    total = {name: float(sum(row[name] for row in rows)) for name in ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den')}
    for kind, numerator, denominator in (('Harrell', 'Harrell_num', 'Harrell_den'), ('Uno60', 'Uno_num', 'Uno_den')):
        total[kind] = total[numerator]/total[denominator]
    for name, denominator in (('repair', 'Harrell_den'), ('damage', 'Harrell_den'), ('Uno_repair', 'Uno_den'), ('Uno_damage', 'Uno_den')):
        total[name] = float(sum(row[name]*row[denominator] for row in rows)/total[denominator])
    total['delta_H_vs_own_offset'] = total['repair']-total['damage']
    total['delta_Uno60_vs_own_offset'] = total['Uno_repair']-total['Uno_damage']
    return total


def mean_available(values):
    # A missing decoder evaluation must not silently change the folds/seeds
    # behind a reported mean. Per-cell values and availability remain visible.
    return float(np.mean([float(value) for value in values])) if values and all(value is not None for value in values) else None


def main_row(arm, seed, partition, epoch, artifact, receipt, prepared, time, event, parent_risk):
    train, valid = prepared['train'], prepared['valid']
    snapshot = artifact['snapshots'][epoch]
    risk = np.asarray(snapshot['risk'], np.float64)
    delta = np.asarray(snapshot['delta'], np.float64)
    offset = np.asarray(prepared['RNA'][valid], np.float32)
    require(np.array_equal((offset+delta.astype(np.float32)).astype(np.float64), risk),
            'Actual float32 RNA offset plus delta does not replay saved risk')
    metrics = independent_metrics(time[valid], event[valid], time[train], event[train], risk, offset.astype(np.float64))
    for name in ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den', 'Harrell', 'Uno60', 'repair', 'damage', 'Uno_repair', 'Uno_damage'):
        require(abs(metrics[name]-snapshot['metrics'][name]) < 1e-9, 'Independent final validation metric differs: '+name)
    parent_metric = independent_metrics(time[valid], event[valid], time[train], event[train], risk, parent_risk)
    curve = next(row for row in artifact['curves'] if row['epoch'] == epoch)
    require(curve['valid'] == snapshot['metrics'], 'Curve and snapshot validation metrics differ')
    training = curve['train']
    retention = artifact['retention']
    train_offset = np.asarray(prepared['RNA'][train], np.float32).astype(np.float64)
    row = dict(arm=arm, seed=seed, partition=partition, epoch=epoch,
        source='original80_parent' if epoch == 80 else 'independent_new_fit_final',
        validation=metrics, training=training, parent80_comparison=parent_metric,
        train_delta_H_vs_own_offset=float(training['repair']-training['damage']),
        train_delta_Uno60_vs_own_offset=float(training['Uno_repair']-training['Uno_damage']),
        valid_delta_H_vs_own_offset=float(metrics['repair']-metrics['damage']),
        valid_delta_Uno60_vs_own_offset=float(metrics['Uno_repair']-metrics['Uno_damage']),
        valid_delta_H_vs_parent80=float(parent_metric['repair']-parent_metric['damage']),
        valid_delta_Uno60_vs_parent80=float(parent_metric['Uno_repair']-parent_metric['Uno_damage']),
        train_offset_SD=float(train_offset.std()), valid_offset_SD=float(offset.astype(np.float64).std()),
        train_delta_SD=float(curve['delta_train_SD']), valid_delta_SD=float(delta.std()),
        train_delta_SD_over_offset_SD=float(curve['delta_train_SD']/max(train_offset.std(), 1e-12)),
        valid_delta_SD_over_offset_SD=float(delta.std()/max(offset.astype(np.float64).std(), 1e-12)),
        valid_actual_risk_shift_SD=float((risk-offset.astype(np.float64)).std()),
        parameters=int(artifact['parameters']), peak_GPU_bytes=int(artifact['peak_GPU_bytes']),
        training_walltime_seconds=float(artifact['elapsed_seconds']), total_cell_walltime_seconds=float(receipt['elapsed_seconds']),
        actual_optimizer_updates=int(artifact['optimizer_updates']), retention=retention,
        train_metric_boundary='saved actual checkpoint curve; no train-risk vector persisted for an independent full train-pair replay',
        reference='each split own frozen RNA512 offset; training crossfit and validation full-fit teacher')
    require(abs(row['valid_delta_SD']-curve['delta_valid_SD']) < 1e-10, 'Recorded delta SD differs from saved validation delta')
    return row


def aggregate_rows(cells):
    groups = {}
    for row in cells:
        groups.setdefault((row['arm'], row['epoch'], row['seed']), []).append(row)
    seeds = []
    for (arm, epoch, seed), rows in sorted(groups.items()):
        require(len({row['partition'] for row in rows}) == len(rows), 'Duplicate partition contributes twice')
        valid, train = aggregate_metrics([row['validation'] for row in rows]), aggregate_metrics([row['training'] for row in rows])
        parent = aggregate_metrics([row['parent80_comparison'] for row in rows])
        seeds.append(dict(arm=arm, epoch=epoch, seed=seed, partitions=len(rows), complete_three_partitions=len(rows) == 3,
            validation=valid, training=train, parent80_comparison=parent,
            complete_tokens_decoder_Pearson=mean_available([row['retention']['complete_tokens']['Pearson'] for row in rows]),
            complete_tokens_decoder_R2=mean_available([row['retention']['complete_tokens']['R2'] for row in rows]),
            pooled32_decoder_Pearson=mean_available([row['retention']['pooled32']['Pearson'] for row in rows]),
            pooled32_decoder_R2=mean_available([row['retention']['pooled32']['R2'] for row in rows]),
            mean_valid_delta_SD_over_offset_SD=float(np.mean([row['valid_delta_SD_over_offset_SD'] for row in rows])),
            sum_per_fit_training_walltime_seconds=float(sum(row['training_walltime_seconds'] for row in rows)),
            sum_per_fit_total_walltime_seconds=float(sum(row['total_cell_walltime_seconds'] for row in rows)),
            parameter_min=min(row['parameters'] for row in rows), parameter_max=max(row['parameters'] for row in rows),
            peak_GPU_bytes=max(row['peak_GPU_bytes'] for row in rows)))
    aggregate = []
    for arm, epoch in sorted({(row['arm'], row['epoch']) for row in seeds}):
        rows = [row for row in seeds if row['arm'] == arm and row['epoch'] == epoch]
        entry = dict(arm=arm, epoch=epoch, seeds=len(rows), cells=sum(row['partitions'] for row in rows),
                     complete_six_cells=len(rows) == 2 and all(row['complete_three_partitions'] for row in rows))
        for split in ('validation', 'training', 'parent80_comparison'):
            entry[split] = {name: float(np.mean([row[split][name] for row in rows]))
                            for name in ('Harrell', 'Uno60', 'repair', 'damage', 'Uno_repair', 'Uno_damage', 'delta_H_vs_own_offset', 'delta_Uno60_vs_own_offset')}
        for name in ('complete_tokens_decoder_Pearson', 'complete_tokens_decoder_R2', 'pooled32_decoder_Pearson', 'pooled32_decoder_R2'):
            entry[name] = mean_available([row[name] for row in rows])
        entry.update(mean_valid_delta_SD_over_offset_SD=float(np.mean([row['mean_valid_delta_SD_over_offset_SD'] for row in rows])),
            sum_per_fit_training_walltime_seconds=sum(row['sum_per_fit_training_walltime_seconds'] for row in rows),
            sum_per_fit_total_walltime_seconds=sum(row['sum_per_fit_total_walltime_seconds'] for row in rows),
            parameter_min=min(row['parameter_min'] for row in rows), parameter_max=max(row['parameter_max'] for row in rows),
            peak_GPU_bytes=max(row['peak_GPU_bytes'] for row in rows))
        aggregate.append(entry)
    return seeds, aggregate


def csv_write(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--primary-private', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/budget_sensitivity_v1.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/bgb_en')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    lock = json.loads(args.config.read_text())
    require(lock['arms'] == ['O3', 'O4'] and lock['new_fit_epochs'] == [160, 320] and lock['required_new_neural_fits'] == 24,
            'Unexpected sensitivity scope')
    assets = json.loads(args.assets.read_text())
    with np.load(assets['endpoints'], allow_pickle=False) as endpoint:
        time, event = np.array(endpoint['DSS_time']), np.array(endpoint['DSS_event'])
    sources = {name: sha(ROOT/name) for name in ('scripts/bgb_en_summarize_budget.py',
        'scripts/bgb_en_budget_retrain.py', 'scripts/bgb_en_anchor_cast_audit.py', 'configs/bgb_en/budget_sensitivity_v1.json')}
    parents, prepared, parent_receipts, provenance = {}, {}, {}, []
    cells, replay_rows, trajectories, initializations, completed = [], [], [], {}, []
    initialization_budgets = {}
    for partition in lock['inner_partitions']:
        path = args.primary_private/'prepared'/f'outer1_inner{partition}.pt'
        receipt_path = path.with_suffix('.json')
        record = json.loads(receipt_path.read_text())
        require(record['sha256'] == sha(path), 'Prepared origin SHA differs')
        prepared[partition] = torch.load(path, map_location='cpu', weights_only=False)
        require(sha(assets['endpoints']) == record['lineage']['endpoints_sha256'], 'Endpoint source SHA differs')
        provenance.append(dict(partition=partition, prepared_sha256=sha(path), prepared_receipt_sha256=sha(receipt_path)))
        for arm in lock['arms']:
            for seed in lock['seeds']:
                folder = args.primary_private/'cells'/f'R2_outer1_inner{partition}_{arm}_seed{seed}'
                record80 = json.loads((folder/'COMPLETE.json').read_text())
                require(record80['lineage'] == record['lineage'] and record80['artifact_sha256'] == sha(folder/'model.pt'),
                        'Parent80 lineage or artifact SHA differs')
                parent = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
                require(parent['epochs'] == 80 and np.array_equal(parent['valid'], prepared[partition]['valid']), 'Parent80 split differs')
                parents[arm, partition, seed], parent_receipts[arm, partition, seed] = parent, record80
                cells.append(main_row(arm, seed, partition, 80, parent, record80, prepared[partition], time, event, parent['snapshots'][80]['risk']))
                for epoch in lock['new_fit_epochs']:
                    name = f'R2_newfit{epoch}_outer1_inner{partition}_{arm}_seed{seed}'
                    directory = args.private/'cells'/name
                    if not (directory/'COMPLETE.json').exists():
                        continue
                    receipt = json.loads((directory/'COMPLETE.json').read_text())
                    require(receipt['status'] == 'COMPLETE_INDEPENDENT_BUDGET_FIT' and receipt['artifact_sha256'] == sha(directory/'model.pt'),
                            'New fit completion/artifact differs')
                    model = torch.load(directory/'model.pt', map_location='cpu', weights_only=False)
                    lineage = model['lineage']
                    require(lineage == receipt['lineage'] and lineage['core_origin'] == record['lineage'], 'New fit core source differs')
                    require(lineage['budget_lock_sha256'] == sha(args.config)
                            and lineage['independent_script_sha256'] == sources['scripts/bgb_en_budget_retrain.py']
                            and lineage['evaluation_script_sha256'] == sources['scripts/bgb_en_anchor_cast_audit.py'], 'New fit source/config differs')
                    require(lineage['parent80_artifact_sha256'] == record80['artifact_sha256']
                            and lineage['parent80_receipt_sha256'] == sha(folder/'COMPLETE.json')
                            and lineage['prepared_origin_sha256'] == sha(path)
                            and lineage['prepared_receipt_sha256'] == sha(receipt_path), 'New fit parent/prepared provenance differs')
                    require(model['optimizer_updates'] == epoch == model['epochs'] == receipt['epochs']
                            and model['actual_neural_optimizer_fits'] == 1 and model['actual_teacher_refits'] == 0
                            and model['actual_preprocessing_refits'] == 0 and model['final_retention_decoder_fits'] == 2
                            and model['optimizer_resumed'] is False and model['select_best_epoch'] is False
                            and model['outer_test_evaluated'] is False and model['R4_training_budget_changed'] is False,
                            'New fit count, resume, selection or outer-budget contract differs')
                    require(np.array_equal(model['train'], prepared[partition]['train']) and np.array_equal(model['valid'], prepared[partition]['valid']),
                            'New fit row partition differs')
                    key = arm, partition, seed
                    if key in initializations:
                        require(initializations[key] == model['initial_state_sha256'], '160/320 initial model states differ')
                    else:
                        initializations[key] = model['initial_state_sha256']
                    initialization_budgets.setdefault(key, set()).add(epoch)
                    train, valid = prepared[partition]['train'], prepared[partition]['valid']
                    comparison = compare_parent80(time[valid], event[valid], time[train], event[train],
                        parent['snapshots'][80]['risk'], model['snapshots'][80]['risk'], lock['parent80_validation_risk_max_abs_tolerance'])
                    require(comparison['within_locked_tolerance'], '80 replay exceeds tolerance; do not aggregate as a comparable budget fit')
                    for metric, value in comparison.items():
                        stored = model['parent80_replay'][metric]
                        require(value == stored or (isinstance(value, (int, float)) and abs(value-stored) < 1e-12), 'Stored parent80 replay certificate differs')
                    replay_rows.append(dict(arm=arm, partition=partition, seed=seed, independently_fitted_epochs=epoch,
                                            **comparison, original_parent_artifact_sha256=record80['artifact_sha256'],
                                            new_fit_artifact_sha256=receipt['artifact_sha256']))
                    cells.append(main_row(arm, seed, partition, epoch, model, receipt, prepared[partition], time, event, parent['snapshots'][80]['risk']))
                    trajectories.append(dict(arm=arm, seed=seed, partition=partition, independent_fit_epochs=epoch,
                                             actual_curve=model['curves'], counts_as_one_fit=True))
                    completed.append(dict(arm=arm, partition=partition, seed=seed, epochs=epoch,
                                          optimizer_updates=model['optimizer_updates'], decoders=2,
                                          artifact_sha256=receipt['artifact_sha256'], initial_state_dict_sha256=model['initial_state_sha256']))
    required = {(arm, part, seed, epoch) for arm in lock['arms'] for part in lock['inner_partitions'] for seed in lock['seeds'] for epoch in lock['new_fit_epochs']}
    actual = {(row['arm'], row['partition'], row['seed'], row['epochs']) for row in completed}
    missing = [dict(arm=arm, partition=part, seed=seed, epochs=epoch) for arm, part, seed, epoch in sorted(required-actual)]
    require(args.allow_partial or not missing, 'Only '+str(len(completed))+'/24 budget fits completed; --allow-partial is required for a progress-only report')
    per_seed, aggregate = aggregate_rows(cells)
    rank_changes = sum(row['changed_pair_credits'] for row in replay_rows)
    result = dict(status='COMPLETE_LOCKED24_BUDGET_SENSITIVITY' if not missing else 'PARTIAL_BUDGET_SENSITIVITY',
        sources=sources, lock=lock, actual_new_neural_fits=len(completed), required_new_neural_fits=24,
        actual_new_optimizer_updates=sum(row['optimizer_updates'] for row in completed), required_new_optimizer_updates=5760,
        actual_new_risk_decoder_fits=sum(row['decoders'] for row in completed), required_new_risk_decoder_fits=48,
        reused_original80_models=12, actual_new_teacher_refits=0, actual_audit_optimizer_updates=0,
        parent80_replay_pairs_with_changed_credit=rank_changes,
        parent80_replay_max_abs_risk_error=max((row['max_abs_risk_error'] for row in replay_rows), default=None),
        parent80_rank_changes_are_not_ignored=True, R4_epochs=80, outer_budget_changed=False, best_budget_selected=False,
        initialization_comparison_groups=len(initializations), required_initialization_comparison_groups=12,
        same_initialization_verified_160_320_groups=sum(epochs == {160, 320} for epochs in initialization_budgets.values()),
        initialization_comparison_coverage=[dict(arm=key[0], partition=key[1], seed=key[2], epochs=sorted(epochs),
                                                both_budgets_verified=epochs == {160, 320})
                                            for key, epochs in sorted(initialization_budgets.items())],
        aggregation='within each arm/epoch/seed sum the three partition concordance numerators and denominators, then mean the two seed ratios; never pool cross-partition risk vectors',
        training_aggregation_boundary='training partitions overlap; training aggregate is descriptive, not a new independent cohort',
        retention_aggregation='mean three decoder evaluations within seed, then mean seeds; each decoder trained only in that partition',
        walltime_boundary='sum of actual per-fit elapsed times is compute exposure, not calendar time for parallel jobs; original80 total time includes its old patch-bank overhead',
        rationale='O3/O4 four-of-six40to80 directions triggered a sensitivity check despite negative meanHarrell changes; insufficient training was not established',
        missing_cells=missing, completed_new_fit_lineage=completed, prepared_origins=provenance,
        per_cell=cells, per_seed=per_seed, seed_mean=aggregate, parent80_replays=replay_rows, new_fit_trajectories=trajectories)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir/'R2_BUDGET_SENSITIVITY'
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    flat = []
    for row in cells:
        record = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for split in ('validation', 'training', 'parent80_comparison'):
            record.update({split+'_'+name: value for name, value in row[split].items()})
        for name, values in row['retention'].items():
            record.update({name+'_decoder_'+key: value for key, value in values.items() if key in ('dimension', 'Pearson', 'R2')})
        flat.append(record)
    csv_write(prefix.with_suffix('.csv'), flat)
    csv_write(args.output_dir/'R2_BUDGET_REPLAY.csv', replay_rows)
    lines = ['# R2 固定预算敏感性', '', '**状态：'+result['status']+'。** 新增'+str(len(completed))+'/24个神经拟合，'+str(result['actual_new_optimizer_updates'])+'/5760次更新，'+str(result['actual_new_risk_decoder_fits'])+'/48个风险解码器。复用12个原始80轮模型，新增教师拟合0。', '',
        '80轮主结果来自原parent；160和320轮分别来自各自同初始化、全新Adam的新拟合终点。320轮内部的80/160快照只作轨迹回放，不重复算独立模型。全部预算并列报告，不选择最佳epoch，R4仍80轮。', '',
        '|arm|epoch|cells|Harrell|Uno60|valid修复|valid损伤|train ΔH/自身offset|valid ΔH/自身offset|ΔH/parent80|ΔUno/parent80|token decoder R²|pooled decoder R²|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in aggregate:
        v, tr, p80 = row['validation'], row['training'], row['parent80_comparison']
        def number(value):
            return 'NA' if value is None else f'{value:.6f}'
        lines.append(f"|{row['arm']}|{row['epoch']}|{row['cells']}/6|{v['Harrell']:.6f}|{v['Uno60']:.6f}|{v['repair']:.6f}|{v['damage']:.6f}|{tr['delta_H_vs_own_offset']:.6f}|{v['delta_H_vs_own_offset']:.6f}|{p80['delta_H_vs_own_offset']:.6f}|{p80['delta_Uno60_vs_own_offset']:.6f}|{number(row['complete_tokens_decoder_R2'])}|{number(row['pooled32_decoder_R2'])}|")
    lines += ['', '每seed先合并三partition分子/分母，再对seed求均值。没有跨fold拼风险排序。训练offset来自CF教师，验证offset来自full-fit教师；绝对train<valid不说明欠拟合，增量各自相对本人offset。', '',
        '80快照回放最大风险误差：'+str(result['parent80_replay_max_abs_risk_error'])+'；发生credit变化的可比较患者对计数：'+str(rank_changes)+'。小于2e-6也不忽略实际排序变化，逐项见R2_BUDGET_REPLAY.csv。', '',
        'O3/O4因多数cell方向触发检查，但此前平均Harrell变化为负。本阶段仅辨别预算影响，不预先认定预算不足。没有选择最佳预算，没有改变外层80轮合同。参数、逐fit实际训练/总耗时、GPU峰值、RNA decoder及全部轨迹见JSON/CSV。并行fit的耗时之和不等于日历walltime。', '']
    prefix.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], actual_new_neural_fits=len(completed),
                         actual_new_optimizer_updates=result['actual_new_optimizer_updates'],
                         actual_new_decoders=result['actual_new_risk_decoder_fits'], optimizer_updates_by_audit=0)))


if __name__ == '__main__':
    main()

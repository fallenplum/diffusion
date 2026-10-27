"""Anonymous reports from BG-BEN inner-cell COMPLETE receipts only.

Never opens model.pt, prepared files, patient predictions, coordinates or IDs.
Incomplete arms retain empty aggregate metrics, rather than silently changing
the comparison population. Fixed epoch-80 metrics are the scientific prototype
endpoint; epoch-20/40 snapshots remain learning-curve diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CONCORDANCE = ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den')
DIAGNOSTICS = ('cox_loss', 'regularizer', 'gradient_norm', 'replay_max', 'delta_train_SD',
               'delta_valid_SD', 'delta_train_bound_fraction', 'gradient_tensor_nonzero_ratio')
METRICS = ('Harrell', 'Uno60', *CONCORDANCE, 'repair', 'damage', 'Uno_repair', 'Uno_damage',
           'risk_shift_sd_ratio')


def deterministic(arm):
    return arm in ('P0', 'O0', 'O1', 'F0', 'F1') or arm.startswith('O2_')


def stored_risk_diagnostics_reference(stage, arm):
    """Actual saved SD/repair reference, distinct from a report's paired arm."""
    if deterministic(arm):
        # convex_unit evaluates every deterministic arm against full LOCAL,
        # even O0/O1/O2 and F1. This is F0 on the validation partition.
        return 'F0'
    return {'R1': 'CS', 'R2': 'O0', 'R3': 'F1'}[stage]


def stored_risk_increment(metric, uno=False):
    repair, damage = ('Uno_repair', 'Uno_damage') if uno else ('repair', 'damage')
    if metric.get(repair) is None or metric.get(damage) is None:
        return None
    return finite_number(metric[repair], repair) - finite_number(metric[damage], damage)


def average(values):
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def all_defined_mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Nonfinite or nonnumeric receipt field: {label}')
    return float(value)


def validate_metrics(value):
    for key in CONCORDANCE:
        finite_number(value[key], key)
    if min(value[k] for k in CONCORDANCE) < 0:
        raise ValueError('Negative concordance count')
    for name, num, den in [('Harrell', 'Harrell_num', 'Harrell_den'), ('Uno60', 'Uno_num', 'Uno_den')]:
        actual = value.get(name)
        if value[den] > 0:
            if actual is None or abs(finite_number(actual, name) - value[num] / value[den]) > 1e-10:
                raise ValueError('Metric disagrees with numerator/denominator: ' + name)
        elif actual is not None:
            raise ValueError('Undefined metric must be null')


def aggregate(rows):
    """Within-partition eligible counts first; no cross-fold patient risks."""
    total = {key: sum(float(row[key]) for row in rows) for key in CONCORDANCE}
    total['Harrell'] = total['Harrell_num'] / total['Harrell_den'] if total['Harrell_den'] else None
    total['Uno60'] = total['Uno_num'] / total['Uno_den'] if total['Uno_den'] else None
    for name, den in [('repair', 'Harrell_den'), ('damage', 'Harrell_den'),
                      ('Uno_repair', 'Uno_den'), ('Uno_damage', 'Uno_den')]:
        if all(row.get(name) is not None for row in rows) and total[den] > 0:
            total[name] = sum(float(row[name]) * row[den] for row in rows) / total[den]
        else:
            total[name] = None
    total['risk_shift_sd_ratio'] = average([row.get('risk_shift_sd_ratio') for row in rows])
    return total


def planned_arms(config, stage):
    values = config[stage]['arms'][:]
    if stage == 'R3':
        values += config[stage].get('additional_real_retraining', [])
    return values


def load_cells(private, config):
    """Only COMPLETE.json under cells is touched. Unknown JSON keys stay private."""
    cells = {}
    sources = set()
    outer = int(config['prototype']['outer_fold'])
    epochs = int(config['training']['epochs'])
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for path in sorted((Path(private) / 'cells').glob('**/COMPLETE.json')):
        raw_bytes = path.read_bytes()
        row = json.loads(raw_bytes)
        if row.get('status') != 'COMPLETE':
            raise ValueError('A COMPLETE receipt has a noncomplete status')
        stage, arm = row['stage'], row['arm']
        if stage not in ('R1', 'R2', 'R3') or arm not in planned_arms(config, stage):
            raise ValueError('Unknown stage/arm in formal receipts')
        seed, part = int(row['seed']), int(row['inner_partition'])
        expected_seeds = [31] if deterministic(arm) else config['prototype']['neural_seeds']
        if seed not in expected_seeds or part not in range(config['prototype']['inner_splits']):
            raise ValueError('Receipt seed/partition outside protocol')
        if int(row['outer_fold']) != outer or row.get('outer_test_evaluated') is not False:
            raise ValueError('Prototype report cannot mix outer folds or outer evaluation')
        validate_metrics(row['metrics'])
        if not deterministic(arm):
            if int(row['epochs']) != epochs:
                raise ValueError('Neural receipt did not complete the fixed training budget')
            curves = row.get('curves', [])
            finals = [c for c in curves if c['epoch'] == epochs]
            if len(finals) != 1 or finals[0]['valid'] != row['metrics']:
                raise ValueError('Scientific endpoint differs from the fixed final snapshot')
        for control in row.get('frozen_controls', {}).values():
            validate_metrics(control)
        key = stage, arm, part, seed
        if key in cells:
            raise ValueError('Duplicate logical formal unit')
        row['_receipt_sha256'] = hashlib.sha256(raw_bytes).hexdigest()
        cells[key] = row
        sources.add(hashlib.sha256(json.dumps(row['lineage'], sort_keys=True).encode()).hexdigest())
    if len(sources) > 1:
        raise ValueError('Mixed source/data lineage: report aborted')
    return cells, {'source_lineage_sha256': next(iter(sources), None),
                   'configuration_canonical_sha256': config_hash}


def get_cell(cells, stage, arm, part, seed):
    return cells.get((stage, arm, part, 31 if deterministic(arm) else seed))


def arm_summary(cells, config, stage, arm):
    partitions = range(config['prototype']['inner_splits'])
    seeds = config['prototype']['neural_seeds']
    actual_seeds = [31] if deterministic(arm) else seeds
    rows = [cells[(stage, arm, part, seed)] for part in partitions for seed in actual_seeds
            if (stage, arm, part, seed) in cells]
    expected = len(partitions) * len(actual_seeds)
    complete = len(rows) == expected
    result = {'stage': stage, 'arm': arm, 'status': 'COMPLETE' if complete else 'RUNNING',
              'stored_risk_diagnostics_reference': stored_risk_diagnostics_reference(stage, arm),
              'completed_unique_cells': len(rows), 'planned_unique_cells': expected,
              'logical_seed_references': len(seeds) * len(partitions) if complete else None,
              'deterministic_seed_reuse': deterministic(arm),
              'actual_cell_optimizer_fits': sum(int(r['actual_optimizer_fits']) for r in rows),
              'parameters_mean': average([r['parameters'] for r in rows]),
              'peak_GPU_bytes_max': max((r['peak_GPU_bytes'] for r in rows), default=None),
              'sum_unit_elapsed_seconds': sum(float(r['elapsed_seconds']) for r in rows),
              'mean_unit_elapsed_seconds': average([r['elapsed_seconds'] for r in rows]),
              'Harrell': None, 'Uno60': None, 'train_Harrell': None, 'train_Uno60': None,
              'train_minus_valid_Harrell': None, 'train_minus_valid_Uno60': None,
              'train_delta_H_vs_stored_reference': None, 'train_delta_U60_vs_stored_reference': None,
              'valid_delta_H_vs_stored_reference': None, 'valid_delta_U60_vs_stored_reference': None,
              'delta_risk_sd_ratio_mean': None, 'repair': None, 'damage': None,
              'Uno_repair': None, 'Uno_damage': None}
    for seed in seeds:
        result[f'Harrell_seed{seed}'] = result[f'Uno60_seed{seed}'] = None
    if not complete:
        return result
    seed_metrics, train_metrics = [], []
    for seed in seeds:
        selected = [get_cell(cells, stage, arm, part, seed) for part in partitions]
        metric = aggregate([r['metrics'] for r in selected])
        seed_metrics.append(metric)
        result[f'Harrell_seed{seed}'] = metric['Harrell']
        result[f'Uno60_seed{seed}'] = metric['Uno60']
        if not deterministic(arm):
            train_metrics.append(aggregate([next(c['train'] for c in r['curves']
                                                  if c['epoch'] == config['training']['epochs'])
                                            for r in selected]))
    for key in ('Harrell', 'Uno60', 'repair', 'damage', 'Uno_repair', 'Uno_damage'):
        result[key] = all_defined_mean([metric[key] for metric in seed_metrics])
    for prefix, values in [('train', train_metrics), ('valid', seed_metrics)]:
        for name, uno in [('H', False), ('U60', True)]:
            result[f'{prefix}_delta_{name}_vs_stored_reference'] = all_defined_mean([
                stored_risk_increment(metric, uno) for metric in values])
    result['delta_risk_sd_ratio_mean'] = average([r['metrics'].get('risk_shift_sd_ratio') for r in rows])
    if train_metrics:
        for key in ('Harrell', 'Uno60'):
            result['train_' + key] = average([m[key] for m in train_metrics])
            if result['train_' + key] is not None and result[key] is not None:
                result['train_minus_valid_' + key] = result['train_' + key] - result[key]
    return result


def paired(cells, config, stage, candidate, control, *, frozen=None):
    seeds = config['prototype']['neural_seeds']
    partitions = range(config['prototype']['inner_splits'])
    label = f'{candidate}-{control}' if frozen is None else f'{candidate}-frozen:{frozen}'
    pairs = []
    for seed in seeds:
        for part in partitions:
            a = get_cell(cells, stage, candidate, part, seed)
            b = get_cell(cells, stage, control, part, seed) if frozen is None else a
            if a is not None and b is not None:
                bm = b['metrics'] if frozen is None else b.get('frozen_controls', {}).get(frozen)
                if bm is not None:
                    for denominator in ('Harrell_den', 'Uno_den'):
                        if not math.isclose(a['metrics'][denominator], bm[denominator], rel_tol=1e-12, abs_tol=1e-10):
                            raise ValueError('Paired models have different eligible pairs or IPCW: ' + label)
                    pairs.append((seed, part, a['metrics'], bm))
    expected = len(seeds) * len(partitions)
    result = {'stage': stage, 'comparison': label, 'candidate': candidate, 'control': control,
              'comparison_type': 'REAL_RETRAINING' if frozen is None else 'FROZEN_INTERVENTION',
              'status': 'COMPLETE' if len(pairs) == expected else 'RUNNING',
              'completed_logical_pairs': len(pairs), 'planned_logical_pairs': expected,
              'paired_delta_H': None, 'paired_delta_U60': None,
              'positive_seed_H': None, 'positive_seed_U60': None,
              'positive_partition_seed_H': None, 'positive_partition_seed_U60': None,
              'stored_risk_diagnostics_reference': 'candidate=' + stored_risk_diagnostics_reference(stage, candidate)
                  + ';control=' + stored_risk_diagnostics_reference(stage, control),
              'candidate_stored_risk_diagnostics_reference': stored_risk_diagnostics_reference(stage, candidate),
              'control_stored_risk_diagnostics_reference': stored_risk_diagnostics_reference(stage, control),
              'paired_delta_source': 'independent_concordance_numerator_denominator_difference',
              'repair_reference': 'candidate=' + stored_risk_diagnostics_reference(stage, candidate)
                  + ';control=' + stored_risk_diagnostics_reference(stage, control) + ';not_candidate_control_pairwise',
              'candidate_repair_vs_F1': None, 'candidate_damage_vs_F1': None,
              'control_repair_vs_F1': None, 'control_damage_vs_F1': None,
              'direct_candidate_control_repair_damage': 'NOT_AVAILABLE_FROM_ANONYMOUS_RECEIPTS'}
    for seed in seeds:
        result[f'delta_H_seed{seed}'] = result[f'delta_U60_seed{seed}'] = None
    if len(pairs) != expected:
        return result
    deltas_h, deltas_u = [], []
    for seed in seeds:
        selected = [p for p in pairs if p[0] == seed]
        a, b = aggregate([p[2] for p in selected]), aggregate([p[3] for p in selected])
        dh = a['Harrell'] - b['Harrell'] if a['Harrell'] is not None and b['Harrell'] is not None else None
        du = a['Uno60'] - b['Uno60'] if a['Uno60'] is not None and b['Uno60'] is not None else None
        result[f'delta_H_seed{seed}'], result[f'delta_U60_seed{seed}'] = dh, du
        deltas_h.append(dh); deltas_u.append(du)
    result['paired_delta_H'], result['paired_delta_U60'] = all_defined_mean(deltas_h), all_defined_mean(deltas_u)
    result['positive_seed_H'] = sum(v is not None and v > 0 for v in deltas_h)
    result['positive_seed_U60'] = sum(v is not None and v > 0 for v in deltas_u)
    result['positive_partition_seed_H'] = sum(a['Harrell'] is not None and b['Harrell'] is not None
                                               and a['Harrell'] > b['Harrell'] for _, _, a, b in pairs)
    result['positive_partition_seed_U60'] = sum(a['Uno60'] is not None and b['Uno60'] is not None
                                                 and a['Uno60'] > b['Uno60'] for _, _, a, b in pairs)
    # Each source field's reference is retained; these cannot recover the
    # candidate/control pairwise repair intersection without patient risks.
    if stage == 'R3' and candidate not in ('F0', 'F1') and control not in ('F0', 'F1'):
        for name, side in [('candidate', 2), ('control', 3)]:
            by_seed = [aggregate([p[side] for p in pairs if p[0] == seed]) for seed in seeds]
            for metric in ('repair', 'damage'):
                result[f'{name}_{metric}_vs_F1'] = average([r[metric] for r in by_seed])
    return result


def cell_metric_rows(cells):
    output = []
    for (stage, arm, partition, seed), r in sorted(cells.items()):
        output.append({'stage': stage, 'arm': arm, 'inner_partition': partition, 'seed': seed,
                       'status': 'COMPLETE', 'epochs': r['epochs'],
                       'actual_optimizer_fits': r['actual_optimizer_fits'],
                       'parameters': r['parameters'], 'peak_GPU_bytes': r['peak_GPU_bytes'],
                       'elapsed_seconds': r['elapsed_seconds'],
                       'stored_risk_diagnostics_reference': stored_risk_diagnostics_reference(stage, arm),
                       'repair_reference': stored_risk_diagnostics_reference(stage, arm),
                       'valid_delta_H_vs_stored_reference': stored_risk_increment(r['metrics']),
                       'valid_delta_U60_vs_stored_reference': stored_risk_increment(r['metrics'], True),
                       **{key: r['metrics'].get(key) for key in METRICS}})
    return output


def retention_rows(cells, config):
    rows = []
    for arm in planned_arms(config, 'R2'):
        selected = [(key, r) for key, r in sorted(cells.items()) if key[:2] == ('R2', arm)]
        if not selected:
            rows.append({'arm': arm, 'status': 'RUNNING', 'representation': 'NOT_AVAILABLE'})
        for (_, _, partition, seed), receipt in selected:
            retention = receipt.get('retention')
            common = {'arm': arm, 'inner_partition': partition, 'seed': seed,
                      'direct_RNA512_retained': arm.startswith(('O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9')),
                      'source': 'fixed_final_epoch' if not deterministic(arm) else 'convex_fixed_fit'}
            if not retention:
                rows.append({**common, 'status': 'NOT_PROVIDED_REFERENCE', 'representation': 'RNA512_reference'})
            elif 'complete_tokens' in retention or 'pooled32' in retention:
                for representation, value in retention.items():
                    rows.append({**common, 'status': 'COMPLETE', 'representation': representation,
                                 'dimension': value.get('dimension'), 'decoder_Pearson': value.get('Pearson'),
                                 'decoder_R2': value.get('R2'), 'reference': value.get('target'),
                                 'decoder_lambda': value.get('decoder_lambda'),
                                 'decoder_fit_rows': value.get('decoder_fit_rows')})
            else:
                rows.append({**common, 'status': 'COMPLETE' if retention.get('decoder_available') else 'NOT_RUN_DECODER_HIGH_DIMENSION',
                             'representation': arm, 'decoder_Pearson': retention.get('risk_decoder_Pearson'),
                             'decoder_R2': retention.get('risk_decoder_R2'),
                             'prediction_risk_Pearson': retention.get('prediction_risk_Pearson'),
                             'reference': retention.get('reference')})
    return rows


def training_rows(cells):
    rows = []
    for (stage, arm, part, seed), receipt in sorted(cells.items()):
        if stage != 'R3':
            continue
        for point in receipt.get('curves', []):
            rows.append({'stage': stage, 'arm': arm, 'inner_partition': part, 'seed': seed,
                         'epoch': point['epoch'], 'scientific_endpoint': point['epoch'] == receipt['epochs'],
                         'stored_risk_diagnostics_reference': stored_risk_diagnostics_reference(stage, arm),
                         'train_offset_scope': 'current_partition_crossfitted_offset',
                         'valid_offset_scope': 'current_partition_full_training_fit_offset',
                         'train_Harrell': point['train']['Harrell'], 'train_Uno60': point['train']['Uno60'],
                         'valid_Harrell': point['valid']['Harrell'], 'valid_Uno60': point['valid']['Uno60'],
                         'train_delta_H_vs_stored_reference': stored_risk_increment(point['train']),
                         'train_delta_U60_vs_stored_reference': stored_risk_increment(point['train'], True),
                         'valid_delta_H_vs_stored_reference': stored_risk_increment(point['valid']),
                         'valid_delta_U60_vs_stored_reference': stored_risk_increment(point['valid'], True),
                         **{key: point.get(key) for key in DIAGNOSTICS}})
    return rows


def attach_mechanism_replay(pairing, cells, config, path):
    """Optional anonymous saved-risk replay; never infer intersections from C."""
    for row in pairing:
        row.update(direct_replay_status='NOT_RUN', direct_replay_logical_pairs=0,
                   direct_replay_pair_reference='full_trained_checkpoint_vs_frozen_control',
                   full_control_repair=None, full_control_damage=None,
                   full_control_Uno_repair=None, full_control_Uno_damage=None,
                   full_control_risk_shift_sd_ratio_mean=None)
    if path is None or not Path(path).exists():
        return {'status': 'NOT_RUN_NO_ANONYMOUS_REPLAY_FILE'}
    raw_bytes = Path(path).read_bytes()
    replay = json.loads(raw_bytes)
    if replay.get('schema') != 'bgb_en_mechanism_pair_replay_v1' or replay.get('optimizer_steps') != 0:
        raise ValueError('Unrecognized mechanism replay contract')
    by_key = {}
    for item in replay['comparisons']:
        key = item['arm'], item['control'], int(item['seed']), int(item['inner_partition'])
        if key in by_key or item['pair_reference'] != 'full_trained_checkpoint_vs_frozen_control':
            raise ValueError('Duplicate or misreferenced mechanism replay row')
        cell = get_cell(cells, 'R3', key[0], key[3], key[2])
        if cell is None or key[1] not in cell.get('frozen_controls', {}):
            raise ValueError('Replay has no corresponding completed intervention receipt')
        for name, source in [('Harrell_full', cell['metrics']['Harrell']),
                              ('Uno60_full', cell['metrics']['Uno60']),
                              ('Harrell_control', cell['frozen_controls'][key[1]]['Harrell']),
                              ('Uno60_control', cell['frozen_controls'][key[1]]['Uno60'])]:
            if not math.isclose(finite_number(item[name], name), source, abs_tol=1e-10, rel_tol=1e-12):
                raise ValueError('Replay metric does not match fixed-final receipt')
        for name in ('repair_num', 'damage_num', 'Uno_repair_num', 'Uno_damage_num', 'Harrell_den', 'Uno_den'):
            finite_number(item[name], name)
        if abs(item['repair_num'] - item['damage_num'] - item['delta_Harrell'] * item['Harrell_den']) > 1e-9:
            raise ValueError('Replay Harrell repair/damage does not reconcile')
        if abs(item['Uno_repair_num'] - item['Uno_damage_num'] - item['delta_Uno60'] * item['Uno_den']) > 1e-8:
            raise ValueError('Replay Uno repair/damage does not reconcile')
        by_key[key] = item
    seeds, n_partitions = config['prototype']['neural_seeds'], config['prototype']['inner_splits']
    for row in pairing:
        control = row['comparison'].split('frozen:', 1)[1]
        selected = [by_key[row['candidate'], control, seed, partition]
                    for seed in seeds for partition in range(n_partitions)
                    if (row['candidate'], control, seed, partition) in by_key]
        row['direct_replay_logical_pairs'] = len(selected)
        row['direct_replay_status'] = 'COMPLETE' if len(selected) == len(seeds) * n_partitions else 'RUNNING'
        if row['direct_replay_status'] != 'COMPLETE':
            continue
        for name, numerator, denominator in [('repair', 'repair_num', 'Harrell_den'),
                                              ('damage', 'damage_num', 'Harrell_den'),
                                              ('Uno_repair', 'Uno_repair_num', 'Uno_den'),
                                              ('Uno_damage', 'Uno_damage_num', 'Uno_den')]:
            seed_values = []
            for seed in seeds:
                group = [item for item in selected if item['seed'] == seed]
                den = sum(item[denominator] for item in group)
                seed_values.append(sum(item[numerator] for item in group) / den if den else None)
            row['full_control_' + name] = all_defined_mean(seed_values)
        row['full_control_risk_shift_sd_ratio_mean'] = average([r.get('risk_shift_sd_ratio') for r in selected])
        row['direct_candidate_control_repair_damage'] = 'COMPLETE_SAVED_RISK_REPLAY'
        for metric, delta in [('Harrell', 'paired_delta_H'), ('Uno', 'paired_delta_U60')]:
            repair = row['full_control_' + ('repair' if metric == 'Harrell' else 'Uno_repair')]
            damage = row['full_control_' + ('damage' if metric == 'Harrell' else 'Uno_damage')]
            if repair is not None and damage is not None and abs(repair - damage - row[delta]) > 1e-10:
                raise ValueError('Aggregated replay does not reconcile with paired final metrics')
    return {'status': 'LOADED_AVAILABLE_ANONYMOUS_REPLAY', 'sha256': hashlib.sha256(raw_bytes).hexdigest(),
            'source_whole_matrix_complete_flag': replay.get('whole_matrix_complete'),
            'completed_pairing_comparisons': sum(r['direct_replay_status'] == 'COMPLETE' for r in pairing),
            'available_replay_rows': len(by_key)}


def write_csv(path, rows, minimum_fields):
    names = list(dict.fromkeys([*minimum_fields, *(key for row in rows for key in row)]))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def fmt(value, signed=False):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:+.6f}' if signed else f'{value:.6f}'
    return str(value)


def table(rows, columns):
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row.get(key), key.startswith('paired_delta')) for key, _ in columns) + ' |')
    return '\n'.join(lines)


def stage_report(stage, summaries, comparisons, config, complete, planned):
    status = 'COMPLETE' if complete == planned else 'RUNNING'
    rows = [r for r in summaries if r['stage'] == stage]
    content = [f'# BG-BEN {stage} 内层正式原型结果', '',
               f'状态：**{status}**；已完成 {complete}/{planned} 个 unique cell。', '',
               '数据层级：outer fold 1 的训练人群内，3 个固定内层分区 × 2 个 neural seeds（31、47）。'
               '这些是重复开发原型，不是完整5外折确认或独立队列。', '',
               '每个 seed 先相加三个分区的 concordant/eligible 分子分母，再计算 Harrell 和 Uno60；'
               '最后均值两个 seed。绝不拼接跨分区风险作排序。缺少任一规定单元时，该臂平均分数留空。', '',
               table(rows, [('arm', '模型'), ('status', '状态'), ('completed_unique_cells', '完成unique cells'),
                            ('Harrell', 'Harrell'), ('Uno60', 'Uno60'), ('train_minus_valid_Harrell', '训练−验证 H'),
                            ('train_delta_H_vs_stored_reference', '训练ΔH/原参考'),
                            ('valid_delta_H_vs_stored_reference', '验证ΔH/原参考'),
                            ('stored_risk_diagnostics_reference', '保存诊断参考'),
                            ('parameters_mean', '参数量均值')]), '']
    if comparisons:
        content += ['## 预定配对比较', '',
                    table(comparisons, [('comparison', '比较'), ('status', '状态'), ('paired_delta_H', 'ΔHarrell'),
                                        ('paired_delta_U60', 'ΔUno60'), ('positive_seed_H', 'H正向seeds/2'),
                                        ('positive_seed_U60', 'U正向seeds/2'), ('positive_partition_seed_H', 'H正向分区×seed/6')]), '']
    content += ['## 结果解释与开销边界', '',
                '- 20/40轮仅用于学习曲线诊断；正式原型成绩固定取80轮，未根据这些验证点改选checkpoint。',
                '- 保存的risk SD比与repair/damage使用各自原始风险参考，不能用表中paired_reference代替：'
                'P0/O0/O1/O2/F0/F1均参照当前训练分区full LOCAL，即验证集上的F0；'
                'R1神经臂参照CS，R2神经臂参照RNA512/O0，R3神经臂参照F1。'
                'stored_risk_diagnostics_reference列明确记录该来源。',
                '- paired_delta_H/paired_delta_U60由候选与对照的concordance分子分母分别聚合后直接相减；'
                '它们不是原repair/damage字段相减，也不自动等于相对保存诊断参考的增量。',
                '- 学习曲线中训练/验证ΔH=repair−damage、ΔUno60=Uno_repair−Uno_damage，均相对各自保存的参考风险；'
                '训练使用cross-fitted offset，验证使用完整当前训练分区拟合的offset。'
                '原始train−valid H只作描述，train H低于valid H不能据此判定欠拟合。',
                '- P0、O0、O1、各O2、F0、F1是确定性单元，只拟合seed31；seed47使用同一引用，未计为新的优化。',
                '- 每个prepare分区为9个teacher Cox＋4个full Cox＋1个same-teacher β优化，共14个监督优化；原upstream_optimizer_fits字段的13只计Cox。',
                '- 核心186cells全部完成时为12个无新优化baseline引用＋162个神经拟合＋12个新凸Cox拟合；另加3分区42个上游优化，共216个监督优化。解析风险decoder单列，外部ST拟合另见独立收据。',
                '- 时间列为各进程耗时之和，不是并行调度的实际makespan。显存为对应进程记录的PyTorch峰值。',
                '- 小于0.003的改善仅作很弱方向性线索；不生成CI、bootstrap或p值，也不把两个seed称为两个验证队列。', '']
    if stage == 'R1':
        content += ['P0使用完整CONCH均值、train-local PCA32与Clinical+S32联合Ridge；'
                    '神经臂使用相同CS固定风险加病理残差。因此胜负同时涉及病理编码和风险读出/正则形式，'
                    '需要结合已有加性与空间对照解释，不能把P0差值全部归因于二维结构。', '',
                    'R1_FEATURE_INTEGRITY与R0资产/拓扑资格仍提供完整输入边界；本脚本没有重新读取H5。']
    elif stage == 'R2':
        content += ['O3–O9均保留RNA512 direct risk；报告中的新表示增量相对O0判断。'
                    'token decoder只在训练分区拟合，目标为同训练分区RNA512教师，验证目标患者未进入教师。', '',
                    '目前预定核心O8实际调用默认4维非线性读出；2/8维补充比较未运行，不能把配置中的候选列表写成完成。'
                    'O5/O6训练使用多顺序共享扫描，但独立真实重训的顺序敏感性尚未运行，详见R2_ORDER_SENSITIVITY.csv。', '',
                    '信息保真CSV中Pearson/R²为逐cell解码诊断，不是病人合并相关；完整token与pool32需在相同cell直接比较。'
                    'O1高维decoder若未运行会明确标注，不能把缺失R²写为0。']
    else:
        content += ['F5是本项目双向Attention控制，**不是官方SurvPath精确复现**；虽已核验公开源码，'
                    '本轮尚未直接port其原生tokenizer/attention/离散生存头。', '',
                    '真实重训比较与冻结干预分别列出。冻结wrong-query/mean-query/whole-omics/attention-permutation/'
                    'interaction-off/direct-off解释已训模型依赖，不能替代重新训练消融。', '',
                    '神经R3收据中的repair/damage参照F1；冻结控制中的repair/damage也参照F1。'
                    '仅COMPLETE匿名汇总不能重建full模型与控制之间的repair/damage交集。'
                    '若另有经saved-risk独立回放的MECHANISM_REPLAY.json，则full_control_*列给出该直接配对分解；'
                    '缺少时明确NOT_RUN，不把相对F1的分解误当full-control分解。', '',
                    '无已通过资格的局部组织学真值时，不将attention/slot变化命名为医学事件或细胞来源。'
                    '外部ST与文本追加候选未完成时不计入核心矩阵结果。']
    return '\n'.join(content) + '\n'


def generate(private, output, config, mechanism_replay=None):
    cells, lineage = load_cells(private, config)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    summaries = [arm_summary(cells, config, stage, arm)
                 for stage in ('R1', 'R2', 'R3') for arm in planned_arms(config, stage)]
    detail = cell_metric_rows(cells)
    comparisons = {}
    for stage, reference in [('R1', 'P0'), ('R2', 'O0'), ('R3', 'F1')]:
        comparisons[stage] = [paired(cells, config, stage, arm, reference)
                              for arm in planned_arms(config, stage) if arm != reference]
        for row in summaries:
            if row['stage'] != stage:
                continue
            row['paired_reference'] = reference
            comparison = next((r for r in comparisons[stage] if r['candidate'] == row['arm']), None)
            for key in ('paired_delta_H', 'paired_delta_U60', 'positive_seed_H', 'positive_seed_U60',
                        'positive_partition_seed_H', 'positive_partition_seed_U60'):
                row[key] = comparison[key] if comparison is not None else None
    comparisons['R3'] += [paired(cells, config, 'R3', a, b) for a, b in
                          [('F1', 'F0'), ('F5', 'F2'), ('F5', 'F3'), ('F5', 'F4'),
                           ('F6', 'F5'), ('F7', 'F5'), ('F8', 'F5'), ('F9', 'F5'), ('F10', 'F5'),
                           ('F5', 'F5_WRONG_QUERY'), ('F5', 'F5_SHUFFLED_MEMBERSHIP')]]
    pairing = [paired(cells, config, 'R3', arm, arm, frozen=control)
               for arm in planned_arms(config, 'R3') if not deterministic(arm)
               for control in ('wrong_query', 'mean_query', 'whole_omics', 'attention_permutation', 'interaction_off', 'direct_off')]
    replay_audit = attach_mechanism_replay(pairing, cells, config,
                                         mechanism_replay if mechanism_replay is not None else output / 'MECHANISM_REPLAY.json')
    counts = {}
    for stage in ('R1', 'R2', 'R3'):
        selected = [r for r in summaries if r['stage'] == stage]
        complete = sum(r['completed_unique_cells'] for r in selected)
        planned = sum(r['planned_unique_cells'] for r in selected)
        counts[stage] = {'completed_unique_cells': complete, 'planned_unique_cells': planned,
                         'status': 'COMPLETE' if complete == planned else 'RUNNING'}
        (output / f'{stage}_RESULTS.md').write_text(stage_report(stage, summaries, comparisons[stage], config, complete, planned), encoding='utf-8')
    write_csv(output / 'R1_ARCH_ABLATION.csv', [r for r in summaries if r['stage'] == 'R1'], ('arm', 'status'))
    write_csv(output / 'R1_COST.csv', [r for r in detail if r['stage'] == 'R1'], ('arm', 'inner_partition', 'seed'))
    write_csv(output / 'R2_RESULTS.csv', [r for r in summaries if r['stage'] == 'R2'], ('arm', 'status'))
    retention = retention_rows(cells, config)
    write_csv(output / 'R2_RNA_RETENTION.csv', retention, ('arm', 'status', 'representation'))
    write_csv(output / 'R2_ORDER_SENSITIVITY.csv',
              [{'arm': arm, 'status': 'NOT_RUN', 'trained_ensemble_orders': config['R2']['sequence_orders'],
                'independent_order_retraining': False,
                'reason': 'Multisequence training ensemble is not an independent order-sensitivity comparison'}
               for arm in ('O5', 'O6')], ('arm', 'status'))
    write_csv(output / 'R3_MODEL_MATRIX.csv', [r for r in summaries if r['stage'] == 'R3'], ('arm', 'status'))
    write_csv(output / 'R3_MECHANISM_ABLATIONS.csv', comparisons['R3'], ('comparison', 'status'))
    write_csv(output / 'R3_PAIRING_CONTROLS.csv', pairing, ('comparison', 'comparison_type', 'status'))
    write_csv(output / 'R3_TRAINING_DIAGNOSTICS.csv', training_rows(cells), ('arm', 'inner_partition', 'seed', 'epoch'))
    audit = {'status': 'COMPLETE_CORE_INNER' if all(v['status'] == 'COMPLETE' for v in counts.values()) else 'RUNNING',
             'input_contract': 'only cells/**/COMPLETE.json;no patient models,arrays,indices,coordinates,OOF or prepared reads',
             'scientific_epoch': config['training']['epochs'], 'stage_counts': counts,
             'completed_unique_cells': len(cells), 'actual_cell_optimizer_fits': sum(r['actual_optimizer_fits'] for r in cells.values()),
             'upstream_fit_accounting': {
                 'counting_basis': 'frozen_source_contract_per_preparation_partition_with_at_least_one_complete_cell;prepared_receipts_not_opened',
                 'covered_preparation_partitions': len({(key[2]) for key in cells}),
                 'Cox_fits_per_preparation': 13, 'same_teacher_beta_fits_per_preparation': 1,
                 'source_contract_upstream_Cox_fits': 13 * len({key[2] for key in cells}),
                 'source_contract_upstream_beta_fits': len({key[2] for key in cells}),
                 'actual_cell_fits_plus_contract_upstream_fits': sum(r['actual_optimizer_fits'] for r in cells.values()) + 14 * len({key[2] for key in cells}),
                 'analytical_risk_decoder_fits_from_complete_receipts': sum(r['status'] == 'COMPLETE' for r in retention),
                 'external_ST_fits': 'NOT_COUNTED_BY_THIS_CORE_INNER_RECEIPT_READER'},
             'mechanism_replay': replay_audit,
             'stored_risk_diagnostics_reference_rule': {
                 'deterministic_P0_O0_O1_O2_F0_F1': 'F0=current_partition_full_LOCAL_risk',
                 'R1_neural': 'CS', 'R2_neural': 'O0=RNA512', 'R3_neural': 'F1',
                 'stored_fields': 'risk_shift_sd_ratio,repair,damage,Uno_repair,Uno_damage',
                 'paired_delta_source': 'independent_numerator_denominator_ratios_candidate_minus_control',
                 'curve_increment_source': 'repair_minus_damage_on_each_partition_then_counts_aggregate_per_seed',
                 'neural_train_offset': 'crossfitted_current_partition',
                 'neural_validation_offset': 'full_fit_on_current_partition_train',
                 'train_below_validation_is_not_by_itself_underfitting_evidence': True},
             'deterministic_reference_rule': 'seed31 actual unique cell reused for logical seed47;never counted twice',
             'no_outer_test_evaluation': True, 'no_bootstrap_CI_pvalues': True,
             'receipt_sha256': sorted(r['_receipt_sha256'] for r in cells.values()), **lineage}
    (output / 'INNER_REPORT_AUDIT.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--mechanism-replay', type=Path, help='Optional anonymous saved-risk pair repair/damage audit')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    audit = generate(args.private, args.output, config, args.mechanism_replay)
    print(json.dumps({key: audit[key] for key in ('status', 'completed_unique_cells', 'stage_counts')}, indent=2))


if __name__ == '__main__':
    main()

"""CPU-only reporting/audit of the separately locked matched-support controls.

No optimizer, model forward, risk concatenation, endpoint loading or selection
occurs. Artifact deserialization verifies the final saved membership and weight
dimensions, and must not be described as independent risk/metric replay.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import bgb_en_run as driver
from scripts.bgb_en_checkpoint_audit import require, sha256, index_hash, verify_current_source
from scripts.bgb_en_matched_support import SPEC_PATH
from scripts.bgb_en_matched_support_plan import verify_constructor_and_gamma
from scripts.bgb_en_report_inner import aggregate, all_defined_mean, validate_metrics

FAMILIES = {
    'O4': {'stage': 'R2', 'true': 'O4', 'matched': 'O4_MATCHED_SUPPORT', 'old': 'O9', 'gamma': .3},
    'F5': {'stage': 'R3', 'true': 'F5', 'matched': 'F5_MATCHED_SUPPORT',
           'old': 'F5_SHUFFLED_MEMBERSHIP', 'gamma': .1},
}


def identity(row):
    return row['stage'], row['arm'], row['inner_partition'], row['seed']


def verify_receipt(row, *, arm, stage, partition, seed, lineage):
    require(row.get('status') == 'COMPLETE', 'NONCOMPLETE_RECEIPT')
    require(row.get('arm') == arm and row.get('stage') == stage, 'RECEIPT_ARCHITECTURE_MISMATCH')
    require(type(row.get('seed')) is int and row['seed'] == seed, 'RECEIPT_SEED_MISMATCH')
    require(type(row.get('inner_partition')) is int and row['inner_partition'] == partition,
            'RECEIPT_PARTITION_MISMATCH')
    require(row.get('outer_fold') == 1 and row.get('outer_test_evaluated') is False,
            'NOT_A_SEALED_INNER_UNIT')
    require(row.get('lineage') == lineage, 'RECEIPT_LINEAGE_MISMATCH')
    require(row.get('epochs') == 80 and row.get('actual_optimizer_fits') == 1, 'NEURAL_BUDGET_MISMATCH')
    validate_metrics(row['metrics'])
    finals = [p for p in row.get('curves', []) if p.get('epoch') == 80]
    require(len(finals) == 1 and finals[0].get('valid') == row['metrics'], 'FIXED_EPOCH80_METRICS_MISMATCH')
    return row


def inspect_saved_state(state, member):
    """Exact sparse membership plus substantive shape/finite checks, no forward."""
    member = torch.as_tensor(member, dtype=torch.bool).cpu()
    require(isinstance(state, dict) and bool(state), 'MISSING_FINAL_STATE_DICT')
    expected_path, expected_gene = torch.where(member)
    prefix = 'omics.rna.'
    require(torch.equal(state[prefix + 'edge_path'].cpu(), expected_path), 'SAVED_EDGE_PATH_MISMATCH')
    require(torch.equal(state[prefix + 'edge_gene'].cpu(), expected_gene), 'SAVED_EDGE_GENE_MISMATCH')
    require(torch.equal(state[prefix + 'membership_count'].cpu(), member.sum(1)), 'SAVED_MEMBERSHIP_COUNT_MISMATCH')
    edges, pathways = len(expected_gene), member.shape[0]
    shapes = {'edge_weight': (edges, 4), 'project.weight': (32, 5), 'project.bias': (32,),
              'identity': (pathways, 32)}
    for name, shape in shapes.items():
        require(tuple(state[prefix + name].shape) == shape, 'SAVED_WEIGHT_DIMENSION_MISMATCH')
    for value in state.values():
        require(torch.is_tensor(value) and value.device.type == 'cpu', 'STATE_NOT_CPU_TENSOR')
        require(bool(torch.isfinite(value).all()), 'NONFINITE_SAVED_STATE')
    return {
        'state_membership_sha256': index_hash(member.numpy()), 'annotated_edge_count': edges,
        'patterns_per_pathway': 4, 'token_width': 32, 'state_tensors_checked': len(state),
        'all_saved_tensors_finite': True, 'actual_membership_buffers_equal': True,
        'weight_dimensions_valid': True,
        # The anonymous shape map contains module names/shapes only.
        'state_shape_signature': {name: list(value.shape) for name, value in sorted(state.items())},
    }


def inspect_artifact(path, row, *, expected_member=None, train=None, valid=None):
    require(sha256(path) == row['artifact_sha256'], 'MODEL_ARTIFACT_SHA256_MISMATCH')
    value = torch.load(path, map_location='cpu', weights_only=False)
    require(value.get('lineage') == row['lineage'], 'SAVED_MODEL_LINEAGE_MISMATCH')
    for key in ('stage', 'arm', 'seed'):
        require(value.get(key) == row[key], 'SAVED_MODEL_IDENTITY_MISMATCH')
    require(value.get('epochs') == 80 and value.get('optimizer_updates') == 80
            and value.get('full_riskset') is True and value.get('gradient_cache') is True,
            'SAVED_TRAINING_CONTRACT_MISMATCH')
    require(value.get('outer_test_evaluated') is False, 'SAVED_MODEL_OUTER_EVALUATION')
    final = value.get('snapshots', {}).get(80)
    require(final is not None and final.get('metrics') == row['metrics'], 'SAVED_SNAPSHOT_METRICS_MISMATCH')
    if train is not None:
        require(np.array_equal(value.get('train'), train) and np.array_equal(value.get('valid'), valid),
                'SAVED_MODEL_TRAIN_VALID_MISMATCH')
    state = final['state_dict']
    if expected_member is not None:
        audit = inspect_saved_state(state, expected_member)
    else:
        require(all(torch.is_tensor(v) and v.device.type == 'cpu' and torch.isfinite(v).all()
                    for v in state.values()), 'INVALID_REFERENCE_STATE')
        audit = {'state_shape_signature': {k: list(v.shape) for k, v in sorted(state.items())}}
    del value
    return audit


def paired_comparison(candidate, reference, *, parts, seeds, label, family):
    expected = {(p, s) for p in parts for s in seeds}
    common = set(candidate) & set(reference)
    require(set(candidate) <= expected and set(reference) <= expected, 'UNPLANNED_PAIRED_CELL')
    for key in common:
        a, b = candidate[key], reference[key]
        for name in ('Harrell_den', 'Uno_den'):
            require(abs(a['metrics'][name] - b['metrics'][name]) <= 1e-10, 'PAIRED_POPULATION_MISMATCH')
        require(a['train_n'] == b['train_n'] and a['valid_n'] == b['valid_n'], 'PAIRED_SAMPLE_SIZE_MISMATCH')
    result = {'family': family, 'comparison': label, 'status': 'COMPLETE' if common == expected else 'RUNNING',
              'completed_pairs': len(common), 'planned_pairs': len(expected),
              'delta_Harrell': None, 'delta_Uno60': None, 'positive_Harrell_cells': None,
              'positive_Uno60_cells': None, 'seed_summaries': []}
    if common != expected:
        return result
    for seed in seeds:
        a = aggregate([candidate[p, seed]['metrics'] for p in parts])
        b = aggregate([reference[p, seed]['metrics'] for p in parts])
        result['seed_summaries'].append({'seed': seed, 'candidate_Harrell': a['Harrell'],
            'reference_Harrell': b['Harrell'], 'delta_Harrell': a['Harrell'] - b['Harrell'],
            'candidate_Uno60': a['Uno60'], 'reference_Uno60': b['Uno60'],
            'delta_Uno60': a['Uno60'] - b['Uno60'],
            'Harrell_den': a['Harrell_den'], 'Uno_den': a['Uno_den']})
    for metric in ('Harrell', 'Uno60'):
        result['delta_' + metric] = all_defined_mean([r['delta_' + metric] for r in result['seed_summaries']])
        result['positive_' + metric + '_cells'] = sum(candidate[key]['metrics'][metric] > reference[key]['metrics'][metric]
                                                      for key in expected)
    return result


def arm_summary(rows, *, arm, stage, parts, seeds):
    expected = {(p, s) for p in parts for s in seeds}
    result = {'stage': stage, 'arm': arm, 'status': 'COMPLETE' if set(rows) == expected else 'RUNNING',
              'completed_cells': len(rows), 'planned_cells': len(expected), 'Harrell': None, 'Uno60': None,
              'seed_summaries': [], 'registered_parameters': None,
              'stored_risk_diagnostics_reference': 'O0' if stage == 'R2' else 'F1'}
    if set(rows) != expected:
        return result
    for seed in seeds:
        combined = aggregate([rows[p, seed]['metrics'] for p in parts])
        result['seed_summaries'].append({'seed': seed, **{key: combined[key] for key in
            ('Harrell', 'Uno60', 'Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den')}})
    for name in ('Harrell', 'Uno60'):
        result[name] = all_defined_mean([r[name] for r in result['seed_summaries']])
    parameters = {r['parameters'] for r in rows.values()}
    require(len(parameters) == 1, 'ARM_PARAMETER_COUNT_CHANGED_ACROSS_CELLS')
    result['registered_parameters'] = next(iter(parameters))
    return result


def expected_null_lineage(core, lock, source_lock_path, locked_part, variant, base_arm):
    result = copy.deepcopy(core)
    result['matched_support_auxiliary'] = {'source': lock['lineage'], 'null': locked_part,
        'lock_sha256': sha256(source_lock_path), 'variant': variant, 'base_arm': base_arm}
    return result


def csv_write(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=fields); out.writeheader(); out.writerows(rows)


def write_reports(output, report):
    output.mkdir(parents=True, exist_ok=True)
    driver.write(output / 'MATCHED_SUPPORT_RESULTS.json', report)
    csv_write(output / 'MATCHED_SUPPORT_METRICS.csv',
              [{k: v for k, v in r.items() if k != 'seed_summaries'} for r in report['arms']])
    csv_write(output / 'MATCHED_SUPPORT_COMPARISONS.csv',
              [{k: v for k, v in r.items() if k != 'seed_summaries'} for r in report['comparisons']])
    csv_write(output / 'MATCHED_SUPPORT_CHECKPOINT_AUDIT.csv', report['checkpoint_audits'])
    def fmt(x): return '' if x is None else f'{x:+.6f}'
    lines = ['# 保持输入支持集的通路成员打乱对照', '',
        f"状态：**{report['status']}**，新增对照已完成 {report['completed_null_cells']}/12 个拟合。", '',
        '这是发现原全基因置换改变输入支持集后追加的探索性修复，不参与已经锁定的 R4 选择。', '',
        '每个 seed 先将三个分区的同终点分子/分母相加，再取两个 seed 的指标均值；不拼接风险，不做 CI、显著性检验或新的选择。', '',
        '| 比较（左减右） | 状态 | 配对单元 | ΔHarrell | ΔUno60 |', '|---|---|---:|---:|---:|']
    for row in report['comparisons']:
        lines.append(f"| {row['comparison']} | {row['status']} | {row['completed_pairs']}/6 | "
                     f"{fmt(row['delta_Harrell'])} | {fmt(row['delta_Uno60'])} |")
    lines += ['', '## 核验范围', '',
        '- 核验完成收据、80轮固定快照、artifact SHA256、训练谱系、分区/seed、base arm与gamma。',
        '- 已完成的新对照在 CPU 反序列化 final80 state_dict，核实际稀疏成员 buffer 与锁定矩阵一致、参数形状匹配原模型、权重有限。',
        '- **没有 GPU/CPU 模型前向、风险重放、终点加载或优化器调用。**权重结构核验不能替代独立风险与指标重放。',
        '- 所有源代码/选择锁保持不变；缺少任一配对单元时不计算该比较的完整汇总。', '',
        '## 优势与局限', '',
        '**优势：**相对旧 null，新 null 保持活跃基因集及训练测量覆盖；同架构、同预算真实重训。', '',
        '**局限：**这是受训练 mask 分层约束的一种固定打乱、事后内层探索；不能证明医学语义、细胞来源或独立外部泛化。', '',
        '列置换保持基因隶属次数的整体多重集，却不固定每个基因原有的隶属次数。因此它检验固定基因并集下的成员分配整体先验（通路组合及基因复用分配），不能把差异再单独归因于某一个通路名称的语义。', '',
        '**置信度：高（限文件、矩阵与统计对象核验）；机制结论须结合真实结果和独立风险重放。**', '']
    (output / 'MATCHED_SUPPORT_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    source_lock_path = args.private / 'LOCK.json'
    lock = json.loads(source_lock_path.read_text())
    launch = json.loads((args.private / 'LAUNCH_LOCK.json').read_text())
    require(launch['source_lock_sha256'] == sha256(source_lock_path), 'SOURCE_LOCK_HASH_MISMATCH')
    require(launch['private_job_plan_sha256'] == sha256(args.private / 'JOB_PLAN.json'), 'PRIVATE_PLAN_HASH_MISMATCH')
    require(launch['planner_sha256'] == sha256(ROOT / 'scripts/bgb_en_matched_support_plan.py'),
            'FROZEN_PLANNER_SOURCE_CHANGED')
    require(lock['lineage']['specification'] == json.loads(SPEC_PATH.read_text()), 'SPECIFICATION_CHANGED')
    for relative, expected in lock['lineage']['auxiliary_source'].items():
        require(sha256(ROOT / relative) == expected, 'FROZEN_AUXILIARY_SOURCE_CHANGED')
    spec, core = lock['lineage']['specification'], lock['lineage']['core_lineage']
    config = json.loads((ROOT / 'configs/bgb_en/prototype_v1.json').read_text())
    verify_current_source(core, config_path=ROOT / 'configs/bgb_en/prototype_v1.json')
    dispatch = verify_constructor_and_gamma(config)
    selection_paths = [args.inner_report, args.selection_lock]
    selection_before = {p.name: sha256(p) for p in selection_paths}
    require(selection_before == launch['selection_sha256'], 'SELECTION_LOCK_CHANGED')
    parts, seeds = spec['partitions'], spec['seeds']
    locked_parts = {r['inner_partition']: r for r in lock['partitions']}
    require(set(locked_parts) == set(parts), 'LOCK_PARTITION_COVERAGE_MISMATCH')
    expected_aux = {(d['stage'], v, p, s) for v, d in spec['variants'].items() for p in parts for s in seeds}
    expected_core = {(d['stage'], d[role], p, s) for d in FAMILIES.values()
                     for role in ('true', 'old') for p in parts for s in seeds}
    core_rows, aux_rows, audit_rows, state_signatures = {}, {}, [], {}
    # Only the 24 relevant core reference receipts are consumed; other frozen
    # core and recovered F7 receipts cannot contaminate these identities.
    for path in sorted((args.primary_private / 'cells').glob('**/COMPLETE.json')):
        row = json.loads(path.read_text())
        if identity(row) not in expected_core:
            continue
        key = identity(row)
        require(key not in core_rows, 'DUPLICATED_CORE_REFERENCE')
        verify_receipt(row, stage=key[0], arm=key[1], partition=key[2], seed=key[3], lineage=core)
        state = inspect_artifact(path.parent / 'model.pt', row)
        state_signatures[key] = state['state_shape_signature']
        core_rows[key] = row
    nulls, split_indices = {}, {}
    for part in parts:
        locked_part = locked_parts[part]
        path = args.private / 'nulls' / f'partition_{part}.pt'
        require(sha256(path) == locked_part['null_artifact_sha256'], 'NULL_ARTIFACT_HASH_MISMATCH')
        value = torch.load(path, map_location='cpu', weights_only=False)
        require(index_hash(value['membership'].numpy()) == locked_part['actual_membership_sha256']
                and index_hash(value['permutation'].numpy()) == locked_part['actual_permutation_sha256'],
                'ACTUAL_NULL_MEMBERSHIP_OR_PERMUTATION_MISMATCH')
        nulls[part] = value['membership']
        prep_path = args.primary_private / 'prepared' / f'outer1_inner{part}.pt'
        require(sha256(prep_path) == locked_part['prepared_sha256'], 'ORIGINAL_PREPARED_HASH_MISMATCH')
        prepared = torch.load(prep_path, map_location='cpu', weights_only=False)
        split_indices[part] = (prepared['train'], prepared['valid'])
        require(index_hash(prepared['train']) == locked_part['training_indices_sha256'], 'PREPARED_TRAIN_HASH_MISMATCH')
        del prepared, value
    for path in sorted((args.private / 'cells').glob('**/COMPLETE.json')):
        row = json.loads(path.read_text()); key = identity(row)
        require(key in expected_aux and key not in aux_rows, 'UNPLANNED_OR_DUPLICATE_NULL_UNIT')
        stage, variant, part, seed = key
        definition = spec['variants'][variant]
        expected = expected_null_lineage(core, lock, source_lock_path, locked_parts[part], variant, definition['base_arm'])
        verify_receipt(row, stage=stage, arm=variant, partition=part, seed=seed, lineage=expected)
        audited = inspect_artifact(path.parent / 'model.pt', row, expected_member=nulls[part],
                                   train=split_indices[part][0], valid=split_indices[part][1])
        reference_key = stage, definition['base_arm'], part, seed
        require(reference_key in core_rows, 'MISSING_TRUE_REFERENCE_FOR_COMPLETED_NULL')
        require(audited.pop('state_shape_signature') == state_signatures[reference_key],
                'NULL_AND_TRUE_WEIGHT_DIMENSIONS_DIFFER')
        require(row['parameters'] == core_rows[reference_key]['parameters'], 'NULL_AND_TRUE_PARAMETER_COUNTS_DIFFER')
        require(audited['state_membership_sha256'] == locked_parts[part]['actual_membership_sha256'],
                'STATE_MEMBERSHIP_HASH_MISMATCH')
        audit_rows.append({'stage': stage, 'variant': variant, 'inner_partition': part, 'seed': seed,
            'base_arm': definition['base_arm'], 'gamma': config[stage]['gamma'],
            'artifact_sha256': row['artifact_sha256'], 'receipt_sha256': sha256(path),
            'registered_parameters': row['parameters'], 'membership_and_dimensions_status': 'PASS',
            **audited, 'forward_risk_replayed': False, 'optimizer_calls': 0, 'GPU_calls': 0})
        aux_rows[key] = row
    arms, comparisons = [], []
    for family, definition in FAMILIES.items():
        rows = {}
        for role in ('true', 'matched', 'old'):
            arm = definition[role]; source_rows = aux_rows if role == 'matched' else core_rows
            rows[role] = {(p, s): source_rows[definition['stage'], arm, p, s] for p in parts for s in seeds
                          if (definition['stage'], arm, p, s) in source_rows}
            arms.append(arm_summary(rows[role], arm=arm, stage=definition['stage'], parts=parts, seeds=seeds))
        for candidate, reference in [('true', 'matched'), ('true', 'old'), ('matched', 'old')]:
            comparisons.append(paired_comparison(rows[candidate], rows[reference], parts=parts, seeds=seeds,
                               label=f"{definition[candidate]} - {definition[reference]}", family=family))
    selection_after = {p.name: sha256(p) for p in selection_paths}
    require(selection_before == selection_after, 'SELECTION_CHANGED_DURING_REPORT')
    complete = len(aux_rows) == 12 and len(core_rows) == 24
    result = {'schema': 'bgb_en_matched_support_report_v1', 'status': 'COMPLETE' if complete else 'RUNNING',
        'completed_null_cells': len(aux_rows), 'planned_null_cells': 12,
        'completed_core_reference_cells': len(core_rows), 'planned_core_reference_cells': 24,
        'source_lock_sha256': sha256(source_lock_path), 'launch_lock_sha256': sha256(args.private / 'LAUNCH_LOCK.json'),
        'reporter_source_sha256': sha256(__file__), 'constructor_gamma_audit': dispatch,
        'report_source_closure': {name: sha256(ROOT / name) for name in (
            'scripts/bgb_en_report_matched_support.py', 'scripts/bgb_en_report_inner.py',
            'scripts/bgb_en_checkpoint_audit.py', 'scripts/bgb_en_matched_support_plan.py')},
        'selection_sha256_before': selection_before, 'selection_sha256_after': selection_after,
        'arms': arms, 'comparisons': comparisons, 'checkpoint_audits': audit_rows,
        'actual_report_optimizer_calls': 0, 'actual_report_model_forwards': 0, 'GPU_calls': 0,
        'artifact_deserialization_not_risk_replay': True, 'patient_or_gene_indices_exported': False,
        'risk_vectors_consumed': False, 'endpoints_loaded': False, 'outer_performance_read': False,
        'R4_selection_modified': False, 'post_hoc_exploratory': True,
        'nulls_audited_by_actual_final80_weights': len(audit_rows),
        'planned_new_neural_fits': 12, 'completed_new_neural_fits': len(aux_rows),
        'teacher_refits': 0, 'analytical_RNA_decoder_fits': sum(r['stage'] == 'R2' for r in aux_rows.values()) * 2,
        'aggregation': 'sum_3partition_numerators_and_denominators_within_seed_then_mean_2seeds',
        'null_preserves_column_degree_multiset': True,
        'null_preserves_each_fixed_gene_degree': False,
        'null_interpretation': 'Membership-assignment prior within fixed gene union, including pathway composition and gene reuse; not isolated pathway-name semantics.',
        'confidence_intervals': False, 'cross_partition_risk_comparisons': False}
    write_reports(args.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private', required=True, type=Path)
    p.add_argument('--primary-private', required=True, type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en')
    p.add_argument('--inner-report', type=Path, default=ROOT / 'reports/bgb_en/INNER_REPORT_FOR_SELECTION.json')
    p.add_argument('--selection-lock', type=Path, default=ROOT / 'reports/bgb_en/OUTER_SELECTION_LOCK.json')
    args = p.parse_args(); torch.set_num_threads(2)
    result = run(args)
    print(json.dumps({'status': result['status'], 'completed_null_cells': result['completed_null_cells'],
                      'optimizer_calls': 0, 'model_forwards': 0, 'GPU_calls': 0}))


if __name__ == '__main__':
    main()

"""Hash-bound saved parameter budgets; CPU tensor reads, zero forwards/fits.

Registered nn.Parameter count, requires_grad eligibility, observed gradient and
shared upstream coefficient dimensions are different quantities. This sidebar
does not rewrite historical receipts or any parameter tie-break rule.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_recovery_audit import matrix_inventory, sha256, require, verify_runtime_receipt

BUFFER_NAMES = {'edge_path', 'edge_gene', 'membership_count', 'orders'}
FROZEN_PREFIXES = ('path.mil.', 'path.output.', 'omics.risk_head.', 'omics.pool_query')


def state_parameter_count(state):
    """Exact for the frozen source closure: explicitly remove known buffers."""
    import torch
    total, excluded = 0, 0
    for name, tensor in state.items():
        require(torch.is_tensor(tensor), 'NON_TENSOR_IN_STATE_DICT')
        if name.rsplit('.', 1)[-1] in BUFFER_NAMES:
            excluded += tensor.numel()
        else:
            require(torch.is_floating_point(tensor) or torch.is_complex(tensor), 'UNKNOWN_INTEGER_BUFFER_NOT_SILENTLY_COUNTED_AS_PARAMETER')
            total += tensor.numel()
    frozen = {prefix: sum(tensor.numel() for name, tensor in state.items() if name.startswith(prefix))
              for prefix in FROZEN_PREFIXES}
    return int(total), int(excluded), {key: int(value) for key, value in frozen.items()}


def neural_counts(artifact, receipt):
    total = int(artifact['parameters']); active = int(artifact['active_parameters'])
    require(total == receipt['parameters'] and 0 < active <= total, 'SAVED_REGISTERED_ACTIVE_PARAMETER_COUNTS_DIFFER')
    state = artifact['snapshots'][80]['state_dict']
    rebuilt, buffers, frozen = state_parameter_count(state)
    require(rebuilt == total, 'REGISTERED_COUNT_DIFFERS_FROM_SAVED_STATE_PARAMETER_ELEMENTS')
    expected_frozen = sum(frozen.values()) if receipt['stage'] == 'R3' else 0
    require(total-active == expected_frozen, 'SAVED_REQUIRES_GRAD_COUNT_DIFFERS_FROM_SOURCE_FROZEN_MODULES')
    return dict(count_kind='neural_registered_and_requires_grad_eligible', total_registered_parameters=total,
        active_parameters=active, frozen_parameters=total-active, incremental_optimized_parameter_count=active,
        convex_coefficient_dimension=None, shared_reference_reported_dimension=None,
        excluded_registered_buffer_elements=buffers, source_frozen_module_counts=frozen if receipt['stage'] == 'R3' else {},
        active_count_source='actual saved model.pt.active_parameters from run_unit sum(p.numel() for p if p.requires_grad)',
        registered_count_independently_recounted_from_state=True,
        active_is_not_observed_gradient_activity=True)


def tie_break_proof(rows, candidates):
    by_identity = {(row['stage'], row['arm'], row['inner_partition'], row['seed']): row for row in rows}
    records = []
    for part in (0, 1, 2):
        for seed in (31, 47):
            selected = [by_identity['R3', arm, part, seed] for arm in candidates]
            differences = {row['total_registered_parameters']-row['active_parameters'] for row in selected}
            require(len(differences) == 1, 'FOUR_CANDIDATES_HAVE_DIFFERENT_FROZEN_BUDGETS')
            pairs = []
            for left, right in itertools.combinations(selected, 2):
                old = left['total_registered_parameters']-right['total_registered_parameters']
                new = left['active_parameters']-right['active_parameters']
                require(old == new, 'REGISTERED_ACTIVE_PARAMETER_TIE_ORDER_CHANGED')
                pairs.append(dict(left_arm=left['arm'], right_arm=right['arm'], registered_difference=old, active_difference=new))
            records.append(dict(inner_partition=part, seed=seed, constant_frozen_difference=differences.pop(),
                                all_parameter_pair_orderings_unchanged=True, pairs=pairs))
    return dict(candidates=candidates, six_prototype_partitions_seed_combinations=records,
        constant_frozen_difference_across_all_candidates_and_cells=len({row['constant_frozen_difference'] for row in records}) == 1,
        frozen_difference_values=sorted({row['constant_frozen_difference'] for row in records}),
        pairwise_comparisons_verified=sum(len(row['pairs']) for row in records),
        nested_selection_or_tiebreak_modified=False, outer_scores_read=False,
        boundary='same constant subtraction preserves parameter differences and exact ties for these four architectures; no performance key or existing nested choice was recomputed')


def run(private, config_path):
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU_AUDIT_MUST_START_WITHOUT_CUDA_CONTEXT')
    config = json.loads(Path(config_path).read_text())
    matrix = matrix_inventory(private, config)
    rows, lineage = [], None
    for identity, (receipt_path, receipt) in sorted(matrix.items()):
        artifact_path = receipt_path.parent/'model.pt'
        if lineage is None:
            lineage = receipt['lineage']
            require(lineage['protocol_sha256'] == sha256(config_path), 'PARAMETER_PROTOCOL_SHA_DIFFERS')
            for name, digest in lineage['source'].items():
                relative = Path(name)
                require(not relative.is_absolute() and '..' not in relative.parts and sha256(ROOT/relative) == digest,
                        'PARAMETER_CORE_SOURCE_SHA_DIFFERS')
        require(receipt['lineage'] == lineage, 'PARAMETER_MATRIX_MIXES_LINEAGES')
        # matrix_inventory verified each artifact SHA before any unpickling.
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=False)
        require(artifact['lineage'] == lineage and all(artifact[key] == receipt[key] for key in ('stage', 'arm', 'seed')),
                'PARAMETER_ARTIFACT_LINEAGE_IDENTITY_DIFFERS')
        require(artifact['parameters'] == receipt['parameters'], 'PARAMETER_ARTIFACT_RECEIPT_COUNT_DIFFERS')
        base = dict(stage=receipt['stage'], arm=receipt['arm'], outer_fold=receipt['outer_fold'],
            inner_partition=receipt['inner_partition'], seed=receipt['seed'],
            legacy_COMPLETE_parameters=receipt['parameters'], artifact_sha256=receipt['artifact_sha256'],
            complete_sha256=sha256(receipt_path), shared_teacher_parameters_excluded_from_incremental_count=True,
            cached_foundation_extractors_excluded=True)
        if receipt['stage'] == 'R3' and receipt['arm'] == 'F7':
            runtime = verify_runtime_receipt(receipt_path.parent, receipt)
            base['runtime_receipt_sha256'] = runtime['runtime_receipt_sha256']
            base['runtime_backend'] = runtime['backend']['mode']
        else:
            base['runtime_backend'] = 'inherited_core_backend'
        if 'snapshots' in artifact:
            require(artifact['epochs'] == 80 and 80 in artifact['snapshots'], 'PARAMETER_NEURAL_FINAL80_MISSING')
            base.update(neural_counts(artifact, receipt))
        elif 'coefficient' in artifact:
            dimension = int(artifact['coefficient'].shape[0])
            require(dimension == artifact['parameters'], 'CONVEX_COEFFICIENT_DIMENSION_DIFFERS')
            base.update(count_kind='new_convex_Cox_coefficient_dimension_not_neural_parameters',
                total_registered_parameters=None, active_parameters=None, frozen_parameters=None,
                incremental_optimized_parameter_count=dimension, convex_coefficient_dimension=dimension,
                shared_reference_reported_dimension=None, active_is_not_observed_gradient_activity=True)
        else:
            require(artifact.get('reference_to_new_upstream_anchor') is True and artifact['actual_optimizer_fits'] == 0,
                    'UNKNOWN_PARAMETER_ARTIFACT_KIND')
            base.update(count_kind='shared_upstream_reference_dimension_zero_new_cell_parameters',
                total_registered_parameters=None, active_parameters=None, frozen_parameters=None,
                incremental_optimized_parameter_count=0, convex_coefficient_dimension=None,
                shared_reference_reported_dimension=int(artifact['parameters']), active_is_not_observed_gradient_activity=True)
        rows.append(base)
        del artifact
    proof = tie_break_proof(rows, config['selection']['nested_interaction_candidates'])
    groups = []
    for stage, arm in sorted({(row['stage'], row['arm']) for row in rows}):
        values = [row for row in rows if row['stage'] == stage and row['arm'] == arm]
        group = dict(stage=stage, arm=arm, cells=len(values), count_kind=values[0]['count_kind'])
        for name in ('legacy_COMPLETE_parameters', 'total_registered_parameters', 'active_parameters', 'frozen_parameters',
                     'incremental_optimized_parameter_count', 'convex_coefficient_dimension', 'shared_reference_reported_dimension'):
            available = [row[name] for row in values if row[name] is not None]
            group[name+'_range'] = [min(available), max(available)] if available else None
        groups.append(group)
    final_source = ROOT/'scripts/bgb_en_final_neural.py'
    expression = 'parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)'
    require(expression in final_source.read_text(), 'R4_FINAL_NEURAL_ACTIVE_COUNT_SOURCE_CHANGED')
    require(not torch.cuda.is_initialized(), 'PARAMETER_AUDIT_INITIALIZED_CUDA')
    return dict(schema='bgb_en_parameter_budget_sidebar_v1', status='COMPLETE_186_SAVED_PARAMETER_AUDIT',
        audited_cells=len(rows), neural_cells=sum(row['count_kind'].startswith('neural_') for row in rows),
        forward_calls=0, backward_calls=0, optimizer_fits=0, CUDA_context_initialized=False,
        audit_source_sha256=sha256(__file__), config_sha256=sha256(config_path),
        core_lineage_sha256=hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest(),
        rows=rows, stage_arm_summary=groups, nested_parameter_tiebreak_proof=proof,
        old_core_parameter_field='registered nn.Parameter elements for neural cells; convex/reference rows use their separately labeled dimensions',
        active_parameter_boundary='requires_grad eligibility saved by the actual trained artifact; not empirical nonzero gradients or effective functional degrees of freedom',
        shared_teacher_boundary='CP/CS/RNA Cox teachers, beta and train-fold preprocessing are shared upstream costs and excluded from incremental cell counts; reference dimensions are labeled, not counted as new fits',
        foundation_boundary='offline UNI2/CONCH/TITAN extractors are not registered in these heads; cached features are inputs',
        R4_final_neural_boundary=dict(source='scripts/bgb_en_final_neural.py', sha256=sha256(final_source),
            expression_statically_verified=expression, parameter_field_is_active_requires_grad_count=True,
            R4_artifacts_or_scores_read=False, warning='R4 active parameter values must not be compared to legacy neural COMPLETE registered counts without this distinction'),
        historical_COMPLETE_files_or_configuration_modified=False)


def write_report(report, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'PARAMETER_BUDGET_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    scalar = [{key: value for key, value in row.items() if not isinstance(value, dict)} for row in report['rows']]
    fields = list(dict.fromkeys(key for row in scalar for key in row))
    with (output_dir/'PARAMETER_BUDGET_AUDIT.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(scalar)
    lines = ['# 参数预算口径审计', '',
        '**旧神经COMPLETE.parameters是registered总量；model.pt.active_parameters是requires_grad资格量，两者都不是实测梯度活跃度。**', '',
        f"已按source/SHA核对{report['audited_cells']}个实际产物，其中神经单元{report['neural_cells']}个；CPU读取，前向/反传/新拟合均0。未改旧收据、配置或nested选择。", '',
        '|stage|arm|cells|registered总量范围|requires_grad范围|冻结范围|新增优化参数范围|', '|---|---|---:|---|---|---|---|']
    for row in report['stage_arm_summary']:
        def value(name):
            result = row[name+'_range']
            return 'NA：见独立coef/reference口径' if result is None else str(result[0]) if result[0] == result[1] else f'{result[0]}–{result[1]}'
        lines.append(f"|{row['stage']}|{row['arm']}|{row['cells']}|{value('total_registered_parameters')}|{value('active_parameters')}|{value('frozen_parameters')}|{value('incremental_optimized_parameter_count')}|")
    proof = report['nested_parameter_tiebreak_proof']
    lines += ['', '## 四候选参数tie-break', '',
        f"锁定候选{', '.join(proof['candidates'])}的registered−active差值为{proof['frozen_difference_values']}；逐partition/seed共{proof['pairwise_comparisons_verified']}个候选两两差值保持完全一致。因此该常数扣除不改变参数大小关系或精确平局；现有选择规则及结果均未重算或修改。", '',
        'R3冻结部分为path.mil、path.output、omics.risk_head和omics.pool_query。它们属于registered总量但不进入Adam的requires_grad参数集合。其存在不代表对应路径有非零梯度；本报告没有逐参数反传。', '',
        '## 凸模型、引用与共享成本', '',
        '新拟合凸Cox只列coefficient维数；P0/O0/F0/F1仅引用共享上游风险，单列参考维数，新增单元优化参数为0。PCA等训练折预处理数组、共同教师Cox和beta、离线基础特征提取器均不混入新增神经参数预算。逐seed/partition及来源SHA见CSV/JSON。', '',
        'R4 final_neural现源码使用sum(requires_grad)作为parameters字段；这是active口径，与旧core registered主表不同。本审计只核对该源码表达式，没有读取R4外折分数或改动参数tie-break。', '']
    (output_dir/'PARAMETER_BUDGET_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/bgb_en')
    args = parser.parse_args()
    result = run(args.private, args.config); write_report(result, args.output_dir)
    print(json.dumps({key: result[key] for key in ('status', 'audited_cells', 'neural_cells', 'forward_calls', 'optimizer_fits')}))


if __name__ == '__main__':
    main()

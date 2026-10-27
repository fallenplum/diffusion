"""CPU-only post-observation PCA penalty-geometry diagnostic; no core changes.

Reuse the exact three saved PCA bases at 32/128/256 dimensions. Fit nine Cox
models to unwhitened PC scores with lambda=1. In an orthonormal retained
subspace this is gene-coordinate L2, unlike uniform L2 after per-PC scaling.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_checkpoint_audit import (
    require, sha256, index_hash, verify_current_source, verify_preprocessing_state,
    verify_replayed_risk, metrics_from_risk, verify_metric_fields,
)
from src.bgb_en.survival import fit_offset_ridge


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pca_coordinates(rna, state, train, dimension):
    """Apply the exact saved PCA and audit its old post-scaling, with no fit."""
    x, fit = np.asarray(rna, float), np.asarray(train, int)
    v, center = np.asarray(state['components'], float), np.asarray(state['mean'], float)
    require(x.ndim == 2 and v.shape == (dimension, x.shape[1]) and center.shape == (x.shape[1],),
            'SAVED_PCA_DIMENSION_MISMATCH')
    require(np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(center).all(), 'NONFINITE_SAVED_PCA')
    orthogonal_error = float(np.max(np.abs(v @ v.T - np.eye(dimension))))
    require(orthogonal_error <= 1e-10, 'SAVED_PCA_COMPONENTS_NOT_ORTHONORMAL')
    require(float(np.max(np.abs(x[fit].mean(0) - center))) <= 1e-10, 'SAVED_PCA_CENTER_NOT_TRAIN_MEAN')
    score = (x - center) @ v.T
    post_mean, post_scale = np.asarray(state['post_mean']), np.asarray(state['post_scale'])
    require(post_mean.shape == post_scale.shape == (dimension,) and (post_scale > 0).all(), 'INVALID_SAVED_PC_SCALING')
    mean_error = float(np.max(np.abs(score[fit].mean(0) - post_mean)))
    expected_sd = score[fit].std(0); expected_sd = np.where(expected_sd > 1e-8, expected_sd, 1.)
    scale_error = float(np.max(np.abs(expected_sd - post_scale)))
    require(mean_error <= 1e-10 and scale_error <= 1e-10, 'SAVED_PC_SCALING_NOT_TRAIN_LOCAL')
    old_score = (score - post_mean) / post_scale
    return score, old_score, dict(orthonormality_max_error=orthogonal_error,
                                  saved_post_mean_error=mean_error, saved_post_scale_error=scale_error,
                                  lowest_PC_train_SD=float(post_scale.min()), highest_PC_train_SD=float(post_scale.max()),
                                  old_gene_L2_directional_penalty_ratio=float((post_scale.max() / post_scale.min())**2))


def own_training_standardize(risk, train):
    x, fit = np.asarray(risk, float), np.asarray(train, int)
    mean, sd = float(x[fit].mean()), float(x[fit].std())
    require(np.isfinite(x).all() and sd > 1e-8, 'DEGENERATE_RISK_SCALE')
    return (x - mean) / sd, dict(mean=mean, sd=sd, moments_from='own_current_training_partition_only')


def source_contract(args):
    from scripts.bgb_en_run import load_inputs
    from src.bgb_en.protocol import fixed_inner_splits
    _, core_config, raw, _, lineage = load_inputs(args)
    verify_current_source(lineage, config_path=args.config)
    spec = json.loads(args.geometry_config.read_text(encoding='utf-8'))
    require(spec['dimensions'] == [32, 128, 256] and spec['inner_partitions'] == [0, 1, 2]
            and spec['new_Cox_fits'] == 9 and spec['lambda'] == 1. and spec['seed'] == 31,
            'GEOMETRY_SPEC_OUTSIDE_FIXED_NINE_FIT_CONTRACT')
    require(spec['PCA_refits'] == spec['teacher_refits'] == spec['upstream_refits'] == 0
            and spec['outer_test_evaluated'] is False and spec['automatically_changes_R4_finalists'] is False,
            'GEOMETRY_SPEC_CHANGED_SCOPE')
    auxiliary, core = args.private.resolve(), args.core_private.resolve()
    require(auxiliary != core and not auxiliary.is_relative_to(core) and not core.is_relative_to(auxiliary)
            and not auxiliary.is_relative_to((ROOT / 'reports').resolve()), 'PRIVATE_DIAGNOSTIC_DIRECTORY_NOT_ISOLATED')
    inputs = {}; prepared_cache = {}
    for part in spec['inner_partitions']:
        stem = f'outer1_inner{part}'
        path, receipt_path = args.core_private / 'prepared' / (stem + '.pt'), args.core_private / 'prepared' / (stem + '.json')
        receipt = json.loads(receipt_path.read_text())
        require(receipt['lineage'] == lineage and receipt['sha256'] == sha256(path), 'GEOMETRY_PREPARED_LINEAGE_OR_HASH_CHANGED')
        prepared = torch.load(path, map_location='cpu', weights_only=False)
        train, valid = fixed_inner_splits(raw['folds'], raw['e'], outer=1)[part]
        require(prepared['lineage'] == lineage and np.array_equal(prepared['train'], train)
                and np.array_equal(prepared['valid'], valid), 'GEOMETRY_PREPARED_SPLIT_CHANGED')
        require(prepared['fit_hash'] == index_hash(train) and prepared['valid_hash'] == index_hash(valid), 'GEOMETRY_PREPARED_FIT_HASH_CHANGED')
        verify_preprocessing_state(prepared['features']['state'], train)
        prepared_cache[part] = prepared
        inputs[f'prepared_p{part}'] = dict(artifact_sha256=sha256(path), receipt_sha256=sha256(receipt_path))
        for arm in ['O0'] + [f'O2_PCA{dim}' for dim in spec['dimensions']]:
            directory = args.core_private / 'cells' / f'R2_outer1_inner{part}_{arm}_seed31'
            r = json.loads((directory / 'COMPLETE.json').read_text()); artifact = directory / 'model.pt'
            wanted = dict(status='COMPLETE', stage='R2', arm=arm, outer_fold=1, inner_partition=part,
                          seed=31, epochs=0, outer_test_evaluated=False)
            require(all(r.get(k) == v for k, v in wanted.items()) and r['lineage'] == lineage,
                    'ORIGINAL_PCA_CELL_IDENTITY_OR_LINEAGE_CHANGED')
            require(r['artifact_sha256'] == sha256(artifact), 'ORIGINAL_PCA_ARTIFACT_HASH_CHANGED')
            inputs[f'{arm}_p{part}'] = dict(artifact_sha256=sha256(artifact), receipt_sha256=sha256(directory / 'COMPLETE.json'))
            if arm == 'O0':
                original = torch.load(artifact, map_location='cpu', weights_only=False)
                require(original['lineage'] == lineage and np.array_equal(original['train'], train)
                        and np.array_equal(original['valid'], valid) and original['arm'] == 'O0', 'ORIGINAL_O0_MODEL_SOURCE_CHANGED')
                verify_replayed_risk(original['risk'], prepared['RNA'][valid], len(valid), 2e-9)
                metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid],
                                          original['risk'], prepared['full_risks']['LOCAL'][valid])
                verify_metric_fields(metric, r['metrics'])
    contract = dict(schema='bgb_en_pca_geometry_source_lock_v1', specification=spec,
        geometry_config_sha256=sha256(args.geometry_config), core_config_sha256=sha256(args.config),
        assets_config_sha256=sha256(args.assets), core_lineage=lineage, original_inputs=inputs,
        source_sha256={name:sha256(ROOT / name) for name in ('scripts/bgb_en_pca_geometry.py',
            'scripts/bgb_en_checkpoint_audit.py', 'src/phase10_trimodal/reference.py')})
    return contract, prepared_cache, raw


def enforce_lock(args, contract, create=False):
    path = args.private / 'LOCK.json'
    if path.exists():
        old = json.loads(path.read_text())
        require(old.get('status') == 'LOCKED_BEFORE_NINE_EXPLORATORY_FITS' and old['contract'] == contract,
                'GEOMETRY_SOURCE_OR_CONFIG_LOCK_CHANGED')
        return old
    require(create, 'GEOMETRY_PRE_FIT_LOCK_MISSING')
    require(not (args.private / 'cells').exists(), 'GEOMETRY_CELLS_EXIST_BEFORE_LOCK')
    args.private.mkdir(parents=True, exist_ok=True)
    value = dict(status='LOCKED_BEFORE_NINE_EXPLORATORY_FITS', created_utc=datetime.now(timezone.utc).isoformat(),
                 contract=contract, optimizer_fits_before_lock=0)
    with path.open('x', encoding='utf-8') as handle: json.dump(value, handle, indent=2, allow_nan=False)
    return value


def load_original(args, prepared, raw, part, dimension, contract):
    arm = f'O2_PCA{dimension}'; directory = args.core_private / 'cells' / f'R2_outer1_inner{part}_{arm}_seed31'
    artifact = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
    receipt = json.loads((directory / 'COMPLETE.json').read_text())
    require(artifact['lineage'] == contract['core_lineage'] and artifact['stage'] == 'R2'
            and artifact['arm'] == arm and artifact['seed'] == 31, 'ORIGINAL_PCA_MODEL_IDENTITY_CHANGED')
    train, valid = prepared['train'], prepared['valid']
    require(np.array_equal(artifact['train'], train) and np.array_equal(artifact['valid'], valid), 'ORIGINAL_PCA_MODEL_SPLIT_CHANGED')
    score, old_score, math_audit = pca_coordinates(prepared['features']['R'], artifact['feature_state'], train, dimension)
    old_all = old_score @ artifact['coefficient']
    verify_replayed_risk(artifact['risk'], old_all[valid], len(valid), 2e-9)
    old_metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid],
                                   artifact['risk'], prepared['full_risks']['LOCAL'][valid])
    verify_metric_fields(old_metric, receipt['metrics'])
    return score, old_all, artifact, receipt, math_audit


def fit_unit(args, contract, prepared, raw, part, dimension):
    label = f'PCA{dimension}_UNWHITENED_p{part}'; directory = args.private / 'cells' / label
    done, artifact_path = directory / 'COMPLETE.json', directory / 'model.pt'
    lock_sha = sha256(args.private / 'LOCK.json')
    identity = dict(lock_sha256=lock_sha, partition=part, dimension=dimension, source_contract_sha256=canonical(contract))
    if done.exists():
        old = json.loads(done.read_text())
        require(args.resume and old['identity'] == identity and old['artifact_sha256'] == sha256(artifact_path), 'GEOMETRY_COMPLETED_UNIT_CHANGED')
        return dict(partition=part, dimension=dimension, status='REUSED_VERIFIED_COMPLETE', new_optimizer_fits=0)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'START.json').open('x') as handle: json.dump(identity, handle)
    started = time.monotonic(); train, valid = prepared['train'], prepared['valid']
    score, old_risk, original, original_receipt, math_audit = load_original(args, prepared, raw, part, dimension, contract)
    coefficient, optimizer = fit_offset_ridge(score[train], raw['t'][train], raw['e'][train], 1.)
    risk = score @ coefficient
    gene_coefficient = np.asarray(original['feature_state']['components']).T @ coefficient
    norm_error = abs(float(coefficient @ coefficient) - float(gene_coefficient @ gene_coefficient))
    require(norm_error <= 1e-10, 'ORTHOGONAL_GENE_L2_EQUIVALENCE_FAILED')
    # Primary sorting remains on the original, unscaled risk vectors. Only
    # risk-amplitude diagnostics use each model's own training mean and SD.
    new_z, new_scale = own_training_standardize(risk, train)
    old_z, old_scale = own_training_standardize(old_risk, train)
    o0_raw = prepared['features']['R'] @ prepared['full_models']['R']['coef']
    o0_z, o0_scale = own_training_standardize(o0_raw, train)
    verify_replayed_risk(prepared['RNA'][valid], o0_z[valid], len(valid), 2e-9)
    new_metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], risk[valid], prepared['RNA'][valid])
    vs_old = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], risk[valid], original['risk'])
    o0_metric = metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][valid], raw['e'][valid], prepared['RNA'][valid])
    diagnostics = dict(standardization='separate_own_training_mean_SD_before_amplitude_comparison',
        new_vs_O0_valid_SD_difference=float(np.std(new_z[valid] - o0_z[valid])),
        old_vs_O0_valid_SD_difference=float(np.std(old_z[valid] - o0_z[valid])),
        new_vs_old_valid_SD_difference=float(np.std(new_z[valid] - old_z[valid])),
        new_vs_O0_valid_Pearson=float(np.corrcoef(new_z[valid], o0_z[valid])[0, 1]),
        old_vs_O0_valid_Pearson=float(np.corrcoef(old_z[valid], o0_z[valid])[0, 1]),
        new_scale=new_scale, old_scale=old_scale, O0_scale=o0_scale)
    result = dict(identity=identity, train=train, valid=valid, coefficient=coefficient, gene_coefficient=gene_coefficient,
                  risk=risk[valid], risk_train=risk[train], optimizer=optimizer, metrics=new_metric,
                  comparisons=dict(O2_original=vs_old), diagnostics=diagnostics,
                  saved_PCA_source=original['feature_state'], PCA_refits=0, teacher_refits=0, actual_optimizer_fits=1)
    torch.save(result, artifact_path)
    receipt = dict(status='COMPLETE', identity=identity, artifact_sha256=sha256(artifact_path),
        actual_optimizer_fits=1, actual_PCA_refits=0, actual_teacher_refits=0, GPU_executions=0,
        metrics=new_metric, original_O2_metrics=original_receipt['metrics'], O0_metrics=o0_metric,
        new_vs_original_repair_damage=vs_old, original_RNA_retention=original_receipt.get('retention'),
        diagnostics=diagnostics, PCA_math_audit=dict(**math_audit, coefficient_gene_L2_norm_error=norm_error),
        parameters=dimension, optimizer=optimizer, elapsed_seconds=time.monotonic() - started,
        outer_test_evaluated=False, changes_R4_finalists=False)
    write(done, receipt)
    return dict(partition=part, dimension=dimension, status='COMPLETE', new_optimizer_fits=1,
                Harrell=new_metric['Harrell'], Uno60=new_metric['Uno60'])


def aggregate(metrics):
    total = {name: float(sum(row[name] for row in metrics)) for name in ('Harrell_num', 'Harrell_den', 'Uno_num', 'Uno_den')}
    total.update(Harrell=total['Harrell_num'] / total['Harrell_den'], Uno60=total['Uno_num'] / total['Uno_den'])
    return total


def report(args, contract):
    rows = []; per_dimension = []
    for dim in contract['specification']['dimensions']:
        cells = []
        for part in contract['specification']['inner_partitions']:
            path = args.private / 'cells' / f'PCA{dim}_UNWHITENED_p{part}' / 'COMPLETE.json'
            if not path.exists(): continue
            value = json.loads(path.read_text())
            wanted = dict(lock_sha256=sha256(args.private / 'LOCK.json'), partition=part, dimension=dim, source_contract_sha256=canonical(contract))
            require(value['status'] == 'COMPLETE' and value['identity'] == wanted
                    and value['artifact_sha256'] == sha256(path.parent / 'model.pt'), 'GEOMETRY_REPORT_SOURCE_CHANGED')
            cells.append(value)
            row = dict(scope='PARTITION', dimension=dim, partition=part, status='COMPLETE',
                new_Harrell=value['metrics']['Harrell'], new_Uno60=value['metrics']['Uno60'],
                original_O2_Harrell=value['original_O2_metrics']['Harrell'], original_O2_Uno60=value['original_O2_metrics']['Uno60'],
                O0_Harrell=value['O0_metrics']['Harrell'], O0_Uno60=value['O0_metrics']['Uno60'],
                delta_H_vs_original_O2=value['metrics']['Harrell'] - value['original_O2_metrics']['Harrell'],
                delta_U60_vs_original_O2=value['metrics']['Uno60'] - value['original_O2_metrics']['Uno60'],
                delta_H_vs_O0=value['metrics']['Harrell'] - value['O0_metrics']['Harrell'],
                delta_U60_vs_O0=value['metrics']['Uno60'] - value['O0_metrics']['Uno60'],
                new_vs_O0_standardized_valid_SD_difference=value['diagnostics']['new_vs_O0_valid_SD_difference'],
                old_vs_O0_standardized_valid_SD_difference=value['diagnostics']['old_vs_O0_valid_SD_difference'],
                new_vs_O0_valid_Pearson=value['diagnostics']['new_vs_O0_valid_Pearson'],
                old_vs_O0_valid_Pearson=value['diagnostics']['old_vs_O0_valid_Pearson'],
                original_PC_gene_L2_directional_penalty_ratio=value['PCA_math_audit']['old_gene_L2_directional_penalty_ratio'],
                original_saved_RNA_decoder_Pearson=(value['original_RNA_retention'] or {}).get('risk_decoder_Pearson'),
                original_saved_RNA_decoder_R2=(value['original_RNA_retention'] or {}).get('risk_decoder_R2'),
                actual_new_Cox_fits=1, PCA_refits=0, teacher_refits=0)
            rows.append(row)
        summary = dict(scope='THREE_PARTITION_COUNTS', dimension=dim, partition='ALL',
                       status='COMPLETE' if len(cells) == 3 else 'RUNNING', completed_new_Cox_fits=len(cells))
        if len(cells) == 3:
            new, old, o0 = (aggregate([v[key] for v in cells]) for key in ('metrics', 'original_O2_metrics', 'O0_metrics'))
            summary.update(new_Harrell=new['Harrell'], new_Uno60=new['Uno60'], original_O2_Harrell=old['Harrell'],
                original_O2_Uno60=old['Uno60'], O0_Harrell=o0['Harrell'], O0_Uno60=o0['Uno60'],
                delta_H_vs_original_O2=new['Harrell']-old['Harrell'], delta_U60_vs_original_O2=new['Uno60']-old['Uno60'],
                delta_H_vs_O0=new['Harrell']-o0['Harrell'], delta_U60_vs_O0=new['Uno60']-o0['Uno60'],
                positive_partitions_H_vs_original_O2=sum(v['metrics']['Harrell'] > v['original_O2_metrics']['Harrell'] for v in cells),
                positive_partitions_U60_vs_original_O2=sum(v['metrics']['Uno60'] > v['original_O2_metrics']['Uno60'] for v in cells),
                new_counts=new, original_O2_counts=old, O0_counts=o0)
        per_dimension.append(summary)
    complete = len(rows) == 9
    value = dict(schema='bgb_en_r2_pca_geometry_report_v1', status='COMPLETE_NINE_EXPLORATORY_COX_FITS' if complete else 'RUNNING',
        analysis_role='new_exploratory_diagnostic_locked_after_inspection_of_existing_inner_results_not_independent_confirmation',
        source_contract_sha256=canonical(contract), private_lock_sha256=sha256(args.private / 'LOCK.json'),
        source_contract=contract, new_Cox_fits=len(rows), PCA_refits=0, teacher_refits=0, GPU_executions=0,
        outer_test_evaluated=False, changes_R4_finalists=False, all_dimensions_reported=True,
        aggregation='within_partition_pair_counts_sum_no_cross_partition_risk_ranking', summaries=per_dimension, partition_rows=rows,
        mathematical_interpretation=dict(unwhitened='lambda*||beta_PC||^2=lambda*||V_transpose_beta_PC||^2_in_the_retained_orthonormal_gene_subspace',
            original_unit_scaled='lambda*sum_j(PC_train_SD_j^2*beta_unscaled_PC_j^2);different_anisotropic_gene_penalty',
            nonzero_intercept='PCA_centering_is_a_constant_risk_shift_and_Cox_ranking_is_invariant',
            information_loss_boundary='High_heldout_teacher_risk_decodability_alone_does_not_guarantee_the_selected_penalized_Cox_readout_retains_that_ranking'))
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'R2_PCA_GEOMETRY_CONTROL.json', value)
    flat = rows + [{k:v for k,v in row.items() if not k.endswith('_counts')} for row in per_dimension]
    columns = list(dict.fromkeys(key for row in flat for key in row)) or ['dimension', 'status']
    with (args.output / 'R2_PCA_GEOMETRY_CONTROL.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(flat)
    fmt = lambda x: '—' if x is None else f'{x:.6f}'
    text = ['# R2 PCA 正则几何对照', '', f"状态：**{value['status']}**；新Cox拟合 {len(rows)}/9，PCA/teacher重拟合0，GPU执行0。", '',
        '本诊断在看到原内层PCA的高RNA教师可解码性、低DSS排序后提出；仅在本诊断的9个拟合之前锁定源码、输入和配置。'
        '属于探索性故障归因，不是独立性能证据，不修改原O2或R4候选。', '',
        '所有维度32/128/256预先全报；逐分区用原R的标准化、原PCA中心与正交基，不做PCA重拟合。'
        '新模型直接在未whiten PC scores上用λ=1的Efron Ridge Cox训练。', '',
        '| PC维度 | 新未whiten H | 新Uno60 | 原O2 H | 原O2 Uno60 | O0 H | O0 Uno60 | 新−原O2 H | 新−O0 H |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in per_dimension:
        text.append('| ' + str(row['dimension']) + ' | ' + ' | '.join(fmt(row.get(k)) for k in
            ('new_Harrell','new_Uno60','original_O2_Harrell','original_O2_Uno60','O0_Harrell','O0_Uno60','delta_H_vs_original_O2','delta_H_vs_O0')) + ' |')
    text += ['', '## 数学与解释边界', '',
        '- 正交基下，未whiten PC系数的L2等于同一保留子空间内gene-space系数L2；原先每PC单位方差再统一λ的做法等价于按PC训练方差加权的不同基因空间惩罚。',
        '- 原始排序指标保留各模型原风险数值。风险差SD诊断先将每个模型按各自训练风险均值/SD标准化，未使用验证分布估计尺度。',
        '- 高维表示可解码出RNA教师排序而其现有Cox读出排序较差，不能直接称为表示信息已丢失；本实验只隔离PCA后标准化造成的正则几何变化。',
        '- 即使未whiten对照改善，也不证明生物语义、交互机制或外部泛化成立；也不消除截断子空间、监督事件数量及模型选择方差的影响。',
        '- 配对差值来自同分区独立concordance分子分母；不拼接跨分区风险，不选最佳维度作为独立确认，不生成CI、bootstrap或p值。', '']
    (args.output / 'R2_PCA_GEOMETRY_CONTROL.md').write_text('\n'.join(text), encoding='utf-8')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['lock', 'run', 'report'], required=True)
    parser.add_argument('--assets', type=Path, required=True); parser.add_argument('--core-private', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True); parser.add_argument('--output', type=Path, default=ROOT/'reports/bgb_en')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--geometry-config', type=Path, default=ROOT/'configs/bgb_en/pca_geometry_v1.json')
    parser.add_argument('--max-workers', type=int, choices=[1, 2], default=2); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    contract, prepared, raw = source_contract(args)
    # Qualify every saved PCA before writing the pre-fit lock or running the
    # first new optimizer. No PCA is learned in this check.
    for dim in [32,128,256]:
        for part in [0,1,2]: load_original(args, prepared[part], raw, part, dim, contract)
    enforce_lock(args, contract, create=args.stage == 'lock')
    if args.stage == 'lock':
        print(json.dumps(dict(status='LOCKED_BEFORE_NINE_EXPLORATORY_FITS', optimizer_fits=0)), flush=True); return
    if args.stage == 'run':
        tasks = [(part, dim) for dim in [32,128,256] for part in [0,1,2]]
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            for future in as_completed([pool.submit(fit_unit,args,contract,prepared[p],raw,p,d) for p,d in tasks]):
                print(json.dumps(future.result()), flush=True)
    result = report(args, contract)
    print(json.dumps({key:result[key] for key in ('status','new_Cox_fits','PCA_refits','teacher_refits','GPU_executions')}), flush=True)


if __name__ == '__main__':
    main()

"""Independent, preregistered R2 sensitivity fits; frozen core code is untouched.

Four variants x three existing inner partitions x two seeds = 24 new neural
fits. Training consumes the existing prepared anchors read-only. This script
never replaces a core COMPLETE receipt or changes its scientific source lock.
The default synthetic qualification uses an explicitly labelled CPU SSM mock
backend, zero optimizer steps and no real cohort data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bgb_en.omics import SparsePathwayEncoder
from src.bgb_en.training import MechanismModel, train_neural


SPEC = {
    'schema': 'bgb_en_r2_sensitivity_v1', 'outer_fold': 1, 'inner_partitions': [0, 1, 2], 'seeds': [31, 47],
    'epochs': 80, 'evaluation_epochs': [20, 40, 80], 'batch_size': 8, 'learning_rate': .001,
    'gamma': .3, 'sequence_backend_formal': 'mamba', 'random_order_seed': 270901, 'random_orders': 4,
    'variants': {
        'O8_F2': {'core_arm': 'O8', 'factor_dimensions': 2},
        'O8_F8': {'core_arm': 'O8', 'factor_dimensions': 8},
        'O5_RANDOM_ORDERS': {'core_arm': 'O5', 'order_strategy': 'four_fixed_independent_random_permutations'},
        'O6_RANDOM_ORDERS': {'core_arm': 'O6', 'order_strategy': 'four_fixed_independent_random_permutations'}},
    'expected_new_neural_fits': 24, 'expected_analytical_risk_decoder_fits': 48,
    'direct_RNA512_anchor': 'exact_existing_prepared_risk_no_refit',
    'scientific_endpoint': 'fixed_epoch80_no_validation_checkpoint_selection',
    'outer_test_evaluated': False,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def locked_orders(pathways):
    generator = torch.Generator().manual_seed(SPEC['random_order_seed'])
    return torch.stack([torch.randperm(pathways, generator=generator) for _ in range(SPEC['random_orders'])])


def make_variant(prepared, membership, variant, seed, device, *, backend='mamba'):
    """Change only factor width or the orders buffer; all other inputs match."""
    if variant not in SPEC['variants'] or seed not in SPEC['seeds']:
        raise ValueError('Variant/seed outside sensitivity lock')
    definition = SPEC['variants'][variant]
    torch.manual_seed(seed); np.random.seed(seed)
    model = MechanismModel('R2', definition['core_arm'], prepared, membership, None, device,
                           gamma=SPEC['gamma'], backend=backend)
    if 'factor_dimensions' in definition:
        original = model.omics.state_dict()
        # Reset to the same construction RNG: all unchanged pooling/token
        # tensors are bitwise identical to the corresponding core O8 seed.
        torch.manual_seed(seed)
        replacement = SparsePathwayEncoder(membership, mode='factors', width=32, patterns=4,
                                            factors=definition['factor_dimensions'], sequence_backend=backend,
                                            order_seed=270101, orders=4).to(device)
        for name, value in replacement.state_dict().items():
            if not name.startswith(('factor.', 'risk_head.')) and not torch.equal(value, original[name]):
                raise ValueError('Factor sensitivity changed the shared token encoder initialization')
        model.omics = replacement
    else:
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        replacement = locked_orders(model.omics.rna.pathways).to(device)
        with torch.no_grad():
            model.omics.sequence.orders.copy_(replacement)
        if any(not torch.equal(before[name], p) for name, p in model.named_parameters()):
            raise ValueError('Order sensitivity changed trainable initialization')
    return model


def synthetic_qualification(output):
    torch.manual_seed(731)
    n, genes = 8, 13
    rna = np.random.default_rng(731).normal(size=(n, genes)).astype(np.float32)
    prepared = {'features': {'rna': rna, 'rna_mask': np.ones_like(rna, bool),
                            'cna_gain': np.zeros_like(rna), 'cna_loss': np.zeros_like(rna),
                            'cna_mask': np.ones_like(rna, bool), 'C': np.zeros((n, 3), np.float32)},
                'train': np.arange(6), 'valid': np.arange(6, 8),
                'RNA': np.linspace(-.5, .5, n), 'CP': np.zeros(n), 'base': np.zeros(n), 'beta': 0.}
    membership = torch.zeros(5, genes, dtype=torch.bool)
    for k in range(5): membership[k, k:k+5] = True
    rows = []
    for variant in SPEC['variants']:
        model = make_variant(prepared, membership, variant, 31, 'cpu', backend='ssm')
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        output_value = model(torch.arange(n)); output_value['risk'].square().mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        unchanged = all(torch.equal(before[k], p) for k, p in model.named_parameters())
        if not unchanged or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Synthetic sensitivity qualification failed')
        rows.append({'variant': variant, 'parameters': sum(p.numel() for p in model.parameters()),
                     'finite_gradients': True, 'gradient_tensors_nonzero': sum(bool(g.abs().sum() > 0) for g in gradients),
                     'weights_unchanged': unchanged, 'direct_RNA_risk_unchanged': bool(torch.equal(model.offset, torch.tensor(prepared['RNA'], dtype=torch.float32))),
                     'parameter_audit': model.omics.parameter_audit()})
    result = {'status': 'PASS_SYNTHETIC_CPU_ONLY_NOT_A_FORMAL_CELL', 'synthetic_only': True,
              'sequence_backend': 'explicit_ssm_only_for_synthetic_constructor_gradient_check',
              'formal_backend_remains_official_mamba': True, 'optimizer_steps': 0,
              'real_cohort_reads': 0, 'source_sha256': digest(__file__), 'specification': SPEC, 'variants': rows}
    write(output, result); return result


def train(args):
    from scripts.bgb_en_run import load_inputs, representation_retention
    if args.variant is None or args.partition not in SPEC['inner_partitions'] or args.seed not in SPEC['seeds']:
        raise ValueError('An explicitly locked sensitivity unit is required')
    if args.private is None or args.core_private is None or args.assets is None:
        raise ValueError('Training requires private auxiliary/core directories and existing assets')
    auxiliary, core = args.private.resolve(), args.core_private.resolve()
    if (auxiliary == core or auxiliary.is_relative_to(core) or core.is_relative_to(auxiliary)
            or auxiliary.is_relative_to((ROOT / 'reports').resolve())):
        raise ValueError('Auxiliary training artifacts require a distinct private directory')
    assets, core_config, raw, membership, core_lineage = load_inputs(args)
    if core_config['training']['epochs'] != SPEC['epochs'] or core_config['training']['learning_rate'] != SPEC['learning_rate']:
        raise ValueError('Core training budget differs from the sensitivity lock')
    if core_config['R2']['gamma'] != SPEC['gamma'] or core_config['training']['batch_size'] != SPEC['batch_size']:
        raise ValueError('Core magnitude/batch contract differs from sensitivity lock')
    stem = f"outer{SPEC['outer_fold']}_inner{args.partition}"
    path = args.core_private / 'prepared' / (stem + '.pt')
    prep_receipt = json.loads((args.core_private / 'prepared' / (stem + '.json')).read_text())
    if prep_receipt['lineage'] != core_lineage or prep_receipt['sha256'] != digest(path):
        raise ValueError('Core prepared source or hash changed; no anchor refitting is permitted')
    prepared = torch.load(path, map_location='cpu', weights_only=False)
    if prepared['lineage'] != core_lineage:
        raise ValueError('Prepared checkpoint lineage differs')
    lineage = {'core_lineage': core_lineage, 'auxiliary_script_sha256': digest(__file__),
               'specification': SPEC, 'prepared_sha256': digest(path), 'variant': args.variant}
    label = f'{args.variant}_p{args.partition}_seed{args.seed}'
    directory = args.private / 'cells' / label
    complete, artifact_path = directory / 'COMPLETE.json', directory / 'model.pt'
    if complete.exists():
        old = json.loads(complete.read_text())
        if not args.resume or old['lineage'] != lineage or old['artifact_sha256'] != digest(artifact_path):
            raise ValueError('Completed auxiliary unit has incompatible source/hash or lacks --resume')
        return {'status': 'REUSED_VERIFIED_COMPLETE', 'actual_new_optimizer_fits': 0}
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'START.json').open('x') as handle:
        json.dump({'lineage': lineage, 'partition': args.partition, 'seed': args.seed}, handle)
    started = time.monotonic()
    try:
        model = make_variant(prepared, membership, args.variant, args.seed, args.device)
        result = train_neural(model, raw, prepared, epochs=SPEC['epochs'], batch_size=SPEC['batch_size'],
                              lr=SPEC['learning_rate'], evaluate_epochs=tuple(SPEC['evaluation_epochs']))
        retention = representation_retention(model, prepared)
        result.update(lineage=lineage, variant=args.variant, core_arm=SPEC['variants'][args.variant]['core_arm'],
                      seed=args.seed, inner_partition=args.partition, train=prepared['train'], valid=prepared['valid'],
                      retention=retention, omics_parameter_audit=model.omics.parameter_audit(), outer_test_evaluated=False)
        temporary = artifact_path.with_suffix('.pt.tmp'); torch.save(result, temporary); temporary.replace(artifact_path)
        receipt = {'status': 'COMPLETE', 'stage': 'R2_SENSITIVITY', 'variant': args.variant,
                   'core_arm': SPEC['variants'][args.variant]['core_arm'], 'seed': args.seed,
                   'inner_partition': args.partition, 'outer_fold': SPEC['outer_fold'], 'lineage': lineage,
                   'metrics': result['snapshots'][80]['metrics'], 'curves': result['curves'],
                   'retention': retention, 'epochs': 80, 'parameters': result['parameters'],
                   'peak_GPU_bytes': result['peak_GPU_bytes'], 'elapsed_seconds': time.monotonic() - started,
                   'actual_new_optimizer_fits': 1, 'analytical_risk_decoder_fits': 2,
                   'upstream_anchor_refits': 0, 'artifact_sha256': digest(artifact_path),
                   'outer_test_evaluated': False, 'omics_parameter_audit': model.omics.parameter_audit()}
        write(complete, receipt); return {k: receipt[k] for k in ('status', 'variant', 'seed', 'inner_partition', 'elapsed_seconds')}
    except Exception as exc:
        write(directory / 'FAILED.json', {'status': 'FAILED', 'error_type': type(exc).__name__, 'lineage': lineage})
        raise


def report(args):
    from scripts.bgb_en_report_inner import aggregate, all_defined_mean, write_csv
    if args.private is None or args.core_private is None:
        raise ValueError('Reporting needs the auxiliary and core receipt directories')
    cells = {}
    for path in (args.private / 'cells').glob('**/COMPLETE.json'):
        row = json.loads(path.read_text()); key = row['variant'], row['inner_partition'], row['seed']
        if (row.get('status') != 'COMPLETE' or row['variant'] not in SPEC['variants']
                or row['inner_partition'] not in SPEC['inner_partitions'] or row['seed'] not in SPEC['seeds']
                or row['outer_fold'] != SPEC['outer_fold'] or row.get('outer_test_evaluated') is not False
                or row['lineage']['specification'] != SPEC or row['epochs'] != 80 or key in cells
                or row['lineage']['variant'] != row['variant'] or row['lineage']['auxiliary_script_sha256'] != digest(__file__)):
            raise ValueError('Auxiliary receipts differ from the locked sensitivity matrix')
        finals=[point for point in row['curves'] if point['epoch']==80]
        if len(finals)!=1 or finals[0]['valid']!=row['metrics']:
            raise ValueError('Auxiliary endpoint differs from its fixed epoch80 snapshot')
        cells[key] = row
    core = {}
    for path in (args.core_private / 'cells').glob('**/COMPLETE.json'):
        row = json.loads(path.read_text())
        if row.get('stage') == 'R2': core[row['arm'], row['inner_partition'], row['seed']] = row
    rows, retention_rows = [], []
    for variant, definition in SPEC['variants'].items():
        selected = [cells[variant, p, s] for p in SPEC['inner_partitions'] for s in SPEC['seeds'] if (variant, p, s) in cells]
        row = {'variant': variant, 'core_reference': definition['core_arm'], 'status': 'COMPLETE' if len(selected) == 6 else 'RUNNING',
               'completed_fits': len(selected), 'planned_fits': 6, 'Harrell': None, 'Uno60': None,
               'paired_delta_H': None, 'paired_delta_U60': None, 'positive_H_cells': None, 'positive_U_cells': None,
               'actual_upstream_refits': 0, 'comparison_type': 'REAL_RETRAINING'}
        for result in selected:
            for name, value in result.get('retention', {}).items():
                retention_rows.append({'variant': variant, 'inner_partition': result['inner_partition'], 'seed': result['seed'],
                                       'representation': name, 'dimension': value['dimension'],
                                       'Pearson': value['Pearson'], 'R2': value['R2']})
        complete_pairs = []
        for result in selected:
            reference = core.get((definition['core_arm'], result['inner_partition'], result['seed']))
            if reference is not None:
                if reference['lineage'] != result['lineage']['core_lineage']:
                    raise ValueError('Auxiliary/core comparison has different source/data lineage')
                if any(abs(result['metrics'][key] - reference['metrics'][key]) > 1e-10 for key in ('Harrell_den', 'Uno_den')):
                    raise ValueError('Auxiliary/core paired test population differs')
                complete_pairs.append((result, reference))
        row['paired_reference_status'] = 'COMPLETE' if len(complete_pairs) == 6 else 'RUNNING'
        if len(selected) == 6:
            for metric in ('Harrell', 'Uno60'):
                row[metric] = all_defined_mean([aggregate([r['metrics'] for r in selected if r['seed'] == seed])[metric]
                                               for seed in SPEC['seeds']])
        if len(complete_pairs) == 6:
            for metric, name in [('Harrell', 'paired_delta_H'), ('Uno60', 'paired_delta_U60')]:
                differences = []
                for seed in SPEC['seeds']:
                    part = [(a, b) for a, b in complete_pairs if a['seed'] == seed]
                    differences.append(aggregate([a['metrics'] for a, _ in part])[metric]
                                       - aggregate([b['metrics'] for _, b in part])[metric])
                    row[f'{name}_seed{seed}'] = differences[-1]
                row[name] = all_defined_mean(differences)
            row['positive_H_cells'] = sum(a['metrics']['Harrell'] > b['metrics']['Harrell'] for a, b in complete_pairs)
            row['positive_U_cells'] = sum(a['metrics']['Uno60'] > b['metrics']['Uno60'] for a, b in complete_pairs)
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'R2_OMICS_SENSITIVITY.csv', rows, ('variant', 'status'))
    write_csv(args.output / 'R2_OMICS_SENSITIVITY_RETENTION.csv', retention_rows, ('variant', 'inner_partition', 'seed'))
    from scripts.bgb_en_report_inner import table
    text = ['# R2 独立预锁敏感性', '', f"状态：{'COMPLETE' if len(cells) == 24 else 'RUNNING'}；完成 {len(cells)}/24 个真实神经拟合。", '',
            '复用原三个prepared锚点，不重新选择RNA/PCA或拟合教师；固定80轮，两个seed分别先合并分区分子分母再均值。', '',
            table(rows, [('variant', '变体'), ('status', '状态'), ('Harrell', 'H'), ('Uno60', 'U60'),
                         ('paired_delta_H', '对原臂ΔH'), ('paired_delta_U60', '对原臂ΔU60')]), '',
            'O8_F2/O8_F8只改变非线性低秩风险读出宽度；O5/O6_RANDOM_ORDERS只替换4×K顺序buffer，固定seed270901、参数与scan次数不变。', '',
            '这是真实重训的内层敏感性，不能称独立患者确认。原核心报告保持不变；缺少的单元或原臂配对结果留空，不外推为完成。', '',
            '每个新神经拟合有2个训练分区解析风险decoder，和DSS优化模型分开计数。没有新增outer测试、bootstrap或事后挑checkpoint。', '']
    (args.output / 'R2_OMICS_SENSITIVITY.md').write_text('\n'.join(text), encoding='utf-8')
    summary = {'status': 'COMPLETE' if len(cells) == 24 else 'RUNNING', 'completed_neural_fits': len(cells),
               'planned_neural_fits': 24, 'actual_upstream_anchor_refits': 0,
               'analytical_risk_decoder_fits': sum(r.get('analytical_risk_decoder_fits', 0) for r in cells.values()),
               'specification': SPEC, 'outer_test_evaluated': False}
    write(args.output / 'R2_OMICS_SENSITIVITY_AUDIT.json', summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['qualify', 'train', 'report'], default='qualify')
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--core-private', type=Path)
    parser.add_argument('--private', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/bgb_en')
    parser.add_argument('--variant', choices=list(SPEC['variants']))
    parser.add_argument('--partition', type=int, choices=SPEC['inner_partitions'], default=0)
    parser.add_argument('--seed', type=int, choices=SPEC['seeds'], default=31)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    result = (synthetic_qualification(args.output / 'R2_OMICS_SENSITIVITY_QUALIFICATION.json') if args.stage == 'qualify'
              else train(args) if args.stage == 'train' else report(args))
    print(json.dumps({k: result[k] for k in ('status', 'completed_neural_fits') if k in result}), flush=True)


if __name__ == '__main__':
    main()

"""Read-only F7 forward/top-k qualification; no optimizer or training step.

Run inherited and deterministic modes in distinct processes, after the active
training queue is idle. CUDA is never selected implicitly. This script does not
repair/overwrite a failed cell, relax 2e-5, or claim CPU tests qualify CUDA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def configure_backend(mode):
    # Main calls this before importing torch or constructing a CUDA context.
    if mode == 'deterministic':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if mode == 'deterministic':
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    return dict(mode=mode, torch_version=torch.__version__, cuda_build=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def collect_forward(model, indices, grad_enabled):
    """Capture detached intermediates only; never backward or mutate weights."""
    import torch
    captures, count, hooks = {}, {}, []
    selected = ('path.project', 'omics.rna', 'omics', 'interaction.patch_project',
                'interaction.omics_project', 'interaction.path_match', 'interaction.omics_match',
                'interaction.path_value', 'interaction.omics_value')
    def record(name):
        def hook(_module, _args, result):
            index = count.get(name, 0); count[name] = index+1
            if name == 'omics.rna':
                result = result[3]  # pre-projection molecular program patterns
            elif name == 'omics':
                result = result['tokens']
            captures[name+':'+str(index)] = result.detach().cpu().clone()
        return hook
    for name, module in model.named_modules():
        if name in selected:
            hooks.append(module.register_forward_hook(record(name)))
    interaction = []
    hooks.append(model.interaction.register_forward_hook(lambda _m, _a, out: interaction.append(out)))
    try:
        with torch.set_grad_enabled(grad_enabled):
            output = model(indices)
        captures['risk'] = output['risk'].detach().cpu().clone()
        captures['delta'] = output['delta'].detach().cpu().clone()
        trace = interaction[-1]['trace'][0]['omics_to_patch']
        captures['competition_score'] = trace['score'].detach().cpu().clone()
        captures['selected_attention'] = trace['attention'].detach().cpu().clone()
        captures['selected_message'] = trace['message'].detach().cpu().clone()
        return captures
    finally:
        for hook in hooks:
            hook.remove()


def compare_topk(left_score, right_score, left_attention, right_attention, *, topk=64):
    """Separate continuous numerical drift from discontinuous support changes."""
    support_left, support_right = left_attention > 0, right_attention > 0
    eligible_rows = support_left.any(-1) | support_right.any(-1)
    changed = (support_left != support_right).any(-1) & eligible_rows
    tv = .5*(left_attention-right_attention).abs().sum(-1)
    def maximum(tensor):
        return float(tensor.abs().max()) if tensor.numel() else 0.
    return dict(score_max_abs_error=maximum(right_score-left_score),
        attention_max_abs_error=maximum(right_attention-left_attention),
        eligible_query_rows=int(eligible_rows.sum()), changed_support_query_rows=int(changed.sum()),
        symmetric_difference_selected_keys=int((support_left != support_right).sum()),
        attention_TV_mean=float(tv[eligible_rows].mean()) if eligible_rows.any() else None,
        attention_TV_max=float(tv.max()),
        boundary_evidence_requires_full_valid_mask=True)


def topk_boundary(score, valid, topk=64):
    import torch
    from src.bgb_en.interaction import masked_softmax
    probability = masked_softmax(score, valid, dim=-2)
    valid = torch.broadcast_to(valid.bool(), score.shape)
    eligible = valid.sum(-1) > topk
    keep = min(topk+1, probability.shape[-1])
    ordered = torch.topk(probability.masked_fill(~valid, -1), keep, dim=-1).values
    if keep <= topk:
        return dict(rows_with_more_than_topk_keys=0, minimum_margin=None, mean_margin=None, exact_boundary_ties=0)
    margin = ordered[..., topk-1]-ordered[..., topk]
    values = margin[eligible]
    return dict(rows_with_more_than_topk_keys=int(eligible.sum()),
                minimum_margin=float(values.min()) if values.numel() else None,
                mean_margin=float(values.mean()) if values.numel() else None,
                exact_boundary_ties=int((values == 0).sum()))


def compare_captures(left, right):
    import torch
    if list(left) != list(right):
        raise ValueError('Forward capture module call order changed')
    intermediate = []
    for name in left:
        delta = right[name].double()-left[name].double()
        intermediate.append(dict(name=name, bitwise_equal=bool(torch.equal(left[name], right[name])),
            max_abs_error=float(delta.abs().max()), RMS_error=float(delta.square().mean().sqrt())))
    return dict(first_non_bitwise_intermediate=next((row['name'] for row in intermediate if not row['bitwise_equal']), None),
        intermediate=intermediate,
        risk_within_unchanged_gradient_cache_threshold=bool((left['risk']-right['risk']).abs().max() <= 2e-5),
        topk=compare_topk(left['competition_score'], right['competition_score'],
                          left['selected_attention'], right['selected_attention']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--partition', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--seed', type=int, choices=(31, 47), required=True)
    parser.add_argument('--backend', choices=('inherited', 'deterministic'), required=True)
    parser.add_argument('--state', choices=('initial', 'final80'), default='initial')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-limit', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Diagnostic output already exists; choose a new path')
    if args.batch_limit < 1 or args.repeats < 2:
        raise ValueError('Positive batch limit and at least two repeats required')
    backend = configure_backend(args.backend)
    import numpy as np
    import torch
    from scripts import bgb_en_run as driver
    from src.bgb_en.training import MechanismModel, PatchBank
    from src.bgb_en.data import read_patch_cache
    torch.set_num_threads(2)
    assets, config, raw, member, lineage = driver.load_inputs(args)
    prepared_path = args.private/'prepared'/f'outer1_inner{args.partition}.pt'
    prepared_receipt = json.loads(prepared_path.with_suffix('.json').read_text())
    if prepared_receipt['sha256'] != sha(prepared_path) or prepared_receipt['lineage'] != lineage:
        raise ValueError('Prepared origin/hash differs')
    prepared = torch.load(prepared_path, map_location='cpu', weights_only=False)
    cell = args.private/'cells'/f'R3_outer1_inner{args.partition}_F7_seed{args.seed}'
    original_receipts = {name: sha(cell/name) for name in ('START.json', 'FAILED.json', 'COMPLETE.json') if (cell/name).exists()}
    report = dict(status='PENDING', neural_fits=0, optimizer_updates=0, backpropagations=0,
        arm='F7', partition=args.partition, seed=args.seed, state=args.state, device=args.device,
        backend=backend, original_receipt_sha256=original_receipts,
        core_lineage=lineage, prepared_sha256=sha(prepared_path),
        diagnostic_sources={name: sha(ROOT/name) for name in ('scripts/bgb_en_f7_determinism_diagnostic.py',)},
        threshold=2e-5, threshold_unchanged=True, original_cell_files_modified=False,
        limitation='No failed-epoch state was persisted; an initial/successful80 read-only forward cannot reproduce an unrecorded failed training state',
        batches=[])
    try:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        bank = PatchBank(read_patch_cache(assets['patch_regions']), args.device)
        model = MechanismModel('R3', 'F7', prepared, member, bank, args.device,
                               gamma=config['R3'].get('gamma', .1)).train()
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if args.state == 'final80':
            receipt = json.loads((cell/'COMPLETE.json').read_text())
            if receipt['artifact_sha256'] != sha(cell/'model.pt') or receipt['lineage'] != lineage:
                raise ValueError('Successful F7 parent source/artifact differs')
            parent = torch.load(cell/'model.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(parent['snapshots'][80]['state_dict'], strict=True)
            report['checkpoint_artifact_sha256'] = receipt['artifact_sha256']
            initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        size = config['training']['batch_size']
        train = np.asarray(prepared['train'])
        for start in range(0, min(len(train), args.batch_limit*size), size):
            indices = torch.as_tensor(train[start:start+size], device=args.device)
            runs = [collect_forward(model, indices, enabled) for enabled in (False, True) for _ in range(args.repeats)]
            # The observed support, rather than membership coverage alone,
            # controls query validity for patients with natural missing values.
            own_args = {key: value[indices] for key, value in model.inputs.items()}
            with torch.no_grad():
                token_mask = model.omics(**own_args)['token_mask'].detach().cpu()
            valid = bank.mask[indices].detach().cpu()[:, :, None, :] & token_mask[:, None, :, None]
            comparisons = dict(no_grad_repeat=[compare_captures(runs[0], run) for run in runs[1:args.repeats]],
                               grad_repeat=[compare_captures(runs[args.repeats], run) for run in runs[args.repeats+1:]],
                               no_grad_vs_grad=[compare_captures(runs[0], run) for run in runs[args.repeats:]])
            report['batches'].append(dict(training_batch_ordinal=start//size, patients=len(indices), comparisons=comparisons,
                boundary_no_grad=topk_boundary(runs[0]['competition_score'], valid),
                boundary_grad=topk_boundary(runs[args.repeats]['competition_score'], valid)))
        report['weights_bitwise_unchanged'] = all(torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in initial.items())
        report['parameter_gradients_absent'] = all(parameter.grad is None for parameter in model.parameters())
        if not report['weights_bitwise_unchanged'] or not report['parameter_gradients_absent']:
            raise ValueError('Read-only forward qualification mutated weights or accumulated gradients')
        all_comparisons = [comparison for batch in report['batches'] for rows in batch['comparisons'].values() for comparison in rows]
        report['all_comparisons_within_threshold'] = all(row['risk_within_unchanged_gradient_cache_threshold'] for row in all_comparisons)
        report['all_comparisons_bitwise_equal'] = all(row['first_non_bitwise_intermediate'] is None for row in all_comparisons)
        report['status'] = 'PASS_SAMPLED_READONLY_FORWARD' if report['all_comparisons_within_threshold'] else 'FAIL_SAMPLED_READONLY_FORWARD'
    except Exception as exc:
        report.update(status='FAILED_DIAGNOSTIC_BACKEND_QUALIFICATION', error_type=type(exc).__name__, error=str(exc))
    if original_receipts != {name: sha(cell/name) for name in original_receipts}:
        raise ValueError('An original cell receipt changed during the read-only diagnostic')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: report.get(name) for name in ('status', 'backend', 'neural_fits', 'optimizer_updates', 'all_comparisons_within_threshold')}))


if __name__ == '__main__':
    main()

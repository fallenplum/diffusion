"""Post-fit Phase28 diagnostics; never selects or updates a predictor.

Two separate entry points are deliberate. ``history`` reads only the saved
training history and fit risk components on CPU. ``mechanism`` additionally
replays three fit inputs chosen by original selected-Patch count, without an
endpoint or optimizer. The latter must be scheduled explicitly on a free GPU.
The locked training closure and its qualification receipts are not modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import torch

from src.phase28_reread import data, runtime, train


FEATURE_KEYS = frozenset(('patient_ids', 'folds', 'rna_genes', 'rna_gene_symbols',
                         'cna_genes', 'cna_gene_symbols', 'cancer_type'))
VERSION = 'phase28_postfit_diagnostics_v1'


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def scalar(value):
    if isinstance(value, torch.Tensor):
        require(value.numel() == 1, 'diagnostic expects one scalar')
        value = value.detach().cpu().double().item()
    value = float(value)
    require(np.isfinite(value), 'nonfinite diagnostic scalar')
    return value


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().double().numpy()
    value = np.asarray(value, np.float64)
    require(value.size > 0 and np.isfinite(value).all(), 'nonfinite or empty diagnostic array')
    return value


def moments(value):
    x = array(value).reshape(-1)
    return {'count': int(len(x)), 'mean': float(x.mean()), 'SD': float(x.std()),
            'RMS': float(np.sqrt(np.mean(x*x))), 'minimum': float(x.min()),
            'median': float(np.median(x)), 'abs_p95': float(np.quantile(np.abs(x), .95)),
            'maximum': float(x.max())}


def rms(value):
    return scalar(value.detach().double().square().mean().sqrt())


def l2(value):
    return scalar(value.detach().double().norm())


def history_summary(record):
    """Descriptive fit-only learning curves, without an early-stop decision."""
    history = record['history']
    require(len(history) == 80 and [r['epoch'] for r in history] == list(range(1, 81)),
            'exactly the locked eighty updates are required')
    names = ('cox_loss', 'regularizer', 'objective', 'risk_RMS', 'main_residual_RMS',
             'interaction_RMS', 'clipped_from_gradient_L2',
             'maximum_absolute_component_replay_error')
    curves, trends = {}, {}
    for name in names:
        y = array([row[name] for row in history])
        curves[name] = [{'epoch': index+1, 'value': float(v)} for index, v in enumerate(y)]
        slope = np.polyfit(np.arange(71, 81, dtype=np.float64), y[-10:], 1)[0]
        trends[name] = {'first': float(y[0]), 'last': float(y[-1]),
            'last10_mean': float(y[-10:].mean()), 'last10_range': float(np.ptp(y[-10:])),
            'last10_OLS_slope_per_epoch': float(slope),
            'last10_first_to_last_change': float(y[-1]-y[-10]),
            'minimum_observed_training_value': float(y.min()),
            'epoch_of_minimum_training_value_diagnostic_only': int(y.argmin()+1)}
    module_names = sorted(set().union(*(row['data_gradient_L2_by_top_level_module'] for row in history)))
    gradients = {}
    for module in module_names:
        d = np.asarray([row['data_gradient_L2_by_top_level_module'].get(module, 0.) for row in history])
        t = np.asarray([row['total_gradient_L2_by_top_level_module'].get(module, 0.) for row in history])
        require(np.isfinite(d).all() and np.isfinite(t).all(), 'nonfinite saved gradient history')
        gradients[module] = {'data_L2_first10_mean': float(d[:10].mean()),
            'data_L2_last10_mean': float(d[-10:].mean()), 'data_L2_maximum': float(d.max()),
            'epochs_with_nonzero_data_gradient': int(np.count_nonzero(d)),
            'total_including_L2_last10_mean': float(t[-10:].mean()),
            'data_gradient_L2_curve': [{'epoch': index+1, 'L2': float(v)} for index, v in enumerate(d)]}
    fit = record['fit_components']
    r, base, interaction = (array(fit[k]) for k in ('risk', 'base', 'interaction'))
    require(r.ndim == base.ndim == interaction.ndim == 1 and r.shape == base.shape == interaction.shape,
            'fit risk components must share one patient order')
    require(np.max(np.abs(r-base-interaction)) <= 1e-6, 'saved fit risk decomposition is inconsistent')
    return {'epochs': 80, 'curves': curves, 'last10_descriptive_trends': trends,
        'data_versus_total_gradient_by_module': gradients,
        'gradient_clip_fraction': float(np.mean([row['clipped_from_gradient_L2'] > 5. for row in history])),
        'saved_fit_risk_components': {k: moments(fit[k]) for k in ('risk', 'base', 'interaction', 'path_main', 'gene_main')},
        'learning_curve_records_are_pre_step': True,
        'final_checkpoint_fit_Efron_NLL_post_step': scalar(record['training_audit']['final_fit_Efron_NLL']),
        'checkpoint_or_epoch_selected_by_this_analysis': False,
        'budget_sufficient_or_global_convergence_claimed': False,
        'interpretation': 'fit-only trajectories diagnose continued movement or instability; eighty nonconvex updates do not prove global convergence'}


def weight_change(base, conditional, support):
    """Real-evidence read changes, retaining the singleton identifiability limit."""
    require(base.shape == conditional.shape == support.shape, 'read weight/support dimensions differ')
    support = support.bool()
    require(torch.isfinite(base).all() and torch.isfinite(conditional).all(), 'nonfinite read weights')
    valid = support.any(-1)
    count = support.sum(-1)
    tv = .5*(conditional-base).abs().sum(-1)
    entropy = -(conditional.clamp_min(torch.finfo(conditional.dtype).tiny).log()*conditional).sum(-1)
    require(valid.any(), 'read has no valid query')
    result = {'valid_queries': scalar(valid.sum()), 'eligible_query_fraction_ge2_candidates': scalar((count[valid] >= 2).double().mean()),
        'TV_mean_all_valid_queries': scalar(tv[valid].double().mean()),
        'TV_maximum': scalar(tv[valid].max()),
        'effective_candidate_fraction_mean': scalar((entropy[valid].exp()/count[valid]).double().mean())}
    eligible = valid & (count >= 2)
    result['TV_mean_queries_ge2_candidates'] = scalar(tv[eligible].double().mean()) if eligible.any() else None
    singleton = valid & (count == 1)
    result['singleton_TV_maximum'] = scalar(tv[singleton].max()) if singleton.any() else None
    return result


def route_summary(route, path_valid, gene_valid):
    """Raw score common energy and learned null mass, not medical confidence."""
    score = route['score'][:, path_valid][:, :, gene_valid]
    require(score.numel() > 0, 'routing score has no valid paired evidence')
    centered = score-score.mean(-1, keepdim=True)
    energy = scalar(score.detach().double().square().sum())
    centered_energy = scalar(centered.detach().double().square().sum())
    ap = route['path_probability'][:, path_valid]
    ag = route['gene_probability'][:, gene_valid]
    return {'score_RMS': rms(score), 'score_row_mean_RMS': rms(score.mean(-1)),
        'score_row_centered_RMS': rms(centered),
        'score_row_centered_energy_fraction': centered_energy/energy if energy else None,
        'path_target_null_probability_mean': scalar(ap[..., -1].double().mean()),
        'gene_target_null_probability_mean': scalar(ag[..., -1].double().mean()),
        'path_target_real_probability_mass_mean': scalar(ap[..., :-1].double().sum(-1).mean()),
        'gene_target_real_probability_mass_mean': scalar(ag[..., :-1].double().sum(-1).mean())}


def _mean_numeric_dict(rows):
    require(bool(rows) and all(set(v) == set(rows[0]) for v in rows), 'diagnostic aggregation schema differs')
    result = {}
    for key in rows[0]:
        vals = [v[key] for v in rows if v[key] is not None]
        result[key] = float(np.mean(vals)) if vals else None
    return result


def gate_summary(gate):
    return {'mean': scalar(gate.detach().double().mean()),
            'fraction_lt_point01': scalar((gate < .01).double().mean()),
            'fraction_gt_point99': scalar((gate > .99).double().mean())}


def trace_summary(model, encoded, output):
    """One patient's equal-WSI measurements at each of the two update rounds."""
    layers = []
    for index, layer in enumerate(output['trace']):
        rows, routes = [], []
        for slide, entry in enumerate(layer['path']):
            stats = weight_change(entry['base_weight'], entry['conditional_weight'], encoded['path_support'][slide])
            stats.update(message_RMS=rms(entry['message']), evidence_read_difference_RMS=rms(entry['difference']),
                         anchored_token_change_RMS=rms(entry['change']),
                         anchored_token_change_maximum_absolute=scalar(entry['change'].abs().max()),
                         **{'gate_'+k: v for k, v in gate_summary(entry['gate']).items()})
            rows.append(stats)
            routes.append(route_summary(layer['own_routing_reference'][slide], encoded['path_valid'][slide], encoded['gene_valid']))
        gene = weight_change(layer['gene_base_weight'], layer['gene_conditional_weight'], encoded['gene_support'])
        gene.update(message_RMS=rms(layer['gene_message']), evidence_read_difference_RMS=rms(layer['gene_difference']),
                    anchored_token_change_RMS=rms(layer['gene_change']),
                    anchored_token_change_maximum_absolute=scalar(layer['gene_change'].abs().max()),
                    **{'gate_'+k: v for k, v in gate_summary(layer['gene_gate']).items()})
        # Trace stores input states. The next trace input or final output is
        # therefore the actual synchronous, anchored output of this round.
        new_path = output['states'][index+1]['path'] if index+1 < len(output['states']) else output['path_final']
        new_gene = output['states'][index+1]['gene'] if index+1 < len(output['states']) else output['gene_final']
        path_risk = torch.stack([model.rho_path(p, v)-model.rho_path(p0, v)
            for p, p0, v in zip(new_path, encoded['path0'], encoded['path_valid'])]).mean()
        gene_risk = model.rho_gene(new_gene, encoded['gene_valid'])-model.rho_gene(encoded['gene0'], encoded['gene_valid'])
        layers.append({'round': index+1, 'path_equal_WSI': _mean_numeric_dict(rows),
            'gene': gene, 'routing_equal_WSI': _mean_numeric_dict(routes),
            'path_risk_correction': scalar(path_risk), 'gene_risk_correction': scalar(gene_risk)})
    return layers


def _grad_norm(output, inputs, *, retain_graph):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    gradients = torch.autograd.grad(output, inputs, retain_graph=retain_graph, allow_unused=True)
    total = output.detach().new_zeros((), dtype=torch.float64)
    for value in gradients:
        if value is not None:
            require(torch.isfinite(value).all(), 'nonfinite path-isolated gradient')
            total = total+value.detach().double().square().sum()
    return scalar(total.sqrt()), gradients


def detached_route_context(output, encoded, *, requires_grad=False):
    """Same-patient layer context, kept separate from original evidence values."""
    rows = []
    for state in output['states']:
        rows.append({'gene': state['gene'].detach().clone().requires_grad_(requires_grad),
            'gene_valid': encoded['gene_valid'],
            'path': [p.detach().clone().requires_grad_(requires_grad) for p in state['path']],
            'path_valid': encoded['path_valid']})
    return rows


def diagnose_encoded(model, raw, encoded, offset):
    """Autograd reports sensitivity only; it does not impose a minimum effect."""
    output = model.read_encoded(encoded, offset, details=True)
    raw_norm, gradients = _grad_norm(output['path_interaction'], raw, retain_graph=True)
    rna_gradient = gradients[0][:, 0] if gradients[0] is not None else torch.zeros_like(raw[:, 0])
    ep_norm, _ = _grad_norm(output['gene_interaction'], encoded['path_evidence'], retain_graph=True)
    total_raw_norm, _ = _grad_norm(output['interaction'], raw, retain_graph=True)
    total_ep_norm, _ = _grad_norm(output['interaction'], encoded['path_evidence'], retain_graph=True)
    context = detached_route_context(output, encoded, requires_grad=True)
    controlled = model.read_encoded(encoded, offset, context=context)
    require(abs(scalar(controlled['risk'])-scalar(output['risk'])) <= 1e-6,
            'same-patient detached layer bank must replay matched risk')
    gc = tuple(row['gene'] for row in context)
    pc = tuple(p for row in context for p in row['path'])
    gc_norm, _ = _grad_norm(controlled['path_interaction'], gc, retain_graph=True)
    pc_norm, _ = _grad_norm(controlled['gene_interaction'], pc, retain_graph=False)
    gradients = {'original_molecular_to_path_risk_L2': raw_norm,
        'original_RNA_to_path_risk_L2': l2(rna_gradient),
        'original_Ep_to_gene_risk_L2': ep_norm,
        'original_molecular_to_total_interaction_L2': total_raw_norm,
        'original_Ep_to_total_interaction_L2': total_ep_norm,
        'isolated_gene_context_to_path_risk_L2': gc_norm,
        'isolated_path_context_to_gene_risk_L2': pc_norm,
        'minimum_nonzero_gradient_threshold_enforced': False}
    result = {'layers': trace_summary(model, encoded, output), 'gradients': gradients,
        'risk': {k: scalar(output[k]) for k in ('risk', 'base', 'offset', 'learned_path_main',
                  'learned_gene_main', 'path_interaction', 'gene_interaction', 'interaction')}}
    return result, output


@contextlib.contextmanager
def no_labels_or_training():
    """Permit fit-feature autograd, forbid all labels and predictor updates."""
    from src.phase13_rna import data as parent
    counters, reads = Counter(), Counter()
    original = np.load

    def blocked(name):
        def reject(*args, **kwargs):
            counters[name] += 1
            raise RuntimeError('post-fit diagnostics forbids '+name)
        return reject

    class FeatureView:
        def __init__(self, wrapped): self.wrapped = wrapped
        def __enter__(self): return self
        def __exit__(self, *args): self.wrapped.close()
        def __getitem__(self, key):
            if key not in FEATURE_KEYS:
                return blocked('central_endpoint_or_nonfeature_key')()
            reads[key] += 1
            return self.wrapped[key]

    def load(file, *args, **kwargs):
        try: path = Path(file).resolve()
        except TypeError: path = None
        if path is not None and path.name == 'endpoints.npz':
            return blocked('endpoint_NPZ')()
        value = original(file, *args, **kwargs)
        return FeatureView(value) if path == data.COHORT_NPZ.resolve() else value

    targets = ((data.Data, 'fit_labels'), (train, 'train_cell'), (train, 'efron_loss'),
               (train, 'risk_gradient_cache'), (torch.optim.Optimizer, '__init__'),
               (torch, 'save'), (parent, 'endpoints'))
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(np, 'load', load))
        for owner, name in targets:
            stack.enter_context(patch.object(owner, name, blocked(name)))
        for name in ('save', 'savez', 'savez_compressed', 'savetxt'):
            stack.enter_context(patch.object(np, name, blocked('np.'+name)))
        yield counters, reads


def geometry_choices(ds, registry):
    """Fixed min/lower-median/max geometry, never a prediction or endpoint."""
    rows = registry['H5_files']
    require(len(rows) == 520, 'original full520 private source registry required')
    counts = Counter()
    for row in rows:
        path = Path(row['path']); stat = path.stat()
        require((stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']),
                'original H5 metadata differs from the input audit')
        counts[int(row['owner'])] += int(row['selected_rows'])
    ordered = sorted((counts[int(patient)], int(patient)) for patient in ds.fit)
    require(len(ordered) >= 3 and all(n > 0 for n, _ in ordered), 'invalid fit input geometry counts')
    answer = [ordered[0], ordered[(len(ordered)-1)//2], ordered[-1]]
    require(len({p for _, p in answer}) == 3, 'geometry choice must contain three different fit patients')
    return answer


def aggregate_patient_diagnostics(rows):
    require(len(rows) == 3, 'exactly three fixed geometry examples required')
    result = {'patients': 3,
        'gradient_sensitivities': {key: moments([row['gradients'][key] for row in rows])
            for key in rows[0]['gradients'] if key != 'minimum_nonzero_gradient_threshold_enforced'},
        'risk_component_distributions': {key: moments([row['risk'][key] for row in rows])
            for key in rows[0]['risk']}, 'layers': []}
    for layer in range(2):
        layer_rows = [row['layers'][layer] for row in rows]
        item = {'round': layer+1}
        for name in ('path_equal_WSI', 'gene', 'routing_equal_WSI'):
            item[name] = {key: (moments([row[name][key] for row in layer_rows if row[name][key] is not None])
                if any(row[name][key] is not None for row in layer_rows) else None) for key in layer_rows[0][name]}
        for name in ('path_risk_correction', 'gene_risk_correction'):
            item[name] = moments([row[name] for row in layer_rows])
        result['layers'].append(item)
    return result


def _checkpoint(identity, seed, fold, arm):
    directory = runtime.cell_dir(seed, fold, arm)
    done = runtime.verify_done(directory, identity, seed, fold, arm)
    path = directory/'checkpoint.pt'
    record = torch.load(path, map_location='cpu', weights_only=False)
    require(record['source'] == identity and record['arm'] == arm and record['seed'] == seed
            and record['fold'] == fold and len(record['history']) == 80, 'checkpoint identity/history differs')
    require(runtime.sha(path) == done['checkpoint_sha256'], 'checkpoint changed while loaded')
    return path, done, record


def mechanism_summary(identity, seed, fold, arm, record):
    require(torch.cuda.is_available(), 'formal mechanism replay requires official CUDA spatial encoder')
    qualifications = runtime.verify_qualifications(identity)
    with no_labels_or_training() as (counters, reads):
        ds = data.Data(fold, seed, device='cpu', cache_slides=0)
        require(np.array_equal(ds.fit, record['fit']) and ds.prepared_sha256 == record['prepared_sha256'],
                'fit preprocessing or partition differs')
        # Existing prepared constructor materializes fit-only endpoint bytes;
        # delete them before geometry selection and never invoke their getter.
        del ds._fit_time, ds._fit_event
        external_offset_before = ds.base.clone()
        registry_path = runtime.PRIVATE/'H5_SOURCES_PRIVATE.json'
        registry = json.loads(registry_path.read_text())
        original_registry_sha = runtime.sha(registry_path)
        choices = geometry_choices(ds, registry)
        model = train.make_model(ds, arm, seed, fold)
        model.load_state_dict(record['model_state'], strict=True); model.eval()
        initial_weight_hash = train.model_hash(model.state_dict())
        wrong, wrong_audit = train._wrong_map(ds, ds.fit, 285000+seed*10+fold)
        wrong = np.asarray(wrong, np.int64)
        if wrong.shape == ds.fit.shape:
            mapping = dict(zip(map(int, ds.fit), map(int, wrong)))
        else:
            require(wrong.shape == (len(ds.patient_ids),), 'unexpected fit-only wrong-context mapping schema')
            mapping = {int(p): int(wrong[int(p)]) for p in ds.fit}
        require(all(j in set(map(int, ds.fit)) and p != j for p, j in mapping.items()),
                'fit-only donor mapping must remain partition-local and zero-self')
        rows, controls = [], []
        for _, patient in choices:
            raw = ds.molecular[patient].to('cuda').detach().requires_grad_()
            mask = ds.molecular_mask[patient].to('cuda')
            slides = ds.load_patient_slides(patient, device='cuda')
            encoded = model.encode(raw, mask, ds.gene_ids.to('cuda'), slides)
            row, out = diagnose_encoded(model, raw, encoded, ds.base[patient])
            with torch.no_grad():
                donor = mapping[patient]
                donor_encoded = train.encode_patient(model, ds, donor)
                donor_output = model.read_encoded(donor_encoded, ds.base[donor])
                bank = train._bank(donor_output, donor_encoded)
                off = model.read_encoded(encoded, ds.base[patient], interaction_off=True)
                constant = model.read_encoded(encoded, ds.base[patient], constant_path_value=True, constant_gene_value=True)
                require(torch.equal(off['risk'], out['base']), 'OFF must recover the same learned main effect exactly')
                base_key = scalar(out['base'])
                control = {'OFF_minus_MATCHED': scalar(off['risk']-out['risk']),
                           'CONSTANT_VALUES_minus_MATCHED': scalar(constant['risk']-out['risk'])}
                for key, gene, path in (('GENE_WRONG', True, False), ('PATH_WRONG', False, True), ('BOTH_WRONG', True, True)):
                    changed = model.read_encoded(encoded, ds.base[patient], context=train._foreign(bank, encoded, 'cuda', gene=gene, path=path))
                    require(scalar(changed['base']) == base_key, 'routing-only control changed the patient main effects')
                    control[key+'_minus_MATCHED'] = scalar(changed['risk']-out['risk'])
                    control[key+'_path_state_change_RMS'] = float(np.mean([
                        rms(a-b) for a, b in zip(changed['path_final'], out['path_final'])]))
                    control[key+'_gene_state_change_RMS'] = rms(changed['gene_final']-out['gene_final'])
                controls.append(control)
            rows.append(row)
            del out, encoded, raw, mask, slides, donor_encoded, donor_output, bank, off, constant
        require(not any(counters.values()), 'forbidden supervision or predictor-update API was used')
        require(train.model_hash(model.state_dict()) == initial_weight_hash, 'mechanism diagnostics changed fitted weights')
        require(torch.equal(ds.base, external_offset_before), 'mechanism diagnostics changed the frozen external offset')
        require(runtime.sha(registry_path) == original_registry_sha, 'source registry changed during diagnosis')
        public = aggregate_patient_diagnostics(rows)
        public.update(frozen_control_sensitivities={key: moments([r[key] for r in controls]) for key in controls[0]},
            source_registry_sha256=original_registry_sha,
            qualification_receipt_sha256=qualifications['qualification_receipt_sha256'],
            selection='fit_only_original_selected_Patch_count_min_lower_median_max;canonical_index_tiebreak',
            selected_Patch_counts_aggregate=moments([n for n, _ in choices]),
            read_or_control_aggregation='first_equal_WSI_within_patient_then_equal_three_geometry_examples',
            fit_geometry_examples_are_not_independent_validation=True,
            forbidden_API_call_counts=dict(counters), central_feature_array_read_counts=dict(reads),
            fit_endpoint_constructor_bytes_deserialized_then_deleted=True,
            fit_endpoint_getter_calls=0, valid_endpoint_reads=0, valid_patient_forward_calls=0,
            new_optimizer_steps=0, model_weights_unchanged=True,
            external_offset_unchanged=True,
            private_fit_donor_audit_uses_only_cancer_and_natural_RNA_missing_pattern=True,
            gradient_nonzero_is_not_DSS_increment_evidence=True,
            gradient_norm_depends_on_input_units_and_evidence_count=True,
            frozen_intervention_is_not_retrained_necessity_evidence=True,
            candidate_count_and_score_scale_can_change_null_mass=True,
            learned_null_is_not_calibrated_medical_confidence=True,
            latent_eta_is_not_risk_bound=True)
        del model, ds
        return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('history', 'mechanism'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--arm', choices=runtime.ARMS, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--execute-diagnostics', action='store_true')
    args = parser.parse_args()
    require(args.execute_diagnostics, 'explicit diagnostic execution flag required')
    identity, began, script_hash = runtime.source(), time.monotonic(), runtime.sha(__file__)
    require(args.source == identity, 'diagnostic source differs from the locked training closure')
    output_path = runtime.REPORT/'diagnostics'/f'seed{args.seed}_fold{args.fold}_{args.arm}_{args.mode}.json'
    require(not output_path.exists(), 'existing diagnostic receipt; silent repeat refused')
    checkpoint_path, done, record = _checkpoint(identity, args.seed, args.fold, args.arm)
    report = {'version': VERSION, 'status': 'PASS_POSTFIT_'+args.mode.upper()+'_DIAGNOSTICS',
        'mode': args.mode, 'source': identity, 'script_path': 'scripts/diagnose_phase28.py',
        'script_sha256': script_hash, 'seed': args.seed, 'fold': args.fold, 'arm': args.arm,
        'checkpoint_sha256': done['checkpoint_sha256'],
        'history': history_summary(record), 'valid_endpoint_reads': 0,
        'new_optimizer_steps': 0, 'hyperparameters_or_checkpoint_changed': False}
    if args.mode == 'mechanism':
        report['mechanism'] = mechanism_summary(identity, args.seed, args.fold, args.arm, record)
    require(runtime.source() == identity and runtime.sha(__file__) == script_hash
            and runtime.sha(checkpoint_path) == done['checkpoint_sha256'], 'diagnostic input/source bytes changed')
    report['wall_seconds'] = time.monotonic()-began
    runtime.atomic_json(output_path, report)
    print(json.dumps({'status': report['status'], 'seed': args.seed, 'fold': args.fold,
                      'arm': args.arm, 'mode': args.mode, 'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()

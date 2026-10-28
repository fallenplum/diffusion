"""Immutable-lock R5 qualification: real-data forward/backward, zero fit steps.

This script never creates an optimizer. It replays an existing inner F5 model,
qualifies the seven fresh readouts on the same real patients, and leaves all
R0--R4 report bytes unchanged. Private vectors are read but never exported.
"""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.bgb_en_r5_common import (parser, context, create_lock,
    qualified_prepared, bank, public_snapshot_guard, pairing, sha256, require)
from src.bgb_en.protocol import array_hash
from src.bgb_en.r5_protocol import canonical_sha256, verify_protocol_lock
from src.bgb_en.r5_readout import R5MechanismModel, RiskReadout
from src.bgb_en.r5_training import predict, state_hash, pure_survival_gradient_probe
from src.bgb_en.training import MechanismModel, predict as legacy_predict
from src.phase10_trimodal.reference import pairs


PUBLIC = ROOT/'reports/bgb_en_r5'
PUBLIC_NAMES = ('R5_0_QUALIFICATION.json', 'DERIVATIVE_CURVES.csv')


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def max_error(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(),
            'QUALIFICATION_RISK_SHAPE_OR_FINITE_GUARD')
    return float(np.max(np.abs(a-b)))


def credit_changes(a, b, time_values, event):
    left, right = pairs(np.asarray(time_values), np.asarray(event))
    def credit(risk):
        difference = np.asarray(risk)[left]-np.asarray(risk)[right]
        return np.where(np.abs(difference) <= 1e-8, .5, (difference > 0).astype(float))
    difference = credit(a)-credit(b)
    return dict(eligible_pairs=int(len(left)), changed_pair_credits=int(np.count_nonzero(difference)),
                total_absolute_credit_change=float(np.abs(difference).sum()))


def tensor_max(value):
    return float(value.detach().abs().max()) if value.numel() else 0.


def readout_options(config):
    return dict(rank=config['readout']['rank'], sham_rank=config['readout']['sham_rank'],
                directional_lambda=config['training']['directional_prior_L2'], extra_init_std=.02)


def build_fresh(arm, prepared, member, patch_bank, device, config):
    torch.manual_seed(31); np.random.seed(31)
    return R5MechanismModel(arm, prepared, member, patch_bank, device,
        gamma=config['readout']['gamma'], readout_options=readout_options(config))


def encode_once(model, index):
    """One actual upstream encoding for exact operator-level isolation tests."""
    path = model._path(index)
    omics = model.omics(**{key:value[index] for key,value in model.inputs.items()})
    return path, omics


def fixed_evidence_forward(model, index, path, omics, *, context=None, off=False):
    off = off or not (model.route_otop or model.route_ptoo)
    output = model.interaction(path['patch_states'], path['patch_mask'], omics['tokens'],
        omics['token_mask'], clinical=model.clinical[index], context_omics=context,
        interaction_off=off, return_details=True)
    readout = model.readout(output, path['patch_mask'], omics['token_mask'], interaction_off=off)
    return output, readout


def old_checkpoint(args, ctx, prepared):
    directory = args.primary_private/'cells/R3_outer1_inner0_F5_seed31'
    receipt, artifact = directory/'COMPLETE.json', directory/'model.pt'
    value = json.loads(receipt.read_text())
    require(value['status'] == 'COMPLETE' and value['lineage'] == ctx['lineage']
        and value['stage'] == 'R3' and value['arm'] == 'F5' and value['seed'] == 31
        and value['outer_fold'] == 1 and value['inner_partition'] == 0
        and value['epochs'] == 80 and value['outer_test_evaluated'] is False,
        'OLD_F5_IDENTITY_OR_SOURCE_CHANGED')
    require(sha256(artifact) == value['artifact_sha256'], 'OLD_F5_ARTIFACT_CHANGED')
    payload = torch.load(artifact, map_location='cpu', weights_only=False)
    require(payload['lineage'] == ctx['lineage'] and payload['seed'] == 31
        and payload['stage'] == 'R3' and payload['arm'] == 'F5'
        and np.array_equal(payload['train'], prepared['train'])
        and np.array_equal(payload['valid'], prepared['valid']), 'OLD_F5_PAYLOAD_SCOPE_CHANGED')
    require(80 in payload['snapshots'], 'OLD_F5_LOCKED_EPOCH80_MISSING')
    return payload['snapshots'][80], dict(receipt_sha256=sha256(receipt), artifact_sha256=sha256(artifact),
        epoch=80, same_train_valid_and_teacher=True, checkpoint_read_only=True)


def qualify_legacy(args, ctx, prepared, patch_bank, tolerance):
    snapshot, origin = old_checkpoint(args, ctx, prepared)
    batch = ctx['config']['training']['batch_size']; valid = prepared['valid']
    torch.manual_seed(31)
    old = MechanismModel('R3', 'F5', prepared, ctx['member'], patch_bank, args.device, gamma=.1)
    old.load_state_dict(snapshot['state_dict'], strict=True)
    replay, _ = legacy_predict(old, valid, batch)
    del old; gc.collect()
    new = build_fresh('A3', prepared, ctx['member'], patch_bank, args.device, ctx['config'])
    load = new.load_legacy_backbone(snapshot['state_dict'])
    current = predict(new, valid, batch)['risk']
    saved = np.asarray(snapshot['risk'], float)
    checks = []
    for name, first, second in (('old_forward_vs_saved', replay, saved),
                               ('new_legacy_vs_old_forward', current, replay),
                               ('new_legacy_vs_saved', current, saved)):
        error = max_error(first, second)
        ranking = credit_changes(first, second, ctx['raw']['t'][valid], ctx['raw']['e'][valid])
        require(error <= tolerance, 'LEGACY_REPLAY_RISK_TOLERANCE_FAILED_'+name)
        require(ranking['changed_pair_credits'] == 0, 'LEGACY_REPLAY_PAIR_CREDITS_CHANGED_'+name)
        checks.append(dict(comparison=name, maximum_risk_error=error, **ranking))
    del new, snapshot; gc.collect()
    return dict(status='PASS_FULL_INNER_VALIDATION_LEGACY_REPLAY', patients=len(valid),
        risk_tolerance=tolerance, changed_pair_credit_limit=0, original_checkpoint=origin,
        strict_backbone_load=load, comparisons=checks, saved_old_risks_reused_not_refitted=True)


def supported_profile(model, index, requested):
    if not requested:
        return dict(status='NOT_REQUESTED', complete_operator_FLOPs_claimed=False)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(model.device).startswith('cuda'):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(model.device)
    try:
        with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as profile:
            output = model(index, control='matched_query')
            if str(model.device).startswith('cuda'): torch.cuda.synchronize(model.device)
            del output
        supported = [dict(operator=row.key, calls=int(row.count), FLOPs=int(row.flops))
                     for row in profile.key_averages() if row.flops and row.flops > 0]
        return dict(status='SUPPORTED_OPERATOR_FLOPs_ONLY', supported_FLOPs=sum(x['FLOPs'] for x in supported),
                    patients=len(index), operators=supported, complete_operator_FLOPs_claimed=False,
                    shape_trace_or_patient_values_exported=False, same_FLOPs_claimed=False)
    except Exception as exc:
        # Runtime/CUPTI support is not evidence that the mathematical model failed.
        return dict(status='PROFILER_UNAVAILABLE', exception_type=type(exc).__name__,
                    complete_operator_FLOPs_claimed=False, same_FLOPs_claimed=False)


def qualify_fresh(args, ctx, prepared, patch_bank, tolerance, profile):
    indices = np.asarray(prepared['train'])[:8]
    index = torch.as_tensor(indices, device=args.device, dtype=torch.long)
    hashes, public_rows, values = {}, [], {}
    for arm in ('A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8'):
        model = build_fresh(arm, prepared, ctx['member'], patch_bank, args.device, ctx['config']); model.eval()
        before = state_hash(model.state_dict()); hashes[arm] = state_hash(model.state_dict(), backbone_only=True)
        with torch.no_grad():
            output = model(index, control='matched_query')
            off = model(index, control='interaction_off')
            require(torch.equal(off['cross_delta'], torch.zeros_like(off['cross_delta'])),
                    'INTERACTION_OFF_NOT_ZERO_'+arm)
            wrapper_main_error = tensor_max(output['main_delta']-off['main_delta'])
            wrapper_sham_error = tensor_max(output['sham_delta']-off['sham_delta'])
            require(max(wrapper_main_error, wrapper_sham_error) <= tolerance,
                    'REPEATED_WRAPPER_MAIN_REPLAY_TOLERANCE_FAILED_'+arm)
            path, omics = encode_once(model, index)
            _, fixed = fixed_evidence_forward(model, index, path, omics)
            _, fixed_off = fixed_evidence_forward(model, index, path, omics, off=True)
            require(torch.equal(fixed['main_delta'], fixed_off['main_delta'])
                    and torch.equal(fixed['sham_delta'], fixed_off['sham_delta']),
                    'FIXED_EVIDENCE_OFF_CHANGED_MAIN_EFFECT_'+arm)
            require(off['scalar_derivative'] is None or torch.equal(off['scalar_derivative'],
                    torch.zeros_like(off['scalar_derivative'])), 'OFF_SCALAR_DERIVATIVE_NOT_ZERO')
            values[arm] = {name: output[name].detach().cpu().numpy().copy()
                           for name in ('risk', 'cross_delta', 'main_delta', 'sham_delta')}
            extra = {}
            if arm == 'A7':
                joined = torch.cat((output['summary_omics_update'], output['summary_path_update']), -1)
                jac = model.readout.summary_jacobian(joined).norm(dim=-1)
                extra = dict(summary_jacobian_L2_mean=float(jac.mean()), summary_jacobian_L2_max=float(jac.max()),
                    scalar_z_derivative_not_defined=True)
            del output, off, path, omics, fixed, fixed_off
        gradient = None
        if arm in ('A4', 'A6'):
            gradient = pure_survival_gradient_probe(model, prepared, ctx['raw'], batch_size=8)
            cross_groups = gradient['groups_by_path']['cross_delta']
            for name in ('interaction.update_omics', 'interaction.update_path'):
                require(cross_groups.get(name, {}).get('nonzero', 0) > 0,
                        'PURE_COX_DIRECTION_GRADIENT_MISSING_'+arm+'_'+name)
            if arm == 'A6':
                for name in ('readout.a_o', 'readout.a_p'):
                    require(cross_groups.get(name, {}).get('nonzero', 0) > 0,
                            'PURE_COX_DIRECTIONAL_COEFFICIENT_GRADIENT_MISSING')
        flops = supported_profile(model, index, profile)
        require(state_hash(model.state_dict()) == before, 'QUALIFICATION_CHANGED_MODEL_WEIGHTS')
        public_rows.append(dict(arm=arm, seed=31, patients=len(indices), backbone_sha256=hashes[arm],
            parameter_budget=model.parameter_budget(), off_cross_exactly_zero=True,
            fixed_evidence_off_pooled_and_sham_exactly_unchanged=True,
            repeated_wrapper_main_maximum_error=wrapper_main_error,
            repeated_wrapper_sham_maximum_error=wrapper_sham_error, optimizer_updates=0,
            pure_Cox_gradient_probe=gradient, supported_operator_profile=flops, **extra))
        del model; gc.collect()
    require(len(set(hashes.values())) == 1, 'READOUT_ARMS_SHARED_INITIALIZATION_DIFFERS')
    comparisons = []
    for first, second in (('A7', 'A4'), ('A7', 'A3'), ('A8', 'A2')):
        errors = {name: max_error(values[first][name], values[second][name])
                  for name in ('risk', 'cross_delta', 'main_delta')}
        require(max(errors.values()) <= tolerance, 'FRESH_INITIALIZATION_PARITY_FAILED_'+first+'_'+second)
        comparisons.append(dict(first=first, second=second, maximum_errors=errors,
            scope='real_initial_batch_only_not_global_rank4_tanh_identity'))
    budgets = {row['arm']:row['parameter_budget'] for row in public_rows}
    require(budgets['A8']['requires_grad']-budgets['A7']['requires_grad'] == 7,
            'A8_WHOLE_MODEL_CAPACITY_MATCH_CHANGED')
    return dict(status='PASS_SEVEN_FRESH_READOUT_REAL_BATCHES', arm_count=7, patients_per_arm=len(indices),
        identical_backbone_initialization=True, backbone_sha256=next(iter(hashes.values())),
        rows=public_rows, initialization_comparisons=comparisons,
        A8_minus_A7_active_parameters=7, whole_model_FLOPs_equivalence_claimed=False)


def query_and_mask_qualification(args, ctx, prepared, patch_bank, tolerance):
    model = build_fresh('A4', prepared, ctx['member'], patch_bank, args.device, ctx['config']); model.eval()
    mapping, pair_audit = pairing(ctx, prepared, 0); model.query_map = mapping
    indices = np.asarray(prepared['train'])[:8]
    idx = torch.as_tensor(indices, device=args.device, dtype=torch.long)
    original_offset = model.offset.clone(); before = state_hash(model.state_dict())
    with torch.no_grad():
        path, omics = encode_once(model, idx)
        source_idx = torch.as_tensor(mapping[indices], device=args.device)
        source = model.omics(**{k:v[source_idx] for k,v in model.inputs.items()})
        own_tokens, own_mask = omics['tokens'].clone(), omics['token_mask'].clone()
        own_path, own_path_mask = path['patch_states'].clone(), path['patch_mask'].clone()
        correct_details, correct = fixed_evidence_forward(model, idx, path, omics)
        wrong_details, wrong = fixed_evidence_forward(model, idx, path, omics,
            context=(source['tokens'], source['token_mask']))
        for name in ('pool_p', 'pool_o', 'main_delta'):
            require(torch.equal(correct[name], wrong[name]), 'WRONG_QUERY_CHANGED_MAIN_EFFECT')
        for name in ('path_initial', 'omics_initial'):
            require(torch.equal(correct_details[name], wrong_details[name]),
                    'WRONG_QUERY_REPLACED_OWN_KEY_VALUE')
        require(torch.equal(own_tokens, omics['tokens']) and torch.equal(own_mask, omics['token_mask'])
                and torch.equal(own_path, path['patch_states']) and torch.equal(own_path_mask, path['patch_mask']),
                'QUERY_CONTROL_MODIFIED_FIXED_OWN_EVIDENCE_OR_MASK')
        attention = wrong_details['trace'][0]['omics_to_patch']['attention']
        absent = ~source['token_mask'][:, None, :, None].expand_as(attention)
        require(tensor_max(attention[absent]) == 0, 'WRONG_QUERY_SOURCE_MASK_IGNORED')
        wrong_change = float((correct['cross_delta']-wrong['cross_delta']).abs().max())
        wrapper_correct = model(idx, control='matched_query'); wrapper_wrong = model(idx, control='wrong_query')
        wrapper_main_error = tensor_max(wrapper_correct['main_delta']-wrapper_wrong['main_delta'])
        wrapper_pool_error = max(tensor_max(wrapper_correct[name]-wrapper_wrong[name]) for name in ('pool_p','pool_o'))
        require(max(wrapper_main_error, wrapper_pool_error) <= tolerance,
                'REPEATED_WRAPPER_QUERY_MAIN_REPLAY_TOLERANCE_FAILED')
        del correct, wrong, correct_details, wrong_details, source, attention, path, omics
        del own_tokens, own_mask, own_path, own_path_mask, wrapper_correct, wrapper_wrong
        mean, source_mask = model.refresh_training_query_mean(batch=16)
        all_tokens, all_masks = [], []
        for start in range(0, len(prepared['train']), 16):
            part = torch.as_tensor(prepared['train'][start:start+16], device=args.device)
            out = model.omics(**{k:v[part] for k,v in model.inputs.items()})
            all_tokens.append(out['tokens']); all_masks.append(out['token_mask'])
        tokens, valid = torch.cat(all_tokens), torch.cat(all_masks)
        expected = torch.where(valid[..., None], tokens, torch.zeros_like(tokens)).sum(0, keepdim=True)
        expected = expected/valid.sum(0, keepdim=True).clamp_min(1)[..., None]
        mean_error = float((mean-expected).abs().max())
        require(mean_error <= tolerance and torch.equal(source_mask, valid.any(0, keepdim=True)),
                'MEAN_QUERY_NOT_CURRENT_TRAIN_MASKED_MEAN')
        fixed_mean, fixed_mask = mean.clone(), source_mask.clone()
        first = model(idx, control='mean_query')['risk']; second = model(idx, control='mean_query')['risk']
        mean_wrapper_error = tensor_max(first-second)
        require(mean_wrapper_error <= tolerance and torch.equal(model.context_mean, fixed_mean)
                and torch.equal(model.context_mean_mask, fixed_mask), 'MEAN_REFERENCE_CHANGED_DURING_CACHE_PASSES')
        del all_tokens, all_masks, tokens, valid, expected, first, second, fixed_mean, fixed_mask
    # Actual measured RNA and projected Patch states, without optimization.
    rna = model.inputs['rna']; rna.requires_grad_(True)
    output = model(idx, control='matched_query')
    projected = output['interaction_details']['path_initial']
    gene_grad, patch_grad = torch.autograd.grad(output['cross_delta'].sum(), (rna, projected), allow_unused=True)
    require(gene_grad is not None and patch_grad is not None, 'REAL_INPUT_GRADIENT_PATH_MISSING')
    measured = model.inputs['rna_mask'].bool(); patch_mask = patch_bank.mask[idx]
    missing_max = tensor_max(gene_grad[~measured]); padded_max = tensor_max(patch_grad[~patch_mask])
    require(missing_max == 0 and padded_max == 0, 'MASKED_REAL_INPUT_HAS_GRADIENT')
    require(torch.isfinite(gene_grad).all() and torch.isfinite(patch_grad).all()
            and gene_grad[measured].abs().sum() > 0 and patch_grad[patch_mask].abs().sum() > 0,
            'REAL_OBSERVED_INPUT_GRADIENT_NOT_FINITE_NONZERO')
    other = torch.ones(len(rna), device=args.device, dtype=torch.bool); other[idx] = False
    require(tensor_max(gene_grad[other]) == 0, 'MATCHED_QUERY_READ_ANOTHER_PATIENT_RNA')
    rna.requires_grad_(False)
    require(torch.equal(model.offset, original_offset) and state_hash(model.state_dict()) == before,
            'QUERY_OR_MASK_QUALIFICATION_CHANGED_ANCHOR_OR_WEIGHTS')
    result = dict(status='PASS_REAL_QUERY_VALUE_AND_MASK_ISOLATION', patients=len(indices),
        pairing=pair_audit, wrong_source_masks_used=True, own_initial_evidence_unchanged=True,
        exact_isolation_scope='one_real_upstream_encoding_then_fixed_tensor_interaction_readout_controls',
        own_CP_RNA_offset_unchanged=True, wrong_query_cross_maximum_change=wrong_change,
        repeated_wrapper_query_main_maximum_error=wrapper_main_error,
        repeated_wrapper_query_pool_maximum_error=wrapper_pool_error,
        repeated_wrapper_mean_risk_maximum_error=mean_wrapper_error,
        independent_train_masked_mean_maximum_error=mean_error,
        query_mean_source_mask_equal_training_support=True, mean_detached=True,
        mean_constant_during_repeated_forward=True, mean_training_rows=len(prepared['train']),
        mean_training_hash=array_hash(prepared['train']),
        input_gradient_target='sum_cross_delta_without_survival_labels_separate_from_full_Cox_probes',
        missing_RNA_gradient_maximum=missing_max, padded_projected_patch_gradient_maximum=padded_max,
        observed_RNA_gradient_L1=float(gene_grad[measured].abs().sum()),
        observed_projected_patch_gradient_L1=float(patch_grad[patch_mask].abs().sum()),
        matched_other_patient_RNA_gradient_exactly_zero=True,
        actual_missing_RNA_entries_in_probe=int((~measured[idx]).sum()),
        actual_padded_projected_patch_entries_in_probe=int((~patch_mask).sum()), optimizer_updates=0)
    del model, output, gene_grad, patch_grad; gc.collect()
    return result


def derivative_csv(path, config):
    grid = config['diagnostics']['derivative_grid']
    z = torch.linspace(grid['minimum'], grid['maximum'], grid['points'], dtype=torch.float64)
    modules = {name:RiskReadout(name).double() for name in ('legacy', 'linear', 'temp4', 'directional')}
    rows = []
    do, dp = modules['directional'].directional_derivatives(z)
    for i, point in enumerate(z.tolist()):
        row = dict(z=point)
        for name in ('legacy', 'linear', 'temp4'):
            row[name+'_cross_risk'] = float(modules[name].scalar_response(z)[i])
            row[name+'_derivative'] = float(modules[name].scalar_derivative(z)[i])
        row.update(directional_initial_d_zo=float(do[i]), directional_initial_d_zp=float(dp[i]))
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return dict(points=len(rows), minimum=float(z[0]), maximum=float(z[-1]),
        common_initial_scalar_slope=.05, legacy_absolute_cross_bound=.05,
        temp4_absolute_cross_bound=.2, linear_finite_absolute_bound=None,
        directional_scalar_coordinates_separate=True, rank4_scalar_z_curve_not_defined=True)


def run(args):
    before = public_snapshot_guard(); started = time.monotonic()
    ctx = context(args, locked=False)
    lock = create_lock(args, ctx); ctx['lock'] = lock
    verify_protocol_lock(lock, ROOT)
    private = args.private/'QUALIFICATION.json'
    if private.exists():
        previous = json.loads(private.read_text())
        require(args.resume, 'R5_QUALIFICATION_EXISTS_USE_RESUME')
        require(previous['status'] == 'PASS_R5_REAL_DATA_AND_LEGACY_PARITY'
                and previous['contract_sha256'] == lock['contract_sha256'], 'EXISTING_QUALIFICATION_NOT_PASS_OR_CHANGED')
        require(all(sha256(PUBLIC/name) == digest for name,digest in previous['public_artifact_sha256'].items()),
                'QUALIFICATION_PUBLIC_ARTIFACT_CHANGED')
        legacy_directory = args.primary_private/'cells/R3_outer1_inner0_F5_seed31'
        require(sha256(legacy_directory/'model.pt') == previous['legacy_artifact_sha256']
                and sha256(legacy_directory/'COMPLETE.json') == previous['legacy_receipt_sha256'],
                'QUALIFIED_LEGACY_CHECKPOINT_CHANGED')
        require(before == public_snapshot_guard(), 'UPSTREAM_REPORT_BYTES_CHANGED')
        return dict(status=previous['status'], contract_sha256=lock['contract_sha256'], resumed=True,
                    new_optimizer_updates=0, new_formal_fits=0)
    require(not (args.private/'QUALIFICATION_ERROR.private.json').exists(),
            'OLD_QUALIFICATION_ERROR_REQUIRES_EXPLICIT_REVIEW')
    write_new(args.private/'QUALIFICATION_STARTED.private.json', dict(status='REAL_QUALIFICATION_STARTED',
        contract_sha256=lock['contract_sha256'], optimizer_updates=0, new_formal_fits=0))
    prepared, prepared_audit = qualified_prepared(args, ctx, 0)
    tolerance = ctx['config']['training']['saved_risk_replay_max_error']
    require(tolerance == 2e-6, 'R5_FIXED_REPLAY_TOLERANCE_CHANGED')
    torch.set_num_threads(ctx['config']['training']['threads_per_process'])
    patch_bank = bank(ctx, args.device)
    if str(args.device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(args.device)
    legacy = qualify_legacy(args, ctx, prepared, patch_bank, tolerance)
    fresh = qualify_fresh(args, ctx, prepared, patch_bank, tolerance, args.profile)
    masks = query_and_mask_qualification(args, ctx, prepared, patch_bank, tolerance)
    curves = derivative_csv(PUBLIC/'DERIVATIVE_CURVES.csv', ctx['config'])
    after = public_snapshot_guard()
    require(before == after, 'UPSTREAM_REPORT_BYTES_CHANGED')
    report = dict(schema='bgb_en_r5_zero_fit_qualification_v1', status='PASS_R5_REAL_DATA_AND_LEGACY_PARITY',
        contract_sha256=lock['contract_sha256'], qualification_source_sha256=sha256(Path(__file__)),
        prepared_reuse=prepared_audit, legacy_parity=legacy, fresh_readouts=fresh,
        query_and_mask_qualification=masks, derivative_curves=curves,
        preserved_upstream_reports=dict(files=len(before), manifest_sha256=canonical_sha256(before), unchanged=True),
        floating_point_boundary='fixed-evidence semantic equality is exact; repeated upstream CUDA index_add encodings and full legacy saved-risk replay use the prelocked 2e-6 risk gate, with zero changed saved pair credits; formal gradient cache remains 2e-5',
        device_kind='cuda' if str(args.device).startswith('cuda') else 'cpu',
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0,
        elapsed_seconds=time.monotonic()-started, new_teacher_fits=0,
        new_formal_fits=0, optimizer_updates=0, outer_test_accessed=False,
        patient_identifiers_risks_coordinates_or_weights_exported=False)
    write_new(PUBLIC/'R5_0_QUALIFICATION.json', report)
    receipt = dict(status=report['status'], contract_sha256=lock['contract_sha256'],
        public_artifact_sha256={name:sha256(PUBLIC/name) for name in PUBLIC_NAMES},
        legacy_artifact_sha256=legacy['original_checkpoint']['artifact_sha256'],
        legacy_receipt_sha256=legacy['original_checkpoint']['receipt_sha256'],
        qualification_source_sha256=sha256(Path(__file__)), new_formal_fits=0, optimizer_updates=0,
        elapsed_seconds=report['elapsed_seconds'])
    write_new(private, receipt)
    return dict(status=report['status'], contract_sha256=lock['contract_sha256'],
                new_formal_fits=0, optimizer_updates=0, legacy_validation_rows=legacy['patients'], fresh_arms=7)


def main():
    command = parser(__doc__)
    command.add_argument('--profile', action='store_true', help='Record profiler-supported real-batch FLOPs only')
    args = command.parse_args()
    args.private.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:
        destination = args.private/'QUALIFICATION_ERROR.private.json'
        if not destination.exists():
            lock_path = args.private/'LOCK.json'
            failed_contract = json.loads(lock_path.read_text()).get('contract_sha256') if lock_path.is_file() else None
            write_new(destination, dict(status='FAILED_R5_QUALIFICATION_NO_AUTOMATIC_RETRY',
                exception_type=type(exc).__name__, traceback=traceback.format_exc(),
                contract_sha256=failed_contract, qualification_source_sha256=sha256(Path(__file__)),
                assets_config_sha256=sha256(args.assets) if args.assets.is_file() else None,
                optimizer_updates=0, new_formal_fits=0))
        print(json.dumps(dict(status='FAILED_R5_QUALIFICATION', exception_type=type(exc).__name__,
                              optimizer_updates=0, new_formal_fits=0)), flush=True)
        raise SystemExit(1)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()

"""Zero-optimizer R3 final80 mechanism replay, separate from frozen training.

Use --readcore during active training to inventory completed checkpoints without
constructing a CUDA model. Actual diagnostics use the first eight validation
patients per cell by default; full-validation performance uses saved risks only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# F7 runtime-repaired checkpoints require an explicit separate process. This
# opt-in is handled before importing torch; non-F7 arms cannot share this flag.
F7_REPAIR_BACKEND = None
if '--runtime-repair' in sys.argv:
    from scripts.bgb_en_f7_determinism_diagnostic import configure_backend
    F7_REPAIR_BACKEND = configure_backend('deterministic')

import numpy as np
import torch
from scripts import bgb_en_run as driver
from scripts.bgb_en_anchor_cast_audit import comparable, censor_weights
from src.bgb_en.data import read_patch_cache
from src.bgb_en.mechanism_audit import (attention_change, attention_summary, bounded_components,
    independent_efron_gradient, loss_chain_probe, statistics)
from src.bgb_en.protocol import patient_mismatch
from src.bgb_en.training import MechanismModel, PatchBank


def metrics_from_saved(time, event, train_time, train_event, risk, reference):
    left, right = comparable(time, event)
    weights = censor_weights(train_time, train_event, time, event)
    def credit(value):
        margin = np.asarray(value, float)[left]-np.asarray(value, float)[right]
        return np.where(np.abs(margin) <= 1e-8, .5, margin > 0)
    value, baseline = credit(risk), credit(reference)
    diff = value-baseline
    hd, ud = len(left), float(weights[left].sum())
    return dict(Harrell=float(value.sum()/hd), Uno60=float(np.dot(weights[left], value)/ud),
                Harrell_num=float(value.sum()), Harrell_den=hd,
                Uno_num=float(np.dot(weights[left], value)), Uno_den=ud,
                repair=float(np.maximum(diff, 0).sum()/hd), damage=float(np.maximum(-diff, 0).sum()/hd),
                Uno_repair=float(np.dot(weights[left], np.maximum(diff, 0))/ud),
                Uno_damage=float(np.dot(weights[left], np.maximum(-diff, 0))/ud))


def grouped_mechanisms(results, fields):
    """Descriptive cell summaries; never turn subset gradients into cohort ones."""
    groups = {}
    for result in results:
        identity = result['identity']
        key = tuple(identity[field] for field in fields)
        groups.setdefault(key, []).append(result)
    output = []
    for key, rows in sorted(groups.items()):
        summary = dict(zip(fields, key))
        summary['cells'] = len(rows)
        for name in ('interaction', 'OtoP_risk_raw', 'PtoO_risk_raw', 'pooled_path', 'pooled_omics'):
            summary[name+'_SD_across_cells'] = statistics(np.array([row['amplitude'][name]['SD'] for row in rows]))
        summary['delta_to_offset_SD_ratio_across_cells'] = statistics(np.array([row['amplitude']['trained_delta_SD_over_offset_SD'] for row in rows]))
        summary['mechanism_patient_scope'] = dict(
            diagnostic_patients_per_cell=statistics(np.asarray([row['diagnostic_validation_patients'] for row in rows])),
            boundary='first <=8 saved validation rows per cell; repeated fits are not independent patient cohorts')
        summary['pure_interaction_loss_gradients'] = {}
        parameter_groups = dict(
            routing_kernel=('path_match.', 'omics_match.', 'path_value.', 'omics_value.',
                            'update_path.', 'update_omics.', 'null_path.', 'null_omics.', 'gate_path.', 'gate_omics.'),
            interaction_input_projection=('patch_project.', 'omics_project.'),
            directional_readout=('otop_head.', 'ptoo_head.'))
        for probe in ('interaction', 'OtoP_drop_one', 'PtoO_drop_one'):
            entries = [row['loss_chain_gradients'][probe] for row in rows]
            grad_summary = dict(cells=len(entries),
                all_exclude_regularization=all(entry['regularization_included'] is False for entry in entries),
                all_exclude_pooled_main_effects=all(entry['pooled_main_effects_included'] is False for entry in entries),
                input_gradients={}, parameter_groups={})
            for input_name in ('RNA_input', 'patch_input'):
                available = [entry['input_gradients'][input_name] for entry in entries]
                grad_summary['input_gradients'][input_name] = dict(
                    gradient_L2_across_cells=statistics(np.asarray([entry['gradient_L2'] for entry in available])),
                    connected_cells=sum(entry['connected'] for entry in available),
                    nonzero_cells=sum(entry['gradient_L2'] > 0 for entry in available))
            for group, prefixes in parameter_groups.items():
                per_cell = []
                for entry in entries:
                    parameters = [record for record in entry['parameter_gradients']
                                  if record['name'].startswith(tuple('interaction.'+prefix for prefix in prefixes))]
                    per_cell.append(dict(
                        gradient_L2=float(np.sqrt(sum(record['gradient_L2']**2 for record in parameters))),
                        parameter_tensors=len(parameters), parameter_elements=sum(record['parameters'] for record in parameters),
                        connected_tensors=sum(record['connected'] for record in parameters),
                        nonzero_tensors=sum(record['gradient_L2'] > 0 for record in parameters)))
                grad_summary['parameter_groups'][group] = {
                    name+'_across_cells': statistics(np.asarray([record[name] for record in per_cell]))
                    for name in ('gradient_L2', 'parameter_tensors', 'parameter_elements', 'connected_tensors', 'nonzero_tensors')}
                grad_summary['parameter_groups'][group]['cells_with_nonzero_gradient'] = sum(record['gradient_L2'] > 0 for record in per_cell)
            # Keep the exclusion itself auditable. Pooled-head gradients never
            # contribute to the routing/interface/readout norms above.
            pooled = [[record for record in entry['parameter_gradients']
                       if record['name'].startswith(('interaction.pooled_path_head.', 'interaction.pooled_omics_head.'))]
                      for entry in entries]
            grad_summary['excluded_pooled_head_check'] = dict(
                connected_tensors_across_cells=statistics(np.asarray([sum(record['connected'] for record in records) for records in pooled])),
                gradient_L2_across_cells=statistics(np.asarray([np.sqrt(sum(record['gradient_L2']**2 for record in records)) for records in pooled])))
            grad_summary['boundary'] = 'frozen full-validation Efron dLoss/dr times first <=8 network paths; not a full-validation parameter/input gradient'
            summary['pure_interaction_loss_gradients'][probe] = grad_summary
        summary['readout_derivative_and_amplitude'] = {}
        for name in ('bounded_interaction_readout_derivative', 'bounded_pooled_path_readout_derivative',
                     'bounded_pooled_omics_readout_derivative'):
            values = [row['amplitude'][name]['mean'] for row in rows]
            summary['readout_derivative_and_amplitude'][name+'_cell_mean'] = statistics(np.asarray(values))
        for name in ('interaction_raw_abs_gt2_fraction', 'interaction_readout_derivative_below_one_percent_max_fraction',
                     'pooled_path_raw_abs_gt2_fraction', 'pooled_omics_raw_abs_gt2_fraction'):
            summary['readout_derivative_and_amplitude'][name+'_across_cells'] = statistics(np.asarray([row['amplitude'][name] for row in rows]))
        summary['last_layer_attention'] = {}
        for direction in ('omics_to_patch', 'path_to_omics'):
            available = []
            for row in rows:
                layers = [value for value in row['attention'].values() if value['direction'] == direction]
                if layers:
                    available.append(max(layers, key=lambda value: value['layer']))
            direction_summary = dict(cells_with_direction=len(available), cells_without_direction=len(rows)-len(available))
            for name in ('attention_entropy_including_null', 'relative_effective_keys', 'relative_effective_key_fraction',
                         'query_key_effective_rank', 'null_mass', 'nonnull_mass', 'gate', 'message_L2'):
                values = [layer['matched'][name]['mean'] for layer in available
                          if layer['matched'].get(name) is not None and layer['matched'][name]['mean'] is not None]
                direction_summary[name+'_cell_mean'] = statistics(np.asarray(values))
            for name in ('gate_fraction_below_005', 'gate_fraction_above_095'):
                values = [layer['matched'][name] for layer in available if layer['matched'].get(name) is not None]
                direction_summary[name+'_across_cells'] = statistics(np.asarray(values))
            direction_summary['valid_query_rows_across_cells'] = statistics(np.asarray([layer['matched']['valid_query_rows'] for layer in available]))
            summary['last_layer_attention'][direction] = direction_summary
            for control in ('wrong', 'mean', 'value_fixed_attention_permutation'):
                values = []
                for layer in available:
                    if control in layer['controls']:
                        value = layer['controls'][control]['TV']['mean']
                        if value is not None:
                            values.append(value)
                summary[direction+'_'+control+'_last_layer_TV_across_cells'] = statistics(np.asarray(values))
        output.append(summary)
    return output


def replay_cell(model, prepared, raw, checkpoint, sample_limit):
    valid, train = np.asarray(prepared['valid']), np.asarray(prepared['train'])
    sampled = valid[:sample_limit]
    idx = torch.as_tensor(sampled, device=model.device)
    # Explicit differentiable raw-input leaves exclude anchor and direct RNA
    # score paths from all RNA/patch interaction sensitivity measurements.
    bank = model.patch_bank
    patch_input = bank.x[idx].detach().clone().requires_grad_(True)
    rna_input = model.inputs['rna'][idx].detach().clone().requires_grad_(True)
    xy, mask, regions = bank.xy[idx], bank.mask[idx], bank.region_ids[idx]
    geometry = model.path.geometry(xy, mask, regions)
    path = model.path(patch_input, xy, mask, region_ids=regions, geometry=geometry, return_details=True)
    omics_args = {key: value[idx] for key, value in model.inputs.items()}
    omics_args['rna'] = rna_input
    omics = model.omics(**omics_args)
    own = dict(patch_states=path['patch_states'], patch_mask=path['patch_mask'],
               omics_tokens=omics['tokens'], omics_mask=omics['token_mask'], clinical=model.clinical[idx])
    model.training_query_mean()
    other = torch.as_tensor(model.query_map[sampled], device=model.device)
    with torch.no_grad():
        foreign = model.omics(**{key: value[other] for key, value in model.inputs.items()})
    contexts = dict(wrong=(foreign['tokens'], foreign['token_mask']),
                    mean=(model.context_mean.expand(len(idx), -1, -1), omics['token_mask']))
    matched = model.interaction(**own, return_details=True)
    alternatives = {}
    with torch.no_grad():
        for mode in ('wrong', 'mean'):
            alternatives[mode] = model.interaction(**own, context_omics=contexts[mode], return_details=True)
        alternatives['value_fixed_attention_permutation'] = model.interaction(**own, permute_attention=True, return_details=True)
        alternatives['interaction_off'] = model.interaction(**own, interaction_off=True, return_details=True)
    actual_output = alternatives['wrong'] if model.intervention == 'wrong_query' else matched
    component = bounded_components(matched, model.gamma)
    actual_component = bounded_components(actual_output, model.gamma)
    delta = actual_component['pooled_path']+actual_component['pooled_omics']+actual_component['interaction']
    reconstructed = model.offset[idx]+delta
    final = checkpoint['snapshots'][80]
    saved = np.asarray(final['risk'])[:len(sampled)]
    replay_error = float(np.max(np.abs(reconstructed.detach().cpu().numpy().astype(float)-saved)))
    if replay_error > 2e-5:
        raise ValueError('Final80 risk replay mismatch on the fixed validation subset: '+str(replay_error))
    amplitude = {name: statistics(value) for name, value in component.items()}
    for name in ('pooled_path_risk', 'pooled_omics_risk', 'OtoP_risk', 'PtoO_risk'):
        amplitude[name+'_raw'] = statistics(matched[name])
    offset = model.offset[idx]
    amplitude['trained_model_delta'] = statistics(delta)
    amplitude['offset'] = statistics(offset)
    raw_cross = matched['OtoP_risk']+matched['PtoO_risk']
    relative_derivative = 1-torch.tanh(raw_cross).square()
    amplitude['bounded_interaction_readout_derivative'] = statistics(model.gamma*.5*relative_derivative)
    amplitude['interaction_readout_derivative_below_one_percent_max_fraction'] = float((relative_derivative.detach() < .01).double().mean())
    amplitude['interaction_raw_abs_gt2_fraction'] = float((raw_cross.detach().abs() > 2).double().mean())
    for name in ('pooled_path', 'pooled_omics'):
        raw_risk = matched[name+'_risk']
        amplitude['bounded_'+name+'_readout_derivative'] = statistics(model.gamma*.25*(1-torch.tanh(raw_risk).square()))
        amplitude[name+'_raw_abs_gt2_fraction'] = float((raw_risk.detach().abs() > 2).double().mean())
    amplitude['trained_delta_SD_over_offset_SD'] = float(delta.detach().std(unbiased=False)/offset.std(unbiased=False).clamp_min(1e-12))
    attention = {}
    for layer_index, trace in enumerate(matched['trace']):
        for direction, row in trace.items():
            key_count = mask.sum(-1)[:, :, None] if direction == 'omics_to_patch' else omics['token_mask'].sum(-1)[:, None, None]
            record = dict(layer=layer_index, direction=direction, matched=attention_summary(row, key_count), controls={})
            for name, alternative in alternatives.items():
                if len(alternative['trace']) > layer_index and direction in alternative['trace'][layer_index]:
                    record['controls'][name] = attention_change(row, alternative['trace'][layer_index][direction])
            attention[f'layer{layer_index}_{direction}'] = record
    interventions = {}
    matched_delta = component['pooled_path']+component['pooled_omics']+component['interaction']
    for name, output in alternatives.items():
        changed = bounded_components(output, model.gamma)
        changed_delta = changed['pooled_path']+changed['pooled_omics']+changed['interaction']
        interventions[name] = dict(delta_difference_from_matched=statistics(changed_delta-matched_delta),
                                   interaction_difference_from_matched=statistics(changed['interaction']-component['interaction']),
                                   pooled_path_exactly_unchanged=bool(torch.equal(matched['pooled_path_risk'], output['pooled_path_risk'])),
                                   pooled_omics_exactly_unchanged=bool(torch.equal(matched['pooled_omics_risk'], output['pooled_omics_risk'])))
    incoming = independent_efron_gradient(raw['t'][valid], raw['e'][valid], np.asarray(final['risk']))
    gradients = {name: loss_chain_probe(component[name], incoming['gradient'][:len(sampled)], model,
                                      rna_input, patch_input, retain_graph=True)
                 for name in ('interaction', 'OtoP_drop_one', 'PtoO_drop_one')}
    # Actual shared-kernel route gradients for the raw directional scores are
    # also reported when the common tanh suppresses otherwise active paths.
    gradients['OtoP_raw'] = loss_chain_probe(matched['OtoP_risk'], incoming['gradient'][:len(sampled)], model, rna_input, patch_input)
    gradients['PtoO_raw'] = loss_chain_probe(matched['PtoO_risk'], incoming['gradient'][:len(sampled)], model, rna_input, patch_input)
    all_saved = dict(final80_trained=np.asarray(final['risk']))
    all_saved.update({name: np.asarray(value['risk']) for name, value in checkpoint.get('frozen_controls', {}).items()})
    performance = {name: metrics_from_saved(raw['t'][valid], raw['e'][valid], raw['t'][train], raw['e'][train], risk, prepared['base'][valid])
                   for name, risk in all_saved.items()}
    return dict(status='PASS_FINAL80_MECHANISM_REPLAY', optimizer_updates=0,
        diagnostic_validation_patients=len(sampled), full_validation_patients=len(valid),
        diagnostic_subset_rule=f'first min({sample_limit},validation_n) patients in the saved validation row order; no label-based selection',
        diagnostic_subset_has_survival_events=int(raw['e'][sampled].sum()),
        sampled_final80_risk_replay_max_abs=replay_error, amplitude=amplitude,
        attention=attention, frozen_intervention_responses=interventions,
        loss_chain_gradients=gradients,
        gradient_probe_loss=dict(full_validation_Efron=incoming['loss'], risk_gradient_sum=incoming['gradient_sum'],
                                 full_validation_gradient=True, sampled_network_paths_only=True, regularization_included=False,
                                 query_pairing_for_gradient='own-patient matched query, including separately labeled wrong-query trained arms'),
        full_validation_saved_prediction_performance=performance,
        paired_performance_boundary='full validation saved risks; not extrapolated from the first-eight mechanism subset',
        trained_default_query='wrong-patient' if model.intervention == 'wrong_query' else 'matched-patient',
        directional_allocation_boundary='shared tanh numeric Shapley allocation is not causal or biological attribution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('audit',), default='audit')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--prepared-private', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--patient-limit', type=int, default=8)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--arm', action='append')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--readcore', action='store_true', help='inventory/verify only; no neural forward or CUDA allocation')
    parser.add_argument('--runtime-repair', action='store_true', help='explicit deterministic backend for repaired F7 only; requires --arm F7')
    args = parser.parse_args()
    if args.runtime_repair and args.arm != ['F7']:
        raise ValueError('--runtime-repair is restricted to a separate --arm F7 audit process')
    if args.patient_limit < 1 or args.patient_limit > 8:
        raise ValueError('Fixed mechanism subset is limited to the first 1..8 validation patients')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    assets, config, raw, member, core_lineage = driver.load_inputs(args)
    audit_sources = {name: driver.sha256(ROOT/name) for name in ('src/bgb_en/mechanism_audit.py',
        'scripts/bgb_en_mechanism_replay.py', 'scripts/bgb_en_anchor_cast_audit.py')}
    cells = []
    for complete in (args.private/'cells').glob('R3_*/COMPLETE.json'):
        receipt = json.loads(complete.read_text())
        if receipt['arm'] in ('F0', 'F1') or (args.arm and receipt['arm'] not in args.arm):
            continue
        cells.append((receipt, complete))
    cells.sort(key=lambda row: (int(row[0]['arm'].split('_')[0][1:]), row[0]['inner_partition'], row[0]['seed']))
    if args.limit is not None:
        cells = cells[:args.limit]
    results, inventory, banks = [], [], {}
    for receipt, complete in cells:
        cell = complete.parent
        artifact = cell/'model.pt'
        if driver.sha256(artifact) != receipt['artifact_sha256']:
            raise ValueError('Completed checkpoint artifact SHA differs')
        checkpoint = torch.load(artifact, map_location='cpu', weights_only=False)
        runtime_path = cell/'RUNTIME_REPAIR_COMPLETE.json'
        if receipt['arm'] == 'F7' and runtime_path.exists() and not args.runtime_repair:
            raise ValueError('Repaired F7 requires a separate --runtime-repair --arm F7 replay; other arms must retain their inherited runtime')
        if args.runtime_repair:
            runtime = json.loads(runtime_path.read_text())
            if (runtime['status'] != 'COMPLETE_F7_DETERMINISTIC_FRESH_FIT'
                or runtime['artifact_sha256'] != receipt['artifact_sha256']
                or runtime['core_receipt_sha256'] != driver.sha256(complete)
                or any(driver.sha256(ROOT/name) != digest for name, digest in runtime['sources'].items())
                or runtime['backend'] != F7_REPAIR_BACKEND):
                raise ValueError('F7 runtime repair source/backend/artifact receipt differs')
        for key, value in core_lineage.items():
            if checkpoint['lineage'][key] != value:
                raise ValueError('Checkpoint frozen core lineage differs at '+key)
        if 80 not in checkpoint.get('snapshots', {}) or checkpoint['epochs'] != 80:
            raise ValueError('This audit requires fixed final80 checkpoints')
        partition = int(receipt['inner_partition'])
        root = args.private if args.prepared_private is None else args.prepared_private
        prepared_path = root/'prepared'/f"outer{receipt['outer_fold']}_inner{partition}.pt"
        prepared = torch.load(prepared_path, map_location='cpu', weights_only=False)
        if not (np.array_equal(prepared['valid'], checkpoint['valid']) and np.array_equal(prepared['train'], checkpoint['train'])):
            raise ValueError('Checkpoint and prepared row partitions differ')
        contract = dict(artifact_sha256=receipt['artifact_sha256'], prepared_sha256=driver.sha256(prepared_path),
                        audit_sources=audit_sources, patient_limit=args.patient_limit, device=args.device)
        if args.runtime_repair:
            contract.update(runtime_repair_receipt_sha256=driver.sha256(runtime_path), runtime_backend=F7_REPAIR_BACKEND)
        identity = dict(arm=receipt['arm'], seed=receipt['seed'], partition=partition, outer_fold=receipt['outer_fold'])
        inventory.append(dict(**identity, final_epoch=80, artifact_verified=True, row_partition_verified=True))
        if args.readcore:
            continue
        destination = args.output/(cell.name+'.json')
        if destination.exists():
            previous = json.loads(destination.read_text())
            if not args.resume or previous['contract'] != contract or previous['status'] != 'PASS_FINAL80_MECHANISM_REPLAY':
                raise ValueError('Existing mechanism report differs or requires explicit --resume')
            results.append(previous)
            continue
        if receipt['arm'] in ('F8', 'F9') and not str(args.device).startswith('cuda'):
            raise ValueError('Official Mamba checkpoint requires --device cuda; no replacement scan is permitted')
        cache_key = 'patch_uni2' if receipt['arm'].endswith('_UNI2') else 'patch_regions'
        if cache_key not in banks:
            banks[cache_key] = PatchBank(read_patch_cache(assets[cache_key]), args.device)
        torch.manual_seed(receipt['seed'])
        model = MechanismModel('R3', receipt['arm'], prepared, member, banks[cache_key], args.device,
                               gamma=config['R3'].get('gamma', .1),
                               intervention='wrong_query' if receipt['arm'] == 'F5_WRONG_QUERY' else None)
        model.load_state_dict(checkpoint['snapshots'][80]['state_dict'], strict=True)
        model.eval()
        strata = np.array([str(raw['cancer'][i])+'|'+str(raw['clinical_raw']['AJCC_PATHOLOGIC_TUMOR_STAGE'][i])+'|'+str(raw['platform'][i])
                           for i in range(len(raw['t']))])
        for name, part in [('train', prepared['train']), ('validation', prepared['valid'])]:
            mapping, _ = patient_mismatch(part, strata, 270077+partition+(name == 'validation'))
            model.query_map[part] = mapping[part]
        result = replay_cell(model, prepared, raw, checkpoint, args.patient_limit)
        result.update(identity=identity, contract=contract)
        driver.write(destination, result)
        results.append(result)
        print(json.dumps(dict(status=result['status'], **identity, sampled_risk_error=result['sampled_final80_risk_replay_max_abs'])), flush=True)
        del model
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    summary = dict(status='EMPTY_COMPLETED_R3_INVENTORY' if not inventory else
                         ('PASS_READCORE_INVENTORY' if args.readcore else 'COMPLETE_REQUESTED_MECHANISM_REPLAYS'),
                   inventoried_cells=len(inventory), completed_mechanism_cells=len(results),
                   optimizer_updates=0, inventory=inventory, audit_sources=audit_sources,
                   all_rows_are_anonymous_cell_identifiers=True, actual_weights_changed=False,
                   F7_runtime_repair=args.runtime_repair, explicit_runtime_backend=F7_REPAIR_BACKEND,
                   per_arm=grouped_mechanisms(results, ('arm',)),
                   per_arm_seed=grouped_mechanisms(results, ('arm', 'seed')),
                   per_arm_partition=grouped_mechanisms(results, ('arm', 'partition')),
                   aggregation_boundary='descriptive aggregation of fixed-first-<=8-row mechanism cells; not independent patient cohorts; full-validation saved-risk performance is separate',
                   gradient_scope='frozen full-validation Efron dLoss/dr, backpropagation through only the first <=8 saved validation network paths; no regularizer or pooled-main-effect objective',
                   readout_derivative_boundary='derivative below 1% of branch maximum is a fixed numerical diagnostic, not an estimated medical threshold')
    driver.write(args.output/'INVENTORY.json', summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

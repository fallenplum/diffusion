"""Fixed full-risk training and frozen, layer-context routing interventions."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import runtime
from .data import Data, array_hash
from .model import BidirectionalReread
from .spatial import OfficialSpatialEncoder
from .training_math import efron_loss, risk_gradient_cache


def model_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def make_model(ds, arm, seed, fold, device='cuda'):
    torch.set_num_threads(1)
    torch.manual_seed(int(seed)+1000*int(fold))
    torch.cuda.manual_seed_all(int(seed)+1000*int(fold))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spatial = OfficialSpatialEncoder()
    if spatial.test_only:
        raise ValueError('a formal model cannot use a synthetic scan')
    return BidirectionalReread(ds.molecular.shape[1], ds.slot_membership.cpu(), spatial,
                               arm=arm, rank=4).to(device)


def regularizer(model):
    """Actual L2, once per full-risk update; no decoupled optimizer decay."""
    c = runtime.cfg()['training']
    head_names = ('main_path', 'main_gene', 'rho_path', 'rho_gene')
    encoder = sum((p.square().sum() for name, p in model.named_parameters()
                   if name.split('.')[0] not in head_names and p.ndim >= 2
                   and not getattr(p, '_no_weight_decay', False)),
                  next(model.parameters()).new_zeros(()))
    readout = next(model.parameters()).new_zeros(())
    for name in head_names:
        head = getattr(model, name)
        effective = head.coefficient/2 if name.endswith('path') else (head.slot @ head.axis)/108
        readout = readout+effective.square().sum()
    return .5*c['encoder_L2_lambda']*encoder+.5*c['effective_readout_L2_lambda']*readout


def encode_patient(model, ds, patient, *, omics_patient=None):
    device = next(model.parameters()).device
    j = int(patient if omics_patient is None else omics_patient)
    return model.encode(ds.molecular[j].to(device), ds.molecular_mask[j].to(device),
                        ds.gene_ids.to(device), ds.load_patient_slides(int(patient), device=device))


def forward_patients(model, ds, patients):
    risks, mains, interactions = [], [], []
    for p in patients:
        encoded = encode_patient(model, ds, int(p))
        out = model.read_encoded(encoded, ds.base[int(p)])
        risks.append(out['risk']); mains.append(out['base']-out['offset']); interactions.append(out['interaction'])
    return {'risk': torch.stack(risks), 'main_residual': torch.stack(mains), 'interaction': torch.stack(interactions)}


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    return copy.deepcopy(value)


def _save(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    torch.save(record, temporary); temporary.replace(path)


def _moments(values):
    x = np.asarray(values, np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('diagnostic requires finite nonempty values')
    return {'count': len(x), 'mean': float(x.mean()), 'SD': float(x.std()),
            'RMS': float(np.sqrt((x*x).mean())), 'abs_q95': float(np.quantile(np.abs(x), .95)),
            'min': float(x.min()), 'max': float(x.max())}


def _bank(out, encoded):
    return [{'gene': _cpu(state['gene']), 'gene_valid': _cpu(encoded['gene_valid']),
             'path': _cpu(state['path']), 'path_valid': _cpu(encoded['path_valid'])} for state in out['states']]


def _foreign(bank, encoded, device, *, gene=False, path=False):
    """Fixed layer bank; keep target original values, masks, anchors and heads.

    Other-patient path regions are count-matched by deterministic quantiles.
    Repeated context states do not represent extra observed tissue regions.
    """
    result = []
    for state in bank:
        current = {}
        if gene:
            current.update(gene=state['gene'].to(device), gene_valid=state['gene_valid'].to(device))
        if path:
            pooled = torch.cat(state['path']).reshape(-1, 2, 128)
            pv = torch.cat(state['path_valid']).reshape(-1, 2)
            ps, masks = [], []
            same_layout = len(state['path']) == len(encoded['path0']) and all(
                len(left) == len(right) for left, right in zip(state['path'], encoded['path0']))
            for slide, own in enumerate(encoded['path0']):
                if same_layout:
                    ps.append(state['path'][slide].to(device))
                    masks.append(state['path_valid'][slide].to(device))
                    continue
                count = len(own)//2
                index = torch.floor((torch.arange(count, dtype=torch.float64)+.5)*len(pooled)/count).long().clamp_max(len(pooled)-1)
                ps.append(pooled[index].reshape(-1, 128).to(device))
                masks.append(pv[index].reshape(-1).to(device))
            current.update(path=ps, path_valid=masks)
        result.append(current)
    return result


def _wrong_map(ds, indices, seed):
    """Outcome-blind zero-self mapping within cancer and natural-RNA-NA class."""
    from src.phase23_cdei.data import partition_derangement, raw_rna_missing_patterns
    from .data import read_feature_npz
    # Measurement absence in a whole gene block reflects assay/platform source;
    # this control does not use time, event, Stage or any predicted error.
    raw = read_feature_npz()
    patterns = raw_rna_missing_patterns(raw['rna_genes'])
    result, audit = partition_derangement(ds.cancer_type, patterns, np.asarray(indices), int(seed))
    return result, audit


def _mean_context(banks):
    """Equal fit-patient / WSI / region means; preserve the two query identities."""
    answer = []
    for layer in range(2):
        states = [v[layer] for v in banks]
        valid = torch.stack([v['gene_valid'] for v in states])
        genes = torch.stack([v['gene'] for v in states])
        g = (genes*valid[:, :, None]).sum(0)/valid.sum(0)[:, None].clamp_min(1)
        paths = []
        for state in states:
            per_slide = [p.reshape(-1, 2, 128).mean(0) for p in state['path']]
            paths.append(torch.stack(per_slide).mean(0))
        p = torch.stack(paths).mean(0)
        answer.append({'gene': g, 'gene_valid': valid.any(0),
                       'path': [p], 'path_valid': [torch.ones(2, dtype=torch.bool)]})
    return answer


@torch.no_grad()
def predict(model, ds):
    model.eval(); device = next(model.parameters()).device
    fit_rows = {k: [] for k in ('risk', 'base', 'interaction', 'path_main', 'gene_main')}
    fit_banks = []
    for p in ds.fit:
        e = encode_patient(model, ds, int(p)); o = model.read_encoded(e, ds.base[int(p)])
        for key, field in (('risk', 'risk'), ('base', 'base'), ('interaction', 'interaction'),
                           ('path_main', 'learned_path_main'), ('gene_main', 'learned_gene_main')):
            fit_rows[key].append(float(o[field]))
        fit_banks.append(_bank(o, e))
    means = _mean_context(fit_banks)
    del fit_banks
    valid_banks, matched = {}, []
    for p in ds.valid:
        e = encode_patient(model, ds, int(p)); o = model.read_encoded(e, ds.base[int(p)])
        valid_banks[int(p)] = _bank(o, e); matched.append(float(o['risk']))
    # The mapping is stored only privately, and every paired source is in valid.
    wrong, wrong_audit = _wrong_map(ds, ds.valid, 280000+ds.seed*10+ds.fold)
    wrong = np.asarray(wrong, np.int64)
    if wrong.shape == (len(ds.patient_ids),):
        selected_wrong = wrong[ds.valid]
    elif wrong.shape == ds.valid.shape:
        selected_wrong = wrong
    else:
        raise ValueError('wrong-context map has unexpected indexing schema')
    if not np.isin(selected_wrong, ds.valid).all() or np.any(selected_wrong == ds.valid):
        raise ValueError('wrong context must be partition-local and zero-self')
    lookup = {int(p): j for j, p in enumerate(ds.valid)}
    risks = {key: [] for key in runtime.CONTROLS}
    base_risk, components = [], []
    for p, j, expected in zip(ds.valid, selected_wrong, matched):
        p, j = int(p), int(j)
        e = encode_patient(model, ds, p); o = model.read_encoded(e, ds.base[p])
        if abs(float(o['risk'])-expected) > 1e-6:
            raise ValueError('matched prediction changed between context and control passes')
        values = {'MATCHED': o['risk'], 'OFF': model.read_encoded(e, ds.base[p], interaction_off=True)['risk'],
                  'CONSTANT_VALUES': model.read_encoded(e, ds.base[p], constant_path_value=True, constant_gene_value=True)['risk']}
        for key, bank, gene, path in (
                ('GENE_MEAN', means, True, False), ('GENE_WRONG', valid_banks[j], True, False),
                ('PATH_MEAN', means, False, True), ('PATH_WRONG', valid_banks[j], False, True),
                ('BOTH_WRONG', valid_banks[j], True, True)):
            values[key] = model.read_encoded(e, ds.base[p], context=_foreign(bank, e, device, gene=gene, path=path))['risk']
        swapped = encode_patient(model, ds, p, omics_patient=j)
        whole_offset = float(ds.CP_valid[lookup[p]]+ds.RNA_valid[lookup[j]])
        values['WHOLE_OMICS_WRONG'] = model.read_encoded(swapped, whole_offset)['risk']
        for key in runtime.CONTROLS:
            risks[key].append(float(values[key]))
        base_risk.append(float(o['base']))
        components.append([float(o[k]) for k in ('learned_path_main', 'learned_gene_main', 'path_interaction', 'gene_interaction')])
        if self_arm_no_interaction(model):
            if any(abs(float(values[k])-float(o['risk'])) > 1e-6 for k in runtime.CONTROLS if k != 'WHOLE_OMICS_WRONG'):
                raise ValueError('C00 frozen routing controls must preserve its own additive output')
        if not torch.equal(values['OFF'], o['base']):
            raise ValueError('OFF must be exact learned own main, not the external anchor')
    return {'fit_risk': np.asarray(fit_rows['risk']), 'fit_components': {k: np.asarray(v) for k, v in fit_rows.items()},
            'valid_risks': {k: np.asarray(v) for k, v in risks.items()}, 'valid_base_risk': np.asarray(base_risk),
            'valid_components': np.asarray(components), 'wrong_context_mapping': selected_wrong,
            'wrong_context_audit': wrong_audit,
            'prediction_audit': {'valid_endpoint_reads': 0, 'all_controls_share_same_weights_and_preprocessing': True,
                'source_contexts_frozen_from_matched_layers': True, 'wrong_control_does_not_refit_model': True,
                'OFF_is_own_learned_main_not_external_M0': True, 'whole_omics_changes_external_RNA_only': True}}


def self_arm_no_interaction(model):
    return model.arm == 'C00'


def train_cell(seed, fold, arm, *, resume=False):
    configuration, identity = runtime.cfg(), runtime.source()
    path = runtime.cell_dir(seed, fold, arm); path.mkdir(parents=True, exist_ok=True)
    latest = path/'LATEST_PRIVATE.pt'
    if not (path/'DONE_PRIVATE.json').exists() and (path/'START_PRIVATE.json').exists() and not (resume and latest.exists()):
        raise ValueError('incomplete prior attempt: explicit same-source resume required')
    qualifications = runtime.verify_qualifications(identity)
    qualification_path = path/'QUALIFICATIONS_PRIVATE.json'
    if qualification_path.exists():
        if json.loads(qualification_path.read_text()) != qualifications:
            raise ValueError('qualification receipts changed across the same-source resume')
    else:
        runtime.atomic_json(qualification_path, qualifications)
    if (path/'DONE_PRIVATE.json').exists():
        return runtime.verify_done(path, identity, seed, fold, arm)
    ds = Data(fold, seed, cache_slides=4)
    model = make_model(ds, arm, seed, fold)
    initial_hash = model_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration['training']['learning_rate'])
    history, first_epoch = [], 1
    if resume:
        old = torch.load(latest, map_location='cpu', weights_only=False)
        if old['source'] != identity or old['prepared_sha256'] != ds.prepared_sha256 or old['initial_state_sha256'] != initial_hash:
            raise ValueError('resume source, preprocessing or initialization mismatch')
        model.load_state_dict(old['model_state']); optimizer.load_state_dict(old['optimizer_state'])
        history, first_epoch = old['history'], old['epoch']+1
    else:
        runtime.atomic_json(path/'START_PRIVATE.json', {'source': identity, 'stage': 'prototype', 'seed': seed,
            'fold': fold, 'arm': arm, 'prepared_sha256': ds.prepared_sha256,
            'initial_state_sha256': initial_hash, 'epochs': 80, 'valid_endpoint_reads': 0})
    t, event = ds.fit_labels()
    offset_before = ds.base.clone()
    started = time.monotonic()
    for epoch in range(first_epoch, 81):
        tick = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
        audit = risk_gradient_cache(model, lambda patients: forward_patients(model, ds, patients),
                                    ds.fit, t, event, microbatch=1, regularizer=lambda: regularizer(model))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if not torch.equal(ds.base, offset_before) or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('offset changed or parameters became nonfinite')
        audit.update(epoch=epoch, seconds=time.monotonic()-tick, clipped_from_gradient_L2=float(norm))
        history.append(audit)
        runtime.atomic_json(path/'PROGRESS_PRIVATE.json', {'source': identity, 'arm': arm, 'seed': seed, 'fold': fold,
            'epoch': epoch, 'total_epochs': 80, 'fit_cox': audit['cox_loss'], 'valid_endpoint_reads': 0,
            'wall_seconds': time.monotonic()-started})
        if epoch % 10 == 0:
            _save(latest, {'source': identity, 'epoch': epoch, 'model_state': _cpu(model.state_dict()),
                'optimizer_state': _cpu(optimizer.state_dict()), 'prepared_sha256': ds.prepared_sha256,
                'initial_state_sha256': initial_hash, 'history': history})
    prediction = predict(model, ds)
    fit_risk = torch.as_tensor(prediction['fit_risk'], dtype=torch.float64)
    training_audit = {'actual_epochs': 80, 'optimizer_steps': 80, 'new_main_anchor_fits': 0,
        'complete_fit_risk_sets': True, 'fixed_final_epoch': True, 'valid_endpoint_reads': 0,
        'external_offset_unchanged': True, 'learned_unimodal_heads_are_not_shared_fitted_coefficients_between_arms': True,
        'Clinical_directly_conditions_local_queries': False, 'maximum_gradient_cache_error': max(
            row['maximum_absolute_component_replay_error'] for row in history),
        'nonconvex_global_convergence_claimed': False,
        'final_fit_Efron_NLL': float(efron_loss(fit_risk, t, event)),
        'fit_main_residual': _moments(prediction['fit_components']['base']-ds.base[ds.fit].cpu().numpy()),
        'fit_interaction': _moments(prediction['fit_components']['interaction']),
        'valid_interaction': _moments(prediction['valid_risks']['MATCHED']-prediction['valid_base_risk'])}
    record = {'source': identity, 'stage': 'prototype', 'seed': seed, 'fold': fold, 'arm': arm,
        'fit': ds.fit, 'valid': ds.valid, 'prepared_sha256': ds.prepared_sha256,
        'model_state': _cpu(model.state_dict()), 'initial_state_sha256': initial_hash,
        'model_parameters': model.parameter_counts(), 'model_init': {'genes': 26708, 'path_dim': 1536, 'rank': 4},
        'CP_valid': ds.CP_valid, 'RNA_valid': ds.RNA_valid, 'M0_valid': ds.M0_valid,
        'history': history, 'training_audit': training_audit, 'data_audit': ds.audit, **prediction}
    if runtime.source() != identity:
        raise ValueError('training source changed before checkpoint sealing')
    _save(path/'checkpoint.pt', record)
    result = {'status': 'COMPLETE', 'source': identity, 'stage': 'prototype', 'seed': seed, 'fold': fold, 'arm': arm,
        'epochs': 80, 'optimizer_steps': 80, 'checkpoint_sha256': runtime.sha(path/'checkpoint.pt'),
        'prepared_sha256': ds.prepared_sha256, 'initial_state_sha256': initial_hash,
        'training_audit': training_audit, 'wall_seconds': time.monotonic()-started}
    runtime.atomic_json(path/'DONE_PRIVATE.json', result)
    # Final checkpoint is sufficient for replay; resumable optimizer state is
    # intentionally kept until the independent post-training audit passes.
    return result

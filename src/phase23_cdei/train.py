"""Fixed-kernel feasibility cells. Protected artifacts stay outside Git."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch

from .data import Data
from .features import (equal_slide_patient_mean, unimodal_feature_matrix,
                       AdditiveTwoDimensionalControl)
from .model import FrozenDualEvidence, module_fingerprint
from .reference import FrozenPatientReferenceBank
from .readout import FitOnlyStandardizer, fit_conditional_cox, risk_with_offset
from .warmup import train_kernel, FORMAL_EPOCHS

ARMS = ('REL', 'DUAL', 'ADD', 'NO_C', 'WRONG_O', 'WRONG_C', 'MEAN', 'CENTERED')


def _cpu(x):
    return x.detach().cpu().numpy().copy()


@torch.no_grad()
def extract_and_fit(backbone, ds, source, fit_readouts=True):
    """No selection_labels call; feature views never contain endpoints.

    Every absolute intervention shares observed slot intersection and reference
    exclusion union. Relative and frozen offset are always the matched ones.
    Returns (private readout payload, private frozen reference state).
    """
    ft, active = ds.ft, np.asarray(ds.active)
    tokens, token_mask = backbone.encode_omics(*(ft[k] for k in ('R','G','L','RM','CM')))
    bank = FrozenPatientReferenceBank(tokens, token_mask, ft['C'], ds.fit,
        provenance={'kernel_sha256': module_fingerprint(backbone), 'source': source,
                    'role': ds.role, 'fold': ds.fold,
                    'feature_cache_sha256': ds.audit.get('feature_cache_sha256')})
    extractor = FrozenDualEvidence(backbone, bank, tau=1.)
    clinical_mean = ft['C'][ds.fit].mean(0)
    pieces = {k: [] for k in ('relative','absolute','NO_C','WRONG_O','WRONG_C',
                              'mean_only','centered_only','pathology')}
    all_available, reference_equal = True, True
    for start in range(0, len(ds.visual), 32):
        slides = np.arange(start, min(start + 32, len(ds.visual)))
        owners = np.asarray(ds.owner)[slides]
        other_o, other_c = ds.wrong_o[owners], ds.wrong_c[owners]
        intersection = token_mask[owners] & token_mask[other_o]
        exclusions = np.stack((owners, other_o), axis=1)
        args = (ds.visual[slides], ds.mask[slides], ft['C'][owners],
                *(ft[k][owners] for k in ('R','G','L','RM','CM')), owners)
        common = {'reference_exclusions': exclusions, 'absolute_token_mask': intersection}
        matched = extractor.extract(*args, **common)
        no_c = extractor.extract(*args, absolute_clinical=clinical_mean.expand(len(owners), -1), **common)
        wrong_c = extractor.extract(*args, absolute_clinical=ft['C'][other_c], **common)
        wrong_o = extractor.extract(*args, absolute_omics=tuple(ft[k][other_o] for k in ('R','G','L','RM','CM')),
                                    omics_source_indices=other_o, **common)
        for intervention in (no_c, wrong_c, wrong_o):
            for key in ('relative','relative_p_to_o','relative_o_to_p'):
                if not torch.equal(matched[key], intervention[key]):
                    raise ValueError('absolute intervention changed matched relative attention')
            reference_equal &= torch.equal(matched['reference_counts'], intervention['reference_counts'])
        if not reference_equal:
            raise ValueError('paired reference donors differ')
        for item in (matched, no_c, wrong_c, wrong_o):
            all_available &= bool(item['absolute_available'].all())
        if not all_available:
            raise ValueError('patient WSI without common observed absolute evidence')
        for key in ('relative','absolute','mean_only','centered_only'):
            pieces[key].append(_cpu(matched[key]))
        for key, item in (('NO_C', no_c), ('WRONG_O', wrong_o), ('WRONG_C', wrong_c)):
            pieces[key].append(_cpu(item['absolute']))
        p = backbone.encode_pathology(ds.visual[slides], ds.mask[slides])
        mask = ds.mask[slides]
        pooled = (p * mask[...,None]).sum(1) / mask.sum(1).clamp_min(1)[:,None]
        pieces['pathology'].append(_cpu(pooled))
    pooled = {key: equal_slide_patient_mean(np.concatenate(value), ds.owner, active)
              for key, value in pieces.items()}
    n = len(ft['C'])
    patient_p = np.zeros((n,32), np.float64)
    patient_p[active] = pooled['pathology']
    unimodal = unimodal_feature_matrix(_cpu(ft['C']), patient_p, _cpu(tokens), _cpu(token_mask))
    additive = AdditiveTwoDimensionalControl().fit(unimodal, ds.fit)
    additive2 = additive.transform(unimodal[active])
    rel = pooled['relative'].astype(np.float64)
    extras = {'DUAL':pooled['absolute'], 'ADD':additive2, 'NO_C':pooled['NO_C'],
              'WRONG_O':pooled['WRONG_O'], 'WRONG_C':pooled['WRONG_C'],
              'MEAN':pooled['mean_only'], 'CENTERED':pooled['centered_only']}
    raw = {'REL': rel}
    raw.update({arm: np.c_[rel, extra] for arm,extra in extras.items()})
    # Index map is explicit: neither active sorting nor WSI order can silently
    # reorder fit supervision or heldout predictions.
    lookup = {int(value): index for index,value in enumerate(active)}
    fi = np.array([lookup[int(x)] for x in ds.fit])
    vi = np.array([lookup[int(x)] for x in ds.valid])
    anchor = _cpu(ft['ANCHOR']).astype(np.float64)
    result = {'source':source, 'role':ds.role, 'seed':ds.seed, 'fold':ds.fold,
              'fit':np.array(ds.fit), 'valid':np.array(ds.valid),
              'anchor_fit':anchor[ds.fit], 'anchor_valid':anchor[ds.valid],
              'J_CS_valid':np.array(ds.inherited_simple['models']['J_CS']['risk']),
              'arms':{}, 'sensitivities':{}, 'data_audit':ds.audit,
              'reference_summary':bank.public_summary(),
              'feature_audit':{'all_absolute_available':all_available,
                 'paired_reference_counts_equal':reference_equal,
                 'matched_relative_bitwise_preserved':True,
                 'equal_WSI_patient_pooling':True, 'shared_absolute_slot_intersection':True,
                 'additive_control':additive.public_summary(),
                 'selection_label_reads':0, 'absolute_patient_SD':np.std(extras['DUAL'][fi],axis=0).tolist()}}
    if fit_readouts:
        labels = ds.fit_labels()
        for arm in ARMS:
            fit_raw, valid_raw = raw[arm][fi], raw[arm][vi]
            scaler = FitOnlyStandardizer().fit(fit_raw)
            xs, xv = scaler.transform(fit_raw), scaler.transform(valid_raw)
            weights, diagnostic = fit_conditional_cox(xs, labels['time'], labels['event'],
                                                      result['anchor_fit'], penalty=10.)
            risk, delta = risk_with_offset(xv, weights, result['anchor_valid'])
            fit_risk, _ = risk_with_offset(xs, weights, result['anchor_fit'])
            diagnostic.update(fit_feature_rank=int(np.linalg.matrix_rank(xs)),
                valid_risk_correction_SD=float(delta.std()),
                valid_risk_correction_RMS=float(np.sqrt(np.mean(delta**2))))
            result['arms'][arm] = {'risk':risk, 'fit_risk':fit_risk,
                'fit_features':fit_raw, 'valid_features':valid_raw, 'weights':weights,
                'scaler':{'mean':scaler.mean, 'scale':scaler.scale,
                          'constant':scaler.constant, 'fit_n':scaler.fit_n}, 'diagnostic':diagnostic}
        dual = result['arms']['DUAL']
        mean, scale, weights = dual['scaler']['mean'], dual['scaler']['scale'], dual['weights']
        xs = (dual['valid_features'] - mean) / scale
        off = xs.copy(); off[:,-2:] = 0
        result['sensitivities']['DUAL_ABS_OFF'] = result['anchor_valid'] + off @ weights
        for key in ('WRONG_O','WRONG_C','NO_C'):
            x = (result['arms'][key]['valid_features'] - mean) / scale
            result['sensitivities']['DUAL_'+key] = result['anchor_valid'] + x @ weights
        result['feature_audit']['readout_solver_calls'] = len(ARMS)
    return result, {key:value.detach().cpu() if isinstance(value,torch.Tensor) else value
                    for key,value in bank.state_dict().items()}


def run_cell(role, seed, fold, device, output_dir, source, step_callback=None):
    started = time.monotonic()
    ds = Data(fold, role, seed, device=device)
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    backbone, warmup_head, warmup_audit = train_kernel(ds, seed, fold,
        epochs=FORMAL_EPOCHS, step_callback=step_callback)
    payload, reference_state = extract_and_fit(backbone, ds, source)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    kernel = {'source':source, 'role':role, 'seed':seed, 'fold':fold,
              'fit':np.array(ds.fit), 'valid':np.array(ds.valid),
              'backbone':{k:v.detach().cpu() for k,v in backbone.state_dict().items()},
              'warmup_head':warmup_head, 'warmup_audit':warmup_audit,
              'reference_state':reference_state, 'clinical_dim':int(ds.ft['C'].shape[1]),
              'membership':_cpu(ds.ft['membership']), 'data_audit':ds.audit}
    for name,value in (('kernel.pt',kernel), ('readouts.pt',payload)):
        path = output_dir/name
        torch.save(value,path.with_suffix('.partial'))
        path.with_suffix('.partial').replace(path)
    return {'status':'COMPLETE', 'source':source, 'role':role, 'seed':seed, 'fold':fold,
            'kernel_epochs':warmup_audit['actual_epochs'], 'readout_fits':len(ARMS),
            'wall_seconds':time.monotonic()-started,
            'peak_GPU_reserved_bytes':int(torch.cuda.max_memory_reserved()) if str(device).startswith('cuda') else 0,
            'heldout_endpoint_reads_in_training':0, 'anchor_bitwise_preserved':True}

"""Immutable Phase25 transforms, independently fitted frozen CP/RNA risks.

Only scalar predictions cross teacher coordinates. Seven new convex heads are
fitted per prepared partition; the original full-fit CP guard is reused exactly.
No Phase25 preprocessing estimator or artifact is fitted, relabelled or edited.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.phase10_trimodal.math import fit_ridge
from src.phase25_local import data as old

LEGACY_SOURCE = '91cad3e88814d92d9f43b3ebf8c0962d11098b3fa80a8ea81c275a6b83f2884c'


def _runtime():
    from . import runtime
    return runtime


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _array(value, name, shape=None):
    value = np.asarray(value, np.float64)
    require(np.isfinite(value).all() and (shape is None or value.shape == shape), name + ' shape/finite mismatch')
    return np.array(value, copy=True)


def _distribution(cf, full):
    cf, full = np.asarray(cf, float), np.asarray(full, float)
    a, b = float(cf.std()), float(full.std())
    correlation = float(np.corrcoef(cf, full)[0, 1]) if a > 0 and b > 0 else None
    return {'CF_mean': float(cf.mean()), 'fullfit_mean': float(full.mean()), 'CF_SD': a, 'fullfit_SD': b,
            'CF_to_fullfit_SD_ratio': a / b if b > 0 else None, 'Pearson': correlation,
            'Pearson_defined': correlation is not None, 'gap_RMS': float(np.sqrt(np.mean((cf-full)**2))),
            'alignment_applied': False}


def build_prepared(legacy, legacy_receipt, raw, *, fold, identity, convex_fitter=fit_ridge):
    """Fit only CP/RNA heads using old teacher states and fit-only labels."""
    require(legacy['source'] == LEGACY_SOURCE and legacy_receipt['source'] == LEGACY_SOURCE,
            'legacy source must remain the exact published Phase25 source')
    require(legacy['fold'] == int(fold) and legacy['stage'] == 'prototype', 'legacy partition mismatch')
    fit, valid, active = (np.array(legacy[name], np.int64, copy=True) for name in ('fit', 'valid', 'active'))
    require(not np.intersect1d(fit, valid).size and np.array_equal(active, np.sort(np.r_[fit, valid])),
            'legacy active patient scope mismatch')
    n = len(legacy['base'])
    time = _array(legacy['fit_time'], 'fit time', fit.shape)
    event = _array(legacy['fit_event'], 'fit event', fit.shape)
    require((time >= 0).all() and np.isin(event, (0, 1)).all() and event.sum() > 0, 'invalid fit-only supervision')
    lookup = np.full(n, -1, int); lookup[fit] = np.arange(len(fit))
    cp_cf, rna_cf, count = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    teachers, actual_fits = [], 0
    require(len(legacy['mainstate']['teachers']) == 3, 'three legacy teachers required')
    for prior in legacy['mainstate']['teachers']:
        train, held = np.asarray(prior['train'], int), np.asarray(prior['held'], int)
        require(not np.intersect1d(train, held).size and np.array_equal(np.union1d(train, held), np.sort(fit)),
                'teacher train/held is not the same fit partition')
        state = prior['state']
        require(np.array_equal(state['fit'], train) and state['fit_hash'] == old.array_hash(train),
                'legacy teacher preprocessing is not train-local')
        require(state['state_hash'] == old.state_hash({key: value for key, value in state.items() if key != 'state_hash'}),
                'legacy teacher preprocessing bytes changed')
        position = lookup[train]
        require((position >= 0).all() and event[position].sum() > 0, 'invalid teacher label scope or zero events')
        # Frozen transformations only: no selection, scaling or PCA estimator
        # is refitted. Clinical dimensions may differ between teachers.
        transformed = old.transform(raw, state, active)
        cp_x, rna_x = transformed['CP'], transformed['R']
        require(rna_x.shape == (n, 512), 'all 512 RNA coordinates must enter its independent head')
        cp_coef, cp_optimizer = convex_fitter(cp_x[train], time[position], event[position], 10.)
        rna_coef, rna_optimizer = convex_fitter(rna_x[train], time[position], event[position], 10.)
        actual_fits += 2
        cp_coef = _array(cp_coef, 'teacher CP coefficients', (cp_x.shape[1],))
        rna_coef = _array(rna_coef, 'teacher RNA coefficients', (512,))
        cp_risk, rna_risk = cp_x[held] @ cp_coef, rna_x[held] @ rna_coef
        require(not np.isfinite(cp_cf[held]).any() and not np.isfinite(rna_cf[held]).any(), 'duplicate teacher heldout prediction')
        cp_cf[held], rna_cf[held], count[held] = cp_risk, rna_risk, count[held] + 1
        teachers.append({'teacher': int(prior['teacher']), 'train': train.copy(), 'held': held.copy(),
            'state': copy.deepcopy(state), 'legacy_state_sha256': state['state_hash'],
            'CP_coefficients': cp_coef, 'RNA_coefficients': rna_coef,
            'CP_coefficient_sha256': old.array_hash(cp_coef), 'RNA_coefficient_sha256': old.array_hash(rna_coef),
            'CP_risk_held': cp_risk, 'RNA_risk_held': rna_risk,
            'CP_optimizer': cp_optimizer, 'RNA_optimizer': rna_optimizer,
            'CP_fit_feature_sha256': old.array_hash(cp_x[train]), 'CP_held_feature_sha256': old.array_hash(cp_x[held]),
            'RNA_fit_feature_sha256': old.array_hash(rna_x[train]), 'RNA_held_feature_sha256': old.array_hash(rna_x[held]),
            'fit_endpoint_sha256': old.array_hash(np.c_[time[position], event[position]]), 'lambda': 10.})
    require((count[fit] == 1).all() and count[valid].sum() == 0
            and np.isfinite(cp_cf[fit]).all() and np.isfinite(rna_cf[fit]).all(), 'crossfit scalar cover is not exact fit-only')
    full_rna = _array(legacy['R'], 'legacy full RNA', (n, 512))
    full_rna_coef, full_rna_optimizer = convex_fitter(full_rna[fit], time, event, 10.)
    actual_fits += 1
    full_rna_coef = _array(full_rna_coef, 'full RNA coefficients', (512,))
    full_cp_coef = np.array(legacy['mainstate']['CP_coefficients'], copy=True)
    cp_valid = np.array(legacy['CP_valid'], np.float64, copy=True)
    rna_valid = full_rna[valid] @ full_rna_coef
    main_valid = cp_valid + rna_valid
    full_cp_fit = np.asarray(legacy['mainstate']['CP_X'], float)[fit] @ full_cp_coef
    full_rna_fit = full_rna[fit] @ full_rna_coef
    full_main_fit = full_cp_fit + full_rna_fit
    base = np.zeros(n, np.float64); base[fit] = cp_cf[fit] + rna_cf[fit]; base[valid] = main_valid
    scale = float(base[fit].std())
    require(actual_fits == 7 and np.isfinite(base[active]).all() and np.isfinite(scale) and scale > 1e-8,
            'independent main budget or risk scale invalid')
    saved = {'source': identity, 'legacy_source': LEGACY_SOURCE,
        'legacy_prepared_sha256': legacy_receipt['sha256'], 'fold': int(fold), 'stage': 'prototype',
        'fit': fit, 'valid': valid, 'active': active, 'fit_time': time, 'fit_event': event,
        'CP_cf_fit': cp_cf[fit], 'RNA_cf_fit': rna_cf[fit], 'CP_valid': cp_valid,
        'RNA_valid': rna_valid, 'M0_valid': main_valid, 'base': base, 'scale': scale,
        'mainstate': {'full_state': copy.deepcopy(legacy['mainstate']['full_state']),
            'full_CP_coefficients': full_cp_coef, 'full_RNA_coefficients': full_rna_coef,
            'full_RNA_optimizer': full_rna_optimizer, 'teachers': teachers,
            'full_CP_fit_risk': full_cp_fit, 'full_RNA_fit_risk': full_rna_fit, 'full_main_fit_risk': full_main_fit},
        'audit': {'schema': 'phase26_exact_CP_plus_independent_RNA_v1', 'source': identity,
            'legacy_source': LEGACY_SOURCE, 'legacy_prepared_sha256': legacy_receipt['sha256'], 'fold': int(fold),
            'fit_n': len(fit), 'valid_n': len(valid), 'fit_sha256': old.array_hash(fit), 'valid_sha256': old.array_hash(valid),
            'new_convex_fits': actual_fits, 'reused_full_CP_fits': 1, 'new_preprocessing_fits': 0,
            'CP_parameters': len(full_cp_coef), 'RNA_parameters': len(full_rna_coef),
            'fullfit_main_parameters': len(full_cp_coef) + len(full_rna_coef),
            'CP_exact_legacy_coefficients_and_valid_risk': True, 'independent_RNA_only_lambda': 10.,
            'CP_plus_RNA_coefficients': [1., 1.], 'meta_training': False, 'moment_alignment': False,
            'CNA_used': False, 'all512_RNA_inputs_retained': True,
            'old_preprocessing_source_relabelled': False, 'teacher_latent_stacking': False,
            'teacher_assignment_outcome_blind': True, 'heldout_endpoint_passed_to_fitters': False,
            'main_distribution': {'CP': _distribution(cp_cf[fit], full_cp_fit),
                                  'RNA': _distribution(rna_cf[fit], full_rna_fit),
                                  'sum': _distribution(base[fit], full_main_fit)},
            'CF_fullfit_shift_assumed_solved': False}}
    validate_prepared(saved, legacy, legacy_receipt, identity)
    return saved


def validate_prepared(saved, legacy, legacy_receipt, identity):
    require(saved.get('source') == identity and saved.get('legacy_source') == LEGACY_SOURCE,
            'new/legacy source mismatch')
    require(saved.get('legacy_prepared_sha256') == legacy_receipt['sha256'] and legacy['source'] == LEGACY_SOURCE,
            'legacy prepared artifact changed')
    fit, valid = saved['fit'], saved['valid']
    require(np.array_equal(fit, legacy['fit']) and np.array_equal(valid, legacy['valid'])
            and np.array_equal(saved['active'], legacy['active']), 'new/legacy patient partitions differ')
    require(np.array_equal(saved['fit_time'], legacy['fit_time']) and np.array_equal(saved['fit_event'], legacy['fit_event']),
            'new/legacy fit supervision differs')
    require(np.array_equal(saved['mainstate']['full_CP_coefficients'], legacy['mainstate']['CP_coefficients'])
            and np.array_equal(saved['CP_valid'], legacy['CP_valid']), 'strong CP anchor was changed')
    full_state = saved['mainstate']['full_state']
    require(full_state['state_hash'] == legacy['mainstate']['full_state']['state_hash']
            and full_state['state_hash'] == old.state_hash({key: value for key, value in full_state.items() if key != 'state_hash'}),
            'full preprocessing reference changed')
    cp, rna, count = np.zeros(len(saved['base'])), np.zeros(len(saved['base'])), np.zeros(len(saved['base']), int)
    require(len(saved['mainstate']['teachers']) == len(legacy['mainstate']['teachers']) == 3, 'three legacy teachers required')
    for current, prior in zip(saved['mainstate']['teachers'], legacy['mainstate']['teachers']):
        train, held = prior['train'], prior['held']
        require(np.array_equal(current['train'], train) and np.array_equal(current['held'], held), 'teacher patient scope changed')
        state = current['state']
        require(current['legacy_state_sha256'] == prior['state']['state_hash'] == state['state_hash']
                and state['state_hash'] == old.state_hash({key: value for key, value in state.items() if key != 'state_hash'})
                and np.array_equal(state['fit'], train), 'teacher preprocessing reference changed')
        require(current['lambda'] == 10.
                and current['CP_coefficient_sha256'] == old.array_hash(current['CP_coefficients'])
                and current['RNA_coefficient_sha256'] == old.array_hash(current['RNA_coefficients']), 'teacher coefficient identity changed')
        cp[held], rna[held], count[held] = current['CP_risk_held'], current['RNA_risk_held'], count[held] + 1
    require((count[fit] == 1).all() and not count[valid].any(), 'teacher crossfit coverage changed')
    require(np.array_equal(saved['CP_cf_fit'], cp[fit]) and np.array_equal(saved['RNA_cf_fit'], rna[fit]), 'branch CF risks changed')
    require(np.array_equal(saved['base'][fit], cp[fit] + rna[fit]), 'fit offset is not exact independent CP+RNA CF sum')
    full_rna_coef = saved['mainstate']['full_RNA_coefficients']
    require(np.asarray(full_rna_coef).shape == (512,), 'full RNA head lost 512-dimensional input')
    require(np.allclose(saved['RNA_valid'], legacy['R'][valid] @ full_rna_coef, rtol=0, atol=1e-10), 'full RNA valid replay failed')
    require(np.array_equal(saved['M0_valid'], saved['CP_valid'] + saved['RNA_valid'])
            and np.array_equal(saved['base'][valid], saved['M0_valid']), 'heldout risk is not the unchanged independent branch sum')
    require(saved['scale'] == float(saved['base'][fit].std()), 'diagnostic SD is not fit-CF-only')
    require(saved['audit']['new_convex_fits'] == 7 and saved['audit']['reused_full_CP_fits'] == 1,
            'convex fitting/reuse count changed')
    for key in ('base', 'CP_cf_fit', 'RNA_cf_fit', 'CP_valid', 'RNA_valid', 'M0_valid', 'fit_time', 'fit_event'):
        require(np.isfinite(saved[key]).all(), 'nonfinite prepared main: ' + key)
    return saved


def prepared_path(fold):
    return _runtime().PRIVATE / 'prepared' / f'prototype_fold{int(fold)}.pt'


def read_prepared(fold):
    runtime = _runtime(); identity = runtime.source(); path = prepared_path(fold)
    require(path.is_file() and path.with_suffix('.json').is_file(), 'new Phase26 prepared main missing')
    receipt = json.loads(path.with_suffix('.json').read_text())
    require(receipt.get('status') == 'COMPLETE' and receipt.get('source') == identity
            and receipt.get('fold') == int(fold) and receipt.get('sha256') == runtime.sha(path), 'Phase26 prepared receipt/source/hash mismatch')
    legacy, legacy_receipt = old.read_prepared(fold, 'prototype')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    return validate_prepared(saved, legacy, legacy_receipt, identity), receipt


def prepare(fold):
    runtime = _runtime(); configuration = runtime.cfg(); identity = runtime.source()
    require(int(fold) in configuration['prototype']['folds'], 'fold outside locked prototype')
    require(configuration['legacy_phase25_source'] == LEGACY_SOURCE, 'legacy source lock drift')
    path = prepared_path(fold)
    with old._file_lock(path.with_suffix('.lock')):
        if path.exists():
            return read_prepared(fold)
        require(not path.with_suffix('.json').exists(), 'orphan new preparation receipt')
        legacy, legacy_receipt = old.read_prepared(fold, 'prototype')
        raw = old.load_raw()
        saved = build_prepared(legacy, legacy_receipt, raw, fold=fold, identity=identity)
        saved['legacy_receipt_sha256'] = runtime.sha(old.prepared_path(fold, 'prototype').with_suffix('.json'))
        saved['audit']['legacy_receipt_sha256'] = saved['legacy_receipt_sha256']
        require(runtime.source() == identity, 'source changed during new head preparation')
        temporary = path.with_name(path.name + '.partial')
        torch.save(saved, temporary); temporary.replace(path)
        receipt = {'status': 'COMPLETE', 'source': identity, 'fold': int(fold), 'stage': 'prototype',
            'sha256': runtime.sha(path), 'legacy_source': LEGACY_SOURCE,
            'legacy_prepared_sha256': legacy_receipt['sha256'], 'new_convex_fits': 7,
            'reused_full_CP_fits': 1, 'audit': saved['audit']}
        runtime.atomic_json(path.with_suffix('.json'), receipt)
        return saved, receipt


class Data:
    """Same raw neural inputs and controls, replaced independent frozen main."""
    def __init__(self, fold, seed, device='cuda'):
        runtime = _runtime()
        require(int(fold) in runtime.cfg()['prototype']['folds'] and int(seed) in runtime.cfg()['prototype']['seeds'],
                'cell outside locked prototype')
        inherited = old.Data(fold, 'prototype', seed, device=device)
        saved, receipt = read_prepared(fold)
        require(inherited.prepared_sha256 == saved['legacy_prepared_sha256'], 'inherited neural inputs have a different legacy artifact')
        self.fold, self.seed, self.stage, self.role = int(fold), int(seed), 'prototype', 'inner'
        for name in ('fit', 'valid', 'active', 'visual', 'mask', 'owner', 'R', 'C', 'wrong_rna', 'wrong_c'):
            setattr(self, name, getattr(inherited, name))
        self.base = torch.as_tensor(np.array(saved['base'], copy=True), dtype=torch.float64, device=device)
        self.R0_valid = np.array(saved['M0_valid'], np.float64, copy=True)
        self.M0_valid = self.R0_valid.copy()
        self.CP_valid, self.RNA_valid = (np.array(saved[key], np.float64, copy=True) for key in ('CP_valid', 'RNA_valid'))
        self.scale = float(saved['scale'])  # Diagnostic only; neural bound is fixed 0.1.
        self._fit_time, self._fit_event = inherited.fit_labels()
        self.private_mainstate = saved['mainstate']
        self.prepared_path, self.prepared_sha256 = str(prepared_path(fold)), receipt['sha256']
        self.audit = {**saved['audit'], 'prepared_sha256': receipt['sha256'],
            'local_visual': copy.deepcopy(inherited.audit['local_visual']),
            'wrong_controls': copy.deepcopy(inherited.audit['wrong_controls']),
            'neural_inputs_reused_from_exact_legacy': True, 'data_constructor_new_fits': 0,
            'risk_correction_uses_diagnostic_SD': False, 'valid_endpoint_retained': False}

    def fit_labels(self):
        return self._fit_time.copy(), self._fit_event.copy()

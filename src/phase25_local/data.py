"""Fit-local raw-feature Ridge anchors and endpoint-separated local inputs.

Every crossfit teacher starts from raw covariates. Only one-dimensional risk
offsets cross teacher boundaries; RNA/PCA latent coordinates never do. Patient
arrays, fitted coefficients and preprocessing states remain in private files.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import warnings

import numpy as np
from sklearn.decomposition import PCA
import torch

from src.phase10_trimodal.features import CLINICAL
from src.phase10_trimodal.math import fit_ridge
from src.phase13_rna import data as parent
from src.phase23_cdei.data import partition_derangement, raw_rna_missing_patterns

GLOBAL_VISUAL = Path('/root/phase16r_light_rna_private')
LOCAL_VISUAL = Path('/root/autodl-tmp/phase17_ssd_private/visual')
GROUPS = Path(__file__).resolve().parents[2] / 'configs/phase10_molecular_groups.json'
RNA_DIM = 512
PCA_DIM = 32


def _runtime():
    from . import runtime
    return runtime


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def array_hash(value):
    value = np.ascontiguousarray(value)
    metadata = json.dumps({'shape': list(value.shape), 'dtype': str(value.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(metadata + value.tobytes()).hexdigest()


def state_hash(state):
    def reduce(value):
        if isinstance(value, np.ndarray):
            return {'array_sha256': array_hash(value)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): reduce(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
        if isinstance(value, (tuple, list)):
            return [reduce(item) for item in value]
        return value
    return hashlib.sha256(json.dumps(reduce(state), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _indices(index, n, name):
    index = np.asarray(index)
    _require(index.ndim == 1 and np.issubdtype(index.dtype, np.integer) and len(index)
             and len(np.unique(index)) == len(index) and (index >= 0).all() and (index < n).all(),
             name + ' must be nonempty unique patient indices')
    return np.array(index, np.int64, copy=True)


def _fit_scale(values, fit):
    values = np.asarray(values, np.float64)
    _require(values.ndim == 2 and np.isfinite(values[fit]).all(), 'nonfinite fit feature')
    mean, sd = values[fit].mean(0), values[fit].std(0)
    return {'mean': mean, 'sd': np.where(sd < 1e-8, 1., sd), 'fit_hash': array_hash(fit)}


def _scale(values, state):
    result = (np.asarray(values, np.float64) - state['mean']) / state['sd']
    _require(np.isfinite(result).all(), 'nonfinite transformed feature')
    return result


def _clinical_raw(raw, coding):
    parts = []
    for field in CLINICAL:
        values = np.asarray(raw['clinical_' + field]).astype(str)
        if field == 'AGE':
            numbers = np.array([float(x) if x.replace('.', '', 1).isdigit() else np.nan for x in values])
            parts.append(np.where(np.isfinite(numbers), numbers, coding[field]['median'])[:, None])
        else:
            parts.append(np.column_stack([values == level for level in coding[field]['levels']]))
    return np.concatenate(parts, axis=1).astype(np.float64)


def _clinical_fit(raw, fit):
    coding = {}
    for field in CLINICAL:
        values = np.asarray(raw['clinical_' + field]).astype(str)
        if field == 'AGE':
            numbers = np.array([float(x) if x.replace('.', '', 1).isdigit() else np.nan for x in values[fit]])
            _require(np.isfinite(numbers).any(), 'no measured training age')
            coding[field] = {'median': float(np.nanmedian(numbers))}
        else:
            coding[field] = {'levels': np.unique(values[fit]).tolist()}
    return coding


def _rna_fit(raw, fit, groups):
    values = np.asarray(raw['rna_genes'], np.float64)
    observed = np.isfinite(values[fit])
    _require(not np.isinf(values[fit]).any(), 'infinite training RNA')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values[fit], axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    filled = np.where(observed, values[fit], median)
    mean, sd = filled.mean(0), filled.std(0)
    eligible = np.flatnonzero((observed.mean(0) >= .9) & (sd > 1e-8))
    symbols = np.asarray(raw['rna_gene_symbols']).astype(str)
    _require(len(symbols) == values.shape[1] and len(np.unique(symbols)) == len(symbols), 'RNA symbols are not unique/aligned')
    union = set().union(*(set(value) for value in groups.values()))
    annotation = np.array([symbol in union for symbol in symbols])
    pathway = eligible[annotation[eligible]]
    selected = pathway[np.argsort(-sd[pathway], kind='stable')[:RNA_DIM]]
    fallback = 0
    if len(selected) < RNA_DIM:
        other = eligible[~annotation[eligible]]
        addition = other[np.argsort(-sd[other], kind='stable')[:RNA_DIM-len(selected)]]
        fallback = len(addition)
        selected = np.r_[selected, addition]
    _require(len(selected) == RNA_DIM, 'fewer than 512 eligible measured RNA genes')
    membership = np.array([[symbol in members for symbol in symbols[selected]] for members in groups.values()], bool)
    _require(membership.shape == (8, RNA_DIM) and membership.any(1).all(), 'empty real RNA program; no invented fallback membership')
    return {'indices': selected, 'symbols': symbols[selected], 'median': median[selected],
            'scale': {'mean': mean[selected], 'sd': np.where(sd[selected] < 1e-8, 1., sd[selected]), 'fit_hash': array_hash(fit)},
            'membership': membership, 'group_names': list(groups), 'fallback_nonHall_genes': fallback,
            'eligible_count': len(eligible), 'fit_hash': array_hash(fit)}


def _aligned_cna(raw, indices):
    source = np.asarray(raw['cna_genes'], np.float64)
    values = np.full((len(source), RNA_DIM), np.nan, np.float64)
    available = indices >= 0
    values[:, available] = source[:, indices[available]]
    _require(not np.isinf(values).any(), 'infinite CNA dosage')
    return values


def _molecular(raw, rna, cna_indices):
    raw_rna = np.asarray(raw['rna_genes'], np.float64)[:, rna['indices']]
    _require(not np.isinf(raw_rna).any(), 'infinite active RNA')
    rna_measured = np.isfinite(raw_rna)
    r = _scale(np.where(rna_measured, raw_rna, rna['median']), rna['scale'])
    cna = _aligned_cna(raw, cna_indices)
    measured = np.isfinite(cna)
    membership = rna['membership'].astype(np.float64)
    denominator = measured.astype(np.float64) @ membership.T
    safe = np.where(measured, cna, 0.)
    gain = np.maximum(safe, 0.) @ membership.T / np.maximum(denominator, 1.)
    loss = np.maximum(-safe, 0.) @ membership.T / np.maximum(denominator, 1.)
    program_sizes = membership.sum(1)
    return {'R': r, 'GAIN': gain, 'LOSS': loss,
            'RCOV': rna_measured.astype(np.float64) @ membership.T / program_sizes,
            'CCOV': denominator / program_sizes}, measured


def fit_preprocessor(raw, fit, groups):
    """No endpoint argument: fit every learned transform on raw train rows."""
    n = len(raw['rna_genes'])
    fit = _indices(fit, n, 'preprocessing fit')
    _require(len(groups) == 8, 'eight externally fixed programs required')
    coding = _clinical_fit(raw, fit)
    c_raw = _clinical_raw(raw, coding)
    static = np.asarray(raw['static'], np.float64)
    global_conch = np.asarray(raw['global_matched_conch'], np.float64)
    _require(static.shape == (n, 32) and global_conch.ndim == 2 and len(global_conch) == n,
             'S32 or global visual dimensions changed')
    _require(PCA_DIM <= min(len(fit), global_conch.shape[1]), 'insufficient teacher patients for fixed PCA32')
    before = _fit_scale(global_conch, fit)
    # Fitting never transforms held rows first; their values cannot affect SVD.
    fit_global = _scale(global_conch[fit], before)
    pca = PCA(PCA_DIM, svd_solver='full', whiten=False).fit(fit_global)
    pca_scores = (_scale(global_conch, before) - pca.mean_) @ pca.components_.T
    rna = _rna_fit(raw, fit, groups)
    cna_symbols = np.asarray(raw['cna_gene_symbols']).astype(str)
    _require(len(np.unique(cna_symbols)) == len(cna_symbols), 'CNA symbols are not unique')
    lookup = {symbol: j for j, symbol in enumerate(cna_symbols)}
    cna_indices = np.array([lookup.get(symbol, -1) for symbol in rna['symbols']], np.int64)
    mol, measured = _molecular(raw, rna, cna_indices)
    scales = {key: _fit_scale(values, fit) for key, values in
              [('C', c_raw), ('S', static), ('T', pca_scores),
               ('GAIN', mol['GAIN']), ('LOSS', mol['LOSS']), ('RCOV', mol['RCOV']), ('CCOV', mol['CCOV'])]}
    state = {'fit': fit, 'fit_hash': array_hash(fit), 'clinical_coding': coding,
             'scales': scales, 'RNA': rna, 'CNA_indices': cna_indices,
             'global': {'before': before, 'components': pca.components_.copy(), 'mean': pca.mean_.copy(),
                        'explained_variance_ratio_sum': float(np.nan_to_num(pca.explained_variance_ratio_).sum())},
             'audit': {'fit_n': len(fit), 'RNA_fallback_nonHall_genes': rna['fallback_nonHall_genes'],
                       'RNA_program_sizes': rna['membership'].sum(1).astype(int).tolist(),
                       'CNA_mapped_selected_genes': int((cna_indices >= 0).sum()),
                       'CNA_measured_fit_gene_values': int(measured[fit].sum()),
                       'CNA_no_measured_fit_gene_values': bool(not measured[fit].any()),
                       'CNA_programs_without_any_fit_coverage': int((mol['CCOV'][fit].sum(0) == 0).sum()),
                       'clinical_fields': list(CLINICAL), 'extra_missing_age_predictor': False,
                       'raw_transform_endpoint_inputs': False}}
    state['state_hash'] = state_hash(state)
    return state


def transform(raw, state, active=None):
    """Transform with frozen state; inactive rows are zero in every branch."""
    n = len(raw['rna_genes'])
    active = np.arange(n) if active is None else _indices(active, n, 'active')
    c = _scale(_clinical_raw(raw, state['clinical_coding']), state['scales']['C'])
    s = _scale(raw['static'], state['scales']['S'])
    global_values = _scale(raw['global_matched_conch'], state['global']['before'])
    scores = (global_values - state['global']['mean']) @ state['global']['components'].T
    t = _scale(scores, state['scales']['T'])
    mol, _ = _molecular(raw, state['RNA'], state['CNA_indices'])
    blocks = {'Clinical': c, 'S32': s, 'GLOBAL_MATCHED_CONCH32': t, 'HALL512': mol['R'],
              'CNA_gain8': _scale(mol['GAIN'], state['scales']['GAIN']),
              'CNA_loss8': _scale(mol['LOSS'], state['scales']['LOSS']),
              'RNA_coverage8': _scale(mol['RCOV'], state['scales']['RCOV']),
              'CNA_coverage8': _scale(mol['CCOV'], state['scales']['CCOV'])}
    positions, start = {}, 0
    for name, values in blocks.items():
        positions[name] = (start, start + values.shape[1]); start += values.shape[1]
    result = {'X': np.concatenate(list(blocks.values()), axis=1),
              'C': c, 'R': mol['R'], 'CP': np.c_[c, s, t], 'blocks': positions}
    inactive = np.ones(n, bool); inactive[active] = False
    for key in ('X', 'C', 'R', 'CP'):
        result[key] = np.array(result[key], np.float64, copy=True)
        result[key][inactive] = 0.
    return result


def teacher_partitions(fit, fold, stage):
    """One fixed outcome-blind split, shared by every neural seed."""
    _require(stage in ('prototype', 'outer'), 'unknown stage')
    fit = np.asarray(fit, np.int64)
    seed = 250101 + int(fold) + (10000 if stage == 'outer' else 0)
    groups = np.array_split(np.random.default_rng(seed).permutation(fit), 3)
    _require(all(len(group) for group in groups), 'empty crossfit teacher holdout')
    return [(fit[~np.isin(fit, held)], np.sort(held)) for held in groups]


def build_prepared(raw, fit, valid, fit_time, fit_event, *, fold, stage, identity, groups,
                   convex_fitter=fit_ridge):
    """Exactly five convex fits; the API receives no heldout endpoint vector."""
    n = len(raw['rna_genes'])
    fit, valid = _indices(fit, n, 'fit'), _indices(valid, n, 'valid')
    _require(not np.intersect1d(fit, valid).size, 'fit/heldout patient overlap')
    active = np.sort(np.r_[fit, valid])
    t, e = np.asarray(fit_time, np.float64), np.asarray(fit_event, np.float64)
    _require(t.shape == e.shape == fit.shape and np.isfinite(t).all() and (t >= 0).all()
             and np.isin(e, (0, 1)).all() and e.sum() > 0, 'invalid fit-only endpoint vector')
    position = np.full(n, -1, int); position[fit] = np.arange(len(fit))
    teachers, oof = [], np.full(n, np.nan)
    fitted_count = 0
    for number, (train, held) in enumerate(teacher_partitions(fit, fold, stage)):
        local = position[train]
        _require(e[local].sum() > 0, 'outcome-blind teacher has zero events; no label-driven resplit')
        state = fit_preprocessor(raw, train, groups)
        features = transform(raw, state, active)
        coef, optimizer = convex_fitter(features['X'][train], t[local], e[local], 10.)
        fitted_count += 1
        coef = np.asarray(coef, np.float64)
        prediction = features['X'][held] @ coef
        _require(np.isfinite(prediction).all() and not np.isfinite(oof[held]).any(), 'teacher OOF prediction invalid/duplicated')
        oof[held] = prediction
        teachers.append({'teacher': number, 'train': train, 'held': held, 'state': state, 'coefficients': coef,
            'optimizer': optimizer, 'risk_held': prediction, 'lambda': 10.,
            'fit_feature_hash': array_hash(features['X'][train]), 'held_feature_hash': array_hash(features['X'][held]),
            'fit_endpoint_hash': array_hash(np.c_[t[local], e[local]]),
            'preprocessing_fit_hash': state['fit_hash'], 'coefficient_hash': array_hash(coef)})
    _require(np.isfinite(oof[fit]).all(), 'crossfit risk fails exact fit-patient cover')
    full_state = fit_preprocessor(raw, fit, groups)
    features = transform(raw, full_state, active)
    full_coef, full_optimizer = convex_fitter(features['X'][fit], t, e, 10.)
    cp_coef, cp_optimizer = convex_fitter(features['CP'][fit], t, e, 10.)
    fitted_count += 2
    full_coef, cp_coef = np.asarray(full_coef, np.float64), np.asarray(cp_coef, np.float64)
    full_fit_risk = features['X'][fit] @ full_coef
    r0_valid, cp_valid = features['X'][valid] @ full_coef, features['CP'][valid] @ cp_coef
    base = np.zeros(n, np.float64); base[fit] = oof[fit]; base[valid] = r0_valid
    scale = float(base[fit].std())
    _require(np.isfinite(base[active]).all() and np.isfinite(scale) and scale > 1e-8,
             'crossfitted R0 scale unavailable')
    correlation = float(np.corrcoef(oof[fit], full_fit_risk)[0, 1])
    _require(np.isfinite(correlation), 'crossfit/fullfit risk correlation undefined')
    audit = {'schema': 'phase25_strict_crossfit_raw_joint_ridge_v1', 'stage': stage, 'fold': int(fold),
        'fit_n': len(fit), 'valid_n': len(valid), 'fit_hash': array_hash(fit), 'valid_hash': array_hash(valid),
        'fullfit_main_parameters': len(full_coef), 'CP_parameters': len(cp_coef),
        'convex_fits': fitted_count, 'R0_full_fits': 1, 'R0_teacher_fits': 3, 'CP_only_fits': 1,
        'shared_across_neural_seeds': True, 'teacher_assignment_outcome_blind': True,
        'teacher_assignment_seed': 250101 + int(fold) + (10000 if stage == 'outer' else 0),
        'all_teacher_preprocessing_from_raw_fit_only': True, 'no_teacher_latent_stacking': True,
        'heldout_labels_passed_to_builder': False, 'lambda': 10., 'lambda_selected': False,
        'crossfit_fit_risk_SD': scale, 'fullfit_in_sample_risk_SD': float(full_fit_risk.std()),
        'crossfit_fullfit_risk_correlation': correlation,
        'crossfit_fullfit_risk_RMS_difference': float(np.sqrt(np.mean((oof[fit] - full_fit_risk) ** 2))),
        'full_preprocessing_audit': full_state['audit'],
        'teacher_preprocessing_audits': [teacher['state']['audit'] for teacher in teachers],
        'risk_scale_shift_solved_claimed': False, 'not_phase15_scalar_reproduction': True}
    result = {'source': identity, 'stage': stage, 'role': 'inner' if stage == 'prototype' else 'outer',
        'fold': int(fold), 'fit': fit, 'valid': valid, 'active': active, 'R': features['R'], 'C': features['C'],
        'base': base, 'R0_valid': r0_valid, 'CP_valid': cp_valid, 'scale': scale,
        'fit_time': t.copy(), 'fit_event': e.copy(), 'audit': audit,
        'mainstate': {'full_state': full_state, 'full_X': features['X'], 'CP_X': features['CP'],
            'blocks': features['blocks'], 'full_R0_coefficients': full_coef, 'CP_coefficients': cp_coef,
            'full_R0_optimizer': full_optimizer, 'CP_optimizer': cp_optimizer,
            'fullfit_risk_fit': full_fit_risk, 'crossfit_risk_fit': oof[fit], 'teachers': teachers}}
    if 'cancer_type' in raw:
        result['cancer_type'] = np.array(raw['cancer_type']).astype(str)
    if 'raw_rna_patterns' in raw:
        result['raw_rna_patterns'] = np.array(raw['raw_rna_patterns']).astype(str)
    validate_prepared(result, identity)
    return result


def validate_prepared(saved, identity):
    _require(saved.get('source') == identity, 'prepared source drift')
    n = len(saved['base']); fit, valid = _indices(saved['fit'], n, 'fit'), _indices(saved['valid'], n, 'valid')
    _require(not np.intersect1d(fit, valid).size, 'prepared fit/valid overlap')
    _require(np.array_equal(saved['active'], np.sort(np.r_[fit, valid])), 'prepared active scope drift')
    _require(np.asarray(saved['R']).shape == (n, RNA_DIM) and np.asarray(saved['C']).shape[0] == n, 'prepared query dimensions')
    ms = saved['mainstate']; full = ms['full_state']
    _require(np.array_equal(full['fit'], fit) and full['fit_hash'] == array_hash(fit), 'full preprocess fit drift')
    _require(full['state_hash'] == state_hash({k: v for k, v in full.items() if k != 'state_hash'}), 'full preprocess state drift')
    counts = np.zeros(n, int); reconstructed = np.zeros(n)
    _require(len(ms['teachers']) == 3, 'three teachers required')
    for teacher, (train, held) in zip(ms['teachers'], teacher_partitions(fit, saved['fold'], saved['stage'])):
        _require(np.array_equal(teacher['train'], train) and np.array_equal(teacher['held'], held), 'teacher partition drift')
        _require(np.array_equal(teacher['state']['fit'], train) and teacher['preprocessing_fit_hash'] == array_hash(train),
                 'teacher preprocess contains heldout rows')
        _require(teacher['state']['state_hash'] == state_hash({k: v for k, v in teacher['state'].items() if k != 'state_hash'}),
                 'teacher state hash drift')
        _require(teacher['lambda'] == 10. and array_hash(teacher['coefficients']) == teacher['coefficient_hash'],
                 'teacher coefficient or penalty drift')
        counts[held] += 1; reconstructed[held] = teacher['risk_held']
    _require((counts[fit] == 1).all() and counts[valid].sum() == 0, 'teacher OOF is not exact fit-only cover')
    _require(np.array_equal(saved['base'][fit], reconstructed[fit])
             and np.array_equal(ms['crossfit_risk_fit'], reconstructed[fit]), 'fit offset is not exact crossfit R0')
    for actual, expected, name in (
        (saved['R0_valid'], ms['full_X'][valid] @ ms['full_R0_coefficients'], 'fullfit heldout R0'),
        (saved['CP_valid'], ms['CP_X'][valid] @ ms['CP_coefficients'], 'CP heldout guard'),
        (ms['fullfit_risk_fit'], ms['full_X'][fit] @ ms['full_R0_coefficients'], 'fullfit fit diagnostic')):
        _require(np.allclose(actual, expected, rtol=0, atol=1e-10), name + ' replay failed')
    _require(np.array_equal(saved['base'][valid], saved['R0_valid']), 'heldout offset changed')
    _require(saved['scale'] == float(saved['base'][fit].std()) and saved['scale'] > 1e-8, 'offset scale is not fit-CF-only')
    _require(saved['audit']['convex_fits'] == 5 and saved['audit']['fullfit_main_parameters'] == len(ms['full_R0_coefficients']),
             'prepared convex accounting drift')
    for key in ('R', 'C', 'base', 'R0_valid', 'CP_valid', 'fit_time', 'fit_event'):
        _require(np.isfinite(saved[key]).all(), 'nonfinite prepared array: ' + key)
    return saved


def load_raw():
    """Read only covariate arrays; endpoint access is a separate legacy call."""
    runtime = _runtime()
    dataset_sha = runtime.sha(parent.DATA)
    local_ready = json.loads((LOCAL_VISUAL / 'READY.json').read_text())
    _require(local_ready.get('status') == 'PASS' and local_ready.get('dataset_sha256') == dataset_sha,
             'audited raw cohort/local visual dataset identity changed')
    # Legacy source metadata can contain object-dtype strings. Only the
    # explicit covariate keys below are loaded, after its audited byte hash.
    with np.load(parent.DATA, allow_pickle=True) as archive:
        keys = ['patient_ids', 'rna_genes', 'rna_gene_symbols', 'cna_genes', 'cna_gene_symbols',
                'static', 'cancer_type'] + ['clinical_' + field for field in CLINICAL]
        raw = {key: np.array(archive[key], copy=True) for key in keys}
        raw['raw_rna_patterns'] = raw_rna_missing_patterns(archive['raw_rna'])
    ids = np.asarray(raw['patient_ids']).astype(str)
    _require(len(ids) == 512 and len(np.unique(ids)) == 512
             and hashlib.sha256('\n'.join(ids).encode()).hexdigest()
             == '76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f',
             'canonical patient ordering changed')
    path = GLOBAL_VISUAL / 'VISUAL_SOURCES.npz'
    receipt_path = GLOBAL_VISUAL / 'VISUAL_SOURCES.json'
    receipt = json.loads(receipt_path.read_text())
    _require(receipt.get('status') == 'PASS' and receipt.get('sha256') == runtime.sha(path), 'GLOBAL_MATCHED source hash drift')
    with np.load(path) as archive:
        raw['global_matched_conch'] = np.asarray(archive['GLOBAL_MATCHED'], np.float64)
    _require(raw['global_matched_conch'].shape == (512, 768), 'full matched-global source shape drift')
    raw['source_audit'] = {'dataset_sha256': dataset_sha, 'global_visual_sha256': runtime.sha(path),
        'global_visual_receipt_sha256': runtime.sha(receipt_path), 'global_visual_scope': 'all_positive_area_matched_candidates_not_only_local512'}
    return raw


@contextlib.contextmanager
def _file_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0'); handle.flush()
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def prepared_path(fold, stage):
    return _runtime().PRIVATE / 'prepared' / f'{stage}_fold{int(fold)}.pt'


def read_prepared(fold, stage):
    runtime = _runtime(); path = prepared_path(fold, stage)
    _require(path.is_file() and path.with_suffix('.json').is_file(), 'prepared main effect missing; run authorized prepare first')
    receipt = json.loads(path.with_suffix('.json').read_text())
    identity = runtime.source()
    _require(receipt.get('status') == 'COMPLETE' and receipt.get('source') == identity
             and receipt.get('sha256') == runtime.sha(path), 'prepared receipt/hash/source drift')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    _require(saved.get('fold') == int(fold) and saved.get('stage') == stage, 'prepared role/fold mismatch')
    return validate_prepared(saved, identity), receipt


def prepare(fold, stage='prototype'):
    runtime = _runtime(); configuration = runtime.cfg(); identity = runtime.source()
    _require(stage in ('prototype', 'outer'), 'unknown preparation stage')
    if stage == 'outer':
        runtime.authorize_outer(identity)
    allowed = configuration[stage if stage == 'prototype' else 'outer']['folds']
    _require(int(fold) in allowed, 'fold outside locked matrix')
    _require(configuration['main']['lambda'] == 10. and configuration['main']['crossfit_folds'] == 3,
             'fixed convex/reference protocol drift')
    path = prepared_path(fold, stage)
    with _file_lock(path.with_suffix('.lock')):
        if path.exists():
            return read_prepared(fold, stage)
        _require(not path.with_suffix('.json').exists(), 'orphan preparation receipt')
        raw = load_raw(); role = 'inner' if stage == 'prototype' else 'outer'
        # The legacy registry momentarily loads endpoints. Only fit rows are
        # passed to the builder; no valid outcome is retained in prepared data.
        part = parent.splits({'patient_ids': raw['patient_ids']}, 'DSS', int(fold))
        fit, valid = (part[key] for key in (('inner_train', 'inner_val') if role == 'inner' else ('outer_train', 'outer_test')))
        endpoint = parent.endpoints({'patient_ids': raw['patient_ids']})['DSS']
        groups = json.loads(GROUPS.read_text(encoding='utf8'))
        saved = build_prepared(raw, fit, valid, endpoint['time'][fit], endpoint['event'][fit],
                               fold=int(fold), stage=stage, identity=identity, groups=groups)
        saved['source_audit'] = raw['source_audit']
        saved['audit']['source_audit'] = raw['source_audit']
        saved['audit']['groups_sha256'] = runtime.sha(GROUPS)
        saved['audit']['legacy_registry_temporarily_loads_protected_endpoints'] = True
        _require(runtime.source() == identity, 'source changed during preparation')
        temporary = path.with_name(path.name + '.partial')
        torch.save(saved, temporary); temporary.replace(path)
        receipt = {'status': 'COMPLETE', 'source': identity, 'stage': stage, 'fold': int(fold),
                   'sha256': runtime.sha(path), 'convex_fits': 5, 'shared_across_neural_seeds': True,
                   'audit': saved['audit']}
        runtime.atomic_json(path.with_suffix('.json'), receipt)
        return saved, receipt


def local_visual(active, dataset_sha256):
    runtime = _runtime()
    receipt_path = LOCAL_VISUAL / 'READY.json'
    ready = json.loads(receipt_path.read_text())
    _require(ready.get('status') == 'PASS' and ready.get('patients') == 512
             and ready.get('dataset_sha256') == dataset_sha256, 'local visual cohort receipt drift')
    names = ('conch', 'mask', 'owner', 'coords', 'local_row')
    for name in names:
        _require(runtime.sha(LOCAL_VISUAL / (name + '.npy')) == ready['hashes'][name], 'local visual hash drift: ' + name)
    owner = np.load(LOCAL_VISUAL / 'owner.npy')
    mask = np.load(LOCAL_VISUAL / 'mask.npy')
    features = np.load(LOCAL_VISUAL / 'conch.npy', mmap_mode='r')
    _require(features.shape == (*mask.shape, 768) and owner.shape == (len(features),)
             and mask.dtype == bool and np.array_equal(np.unique(owner), np.arange(512)), 'local visual shape/coverage drift')
    take = np.flatnonzero(np.isin(owner, active))
    selected, selected_mask, selected_owner = np.array(features[take], np.float32), np.array(mask[take]), np.array(owner[take], int)
    _require(selected_mask.any(1).all() and np.array_equal(np.unique(selected_owner), np.sort(active)), 'active patients missing local patches')
    _require(np.bincount(selected_owner, weights=selected_mask.sum(1), minlength=512).max() <= 512,
             'local patient patch budget exceeded')
    valid_features = selected[selected_mask]
    norm = np.linalg.norm(valid_features, axis=1, keepdims=True)
    _require(np.isfinite(valid_features).all() and (norm > 1e-12).all(), 'invalid frozen CONCH row')
    selected[selected_mask] = valid_features / norm
    selected[~selected_mask] = 0.
    return selected, selected_mask, selected_owner, {'receipt_sha256': runtime.sha(receipt_path),
        'hashes': {name: ready['hashes'][name] for name in names}, 'slides': len(take),
        'patches': int(selected_mask.sum()), 'same_candidates_no_RNA_preselection': True,
        'row_normalized_again_after_float16_storage': True, 'sqrt768_applied_in_model_not_data': True}


class Data:
    """Safe local inputs: fit labels only, shared prepared anchors, no auto-fit."""
    def __init__(self, fold, stage, seed, device='cuda'):
        if stage == 'inner':
            stage = 'prototype'
        runtime = _runtime(); configuration = runtime.cfg()
        _require(stage in ('prototype', 'outer'), 'unknown stage')
        _require(int(seed) in configuration['prototype' if stage == 'prototype' else 'outer']['seeds'], 'seed outside locked matrix')
        saved, receipt = read_prepared(int(fold), stage)
        self.fold, self.stage, self.seed = int(fold), stage, int(seed)
        self.role = 'inner' if stage == 'prototype' else 'outer'
        self.fit, self.valid, self.active = (np.array(saved[key], np.int64, copy=True) for key in ('fit', 'valid', 'active'))
        visual, mask, owner, visual_audit = local_visual(self.active, saved['source_audit']['dataset_sha256'])
        self.visual = torch.as_tensor(visual, dtype=torch.float32, device=device)
        self.mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        self.owner = owner
        self.R = torch.as_tensor(np.array(saved['R'], copy=True), dtype=torch.float32, device=device)
        self.C = torch.as_tensor(np.array(saved['C'], copy=True), dtype=torch.float32, device=device)
        self.base = torch.as_tensor(np.array(saved['base'], copy=True), dtype=torch.float64, device=device)
        self.scale = float(saved['scale'])
        self.R0_valid, self.CP_valid = (np.array(saved[key], np.float64, copy=True) for key in ('R0_valid', 'CP_valid'))
        self._fit_time, self._fit_event = np.array(saved['fit_time'], copy=True), np.array(saved['fit_event'], copy=True)
        self.wrong_rna, self.wrong_c = np.arange(len(self.base)), np.arange(len(self.base))
        pairing = {}
        for name, indices in (('fit', self.fit), ('valid', self.valid)):
            rna_map, ra = partition_derangement(saved['cancer_type'], saved['raw_rna_patterns'], indices, seed + 25000 + fold)
            c_map, ca = partition_derangement(saved['cancer_type'], saved['raw_rna_patterns'], indices, seed + 26000 + fold)
            self.wrong_rna[indices], self.wrong_c[indices] = rna_map[indices], c_map[indices]
            pairing[name] = {'RNA': ra, 'Clinical': ca}
        self.prepared_path = str(prepared_path(fold, stage))
        self.prepared_sha256 = receipt['sha256']
        self.private_mainstate = saved['mainstate']
        self.audit = {**saved['audit'], 'prepared_sha256': receipt['sha256'], 'source': saved['source'],
                      'local_visual': visual_audit, 'wrong_controls': pairing,
                      'full_cohort_endpoints_retained': False, 'valid_endpoint_retained': False,
                      'data_constructor_convex_fits': 0, 'frozen_main_shared_by_seed': True}

    def fit_labels(self):
        return self._fit_time.copy(), self._fit_event.copy()

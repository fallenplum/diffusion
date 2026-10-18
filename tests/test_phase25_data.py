"""Synthetic source/teacher isolation checks; never read actual cohort files.

The injected deterministic fitter tests data and label routing only. Numerical
Efron/stationarity qualification is provided by the inherited Cox tests and the
separately authorized real-data preparation qualification.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase25_local import data as mod


IDENTITY = 'f' * 64


@pytest.fixture(scope='module')
def cohort():
    rng = np.random.default_rng(250125)
    n, genes = 96, 640
    symbols = np.array(['gene' + str(j) for j in range(genes)])
    groups = {'program' + str(j): symbols[j * 80:(j + 1) * 80].tolist() for j in range(8)}
    raw = {'rna_genes': rng.normal(size=(n, genes)), 'rna_gene_symbols': symbols,
           'cna_genes': rng.normal(size=(n, genes)), 'cna_gene_symbols': symbols.copy(),
           'static': rng.normal(size=(n, 32)), 'global_matched_conch': rng.normal(size=(n, 40)),
           'cancer_type': np.where(np.arange(n) % 2, 'A', 'B'),
           'raw_rna_patterns': np.where(np.arange(n) % 4 < 2, 'pattern0', 'pattern1'),
           'clinical_AGE': np.array([str(30 + j % 50) for j in range(n)]),
           'DSS_time': np.arange(1., n + 1), 'DSS_event': (np.arange(n) % 3 == 0).astype(float)}
    raw['cna_genes'][rng.random((n, genes)) < .1] = np.nan
    raw['rna_genes'][rng.random((n, genes)) < .01] = np.nan
    for field in mod.CLINICAL[1:]:
        raw['clinical_' + field] = np.where(np.arange(n) % 2, 'categoryA', 'categoryB')
    return raw, groups, np.arange(72), np.arange(72, n)


def recorder(log):
    def fitter(x, time, event, lam):
        log.append({'x': np.array(x), 'time': np.array(time), 'event': np.array(event), 'lambda': lam})
        # A fixed-coordinate deterministic oracle deliberately responds to the
        # supplied supervision; it does not imitate a learned Cox estimator.
        coefficient = np.linspace(-.03, .05, x.shape[1]) * (1. + np.mean(time) / 100 + np.mean(event))
        return coefficient, {'gradient_max': 0., 'test_routing_oracle_not_Cox_fit': True}
    return fitter


def build(raw, groups, fit, valid, *, time=None, event=None, log=None):
    log = [] if log is None else log
    return mod.build_prepared(raw, fit, valid,
        raw['DSS_time'][fit] if time is None else time,
        raw['DSS_event'][fit] if event is None else event,
        fold=1, stage='prototype', identity=IDENTITY, groups=groups, convex_fitter=recorder(log))


@pytest.fixture(scope='module')
def prepared(cohort):
    return build(*cohort)


def test_training_state_and_train_features_ignore_all_held_covariates(cohort):
    raw, groups, fit, valid = cohort
    changed = copy.deepcopy(raw)
    for key in ('rna_genes', 'cna_genes', 'static', 'global_matched_conch'):
        changed[key][valid] = np.nan_to_num(changed[key][valid]) * 300 + 700
    changed['clinical_AGE'][valid] = '999'
    for field in mod.CLINICAL[1:]:
        changed['clinical_' + field][valid] = 'unseen_held_category'
    state = mod.fit_preprocessor(raw, fit, groups)
    modified = mod.fit_preprocessor(changed, fit, groups)
    assert state['state_hash'] == modified['state_hash']
    original_x = mod.transform(raw, state)
    changed_x = mod.transform(changed, modified)
    assert np.array_equal(original_x['X'][fit], changed_x['X'][fit])
    assert not np.array_equal(original_x['X'][valid], changed_x['X'][valid])


def test_all_main_columns_fit_scaled_and_query_coordinates_match_main(cohort):
    raw, groups, fit, valid = cohort
    state = mod.fit_preprocessor(raw, fit, groups)
    features = mod.transform(raw, state)
    x = features['X'][fit]
    assert np.max(np.abs(x.mean(0))) < 1e-12
    sd = x.std(0)
    assert np.allclose(sd[sd > 1e-8], 1., atol=1e-12)
    start, end = features['blocks']['HALL512']
    assert np.array_equal(features['R'], features['X'][:, start:end])
    c0, c1 = features['blocks']['Clinical']
    assert np.array_equal(features['C'], features['X'][:, c0:c1])
    assert features['CP'].shape[1] == features['C'].shape[1] + 64
    assert np.array_equal(features['CP'], features['X'][:, :features['CP'].shape[1]])
    assert not state['audit']['extra_missing_age_predictor']


def test_teacher_preprocessing_is_refit_from_raw_not_sliced_fullfit(cohort):
    raw, groups, fit, valid = cohort
    train, held = mod.teacher_partitions(fit, 1, 'prototype')[0]
    teacher = mod.fit_preprocessor(raw, train, groups)
    full = mod.fit_preprocessor(raw, fit, groups)
    assert teacher['fit_hash'] != full['fit_hash']
    assert not np.array_equal(teacher['global']['components'], full['global']['components'])
    changed = copy.deepcopy(raw)
    changed['rna_genes'][held] = np.nan_to_num(changed['rna_genes'][held]) * 500
    changed['global_matched_conch'][held] += 1000
    altered = mod.fit_preprocessor(changed, train, groups)
    assert teacher['state_hash'] == altered['state_hash']
    assert np.array_equal(mod.transform(raw, teacher)['X'][train], mod.transform(changed, altered)['X'][train])


def test_CNA_gain_loss_split_retains_simultaneous_aberrations_and_missingness(cohort):
    raw, groups, fit, valid = cohort
    state = mod.fit_preprocessor(raw, fit, groups)
    mutated = copy.deepcopy(raw)
    indices = state['CNA_indices']
    members = np.flatnonzero(state['RNA']['membership'][0])
    mutated['cna_genes'][0] = np.nan
    mutated['cna_genes'][0, indices[members[:4]]] = [1, 1, -1, -1]
    mol, measured = mod._molecular(mutated, state['RNA'], indices)
    assert mol['GAIN'][0, 0] == .5 and mol['LOSS'][0, 0] == .5
    assert mol['CCOV'][0, 0] == pytest.approx(4 / len(members))
    assert measured[0].sum() == 4
    mutated['cna_genes'][0] = np.nan
    mol, _ = mod._molecular(mutated, state['RNA'], indices)
    assert not mol['GAIN'][0].any() and not mol['LOSS'][0].any() and not mol['CCOV'][0].any()


def test_empty_real_RNA_program_fails_instead_of_inventing_membership(cohort):
    raw, groups, fit, valid = cohort
    changed = copy.deepcopy(groups)
    changed['program0'] = ['absent_gene']
    with pytest.raises(ValueError, match='empty real RNA program'):
        mod.fit_preprocessor(raw, fit, changed)


def test_missing_all_teacher_CNA_is_explicit_audit_not_silent_imputation(cohort):
    raw, groups, fit, valid = cohort
    changed = copy.deepcopy(raw)
    changed['cna_genes'][fit] = np.nan
    state = mod.fit_preprocessor(changed, fit, groups)
    assert state['audit']['CNA_no_measured_fit_gene_values']
    assert state['audit']['CNA_programs_without_any_fit_coverage'] == 8
    assert np.isfinite(mod.transform(changed, state)['X']).all()


def test_crossfit_assignment_uses_no_outcome_and_shared_seed_base(cohort):
    _, _, fit, _ = cohort
    splits = mod.teacher_partitions(fit, 1, 'prototype')
    direct = np.array_split(np.random.default_rng(250102).permutation(fit), 3)
    assert all(np.array_equal(held, np.sort(expected)) for (_, held), expected in zip(splits, direct))
    assert np.array_equal(np.sort(np.concatenate([held for _, held in splits])), fit)
    assert all(not np.intersect1d(train, held).size for train, held in splits)
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(splits, mod.teacher_partitions(fit, 1, 'outer')))


def test_only_five_fits_and_each_receives_exact_teacher_or_fullfit_labels(cohort):
    raw, groups, fit, valid = cohort
    calls = []
    saved = build(raw, groups, fit, valid, log=calls)
    assert len(calls) == 5 and saved['audit']['convex_fits'] == 5
    for call, (train, held) in zip(calls, mod.teacher_partitions(fit, 1, 'prototype')):
        assert np.array_equal(call['time'], raw['DSS_time'][train])
        assert np.array_equal(call['event'], raw['DSS_event'][train])
        assert call['lambda'] == 10.
    assert all(np.array_equal(call['time'], raw['DSS_time'][fit]) for call in calls[3:])
    assert calls[3]['x'].shape[1] > calls[4]['x'].shape[1]


def test_heldout_label_poison_cannot_change_any_teacher_fullfit_or_offset(cohort, prepared):
    raw, groups, fit, valid = cohort
    changed = copy.deepcopy(raw)
    changed['DSS_time'][valid] += 99999
    changed['DSS_event'][valid] = 1 - changed['DSS_event'][valid]
    altered = build(changed, groups, fit, valid)
    for key in ('base', 'R0_valid', 'CP_valid', 'R', 'C'):
        assert np.array_equal(prepared[key], altered[key])
    assert prepared['mainstate']['full_state']['state_hash'] == altered['mainstate']['full_state']['state_hash']


def test_teacher_held_label_poison_changes_other_models_but_not_that_teacher(cohort, prepared):
    raw, groups, fit, valid = cohort
    train, held = mod.teacher_partitions(fit, 1, 'prototype')[0]
    time, event = raw['DSS_time'][fit].copy(), raw['DSS_event'][fit].copy()
    time[np.isin(fit, held)] += 777
    changed = build(raw, groups, fit, valid, time=time, event=event)
    old, new = prepared['mainstate']['teachers'][0], changed['mainstate']['teachers'][0]
    assert np.array_equal(old['coefficients'], new['coefficients'])
    assert np.array_equal(old['risk_held'], new['risk_held'])
    assert old['fit_endpoint_hash'] == new['fit_endpoint_hash']
    assert not np.array_equal(prepared['mainstate']['full_R0_coefficients'], changed['mainstate']['full_R0_coefficients'])


def test_fit_offsets_are_true_crossfit_while_valid_uses_fullfit(prepared):
    fit, valid = prepared['fit'], prepared['valid']
    state = prepared['mainstate']
    assert np.array_equal(prepared['base'][fit], state['crossfit_risk_fit'])
    assert not np.allclose(prepared['base'][fit], state['fullfit_risk_fit'])
    assert np.array_equal(prepared['base'][valid], prepared['R0_valid'])
    assert prepared['scale'] == float(prepared['base'][fit].std())
    assert not prepared['audit']['risk_scale_shift_solved_claimed']


@pytest.mark.parametrize('tamper,match', [
    (lambda x: x['base'].__setitem__(x['fit'], x['mainstate']['fullfit_risk_fit']), 'not exact crossfit'),
    (lambda x: x['mainstate']['teachers'][0]['train'].__setitem__(0, x['valid'][0]), 'partition drift'),
    (lambda x: x['mainstate']['teachers'][0]['state']['scales']['S']['mean'].__setitem__(0, 999), 'state hash drift'),
    (lambda x: x.__setitem__('scale', x['scale'] * 2), 'scale is not fit-CF'),
    (lambda x: x['R0_valid'].__setitem__(0, 999), 'replay failed'),
])
def test_readout_scope_or_state_tamper_fails_closed(prepared, tamper, match):
    changed = copy.deepcopy(prepared)
    tamper(changed)
    with pytest.raises(ValueError, match=match):
        mod.validate_prepared(changed, IDENTITY)


def test_zero_event_teacher_blocks_without_outcome_resplit(cohort):
    raw, groups, fit, valid = cohort
    train, held = mod.teacher_partitions(fit, 1, 'prototype')[0]
    event = np.zeros(len(fit))
    event[np.isin(fit, held)] = 1
    calls = []
    with pytest.raises(ValueError, match='zero events; no label-driven resplit'):
        build(raw, groups, fit, valid, event=event, log=calls)
    assert calls == []


def test_constructor_only_exposes_fit_labels_and_shares_float64_main(cohort, prepared, monkeypatch):
    raw, groups, fit, valid = cohort
    saved = copy.deepcopy(prepared)
    saved['source_audit'] = {'dataset_sha256': 'd' * 64}
    configuration = {'prototype': {'seeds': [31, 47]}, 'outer': {'seeds': [31, 47, 83]}}
    fake_runtime = SimpleNamespace(cfg=lambda: configuration, PRIVATE=Path('private_test'))
    monkeypatch.setattr(mod, '_runtime', lambda: fake_runtime)
    monkeypatch.setattr(mod, 'read_prepared', lambda *a: (copy.deepcopy(saved), {'sha256': 'e' * 64}))
    def visual(active, dataset_sha):
        features = np.ones((len(active), 2, 768), np.float32) / np.sqrt(768)
        return features, np.ones(features.shape[:2], bool), active.copy(), {'synthetic': True}
    monkeypatch.setattr(mod, 'local_visual', visual)
    first, second = mod.Data(1, 'prototype', 31, device='cpu'), mod.Data(1, 'prototype', 47, device='cpu')
    assert first.stage == 'prototype' and first.role == 'inner'
    assert first.base.dtype == torch.float64
    assert first.R.dtype == first.C.dtype == first.visual.dtype == torch.float32
    assert np.array_equal(first.base.numpy(), second.base.numpy())
    assert np.array_equal(first.base[valid].numpy(), first.R0_valid)
    assert first.audit['fullfit_main_parameters'] == len(saved['mainstate']['full_R0_coefficients'])
    assert first.audit['data_constructor_convex_fits'] == 0
    assert not any(name in first.__dict__ for name in ('raw', 'd', 't', 'e', 'ep', 'valid_labels'))
    time, event = first.fit_labels(); time[0] += 1
    assert not np.array_equal(time, first.fit_labels()[0])
    for mapping in (first.wrong_rna, first.wrong_c):
        for index in (fit, valid):
            assert set(mapping[index]) == set(index) and not np.any(mapping[index] == index)
            assert np.array_equal(raw['raw_rna_patterns'][index], raw['raw_rna_patterns'][mapping[index]])


def test_inactive_rows_zeroed_in_every_query_and_main_feature(cohort):
    raw, groups, fit, valid = cohort
    state = mod.fit_preprocessor(raw, fit, groups)
    active = np.arange(80)
    features = mod.transform(raw, state, active)
    assert all(not features[key][80:].any() for key in ('X', 'CP', 'C', 'R'))


def test_prepare_outer_never_starts_without_explicit_runtime_authorization(monkeypatch):
    called = []
    fake = SimpleNamespace(cfg=lambda: {}, source=lambda: IDENTITY,
        authorize_outer=lambda _: (_ for _ in ()).throw(ValueError('outer not admitted')))
    monkeypatch.setattr(mod, '_runtime', lambda: fake)
    monkeypatch.setattr(mod, 'load_raw', lambda: called.append(1))
    with pytest.raises(ValueError, match='outer not admitted'):
        mod.prepare(1, 'outer')
    assert called == []

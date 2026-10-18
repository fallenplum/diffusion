"""Small synthetic routing tests; no actual cohort, PCA or Cox fitting."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase26_multiquery import data as mod

IDENTITY = 'c' * 64


@pytest.fixture(scope='module')
def inputs():
    rng = np.random.default_rng(2601); n = 18
    fit, valid = np.arange(12), np.arange(12, n)
    raw = {'CP': rng.normal(size=(n, 4)), 'R': rng.normal(size=(n, 512))}
    teachers = []
    for number, held in enumerate(np.array_split(fit, 3)):
        train = fit[~np.isin(fit, held)]
        state = {'fit': train.copy(), 'fit_hash': mod.old.array_hash(train), 'teacher_tag': number}
        state['state_hash'] = mod.old.state_hash(state)
        teachers.append({'teacher': number, 'train': train, 'held': held, 'state': state})
    full_state = {'fit': fit, 'fit_hash': mod.old.array_hash(fit)}
    full_state['state_hash'] = mod.old.state_hash(full_state)
    cp_coef = np.array([.05, -.1, .2, .3])
    legacy = {'source': mod.LEGACY_SOURCE, 'fold': 1, 'stage': 'prototype', 'fit': fit,
        'valid': valid, 'active': np.arange(n), 'base': np.zeros(n), 'R': raw['R'].copy(),
        'fit_time': np.arange(1., 13.), 'fit_event': (np.arange(12) % 2 == 0).astype(float),
        'CP_valid': raw['CP'][valid] @ cp_coef,
        'mainstate': {'teachers': teachers, 'full_state': full_state, 'CP_coefficients': cp_coef, 'CP_X': raw['CP']}}
    return raw, legacy, {'source': mod.LEGACY_SOURCE, 'sha256': 'a' * 64}


def transformation(raw, state, active):
    # Saved teacher-specific coordinates: distinct scales demonstrate that
    # only risks, rather than teacher latent vectors, may be assembled.
    multiplier = 1 + state['teacher_tag'] / 10
    return {'CP': raw['CP'] * multiplier, 'R': raw['R'] * multiplier}


def fit_oracle(calls):
    def fit(x, time, event, penalty):
        calls.append({'time': np.array(time), 'event': np.array(event), 'shape': x.shape, 'lambda': penalty})
        coef = np.linspace(-.02, .03, x.shape[1]) * (1 + np.mean(time) / 20)
        return coef, {'gradient_max': 0., 'routing_oracle_not_actual_Cox': True}
    return fit


def build(inputs, monkeypatch, *, mutate=None, calls=None):
    raw, legacy, receipt = copy.deepcopy(inputs)
    if mutate is not None:
        mutate(legacy)
    monkeypatch.setattr(mod.old, 'transform', transformation)
    calls = [] if calls is None else calls
    result = mod.build_prepared(legacy, receipt, raw, fold=1, identity=IDENTITY, convex_fitter=fit_oracle(calls))
    return result, legacy, receipt


def test_exactly_seven_fits_and_original_fullCP_coefficients_risk_unchanged(inputs, monkeypatch):
    calls = []
    result, legacy, _ = build(inputs, monkeypatch, calls=calls)
    assert len(calls) == 7 and result['audit']['new_convex_fits'] == 7
    assert [call['shape'][1] for call in calls] == [4, 512, 4, 512, 4, 512, 512]
    assert all(call['lambda'] == 10 for call in calls)
    assert np.array_equal(result['mainstate']['full_CP_coefficients'], legacy['mainstate']['CP_coefficients'])
    assert np.array_equal(result['CP_valid'], legacy['CP_valid'])
    assert not result['audit']['moment_alignment'] and not result['audit']['meta_training']


def test_teacher_label_indices_are_exact_and_both_branches_exclude_held(inputs, monkeypatch):
    calls = []
    result, legacy, _ = build(inputs, monkeypatch, calls=calls)
    position = {int(patient): i for i, patient in enumerate(legacy['fit'])}
    for teacher, cp_call, rna_call in zip(legacy['mainstate']['teachers'], calls[::2], calls[1::2]):
        take = np.array([position[int(patient)] for patient in teacher['train']])
        assert np.array_equal(cp_call['time'], legacy['fit_time'][take])
        assert np.array_equal(rna_call['time'], legacy['fit_time'][take])
        assert np.array_equal(cp_call['event'], legacy['fit_event'][take])
    assert np.array_equal(calls[-1]['time'], legacy['fit_time'])


def test_teacher_held_labels_cannot_change_its_CP_or_RNA_head(inputs, monkeypatch):
    original, _, _ = build(inputs, monkeypatch)
    def poison(legacy):
        held = legacy['mainstate']['teachers'][0]['held']
        legacy['fit_time'][np.isin(legacy['fit'], held)] += 500
    changed, _, _ = build(inputs, monkeypatch, mutate=poison)
    for key in ('CP_coefficients', 'RNA_coefficients', 'CP_risk_held', 'RNA_risk_held', 'fit_endpoint_sha256'):
        assert np.array_equal(original['mainstate']['teachers'][0][key], changed['mainstate']['teachers'][0][key])
    assert not np.array_equal(original['mainstate']['full_RNA_coefficients'], changed['mainstate']['full_RNA_coefficients'])


def test_scalar_branch_sums_preserve_exact_fit_CF_and_valid_fullfit(inputs, monkeypatch):
    saved, legacy, _ = build(inputs, monkeypatch)
    fit, valid = saved['fit'], saved['valid']
    assert np.array_equal(saved['base'][fit], saved['CP_cf_fit'] + saved['RNA_cf_fit'])
    assert np.array_equal(saved['base'][valid], saved['CP_valid'] + saved['RNA_valid'])
    assert np.array_equal(saved['M0_valid'], saved['CP_valid'] + saved['RNA_valid'])
    assert not np.allclose(saved['base'][fit], saved['mainstate']['full_main_fit_risk'])
    assert not saved['audit']['CF_fullfit_shift_assumed_solved']


@pytest.mark.parametrize('mutation,match', [
    (lambda s: s['CP_valid'].__setitem__(0, 999), 'strong CP'),
    (lambda s: s['mainstate']['full_CP_coefficients'].__setitem__(0, 999), 'strong CP'),
    (lambda s: s['base'].__setitem__(s['fit'], s['mainstate']['full_main_fit_risk']), 'exact independent'),
    (lambda s: s.__setitem__('legacy_source', IDENTITY), 'source mismatch'),
    (lambda s: s['mainstate']['teachers'][0]['state'].__setitem__('teacher_tag', 999), 'preprocessing reference'),
])
def test_replay_scope_anchor_or_source_corruption_rejected(inputs, monkeypatch, mutation, match):
    saved, legacy, receipt = build(inputs, monkeypatch)
    mutation(saved)
    with pytest.raises(ValueError, match=match):
        mod.validate_prepared(saved, legacy, receipt, IDENTITY)


def test_no_legacy_preprocessor_or_PCA_fit_is_called(inputs, monkeypatch):
    import sklearn.decomposition
    def fail(*args, **kwargs):
        raise AssertionError('preprocessing refit attempted')
    monkeypatch.setattr(mod.old, 'fit_preprocessor', fail)
    monkeypatch.setattr(sklearn.decomposition.PCA, 'fit', fail)
    result, _, _ = build(inputs, monkeypatch)
    assert result['audit']['new_preprocessing_fits'] == 0


def test_Data_reuses_exact_neural_inputs_and_only_replaces_own_main(inputs, monkeypatch):
    saved, legacy, _ = build(inputs, monkeypatch)
    fit, valid = legacy['fit'], legacy['valid']; n = len(legacy['base'])
    prior = SimpleNamespace(fit=fit, valid=valid, active=legacy['active'],
        visual=torch.ones((n, 2, 768)), mask=torch.ones((n, 2), dtype=torch.bool), owner=np.arange(n),
        R=torch.tensor(inputs[0]['R'], dtype=torch.float32), C=torch.ones((n, 4)),
        wrong_rna=np.roll(np.arange(n), 1), wrong_c=np.roll(np.arange(n), 2),
        prepared_sha256=saved['legacy_prepared_sha256'],
        audit={'local_visual': {'old_exact': True}, 'wrong_controls': {'old_exact': True}},
        fit_labels=lambda: (legacy['fit_time'].copy(), legacy['fit_event'].copy()))
    monkeypatch.setattr(mod.old, 'Data', lambda *a, **k: prior)
    monkeypatch.setattr(mod, 'read_prepared', lambda _: (saved, {'sha256': 'b' * 64}))
    monkeypatch.setattr(mod, '_runtime', lambda: SimpleNamespace(PRIVATE=Path('synthetic'),
        cfg=lambda: {'prototype': {'folds': [1, 2, 3], 'seeds': [31, 47]}}))
    ds = mod.Data(1, 31, device='cpu')
    assert ds.visual is prior.visual and ds.R is prior.R and ds.C is prior.C
    assert ds.owner is prior.owner and ds.wrong_rna is prior.wrong_rna
    assert ds.base.dtype == torch.float64
    assert np.array_equal(ds.base[valid].numpy(), ds.R0_valid)
    assert np.array_equal(ds.R0_valid, ds.CP_valid + ds.RNA_valid)
    assert not ds.audit['risk_correction_uses_diagnostic_SD']
    assert not any(key in ds.__dict__ for key in ('d', 'raw', 'ep', 'valid_time', 'valid_event'))

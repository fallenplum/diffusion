import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/bgb_en_outer.py'
SPEC = importlib.util.spec_from_file_location('bgb_en_outer_guard_tests', SOURCE)
outer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(outer)


def selection_fixture():
    config = {'prototype': {'inner_splits': 3, 'neural_seeds': [31, 47]},
              'R1': {'arms': ['P0', 'P1']}, 'R2': {'arms': ['O0', 'O1', 'O3']},
              'R3': {'arms': ['F0', 'F1', 'F5'], 'additional_real_retraining': []}}
    lineage = {'source': {'synthetic_model.py': 'frozen'}, 'endpoints_sha256': 'frozen_endpoint'}
    digest = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
    lock = {'status': 'LOCKED_BEFORE_OUTER_EVALUATION', 'outer_test_evaluated': False,
            'outer_folds': [1, 2, 3, 4, 5], 'source_lineage_sha256': digest,
            'config_sha256': 'configuration_bytes_sha', 'inner_report_sha256': 'inner_report_bytes_sha',
            'finalists': ['CP_STRONG', 'CP_SCALAR']}
    report = {'status': 'COMPLETE_CORE_INNER', 'no_outer_test_evaluation': True,
              'source_lineage_sha256': digest,
              'configuration_canonical_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              'completed_unique_cells': 33,
              'stage_counts': {stage: {'status': 'COMPLETE', 'completed_unique_cells': n,
                                       'planned_unique_cells': n} for stage, n in [('R1', 9), ('R2', 12), ('R3', 12)]}}
    return lock, report, lineage, config


def validate(lock, report, lineage, config, fold=1):
    return outer.validate_selection_lock(lock, report, lineage, config,
                                         'configuration_bytes_sha', 'inner_report_bytes_sha', fold)


def test_complete_outer_lock_binds_report_source_configuration_and_scope():
    lock, report, lineage, config = selection_fixture()
    assert validate(lock, report, lineage, config)['this_tool_fits_neural_models'] is False
    lock['finalists'].append('NESTED_INTERACTION')
    scope = validate(lock, report, lineage, config, fold=5)
    assert scope['executed_finalists'] == ['CP_STRONG', 'CP_SCALAR']
    assert scope['neural_finalists_not_executed'] == ['NESTED_INTERACTION']


@pytest.mark.parametrize('mutation', ['status', 'outer_labels', 'config', 'report_bytes', 'lineage',
                                      'finalist_missing', 'finalist_unknown', 'duplicate_finalist', 'fold_scope'])
def test_selection_lock_rejects_incompatible_decisions(mutation):
    lock, report, lineage, config = selection_fixture()
    if mutation == 'status': lock['status'] = 'RUNNING'
    elif mutation == 'outer_labels': lock['outer_test_evaluated'] = True
    elif mutation == 'config': lock['config_sha256'] = 'other'
    elif mutation == 'report_bytes': lock['inner_report_sha256'] = 'other'
    elif mutation == 'lineage': lock['source_lineage_sha256'] = 'other'
    elif mutation == 'finalist_missing': lock['finalists'] = ['CP_STRONG']
    elif mutation == 'finalist_unknown': lock['finalists'].append('UNLOCKED_NEURAL')
    elif mutation == 'duplicate_finalist': lock['finalists'].append('CP_SCALAR')
    else: lock['outer_folds'] = [1, 2]
    with pytest.raises(ValueError):
        validate(lock, report, lineage, config)


@pytest.mark.parametrize('mutation', ['source', 'canonical_config', 'not_complete', 'count', 'outer_access'])
def test_incomplete_or_altered_inner_evidence_cannot_open_outer(mutation):
    lock, report, lineage, config = selection_fixture()
    if mutation == 'source': report['source_lineage_sha256'] = 'other'
    elif mutation == 'canonical_config': config['R1']['arms'].append('P_UNLOCKED')
    elif mutation == 'not_complete': report['status'] = 'RUNNING'
    elif mutation == 'count': report['stage_counts']['R2']['completed_unique_cells'] = 11
    else: report['no_outer_test_evaluation'] = False
    with pytest.raises(ValueError):
        validate(lock, report, lineage, config)


def test_fold_outside_canonical_five_is_rejected():
    with pytest.raises(ValueError, match='fold scope'):
        validate(*selection_fixture(), fold=6)


def test_each_scalar_seed_has_its_own_crossfit_diagnostics_and_lean_replay_state(monkeypatch):
    rng = np.random.default_rng(19); n = 18; train = np.arange(12); valid = np.arange(12, n)
    raw = {'t': np.arange(1., n + 1), 'e': (np.arange(n) % 2 == 0).astype(int)}
    x = rng.normal(size=(n, 2)); rna = rng.normal(size=(n, 2))
    cp_full = x @ np.array([.2, -.1]); cp_full = (cp_full - cp_full[train].mean()) / cp_full[train].std()
    rna_full = rna @ np.array([.1, .3]); rna_full = (rna_full - rna_full[train].mean()) / rna_full[train].std()
    base = {'train': train, 'valid': valid, 'full_risks': {'LOCAL': cp_full, 'R': rna_full}}
    seen = []

    def prep(_raw, fit, membership):
        seen.append(fit.copy())
        return {'LOCAL': x, 'R': rna, 'state': {'fit': fit.copy()}}

    def fit(values, time, event, lam):
        return np.array([time.mean() / 20, time.std() / 10]), {'synthetic_only': True}

    monkeypatch.setattr(outer, 'prepare_partition', prep)
    monkeypatch.setattr(outer, 'fit_offset_ridge', fit)
    monkeypatch.setattr(outer, 'fit_scalar_same_teacher', lambda *args: (.2, {'synthetic_only': True}))
    # calibration_risks expects the original complete CP/RNA vectors only for
    # their n-patient dimension; their heldout values remain the full-fit risk.
    base.update(CP=cp_full.copy(), RNA=rna_full.copy())
    a = outer.recalibrate(raw, base, np.ones((1, 2), bool), seed=31)
    b = outer.recalibrate(raw, base, np.ones((1, 2), bool), seed=47)
    assert a['actual_new_Cox_fits'] == b['actual_new_Cox_fits'] == 6
    assert a['actual_new_scalar_fits'] == b['actual_new_scalar_fits'] == 1
    assert a['risk_distribution'] != b['risk_distribution']
    for calibration in (a, b):
        assert calibration['CP_cf'].shape == calibration['RNA_cf'].shape == (n,)
        assert np.array_equal(calibration['teacher_groups'][valid], np.full(len(valid), -1))
        assert np.array_equal(calibration['CP_cf'][valid], np.zeros(len(valid)))
        assert np.array_equal(calibration['CP'][valid], cp_full[valid])
        assert np.array_equal(calibration['RNA'][valid], rna_full[valid])
        assert np.array_equal(calibration['base'], calibration['CP'] + .2 * calibration['RNA'])
        for teacher in calibration['teachers']:
            assert not np.intersect1d(teacher['train'], teacher['held']).size
            assert not np.intersect1d(teacher['train'], valid).size
            assert set(teacher) == {'train', 'held', 'preprocessing', 'models'}
            assert np.array_equal(teacher['preprocessing']['fit'], teacher['train'])
            assert set(teacher['models']) == {'LOCAL', 'R'}

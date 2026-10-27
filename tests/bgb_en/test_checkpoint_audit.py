import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.bgb_en.survival import evaluate


SOURCE = Path(__file__).resolve().parents[2] / 'scripts/bgb_en_checkpoint_audit.py'
SPEC = importlib.util.spec_from_file_location('bgb_en_checkpoint_audit_tests', SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def state(fit):
    h = audit.index_hash(fit)
    return {'fit': fit.copy(), 'fit_hash': h, 'clinical': {'scale': {'fit_hash': h}}, 'S32': {'fit_hash': h},
            'molecular': {'scale': {'fit_hash': h}, 'selected_scale': {'fit_hash': h},
                          'fit_hash': h, 'selected': np.array([0, 1])},
            'visual': {name: {'before': {'fit_hash': h}, 'after': {'fit_hash': h}}
                       for name in ('LOCAL', 'GLOBAL')}}


def prepared_fixture():
    n = 9; train = np.arange(6); valid = np.arange(6, 9)
    rng = np.random.default_rng(4)
    features = {key: rng.normal(size=(n, 2)) for key in ('LOCAL', 'GLOBAL', 'CS', 'R')}
    features['state'] = state(train)
    models, risks = {}, {}
    for key in ('LOCAL', 'GLOBAL', 'CS', 'R'):
        coef = rng.normal(size=2); raw = features[key] @ coef
        mean, sd = raw[train].mean(), raw[train].std()
        models[key] = {'coef': coef, 'scale': {'mean': mean, 'sd': sd}}
        risks[key] = (raw - mean) / sd
    groups = np.full(n, -1); teachers = []
    for group, held in enumerate(np.array_split(train, 3)):
        fit = np.setdiff1d(train, held); groups[held] = group
        teachers.append({'train': fit, 'held': held, 'preprocessing': state(fit),
                         'models': {key: {'coef': np.array([.1, -.1]), 'scale': {'mean': 0., 'sd': 1.}}
                                    for key in ('LOCAL', 'CS', 'R')}})
    cp, cs, rna = risks['LOCAL'].copy(), risks['CS'].copy(), risks['R'].copy()
    cp[train] *= .5; cs[train] *= .8; rna[train] *= 1.2
    return {'features': features, 'train': train, 'valid': valid, 'fit_hash': audit.index_hash(train),
            'valid_hash': audit.index_hash(valid), 'teacher_groups': groups, 'teachers': teachers,
            'full_models': models, 'full_risks': risks, 'CP': cp, 'CS': cs, 'RNA': rna,
            'beta': .2, 'base': cp + .2 * rna}


def test_wrong_source_config_or_data_lineage_rejected_before_checkpoint_load(tmp_path):
    source = tmp_path / 'operator.py'; source.write_text('synthetic source\n')
    config = tmp_path / 'lock.json'; config.write_text('{}')
    lineage = {'protocol_sha256': audit.sha256(config), 'source': {'operator.py': audit.sha256(source)},
               'cohort_sha256': 'fixed-synthetic'}
    audit.verify_current_source(lineage, config_path=config, root=tmp_path, expected_lineage=lineage)
    wrong = copy.deepcopy(lineage); wrong['cohort_sha256'] = 'changed'
    with pytest.raises(audit.AuditError, match='SOURCE_DATA_LINEAGE'):
        audit.verify_current_source(lineage, config_path=config, root=tmp_path, expected_lineage=wrong)
    source.write_text('different mathematical implementation\n')
    with pytest.raises(audit.AuditError, match='SOURCE_SHA256'):
        audit.verify_current_source(lineage, config_path=config, root=tmp_path)
    config.write_text('{"changed":true}')
    with pytest.raises(audit.AuditError, match='CONFIG_SHA256'):
        audit.verify_current_source(lineage, config_path=config, root=tmp_path)


def test_teacher_guard_checks_both_heldout_members_and_all_preprocessor_hashes():
    prepared = prepared_fixture()
    result = audit.verify_teacher_lineage(prepared, 9)
    assert result['pair_both_members_excluded']
    assert result['teacher_Cox_models_checked'] == 9
    wrong = copy.deepcopy(prepared)
    wrong['teachers'][0]['train'][0] = wrong['teachers'][0]['held'][0]
    with pytest.raises(audit.AuditError, match='TEACHER_HELDOUT_OVERLAP'):
        audit.verify_teacher_lineage(wrong, 9)
    wrong = copy.deepcopy(prepared)
    wrong['teachers'][1]['preprocessing']['visual']['LOCAL']['before']['fit_hash'] = 'wrong'
    with pytest.raises(audit.AuditError, match='VISUAL_BEFORE_FIT_HASH'):
        audit.verify_teacher_lineage(wrong, 9)
    wrong = copy.deepcopy(prepared); wrong['teacher_groups'][6] = 0
    with pytest.raises(audit.AuditError, match='VALIDATION_LEAKAGE'):
        audit.verify_teacher_lineage(wrong, 9)


def test_actual_full_coefficient_replay_keeps_crossfit_and_fullfit_risks_distinct():
    prepared = prepared_fixture()
    result = audit.verify_full_anchor_predictions(prepared)
    assert result['maximum_risk_error'] == 0
    assert not np.array_equal(prepared['CP'][prepared['train']], prepared['full_risks']['LOCAL'][prepared['train']])
    wrong = copy.deepcopy(prepared); wrong['full_models']['LOCAL']['coef'][0] += 1.
    with pytest.raises(audit.AuditError, match='FULL_ANCHOR_RISK_CENTER|FULL_ANCHOR_RISK_SCALE|COEFFICIENT_REPLAY'):
        audit.verify_full_anchor_predictions(wrong)


def test_convex_saved_coefficients_and_saved_pca_are_replayed_without_refitting():
    prepared = prepared_fixture(); valid = prepared['valid']
    coef = np.array([.3, -.2])
    prepared['features']['rna'] = prepared['features']['R'].astype(np.float32)
    expected = prepared['features']['rna'].astype(float)[valid] @ coef
    actual = audit.rebuild_convex_risk({'coefficient': coef}, {'arm': 'O1'}, prepared)
    assert np.array_equal(actual, expected)
    pca = {'components': np.array([[0., 1.], [1., 0.]]), 'mean': np.array([.1, -.2]),
           'post_mean': np.array([.4, -.3]), 'post_scale': np.array([2., .5])}
    expected = (((prepared['features']['R'] - pca['mean']) @ pca['components'].T
                 - pca['post_mean']) / pca['post_scale'])[valid] @ coef
    actual = audit.rebuild_convex_risk({'coefficient': coef, 'feature_state': pca}, {'arm': 'O2_PCA32'}, prepared)
    assert np.array_equal(actual, expected)


def test_independent_risk_metrics_tied_pairs_ipcw_and_repair_match_saved_definition():
    tt, te = np.array([2., 2., 4., 6., 8.]), np.array([1, 0, 1, 0, 1])
    t, e = np.array([2., 3., 5., 7.]), np.ones(4, int)
    r, ref = np.array([3., 1., 2., 0.]), np.array([1., 3., 0., 2.])
    independent = audit.metrics_from_risk(tt, te, t, e, r, ref)
    saved = evaluate(tt, te, t, e, r, ref)
    assert audit.verify_metric_fields(independent, saved) == 0
    wrong = copy.deepcopy(saved); wrong['Uno_num'] += .01
    with pytest.raises(audit.AuditError, match='INDEPENDENT_METRIC'):
        audit.verify_metric_fields(independent, wrong)


def test_checkpoint_forward_error_gate_and_shape_fail_closed():
    saved = np.array([.2, -.1, .7])
    assert audit.verify_replayed_risk(saved, saved + 1e-7, 3) <= 2e-6
    with pytest.raises(audit.AuditError, match='FORWARD_RISK'):
        audit.verify_replayed_risk(saved, saved + 3e-6, 3)
    with pytest.raises(audit.AuditError, match='RISK_SHAPE'):
        audit.verify_replayed_risk(saved, saved[:2], 3)
    with pytest.raises(audit.AuditError, match='FORWARD_RISK'):
        audit.verify_replayed_risk(saved, np.full(3, np.nan), 3)


def test_f7_import_requires_separate_actual_same_source_runtime_replays():
    report = dict(schema='bgb_en_checkpoint_replay_audit_v1', F7_runtime_repair=True, arm_filter=['F7'], device='cuda',
        passed_units=6, failed_units=0, whole_requested_matrix_complete_and_replayed=True,
        audit_script_sha256='audit', independent_reference_sha256='ref', recovery_audit_sha256='guard',
        source_lineage_sha256='core', units=[dict(stage='R3', arm='F7', inner_partition=part, seed=seed,
            status='PASS', replay_type='ACTUAL_FINAL80_STATE_DICT_FORWARD', F7_runtime_repair=True,
            runtime_receipt_sha256='runtime') for part in (0, 1, 2) for seed in (31, 47)])
    result = audit.imported_f7_units(report, lineage_sha='core', audit_sha='audit', reference_sha='ref', recovery_audit_sha='guard')
    assert len(result) == 6
    wrong = copy.deepcopy(report); wrong['units'][0]['runtime_receipt_sha256'] = None
    with pytest.raises(audit.AuditError, match='CELL_SET_DIFFERS'):
        audit.imported_f7_units(wrong, lineage_sha='core', audit_sha='audit', reference_sha='ref', recovery_audit_sha='guard')
    wrong = copy.deepcopy(report); wrong['audit_script_sha256'] = 'stale'
    with pytest.raises(audit.AuditError, match='SOURCE_DIFFERS'):
        audit.imported_f7_units(wrong, lineage_sha='core', audit_sha='audit', reference_sha='ref', recovery_audit_sha='guard')


def test_programmatic_runtime_flag_cannot_bypass_pre_torch_backend_bootstrap():
    from types import SimpleNamespace
    assert audit.F7_REPAIR_BACKEND is None
    with pytest.raises(audit.AuditError, match='BOOTSTRAP_MISSING'):
        audit.run(SimpleNamespace(runtime_repair=True))

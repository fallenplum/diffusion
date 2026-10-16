"""Synthetic leakage, identity and private-cache checks; no patient files."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.phase19_attention import data


def test_measured_program_mean_separates_missingness_and_signed_channels():
    values = np.array([[2., 4., np.nan, 1000.], [6., np.nan, -2., 2.]])
    measured = np.array([[True, True, False, False], [True, False, True, True]])
    membership = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]], bool)
    result, present = data.measured_program_mean(values, measured, membership)
    np.testing.assert_array_equal(result, [[3., 0., 4.], [6., 0., 0.]])
    np.testing.assert_array_equal(present, [[True, False, True], [True, True, False]])


def test_fit_pca_is_unchanged_by_validation_and_inactive_feature_mutations():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(90, 50))
    fit, active = np.arange(50), np.arange(75)
    original, state, audit = data.fit_pca_active(values, fit, active, 32)
    modified = values.copy()
    modified[50:75] += rng.normal(size=(25, 50)) * 1e6
    modified[75:] = np.nan  # Not even transformed by this role.
    changed, other, other_audit = data.fit_pca_active(modified, fit, active, 32)
    np.testing.assert_array_equal(original[fit], changed[fit])
    np.testing.assert_array_equal(state['pca'].components_, other['pca'].components_)
    assert audit == other_audit
    np.testing.assert_array_equal(original[75:], np.zeros((15, 32)))
    np.testing.assert_array_equal(changed[75:], np.zeros((15, 32)))
    np.testing.assert_allclose(original[fit].mean(0), 0., atol=1e-7)
    np.testing.assert_allclose(original[fit].std(0), 1., atol=2e-7)


def test_pca_rejects_nonfinite_active_or_insufficient_fit():
    x = np.ones((40, 45))
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        data.fit_pca_active(x, np.arange(35), np.arange(40), 32)
    with pytest.raises(ValueError):
        data.fit_pca_active(np.ones((40, 45)), np.arange(10), np.arange(40), 32)


def test_wrong_clinical_uses_only_cancer_and_no_other_fields():
    d = {'cancer_type': np.array(['A'] * 4 + ['B'] * 4)}
    ix = np.arange(8)
    a = data.cancer_derangement(d, ix, 19)
    d.update(clinical_AJCC_PATHOLOGIC_TUMOR_STAGE=np.arange(8),
             clinical_AGE=np.arange(8) + 50, clinical_SEX=np.arange(8))
    b = data.cancer_derangement(d, ix, 19)
    np.testing.assert_array_equal(a, b)
    assert not np.any(a == ix)
    assert set(a) == set(ix)
    assert data.mismatch_audit(d, ix, a)['cancer_mismatch_count'] == 0


@pytest.mark.parametrize('cancers', [['A', 'B', 'B', 'B'], ['A', 'B'], ['A', 'B', 'C']])
def test_singleton_cancer_fallback_is_reported_and_stays_in_partition(cancers):
    d = {'cancer_type': np.array(['X'] + cancers + ['Z'])}
    ix = np.arange(1, len(cancers) + 1)
    mapping = data.cancer_derangement(d, ix, 99)
    assert mapping[0] == 0 and mapping[-1] == len(mapping) - 1
    audit = data.mismatch_audit(d, ix, mapping)
    assert audit['identity_count'] == 0
    assert audit['cancer_mismatch_count'] > 0
    with pytest.raises(ValueError):
        data.mismatch_audit(d, ix[:-1], mapping)


def test_data_index_guard_rejects_outer_test_floats_and_negative_indices():
    instance = data.Data.__new__(data.Data)
    instance.active = np.array([0, 2, 4, 6])
    np.testing.assert_array_equal(instance.check_indices([2, 0]), [2, 0])
    for forbidden in ([1], [-1], [1.0], [[0, 2]]):
        with pytest.raises(ValueError):
            instance.check_indices(forbidden)


def test_cache_fits_once_and_rejects_corruption_and_provenance_drift(tmp_path):
    path = tmp_path / 'fold1_inner.pt'
    provenance = {'source': 'a', 'fit_hash': 'fit-only'}
    fits = []
    def producer():
        fits.append(1)
        return {'provenance': provenance, 'features': {'x': np.arange(10)}}
    a, digest = data._cached_bank(path, provenance, producer)
    b, other = data._cached_bank(path, provenance, producer)
    assert len(fits) == 1 and digest == other
    np.testing.assert_array_equal(a['features']['x'], b['features']['x'])
    with pytest.raises(ValueError, match='provenance'):
        data._cached_bank(path, {'source': 'new'}, producer)
    with path.open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='hash'):
        data._cached_bank(path, provenance, producer)


def test_orphan_cache_and_receipt_are_not_silently_rebuilt(tmp_path):
    path = tmp_path / 'fold1_outer.pt'
    path.with_suffix('.json').write_text(json.dumps({'sha256': 'bad'}))
    with pytest.raises(ValueError, match='orphan'):
        data._cached_bank(path, {}, lambda: {})
    path.with_suffix('.json').unlink()
    path.write_bytes(b'not-a-valid-bank')
    with pytest.raises(ValueError, match='receipt'):
        data._cached_bank(path, {}, lambda: {})


def test_active_visual_mean_uses_equal_slide_weights_not_patch_weights(tmp_path, monkeypatch):
    owner = np.array([0, 0, 1, 2])
    mask = np.zeros((4, 512), bool)
    mask[0, :1] = True; mask[1, :3] = True
    mask[2, :2] = True; mask[3, :4] = True
    visual = np.zeros((4, 512, 1536), np.float16)
    visual[0, :1] = 2; visual[1, :3] = 6
    visual[2, :2] = 10; visual[3, :4] = 20
    for name, value in [('owner', owner), ('mask', mask), ('u', visual)]:
        np.save(tmp_path / (name + '.npy'), value)
    monkeypatch.setattr(data, 'VISUAL', tmp_path)
    raw, valid, retained_owner, slides, mean = data.active_visuals([0, 2], patients=3)
    np.testing.assert_array_equal(retained_owner, [0, 0, 2])
    np.testing.assert_array_equal(slides, [0, 1, 3])
    np.testing.assert_array_equal(mean[0], np.full(1536, 4.))  # Not 5 by patch weighting.
    np.testing.assert_array_equal(mean[1], np.zeros(1536))
    np.testing.assert_array_equal(mean[2], np.full(1536, 20.))
    assert raw.shape == (3, 512, 1536) and valid.shape == (3, 512)


def test_import_closure_resolves_relative_and_module_imports_without_execution(tmp_path, monkeypatch):
    (tmp_path / 'src/example').mkdir(parents=True)
    (tmp_path / 'src/example/a.py').write_text('from . import b\nfrom src.example.c import value\n')
    (tmp_path / 'src/example/b.py').write_text('raise RuntimeError("must never execute")\n')
    (tmp_path / 'src/example/c.py').write_text('value = 1\n')
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    files = data._import_closure([tmp_path / 'src/example/a.py'])
    assert {p.relative_to(tmp_path).as_posix() for p in files} == {
        'src/example/a.py', 'src/example/b.py', 'src/example/c.py'}


def test_source_requires_all_five_training_tests_and_excludes_evaluation(tmp_path, monkeypatch):
    (tmp_path / 'src/phase19_attention').mkdir(parents=True)
    (tmp_path / 'configs').mkdir(); (tmp_path / 'scripts').mkdir(); (tmp_path / 'tests').mkdir()
    (tmp_path / 'src/phase19_attention/attention.py').write_text('x = 1\n')
    config = tmp_path / 'configs/new.json'; config.write_text('{}')
    (tmp_path / 'configs/phase19_baseline_sources.json').write_text('{}')
    (tmp_path / 'scripts/run_phase19.py').write_text('x = 1\n')
    for name in data.TRAINING_TESTS:
        (tmp_path / 'tests' / f'test_phase19_{name}.py').write_text('x = 1\n')
    evaluation = tmp_path / 'tests/test_phase19_evaluation.py'; evaluation.write_text('x = 1\n')
    monkeypatch.setattr(data, 'ROOT', tmp_path); monkeypatch.setattr(data, 'CONFIG', config)
    monkeypatch.setattr(data, 'ANCESTOR_CONFIGS', [])
    monkeypatch.setattr(data, '_verify_external_sources', lambda: {'passed': True})
    a = data.source()
    evaluation.write_text('x = 2\n')
    assert data.source() == a
    (tmp_path / 'tests/test_phase19_model.py').unlink()
    with pytest.raises(ValueError, match='required training source missing'):
        data.source()


def test_feature_bank_zeroes_inactive_patients_and_keeps_separate_cna_channels(monkeypatch):
    rng = np.random.default_rng(1902)
    membership = np.zeros((8, 512), bool)
    for k in range(8):
        membership[k, k * 64:(k + 1) * 64] = True
    original = {key: rng.normal(size=(512, width)).astype(np.float32)
                for key, width in [('C', 6), ('S', 32), ('R', 512), ('G', 512), ('L', 512)]}
    original.update(RM=np.ones((512, 512), np.float32), CM=np.ones((512, 512), np.float32),
                    membership=membership.astype(np.float32))
    original['RM'][0, :64] = 0
    original['CM'][1, 64:128] = 0
    monkeypatch.setattr(data.old, 'molecular_features', lambda d, fold, role: (original, {'source': 'synthetic'}))
    fit, valid, active = np.arange(40), np.arange(40, 50), np.arange(50)
    visual_mean = rng.normal(size=(512, 64))
    bank = data._feature_bank({}, 1, 'inner', fit, valid, active, visual_mean, {'source': 'synthetic'})
    features = bank['features']
    for key, array in features.items():
        if key != 'membership':
            np.testing.assert_array_equal(array[50:], np.zeros_like(array[50:]), err_msg=key)
    np.testing.assert_array_equal(features['CS'], np.c_[features['C'], features['S']])
    assert bank['audit']['X_order'] == ['C', 'S', 'R', 'GN8', 'LN8', 'UNI32', 'RCOV8', 'CCOV8']
    assert features['RPCA32'].shape == (512, 32)
    assert features['CNAPCA32'].shape == (512, 32)
    assert not np.array_equal(features['GN8'], features['LN8'])
    assert bank['state']['CNAPCA32']['pca'].components_.shape == (32, 1024)
    for name, key in [('RCOV8', 'RM'), ('CCOV8', 'CM')]:
        raw_coverage = original[key] @ membership.T / membership.sum(1)[None]
        expected, _ = data.fit_scale_active(raw_coverage, fit, active)
        np.testing.assert_array_equal(features[name], expected)

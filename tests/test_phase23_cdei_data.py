"""Endpoint separation, read-only mask conversion and outcome-blind pairing."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase23_cdei import data


def example_parent(role='inner'):
    n = 12
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(n, 5))
    raw[1::2, -1] = np.nan
    R = rng.normal(size=(n, 512)).astype(np.float32)
    membership = np.zeros((8, 512), np.float32)
    for k in range(8): membership[k, k*64:(k+1)*64] = 1
    rm = np.ones_like(R)
    rm[1::2, :10] = 0
    cm = np.ones_like(R)
    cm[0, :64] = 0
    f = {'C': rng.normal(size=(n, 3)).astype(np.float32),
         'S': rng.normal(size=(n, 32)).astype(np.float32), 'R': R,
         'G': abs(R).copy(), 'L': abs(R).copy(), 'RM': rm, 'CM': cm,
         'membership': membership, 'X': rng.normal(size=(n, 9)).astype(np.float32),
         'ANCHOR': rng.normal(size=n).astype(np.float32)}
    f['CS'] = np.c_[f['C'], f['S']]
    fit, valid = np.arange(4), np.arange(4, 8)
    simple = {'source': 'old', 'fit': fit.copy(), 'valid': valid.copy(),
              'preprocess_audit': {'cache_sha256': 'b'*64},
              'models': {key: {'w': np.arange(9, dtype=np.float64),
                               'risk': f['ANCHOR'][valid].copy()}
                         for key in ('J0', 'J_CS')}}
    return SimpleNamespace(f=f, ft={k:torch.as_tensor(v) for k,v in f.items()},
        fold=1, role=role, fit=fit, valid=valid, owner=np.arange(8),
        visual=torch.ones(8, 2, 1536), mask=torch.ones(8, 2, dtype=torch.bool),
        ep={'time': np.arange(1, n+1, dtype=float), 'event': np.arange(n)%2},
        d={'rna_genes': raw, 'cancer_type': np.where(np.arange(n)%2, 'B', 'A')},
        anchor_w=np.arange(9, dtype=np.float64), anchor_sd_fit=1.25,
        anchor_reference={'checkpoint_sha256':'c'*64}, inherited_simple=simple,
        upstream_audit={'cache_sha256':'b'*64})


def make_data(monkeypatch, role='inner', inherited=None):
    inherited = example_parent(role) if inherited is None else inherited
    calls = []
    def loader(fold, stage, seed, device):
        calls.append((fold, stage, seed, device))
        return inherited
    monkeypatch.setattr(data.prior, 'Data', loader)
    return data.Data(1, role, 31, device='cpu'), inherited, calls


def test_wraps_qualified_parent_once_and_never_keeps_endpoint_or_raw_cohort(monkeypatch):
    ds, inherited, calls = make_data(monkeypatch)
    assert calls == [(1, 'inner', 31, 'cpu')]
    for obj in (ds, ds.feature_view):
        for name in ('d', 'ep', 'ev', 't', 'e', 'state', 'parts', 'part'):
            assert not hasattr(obj, name)
    view = ds.feature_view
    assert not hasattr(view, 'fit_labels')
    assert not hasattr(view, 'selection_labels')
    assert set(ds.feat(['C', 'R', 'G', 'L', 'RM', 'CM', 'X', 'ANCHOR'])) == {
        'C', 'R', 'G', 'L', 'RM', 'CM', 'X', 'ANCHOR'}
    with pytest.raises(ValueError, match='outside'):
        ds.feat(['event'])
    assert ds.audit['outer_test_labels_retained'] is False


def test_binary_mask_conversion_is_new_boolean_copy_and_preserves_own_features(monkeypatch):
    ds, inherited, _ = make_data(monkeypatch)
    for key in ('RM', 'CM', 'membership'):
        assert ds.ft[key].dtype == torch.bool
        assert ds.f[key].dtype == bool
        assert inherited.f[key].dtype == np.float32
        np.testing.assert_array_equal(ds.f[key], inherited.f[key])
    for key in ('C', 'R', 'G', 'L', 'X', 'ANCHOR'):
        assert torch.equal(ds.ft[key], inherited.ft[key])
    ds.ft['RM'].fill_(False)
    assert inherited.f['RM'].any()
    assert ds.anchor_sd_fit == 1.25
    assert ds.anchor_w.dtype == np.float64


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.1, .5, 2.])
def test_nonbinary_mask_never_silently_casts(bad):
    mask = np.ones((4, 512), np.float32)
    mask[0, 0] = bad
    with pytest.raises(ValueError, match='finite binary'):
        data.validated_binary_mask(mask)


def test_only_fit_and_explicit_inner_selection_labels_available_and_return_copies(monkeypatch):
    ds, inherited, _ = make_data(monkeypatch)
    expected = inherited.ep['time'][ds.fit].copy()
    assert set(ds.fit_labels()) == {'time', 'event'}
    np.testing.assert_array_equal(ds.fit_labels()['time'], expected)
    np.testing.assert_array_equal(ds.selection_labels()['time'], inherited.ep['time'][ds.valid])
    labels = ds.fit_labels()
    labels['time'].fill_(-10) if isinstance(labels['time'], torch.Tensor) else labels['time'].fill(-10)
    np.testing.assert_array_equal(ds.fit_labels()['time'], expected)


def test_outer_loader_never_indexes_or_retains_test_labels(monkeypatch):
    inherited = example_parent('outer')
    class FitOnlyLabels:
        def __init__(self, values): self.values = values
        def __getitem__(self, index):
            if np.intersect1d(index, inherited.valid).size:
                raise AssertionError('outer-test label was read')
            return self.values[index]
    inherited.ep = {k:FitOnlyLabels(v) for k,v in inherited.ep.items()}
    ds, _, _ = make_data(monkeypatch, 'outer', inherited)
    assert len(ds.fit_labels()['time']) == len(ds.fit)
    assert ds._selection_supervision is None
    with pytest.raises(ValueError, match='sealed'):
        ds.selection_labels()


def test_heldout_endpoint_poison_does_not_change_features_pairing_or_fit_supervision(monkeypatch):
    parent = example_parent('outer')
    first, _, _ = make_data(monkeypatch, 'outer', parent)
    parent.ep['time'][parent.valid] = np.nan
    parent.ep['event'][parent.valid] = 999
    second, _, _ = make_data(monkeypatch, 'outer', parent)
    assert first.audit == second.audit
    np.testing.assert_array_equal(first.wrong_o, second.wrong_o)
    np.testing.assert_array_equal(first.wrong_c, second.wrong_c)
    np.testing.assert_array_equal(first.fit_labels()['time'], second.fit_labels()['time'])
    for key in first.ft:
        assert torch.equal(first.ft[key], second.ft[key])


def test_derangements_are_partition_local_zero_self_and_pattern_preserving():
    cancer = np.array(['A','A','B','B','A','A','B','B','unused'])
    pattern = np.array(['x']*4+['y']*4+['unused'])
    ix = np.arange(8)
    mapping, audit = data.partition_derangement(cancer, pattern, ix, 31)
    assert (mapping[ix] != ix).all()
    assert set(mapping[ix]) == set(ix)
    np.testing.assert_array_equal(pattern[ix], pattern[mapping[ix]])
    np.testing.assert_array_equal(cancer[ix], cancer[mapping[ix]])
    assert mapping[8] == 8
    assert audit['relaxed_cancer_match_count'] == 0
    assert audit['no_DSS_Stage_Age_Sex_matching'] is True
    again, _ = data.partition_derangement(cancer, pattern, ix, 31)
    np.testing.assert_array_equal(mapping, again)


@pytest.mark.parametrize('cancer', [['A','A','B'], ['A','B'], ['A','B','C'], ['A','A','B','C']])
def test_singleton_cancer_relaxation_stays_inside_missing_pattern(cancer):
    patterns = ['same']*len(cancer)
    ix = np.arange(len(cancer))
    mapping, audit = data.partition_derangement(cancer, patterns, ix, 47)
    assert (mapping[ix] != ix).all()
    assert set(mapping[ix]) == set(ix)
    assert audit['relaxed_cancer_match_count'] >= 2
    assert audit['missing_pattern_mismatch_count'] == 0


def test_missing_pattern_singleton_blocks_instead_of_self_pair_or_patient_exclusion():
    with pytest.raises(data.PairingUnavailableError) as found:
        data.partition_derangement(['A','A','A'], ['same','same','unique'], [0,1,2], 31)
    assert found.value.audit['status'] == 'UNAVAILABLE_PATTERN_SINGLETON'
    assert found.value.audit['patients'] == 3
    assert found.value.audit['pattern_singleton_patients'] == 1
    assert 'indices' not in found.value.audit


def test_full_raw_rna_missing_patterns_do_not_depend_on_observed_expression_values():
    raw = np.array([[1,np.nan,4],[7,np.nan,3],[np.nan,9,3]], float)
    before = data.raw_rna_missing_patterns(raw)
    raw[np.isfinite(raw)] *= -1024
    after = data.raw_rna_missing_patterns(raw)
    np.testing.assert_array_equal(before, after)
    assert before[0] == before[1] and before[0] != before[2]


def test_new_wrong_controls_do_not_access_stage_age_sex_or_endpoint_dictionary(monkeypatch):
    inherited = example_parent()
    class AllowedSource(dict):
        def __getitem__(self, key):
            if key not in ('rna_genes','cancer_type'):
                raise AssertionError('unapproved matching variable')
            return super().__getitem__(key)
    inherited.d = AllowedSource(inherited.d)
    ds, _, _ = make_data(monkeypatch, inherited=inherited)
    assert ds.audit['wrong_controls']['fit']['omics']['identity_count'] == 0
    assert ds.audit['wrong_controls']['valid']['clinical']['identity_count'] == 0


def test_pair_support_audit_retains_actual_cna_missingness():
    parent = example_parent()
    mapping = np.arange(12)
    mapping[[0, 2]] = [2, 0]
    audit = data.pair_mask_summary(parent.f['RM'], parent.f['CM'], parent.f['membership'],
                                   np.array([0, 2]), mapping)
    assert audit['patients'] == 2
    assert audit['token_slots'] == 24
    assert audit['support_changed_patients'] == 2
    assert audit['shared_valid_tokens_patient_min'] == 22
    assert not any(isinstance(v, (list, np.ndarray)) for v in audit.values())


def test_feature_queries_reject_inactive_patients_or_noninteger_indices(monkeypatch):
    ds, _, _ = make_data(monkeypatch)
    np.testing.assert_array_equal(ds.check_indices([1,1,2]), [1,1,2])
    with pytest.raises(ValueError, match='active'):
        ds.check_indices([9])
    with pytest.raises(ValueError, match='integer'):
        ds.check_indices([1.0])


def test_new_source_identity_binds_new_config_without_expectation_of_future_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    config = tmp_path/'configs/phase23_cdei_g0.json'
    config.parent.mkdir()
    config.write_text('{}')
    monkeypatch.setattr(data, 'CONFIG', config)
    monkeypatch.setattr(data.prior, 'source_files', lambda: {})
    monkeypatch.setattr(data.prior.prior, '_import_closure', lambda files: files)
    (tmp_path/'src/phase23_cdei').mkdir(parents=True)
    (tmp_path/'src/phase23_cdei/data.py').write_text('# temporary source identity\n')
    before = data.source()
    config.write_text('{"locked":true}')
    assert before != data.source()

"""Synthetic fold/mask/source/H5 tests; no real cohort or neural fits."""
import copy
import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from src.phase28_reread import data as mod


def aligned_example():
    rna = np.array([[2., 0.], [4., np.nan], [99., 9.], [3., 1.]])
    cna = np.array([[2., -2.], [4., -4.], [-8., 8.], [np.nan, 0.]])
    return mod.align_all_genes(rna, ['R', 'COMMON'], cna, ['C', 'COMMON'])


def test_union_real_gene_identity_and_raw_gain_loss_before_scaling():
    values, mask, symbols = aligned_example()
    assert symbols.tolist() == ['C', 'COMMON', 'R']
    assert values.dtype == np.float32 and mask.dtype == bool
    assert values.shape == mask.shape == (4, 3, 3)
    # Existing RNA numbers are retained: there is no second log transform.
    np.testing.assert_array_equal(values[:, 2, 0], [2., 4., 99., 3.])
    assert values[0, 1, 0] == 0 and mask[0, 1, 0]  # observed zero
    assert values[1, 1, 0] == 0 and not mask[1, 1, 0]  # missing
    np.testing.assert_array_equal(values[:, 0, 1], [2., 4., 0., 0.])
    np.testing.assert_array_equal(values[:, 0, 2], [0., 0., 8., 0.])
    assert not mask[3, 0, 1] and not mask[3, 0, 2]
    assert mask[3, 1, 1] and mask[3, 1, 2]  # measured neutral CNA
    assert np.array_equal(mask[:, :, 1], mask[:, :, 2])
    assert not mask[:, 0, 0].any() and not mask[:, 2, 1:].any()


def test_measured_fit_statistics_population_SD_zeroSD_and_no_valid_selection():
    values, mask, _ = aligned_example()
    state = mod.fit_molecular_state(values, mask, np.array([0, 1]))
    assert state['count'][1, 0] == 1 and state['mean'][1, 0] == 0 and state['sd'][1, 0] == 1
    assert state['mean'][2, 0] == 3 and state['sd'][2, 0] == 1
    assert state['mean'][0, 1] == 3 and state['sd'][0, 1] == 1
    assert state['mean'][0, 2] == 0 and state['sd'][0, 2] == 1
    poisoned = values.copy(); poisoned[2:] += 1e8
    altered_mask = mask.copy(); altered_mask[2:] = ~altered_mask[2:]
    again = mod.fit_molecular_state(poisoned, altered_mask, np.array([0, 1]))
    for key in ('count', 'mean', 'sd'): np.testing.assert_array_equal(state[key], again[key])


def test_no_fit_measurement_disables_channel_even_when_valid_is_measured():
    values, mask, _ = aligned_example()
    mask[:2, 1, 0] = False
    state = mod.fit_molecular_state(values, mask, np.array([0, 1]))
    result, effective = mod.transform_molecular(values, mask, state)
    assert state['count'][1, 0] == 0
    assert not effective[:, 1, 0].any() and not result[:, 1, 0].any()
    assert state['mean'][1, 0] == 0 and state['sd'][1, 0] == 1


def test_transform_preserves_observed_zero_missing_mask_and_inactive_scope():
    values, mask, _ = aligned_example()
    state = mod.fit_molecular_state(values, mask, [0, 1])
    result, effective = mod.transform_molecular(values, mask, state, [0, 1, 2])
    assert effective[0, 1, 0] and not effective[1, 1, 0]
    assert not effective[3].any() and not result[3].any()
    np.testing.assert_array_equal(result[:2, 2, 0], [-1., 1.])
    assert np.isfinite(result).all() and not result[~effective].any()


@pytest.mark.parametrize('bad', ['duplicate', 'blank', 'infinite', 'wrong_axis'])
def test_canonical_inputs_fail_closed(bad):
    rna = np.ones((3, 2)); cna = np.ones((3, 1)); symbols = ['A', 'B']
    if bad == 'duplicate': symbols[1] = 'A'
    if bad == 'blank': symbols[1] = ''
    if bad == 'infinite': rna[0, 0] = np.inf
    if bad == 'wrong_axis': symbols.append('C')
    with pytest.raises(ValueError): mod.align_all_genes(rna, symbols, cna, ['C'])


def test_gene_order_is_UTF8_sorted_and_independent_of_source_column_order():
    rna = np.array([[1., 2.], [3., 4.]])
    a, m, names = mod.align_all_genes(rna, ['Z', 'A'], np.array([[.2], [-.3]]), ['C'])
    b, mm, nn = mod.align_all_genes(rna[:, ::-1], ['A', 'Z'], np.array([[.2], [-.3]]), ['C'])
    np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(m, mm)
    assert names.tolist() == nn.tolist() == ['A', 'C', 'Z']


def test_pinned_50_hallmark_support_has_two_slots_and_eight_allgene(tmp_path):
    path = tmp_path/'synthetic.gmt'
    path.write_text('\n'.join(f'HALLMARK_SYNTHETIC_{k}\tannotation\tA\tB' for k in range(50)))
    digest = mod.sha256(path)
    omega, slots, names = mod.hallmark_membership(path, ['A', 'C', 'B'], digest)
    assert omega.shape == (50, 3) and slots.shape == (108, 3) and len(names) == 50
    assert omega.dtype == slots.dtype == bool
    for k in range(50):
        np.testing.assert_array_equal(slots[2*k], omega[k])
        np.testing.assert_array_equal(slots[2*k+1], omega[k])
    assert slots[100:].all() and not slots[:100, 1].any()
    with pytest.raises(ValueError, match='SHA'): mod.hallmark_membership(path, ['A', 'B'], 'bad')
    with pytest.raises(ValueError, match='no canonical'): mod.hallmark_membership(path, ['MISSING'], digest)


def write_h5(path, *, singleton=True, valid_attrs=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(6*1536, dtype=np.float32).reshape(6, 1536)
    xy = np.array([[0, 0], [512, 0], [1024, 0], [0, 512], [512, 512], [1024, 512]], np.int64)
    with h5py.File(path, 'w') as h:
        h.create_dataset('features', data=values[None] if singleton else values)
        c = h.create_dataset('coords', data=xy[None] if singleton else xy)
        for key, value in [('patch_size_level0', 512), ('target_magnification', 20), ('level0_magnification', 40)]:
            c.attrs[key] = value
        if not valid_attrs: c.attrs['patch_size_level0'] = 256
    return values, xy


@pytest.mark.parametrize('singleton', [True, False])
def test_H5_fancy_row_order_is_restored_without_random_patch_sampling(tmp_path, singleton):
    path = tmp_path/'test.h5'; original, _ = write_h5(path, singleton=singleton)
    selection = np.array([4, 0, 5, 2])
    before = mod.sha256(path)
    with h5py.File(path, 'r') as h:
        selected = mod._read_selected(h, selection)
    np.testing.assert_array_equal(selected, original[selection])
    assert selected.dtype == np.float32 and mod.sha256(path) == before
    with h5py.File(path, 'r') as h:
        with pytest.raises(ValueError): mod._read_selected(h, [1, 1])
        with pytest.raises(ValueError): mod._read_selected(h, [6])


def test_H5_real_coordinate_scale_and_full_patient_coverage_are_required(tmp_path):
    path = tmp_path/'cohort'/'TEST00000000_slide.h5'
    write_h5(path, valid_attrs=False)
    with pytest.raises(ValueError, match='scale'): mod.slide_catalog(tmp_path, ['TEST00000000'])
    write_h5(path)
    with pytest.raises(ValueError, match='no complete'): mod.slide_catalog(tmp_path, ['TEST00000000', 'TEST00000001'])


def frozen_fixture(tmp_path):
    folder = tmp_path/'prepared'; folder.mkdir()
    fit, valid = np.array([1, 0]), np.array([3, 2])
    cp, rna = np.array([.25, .5]), np.array([-.125, .25])
    cp_valid, rna_valid = np.array([.5, .75]), np.array([.125, -.25])
    base = np.full(5, np.nan); base[fit] = cp+rna; base[valid] = cp_valid+rna_valid
    saved = {'source': mod.MAIN_SOURCE, 'stage': 'prototype', 'fold': 1, 'fit': fit, 'valid': valid,
             'active': np.array([0, 1, 2, 3]), 'base': base, 'CP_cf_fit': cp, 'RNA_cf_fit': rna,
             'CP_valid': cp_valid, 'RNA_valid': rna_valid, 'M0_valid': cp_valid+rna_valid,
             'fit_time': np.array([2., 4.]), 'fit_event': np.array([1., 0.]),
             'legacy_prepared_sha256': 'a'*64}
    path = folder/'prototype_fold1.pt'; torch.save(saved, path); digest = mod.sha256(path)
    path.with_suffix('.json').write_text(json.dumps({'status': 'COMPLETE', 'source': mod.MAIN_SOURCE,
                'stage': 'prototype', 'fold': 1, 'sha256': digest}))
    audit = tmp_path/'main_audit.json'
    audit.write_text(json.dumps({'status': 'PASS_NO_FIT_MAIN_REPLAY', 'source': mod.MAIN_SOURCE,
            'prepared_sha256_by_fold': {'1': digest}, 'maximum_absolute_risk_error': 0,
            'endpoint_API_reads': 0, 'new_predictor_fits': 0}))
    return folder, audit, saved


def test_frozen_exact_crossfit_and_heldout_offsets_reused_without_any_fitter(tmp_path, monkeypatch):
    folder, audit, saved = frozen_fixture(tmp_path)
    def no_optimizer(*args, **kwargs): raise AssertionError('no optimizer allowed')
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', no_optimizer)
    result = mod.read_frozen_main(1, folder, audit)
    for key in ('fit', 'valid', 'active', 'base', 'CP_cf_fit', 'RNA_cf_fit', 'CP_valid', 'RNA_valid', 'M0_valid'):
        np.testing.assert_array_equal(result[key], saved[key])
    assert '_fit_time' in result and '_fit_event' in result and 'valid_time' not in result
    np.testing.assert_array_equal(result['_fit_time'], saved['fit_time'])


@pytest.mark.parametrize('mutation,match', [('base', 'risk sum'), ('valid_time', 'held-out'), ('source', 'identity')])
def test_frozen_main_rejects_changed_sum_valid_labels_and_wrong_source(tmp_path, mutation, match):
    folder, audit, saved = frozen_fixture(tmp_path)
    if mutation == 'base': saved['base'][saved['valid'][0]] += .01
    elif mutation == 'valid_time': saved['valid_time'] = np.ones(2)
    else: saved['source'] = 'x'*64
    path = folder/'prototype_fold1.pt'; torch.save(saved, path)
    receipt = json.loads(path.with_suffix('.json').read_text()); receipt['sha256'] = mod.sha256(path)
    path.with_suffix('.json').write_text(json.dumps(receipt))
    evidence = json.loads(audit.read_text()); evidence['prepared_sha256_by_fold']['1'] = mod.sha256(path)
    audit.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match=match): mod.read_frozen_main(1, folder, audit)


def test_streamed_slide_uses_plan_original_rows_and_LRU_is_RAM_only(tmp_path):
    path = tmp_path/'cohort'/'TEST00000000_slide.h5'; values, _ = write_h5(path)
    dataset = object.__new__(mod.Data)
    dataset.slide_catalog = mod.slide_catalog(tmp_path, ['TEST00000000'])
    dataset.patient_ids = np.array(['TEST00000000']); dataset.patient_slides = [[0]]
    dataset._plans, dataset._slide_cache, dataset._cache_slides = {}, mod.OrderedDict(), 1
    selection = np.array([5, 1, 3])
    dataset._plan = lambda index, xy: {'selected_indices': selection, 'region_indices': [np.arange(3)]}
    before = {str(p.relative_to(tmp_path)): mod.sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    one = dataset.load_slide(0)
    np.testing.assert_array_equal(one['features'].numpy(), values[selection])
    assert one['owner'] == 0 and one['features'].device.type == 'cpu'
    again = dataset.load_patient_slides(0)[0]
    assert torch.equal(one['features'], again['features']) and len(dataset._slide_cache) == 1
    one['features'].zero_()
    np.testing.assert_array_equal(dataset.load_slide(0)['features'].numpy(), values[selection])
    after = {str(p.relative_to(tmp_path)): mod.sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after


def test_feature_NPZ_reads_only_whitelisted_keys_not_endpoints(tmp_path, monkeypatch):
    path = tmp_path/'fake.npz'; path.write_bytes(b'synthetic marker')
    monkeypatch.setattr(mod, 'COHORT_SHA256', mod.sha256(path))
    ids = np.array([f'TEST{i:08d}' for i in range(512)])
    monkeypatch.setattr(mod, 'PATIENT_SHA256', hashlib.sha256('\n'.join(ids).encode()).hexdigest())
    data = {'patient_ids': ids, 'folds': np.resize(np.arange(1, 6), 512),
            'rna_genes': np.zeros((512, 20511), np.float32), 'rna_gene_symbols': np.arange(20511).astype(str),
            'cna_genes': np.zeros((512, 25101), np.float32), 'cna_gene_symbols': np.arange(25101).astype(str),
            'cancer_type': np.full(512, 'CRC')}
    accessed = []
    class Archive:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key):
            accessed.append(key)
            assert key in data, 'endpoint array requested'
            return data[key]
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: Archive())
    result = mod.read_feature_npz(path)
    assert set(result) == set(accessed) == set(data)
    assert not any('DSS' in name or 'event' in name or 'time' in name for name in accessed)

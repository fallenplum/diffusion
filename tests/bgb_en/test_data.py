import h5py
import numpy as np
import pytest

from src.bgb_en.data import (build_patch_cache, canonical_order, inner_partitions,
                            coordinate_step, patient_quotas, read_h5_slide,
                            select_patch_rows, select_region_rows)


def test_geometry_selection_invariant_to_input_row_order():
    rng = np.random.default_rng(8)
    xy = np.array([(x, y) for y in range(7) for x in range(8)], np.int64) * 512
    x = rng.normal(size=(len(xy), 8))
    permutation = rng.permutation(len(x))
    for selector in ('GRID', 'DIVERSE', 'RAND', 'REGIONS'):
        a = select_patch_rows(x, xy, 17, selector=selector, seed=31)
        b = select_patch_rows(x[permutation], xy[permutation], 17, selector=selector, seed=31)
        np.testing.assert_array_equal(xy[a], xy[permutation][b])
        np.testing.assert_array_equal(x[a], x[permutation][b])


def test_regions_keep_real_connected_windows_with_bounded_capacity():
    xy = np.array([(x, y) for y in range(20) for x in range(24) if not (8 <= x <= 10)], np.int64)*1024
    selected, region = select_region_rows(xy, 190, max_window=64)
    assert len(np.unique(selected)) == 190 and coordinate_step(xy) == 1024
    for label in np.unique(region):
        points = set(map(tuple, xy[selected[region == label]]))
        assert 0 < len(points) <= 64
        seen = {next(iter(points))}; queue = list(seen)
        for x, y in queue:
            for at in ((x-1024,y),(x+1024,y),(x,y-1024),(x,y+1024)):
                if at in points and at not in seen:
                    seen.add(at); queue.append(at)
        assert seen == points


def test_coordinate_hop_comes_from_native_grid_with_holes():
    xy = np.array([[150, 70], [2198, 70], [3222, 70], [3222, 2118]], np.int64)
    assert coordinate_step(xy) == 1024


def test_geometry_rejects_missing_duplicates_and_empty():
    for coordinates in (np.array([[0., 0.], [np.nan, 1.]]), np.array([[0, 0], [0, 0]]), np.empty((0, 2))):
        with pytest.raises(ValueError):
            canonical_order(coordinates)


def test_patient_budget_is_shared_across_slides():
    slides = [{'patient': 0, 'n': 10}, {'patient': 0, 'n': 30}, {'patient': 1, 'n': 9}]
    quota = patient_quotas(slides, 16, 2)
    assert quota[:2].sum() == 16 and quota[2] == 9 and (quota > 0).all()
    with pytest.raises(ValueError, match='retain all actual slides'):
        patient_quotas(slides, 1, 2)


def test_inner_partitions_seal_outer_patients_and_labels():
    folds = np.repeat(np.arange(1, 6), 15)
    event = (np.arange(len(folds)) % 3 == 0).astype(int)
    a = inner_partitions(folds, event, outer_fold=1)
    changed = event.copy(); changed[folds == 1] = 1 - changed[folds == 1]
    b = inner_partitions(folds, changed, outer_fold=1)
    assert len(a) == 3
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left['fit'], right['fit'])
        np.testing.assert_array_equal(left['valid'], right['valid'])
        assert not set(left['fit']) & set(left['valid'])
        assert not set(left['fit']) & set(left['outer_test'])
    assert sorted(np.concatenate([row['valid'] for row in a])) == list(np.flatnonzero(folds != 1))


def test_h5_singleton_axis_and_masked_patch_cache(tmp_path):
    path = tmp_path / 'one.h5'
    with h5py.File(path, 'w') as handle:
        handle['features'] = np.ones((1, 3, 1536), np.float32)
        handle['coords'] = np.array([[[0, 0], [512, 0], [0, 512]]], np.int64)
    slide = read_h5_slide(path, feature_dim=1536)
    assert slide['features'].shape == (3, 1536)
    np.testing.assert_allclose(np.linalg.norm(slide['features'], axis=1), 1., atol=1e-6)
    features = np.arange(7 * 768, dtype=np.float32).reshape(7, 768) + 1
    candidate = {'features': features, 'coords': np.array([[i, 0] for i in range(7)]),
                 'slides': [{'offset': 0, 'n': 3, 'patient': 0}, {'offset': 3, 'n': 4, 'patient': 1}],
                 'manifest': {'patients': 2}, 'manifest_sha256': 'synthetic'}
    packed = build_patch_cache(candidate, budget=5)
    np.testing.assert_array_equal(packed['owner'], [0, 1])
    assert packed['mask'].sum() == 7
    assert np.all(packed['features'][~packed['mask']] == 0)
    assert np.all(packed['rowids'][~packed['mask']] == -1)
    for slide_index, owner in enumerate(packed['owner']):
        ids = packed['rowids'][slide_index][packed['mask'][slide_index]]
        assert np.all(ids < 3) if owner == 0 else np.all(ids >= 3)

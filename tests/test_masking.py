import numpy as np
import pytest

from phase1_neural import make_mask


def test_random_and_spatial_masking_have_valid_bounds():
    coords = np.stack(np.meshgrid(np.arange(4), np.arange(2)), axis=-1).reshape(-1, 2).astype(float)
    for kind in ("random", "spatial"):
        mask = make_mask(coords, 0.4, kind, np.random.default_rng(3))
        assert mask.shape == (8,)
        assert 0 < int(mask.sum()) < len(coords)
    assert int(make_mask(coords, 0.0, "random", np.random.default_rng(1)).sum()) == 1
    assert int(make_mask(coords, 1.0, "random", np.random.default_rng(1)).sum()) == len(coords) - 1


@pytest.mark.parametrize("coords", [np.zeros((8,)), np.zeros((8, 3)), np.full((8, 2), np.nan)])
def test_masking_rejects_bad_coordinates(coords):
    with pytest.raises(ValueError):
        make_mask(coords, 0.4, "random", np.random.default_rng(1))


@pytest.mark.parametrize("ratio", [-0.1, 1.1, np.nan])
def test_masking_rejects_bad_ratio(ratio):
    with pytest.raises(ValueError):
        make_mask(np.zeros((8, 2)), ratio, "random", np.random.default_rng(1))


def test_masking_rejects_single_token_and_unknown_kind():
    with pytest.raises(ValueError):
        make_mask(np.zeros((1, 2)), 0.4, "random", np.random.default_rng(1))
    with pytest.raises(ValueError):
        make_mask(np.zeros((8, 2)), 0.4, "unknown", np.random.default_rng(1))

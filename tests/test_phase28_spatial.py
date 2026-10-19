import copy
import sys

import numpy as np
import pytest
import torch
from torch import nn

from src.phase28_reread.spatial import (DIRECTIONS, OfficialSpatialEncoder,
    build_plan, pack_regions, pack_segments, scatter_segments, validate_plan)


def pixels(points, phase=(0, 0)):
    return np.asarray(points, np.int64)*512+np.asarray(phase, np.int64)


def test_real_rectangle_is_partitioned_and_direction_reversals_are_exact():
    xy = pixels([(x, y) for y in range(3) for x in range(4)], (73, 19))
    xy = xy[np.array([7, 2, 4, 0, 9, 3, 6, 11, 1, 8, 10, 5])]
    plan = build_plan(xy)
    assert plan['audit']['selected_regions'] == 1
    assert plan['audit']['patch_coverage'] == 1
    assert plan['audit']['segments_per_direction'] == dict(zip(DIRECTIONS, [3, 3, 4, 4]))
    assert np.array_equal(np.sort(plan['selected_indices']), np.arange(12))
    for left, right in [('h_forward', 'h_reverse'), ('v_forward', 'v_reverse')]:
        for a, b in zip(plan['segments'][left], plan['segments'][right]):
            np.testing.assert_array_equal(a[::-1], b)
    assert validate_plan(plan)


def test_connected_region_still_resets_at_missing_locations_in_a_row():
    xy = pixels([(0, 0), (1, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)])
    plan = build_plan(xy)
    assert len(plan['regions']) == 1  # Connected via the second real row.
    assert sorted(map(len, plan['segments']['h_forward'])) == [1, 2, 4]
    for run in plan['segments']['h_forward']:
        if len(run) > 1:
            np.testing.assert_array_equal(np.diff(plan['grid'][run, 0]), np.ones(len(run)-1))


def test_tile_and_component_boundaries_never_become_contiguous_runs():
    tile = build_plan(pixels([(15, 0), (16, 0)]))
    assert len(tile['regions']) == 2
    assert all(len(run) == 1 for run in tile['segments']['h_forward'])
    component = build_plan(pixels([(0, 0), (2, 0)]))
    assert len(component['regions']) == 2
    assert component['audit']['fabricated_patch_locations'] == 0


def test_spatial_bucket_cap_is_fixed_and_invariant_to_H5_row_order():
    xy = pixels([(20*x, 20*y) for y in range(9) for x in range(9)])
    first = build_plan(xy, seed=2801)
    permutation = np.random.default_rng(7).permutation(len(xy))
    second = build_plan(xy[permutation], seed=2801)
    assert first['audit']['selected_regions'] == 64
    assert first['audit']['selected_patches'] == 64
    np.testing.assert_array_equal(xy[first['selected_indices']],
                                  xy[permutation][second['selected_indices']])
    ranks = first['selected_region_ranks']
    for bucket in np.array_split(np.arange(81), 64):
        assert sum(int(rank in bucket) for rank in ranks) == 1
    np.testing.assert_array_equal(ranks, build_plan(xy, seed=2801)['selected_region_ranks'])


def test_entire_16_by_16_region_is_preserved_without_internal_patch_sampling():
    xy = pixels([(x, y) for y in range(16) for x in range(16)])
    plan = build_plan(xy)
    assert plan['audit']['largest_region_patches'] == 256
    assert plan['audit']['largest_segment_patches'] == 16
    assert plan['audit']['selected_patches'] == 256
    assert plan['audit']['maximum_local_slots'] == 2


@pytest.mark.parametrize('coords,message', [
    (np.array([[0, 0], [0, 0]], np.int64), 'duplicate'),
    (np.array([[0, 0], [513, 0]], np.int64), 'phase'),
    (np.array([[0., 0.]]), 'integer'),
    (np.array([[-1, 0]], np.int64), 'nonnegative'),
])
def test_invalid_coordinate_sources_fail_closed(coords, message):
    with pytest.raises(ValueError, match=message):
        build_plan(coords)


def test_valid_mask_does_not_make_a_missing_middle_patch_real():
    plan = build_plan(pixels([(0, 0), (1, 0), (2, 0)]), np.array([True, False, True]))
    assert len(plan['regions']) == 2
    np.testing.assert_array_equal(np.sort(plan['selected_indices']), [0, 2])
    with pytest.raises(ValueError, match='valid'):
        build_plan(pixels([(0, 0)]), np.array([False]))


def test_pack_scatter_roundtrip_ignores_padding_and_preserves_gradients():
    plan = build_plan(pixels([(0, 0), (1, 0), (2, 0), (0, 1)]))
    values = torch.arange(4*3, dtype=torch.float64).reshape(4, 3).requires_grad_()
    packed, mask, indices = pack_segments(values, plan, 'h_reverse')
    assert (~mask).any()
    assert ((~mask).cumsum(1).bool() & mask).sum() == 0  # Suffix padding only.
    poisoned = torch.where(mask[..., None], packed, torch.full_like(packed, 1e30))
    restored = scatter_segments(poisoned, mask, indices, size=4)
    torch.testing.assert_close(restored, values)
    restored.square().sum().backward()
    torch.testing.assert_close(values.grad, 2*values.detach())
    region, region_mask, _ = pack_regions(values, plan)
    assert region.shape == (1, 4, 3) and region_mask.all()


def test_serialized_plan_cannot_join_gaps_or_cross_component_boundaries():
    plan = build_plan(pixels([(0, 0), (1, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]))
    forged = copy.deepcopy(plan)
    forged['segments']['h_forward'][0] = np.r_[forged['segments']['h_forward'][0],
                                              forged['segments']['h_forward'][1]]
    forged['segments']['h_forward'].pop(1)
    forged['segment_regions']['h_forward'] = np.zeros(len(forged['segments']['h_forward']), np.int64)
    with pytest.raises(ValueError, match='gap'):
        validate_plan(forged)


class CumulativeTestScan(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return x.cumsum(1)


def test_encoder_uses_four_independent_runs_and_one_shared_scan_with_gradient():
    plan = build_plan(pixels([(0, 0), (1, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]))
    values = torch.randn(7, 128, dtype=torch.float64, requires_grad=True)
    scan = CumulativeTestScan()
    encoder = OfficialSpatialEncoder(test_only_scan=scan).double()
    result = encoder(values, plan)
    normalized = encoder.norm(values)
    messages = []
    for direction in DIRECTIONS:
        reference = torch.empty_like(values)
        for run in plan['segments'][direction]:
            accumulated = torch.zeros(128, dtype=torch.float64)
            for j in run:
                accumulated = accumulated+normalized[int(j)]
                reference[int(j)] = accumulated
        messages.append(reference)
    torch.testing.assert_close(result, values+torch.stack(messages).mean(0))
    assert scan.calls == 4
    result.square().mean().backward()
    assert torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0


def test_formal_constructor_has_no_missing_dependency_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mamba_ssm', None)
    with pytest.raises(ModuleNotFoundError):
        OfficialSpatialEncoder()


def test_vectorized_pack_matches_original_rowwise_values_and_gradients():
    plan = build_plan(pixels([(0, 0), (1, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]))
    reference = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    actual = reference.detach().clone().requires_grad_()
    rows = plan['segments']['h_reverse']
    width = max(map(len, rows))
    old_indices = torch.full((len(rows), width), -1, dtype=torch.long)
    for j, row in enumerate(rows):
        old_indices[j, :len(row)] = torch.as_tensor(row)
    old_mask = old_indices >= 0
    old_packed = reference.new_zeros((len(rows), width, 5))
    old_packed[old_mask] = reference[old_indices[old_mask]]
    packed, mask, indices = pack_segments(actual, plan, 'h_reverse')
    assert torch.equal(indices, old_indices) and torch.equal(mask, old_mask)
    assert torch.equal(packed, old_packed)
    probe = torch.randn_like(packed)
    (packed*probe).sum().backward(); (old_packed*probe).sum().backward()
    assert torch.equal(actual.grad, reference.grad)

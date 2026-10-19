"""Real-coordinate regions and independent four-axis official Mamba runs.

No missing grid location is inserted. Each tile/component/row gap/column gap is
a recurrent boundary. Plans are built separately for each WSI and have no
outcome, fold, or epoch input. This is a local four-axis adaptation, not SS2D.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import torch
from torch import nn

DIRECTIONS = ('h_forward', 'h_reverse', 'v_forward', 'v_reverse')


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _morton(x, y):
    """Deterministic spatial ordering, independent of input row order."""
    value, bit = 0, 0
    while x or y:
        value |= (x & 1) << (2*bit)
        value |= (y & 1) << (2*bit+1)
        x >>= 1; y >>= 1; bit += 1
    return value


def _region_hash(points, seed):
    content = np.asarray(points, dtype='<i8').tobytes()
    return hashlib.sha256(str(int(seed)).encode()+b'|'+content).hexdigest()


def build_plan(coords, valid_mask=None, *, step=512, tile_size=16,
               max_regions=64, seed=2801):
    """Return original H5 rows and selected-local indices for one actual WSI.

    ``regions`` contains original H5 row indices. ``selected_indices`` is their
    concatenation, which defines the order expected by all packing helpers and
    the encoder. ``region_indices`` and ``segments`` index that selected order.
    Regions exceeding the cap are spatially sorted with Morton order, divided
    into equal-count buckets, and one seed-hashed region is chosen per bucket.
    Every selected region retains all its real patches (at most tile_size**2).
    """
    xy = np.asarray(coords)
    _require(xy.ndim == 2 and xy.shape[1] == 2 and len(xy) > 0,
             'one WSI requires nonempty [N,2] original coordinates')
    _require(np.issubdtype(xy.dtype, np.integer), 'coordinates must be original integer pixels')
    for value, name in [(step, 'step'), (tile_size, 'tile_size'), (max_regions, 'max_regions')]:
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                 and int(value) > 0, name+' must be a positive integer')
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool), 'integer seed required')
    mask = np.ones(len(xy), bool) if valid_mask is None else np.asarray(valid_mask)
    _require(mask.shape == (len(xy),) and mask.dtype == np.bool_, 'valid mask must be bool [N]')
    original = np.flatnonzero(mask)
    _require(len(original) > 0, 'WSI has no real valid patches')
    points = np.asarray(xy[original], np.int64)
    _require((points >= 0).all(), 'real level-zero coordinates must be nonnegative')
    _require(len(np.unique(points, axis=0)) == len(points), 'duplicate real patch coordinates')
    phase = points[0] % int(step)
    _require(((points-phase) % int(step) == 0).all(), 'inconsistent actual 512-grid coordinate phase')
    grid = (points-phase)//int(step)
    by_tile = defaultdict(list)
    for j, (x, y) in enumerate(grid):
        by_tile[(int(x)//tile_size, int(y)//tile_size)].append(j)
    candidates = []
    for tile, members in sorted(by_tile.items(), key=lambda item: (item[0][1], item[0][0])):
        at = {(int(grid[j, 0]), int(grid[j, 1])): j for j in members}
        remaining = set(at)
        while remaining:
            start = min(remaining, key=lambda p: (p[1], p[0]))
            remaining.remove(start); queue = [start]; component = []
            for x, y in queue:
                component.append(at[(x, y)])
                for neighbor in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor); queue.append(neighbor)
            component.sort(key=lambda j: (int(grid[j, 1]), int(grid[j, 0])))
            index = np.asarray(component, np.int64)
            rows = original[index]
            g = grid[index]
            digest = _region_hash(points[index], seed)
            cx, cy = int(g[:, 0].sum())//len(g), int(g[:, 1].sum())//len(g)
            key = (_morton(cx, cy), int(g[:, 1].min()), int(g[:, 0].min()), digest)
            candidates.append({'rows': rows, 'grid': g, 'tile': tile, 'hash': digest, 'key': key})
    candidates.sort(key=lambda row: row['key'])
    total_regions = len(candidates)
    if total_regions > max_regions:
        buckets = np.array_split(np.arange(total_regions), max_regions)
        selected_ranks = sorted(min(bucket, key=lambda j: candidates[int(j)]['hash'])
                                for bucket in buckets)
    else:
        selected_ranks = list(range(total_regions))
    chosen = [candidates[int(rank)] for rank in selected_ranks]
    regions = [np.array(row['rows'], np.int64, copy=True) for row in chosen]
    selected = np.concatenate(regions)
    region_indices, region_grids = [], []
    start = 0
    for row in chosen:
        count = len(row['rows'])
        _require(count <= tile_size**2, 'region exceeds actual tile capacity')
        region_indices.append(np.arange(start, start+count, dtype=np.int64))
        region_grids.append(row['grid']); start += count
    segments = {name: [] for name in DIRECTIONS}
    segment_regions = {name: [] for name in DIRECTIONS}
    for region, (indices, g) in enumerate(zip(region_indices, region_grids)):
        for axis, forward, reverse in [(0, 'h_forward', 'h_reverse'),
                                       (1, 'v_forward', 'v_reverse')]:
            groups = defaultdict(list)
            for local, point in zip(indices, g):
                groups[int(point[1-axis])].append((int(point[axis]), int(local)))
            for _, group in sorted(groups.items()):
                group.sort(); runs = []; run = []; previous = None
                for coordinate, local in group:
                    if previous is not None and coordinate != previous+1:
                        runs.append(run); run = []
                    run.append(local); previous = coordinate
                if run: runs.append(run)
                for run in runs:
                    index = np.asarray(run, np.int64)
                    segments[forward].append(index)
                    segments[reverse].append(index[::-1].copy())
                    segment_regions[forward].append(region)
                    segment_regions[reverse].append(region)
    selected_grids = np.concatenate(region_grids)
    plan = {'version': 1, 'selected_indices': selected, 'regions': regions,
            'region_indices': region_indices, 'grid': selected_grids,
            'segments': segments,
            'segment_regions': {k: np.asarray(v, np.int64) for k, v in segment_regions.items()},
            'selected_region_ranks': np.asarray(selected_ranks, np.int64),
            'audit': {'input_rows': len(xy), 'valid_patches': len(original),
                'all_regions': total_regions, 'selected_regions': len(regions),
                'selected_patches': len(selected), 'patch_coverage': len(selected)/len(original),
                'region_coverage': len(regions)/total_regions,
                'largest_region_patches': max(map(len, regions)),
                'largest_segment_patches': max(len(run) for run in segments['h_forward']+segments['v_forward']),
                'segments_per_direction': {k: len(v) for k, v in segments.items()},
                'step': int(step), 'tile_size': int(tile_size), 'max_regions': int(max_regions),
                'seed': int(seed), 'read_queries_per_region': 2,
                'maximum_local_slots': 2*len(regions),
                'selection': 'Morton-spatial quantile buckets then deterministic minimum coordinate hash',
                'fabricated_patch_locations': 0, 'outcome_or_epoch_inputs': False,
                'boundary_reset': 'WSI / tile / four-neighbor component / actual axis gap'}}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Fail closed if a serialized plan introduces overlap or skips reset gaps."""
    selected = np.asarray(plan['selected_indices'])
    size = len(selected)
    _require(size > 0 and np.issubdtype(selected.dtype, np.integer)
             and len(np.unique(selected)) == size, 'invalid selected original rows')
    region_indices = plan['region_indices']
    region_membership = np.full(size, -1, np.int64)
    for region, rows in enumerate(region_indices):
        rows = np.asarray(rows)
        _require(rows.ndim == 1 and len(rows) > 0 and (rows >= 0).all() and (rows < size).all(),
                 'invalid region local indices')
        _require((region_membership[rows] == -1).all(), 'overlapping regions')
        region_membership[rows] = region
    _require((region_membership >= 0).all(), 'regions do not cover selected patches')
    grid = np.asarray(plan['grid'])
    _require(grid.shape == (size, 2), 'selected real-grid shape mismatch')
    for direction in DIRECTIONS:
        runs = plan['segments'][direction]
        all_rows = np.concatenate(runs)
        _require(np.array_equal(np.sort(all_rows), np.arange(size)),
                 'direction must visit every selected patch exactly once')
        owners = np.asarray(plan['segment_regions'][direction])
        _require(len(owners) == len(runs), 'segment region metadata mismatch')
        axis, sign = (0 if direction.startswith('h_') else 1), (-1 if direction.endswith('reverse') else 1)
        for rows, owner in zip(runs, owners):
            rows = np.asarray(rows)
            _require((region_membership[rows] == owner).all(), 'segment crossed component/tile')
            if len(rows) > 1:
                differences = np.diff(grid[rows], axis=0)
                _require((differences[:, axis] == sign).all()
                         and (differences[:, 1-axis] == 0).all(), 'segment crossed missing patch/gap')
    return True


def _pack(values, rows):
    _require(isinstance(values, torch.Tensor) and values.ndim == 2, 'values must be selected [M,D] tensor')
    _require(len(rows) > 0 and all(len(row) > 0 for row in rows), 'empty run/region')
    # Geometry stays on the host. Construct all rows at once and make exactly
    # one index transfer, avoiding thousands of per-run CUDA allocations/copies.
    # No transient host/device cache is inserted into the serialized plan.
    lengths = np.asarray([len(row) for row in rows], np.int64)
    flat = np.concatenate(rows).astype(np.int64, copy=False)
    _require((flat >= 0).all() and (flat < len(values)).all(), 'packing values/plan scope mismatch')
    host = np.full((len(rows), int(lengths.max())), -1, dtype=np.int64)
    host_mask = np.arange(host.shape[1])[None, :] < lengths[:, None]
    host[host_mask] = flat
    indices = torch.from_numpy(host).to(device=values.device)
    mask = indices >= 0
    gathered = values[indices.clamp_min(0)]
    packed = torch.where(mask[..., None], gathered, values.new_zeros(()))
    return packed, mask, indices


def pack_segments(values, plan, direction):
    _require(direction in DIRECTIONS, 'unknown spatial direction')
    _require(len(values) == len(plan['selected_indices']), 'selected values/plan size mismatch')
    return _pack(values, plan['segments'][direction])


def pack_regions(values, plan):
    _require(len(values) == len(plan['selected_indices']), 'selected values/plan size mismatch')
    return _pack(values, plan['region_indices'])


def scatter_segments(packed, mask, indices, *, size):
    """Undo one complete direction. Padded outputs never enter messages."""
    _require(packed.ndim == 3 and mask.shape == packed.shape[:2] and indices.shape == mask.shape,
             'scatter shape mismatch')
    _require(mask.dtype == torch.bool and indices.dtype == torch.long, 'scatter mask/index types')
    _require(torch.equal(mask, indices >= 0) and (indices[~mask] == -1).all(), 'invalid suffix padding index')
    valid = indices[mask]
    _require(len(valid) == size and torch.equal(torch.sort(valid).values,
             torch.arange(size, device=valid.device)), 'scatter must be an exact nonoverlapping partition')
    result = packed.new_zeros((size, packed.shape[-1]))
    result.index_copy_(0, valid, packed[mask])
    return result


class OfficialSpatialEncoder(nn.Module):
    """Pre-LN residual, four directions sharing one official Mamba instance.

    Default construction performs a normal official package import and has no
    fallback. ``test_only_scan`` is an explicit CPU-only synthetic test hook;
    it is prohibited on CUDA and must never be passed by a formal runner.
    """
    def __init__(self, d_model=128, d_state=16, expand=2, d_conv=4, *, test_only_scan=None):
        super().__init__()
        _require((d_model, d_state, expand, d_conv) == (128, 16, 2, 4), 'locked official core dimensions changed')
        self.test_only = test_only_scan is not None
        self.norm = nn.LayerNorm(d_model)
        if self.test_only:
            _require(isinstance(test_only_scan, nn.Module), 'explicit test-only scan must be a module')
            self.scan = test_only_scan
        else:
            from mamba_ssm import Mamba
            self.scan = Mamba(d_model=d_model, d_state=d_state, expand=expand,
                              d_conv=d_conv, use_fast_path=False)
        self.d_model = d_model

    def forward(self, values, plan):
        _require(values.ndim == 2 and values.shape[1] == self.d_model,
                 'spatial encoder requires selected [M,128] features')
        if self.test_only:
            _require(not values.is_cuda, 'test-only scan is forbidden for CUDA/formal execution')
        else:
            _require(values.is_cuda, 'official selective scan requires CUDA; no CPU fallback')
        normalized = self.norm(values)
        messages = []
        for direction in DIRECTIONS:
            packed, mask, indices = pack_segments(normalized, plan, direction)
            # Each batch element is a true contiguous run with fresh zero state.
            # No inference cache or return_last_state is requested.
            sequence = self.scan(packed)
            _require(isinstance(sequence, torch.Tensor) and sequence.shape == packed.shape,
                     'official scan must return the full sequence tensor')
            messages.append(scatter_segments(sequence, mask, indices, size=len(values)))
        return values+torch.stack(messages).mean(0)

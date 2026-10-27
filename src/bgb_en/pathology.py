"""Small patch encoders for BG-BEN, with explicit real-coordinate boundaries.

P3/P4/P6 use the installed official ``mamba_ssm.Mamba`` implementation.  This
module does not implement or silently substitute an alternative Mamba.  A
deliberate CPU-only scan injection exists for numerical unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("P1", "P2", "P3", "P4", "P5", "P6")
SPATIAL_ARMS = ("P2", "P3", "P4", "P6")
SCAN_ARMS = ("P3", "P4", "P5", "P6")


def _require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_order(coords: np.ndarray) -> np.ndarray:
    return np.lexsort((coords[:, 0], coords[:, 1]))


def permute_coordinate_assignment(coords: np.ndarray, seed: int) -> np.ndarray:
    """Change patch-coordinate pairing, independently of loader row ordering.

    The same canonical-coordinate derangement is used at every epoch.  It is a
    coordinate assignment intervention and never a permutation of feature,
    coordinate, and mask rows together.  Two points necessarily exchange.
    """
    xy = np.asarray(coords)
    _require(xy.ndim == 2 and xy.shape[1] == 2, "coordinates must be [N,2]")
    if len(xy) <= 1:
        return np.array(xy, copy=True)
    order = _canonical_order(xy)
    rng = np.random.default_rng(int(seed))
    # A single shuffled cycle is a fixed-point-free permutation for N > 1.
    cycle = rng.permutation(len(order))
    mapping = np.empty(len(order), dtype=np.int64)
    mapping[cycle] = np.roll(cycle, -1)
    result = np.empty_like(xy)
    result[order] = xy[order[mapping]]
    return result


def spatial_segments(coords: np.ndarray, step: float) -> list[np.ndarray]:
    """Four directed partitions; every real point occurs once per direction.

    Each run is a single contiguous real row or column.  A missing lattice
    position terminates recurrence, including holes inside a tissue component.
    No nonexistent patch is fabricated.  Runs never cross WSI boundaries.
    """
    xy = np.asarray(coords, dtype=np.float64)
    _require(xy.ndim == 2 and xy.shape[1] == 2 and len(xy) > 0,
             "nonempty real coordinates required")
    _require(np.isfinite(xy).all() and (xy >= 0).all(), "coordinates must be finite nonnegative")
    _require(np.isfinite(step) and step > 0, "audited coordinate_step must be positive")
    _require(len(np.unique(xy, axis=0)) == len(xy), "duplicate real coordinates")
    origin = xy.min(axis=0)
    unrounded = (xy - origin) / float(step)
    grid = np.rint(unrounded).astype(np.int64)
    _require(np.allclose(unrounded, grid, atol=1e-5, rtol=0),
             "coordinates do not share the audited patch lattice")
    segments: list[np.ndarray] = []
    for axis in (0, 1):
        for reverse in (False, True):
            for other_value in np.unique(grid[:, 1-axis]):
                index = np.flatnonzero(grid[:, 1-axis] == other_value)
                index = index[np.argsort(grid[index, axis], kind="stable")]
                cuts = np.flatnonzero(np.diff(grid[index, axis]) != 1) + 1
                for run in np.split(index, cuts):
                    segments.append(run[::-1].copy() if reverse else run.copy())
    visits = np.bincount(np.concatenate(segments), minlength=len(xy))
    _require(np.array_equal(visits, np.full(len(xy), 4)), "incomplete spatial scan partition")
    return segments


def spatial_knn(coords: np.ndarray, step: float, k: int = 8,
                radius_steps: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Real kNN edges with a fixed radius; tie-breaking uses coordinates.

    A node with no local neighbor receives a zero neighbor message.  Self
    evidence uses the separate GraphSAGE self projection, never a duplicate
    self-loop in the neighbor mean.  The radius forbids jumping large gaps.
    """
    xy = np.asarray(coords, dtype=np.float64)
    _require(k >= 1 and radius_steps > 0 and step > 0, "invalid spatial graph geometry")
    _require(xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all(), "invalid graph coordinates")
    _require(len(np.unique(xy, axis=0)) == len(xy), "duplicate graph coordinates")
    order = _canonical_order(xy)
    ordered = xy[order]
    distance2 = ((ordered[:, None] - ordered[None, :])**2).sum(-1)
    np.fill_diagonal(distance2, np.inf)
    candidates = np.argsort(distance2, axis=1, kind="stable")[:, :min(k, len(xy))]
    neighbor = np.full((len(xy), k), -1, dtype=np.int64)
    valid = np.zeros((len(xy), k), dtype=bool)
    for row in range(len(xy)):
        chosen = candidates[row]
        chosen = chosen[distance2[row, chosen] <= (float(step)*radius_steps)**2]
        neighbor[order[row], :len(chosen)] = order[chosen]
        valid[order[row], :len(chosen)] = True
    return neighbor, valid


@dataclass(frozen=True)
class Geometry:
    """Fold-independent host geometry, safe to cache with a patch-cache hash."""
    shape: tuple[int, int, int]
    segments: tuple[np.ndarray, ...]
    neighbor: np.ndarray | None
    neighbor_mask: np.ndarray | None
    valid_flat: np.ndarray
    scan_directions: int
    real_nodes: int
    nonempty_slides: int
    segment_count: int
    longest_segment: int
    source_fingerprint: str
    geometry_contract: tuple[float | None, int, float, int]
    real_regions: int
    region_boundaries_supplied: bool


def _geometry_fingerprint(coords: torch.Tensor | np.ndarray | None,
                          mask: torch.Tensor | np.ndarray, arm: str,
                          region_ids: torch.Tensor | np.ndarray | None = None) -> str:
    vm = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    digest = hashlib.sha256(arm.encode()+np.asarray(vm.shape, dtype="<i8").tobytes()+vm.tobytes())
    if arm in SPATIAL_ARMS:
        _require(coords is not None, "spatial encoder requires original coordinates")
        xy = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
        _require(xy.shape == vm.shape+(2,), "coordinate shape must match patches")
        digest.update(np.ascontiguousarray(xy[vm], dtype="<f8").tobytes())
    regions = _region_array(region_ids, vm)
    digest.update(np.ascontiguousarray(regions[vm], dtype="<i8").tobytes())
    digest.update(str(region_ids is not None).encode())
    return digest.hexdigest()


def _region_array(region_ids: torch.Tensor | np.ndarray | None, mask: np.ndarray) -> np.ndarray:
    if region_ids is None:
        return np.zeros(mask.shape, dtype=np.int64)
    ids = region_ids.detach().cpu().numpy() if isinstance(region_ids, torch.Tensor) else np.asarray(region_ids)
    _require(ids.shape == mask.shape and np.issubdtype(ids.dtype, np.integer),
             "region_ids must be integer [B,S,N], matching the patch mask")
    _require((ids[mask] >= 0).all(), "real patches require nonnegative region identities")
    return ids


def build_geometry(coords: torch.Tensor | np.ndarray | None,
                   mask: torch.Tensor | np.ndarray, *, arm: str,
                   coordinate_step: float | None, k: int = 8,
                   radius_steps: float = 1.5, coordinate_shuffle_seed: int = 9031006,
                   region_ids: torch.Tensor | np.ndarray | None = None) -> Geometry:
    """Build geometry without accessing features, labels, or patient identity."""
    _require(arm in ARMS, "unknown pathology arm")
    vm = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    _require(vm.ndim == 3 and vm.dtype == np.bool_, "mask must be bool [B,S,N]")
    b, s, n = vm.shape
    _require(b > 0 and s > 0 and n > 0, "empty tensor dimensions")
    _require(vm.reshape(b, -1).any(-1).all(), "each patient must have at least one real patch")
    regions = _region_array(region_ids, vm)
    xy = None
    if arm in SPATIAL_ARMS:
        _require(coords is not None, "spatial encoder requires original coordinates")
        _require(coordinate_step is not None and np.isfinite(coordinate_step) and coordinate_step > 0,
                 "spatial encoder requires an explicitly audited coordinate_step")
        xy = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
        _require(xy.shape == (b, s, n, 2), "coordinate shape must match patches")
        _require(np.isfinite(xy[vm]).all() and (xy[vm] >= 0).all(), "invalid real coordinates")
    segments: list[np.ndarray] = []
    all_neighbor = np.full((b*s*n, k), -1, np.int64) if arm in ("P2", "P4") else None
    all_neighbor_mask = np.zeros((b*s*n, k), bool) if all_neighbor is not None else None
    nonempty, real_regions = 0, 0
    for batch in range(b):
        for slide in range(s):
            rows = np.flatnonzero(vm[batch, slide])
            if not len(rows):
                continue
            nonempty += 1
            global_rows = (batch*s+slide)*n + rows
            real_coords = None if xy is None else xy[batch, slide, rows]
            slide_region = regions[batch, slide, rows]
            unique_regions = np.unique(slide_region)
            real_regions += len(unique_regions)
            if real_coords is not None:
                _require(len(np.unique(real_coords, axis=0)) == len(rows), "duplicate real patch coordinates")
            if arm == "P6":
                real_coords = real_coords.copy()
                # Preserve every region's actual coordinate multiset and scan
                # lengths while breaking feature-position pairing within it.
                for region in unique_regions:
                    sub = np.flatnonzero(slide_region == region)
                    real_coords[sub] = permute_coordinate_assignment(real_coords[sub], coordinate_shuffle_seed)
            if arm in SPATIAL_ARMS:
                for region in unique_regions:
                    sub = np.flatnonzero(slide_region == region)
                    region_rows, region_coords = global_rows[sub], real_coords[sub]
                    if arm in ("P2", "P4"):
                        neighbor, neighbor_mask = spatial_knn(region_coords, float(coordinate_step), k, radius_steps)
                        all_neighbor[region_rows] = np.where(neighbor_mask, region_rows[neighbor.clip(min=0)], -1)
                        all_neighbor_mask[region_rows] = neighbor_mask
                    if arm in ("P3", "P4", "P6"):
                        segments.extend(region_rows[run] for run in spatial_segments(region_coords, float(coordinate_step)))
            if arm == "P5":
                # Deliberately depends on native selected-H5 row sequence. This
                # is the nonspatial order-sensitive control, not a spatial map.
                segments.extend((global_rows.copy(), global_rows[::-1].copy()))
    return Geometry(shape=(b, s, n), segments=tuple(segments), neighbor=all_neighbor,
                    neighbor_mask=all_neighbor_mask, valid_flat=np.flatnonzero(vm.reshape(-1)),
                    scan_directions=2 if arm == "P5" else (4 if arm in SCAN_ARMS else 0),
                    real_nodes=int(vm.sum()), nonempty_slides=nonempty,
                    segment_count=len(segments), longest_segment=max(map(len, segments), default=0),
                    source_fingerprint=_geometry_fingerprint(coords, mask, arm, region_ids),
                    geometry_contract=(coordinate_step, k, radius_steps, coordinate_shuffle_seed),
                    real_regions=real_regions, region_boundaries_supplied=region_ids is not None)


class GatedMIL(nn.Module):
    def __init__(self, width: int, attention_width: int = 32):
        super().__init__()
        self.tanh = nn.Linear(width, attention_width)
        self.sigmoid = nn.Linear(width, attention_width)
        self.score = nn.Linear(attention_width, 1, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(torch.tanh(self.tanh(values))*torch.sigmoid(self.sigmoid(values))).squeeze(-1)
        present = mask.any(-1, keepdim=True)
        safe = torch.where(present, logits.masked_fill(~mask, -torch.inf), torch.zeros_like(logits))
        attention = torch.softmax(safe, -1)*mask
        return (attention[..., None]*values).sum(-2), attention


class GraphSAGEBlock(nn.Module):
    """Mean-neighbor GraphSAGE with separate self and neighbor projections."""
    def __init__(self, width: int):
        super().__init__()
        self.self_project = nn.Linear(width, width)
        self.neighbor_project = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor, geometry: Geometry) -> torch.Tensor:
        _require(geometry.neighbor is not None, "graph arm requires audited neighbor geometry")
        flat = values.reshape(-1, values.shape[-1])
        indices = torch.as_tensor(geometry.neighbor, device=values.device)
        mask = torch.as_tensor(geometry.neighbor_mask, device=values.device)
        neighbors = flat[indices.clamp_min(0)]*mask[..., None]
        mean = neighbors.sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
        result = self.norm(flat+F.gelu(self.self_project(flat)+self.neighbor_project(mean)))
        return result.reshape_as(values)


class PackedSpatialMamba(nn.Module):
    """One official call for all true runs and directions; zero initial state.

    Every batch row is one independent contiguous run. Tail padding is discarded
    after the causal scan. We do not use ``last_state`` or an inference cache.
    """
    def __init__(self, width: int = 64, state_dim: int = 8, *, test_only_scan: nn.Module | None = None):
        super().__init__()
        self.test_only = test_only_scan is not None
        self.norm = nn.LayerNorm(width)
        if self.test_only:
            _require(isinstance(test_only_scan, nn.Module), "synthetic scan must be an explicit module")
            self.scan = test_only_scan
        else:
            from mamba_ssm import Mamba
            self.scan = Mamba(d_model=width, d_state=state_dim, d_conv=4, expand=2, use_fast_path=False)

    def forward(self, values: torch.Tensor, geometry: Geometry) -> torch.Tensor:
        if self.test_only:
            _require(not values.is_cuda, "CPU-only synthetic scan prohibited in CUDA/formal execution")
        else:
            _require(values.is_cuda, "official Mamba requires CUDA; no CPU fallback")
        _require(len(geometry.segments) > 0 and geometry.scan_directions > 0, "scan requires real segments")
        flat = values.reshape(-1, values.shape[-1])
        lengths = np.fromiter(map(len, geometry.segments), dtype=np.int64)
        host_indices = np.full((len(lengths), int(lengths.max())), -1, np.int64)
        host_mask = np.arange(host_indices.shape[1])[None, :] < lengths[:, None]
        host_indices[host_mask] = np.concatenate(geometry.segments)
        indices = torch.as_tensor(host_indices, device=values.device)
        valid = indices >= 0
        normalized = self.norm(flat)
        packed = normalized[indices.clamp_min(0)]*valid[..., None]
        sequence = self.scan(packed)
        _require(isinstance(sequence, torch.Tensor) and sequence.shape == packed.shape,
                 "official scan must return full output sequence")
        # Avoid repeated-index CUDA atomic summation: sort the host membership,
        # then reduce each patch's 2/4 outputs in an explicit fixed order. This
        # keeps repeated deterministic gradient-cache forwards reproducible.
        visits = np.concatenate(geometry.segments)
        counts = np.bincount(visits, minlength=len(flat))[geometry.valid_flat]
        _require(np.array_equal(counts, np.full(len(counts), geometry.scan_directions)),
                 "scan geometry does not visit each real patch once per direction")
        gather_order = torch.as_tensor(np.argsort(visits, kind="stable"), device=values.device)
        ordered = sequence[valid][gather_order]
        per_patch = ordered.reshape(geometry.real_nodes, geometry.scan_directions, -1).mean(1)
        real_index = torch.as_tensor(geometry.valid_flat, device=values.device)
        message = flat.new_zeros(flat.shape).index_copy(0, real_index, per_patch)
        return (flat+message).reshape_as(values)


class PathologyEncoder(nn.Module):
    """Shared patient-level [B,S,N,D] -> [B,32] interface without clinical data."""
    def __init__(self, arm: str = "P1", input_dim: int = 768, width: int = 64,
                 embedding_dim: int = 32, state_dim: int = 8, graph_layers: int = 1,
                 coordinate_step: float | None = None, k: int = 8,
                 radius_steps: float = 1.5, coordinate_shuffle_seed: int = 9031006,
                 *, test_only_scan: nn.Module | None = None):
        super().__init__()
        _require(arm in ARMS, "unknown pathology arm")
        _require(input_dim > 0 and width > 0 and embedding_dim > 0 and state_dim in (8, 16), "invalid model dimensions")
        _require(graph_layers in (1, 2), "graph_layers must be 1 or 2")
        if arm in SPATIAL_ARMS:
            _require(coordinate_step is not None and np.isfinite(coordinate_step) and coordinate_step > 0,
                     "spatial model requires explicitly audited coordinate_step")
        _require(test_only_scan is None or arm in SCAN_ARMS, "test scan supplied to a nonscan model")
        self.arm, self.input_dim, self.width = arm, input_dim, width
        self.coordinate_step, self.k, self.radius_steps = coordinate_step, k, radius_steps
        self.coordinate_shuffle_seed = int(coordinate_shuffle_seed)
        self.test_only = test_only_scan is not None
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.graph = nn.ModuleList(GraphSAGEBlock(width) for _ in range(graph_layers)) if arm in ("P2", "P4") else nn.ModuleList()
        self.spatial = PackedSpatialMamba(width, state_dim, test_only_scan=test_only_scan) if arm in SCAN_ARMS else None
        self.mil = GatedMIL(width)
        self.output = nn.Linear(width, embedding_dim)

    def geometry(self, coords: torch.Tensor | np.ndarray | None, mask: torch.Tensor | np.ndarray,
                 region_ids: torch.Tensor | np.ndarray | None = None) -> Geometry:
        return build_geometry(coords, mask, arm=self.arm, coordinate_step=self.coordinate_step,
                              k=self.k, radius_steps=self.radius_steps,
                              coordinate_shuffle_seed=self.coordinate_shuffle_seed, region_ids=region_ids)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None,
                mask: torch.Tensor, *, return_details: bool = False,
                geometry: Geometry | None = None,
                region_ids: torch.Tensor | None = None) -> torch.Tensor | dict[str, Any]:
        _require(x.ndim == 4 and x.shape[-1] == self.input_dim, "patches must be [B,S,N,input_dim]")
        _require(mask.shape == x.shape[:3] and mask.dtype == torch.bool, "patch mask must be bool [B,S,N]")
        _require(mask.device == x.device, "patch mask and features must share a device")
        _require(torch.isfinite(x[mask]).all(), "real patch features contain NaN/Inf")
        if geometry is None:
            geometry = self.geometry(coords, mask, region_ids)
        _require(geometry.shape == tuple(mask.shape), "cached geometry tensor shape mismatch")
        current = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        _require(np.array_equal(geometry.valid_flat, current), "cached geometry valid-patch identity mismatch")
        _require(geometry.source_fingerprint == _geometry_fingerprint(coords, mask, self.arm, region_ids),
                 "cached geometry coordinates or arm differ from current input")
        _require(geometry.geometry_contract == (self.coordinate_step, self.k, self.radius_steps,
                                                self.coordinate_shuffle_seed),
                 "cached geometry hyperparameters differ from current model")
        # Padded feature/coordinate NaNs are deliberately ignored; no invalid
        # value is projected, normalized, included in graph values or pooled.
        values = self.project(torch.where(mask[..., None], x, torch.zeros_like(x)))*mask[..., None]
        for block in self.graph:
            values = block(values, geometry)*mask[..., None]
        if self.spatial is not None:
            values = self.spatial(values, geometry)*mask[..., None]
        slide_state, attention = self.mil(values, mask)
        valid_slide = mask.any(-1)
        slide_embedding = self.output(slide_state)*valid_slide[..., None]
        embedding = slide_embedding.sum(1)/valid_slide.sum(1, keepdim=True).clamp_min(1)
        if not return_details:
            return embedding
        return {"embedding": embedding, "patch_states": values, "patch_mask": mask,
                "slide_embedding": slide_embedding, "slide_mask": valid_slide,
                "attention": attention, "geometry": geometry,
                "diagnostics": {"arm": self.arm, "registered_parameters": self.parameter_count(),
                    "real_nodes": geometry.real_nodes, "nonempty_slides": geometry.nonempty_slides,
                    "scan_directions": geometry.scan_directions, "segment_count": geometry.segment_count,
                    "longest_segment": geometry.longest_segment, "test_only_scan": self.test_only,
                    "real_regions": geometry.real_regions,
                    "region_boundaries_supplied": geometry.region_boundaries_supplied,
                    "region_scan_reset": self.arm in ("P3", "P4", "P6") and geometry.region_boundaries_supplied,
                    "graph_same_region_only": self.arm in ("P2", "P4") and geometry.region_boundaries_supplied,
                    "window_pooling": False,
                    "scan_nontrivial_transition_fraction": (1.0-geometry.segment_count/
                        (geometry.real_nodes*geometry.scan_directions)) if geometry.scan_directions else None,
                    "graph_nonisolated_fraction": float(geometry.neighbor_mask[geometry.valid_flat].any(-1).mean())
                        if geometry.neighbor_mask is not None else None,
                    "graph_mean_degree": float(geometry.neighbor_mask[geometry.valid_flat].sum(-1).mean())
                        if geometry.neighbor_mask is not None else None,
                    "slide_aggregation": "equal mean of nonempty WSI embeddings",
                    "dropout": 0.0, "position_encoding": "geometry only; no learned absolute XY"}}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_pathology_encoder(arm: str, **kwargs: Any) -> PathologyEncoder:
    return PathologyEncoder(arm=arm, **kwargs)


class P1GatedABMIL(PathologyEncoder):
    def __init__(self, **kwargs: Any):
        super().__init__("P1", **kwargs)


class P2SpatialGraphSAGE(PathologyEncoder):
    def __init__(self, **kwargs: Any):
        super().__init__("P2", **kwargs)


class P3SpatialMamba(PathologyEncoder):
    def __init__(self, **kwargs: Any):
        super().__init__("P3", **kwargs)


class P4GraphSpatialMamba(PathologyEncoder):
    def __init__(self, **kwargs: Any):
        super().__init__("P4", **kwargs)


class P5SequenceMamba(PathologyEncoder):
    def __init__(self, **kwargs: Any):
        super().__init__("P5", **kwargs)


class P6ShuffledCoordinateMamba(PathologyEncoder):
    def __init__(self, **kwargs: Any):
        super().__init__("P6", **kwargs)

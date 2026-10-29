"""Fixed source geometry and scoped hard negatives; never model features."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .contracts import array_hash, require


def build_neighbors(xy_um, sections, *, radius_um=300., maximum_neighbors=16):
    xy, section = np.asarray(xy_um, np.float64), np.asarray(sections)
    require(xy.ndim == 2 and xy.shape[1] == 2 and section.shape == (len(xy),)
            and np.isfinite(xy).all() and radius_um > 0 and 1 <= maximum_neighbors <= 16,
            "SPGEL_NEIGHBOR_INPUTS")
    out = np.full((len(xy), maximum_neighbors), -1, np.int64)
    for slide in np.unique(section):
        rows = np.flatnonzero(section == slide)
        require(len(np.unique(xy[rows], axis=0)) == len(rows), "DUPLICATE_WITHIN_SLIDE_COORDINATES")
        tree = cKDTree(xy[rows])
        for local, row in enumerate(rows):
            neighbors = np.asarray(tree.query_ball_point(xy[row], radius_um), np.int64)
            dist = np.square(xy[rows[neighbors]] - xy[row]).sum(-1)
            chosen = rows[neighbors[np.lexsort((rows[neighbors], dist))[:maximum_neighbors]]]
            require(len(chosen) and chosen[0] == row, "SELF_NEIGHBOR_REQUIRED")
            out[row, :len(chosen)] = chosen
    return out


def hard_negative_rows(*, section, tissue, library, xy_um, rows, train_depth_quartiles,
                       count=8, exclusion_radius_um=300., seed=31):
    """Same-tissue/depth first; same-slide near neighbors are not strong negatives.

    The sampler can read metadata but returns only row indices to the loss. No
    model input contains tissue truth, depth, absolute coordinate or donor ID.
    Targets/expression are not inputs to this selection function.
    """
    section, tissue = np.asarray(section), np.asarray(tissue)
    library, xy = np.asarray(library, float), np.asarray(xy_um, float)
    rows = np.asarray(rows, np.int64)
    require(section.shape == tissue.shape == library.shape == (len(xy),)
            and xy.shape[1:] == (2,) and len(np.unique(rows)) == len(rows)
            and len(rows) >= 2 and np.isfinite(library[rows]).all() and (library[rows] > 0).all(),
            "SPGEL_HARD_NEGATIVE_METADATA")
    boundaries = np.asarray(train_depth_quartiles, float)
    require(boundaries.shape == (3,) and np.isfinite(boundaries).all() and np.all(np.diff(boundaries) >= 0)
            and 1 <= count <= 32, "SPGEL_HARD_NEGATIVE_TRAIN_DEPTH_CONTRACT")
    bins = np.searchsorted(boundaries, np.log1p(library), side="right")
    rng = np.random.default_rng(seed)
    result = np.full((len(xy), count), -1, np.int64)
    exact, same_tissue, cross_slide, active = 0, 0, 0, 0
    for row in rows:
        distance = np.square(xy[rows] - xy[row]).sum(1)
        eligible = (rows != row) & ((section[rows] != section[row]) | (distance > exclusion_radius_um**2))
        candidates = rows[eligible]
        if not len(candidates):
            continue
        priority = (tissue[candidates] != tissue[row]).astype(int) * 2 + (bins[candidates] != bins[row]).astype(int)
        ordered = candidates[np.lexsort((candidates, rng.random(len(candidates)), priority))[:count]]
        result[row, :len(ordered)] = ordered
        active += len(ordered)
        exact += int(np.sum((tissue[ordered] == tissue[row]) & (bins[ordered] == bins[row])))
        same_tissue += int(np.sum(tissue[ordered] == tissue[row]))
        cross_slide += int(np.sum(section[ordered] != section[row]))
    allowed = result[rows] >= 0
    require(np.isin(result[rows][allowed], rows).all()
            and not (allowed & (result[rows] == rows[:, None])).any(), "HARD_NEGATIVE_SCOPE_OR_SELF_MATCH")
    return result, dict(scope="within_current_source_split_metadata_only", requested_candidates=count,
        valid_negative_count=active, anchors_count=len(rows), same_tissue_depth_count=exact,
        same_tissue_count=same_tissue, cross_slide_count=cross_slide,
        excluded_same_slide_radius_um=exclusion_radius_um,
        mapping_sha256=array_hash(result[rows]), expression_targets_used_for_selection=False,
        depth_boundaries_fitted_on_train=True, inference_features_include_metadata=False)


def localize_negative_indices(global_negative, rows):
    rows, mapping = np.asarray(rows, np.int64), np.asarray(global_negative, np.int64)
    lookup = np.full(len(mapping), -1, np.int64)
    lookup[rows] = np.arange(len(rows))
    selected = mapping[rows]
    valid = selected >= 0
    require((lookup[np.maximum(selected, 0)][valid] >= 0).all(), "NEGATIVE_CROSSES_CURRENT_SPLIT")
    return np.where(valid, lookup[np.maximum(selected, 0)], -1)

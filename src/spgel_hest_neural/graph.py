"""Per-slide pixel kNN; no expression or cross-slide edges enter construction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.spatial import cKDTree
import torch


@dataclass
class Graph:
    # Destination/readout row first, source/value column second.
    edge_index: np.ndarray
    degree: np.ndarray
    audit: dict

    def torch_adjacency(self, device="cpu", dtype=torch.float32):
        rows = self.edge_index[0]
        values = 1.0 / self.degree[rows].astype(np.float64)
        return torch.sparse_coo_tensor(
            torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            torch.as_tensor(values, dtype=dtype, device=device),
            (len(self.degree), len(self.degree)), device=device, dtype=dtype,
        ).coalesce()


def build_graph(coords, specimen_index, identity=False):
    """Return self plus <=15 neighbours, deterministic at complete kth ties.

    The kth radius is an algorithmic boundary for collecting all distance ties,
    not a physical radius or a claim of histological correspondence. At equal
    distances, original within-slide row order determines selection. Own row
    is always included, even if several coordinates coincide.
    """
    coords = np.asarray(coords, dtype=np.float64)
    specimen_index = np.asarray(specimen_index)
    if (coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0
            or specimen_index.shape != (len(coords),) or not np.isfinite(coords).all()):
        raise ValueError("FINITE_COORDINATES_AND_ALIGNED_SPECIMEN_ROWS_REQUIRED")
    if not (np.issubdtype(specimen_index.dtype, np.integer)
            and np.all(specimen_index >= 0)):
        raise ValueError("NONNEGATIVE_INTEGER_SPECIMEN_INDEX_REQUIRED")
    source_rows, destination_rows = [], []
    degree = np.empty(len(coords), dtype=np.int64)
    groups = []
    for specimen in np.unique(specimen_index):
        indices = np.flatnonzero(specimen_index == specimen)
        # Translation before distance arithmetic improves translation stability.
        local = coords[indices] - coords[indices[0]]
        tree = cKDTree(local)
        count = 1 if identity else min(16, len(indices))
        groups.append({"specimen_code": int(specimen), "rows": len(indices), "degree": count})
        kth_distances = None if count == 1 else tree.query(local, k=count)[0][:, -1]
        for local_row, global_row in enumerate(indices):
            if count == 1:
                selected = np.asarray([local_row], dtype=np.int64)
            else:
                radius = np.nextafter(kth_distances[local_row], np.inf)
                candidates = np.asarray(tree.query_ball_point(local[local_row], radius), dtype=np.int64)
                others = candidates[candidates != local_row]
                differences = local[others] - local[local_row]
                distances_squared = np.sum(differences * differences, axis=1)
                order = np.lexsort((others, distances_squared))
                selected = np.concatenate(([local_row], others[order][:count - 1]))
                if len(selected) != count:
                    raise ValueError("INCOMPLETE_KTH_BOUNDARY_CANDIDATES")
            degree[global_row] = len(selected)
            destination_rows.extend([int(global_row)] * len(selected))
            source_rows.extend(indices[selected].tolist())
    edge_index = np.asarray([destination_rows, source_rows], dtype=np.int64)
    if np.any(specimen_index[edge_index[0]] != specimen_index[edge_index[1]]):
        raise ValueError("CROSS_SPECIMEN_EDGE_FORBIDDEN")
    return Graph(edge_index, degree, {
        "schema": "spgel_hest_neural_pixel_knn_v1",
        "method": "IDENTITY" if identity else "SELF_PLUS_15_PIXEL_KNN_COMPLETE_DISTANCE_TIES",
        "maximum_neighbours_including_self": 1 if identity else 16,
        "tie_break": "distance_squared_then_original_within_slide_row_index",
        "physical_radius_used": False, "cross_specimen_edges": 0,
        "edges": edge_index.shape[1], "rows": len(coords), "groups": groups,
        "edge_index_sha256": hashlib.sha256(edge_index.tobytes()).hexdigest(),
        "degree_sha256": hashlib.sha256(degree.tobytes()).hexdigest(),
    })

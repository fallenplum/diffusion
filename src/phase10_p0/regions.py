"""Outcome-blind fold-local prototypes and coordinate-connected regions."""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from .data import Slide


def fit_prototypes(patients: dict, train_patient_ids, seed: int, k: int = 16, per_slide: int = 256):
    rng = np.random.default_rng(seed)
    samples = []
    for patient_id in sorted(map(str, train_patient_ids)):
        for slide in sorted(patients[patient_id], key=lambda value: value.filename):
            indices = rng.choice(len(slide.features), size=min(per_slide, len(slide.features)), replace=False)
            samples.append(normalize(slide.features[indices].astype(np.float32), norm="l2"))
    if not samples:
        raise ValueError("no training patches available for outcome-blind prototypes")
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=8192, n_init=3,
                            max_iter=100, reassignment_ratio=0.01)
    model.fit(np.concatenate(samples, axis=0))
    centers = normalize(model.cluster_centers_.astype(np.float32), norm="l2")
    return centers, {"status": "PASS", "k": k, "sampled_patches": int(sum(map(len, samples))),
                     "per_slide_cap": int(per_slide), "train_patient_count": len(set(train_patient_ids)),
                     "outcome_blind": True, "seed": int(seed)}


def _assign(x: np.ndarray, centers: np.ndarray, chunk: int = 8192):
    result = np.empty(len(x), dtype=np.int16)
    for start in range(0, len(x), chunk):
        block = normalize(x[start:start + chunk].astype(np.float32), norm="l2")
        result[start:start + len(block)] = np.argmax(block @ centers.T, axis=1)
    return result


def _patch_edges(coords: np.ndarray, step: int, radius_multiplier: float):
    n = len(coords)
    index = {tuple(p): i for i, p in enumerate(coords.tolist())}
    edges = set()
    grid_ok = True
    for i, (x, y) in enumerate(coords):
        for q in ((x + step, y), (x, y + step)):
            j = index.get(tuple(q))
            if j is not None:
                edges.add((i, j))
    # Sparse or irregular coordinates use a strict physical radius. Existing
    # grid edges are retained; the radius never exceeds 1.05 patch widths.
    if not edges or len(edges) < n * 0.15:
        grid_ok = False
        pairs = cKDTree(coords.astype(np.float64)).query_pairs(r=step * radius_multiplier, output_type="ndarray")
        for i, j in pairs:
            dx, dy = np.abs(coords[i] - coords[j])
            if max(dx, dy) > step * radius_multiplier:
                continue
            edges.add((int(min(i, j)), int(max(i, j))))
    e = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
    return e, {"coordinate_rule": "four-neighbor grid" if grid_ok else "bounded-radius cKDTree",
               "radius_multiplier": float(radius_multiplier), "radius_level0": float(step * radius_multiplier),
               "edge_count": int(len(e)), "mean_degree": float(2 * len(e) / max(n, 1))}


def _region_components(labels: np.ndarray, edges: np.ndarray):
    keep = labels[edges[:, 0]] == labels[edges[:, 1]] if len(edges) else np.zeros(0, dtype=bool)
    same = edges[keep]
    n = len(labels)
    graph = coo_matrix((np.ones(2 * len(same)), (np.r_[same[:, 0], same[:, 1]], np.r_[same[:, 1], same[:, 0]])),
                       shape=(n, n)).tocsr()
    _, component = connected_components(graph, directed=False, return_labels=True)
    return component.astype(np.int64), same


def _merge_region_graph(component, x, coords, proto, edges, max_regions):
    n0 = int(component.max()) + 1
    counts = np.bincount(component, minlength=n0).astype(np.int64)
    if n0 <= max_regions:
        remap = component.copy()
        region_edges = np.asarray(sorted({tuple(sorted((int(remap[a]), int(remap[b]))))
                                         for a, b in edges if remap[a] != remap[b]}), dtype=np.int64).reshape(-1, 2)
        return remap, region_edges, {"initial_regions": n0, "final_regions": n0, "merged_regions": 0,
                              "other_token_patches": 0, "contact_retention": 1.0}
    contact = defaultdict(int)
    for a, b in edges:
        u, v = int(component[a]), int(component[b])
        if u != v:
            p = (min(u, v), max(u, v))
            contact[p] += 1
    active = set(range(n0))
    adjacency = {i: {} for i in range(n0)}
    for (u, v), weight in contact.items():
        adjacency[u][v] = weight; adjacency[v][u] = weight
    area = {i: int(counts[i]) for i in range(n0)}
    parent = list(range(n0))
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v
    queue = [(area[i], i) for i in range(n0)]
    heapq.heapify(queue)
    merged = 0
    while len(active) > max_regions and queue:
        _, source0 = heapq.heappop(queue); source = find(source0)
        if source not in active:
            continue
        neighbors = [(w, area[find(v)], find(v)) for v, w in adjacency[source].items()
                     if find(v) in active and find(v) != source]
        if not neighbors:
            continue
        # Merge into the adjacent region with the largest shared boundary;
        # break ties toward the larger region, then stable integer ID.
        _, _, target = max(neighbors, key=lambda item: (item[0], item[1], -item[2]))
        parent[source] = target
        active.remove(source); merged += 1
        area[target] += area[source]
        for neighbor0, weight in list(adjacency[source].items()):
            neighbor = find(neighbor0)
            if neighbor not in active or neighbor == target:
                continue
            total = adjacency[target].get(neighbor, 0) + weight
            adjacency[target][neighbor] = total; adjacency[neighbor][target] = total
            adjacency[neighbor].pop(source, None)
        adjacency[target].pop(source, None); adjacency[source].clear()
        heapq.heappush(queue, (area[target], target))
    # Disconnected singleton/small components cannot be joined through real
    # tissue contact; if necessary represent them as one explicit other token.
    other = 0
    other_root = None
    if len(active) > max_regions:
        isolated = sorted(active, key=lambda i: (area[i], i))[:len(active) - max_regions + 1]
        other_root = isolated[0]
        other = area[other_root]
        for node in isolated[1:]:
            parent[node] = other_root; active.remove(node); other += area[node]
        area[other_root] = other
        adjacency[other_root].clear()
        active = {find(i) for i in active}
    roots = sorted({find(i) for i in range(n0)})
    root_map = {root: j for j, root in enumerate(roots)}
    remap = np.asarray([root_map[find(int(c))] for c in component], dtype=np.int64)
    kept_edges = set()
    for a, b in edges:
        u, v = remap[a], remap[b]
        if u != v:
            kept_edges.add((min(int(u), int(v)), max(int(u), int(v))))
    other_index = root_map[other_root] if other_root is not None else None
    if other_index is not None:
        kept_edges = {edge for edge in kept_edges if other_index not in edge}
    retained = sum(1 for a, b in edges if remap[a] == remap[b] or
                   (other_index not in (remap[a], remap[b]) if other_index is not None else True))
    total_contacts = max(1, len(edges))
    return remap, np.asarray(sorted(kept_edges), dtype=np.int64).reshape(-1, 2), {
        "initial_regions": n0, "final_regions": len(roots), "merged_regions": int(merged),
        "other_token_patches": int(other), "other_token_index": other_index,
        "contact_retention": float(retained / total_contacts),
    }


def build_slide_regions(slide: Slide, centers, max_regions=256, radius_multiplier=1.05):
    x = slide.features.astype(np.float32, copy=False)
    labels = _assign(x, centers)
    edges, topology = _patch_edges(slide.coords, slide.patch_size_level0, radius_multiplier)
    component, _ = _region_components(labels, edges)
    remap, region_edges, cap = _merge_region_graph(component, x, slide.coords, labels, edges, max_regions)
    n_regions = int(remap.max()) + 1
    counts = np.bincount(remap, minlength=n_regions).astype(np.float64)
    means = np.zeros((n_regions, 768), dtype=np.float64)
    second = np.zeros(n_regions, dtype=np.float64)
    centers_xy = np.zeros((n_regions, 2), dtype=np.float64)
    hist = np.zeros((n_regions, len(centers)), dtype=np.float64)
    np.add.at(means, remap, x)
    np.add.at(second, remap, np.einsum("ij,ij->i", x, x))
    np.add.at(centers_xy, remap, slide.coords)
    np.add.at(hist, (remap, labels), 1)
    means /= counts[:, None]; centers_xy /= counts[:, None]; hist /= counts[:, None]
    dispersion = np.sqrt(np.maximum(second / counts - np.einsum("ij,ij->i", means, means), 0.0))
    span = np.maximum(slide.coords.max(axis=0) - slide.coords.min(axis=0), slide.patch_size_level0)
    centers_xy = (centers_xy - slide.coords.min(axis=0)) / span
    feat = np.concatenate([means, np.log1p(counts)[:, None], dispersion[:, None], centers_xy, hist], axis=1).astype(np.float32)
    return {"features": feat, "edges": region_edges, "centroids": centers_xy.astype(np.float32), "patch_count": counts.astype(np.int32),
            "patch_to_region": remap.astype(np.int32), "prototype": labels,
            "topology": {**topology, **cap, "patches": len(x), "regions": n_regions,
                         "rare_area_fraction_le_5_patches": float(counts[counts <= 5].sum() / counts.sum()),
                         "rare_region_count_le_5_patches": int((counts <= 5).sum())}}


def build_region_cohort(patients, centers, patient_ids, max_regions=256, radius_multiplier=1.05):
    result = {}
    audit = []
    for patient_id in map(str, patient_ids):
        result[patient_id] = []
        for slide in sorted(patients[patient_id], key=lambda value: value.filename):
            region = build_slide_regions(slide, centers, max_regions, radius_multiplier)
            result[patient_id].append((slide.filename, region))
            audit.append({"slide": slide.filename, **region["topology"]})
    return result, audit


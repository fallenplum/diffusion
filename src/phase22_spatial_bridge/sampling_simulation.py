"""Outcome-blind spatial pseudo-bulk and disjoint raw-count thinning.

One raw count molecule is assigned to condition or evaluation, never both.
Conditional on fixed observed counts these two partitions are complementary,
not independent. Independence follows under an additional Poisson sampling
model, and is not asserted as an empirical property of the input experiment.

Regions are connected prefixes of a physical-micrometre neighbour graph. Each
study/tissue/donor and each disconnected spatial component is handled separately.
The fractions describe *declared capture area* inside that component, not true
patient bulk/WSI overlap or polygonal tissue coverage. Overlapping image patch
areas must not be passed as nonoverlapping molecular capture areas.

No survival outcome, model prediction, or clinical label is accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


LOCKED_AREA_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@dataclass
class RawCountSplit:
    condition: np.ndarray
    evaluation: np.ndarray
    measured_mask: np.ndarray
    condition_probability: float
    random_seed: int
    measured_zero_entries: int
    original_measured_entries: int
    source_sha256: str
    partition_sha256: str
    count_unit: str = "raw_integer_molecule_counts"
    partition_contract: str = "disjoint_complementary_binomial_split"
    independence_contract: str = "independent_only_under_a_Poisson_sampling_model_not_conditional_on_fixed_raw_counts"


@dataclass
class PositiveComposition:
    composition: np.ndarray
    measured_gene_mask: np.ndarray
    aggregated_raw_counts: np.ndarray
    pseudocount_per_measured_gene: float
    raw_measured_zero_genes: int
    raw_total_count_mass: float
    input_unit: str = "sum_of_raw_integer_molecule_counts_before_log"
    output_unit: str = "positive_relative_composition_on_common_measured_genes"
    transform_formula: str = "(aggregated_raw_count_g+pseudocount)/sum_common_measured_genes(aggregated_raw_count+pseudocount)"


@dataclass
class SamplingScenario:
    # Original IDs and arrays remain private runtime data. They are omitted by
    # aggregate_diagnostics; repeated regions are not independent donors.
    study_id_private: Any
    donor_id_private: Any
    tissue_id_private: Any
    component_index: int
    bulk_position_indices: np.ndarray
    image_position_indices: np.ndarray
    condition_bulk: PositiveComposition
    evaluation_image_counts: np.ndarray
    evaluation_measured_mask: np.ndarray
    geometry_measurement_weights: np.ndarray
    rna_mass_oracle_weights: np.ndarray | None
    requested_capture_area_fraction: float
    achieved_capture_area_fraction: float
    rho_simulation: float
    bulk_capture_area_um2: float
    image_capture_area_um2: float
    graph_neighbor_radius_um: float
    graph_neighbor_cap: int
    thinning_source_sha256: str
    thinning_partition_sha256: str
    thinning_random_seed: int
    thinning_condition_probability: float
    area_semantics: str = "sum_of_declared_nonoverlapping_expression_capture_areas_within_connected_component"
    rho_semantics: str = "fixed_synthetic_model_assumption_not_measured_patient_overlap"
    rna_oracle_provenance: str = "original_raw_count_mass_on_bulk_common_measured_genes_ORACLE_ONLY_not_deployment_or_conditioning_input"

    def aggregate_diagnostics(self) -> dict[str, Any]:
        return dict(
            component_index=self.component_index,
            bulk_positions=len(self.bulk_position_indices), image_positions=len(self.image_position_indices),
            requested_capture_area_fraction=self.requested_capture_area_fraction,
            achieved_capture_area_fraction=self.achieved_capture_area_fraction,
            rho_simulation=self.rho_simulation, rho_semantics=self.rho_semantics,
            bulk_capture_area_um2=self.bulk_capture_area_um2, image_capture_area_um2=self.image_capture_area_um2,
            area_semantics=self.area_semantics,
            common_measured_genes=int(self.condition_bulk.measured_gene_mask.sum()),
            condition_bulk_measured_zero_genes=self.condition_bulk.raw_measured_zero_genes,
            evaluation_measured_zero_entries=int(np.sum(self.evaluation_image_counts[self.evaluation_measured_mask] == 0)),
            evaluation_measured_entries=int(self.evaluation_measured_mask.sum()),
            geometry_weights_sum=float(self.geometry_measurement_weights.sum()),
            rna_mass_oracle_available=self.rna_mass_oracle_weights is not None,
            rna_oracle_provenance=self.rna_oracle_provenance,
            thinning_random_seed=self.thinning_random_seed,
            thinning_condition_probability=self.thinning_condition_probability,
            thinning_independence="Poisson_model_only_not_conditional_on_fixed_counts",
            analysis_unit="donor_not_spot_or_sampling_scenario", joint_condition_bulk_vectors=1,
        )


def _raw_counts(raw_counts: np.ndarray, measured_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(raw_counts)
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("raw_counts must have nonempty [position, gene] shape")
    if measured_mask is None:
        measured = np.ones(counts.shape, dtype=bool)
    else:
        measured = np.asarray(measured_mask)
        if measured.dtype != np.dtype(bool):
            raise ValueError("measured_mask must be explicitly boolean")
        if measured.shape == (counts.shape[1],):
            measured = np.broadcast_to(measured, counts.shape).copy()
        elif measured.shape != counts.shape:
            raise ValueError("measured_mask must have shape [gene] or [position, gene]")
        else:
            measured = measured.copy()
    values = np.asarray(counts[measured], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("measured entries must be finite nonnegative integer raw counts, not log/normalized expression")
    if np.any(values > 2**53-1):
        raise ValueError("counts above exact float64 integer range are unsupported")
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    result[measured] = values
    return result, measured


def split_raw_counts(
    raw_counts: np.ndarray, *, measured_mask: np.ndarray | None = None,
    condition_probability: float = 0.5, random_seed: int = 31,
) -> RawCountSplit:
    """Split actual raw counts with no shared molecules and explicit missingness.

    This is not two independent binomial draws from the same observed counts,
    which could reuse molecules. All measured entries, including measured zero,
    stay measured in both partitions; unmeasured entries stay NaN in both.
    """
    counts, measured = _raw_counts(raw_counts, measured_mask)
    if not np.isscalar(condition_probability) or not np.isfinite(condition_probability) or not 0 < condition_probability < 1:
        raise ValueError("condition_probability must lie strictly between zero and one")
    if not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    observed = counts[measured].astype(np.int64)
    condition_values = np.random.default_rng(random_seed).binomial(observed, condition_probability)
    condition, evaluation = np.full(counts.shape, np.nan), np.full(counts.shape, np.nan)
    condition[measured] = condition_values
    evaluation[measured] = observed-condition_values
    digest = hashlib.sha256()
    digest.update(str(counts.shape).encode())
    digest.update(measured.tobytes())
    digest.update(observed.tobytes())
    partition_digest = hashlib.sha256()
    partition_digest.update(digest.digest())
    partition_digest.update(repr((int(random_seed), float(condition_probability))).encode())
    partition_digest.update(condition_values.tobytes())
    return RawCountSplit(condition, evaluation, measured, float(condition_probability), int(random_seed),
                         int(np.sum(observed == 0)), int(measured.sum()), digest.hexdigest(), partition_digest.hexdigest())


def positive_count_composition(
    aggregated_raw_counts: np.ndarray, measured_gene_mask: np.ndarray, *, pseudocount: float,
) -> PositiveComposition:
    """Apply an explicit count-scale pseudocount before composition normalization.

    This is a declared pseudo-bulk transform, not validation that UMI composition
    is interchangeable with TCGA TPM, FPKM, or microarray intensity. Unmeasured
    genes remain NaN; measured zero genes become positive solely because the
    declared pseudocount was applied, and their original count is retained.
    """
    values = np.asarray(aggregated_raw_counts)
    measured = np.asarray(measured_gene_mask)
    if values.ndim != 1 or measured.shape != values.shape or measured.dtype != np.dtype(bool):
        raise ValueError("aggregate and boolean gene mask must share nonempty [gene] shape")
    if not len(values) or not measured.any():
        raise ValueError("at least one common measured gene is required")
    if not np.isscalar(pseudocount) or not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be explicit, finite, and strictly positive")
    counts, _ = _raw_counts(values[None, :], measured)
    raw = counts[0]
    shifted = raw[measured]+float(pseudocount)
    total = shifted.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("pseudocount composition is not finite")
    composition = np.full(values.shape, np.nan)
    composition[measured] = shifted/total
    return PositiveComposition(composition, measured.copy(), raw, float(pseudocount),
                               int(np.sum(raw[measured] == 0)), float(raw[measured].sum()))


def aggregate_count_composition(
    raw_counts: np.ndarray, *, measured_mask: np.ndarray | None = None, pseudocount: float,
) -> PositiveComposition:
    """Sum raw counts before transforming; use intersection of measured genes.

    A gene missing at any aggregated position is not assigned a partial-coverage
    pseudo-bulk sum. This prevents measurement availability becoming expression.
    """
    counts, measured = _raw_counts(raw_counts, measured_mask)
    common = measured.all(axis=0)
    aggregated = np.full(counts.shape[1], np.nan)
    aggregated[common] = counts[:, common].sum(axis=0, dtype=np.float64)
    return positive_count_composition(aggregated, common, pseudocount=pseudocount)


def _group_ids(values: Sequence, size: int, name: str) -> list:
    items = list(values)
    if len(items) != size:
        raise ValueError(f"{name} must have one entry per position")
    for value in items:
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            raise ValueError(f"{name} cannot contain missing identifiers")
        try:
            hash(value)
        except TypeError as error:
            raise ValueError(f"{name} identifiers must be scalar/hashable") from error
    return items


def _physical_components(coords: np.ndarray, radius: float, cap: int) -> tuple[list[list[int]], list[set[int]]]:
    n = len(coords)
    if len(np.unique(coords, axis=0)) != n:
        raise ValueError("duplicate physical coordinates within one tissue require an explicit aggregation decision")
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=min(n, cap+1), distance_upper_bound=radius)
    if n == 1:
        distances, indices = distances[:, None], indices[:, None]
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        for distance, j in zip(distances[i], indices[i]):
            if np.isfinite(distance) and j < n and j != i:
                adjacency[i].add(int(j))
                adjacency[int(j)].add(i)
    components, unseen = [], set(range(n))
    while unseen:
        anchor = min(unseen, key=lambda x: (coords[x, 0], coords[x, 1]))
        component, frontier = [], [anchor]
        unseen.remove(anchor)
        while frontier:
            node = frontier.pop()
            component.append(node)
            for other in sorted(adjacency[node]):
                if other in unseen:
                    unseen.remove(other)
                    frontier.append(other)
        components.append(component)
    return components, adjacency


def _connected_prefix(component: list[int], adjacency: list[set[int]], coords: np.ndarray) -> list[int]:
    center = coords[component].mean(axis=0)
    anchor = min(component, key=lambda i: (float(np.sum((coords[i]-center)**2)), coords[i, 0], coords[i, 1]))
    allowed, visited, queued = set(component), set(), {anchor}
    queue = [(0., float(coords[anchor, 0]), float(coords[anchor, 1]), anchor)]
    ordering = []
    while queue:
        _, _, _, node = heapq.heappop(queue)
        visited.add(node)
        ordering.append(node)
        for other in adjacency[node]:
            if other in allowed and other not in queued:
                queued.add(other)
                distance = float(np.sum((coords[other]-coords[anchor])**2))
                heapq.heappush(queue, (distance, float(coords[other, 0]), float(coords[other, 1]), other))
    if len(visited) != len(component):
        raise RuntimeError("spatial component traversal failed")
    return ordering


def simulate_partial_sampling(
    raw_counts: np.ndarray, coords_um: np.ndarray, tissue_ids: Sequence, donor_ids: Sequence, *,
    study_ids: Sequence, capture_area_um2: float | np.ndarray,
    neighbor_radius_um: float, rho_simulation: float, pseudocount: float,
    measured_mask: np.ndarray | None = None, condition_probability: float = 0.5,
    random_seed: int = 31, neighbor_cap: int = 16, coordinate_unit: str = "um",
) -> list[SamplingScenario]:
    """Create the locked 25/50/75/100% connected-area simulations.

    The conditioning bulk covers a whole connected capture component; the image
    subset grows continuously from a physical anchor. Fractions are of summed
    declared nonoverlapping capture areas, with unavoidable discrete overshoot
    reported exactly. Disconnected components stay separate but do not become
    independent donors. Geometry a_i are deployable assumptions; original raw
    count-mass weights are explicitly separate oracle outputs.

    ``rho_simulation`` is an independently fixed model parameter, not inferred
    from these area fractions and never claimed as observed human overlap.
    """
    counts, measured = _raw_counts(raw_counts, measured_mask)
    n = counts.shape[0]
    coords = np.asarray(coords_um, dtype=np.float64)
    if coordinate_unit != "um" or coords.shape != (n, 2) or not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite [position,2] in explicitly declared micrometres (um)")
    if not np.isscalar(neighbor_radius_um) or not np.isfinite(neighbor_radius_um) or neighbor_radius_um <= 0:
        raise ValueError("neighbor_radius_um must be finite and positive")
    if not isinstance(neighbor_cap, int) or neighbor_cap < 1:
        raise ValueError("neighbor_cap must be a positive integer")
    if not np.isscalar(rho_simulation) or not np.isfinite(rho_simulation) or not 0 <= rho_simulation <= 1:
        raise ValueError("rho_simulation must be a fixed scalar in [0,1]")
    area = np.asarray(capture_area_um2, dtype=np.float64)
    if area.ndim == 0:
        area = np.full(n, float(area))
    if area.shape != (n,) or np.any(~np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("capture_area_um2 must supply a finite positive capture area per position")
    tissue = _group_ids(tissue_ids, n, "tissue_ids")
    donor = _group_ids(donor_ids, n, "donor_ids")
    study = _group_ids(study_ids, n, "study_ids")
    donor_by_tissue, groups = {}, {}
    for i, key in enumerate(zip(study, tissue)):
        if key in donor_by_tissue and donor_by_tissue[key] != donor[i]:
            raise ValueError("one study/tissue identifier cannot refer to multiple donors")
        donor_by_tissue[key] = donor[i]
        groups.setdefault((study[i], donor[i], tissue[i]), []).append(i)
    split = split_raw_counts(counts, measured_mask=measured, condition_probability=condition_probability, random_seed=random_seed)
    scenarios = []
    for (study_id, donor_id, tissue_id), group in groups.items():
        group = np.asarray(group, dtype=np.int64)
        local_coords = coords[group]
        components, adjacency = _physical_components(local_coords, neighbor_radius_um, neighbor_cap)
        for component_index, component in enumerate(components):
            ordered = group[_connected_prefix(component, adjacency, local_coords)]
            bulk = aggregate_count_composition(split.condition[ordered], measured_mask=measured[ordered], pseudocount=pseudocount)
            cumulative_area = np.cumsum(area[ordered])
            total_area = float(cumulative_area[-1])
            for fraction in LOCKED_AREA_FRACTIONS:
                count = min(len(ordered), int(np.searchsorted(cumulative_area, fraction*total_area, side="left"))+1)
                selected = ordered[:count]
                image_area = float(area[selected].sum())
                geometry_weights = area[selected]/image_area
                # Oracle uses true raw counts, including evaluation molecules.
                # It is never supplied as the deployable measurement_weights.
                common = bulk.measured_gene_mask
                masses = counts[selected][:, common].sum(axis=1) if common.any() else np.zeros(count)
                oracle = masses/masses.sum() if masses.sum() > 0 else None
                scenarios.append(SamplingScenario(
                    study_id, donor_id, tissue_id, component_index, ordered.copy(), selected.copy(), bulk,
                    split.evaluation[selected].copy(), measured[selected].copy(), geometry_weights, oracle,
                    fraction, image_area/total_area, float(rho_simulation), total_area, image_area,
                    float(neighbor_radius_um), neighbor_cap, split.source_sha256, split.partition_sha256,
                    split.random_seed, split.condition_probability,
                ))
    return scenarios

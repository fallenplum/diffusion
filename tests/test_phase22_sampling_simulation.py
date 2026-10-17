"""Finite synthetic checks of spatial sampling and raw count contracts."""

import numpy as np
import pytest

from src.phase22_spatial_bridge.sampling_simulation import (
    LOCKED_AREA_FRACTIONS,
    aggregate_count_composition,
    positive_count_composition,
    simulate_partial_sampling,
    split_raw_counts,
)


def _grid():
    coords = np.array([(x*100., y*100.) for x in range(4) for y in range(4)])
    counts = np.column_stack([np.arange(16), np.arange(16)*7, np.ones(16)*3]).astype(int)
    kwargs = dict(study_ids=["study-A"]*16, capture_area_um2=25., neighbor_radius_um=101.,
                  rho_simulation=.5, pseudocount=.5)
    return counts, coords, ["section-A"]*16, ["donor-A"]*16, kwargs


def _connected(coords, radius):
    visited, frontier = {0}, [0]
    while frontier:
        node = frontier.pop()
        for other in np.flatnonzero(np.linalg.norm(coords-coords[node], axis=1) <= radius):
            if int(other) not in visited:
                visited.add(int(other))
                frontier.append(int(other))
    return len(visited) == len(coords)


def test_disjoint_thinning_conserves_counts_and_is_reproducible():
    counts = np.arange(120).reshape(20, 6)
    one = split_raw_counts(counts, random_seed=31)
    two = split_raw_counts(counts, random_seed=31)
    other = split_raw_counts(counts, random_seed=47)
    np.testing.assert_array_equal(one.condition+one.evaluation, counts)
    np.testing.assert_array_equal(one.condition, two.condition)
    assert np.any(one.condition != other.condition)
    assert one.source_sha256 == other.source_sha256
    assert one.partition_sha256 == two.partition_sha256 != other.partition_sha256
    assert "not_conditional_on_fixed_raw_counts" in one.independence_contract


def test_one_molecule_cannot_enter_both_partitions():
    result = split_raw_counts(np.ones((1000, 3), dtype=int), random_seed=31)
    assert np.all(result.condition*result.evaluation == 0)
    assert result.measured_mask.all()


def test_missing_is_not_measured_zero():
    counts = np.array([[0., np.nan, 4.], [0., -123., 9.]])
    mask = np.array([True, False, True])
    result = split_raw_counts(counts, measured_mask=mask)
    assert result.measured_zero_entries == 2 and result.original_measured_entries == 4
    assert np.isnan(result.condition[:, 1]).all() and np.isnan(result.evaluation[:, 1]).all()
    assert (result.condition[:, 0] == 0).all() and result.measured_mask[:, 0].all()
    composition = aggregate_count_composition(counts, measured_mask=mask, pseudocount=.5)
    assert composition.raw_measured_zero_genes == 1
    assert composition.composition[0] > 0 and np.isnan(composition.composition[1])
    assert composition.aggregated_raw_counts[0] == 0


def test_raw_counts_are_summed_before_positive_transform():
    raw = np.array([[1, 0], [0, 9]])
    result = aggregate_count_composition(raw, pseudocount=.5)
    np.testing.assert_array_equal(result.aggregated_raw_counts, [1., 9.])
    np.testing.assert_allclose(result.composition, [1.5/11, 9.5/11], rtol=0, atol=1e-16)
    wrong = np.mean([(raw[0]+.5)/(raw[0]+.5).sum(), (raw[1]+.5)/(raw[1]+.5).sum()], axis=0)
    assert np.max(np.abs(result.composition-wrong)) > .1
    assert result.pseudocount_per_measured_gene == .5 and "before_log" in result.input_unit


def test_heterogeneous_missingness_uses_measured_intersection_not_partial_sum():
    raw = np.array([[1., 2., 0.], [3., np.nan, 0.]])
    mask = np.array([[True, True, True], [True, False, True]])
    result = aggregate_count_composition(raw, measured_mask=mask, pseudocount=1.)
    np.testing.assert_array_equal(result.measured_gene_mask, [True, False, True])
    assert np.isnan(result.composition[1]) and np.isnan(result.aggregated_raw_counts[1])
    assert result.raw_measured_zero_genes == 1


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_pseudocount_is_explicit_and_positive(value):
    with pytest.raises(ValueError, match="pseudocount"):
        positive_count_composition(np.array([0, 2]), np.array([True, True]), pseudocount=value)


@pytest.mark.parametrize("counts", [np.array([[.5, 2.]]), np.array([[np.nan, 1.]]), np.array([[-1., 2.]])])
def test_log_normalized_or_invalid_observed_counts_fail(counts):
    with pytest.raises(ValueError, match="raw counts"):
        split_raw_counts(counts)


def test_locked_fractions_are_nested_connected_and_bulk_is_not_duplicated_per_image():
    counts, coords, tissue, donor, kwargs = _grid()
    scenarios = simulate_partial_sampling(counts, coords, tissue, donor, **kwargs)
    assert len(scenarios) == 4
    assert tuple(x.requested_capture_area_fraction for x in scenarios) == LOCKED_AREA_FRACTIONS
    assert [len(x.image_position_indices) for x in scenarios] == [4, 8, 12, 16]
    for item in scenarios:
        assert _connected(coords[item.image_position_indices], 101.)
        assert len(item.bulk_position_indices) == 16
        assert item.achieved_capture_area_fraction == item.requested_capture_area_fraction
        assert item.rho_simulation == .5
        assert abs(item.geometry_measurement_weights.sum()-1) < 1e-15
        assert item.condition_bulk is scenarios[0].condition_bulk
        assert item.aggregate_diagnostics()["joint_condition_bulk_vectors"] == 1
    for previous, following in zip(scenarios, scenarios[1:]):
        assert set(previous.image_position_indices) < set(following.image_position_indices)


def test_geometry_and_original_count_mass_oracle_are_separate():
    counts, coords, tissue, donor, kwargs = _grid()
    result = simulate_partial_sampling(counts, coords, tissue, donor, **kwargs)[-1]
    np.testing.assert_allclose(result.geometry_measurement_weights, np.full(16, 1/16))
    original_mass = counts[result.image_position_indices].sum(axis=1)
    np.testing.assert_allclose(result.rna_mass_oracle_weights, original_mass/original_mass.sum())
    assert not np.allclose(result.geometry_measurement_weights, result.rna_mass_oracle_weights)
    assert "ORACLE_ONLY" in result.rna_oracle_provenance


def test_zero_count_components_keep_uniform_transform_and_oracle_unavailable():
    counts, coords, tissue, donor, kwargs = _grid()
    results = simulate_partial_sampling(np.zeros_like(counts), coords, tissue, donor, **kwargs)
    for item in results:
        assert item.rna_mass_oracle_weights is None
        np.testing.assert_allclose(item.condition_bulk.composition, np.ones(3)/3)
        assert item.condition_bulk.raw_measured_zero_genes == 3


def test_unequal_capture_areas_have_reported_discrete_overshoot():
    counts, coords, tissue, donor, kwargs = _grid()
    kwargs["capture_area_um2"] = np.arange(1, 17, dtype=float)
    results = simulate_partial_sampling(counts, coords, tissue, donor, **kwargs)
    for item in results:
        assert item.achieved_capture_area_fraction+1e-15 >= item.requested_capture_area_fraction
        assert item.achieved_capture_area_fraction-item.requested_capture_area_fraction <= 16/136+1e-15
        areas = kwargs["capture_area_um2"][item.image_position_indices]
        np.testing.assert_allclose(item.geometry_measurement_weights, areas/areas.sum())


def test_tissues_and_donors_never_mix_even_with_identical_coordinates():
    counts, coords, tissue, donor, kwargs = _grid()
    all_counts, all_coords = np.r_[counts, counts+100], np.r_[coords, coords]
    kwargs["study_ids"] = ["study-A"]*32
    result = simulate_partial_sampling(all_counts, all_coords, tissue+["section-B"]*16,
                                      donor+["donor-B"]*16, **kwargs)
    assert len(result) == 8
    for item in result:
        expected = np.arange(16) if item.donor_id_private == "donor-A" else np.arange(16, 32)
        np.testing.assert_array_equal(np.sort(item.bulk_position_indices), expected)
        assert set(item.image_position_indices).issubset(set(expected))
        aggregate = item.aggregate_diagnostics()
        assert "donor_id_private" not in aggregate and "tissue_id_private" not in aggregate


def test_same_tissue_cannot_be_assigned_multiple_donors():
    counts, coords, tissue, donor, kwargs = _grid()
    donor[0] = "other-donor"
    with pytest.raises(ValueError, match="multiple donors"):
        simulate_partial_sampling(counts, coords, tissue, donor, **kwargs)


def test_disconnected_components_are_separate_not_independent_new_donors():
    counts, coords, tissue, donor, kwargs = _grid()
    coords[8:, 0] += 10000.
    result = simulate_partial_sampling(counts, coords, tissue, donor, **kwargs)
    assert len(result) == 8 and len({x.component_index for x in result}) == 2
    assert {x.donor_id_private for x in result} == {"donor-A"}
    for item in result:
        assert _connected(coords[item.bulk_position_indices], 101.)
        assert item.aggregate_diagnostics()["analysis_unit"] == "donor_not_spot_or_sampling_scenario"


def test_pixels_or_duplicate_positions_fail_before_region_creation():
    counts, coords, tissue, donor, kwargs = _grid()
    with pytest.raises(ValueError, match="micrometres"):
        simulate_partial_sampling(counts, coords, tissue, donor, **kwargs, coordinate_unit="pixels")
    coords[1] = coords[0]
    with pytest.raises(ValueError, match="duplicate"):
        simulate_partial_sampling(counts, coords, tissue, donor, **kwargs)


def test_survival_labels_are_not_accepted():
    counts, coords, tissue, donor, kwargs = _grid()
    with pytest.raises(TypeError):
        simulate_partial_sampling(counts, coords, tissue, donor, **kwargs, DSS=np.zeros(16))

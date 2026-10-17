"""Train discrepancy calibration must not turn data volume into confidence."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from src.phase22_spatial_bridge.measurement_scale import (
    MeasurementScaleTrainingContract, fit_train_measurement_scale,
    verify_measurement_scale,
)
from src.phase22_spatial_bridge.molecular_dictionary import counts_sha256, sequence_sha256


def example():
    counts = np.array([[8, 1, 1], [5, 3, 2], [2, 7, 1], [1, 1, 8], [3, 2, 5]])
    donors = ["private-A"] * 3 + ["private-B"] * 2
    d = np.eye(3, dtype=np.float64)
    pi = counts / counts.sum(1, keepdims=True)
    return counts, pi, d, donors


def contract(counts, d, donors):
    return MeasurementScaleTrainingContract("ST-train", "a" * 64, counts_sha256(counts),
        sequence_sha256(donors), counts_sha256(d),
        tuple((donor, "train") for donor in sorted(set(donors))))


def fit(counts, pi, d, donors, **kwargs):
    return fit_train_measurement_scale(counts, pi, d, donors, oracle_converged=True,
        contract=contract(counts, d, donors), **kwargs)


def verify(result, d):
    verify_measurement_scale(d, result.sigma, result.omega, result.residual_rms,
                             result.within_variance, result.metadata, split_lock_sha256="a" * 64)


def test_exact_reconstruction_uses_train_variation_floor_and_finite_gene_weight_bound():
    counts, pi, d, donors = example()
    result = fit(counts, pi, d, donors)
    y = counts / counts.sum(1, keepdims=True)
    v = (y[:3].var(0, ddof=0) + y[3:].var(0, ddof=0)) / 2
    reference = np.sqrt(v.mean())
    expected = np.maximum(.1 * np.sqrt(v), .1 * reference)
    np.testing.assert_array_equal(result.residual_rms, np.zeros(3))
    np.testing.assert_allclose(result.sigma, expected)
    np.testing.assert_allclose(result.omega, 1. / (np.sqrt(3) * expected))
    assert np.isfinite(result.omega).all()
    assert result.omega.max() <= result.metadata["inverse_weight_upper_bound"]
    assert result.metadata["any_floor_active_genes"] == 3
    assert result.metadata["training_donors"] == 2
    verify(result, d)


def test_donor_equal_rms_and_variance_ignore_replication_of_one_donor():
    counts, pi, d, donors = example()
    # A deliberately imperfect reconstruction exercises the residual RMS term.
    pi = .7 * pi + .3 / 3
    original = fit(counts, pi, d, donors)
    replicated_counts = np.r_[np.tile(counts[:3], (9, 1)), counts[3:]]
    replicated_pi = np.r_[np.tile(pi[:3], (9, 1)), pi[3:]]
    repeated = fit(replicated_counts, replicated_pi, d, ["private-A"] * 27 + ["private-B"] * 2)
    np.testing.assert_allclose(original.residual_rms, repeated.residual_rms, rtol=1e-14)
    np.testing.assert_allclose(original.within_variance, repeated.within_variance, rtol=1e-14)
    np.testing.assert_allclose(original.omega, repeated.omega, rtol=1e-14)


def test_raw_count_exposure_rescaling_does_not_change_composition_calibration():
    counts, pi, d, donors = example()
    original = fit(counts, pi, d, donors)
    changed = fit(counts * np.array([2, 13, 7, 5, 17])[:, None], pi, d, donors)
    np.testing.assert_allclose(original.sigma, changed.sigma, rtol=1e-14)
    np.testing.assert_allclose(original.omega, changed.omega, rtol=1e-14)


def test_sqrt_gene_normalization_keeps_standardized_error_under_vocabulary_duplication():
    counts, pi, d, donors = example()
    pi = .7 * pi + .3 / 3
    a = fit(counts, pi, d, donors)
    doubled_counts = np.tile(counts, (1, 2))
    doubled_d = np.tile(d, (1, 2)) / 2
    b = fit(doubled_counts, pi, doubled_d, donors)
    np.testing.assert_allclose(b.sigma, np.tile(a.sigma, 2) / 2)
    np.testing.assert_allclose(b.omega, np.tile(a.omega, 2) * np.sqrt(2))
    residual = np.array([.02, -.01, -.01])
    assert np.square(a.omega * residual).sum() == pytest.approx(
        np.square(b.omega * np.tile(residual / 2, 2)).sum())


def test_no_within_donor_variation_is_refused_without_invented_numeric_floor():
    counts, pi, d, donors = example()
    counts = np.array([[1, 2, 7]] * 3 + [[3, 2, 5]] * 2)
    pi = counts / counts.sum(1, keepdims=True)
    with pytest.raises(ValueError, match="NO_TRAINING_VARIATION"):
        fit(counts, pi, d, donors)


@pytest.mark.parametrize("role", ["validation", "test"])
def test_non_train_partition_and_non_train_donor_contracts_fail(role):
    counts, pi, d, donors = example()
    lock = contract(counts, d, donors)
    with pytest.raises(ValueError, match="ST-train"):
        fit_train_measurement_scale(counts, pi, d, donors, oracle_converged=True,
                                    contract=replace(lock, partition_role=role))
    with pytest.raises(ValueError, match="non-train"):
        fit_train_measurement_scale(counts, pi, d, donors, oracle_converged=True,
            contract=replace(lock, donor_roles=(("private-A", "train"), ("private-B", role))))


def test_nonconverged_oracle_and_changed_input_hashes_fail():
    counts, pi, d, donors = example()
    with pytest.raises(ValueError, match="nonconverged"):
        fit_train_measurement_scale(counts, pi, d, donors, oracle_converged=False,
                                    contract=contract(counts, d, donors))
    changed = counts.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_train_measurement_scale(changed, pi, d, donors, oracle_converged=True,
                                    contract=contract(counts, d, donors))


def test_loaded_scale_receipt_and_arrays_are_bound_to_train_dictionary():
    counts, pi, d, donors = example()
    result = fit(counts, pi, d, donors)
    verify(result, d)
    corrupted = copy.deepcopy(result.metadata)
    corrupted["source"] = "unit_identity_weights"
    with pytest.raises(ValueError, match="receipt"):
        verify_measurement_scale(d, result.sigma, result.omega, result.residual_rms,
                                 result.within_variance, corrupted, split_lock_sha256="a" * 64)
    with pytest.raises(ValueError, match="array hash"):
        verify_measurement_scale(d, result.sigma, result.omega * 2, result.residual_rms,
                                 result.within_variance, result.metadata, split_lock_sha256="a" * 64)
    with pytest.raises(ValueError, match="dictionary lineage"):
        verify(result, d[[1, 0, 2]])

"""Meaningful statistic/object and privacy checks for the new outer evaluation."""
import numpy as np
import pytest
from scripts.validate_phase19 import (PRIMARY_COMPARISONS, PRIMARY_FAMILY_TESTS,
    check_model_seed_coverage, comparison_definitions, point_tensor, flatten_diagnostics,
    components, resampled_c_indices, linear_c_contrast)


def test_six_comparisons_reserve_twelve_metric_tests():
    risks = {(arm, seed): np.arange(7, dtype=float) for arm in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
             for seed in [31, 47, 83]}
    definitions = comparison_definitions(risks, [31, 47, 83])
    assert [(a, b) for a, b, family in definitions if family == "primary"] == PRIMARY_COMPARISONS
    assert len(PRIMARY_COMPARISONS) * 2 == PRIMARY_FAMILY_TESTS == 12
    risks.pop(("A5", 83))
    with pytest.raises(ValueError, match="seeds"):
        check_model_seed_coverage(risks, [31, 47, 83])


def test_algorithm_mean_c_is_different_from_c_mean_risk():
    t = np.array([1., 2., 3.]); e = np.array([1, 1, 0], bool); w = np.ones(3)
    first, second = np.array([100., 2., 1.]), np.array([-1., 0., 100.])
    c = []
    for risk in [first, second]:
        nn, dd = components(t, e, risk, w); c.append(nn / dd)
    ensemble_num, ensemble_den = components(t, e, (first + second) / 2, w)
    np.testing.assert_allclose(np.mean(c, axis=0), [.5, .5])
    np.testing.assert_allclose(ensemble_num / ensemble_den, [1/3, 1/3])


def test_one_patient_bootstrap_count_matrix_shared_across_models_and_seeds():
    t = np.array([1., 2., 4., 1., 3., 5.]); e = np.array([1, 1, 0, 1, 0, 0], bool)
    fold = np.repeat([1, 2], 3); valid = np.ones(6, bool); weights = np.array([1., 2., 0., 1.5, 0., 0.])
    risk = np.array([6., 3., 1., 4., 2., 1.])
    risks = {("A3", 31): risk, ("A3", 47): risk, ("A0", 31): -risk, ("A0", 47): -risk}
    counts = {1: np.array([[2, 0, 1], [1, 1, 1]]), 2: np.array([[1, 1, 1], [2, 0, 1]])}
    audit = []
    keys, values, denominators, good = resampled_c_indices(t, e, valid, fold, risks, weights,
        draws=2, seed=1, forced_counts=counts, count_audit=audit)
    assert good.all() and len(audit) == 2
    assert all(row["one_shared_count_matrix_for_model_seed_vectors"] == 4 for row in audit)
    delta = linear_c_contrast(keys, values, {"A3": 1., "A0": -1.}, [31, 47])
    np.testing.assert_allclose(delta, np.ones((2, 2)))
    assert (denominators > 0).all()


def test_point_and_resamples_must_have_identical_model_seed_keys():
    rows = [{"arm": "A3", "seed": 31, "Harrell": .8, "Uno60": .75}]
    actual = point_tensor(rows, [("A3", 31)])
    np.testing.assert_allclose(actual, [[[.8, .75]]])
    with pytest.raises(ValueError, match="same models"):
        point_tensor(rows, [("A3", 31), ("A0", 31)])
    with pytest.raises(ValueError, match="repeated"):
        point_tensor(rows + rows, [("A3", 31)])


def test_single_patient_risk_targets_stay_private():
    row = {"arm": "A3", "target_value": 9., "risk_raw": 8., "attention_entropy_mean": .7,
           "position_gradient_probes": [{"target": "final_patient_risk", "target_value": 7., "entropy": .6}]}
    result = flatten_diagnostics([row])[0]
    assert "target_value" not in result and "risk_raw" not in result and "position_probe_target_value" not in result
    assert result["attention_entropy_mean"] == .7
    assert row["position_gradient_probes"][0]["target_value"] == 7.

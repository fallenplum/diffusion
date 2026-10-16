"""Independent explicit patient replication and four-C DID regression tests."""
import numpy as np
from scripts.validate_phase18a import comparable_pairs, components, linear_c_contrast, resampled_c_indices, flatten_mechanisms


def test_shared_patient_counts_equal_explicit_replication():
    time = np.array([1., 2., 2., 5., 1., 3., 4., 6.])
    event = np.array([1, 1, 0, 0, 1, 1, 1, 0], bool)
    fold = np.repeat([1, 2], 4)
    valid = np.ones(8, bool)
    weight = np.array([1., 2., 0., 0., 1.5, 2., 3., 0.])
    a = np.array([4., 1., 1., 2., 4., 3., 2., 1.])
    b = a[::-1].copy()
    risks = {("A", 31): a, ("B", 31): b}
    counts = {1: np.array([[2, 1, 0, 1], [1, 1, 1, 1]]),
              2: np.array([[1, 0, 2, 1], [1, 1, 1, 1]])}
    keys, values, den, good = resampled_c_indices(time, event, valid, fold, risks, weight,
                                                 draws=2, seed=0, forced_counts=counts)
    assert good.all()
    for draw in range(2):
        numerators = np.zeros((2, 2)); denominators = np.zeros(2)
        for f in [1, 2]:
            original = np.flatnonzero(fold == f)
            ix = np.repeat(original, counts[f][draw])
            for k, key in enumerate(keys):
                nn, dd = components(time[ix], event[ix], risks[key][ix], weight[ix])
                numerators[k] += nn
                if k == 0:
                    denominators += dd
        np.testing.assert_allclose(den[draw], denominators)
        np.testing.assert_allclose(values[draw], numerators / denominators)


def test_did_is_four_complete_c_values_per_draw():
    keys = [(arm, seed) for arm in ["F0", "F1", "L0", "L1"] for seed in [31, 47, 83]]
    values = np.random.default_rng(83).uniform(.3, .9, (19, 12, 2))
    coeff = {"L1": 1., "L0": -1., "F1": -1., "F0": 1.}
    actual = linear_c_contrast(keys, values, coeff, [31, 47, 83])
    means = {arm: values[:, [i for i, key in enumerate(keys) if key[0] == arm]].mean(1)
             for arm in coeff}
    expected = (means["L1"] - means["L0"]) - (means["F1"] - means["F0"])
    np.testing.assert_allclose(actual, expected)


def test_event_ties_and_padding_not_comparable_self_pairs():
    i, j = comparable_pairs(np.array([1., 1., 1., 2.]), np.array([1, 1, 0, 0], bool))
    assert set(zip(i, j)) == {(0, 2), (1, 2), (0, 3), (1, 3)}
    assert not np.any(i == j)


def test_seeds_averaged_after_c_not_risk():
    keys = [("A", 31), ("A", 47), ("B", 31), ("B", 47)]
    values = np.array([[[.9, .8], [.5, .6], [.6, .5], [.7, .9]]])
    out = linear_c_contrast(keys, values, {"A": 1., "B": -1.}, [31, 47])
    np.testing.assert_allclose(out, [[.05, 0.]])


def test_public_gradient_diagnostics_omit_single_patient_risk():
    private = {"arm": "F1", "position_gradient_probes": [
        {"target": "final_patient_risk", "target_value": 3.7654,
         "tail10_gradient_L2_mass": .72}]}
    public = flatten_mechanisms([private])
    assert "position_probe_target_value" not in public[0]
    assert public[0]["position_probe_target"] == "final_patient_risk"
    assert public[0]["position_probe_tail10_gradient_L2_mass"] == .72
    assert private["position_gradient_probes"][0]["target_value"] == 3.7654

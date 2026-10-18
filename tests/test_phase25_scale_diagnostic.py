"""Small mathematical invariants only; no datasets, checkpoints or fitting."""
import numpy as np
import pytest

from src.phase25_scale_diagnostic import moments, paired_distributions, residual_summary, scaled_risk


def test_alpha_endpoints_exact_float64_even_when_subtraction_would_overflow():
    base = np.array([-1e308, 1e308])
    trained = -base
    assert np.array_equal(scaled_risk(base, trained, 0), base)
    assert np.array_equal(scaled_risk(base, trained, 1), trained)
    assert np.array_equal(scaled_risk(base, trained, .5), np.zeros(2))
    assert scaled_risk(np.array([1, 2]), np.array([3, 4]), 0).dtype == np.float64


def test_scaled_risk_changes_only_the_saved_residual_and_does_not_mutate_inputs():
    base = np.array([2., 4., 8.]); trained = np.array([6., 0., 16.])
    result = scaled_risk(base, trained, .25)
    assert np.array_equal(result, np.array([3., 3., 10.]))
    result[0] = 999
    assert base[0] == 2 and trained[0] == 6


@pytest.mark.parametrize('a,b,alpha', [([1, 2], [1], .5), ([np.nan], [1], 0),
    ([1], [np.inf], 1), ([1], [2], np.nan), ([1], [2], True), ([1j], [2], .5)])
def test_invalid_shapes_nonfinite_or_complex_values_fail_closed(a, b, alpha):
    with pytest.raises(ValueError):
        scaled_risk(a, b, alpha)


def test_known_population_moments_and_quantiles():
    result = moments([-2, -1, 0, 1, 2])
    assert result['count'] == 5 and result['mean'] == 0 and result['median'] == 0
    assert result['SD'] == pytest.approx(np.sqrt(2))
    assert result['RMS'] == pytest.approx(np.sqrt(2))
    assert result['IQR'] == 2
    assert result['q05'] == pytest.approx(-1.8) and result['q95'] == pytest.approx(1.8)
    assert result['abs_q95'] == 2
    large = moments(np.array([-1e300, 1e300]))
    assert large['mean'] == 0 and large['RMS'] == 1e300 and large['SD'] == 1e300


def test_affine_risks_keep_correlations_but_expose_scale_and_centered_gap():
    full = np.array([-2., -1., 0., 1., 2.]); cf = 2 * full + 3
    result = paired_distributions(cf, full)
    assert result['Pearson'] == pytest.approx(1)
    assert result['Spearman'] == pytest.approx(1)
    assert result['SD_ratio_cf_over_full'] == pytest.approx(2)
    assert result['gap_raw_RMS'] == pytest.approx(np.sqrt(11))
    assert result['gap_centered_RMS'] == pytest.approx(np.sqrt(2))
    assert result['gap_centered_RMS_over_full_SD'] == pytest.approx(1)


def test_spearman_uses_average_tie_ranks_and_differs_from_raw_pearson():
    result = paired_distributions([1, 1, 2, 3], [4, 4, 1, 0])
    assert result['Spearman'] == pytest.approx(-1)
    assert result['Pearson'] > -1


def test_zero_SD_correlations_and_ratios_are_explicitly_undefined():
    result = paired_distributions([2, 2, 2], [7, 7, 7])
    assert result['Pearson'] is None and result['Spearman'] is None
    assert result['SD_ratio_cf_over_full'] is None
    assert result['gap_centered_RMS_over_full_SD'] is None
    assert result['gap_raw_RMS'] == 5 and result['gap_centered_RMS'] == 0
    assert result['zero_SD'] == {'cf': True, 'full': True}
    assert 'UNDEFINED' in result['correlation_status']['Pearson']


def test_residual_ratios_include_constant_offsets_without_inventing_correlation():
    base = np.array([0., 1., 2.])
    result = residual_summary(base, base + 2, 4)
    assert result['delta']['mean'] == 2 and result['delta']['RMS'] == 2
    assert result['SD_delta_over_SD_base'] == 0
    assert result['RMS_delta_over_reference_SD'] == .5
    assert result['Pearson_delta_base'] is None and result['zero_SD']['delta']
    missing = residual_summary([0, 0, 0], [1, 2, 3], 0)
    assert missing['SD_delta_over_SD_base'] is None and missing['RMS_delta_over_reference_SD'] is None
    assert missing['Pearson_delta_base'] is None
    with pytest.raises(ValueError):
        residual_summary(base, base, -1)

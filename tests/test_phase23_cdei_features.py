import numpy as np
import pytest
from src.phase23_cdei.features import (AdditiveTwoDimensionalControl,
    equal_slide_patient_mean, unimodal_feature_matrix)


def test_patient_pooling_equal_slides_keeps_requested_order():
    x = np.array([[1., 2.], [3., 4.], [10., 20.]])
    out = equal_slide_patient_mean(x, [0, 0, 1], [1, 0])
    np.testing.assert_array_equal(out, [[10., 20.], [2., 3.]])
    with pytest.raises(ValueError): equal_slide_patient_mean(x, [0, 0, 1], [2])
    with pytest.raises(ValueError): equal_slide_patient_mean(x, [0, 0, 1], [0, 0])


def test_additive_input_mask_excludes_unmeasured_nan_and_retains_coverage():
    c, p, o = np.ones((5, 4)), np.ones((5, 32)), np.ones((5, 24, 32))
    mask = np.ones((5, 24), bool); mask[:, -1] = False; o[:, -1] = np.nan
    x = unimodal_feature_matrix(c, p, o, mask)
    assert x.shape == (5, 4+32+768+24) and np.isfinite(x).all()
    np.testing.assert_array_equal(x[:, -1], 0.)
    o[0, 0, 0] = np.nan
    with pytest.raises(ValueError): unimodal_feature_matrix(c, p, o, mask)


def test_fit_reference_does_not_inspect_holdout_values_or_reestimate_on_transform():
    rng = np.random.default_rng(23); x = rng.normal(size=(10, 30)); fit = np.arange(7)
    a = AdditiveTwoDimensionalControl().fit(x, fit)
    poisoned = x.copy(); poisoned[7:] = np.nan
    b = AdditiveTwoDimensionalControl().fit(poisoned, fit)
    assert a.public_summary() == b.public_summary()
    np.testing.assert_array_equal(a.transform(x[:7]), b.transform(x[:7]))
    original = a.public_summary(); assert a.transform(x[7:] * 1e8).shape == (3, 2)
    assert a.public_summary() == original


def test_control_is_affine_additive_not_a_cross_modal_product():
    rng = np.random.default_rng(24); x = rng.normal(size=(12, 40))
    control = AdditiveTwoDimensionalControl().fit(x, np.arange(9))
    a, b = rng.normal(size=(2, 3, 40)); zero = np.zeros_like(a)
    np.testing.assert_allclose(control.transform(a+b),
        control.transform(a)+control.transform(b)-control.transform(zero), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('fit', [[0, 0, 1], [-1, 0, 1], [0, 1, 8], [0., 1., 2.], [0, 1]])
def test_invalid_additive_fit_indices(fit):
    with pytest.raises(ValueError): AdditiveTwoDimensionalControl().fit(np.ones((6, 4)), fit)


def test_constant_or_rank_one_control_cannot_claim_two_effective_features():
    for x in (np.ones((6, 4)), np.arange(6.)[:, None] * np.ones((1, 4))):
        with pytest.raises(ValueError): AdditiveTwoDimensionalControl().fit(x, [0, 1, 2, 3])

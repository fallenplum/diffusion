"""Hand-computed cases for the independent SAMI Stage B metrics."""
import json
import numpy as np
import pytest

from sami_surv.program_metrics import evaluate_programs


def base(n=60):
    y = np.repeat(np.arange(n, dtype=float)[:, None], 4, axis=1)
    return y, np.zeros(n, int), np.zeros(n, int), np.repeat(np.arange(3), n//3)


def test_perfect_original_unit_predictions_and_identity_free_output():
    y, d, s, t = base()
    out = evaluate_programs(y, y, np.full(60, 'PRIVATE_DONOR'), np.full(60, 'PRIVATE_SLIDE'),
                            t, np.array([8., 19., -3., 12.]), np.array([1., 2., 3., 4.]))
    assert out['huber'] == out['mse'] == 0
    assert out['within_tissue_spearman'] == out['within_section_spearman'] == 1
    assert all(p['eligible_groups'] == 3 for p in out['per_program'])
    text = json.dumps(out, allow_nan=False)
    assert 'PRIVATE_DONOR' not in text and 'PRIVATE_SLIDE' not in text


def test_train_scale_and_exact_huber_piecewise_objective():
    y, d, s, t = base()
    scale = np.array([1., 2., 3., 4.])
    # Standardized errors .5,1,2,3 -> Huber .125,.5,1.5,2.5.
    pred = y + scale*np.array([.5, 1., 2., 3.])
    out = evaluate_programs(y, pred, d, s, t, np.zeros(4), scale)
    np.testing.assert_allclose([p['huber'] for p in out['per_program']], [.125, .5, 1.5, 2.5])
    assert out['huber'] == pytest.approx(1.15625)
    assert out['mse'] == pytest.approx((.25+1+4+9)/4)


def test_donor_then_section_equal_errors_not_pooled_spots():
    # donor 0 sections contain 20 and 80 spots, donor 1 only 20.
    n = 120
    y = np.repeat(np.arange(n)[:, None], 4, axis=1).astype(float)
    d = np.r_[np.zeros(100, int), np.ones(20, int)]
    s = np.r_[np.zeros(20, int), np.ones(80, int), np.full(20, 2)]
    error = np.r_[np.zeros(20), np.full(80, 2.), np.full(20, 4.)]
    out = evaluate_programs(y, y+error[:, None], d, s, np.zeros(n), np.zeros(4), np.ones(4))
    assert out['huber'] == pytest.approx(((0+1.5)/2+3.5)/2)
    assert out['mse'] == pytest.approx(((0+4)/2+16)/2)


def test_sections_are_averaged_inside_tissues_before_tissues():
    # Tissue 0: section 0 (+1), section 1 (-1) -> 0.
    # Tissue 1: section 0 (+1) -> 1. Tissue-equal -> .5, not pooled 1/3.
    y = np.repeat(np.tile(np.arange(20), 3)[:, None], 4, axis=1).astype(float)
    pred = y.copy(); pred[20:40] *= -1
    s = np.r_[np.zeros(20), np.ones(20), np.zeros(20)]
    t = np.r_[np.zeros(40), np.ones(20)]
    out = evaluate_programs(y, pred, np.zeros(60), s, t, np.zeros(4), np.ones(4))
    assert out['within_tissue_spearman'] == pytest.approx(.5)


def test_large_tissue_does_not_override_small_tissue_correlation():
    y = np.repeat(np.arange(100)[:, None], 4, axis=1).astype(float)
    p = y.copy(); p[20:] = -p[20:]
    t = np.r_[np.zeros(20), np.ones(80)]
    out = evaluate_programs(y, p, np.zeros(100), np.zeros(100), t, np.zeros(4), np.ones(4))
    assert out['within_tissue_spearman'] == pytest.approx(0.)


def test_constant_prediction_is_zero_not_dropped():
    y, d, s, t = base()
    out = evaluate_programs(y, np.ones_like(y), d, s, t, np.zeros(4), np.ones(4))
    assert out['within_tissue_spearman'] == out['within_section_spearman'] == 0
    assert all(p['within_tissue_coverage']['constant_prediction_groups'] == 3 for p in out['per_program'])


def test_constant_truth_has_null_and_explicit_coverage():
    y, d, s, t = base(); y[:, 0] = 8
    out = evaluate_programs(y, y, d, s, t, np.zeros(4), np.ones(4))
    assert out['within_tissue_spearman'] is None
    assert out['per_program'][0]['within_tissue_spearman'] is None
    assert out['per_program'][0]['within_tissue_coverage']['constant_truth_groups'] == 3
    assert out['within_tissue_complete_donors'] == 0
    json.dumps(out, allow_nan=False)


def test_minimum_20_excludes_19_but_accepts_20():
    y = np.repeat(np.arange(39)[:, None], 4, axis=1).astype(float)
    out = evaluate_programs(y, y, np.zeros(39), np.zeros(39), np.r_[np.zeros(19), np.ones(20)],
                            np.zeros(4), np.ones(4))
    assert out['within_tissue_spearman'] == 1
    assert all(p['eligible_groups'] == 1 and p['within_tissue_coverage']['below_minimum_n_groups'] == 1
               for p in out['per_program'])


def test_missing_entire_donor_not_silently_dropped():
    y = np.repeat(np.arange(39)[:, None], 4, axis=1).astype(float)
    d = np.r_[np.zeros(19), np.ones(20)]
    out = evaluate_programs(y, y, d, np.zeros(39), np.zeros(39), np.zeros(4), np.ones(4))
    assert out['within_tissue_spearman'] is None
    assert out['available_donor_description']['within_tissue_spearman'] == 1
    assert out['within_tissue_complete_donors'] == 1


def test_row_order_and_positive_affine_program_scale_invariance():
    y, d, s, t = base()
    p = y.copy(); p[::5] += 3
    a = evaluate_programs(y, p, d, s, t, np.zeros(4), np.ones(4))
    ii = np.random.default_rng(91).permutation(len(y))
    fac, shift = np.array([2., 3., 4., 5.]), np.array([-1., 10., 3., 50.])
    b = evaluate_programs((y*fac+shift)[ii], (p*fac+shift)[ii], d[ii], s[ii], t[ii], shift, fac)
    for key in ['huber', 'mse', 'within_tissue_spearman', 'within_section_spearman']:
        assert a[key] == pytest.approx(b[key])


def test_tied_ranks_use_spearman_average_rank_convention():
    y, d, s, t = base(); y = np.floor(y/2)
    p = -y
    out = evaluate_programs(y, p, d, s, t, np.zeros(4), np.ones(4))
    assert out['within_tissue_spearman'] == pytest.approx(-1)


def test_qc_descriptions_do_not_change_primary():
    y, d, s, t = base()
    a = evaluate_programs(y, y, d, s, t, np.zeros(4), np.ones(4))
    b = evaluate_programs(y, y, d, s, t, np.zeros(4), np.ones(4),
                          library_size=np.arange(60), n_detected_genes=np.arange(60)+1)
    for key in ['huber', 'mse', 'within_tissue_spearman', 'within_section_spearman']:
        assert a[key] == b[key]
    assert b['qc_associations']['library_size']['truth']['aggregate'] == 1


@pytest.mark.parametrize('bad_scale', [[1, 1, 0, 1], [1, 1, -1, 1], [1, 1, np.nan, 1]])
def test_invalid_training_scale_fails(bad_scale):
    y, d, s, t = base()
    with pytest.raises(ValueError, match='positive scale'):
        evaluate_programs(y, y, d, s, t, np.zeros(4), bad_scale)


def test_nonfinite_prediction_and_wrong_dimensions_fail():
    y, d, s, t = base(); p = y.copy(); p[0, 0] = np.inf
    with pytest.raises(ValueError, match='finite'):
        evaluate_programs(y, p, d, s, t, np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match='matching'):
        evaluate_programs(y[:, :3], y[:, :3], d, s, t, np.zeros(4), np.ones(4))

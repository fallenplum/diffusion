import numpy as np
import pytest
from scripts.semantic_m123r_drift_audit import decompose,ranks


def test_positive_affine_scale_drift_is_not_ranking_damage():
    x=np.array([-2.,-1.,1.,2.]);d=decompose(1.2*x+3,x)
    assert d['centered_risk_drift_ratio']==pytest.approx(.2)
    assert d['non_affine_drift_ratio']<1e-14
    assert d['rank_Spearman_to_anchor']==pytest.approx(1.)


def test_ties_get_average_rank_and_reversal_is_detected():
    assert np.array_equal(ranks([3.,1.,1.,2.]),[3.,.5,.5,2.])
    assert decompose(np.array([3.,2.,1.]),np.array([1.,2.,3.]))['rank_Spearman_to_anchor']==pytest.approx(-1.)

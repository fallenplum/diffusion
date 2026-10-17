"""Regression coverage for the historical-only Phase19 qualification rule."""
import numpy as np
import pytest
from scripts.validate_phase20 import historical_reference_identity

def check(a,b,**kwargs):
    parameters=dict(cached=a,stored=b,expected_indices=np.arange(len(a)),stored_indices=np.arange(len(a)),artifact_sha='fixed',receipt_sha='fixed')
    parameters.update(kwargs)
    return historical_reference_identity(**parameters)

def test_original_float64_replay_roundoff_preserves_values_and_all_pair_classes():
    stored=np.array([.2,.4,.8],dtype=np.float64);cached=stored.copy();cached[1]=np.nextafter(cached[1],np.inf)
    result=check(cached,stored)
    assert result['maximum_error']>0 and result['maximum_error']<1e-9
    assert result['all_pair_strict_order_identical'] and result['all_pair_tie_classification_identical']
    assert not result['OOF_values_changed_or_cast']
    assert cached[1]!=stored[1]

def test_original_artifact_hash_and_exact_patient_fold_are_required():
    a=np.array([.2,.4,.8])
    with pytest.raises(ValueError,match='hash'):check(a,a,receipt_sha='changed')
    with pytest.raises(ValueError,match='fold'):check(a,a,stored_indices=np.array([1,0,2]))

def test_original_float64_shape_and_finite_values_are_required():
    a=np.array([.2,.4,.8])
    with pytest.raises(ValueError,match='float64'):check(a.astype(np.float32),a)
    with pytest.raises(ValueError,match='shape'):check(a,a[:2])
    b=a.copy();b[1]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):check(a,b)

def test_published_absolute_limit_does_not_allow_changed_order():
    a=np.array([.2,.4,.8]);b=a.copy();b[1]+=2e-9
    with pytest.raises(ValueError,match='tolerance'):check(a,b)
    a=np.array([0.,1e-10]);b=np.array([1e-10,0.])
    with pytest.raises(ValueError,match='ordering'):check(a,b)

def test_tie_classification_must_also_match_below_absolute_replay_limit():
    a=np.array([0.,.99e-8]);b=np.array([0.,1.01e-8])
    with pytest.raises(ValueError,match='tie'):check(a,b)

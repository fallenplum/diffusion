import copy
import itertools
import pytest
from scripts import semantic_b0b7_p2s_readout as producer


def states():
    return [dict(zip(('split','family','seed','lambda_pair','epoch'),x)) for x in itertools.product(
        ('outer1_inner0','outer1_inner1','outer1_inner2'),('MAIN','ATTN_MATCHED'),(31,47,83),(.1,.03),(400,800))]


def test_exact72_immutable_scopes():
    values=states();producer.verify_state_set(values)
    assert len({producer.job_name(x) for x in values})==72
    assert producer.job_name(values[0])=='outer1_inner0_MAIN_seed31_pair010_epoch400'


@pytest.mark.parametrize('change',[{'split':'outer2_inner0'},{'split':'outer1_final'},{'epoch':1600},
    {'family':'ATTN_NATIVE'},{'seed':2024},{'lambda_pair':.0}])
def test_no_outer_or_extra_candidate(change):
    value=states()[0];value.update(change)
    with pytest.raises(ValueError,match='FIXED_SCOPE'):producer.job_name(value)


def test_duplicate_or_missing_state_fails():
    x=states()
    with pytest.raises(ValueError,match='ALL72'):producer.verify_state_set(x[:-1])
    x[-1]=copy.deepcopy(x[0])
    with pytest.raises(ValueError,match='ALL72'):producer.verify_state_set(x)


def test_child_lock_must_match_fixed_science():
    with pytest.raises(ValueError,match='NOT_PRELOCKED'):producer.verify_child({}, {},producer.STUDY_SHA)
    child={'schema':'semantic_b0b7_p2s_readout_lock_v1','status':'LOCKED_BEFORE_P2S_DSS_FITS',
        'parent_study_lock_sha256':producer.STUDY_SHA,'expected_feature_states':72,
        'expected_Cox_fits':2376,'primary_ridge_lambda':10.,'ridge_candidates':[.1,1.,10.],
        'outer_evaluation':False}
    with pytest.raises(ValueError,match='GRID_CHANGED'):producer.verify_child(child,{},producer.STUDY_SHA)


def test_existing_fit_then_evaluation_order_is_preserved_not_sorted():
    import numpy as np
    ordered=np.r_[np.arange(0,409,2),np.arange(1,409,2)]
    producer.verify_population_order(ordered,ordered.copy())
    with pytest.raises(ValueError,match='ROW_ORDER_CHANGED'):
        producer.verify_population_order(ordered,np.sort(ordered))


def test_registered_rank_na_is_terminal_and_never_zero_score():
    assert producer.terminal_worker_status('COMPLETE_WITH_NONPRIMARY_RANK_NA')
    assert producer.terminal_worker_status('PARTIAL_PRIMARY_UNSUPPORTED_RANK')
    assert not producer.terminal_worker_status('FAILED')


def test_frozen_clinical_field_order_is_not_alphabetical():
    from src.semantic_b0b7.data import CLINICAL_FIELDS
    source={k:None for k in sorted(CLINICAL_FIELDS)}
    assert producer.clinical_order(source)==list(CLINICAL_FIELDS)
    assert producer.clinical_order(source)!=sorted(CLINICAL_FIELDS)
    source['OUTCOME_FIELD']=None
    with pytest.raises(ValueError,match='CLINICAL_FIELDS'):producer.clinical_order(source)

from types import SimpleNamespace
import numpy as np
import pytest
from scripts.semantic_m123_schedule import choose,inner_jobs,final_jobs,identifier
from scripts.semantic_m123_worker import verify_job
from src.semantic_m123.data import prepare,standardize


def metric(n,d,u=None,ud=None):
    return {'Harrell':n/d,'Harrell_num':n,'Harrell_den':d,
        'Uno60':None if u is None else u/ud,'Uno_num':u,'Uno_den':ud}


def test_exact_registered_grid_and_unique_identifiers():
    jobs=inner_jobs('a'*64)
    assert len(jobs)==945 and len({identifier(j) for j in jobs})==945
    assert sum(j['arm']=='M0PLUS' for j in jobs)==45
    assert sum(j['arm']!='M0PLUS' for j in jobs)==900
    assert set(j['seed'] for j in jobs if j['arm']!='M0PLUS')=={31,47,83}
    assert all('final' not in j['split'] for j in jobs)


def test_micro_components_not_equal_fold_score_average():
    # A wins the unweighted fold mean but loses the prelocked pair-micro.
    a=[metric(9,10,9,10),metric(9,10,9,10),metric(10,100,10,100)]
    b=[metric(5,10,5,10),metric(5,10,5,10),metric(60,100,60,100)]
    best,_=choose([(.001,400,a),(.01,200,b)])
    assert best['lambda']==.01 and best['Harrell']==70/120


def test_harrell_then_uno_then_stronger_l2_then_shorter_epoch():
    low=[metric(3,4,2,4)]*3;high=[metric(3,4,3,4)]*3
    best,_=choose([(.001,50,low),(.001,400,high),(.01,200,high),(.01,100,high)])
    assert best['lambda']==.01 and best['epoch']==100


def test_unsupported_uno_is_not_assigned_half_or_zero():
    x,_=choose([(.001,0,[metric(3,4)]*3)])
    assert x['Uno60'] is None


def test_selection_requires_three_distinct_registered_partitions():
    with pytest.raises(ValueError,match='THREE'):
        choose([(.001,20,[metric(3,4)]*2)])


def test_inner_job_rejects_unregistered_budget_or_seed(tmp_path):
    lock={'partitions':{'outer1_inner0':{}}}
    base={'split':'outer1_inner0','role':'inner','arm':'M1_WARM','seed':31,'lambda':.001,'epochs':400,'lock_sha256':'a'*64}
    verify_job(base,lock,tmp_path)
    for replacement in ({'seed':99},{'epochs':800},{'lambda':.03},{'role':'unknown'}):
        with pytest.raises(ValueError):verify_job({**base,**replacement},lock,tmp_path)


def test_new_warm_preprocessor_refuses_teacher_with_different_fit_rows():
    anchor=SimpleNamespace(train_rows=np.array([1,2]),evaluation_rows=np.array([3,4]))
    with pytest.raises(ValueError,match='CURRENT_PARTITION'):
        prepare({},None,anchor,[0,2],[3,4],None)


def test_new_preprocessor_refuses_any_endpoint_bearing_raw_mapping():
    with pytest.raises(ValueError,match='ENDPOINT'):
        prepare({'t':np.array([1,2])},None,None,[0,1],[2,3],None)


def test_new_feature_moments_use_training_only():
    x=np.array([[1.,2.],[3.,4.],[10.,20.]])
    z,a=standardize(x,[0,1]);y=x.copy();y[-1]*=10000
    zz,b=standardize(y,[0,1])
    assert np.array_equal(z[:2],zz[:2])
    assert np.array_equal(a['mean'],b['mean']) and np.array_equal(a['sd'],b['sd'])


def test_final_job_configs_are_exact_barrier_configs():
    b={'choices':{'outer1/M1_WARM/31':{'lambda':.01,'epoch':50},
                  'outer1/M0PLUS/0':{'lambda':10.,'epoch':0}}}
    jobs=final_jobs('a'*64,b,'b'*64)
    assert len(jobs)==2
    assert jobs[0]['epochs']==50 and jobs[0]['selection_sha256']=='b'*64
    assert all(j['role']=='final' for j in jobs)


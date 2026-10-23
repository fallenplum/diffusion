"""Synthetic SAMI Stage B qualification; no real data or model experiment."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sami_surv import spatial_programs as m
from sami_surv.morphology import hierarchical_weights


def small(seed=11):
    rng=np.random.default_rng(seed)
    donor=np.repeat(['a','b','c','d'],[27,30,33,36])
    section=np.array([d+str(i%2) for i,d in enumerate(donor)])
    tissue=np.arange(len(donor))%3
    design=rng.normal(size=(len(donor),9))
    target=design@rng.normal(size=(9,4))*.2+rng.normal(size=(len(donor),4))*.1
    return design,target,donor,section,tissue


@pytest.fixture(scope='module')
def pca_data():
    rng=np.random.default_rng(8)
    n=96
    X=(rng.normal(size=(n,40))@rng.normal(size=(40,1536))+rng.normal(size=(n,1536))*.01).astype(np.float32)
    donor=np.repeat(['a','b','c'],32)
    section=np.array([d+str(i%2) for i,d in enumerate(donor)])
    pca=m.PCATransform.fit(X,donor,section)
    return X,donor,section,pca


def test_weighted_pca_has_locked_dimensions_and_train_only_scaler(pca_data):
    X,donor,section,pca=pca_data
    assert pca.components.shape==(32,1536)
    assert pca.transform(X).shape==(96,32)
    w=hierarchical_weights(donor,section)
    np.testing.assert_allclose(pca.standardizer.mean,w@X.astype(float),atol=1e-14)
    np.testing.assert_allclose(pca.components@pca.components.T,np.eye(32),atol=1e-10)
    scores=pca.transform(X)
    np.testing.assert_allclose(w@scores,0.,atol=1e-12)
    variance=np.sum(w[:,None]*(scores-w@scores)**2,axis=0)
    np.testing.assert_allclose(variance,np.ones(32),atol=1e-12,rtol=1e-12)
    pivot=np.argmax(np.abs(pca.components),axis=1)
    assert np.all(pca.components[np.arange(32),pivot]>0)
    before={k:v.copy() for k,v in pca.arrays().items()}
    pca.transform(np.full((4,1536),1e9))
    for k,v in before.items():np.testing.assert_array_equal(v,pca.arrays()[k])


def test_same_transform_for_local_and_neighbor_and_replay(pca_data,tmp_path):
    X,donor,section,pca=pca_data
    local=pca.transform(X)
    np.testing.assert_array_equal(local,pca.transform(X.copy()))
    path=tmp_path/'pca.npz';np.savez(path,**pca.arrays())
    with np.load(path,allow_pickle=False) as arrays:
        restored=m.PCATransform.from_arrays(arrays)
        np.testing.assert_array_equal(local,restored.transform(X))


def test_missing_pca_features_fail_closed():
    X=np.ones((40,1536));X[0,0]=np.nan
    with pytest.raises(ValueError,match='no missing'):
        m.PCATransform.fit(X,['d']*40,['s']*40)


def test_zero_variance_pca_cannot_silently_whiten():
    with pytest.raises(ValueError,match='whitening'):
        m.PCATransform.fit(np.ones((40,1536)),['d']*40,['s']*40)


def design_input():
    rng=np.random.default_rng(7)
    local=rng.normal(size=(13,32));neighbor=rng.normal(size=(13,32))
    p=rng.dirichlet(np.ones(3),13);q=rng.dirichlet(np.ones(3),13)
    return local,neighbor,p,q


def test_designs_preserve_same_baseline_and_exact_interaction():
    local,neighbor,p,q=design_input()
    designs={arm:m.build_design(local,neighbor,p,q,arm) for arm in m.ARMS}
    for arm,dim in m.DESIGN_DIMENSIONS.items():assert designs[arm].shape==(13,dim)
    np.testing.assert_array_equal(designs['COMPOSITION'],np.concatenate([p,q],1))
    for arm in ['TISSUE_COND','RANDOM_GATE','WRONG_DONOR','SPATIAL_SHUFFLE']:
        np.testing.assert_array_equal(designs[arm][:,:70],designs['ADDITIVE'])
    expected=np.concatenate([local*q[:,c:c+1] for c in range(3)],1)
    np.testing.assert_array_equal(designs['TISSUE_COND'][:,70:],expected)
    np.testing.assert_array_equal(designs['TISSUE_COND'],designs['WRONG_DONOR'])
    np.testing.assert_array_equal(designs['TISSUE_COND'],designs['SPATIAL_SHUFFLE'])


def test_random_gate_fixed_seed_uses_only_neighbor_features():
    local,neighbor,p,q=design_input()
    first=m.random_gates(neighbor)
    np.testing.assert_array_equal(first,m.random_gates(neighbor.copy()))
    np.testing.assert_allclose(first.sum(1),1.)
    a=m.build_design(local,neighbor,p,q,'RANDOM_GATE')
    b=m.build_design(local,neighbor,p[:,::-1],q[:,::-1],'RANDOM_GATE')
    np.testing.assert_array_equal(a[:,70:],b[:,70:])
    assert not np.allclose(first,m.random_gates(neighbor+.5))


def test_invalid_probabilities_and_missing_design_fail():
    local,neighbor,p,q=design_input()
    with pytest.raises(ValueError,match='probabilities'):
        m.build_design(local,neighbor,p*2,q,'COMPOSITION')
    local[0,0]=np.nan
    with pytest.raises(ValueError,match='no missing'):
        m.build_design(local,neighbor,p,q,'ADDITIVE')


def test_huber_gradient_matches_finite_difference_and_task_mean():
    rng=np.random.default_rng(12)
    x=rng.normal(size=(11,5));target=rng.normal(size=(11,4))*2
    weight=rng.uniform(size=11);weight/=weight.sum()
    theta=rng.normal(size=24)*.2
    value,g=m.huber_objective_gradient(theta,x,target,weight)
    epsilon=1e-6;numeric=np.zeros_like(g)
    for j in range(len(theta)):
        hi,lo=theta.copy(),theta.copy();hi[j]+=epsilon;lo[j]-=epsilon
        numeric[j]=(m.huber_objective_gradient(hi,x,target,weight)[0]-m.huber_objective_gradient(lo,x,target,weight)[0])/(2*epsilon)
    np.testing.assert_allclose(g,numeric,atol=3e-9,rtol=2e-6)
    W=theta[:20].reshape(5,4);b=theta[20:]
    r=x@W+b-target
    huber=np.where(np.abs(r)<=1,.5*r*r,np.abs(r)-.5)
    assert np.isclose(value,np.sum(weight[:,None]*huber)/4+.05*np.square(W).sum())


def test_huber_intercept_unpenalized():
    x=np.zeros((9,2));target=np.arange(36).reshape(9,4)/10;w=np.ones(9)/9
    theta=np.r_[np.zeros(8),[2.,-.4,1.,.1]]
    a,ga=m.huber_objective_gradient(theta,x,target,w,ridge=.1)
    b,gb=m.huber_objective_gradient(theta,x,target,w,ridge=100)
    assert a==b
    np.testing.assert_array_equal(ga,gb)


def test_regressor_predict_replay_and_convergence(tmp_path):
    x,y,donor,section,_=small()
    fit=m.ProgramRegressor.fit(x,y,donor,section)
    assert fit.fit_audit['success'] and fit.fit_audit['gradient_inf']<=1e-5
    assert fit.fit_audit['allocated_parameters']==40
    assert fit.fit_audit['regularized_parameters']==36
    path=tmp_path/'regression.npz';np.savez(path,**fit.arrays())
    with np.load(path,allow_pickle=False) as a:
        restored=m.ProgramRegressor.from_arrays(a)
        np.testing.assert_array_equal(fit.predict(x),restored.predict(x))
    np.testing.assert_allclose(fit.predict(x),fit.predict_standardized(x)*fit.y_standardizer.scale+fit.y_standardizer.mean)


def test_target_remapping_never_refits_y_normalizer():
    x,y,donor,section,tissue=small()
    index,audit=m.wrong_donor_target_map(donor,section,tissue)
    true=m.ProgramRegressor.fit(x,y,donor,section)
    wrong=m.ProgramRegressor.fit(x,y,donor,section,index)
    np.testing.assert_array_equal(true.arrays()['ymean'],wrong.arrays()['ymean'])
    np.testing.assert_array_equal(true.arrays()['yscale'],wrong.arrays()['yscale'])
    np.testing.assert_array_equal(true.arrays()['xmean'],wrong.arrays()['xmean'])
    assert not np.allclose(true.W,wrong.W)


def test_identity_target_map_matches_original_fit_exactly():
    x,y,donor,section,_=small()
    a=m.ProgramRegressor.fit(x,y,donor,section)
    b=m.ProgramRegressor.fit(x,y,donor,section,np.arange(len(y)))
    for k in a.arrays():np.testing.assert_array_equal(a.arrays()[k],b.arrays()[k])


def test_negative_control_maps_stay_in_train_rows_and_tissue():
    x,y,donor,section,tissue=small()
    idx,audit=m.wrong_donor_target_map(donor,section,tissue)
    assert np.all(donor[idx]!=donor)
    np.testing.assert_array_equal(tissue[idx],tissue)
    assert idx.min()>=0 and idx.max()<len(y)
    assert audit['global_bijection_claim'] is False
    assert audit['repeated_source_row_uses']>0
    np.testing.assert_array_equal(idx,m.wrong_donor_target_map(donor,section,tissue)[0])
    spatial,audit=m.spatial_shuffle_target_map(donor,section,tissue)
    np.testing.assert_array_equal(np.sort(spatial),np.arange(len(y)))
    np.testing.assert_array_equal(section[spatial],section)
    np.testing.assert_array_equal(tissue[spatial],tissue)
    assert np.all(spatial!=np.arange(len(y)))


def test_wrong_donor_requires_same_class_in_cyclic_source():
    with pytest.raises(ValueError,match='lacks'):
        m.wrong_donor_target_map(['a','a','b','b'],['a1','a1','b1','b1'],[0,1,0,0])
    with pytest.raises(ValueError,match='at least two'):
        m.wrong_donor_target_map(['a','a'],['s','s'],[0,1])


def test_spatial_singletons_are_explicit_and_other_groups_deranged():
    donor=np.array(['a']*5);section=np.array(['s']*5);tissue=np.array([0,1,1,2,2])
    idx,a=m.spatial_shuffle_target_map(donor,section,tissue)
    assert idx[0]==0 and np.all(idx[1:]!=np.arange(1,5))
    assert a['singleton_groups']==1 and a['fixed_rows']==1


def test_invalid_target_maps_missing_y_and_crossdonor_sections_rejected():
    x,y,donor,section,tissue=small()
    with pytest.raises(ValueError,match='in-bounds'):
        m.ProgramRegressor.fit(x,y,donor,section,np.arange(len(y))+1)
    with pytest.raises(ValueError,match='in-bounds'):
        m.ProgramRegressor.fit(x,y,donor,section,np.arange(len(y),dtype=float))
    y[0,0]=np.nan
    with pytest.raises(ValueError,match='no missing'):
        m.ProgramRegressor.fit(x,y,donor,section)
    with pytest.raises(ValueError,match='cross donors'):
        m.spatial_shuffle_target_map(['a','b'],['same','same'],[0,0])


def test_solver_unsuccessful_flag_blocks_publication(monkeypatch):
    x,y,donor,section,_=small()
    def failed(fun,theta,**kwargs):
        return SimpleNamespace(x=theta,success=False,status=1,message='synthetic maxiter',nit=2000,nfev=2001)
    monkeypatch.setattr(m,'minimize',failed)
    with pytest.raises(RuntimeError,match='OPTIMIZATION_FAILED'):
        m.ProgramRegressor.fit(x,y,donor,section)


def test_false_success_with_large_gradient_blocks(monkeypatch):
    x,y,donor,section,_=small()
    def false_success(fun,theta,**kwargs):
        return SimpleNamespace(x=theta,success=True,status=0,message='synthetic false success',nit=1,nfev=1)
    monkeypatch.setattr(m,'minimize',false_success)
    with pytest.raises(RuntimeError,match='OPTIMIZATION_FAILED'):
        m.ProgramRegressor.fit(x,y,donor,section)

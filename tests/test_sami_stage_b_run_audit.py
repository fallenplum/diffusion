"""Synthetic independent audit qualification, without any optimizer/model fit."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('sami_b_independent_audit',ROOT/'scripts/audit_sami_stage_b_run.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
from sami_surv import morphology as morph
from sami_surv import spatial_programs as spatial
from sami_surv.program_metrics import evaluate_programs


def groups():
    donor=np.repeat(['a','b','c','d','e','f'],[99,102,105,108,111,114])
    section=np.array([str(d)+str(i%2) for i,d in enumerate(donor)])
    tissue=np.arange(len(donor))%3
    return donor,section,tissue


def test_independent_hierarchical_weights_and_scaler():
    d,s,y=groups();rng=np.random.default_rng(1)
    for labels in (None,y):
        np.testing.assert_allclose(a.weights(d,s,labels),morph.hierarchical_weights(d,s,labels),atol=1e-16)
    x=rng.normal(size=(len(d),13));x[:,0]=2.
    m,scale=a.mean_scale(x,a.weights(d,s))
    original=morph.Standardizer.fit(x,d,s)
    np.testing.assert_allclose(m,original.mean,atol=1e-14)
    np.testing.assert_allclose(scale,original.scale,atol=1e-14)
    assert scale[0]==1.


def test_independent_teacher_objective_and_gradient():
    rng=np.random.default_rng(2);n=60;p=7
    x=rng.normal(size=(n,p));y=np.arange(n)%3;w=rng.uniform(size=n);w/=w.sum()
    model=dict(mean=rng.normal(size=p),scale=rng.uniform(.2,3,p),W=rng.normal(size=(p,3))*.1,b=rng.normal(size=3))
    z=(x-model['mean'])/model['scale'];theta=np.r_[model['W'].ravel(),model['b']]
    expected,grad=morph.objective_gradient(theta,z,y,w,3,1.)
    actual,ginf=a.teacher_objective_gradient(model,x,y,w)
    np.testing.assert_allclose([actual,ginf],[expected,np.abs(grad).max()],atol=1e-12)
    np.testing.assert_allclose(a.probabilities(x,model).sum(1),1.)


def test_independent_huber_objective_gradient_preserves_true_scaler():
    rng=np.random.default_rng(3);n=70;p=8
    x=rng.normal(size=(n,p));y=rng.normal(size=(n,4))*2;w=rng.uniform(size=n);w/=w.sum()
    model=dict(xmean=rng.normal(size=p),xscale=rng.uniform(.5,2,p),
        ymean=rng.normal(size=4),yscale=rng.uniform(.5,2,4),W=rng.normal(size=(p,4))*.2,b=rng.normal(size=4))
    mapping=np.arange(n)[::-1]
    expected,g=spatial.huber_objective_gradient(np.r_[model['W'].ravel(),model['b']],
        (x-model['xmean'])/model['xscale'],((y-model['ymean'])/model['yscale'])[mapping],w)
    loss,data,reg,ginf=a.huber_objective_gradient(model,x,y,w,mapping)
    np.testing.assert_allclose([loss,ginf],[expected,np.abs(g).max()],atol=1e-12)
    assert loss==data+reg


@pytest.mark.parametrize('arm',a.ARMS[1:])
def test_design_replay_all_six_arms(arm):
    rng=np.random.default_rng(4);p=rng.normal(size=(71,32));q=rng.normal(size=(71,32))
    pl=rng.dirichlet(np.ones(3),71);pn=rng.dirichlet(np.ones(3),71)
    np.testing.assert_array_equal(a.design_matrix(p,q,pl,pn,arm),spatial.build_design(p,q,pl,pn,arm))


def test_both_negative_remaps_match_locked_rules_and_no_donor_leakage():
    d,s,y=groups()
    wrong=a.target_mapping(d,s,y,'WRONG_DONOR')
    np.testing.assert_array_equal(wrong,spatial.wrong_donor_target_map(d,s,y)[0])
    assert np.all(d[wrong]!=d)
    np.testing.assert_array_equal(y[wrong],y)
    shuffled=a.target_mapping(d,s,y,'SPATIAL_SHUFFLE')
    np.testing.assert_array_equal(shuffled,spatial.spatial_shuffle_target_map(d,s,y)[0])
    np.testing.assert_array_equal(s[shuffled],s)
    np.testing.assert_array_equal(np.sort(shuffled),np.arange(len(d)))


def test_crossfit_teacher_requires_four_donors_not_five():
    d,s,y=groups()
    for outer in np.unique(d):
        for recipient in np.unique(d):
            rows=a.expected_teacher_rows(d,outer,recipient)
            assert outer not in d[rows] and recipient not in d[rows]
            assert len(np.unique(d[rows]))==(5 if recipient==outer else 4)
            if recipient!=outer:
                assert not np.array_equal(rows,np.flatnonzero(d!=outer))


@pytest.mark.parametrize('constant', ['none','truth','prediction'])
def test_independent_metric_tree_with_coverage_and_qc(constant):
    rng=np.random.default_rng(5)
    d=np.repeat(['a','b'],360);s=np.array([v+str((i//60)%2) for i,v in enumerate(d)])
    tissue=np.arange(len(d))%3;y=rng.normal(size=(len(d),4));prediction=y*.4+rng.normal(size=y.shape)
    if constant=='truth':y[:,2]=7.
    if constant=='prediction':prediction[:,1]=2.
    mean=rng.normal(size=4);scale=rng.uniform(.2,2,size=4)
    lib=rng.uniform(1,500,size=len(d));genes=rng.integers(50,500,size=len(d))
    expected=evaluate_programs(y,prediction,d,s,tissue,mean,scale,lib,genes)
    actual=a.program_metrics(y,prediction,d,s,tissue,mean,scale,lib,genes)
    assert a.compare_tree(actual,expected)<1e-12


def test_comparison_rejects_changed_values_and_false_convergence():
    with pytest.raises(ValueError,match='numeric mismatch'):
        a.compare_tree({'a':[1.,2.]},{'a':[1.,2.1]})
    fit=dict(success=True,accepted_gradient_inf=1e-5,gradient_inf=2e-6,objective=.3)
    a.check_success(fit,.3,2e-6)
    with pytest.raises(ValueError,match='convergence'):a.check_success(fit,.3,2e-4)
    with pytest.raises(ValueError,match='convergence'):a.check_success(fit,.3,np.nan)
    with pytest.raises(ValueError,match='convergence'):a.check_success(dict(fit,success=False),.3,2e-6)


def test_gate_diagnostics_constant_uniform_and_one_hot():
    w=np.ones(12)/12
    uniform=a.gate_diagnostics(np.ones((12,3))/3,w)
    assert np.isclose(uniform['mean_entropy_nats'],np.log(3))
    assert uniform['centered_covariance_effective_rank']==0.
    onehot=a.gate_diagnostics(np.eye(3)[np.arange(12)%3],w)
    assert onehot['mean_entropy_nats']==0.
    assert np.isclose(onehot['mean_max_probability'],1.)
    assert np.isclose(onehot['centered_covariance_effective_rank'],2.)

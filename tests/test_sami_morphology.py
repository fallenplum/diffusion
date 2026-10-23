"""Synthetic qualification only: no real tissue/patient inputs or scientific fits."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sami_surv.morphology import (Standardizer, MultinomialModel, classification_metrics,
    fit_model, hierarchical_weights, objective_gradient, outer_inner_splits, select_c, train_prior)

spec = importlib.util.spec_from_file_location('sami_runner', ROOT/'scripts/run_sami_stage_a.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic(seed=1):
    rng = np.random.default_rng(seed)
    donor, section, y = [], [], []
    for d in range(6):
        for s in range(2):
            for c in range(3):
                for _ in range(3):
                    donor.append(f'private_donor_{d}')
                    section.append(f'private_section_{d}_{s}')
                    y.append(c)
    y = np.array(y, np.int64)
    x = rng.normal(size=(len(y), 7)).astype(np.float32)
    x[np.arange(len(x)), y] += 1.5
    return dict(X=x,y=y,donor=np.array(donor),section=np.array(section),
                barcodes=np.array([f'private_barcode_{i}' for i in range(len(x))]))


def config():
    return dict(label_contract=dict(class_order=['tumor','stroma','tumor_stroma_mixed']),
                model=dict(C_candidates=[.01,.1,1.],maxiter=2000,gtol=1e-6,ftol=1e-12,
                           maxls=50,accepted_gradient_inf=1e-5),
                selection=dict(tie_tolerance=1e-12))


def test_weights_give_each_donor_equal_mass():
    donor = np.array(['a']*5+['b']*3)
    section = np.array(['a1']*3+['a2']*2+['b1']*3)
    y = np.array([0,0,1,0,1,0,1,1])
    for labels in [None,y]:
        w = hierarchical_weights(donor,section,labels)
        assert np.isclose(w.sum(),1)
        assert np.isclose(w[donor=='a'].sum(),.5)
        assert np.isclose(w[donor=='b'].sum(),.5)
    w = hierarchical_weights(donor,section,y)
    for d in np.unique(donor):
        for c in np.unique(y):
            assert np.isclose(w[(donor==d)&(y==c)].sum(),.25)


def test_weighted_gradient_finite_difference():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(13,4)); y=np.arange(13)%3
    weight=rng.uniform(size=13);weight/=weight.sum()
    theta=rng.normal(size=15)*.2
    value,g=objective_gradient(theta,x,y,weight,3,.1)
    epsilon=1e-6
    numeric=np.empty_like(g)
    for j in range(len(theta)):
        lo,hi=theta.copy(),theta.copy();lo[j]-=epsilon;hi[j]+=epsilon
        numeric[j]=(objective_gradient(hi,x,y,weight,3,.1)[0]-objective_gradient(lo,x,y,weight,3,.1)[0])/(2*epsilon)
    np.testing.assert_allclose(g,numeric,atol=2e-9,rtol=2e-7)


def test_intercept_is_unpenalized():
    x=np.zeros((6,2));y=np.arange(6)%3;w=np.ones(6)/6
    theta=np.array([0.]*6+[.2,-.5,.7])
    a,ga=objective_gradient(theta,x,y,w,3,.01)
    b,gb=objective_gradient(theta,x,y,w,3,1.)
    assert a==b
    np.testing.assert_array_equal(ga,gb)


def test_standardization_uses_only_training_rows_and_no_labels():
    x=np.array([[1.,7.],[3.,7.],[1e9,-1e9]])
    scaler=Standardizer.fit(x[:2],np.array(['a','a']),np.array(['s','s']))
    np.testing.assert_array_equal(scaler.mean,[2.,7.])
    np.testing.assert_array_equal(scaler.scale,[1.,1.])
    np.testing.assert_array_equal(scaler.transform(x[:2]),[[-1,0],[1,0]])
    assert scaler.transform(x[2:])[0,0]>1e8


def test_hierarchical_standardization_not_pooled_spot_mean():
    x=np.array([[0.],[2.],[100.]])
    scaler=Standardizer.fit(x,['a','a','b'],['s1','s1','s2'])
    np.testing.assert_allclose(scaler.mean,[50.5])


def test_nested_folds_never_share_donors():
    data=synthetic()
    folds=list(outer_inner_splits(data['donor']))
    assert len(folds)==6
    seen=np.zeros(len(data['y']),int)
    for _,tr,te,inners in folds:
        seen[te]+=1
        assert not set(data['donor'][tr])&set(data['donor'][te])
        assert len(inners)==5
        for it,iv in inners:
            assert not set(it)&set(iv)
            assert set(it)|set(iv)==set(tr)
            assert not set(data['donor'][it])&set(data['donor'][iv])
            assert not set(iv)&set(te)
    np.testing.assert_array_equal(seen,1)


def test_fit_prediction_checkpoint_replay(tmp_path):
    d=synthetic();train=np.arange(72)
    model=fit_model(d['X'][train],d['y'][train],d['donor'][train],d['section'][train],n_classes=3,C=.1)
    assert model.fit_audit['success']
    assert model.fit_audit['gradient_inf']<=1e-5
    path=tmp_path/'model.npz';np.savez(path,**model.arrays())
    with np.load(path,allow_pickle=False) as a:
        replay=MultinomialModel.from_arrays(a)
        np.testing.assert_array_equal(model.predict_proba(d['X'][72:]),replay.predict_proba(d['X'][72:]))


def test_nonconvergence_blocks():
    d=synthetic()
    with pytest.raises(RuntimeError,match='OPTIMIZATION_FAILED'):
        fit_model(d['X'],d['y'],d['donor'],d['section'],n_classes=3,C=1.,maxiter=0)


def test_missing_training_class_blocks():
    d=synthetic();mask=d['y']<2
    with pytest.raises(ValueError,match='Every class'):
        fit_model(d['X'][mask],d['y'][mask],d['donor'][mask],d['section'][mask],n_classes=3,C=1.)


def test_balanced_accuracy_is_class_then_section_equal():
    y=np.array([0,0,0,1,1,1]); pred=np.array([0,0,1,1,1,0])
    p=np.eye(2)[pred]
    metrics=classification_metrics(y,p,np.array(['d']*6),np.array(['a','a','b','b','b','b']),2)
    np.testing.assert_allclose(metrics['class_recall'],[.5,2/3])
    assert np.isclose(metrics['balanced_accuracy'],7/12)


def test_probability_metrics_are_natural_not_class_balanced():
    y=np.array([0,0,1]);p=np.array([[.8,.2],[.8,.2],[.3,.7]])
    m=classification_metrics(y,p,['d']*3,['a']*3,2)
    assert np.isclose(m['negative_log_likelihood'],-(2*np.log(.8)+np.log(.7))/3)
    assert np.isclose(m['brier_multiclass_sum'],(.08+.08+.18)/3)
    assert np.isclose(m['ece_10_equal_width_bins'],(2*.2+.3)/3)


def test_prior_is_train_natural_weighted():
    p=train_prior(np.array([0,0,1]),['a','a','b'],['a1','a1','b1'],2)
    np.testing.assert_allclose(p,[.5,.5])


def test_selection_requires_all_inner_scores_and_fixed_tie():
    c,means=select_c({.01:[.7,.7],.1:[.7,.7],1.:[.6,.8]},[.01,.1,1.])
    assert c==.01
    with pytest.raises(ValueError):select_c({.01:[.5],.1:[]},[.01,.1])
    with pytest.raises(ValueError):select_c({.01:[np.nan]},[.01])


def test_outer_labels_do_not_change_selection_or_fit(tmp_path):
    a=synthetic();b={k:v.copy() for k,v in a.items()}
    split=list(outer_inner_splits(a['donor']))[0]
    b['y'][split[2]]=(b['y'][split[2]]+1)%3
    lineage={'synthetic':'only'}
    ra=runner.run_outer(tmp_path/'a',a,config(),lineage,split)
    rb=runner.run_outer(tmp_path/'b',b,config(),lineage,split)
    assert ra['selected_C']==rb['selected_C']
    assert ra['inner_scores']==rb['inner_scores']
    for field in ['W','b','mean','scale']:
        with np.load(tmp_path/'a'/'cells'/'outer_00'/ra['final_cell']/'model.npz') as ma, np.load(tmp_path/'b'/'cells'/'outer_00'/rb['final_cell']/'model.npz') as mb:
            np.testing.assert_array_equal(ma[field],mb[field])


def test_complete_cell_resume_and_tamper_refusal(tmp_path):
    d=synthetic();split=list(outer_inner_splits(d['donor']))[0]
    tr,va=split[3][0];cell=tmp_path/'cell'
    first=runner.fitted_cell(cell,d,tr,va,.1,config()['model'],{'source':'abc'})
    second=runner.fitted_cell(cell,d,tr,va,.1,config()['model'],{'source':'abc'})
    assert first==second
    with pytest.raises(ValueError,match='lineage'):
        runner.fitted_cell(cell,d,tr,va,.1,config()['model'],{'source':'different'})
    (cell/'model.npz').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='hash'):
        runner.fitted_cell(cell,d,tr,va,.1,config()['model'],{'source':'abc'})


def test_incomplete_cell_is_never_silently_refit(tmp_path):
    d=synthetic();tr,va=list(outer_inner_splits(d['donor']))[0][3][0]
    cell=tmp_path/'cell';cell.mkdir();(cell/'START.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Incomplete'):
        runner.fitted_cell(cell,d,tr,va,.1,config()['model'],{'source':'abc'})


def test_synthetic_96_fit_finalization_anonymous_and_complete(tmp_path):
    d=synthetic();cfg=config();lineage={'synthetic':'only'}
    splits=list(outer_inner_splits(d['donor']))
    for split in splits:
        runner.run_outer(tmp_path/'private',d,cfg,lineage,split)
    receipt=runner.finalize(tmp_path/'private',tmp_path/'public',d,cfg,lineage,splits)
    assert receipt['completed_fits']==96
    text=(tmp_path/'public'/'RESULTS.json').read_text()
    assert 'private_donor_' not in text
    assert 'private_section_' not in text
    assert 'private_barcode_' not in text
    with np.load(tmp_path/'private'/'oof_predictions.npz') as a:
        assert len(a['probabilities'])==len(d['X'])
        np.testing.assert_allclose(a['probabilities'].sum(1),1.)
    assert runner.finalize(tmp_path/'private',tmp_path/'public',d,cfg,lineage,splits)==receipt


def test_private_outputs_cannot_enter_repository():
    with pytest.raises(ValueError,match='outside repository'):
        runner.validate_output_paths(ROOT/'reports'/'bad',ROOT/'reports'/'sami','x.npz','m.json')
    with pytest.raises(ValueError,match='outside repository'):
        runner.validate_output_paths(ROOT/'src'/'bad',ROOT/'reports'/'sami','x.npz','m.json')


def test_ece_confidence_boundary_has_single_bin():
    y=np.array([0,1,0]);p=np.array([[.7,.3],[.6,.4],[1.,0.]])
    m=classification_metrics(y,p,['d']*3,['s']*3,2)
    assert np.isclose(m['ece_10_equal_width_bins'],(.3+.6)/3)


def test_original_split_lock_must_match_protocol_before_loading_arrays(tmp_path):
    protocol = runner.json_read(ROOT/'configs/sami_surv_stage_a_v1.json')
    config_path = tmp_path/'config.json'
    runner.json_write(config_path,protocol)
    prepared = tmp_path/'prepared.npz'
    prepared.write_bytes(b'not loaded because provenance must fail first')
    manifest = dict(schema_version='sami_stage_a_prepared_v1',
                    status='PASS_PREPARED_FROZEN_TISSUE3',data_sha256=runner.sha256(prepared),
                    protocol_sha256=runner.sha256(config_path),
                    classes=protocol['label_contract']['class_order'],
                    mapping_sha256=protocol['label_contract']['mapping_sha256'],
                    DSS_read=False,expression_array_accessed=False,original_ST_test_arrays_read=False,
                    original_split_lock_sha256='wrong-source-split')
    manifest_path=tmp_path/'manifest.json'
    runner.json_write(manifest_path,manifest)
    with pytest.raises(ValueError,match='Original development split lock'):
        runner.load_locked_inputs(prepared,manifest_path,config_path)

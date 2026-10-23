"""Synthetic routing/scheduling checks; every optimizer is replaced by a stub."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_sami_stage_b.py'
spec = importlib.util.spec_from_file_location('sami_stage_b_runner_under_test', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic_data():
    n = 36
    donor = np.repeat(np.array(['D0','D1','D2','D3','D4','D5']), 6)
    section = np.array([f'S{i//3}' for i in range(n)])
    x = np.c_[np.arange(n), np.arange(n)+0.25].astype(float)
    return dict(X=x, neighbor_X=x+1000, donor=donor, section=section,
                tissue_y=np.tile([0,1,2,0,1,2], 6),
                program_y=np.c_[np.arange(n),np.arange(n)+1,np.arange(n)+2,np.arange(n)+3],
                library_size=np.ones(n)*100, n_detected_genes=np.ones(n)*25)


def install_teacher_stubs(monkeypatch, plan):
    """Receipts and models are synthetic; captures prove the actual predict inputs."""
    predictions = []
    receipts = []
    class Model:
        def __init__(self, teacher_index):
            self.teacher_index = teacher_index
        def predict_proba(self, x):
            predictions.append((self.teacher_index, x.copy()))
            a = 0.1 + 0.001*self.teacher_index + 0.00001*x[:, 0]
            return np.c_[a, np.full(len(x), .3), .7-a]
    class Loader:
        @staticmethod
        def from_arrays(a):
            return Model(int(a['teacher_index']))
    def fake_check(directory, lineage, artifacts):
        ti = int(directory.name.split('_')[1])
        assert lineage == plan['lineage']
        assert artifacts == ['model','indices']
        receipts.append(ti)
        return dict(status='COMPLETE',train_index_sha256=plan['teachers'][ti]['train_index_sha256'])
    def fake_arrays(path):
        assert path.name == 'model.npz'
        return dict(teacher_index=int(path.parent.name.split('_')[1]))
    monkeypatch.setattr(runner,'check_done',fake_check)
    monkeypatch.setattr(runner,'arrays',fake_arrays)
    monkeypatch.setattr(runner,'MultinomialModel',Loader)
    return predictions, receipts


def test_exact_plan_has_21_distinct_teacher_training_sets_and_six_outer_folds():
    data = synthetic_data()
    plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    sets = [tuple(s['train_donors']) for s in plan['teachers']]
    donors = np.unique(data['donor']).tolist()
    assert len(sets) == len(set(sets)) == 21
    assert sets == list(itertools.combinations(donors,4))+list(itertools.combinations(donors,5))
    assert [s['teacher_index'] for s in plan['teachers']] == list(range(21))
    assert len(plan['outer']) == 6
    assert plan['optimized_models'] == 57 and plan['hyperparameter_selection'] is False
    for item in plan['teachers']:
        expected = np.flatnonzero(np.isin(data['donor'],item['train_donors']))
        assert item['train_index_sha256'] == runner.index_hash(expected)
    for oi, fold in enumerate(plan['outer']):
        assert fold['outer_index'] == oi and fold['heldout_donor'] == donors[oi]
        assert fold['test_index_sha256'] == runner.index_hash(np.flatnonzero(data['donor']==donors[oi]))


@pytest.mark.parametrize('outer_index', range(6))
def test_teacher_features_exclude_outer_and_recipient_with_exact_model_inputs(monkeypatch, tmp_path, outer_index):
    data = synthetic_data()
    plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    calls, receipts = install_teacher_stubs(monkeypatch,plan)
    pl,pn,assignment = runner.teacher_features(tmp_path,data,plan,outer_index)
    held = plan['outer'][outer_index]['heldout_donor']
    outer_train = set(np.unique(data['donor']))-{held}
    assert len(calls)==12 and len(receipts)==6
    for j, recipient in enumerate(np.unique(data['donor'])):
        rows = np.flatnonzero(data['donor']==recipient)
        ti = int(assignment[rows[0]])
        assert np.all(assignment[rows]==ti)
        actual_train = set(plan['teachers'][ti]['train_donors'])
        assert actual_train == outer_train-{recipient}
        assert held not in actual_train and recipient not in actual_train
        assert len(actual_train)==(5 if recipient==held else 4)
        assert calls[2*j][0] == calls[2*j+1][0] == ti
        np.testing.assert_array_equal(calls[2*j][1],data['X'][rows])
        np.testing.assert_array_equal(calls[2*j+1][1],data['neighbor_X'][rows])
        assert np.all(pl[rows,0] != pn[rows,0])
    np.testing.assert_allclose(pl.sum(1),1.)
    np.testing.assert_allclose(pn.sum(1),1.)


def test_changing_outer_labels_and_all_program_targets_cannot_change_teacher_features(monkeypatch,tmp_path):
    data = synthetic_data()
    plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    install_teacher_stubs(monkeypatch,plan)
    expected = runner.teacher_features(tmp_path,data,plan,2)
    modified = {k:v.copy() for k,v in data.items()}
    held = modified['donor']=='D2'
    modified['tissue_y'][held] = (modified['tissue_y'][held]+1)%3
    # The teacher feature routine has no legitimate need for ANY molecular target.
    modified['program_y'] = np.full(modified['program_y'].shape,np.nan)
    observed = runner.teacher_features(tmp_path,modified,plan,2)
    for a,b in zip(expected,observed):
        np.testing.assert_array_equal(a,b)


def test_changed_outer_visual_values_cannot_change_training_donor_features(monkeypatch,tmp_path):
    data = synthetic_data()
    plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    install_teacher_stubs(monkeypatch,plan)
    a = runner.teacher_features(tmp_path,data,plan,3)
    modified = {k:v.copy() for k,v in data.items()}
    held = modified['donor']=='D3'
    modified['X'][held] += 11
    modified['neighbor_X'][held] -= 23
    b = runner.teacher_features(tmp_path,modified,plan,3)
    np.testing.assert_array_equal(a[0][~held],b[0][~held])
    np.testing.assert_array_equal(a[1][~held],b[1][~held])
    assert np.any(a[0][held]!=b[0][held]) and np.any(a[1][held]!=b[1][held])


def test_teacher_cache_from_wrong_subset_is_rejected(monkeypatch,tmp_path):
    data = synthetic_data()
    plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    install_teacher_stubs(monkeypatch,plan)
    monkeypatch.setattr(runner,'check_done',lambda *a,**k:dict(train_index_sha256='wrong-subset'))
    with pytest.raises(ValueError,match='another training subset'):
        runner.teacher_features(tmp_path,data,plan,0)


def test_missing_teacher_receipt_is_not_fitted_by_feature_loading(monkeypatch,tmp_path):
    data = synthetic_data()
    plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    def forbidden(*a,**k):
        pytest.fail('Feature loading attempted a model fit')
    monkeypatch.setattr(runner,'fit_model',forbidden)
    with pytest.raises(FileNotFoundError):
        runner.teacher_features(tmp_path,data,plan,0)


def test_teacher_fit_receives_only_prescribed_rows_and_fixed_C(monkeypatch,tmp_path):
    data = synthetic_data()
    plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    item = plan['teachers'][4]
    called = []
    class FakeFittedModel:
        fit_audit = dict(success=True,gradient_inf=0.,accepted_gradient_inf=1e-5)
        def arrays(self):
            return dict(fake=np.zeros(1))
    def fake_fit(x,y,d,s,**kwargs):
        called.append((x.copy(),y.copy(),d.copy(),s.copy(),kwargs))
        return FakeFittedModel()
    monkeypatch.setattr(runner,'fit_model',fake_fit)
    receipt = runner.run_teacher(tmp_path,data,plan['lineage'],item)
    rows = np.flatnonzero(np.isin(data['donor'],item['train_donors']))
    assert len(called)==1
    for actual, key in zip(called[0][:4],['X','tissue_y','donor','section']):
        np.testing.assert_array_equal(actual,data[key][rows])
    assert called[0][4] == dict(n_classes=3,C=1.)
    assert receipt['status']=='COMPLETE'
    runner.run_teacher(tmp_path,data,plan['lineage'],item)
    assert len(called)==1, 'Completed teacher was unnecessarily retrained'


def test_interrupted_teacher_is_not_silently_retrained(monkeypatch,tmp_path):
    data = synthetic_data(); plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    item = plan['teachers'][0]
    cell = tmp_path/'teachers'/'t_00'
    runner.claim(cell,dict(lineage=plan['lineage']))
    monkeypatch.setattr(runner,'fit_model',lambda *a,**k:pytest.fail('Hidden retry'))
    with pytest.raises(FileExistsError):
        runner.run_teacher(tmp_path,data,plan['lineage'],item)


@pytest.mark.parametrize('metric', ['huber','mse'])
def test_bootstrap_lower_error_has_positive_improvement(metric):
    result = runner.paired_bootstrap([1.]*6,[2.]*6,metric)
    assert result['mean_improvement']==1 and result['ci95']==[1.,1.]
    assert result['positive_donors']==6
    assert result['bootstrap_repeats']==20000 and result['bootstrap_unit']=='donor'


@pytest.mark.parametrize('metric', ['within_tissue_spearman','within_section_spearman'])
def test_bootstrap_higher_correlation_has_positive_improvement(metric):
    result = runner.paired_bootstrap([.5]*6,[.25]*6,metric)
    assert result['mean_improvement']==.25 and result['ci95']==[.25,.25]
    assert result['positive_donors']==6


def test_bootstrap_pairs_donors_instead_of_independently_resampling_arms():
    # Large shared donor offsets cancel exactly. Independent arm resampling would
    # produce an interval hundreds of times wider than the interval of deltas.
    shared = [0.,100.,200.,300.,400.,500.]
    delta = [-.3,-.2,-.1,.1,.2,.3]
    candidate = [x+d for x,d in zip(shared,delta)]
    result = runner.paired_bootstrap(candidate,shared,'within_tissue_spearman')
    assert abs(result['mean_improvement'])<1e-12
    assert -0.3<=result['ci95'][0]<0<result['ci95'][1]<=0.3
    assert result['positive_donors']==3
    assert result == runner.paired_bootstrap(candidate,shared,'within_tissue_spearman')


def test_bootstrap_missing_values_are_explicit_not_dropped():
    result = runner.paired_bootstrap([None,1.,1.,1.,1.,1.],[1.]*6,'huber')
    assert result['status']=='UNDEFINED_MISSING_PROGRAM_OR_DONOR'
    assert 'mean_improvement' not in result
    with pytest.raises(ValueError,match='six'):
        runner.paired_bootstrap([1.]*5,[2.]*5,'huber')


def test_finalize_direct_api_never_starts_missing_outer_fit(monkeypatch,tmp_path):
    data = synthetic_data(); plan = runner.make_plan(data,{'fake_lineage':'fixed'})
    # Satisfy teachers without introducing real model files. All outer receipts
    # deliberately remain absent; even direct finalize() must fail BEFORE run_outer.
    monkeypatch.setattr(runner,'check_done',lambda *a,**k:dict(fit_audit={}))
    def forbidden(*a,**k):
        pytest.fail('finalize() attempted to execute a missing outer training cell')
    monkeypatch.setattr(runner,'run_outer',forbidden)
    with pytest.raises((ValueError,FileNotFoundError),match='incomplete|Incomplete|COMPLETE'):
        runner.finalize(tmp_path,data,plan)


def test_cli_finalize_missing_fold_fails_without_calling_finalize(monkeypatch,tmp_path):
    data = synthetic_data(); lineage={'fake_lineage':'fixed'}
    plan = runner.make_plan(data,lineage)
    run = tmp_path/'run'; run.mkdir()
    runner.write_json(run/'PLAN.json',plan)
    monkeypatch.setattr(runner,'load_inputs',lambda p:(data,{},lineage))
    monkeypatch.setattr(runner,'finalize',lambda *a,**k:pytest.fail('Incomplete CLI finalize was executed'))
    monkeypatch.setattr(runner.sys,'argv',['run_sami_stage_b.py','--prepared',str(tmp_path),
                                         '--run',str(run),'--mode','finalize'])
    with pytest.raises(ValueError,match='incomplete'):
        runner.main()

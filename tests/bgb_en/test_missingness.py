import importlib.util
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[2]/'scripts/bgb_en_missingness.py'
SPEC=importlib.util.spec_from_file_location('bgb_missingness_tested',PATH)
missingness=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(missingness)


def test_subgroups_keep_original_fit_censor_distribution_once(monkeypatch):
    tt=np.array([1.,2.,3.,5.,7.,9.]);te=np.array([1,0,1,0,1,1])
    t=np.array([1.5,2.5,4.,6.]);e=np.array([1,1,0,0]);r=np.array([3.,2.,1.,0.])
    calls=[];original=missingness.uno_weights
    def observe(train_time,train_event,time,event,tau):
        calls.append((np.asarray(train_time).copy(),np.asarray(train_event).copy()))
        return original(train_time,train_event,time,event,tau)
    monkeypatch.setattr(missingness,'uno_weights',observe)
    rows=missingness.stratified_metrics(tt,te,t,e,r,{'ALL':np.ones(4,bool),'SUBSET':np.array([1,1,0,0],bool),'EMPTY':np.zeros(4,bool)})
    assert len(calls)==1
    np.testing.assert_array_equal(calls[0][0],tt);np.testing.assert_array_equal(calls[0][1],te)
    assert rows['ALL']['Harrell']==rows['SUBSET']['Harrell']==1.
    assert rows['EMPTY']['Harrell'] is None and rows['EMPTY']['Uno60'] is None
    assert rows['EMPTY']['Harrell_den']==rows['EMPTY']['Uno_den']==0.


def test_measured_zero_is_not_missing_and_strata_do_not_rewrite_data():
    rna=np.array([[0.,1.],[np.nan,2.],[3.,0.]])
    cna=np.array([[0.,0.],[1.,np.nan],[0.,2.]])
    before=rna.copy()
    groups=missingness.missingness_groups({'rna':rna,'cna':cna},canonical=False)
    np.testing.assert_array_equal(groups['RNA_COMPLETE'],[True,False,True])
    np.testing.assert_array_equal(groups['CNA_MINOR_MISSING'],[False,True,False])
    np.testing.assert_allclose(rna,before,equal_nan=True)

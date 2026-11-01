import copy
import numpy as np
import pytest
import torch
from src.rscil.data import generate,fit_transform,fit_warm,disjoint
from src.rscil.model import RSCIL,encode
from src.rscil.train import fit,predict,state_hash
from src.rscil.protocol import require_synthetic_go
from src.semantic_m123.models import Branch


def test_explicit_branch_encode_exact():
    torch.manual_seed(99);branch=Branch(7).double();x=torch.randn(13,7,dtype=torch.float64)
    assert torch.equal(branch(x),branch.head(encode(branch,x)).squeeze(-1))


def setup():
    raw,ev=generate(101,512,'INTERACTION');tr,va,_=fit_transform(raw,ev)
    warm=fit_warm(tr,raw['time'],raw['event'])
    return raw,ev,tr,va,warm


def test_warm_and_complete_r2_match_including_nonzero_mean_path():
    raw,ev,tr,va,warm=setup();torch.manual_seed(19)
    model=RSCIL('S0',12,12,6);model.warm(warm)
    expected=sum(tr[k]@warm[l+'_coef'] for k,l in [('P','p'),('G','g'),('C','c')])
    assert (model(tr)['risk']-expected).abs().max()<1e-10
    with torch.no_grad():model.additive.local_out.weight.normal_(std=.01)
    tr['patch']=tr['mean_patch'][:,None,:];tr['mask']=torch.ones(len(tr['P']),1,dtype=torch.bool);tr['slide']=torch.zeros(len(tr['P']),1,dtype=torch.long)
    assert (model(tr)['risk']-model.additive(tr)['risk']).abs().max()<1e-12


def test_added_parameter_budget():
    for rank in (4,8):
        a=RSCIL('S1',12,12,6,rank).added_parameters();b=RSCIL('S1S',12,12,6,rank).added_parameters()
        assert abs(a-b)/a<=.05


def test_outcome_isolation_actual_training_gradient_and_disk_replay(tmp_path):
    raw,ev,tr,va,warm=setup();hashes=[];risks=[]
    for contaminate in (False,True):
        changed=copy.deepcopy(ev)
        if contaminate:changed['time']*=999;changed['event']=~changed['event']
        torch.manual_seed(21);m=RSCIL('S3',12,12,6);m.warm(warm)
        result=fit(m,tr,raw['time'],raw['event'],epochs=30)
        hashes.append(state_hash(m));risks.append(predict(m,va)['risk'])
    assert hashes[0]==hashes[1] and torch.equal(risks[0],risks[1])
    assert result['updates']['generator']==20
    assert result['history'][-1]['grad_Z_Cox_norm']>0
    assert result['history'][10]['Cox_gradient_generator']>0
    path=tmp_path/'synthetic_replay.pt';torch.save(m.state_dict(),path)
    replay=RSCIL('S3',12,12,6);replay.load_state_dict(torch.load(path,weights_only=True));replay.stage(30)
    assert torch.equal(predict(replay,va)['risk'],risks[-1])


def test_no_a_at_test_and_id_overlap_fail_closed():
    raw,ev,tr,va,warm=setup();m=RSCIL('S3',12,12,6);m.warm(warm)
    with pytest.raises(ValueError,match='TRAIN_FITTED_A'):predict(m,va)
    with pytest.raises(ValueError,match='OVERLAP'):disjoint(['a','b'],['b','c'])
    with pytest.raises(ValueError,match='DENIED'):require_synthetic_go({'theory_pass':True,'synthetic_neural_pass':False})


def test_synthetic_splits_independent_and_event_counts_exact():
    for n in (512,2048):
        tr,te=generate(101,n,'CORRELATED_INTERACTION')
        assert len(tr['P'])+len(te['P'])==n
        assert tr['event'].sum()+te['event'].sum()==(65 if n==512 else 260)

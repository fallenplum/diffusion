"""Actual complete-risk/checkpoint gradients, intervention scope and refits."""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase20_selective import train
from src.phase19_attention.data import Data as ParentData
from src.phase20_selective.model import SelectiveModel
from src.phase17_ssd.losses import full_risk_efron


def config():
    return {'hidden':32,'tokenizer':{'filters':16},'training':{
        'dropout':.2,'microbatch_patients':2,'max_epochs':4,'minimum_epoch':1,
        'early_stopping_patience':3,'validation_ema_alpha':.2,'learning_rate':.001,
        'weight_decay':.01,'gradient_clip':5.}}


def synthetic(role='inner',device='cpu',dtype=torch.float64):
    rng=np.random.default_rng(202019)
    n=10; g=24
    f={key:rng.normal(size=(n,d)).astype(float) for key,d in [('C',3),('R',g),('G',g),('L',g),('X',7)]}
    f['RM']=np.ones((n,g));f['CM']=np.ones((n,g))
    f['RM'][2,:3]=0; f['CM'][3,3:6]=0
    f['membership']=np.zeros((8,g),bool)
    for j in range(8): f['membership'][j,j*3:j*3+3]=True
    w=np.linspace(-.1,.1,7)
    f['ANCHOR']=np.asarray(f['X']@w,np.float32)
    owner=np.array([0,0,1,2,3,3,4,5,5,6,7])
    visual=torch.as_tensor(rng.normal(size=(len(owner),3,1536)),dtype=dtype,device=device)
    mask=torch.ones(len(owner),3,dtype=torch.bool,device=device);mask[0,1:]=False
    fit=np.arange(6);valid=np.array([6,7]);active=np.arange(8)
    wrong=np.arange(n);wrong[fit]=np.roll(fit,1);wrong[valid]=valid[::-1]
    ds=SimpleNamespace(fold=1,role=role,seed=31,f=f,
        ft={k:torch.as_tensor(v,dtype=torch.bool if k=='membership' else dtype,device=device) for k,v in f.items()},
        owner=owner,visual=visual,mask=mask,fit=fit,valid=valid,active=active,
        part={'outer_test':np.array([8,9])},wrong_c=wrong.copy(),wrong_o=wrong.copy(),
        global_wsi_indices=np.arange(len(owner)),anchor_w=w.copy(),anchor_sd_fit=float(np.std(f['X'][fit]@w)),
        anchor_reference={'source':'synthetic'},audit={'source':'synthetic'},
        gradient_probe_indices=fit.copy())
    ds.inherited_simple={'models':{'J0':{'w':w,'risk':f['ANCHOR'][valid].copy()}}}
    ds.ep={'time':np.array([1,2,2,3,4,5,6,7,np.nan,np.nan]),
           'event':np.array([1,1,0,1,0,1,1,0,np.nan,np.nan]),'valid':np.isin(np.arange(n),active)}
    ds.check_indices=lambda ix: ParentData.check_indices(ds,ix)
    return ds


@pytest.fixture(autouse=True)
def deterministic_state():
    enabled=torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(enabled)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_checkpoint_dropout_matches_one_complete_risk_graph(monkeypatch,device):
    monkeypatch.setattr(train,'cfg',config)
    ds=synthetic(device=device)
    ds.visual.requires_grad_()
    for key in ['C','R','G','L']: ds.ft[key].requires_grad_()
    ref=copy.deepcopy(ds);ref.check_indices=lambda ix: ParentData.check_indices(ref,ix)
    torch.manual_seed(812)
    model=SelectiveModel(3,ds.f['membership'],ds.anchor_w,ds.f['C'][ds.fit].mean(0),ds.anchor_sd_fit,'S16',config()).to(device=device,dtype=torch.float64)
    oracle=copy.deepcopy(model)
    model.train();oracle.train()
    torch.manual_seed(113)
    a=train.forward(model,ds,ds.fit,training=True)
    full_risk_efron(a,ds.ep['time'][ds.fit],ds.ep['event'][ds.fit]).backward()
    torch.manual_seed(113)
    b=train.forward(oracle,ref,ref.fit,training=False)
    full_risk_efron(b,ref.ep['time'][ref.fit],ref.ep['event'][ref.fit]).backward()
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    for (name,p),(_,q) in zip(model.named_parameters(),oracle.named_parameters()):
        if p.grad is None: assert q.grad is None,name
        else: torch.testing.assert_close(p.grad,q.grad,atol=1e-9,rtol=1e-8,msg=name)
    for key in ['C','R','G','L']:
        torch.testing.assert_close(ds.ft[key].grad,ref.ft[key].grad,atol=1e-10,rtol=1e-8)
    assert 'main' not in dict(model.named_parameters())
    assert torch.equal(model.main.cpu(),torch.as_tensor(ds.anchor_w,dtype=torch.float64))


def test_interventions_preserve_exact_anchor_and_base(monkeypatch):
    monkeypatch.setattr(train,'cfg',config)
    ds=synthetic(dtype=torch.float32)
    model=SelectiveModel(3,ds.f['membership'],ds.anchor_w,ds.f['C'][ds.fit].mean(0),ds.anchor_sd_fit,'S16',config()).eval()
    results={mode:train.forward(model,ds,ds.valid,mode,diagnostic=True) for mode in train.MODES}
    for mode,(_,diags) in results.items():
        for j,diag in enumerate(diags):
            reference=results['matched'][1][j]
            for key in ['main_risk','pair_risk','base_p_to_o','base_o_to_p']:
                torch.testing.assert_close(diag[key],reference[key],atol=0,rtol=0)
    torch.testing.assert_close(results['constantC'][0],results['off'][0],atol=0,rtol=0)
    np.testing.assert_array_equal(train.forward(model,ds,ds.valid,'anchor').detach().numpy(),ds.inherited_simple['models']['J0']['risk'])


def test_gradient_override_only_reaches_new_correction(monkeypatch):
    monkeypatch.setattr(train,'cfg',config)
    ds=synthetic(dtype=torch.float32)
    model=SelectiveModel(3,ds.f['membership'],ds.anchor_w,ds.f['C'][ds.fit].mean(0),ds.anchor_sd_fit,'S16',config()).eval()
    c=ds.ft['C'].detach().clone().requires_grad_();r=ds.ft['R'].detach().clone().requires_grad_()
    kwargs={'context_overrides':{'clinical':c,'R':r}}
    corrected=train.forward(model,ds,ds.valid,**kwargs)
    grads=torch.autograd.grad(corrected.sum(),(c,r),allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(g.abs().sum()>0 for g in grads)
    off=train.forward(model,ds,ds.valid,'off',**kwargs)
    gg=torch.autograd.grad(off.sum(),(c,r),allow_unused=True)
    assert gg==(None,None)
    assert torch.equal(grads[0][ds.fit],torch.zeros_like(grads[0][ds.fit]))


def test_fresh_exact_outer_epoch_does_not_read_test_outcomes(monkeypatch):
    monkeypatch.setattr(train,'cfg',config)
    monkeypatch.setattr(train,'source',lambda:'synthetic')
    monkeypatch.setitem(sys.modules,'src.phase20_selective.diagnostics',SimpleNamespace(inspect_model=lambda *args:{}))
    with pytest.raises(ValueError,match='locked inner epoch'):
        train.train_cell('P16',31,1,'outer',dataset=synthetic('outer',dtype=torch.float32))
    init=[];original=train.make_model
    def maker(ds,arm,w=None):
        m=original(ds,arm,w);init.append({k:v.detach().clone() for k,v in m.state_dict().items()});return m
    monkeypatch.setattr(train,'make_model',maker)
    inner=train.train_cell('P16',31,1,'inner',epochs=2,dataset=synthetic(dtype=torch.float32))
    d=synthetic('outer',dtype=torch.float32);d.ep['time'][d.valid]=np.nan;d.ep['event'][d.valid]=np.nan
    outer=train.train_cell('P16',31,1,'outer',epochs=3,dataset=d)
    assert all(torch.equal(init[0][k],init[1][k]) for k in init[0])
    assert outer['selected_epoch']==3 and len(outer['history'])==3
    assert all('validation_loss' not in q for q in outer['history'])
    assert np.array_equal(inner['anchor_w'],outer['anchor_w'])
    assert not any(name=='main' for name,_ in original(d,'P16').named_parameters())


def test_full_omics_and_uniform_are_distinct_from_correction_only_controls(monkeypatch):
    monkeypatch.setattr(train,'cfg',config)
    ds=synthetic(dtype=torch.float32)
    model=SelectiveModel(3,ds.f['membership'],ds.anchor_w,ds.f['C'][ds.fit].mean(0),ds.anchor_sd_fit,'P16',config()).eval()
    a=train.forward(model,ds,ds.valid)
    torch.testing.assert_close(a,train.forward(model,ds,ds.valid,'wrongO'),atol=0,rtol=0)
    assert not torch.equal(a,train.forward(model,ds,ds.valid,'fullOmicsWrong'))
    assert not torch.equal(a,train.forward(model,ds,ds.valid,'uniformPair'))


def test_patient_scalar_aggregation_is_equal_wsi_not_patch_weighted(monkeypatch):
    monkeypatch.setattr(train,'cfg',config)
    ds=synthetic()
    class Scalar(torch.nn.Module):
        def forward(self,visual,*args,**kwargs): return visual[:,0,0]
    expected=torch.tensor([ds.visual[ds.owner==p,0,0].mean() for p in [3,0,5,1]])
    actual=train.forward(Scalar(),ds,np.array([3,0,5,1]))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_anchor_audit_preserves_three_wsi_float32_bits_after_model_validation(monkeypatch,device):
    monkeypatch.setattr(train,'cfg',config)
    ds=synthetic(dtype=torch.float32,device=device)
    ds.owner=np.repeat(ds.valid,3)
    ds.visual=ds.visual[:6].clone();ds.mask=ds.mask[:6].clone()
    ds.ft['ANCHOR'][ds.valid]=torch.tensor([.123456789,.8],device=device)
    expected=ds.ft['ANCHOR'][ds.valid]
    legacy=torch.stack([value.expand(3).mean() for value in expected])
    # A real float32 regression: the former mean rule can change the exact
    # offset despite every WSI returning the same correct stored value.
    assert not torch.equal(legacy,expected)
    class AnchorOutput(torch.nn.Module):
        calls=0
        def forward(self,*args,anchor_override=None,main_only=False,**kwargs):
            assert main_only
            self.calls+=1
            return anchor_override.clone()
    model=AnchorOutput()
    actual=train.forward(model,ds,ds.valid,'anchor')
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert model.calls==1


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_anchor_audit_rejects_corruption_of_a_nonfirst_wsi(monkeypatch,device):
    monkeypatch.setattr(train,'cfg',config)
    ds=synthetic(dtype=torch.float32,device=device)
    ds.owner=np.repeat(ds.valid,3)
    ds.visual=ds.visual[:6].clone();ds.mask=ds.mask[:6].clone()
    class CorruptAnchorOutput(torch.nn.Module):
        def forward(self,*args,anchor_override=None,main_only=False,**kwargs):
            assert main_only
            result=anchor_override.clone()
            result[1]=torch.nextafter(result[1],torch.full_like(result[1],torch.inf))
            return result
    with pytest.raises(ValueError,match='per-WSI frozen anchor output identity'):
        train.forward(CorruptAnchorOutput(),ds,ds.valid,'anchor')

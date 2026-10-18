"""Integration checks for complete-risk gradients and patient aggregation."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.phase25_local.train import _forward,train_arm
from src.phase25_local.model import LocalRouting
from src.phase17_ssd.losses import full_risk_efron


def data(nfit=35,nvalid=4):
    rng=np.random.default_rng(15);n=nfit+nvalid
    owner=np.r_[np.arange(n),0,0,nfit]
    x=torch.tensor(rng.normal(size=(len(owner),5,768)),dtype=torch.float32)
    x=x/x.norm(dim=-1,keepdim=True)
    ds=SimpleNamespace(fit=np.arange(nfit),valid=np.arange(nfit,n),owner=owner,
        visual=x,mask=torch.ones(len(owner),5,dtype=torch.bool),R=torch.tensor(rng.normal(size=(n,512)),dtype=torch.float32),
        C=torch.tensor(rng.normal(size=(n,3)),dtype=torch.float32),base=torch.tensor(rng.normal(size=n),dtype=torch.float64),
        scale=.7,wrong_rna=np.r_[np.roll(np.arange(nfit),1),np.roll(np.arange(nfit,n),1)],
        wrong_c=np.r_[np.roll(np.arange(nfit),-1),np.roll(np.arange(nfit,n),-1)])
    ds.fit_labels=lambda:(np.arange(nfit,dtype=float)+1,(np.arange(nfit)%4==0).astype(float))
    ds.R0_valid=ds.base[ds.valid].numpy().copy()
    return ds


def test_patient_mean_anchor_once_and_frozen_off():
    ds=data(4,2)
    class ConstantWSI:
        def __call__(self,p,m,r,c,**kw):return {'unit':p[:,0,0]}
    risk,u,_=_forward(ConstantWSI(),ds,np.r_[ds.fit,ds.valid])
    expected=np.array([ds.visual[ds.owner==i,0,0].double().mean() for i in range(6)])
    np.testing.assert_allclose(u.numpy(),expected,rtol=0,atol=0)
    np.testing.assert_allclose(risk.numpy(),ds.base.numpy()+ds.scale*expected,rtol=0,atol=0)
    off,_,_=_forward(ConstantWSI(),ds,ds.valid,intervention='OFF')
    np.testing.assert_array_equal(off.numpy(),ds.R0_valid)
    with pytest.raises(ValueError,match='heldout'):_forward(ConstantWSI(),ds,ds.fit,intervention='OFF')


def test_chunked_checkpoint_gradient_matches_whole_patient_risk():
    ds=data();torch.manual_seed(1);a=LocalRouting(3,'R3');b=LocalRouting(3,'R3');b.load_state_dict(a.state_dict())
    t,e=ds.fit_labels();t=torch.tensor(t);e=torch.tensor(e)
    r,u,_=_forward(a,ds,ds.fit,training=True)
    loss=full_risk_efron(r,t,e)+.5*u.square().mean();loss.backward()
    slides=np.flatnonzero(np.isin(ds.owner,ds.fit));own=ds.owner[slides]
    wsi=b(ds.visual[slides],ds.mask[slides],ds.R[own],ds.C[own])['unit']
    direct=torch.stack([wsi[torch.tensor(own==p)].double().mean() for p in ds.fit])
    ref=full_risk_efron(ds.base[ds.fit]+ds.scale*direct,t,e)+.5*direct.square().mean();ref.backward()
    torch.testing.assert_close(loss,ref,rtol=1e-7,atol=1e-9)
    for (_,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-4,atol=2e-7)


def test_nonfit_covariates_and_offsets_cannot_change_training_state():
    ds=data(7,4);other=data(7,4)
    other.R[other.valid]=1000;other.C[other.valid]=-1000;other.base[other.valid]=1000
    first,_,h1,a1=train_arm(ds,'R3',31,1,epochs=2)
    second,_,h2,a2=train_arm(other,'R3',31,1,epochs=2)
    for k,v in first.state_dict().items():assert torch.equal(v,second.state_dict()[k])
    assert [(h['training_loss'],h['fit_loss']) for h in h1]==[(h['training_loss'],h['fit_loss']) for h in h2]
    assert a1['heldout_endpoint_reads']==a2['heldout_endpoint_reads']==0


def test_frozen_query_intervention_keeps_own_main():
    ds=data(6,4);torch.manual_seed(8);model=LocalRouting(3,'R3')
    with torch.no_grad():model.rna.weight.normal_(0,.02);model.clinical.weight.normal_(0,.02)
    a,u,_=_forward(model,ds,ds.valid)
    b,v,_=_forward(model,ds,ds.valid,intervention='WRONG_RNA')
    torch.testing.assert_close(a-ds.scale*u,ds.base[ds.valid],atol=1e-12,rtol=0)
    torch.testing.assert_close(b-ds.scale*v,ds.base[ds.valid],atol=1e-12,rtol=0)
    assert not torch.equal(u,v)


@pytest.mark.parametrize('arm',['R1','R2','R3'])
def test_short_fit_finite_complete_history(arm):
    ds=data(6,3);m,initial,h,a=train_arm(ds,arm,47,2,epochs=2)
    assert len(h)==2 and a['optimizer_steps']==2 and a['fixed_final_epoch']
    assert a['fit_offset_kind']=='strict_threefold_crossfit_R0'
    assert all(np.isfinite(x['fit_loss']) for x in h)
    assert any(not torch.equal(v,initial[k]) for k,v in m.state_dict().items())

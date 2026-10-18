from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.phase26_multiquery import train
from src.phase26_multiquery.model import MultiQuery
from src.phase17_ssd.losses import full_risk_efron


def dataset(n=12):
    rng=np.random.default_rng(26);fit=np.arange(n-4);valid=np.arange(n-4,n)
    labels=(np.arange(1,len(fit)+1,dtype=float),np.array([int(i%2==0) for i in fit],float))
    ds=SimpleNamespace(visual=torch.tensor(rng.normal(size=(n,5,768)),dtype=torch.float32),
        mask=torch.ones(n,5,dtype=torch.bool),owner=np.arange(n),fit=fit,valid=valid,
        R=torch.tensor(rng.normal(size=(n,512)),dtype=torch.float32),
        C=torch.tensor(rng.normal(size=(n,4)),dtype=torch.float32),
        base=torch.tensor(rng.normal(size=n)*.3,dtype=torch.float64),
        wrong_rna=np.roll(np.arange(n),1),wrong_c=np.roll(np.arange(n),1))
    ds.R0_valid=ds.base[valid].numpy().copy();ds.fit_labels=lambda:tuple(x.copy() for x in labels)
    ds.scale=float(ds.base[fit].numpy().std())
    ds.wrong_rna[valid]=np.roll(valid,1);ds.wrong_c[valid]=np.roll(valid,1)
    return ds


def test_zero_head_initializes_exact_main_and_hard_bound():
    ds=dataset();train.seed_all(31,1)
    for arm in ('QUERY','ATTN','COMP'):
        model=MultiQuery(4,arm).eval()
        with torch.no_grad():
            r,u,_=train._forward(model,ds,ds.valid)
            assert torch.equal(r,ds.base[ds.valid]) and not u.any()
            model.head[-1].weight.fill_(1e4)
            r,u,_=train._forward(model,ds,ds.valid)
            assert (r-ds.base[ds.valid]).abs().max()<=.100000000000001


def test_complete_riskset_checkpoint_gradient_matches_plain_two_microbatches():
    ds=dataset(44);train.seed_all(31,1);model=MultiQuery(4,'ATTN')
    with torch.no_grad():model.head[-1].weight.fill_(.01)
    t,e=ds.fit_labels();grads=[]
    for checkpointed in (False,True):
        model.zero_grad(set_to_none=True)
        r,u,_=train._forward(model,ds,ds.fit,training=checkpointed)
        loss=full_risk_efron(r,t,e)+.005*u.square().mean();loss.backward()
        grads.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys()==grads[1].keys()
    for k in grads[0]:torch.testing.assert_close(grads[0][k],grads[1][k],rtol=1e-6,atol=1e-8)


def test_two_steps_activate_upstream_and_preserve_frozen_main():
    ds=dataset();saved=ds.base.clone()
    model,initial,history,audit,counts=train.train_arm(ds,'COMP',31,1,epochs=2)
    assert torch.equal(saved,ds.base) and len(history)==2
    assert history[0]['gradients']['rna']==0
    assert history[1]['gradients']['rna']>0 and history[1]['gradients']['value']>0
    assert audit['heldout_endpoint_reads']==0 and counts['allocated_trainable']>0


def test_frozen_interventions_and_public_mechanism_schema():
    ds=dataset();train.seed_all(31,1);model=MultiQuery(4,'QUERY').eval()
    with torch.no_grad():model.head[-1].weight.fill_(.01)
    record=train.predict_arm(model,ds)
    for name in train.CONTROLS:
        if name!='OFF':np.testing.assert_array_equal(record['risk'],record['sensitivities'][name])
    np.testing.assert_array_equal(record['sensitivities']['OFF'],ds.R0_valid)
    with pytest.raises(ValueError,match='valid-only'):
        train._forward(model,ds,ds.fit,intervention='VALUE_OFF')
    from src.phase26_multiquery.evaluation import mechanism_summary
    mechanism_summary(record['mechanism'],len(ds.fit),len(ds.valid))

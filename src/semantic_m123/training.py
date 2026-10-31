"""Deterministic, full-risk-set Efron training; no validation labels enter updates."""
from __future__ import annotations
import math,time
import numpy as np
import torch


def efron_log_weights(time,event,*,device='cpu'):
    t=np.asarray(time,np.float64);e=np.asarray(event)
    if t.ndim!=1 or e.shape!=t.shape or not np.isfinite(t).all() or (t<0).any() or not np.isin(e,[0,1]).all() or e.sum()<1:
        raise ValueError('CURRENT_TRAINING_ENDPOINTS_REQUIRED')
    rows=[]
    for at in np.unique(t[e==1]):
        died=(t==at)&(e==1);count=int(died.sum())
        for ell in range(count):
            row=np.full(len(t),-np.inf);row[t>=at]=0.
            row[died]=math.log1p(-ell/count)
            rows.append(row)
    return torch.as_tensor(np.stack(rows),dtype=torch.float64,device=device),torch.as_tensor(e,dtype=torch.bool,device=device)


def efron(risk,log_weights,event):
    if risk.ndim!=1 or risk.shape!=event.shape or log_weights.shape!=(int(event.sum()),len(risk)):
        raise ValueError('FULL_TRAINING_RISK_SET_SHAPE_MISMATCH')
    return (torch.logsumexp(log_weights+risk.double()[None,:],dim=1).sum()-risk.double()[event].sum())/event.sum()


def subset(batch,rows,device):
    ix=torch.as_tensor(rows,dtype=torch.long)
    return {k:v[ix].to(device) for k,v in batch.items()}


def fit(model,train_batch,train_time,train_event,*,seed,l2,max_epochs,head_epochs=20,
        encoder_lr=1e-4,head_lr=3e-4,snapshot_epochs=(0,20,50,100,200,400),
        evaluation_batch=None,snapshot_callback=None,heartbeat_callback=None):
    """All labels in this signature belong to fit-train; no test metric selection.

    The caller may record held-out predictions without backpropagating them.
    No dropout, mini-batch Cox, risk-offset stacking or stochastic sampling.
    """
    if max_epochs<0 or l2<=0:raise ValueError('UNREGISTERED_TRAINING_BUDGET')
    device=next(model.parameters()).device
    logw,event=efron_log_weights(train_time,train_event,device=device)
    if len(train_time)!=len(train_batch['P']):raise ValueError('TRAINING_ROWS_NOT_FULL_RISK_SET')
    heads=model.head_parameters();head_ids={id(p) for p in heads}
    other=[p for p in model.parameters() if id(p) not in head_ids]
    optimizer=torch.optim.Adam([{'params':heads,'lr':head_lr},{'params':other,'lr':encoder_lr}])
    history=[];start=time.monotonic()
    def snapshot(epoch):
        if snapshot_callback is None:return
        model.eval()
        with torch.no_grad():
            tr={k:v.detach().cpu().numpy() for k,v in model(train_batch).items()}
            ev={k:v.detach().cpu().numpy() for k,v in model(evaluation_batch).items()} if evaluation_batch is not None else {}
        snapshot_callback(epoch,tr,ev)
    if 0 in snapshot_epochs:snapshot(0)
    for epoch in range(1,max_epochs+1):
        model.set_stage(epoch<=head_epochs);model.train();optimizer.zero_grad(set_to_none=True)
        out=model(train_batch);loss_cox=efron(out['risk'],logw,event)
        penalty=l2*model.regularizer();loss=loss_cox+penalty
        if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_EFRON_OR_L2')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        row={'epoch':epoch,'stage':'heads' if epoch<=head_epochs else 'joint','Cox':float(loss_cox.detach()),
            'L2':float(penalty.detach()),'objective':float(loss.detach()),'gradient_norm_before_clip':float(norm),
            'risk_sd':float(out['risk'].detach().std(unbiased=False)),
            'interaction_risk_sd':float(out['interaction_risk'].detach().std(unbiased=False))}
        for group in ('patch','program','query_projection','attention','interaction_head'):
            pp=[p.grad for name,p in model.named_parameters() if name.startswith(group+'.') and p.grad is not None]
            row[group+'_gradient_norm_with_regularizer']=float(torch.sqrt(sum(x.square().sum() for x in pp))) if pp else 0.
        optimizer.step();history.append(row)
        if epoch in snapshot_epochs:snapshot(epoch)
        if heartbeat_callback is not None and (epoch%10==0 or epoch==max_epochs):
            heartbeat_callback(epoch,row)
    model.set_stage(False);model.eval()
    return {'history':history,'optimizer_updates':max_epochs,'elapsed_seconds':time.monotonic()-start,
            'full_risk_set_patients':len(train_time),'DSS_in_encoder_updates':True,
            'validation_endpoints_available_to_optimizer':False,'seed':seed,'L2_lambda':l2}


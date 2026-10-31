"""The user's literal sum penalties and activated full-risk-set schedule."""
import math,time
import numpy as np
import torch
from src.semantic_m123.training import efron_log_weights,efron

SNAPSHOTS=(0,10,30,50,90,140)


def center(x):return x-x.mean()


def function_penalty(risk,reference):
    if risk.shape!=reference.shape or risk.ndim!=1:raise ValueError('CURRENT_TRAINING_REFERENCE_SHAPE')
    return (center(risk)-center(reference)).square().sum()


def a_multiplier(epoch):return 1. if epoch<=90 else 1.-.9*(epoch-90)/50.


def parameter_group(name):
    if name.startswith('base.'):return 'base'
    if name.startswith(('local_out.','interaction_head.')):return 'output'
    if name.startswith('patch.'):return 'patch'
    if name=='mil_query':return 'mil_query'
    if name.startswith('mil_attention.'):return 'mil_attention'
    if name.startswith(('program.','query_projection.')) or name=='identity':return 'molecular_query'
    if name.startswith('cross_attention.'):return 'cross_attention'
    return 'other_new'


def fit(model,train,time,event,reference,config,*,snapshot_callback=None,heartbeat=None,max_epochs=140):
    if max_epochs not in SNAPSHOTS:raise ValueError('LOCKED_SCHEDULE_CHECKPOINT')
    device=next(model.parameters()).device
    lw,ev=efron_log_weights(time,event,device=device)
    ref=torch.as_tensor(reference,dtype=torch.float64,device=device).detach()
    if len(ref)!=len(train['P']):raise ValueError('TRAIN_ONLY_FUNCTION_ANCHOR')
    named=list(model.named_parameters());groups={k:[p for n,p in named if parameter_group(n)==k] for k in sorted({parameter_group(n) for n,p in named})}
    optim=torch.optim.Adam([{'params':[p for n,p in named if n.startswith('base.')],'lr':1e-5},
        {'params':[p for n,p in named if not n.startswith('base.') and parameter_group(n)=='output'],'lr':3e-4},
        {'params':[p for n,p in named if not n.startswith('base.') and parameter_group(n)!='output'],'lr':1e-4}])
    counts={g:0 for g in groups};max_cox_gradient={g:0. for g in groups};history=[];started=time_module()
    def snapshot(epoch):
        if snapshot_callback is not None:
            model.eval();snapshot_callback(epoch,dict(counts),dict(max_cox_gradient))
    snapshot(0)
    for epoch in range(1,max_epochs+1):
        stage=model.stage(epoch);model.train();optim.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in named if p.requires_grad}
        out=model(train);lc=efron(out['risk'],lw,ev);pb,pn=model.penalties();fa=function_penalty(out['risk'],ref)
        la=config['a']*a_multiplier(epoch)
        loss=lc+la*fa+config['b']*pb+config['n']*pn
        if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_REPAIR_LOSS')
        probe={}
        if epoch in (1,10,11,30,90,91,140):
            selected=[(n,p) for n,p in named if p.requires_grad]
            gg=torch.autograd.grad(lc,[p for n,p in selected],retain_graph=True,allow_unused=True)
            for group in groups:
                vals=[g.double().square().sum() for (n,p),g in zip(selected,gg) if parameter_group(n)==group and g is not None]
                probe['Cox_only_gradient_'+group]=float(torch.stack(vals).sum().sqrt()) if vals else 0.
                max_cox_gradient[group]=max(max_cox_gradient[group],probe['Cox_only_gradient_'+group])
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optim.step()
        delta=[]
        for group in groups:
            vals=[(p.detach()-before[n]).double().square().sum() for n,p in named if n in before and parameter_group(n)==group]
            delta.append(torch.stack(vals).sum() if vals else loss.detach().new_zeros(()))
        changed=torch.stack(delta).detach().cpu().numpy()
        for group,value in zip(groups,changed):counts[group]+=int(value>0)
        row={'epoch':epoch,'stage':stage,'Cox':float(lc.detach()),'function_sum':float(fa.detach()),
            'base_proximal_sum':float(pb.detach()),'new_square_sum':float(pn.detach()),'effective_a':la,
            'objective':float(loss.detach()),'gradient_norm_before_clip':float(norm),
            'train_risk_sd':float(out['risk'].detach().std(unbiased=False)),
            'train_base_drift_sd':float((out['base_risk'].detach()-ref).std(unbiased=False)),**probe}
        history.append(row)
        if epoch in SNAPSHOTS:snapshot(epoch)
        if heartbeat is not None and (epoch%10==0 or epoch==max_epochs):heartbeat(epoch,stage)
    model.eval()
    return {'history':history,'actual_parameter_update_counts':counts,'optimizer_updates':max_epochs,
        'elapsed_seconds':time_module()-started,'validation_labels_in_optimizer':False,
        'full_risk_set_patients':len(ref),'penalties_are_literal_sums_no_variance_or_sample_normalization':True}


def time_module():return __import__('time').monotonic()

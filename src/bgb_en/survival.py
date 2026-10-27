"""Full-risk Efron objectives, honest scalar fusion and pair repair diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
import torch
from src.phase10_trimodal.math import efron_weights,torch_cox,log_weights
from src.phase10_trimodal.reference import components,uno_weights,pairs

def fit_offset_ridge(x,t,e,lam=1.,offset=None,maxiter=2000):
    """L-BFGS only: never allocate a G-by-G Hessian for the full-RNA control."""
    x=np.asarray(x,float);t=np.asarray(t,float);e=np.asarray(e,int)
    if offset is None:offset=np.zeros(len(t))
    offset=np.asarray(offset,float)
    a=efron_weights(t,e)
    with np.errstate(divide='ignore'):la=np.log(a)
    def fun(w):
        risk=offset+x@w;scores=la+risk[None,:];den=logsumexp(scores,axis=1)
        dr=(np.exp(scores-den[:,None]).sum(0)-e)/e.sum()
        return float((den.sum()-risk[e==1].sum())/e.sum()+lam/2*(w@w)),x.T@dr+lam*w
    opt=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=maxiter,ftol=1e-12,gtol=1e-7,maxls=50))
    value,grad=fun(opt.x)
    if not np.isfinite(value) or not np.isfinite(opt.x).all() or np.max(np.abs(grad))>2e-5:
        raise RuntimeError('Cox optimizer did not meet gradient contract')
    return opt.x,dict(loss=value,gradient_max=float(np.max(np.abs(grad))),iterations=int(opt.nit),
                      solver='L-BFGS-B',success=bool(opt.success),parameters=x.shape[1])

def standardized_risk(train_risk,all_risk):
    mean=float(np.mean(train_risk));sd=float(np.std(train_risk))
    if not np.isfinite(sd) or sd<=1e-8:raise ValueError('Degenerate teacher risk scale')
    return (all_risk-mean)/sd,dict(mean=mean,sd=sd)

def fit_scalar_same_teacher(cp,rna,t,e,teacher_group,lam=1.):
    left=[];right=[]
    for group in np.unique(teacher_group):
        idx=np.flatnonzero(teacher_group==group);a,b=pairs(t[idx],e[idx]);left.extend(idx[a]);right.extend(idx[b])
    i,j=np.asarray(left,int),np.asarray(right,int)
    if not len(i):raise ValueError('No same-teacher heldout comparable pairs')
    margin=cp[i]-cp[j];difference=rna[i]-rna[j]
    def fun(beta):
        v=margin+beta[0]*difference
        return float(np.logaddexp(0.,-v).mean()+lam/2*beta[0]**2),np.array([np.mean(-expit(-v)*difference)+lam*beta[0]])
    result=minimize(fun,np.zeros(1),jac=True,method='L-BFGS-B',options=dict(gtol=1e-9,ftol=1e-14,maxiter=1000))
    value,grad=fun(result.x)
    if not np.isfinite(value) or abs(grad[0])>1e-6:raise ValueError('Scalar calibration did not converge')
    return float(result.x[0]),dict(loss=value,gradient_max=float(abs(grad[0])),pairs=len(i),lambda_beta=lam)

def evaluate(train_time,train_event,time,event,risk,reference=None):
    w=uno_weights(train_time,train_event,time,event,tau=60.)
    hn,hd=components(time,event,risk);un,ud=components(time,event,risk,w)
    out=dict(Harrell=hn/hd if hd else None,Uno60=un/ud if ud else None,
             Harrell_num=hn,Harrell_den=hd,Uno_num=un,Uno_den=ud)
    if reference is not None:
        i,j=pairs(time,event)
        def credit(x):
            delta=np.asarray(x)[i]-np.asarray(x)[j]
            return np.where(np.abs(delta)<=1e-8,.5,(delta>0).astype(float))
        d=credit(risk)-credit(reference)
        out['repair']=float(np.maximum(d,0).sum()/hd) if hd else None
        out['damage']=float(np.maximum(-d,0).sum()/hd) if hd else None
        out['Uno_repair']=float(np.dot(w[i],np.maximum(d,0))/ud) if ud else None
        out['Uno_damage']=float(np.dot(w[i],np.maximum(-d,0))/ud) if ud else None
        out['risk_shift_sd_ratio']=float(np.std(np.asarray(risk)-reference)/max(np.std(reference),1e-12))
    return out

def combine_metrics(rows):
    out={k:float(sum(r[k] for r in rows)) for k in ('Harrell_num','Harrell_den','Uno_num','Uno_den')}
    out['Harrell']=out['Harrell_num']/out['Harrell_den'] if out['Harrell_den'] else None
    out['Uno60']=out['Uno_num']/out['Uno_den'] if out['Uno_den'] else None
    return out

def cached_full_risk_step(model,batches,time,event,optimizer,device,extra_weight=.1):
    """Exact deterministic two-pass full-risk gradient; batches are index tensors.

    No dropout or stochastic feature sampling is permitted in this path. Micro-
    batch Cox losses are never substituted for the full training risk set.
    """
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():risk=torch.cat([model(idx)['risk'] for idx in batches])
    leaf=risk.detach().requires_grad_(True)
    loss=torch_cox(leaf,log_weights(time,event,device),torch.as_tensor(event,device=device))
    grad=torch.autograd.grad(loss,leaf)[0].detach()
    position=0;maximum=0.
    for idx in batches:
        output=model(idx);pred=output['risk'];n=len(pred)
        maximum=max(maximum,float((pred.detach()-risk[position:position+n]).abs().max()))
        if maximum>2e-5:raise RuntimeError('Gradient-cache forward was not deterministic')
        # This matches the exact additive risk-magnitude penalty over ALL rows.
        penalty=extra_weight*output['delta'].square().sum()/len(risk)
        (torch.dot(pred,grad[position:position+n])+penalty).backward();position+=n
    regularizer=model.regularization()
    regularizer.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
    optimizer.step()
    return dict(cox_loss=float(loss.detach()),regularizer=float(regularizer.detach()),
                gradient_norm=norm,replay_max=maximum)

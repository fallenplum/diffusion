"""Training math. Evaluation lives in a separate, independent module."""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp


def efron_weights(time, event):
    t, e = np.asarray(time,float), np.asarray(event,int)
    if t.ndim != 1 or e.shape != t.shape or not np.isfinite(t).all() or not np.isin(e,[0,1]).all() or e.sum()==0:
        raise ValueError('invalid survival input or zero events')
    rows=[]
    for u in np.unique(t[e==1]):
        deaths=(t==u)&(e==1); d=deaths.sum()
        for j in range(d): rows.append((t>=u).astype(float)-j/d*deaths)
    return np.stack(rows)


def log_weights(t,e,device):
    a=efron_weights(t,e)
    with np.errstate(divide='ignore'): a=np.log(a)
    return torch.tensor(a,dtype=torch.float64,device=device)


def torch_cox(risk, loga, event):
    r=risk.double(); e=event.bool()
    return (torch.logsumexp(loga+r[None,:],dim=1).sum()-r[e].sum())/e.sum()


def fit_ridge(x,t,e,lam):
    x=np.asarray(x,float); e=np.asarray(e,int); a=efron_weights(t,e)
    with np.errstate(divide='ignore'): la=np.log(a)
    def obj(w):
        r=x@w; scores=la+r[None,:]; ld=logsumexp(scores,axis=1)
        proba=np.exp(scores-ld[:,None]); dr=(proba.sum(0)-e)/e.sum()
        return float((ld.sum()-r[e==1].sum())/e.sum()+lam/2*(w@w)),x.T@dr+lam*w
    fit=minimize(obj,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7,'maxls':50})
    w=fit.x.copy();loss,grad=obj(w);polish=0
    # PCA axes have strongly different variances. L-BFGS's relative objective
    # stopping can trigger while the absolute gradient is still too large.
    # Exact Newton refinement optimizes the SAME objective and lambda.
    while np.abs(grad).max()>1e-7 and polish<30:
        scores=la+(x@w)[None,:];ld=logsumexp(scores,axis=1);prob=np.exp(scores-ld[:,None]);ex=prob@x
        h=(x.T@(x*prob.sum(0)[:,None])-ex.T@ex)/e.sum()+lam*np.eye(x.shape[1])
        direction=np.linalg.solve(h,grad);step=1.
        for _ in range(40):
            candidate=w-step*direction;newloss,newgrad=obj(candidate)
            if newloss<=loss-1e-4*step*(grad@direction)+1e-14:break
            step*=.5
        w=candidate;loss,grad=newloss,newgrad;polish+=1
    if not np.isfinite(loss) or np.abs(grad).max()>2e-5: raise RuntimeError('Ridge did not converge: '+str(fit.message))
    return w,{'loss':loss,'gradient_max':float(np.abs(grad).max()),'iterations':int(fit.nit),'newton_polish_steps':polish}


def comparable_pairs(t,e):
    # Risk ranking distillation includes event/censor ties, excludes event/event ties.
    t=np.asarray(t);e=np.asarray(e)
    return np.where((e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0))))

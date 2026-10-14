"""Cox fitting only; independent evaluation lives in a separate script."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from src.phase10_trimodal.math import efron_weights,fit_ridge


def block_fit(x,t,e,penalty):
    x=np.asarray(x,float);e=np.asarray(e,int);penalty=np.asarray(penalty,float)
    if penalty.shape!=(x.shape[1],) or (penalty<=0).any():raise ValueError('nonpositive block penalty')
    a=efron_weights(t,e)
    with np.errstate(divide='ignore'):loga=np.log(a)
    def objective(w):
        r=x@w;scores=loga+r;ld=logsumexp(scores,axis=1);p=np.exp(scores-ld[:,None]);g=x.T@((p.sum(0)-e)/e.sum())+penalty*w
        return float((ld.sum()-r[e==1].sum())/e.sum()+.5*np.sum(penalty*w*w)),g
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7,'maxls':50})
    w=result.x.copy();loss,gradient=objective(w);polish=0
    while abs(gradient).max()>1e-7 and polish<30:
        scores=loga+x@w;ld=logsumexp(scores,axis=1);p=np.exp(scores-ld[:,None]);ex=p@x
        h=(x.T@(x*p.sum(0)[:,None])-ex.T@ex)/e.sum()+np.diag(penalty);direction=np.linalg.solve(h,gradient);step=1.
        for _ in range(40):
            candidate=w-step*direction;nl,ng=objective(candidate)
            if nl<=loss-1e-4*step*(gradient@direction)+1e-14:break
            step*=.5
        w=candidate;loss,gradient=nl,ng;polish+=1
    if not np.isfinite(loss) or abs(gradient).max()>2e-5:raise RuntimeError('block Cox did not converge')
    return w,{'loss':loss,'gradient_max':float(abs(gradient).max()),'iterations':int(result.nit),'newton_polish_steps':polish}


def inner_score(t,e,r):
    a,b=np.where((e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0))))
    if not len(a):raise ValueError('no inner comparable pairs')
    delta=r[a]-r[b];return float(np.where(abs(delta)<=1e-8,.5,delta>0).mean())


def choose(xa,xv,t,e,tv,ev,grid):
    rows=[];best=None
    for lam in grid:
        w,diagnostic=fit_ridge(xa,t,e,lam);score=inner_score(tv,ev,xv@w);rows.append({'lambda':lam,'inner_Harrell':score,**diagnostic})
        if best is None or score>best[0]+1e-12:best=(score,lam)
    return best[1],rows


def choose_block(a,r,av,rv,t,e,tv,ev,base_lambda,grid):
    baseline,bd=fit_ridge(a,t,e,base_lambda);rows=[];best=None
    for lam in grid:
        if lam=='OFF':risk=av@baseline;diagnostic=bd
        else:
            w,diagnostic=block_fit(np.c_[a,r],t,e,np.r_[np.full(a.shape[1],base_lambda),np.full(r.shape[1],lam)])
            risk=np.c_[av,rv]@w
        score=inner_score(tv,ev,risk);rows.append({'RNA_lambda':lam,'inner_Harrell':score,**diagnostic})
        if best is None or score>best[0]+1e-12 or (lam=='OFF' and abs(score-best[0])<=1e-12):best=(score,lam)
    return best[1],rows


def linear_map(x,y,lam):
    return np.linalg.solve(x.T@x/len(x)+lam*np.eye(x.shape[1]),x.T@y/len(x))

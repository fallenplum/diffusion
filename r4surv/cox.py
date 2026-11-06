"""与diffusion R4一致的Efron-ridge、teacher折内RNA风险标量融合。"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from .reference import efron_weights, pairs


def ridge_cox(x, t, e, lam=10., maxiter=2000):
    x, t, e = np.asarray(x, float), np.asarray(t, float), np.asarray(e, int)
    a = efron_weights(t, e)
    with np.errstate(divide='ignore'): la = np.log(a)
    def objective(w):
        risk = x @ w; scores = la + risk[None, :]
        den = logsumexp(scores, axis=1)
        dr = (np.exp(scores - den[:, None]).sum(0) - e) / e.sum()
        loss = (den.sum()-risk[e == 1].sum()) / e.sum() + lam / 2 * (w @ w)
        return float(loss), x.T @ dr + lam * w
    ans = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-7, maxls=50))
    loss, grad = objective(ans.x)
    if not np.isfinite(loss) or np.max(np.abs(grad)) > 2e-5:
        raise RuntimeError(f'Cox优化没有达到收敛合同: gradient_max={np.max(np.abs(grad))}')
    return ans.x, {'loss': loss, 'iterations': int(ans.nit), 'gradient_max': float(np.abs(grad).max()), 'lambda': lam}


def scale_risk(train_risk, all_risk):
    mean = float(train_risk.mean()); sd = float(train_risk.std())
    if sd <= 1e-8: raise ValueError('退化风险尺度')
    return (all_risk - mean)/sd, {'mean':mean, 'sd':sd}


def scalar_weight(cp_cf, rna_cf, t, e, groups, lam=1.):
    left, right = [], []
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group); a,b = pairs(t[ix],e[ix])
        left.extend(ix[a]);right.extend(ix[b])
    i,j = np.asarray(left,int),np.asarray(right,int)
    if not len(i): raise ValueError('交叉拟合teacher组内无可比较患者对')
    margin = cp_cf[i]-cp_cf[j]; diff = rna_cf[i]-rna_cf[j]
    def fun(beta):
        risk = margin + beta[0]*diff
        return float(np.logaddexp(0.,-risk).mean()+lam/2*beta[0]**2), \
               np.array([np.mean(-expit(-risk)*diff)+lam*beta[0]])
    opt = minimize(fun,np.zeros(1),jac=True,method='L-BFGS-B',options=dict(gtol=1e-9,ftol=1e-14,maxiter=1000))
    val,grad = fun(opt.x)
    if not np.isfinite(val) or abs(grad[0]) > 1e-6:raise RuntimeError('RNA标量校准不收敛')
    return float(opt.x[0]), {'pairs':len(i),'loss':val,'gradient_max':float(abs(grad[0]))}

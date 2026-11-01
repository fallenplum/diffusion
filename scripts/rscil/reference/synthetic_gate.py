"""Three falsifiable synthetic cases; NO TCGA labels or hidden patient data.

Usage: python synthetic_gate.py --reps 12 --out synthetic_results.json
Builds true independent interaction, redundant interaction, and no interaction.
Does not assert the proposed deep multimodal architecture is validated.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from riskset_core import orthogonalize_interaction, efron_nll


def simulate(seed, scenario, n=512, events_target=65):
    rg = np.random.default_rng(seed)
    c, ep, eg = rg.normal(size=(3, n))
    p = 0.75 * c + ep
    g = 0.75 * c + eg
    B = np.stack([c, p, g], axis=1)
    # The interaction is genuinely conditional on both p and g.
    z = (p - 0.75*c) * (g - 0.75*c)
    if scenario == 'redundant':
        z = 2.0*p - 0.4*g + 0.7*c
    gamma = 0.0 if scenario in ('null', 'redundant') else 0.85
    truth = 1.0*c + 0.25*ep + 0.25*eg + gamma*((p-0.75*c)*(g-0.75*c))
    # Event rate controlled by independent censoring uniforms. The resulting
    # administrative scale is chosen once for the synthetic cohort.
    u = rg.uniform(size=n)
    t_event = -np.log(u) / np.exp(np.clip(truth, -7, 7))
    t_cens0 = -np.log(rg.uniform(size=n))
    ratio = t_cens0 / t_event
    censor_scale = np.partition(ratio, events_target-1)[events_target-1]
    # exactly the target number of observed failures, barring floating ties
    t_cens = t_cens0 / censor_scale
    observed = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    return B, z[:, None], observed, event


def fit_cox(X, t, d, alpha=0.0):
    # Precompute risk-set incidence (continuous synthetic event times, no ties).
    ev_idx = np.flatnonzero(d.astype(bool))
    risk_mask = (t[None, :] >= t[ev_idx, None])
    xev = X[ev_idx].sum(0)
    def fg(w):
        r = X@w
        logits = np.where(risk_mask, r[None, :], -np.inf)
        m = logits.max(1, keepdims=True)
        a = np.exp(logits-m) * risk_mask
        probs = a / a.sum(1, keepdims=True)
        l = -(xev @ w) + np.log(a.sum(1)).sum() + m.sum()
        grad = -xev + probs.sum(0) @ X
        return l + 0.5 * alpha * (w@w), grad + alpha*w
    sol = minimize(fg, np.zeros(X.shape[1]), jac=True, method='BFGS',
                   options={'gtol':1e-7,'maxiter':180})
    return sol.x


def within_pairs_concordance(r, t, d):
    ii = np.flatnonzero(d)
    earlier = t[ii,None] < t[None,:]
    ok = earlier & (np.arange(len(t))[ii,None] != np.arange(len(t))[None,:])
    delta = r[ii,None] - r[None,:]
    return float(((delta > 0)*ok + 0.5*(delta == 0)*ok).sum()/max(ok.sum(),1))


def single(seed, scenario, n=512, event_count=65):
    B,z,t,e=simulate(seed, scenario, n, event_count)
    rg=np.random.default_rng(seed+100000)
    idx=rg.permutation(n); cut=int(.8*n)
    tr,te=idx[:cut],idx[cut:]
    means=B[tr].mean(0); std=B[tr].std(0).clip(1e-8)
    b=(B-means)/std
    zm=z[tr].mean(0);zs=z[tr].std(0).clip(1e-8)
    zn=(z-zm)/zs
    # Both feature families see exactly the same training data.
    wb=fit_cox(b[tr],t[tr],e[tr],alpha=.5)
    r_base=b@wb
    zi,coef,inf=orthogonalize_interaction(zn[tr],b[tr],r_base[tr],t[tr],e[tr],ridge=1e-8)
    zperp=zn-b@coef.numpy()
    # Tests the purely geometric orthogonality property, not an asymptotic p-value.
    q=b.shape[1]
    cross=inf[:q,q:] - inf[:q,:q]@coef
    orth_error=float(torch.max(torch.abs(cross)).item())
    wraw=fit_cox(np.c_[b[tr],zn[tr]],t[tr],e[tr],alpha=.5)
    wortho=fit_cox(np.c_[b[tr],zperp[tr]],t[tr],e[tr],alpha=.5)
    # The unregularized model classes are identical after linear projection.
    wur=fit_cox(np.c_[b[tr],zn[tr]],t[tr],e[tr],alpha=0)
    wuo=fit_cox(np.c_[b[tr],zperp[tr]],t[tr],e[tr],alpha=0)
    equal_pred=float(np.max(np.abs((np.c_[b,zn]@wur)-(np.c_[b,zperp]@wuo))))
    return {'seed':seed,'scenario':scenario,'N':n,'events':int(e.sum()),
        'train_events':int(e[tr].sum()),'test_events':int(e[te].sum()),
        'base_C':within_pairs_concordance(r_base[te],t[te],e[te]),
        'raw_C':within_pairs_concordance((np.c_[b,zn]@wraw)[te],t[te],e[te]),
        'ortho_C':within_pairs_concordance((np.c_[b,zperp]@wortho)[te],t[te],e[te]),
        'orthogonality_error':orth_error,'unpenalized_equal_risk_error':equal_pred}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=12)
    ap.add_argument('--n',type=int,default=512)
    ap.add_argument('--events',type=int,default=65)
    ap.add_argument('--out',type=str,default='synthetic_results.json')
    a=ap.parse_args()
    rows=[single(710000+j,case,a.n,a.events) for case in ['null','redundant','interaction'] for j in range(a.reps)]
    summary={}
    for case in ['null','redundant','interaction']:
        rr=[r for r in rows if r['scenario']==case]
        summary[case]={k:float(np.mean([x[k] for x in rr])) for k in ['base_C','raw_C','ortho_C','orthogonality_error','unpenalized_equal_risk_error']}
        summary[case]['mean_raw_minus_base']=summary[case]['raw_C']-summary[case]['base_C']
        summary[case]['mean_ortho_minus_base']=summary[case]['ortho_C']-summary[case]['base_C']
        summary[case]['ortho_better_than_raw_repetitions']=sum(x['ortho_C']>x['raw_C'] for x in rr)
        summary[case]['raw_better_than_base_repetitions']=sum(x['raw_C']>x['base_C'] for x in rr)
    obj={'status':'SYNTHETIC_ONLY','data_label':'generated not TCGA','reps':a.reps,'n':a.n,'events_target':a.events,'summary':summary,'rows':rows}
    Path(a.out).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

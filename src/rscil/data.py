"""Independent synthetic cohorts and training-only transforms; no TCGA access."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from .geometry import EfronRiskSets

SCENARIOS=('NULL','REDUNDANT','INTERACTION','CORRELATED_INTERACTION')


def disjoint(train_ids,evaluation_ids):
    if len(set(train_ids))!=len(train_ids) or len(set(evaluation_ids))!=len(evaluation_ids):raise ValueError('DUPLICATE_IDS')
    if set(train_ids)&set(evaluation_ids):raise ValueError('TRAIN_EVALUATION_OVERLAP')


def _cohort(rng,n,events,scenario):
    c=rng.normal(size=(n,6));ep=rng.normal(size=(n,12));eg=rng.normal(size=(n,12))
    p=.65*c[:,[0]]+np.sqrt(1-.65**2)*ep
    g=.65*c[:,[0]]+np.sqrt(1-.65**2)*eg
    main=.7*c[:,0]+.5*p[:,0]+.4*g[:,0]
    if scenario=='NULL':risk=main
    elif scenario=='REDUNDANT':risk=main+.6*p[:,0]-.4*g[:,0]+.3*c[:,0]
    elif scenario=='INTERACTION':risk=main+.8*ep[:,1]*eg[:,1]
    elif scenario=='CORRELATED_INTERACTION':risk=main+1.2*c[:,0]+.4*ep[:,1]*eg[:,1]
    else:raise ValueError('UNLOCKED_SCENARIO')
    latent=-np.log(rng.uniform(.00001,.99999,n))*np.exp(-risk)
    # Independent cohort-specific type-II administrative censoring. It fixes
    # event count by study design, never by a model's score or selected epoch.
    ordered=np.sort(latent);cut=(ordered[events-1]+ordered[events])/2
    t=np.minimum(latent,cut);e=latent<=cut
    if int(e.sum())!=events:raise ValueError('EVENT_COUNT')
    return {'P':p,'G':g,'C':c,'time':t,'event':e,'oracle':risk,
            'censor_cut':float(cut)}


def generate(seed,n,scenario):
    nt=int(.8*n);ne=65 if n==512 else 260
    # Independent SeedSequence children ensure neither split changes when the
    # other split's outcomes or sample generation is replaced.
    a,b=np.random.SeedSequence(seed).spawn(2)
    train=_cohort(np.random.default_rng(a),nt,52 if n==512 else 208,scenario)
    test=_cohort(np.random.default_rng(b),n-nt,13 if n==512 else 52,scenario)
    assert train['event'].sum()+test['event'].sum()==ne
    return train,test


def fit_transform(fit,evaluation,*,device='cpu'):
    state={};batches=[{},{}]
    for key in ('P','G','C'):
        mu=fit[key].mean(0);sd=fit[key].std(0);sd=np.where(sd>1e-8,sd,1.)
        state[key]={'mean':mu,'sd':sd}
        for b,raw in zip(batches,(fit,evaluation)):
            b[key]=torch.as_tensor((raw[key]-mu)/sd,dtype=torch.float64,device=device)
    # Fixed, nonlearned expansion of P into 768D mean-patch input, retaining
    # the complete R2 mean/local_out architecture in synthetic controls.
    projection=np.random.default_rng(70001).normal(size=(12,768))/np.sqrt(12)
    for b in batches:b['mean_patch']=b['P']@torch.as_tensor(projection,dtype=torch.float64,device=device)
    return batches[0],batches[1],state


def fit_warm(batch,time,event,*,ridge=1.):
    keys=('P','G','C');dims=[batch[k].shape[1] for k in keys]
    x=torch.cat([batch[k] for k in keys],1)
    rs=EfronRiskSets(time,event,device=x.device)
    coef=torch.zeros(x.shape[1],dtype=torch.float64,device=x.device,requires_grad=True)
    opt=torch.optim.LBFGS([coef],lr=1.,max_iter=100,tolerance_grad=1e-10,tolerance_change=1e-13,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=rs.loss(x@coef)+.5*ridge*coef.square().sum();loss.backward();return loss
    opt.step(closure)
    out={};start=0
    for k,d in zip(('p','g','c'),dims):out[k+'_coef']=coef.detach()[start:start+d].clone();start+=d
    return out


def array_hash(x):
    a=np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

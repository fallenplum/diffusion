"""训练折严格交叉拟合，冻结推理状态，不保留训练样本患者数据。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .preprocess import fit_features, apply_features
from .cox import ridge_cox, scale_risk, scalar_weight

VERSION='1.0.0'

def train_model(raw, train, *, seed=27011):
    train=np.asarray(train,int)
    if len(np.unique(train))!=len(train) or len(train)<90:
        raise ValueError('必须至少90位互异训练患者，以保证3-fold teacher局部PCA与事件支持')
    t,e=np.asarray(raw['t'])[train],np.asarray(raw['e'])[train]
    rng=np.random.default_rng(seed)
    blocks=np.array_split(rng.permutation(train),3)
    cp_cf=np.empty(len(train)); rna_cf=np.empty(len(train)); group=np.empty(len(train),int)
    pos={int(j):k for k,j in enumerate(train)}
    teachers=[]
    for gid,held in enumerate(blocks):
        fit=np.setdiff1d(train,held)
        cp,rna,_=fit_features(raw,fit)
        m1,d1=ridge_cox(cp[fit],np.asarray(raw['t'])[fit],np.asarray(raw['e'])[fit],10.)
        m2,d2=ridge_cox(rna[fit],np.asarray(raw['t'])[fit],np.asarray(raw['e'])[fit],1.)
        cpall,_=scale_risk(cp[fit]@m1,cp@m1)
        rall,_=scale_risk(rna[fit]@m2,rna@m2)
        for h in held:
            j=pos[int(h)]; cp_cf[j]=cpall[h];rna_cf[j]=rall[h];group[j]=gid
        teachers.append({'teacher':gid,'train_patients':len(fit),'held_patients':len(held),
                         'cp_max_gradient':d1['gradient_max'],'rna_max_gradient':d2['gradient_max']})
    beta,cal=scalar_weight(cp_cf,rna_cf,t,e,group)
    cp,rna,pre=fit_features(raw,train)
    cpcoef,cpinfo=ridge_cox(cp[train],t,e,10.)
    rnacoef,rnainfo=ridge_cox(rna[train],t,e,1.)
    cpmu=cp[train]@cpcoef; rnamu=rna[train]@rnacoef
    _,cps=scale_risk(cpmu,cpmu);_,rnas=scale_risk(rnamu,rnamu)
    state={'preprocessing':pre,'cp_coefficients':cpcoef,'rna_coefficients':rnacoef,
           'cp_risk_scale':cps,'rna_risk_scale':rnas,'beta':beta,
           'info':{'version':VERSION,'seed':int(seed),'train_count':len(train),'train_event_count':int(e.sum()),
                   'teacher':teachers,'calibration':cal,'cp_optimizer':cpinfo,'rna_optimizer':rnainfo,
                   'source_algorithm':'diffusion R4 CP+RNA scalar; feature-level refit'}}
    return state


def predict_model(raw,state):
    cp,rna=apply_features(raw,state['preprocessing'])
    c=cp@state['cp_coefficients'];g=rna@state['rna_coefficients']
    sc=state['cp_risk_scale'];sg=state['rna_risk_scale']
    c=(c-sc['mean'])/sc['sd'];g=(g-sg['mean'])/sg['sd']
    score=c+state['beta']*g
    if not np.isfinite(score).all():raise ValueError('输出风险非有限数值')
    return {'cp_risk':c,'rna_risk':g,'risk_score':score}


def _to_serializable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.integer,np.floating)):return x.item()
    if isinstance(x,dict):return {k:_to_serializable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_to_serializable(y) for y in x]
    return x


def save_model(state,path):
    """JSON无Pickle模型工件，拒绝隐式执行任意对象；可读且易审计。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    import os,tempfile
    fd,tmp=tempfile.mkstemp(dir=str(path.parent),suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(_to_serializable(state),f,ensure_ascii=False,allow_nan=False,sort_keys=True)
            f.write('\n')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def load_model(path):
    s=json.loads(Path(path).read_text(encoding='utf-8'))
    if s['info']['version']!=VERSION:raise ValueError('模型版本不匹配')
    st=s['preprocessing']
    for x in [st['s32'],st['conch_before'],st['conch_after'],st['rna']['before'],st['rna']['after']]:
        for k in ('mean','sd'):x[k]=np.asarray(x[k],float)
    st['pca_components']=np.asarray(st['pca_components'],float);st['pca_mean']=np.asarray(st['pca_mean'],float)
    st['rna']['median']=np.asarray(st['rna']['median'],float)
    st['rna']['measured']=np.asarray(st['rna']['measured'],int)
    st['rna']['selected']=np.asarray(st['rna']['selected'],int)
    st['clinical']['scale']['mean']=np.asarray(st['clinical']['scale']['mean'],float)
    st['clinical']['scale']['sd']=np.asarray(st['clinical']['scale']['sd'],float)
    s['cp_coefficients']=np.asarray(s['cp_coefficients'],float)
    s['rna_coefficients']=np.asarray(s['rna_coefficients'],float)
    return s

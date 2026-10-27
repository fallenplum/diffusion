"""Common protocol and outcome-free, partition-local preprocessing."""
from __future__ import annotations
import hashlib
import json
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

CLINICAL=('AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE')

def array_hash(value):
    x=np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()

def require(condition,message):
    if not bool(condition):raise ValueError(message)

def fixed_inner_splits(folds,event,outer=1,seed=27001):
    """Select only within this outer training partition; no outer scores enter."""
    folds,event=np.asarray(folds),np.asarray(event)
    train=np.flatnonzero(folds!=outer)
    require(event.shape==folds.shape and np.isin(event,[0,1]).all(),'Invalid fold/endpoint vectors')
    splitter=StratifiedKFold(3,shuffle=True,random_state=seed+int(outer))
    return [(train[a],train[b]) for a,b in splitter.split(train,event[train])]

def scale_fit(x,train):
    x=np.asarray(x,np.float64);train=np.asarray(train,int)
    mean=x[train].mean(0);sd=x[train].std(0)
    return dict(mean=mean,sd=np.where(sd>1e-8,sd,1.),fit_hash=array_hash(train))

def scale_apply(x,state):return (np.asarray(x,np.float64)-state['mean'])/state['sd']

def clinical_fit(raw,train):
    columns=[];coding={}
    for name in CLINICAL:
        v=np.asarray(raw['clinical_raw'][name]).astype(str)
        if name=='AGE':
            values=np.array([float(s) if s.replace('.','',1).isdigit() else np.nan for s in v])
            require(np.isfinite(values[train]).any(),'No training age')
            median=float(np.nanmedian(values[train]));coding[name]=dict(median=median)
            columns.append(np.where(np.isfinite(values),values,median)[:,None])
        else:
            levels=np.unique(v[train]);coding[name]=dict(levels=levels)
            columns.append(np.stack([v==level for level in levels],1))
    x=np.concatenate(columns,1).astype(np.float64)
    state=scale_fit(x,train)
    return scale_apply(x,state),dict(coding=coding,scale=state)

def molecular_fit(raw,train,membership):
    x=np.asarray(raw['rna'],np.float64);mask=np.isfinite(x)
    require(not np.isinf(x).any(),'Infinite RNA')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(x[train],axis=0)
    measured=mask[train].sum(0);median=np.where(np.isfinite(median),median,0.)
    filled=np.where(mask,x,median)
    state=scale_fit(filled,train)
    # Membership is externally fixed; selection uses training variance only.
    annotated=np.asarray(raw.get('_direct_gene_universe',np.asarray(membership,bool).any(0)),bool)
    variance=np.var(filled[train],axis=0)
    eligible=(measured>=.9*len(train))&(variance>1e-16)
    order=np.argsort(-variance,kind='stable')
    hallmark=order[eligible[order]&annotated[order]]
    outside=order[eligible[order]&~annotated[order]]
    selected=np.r_[hallmark[:512],outside[:max(0,512-len(hallmark))]]
    require(len(selected)==512,'Not enough eligible RNA512 genes')
    z=scale_apply(filled,state);z=np.where(mask&(measured[None,:]>0),z,0.).astype(np.float32)
    r=z[:,selected].astype(np.float64)
    # Explicit training-only standardization after the legacy float32 boundary.
    rs=scale_fit(r,train);r=scale_apply(r,rs)
    cna=np.asarray(raw['cna'],np.float64)
    lookup={str(s):i for i,s in enumerate(raw['cna_symbols'])}
    mapping=np.array([lookup.get(str(s),-1) for s in raw['rna_symbols']],int)
    c=np.full(x.shape,np.nan)
    available=mapping>=0;c[:,available]=cna[:,mapping[available]]
    cm=np.isfinite(c);gain=np.where(cm,np.maximum(c,0.),0.);loss=np.where(cm,np.maximum(-c,0.),0.)
    # Gain/loss are split BEFORE centering; unmeasured channels stay masked.
    channels=[];cs=[]
    for v in (gain,loss):
        count=cm[train].sum(0);mean=np.where(cm[train],v[train],0.).sum(0)/np.maximum(count,1)
        sd=np.sqrt(np.where(cm[train],(v[train]-mean)**2,0.).sum(0)/np.maximum(count,1))
        st=dict(mean=mean,sd=np.where(sd>1e-8,sd,1.),count=count)
        channels.append(np.where(cm&(count[None,:]>0),(v-mean)/st['sd'],0.).astype(np.float32));cs.append(st)
    return dict(rna=z,rna_mask=mask&(measured[None,:]>0),R=r,cna_gain=channels[0],cna_loss=channels[1],cna_mask=cm&(cm[train].sum(0)[None,:]>0),
        state=dict(median=median,scale=state,measured=measured,selected=selected,selected_scale=rs,
                   cna_mapping=mapping,cna_channel_scales=cs,fit_hash=array_hash(train),eligible_genes=int(eligible.sum())))

def prepare_partition(raw,train,membership):
    train=np.asarray(train,int)
    require(len(np.unique(train))==len(train) and len(train)>32,'Invalid training indices')
    c,clinical=clinical_fit(raw,train)
    ss=scale_fit(raw['S32'],train);s=scale_apply(raw['S32'],ss)
    result=dict(C=c,CS=np.c_[c,s]);state=dict(clinical=clinical,S32=ss,fit=train,fit_hash=array_hash(train),visual={})
    for name,key in [('LOCAL','conch_local'),('GLOBAL','conch_mean')]:
        x=np.asarray(raw[key],float);before=scale_fit(x,train);z=scale_apply(x,before)
        pca=PCA(32,svd_solver='full').fit(z[train]);pc=pca.transform(z);after=scale_fit(pc,train)
        result[name]=np.c_[result['CS'],scale_apply(pc,after)]
        state['visual'][name]=dict(before=before,components=pca.components_,mean=pca.mean_,after=after)
    mol=molecular_fit(raw,train,membership)
    result.update({k:v for k,v in mol.items() if k!='state'});state['molecular']=mol['state']
    result['state']=state
    return result

def patient_mismatch(indices,strata,seed):
    """Within-partition derangement, preferring exact strata; record fallback."""
    indices=np.asarray(indices,int);rng=np.random.default_rng(seed);mapping=np.arange(len(strata));single=[]
    for value in np.unique(np.asarray(strata)[indices]):
        rows=indices[np.asarray(strata)[indices]==value]
        if len(rows)<2:single.extend(rows.tolist());continue
        order=rng.permutation(rows);mapping[order]=np.roll(order,1)
    # Singleton strata are paired within the same partition using a global cycle.
    # This intentionally relaxes the stratum condition, never patient disjointness.
    if single:
        if len(single)==1:
            j=single[0];other=indices[indices!=j][0];old=mapping[other]
            mapping[other]=j;mapping[j]=old
        else:
            order=rng.permutation(single);mapping[order]=np.roll(order,1)
    require(np.all(mapping[indices]!=indices) and set(mapping[indices])==set(indices),'Mismatch is not a within-partition derangement')
    return mapping,dict(singleton_stratum_fallbacks=len(single),n=len(indices),fixed_points=0)

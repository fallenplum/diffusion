"""New common anchor fits; old model predictions are not training inputs."""
from __future__ import annotations
import numpy as np
from .protocol import prepare_partition,array_hash
from .survival import fit_offset_ridge,standardized_risk,fit_scalar_same_teacher

def prepare_anchors(raw,train,valid,membership,seed=27011):
    train,valid=np.asarray(train,int),np.asarray(valid,int)
    if np.intersect1d(train,valid).size:raise ValueError('Train and evaluation overlap')
    n=len(raw['t']);full=prepare_partition(raw,train,membership)
    t,e=np.asarray(raw['t']),np.asarray(raw['e'])
    cp_cf=np.zeros(n);cs_cf=np.zeros(n);rna_cf=np.zeros(n);groups=np.full(n,-1,int);teachers=[]
    blocks=np.array_split(np.random.default_rng(seed).permutation(train),3)
    for group,held in enumerate(blocks):
        fit=np.setdiff1d(train,held);features=prepare_partition(raw,fit,membership)
        teacher=dict(train=fit,held=held,preprocessing=features['state'],models={})
        for key,destination,lam in [('LOCAL',cp_cf,10.),('CS',cs_cf,10.),('R',rna_cf,1.)]:
            coef,info=fit_offset_ridge(features[key][fit],t[fit],e[fit],lam)
            prediction=features[key]@coef
            scaled,scale=standardized_risk(prediction[fit],prediction)
            destination[held]=scaled[held]
            teacher['models'][key]=dict(coef=coef,scale=scale,optimizer=info)
        groups[held]=group;teachers.append(teacher)
    if np.any(groups[train]<0):raise ValueError('Missing crossfit risk')
    full_models={};full_risks={}
    for key,lam in [('LOCAL',10.),('GLOBAL',10.),('CS',10.),('R',1.)]:
        coef,info=fit_offset_ridge(full[key][train],t[train],e[train],lam)
        prediction=full[key]@coef
        scaled,scale=standardized_risk(prediction[train],prediction)
        full_models[key]=dict(coef=coef,scale=scale,optimizer=info)
        full_risks[key]=scaled
    beta,scalar=fit_scalar_same_teacher(cp_cf[train],rna_cf[train],t[train],e[train],groups[train],1.)
    cp=full_risks['LOCAL'].copy();cp[train]=cp_cf[train]
    cs=full_risks['CS'].copy();cs[train]=cs_cf[train]
    rna=full_risks['R'].copy();rna[train]=rna_cf[train]
    diagnostics={}
    for name,cf,key in [('CP',cp_cf,'LOCAL'),('CS',cs_cf,'CS'),('RNA',rna_cf,'R')]:
        a,b=cf[train],full_risks[key][train]
        diagnostics[name]=dict(crossfit_mean=float(a.mean()),fullfit_mean=float(b.mean()),
            crossfit_sd=float(a.std()),fullfit_sd=float(b.std()),pearson=float(np.corrcoef(a,b)[0,1]),
            sd_ratio=float(a.std()/b.std()),RMS_gap=float(np.sqrt(np.mean((a-b)**2))),
            alignment='each_teacher_own_train_mean_SD_not_heldout_moments')
    return dict(features=full,train=train,valid=valid,fit_time=t[train],fit_event=e[train],
        CP=cp,CS=cs,RNA=rna,base=cp+beta*rna,beta=beta,scalar_audit=scalar,
        teacher_groups=groups,teachers=teachers,full_models=full_models,full_risks=full_risks,
        risk_distribution=diagnostics,fit_hash=array_hash(train),valid_hash=array_hash(valid),
        upstream_optimizer_fits=13,upstream_preprocessing_fits=4,
        pair_teacher_rule='both_pair_members_in_one_excluded_teacher_block')

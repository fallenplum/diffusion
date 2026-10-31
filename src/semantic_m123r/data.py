"""Current-fit-only feature scaling; endpoints are handled separately."""
import numpy as np
import torch
from src.semantic_m123.data import historical_covariates,equal_slide_mean


def patch_scale(patch,mask,slide,nt):
    first=[];second=[]
    for i in range(nt):
        one=[];two=[]
        for sid in np.unique(slide[i,mask[i]]):
            x=patch[i,mask[i]&(slide[i]==sid)].astype(np.float64)
            one.append(x.mean(0));two.append((x*x).mean(0))
        if not one:raise ValueError('EMPTY_TRAINING_PATIENT')
        first.append(np.mean(one,0));second.append(np.mean(two,0))
    mu=np.mean(first,0);sd=np.sqrt(np.maximum(np.mean(second,0)-mu*mu,1e-8))
    z=((patch.astype(np.float64)-mu)/sd).astype(np.float32);z[~mask]=0.
    return z,{'patch_mean':mu,'patch_sd':sd,'moments':'equal_patient_equal_slide_observed_patch_moments',
        'fit_patients':nt,'mean_and_patch_share_exact_same_feature_scale':True,'DSS_used':False}


def repair_inputs(data,raw):
    b=dict(data.batch);st=data.state;rows=data.rows;nt=len(data.train)
    c,s,grid,r=historical_covariates(raw,rows,st['historical_preprocessing'])
    def tr(key,name):
        # Original batch is already rounded. Recover pre-rounding global
        # covariates from the exact fixed feature source supplied by caller.
        x=np.asarray(raw[name],np.float64)[rows];v=st[key];return (x-v['mean'])/v['sd']
    native=tr('native','m123r_titan_native');aligned=tr('aligned','m123r_titan_aligned')
    b['P']=torch.from_numpy(np.c_[s,grid,native,aligned])
    oldg=data.batch['G'].numpy();b['G']=torch.from_numpy(np.c_[r,oldg[:,512:].astype(np.float64)])
    b['C']=torch.from_numpy(c)
    patch=b['patch'].numpy();mask=b['mask'].numpy();slide=b['slide'].numpy()
    z,scale=patch_scale(patch,mask,slide,nt)
    b['patch']=torch.from_numpy(z)
    b['mean_patch']=torch.from_numpy(equal_slide_mean(z.astype(np.float64),mask,slide))
    return b,scale

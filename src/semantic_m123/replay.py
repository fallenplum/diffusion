"""Rebuild frozen inputs from saved fold transforms without refitting anything."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
import torch
from src.semantic_m123.data import historical_covariates,equal_slide_mean


def load_saved_inputs(raw,frozen,preprocessing,fit,evaluation,*,expected_input_sha256):
    rows=np.r_[fit,evaluation];n=len(rows);state=preprocessing
    c,s,grid,r=historical_covariates(raw,rows,state['historical_preprocessing'])
    registry=np.asarray(frozen['global_rows'],int);lookup={int(v):i for i,v in enumerate(registry)}
    take=torch.tensor([lookup[int(v)] for v in rows],dtype=torch.long)
    image=frozen['image'];patch=image['patch_values'][take].numpy()
    mask=image['patch_mask'][take].numpy();slide=image['patch_slide'][take].numpy()
    def transform(x,key):
        st=state[key]
        return (np.asarray(x,np.float64)-st['mean'])/st['sd']
    mean=transform(equal_slide_mean(patch,mask,slide),'mean')
    native=transform(image['titan_native'][take].numpy(),'native')
    aligned=transform(image['titan_aligned'][take].numpy(),'aligned')
    rr=np.where(np.asarray(raw['rna_measured'])[rows],np.asarray(raw['rna'])[rows],np.nan)
    cc=np.where(np.asarray(raw['cna_measured'])[rows],np.asarray(raw['cna'])[rows],np.nan)
    program=state['program'].transform(rr,cc)
    p=np.c_[s,grid,native,aligned]
    g=np.c_[r,program['values'].reshape(n,-1),program['measured'].reshape(n,-1),program['coverage'].reshape(n,-1)]
    batch={'patch':torch.from_numpy(patch),'mask':torch.from_numpy(mask),'slide':torch.from_numpy(slide),
        'mean':torch.from_numpy(mean.astype(np.float32)),'P':torch.from_numpy(p.astype(np.float32)),
        'G':torch.from_numpy(g.astype(np.float32)),'C':torch.from_numpy(c.astype(np.float32)),
        'program':torch.from_numpy(program['values']),'program_mask':torch.from_numpy(program['measured']),
        'coverage':torch.from_numpy(program['coverage'])}
    from src.semantic_b0b7.p2_anchor import array_sha
    inputsha=hashlib.sha256(json.dumps({k:array_sha(v.numpy()) for k,v in batch.items()},sort_keys=True).encode()).hexdigest()
    if inputsha!=expected_input_sha256:raise ValueError('SAVED_PREPROCESSOR_INPUT_REPLAY_FAILED')
    flat=transform(np.c_[p,g,c,mean],'flat')
    return batch,flat,rows


def repair_damage(time,event,risk,reference,weights=None):
    t=np.asarray(time);e=np.asarray(event,dtype=bool);r=np.asarray(risk,np.float64);b=np.asarray(reference,np.float64)
    w=np.ones(len(t)) if weights is None else np.asarray(weights,np.float64)
    repair=damage=den=0.
    for i in np.flatnonzero(e):
        j=np.flatnonzero((t>t[i])|((t==t[i])&~e))
        def credit(z):
            d=z[i]-z[j]
            return np.where(np.abs(d)<=1e-8,.5,(d>0).astype(float))
        d=credit(r)-credit(b)
        repair+=w[i]*np.maximum(d,0).sum();damage+=w[i]*np.maximum(-d,0).sum();den+=w[i]*len(j)
    return {'repair_num':float(repair),'damage_num':float(damage),'den':float(den),
            'net_delta':float((repair-damage)/den) if den>0 else None}


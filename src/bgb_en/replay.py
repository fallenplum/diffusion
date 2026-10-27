"""Apply saved fold transformations without fitting or reading outcome labels."""
from __future__ import annotations
import numpy as np
from .protocol import CLINICAL,scale_apply,array_hash


def apply_saved_preprocessing(raw,state):
    train=np.asarray(state['fit'],int)
    if state['fit_hash']!=array_hash(train):raise ValueError('Saved preprocessing fit identity changed')
    columns=[]
    for name in CLINICAL:
        value=np.asarray(raw['clinical_raw'][name]).astype(str);coding=state['clinical']['coding'][name]
        if name=='AGE':
            number=np.array([float(s) if s.replace('.','',1).isdigit() else np.nan for s in value])
            columns.append(np.where(np.isfinite(number),number,coding['median'])[:,None])
        else:columns.append(np.stack([value==level for level in coding['levels']],1))
    c=scale_apply(np.concatenate(columns,1).astype(np.float64),state['clinical']['scale'])
    s=scale_apply(raw['S32'],state['S32']);out=dict(C=c,CS=np.c_[c,s])
    for name,key in [('LOCAL','conch_local'),('GLOBAL','conch_mean')]:
        saved=state['visual'][name];z=scale_apply(raw[key],saved['before'])
        pc=(z-saved['mean'])@saved['components'].T
        out[name]=np.c_[out['CS'],scale_apply(pc,saved['after'])]
    st=state['molecular'];x=np.asarray(raw['rna'],np.float64);mask=np.isfinite(x)
    measured=np.asarray(st['measured']);filled=np.where(mask,x,st['median'])
    z=scale_apply(filled,st['scale'])
    z=np.where(mask&(measured[None,:]>0),z,0.).astype(np.float32)
    out.update(rna=z,rna_mask=mask&(measured[None,:]>0),R=scale_apply(z[:,st['selected']].astype(np.float64),st['selected_scale']))
    mapping=np.asarray(st['cna_mapping']);available=mapping>=0
    aligned=np.full(x.shape,np.nan);aligned[:,available]=np.asarray(raw['cna'],float)[:,mapping[available]]
    cm=np.isfinite(aligned)
    for name,v,channel in [('cna_gain',np.where(cm,np.maximum(aligned,0),0),0),
                           ('cna_loss',np.where(cm,np.maximum(-aligned,0),0),1)]:
        saved=st['cna_channel_scales'][channel]
        out[name]=np.where(cm&(saved['count'][None,:]>0),(v-saved['mean'])/saved['sd'],0.).astype(np.float32)
    out['cna_mask']=cm&(st['cna_channel_scales'][0]['count'][None,:]>0)
    out['state']=state
    return out


def replay_full_risks(features,models):
    risks={}
    for key,saved in models.items():
        prediction=features[key]@saved['coef']
        risks[key]=(prediction-saved['scale']['mean'])/saved['scale']['sd']
    return risks

"""Outcome-free interpretation of frozen attention; no medical ground truth claim."""
import numpy as np


def effective_rank(matrix):
    s=np.linalg.svd(np.asarray(matrix,np.float64),compute_uv=False)
    total=s.sum()
    if total<=1e-12:return 0.
    p=s/total;p=p[p>0]
    return float(np.exp(-np.sum(p*np.log(p))))


def summarize_attention(attention,tokens,patch_mask,query_mask=None):
    a=np.asarray(attention,np.float64).mean(1)
    z=np.asarray(tokens,np.float64);pm=np.asarray(patch_mask,bool)
    qm=np.ones(a.shape[:2],bool) if query_mask is None else np.asarray(query_mask,bool)
    values={k:[] for k in ('effective_patch_fraction','attention_pair_cosine','slot_effective_rank','slot_rank_fraction',
        'centered_slot_effective_rank','centered_slot_energy_fraction')}
    for i in range(len(a)):
        if not qm[i].any():continue
        w=a[i][qm[i]][:,pm[i]];v=z[i][qm[i]]
        if not np.allclose(w.sum(1),1,atol=1e-6):raise ValueError('ATTENTION_NOT_NORMALIZED_ON_REAL_PATCHES')
        positive=np.maximum(w,1e-300)
        entropy=-np.sum(w*np.log(positive),axis=1)
        values['effective_patch_fraction'].append(float(np.mean(np.exp(entropy)/pm[i].sum())))
        if len(w)>1:
            norm=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-30)
            gram=norm@norm.T
            values['attention_pair_cosine'].append(float((gram.sum()-np.trace(gram))/(len(w)*(len(w)-1))))
        rank=effective_rank(v)
        values['slot_effective_rank'].append(rank)
        values['slot_rank_fraction'].append(rank/len(v))
        values['centered_slot_effective_rank'].append(effective_rank(v-v.mean(0,keepdims=True)))
        values['centered_slot_energy_fraction'].append(float(np.sum((v-v.mean(0,keepdims=True))**2)/max(np.sum(v**2),1e-30)))
    result={'patients_total':len(a),'patients_with_observed_queries':len(values['slot_effective_rank'])}
    for k,x in values.items():
        result[k+'_mean']=float(np.mean(x)) if x else None
        result[k+'_min']=float(np.min(x)) if x else None
        result[k+'_max']=float(np.max(x)) if x else None
    return result


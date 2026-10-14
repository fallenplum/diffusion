"""Inherit exact Phase11 schedule/Cox; explicit factory and split-local inputs."""
from contextlib import contextmanager
import numpy as np
import src.phase11_semantic.train as old
from src.phase10_trimodal.features import ahash
from .models import C4Model


def derange(d,ix,seed):
    ix=np.asarray(ix,int);rng=np.random.default_rng(seed);src=ix.copy()
    missing=np.isnan(d['raw_rna']).sum(1)>0
    cancer=d['cancer_type'].astype(str);fine=np.char.add(cancer,np.where(missing,'_missing','_complete'))
    blocks=[np.flatnonzero(fine[ix]==k) for k in np.unique(fine[ix])];merged=[];fallback=[]
    # Locked hierarchy: merge singleton fine block into SAME-cancer complete
    # block; if unavailable combine remaining singletons; final full-split ring.
    single=[v for v in blocks if len(v)==1];blocks=[v for v in blocks if len(v)>1]
    for one in single:
        target=next((j for j,v in enumerate(blocks) if cancer[ix[v[0]]]==cancer[ix[one[0]]]),None)
        if target is not None:blocks[target]=np.r_[blocks[target],one];fallback.append('same_cancer_relax_missing')
        else:merged.extend(one.tolist())
    if merged:
        if len(merged)>1:blocks.append(np.array(merged));fallback.append('combine_singletons_relax_cancer')
        elif blocks:blocks[0]=np.r_[blocks[0],merged];fallback.append('one_remaining_singleton_merge')
        else:raise ValueError('split too small for derangement')
    for block in blocks:
        order=rng.permutation(block);src[order]=ix[np.roll(order,1)]
    if np.any(src==ix) or set(src)!=set(ix):raise ValueError('wrong RNA mapping crosses split or has fixed points')
    return src,{'fallback':fallback,'map_hash':ahash(src),'preserved_cancer':bool(np.array_equal(cancer[ix],cancer[src])),'preserved_missing':bool(np.array_equal(missing[ix],missing[src]))}


def prepare_inputs(d,f,partitions,kind,split_seed):
    out={**f,'R':f['R'].copy()};maps={}
    for name,ix,offset in partitions:
        source,meta=derange(d,ix,split_seed+offset) if kind=='wrong' else (np.asarray(ix),{'fallback':[],'preserved_cancer':True,'preserved_missing':True})
        out['R'][ix]=f['R'][source];maps[name]={'RNA':source.tolist(),'metadata':meta}
    return out,maps


def train(d,f,clinical_mask,membership,cache,fit,val,seed,cfg,lam,kind='real',no_anchor=False,epochs=None,checkpoint_path=None):
    # No Phase11 source file is modified. Factory replacement is process-local
    # and restored before returning; parent loss/optimizer/selection is exact.
    original=old.SemanticModel
    try:
        old.SemanticModel=lambda arm,cm,gm,hidden:C4Model(cm,gm,hidden,kind,no_anchor)
        return old.train(d,f,{'C':clinical_mask,'random':membership},cache,fit,val,'C4',seed,cfg,lam,epochs,checkpoint_path)
    finally:old.SemanticModel=original


forward_partition=old.forward_partition

import hashlib,json,warnings
from pathlib import Path
import numpy as np
import h5py
import torch
from src.phase13_rna import data as parent
from src.phase10_trimodal.features import clinical,scale,gene_fit,ahash
from src.phase10_trimodal.data import digest
from src.phase10_trimodal.math import fit_ridge

ROOT=Path(__file__).resolve().parents[2]
P=Path('/root/autodl-tmp/phase17_ssd_private')
R=ROOT/'reports/phase17_ssd_fusion'
CONFIG=ROOT/'configs/phase17_ssd_fusion.json'

def cfg():return json.loads(CONFIG.read_text())
def write(p,x):
    def cast(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=cast,allow_nan=False)+'\n');tmp.replace(p)

def source():
    paths=[CONFIG]+sorted((ROOT/'src/phase17_ssd').glob('*.py'))
    # Evaluation/report scripts can be corrected without changing training identity.
    paths+=[ROOT/'scripts/run_phase17.py']
    return hashlib.sha256(''.join(str(p.relative_to(ROOT))+digest(p) for p in paths).encode()).hexdigest()

def prepare_visual():
    target=P/'visual';target.mkdir(parents=True,exist_ok=True)
    if (target/'READY.json').exists():
        a=json.loads((target/'READY.json').read_text())
        for k in ['u','conch','mask','owner','coords','local_row']:
            if digest(target/(k+'.npy'))!=a['hashes'][k]:raise ValueError('visual cache changed')
        return a
    old=Path('/root/phase16_routing_private');prev=Path('/root/phase16_prompt_rna_diagnostic_private')
    inv=json.loads((old/'ASSET_INVENTORY.json').read_text())
    con=sorted(inv['chosen_conch'],key=lambda r:(r['patient_index'],r['stem']))
    uni=[r for r in inv['h5'] if r['kind']=='UNI2H_CANDIDATE']
    selection=np.load(prev/'PAIRED_VISUAL_SELECTION_PRIVATE.npz')
    n=len(con);cap=512
    u=np.lib.format.open_memmap(target/'u.npy',mode='w+',dtype='float16',shape=(n,cap,1536));u[:]=0
    c=np.lib.format.open_memmap(target/'conch.npy',mode='w+',dtype='float16',shape=(n,cap,768));c[:]=0
    mask=np.zeros((n,cap),bool);xyall=np.zeros((n,cap,2),np.int64);rowsall=np.full((n,cap),-1,np.int64);owner=[];cover=[];private=[]
    for i,co in enumerate(con):
        pi=co['patient_index'];owner.append(pi)
        up=sorted([r for r in uni if r['stem']==co['stem'] and r['patient_index']==pi],key=lambda r:len(r['path']))[0]
        m=dict(np.load(old/'coordinate_maps'/f'slide_{i+1:04d}.npz'))
        if digest(co['path'])!=str(m['conch_h5_sha256']) or digest(up['path'])!=str(m['uni2_h5_sha256']):raise ValueError('source H5 changed')
        take=selection[str(i)]
        with h5py.File(co['path'],'r') as h:cx=h['features'][:].reshape(-1,768).astype('float32');xy=h['coords'][:].reshape(-1,2)
        # Lock one geometry order for every arm, preserving the exact old selected IDs.
        take=take[np.lexsort((xy[take,1],xy[take,0]))];q=len(take)
        with h5py.File(up['path'],'r') as h:ux=h['features'][:].reshape(-1,1536).astype('float32')
        ux/=np.maximum(np.linalg.norm(ux,axis=1,keepdims=True),1e-12);cx/=np.maximum(np.linalg.norm(cx,axis=1,keepdims=True),1e-12)
        keep=np.isin(m['conch_row'],take);rr=m['conch_row'][keep];jj=m['uni2_row'][keep];area=m['intersection_area'][keep].astype(float)
        order=np.argsort(take);pos=order[np.searchsorted(take[order],rr)]
        aligned=np.zeros((q,1536),np.float64);den=np.bincount(pos,weights=area,minlength=q)
        if (den<=0).any():raise ValueError('unmatched selected footprint')
        np.add.at(aligned,pos,ux[jj]*area[:,None]);aligned/=den[:,None]
        aligned/=np.maximum(np.linalg.norm(aligned,axis=1,keepdims=True),1e-12)
        if not np.isfinite(aligned).all():raise ValueError('nonfinite visual')
        u[i,:q]=aligned;c[i,:q]=cx[take];mask[i,:q]=True;xyall[i,:q]=xy[take];rowsall[i,:q]=take
        cover.append({'anonymous_slide':i+1,'selected_patches':q,'mean_overlap_area':float(den.mean()),'minimum_overlap_area':float(den.min()),'mean_UNI2_subpatches':len(jj)/q})
        private.append({'patient_index':pi,'slide':co['stem'],'conch_sha256':digest(co['path']),'uni2_sha256':digest(up['path'])})
        if i%40==0:print('VISUAL_PREPARE',i,n,flush=True)
    u.flush();c.flush()
    for k,v in [('mask',mask),('owner',np.asarray(owner)),('coords',xyall),('local_row',rowsall)]:np.save(target/(k+'.npy'),v)
    if set(owner)!=set(range(512)):raise ValueError('missing patient')
    assert max(np.bincount(np.asarray(owner),weights=mask.sum(1)))<=512
    write(target/'PRIVATE_SOURCES.json',private)
    a={'status':'PASS','patients':512,'slides':n,'patches':int(mask.sum()),'dimensions':1536,'visual':'UNI2h normalized,physical CONCH footprint area weighted,normalized again','patch_IDs':'exact Phase16 dual-visual selection;coordinate order;no outcomes','validation':'candidate-pool-limited;geometry-only-validation','raw_image_QA':False,'hashes':{k:digest(target/(k+'.npy')) for k in ['u','conch','mask','owner','coords','local_row']},'dataset_sha256':digest(parent.DATA)}
    import pandas as pd
    R.mkdir(parents=True,exist_ok=True);pd.DataFrame(cover).to_csv(R/'VISUAL_COVERAGE.csv',index=False)
    write(target/'READY.json',a);write(R/'VISUAL_AUDIT.json',a);return a

def molecular_features(d,fold,role):
    part=parent.splits(d,'DSS',fold);fit=part['inner_train' if role=='inner' else 'outer_train']
    meta,z=parent.parent_features(fold,role)
    if meta['fit_indices']!=fit.tolist():raise ValueError('upstream preprocessing leakage')
    idx=np.asarray(meta['phase12']['enriched_indices'],int);symbols=d['rna_gene_symbols'].astype(str)[idx]
    r=np.asarray(z['f__enriched__R'],np.float32);rm=np.isfinite(d['rna_genes'][:,idx])
    loc={g:j for j,g in enumerate(d['cna_gene_symbols'].astype(str))}
    raw=np.full((512,512),np.nan)
    for k,g in enumerate(symbols):
        if g in loc:raw[:,k]=d['cna_genes'][:,loc[g]]
    cm=np.isfinite(raw);g,gs=gene_fit(np.maximum(raw,0),fit);l,ls=gene_fit(np.maximum(-raw,0),fit)
    # Unmeasured values are masked in pooling; direct shortcut receives median-imputed train-local values.
    cv,coding=clinical(d,fit);age=np.asarray(d['clinical_AGE']).astype(str)
    missing_age=np.array([not v.replace('.','',1).isdigit() for v in age],float)
    cv=np.c_[cv,missing_age];c,cs=scale(cv,fit);s,ss=scale(np.asarray(d['static'],float),fit)
    membership=np.asarray(z['gm_enriched_true'],bool)
    if membership.shape!=(8,512) or (membership.sum(1)==0).any():raise ValueError('invalid memberships')
    out={'C':c,'S':s,'R':r,'G':g,'L':l,'RM':rm,'CM':cm,'membership':membership}
    audit={'fold':fold,'role':role,'fit_hash':ahash(fit),'RNA_selection':meta['phase12']['enriched_hash'],'upstream_npz':meta['npz_sha256'],'CNA_source':'same RNA gene symbol;gain/loss before standardization;explicit mask','missing_RNA':int((~rm).sum()),'missing_CNA':int((~cm).sum()),'membership_sizes':membership.sum(1),'clinical_fields':list(coding),'RNA_is_log2_already':True}
    return {k:v.astype(np.float32) for k,v in out.items()},audit

def wrong_map(d,indices,seed):
    out=np.arange(512);ix=np.asarray(indices,int);rng=np.random.default_rng(seed)
    cancer=d['cancer_type'].astype(str);stage=d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str)
    pending=[]
    for ca in np.unique(cancer[ix]):
        ci=ix[cancer[ix]==ca];single=[]
        for st in np.unique(stage[ci]):
            block=ci[stage[ci]==st]
            if len(block)>1:
                block=rng.permutation(block);out[block]=np.roll(block,1)
            else:single.extend(block)
        if len(single)>1:
            block=rng.permutation(single);out[block]=np.roll(block,1)
        else:pending.extend(single)
    if len(pending)==1:
        # Merge singleton and one already mapped stratum as a full cycle.
        pi=pending[0];other=ix[(cancer[ix]==cancer[pi])&(ix!=pi)]
        if not len(other):other=ix[ix!=pi]
        chosen=other[0];partner=out[chosen];out[chosen]=pi;out[pi]=partner
    elif pending:
        block=rng.permutation(pending);out[block]=np.roll(block,1)
    if np.any(out[ix]==ix) or set(out[ix])!=set(ix):raise ValueError('wrong context not partition-local derangement')
    return out

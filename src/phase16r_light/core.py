import hashlib,json,os
from pathlib import Path
import numpy as np,torch
from sklearn.decomposition import PCA
from src.phase16_semantic.core import base,parent,ROOT,sha,write
from src.phase10_trimodal.features import gene_fit,scale,ahash
from src.phase16_semantic.prompts import derange

CONFIG=ROOT/'configs/phase16r_light_rna.json';P=Path('/root/phase16r_light_rna_private');R=ROOT/'reports/phase16r_light_baseline_rna_compact';PREVIOUS=Path('/root/phase16_prompt_rna_diagnostic_private')

def cfg():return json.loads(CONFIG.read_text())
def source():return hashlib.sha256((base.source()+sha(CONFIG)+sha(Path(__file__))+sha(ROOT/'scripts/run_phase16r_light_rna.py')).encode()).hexdigest()
def label(seed,fold):return f'seed{seed}_fold{fold}'
def lock():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True);expected={'status':'LOCKED','source':source(),'config_sha256':sha(CONFIG),'data_sha256':sha(parent.DATA),'parent_commit':cfg()['parent_commit']}
    if (P/'LOCK.json').exists():
        if json.loads((P/'LOCK.json').read_text())!=expected:raise ValueError('locked source/data changed')
    else:write(P/'LOCK.json',expected)
    write(R/'PROTOCOL_LOCK.json',expected)

def visual_sources():
    path=P/'VISUAL_SOURCES.npz'
    if path.exists():
        m=json.loads((P/'VISUAL_SOURCES.json').read_text())
        if sha(path)!=m['sha256']:raise ValueError('visual source changed')
        return dict(np.load(path))
    meta=base.manifest();raw=np.load(meta['features'],mmap_mode='r')
    if sha(meta['features'])!=meta['cache_sha256']:raise ValueError('raw visual cache mismatch')
    matched=np.zeros((512,768));counts=np.zeros(512);coverage=[]
    for i,s in enumerate(meta['slides']):
        m=np.load(base.P/'coordinate_maps'/f'slide_{i+1:04d}.npz');idx=np.unique(m['conch_row']);patient=s['patient']
        if len(idx):matched[patient]+=np.asarray(raw[s['offset']+idx]).astype(float).mean(0);counts[patient]+=1
        coverage.append({'anonymous_slide_number':i+1,'full_candidate_rows':s['n'],'matched_candidate_rows':len(idx)})
    if np.any(counts==0):raise ValueError('no matched global representation for a patient')
    old=np.load(PREVIOUS/'PAIRED_VISUAL.npz');oldmeta=json.loads((PREVIOUS/'PAIRED_VISUAL.json').read_text())
    if sha(PREVIOUS/'PAIRED_VISUAL.npz')!=oldmeta['sha256']:raise ValueError('previous local cache changed')
    # Raw patch rows are already normalized. The matched global pool uses
    # float64 means; the historical LOCAL and GLOBAL_ALL arrays stay exact.
    arrays={'LOCAL':old['C'],'GLOBAL_MATCHED':matched/counts[:,None],'GLOBAL_ALL':np.load(base.P/'conch_global.npy')}
    np.savez_compressed(path,**arrays);write(P/'VISUAL_SOURCES.json',{'status':'PASS','source':source(),'sha256':sha(path),'old_local_sha256':oldmeta['sha256'],'raw_visual_sha256':meta['cache_sha256'],'candidate_source_sha256':meta['source_sha256']})
    import pandas as pd
    pd.DataFrame(coverage).to_csv(R/'VISUAL_CANDIDATES.csv',index=False);return arrays

def quotas(capacity,total):
    counts=np.zeros(len(capacity),int)
    while counts.sum()<total:
        changed=False
        for i in range(len(counts)):
            if counts.sum()==total:break
            if counts[i]<capacity[i]:counts[i]+=1;changed=True
        if not changed:raise ValueError('insufficient pathway PCA dimensions')
    return counts

def bank(d,visual,fit):
    """Fit everything on this exact partition, never on its validation rows."""
    fit=np.asarray(fit,int);key=ahash(fit);path=P/'features'/f'{key}.pt';path.parent.mkdir(exist_ok=True)
    import fcntl
    with path.with_suffix('.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if path.exists():
            z=torch.load(path,map_location='cpu',weights_only=False)
            if z['source']!=source() or not np.array_equal(z['fit'],fit):raise ValueError('feature bank source/fit changed')
            return z,path
        cs,csstate=base.cs_features(d,fit);x={'CS':cs};state={'CS':csstate,'visual':{}}
        for k in ['LOCAL','GLOBAL_MATCHED','GLOBAL_ALL']:
            v,vs=scale(visual[k],fit);pca=PCA(32,svd_solver='full').fit(v[fit]);pc=pca.transform(v);pc,ps=scale(pc,fit);x[k]=np.c_[cs,pc];state['visual'][k]={'scale':vs,'components':pca.components_,'mean':pca.mean_,'pc_scale':ps}
        full,st=gene_fit(d['rna_genes'],fit);names=d['rna_gene_symbols'].astype(str);groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());union=set().union(*[set(v) for v in groups.values()]);eligible=np.flatnonzero(st['usable']);annotated=np.array([g in union for g in names]);annot=eligible[annotated[eligible]];indices=annot[np.argsort(-st['sd'][annot],kind='stable')[:512]];fallback=0
        if len(indices)<512:
            other=eligible[~annotated[eligible]];add=other[np.argsort(-st['sd'][other],kind='stable')[:512-len(indices)]];fallback=len(add);indices=np.r_[indices,add]
        if len(indices)!=512:raise ValueError('Hallmark512 selection failed')
        raw=full[:,indices].astype(np.float32).astype(float);h,hs=scale(raw,fit);x['HALL512']=h;pca=PCA(128,svd_solver='full').fit(h[fit]);pc=pca.transform(h);pcstates={}
        for dim in [8,16,32,64,128]:
            x['ORTHO'+str(dim)]=pc[:,:dim]
            if dim!=8:x['PCA'+str(dim)],pcstates['PCA'+str(dim)]=scale(pc[:,:dim],fit)
        members=[np.flatnonzero(np.array([g in values for g in names[indices]])) for values in groups.values()];capacity=np.array([min(len(g),len(fit)-1) for g in members]);maximum=quotas(capacity,128);blocks=[];bp=[]
        for ix,nn in zip(members,maximum):
            model=PCA(int(nn),svd_solver='full').fit(h[fit][:,ix]);blocks.append(model.transform(h[:,ix]));bp.append({'indices':ix,'components':model.components_,'mean':model.mean_})
        allocations={};bscales={}
        for dim in [16,32,64,128]:
            alloc=quotas(capacity,dim);v=np.concatenate([b[:,:q] for b,q in zip(blocks,alloc)],1);x['BPCA'+str(dim)],bscales['BPCA'+str(dim)]=scale(v,fit);allocations[str(dim)]=alloc
        state['RNA']={'gene_state':st,'indices':indices,'fallback':fallback,'scale':hs,'components':pca.components_,'pca_mean':pca.mean_,'pc_scales':pcstates,'blocks':bp,'block_allocations':allocations,'block_scales':bscales,'group_names':list(groups)}
        z={'source':source(),'fit':fit,'x':x,'state':state};tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(z,tmp);tmp.replace(path);return z,path

def wrong_design(d,z,fit,test,seed):
    strata=np.char.add(np.char.add(d['cancer_type'].astype(str),'|'),d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str));perm=np.arange(512);counts={}
    for k,part in [('fit',fit),('test',test)]:perm[part],counts[k]=derange(part,strata,seed+(k=='test'))
    n=z['x']['CS'].shape[1];x=np.c_[z['x']['CS'],z['x']['LOCAL'][perm,n:]];return x,{'permutation':perm,'fallback':counts,'fit':fit,'test':test}

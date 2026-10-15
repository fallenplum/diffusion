"""Read-only full-pool cache and outcome-blind patient-budget selectors."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,h5py,pandas as pd
from src.phase16_routing.core import *

def quotas(slides,m):
    q=np.zeros(len(slides),int)
    for patient in range(512):
        ix=[i for i,r in enumerate(slides) if r['patient']==patient];n=np.array([slides[i]['n'] for i in ix]);target=min(m,n.sum());used=np.zeros(len(ix),int)
        for _ in range(target):
            eligible=used<n;priority=n/(used+1);priority[~eligible]=-1;used[np.argmax(priority)]+=1
        q[ix]=used
        if used.sum()!=target or np.any(used==0):raise ValueError('patient quota failure')
    return q
def fps(points,k):
    # Independent WSI row selection. Batched masked farthest point sampling.
    b,n,dim=points.shape;device=points.device;finite=points[:,:,0].isfinite();x=torch.nan_to_num(points);count=finite.sum(1);centroid=x.sum(1)/count[:,None]
    ds=((x-centroid[:,None])**2).sum(-1).masked_fill(~finite,torch.inf);at=ds.argmin(1);distance=torch.full((b,n),torch.inf,device=device);result=[];batch=torch.arange(b,device=device)
    for step in range(min(k,n)):
        result.append(at);v=x[batch,at];distance=torch.minimum(distance,((x-v[:,None])**2).sum(-1));distance=distance.masked_fill(~finite,-torch.inf);distance[batch,at]=-torch.inf;at=distance.argmax(1)
    return torch.stack(result,1).cpu().numpy()
def main():
    started=time.time();P.mkdir(exist_ok=True);(P/'selectors').mkdir(exist_ok=True);inventory=json.loads((P/'ASSET_INVENTORY.json').read_text());chosen=sorted(inventory['chosen_conch'],key=lambda r:(r['patient_index'],r['stem']));total=sum(r['patches'] for r in chosen)
    cache=Path('/dev/shm/phase16_routing_full.npy');out=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(total,768));coord=np.zeros((total,2),np.int64);owners=np.zeros(total,np.int16);slides=[];offset=0;globals=[]
    for row in chosen:
        with h5py.File(row['path'],'r') as h:
            x=h['features'][:].reshape(-1,768).astype(np.float32);xy=h['coords'][:]
        n=len(x);norm=np.linalg.norm(x,axis=1)
        if np.any(norm<1e-12) or not np.isfinite(x).all():raise ValueError('zero/nonfinite patch')
        x/=norm[:,None];out[offset:offset+n]=x;coord[offset:offset+n]=xy;owners[offset:offset+n]=row['patient_index'];globals.append(x.mean(0))
        slides.append({'offset':offset,'n':n,'patient':row['patient_index'],'source_h5_sha256':row['sha256'],'geometry':row['coordinate_attrs']});offset+=n
    out.flush();np.save(P/'coords.npy',coord);np.save(P/'row_owners.npy',owners);g=np.zeros((512,768));counts=np.zeros(512)
    for r,x in zip(slides,globals):g[r['patient']]+=x;counts[r['patient']]+=1
    np.save(P/'conch_global.npy',g/counts[:,None]);source_sha=hashlib.sha256(''.join(r['source_h5_sha256'] for r in slides).encode()).hexdigest()
    info={'features':str(cache),'normalized_visual_only':True,'semantic_projection_applied':False,'slides':slides,'patches':total,'patients':512,'source_sha256':source_sha,'cache_sha256':sha(cache),'coordinate_sha256':sha(P/'coords.npy'),'owners_sha256':sha(P/'row_owners.npy'),'volatile_cache':True}
    write(P/'CANDIDATE_CACHE.json',info);allq={m:quotas(slides,m) for m in [64,128,256,512,1024]}
    for m,q in allq.items():np.save(P/'selectors'/f'quota_{m}.npy',q)
    # Outcome-blind per-slide complete rankings, truncated by same patient quotas.
    rankings={};order=sorted(range(len(slides)),key=lambda i:slides[i]['n']);projection=(np.random.default_rng(1600).normal(size=(768,32))/np.sqrt(32)).astype(np.float32);projection=torch.tensor(projection,device='cuda')
    for selector in ['GRID','DIVERSE']:
        ranks={}
        for start in range(0,len(order),24):
            subset=order[start:start+24];nn=max(slides[i]['n'] for i in subset);dim=2 if selector=='GRID' else 34;pad=np.full((len(subset),nn,dim),np.nan,np.float32)
            for j,i in enumerate(subset):
                r=slides[i];a=r['offset'];b=a+r['n'];xy=coord[a:b].astype(np.float32);xy=(xy-xy.min(0))/np.maximum(xy.max(0)-xy.min(0),1)
                if selector=='GRID':v=xy
                else:
                    with torch.no_grad():v=(torch.tensor(np.asarray(out[a:b]),device='cuda')@projection).cpu().numpy()
                    v=np.c_[v,xy*.1]
                pad[j,:len(v)]=v
            selected=fps(torch.tensor(pad,device='cuda'),1024)
            for j,i in enumerate(subset):ranks[i]=selected[j,:min(1024,slides[i]['n'])]+slides[i]['offset']
        rankings[selector]=ranks
    rows=[]
    for seed in cfg()['seeds']:
        rand={i:np.random.default_rng(seed+patient_seed(r['patient'],i)).permutation(r['n'])[:1024]+r['offset'] for i,r in enumerate(slides)}
        for selector,ranks in [('RAND',rand)]+list(rankings.items()):
            for m,q in allq.items():
                selected=np.full((len(slides),int(q.max())),-1,np.int64)
                for i,r in enumerate(slides):
                    ids=ranks[i][:q[i]];check_rows(ids,owners,r['patient']);selected[i,:len(ids)]=ids
                    xy=coord[ids];full=coord[r['offset']:r['offset']+r['n']];lo=full.min(0);span=np.maximum(full.max(0)-lo,1);cells=np.minimum(((xy-lo)/span*8).astype(int),7);allcells=np.minimum(((full-lo)/span*8).astype(int),7)
                    coverage=len(np.unique(cells,axis=0))/len(np.unique(allcells,axis=0));rows.append({'selector':selector,'seed':seed,'budget_per_patient':m,'slide_number':i+1,'selected':len(ids),'original':r['n'],'occupied_grid_fraction':coverage,'duplicates':len(ids)-len(np.unique(ids)),'raw_histology_QA':'BLOCKED_NO_RAW_WSI'})
                np.save(P/'selectors'/f'{selector}_seed{seed}_{m}.npy',selected)
    pd.DataFrame(rows).to_csv(R/'G1_COVERAGE.csv',index=False)
    d=parent.load();missing=np.isnan(d['raw_rna']);block=missing[missing.any(1)];extra={'RNA_missing_block_patients':len(block),'RNA_missing_columns':int(block[0].sum()),'all_missing_patients_same_exact_columns':bool(np.all(block==block[0])),'platform_identity':'unconfirmed','CNA_gain_loss':'separate;no semantic routing arm trained while gate blocked'}
    write(R/'MOLECULAR_SOURCE_AUDIT.json',extra);write(P/'PREPARE_DONE.json',{'status':'PASS','source_sha256':source_sha,'cache_sha256':info['cache_sha256'],'seconds':time.time()-started,'no_outcome_labels_used':True});print(json.dumps({'status':'PASS','patches':total,'seconds':time.time()-started}),flush=True)
def patient_seed(patient,slide):return patient*7919+slide*104729
if __name__=='__main__':main()

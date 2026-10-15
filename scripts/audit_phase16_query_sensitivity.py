"""Frozen retrieval diagnostics. No survival labels enter any query or ROI."""
import sys,json,argparse,hashlib,time,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch,h5py
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import spearmanr,rankdata
from src.phase16_diagnostic.core import *
from src.phase16_semantic.core import directory,P as OLD_P
from src.phase16_semantic.prompts import captions,clinical_values,clinical_caption,RNA_caption
from src.phase16_semantic import runtime

def spatial(a,b,s,xy):
    a=np.asarray(a);b=np.asarray(b);common=len(np.intersect1d(a,b));full=xy[s['offset']:s['offset']+s['n']].astype(float);lo=full.min(0);span=np.maximum(full.max(0)-lo,1);diag=np.linalg.norm(span)
    u=xy[a].astype(float);v=xy[b].astype(float)
    def hist(x):g=np.minimum(((x-lo)/span*8).astype(int),7);return np.bincount(g[:,0]*8+g[:,1],minlength=64)/len(x)
    return {'intersection_fraction':common/len(a),'Jaccard':common/(len(a)+len(b)-common),'grid_total_variation':float(abs(hist(u)-hist(v)).sum()/2),
            'centroid_shift':float(np.linalg.norm(u.mean(0)-v.mean(0))/diag),'symmetric_nearest_shift':float((cKDTree(u).query(v)[0].mean()+cKDTree(v).query(u)[0].mean())/2/diag)}

def spatial_audit():
    d=parent.load();meta=base.manifest();xy=np.load(base.P/'coords.npy');records=[]
    for seed in cfg()['seeds']:
        for fold in cfg()['folds']:
            part=parent.splits(d,'DSS',fold);test=set(part['outer_test']);path=directory(seed,fold,'outer');st=torch.load(path/'preprocessing.pt',map_location='cpu',weights_only=False)
            if not np.array_equal(st['test'],part['outer_test']):raise ValueError('retrieval heldout partition')
            for first,second in cfg()['query_pairs']:
                for budget in cfg()['budgets']:
                    aa=np.load(path/f'{first}_{budget}.npy');bb=np.load(path/f'{second}_{budget}.npy');patients={}
                    for i,s in enumerate(meta['slides']):
                        patient=s['patient']
                        if patient not in test:continue
                        a=aa[i][aa[i]>=0];b=bb[i][bb[i]>=0]
                        if len(a)!=len(b) or len(a)!=len(np.unique(a)):raise ValueError('ROI count/duplicates')
                        r=spatial(a,b,s,xy);patients.setdefault(patient,[]).append((len(a),r))
                    for patient,values in patients.items():
                        w=np.array([v[0] for v in values],float);w/=w.sum();scores={k:float(w@np.array([v[1][k] for v in values])) for k in values[0][1]}
                        hamming=float(np.mean(st['RNA_states'][patient]!=st['RNA_states'][st['RNA_permutation'][patient]]))
                        records.append({'seed':seed,'fold':fold,'patient_index':patient,'comparison':first+'-'+second,'budget':budget,'RNA_state_hamming_fraction':hamming,
                                        'missing_block':bool(np.isnan(d['raw_rna'][patient]).any()),**scores})
    private=pd.DataFrame(records);private.to_csv(P/'SPATIAL_PATIENT_PRIVATE.csv',index=False)
    metrics=['intersection_fraction','Jaccard','grid_total_variation','centroid_shift','symmetric_nearest_shift']
    out=[]
    for keys,g in private.groupby(['seed','fold','comparison','budget']):
        row=dict(zip(['seed','fold','comparison','budget'],keys));row['patients']=len(g)
        for k in metrics:row[k+'_mean']=g[k].mean();row[k+'_median']=g[k].median();row[k+'_p10']=g[k].quantile(.1);row[k+'_p90']=g[k].quantile(.9)
        out.append(row)
    pd.DataFrame(out).to_csv(R/'HELDOUT_SPATIAL_FOLD.csv',index=False)
    out=[]
    for keys,g in private.groupby(['seed','comparison','budget']):
        if len(g)!=512 or g.patient_index.nunique()!=512:raise ValueError('heldout spatial coverage')
        row=dict(zip(['seed','comparison','budget'],keys));row.update(patients=512)
        for k in metrics:row[k+'_mean']=g[k].mean();row[k+'_median']=g[k].median();row[k+'_p10']=g[k].quantile(.1);row[k+'_p90']=g[k].quantile(.9)
        row['state_distance_vs_replaced_fraction_spearman']=float(spearmanr(g.RNA_state_hamming_fraction,1-g.intersection_fraction).statistic) if g.RNA_state_hamming_fraction.nunique()>1 else 0.
        out.append(row)
    pd.DataFrame(out).to_csv(R/'HELDOUT_SPATIAL_SEED.csv',index=False)
    write(R/'SPATIAL_AUDIT.json',{'status':'PASS','heldout_patient_records':len(private),'patients_per_arm_seed':512,'seeds':3,'no_survival_labels_read':True,'raw_image_QA':'BLOCKED','aggregation':'patient mean; multi-slide metrics weighted by selected patch count; Jaccard is weighted slide mean, not a single global Jaccard'})
    print('SPATIAL_AUDIT_PASS',flush=True)

def prepare(model):
    info=base.manifest();raw=Path(info['features']);projected=Path('/dev/shm/phase16_conch_z_projected.npy');meta=json.loads((OLD_P/'PROJECTION_CACHE.json').read_text())
    if not raw.exists():
        arr=np.lib.format.open_memmap(raw,mode='w+',dtype=np.float32,shape=(info['patches'],768));inv=json.loads((base.P/'ASSET_INVENTORY.json').read_text())
        rows=sorted(inv['chosen_conch'],key=lambda x:(x['patient_index'],x['stem']))
        for n,(r,s) in enumerate(zip(rows,info['slides'])):
            if sha(r['path'])!=s['source_h5_sha256']:raise ValueError('source H5 changed')
            with h5py.File(r['path']) as h:x=h['features'][:].reshape(-1,768).astype(np.float32)
            x/=np.linalg.norm(x,axis=1)[:,None];arr[s['offset']:s['offset']+s['n']]=x
        arr.flush()
    if sha(raw)!=info['cache_sha256']:raise ValueError('rebuilt original visual cache differs')
    if not projected.exists():
        x=np.load(raw,mmap_mode='r');out=np.lib.format.open_memmap(projected,mode='w+',dtype=np.float32,shape=x.shape);proj=model.visual.proj_contrast.detach().float()
        with torch.no_grad():
            for start in range(0,len(x),8192):out[start:start+8192]=F.normalize(torch.tensor(np.asarray(x[start:start+8192]),device='cuda')@proj,dim=-1).cpu().numpy()
        out.flush()
    if sha(projected)!=meta['cache_sha256']:raise ValueError('official projected cache not exact')
    print('PROJECTION_CACHE_EXACT_REBUILT',flush=True);return np.load(projected,mmap_mode='r')

def query_metrics(image,a,b,budget):
    x=image@a.T;y=image@b.T
    cosine=np.clip(np.sum(a.cpu().numpy()*b.cpu().numpy(),1),-1,1)
    top=max(1,int(np.ceil(budget/32)));ix=torch.topk(x,top,dim=0).indices.cpu().numpy();iy=torch.topk(y,top,dim=0).indices.cpu().numpy()
    aa=x.cpu().numpy();bb=y.cpu().numpy();rho=[]
    for k in range(32):rho.append(float(np.corrcoef(rankdata(aa[:,k]),rankdata(bb[:,k]))[0,1]))
    row={'query_cosine_mean':float(cosine.mean()),'query_cosine_min':float(cosine.min()),'patch_rank_spearman_mean':float(np.mean(rho)),
         'per_concept_top_intersection_mean':float(np.mean([len(np.intersect1d(ix[:,k],iy[:,k]))/top for k in range(32)]))}
    return row,x,y

def semantic_audit():
    base.seed_all(31);runtime.R=R/'official_runtime';model=runtime.official_model();projection=prepare(model);model=model.half();cache=runtime.TextCache(model)
    capsule=np.load(OLD_P/'QUERY_TEXT_REPLAY_CACHE.npz')
    cache.db.executemany('INSERT OR IGNORE INTO embeddings VALUES(?,?,?)',[(str(k),'',v.astype(np.float32).tobytes()) for k,v in zip(capsule['keys'],capsule['vectors'])]);cache.db.commit()
    d=parent.load();meta=base.manifest();xy=np.load(base.P/'coords.npy');concepts=json.loads((ROOT/'configs/phase6a_titan_crc_morphology_concepts.json').read_text())['concepts'];static_words=[runtime.cfg()['concept_template'].format(morphology=c['text']) for c in concepts];fixed=cache.encode(static_words)
    rows=[];cfrows=[];numerical=[];evidence=[];states_out=[];replay=[]
    for fold in cfg()['folds']:
        parts=parent.splits(d,'DSS',fold);fit,test=parts['outer_train'],parts['outer_test'];texts,st=captions(d,fit,test,31,fold);values,_=clinical_values(d,fit)
        selected=set(np.random.default_rng(cfg()['counterfactual_seed']+fold).permutation(test)[:6].tolist())
        selectors=sorted(set(s for pair in cfg()['query_pairs'] for s in pair))
        words=[word for s in selectors for i in test for word in texts[s][i]];encoded=cache.encode(words).reshape(len(selectors),len(test),32,768);embedding={s:{int(i):encoded[k,j] for j,i in enumerate(test)} for k,s in enumerate(selectors)}
        arrays={(s,m):np.load(directory(31,fold,'outer')/f'{s}_{m}.npy') for s in selectors for m in cfg()['budgets']}
        for patient in test:
            if patient in selected:
                varied=[];labels=[]
                for group in range(8):
                    for val in ['low','high']:
                        state=st['RNA_states'][patient].astype('U16').copy();state[group]=val;context=clinical_caption(values[patient])+' '+RNA_caption(state)
                        varied.extend([v+' '+context for v in static_words]);labels.append((group,val))
                cf=cache.encode(varied).reshape(8,2,32,768)
                with torch.no_grad():fresh=F.normalize(model.encode_text(model.tokenizer(texts['CLIN_RNA_TEXT'][patient]).cuda()).float(),dim=-1).cpu().numpy()
            state_row={'patient_index':int(patient),'fold':fold,**{'state'+str(k):str(v) for k,v in enumerate(st['RNA_states'][patient])}};states_out.append(state_row)
            for slide,s in enumerate(meta['slides']):
                if s['patient']!=patient:continue
                offset=s['offset'];image=torch.tensor(np.asarray(projection[offset:offset+s['n']]),device='cuda');static_scores=(image@torch.tensor(fixed.T,device='cuda')).cpu().numpy()
                for first,second in cfg()['query_pairs']:
                    a=torch.tensor(embedding[first][int(patient)],device='cuda');b=torch.tensor(embedding[second][int(patient)],device='cuda')
                    for budget in cfg()['budgets']:
                        row,xx,yy=query_metrics(image,a,b,budget);rows.append({'patient_index':int(patient),'slide':slide,'fold':fold,'comparison':first+'-'+second,'budget':budget,**row})
                for selector in ['FIXED_TEXT','CLIN_TEXT','RNA_TEXT','CLIN_RNA_TEXT','JOINT_WRONG_RNA']:
                    for budget in cfg()['budgets']:
                        ids=arrays[selector,budget][slide];ids=ids[ids>=0];evidence.append({'patient_index':int(patient),'fold':fold,'selector':selector,'budget':budget,'C10_static_evidence':float(static_scores[ids-offset,9].mean()),'C10_full_evidence':float(static_scores[:,9].mean())})
                if patient in selected:
                    true=torch.tensor(embedding['CLIN_RNA_TEXT'][int(patient)],device='cuda');fr=torch.tensor(fresh,device='cuda')
                    for budget in cfg()['budgets']:
                        quota=int(np.load(base.P/'selectors'/f'quota_{budget}.npy')[slide]);old=arrays['CLIN_RNA_TEXT',budget][slide,:quota];base_rank=runtime.round_robin(image@true.T,quota)+offset;fresh_rank=runtime.round_robin(image@fr.T,quota)+offset
                        numerical.append({'patient_index':int(patient),'fold':fold,'slide':slide,'budget':budget,'query_abs_error':float(abs(fresh-embedding['CLIN_RNA_TEXT'][int(patient)]).max()),**spatial(base_rank,fresh_rank,s,xy)})
                        replay.append({'patient_index':int(patient),'fold':fold,'budget':budget,'intersection_fraction':len(np.intersect1d(base_rank,old))/quota})
                    for group in range(8):
                        low=torch.tensor(cf[group,0],device='cuda');high=torch.tensor(cf[group,1],device='cuda')
                        for budget in cfg()['budgets']:
                            row,ll,hh=query_metrics(image,low,high,budget);quota=int(np.load(base.P/'selectors'/f'quota_{budget}.npy')[slide]);a=runtime.round_robin(ll,quota)+offset;b=runtime.round_robin(hh,quota)+offset;cfrows.append({'patient_index':int(patient),'fold':fold,'slide':slide,'group':group,'budget':budget,**row,**spatial(a,b,s,xy)})
                            if budget==512:
                                folder=P/'raw_QA';folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'case_{fold}_{int(patient)}_group{group}.npz',low=a,high=b,coords=xy[np.union1d(a,b)],slide=slide)
                del image
        print('QUERY_FOLD_DONE',fold,'NEW_TEXT',cache.generated,'CACHED',cache.reused,flush=True)
    for name,data in [('QUERY_PATIENT_PRIVATE',rows),('COUNTERFACTUAL_PRIVATE',cfrows),('NUMERICAL_PRIVATE',numerical),('STATIC_EVIDENCE_PRIVATE',evidence),('STATES_PRIVATE',states_out),('REPLAY_PRIVATE',replay)]:pd.DataFrame(data).to_csv(P/(name+'.csv'),index=False)
    query=pd.DataFrame(rows);cf=pd.DataFrame(cfrows);noise=pd.DataFrame(numerical)
    def summarize(table,groups,filename):
        numeric=[k for k in table.select_dtypes(include='number').columns if k not in groups+['patient_index','slide']]
        result=table.groupby(groups)[numeric].mean().reset_index();result['units']=table.groupby(groups).size().to_numpy();result.to_csv(R/filename,index=False)
    # Average multiple slides within patient before the public fold summaries.
    q=query.groupby(['patient_index','fold','comparison','budget'],as_index=False).mean(numeric_only=True);c=cf.groupby(['patient_index','fold','group','budget'],as_index=False).mean(numeric_only=True)
    summarize(q,['fold','comparison','budget'],'QUERY_SENSITIVITY_FOLD.csv');summarize(c,['fold','group','budget'],'COUNTERFACTUAL_FOLD.csv');summarize(noise,['fold','budget'],'NUMERICAL_CONTROL.csv')
    ev=pd.DataFrame(evidence).groupby(['patient_index','fold','selector','budget'],as_index=False).mean(numeric_only=True);ev.to_csv(P/'PATIENT_MORPHOLOGY_EVIDENCE_PRIVATE.csv',index=False)
    state=pd.DataFrame(states_out);joined=cf.merge(state,on=['patient_index','fold']);grouped=[]
    for (group,budget),g in joined.groupby(['group','budget']):
        for val in ['low','typical','high','unmeasured']:
            sub=g[g['state'+str(group)]==val]
            if sub.patient_index.nunique()>=10:grouped.append({'group':group,'budget':budget,'original_state':val,'patients':sub.patient_index.nunique(),'intersection_fraction':sub.intersection_fraction.mean(),'grid_total_variation':sub.grid_total_variation.mean()})
    pd.DataFrame(grouped).to_csv(R/'STATE_STRATIFIED_COUNTERFACTUAL.csv',index=False)
    write(R/'QUERY_AUDIT.json',{'status':'PASS','query_seed':31,'heldout_patients':512,'counterfactual_patients':30,'RNA_states':8,'text_vectors_generated':cache.generated,'cache_vectors_reused':cache.reused,'raw_visual_cache_exact':True,'official_projection_cache_exact':True,'survival_labels_used':False,'raw_image_QA':'BLOCKED','cached_ROI_replay_mean_intersection':float(pd.DataFrame(replay).intersection_fraction.mean()),'numeric_query_max_abs_error':float(noise.query_abs_error.max()),'query_source_sha256':sha(Path(__file__))})
    print('QUERY_AUDIT_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spatial-only',action='store_true');a=p.parse_args();lock()
    query_lock=P/'QUERY_SOURCE_LOCK.json';identity={'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(CONFIG)}
    if query_lock.exists() and json.loads(query_lock.read_text())!=identity:raise ValueError('query audit source changed after lock')
    write(query_lock,identity);spatial_audit()
    if not a.spatial_only:semantic_audit()

"""Independent sampled geometry/aggregation and source morphology associations."""
import sys,json,hashlib,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch,h5py
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from src.phase16_diagnostic.core import *
from src.phase16_semantic.core import P as OLD_P,directory
from src.phase16_semantic.prompts import captions

def geometry():
    meta=base.manifest();inv=json.loads((base.P/'ASSET_INVENTORY.json').read_text());con=sorted(inv['chosen_conch'],key=lambda r:(r['patient_index'],r['stem']));uni=[r for r in inv['h5'] if r['kind']=='UNI2H_CANDIDATE'];selections=np.load(P/'PAIRED_VISUAL_SELECTION_PRIVATE.npz');xy=np.load(base.P/'coords.npy');d=parent.load();records=[];queue=[]
    slides=sorted(np.random.default_rng(1610719).choice(len(con),30,replace=False))
    for i in slides:
        s=meta['slides'][i];c=con[i];u=sorted([r for r in uni if r['stem']==c['stem'] and r['patient_index']==s['patient']],key=lambda r:len(r['path']))[0];m=np.load(base.P/'coordinate_maps'/f'slide_{i+1:04d}.npz');take=selections[str(i)]
        with h5py.File(c['path']) as h:cc=h['coords'][:].reshape(-1,2).astype(float);cx=h['features'][:].reshape(-1,768).astype(float)
        with h5py.File(u['path']) as h:uu=h['coords'][:].reshape(-1,2).astype(float);ux=h['features'][:].reshape(-1,1536).astype(float)
        ca,ua=c['coordinate_attrs'],u['coordinate_attrs'];cs,us=ca['patch_size_level0'],ua['patch_size_level0'];width,height=ca['level0_width'],ca['level0_height'];errors=[]
        # Recompute rectangle intersections from coordinates, without loading
        # the training map's edge weights into the expected result.
        targets=np.random.default_rng(161080+i).choice(take,min(8,len(take)),replace=False)
        for row in targets:
            low=np.maximum(cc[row],uu);high=np.minimum(np.minimum(cc[row]+cs,uu+us),[width,height]);area=np.maximum(high-low,0).prod(1);jj=np.flatnonzero(area>0);old=m['uni2_row'][m['conch_row']==row];aa=m['intersection_area'][m['conch_row']==row]
            if not np.array_equal(jj,old) or not np.array_equal(area[jj],aa):raise ValueError('independent raw coordinate geometry mismatch')
            expected=np.average(ux[jj]/np.linalg.norm(ux[jj],axis=1)[:,None],axis=0,weights=area[jj]);expected/=np.linalg.norm(expected)
            original=ux[old]/np.linalg.norm(ux[old],axis=1)[:,None];matrix=csr_matrix(aa.reshape(1,-1));actual=np.asarray(matrix@original).reshape(-1)/aa.sum();actual/=np.linalg.norm(actual);errors.append(float(abs(expected-actual).max()))
        records.append({'anonymous_slide_number':int(i+1),'sampled_footprints':len(targets),'geometry_exact':True,'independent_sparse_aggregation_max_error':max(errors,default=0),'raw_image_QA':False})
        queue.append({'case':len(queue)+1,'anonymous_slide_number':int(i+1),'conch_source':c['path'],'uni2_source':u['path'],'candidate_selected_local_rows':take[:8].tolist(),'level0_coords':cc[take[:8]].tolist(),'raw_WSI_available':False})
    pd.DataFrame(records).to_csv(R/'GEOMETRY_INDEPENDENT_SAMPLES.csv',index=False);write(P/'RAW_WSI_REVIEW_QUEUE_PRIVATE.json',queue);return records

def morphology():
    d=parent.load();manifest=json.loads(Path('/root/phase7c_morphology_conditional_audit_v1/manifest.json').read_text());source=next(r for r in manifest['sources'] if r['path'].endswith('coadread_tcga/data_clinical_patient.txt'));path=Path(source['path'])
    if sha(path)!=source['sha256']:raise ValueError('independent morphology source changed')
    table=pd.read_csv(path,sep='\t',comment='#',dtype=str).fillna('').drop_duplicates('PATIENT_ID').set_index('PATIENT_ID').reindex(d['patient_ids'].astype(str)).fillna('');diag=table.HISTOLOGICAL_DIAGNOSIS.str.lower();valid=diag.str.contains('adenocarcinoma').to_numpy();label=diag.str.contains('mucinous').to_numpy().astype(int);ix=np.flatnonzero(valid)
    if len(ix)!=505 or label[ix].sum()!=66:raise ValueError('morphology label coverage changed')
    ev=pd.read_csv(P/'PATIENT_MORPHOLOGY_EVIDENCE_PRIVATE.csv');score={}
    for (selector,budget),g in ev.groupby(['selector','budget']):x=np.full(512,np.nan);x[g.patient_index.astype(int)]=g.C10_static_evidence;score[selector,int(budget)]=x
    records=[];comparisons=[];rng=np.random.default_rng(1610831);keys=list(score);draws=[];strata=np.char.add(d['cancer_type'].astype(str),label.astype(str));groups=[ix[strata[ix]==s] for s in np.unique(strata[ix])]
    for start in range(0,5000):
        sample=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);draws.append([roc_auc_score(label[sample],score[k][sample]) for k in keys])
    draws=np.asarray(draws)
    for j,k in enumerate(keys):lo,hi=np.quantile(draws[:,j],[.025,.975]);records.append({'selector':k[0],'budget':k[1],'n':505,'positive':66,'AUC':roc_auc_score(label[ix],score[k][ix]),'CI_low':lo,'CI_high':hi,'label':'source HISTOLOGICAL_DIAGNOSIS;not patch truth'})
    for budget in [128,512]:
        for a,b in [('CLIN_RNA_TEXT','JOINT_WRONG_RNA'),('CLIN_RNA_TEXT','CLIN_TEXT'),('CLIN_TEXT','FIXED_TEXT')]:
            aa,bb=keys.index((a,budget)),keys.index((b,budget));lo,hi=np.quantile(draws[:,aa]-draws[:,bb],[.025,.975]);comparisons.append({'comparison':a+'-'+b,'budget':budget,'delta_AUC':roc_auc_score(label[ix],score[a,budget][ix])-roc_auc_score(label[ix],score[b,budget][ix]),'CI_low':lo,'CI_high':hi,'scope':'exploratory patient-level association;not regional validation'})
    pd.DataFrame(records).to_csv(R/'MUCINOUS_PATIENT_ASSOCIATION.csv',index=False);pd.DataFrame(comparisons).to_csv(R/'MUCINOUS_PAIRED_DIAGNOSTIC.csv',index=False)
    write(R/'MORPHOLOGY_SOURCE_AUDIT.json',{'status':'PASS','patients':505,'positive':66,'unknown':7,'source_field':'supplementary.HISTOLOGICAL_DIAGNOSIS','source_sha256':source['sha256'],'no_scores_used_as_labels':True,'patch_level_truth':False,'DSS_labels_used':False})

def decomposition():
    d=parent.load();db=sqlite3.connect('file:/dev/shm/phase16_conch_z_text.sqlite?mode=ro',uri=True);weight=sha(Path('/root/phase16_conch_z/model/model.safetensors'));records=[]
    for fold in cfg()['folds']:
        part=parent.splits(d,'DSS',fold);texts,_=captions(d,part['outer_train'],part['outer_test'],31,fold)
        for a,b in cfg()['query_pairs']:
            ratios=[];norms=[]
            for i in part['outer_test']:
                arrays=[]
                for selector in [a,b]:
                    keys=[hashlib.sha256((weight+'|official_half_float32_normalized_v1|'+word).encode()).hexdigest() for word in texts[selector][i]];arrays.append(np.stack([np.frombuffer(db.execute('SELECT value FROM embeddings WHERE key=?',(key,)).fetchone()[0],np.float32) for key in keys]))
                delta=arrays[0]-arrays[1];total=np.sum(delta*delta);res=delta-delta.mean(0);ratios.append(float(np.sum(res*res)/total) if total else 0);norms.append(float(np.sqrt(total/32)))
            records.append({'fold':fold,'comparison':a+'-'+b,'patients':len(ratios),'concept_specific_delta_energy_fraction':np.mean(ratios),'common_delta_energy_fraction':1-np.mean(ratios),'mean_query_L2_change':np.mean(norms),'scope':'posthoc descriptive vector decomposition;not a causal semantic measure'})
    pd.DataFrame(records).to_csv(R/'QUERY_DELTA_DECOMPOSITION.csv',index=False)

if __name__=='__main__':
    samples=geometry();morphology();decomposition();write(R/'SOURCES_INDEPENDENT_AUDIT.json',{'status':'PASS','coordinate_slides_sampled':30,'footprints_recomputed':sum(r['sampled_footprints'] for r in samples),'raw_histology_QA':False,'sample_scope_not_full_dataset':True});print('SOURCES_INDEPENDENT_AUDIT_PASS',flush=True)

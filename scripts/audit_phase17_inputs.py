"""Frozen source hashes, visual units and wrong-context strata, no model tuning."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from src.phase17_ssd.data import P,R,write,parent,source
from src.phase10_trimodal.data import digest

def main():
    v=P/'visual';raw=np.load(v/'u.npy',mmap_mode='r');mask=np.load(v/'mask.npy');owner=np.load(v/'owner.npy');old=np.load('/root/phase16_prompt_rna_diagnostic_private/PAIRED_VISUAL_SELECTION_PRIVATE.npz');rows=np.load(v/'local_row.npy');norms=[]
    ready=json.loads((v/'READY.json').read_text())
    for name,value in ready['hashes'].items():assert digest(v/(name+'.npy'))==value
    assert digest(parent.DATA)==ready['dataset_sha256']=='4aaeb4cc22374f32615148605d824abed679fcb957c64286b6a8e44f12a294a1'
    for k in range(len(owner)):
        assert set(rows[k,mask[k]])==set(old[str(k)])
        norms.extend(np.linalg.norm(np.asarray(raw[k,mask[k]],np.float32),axis=1).tolist())
    norms=np.asarray(norms);assert np.isfinite(norms).all() and norms.min()>.99 and norms.max()<1.01
    counts=np.bincount(owner,weights=mask.sum(1));assert len(counts)==512 and counts.min()>0 and counts.max()<=512
    protected=[]
    paths=['reports/phase16_semantic_routing/ARTIFACT_MANIFEST.json','reports/phase16_semantic_routing/semantic_completion/ARTIFACT_MANIFEST.json','reports/phase16_prompt_rna_diagnostic/ARTIFACT_MANIFEST.json','reports/phase16r_light_baseline_rna_compact/ARTIFACT_MANIFEST.json']
    for path in paths:
        m=json.loads((ROOT/path).read_text());n=0
        for row in m['public_files']:
            if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('protected history changed '+row['path'])
            n+=1
        protected.append({'manifest':path,'sha256':digest(ROOT/path),'files_verified':n})
    write(R/'HISTORICAL_IMMUTABILITY_AUDIT.json',{'status':'PASS','manifests':protected})
    write(R/'VISUAL_CACHE_INDEPENDENT_AUDIT.json',{'status':'PASS','selected_patch_ID_sets_rechecked':len(owner),'patients':512,'patches':int(mask.sum()),'minimum_norm':float(norms.min()),'maximum_norm':float(norms.max()),'zero_vectors':int((norms==0).sum()),'minimum_patient_patches':int(counts.min()),'maximum_patient_patches':int(counts.max()),'normalization_precision':'float16 persisted;float32 network input','raw_WSI_QA':False})
    d=parent.load();tab=[]
    for stage in ['inner','outer']:
        for p in (P/stage).glob('A10_*.pt'):
            q=torch.load(p,map_location='cpu',weights_only=False);mapping=q['wrong_map']
            for part,ix in [('fit',q['fit']),('evaluation',q['validation_or_test'])]:
                jj=mapping[ix];assert set(jj)==set(ix) and not np.any(jj==ix)
                cancer=d['cancer_type'].astype(str);ss=d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str)
                tab.append({'stage':stage,'seed':q['seed'],'fold':q['fold'],'partition':part,'patients':len(ix),'self_pairs':int(np.sum(jj==ix)),'cross_partition_pairs':int(np.sum(~np.isin(jj,ix))),'same_cancer_fraction':float(np.mean(cancer[jj]==cancer[ix])),'same_raw_Stage_fraction':float(np.mean(ss[jj]==ss[ix])),'Stage_fallback_pairs':int(np.sum(ss[jj]!=ss[ix])),'cancer_fallback_pairs':int(np.sum(cancer[jj]!=cancer[ix]))})
    if tab:pd.DataFrame(tab).to_csv(R/'WRONG_CONTEXT_STRATA_AUDIT.csv',index=False)
    print('INPUT_AND_HISTORY_AUDIT_PASS',len(protected),len(owner),flush=True)

if __name__=='__main__':main()

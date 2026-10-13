"""Rebuild clinical/endpoints from original tables, check representation sources."""
import sys,json,re,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.phase10_trimodal.data import write_json,digest
from src.phase10_trimodal.features import prepare,clinical,CLINICAL
from src.phase10_trimodal.train import inner_split

def table(p,ids):
 t=pd.read_csv(p,sep='\t',comment='#',dtype=str);t['pid']=t['PATIENT_ID'].str.extract(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')[0];return t.drop_duplicates('pid').set_index('pid').reindex(ids)

def main():
 root=Path(__file__).resolve().parents[1];report=root/'reports/phase10_trimodal_mechanism_audit';d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));ids=d['patient_ids'].astype(str);base=Path('/root/autodl-tmp/diffusion_data/data/data_8');paths=[base/'coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt',base/'coadread_tcga/data_clinical_patient.txt'];a,b=[table(p,ids) for p in paths];checks={}
 for name in CLINICAL:
  v=a[name] if name in a else pd.Series(np.nan,index=ids)
  if name!='AGE' and name in b:v=v.where(v.notna(),b[name])
  v=v.fillna('UNKNOWN').astype(str).to_numpy();ok=np.array_equal(v,d['clinical_'+name].astype(str));assert ok;checks[name]={'exact_source_agreement':True,'supplement_fill_only':name!='AGE','unknown_count':int(sum(v=='UNKNOWN'))}
 t=pd.to_numeric(a['DSS_MONTHS'],errors='coerce').to_numpy();e=a['DSS_STATUS'].str.extract(r'^\s*([01])')[0].astype(float).to_numpy();assert np.array_equal(t,d['DSS_time']) and np.array_equal(e,d['DSS_event']);checks['DSS']={'exact_time_and_status_agreement':True,'status_dictionary':a['DSS_STATUS'].value_counts().to_dict()}
 native=Path('/root/phase6a_titan_runtime/cache_v1/patient_representations.npz');z=dict(np.load(native,allow_pickle=True));assert np.array_equal(z['patient_ids'].astype(str),ids) and np.array_equal(z['native'],d['titan']) and np.array_equal(z['concept_cosine'],d['static']);checks['TITAN_native_S32']={'exact_agreement':True,'source_sha256':digest(native),'native_key':'native','S32_key':'concept_cosine','native_not_text_aligned':True}
 groups=json.loads((root/'configs/phase10_molecular_groups.json').read_text());tr=np.flatnonzero(d['folds']!=1);f,ha=prepare(d,tr,groups);d2=copy.deepcopy(d);te=np.flatnonzero(d['folds']==1)
 for key in ['rna_genes','cna_genes','static','titan']:d2[key][te]=1e12
 d2['DSS_event'][te]=1-d2['DSS_event'][te];d2['DSS_time'][te]+=1e9;g,hb=prepare(d2,tr,groups);assert ha==hb and all(np.array_equal(f[k][tr],g[k][tr]) for k in f)
 ia=inner_split(d,tr,7302);ib=inner_split(d2,tr,7302);assert all(np.array_equal(x,y) for x,y in zip(ia,ib));checks['actual_outer_test_perturbation']={'state_hashes_identical':True,'train_tensors_identical':True,'inner_selection_split_identical':True,'test_features_and_labels_perturbed':True,'model_weights_invariance':'same train/inner-val tensors, RNG and schedule; checkpoint replay independently verified; no repeat training performed'}
 try:clinical(d,tr,requested_fields=['normal_score'],provenance={'normal_score':{'source_field':'DSS_STATUS','source_kind':'diagnosis_clinical','label_derived':True}})
 except ValueError:checks['constructor_red_team']={'status':'PASS','disguised_test_label_alias_rejected':True}
 else:raise AssertionError('future input admitted')
 write_json(report/'raw_clinical_representation_audit.json',{'status':'PASS','checks':checks,'source_sha256':{p.name+str(i):digest(p) for i,p in enumerate(paths)}});print('PASS raw clinical, endpoints, TITAN, test perturbation and constructor red team',flush=True)
if __name__=='__main__':main()

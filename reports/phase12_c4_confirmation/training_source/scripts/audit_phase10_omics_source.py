import sys,json,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.phase10_trimodal.data import digest,write_json
repo=Path(__file__).resolve().parents[1];d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));ids=d['patient_ids'].astype(str);base=Path('/root/autodl-tmp/diffusion_data/data/data_8/coadread_tcga_pan_can_atlas_2018');result={}
for typ,file in [('RNA','data_mrna_seq_v2_rsem.txt'),('CNA','data_log2_cna.txt')]:
 tab=pd.read_csv(base/file,sep='\t',comment='#',low_memory=False);source=tab.iloc[:,2:].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32);col={str(c):j for j,c in enumerate(tab.columns[2:])};columns={}
 for name,j in col.items():
  pid=name[:12]
  if pid in set(ids) and name[13:15]=='01':columns.setdefault(pid,[]).append(j)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);raw=np.stack([np.nanmean(source[:,columns[pid]],axis=1) for pid in ids])
 expected=d['raw_rna' if typ=='RNA' else 'raw_cna'];same=np.allclose(raw,expected,equal_nan=True,rtol=0,atol=0)
 if not same:raise ValueError(typ+' raw cache not exact')
 sym=tab.iloc[:,0].fillna('').astype(str).str.strip().str.upper().to_numpy(dtype='U64');valid=sorted(set(s for s in sym if s and s!='NAN'));gene=raw.copy()
 if typ=='RNA':
  if np.any(gene[np.isfinite(gene)]<=-1):raise ValueError('RNA log domain');
  gene=np.log2(gene+1)
 loc={}
 for j,s in enumerate(sym):loc.setdefault(s,[]).append(j)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);collapsed=np.stack([np.nanmean(gene[:,loc[s]],axis=1) for s in valid],axis=1).astype(float)
 key='rna' if typ=='RNA' else 'cna'
 expected_gene=d[key+'_genes'];exact=np.array_equal(np.asarray(valid),d[key+'_gene_symbols']) and np.allclose(collapsed,expected_gene,equal_nan=True,rtol=0,atol=0)
 precision={}
 if not exact:
  if typ!='RNA' or not np.array_equal(np.asarray(valid),d[key+'_gene_symbols']) or not np.array_equal(np.isnan(collapsed),np.isnan(expected_gene)):raise ValueError('symbol/NA or CNA cache mismatch')
  delta=np.abs(collapsed-expected_gene);ulp=np.abs(np.spacing(expected_gene.astype(np.float32))).astype(float);maxulp=float(np.nanmax(delta/ulp))
  precision={'max_abs_difference':float(np.nanmax(delta)),'max_float32_ULP':maxulp,'changed_finite_entries':int(np.sum(np.isfinite(delta)&(delta>0))),'diagnosis':'float32 log2 kernel rounding differs from historical cache; raw matrix, symbols, NA and transform identical; historical cache retained, no formal feature alteration'}
  print(json.dumps(precision),flush=True)
  ref=np.log2((raw+np.float32(1)).astype(np.float64)).astype(np.float32)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);ref=np.stack([np.nanmean(ref[:,loc[s]],axis=1) for s in valid],axis=1).astype(float)
  rd=np.abs(ref-expected_gene);precision['float64_log_reference_max_ULP']=float(np.nanmax(rd/ulp));precision['float64_log_reference_max_abs']=float(np.nanmax(rd));precision['float64_log_reference_exact_entries']=int(np.sum(np.isfinite(rd)&(rd==0)))
  if precision['float64_log_reference_max_ULP']>1:raise ValueError('cached RNA log differs from correctly rounded reference by more than one float32 ULP')
 result[typ]={'status':'PASS_WITH_ROUNDING_NOTE' if precision else 'PASS','source_sha256':digest(base/file),'source_rows':len(tab),'raw_shape':list(raw.shape),'collapsed_shape':list(collapsed.shape),'exact_raw_cache_agreement':True,'exact_collapsed_cache_agreement':bool(exact),'precision_audit':precision,'removed_source_rows':len(sym)-len(valid),'duplicate_symbol_rows':len(sym)-len(valid)-int(sum(not s or s=='NAN' for s in sym)),'symbol_free_rows':int(sum(not s or s=='NAN' for s in sym)),'primary_01_only':True,'missing_counts_by_patient':{str(int(a)):int(b) for a,b in zip(*np.unique(np.isnan(raw).sum(1),return_counts=True))},'infinite_values':int(np.isinf(raw).sum())}
 # Audit NA pattern source; do not infer confirmed platform identity.
 if typ=='RNA':result[typ]['NA_source']='exact original PanCancer RNA table; GAII mechanism compatible, patient aliquot platform not independently mapped'
foldp=repo/'outputs/phase_m1r/fixed_patient_folds.csv'
if not foldp.exists():foldp=Path('/root/phase10_trimodal_private/cache/private_fixed_patient_folds.csv')
if foldp.exists():
 f=pd.read_csv(foldp);ic='patient_id' if 'patient_id' in f else 'case_id';fc='fold' if 'fold' in f else 'outer_fold';mp=dict(zip(f[ic].astype(str),f[fc]))
 ok=all(mp.get(pid)==fold for pid,fold in zip(ids,d['folds']))
 if not ok:raise ValueError('original fold assignments differ')
 result['original_folds']={'status':'PASS','source_sha256':digest(foldp),'exact_assignment_agreement':True}
else:result['original_folds']={'status':'BLOCKED','reason':'original fold CSV not on this checkout'}
write_json(repo/'reports/phase10_trimodal_mechanism_audit/raw_source_provenance.json',result);print(json.dumps(result),flush=True)

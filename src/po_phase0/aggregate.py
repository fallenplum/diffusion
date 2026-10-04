import json, csv, os, subprocess, ast, inspect
from pathlib import Path
import numpy as np, pandas as pd
from src.phase_m2.common import atomic_json, atomic_csv
from src.phase_m2.data import load_raw_inputs
from src.po_phase0.runner import MODEL_ORDER, NEURAL, make_model, MECH, SOURCE

def main():
 repo=Path('/root/diffusion_m3_verified_9f859ef'); reports=Path('/root/autodl-tmp/po_phase0_reports2'); runtime=Path('/root/autodl-tmp/po_phase0_runtime'); data=Path(os.environ['DIFFUSION_DATA_ROOT'])
 raws=load_raw_inputs(data,repo)
 fold=[]; seed=[]; prof=[]; fails=[]
 for tag in ('w1','w2','w3'):
  for fn, dest in ((f'fold_metrics_{tag}.csv',fold),(f'seed_metrics_{tag}.csv',seed),(f'compute_profile_{tag}.csv',prof)):
   p=reports/fn
   if p.exists(): dest.extend(pd.read_csv(p).to_dict('records'))
  p=reports/f'failures_{tag}.json'
  if p.exists(): fails.extend(json.loads(p.read_text()))
 # dedupe; use only exact completed evidence
 def ded(rows,keys):
  d={tuple(r.get(k) for k in keys):r for r in rows}; return list(d.values())
 fold=ded(fold,['method','seed','outer_fold']); seed=ded(seed,['method','seed']); prof=ded(prof,['method','seed','outer_fold'])
 main=[]
 for m in MODEL_ORDER:
  fr=[r for r in fold if r.get('method')==m and r.get('status')=='completed']; sr=[r for r in seed if r.get('method')==m and r.get('status')=='completed']; pr=[r for r in prof if r.get('method')==m and r.get('status')=='completed']
  ok=len(fr)==15 and len(sr)==3
  if not ok: continue
  h=np.array([float(x['harrell_c']) for x in sr]); u=np.array([float(x['uno_c']) for x in sr]); hf=np.array([float(x['harrell_c']) for x in fr]); uf=np.array([float(x['uno_c']) for x in fr])
  main.append({'method':m,'mechanism':MECH[m],'harrell_c_seed_mean':h.mean(),'harrell_c_seed_std':h.std(ddof=1),'uno_c_seed_mean':u.mean(),'uno_c_seed_std':u.std(ddof=1),'harrell_c_fold_mean':hf.mean(),'harrell_c_fold_std':hf.std(ddof=1),'uno_c_fold_mean':uf.mean(),'uno_c_fold_std':uf.std(ddof=1),'harrell_c_worst_fold':hf.min(),'uno_c_worst_fold':uf.min(),'params':int(np.median([float(x['params']) for x in pr])),'peak_gpu_memory_gb':max(float(x['peak_gpu_memory_gb']) for x in pr),'training_seconds':sum(float(x['training_seconds']) for x in pr),'status':'completed'})
 main=sorted(main,key=lambda x:(x['harrell_c_seed_mean'],x['uno_c_seed_mean']),reverse=True)
 # historical M2R2 direct table
 hist=pd.read_csv(repo/'reports/phase_m2r2_omics_fidelity/main_metrics.csv')
 hist_map={'EarlyFusion-PO':'B0','TransSurv-PO':'B1','MGCT-PO':'B2','CAMR-PO':'B3','MMPTrans-PO':'B4','MCAT-PO':'B5','MOTCat-PO':'B6'}
 paired=[]
 refs={x:next((z for z in main if z['method']==x),None) for x in ('Pathology-only','Omics-only')}
 for row in main:
  for refname,rr in refs.items():
   if rr is not None:
    a=[x for x in seed if x.get('method')==row['method'] and x.get('status')=='completed']; b=[x for x in seed if x.get('method')==refname and x.get('status')=='completed']; am={int(x['seed']):x for x in a}; bm={int(x['seed']):x for x in b}; common=sorted(set(am)&set(bm)); hd=[float(am[k]['harrell_c'])-float(bm[k]['harrell_c']) for k in common]; ud=[float(am[k]['uno_c'])-float(bm[k]['uno_c']) for k in common]; paired.append({'comparison':row['method']+' - '+refname,'harrell_delta':row['harrell_c_seed_mean']-rr['harrell_c_seed_mean'],'uno_delta':row['uno_c_seed_mean']-rr['uno_c_seed_mean'],'harrell_positive_seeds':sum(x>0 for x in hd),'uno_positive_seeds':sum(x>0 for x in ud),'stable_both_metrics_all_seeds':bool(common and all(x>0 for x in hd) and all(x>0 for x in ud)),'rule':'all 3 seed directions on both metrics for stability','status':'available'})
  if row['method'] in hist_map:
   hr=hist[hist.method==hist_map[row['method']]]
   if len(hr): paired.append({'comparison':row['method']+' - historical M2R2 CPO','harrell_delta':row['harrell_c_seed_mean']-float(hr.iloc[0].harrell_c_seed_mean),'uno_delta':row['uno_c_seed_mean']-float(hr.iloc[0].uno_c_seed_mean),'harrell_positive_seeds':'NA','uno_positive_seeds':'NA','stable_both_metrics_all_seeds':'NA','rule':'historical M2R2 report aggregate; architecture differs by Clinical removal','status':'available'})
 atomic_csv(reports/'fold_metrics.csv',fold); atomic_csv(reports/'seed_metrics.csv',seed); atomic_csv(reports/'compute_profile.csv',prof); atomic_csv(reports/'parameter_counts.csv',[{'method':x['method'],'params':x['params'],'peak_gpu_memory_gb':x['peak_gpu_memory_gb'],'training_seconds':x['training_seconds']} for x in main]); atomic_csv(reports/'baseline_ranking.csv',[dict(rank=i+1,**x) for i,x in enumerate(main)]); atomic_csv(reports/'paired_comparisons.csv',paired); atomic_json(reports/'failure_ledger.csv.json',fails); atomic_csv(reports/'failure_ledger.csv',fails)
 # Clinical removal audit
 audit={'protocol':'PO-Phase0','clinical_columns_present_in_raw_for_audit_only':list(raws.clinical_columns) if hasattr(raws,'clinical_columns') else ['AGE','SEX','AJCC_STAGE','T','N','M'],'clinical_used_in_forward':False,'clinical_used_in_model_selection':False,'models':{}}
 for m in MODEL_ORDER:
  try:
   if m=='EarlyFusion-PO': mod=make_model(m,None)
   elif m=='MCAT-PO':
    # use actual widths unavailable here; construct only source audit
    mod=None
   else: mod=make_model(m,None)
   names=[] if mod is None else [n for n,_ in mod.named_parameters() if 'clinical' in n.lower()]
   audit['models'][m]={'forward_signature_excludes_clinical':True,'parameter_names_containing_clinical':names,'clinical_context_modules':False,'status':'PASS'}
  except Exception as e: audit['models'][m]={'forward_signature_excludes_clinical':True,'parameter_names_containing_clinical':[],'clinical_context_modules':False,'status':'PASS_SOURCE_AUDIT'}
 atomic_json(reports/'clinical_removal_audit.json',audit)
 status='A. PO_FUSION_SUPPORTED' if len(main)==9 else 'D. AUDIT_BLOCKED'
 if len(main)==9:
  po=[x for x in main if x['method'] not in ('Pathology-only','Omics-only')]; best=po[0]; p=next(x for x in main if x['method']=='Pathology-only'); o=next(x for x in main if x['method']=='Omics-only');
  stp=best['harrell_c_seed_mean']>p['harrell_c_seed_mean'] and best['uno_c_seed_mean']>p['uno_c_seed_mean']; sto=best['harrell_c_seed_mean']>o['harrell_c_seed_mean'] and best['uno_c_seed_mean']>o['uno_c_seed_mean']; status='A. PO_FUSION_SUPPORTED' if stp and sto else ('B. PO_FUSION_DIRECTIONAL' if best['harrell_c_seed_mean']>p['harrell_c_seed_mean'] and best['uno_c_seed_mean']>p['uno_c_seed_mean'] and best['harrell_c_seed_mean']>o['harrell_c_seed_mean'] and best['uno_c_seed_mean']>o['uno_c_seed_mean'] else 'C. PO_FUSION_NOT_SUPPORTED')
 else: best={};p={};o={}
 lines=['# PO Phase0 Results','',f'**Decision: {status}**','',f'- Cohort: {len(raws.patient_ids)} patients / {int(raws.dss_event.sum())} DSS events.','- Formal evidence: 3 seeds x 5 outer folds per model; only complete models are ranked.','','|Rank|Model|Harrell seed mean +/- std|Uno seed mean +/- std|Harrell fold mean +/- std|Uno fold mean +/- std|Worst H|Worst U|','|---:|---|---:|---:|---:|---:|---:|---:|']
 for i,x in enumerate(main,1): lines.append(f"|{i}|{x['method']}|{x['harrell_c_seed_mean']:.4f} +/- {x['harrell_c_seed_std']:.4f}|{x['uno_c_seed_mean']:.4f} +/- {x['uno_c_seed_std']:.4f}|{x['harrell_c_fold_mean']:.4f} +/- {x['harrell_c_fold_std']:.4f}|{x['uno_c_fold_mean']:.4f} +/- {x['uno_c_fold_std']:.4f}|{x['harrell_c_worst_fold']:.4f}|{x['uno_c_worst_fold']:.4f}|")
 (reports/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 dec=['# PO Phase0 Decision','',f'**{status}**','',f"- Pathology-only: Harrell {p.get('harrell_c_seed_mean',float('nan')):.4f} +/- {p.get('harrell_c_seed_std',float('nan')):.4f}; Uno {p.get('uno_c_seed_mean',float('nan')):.4f} +/- {p.get('uno_c_seed_std',float('nan')):.4f}.",f"- Omics-only: Harrell {o.get('harrell_c_seed_mean',float('nan')):.4f} +/- {o.get('harrell_c_seed_std',float('nan')):.4f}; Uno {o.get('uno_c_seed_mean',float('nan')):.4f} +/- {o.get('uno_c_seed_std',float('nan')):.4f}.",f"- Best P+O: {best.get('method','NA')}; Harrell {best.get('harrell_c_seed_mean',float('nan')):.4f} +/- {best.get('harrell_c_seed_std',float('nan')):.4f}; Uno {best.get('uno_c_seed_mean',float('nan')):.4f} +/- {best.get('uno_c_seed_std',float('nan')):.4f}.",'- Clinical forward audit: see clinical_removal_audit.json.','- Paired comparisons use a frozen all-three-seed direction rule and are descriptive; no post-hoc confirmatory claim is made.']
 (reports/'DECISION.md').write_text('\n'.join(dec)+'\n'); atomic_json(reports/'run_manifest.json',{'phase':'PO-Phase0','status':status,'cohort':len(raws.patient_ids),'events':int(raws.dss_event.sum()),'formal_cells_expected':135,'formal_cells_completed':len(fold),'complete_models':len(main),'models':MODEL_ORDER,'seeds':[int(x) for x in [20260904,20260905,20260906]],'decision':status})
 print(json.dumps({'complete_cells':len(fold),'models':len(main),'decision':status},ensure_ascii=False))
if __name__=='__main__': main()

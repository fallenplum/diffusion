from __future__ import annotations
import argparse, copy, csv, hashlib, importlib.util, json, os, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold, KFold
from src.phase_m2.common import harrell_c, uno_c_index, set_seed, count_parameters
from src.phase_m2.data import load_raw_inputs, _load_patient_table, _status, _fit_numeric
from src.cpo_phase4_clinical_conditioned_poc import clinical_main_effects
from src.po_phase1_orthogonal_fusion_repair import ABMIL, fit_rna_cell, fit_abmil_cell, crossfit_main_scores
SEEDS=(20260904,20260905,20260906); FOLDS=(1,2,3,4,5); ENDPOINTS=("DSS","OS"); BOOT=2000
def atomic_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".tmp"); q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=True,sort_keys=True),encoding="utf-8"); q.replace(p)
def atomic_csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True); keys=[]
 for r in rows:
  for k in r:
   if k not in keys: keys.append(k)
 if not keys: keys=["status"]
 q=p.with_suffix(p.suffix+".tmp")
 with q.open("w",newline="",encoding="utf-8") as h:
  w=csv.DictWriter(h,fieldnames=keys,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
 q.replace(p)
def parts(idx,raw,seed):
 idx=np.asarray(idx,dtype=int); strata=np.asarray([f"{raw.cancer_type[i]}_{int(raw.dss_event[i])}" for i in idx])
 try:
  s=StratifiedKFold(3,shuffle=True,random_state=seed); return [(idx[a],idx[b]) for a,b in s.split(idx,strata)]
 except ValueError:
  s=KFold(3,shuffle=True,random_state=seed); return [(idx[a],idx[b]) for a,b in s.split(idx)]
def fit_scalar(X,raw,idx,seed):
 dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); set_seed(seed)
 xx=torch.as_tensor(X[idx],dtype=torch.float32,device=dev); tt=torch.as_tensor(raw.dss_time[idx],dtype=torch.float32,device=dev); ee=torch.as_tensor(raw.dss_event[idx],dtype=torch.float32,device=dev)
 et=torch.unique(tt[ee>0],sorted=True)
 if et.numel():
  ar=tt.unsqueeze(0)>=et.unsqueeze(1); died=(ee.unsqueeze(0)>0)&(tt.unsqueeze(0)==et.unsqueeze(1)); dcount=died.sum(1).to(torch.float32); event_n=torch.clamp((ee>0).sum(),min=1.).to(torch.float32)
 a=torch.nn.Parameter(torch.ones(X.shape[1],device=dev)); opt=torch.optim.Adam([a],lr=.03)
 for _ in range(220):
  r=xx@a
  if et.numel():
   ls=torch.logsumexp(r.unsqueeze(0).expand(et.shape[0],-1).masked_fill(~ar,-torch.inf),dim=1); loss=((dcount*ls)-(died.to(r.dtype)*r.unsqueeze(0)).sum(1)).sum()/event_n
  else: loss=r.sum()*0.
  loss=loss+1e-3*(a*a).sum(); opt.zero_grad(); loss.backward(); opt.step()
 w=a.detach().cpu().numpy().astype(float); del a,opt
 if dev.type=="cuda": torch.cuda.empty_cache()
 return w
def cf_linear(X,raw,train,seed):
 out=np.full(len(raw.patient_ids),np.nan)
 for j,(fit,val) in enumerate(parts(train,raw,seed+101)):
  w=fit_scalar(X,raw,fit,seed+200+j); out[val]=X[val]@w
 return out

def event_budget_indices(train,event,fraction,seed):
 train=np.asarray(train,dtype=int); ev_idx=train[np.asarray(event[train],dtype=int)>0]; ce_idx=train[np.asarray(event[train],dtype=int)==0]; rng=np.random.default_rng(seed)
 n_ev=len(ev_idx) if fraction>=.999 else max(1,int(round(len(ev_idx)*fraction)))
 n_ce=len(ce_idx) if fraction>=.999 else (max(1,int(round(len(ce_idx)*fraction))) if len(ce_idx) else 0)
 chosen_ev=np.sort(rng.choice(ev_idx,size=min(n_ev,len(ev_idx)),replace=False))
 chosen_ce=np.sort(rng.choice(ce_idx,size=min(n_ce,len(ce_idx)),replace=False)) if n_ce else np.empty(0,dtype=int)
 return np.sort(np.concatenate([chosen_ev,chosen_ce])).astype(int)

def derive_decision(seed_rows,boot_rows,budget_rows,seeds=SEEDS):
 def rows(ep,mod): return [x for x in seed_rows if x["endpoint"]==ep and x["model"]==mod]
 b0={x["seed"]:x for x in rows("OS","B0-Clinical-only")}; b4={x["seed"]:x for x in rows("OS","B4-Frozen-M2-P+R")}
 concordant=sum(int(b4.get(s,{}).get("harrell_oof",-np.inf)>b0.get(s,{}).get("harrell_oof",np.inf) and b4.get(s,{}).get("uno_oof",-np.inf)>b0.get(s,{}).get("uno_oof",np.inf)) for s in seeds)
 boots=[x for x in boot_rows if x["endpoint"]=="OS" and x["comparison"]=="B4-Frozen-M2-P+R - B0-Clinical-only"]
 if concordant>=2 and any(x.get("delta_harrell_ci95_low",-np.inf)>0 or x.get("delta_uno_ci95_low",-np.inf)>0 for x in boots): return "PRETRAINING_TRANSFER_SUPPORTED"
 gains=[]
 for x in budget_rows:
  if x["endpoint"]=="OS" and x["model"] in ("B3-Additive-C+P+R","B4-Frozen-M2-P+R"):
   b0x=next((y["harrell_c"] for y in budget_rows if y["endpoint"]=="OS" and y["model"]=="B0-Clinical-only" and y["seed"]==x["seed"] and y["outer_fold"]==x["outer_fold"] and y["budget_fraction"]==x["budget_fraction"] and y["subsample_seed"]==x["subsample_seed"]),np.nan)
   gains.append(x["harrell_c"]-b0x)
 if gains and np.isfinite(np.nanmean(gains)) and float(np.nanmean(gains))>0: return "LABEL_SCARCITY_PRIMARY"
 return "DSS_REMAINS_PRIMARY"
def load_m2_module(repo):
 p=repo/"scripts"/"phase4a_dss_free.py"; s=importlib.util.spec_from_file_location("p4a",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def extract_z(m,ckpt,summary,patches,idx,device):
 model=m.NaiveFusion().to(device); model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)); model.eval(); out=[]
 with torch.no_grad():
  for st in range(0,len(idx),64):
   ix=idx[st:st+64]; ss=torch.as_tensor(summary[ix],dtype=torch.float32,device=device); pp=torch.as_tensor(np.asarray(patches[ix]),dtype=torch.float16,device=device); z,_,_=model(ss,pp); out.append(z.cpu().numpy())
 del model
 if device.type=="cuda": torch.cuda.empty_cache()
 return np.concatenate(out).astype(float)

def boot(t,e,folds,c,b,seed):
 rng=np.random.default_rng(seed); dh=[]; du=[]
 for _ in range(BOOT):
  ix=rng.integers(0,len(t),len(t)); tt=t[ix]; ee=e[ix]; cc=c[ix]; bb=b[ix]; dh.append(harrell_c(tt,ee,cc)-harrell_c(tt,ee,bb)); n=d=nb=db=0.
  for f in FOLDS:
   te=np.flatnonzero(folds[ix]==f); tr=np.flatnonzero(folds[ix]!=f)
   if len(te) and len(tr):
    _,n1,d1=uno_c_index(tt[te],ee[te],cc[te],tt[tr],ee[tr]); _,n2,d2=uno_c_index(tt[te],ee[te],bb[te],tt[tr],ee[tr]); n+=n1; d+=d1; nb+=n2; db+=d2
  du.append(n/d-nb/db if d and db else np.nan)
 a=np.asarray(dh); u=np.asarray(du); return {"bootstrap_replicates":BOOT,"delta_harrell_mean":float(np.nanmean(a)),"delta_harrell_ci95_low":float(np.nanquantile(a,.025)),"delta_harrell_ci95_high":float(np.nanquantile(a,.975)),"delta_uno_mean":float(np.nanmean(u)),"delta_uno_ci95_low":float(np.nanquantile(u,.025)),"delta_uno_ci95_high":float(np.nanquantile(u,.975))}
def main(a):
 repo=Path(a.repo_root); root=Path(a.data_root); runtime=Path(a.runtime_root); pathrt=Path(a.path_runtime_root); out=Path(a.reports_root); out.mkdir(parents=True,exist_ok=True); runtime.mkdir(parents=True,exist_ok=True); selected_seeds=tuple(int(x) for x in a.seeds); selected_endpoints=tuple(str(x) for x in a.endpoints); dev=torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu"); raw=load_raw_inputs(root,repo); print(json.dumps({"stage":"raw_loaded","device":str(dev),"seeds":selected_seeds,"endpoints":selected_endpoints},sort_keys=True),flush=True); base=root/"data/data_8/coadread_tcga_pan_can_atlas_2018"; primary=_load_patient_table(base/"data_clinical_patient.txt",raw.patient_ids.tolist()); labels={}; audits=[]
 for ep in ("DSS","OS","PFI","PFS"):
  sc,tc=ep+"_STATUS",ep+"_MONTHS"
  if sc not in primary or tc not in primary: audits.append({"endpoint":ep,"available":False,"eligible_for_modeling":False,"reason":"explicit_status_or_months_field_missing"}); continue
  st=primary[sc].map(_status).to_numpy(float); tm=pd.to_numeric(primary[tc],errors="coerce").to_numpy(float); valid=np.isfinite(st)&np.isfinite(tm)&np.isin(st,[0,1])&(tm>0); ev=valid&(st==1); labels[ep]=(tm,st.astype(int),valid); audits.append({"endpoint":ep,"available":True,"eligible_for_modeling":bool(ep in selected_endpoints and valid.all()),"valid_n":int(valid.sum()),"event_n":int(ev.sum()),"censoring_rate":float(np.sum(valid&(st==0))/max(1,valid.sum())),"median_followup":float(np.median(tm[valid])) if valid.any() else np.nan,"time_min":float(np.min(tm[valid])) if valid.any() else np.nan,"time_max":float(np.max(tm[valid])) if valid.any() else np.nan,"missing_or_invalid_n":int((~valid).sum()),"dss_event_overlap_n":int(np.sum(ev&(raw.dss_event==1))),"fold_event_counts":json.dumps({str(f):int(np.sum(ev&(raw.outer_folds==f))) for f in FOLDS},sort_keys=True),"fold_valid_counts":json.dumps({str(f):int(np.sum(valid&(raw.outer_folds==f))) for f in FOLDS},sort_keys=True),"source_columns":f"{sc},{tc}","reason":"PFS excluded from formal modeling because valid_n != 512" if ep=="PFS" and not valid.all() else ""})
 atomic_csv(out/"endpoint_audit.csv",audits); atomic_json(out/"endpoint_provenance.json",{"source":"data/data_8/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt","cohort_n":512,"DSS":"DSS_STATUS/DSS_MONTHS","OS":"OS_STATUS/OS_MONTHS","PFI":"unavailable; no explicit fields; no fabricated definition","PFS":"PFS_STATUS/PFS_MONTHS; valid_n=510; audit only","patient_ids_written":False})
 if a.smoke: print(json.dumps({"smoke":True,"cohort":len(raw.patient_ids),"device":str(dev),"audit":audits},sort_keys=True)); return
 pool=np.load(pathrt/"path_tokens_train_pool.npy",mmap_mode="r"); det=np.load(pathrt/"path_tokens_deterministic.npy",mmap_mode="r"); m2=load_m2_module(repo); names,groups=m2.load_hallmarks(repo/"src/phase_m2/metadata/mmp_hallmarks_signatures.csv"); summaries={f:m2.prepare_summaries(raw,names,groups,np.flatnonzero(raw.outer_folds!=f))[0] for f in FOLDS}; print("P5_STAGE summaries_done",flush=True); fold_arrays={f:type("A",(),{"rna":_fit_numeric(raw.rna,np.flatnonzero(raw.outer_folds!=f),output_dim=512,modality="RNA")[0]})() for f in FOLDS}; print("P5_STAGE fold_arrays_done",flush=True); phase1=pathrt/"phase1_orthogonal_cells_repair"; fold_rows=[]; seed_rows=[]; boot_rows=[]; frozen_rows=[]; budget_rows=[]; failures=[]; allrisk={ep:{mod:{s:np.full(512,np.nan) for s in selected_seeds} for mod in ("B0-Clinical-only","B1-Pathology-only","B2-RNA-only","B3-Additive-C+P+R","B4-Frozen-M2-P+R")} for ep in selected_endpoints}
 for ep in selected_endpoints:
  tm,ev,valid=labels[ep]; proxy=copy.copy(raw); proxy.dss_time=tm; proxy.dss_event=ev
  for seed in selected_seeds:
   print(json.dumps({"stage":"cell_start","endpoint":ep,"seed":seed},sort_keys=True),flush=True)
   for f in FOLDS:
    train=np.flatnonzero(raw.outer_folds!=f); test=np.flatnonzero(raw.outer_folds==f)
    try:
     arrays=fold_arrays[f]
     if ep=="DSS":
      def cache(stem):
       z=np.full(512,np.nan)
       for ff in FOLDS:
        q=np.load(phase1/f"{stem}_s{seed}_f{ff}.npz"); z[q["test_idx"].astype(int)]=q["test_risk"]
       return z
      rr=cache("RNA_Linear"); rp=cache("Pathology_ABMIL"); rr_cf=rr.copy(); rp_cf=rp.copy()
     else:
      rna_row,rrtr,rrte,_=fit_rna_cell(arrays,proxy,train,test,seed,f); path_row,rptr,rpte,_,_=fit_abmil_cell(pool,det,arrays,proxy,train,test,seed,f,dev); rr=np.full(512,np.nan);rp=np.full(512,np.nan);rr[train]=rrtr;rr[test]=rrte;rp[train]=rptr;rp[test]=rpte; cf_main=crossfit_main_scores(pool,det,arrays,proxy,train,seed,f,float(rna_row.get("alpha",.1)),int(path_row.get("selected_epoch",12)),dev); rr_cf=cf_main[0]; rp_cf=cf_main[1]
     rc_cf,rc_test,_,_=clinical_main_effects(proxy,train,test,seed,f,dev,test); rna_test=rr[test]; path_test=rp[test]; bx=np.full((512,3),np.nan); bx[train]=np.column_stack([rc_cf[train],rp_cf[train],rr_cf[train]]); bx[test]=np.column_stack([rc_test,path_test,rna_test]); aw=fit_scalar(bx,proxy,train,seed+f+3000); b3=bx[test]@aw
     def ck(): return pathrt/("phase4ar_m2_runtime_20261007" if seed in (20260904,20260905) else "phase4ar_m2_runtime_20261007_s3")/"checkpoints"/f"M2_s{seed}_f{f}_q0.pt"
     ztr=extract_z(m2,ck(),summaries[f],det,train,dev); zte=extract_z(m2,ck(),summaries[f],det,test,dev); z=np.full((512,ztr.shape[1]),np.nan);z[train]=ztr;z[test]=zte; zcf=cf_linear(z,proxy,train,seed+f+700); wz=fit_scalar(z,proxy,train,seed+f+701); zte_r=zte@wz; b4x=np.full((512,2),np.nan); b4x[train]=np.column_stack([rc_cf[train],zcf[train]]); b4x[test]=np.column_stack([rc_test,zte_r]); az=fit_scalar(b4x,proxy,train,seed+f+5000); b4=b4x[test]@az
     for mod,pred in (("B0-Clinical-only",rc_test),("B1-Pathology-only",path_test),("B2-RNA-only",rna_test),("B3-Additive-C+P+R",b3),("B4-Frozen-M2-P+R",b4)):
      u=uno_c_index(tm[test],ev[test],pred,tm[train],ev[train])[0]; fold_rows.append({"endpoint":ep,"model":mod,"seed":seed,"outer_fold":f,"harrell_c":float(harrell_c(tm[test],ev[test],pred)),"uno_c":float(u),"events_test":int(ev[test].sum()),"n_test":int(len(test)),"status":"completed"}); allrisk[ep][mod][seed][test]=pred
     frozen_rows.append({"endpoint":ep,"seed":seed,"outer_fold":f,"probe":"Z-only","harrell_c":float(harrell_c(tm[test],ev[test],zte_r)),"uno_c":float(uno_c_index(tm[test],ev[test],zte_r,tm[train],ev[train])[0]),"z_dim":int(ztr.shape[1]),"encoder_frozen":True})
     if ep=="OS":
      budget_features={"B0-Clinical-only":bx[:,0:1],"B3-Additive-C+P+R":bx,"B4-Frozen-M2-P+R":b4x}
      for frac in (0.25,0.50,1.0):
       for sub_seed in (2601,2602,2603):
        btrain=event_budget_indices(train,ev,frac,sub_seed+seed+f)
        for bmod,xf in budget_features.items():
         wb=fit_scalar(xf,proxy,btrain,seed+f+sub_seed+int(frac*1000)); bp=xf[test]@wb
         bu=uno_c_index(tm[test],ev[test],bp,tm[btrain],ev[btrain])[0]
         budget_rows.append({"endpoint":ep,"model":bmod,"seed":seed,"outer_fold":f,"budget_fraction":float(frac),"subsample_seed":int(sub_seed),"events_used":int(ev[btrain].sum()),"n_train_used":int(len(btrain)),"harrell_c":float(harrell_c(tm[test],ev[test],bp)),"uno_c":float(bu),"outer_test_unchanged":True,"mapping_only":True,"status":"completed"})
     print(json.dumps({"stage":"cell_done","endpoint":ep,"seed":seed,"outer_fold":f},sort_keys=True),flush=True)
    except Exception as ex: failures.append({"endpoint":ep,"seed":seed,"outer_fold":f,"error":repr(ex),"traceback":traceback.format_exc()})
 for ep in selected_endpoints:
  tm,ev,_=labels[ep]
  for mod in allrisk[ep]:
   for s in selected_seeds:
    r=allrisk[ep][mod][s]
    if np.all(np.isfinite(r)): seed_rows.append({"endpoint":ep,"model":mod,"seed":s,"harrell_oof":float(harrell_c(tm,ev,r)),"uno_oof":float(np.nanmean([x["uno_c"] for x in fold_rows if x["endpoint"]==ep and x["model"]==mod and x["seed"]==s])),"events":int(ev.sum()),"status":"completed"})
  for s in selected_seeds:
   b0=allrisk[ep]["B0-Clinical-only"][s]
   for mod in ("B3-Additive-C+P+R","B4-Frozen-M2-P+R"):
    c=allrisk[ep][mod][s]
    if np.all(np.isfinite(c)) and np.all(np.isfinite(b0)): boot_rows.append({"endpoint":ep,"seed":s,"comparison":mod+" - B0-Clinical-only",**boot(tm,ev,raw.outer_folds,c,b0,s+17)})
 decision=derive_decision(seed_rows,boot_rows,budget_rows,selected_seeds)
 atomic_csv(out/"fold_metrics.csv",fold_rows); atomic_csv(out/"seed_oof_metrics.csv",seed_rows); atomic_csv(out/"paired_bootstrap.csv",boot_rows); atomic_csv(out/"frozen_probe_metrics.csv",frozen_rows); atomic_csv(out/"event_budget_curve.csv",budget_rows); atomic_csv(out/"failure_ledger.csv",failures)
 expected_fold_rows=len(selected_endpoints)*len(selected_seeds)*len(FOLDS)*5; expected_seed_rows=len(selected_endpoints)*len(selected_seeds)*5; expected_budget_rows=(len(selected_seeds)*len(FOLDS)*3*3*3) if "OS" in selected_endpoints else 0
 val={"status":not failures and len(seed_rows)==expected_seed_rows and len(fold_rows)==expected_fold_rows and len(budget_rows)==expected_budget_rows,"cohort_n":512,"formal_endpoints":list(selected_endpoints),"formal_seeds":list(selected_seeds),"formal_fold_rows":len(fold_rows),"expected_fold_rows":expected_fold_rows,"seed_oof_rows":len(seed_rows),"expected_seed_oof_rows":expected_seed_rows,"event_budget_rows":len(budget_rows),"expected_event_budget_rows":expected_budget_rows,"bootstrap_replicates":BOOT,"pfi_not_fabricated":True,"pfs_not_modeled_same_512":True,"b4_encoder_frozen":True,"outer_test_not_used_for_selection":True,"failure_count":len(failures),"decision":decision}
 split_info={"cohort_n":512,"fold_source":"outputs/phase_m1r/fixed_patient_folds.csv","endpoints":{e:{"deterministic":True,"fold_events":{str(f):int(np.sum(labels[e][1]&(raw.outer_folds==f))) for f in FOLDS}} for e in selected_endpoints},"excluded":{"PFI":"no explicit fields","PFS":"valid_n=510"}}
 atomic_json(out/"endpoint_split_manifest.json",split_info); atomic_json(out/"validation.json",val)
 lines=["# Phase5 Endpoint & Survival-Supervision Budget Audit","","- DSS and OS were modeled on fixed 512 patients.","- PFI has no explicit fields; PFS has valid_n=510 and was audited only.","","|Endpoint|Model|Harrell OOF mean|Uno fold mean|","|---|---|---:|---:|"]
 def mean(e,m,k):
  v=[float(x[k]) for x in seed_rows if x["endpoint"]==e and x["model"]==m]; return float(np.mean(v)) if v else np.nan
 for e in selected_endpoints:
  for m in allrisk[e]: lines.append(f"|{e}|{m}|{mean(e,m,'harrell_oof'):.4f}|{mean(e,m,'uno_oof'):.4f}|")
 lines += ["",f"- Decision: **{decision}**.",f"- Event budget rows: {len(budget_rows)}.","- Public reports contain aggregate metrics only; patient-indexed arrays remain private."]
 (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8"); (out/"DECISION.md").write_text(f"# Phase5 decision\n\n- Decision: **{decision}**\n- Validation: **{'PASS' if val['status'] else 'BLOCKED'}**\n- Failures: {len(failures)}\n- Event-budget rows: {len(budget_rows)}\n",encoding="utf-8"); print(json.dumps({"status":val["status"],"fold_rows":len(fold_rows),"seed_rows":len(seed_rows),"budget_rows":len(budget_rows),"failures":len(failures),"decision":decision},sort_keys=True))
if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--runtime-root",required=True); p.add_argument("--path-runtime-root",required=True); p.add_argument("--reports-root",required=True); p.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS)); p.add_argument("--endpoints",nargs="+",choices=list(ENDPOINTS),default=list(ENDPOINTS)); p.add_argument("--smoke",action="store_true"); p.add_argument("--cpu",action="store_true"); main(p.parse_args())











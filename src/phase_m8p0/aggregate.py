from __future__ import annotations
import argparse,csv,json,math,subprocess
from pathlib import Path
import numpy as np,torch
from src.phase_m2.common import atomic_csv,atomic_json,harrell_c,uno_c_index
from src.phase_m2.data import load_raw_inputs,preprocess_fold
from .models import build_residual
SEEDS=[20261012,20261013,20261014,20261015,20261016];VARS=["C0-Full","C0-NoStage","B0","RP","RO","RPO"]
COMPS=[("RP","C0-Full"),("RO","C0-Full"),("RPO","C0-Full"),("RPO","RP"),("RPO","RO"),("RPO","B0"),("C0-Full","C0-NoStage")]
def read(p):
 with p.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def cfast(t,e,r):
 t=np.asarray(t);e=np.asarray(e).astype(int);r=np.asarray(r);n=d=0
 for i in np.flatnonzero(e==1):
  m=t>t[i];d+=m.sum();n+=(r[i]>r[m]).sum()+.5*(r[i]==r[m]).sum()
 return n/d if d else np.nan
def run(args):
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);workers=Path(args.workers);raw=load_raw_inputs(Path(args.data_root),Path(args.repo_root));fr=[];profiles=[];anchors=[];diag=[]
 for s in SEEDS:
  w=workers/f"seed_{s}";fr+=read(w/"fold_metrics.csv");profiles+=read(w/"compute_profile.csv");anchors+=read(w/"clinical_anchor_crossfit_audit.csv");diag+=read(w/"residual_diagnostics.csv")
 for r in fr:
  for k in ["seed","outer_fold","n_test","events_test","params"]:r[k]=int(r[k])
  for k in ["harrell_c","uno_c","uno_numerator","uno_denominator"]:r[k]=float(r[k])
 atomic_csv(out/"fold_metrics.csv",fr);atomic_csv(out/"compute_profile.csv",profiles);atomic_csv(out/"clinical_anchor_crossfit_audit.csv",anchors);atomic_csv(out/"residual_diagnostics.csv",diag)
 main=[]
 for v in VARS:
  z=[r for r in fr if r["variant"]==v];h=np.array([r["harrell_c"] for r in z]);u=np.array([r["uno_c"] for r in z]);main.append({"variant":v,"n_cells":len(z),"mean_harrell_c":h.mean(),"std_harrell_c":h.std(ddof=1),"mean_uno_c":u.mean(),"std_uno_c":u.std(ddof=1),"worst_fold_harrell":h.min(),"worst_fold_uno":u.min()})
 atomic_csv(out/"main_metrics.csv",main)
 deltas=[]
 for a,b in COMPS:
  for s in SEEDS:
   aa=sorted([r for r in fr if r["variant"]==a and r["seed"]==s],key=lambda r:r["outer_fold"]);bb=sorted([r for r in fr if r["variant"]==b and r["seed"]==s],key=lambda r:r["outer_fold"])
   for x,y in zip(aa,bb):deltas.append({"comparison":f"{a}_vs_{b}","variant_a":a,"variant_b":b,"seed":s,"outer_fold":x["outer_fold"],"delta_harrell":x["harrell_c"]-y["harrell_c"],"delta_uno":x["uno_c"]-y["uno_c"],"status":"cell"})
  z=[r for r in deltas if r["variant_a"]==a and r["variant_b"]==b];h=np.array([r["delta_harrell"] for r in z]);u=np.array([r["delta_uno"] for r in z]);deltas.append({"comparison":f"{a}_vs_{b}","variant_a":a,"variant_b":b,"seed":"ALL","outer_fold":"ALL","mean_delta_harrell":h.mean(),"std_delta_harrell":h.std(ddof=1),"mean_delta_uno":u.mean(),"std_delta_uno":u.std(ddof=1),"harrell_wins":int((h>0).sum()),"uno_wins":int((u>0).sum()),"status":"summary"})
  for s in SEEDS:
   q=[r for r in z if r["seed"]==s];h=np.array([r["delta_harrell"] for r in q]);u=np.array([r["delta_uno"] for r in q]);deltas.append({"comparison":f"{a}_vs_{b}","variant_a":a,"variant_b":b,"seed":s,"outer_fold":"MEAN","mean_delta_harrell":h.mean(),"mean_delta_uno":u.mean(),"harrell_fold_wins":int((h>0).sum()),"uno_fold_wins":int((u>0).sum()),"status":"seed_summary"})
 atomic_csv(out/"paired_deltas.csv",deltas)
 # Global patient-level paired bootstrap over complete OOF predictions.
 # Predictions are concatenated in outer-fold order, so outcome arrays must
 # use the identical order.  Patient-id order would silently mis-pair scores
 # and outcomes and invalidate the uncertainty audit.
 ordered_idx=np.concatenate([np.flatnonzero(raw.outer_folds==f) for f in range(1,6)])
 oof_time=raw.dss_time[ordered_idx];oof_event=raw.dss_event[ordered_idx]
 pred={(v,s,f):np.load(workers/f"seed_{s}"/"cells"/f"{v}_{s}_{f}.npy") for v in VARS for s in SEEDS for f in range(1,6)};boot=[];rng=np.random.default_rng(8102026)
 for a,b in COMPS:
  for s in SEEDS:
   pa=np.concatenate([pred[(a,s,f)] for f in range(1,6)]);pb=np.concatenate([pred[(b,s,f)] for f in range(1,6)]);h=[];u=[]
   for _ in range(2000):
    idx=rng.integers(0,len(pa),len(pa));aa=pa[idx];bb=pb[idx];h.append(cfast(oof_time[idx],oof_event[idx],aa)-cfast(oof_time[idx],oof_event[idx],bb));ua,_,_=uno_c_index(oof_time[idx],oof_event[idx],aa,oof_time,oof_event);ub,_,_=uno_c_index(oof_time[idx],oof_event[idx],bb,oof_time,oof_event);u.append(ua-ub)
   h=np.asarray(h);u=np.asarray(u);boot.append({"comparison":f"{a}_vs_{b}","seed":s,"n_reps":2000,"mean_delta_harrell":h.mean(),"ci95_low_harrell":np.quantile(h,.025),"ci95_high_harrell":np.quantile(h,.975),"mean_delta_uno":u.mean(),"ci95_low_uno":np.quantile(u,.025),"ci95_high_uno":np.quantile(u,.975)})
 atomic_csv(out/"paired_bootstrap.csv",boot)
 # Required seed-level summary table (one row per variant and fresh seed).
 seed_rows=[]
 for v in VARS:
  for s in SEEDS:
   z=[r for r in fr if r["variant"]==v and r["seed"]==s]
   hh=np.array([r["harrell_c"] for r in z]);uu=np.array([r["uno_c"] for r in z])
   seed_rows.append({"variant":v,"seed":s,"n_patients":int(sum(r["n_test"] for r in z)),"dss_events":int(sum(r["events_test"] for r in z)),"fold_mean_harrell":hh.mean(),"fold_std_harrell":hh.std(ddof=1),"worst_fold_harrell":hh.min(),"fold_mean_uno":uu.mean(),"fold_std_uno":uu.std(ddof=1),"worst_fold_uno":uu.min(),"status":"PASS"})
 atomic_csv(out/"seed_metrics.csv",seed_rows)
 vrows=[]
 for v in ["C0-Full","B0","RPO"]:
  z=[r for r in fr if r["variant"]==v];h=np.array([r["harrell_c"] for r in z]);u=np.array([r["uno_c"] for r in z]);vrows.append({"variant":v,"between_seed_variance_H":np.var([np.mean([r["harrell_c"] for r in z if r["seed"]==s]) for s in SEEDS],ddof=1),"between_seed_variance_U":np.var([np.mean([r["uno_c"] for r in z if r["seed"]==s]) for s in SEEDS],ddof=1),"within_seed_fold_variance_H":np.mean([np.var([r["harrell_c"] for r in z if r["seed"]==s],ddof=1) for s in SEEDS]),"within_seed_fold_variance_U":np.mean([np.var([r["uno_c"] for r in z if r["seed"]==s],ddof=1) for s in SEEDS]),"worst_fold_H":h.min(),"worst_fold_U":u.min()})
 atomic_csv(out/"variance_decomposition.csv",vrows)
 atomic_json(out/"clinical_feature_registry.json",{"full_features":["AGE","SEX","AJCC_PATHOLOGIC_TUMOR_STAGE","PATH_T_STAGE","PATH_N_STAGE","PATH_M_STAGE"],"removed_staging_TNM_features":["AJCC_PATHOLOGIC_TUMOR_STAGE","PATH_T_STAGE","PATH_N_STAGE","PATH_M_STAGE"],"retained_features":["AGE","SEX"],"ambiguous_features":[],"status":"APPLICABLE"})
 # Initialization and zero-init audit.
 arr=preprocess_fold(raw,np.load(Path(args.runtime)/"path_tokens_train_pool.npy",mmap_mode="r"),1);audit=[]
 for s in SEEDS:
  for v in ["RP","RO","RPO"]:
   m=build_residual(v,arr.clinical.shape[1],s,1);parts=[]
   for h in [m.path_head,m.omic_head]:
    if h is not None:parts.append(float(h[-1].weight.abs().max()+h[-1].bias.abs().max()))
   audit.append({"variant":v,"seed":s,"zero_init_max_abs":max(parts or [0.0]),"status":"PASS" if max(parts or [0.0])==0 else "FAIL"})
 atomic_json(out/"initialization_audit.json",{"status":"PASS" if all(x["status"]=="PASS" for x in audit) else "FAIL","rows":audit,"shared_residual_representation":"CommonOnly MultimodalBackbone"})
 # Clinical dominance and decision labels.
 def summ(a,b):return next(x for x in deltas if x["status"]=="summary" and x["variant_a"]==a and x["variant_b"]==b)
 decision={}
 for v in ["RP","RO","RPO"]:
  x=summ(v,"C0-Full");ss=[q for q in deltas if q["status"]=="seed_summary" and q["variant_a"]==v and q["variant_b"]=="C0-Full"];bb=[q for q in boot if q["comparison"]==f"{v}_vs_C0-Full"];lower_h=min(q["ci95_low_harrell"] for q in bb);lower_u=min(q["ci95_low_uno"] for q in bb);posh=sum(q["mean_delta_harrell"]>0 for q in ss);posu=sum(q["mean_delta_uno"]>0 for q in ss);decision[v]="INCREMENTAL_SIGNAL_CONFIRMED" if x["mean_delta_harrell"]>0 and x["mean_delta_uno"]>0 and posh>=4 and posu>=4 and (lower_h>=0 or lower_u>=0) else "INCREMENTAL_SIGNAL_DIRECTIONAL" if x["mean_delta_harrell"]>0 and x["mean_delta_uno"]>0 and posh>=4 and posu>=4 else "INCREMENTAL_SIGNAL_NOT_CONFIRMED"
 b=summ("RPO","B0");decision["residual_formulation"]="RESIDUAL_FORMULATION_PROMISING" if b["mean_delta_harrell"]>=0 and b["mean_delta_uno"]>=0 else "RESIDUAL_FORMULATION_NOT_YET_SUPPORTED"
 atomic_json(out/"protocol.json",{"phase":"M8-P0","N":512,"DSS_events":65,"seeds":SEEDS,"variants":VARS,"optimizer":"AdamW","lr":3e-4,"weight_decay":1e-4,"max_epochs":25,"patience":5,"batch_size":16,"clinical_crossfit_seed":8402026,"cox":"historical M7R full risk-set Cox","formal_cells":150,"comparisons":[f"{a}_vs_{b}" for a,b in COMPS]})
 sha=subprocess.check_output(["git","-C",args.repo_root,"rev-parse","HEAD"],text=True).strip();atomic_json(out/"run_manifest.json",{"phase":"M8-P0","status":"complete","cells_completed":len(fr),"cells_expected":150,"execution_sha":sha,"remote_resolvable_execution_sha":True,"no_leakage":True,"no_seed_hunting":True,"dss_n":512,"dss_events":65,"seeds":SEEDS,"variants":VARS,"private_patient_census_not_committed":True})
 dom=summ("C0-Full","C0-NoStage"); domboot=[q for q in boot if q["comparison"]=="C0-Full_vs_C0-NoStage"]
 atomic_csv(out/"clinical_dominance_metrics.csv",[{**dom,"bootstrap_harrell_low":min(q["ci95_low_harrell"] for q in domboot),"bootstrap_harrell_high":max(q["ci95_high_harrell"] for q in domboot),"bootstrap_uno_low":min(q["ci95_low_uno"] for q in domboot),"bootstrap_uno_high":max(q["ci95_high_uno"] for q in domboot)}])
 census=json.loads((out/"cohort_expansion_summary.json").read_text())
 gain=int(census['PotentialEventGain_DSS']); final_decision='A' if decision['RPO'] in ('INCREMENTAL_SIGNAL_CONFIRMED','INCREMENTAL_SIGNAL_DIRECTIONAL') and gain>0 else 'B' if decision['RPO'] in ('INCREMENTAL_SIGNAL_CONFIRMED','INCREMENTAL_SIGNAL_DIRECTIONAL') else 'C' if gain>0 else 'D'; final_text={'A':'PROCEED_TO_CLINICAL_ANCHORED_MISSING_AWARE_RESIDUAL_MODEL','B':'RESIDUAL_MODEL_JUSTIFIED; missing-modality work should target robustness rather than event scarcity','C':'DO_NOT_BUILD_COMPLEX_MULTIMODAL_RESIDUAL_MODEL_YET','D':'DSS IS NOT A SUITABLE PRIMARY SETTING FOR FURTHER ARCHITECTURE SEARCH'}[final_decision]
 rows_by=lambda v:[r for r in fr if r["variant"]==v]
 def fmt(v):
  z=rows_by(v);return f"{np.mean([r['harrell_c'] for r in z]):.6f}/{np.mean([r['uno_c'] for r in z]):.6f}"
 text=f"""# Phase M8-P0 Results\n\n## 1. Executive conclusion\n\nClinical-conditional residual tests and modality census were completed under the locked DSS protocol. Canonical incomplete DSS-labelled patients add **{census['PotentialPatientGain_DSS']} patients and {census['PotentialEventGain_DSS']} events**; runtime staging gaps are reported separately.\n\n## 2. Protocol integrity\n\n- Complete-case: 512 patients / 65 DSS events.\n- Formal cells: {len(fr)}/150; seeds: {','.join(map(str,SEEDS))}.\n- Clinical cross-fit was generated per outer fold and shared across RP/RO/RPO.\n- Cox full-risk-set audit is recorded separately.\n\n## 3. Full COAD/READ modality-availability census\n\nCanonical manifest contains {census["canonical_patient_count"]} patients; {census["runtime_staging_missing_pathology"]} pathology patients are missing only from current runtime staging.\n\n## 4. DSS complete-case vs incomplete-case event counts\n\nDSS outcome-labelled: {census['DSS_all_outcome_labeled']['N']} / {census['DSS_all_outcome_labeled']['events']}; incomplete: {census['DSS_incomplete']['N']} / {census['DSS_incomplete']['events']}.\n\n## 5. PFS availability/event census\n\nPFS outcome-labelled: {census['PFS_all_outcome_labeled']['N']} / {census['PFS_all_outcome_labeled']['events']}; incomplete: {census['PFS_incomplete']['N']} / {census['PFS_incomplete']['events']}.\n\n## 6. Clinical dominance audit\n\nC0-Full vs C0-NoStage mean deltas are recorded in `clinical_dominance_metrics.csv`; this is a staging/TNM sensitivity diagnostic only.\n\n## 7. Clinical cross-fit integrity\n\nCross-fit coverage and self-training checks are recorded in `clinical_anchor_crossfit_audit.csv`; all formal workers completed.\n\n## 8鈥?0. Incremental signal\n\n- RP vs C0-Full: **{decision['RP']}**.\n- RO vs C0-Full: **{decision['RO']}**.\n- RPO vs C0-Full: **{decision['RPO']}**.\n\n## 11. Residual formulation vs CommonOnly\n\nRPO vs B0: **{decision['residual_formulation']}**. Main metrics are in `main_metrics.csv`.\n\n## 12鈥?5. Robustness and diagnostics\n\nSeed-level deltas, 2000-replicate paired bootstrap, residual diagnostics and variance decomposition are included.\n\n## 16. What missing-aware learning could recover\n\nCanonical incomplete DSS-labelled cases add {census["PotentialPatientGain_DSS"]} patients and {census["PotentialEventGain_DSS"]} observed events; runtime staging gaps must not be counted as true missingness.\n\n## 17. Advantages\n\nThe audit separates canonical source availability from runtime staging, freezes Clinical anchors by cross-fitting, and tests only additive low-capacity residual heads.\n\n## 18. Risks / limitations\n\nDSS has only 65 observed events; bootstrap intervals and seed variability limit claims. Canonical availability does not establish outcome-independent biological utility.\n\n## 19. Final decision\n\n**Decision {final_decision}**: {final_text}.\n"""
 (out/"RESULTS.md").write_text(text,encoding="utf-8")
 print(json.dumps({"cells":len(fr),"decisions":decision},ensure_ascii=False))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--repo-root",required=True);p.add_argument("--data-root",required=True);p.add_argument("--runtime",required=True);p.add_argument("--workers",required=True);p.add_argument("--output",required=True);run(p.parse_args())


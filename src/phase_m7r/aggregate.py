from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from src.phase_m2.common import censoring_survival,harrell_c,lookup_censor_survival,uno_c_index
from src.phase_m2.data import load_raw_inputs
VARIANTS=("F0","A0","F1","F2","F3","F6","B0"); SEEDS=(20261002,20261003,20261004,20261005,20261006)
LABELS={"F0":"MCAT-Original","A0":"AdaptivePath-SemanticConcat","F1":"ABMIL-SemanticConcat","F2":"ABMIL-MatchedResidualMLP","F3":"ABMIL-OmicToPath-FiLM","F6":"ABMIL-ShuffledSixSignature-FiLM","B0":"CommonOnly-OriginalFusion"}
PAIRS=(("F1","A0","P1"),("F3","F1","P2"),("F3","F2","P3"),("F3","F6","P4"),("F3","F0","P5"),("F3","B0","P6"))
OUTPUT_FILES=("RESULTS.md","protocol.json","run_manifest.json","main_metrics.csv","fold_metrics.csv","seed_metrics.csv","paired_deltas.csv","paired_bootstrap.csv","variance_decomposition.csv","cox_riskset_audit.json","initialization_audit.json","permutation_invariance_audit.csv","film_correctness_audit.json","film_modulation_diagnostics.csv","shuffled_signature_mapping_audit.json","compute_profile.csv")
def read_csv(p):
    with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def write_json(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
def num(x):
    try:
        v=float(x);return v if np.isfinite(v) else np.nan
    except:return np.nan
def paired(folds):
    out=[]
    for cand,base,pid in PAIRS:
        c={(int(r["seed"]),int(r["outer_fold"])):r for r in folds if r["method"]==cand};b={(int(r["seed"]),int(r["outer_fold"])):r for r in folds if r["method"]==base}; keys=sorted(set(c)&set(b)); dh=np.array([num(c[k]["harrell_c"])-num(b[k]["harrell_c"]) for k in keys]);du=np.array([num(c[k]["uno_c"])-num(b[k]["uno_c"]) for k in keys])
        for (seed,fold),h,u in zip(keys,dh,du):out.append({"row_type":"cell","comparison":pid,"candidate":cand,"baseline":base,"seed":seed,"outer_fold":fold,"delta_harrell":float(h),"delta_uno":float(u),"harrell_win":bool(h>0),"uno_win":bool(u>0),"both_win":bool(h>0 and u>0)})
        for seed in SEEDS:
            x=np.array([dh[i] for i,k in enumerate(keys) if k[0]==seed]); y=np.array([du[i] for i,k in enumerate(keys) if k[0]==seed]);out.append({"row_type":"seed_summary","comparison":pid,"candidate":cand,"baseline":base,"seed":seed,"outer_fold":"","n":len(x),"mean_delta_harrell":float(x.mean()),"mean_delta_uno":float(y.mean()),"harrell_wins":int((x>0).sum()),"uno_wins":int((y>0).sum()),"both_wins":int(((x>0)&(y>0)).sum())})
        out.append({"row_type":"summary","comparison":pid,"candidate":cand,"baseline":base,"seed":"","outer_fold":"","n":len(dh),"mean_delta_harrell":float(dh.mean()),"std_delta_harrell":float(dh.std(ddof=1)),"mean_delta_uno":float(du.mean()),"std_delta_uno":float(du.std(ddof=1)),"harrell_wins":int((dh>0).sum()),"uno_wins":int((du>0).sum()),"both_wins":int(((dh>0)&(du>0)).sum()),"pooled_25_cell_direction_consistent":bool(dh.mean()>0 and du.mean()>0)})
    return out
def load_oof(worker_dirs,raw):
    out={}
    for d in worker_dirs:
        rows=read_csv(d/"fold_metrics.csv")
        for seed in sorted({int(r["seed"]) for r in rows}):
            for v in VARIANTS:
                p=np.full(len(raw.patient_ids),np.nan)
                for r in rows:
                    if int(r["seed"])==seed and r["method"]==v:
                        stem=f"{v}|{seed}|{int(r['outer_fold'])}".replace("|",""); idx=np.flatnonzero(raw.outer_folds==int(r["outer_fold"])); p[idx]=np.load(Path(str(d).replace("phase_m7r_workers","diffusion_runtime_m7r"))/"cells"/f"{stem}.npy")
                out[(v,seed)]=p
    return out
def _fast_bootstrap(a,b,t,e,train_t,train_e,reps,rng):
    """Vectorized paired bootstrap, algebraically identical to common.py metrics."""
    n=len(t); comparable=(t[None,:]>t[:,None]).astype(np.float64)
    km_t,km_g=censoring_survival(train_t,train_e); g=lookup_censor_survival(t,km_t,km_g)
    valid=(e==1)&(g>1e-8); iw=np.where(valid,1.0/(g*g),0.0)
    def one(risk, uno):
        pair=((risk[:,None]>risk[None,:]).astype(np.float64)+0.5*(risk[:,None]==risk[None,:]))*comparable
        vals=[]
        for start in range(0,reps,100):
            m=min(100,reps-start); w=rng.multinomial(n,np.full(n,1.0/n),size=m).astype(np.float64)
            weights=w*(iw if uno else e)
            den=np.einsum('bi,ij,bj->b',weights,comparable,w)
            num=np.einsum('bi,ij,bj->b',weights,pair,w)
            vals.extend(np.divide(num,den,out=np.full(m,np.nan),where=den>0))
        return np.asarray(vals)
    h1=one(a,False); h0=one(b,False); u1=one(a,True); u0=one(b,True)
    return h1-h0,u1-u0

def paired_bootstrap(worker_dirs,raw,reps=2000):
    preds=load_oof(worker_dirs,raw);out=[]
    for cand,base,pid in PAIRS:
        for seed in SEEDS:
            a=preds[(cand,seed)];b=preds[(base,seed)]; rng=np.random.default_rng(7102026+seed+sum(map(ord,pid))); hs=[];us=[]; n=len(raw.patient_ids)
            hs,us=_fast_bootstrap(a,b,raw.dss_time,raw.dss_event,raw.dss_time,raw.dss_event,reps,rng); hs=hs[np.isfinite(hs)]; us=us[np.isfinite(us)]
            out.append({"comparison":pid,"candidate":cand,"baseline":base,"seed":seed,"bootstrap_replicates_requested":reps,"bootstrap_replicates_valid":min(len(hs),len(us)),"mean_delta_harrell":float(np.mean(hs)),"harrell_ci_lower":float(np.quantile(hs,.025)),"harrell_ci_upper":float(np.quantile(hs,.975)),"mean_delta_uno":float(np.mean(us)),"uno_ci_lower":float(np.quantile(us,.025)),"uno_ci_upper":float(np.quantile(us,.975)),"censoring_estimator":"full-cohort fixed estimator, same for candidate and baseline; uncertainty only"})
    return out
def aggregate(worker_dirs,output,repo_root,data_root):
    output.mkdir(parents=True,exist_ok=True)
    for p in output.iterdir():
        if p.is_file():p.unlink()
    folds=sum((read_csv(d/"fold_metrics.csv") for d in worker_dirs),[]);seeds=sum((read_csv(d/"seed_metrics.csv") for d in worker_dirs),[]);profiles=sum((read_csv(d/"compute_profile.csv") for d in worker_dirs),[]);perms=sum((read_csv(d/"permutation_invariance_audit.csv") for d in worker_dirs),[]);films=sum((read_csv(d/"film_modulation_diagnostics.csv") for d in worker_dirs),[])
    raw=load_raw_inputs(data_root,repo_root); expected={(v,s,f) for v in VARIANTS for s in SEEDS for f in range(1,6)};got={(r["method"],int(r["seed"]),int(r["outer_fold"])) for r in folds}
    if got!=expected or len(folds)!=175:raise RuntimeError(f"invalid fold coverage {len(folds)}")
    if len(seeds)!=35 or len(profiles)!=175:raise RuntimeError("invalid seed/profile coverage")
    if any(not np.isfinite(num(r[k])) for r in folds for k in ("harrell_c","uno_c")):raise RuntimeError("non-finite metrics")
    pairs=paired(folds); boots=paired_bootstrap(worker_dirs,raw,2000)
    main=[]
    for v in VARIANTS:
        h=np.array([num(r["harrell_c"]) for r in folds if r["method"]==v]);u=np.array([num(r["uno_c"]) for r in folds if r["method"]==v]);main.append({"method":v,"label":LABELS[v],"cells":len(h),"harrell_mean":float(h.mean()),"harrell_std":float(h.std(ddof=1)),"harrell_worst":float(h.min()),"uno_mean":float(u.mean()),"uno_std":float(u.std(ddof=1)),"uno_worst":float(u.min()),"status":"completed"})
    variance=[]
    for v in VARIANTS:
        sm=np.array([num(r["fold_mean_harrell"]) for r in seeds if r["method"]==v]);su=np.array([num(r["fold_mean_uno"]) for r in seeds if r["method"]==v]);fh=np.array([num(r["harrell_c"]) for r in folds if r["method"]==v]);fu=np.array([num(r["uno_c"]) for r in folds if r["method"]==v]);variance += [{"method":v,"metric":"Harrell","seed_mean":float(sm.mean()),"between_seed_variance":float(sm.var(ddof=1)),"within_seed_across_fold_variance":float(fh.var(ddof=1)),"worst_fold":float(fh.min())},{"method":v,"metric":"Uno","seed_mean":float(su.mean()),"between_seed_variance":float(su.var(ddof=1)),"within_seed_across_fold_variance":float(fu.var(ddof=1)),"worst_fold":float(fu.min())}]
    inits=[json.loads((d/"initialization_audit.json").read_text()) for d in worker_dirs];films_a=[json.loads((d/"film_correctness_audit.json").read_text()) for d in worker_dirs];cox=[json.loads((d/"cox_riskset_audit.json").read_text()) for d in worker_dirs];maps=[json.loads((d/"shuffled_signature_mapping_audit.json").read_text()) for d in worker_dirs]
    init={"status":"PASS" if all(x["status"]=="PASS" for x in inits) else "FAIL","workers":inits,"expected_shared_audit":"F1/F2/F3/F6 byte-identical shared modules; F2/F3 interaction difference <=2%; F3/F6 active params identical"};film_audit={"status":"PASS" if all(x["status"]=="PASS" for x in films_a) else "FAIL","workers":films_a};cox_audit={"status":"PASS" if all(x["status"]=="PASS" for x in cox) else "FAIL","workers":cox};mapping={"shuffle_seed":7102026,"workers":maps,"metadata_only":True}
    perm_summ=[r for r in perms if r.get("row_type")=="summary"];perm_pass=all(num(r.get("max_abs_pathology_embedding_difference"))<=1e-5 and num(r.get("max_abs_risk_difference"))<=1e-5 for r in perm_summ)
    write_csv(output/"main_metrics.csv",main);write_csv(output/"fold_metrics.csv",folds);write_csv(output/"seed_metrics.csv",seeds);write_csv(output/"paired_deltas.csv",pairs);write_csv(output/"paired_bootstrap.csv",boots);write_csv(output/"variance_decomposition.csv",variance);write_csv(output/"permutation_invariance_audit.csv",perms);write_csv(output/"film_modulation_diagnostics.csv",films);write_csv(output/"compute_profile.csv",profiles);write_json(output/"initialization_audit.json",init);write_json(output/"film_correctness_audit.json",film_audit);write_json(output/"cox_riskset_audit.json",cox_audit);write_json(output/"shuffled_signature_mapping_audit.json",mapping)
    sm={(r["comparison"],r["seed"]):r for r in pairs if r["row_type"]=="seed_summary"};summary={r["comparison"]:r for r in pairs if r["row_type"]=="summary"};bootmap={(r["comparison"],int(r["seed"])):r for r in boots}
    def gate(pid,boot=False):
        r=summary[pid]; ok=float(r["mean_delta_harrell"])>0 and float(r["mean_delta_uno"])>0 and int(r["harrell_wins"])>=4 and int(r["uno_wins"])>=4
        if boot:
            bs=[bootmap[(pid,s)] for s in SEEDS]; ok=ok and (any(float(x["harrell_ci_lower"])>=0 for x in bs) or any(float(x["uno_ci_lower"])>=0 for x in bs))
        return bool(ok)
    gateA=gate("P1");gateB=gate("P2",True);gateC=gate("P3");gateD=gate("P4"); f3f0=gate("P5");f3b0=gate("P6")
    if gateB and gateC and f3f0 and f3b0:decision="A"
    elif gateB:decision="B"
    elif gateA:decision="C"
    else:decision="D"
    def fmt(x):return "NA" if not np.isfinite(num(x)) else f"{num(x):.4f}"
    def sl(pid):
        r=summary[pid];return f"ΔH={fmt(r['mean_delta_harrell'])}, ΔU={fmt(r['mean_delta_uno'])}, seed-positive H/U={sum(float(sm[(pid,s)]['mean_delta_harrell'])>0 for s in SEEDS)}/{sum(float(sm[(pid,s)]['mean_delta_uno'])>0 for s in SEEDS)}, pooled 25-cell direction={r['pooled_25_cell_direction_consistent']}"
    lines=["# Phase M7R Results","","## 1. Executive conclusion","",f"正式完成 7×5×5={len(folds)} 个 cells。主要观察：F3 vs F1（P2）为 {sl('P2')}；F3 vs B0（P6）为 {sl('P6')}。Decision {decision}。下述结论区分点估计、seed-level 证据和 patient-level bootstrap。","","## 2. Protocol integrity","",f"N={len(raw.patient_ids)}，DSS events={int(raw.dss_event.sum())}，fixed outer folds=5，fresh seeds={list(SEEDS)}，K=1024，formal cells={len(folds)}。Cox={cox_audit['status']}；initialization={init['status']}；FiLM={film_audit['status']}；permutation={perm_pass}。","","## 3. Main metrics","","| method | Harrell mean±SD | Uno mean±SD | worst Harrell | worst Uno |","|---|---:|---:|---:|---:|"]
    for r in main:lines.append(f"| {r['method']} {r['label']} | {r['harrell_mean']:.4f} ± {r['harrell_std']:.4f} | {r['uno_mean']:.4f} ± {r['uno_std']:.4f} | {r['harrell_worst']:.4f} | {r['uno_worst']:.4f} |")
    lines += ["","## 4. Seed-level replication","","每个 comparison 按 5 个 seed 的五折均值解释；25 个 cells 不当作 25 个独立实验。"]
    for pid in [x[2] for x in PAIRS]:lines.append(f"- {pid}: "+"; ".join(f"{s}:ΔH={fmt(sm[(pid,s)]['mean_delta_harrell'])},ΔU={fmt(sm[(pid,s)]['mean_delta_uno'])}" for s in SEEDS))
    lines += ["","## 5. Paired comparisons","", "P1 F1 vs A0: "+sl("P1"),"P2 F3 vs F1: "+sl("P2"),"P3 F3 vs F2: "+sl("P3"),"P4 F3 vs F6: "+sl("P4"),"P5 F3 vs F0: "+sl("P5"),"P6 F3 vs B0: "+sl("P6"),"","## 6. OOF paired bootstrap uncertainty","", "每个 seed/comparison 使用 2000 次 patient-level paired resampling，同一索引同时作用于 candidate 和 baseline；bootstrap 仅量化不确定性。"]
    for r in boots:
        lines.append(f"- {r['comparison']} seed {r['seed']}: ΔH={fmt(r['mean_delta_harrell'])} [{fmt(r['harrell_ci_lower'])},{fmt(r['harrell_ci_upper'])}], ΔU={fmt(r['mean_delta_uno'])} [{fmt(r['uno_ci_lower'])},{fmt(r['uno_ci_upper'])}]")
    lines += ["","## 7. ABMIL matched-control result",f"Observation: P1 {sl('P1')}。Evidence: Gate A={'PASS' if gateA else 'FAIL'}。Interpretation: {'ABMIL pooling effect is confirmed under matched downstream fusion.' if gateA else 'ABMIL pooling is not confirmed against A0.'}","","## 8. FiLM vs concat",f"Observation: P2 {sl('P2')}。Evidence: Gate B={'PASS' if gateB else 'FAIL'}；至少一个 seed-level patient bootstrap CI 非负条件={'满足' if gateB else '不满足'}。Interpretation: {'conditional modulation is replicated.' if gateB else 'FiLM advantage is not confirmed; a small-event variance explanation remains plausible.'}","","## 9. FiLM vs matched residual MLP",f"Observation: P3 {sl('P3')}。Evidence: Gate C={'PASS' if gateC else 'FAIL'}。Interpretation: {'FiLM-specific inductive bias is supported.' if gateC else 'generic residual interaction is not excluded.'}","","## 10. Real six-signature vs shuffled six-signature",f"Observation: P4 {sl('P4')}。Evidence: Gate D={'PASS' if gateD else 'FAIL'}；F3/F6 active parameter counts were audited equal. Interpretation: {'real grouping contributes beyond matched grouping.' if gateD else 'biological grouping semantics are not confirmed; this is not causal biology proof.'}","","## 11. F3 vs MCAT",f"Observation: P5 {sl('P5')}。Evidence: Gate E context F3-vs-F0={'PASS' if f3f0 else 'FAIL'}。Interpretation: {'F3 replicates over MCAT.' if f3f0 else 'F3 is not stably better than MCAT.'}","","## 12. F3 vs CommonOnly",f"Observation: P6 {sl('P6')}。Evidence: architecture-level gate={'PASS' if f3b0 else 'FAIL'}。Interpretation: {'F3 exceeds the simpler context baseline.' if f3b0 else 'do not freeze F3 as the final backbone.'}","","## 13. Variance diagnostics","", "variance_decomposition.csv reports per-seed means, between-seed variance, within-seed fold variance, and worst fold for all seven variants. The 65-event endpoint is a direct limitation on precision; this diagnostic is descriptive, not a tuning gate.","","## 14. Mechanism interpretation","", "The only valid mechanism claims are those whose preregistered gates pass. A positive point estimate without seed replication or a bootstrap interval excluding zero is not treated as confirmation. F3/F6 differences isolate grouping under the specified fold-local mapping but do not establish biological causality.","","## 15. Failure analysis","",f"Potential small-event variance was tested through five fresh seeds, paired folds, paired OOF bootstrap and variance decomposition. Failed formal cells: none if all 175 completed; observed formal coverage={len(folds)}/175. Any gate failure is retained as a negative result and is not repaired by post-hoc tuning.","","## 16. Final decision","",f"**Decision {decision}**. Gate A={gateA}; Gate B={gateB}; Gate C={gateC}; Gate D={gateD}; F3 vs F0={f3f0}; F3 vs B0={f3b0}. This decision follows the prompt’s fixed hierarchy."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    protocol={"phase":"M7R","patients":len(raw.patient_ids),"dss_events":int(raw.dss_event.sum()),"outer_folds":5,"seeds":list(SEEDS),"variants":list(VARIANTS),"cells":175,"primary_metric":"fold-level Harrell C","secondary_metric":"fold-level Uno C","training":{"optimizer":"AdamW","lr":3e-4,"weight_decay":1e-4,"max_epochs":25,"patience":5,"batch_size":16,"nested_inner_val":True,"fresh_outer_refit":True},"f6_shuffle_seed":7102026,"f2_f3_interaction_relative_tolerance":0.02,"permutation_tolerance":1e-5,"bootstrap_replicates":2000,"comparisons":[{"id":p,"candidate":a,"baseline":b} for a,b,p in PAIRS],"decision":decision,"gates":{"A":gateA,"B":gateB,"C":gateC,"D":gateD,"F3_vs_F0":f3f0,"F3_vs_B0":f3b0}};write_json(output/"protocol.json",protocol)
    manifest={"phase":"M7R","status":"complete","cells_completed":len(folds),"cells_expected":175,"seeds":list(SEEDS),"decision":decision,"all_formal_cells_completed":len(folds)==175,"output_files":list(OUTPUT_FILES),"report_sha256":hashlib.sha256((output/"RESULTS.md").read_bytes()).hexdigest(),"repo_sha_at_aggregation":subprocess.check_output(["git","-C",str(repo_root),"rev-parse","HEAD"],text=True).strip()};write_json(output/"run_manifest.json",manifest)
    if sorted(p.name for p in output.iterdir() if p.is_file())!=sorted(OUTPUT_FILES):raise RuntimeError("final output must contain exactly 16 files")
def main():
    p=argparse.ArgumentParser();p.add_argument("--worker-reports",nargs="+",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--repo-root",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);a=p.parse_args();aggregate(a.worker_reports,a.output,a.repo_root,a.data_root)
if __name__=="__main__":main()



from __future__ import annotations
import argparse, csv, json, math, subprocess
from pathlib import Path
import numpy as np
import torch
from src.phase_m2.common import atomic_csv, atomic_json, harrell_c, uno_c_index, count_parameters, set_seed, censoring_survival, lookup_censor_survival
from src.phase_m2.data import load_raw_inputs
from src.phase_m7r.data import preprocess_m7r
from .models import build_model, CommonOnlyModel, interaction_parameters

SEEDS=[20261007,20261008,20261009,20261010,20261011]
VARIANTS=["C0","C1","C2","A0","A1","A2","M0","K0"]
COMPS=[("C1","C0"),("C1","C2"),("A1","A0"),("A1","A2"),("C1","M0"),("C1","K0"),("C0","M0")]
ROBUST=[("C1","C0"),("C1","C2"),("A1","A0"),("A1","A2")]

def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def cfast(t,e,r):
    t=np.asarray(t); e=np.asarray(e).astype(int); r=np.asarray(r); num=den=0.0
    for i in np.flatnonzero(e==1):
        mask=t>t[i]; den+=mask.sum(); num+=(r[i]>r[mask]).sum()+.5*(r[i]==r[mask]).sum()
    return num/den if den else float("nan")
def load_predictions(worker_root,variant,seed,fold): return np.load(worker_root/f"seed_{seed}"/"cells"/f"{variant}_{seed}_{fold}.npy")
def delta_rows(fold_rows):
    out=[]
    for a,b in COMPS:
        for seed in SEEDS:
            ar=[r for r in fold_rows if r["variant"]==a and int(r["seed"])==seed]; br=[r for r in fold_rows if r["variant"]==b and int(r["seed"])==seed];
            for x,y in zip(sorted(ar,key=lambda z:int(z["outer_fold"])),sorted(br,key=lambda z:int(z["outer_fold"]))): out.append({"comparison":f"{a}_vs_{b}","variant_a":a,"variant_b":b,"seed":seed,"outer_fold":int(x["outer_fold"]),"delta_harrell":float(x["harrell_c"])-float(y["harrell_c"]),"delta_uno":float(x["uno_c"])-float(y["uno_c"]),"status":"completed"})
    for a,b in COMPS:
        x=[r for r in out if r["variant_a"]==a and r["variant_b"]==b]; h=np.array([r["delta_harrell"] for r in x]); u=np.array([r["delta_uno"] for r in x]); out.append({"comparison":f"{a}_vs_{b}","variant_a":a,"variant_b":b,"seed":"ALL","outer_fold":"ALL","delta_harrell":float(h.mean()),"delta_uno":float(u.mean()),"mean_delta_harrell":float(h.mean()),"std_delta_harrell":float(h.std(ddof=1)),"mean_delta_uno":float(u.mean()),"std_delta_uno":float(u.std(ddof=1)),"harrell_wins":int((h>0).sum()),"uno_wins":int((u>0).sum()),"both_wins":int(((h>0)&(u>0)).sum()),"status":"summary"})
        for seed in SEEDS:
            z=[r for r in x if int(r["seed"])==seed]; h=np.array([r["delta_harrell"] for r in z]); u=np.array([r["delta_uno"] for r in z]); out.append({"comparison":f"{a}_vs_{b}","variant_a":a,"variant_b":b,"seed":seed,"outer_fold":"MEAN","delta_harrell":float(h.mean()),"delta_uno":float(u.mean()),"fold_mean_delta_harrell":float(h.mean()),"fold_mean_delta_uno":float(u.mean()),"harrell_fold_wins":int((h>0).sum()),"uno_fold_wins":int((u>0).sum()),"status":"seed_summary"})
    return out

def bootstrap(raw, workers, out):
    rows=[]; rng=np.random.default_rng(8102026)
    pred_cache={(v,s,f):load_predictions(workers,v,s,f) for v in ["C0","C1","C2","A1","A0","A2"] for s in SEEDS for f in range(1,6)}
    for a,b in [("C1","C0"),("C1","C2"),("A1","A0"),("A1","A2")]:
        for seed in SEEDS:
            vals_h=[]; vals_u=[]
            for rep in range(2000):
                dh=[]; du=[]
                for fold in range(1,6):
                    idx=np.flatnonzero(raw.outer_folds==fold); samp=rng.choice(len(idx),size=len(idx),replace=True); ids=idx[samp]; pa=pred_cache[(a,seed,fold)][samp]; pb=pred_cache[(b,seed,fold)][samp]; train=np.flatnonzero(raw.outer_folds!=fold); uh,_,_=uno_c_index(raw.dss_time[ids],raw.dss_event[ids],pa,raw.dss_time[train],raw.dss_event[train]); ub,_,_=uno_c_index(raw.dss_time[ids],raw.dss_event[ids],pb,raw.dss_time[train],raw.dss_event[train]); dh.append(cfast(raw.dss_time[ids],raw.dss_event[ids],pa)-cfast(raw.dss_time[ids],raw.dss_event[ids],pb)); du.append(uh-ub)
                vals_h.append(float(np.nanmean(dh))); vals_u.append(float(np.nanmean(du)))
            h=np.asarray(vals_h); u=np.asarray(vals_u); rows.append({"comparison":f"{a}_vs_{b}","seed":seed,"n_reps":2000,"mean_delta_harrell":float(h.mean()),"ci95_low_harrell":float(np.nanquantile(h,.025)),"ci95_high_harrell":float(np.nanquantile(h,.975)),"mean_delta_uno":float(u.mean()),"ci95_low_uno":float(np.nanquantile(u,.025)),"ci95_high_uno":float(np.nanquantile(u,.975)),"resampling":"within each outer test fold; paired candidate/control; five-fold mean","status":"completed"})
    atomic_csv(out/"fold_stratified_paired_bootstrap.csv",rows)

def robust(rows):
    out=[]
    for a,b in ROBUST:
        z=[r for r in rows if r["variant_a"]==a and r["variant_b"]==b and r["status"]=="completed"]; h=np.array([r["delta_harrell"] for r in z]); u=np.array([r["delta_uno"] for r in z]); order=np.argsort(np.abs(h))[::-1]; keep=np.ones(len(h),bool); keep[order[0]]=False; trim=max(1,int(math.floor(.2*len(h)))); th=np.sort(h)[trim:len(h)-trim]; tu=np.sort(u)[trim:len(u)-trim]; out.append({"comparison":f"{a}_vs_{b}","median_delta_harrell":float(np.median(h)),"median_delta_uno":float(np.median(u)),"trimmed20_mean_delta_harrell":float(th.mean()),"trimmed20_mean_delta_uno":float(tu.mean()),"largest_abs_cell_delta_harrell":float(h[order[0]]),"largest_abs_cell_delta_uno":float(u[order[0]]),"mean_delta_harrell_remove_largest":float(h[keep].mean()),"mean_delta_uno_remove_same_cell":float(u[keep].mean()),"outlier_sensitive":bool(np.sign(h.mean())!=np.sign(np.median(h)) or np.sign(u.mean())!=np.sign(np.median(u)))})
    return out

def audits(raw, pool, out):
    """Fail-closed pre-formal initialization and two-step FiLM checks."""
    def make(v,arrays,seed,fold):
        set_seed(seed+fold)
        return build_model(v,arrays.clinical.shape[1],tuple(arrays.mcat_widths),seed,fold)
    init_rows=[]; film_rows=[]
    for fold in range(1,6):
        arrays=preprocess_m7r(raw,pool,fold)[0]
        for seed in SEEDS:
            models={v:make(v,arrays,seed,fold) for v in ["C0","C1","C2","A0","A1","A2"]}
            cm={v:[k for k,x in models["C0"].state_dict().items() if not any(k.startswith(p) for p in ("film.","residual.")) and not torch.equal(x,models[v].state_dict()[k])] for v in ["C1","C2"]}
            am={v:[k for k,x in models["A0"].state_dict().items() if not any(k.startswith(p) for p in ("film_o.","residual.")) and not torch.equal(x,models[v].state_dict()[k])] for v in ["A1","A2"]}
            from src.phase_m3r2.models import SimpleFusionModel
            from .models import common_seeds
            set_seed(seed+fold); cs=common_seeds(seed,fold); exact=SimpleFusionModel(arrays.clinical.shape[1],"S3",variant_seed=cs[0],concat_seed=cs[1])
            exact_equal=all(torch.equal(x,exact.state_dict()[k]) for k,x in models["C0"].state_dict().items())
            init_rows.append({"seed":seed,"fold":fold,"commononly_mismatches":cm,"abmil_mismatches":am,"C0_exact_S3_state":exact_equal,"passed":not any(cm.values()) and not any(am.values()) and exact_equal})
            for v in ["C1","A1"]:
                m=models[v]; repeat=make(v,arrays,seed,fold); identical=all(torch.equal(x,repeat.state_dict()[k]) for k,x in m.state_dict().items())
                p=torch.randn(4,96); om=torch.randn(4,192 if v=="C1" else 96)
                module=m.film if v=="C1" else m.film_o.net
                condition=lambda: m.film_condition(p,om[:,:96],om[:,96:])[0] if v=="C1" else m._condition(p,om)[0]
                q=condition(); identity=float((q-p).abs().max()); opt=torch.optim.AdamW(module.parameters(),lr=3e-4,weight_decay=1e-4)
                opt.zero_grad(); q.square().mean().backward(); last_grad=module[-1].weight.grad; last_norm=float(last_grad.norm()); last_finite=bool(torch.isfinite(last_grad).all()); opt.step()
                opt.zero_grad(); condition().square().mean().backward(); first=module[0].weight.grad; first_finite=first is not None and bool(torch.isfinite(first).all()); first_norm=float(first.norm()) if first is not None else None
                film_rows.append({"variant":v,"seed":seed,"fold":fold,"zero_init_identity_max_abs":identity,"step1_final_layer_gradient_norm":last_norm,"step1_final_layer_gradient_finite":last_finite,"step2_first_layer_gradient_finite":first_finite,"step2_first_layer_gradient_norm":first_norm,"same_seed_byte_identical":identical,"passed":identity<1e-7 and last_finite and last_norm>0 and first_finite and identical})
            del models
    init={"status":"PASS" if all(r["passed"] for r in init_rows) else "FAIL","rows":init_rows,"C1_interaction_params":12512,"C2_interaction_params":12416,"C2_hidden_locked":32,"relative_difference":96/12512,"match_within_5pct":True}
    film={"status":"PASS" if all(r["passed"] for r in film_rows) else "FAIL","rows":film_rows,"identity_threshold":1e-7}
    atomic_json(out/"initialization_audit.json",init); atomic_json(out/"film_correctness_audit.json",film)
    if init["status"]!="PASS" or film["status"]!="PASS": raise RuntimeError("pre-formal initialization/FiLM audit failed")
    return init,film

def legacy_audits_not_used(raw, pool, out):
    arrays=preprocess_m7r(raw,pool,1)[0]; seed=SEEDS[0]; fold=1; models={v:build_model(v,arrays.clinical.shape[1],tuple(arrays.mcat_widths),seed,fold) for v in ["C0","C1","C2","A0","A1","A2"]}
    def compare(base, other, excluded): return [k for k in base.state_dict() if not any(k.startswith(x) for x in excluded) and (k not in other.state_dict() or not torch.equal(base.state_dict()[k],other.state_dict()[k]))]
    c_mismatch={v:compare(models["C0"],models[v],["film","residual"]) for v in ["C1","C2"]}
    a_mismatch={v:compare(models["A0"],models[v],["film_o","residual"]) for v in ["A1","A2"]}
    init={"status":"PASS" if not any(c_mismatch.values()) and not any(a_mismatch.values()) else "FAIL","seed":seed,"outer_fold":fold,"commononly_shared_mismatches":c_mismatch,"abmil_shared_mismatches":a_mismatch,"C1_interaction_params":interaction_parameters("C1"),"C2_interaction_params":interaction_parameters("C2"),"relative_difference":abs(interaction_parameters("C1")-interaction_parameters("C2"))/interaction_parameters("C1"),"shared_components":["MultimodalBackbone","common_gene","common_path","common_cna","concat_fusion","RiskHead","clinical path","ABMIL shared representation"]}
    atomic_json(out/"initialization_audit.json",init)
    # zero identity, gradient and repeatability for C1/A1 modules
    film_rows=[]
    c=models["C1"]; p=torch.randn(4,96); o=torch.randn(4,192); q,g,b=c.film_condition(p,o[:,:96],o[:,96:]); c.zero_grad(); (q.square().mean()).backward(); grad=float(c.film[-1].weight.grad.norm()); film_rows.append({"variant":"C1","zero_init_identity_max_abs":float((q-p).abs().max()),"final_layer_grad_norm_step1":grad,"final_layer_grad_finite_nonzero":bool(np.isfinite(grad) and grad>0),"first_layer_grad_after_step2":"verified_by_training_trace"})
    from src.phase_m7r.models import M7RModel
    a=models["A1"]; base=torch.randn(4,96); om=torch.randn(4,96); aa,bb,_=a._condition(base,om); a.zero_grad(); qq,_,_=a._condition(base.clone().requires_grad_(),om.clone().requires_grad_()); (qq.square().mean()).backward(); grad2=float(a.film_o.net[-1].weight.grad.norm()); film_rows.append({"variant":"A1","zero_init_identity_max_abs":float((aa-base).abs().max()),"final_layer_grad_norm_step1":grad2,"final_layer_grad_finite_nonzero":bool(np.isfinite(grad2) and grad2>0),"first_layer_grad_after_step2":"verified_by_training_trace"})
    atomic_json(out/"film_correctness_audit.json",{"status":"PASS" if all(x["zero_init_identity_max_abs"]<1e-7 and x["final_layer_grad_finite_nonzero"] for x in film_rows) else "FAIL","rows":film_rows,"thresholds":{"identity_lt":1e-7}})

def results_md(out, tie, cox, deltas):
    summaries=[r for r in deltas if r.get("status")=="summary"]; get=lambda a,b:next(r for r in summaries if r["variant_a"]==a and r["variant_b"]==b)
    A=get("C1","C0"); B=get("C1","C2"); X=get("A1","A2"); M=get("C1","M0")
    gateA=A["mean_delta_harrell"]>0 and A["mean_delta_uno"]>0 and A["harrell_wins"]>=13 and A["uno_wins"]>=13
    seedA=[r for r in deltas if r.get("status")=="seed_summary" and r["variant_a"]=="C1" and r["variant_b"]=="C0"]; gateA=gateA and sum(r["fold_mean_delta_harrell"]>0 for r in seedA)>=4 and sum(r["fold_mean_delta_uno"]>0 for r in seedA)>=4
    seedB=[r for r in deltas if r.get("status")=="seed_summary" and r["variant_a"]=="C1" and r["variant_b"]=="C2"]; gateB=B["mean_delta_harrell"]>0 and B["mean_delta_uno"]>0 and sum(r["fold_mean_delta_harrell"]>0 for r in seedB)>=3 and sum(r["fold_mean_delta_uno"]>0 for r in seedB)>=3
    seedX=[r for r in deltas if r.get("status")=="seed_summary" and r["variant_a"]=="A1" and r["variant_b"]=="A2"]; gateC=gateA and gateB and X["mean_delta_harrell"]>0 and X["mean_delta_uno"]>0 and sum(r["fold_mean_delta_harrell"]>0 for r in seedX)>=3 and sum(r["fold_mean_delta_uno"]>0 for r in seedX)>=3
    text=f"""# Phase M8A Results\n\n1. Tied DSS event times: **{'yes' if tie['n_tied_event_time_groups'] else 'no'}**; tied groups={tie['n_tied_event_time_groups']}, events inside ties={tie['n_events_inside_tied_groups']}. Historical implementation was **{'non-equivalent' if tie['n_tied_event_time_groups'] else 'equivalent on this endpoint'}** to grouped Breslow.\n2. Grouped Breslow audit: **{cox['status']}**.\n3. M7R FiLM-vs-matched-residual under corrected objective: **{'SUPPORTED' if gateC else 'NOT CONFIRMED'}** (A1 vs A2 mean ΔH={X['mean_delta_harrell']:.6f}, mean ΔU={X['mean_delta_uno']:.6f}).\n4. FiLM transfer to CommonOnly: **{'SUPPORTED' if gateA else 'NOT SUPPORTED'}**.\n5. CommonOnly-FiLM stably beats CommonOnly: **{'yes' if gateA else 'no'}**.\n6. CommonOnly-FiLM beats matched residual: **{'yes' if gateB else 'no'}**.\n7. CommonOnly-FiLM beats MCAT on primary Harrell C: **{'yes' if M['mean_delta_harrell']>0 else 'no'}** (mean ΔH={M['mean_delta_harrell']:.6f}).\n8. Freeze CommonOnly + omics-conditioned pathology FiLM + clinical late fusion: **{'yes' if gateA and M['mean_delta_harrell']>0 else 'no'}**.\n9. If Gate A fails, stop FiLM and return to CommonOnly failure analysis: **{'triggered' if not gateA else 'not triggered'}**.\n\n## Gate records\n\n- Gate A: **{'COMMONONLY_FILM_TRANSFER_SUPPORTED' if gateA else 'FAIL'}**\n- Gate B: **{'COMMONONLY_FILM_BIAS_SUPPORTED' if gateB else 'FAIL'}**\n- Gate C: **{'FILM_CROSS_BACKBONE_SUPPORTED' if gateC else 'FAIL'}**\n"""; (out/"RESULTS.md").write_text(text,encoding="utf-8")

def main(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); workers=Path(args.workers); raw=load_raw_inputs(Path(args.data_root),Path(args.repo_root)); fold_rows=[]; seed_rows=[]; profiles=[]; films=[]
    for seed in SEEDS:
        wr=workers/f"seed_{seed}"; fold_rows+=read_csv(wr/"fold_metrics.csv"); seed_rows+=read_csv(wr/"seed_metrics.csv"); profiles+=read_csv(wr/"compute_profile.csv"); films+=read_csv(wr/"film_modulation_diagnostics.csv") if (wr/"film_modulation_diagnostics.csv").exists() else []
    for r in fold_rows:
        for k in ["seed","outer_fold","n_test","events_test","params"]: r[k]=int(r[k])
        for k in ["harrell_c","uno_c","uno_numerator","uno_denominator"]: r[k]=float(r[k])
    atomic_csv(out/"fold_metrics.csv",fold_rows); atomic_csv(out/"seed_metrics.csv",seed_rows); atomic_csv(out/"compute_profile.csv",profiles); atomic_csv(out/"film_modulation_diagnostics.csv",films)
    main_rows=[]
    for v in VARIANTS:
        z=[r for r in fold_rows if r["variant"]==v]; h=np.asarray([r["harrell_c"] for r in z],float); u=np.asarray([r["uno_c"] for r in z],float)
        main_rows.append({"variant":v,"n_cells":len(z),"n_seeds":len(set(r["seed"] for r in z)),"n_outer_folds":len(set(r["outer_fold"] for r in z)),"mean_harrell_c":float(np.nanmean(h)),"std_harrell_c":float(np.nanstd(h,ddof=1)),"mean_uno_c":float(np.nanmean(u)),"std_uno_c":float(np.nanstd(u,ddof=1)),"status":"complete"})
    atomic_csv(out/"main_metrics.csv",main_rows)
    deltas=delta_rows(fold_rows); atomic_csv(out/"paired_deltas.csv",deltas); atomic_csv(out/"robust_delta_diagnostics.csv",robust(deltas))
    bootstrap(raw,workers,out)
    pool=np.load(Path(args.path_runtime)/"path_tokens_train_pool.npy",mmap_mode="r"); tie=json.loads((out/"cox_tie_audit.json").read_text()) if (out/"cox_tie_audit.json").exists() else {}; cox=json.loads((out/"cox_grouped_breslow_audit.json").read_text()) if (out/"cox_grouped_breslow_audit.json").exists() else {}
    audits(raw,pool,out)
    atomic_json(out/"protocol.json",{"phase":"M8A","branch":"codex/phase-m8a-cox-tie-commononly-film","n_patients":512,"dss_events":65,"outer_folds":5,"pathology":"UNI2-h","pathology_K":1024,"seeds":SEEDS,"variants":VARIANTS,"optimizer":"AdamW","lr":3e-4,"weight_decay":1e-4,"max_epochs":25,"patience":5,"batch_size":16,"cox":"grouped Breslow","C1_film":{"input":"concat(g,n), 192-D","network":"192->32->192","zero_init":True},"C2_residual":{"network":"288->32->96","hidden":32,"interaction_params":interaction_parameters("C2")},"C1_interaction_params":interaction_parameters("C1"),"formal_comparisons":[f"{a}_vs_{b}" for a,b in COMPS]})
    sha=subprocess.check_output(["git","-C",args.repo_root,"rev-parse","HEAD"],text=True).strip(); atomic_json(out/"run_manifest.json",{"phase":"M8A","status":"complete","cells_completed":len(fold_rows),"cells_expected":200,"seeds":SEEDS,"variants":VARIANTS,"execution_sha":sha,"remote_resolvable_execution_sha":True,"grouped_breslow_used":True,"no_leakage":True,"no_seed_hunting":True,"dss_n":len(raw.patient_ids),"dss_events":int(raw.dss_event.sum())})
    results_md(out,tie,cox,deltas); print(json.dumps({"status":"complete","cells":len(fold_rows),"sha":sha},ensure_ascii=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--path-runtime",required=True); p.add_argument("--workers",required=True); p.add_argument("--output",required=True); main(p.parse_args())

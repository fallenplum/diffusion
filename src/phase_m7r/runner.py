from __future__ import annotations
import argparse,gc,json,os,subprocess,time
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import StratifiedShuffleSplit
from src.phase_m2.common import Timer,atomic_csv,atomic_json,count_parameters,cox_loss,gpu_memory_gb,harrell_c,set_seed,stable_int,uno_c_index
from src.phase_m2.runner import PathTokenProvider
from .data import M7RFoldArrays,RawInputs,load_raw_inputs,prepare_path_cache,preprocess_m7r
from .models import DIM,M7RModel,build_model
VARIANTS=("F0","A0","F1","F2","F3","F6","B0")
MCAT_VARIANTS=("F0","A0","F1","F2","F3","F6")
PERM_VARIANTS=("F1","F2","F3","F6")
FILM_VARIANTS=("F3","F6")
LABELS={"F0":"MCAT-Original","A0":"AdaptivePath-SemanticConcat","F1":"ABMIL-SemanticConcat","F2":"ABMIL-MatchedResidualMLP","F3":"ABMIL-OmicToPath-FiLM","F6":"ABMIL-ShuffledSixSignature-FiLM","B0":"CommonOnly-OriginalFusion"}

def _git_revision(root):
    try:return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "unknown"
def _batches(indices,batch_size,rng=None):
    v=np.asarray(indices,dtype=int); v=v[rng.permutation(len(v))] if rng is not None else v; return [v[i:i+batch_size] for i in range(0,len(v),batch_size)]
def _group_tensor(arrays,indices,variant,device):
    g=arrays.shuffled_groups if variant=="F6" else arrays.mcat_groups; return torch.from_numpy(g[indices]).to(device=device,non_blocking=True)
def _tensor_batch(arrays,provider,indices,device,deterministic,epoch,variant):
    path=torch.from_numpy(provider.batch(indices,epoch=epoch,deterministic=deterministic)).to(device=device,non_blocking=True); rna=torch.from_numpy(arrays.rna[indices]).to(device=device,non_blocking=True); cna=torch.from_numpy(arrays.cna[indices]).to(device=device,non_blocking=True); clinical=torch.from_numpy(arrays.clinical[indices]).to(device=device,non_blocking=True)
    if variant in MCAT_VARIANTS:return path,rna,cna,clinical,_group_tensor(arrays,indices,variant,device)
    return path,rna,cna,clinical
def _make_model(variant,arrays,seed,fold):
    set_seed(seed+fold); widths=arrays.shuffled_widths if variant=="F6" else arrays.mcat_widths; return build_model(variant,arrays.clinical.shape[1],tuple(widths),seed,fold)
def _forward_epoch(model,arrays,provider,indices,times,events,device,batch_size,train,rng,epoch,variant):
    model.train(train); risks=[]; batches=_batches(indices,batch_size,rng if train else None); ordered=np.concatenate(batches) if batches else np.empty(0,dtype=int)
    for b in batches: risks.append(model(*_tensor_batch(arrays,provider,b,device,not train,epoch,variant))[0])
    all_risk=torch.cat(risks); loss=cox_loss(torch.tensor(times[ordered],dtype=torch.float32,device=device),torch.tensor(events[ordered],dtype=torch.float32,device=device),all_risk); return float(loss.detach().cpu()),None,loss
def _predict(model,arrays,provider,indices,device,batch_size,variant):
    model.eval(); out=[]
    with torch.no_grad():
        for b in _batches(indices,batch_size):out.append(model(*_tensor_batch(arrays,provider,b,device,True,0,variant))[0].detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
def _inner_split(raw,fold,seed):
    outer=np.flatnonzero(raw.outer_folds!=fold); strata=np.char.add(np.char.add(raw.cancer_type[outer].astype(str),"_"),raw.dss_event[outer].astype(str))
    try:
        a,b=next(StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=seed+fold).split(outer,strata)); return outer[a],outer[b]
    except ValueError:
        order=np.random.default_rng(seed+fold).permutation(outer); n=max(1,round(.15*len(order))); return order[n:],order[:n]
def parameter_breakdown(model):
    total=count_parameters(model); interaction=0; projector=0
    if isinstance(model,M7RModel):
        projector=count_parameters(model.representation.path_pool.projection); interaction=sum(count_parameters(x) for x in (model.residual,model.film_o) if x is not None); fusion=count_parameters(model.representation.fusion)+count_parameters(model.representation.clinical_gate)+count_parameters(model.representation.head)+interaction
    else:fusion=sum(p.numel() for n,p in model.named_parameters() if any(x in n.lower() for x in ("fusion","gate","head")))
    return {"total_params":total,"active_params":total,"registered_unused_params":0,"representation_params":total-fusion-projector,"projector_params":projector,"fusion_head_params":fusion,"interaction_params":interaction}
def _fit_fold(variant,raw,arrays,provider,fold,seed,device,max_epochs,patience,batch_size):
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    inner_train,inner_val=_inner_split(raw,fold,seed); timer=Timer(); model=_make_model(variant,arrays,seed,fold).to(device); breakdown=parameter_breakdown(model); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); best=-float("inf"); best_epoch=1; wait=0; rng=np.random.default_rng(seed+fold+101)
    for epoch in range(1,max_epochs+1):
        opt.zero_grad(set_to_none=True); train_loss,_,loss=_forward_epoch(model,arrays,provider,inner_train,raw.dss_time,raw.dss_event,device,batch_size,True,rng,epoch,variant); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); val=_predict(model,arrays,provider,inner_val,device,batch_size,variant); score=harrell_c(raw.dss_time[inner_val],raw.dss_event[inner_val],val); score=score if np.isfinite(score) else -train_loss
        if score>best+1e-5:best,best_epoch,wait=score,epoch,0
        else:wait+=1
        if wait>=patience:break
    del model,opt; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None
    model=_make_model(variant,arrays,seed,fold).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); outer_train=np.flatnonzero(raw.outer_folds!=fold); rng=np.random.default_rng(seed+fold+101)
    for epoch in range(1,max(1,best_epoch)+1):
        opt.zero_grad(set_to_none=True); _,_,loss=_forward_epoch(model,arrays,provider,outer_train,raw.dss_time,raw.dss_event,device,batch_size,True,rng,epoch,variant); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
    test_idx=np.flatnonzero(raw.outer_folds==fold); pred=_predict(model,arrays,provider,test_idx,device,batch_size,variant); p=parameter_breakdown(model); profile={"method":variant,"seed":seed,"outer_fold":fold,**p,"best_epoch":int(best_epoch),"inner_best_harrell_c":float(best),"training_seconds":timer.seconds,"peak_gpu_memory_gb":gpu_memory_gb(device),"outer_refit":"fresh reinitialize same seed full outer train","epochwise_sampling":True,"execution_commit":_git_revision(Path(__file__).resolve().parents[2]),"status":"completed"}; del opt; return pred,profile,model
def _audit_predict(model,arrays,path_values,indices,device,batch_size,variant):
    vals=[]; paths=[]
    model.eval()
    with torch.no_grad():
        for pos in _batches(np.arange(len(indices)),batch_size):
            b=indices[pos]; path=torch.from_numpy(np.asarray(path_values[pos],dtype=np.float16)).to(device=device); rna=torch.from_numpy(arrays.rna[b]).to(device=device); cna=torch.from_numpy(arrays.cna[b]).to(device=device); clinical=torch.from_numpy(arrays.clinical[b]).to(device=device); g=_group_tensor(arrays,b,variant,device); risk,p=model.audit_forward(path,rna,cna,clinical,g); vals.append(risk.detach().cpu().numpy()); paths.append(p.detach().cpu().numpy())
    return np.concatenate(vals),np.concatenate(paths)
def _permutation_audit(model,arrays,provider,raw,test_idx,device,batch_size,variant,seed,fold,stage="posttrain"):
    original=np.asarray(provider.deterministic[test_idx],dtype=np.float16); preds=[]; paths=[]; rngs=[]
    for perm_id in range(6):
        x=original.copy()
        if perm_id:
            for row,pid in enumerate(test_idx.tolist()):x[row]=x[row,np.random.default_rng(stable_int(f"m7r-permutation:{seed}:{fold}:{pid}:{perm_id}")).permutation(x.shape[1])]
        r,p=_audit_predict(model,arrays,x,test_idx,device,batch_size,variant); preds.append(r); paths.append(p)
    pm=np.stack(paths,axis=1); rm=np.stack(preds,axis=1); pdiff=np.max(np.abs(pm-pm[:,0:1]),axis=(1,2)); rdiff=np.max(np.abs(rm-rm[:,0:1]),axis=1); rows=[]
    for pid,dp,dr in zip(test_idx,pdiff,rdiff):rows.append({"row_type":"patient","stage":stage,"method":variant,"seed":seed,"outer_fold":fold,"patient_index":int(pid),"permutation_count":5,"max_abs_pathology_embedding_difference":float(dp),"max_abs_risk_difference":float(dr),"invariance_pass":bool(dp<=1e-5 and dr<=1e-5)})
    rows.append({"row_type":"summary","stage":stage,"method":variant,"seed":seed,"outer_fold":fold,"patient_index":"","permutation_count":5,"max_abs_pathology_embedding_difference":float(pdiff.max()),"max_abs_risk_difference":float(rdiff.max()),"pathology_diff_p95":float(np.quantile(pdiff,.95)),"risk_diff_p95":float(np.quantile(rdiff,.95)),"invariance_pass":bool(pdiff.max()<=1e-5 and rdiff.max()<=1e-5)})
    return rows
def _film_diagnostics(model,arrays,provider,test_idx,device,batch_size,variant,seed,fold):
    rows=[]; model.eval()
    for b in _batches(test_idx,batch_size):
        path=torch.from_numpy(provider.batch(b,epoch=0,deterministic=True)).to(device=device); rna=torch.from_numpy(arrays.rna[b]).to(device=device); cna=torch.from_numpy(arrays.cna[b]).to(device=device); clinical=torch.from_numpy(arrays.clinical[b]).to(device=device); g=_group_tensor(arrays,b,variant,device); d=model.modulation_diagnostics(path,rna,cna,clinical,g); n=len(b)
        for i,pid in enumerate(b):
            gam=d["gamma_o"][i]; bet=d["beta_o"][i]; rows.append({"method":variant,"seed":seed,"outer_fold":fold,"patient_index":int(pid),"mean_abs_gamma":float(gam.abs().mean()),"mean_abs_beta":float(bet.abs().mean()),"gamma_std":float(gam.std()),"beta_std":float(bet.std()),"fraction_abs_gamma_lt_0_05":float((gam.abs()<.05).float().mean()),"path_correction_norm":float(d["path_correction_norm"][i]),"path_norm":float(model.representation.encode(path,rna,cna,clinical,g)[0].path[i].norm()),"correction_over_path_norm":float(d["path_correction_norm"][i]/(model.representation.encode(path,rna,cna,clinical,g)[0].path[i].norm()+1e-8))})
    return rows
def _initialization_audit(arrays,seed,fold):
    models={v:_make_model(v,arrays,seed,fold) for v in ("F1","F2","F3","F6")}; prefixes=("residual","film_o"); shared=[k for k in models["F1"].state_dict() if not k.startswith(prefixes)]; mism={v:[k for k in shared if not torch.equal(models["F1"].state_dict()[k],models[v].state_dict()[k])] for v in ("F2","F3","F6")}; f2=count_parameters(models["F2"].residual); f3=count_parameters(models["F3"].film_o); active_equal=count_parameters(models["F3"])==count_parameters(models["F6"]); return {"status":"PASS" if all(not x for x in mism.values()) and abs(f2-f3)/f3<=.02 and active_equal else "FAIL","seed":seed,"outer_fold":fold,"shared_upstream_mismatches":mism,"F2_interaction_params":f2,"F3_interaction_params":f3,"relative_interaction_difference":abs(f2-f3)/f3,"F2_F3_match_within_2pct":abs(f2-f3)/f3<=.02,"F3_F6_active_params_identical":active_equal,"shared_components":["ABMIL patch projection/attention","SixSignatureEncoder","omic Transformer","omic gated pool","omic_rho","clinical encoder","late fusion","clinical gate","RiskHead"]}
def _film_correctness(clinical_dim,widths,seed,fold):
    out={}; a=type("A",(),{"clinical":np.zeros((1,clinical_dim),np.float32),"mcat_widths":widths,"shuffled_widths":widths})(); rows=[]; base=torch.randn(4,DIM); om=torch.randn(4,DIM)
    for v in ("F3","F6"):
        m=_make_model(v,a,seed,fold); p,o,d=m._condition(base,om); m.zero_grad(set_to_none=True); pp=base.clone().requires_grad_(); oo=om.clone().requires_grad_(); q,_,_=m._condition(pp,oo); (q.square().mean()).backward(); grad=float(m.film_o.net[-1].weight.grad.norm()); rep=_make_model(v,a,seed,fold); rows.append({"variant":v,"identity_max_abs":float((p-base).abs().max()),"gradient_norm":grad,"gradient_finite_nonzero":bool(np.isfinite(grad) and grad>0),"same_seed_byte_identical":all(torch.equal(t,rep.state_dict()[k]) for k,t in m.state_dict().items())})
    out={"status":"PASS" if all(x["identity_max_abs"]<1e-7 and x["gradient_finite_nonzero"] and x["same_seed_byte_identical"] for x in rows) else "FAIL","zero_init_threshold":1e-7,"rows":rows}; return out
def _cox_audit(raw):
    times=torch.tensor([4.,3.,2.,1.]); events=torch.tensor([1.,0.,1.,0.]); risks=torch.tensor([.2,.1,.4,.3],requires_grad=True); loss=cox_loss(times,events,risks); loss.backward(); return {"status":"PASS" if torch.isfinite(loss) and torch.isfinite(risks.grad).all() and float(risks.grad[0])>0 and float(risks.grad[2])<0 else "FAIL","implementation":"src.phase_m2.common.cox_loss exact full risk set Breslow logcumsumexp; batch_size is memory microbatch","synthetic_loss":float(loss.detach()),"gradient":risks.grad.detach().tolist(),"patient_count":int(len(raw.patient_ids)),"dss_events":int(raw.dss_event.sum()),"risk_order":"descending time","censoring":"events==1 only"}
def run(args):
    repo=Path(args.repo_root).resolve(); root=Path(os.environ[args.data_root_env]).resolve(); runtime=Path(args.runtime_root).resolve(); reports=Path(args.reports_root).resolve(); runtime.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True,warn_only=True); raw=load_raw_inputs(root,repo); cache=prepare_path_cache(raw,runtime,k=1024,pool_k=2048); deterministic=np.load(runtime/"path_tokens_deterministic.npy",mmap_mode="r"); train_pool=np.load(runtime/"path_tokens_train_pool.npy",mmap_mode="r"); device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"); seeds=[int(x) for x in args.seeds.split(",")];
    if len(seeds)!=1: raise ValueError("one seed per worker")
    seed=seeds[0]
    arrays1,shuffle_audit=preprocess_m7r(raw,train_pool,1); init=_initialization_audit(arrays1,seed,1); film=_film_correctness(arrays1.clinical.shape[1],arrays1.mcat_widths,seed,1); cox=_cox_audit(raw)
    atomic_json(reports/"initialization_audit.json",init); atomic_json(reports/"film_correctness_audit.json",film); atomic_json(reports/"cox_riskset_audit.json",cox); atomic_json(reports/"shuffled_signature_mapping_audit.json",{"shuffle_seed":7102026,"folds":{"1":shuffle_audit},"note":"counts, hashes and reproducibility metadata only; no patient-level information"})
    if any(x["status"]=="FAIL" for x in (init,film,cox)): raise RuntimeError("pre-formal audit failed")
    requested=[x.strip() for x in args.variants.split(",") if x.strip()]; bad=set(requested)-set(VARIANTS);
    if bad:raise ValueError(bad)
    cells=runtime/"cells"; cells.mkdir(exist_ok=True); state_path=runtime/"run_state.json"; state=json.loads(state_path.read_text()) if state_path.exists() else {"completed":[],"failed":[]}; fold_rows=[]; seed_rows=[]; profiles=[]; mapping={"1":shuffle_audit}; arrays_cache={1:arrays1}; started=time.perf_counter(); perm_rows=[]; film_rows=[]; pre_provider=PathTokenProvider(train_pool=train_pool,deterministic=deterministic,train_seed=seed,k=1024); pre_idx=np.flatnonzero(raw.outer_folds==1);
    for v in PERM_VARIANTS:
        pre_model=_make_model(v,arrays1,seed,1).to(device); perm_rows.extend(_permutation_audit(pre_model,arrays1,pre_provider,raw,pre_idx,device,args.batch_size,v,seed,1,stage="preformal")); del pre_model
    for variant in requested:
        oof=np.full(len(raw.patient_ids),np.nan); vr=[]
        for fold in range(1,6):
            if fold not in arrays_cache:arrays_cache[fold],mapping[str(fold)]=preprocess_m7r(raw,train_pool,fold)
            arrays=arrays_cache[fold]; test_idx=np.flatnonzero(raw.outer_folds==fold); provider=PathTokenProvider(train_pool=train_pool,deterministic=deterministic,train_seed=seed,k=1024); cell=f"{variant}|{seed}|{fold}"; stem=cell.replace("|","")
            pred,profile,model=_fit_fold(variant,raw,arrays,provider,fold,seed,device,args.max_epochs,args.patience,args.batch_size); train_idx=np.flatnonzero(raw.outer_folds!=fold); uno,num,den=uno_c_index(raw.dss_time[test_idx],raw.dss_event[test_idx],pred,raw.dss_time[train_idx],raw.dss_event[train_idx]); row={"method":variant,"label":LABELS[variant],"seed":seed,"outer_fold":fold,"n_test":len(test_idx),"events_test":int(raw.dss_event[test_idx].sum()),"harrell_c":harrell_c(raw.dss_time[test_idx],raw.dss_event[test_idx],pred),"uno_c":uno,"uno_numerator":num,"uno_denominator":den,"params":profile["total_params"],"status":"completed"}; np.save(cells/f"{stem}.npy",pred); atomic_json(cells/f"{stem}.json",{"row":row,"profile":profile}); fold_rows.append(row); vr.append(row); profiles.append(profile); oof[test_idx]=pred
            if variant in PERM_VARIANTS:perm_rows.extend(_permutation_audit(model,arrays,provider,raw,test_idx,device,args.batch_size,variant,seed,fold));
            if variant in FILM_VARIANTS:film_rows.extend(_film_diagnostics(model,arrays,provider,test_idx,device,args.batch_size,variant,seed,fold));
            state.setdefault("completed",[]).append(cell); atomic_json(state_path,state); del model; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None
        h=np.array([float(x["harrell_c"]) for x in vr]); u=np.array([float(x["uno_c"]) for x in vr]); seed_rows.append({"method":variant,"label":LABELS[variant],"seed":seed,"n_patients":len(raw.patient_ids),"dss_events":int(raw.dss_event.sum()),"fold_mean_harrell":float(h.mean()),"fold_std_harrell":float(h.std(ddof=1)),"worst_fold_harrell":float(h.min()),"fold_mean_uno":float(u.mean()),"fold_std_uno":float(u.std(ddof=1)),"worst_fold_uno":float(u.min()),"status":"completed"})
    atomic_csv(reports/"fold_metrics.csv",fold_rows); atomic_csv(reports/"seed_metrics.csv",seed_rows); atomic_csv(reports/"compute_profile.csv",profiles); atomic_csv(reports/"permutation_invariance_audit.csv",perm_rows); atomic_csv(reports/"film_modulation_diagnostics.csv",film_rows); atomic_json(reports/"run_manifest.json",{"phase":"M7R","status":"worker_completed","variants":requested,"seeds":seeds,"cells_completed":len(fold_rows),"device":str(device),"cache":cache,"n_patients":len(raw.patient_ids),"dss_events":int(raw.dss_event.sum()),"pathology_K":1024,"shuffle_seed":7102026,"execution_commit":_git_revision(repo),"no_leakage":True,"no_seed_hunting":True}); atomic_json(reports/"shuffled_signature_mapping_audit.json",{"shuffle_seed":7102026,"folds":mapping,"note":"counts, hashes and reproducibility metadata only; no patient-level information"}); print(json.dumps({"status":"worker_completed","seed":seed,"cells":len(fold_rows),"elapsed_seconds":time.perf_counter()-started},ensure_ascii=False))
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",type=Path,required=True); p.add_argument("--data-root-env",default="DIFFUSION_DATA_ROOT"); p.add_argument("--runtime-root",type=Path,required=True); p.add_argument("--reports-root",type=Path,required=True); p.add_argument("--variants",default=",".join(VARIANTS)); p.add_argument("--seeds",required=True); p.add_argument("--max-epochs",type=int,default=25); p.add_argument("--patience",type=int,default=5); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--cpu",action="store_true"); return p.parse_args()
if __name__=="__main__":run(parse_args())




from __future__ import annotations
import argparse, gc, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from src.phase_m2.common import atomic_csv, atomic_json, count_parameters, gpu_memory_gb, harrell_c, set_seed, stable_int, Timer, uno_c_index
from src.phase_m2.runner import PathTokenProvider
from src.phase_m7r.data import load_raw_inputs, preprocess_m7r
from .models import build_model, CommonOnlyModel, DIM, interaction_parameters
from .survival import cox_loss_breslow_grouped

VARIANTS=("C0","C1","C2","A0","A1","A2","M0","K0")
GROUP_VARIANTS={"A0","A1","A2","M0"}
LABELS={"C0":"CommonOnly-Original","C1":"CommonOnly-OmicToPath-FiLM","C2":"CommonOnly-MatchedResidualMLP","A0":"ABMIL-SemanticConcat","A1":"ABMIL-OmicToPath-FiLM","A2":"ABMIL-MatchedResidualMLP","M0":"MCAT-Original","K0":"ClinicalOnly"}

def git_sha(root):
    try: return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    except Exception: return "unknown"

def batches(indices, batch_size, rng=None):
    x=np.asarray(indices,dtype=int); x=x[rng.permutation(len(x))] if rng is not None else x
    return [x[i:i+batch_size] for i in range(0,len(x),batch_size)]

def tensor_batch(arrays, provider, idx, device, epoch, variant, train):
    path=torch.from_numpy(provider.batch(idx,epoch=epoch,deterministic=not train)).to(device=device,non_blocking=True)
    rna=torch.from_numpy(arrays.rna[idx]).to(device=device,non_blocking=True); cna=torch.from_numpy(arrays.cna[idx]).to(device=device,non_blocking=True); clinical=torch.from_numpy(arrays.clinical[idx]).to(device=device,non_blocking=True)
    if variant in GROUP_VARIANTS:
        return path,rna,cna,clinical,torch.from_numpy(arrays.mcat_groups[idx]).to(device=device,non_blocking=True)
    return path,rna,cna,clinical

def make_model(variant, arrays, seed, fold):
    set_seed(seed+fold)
    return build_model(variant, arrays.clinical.shape[1], tuple(arrays.mcat_widths), seed, fold)

def train_epoch(model, arrays, provider, idx, times, events, device, batch_size, epoch, variant, optimizer=None, rng=None):
    train=optimizer is not None; model.train(train); bs=batches(idx,batch_size,rng if train else None); order=np.concatenate(bs) if bs else np.empty(0,dtype=int); risks=[]
    for b in bs: risks.append(model(*tensor_batch(arrays,provider,b,device,epoch,variant,train))[0])
    risk=torch.cat(risks); t=torch.tensor(times[order],dtype=torch.float32,device=device); e=torch.tensor(events[order],dtype=torch.float32,device=device); loss=cox_loss_breslow_grouped(t,e,risk)
    if train:
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
    return float(loss.detach().cpu())

@torch.no_grad()
def predict(model, arrays, provider, idx, device, batch_size, variant):
    model.eval(); out=[]
    for b in batches(idx,batch_size): out.append(model(*tensor_batch(arrays,provider,b,device,0,variant,False))[0].detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)

def inner_split(raw, fold, seed):
    outer=np.flatnonzero(raw.outer_folds!=fold); strata=np.asarray([f"{a}_{b}" for a,b in zip(raw.cancer_type[outer],raw.dss_event[outer])])
    try:
        a,b=next(StratifiedShuffleSplit(n_splits=1,test_size=.15,random_state=seed+fold).split(outer,strata)); return outer[a],outer[b]
    except ValueError:
        p=np.random.default_rng(seed+fold).permutation(outer); n=max(1,int(round(.15*len(p)))); return p[n:],p[:n]

def parameter_profile(model, variant):
    total=count_parameters(model); inter=interaction_parameters(variant); return {"total_params":total,"interaction_params":inter,"variant":variant}

def fit_fold(variant, raw, arrays, provider, fold, seed, device, max_epochs, patience, batch_size):
    timer=Timer(); inner_train,inner_val=inner_split(raw,fold,seed); model=make_model(variant,arrays,seed,fold).to(device); profile=parameter_profile(model,variant); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); best=-1e9; best_epoch=1; wait=0; rng=np.random.default_rng(seed+fold+101)
    for epoch in range(1,max_epochs+1):
        loss=train_epoch(model,arrays,provider,inner_train,raw.dss_time,raw.dss_event,device,batch_size,epoch,variant,opt,rng); val=predict(model,arrays,provider,inner_val,device,batch_size,variant); score=harrell_c(raw.dss_time[inner_val],raw.dss_event[inner_val],val); score=score if np.isfinite(score) else -loss
        if score>best+1e-5: best=float(score); best_epoch=epoch; wait=0
        else: wait+=1
        if wait>=patience: break
    del model,opt; gc.collect();
    if device.type=="cuda": torch.cuda.empty_cache()
    model=make_model(variant,arrays,seed,fold).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); rng=np.random.default_rng(seed+fold+101); outer_train=np.flatnonzero(raw.outer_folds!=fold)
    for epoch in range(1,max(1,best_epoch)+1): train_epoch(model,arrays,provider,outer_train,raw.dss_time,raw.dss_event,device,batch_size,epoch,variant,opt,rng)
    test=np.flatnonzero(raw.outer_folds==fold); pred=predict(model,arrays,provider,test,device,batch_size,variant); profile.update({"seed":seed,"outer_fold":fold,"best_epoch":best_epoch,"inner_best_harrell_c":best,"training_seconds":timer.seconds,"peak_gpu_memory_gb":gpu_memory_gb(device),"execution_commit":git_sha(Path(__file__).resolve().parents[2])}); return pred,profile,model

def film_rows(model, arrays, provider, idx, device, batch_size, variant, seed, fold):
    if variant not in {"C1","A1"}: return []
    rows=[]; model.eval()
    for b in batches(idx,batch_size):
        with torch.no_grad():
            path,rna,cna,clinical,*rest=tensor_batch(arrays,provider,b,device,0,variant,False)
            if variant=="C1":
                enc=model.backbone.encode(path,rna,cna,clinical); base={"gene":enc.omics[:,:8].mean(1),"path":enc.pathology.mean(1),"cna":enc.omics[:,8:].mean(1)}; common={n:model.common[n](v) for n,v in base.items()}; _,g,be=model.film_condition(common["path"],common["gene"],common["cna"])
                corr=(torch.tanh(g)*torch.nn.functional.layer_norm(common["path"],(DIM,))+be).norm(dim=-1)
            else:
                d=model.modulation_diagnostics(path,rna,cna,clinical,rest[0]); g=d["gamma_o"]; be=d["beta_o"]; corr=d["path_correction_norm"]
            for i,pid in enumerate(b): rows.append({"variant":variant,"seed":seed,"outer_fold":fold,"patient_index":int(pid),"mean_abs_gamma":float(g[i].abs().mean()),"mean_abs_beta":float(be[i].abs().mean()),"path_correction_norm":float(corr[i])})
    return rows

def run(args):
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    repo=Path(args.repo_root).resolve(); root=Path(args.data_root).resolve(); worker=Path(args.worker_root).resolve(); worker.mkdir(parents=True,exist_ok=True); cells=worker/"cells"; cells.mkdir(exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu"); seeds=[int(x) for x in args.seeds.split(",")]; assert len(seeds)==1; seed=seeds[0]
    raw=load_raw_inputs(root,repo); det=np.load(Path(args.path_runtime)/"path_tokens_deterministic.npy",mmap_mode="r"); pool=np.load(Path(args.path_runtime)/"path_tokens_train_pool.npy",mmap_mode="r"); requested=[x for x in args.variants.split(",") if x]
    fold_rows=[]; seed_rows=[]; profiles=[]; film=[]; arrays_cache={}; started=time.perf_counter()
    for variant in requested:
        vr=[]
        for fold in range(1,6):
            if fold not in arrays_cache: arrays_cache[fold]=preprocess_m7r(raw,pool,fold)[0]
            arrays=arrays_cache[fold]; provider=PathTokenProvider(pool,det,train_seed=seed,k=1024); pred,prof,model=fit_fold(variant,raw,arrays,provider,fold,seed,device,args.max_epochs,args.patience,args.batch_size); test=np.flatnonzero(raw.outer_folds==fold); train=np.flatnonzero(raw.outer_folds!=fold); uno,num,den=uno_c_index(raw.dss_time[test],raw.dss_event[test],pred,raw.dss_time[train],raw.dss_event[train]); row={"variant":variant,"label":LABELS[variant],"seed":seed,"outer_fold":fold,"n_test":len(test),"events_test":int(raw.dss_event[test].sum()),"harrell_c":float(harrell_c(raw.dss_time[test],raw.dss_event[test],pred)),"uno_c":float(uno),"uno_numerator":float(num),"uno_denominator":float(den),"params":prof["total_params"],"status":"completed"}; fold_rows.append(row); vr.append(row); profiles.append({**prof,"elapsed_seconds":time.perf_counter()-started}); np.save(cells/f"{variant}_{seed}_{fold}.npy",pred); atomic_json(cells/f"{variant}_{seed}_{fold}.json",{"row":row,"profile":prof}); film.extend(film_rows(model,arrays,provider,test,device,args.batch_size,variant,seed,fold)); del model; gc.collect();
            print(json.dumps({"cell_completed":f"{variant}|{seed}|{fold}","cells_completed":len(fold_rows)},ensure_ascii=False),flush=True)
            if device.type=="cuda": torch.cuda.empty_cache()
        h=np.asarray([x["harrell_c"] for x in vr]); u=np.asarray([x["uno_c"] for x in vr]); seed_rows.append({"variant":variant,"label":LABELS[variant],"seed":seed,"n_patients":len(raw.patient_ids),"dss_events":int(raw.dss_event.sum()),"fold_mean_harrell":float(np.nanmean(h)),"fold_std_harrell":float(np.nanstd(h,ddof=1)),"fold_mean_uno":float(np.nanmean(u)),"fold_std_uno":float(np.nanstd(u,ddof=1)),"status":"completed"})
    atomic_csv(worker/"fold_metrics.csv",fold_rows); atomic_csv(worker/"seed_metrics.csv",seed_rows); atomic_csv(worker/"compute_profile.csv",profiles); atomic_csv(worker/"film_modulation_diagnostics.csv",film); atomic_json(worker/"worker_manifest.json",{"phase":"M8A","seed":seed,"variants":requested,"cells_completed":len(fold_rows),"device":str(device),"elapsed_seconds":time.perf_counter()-started,"execution_commit":git_sha(repo),"grouped_breslow_used":True,"no_leakage":True,"no_seed_hunting":True}); print(json.dumps({"status":"completed","seed":seed,"cells":len(fold_rows),"seconds":time.perf_counter()-started},ensure_ascii=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--path-runtime",required=True); p.add_argument("--worker-root",required=True); p.add_argument("--seeds",required=True); p.add_argument("--variants",default=",".join(VARIANTS)); p.add_argument("--max-epochs",type=int,default=25); p.add_argument("--patience",type=int,default=5); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--cpu",action="store_true"); run(p.parse_args())

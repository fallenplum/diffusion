from __future__ import annotations
import argparse,gc,json,os,subprocess,time
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from src.phase_m2.common import atomic_csv,atomic_json,count_parameters,gpu_memory_gb,harrell_c,set_seed,stable_int,uno_c_index,Timer
from src.phase_m2.data import load_raw_inputs,prepare_path_cache,preprocess_fold
from src.phase_m3r.runner import PathTokenProvider
from src.phase_m3r2.models import ClinicalOnlyModel,SimpleFusionModel
from .models import build_residual,ResidualRiskModel
from .survival import cox_loss_full
VARIANTS=("C0-Full","C0-NoStage","B0","RP","RO","RPO")
LABELS={"C0-Full":"ClinicalOnly-Full","C0-NoStage":"ClinicalOnly-NoStage","B0":"CommonOnly","RP":"PathologyResidual","RO":"OmicsResidual","RPO":"AdditivePathOmicsResidual"}
def git_sha(root):
 try:return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
 except Exception:return "unknown"
def batches(idx,bs,rng=None):
 x=np.asarray(idx,dtype=int);x=x[rng.permutation(len(x))] if rng is not None else x;return [x[i:i+bs] for i in range(0,len(x),bs)]
def clinical_view(arrays,variant): return arrays.clinical if variant!="C0-NoStage" else arrays.clinical[:,:2]
def tensor_batch(arrays,provider,b,device,epoch,variant,train):
 path=torch.from_numpy(provider.batch(b,epoch=epoch,deterministic=not train)).to(device=device);rna=torch.from_numpy(arrays.rna[b]).to(device=device);cna=torch.from_numpy(arrays.cna[b]).to(device=device);clin=torch.from_numpy(arrays.clinical[b] if variant!="C0-NoStage" else arrays.clinical[b,:2]).to(device=device);return path,rna,cna,clin
def anchor_forward(model,clinical,device): return model.encoder(clinical.float()) if hasattr(model,"encoder") else model(clinical)
def fit_anchor(clinical,times,events,train_idx,val_idx,seed,device,max_epochs=25,patience=5):
 set_seed(seed);m=ClinicalOnlyModel(clinical.shape[1],variant_seed=500000+stable_int(f"m8p0-anchor:{seed}"));m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);best=-1e9;be=1;wait=0
 for ep in range(1,max_epochs+1):
  m.train();r=m.encoder(torch.from_numpy(clinical[train_idx]).to(device).float());risk=m.head(r);loss=cox_loss_full(torch.tensor(times[train_idx],device=device),torch.tensor(events[train_idx],device=device,dtype=torch.float32),risk);opt.zero_grad();loss.backward();opt.step();m.eval();
  with torch.no_grad():vr=m.head(m.encoder(torch.from_numpy(clinical[val_idx]).to(device).float())).cpu().numpy()
  sc=harrell_c(times[val_idx],events[val_idx],vr);sc=sc if np.isfinite(sc) else -float(loss)
  if sc>best+1e-5:best,be,wait=float(sc),ep,0
  else:wait+=1
  if wait>=patience:break
 del opt;return m,be,best
def crossfit_anchor(clinical,times,events,outer_train,outer_test,seed,fold,device):
 strata=np.asarray([str(int(events[i])) for i in outer_train]);split=StratifiedKFold(5,shuffle=True,random_state=8402026+fold);cf=np.full(len(outer_train),np.nan);rows=[]
 for k,(tr,va) in enumerate(split.split(outer_train,strata),1):
  m,ep,sc=fit_anchor(clinical,times,events,outer_train[tr],outer_train[va],8402026+fold+k,device);m.eval()
  with torch.no_grad():cf[va]=m.head(m.encoder(torch.from_numpy(clinical[outer_train[va]]).to(device).float())).cpu().numpy()
  rows.append({"seed":seed,"outer_fold":fold,"crossfit_subfold":k,"train_count":len(tr),"heldout_count":len(va),"selected_epoch":ep,"heldout_harrell":sc,"self_training":False})
 # full outer train refit with deterministic selected epoch
 m,ep,sc=fit_anchor(clinical,times,events,outer_train,outer_test,900000+seed+fold,device)
 with torch.no_grad():test=m.head(m.encoder(torch.from_numpy(clinical[outer_test]).to(device).float())).cpu().numpy()
 del m;gc.collect();return cf,test,rows
def inner_split(raw,fold,seed):
 outer=np.flatnonzero(raw.outer_folds!=fold);strata=np.asarray([f"{raw.cancer_type[i]}_{raw.dss_event[i]}" for i in outer]);
 try:a,b=next(StratifiedShuffleSplit(1,test_size=.15,random_state=seed+fold).split(outer,strata));return outer[a],outer[b]
 except ValueError:return outer[:-max(1,len(outer)//7)],outer[-max(1,len(outer)//7):]
def direct_model(variant,dim,seed,fold):
 set_seed(seed+fold)
 if variant.startswith("C0"):return ClinicalOnlyModel(dim,variant_seed=500000+stable_int(f"m8p0-c0:{variant}:{seed}:{fold}"))
 return SimpleFusionModel(dim,variant="S3",variant_seed=600000+stable_int(f"m8p0-b0:{seed}:{fold}"),concat_seed=700000+stable_int(f"m8p0-b0-f:{seed}:{fold}"))
def direct_forward(m,arr,p,b,device,ep,v,train):return m(*tensor_batch(arr,p,b,device,ep,v,train))[0] if v=="B0" else m(*tensor_batch(arr,p,b,device,ep,v,train))[0]
def fit_direct(v,raw,arr,provider,fold,seed,device,args):
 dim=2 if v=="C0-NoStage" else arr.clinical.shape[1];tr,va=inner_split(raw,fold,seed);m=direct_model(v,dim,seed,fold).to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);best=-1e9;be=1;wait=0;rng=np.random.default_rng(seed+fold+101)
 for ep in range(1,args.max_epochs+1):
  bs=batches(tr,args.batch_size,rng);order=np.concatenate(bs);risks=[direct_forward(m,arr,provider,b,device,ep,v,True) for b in bs]
  loss=cox_loss_full(torch.tensor(raw.dss_time[order],device=device),torch.tensor(raw.dss_event[order],device=device,dtype=torch.float32),torch.cat(risks));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.eval();
  with torch.no_grad():vr=np.concatenate([direct_forward(m,arr,provider,b,device,0,v,False).cpu().numpy() for b in batches(va,args.batch_size)])
  sc=harrell_c(raw.dss_time[va],raw.dss_event[va],vr);sc=sc if np.isfinite(sc) else -float(loss)
  if sc>best+1e-5:best,be,wait=float(sc),ep,0
  else:wait+=1
  if wait>=args.patience:break
 del m,opt;gc.collect();m=direct_model(v,dim,seed,fold).to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);outer=np.flatnonzero(raw.outer_folds!=fold);rng=np.random.default_rng(seed+fold+101)
 for ep in range(1,be+1):
  bs=batches(outer,args.batch_size,rng);order=np.concatenate(bs);risks=[direct_forward(m,arr,provider,b,device,ep,v,True) for b in bs];loss=cox_loss_full(torch.tensor(raw.dss_time[order],device=device),torch.tensor(raw.dss_event[order],device=device,dtype=torch.float32),torch.cat(risks));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 test=np.flatnonzero(raw.outer_folds==fold);m.eval();
 with torch.no_grad():pred=np.concatenate([direct_forward(m,arr,provider,b,device,0,v,False).cpu().numpy() for b in batches(test,args.batch_size)])
 return pred,{"variant":v,"seed":seed,"outer_fold":fold,"params":count_parameters(m),"best_epoch":be,"inner_best_harrell":best,"status":"completed"}
def fit_residual(v,raw,arr,provider,fold,seed,device,args,cf,testoff,auditrows):
 tr,va=inner_split(raw,fold,seed);m=build_residual(v,arr.clinical.shape[1],seed,fold).to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);best=-1e9;be=1;wait=0;rng=np.random.default_rng(seed+fold+101);outer=np.flatnonzero(raw.outer_folds!=fold)
 for p in m.backbone.parameters():p.requires_grad=True
 for ep in range(1,args.max_epochs+1):
  bs=batches(tr,args.batch_size,rng);order=np.concatenate(bs);risks=[]
  for b in bs:
   off=torch.tensor(cf[np.searchsorted(outer,b)],device=device,dtype=torch.float32);risks.append(m(*tensor_batch(arr,provider,b,device,ep,v,True),off)[0])
  loss=cox_loss_full(torch.tensor(raw.dss_time[order],device=device),torch.tensor(raw.dss_event[order],device=device,dtype=torch.float32),torch.cat(risks));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.eval();
  with torch.no_grad():
   rv=[]
   for b in batches(va,args.batch_size):
    off=torch.tensor(cf[np.searchsorted(outer,b)],device=device,dtype=torch.float32);rv.append(m(*tensor_batch(arr,provider,b,device,0,v,False),off)[0].cpu().numpy())
   sc=harrell_c(raw.dss_time[va],raw.dss_event[va],np.concatenate(rv))
  if sc>best+1e-5:best,be,wait=float(sc),ep,0
  else:wait+=1
  if wait>=args.patience:break
 del m,opt;gc.collect();m=build_residual(v,arr.clinical.shape[1],seed,fold).to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(seed+fold+101)
 for ep in range(1,be+1):
  bs=batches(outer,args.batch_size,rng);order=np.concatenate(bs);risks=[]
  for b in bs:
   off=torch.tensor(cf[np.searchsorted(outer,b)],device=device,dtype=torch.float32);risks.append(m(*tensor_batch(arr,provider,b,device,ep,v,True),off)[0])
  loss=cox_loss_full(torch.tensor(raw.dss_time[order],device=device),torch.tensor(raw.dss_event[order],device=device,dtype=torch.float32),torch.cat(risks));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 test=np.flatnonzero(raw.outer_folds==fold);m.eval();pred=[];drows=[]
 with torch.no_grad():
  for b in batches(test,args.batch_size):
   off=torch.tensor(testoff[np.searchsorted(test,b)],device=device,dtype=torch.float32);path,rna,cna,clin=tensor_batch(arr,provider,b,device,0,v,False);risk,delta=m(path,rna,cna,clin,off);pred.append(risk.cpu().numpy());dp,do=m.delta_parts(path,rna,cna,clin);drows.extend({"variant":v,"seed":seed,"outer_fold":fold,"patient_index":int(i),"clinical_offset":float(off[j]),"delta_pathology":float(dp[j]),"delta_omics":float(do[j])} for j,i in enumerate(b))
 return np.concatenate(pred),{"variant":v,"seed":seed,"outer_fold":fold,"params":count_parameters(m),"best_epoch":be,"inner_best_harrell":best,"status":"completed"},auditrows,drows
def run(args):
 repo=Path(args.repo_root);root=Path(args.data_root);runtime=Path(args.runtime);worker=Path(args.worker);worker.mkdir(parents=True,exist_ok=True);(worker/"cells").mkdir(exist_ok=True);device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu");raw=load_raw_inputs(root,repo);det=np.load(runtime/"path_tokens_deterministic.npy",mmap_mode="r");pool=np.load(runtime/"path_tokens_train_pool.npy",mmap_mode="r");seed=int(args.seed);foldrows=[];profiles=[];audits=[];diag=[];arrays={}
 for v in [x for x in args.variants.split(",") if x]:
  for fold in range(1,6):
   if fold not in arrays:arrays[fold]=preprocess_fold(raw,pool,fold)
   arr=arrays[fold];provider=PathTokenProvider(pool,det,train_seed=seed,k=1024);test=np.flatnonzero(raw.outer_folds==fold);outer=np.flatnonzero(raw.outer_folds!=fold)
   if v in {"RP","RO","RPO"}:cf,toff,ar=crossfit_anchor(arr.clinical,raw.dss_time,raw.dss_event,outer,test,seed,fold,device);pred,prof,ar,dr=fit_residual(v,raw,arr,provider,fold,seed,device,args,cf,toff,ar);audits+=ar;diag+=dr
   else:pred,prof=fit_direct(v,raw,arr,provider,fold,seed,device,args)
   uno,num,den=uno_c_index(raw.dss_time[test],raw.dss_event[test],pred,raw.dss_time[outer],raw.dss_event[outer]);row={"variant":v,"label":LABELS[v],"seed":seed,"outer_fold":fold,"n_test":len(test),"events_test":int(raw.dss_event[test].sum()),"harrell_c":float(harrell_c(raw.dss_time[test],raw.dss_event[test],pred)),"uno_c":float(uno),"uno_numerator":float(num),"uno_denominator":float(den),"params":prof["params"],"status":"completed"};foldrows.append(row);profiles.append(prof);np.save(worker/"cells"/f"{v}_{seed}_{fold}.npy",pred);atomic_json(worker/"cells"/f"{v}_{seed}_{fold}.json",{"row":row,"profile":prof})
 atomic_csv(worker/"fold_metrics.csv",foldrows);atomic_csv(worker/"compute_profile.csv",profiles);atomic_csv(worker/"clinical_anchor_crossfit_audit.csv",audits);atomic_csv(worker/"residual_diagnostics.csv",diag);atomic_json(worker/"worker_manifest.json",{"seed":seed,"variants":args.variants,"cells_completed":len(foldrows),"execution_sha":git_sha(repo),"status":"complete"});print(json.dumps({"seed":seed,"cells":len(foldrows)},ensure_ascii=False))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--repo-root",required=True);p.add_argument("--data-root",required=True);p.add_argument("--runtime",required=True);p.add_argument("--worker",required=True);p.add_argument("--seed",required=True);p.add_argument("--variants",default=",".join(VARIANTS));p.add_argument("--max-epochs",type=int,default=25);p.add_argument("--patience",type=int,default=5);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--cpu",action="store_true");run(p.parse_args())


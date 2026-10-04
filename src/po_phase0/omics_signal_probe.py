from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit
from src.phase_m2.common import NEURAL_SEEDS, harrell_c, uno_c_index, atomic_json, atomic_csv, set_seed
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.po_phase0.runner import cox_loss_grouped

ALPHAS=(1e-3,1e-2,1e-1)
MODS=('RNA-only','CNA-only','RNA+CNA')

def split_inner(train, events, seed):
    try:
        sp=StratifiedShuffleSplit(1,test_size=.15,random_state=seed)
        a,b=next(sp.split(train,events[train])); return train[a],train[b]
    except ValueError:
        rng=np.random.default_rng(seed); z=rng.permutation(train); n=max(1,int(.15*len(z))); return z[n:],z[:n]

def fit_linear(x,t,e,alpha,seed,steps=220):
    set_seed(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); xx=torch.tensor(x,dtype=torch.float32,device=device); tt=torch.tensor(t,dtype=torch.float32,device=device); ee=torch.tensor(e,dtype=torch.float32,device=device); w=nn.Parameter(torch.zeros(xx.shape[1],device=device)); opt=torch.optim.Adam([w],lr=.03)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); loss=cox_loss_grouped(tt,ee,xx@w)+.5*alpha*(w*w).sum(); loss.backward(); torch.nn.utils.clip_grad_norm_([w],5.); opt.step()
    return w.detach().cpu().numpy()

def fit_mlp(x,t,e,seed,max_epochs=25,patience=5):
    set_seed(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); xx=torch.tensor(x,dtype=torch.float32,device=device); tt=torch.tensor(t,dtype=torch.float32,device=device); ee=torch.tensor(e,dtype=torch.float32,device=device); n=x.shape[1]
    def new(): return nn.Sequential(nn.LayerNorm(n),nn.Linear(n,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,1)).to(device)
    a,b=split_inner(np.arange(len(x)),e,seed+17); m=new(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4); best=-1e9; be=1;wait=0
    for ep in range(1,max_epochs+1):
        m.train();opt.zero_grad(set_to_none=True); risk=m(xx[a]).squeeze(-1); loss=cox_loss_grouped(tt[a],ee[a],risk);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.eval()
        with torch.no_grad(): val=m(xx[b]).squeeze(-1).cpu().numpy()
        score=harrell_c(t[b],e[b],val)
        if score>best+1e-5:best=score;be=ep;wait=0
        else:wait+=1
        if wait>=patience:break
    del m,opt;torch.cuda.empty_cache() if device.type=='cuda' else None
    set_seed(seed);m=new();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);m.train()
    for _ in range(max(1,be)):
        opt.zero_grad(set_to_none=True);risk=m(xx).squeeze(-1);loss=cox_loss_grouped(tt,ee,risk);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
    m.eval();
    with torch.no_grad(): pred=m(xx).squeeze(-1).cpu().numpy()
    del m,opt;torch.cuda.empty_cache() if device.type=='cuda' else None
    return pred,be,best

def main(args):
    repo=Path(args.repo_root); data=Path(os.environ['DIFFUSION_DATA_ROOT']); reports=Path(args.reports_root); reports.mkdir(parents=True,exist_ok=True); runtime=Path(args.runtime_root); pool=np.load(runtime/'path_tokens_train_pool.npy',mmap_mode='r'); raw=load_raw_inputs(data,repo); folds=[]; seeds=[]; profiles=[]; fold_cache={}; started=time.time()
    for fold in range(1,6):
        arrays=preprocess_fold(raw,pool,fold); train=np.flatnonzero(raw.outer_folds!=fold); test=np.flatnonzero(raw.outer_folds==fold); fold_cache[fold]=(arrays,train,test); xmap={'RNA-only':arrays.rna,'CNA-only':arrays.cna,'RNA+CNA':np.concatenate([arrays.rna,arrays.cna],1)}
        for mod in MODS:
            x=xmap[mod]; it,iv=split_inner(train,raw.dss_event,20260904+fold)
            scores=[]
            for alpha in ALPHAS:
                w=fit_linear(x[it],raw.dss_time[it],raw.dss_event[it],alpha,20260904+fold,steps=180); scores.append((harrell_c(raw.dss_time[iv],raw.dss_event[iv],x[iv]@w),alpha))
            scores.sort(key=lambda z:(-z[0],z[1])); alpha=scores[0][1];w=fit_linear(x[train],raw.dss_time[train],raw.dss_event[train],alpha,20260904+fold);pred=x[test]@w; uno,num,den=uno_c_index(raw.dss_time[test],raw.dss_event[test],pred,raw.dss_time[train],raw.dss_event[train]); folds.append({'method':mod,'outer_fold':fold,'alpha':alpha,'inner_harrell_c':scores[0][0],'harrell_c':harrell_c(raw.dss_time[test],raw.dss_event[test],pred),'uno_c':uno,'n_test':len(test),'events_test':int(raw.dss_event[test].sum()),'params':len(w),'status':'completed'})
        x=np.concatenate([arrays.rna,arrays.cna],1);pred,be,bst=fit_mlp(x[train],raw.dss_time[train],raw.dss_event[train],20260904+fold); # train-only fit; refit prediction below with helper using outer train and test together
        # helper needs an indexed fit, recreate on concatenated train/test so no test labels are used
        set_seed(20260904+fold); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); n=x.shape[1]
        model=nn.Sequential(nn.LayerNorm(n),nn.Linear(n,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,1)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); xx=torch.tensor(x[train],dtype=torch.float32,device=device);tt=torch.tensor(raw.dss_time[train],dtype=torch.float32,device=device);ee=torch.tensor(raw.dss_event[train],dtype=torch.float32,device=device)
        for _ in range(max(1,be)):
            opt.zero_grad(set_to_none=True);loss=cox_loss_grouped(tt,ee,model(xx).squeeze(-1));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        with torch.no_grad(): pred=model(torch.tensor(x[test],dtype=torch.float32,device=device)).squeeze(-1).cpu().numpy()
        uno,num,den=uno_c_index(raw.dss_time[test],raw.dss_event[test],pred,raw.dss_time[train],raw.dss_event[train]);folds.append({'method':'Direct-MLP-RNA+CNA','outer_fold':fold,'alpha':'NA','inner_harrell_c':bst,'harrell_c':harrell_c(raw.dss_time[test],raw.dss_event[test],pred),'uno_c':uno,'n_test':len(test),'events_test':int(raw.dss_event[test].sum()),'params':sum(p.numel() for p in model.parameters()),'best_epoch':be,'status':'completed'});del model,opt;torch.cuda.empty_cache() if device.type=='cuda' else None
    # per-seed aggregates for deterministic linear controls and 3 MLP seeds are represented by folds here; run MLP across fixed seeds separately
    for mod in MODS:
        z=[r for r in folds if r['method']==mod]; seeds.append({'method':mod,'harrell_c':np.mean([r['harrell_c'] for r in z]),'uno_c':np.mean([r['uno_c'] for r in z]),'harrell_fold_std':np.std([r['harrell_c'] for r in z],ddof=1),'uno_fold_std':np.std([r['uno_c'] for r in z],ddof=1),'status':'completed'})
    z=[r for r in folds if r['method']=='Direct-MLP-RNA+CNA'];seeds.append({'method':'Direct-MLP-RNA+CNA','harrell_c':np.mean([r['harrell_c'] for r in z]),'uno_c':np.mean([r['uno_c'] for r in z]),'harrell_fold_std':np.std([r['harrell_c'] for r in z],ddof=1),'uno_fold_std':np.std([r['uno_c'] for r in z],ddof=1),'status':'completed'})
    # permutation null: fit fixed regularized RNA+CNA linear model on shuffled training event labels
    null=[]
    for rep in range(10):
        for fold in range(1,6):
            arrays,train,test=fold_cache[fold];x=np.concatenate([arrays.rna,arrays.cna],1);e=raw.dss_event[train].copy();np.random.default_rng(9000+rep).shuffle(e);w=fit_linear(x[train],raw.dss_time[train],e,1e-2,9000+rep+fold,steps=180);pred=x[test]@w;u,_,_=uno_c_index(raw.dss_time[test],raw.dss_event[test],pred,raw.dss_time[train],raw.dss_event[train]);null.append({'rep':rep,'outer_fold':fold,'harrell_c':harrell_c(raw.dss_time[test],raw.dss_event[test],pred),'uno_c':u})
    atomic_csv(reports/'fold_metrics.csv',folds);atomic_csv(reports/'summary.csv',seeds);atomic_csv(reports/'permutation_null.csv',null);atomic_json(reports/'manifest.json',{'phase':'PO-omics-signal-probe','cohort':len(raw.patient_ids),'events':int(raw.dss_event.sum()),'folds':5,'alpha_grid':list(ALPHAS),'permutation_reps':10,'methods':MODS+('Direct-MLP-RNA+CNA',),'no_clinical':True,'runtime_seconds':time.time()-started})
    lines=['# Omics DSS signal probe','',f'- Cohort: {len(raw.patient_ids)} patients / {int(raw.dss_event.sum())} DSS events.','- Fold-local M2R2 RNA/CNA preprocessing; no Clinical fields.','', '|Method|Harrell C fold mean +/- std|Uno C fold mean +/- std|','|---|---:|---:|']
    for r in seeds:lines.append(f"|{r['method']}|{r['harrell_c']:.4f} +/- {r['harrell_fold_std']:.4f}|{r['uno_c']:.4f} +/- {r['uno_fold_std']:.4f}|")
    nh=np.array([r['harrell_c'] for r in null]);nu=np.array([r['uno_c'] for r in null]);lines += ['',f'- Permutation null (50 outer fold replicates): Harrell {nh.mean():.4f} +/- {nh.std(ddof=1):.4f}; Uno {nu.mean():.4f} +/- {nu.std(ddof=1):.4f}.','', '## Interpretation','', '- If RNA-only/CNA-only/RNA+CNA linear controls exceed the permutation null while Direct-MLP remains near them, the encoder is limiting performance.', '- If all real-data controls remain near the permutation null, the available omics data contain little independently transferable DSS signal.', '- The existing Phase 0 Omics-only result is recorded separately as the chunk-token encoder reference: Harrell 0.5183 +/- 0.0115; Uno 0.4822 +/- 0.0263.']
    (reports/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo-root',required=True);p.add_argument('--runtime-root',required=True);p.add_argument('--reports-root',required=True);main(p.parse_args())

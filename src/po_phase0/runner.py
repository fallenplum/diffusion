from __future__ import annotations
import argparse, gc, json, os, platform, subprocess, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit
from ..phase_m2.common import NEURAL_SEEDS, BASELINE_SEED, stable_int, set_seed, harrell_c, uno_c_index, aggregate_uno, count_parameters, gpu_memory_gb, atomic_json, atomic_csv, Timer
from ..phase_m2.data import load_raw_inputs, preprocess_fold
from ..phase_m2.nn_blocks import PathologyEncoder, OmicsEncoder, RiskHead
from ..phase_m2.mgct import GatedAttentionPool, MutualGuidedFusionLayer
from ..phase_m2.mcat import SixSignatureEncoder
from ..phase_m2.motcat import sinkhorn_transport
from ..phase_m2.mmp import fit_morphology_prototypes, PrototypeSpecificFFN

MODEL_ORDER=('EarlyFusion-PO','MCAT-PO','MGCT-PO','CAMR-PO','MOTCat-PO','MMPTrans-PO','TransSurv-PO','Pathology-only','Omics-only')
NEURAL=MODEL_ORDER[1:]
LABELS={m:m for m in MODEL_ORDER}
MECH={
'EarlyFusion-PO':'pathology summary + RNA/CNA concat + regularized Cox','MCAT-PO':'omic-query pathology co-attention + dual transformer pooling','MGCT-PO':'bidirectional mutual-guided cross-attention + gated pooling','CAMR-PO':'invariant/specific branches + low-rank cross-modal fusion + auxiliary alignment','MOTCat-PO':'entropic OT pathology/omics token alignment','MMPTrans-PO':'outer-train morphology prototypes + Hallmark tokens + bidirectional transformer','TransSurv-PO':'adapted P/O cross-modal transformer; not canonical TransSurv','Pathology-only':'Gated ABMIL pathology encoder','Omics-only':'RNA/CNA chunk-token encoder'}
SOURCE={m:'PO Phase0 audited adaptation' for m in MODEL_ORDER}

def cox_loss_grouped(times, events, risks):
    mask=events>0; n=int(mask.sum().item())
    if n==0: return risks.sum()*0
    terms=[]
    for t in torch.unique(times[mask], sorted=True):
        d=mask & (times==t); r=times>=t
        terms.append(d.sum().to(risks.dtype)*torch.logsumexp(risks[r],0)-risks[d].sum())
    return torch.stack(terms).sum()/n

class POBackbone(nn.Module):
    def __init__(self,dim=96,path_tokens=32):
        super().__init__(); self.pathology_encoder=PathologyEncoder(dim=dim,output_tokens=path_tokens); self.omics_encoder=OmicsEncoder(dim=dim)
    def encode(self,pathology,rna,cna): return self.pathology_encoder(pathology), self.omics_encoder(rna,cna)

class ABMIL(nn.Module):
    def __init__(self,input_dim=1536,dim=96):
        super().__init__(); self.proj=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,dim),nn.GELU()); h=dim//2
        self.v=nn.Linear(dim,h); self.u=nn.Linear(dim,h); self.w=nn.Linear(h,1); self.norm=nn.LayerNorm(dim); self.head=RiskHead(dim)
    def forward(self,path):
        x=self.proj(path.float()); a=torch.softmax(self.w(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))).squeeze(-1),dim=1); return self.head(self.norm((a.unsqueeze(-1)*x).sum(1))),path.new_zeros(())

class OmicsOnly(nn.Module):
    def __init__(self,dim=96): super().__init__(); self.enc=OmicsEncoder(dim=dim); self.head=RiskHead(dim)
    def forward(self,path,rna,cna): return self.head(self.enc(rna,cna).mean(1)),path.new_zeros(())

class EarlyPO(nn.Module):
    def __init__(self): super().__init__(); self.path=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,96),nn.GELU()); self.omics=OmicsEncoder(dim=96); self.head=nn.Linear(96+96,1,bias=False)
    def forward(self,path,rna,cna): return self.head(torch.cat([self.path(path.float()).mean(1),self.omics(rna,cna).mean(1)],1)).squeeze(-1),path.new_zeros(())

class MCATPO(nn.Module):
    def __init__(self,widths):
        super().__init__(); self.back=POBackbone(); self.sig=SixSignatureEncoder(widths); d=96; self.co=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True); self.pt=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*2,dropout=.25,batch_first=True),2); self.ot=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*2,dropout=.25,batch_first=True),2); self.pp=GatedAttentionPool(d); self.op=GatedAttentionPool(d); self.pr=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(.25)); self.or_=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(.25)); self.mm=nn.Sequential(nn.Linear(d*2,d),nn.ReLU(),nn.Linear(d,d),nn.ReLU()); self.head=RiskHead(d)
    def forward(self,p,r,c,g):
        hp,_=self.back.encode(p,r,c); ho=self.sig(g); pt,_=self.co(ho,hp,hp); pt=self.pt(pt); ot=self.ot(ho); ap,_=self.pp(pt); ao,_=self.op(ot); return self.head(self.mm(torch.cat([self.pr(ap),self.or_(ao)],1))),p.new_zeros(())

class MGCTPO(nn.Module):
    def __init__(self):
        super().__init__(); self.back=POBackbone(); d=96; self.s1=MutualGuidedFusionLayer(d); self.s2=nn.ModuleList([MutualGuidedFusionLayer(d) for _ in range(2)]); self.head=RiskHead(d)
    def forward(self,p,r,c):
        hp,ho=self.back.encode(p,r,c); hp,ho,f=self.s1(hp,ho)
        for layer in self.s2: hp,ho,f=layer(torch.cat([hp,f],1),torch.cat([ho,f],1))
        return self.head(f[:,0]),p.new_zeros(())

class CAMRPO(nn.Module):
    def __init__(self):
        super().__init__(); d=96; self.back=POBackbone(); self.common=nn.ModuleDict({n:nn.Sequential(nn.Linear(d,d*2),nn.ReLU(),nn.Linear(d*2,d),nn.ReLU()) for n in ('gene','path','cna')}); self.specific=nn.ModuleDict({n:nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(.1)) for n in ('gene','path','cna')}); self.reconstruct=nn.ModuleDict({n:nn.Sequential(nn.Linear(d*2,d*2),nn.ReLU(),nn.Linear(d*2,d)) for n in ('gene','path','cna')}); self.lmf_a=nn.Parameter(torch.randn(16,d+1,d)*.02); self.lmf_b=nn.Parameter(torch.randn(16,d+1,d)*.02); self.lmf_c=nn.Parameter(torch.randn(16,d+1,d)*.02); self.lmf_w=nn.Parameter(torch.randn(1,16)*.02); self.bias=nn.Parameter(torch.zeros(1,d)); self.cm=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True); self.norm=nn.LayerNorm(d); self.disc=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,3)); self.head=RiskHead(d)
    def lmf(self,a,b,c):
        one=a.new_ones((a.shape[0],1)); a=torch.cat([one,a],1); b=torch.cat([one,b],1); c=torch.cat([one,c],1); prod=torch.matmul(a,self.lmf_a)*torch.matmul(b,self.lmf_b)*torch.matmul(c,self.lmf_c); return torch.matmul(self.lmf_w,prod.permute(1,0,2)).squeeze(1)+self.bias
    def forward(self,p,r,c):
        hp,ho=self.back.encode(p,r,c); base={'gene':ho[:,:8].mean(1),'path':hp.mean(1),'cna':ho[:,8:].mean(1)}; com={n:self.common[n](v) for n,v in base.items()}; spec={n:self.specific[n](v) for n,v in base.items()}; l=self.lmf(com['gene'],com['path'],com['cna']); toks=torch.stack([com['gene'],com['path'],com['cna']],1); att,_=self.cm(toks,toks,toks); fused=self.norm(att+toks).mean(1)+l; rec=sum((self.reconstruct[n](torch.cat([com[n],spec[n]],1))-base[n]).square().mean() for n in base)/3; ortho=sum((com[n]*spec[n]).sum(1).square().mean() for n in base)/3; di=torch.cat([com[n] for n in ('gene','path','cna')],0); y=torch.arange(3,device=di.device).repeat_interleave(com['gene'].shape[0]); adv=nn.functional.cross_entropy(self.disc(di),y); return self.head(fused),.10*rec+.01*ortho+.05*adv

class MOTCatPO(nn.Module):
    def __init__(self):
        super().__init__(); d=96; self.back=POBackbone(); self.pp=nn.Linear(d,d); self.op=nn.Linear(d,d); self.pu=nn.Sequential(nn.Linear(d,d),nn.GELU()); self.ou=nn.Sequential(nn.Linear(d,d),nn.GELU()); self.pn=nn.LayerNorm(d); self.on_=nn.LayerNorm(d); self.p_pool=GatedAttentionPool(d); self.o_pool=GatedAttentionPool(d); self.f=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.LayerNorm(d)); self.head=RiskHead(d)
    def forward(self,p,r,c):
        hp,ho=self.back.encode(p,r,c); hp=self.pp(hp); ho=self.op(ho); sim=torch.matmul(hp,ho.transpose(1,2))/np.sqrt(96); cou=sinkhorn_transport(sim); hp=self.pn(hp+self.pu(torch.matmul(cou,ho))); ho=self.on_(ho+self.ou(torch.matmul(cou.transpose(1,2),hp))); ap,_=self.p_pool(hp); ao,_=self.o_pool(ho); pathm=cou.sum(-1); om=cou.sum(-2); st=(pathm-1/hp.shape[1]).square().mean()+(om-1/ho.shape[1]).square().mean(); return self.head(self.f(torch.cat([ap,ao],1))),.01*st

class MMPPO(nn.Module):
    def __init__(self,proto,widths):
        super().__init__(); d=96; self.back=POBackbone(); self.register_buffer('proto',torch.from_numpy(np.asarray(proto,dtype=np.float32)),persistent=False); self.mp=nn.Linear(1536,d); self.mffn=PrototypeSpecificFFN(d,len(proto)); self.widths=tuple(max(1,int(x)) for x in widths); self.enc=nn.ModuleList([nn.Sequential(nn.LayerNorm(w),nn.Linear(w,d),nn.GELU()) for w in self.widths]); self.cross=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True); self.rev=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True); self.pp=GatedAttentionPool(d); self.op=GatedAttentionPool(d); self.f=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.LayerNorm(d)); self.head=RiskHead(d)
    def forward(self,p,r,c,g):
        hp,ho=self.back.encode(p,r,c); pr=self.proto.to(p.device,dtype=p.dtype); vp=self.mp(p.float()); pp=self.mp(pr.float()); ass=torch.softmax(torch.matmul(vp,pp.T)/np.sqrt(96),1); morph=self.mffn(torch.matmul(ass.transpose(1,2),vp)); ts=torch.cat([e(g[:,i,:w].float()).unsqueeze(1) for i,(e,w) in enumerate(zip(self.enc,self.widths))],1); mu,_=self.cross(morph,ts,ts); tu,_=self.rev(ts,morph,morph); morph=morph+mu; ts=ts+tu; a,_=self.pp(morph); b,_=self.op(ts); return self.head(self.f(torch.cat([a,b],1))),p.new_zeros(())

class TransPO(nn.Module):
    def __init__(self):
        super().__init__(); d=96; self.back=POBackbone(); self.pt=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*2,dropout=.1,activation='gelu',batch_first=True,norm_first=True),2); self.ot=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*2,dropout=.1,activation='gelu',batch_first=True,norm_first=True),2); self.p2o=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True); self.o2p=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True); self.pn=nn.LayerNorm(d); self.on_=nn.LayerNorm(d); self.fuse=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.LayerNorm(d)); self.head=RiskHead(d)
    def forward(self,p,r,c):
        hp,ho=self.back.encode(p,r,c); hp=self.pt(hp); ho=self.ot(ho); pu,_=self.p2o(hp,ho,ho); ou,_=self.o2p(ho,hp,hp); hp=self.pn(hp+pu).mean(1); ho=self.on_(ho+ou).mean(1); return self.head(self.fuse(torch.cat([hp,ho],1))),p.new_zeros(())

def make_model(name,arrays,proto=None):
    if name=='Pathology-only': return ABMIL()
    if name=='Omics-only': return OmicsOnly()
    if name=='EarlyFusion-PO': return EarlyPO()
    if name=='MCAT-PO': return MCATPO(arrays.mcat_widths)
    if name=='MGCT-PO': return MGCTPO()
    if name=='CAMR-PO': return CAMRPO()
    if name=='MOTCat-PO': return MOTCatPO()
    if name=='MMPTrans-PO': return MMPPO(proto,arrays.mmp_widths)
    if name=='TransSurv-PO': return TransPO()
    raise ValueError(name)

def inputs(arrays,idx,device,name,provider,det=False,epoch=0):
    p=torch.from_numpy(provider(idx,det,epoch)).to(device); r=torch.from_numpy(arrays.rna[idx]).to(device); c=torch.from_numpy(arrays.cna[idx]).to(device)
    if name=='MCAT-PO': return (p,r,c,torch.from_numpy(arrays.mcat_groups[idx]).to(device))
    if name=='MMPTrans-PO': return (p,r,c,torch.from_numpy(arrays.mmp_groups[idx]).to(device))
    if name=='Pathology-only': return (p,)
    return (p,r,c)

def fit_cell(name,raw,arrays,train_pool,det,fold,seed,device,max_epochs,patience,batch_size,proto):
    set_seed(seed+fold); timer=Timer(); train=np.flatnonzero(raw.outer_folds!=fold); test=np.flatnonzero(raw.outer_folds==fold)
    strata=np.char.add(raw.cancer_type[train].astype(str),'_'+raw.dss_event[train].astype(str))
    try:
        sp=StratifiedShuffleSplit(1,test_size=.15,random_state=seed+fold); a,b=next(sp.split(train,strata)); it,iv=train[a],train[b]
    except ValueError:
        rng=np.random.default_rng(seed+fold); z=rng.permutation(train); n=max(1,int(.15*len(z))); it,iv=z[n:],z[:n]
    def provider(idx,deterministic,epoch):
        if deterministic:return np.asarray(det[idx],dtype=np.float16)
        out=np.empty((len(idx),1024,train_pool.shape[2]),dtype=np.float16)
        for row,i in enumerate(idx):
            rng=np.random.default_rng(seed ^ stable_int(f'patient:{int(i)}') ^ stable_int(f'epoch:{epoch}')); ch=rng.choice(train_pool.shape[1],1024,replace=False); out[row]=train_pool[int(i),ch]
        return out
    model=make_model(name,arrays,proto).to(device); params=count_parameters(model); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); best=-1e9; be=1; wait=0; rng=np.random.default_rng(seed+fold+101)
    for ep in range(1,max_epochs+1):
        model.train(); order=it[rng.permutation(len(it))]; risks=[]; losses=[]
        for s in range(0,len(order),batch_size):
            ix=order[s:s+batch_size]; opt.zero_grad(set_to_none=True); risk,aux=model(*inputs(arrays,ix,device,name,provider,False,ep)); risks.append(risk); losses.append(aux)
        rr=torch.cat(risks); tt=torch.tensor(raw.dss_time[order],dtype=torch.float32,device=device); ee=torch.tensor(raw.dss_event[order],dtype=torch.float32,device=device); loss=cox_loss_grouped(tt,ee,rr)+torch.stack(losses).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval(); preds=[]
        with torch.no_grad():
            for s in range(0,len(iv),batch_size): preds.append(model(*inputs(arrays,iv[s:s+batch_size],device,name,provider,True,0))[0].cpu().numpy())
        vc=harrell_c(raw.dss_time[iv],raw.dss_event[iv],np.concatenate(preds)); score=vc if np.isfinite(vc) else -float(loss.detach().cpu());
        if score>best+1e-5:best=score;be=ep;wait=0
        else:wait+=1
        if wait>=patience:break
    del model,opt;gc.collect();torch.cuda.empty_cache() if device.type=='cuda' else None
    set_seed(seed+fold); model=make_model(name,arrays,proto).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); rng=np.random.default_rng(seed+fold+101)
    model.train()
    order=train[rng.permutation(len(train))]
    for ep in range(1,max(1,be)+1):
        risks=[]; losses=[]
        for s in range(0,len(order),batch_size):
            ix=order[s:s+batch_size]; opt.zero_grad(set_to_none=True); risk,aux=model(*inputs(arrays,ix,device,name,provider,False,ep)); risks.append(risk); losses.append(aux)
        rr=torch.cat(risks); tt=torch.tensor(raw.dss_time[order],dtype=torch.float32,device=device); ee=torch.tensor(raw.dss_event[order],dtype=torch.float32,device=device); loss=cox_loss_grouped(tt,ee,rr)+torch.stack(losses).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    model.eval(); preds=[]
    with torch.no_grad():
        for s in range(0,len(test),batch_size): preds.append(model(*inputs(arrays,test[s:s+batch_size],device,name,provider,True,0))[0].cpu().numpy())
    pred=np.concatenate(preds); prof={'method':name,'seed':seed,'outer_fold':fold,'params':params,'best_epoch':be,'inner_best_harrell_c':float(best),'training_seconds':timer.seconds,'peak_gpu_memory_gb':gpu_memory_gb(device),'status':'completed'}; del model,opt;gc.collect();torch.cuda.empty_cache() if device.type=='cuda' else None
    return pred,prof

def b0_cell(raw,arrays,train_pool,det,fold,seed,device):
    train=np.flatnonzero(raw.outer_folds!=fold);test=np.flatnonzero(raw.outer_folds==fold); ptrain=train_pool[train].astype(np.float32).mean(1);ptest=det[test].astype(np.float32).mean(1); x=np.concatenate([ptrain,arrays.rna[train],arrays.cna[train]],1);xt=np.concatenate([ptest,arrays.rna[test],arrays.cna[test]],1); model=nn.Linear(x.shape[1],1,bias=False).to(device); nn.init.zeros_(model.weight); opt=torch.optim.Adam([model.weight],lr=.03); tt=torch.tensor(raw.dss_time[train],dtype=torch.float32,device=device);ee=torch.tensor(raw.dss_event[train],dtype=torch.float32,device=device); xx=torch.tensor(x,dtype=torch.float32,device=device)
    for _ in range(220):opt.zero_grad();risk=model(xx).squeeze();loss=cox_loss_grouped(tt,ee,risk)+.005*model.weight.square().sum();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
    pred=model(torch.tensor(xt,dtype=torch.float32,device=device)).squeeze().detach().cpu().numpy();return pred,{'method':'EarlyFusion-PO','seed':seed,'outer_fold':fold,'params':int(x.shape[1]),'best_epoch':220,'inner_best_harrell_c':float('nan'),'training_seconds':0.,'peak_gpu_memory_gb':gpu_memory_gb(device),'status':'completed'}

def main(args):
    repo=Path(args.repo_root).resolve(); data=Path(os.environ['DIFFUSION_DATA_ROOT']).resolve(); runtime=Path(args.runtime_root); reports=Path(args.reports_root); runtime.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True); raw=load_raw_inputs(data,repo);det=np.load(runtime/'path_tokens_deterministic.npy',mmap_mode='r');pool=np.load(runtime/'path_tokens_train_pool.npy',mmap_mode='r'); device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'); requested=[x.strip() for x in args.models.split(',') if x.strip()]; seeds=NEURAL_SEEDS if args.seeds=='all' else tuple(int(x) for x in args.seeds.split(',')); cellroot=runtime/'cells';cellroot.mkdir(exist_ok=True); done=[]; failures=[]; profiles=[]; foldrows=[];seedrows=[]; proto_cache={};
    for name in requested:
      for seed in seeds:
       oof=np.full(len(raw.patient_ids),np.nan); sr=[]
       for fold in range(1,6):
        stem=f'{name}_{seed}_{fold}'.replace('-','_'); cf=cellroot/(stem+'.json'); nf=cellroot/(stem+'.npy')
        if cf.exists() and nf.exists():
         pay=json.loads(cf.read_text());row=pay['row'];prof=pay['profile'];pred=np.load(nf);done.append(f'{name}|{seed}|{fold}')
        else:
         try:
          arrays=preprocess_fold(raw,pool,fold); proto=None
          if name=='MMPTrans-PO':
           if fold not in proto_cache:
            ti=np.flatnonzero(raw.outer_folds!=fold); sm=np.stack([np.asarray(pool[int(i),::128,:],dtype=np.float16) for i in ti]); proto,_=fit_morphology_prototypes(sm,BASELINE_SEED+fold,sampled_stride=1);proto_cache[fold]=proto
           proto=proto_cache[fold]
          if name=='EarlyFusion-PO':pred,prof=b0_cell(raw,arrays,pool,det,fold,seed,device)
          else:pred,prof=fit_cell(name,raw,arrays,pool,det,fold,seed,device,args.max_epochs,args.patience,args.batch_size,proto)
          te=np.flatnonzero(raw.outer_folds==fold); tr=np.flatnonzero(raw.outer_folds!=fold); uno,num,den=uno_c_index(raw.dss_time[te],raw.dss_event[te],pred,raw.dss_time[tr],raw.dss_event[tr]); row={'method':name,'seed':seed,'outer_fold':fold,'n_test':len(te),'events_test':int(raw.dss_event[te].sum()),'harrell_c':harrell_c(raw.dss_time[te],raw.dss_event[te],pred),'uno_c':uno,'uno_numerator':num,'uno_denominator':den,'params':prof['params'],'status':'completed'};np.save(nf,pred);atomic_json(cf,{'row':row,'profile':prof});done.append(f'{name}|{seed}|{fold}')
         except Exception as e: failures.append({'method':name,'seed':seed,'fold':fold,'error':repr(e),'traceback':traceback.format_exc()});continue
        foldrows.append(row);profiles.append(prof);sr.append(row);te=np.flatnonzero(raw.outer_folds==fold);oof[te]=pred
       if len(sr)==5:seedrows.append({'method':name,'seed':seed,'n_patients':len(raw.patient_ids),'dss_events':int(raw.dss_event.sum()),'harrell_c':harrell_c(raw.dss_time,raw.dss_event,oof),'uno_c':aggregate_uno(sr),'status':'completed'})
       else:seedrows.append({'method':name,'seed':seed,'status':'failed'})
    atomic_json(reports/'run_manifest.json',{'phase':'PO-Phase0','status':'running','cohort':len(raw.patient_ids),'events':int(raw.dss_event.sum()),'models':requested,'seeds':list(seeds),'device':str(device),'git_revision':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()})
    atomic_csv(reports/(f'fold_metrics_{args.tag}.csv'),foldrows);atomic_csv(reports/(f'seed_metrics_{args.tag}.csv'),seedrows);atomic_csv(reports/(f'compute_profile_{args.tag}.csv'),profiles);atomic_json(reports/(f'failures_{args.tag}.json'),failures);atomic_json(runtime/(f'run_{args.tag}.json'),{'completed':done,'failures':failures});print(json.dumps({'tag':args.tag,'done':len(done),'failures':len(failures),'reports':str(reports)},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo-root',required=True);p.add_argument('--runtime-root',required=True);p.add_argument('--reports-root',required=True);p.add_argument('--models',required=True);p.add_argument('--seeds',default='all');p.add_argument('--tag',required=True);p.add_argument('--max-epochs',type=int,default=25);p.add_argument('--patience',type=int,default=5);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--cpu',action='store_true');main(p.parse_args())



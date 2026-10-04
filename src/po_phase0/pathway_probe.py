from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit
from src.phase_m2.common import NEURAL_SEEDS,harrell_c,uno_c_index,set_seed,atomic_json,atomic_csv
from src.phase_m2.data import load_raw_inputs,preprocess_fold
from src.phase_m2.mcat import SixSignatureEncoder
from src.phase_m2.nn_blocks import ChunkTokenEncoder,RiskHead
from src.po_phase0.runner import cox_loss_grouped

class GatedPool(nn.Module):
 def __init__(self,d):
  super().__init__();h=max(16,d//2);self.v=nn.Linear(d,h);self.u=nn.Linear(d,h);self.w=nn.Linear(h,1)
 def forward(self,x):
  a=torch.softmax(self.w(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))).squeeze(-1),1);return (a.unsqueeze(-1)*x).sum(1)
class PathwayEncoder(nn.Module):
 def __init__(self,widths,d=64):
  super().__init__();self.widths=tuple(int(x) for x in widths);self.enc=nn.ModuleList([nn.Sequential(nn.LayerNorm(max(1,w)),nn.Linear(max(1,w),d),nn.GELU()) for w in self.widths]);self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*2,dropout=.1,batch_first=True),1);self.pool=GatedPool(d);self.head=RiskHead(d)
 def tokens(self,g):return torch.cat([e(g[:,i,:w].float()).unsqueeze(1) for i,(e,w) in enumerate(zip(self.enc,self.widths))],1)
 def forward(self,g):return self.head(self.pool(self.tr(self.tokens(g))))
class PathwayCNA(nn.Module):
 def __init__(self,widths,d=64):
  super().__init__();self.p=PathwayEncoder(widths,d);self.cna=ChunkTokenEncoder(256,4,d);self.f=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.LayerNorm(d));self.head=RiskHead(d)
 def forward(self,g,c):return self.head(self.f(torch.cat([self.p.pool(self.p.tr(self.p.tokens(g))),self.cna(c).mean(1)],1)))
class SignaturePO(nn.Module):
 def __init__(self,widths,d=64):
  super().__init__();self.s=SixSignatureEncoder(widths,d);self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*2,dropout=.1,batch_first=True),1);self.pool=GatedPool(d);self.head=RiskHead(d)
 def forward(self,g):return self.head(self.pool(self.tr(self.s(g))))
def split(train,e,seed):
 try:
  a,b=next(StratifiedShuffleSplit(1,test_size=.15,random_state=seed).split(train,e[train]));return train[a],train[b]
 except ValueError:
  z=np.random.default_rng(seed).permutation(train);n=max(1,int(.15*len(z)));return z[n:],z[:n]
def inps(name,arrays,ix,dev):
 g=torch.from_numpy(arrays.mcat_groups[ix] if name=='Signature-RNA+CNA' else arrays.mmp_groups[ix]).to(dev)
 if name=='Pathway-RNA+CNA':return g,torch.from_numpy(arrays.cna[ix]).to(dev)
 return (g,)
def make(name,arrays):
 if name=='Pathway-RNA':return PathwayEncoder(arrays.mmp_widths)
 if name=='Pathway-RNA+CNA':return PathwayCNA(arrays.mmp_widths)
 if name=='Signature-RNA+CNA':return SignaturePO(arrays.mcat_widths)
 raise ValueError(name)
def fit(name,arrays,raw,tr,te,seed,device):
 set_seed(seed);it,iv=split(tr,raw.dss_event,seed+101);m=make(name,arrays).to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);best=-1e9;be=1;wait=0
 for ep in range(1,26):
  m.train();opt.zero_grad(set_to_none=True);risk=m(*inps(name,arrays,it,device));tt=torch.tensor(raw.dss_time[it],dtype=torch.float32,device=device);ee=torch.tensor(raw.dss_event[it],dtype=torch.float32,device=device);loss=cox_loss_grouped(tt,ee,risk);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.eval()
  with torch.no_grad():pred=m(*inps(name,arrays,iv,device)).detach().cpu().numpy()
  score=harrell_c(raw.dss_time[iv],raw.dss_event[iv],pred)
  if score>best+1e-5:best=score;be=ep;wait=0
  else:wait+=1
  if wait>=5:break
 del m,opt;torch.cuda.empty_cache() if device.type=='cuda' else None
 set_seed(seed);m=make(name,arrays).to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);m.train();tt=torch.tensor(raw.dss_time[tr],dtype=torch.float32,device=device);ee=torch.tensor(raw.dss_event[tr],dtype=torch.float32,device=device)
 for _ in range(max(1,be)):
  opt.zero_grad(set_to_none=True);loss=cox_loss_grouped(tt,ee,m(*inps(name,arrays,tr,device)));loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval();
 with torch.no_grad():pred=m(*inps(name,arrays,te,device)).detach().cpu().numpy()
 u,_,_=uno_c_index(raw.dss_time[te],raw.dss_event[te],pred,raw.dss_time[tr],raw.dss_event[tr]);params=sum(x.numel() for x in m.parameters());del m,opt;torch.cuda.empty_cache() if device.type=='cuda' else None
 return {'method':name,'seed':seed,'outer_fold':int(raw.outer_folds[te[0]]),'harrell_c':harrell_c(raw.dss_time[te],raw.dss_event[te],pred),'uno_c':u,'best_epoch':be,'inner_best_harrell':best,'params':params,'n_test':len(te),'events_test':int(raw.dss_event[te].sum()),'status':'completed'}
def main(a):
 repo=Path(a.repo_root);raw=load_raw_inputs(Path(os.environ['DIFFUSION_DATA_ROOT']),repo);runtime=Path(a.runtime_root);pool=np.load(runtime/'path_tokens_train_pool.npy',mmap_mode='r');reports=Path(a.reports_root);reports.mkdir(parents=True,exist_ok=True);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');names=[x for x in a.models.split(',') if x];seeds=list(NEURAL_SEEDS if a.seeds=='all' else [int(x) for x in a.seeds.split(',')]);rows=[];start=time.time()
 for fold in range(1,6):
  arrays=preprocess_fold(raw,pool,fold);tr=np.flatnonzero(raw.outer_folds!=fold);te=np.flatnonzero(raw.outer_folds==fold)
  for name in names:
   for seed in seeds:rows.append(fit(name,arrays,raw,tr,te,int(seed),dev))
 summary=[]
 for name in names:
  z=[x for x in rows if x['method']==name];summary.append({'method':name,'harrell_c_seed_mean':float(np.mean([x['harrell_c'] for x in z])),'harrell_c_seed_std':float(np.std([x['harrell_c'] for x in z],ddof=1)),'uno_c_seed_mean':float(np.mean([x['uno_c'] for x in z])),'uno_c_seed_std':float(np.std([x['uno_c'] for x in z],ddof=1)),'harrell_c_fold_mean':float(np.mean([x['harrell_c'] for x in z])),'uno_c_fold_mean':float(np.mean([x['uno_c'] for x in z])),'params':int(np.median([x['params'] for x in z])),'status':'completed'})
 atomic_csv(reports/'fold_metrics.csv',rows);atomic_csv(reports/'summary.csv',summary);atomic_json(reports/'manifest.json',{'phase':'PO-pathway-representation-probe','cohort':len(raw.patient_ids),'events':int(raw.dss_event.sum()),'folds':5,'seeds':seeds,'models':names,'no_clinical':True,'knowledge_prior':'M2R2 fold-local MMP Hallmark groups and MCAT signature groups','runtime_seconds':time.time()-start})
 lines=['# Pathway representation probe','','- Fold-local M2R2 preprocessing; no Clinical fields.','- Hallmark groups and MCAT signatures are fixed metadata priors; no outcome-guided pathway selection.','','|Method|Harrell seed mean +/- std|Uno seed mean +/- std|Params|','|---|---:|---:|---:|']
 for x in summary:lines.append(f"|{x['method']}|{x['harrell_c_seed_mean']:.4f} +/- {x['harrell_c_seed_std']:.4f}|{x['uno_c_seed_mean']:.4f} +/- {x['uno_c_seed_std']:.4f}|{x['params']}|")
 (reports/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo-root',required=True);p.add_argument('--runtime-root',required=True);p.add_argument('--reports-root',required=True);p.add_argument('--models',default='Pathway-RNA,Pathway-RNA+CNA,Signature-RNA+CNA');p.add_argument('--seeds',default='all');main(p.parse_args())

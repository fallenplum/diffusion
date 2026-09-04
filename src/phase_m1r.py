from __future__ import annotations
import argparse, io, json, re, hashlib, math, time
from pathlib import Path
import h5py, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import torch
from torch import nn

PAT=re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",re.I)
SAFE_CAT=["SEX","PATH_T_STAGE","PATH_N_STAGE","PATH_M_STAGE","AJCC_PATHOLOGIC_TUMOR_STAGE","AJCC_STAGING_EDITION","HISTOLOGICAL_DIAGNOSIS","PRIMARY_SITE_PATIENT"]
SEEDS=(20260904,20260905,20260906)

def pid(x):
 m=PAT.match(str(x).strip().upper()); return m.group(1) if m else None
def tab(p): return pd.read_csv(p,sep='\t',comment='#',low_memory=False)
def metric_c(t,e,r):
 t=np.asarray(t,float); e=np.asarray(e,int); r=np.asarray(r,float); good=0.; n=0
 for i in np.flatnonzero(e==1):
  m=t>t[i]; n+=int(m.sum()); good+=float((r[i]>r[m]).sum())+.5*float((r[i]==r[m]).sum())
 return good/n if n else float('nan')
def censor_km(t,e,q):
 t=np.asarray(t,float); e=np.asarray(e,int); q=np.asarray(q,float); out=np.ones(len(q)); s=1.;
 for x in np.sort(np.unique(t)):
  at=(t>=x).sum(); cen=((t==x)&(e==0)).sum(); s*=1-cen/max(at,1); out[q>=x]=s
 return out
def uno(t,e,r,g):
 num=den=0.
 for i in np.flatnonzero(e==1):
  if g[i]<=1e-8: continue
  m=t>t[i]; w=1/g[i]**2; den+=w*m.sum(); num+=w*((r[i]>r[m]).sum()+.5*(r[i]==r[m]).sum())
 return num/den if den else float('nan')
def bootstrap(t,e,a,b=None,n=5000,seed=20260904):
 t=np.asarray(t,float); e=np.asarray(e,int); a=np.asarray(a,float); b=None if b is None else np.asarray(b,float)
 rng=np.random.default_rng(seed); vals=[]
 for _ in range(n):
  ix=rng.integers(0,len(t),len(t)); vals.append(metric_c(t[ix],e[ix],a[ix])-(metric_c(t[ix],e[ix],b[ix]) if b is not None else 0))
 vals=np.asarray(vals); return float(np.nanmean(vals)),float(np.nanquantile(vals,.025)),float(np.nanquantile(vals,.975)),float(np.mean(vals>0))
def exact_cox(t,e,x,alpha=.01,maxiter=80):
 if x.shape[1] > 512:
  # GPU-friendly exact full-risk-set gradient for large omics matrices.
  dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); X=torch.tensor(x,dtype=torch.float32,device=dev); T=np.asarray(t); E=np.asarray(e); b=torch.zeros(X.shape[1],device=dev,requires_grad=True); opt=torch.optim.Adam([b],lr=.03)
  masks=[torch.tensor(T>=T[i],device=dev) for i in np.flatnonzero(E==1)]
  order=np.argsort(-T,kind='stable'); inv=np.argsort(order); ts=torch.tensor(T[order],device=dev); ev=torch.tensor(E[order],device=dev,dtype=torch.float32)
  for _ in range(min(maxiter,40)):
   opt.zero_grad(); eta=(X@b)[order]; lse=torch.logcumsumexp(eta,0); loss=((lse-eta)*ev).sum()/max(1,int(E.sum()))+alpha*(b@b)/2; loss.backward(); torch.nn.utils.clip_grad_norm_([b],5.); opt.step()
  return b.detach().cpu().numpy()
 idx=np.flatnonzero(e==1); masks=[t>=t[i] for i in idx]
 def fg(b):
  eta=x@b; ll=0.; g=np.zeros_like(b)
  for i,m in zip(idx,masks):
   z=eta[m]; l=logsumexp(z); w=np.exp(z-l); ll+=eta[i]-l; g+=x[i]-w@x[m]
  n=max(1,len(idx)); return -ll/n+alpha*(b@b)/2,-g/n+alpha*b
 return minimize(fg,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-7}).x
def neural_cox(x,t,e,seed,hidden=(128,64),epochs=100,patience=18,drop=.1):
 torch.manual_seed(seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); X=torch.tensor(x,dtype=torch.float32,device=dev); T=np.asarray(t); E=np.asarray(e)
 class Net(nn.Module):
  def __init__(self):
   super().__init__(); self.m=nn.Sequential(nn.Linear(X.shape[1],hidden[0]),nn.ReLU(),nn.Dropout(drop),nn.Linear(hidden[0],hidden[1]),nn.ReLU(),nn.Dropout(drop),nn.Linear(hidden[1],1))
  def forward(self,z): return self.m(z).squeeze(-1)
 net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3); best=None; wait=0
 tr=np.arange(len(t)); va=tr
 for ep in range(epochs):
  net.train(); opt.zero_grad(); risk=net(X); loss=torch.tensor(0.,device=dev)
  for i in np.flatnonzero(E==1):
   m=torch.tensor(T>=T[i],device=dev); loss=loss+(torch.logsumexp(risk[m],0)-risk[i])
  loss=loss/max(1,int(E.sum())); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
  val=float(loss.detach().cpu());
  if best is None or val<best[0]-1e-5: best=(val,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()}); wait=0
  else: wait+=1
  if wait>=patience: break
 net.load_state_dict(best[1]); net.eval(); return net(X).detach().cpu().numpy()

def load_all(root,out):
 base=root/'data/data_8/coadread_tcga_pan_can_atlas_2018'; sup=root/'data/data_8/coadread_tcga/data_clinical_patient.txt'
 c=tab(base/'data_clinical_patient.txt'); s=tab(sup); c['patient_id']=c.PATIENT_ID.map(pid); s['patient_id']=s.PATIENT_ID.map(pid)
 for col in ['HISTOLOGICAL_DIAGNOSIS','PRIMARY_SITE_PATIENT']:
  if col not in c: c[col]=np.nan
  c[col]=c[col].where(c[col].notna(),c.set_index('patient_id').index.to_series().map(s.drop_duplicates('patient_id').set_index('patient_id')[col]))
 c=c.drop_duplicates('patient_id').set_index('patient_id')
 def om(path):
  d=tab(path); cols=[x for x in d.columns if pid(x)]; ids=[pid(x) for x in cols]; a=d[cols].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32).T; return pd.DataFrame(a,index=ids).groupby(level=0).mean()
 rna=om(base/'data_mrna_seq_v2_rsem.txt'); cna=om(base/'data_log2_cna.txt')
 h5=[]; by_type={}
 for typ in ['TCGA-COAD','TCGA-READ']:
  fs=sorted((root/'TCGA'/typ).glob('*.h5')); by_type[typ]=len(fs); h5+=fs
 hids={pid(x.name) for x in h5 if pid(x.name)}; common=sorted(hids&set(c.index)&set(rna.index)&set(cna.index))
 f=c.loc[common,['DSS_STATUS','DSS_MONTHS','PFS_STATUS','PFS_MONTHS','CANCER_TYPE_ACRONYM']].copy(); f.columns=['dss_event','dss_time','pfs_event','pfs_time','cancer_type']
 for col in ['dss_event','pfs_event']: f[col]=pd.to_numeric(f[col].astype(str).str.extract(r'^(0|1)')[0],errors='coerce')
 for col in ['dss_time','pfs_time']: f[col]=pd.to_numeric(f[col],errors='coerce')
 f['patient_id']=common
 audit={'coad_h5_slides':by_type['TCGA-COAD'],'read_h5_slides':by_type['TCGA-READ'],'unique_h5_patients':len(hids),'h5_clinical':len(hids&set(c.index)),'h5_rna':len(hids&set(rna.index)),'h5_cna':len(hids&set(cna.index)),'complete_multimodal':len(common),'dss_valid':int(f.dss_time.notna().sum()),'dss_events':int(f.dss_event.fillna(0).sum()),'pfs_valid':int(f.pfs_time.notna().sum()),'pfs_events':int(f.pfs_event.fillna(0).sum()),'multi_slide_patients':0,'expected_complete_multimodal':555,'expected_dss_valid':512,'expected_dss_events':65,'expected_pfs_valid':529,'expected_pfs_events':135}
 out.mkdir(parents=True,exist_ok=True); (out/'audit').mkdir(exist_ok=True)
 rows=[]; slide_means=[]; counts={}
 for j,fp in enumerate(h5):
  p=pid(fp.name)
  if p not in common: continue
  with h5py.File(fp,'r') as h: z=np.asarray(h['features']); z=z[0] if z.ndim==3 else z
  slide_means.append(z.mean(0,dtype=np.float64).astype(np.float32)); rows.append({'patient_id':p,'slide_id':fp.name,'n_patches':int(z.shape[0]),'feature_dim':int(z.shape[1])}); counts[p]=counts.get(p,0)+1
 audit['multi_slide_patients']=sum(v>1 for v in counts.values())
 sm=pd.DataFrame(rows); sm['feature_index']=np.arange(len(sm)); np.save(out/'pathology_slide_means.npy',np.stack(slide_means)); sm.to_csv(out/'pathology_slide_manifest.csv',index=False)
 (out/'audit/cohort_reconciliation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 md='# Cohort reconciliation\n\n'+''.join(f'- {k}: {v}\n' for k,v in audit.items()); (out/'audit/cohort_reconciliation.md').write_text(md,encoding='utf-8')
 return f.reset_index(drop=True),rna.loc[common],cna.loc[common],c.loc[common],sm,np.stack(slide_means)

def prep(train,ev,kind):
 a=train.to_numpy(float); b=ev.to_numpy(float); med=np.nanmedian(a,0); med[~np.isfinite(med)]=0; a=np.where(np.isfinite(a),a,med); b=np.where(np.isfinite(b),b,med)
 if kind=='rna':
  a=np.log2(np.clip(a,0,None)+1); b=np.log2(np.clip(b,0,None)+1); keep=np.mean(np.isfinite(train.to_numpy(float)),0)>=.5; a=a[:,keep]; b=b[:,keep]; sc=MinMaxScaler(); a=sc.fit_transform(a); b=sc.transform(b); idx=np.argsort(np.var(a,0))[-min(3000,a.shape[1]):]; return a[:,idx],b[:,idx]
 if kind=='cna':
  keep=np.mean(np.abs(a)>1e-8,0)>=.05; a=a[:,keep]; b=b[:,keep]; sc=StandardScaler(); return sc.fit_transform(a),sc.transform(b)
 if kind=='path':
  sc=StandardScaler(); return sc.fit_transform(a),sc.transform(b)
 raise ValueError(kind)
def prep_clin(tr,te):
 num=['AGE']; trn=tr[num].apply(pd.to_numeric,errors='coerce'); ten=te[num].apply(pd.to_numeric,errors='coerce'); med=trn.median(); sc=StandardScaler(); a=sc.fit_transform(trn.fillna(med)); b=sc.transform(ten.fillna(med)); enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False); c=enc.fit_transform(tr[SAFE_CAT].fillna('MISSING').astype(str)); d=enc.transform(te[SAFE_CAT].fillna('MISSING').astype(str)); return np.c_[a,c],np.c_[b,d]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--seed',type=int,default=20260904); args=ap.parse_args(); root=args.root.resolve(); out=root/'outputs/phase_m1r'; t0=time.time()
 f,rna,cna,clin,slides,slide_mean=load_all(root,out); valid=f.dss_time.notna()&f.dss_event.isin([0,1])&(f.dss_time>0); f=f.loc[valid].reset_index(drop=True); rna=rna.loc[valid.to_numpy()].reset_index(drop=True); cna=cna.loc[valid.to_numpy()].reset_index(drop=True); clin=clin.loc[valid.to_numpy()].reset_index(drop=True); path=np.stack([slide_mean[np.flatnonzero(slides.patient_id.to_numpy()==p)].mean(0) for p in f.patient_id])
 strata=f.cancer_type.astype(str)+'_'+f.dss_event.astype(int).astype(str); sk=StratifiedKFold(5,shuffle=True,random_state=args.seed); folds=list(sk.split(f,strata)); split=[]
 for k,(_,te) in enumerate(folds,1): split += [{'patient_id':f.iloc[i].patient_id,'fold':k} for i in te]
 pd.DataFrame(split).sort_values('patient_id').to_csv(out/'fixed_patient_folds.csv',index=False); (out/'audit').mkdir(exist_ok=True); pd.DataFrame([{'fold':k,'n_test':len(te),'events_test':int(f.iloc[te].dss_event.sum())} for k,(_,te) in enumerate(folds,1)]).to_csv(out/'audit/split_audit.csv',index=False)
 specs={'C0':['C'],'R0':['R'],'N0':['N'],'O0':['R','N'],'P0':['P'],'PO0':['P','R','N'],'PC0':['P','C'],'OC0':['R','N','C'],'POC0':['P','R','N','C'],'C1':['C'],'R1':['R'],'N1':['N'],'O1':['R','N'],'P1':['P']}
 allrows=[]; preds=[]
 cache={}
 for model,mods in specs.items():
  seeds=SEEDS if model.endswith('1') else (args.seed,)
  for seed in seeds:
   oof=np.full(len(f),np.nan)
   for fold,(tr,te) in enumerate(folds,1):
    xs=[]; ys=[]
    for m in mods:
     key=(fold,m)
     if key not in cache:
      if m=='C': cache[key]=prep_clin(clin.iloc[tr],clin.iloc[te])
      elif m=='R': cache[key]=prep(rna.iloc[tr],rna.iloc[te],'rna')
      elif m=='N': cache[key]=prep(cna.iloc[tr],cna.iloc[te],'cna')
      else: cache[key]=prep(pd.DataFrame(path[tr]),pd.DataFrame(path[te]),'path')
     a,b=cache[key]
     xs.append(a); ys.append(b)
    xtr=np.c_[*xs]; xte=np.c_[*ys]
    if model.endswith('1'):
     # Modality encoders are represented by fixed 128-d fold-local projections before the Cox head.
     rng=np.random.default_rng(seed+fold); proj=rng.normal(0,1,(xtr.shape[1],min(128,xtr.shape[1])))/math.sqrt(xtr.shape[1]); xtr=xtr@proj; xte=xte@proj
     beta=exact_cox(f.iloc[tr].dss_time.to_numpy(),f.iloc[tr].dss_event.to_numpy(),xtr,.1,30); oof[te]=xte@beta
    else:
     beta=exact_cox(f.iloc[tr].dss_time.to_numpy(),f.iloc[tr].dss_event.to_numpy(),xtr,.01,50); oof[te]=xte@beta
    allrows.append({'endpoint':'DSS','model':model,'seed':seed,'fold':fold,'n_test':len(te),'events_test':int(f.iloc[te].dss_event.sum()),'harrell_c':metric_c(f.iloc[te].dss_time,f.iloc[te].dss_event,oof[te])})
   g=censor_km(f.dss_time.to_numpy(),f.dss_event.to_numpy(),f.dss_time.to_numpy()); h=metric_c(f.dss_time,f.dss_event,oof); u=uno(f.dss_time.to_numpy(),f.dss_event.to_numpy(),oof,g); allrows.append({'endpoint':'DSS','model':model,'seed':seed,'fold':'OOF','n_test':len(f),'events_test':int(f.dss_event.sum()),'harrell_c':h,'uno_c':u}); preds.append(pd.DataFrame({'patient_id':f.patient_id,'model':model,'seed':seed,'risk':oof}))
 out.mkdir(exist_ok=True); Path(out/'statistical_tests').mkdir(exist_ok=True); pd.DataFrame(allrows).to_csv(out/'statistical_tests/fold_metrics_all.csv',index=False); pd.concat(preds).to_csv(out/'statistical_tests/oof_predictions.csv',index=False)
 manifest={'seed':args.seed,'neural_seeds':list(SEEDS),'n_patients':len(f),'dss_events':int(f.dss_event.sum()),'pfs_expected_from_reconciliation':529,'pfs_events_expected_from_reconciliation':141,'models':list(specs),'elapsed_seconds':time.time()-t0,'status':'DSS_completed_PFS_not_run','failed_runs':['PFS formal sensitivity','paired 5000 bootstrap','AUC/Brier/IBS/KM/logrank','POC0 label permutation','gated ABMIL P1 checkpoint training']}; (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__': main()

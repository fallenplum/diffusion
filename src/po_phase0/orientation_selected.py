import os,numpy as np,pandas as pd
from pathlib import Path
from src.phase_m2.data import load_raw_inputs,preprocess_fold
from src.phase_m2.common import harrell_c,uno_c_index
from src.po_phase0.omics_signal_probe import fit_linear,split_inner,ALPHAS
repo=Path("/root/diffusion_m3_verified_9f859ef");raw=load_raw_inputs(Path(os.environ["DIFFUSION_DATA_ROOT"]),repo);pool=np.load("/root/autodl-tmp/po_phase0_runtime/path_tokens_train_pool.npy",mmap_mode="r");out=[]
for fold in range(1,6):
 a=preprocess_fold(raw,pool,fold);tr=np.flatnonzero(raw.outer_folds!=fold);te=np.flatnonzero(raw.outer_folds==fold);it,iv=split_inner(tr,raw.dss_event,20260904+fold);mods={"RNA-only":a.rna,"CNA-only":a.cna,"RNA+CNA":np.concatenate([a.rna,a.cna],1)}
 for mod,x in mods.items():
  choices=[]
  for alpha in ALPHAS:
   w=fit_linear(x[it],raw.dss_time[it],raw.dss_event[it],alpha,20260904+fold,steps=180);base=x[iv]@w
   for sign in (1,-1):choices.append((harrell_c(raw.dss_time[iv],raw.dss_event[iv],sign*base),alpha,sign))
  inner,alpha,sign=max(choices,key=lambda z:(z[0],-z[1]));w=fit_linear(x[tr],raw.dss_time[tr],raw.dss_event[tr],alpha,20260904+fold,steps=220);pred=sign*x[te]@w;u,_,_=uno_c_index(raw.dss_time[te],raw.dss_event[te],pred,raw.dss_time[tr],raw.dss_event[tr]);out.append({"method":mod,"fold":fold,"alpha":alpha,"sign":sign,"inner_harrell":inner,"harrell_c":harrell_c(raw.dss_time[te],raw.dss_event[te],pred),"uno_c":u})
d=pd.DataFrame(out);d.to_csv('/root/autodl-tmp/po_omics_signal_probe/orientation_selected.csv',index=False);print(d.to_string(index=False));print(d.groupby("method")[['harrell_c','uno_c']].mean().to_string());print(d.groupby("method").sign.apply(list).to_string())

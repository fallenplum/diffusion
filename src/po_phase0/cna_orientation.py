import os, json, numpy as np, pandas as pd
from pathlib import Path
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.phase_m2.common import harrell_c, uno_c_index
from src.po_phase0.omics_signal_probe import fit_linear, split_inner, ALPHAS
repo=Path("/root/diffusion_m3_verified_9f859ef"); raw=load_raw_inputs(Path(os.environ["DIFFUSION_DATA_ROOT"]),repo); pool=np.load("/root/autodl-tmp/po_phase0_runtime/path_tokens_train_pool.npy",mmap_mode="r"); out=[]
for fold in range(1,6):
 a=preprocess_fold(raw,pool,fold); tr=np.flatnonzero(raw.outer_folds!=fold); te=np.flatnonzero(raw.outer_folds==fold); it,iv=split_inner(tr,raw.dss_event,20260904+fold); scores=[]
 for alpha in ALPHAS:
  w=fit_linear(a.cna[it],raw.dss_time[it],raw.dss_event[it],alpha,20260904+fold,steps=180); scores.append((harrell_c(raw.dss_time[iv],raw.dss_event[iv],a.cna[iv]@w),alpha))
 alpha=sorted(scores,key=lambda z:(-z[0],z[1]))[0][1]; w=fit_linear(a.cna[tr],raw.dss_time[tr],raw.dss_event[tr],alpha,20260904+fold); pred=a.cna[te]@w
 for direction,pr in [("forward",pred),("inverted",-pred)]:
  u,_,_=uno_c_index(raw.dss_time[te],raw.dss_event[te],pr,raw.dss_time[tr],raw.dss_event[tr]);out.append({"fold":fold,"alpha":alpha,"direction":direction,"harrell_c":harrell_c(raw.dss_time[te],raw.dss_event[te],pr),"uno_c":u})
print(pd.DataFrame(out).to_string(index=False)); print(pd.DataFrame(out).groupby("direction")[['harrell_c','uno_c']].mean())

"""Post-matrix descriptive requirement supplement, excluded primary hypothesis family."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from src.phase10_trimodal.features import prepare,scale
from src.phase10_trimodal.train import inner_split,choose_ridge
from src.phase10_trimodal.math import fit_ridge
from src.phase10_trimodal.reference import metrics,uno_weights,components
from src.phase10_trimodal.data import write_json

def main():
 root=Path(__file__).resolve().parents[1];report=root/'reports/phase10_trimodal_mechanism_audit';private=Path('/root/phase10_trimodal_private/fixed_rep_supplement');private.mkdir(exist_ok=True);d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));cfg=json.loads((root/'configs/phase10_trimodal_mechanism_audit.json').read_text());groups=json.loads((root/'configs/phase10_molecular_groups.json').read_text());arms=['P_S32','P_TITAN_PCA32','P_UNI2_PCA32','R_PCA16','R_molecular8'];risk={k:np.zeros(512) for k in arms};weights=np.zeros(512);rows=[]
 write_json(report/'fixed_rep_supplement_protocol.json',{'scope':'post-matrix descriptive supplement; no model or primary comparison changes','arms':arms,'ridge_grid':cfg['ridge_grid'],'folds':cfg['folds'],'initialization':'deterministic convex fits once per fold; no fake three independent initializations','lambda_selection':'same train-only inner split and Harrell; no test selection','new_prognostic_claims':'exploratory only'})
 for fold in cfg['folds']:
  tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold);it,iv=inner_split(d,tr,cfg['split_seed']+fold);fi,_=prepare(d,it,groups);fo,_=prepare(d,tr,groups)
  for f,fit in [(fi,it),(fo,tr)]:
   u,_=scale(d['uni2'].astype(float),fit);pp=PCA(32,svd_solver='full').fit(u[fit]);f['U']=pp.transform(u).astype(np.float32)
  weights[te]=uno_weights(d['DSS_time'][tr],d['DSS_event'][tr],d['DSS_time'][te],d['DSS_event'][te]);keys=dict(zip(arms,['S','T','U','RPCA','M']))
  for arm,key in keys.items():
   lam,sel=choose_ridge(fi[key].astype(float),d['DSS_time'],d['DSS_event'],it,iv,cfg['ridge_grid']);w,info=fit_ridge(fo[key][tr].astype(float),d['DSS_time'][tr],d['DSS_event'][tr],lam);risk[arm][te]=fo[key][te].astype(float)@w;hn,hd=components(d['DSS_time'][te],d['DSS_event'][te],risk[arm][te]);un,ud=components(d['DSS_time'][te],d['DSS_event'][te],risk[arm][te],weights[te]);rows.append({'arm':arm,'fold':fold,'lambda':lam,'parameters':len(w),'Harrell':hn/hd,'Uno60':un/ud,'Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud,'gradient_max':info['gradient_max']});np.savez_compressed(private/f'{arm}_fold{fold}.npz',train_indices=tr,test_indices=te,risk=risk[arm][te],coefficients=w);write_json(private/f'{arm}_fold{fold}.json',{'inner_selection':sel,'optimizer':info})
 pd.DataFrame(rows).to_csv(report/'fixed_rep_fold_metrics.csv',index=False);pd.DataFrame([{'arm':a,**metrics(d['DSS_time'],d['DSS_event'],r,d['folds'],weights)} for a,r in risk.items()]).to_csv(report/'fixed_rep_oof_metrics.csv',index=False);print('DONE fixed-representation descriptive supplement 25 fits',flush=True)
if __name__=='__main__':main()

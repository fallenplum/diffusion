"""Additional independent provenance, clinical anchor and training diagnostics."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.special import logsumexp
from src.phase10_trimodal.reference import components,uno_weights,metrics
from src.phase10_trimodal.data import digest,write_json

def independent_clinical(d,fit,simple=False):
 parts=[]
 for name in (['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE'] if simple else ['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']):
  v=np.asarray(d['clinical_'+name]).astype(str)
  if name=='AGE':
   x=pd.to_numeric(pd.Series(v),errors='coerce').to_numpy();x[~np.isfinite(x)]=np.nanmedian(x[fit]);parts.append(x[:,None])
  else:parts.append(np.column_stack([v==level for level in sorted(set(v[fit]))]))
 x=np.column_stack(parts).astype(float);mu=x[fit].mean(0);sd=x[fit].std(0);sd[sd<1e-8]=1;return ((x-mu)/sd).astype(np.float32)

def objective_gradient(x,t,e,w,lam):
 risk=x@w;loss=-risk[e==1].sum();dr=-e.astype(float)
 for u in np.unique(t[e==1]):
  dead=(t==u)&(e==1);n=int(dead.sum());at=t>=u
  for j in range(n):
   a=at.astype(float)-j/n*dead;ix=np.flatnonzero(a>0);v=risk[ix]+np.log(a[ix]);den=logsumexp(v);loss+=den;dr[ix]+=np.exp(v-den)
 return loss/e.sum()+lam*(w@w)/2,x.T@dr/e.sum()+lam*w

def main():
 root=Path(__file__).resolve().parents[1];rep=root/'reports/phase10_trimodal_mechanism_audit';out=Path('/root/phase10_trimodal_private/formal');d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));cfg=json.loads((root/'configs/phase10_trimodal_mechanism_audit.json').read_text());checks=[];logs=[];subrows=[]
 weights=np.zeros(512)
 for fold in cfg['folds']:
  tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold);weights[te]=uno_weights(d['DSS_time'][tr],d['DSS_event'][tr],d['DSS_time'][te],d['DSS_event'][te])
  for seed in cfg['seeds']:
   for arm in ['A0','A0_min','A3']:
    x=independent_clinical(d,tr,arm=='A0_min')
    if arm=='A3':
     s=d['static'].astype(float);sd=s[tr].std(0);sd[sd<1e-8]=1;s=((s-s[tr].mean(0))/sd).astype(np.float32);x=np.c_[x,s]
    x=x.astype(float);w=np.load(out/f'private_{arm}_seed{seed}_fold{fold}_coefficients.npz')['coefficients'];cell=json.loads((out/'cells'/f'{arm}_seed{seed}_fold{fold}.json').read_text());original=np.load(out/'oof'/f'{arm}_seed{seed}_fold{fold}.npz')['risk'];err=float(np.max(np.abs(x[te]@w-original)));loss,grad=objective_gradient(x[tr],d['DSS_time'][tr],d['DSS_event'][tr],w,cell['lambda']);checks.append({'arm':arm,'seed':seed,'fold':fold,'independent_input_and_risk_max_error':err,'independent_objective_gradient_max':float(abs(grad).max())});assert err==0 and abs(grad).max()<2e-5
 for p in sorted((out/'cells').glob('*.json')):
  z=json.loads(p.read_text());inner=z.get('inner_training',{}).get('training_logs',[]);outer=z.get('outer_training',{}).get('training_logs',[])
  if not inner:continue
  a=inner[0];b=inner[-1];last=outer[-1];sl=z['selected_epoch'];row={'arm':z['arm'],'seed':z['seed'],'fold':z['fold'],'selected_epoch':sl,'inner_epochs_run':len(inner),'refit_epochs':len(outer),'minimum_boundary':sl==10,'maximum_boundary':sl==40,'parameters_declared':z['total_parameters'],'parameters_with_gradient':last['participating_parameters'],'inner_train_loss_start':a['train_cox_after_step'],'inner_train_loss_end':b['train_cox_after_step'],'inner_val_EMA_selected':inner[sl-1]['inner_val_loss_EMA'],'inner_val_EMA_end':b['inner_val_loss_EMA'],'inner_val_EMA_last5_slope':float(np.polyfit(np.arange(5),[q['inner_val_loss_EMA'] for q in inner[-5:]],1)[0]),'refit_train_loss_start':outer[0]['train_cox_after_step'],'refit_train_loss_end':last['train_cox_after_step'],'correction_sd_end':last['correction_sd'],'correction_max_abs_end':last['correction_max_abs'],'risk_sd_end':last['risk_sd'],'peak_cuda_MB':z['peak_cuda_MB'],'seconds':z['seconds'],'gate_P_mean_end':last['gate_mean'][0],'gate_R_mean_end':last['gate_mean'][1],'SEL_end':last['SEL'],'PAL_end':last['PAL']}
  for k,v in last['gradients'].items():row['gradient_'+k+'_end']=v
  logs.append(row)
 pd.DataFrame(logs).to_csv(rep/'training_diagnostics.csv',index=False);pd.DataFrame(checks).to_csv(rep/'anchor_elementwise_audit.csv',index=False)
 blocks={'cancer':d['cancer_type'].astype(str),'RNA_missing_pattern':np.where(np.isnan(d['raw_rna']).sum(1)>0,'3024_missing','complete'),'clinical_stage':d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str)}
 for arm in cfg['arms']:
  for seed in cfg['seeds']:
   risk=np.zeros(512)
   for fold in cfg['folds']:
    z=np.load(out/'oof'/f'{arm}_seed{seed}_fold{fold}.npz');risk[z['test_indices']]=z['risk']
   for kind,values in blocks.items():
    for group in np.unique(values):
     ix=values==group;n=int(ix.sum());ev=int(d['DSS_event'][ix].sum());m=metrics(d['DSS_time'][ix],d['DSS_event'][ix],risk[ix],d['folds'][ix],weights[ix]) if ev>=10 else {'Harrell':np.nan,'Uno60':np.nan}
     subrows.append({'arm':arm,'seed':seed,'stratum':kind,'group':group,'patients':n,'events':ev,'Harrell':m['Harrell'],'Uno60':m['Uno60'],'small_event_suppressed':ev<10})
 pd.DataFrame(subrows).to_csv(rep/'subgroup_diagnostics.csv',index=False)
 write_json(rep/'extra_independent_validation.json',{'status':'PASS','clinical_anchor_elements':len(checks),'independent_objective_gradient':True,'elementwise_zero_error_all':True,'historical_Phase8_private_OOF_available':False,'source_hashes_in_cells':sorted(set(json.loads(p.read_text())['source_hash'] for p in (out/'cells').glob('*.json'))),'RNA_NA_proxy_not_confirmed_platform':True})
 print('PASS independent clinical anchor and training diagnostics',len(checks),len(logs),flush=True)
if __name__=='__main__':main()

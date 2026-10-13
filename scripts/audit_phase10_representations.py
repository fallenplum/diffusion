"""Frozen tap probes and checkpoint replay; no new architecture/selection."""
import argparse,sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.phase10_trimodal.features import prepare
from src.phase10_trimodal.models import MechanismModel
from src.phase10_trimodal.math import fit_ridge
from src.phase10_trimodal.reference import components,uno_weights
from src.phase10_trimodal.train import torch_inputs,inputs

def main():
 a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--fold',type=int,required=True);a=a.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 root=Path(__file__).resolve().parents[1];out=Path('/root/phase10_trimodal_private/formal');d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));cfg=json.loads((root/'configs/phase10_trimodal_mechanism_audit.json').read_text());groups=json.loads((root/'configs/phase10_molecular_groups.json').read_text());tr=np.flatnonzero(d['folds']!=a.fold);te=np.flatnonzero(d['folds']==a.fold)
 f,_=prepare(d,tr,groups);p=np.load('/root/phase10_trimodal_private/cache/patch.npy',mmap_mode='r');mask=np.load('/root/phase10_trimodal_private/cache/mask.npy');tf,pp,pm=torch_inputs(f,p,mask,'cuda');weights=uno_weights(d['DSS_time'][tr],d['DSS_event'][tr],d['DSS_time'][te],d['DSS_event'][te]);rows=[];checks=[]
 initial_cache={}
 for arm in cfg['arms']:
  ck=out/'checkpoints'/f'{arm}_seed{a.seed}_fold{a.fold}.pt'
  if not ck.exists():continue
  z=torch.load(ck,map_location='cpu',weights_only=False);torch.manual_seed(a.seed);model=MechanismModel('B0' if arm.startswith('D') else arm,f['C'].shape[1],d=64).cuda().eval();testin=inputs(tf,pp,pm,te,arm,d,cfg['split_seed']+303)
  if arm in ['A5','A6','B0','C0','D1','D2','D3']:
   with torch.no_grad():before={k:v.cpu().numpy() for k,v in model(*testin,return_details=True)[1].items()}
  model.load_state_dict(z['state'])
  with torch.no_grad():risk=model(*testin).cpu().numpy()
  original=np.load(out/'oof'/f'{arm}_seed{a.seed}_fold{a.fold}.npz')['risk'];err=float(np.max(np.abs(risk-original)));checks.append({'arm':arm,'seed':a.seed,'fold':a.fold,'max_checkpoint_risk_error':err,'exact_identity':bool(np.array_equal(risk,original))});assert err<1e-6
  if arm in ['A5','A6','B0','C0','D1','D2','D3']:
   rep=dict(np.load(out/'representations'/f'{arm}_seed{a.seed}_fold{a.fold}.npz'))
   for mod in ['C','R','P']:
    for phase in ['before','after']:
     xt=rep[phase+'_'+mod].astype(float);xv=before[mod] if phase=='before' else rep['test_'+mod];mu=xt.mean(0);sd=xt.std(0);sd[sd<1e-8]=1;xt=(xt-mu)/sd;xv=(xv-mu)/sd
     w,info=fit_ridge(xt,d['DSS_time'][tr],d['DSS_event'][tr],cfg['linear_probe_lambda']);rr=xv@w;hn,hd=components(d['DSS_time'][te],d['DSS_event'][te],rr);un,ud=components(d['DSS_time'][te],d['DSS_event'][te],rr,weights)
     rows.append({'arm':arm,'seed':a.seed,'fold':a.fold,'modality':mod,'phase':phase,'probe':'frozen_representation_linear_Cox_lambda1','Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud,'Harrell':hn/hd,'Uno60':un/ud,'ridge_gradient_max':info['gradient_max']})
    rr=rep['aux_test_'+mod];hn,hd=components(d['DSS_time'][te],d['DSS_event'][te],rr);un,ud=components(d['DSS_time'][te],d['DSS_event'][te],rr,weights);rows.append({'arm':arm,'seed':a.seed,'fold':a.fold,'modality':mod,'phase':'after','probe':'joint_aux_head_trained' if arm in ['D1','D2','D3'] else 'untrained_aux_head_negative_control','Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud,'Harrell':hn/hd,'Uno60':un/ud})
  del model
 target=out/'extra_audit';target.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(target/f'probe_seed{a.seed}_fold{a.fold}.csv',index=False);pd.DataFrame(checks).to_csv(target/f'replay_seed{a.seed}_fold{a.fold}.csv',index=False);print('DONE probes/replay',a.seed,a.fold,len(checks),flush=True)
if __name__=='__main__':main()

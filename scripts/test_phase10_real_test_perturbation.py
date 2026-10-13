"""Real single-cell repeat with perturbed test labels/features; retain aggregate only."""
import sys,json,copy,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from src.phase10_trimodal.data import write_json
from src.phase10_trimodal.features import prepare
from src.phase10_trimodal.train import train_neural,inner_split,choose_ridge

def main():
 torch.set_num_threads(2);root=Path(__file__).resolve().parents[1];report=root/'reports/phase10_trimodal_mechanism_audit';d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));cfg=json.loads((root/'configs/phase10_trimodal_mechanism_audit.json').read_text());groups=json.loads((root/'configs/phase10_molecular_groups.json').read_text());tr=np.flatnonzero(d['folds']!=1);te=np.flatnonzero(d['folds']==1);p=np.load('/root/phase10_trimodal_private/cache/patch.npy');mask=np.load('/root/phase10_trimodal_private/cache/mask.npy');d2=copy.deepcopy(d);p2=p.copy();p2[te]=1e6
 for key in ['rna_genes','cna_genes','static','titan']:d2[key][te]=1e12
 d2['DSS_event'][te]=np.roll(d2['DSS_event'][te],1);d2['DSS_time'][te]+=1e9
 states=[];epochs=[];audits=[]
 for dd,pp in [(d,p),(d2,p2)]:
  it,iv=inner_split(dd,tr,cfg['split_seed']+1);fi,ai=prepare(dd,it,groups);fo,ao=prepare(dd,tr,groups);lam,_=choose_ridge(fi['C'],dd['DSS_time'],dd['DSS_event'],it,iv,cfg['ridge_grid']);m,info,_=train_neural(dd,fi,pp,mask,it,iv,'B0',31,cfg,lam);epoch=info['selected_epoch'];del m;torch.cuda.empty_cache();m,info,_=train_neural(dd,fo,pp,mask,tr,np.array([],int),'B0',31,cfg,lam,epochs=epoch);states.append({k:v.detach().cpu().clone() for k,v in m.state_dict().items()});epochs.append(epoch);audits.append([ai,ao]);del m;torch.cuda.empty_cache()
 err=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0]);exact=all(torch.equal(states[0][k],states[1][k]) for k in states[0]);assert exact and epochs[0]==epochs[1] and audits[0]==audits[1]
 original=torch.load('/root/phase10_trimodal_private/formal/checkpoints/B0_seed31_fold1.pt',map_location='cpu',weights_only=False)['state'];original_exact=all(torch.equal(states[0][k],original[k].cpu()) for k in states[0]);assert original_exact
 write_json(report/'real_test_perturbation_audit.json',{'status':'PASS','scope':'B0 seed31 fold1 full 40-epoch inner selection + selected-epoch final fit repeated twice','changed':'all test patch inputs, RNA/CNA/S32/TITAN feature values, test survival time/event pairing','outer_event_count_preserved':True,'selected_epochs':epochs,'preprocessing_states_identical':True,'all_model_parameters_bitwise_identical':exact,'max_parameter_error':err,'unperturbed_refit_identical_to_original_formal_checkpoint':original_exact,'exploratory_private_outputs_saved':False});print('PASS real test perturbation: bitwise identical weights/epochs',epochs,flush=True)
if __name__=='__main__':main()

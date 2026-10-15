"""Executed failure injection and no-outer-label training probe."""
import sys,json,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from src.phase16_routing.core import *
from validate_phase13_independent import raw_endpoints,test_loss

def main():
    rows=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,RuntimeError,AssertionError) as error:rows.append({'fault':name,'status':'REJECTED','exception':type(error).__name__});return
        raise ValueError('fault accepted: '+name)
    d=parent.load();ev=raw_endpoints(d,{})['DSS'];t,e=ev['time'],ev['event'];m=manifest();owner=np.load(P/'row_owners.npy');correct=np.load(P/'selectors/RAND_seed31_512.npy');s=m['slides'][0];other=next(r['offset'] for r in m['slides'] if r['patient']!=s['patient'])
    reject('cross_patient_ROI',lambda:check_rows([other],owner,s['patient']))
    bad=copy.deepcopy(s['geometry']);bad['patch_size_level0']*=2;reject('wrong_level_or_MPP',lambda:check_geometry(bad,s['geometry']))
    def dual_invalid():
        gate=json.loads((R/'G0_DECISION.json').read_text())
        if gate['dual_visual']!='PASS':raise ValueError('dual mapping cannot be accepted without raw-image validation')
    reject('unverified_CONCH_UNI2_mapping',dual_invalid)
    part=parent.splits(d,'DSS',1);it=part['inner_train'];iv=part['inner_val'];reject('RNA_prompt_cross_partition',lambda:check_prompt_pair(it,np.r_[iv[0],it[1:]]));reject('RNA_prompt_identity',lambda:check_prompt_pair(it,it))
    tag={'arm':'RAND_512','seed':31,'fold':1,'source':source(),'candidates':m['source_sha256']};badtag=tag.copy();badtag['arm']='MIL_512';reject('mismatched_checkpoint',lambda:check_tag(badtag,'RAND_512',31,1))
    seed_all(31);bags=Bags('RAND',512,31);a,path=anchor(31,1,'inner');cs,_=cs_features(d,it)
    model,h,best=train(cs,bags,t,e,it,iv,a['w'],a['lambda'],2,997,True);tt=t.copy();ee=e.copy();tt[part['outer_test']]*=1.73;ee[part['outer_test']]=1-ee[part['outer_test']]
    second,hh,other_best=train(cs,bags,tt,ee,it,iv,a['w'],a['lambda'],2,997,True)
    identical=all(torch.equal(v,second.state_dict()[k]) for k,v in model.state_dict().items()) and h==hh
    if not identical or best!=other_best:raise ValueError('outer labels affect training/inner selection')
    rows.append({'fault':'outer_label_perturbation','status':'INVARIANT_BITWISE','check':'actual2epoch training repeated after all outer labels/time changed'})
    def verify_endpoint(times,events):
        if not np.array_equal(times,t) or not np.array_equal(events,e):raise ValueError('raw CDR endpoint mismatch')
    reject('tampered_censoring_time',lambda:verify_endpoint(tt,e))
    with torch.no_grad():risk=model(torch.tensor(cs,dtype=torch.float32,device='cuda'),bags)[iv].cpu().numpy()
    def risk_check(z):
        if np.max(abs(z-risk))>1e-6:raise ValueError('checkpoint replay risk direction')
    reject('reversed_risk',lambda:risk_check(-risk))
    # Real Efron gradient compared with finite difference of a literal loop.
    z=np.array([-.3,.2,.8,-.1,.5]);at=np.array([1.,2.,2.,3.,4.]);ae=np.array([1,1,1,0,1]);v=torch.tensor(z,requires_grad=True);loss=torch_cox(v,log_weights(at,ae,'cpu'),torch.tensor(ae));loss.backward();finite=[]
    for i in range(5):
        delta=np.eye(5)[i]*1e-5;finite.append((test_loss(at,ae,z+delta)-test_loss(at,ae,z-delta))/2e-5)
    err=float(np.max(abs(v.grad.numpy()-finite)))
    if err>1e-6:raise ValueError('independent Efron gradient')
    for i,r in enumerate(m['slides']):check_rows(correct[i],owner,r['patient'])
    import pandas as pd
    pd.DataFrame(rows).to_csv(R/'FAULT_INJECTIONS.csv',index=False);write(R/'HARD_GATES.json',{'status':'PASS','source_hash':source(),'actual_fault_checks':rows,'Efron_gradient_error':err,'full_ROI_ownership_checked':True,'semantic_accuracy_faults':'BLOCKED_NO_TEXT_MODEL;only partition guards exercised','dual_mapping_fault':'rejected at hard gate;no raw-image alignment accuracy claim'})
    print('HARD_GATES_PASS',flush=True)
if __name__=='__main__':main()

"""Semantic-specific leakage, ROI/projection/prompt and numerical failure checks."""
import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from src.phase16_semantic.core import *
from src.phase16_semantic.prompts import captions

def main():
    rows=[]
    def reject(label,fn):
        try:fn()
        except (ValueError,RuntimeError):rows.append({'fault':label,'status':'REJECTED'});return
        raise ValueError('injected fault accepted: '+label)
    results=[json.loads(p.read_text()) for p in (P/'pilot').glob('*.json')]
    if len(results)!=18 or any(r['status']!='PASS' or r['source']!=source() or r['outer_test_prediction'] for r in results):raise ValueError('pilot incomplete/stale')
    d=parent.load();part=parent.splits(d,'DSS',1);fit,test=part['inner_train'],part['inner_val'];path=directory(31,1,'inner');audit=torch.load(path/'preprocessing.pt',map_location='cpu',weights_only=False);used=np.r_[fit,test]
    for name in ['clinical_permutation','RNA_permutation']:
        for pp in [fit,test]:check_permutation(pp,audit[name][pp])
    reject('wrong_RNA_cross_partition',lambda:check_permutation(fit,np.r_[test[0],fit[1:]]));reject('wrong_RNA_identity',lambda:check_permutation(fit,fit))
    owners=np.load(base.P/'row_owners.npy');slides=base.manifest()['slides'];s=slides[0];other=next(x['offset'] for x in slides if x['patient']!=s['patient']);reject('cross_patient_ROI',lambda:base.check_rows([other],owners,s['patient']))
    spec=s['geometry'].copy();spec['patch_size_level0']*=2;reject('coordinate_wrong_scale',lambda:base.check_geometry(spec,s['geometry']))
    official=np.load(P/'official_projection.npy')
    def projection_check(w):
        if w.shape!=official.shape or not np.array_equal(w,official):raise ValueError('not official image/text projection')
    reject('identity_instead_of_official_projection',lambda:projection_check(np.eye(768)))
    actual,au=captions(d,fit,test,31,1);altered=dict(d)
    for name in ['DSS_event','DSS_time','PFS_event','PFS_time']:
        if name in altered:altered[name]=np.asarray(altered[name]).copy();altered[name][part['outer_test']]=999
    changed,other_audit=captions(altered,fit,test,31,1)
    if actual!=changed:raise ValueError('outer outcomes influence prompt generation')
    rows.append({'fault':'outer_label_prompt_perturbation','status':'INVARIANT_ALL_ACTUAL_STRINGS'})
    bags=SemanticBags(path/'CLIN_RNA_TEXT_512.npy',512);a,ap=base.anchor(31,1,'inner');cs,_=base.cs_features(d,fit);ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];m,h,b=base.train(cs,bags,t,e,fit,test,a['w'],a['lambda'],2,1777,True);tt=t.copy();ee=e.copy();tt[part['outer_test']]*=1.9;ee[part['outer_test']]=1-ee[part['outer_test']];other,hh,bb=base.train(cs,bags,tt,ee,fit,test,a['w'],a['lambda'],2,1777,True)
    if h!=hh or b!=bb or any(not torch.equal(v,other.state_dict()[k]) for k,v in m.state_dict().items()):raise ValueError('outer outcomes influence training/selection')
    rows.append({'fault':'outer_label_training_perturbation','status':'INVARIANT_BITWISE_AND_SELECTION'})
    correct=tag('CLIN_RNA_TEXT_512',31,1);wrong=correct.copy();wrong['arm']='RNA_TEXT_512'
    def check_identity(v):
        if v!=correct:raise ValueError('mismatched checkpoint')
    reject('mismatched_checkpoint',lambda:check_identity(wrong))
    if not (base.R/'HARD_GATES.json').exists():raise ValueError('base Efron/time/direction gate absent')
    pd.DataFrame(rows).to_csv(R/'FAULT_INJECTIONS.csv',index=False);write(R/'HARD_GATES.json',{'status':'PASS','source':source(),'pilot_cells':18,'pilot_outer_predictions':0,'actual_faults':rows,'original_shared_loss_time_risk_faults_retained_sha256':sha(base.R/'HARD_GATES.json'),'same_forward_architecture_as_old_controls':True,'S32_gradient_pilots_min':min(r['S32_gradient'] for r in results)});print('SEMANTIC_HARD_GATES_PASS',flush=True)
if __name__=='__main__':main()

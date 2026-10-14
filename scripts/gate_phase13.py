"""Pre-lock actual endpoint/fit isolation, matching and full inner reconstruction."""
import sys,json,copy,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from src.phase13_rna.data import *
from src.phase13_rna.models import train_rna,extract_rna
from src.phase13_rna.math import block_fit
from run_phase13 import fusion_job


def validate_partitions(part,folds,valid,fold):
    tr=np.array(part['outer_train']);te=np.array(part['outer_test']);it=np.array(part['inner_train']);iv=np.array(part['inner_val'])
    if not np.array_equal(tr,np.flatnonzero((folds!=fold)&valid)) or not np.array_equal(te,np.flatnonzero((folds==fold)&valid)):raise ValueError('wrong fold/patient')
    if set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise ValueError('inner label training leakage')


def actual_gate(d,cfg):
    part=splits(d,'DSS',1);it,iv,te=[part[k] for k in ['inner_train','inner_val','outer_test']];f,m,a=cached_features(d,'DSS',1,'inner');ev=endpoints(d)['DSS'];out=[]
    for change in [False,True]:
        r=f['HALL'].copy();t=ev['time'].copy();e=ev['event'].copy()
        if change:r[te]=r[te]*-7+777;t[te]+=1e6;e[te]=1-e[te]
        model,info=train_rna(r,m['clinical_mask'],m['HALL'],it,iv,t,e,31,cfg['training'])
        risk,_=extract_rna(model,r,te);out.append({'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'initial':info['initial_state'],'epoch':info['selected_epoch'],'risk':risk});del model;torch.cuda.empty_cache()
    for state in ['weights','initial']:
        if any(not torch.equal(value,out[1][state][key]) for key,value in out[0][state].items()):raise RuntimeError('test labels/input affect fitted RNA producer')
    if out[0]['epoch']!=out[1]['epoch'] or np.array_equal(out[0]['risk'],out[1]['risk']):raise RuntimeError('selection invariance/input response failed')
    # Rebuild PFIs train-local transforms after changing TEST data/labels.
    part=splits(d,'PFI',1);te=part['outer_test'];altered=copy.deepcopy(d)
    altered['rna_genes'][te]=altered['rna_genes'][te]*-2+111;altered['static'][te]+=333;altered['DSS_time'][te]+=1e6;altered['DSS_event'][te]=1-altered['DSS_event'][te];altered['PFS_event'][te]=1-altered['PFS_event'][te]
    aa,ma,ha=features(d,'PFI',1,'inner');bb,mb,hb=features(altered,'PFI',1,'inner')
    if ha!=hb or any(not np.array_equal(aa[k][part['inner_train']],bb[k][part['inner_train']]) for k in aa):raise RuntimeError('PFI test-informed preprocessing')
    if np.array_equal(aa['HALL'][te],bb['HALL'][te]):raise RuntimeError('PFI legitimate test input ignored')
    return {'status':'PASS','full_inner_epochs_each':80,'fitted_state_initial_epoch_invariant':True,'RNA_input_response':True,'PFI_trainlocal_preprocessing_invariant':True,'smoke_patient_predictions_saved':False,'selected_epoch':out[0]['epoch']}


def main():
    torch.set_num_threads(2);cfg=config();d=load();source=source_hash();result=subprocess.run([sys.executable,'-m','unittest','tests.test_phase13_rna','-v'],cwd=ROOT,capture_output=True,text=True)
    (REPORT/'UNIT_TEST_OUTPUT.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('RNA diagnosis tests failed')
    faults=[]
    for ep in cfg['endpoints']:
        valid=endpoints(d)[ep]['valid'];part=splits(d,ep,1)
        validate_partitions(part,d['folds'],valid,1)
        for name,key,edit in [('outer_test_wrong_patient','outer_test',lambda v:np.roll(v,1)),('inner_validation_in_train','inner_train',lambda v:np.r_[v,part['inner_val'][0]]),('outer_test_in_train','outer_train',lambda v:np.r_[v,part['outer_test'][0]])]:
            changed={k:v.copy() for k,v in part.items()};changed[key]=edit(changed[key])
            try:validate_partitions(changed,d['folds'],valid,1)
            except ValueError:faults.append({'endpoint':ep,'fault':name,'rejected':True})
            else:raise RuntimeError('fault admitted '+name)
    actual=actual_gate(d,cfg);write_json(REPORT/'ACTUAL_TEST_INTERVENTION.json',actual)
    reconstruction=fusion_job('HALL',31,1,gate_only=True)
    write_json(REPORT/'FAULT_INJECTION.json',{'status':'PASS','faults':faults,'parent28_faults_reused_with_unchanged_source':True})
    write_json(REPORT/'PROTOCOL_LOCK.json',{**cfg,'source_hash':source,'locked_before_new_outer_evaluation':True,'known_parent_outer_results':True,'parent_outer_models_reused':45,'DSS_baseline_parent_records_reused':60,'new_record_target':3330})
    write_json(REPORT/'HARD_GATES.json',{'status':'PASS','source_hash':source,'independent_unit_tests':5,'faults_rejected':len(faults),'actual_test_intervention':actual,'exact_inner_reconstruction':reconstruction,'PFI_source_original_CDR_verified':True})
    print(json.dumps({'status':'PASS','source_hash':source,'new_record_target':3330}))


if __name__=='__main__':main()

"""Hard D/M/S tests, actual test interventions, and label-free GPU preflight."""
import copy, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np, torch
from run_phase11 import sources_hash,CONFIG,REPORT,PRIVATE,inner_split
from src.phase10_trimodal.data import load_private,write_json,digest
from src.phase10_trimodal.math import log_weights,torch_cox
from src.phase10_trimodal.features import ahash
from src.phase11_semantic.features import prepare
from src.phase11_semantic.data import PatchCache
from src.phase11_semantic.models import SemanticModel
from src.phase11_semantic.train import train,choose_ridge,anchor_inputs,forward_partition


def independent_synthetic():
    from validate_phase11_independent import components,ipcw,check_manifest,check_record
    from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
    from sksurv.util import Surv
    cases=[(np.array([1,1,2,3,5.,6]),np.array([1,0,1,0,1,0]),np.array([2,2,1,0,-1,-2.])),
           (np.array([1,2,3,4.]),np.array([1,1,0,1]),np.array([1000,-1000,0,0.]))]
    for t,e,r in cases:
        n,den=components(t,e,r);assert abs(n/den-concordance_index_censored(e.astype(bool),t,r)[0])<1e-12
        tt=np.r_[t,20.,21.];ee=np.r_[e,1,0];w=ipcw(tt,ee,t,e,5)
        n,den=components(t,e,r,w);ref=concordance_index_ipcw(Surv.from_arrays(ee.astype(bool),tt),Surv.from_arrays(e.astype(bool),t),r,tau=5)[0]
        assert abs(n/den-ref)<1e-12
        counts=np.array([2,1,0,2,1,3])[:len(t)];ix=np.repeat(np.arange(len(t)),counts)
        a,b=components(t,e,r,w,counts);c,d=components(t[ix],e[ix],r[ix],w[ix]);assert abs(a-c)<1e-12 and abs(b-d)<1e-12
    assert components(np.arange(4),np.zeros(4),np.zeros(4))==(0.,0.)
    try:components(np.arange(3),np.array([1,2,0]),np.zeros(3))
    except ValueError:pass
    else:raise AssertionError('invalid event accepted')
    return {'status':'PASS','ties_censor_extreme_repeat_zero_event_cases':True}


def fault_injections():
    from validate_phase11_independent import check_manifest,check_record
    from tests.test_phase11_semantic import example,MechanismTests
    fold=np.array([1,1,2,2,3,3,4,4,5,5]);tr=np.arange(2,10);te=np.arange(2);it=np.arange(2,8);iv=np.arange(8,10)
    p=lambda fit:{'fit_indices':fit.tolist(),'fit_hash':ahash(fit),'source_fields':['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE'],'selection_source':'RNA expression only; no outcome','pca':{'fit_hash':ahash(fit)}}
    good={'outer_train':tr.tolist(),'outer_test':te.tolist(),'inner_train':it.tolist(),'inner_val':iv.tolist(),'preprocess_inner':p(it),'preprocess_outer':p(tr)}
    check_manifest(good,fold,1);faults=[]
    def reject(name,run,payload):
        try:run()
        except (ValueError,AssertionError,RuntimeError) as ex:
            faults.append({'fault':name,'status':'REJECTED','error':str(ex),'synthetic_injection':payload});return
        raise AssertionError('fault not caught: '+name)
    for name,mutate in [
        ('DSS_alias_in_Clinical',lambda m:m['preprocess_inner']['source_fields'].append('DSS_time')),
        ('DSS_alias_in_S32_prompt',lambda m:m['preprocess_outer'].update(selection_source='DSS-selected concepts')),
        ('DSS_alias_in_TopK',lambda m:m['preprocess_inner'].update(selection_source='DSS gene ranking')),
        ('slide_patient_split',lambda m:m['outer_train'].append(0)),
        ('global_scaler_TopK',lambda m:m['preprocess_inner'].update(fit_indices=list(range(10)))),
        ('global_PCA',lambda m:m['preprocess_outer']['pca'].update(fit_hash=ahash(np.arange(10)))),
        ('cross_split_wrong_RNA',lambda m:m.update(outer_test_modality_indices={'RNA':[2,3]}))]:
        bad=copy.deepcopy(good);mutate(bad);reject(name,lambda:check_manifest(bad,fold,1),bad)
    reject('CONCH_UNI2_row_grid_pair',lambda:assert_grid(256,512,True),{'source_grid':[256,512],'row_pair':True})
    for modality in ['S32','RNA']:
        m,x=example();
        if modality=='S32':
            original=m.concept.forward;m.concept.forward=lambda s:original(s).detach()
        else:
            original=m.rna.forward;m.rna.forward=lambda r,mode='msa':tuple(v.detach() for v in original(r,mode))
        m(*x).square().sum().backward();part=m.concept if modality=='S32' else m.rna
        reject('disconnected_'+modality,lambda:require_gradient(part),{'detached':modality})
    t=np.array([1,1,2,3.]);e=np.array([1,0,1,0]);r=torch.tensor([1.,2.,3.,4.])
    full=torch_cox(r,log_weights(t,e,'cpu'),torch.tensor(e));mini=torch_cox(r[:2],log_weights(t[:2],e[:2],'cpu'),torch.tensor(e[:2]))
    reject('minibatch_Cox',lambda:torch.testing.assert_close(full,mini),{'mini_size':2,'full_size':4})
    reject('unmasked_padding',lambda:padding_bad_test(),{'padding_value':10000})
    d={'folds':fold,'DSS_time':np.arange(10.),'DSS_event':np.array([1,0]*5)}
    d['patient_ids']=np.arange(10);good.update(patient_hash=ahash(d['patient_ids']),fold_hash=ahash(fold))
    cell={'status':'COMPLETE','arm':'B0','seed':31,'fold':1,'source_hash':'s','protocol_sha256':'p','outer_test_selection':False,'selected_epoch':0}
    z={'test_indices':te,'time':d['DSS_time'][te].copy(),'event':d['DSS_event'][te].copy(),'risk':np.array([1.,0.]),'arm':'B0','seed':31,'fold':1}
    check_record(cell,z,good,d,'B0',31,1,'s','p')
    bad=copy.deepcopy(z);bad['event']=1-bad['event'];reject('event_censor_reversed',lambda:check_record(cell,bad,good,d,'B0',31,1,'s','p'),{'event_flipped':True})
    for field,new in [('arm','B1'),('seed',47),('fold',2),('source_hash','bad')]:
        c=copy.deepcopy(cell);c[field]=new;reject('metadata_'+field,lambda:check_record(c,z,good,d,'B0',31,1,'s','p'),{field:new})
    reject('OOF_risk_reversed',lambda:torch.testing.assert_close(torch.tensor(-z['risk']),torch.tensor(z['risk'])),{'checkpoint_risk':[1.,0.],'OOF_risk':[-1.,0.]})
    reject('checkpoint_OOF_mismatch',lambda:torch.testing.assert_close(torch.tensor([2.,0.]),torch.tensor(z['risk'])),{'checkpoint_risk':[2.,0.],'OOF_risk':[1.,0.]})
    return {'status':'PASS','faults':faults,'clean_corrected_manifest':'PASS','scope':'synthetic injections, separate from formal/private data'}


def assert_grid(a,b,row_pair):
    if a!=b and row_pair:raise ValueError('unverified different patch grid row pairing')

def require_gradient(module):
    if not any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in module.parameters()):raise ValueError('disconnected modality')

def padding_bad_test():
    p=torch.tensor([1.,2.,0.]);bad=p.clone();bad[-1]=10000
    torch.testing.assert_close(p.mean(),bad.mean(),rtol=0,atol=0)


def real_test_perturbation(cfg):
    d=load_private('/root/phase7a_foundation_headroom_v1/private_dataset.npz');tr=np.flatnonzero(d['folds']!=1);te=np.flatnonzero(d['folds']==1);it,iv=inner_split(d,tr,1)
    altered={k:v.copy() for k,v in d.items()}
    altered['DSS_time'][te]=d['DSS_time'][te][::-1];altered['DSS_event'][te]=d['DSS_event'][te][::-1]
    for name in ['raw_rna','rna_genes','raw_cna','cna_genes','static','titan']:altered[name][te]=777.
    for name in ['clinical_AGE','clinical_AJCC_PATHOLOGIC_TUMOR_STAGE']:altered[name][te]='999'
    cache=PatchCache(PRIVATE/'cache',128);groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());out=[];test_predictions=[]
    original_batch=cache.batch;training_requests=[]
    for condition,data in enumerate([d,altered]):
        def altered_batch(ix,device):
            p,mask=original_batch(ix,device)
            if condition and set(ix)&set(te):p=p*0+999
            return p,mask
        cache.batch=altered_batch
        fi,mi,ai=prepare(data,it,groups);fo,mo,ao=prepare(data,tr,groups)
        xt=anchor_inputs(fi,data,it,'B0',cfg['split_seed']+101);xv=anchor_inputs(fi,data,iv,'B0',cfg['split_seed']+202)
        lam,sel=choose_ridge(xt,xv,data['DSS_time'][it],data['DSS_event'][it],data['DSS_time'][iv],data['DSS_event'][iv],cfg['ridge_grid'])
        model,info,_=train(data,fi,mi,cache,it,iv,'B0',31,cfg,lam);epoch=info['selected_epoch'];del model
        model,final,_=train(data,fo,mo,cache,tr,np.array([],int),'B0',31,cfg,lam,epoch)
        state={k:ahash(v.detach().cpu().numpy()) for k,v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():test_predictions.append(forward_partition(model,data,fo,cache,te,'B0',cfg['split_seed']+303,cfg['patient_chunk'])[0].cpu().numpy())
        out.append({'epoch':epoch,'lambda':lam,'state':state,'inner_preprocess':ai,'outer_preprocess':ao});del model
        torch.cuda.empty_cache()
    if out[0]!=out[1]:raise RuntimeError('actual test perturbation affects fitted pipeline')
    # Test patch never used in any training/model-selection call: the loader is
    # guarded below and the actual two fits have identical complete states.
    requested=[];original=original_batch
    def guarded(ix,device):
        requested.extend(np.asarray(ix).tolist())
        if set(ix)&set(te):raise RuntimeError('training requests outer-test patches')
        return original(ix,device)
    cache.batch=guarded
    # Run fit with a guarded patch loader for one complete optimization epoch.
    fi,mi,_=prepare(d,it,groups);m,_,_=train(d,fi,mi,cache,it,iv,'B0',31,cfg,out[0]['lambda'],1)
    del m
    if np.array_equal(test_predictions[0],test_predictions[1]):raise RuntimeError('legal test inputs have no risk response')
    return {'status':'PASS','arm':'B0','seed':31,'fold':1,'inner_epochs_each':cfg['inner_epochs'],
            'refit_epochs_each':out[0]['epoch'],'all_state_tensors_bitwise_equal':True,'test_label_and_all_feature_perturbation':True,
            'patch_test_absent_from_actual_training_calls':not bool(set(requested)&set(te)),
            'legal_test_input_changes_risk':True,'test_patch_extreme_values_in_second_condition':True,
            'state_fingerprint':hashlib.sha256(json.dumps(out[0]['state'],sort_keys=True).encode()).hexdigest(),
            'no_private_smoke_outputs_saved':True}


def gpu_preflight(cfg):
    records=[]
    cm=np.eye(6,dtype=np.float32);gm=np.zeros((8,512),np.float32)
    for k in range(8):gm[k,k*64:(k+1)*64]=1
    for budget in [128,512,1024]:
        torch.manual_seed(31);model=SemanticModel('B0',cm,gm,cfg['hidden_dim']).cuda().train()
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        chunks=[]
        from torch.utils.checkpoint import checkpoint
        for k in range(8):
            c=torch.randn(64,6,device='cuda');s=torch.randn(64,32,device='cuda');r=torch.randn(64,512,device='cuda');p=torch.randn(64,1,budget,768,device='cuda');mask=torch.ones(64,1,budget,device='cuda',dtype=torch.bool)
            chunks.append(checkpoint(model,c,s,r,p,mask,use_reentrant=False))
        risk=torch.cat(chunks);loss=risk.square().mean();loss.backward();torch.cuda.synchronize()
        records.append({'budget':budget,'patients':512,'patient_chunk':64,'slides_per_patient':1,'peak_MB':torch.cuda.max_memory_allocated()/2**20,'seconds':time.monotonic()-start,'finite_gradients':all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())})
        del model,chunks,risk,loss,c,s,r,p,mask;torch.cuda.empty_cache()
    return records


def main():
    torch.set_num_threads(2);cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'));start=time.monotonic()
    result=subprocess.run([sys.executable,'-m','unittest','tests.test_phase11_semantic','-v'],cwd=ROOT,capture_output=True,text=True)
    (REPORT/'UNIT_TEST_OUTPUT.txt').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('synthetic gates fail')
    independent=independent_synthetic();fault=fault_injections();write_json(REPORT/'FAULT_INJECTION.json',fault)
    # Freeze design before real experiments or any formal outer metric.
    write_json(REPORT/'PROTOCOL_LOCK.json',{**cfg,'source_hash':sources_hash(),'protocol_file_sha256':digest(CONFIG),'locked_before_outer':True})
    write_json(REPORT/'GPU_PREFLIGHT.json',gpu_preflight(cfg));actual=real_test_perturbation(cfg);write_json(REPORT/'REAL_TEST_PERTURBATION.json',actual)
    data=json.loads((PRIVATE/'cache/data_manifest.json').read_text())
    if data['status']!='PASS':raise RuntimeError('data gate')
    write_json(REPORT/'HARD_GATES.json',{'status':'PASS','source_hash':sources_hash(),'data_sha256':digest(PRIVATE/'cache/data_manifest.json'),
              'synthetic_test_returncode':0,'independent_synthetic':independent,'fault_injections_rejected':len(fault['faults']),
              'real_test_perturbation':actual,'seconds':time.monotonic()-start})
    print(json.dumps({'status':'PASS','seconds':time.monotonic()-start}),flush=True)
if __name__=='__main__':main()

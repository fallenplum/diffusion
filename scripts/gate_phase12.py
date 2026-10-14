"""Control/data leakage/math faults and exact cache preflight before training."""
import copy,json,sys,time,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from run_phase12 import config,PRIVATE,PARENT,REPORT,DATA,features_for,sources_hash
from src.phase10_trimodal.data import load_private,write_json
from src.phase10_trimodal.features import ahash
from src.phase11_semantic.data import PatchCache
from src.phase11_semantic.train import choose_ridge
from src.phase12_c4.features import prepare
from src.phase12_c4.train import train,forward_partition,prepare_inputs,derange
from src.phase12_c4.models import C4Model,convert_parent_state
from src.phase12_c4.cache import CachedPatches
from gate_phase11 import independent_synthetic,fault_injections


def phase12_faults():
    from validate_phase12_independent import check_record
    from replay_phase12 import graph_contract,assert_risk_matches
    from tests.test_phase12_c4 import example
    faults=[]
    def reject(name,call,payload):
        try:call()
        except (ValueError,RuntimeError,AssertionError) as ex:
            faults.append({'fault':name,'status':'REJECTED','synthetic_injection':payload,'error':str(ex)});return
        raise RuntimeError('Phase12 fault not rejected '+name)
    fold=np.repeat(np.arange(1,6),2);tr=np.arange(2,10);te=np.arange(2);it=np.arange(2,8);iv=np.arange(8,10)
    d={'folds':fold,'patient_ids':np.arange(10),'DSS_time':np.arange(10.),'DSS_event':np.array([1,0]*5)}
    gene=np.arange(512,dtype=np.int64)
    def prep(fit):return {'fit_indices':fit.tolist(),'fit_hash':ahash(fit),'pca':{'fit_hash':ahash(fit)},'source_fields':['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE'],'selection_source':'RNA expression only; no outcome','phase12':{'selection_source':'frozen annotation + TRAIN observation>=.9 and variance; NO survival labels','enriched_indices':gene.tolist(),'enriched_hash':ahash(gene),'matched_indices':(gene+512).tolist(),'matched_hash':ahash(gene+512)}}
    spec={'kind':'real','genes':'top','group':'g0','budget':128};mapping={k:{'RNA':v.tolist()} for k,v in [('inner_train',it),('inner_val',iv),('outer_train',tr),('outer_test',te)]}
    m={'outer_train':tr.tolist(),'outer_test':te.tolist(),'inner_train':it.tolist(),'inner_val':iv.tolist(),'patient_hash':ahash(d['patient_ids']),'fold_hash':ahash(fold),'preprocess_inner':prep(it),'preprocess_outer':prep(tr),'effective_modality_mapping':mapping}
    cell={'status':'COMPLETE','arm':'R0_128','seed':31,'fold':1,'source_hash':'S','protocol_sha256':'P','spec':spec,'outer_test_selection':False,'actual_fit':False}
    z={'arm':'R0_128','seed':31,'fold':1,'test_indices':te,'time':d['DSS_time'][te],'event':d['DSS_event'][te],'risk':np.array([1.,0.])};check_record(cell,z,m,d,'R0_128',31,1,spec,'S','P')
    for name,change in [('real_RNA_wrong_patient',lambda a:a['effective_modality_mapping']['outer_test'].update(RNA=[1,0])),('cross_split_RNA',lambda a:a['effective_modality_mapping']['outer_test'].update(RNA=[2,3])),('DSS_selected_enriched_genes',lambda a:a['preprocess_inner']['phase12'].update(selection_source='survival selected'))]:
        bad=copy.deepcopy(m);change(bad);reject(name,lambda:check_record(cell,z,bad,d,'R0_128',31,1,spec,'S','P'),bad)
    for part in ['RNA','S32']:
        model,x,_,_=example();model.eval()
        if part=='RNA':
            original=model.rna.encoder.forward;model.rna.encoder.forward=lambda r,mode='msa':tuple(t.detach() for t in original(r,mode))
        else:
            original=model.concept.forward;model.concept.forward=lambda s:original(s).detach()
        reject('Phase12_detached_'+part,lambda:graph_contract(model,x,'real',False,True),{'detach':part})
    model,x,_,_=example('null');model.rna.constant_mode=False
    reject('Null_uses_real_RNA',lambda:graph_contract(model,x,'null',False,True),{'constant_mode':False})
    model,x,_,_=example();original=model.forward
    model.forward=lambda c,s,r,p,mask:original(c,s,r,p,torch.ones_like(mask))
    reject('Phase12_unmasked_softmax_padding',lambda:graph_contract(model,x,'real'),{'mask_ignored':True,'padding':'nonuniform extreme vector'})
    model,x,_,_=example();model.eval();expected=model(*x).detach().numpy()
    reject('Phase12_reversed_OOF',lambda:assert_risk_matches(-expected,expected),{'direction_reversed':True})
    reject('Phase12_corrupted_OOF',lambda:assert_risk_matches(expected+.01,expected),{'perturbation':.01})
    return faults


def gpu_gate(d,cfg):
    tr,te,it,iv,inner,outer=features_for(d,1);f,cm,g,a=outer;f=f['top'];records=[]
    for budget in [128,512,1024]:
        fast=CachedPatches(PARENT/'cache',budget,True);slow=PatchCache(PARENT/'cache',budget);max_error=0.;old_time=0.;new_time=0.
        for start in range(0,512,64):
            ix=np.arange(start,min(start+64,512));tic=time.monotonic();oldx,oldm=slow.batch(ix,'cuda');torch.cuda.synchronize();old_time+=time.monotonic()-tic
            tic=time.monotonic();newx,newm=fast.batch(ix,'cuda');torch.cuda.synchronize();new_time+=time.monotonic()-tic
            if not torch.equal(oldx,newx) or not torch.equal(oldm,newm):raise RuntimeError('GPU cache changes patch values/masks')
            del oldx,oldm,newx,newm
        outputs=[];peak=[]
        for cache in [slow,fast]:
            torch.manual_seed(31);model=C4Model(cm,g['g0'],cfg['hidden_dim']).cuda().train();torch.cuda.reset_peak_memory_stats();torch.manual_seed(801)
            tic=time.monotonic();risk=forward_partition(model,d,f,cache,np.arange(512),'C4',cfg['split_seed']+101,cfg['patient_chunk'],True)[0];loss=risk.square().mean();loss.backward();torch.cuda.synchronize()
            outputs.append((risk.detach().cpu(),{k:v.grad.detach().cpu().clone() for k,v in model.named_parameters() if v.grad is not None}));peak.append(torch.cuda.max_memory_reserved()/2**20)
            elapsed=time.monotonic()-tic;del model,risk,loss;torch.cuda.empty_cache()
        torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=2e-6)
        for name in outputs[0][1]:torch.testing.assert_close(outputs[0][1][name],outputs[1][1][name],rtol=1e-6,atol=1e-7)
        total=torch.cuda.get_device_properties(0).total_memory/2**20
        workers=max(1,min(12,int((total-4096)/(peak[-1]+600))))
        records.append({'budget':budget,'all512_input_values_masks_exact':True,'gradient_and_risk_equivalence':True,'GPU_cached_max_reserved_MB':peak[-1],'worker_limit':workers,'CPU_loader_full512_seconds':old_time,'GPU_loader_full512_seconds_synchronized':new_time,'last_full_gradient_seconds':elapsed,'peak_rule':'4GiB global margin +600MiB/context,cap12'})
        del fast,slow,outputs;torch.cuda.empty_cache()
    return records


def actual_test_gate(d,cfg):
    tr,te,it,iv,_,_=features_for(d,1);groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text())
    altered={k:v.copy() for k,v in d.items()};altered['DSS_time'][te]=d['DSS_time'][te][::-1]+999;altered['DSS_event'][te]=1-d['DSS_event'][te]
    for name in ['raw_rna','rna_genes','raw_cna','cna_genes','static','titan']:altered[name][te]=888
    for name in ['clinical_AGE','clinical_AJCC_PATHOLOGIC_TUMOR_STAGE']:altered[name][te]='999'
    result=[];base_cache=CachedPatches(PARENT/'cache',128,True);original=base_cache.batch
    for j,data in enumerate([d,altered]):
        vi,cmi,gmi,ai=prepare(data,it,groups,cfg['group_seeds'],cfg['matched_gene_set_seed']);vo,cmo,gmo,ao=prepare(data,tr,groups,cfg['group_seeds'],cfg['matched_gene_set_seed'])
        # Prove all three gene-set variants and ALL maps independent of test.
        states=[ai,ao];f=vi['top'];ff=vo['top'];x=np.c_[f['C'],f['S']].astype(float)
        lam,sel=choose_ridge(x[it],x[iv],data['DSS_time'][it],data['DSS_event'][it],data['DSS_time'][iv],data['DSS_event'][iv],cfg['ridge_grid'])
        def batch(ix,device):
            p,mask=original(ix,device)
            if j and set(ix)&set(te):p=p*0+888
            return p,mask
        base_cache.batch=batch
        model,inner,_=train(data,f,cmi,gmi['g0'],base_cache,it,iv,31,cfg,lam);epoch=inner['selected_epoch'];initial={k:ahash(v.detach().cpu().numpy()) for k,v in inner['initial_state'].items()};del model;torch.cuda.empty_cache()
        model,final,_=train(data,ff,cmo,gmo['g0'],base_cache,tr,np.array([],int),31,cfg,lam,epochs=epoch)
        finalstate={k:ahash(v.detach().cpu().numpy()) for k,v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():risk=forward_partition(model,data,ff,base_cache,te,'C4',cfg['split_seed']+303,cfg['patient_chunk'])[0].cpu().numpy()
        result.append({'states':states,'initial':initial,'final':finalstate,'epoch':epoch,'lambda':lam,'risk':risk});del model;torch.cuda.empty_cache()
    a,b=result
    for key in ['states','initial','final','epoch','lambda']:
        if a[key]!=b[key]:raise RuntimeError('outer test perturbation affects fitted pipeline '+key)
    if np.array_equal(a['risk'],b['risk']):raise RuntimeError('legitimate test input should affect risk')
    return {'status':'PASS','full_inner_epochs_each':80,'fresh_refit_epochs_each':a['epoch'],'fitted_states_and_init_bitwise_equal':True,'all3_RNA_gene_variants_8group_maps_train_only':True,'legal_test_input_risk_response':True,'private_smoke_predictions_not_saved':True,'state_hash':hashlib.sha256(json.dumps(a['final'],sort_keys=True).encode()).hexdigest()}


def main():
    torch.set_num_threads(2);cfg=config();parent=json.loads((REPORT/'PARENT_HARD_GATE.json').read_text())
    if parent['status']!='PASS' or parent['source_hash']!=sources_hash():raise RuntimeError('parent gate absent/stale')
    result=subprocess.run([sys.executable,'-m','unittest','tests.test_phase12_c4','-v'],cwd=ROOT,capture_output=True,text=True)
    (REPORT/'UNIT_TEST_OUTPUT.txt').write_text(result.stdout+result.stderr)
    if result.returncode or 'Ran 10 tests' not in result.stderr:raise RuntimeError('control tests failed')
    inherited_faults=fault_injections();inherited_faults['faults'].extend(phase12_faults());write_json(REPORT/'FAULT_INJECTION.json',inherited_faults)
    metric=independent_synthetic();d=load_private(DATA);print('START_EQUIVALENT_GPU_CACHE_PREFLIGHT',flush=True);profile=gpu_gate(d,cfg);write_json(REPORT/'GPU_PREFLIGHT.json',profile)
    print('START_ACTUAL_OUTER_TEST_INTERVENTION',flush=True);actual=actual_test_gate(d,cfg);write_json(REPORT/'REAL_TEST_PERTURBATION.json',actual)
    write_json(REPORT/'PROTOCOL_LOCK.json',{**cfg,'source_hash':sources_hash(),'locked_before_new_outer_inspection':True,'inherited_old_outer_results_known':True,'execution_workers':{str(v['budget']):v['worker_limit'] for v in profile}})
    write_json(REPORT/'HARD_GATES.json',{'status':'PASS','source_hash':sources_hash(),'historical_gate':parent,'control_unit_tests':10,'faults_rejected':len(inherited_faults['faults']),'independent_metric_synthetic':metric,'actual_test_intervention':actual})
    print(json.dumps({'status':'PASS','workers':{v['budget']:v['worker_limit'] for v in profile}}),flush=True)


if __name__=='__main__':main()

"""One real registered inner fit; the optimizer receives fit labels only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,gzip,json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import verify_lock,build,folder,read,write,sha,atomic,shared_preprocessing
from src.semantic_m123r.models import RepairModel
from src.semantic_m123r.training import fit,SNAPSHOTS
from src.semantic_m123r.io import partition_endpoints,stratified_cycle,save_checkpoint,load_checkpoint
from src.semantic_m123.training import subset
from src.semantic_m123.diagnostics import summarize_attention
from src.semantic_b0b7.p2_survival import evaluate_safe


def describe(model,batch,reference,eligible,wrong,mean):
    with torch.no_grad():
        out=model(batch,diagnostics=True)
        ref=torch.as_tensor(reference,dtype=torch.float64,device=out['risk'].device)
        denom=float(ref.std(unbiased=False));denom=max(denom,1e-12)
        sd=lambda x:float(x.std(unbiased=False))
        d={'anchor_sd':denom,'risk_sd':sd(out['risk']),'base_drift_sd':sd(out['base_risk']-ref),
            'full_drift_sd':sd(out['risk']-ref),'base_drift_ratio':sd(out['base_risk']-ref)/denom,
            'local_risk_sd':sd(out['local_risk']),'interaction_risk_sd':sd(out['interaction_risk']),
            'local_out_norm':float(model.local_out.weight.norm()),'wrong_pair_eligible':int(eligible.sum())}
        vectors={k:v.cpu().numpy() for k,v in out.items() if isinstance(v,torch.Tensor) and v.ndim==1}
        if out['mil_attention'] is not None:
            g=summarize_attention(out['mil_attention'].cpu().numpy(),out['mil_tokens'].cpu().numpy(),batch['mask'].cpu().numpy())
            d.update({'MIL_'+k:v for k,v in g.items()})
            off=model(batch,disable_mil=True)['risk'];vectors['no_mil_risk']=off.cpu().numpy()
            d['MIL_risk_dependence_ratio']=sd(out['risk']-off)/denom
        if out['cross_attention'] is not None:
            g=summarize_attention(out['cross_attention'].cpu().numpy(),out['cross_tokens'].cpu().numpy(),
                batch['mask'].cpu().numpy(),batch['program_mask'].any(-1).cpu().numpy())
            d.update({'CROSS_'+k:v for k,v in g.items()})
            for label,context in (('wrong_query',wrong),('mean_query',mean)):
                other=model(batch,query_context=context,diagnostics=True)
                for key in ('base_risk','local_risk','pathology_main_risk','omics_main_risk','clinical_main_risk'):
                    if not torch.equal(out[key],other[key]):raise ValueError('CONDITION_INTERVENTION_CHANGED_OWNER_VALUES_OR_MAIN_EFFECTS')
                delta=out['risk']-other['risk'];ix=torch.as_tensor(eligible,device=delta.device)
                d[label+'_risk_delta_sd']=sd(delta[ix]) if ix.any() else None
                d[label+'_risk_dependence_ratio']=sd(delta[ix])/denom if ix.any() else None
                diff=(out['cross_attention']-other['cross_attention']).abs().sum(-1)
                valid=batch['program_mask'].any(-1)
                d[label+'_attention_L1_mean']=float((diff*valid[:,None]).sum()/(valid.sum()*diff.shape[1]).clamp_min(1))
                vectors[label+'_risk']=other['risk'].cpu().numpy()
            vectors['no_interaction_risk']=model(batch,disable_interaction=True)['risk'].cpu().numpy()
        return d,vectors


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','job','expected-job-sha256','private'):p.add_argument('--'+k,required=True)
    a=p.parse_args();lock,parent=verify_lock(a.lock,a.expected_lock_sha256)
    if sha(a.job)!=a.expected_job_sha256:raise ValueError('JOB_CHANGED')
    job=read(a.job)
    if job['lock_sha256']!=a.expected_lock_sha256 or job['phase'] not in ('qualification','formal') or job['role'] not in ('inner','final'):raise ValueError('JOB_SCOPE_CHANGED')
    if job['phase']=='qualification' and job['outer']!=1:raise ValueError('QUALIFICATION_OUTER1_ONLY')
    from src.semantic_m123r.protocol import configurations
    final=job['role']=='final'
    if job['config'] not in configurations(job['arm']) or job['seed'] not in (31,47,83):raise ValueError('UNREGISTERED_GRID')
    if not final and job['epochs']!=140:raise ValueError('INNER_MUST_COMPLETE_FULL_BUDGET')
    if final:
        barrier=Path(a.private)/'ALL_OUTER_CHOICES_FROZEN.private.json'
        if job['phase']!='formal' or sha(barrier)!=job['selection_sha256']:raise ValueError('FINAL_REQUIRES_GLOBAL_SELECTION_BARRIER')
        global_selection=read(barrier)
        if global_selection['lock_sha256']!=a.expected_lock_sha256:raise ValueError('FINAL_MASTER_MISMATCH')
        allowed=[v for v in global_selection['selections'][f"outer{job['outer']}/{job['arm']}/{job['seed']}"]['choices'].values() if v['config']==job['config']]
        if not allowed or job['epochs']!=max(v['epoch'] for v in allowed):raise ValueError('FINAL_CONFIGURATION_NOT_SELECTED')
    out=folder(a.private,job);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    split=f"outer{job['outer']}_final" if final else f"outer{job['outer']}_inner{job['inner']}"
    b,data,scale,raw,assets,part,digest=build(parent,split)
    if split in lock['input_sha256'] and lock['input_sha256'][split]!=digest:raise ValueError('QUALIFIED_COVARIATES_CHANGED')
    plan=read(parent['parent_split_plan'])['partitions'];allowed=plan[f"outer{job['outer']}_final"]['fit']
    nt=len(data.train)
    tt,ee=partition_endpoints(assets['endpoints'],raw['patient_ids'],data.rows[:nt] if final else data.rows,allowed)
    t,e=tt[:nt],ee[:nt];vt,ve=(None,None) if final else (tt[nt:],ee[nt:])
    prep=shared_preprocessing(a.private,split,{'original':data.state,'repair_scale':scale})
    torch.set_num_threads(2);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dims=[b[k].shape[1] for k in ('P','G','C')]
    model=RepairModel(job['arm'],*dims).cuda();model.warm(data.warm)
    base0={n:v.detach().cpu().numpy().copy() for n,v in model.named_parameters() if n.startswith('base.')}
    base_ref=shared_preprocessing(a.private,split+'_base_seed'+str(job['seed']),base0)
    train=subset(b,data.train,'cuda');valid=subset(b,data.evaluation,'cuda')
    r0train=data.baseline[:nt];r0valid=data.baseline[nt:]
    contexts={}
    for label,batch,rows in (('train',train,data.rows[:nt]),('validation',valid,data.rows[nt:])):
        order,eligible=stratified_cycle(raw['rna_measured'][rows],raw['cna_measured'][rows])
        ix=torch.as_tensor(order,device='cuda')
        contexts[label]=({'program':batch['program'][ix],'program_mask':batch['program_mask'][ix],'coverage':batch['coverage'][ix]},eligible)
    mask=train['program_mask'];count=mask.sum(0).clamp_min(1)
    meanbase={'program':torch.where(mask,train['program'],0.).sum(0,keepdim=True)/count[None],
        'program_mask':mask.any(0,keepdim=True),'coverage':train['coverage'].mean(0,keepdim=True)}
    snapshots=[];pred={'rows':data.rows,'fit_rows':data.rows[:nt],'evaluation_rows':data.rows[nt:],
        'train_time':t,'train_event':e,'r0_train':r0train,'r0_validation':r0valid}
    if not final:pred.update(evaluation_time=vt,evaluation_event=ve)
    cps={};maximum=0.;replay=RepairModel(job['arm'],*dims).cuda()
    def snapshot(epoch,counts,max_cox_gradient):
        nonlocal maximum
        node={'epoch':epoch,'actual_update_counts':counts,'maximum_observed_training_Cox_only_gradients':max_cox_gradient,'diagnostics':{},'metrics':{}}
        for label,batch,rt,et,ref in (('train',train,t,e,r0train),('validation',valid,vt,ve,r0valid)):
            mean={k:v.expand(len(batch['P']),-1,-1) for k,v in meanbase.items()}
            d,vectors=describe(model,batch,ref,contexts[label][1],contexts[label][0],mean)
            node['diagnostics'][label]=d
            node['metrics'][label]={k:evaluate_safe(t,e,rt,et,value) for k,value in vectors.items() if k in ('risk','base_risk','no_mil_risk','wrong_query_risk','mean_query_risk','no_interaction_risk')} if rt is not None else {}
            for k,value in vectors.items():pred[f'{epoch}_{label}_{k}']=value
        if epoch==0:
            err=float(np.max(np.abs(pred['0_validation_risk']-r0valid)))
            if err>1e-10:raise ValueError('S0_EXACT_R4_OUTPUT_FAILED')
            node['S0_FP64_error']=err
        cp=save_checkpoint(out/f'checkpoint_{epoch}.private.pt.gz',model,epoch,job,dims,base_ref,digest)
        load_checkpoint(out/cp['file'],a.private,replay)
        with torch.no_grad():r=replay(valid)['risk'].cpu().numpy()
        error=float(np.max(np.abs(r-pred[f'{epoch}_validation_risk'])))
        if error>1e-9:raise ValueError('SAVED_SNAPSHOT_REPLAY_FAILED')
        maximum=max(maximum,error);node['checkpoint']=cp;node['actual_saved_replay_error']=error
        cps[str(epoch)]=cp;snapshots.append(node)
    result=fit(model,train,t,e,r0train,job['config'],snapshot_callback=snapshot,
        heartbeat=lambda epoch,stage:atomic(out/'LIVE.private.json',{'epoch':epoch,'stage':stage,'seconds':time.monotonic()-started}),max_epochs=job['epochs'])
    with (out/'PREDICTIONS.private.npz').open('xb') as f:np.savez_compressed(f,**pred)
    with gzip.open(out/'HISTORY.private.json.gz','xt') as f:json.dump(result['history'],f,separators=(',',':'))
    write(out/'COMPLETE.private.json',{'status':'COMPLETE','job':job,'lock_sha256':a.expected_lock_sha256,
        'snapshots':snapshots,'checkpoints':cps,'parameters':sum(p.numel() for p in model.parameters()),
        'input_sha256':digest,'preprocessing':prep,'initial_base':base_ref,
        'predictions_sha256':sha(out/'PREDICTIONS.private.npz'),'history_sha256':sha(out/'HISTORY.private.json.gz'),
        'maximum_saved_replay_error':maximum,'optimizer_updates':result['optimizer_updates'],
        'actual_parameter_update_counts':result['actual_parameter_update_counts'],
        'elapsed_seconds':time.monotonic()-started,'outer_endpoint_rows_decoded':False,
        'label_access':'ONLY_current_outer_train_rows_individual_NPY_elements','no_patient_arrays_public':True})
    print(json.dumps({'status':'COMPLETE','arm':job['arm'],'seed':job['seed'],'inner':job.get('inner'),'updates':job['epochs']}))


if __name__=='__main__':main()

"""One registered M1--M3 partition fit; private predictions, no outer scoring."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,gzip,hashlib,json,pickle,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.semantic_m123.models import JointModel,ARMS
from src.semantic_m123.data import prepare
from src.semantic_m123.training import fit,subset
from src.semantic_b0b7.p2_data import read_dss_only
from src.semantic_b0b7.p2_anchor import load_legacy_backend
from src.semantic_b0b7.p2_survival import evaluate_safe
from scripts.semantic_b0b7_p2_matrix import raw_inputs,shared_preprocessing
from types import SimpleNamespace


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_bytes())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def atomic(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f)
    os.replace(tmp,p)


def check_lock(path,expected):
    if sha(path)!=expected:raise ValueError('M123_MASTER_LOCK_CHANGED')
    lock=read(path)
    if lock['status']!='LOCKED_BEFORE_NEW_DSS_OPTIMIZATION':raise ValueError('M123_LOCK_STATUS')
    for name,h in lock['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('M123_SOURCE_CHANGED:'+name)
    if sha(ROOT/'configs/semantic_m123/design_precommit.json')!=lock['design_precommit_sha256']:
        raise ValueError('M123_DESIGN_CHANGED')
    if sha(lock['parent_split_plan'])!=lock['parent_split_sha256']:raise ValueError('M123_SPLITS_CHANGED')
    return lock


def build(lock,split):
    a=SimpleNamespace(assets=lock['assets_path'],legacy_root=lock['legacy_root'])
    raw,member,assets=raw_inputs(a,endpoints=False)
    ap=Path(lock['parent_private'])/'anchors'/(split+'.private.pkl')
    binding=lock['partitions'][split]
    if sha(ap)!=binding['anchor_sha256'] or sha(ap.with_suffix('.json'))!=binding['anchor_receipt_sha256']:
        raise ValueError('M123_CURRENT_PARTITION_TEACHER_CHANGED')
    with ap.open('rb') as f:anchor=pickle.load(f)
    plan=read(lock['parent_split_plan'])['partitions'][split]
    frozen=torch.load(lock['frozen_images'],map_location='cpu',weights_only=False,mmap=True)
    data=prepare(raw,frozen,anchor,plan['fit'],plan['evaluation'],member)
    if data.receipt['input_sha256']!=binding['input_sha256']:raise ValueError('M123_COVARIATES_OR_TRANSFORMS_CHANGED')
    endpoints,_=read_dss_only(assets['endpoints'],raw['patient_ids'])
    # The optimizer is passed only these fit-train labels, never full512 arrays.
    train_t=endpoints['t'][plan['fit']];train_e=endpoints['e'][plan['fit']]
    eval_t=endpoints['t'][plan['evaluation']] if plan['role']=='inner' else None
    eval_e=endpoints['e'][plan['evaluation']] if plan['role']=='inner' else None
    return data,plan,train_t,train_e,eval_t,eval_e


def verify_job(job,lock,private):
    if job['role'] not in ('inner','final'):raise ValueError('INVALID_MODEL_SELECTION_ROLE')
    if job['split'] not in lock['partitions']:raise ValueError('UNREGISTERED_PARTITION')
    if job['arm']!='M0PLUS' and job['arm'] not in ARMS:raise ValueError('UNREGISTERED_ARM')
    if job['role']=='inner':
        if not job['split'].endswith(tuple('inner'+str(i) for i in range(3))):raise ValueError('NOT_INNER')
        if job['arm']=='M0PLUS':
            if job['seed']!=0 or job['lambda'] not in (.1,1.,10.):raise ValueError('M0PLUS_GRID')
        elif job['seed'] not in (31,47,83) or job['lambda'] not in (.001,.01) or job['epochs']!=400:
            raise ValueError('NN_INNER_GRID')
    else:
        p=Path(private)/'ALL_OUTER_SELECTIONS_FROZEN.private.json'
        selections=read(p)
        if selections['lock_sha256']!=job['lock_sha256'] or sha(p)!=job['selection_sha256']:
            raise ValueError('FINAL_SELECTION_BARRIER_CHANGED')
        key=f"{job['split'].split('_')[0]}/{job['arm']}/{job['seed']}"
        selected=selections['choices'][key]
        if selected['lambda']!=job['lambda'] or selected['epoch']!=job['epochs']:raise ValueError('FINAL_CONFIG_NOT_LOCKED')


def route_audit(model,batch):
    if not model.arm.startswith('M3'):return {'applicable':False}
    model.zero_grad(set_to_none=True)
    n=len(batch['P']);order=torch.arange(n,device=batch['P'].device).roll(1)
    risk={};attdiff=0.;units=0;main_error=0.
    with torch.no_grad():
        for start in range(0,n,16):
            sl=slice(start,min(n,start+16));b={k:v[sl] for k,v in batch.items()}
            wrong={k:batch[k][order[sl]] for k in ('program','program_mask','coverage')}
            a=model(b,diagnostics=True);w=model(b,query_context=wrong,diagnostics=True)
            for key in ('pathology_risk','omics_risk','clinical_risk'):
                main_error=max(main_error,float((a[key]-w[key]).abs().max()))
            aa=a['patch_attention'];ww=w['patch_attention'];valid=b['program_mask'].any(-1)
            attdiff+=float(((aa-ww).abs().sum(-1)*valid[:,None]).sum())
            units+=int(valid.sum())*aa.shape[1]
            for key,out in [('matched',a),('wrong_query',w)]:
                risk.setdefault(key,[]).append(out['interaction_risk'].cpu().numpy())
    rr={k:np.concatenate(v) for k,v in risk.items()}
    if main_error!=0:raise ValueError('WRONG_QUERY_CHANGED_OWNER_MAIN_EFFECT')
    probe={k:v[:min(16,n)] for k,v in batch.items()}
    interaction=model(probe)['interaction_risk']
    names=[(k,p) for k,p in model.named_parameters() if k in ('patch.0.weight','program.0.weight','query_projection.weight','attention.q.weight','attention.k.weight')]
    gg=torch.autograd.grad(interaction.sum(),[p for k,p in names],allow_unused=True)
    path={k:float(g.norm()) if g is not None else 0. for (k,p),g in zip(names,gg)}
    return {'applicable':True,'partition':'fit_train_only_frozen_intervention_not_retraining',
        'isolated_interaction_risk_gradient_norms_without_L2':path,
        'main_effect_max_abs_change':main_error,'attention_query_L1_mean':attdiff/units if units else None,
        'matched_interaction_risk_sd':float(np.std(rr['matched'])),
        'wrong_query_interaction_risk_sd':float(np.std(rr['wrong_query'])),
        'matched_minus_wrong_risk_sd':float(np.std(rr['matched']-rr['wrong_query']))}


def run(args):
    lock=check_lock(args.lock,args.expected_lock_sha256)
    if sha(args.job)!=args.expected_job_sha256:raise ValueError('JOB_CHANGED')
    job=read(args.job);verify_job(job,lock,args.private)
    if job['lock_sha256']!=args.expected_lock_sha256:raise ValueError('JOB_LOCK_BINDING')
    out=Path(args.output_newdir)
    if not out.resolve().is_relative_to((Path(args.private)/'fits').resolve()):raise ValueError('OUTPUT_OUTSIDE_NEW_STUDY_FITS')
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();data,part,tt,te,vt,ve=build(lock,job['split'])
    prep=shared_preprocessing(args.private,job['split'],data.state)
    pred={};snapshots=[]
    def record(epoch,tr,ev):
        for side,d in [('train',tr),('evaluation',ev)]:
            for k,v in d.items():pred[f'{epoch}_{side}_{k}']=v
        metrics=evaluate_safe(tt,te,vt,ve,ev['risk']) if job['role']=='inner' else None
        snapshots.append({'epoch':epoch,'evaluation_metrics':metrics,'outer_evaluation_performed':False})
    if job['arm']=='M0PLUS':
        backend,_=load_legacy_backend(lock['legacy_root'],expected_source_hashes=lock['legacy_dependencies_sha256'])
        solver=backend.__globals__['fit_offset_ridge']
        coef,optimization=solver(data.flat[data.train],tt,te,job['lambda'])
        risk=data.flat@coef
        record(0,{'risk':risk[data.train]},{'risk':risk[data.evaluation]})
        with gzip.open(out/'MODEL.private.pkl.gz','xb') as f:pickle.dump({'coef':coef,'preprocessing':prep},f,protocol=5)
        statefile='MODEL.private.pkl.gz';history=[];updates=0;route={'applicable':False};optimization_info=optimization
        # Actual saved model coefficient reload, not a check of the in-memory vector.
        with gzip.open(out/statefile,'rb') as f:loaded=pickle.load(f)
        replay=data.flat@loaded['coef'];error=float(np.max(np.abs(replay-risk)))
        parameters=len(coef)
    else:
        torch.set_num_threads(2);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        dims=[data.batch[k].shape[1] for k in ('P','G','C')]
        model=JointModel(job['arm'],*dims).cuda()
        if job['arm']!='M1_SCRATCH':
            warm={k:torch.as_tensor(v,dtype=torch.float32,device='cuda') if k.endswith('coef') else v for k,v in data.warm.items()}
            model.warm(**warm)
        train=subset(data.batch,data.train,'cuda');evaluation=subset(data.batch,data.evaluation,'cuda')
        def heartbeat(epoch,row):atomic(out/'LIVE.private.json',{'epoch':epoch,'stage':row['stage'],'elapsed_seconds':time.monotonic()-start})
        result=fit(model,train,tt,te,seed=job['seed'],l2=job['lambda'],max_epochs=job['epochs'],
            evaluation_batch=evaluation,snapshot_epochs=(0,20,50,100,200,400) if job['role']=='inner' else (0,job['epochs']),
            snapshot_callback=record,heartbeat_callback=heartbeat)
        if job['epochs'] not in [r['epoch'] for r in snapshots]:
            model.eval()
            with torch.no_grad():record(job['epochs'],{k:v.cpu().numpy() for k,v in model(train).items()},
                {k:v.cpu().numpy() for k,v in model(evaluation).items()})
        route=route_audit(model,train);history=result['history'];updates=result['optimizer_updates']
        checkpoint={'state_dict':model.cpu().state_dict(),'dims':dims,'arm':job['arm'],'job':job,
            'input_sha256':data.receipt['input_sha256'],'sources':lock['sources']}
        statefile='MODEL.private.pt';torch.save(checkpoint,out/statefile)
        loaded=torch.load(out/statefile,map_location='cpu',weights_only=False)
        replay_model=JointModel(loaded['arm'],*loaded['dims']).cuda()
        replay_model.load_state_dict(loaded['state_dict']);replay_model.eval()
        with torch.no_grad():replay=replay_model(evaluation)['risk'].cpu().numpy()
        error=float(np.max(np.abs(replay-pred[f"{job['epochs']}_evaluation_risk"])))
        parameters=model.parameter_audit()['parameters'];optimization_info={'solver':'Adam','last_history_row':history[-1] if history else None}
        if error>2e-6:raise ValueError('SAVED_CHECKPOINT_RISK_REPLAY_FAILED')
    with (out/'PREDICTIONS.private.npz').open('xb') as f:
        np.savez_compressed(f,**pred,rows=data.rows,train=data.train,evaluation=data.evaluation,
            training_time=tt,training_event=te,**({'evaluation_time':vt,'evaluation_event':ve} if job['role']=='inner' else {}))
    # gzip history keeps every update while respecting the finite disk budget.
    with gzip.open(out/'HISTORY.private.json.gz','xt',encoding='utf-8') as f:json.dump(history,f,separators=(',',':'))
    receipt={'schema':'m123_private_fit_v1','status':'COMPLETE','job':job,'input':data.receipt,
        'snapshots':snapshots,'parameters':parameters,'optimizer_updates':updates,
        'elapsed_seconds':time.monotonic()-start,'route_audit':route,'shared_preprocessing':prep,
        'optimization':optimization_info,
        'checkpoint_file':statefile,'checkpoint_sha256':sha(out/statefile),
        'prediction_sha256':sha(out/'PREDICTIONS.private.npz'),'history_sha256':sha(out/'HISTORY.private.json.gz'),
        'saved_checkpoint_prediction_max_error':error,'outer_metrics_read':False,
        'history_is_pre_update_snapshot_predictions_post_update':True}
    write(out/'COMPLETE.private.json',receipt)
    print(json.dumps({'status':'COMPLETE','split':job['split'],'arm':job['arm'],'updates':updates,'checkpoint_replay_max_error':error}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','job','expected-job-sha256','private','output-newdir'):p.add_argument('--'+k,required=True)
    run(p.parse_args())


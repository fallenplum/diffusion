"""Locked neural experiment: independent test scored only after selection/refit."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.rscil.data import generate,fit_transform,fit_warm,array_hash
from src.rscil.geometry import EfronRiskSets
from src.rscil.model import RSCIL
from src.rscil.train import fit,predict,state_hash
from src.rscil.report import harrell,write_json


def subset(raw,indices):return {k:v[indices] if isinstance(v,np.ndarray) else v for k,v in raw.items()}


def run(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    lock=json.loads((ROOT/'configs/rscil/synthetic_lock.json').read_text(encoding='utf-8'))
    source_lock=json.loads((ROOT/'configs/rscil/run_lock.json').read_text(encoding='utf-8'))
    for name,sha in source_lock['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('SOURCE_LOCK_CHANGED:'+name)
    if args.seed not in lock['seeds'] or args.n not in lock['sample_sizes']:raise ValueError('UNLOCKED_REPLICATION')
    if args.scenario not in lock['scenarios']:raise ValueError('UNLOCKED_SCENARIO')
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    final=target/f'{args.n}_{args.scenario}_{args.seed}.json'
    if final.exists():raise ValueError('COMPLETED_CELL_MUST_NOT_BE_OVERWRITTEN')
    started=time.monotonic();raw,test=generate(args.seed,args.n,args.scenario)
    permutation=np.random.default_rng(args.seed+81001).permutation(len(raw['P']))
    val=permutation[:max(1,int(.2*len(permutation)))];tr=permutation[len(val):]
    inner,valid=fit_transform(subset(raw,tr),subset(raw,val),device=args.device)[:2]
    full,hold=fit_transform(raw,test,device=args.device)[:2]
    warm_inner=fit_warm(inner,raw['time'][tr],raw['event'][tr]);warm_full=fit_warm(full,raw['time'],raw['event'])
    valrs=EfronRiskSets(raw['time'][val],raw['event'][val],device=args.device)
    testrs=EfronRiskSets(test['time'],test['event'],device=args.device)
    rows=[];audit=[]
    for arm in lock['arms']:
        stages=[]
        for phase,batch,warm,t,e in [('selection',inner,warm_inner,raw['time'][tr],raw['event'][tr]),
                                     ('refit',full,warm_full,raw['time'],raw['event'])]:
            torch.manual_seed(args.seed+90001)
            model=RSCIL(arm,12,12,6,rank=8).to(args.device);model.warm(warm)
            reference=sum(batch[k]@warm[l+'_coef'] for k,l in [('P','p'),('G','g'),('C','c')])
            # Projection is not required for the zero-head exact initial risk.
            init=model(batch,a=model.A)['risk'];error=float((init-reference).abs().max())
            if error>=1e-10:raise ValueError('WARM_FUNCTION_MISMATCH')
            training=fit(model,batch,t,e,epochs=90)
            replay_receipts=[]
            replay_eval=valid if phase=='selection' else hold
            for ep,ss in training['snapshots'].items():
                model.load_state_dict(ss['state']);model.stage(ep)
                expected_train=predict(model,batch)['risk'];expected_eval=predict(model,replay_eval)['risk']
                temporary=target/f'.snapshot_{args.n}_{args.scenario}_{args.seed}_{arm}.pt'
                torch.save(model.state_dict(),temporary)
                restored=RSCIL(arm,12,12,6,rank=8).to(args.device)
                restored.load_state_dict(torch.load(temporary,weights_only=True,map_location=args.device));restored.stage(ep)
                train_error=float((expected_train-predict(restored,batch)['risk']).abs().max())
                eval_error=float((expected_eval-predict(restored,replay_eval)['risk']).abs().max())
                if max(train_error,eval_error)>1e-12:raise ValueError('ALL_SNAPSHOT_DISK_REPLAY_FAILED')
                replay_receipts.append({'epoch':ep,'checkpoint_tensor_sha256':state_hash(restored),
                                        'checkpoint_file_sha256':hashlib.sha256(temporary.read_bytes()).hexdigest(),
                                        'train_max_abs':train_error,'evaluation_max_abs':eval_error})
                temporary.unlink()
            if phase=='selection':
                curve=[]
                for epoch,snap in training['snapshots'].items():
                    model.load_state_dict(snap['state']);pred=predict(model,valid)['risk'].to(args.device)
                    curve.append({'epoch':epoch,'validation_Cox':float(valrs.loss(pred)),
                                  **harrell(raw['time'][val],raw['event'][val],pred.cpu().numpy())})
                # Raw early best is retained but cannot masquerade as a fully
                # updated generator. Equal 30/60/90 selection in every arm.
                selected=min((x for x in curve if x['epoch']>=30),key=lambda x:(x['validation_Cox'],x['epoch']))['epoch']
                early=min(curve,key=lambda x:(x['validation_Cox'],x['epoch']))['epoch']
            else:
                model.load_state_dict(training['snapshots'][selected]['state'])
                model.stage(selected)
                with torch.no_grad():pred=predict(model,hold)['risk']
                checkpoint=target/f'.replay_{args.n}_{args.scenario}_{args.seed}_{arm}.pt'
                torch.save(model.state_dict(),checkpoint)
                tensor_sha=state_hash(model);file_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                replay=RSCIL(arm,12,12,6,rank=8).to(args.device);replay.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=args.device));replay.stage(selected)
                r2=predict(replay,hold)['risk'];replay_error=float((pred-r2).abs().max())
                if replay_error>1e-12:raise ValueError('DISK_REPLAY_FAILED')
                checkpoint.unlink()  # Synthetic weights are not a durable deliverable.
                row={'role':'synthetic_neural','arm':arm,'seed':args.seed,'n':args.n,'scenario':args.scenario,
                     'selected_epoch':selected,'raw_early_best_epoch':early,
                     **harrell(test['time'],test['event'],pred.numpy()),'Cox_NLL':float(testrs.loss(pred.to(args.device))),
                     'oracle_Harrell_diagnostic_only':harrell(test['time'],test['event'],test['oracle'])['Harrell'],
                     'warm_error':error,'added_parameters':model.added_parameters(),
                     'total_parameters':sum(p.numel() for p in model.parameters()),'selected_model_tensor_sha256':tensor_sha,
                     'checkpoint_file_sha256':file_sha,'disk_replay_max_abs':replay_error,
                     'selected_update_counts':training['snapshots'][selected]['counts'],
                     'refit_geometry':training['snapshots'][selected]['geometry']}
                rows.append(row)
            stages.append({'phase':phase,'warm_error':error,'updates':training['updates'],
                           'optimizer_updates':training['optimizer_updates'],'geometry_evaluations':training['geometry_evaluations'],
                           'elapsed_seconds':training['elapsed_seconds'],'history':training['history'],'all_snapshot_disk_replays':replay_receipts,
                           'snapshots':[{k:v for k,v in ss.items() if k!='state'}|{'epoch':ep} for ep,ss in training['snapshots'].items()]})
        audit.append({'arm':arm,'validation_curve':curve,'stages':stages})
        print(json.dumps({'n':args.n,'scenario':args.scenario,'seed':args.seed,'arm':arm,'done':True}),flush=True)
    write_json(final,{'lock_sha256':hashlib.sha256((ROOT/'configs/rscil/run_lock.json').read_bytes()).hexdigest(),
                     'rows':rows,'audit':audit,'elapsed_seconds':time.monotonic()-started,
                     'train_events':int(raw['event'].sum()),'test_events':int(test['event'].sum()),
                     'selection_fit_hash':array_hash(tr),'selection_valid_hash':array_hash(val),
                     'no_TCGA_loaded':True,'test_labels_seen_only_in_scoring_after_epoch_lock':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--scenario',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    run(p.parse_args())

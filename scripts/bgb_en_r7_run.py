"""One prelocked R7 cell, 80 complete-risk Efron updates, no automatic retry."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r7_common import (parser,context,qualified_prepared,construct,validate_qualification,verify_target_lock,
    public_snapshot_guard,write,torch_write,read,require,sha256,PARTITIONS,SEEDS,TARGET_ARMS,EVALUATION_EPOCHS,
    DIAGNOSTIC_EPOCHS,frozen_controls_for_arm,configure_deterministic_runtime,validate_deterministic_runtime)
from src.bgb_en.r7_training import saved_state,state_hash,full_risk_step,predict,metrics_for,branch_diagnostics,checkpoint_diagnostics
from src.bgb_en.survival import evaluate


def cell_name(arm,part,seed):return f'{arm}_p{part}_seed{seed}'


def make_references(args,ctx,prepared,origin):
    raw,valid=ctx['raw'],prepared['valid']
    for arm,key in (('B0','CP'),('B1','base')):
        metrics=metrics_for(raw,prepared,valid,{'risk':np.asarray(prepared[key][valid],float)})
        for seed in SEEDS:
            path=args.private/'references'/f'{arm}_p{args.partition}_seed{seed}.json'
            value=dict(status='R7_BASELINE_REFERENCE',group='REFERENCE',arm=arm,partition=args.partition,seed=seed,
                prepared_origin=origin,contract_sha256=ctx['lock']['contract_sha256'],metrics=metrics['CP_SCALAR'],metrics_vs=metrics,
                actual_new_optimizer_fits=0,teacher_refits=0,deterministic_seed_reference=seed!=31)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():require(read(path)==value,'R7_EXISTING_REFERENCE_CHANGED')
            else:
                with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,allow_nan=False)


def frozen_predictions(model,prepared,raw,prediction):
    valid,train=prepared['valid'],prepared['train'];controls={};predictions={}
    for tag in frozen_controls_for_arm(model.arm):
        value=predict(model,valid,control=tag);predictions[tag]=value
        controls[tag]=dict(metrics_vs_base=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],value['risk'],prepared['base'][valid]),
            relative_to_trained=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],prediction['risk'],value['risk']),
            risk_SD_ratio=float(np.std(prediction['risk']-value['risk'])/max(np.std(prepared['base'][valid]),1e-12)),
            frozen_not_retrained=True,whole_modality_replaced=False,
            intervention_scope='path_query_only' if tag=='wrong_path' else 'RNA_query_only' if tag in ('wrong_query','mean_query') else tag)
    return controls,predictions


def run_unit(args,ctx,prepared,origin):
    require(args.arm in TARGET_ARMS and args.partition in PARTITIONS and args.seed in SEEDS,'R7_UNIT_OUTSIDE_LOCKED_MATRIX')
    qualification=validate_qualification(args,ctx)
    contract=dict(contract_sha256=ctx['lock']['contract_sha256'],prepared_origin=origin,arm=args.arm,partition=args.partition,
        seed=args.seed,qualification_sha256=qualification,config_sha256=sha256(args.config),source_config_sha256=sha256(args.source_config),
        upstream_r6=ctx['upstream_r6'],fixed_deterministic_runtime=validate_deterministic_runtime(),epochs=80,
        teacher_refits=0,source_training_updates=0,outer_test_used=False)
    directory=args.private/'cells'/cell_name(args.arm,args.partition,args.seed);receipt=directory/'COMPLETE.json';artifact=directory/'model.pt'
    if receipt.exists():
        previous=read(receipt)
        require(args.resume and previous['status']=='COMPLETE_R7_TARGET_FIT' and previous['contract']==contract
            and sha256(artifact)==previous['artifact_sha256'],'R7_COMPLETED_CELL_CHANGED')
        return previous
    require(not directory.exists(),'R7_INTERRUPTED_OR_FAILED_CELL_NO_AUTORETRY')
    directory.mkdir(parents=True);start=time.monotonic();updates=0;historical=public_snapshot_guard()
    write(directory/'START.json',dict(status='STARTED_R7_TARGET_CELL',contract=contract,optimizer_updates_completed=0))
    try:
        model,model_origin=construct(args,ctx,prepared,args.arm,args.seed);initial=saved_state(model)
        initialization=dict(full_state_sha256=state_hash(initial),target_state_sha256=state_hash(initial,target_only=True))
        train,valid,raw=prepared['train'],prepared['valid'],ctx['raw']
        batches=[torch.as_tensor(train[j:j+8],device=args.device) for j in range(0,len(train),8)]
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        frozen={id(p) for p in model.core.foundation.parameters()}
        require(not any(id(p) in frozen for group in optimizer.param_groups for p in group['params']),'R7_FROZEN_SOURCE_IN_OPTIMIZER')
        if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        curves=[];snapshots={};diagnostics={}
        for epoch in range(1,81):
            step=full_risk_step(model,batches,raw['t'][train],raw['e'][train],optimizer);updates+=1
            write(directory/'PROGRESS.json',dict(status='RUNNING',epoch=epoch,optimizer_updates_completed=updates,
                contract_sha256=ctx['lock']['contract_sha256'],last_step=step,elapsed_seconds=time.monotonic()-start,
                peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else 0))
            if epoch in DIAGNOSTIC_EPOCHS:
                model.refresh_training_query_mean();tp=predict(model,train)
                vp=predict(model,valid) if epoch in EVALUATION_EPOCHS else None
                diagnostics[str(epoch)]=checkpoint_diagnostics(model,prepared,raw,epoch,include_validation=vp is not None,
                    train_prediction=tp,valid_prediction=vp)
                write(directory/f'DIAGNOSTIC_epoch{epoch}.json',diagnostics[str(epoch)])
                if epoch in EVALUATION_EPOCHS:
                    tm=metrics_for(raw,prepared,train,tp,include_pfi=epoch==80)
                    vm=metrics_for(raw,prepared,valid,vp,include_pfi=epoch==80)
                    controls,control_predictions=frozen_predictions(model,prepared,raw,vp)
                    curves.append(dict(epoch=epoch,optimizer_updates_completed=updates,**step,train=tm['CP_SCALAR'],valid=vm['CP_SCALAR'],
                        train_SD_delta_over_offset=float(tp['delta'].std()/max(np.std(prepared['base'][train]),1e-12)),
                        valid_SD_delta_over_offset=float(vp['delta'].std()/max(np.std(prepared['base'][valid]),1e-12)),
                        checkpoint_selected_by_prelock=epoch==80,outer_test_read=False,
                        step_loss_pre_update_metrics_post_update=True))
                    state=saved_state(model)
                    snapshots[epoch]=dict(state_dict=state,predictions=dict(train=tp,validation=vp),metrics_vs=vm,train_metrics=tm,
                        frozen_controls=controls,frozen_control_predictions=control_predictions,
                        mean_query=model.mean_context.tokens.detach().cpu(),mean_query_mask=model.mean_context.program_mask.detach().cpu(),
                        optimizer_updates_completed=updates)
                    torch_write(directory/f'checkpoint_epoch{epoch}.pt',dict(contract=contract,epoch=epoch,state_dict=state,
                        optimizer_state=optimizer.state_dict(),model_origin=model_origin,optimizer_updates_completed=updates,
                        torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),
                        cuda_rng_state=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None,
                        snapshot=snapshots[epoch],curves=list(curves)))
        final=snapshots[80]
        torch_write(artifact,dict(contract=contract,train=train,valid=valid,model_origin=model_origin,final_state_dict=final['state_dict'],
            initialization=initialization,snapshots=snapshots,curves=curves,query_map=model.query_map,
            predictions=final['predictions'],frozen_control_predictions=final['frozen_control_predictions'],
            mean_query=final['mean_query'],mean_query_mask=final['mean_query_mask'],epochs=80,optimizer_updates=80))
        result=dict(status='COMPLETE_R7_TARGET_FIT',contract=contract,arm=args.arm,partition=args.partition,seed=args.seed,
            epochs=80,optimizer_updates=80,actual_new_optimizer_fits=1,teacher_refits=0,source_training_updates=0,
            source_origin=model_origin,initialization=initialization,parameters=model.parameter_budget(),metrics=final['metrics_vs']['CP_SCALAR'],
            metrics_vs=final['metrics_vs'],train_metrics=final['train_metrics'],curves=curves,
            risk_diagnostics=dict(train=branch_diagnostics(final['predictions']['train'],prepared['base'][train]),
                validation=branch_diagnostics(final['predictions']['validation'],prepared['base'][valid])),
            frozen_controls=final['frozen_controls'],gradient_probe=diagnostics['80']['gradient_probe'],checkpoint_diagnostics=diagnostics,
            checkpoint_artifact_sha256={str(e):sha256(directory/f'checkpoint_epoch{e}.pt') for e in EVALUATION_EPOCHS},
            snapshot_replay_contract=dict(required_epochs=list(EVALUATION_EPOCHS),every_state_requires_actual_forward=True,
                max_risk_error=2e-6,max_changed_pair_credits=0,primary_epoch=80),artifact_sha256=sha256(artifact),
            elapsed_seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else 0,
            outer_test_used=False,primary_epoch_locked_before_training=80)
        require(public_snapshot_guard()==historical,'R7_HISTORICAL_REPORTS_CHANGED');verify_target_lock(args,ctx)
        write(receipt,result);write(directory/'PROGRESS.json',dict(status='COMPLETE',epoch=80,optimizer_updates_completed=80,
            artifact_sha256=result['artifact_sha256']))
        return result
    except BaseException as exc:
        write(directory/'ERROR_RECEIPT.json',dict(status='FAILED_NOT_RETRIED',contract=contract,error_type=type(exc).__name__,
            optimizer_updates_completed=updates,elapsed_seconds=time.monotonic()-start,artifact_exists=artifact.exists()))
        (directory/'ERROR_PRIVATE.log').write_text(traceback.format_exc(),encoding='utf-8');raise


def main():
    configure_deterministic_runtime();torch.set_num_threads(2)
    p=parser(__doc__);p.add_argument('--mode',choices=('references','fit'),default='fit')
    p.add_argument('--partition',type=int,choices=PARTITIONS,required=True);p.add_argument('--arm',choices=TARGET_ARMS)
    p.add_argument('--seed',type=int,choices=SEEDS,default=31);args=p.parse_args();ctx=context(args)
    prepared,origin=qualified_prepared(args,ctx,args.partition)
    if args.mode=='references':make_references(args,ctx,prepared,origin);result=dict(status='COMPLETE_R7_REFERENCES',optimizer_updates=0)
    else:result=run_unit(args,ctx,prepared,origin)
    print(json.dumps({k:result[k] for k in ('status','optimizer_updates')}),flush=True)


if __name__=='__main__':main()

"""One locked R6 target cell; no source/teacher fit, scheduler or automatic retry."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r6_common import (parser,context,qualified_prepared,construct,validate_qualification,
    public_snapshot_guard,verify_target_lock,write,torch_write,require,sha256,read,PARTITIONS,SEEDS,TARGET_ARMS,
    configure_deterministic_runtime,validate_deterministic_runtime)
from src.bgb_en.r6_training import (saved_state,state_hash,full_risk_step,predict,metrics_for,
    risk_diagnostics,pure_survival_gradient_probe,frozen_attention_probe)
from src.bgb_en.survival import evaluate


def cell_name(arm,partition,seed):return f'{arm}_p{partition}_seed{seed}'


def make_references(args,ctx,prepared,origin):
    """Explicit separate action once per partition; zero optimizer fits."""
    raw,valid=ctx['raw'],prepared['valid']
    for arm,key in (('B0','CP'),('B1','base')):
        metrics=metrics_for(raw,prepared,valid,{'risk':np.asarray(prepared[key][valid],float)})
        for seed in SEEDS:
            path=args.private/'references'/f'{arm}_p{args.partition}_seed{seed}.json'
            value=dict(status='R6_BASELINE_REFERENCE',group='REFERENCE',arm=arm,partition=args.partition,
                seed=seed,prepared_origin=origin,contract_sha256=ctx['lock']['contract_sha256'],
                metrics=metrics['CP_SCALAR'],metrics_vs=metrics,actual_new_optimizer_fits=0,
                deterministic_seed_reference=seed!=31,teacher_refits=0)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():require(read(path)==value,'R6_EXISTING_BASELINE_REFERENCE_CHANGED')
            else:
                with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,allow_nan=False)


def run_unit(args,ctx,prepared,origin):
    require(args.arm in TARGET_ARMS and args.partition in PARTITIONS and args.seed in SEEDS,'R6_TARGET_CELL_OUTSIDE_LOCKED_MATRIX')
    qualification=validate_qualification(args,ctx)
    contract=dict(contract_sha256=ctx['lock']['contract_sha256'],prepared_origin=origin,arm=args.arm,
        partition=args.partition,seed=args.seed,qualification_sha256=qualification,config_sha256=sha256(args.config),
        source_config_sha256=sha256(args.source_config),fixed_deterministic_runtime=validate_deterministic_runtime(),
        epochs=80,teacher_refits=0,source_training_updates=0,outer_test_used=False)
    directory=args.private/'cells'/cell_name(args.arm,args.partition,args.seed)
    receipt,artifact=directory/'COMPLETE.json',directory/'model.pt'
    if receipt.exists():
        previous=read(receipt)
        require(args.resume and previous['status']=='COMPLETE_R6_TARGET_FIT' and previous['contract']==contract
                and previous['artifact_sha256']==sha256(artifact),'R6_COMPLETED_CELL_SOURCE_OR_PAYLOAD_CHANGED')
        return previous
    require(not directory.exists(),'R6_INTERRUPTED_OR_FAILED_CELL_REQUIRES_REVIEW_NO_AUTORETRY')
    directory.mkdir(parents=True);started=time.monotonic();updates=0
    write(directory/'START.json',dict(status='STARTED_R6_TARGET_CELL',contract=contract,optimizer_updates_completed=0))
    historical=public_snapshot_guard()
    try:
        model,model_origin=construct(args,ctx,prepared,args.arm,args.seed)
        initial=saved_state(model)
        initialization=dict(full_state_sha256=state_hash(initial),target_state_sha256=state_hash(initial,target_only=True))
        train,valid,raw=prepared['train'],prepared['valid'],ctx['raw']
        batches=[torch.as_tensor(train[j:j+8],device=args.device) for j in range(0,len(train),8)]
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        frozen={id(p) for p in model.core.foundation.parameters()}
        require(not any(id(p) in frozen for group in optimizer.param_groups for p in group['params']),'R6_FROZEN_SOURCE_ENTERED_OPTIMIZER')
        if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        curves=[];snapshots={}
        for epoch in range(1,81):
            step=full_risk_step(model,batches,raw['t'][train],raw['e'][train],optimizer);updates+=1
            write(directory/'PROGRESS.json',dict(status='RUNNING',epoch=epoch,optimizer_updates_completed=updates,
                contract_sha256=ctx['lock']['contract_sha256'],last_step=step))
            if epoch in (20,40,80):
                tp,vp=predict(model,train),predict(model,valid)
                tm,vm=metrics_for(raw,prepared,train,tp,include_pfi=False),metrics_for(raw,prepared,valid,vp,include_pfi=False)
                curves.append(dict(epoch=epoch,optimizer_updates_completed=updates,**step,train=tm['CP_SCALAR'],valid=vm['CP_SCALAR'],
                    train_SD_delta_over_offset=float(tp['delta'].std()/max(np.std(prepared['base'][train]),1e-12)),
                    valid_SD_delta_over_offset=float(vp['delta'].std()/max(np.std(prepared['base'][valid]),1e-12)),
                    checkpoint_selected_by_prelock=epoch==80,outer_test_read=False))
                state=saved_state(model)
                snapshots[epoch]=dict(state_dict=state,risk=vp['risk'],delta=vp['delta'],cross_delta=vp['cross_delta'])
                torch_write(directory/f'checkpoint_epoch{epoch}.pt',dict(contract=contract,epoch=epoch,state_dict=state,
                    optimizer_state=optimizer.state_dict(),model_origin=model_origin,torch_rng_state=torch.get_rng_state(),
                    numpy_rng_state=np.random.get_state(),cuda_rng_state=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None,
                    optimizer_updates_completed=updates,curves=curves))
        model.refresh_training_query_mean()
        train_prediction,valid_prediction=predict(model,train),predict(model,valid)
        vm,tm=metrics_for(raw,prepared,valid,valid_prediction),metrics_for(raw,prepared,train,train_prediction)
        controls={};control_predictions={}
        for control in ('matched_query','wrong_query','mean_query','interaction_off','value_off','attention_permutation','whole_omics'):
            prediction=predict(model,valid,control=control);control_predictions[control]=prediction
            controls[control]=dict(metrics_vs_base=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],prediction['risk'],prepared['base'][valid]),
                relative_to_trained=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],valid_prediction['risk'],prediction['risk']),
                risk_SD_ratio=float(np.std(valid_prediction['risk']-prediction['risk'])/max(np.std(prepared['base'][valid]),1e-12)),
                frozen_not_retrained=True,whole_modality_replaced=control=='whole_omics')
        attention=frozen_attention_probe(model,valid);gradient=pure_survival_gradient_probe(model,prepared,raw)
        require(public_snapshot_guard()==historical,'R6_HISTORICAL_FORMAL_REPORTS_CHANGED');verify_target_lock(ctx['lock'],ctx,args)
        torch_write(artifact,dict(contract=contract,train=train,valid=valid,model_origin=model_origin,final_state_dict=saved_state(model),
            initialization=initialization,snapshots=snapshots,curves=curves,query_map=model.query_map,
            predictions=dict(train=train_prediction,validation=valid_prediction),frozen_control_predictions=control_predictions,
            mean_query=model.mean_context.tokens.detach().cpu(),mean_query_mask=model.mean_context.program_mask.detach().cpu(),epochs=80,optimizer_updates=80))
        result=dict(status='COMPLETE_R6_TARGET_FIT',contract=contract,arm=args.arm,partition=args.partition,seed=args.seed,
            epochs=80,optimizer_updates=80,actual_new_optimizer_fits=1,teacher_refits=0,source_training_updates=0,
            source_origin=model_origin,initialization=initialization,parameters=model.parameter_budget(),metrics=vm['CP_SCALAR'],
            metrics_vs=vm,train_metrics=tm,curves=curves,risk_diagnostics=dict(train=risk_diagnostics(train_prediction,prepared['base'][train]),
            validation=risk_diagnostics(valid_prediction,prepared['base'][valid])),frozen_controls=controls,attention_probe=attention,gradient_probe=gradient,
            artifact_sha256=sha256(artifact),elapsed_seconds=time.monotonic()-started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else 0,
            outer_test_used=False,primary_epoch_locked_before_training=80)
        write(receipt,result);write(directory/'PROGRESS.json',dict(status='COMPLETE',epoch=80,optimizer_updates_completed=80,artifact_sha256=result['artifact_sha256']))
        return result
    except BaseException as exc:
        write(directory/'ERROR_RECEIPT.json',dict(status='FAILED_NOT_RETRIED',contract=contract,error_type=type(exc).__name__,
            optimizer_updates_completed=updates,elapsed_seconds=time.monotonic()-started,artifact_exists=artifact.exists()))
        (directory/'ERROR_PRIVATE.log').write_text(traceback.format_exc(),encoding='utf-8');raise


def main():
    configure_deterministic_runtime()
    p=parser(__doc__);p.add_argument('--mode',choices=('references','fit'),default='fit')
    p.add_argument('--partition',type=int,choices=PARTITIONS,required=True);p.add_argument('--arm',choices=TARGET_ARMS)
    p.add_argument('--seed',type=int,choices=SEEDS,default=31);args=p.parse_args();ctx=context(args)
    prepared,origin=qualified_prepared(args,ctx,args.partition)
    if args.mode=='references':make_references(args,ctx,prepared,origin);result=dict(status='COMPLETE_R6_ZERO_FIT_REFERENCES',optimizer_updates=0)
    else:result=run_unit(args,ctx,prepared,origin)
    print(json.dumps({k:result[k] for k in ('status','optimizer_updates')}),flush=True)


if __name__=='__main__':main()

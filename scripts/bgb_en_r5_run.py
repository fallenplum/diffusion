"""One complete R5 fitting unit, or immutable convex-reference reuse."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r5_common import (parser,context,qualified_prepared,pairing,bank,write,torch_write,
    validate_qualification,cell_name,cell_contract,subgroup_diagnostics,sha256,require,public_snapshot_guard)
from src.bgb_en.r5_readout import R5MechanismModel
from src.bgb_en.r5_protocol import MAIN_ARMS,MECHANISM_CONTROLS
from src.bgb_en.r5_training import (state_hash,saved_state,full_risk_step,predict,metrics_for,
    risk_diagnostics,refresh_reference,frozen_attention_probe,pure_survival_gradient_probe)
from src.bgb_en.survival import evaluate


def construct(ctx,prepared,patch_bank,arm,seed,control='MATCHED',*,device='cuda'):
    directions={'ONEWAY_O2P_ONLY':'O2P','ONEWAY_P2O_ONLY':'P2O'}
    interventions={'WRONG_QUERY':'wrong_query','QUERY_MEAN':'mean_query',
                   'INTERACTION_OFF_RETRAIN':'interaction_off'}
    torch.manual_seed(seed);np.random.seed(seed)
    return R5MechanismModel(arm,prepared,ctx['member'],patch_bank,device,
        gamma=ctx['config']['readout']['gamma'],direction=directions.get(control,'both'),
        intervention=interventions.get(control),readout_options=dict(rank=4,
            sham_rank=ctx['config']['readout']['sham_rank'],directional_lambda=1e-4,extra_init_std=.02))


def make_references(args,ctx,prepared,origin):
    raw=ctx['raw'];tr,va=prepared['train'],prepared['valid']
    for arm,key in (('A0','CP'),('A1','base')):
        risk=np.asarray(prepared[key][va],float)
        for seed in (31,47,83):
            path=args.private/'references'/f'{arm}_p{args.partition}_seed{seed}.json'
            m=evaluate(raw['t'][tr],raw['e'][tr],raw['t'][va],raw['e'][va],risk,prepared['base'][va])
            vm=metrics_for(raw,prepared,va,{'risk':risk})
            value=dict(status='R5_BASELINE_REFERENCE',group='REFERENCE',arm=arm,report_arm=arm,
                partition=args.partition,seed=seed,metrics=m,metrics_vs=vm,
                actual_new_optimizer_fits=0,deterministic_seed_reference=seed!=31,
                source='same_verified_prepared_bundle_not_a_new_optimization',prepared_origin=origin,
                contract_sha256=ctx['lock']['contract_sha256'])
            if path.exists():require(json.loads(path.read_text())==value,'EXISTING_REFERENCE_CHANGED')
            else:write(path,value)


def run_unit(args,ctx,prepared,origin):
    require(args.arm in MAIN_ARMS and args.control in ('MATCHED',)+MECHANISM_CONTROLS,'UNKNOWN_FORMAL_R5_ARM')
    if args.control!='MATCHED':
        selected=json.loads((args.private/'SELECTION.json').read_text())
        require(selected['status']=='LOCKED_AFTER_COMPLETE_MAIN_BEFORE_MECHANISM'
                and selected['selected_arm']==args.arm and selected['contract_sha256']==ctx['lock']['contract_sha256'],
                'MECHANISM_RETRAIN_NOT_PRESELECTED')
    qualification=validate_qualification(args,ctx)
    contract=cell_contract(ctx,origin,args.arm,args.partition,args.seed,args.control,qualification)
    directory=args.private/'cells'/cell_name(args.arm,args.partition,args.seed,args.control)
    receipt=directory/'COMPLETE.json';artifact=directory/'model.pt'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        require(args.resume and old['contract']==contract and old['artifact_sha256']==sha256(artifact),
                'RESUME_COMPLETED_SOURCE_OR_ARTIFACT_CHANGED')
        return
    require(not directory.exists(),'INTERRUPTED_UNIT_REQUIRES_EXPLICIT_REVIEW_NOT_SILENT_RETRY')
    directory.mkdir(parents=True);started=time.monotonic();updates=0
    write(directory/'START.json',dict(contract=contract,started_unix=time.time(),optimizer_updates_completed=0))
    parent_guard=public_snapshot_guard()
    try:
        patch_bank=bank(ctx,args.device)
        model=construct(ctx,prepared,patch_bank,args.arm,args.seed,args.control,device=args.device)
        mapping,pair_audit=pairing(ctx,prepared,args.partition);model.query_map=mapping
        initial=saved_state(model);initialization=dict(backbone_sha256=state_hash(initial,backbone_only=True),
            full_state_sha256=state_hash(initial))
        train,valid=prepared['train'],prepared['valid'];raw=ctx['raw'];cfg=ctx['config']['training']
        batches=[torch.as_tensor(train[j:j+8],device=args.device) for j in range(0,len(train),8)]
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        curves=[];snapshots={}
        for epoch in range(1,81):
            step=full_risk_step(model,batches,raw['t'][train],raw['e'][train],optimizer,
                total_weight=cfg['total_delta_L2'],cross_weight=cfg['interaction_delta_L2'])
            updates+=1
            write(directory/'PROGRESS.json',dict(status='RUNNING',epoch=epoch,optimizer_updates_completed=updates,
                contract_sha256=ctx['lock']['contract_sha256'],last_step=step))
            if epoch in (20,40,80):
                if model.intervention=='mean_query':refresh_reference(model)
                tp=predict(model,train);vp=predict(model,valid)
                tm=metrics_for(raw,prepared,train,tp,include_pfi=False);vm=metrics_for(raw,prepared,valid,vp,include_pfi=False)
                record=dict(epoch=epoch,optimizer_updates_completed=updates,**step,
                    train=tm['CP_SCALAR'],valid=vm['CP_SCALAR'],
                    train_SD_delta_over_offset=float(tp['delta'].std()/max(np.std(prepared['base'][train]),1e-12)),
                    valid_SD_delta_over_offset=float(vp['delta'].std()/max(np.std(prepared['base'][valid]),1e-12)),
                    train_SD_cross_over_offset=float(tp['cross_delta'].std()/max(np.std(prepared['base'][train]),1e-12)),
                    valid_SD_cross_over_offset=float(vp['cross_delta'].std()/max(np.std(prepared['base'][valid]),1e-12)),
                    checkpoint_selected_by_prelock=epoch==80,outer_test_read=False)
                curves.append(record)
                snapshots[epoch]=dict(state_dict=saved_state(model),risk=vp['risk'],delta=vp['delta'],cross_delta=vp['cross_delta'])
                torch_write(directory/f'checkpoint_epoch{epoch}.pt',dict(contract=contract,epoch=epoch,
                    state_dict=snapshots[epoch]['state_dict'],optimizer_state=optimizer.state_dict(),
                    torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None,
                    optimizer_updates_completed=updates,curves=curves))
        refresh_reference(model)
        train_prediction=predict(model,train,attention=True);valid_prediction=predict(model,valid,attention=True)
        vm=metrics_for(raw,prepared,valid,valid_prediction);tm=metrics_for(raw,prepared,train,train_prediction)
        controls={};control_predictions={}
        for tag in ('matched_query','wrong_query','mean_query','attention_permutation','interaction_off','direct_off','whole_omics'):
            cp=predict(model,valid,control=tag);control_predictions[tag]=cp
            controls[tag]=dict(metrics_vs_base=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],cp['risk'],prepared['base'][valid]),
                relative_to_trained=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],valid_prediction['risk'],cp['risk']),
                risk_SD_ratio=float(np.std(valid_prediction['risk']-cp['risk'])/max(np.std(prepared['base'][valid]),1e-12)),
                frozen_not_retrained=True,whole_modality_replaced=tag=='whole_omics')
        diagnostics=dict(train=risk_diagnostics(train_prediction,prepared['base'][train]),
                         validation=risk_diagnostics(valid_prediction,prepared['base'][valid]))
        attention_probe=frozen_attention_probe(model,valid)
        gradient_probe=pure_survival_gradient_probe(model,prepared,raw)
        result=dict(contract=contract,train=train,valid=valid,query_map=mapping,
            final_state_dict=saved_state(model),snapshots=snapshots,curves=curves,
            predictions=dict(train=train_prediction,validation=valid_prediction),frozen_control_predictions=control_predictions,
            mean_query=model.context_mean.detach().cpu(),mean_query_mask=model.context_mean_mask.detach().cpu(),
            initialization=initialization,pairing=pair_audit,optimizer_updates=updates,epochs=80)
        torch_write(artifact,result)
        require(public_snapshot_guard()==parent_guard,'HISTORICAL_REPORT_MUTATED')
        public=dict(status='COMPLETE_R5_FIT',contract=contract,group='MAIN' if args.control=='MATCHED' else 'MECHANISM',
            arm=args.arm,report_arm=args.arm if args.control=='MATCHED' else args.control,
            training_control=args.control,partition=args.partition,seed=args.seed,
            epochs=80,optimizer_updates=updates,actual_new_optimizer_fits=1,new_teacher_fits=0,
            metrics=vm['CP_SCALAR'],metrics_vs=vm,train_metrics=tm,
            parameters=model.parameter_budget(),initialization=initialization,
            readout_coefficients={key:float(getattr(model.readout,key).detach()) for key in ('a_o','a_p')}
                if model.readout_mode=='directional' else None,
            risk_diagnostics=diagnostics,curves=curves,frozen_controls=controls,pairing_audit=pair_audit,
            attention_probe=attention_probe,gradient_probe=gradient_probe,
            subgroups=subgroup_diagnostics(raw,prepared,valid_prediction['risk']),
            elapsed_seconds=time.monotonic()-started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(args.device).startswith('cuda') else 0,
            artifact_sha256=sha256(artifact),outer_test_used=False,primary_epoch_locked_before_training=80)
        write(receipt,public)
        write(directory/'PROGRESS.json',dict(status='COMPLETE',epoch=80,optimizer_updates_completed=80,artifact_sha256=public['artifact_sha256']))
        print(json.dumps({k:public[k] for k in ('status','arm','training_control','partition','seed','elapsed_seconds')}),flush=True)
    except Exception as exc:
        write(directory/'ERROR_RECEIPT.json',dict(status='FAILED_NOT_RETRIED',contract=contract,
            error_type=type(exc).__name__,optimizer_updates_completed=updates,
            elapsed_seconds=time.monotonic()-started,artifact_exists=artifact.exists()))
        (directory/'ERROR_PRIVATE.log').write_text(traceback.format_exc())
        raise


def main():
    p=parser(__doc__);p.add_argument('--stage',choices=('reference','train'),required=True)
    p.add_argument('--partition',type=int,choices=(0,1,2),required=True)
    p.add_argument('--seed',type=int,choices=(31,47,83),default=31)
    p.add_argument('--arm',choices=MAIN_ARMS,default='A3')
    p.add_argument('--control',choices=('MATCHED',)+MECHANISM_CONTROLS,default='MATCHED')
    args=p.parse_args();torch.set_num_threads(2);ctx=context(args)
    prepared,origin=qualified_prepared(args,ctx,args.partition)
    if args.stage=='reference':make_references(args,ctx,prepared,origin)
    else:run_unit(args,ctx,prepared,origin)


if __name__=='__main__':main()

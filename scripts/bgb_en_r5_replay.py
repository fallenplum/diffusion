"""Independent saved-state replay and direct retrained patient-pair credits."""
from __future__ import annotations
from pathlib import Path
import sys
import json
import re
import traceback
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r5_common import (parser,context,qualified_prepared,bank,write,sha256,require,
    validate_qualification,public_snapshot_guard,cell_name)
from scripts.bgb_en_r5_run import construct
from src.bgb_en.r5_training import predict,refresh_reference,metrics_for
from src.bgb_en.replay import apply_saved_preprocessing,replay_full_risks
from src.bgb_en.survival import evaluate
from src.phase10_trimodal.reference import components,pairs,uno_weights


def credit(time,event,risk):
    i,j=pairs(time,event);difference=np.asarray(risk)[i]-np.asarray(risk)[j]
    return np.where(np.abs(difference)<=1e-8,.5,(difference>0).astype(float))


def independent_metrics(raw,prepared,risk):
    train,valid=prepared['train'],prepared['valid']
    w=uno_weights(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],tau=60.)
    hn,hd=components(raw['t'][valid],raw['e'][valid],risk)
    un,ud=components(raw['t'][valid],raw['e'][valid],risk,w)
    return dict(Harrell_num=hn,Harrell_den=hd,Uno_num=un,Uno_den=ud,Harrell=hn/hd,Uno60=un/ud)


def upstream_replay(ctx,prepared):
    raw=ctx['raw'];errors=[];coefficient_models=0
    features=apply_saved_preprocessing(raw,prepared['features']['state'])
    for key,value in features.items():
        if key=='state':continue
        target=prepared['features'][key];error=float(np.max(np.abs(value.astype(float)-target.astype(float))))
        require(error<=(2e-12 if value.dtype==np.float64 else 0.),'SAVED_FULL_PREPROCESSING_REPLAY_FAILED')
    full=replay_full_risks(features,prepared['full_models']);coefficient_models+=len(full)
    errors.extend(float(np.max(np.abs(v-prepared['full_risks'][k]))) for k,v in full.items())
    for teacher in prepared['teachers']:
        feature=apply_saved_preprocessing(raw,teacher['preprocessing'])
        risks=replay_full_risks(feature,teacher['models']);coefficient_models+=len(risks)
        for key,target in (('LOCAL','CP'),('CS','CS'),('R','RNA')):
            held=teacher['held'];errors.append(float(np.max(np.abs(risks[key][held]-prepared[target][held]))))
    require(max(errors)<=2e-12,'UPSTREAM_COX_COEFFICIENT_RISK_REPLAY_FAILED')
    train=prepared['train'];left=[];right=[]
    for group in np.unique(prepared['teacher_groups'][train]):
        idx=train[prepared['teacher_groups'][train]==group]
        a,b=pairs(raw['t'][idx],raw['e'][idx]);left.extend(idx[a]);right.extend(idx[b])
    a,b=np.asarray(left,int),np.asarray(right,int)
    dr=prepared['RNA'][a]-prepared['RNA'][b]
    margin=prepared['CP'][a]-prepared['CP'][b]+prepared['beta']*dr
    gradient=float(np.mean(-expit(-margin)*dr)+prepared['beta'])
    require(abs(gradient)<=1e-6,'SAME_TEACHER_BETA_GRADIENT_REPLAY_FAILED')
    return dict(status='PASS',coefficient_models=coefficient_models,beta_gradient_abs=abs(gradient),
        maximum_risk_error=max(errors),new_optimizer_fits=0,new_preprocessing_fits=0,
        teacher_excluded_patient_labels_not_used_for_fitting=True)


def main():
    p=parser(__doc__);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    require(args.limit>=0,'NEGATIVE_REPLAY_LIMIT');torch.set_num_threads(2)
    ctx=context(args);validate_qualification(args,ctx);parent=public_snapshot_guard()
    entries=[]
    for receipt in sorted((args.private/'cells').glob('*/COMPLETE.json')):
        row=json.loads(receipt.read_text());require(row['status']=='COMPLETE_R5_FIT','WRONG_COMPLETION_MARKER')
        require(row['contract']['contract_sha256']==ctx['lock']['contract_sha256'],'R5_COMPLETE_CONTRACT_CHANGED')
        require(receipt.parent.name==cell_name(row['arm'],row['partition'],row['seed'],row['training_control']),
                'COMPLETE_UNIT_IDENTITY_CHANGED')
        artifact=receipt.parent/'model.pt';require(sha256(artifact)==row['artifact_sha256'],'R5_COMPLETE_ARTIFACT_CHANGED')
        entries.append((receipt,row,artifact))
    output=args.private/'REPLAY_AUDIT.json';previous={}
    if args.resume and output.exists():
        old=json.loads(output.read_text());require(old['contract_sha256']==ctx['lock']['contract_sha256'],'REPLAY_ORIGIN_CHANGED')
        previous={row['unit']:row for row in old.get('units',[]) if row['status']=='PASS'}
    prepared_cache={};upstream=[]
    for part in (0,1,2):
        prepared,origin=qualified_prepared(args,ctx,part);prepared_cache[part]=prepared
        upstream.append(dict(partition=part,**upstream_replay(ctx,prepared)))
    patch_bank=bank(ctx,args.device);rows=[];new=0;risks={}
    for receipt,metadata,artifact_path in entries:
        unit=receipt.parent.name;part,seed,arm,control=(metadata[k] for k in ('partition','seed','arm','training_control'))
        prepared=prepared_cache[part];artifact=torch.load(artifact_path,map_location='cpu',weights_only=False)
        require(artifact['contract']==metadata['contract'],'PRIVATE_PUBLIC_CONTRACT_MISMATCH')
        require(np.array_equal(artifact['train'],prepared['train']) and np.array_equal(artifact['valid'],prepared['valid']),
                'SAVED_PATIENT_SCOPE_CHANGED')
        risks[(part,seed,metadata['report_arm'])]=artifact['predictions']['validation']['risk']
        if unit in previous:
            prior=previous[unit];require(prior['artifact_sha256']==metadata['artifact_sha256'],'REPLAYED_ARTIFACT_CHANGED')
            rows.append(prior);continue
        if args.limit and new>=args.limit:continue
        result=dict(unit=unit,group=metadata['group'],arm=arm,control=control,partition=part,seed=seed,
            artifact_sha256=metadata['artifact_sha256'],status='FAIL')
        try:
            model=construct(ctx,prepared,patch_bank,arm,seed,control,device=args.device)
            model.load_state_dict(artifact['final_state_dict'],strict=True);model.query_map=np.asarray(artifact['query_map'])
            refresh_reference(model)
            mean_error=float((model.context_mean.cpu()-artifact['mean_query']).abs().max())
            require(mean_error<=2e-6 and torch.equal(model.context_mean_mask.cpu(),artifact['mean_query_mask']),
                    'TRAIN_ONLY_MEAN_REFERENCE_CHANGED')
            valid=prepared['valid'];prediction=predict(model,valid)
            saved=artifact['predictions']['validation'];maximum=max(float(np.max(np.abs(prediction[k]-saved[k])))
                for k in ('risk','delta','cross_delta'))
            require(maximum<=2e-6,'SAVED_RISK_NUMERIC_TOLERANCE_FAILED')
            time,event=ctx['raw']['t'][valid],ctx['raw']['e'][valid]
            changed=int(np.sum(credit(time,event,prediction['risk'])!=credit(time,event,saved['risk'])))
            require(changed==0,'SAVED_RISK_PAIR_CREDIT_CHANGED')
            independent=independent_metrics(ctx['raw'],prepared,prediction['risk'])
            error=max(abs(independent[k]-metadata['metrics'][k]) for k in independent)
            require(error<=1e-10,'INDEPENDENT_CONCORDANCE_CHANGED')
            metrics=metrics_for(ctx['raw'],prepared,valid,prediction)
            pfi_error=max(abs(metrics['PFI_frozen_DSS_risk'][k]-metadata['metrics_vs']['PFI_frozen_DSS_risk'][k])
                for k in ('Harrell_num','Harrell_den','Uno_num','Uno_den'))
            require(pfi_error<=1e-10,'FROZEN_PFI_REPLAY_CHANGED')
            control_errors={}
            for tag,target in artifact['frozen_control_predictions'].items():
                current=predict(model,valid,control=tag)
                value=float(np.max(np.abs(current['risk']-target['risk'])))
                require(value<=2e-6,'FROZEN_QUERY_RISK_REPLAY_FAILED')
                require(np.array_equal(credit(time,event,current['risk']),credit(time,event,target['risk'])),
                        'FROZEN_QUERY_PAIR_CREDIT_CHANGED')
                control_errors[tag]=value
                full_metrics=evaluate(ctx['raw']['t'][prepared['train']],ctx['raw']['e'][prepared['train']],time,event,
                    prediction['risk'],current['risk'])
                stored=metadata['frozen_controls'][tag]['relative_to_trained']
                require(all(abs(full_metrics[k]-stored[k])<=1e-10 for k in
                    ('Harrell_num','Harrell_den','Uno_num','Uno_den','repair','damage','Uno_repair','Uno_damage')),
                    'FROZEN_CONTROL_DIRECT_PAIR_CREDIT_CHANGED')
            result.update(status='PASS',maximum_risk_error=maximum,changed_pair_credits=changed,
                maximum_metric_error=error,maximum_PFI_component_error=pfi_error,
                frozen_control_risk_errors=control_errors,mean_query_error=mean_error,
                actual_state_dict_forward_replayed=True,new_optimizer_updates=0)
            del model
        except Exception as exc:
            message=str(exc);code=message if re.fullmatch('[A-Z0-9_]+',message) else 'RUNTIME_REPLAY_FAILED'
            result.update(failure_type=type(exc).__name__,failure_code=code)
            logs=args.private/'REPLAY_PRIVATE';logs.mkdir(exist_ok=True)
            (logs/(unit+'.log')).write_text(traceback.format_exc())
        rows.append(result);new+=1
        write(args.private/'REPLAY_PROGRESS.json',dict(contract_sha256=ctx['lock']['contract_sha256'],units=rows,new_actual_replays=new))
        print(json.dumps({k:result[k] for k in ('unit','status')}),flush=True)
    passed=sum(r['status']=='PASS' for r in rows);failed=len(rows)-passed
    main_passed=sum(r['status']=='PASS' and r['group']=='MAIN' for r in rows)
    mechanism_passed=sum(r['status']=='PASS' and r['group']=='MECHANISM' for r in rows)
    complete=main_passed==63 and mechanism_passed==45 and not failed
    paired=[];main_paired=[]
    if complete:
        chosen=json.loads((args.private/'SELECTION.json').read_text())['selected_arm']
        for part in (0,1,2):
            prepared=prepared_cache[part];train,valid=prepared['train'],prepared['valid'];raw=ctx['raw']
            for seed in (31,47,83):
                matched=risks[(part,seed,chosen)]
                for reference in ('A1','A2','A3','A8'):
                    control_risk=prepared['base'][valid] if reference=='A1' else risks[(part,seed,reference)]
                    main_paired.append(dict(group='MAIN',selected_arm=chosen,reference=reference,partition=part,seed=seed,
                        matched_vs_control=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],matched,control_risk),
                        risk_SD_ratio=float(np.std(matched-control_risk)/max(np.std(prepared['base'][valid]),1e-12))))
                for control in ('WRONG_QUERY','QUERY_MEAN','INTERACTION_OFF_RETRAIN','ONEWAY_O2P_ONLY','ONEWAY_P2O_ONLY'):
                    cr=risks[(part,seed,control)]
                    paired.append(dict(group='MECHANISM',selected_arm=chosen,control=control,partition=part,seed=seed,
                        matched_vs_control=evaluate(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],matched,cr),
                        risk_SD_ratio=float(np.std(matched-cr)/max(np.std(prepared['base'][valid]),1e-12))))
    require(public_snapshot_guard()==parent,'HISTORICAL_REPORT_MUTATED_BY_REPLAY')
    result=dict(status='PASS_COMPLETE_R5_CHECKPOINT_REPLAY' if complete else
        'FAILED_R5_CHECKPOINT_REPLAY' if failed else 'PASS_AVAILABLE_SUBSET_NOT_COMPLETE',
        contract_sha256=ctx['lock']['contract_sha256'],passed_units=passed,failed_units=failed,
        main_passed=main_passed,mechanism_passed=mechanism_passed,available_units=len(entries),
        new_actual_replays=new,reused_actual_replays=len(rows)-new,units=rows,
        upstream_reference_replays=upstream,paired_retrained_controls=paired,paired_main_controls=main_paired,
        patient_arrays_exported=False,new_optimizer_fits=0,new_preprocessing_fits=0,
        numerical_risk_tolerance=2e-6,maximum_allowed_changed_pair_credits=0)
    write(output,result)
    write(ROOT/'reports/bgb_en_r5/R5_CHECKPOINT_REPLAY_AUDIT.json',result)
    print(json.dumps({k:result[k] for k in ('status','passed_units','failed_units','main_passed','mechanism_passed')}))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()

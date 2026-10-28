"""Actual R7 replay of every saved 20/40/80 state, no optimizer updates."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import json
import re
import sys
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r7_common import (parser,context,qualified_prepared,construct,validate_qualification,require,read,write,
    sha256,PARTITIONS,SEEDS,TARGET_ARMS,EVALUATION_EPOCHS,frozen_controls_for_arm,configure_deterministic_runtime,
    validate_deterministic_runtime,public_snapshot_guard)
from scripts.bgb_en_r7_run import cell_name
from scripts.bgb_en_r6_replay import upstream_replay,credit,independent_metrics,assert_metrics
from src.bgb_en.r7_training import predict,saved_state,state_hash,SCALARS
from src.bgb_en.r6_pathology import source_model_state_sha256


def planned_units():return {cell_name(a,p,s):(a,p,s) for a in TARGET_ARMS for p in PARTITIONS for s in SEEDS}


def score(raw,prepared,indices,risk,reference=None,*,endpoint='DSS'):
    train=prepared['train']
    if endpoint=='DSS':return independent_metrics(raw['t'][train],raw['e'][train],raw['t'][indices],raw['e'][indices],risk,reference)
    require(endpoint=='PFI','R7_UNKNOWN_ENDPOINT')
    usable=raw['pfi_valid'][indices];fit=train[raw['pfi_valid'][train]]
    return independent_metrics(raw['pfi_t'][fit],raw['pfi_e'][fit],raw['pfi_t'][indices][usable],raw['pfi_e'][indices][usable],
        np.asarray(risk)[usable],None if reference is None else np.asarray(reference)[usable])


def upstream_audit(args,ctx,prepared,origins):
    rows=[dict(partition=p,**upstream_replay(ctx['r6_ctx'],prepared[p])) for p in PARTITIONS]
    path=args.private/'UPSTREAM_REPLAY_AUDIT.json'
    result=read(path);qualification=read(args.private/'QUALIFICATION.json')
    require(sha256(path)==qualification['upstream_replay_audit_sha256']
        and result.get('status')=='PASS_R7_REUSED_UPSTREAM_39_COX_AND_3_BETA' and result.get('rows')==rows
        and sum(r['coefficient_models'] for r in rows)==39
        and result.get('inherited_R6_contract_sha256')==ctx['upstream_r6']['contract_sha256']
        and result.get('teacher_refits')==result.get('source_refits')==result.get('new_optimizer_fits')==0,
        'R7_QUALIFIED_UPSTREAM_AUDIT_CHANGED')
    return result,sha256(path)


def reference_replay(args,ctx,prepared,origins):
    expected={(a,p,s) for a in ('B0','B1') for p in PARTITIONS for s in SEEDS};rows=[]
    paths=list((args.private/'references').glob('*.json'))
    require(len(paths)==18,'R7_EXACT_18_ZERO_FIT_REFERENCE_ROWS_REQUIRED')
    seen=set()
    for path in paths:
        value=read(path);key=(value['arm'],value['partition'],value['seed'])
        require(key in expected and key not in seen and path.name==f'{key[0]}_p{key[1]}_seed{key[2]}.json','R7_REFERENCE_IDENTITY_CHANGED');seen.add(key)
        require(value['status']=='R7_BASELINE_REFERENCE' and value['group']=='REFERENCE'
            and value['contract_sha256']==ctx['lock']['contract_sha256'] and value['prepared_origin']==origins[key[1]]
            and value['actual_new_optimizer_fits']==value['teacher_refits']==0,'R7_REFERENCE_CONTRACT_CHANGED')
        bundle=prepared[key[1]];indices=bundle['valid'];risk=bundle['CP' if key[0]=='B0' else 'base'][indices]
        for label,ref in (('CP_STRONG',bundle['CP'][indices]),('CP_SCALAR',bundle['base'][indices])):
            assert_metrics(score(ctx['raw'],bundle,indices,risk,ref),value['metrics_vs'][label])
        assert_metrics(score(ctx['raw'],bundle,indices,risk,endpoint='PFI'),value['metrics_vs']['PFI_frozen_DSS_risk'])
        assert_metrics(score(ctx['raw'],bundle,indices,risk,bundle['base'][indices]),value['metrics'])
        rows.append(dict(arm=key[0],partition=key[1],seed=key[2],status='PASS',receipt_sha256=sha256(path),
            independent_fixed_anchor_metrics_recomputed=True,new_optimizer_fits=0))
    return rows


def validate_entry(receipt,metadata,artifact,ctx,args,origin,qualification_sha):
    arm,part,seed=metadata['arm'],metadata['partition'],metadata['seed']
    require(planned_units().get(receipt.parent.name)==(arm,part,seed) and metadata['status']=='COMPLETE_R7_TARGET_FIT',
        'R7_COMPLETE_IDENTITY_CHANGED')
    c=metadata['contract']
    require(c==artifact['contract'] and c['contract_sha256']==ctx['lock']['contract_sha256'] and c['prepared_origin']==origin
        and c['qualification_sha256']==qualification_sha and c['config_sha256']==sha256(args.config)
        and c['source_config_sha256']==sha256(args.source_config) and c['upstream_r6']==ctx['upstream_r6']
        and c['fixed_deterministic_runtime']==validate_deterministic_runtime()
        and (c['arm'],c['partition'],c['seed'])==(arm,part,seed)
        and c['epochs']==metadata['epochs']==metadata['optimizer_updates']==artifact['epochs']==artifact['optimizer_updates']==80
        and metadata['actual_new_optimizer_fits']==1 and metadata['teacher_refits']==metadata['source_training_updates']==0
        and metadata['outer_test_used'] is False,'R7_COMPLETE_INPUT_SOURCE_BUDGET_CHANGED')
    require(metadata['source_origin']==artifact['model_origin'] and metadata['source_origin']['source_key']=='MATCHED'
        and metadata['source_origin']['source_origin']==ctx['source_audits']['MATCHED'],'R7_WRONG_FOUNDATION_ORIGIN')


def compare_predictions(raw,indices,current,target,*,code):
    require(set(current)==set(target),code+'_SCALAR_INVENTORY_CHANGED')
    maximum=0.
    for name,value in current.items():
        expected=np.asarray(target[name]);require(value.shape==expected.shape and len(value)==len(indices),code+'_SHAPE_CHANGED')
        require(np.isfinite(value).all() and np.isfinite(expected).all(),code+'_NONFINITE_SAVED_SCALAR')
        maximum=max(maximum,float(np.max(np.abs(value-expected))))
    require(maximum<=2e-6,code+'_NUMERIC_TOLERANCE_FAILED')
    require(np.array_equal(credit(raw['t'][indices],raw['e'][indices],current['risk']),
        credit(raw['t'][indices],raw['e'][indices],target['risk'])),code+'_PAIR_CREDITS_CHANGED')
    return maximum


def replay_one(args,ctx,prepared,origin,receipt,metadata,artifact,qualification_sha):
    validate_entry(receipt,metadata,artifact,ctx,args,origin,qualification_sha)
    require(np.array_equal(artifact['train'],prepared['train']) and np.array_equal(artifact['valid'],prepared['valid']),
        'R7_SAVED_PATIENT_SCOPE_CHANGED')
    local=argparse.Namespace(**(vars(args)|dict(partition=metadata['partition'])))
    model,current_origin=construct(local,ctx,prepared,metadata['arm'],metadata['seed'])
    require(current_origin==metadata['source_origin'],'R7_RECONSTRUCTED_MODEL_ORIGIN_CHANGED')
    initial=saved_state(model)
    require(state_hash(initial)==metadata['initialization']['full_state_sha256']
        and state_hash(initial,target_only=True)==metadata['initialization']['target_state_sha256']
        and artifact['initialization']==metadata['initialization'],'R7_INITIALIZATION_CHANGED')
    require(set(artifact['snapshots'])==set(EVALUATION_EPOCHS)
        and state_hash(artifact['snapshots'][80]['state_dict'])==state_hash(artifact['final_state_dict'])
        and np.array_equal(model.query_map,artifact['query_map']),'R7_FINAL_OR_QUERY_MAPPING_CHANGED')
    checkpoints=[];final_risk=None;raw=ctx['raw']
    for epoch in EVALUATION_EPOCHS:
        checkpoint=receipt.parent/f'checkpoint_epoch{epoch}.pt'
        require(sha256(checkpoint)==metadata['checkpoint_artifact_sha256'][str(epoch)],'R7_CHECKPOINT_BYTES_CHANGED')
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False);snapshot=artifact['snapshots'][epoch]
        require(cp['contract']==metadata['contract'] and cp['model_origin']==metadata['source_origin']
            and cp['epoch']==cp['optimizer_updates_completed']==snapshot['optimizer_updates_completed']==epoch
            and state_hash(cp['state_dict'])==state_hash(snapshot['state_dict'])==state_hash(cp['snapshot']['state_dict']),
            'R7_CHECKPOINT_INTERNAL_EPOCH_STATE_CHANGED')
        model.load_state_dict(snapshot['state_dict'],strict=True)
        require(source_model_state_sha256(model.core.foundation)==current_origin['source_origin']['complete_source_state_sha256'],
            'R7_FROZEN_SOURCE_CHANGED_IN_CHECKPOINT')
        model.refresh_training_query_mean()
        mean_error=float((model.mean_context.tokens.cpu()-snapshot['mean_query']).abs().max())
        require(mean_error<=2e-6 and torch.equal(model.mean_context.program_mask.cpu(),snapshot['mean_query_mask']),
            'R7_CHECKPOINT_TRAIN_MEAN_CHANGED')
        errors={};risks={};metric_errors=[]
        for split,indices in (('train',prepared['train']),('validation',prepared['valid'])):
            current=predict(model,indices);target=snapshot['predictions'][split]
            errors[split]=compare_predictions(raw,indices,current,target,code='R7_CHECKPOINT_'+split.upper())
            expected=snapshot['train_metrics'] if split=='train' else snapshot['metrics_vs']
            for label,ref in (('CP_STRONG',prepared['CP'][indices]),('CP_SCALAR',prepared['base'][indices])):
                metric_errors.append(assert_metrics(score(raw,prepared,indices,current['risk'],ref),expected[label]))
            if epoch==80:metric_errors.append(assert_metrics(score(raw,prepared,indices,current['risk'],endpoint='PFI'),expected['PFI_frozen_DSS_risk']))
            risks[split]=current['risk']
        controls={};valid=prepared['valid'];expected_tags=set(frozen_controls_for_arm(model.arm))
        require(set(snapshot['frozen_controls'])==set(snapshot['frozen_control_predictions'])==expected_tags,
            'R7_CHECKPOINT_FROZEN_INVENTORY_CHANGED')
        for tag in frozen_controls_for_arm(model.arm):
            current=predict(model,valid,control=tag)
            controls[tag]=compare_predictions(raw,valid,current,snapshot['frozen_control_predictions'][tag],code='R7_CHECKPOINT_FROZEN')
            stored=snapshot['frozen_controls'][tag]
            assert_metrics(score(raw,prepared,valid,current['risk'],prepared['base'][valid]),stored['metrics_vs_base'])
            assert_metrics(score(raw,prepared,valid,risks['validation'],current['risk']),stored['relative_to_trained'])
        checkpoints.append(dict(epoch=epoch,status='PASS',checkpoint_sha256=sha256(checkpoint),
            maximum_risk_error=max([*errors.values(),*controls.values()]),risk_errors_by_split=errors,
            changed_pair_credits=0,frozen_control_risk_errors=controls,maximum_metric_error=max(metric_errors),
            mean_query_error=mean_error,actual_state_dict_forward_replayed=True,new_optimizer_updates=0))
        if epoch==80:final_risk=risks['validation']
    require(all(np.array_equal(artifact['predictions'][split][key],artifact['snapshots'][80]['predictions'][split][key])
        for split in ('train','validation') for key in SCALARS),'R7_FINAL_PREDICTIONS_NOT_EPOCH80')
    for epoch in (20,40,60,80):
        path=receipt.parent/f'DIAGNOSTIC_epoch{epoch}.json'
        require(read(path)==metadata['checkpoint_diagnostics'][str(epoch)],'R7_SAVED_DIAGNOSTIC_RECEIPT_CHANGED')
    return dict(status='PASS',checkpoints=checkpoints,actual_checkpoint_count=3,
        maximum_risk_error=max(row['maximum_risk_error'] for row in checkpoints),changed_pair_credits=0,
        all_checkpoints_actual_forward_replayed=True,actual_state_dict_forward_replayed=True,
        source_frozen_weight_hash_replayed=True,saved_query_mapping_recomputed=True,new_optimizer_updates=0),final_risk


def paired_comparisons(ctx,prepared,risks):
    result=[]
    for comparison in ctx['config']['paired_comparisons']:
        candidate,reference=comparison['candidate'],comparison['reference']
        for part in PARTITIONS:
            bundle=prepared[part];valid=bundle['valid']
            for seed in SEEDS:
                a=risks[(part,seed,candidate)]
                b=bundle['base'][valid] if reference=='B1' else bundle['CP'][valid] if reference=='B0' else risks[(part,seed,reference)]
                result.append(dict(candidate=candidate,reference=reference,partition=part,seed=seed,
                    candidate_vs_reference=score(ctx['raw'],bundle,valid,a,b),PFI_frozen_DSS_risk=score(ctx['raw'],bundle,valid,a,b,endpoint='PFI'),
                    risk_SD_ratio=float(np.std(a-b)/max(np.std(bundle['base'][valid]),1e-12)),direct_risk_pair_credits=True,
                    bootstrap_or_CI_computed=False,primary_checkpoint_epoch=80))
    return result


def persist(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(read(path)==value,'R7_EXISTING_TERMINAL_RECEIPT_PRESERVED')
    else:
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def verify_saved_auxiliary_files(receipt,metadata):
    for epoch in EVALUATION_EPOCHS:
        require(sha256(receipt.parent/f'checkpoint_epoch{epoch}.pt')==metadata['checkpoint_artifact_sha256'][str(epoch)],
            'R7_CHECKPOINT_BYTES_CHANGED')
    for epoch in (20,40,60,80):
        require(read(receipt.parent/f'DIAGNOSTIC_epoch{epoch}.json')==metadata['checkpoint_diagnostics'][str(epoch)],
            'R7_SAVED_DIAGNOSTIC_RECEIPT_CHANGED')


def run(args,ctx=None):
    ctx=context(args) if ctx is None else ctx;qualification=validate_qualification(args,ctx)
    require(ctx['lock']['sources'].get('scripts/bgb_en_r7_replay.py')==sha256(__file__),'R7_REPLAY_SOURCE_NOT_LOCKED')
    parent=public_snapshot_guard();prepared={};origins={}
    for part in PARTITIONS:prepared[part],origins[part]=qualified_prepared(args,ctx,part)
    upstream,upstream_sha=upstream_audit(args,ctx,prepared,origins);references=reference_replay(args,ctx,prepared,origins)
    terminal=args.private/'REPLAY_PRIVATE';output=args.private/'REPLAY_AUDIT.json';rows=[];risks={};new=0;reused=0
    require(args.resume or not output.exists(),'R7_REPLAY_EXPLICIT_RESUME_REQUIRED')
    if output.exists():require(read(output)['contract_sha256']==ctx['lock']['contract_sha256'],'R7_EXISTING_REPLAY_CONTRACT_CHANGED')
    expected=planned_units();entries=[]
    for receipt in sorted((args.private/'cells').glob('*/COMPLETE.json')):
        require(receipt.parent.name in expected and not (receipt.parent/'ERROR_RECEIPT.json').exists(),'R7_UNEXPECTED_OR_FAILED_COMPLETION')
        entries.append((receipt,read(receipt)))
    for receipt,metadata in entries:
        name=receipt.parent.name;artifact_path=receipt.parent/'model.pt'
        verify_saved_auxiliary_files(receipt,metadata)
        identity=dict(unit=name,arm=metadata['arm'],partition=metadata['partition'],seed=metadata['seed'],
            contract_sha256=ctx['lock']['contract_sha256'],completion_sha256=sha256(receipt),artifact_sha256=sha256(artifact_path),
            qualification_sha256=qualification,upstream_replay_audit_sha256=upstream_sha,replay_script_sha256=sha256(__file__))
        passed,failed=terminal/(name+'.PASS.json'),terminal/(name+'.FAIL.json')
        require(not (passed.exists() and failed.exists()),'R7_CONFLICTING_REPLAY_TERMINALS')
        if passed.exists() or failed.exists():
            require(args.resume,'R7_EXISTING_REPLAY_REQUIRES_RESUME');prior=read(passed if passed.exists() else failed)
            require(all(prior.get(k)==v for k,v in identity.items()),'R7_RESUMED_REPLAY_CONTRACT_OR_ARTIFACT_CHANGED')
            rows.append(prior);reused+=1
            if passed.exists():
                require(prior['status']=='PASS' and prior.get('actual_checkpoint_count')==3
                    and prior.get('all_checkpoints_actual_forward_replayed') is True,'R7_RESUMED_PASS_NOT_THREE_ACTUAL_REPLAYS')
                artifact=torch.load(artifact_path,map_location='cpu',weights_only=False)
                validate_entry(receipt,metadata,artifact,ctx,args,origins[metadata['partition']],qualification)
                risks[(metadata['partition'],metadata['seed'],metadata['arm'])]=artifact['predictions']['validation']['risk']
            else:require(prior['status']=='FAIL','R7_INVALID_PRESERVED_FAILURE_STATUS')
            continue
        if args.limit and new>=args.limit:continue
        result=dict(identity,status='FAIL')
        try:
            require(identity['artifact_sha256']==metadata['artifact_sha256'],'R7_COMPLETE_ARTIFACT_SHA_CHANGED')
            artifact=torch.load(artifact_path,map_location='cpu',weights_only=False)
            details,risk=replay_one(args,ctx,prepared[metadata['partition']],origins[metadata['partition']],receipt,metadata,artifact,qualification)
            result.update(details);risks[(metadata['partition'],metadata['seed'],metadata['arm'])]=risk;persist(passed,result)
        except Exception as exc:
            message=str(exc);result.update(failure_type=type(exc).__name__,failure_code=message if re.fullmatch('[A-Z0-9_]+',message) else 'R7_RUNTIME_REPLAY_FAILED')
            terminal.mkdir(parents=True,exist_ok=True)
            with (terminal/(name+'.log')).open('x',encoding='utf-8') as stream:stream.write(traceback.format_exc())
            persist(failed,result)
        rows.append(result);new+=1
        write(args.private/'REPLAY_PROGRESS.json',dict(contract_sha256=ctx['lock']['contract_sha256'],units=rows,expected_units=99,
            new_actual_replays=new,new_optimizer_fits=0))
        print(json.dumps(dict(unit=name,status=result['status'])),flush=True)
    n_pass=sum(row['status']=='PASS' for row in rows);n_fail=sum(row['status']=='FAIL' for row in rows)
    complete=len(entries)==len(rows)==n_pass==99 and n_fail==0
    require(public_snapshot_guard()==parent,'R7_HISTORICAL_REPORTS_CHANGED_DURING_REPLAY')
    result=dict(status='PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY' if complete else 'FAILED_R7_CHECKPOINT_REPLAY' if n_fail else 'INCOMPLETE_R7_CHECKPOINT_REPLAY',
        contract_sha256=ctx['lock']['contract_sha256'],qualification_sha256=qualification,upstream_replay_audit_sha256=upstream_sha,
        expected_units=99,available_units=len(entries),passed_units=n_pass,failed_units=n_fail,expected_checkpoints=297,
        passed_checkpoints=sum(row.get('actual_checkpoint_count',0) for row in rows if row['status']=='PASS'),
        missing_or_unreplayed_units=sorted(set(expected)-{row['unit'] for row in rows}),units=rows,new_actual_replays=new,
        reused_terminal_receipts=reused,upstream_reference_replay=upstream,zero_fit_baseline_reference_replays=references,
        paired_retrained_controls=paired_comparisons(ctx,prepared,risks) if complete else [],new_optimizer_fits=0,new_preprocessing_fits=0,
        primary_checkpoint_epoch=80,diagnostic_checkpoints_not_selected=True,patient_arrays_exported=False)
    write(output,result);return result


def main():
    configure_deterministic_runtime();torch.set_num_threads(2)
    p=parser(__doc__);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    require(args.limit>=0,'R7_NEGATIVE_REPLAY_LIMIT');result=run(args)
    print(json.dumps({k:result[k] for k in ('status','passed_units','failed_units','passed_checkpoints')}),flush=True)
    if result['failed_units']:raise SystemExit(1)


if __name__=='__main__':main()

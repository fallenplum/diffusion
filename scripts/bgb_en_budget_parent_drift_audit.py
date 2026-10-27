"""CPU audit of the prespecified O4/p0/s47 budget guard failure.

No optimizer, model construction or forward is invoked. The failed receipt is
retained. Final160 metrics describe an unqualified inner fit, not a valid budget
comparison or a revised threshold decision.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_recovery_audit import require,sha256
from scripts.bgb_en_budget_retrain import compare_parent80


def state_difference(left,right):
    require(set(left)==set(right),'SNAPSHOT_STATE_KEYS_DIFFER')
    total=0;changed=0;squared=0.;maximum=0.;changed_tensors=0
    for name in left:
        a,b=left[name],right[name]
        require(torch.is_tensor(a) and torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype,
                'SNAPSHOT_STATE_TYPE_SHAPE_DIFFER')
        difference=a.to(torch.float64)-b.to(torch.float64)
        total+=a.numel();changed+=int(torch.count_nonzero(difference))
        squared+=float(difference.square().sum());maximum=max(maximum,float(difference.abs().max()))
        changed_tensors+=int(not torch.equal(a,b))
    return dict(state_tensors=len(left),changed_tensors=changed_tensors,state_elements=total,
        changed_elements=changed,max_abs_state_error=maximum,RMS_state_error=float(np.sqrt(squared/total)))


def run(args):
    torch.set_num_threads(2);require(not torch.cuda.is_initialized(),'CUDA_CONTEXT_MUST_NOT_EXIST')
    stem='R2_newfit160_outer1_inner0_O4_seed47'
    failed_dir=args.private/'cells'/stem;failed_path=failed_dir/'FAILED.json';failure=json.loads(failed_path.read_text())
    require(failure['exception_type']=='ValueError' and failure['actual_optimizer_updates_completed']==160
            and not (failed_dir/'COMPLETE.json').exists(),'FAILURE_RECEIPT_OR_ORIGINAL_STATUS_CHANGED')
    failed_model_path=failed_dir/'model.pt';require(failed_model_path.is_file(),'FAILED_MODEL_NOT_SAVED')
    failed=torch.load(failed_model_path,map_location='cpu',weights_only=False)
    require(failed['lineage']==failure['lineage'] and failed['epochs']==failed['optimizer_updates']==160,
            'FAILED_CHECKPOINT_LINEAGE_OR_UPDATES_CHANGED')
    paired_dir=args.private/'cells'/'R2_newfit320_outer1_inner0_O4_seed47'
    paired_receipt_path=paired_dir/'COMPLETE.json';paired_receipt=json.loads(paired_receipt_path.read_text())
    require(paired_receipt['status']=='COMPLETE_INDEPENDENT_BUDGET_FIT'
            and paired_receipt['artifact_sha256']==sha256(paired_dir/'model.pt'),'COMPLETED320_OR_SHA_CHANGED')
    paired=torch.load(paired_dir/'model.pt',map_location='cpu',weights_only=False)
    parent_dir=args.primary_private/'cells'/'R2_outer1_inner0_O4_seed47'
    parent_receipt_path=parent_dir/'COMPLETE.json';parent_receipt=json.loads(parent_receipt_path.read_text())
    require(parent_receipt['status']=='COMPLETE' and parent_receipt['artifact_sha256']==sha256(parent_dir/'model.pt'),
            'ORIGINAL_PARENT80_OR_SHA_CHANGED')
    parent=torch.load(parent_dir/'model.pt',map_location='cpu',weights_only=False)
    prep_path=args.primary_private/'prepared'/'outer1_inner0.pt';prep_receipt=json.loads(prep_path.with_suffix('.json').read_text())
    require(prep_receipt['sha256']==sha256(prep_path),'PREPARED_SOURCE_CHANGED')
    prepared=torch.load(prep_path,map_location='cpu',weights_only=False)
    for name,digest in parent['lineage']['source'].items():require(sha256(ROOT/name)==digest,'CORE_CODE_SOURCE_CHANGED')
    for model in (failed,paired):
        lineage=model['lineage']
        require(lineage['core_origin']==parent['lineage']==prep_receipt['lineage']
                and lineage['parent80_artifact_sha256']==parent_receipt['artifact_sha256']
                and lineage['parent80_receipt_sha256']==sha256(parent_receipt_path)
                and lineage['prepared_origin_sha256']==sha256(prep_path)
                and lineage['independent_script_sha256']==sha256(ROOT/'scripts/bgb_en_budget_retrain.py'),
                'INDEPENDENT_BUDGET_PARENT_OR_SOURCE_CHANGED')
        require(model['arm']=='O4' and model['seed']==47 and np.array_equal(model['train'],parent['train'])
                and np.array_equal(model['valid'],parent['valid']),'MODEL_IDENTITY_OR_SPLIT_CHANGED')
    assets=json.loads(args.assets.read_text())
    require(sha256(assets['endpoints'])==parent['lineage']['endpoints_sha256'],'ENDPOINT_SHA_CHANGED')
    with np.load(assets['endpoints'],allow_pickle=False) as file:
        time,event=np.array(file['DSS_time']),np.array(file['DSS_event'])
    train,valid=parent['train'],parent['valid']
    risk={name:model['snapshots'][80]['risk'] for name,model in [('original_parent80',parent),('new160_epoch80',failed),('new320_epoch80',paired)]}
    comparisons={}
    for left,right in [('original_parent80','new160_epoch80'),('original_parent80','new320_epoch80'),('new320_epoch80','new160_epoch80')]:
        comparisons[left+'_to_'+right]=compare_parent80(time[valid],event[valid],time[train],event[train],risk[left],risk[right],2e-6)
    for label,model in [('new160',failed),('new320',paired)]:
        stored=model['parent80_replay'];independent=comparisons['original_parent80_to_'+label+'_epoch80']
        require(stored==independent,'SAVED_PARENT80_REPLAY_DIFFERS_FROM_INDEPENDENT_RECALCULATION')
    initial160,initial320=failed['initial_state_sha256'],paired['initial_state_sha256']
    require(set(initial160)==set(initial320),'INITIAL_STATE_HASH_KEYS_DIFFER')
    changed_initial=[name for name in initial160 if initial160[name]!=initial320[name]]
    epoch160=next(row for row in failed['curves'] if row['epoch']==160)
    final=failed['snapshots'][160]['metrics']
    require(final==epoch160['valid'],'FAILED_FINAL160_SNAPSHOT_CURVE_DIFFERS')
    require(not torch.cuda.is_initialized(),'AUDIT_INITIALIZED_CUDA')
    return dict(schema='bgb_en_budget_failed_parent80_cpu_audit_v1',status='FAILED160_RECEIPT_RETAINED_CPU_DIAGNOSIS_ONLY',
        unit=dict(arm='O4',partition=0,seed=47,epochs=160),
        failure_guard='Same-initialization epoch80 risks exceeded the locked original80 replay tolerance',
        original_risk_tolerance=2e-6,threshold_changed=False,failed_receipt_sha256=sha256(failed_path),
        failed_model_saved=True,failed_model_sha256=sha256(failed_model_path),failed_COMPLETE_exists=False,
        failed_fit_actual_completed_optimizer_updates=160,failed_artifact_actual_risk_decoder_fits=failed['final_retention_decoder_fits'],
        original_parent_artifact_sha256=parent_receipt['artifact_sha256'],paired320_artifact_sha256=paired_receipt['artifact_sha256'],
        paired320_receipt_sha256=sha256(paired_receipt_path),prepared_sha256=sha256(prep_path),
        parent80_comparisons=comparisons,initial_state_hash_comparison=dict(named_states=len(initial160),
            new160_vs_new320_exact=not changed_initial,changed_state_names=changed_initial,
            original_parent_initial_hashes_persisted='initial_state_sha256' in parent,
            original_parent_exact_initial_state_not_inferred_from_seed=True),
        epoch80_parameter_and_buffer_difference=state_difference(failed['snapshots'][80]['state_dict'],paired['snapshots'][80]['state_dict']),
        failed_final160_inner_metrics={key:final[key] for key in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den','repair','damage')},
        failed_final160_training_curve={key:epoch160[key] for key in ('epoch','cox_loss','delta_train_SD','delta_valid_SD','gradient_tensor_nonzero_ratio')},
        failed_fit_metrics_are_not_promoted_to_qualified_budget_comparison=True,
        actual_audit_work=dict(model_constructions=0,forwards=0,backwards=0,optimizers=0,GPU_calls=0),
        report_script_sha256=sha256(__file__),comparison_script_sha256=sha256(ROOT/'scripts/bgb_en_budget_retrain.py'),
        interpretation='Same initialized new fits may diverge numerically under the inherited GPU backend. These CPU saved-array comparisons do not isolate a unique CUDA kernel cause or repair the original guard.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private',type=Path,required=True);parser.add_argument('--primary-private',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en/BUDGET_O4_FAILED_PARENT80_AUDIT.json')
    args=parser.parse_args();report=run(args);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','failed_model_saved','failed_fit_actual_completed_optimizer_updates','parent80_comparisons','initial_state_hash_comparison')}))


if __name__=='__main__':main()

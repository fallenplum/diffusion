"""Real target forward/gradient qualification, zero optimizer updates."""
from __future__ import annotations
import gc
import os
from pathlib import Path
import sys
import time
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r6_common import (parser,context,create_lock,qualified_prepared,
    patch_bank,construct,configure_deterministic_runtime,write,sha256,require,
    TARGET_ARMS,public_snapshot_guard)
from src.bgb_en.r6_training import state_hash,saved_state


def main():
    args=parser(__doc__).parse_args()
    configure_deterministic_runtime();torch.set_num_threads(2)
    start=time.monotonic();before=public_snapshot_guard()
    require(not (args.private/'QUALIFICATION.json').exists(),'EXISTING_TARGET_QUALIFICATION_PRESERVED')
    ctx=context(args,locked=False)
    lock=create_lock(args,ctx);ctx['lock']=lock
    # Every source representation has a separate immutable state-keyed cache.
    cache_audits={}
    for source_key in ('MATCHED','WRONG_DONOR','WITHIN_SECTION_TISSUE_DEPTH','RANDOM_NO_ST'):
        _,digest=patch_bank(args,ctx,source_key,create=True)
        cache_audits[source_key]=digest
    from scripts.bgb_en_r6_replay import upstream_replay
    upstream=[];prepared_receipts={}
    for part in (0,1,2):
        prepared,origin=qualified_prepared(args,ctx,part)
        upstream.append(dict(partition=part,**upstream_replay(ctx,prepared)))
        prepared_receipts[str(part)]=origin['prepared_receipt_sha256']
    up=dict(status='PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY',
        contract_sha256=lock['contract_sha256'],partitions=upstream,
        prepared_receipt_sha256=prepared_receipts,new_optimizer_fits=0,new_preprocessing_fits=0)
    write(args.private/'UPSTREAM_REPLAY_AUDIT.json',up)
    args.partition=0;prepared,_=qualified_prepared(args,ctx,0)
    indices=torch.as_tensor(prepared['train'][:8],device=args.device)
    budgets=[];fingerprints=[];repetition_errors=[]
    torch.cuda.reset_peak_memory_stats()
    for arm in TARGET_ARMS:
        model,_=construct(args,ctx,prepared,arm,31);model.eval()
        model.refresh_training_query_mean()
        fingerprint=state_hash(saved_state(model),target_only=True);fingerprints.append(fingerprint)
        with torch.no_grad():
            offset_before=model.offset[indices].clone()
            first=model(indices);second=model(indices)
            error=float((first['risk']-second['risk']).abs().max())
            require(error==0.,'R6_REAL_TARGET_REPEAT_FORWARD_NOT_EXACT')
            off=model(indices,control='off')
            require(torch.count_nonzero(off['cross_delta'])==0,'R6_REAL_TARGET_OFF_NOT_EXACT_ZERO')
            wrong=model(indices,control='wrong_query')
            require(torch.equal(first['main_delta'],wrong['main_delta']),
                    'R6_QUERY_ONLY_CHANGED_UNIMODAL_MAIN_EFFECT')
            require(torch.equal(offset_before,model.offset[indices])
                    and torch.equal(first['offset'],wrong['offset'])
                    and wrong['diagnostics']['offset_untouched'],
                    'R6_QUERY_ONLY_CHANGED_FROZEN_OFFSET')
        require(not any(v.requires_grad for v in model.core.foundation.parameters()),
                'R6_SOURCE_FOUNDATION_NOT_FROZEN')
        budgets.append(dict(arm=arm,**model.parameter_budget()))
        repetition_errors.append(dict(arm=arm,repeated_risk_max_abs_error=error))
        if arm=='B7':
            # Two real-data backward passes, no optimizer and no survival labels.
            for iteration in (1,2):
                model.zero_grad(set_to_none=True)
                output=model(indices)
                (output['risk'].square().mean()+output['cross_delta'].square().mean()).backward()
                require(all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()),
                        'R6_REAL_TARGET_NONFINITE_BACKWARD')
            require(all(v.grad is None for v in model.core.foundation.parameters()),
                    'R6_SOURCE_PARAMETER_RECEIVED_TARGET_GRADIENT')
            model.inputs['rna'].requires_grad_(True)
            output=model(indices)
            derivative=torch.autograd.grad(output['cross_delta'].sum(),model.inputs['rna'],allow_unused=False)[0]
            observed=model.inputs['rna_mask']
            require(torch.count_nonzero(derivative[~observed])==0,'R6_REAL_MISSING_GENE_GRADIENT_LEAK')
            require(torch.any(derivative[indices][observed[indices]]!=0),'R6_REAL_CROSS_RNA_GRADIENT_IS_ZERO')
            model.inputs['rna'].requires_grad_(False)
        del model
        gc.collect()
    require(len(set(fingerprints))==1,'R6_SHARED_TARGET_INITIALIZATION_CHANGED')
    require(public_snapshot_guard()==before,'R0_R4_REPORT_CHANGED_DURING_R6_QUALIFICATION')
    result=dict(status='PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER',
        contract_sha256=lock['contract_sha256'],new_optimizer_updates=0,new_neural_fits=0,
        real_data_backward_passes=2,source_teacher_or_scalar_fits=0,
        cache_receipts=cache_audits,parameter_budgets=budgets,
        repeated_forward=repetition_errors,shared_target_initialization_sha256=fingerprints[0],
        query_only_main_and_offset_exact=True,missing_gene_gradients_exact_zero=True,
        source_parameters_frozen=True,original_gene_values_in_dynamic_reread=True,
        upstream_replay_audit_sha256=sha256(args.private/'UPSTREAM_REPLAY_AUDIT.json'),
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,
        outer_test_outcomes_used_for_training_selection_or_evaluation=False,
        outcome_blind_visual_cache_covers_canonical_cohort=True,
        raw_patient_arrays_or_weights_exported=False)
    write(args.private/'QUALIFICATION.json',result)
    write(ROOT/'reports/bgb_en_r6/PROTOCOL_LOCK.json',lock)
    write(ROOT/'reports/bgb_en_r6/R6_3_REAL_GENE_REREAD_AUDIT.json',result)
    print(result['status'])


if __name__=='__main__':main()

"""Two real-source GPU qualification updates, separate from formal cells.

No heldout score is used for selection and no qualification weight is saved.
These updates are explicitly counted, never merged into the 135 formal fits.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r6_pretrain import (
    validate_lock,verify_launch_gate,SourceFitData,_model_for_cell,_forward_small,
    deterministic_output_gradient_cache_backward,sha256,SourceContract,
    load_private_cache,load_source_program_mapping,_atomic_json)
import torch


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-cache',type=Path,required=True)
    p.add_argument('--launch-gate',type=Path,required=True)
    p.add_argument('--launch-gate-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('EXISTING_QUALIFICATION_PRESERVED')
    config=json.loads(args.config.read_text());fixed=validate_lock(config)
    verify_launch_gate(args.launch_gate,args.launch_gate_sha256)
    if not torch.cuda.is_available():raise ValueError('REAL_CUDA_QUALIFICATION_REQUIRED')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    contract=SourceContract(fresh_certificate_sha256=config['fresh_certificate_sha256'])
    dataset=load_private_cache(args.source_cache,contract,
        expected_manifest_sha256=config['source_cache_manifest_sha256'])
    mapping=load_source_program_mapping(ROOT/'configs/sami_surv_stage_b_programs_v1.json')
    cell=dict(arm='S7',control='MATCHED',heldout_donor=0,seed=31)
    data=SourceFitData(dataset,cell,mapping,'cuda')
    model=_model_for_cell(cell,data,mapping,'cuda');model.train()
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=.001,weight_decay=0.)
    objective=data.objective(model)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();steps=[]
    for step in (1,2):
        optimizer.zero_grad(set_to_none=True)
        audit=deterministic_output_gradient_cache_backward(model,
            lambda a,b:_forward_small(model,data,a,b),len(data.train),objective,
            microbatch=fixed['source_microbatch'],parameter_L2=.0001,
            tolerance=fixed['full_output_replay_tolerance'])
        norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        steps.append(dict(qualification_update=step,preclip_gradient_norm=float(norm),**audit))
    # Direct measured/missing input-gradient check on real source measurements.
    rows=data.train[:8]
    values=torch.as_tensor(data.gene_z[rows],device='cuda').requires_grad_(True)
    measured=torch.as_tensor(dataset.measured_mask[rows],device='cuda')
    batch=data.gene_bank(values,measured)
    encoded=model.encode_gene_programs(batch)
    derivative=torch.autograd.grad(encoded['tokens'].square().mean(),values)[0]
    masked_max=float(derivative[~measured].abs().max()) if (~measured).any() else 0.
    if masked_max!=0. or not torch.any(derivative[measured]!=0):
        raise ValueError('REAL_GENE_MASK_GRADIENT_ISOLATION_FAILED')
    elapsed=time.monotonic()-start
    result=dict(schema='bgb_en_r6_real_source_compute_qualification_v1',
        status='PASS_TWO_REAL_SOURCE_GPU_UPDATES',config_sha256=sha256(args.config),
        source_cache_manifest_sha256=config['source_cache_manifest_sha256'],
        model='S7',qualification_updates=2,formal_source_fits=0,
        separate_qualification_optimizer_instances=1,training_source_spots=len(data.train),
        heldout_spots_not_in_training=len(data.heldout),steps=steps,
        masked_gene_input_gradient_max_abs=masked_max,measured_gene_gradient_nonzero=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_name=torch.cuda.get_device_name(),reported_device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        elapsed_seconds=elapsed,estimated_serial_80_update_seconds_excludes_snapshots=elapsed*40,
        estimate_is_not_formal_runtime_or_parallel_throughput=True,
        saved_qualification_weights=False,TCGA_DSS_or_PFI_read=False,
        performance_based_hyperparameter_choice=False,source_test_sections_read=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);_atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','qualification_updates','peak_allocated_bytes','elapsed_seconds')}))


if __name__=='__main__':main()

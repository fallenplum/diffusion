"""Fit-only reference refresh and real heldout-label poisoning qualification.

Two two-epoch runs of all four arms remain in memory. Only an aggregate
receipt is persisted. No smoke checkpoint, prediction array or model is saved.
"""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.phase23_cdei.data import Data
from src.phase24_cdei_g1.runtime import PRIVATE,source,atomic_json
from src.phase24_cdei_g1.train import run_cell
# The ancestor utility changes only in-memory endpoint-loader output.
from scripts.qualify_phase23_cdei_real import (tree_hash,poison_nonfit_dss_endpoints,
                                              install_label_barriers)


def numbers(payload):
    keep={key:payload[key] for key in ('fit','valid','anchor_fit','anchor_valid','J_CS_valid')}
    keep['arms']={}
    for arm,row in payload['arms'].items():
        keep['arms'][arm]={k:row[k] for k in ('risk','fit_risk','model_state')}
        keep['arms'][arm]['history']=[{k:v for k,v in x.items() if k!='epoch_seconds'} for x in row['history']]
        if 'sensitivities' in row:keep['arms'][arm]['sensitivities']=row['sensitivities']
    return keep


def main():
    a=argparse.ArgumentParser();a.add_argument('--device',default='cuda');args=a.parse_args()
    if not torch.cuda.is_available():raise ValueError('server CUDA required')
    torch.set_num_threads(2);s=source();out=PRIVATE/'REAL_DATA_QUALIFICATION.json'
    if out.exists():raise ValueError('prior qualification requires explicit review; never silently rerun')
    report={'status':'RUNNING','source':s,'formal_fits':0,'short_models_completed':0,
            'kernel_updates':0,'persistent_weights_arrays':False,'runs':[]}
    atomic_json(out,report);started=time.monotonic()
    try:
        # Prior fold-local one-hot dimensions are legal. Audit every role before
        # any fit, without using any selection endpoint for model decisions.
        shapes=[]
        for role in ('inner','outer'):
            for fold in range(1,6):
                ds=Data(fold,role,31,device='cpu')
                support=[v['omics']['token_support'] for v in ds.audit['wrong_controls'].values()]
                if any(x['shared_valid_tokens_patient_min']!=24 or x['support_changed_patients'] for x in support):
                    raise ValueError('locked prototype requires all24 matching token slots; no silent change of support')
                shapes.append({'role':role,'fold':fold,'clinical_dim':int(ds.ft['C'].shape[1]),'common_slots':24})
                del ds
        report['all10_shape_and_anchor_audits']=shapes;atomic_json(out,report)
        first=None;original_fit=None;labelhash=None
        for index in range(2):
            if index==0:
                ds=Data(1,'inner',31,device=args.device);original_fit=np.array(ds.fit);n=len(ds.ft['C'])
                labelhash=tree_hash(ds.selection_labels())
            else:
                with poison_nonfit_dss_endpoints(original_fit,n) as poison:
                    ds=Data(1,'inner',31,device=args.device)
                if poison['loader_calls']<1 or tree_hash(ds.selection_labels())==labelhash:
                    raise ValueError('real heldout endpoint poison did not take effect')
            barrier,_=install_label_barriers(ds)
            torch.cuda.reset_peak_memory_stats();run_start=time.monotonic()
            def step(arm,epoch):
                report['kernel_updates']+=1;atomic_json(out,report)
            result=run_cell(ds,s,None,epochs=2,step_callback=step)
            statehash=tree_hash(numbers(result))
            initials={r['training_audit']['initial_backbone_sha256'] for r in result['arms'].values()}
            if len(initials)!=1:raise ValueError('four arms have different backbone initializations')
            local=result['arms']['LOCAL'];ctrl=result['arms']['CTRL']
            if local['parameter_counts']!=ctrl['parameter_counts']:raise ValueError('LOCAL CTRL parameter mismatch')
            if first is None:first=statehash
            elif statehash!=first:raise ValueError('heldout label poisoning changed joint training state or predictions')
            if barrier['selection_label_calls']:raise ValueError('joint training accessed selection labels')
            report['short_models_completed']+=4
            report['runs'].append({'run':index+1,'poisoned_nonfit_DSS':bool(index),'state_sha256':statehash,
                'kernel_updates':8,'short_models':4,'selection_label_calls_in_training':0,
                'peak_GPU_reserved_bytes':int(torch.cuda.max_memory_reserved()),
                'wall_seconds':time.monotonic()-run_start,'LOCAL_parameter_count':local['parameter_counts']['total'],
                'LOCAL_CTRL_parameters_equal':True,'all_arms_same_backbone_initialization':True})
            atomic_json(out,report);del result,ds;gc.collect();torch.cuda.empty_cache()
        if source()!=s:raise ValueError('source drift during qualification')
        report.update(status='PASS',wall_seconds=time.monotonic()-started,
                      heldout_poison_all_numeric_state_exact=True,qualification_weights_or_arrays_saved=0)
    except BaseException as error:
        report.update(status='FAILED',error_type=type(error).__name__);raise
    finally:atomic_json(out,report)
    print('G1_REAL_DATA_QUALIFICATION',report['status'],s,flush=True)


if __name__=='__main__':main()

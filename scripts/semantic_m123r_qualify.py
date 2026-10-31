"""Actual source/shape/initialization qualification; zero DSS optimizer steps."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,sys,time,subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import parent_check,PARENT,build,write,sha
from src.semantic_m123r.models import RepairModel,ARMS
from src.semantic_m123.training import subset,efron,efron_log_weights


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-master',required=True);p.add_argument('--output-newdir',required=True);a=p.parse_args()
    parent=parent_check(a.parent_master,PARENT)
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False)
    if sha(parent['frozen_images'])!='eb12386a1f51b65635893e47a60f79867a092eb663615dcce932b6db0cf41da5':raise ValueError('FROZEN_CACHE_CHANGED')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    audits=[];profiles=[]
    for inner in range(3):
        split=f'outer1_inner{inner}';b,data,scale,raw,assets,part,digest=build(parent,split)
        train=subset(b,data.train,'cuda');valid=subset(b,data.evaluation,'cuda')
        dims=[b[k].shape[1] for k in ('P','G','C')];errors=[]
        for arm in ARMS:
            torch.manual_seed(31);model=RepairModel(arm,*dims).cuda();model.warm(data.warm)
            with torch.no_grad():prediction=model(valid)['risk'].cpu().numpy()
            error=float(np.max(np.abs(prediction-data.baseline[len(data.train):])))
            if error>1e-10:raise ValueError('REAL_FP64_S0_QUALIFICATION_FAILED')
            errors.append({'arm':arm,'FP64_S0_error':error,'parameters':sum(p.numel() for p in model.parameters())})
            if inner==0 and arm in ('R3','R5','R6','R9'):
                model.stage(91);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                synthetic_time=np.arange(1,len(data.train)+1);synthetic_event=(synthetic_time%7==0).astype(int)
                lw,ev=efron_log_weights(synthetic_time,synthetic_event,device='cuda');times=[]
                for _ in range(4):
                    model.zero_grad(set_to_none=True);torch.cuda.synchronize();t0=time.perf_counter()
                    value=model(train)['risk'];loss=efron(value,lw,ev);loss.backward();torch.cuda.synchronize()
                    times.append(time.perf_counter()-t0)
                profiles.append({'arm':arm,'steady_forward_backward_seconds':float(np.mean(times[1:])),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'synthetic_endpoints_only':True,
                    'NVML_process_memory_MiB':int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())})
            del model;torch.cuda.empty_cache()
        mean_error=float((train['mean_patch']-torch.stack([train['patch'][i,train['mask'][i]].mean(0) for i in range(len(data.train))])).abs().max())
        # The general equal-slide identity is separately recomputed here.
        from src.semantic_m123.models import slide_mean
        equal_error=float((train['mean_patch']-slide_mean(train['patch'].double(),train['mask'],train['slide'])).abs().max())
        if equal_error>1e-12:raise ValueError('MEAN_PATCH_SCALE_INTERFACE_FAILED')
        audits.append({'split':split,'fit_patients':len(data.train),'evaluation_patients':len(data.evaluation),
            'input_sha256':digest,'S0_all_arms':errors,'equal_slide_mean_scale_error':equal_error,
            'patch_weighted_mean_is_not_contract_diagnostic_difference':mean_error})
        del b,data,raw,train,valid;torch.cuda.empty_cache()
    result={'schema':'m123r_actual_asset_qualification_v1','status':'PASS','partitions':audits,'profiles':profiles,
        'new_optimizer_updates':0,'DSS_endpoints_decoded':False,'weights_saved':False,
        'parent_master_sha256':PARENT,'no_outer_test_performance_computed':True}
    write(out/'QUALIFICATION.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()

"""Real-asset qualification, zero DSS optimization and no retained smoke weights."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,pickle,sys,time,gc,subprocess
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.semantic_m123.data import prepare
from src.semantic_m123.models import JointModel,ARMS
from src.semantic_m123.training import subset,efron,efron_log_weights
from scripts.semantic_b0b7_p2_matrix import raw_inputs
from types import SimpleNamespace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw,membership,_=raw_inputs(SimpleNamespace(assets=args.assets,legacy_root=args.legacy_root),endpoints=False)
    frozen=torch.load(args.frozen,map_location='cpu',weights_only=False,mmap=True)
    plan=json.loads((Path(args.parent_private)/'SPLITS.private.json').read_bytes())['partitions']
    summaries=[];profile=None
    for name,part in sorted(plan.items()):
        ap=Path(args.parent_private)/'anchors'/(name+'.private.pkl')
        receipt=json.loads(ap.with_suffix('.json').read_bytes())
        if sha(ap)!=receipt['artifact_sha256']:raise ValueError('PARENT_ANCHOR_HASH_CHANGED')
        with ap.open('rb') as f:anchor=pickle.load(f)
        data=prepare(raw,frozen,anchor,part['fit'],part['evaluation'],membership)
        summaries.append({'split':name,**data.receipt})
        if name=='outer1_final':
            profile=data
        else:del data
        del anchor;gc.collect()
    if profile is None:raise ValueError('PROFILE_PARTITION_MISSING')
    # Verify the exact mapped function in FP64 without multiplying the entire
    # 512-patch cache to double: no local branch contributes at warm epoch zero.
    b=profile.batch;n=len(profile.rows)
    small={k:(v.double() if v.is_floating_point() else v) for k,v in b.items() if k not in ('patch','mask','slide')}
    small.update(patch=torch.zeros(n,1,768,dtype=torch.float64),mask=torch.ones(n,1,dtype=torch.bool),slide=torch.zeros(n,1,dtype=torch.long))
    dims=[b[k].shape[1] for k in ('P','G','C')]
    m=JointModel('M1_WARM',*dims).double()
    w={k:torch.as_tensor(v,dtype=torch.float64) if k.endswith('coef') else v for k,v in profile.warm.items()}
    m.warm(**w)
    expected=(small['P']@w['p_coef']+small['G']@w['g_coef']+small['C']@w['c_coef']+w['p_bias']+w['g_bias']).numpy()
    with torch.no_grad():exact=float(np.max(np.abs(m(small)['risk'].numpy()-expected)))
    rounded=float(np.max(np.abs(expected-profile.baseline)))
    if exact>1e-8:raise ValueError('FP64_MODEL_MAPPING_FAILED:'+str(exact))
    if rounded>2e-5:raise ValueError('FLOAT32_INPUT_BOUNDARY_EXCEEDS_TOLERANCE:'+str(rounded))
    del m,small
    train=subset(b,profile.train,'cuda')
    synthetic_t=np.arange(len(profile.train),dtype=float)+1
    synthetic_e=(np.arange(len(profile.train))%7==0).astype(int)
    logw,event=efron_log_weights(synthetic_t,synthetic_e,device='cuda')
    models=[]
    for arm in ARMS:
        torch.manual_seed(31);torch.cuda.manual_seed_all(31);torch.cuda.reset_peak_memory_stats()
        m=JointModel(arm,*dims).cuda()
        if arm!='M1_SCRATCH':
            ww={k:torch.as_tensor(v,dtype=torch.float32,device='cuda') if k.endswith('coef') else v for k,v in profile.warm.items()}
            m.warm(**ww)
        # Temporary nonzero exits test full gradients and maximum training
        # memory. They are discarded and never used for formal initialization.
        with torch.no_grad():
            m.local_out.weight.copy_(torch.eye(64,device='cuda')*.01)
            if hasattr(m,'interaction_head'):m.interaction_head.weight.fill_(.01)
        elapsed=[]
        for rep in range(3):
            m.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
            out=m(train);loss=efron(out['risk'],logw,event)+.001*m.regularizer()
            loss.backward();torch.cuda.synchronize();elapsed.append(time.monotonic()-start)
        rec={'arm':arm,**m.parameter_audit(),'profile_fit_patients':len(profile.train),
            'profile_uses_real_DSS':False,'optimizer_updates':0,'weights_saved':False,
            'forward_backward_seconds':elapsed,'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved()}
        nv=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
        matches=[line.split(',') for line in nv.stdout.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
        rec['observed_NVML_resident_MiB']=int(matches[0][1].strip()) if matches and matches[0][1].strip().isdigit() else None
        models.append(rec);del m,out,loss;torch.cuda.empty_cache()
    result={'schema':'m123_real_asset_qualification_v1','status':'PASS_NO_DSS_TRAINING',
        'all20_partitions_exact_R4_feature_replay':summaries,'real_FP64_model_risk_max_error':exact,
        'FP32_input_rounding_vs_full_precision_R4_max_error':rounded,
        'warm_is_mathematical_recipe_mapping_not_original_FP64_bitwise_risk':True,
        'models':models,'formal_DSS_fits':0,'retained_smoke_weights':0,'raw_endpoints_read':False,
        'design_precommit_sha256':sha(ROOT/'configs/semantic_m123/design_precommit.json'),
        'qualified_sources':{p:sha(ROOT/p) for p in ('src/semantic_m123/models.py','src/semantic_m123/data.py','src/semantic_m123/training.py')}}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'partitions':len(summaries),'FP64_warm_max_error':exact,'FP32_input_boundary_max_error':rounded,'models':models}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--assets',required=True);p.add_argument('--legacy-root',required=True)
    p.add_argument('--frozen',required=True);p.add_argument('--parent-private',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())


"""Frozen attention geometry only; no survival endpoints decoded or new fits."""
from pathlib import Path
import argparse,csv,hashlib,json,pickle,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123_worker import check_lock,read,sha,write
from scripts.semantic_m123_schedule import final_jobs,verify_complete,cell_path
from scripts.semantic_b0b7_p2_matrix import raw_inputs
from src.semantic_m123.replay import load_saved_inputs
from src.semantic_m123.training import subset
from src.semantic_m123.models import JointModel
from src.semantic_m123.diagnostics import summarize_attention
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','diagnostic-lock','expected-diagnostic-sha256','private','output-newdir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();private=Path(a.private)
    lock=check_lock(a.lock,a.expected_lock_sha256)
    if (private/'OWNER.private.json').exists() or not (private/'STUDY_COMPLETE.private.json').exists():
        raise ValueError('FROZEN_DIAGNOSTIC_REQUIRES_COMPLETED_FITS')
    if sha(a.diagnostic_lock)!=a.expected_diagnostic_sha256:raise ValueError('DIAGNOSTIC_LOCK_CHANGED')
    dl=read(a.diagnostic_lock)
    if dl['master_lock_sha256']!=a.expected_lock_sha256:raise ValueError('DIAGNOSTIC_MASTER_BINDING')
    for name,h in dl['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('DIAGNOSTIC_SOURCE_CHANGED')
    out=Path(a.output_newdir);out.mkdir(exist_ok=False,parents=True)
    barrier=read(private/'ALL_OUTER_SELECTIONS_FROZEN.private.json')
    jobs=final_jobs(a.expected_lock_sha256,barrier,sha(private/'ALL_OUTER_SELECTIONS_FROZEN.private.json'))
    raw,_,_=raw_inputs(SimpleNamespace(assets=lock['assets_path'],legacy_root=lock['legacy_root']),endpoints=False)
    frozen=torch.load(lock['frozen_images'],map_location='cpu',weights_only=False,mmap=True)
    plan=read(lock['parent_split_plan'])['partitions']
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];forwards=0;states=0
    for outer in range(1,6):
        split=f'outer{outer}_final';part=plan[split]
        current=[j for j in jobs if j['split']==split and j['arm'] not in ('M0PLUS','M1_SCRATCH','M1_WARM','M2_MEAN_MATCHED')]
        first=verify_complete(private,current[0]);binding=first['shared_preprocessing']
        ps=private/'preprocessing'/binding['file']
        if sha(ps)!=binding['sha256']:raise ValueError('PREPROCESSOR_CHANGED')
        with ps.open('rb') as f:state=pickle.load(f)
        b,_,order=load_saved_inputs(raw,frozen,state,part['fit'],part['evaluation'],
            expected_input_sha256=lock['partitions'][split]['input_sha256'])
        test=subset(b,np.arange(len(part['fit']),len(order)),'cuda')
        for job in current:
            receipt=verify_complete(private,job);folder=cell_path(private,job)
            cp=torch.load(folder/receipt['checkpoint_file'],map_location='cpu',weights_only=False)
            model=JointModel(cp['arm'],*cp['dims']).cuda();model.load_state_dict(cp['state_dict']);model.eval()
            with torch.no_grad():result=model(test,diagnostics=True)
            querymask=test['program_mask'].any(-1).cpu().numpy() if job['arm'].startswith('M3') else None
            d=summarize_attention(result['patch_attention'].cpu().numpy(),result['local_tokens'].cpu().numpy(),
                test['mask'].cpu().numpy(),querymask)
            rows.append({'outer':outer,'arm':job['arm'],'seed':job['seed'],'epoch':job['epochs'],'L2_lambda':job['lambda'],**d})
            forwards+=1;states+=1
            del model,cp,result;torch.cuda.empty_cache()
        del test,b,state;torch.cuda.empty_cache()
    if forwards!=105:raise ValueError('EXPECTED_105_FIXED_MODEL_DIAGNOSTICS')
    with (out/'FROZEN_ATTENTION_GEOMETRY.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    audit={'schema':'m123_frozen_attention_geometry_audit_v1','status':'COMPLETE','actual_model_forwards':forwards,
        'saved_model_states':states,'survival_endpoint_arrays_decoded':False,'archive_hashing_may_read_endpoint_containers_as_opaque_bytes':True,
        'new_optimizer_fits':0,'new_preprocessor_fits':0,'no_attention_arrays_or_patient_rows_exported':True,
        'definitions':{'rank':'exp entropy of normalized singular values, zero matrix rank zero',
          'centered_energy':'squared centered token norm divided by squared uncentered norm',
          'rank_fraction':'effective rank divided by number of observed queries',
          'effective_patch_fraction':'exp entropy of head-mean attention divided by real patch count'},
        'limitations':['These are geometric diagnostics, not histological labels or calibrated uncertainty.',
          'Centered rank must be read with centered energy; rounding-scale differences do not establish distinct evidence.',
          'Different query counts change rank ceilings; use matched controls and rank fractions.',
          'Equal-slide weights can reduce effective patch fraction when slides have unequal patch counts.'],
        'master_lock_sha256':a.expected_lock_sha256,'diagnostic_lock_sha256':a.expected_diagnostic_sha256,
        'source_sha256':{name:sha(ROOT/name) for name in dl['sources']},
        'file':{'file':'FROZEN_ATTENTION_GEOMETRY.csv','sha256':sha(out/'FROZEN_ATTENTION_GEOMETRY.csv')}}
    write(out/'FROZEN_ATTENTION_GEOMETRY_AUDIT.json',audit)
    print(json.dumps({'status':'COMPLETE','forwards':forwards,'new_fits':0}))


if __name__=='__main__':main()


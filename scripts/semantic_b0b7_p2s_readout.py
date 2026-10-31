"""Produce fixed inner-only frozen features and run the finite P2S Cox probes.

Neural optimization must have finished. One frozen feature extraction per
registered state feeds at most two independent CPU workers; no outer risk or
outer labels are scored, and no configuration is selected from the results.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

STARTUP_THREADS={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
STUDY_SHA="3babf37b356e7d27f3cb5da0985b94d0b8d983edadc333eb8333b627539b5c4b"
PARENT_SHA="b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
EXECUTION_SHA="bc2df57fc7593acc92d652978fe288295c3131e3bfbedeca2cbeed82b92017a0"
SCRIPT="scripts/semantic_b0b7_p2s_readout.py"
WORKER="scripts/semantic_b0b7_p2s_risk_worker.py"
GIB=1024**3


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def meta_key(meta):
    require(re.fullmatch(r'outer1_inner[0-2]',meta.get('split','')) is not None and
        meta.get('family') in ('MAIN','ATTN_MATCHED') and meta.get('seed') in (31,47,83) and
        meta.get('lambda_pair') in (.1,.03) and meta.get('epoch') in (400,800),'P2S_READOUT_FIXED_SCOPE')
    return (meta['split'],meta['family'],meta['seed'],float(meta['lambda_pair']),meta['epoch'])


def job_name(meta):
    split,family,seed,pair,epoch=meta_key(meta)
    return f'{split}_{family}_seed{seed}_pair{round(pair*100):03d}_epoch{epoch}'


def verify_child(lock,study,expected_study_sha,*,root=ROOT):
    require(expected_study_sha==STUDY_SHA and lock.get('schema')=='semantic_b0b7_p2s_readout_lock_v1' and
        lock.get('status')=='LOCKED_BEFORE_P2S_DSS_FITS' and
        lock.get('parent_study_lock_sha256')==STUDY_SHA,'P2S_READOUT_LOCK_NOT_PRELOCKED')
    require(lock.get('expected_feature_states')==72 and lock.get('expected_Cox_fits')==2376 and
        lock.get('primary_ridge_lambda')==1. and lock.get('ridge_candidates')==[.1,1.,10.] and
        lock.get('outer_evaluation') is False,'P2S_READOUT_GRID_CHANGED')
    sources=lock.get('readout_sources',{})
    require({SCRIPT,WORKER,'tests/test_semantic_b0b7_p2s_readout.py',
        'tests/test_semantic_b0b7_p2s_risk_worker.py','src/semantic_b0b7/p2s_survival.py',
        'tests/test_semantic_b0b7_p2s_survival.py'}.issubset(sources),'P2S_READOUT_SOURCE_CLOSURE')
    for name,digest in {**study['frozen_sources'],**sources}.items():
        p=(root/name).resolve()
        require(p.is_relative_to(root.resolve()) and p.is_file() and sha(p)==digest,'P2S_READOUT_SOURCE_CHANGED:'+name)
    require(set(lock.get('anchors',{}))=={'outer1_inner0','outer1_inner1','outer1_inner2'},'P2S_THREE_OLD_INNER_ANCHORS_ONLY')


def verify_state_set(states):
    expected=set(itertools.product(('outer1_inner0','outer1_inner1','outer1_inner2'),
        ('MAIN','ATTN_MATCHED'),(31,47,83),(.1,.03),(400,800)))
    keys=[meta_key(row) for row in states]
    require(len(keys)==72 and set(keys)==expected,'P2S_ALL72_STATES_EXACT_ONCE')


def verify_population_order(global_rows,expected_ordered_rows):
    import numpy as np
    a=np.asarray(global_rows);b=np.asarray(expected_ordered_rows)
    require(a.shape==b.shape==(409,) and len(np.unique(a))==409 and np.array_equal(a,b),
        'P2S_FIT_THEN_EVALUATION_ROW_ORDER_CHANGED')


def terminal_worker_status(status):
    return status in ('COMPLETE','COMPLETE_WITH_NONPRIMARY_RANK_NA','PARTIAL_PRIMARY_UNSUPPORTED_RANK')


def clinical_order(clinical_raw):
    from src.semantic_b0b7.data import CLINICAL_FIELDS
    require(set(clinical_raw)==set(CLINICAL_FIELDS),'P2S_SIX_RAW_CLINICAL_FIELDS')
    return list(CLINICAL_FIELDS)


def collect_states(args,study):
    from scripts import semantic_b0b7_p2s_schedule as schedule
    require(not (Path(args.diagnostic_private)/'ACTIVE_SCHEDULER_OWNER.private.json').exists(),
        'P2S_NEURAL_OWNER_STILL_ACTIVE')
    helper=SimpleNamespace(lock=args.study_lock,expected_lock_sha256=STUDY_SHA,
        private=args.formal_private,output_root=args.diagnostic_private)
    states=[];counts=[]
    for cell in study['cells']:
        checked=schedule.checked_complete(helper,cell)
        require(checked is not None,'P2S_ALL_NEURAL_CELLS_REQUIRED_BEFORE_READOUT')
        counts.append(checked)
        folder=schedule.cell_path(args.diagnostic_private,cell);rp=folder/'cell.private.json';receipt=read(rp)
        for row in receipt['states']:
            p=(folder/row['file']).resolve()
            states.append({**{k:cell[k] for k in ('split','family','seed','lambda_pair')},
                'epoch':row['epoch'],'input_sha256':cell['input_sha256'],'checkpoint_sha256':row['sha256'],
                'checkpoint_file':str(p),'neural_receipt_file':str(rp),'neural_receipt_sha256':sha(rp),
                'reused_formal400':cell['lambda_pair']==.1 and row['epoch']==400})
    verify_state_set(states)
    require(sum(x['optimizer_steps'] for x in counts)==21600 and sum(x['actual_replays'] for x in counts)==72 and
        sum(x['new_snapshots'] for x in counts)==54,'P2S_NEURAL_COMPUTE_LEDGER_CHANGED')
    return sorted(states,key=meta_key)


def disk_guard(root,cache):
    require(shutil.disk_usage(root).free>2*GIB and shutil.disk_usage(cache).free>2*GIB,
        'P2S_READOUT_DISK_RESERVE_BEFORE_NEW_WRITE')
    size=sum(p.stat().st_size for p in Path(cache).rglob('*') if p.is_file())
    require(size<=800_000_000,'P2S_LOCKED_FEATURE_CACHE_CEILING')


def prepare_endpoints(args,master,states,out):
    import numpy as np
    from scripts import semantic_b0b7_p2_matrix as matrix
    from src.semantic_b0b7.p2_data import read_dss_only
    raw,_,_=matrix.raw_inputs(args,endpoints=False)
    require(sha(read(args.assets)['endpoints'])==master['assets']['endpoints_sha256'],'P2S_DSS_SOURCE_CHANGED')
    endpoints,audit=read_dss_only(read(args.assets)['endpoints'],raw['patient_ids'])
    plan=read(Path(args.private)/'SPLITS.private.json')
    populations=[]
    for split in ('outer1_inner0','outer1_inner1','outer1_inner2'):
        entry=plan['partitions'][split]
        populations.append(np.r_[entry['fit'],entry['evaluation']])
    require(all(np.array_equal(np.sort(populations[0]),np.sort(p)) for p in populations) and len(populations[0])==409,
        'P2S_ONLY_OUTER1_TRAINING409_ENDPOINT_SCOPE')
    scopes={};orders={}
    for split,rows in zip(('outer1_inner0','outer1_inner1','outer1_inner2'),populations):
        path=out/(split+'_READOUT_ENDPOINT_SCOPE.private.npz')
        with path.open('xb') as f:np.savez_compressed(f,global_rows=rows,t=endpoints['t'][rows],e=endpoints['e'][rows])
        scopes[split]={'path':str(path),'sha256':sha(path)};orders[split]=rows
    return scopes,orders


def extract_one(state,data,cache):
    import numpy as np
    import torch
    from scripts.semantic_b0b7_p1r import load_checkpoint_model,extract_frozen_features
    require(sha(state['checkpoint_file'])==state['checkpoint_sha256'],'P2S_NEURAL_STATE_CHANGED_BEFORE_EXTRACTION')
    model,_=load_checkpoint_model(state['checkpoint_file'],device='cuda',freeze=True,
        expected_input_sha256=state['input_sha256'])
    started=time.monotonic()
    outputs=extract_frozen_features(model,data,torch.arange(len(data['global_rows'])),batch_size=32)
    arrays={'P':outputs['pre_residual_visual_tokens'],'G':outputs['pre_residual_gene_tokens'],
        'C':outputs['additive_clinical'],'mask':outputs['gene_mask'],'M2':outputs['interaction_features'],
        'u':model.u.weight.detach().cpu().numpy(),'v':model.v.weight.detach().cpu().numpy(),
        'prior':model.relation_prior.detach().cpu().numpy(),'fit_local':data['train_indices'].numpy(),
        'eval_local':data['valid_indices'].numpy(),'global_rows':data['global_rows'].numpy(),
        'fit_global':data['global_rows'][data['train_indices']].numpy(),
        'eval_global':data['global_rows'][data['valid_indices']].numpy()}
    keys=clinical_order(data['clinical_raw'])
    arrays.update({f'clinical_{i}':np.asarray(data['clinical_raw'][k]).astype(str) for i,k in enumerate(keys)})
    p=cache/(job_name(state)+'.private.npz')
    require(not p.exists(),'P2S_FEATURE_CACHE_MUST_BE_NEW')
    with p.open('xb') as f:np.savez_compressed(f,**arrays)
    model.cpu();del model,outputs
    torch.cuda.empty_cache()
    return {'path':str(p),'sha256':sha(p)},keys,{'frozen_encoder_feature_extractions':1,
        'branch_batches_per_modality':int(np.ceil(len(arrays['global_rows'])/32)),
        'elapsed_seconds':time.monotonic()-started,'new_neural_optimizer_steps':0,
        'stored_feature_bytes':p.stat().st_size}


def run(args):
    require(args.allow_completed_neural_training,'P2S_READOUT_EXPLICIT_POSTTRAINING_AUTHORIZATION')
    require(STARTUP_THREADS=={k:'2' for k in STARTUP_THREADS},'P2S_READOUT_STARTUP_BLAS2_REQUIRED')
    require(sha(args.study_lock)==STUDY_SHA and sha(args.readout_lock)==args.expected_readout_lock_sha256,
        'P2S_READOUT_LOCK_BYTES_CHANGED')
    study=read(args.study_lock);child=read(args.readout_lock);verify_child(child,study,STUDY_SHA)
    require(args.max_cpu_workers in (1,2),'P2S_CPU_WORKERS_LOCKED_UPPER2')
    require(sha(args.parent_execution_lock)==EXECUTION_SHA,'P2S_PARENT_EXECUTION_SOURCE_LOCK_CHANGED')
    for n,d in read(args.parent_execution_lock)['execution_sources'].items():require(sha(ROOT/n)==d,'P2S_PARENT_SCHEDULER_SOURCE_CHANGED')
    states=collect_states(args,study)
    out=Path(args.output_newdir);cache=Path(args.feature_cache_newdir)
    require(not out.exists() and not cache.exists(),'P2S_READOUT_OUTPUTS_MUST_BE_NEW_NO_PARTIAL_RESTART')
    out.mkdir(parents=True);cache.mkdir(parents=True)
    (out/'jobs').mkdir();(out/'logs').mkdir();(out/'results').mkdir()
    disk_guard(out,cache)
    registry=out/'ALL_STATES_FROZEN_FOR_READOUT.private.json'
    write(registry,{'schema':'semantic_p2s_completed_states_v1','study_lock_sha256':STUDY_SHA,
        'readout_lock_sha256':args.expected_readout_lock_sha256,'states':states})
    from scripts import semantic_b0b7_p2_matrix as matrix
    from scripts.semantic_b0b7_p1r import legacy
    import numpy as np
    import torch
    legacy.configure_runtime();torch.set_num_threads(2)
    build_args=SimpleNamespace(lock=args.p2_lock,expected_lock_sha256=PARENT_SHA,private=args.formal_private,
        assets=args.assets,legacy_root=args.legacy_root,frozen_images=args.frozen_images,
        split='outer1_inner0',family='MAIN',seed=31,threads=2)
    master=matrix.verify_lock(build_args)
    endpoints,population=prepare_endpoints(build_args,master,states,out)
    binding=out/'READOUT_SESSION_BINDING.private.json'
    write(binding,{'schema':'semantic_p2s_readout_session_v1','study_lock_sha256':STUDY_SHA,
        'readout_lock_sha256':args.expected_readout_lock_sha256,'state_registry_sha256':sha(registry),
        'endpoint_scope_sha256':{k:v['sha256'] for k,v in endpoints.items()},'full_outer_endpoint_scoring':False,
        'canonical_reader_decodes512_DSS_values_before_selecting_training409':True,
        'outer103_DSS_values_used_for_fit_or_scoring':False,
        'state_registry_frozen_before_first_new_DSS_fit':True})
    active=[];completed=[];failures=[];data=None;current_split=None
    def collect(block=False):
        while True:
            for entry in list(active):
                rc=entry['process'].poll()
                if rc is None:continue
                entry['log'].close();active.remove(entry)
                if rc:
                    failures.append({'job':entry['name'],'exit_code':rc});continue
                rp=entry['output']/'READOUT.private.json'
                receipt=read(rp)
                require(terminal_worker_status(receipt.get('status')),'P2S_WORKER_NOT_COMPLETE')
                completed.append({**entry['meta'],'receipt_file':str(rp.relative_to(out)),'receipt_sha256':sha(rp)})
            if failures or not block or len(active)<args.max_cpu_workers:return
            time.sleep(1)
    try:
        for i,state in enumerate(states):
            collect(block=True)
            require(not failures,'P2S_CPU_WORKER_FAILURE_STOPS_NEW_ADMISSION')
            disk_guard(out,cache)
            if current_split!=state['split']:
                del data;current_split=state['split'];build_args.split=current_split
                data,_,split,input_sha=matrix.build_inputs(build_args,master)
                verify_population_order(data['global_rows'].numpy(),population[current_split])
            require(matrix.training_input_binding(data,master['assets']['frozen_images_sha256'])==state['input_sha256'],
                'P2S_FROZEN_PARTITION_INPUT_CHANGED')
            feature,clinical_keys,compute=extract_one(state,data,cache)
            tr=data['train_indices'].numpy();ev=data['valid_indices'].numpy();gr=data['global_rows'].numpy()
            # Append exact global row scopes without duplicating raw image tensors.
            # The initial cache already binds local indices and canonical registry.
            anchor=child['anchors'][state['split']]
            ap=Path(args.formal_private)/anchor['file'];ar=ap.with_suffix('.json')
            require(sha(ap)==anchor['sha256'] and sha(ar)==anchor['receipt_sha256'],'P2S_SHARED_INNER_ANCHOR_CHANGED')
            meta={k:state[k] for k in ('split','family','seed','lambda_pair','epoch','input_sha256')}
            meta['checkpoint_sha256']=state['checkpoint_sha256']
            name=job_name(state);jp=out/'jobs'/(name+'.private.json');dest=out/'results'/name
            payload={**meta,'feature_cache':feature,'clinical_keys':clinical_keys,'endpoint_scope':endpoints[current_split],
                'anchor':{'path':str(ap),'sha256':sha(ap),'receipt_path':str(ar),'receipt_sha256':sha(ar)},
                'state_registry':{'path':str(registry),'sha256':sha(registry)},
                'session_binding':{'path':str(binding),'sha256':sha(binding)},
                'legacy_root':args.legacy_root,'feature_extraction_compute':compute,
                'fit_global_sha256':matrix.array_sha(gr[tr]),'eval_global_sha256':matrix.array_sha(gr[ev]),
                'readout_lock_sha256':args.expected_readout_lock_sha256,'study_lock_sha256':STUDY_SHA}
            write(jp,payload)
            argv=[sys.executable,str(ROOT/WORKER),'--readout-lock',args.readout_lock,
                '--expected-readout-lock-sha256',args.expected_readout_lock_sha256,
                '--study-lock',args.study_lock,'--expected-study-lock-sha256',STUDY_SHA,
                '--job',str(jp),'--expected-job-sha256',sha(jp),'--output-newdir',str(dest)]
            stream=(out/'logs'/(name+'.private.log')).open('xb')
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
            active.append({'process':process,'log':stream,'name':name,'meta':meta,'output':dest})
            print(json.dumps({'feature_state':i+1,'of':72,'active_CPU_workers':len(active),'job':name}),flush=True)
            if i==0:
                # First complete formal block qualifies the pipeline; no extra fit.
                while active:collect();time.sleep(1)
                require(not failures,'P2S_FIRST_REAL_READOUT_BLOCK_FAILED')
        while active:collect();time.sleep(1)
        require(not failures and len(completed)==72,'P2S_FULL_READOUT_INCOMPLETE')
        manifest={'schema':'semantic_b0b7_p2s_readout_manifest_v1','status':'COMPLETE',
            'pretraining_lock_sha256':STUDY_SHA,'readout_lock_sha256':args.expected_readout_lock_sha256,
            'jobs':sorted(completed,key=meta_key),'expected_registered_Cox_cells':2376,
            'new_neural_optimizer_updates':0,'outer_evaluation':False,'confidence_intervals_computed':False}
        write(out/'READOUT_MANIFEST.private.json',manifest)
        print(json.dumps({'status':'COMPLETE','feature_states':72,'new_neural_updates':0}))
    except BaseException as error:
        while active:collect();time.sleep(1)
        write(out/'READOUT_FAILURE.private.json',{'status':'FAILED_NO_PARTIAL_RESTART','error':str(error),
            'failures':failures,'completed_jobs':completed,'worker_signals_sent':0})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study-lock','readout-lock','expected-readout-lock-sha256','parent-execution-lock','p2-lock',
        'formal-private','diagnostic-private','assets','legacy-root','frozen-images','output-newdir','feature-cache-newdir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-cpu-workers',type=int,default=2)
    p.add_argument('--allow-completed-neural-training',action='store_true')
    run(p.parse_args())

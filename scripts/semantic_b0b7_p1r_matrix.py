"""Locked P1-R 36-development/48-replication dispatcher; no survival data.

The manager preserves completed units and never resumes a partial optimizer.
Every neural unit is an independent existing p1r.py CLI process. Selection of
the two-channel fraction occurs exactly once before replication is admitted.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(31,47,83)
OWNED_CHILDREN=[]


def stop_owned_children():
    for process in OWNED_CHILDREN:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass


def stop_signal(signum,frame):
    raise SystemExit(128+signum)


atexit.register(stop_owned_children)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()


def require(test,message):
    if not test:raise ValueError(message)


def write(path,value,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not replace:
        with path.open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
    else:
        temp=path.with_name(path.name+".writing")
        with temp.open("w",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
        os.replace(temp,path)


def hard_gap(metric):
    values=[metric["retrieval_hard"][d]["paired_gap_mean"] for d in ("image_to_gene","gene_to_image")]
    values=[v for v in values if v is not None and math.isfinite(v)]
    return math.fsum(values)/len(values) if values else None


def select_epoch(history,tolerance):
    rows=[r for r in history if "validation" in r]
    require(bool(rows),"NO_VALIDATION_CHECKPOINTS")
    best=min(r["validation"]["program"]["patient_equal_masked_huber"] for r in rows)
    rows=[r for r in rows if r["validation"]["program"]["patient_equal_masked_huber"]<=best+tolerance]
    return min(rows,key=lambda r:(-(hard_gap(r["validation"]) if hard_gap(r["validation"]) is not None else -math.inf),r["epoch"]))


def select_fraction(receipts):
    summaries=[]
    for a,tag in ((.5,"A050"),(.75,"A075")):
        names={"T3_"+arm+"_DUAL_"+tag for arm in ("B3","B4")}
        use=[r for r in receipts if r["variant"] in names]
        require(len(use)==6 and {(r["variant"],r["seed"]) for r in use}=={(n,s) for n in names for s in SEEDS},
                "INCOMPLETE_PAIRED_DUAL_SELECTION_GROUP")
        h=math.fsum(r["selected_validation"]["program"]["patient_equal_masked_huber"] for r in use)/6
        gaps=[hard_gap(r["selected_validation"]) for r in use]
        require(all(g is not None for g in gaps),"UNSUPPORTED_DUAL_SELECTION_HARD_GAP")
        summaries.append({"fraction":a,"mean_program_Huber":h,"mean_hard_gap":math.fsum(gaps)/6,
                          "mean_training_tolerance":math.fsum(r["program_huber_tolerance"] for r in use)/6})
    tolerance=math.fsum(x["mean_training_tolerance"] for x in summaries)/2
    best=min(x["mean_program_Huber"] for x in summaries)
    eligible=[x for x in summaries if x["mean_program_Huber"]<=best+tolerance]
    selected=min(eligible,key=lambda x:(-x["mean_hard_gap"],x["fraction"]))
    return {"selected_fraction":selected["fraction"],"summaries":summaries,"tolerance":tolerance,
            "selection":"six_B3_B4_same_seed_cells_mean_Huber_within_training_tol_higher_hardgap_then_a0.5",
            "replication_scores_used":False,"DSS_used":False}


def replication_names(fraction,definitions):
    tag="A050" if fraction==.5 else "A075" if fraction==.75 else None
    require(tag is not None,"UNLOCKED_DUAL_FRACTION")
    fixed=["ORIGINAL_B3","ORIGINAL_B4","T3_B3_DUAL_"+tag,"T3_B4_DUAL_"+tag,
           "T3_B4_DUAL_MAP_SHUFFLE_"+tag,"T4_B3_LAMBDA003","T4_B4_LAMBDA003"]
    candidates=[v["name"] for v in definitions if v.get("wrong_aux") and v.get("relation_mode")=="dual"
                and v.get("known_fraction")==fraction]
    require(len(candidates)==1,"EXACT_ONE_PRELOCKED_DUAL_WRONG_AUX_REQUIRED")
    return fixed[:5]+candidates+fixed[5:]


def resources(path):
    limit=Path('/sys/fs/cgroup/memory/memory.limit_in_bytes');usage=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    if limit.exists():memory={"usage_bytes":int(usage.read_text()),"limit_bytes":int(limit.read_text())}
    else:
        lim=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        memory={"usage_bytes":int(Path('/sys/fs/cgroup/memory.current').read_text()),
                "limit_bytes":int(lim) if lim!='max' else None}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True)
    return {"cgroup_memory":memory,"filesystem_free_bytes":shutil.disk_usage(path).free,
            "GPU_telemetry":gpu.stdout.strip() if gpu.returncode==0 else None,"epoch_seconds":time.time()}


def admission(resource):
    m=resource["cgroup_memory"]
    return resource["filesystem_free_bytes"]>=2*1024**3 and (m["limit_bytes"] is None or
           m["limit_bytes"]-m["usage_bytes"]>=8*1024**3)


def verify_sources(lock):
    require(lock.get("schema")=="semantic_b0b7_p1r_lock_v1" and lock.get("status") in
            ("LOCKED_BEFORE_OPTIMIZER","LOCKED_BEFORE_P1R_OPTIMIZER"),"MASTER_LOCK_NOT_FROZEN")
    source=lock["frozen_sources"]
    for name in ('scripts/semantic_b0b7_p1r_matrix.py','scripts/semantic_b0b7_p1r.py',
                 'src/semantic_b0b7/p1r_models.py','configs/semantic_b0b7/p1r_variants_v1.json'):
        require(name in source,"SCHEDULER_OR_WORKER_NOT_FROZEN:"+name)
    for name,expected in source.items():
        p=ROOT/name;require(p.resolve().is_relative_to(ROOT) and sha(p)==expected,"FROZEN_SOURCE_CHANGED:"+name)


def partitions(path):
    value=read(path);rows=value['partitions']
    require(isinstance(rows,list) and {int(x['partition']) for x in rows}=={0,1,2} and len(rows)==3,
            "THREE_PRELOCKED_PARTITION_FILES_REQUIRED")
    result={}
    for row in rows:
        require(Path(row['file']).is_file() and sha(row['file'])==row['sha256'],"PARTITION_FILE_SHA_CHANGED")
        result[int(row['partition'])]=dict(row)
    return result


def prepare_references(args,part,lock):
    # CPU-only input qualification; never import or read a survival loader.
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from argparse import Namespace
    from scripts import semantic_b0b7_p1 as legacy
    from scripts.semantic_b0b7_p1r import training_huber_tolerance
    data,binding=legacy.load_gate_and_inputs(Namespace(inputs=part['file'],expected_input_sha256=part['sha256'],gate=args.historical_gate))
    tr=data['train_indices'];tol,median=training_huber_tolerance(data['target'][tr],data['target_mask'][tr])
    refs=[]
    for arm in ('B3','B4'):
        for seed in SEEDS:
            directory=Path(args.original_fits_root)/'cells'/(arm+'_seed'+str(seed));receipt=read(directory/'cell.private.json')
            require(receipt['input_sha256']==part['sha256'] and receipt['frozen_sources']==binding['frozen_sources'],
                    "ORIGINAL_REFERENCE_SOURCE_OR_INPUT_CHANGED")
            require(len(receipt['states'])==4,"ORIGINAL_REFERENCE_FOUR_STATES_REQUIRED")
            for state in receipt['states']:require(sha(directory/state['file'])==state['sha256'],"ORIGINAL_REFERENCE_STATE_CHANGED")
            old=min((x for x in receipt['history'] if 'validation' in x),key=lambda r:(r['validation']['auxiliary_loss'],r['epoch']))
            selected=select_epoch(receipt['history'],tol)
            state=next(x for x in receipt['states'] if x['epoch']==selected['epoch'])
            refs.append({'identity':'REPLAY_ONLY','variant':'ORIGINAL_'+arm,'seed':seed,'partition':0,
                'input_sha256':part['sha256'],'source_receipt_sha256':sha(directory/'cell.private.json'),
                'historical_Laux_selected_epoch_unchanged':old['epoch'],'selected_epoch':selected['epoch'],
                'selected_validation':selected['validation'],'selected_checkpoint_sha256':state['sha256'],
                'program_huber_tolerance':tol,'training_median_program_Huber':median,
                'new_fits':0,'new_model_forwards':0,'uses_previously_completed_R0_exact_replays':True})
    return {'schema':'semantic_b0b7_p1r_uniform_legacy_references_v1','status':'COMPLETE_REPLAY_ONLY_RESELECTION',
            'source_sha256':sha(__file__),'input_sha256':part['sha256'],'historical_gate_sha256':sha(args.historical_gate),
            'training_tolerance':tol,'training_median_program_Huber':median,'references':refs,
            'T2_old_selected_encoders_changed':False,'new_fits':0,'new_forwards':0,'DSS_loaded':False}


def checked_receipt(directory,job,lock_sha,source):
    p=Path(directory)/'cell.private.json';r=read(p)
    require(r['schema']=='semantic_b0b7_p1r_cell_v1' and r['status']=='COMPLETE' and r['identity']=='NEW_FIT',"CELL_INCOMPLETE")
    require(r['variant']==job['variant'] and r['seed']==job['seed'] and r['input_sha256']==job['input_sha256']
            and r['master_lock_sha256']==lock_sha and r['frozen_sources']==source,"CELL_SOURCE_OR_SCOPE_CHANGED")
    require(r['optimizer_steps']==400 and len(r['history'])==400 and len(r['states'])==4 and
            {x['epoch'] for x in r['states']}=={50,100,200,400},"CELL_BUDGET_CHANGED")
    require(r['DSS_OS_PFI_accessed'] is False,"OUTCOME_ACCESS_DETECTED")
    for s in r['states']:
        f=Path(directory)/s['file'];require(f.resolve().parent==Path(directory).resolve()
            and sha(f)==s['sha256'] and s['actual_replay'] is True and s['replay_max_error']==0.,"STATE_REPLAY_OR_HASH_CHANGED")
    selected=select_epoch(r['history'],r['program_huber_tolerance'])
    require(selected['epoch']==r['selected_epoch'],"CELL_SELECTION_CHANGED")
    return r,sha(p)


def make_jobs(stage,names,partitions_by_id,ids):
    return [{'stage':stage,'partition':p,'variant':name,'seed':seed,
             'input_file':partitions_by_id[p]['file'],'input_sha256':partitions_by_id[p]['sha256'],
             'key':'p'+str(p)+'_'+name+'_seed'+str(seed)} for p in ids for name in names for seed in SEEDS]


def schedule(args,jobs,lock,root,phase):
    statepath=root/(phase+'_RUN_STATE.private.json');active={};completed=[];receipts=[];pending=list(jobs)
    locksha=sha(args.lock);logs=root/'logs';logs.mkdir(exist_ok=True)
    while pending or active:
        for key,entry in list(active.items()):
            if entry['process'].poll() is None:continue
            job=entry['job'];code=entry['process'].returncode;entry['log'].close()
            if code!=0:
                for other in active.values():
                    if other['process'].poll() is None:os.killpg(other['process'].pid,signal.SIGTERM)
                failure={'status':'FAILED_STOPPED_NO_AUTORETRY','failed_job':job,'returncode':code,
                    'already_completed':completed,'pending':pending,'source_sha256':sha(__file__),
                    'partial_outputs_preserved':True,'rerun_requires_new_affected_group_root_and_lock':True}
                write(root/(phase+'_FAILURE.private.json'),failure)
                raise RuntimeError('CELL_PROCESS_FAILED:'+key)
            receipt,digest=checked_receipt(entry['destination'],job,locksha,lock['frozen_sources'])
            selected={**job,'identity':'NEW_FIT','receipt_sha256':digest,'selected_epoch':receipt['selected_epoch'],
                      'checkpoint_sha256':receipt['selected_checkpoint_sha256'],'optimizer_steps':400,
                      'actual_saved_replays':4,'selected_validation':receipt['selected_validation'],
                      'program_huber_tolerance':receipt['program_huber_tolerance'],
                      'elapsed_seconds':receipt['elapsed_seconds'],'peak_cuda_allocated_bytes':receipt['peak_cuda_allocated_bytes']}
            completed.append(selected);receipts.append(receipt);del active[key]
            print(json.dumps({'stage':phase,'completed':len(completed),'total':len(jobs),'job':key}),flush=True)
        resource=resources(root)
        while pending and len(active)<args.max_parallel and admission(resource):
            verify_sources(lock);job=pending.pop(0);destination=root/'cells'/job['key']
            require(not destination.exists(),'PARTIAL_CELL_IS_NOT_RESUMED:'+job['key'])
            destination.parent.mkdir(exist_ok=True)
            command=[sys.executable,str(ROOT/'scripts/semantic_b0b7_p1r.py'),'--inputs',job['input_file'],
                '--expected-input-sha256',job['input_sha256'],'--lock',str(args.lock),'--variant',job['variant'],
                '--seed',str(job['seed']),'--output',str(destination),'--device',args.device,'--threads','2']
            env=dict(os.environ);env.update({'PYTHONUNBUFFERED':'1','PYTHONPATH':str(ROOT),
                'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
            log=(logs/(job['key']+'.log')).open('xb')
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            OWNED_CHILDREN.append(process)
            active[job['key']]={'process':process,'job':job,'destination':destination,'log':log}
            # Admission is bounded by the measured six-worker profile; no
            # host-free-RAM estimate is used as a container memory budget.
            resource=resources(root)
        state={'schema':'semantic_b0b7_p1r_live_matrix_v1','status':'RUNNING' if pending or active else 'COMPLETE',
            'phase':phase,'master_lock_sha256':locksha,'source_sha256':sha(__file__),
            'completed':completed,'active':[{'job':e['job'],'pid':e['process'].pid} for e in active.values()],
            'pending':pending,'resource':resource,'DSS_fits':0,'max_parallel':args.max_parallel}
        write(statepath,state,replace=True)
        if pending and not active and not admission(resource):
            write(root/(phase+'_RESOURCE_BLOCKED.private.json'),state,replace=True)
        if pending or active:time.sleep(args.poll_seconds)
    return completed,receipts


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('lock','inputs-manifest','registry','private-root','original-fits-root','historical-gate'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--stage',choices=('development','replication','all'),default='all')
    p.add_argument('--max-parallel',type=int,default=6);p.add_argument('--poll-seconds',type=float,default=5.)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    require(1<=args.max_parallel<=6 and 1<=args.poll_seconds<=30,'UNQUALIFIED_PARALLELISM_OR_POLLING')
    lock=read(args.lock);verify_sources(lock);book=read(args.registry)
    signal.signal(signal.SIGTERM,stop_signal)
    require(sha(args.registry)==lock['frozen_sources']['configs/semantic_b0b7/p1r_variants_v1.json'],'REGISTRY_CHANGED')
    parts=partitions(args.inputs_manifest);root=args.private_root;root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    write(root/'PRE_MATRIX_BINDING.private.json',{'schema':'semantic_b0b7_p1r_matrix_binding_v1',
        'master_lock_sha256':sha(args.lock),'input_registry_sha256':sha(args.inputs_manifest),
        'scheduler_sha256':sha(__file__),'partitions':parts,'stage':args.stage,'max_parallel':args.max_parallel,
        'new_fits_before_binding':0,'DSS_access':False})
    devnames=[v['name'] for v in book['new_fit_variants']]
    require(len(devnames)==12 and len(set(devnames))==12,'EXACT12_DEVELOPMENT_VARIANTS_REQUIRED')
    refs=prepare_references(args,parts[0],lock);write(root/'HISTORICAL_REFERENCES.private.json',refs)
    allcells=[]
    if args.stage in ('development','all'):
        jobs=make_jobs('development',devnames,parts,(0,));require(len(jobs)==36,'EXACT36_DEVELOPMENT_FITS')
        cells,receipts=schedule(args,jobs,lock,root,'DEVELOPMENT');allcells+=cells
        choice=select_fraction(receipts);choice.update(master_lock_sha256=sha(args.lock),
            development_receipt_sha256=[c['receipt_sha256'] for c in cells])
        write(root/'DEVELOPMENT_SELECTION.private.json',choice)
    else:
        source=lock.get('development_selection_file')
        require(source and sha(source)==lock.get('development_selection_sha256'),'REPLICATION_NEEDS_FROZEN_DEVELOPMENT_SELECTION')
        choice=read(source)
    if args.stage in ('replication','all'):
        definitions=book['new_fit_variants']+book['reference_variants']+lock.get('replication_variants',[])
        names=replication_names(choice['selected_fraction'],definitions)
        known={v['name'] for v in definitions};require(set(names)<=known,'REPLICATION_VARIANT_NOT_PRELOCKED')
        jobs=make_jobs('replication',names,parts,(1,2));require(len(jobs)==48,'EXACT48_REPLICATION_FITS')
        write(root/'PRE_REPLICATION_BINDING.private.json',{'selection_sha256':sha(root/'DEVELOPMENT_SELECTION.private.json')
              if (root/'DEVELOPMENT_SELECTION.private.json').exists() else lock['development_selection_sha256'],
              'jobs':jobs,'replication_validation_results_used_for_choice':False})
        cells,_=schedule(args,jobs,lock,root,'REPLICATION');allcells+=cells
    result={'schema':'semantic_b0b7_p1r_actual_matrix_v1','status':'COMPLETE_ACTUAL_REQUESTED_MATRIX',
        'master_lock_sha256':sha(args.lock),'input_registry_sha256':sha(args.inputs_manifest),
        'scheduler_sha256':sha(__file__),'source_sha256':lock['frozen_sources'],'stage':args.stage,
        'cells':allcells,'historical_references':refs,'dual_fraction_selection':choice,
        'new_neural_fits':len(allcells),'optimizer_updates':sum(x['optimizer_steps'] for x in allcells),
        'actual_saved_checkpoint_replays':sum(x['actual_saved_replays'] for x in allcells),
        'elapsed_wall_seconds':time.perf_counter()-started,'CI_computed':False,'DSS_fits':0,'P2_started':False}
    write(root/'MATRIX_REPORT.private.json',result)
    print(json.dumps({'status':result['status'],'new_neural_fits':len(allcells),
                      'actual_saved_replays':result['actual_saved_checkpoint_replays']}),flush=True)


if __name__=='__main__':main()

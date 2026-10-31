"""Bounded M1--M3 scheduler: inner selection barrier, then fixed outer fits.

Resources change admission only; they never alter sources, candidates or epochs.
No outer endpoint metric is computed here.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
from pathlib import Path
import argparse,hashlib,json,math,shutil,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123_worker import check_lock,read,sha,write,atomic
from src.semantic_m123.models import ARMS

GIB=1024**3


def inner_jobs(lock_sha):
    jobs=[]
    for outer in range(1,6):
        for inner in range(3):
            split=f'outer{outer}_inner{inner}'
            for lam in (.1,1.,10.):
                jobs.append({'split':split,'role':'inner','arm':'M0PLUS','seed':0,'lambda':lam,'epochs':0,'lock_sha256':lock_sha})
            for arm in ARMS:
                for seed in (31,47,83):
                    for lam in (.001,.01):
                        jobs.append({'split':split,'role':'inner','arm':arm,'seed':seed,'lambda':lam,'epochs':400,'lock_sha256':lock_sha})
    return jobs


def identifier(job):return f"{job['split']}/{job['arm']}_seed{job['seed']}_lambda{job['lambda']}"


def cell_path(private,job):return Path(private)/'fits'/identifier(job)


def verify_complete(private,job):
    p=cell_path(private,job);r=p/'COMPLETE.private.json'
    if not r.exists():return None
    row=read(r)
    if row['status']!='COMPLETE' or row['job']!=job:raise ValueError('COMPLETION_JOB_BINDING_CHANGED')
    for key,name in (('checkpoint_sha256',row['checkpoint_file']),('prediction_sha256','PREDICTIONS.private.npz'),('history_sha256','HISTORY.private.json.gz')):
        if sha(p/name)!=row[key]:raise ValueError('COMPLETED_PRIVATE_ARTIFACT_CHANGED')
    if row['saved_checkpoint_prediction_max_error']>2e-6:raise ValueError('COMPLETION_REPLAY_ERROR')
    return row


def choose(candidates):
    """Metrics are within-partition pair components, never pooled patient risks."""
    scored=[]
    for lam,epoch,rows in candidates:
        if len(rows)!=3:raise ValueError('EXACT_THREE_INNER_PARTITIONS_REQUIRED')
        for row in rows:
            if row.get('Harrell') is None:raise ValueError('NO_HARRELL_SUPPORT_NO_FAKE_SCORE')
        hn=sum(x['Harrell_num'] for x in rows);hd=sum(x['Harrell_den'] for x in rows)
        uno_ok=all(x.get('Uno60') is not None for x in rows)
        un=sum(x['Uno_num'] for x in rows) if uno_ok else None
        ud=sum(x['Uno_den'] for x in rows) if uno_ok else None
        h=hn/hd;u=un/ud if uno_ok and ud>0 else None
        scored.append({'lambda':lam,'epoch':epoch,'Harrell':h,'Uno60':u,
            'Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud})
    # Stable explicit tolerance prevents floating summation noise from becoming
    # the rationale for selecting a more complex configuration.
    best=None
    for x in sorted(scored,key=lambda x:(-x['lambda'],x['epoch'])):
        if best is None or x['Harrell']>best['Harrell']+1e-12:
            best=x;continue
        if abs(x['Harrell']-best['Harrell'])<=1e-12:
            u=x['Uno60'] if x['Uno60'] is not None else -math.inf
            b=best['Uno60'] if best['Uno60'] is not None else -math.inf
            if u>b+1e-12:best=x
    return best,scored


def freeze_choices(args):
    path=Path(args.private)/'ALL_OUTER_SELECTIONS_FROZEN.private.json'
    if path.exists():raise ValueError('SELECTION_BARRIER_ALREADY_EXISTS_NO_OVERWRITE')
    choices={};audit={}
    for outer in range(1,6):
        for arm,seeds,lambdas,epochs in [('M0PLUS',(0,),(.1,1.,10.),(0,))]+[(a,(31,47,83),(.001,.01),(0,20,50,100,200,400)) for a in ARMS]:
            for seed in seeds:
                candidates=[]
                for lam in lambdas:
                    fitrows=[]
                    for inner in range(3):
                        job={'split':f'outer{outer}_inner{inner}','role':'inner','arm':arm,'seed':seed,'lambda':lam,
                            'epochs':0 if arm=='M0PLUS' else 400,'lock_sha256':args.expected_lock_sha256}
                        r=verify_complete(args.private,job)
                        if r is None:raise ValueError('CANNOT_SELECT_BEFORE_ALL_INNER_FITS')
                        fitrows.append(r)
                    for epoch in epochs:
                        metrics=[next(s['evaluation_metrics'] for s in r['snapshots'] if s['epoch']==epoch) for r in fitrows]
                        candidates.append((lam,epoch,metrics))
                key=f'outer{outer}/{arm}/{seed}'
                selected,allrows=choose(candidates)
                choices[key]=selected;audit[key]=allrows
    result={'schema':'m123_global_inner_selection_barrier_v1','status':'ALL_FIVE_OUTERS_SELECTED_BEFORE_ANY_NEW_OUTER_SCORE',
        'lock_sha256':args.expected_lock_sha256,'choices':choices,'all_fixed_inner_candidates':audit,
        'outer_metrics_read':False,'independent_confirmation_claim':False}
    write(path,result)
    return result,sha(path)


def final_jobs(lock_sha,barrier,barrier_sha):
    jobs=[]
    for key,choice in barrier['choices'].items():
        outer,arm,seed=key.split('/')
        jobs.append({'split':outer+'_final','role':'final','arm':arm,'seed':int(seed),'lambda':choice['lambda'],
            'epochs':choice['epoch'],'lock_sha256':lock_sha,'selection_sha256':barrier_sha})
    return jobs


def available_ram():
    free=psutil.virtual_memory().available
    for base in ('/sys/fs/cgroup','/sys/fs/cgroup/memory'):
        try:
            p=Path(base)
            if (p/'memory.max').exists():
                maximum=(p/'memory.max').read_text().strip();current=int((p/'memory.current').read_text())
            else:
                maximum=(p/'memory.limit_in_bytes').read_text().strip();current=int((p/'memory.usage_in_bytes').read_text())
            if maximum!='max':free=min(free,max(0,int(maximum)-current))
        except (OSError,ValueError):pass
    return free


def gpu_free():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return int(r.stdout.strip().splitlines()[0])*1024**2


def admit(args,private,gpu):
    diskfree=shutil.disk_usage(private).free
    bytes_written=sum(p.stat().st_size for p in Path(private).rglob('*') if p.is_file())
    if diskfree<1*GIB+20_000_000 or bytes_written>1_500_000_000:return False
    if available_ram()<8*GIB:return False
    return (not gpu) or gpu_free()>2*GIB+4_700_000_000


def run_stage(args,jobs,*,qualify_first=False):
    pending=[]
    for job in jobs:
        if verify_complete(args.private,job) is not None:continue
        p=cell_path(args.private,job)
        if p.exists():raise ValueError('PARTIAL_CELL_NOT_AUTOMATICALLY_RETRAINED:'+identifier(job))
        pending.append(job)
    active={};completed=0;failure=False
    if qualify_first:
        pending.sort(key=lambda j:(not(j['arm']=='M1_WARM' and j['seed']==31 and j['lambda']==.01),identifier(j)))
    qualified=not qualify_first
    while pending or active:
        for pid,item in list(active.items()):
            p,job,log=item
            code=p.poll()
            if code is None:continue
            log.close();del active[pid]
            if code!=0 or verify_complete(args.private,job) is None:
                failure=True
                atomic(Path(args.private)/'FAILURE.private.json',{'job':job,'exit_code':code,'pid':pid})
            else:completed+=1;qualified=True
        if failure:
            if active:time.sleep(5);continue
            raise RuntimeError('REGISTERED_FIT_FAILED_NO_NEW_ADMISSIONS')
        launched=False
        for job in list(pending):
            is_gpu=job['arm']!='M0PLUS'
            gpu_count=sum(x[1]['arm']!='M0PLUS' for x in active.values())
            cpu_count=len(active)-gpu_count
            if not qualified and active:break
            if is_gpu and gpu_count>=(args.max_parallel if qualified else 1):continue
            if not is_gpu and cpu_count>=1:continue
            if not admit(args,args.private,is_gpu):continue
            folder=Path(args.private)/'jobs';folder.mkdir(exist_ok=True)
            jp=folder/(identifier(job).replace('/','__')+'.json')
            write(jp,job)
            logpath=Path(args.private)/'logs'/(jp.stem+'.log');logpath.parent.mkdir(exist_ok=True)
            log=logpath.open('xb')
            cmd=[sys.executable,str(ROOT/'scripts/semantic_m123_worker.py'),'--lock',str(args.lock),'--expected-lock-sha256',args.expected_lock_sha256,
                '--job',str(jp),'--expected-job-sha256',sha(jp),'--private',str(args.private),'--output-newdir',str(cell_path(args.private,job))]
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'),start_new_session=True)
            active[p.pid]=(p,job,log);pending.remove(job);launched=True
            if not qualified:break
        atomic(Path(args.private)/'LIVE.private.json',{'pending':len(pending),'completed_in_this_stage':completed,
            'registered_in_stage':len(jobs),'active':[{'pid':pid,'job':j,'runtime_seconds':None} for pid,(_,j,_) in active.items()],
            'disk_free_bytes':shutil.disk_usage(args.private).free,'cgroup_available_bytes':available_ram(),
            'GPU_free_bytes':gpu_free(),'new_outer_metrics_read':False})
        if pending and not active and not launched:raise RuntimeError('RESOURCE_RESERVE_BLOCKED_WITH_NO_ACTIVE_FIT')
        if active:time.sleep(5)


def run(args):
    check_lock(args.lock,args.expected_lock_sha256)
    if not 1<=args.max_parallel<=6:raise ValueError('LOCKED_PARALLEL_LIMIT')
    private=Path(args.private);private.mkdir(parents=True,exist_ok=False)
    owner=private/'OWNER.private.json';write(owner,{'pid':os.getpid(),'lock_sha256':args.expected_lock_sha256})
    try:
        jobs=inner_jobs(args.expected_lock_sha256);assert len(jobs)==945
        run_stage(args,jobs,qualify_first=True)
        barrier,bsha=freeze_choices(args)
        outer=final_jobs(args.expected_lock_sha256,barrier,bsha);assert len(outer)==155
        run_stage(args,outer)
        totals={'neural_initializations':0,'Cox_fits':0,'new_optimizer_updates':0,'actual_checkpoint_replays':0}
        for j in jobs+outer:
            r=verify_complete(private,j);totals['Cox_fits']+=int(j['arm']=='M0PLUS')
            totals['neural_initializations']+=int(j['arm']!='M0PLUS')
            totals['new_optimizer_updates']+=r['optimizer_updates'];totals['actual_checkpoint_replays']+=1
        write(private/'STUDY_COMPLETE.private.json',{'schema':'m123_completed_training_v1','status':'ALL_REGISTERED_FITS_COMPLETE',
            'lock_sha256':args.expected_lock_sha256,'selection_barrier_sha256':bsha,'totals':totals,'outer_metrics_read':False})
    finally:
        if owner.exists():owner.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','private'):p.add_argument('--'+k,required=True)
    p.add_argument('--max-parallel',type=int,default=6)
    run(p.parse_args())


"""Qualified six-worker execution; eligibility precedes any formal outer fit."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,shutil,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import verify_lock,verify_cell,folder,identifier,write,read,sha,atomic
from scripts.semantic_m123r_selection import block_jobs,select_block
from src.semantic_m123r.models import ARMS

GIB=1024**3


def final_jobs(barrier,barrier_sha):
    jobs=[]
    for key,selection in barrier['selections'].items():
        outer,arm,seed=key.split('/');configs={}
        for c in selection['choices'].values():configs.setdefault(json.dumps(c['config'],sort_keys=True),[]).append(c['epoch'])
        for raw,epochs in configs.items():
            jobs.append({'phase':'formal','role':'final','outer':int(outer[5:]),'inner':None,'arm':arm,'seed':int(seed),
                'config':json.loads(raw),'epochs':max(epochs),'lock_sha256':barrier['lock_sha256'],'selection_sha256':barrier_sha})
    return jobs


def resource_ok(private):
    if min(shutil.disk_usage('/').free,shutil.disk_usage(private).free)<GIB+64*1024**2:return False
    from scripts.semantic_m123_schedule import available_ram,gpu_free
    return available_ram()>12*GIB and gpu_free()>GIB*2+4_700_000_000


def run_jobs(a,jobs,*,first_qualification=False):
    pending=[]
    for job in jobs:
        if verify_cell(a.private,job) is not None:continue
        if folder(a.private,job).exists():raise ValueError('PARTIAL_FIT_REQUIRES_REVIEW_NO_BLIND_RESTART')
        pending.append(job)
    active={};completed=0;qualified=not first_qualification;failed=False
    while pending or active:
        for pid,(proc,job,log) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del active[pid]
            if code!=0 or verify_cell(a.private,job) is None:
                failed=True;atomic(Path(a.private)/'FAILURE.private.json',{'job':job,'exit_code':code,'pid':pid})
            else:completed+=1;qualified=True
        if failed:
            if active:time.sleep(5);continue
            raise RuntimeError('FIT_FAILED_ACTIVE_WORKERS_DRAINED_NO_NEW_ADMISSIONS')
        while pending and len(active)<(a.max_parallel if qualified else 1) and resource_ok(a.private):
            job=pending.pop(0);name=identifier(job).replace('/','__')+f"_stop{job['epochs']}"
            jp=Path(a.private)/'jobs'/(name+'.json');write(jp,job)
            lp=Path(a.private)/'logs'/(name+'.log');lp.parent.mkdir(exist_ok=True);log=lp.open('xb')
            cmd=[sys.executable,str(ROOT/'scripts/semantic_m123r_worker.py'),'--lock',a.lock,'--expected-lock-sha256',a.expected_lock_sha256,
                '--job',str(jp),'--expected-job-sha256',sha(jp),'--private',a.private]
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[proc.pid]=(proc,job,log)
            if not qualified:break
        atomic(Path(a.private)/'LIVE.private.json',{'stage':a.stage,'active':[{'pid':pid,'job':j} for pid,(_,j,_) in active.items()],
            'pending_in_current_block':len(pending),'completed_in_current_block':completed,'registered_in_current_block':len(jobs),
            'root_free_bytes':shutil.disk_usage('/').free,'data_free_bytes':shutil.disk_usage(a.private).free,'outer_scores_computed':False})
        if pending and not active:raise RuntimeError('RESOURCE_FLOOR_PREVENTED_NEW_ADMISSION')
        if active:time.sleep(5)


def retain_final(private,jobs,barrier_path):
    b=read(barrier_path)
    for j in jobs:
        r=verify_cell(private,j);choice=b['selections'][f"outer{j['outer']}/{j['arm']}/{j['seed']}"]['choices']
        keep={str(c['epoch']) for c in choice.values() if c['config']==j['config']}
        f=folder(private,j);removed={c['file']:c['sha256'] for e,c in r['checkpoints'].items() if e not in keep}
        m=f/'WEIGHT_RETENTION.private.json'
        if m.exists():continue
        write(m,{'verified_removed':removed,'kept_epochs':sorted(map(int,keep)),'selection_sha256':sha(barrier_path),
            'all_predictions_metrics_histories_preserved':True,'no_outer_metric_selection':True})
        for n,h in removed.items():
            target=f/n
            if target.is_symlink() or not target.resolve().is_relative_to((Path(private)/'fits').resolve()) or sha(target)!=h:raise ValueError('UNSAFE_FINAL_RETENTION')
            target.unlink()


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','private'):p.add_argument('--'+k,required=True)
    p.add_argument('--stage',choices=('qualification','formal'),required=True)
    p.add_argument('--max-parallel',type=int,default=6);p.add_argument('--decision');p.add_argument('--decision-sha256');p.add_argument('--resume',action='store_true')
    a=p.parse_args();lock,parent=verify_lock(a.lock,a.expected_lock_sha256)
    if not 1<=a.max_parallel<=6:raise ValueError('LOCKED_PARALLEL_CAP')
    private=Path(a.private)
    if not private.exists():
        if a.stage!='qualification':raise ValueError('QUALIFICATION_MUST_RUN_FIRST')
        private.mkdir(parents=True);write(private/'EXECUTION_BINDING.private.json',{'lock_sha256':a.expected_lock_sha256,'created_epoch':time.time()})
    elif a.stage=='qualification' and not a.resume:raise ValueError('NEW_QUALIFICATION_DIRECTORY_REQUIRED')
    if read(private/'EXECUTION_BINDING.private.json')['lock_sha256']!=a.expected_lock_sha256:raise ValueError('RESUME_MASTER_CHANGED')
    owner=private/'OWNER.private.json'
    if owner.exists():raise ValueError('ANOTHER_SCHEDULER_OWNS_STUDY')
    write(owner,{'pid':os.getpid(),'lock_sha256':a.expected_lock_sha256,'stage':a.stage,'created_epoch':time.time()})
    try:
        if a.stage=='qualification':
            order=('R6','R2','R3','R4','R5','R7','R8','R9')
            first=True
            for arm in order:
                for seed in (31,47,83):
                    jobs=block_jobs(1,arm,seed,a.expected_lock_sha256)
                    run_jobs(a,jobs,first_qualification=first);first=False
                    select_block(private,1,arm,seed,a.expected_lock_sha256)
            alljobs=[j for arm in ARMS for seed in (31,47,83) for j in block_jobs(1,arm,seed,a.expected_lock_sha256)]
            receipts=[verify_cell(private,j) for j in alljobs]
            write(private/'QUALIFICATION_COMPLETE.private.json',{'status':'ALL540_QUALIFICATION_FITS_COMPLETE','lock_sha256':a.expected_lock_sha256,
                'registered_fits':len(alljobs),'new_optimizer_updates':sum(r['optimizer_updates'] for r in receipts),
                'scheduler_wall_seconds':time.time()-read(private/'EXECUTION_BINDING.private.json')['created_epoch'],
                'outer_test_scores_computed':False,'source_model_choices_not_changed':True})
        else:
            if sha(a.decision)!=a.decision_sha256:raise ValueError('QUALIFICATION_DECISION_CHANGED')
            decision=read(a.decision)
            from src.semantic_m123r.protocol import formal_arms
            if decision['status']!='COMPLETE_DEVELOPMENT_QUALIFICATION' or decision['lock_sha256']!=a.expected_lock_sha256 or decision['training_complete_sha256']!=sha(private/'QUALIFICATION_COMPLETE.private.json'):
                raise ValueError('FORMAL_DECISION_LINEAGE')
            arms=decision['formal_arms']
            if not arms or arms!=formal_arms(decision['gates']):raise ValueError('NO_QUALIFIED_FORMAL_ARMS')
            for outer in range(2,6):
                for arm in arms:
                    for seed in (31,47,83):
                        run_jobs(a,block_jobs(outer,arm,seed,a.expected_lock_sha256))
                        select_block(private,outer,arm,seed,a.expected_lock_sha256)
            selections={}
            for outer in range(1,6):
                for arm in arms:
                    for seed in (31,47,83):selections[f'outer{outer}/{arm}/{seed}']=read(private/'selections'/f'outer{outer}_{arm}_seed{seed}.private.json')
            bp=private/'ALL_OUTER_CHOICES_FROZEN.private.json'
            barrier={'status':'ALL_FIVE_OUTERS_SELECTED_BEFORE_NEW_OUTER_SCORE','lock_sha256':a.expected_lock_sha256,
                'qualification_decision_sha256':a.decision_sha256,'selections':selections,'outer_scores_computed':False}
            if bp.exists():
                if read(bp)!=barrier:raise ValueError('GLOBAL_OUTER_SELECTION_CHANGED')
            else:write(bp,barrier)
            jobs=final_jobs(barrier,sha(bp));run_jobs(a,jobs);retain_final(private,jobs,bp)
            write(private/'FORMAL_COMPLETE.private.json',{'status':'ALL_FORMAL_FITS_COMPLETE_NO_OUTER_SCORING_YET','lock_sha256':a.expected_lock_sha256,
                'selection_sha256':sha(bp),'actual_final_initializations':len(jobs),'formal_arms':arms,'outer_scores_computed':False})
    finally:
        if owner.exists():owner.unlink()


if __name__=='__main__':main()

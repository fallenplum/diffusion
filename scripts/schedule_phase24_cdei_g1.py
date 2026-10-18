"""Finite fifteen-cell queue; never launch outer implicitly or retry failures."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase24_cdei_g1.runtime import (cfg,source,PRIVATE,cell_dir,authorization,verify_done,atomic_json)


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['inner','outer'],required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--device',default='cuda');a=p.parse_args()
    if not 1<=a.workers<=15:raise ValueError('invalid worker count')
    import fcntl
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/(a.role+'_QUEUE.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=source();authorization(a.role,s);c=cfg()
        todo=[(seed,fold) for fold in c['folds'] for seed in c['seeds']]
        assert len(todo)==15
        running=[];done=[];failed=[];logs=PRIVATE/'logs';logs.mkdir(exist_ok=True)
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        started=time.time()
        while todo or running:
            while todo and len(running)<a.workers:
                seed,fold=todo.pop(0);d=cell_dir(a.role,seed,fold);name=d.name
                if (d/'DONE_PRIVATE.json').exists():
                    verify_done(d,s,a.role,seed,fold);done.append(name);continue
                if (d/'ATTEMPTS_PRIVATE.json').exists():
                    failed.append({'cell':name,'reason':'PRIOR_ATTEMPT_REVIEW_REQUIRED'});continue
                if source()!=s:raise ValueError('source drift')
                f=(logs/(a.role+'_'+name+'.log')).open('ab')
                child=subprocess.Popen([sys.executable,'scripts/run_phase24_cdei_g1.py','--role',a.role,
                     '--seed',str(seed),'--fold',str(fold),'--device',a.device],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                running.append((child,f,seed,fold));print('START',name,child.pid,flush=True)
            for row in list(running):
                child,f,seed,fold=row;exitcode=child.poll()
                if exitcode is None:continue
                running.remove(row);f.close();d=cell_dir(a.role,seed,fold)
                if exitcode:failed.append({'cell':d.name,'exitcode':exitcode})
                else:verify_done(d,s,a.role,seed,fold);done.append(d.name)
            atomic_json(PRIVATE/(a.role+'_SCHEDULE_PRIVATE.json'),{'status':'RUNNING' if todo or running else ('COMPLETE' if len(done)==15 and not failed else 'PARTIAL_FAILED'),
               'source':s,'completed':len(done),'failed':len(failed),'pending':len(todo),'workers':a.workers,
               'completed_cells':done,'failed_cells':failed,'running':[{'seed':x[2],'fold':x[3],'pid':x[0].pid} for x in running],
               'unix_start':started,'unix_update':time.time(),'automatic_retries':0})
            if running:time.sleep(3)
        if failed:raise SystemExit(1)


if __name__=='__main__':main()

"""Finite six-cell prototype queue; no model search and no failed retries."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase25_local.runtime import cfg,source,PRIVATE,cell_dir,authorization,verify_done,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--device',default='cuda');a=p.parse_args()
    if not 1<=a.workers<=6:raise ValueError('invalid concurrency')
    import fcntl
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'PROTOTYPE_QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=cfg();identity=source();authorization('prototype',identity)
        todo=[(s,f) for f in c['prototype']['folds'] for s in c['prototype']['seeds']]
        if len(todo)!=6:raise ValueError('locked six-cell matrix changed')
        running=[];done=[];failed=[];logs=PRIVATE/'logs';logs.mkdir(exist_ok=True);started=time.time()
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        while todo or running:
            while todo and len(running)<a.workers:
                seed,fold=todo.pop(0);path=cell_dir('prototype',seed,fold)
                if (path/'DONE_PRIVATE.json').exists():verify_done(path,identity,'prototype',seed,fold);done.append(path.name);continue
                if (path/'ATTEMPTS_PRIVATE.json').exists():failed.append({'cell':path.name,'reason':'PRIOR_ATTEMPT_REVIEW_REQUIRED'});continue
                if source()!=identity:raise ValueError('source drift')
                log=(logs/(path.name+'.log')).open('ab')
                child=subprocess.Popen([sys.executable,'scripts/run_phase25.py','--stage','prototype','--seed',str(seed),'--fold',str(fold),'--device',a.device],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                running.append((child,log,seed,fold));print('START',path.name,child.pid,flush=True)
            for row in list(running):
                child,log,seed,fold=row;code=child.poll()
                if code is None:continue
                running.remove(row);log.close();path=cell_dir('prototype',seed,fold)
                if code:failed.append({'cell':path.name,'exitcode':code})
                else:verify_done(path,identity,'prototype',seed,fold);done.append(path.name)
            atomic_json(PRIVATE/'PROTOTYPE_SCHEDULE_PRIVATE.json',{'status':'RUNNING' if todo or running else ('COMPLETE' if len(done)==6 and not failed else 'PARTIAL_FAILED'),
                'source':identity,'completed':len(done),'failed':len(failed),'pending':len(todo),'workers':a.workers,
                'completed_cells':done,'failed_cells':failed,'running':[{'seed':r[2],'fold':r[3],'pid':r[0].pid} for r in running],
                'unix_start':started,'unix_update':time.time(),'automatic_retries':0})
            if running:time.sleep(3)
        if failed:raise SystemExit(1)


if __name__=='__main__':main()

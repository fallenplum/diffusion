"""Bounded concurrency, per-cell persistence, no rerun of valid completed cells."""
import argparse,subprocess,time,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--stage',choices=['baselines','mechanisms'],required=True);ap.add_argument('--output',default='/root/phase10_trimodal_private/formal');a=ap.parse_args();out=Path(a.output);logs=out/'scheduler_logs';logs.mkdir(exist_ok=True,parents=True)
    queue=[(s,f) for s in [31,47,83] for f in [1,2,3,4,5]];active=[];fail=[];done=[]
    env=os.environ.copy();env.update({'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PHASE10_THREADS':'2'})
    while queue or active:
        while queue and len(active)<a.workers:
            s,f=queue.pop(0);fp=(logs/f'{a.stage}_seed{s}_fold{f}.log').open('a');cmd=[sys.executable,'scripts/run_phase10_trimodal.py','run','--stage',a.stage,'--seed',str(s),'--fold',str(f),'--output',a.output]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT);active.append((proc,s,f,fp));print('START',a.stage,s,f,proc.pid,flush=True)
        for proc,s,f,fp in active[:]:
            if proc.poll() is not None:
                fp.close();active.remove((proc,s,f,fp));record={'seed':s,'fold':f,'exit':proc.returncode};(done if proc.returncode==0 else fail).append(record);print('EXIT',record,flush=True)
                if proc.returncode!=0:
                    # Fail closed. Other jobs finish their current cells; don't
                    # claim partial matrix complete or silently change budget.
                    queue.clear()
        state={'stage':a.stage,'completed_jobs':done,'failed_jobs':fail,'active':[{'seed':s,'fold':f,'pid':p.pid} for p,s,f,fp in active],'pending_jobs':queue}
        (out/f'{a.stage}_scheduler_state.json').write_text(json.dumps(state,indent=2))
        if active:time.sleep(10)
    if fail:return 2
    if a.stage=='baselines':
        return subprocess.call([sys.executable,'scripts/validate_phase10_independent.py','--baseline','--output',a.output],cwd=ROOT,env=env)
    return subprocess.call([sys.executable,'scripts/validate_phase10_independent.py','--output',a.output],cwd=ROOT,env=env)

if __name__=='__main__':raise SystemExit(main())

"""Detached bounded GPU scheduler and independent final validation."""
import itertools,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_phase14 import cfg,PRIVATE,REPORT,source,write,unit


def execute():
    configuration=cfg();jobs=list(itertools.product(configuration['endpoints'],configuration['genes'],configuration['seeds'],configuration['folds']));active=[];done=0;fail=[];telemetry=[]
    (PRIVATE/'logs').mkdir(parents=True,exist_ok=True)
    while jobs or active:
        while jobs and len(active)<configuration['workers']:
            ep,gene,seed,fold=jobs.pop(0);name=unit(ep,gene,seed,fold);fp=(PRIVATE/'logs'/f'{name}.log').open('w')
            p=subprocess.Popen([sys.executable,'-u','scripts/run_phase14.py','--endpoint',ep,'--gene',gene,'--seed',str(seed),'--fold',str(fold)],stdout=fp,stderr=subprocess.STDOUT,cwd=ROOT)
            active.append((p,name,fp))
        for p,name,fp in active[:]:
            if p.poll() is not None:
                fp.close();active.remove((p,name,fp));done+=1
                if p.returncode:fail.append({'job':name,'exit':p.returncode});jobs.clear()
                print('EXIT',name,p.returncode,flush=True)
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip();telemetry.append({'time':time.time(),'gpu':gpu,'active':len(active)})
        write(PRIVATE/'SCHEDULER_STATE.json',{'completed':done,'pending':len(jobs),'active':[{'pid':p.pid,'name':n} for p,n,f in active],'failed':fail});write(PRIVATE/'GPU_TELEMETRY.json',telemetry)
        if active:time.sleep(10)
    if fail:raise RuntimeError(str(fail))
    subprocess.check_call([sys.executable,'scripts/evaluate_phase14.py'],cwd=ROOT)
    print('FORMAL_AND_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    if '--worker' in sys.argv:execute()
    else:
        env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        PRIVATE.mkdir(exist_ok=True)
        with (PRIVATE/'full_run.log').open('w') as fp:p=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':p.pid,'log':str(PRIVATE/'full_run.log')}))

"""Bounded parallel anchors then complete pipelines, no duplicate anchor writes."""
import itertools,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase15_complement.core import *


def schedule():
    configuration=cfg();gate=json.loads((REPORT/'HARD_GATES.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=source():raise RuntimeError('gate absent')
    base=[['--endpoint',ep,'--seed',str(seed),'--fold',str(fold),'--prepare'] for ep,seed,fold in itertools.product(configuration['endpoints'],configuration['seeds'],configuration['folds'])]
    formal=[['--endpoint',ep,'--gene',gene,'--seed',str(seed),'--fold',str(fold)] for ep,gene,seed,fold in itertools.product(configuration['endpoints'],configuration['genes'],configuration['seeds'],configuration['folds'])]
    logs=PRIVATE/'logs';logs.mkdir(exist_ok=True);telemetry=[]
    for stage,jobs in [('ANCHOR',base),('FORMAL',formal)]:
        active=[];done=0;fail=[];counter=0
        while jobs or active:
            while jobs and len(active)<configuration['workers']:
                args=jobs.pop(0);counter+=1;fp=(logs/f'{stage}_{counter}.log').open('w');p=subprocess.Popen([sys.executable,'-u','scripts/run_phase15.py',*args],stdout=fp,stderr=subprocess.STDOUT,cwd=ROOT);active.append((p,args,fp))
            for p,args,fp in active[:]:
                if p.poll() is not None:
                    fp.close();active.remove((p,args,fp));done+=1
                    if p.returncode:fail.append(args);jobs.clear()
                    print('EXIT',stage,args,p.returncode,flush=True)
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip();telemetry.append({'stage':stage,'time':time.time(),'gpu':gpu})
            write(PRIVATE/'SCHEDULER_STATE.json',{'stage':stage,'completed':done,'pending':len(jobs),'active':[{'pid':p.pid,'args':a} for p,a,f in active],'failed':fail});write(PRIVATE/'GPU_TELEMETRY.json',telemetry)
            if active:time.sleep(10)
        if fail:raise RuntimeError(str(fail))
    subprocess.check_call([sys.executable,'scripts/evaluate_phase15.py'],cwd=ROOT);print('FORMAL_AND_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    if '--worker' in sys.argv:schedule()
    else:
        env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        with (PRIVATE/'run.log').open('w') as fp:p=subprocess.Popen([sys.executable,'-u',__file__,'--worker'],cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':p.pid,'log':str(PRIVATE/'run.log')}))

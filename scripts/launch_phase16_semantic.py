import sys,os,time,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_semantic.core import *
def main(pilot=False):
    (P/'logs').mkdir(parents=True,exist_ok=True);env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    task=[(arm,seed,fold) for arm in arms() if not pilot or arm.endswith('_512') for seed in ([31] if pilot else cfg()['seeds']) for fold in ([1,2] if pilot else cfg()['folds'])];running=[];failures=[];completed=0;started=time.time()
    while task or running:
        while task and len(running)<(4 if pilot else cfg()['workers']):
            eligible=next((i for i,(arm,seed,fold) in enumerate(task) if (directory(seed,fold,'inner')/'DONE.json').exists() and (directory(seed,fold,'outer')/'DONE.json').exists()),None)
            if eligible is None:break
            arm,seed,fold=task.pop(eligible);name=label(arm,seed,fold);log=open(P/'logs'/f'{"pilot_" if pilot else ""}{name}.log','w');cmd=[sys.executable,str(ROOT/'scripts/run_phase16_semantic.py'),'--arm',arm,'--seed',str(seed),'--fold',str(fold)]+(['--pilot'] if pilot else []);p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);running.append((p,log,name))
        pending=[]
        for process,log,name in running:
            code=process.poll()
            if code is None:pending.append((process,log,name));continue
            log.close()
            if code:failures.append({'cell':name,'exit_code':code});print('FAILED',name,flush=True)
            else:completed+=1;print('DONE',completed,name,flush=True)
        running=pending;write(P/('PILOT_SCHEDULER.json' if pilot else 'SCHEDULER.json'),{'completed':completed,'running':[r[2] for r in running],'remaining':len(task),'failures':failures,'seconds':time.time()-started})
        if failures and not running:raise SystemExit(2)
        time.sleep(2)
    write(P/('PILOT_DONE.json' if pilot else 'TRAINING_DONE.json'),{'status':'PASS','source':source(),'cells':completed,'expected':18 if pilot else 270,'seconds':time.time()-started})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)

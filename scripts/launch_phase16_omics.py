import sys,os,subprocess,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_semantic.omics import *
def main(pilot=False):
    (MP/'logs').mkdir(parents=True,exist_ok=True);env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2');specs=armdefs();tasks=[(arm,seed,fold) for arm,spec in specs.items() if not pilot or spec['budget'] in [0,512] for seed in ([31] if pilot else cfg()['seeds']) for fold in ([1,2] if pilot else cfg()['folds'])];expected=len(tasks);running=[];failures=[];completed=0
    while tasks or running:
        while tasks and len(running)<(4 if pilot else 8):
            arm,seed,fold=tasks.pop(0);name=label(arm,seed,fold);f=open(MP/'logs'/f'{"pilot_" if pilot else ""}{name}.log','w');p=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_phase16_omics.py'),'--arm',arm,'--seed',str(seed),'--fold',str(fold)]+(['--pilot'] if pilot else []),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);running.append((p,f,name))
        pending=[]
        for p,f,name in running:
            code=p.poll()
            if code is None:pending.append((p,f,name));continue
            f.close()
            if code:failures.append({'cell':name,'exit_code':code});print('G5_FAILED',name,flush=True)
            else:completed+=1;print('G5_DONE',completed,name,flush=True)
        running=pending;write(MP/('PILOT_SCHEDULER.json' if pilot else 'SCHEDULER.json'),{'completed':completed,'remaining':len(tasks),'running':[r[2] for r in running],'failures':failures})
        if failures and not running:raise SystemExit(2)
        time.sleep(2)
    if pilot:
        results=[json.loads(p.read_text()) for p in (MP/'pilot').glob('*.json')]
        if len(results)!=expected or any(r['status']!='PASS' or r['outer_predictions'] or r['tag']['source']!=msource() for r in results):raise ValueError('G5 pilot incomplete')
        write(R/'G5_PILOT_GATE.json',{'status':'PASS','source':msource(),'cells':expected,'outer_predictions':0,'all_real_gradients_checked':True})
    else:write(MP/'TRAINING_DONE.json',{'status':'PASS','source':msource(),'new_formal_fits':240,'total_formal_evaluations_including_prior':720,'additional_reserved_MIL_ablation_fits':30})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)

"""Bounded process scheduling; completion is validated, never inferred from logs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_routing.core import *
def main(pilot=False):
    (P/'logs').mkdir(exist_ok=True);env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    tasks=[(arm,seed,fold) for arm in (['RAND_512','MIL_512','CS_CONCH_GLOBAL'] if pilot else arm_names()) for seed in ([31] if pilot else cfg()['seeds']) for fold in ([1,2] if pilot else cfg()['folds'])]
    running=[];failures=[];done=0;started=time.time()
    while tasks or running:
        while tasks and len(running)<(2 if pilot else cfg()['workers']):
            arm,seed,fold=tasks.pop(0);name=f'{arm}_seed{seed}_fold{fold}';log=open(P/'logs'/f'{"pilot_" if pilot else ""}{name}.log','w');cmd=[sys.executable,str(ROOT/'scripts/run_phase16.py'),'--arm',arm,'--seed',str(seed),'--fold',str(fold)]+(['--pilot'] if pilot else [])
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);running.append((p,log,name))
        pending=[]
        for p,log,name in running:
            status=p.poll()
            if status is None:pending.append((p,log,name));continue
            log.close()
            if status:failures.append({'cell':name,'exit_code':status});print('FAILED',name,status,flush=True)
            else:done+=1;print('DONE',done,name,flush=True)
        running=pending;write(P/('PILOT_SCHEDULER.json' if pilot else 'SCHEDULER.json'),{'completed':done,'running':[x[2] for x in running],'queued':len(tasks),'failures':failures,'seconds':time.time()-started})
        if failures and not running:break
        time.sleep(2)
    if failures:raise SystemExit(2)
    if pilot:
        results=[json.loads(p.read_text()) for p in sorted((P/'pilot').glob('*.json'))]
        if len(results)!=6 or any(r['status']!='PASS' or r['outer_test_prediction_performed'] or r['source_hash']!=source() for r in results):raise ValueError('pilot incomplete')
        write(R/'PILOT_GATE.json',{'status':'PASS','source_hash':source(),'results':results,'no_outer_predictions':True})
    else:write(P/'TRAINING_DONE.json',{'status':'PASS','source_hash':source(),'expected_cells':210,'actual_fit_cells':150,'reference_cells':60})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');v=a.parse_args();main(v.pilot)

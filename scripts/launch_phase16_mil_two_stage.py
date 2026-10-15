import sys,os,subprocess,time,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_phase16_mil_two_stage import LP,lsource,write
env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def invoke(task):
    budget,seed,fold,pilot=task;(LP/'logs').mkdir(parents=True,exist_ok=True)
    with (LP/'logs'/f'{"pilot_" if pilot else ""}{budget}_{seed}_{fold}.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'scripts/run_phase16_mil_two_stage.py'),'--budget',str(budget),'--seed',str(seed),'--fold',str(fold)]+(['--pilot'] if pilot else []),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('TWO_STAGE_DONE',task,flush=True)
def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(invoke,[(512,31,f,True) for f in [1,2]]))
    write(LP/'PILOT_DONE.json',{'status':'PASS','source':lsource(),'cells':2,'outer_predictions':0})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(invoke,[(m,s,f,False) for m in [128,512] for s in [31,47,83] for f in [1,2,3,4,5]]))
    write(LP/'TRAINING_DONE.json',{'status':'PASS','source':lsource(),'new_outer_fits':30,'extra_StageA_role_fits':60,'total_with_other_Phase16_modules':750})
if __name__=='__main__':main()

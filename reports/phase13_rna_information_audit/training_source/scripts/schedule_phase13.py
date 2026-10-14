"""Bounded parallel complete pipelines; stop on any failed scientific cell."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.phase13_rna.data import PRIVATE,REPORT,config,source_hash,write_json


def main():
    cfg=config();gate=json.loads((REPORT/'HARD_GATES.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=source_hash():raise RuntimeError('gate not satisfied')
    jobs=[]
    for ep in cfg['endpoints']:
        for fold in cfg['folds']:jobs.append(['--family','BASE','--endpoint',ep,'--fold',str(fold)])
    base=jobs[:];rna=[];fusion=[]
    for gene in cfg['genes']:
        for seed in cfg['seeds']:
            for fold in cfg['folds']:
                fusion.append(['--family','FUSION','--gene',gene,'--seed',str(seed),'--fold',str(fold)])
                for ep in cfg['endpoints']:rna.append(['--family','RNA','--endpoint',ep,'--gene',gene,'--seed',str(seed),'--fold',str(fold)])
    env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    logs=PRIVATE/'logs';logs.mkdir(exist_ok=True);telemetry=[]
    for family,queue in [('BASE',base),('ENCODING',fusion+rna)]:
        active=[];completed=0;failed=[];number=0
        while queue or active:
            while queue and len(active)<cfg['execution_workers']:
                args=queue.pop(0);number+=1;path=logs/f'{family}_{number}.log';fp=path.open('w');p=subprocess.Popen([sys.executable,'scripts/run_phase13.py',*args],cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT);active.append((p,args,fp));print('START',family,p.pid,args,flush=True)
            for p,args,fp in active[:]:
                if p.poll() is not None:
                    fp.close();active.remove((p,args,fp));completed+=1;print('EXIT',family,args,p.returncode,flush=True)
                    if p.returncode:failed.append({'args':args,'exit':p.returncode});queue.clear()
            try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
            except Exception as error:gpu=str(error)
            telemetry.append({'time':time.time(),'family':family,'active':len(active),'gpu':gpu})
            write_json(PRIVATE/'SCHEDULER_STATE.json',{'family':family,'completed_jobs':completed,'pending_jobs':len(queue),'active':[{'pid':p.pid,'args':args} for p,args,fp in active],'failed':failed,'records':len(list((PRIVATE/'formal/cells').glob('*.json')))});write_json(PRIVATE/'GPU_TELEMETRY.json',telemetry)
            if active:time.sleep(10)
        if failed:write_json(PRIVATE/'PIPELINE_FAILURE.json',{'family':family,'failed':failed});return 2
    print('ALL_DIAGNOSTIC_RECORDS_COMPLETE',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

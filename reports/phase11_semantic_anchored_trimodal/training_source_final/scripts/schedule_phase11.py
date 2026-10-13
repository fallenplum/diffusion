"""Stage-ordered, bounded concurrency; fail closed on a failed formal cell."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_phase11 import CONFIG,PRIVATE,REPORT,features_for,sources_hash
from src.phase10_trimodal.data import load_private,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args();cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    out=PRIVATE/'formal';logs=out/'scheduler_logs';logs.mkdir(parents=True,exist_ok=True)
    d=load_private('/root/phase7a_foundation_headroom_v1/private_dataset.npz')
    for fold in cfg['folds']:features_for(d,fold)
    del d
    env=os.environ.copy();env.update({'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PHASE11_THREADS':'2'})
    telemetry=[]
    for family in cfg['family_order']:
        arms=[v for v in cfg['arms'] if v.startswith(family)] if family!='P' else ['P0','P3']+list(cfg['patch_arms'])
        queue=[(s,f) for s in cfg['seeds'] for f in cfg['folds']];active=[];done=[];failed=[]
        # 1024 activation inputs are large; preflight determines safe concurrency.
        workers=a.workers if family!='P' else min(a.workers,4)
        while queue or active:
            while queue and len(active)<workers:
                s,f=queue.pop(0);fp=(logs/f'{family}_seed{s}_fold{f}.log').open('a')
                cmd=[sys.executable,'scripts/run_phase11.py','--arms',','.join(arms),'--seed',str(s),'--fold',str(f)]
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT);active.append((proc,s,f,fp));print('START',family,s,f,proc.pid,flush=True)
            for proc,s,f,fp in active[:]:
                if proc.poll() is not None:
                    fp.close();active.remove((proc,s,f,fp));row={'seed':s,'fold':f,'exit':proc.returncode};(done if proc.returncode==0 else failed).append(row);print('EXIT',family,row,flush=True)
                    if proc.returncode:queue.clear()
            try:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
                telemetry.append({'unix_time':time.time(),'family':family,'GPU':gpu,'active_jobs':len(active)})
            except Exception as ex:telemetry.append({'error':str(ex)})
            write_json(out/'SCHEDULER_STATE.json',{'family':family,'done':done,'failed':failed,'active':[{'seed':s,'fold':f,'pid':p.pid} for p,s,f,fp in active],'pending':queue,'source_hash':sources_hash()})
            write_json(out/'GPU_TELEMETRY.json',telemetry)
            if active:time.sleep(10)
        if failed:return 2
        result=subprocess.call([sys.executable,'scripts/validate_phase11_independent.py','--family',family],cwd=ROOT,env=env)
        if result:return result
        (REPORT/f'{family}_STAGE_STATUS.md').write_text(f'# Family {family}\n\nAll {len(arms)*15} records persisted; independent raw OOF/library verification passed. Later model list unchanged.\n')
    return subprocess.call([sys.executable,'scripts/validate_phase11_independent.py','--statistics'],cwd=ROOT,env=env)
if __name__=='__main__':raise SystemExit(main())

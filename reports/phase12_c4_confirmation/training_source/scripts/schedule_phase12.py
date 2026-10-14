"""Locked family order, measured GPU memory limits, durable failure handling."""
import json,os,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_phase12 import config,PRIVATE,REPORT,sources_hash
from src.phase10_trimodal.data import write_json


def families(cfg):
    return {'REUSE':[a for a,s in cfg['arms'].items() if s.get('reuse')],
       'A128':['N_128','W_128'],'B512':['R0_512','N_512','W_512'],'B1024':['R0_1024','N_1024','W_1024'],
       'C':['G'+str(g)+'_128' for g in range(1,8)],'D128':['H1_128','HR_128','MR_128'],
       'D512':['H0_512','H1_512','HR_512','MR_512'],'E':['J_128','JN_128'],'RNA':['RNA_H1','RNA_MR']}


def main():
    cfg=config();gate=json.loads((REPORT/'HARD_GATES.json').read_text());lock=json.loads((REPORT/'PROTOCOL_LOCK.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=sources_hash():raise RuntimeError('formal gates fail')
    groups=families(cfg)
    if set(sum(groups.values(),[]))!=set(cfg['arms']) or sum(map(len,groups.values()))!=len(cfg['arms']):raise RuntimeError('matrix coverage')
    out=PRIVATE/'formal';logdir=out/'scheduler_logs';logdir.mkdir(parents=True,exist_ok=True);telemetry=[]
    env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    for family in cfg['family_order']:
        arms=groups[family];budget=max(cfg['arms'][a]['budget'] for a in arms);workers=lock['execution_workers'].get(str(budget),8)
        amendment=REPORT/'EXECUTION_AMENDMENT.json'
        if amendment.exists():
            change=json.loads(amendment.read_text())
            if change['training_source_hash']!=sources_hash() or change['changes_model_or_training_schedule']:raise RuntimeError('invalid resource amendment')
            workers=min(workers,change['worker_caps'].get(str(budget),workers))
        if family=='REUSE':workers=5
        if family=='RNA':workers=4
        queue=[(seed,fold) for seed in cfg['seeds'] for fold in cfg['folds']];active=[];done=[];failed=[]
        while queue or active:
            while queue and len(active)<workers:
                seed,fold=queue.pop(0);fp=(logdir/f'{family}_seed{seed}_fold{fold}.log').open('a');cmd=[sys.executable,'scripts/run_phase12.py','--arms',','.join(arms),'--seed',str(seed),'--fold',str(fold)]
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT);active.append((process,seed,fold,fp));print('START',family,seed,fold,process.pid,flush=True)
            for proc,seed,fold,fp in active[:]:
                if proc.poll() is not None:
                    fp.close();active.remove((proc,seed,fold,fp));item={'seed':seed,'fold':fold,'exit':proc.returncode};(failed if proc.returncode else done).append(item);print('EXIT',family,item,flush=True)
                    if proc.returncode:queue.clear()
            try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip();telemetry.append({'unix_time':time.time(),'family':family,'active':len(active),'GPU':gpu})
            except Exception as ex:telemetry.append({'error':str(ex)})
            write_json(out/'SCHEDULER_STATE.json',{'family':family,'done':done,'failed':failed,'pending':queue,'active':[{'pid':p.pid,'seed':s,'fold':f} for p,s,f,fp in active],'source_hash':sources_hash()});write_json(out/'GPU_TELEMETRY.json',telemetry)
            if active:time.sleep(10)
        if failed:return 2
        status=subprocess.call([sys.executable,'scripts/validate_phase12_independent.py','--arms',','.join(arms)],cwd=ROOT,env=env)
        if status:return status
        (REPORT/f'{family}_STATUS.md').write_text(f'# {family}\n\nAll{len(arms)*15} records persisted and independently recomputed; later locked models unchanged.\n')
    return subprocess.call([sys.executable,'scripts/validate_phase12_independent.py','--statistics'],cwd=ROOT,env=env)


if __name__=='__main__':raise SystemExit(main())

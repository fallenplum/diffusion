"""Locked formal run. Scores are evaluated only after this queue has finished."""
import sys,os,time,json,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17_ssd.data import P,R,cfg,source,write

def launch(kind,arm,seed,fold,stage):
    stem=f'{arm}_seed{seed}_fold{fold}';logpath=P/'logs'/f'formal_{kind}_{stage}_{stem}.log';handle=logpath.open('w')
    script='scripts/run_phase17_paper.py' if kind=='paper' else 'scripts/run_phase17.py';flag='--model' if kind=='paper' else '--arm'
    args=[sys.executable,script,flag,arm,'--seed',str(seed),'--fold',str(fold),'--stage',stage]
    proc=subprocess.Popen(args,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy());print('START',kind,stage,stem,proc.pid,flush=True)
    return {'kind':kind,'arm':arm,'seed':seed,'fold':fold,'stage':stage,'proc':proc,'handle':handle,'log':str(logpath)}

def complete(kind,arm,seed,fold,stage):
    base=P/(('paper_'+stage) if kind=='paper' else stage);path=base/f'{arm}_seed{seed}_fold{fold}.json'
    if not path.exists():return False
    q=json.loads(path.read_text());assert q['status']=='COMPLETE' and q['source']==source();return True

def main():
    lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());assert lock['training_source']==source();start=time.time()
    priority=['A2','A3','A5','A6','A7','A8','A10','A4','A9','A11_NO_S32','A11_NO_CLIN']
    core=[('core',a,s,f) for f in cfg()['folds'] for s in cfg()['seeds'] for a in priority]
    paper=[('paper',a,s,f) for f in cfg()['folds'] for s in cfg()['seeds'] for a in ['MGCT','MambaMIL']]
    live=[];done=[];fail=[];monitor=[];last_gpu=0
    while core or paper or live:
        # One paper slot, three S6 slots; freed slots are shared at queue tails.
        while len(live)<4 and (core or paper):
            haspaper=any(v['kind']=='paper' for v in live)
            if paper and not haspaper:job=paper.pop(0)
            elif core:job=core.pop(0)
            else:break
            kind,arm,seed,fold=job
            if complete(kind,arm,seed,fold,'outer'):done.append(f'{arm}_seed{seed}_fold{fold}');continue
            stage='outer' if complete(kind,arm,seed,fold,'inner') else 'inner';live.append(launch(kind,arm,seed,fold,stage))
        for job in live[:]:
            code=job['proc'].poll()
            if code is None:continue
            job['handle'].close();live.remove(job)
            if code:
                fail.append({k:v for k,v in job.items() if k not in ['proc','handle']});fail[-1]['exit']=code;print('FAIL',fail[-1],flush=True)
            elif job['stage']=='inner':live.append(launch(job['kind'],job['arm'],job['seed'],job['fold'],'outer'))
            else:
                stem=f"{job['arm']}_seed{job['seed']}_fold{job['fold']}";done.append(stem);print('PAIR_DONE',stem,len(done),flush=True)
        if time.time()-last_gpu>15:
            q=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if q.returncode==0:
                memory,util=[float(v.strip()) for v in q.stdout.split(',')];monitor.append({'unix_time':time.time(),'GPU_memory_MB':memory,'GPU_utilization_percent':util})
            last_gpu=time.time()
        write(P/'FORMAL_PROGRESS.json',{'source':source(),'start_unix':start,'completed_outer_pairs':len(done),'expected_outer_pairs':195,'remaining_core':len(core),'remaining_paper':len(paper),'running':[{k:v for k,v in q.items() if k not in ['proc','handle']} for q in live],'failures':fail})
        if fail:core=[];paper=[]
        if live:time.sleep(3)
    import pandas as pd
    pd.DataFrame(monitor).to_csv(R/'DEVICE_MONITOR.csv',index=False)
    write(P/'FORMAL_TRAINING_DONE.json',{'status':'FAIL' if fail else 'PASS','source':source(),'completed_outer_pairs':len(done),'expected_outer_pairs':195,'failures':fail,'wall_hours':(time.time()-start)/3600})
    if fail:raise SystemExit(1)

if __name__=='__main__':main()

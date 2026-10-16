"""Resumable cell scheduler; never starts outer evaluation before protocol lock."""
import sys,json,os,time,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17_ssd.data import P,R,cfg,source,write

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--stage',choices=['pilot','inner','outer'],default='pilot');pa.add_argument('--workers',type=int,default=3);a=pa.parse_args()
    arms=[s for s in cfg()['arms'] if s not in ['A0','A1']]
    seeds=[31] if a.stage=='pilot' else cfg()['seeds'];folds=[1,2] if a.stage=='pilot' else cfg()['folds']
    queue=[(arm,seed,fold) for fold in folds for seed in seeds for arm in arms];live=[];done=[];failed=[]
    while queue or live:
        while queue and len(live)<a.workers:
            arm,seed,fold=queue.pop(0);stem=f'{arm}_seed{seed}_fold{fold}';cell=P/a.stage/(stem+'.json')
            if cell.exists():
                z=json.loads(cell.read_text());assert z['status']=='COMPLETE' and z['source']==source();done.append(stem);continue
            log=P/'logs'/f'{a.stage}_{stem}.log';handle=log.open('w')
            command=[sys.executable,str(ROOT/'scripts/run_phase17.py'),'--stage',a.stage,'--arm',arm,'--seed',str(seed),'--fold',str(fold)]
            job=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy());live.append((job,handle,stem))
            print('START',a.stage,stem,job.pid,flush=True)
        for job,handle,stem in live[:]:
            code=job.poll()
            if code is None:continue
            handle.close();live.remove((job,handle,stem))
            if code:failed.append({'cell':stem,'exit':code});print('FAILED',stem,code,flush=True)
            else:done.append(stem);print('DONE',stem,flush=True)
        write(P/f'{a.stage}_PROGRESS.json',{'stage':a.stage,'source':source(),'completed':len(done),'running':[v[2] for v in live],'queued':len(queue),'failures':failed})
        if failed:
            # Finish already running authorized cells, do not fan out a broken run.
            queue=[]
        if live:time.sleep(5)
    write(P/f'{a.stage}_DONE.json',{'status':'FAIL' if failed else 'PASS','source':source(),'completed':len(done),'failures':failed})

if __name__=='__main__':main()

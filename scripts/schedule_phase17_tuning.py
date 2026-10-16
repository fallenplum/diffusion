import sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17_ssd.data import P,write,source
queue=[(arm,f) for f in [1,2] for arm in ['A3','A7','A8']];live=[];rows=[]
while queue or live:
    while queue and len(live)<4:
        arm,fold=queue.pop(0);log=(P/'logs'/f'tune_T1_{arm}_fold{fold}.log').open('w')
        args=[sys.executable,'scripts/tune_phase17_inner.py','--arm',arm,'--fold',str(fold)]
        job=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);live.append((job,log,arm,fold))
    for job,log,arm,fold in live[:]:
        code=job.poll()
        if code is not None:log.close();rows.append({'arm':arm,'fold':fold,'exit':code});live.remove((job,log,arm,fold));print('TUNE_DONE',arm,fold,code,flush=True)
    if live:time.sleep(5)
write(P/'TUNING_DONE.json',{'source':source(),'cells':rows,'status':'PASS' if all(r['exit']==0 for r in rows) else 'FAIL'})

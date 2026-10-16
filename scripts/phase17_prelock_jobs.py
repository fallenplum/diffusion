import sys,subprocess,os,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17_ssd.data import P,write,source

jobs=[]
for fold in [1,2]:
    args=['scripts/run_phase17.py','--stage','repair_inner','--arm','A8','--fold',str(fold)]
    log=(P/'logs'/f'repair_A8_fold{fold}.log').open('w');jobs.append((args,log,subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)))
for name in ['MGCT','MambaMIL']:
    args=['scripts/run_phase17_paper.py','--stage','pilot','--model',name]
    log=(P/'logs'/f'paper_pilot_{name}.log').open('w');jobs.append((args,log,subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)))
rows=[]
for args,log,proc in jobs:
    code=proc.wait();log.close();rows.append({'args':args,'exit':code});print('PRELOCK_JOB',args,code,flush=True)
write(P/'PRELOCK_JOBS_DONE.json',{'source':source(),'jobs':rows})

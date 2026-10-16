import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17_ssd.data import P
jobs=[]
for fold in [1,2]:
    log=(P/'logs'/f'film_recheck_fold{fold}.log').open('w');job=subprocess.Popen([sys.executable,'scripts/run_phase17.py','--stage','repair_inner','--arm','A4','--fold',str(fold)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);jobs.append((job,log,fold))
for job,log,fold in jobs:
    code=job.wait();log.close();print('FILM_RECHECK',fold,code,flush=True)
    if code:raise SystemExit(code)

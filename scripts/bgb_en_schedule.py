"""Bounded formal BG-BEN inner matrix; no automatic retries or hidden fits."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--max-workers',type=int,default=2);p.add_argument('--stage',choices=['R1','R2','R3','all'],default='all')
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda')
    args=p.parse_args();args.private.mkdir(parents=True,exist_ok=True);logs=args.private/'logs';logs.mkdir(exist_ok=True)
    if not 1<=args.max_workers<=4:raise ValueError('Measured worker ceiling is four')
    config=json.loads((ROOT/'configs/bgb_en/prototype_v1.json').read_text())
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    base=[sys.executable,str(ROOT/'scripts/bgb_en_run.py'),'--assets',str(args.assets),'--private',str(args.private),'--device',args.device]
    def unit(stage,arm,partition,seed):
        label=f'{stage}_p{partition}_{arm}_s{seed}';command=base+['--stage',stage,'--arm',arm,'--partition',str(partition),'--seed',str(seed)]
        if args.resume:command+=['--resume']
        start=time.monotonic()
        with (logs/(label+'.log')).open('ab' if args.resume else 'xb') as stream:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        print(json.dumps(dict(unit=label,returncode=result.returncode,seconds=time.monotonic()-start)),flush=True)
        if result.returncode:raise RuntimeError('Formal unit failed: '+label)
        return dict(unit=label,returncode=0,elapsed_seconds=time.monotonic()-start)
    start=time.monotonic();receipts=[]
    try:
        for partition in range(3):receipts.append(unit('prepare','P0',partition,31))
        stages=['R1','R2','R3'] if args.stage=='all' else [args.stage]
        for stage in stages:
            arms=config[stage]['arms'][:]
            if stage=='R3':arms+=config['R3']['additional_real_retraining']
            tasks=[]
            for arm in arms:
                deterministic=arm in ('P0','O0','O1','F0','F1') or arm.startswith('O2_')
                seeds=[31] if deterministic else [31,47]
                for partition in range(3):
                    for seed in seeds:tasks.append((stage,arm,partition,seed))
            print(json.dumps(dict(stage=stage,planned_unique_units=len(tasks),workers=args.max_workers)),flush=True)
            # Independent mechanisms keep running if a different unit fails.
            failures=[]
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                fs={pool.submit(unit,*task):task for task in tasks}
                for future in as_completed(fs):
                    try:receipts.append(future.result())
                    except Exception as exc:failures.append(str(exc));print(str(exc),flush=True)
            if failures:
                write=dict(status='PARTIAL_FAILED_UNITS_CONTINUE_INDEPENDENT_MECHANISMS',failures=failures)
                (args.private/(stage+'_SCHEDULE.json')).write_text(json.dumps(write,indent=2)+'\n')
            else:
                (args.private/(stage+'_SCHEDULE.json')).write_text(json.dumps(dict(status='COMPLETE',unique_units=len(tasks)),indent=2)+'\n')
        status='FINISHED_CHECK_UNIT_FAILURES'
    except Exception as exc:
        status='FAILED_DEPENDENCY_NO_AUTOMATIC_RETRY';print(type(exc).__name__+': '+str(exc),flush=True)
    receipt=dict(status=status,elapsed_seconds=time.monotonic()-start,workers=args.max_workers,units=receipts)
    (args.private/'SCHEDULER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()

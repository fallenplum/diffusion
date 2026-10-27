"""Execute a finite explicit sensitivity plan after the primary queue finishes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.bgb_en.data import sha256
from scripts.bgb_en_run import write


def validate_plan(value):
    if value.get('status')!='LOCKED_BEFORE_ADDITIONAL_FITS' or value.get('outer_test_evaluated') is not False:
        raise ValueError('A frozen explicit inner-only plan is required')
    tasks=value['tasks'];labels=set()
    if not tasks:raise ValueError('Empty plan')
    for task in tasks:
        name=Path(task['script'])
        if len(name.parts)!=2 or name.parts[0]!='scripts' or not name.name.startswith('bgb_en_') or name.suffix!='.py':
            raise ValueError('Plan script outside the BG-BEN namespace')
        if task['label'] in labels or not task['label'].replace('_','').isalnum():raise ValueError('Duplicate or unsafe unit label')
        if not isinstance(task['args'],list) or not all(isinstance(x,str) for x in task['args']):raise ValueError('Arguments must be an explicit list')
        if any(x in task['args'] for x in ('--dry-run','--stage=report')):raise ValueError('A qualification is not a formal optimizer fit')
        labels.add(task['label'])
    return tasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--logs',required=True,type=Path)
    p.add_argument('--wait-for-core',required=True,type=Path);p.add_argument('--max-workers',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if not 1<=args.max_workers<=4:raise ValueError('Qualified worker ceiling is four')
    plan=json.loads(args.plan.read_text());tasks=validate_plan(plan)
    sources={task['script']:sha256(ROOT/task['script']) for task in tasks}
    contract=dict(plan_sha256=sha256(args.plan),scripts=sources,scheduler_sha256=sha256(__file__),max_workers=args.max_workers)
    if args.dry_run:print(json.dumps(dict(status='PASS_EXPLICIT_PLAN_NO_FIT',tasks=len(tasks),contract=contract)));return
    args.logs.mkdir(parents=True,exist_ok=True)
    receipt_path=args.logs/'BATCH_RECEIPT.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if not args.resume or old['contract']!=contract:raise ValueError('A completed batch exists or its plan changed')
        if old['status']=='COMPLETE':return
        raise ValueError('Investigate failed units before creating a new bounded recovery plan')
    print(json.dumps(dict(status='WAITING_PRIMARY_QUEUE',tasks=len(tasks),contract=contract)),flush=True)
    while not (args.wait_for_core/'SCHEDULER_RECEIPT.json').is_file():time.sleep(15)
    for stage in ('R1','R2','R3'):
        if json.loads((args.wait_for_core/(stage+'_SCHEDULE.json')).read_text()).get('status')!='COMPLETE':
            raise ValueError('Primary queue has unresolved failed units; do not relabel the matrix complete')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    start=time.monotonic();rows=[]
    def unit(task):
        if sha256(ROOT/task['script'])!=sources[task['script']]:raise ValueError('Planned driver changed after batch start')
        command=[sys.executable,str(ROOT/task['script'])]+task['args']
        if args.resume and '--resume' not in command:command+=['--resume']
        began=time.monotonic()
        with (args.logs/(task['label']+'.log')).open('ab' if args.resume else 'xb') as handle:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        row=dict(unit=task['label'],returncode=result.returncode,seconds=time.monotonic()-began)
        print(json.dumps(row),flush=True);return row
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures={pool.submit(unit,task):task['label'] for task in tasks}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as exc:rows.append(dict(unit=futures[future],returncode=-1,error_type=type(exc).__name__))
            write(args.logs/'BATCH_PROGRESS.json',dict(contract=contract,completed_units=len(rows),planned_units=len(tasks),units=rows))
    complete=len(rows)==len(tasks) and all(row['returncode']==0 for row in rows)
    write(receipt_path,dict(status='COMPLETE' if complete else 'PARTIAL_FAILED_UNITS_NO_AUTOMATIC_RETRIES',contract=contract,
        elapsed_seconds=time.monotonic()-start,completed_units=len(rows),planned_units=len(tasks),units=rows))


if __name__=='__main__':main()

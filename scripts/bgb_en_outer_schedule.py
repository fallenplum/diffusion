"""Bounded outer pipelines with no metric-based scheduler decisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_run import write
from src.bgb_en.data import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--primary-private',type=Path,required=True);p.add_argument('--selection-lock',type=Path,required=True)
    p.add_argument('--inner-report',type=Path,required=True);p.add_argument('--stage',choices=['baselines','nested_neural'],required=True)
    p.add_argument('--max-workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--wait-for-batch',type=Path,action='append',default=[])
    args=p.parse_args()
    if not 1<=args.max_workers<=4:raise ValueError('GPU worker ceiling is four')
    if args.stage=='baselines' and args.max_workers>2:raise ValueError('Concurrent CPU anchor fit ceiling is two')
    lock=json.loads(args.selection_lock.read_text());args.private.mkdir(parents=True,exist_ok=True)
    if lock['status']!='LOCKED_BEFORE_OUTER_EVALUATION' or lock['outer_test_evaluated'] is not False:raise ValueError('Missing immutable finalist lock')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    logs=args.private/'logs';logs.mkdir(exist_ok=True);rows=[];start=time.monotonic()
    def run(label,script,argv):
        command=[sys.executable,str(ROOT/'scripts'/script)]+[str(x) for x in argv]
        if args.resume:command+=['--resume']
        began=time.monotonic()
        with (logs/(label+'.log')).open('ab' if args.resume else 'xb') as stream:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        row=dict(unit=label,returncode=code,seconds=time.monotonic()-began);rows.append(row)
        print(json.dumps(row),flush=True)
        if code:raise RuntimeError('Outer pipeline unit failed: '+label)
    common=['--assets',args.assets,'--private',args.private,'--selection-lock',args.selection_lock,'--inner-report',args.inner_report]
    if args.stage=='baselines':
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures=[pool.submit(run,f'baseline_fold{fold}','bgb_en_outer.py',common+['--fold',fold]) for fold in range(1,6)]
            for future in as_completed(futures):future.result()
    else:
        if 'NESTED_INTERACTION' not in lock['finalists']:
            write(args.private/'NESTED_NEURAL_SCHEDULE.json',dict(status='NOT_QUALIFIED_BY_PRELOCKED_PROTOTYPE_GATE',actual_fits=0));return
        for dependency in args.wait_for_batch:
            while not dependency.exists():time.sleep(15)
            if json.loads(dependency.read_text())['status']!='COMPLETE':raise ValueError('A preceding locked batch has unresolved failures')
        # Seal every within-outer architecture choice before any final neural
        # model is evaluated. Baseline risk outputs are not selection inputs.
        for fold in range(1,6):
            run(f'nested_inner_fold{fold}','bgb_en_nested.py',
                ['--assets',args.assets,'--private',args.private,'--primary-private',args.primary_private,
                 '--global-selection-lock',args.selection_lock,'--inner-report',args.inner_report,'--fold',fold,
                 '--stage','select' if fold==1 else 'inner','--max-workers',args.max_workers,'--device','cuda'])
        for fold in range(1,6):
            path=args.private/'outer'/f'fold{fold}'/'COMPLETE.json'
            while not path.exists():time.sleep(10)
            if json.loads(path.read_text())['status']!='COMPLETE_OUTER_BASELINES':raise ValueError('Shared final anchors are not complete')
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures=[pool.submit(run,f'neural_fold{fold}_seed{seed}','bgb_en_final_neural.py',
                common+['--primary-private',args.primary_private,'--fold-lock',args.private/'selection'/f'outer{fold}.json',
                        '--fold',fold,'--seed',seed,'--device','cuda']) for fold in range(1,6) for seed in (31,47,83)]
            for future in as_completed(futures):future.result()
    write(args.private/('BASELINE_SCHEDULE.json' if args.stage=='baselines' else 'NESTED_NEURAL_SCHEDULE.json'),
        dict(status='COMPLETE',stage=args.stage,workers=args.max_workers,units=rows,elapsed_seconds=time.monotonic()-start,
            lock_sha256=sha256(args.selection_lock),script_sha256=sha256(__file__),scheduler_reads_outer_metrics_for_decisions=False))


if __name__=='__main__':main()

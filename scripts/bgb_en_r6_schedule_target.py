"""Finite 54 primary + 36 mandatory R6 target fits; no architecture search."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r6_common import (parser,context,qualified_prepared,
    validate_qualification,configure_deterministic_runtime,write,require,
    public_snapshot_guard,TARGET_ARMS)


def main():
    p=parser(__doc__);p.add_argument('--max-workers',type=int,default=4)
    p.add_argument('--source-decision',type=Path,required=True)
    args=p.parse_args()
    require(1<=args.max_workers<=4,'R6_TARGET_WORKER_CEILING_FOUR')
    configure_deterministic_runtime();torch.set_num_threads(2)
    ctx=context(args);validate_qualification(args,ctx)
    old_reports=public_snapshot_guard();start=time.monotonic()
    logs=args.private/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    common=[]
    for option in ('assets','private','primary_private','source_private','config','source_config'):
        common+=['--'+option.replace('_','-'),str(getattr(args,option))]
    common+=['--device',args.device]
    if args.resume:common+=['--resume']
    records=[]

    def execute(label,script,extra):
        begin=time.monotonic()
        with (logs/(label+'.log')).open('ab' if args.resume else 'xb') as handle:
            code=subprocess.run([sys.executable,str(ROOT/'scripts'/script)]+common+extra,
                cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
        record=dict(unit=label,returncode=code,elapsed_seconds=time.monotonic()-begin)
        records.append(record);print(json.dumps(record),flush=True)
        require(code==0,'R6_TARGET_SUBPROCESS_FAILED_NO_AUTO_RETRY')

    try:
        # References are explicitly serialized before the concurrent cells.
        for part in (0,1,2):
            qualified_prepared(args,ctx,part)
            execute('references_p'+str(part),'bgb_en_r6_run.py',
                ['--mode','references','--partition',str(part)])
        for group,arms in (('PRIMARY',TARGET_ARMS[:6]),('MANDATORY_CONTROLS',TARGET_ARMS[6:])):
            pool=ThreadPoolExecutor(max_workers=args.max_workers);futures=[]
            try:
                for arm in arms:
                    for part in (0,1,2):
                        for seed in (31,47,83):
                            futures.append(pool.submit(execute,f'{arm}_p{part}_seed{seed}',
                                'bgb_en_r6_run.py',['--mode','fit','--arm',arm,
                                '--partition',str(part),'--seed',str(seed)]))
                for future in as_completed(futures):future.result()
            except Exception:
                for future in futures:future.cancel()
                pool.shutdown(wait=True,cancel_futures=True);raise
            else:pool.shutdown(wait=True)
            write(args.private/(group+'_SCHEDULE_COMPLETE.json'),dict(status='COMPLETE',
                group=group,arms=list(arms),contract_sha256=ctx['lock']['contract_sha256']))
        execute('actual_checkpoint_replay','bgb_en_r6_replay.py',[])
        command=[sys.executable,str(ROOT/'scripts/bgb_en_r6_report_target.py'),
            '--private-root',str(args.private),'--config',str(args.config),
            '--source-decision',str(args.source_decision),
            '--output',str(ROOT/'reports/bgb_en_r6')]
        with (logs/'report_target.log').open('ab' if args.resume else 'xb') as handle:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
        require(code==0,'R6_TARGET_REPORT_FAILED_NO_MODEL_RETRY')
        require(public_snapshot_guard()==old_reports,'R0_R4_REPORTS_CHANGED_DURING_R6')
        write(args.private/'TARGET_SCHEDULE_COMPLETE.json',dict(status='COMPLETE_90_R6_TARGET_FITS_AND_REPLAY',
            contract_sha256=ctx['lock']['contract_sha256'],records=records,
            elapsed_seconds=time.monotonic()-start,worker_ceiling=args.max_workers,
            new_unplanned_candidates=0,optional_controls_excluded_from_this_90_unit_denominator=True))
    except Exception as exc:
        write(args.private/'TARGET_SCHEDULE_ERROR.json',dict(status='STOPPED_NO_AUTOMATIC_RETRY',
            error_type=type(exc).__name__,records=records,elapsed_seconds=time.monotonic()-start,
            contract_sha256=ctx['lock']['contract_sha256']))
        raise


if __name__=='__main__':main()

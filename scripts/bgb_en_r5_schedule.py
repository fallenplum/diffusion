"""Finite R5 63+45 matrix; no score-dependent concurrency or extra searches."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r5_common import (parser,context,qualified_prepared,validate_qualification,
    cell_name,cell_contract,sha256,require,write,public_snapshot_guard)
from scripts.bgb_en_r5_run import make_references
from src.bgb_en.r5_protocol import MAIN_ARMS,MECHANISM_CONTROLS


def main():
    p=parser(__doc__);p.add_argument('--stage',choices=('all','main','mechanism'),default='all')
    p.add_argument('--max-workers',type=int,default=4);args=p.parse_args()
    require(1<=args.max_workers<=4,'FORMAL_WORKER_CEILING_FOUR')
    torch.set_num_threads(2);ctx=context(args);qualification=validate_qualification(args,ctx)
    before=public_snapshot_guard();start=time.monotonic();logs=args.private/'logs';logs.mkdir(exist_ok=True)
    origins={};records=[]
    for part in (0,1,2):
        prepared,origin=qualified_prepared(args,ctx,part);origins[part]=origin
        reference_args=argparse.Namespace(**vars(args));reference_args.partition=part
        make_references(reference_args,ctx,prepared,origin)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    common=['--assets',str(args.assets),'--private',str(args.private),'--primary-private',str(args.primary_private)]

    def process(label,script,extra):
        command=[sys.executable,str(ROOT/'scripts'/script)]+common+extra
        if args.resume:command+=['--resume']
        began=time.monotonic()
        with (logs/(label+'.log')).open('ab' if args.resume else 'xb') as stream:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        record=dict(unit=label,returncode=code,elapsed_seconds=time.monotonic()-began)
        records.append(record);print(json.dumps(record),flush=True)
        if code:raise RuntimeError('R5_SUBPROCESS_FAILED_'+label)

    def report(stage):
        label='report_'+stage
        command=[sys.executable,str(ROOT/'scripts/bgb_en_r5_report.py'),'--private',str(args.private),
                 '--stage',stage,'--output',str(ROOT/'reports/bgb_en_r5')]
        with (logs/(label+'.log')).open('ab' if args.resume else 'xb') as stream:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        require(code==0,'R5_REPORT_FAILED_'+stage)

    def batch(group,tasks):
        todo=[];reused=0
        for arm,part,seed,control in tasks:
            label=cell_name(arm,part,seed,control);directory=args.private/'cells'/label
            done=directory/'COMPLETE.json';contract=cell_contract(ctx,origins[part],arm,part,seed,control,qualification)
            if done.exists():
                value=json.loads(done.read_text())
                require(args.resume and value['status']=='COMPLETE_R5_FIT' and value['contract']==contract
                        and sha256(directory/'model.pt')==value['artifact_sha256'],'COMPLETED_UNIT_REUSE_CHANGED')
                reused+=1;continue
            require(not directory.exists(),'UNFINISHED_OR_FAILED_UNIT_REQUIRES_REVIEW')
            todo.append((arm,part,seed,control,label))
        pool=ThreadPoolExecutor(max_workers=args.max_workers);futures=[]
        try:
            for arm,part,seed,control,label in todo:
                futures.append(pool.submit(process,label,'bgb_en_r5_run.py',
                    ['--stage','train','--arm',arm,'--partition',str(part),'--seed',str(seed),'--control',control,'--device',args.device]))
            for future in as_completed(futures):future.result()
        except Exception:
            for future in futures:future.cancel()
            pool.shutdown(wait=True,cancel_futures=True)
            raise
        else:pool.shutdown(wait=True)
        write(args.private/(group+'_BATCH.json'),dict(status='COMPLETE',group=group,planned=len(tasks),
            newly_run=len(todo),completed_reused=reused,worker_ceiling=args.max_workers,
            contract_sha256=ctx['lock']['contract_sha256']))

    try:
        if args.stage in ('all','main'):
            tasks=[(arm,part,seed,'MATCHED') for arm in MAIN_ARMS for part in (0,1,2) for seed in (31,47,83)]
            batch('MAIN',tasks);report('main')
            audit=json.loads((ROOT/'reports/bgb_en_r5/R5_REPORT_AUDIT.json').read_text())
            selected=dict(audit['proposed_selection'],contract_sha256=ctx['lock']['contract_sha256'])
            require(selected['status']=='LOCKED_AFTER_COMPLETE_MAIN_BEFORE_MECHANISM','MAIN_SELECTION_NOT_COMPLETE')
            path=args.private/'SELECTION.json'
            if path.exists():require(json.loads(path.read_text())==selected,'EXISTING_FINALIST_CHANGED')
            else:write(path,selected)
        if args.stage in ('all','mechanism'):
            selected=json.loads((args.private/'SELECTION.json').read_text())
            require(selected['contract_sha256']==ctx['lock']['contract_sha256'],'SELECTION_SOURCE_CHANGED')
            tasks=[(selected['selected_arm'],part,seed,control) for control in MECHANISM_CONTROLS
                   for part in (0,1,2) for seed in (31,47,83)]
            batch('MECHANISM',tasks)
            process('independent_replay','bgb_en_r5_replay.py',['--device',args.device])
            report('final')
        require(before==public_snapshot_guard(),'OLD_R0_R4_REPORT_CHANGED')
        write(args.private/'SCHEDULE_COMPLETE.json',dict(status='COMPLETE_R5_ALL' if args.stage=='all' else 'COMPLETE_R5_'+args.stage.upper(),
            contract_sha256=ctx['lock']['contract_sha256'],units=records,elapsed_seconds=time.monotonic()-start,
            requested_stage=args.stage,worker_ceiling=args.max_workers,new_unplanned_candidates=0))
    except Exception as exc:
        write(args.private/'SCHEDULE_ERROR.json',dict(status='STOPPED_WITH_ERRORS_NO_AUTOMATIC_RETRY',
            error_type=type(exc).__name__,units=records,elapsed_seconds=time.monotonic()-start,
            contract_sha256=ctx['lock']['contract_sha256']))
        raise


if __name__=='__main__':main()

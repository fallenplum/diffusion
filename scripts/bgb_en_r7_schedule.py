"""Finite 99-cell R7 scheduler; resource-qualified ceiling, no fit retries."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from collections import deque
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r7_common import (parser,context,qualified_prepared,validate_qualification,
    configure_deterministic_runtime,write,read,require,sha256,public_snapshot_guard,TARGET_ARMS,PARTITIONS,SEEDS)


def resource_snapshot(private):
    result=dict(UTC=datetime.now(timezone.utc).isoformat(),disk_free_bytes=shutil.disk_usage(private).free)
    try:
        import psutil
        ram=psutil.virtual_memory();result.update(RAM_total_bytes=ram.total,RAM_available_bytes=ram.available)
    except ImportError:result['RAM_measurement']='PSUTIL_UNAVAILABLE'
    try:
        command=['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
        output=subprocess.run(command,capture_output=True,text=True,timeout=10,check=True).stdout.strip().splitlines()
        result['GPU']=[dict(zip(('total_MiB','used_MiB','free_MiB','utilization_percent'),[float(v.strip()) for v in row.split(',')])) for row in output]
    except (OSError,subprocess.SubprocessError,ValueError):result['GPU_measurement']='UNAVAILABLE'
    return result


def qualified_workers(args,ctx,qualification_sha):
    initial=min(args.max_workers,4)
    path=args.private/'RESOURCE_QUALIFICATION.json'
    if args.max_workers<=4 or not path.exists():return initial
    value=read(path)
    require(value.get('status')=='PASS_R7_CONCURRENT_RESOURCE_QUALIFICATION'
        and value.get('contract_sha256')==ctx['lock']['contract_sha256']
        and value.get('qualification_sha256')==qualification_sha
        and value.get('observed_real_forward_backward') is True
        and value.get('observed_concurrent_workers',0)>=4
        and value.get('qualified_max_workers') in (5,6),
        'R7_INVALID_REAL_RESOURCE_QUALIFICATION')
    measured=value.get('resource_measurements',{})
    require(value.get('largest_two_round_arm_measured') is True,'R7_LARGEST_ARM_RESOURCE_PROBE_REQUIRED')
    extra=max(value['qualified_max_workers']-value['observed_concurrent_workers'],0)
    for kind in ('GPU','RAM'):
        total=measured.get(kind+'_total_bytes');used=measured.get(kind+'_peak_used_bytes_at_observed_workers')
        bound=measured.get('per_extra_worker_'+kind+'_bytes_upper_bound')
        require(all(type(x) in (int,float) and x>0 for x in (total,used,bound))
            and used+extra*bound<=.9*total,'R7_MEASURED_'+kind+'_HEADROOM_INSUFFICIENT')
    return min(args.max_workers,value['qualified_max_workers'])


def run(args):
    require(1<=args.max_workers<=6,'R7_WORKER_CEILING_SIX')
    configure_deterministic_runtime();torch.set_num_threads(2);ctx=context(args)
    qualification=validate_qualification(args,ctx);historical=public_snapshot_guard();start=time.monotonic()
    logs=args.private/'logs';logs.mkdir(exist_ok=True);stop=threading.Event();records=[];record_lock=threading.Lock()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    options=[]
    for name in ('assets','private','primary_private','source_private','config','source_config','r6_private','r6_config'):
        options+=['--'+name.replace('_','-'),str(getattr(args,name))]
    options+=['--device',args.device]
    if args.resume:options+=['--resume']
    monitor_file=args.private/'RESOURCE_MONITOR.jsonl'
    def monitor():
        with monitor_file.open('a',encoding='utf-8') as handle:
            while not stop.is_set():
                handle.write(json.dumps(resource_snapshot(args.private),allow_nan=False)+'\n');handle.flush()
                stop.wait(15)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    def execute(label,script,extra):
        require(shutil.disk_usage(args.private).free>=512*1024**2,'R7_DISK_HEADROOM_REQUIRED_BEFORE_NEW_PROCESS')
        begin=time.monotonic()
        with (logs/(label+'.log')).open('ab' if args.resume else 'xb') as handle:
            code=subprocess.run([sys.executable,str(ROOT/'scripts'/script)]+options+extra,cwd=ROOT,env=env,
                stdout=handle,stderr=subprocess.STDOUT).returncode
        row=dict(unit=label,returncode=code,elapsed_seconds=time.monotonic()-begin)
        with record_lock:records.append(row)
        print(json.dumps(row),flush=True);require(code==0,'R7_SUBPROCESS_FAILED_NO_AUTOMATIC_RETRY');return row
    try:
        for part in PARTITIONS:
            qualified_prepared(args,ctx,part)
            execute('references_p'+str(part),'bgb_en_r7_run.py',['--mode','references','--partition',str(part)])
        pending=deque((arm,part,seed) for arm in TARGET_ARMS for part in PARTITIONS for seed in SEEDS)
        pool=ThreadPoolExecutor(max_workers=6);active={}
        try:
            while pending or active:
                allowed=qualified_workers(args,ctx,qualification)
                while pending and len(active)<allowed:
                    arm,part,seed=pending.popleft();name=f'{arm}_p{part}_seed{seed}'
                    future=pool.submit(execute,name,'bgb_en_r7_run.py',['--mode','fit','--arm',arm,'--partition',str(part),'--seed',str(seed)])
                    active[future]=name
                complete,_=wait(active,timeout=15,return_when=FIRST_COMPLETED)
                for future in complete:future.result();del active[future]
                write(args.private/'SCHEDULE_PROGRESS.json',dict(status='RUNNING',active_units=list(active.values()),pending_units=len(pending),
                    allowed_workers=allowed,completed_fit_processes=sum(r['unit'].split('_p')[0] in TARGET_ARMS for r in records),
                    contract_sha256=ctx['lock']['contract_sha256'],elapsed_seconds=time.monotonic()-start))
        except BaseException:
            for future in active:future.cancel()
            pool.shutdown(wait=True,cancel_futures=True);raise
        else:pool.shutdown(wait=True)
        execute('actual_all_checkpoint_replay','bgb_en_r7_replay.py',[])
        command=[sys.executable,str(ROOT/'scripts/bgb_en_r7_report.py'),'--private-root',str(args.private),
            '--config',str(args.config),'--output',str(ROOT/'reports/bgb_en_r7')]
        with (logs/'report.log').open('ab' if args.resume else 'xb') as handle:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
        require(code==0,'R7_REPORT_FAILED_NO_MODEL_RETRY')
        require(public_snapshot_guard()==historical,'R7_HISTORICAL_REPORTS_CHANGED_BY_SCHEDULE')
        result=dict(status='COMPLETE_99_R7_TARGET_FITS_AND_297_CHECKPOINT_REPLAYS',contract_sha256=ctx['lock']['contract_sha256'],
            records=records,elapsed_seconds=time.monotonic()-start,worker_ceiling=args.max_workers,initial_worker_ceiling=min(4,args.max_workers),
            formal_fit_count=99,formal_optimizer_updates=7920,unplanned_candidates=0,performance_adaptive_budget=False)
        write(args.private/'SCHEDULE_COMPLETE.json',result);return result
    except BaseException as exc:
        write(args.private/'SCHEDULE_ERROR.json',dict(status='STOPPED_NO_AUTOMATIC_RETRY',error_type=type(exc).__name__,records=records,
            elapsed_seconds=time.monotonic()-start,contract_sha256=ctx['lock']['contract_sha256']))
        raise
    finally:stop.set();watcher.join(timeout=20)


def main():
    p=parser(__doc__);p.add_argument('--max-workers',type=int,default=4);args=p.parse_args();run(args)


if __name__=='__main__':main()

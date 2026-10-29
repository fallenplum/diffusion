"""Finite S5-A GPU queue with resource-only concurrency and zero automatic retries.

Exactly 84 fits, followed by exactly 84 replay processes (252 saved states).
Concurrency changes never inspect scientific scores or choose experiments.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import psutil

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_name(path.name+".writing")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,required=True);p.add_argument("--private",type=Path,required=True)
    p.add_argument("--config",type=Path,default=ROOT/"configs/spgel_s5/mechanism_v1.json")
    p.add_argument("--workers",type=int,default=4,choices=range(1,9));p.add_argument("--replay-workers",type=int,default=4,choices=range(1,9))
    a=p.parse_args();private=a.private.resolve();lock=read(private/"S5_RUN_LOCK.json")
    config=read(a.config);assert config["budget"]["new_neural_fits"]==84
    assert not (private/"ADMIN_STARTED.private.json").exists(),"EXISTING_QUEUE_NO_AUTOMATIC_RETRY"
    admin=private/"admin";admin.mkdir(exist_ok=False)
    state=dict(schema="spgel_s5_administrative_queue_v1",status="STARTED",controller_pid=os.getpid(),
        completed_fits=0,completed_replay_units=0,failed_units=[],active=[],new_fits_during_replay=0,
        requested_workers=a.workers,requested_replay_workers=a.replay_workers,contract_sha256=lock["contract_sha256"])
    write(private/"ADMIN_STARTED.private.json",dict(config_sha256=sha(a.config),lock_sha256=sha(private/"S5_RUN_LOCK.json"),
        workers=a.workers,replay_workers=a.replay_workers,controller_pid=os.getpid(),start_unix=time.time()))
    guard=threading.Lock();stop=threading.Event();samples=[];started=time.monotonic()
    def monitor():
        while not stop.is_set():
            try:
                gpu=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")
                memory=psutil.virtual_memory();proc=psutil.Process();children=proc.children(recursive=True)
                rss=proc.memory_info().rss+sum(c.memory_info().rss for c in children if c.is_running())
                sample=dict(elapsed_seconds=time.monotonic()-started,GPU_used_MiB=int(gpu[0]),GPU_util_percent=int(gpu[1]),
                    processes_RSS_bytes=rss,available_RAM_bytes=memory.available,CPU_percent=psutil.cpu_percent(),
                    data_disk_free_bytes=psutil.disk_usage(str(private)).free)
                samples.append(sample)
                with guard:state["latest_telemetry"]=sample;write(private/"RUNNING_STATUS.private.json",state)
            except (psutil.Error,subprocess.SubprocessError):pass
            stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    units=[(arm,d,s) for d in range(6) for s in (31,47) for arm in config["arms"]]
    def run(operation,unit):
        arm,d,s=unit;name=f"{arm}_d{d}_seed{s}"
        command=[sys.executable,str(ROOT/"scripts/spgel_s5.py"),operation,"--assets",str(a.assets),"--private",str(private),
            "--config",str(a.config),"--arm",arm,"--donor",str(d),"--seed",str(s),"--device","cuda"]
        t=time.monotonic()
        with (admin/f"{operation}_{name}.log").open("xb") as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            with guard:state["active"].append(dict(operation=operation,unit=name,pid=process.pid))
            result=process.wait()
        with guard:
            state["active"]=[x for x in state["active"] if x["pid"]!=process.pid]
            if result==0:state["completed_fits" if operation=="fit" else "completed_replay_units"]+=1
            else:state["failed_units"].append(dict(operation=operation,unit=name,exit_code=result))
            write(private/"RUNNING_STATUS.private.json",state)
        return dict(operation=operation,unit=name,exit_code=result,elapsed_seconds=time.monotonic()-t)
    results=[]
    try:
        for operation,workers in (("fit",a.workers),("replay",a.replay_workers)):
            with guard:state["status"]="RUNNING_84_FITS" if operation=="fit" else "REPLAYING_252_ACTUAL_STATES"
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures={pool.submit(run,operation,u):u for u in units}
                for future in as_completed(futures):
                    row=future.result();results.append(row)
                    print(json.dumps(row),flush=True)
                    if row["exit_code"]!=0:
                        for pending in futures:pending.cancel()
                        raise RuntimeError("S5_FAILED_CELL_NO_AUTOMATIC_RETRY")
            assert not state["failed_units"]
        assert state["completed_fits"]==state["completed_replay_units"]==84
        state["status"]="COMPLETE_84_FITS_252_ACTUAL_REPLAYS"
    except BaseException as error:
        state.update(status="PARTIAL_FAILED_NO_AUTOMATIC_RETRY",failure_type=type(error).__name__)
        (admin/"controller_error.private.log").write_text(traceback.format_exc());raise
    finally:
        stop.set();thread.join(timeout=7)
        state["elapsed_seconds"]=time.monotonic()-started
        write(private/"RUNNING_STATUS.private.json",state)
        write(private/"ADMIN_EXECUTION.private.json",dict(state=state,results=results,telemetry=samples,new_fits_during_replay=0))


if __name__=="__main__":main()

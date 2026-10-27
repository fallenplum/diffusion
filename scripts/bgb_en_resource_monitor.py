"""Anonymous resource samples; no process command lines or patient inputs."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time


def cpu_ticks():
    values=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    return sum(values[:8]),values[3]+values[4]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--stop-file',required=True,type=Path)
    p.add_argument('--interval',type=float,default=10);p.add_argument('--maximum-seconds',type=float,default=14400);args=p.parse_args()
    if args.interval<5:raise ValueError('Monitor sampling interval must be at least five seconds')
    args.output.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();old=cpu_ticks()
    with args.output.open('x',newline='') as handle:
        columns=['elapsed_seconds','GPU_util_percent','GPU_memory_MiB','GPU_total_MiB','CPU_util_percent','RAM_available_MiB','disk_available_MiB']
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        while time.monotonic()-start<args.maximum_seconds and not args.stop_file.exists():
            result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            gpu=[float(x) for x in result.stdout.splitlines()[0].split(',')]
            current=cpu_ticks();total,idle=current[0]-old[0],current[1]-old[1];old=current
            memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            row=dict(elapsed_seconds=round(time.monotonic()-start,3),GPU_util_percent=gpu[0],GPU_memory_MiB=gpu[1],GPU_total_MiB=gpu[2],
                CPU_util_percent=100*(1-idle/total) if total else None,RAM_available_MiB=memory['MemAvailable']/1024,
                disk_available_MiB=shutil.disk_usage(args.output.parent).free/1024**2)
            writer.writerow(row);handle.flush();time.sleep(args.interval)
    print(json.dumps(dict(status='MONITOR_STOPPED',elapsed_seconds=time.monotonic()-start)))


if __name__=='__main__':main()

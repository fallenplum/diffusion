"""Detach a source-locked finite scheduler on the authorized server."""
import json
import os
import subprocess
import sys
from src.phase21_relative.data import ROOT,P,source,write

def main():
    if (P/'STARTED.json').exists():
        q=json.loads((P/'STARTED.json').read_text())
        try: os.kill(q['pid'],0)
        except ProcessLookupError: pass
        else: raise ValueError('scheduler active')
    write(P/'WORKERS.json',{'workers':8})
    env=dict(os.environ,PYTHONPATH='.:/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log=(P/'logs/scheduler.log').open('a')
    p=subprocess.Popen([sys.executable,'scripts/schedule_phase21.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(P/'STARTED.json',{'pid':p.pid,'source':source(),'workers':8});print('PHASE21_STARTED',p.pid,flush=True)

if __name__=='__main__': main()

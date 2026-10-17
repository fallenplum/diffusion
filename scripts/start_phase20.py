"""Launch the qualified finite queue on the authorized GPU server."""
import json
import os
import subprocess
import sys
from src.phase20_selective.data import P,ROOT,source,write


def main():
    P.mkdir(parents=True,exist_ok=True);(P/'logs').mkdir(exist_ok=True)
    if (P/'STARTED.json').exists():
        record=json.loads((P/'STARTED.json').read_text())
        try: os.kill(record['pid'],0)
        except ProcessLookupError: pass
        else: raise ValueError('Phase20 scheduler already active; inspect progress')
    if not (P/'WORKERS.json').exists(): write(P/'WORKERS.json',{'workers':6})
    env=dict(os.environ,PYTHONPATH='.:/root/phase10_trimodal_eval_deps',
             OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log=(P/'logs/scheduler.log').open('a')
    child=subprocess.Popen([sys.executable,'scripts/schedule_phase20.py'],cwd=ROOT,env=env,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(P/'STARTED.json',{'pid':child.pid,'source':source(),'default_workers':6})
    print('PHASE20_SCHEDULER_PID',child.pid,flush=True)


if __name__=='__main__': main()

"""Detach the authorized server run with durable PID and logs."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];private=Path('/root/phase11_semantic_anchor_private')
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PHASE11_THREADS':'2'})
with (private/'full_run.log').open('a') as fp:
    p=subprocess.Popen([sys.executable,'scripts/schedule_phase11.py','--workers','12'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
(private/'full_run.pid').write_text(str(p.pid));print(json.dumps({'pid':p.pid,'log':str(private/'full_run.log')}))

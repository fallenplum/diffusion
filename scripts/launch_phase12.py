"""Detach only the explicitly authorized gate and formal experiment chain."""
import subprocess,sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/root/phase12_c4_private')
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
with (P/'full_run.log').open('a') as log:
    process=subprocess.Popen([sys.executable,'scripts/run_phase12_pipeline.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(P/'full_run.pid').write_text(str(process.pid));print(json.dumps({'pid':process.pid,'log':str(P/'full_run.log')}))

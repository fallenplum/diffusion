"""Run sequential gates then bounded formal scheduling and independent audit."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for script,args in [('gate_phase16.py',[]),('launch_phase16.py',[]),('validate_phase16_independent.py',[])]:
    print('START',script,flush=True);subprocess.run([sys.executable,str(ROOT/'scripts'/script)]+args,cwd=ROOT,env=env,check=True);print('FINISHED',script,flush=True)

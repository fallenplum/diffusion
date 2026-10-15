import sys,time,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
while not Path('/root/phase16_omics_private/TRAINING_DONE.json').exists():time.sleep(2)
subprocess.run([sys.executable,str(ROOT/'scripts/validate_phase16_omics_independent.py')],cwd=ROOT,env=env,check=True)
print('G5_INDEPENDENT_STAGE_DONE',flush=True)

"""Wait for formal completion, then fail closed through independent audits."""
import sys,time,json,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_semantic.core import P
env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
while not (P/'FIT_STAGE_DONE.json').exists():time.sleep(2)
for script in ['validate_phase16_semantic_independent.py','audit_phase16_semantic_retrieval.py','replay_phase16_semantic_queries.py','ablate_phase16_semantic_inference.py','launch_phase16_omics.py']:
    args=['--pilot'] if script=='launch_phase16_omics.py' else [];print('START',script,flush=True);subprocess.run([sys.executable,str(ROOT/'scripts'/script)]+args,cwd=ROOT,env=env,check=True);print('FINISHED',script,flush=True)
subprocess.run([sys.executable,str(ROOT/'scripts/launch_phase16_omics.py')],cwd=ROOT,env=env,check=True)
print('ALL_G5_FITS_DONE',flush=True)

"""Detached complete preflight and all locked diagnostic training pipelines."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PRIVATE=Path('/root/phase13_rna_private')
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
code="""
import json,subprocess,sys
from pathlib import Path
private=Path('/root/phase13_rna_private')
for script in ['prepare_phase13.py','gate_phase13.py','schedule_phase13.py']:
    print('RUN',script,flush=True)
    result=subprocess.call([sys.executable,'scripts/'+script])
    if result:
        (private/'PIPELINE_FAILURE.json').write_text(json.dumps({'script':script,'exit':result}))
        raise SystemExit(result)
print('ALL_FORMAL_RECORDS_COMPLETE_READY_FOR_INDEPENDENT_EVALUATION',flush=True)
"""
with (PRIVATE/'full_run.log').open('w') as log:
    process=subprocess.Popen([sys.executable,'-u','-c',code],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'status':'LAUNCHED','pid':process.pid,'log':str(PRIVATE/'full_run.log')}))

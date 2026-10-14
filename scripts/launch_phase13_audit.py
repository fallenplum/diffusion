import os,sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/root/phase13_rna_private')
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
code="""
import sys,subprocess,json
from pathlib import Path
p=Path('/root/phase13_rna_private')
for script in ['validate_phase13_independent.py','replay_phase13.py','audit_phase13_statistics.py']:
    print('RUN',script,flush=True)
    result=subprocess.call([sys.executable,'scripts/'+script])
    if result:
        (p/'AUDIT_FAILURE.json').write_text(json.dumps({'script':script,'exit':result}))
        raise SystemExit(result)
print('ALL_INDEPENDENT_AUDITS_COMPLETE',flush=True)
"""
with (P/'independent_audit.log').open('w') as log:proc=subprocess.Popen([sys.executable,'-u','-c',code],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'status':'AUDITS_LAUNCHED','pid':proc.pid}))

"""Start the authorized final audit chain; keep its process/log persistent."""
import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];private=Path('/root/phase11_semantic_anchor_private')
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PHASE11_THREADS':'2'})
with (private/'final_audit.log').open('a') as fp:
    p=subprocess.Popen([sys.executable,'scripts/finish_phase11.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
(private/'final_audit.pid').write_text(str(p.pid));print(json.dumps({'pid':p.pid,'log':str(private/'final_audit.log')}))

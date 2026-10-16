import sys,os,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17r_ssd.data import P,write
env=os.environ.copy();env.update(PYTHONPATH='.:/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
if (P/'COMPLETION_PID.json').exists():
    q=json.loads((P/'COMPLETION_PID.json').read_text())
    try:os.kill(q['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('completion process already alive')
with (P/'COMPLETION.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'scripts/complete_phase17r.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
write(P/'COMPLETION_PID.json',{'pid':p.pid});print('COMPLETION_STARTED',p.pid)

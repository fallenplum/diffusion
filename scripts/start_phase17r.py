import sys,os,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17r_ssd.data import P,R,source,write
assert json.loads((R/'PROTOCOL_LOCK.json').read_text())['source']==source()
env=os.environ.copy();env.update(PYTHONPATH='.:/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
if (P/'SCHEDULER_PID.json').exists():
    prior=json.loads((P/'SCHEDULER_PID.json').read_text())
    try:os.kill(prior['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('scheduler already alive')
with (P/'SCHEDULER.log').open('a') as log:
    p=subprocess.Popen([sys.executable,'scripts/schedule_phase17r.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
write(P/'SCHEDULER_PID.json',{'pid':p.pid,'source':source()});print('STARTED',p.pid,source())

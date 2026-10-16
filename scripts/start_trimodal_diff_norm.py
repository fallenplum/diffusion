import os,subprocess,sys,time
from src.trimodal_interaction.data import P,ROOT,write

if '--worker' in sys.argv:
    while not (P/'budget160/DONE.json').exists():time.sleep(5)
    raise SystemExit(subprocess.call([sys.executable,'scripts/repair_trimodal_diff_norm.py'],cwd=ROOT))
env=os.environ.copy();env.update(PYTHONPATH='.:/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
h=(P/'A3F_SCHEDULER.log').open('a');p=subprocess.Popen([sys.executable,__file__,'--worker'],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env=env,start_new_session=True)
write(P/'A3F_SCHEDULER_PID.json',{'pid':p.pid});print('A3F_QUEUED',p.pid)

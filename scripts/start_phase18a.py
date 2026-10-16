import os,subprocess,sys
from src.phase18a_readout.data import P,ROOT,write
env=os.environ.copy();env.update(PYTHONPATH='.:/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
h=(P/'SCHEDULER.log').open('a');p=subprocess.Popen([sys.executable,'scripts/schedule_phase18a.py'],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env=env,start_new_session=True);write(P/'SCHEDULER_PID.json',{'pid':p.pid});print('STARTED',p.pid)

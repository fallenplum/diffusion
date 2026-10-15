"""One-command reproduction of the currently executable, explicitly limited scope."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
steps=[('audit_phase16.py',[]),('prepare_phase16.py',[]),('coordinate_audit_phase16.py',[]),('pin_phase16_inputs.py',[]),('launch_phase16.py',['--pilot']),('execute_phase16.py',[]),('pin_phase16_inputs.py',[]),('budget_diagnostic_phase16.py',[]),('components_phase16.py',[]),('profile_phase16.py',[]),('reconcile_phase16_scope.py',[]),('report_phase16.py',[])]+([('publish_phase16.py',[])] if '--publish' in sys.argv else [])
for script,args in steps:
    print('PIPELINE',script,flush=True);subprocess.run([sys.executable,str(ROOT/'scripts'/script)]+args,cwd=ROOT,env=env,check=True)

"""Wait for full matrix; run final audits and package without human polling."""
import os,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/root/phase12_c4_private');R=ROOT/'reports/phase12_c4_confirmation'
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
while True:
    if (R/'ALL_INDEPENDENT_VALIDATION.json').exists() and (R/'BOOTSTRAP_AUDIT.json').exists():
        result=json.loads((R/'ALL_INDEPENDENT_VALIDATION.json').read_text())
        if result['status']!='PASS' or result['verified_records']!=465:raise RuntimeError('all OOF not verified')
        break
    # Resource-only retries may resume a failed scheduler; wait, never claim
    # completion from progress files or partial results.
    time.sleep(10)
for script in ['replay_phase12.py','audit_phase12_statistics.py','audit_phase12_test_sensitivity.py','report_phase12.py','package_phase12.py','check_phase12_delivery.py']:
    print('RUN',script,flush=True);code=subprocess.call([sys.executable,'scripts/'+script],cwd=ROOT,env=env)
    if code:
        (P/'FINAL_AUDIT_FAILURE.json').write_text(json.dumps({'script':script,'exit':code}));raise SystemExit(code)
print('COMPLETE_AUDITED_PRIVATE_PUBLIC_PACKAGED',flush=True)

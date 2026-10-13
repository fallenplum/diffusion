"""Finish required audits after all locked cells and statistics exist."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/root/phase11_semantic_anchor_private');R=ROOT/'reports/phase11_semantic_anchored_trimodal'
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PHASE11_THREADS':'2'})
while True:
    state=P/'formal/SCHEDULER_STATE.json'
    if state.exists() and json.loads(state.read_text()).get('failed'):
        raise RuntimeError('scheduler failed; final audit must not claim COMPLETE')
    if (R/'ALL_INDEPENDENT_VALIDATION.json').exists() and (R/'BOOTSTRAP_AUDIT.json').exists():
        a=json.loads((R/'ALL_INDEPENDENT_VALIDATION.json').read_text())
        if a['status']!='PASS' or a['verified_records']!=450:raise RuntimeError('formal independent OOF gate failed')
        break
    time.sleep(10)
for script in ['replay_phase11.py','audit_phase11_statistics.py','audit_phase11_anchor_comparisons.py','report_phase11.py','interpret_phase11.py','package_phase11.py','check_phase11_final_contract.py']:
    print('RUN',script,flush=True)
    status=subprocess.call([sys.executable,'scripts/'+script],cwd=ROOT,env=env)
    if status:
        (P/'FINAL_AUDIT_FAILURE.json').write_text(json.dumps({'script':script,'exit':status}));raise SystemExit(status)
print('COMPLETE_AUDITED_PRIVATE_AND_PUBLIC_PACKAGED',flush=True)

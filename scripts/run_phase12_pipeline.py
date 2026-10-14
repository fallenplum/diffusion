"""Persist until gates, raw source audit and all locked training cells finish."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/root/phase12_c4_private')
sys.path.insert(0,str(ROOT/'scripts'))
from run_phase12 import sources_hash,REPORT
for script in ['audit_phase12_sources.py','gate_phase12.py','schedule_phase12.py']:
    if script in ['audit_phase12_sources.py','gate_phase12.py'] and (REPORT/'HARD_GATES.json').exists():
        gate=json.loads((REPORT/'HARD_GATES.json').read_text())
        if gate['status']=='PASS' and gate['source_hash']==sources_hash():
            print('REUSE_VERIFIED_GATE_NUMERICAL_SOURCE_UNCHANGED',script,flush=True);continue
    print('RUN',script,flush=True);status=subprocess.call([sys.executable,'scripts/'+script],cwd=ROOT)
    if status:
        (P/'PIPELINE_FAILURE.json').write_text(json.dumps({'script':script,'exit':status}));raise SystemExit(status)
print('TRAINING_AND_INDEPENDENT_METRICS_COMPLETE',flush=True)

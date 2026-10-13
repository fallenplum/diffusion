"""Stop v1 processes, retain all evidence, start no training by itself."""
import json,os,signal,time
from pathlib import Path
private=Path('/root/phase11_semantic_anchor_private');repo=Path('/root/phase11_semantic_repo')
pid=int((private/'full_run.pid').read_text());state=json.loads((private/'formal/SCHEDULER_STATE.json').read_text())
children=[a['pid'] for a in state.get('active',[])]
for p in [pid]+children:
    try:os.kill(p,signal.SIGTERM)
    except ProcessLookupError:pass
time.sleep(2)
for p in children:
    try:os.kill(p,signal.SIGKILL)
    except ProcessLookupError:pass
archive=private/'invalid_v1_missing_CP_reverse';archive.mkdir()
for name in ['formal','preprocessing','full_run.log','full_run.pid','gates.log']:
    p=private/name
    if p.exists():p.rename(archive/name)
report=repo/'reports/phase11_semantic_anchored_trimodal'
for name in ['HARD_GATES.json','PROTOCOL_LOCK.json','GPU_PREFLIGHT.json','REAL_TEST_PERTURBATION.json','UNIT_TEST_OUTPUT.txt','FAULT_INJECTION.json']:
    p=report/name
    if p.exists():(archive/name).write_bytes(p.read_bytes())
cells=list((archive/'formal/cells').glob('*.json'))
record={'status':'INVALID','reason':'C/S32 to Patch implemented, missing Patch to C/S32 direction required by B0; detected by architecture review before outer result inspection',
        'severity':'P0','completed_records_preserved':len(cells),'outer_metrics_inspected':False,'retrain_scope':'fresh entire phase11 matrix, same data, no Phase10 outputs modified'}
(archive/'INVALID.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))

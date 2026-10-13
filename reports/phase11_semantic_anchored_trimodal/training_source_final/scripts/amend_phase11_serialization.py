"""Record numeric-equivalent A lineage; preserve failed B save artifacts."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_phase11 import sources_hash,PRIVATE,REPORT
from validate_phase11_independent import digest
old=json.loads((REPORT/'HARD_GATES.json').read_text());audit=json.loads((REPORT/'REPLAY_AND_PROVENANCE_AUDIT.json').read_text())
if audit['status']!='PASS' or audit['checkpoints']['actual_fit_checkpoints']!=120:raise RuntimeError('A replay evidence absent')
archive=PRIVATE/'P2_serialization_failure';archive.mkdir(exist_ok=True)
for path in [REPORT/'HARD_GATES.json',REPORT/'PROTOCOL_LOCK.json',REPORT/'REPLAY_AND_PROVENANCE_AUDIT.json',REPORT/'CHECKPOINT_REPLAY.csv',ROOT/'phase11_v2_gate.tar']:
    shutil.copy2(path,archive/path.name)
for directory in ['checkpoints','probes','representations','scheduler_logs']:
    for path in (PRIVATE/'formal'/directory).glob('B*'):
        dest=archive/directory/path.name;dest.parent.mkdir(parents=True,exist_ok=True);path.rename(dest)
record={'severity':'P2','scope':'serialization namespace and explicit source lineage only','prior_source_hash':old['source_hash'],'new_source_hash':sources_hash(),
        'pre_repair_A_checkpoint_audit':'PASS','prior_A_records_preserved':120,'independent_preprocessing_matrices_rebuilt':50,
        'changed_numerical_model_math':False,'old_A_OOF_modified':False,'old_A_cell_metadata_modified':False,
        'A_checkpoint_audit_sha256':digest(archive/'CHECKPOINT_REPLAY.csv'),'archive':str(archive),'B0_retrain_reason':'inner/outer diagnostic state not durably persisted before failed save; no COMPLETE B record',
        'outer_metrics_used_to_choose_repair':False}
(REPORT/'P2_SOURCE_AMENDMENT.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))

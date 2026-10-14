"""Preserve the actual OOM failure; amend concurrency only, never model code."""
import json,shutil,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_phase12 import PRIVATE,REPORT,sources_hash,DATA
from src.phase10_trimodal.data import digest,write_json
source=sources_hash();gate=json.loads((REPORT/'HARD_GATES.json').read_text())
if source!=gate['source_hash']:raise RuntimeError('numerical source changed')
data=json.loads((REPORT/'DATA_AUDIT.json').read_text())
if digest(DATA)!=data['data_hashes']['dataset']:raise RuntimeError('dataset changed')
archive=PRIVATE/'resource_failure_1024_workers5';archive.mkdir(exist_ok=True)
for path in [PRIVATE/'PIPELINE_FAILURE.json',PRIVATE/'full_run.log',PRIVATE/'formal/SCHEDULER_STATE.json',PRIVATE/'formal/GPU_TELEMETRY.json',PRIVATE/'formal/scheduler_logs/B1024_seed31_fold2.log',ROOT/'scripts/schedule_phase12.py',REPORT/'PROTOCOL_LOCK.json',REPORT/'GPU_PREFLIGHT.json']:
    if path.exists():shutil.copy2(path,archive/path.name)
verified=0
for path in (PRIVATE/'formal/cells').glob('*.json'):
    c=json.loads(path.read_text());stem=f"{c['arm']}_seed{c['seed']}_fold{c['fold']}"
    if digest(PRIVATE/'formal/oof'/f'{stem}.npz')!=c['prediction_sha256']:raise RuntimeError('completed OOF changed')
    verified+=1
for arm in ['R0_1024','N_1024','W_1024']:
    stem=arm+'_seed31_fold2'
    if (PRIVATE/'formal/cells'/f'{stem}.json').exists():continue
    for folder in ['checkpoints','oof','manifests','probes','representations']:
        for path in (PRIVATE/'formal'/folder).glob(stem+'.*'):
            if not path.resolve().is_relative_to(PRIVATE.resolve()):raise RuntimeError('unsafe path')
            dest=archive/'incomplete'/folder/path.name;dest.parent.mkdir(parents=True,exist_ok=True);path.rename(dest)
amendment={'status':'LOCKED_RESOURCE_ONLY_AMENDMENT','training_source_hash':source,'changes_model_or_training_schedule':False,'changes_loss_genes_patch_subset_rng_or_selection':False,'worker_caps':{'1024':4},'original_worker_cap_1024':5,'reason':'observed5-process CUDA OOM,failed first R0_1024 inner fitseed31fold2;192MiB allocation with162MiB free','outer_performance_used':False,'completed_records_unchanged_hash_verified':verified,'archive':str(archive),'timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
write_json(REPORT/'EXECUTION_AMENDMENT.json',amendment)
journal=PRIVATE/'BUG_JOURNAL.json';entries=json.loads(journal.read_text()) if journal.exists() else []
entries.append({'severity':'P1_RESOURCE','stage':'B1024_concurrency','bug':'5process memory preflight margin insufficient for actual Cox train','affected':'R0_1024 seed31 fold2 has no completed record;pending queue stopped','repair':'archive original logs,cap1024 to4,resume only missing cells;all numerical/source/protocol unchanged','post_hoc':False,'source_hash':source});write_json(journal,entries)
failure=PRIVATE/'PIPELINE_FAILURE.json'
if failure.exists():failure.rename(archive/'PIPELINE_FAILURE.json')
print(json.dumps(amendment))

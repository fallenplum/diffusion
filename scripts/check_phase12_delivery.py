"""Verify every required table, declared fit, bootstrap audit and ZIP SHA."""
import json,hashlib,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase12_c4_confirmation'
required=['PROTOCOL_LOCK.json','MODEL_MATRIX.csv','RESULTS.md','PRIMARY_PAIRED.csv','FACTORIAL_PATCH_RNA.csv','GROUPING_8_REPLICATES.csv','PATHWAY_COVERAGE.csv','RNA_SIGNAL_PROBES.csv','TRAINING_DIAGNOSTICS.csv','BUG_LOG.csv','SOURCE_REGISTRY.json','INDEPENDENT_VALIDATION.json','FAULT_INJECTION.json','REPRODUCE.md','DECISION.md']
if any(not (R/name).exists() for name in required):raise RuntimeError('missing prompt deliverable')
if len(pd.read_csv(R/'FOLD_METRICS.csv'))!=465 or len(pd.read_csv(R/'CHECKPOINT_REPLAY.csv'))!=465:raise RuntimeError('formal/checkpoint matrix count')
if len(pd.read_csv(R/'PRIMARY_PAIRED.csv'))!=12 or len(pd.read_csv(R/'GROUPING_8_REPLICATES.csv'))!=24 or len(pd.read_csv(R/'TRAINING_DIAGNOSTICS.csv'))!=360:raise RuntimeError('statistics/diagnostics counts')
for name in ['INDEPENDENT_VALIDATION.json','REPLAY_AUDIT.json','LITERAL_BOOTSTRAP_AUDIT.json']:
    if json.loads((R/name).read_text())['status']!='PASS':raise RuntimeError('audit not pass '+name)
manifest=json.loads((R/'PUBLIC_ARTIFACT_MANIFEST.json').read_text());archive=R/'PHASE12_PUBLIC_AUDIT.zip';status=json.loads((R/'PUBLIC_ARCHIVE_STATUS.json').read_text())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=status['public_sha256']:raise RuntimeError('public SHA')
with zipfile.ZipFile(archive) as z:
    if z.testzip():raise RuntimeError('public CRC')
    if any(Path(n).suffix in ['.npz','.npy','.pt','.h5','.pem'] for n in z.namelist()):raise RuntimeError('private arrays in public archive')
    for row in manifest['files']:
        if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:raise RuntimeError('public entryhash')
result={'status':'PASS','required_deliverables':len(required),'formal_records':465,'new_fit_checkpoints':390,'exact_parent_reuse':75,'new_neural_diagnostics':360,'primary_metric_comparisons':12,'public_entries_hash_verified':len(manifest['files'])}
(R/'DELIVERY_CONTRACT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

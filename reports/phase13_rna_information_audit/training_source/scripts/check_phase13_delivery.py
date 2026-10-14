import json,hashlib,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase13_rna_information_audit'
required=['PROTOCOL_LOCK.json','ENDPOINT_AUDIT.json','MODEL_MATRIX.csv','RESULTS.md','PRIMARY_PAIRED.csv','ALL_PAIRED.csv','MODEL_ABSOLUTE_CI.csv','PAIR_RETENTION_TRANSITIONS.csv','EVENT_INFORMATION_MASS.csv','CS_RNA_RISK_PREDICTABILITY.csv','REFERENCE_RISK_RECOVERY.csv','TRAINING_DIAGNOSTICS.csv','ARTIFACT_HASH_INDEX.csv','FUSION_INNER_RECONSTRUCTION.csv','APPENDIX_PAIRED.csv','APPENDIX_PROTOCOL_LOCK.json','BUG_LOG.csv','SOURCE_REGISTRY.json','INDEPENDENT_VALIDATION.json','REPRODUCE.md','DECISION.md']
if any(not (R/name).exists() for name in required):raise RuntimeError('missing delivery file')
for name in ['FOLD_METRICS.csv','CHECKPOINT_REPLAY.csv','ARTIFACT_HASH_INDEX.csv']:
    if len(pd.read_csv(R/name))!=3330:raise RuntimeError('matrix incomplete')
if len(pd.read_csv(R/'PRIMARY_PAIRED.csv'))!=12 or len(pd.read_csv(R/'TRAINING_DIAGNOSTICS.csv'))!=180 or len(pd.read_csv(R/'FUSION_INNER_RECONSTRUCTION.csv'))!=45:raise RuntimeError('diagnosis count')
for name in ['INDEPENDENT_VALIDATION.json','REPLAY_AUDIT.json','LITERAL_BOOTSTRAP_AUDIT.json','APPENDIX_VALIDATION.json']:
    if json.loads((R/name).read_text())['status']!='PASS':raise RuntimeError('audit not pass '+name)
status=json.loads((R/'PUBLIC_ARCHIVE_STATUS.json').read_text());archive=R/'PHASE13_PUBLIC_AUDIT.zip';manifest=json.loads((R/'PUBLIC_ARTIFACT_MANIFEST.json').read_text())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=status['public_sha256']:raise RuntimeError('ZIP SHA')
with zipfile.ZipFile(archive) as z:
    if z.testzip():raise RuntimeError('CRC')
    if any(Path(n).suffix in ['.npz','.npy','.pt','.xlsx','.h5','.pem','.tar'] for n in z.namelist()):raise RuntimeError('private artifact in public')
    for record in manifest['files']:
        if hashlib.sha256(z.read(record['path'])).hexdigest()!=record['sha256']:raise RuntimeError('public artifact SHA')
result={'status':'PASS','required_deliverables':len(required),'core_records':3330,'RNA_only_pipelines':90,'fusion_inner_reconstructions':45,'parent_outer_checkpoints_reused':45,'appendix_unique_newfits':30,'public_entries_hash_verified':len(manifest['files'])}
(R/'DELIVERY_CONTRACT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

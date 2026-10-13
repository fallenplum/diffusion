"""Final executable completion/privacy contract, not a self-awarded PASS."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase11_semantic_anchored_trimodal';P=Path('/root/phase11_semantic_anchor_private')
required=['README.md','RESULTS.md','DECISION.md','PROTOCOL_LOCK.json','SOURCE_REGISTRY.json','DATA_AUDIT.md','PATIENT_SLIDE_ALIGNMENT_AUDIT.json','S32_SEMANTIC_SOURCE_AUDIT.md','RNA_PREPROCESS_AND_TOKEN_AUDIT.md','MODEL_MATRIX.csv','FOLD_METRICS.csv','SEED_OOF_METRICS.csv','PAIRED_COMPARISONS.csv','MODULE_ABLATIONS.csv','RNA_SIGNAL_PROBES.csv','PATCH_BUDGET_COMPARISONS.csv','NEGATIVE_CONTROLS.csv','TRAINING_DIAGNOSTICS.csv','ARCHITECTURE_GRADIENT_AUDIT.md','INDEPENDENT_AUDIT.md','INDEPENDENT_VALIDATION.json','FAULT_INJECTION_REPORT.md','BUG_LOG.csv','KNOWN_LIMITATIONS.md','RUN_MANIFEST.json','PUBLIC_ARTIFACT_MANIFEST.json','PUBLIC_AUDIT_PACKAGE_SHA256.txt','RUN_INSTRUCTIONS.md']
missing=[p for p in required if not (R/p).exists()]
if missing:raise RuntimeError('missing requested deliverables: '+str(missing))
import pandas as pd
cfg=json.loads((ROOT/'configs/phase11_semantic_anchored.json').read_text(encoding='utf-8-sig'));fold=pd.read_csv(R/'FOLD_METRICS.csv');pairs=pd.read_csv(R/'PAIRED_COMPARISONS.csv')
if len(fold)!=450 or len(pairs)!=2*(len(cfg['primary_comparisons'])+len(cfg['secondary_comparisons'])):raise RuntimeError('matrix/statistical comparison count')
if len(pd.read_csv(R/'CHECKPOINT_REPLAY.csv'))!=420:raise RuntimeError('actual fit checkpoint count')
if len(pd.read_csv(R/'A1_INDEPENDENT_REFIT.csv'))!=15:raise RuntimeError('independent fresh strong anchor refits missing')
if len(pd.read_csv(R/'TRAINING_DIAGNOSTICS.csv'))!=345:raise RuntimeError('neural training diagnostic count')
if len(pd.read_csv(R/'RNA_SIGNAL_PROBES.csv'))!=144:raise RuntimeError('6 arms x 3 seeds x 8 probes expected')
if len(pd.read_csv(R/'SYSTEM_ANCHOR_COMPARISONS.csv'))!=54:raise RuntimeError('supplementary ALL-model anchor family incomplete')
for name in ['INDEPENDENT_VALIDATION.json','REPLAY_AND_PROVENANCE_AUDIT.json','LITERAL_BOOTSTRAP_AUDIT.json']:
    if json.loads((R/name).read_text())['status']!='PASS':raise RuntimeError('audit FAIL: '+name)
archives=json.loads((R/'PUBLIC_ARCHIVE_STATUS.json').read_text())
public=R/'PHASE11_PUBLIC_AUDIT.zip'
if hashlib.sha256(public.read_bytes()).hexdigest()!=archives['public_sha256']:raise RuntimeError('public package SHA')
with zipfile.ZipFile(public) as z:
    if z.testzip():raise RuntimeError('public package CRC')
    if any(Path(n).suffix in ['.npz','.npy','.pt','.h5','.pem'] for n in z.namelist()):raise RuntimeError('private arrays/checkpoints in PUBLIC')
    manifest=json.loads(z.read('reports/phase11_semantic_anchored_trimodal/PUBLIC_ARTIFACT_MANIFEST.json'))
    for row in manifest['files']:
        if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:raise RuntimeError('public entry SHA '+row['path'])
result={'status':'PASS','deliverables':len(required),'formal_records':450,'actual_fit_checkpoints':420,'neural_training_diagnostics':345,'probe_seed_records':144,'public_archive_entry_hashes_validated':len(manifest['files'])}
(R/'DELIVERY_CONTRACT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

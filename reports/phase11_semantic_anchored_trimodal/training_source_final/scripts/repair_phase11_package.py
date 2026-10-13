"""Preserve a package-only failure; prove formal source/artifacts unchanged."""
import json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_phase11 import sources_hash
from src.phase10_trimodal.data import digest
P=Path('/root/phase11_semantic_anchor_private');R=ROOT/'reports/phase11_semantic_anchored_trimodal'
before=sources_hash();gate=json.loads((R/'HARD_GATES.json').read_text())
if before!=gate['source_hash']:raise RuntimeError('numerical training source changed')
archive=P/'packaging_failure';archive.mkdir(exist_ok=True)
shutil.copy2(P/'final_audit.log',archive/'original_final_audit.log')
failure=P/'FINAL_AUDIT_FAILURE.json'
if failure.exists():shutil.copy2(failure,archive/'original_failure.json')
env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
with (R/'PACKAGE_UNITTEST_RECHECK.txt').open('w') as out:
    status=subprocess.call([sys.executable,'-m','unittest','tests.test_phase11_semantic','-v'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
if status:raise RuntimeError('documented unittest entrypoint failed')
if 'Ran 12 tests' not in (R/'PACKAGE_UNITTEST_RECHECK.txt').read_text():raise RuntimeError('unit test count mismatch')
records=0
for path in (P/'formal/cells').glob('*.json'):
    cell=json.loads(path.read_text());stem=f"{cell['arm']}_seed{cell['seed']}_fold{cell['fold']}"
    if digest(P/'formal/oof'/f'{stem}.npz')!=cell['prediction_sha256']:raise RuntimeError('OOF mutated')
    if cell['actual_fit']:
        checkpoint=P/'formal/checkpoints'/f'{stem}.npz'
        if not checkpoint.exists():checkpoint=checkpoint.with_suffix('.pt')
        if digest(checkpoint)!=cell['checkpoint_sha256']:raise RuntimeError('checkpoint mutated')
    records+=1
after=sources_hash()
if after!=before:raise RuntimeError('training hash mutated during packaging repair')
result={'status':'RESOLVED','severity':'P2','scope':'POST_HOC_PACKAGING_ONLY','outer_values_inspected_before_repair':True,'repair_chosen_from_outer_performance':False,'training_source_hash_before':before,'training_source_hash_after':after,'risk_record_files_hash_verified':records,'formal_checkpoints_unchanged':True,'unit_tests':'12 passed via documented package entrypoint','archive':str(archive)}
(R/'PACKAGING_REPAIR.json').write_text(json.dumps(result,indent=2))
if failure.exists():failure.rename(archive/'FINAL_AUDIT_FAILURE.json')
print(json.dumps(result))

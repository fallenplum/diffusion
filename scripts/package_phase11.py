"""Public allowlist package and server-only private reproducibility archive."""
import hashlib,json,re,subprocess,zipfile,tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase11_semantic_anchored_trimodal';P=Path('/root/phase11_semantic_anchor_private')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def source_files():
    names=['run_phase11.py','schedule_phase11.py','launch_phase11.py','prepare_phase11.py','gate_phase11.py','validate_phase11_independent.py','replay_phase11.py','audit_phase11_sources.py','audit_phase11_statistics.py','report_phase11.py','package_phase11.py','finish_phase11.py','check_phase11_final_contract.py','recheck_phase11_papers.py','read_phase11_history.py','repair_phase11_pre_evaluation.py','amend_phase11_serialization.py']
    names+=['audit_phase10_raw_sources.py','audit_phase10_omics_source.py','launch_phase11_finish.py','interpret_phase11.py','stage_phase11_public.py','audit_phase11_anchor_comparisons.py','repair_phase11_package.py']
    paths=[ROOT/'scripts'/n for n in names]
    paths+=list((ROOT/'src/phase11_semantic').glob('*.py'))
    paths+=[ROOT/'src/phase10_trimodal'/n for n in ['__init__.py','math.py','features.py','data.py']]
    paths+=[ROOT/'configs/phase11_semantic_anchored.json',ROOT/'configs/phase10_molecular_groups.json',ROOT/'tests/test_phase11_semantic.py',ROOT/'tests/__init__.py']
    paths+=[ROOT/'PHASE11_SEMANTIC_ANCHORED_TRIMODAL_CODEX_PROMPT.md']
    return paths

def main():
    manifest=json.loads((R/'RUN_MANIFEST.json').read_text())
    if manifest['status']!='COMPLETE_AUDITED' or manifest['actual_fits']!=420:raise RuntimeError('completion gate not met')
    sources=source_files()
    # Keep exact training versions for the old A lineage and repaired B/C/D.
    current=R/'training_source_final'
    for source in sources:
        dest=current/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
    for directory,archive in [('training_source_v1_INVALID',P/'invalid_v1_missing_CP_reverse/exact_source_v1.tar'),('training_source_v2_prior_A',P/'P2_serialization_failure/phase11_v2_gate.tar')]:
        with tarfile.open(archive) as tar:
            for entry in tar.getmembers():
                if not entry.isfile() or not entry.name.endswith(('.py','.json')):continue
                dest=(R/directory/entry.name).resolve()
                if not dest.is_relative_to(R.resolve()):raise RuntimeError('unsafe archive member')
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(tar.extractfile(entry).read())
    # Version2 prior runner retained exactly in archive tar (server-only); public
    # JSON lineage + unchanged A replay prevents misclaiming one training hash.
    all_files=sources+[p for p in R.rglob('*') if p.is_file() and p.suffix in ['.md','.json','.csv','.txt','.py'] and p.name not in ['PUBLIC_ARTIFACT_MANIFEST.json','PUBLIC_AUDIT_PACKAGE_SHA256.txt','PUBLIC_ARCHIVE_STATUS.json']]
    all_files=sorted(set(all_files));forbidden=re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    violations=[]
    for p in all_files:
        text=p.read_text(encoding='utf8',errors='replace')
        if forbidden.search(text):violations.append(str(p.relative_to(ROOT)))
    if violations:raise RuntimeError('public protected-data pattern detected: '+str(violations))
    write(R/'PUBLIC_ARTIFACT_MANIFEST.json',{'status':'PASS','files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size} for p in all_files],'private_input_included':False})
    all_files.append(R/'PUBLIC_ARTIFACT_MANIFEST.json');public=R/'PHASE11_PUBLIC_AUDIT.zip'
    with zipfile.ZipFile(public,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in all_files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(public) as z:
        if z.testzip():raise RuntimeError('public archive CRC failed')
    public_sha=digest(public);(R/'PUBLIC_AUDIT_PACKAGE_SHA256.txt').write_text(public_sha+'  PHASE11_PUBLIC_AUDIT.zip\n')
    d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True))
    labels=P/'private_labels_and_folds.npz';np.savez_compressed(labels,patient_ids=d['patient_ids'],folds=d['folds'],DSS_time=d['DSS_time'],DSS_event=d['DSS_event'])
    dependencies=[Path('/root/phase7a_foundation_headroom_v1/private_dataset.npz'),P/'cache/patch1024.npy',P/'cache/coords1024.npy',P/'cache/private_slide_manifest.json']
    write(P/'REQUIRED_PRIVATE_INPUTS.json',{'retained_server_dependencies':[{'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size} for p in dependencies],
          'CONCH_source_root':'/root/autodl-tmp/phase10_conch','H5_hashes':'cache/private_slide_manifest.json','notice':'private ZIP has labels/OOF/fitted transforms/checkpoints; raw matrix and large patch cache retained separately, not duplicated'})
    private=P/'PHASE11_PRIVATE_REPRO.zip'
    with zipfile.ZipFile(private,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for folder in [P/'formal',P/'preprocessing',P/'invalid_v1_missing_CP_reverse',P/'P2_serialization_failure',P/'packaging_failure']:
            for p in sorted(folder.rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(P)))
        for p in [labels,P/'REQUIRED_PRIVATE_INPUTS.json',P/'cache/private_slide_manifest.json']:
            z.write(p,str(p.relative_to(P)))
        for p in all_files:z.write(p,'source_and_public/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(private) as z:
        if z.testzip():raise RuntimeError('private archive CRC failed')
        count=len(z.infolist())
    status={'status':'PASS','public_sha256':public_sha,'public_bytes':public.stat().st_size,'public_files':len(all_files),'private_sha256':digest(private),'private_bytes':private.stat().st_size,'private_files':count,'private_path':str(private),'private_pushed_to_Git':False}
    write(R/'PUBLIC_ARCHIVE_STATUS.json',status);write(P/'DONE.json',{'status':'COMPLETE','records':450,'record_files_with_30_alias_bookkeeping':480,'actual_fits':420,'archives':status,'decision':json.loads((R/'decision.json').read_text())['overall']})
    print(json.dumps(status))
if __name__=='__main__':main()

"""Publish only explicit nonidentifying sources and aggregated audit evidence."""
import json,hashlib,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase12_c4_confirmation';P=Path('/root/phase12_c4_private')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def sources():
    names=['run_phase12.py','prepare_phase12.py','audit_phase12_sources.py','gate_phase12.py','validate_phase12_independent.py','replay_phase12.py','schedule_phase12.py','launch_phase12.py','run_phase12_pipeline.py','repair_phase12_resource.py','audit_phase12_statistics.py','audit_phase12_test_sensitivity.py','report_phase12.py','package_phase12.py','check_phase12_delivery.py','finish_phase12.py','launch_phase12_finish.py','stage_phase12_public.py']
    names+=['repair_phase12_audit_device.py','run_phase11.py','replay_phase11.py','validate_phase11_independent.py','gate_phase11.py','audit_phase10_raw_sources.py','audit_phase10_omics_source.py']
    files=[ROOT/'scripts'/name for name in names]
    files+=list((ROOT/'src/phase12_c4').glob('*.py'))+list((ROOT/'src/phase11_semantic').glob('*.py'))
    files+=[ROOT/'src/phase10_trimodal'/name for name in ['__init__.py','math.py','features.py','data.py']]
    files+=[ROOT/'configs'/name for name in ['phase12_c4_confirmation.json','phase11_semantic_anchored.json','phase10_molecular_groups.json']]
    files+=[ROOT/'tests'/name for name in ['__init__.py','test_phase12_c4.py','test_phase11_semantic.py']]
    files+=[ROOT/'src/phase_m2/metadata/mmp_hallmarks_signatures.csv',ROOT/'PHASE12_C4_MECHANISM_CONFIRMATION_CODEX_PROMPT.md']
    return files


def main():
    m=json.loads((R/'RUN_MANIFEST.json').read_text())
    if m['status']!='COMPLETE_AUDITED' or m['new_fits']!=390 or m['records']!=465:raise RuntimeError('not complete')
    files=sources();snapshot=R/'training_source'
    for p in files:
        q=snapshot/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    files=sorted(set(files+[p for p in R.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.txt'] and p.name not in ['PUBLIC_ARTIFACT_MANIFEST.json','PUBLIC_ARCHIVE_STATUS.json','PUBLIC_AUDIT_PACKAGE_SHA256.txt','DELIVERY_CONTRACT.json']]))
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for p in files:
        if pattern.search(p.read_bytes()):raise RuntimeError('patient/secret pattern in PUBLIC '+str(p.relative_to(ROOT)))
    manifest={'status':'PASS','files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size} for p in files],'patient_arrays_checkpoints_raw_WSI_included':False}
    (R/'PUBLIC_ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));files.append(R/'PUBLIC_ARTIFACT_MANIFEST.json');public=R/'PHASE12_PUBLIC_AUDIT.zip'
    with zipfile.ZipFile(public,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(public) as z:
        if z.testzip():raise RuntimeError('public CRC')
    pubsha=digest(public);(R/'PUBLIC_AUDIT_PACKAGE_SHA256.txt').write_text(pubsha+'  PHASE12_PUBLIC_AUDIT.zip\n')
    dependencies={'notice':'historical checkpoints referenced by exact paths/hash,raw matrix andlarge read-only cache retained separately,not duplicated','parent_checkpoint_references':[{'arm':c['arm'],'seed':c['seed'],'fold':c['fold'],'path':c['parent_checkpoint_path'],'sha256':c['checkpoint_sha256']} for p in (P/'formal/cells').glob('*.json') if (c:=json.loads(p.read_text())).get('reuse_parent')],'data':json.loads((R/'DATA_AUDIT.json').read_text()),'raw_dataset':'/root/phase7a_foundation_headroom_v1/private_dataset.npz','cache':'/root/phase11_semantic_anchor_private/cache','H5':'/root/autodl-tmp/phase10_conch'}
    (P/'REQUIRED_PRIVATE_INPUTS.json').write_text(json.dumps(dependencies,indent=2));private=P/'PHASE12_PRIVATE_REPRO.zip'
    with zipfile.ZipFile(private,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for folder in ['formal','preprocessing','fixed_model_sensitivity','resource_failure_1024_workers5','audit_failure_device_comparison','delivery_source_before_review']:
            for p in sorted((P/folder).rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(P)))
        for p in [P/'REQUIRED_PRIVATE_INPUTS.json',P/'BUG_JOURNAL.json']:
            if p.exists():z.write(p,str(p.relative_to(P)))
        for p in files:z.write(p,'public_and_sources/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(private) as z:
        if z.testzip():raise RuntimeError('private CRC')
        private_entries=len(z.infolist())
    result={'status':'PASS','public_sha256':pubsha,'public_bytes':public.stat().st_size,'public_entries':len(files),'private_sha256':digest(private),'private_bytes':private.stat().st_size,'private_entries':private_entries,'private_path':str(private),'private_pushed_to_Git':False}
    (R/'PUBLIC_ARCHIVE_STATUS.json').write_text(json.dumps(result,indent=2));(P/'DONE.json').write_text(json.dumps({'status':'COMPLETE','records':465,'new_fits':390,'archives':result},indent=2));print(json.dumps(result))


if __name__=='__main__':main()

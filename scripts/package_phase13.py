"""Explicit public allowlist and private reproducibility archive on data disk."""
import hashlib,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase13_rna_information_audit';P=Path('/root/phase13_rna_private')


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def sources():
    scripts=['prepare_phase13.py','gate_phase13.py','run_phase13.py','schedule_phase13.py','launch_phase13.py','validate_phase13_independent.py','replay_phase13.py','audit_phase13_statistics.py','audit_phase13_absolute_ci.py','phase13_matched_conditional_appendix.py','launch_phase13_audit.py','report_phase13.py','package_phase13.py','check_phase13_delivery.py','stage_phase13_public.py',
             'run_phase12.py','run_phase11.py','validate_phase11_independent.py','replay_phase11.py']
    files=[ROOT/'scripts'/name for name in scripts]+list((ROOT/'src/phase13_rna').glob('*.py'))+list((ROOT/'src/phase12_c4').glob('*.py'))+list((ROOT/'src/phase11_semantic').glob('*.py'))
    files+=[ROOT/'src/phase10_trimodal'/name for name in ['data.py','features.py','math.py','__init__.py']]
    files+=[ROOT/'configs'/name for name in ['phase13_rna_information_audit.json','phase12_c4_confirmation.json','phase11_semantic_anchored.json','phase10_molecular_groups.json']]
    files+=[ROOT/'tests'/name for name in ['__init__.py','test_phase13_rna.py','test_phase12_c4.py']]
    files+=[ROOT/'src/phase_m2/metadata/mmp_hallmarks_signatures.csv',ROOT/'PHASE13_R0_RNA_INFORMATION_AUDIT_PROTOCOL.md']
    return files


def main():
    run=json.loads((R/'RUN_MANIFEST.json').read_text())
    if run['status']!='COMPLETE_AUDITED' or run['records']!=3330:raise RuntimeError('not complete')
    if json.loads((R/'APPENDIX_VALIDATION.json').read_text())['status']!='PASS':raise RuntimeError('matched control missing')
    files=sources();snapshot=R/'training_source'
    for p in files:
        q=snapshot/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    excluded={'PUBLIC_ARTIFACT_MANIFEST.json','PUBLIC_ARCHIVE_STATUS.json','PUBLIC_PACKAGE_SHA256.txt','DELIVERY_CONTRACT.json'}
    files=sorted(set(files+[p for p in R.rglob('*') if p.is_file() and p.suffix in ['.py','.md','.json','.csv','.txt'] and p.name not in excluded]))
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for p in files:
        if pattern.search(p.read_bytes()):raise RuntimeError('patient/credential pattern '+str(p.relative_to(ROOT)))
    manifest={'status':'PASS','files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size} for p in files],'patient_arrays_or_checkpoints_public':False}
    m=R/'PUBLIC_ARTIFACT_MANIFEST.json';m.write_text(json.dumps(manifest,indent=2));files.append(m);archive=R/'PHASE13_PUBLIC_AUDIT.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('public CRC')
    sha=digest(archive);(R/'PUBLIC_PACKAGE_SHA256.txt').write_text(sha+'  PHASE13_PUBLIC_AUDIT.zip\n')
    references=[]
    for p in (P/'producers').glob('*_reconstruction.json'):
        c=json.loads(p.read_text());references.append({'gene':c['gene'],'seed':c['seed'],'fold':c['fold'],'path':c['outer_parent_checkpoint'],'sha256':c['outer_parent_checkpoint_sha256']})
    dependency={'originalCDR':'sources/TCGA-CDR-SupplementalTableS1.xlsx','parent_data':'/root/phase7a_foundation_headroom_v1/private_dataset.npz','parent_fold_gene_preprocessing':'/root/phase12_c4_private/preprocessing','read_only_CONCH_cache':'/root/phase11_semantic_anchor_private/cache','raw_clinical':'/root/autodl-tmp/diffusion_data/data/data_8/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt','parent_checkpoint_references':references,'registry':json.loads((R/'SOURCE_REGISTRY.json').read_text()),'notice':'large historical matrices/H5 and45 parentcheckpoints remain separate,not duplicated'}
    (P/'REQUIRED_PRIVATE_INPUTS.json').write_text(json.dumps(dependency,indent=2))
    target=Path('/root/autodl-tmp/phase13_rna_information_audit');target.mkdir(exist_ok=True);private=target/'PHASE13_PRIVATE_REPRO.zip'
    with zipfile.ZipFile(private,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for folder in ['formal','producers','preprocessing','transforms','diagnostics','matched_conditional_appendix','sources','logs']:
            for p in sorted((P/folder).rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(P)))
        for name in ['endpoints.npz','REQUIRED_PRIVATE_INPUTS.json','BUG_JOURNAL.json','GPU_TELEMETRY.json','full_run.log','independent_audit.log','phase13_initial.tar']:
            p=P/name
            if p.exists():z.write(p,name)
        for p in files:z.write(p,'public_and_sources/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(private) as z:
        if z.testzip():raise RuntimeError('private CRC')
        count=len(z.infolist())
    result={'status':'PASS','public_sha256':sha,'public_bytes':archive.stat().st_size,'public_entries':len(files),'private_sha256':digest(private),'private_bytes':private.stat().st_size,'private_entries':count,'private_path':str(private),'private_data_disk_used_to_avoid_root_fill':True,'private_pushed_to_Git':False}
    (R/'PUBLIC_ARCHIVE_STATUS.json').write_text(json.dumps(result,indent=2));(P/'DONE.json').write_text(json.dumps({'status':'COMPLETE','core_records':3330,'appendix_uniquefits':30,'archives':result},indent=2));print(json.dumps(result))


if __name__=='__main__':main()

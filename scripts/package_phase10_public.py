"""Explicit allowlist package, refuse case identifiers, tokens and private arrays."""
import json,hashlib,re,zipfile,argparse,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=['run_phase10_trimodal.py','schedule_phase10_trimodal.py','validate_phase10_independent.py','repair_phase10_reuse_metadata.py','audit_phase10_representations.py','audit_phase10_final.py','audit_phase10_raw_sources.py','audit_phase10_omics_source.py','collect_phase10_audit_tables.py','supplement_phase10_fixed_representations.py','add_phase10_descriptive_pairs.py','test_phase10_real_test_perturbation.py','run_phase10_complete.py','package_phase10_public.py']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=ROOT/'reports/phase10_trimodal_mechanism_audit';paths=list((ROOT/'src/phase10_trimodal').glob('*.py'))+[ROOT/'scripts'/s for s in SCRIPTS]+[ROOT/'tests/test_phase10_trimodal_audit.py',ROOT/'configs/phase10_trimodal_mechanism_audit.json',ROOT/'configs/phase10_molecular_groups.json']
 paths+=sorted(p for p in report.rglob('*') if p.is_file() and p.name not in ['PHASE10_PUBLIC_AUDIT.zip','public_artifact_manifest.json','package_checksums.json'])
 records=[];bad=[]
 for p in paths:
  if p.suffix in ['.npz','.npy','.pt','.pth','.h5','.pkl']:bad.append(str(p));continue
  if p.suffix in ['.py','.json','.md','.txt','.csv']:
   txt=p.read_text(encoding='utf8')
   if re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',txt) or re.search(r'hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY',txt):bad.append(str(p))
  if p.suffix=='.tgz':
   if p.name!='TRAINING_SOURCE_V2.tgz':bad.append(str(p));continue
   with tarfile.open(p) as t:
    for member in t.getmembers():
     if not member.isfile() or Path(member.name).suffix not in ['.py','.json']:bad.append(str(p));continue
     text=t.extractfile(member).read().decode('utf8')
     if re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|PRIVATE KEY',text):bad.append(str(p))
  records.append({'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'size_bytes':p.stat().st_size,'models':'ALL' if '/phase10_trimodal/' in p.as_posix() or p.name.startswith(('fold_','seed_','paired_')) else 'see MODEL_MATRIX','associated_script':'scripts/run_phase10_trimodal.py + scripts/validate_phase10_independent.py','role':'source/config' if p.suffix=='.py' or p.parent.name=='configs' else 'aggregate evidence/frozen source'})
 if bad:raise ValueError('public privacy allowlist rejected '+str(bad))
 manifest=report/'public_artifact_manifest.json';manifest.write_text(json.dumps({'status':'PASS','contains_patient_level_data':False,'files':records,'self_reference':'manifest and package_checksums excluded from per-file hash list to avoid recursive checksums','archives':'PUBLIC zip + frozen code tgz local/server artifacts; not tracked in Git under binary guard'},indent=2)+'\n')
 with zipfile.ZipFile(report/'PHASE10_PUBLIC_AUDIT.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in paths+[manifest]:z.write(p,p.relative_to(ROOT).as_posix())
 checks=report/'package_checksums.json';prior=json.loads(checks.read_text()) if checks.exists() else {};prior['public']={'path':'reports/phase10_trimodal_mechanism_audit/PHASE10_PUBLIC_AUDIT.zip','sha256':digest(report/'PHASE10_PUBLIC_AUDIT.zip'),'size_bytes':(report/'PHASE10_PUBLIC_AUDIT.zip').stat().st_size};checks.write_text(json.dumps(prior,indent=2)+'\n')
 with zipfile.ZipFile(report/'PHASE10_PUBLIC_AUDIT.zip') as z:assert z.testzip() is None
 print(json.dumps({'status':'PASS','files':len(paths)+1,'public_size_bytes':prior['public']['size_bytes'],'public_sha256':prior['public']['sha256']}))
if __name__=='__main__':main()

"""Explicit allowlist staging after the executable delivery/privacy contract."""
import hashlib,json,re,subprocess,zipfile
from pathlib import Path
from package_phase11 import source_files
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase11_semantic_anchored_trimodal'
if json.loads((R/'DELIVERY_CONTRACT.json').read_text())['status']!='PASS':
    raise RuntimeError('delivery contract incomplete')
branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
if branch!='codex/phase11-semantic-anchored-trimodal':raise RuntimeError('wrong publication branch')
paths=sorted(set(source_files()+[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts]))
forbidden=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
def inspect(name,content):
    if Path(name).suffix.lower() in ['.npz','.npy','.pt','.h5','.pem','.tar']:
        raise RuntimeError('private artifact in publish set '+name)
    if forbidden.search(content):raise RuntimeError('patient/secret pattern '+name)
for p in paths:
    if p.suffix=='.zip':
        if p.name!='PHASE11_PUBLIC_AUDIT.zip':raise RuntimeError('unexpected archive')
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():inspect(name,z.read(name))
    else:inspect(str(p.relative_to(ROOT)),p.read_bytes())
report={'status':'PASS','branch':branch,'public_files':len(paths),'public_bytes':sum(p.stat().st_size for p in paths),'private_arrays_checkpoints_patient_ids_included':False,'public_zip_sha256':hashlib.sha256((R/'PHASE11_PUBLIC_AUDIT.zip').read_bytes()).hexdigest()}
(R/'PUBLISH_PREFLIGHT.json').write_text(json.dumps(report,indent=2),encoding='utf8')
paths.append(R/'PUBLISH_PREFLIGHT.json')
subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
print(json.dumps(report))
subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)

"""Explicit protected-data-checked Git allowlist after completion."""
import json,re,subprocess,zipfile
from pathlib import Path
from package_phase12 import sources
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase12_c4_confirmation'
if json.loads((R/'DELIVERY_CONTRACT.json').read_text())['status']!='PASS':raise RuntimeError('delivery incomplete')
if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='codex/phase12-c4-mechanism-confirmation':raise RuntimeError('wrong branch')
files=sorted(set(sources()+[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts]));pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
def inspect(name,content):
    if Path(name).suffix in ['.npz','.npy','.pt','.h5','.pem','.tar'] or pattern.search(content):raise RuntimeError('protected artifact '+name)
for p in files:
    if p.suffix=='.zip':
        if p.name!='PHASE12_PUBLIC_AUDIT.zip':raise RuntimeError('unexpected zip')
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():inspect(name,z.read(name))
    else:inspect(str(p.relative_to(ROOT)),p.read_bytes())
public=R/'PHASE12_PUBLIC_AUDIT.zip';plain=[p for p in files if p!=public]
subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in plain]],cwd=ROOT,check=True)
# Only this verified small public artifact bypasses the repository ZIP ignore.
subprocess.run(['git','add','-f','--',str(public.relative_to(ROOT))],cwd=ROOT,check=True)
print(json.dumps({'status':'PASS','public_files':len(files),'private_data_published':False}))

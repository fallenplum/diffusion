import json,re,subprocess,zipfile
from pathlib import Path
from package_phase13 import sources
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/phase13_rna_information_audit'
if json.loads((R/'DELIVERY_CONTRACT.json').read_text())['status']!='PASS':raise RuntimeError('incomplete')
if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='codex/phase13-rna-information-retention-audit':raise RuntimeError('wrong branch')
files=sorted(set(sources()+[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts]));pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
def check(name,data):
    if Path(name).suffix in ['.npz','.npy','.pt','.xlsx','.h5','.pem','.tar'] or pattern.search(data):raise RuntimeError('protected file '+name)
for p in files:
    if p.suffix=='.zip':
        if p.name!='PHASE13_PUBLIC_AUDIT.zip':raise RuntimeError('unknown ZIP')
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():check(name,z.read(name))
    else:check(str(p.relative_to(ROOT)),p.read_bytes())
public=R/'PHASE13_PUBLIC_AUDIT.zip';subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in files if p!=public]],cwd=ROOT,check=True);subprocess.run(['git','add','-f','--',str(public.relative_to(ROOT))],cwd=ROOT,check=True)
print(json.dumps({'status':'PASS','public_files':len(files),'private_data_published':False}))

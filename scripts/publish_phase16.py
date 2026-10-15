"""Publish only manifest-listed anonymous evidence; verify remote SHA and blobs."""
import sys,json,subprocess,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_routing.core import *
BRANCH='codex/phase16-semantic-roi-routing'
def main():
    delivery=json.loads((R/'DELIVERY.json').read_text());m=json.loads((R/'ARTIFACT_MANIFEST.json').read_text())
    if delivery['status']!='PASS_PARTIAL_G0_SCOPE' or m['status']!='PASS' or sha(R/'ARTIFACT_MANIFEST.json')!=delivery['manifest_sha256']:raise ValueError('delivery not accepted')
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----');paths=[]
    for row in m['public_files']:
        p=ROOT/row['path']
        if sha(p)!=row['sha256'] or pattern.search(p.read_bytes()):raise ValueError('public privacy/hash check')
        paths.append(row['path'])
    paths.extend(str((R/x).relative_to(ROOT)) for x in ['ARTIFACT_MANIFEST.json','DELIVERY.json'])
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('branch','--show-current')!=BRANCH:raise ValueError('wrong branch')
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);staged=set(git('diff','--cached','--name-only').splitlines())
    if not staged or not staged<=set(paths):raise ValueError('empty/unrelated staging')
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Run Phase16 full-pool ROI controls and document blocked semantic gates'],cwd=ROOT,check=True)
    ssh='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new';subprocess.run(['git','-c','core.sshCommand='+ssh,'push','-u','origin',BRANCH],cwd=ROOT,check=True);head=git('rev-parse','HEAD');remote=git('-c','core.sshCommand='+ssh,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if head!=remote:raise ValueError('remote commit mismatch')
    for row in m['public_files']:
        content=subprocess.check_output(['git','show',head+':'+row['path']],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('committed blob mismatch')
    result={'status':'PASS','branch':BRANCH,'commit':head,'remote_commit':remote,'verified_public_files':len(m['public_files'])+2,'private_patient_data_published':False};write(P/'GITHUB_PUSH_VERIFIED.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()

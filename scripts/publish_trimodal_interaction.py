import sys,json,re,hashlib,subprocess
from pathlib import PurePosixPath
from src.trimodal_interaction.data import ROOT,P,R,write,digest
BRANCH='research/trimodal-conditional-interaction-v1';SSH='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new'

def git(*args):return subprocess.check_output(['git','-c','core.sshCommand='+SSH,*args],cwd=ROOT)

def main():
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert git('remote','get-url','origin').decode().strip()=='git@github.com:fallenplum/diffusion.git'
    m=R/'ARTIFACT_MANIFEST.json';a=json.loads(m.read_text());assert a['status']=='PASS'
    rows=a['public_files']+[{'path':m.relative_to(ROOT).as_posix(),'bytes':m.stat().st_size,'sha256':digest(m)}];paths=[r['path'] for r in rows]
    assert len(paths)==len(set(paths));pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for row in rows:
        p=PurePosixPath(row['path']);assert not p.is_absolute() and '..' not in p.parts;b=(ROOT/row['path']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'] and not pattern.search(b)
    for args in [('diff','--cached','--name-only','-z'),('diff','--name-only','-z')]:assert not (set(git(*args).decode().strip('\0').split('\0'))-{''})-set(paths)
    git('add','--',*paths);check=subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
    scope=json.loads((R/'PUBLICATION_SCOPE_AUDIT.json').read_text());guard=subprocess.run([sys.executable,'scripts/check_protected_data.py'],cwd=ROOT,capture_output=True,text=True)
    if guard.returncode:
        assert guard.returncode==1 and set(json.loads(guard.stdout)['violations'])=={r['path'] for r in scope['unchanged_inherited_violations']},guard.stdout+guard.stderr
        for row in scope['unchanged_inherited_violations']:assert hashlib.sha256(git('show',':'+row['path'])).hexdigest()==row['sha256']
    if git('diff','--cached','--name-only').strip():print(git('commit','-m','Audit reference-defined trimodal attention and inner training budget').decode(),flush=True)
    sha=git('rev-parse','HEAD').decode().strip();print(git('push','origin','HEAD:refs/heads/'+BRANCH).decode(),flush=True)
    assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==sha
    for row in rows:assert hashlib.sha256(git('show',sha+':'+row['path'])).hexdigest()==row['sha256']
    receipt={'status':'PASS','branch':BRANCH,'commit':sha,'verified_remote_blobs':len(rows),'no_patient_level_files_uploaded':True};write(P/'GITHUB_PUSH_VERIFIED.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

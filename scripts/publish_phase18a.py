import sys,json,re,hashlib,subprocess
from pathlib import PurePosixPath
from src.phase18a_readout.data import ROOT,P,R,write,digest
BRANCH='codex/phase18a-local-state-readout';SSH='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new'

def git(*args):return subprocess.check_output(['git','-c','core.sshCommand='+SSH,*args],cwd=ROOT)

def main():
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert git('remote','get-url','origin').decode().strip()=='git@github.com:fallenplum/diffusion.git'
    manifest=R/'ARTIFACT_MANIFEST.json';m=json.loads(manifest.read_text());assert m['status']=='PASS'
    rows=m['public_files']+[{'path':manifest.relative_to(ROOT).as_posix(),'bytes':manifest.stat().st_size,'sha256':digest(manifest)}];paths=[r['path'] for r in rows];assert len(paths)==len(set(paths))
    private=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for row in rows:
        p=PurePosixPath(row['path']);assert not p.is_absolute() and '..' not in p.parts;raw=(ROOT/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'] and not private.search(raw)
    for args in [('diff','--cached','--name-only','-z'),('diff','--name-only','-z')]:assert not (set(git(*args).decode().strip('\0').split('\0'))-{''})-set(paths)
    git('add','--',*paths)
    check=subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,capture_output=True,text=True)
    # This pretraining-locked file contains one harmless final blank line.
    # Retain its exact bytes/source hash; reject every other whitespace error.
    locked_eof_note='src/phase18a_readout/data.py:21: new blank line at EOF.\n'
    assert check.returncode==0 or (check.returncode==2 and check.stdout==locked_eof_note and not check.stderr),check.stdout+check.stderr
    if check.returncode:print('FROZEN_SOURCE_EOF_NOTE_RETAINED',locked_eof_note.strip(),flush=True)
    scope=json.loads((R/'PUBLICATION_SCOPE_AUDIT.json').read_text());guard=subprocess.run([sys.executable,'scripts/check_protected_data.py'],cwd=ROOT,capture_output=True,text=True)
    if guard.returncode:
        assert guard.returncode==1 and set(json.loads(guard.stdout)['violations'])=={r['path'] for r in scope['unchanged_inherited_violations']},guard.stdout+guard.stderr
        for row in scope['unchanged_inherited_violations']:assert hashlib.sha256(git('show',':'+row['path'])).hexdigest()==row['sha256']
    if git('diff','--cached','--name-only').strip():print(git('commit','-m','Audit Phase18A paired local S6 readout and patient conditioning').decode(),flush=True)
    sha=git('rev-parse','HEAD').decode().strip();print(git('push','origin','HEAD:refs/heads/'+BRANCH).decode(),flush=True)
    remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0];assert remote==sha
    for row in rows:assert hashlib.sha256(git('show',sha+':'+row['path'])).hexdigest()==row['sha256']
    result={'status':'PASS','branch':BRANCH,'commit':sha,'remote_commit':remote,'verified_public_blobs':len(rows),'no_patient_level_files_uploaded':True};write(P/'GITHUB_PUSH_VERIFIED.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

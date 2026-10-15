"""Explicit-file publication with anonymous-content and remote-blob checks."""
import sys,json,hashlib,re,subprocess
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_diagnostic.core import P,R,sha,write
BRANCH='codex/phase16-prompt-rna-diagnostic'
SSH='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new'

def git(*args):return subprocess.check_output(['git','-c','core.sshCommand='+SSH,*args],cwd=ROOT)

def main():
    if git('branch','--show-current').decode().strip()!=BRANCH or git('remote','get-url','origin').decode().strip()!='git@github.com:fallenplum/diffusion.git':raise ValueError('publication branch/origin mismatch')
    manifest=R/'ARTIFACT_MANIFEST.json';delivery=json.loads((R/'DELIVERY.json').read_text());m=json.loads(manifest.read_text())
    if delivery['status']!='PASS' or m['status']!='PASS' or delivery['manifest_sha256']!=sha(manifest):raise ValueError('delivery gate failed')
    rows=m['public_files']+[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in [manifest,R/'DELIVERY.json']];paths=[r['path'] for r in rows]
    if len(set(paths))!=len(paths):raise ValueError('duplicate public paths')
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for row in rows:
        relative=PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix not in {'.py','.json','.md','.csv'}:raise ValueError('unsafe public path')
        path=ROOT/row['path'];data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256'] or len(data)!=row['bytes'] or pattern.search(data):raise ValueError('public checksum/privacy failure '+row['path'])
    staged=set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))-{''}
    if staged-set(paths):raise ValueError('unrelated staged files')
    # Refuse any modification to previously published tracked files.
    tracked=set(git('diff','--name-only','-z').decode().strip('\0').split('\0'))-{''}
    if tracked-set(paths):raise ValueError('previous tracked content modified')
    git('add','--',*paths);git('diff','--cached','--check')
    if git('diff','--cached','--name-only').strip():print(git('commit','-m','Audit RNA compression, prompt sensitivity and coordinate-matched visual complementarity').decode(),flush=True)
    commit=git('rev-parse','HEAD').decode().strip();print(git('push','origin','HEAD:refs/heads/'+BRANCH).decode(),flush=True);remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]
    if remote!=commit:raise ValueError('remote commit not verified')
    for row in rows:
        if hashlib.sha256(git('show',commit+':'+row['path'])).hexdigest()!=row['sha256']:raise ValueError('remote commit blob mismatch '+row['path'])
    result={'status':'PASS','commit':commit,'remote_commit':remote,'branch':BRANCH,'public_blobs_verified':len(rows),'no_patient_data_uploaded':True,'results_url':'https://github.com/fallenplum/diffusion/blob/'+BRANCH+'/reports/phase16_prompt_rna_diagnostic/RESULTS.md'};write(P/'GITHUB_PUSH_VERIFIED.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

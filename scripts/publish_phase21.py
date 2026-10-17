"""Publish exactly the complete anonymous manifest, then verify remote bytes."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
from scripts.package_phase21 import P,R,ROOT,PARENT,BRANCH,allowed_path,audit_public_file,verify_completion

def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=ROOT,env=env)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    complete=verify_completion()
    manifest=json.loads((R/'ARTIFACT_MANIFEST.json').read_text())
    if manifest['source']!=complete['source'] or manifest['status']!='PASS': raise ValueError('release incomplete')
    paths={q['path']:q['sha256'] for q in manifest['public_files']}
    paths[(R/'ARTIFACT_MANIFEST.json').relative_to(ROOT).as_posix()]=sha(R/'ARTIFACT_MANIFEST.json')
    if git('remote','get-url','origin').decode().strip()!='git@github.com:fallenplum/diffusion.git': raise ValueError('unexpected origin')
    if git('branch','--show-current').decode().strip()!=BRANCH: raise ValueError('unexpected branch')
    for relative,expected in paths.items():
        if not allowed_path(relative) or audit_public_file(ROOT/relative)['sha256']!=expected: raise ValueError('publication drift '+relative)
    current=git('rev-parse','HEAD').decode().strip()
    if current==PARENT:
        subprocess.run(['git','add','--',*sorted(paths)],cwd=ROOT,check=True)
        staged=git('diff','--cached','--name-status').decode().splitlines()
        if {(q.split('\t')[0],q.split('\t')[1]) for q in staged}!={('A',p) for p in paths}: raise ValueError('staged changes outside new manifest')
        subprocess.run(['git','commit','-m','Phase21: audit relative attention corrections and paired risk repair'],cwd=ROOT,check=True)
        current=git('rev-parse','HEAD').decode().strip()
    elif git('rev-parse','HEAD^').decode().strip()!=PARENT:
        raise ValueError('unexpected publication ancestry')
    for relative,expected in paths.items():
        if hashlib.sha256(git('show',current+':'+relative)).hexdigest()!=expected: raise ValueError('local commit blob mismatch')
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new')
    subprocess.run(['git','push','-u','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,env=env,check=True)
    remote=git('ls-remote','--heads','origin',BRANCH,env=env).decode().strip().split()
    if not remote or remote[0]!=current: raise ValueError('remote SHA differs')
    subprocess.run(['git','fetch','origin','refs/heads/'+BRANCH],cwd=ROOT,env=env,check=True)
    if git('rev-parse','FETCH_HEAD').decode().strip()!=current: raise ValueError('fetch SHA differs')
    for relative,expected in paths.items():
        if hashlib.sha256(git('show','FETCH_HEAD:'+relative)).hexdigest()!=expected: raise ValueError('remote fetched blob mismatch')
    record={'status':'PASS','source':complete['source'],'commit':current,'branch':BRANCH,
        'remote':'fallenplum/diffusion','public_files_verified':len(paths),'remote_SHA_and_all_blobs_verified':True,
        'public_manifest_sha256':sha(R/'ARTIFACT_MANIFEST.json')}
    (P/'GITHUB_PUSH_VERIFIED.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PHASE21_PUSH_VERIFIED',current,len(paths),flush=True)

if __name__=='__main__':main()

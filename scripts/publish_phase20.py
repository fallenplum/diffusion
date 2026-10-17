"""Explicitly publish only complete, immutable-parent Phase20 public evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_phase20 import (BRANCH,PARENT,P,R,allowed_path,audit_parent_immutability,
    audit_public_file,read,require,safe_path,verify_completion)
from src.phase20_selective.data import write
from src.phase10_trimodal.data import digest

SSH='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new'
REMOTE='git@github.com:fallenplum/diffusion.git'


def git(*args):
    return subprocess.check_output(['git','-c','core.sshCommand='+SSH,*args],cwd=ROOT)


def names(*args):
    return {p.decode('utf-8') for p in git(*args,'-z').split(b'\0') if p}


def main():
    require(git('branch','--show-current').decode().strip()==BRANCH,'wrong target branch')
    require(git('remote','get-url','origin').decode().strip()==REMOTE,'wrong repository')
    completion=verify_completion();parent,legacy=audit_parent_immutability()
    manifest_path=R/'ARTIFACT_MANIFEST.json';manifest=read(manifest_path)
    require(all(manifest.get(k)==v for k,v in completion.items()),'packaged completion receipts changed')
    require(manifest['branch']==BRANCH and manifest['parent_commit']==PARENT,'publication target changed')
    require(read(R/'PARENT_IMMUTABILITY_AUDIT.json')==parent,'parent evidence changed')
    scope=read(R/'PUBLICATION_SCOPE_AUDIT.json')
    require(scope['status']=='PASS' and scope['unchanged_inherited_violations']==legacy,'legacy exception list changed')
    rows=manifest['public_files']+[audit_public_file(manifest_path)]
    paths=[q['path'] for q in rows];allowed=set(paths)
    require(len(paths)==len(allowed) and all(allowed_path(p) for p in paths),'duplicate/out-of-scope publication path')
    for row in rows:
        require(audit_public_file(safe_path(row['path']))==row,'public bytes changed after packaging: '+row['path'])
    for command in [('diff','--cached','--name-only'),('diff','--name-only'),('diff','--name-only',PARENT,'HEAD')]:
        require(not names(*command)-allowed,'unrelated working/staged/earlier branch changes')
    git('add','--',*paths)
    require(not names('diff','--cached','--name-only')-allowed,'staged files outside allowlist')
    for row in rows:
        require(hashlib.sha256(git('show',':'+row['path'])).hexdigest()==row['sha256'],
                'Git filter or staging changed audited bytes: '+row['path'])
    whitespace=subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,capture_output=True,text=True)
    require(whitespace.returncode==0,'new publication whitespace check failed: '+whitespace.stdout+whitespace.stderr)
    guard=subprocess.run([sys.executable,'scripts/check_protected_data.py'],cwd=ROOT,capture_output=True,text=True)
    require(guard.returncode==1 and set(json.loads(guard.stdout)['violations'])=={q['path'] for q in legacy},
            'whole-repository guard differs from five unchanged inherited ZIP exceptions')
    for row in legacy:
        require(hashlib.sha256(git('show',':'+row['path'])).hexdigest()==row['sha256'],'legacy staged bytes changed')
    if names('diff','--cached','--name-only'):
        print(git('commit','-m','Audit selective clinical correction and pathway token capacity with frozen strong anchors').decode(),flush=True)
    commit=git('rev-parse','HEAD').decode().strip()
    print(git('push','origin','HEAD:refs/heads/'+BRANCH).decode(),flush=True)
    remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]
    require(remote==commit,'remote branch differs from committed publication')
    git('fetch','--no-tags','origin','refs/heads/'+BRANCH)
    fetched=git('rev-parse','FETCH_HEAD').decode().strip()
    require(fetched==commit,'fetched remote commit differs')
    for row in rows:
        require(hashlib.sha256(git('show','FETCH_HEAD:'+row['path'])).hexdigest()==row['sha256'],
                'remote blob SHA256 differs: '+row['path'])
    result={'status':'PASS','branch':BRANCH,'commit':commit,'remote_commit':remote,'fetched_remote_commit':fetched,
        'verified_public_blobs':len(rows),'source':completion['source'],'public_manifest_sha256':digest(manifest_path),
        'retained_original_source':completion['retained_original_source'],
        'retained_original_source_checkpoints':completion['retained_original_source_checkpoints'],
        'fresh_current_source_checkpoints':completion['fresh_current_source_checkpoints'],
        'private_audit_repair_inventory_sha256':completion['private_audit_repair_inventory_sha256'],
        'audit_repair_lock_sha256':completion['audit_repair_lock_sha256'],
        'historical_reference_repair_audit_sha256':completion['historical_reference_repair_audit_sha256'],
        'historical_reference_audit_program_sha256':completion['historical_reference_audit_program_sha256'],
        'no_patient_level_files_uploaded':True,'force_push_used':False,
        'legacy_exceptions':'exactly five unchanged inherited ZIPs; no whole-repository privacy PASS claim'}
    write(P/'GITHUB_PUSH_VERIFIED.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()

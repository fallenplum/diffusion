"""Only audited anonymous Phase17 artifacts; verify remote commit and blobs."""
import sys,json,re,hashlib,subprocess
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17_ssd.data import P,R,write
from src.phase10_trimodal.data import digest
BRANCH='codex/phase17-ssd-fusion';SSH='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new'
def git(*args):return subprocess.check_output(['git','-c','core.sshCommand='+SSH,*args],cwd=ROOT)

def main():
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert git('remote','get-url','origin').decode().strip()=='git@github.com:fallenplum/diffusion.git'
    manifest=R/'ARTIFACT_MANIFEST.json';delivery=R/'DELIVERY.json';m=json.loads(manifest.read_text());q=json.loads(delivery.read_text())
    assert m['status']=='PASS' and q['status']=='CORE_COMPLETE_BASELINE_TARGET_PARTIAL' and q['manifest_sha256']==digest(manifest)
    rows=m['public_files']+[{'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size} for p in [manifest,delivery]];paths=[r['path'] for r in rows]
    assert len(set(paths))==len(paths)
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for r in rows:
        path=PurePosixPath(r['path']);assert not path.is_absolute() and '..' not in path.parts
        b=(ROOT/r['path']).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'] and not pattern.search(b)
    for args in [('diff','--cached','--name-only','-z'),('diff','--name-only','-z')]:
        changed=set(git(*args).decode().strip('\0').split('\0'))-{''};assert not changed-set(paths),changed-set(paths)
    git('add','--',*paths);git('diff','--cached','--check')
    guard=subprocess.run([sys.executable,'scripts/check_protected_data.py'],cwd=ROOT,capture_output=True,text=True)
    scope=json.loads((R/'PUBLICATION_SCOPE_AUDIT.json').read_text())
    inherited=scope['inherited_unchanged_violations'];assert scope['status']=='PASS'
    if guard.returncode:
        assert guard.returncode==1 and set(json.loads(guard.stdout)['violations'])=={r['path'] for r in inherited},guard.stdout+guard.stderr
        for row in inherited:
            assert hashlib.sha256(git('show',scope['base_commit']+':'+row['path'])).hexdigest()==row['sha256']
            assert hashlib.sha256(git('show',':'+row['path'])).hexdigest()==row['sha256']
        print('Phase17 incremental guard PASS; inherited unchanged archive extension findings:',len(inherited),flush=True)
    else:assert not inherited
    if git('diff','--cached','--name-only').strip():print(git('commit','-m','Complete Phase17 conditional S6 ablations, independent audits and negative mechanism findings').decode(),flush=True)
    commit=git('rev-parse','HEAD').decode().strip();print(git('push','origin','HEAD:refs/heads/'+BRANCH).decode(),flush=True)
    remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0];assert remote==commit
    for row in rows:assert hashlib.sha256(git('show',commit+':'+row['path'])).hexdigest()==row['sha256']
    result={'status':'PASS','branch':BRANCH,'commit':commit,'remote_commit':remote,'public_blobs_verified':len(rows),'no_patient_arrays_uploaded':True,'report_url':'https://github.com/fallenplum/diffusion/blob/'+BRANCH+'/reports/phase17_ssd_fusion/RESULTS.md','remaining_blocker':'TransSurv official baseline unavailable'}
    write(P/'GITHUB_PUSH_VERIFIED.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

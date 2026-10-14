"""Stage only an audited explicit public inventory; verify remote branch SHA."""
import json,re,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_phase14 import REPORT,PRIVATE,sha,write
BRANCH='codex/phase14-rna-ranking-conditional-audit'


def main():
    delivery=json.loads((REPORT/'DELIVERY.json').read_text());manifest=json.loads((REPORT/'PUBLIC_FILES.json').read_text())
    if delivery['status']!='PASS' or manifest['status']!='PASS':raise RuntimeError('delivery incomplete')
    archive=REPORT/'PHASE14_PUBLIC_AUDIT.zip'
    if sha(archive)!=delivery['public_zip_sha256']:raise RuntimeError('public archive hash')
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP CRC')
    paths=[];pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for item in manifest['files']:
        p=ROOT/item['path']
        if sha(p)!=item['sha256'] or pattern.search(p.read_bytes()):raise RuntimeError('protected or modified file '+str(p))
        paths.append(str(p.relative_to(ROOT)))
    paths.extend(str((REPORT/n).relative_to(ROOT)) for n in ['PUBLIC_FILES.json','DELIVERY.json'])
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('branch','--show-current')!=BRANCH:raise RuntimeError('wrong branch')
    subprocess.check_call(['git','add','--',*paths],cwd=ROOT);subprocess.check_call(['git','add','-f','--',str(archive.relative_to(ROOT))],cwd=ROOT)
    staged=git('diff','--cached','--name-only').splitlines();allowed=set(paths+[str(archive.relative_to(ROOT))])
    if not set(staged)<=allowed or not staged:raise RuntimeError('unrelated/empty staging')
    subprocess.check_call(['git','diff','--cached','--check'],cwd=ROOT)
    subprocess.check_call(['git','commit','-m','Complete Phase14 RNA ranking retention and conditional interaction audit'],cwd=ROOT)
    head=git('rev-parse','HEAD');ssh='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new'
    subprocess.check_call(['git','-c','core.sshCommand='+ssh,'push','-u','origin',BRANCH],cwd=ROOT)
    remote=git('-c','core.sshCommand='+ssh,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if head!=remote:raise RuntimeError('remote commit mismatch')
    for item in manifest['files']:
        data=subprocess.check_output(['git','show',head+':'+item['path']],cwd=ROOT)
        import hashlib
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise RuntimeError('committed blob mismatch')
    result={'status':'PASS','branch':BRANCH,'commit':head,'remote_commit':remote,'verified_public_files':len(manifest['files']),'public_zip_sha256':delivery['public_zip_sha256'],'private_data_published':False}
    write(PRIVATE/'GITHUB_PUSH_VERIFIED.json',result);print(json.dumps(result))


if __name__=='__main__':main()

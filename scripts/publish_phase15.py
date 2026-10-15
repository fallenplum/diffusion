"""Publish an explicit audited list, then verify remote commit and every blob."""
import hashlib,json,re,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase15_complement.core import REPORT,PRIVATE,sha,write
BRANCH='codex/phase15-rna-conditional-complementarity'


def main():
    status=json.loads((REPORT/'DELIVERY.json').read_text());manifest=json.loads((REPORT/'PUBLIC_FILES.json').read_text());archive=REPORT/'PHASE15_PUBLIC_AUDIT.zip'
    if status['status']!='PASS' or manifest['status']!='PASS' or sha(archive)!=status['public_zip_sha256']:raise RuntimeError('incomplete or modified delivery')
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----');paths=[]
    for row in manifest['files']:
        file=ROOT/row['path']
        if sha(file)!=row['sha256'] or pattern.search(file.read_bytes()):raise RuntimeError('privacy/hash failure '+str(file))
        paths.append(row['path'])
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP CRC')
        for row in manifest['files']:
            if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:raise RuntimeError('ZIP hash')
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('branch','--show-current')!=BRANCH:raise RuntimeError('wrong branch')
    paths += [str((REPORT/n).relative_to(ROOT)) for n in ['PUBLIC_FILES.json','DELIVERY.json']]
    subprocess.check_call(['git','add','--',*paths],cwd=ROOT);subprocess.check_call(['git','add','-f','--',str(archive.relative_to(ROOT))],cwd=ROOT)
    expected=set(paths+[str(archive.relative_to(ROOT))]);staged=set(git('diff','--cached','--name-only').splitlines())
    if not staged or not staged<=expected:raise RuntimeError('unrelated/empty staged files')
    subprocess.check_call(['git','diff','--cached','--check'],cwd=ROOT);subprocess.check_call(['git','commit','-m','Complete Phase15 honest anchor conditional RNA complementarity audit'],cwd=ROOT)
    ssh='ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new';subprocess.check_call(['git','-c','core.sshCommand='+ssh,'push','-u','origin',BRANCH],cwd=ROOT)
    head=git('rev-parse','HEAD');remote=git('-c','core.sshCommand='+ssh,'ls-remote','origin','refs/heads/'+BRANCH).split()[0]
    if head!=remote:raise RuntimeError('remote SHA mismatch')
    for row in manifest['files']:
        data=subprocess.check_output(['git','show',head+':'+row['path']],cwd=ROOT)
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise RuntimeError('committed blob mismatch')
    result={'status':'PASS','branch':BRANCH,'commit':head,'remote_commit':remote,'verified_public_files':len(manifest['files']),'public_zip_sha256':status['public_zip_sha256'],'private_data_published':False};write(PRIVATE/'GITHUB_PUSH_VERIFIED.json',result);print(json.dumps(result))


if __name__=='__main__':main()

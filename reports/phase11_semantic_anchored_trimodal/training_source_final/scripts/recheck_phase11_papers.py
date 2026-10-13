"""Read pinned local official source clones and record file-level provenance."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];old=json.loads((ROOT/'reports/phase10_trimodal_mechanism_audit/SOURCE_REGISTRY.json').read_text())
rows=[]
for source in old['third_party_code']:
    if source['name'] not in ['SurvPGC','CiMP']:continue
    repo=ROOT/'tmp/phase10_trimodal_sources'/source['name'];ref=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if ref!=source['commit_sha']:raise ValueError('pinned official ref drift')
    for entry in source['files']:
        path=repo/entry['path'];contents=subprocess.check_output(['git','show',ref+':'+entry['path']],cwd=repo);sha=hashlib.sha256(contents).hexdigest()
        if sha!=entry['sha256']:raise ValueError('official source blob drift')
        rows.append({'repository':source['name'],'ref':ref,'path':entry['path'],'sha256':sha,'bytes':len(contents),'status':'READ_VERIFIED_PINNED','working_copy_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'working_copy_LF_equivalent':path.read_bytes().replace(b'\r\n',b'\n')==contents})
report=ROOT/'reports/phase11_semantic_anchored_trimodal';report.mkdir(parents=True,exist_ok=True)
(report/'PAPER_SOURCE_RECHECK.json').write_text(json.dumps({'status':'PASS','sources':rows,'scope':'pinned official code, not current GitHub tip or full CiMP paper'},indent=2))
print(json.dumps(rows))

import sys,json,re,hashlib,subprocess
from pathlib import Path
import pandas as pd
from src.trimodal_interaction.data import ROOT,P,R,cfg,source,write,digest
sys.path.insert(0,str(ROOT/'src'))
from reproducibility import protected_path_violations

def main():
    for name in ['OPERATOR_AUDIT','PARENT_IMMUTABILITY_AUDIT','DATA_BOUNDARY_AUDIT','VALIDATION','QUERY_REPRESENTATION_AUDIT']:
        assert json.loads((R/(name+'.json')).read_text())['status']=='PASS',name
    for name in ['VALIDATION','QUERY_REPRESENTATION_AUDIT']:
        assert json.loads((R/'budget160'/(name+'.json')).read_text())['status']=='PASS',name
    assert json.loads((R/'a3f_norm_repair/VALIDATION.json').read_text())['status']=='PASS'
    assert json.loads((P/'DONE.json').read_text())['source']==source()
    assert json.loads((P/'budget160/DONE.json').read_text())['source']==source()
    assert (R/'RESULTS.md').exists() and (R/'decision.json').exists()
    # A repository-wide old ZIP violation is not silently relabelled as PASS.
    base=cfg()['parent_commit'];tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().strip('\0').split('\0');inherited=[]
    for path in protected_path_violations(tracked):
        old=subprocess.check_output(['git','show',base+':'+path],cwd=ROOT);assert old==(ROOT/path).read_bytes()
        inherited.append({'path':path,'sha256':hashlib.sha256(old).hexdigest()})
    write(R/'PUBLICATION_SCOPE_AUDIT.json',{'status':'PASS','scope':'new anonymous conditional-attention audit files only','base_commit':base,'repository_wide_guard':'LEGACY_EXTENSION_VIOLATIONS' if inherited else 'PASS','unchanged_inherited_violations':inherited,'no_repository_wide_guard_PASS_claim':True})
    files=list((ROOT/'src/trimodal_interaction').glob('*.py'))+list((ROOT/'scripts').glob('*trimodal*interaction*.py'))
    files += [ROOT/'scripts/extend_trimodal_budget.py',ROOT/'scripts/start_trimodal_budget.py',ROOT/'scripts/audit_trimodal_query.py',ROOT/'scripts/audit_trimodal_environment.py',ROOT/'tests/test_trimodal_interaction.py',ROOT/'configs/trimodal_interaction_v1.json',ROOT/'configs/trimodal_interaction_budget160.json',ROOT/'docs/literature_mechanism_matrix.md',ROOT/'docs/novelty_ledger.md']
    files += [ROOT/'scripts/repair_trimodal_diff_norm.py',ROOT/'scripts/start_trimodal_diff_norm.py',ROOT/'scripts/monitor_trimodal_resources.py',ROOT/'configs/trimodal_diff_norm_repair.json']
    files += [p for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json']
    files=sorted(set(files));assert not protected_path_violations([p.relative_to(ROOT).as_posix() for p in files]);rows=[]
    private=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for p in files:
        assert p.suffix in ['.py','.json','.md','.csv'] and p.stat().st_size<8*1024**2,p
        raw=p.read_bytes();assert not private.search(raw),p
        if p.suffix=='.csv':assert not set(pd.read_csv(p,nrows=0).columns.str.lower())&{'patient_id','patient_ids','risk','prediction','barcode','sample_id'},p
        rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':digest(p)})
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','source':source(),'public_files':rows,'scope':'completed exploratory inner-only40 plus160 budget sensitivity; no formal outer inference','private_exclusions':['patient data','risks','checkpoints','preprocessing fit states','gene identities','permutations','tokens']})
    print('PACKAGE_PASS',len(rows),flush=True)

if __name__=='__main__':main()

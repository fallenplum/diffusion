"""Fail-closed completion, parent immutability and anonymous publication scope."""
import sys,json,re,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd
from src.phase17r_ssd.data import P,R,cfg,source,write,old
from src.phase10_trimodal.data import digest
from reproducibility import protected_path_violations

def main():
    for name in ['OPERATOR_AUDIT','DATA_LEAKAGE_AUDIT','INDEPENDENT_VALIDATION','CHECKPOINT_REPLAY','BOOTSTRAP_AUDIT','FIGURE_VISUAL_QA','PARENT_FULL_A8_PARITY','COMPUTE_BUDGET_AUDIT']:
        q=json.loads((R/(name+'.json')).read_text());assert q['status']=='PASS',name
    lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());dec=json.loads((R/'STAGE_DECISION.json').read_text());done=json.loads((P/'TRAINING_DONE.json').read_text());counts=json.loads((R/'EXECUTION_COUNTS.json').read_text())
    assert lock['source']==dec['source']==done['source']==source() and done['status']=='PASS' and dec['outer_files_present_at_decision']==0
    assert dec['architecture_decisions_locked_before_outer'] and counts['outer_fits']==json.loads((R/'CHECKPOINT_REPLAY.json').read_text())['checkpoints']==done['outer_fits']
    if not dec['R3']['run']:assert not list((P/'outer_reverse').glob('*.pt'))
    folds=pd.read_csv(R/'FOLD_METRICS.csv');seeds=pd.read_csv(R/'SEED_OOF_METRICS.csv')
    assert not folds.duplicated(['arm','seed','fold']).any()
    for (arm,s),g in folds.groupby(['arm','seed']):
        row=seeds[(seeds.arm==arm)&(seeds.seed==s)].iloc[0];assert len(g)==5 and g.n.sum()==512 and g.events.sum()==65
        assert abs(g.H_num.sum()/g.H_den.sum()-row.Harrell)<1e-12 and abs(g.U_num.sum()/g.U_den.sum()-row.Uno60)<1e-12
    parent=json.loads((old.R/'ARTIFACT_MANIFEST.json').read_text())
    for row in parent['public_files']:assert digest(ROOT/row['path'])==row['sha256'],row['path']
    visual=json.loads((old.P/'visual/READY.json').read_text());dataset_sha=digest(old.parent.DATA)
    assert dataset_sha==visual['dataset_sha256'],'private raw dataset changed since visual/preprocessing audit'
    write(R/'DATA_SOURCE_AUDIT.json',{'status':'PASS','private_dataset_sha256':dataset_sha,'matches_parent_audited_dataset':True,'patients':512,'DSS_events':65,'visual_cache_source':'parent locked hashes checked before training','preprocessing':'per-call parent cache hashes and fit-index checks; no feature change in repair'})
    write(R/'PARENT_IMMUTABILITY_AUDIT.json',{'status':'PASS','parent_commit':cfg()['parent_commit'],'parent_manifest_files_verified':len(parent['public_files']),'parent_training_source_unchanged':old.source()==cfg()['parent_training_source']})
    base=cfg()['parent_commit'];tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().strip('\0').split('\0');inherited=[]
    for path in protected_path_violations(tracked):
        b=subprocess.check_output(['git','show',base+':'+path],cwd=ROOT);assert b==(ROOT/path).read_bytes()
        inherited.append({'path':path,'sha256':hashlib.sha256(b).hexdigest()})
    write(R/'PUBLICATION_SCOPE_AUDIT.json',{'status':'PASS','scope':'new Phase17R artifacts only','base_commit':base,'repository_wide_guard':'LEGACY_EXTENSION_VIOLATIONS' if inherited else 'PASS','unchanged_inherited_violations':inherited,'no_repository_wide_guard_PASS_claim':True})
    files=[ROOT/'configs/phase17r_conditional_repair.json']+sorted((ROOT/'src/phase17r_ssd').glob('*.py'))+sorted((ROOT/'scripts').glob('*phase17r*.py'))+sorted(p for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json')
    files=sorted(set(files));assert not protected_path_violations([p.relative_to(ROOT).as_posix() for p in files]);rows=[]
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for p in files:
        assert p.suffix in ['.py','.json','.md','.csv','.svg','.png'] and p.stat().st_size<8*1024**2,p
        b=p.read_bytes();assert not pattern.search(b),p
        if p.suffix=='.csv':assert not set(x.lower() for x in pd.read_csv(p,nrows=0).columns)&{'patient_id','patient_ids','sample_id','barcode','risk','prediction','rna_vector','clinical_text'},p
        rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':digest(p)})
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','source':source(),'public_files':rows,'scope':'verified exploratory Phase17R results; stopped/skipped stages explicitly retained','private_assets_excluded':['OOF risks','patient arrays','checkpoints','weights','H5/coords'],'counts':counts})
    print('PACKAGE_PASS',len(rows),digest(R/'ARTIFACT_MANIFEST.json'),flush=True)

if __name__=='__main__':main()

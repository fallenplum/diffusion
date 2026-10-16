"""Completion/privacy boundary for the finite Phase18A 2x2, preserving history."""
import sys,json,re,hashlib,subprocess
import pandas as pd
from src.phase18a_readout.data import ROOT,P,R,cfg,source,write,digest,old
sys.path.insert(0,str(ROOT/'src'))
from reproducibility import protected_path_violations

def main():
    for name in ['OPERATOR_PARITY','HISTORICAL_REUSE_AUDIT','HISTORICAL_READOUT_PROBE_AUDIT','DATA_LEAKAGE_AUDIT','INDEPENDENT_VALIDATION','CHECKPOINT_REPLAY','BOOTSTRAP_AUDIT','EVALUATION_MATH_AUDIT','CONTRAST_POINT_AUDIT','PARENT_PUBLISHED_METRIC_PARITY']:
        assert json.loads((R/(name+'.json')).read_text())['status']=='PASS',name
    c=cfg();s=source();done=json.loads((P/'TRAINING_DONE.json').read_text());lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());decision=json.loads((R/'decision.json').read_text())
    assert done['status']=='PASS' and done['source']==lock['source']==s
    assert done['new_inner_fits']==done['new_outer_fits']==30 and not done['phase18b_started']
    assert (R/'RESULTS.md').exists() and (R/'DECISION.md').exists() and (R/'FAILURE_LOG.md').exists()
    manifest=pd.read_csv(R/'CELL_MANIFEST.csv');assert len(manifest)==135 and int(manifest.new_optimization.sum())==60
    folds=pd.read_csv(R/'FOLD_METRICS.csv');seeds=pd.read_csv(R/'SEED_METRICS.csv')
    for (arm,seed),g in folds.groupby(['arm','seed']):
        row=seeds[(seeds.arm==arm)&(seeds.seed==seed)].iloc[0]
        assert len(g)==5 and g.patients.sum()==512 and g.events.sum()==65
        assert abs(g.H_num.sum()/g.H_den.sum()-row.Harrell)<1e-12 and abs(g.U_num.sum()/g.U_den.sum()-row.Uno60)<1e-12
    paired=pd.read_csv(R/'PAIRED_COMPARISONS.csv');primary=paired[paired.family=='primary'];assert len(primary)==8 and set(primary.family_tests)=={8} and set(primary.draws_requested)=={20000}
    assert old.source()==c['parent_training_source']
    metadata=json.loads((R/'OFFICIAL_SOURCE_METADATA.json').read_text())
    for path,row in metadata['parent_local_files'].items():assert digest(ROOT/path)==row['sha256'],path
    parent=json.loads((old.R/'ARTIFACT_MANIFEST.json').read_text())
    for row in parent['public_files']:assert digest(ROOT/row['path'])==row['sha256'],row['path']
    base=c['parent_commit'];base_paths=set(subprocess.check_output(['git','ls-tree','-r','--name-only',base],cwd=ROOT,text=True).splitlines());changed=set(subprocess.check_output(['git','diff','--name-only',base],cwd=ROOT,text=True).splitlines());assert not changed&base_paths
    write(R/'PARENT_IMMUTABILITY_AUDIT.json',{'status':'PASS','parent_commit':base,'parent_public_manifest_files_verified':len(parent['public_files']),'all_inherited_tracked_paths_unchanged':True,'parent_training_source':old.source(),'earlier_attention_branch_untouched':'a60aa99e7482d398b46c15738975105226871024'})
    inherited=[]
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().strip('\0').split('\0')
    for path in protected_path_violations(tracked):
        original=subprocess.check_output(['git','show',base+':'+path],cwd=ROOT);assert original==(ROOT/path).read_bytes();inherited.append({'path':path,'sha256':hashlib.sha256(original).hexdigest()})
    write(R/'PUBLICATION_SCOPE_AUDIT.json',{'status':'PASS','scope':'new Phase18A anonymous artifacts only','base_commit':base,'repository_wide_guard':'LEGACY_EXTENSION_VIOLATIONS' if inherited else 'PASS','unchanged_inherited_violations':inherited,'no_repository_wide_PASS_claim':True})
    files=[ROOT/'configs/phase18a_local_state_readout.json']+list((ROOT/'src/phase18a_readout').glob('*.py'))+list((ROOT/'scripts').glob('*phase18a*.py'))+list((ROOT/'tests').glob('test_phase18a*.py'))
    files += [p for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json']
    files=sorted(set(files));assert not protected_path_violations([p.relative_to(ROOT).as_posix() for p in files]);rows=[]
    private=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for p in files:
        assert p.suffix in ['.py','.json','.md','.csv'] and p.stat().st_size<8*1024**2,p
        raw=p.read_bytes();assert not private.search(raw),p
        if p.suffix=='.csv':assert not set(pd.read_csv(p,nrows=0).columns.str.lower())&{'patient_id','patient_ids','sample_id','barcode','risk','prediction','target_value','position_probe_target_value'},p
        rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':digest(p)})
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','source':s,'public_files':rows,'formal_new_fits':60,'historical_primary_references':60,'historical_secondary_A3_references':15,'historical_CONCH32_outer_references':15,'diagnostic_short_fits':done['diagnostic_short_fits'],'no_Phase18B':True,'private_excluded':['patient OOF','raw matrices','trained weights','patient identifiers','private probe target risks','credentials']})
    print('PHASE18A_PACKAGE_PASS',len(rows),flush=True)

if __name__=='__main__':main()

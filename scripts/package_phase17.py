"""Artifact, completion, privacy and immutability checks before publication."""
import sys,json,hashlib,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from src.phase17_ssd.data import P,R,source,write
from src.phase10_trimodal.data import digest

def main():
    gates=['SSM_OPERATOR_AUDIT','DATA_LEAKAGE_AUDIT','STATISTICAL_OPERATOR_AUDIT','PREPROCESSING_AUDIT','ABLATION_OPERATOR_AUDIT','MAMBAMIL_KERNEL_PORT_AUDIT','FULL_BUDGET_AUDIT','HISTORICAL_IMMUTABILITY_AUDIT','VISUAL_CACHE_INDEPENDENT_AUDIT','INDEPENDENT_VALIDATION','CHECKPOINT_REPLAY','SIMPLE_CHECKPOINT_REPLAY','DETAILED_MECHANISM_AUDIT','SUBGROUP_AUDIT','FIGURE_VISUAL_QA']
    for name in gates:
        result=json.loads((R/(name+'.json')).read_text());assert result['status']=='PASS',name
    assert json.loads((R/'CHECKPOINT_REPLAY.json').read_text())['replayed']==195
    assert json.loads((R/'PROTOCOL_LOCK.json').read_text())['training_source']==source()
    count=json.loads((R/'EXECUTION_COUNTS.json').read_text());assert count['all_outer_fits']==205 and count['new_inner_optimizer_fits_total']==249
    fold=pd.read_csv(R/'FOLD_METRICS.csv');seed=pd.read_csv(R/'SEED_OOF_METRICS.csv');assert len(fold)==255 and len(seed)==51
    assert not fold.duplicated(['arm','seed','fold']).any() and not seed.duplicated(['arm','seed']).any()
    for (arm,s),g in fold.groupby(['arm','seed']):
        row=seed[(seed.arm==arm)&(seed.seed==s)].iloc[0]
        assert len(g)==5 and int(g.events.sum())==65 and int(g.n.sum())==512
        assert abs(g.H_num.sum()/g.H_den.sum()-row.Harrell)<1e-12 and abs(g.U_num.sum()/g.U_den.sum()-row.Uno60)<1e-12
    primary=pd.read_csv(R/'PAIRED_COMPARISONS.csv').query("family=='primary'");assert len(primary)==4 and (primary.family_tests==4).all() and (primary.draws_valid==20000).all()
    required=['INPUT_AUDIT.md','LITERATURE_BASELINES.md','INNER_DIAGNOSIS.md','ABLATION_REGISTRY.csv','FOLD_METRICS.csv','SEED_OOF_METRICS.csv','PAIRED_COMPARISONS.csv','MODULE_MECHANISM.csv','ORDER_SENSITIVITY.csv','TRAINING_DIAGNOSTICS.csv','MODEL_SUMMARY.csv','BUG_LOG.csv','RESULTS.md','DECISION.md','REPRODUCE.md']
    for name in required:assert (R/name).stat().st_size>0,name
    # Matplotlib emits spaces before SVG newlines; preserve geometry and XML.
    import xml.etree.ElementTree as ET
    for svg in (R/'figures').glob('*.svg'):
        old=svg.read_bytes();new=re.sub(rb'[ \t]+(?=\r?$)',b'',old,flags=re.MULTILINE)
        def tree_signature(data):
            return [(e.tag,sorted((k,' '.join(v.split())) for k,v in e.attrib.items()),(e.text or '').strip(),(e.tail or '').strip()) for e in ET.fromstring(data).iter()]
        assert tree_signature(old)==tree_signature(new),svg
        if new!=old:svg.write_bytes(new)
    # Record inherited repository policy findings without changing old reports.
    sys.path.insert(0,str(ROOT/'src'))
    from reproducibility import protected_path_violations
    base='7dca7cf9d72b61517baa85b7493b1f6ea750460d'
    subprocess.run(['git','merge-base','--is-ancestor',base,'HEAD'],cwd=ROOT,check=True)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().strip('\0').split('\0')
    inherited=[]
    for path in protected_path_violations(tracked):
        old=subprocess.check_output(['git','show',base+':'+path],cwd=ROOT)
        assert old==(ROOT/path).read_bytes(),path
        assert old==subprocess.check_output(['git','show',':'+path],cwd=ROOT),path
        inherited.append({'path':path,'sha256':hashlib.sha256(old).hexdigest(),'changed_in_phase17':False})
    write(R/'PUBLICATION_SCOPE_AUDIT.json',{'status':'PASS','scope':'Phase17 incremental publication only','repository_wide_guard':'LEGACY_EXTENSION_VIOLATIONS' if inherited else 'PASS','base_commit':base,'inherited_unchanged_violations':inherited,'note':'Historical public audit ZIPs trigger the general extension ban; no new ZIP or inherited artifact is included in this change. This is not a repository-wide guard PASS.'})
    files=[ROOT/'configs/phase17_ssd_fusion.json']+sorted((ROOT/'src/phase17_ssd').glob('*.py'))+sorted((ROOT/'scripts').glob('*phase17*.py'))
    files+=sorted(p for p in R.rglob('*') if p.is_file() and p.name not in ['ARTIFACT_MANIFEST.json','DELIVERY.json'])
    files=sorted(set(files));pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    assert not protected_path_violations([p.relative_to(ROOT).as_posix() for p in files])
    rows=[]
    for p in files:
        assert p.suffix in ['.py','.json','.md','.csv','.svg','.png'] and p.stat().st_size<8*1024**2,p
        data=p.read_bytes();assert not pattern.search(data),p
        if p.suffix=='.csv':
            cols=pd.read_csv(p,nrows=0).columns
            assert not any(c.lower() in ['patient_id','patient_ids','sample_id','barcode','risk','prediction','rna_vector','clinical_text'] for c in cols),p
        rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':digest(p)})
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','scope':'public artifact checksum/privacy/completion gates;not a claim that missing TransSurv was run','training_source':source(),'public_files':rows,'private_assets_excluded':['patient arrays','OOF vectors','checkpoints','original H5/coords','weights','third-party private checkouts'],'gate_files':gates,'paper_baseline_target':'PARTIAL:MGCT and MambaMIL15cells each;TransSurv BLOCKED'})
    write(R/'DELIVERY.json',{'status':'CORE_COMPLETE_BASELINE_TARGET_PARTIAL','manifest_sha256':digest(R/'ARTIFACT_MANIFEST.json'),'public_files':len(rows),'core_outer_cells':165,'paper_outer_cells':30,'simple_outer_fits':10,'individual_patient_data_uploaded':False,'conclusion':'MECHANISM_NOT_SUPPORTED','remaining_blocker':'TransSurv verifiable official code and multiscale input absent'})
    print('PACKAGE_PASS',len(rows),digest(R/'ARTIFACT_MANIFEST.json'),flush=True)

if __name__=='__main__':main()

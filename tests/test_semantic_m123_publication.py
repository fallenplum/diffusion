import csv,hashlib,json,re
from pathlib import Path
import pytest
from scripts import semantic_m123_publication as pub


@pytest.mark.parametrize('name',[
 'src/semantic_m123/models.py','scripts/semantic_m123_report.py','tests/test_semantic_m123_report.py',
 'configs/semantic_m123/master_public.json','reports/semantic_m123/RESULTS.md',
 'M1_M5_JOINT_TRIMODAL_REDESIGN_PROMPT.md'])
def test_only_explicit_new_namespace_accepted(name):assert pub.allowed(name)


@pytest.mark.parametrize('name',[
 '../secrets.json','/absolute/key.json','reports/semantic_m123/x/array.csv',
 'reports/semantic_m123/risk.npz','reports/semantic_b0b7_p2/RESULTS.md',
 r'reports\semantic_m123\x.json','src/semantic_m123/x.txt','configs/semantic_m123/master_lock.json'])
def test_foreign_nested_binary_and_escaping_paths_blocked(name):assert not pub.allowed(name)


def test_original_scanner_rejects_patient_vector_json_without_rule_changes(tmp_path):
    p=tmp_path/'reports/semantic_m123/x.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'risk':[.1,.2]}))
    row=pub.row(p,'reports/semantic_m123/x.json')
    old=pub.original.ALLOWLIST
    with pytest.raises(ValueError,match='raw array'):
        pub.scan(tmp_path,[row],re.compile('(?!)'))
    assert pub.original.ALLOWLIST==old
    assert pub.sha(pub.original.__file__)==pub.SCANNER


def test_anonymous_aggregate_passes_real_original_format_rules(tmp_path):
    p=tmp_path/'reports/semantic_m123/x.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'Harrell':.7,'folds':5,'methodological_risks':['reused cohort']}))
    row=pub.row(p,'reports/semantic_m123/x.json')
    assert pub.scan(tmp_path,[row],re.compile('(?!)'))==[row]


def test_changed_candidate_bytes_fail_even_if_text_still_safe(tmp_path):
    p=tmp_path/'reports/semantic_m123/x.md';p.parent.mkdir(parents=True);p.write_text('safe')
    row=pub.row(p,'reports/semantic_m123/x.md');p.write_text('changed')
    with pytest.raises(ValueError,match='CHANGED'):pub.scan(tmp_path,[row],re.compile('(?!)'))


def test_old_parent_baseline_remains_fixed():
    p=Path('reports/semantic_m123/PARENT536_BASELINE.json')
    b=pub.verify_baseline(p)
    assert b['file_count']==536 and b['commit']==pub.BASE and b['tree']==pub.TREE


def geometry_fixture(root):
    name='configs/semantic_m123/diagnostic_lock.json'
    raw=Path(name).read_bytes();lock=json.loads(raw)
    path=root/name;path.parent.mkdir(parents=True);path.write_bytes(raw)
    indexed={name:pub.row(path,name)}
    for name in lock['sources']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(name).read_bytes())
        indexed[name]=pub.row(p,name)
    table=root/(pub.REPORTS+'FROZEN_ATTENTION_GEOMETRY.csv');table.parent.mkdir(parents=True)
    with table.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['outer','arm','seed'])
        for o in range(1,6):
            for a in ('M2_SINGLE','M2_EIGHT','M2_EIGHT_MEAN','M3_FULL','M3_LEARNED_QUERY','M3_NO_PRODUCT','M3_BIDIRECTIONAL'):
                for s in (31,47,83):w.writerow([o,a,s])
    indexed[pub.REPORTS+table.name]=pub.row(table,pub.REPORTS+table.name)
    audit={'schema':'m123_frozen_attention_geometry_audit_v1','status':'COMPLETE',
        'actual_model_forwards':105,'saved_model_states':105,'master_lock_sha256':pub.MASTER,
        'diagnostic_lock_sha256':pub.DIAGNOSTIC,'source_sha256':lock['sources'],
        'survival_endpoint_arrays_decoded':False,'new_optimizer_fits':0,'new_preprocessor_fits':0,
        'file':{'file':table.name,'sha256':pub.sha(table)}}
    p=root/(pub.REPORTS+'FROZEN_ATTENTION_GEOMETRY_AUDIT.json');p.write_text(json.dumps(audit))
    return indexed,p,audit,table


def test_geometry_requires_actual_complete_matrix_and_exact_sources(tmp_path):
    indexed,_,_,_=geometry_fixture(tmp_path)
    result=pub.verify_geometry(tmp_path,indexed)
    assert result['frozen_geometry_forwards']==105 and result['new_geometry_training_fits']==0
    indexed['src/semantic_m123/diagnostics.py']['sha256']='0'*64
    with pytest.raises(ValueError,match='SOURCE_CLOSURE'):pub.verify_geometry(tmp_path,indexed)


@pytest.mark.parametrize('key,value',[
 ('actual_model_forwards',104),('new_optimizer_fits',1),('survival_endpoint_arrays_decoded',True)])
def test_geometry_rejects_incomplete_or_expanded_diagnostics(tmp_path,key,value):
    indexed,p,audit,_=geometry_fixture(tmp_path);audit[key]=value;p.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='NOT_ACTUALLY_COMPLETE'):pub.verify_geometry(tmp_path,indexed)


def test_geometry_duplicate_and_missing_cell_cannot_pass_by_row_count(tmp_path):
    indexed,p,audit,table=geometry_fixture(tmp_path)
    lines=table.read_text().splitlines();lines[-1]=lines[-2];table.write_text('\n'.join(lines)+'\n')
    indexed[pub.REPORTS+table.name]=pub.row(table,pub.REPORTS+table.name)
    audit['file']['sha256']=pub.sha(table);p.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='COVERAGE'):pub.verify_geometry(tmp_path,indexed)


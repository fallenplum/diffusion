import json
from scripts import bgb_en_publication_dependencies as audit


def test_ast_closure_finds_real_module_without_mistaking_imported_symbol(tmp_path):
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/a.py').write_text('from scripts import bgb_en_target\nfrom scripts.bgb_en_target import helper\n')
    (tmp_path/'scripts/bgb_en_target.py').write_text('def helper(): pass\n')
    report=audit.analyze(tmp_path,{'scripts/a.py'})
    assert report['unlisted_BG_BEN_dependencies']==['scripts/bgb_en_target.py']
    assert report['status'].startswith('FAIL')
    assert audit.analyze(tmp_path,{'scripts/a.py','scripts/bgb_en_target.py'})['status'].startswith('PASS')


def test_certificate_reference_requires_old_script_but_file_presence_alone_does_not(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'reports').mkdir()
    (tmp_path/'scripts/bgb_en_audit_offset_cast.py').write_text('# old unused\n')
    report=audit.analyze(tmp_path,set())
    assert report['old_offset_cast']['file_present'] is True
    assert report['old_offset_cast']['publication_required_by_reviewed_files'] is False
    (tmp_path/'reports/ok.json').write_text(json.dumps({'source_sha256':{'scripts/bgb_en_audit_offset_cast.py':'a'*64}}))
    report=audit.analyze(tmp_path,{'reports/ok.json'})
    assert report['old_offset_cast']['publication_required_by_reviewed_files'] is True
    assert report['unlisted_BG_BEN_dependencies']==['scripts/bgb_en_audit_offset_cast.py']


def test_peer_source_comparison_preserves_drift_without_calling_it_completed():
    a={'files_reviewed':[{'file':'scripts/a.py','sha256':'a'},
        {'file':'tests/bgb_en/test_a.py','sha256':'a'},
        {'file':'reports/bgb_en/R1_RESULTS.md','sha256':'a'}]}
    b={'files_reviewed':[{'file':'scripts/a.py','sha256':'b'},
        {'file':'src/bgb_en/data.py','sha256':'b'}]}
    result=audit.peer_comparison(a,b)
    assert result['only_current']==['tests/bgb_en/test_a.py']
    assert result['only_peer']==['src/bgb_en/data.py']
    assert result['different_source_bytes']==[{'file':'scripts/a.py','current_sha256':'a','peer_sha256':'b'}]


def test_full_suite_is_not_claimed_for_later_plot_changes_and_core_drift_fails(tmp_path):
    (tmp_path/'scripts').mkdir();(tmp_path/'reports/bgb_en').mkdir(parents=True)
    plot='scripts/bgb_en_plot.py';report='reports/bgb_en/TEST_SUITE_AUDIT.json'
    (tmp_path/plot).write_text('# newer plot\n')
    value=dict(schema='bgb_en_synthetic_cpu_test_suite_audit_v1',status='PASS_SYNTHETIC_CPU_TEST_SUITE',
               production_and_test_source_sha256={plot:'old'},counts={'passed':336})
    (tmp_path/report).write_text(json.dumps(value))
    result=audit.analyze(tmp_path,{plot,report})
    assert result['status'].startswith('PASS')
    assert result['full_suite_byte_scope']['full_suite_not_claimed_to_cover_post_suite_changes'] is True
    core='src/bgb_en/training.py';(tmp_path/'src/bgb_en').mkdir(parents=True);(tmp_path/core).write_text('# changed\n')
    value['production_and_test_source_sha256'][core]='old'
    (tmp_path/report).write_text(json.dumps(value))
    assert audit.analyze(tmp_path,{plot,report,core})['status']=='FAIL_POST_SUITE_UNREVIEWED_SOURCE_CHANGE'

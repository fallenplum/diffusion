from scripts import bgb_en_test_suite_audit as audit
import pytest
import json


def test_junit_counts_use_real_failures_errors_skips(tmp_path):
    path=tmp_path/'junit.xml'
    path.write_text('<testsuites><testsuite tests="9" failures="1" errors="2" skipped="3"/></testsuites>')
    assert audit.junit_counts(path)==dict(tests=9,failures=1,errors=2,skipped=3,passed=3)
    assert audit.warning_count('old 3 warnings\n15 passed, 2 warnings in 3.5s')==2
    assert audit.warning_count('18 passed in 2.1s')==0


def test_existing_basetemp_is_never_deleted_or_reused(tmp_path):
    base=tmp_path/'already_used';base.mkdir();sentinel=base/'keep.txt';sentinel.write_text('preserve')
    with pytest.raises(ValueError,match='already exists'):
        audit.execute(base)
    assert sentinel.read_text()=='preserve'


def test_prior_fixture_failure_is_recorded_without_promoting_first_attempt(tmp_path):
    xml=tmp_path/'old.xml';log=tmp_path/'old.log';receipt=tmp_path/'old.json'
    xml.write_text('<testsuites><testsuite tests="336" failures="2"><testcase><failure message="FileNotFoundError O9_FLOAT32_DRIFT_DEFAULT.json"/></testcase><testcase><failure message="FileNotFoundError O9_FLOAT32_DRIFT_DEFAULT.json"/></testcase></testsuite></testsuites>')
    log.write_text('334 passed, 2 failed in 10s')
    snapshot={'scripts/bgb_en_plot.py':'old'}
    receipt.write_text(json.dumps(dict(source_before=snapshot,source_after=snapshot,
        private_junit=str(xml),private_log=str(log),returncode=1)))
    report=dict(status='PASS_SYNTHETIC_CPU_TEST_SUITE',production_and_test_source_sha256=snapshot,
                counts={'passed':336})
    result=audit.add_history(report,[receipt],{'scripts/bgb_en_plot.py':'new'})
    assert result['counts']['passed']==336 and result['prior_attempts'][0]['counts']['passed']==334
    assert result['first_attempt_was_not_all_passed'] is True
    assert result['post_suite_diagnostic_source_changes'][0]['tested_sha256']=='old'
    with pytest.raises(ValueError,match='outside the approved'):
        audit.add_history(report,[receipt],{'src/bgb_en/training.py':'changed'})

"""Only synthetic scheduling metadata; no process or GPU launch."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from scripts import tune_phase28_wide_concurrency as wide
from scripts import publish_phase28_formal as publication
spec = importlib.util.spec_from_file_location('wide_publisher_fixture', Path(__file__).with_name('test_phase28_formal_publish.py'))
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
bound_final_toolchain = fixture.bound_final_toolchain


def snapshot(parallel=8, epoch=3, when=1000):
    raw = {'cgroup_v2': {'memory.max': str(62*wide.GIB), 'memory.current': str(27*wide.GIB),
                        'cpu.max': '1600000 100000'},
           'GPU_csv_MiB_MiB_percent_percent': '16384, 32768, 60, 10',
           'disk_available_bytes': 8*wide.GIB}
    return {'target': parallel, 'observed_time': when,
            'running': {str(k): {'pid': k+100, 'created': 100., 'epoch': epoch,
                                 'arm': 'C11', 'fold': 1} for k in range(parallel)},
            'resource': wide.old.resource_summary(raw)}


@pytest.mark.parametrize('requested', [12, 13])
def test_wide_admission_retains_gpu_and_cgroup_margin(requested):
    check = wide.admission(snapshot()['resource'], 8, requested)
    assert check['safe'] and check['projected_GPU_bytes'] <= 28*wide.GIB
    assert check['projected_cgroup_bytes'] <= 54*wide.GIB


@pytest.mark.parametrize('kind', ['GPU', 'RAM', 'CPU', 'DISK'])
def test_wide_admission_rejects_actual_resource_constraints(kind):
    r = snapshot()['resource']
    if kind == 'GPU': r.update(GPU_used_bytes=23*wide.GIB, GPU_available_bytes=9*wide.GIB)
    if kind == 'RAM': r.update(cgroup_memory_current_bytes=40*wide.GIB, cgroup_memory_available_bytes=22*wide.GIB)
    if kind == 'CPU': r['cgroup_CPU_cores_quota'] = 10
    if kind == 'DISK': r['disk_available_bytes'] = wide.GIB-1
    assert not wide.admission(r, 8, 12)['safe']


def test_live_estimator_discards_first_interval_and_requires_complete_intervals():
    s = snapshot(); w = wide.window(s); wide.observe(w, s)
    second = snapshot(epoch=4, when=1200); wide.observe(w, second)
    assert wide.finish(w) is None
    third = snapshot(epoch=5, when=1500); wide.observe(w, third)
    result = wide.finish(w)
    assert result['throughput_estimated_epochs_per_hour'] == pytest.approx(8*3600/300)
    assert result['minimum_complete_intervals_per_process'] == 1
    assert result['first_live_observation_discarded']
    publication.privacy_json(result)


def test_pid_or_epoch_rollback_never_mixes_throughput():
    s = snapshot(); w = wide.window(s); wide.observe(w, s)
    changed = copy.deepcopy(s); changed['running']['0']['pid'] = 999
    with pytest.raises(ValueError, match='roster'): wide.observe(w, changed)
    changed = copy.deepcopy(s); changed['running']['0']['epoch'] = 2
    with pytest.raises(ValueError, match='backwards'): wide.observe(w, changed)


def test_historical_reference_uses_only_warm_current_roster(tmp_path):
    s = snapshot(epoch=6, when=wide.old.utc_seconds('2026-10-20T00:40:00Z')); path = tmp_path/'history.jsonl'
    raw_resource = {'cgroup_v2': {'memory.max': str(62*wide.GIB), 'memory.current': str(27*wide.GIB),
                                'cpu.max': '1600000 100000'},
        'GPU_csv_MiB_MiB_percent_percent': '16384, 32768, 60, 10', 'disk_available_bytes': 8*wide.GIB}
    lines = []
    for epoch, stamp in ((0, '00:01:00'), (3, '00:20:00'), (4, '00:25:00'), (5, '00:30:00')):
        lines.append(json.dumps({'utc': '2026-10-20T'+stamp+'Z', 'parallel': 8,
            'running': list(s['running']), 'epochs': {name: epoch for name in s['running']},
            'resource': raw_resource}))
    path.write_text('\n'.join(lines))
    result = wide.historical_baseline(s, path)
    assert result['throughput_estimated_epochs_per_hour'] == pytest.approx(96.)
    assert result['historical_warm_roster_metadata']


def test_control_change_preserves_resume_flags_and_rejects_stale_or_unsafe_write(tmp_path, monkeypatch):
    p = tmp_path/'CONTROL.json'; before = {'max_parallel': 8, 'resume_failed': {'x': False}}
    p.write_text(json.dumps(before)); monkeypatch.setattr(wide.old, 'CONTROL', p)
    wide.update_control(8, 12)
    assert json.loads(p.read_text()) == {**before, 'max_parallel': 12}
    with pytest.raises(ValueError, match='external'): wide.update_control(8, 13)
    with pytest.raises(ValueError, match='cap1'): wide.update_control(12, 14)


def test_wide_receipt_hides_roster_and_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(wide, 'STATE', tmp_path/'private.json')
    monkeypatch.setattr(wide, 'REPORT', tmp_path/'public.json')
    state = {'schema': 'synthetic', 'status': 'OBSERVING', 'source': 'a'*64,
             'script_sha256': 'b'*64, 'stages': [], 'decisions': [], 'target': 12,
             'reason': None, 'started_utc': 'synthetic', 'owner_pid_private': 777}
    wide.save(state)
    public = json.loads((tmp_path/'public.json').read_text())
    assert 'owner_pid_private' not in public and public['existing_workers_signaled'] == 0
    publication.privacy_json(public)


@pytest.mark.parametrize('tamper', [None, 'previous_receipt', 'new_tool_hash'])
def test_wide_qualification_is_bound_to_new_tools_and_previous_final(tmp_path, tamper):
    report = bound_final_toolchain(tmp_path)
    directory = tmp_path/publication.REPORT_REL
    report['revision'] = 'WIDE'
    report['previous_FINAL_receipt_sha256'] = publication.sha(directory/'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json')
    for name in publication.WIDE_TEST_FILES|publication.WIDE_TOOL_FILES:
        path = tmp_path/name
        if not path.exists(): path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic = True\n')
    report['test_files_sha256'] = {name: publication.sha(tmp_path/name) for name in publication.WIDE_TEST_FILES}
    report['tool_files_sha256'] = {name: publication.sha(tmp_path/name) for name in publication.WIDE_TOOL_FILES}
    if tamper == 'previous_receipt': report['previous_FINAL_receipt_sha256'] = 'f'*64
    if tamper == 'new_tool_hash': report['tool_files_sha256']['scripts/tune_phase28_wide_concurrency.py'] = 'f'*64
    (directory/'FORMAL_TOOLCHAIN_QUALIFICATION_WIDE.json').write_text(json.dumps(report))
    if tamper:
        with pytest.raises(ValueError): publication.verify_final_toolchain(root=tmp_path, source='a'*64)
    else:
        verified = publication.verify_final_toolchain(root=tmp_path, source='a'*64)
        assert verified['bound_test_files'] == 7 and verified['bound_tool_files'] == 10
        publication.privacy_json(report)


@pytest.mark.parametrize('tamper', [None, 'evaluation_changed', 'previous_wide_changed'])
def test_postfit_numeric_qualification_preserves_once_only_evaluation(tmp_path, tamper):
    report = bound_final_toolchain(tmp_path)
    directory = tmp_path/publication.REPORT_REL
    report['revision'] = 'WIDE'
    report['previous_FINAL_receipt_sha256'] = publication.sha(directory/'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json')
    for name in publication.WIDE_TEST_FILES|publication.WIDE_TOOL_FILES:
        path = tmp_path/name
        if not path.exists(): path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic = True\n')
    report['test_files_sha256'] = {name: publication.sha(tmp_path/name) for name in publication.WIDE_TEST_FILES}
    report['tool_files_sha256'] = {name: publication.sha(tmp_path/name) for name in publication.WIDE_TOOL_FILES}
    previous = directory/'FORMAL_TOOLCHAIN_QUALIFICATION_WIDE.json'; previous.write_text(json.dumps(report))
    report['revision'] = 'POSTFIT'; report['previous_WIDE_receipt_sha256'] = publication.sha(previous)
    names = ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json', 'REPLAY_AUDIT.json')
    for name in names: (directory/name).write_text('{"synthetic":true}')
    report['preserved_evaluation_report_sha256'] = {name: publication.sha(directory/name) for name in names}
    if tamper == 'evaluation_changed': (directory/'INNER_RESULTS.json').write_text('{"synthetic":false}')
    if tamper == 'previous_wide_changed': previous.write_text('{"synthetic":false}')
    (directory/'FORMAL_TOOLCHAIN_QUALIFICATION_POSTFIT.json').write_text(json.dumps(report))
    if tamper:
        with pytest.raises(ValueError): publication.verify_final_toolchain(root=tmp_path, source='a'*64)
    else:
        result = publication.verify_final_toolchain(root=tmp_path, source='a'*64)
        assert result['bound_tool_files'] == 10


@pytest.mark.parametrize('tamper', [None, 'completed_report', 'previous_POSTFIT'])
def test_publication_only_qualification_keeps_completed_scientific_reports(tmp_path, tamper):
    report = bound_final_toolchain(tmp_path); directory = tmp_path/publication.REPORT_REL
    report['revision'] = 'PUBLICATION'
    report['previous_FINAL_receipt_sha256'] = publication.sha(directory/'FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json')
    for name in publication.WIDE_TEST_FILES|publication.WIDE_TOOL_FILES:
        path = tmp_path/name
        if not path.exists(): path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic = True\n')
    report['test_files_sha256'] = {name: publication.sha(tmp_path/name) for name in publication.WIDE_TEST_FILES}
    report['tool_files_sha256'] = {name: publication.sha(tmp_path/name) for name in publication.WIDE_TOOL_FILES}
    for revision in ('WIDE', 'POSTFIT'):
        path = directory/('FORMAL_TOOLCHAIN_QUALIFICATION_'+revision+'.json')
        path.write_text('{"synthetic":true}')
        report['previous_'+revision+'_receipt_sha256'] = publication.sha(path)
    names = ('INNER_RESULTS.json', 'INNER_GATE.json', 'decision.json', 'REPLAY_AUDIT.json')
    completed = ('BUDGET_READOUT_AUDIT.json', 'FORMAL_DELIVERY.json', 'RESULTS_FORMAL.md',
                 'MECHANISM_SUMMARY.json', 'MECHANISM_RESULTS.md')
    for name in names+completed: (directory/name).write_text('synthetic completed report')
    report['preserved_evaluation_report_sha256'] = {name: publication.sha(directory/name) for name in names}
    report['preserved_completed_report_sha256'] = {name: publication.sha(directory/name) for name in completed}
    if tamper == 'completed_report': (directory/'RESULTS_FORMAL.md').write_text('modified')
    if tamper == 'previous_POSTFIT': report['previous_POSTFIT_receipt_sha256'] = 'f'*64
    (directory/'FORMAL_TOOLCHAIN_QUALIFICATION_PUBLICATION.json').write_text(json.dumps(report))
    if tamper:
        with pytest.raises(ValueError): publication.verify_final_toolchain(root=tmp_path, source='a'*64)
    else:
        assert publication.verify_final_toolchain(root=tmp_path, source='a'*64)['bound_test_files'] == 7

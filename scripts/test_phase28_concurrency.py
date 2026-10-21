"""Synthetic metadata-only tests; no processes, GPU or experiment files."""
import copy
import json
import time

import pytest

from scripts import tune_phase28_concurrency as tune


def resource(memory_used=8, gpu_used=4):
    return {'utc': '2026-10-20T00:00:00Z', 'RAM_total_bytes': 540*tune.GIB,
        'RAM_available_bytes': 500*tune.GIB,
        'cgroup_v2': {'memory.max': str(62*tune.GIB), 'memory.current': str(memory_used*tune.GIB),
                      'cpu.max': '1600000 100000'},
        'GPU_csv_MiB_MiB_percent_percent': f'{gpu_used*1024}, 32768, 68, 31',
        'disk_available_bytes': 8*tune.GIB}


def snapshot(parallel=2, epoch=3, when=1000, active=None):
    count = parallel if active is None else active
    return {'observed_time': float(when), 'source': tune.SOURCE, 'status': 'RUNNING',
        'target': parallel, 'complete': 0, 'failed_count': 0,
        'running': {f'cell{j}': {'pid': j+100, 'created': 100., 'epoch': epoch,
                                  'attempt_wall_seconds': 800.} for j in range(count)},
        'resource': tune.resource_summary(resource())}


def test_resource_uses_cgroup_limit_and_not_540_gb_host_ram():
    r = tune.resource_summary(resource())
    assert r['cgroup_memory_limit_bytes'] == 62*tune.GIB
    assert r['cgroup_CPU_cores_quota'] == 16
    assert r['cgroup_memory_available_bytes'] == 54*tune.GIB


def test_unlimited_or_missing_cgroup_memory_is_not_a_host_fallback():
    raw = resource(); raw['cgroup_v2']['memory.max'] = 'max'
    with pytest.raises(ValueError, match='cgroup memory'):
        tune.resource_summary(raw)


def test_multiple_gpu_rows_fail_closed():
    raw = resource(); raw['GPU_csv_MiB_MiB_percent_percent'] += '\n0, 32000, 0, 0'
    with pytest.raises(ValueError, match='one GPU'):
        tune.resource_summary(raw)


def test_current_memory_pressure_stops_new_admission_without_pid_actions():
    s = snapshot(); s['resource'] = tune.resource_summary(resource(memory_used=58))
    t = tune.initialize(s, 8)
    assert tune.step(t, s) == 1
    assert t['status'] == 'STOPPED_RESOURCE_MARGIN'


def test_full_roster_and_two_commits_per_worker_are_required():
    s = snapshot(active=1); t = tune.initialize(s, 8)
    assert tune.step(t, s) is None and t['window'] is None
    s = snapshot(); assert tune.step(t, s) is None
    one = snapshot(epoch=4, when=1300)
    assert tune.step(t, one) is None
    two = snapshot(epoch=5, when=1600)
    assert tune.step(t, two) == 4
    assert t['stages'][0]['committed_epoch_count'] == 4
    assert t['stages'][0]['window_seconds'] == 600
    assert t['stages'][0]['guaranteed_wholly_observed_epochs_per_process_at_least'] == 1
    assert t['stages'][0]['all_workers_had_committed_epoch_at_window_start']
    assert not t['stages'][0]['cold_start_excluded_by_stable_roster_window']


def test_roster_or_pid_change_restarts_window_without_mixing_progress():
    s = snapshot(); t = tune.initialize(s, 8); tune.step(t, s)
    changed = snapshot(epoch=20, when=1300)
    changed['running']['cell0']['pid'] = 999
    assert tune.step(t, changed) is None
    assert t['window']['epochs_at_start']['cell0'] == 20
    assert not t['stages']


def test_lower_throughput_caps_to_previous_level_only():
    s = snapshot(parallel=4)
    historical = {'parallel': 2, 'committed_epochs_per_hour': 100.}
    t = tune.initialize(s, 8, historical); tune.step(t, s)
    later = snapshot(parallel=4, epoch=5, when=2800)
    assert tune.step(t, later) == 2
    assert t['status'] == 'STOPPED_THROUGHPUT_DECLINE'


def test_projected_margin_prevents_next_level_without_changing_training():
    s = snapshot(parallel=2); s['resource'] = tune.resource_summary(resource(memory_used=30))
    t = tune.initialize(s, 8); tune.step(t, s)
    later = copy.deepcopy(s); later['observed_time'] = 1600
    for row in later['running'].values(): row['epoch'] += 2
    assert tune.step(t, later) is None
    assert t['status'] == 'STOPPED_PROJECTED_RESOURCE_MARGIN'
    assert t['target'] == 2


def test_coordinator_snapshot_lag_is_tolerated_but_external_changes_are_not():
    s = snapshot(parallel=2); t = tune.initialize(s, 8)
    t['target'] = 4; t['pending_control'] = {'before': 2, 'after': 4, 'issued_time': 990.}
    assert tune.step(t, s) is None
    s['observed_time'] = 1200
    with pytest.raises(ValueError, match='external concurrency'):
        tune.step(t, s)


def test_maximum_level_completes_after_full_observation():
    s = snapshot(parallel=8); t = tune.initialize(s, 8); tune.step(t, s)
    later = snapshot(parallel=8, epoch=5, when=2000)
    assert tune.step(t, later) is None
    assert t['status'] == 'COMPLETE_MAXIMUM_TESTED'
    assert len(t['stages']) == 1
    public = tune.public_report(t, 'synthetic-script-sha')
    def inspect(value):
        if isinstance(value, dict):
            assert not {'pid', 'created', 'roster', 'epochs_at_start', 'latest_process_start', 'window'} & set(value)
            for item in value.values(): inspect(item)
        elif isinstance(value, list):
            assert not (value and all(isinstance(item, (int, float)) for item in value))
            for item in value: inspect(item)
    inspect(public)


def test_control_update_preserves_every_other_field_and_never_exceeds_eight(tmp_path, monkeypatch):
    p = tmp_path/'CONTROL_PRIVATE.json'
    content = {'max_parallel': 2, 'resume_failed': {'x': False}, 'other': {'locked': True}}
    p.write_text(json.dumps(content)); monkeypatch.setattr(tune, 'CONTROL', p)
    receipt = tune.update_control(4)
    assert json.loads(p.read_text()) == {**content, 'max_parallel': 4}
    assert receipt['existing_processes_signaled'] == 0
    with pytest.raises(ValueError, match='1..8'): tune.update_control(12)


def test_sanitized_snapshot_never_reads_cox_and_refuses_stale_resources():
    when = tune.utc_seconds('2026-10-20T00:00:00Z')
    class Progress(dict):
        def get(self, key, default=None):
            assert key != 'fit_cox'
            return super().get(key, default)
    state = {'source': tune.SOURCE, 'utc': '2026-10-20T00:00:00Z', 'status': 'RUNNING',
        'max_parallel': 1, 'complete_count': 0, 'failed': {}, 'resource': resource(),
        'running': {'x': {'pid': 1, 'create_time': when-100}}}
    progress = {'x': Progress(source=tune.SOURCE, epoch=1, wall_seconds=80, fit_cox=object())}
    result = tune.sanitized_snapshot(state, progress, observed_time=when+10)
    assert result['running']['x']['epoch'] == 1
    with pytest.raises(ValueError, match='stale'):
        tune.sanitized_snapshot(state, progress, observed_time=when+100)

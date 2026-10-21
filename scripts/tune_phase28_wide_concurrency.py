"""User-authorized 8 -> 12 -> 13 scheduling, never predictor selection.

Uses existing live-process epoch/resource metadata only. Existing workers are
never stopped. Reference throughput uses complete commit intervals, with the
same estimate at every level; workload and sampling differences remain explicit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import tune_phase28_concurrency as old
from src.phase28_reread import runtime as rt

LEVELS = (8, 12, 13)
RUN = old.RUN
STATE = RUN/'WIDE_TUNING_PRIVATE.json'
REPORT = rt.REPORT/'CONCURRENCY_WIDE_AUDIT.json'
GIB = 1024**3
SOURCE = old.SOURCE


def require(value, message):
    if not bool(value): raise ValueError(message)


def admission(resource, active, requested):
    base = old.safety(resource, active)
    require(requested in LEVELS and requested >= active > 0, 'invalid wide projection')
    ram_unit = max(2*GIB, resource['cgroup_memory_current_bytes']/active)
    gpu_unit = max(2*GIB, resource['GPU_used_bytes']/active)
    ram = resource['cgroup_memory_current_bytes']+(requested-active)*ram_unit
    gpu = resource['GPU_used_bytes']+(requested-active)*gpu_unit
    reasons = list(base['reasons'])
    if ram > min(.9*resource['cgroup_memory_limit_bytes'], resource['cgroup_memory_limit_bytes']-8*GIB):
        reasons.append('PROJECTED_CGROUP_MARGIN')
    if gpu > min(.9*resource['GPU_total_bytes'], resource['GPU_total_bytes']-4*GIB):
        reasons.append('PROJECTED_GPU_MARGIN')
    if requested > int(resource['cgroup_CPU_cores_quota']): reasons.append('CPU_QUOTA')
    return {'safe': not reasons, 'reasons': reasons, 'requested': requested,
            'projected_GPU_bytes': gpu, 'projected_cgroup_bytes': ram,
            'reserved_GPU_bytes': 4*GIB, 'reserved_cgroup_bytes': 8*GIB}


def window(snapshot):
    arms, folds = {}, {}
    for p in snapshot['running'].values():
        arm, fold = p.get('arm', 'UNSPECIFIED'), str(p.get('fold', 'UNSPECIFIED'))
        arms[arm] = arms.get(arm, 0)+1; folds[fold] = folds.get(fold, 0)+1
    return {'parallel': snapshot['target'], 'began': snapshot['observed_time'],
            'roster': old.roster(snapshot), 'commits': {key: [] for key in snapshot['running']},
            'GPU_samples': [], 'GPU_peak': 0, 'RAM_peak': 0,
            'workload': {'arms': [{'arm': k, 'processes': v} for k, v in sorted(arms.items())],
                         'partitions': [{'fold': k, 'processes': v} for k, v in sorted(folds.items())]}}


def observe(w, snapshot):
    require(old.roster(snapshot) == w['roster'], 'process roster changed; no mixed throughput window')
    for name, process in snapshot['running'].items():
        values = w['commits'][name]
        epoch = process['epoch']
        if epoch > 0 and (not values or values[-1][0] != epoch):
            require(not values or epoch > values[-1][0], 'epoch counter moved backwards')
            values.append((epoch, snapshot['observed_time']))
    r = snapshot['resource']
    w['GPU_samples'].append(r['GPU_utilization_percent'])
    w['GPU_peak'] = max(w['GPU_peak'], r['GPU_used_bytes'])
    w['RAM_peak'] = max(w['RAM_peak'], r['cgroup_memory_current_bytes'])


def finish(w, *, historical=False):
    """Discard the first partial interval in live windows, no startup-as-speed."""
    usable = {name: (rows if historical else rows[1:]) for name, rows in w['commits'].items()}
    if not usable or any(len(rows) < 2 for rows in usable.values()): return None
    rates, intervals = [], []
    for rows in usable.values():
        first, last = rows[0], rows[-1]
        duration = last[1]-first[1]
        require(duration > 0 and last[0] > first[0], 'no completed epoch interval')
        rates.append((last[0]-first[0])*3600/duration)
        intervals.append(last[0]-first[0])
    samples = w['GPU_samples']
    return {'parallel': w['parallel'], 'throughput_estimated_epochs_per_hour': sum(rates),
            'estimator': 'sum_worker_complete_commit_interval_rates',
            'minimum_complete_intervals_per_process': min(intervals),
            'first_live_observation_discarded': not historical,
            'historical_warm_roster_metadata': historical,
            'sampling_timestamp_resolution_seconds': 20,
            'GPU_utilization_sample_mean': sum(samples)/len(samples) if samples else None,
            'GPU_memory_peak_bytes': w['GPU_peak'], 'cgroup_memory_peak_bytes': w['RAM_peak'],
            'workload_metadata': w['workload'],
            'different_cells_and_non_synchronous_intervals_limit_comparability': True,
            'exact_speedup_or_optimality_claimed': False}


def historical_baseline(snapshot, path):
    """Only the current eight live identities, after every worker's first commit."""
    require(snapshot['target'] == 8 and len(snapshot['running']) == 8, 'wide extension starts from live eight')
    w = window(snapshot)
    latest_birth = max(v['created'] for v in snapshot['running'].values())
    names = set(snapshot['running'])
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        when = old.utc_seconds(row['utc'])
        if (row['parallel'] != 8 or set(row['running']) != names or when <= latest_birth
                or when > snapshot['observed_time']): continue
        if any(row['epochs'].get(name, 0) < 1 for name in names): continue
        pseudo = {**snapshot, 'observed_time': when,
                  'running': {name: {**v, 'epoch': row['epochs'][name]} for name, v in snapshot['running'].items()},
                  'resource': old.resource_summary(row['resource'])}
        observe(w, pseudo)
    for name, rows in w['commits'].items(): w['commits'][name] = rows[-3:]
    answer = finish(w, historical=True)
    require(answer is not None, 'warm eight-worker baseline lacks two complete metadata commits')
    return answer


def update_control(expected, requested):
    require(type(requested) is int and 1 <= requested <= 13, 'wide cap1..13')
    before = old.CONTROL.read_bytes(); control = json.loads(before)
    require(control['max_parallel'] == expected, 'external concurrency change, no overwrite')
    changed = {**control, 'max_parallel': requested}
    target = old.CONTROL.with_name(old.CONTROL.name+'.wide.partial')
    require(not target.exists(), 'orphan wide control write')
    target.write_text(json.dumps(changed, indent=2, sort_keys=True)+'\n')
    require(old.CONTROL.read_bytes() == before, 'concurrent operator control write')
    target.replace(old.CONTROL)


def save(state):
    rt.atomic_json(STATE, state)
    public = {k: state[k] for k in ('schema', 'status', 'source', 'script_sha256', 'stages',
                                  'decisions', 'target', 'reason', 'started_utc')}
    public.update(script_path='scripts/tune_phase28_wide_concurrency.py', max_parallel=13,
                  new_training_cells=0, training_source_changed=False, existing_workers_signaled=0,
                  labels_checkpoints_patient_arrays_or_cox_loss_used=False,
                  allocation_not_the_optimization_target=True)
    rt.atomic_json(REPORT, public)


def command(action):
    import os
    import subprocess
    import fcntl
    if action == 'launch':
        require(not STATE.exists(), 'wide tuning already has a receipt; no duplicate launch')
        state = old.read_snapshot()
        require(state['status'] == 'RUNNING' and state['target'] == 8 and not state['failed_count'],
                'healthy eight-worker matrix required')
        with (RUN/'WIDE_TUNER_PRIVATE.log').open('a') as log:
            p = subprocess.Popen([str(rt.ENV_PYTHON), str(Path(__file__)), 'run', '--execute-tuning'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        print(json.dumps({'status': 'WIDE_TUNER_LAUNCHED', 'pid': p.pid})); return
    lock = (RUN/'WIDE_TUNER.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not STATE.exists() and rt.source() == SOURCE, 'wide tuner duplicate or source drift')
    first = old.read_snapshot()
    baseline = historical_baseline(first, RUN/'RESOURCE_SAMPLES_PRIVATE.jsonl')
    state = {'schema': 'phase28_user_authorized_wide_concurrency_v1', 'status': 'OBSERVING',
             'source': SOURCE, 'script_sha256': rt.sha(__file__), 'stages': [baseline],
             'decisions': [], 'target': 8, 'reason': None, 'owner_pid_private': os.getpid(),
             'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    check = admission(first['resource'], 8, 12)
    if not check['safe']:
        state.update(status='STOPPED_PROJECTED_MARGIN', reason=check); save(state); return
    update_control(8, 12)
    state['decisions'].append({'before': 8, 'after': 12, 'reason': 'user_authorized_memory_qualified_extension', 'admission': check})
    state['target'] = 12; save(state)
    w = None; pending_at = time.time()
    while True:
        time.sleep(20)
        require(rt.source() == SOURCE and rt.sha(__file__) == state['script_sha256'], 'training/tuner source drift')
        s = old.read_snapshot()
        if s['failed_count'] or s['status'] != 'RUNNING':
            state.update(status='STOPPED_MATRIX_NOT_RUNNING', reason=s['status']); save(state); return
        if s['target'] != state['target']:
            require(time.time()-pending_at <= 90, 'external control override; no automatic overwrite')
            continue
        safe = old.safety(s['resource'], max(1, len(s['running'])))
        if not safe['safe']:
            update_control(state['target'], 8)
            state['decisions'].append({'before': state['target'], 'after': 8, 'reason': safe})
            state.update(target=8, status='STOPPED_RESOURCE_MARGIN', reason=safe); save(state); return
        if len(s['running']) != state['target']: w = None; continue
        if w is None or old.roster(s) != w['roster']: w = window(s)
        observe(w, s); result = finish(w)
        if result is None: save(state); continue
        previous = state['stages'][-1]
        result['ratio_to_previous_estimate'] = result['throughput_estimated_epochs_per_hour']/previous['throughput_estimated_epochs_per_hour']
        state['stages'].append(result)
        if result['ratio_to_previous_estimate'] < 1.0:
            update_control(state['target'], previous['parallel'])
            state['decisions'].append({'before': state['target'], 'after': previous['parallel'], 'reason': 'observed_throughput_decline'})
            state.update(target=previous['parallel'], status='STOPPED_THROUGHPUT_DECLINE', reason='queue_cap_only_existing_workers_continue')
            save(state); return
        future = [x for x in LEVELS if x > state['target']]
        if not future:
            state.update(status='COMPLETE_MAXIMUM_QUALIFIED', reason='cap13_reached'); save(state); return
        requested = future[0]; check = admission(s['resource'], len(s['running']), requested)
        if not check['safe']:
            state.update(status='STOPPED_PROJECTED_MARGIN', reason=check); save(state); return
        update_control(state['target'], requested)
        state['decisions'].append({'before': state['target'], 'after': requested, 'reason': 'throughput_not_decreased', 'admission': check})
        state.update(target=requested); w = None; pending_at = time.time(); save(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'run'))
    parser.add_argument('--execute-tuning', action='store_true', required=True)
    args = parser.parse_args()
    try: command(args.action)
    except Exception as exc:
        if STATE.exists():
            state = old.read_json(STATE)
            if state['status'] == 'OBSERVING' and state.get('owner_pid_private') == __import__('os').getpid():
                state.update(status='STOPPED_TUNER_EXCEPTION', reason=type(exc).__name__); save(state)
        raise

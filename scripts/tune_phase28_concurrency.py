"""Outcome-blind concurrency calibration for the running, locked matrix.

This tool never launches, stops, resumes or changes a training process. Its only
training control write is an atomic change to CONTROL_PRIVATE.max_parallel in
the coordinator's existing [1, 8] range. It reads epoch/wall metadata, not Cox
losses, checkpoints, patient features or endpoints. CPU and memory limits come
from the coordinator's recent cgroup/GPU resource samples, never host RAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase28_reread import runtime as rt

SOURCE = '0ccd4897d30e654ebcb3f55d4891a3b76312f5e6a28dcc30070520000fc5372b'
RUN = rt.PRIVATE/'matrix_execution'
CONTROL = RUN/'CONTROL_PRIVATE.json'
STATE = RUN/'STATE_PRIVATE.json'
TUNING = RUN/'CONCURRENCY_TUNING_PRIVATE.json'
EVENTS = RUN/'CONCURRENCY_EVENTS_PRIVATE.jsonl'
REPORT = rt.REPORT/'CONCURRENCY_AUDIT.json'
LEVELS = (1, 2, 4, 8)
GIB = 1024**3
VERSION = 'phase28_outcome_blind_concurrency_v1'


def require(value, message):
    if not bool(value): raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def utc_seconds(text):
    return float(__import__('calendar').timegm(time.strptime(text, '%Y-%m-%dT%H:%M:%SZ')))


def resource_summary(resource):
    """Whitelisted cgroup and GPU quantities; host memory is never consulted."""
    cg = resource.get('cgroup_v2', {})
    maximum, current, cpu = cg.get('memory.max'), cg.get('memory.current'), cg.get('cpu.max')
    require(str(maximum).isdigit() and str(current).isdigit(), 'finite cgroup memory.max/current required')
    limit, used = int(maximum), int(current)
    require(limit > 0 and 0 <= used <= limit, 'invalid cgroup memory accounting')
    parts = str(cpu).split()
    require(len(parts) == 2 and all(p.isdigit() for p in parts) and int(parts[1]) > 0,
            'finite cgroup CPU quota/period required')
    quota = int(parts[0])/int(parts[1])
    gpu = resource.get('GPU_csv_MiB_MiB_percent_percent', '')
    require(isinstance(gpu, str) and '\n' not in gpu, 'exactly one GPU resource row required')
    fields = [float(v.strip()) for v in gpu.split(',')]
    require(len(fields) == 4 and fields[0] >= 0 and fields[1] > 0 and fields[0] <= fields[1]
            and all(0 <= v <= 100 for v in fields[2:]), 'invalid GPU resource counters')
    return {'cgroup_memory_limit_bytes': limit, 'cgroup_memory_current_bytes': used,
        'cgroup_memory_available_bytes': limit-used, 'cgroup_CPU_cores_quota': quota,
        'GPU_used_bytes': int(fields[0]*1024**2), 'GPU_total_bytes': int(fields[1]*1024**2),
        'GPU_available_bytes': int((fields[1]-fields[0])*1024**2),
        'GPU_utilization_percent': fields[2], 'GPU_memory_utilization_percent': fields[3],
        'disk_available_bytes': int(resource['disk_available_bytes'])}


def safety(resource, active, requested=None):
    """Conservative admission; it is a memory margin, not a fitted cost model."""
    r = resource
    problems = []
    if r['cgroup_memory_available_bytes'] < 8*GIB or r['cgroup_memory_current_bytes'] > .90*r['cgroup_memory_limit_bytes']:
        problems.append('CGROUP_MEMORY_CURRENT_MARGIN')
    if r['GPU_available_bytes'] < 4*GIB or r['GPU_used_bytes'] > .90*r['GPU_total_bytes']:
        problems.append('GPU_MEMORY_CURRENT_MARGIN')
    if r['disk_available_bytes'] < GIB:
        problems.append('DISK_CURRENT_MARGIN')
    projected = {}
    if requested is not None:
        require(requested in LEVELS and 1 <= active <= 8 and requested >= active,
                'projection requires an allowed nondecreasing concurrency level')
        increment = requested-active
        # No host-size inference or GPU-size division is used. Initial reserve
        # assumptions are intentionally explicit and can only deny admission.
        ram_estimate = max(2*GIB, r['cgroup_memory_current_bytes']/active)
        gpu_estimate = max(2*GIB, r['GPU_used_bytes']/active)
        ram = r['cgroup_memory_current_bytes']+increment*ram_estimate
        gpu = r['GPU_used_bytes']+increment*gpu_estimate
        projected = {'parallel': requested, 'incremental_processes': increment,
            'conservative_incremental_RAM_bytes_per_process': ram_estimate,
            'conservative_incremental_GPU_bytes_per_process': gpu_estimate,
            'projected_cgroup_memory_bytes': ram, 'projected_GPU_memory_bytes': gpu}
        if ram > min(.90*r['cgroup_memory_limit_bytes'], r['cgroup_memory_limit_bytes']-8*GIB):
            problems.append('CGROUP_PROJECTED_MARGIN')
        if gpu > min(.90*r['GPU_total_bytes'], r['GPU_total_bytes']-4*GIB):
            problems.append('GPU_PROJECTED_MARGIN')
        if requested > int(r['cgroup_CPU_cores_quota']):
            problems.append('CPU_QUOTA_ADMISSION')
    return {'safe': not problems, 'reasons': problems, 'projection': projected}


def sanitized_snapshot(state, progress_by_name, *, observed_time=None):
    """Explicit whitelist: no fit_cox or other outcome-dependent field access."""
    now = time.time() if observed_time is None else float(observed_time)
    require(state.get('source') == SOURCE, 'coordinator source differs')
    stamp = utc_seconds(state['utc'])
    resource_stamp = utc_seconds(state['resource']['utc'])
    require(-5 <= now-stamp <= 90 and -5 <= now-resource_stamp <= 90, 'stale coordinator/resource sample')
    running = {}
    for name, row in state['running'].items():
        p = progress_by_name.get(name) or {}
        epoch = int(p.get('epoch', 0)); wall = p.get('wall_seconds')
        require(0 <= epoch <= 80 and (wall is None or float(wall) >= 0), 'invalid committed epoch metadata')
        if p:
            require(p.get('source') == SOURCE, 'progress source differs')
        born = float(row['create_time'])
        require(0 < born <= now+5 and type(row['pid']) is int, 'unbound process identity')
        cell = row.get('cell', (None, None, 'UNSPECIFIED'))
        require(len(cell) == 3, 'coordinator cell metadata requires seed/fold/arm')
        running[name] = {'pid': row['pid'], 'created': born, 'epoch': epoch,
                         'arm': str(cell[2]), 'fold': cell[1],
                         'attempt_wall_seconds': None if wall is None else float(wall)}
    return {'observed_time': now, 'coordinator_time': stamp, 'status': state['status'],
        'source': SOURCE, 'target': int(state['max_parallel']), 'complete': int(state['complete_count']),
        'failed_count': len(state.get('failed', {})), 'running': running,
        'resource': resource_summary(state['resource'])}


def read_snapshot():
    state = read_json(STATE)
    progress = {}
    for name, row in state['running'].items():
        cell = tuple(row['cell'])
        path = rt.cell_dir(*cell)/'PROGRESS_PRIVATE.json'
        if path.exists():
            # Deserialization is unavoidable because the existing JSON also
            # holds fit_cox. Only the following metadata keys are retained or
            # used; the outcome-dependent value is never dereferenced.
            raw = read_json(path)
            progress[name] = {k: raw[k] for k in ('source', 'epoch', 'wall_seconds') if k in raw}
    return sanitized_snapshot(state, progress)


def roster(snapshot):
    return {name: (row['pid'], row['created']) for name, row in snapshot['running'].items()}


def new_window(snapshot, target):
    require(len(snapshot['running']) == target, 'full target roster required before timing')
    arm_counts, fold_counts = {}, {}
    for row in snapshot['running'].values():
        arm = row.get('arm', 'UNSPECIFIED'); fold = str(row.get('fold', 'UNSPECIFIED'))
        arm_counts[arm] = arm_counts.get(arm, 0)+1
        fold_counts[fold] = fold_counts.get(fold, 0)+1
    return {'target': target, 'began': snapshot['observed_time'],
        'latest_process_start': max(row['created'] for row in snapshot['running'].values()),
        'roster': {name: [row['pid'], row['created']] for name, row in snapshot['running'].items()},
        'epochs_at_start': {name: row['epoch'] for name, row in snapshot['running'].items()},
        'arm_counts': arm_counts, 'fold_counts': fold_counts,
        'samples': 0, 'GPU_utilization_sum': 0., 'GPU_utilization_maximum': 0.,
        'GPU_memory_peak_bytes': 0, 'cgroup_memory_peak_bytes': 0}


def add_sample(window, snapshot):
    window['samples'] += 1
    r = snapshot['resource']
    window['GPU_utilization_sum'] += r['GPU_utilization_percent']
    window['GPU_utilization_maximum'] = max(window['GPU_utilization_maximum'], r['GPU_utilization_percent'])
    window['GPU_memory_peak_bytes'] = max(window['GPU_memory_peak_bytes'], r['GPU_used_bytes'])
    window['cgroup_memory_peak_bytes'] = max(window['cgroup_memory_peak_bytes'], r['cgroup_memory_current_bytes'])


def finish_window(window, snapshot):
    """Two commits each ensure at least one wholly observed epoch per process."""
    if {k: tuple(v) for k, v in window['roster'].items()} != roster(snapshot): return None
    deltas = {name: snapshot['running'][name]['epoch']-epoch
              for name, epoch in window['epochs_at_start'].items()}
    require(all(value >= 0 for value in deltas.values()), 'epoch counter decreased without a new process')
    if min(deltas.values()) < 2: return None
    seconds = snapshot['observed_time']-window['began']
    require(seconds > 0 and window['began'] >= window['latest_process_start'], 'invalid observation/startup window')
    committed = sum(deltas.values())
    return {'parallel': window['target'], 'window_seconds': seconds,
        'committed_epoch_count': committed, 'committed_epochs_per_hour': committed/seconds*3600,
        'minimum_committed_epochs_per_process': min(deltas.values()),
        'maximum_committed_epochs_per_process': max(deltas.values()),
        'guaranteed_wholly_observed_epochs_per_process_at_least': min(deltas.values())-1,
        'leading_partial_epoch_not_treated_as_a_fully_observed_epoch': True,
        'throughput_counts_actual_commits_in_common_wall_window': True,
        'GPU_utilization_mean_of_periodic_samples': window['GPU_utilization_sum']/window['samples'],
        'GPU_utilization_maximum_of_periodic_samples': window['GPU_utilization_maximum'],
        'GPU_memory_peak_bytes': window['GPU_memory_peak_bytes'],
        'cgroup_memory_peak_bytes': window['cgroup_memory_peak_bytes'],
        'resource_samples': window['samples'],
        'workload_metadata': {'arms': [{'arm': k, 'processes': v} for k, v in sorted(window['arm_counts'].items())],
            'partitions': [{'fold': k, 'processes': v} for k, v in sorted(window['fold_counts'].items())],
            'raw_Patch_or_patient_inputs_not_opened_for_FLOP_estimation': True},
        'different_cell_sizes_or_partial_boundary_can_confound_throughput': True,
        'equal_FLOPs_claimed': False,
        'all_workers_had_committed_epoch_at_window_start': all(
            value >= 1 for value in window['epochs_at_start'].values()),
        'cold_start_excluded_by_stable_roster_window': False,
        'initial_Data_plan_or_first_forward_cost_may_remain_in_wall_window': True}


def historical_level_one(path, snapshot):
    """Admit only a complete old single-worker window bound to a live PID.

    A single cold epoch is not silently converted to a warm throughput
    benchmark. Insufficient history is reported and the current level is timed.
    """
    if not Path(path).exists(): return None
    window, latest, answer = None, None, None
    for line in Path(path).read_text().splitlines():
        raw = json.loads(line)
        names = raw.get('running', [])
        if raw.get('parallel') != 1 or len(names) != 1 or names[0] not in snapshot['running']:
            window = None; continue
        name = names[0]; when = utc_seconds(raw['utc'])
        live = snapshot['running'][name]
        if when < live['created'] or when > snapshot['observed_time']:
            window = None; continue
        sample = {'observed_time': when, 'running': {name: {**live, 'epoch': int(raw['epochs'][name])}},
                  'resource': resource_summary(raw['resource'])}
        if window is None:
            window = new_window(sample, 1)
        add_sample(window, sample); latest = sample
        candidate = finish_window(window, sample)
        if candidate is not None:
            candidate['retrospective_resource_metadata_window'] = True
            answer = candidate
    return answer


def initialize(snapshot, cap, history=None):
    target = snapshot['target']
    require(target in LEVELS and target <= cap, 'current target is not a locked tuning level within the cap')
    stages = [history] if history is not None and target > 1 else []
    return {'version': VERSION, 'source': SOURCE, 'status': 'OBSERVING', 'initial_parallel': target,
        'target': target, 'cap': cap, 'window': None, 'stages': stages,
        'baseline_1_status': 'OBSERVED_FULL_METADATA_WINDOW' if stages or target == 1 else 'INSUFFICIENT_HISTORY_NOT_INVENTED',
        'stop_reason': None, 'last_control': target, 'decisions': []}


def step(tuning, snapshot):
    """Pure state machine: returns an optional new concurrency, never a PID action."""
    require(tuning['status'] == 'OBSERVING', 'tuner is no longer observing')
    if snapshot['failed_count'] or snapshot['status'] != 'RUNNING':
        tuning.update(status='STOPPED_MATRIX_NOT_RUNNING', stop_reason=snapshot['status'])
        return None
    safe = safety(snapshot['resource'], max(1, len(snapshot['running'])))
    if not safe['safe']:
        tuning.update(status='STOPPED_RESOURCE_MARGIN', stop_reason=safe['reasons'], window=None)
        return 1  # Existing workers continue. This only prevents replenishment above one.
    if snapshot['target'] != tuning['target']:
        pending = tuning.get('pending_control')
        if (pending and snapshot['target'] == pending['before']
                and snapshot['observed_time']-pending['issued_time'] <= 90):
            return None  # Coordinator snapshots may lag our atomic control write.
        raise ValueError('external concurrency change; tuner must not overwrite it')
    tuning.pop('pending_control', None)
    if len(snapshot['running']) != tuning['target']:
        tuning['window'] = None
        return None
    window = tuning.get('window')
    if window is None or {k: tuple(v) for k, v in window['roster'].items()} != roster(snapshot):
        window = tuning['window'] = new_window(snapshot, tuning['target'])
    add_sample(window, snapshot)
    result = finish_window(window, snapshot)
    if result is None: return None
    previous = tuning['stages'][-1] if tuning['stages'] else None
    result['ratio_to_previous_observed_level'] = (result['committed_epochs_per_hour']/previous['committed_epochs_per_hour']) if previous else None
    tuning['stages'].append(result); tuning['window'] = None
    if previous and result['committed_epochs_per_hour'] < previous['committed_epochs_per_hour']:
        tuning.update(status='STOPPED_THROUGHPUT_DECLINE', stop_reason='observed_commit_throughput_below_previous_level')
        return previous['parallel']
    next_levels = [v for v in LEVELS if tuning['target'] < v <= tuning['cap']]
    if not next_levels:
        tuning.update(status='COMPLETE_MAXIMUM_TESTED', stop_reason='predeclared_cap_reached')
        return None
    requested = next_levels[0]
    admission = safety(snapshot['resource'], len(snapshot['running']), requested)
    result['next_level_admission'] = admission
    if not admission['safe']:
        tuning.update(status='STOPPED_PROJECTED_RESOURCE_MARGIN', stop_reason=admission['reasons'])
        return None
    tuning['target'] = requested
    return requested


def update_control(parallel):
    require(type(parallel) is int and 1 <= parallel <= 8, 'coordinator supports only max_parallel 1..8')
    before = CONTROL.read_bytes(); control = json.loads(before)
    updated = {**control, 'max_parallel': parallel}
    temporary = CONTROL.with_name(CONTROL.name+'.concurrency.partial')
    require(not temporary.exists(), 'orphan control update; explicit review required')
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True)+'\n')
    if CONTROL.read_bytes() != before:
        temporary.unlink()
        raise ValueError('CONTROL changed concurrently; not overwriting another operator')
    temporary.replace(CONTROL)
    return {'before': control['max_parallel'], 'after': parallel,
            'only_changed_key': 'max_parallel', 'existing_processes_signaled': 0}


def public_report(tuning, script_hash):
    return {'version': VERSION, 'status': tuning['status'], 'source': SOURCE,
        'script_path': 'scripts/tune_phase28_concurrency.py', 'script_sha256': script_hash,
        'initial_parallel': tuning['initial_parallel'], 'maximum_allowed_parallel': tuning['cap'],
        'current_requested_parallel': tuning.get('last_control', tuning['target']),
        'baseline_1_status': tuning['baseline_1_status'], 'stages': tuning['stages'],
        'stop_reason': tuning['stop_reason'], 'control_decisions': tuning['decisions'],
        'training_processes_started_stopped_or_restarted_by_tuner': 0,
        'training_protocol_or_source_changed': False,
        'outcome_loss_checkpoint_patient_input_accesses': 0,
        'host_RAM_used_for_admission': False,
        'lowering_cap_does_not_stop_existing_workers': True,
        'minimum_cap_one_is_not_a_permanent_pause_when_running_workers_finish': True,
        'raw_progress_JSON_contains_fit_cox_but_only_epoch_and_wall_keys_consumed': True,
        'two_commits_required_per_unchanged_process_before_a_stage_comparison': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--max-parallel', choices=(1, 2, 4, 8), type=int, default=8)
    parser.add_argument('--execute-tuning', action='store_true')
    args = parser.parse_args()
    require(args.execute_tuning and args.source == SOURCE, 'explicit current-source tuning authorization required')
    require(rt.source() == SOURCE, 'locked training source changed')
    # A separate lock prevents duplicate tuners; no coordinator or worker lock
    # is acquired, and no process-control API exists in this tool.
    import fcntl
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN/'CONCURRENCY_TUNER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    script_hash = rt.sha(__file__)
    snapshot = read_snapshot()
    if TUNING.exists():
        tuning = read_json(TUNING)
        require(tuning['version'] == VERSION and tuning['source'] == SOURCE
                and tuning['script_sha256'] == script_hash and tuning['cap'] == args.max_parallel,
                'tuner restart source/script/cap changed')
        require(tuning['status'] == 'OBSERVING', 'completed/stopped tuner requires review, not silent restart')
    else:
        historical = historical_level_one(RUN/'RESOURCE_SAMPLES_PRIVATE.jsonl', snapshot)
        tuning = initialize(snapshot, args.max_parallel, historical)
        tuning['script_sha256'] = script_hash
    while True:
        before_target = tuning['target']
        requested = step(tuning, snapshot)
        if requested is not None:
            require(rt.source() == SOURCE and rt.sha(__file__) == script_hash, 'source/script drift before control update')
            change = update_control(int(requested))
            tuning['last_control'] = int(requested)
            tuning['pending_control'] = {'before': change['before'], 'after': change['after'],
                                         'issued_time': snapshot['observed_time']}
            tuning['decisions'].append({'observed_stage_parallel': before_target,
                'before': change['before'], 'after': change['after'], 'reason': tuning['stop_reason'] or 'full_observation_and_resource_admission_passed'})
            with EVENTS.open('a') as stream:
                stream.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'stage': 'CONTROL_MAX_PARALLEL_CHANGED', **change,
                    'sanitized_resource': snapshot['resource']}, sort_keys=True)+'\n')
            print(json.dumps({'status': tuning['status'], 'parallel': requested,
                              'observed_levels': len(tuning['stages'])}), flush=True)
        rt.atomic_json(TUNING, tuning)
        rt.atomic_json(REPORT, public_report(tuning, script_hash))
        if tuning['status'] != 'OBSERVING':
            print(json.dumps({'status': tuning['status'], 'reason': tuning['stop_reason']}), flush=True)
            return
        # No blocking wait beyond one coordinator sampling interval.
        time.sleep(20)
        snapshot = read_snapshot()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Fail closed without touching CONTROL or any worker. Preserve an
        # interrupted metadata window for review instead of claiming a result.
        if TUNING.exists():
            state = read_json(TUNING)
            if state.get('status') == 'OBSERVING':
                state.update(status='STOPPED_TUNER_EXCEPTION', stop_reason=type(error).__name__)
                rt.atomic_json(TUNING, state)
                rt.atomic_json(REPORT, public_report(state, rt.sha(__file__)))
                with EVENTS.open('a') as stream:
                    stream.write(json.dumps({'stage': 'TUNER_FAILED_NO_WORKER_ACTION',
                        'error_type': type(error).__name__, 'message_private': str(error)})+'\n')
        raise

"""Persistent source-locked 30-cell coordinator; changes concurrency only."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase28_reread import runtime as rt

SOURCE = '0ccd4897d30e654ebcb3f55d4891a3b76312f5e6a28dcc30070520000fc5372b'
RUN = rt.PRIVATE/'matrix_execution'
CONTROL = RUN/'CONTROL_PRIVATE.json'
STATE = RUN/'STATE_PRIVATE.json'
CELLS = [(s, f, a) for f in (1, 2, 3) for s in (31, 47) for a in rt.ARMS]
MAX_PARALLEL = 13  # User-authorized execution-only extension; no new model cells.


def key(cell): return f'seed{cell[0]}_fold{cell[1]}_{cell[2]}'
def read(path): return json.loads(Path(path).read_text()) if Path(path).exists() else None
def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def resource():
    v = psutil.virtual_memory(); d = psutil.disk_usage(rt.PRIVATE)
    io = psutil.disk_io_counters()
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory',
                              '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
        gpu_text = gpu.stdout.strip() if gpu.returncode == 0 else 'QUERY_ERROR'
    except (subprocess.TimeoutExpired, OSError): gpu_text = 'QUERY_TIMEOUT_OR_OS_ERROR'
    cgroup = {}
    for name in ('memory.max', 'memory.current', 'cpu.max'):
        path = Path('/sys/fs/cgroup')/name
        cgroup[name] = path.read_text().strip() if path.exists() else None
    return {'utc': now(), 'RAM_available_bytes': v.available, 'RAM_total_bytes': v.total,
            'CPU_count': psutil.cpu_count(), 'CPU_percent': psutil.cpu_percent(),
            'disk_available_bytes': d.free, 'physical_disk_read_bytes': io.read_bytes if io else None,
            'GPU_csv_MiB_MiB_percent_percent': gpu_text, 'cgroup_v2': cgroup}


def active_cells():
    result = {}
    for p in psutil.process_iter(['pid', 'cmdline', 'create_time', 'status']):
        args = p.info['cmdline'] or []
        if str(ROOT/'scripts/run_phase28.py') not in args or p.info['status'] == psutil.STATUS_ZOMBIE:
            continue
        try:
            cell = (int(args[args.index('--seed')+1]), int(args[args.index('--fold')+1]), args[args.index('--arm')+1])
            identity = args[args.index('--source')+1]
        except (ValueError, IndexError):
            raise RuntimeError('unrecognized active formal process; no new task launched')
        if cell not in CELLS or identity != SOURCE or key(cell) in result:
            raise RuntimeError('duplicate/out-of-lock active formal task; no new task launched')
        result[key(cell)] = {'pid': p.info['pid'], 'create_time': p.info['create_time'], 'cell': cell}
    return result


def worker(cell):
    """Cell lock survives exec and scheduler failure; global lock is not inherited."""
    RUN.mkdir(parents=True, exist_ok=True)
    fd = os.open(RUN/(key(cell)+'.lock'), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(fd, True)
    command = [str(rt.ENV_PYTHON), str(ROOT/'scripts/run_phase28.py'), '--seed', str(cell[0]),
               '--fold', str(cell[1]), '--arm', cell[2], '--source', SOURCE, '--execute-formal']
    path = rt.cell_dir(*cell)
    if (path/'START_PRIVATE.json').exists() and not (path/'DONE_PRIVATE.json').exists():
        if not (path/'LATEST_PRIVATE.pt').exists():
            raise RuntimeError('incomplete attempt before epoch10 has no lawful automatic resume')
        command.append('--resume')
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    os.execve(str(rt.ENV_PYTHON), command, env)


def event(payload):
    with (RUN/'EVENTS_PRIVATE.jsonl').open('a') as stream:
        stream.write(json.dumps({'utc': now(), **payload}, sort_keys=True)+'\n')


def archive_p0_decision():
    target, archive = rt.REPORT/'decision.json', rt.REPORT/'P0_DECISION.json'
    if archive.exists():
        expected = read(RUN/'P0_DECISION_ARCHIVE_PRIVATE.json')
        if not expected or rt.sha(archive) != expected['sha256']:
            raise RuntimeError('P0 decision archive changed')
        return
    old = read(target)
    if not old or old.get('scientific_decision') != 'NOT_YET_TESTED' or old.get('new_patient_training_fits') != 0:
        raise RuntimeError('existing decision is not the known P0 qualification decision')
    content = target.read_bytes(); digest = rt.sha(target)
    archive.write_bytes(content)
    if archive.read_bytes() != content: raise RuntimeError('P0 decision archive mismatch')
    rt.atomic_json(RUN/'P0_DECISION_ARCHIVE_PRIVATE.json', {'status': 'P0_BYTES_PRESERVED', 'sha256': digest,
                    'purpose': 'formal once-only evaluator requires a fresh decision path', 'utc': now()})
    target.unlink()


def postprocess():
    def command(stage, arguments):
        state = read(STATE) or {}
        state.update(status='POSTPROCESS_RUNNING', postprocess_stage=stage, utc=now())
        rt.atomic_json(STATE, state)
        event({'stage': stage+'_START'})
        with (RUN/(stage+'_PRIVATE.log')).open('a') as log:
            child = subprocess.Popen([str(rt.ENV_PYTHON), *arguments], cwd=ROOT, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                        env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1',
                             'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'})
            while child.poll() is None:
                state.update(postprocess_pid=child.pid, utc=now(), resource=resource())
                rt.atomic_json(STATE, state)
                time.sleep(5)
        if child.returncode:
            raise RuntimeError(stage+' failed; private log retained, no silent retry')
        event({'stage': stage+'_COMPLETE'})

    if (rt.REPORT/'REPLAY_AUDIT.json').exists():
        if read(rt.REPORT/'REPLAY_AUDIT.json').get('status') != 'PASS' or read(rt.REPORT/'REPLAY_AUDIT.json').get('source') != SOURCE:
            raise RuntimeError('existing replay is not PASS; no automatic retry')
    else:
        command('INDEPENDENT_REPLAY', [str(ROOT/'scripts/replay_phase28.py'), '--stage', 'prototype',
                                      '--source', SOURCE, '--execute-replay'])
    from src.phase28_reread.evaluation import verify_stage, verify_qualifications, verify_replay
    qualified = verify_qualifications(rt.REPORT, SOURCE, rt.source_files(), project_root=ROOT, private_root=rt.PRIVATE)
    _, artifact = verify_stage(rt.PRIVATE, 'prototype', SOURCE)
    verify_replay(rt.REPORT/'REPLAY_AUDIT.json', SOURCE, artifact, qualified, rt.source_files(), project_root=ROOT)
    if (rt.REPORT/'INNER_RESULTS.json').exists():
        result = read(rt.REPORT/'INNER_RESULTS.json')
        if (result.get('status') != 'COMPLETE_BIDIRECTIONAL_REREAD_INNER_PROTOTYPE'
                or result.get('source') != SOURCE or result.get('valid_endpoint_callback_calls') != 1
                or not (rt.REPORT/'INNER_GATE.json').exists() or not (rt.REPORT/'decision.json').exists()):
            raise RuntimeError('existing formal evaluation is incomplete; no endpoint reread')
    else:
        archive_p0_decision()
        command('ONCE_ONLY_FORMAL_EVALUATION', [str(ROOT/'scripts/evaluate_phase28.py'), '--stage', 'prototype',
                                              '--source', SOURCE, '--execute-evaluation'])
    budget = rt.REPORT/'BUDGET_READOUT_AUDIT.json'
    if not budget.exists():
        command('BUDGET_READOUT_AUDIT', [str(ROOT/'scripts/audit_phase28_budget_readout.py'),
                                       '--source', SOURCE, '--execute-audit'])
    else:
        saved = read(budget)
        if saved.get('source') != SOURCE or saved.get('script_sha256') != rt.sha(ROOT/'scripts/audit_phase28_budget_readout.py'):
            raise RuntimeError('existing budget audit source/executable changed')
    for cell in CELLS:
        done = rt.verify_done(rt.cell_dir(*cell), SOURCE, *cell)
        for mode in ('history', 'mechanism'):
            output = rt.REPORT/'diagnostics'/f'seed{cell[0]}_fold{cell[1]}_{cell[2]}_{mode}.json'
            if output.exists():
                saved = read(output)
                if (saved.get('source') != SOURCE or saved.get('checkpoint_sha256') != done['checkpoint_sha256']
                        or saved.get('script_sha256') != rt.sha(ROOT/'scripts/diagnose_phase28.py')
                        or saved.get('status') != 'PASS_POSTFIT_'+mode.upper()+'_DIAGNOSTICS'):
                    raise RuntimeError('existing diagnostic no longer matches its source/checkpoint')
                continue
            command('DIAGNOSTIC_'+key(cell)+'_'+mode, [str(ROOT/'scripts/diagnose_phase28.py'), '--mode', mode,
                    '--seed', str(cell[0]), '--fold', str(cell[1]), '--arm', cell[2], '--source', SOURCE,
                    '--execute-diagnostics'])
    command('MECHANISM_SUMMARY', [str(ROOT/'scripts/report_phase28_mechanism.py'),
                                '--source', SOURCE, '--execute-report'])
    command('FORMAL_REPORT', [str(ROOT/'scripts/report_phase28_formal.py'), '--source', SOURCE, '--execute-report'])
    extra = ['tests/test_phase28_budget_readout.py', 'tests/test_phase28_formal_report.py',
             'scripts/test_phase28_diagnostics.py', 'scripts/tune_phase28_concurrency.py',
             'scripts/test_phase28_concurrency.py', 'scripts/qualify_phase28_execution.py',
             'reports/phase28_bidir_reread/POSTFIT_DIAGNOSTIC_PLAN.md',
             'reports/phase28_bidir_reread/POSTFIT_DIAGNOSTIC_QUALIFICATION.json',
             'reports/phase28_bidir_reread/POSTFIT_EVALUATION_REVIEW.md',
             'reports/phase28_bidir_reread/SCHEDULER_READONLY_REVIEW.json',
             'reports/phase28_bidir_reread/FORMAL_EXECUTION_AUTHORIZATION.json',
             'reports/phase28_bidir_reread/CONCURRENCY_AUDIT.json',
             'reports/phase28_bidir_reread/CONCURRENCY_PLAN.md',
             'reports/phase28_bidir_reread/CONCURRENCY_QUALIFICATION.json',
             'reports/phase28_bidir_reread/FORMAL_TOOLCHAIN_QUALIFICATION.json',
             'reports/phase28_bidir_reread/FORMAL_TOOLCHAIN_QUALIFICATION_FINAL.json']
    flags = [item for path in extra for item in ('--include-new', path)]
    git_head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    action = '--publish' if git_head == 'c12494606af9e95b704925c03760b5b2c767db1b' else '--push-committed'
    command('FORMAL_PUBLICATION', [str(ROOT/'scripts/publish_phase28_formal.py'), action, *flags])
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    rt.atomic_json(RUN/'PUSH_VERIFIED_PRIVATE.json', {'status': 'FORMAL_PUSH_VERIFIED', 'commit': commit,
          'source': SOURCE, 'branch': 'research/phase28-bidirectional-evidence-reread', 'utc': now()})
    event({'stage': 'TRAIN_REPLAY_EVALUATION_DIAGNOSTICS_PUBLICATION_COMPLETE', 'commit': commit})


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    global_lock = (RUN/'COORDINATOR.lock').open('a')
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if rt.source() != SOURCE: raise RuntimeError('locked training source changed')
    rt.verify_qualifications(SOURCE)
    inherited = active_cells()
    for p in psutil.process_iter(['cmdline']):
        args = p.info['cmdline'] or []
        if any(str(ROOT/'scripts'/s) in args for s in ('replay_phase28.py', 'evaluate_phase28.py',
                'audit_phase28_budget_readout.py', 'diagnose_phase28.py', 'report_phase28_formal.py',
                'report_phase28_mechanism.py',
                'publish_phase28_formal.py')):
            raise RuntimeError('postprocessing process already active; do not duplicate it')
    running, failed, done_cache = {}, {}, {}
    prior = read(STATE) or {}
    failed.update(prior.get('failed', {}))
    for name, row in inherited.items(): running[name] = {'cell': tuple(row['cell']), 'pid': row['pid'], 'process': None,
                                                       'create_time': row['create_time'], 'adopted': True}
    event({'stage': 'COORDINATOR_STARTED', 'pid': os.getpid(), 'adopted_cells': list(inherited),
           'source': SOURCE, 'scheduler_sha256': rt.sha(__file__)})
    while True:
        started = time.monotonic()
        control = read(CONTROL) or {'max_parallel': 1}
        parallel = int(control['max_parallel'])
        if not 1 <= parallel <= MAX_PARALLEL: raise RuntimeError('concurrency outside reviewed explicit range1..13')
        rows, complete = {}, []
        for cell in CELLS:
            name, path = key(cell), rt.cell_dir(*cell)
            receipt, progress = read(path/'DONE_PRIVATE.json'), read(path/'PROGRESS_PRIVATE.json')
            if receipt:
                signature = (rt.sha(path/'DONE_PRIVATE.json'), receipt.get('checkpoint_sha256'))
                if done_cache.get(name) != signature:
                    rt.verify_done(path, SOURCE, *cell); done_cache[name] = signature
                complete.append(name)
            rows[name] = {'complete': receipt is not None, 'epoch': 80 if receipt else (progress or {}).get('epoch', 0),
                          'seconds_last_epoch_unknown': True, 'fit_cox': (progress or {}).get('fit_cox'),
                          'attempt_progress_wall_seconds': (progress or {}).get('wall_seconds')}
        for name, item in list(running.items()):
            process = item['process']
            if process is not None:
                status = process.poll(); alive = status is None
            else:
                try:
                    p = psutil.Process(item['pid'])
                    alive = p.create_time() == item['create_time'] and p.status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess: alive = False
                status = None if alive else 'ADOPTED_EXIT'
            if not alive:
                if name not in complete:
                    failed[name] = {'exit': status, 'utc': now(), 'last_progress': rows[name]}
                    event({'stage': 'CHILD_FAILED', 'cell': name, **failed[name]})
                else: event({'stage': 'CHILD_COMPLETE', 'cell': name})
                del running[name]
        # Recovery is explicit and single-use, never an architecture restart.
        for name, approved in control.get('resume_failed', {}).items():
            if approved is True and name in failed and name not in running:
                cell = next(c for c in CELLS if key(c) == name)
                if not (rt.cell_dir(*cell)/'LATEST_PRIVATE.pt').exists():
                    raise RuntimeError('requested resume lacks a formal epoch10+ checkpoint')
                event({'stage': 'EXPLICIT_SAME_SOURCE_RESUME_APPROVED', 'cell': name, 'previous_failure': failed.pop(name)})
                control['resume_failed'][name] = False
                rt.atomic_json(CONTROL, control)
        if not failed:
            for cell in CELLS:
                if len(running) >= parallel: break
                name, path = key(cell), rt.cell_dir(*cell)
                if name in complete or name in running: continue
                if (path/'START_PRIVATE.json').exists() and not (path/'LATEST_PRIVATE.pt').exists():
                    failed[name] = {'exit': 'NO_LATEST_FOR_INCOMPLETE_START', 'utc': now()}; break
                launch_record = {'stage': 'CHILD_LAUNCH', 'cell': name, 'resume': (path/'LATEST_PRIVATE.pt').exists(),
                                 'previous_progress': read(path/'PROGRESS_PRIVATE.json')}
                event(launch_record)
                log = (RUN/(name+'.log')).open('a')
                process = subprocess.Popen([str(rt.ENV_PYTHON), str(Path(__file__)), 'worker', '--seed', str(cell[0]),
                        '--fold', str(cell[1]), '--arm', cell[2]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, close_fds=True)
                log.close()
                try: born = psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess: born = None  # poll() records a fast failure next loop.
                running[name] = {'cell': cell, 'pid': process.pid, 'process': process,
                                  'create_time': born, 'adopted': False}
        snapshot = {'status': 'BLOCKED_CHILD_FAILURE' if failed else 'RUNNING', 'source': SOURCE,
             'utc': now(), 'coordinator_pid': os.getpid(), 'max_parallel': parallel, 'complete_count': len(complete),
             'planned_cells': 30, 'failed': failed, 'cells': rows,
             'running': {name: {k: v for k, v in item.items() if k != 'process'} for name, item in running.items()},
             'resource': resource(), 'valid_endpoints_opened_by_coordinator': False}
        rt.atomic_json(STATE, snapshot)
        with (RUN/'RESOURCE_SAMPLES_PRIVATE.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc': snapshot['utc'], 'parallel': parallel, 'complete': len(complete),
                         'resource': snapshot['resource'], 'running': list(snapshot['running']),
                         'epochs': {name: row['epoch'] for name, row in rows.items()}}, sort_keys=True)+'\n')
        if len(complete) == 30 and not running:
            rt.verify_qualifications(SOURCE)
            postprocess()
            snapshot.update(status='FORMAL_TRAIN_REPLAY_EVALUATION_AND_PUSH_COMPLETE', utc=now(),
                            publication=read(RUN/'PUSH_VERIFIED_PRIVATE.json'))
            rt.atomic_json(STATE, snapshot)
            return
        time.sleep(max(1, 20-(time.monotonic()-started)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('start', 'run', 'worker', 'status', 'restart-coordinator', 'set-parallel', 'start-tuner'))
    p.add_argument('--parallel', type=int)
    p.add_argument('--seed', type=int); p.add_argument('--fold', type=int); p.add_argument('--arm', choices=rt.ARMS)
    a = p.parse_args()
    if a.action == 'worker': return worker((a.seed, a.fold, a.arm))
    if a.action == 'run': return run()
    if a.action == 'status':
        print(json.dumps({'state': read(STATE), 'active': active_cells(), 'resource': resource()})); return
    if a.action == 'start-tuner':
        state = read(STATE)
        if not state or state['status'] != 'RUNNING': raise RuntimeError('matrix must be running before throughput calibration')
        for process in psutil.process_iter(['cmdline']):
            if str(ROOT/'scripts/tune_phase28_concurrency.py') in (process.info['cmdline'] or []):
                raise RuntimeError('an existing tuner already owns concurrency decisions')
        with (RUN/'CONCURRENCY_TUNER_PRIVATE.log').open('a') as log:
            process = subprocess.Popen([str(rt.ENV_PYTHON), str(ROOT/'scripts/tune_phase28_concurrency.py'),
                    '--source', SOURCE, '--max-parallel', '8', '--execute-tuning'], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        print(json.dumps({'status': 'CONCURRENCY_TUNER_LAUNCHED', 'pid': process.pid, 'cap': 8}))
        return
    if a.action == 'set-parallel':
        if a.parallel is None or not 1 <= a.parallel <= MAX_PARALLEL:
            raise RuntimeError('explicit concurrency1..13 required; no training setting changed')
        state = read(STATE)
        if not state or state['status'] not in ('RUNNING', 'BLOCKED_CHILD_FAILURE'):
            raise RuntimeError('no active coordinator stage to adjust')
        old = read(CONTROL)
        new = {**old, 'max_parallel': a.parallel}
        rt.atomic_json(CONTROL, new)
        event({'stage': 'CONCURRENCY_CONTROL_CHANGED', 'before': old['max_parallel'], 'after': a.parallel,
               'resource': resource(), 'current_epochs': {n: r['epoch'] for n, r in state['cells'].items()}})
        print(json.dumps({'status': 'CONCURRENCY_CONTROL_UPDATED', 'before': old['max_parallel'], 'after': a.parallel}))
        return
    RUN.mkdir(parents=True, exist_ok=True)
    if a.action == 'restart-coordinator':
        state = read(STATE)
        if not state or state['status'] not in ('RUNNING', 'BLOCKED_CHILD_FAILURE'):
            raise RuntimeError('only an actively monitored training stage can be adopted')
        preserved = active_cells()
        coordinator = psutil.Process(state['coordinator_pid'])
        command = coordinator.cmdline()
        if str(Path(__file__)) not in command or 'run' not in command:
            raise RuntimeError('coordinator PID identity differs; no process terminated')
        coordinator.terminate(); coordinator.wait(timeout=10)
        if active_cells() != preserved:
            raise RuntimeError('training PID roster changed during coordinator-only restart')
        event({'stage': 'COORDINATOR_ONLY_RESTART_TRAINING_PIDS_PRESERVED',
               'old_pid': state['coordinator_pid'], 'preserved': preserved, 'new_scheduler_sha256': rt.sha(__file__)})
    elif active_cells() or (read(STATE) or {}).get('status') == 'RUNNING':
        raise RuntimeError('existing matrix requires adoption via explicit run, not a second start')
    if not CONTROL.exists(): rt.atomic_json(CONTROL, {'max_parallel': 1, 'resume_failed': {}})
    with (RUN/'COORDINATOR_PRIVATE.log').open('a') as log:
        process = subprocess.Popen([str(rt.ENV_PYTHON), str(Path(__file__)), 'run'], cwd=ROOT,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps({'status': 'COORDINATOR_LAUNCHED', 'pid': process.pid, 'control': str(CONTROL)}))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        RUN.mkdir(parents=True, exist_ok=True)
        event({'stage': 'COORDINATOR_OR_WORKER_EXCEPTION', 'error_type': type(exc).__name__, 'message_private': str(exc)})
        current = read(STATE)
        if current and current.get('coordinator_pid') == os.getpid():
            current.update(status='BLOCKED_COORDINATOR_EXCEPTION', error_type=type(exc).__name__, utc=now())
            rt.atomic_json(STATE, current)
        raise

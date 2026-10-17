"""Schedule the fixed nine Phase22 A cells with bounded GPU concurrency.

No selection, budget alteration or survival work is performed. Each child
revalidates the frozen data/feature/dictionary/source contracts itself.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--concurrency', type=int, choices=[1, 2, 3, 6, 9], default=3)
    p.add_argument('--wait-for-development-features', action='store_true')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root, out = a.data_root.resolve(), a.output_root.resolve()
    if out.is_relative_to(repo/'reports') or out.is_relative_to(repo/'src'):
        raise ValueError('formal runtime must remain private')
    if not (out/'dictionary/DICT_RECEIPT_PRIVATE.json').is_file():
        raise ValueError('qualify the train-only dictionary before scheduling')
    rows = json.loads((root/'metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json').read_text())['rows']
    lock = out/'SCHEDULER_RUNNING_PRIVATE.json'
    if lock.exists():
        previous = json.loads(lock.read_text())
        try:
            os.kill(int(previous['pid']), 0)
        except ProcessLookupError:
            lock.rename(out/('SCHEDULER_STALE_%d_PRIVATE.json' % int(time.time())))
        else:
            raise RuntimeError('existing scheduler PID is live; inspect before restart')
    with lock.open('x') as f:
        json.dump({'pid': os.getpid(), 'started': time.time()}, f)
    last_missing = None
    wait_started = time.monotonic()
    while True:
        missing = 0
        for row in rows:
            if row['split'] in ['train', 'validation']:
                f = root/'features/uni2-h'/(row['section_id_private']+'.json')
                if not f.is_file() or json.loads(f.read_text()).get('status') != 'COMPLETE':
                    missing += 1
        if not missing:
            break
        if not a.wait_for_development_features or time.monotonic()-wait_started > 7200:
            lock.unlink(missing_ok=True)
            raise ValueError('all development feature receipts must exist; no formal cell launched')
        if missing != last_missing:
            print('WAIT_FOR_FROZEN_DEVELOPMENT_FEATURES', missing, flush=True)
            last_missing = missing
        time.sleep(10)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    pending = [(arm, seed) for arm in ['A0', 'A1', 'A2'] for seed in [31, 47, 83]]
    running, finished = {}, []
    logs = out/'process_logs'; logs.mkdir(exist_ok=True)
    started = time.monotonic()

    def save(status):
        path = out/'MATRIX_PROGRESS_PRIVATE.json'; tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'status': status, 'expected_formal_cells': 9, 'finished': finished,
            'pending': len(pending), 'running': len(running), 'concurrency': a.concurrency,
            'seconds': time.monotonic()-started}, indent=2)+'\n')
        tmp.replace(path)

    try:
        while pending or running:
            while pending and len(running) < a.concurrency and not any(x['exit_code'] for x in finished):
                arm, seed = pending.pop(0)
                log = (logs/('%s_seed%d.log' % (arm, seed))).open('ab', buffering=0)
                command = [sys.executable, '-u', 'scripts/run_phase22_external_spatial.py',
                    '--data-root', str(root), '--split-lock', str(root/'metadata/CRC16_ST_DONOR_SPLIT_LOCK_PRIVATE.json'),
                    '--qualification', str(root/'processed/DATA_QUALIFICATION_PRIVATE.json'),
                    '--output-root', str(out), '--arm', arm, '--seed', str(seed)]
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[child.pid] = (child, log, arm, seed)
                print('FORMAL_CELL_LAUNCHED', arm, seed, child.pid, flush=True)
            for pid, (child, log, arm, seed) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                log.close(); finished.append({'arm': arm, 'seed': seed, 'exit_code': code})
                del running[pid]
                print('FORMAL_CELL_EXIT', arm, seed, code, flush=True)
            failed = any(x['exit_code'] for x in finished)
            save('FAILED_STOPPED_PENDING_CELLS' if failed else 'RUNNING')
            if failed and not running:
                raise RuntimeError('formal cell failed; inspect preserved receipt/log')
            if pending or running:
                time.sleep(5)
        save('COMPLETE'); print('FORMAL_NINE_CELL_MATRIX_COMPLETE', flush=True)
    finally:
        if not running:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

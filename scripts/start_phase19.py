import json
import argparse
import os
import subprocess
import sys
from src.phase19_attention.data import P, ROOT, source, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--native-workers', type=int, default=2)
    args = parser.parse_args()
    assert args.workers > 0 and 0 < args.native_workers <= args.workers
    assert json.loads((P/'PREPARATION_DONE.json').read_text())['source'] == source()
    pid_file = P/'SCHEDULER_PID.json'
    if pid_file.exists():
        prior = json.loads(pid_file.read_text())
        try:
            os.kill(prior['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('scheduler is already alive')
    log = (P/'logs/scheduler.log').open('a')
    write(P/'WORKERS.json', {'workers': args.workers, 'native_workers': args.native_workers})
    process = subprocess.Popen([sys.executable, 'scripts/schedule_phase19.py'], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(pid_file, {'pid': process.pid, 'source': source()})
    print('PHASE19_SCHEDULER_STARTED', process.pid, flush=True)


if __name__ == '__main__':
    main()

"""Read-only compact formal matrix progress; no endpoints or model loading."""
import json
from pathlib import Path
import sys
import time
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase28_reread import runtime as rt


def main():
    path = rt.PRIVATE/'matrix_execution/STATE_PRIVATE.json'
    if not path.exists():
        print(json.dumps({'status': 'NO_STATE'})); return
    state = json.loads(path.read_text())
    coordinator_alive = False
    try:
        p = psutil.Process(state['coordinator_pid'])
        coordinator_alive = (str(ROOT/'scripts/coordinate_phase28.py') in p.cmdline()
                             and p.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess: pass
    running = []
    for name, row in state['running'].items():
        cell = row['cell']; base = rt.cell_dir(*cell)
        progress = json.loads((base/'PROGRESS_PRIVATE.json').read_text()) if (base/'PROGRESS_PRIVATE.json').exists() else {}
        entry = {'cell': name, 'pid': row['pid'], 'epoch': progress.get('epoch', 0),
                 'attempt_wall_seconds': progress.get('wall_seconds'), 'fit_cox': progress.get('fit_cox')}
        try:
            p = psutil.Process(row['pid']); cpu = p.cpu_times()
            entry.update(RSS_bytes=p.memory_info().rss, CPU_seconds=cpu.user+cpu.system,
                         process_age_seconds=time.time()-p.create_time(), status=p.status())
        except psutil.NoSuchProcess: entry['status'] = 'EXITED_WAITING_COORDINATOR'
        running.append(entry)
    status = state['status']
    if not coordinator_alive and status in ('RUNNING', 'BLOCKED_CHILD_FAILURE'):
        status = 'COORDINATOR_EXITED_NEEDS_INSPECTION'
    print(json.dumps({'status': status, 'coordinator_alive': coordinator_alive, 'utc': state['utc'], 'source': state['source'],
           'complete': state['complete_count'], 'planned': 30, 'max_parallel': state['max_parallel'],
           'running': running, 'failed': state['failed'], 'resource': state['resource']}))


if __name__ == '__main__': main()

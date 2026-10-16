import json
import subprocess
import sys
import time
from src.phase19_attention.data import P, R, ROOT, cfg, source, write
from src.phase10_trimodal.data import digest
from scripts.run_phase19 import stem


def queue(stage):
    c = cfg(); identity = source()
    jobs = [(arm, seed, fold) for fold in c['folds'] for seed in c['seeds'] for arm in c['arms']]
    todo, live, completed, telemetry = list(jobs), [], [], []
    begin, last = time.monotonic(), 0.
    (P/'logs').mkdir(parents=True, exist_ok=True)
    while todo or live:
        limits = json.loads((P/'WORKERS.json').read_text()) if (P/'WORKERS.json').exists() else {'workers': 4, 'native_workers': 2}
        while todo and len(live) < limits['workers']:
            candidates = [i for i, j in enumerate(todo) if j[0].startswith('A') or
                          sum(not x[3].startswith('A') for x in live) < limits.get('native_workers', 2)]
            if not candidates:
                break
            arm, seed, fold = todo.pop(candidates[0]); label = stem(arm, seed, fold)
            receipt = P/stage/(label+'.json')
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                assert saved['source'] == identity and saved['checkpoint_sha256'] == digest(receipt.with_suffix('.pt'))
                completed.append(label); continue
            log = (P/'logs'/f'{stage}_{label}.log').open('w')
            process = subprocess.Popen([sys.executable, 'scripts/run_phase19.py', '--arm', arm,
                       '--seed', str(seed), '--fold', str(fold), '--stage', stage],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            live.append((process, log, label, arm)); print('START', stage, label, process.pid, flush=True)
        for row in list(live):
            process, log, label, arm = row
            code = process.poll()
            if code is None:
                continue
            log.close(); live.remove(row)
            if code:
                for pp, ll, _, _ in live:
                    pp.terminate(); ll.close()
                write(P/'FAILED.json', {'source': identity, 'role': stage, 'cell': label, 'exit': code})
                raise SystemExit('FAILED '+label)
            completed.append(label); print('DONE', stage, label, len(completed), flush=True)
        if time.monotonic()-last >= 10:
            memory, utilization = map(float, subprocess.check_output(['nvidia-smi',
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            telemetry.append({'unix_time': time.time(), 'role': stage, 'memory_MiB': memory,
                              'GPU_percent': utilization, 'running': len(live), 'workers_limit': limits['workers']})
            write(P/(stage+'_GPU.json'), telemetry); last = time.monotonic()
        write(P/'PROGRESS.json', {'source': identity, 'role': stage, 'completed': len(completed),
              'total': len(jobs), 'running': [j[2] for j in live], 'pending': len(todo),
              'wall_seconds': time.monotonic()-begin})
        if live:
            time.sleep(2)
    write(P/(stage+'_DONE.json'), {'status': 'PASS', 'source': identity,
                                 'formal_fits': len(completed), 'seconds': time.monotonic()-begin})


def main():
    assert json.loads((P/'PREPARATION_DONE.json').read_text())['source'] == source()
    queue('inner')
    import pandas as pd
    rows = [json.loads(path.read_text()) for path in (P/'inner').glob('*.json')]
    by_arm = {a: [r for r in rows if r['arm'] == a] for a in cfg()['arms']}
    fractions = {a: sum(r['actual_epochs'] == cfg()['training']['max_epochs'] for r in rs)/len(rs)
                 for a, rs in by_arm.items()}
    budget_ok = all(x <= cfg()['engineering_gate']['maximum_cap_fraction_per_arm'] for x in fractions.values())
    write(R/'BUDGET_AUDIT.json', {'status': 'PASS' if budget_ok else 'BUDGET_UNRESOLVED',
          'source': source(), 'inner_cap_fractions': fractions,
          'no_outer_results_used': True, 'zero_cap_does_not_prove_global_convergence': True})
    write(R/'EPOCH_SELECTION_LOCK.json', {'status': 'ALL150_INNER_SELECTIONS_LOCKED',
          'source': source(), 'selections': [{'arm': r['arm'], 'seed': r['seed'], 'fold': r['fold'],
          'epoch': r['selected_epoch'], 'checkpoint_sha256': r['checkpoint_sha256']} for r in rows],
          'no_outer_scores_used': True})
    # Even unresolved optimization is reported honestly; the planned finite
    # outer matrix is not reselected by validation or test performance.
    queue('outer')
    write(P/'TRAINING_DONE.json', {'status': 'PASS', 'source': source(), 'new_inner_fits': 150,
          'new_outer_fits': 150, 'new_formal_fits': 300, 'unique_convex_reference_fits': 20,
          'smoke': json.loads((P/'SMOKE_COUNTS.json').read_text()),
          'budget_status': 'PASS' if budget_ok else 'BUDGET_UNRESOLVED',
          'old_neural_checkpoints_used': 0})
    print('ALL_PHASE19_TRAINING_DONE', flush=True)


if __name__ == '__main__':
    main()

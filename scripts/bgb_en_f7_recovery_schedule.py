"""Finite F7 runtime recovery; preserve originals and require CUDA qualification."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_f7_runtime_repair import original_inventory, require
from scripts.bgb_en_f7_determinism_diagnostic import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--origin-manifest', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--max-workers', type=int, default=4)
    args = p.parse_args()
    require(1 <= args.max_workers <= 4, 'The qualified ceiling is four workers')
    lock_path = ROOT/'configs/bgb_en/F7_deterministic_runtime_v1.json'
    lock = json.loads(lock_path.read_text())
    origin = json.loads(args.origin_manifest.read_text())
    args.work.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    contract = dict(status='LOCKED_FINITE_RUNTIME_RECOVERY', script_sha256=sha(__file__),
                    lock_sha256=sha(lock_path), origin_manifest_sha256=sha(args.origin_manifest),
                    units=[dict(partition=part, seed=seed) for part in (0, 1, 2) for seed in (31, 47)],
                    maximum_workers=args.max_workers, teacher_refits=0, original_guard_threshold=2e-5,
                    changes_architecture_or_training_protocol=False)
    write(args.work/'PLAN.json', contract)
    print(json.dumps(dict(status='WAITING_ORIGINAL_QUEUE', **{'planned_new_fits': 6})), flush=True)
    while not (args.private/'SCHEDULER_RECEIPT.json').exists():
        time.sleep(15)
    for stage in ('R1', 'R2'):
        require(json.loads((args.private/(stage+'_SCHEDULE.json')).read_text())['status']=='COMPLETE',
                'Unexpected non-F7 failure in earlier stages')
    require(json.loads((args.private/'R3_SCHEDULE.json').read_text())['status']=='PARTIAL_FAILED_UNITS_CONTINUE_INDEPENDENT_MECHANISMS',
            'The original F7 failures must remain recorded')
    require(original_inventory(args.private/'cells', lock)==origin['original_cells'],
            'Original F7 inventory changed before qualification')
    require(len(list((args.private/'cells').glob('*/COMPLETE.json')))==183,
            'The original queue has unresolved cells beyond the three known F7 failures')
    results = []
    def run(label, script, argv):
        began = time.monotonic()
        with (args.work/(label+'.log')).open('xb') as stream:
            code = subprocess.run([sys.executable, str(ROOT/'scripts'/script)]+list(map(str, argv)),
                                  cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        row = dict(unit=label, returncode=code, elapsed_seconds=time.monotonic()-began)
        results.append(row)
        print(json.dumps(row), flush=True)
        require(code==0, 'Runtime recovery unit failed: '+label)
    # A fixed first successful original checkpoint provides a read-only legacy
    # numerical diagnostic. It is not the unrecorded failed-epoch state.
    run('original_forward', 'bgb_en_f7_determinism_diagnostic.py',
        ['--assets',args.assets,'--private',args.private,'--partition',0,'--seed',31,
         '--backend','inherited','--state','final80','--device','cuda','--batch-limit',999,
         '--repeats',3,'--output',args.work/'ORIGINAL_FORWARD.json'])
    def qualify(part, seed):
        output = args.work/f'CUDA_QUAL_p{part}_s{seed}.json'
        run(f'qual_p{part}_s{seed}', 'bgb_en_f7_determinism_diagnostic.py',
            ['--assets',args.assets,'--private',args.private,'--partition',part,'--seed',seed,
             '--backend','deterministic','--device','cuda','--batch-limit',999,'--repeats',3,'--output',output])
        require(json.loads(output.read_text())['status']=='PASS_SAMPLED_READONLY_FORWARD',
                'Deterministic CUDA qualification failed; originals are untouched')
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for future in as_completed([pool.submit(qualify, part, seed) for part in (0,1,2) for seed in (31,47)]):
            future.result()
    # Both roots are explicit private directories. Verify resolved source and
    # destination parents before moving each of exactly six original folders.
    source_root = (args.private/'cells').resolve()
    archive = args.archive_root.resolve()
    require(archive.name=='archived_F7_original' and archive.parent==args.private.parent.resolve(),
            'Archive must stay in the explicitly named BG-BEN private parent')
    archive.mkdir(exist_ok=False)
    for row in origin['original_cells']:
        source, destination = source_root/row['cell'], archive/row['cell']
        require(source.resolve().parent==source_root and destination.parent==archive and not destination.exists(),
                'Unsafe or existing archive target')
        source.rename(destination)
    require(original_inventory(archive, lock)==origin['original_cells'], 'Archived original files changed')
    write(args.work/'ARCHIVE_RECEIPT.json', dict(status='PRESERVED_ALL_SIX_ORIGINAL_F7_CELLS',
        origin_manifest_sha256=sha(args.origin_manifest), original_cells=origin['original_cells']))
    def train(part, seed):
        run(f'train_p{part}_s{seed}', 'bgb_en_f7_runtime_repair.py',
            ['--stage','train','--assets',args.assets,'--private',args.private,'--original-root',archive,
             '--origin-manifest',args.origin_manifest,'--partition',part,'--seed',seed,'--device','cuda',
             '--qualification',args.work/f'CUDA_QUAL_p{part}_s{seed}.json',
             '--output',args.work/f'RUNTIME_TRAIN_p{part}_s{seed}.json'])
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for future in as_completed([pool.submit(train, part, seed) for part in (0,1,2) for seed in (31,47)]):
            future.result()
    require(original_inventory(archive, lock)==origin['original_cells'], 'Original archive changed during training')
    write(args.work/'QUEUE_RECEIPT.json', dict(status='SIX_FITS_COMPLETE_PENDING_INDEPENDENT_CERTIFICATION',
        contract=contract, actual_new_fits=6, actual_new_updates=480, teacher_refits=0,
        elapsed_seconds=time.monotonic()-start, units=results,
        original_schedule_modified=False, original_failed_updates=None))


if __name__ == '__main__':
    main()

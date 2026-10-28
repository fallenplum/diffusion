"""Execute only the finite outcome-free R6 source matrix and saved-state replay."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--private-root', type=Path, required=True)
    parser.add_argument('--launch-gate', type=Path, required=True)
    parser.add_argument('--launch-gate-sha256', required=True)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        raise ValueError('SOURCE_WORKER_CEILING_EIGHT')
    config = json.loads(args.config.read_text())
    if config['status'] != 'LOCKED_BEFORE_SOURCE_FITS':
        raise ValueError('FORMAL_SOURCE_LOCK_REQUIRED')
    if config.get('maximum_source_workers') != 8:
        raise ValueError('LOCKED_SOURCE_RESOURCE_CEILING_REQUIRED')
    cells = config['cells']
    identities = [(x['arm'], x['control'], x['heldout_donor'], x['seed']) for x in cells]
    if len(cells) != 147 or len(set(identities)) != 147:
        raise ValueError('R6_FINITE_SOURCE_MATRIX_CHANGED')
    if sha(args.launch_gate) != args.launch_gate_sha256:
        raise ValueError('SOURCE_LAUNCH_GATE_SHA_CHANGED')
    gate = json.loads(args.launch_gate.read_text())
    if gate.get('r6_formal_training_allowed') is not True:
        raise ValueError('COMPLETE_R5_DECISION_REQUIRED')
    args.private_root.mkdir(parents=True, exist_ok=True)
    logs = args.private_root/'logs'; logs.mkdir(exist_ok=True)
    source_hashes = config['source_code_sha256']
    for relative, digest in source_hashes.items():
        if sha(ROOT/relative) != digest:
            raise ValueError('SOURCE_CODE_CHANGED_AFTER_LOCK')
    schedule_start = time.monotonic()
    records = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    common = ['--config',str(args.config),'--source-cache',str(args.source_cache),
              '--private-root',str(args.private_root),'--launch-gate',str(args.launch_gate),
              '--launch-gate-sha256',args.launch_gate_sha256,'--device',args.device]
    if args.resume:
        common += ['--resume']

    def execute(index):
        cell = cells[index]
        name = f"{index:03d}_{cell['arm']}_{cell['control']}_d{cell['heldout_donor']}_s{cell['seed']}"
        began = time.monotonic()
        with (logs/(name+'.log')).open('ab' if args.resume else 'xb') as stream:
            for mode in ('fit','replay'):
                command = [sys.executable, str(ROOT/'scripts/bgb_en_r6_pretrain.py'),
                           '--mode',mode,'--cell-index',str(index)]+common
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    record = dict(cell_index=index, mode=mode, status='FAILED', returncode=result.returncode,
                                  elapsed_seconds=time.monotonic()-began)
                    records.append(record); print(json.dumps(record), flush=True)
                    raise RuntimeError('R6_SOURCE_UNIT_FAILED_NO_AUTOMATIC_RETRY')
        record = dict(cell_index=index, status='FIT_AND_ACTUAL_REPLAY_COMPLETE',
                      elapsed_seconds=time.monotonic()-began)
        records.append(record);print(json.dumps(record),flush=True)

    try:
        # Preserve source-family order; controls are not selected using scores.
        for phase in ('PRIMARY','SPATIAL','SPOT_ALIGNMENT','FULL_TRANSFER'):
            indices = [i for i,c in enumerate(cells) if c['schedule_phase']==phase]
            pool = ThreadPoolExecutor(max_workers=args.max_workers)
            futures=[]
            try:
                futures = [pool.submit(execute,i) for i in indices]
                for future in as_completed(futures):
                    future.result()
            except Exception:
                for future in futures:future.cancel()
                pool.shutdown(wait=True,cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
            write(args.private_root/(phase+'_SCHEDULE_COMPLETE.private.json'),
                  dict(status='COMPLETE', phase=phase, cell_indices=indices,
                       config_sha256=sha(args.config)))
        for relative,digest in source_hashes.items():
            if sha(ROOT/relative)!=digest:raise ValueError('SOURCE_CHANGED_DURING_FITS')
        write(args.private_root/'SOURCE_SCHEDULE_COMPLETE.private.json',
              dict(status='COMPLETE_147_SOURCE_UNITS_AND_ACTUAL_REPLAY',
                   planned_logical_units=147,planned_optimized_units=135,
                   records=records,source_config_sha256=sha(args.config),
                   elapsed_seconds=time.monotonic()-schedule_start,
                   no_DSS_or_PFI_loaded_by_scheduler=True))
    except Exception as exc:
        write(args.private_root/'SOURCE_SCHEDULE_ERROR.private.json',
              dict(status='STOPPED_NO_AUTOMATIC_RETRY',error_type=type(exc).__name__,
                   records=records,source_config_sha256=sha(args.config),
                   elapsed_seconds=time.monotonic()-schedule_start))
        raise


if __name__=='__main__':main()

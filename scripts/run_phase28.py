"""Execute one source-locked formal inner cell; never inspect valid outcomes."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--fold', type=int, required=True)
    p.add_argument('--arm', choices=('C00', 'C10', 'C01', 'C11', 'DIRECT'), required=True)
    p.add_argument('--source', required=True); p.add_argument('--execute-formal', action='store_true')
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    from src.phase28_reread import runtime
    if not a.execute_formal or runtime.source() != a.source:
        raise ValueError('explicit formal command and exact source are required')
    from src.phase28_reread.train import train_cell
    value = train_cell(a.seed, a.fold, a.arm, resume=a.resume)
    print('PHASE28_CELL_COMPLETE', value['seed'], value['fold'], value['arm'], flush=True)

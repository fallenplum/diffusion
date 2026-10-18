"""Exactly one fixed extraction or convex readout cell; no validation metrics."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase27_readout import runtime
from src.phase27_readout.pipeline import extract,fit_cell


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--operation',choices=('extract','fit'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);a=p.parse_args()
    if a.seed not in runtime.cfg()['seeds'] or a.fold not in runtime.cfg()['folds']:raise ValueError('cell outside fixed lock')
    (extract if a.operation=='extract' else fit_cell)(a.seed,a.fold)
    print('PHASE27_CELL_COMPLETE',a.operation,a.seed,a.fold,flush=True)

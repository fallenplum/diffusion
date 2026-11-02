"""Guarded CLI; an entry point is not evidence that a stage was executed."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.qpr.protocol import run_stage
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--lock',required=True);p.add_argument('--reports',default=str(ROOT/'reports/qpr'));a=p.parse_args()
    run_stage(a.stage,report_root=a.reports,lock=a.lock)

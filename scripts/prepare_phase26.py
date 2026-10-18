"""Create each new main once; preserve original Phase25 preprocessors."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase26_multiquery.data import prepare


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int);a=p.parse_args()
    prepare(a.fold)
    print('PHASE26_MAIN_READY',a.fold,flush=True)

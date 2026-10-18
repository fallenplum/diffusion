"""Prepare only the admitted shared raw-feature main effects; no local model."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--stage', choices=('prototype', 'outer'), default='prototype')
    args = parser.parse_args()
    from src.phase25_local.data import prepare
    saved, receipt = prepare(args.fold, args.stage)
    print('PHASE25_MAIN_PREPARED', args.stage, args.fold, receipt['source'],
          receipt['sha256'], saved['audit']['convex_fits'], flush=True)


if __name__ == '__main__':
    main()

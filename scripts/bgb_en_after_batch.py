"""Serialize two bounded GPU batches without editing the original scheduler."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--after',required=True,type=Path);p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--logs',required=True,type=Path);p.add_argument('--core-private',required=True,type=Path);args=p.parse_args()
    while not args.after.exists():time.sleep(15)
    if json.loads(args.after.read_text())['status']!='COMPLETE':raise ValueError('Prior batch contains failed units; resolve before this batch')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/bgb_en_batch.py'),'--plan',str(args.plan),'--logs',str(args.logs),
        '--wait-for-core',str(args.core_private),'--max-workers','4','--resume'],cwd=root)
    if result.returncode or json.loads((args.logs/'BATCH_RECEIPT.json').read_text())['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()

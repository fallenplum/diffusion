"""Exactly four fresh end-to-end models in one qualified patient split."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase24_cdei_g1.runtime import (cfg,source,cell_dir,authorization,verify_done,
                                      atomic_json,sha)


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['inner','outer'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();c=cfg();s=source()
    if a.seed not in c['seeds'] or a.fold not in c['folds']:raise ValueError('outside locked matrix')
    authorization(a.role,s);d=cell_dir(a.role,a.seed,a.fold);d.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (d/'CELL.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if (d/'DONE_PRIVATE.json').exists():
            verify_done(d,s,a.role,a.seed,a.fold);print('VERIFIED_COMPLETE',d.name,flush=True);return
        ledger=d/'ATTEMPTS_PRIVATE.json'
        if ledger.exists():raise ValueError('incomplete prior attempt requires explicit review; no automatic restart')
        attempt={'source':s,'status':'RUNNING','kernel_updates':0,'arm_epochs':{},'unix_start':time.time()}
        atomic_json(ledger,attempt)
        def step(arm,epoch):
            attempt['kernel_updates']+=1;attempt['arm_epochs'][arm]=epoch;atomic_json(ledger,attempt)
        try:
            import torch
            from src.phase23_cdei.data import Data
            from src.phase24_cdei_g1.train import run_cell
            ds=Data(a.fold,a.role,a.seed,device=a.device)
            if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            result=run_cell(ds,s,d,epochs=80,step_callback=step)
            if source()!=s:raise ValueError('source changed during training')
            if len(result['arms'])!=4 or attempt['kernel_updates']!=320:raise ValueError('incomplete formal budget')
            done={'status':'COMPLETE','source':s,'role':a.role,'seed':a.seed,'fold':a.fold,
                  'arms':4,'epochs':80,'checkpoint_sha256':sha(d/'checkpoint.pt'),
                  'wall_seconds':time.time()-attempt['unix_start'],
                  'peak_GPU_reserved_bytes':int(torch.cuda.max_memory_reserved()) if a.device.startswith('cuda') else 0}
            atomic_json(d/'DONE_PRIVATE.json',done);verify_done(d,s,a.role,a.seed,a.fold)
            attempt['status']='COMPLETE'
        except BaseException as error:
            attempt.update(status='FAILED',error_type=type(error).__name__);raise
        finally:
            attempt['unix_end']=time.time();atomic_json(ledger,attempt)
        print('CELL_COMPLETE',d.name,flush=True)


if __name__=='__main__':main()

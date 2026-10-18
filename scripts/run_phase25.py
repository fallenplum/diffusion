"""One immutable prototype cell: common R0 plus three fresh local models."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase25_local.runtime import cfg,source,PRIVATE,cell_dir,authorization,verify_done,atomic_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prototype','outer'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();c=cfg();identity=source()
    if a.seed not in c[a.stage]['seeds'] or a.fold not in c[a.stage]['folds']:raise ValueError('outside fixed matrix')
    authorization(a.stage,identity);path=cell_dir(a.stage,a.seed,a.fold);path.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (path/'CELL.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (path/'DONE_PRIVATE.json').exists():
            verify_done(path,identity,a.stage,a.seed,a.fold);print('VERIFIED_COMPLETE',path.name,flush=True);return
        journal=path/'ATTEMPTS_PRIVATE.json'
        if journal.exists():raise ValueError('prior incomplete attempt requires explicit review; no automatic retry')
        attempt={'source':identity,'status':'RUNNING','optimizer_steps':0,'arm_epochs':{},'unix_start':time.time()}
        atomic_json(journal,attempt)
        def step(arm,epoch):
            attempt['optimizer_steps']+=1;attempt['arm_epochs'][arm]=epoch;atomic_json(journal,attempt)
        try:
            import torch
            from src.phase25_local.data import Data
            from src.phase25_local.train import run_cell
            ds=Data(a.fold,a.stage,a.seed,device=a.device)
            if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            record=run_cell(ds,identity,a.seed,a.fold,a.stage,80,step)
            if source()!=identity:raise ValueError('source drift during formal fitting')
            if len(record['arms'])!=3 or attempt['optimizer_steps']!=240:raise ValueError('incomplete formal budget')
            torch.save(record,path/'checkpoint.partial');(path/'checkpoint.partial').replace(path/'checkpoint.pt')
            done={'status':'COMPLETE','source':identity,'stage':a.stage,'seed':a.seed,'fold':a.fold,
                'model_slots':4,'neural_models':3,'epochs':80,'optimizer_steps':240,
                'convex_main_shared_reference':True,'checkpoint_sha256':sha(path/'checkpoint.pt'),
                'wall_seconds':time.time()-attempt['unix_start'],
                'peak_GPU_reserved_bytes':int(torch.cuda.max_memory_reserved()) if a.device.startswith('cuda') else 0}
            atomic_json(path/'DONE_PRIVATE.json',done);verify_done(path,identity,a.stage,a.seed,a.fold)
            attempt['status']='COMPLETE'
        except BaseException as error:
            attempt.update(status='FAILED',error_type=type(error).__name__);raise
        finally:
            attempt['unix_end']=time.time();atomic_json(journal,attempt)
        print('CELL_COMPLETE',path.name,flush=True)


if __name__=='__main__':main()

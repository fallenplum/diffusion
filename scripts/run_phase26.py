"""One authorized, fail-closed Phase26 prototype cell; no endpoint evaluation."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase26_multiquery import runtime
from src.phase26_multiquery.data import Data
from src.phase26_multiquery.train import run_cell


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();c=runtime.cfg()
    if a.fold not in c['prototype']['folds'] or a.seed not in c['prototype']['seeds']:raise ValueError('cell outside lock')
    identity=runtime.source();runtime.authorization(identity);path=runtime.cell_dir(a.seed,a.fold)
    if (path/'DONE_PRIVATE.json').exists():
        runtime.verify_done(path,identity,a.seed,a.fold);print('REUSE_COMPLETE',flush=True);return
    path.mkdir(parents=True,exist_ok=True)
    if (path/'STARTED_PRIVATE.json').exists():raise ValueError('existing attempt requires explicit review')
    runtime.atomic_json(path/'STARTED_PRIVATE.json',{'source':identity,'seed':a.seed,'fold':a.fold})
    began=time.monotonic()
    ds=Data(a.fold,a.seed,device='cuda')
    def progress(arm,epoch):
        if epoch%20==0:print(f'PHASE26 seed={a.seed} fold={a.fold} {arm} epoch={epoch}',flush=True)
    result=run_cell(ds,identity,a.seed,a.fold,80,progress)
    if runtime.source()!=identity:raise ValueError('source drift during formal fit')
    torch.save(result,path/'checkpoint.pt')
    runtime.atomic_json(path/'DONE_PRIVATE.json',{'status':'COMPLETE','source':identity,'stage':'prototype',
        'seed':a.seed,'fold':a.fold,'model_slots':4,'neural_models':3,'epochs':80,'optimizer_steps':240,
        'checkpoint_sha256':runtime.sha(path/'checkpoint.pt'),'wall_seconds':time.monotonic()-began})
    print('PHASE26_CELL_COMPLETE',a.seed,a.fold,flush=True)


if __name__=='__main__':main()

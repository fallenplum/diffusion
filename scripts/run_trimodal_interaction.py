import argparse,json,os
from pathlib import Path
import torch
from src.trimodal_interaction.data import P,R,cfg,source,write,digest
from src.trimodal_interaction.train import train_cell


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);a=p.parse_args();c=cfg()
    if a.arm not in c['arms'] or a.seed not in c['seeds'] or a.fold not in c['folds']:raise ValueError('not a locked cell')
    lock=json.loads((P/'LOCK.json').read_text())
    if lock['source']!=source():raise ValueError('training source changed after lock')
    label=f'{a.arm}_seed{a.seed}_fold{a.fold}';target=P/'cells'/f'{label}.json';ck=P/'checkpoints'/f'{label}.pt'
    target.parent.mkdir(parents=True,exist_ok=True);ck.parent.mkdir(exist_ok=True)
    if target.exists():
        old=json.loads(target.read_text())
        if old['source']!=source() or digest(ck)!=old['checkpoint_sha256']:raise ValueError('completed cell changed')
        print('ALREADY_VERIFIED',label,flush=True);return
    result=train_cell(a.arm,a.seed,a.fold);tmp=ck.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(ck)
    write(target,{'source':source(),'checkpoint_sha256':digest(ck),'summary':result['summary']});print('DONE',label,json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()

"""One fresh DIRECT fit; source-locked, resumable and separately accounted."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from src.phase21_relative.data import P,R,cfg,source,write
from src.phase21_relative.train import train_cell
from src.phase10_trimodal.data import digest

def stem(seed,fold): return f'DIRECT_seed{seed}_fold{fold}'

def record(path):
    value=json.loads(Path(path).read_text())
    if value['source']!=source(): raise ValueError('training source drift')
    return value

def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fold',type=int,required=True);parser.add_argument('--stage',choices=['inner','outer'],required=True)
    args=parser.parse_args();c=cfg();identity=source()
    if args.seed not in c['seeds'] or args.fold not in c['folds']: raise ValueError('outside finite matrix')
    record(R/'PROTOCOL_LOCK.json')
    if record(R/'QUALIFICATION.json')['status']!='PASS': raise ValueError('qualification missing')
    target=P/args.stage/(stem(args.seed,args.fold)+'.pt');target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=target.with_suffix('.json')
        if receipt.exists():
            q=record(receipt)
            if q['checkpoint_sha256']!=digest(target): raise ValueError('existing cell changed')
            print('VERIFIED_PHASE21',args.stage,target.stem,flush=True);return
        epoch=None;inner_sha=None
        if args.stage=='outer':
            selected=record(R/'EPOCH_SELECTION_LOCK.json')
            if len(selected['selections'])!=15: raise ValueError('all15 inner selections required')
            inner=P/'inner'/target.name;q=record(inner.with_suffix('.json'))
            if digest(inner)!=q['checkpoint_sha256']: raise ValueError('inner changed')
            epoch=q['selected_epoch'];inner_sha=digest(inner)
        ledger_path=P/'attempts'/args.stage/(target.stem+'.json');ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
        attempt={'source':identity,'status':'RUNNING','steps':0,'unix_start':time.time()};ledger['attempts'].append(attempt);write(ledger_path,ledger)
        def stepped(kind,epoch):
            if kind!='generator': raise ValueError('unexpected optimizer')
            attempt['steps']+=1;attempt['last_epoch']=epoch;write(ledger_path,ledger)
        try:
            ck=train_cell('DIRECT',args.seed,args.fold,args.stage,epochs=epoch,progress_callback=stepped)
            if ck['source']!=identity or source()!=identity: raise ValueError('source drift during fit')
            ck['inner_checkpoint_sha256']=inner_sha
            temp=target.with_name(target.name+f'.{os.getpid()}.tmp');torch.save(ck,temp);temp.replace(target)
            write(receipt,{'status':'PASS','source':identity,'seed':args.seed,'fold':args.fold,'role':args.stage,
                 'arm':'DIRECT','selected_epoch':ck['selected_epoch'],'summary':ck['summary'],
                 'checkpoint_sha256':digest(target),'inner_checkpoint_sha256':inner_sha})
        except Exception as error:
            attempt.update(status='FAILED',error_type=type(error).__name__,unix_end=time.time());write(ledger_path,ledger);raise
        attempt.update(status='SUCCESS',unix_end=time.time());write(ledger_path,ledger)
        print('PHASE21_CELL_COMPLETE',args.stage,target.stem,flush=True)

if __name__=='__main__': main()

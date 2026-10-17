"""One finite, source-locked Phase20 fit; retries retain actual-step ledgers."""
import argparse
import json
import os
import time
import torch
from src.phase20_selective.data import P,R,cfg,source,write
from src.phase20_selective.train import train_cell
from src.phase10_trimodal.data import digest
from src.phase20_selective.provenance import require_record, verify_qualified_tests


def stem(arm,seed,fold): return f'{arm}_seed{seed}_fold{fold}'


def main():
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fold',type=int,required=True);parser.add_argument('--stage',choices=['inner','outer'],required=True)
    args=parser.parse_args();config=cfg();identity=source()
    if args.arm not in config['arms'] or args.seed not in config['seeds'] or args.fold not in config['folds']:
        raise ValueError('fit outside locked finite matrix')
    label=stem(args.arm,args.seed,args.fold);target=P/args.stage/(label+'.pt');target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in [R/'PROTOCOL_LOCK.json',P/'PREPARATION_DONE.json']:
            require_record(path)
        for name in ['OPERATOR_AND_PIPELINE_TESTS','DATA_AUDIT','DATA_LEAKAGE_AUDIT','MODEL_QUALIFICATION']:
            gate=require_record(R/(name+'.json'))
            if gate['status']!='PASS':
                raise ValueError('missing/stale qualification '+name)
            if name=='OPERATOR_AND_PIPELINE_TESTS': verify_qualified_tests(gate)
        receipt=target.with_suffix('.json')
        if receipt.exists():
            require_record(receipt,target)
            print('VERIFIED_PHASE20_CELL',label,flush=True);return
        epochs,inner_hash=None,None
        if args.stage=='outer':
            epoch_lock=require_record(R/'EPOCH_SELECTION_LOCK.json')
            if epoch_lock['status']!='ALL135_INNER_SELECTIONS_LOCKED':
                raise ValueError('all inner selections must be locked before outer')
            selection=P/'inner'/(label+'.json');inner=selection.with_suffix('.pt')
            row=require_record(selection,inner);inner_hash=digest(inner)
            epochs=row['selected_epoch']
        ledger_path=P/'attempts'/args.stage/(label+'.json')
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts':[]}
        attempt={'source':identity,'arm':args.arm,'seed':args.seed,'fold':args.fold,'role':args.stage,
                 'status':'RUNNING','generator_steps':0,'unix_start':time.time()}
        ledger['attempts'].append(attempt);write(ledger_path,ledger)
        def stepped(kind,epoch):
            if kind!='generator': raise ValueError('unexpected optimizer kind')
            attempt['generator_steps']+=1;attempt['last_epoch']=epoch;write(ledger_path,ledger)
        try:
            ck=train_cell(args.arm,args.seed,args.fold,args.stage,epochs=epochs,progress_callback=stepped)
            if ck['source']!=identity or source()!=identity: raise ValueError('training source drift')
            ck['inner_checkpoint_sha256']=inner_hash
            if args.stage=='outer': ck['inner_checkpoint_actual_source']=row['source']
            temp=target.with_name(target.name+f'.{os.getpid()}.tmp');torch.save(ck,temp);temp.replace(target)
            write(receipt,{'status':'PASS','source':identity,'arm':args.arm,'seed':args.seed,'fold':args.fold,
                'role':args.stage,'selected_epoch':ck['selected_epoch'],'actual_epochs':ck['summary']['actual_epochs'],
                'summary':ck['summary'],'checkpoint_sha256':digest(target),'inner_checkpoint_sha256':inner_hash})
        except Exception as error:
            attempt['status']='FAILED';attempt['error_type']=type(error).__name__;write(ledger_path,ledger);raise
        attempt['status']='RETURNED_SUCCESSFULLY';attempt['unix_end']=time.time();write(ledger_path,ledger)
        print('PHASE20_CELL_COMPLETE',label,args.stage,flush=True)


if __name__=='__main__': main()

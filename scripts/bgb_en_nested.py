"""Run the fixed four-arm R4 inner pool separately within each outer train."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_run import load_inputs,write
from scripts.bgb_en_outer import validate_selection_lock
from src.bgb_en.data import sha256
from src.bgb_en.survival import combine_metrics


def select_receipts(private,fold,config,lineage):
    results=[];receipts=[]
    for arm in config['selection']['nested_interaction_candidates']:
        by_seed={};parameters=[]
        for seed in config['prototype']['neural_seeds']:
            parts=[]
            for part in range(3):
                path=private/'cells'/f'R3_outer{fold}_inner{part}_{arm}_seed{seed}'/'COMPLETE.json'
                value=json.loads(path.read_text())
                identity=dict(stage='R3',arm=arm,outer_fold=fold,inner_partition=part,seed=seed)
                if any(value.get(key)!=expected for key,expected in identity.items()):raise ValueError('Nested receipt unit identity mismatch')
                if (value['status']!='COMPLETE' or value['lineage']!=lineage or value['outer_test_evaluated'] is not False
                    or value['epochs']!=config['training']['epochs']):raise ValueError('Nested selection found an invalid or changed inner fit')
                if sha256(path.parent/'model.pt')!=value['artifact_sha256']:raise ValueError('Nested model changed')
                parts.append(value['metrics']);parameters.append(value['parameters'])
                receipts.append(dict(arm=arm,partition=part,seed=seed,receipt_sha256=sha256(path),artifact_sha256=value['artifact_sha256']))
            by_seed[str(seed)]=combine_metrics(parts)
        results.append(dict(arm=arm,Harrell=float(np.mean([row['Harrell'] for row in by_seed.values()])),
            Uno60=float(np.mean([row['Uno60'] for row in by_seed.values()])),parameters=max(parameters),seeds=by_seed))
    chosen=sorted(results,key=lambda row:(-row['Harrell'],-row['Uno60'],row['parameters'],row['arm']))[0]
    return chosen,results,receipts


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--primary-private',type=Path,required=True);p.add_argument('--global-selection-lock',type=Path,required=True)
    p.add_argument('--inner-report',type=Path,required=True);p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--stage',choices=['inner','select'],required=True);p.add_argument('--max-workers',type=int,default=4)
    p.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json');p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if not 1<=args.max_workers<=4:raise ValueError('Locked GPU worker ceiling is four')
    _,config,_,_,lineage=load_inputs(args)
    lock=json.loads(args.global_selection_lock.read_text());inner=json.loads(args.inner_report.read_text())
    validate_selection_lock(lock,inner,lineage,config,sha256(args.config),sha256(args.inner_report),args.fold)
    if 'NESTED_INTERACTION' not in lock['finalists']:raise ValueError('Prototype gate did not authorize this neural finalist')
    fits=args.primary_private if args.fold==1 else args.private/'nested_inner'
    logdir=args.private/'logs';logdir.mkdir(parents=True,exist_ok=True)
    if args.stage=='inner' and args.fold!=1:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        base=[sys.executable,str(ROOT/'scripts/bgb_en_run.py'),'--assets',str(args.assets),'--private',str(fits),
              '--outer-fold',str(args.fold),'--device',args.device,'--config',str(args.config)]
        def unit(stage,arm,part,seed):
            command=base+['--stage',stage,'--arm',arm,'--partition',str(part),'--seed',str(seed)]
            if args.resume:command+=['--resume']
            label=f'outer{args.fold}_{stage}_p{part}_{arm}_s{seed}';start=time.monotonic()
            with (logdir/(label+'.log')).open('ab' if args.resume else 'xb') as handle:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            print(json.dumps(dict(unit=label,returncode=result.returncode,seconds=time.monotonic()-start)),flush=True)
            if result.returncode:raise RuntimeError('Nested inner unit failed: '+label)
        for part in range(3):unit('prepare','P0',part,31)
        tasks=[('R3',arm,part,seed) for arm in config['selection']['nested_interaction_candidates'] for part in range(3) for seed in config['prototype']['neural_seeds']]
        failures=[]
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            for future in as_completed([pool.submit(unit,*task) for task in tasks]):
                try:future.result()
                except Exception as exc:failures.append(str(exc))
        if failures:raise RuntimeError('Nested inner matrix incomplete; do not evaluate outer')
    chosen,results,receipts=select_receipts(fits,args.fold,config,lineage)
    selected=dict(status='LOCKED_WITHIN_OUTER_TRAIN_BEFORE_TEST',outer_fold=args.fold,selected_arm=chosen['arm'],
        selection_metric='mean_seed_of_within_inner_partition_concordance_counts_Harrell_then_Uno60_then_parameters',
        selected_epochs=80,selected_gamma=.1,outer_test_evaluated=False,
        selection_lock_sha256=sha256(args.global_selection_lock),inner_report_sha256=sha256(args.inner_report),
        source_lineage_sha256=hashlib.sha256(json.dumps(lineage,sort_keys=True).encode()).hexdigest(),
        source_sha256=sha256(__file__),candidates=results,receipts=receipts,
        source_reuse='preexisting_outer1_inner_only_cells' if args.fold==1 else 'new_current_outer_train_inner_fits',
        unique_neural_fits=0 if args.fold==1 else 24,unique_Cox_fits=0 if args.fold==1 else 39,
        unique_beta_fits=0 if args.fold==1 else 3)
    output=args.private/'selection'/f'outer{args.fold}.json'
    if output.exists() and json.loads(output.read_text())!=selected:raise ValueError('Existing outer selection differs')
    write(output,selected)
    print(json.dumps(dict(status=selected['status'],outer_fold=args.fold,selected_arm=chosen['arm'])))


if __name__=='__main__':main()

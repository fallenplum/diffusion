"""Qualified read-only replay and diagnostics of the thirty saved P20 S16s.

Writes only new Phase21 outputs. There is no optimizer, new fit, epoch selection
or replacement of an ancestor receipt. Raw risks and indices stay private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase20_selective import data as parent_data, train as parent_train
from src.phase20_selective.provenance import require_record
from src.phase10_trimodal.data import digest
from scripts.validate_phase20 import qualified_checkpoint, qualify

P=Path('/root/autodl-tmp/phase21_relative_private')
R=ROOT/'reports/phase21_relative_attention'
INVENTORY=P/'SAVED_S16_ORIGINAL_INVENTORY.json'
PARENT_SOURCE='e7b6383cfc23e4fbb8f94419a88bff5db1113babcb967672f87e0f13c74ba548'
SEEDS=[31,47,83]
FOLDS=[1,2,3,4,5]
ROLES=['inner','outer']
MODES=[('matched','matched_risk'),('off','interaction_off_risk'),
       ('constantC','constantC_risk'),('anchor','anchor_risk')]


def require(value,message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    def cast(item):
        if isinstance(item,np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,default=cast,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def stem(role,seed,fold):
    return f'S16_{role}_seed{seed}_fold{fold}'


def checkpoint_path(role,seed,fold):
    return parent_data.P/role/f'S16_seed{seed}_fold{fold}.pt'


def audit_identity():
    diagnostic=ROOT/'src/phase21_relative/diagnostics.py'
    require(diagnostic.is_file(),'Phase21 diagnostic implementation is not ready')
    initial=[Path(__file__).resolve(),diagnostic]
    files=dict(parent_data.source_files())
    closure=parent_data.prior._import_closure(initial)
    require(not any(path.as_posix().endswith('/src/phase21_relative/train.py') for path in closure),
            'saved-model diagnostics must not import a mutable new training pipeline')
    files.update({path.relative_to(ROOT).as_posix():digest(path) for path in closure})
    require(parent_data.source()==PARENT_SOURCE,'Phase20 runtime/data ancestry changed')
    value={'parent_runtime_source':PARENT_SOURCE,'actual_imported_source_files':files}
    identity=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return identity,value


def _published_s16_cells():
    with (parent_data.R/'CELL_MANIFEST.csv').open(encoding='utf-8',newline='') as stream:
        rows=[row for row in csv.DictReader(stream) if row['arm']=='S16' and row['fit_classification']=='new_neural_fit']
    mapping={(row['role'],int(row['seed']),int(row['fold'])):row for row in rows}
    require(len(rows)==len(mapping)==30 and set(mapping)=={(r,s,f) for r in ROLES for s in SEEDS for f in FOLDS},
            'published Phase20 S16 matrix differs')
    return mapping


def snapshot_inventory(verify_all=False):
    """Concurrent-safe immutable snapshot; returns original receipts unchanged."""
    import fcntl
    P.mkdir(parents=True,exist_ok=True)
    with (P/'SAVED_S16_ORIGINAL_INVENTORY.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        require(parent_data.source()==PARENT_SOURCE,'Phase20 runtime source differs')
        if INVENTORY.exists():
            value=read(INVENTORY)
            require(value['status']=='PASS' and value['parent_runtime_source']==PARENT_SOURCE,
                    'saved S16 inventory from another parent')
            expected={(r,s,f) for r in ROLES for s in SEEDS for f in FOLDS}
            require(len(value['checkpoints'])==30 and {(q['role'],q['seed'],q['fold']) for q in value['checkpoints']}==expected,
                    'saved S16 inventory incomplete')
            if verify_all:
                for item in value['checkpoints']:
                    path=checkpoint_path(item['role'],item['seed'],item['fold'])
                    receipt=require_record(path.with_suffix('.json'),checkpoint_path=path)
                    require(digest(path)==item['checkpoint_sha256'] and digest(path.with_suffix('.json'))==item['receipt_sha256']
                            and receipt==item['original_receipt'],'original saved S16 artifact/receipt changed')
            for name,expected in value['parent_public_files'].items():
                require(digest(parent_data.R/name)==expected,'parent published evidence changed')
            return value
        rows=[];published=_published_s16_cells()
        for role in ROLES:
            for seed in SEEDS:
                for fold in FOLDS:
                    path=checkpoint_path(role,seed,fold)
                    receipt=require_record(path.with_suffix('.json'),checkpoint_path=path)
                    require(receipt['status']=='PASS' and (receipt['arm'],receipt['seed'],receipt['fold'],receipt['role'])==
                            ('S16',seed,fold,role),'saved S16 checkpoint identity differs')
                    original=published[role,seed,fold]
                    require(receipt['checkpoint_sha256']==original['checkpoint_sha256']
                            and receipt['source']==original['actual_fit_source'],
                            'saved S16 does not match the published Phase20 cell manifest')
                    rows.append({'arm':'S16','role':role,'seed':seed,'fold':fold,
                        'checkpoint_sha256':digest(path),'receipt_sha256':digest(path.with_suffix('.json')),
                        'actual_fit_source':receipt['source'],'original_receipt':receipt})
        parent_names=['ARTIFACT_MANIFEST.json','PROTOCOL_LOCK.json','AUDIT_REPAIR_LOCK.json',
                      'AUDIT_REPAIR_COMPATIBILITY.json','CHECKPOINT_REPLAY.json','CELL_MANIFEST.csv']
        public={name:digest(parent_data.R/name) for name in parent_names}
        value={'status':'PASS','version':'phase21_saved_S16_immutable_inventory_v1',
            'parent_runtime_source':PARENT_SOURCE,'checkpoints':rows,'parent_public_files':public,
            'actual_fit_source_counts':dict(Counter(row['actual_fit_source'] for row in rows)),
            'ancestor_files_modified':False,'new_model_fits':0}
        write(INVENTORY,value)
        return value


def model_digest(model):
    result=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().cpu().numpy()
        result.update(name.encode());result.update(str(array.dtype).encode())
        result.update(json.dumps(list(array.shape)).encode());result.update(array.tobytes())
    return result.hexdigest()


def assert_public_aggregate(value,key='root'):
    denied={'patient_id','patient_ids','patient_indices','patient_index','sample_id','slide_id',
        'risk','risks','raw_risk','raw_risks','prediction','predictions','gene_ids','gene_symbols',
        'selected_genes','coordinates','raw_maps','raw_tokens'}
    if isinstance(value,dict):
        for name,item in value.items():
            require(str(name).lower() not in denied,'private diagnostic field: '+key+'.'+str(name))
            assert_public_aggregate(item,key+'.'+str(name))
    elif isinstance(value,(list,tuple)):
        require(not any(isinstance(item,(int,float,np.number)) and not isinstance(item,bool) for item in value),
                'raw numeric list in aggregate diagnostic: '+key)
        for item in value:
            assert_public_aggregate(item,key+'[]')
    elif isinstance(value,(np.ndarray,torch.Tensor)):
        raise ValueError('raw diagnostic array: '+key)
    elif isinstance(value,(float,np.floating)):
        require(math.isfinite(float(value)),'nonfinite aggregate diagnostic: '+key)
    else:
        require(value is None or isinstance(value,(str,int,bool,np.integer,np.bool_)),
                'unsupported aggregate diagnostic value: '+key)


def _save_risks(path,values,indices):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**values,validation_or_test_indices=np.asarray(indices,int))
    temporary.replace(path)


def diagnose_partitions(model,ds,diagnostic_function):
    """Reuse a frozen fit-only diagnostic on explicit, disjoint read-only views."""
    original_fit=np.asarray(ds.fit).copy()
    original_valid=np.asarray(ds.valid).copy()
    require(not np.intersect1d(original_fit,original_valid).size,'fit/heldout overlap')
    fit=diagnostic_function(model,ds,chunk_size=32)
    view=copy.copy(ds)
    view.fit=original_valid.copy()
    heldout=diagnostic_function(model,view,chunk_size=32)
    require(fit['status']==heldout['status']=='PASS','saved S16 partition diagnosis failed')
    heldout['heldout_patients']=heldout.pop('fit_patients')
    heldout['heldout_WSIs']=heldout.pop('fit_WSIs')
    fit['partition']='fit'
    heldout['partition']='heldout'
    heldout['scope']=('frozen heldout label-blind diagnostic; original fitted weights and preprocessing fixed; '
        'shallow partition view only, no fitting, epoch selection or endpoint access; '
        'query/WSI/patient moments separately named; undefined ratios omitted with explicit counts')
    require(np.array_equal(ds.fit,original_fit) and np.array_equal(ds.valid,original_valid),
            'read-only partition view mutated original data partition')
    for result in (fit,heldout):
        require(result['additional_gradient_calls']==result['optimizer_steps']==0,
                'read-only partition diagnostic invoked optimization or gradients')
    return {'fit':fit,'heldout':heldout,
        'additional_model_forward_calls':sum(q['additional_model_forward_calls'] for q in (fit,heldout)),
        'additional_operator_calls':sum(q['additional_operator_calls'] for q in (fit,heldout)),
        'additional_gradient_calls':0,'optimizer_steps':0,
        'original_preprocessing_unchanged':True,'original_partition_unchanged':True,
        'heldout_partition_view_only':True}


def audit_one(role,seed,fold,inventory,identity,identity_record,device='cuda'):
    import fcntl
    from src.phase21_relative.diagnostics import diagnose_checkpoint
    name=stem(role,seed,fold)
    private=P/'saved_audits'/(name+'.json')
    public=R/'saved_checkpoints'/(name+'.json')
    risk_path=P/'frozen_saved_risks'/(name+'.npz')
    private.parent.mkdir(parents=True,exist_ok=True)
    with private.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        item=next(q for q in inventory['checkpoints'] if (q['role'],q['seed'],q['fold'])==(role,seed,fold))
        path=checkpoint_path(role,seed,fold)
        if private.exists():
            saved=read(private)
            require(saved['status']=='PASS' and saved['audit_source']==identity
                    and saved['checkpoint_sha256']==item['checkpoint_sha256']
                    and saved['original_inventory_sha256']==digest(INVENTORY)
                    and saved['public_sha256']==digest(public) and saved['private_risks_sha256']==digest(risk_path),
                    'completed saved audit differs from immutable source/artifacts')
            return read(public)
        parent_train.seed_all(seed+100*fold)
        ck,receipt=qualified_checkpoint(path,'S16',seed,fold,role)
        require(digest(path)==item['checkpoint_sha256'] and digest(path.with_suffix('.json'))==item['receipt_sha256']
                and receipt==item['original_receipt'] and ck['source']==item['actual_fit_source'],
                'saved checkpoint changed since original snapshot')
        ds=parent_data.Data(fold,role,seed,device=device)
        # Donors vary by seed, so each seed gets its exact original Data.
        qualify(ck,ds,parent_data.cfg(),ds.inherited_simple)
        model=parent_train.make_model(ds,'S16',ck['anchor_w'])
        model.load_state_dict(ck['state'],strict=True);model.eval()
        require(model.main.dtype==torch.float64,'frozen main coefficients were cast')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before=model_digest(model);values={};begin=time.monotonic()
        with torch.no_grad():
            for mode,field in MODES:
                actual=parent_train.forward(model,ds,ds.valid,mode,training=False).detach().cpu().numpy()
                require(actual.shape==np.asarray(ck[field]).shape and actual.dtype==np.asarray(ck[field]).dtype
                        and np.isfinite(actual).all() and np.array_equal(actual,ck[field]),
                        'saved S16 replay is not bitwise identical: '+mode)
                values[mode]=actual.copy()
        require(np.array_equal(values['matched'],ck['risk']),'S16 primary risk is not matched context')
        require(np.array_equal(values['anchor'],ds.inherited_simple['models']['J0']['risk']),
                'independent frozen anchor identity failed')
        require(np.array_equal(values['constantC'],values['off']),'saved reference Clinical did not disable correction')
        diagnostics=diagnose_partitions(model,ds,diagnose_checkpoint)
        require(model_digest(model)==before,'saved-model diagnostic changed parameters or buffers')
        require(all(parameter.grad is None for parameter in model.parameters()),'read-only diagnosis created parameter gradients')
        require(audit_identity()[0]==identity,'saved-audit source changed during run')
        require(digest(path)==item['checkpoint_sha256'] and digest(path.with_suffix('.json'))==item['receipt_sha256'],
                'ancestor checkpoint or receipt modified during diagnosis')
        record={'status':'PASS','audit_source':identity,'parent_runtime_source':PARENT_SOURCE,
            'actual_fit_source':ck['source'],'arm':'S16','role':role,'seed':seed,'fold':fold,
            'checkpoint_sha256':item['checkpoint_sha256'],'original_receipt_sha256':item['receipt_sha256'],
            'selected_epoch':int(ck['selected_epoch']),'fit_patients':len(ds.fit),'heldout_patients':len(ds.valid),
            'heldout_WSIs':int(np.isin(ds.owner,ds.valid).sum()),'prediction_vectors_replayed':4,
            'matched_primary_bitwise_identical':True,'anchor_bitwise_identical':True,
            'off_bitwise_identical':True,'constantC_bitwise_identical':True,'constantC_equals_off':True,
            'model_values_unchanged':True,'model_gradients_created':False,'main_dtype':'float64',
            'new_training_fits':0,'optimizer_calls':0,'seconds':time.monotonic()-begin,
            'sampling_uses_DSS_outcomes':False,'existing_Data_loads_protected_endpoints':True,
            'diagnostic_routine_reads_endpoints':False,'raw_risks_indices_maps_public':False,
            'diagnostics':diagnostics}
        assert_public_aggregate(record)
        _save_risks(risk_path,values,ds.valid)
        write(public,record)
        write(private,{'status':'PASS','audit_source':identity,'checkpoint_sha256':item['checkpoint_sha256'],
            'original_inventory_sha256':digest(INVENTORY),'public_sha256':digest(public),
            'private_risks_sha256':digest(risk_path),'audit_source_record':identity_record})
        del model,ds,ck
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        return record


def collect():
    import pandas as pd
    inventory=snapshot_inventory(verify_all=True);identity,identity_record=audit_identity()
    rows=[];records=[];calls=0
    def flatten(obj,prefix=''):
        result={}
        for key,value in obj.items():
            name=prefix+str(key)
            if isinstance(value,dict):
                result.update(flatten(value,name+'.'))
            elif not isinstance(value,(list,tuple)):
                result[name]=value
        return result
    for role in ROLES:
        for seed in SEEDS:
            for fold in FOLDS:
                name=stem(role,seed,fold);private=read(P/'saved_audits'/(name+'.json'))
                path=R/'saved_checkpoints'/(name+'.json');item=read(path)
                require(private['status']==item['status']=='PASS' and private['audit_source']==item['audit_source']==identity
                        and private['original_inventory_sha256']==digest(INVENTORY)
                        and private['public_sha256']==digest(path)
                        and private['private_risks_sha256']==digest(P/'frozen_saved_risks'/(name+'.npz')),
                        'saved audit matrix incomplete or changed')
                records.append(item);rows.append(flatten(item))
                calls+=int(item['diagnostics']['additional_model_forward_calls'])
    R.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(R/'SAVED_S16_DIAGNOSTICS.csv',index=False)
    summary={'status':'PASS','audit_source':identity,'parent_runtime_source':PARENT_SOURCE,
        'checkpoint_count':30,'inner_checkpoints':15,'outer_checkpoints':15,
        'actual_fit_source_counts':dict(Counter(q['actual_fit_source'] for q in records)),
        'immutable_original_inventory_sha256':digest(INVENTORY),'prediction_vectors_replayed':120,
        'diagnostic_model_forward_calls':calls,'new_training_fits':0,'optimizer_calls':0,
        'all_heldout_replays_bitwise_identical':True,'ancestor_checkpoints_and_receipts_unchanged':True,
        'raw_risks_maps_indices_public':False,'source_record':identity_record,
        'diagnostic_CSV_sha256':digest(R/'SAVED_S16_DIAGNOSTICS.csv')}
    assert_public_aggregate(summary)
    write(R/'SAVED_S16_AUDIT.json',summary)
    print('PHASE21_SAVED_S16_AUDIT_COMPLETE',identity,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',action='store_true');parser.add_argument('--collect',action='store_true')
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--fold',type=int,choices=FOLDS)
    parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    require(not (args.snapshot and args.collect),'choose snapshot or collect')
    if args.collect:
        collect();return
    require(args.snapshot or (args.role is not None and args.fold is not None),
            'a saved audit job requires --role and --fold')
    inventory=snapshot_inventory(verify_all=args.snapshot)
    if args.snapshot:
        print('PHASE21_SAVED_S16_INVENTORY_PASS',len(inventory['checkpoints']),digest(INVENTORY),flush=True);return
    identity,identity_record=audit_identity()
    for seed in [args.seed] if args.seed else SEEDS:
        value=audit_one(args.role,seed,args.fold,inventory,identity,identity_record,args.device)
        print('PHASE21_SAVED_S16_CELL_PASS',args.role,seed,args.fold,value['actual_fit_source'],flush=True)


if __name__=='__main__':
    main()

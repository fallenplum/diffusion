"""Prelock and run independent R1 patch-budget units with exact core origins.

The default plan/space/cache stages never train a survival model or use a GPU.
Only explicit --stage train invokes the unchanged core neural fit, after a
complete sensitivity lock has been written. Core source/data are not edited.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.bgb_en.data import (native_conch_catalog,patient_quotas,prepare_native_patch_cache,
                             read_patch_cache,sha256)


def require(value,message):
    if not bool(value):raise ValueError(message)


def estimate_cache_disk(rows,budget,feature_dim=768,n_patients=512):
    quota=patient_quotas(rows,int(budget),int(n_patients))
    slides,width=len(rows),int(quota.max())
    shapes={
        'features':((slides,width,int(feature_dim)),np.float32),
        'coords':((slides,width,2),np.int64),'mask':((slides,width),np.bool_),
        'rowids':((slides,width),np.int64),'region_ids':((slides,width),np.int64),
        'owner':((slides,),np.int64),'quota':((slides,),np.int64),
        'coordinate_step':((slides,),np.int64),
        'full_mean':((int(n_patients),int(feature_dim)),np.float64),
        'selected_mean':((int(n_patients),int(feature_dim)),np.float64)}
    array_bytes={key:int(np.prod(shape))*np.dtype(dtype).itemsize for key,(shape,dtype) in shapes.items()}
    # Every array is written once at its final path. The source function has
    # no disk copy of raw H5 or second complete temporary feature cache.
    header_allowance=len(shapes)*512
    metadata_allowance=1024**2
    return dict(slides=slides,feature_dim=int(feature_dim),maximum_slide_quota=width,
                selected_patches=int(quota.sum()),array_bytes=array_bytes,
                final_array_bytes=sum(array_bytes.values()),
                header_allowance_bytes=header_allowance,metadata_allowance_bytes=metadata_allowance,
                peak_additional_disk_bytes=sum(array_bytes.values())+header_allowance+metadata_allowance,
                full_raw_H5_copied=False,second_complete_cache_disk_copy=False)


def check_reserve(free_bytes,required_bytes,reserve_bytes):
    require(int(free_bytes)-int(required_bytes)>=int(reserve_bytes),
            'Cache preparation blocked: exact allocation would violate reserved free disk space')
    return dict(filesystem_free_bytes=int(free_bytes),peak_additional_disk_bytes=int(required_bytes),
                guaranteed_remaining_free_bytes=int(free_bytes)-int(required_bytes),
                required_reserve_bytes=int(reserve_bytes),status='PASS_EXACT_DISK_PREFLIGHT')


def existing_parent(path):
    path=Path(path).resolve()
    while not path.exists():
        require(path.parent!=path,'No existing filesystem parent for private output')
        path=path.parent
    require(path.is_dir(),'Private output parent is not a directory')
    return path


def cache_path(assets,private,encoder,budget):
    if encoder=='UNI2':
        require(int(budget)==128,'UNI2 sensitivity is prelocked only for the existing budget128 cache')
        return Path(assets['patch_uni2'])
    return Path(private)/'assets'/f'conch_regions{int(budget)}_seed270003'


def check_cache(path,encoder,budget,full_hash=True):
    cache=read_patch_cache(path,verify_hash=full_hash);receipt=cache['audit']
    require(receipt['selector']=='REGIONS' and receipt['budget_per_patient']==int(budget)
            and receipt['patients']==512 and receipt.get('region_boundary_reset_required') is True,
            'Sensitivity cache does not match the fixed geometry/quota contract')
    require(receipt['feature_shape'][-1]==(1536 if encoder=='UNI2' else 768),
            'Sensitivity cache encoder dimensions mismatch')
    require(receipt['coordinate_step_counts']==({'512':520} if encoder=='UNI2' else {'1024':519}),
            'Sensitivity cache lacks original native geometry')
    return dict(encoder=encoder,budget=int(budget),READY_sha256=sha256(Path(path)/'READY.json'),
                feature_shape=receipt['feature_shape'],selected_patches=receipt['selected_patches'],
                source_h5_collection_sha256=receipt['source_h5_collection_sha256'],
                array_sha256=receipt['array_sha256'],coordinate_step_counts=receipt['coordinate_step_counts'])


def source_origin(primary):
    origins={};lineage=None
    for partition in range(3):
        stem=Path(primary)/'prepared'/f'outer1_inner{partition}'
        receipt=json.loads(stem.with_suffix('.json').read_text())
        require(receipt.get('status')=='COMPLETE' and sha256(stem.with_suffix('.pt'))==receipt['sha256'],
                'Sensitivity cannot create or replace missing/unverified core preparation')
        if lineage is None:lineage=receipt['lineage']
        require(receipt['lineage']==lineage,'Core preparations have different scientific source origins')
        origins[str(partition)]=dict(prepared_sha256=receipt['sha256'],receipt_sha256=sha256(stem.with_suffix('.json')))
    for relative,digest in lineage['source'].items():
        require((ROOT/relative).is_file() and sha256(ROOT/relative)==digest,'Current core differs from the exact teacher origin')
    return lineage,origins


def config_sources(config_path):
    return {name:sha256(ROOT/name) for name in ('scripts/bgb_en_path_sensitivity.py',
            'configs/bgb_en/path_budget_sensitivity_v1.json')}


def build_lock(assets,private,primary,configuration,configuration_path,full_hash=True):
    require(not (Path(private)/'survival').exists()
            or not list((Path(private)/'survival').rglob('START.json')),
            'Cannot register a new sensitivity lock after its survival fitting started')
    origin,prepared=source_origin(primary)
    cases={}
    for encoder in ('CONCH','UNI2'):
        for budget in configuration[encoder+'_budgets']:
            cases[encoder+'_B'+str(budget)]=check_cache(cache_path(assets,private,encoder,budget),encoder,budget,full_hash)
    original_pool=json.loads((Path(assets['patch_grid'])/'READY.json').read_text())
    for key,case in cases.items():
        if key.startswith('CONCH_'):
            require(case['array_sha256']['full_mean']==original_pool['array_sha256']['full_mean']
                    and case['source_h5_collection_sha256']==original_pool['source_h5_collection_sha256'],
                    'Budget changed the full-source P0 pooling reference')
    p0={}
    for partition in range(3):
        path=Path(primary)/'cells'/f'R1_outer1_inner{partition}_P0_seed31'/'COMPLETE.json'
        require(path.is_file(),'Exact completed full-mean P0 reference is required; no sensitivity refit')
        receipt=json.loads(path.read_text())
        require(receipt.get('status')=='COMPLETE' and receipt.get('arm')=='P0'
                and receipt.get('epochs')==0 and receipt.get('lineage')==origin,
                'P0 origin or training protocol differs')
        require(sha256(path.parent/'model.pt')==receipt['artifact_sha256'],'Completed P0 reference artifact changed')
        p0[str(partition)]=dict(receipt_sha256=sha256(path),artifact_sha256=receipt['artifact_sha256'])
    return dict(schema='bgb_en_path_sensitivity_lock_v1',status='LOCKED_BEFORE_SENSITIVITY_SURVIVAL_FITS',
        configuration=configuration,configuration_sha256=sha256(configuration_path),
        sensitivity_sources=config_sources(configuration_path),shared_teacher_origin_lineage=origin,
        shared_prepared_source=prepared,cases=cases,planned_new_neural_fits=36,
        selector_or_budget_chosen_from_survival_results=False,
        survival_fits_started=False,P0_new_fits=0,P0_existing_reference=p0,outer_test_evaluated=False,
        scope='predefined_inner_patch_budget_and_nonpure_foundation_sensitivity')


def train_case(args,assets,configuration,lock):
    """Explicit later execution only; retain core architecture arm P1/P3 exactly."""
    import torch
    from scripts import bgb_en_run as driver
    key=args.encoder+'_B'+str(args.budget)
    require(key in lock['cases'] and args.arm in configuration['arms']
            and args.partition in configuration['partitions'] and args.seed in configuration['seeds'],
            'Sensitivity fitting unit is outside its prelocked matrix')
    require(lock['sensitivity_sources']==config_sources(args.sensitivity_config),
            'Sensitivity wrapper/config source changed after locking')
    require(sha256(args.sensitivity_config)==lock['configuration_sha256'],'Sensitivity configuration changed')
    selected=cache_path(assets,args.private,args.encoder,args.budget)
    require(sha256(selected/'READY.json')==lock['cases'][key]['READY_sha256'],'Selected sensitivity cache changed')
    original_private=args.private;args.stage='R1';args.outer_fold=1
    args.config=ROOT/'configs/bgb_en/prototype_v1.json'
    core_assets,core_config,raw,member,core_lineage=driver.load_inputs(args)
    require(core_lineage==lock['shared_teacher_origin_lineage'],'Sensitivity teacher source would be relabelled')
    core_prepare=driver.prepare
    def reuse_prepared(inner_args,raw,member,lineage):
        parent_args=copy.copy(inner_args);parent_args.private=args.primary_private
        path,receipt=driver.prep_paths(parent_args)
        expected=lock['shared_prepared_source'][str(args.partition)]
        require(path.is_file() and receipt.is_file() and sha256(path)==expected['prepared_sha256']
                and sha256(receipt)==expected['receipt_sha256'],'Original core preparation changed or disappeared')
        return core_prepare(parent_args,raw,member,core_lineage)
    expected=copy.deepcopy(core_lineage)
    expected['path_budget_sensitivity']=dict(scenario=key,base_arm=args.arm,
        lock_sha256=sha256(original_private/'PATH_SENSITIVITY_LOCK.json'),
        selected_cache_READY_sha256=lock['cases'][key]['READY_sha256'],
        sources=lock['sensitivity_sources'],shared_teacher_origin_lineage=core_lineage)
    # Every treatment uses the ordinary base P1/P3 arm. Replacing the cache is
    # explicit in its added lineage; the teacher lineage remains byte-identical.
    core_assets=copy.deepcopy(core_assets);core_assets['patch_regions']=str(selected)
    args.private=original_private/'survival'/key
    torch.set_num_threads(2)
    driver.prepare=reuse_prepared
    try:driver.run_unit(args,core_assets,core_config,raw,member,expected)
    finally:driver.prepare=core_prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',required=True,type=Path);parser.add_argument('--private',required=True,type=Path)
    parser.add_argument('--primary-private',type=Path)
    parser.add_argument('--sensitivity-config',type=Path,default=ROOT/'configs/bgb_en/path_budget_sensitivity_v1.json')
    parser.add_argument('--stage',choices=('space','prepare','lock','plan','train'),default='plan')
    parser.add_argument('--encoder',choices=('CONCH','UNI2'),default='CONCH')
    parser.add_argument('--budget',type=int,choices=(128,1024),default=128)
    parser.add_argument('--arm',choices=('P1','P3'),default='P1')
    parser.add_argument('--partition',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--seed',type=int,choices=(31,47),default=31)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--device',default='cuda')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();configuration=json.loads(args.sensitivity_config.read_text())
    require(configuration['start_training_automatically'] is False,'Sensitivity requires an explicit training stage')
    assets=json.loads(args.assets.read_text());target=cache_path(assets,args.private,args.encoder,args.budget)
    if args.dry_run:args.stage='plan'
    if args.stage in ('space','prepare'):
        if target.exists() and (target/'READY.json').is_file():
            info=check_cache(target,args.encoder,args.budget,full_hash=args.stage=='prepare')
            print(json.dumps(dict(status='VERIFIED_EXISTING_CACHE_NO_NEW_FIT',cache=info)),flush=True);return
        require(args.encoder=='CONCH','Only the existing UNI2-128 cache is registered')
        rows=native_conch_catalog(assets['conch_inventory'],root_override=assets.get('conch_root'))
        estimate=estimate_cache_disk(rows,args.budget)
        reserve=configuration['reserve_system_free_bytes_after_cache']
        space=check_reserve(shutil.disk_usage(existing_parent(args.private)).free,
                            estimate['peak_additional_disk_bytes'],reserve)
        print(json.dumps(dict(encoder=args.encoder,budget=args.budget,estimate=estimate,space=space,
                              source_data_module_sha256=sha256(ROOT/'src/bgb_en/data.py'),survival_fits=0,GPU_used=False)),flush=True)
        if args.stage=='prepare':
            receipt=prepare_native_patch_cache(assets['conch_inventory'],target,budget=args.budget,
                      selector='REGIONS',seed=270003,root_override=assets.get('conch_root'))
            require(shutil.disk_usage(existing_parent(args.private)).free>=reserve,'Disk reserve changed during cache preparation')
            print(json.dumps(dict(status='COMPLETE_GEOMETRY_CACHE_ZERO_SURVIVAL_FITS',
                                  READY_sha256=sha256(target/'READY.json'),feature_shape=receipt['feature_shape'])),flush=True)
        return
    lock_path=args.private/'PATH_SENSITIVITY_LOCK.json'
    if args.stage=='lock':
        require(args.primary_private is not None,'Exact core-preparation origin is required')
        value=build_lock(assets,args.private,args.primary_private,configuration,args.sensitivity_config)
        if lock_path.exists():require(json.loads(lock_path.read_text())==value,'Existing sensitivity lock differs')
        else:
            args.private.mkdir(parents=True,exist_ok=True)
            lock_path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=value['status'],planned_new_neural_fits=36,
                              lock_sha256=sha256(lock_path),survival_fits_started=False)),flush=True);return
    if args.stage=='plan':
        print(json.dumps(dict(status='LOCKED_READY_NOT_STARTED' if lock_path.is_file() else 'PLANNED_NOT_LOCKED',
                              CONCH_budgets=[128,1024],UNI2_budgets=[128],arms=['P1','P3'],
                              planned_new_neural_fits=36,automatic_GPU_training=False,P0_new_fits=0)),flush=True);return
    require(args.primary_private is not None and lock_path.is_file(),'Training requires exact origin and a prior complete sensitivity lock')
    lock=json.loads(lock_path.read_text())
    require(lock.get('status')=='LOCKED_BEFORE_SENSITIVITY_SURVIVAL_FITS' and lock['configuration']==configuration,
            'Sensitivity training has no matching prefit lock')
    train_case(args,assets,configuration,lock)


if __name__=='__main__':main()

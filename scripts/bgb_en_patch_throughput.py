"""Contract-derived logical valid-patch admissions per real completed unit.

Prepare-indices CPU-deserializes existing mixed prepared bundles once,
extracting only train/valid into a private cache. Report reads those small
indices plus cache mask/owner and JSON receipts, never features or coordinates.
No image, model construction, forward pass or optimizer is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.bgb_en_execution_ledger import validate_roots,enumerate_records,sha,read,require
from scripts.bgb_en_report_aux import write_csv

CONTROLS=('wrong_query','mean_query','whole_omics','attention_permutation','interaction_off','direct_off')
CALL_SOURCES=('src/bgb_en/training.py','src/bgb_en/survival.py','scripts/bgb_en_run.py',
              'scripts/bgb_en_final_neural.py','scripts/bgb_en_path_sensitivity.py','scripts/bgb_en_aux_run.py',
              'scripts/bgb_en_omics_sensitivity.py','scripts/bgb_en_budget_retrain.py',
              'scripts/bgb_en_matched_support.py','scripts/bgb_en_matched_support_io_plan.py',
              'scripts/bgb_en_precompute_nested_prepare.py')


def array_hash(value):
    x=np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def canonical_folds(assets,registry):
    """Access only the folds member; do not deserialize IDs or any endpoint."""
    path=Path(assets['cohort'])
    require(sha(path)==registry['hashes']['cohort_sha256'],'Canonical cohort file hash differs')
    with np.load(path,allow_pickle=False) as archive:
        folds=np.array(archive['folds'],copy=True)
    require(folds.shape==(registry['patients'],) and folds.dtype.kind in 'iu'
            and set(folds)=={1,2,3,4,5},'Canonical outer fold registry differs')
    return folds,dict(cohort_sha256=registry['hashes']['cohort_sha256'],
        folds_array_sha256=array_hash(folds),patient_order_sha256=registry['hashes']['patient_order_sha256'],
        cohort_members_accessed=['folds'],patient_names_accessed=False,endpoint_or_measurement_members_accessed=False)


def canonical_outer_indices(folds,fold,contract=None):
    require(fold in (1,2,3,4,5),'Unsupported canonical outer fold')
    train,test=checked_indices(np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold),len(folds))
    if contract is not None:
        require(contract['train_hash']==array_hash(train) and contract['test_hash']==array_hash(test),
                'Final neural contract indices differ from canonical folds')
    return train,test


def checked_indices(train,valid,n_patients):
    train,valid=np.asarray(train),np.asarray(valid)
    require(train.ndim==valid.ndim==1 and train.dtype.kind in 'iu' and valid.dtype.kind in 'iu'
            and len(train)>0 and len(valid)>0,'Invalid private split-index arrays')
    require(len(np.unique(train))==len(train) and len(np.unique(valid))==len(valid)
            and not np.intersect1d(train,valid).size and min(train.min(),valid.min())>=0
            and max(train.max(),valid.max())<n_patients,'Private split indices overlap or exceed patient registry')
    return train,valid


def index_cache(bundle,receipt_path,destination,n_patients,patient_order_sha256,create=False):
    """Only the explicit preparation stage loads an existing mixed bundle."""
    receipt=read(receipt_path);identity=dict(source_bundle_sha256=receipt['sha256'],
        source_prepared_receipt_sha256=sha(receipt_path),patient_order_sha256=patient_order_sha256,
        source_prepared_lineage_sha256=hashlib.sha256(json.dumps(receipt.get('lineage'),sort_keys=True).encode()).hexdigest())
    metadata=destination.with_suffix('.json')
    if metadata.is_file():
        saved=read(metadata)
        require(all(saved.get(key)==value for key,value in identity.items()) and sha(destination)==saved['index_payload_sha256'],
                'Private index cache changed or has a different prepared origin')
        with np.load(destination,allow_pickle=False) as archive:
            require(set(archive.files)=={'train','valid'},'Index-only payload contains unexpected arrays')
            train,valid=checked_indices(archive['train'],archive['valid'],n_patients)
        return train,valid,saved
    require(create,'Private index cache absent; run --stage prepare-indices first')
    require(sha(bundle)==identity['source_bundle_sha256'],'Existing prepared bundle differs from its completed receipt')
    import torch
    # This does deserialize the bundle's other mixed arrays. They are not
    # inspected, analyzed or exported; the object is immediately discarded.
    existing=torch.load(bundle,map_location='cpu',weights_only=False)
    train,valid=checked_indices(np.array(existing['train'],copy=True),np.array(existing['valid'],copy=True),n_patients)
    require(existing.get('fit_hash')==array_hash(train) and existing.get('valid_hash')==array_hash(valid),
            'Saved prepared index hashes differ')
    del existing
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,train=train,valid=valid)
    saved=dict(schema='bgb_en_private_index_only_cache_v1',**identity,index_payload_sha256=sha(destination),
        train_index_sha256=array_hash(train),valid_index_sha256=array_hash(valid),train_n=len(train),valid_n=len(valid),
        extraction='CPU deserialize existing mixed prepared bundle once; only train/valid extracted; no measurement analysis or export')
    metadata.write_text(json.dumps(saved,indent=2)+'\n')
    return train,valid,saved


def patch_counts(path,n_patients,conch_inventory=None):
    """Deliberately avoid read_patch_cache: touch exactly owner.npy/mask.npy."""
    path=Path(path);receipt=read(path/'READY.json')
    require(receipt.get('schema')=='bgb_en_native_patch_cache_v1' and receipt.get('status')=='COMPLETE'
            and receipt.get('patients')==n_patients,'Unqualified patch-cache receipt')
    for name in ('mask','owner'):
        require(sha(path/(name+'.npy'))==receipt['array_sha256'][name],'Mask or owner array hash differs from READY')
    mask=np.load(path/'mask.npy',mmap_mode='r',allow_pickle=False)
    owner=np.load(path/'owner.npy',mmap_mode='r',allow_pickle=False)
    require(mask.dtype==np.bool_ and mask.ndim==2 and owner.dtype.kind in 'iu'
            and owner.shape==(len(mask),) and tuple(mask.shape)==tuple(receipt['feature_shape'][:2]),
            'Mask/owner dimensions differ from selected-cache contract')
    require(owner.min()>=0 and owner.max()<n_patients and len(np.unique(owner))==n_patients,
            'Owner mapping does not cover the patient registry')
    if 'canonical patient index' in receipt.get('source_order',''):
        mapping_evidence='explicit READY source_order and original qualified encoder source'
    else:
        # CONCH v1 READY predates the source_order field. Verify the actual
        # private metadata mapping, without opening any source H5 or ID export.
        require(conch_inventory is not None and sha(conch_inventory)==receipt.get('source_inventory_sha256'),
                'Legacy CONCH owner order needs its hash-bound original inventory')
        catalog=read(conch_inventory)
        require(not catalog.get('errors') and not catalog.get('missing_conch'),
                'Historical CONCH source inventory is incomplete')
        ordered=sorted(catalog['chosen_conch'],key=lambda row:(int(row['patient_index']),str(row['stem'])))
        expected=np.asarray([int(row['patient_index']) for row in ordered],dtype=np.int64)
        require(np.array_equal(owner,expected),'Actual CONCH owners differ from canonical source inventory order')
        mapping_evidence='actual owner array equals hash-bound inventory patient_index sorted by patient and slide stem'
    result=np.zeros(n_patients,np.int64);np.add.at(result,owner,np.asarray(mask.sum(1),np.int64))
    require(int(result.sum())==receipt['selected_patches'] and (result>0).all(),'Actual valid-mask counts differ from READY')
    return result,dict(READY_sha256=sha(path/'READY.json'),mask_array_sha256=receipt['array_sha256']['mask'],
        owner_array_sha256=receipt['array_sha256']['owner'],source_H5_collection_sha256=receipt['source_h5_collection_sha256'],
        selected_valid_nodes=int(result.sum()),patient_owner_mapping='canonical patient order',
        budget_per_patient=int(receipt['budget_per_patient']),
        visual_feature_dim=int(receipt['feature_shape'][-1]),cache_selector=receipt['selector'],
        owner_order_evidence=mapping_evidence,
        source_inventory_sha256=receipt.get('source_inventory_sha256'),
        features_array_opened=False,coordinates_array_opened=False,raw_H5_opened=False)


def admissions(schedule,epochs,checkpoints,train_nodes,valid_nodes,train_n,valid_n,batch_size=8):
    require(epochs>0 and checkpoints>=0 and train_nodes>0 and valid_nodes>0,'Invalid completed visual schedule')
    if schedule=='INNER_R1':train_passes=2*epochs+checkpoints;valid_passes=checkpoints
    elif schedule=='INNER_R3':train_passes=2*epochs+checkpoints;valid_passes=checkpoints+6
    elif schedule=='OUTER_R3_FINAL':train_passes=2*epochs;valid_passes=1+6
    else:raise ValueError('Unsupported visual forward schedule')
    train_batches=(train_n+batch_size-1)//batch_size;valid_batches=(valid_n+batch_size-1)//batch_size
    return dict(train_complete_patient_passes=train_passes,validation_complete_patient_passes=valid_passes,
        actual_visual_forward_invocations=train_passes*train_batches+valid_passes*valid_batches,
        train_logical_valid_node_admissions=int(train_nodes*train_passes),
        validation_logical_valid_node_admissions=int(valid_nodes*valid_passes),
        logical_valid_node_admissions=int(train_nodes*train_passes+valid_nodes*valid_passes))


def index_identity(core,cache,fold,part,registry,create=False,outer_private=None,folds=None):
    require(fold in (1,2,3,4,5) and part in (0,1,2),'Unsupported prepared partition identity')
    if fold==1:prepared_root=Path(core)
    else:
        require(outer_private is not None,'Additional outer prepared index root must be explicit')
        prepared_root=Path(outer_private)/'nested_inner'
    stem=f'outer{fold}_inner{part}'
    source=read(prepared_root/'prepared'/(stem+'.json'))
    require(source.get('status')=='COMPLETE' and source['lineage']['cohort_sha256']==registry['hashes']['cohort_sha256'],
            'Prepared split is not bound to the audited canonical patient registry')
    result=index_cache(prepared_root/'prepared'/(stem+'.pt'),prepared_root/'prepared'/(stem+'.json'),
                       Path(cache)/(stem+'.npz'),registry['patients'],registry['hashes']['patient_order_sha256'],create)
    if folds is not None:
        train,valid,_=result;outer_train,_=canonical_outer_indices(folds,fold)
        require(np.array_equal(np.sort(np.r_[train,valid]),outer_train),
                'Prepared inner train/valid does not exactly partition its canonical outer train')
    return result


def qualify_outer_indices(args,assets,registry):
    require(args.outer_private is not None,'R4 index preparation needs explicit --outer-private')
    folds,audit=canonical_folds(assets,registry);rows=[]
    expected=read(Path(args.core_private)/'prepared'/'outer1_inner0.json')['lineage']
    expected_hash=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    for fold in (2,3,4,5):
        for part in (0,1,2):
            train,valid,identity=index_identity(args.core_private,args.index_cache,fold,part,registry,
                create=True,outer_private=args.outer_private,folds=folds)
            require(identity['source_prepared_lineage_sha256']==expected_hash,
                    'R4 prepared origin does not match the frozen common scientific lineage')
            rows.append(dict(outer_fold=fold,inner_partition=part,train_n=len(train),validation_n=len(valid),
                prepared_bundle_sha256=identity['source_bundle_sha256'],
                prepared_receipt_sha256=identity['source_prepared_receipt_sha256'],
                private_index_payload_sha256=identity['index_payload_sha256'],
                inner_union_equals_canonical_outer_train=True,outer_test_rows_in_inner=False))
    return dict(schema='bgb_en_R4_patch_index_qualification_v1',status='PASS_12_SAVED_INDEX_CACHES_ONLY',
        prepared_bundles_verified=12,partitions=rows,canonical_fold_access=audit,
        extraction='CPU deserialize existing measurement-containing prepared bundle once, extract only train/valid; other arrays deserialized but not analyzed or exported',
        qualifies_future_completed_unit_counting_only=True,neural_throughput_created_from_plan=False,
        new_optimizer_updates=0,model_forward_calls=0,GPU_used=False,
        upstream_168_optimizations_already_in_original_plan_not_repeated=True)


def choose_cache(row,assets,path_private):
    lineage=row['lineage'];arm=row['arm']
    treatment=lineage.get('path_budget_sensitivity')
    if treatment:
        scenario=treatment['scenario']
        if scenario=='UNI2_B128':path=Path(assets['patch_uni2'])
        elif scenario in ('CONCH_B128','CONCH_B1024'):
            path=Path(path_private)/'assets'/('conch_regions'+scenario.split('_B')[1]+'_seed270003')
        else:raise ValueError('Unknown prelocked pathology-budget cache')
        expected=treatment['selected_cache_READY_sha256']
    else:
        key='patch_uni2' if arm.endswith('_UNI2') else 'patch_grid' if arm=='P7_GRID' else 'patch_rand' if arm=='P7_RAND' else 'patch_diverse' if arm=='P7_DIVERSE' else 'patch_regions'
        path=Path(assets[key]);expected=lineage['UNI2_cache_receipt_sha256'] if key=='patch_uni2' else lineage['patch_cache_receipts'][key]
    require(sha(path/'READY.json')==expected,'Visual schedule cache differs from trained lineage')
    return path,expected


def base_result(record):
    keys=('root','role','category','stage','arm','variant','seed','inner_partition','outer_fold','fold')
    return {key:record[key] for key in keys if key in record}


def original_lineage(receipt):
    lineage=receipt.get('lineage',receipt.get('contract',{}).get('core_lineage',{}))
    if 'matched_support_auxiliary' in lineage:
        lineage={key:value for key,value in lineage.items() if key!='matched_support_auxiliary'}
    for key in ('shared_teacher_origin_lineage','core_lineage','core_origin'):
        if key in lineage:return lineage[key]
    return lineage.get('path_budget_sensitivity',{}).get('shared_teacher_origin_lineage',lineage)


def classify_nonvisual(record,root_kind=None):
    if record['status']!='COMPLETE':return 'NA_FAILED_OR_UNFINISHED_EPOCH_UNKNOWN'
    if record['category']=='REFERENCE':return 'NA_DETERMINISTIC_REFERENCE_NO_VISUAL_FORWARD'
    if record['category']=='EXTERNAL_SOURCE':return 'NOT_APPLICABLE_ST_SPOT_FEATURE_UNITS_DIFFER_FROM_WSI_PATCHES'
    if record['category'] in ('COX','OUTER_ANCHOR_BUNDLE'):return 'SUPPORTED_ZERO_VISUAL_WORK'
    if record.get('stage') in ('R2','R2_SENSITIVITY') or root_kind=='budget':return 'SUPPORTED_ZERO_VISUAL_WORK'
    return None


def protected_snapshot(core):
    receipt=read(Path(core)/'prepared'/'outer1_inner0.json')
    protected={}
    for name,expected in receipt['lineage']['source'].items():
        actual=sha(ROOT/name);require(actual==expected,'Frozen core source/config changed before throughput audit')
        protected[name]=actual
    for name in ('reports/bgb_en/INNER_REPORT_FOR_SELECTION.json','reports/bgb_en/OUTER_SELECTION_LOCK.json',
                 'scripts/bgb_en_patch_throughput.py','scripts/bgb_en_final_neural.py',
                 'scripts/bgb_en_nested.py'):
        protected[name]=sha(ROOT/name) if (ROOT/name).is_file() else None
    return protected


def unchanged_guard(before,after):
    require(before==after,'Throughput audit source or protected selection files changed during read-only execution')
    return dict(status='PASS_CORE_SOURCE_AND_SELECTION_UNCHANGED',before=before,after=after)


def generate(args,configuration,assets,registry):
    roots=validate_roots(configuration);records=enumerate_records(roots);rows=[];caches={};prepared={};sources={}
    folds,fold_audit=canonical_folds(assets,registry) if args.outer_private is not None else (None,None)
    for record in records:
        if record['category']=='UPSTREAM_ANCHOR_BUNDLE':continue
        out=dict(**base_result(record),receipt_sha256=record['receipt_sha256'],
            elapsed_seconds=record['unit_elapsed_seconds'],logical_valid_node_admissions=None,
            logical_valid_node_admissions_per_second=None,
            elapsed_scope='complete unit elapsed: includes setup/post-fit/report/serialization according to its driver')
        receipt=None
        if record['status']=='COMPLETE' and record['category']=='NEURAL':
            receipt=read(Path(record['_directory'])/'COMPLETE.json');origin=original_lineage(receipt)
            for name in ('src/bgb_en/training.py','src/bgb_en/survival.py','scripts/bgb_en_run.py'):
                require(origin['source'][name]==sha(ROOT/name),'Frozen visual forward-count source differs')
            source_id=hashlib.sha256(json.dumps(origin['source'],sort_keys=True).encode()).hexdigest()
            sources[source_id]=origin['source'];out['frozen_training_source_group_sha256']=source_id
        nonvisual=classify_nonvisual(record,next(root['kind'] for root in roots if root['name']==record['root']))
        if nonvisual:
            out['throughput_status']=nonvisual
            if nonvisual=='SUPPORTED_ZERO_VISUAL_WORK':
                out['logical_valid_node_admissions']=0
                out['logical_valid_node_admissions_per_second']=0. if out['elapsed_seconds'] and out['elapsed_seconds']>0 else None
            rows.append(out);continue
        path=Path(record['_directory'])/'COMPLETE.json';receipt=receipt or read(path)
        if receipt.get('status')=='COMPLETE_OUTER_NEURAL':
            if folds is None:
                out.update(throughput_status='NOT_SUPPORTED_OUTER_PRIVATE_INDEX_SOURCE',
                    reason='Explicit outer-private/folds qualification not supplied; no patch count guessed')
                rows.append(out);continue
            contract=receipt['contract'];fold=receipt['fold'];origin=original_lineage(receipt)
            require(receipt['epochs']==80 and receipt.get('outer_test_read_for_selection') is False
                    and set(receipt.get('frozen_controls',{}))==set(CONTROLS),
                    'Final neural schedule or sealed-test contract differs')
            require(contract['additional_sources']['scripts/bgb_en_final_neural.py']==sha(ROOT/'scripts/bgb_en_final_neural.py')
                    and contract['selection_lock_sha256']==sha(ROOT/'reports/bgb_en/OUTER_SELECTION_LOCK.json')
                    and origin['cohort_sha256']==registry['hashes']['cohort_sha256'],
                    'Final neural source or canonical cohort identity differs')
            train,test=canonical_outer_indices(folds,fold,contract)
            cache_path,ready=choose_cache({'lineage':origin,'arm':receipt['selected_arm']},assets,args.path_private)
            if ready not in caches:caches[ready]=patch_counts(cache_path,registry['patients'],assets.get('conch_inventory'))
            count,audit=caches[ready];work=admissions('OUTER_R3_FINAL',receipt['epochs'],0,
                int(count[train].sum()),int(count[test].sum()),len(train),len(test),8)
            elapsed=float(receipt['elapsed_seconds']);require(np.isfinite(elapsed) and elapsed>0,'Final neural elapsed is invalid')
            out.update(throughput_status='SUPPORTED_CONTRACT_DERIVED_LOGICAL_VALID_NODES',
                stage='R4_FINAL',arm=receipt['selected_arm'],
                epochs=receipt['epochs'],checkpoint_evaluations=0,frozen_visual_control_evaluations=6,
                train_n=len(train),validation_n=len(test),**work,
                logical_valid_node_admissions_per_second=work['logical_valid_node_admissions']/elapsed,
                cache_READY_sha256=ready,canonical_folds_sha256=fold_audit['folds_array_sha256'],
                **{key:audit[key] for key in ('budget_per_patient','visual_feature_dim','cache_selector')},
                patient_order_sha256=fold_audit['patient_order_sha256'],
                final_train_index_sha256=contract['train_hash'],final_test_index_sha256=contract['test_hash'],
                forward_source_sha256=sha(ROOT/'scripts/bgb_en_final_neural.py'))
            rows.append(out);continue
        stage=receipt.get('stage');require(stage in ('R1','R3'),'Unexpected nonzero visual stage')
        epochs=receipt['epochs'];checkpoints=receipt.get('curves',[])
        require(epochs==80 and sorted(point['epoch'] for point in checkpoints)==[20,40,80],
                'Visual cell did not use the frozen three-snapshot 80-epoch schedule')
        if stage=='R3':require(set(receipt.get('frozen_controls',{}))==set(CONTROLS),'R3 post-fit visual interventions differ')
        lineage=receipt['lineage'];origin=original_lineage(receipt)
        part,fold=receipt['inner_partition'],receipt['outer_fold'];key=(fold,part)
        if key not in prepared:prepared[key]=index_identity(args.core_private,args.index_cache,fold,part,registry,
            outer_private=args.outer_private,folds=folds)
        train,valid,identity=prepared[key]
        require(receipt['train_n']==len(train) and receipt['valid_n']==len(valid)
                and lineage['cohort_sha256']==registry['hashes']['cohort_sha256']
                and hashlib.sha256(json.dumps(origin,sort_keys=True).encode()).hexdigest()==identity['source_prepared_lineage_sha256'],
                'Completed visual cell does not match original prepared population')
        cache_path,ready=choose_cache(receipt,assets,args.path_private)
        if ready not in caches:caches[ready]=patch_counts(cache_path,registry['patients'],assets.get('conch_inventory'))
        count,audit=caches[ready];train_nodes=int(count[train].sum());valid_nodes=int(count[valid].sum())
        work=admissions('INNER_'+stage,epochs,len(checkpoints),train_nodes,valid_nodes,len(train),len(valid),8)
        elapsed=float(receipt['elapsed_seconds']);require(np.isfinite(elapsed) and elapsed>0,'Completed unit elapsed is invalid')
        source_id=hashlib.sha256(json.dumps(origin['source'],sort_keys=True).encode()).hexdigest();sources[source_id]=origin['source']
        out.update(throughput_status='SUPPORTED_CONTRACT_DERIVED_LOGICAL_VALID_NODES',epochs=epochs,
            checkpoint_evaluations=len(checkpoints),frozen_visual_control_evaluations=6 if stage=='R3' else 0,
            train_valid_nodes_per_pass=train_nodes,validation_valid_nodes_per_pass=valid_nodes,
            train_n=len(train),validation_n=len(valid),**work,
            logical_valid_node_admissions_per_second=work['logical_valid_node_admissions']/elapsed,
            cache_READY_sha256=ready,prepared_bundle_sha256=identity['source_bundle_sha256'],
            **{key:audit[key] for key in ('budget_per_patient','visual_feature_dim','cache_selector')},
            prepared_receipt_sha256=identity['source_prepared_receipt_sha256'],
            private_index_payload_sha256=identity['index_payload_sha256'],
            patient_order_sha256=identity['patient_order_sha256'],frozen_training_source_group_sha256=source_id)
        rows.append(out)
    coverage=[]
    for root in roots:
        selected=[record for record in records if record['root']==root['name'] and record['category']!='UPSTREAM_ANCHOR_BUNDLE']
        complete=sum(record['status']=='COMPLETE' for record in selected)
        coverage.append(dict(root=root['name'],role=root['role'],expected_units=root.get('expected_units'),
            completed_receipts=complete,failed_or_unfinished_NA_attempts=len(selected)-complete,
            status='NOT_RUN' if not selected else 'COMPLETE_EXPECTED_RECEIPTS' if root.get('expected_units')==complete else 'RUNNING_OR_PARTIAL'))
    result=dict(schema='bgb_en_logical_patch_throughput_audit_v1',status='RUNNING_AVAILABLE_COMPLETED_UNIT_CONTRACTS',
        definition='sum actual valid mask nodes admitted to each visual forward / original completed unit elapsed_seconds',
        metric_is_contract_derived=True,not_kernel_time_or_FLOPs=True,not_unique_tissue_coverage=True,
        four_scan_directions_not_counted_as_four_raw_patch_admissions=True,
        valid_mask_nodes_exclude_padding_but_actual_hardware_may_compute_padding=True,
        patient_node_repetitions_across_epochs_are_counted=True,
        rows=rows,root_coverage=coverage,available_receipt_or_attempt_units=len(rows),
        positive_visual_throughput_units=sum(row['throughput_status']=='SUPPORTED_CONTRACT_DERIVED_LOGICAL_VALID_NODES' for row in rows),
        zero_visual_units=sum(row['throughput_status']=='SUPPORTED_ZERO_VISUAL_WORK' for row in rows),
        NA_or_unsupported_units=sum(row['logical_valid_node_admissions'] is None for row in rows),
        cache_audits=list(audit for count,audit in caches.values()),training_source_groups=sources,
        canonical_fold_access=fold_audit,
        forward_contract_sources={name:sha(ROOT/name) for name in CALL_SOURCES},
        index_extraction_boundary='Read original three and explicitly authorized twelve R4 mixed measurement-containing prepared bundles on CPU once to extract only train/valid; other arrays deserialized but not analyzed/exported; reporting reuses index-only cache. Final outer indices access canonical folds only.',
        source_ST_spot_throughput_not_in_WSI_patch_table=True,model_forward_calls=0,new_optimizer_updates=0,
        GPU_used=False,raw_images_or_H5_opened=False,features_npy_opened=False,coordinates_opened=False,
        patient_indices_or_per_patient_node_counts_exported=False,script_sha256=sha(__file__))
    return result


def write_reports(result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'PATCH_THROUGHPUT_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(output/'PATCH_THROUGHPUT_AUDIT.csv',result['rows'],
              ('root','role','throughput_status','logical_valid_node_admissions','logical_valid_node_admissions_per_second','elapsed_seconds'))
    lines=['# 逻辑有效Patch admissions吞吐诊断','',
        '**状态：可用完整收据快照；不是所有计划已完成的证明。**','',
        f"已覆盖 {result['available_receipt_or_attempt_units']} 个收据/尝试单元；正视觉工作 {result['positive_visual_throughput_units']}，明确零视觉工作 {result['zero_visual_units']}，NA或未支持 {result['NA_or_unsupported_units']}。",'',
        '定义：真实mask有效节点按每次视觉前向重复准入的总数，除以对应COMPLETE实测elapsed_seconds。它是冻结调用合同推导的逻辑valid-node吞吐，不是硬件kernel速度、FLOP、去重组织覆盖或实际padding计算量。',
        '', '## 已核调用合同','',
        '- 内层R1：每轮两遍训练风险缓存，共2×80次train；20/40/80各一次train及valid，故163次train、3次valid。',
        '- 内层R3：相同训练/快照调用，再增加6种冻结valid干预，故163次train、9次valid。training_query_mean仅经过omics，不增加视觉工作。',
        '- R2没有path分支，视觉准入数为0；即使曾分配PatchBank也不计成视觉前向。P0/F0/F1/O0等确定性引用为NA，不制造高吞吐排名。',
        '- final_neural源码为160次train加最终预测与6控制共7次test，无中途train/test预测。只有实际COMPLETE且train/test hash与canonical folds一致后才计算吞吐；尚未训练的计划不产生数值。显式缺少outer-private时仍标NOT_SUPPORTED。',
        '- 外部ST source操作单位是spot特征，并且训练timer先于post-fit评估停止；本WSI Patch主表标NOT_APPLICABLE，不混入不同单位的吞吐。',
        '', '## 数据与计时范围','',
        '只读缓存READY、mask.npy、owner.npy及小型私有train/valid索引缓存。首次明确读取原三个以及获授权的R4十二个含测量的既有prepared bundle，经CPU torch.load提取索引；其他数组确实随bundle反序列化，但不分析、不导出。随后报告不再读取bundle。最终outer train/test只访问canonical cohort的folds成员，完全不读取患者名称、终点或风险。没有读取features.npy、原始H5、图像或坐标。',
        '', '每患者有效节点按owner累加全部原slides，与PatchBank实际输入一致。来源READY、mask/owner、prepared及patient-order均绑定哈希，不输出索引或逐患者节点数。不同预算、选区和UNI2/CONCH分别使用实际训练cache。',
        '', '单位elapsed依训练driver可能包含bank设置、训练、快照/冻结干预及序列化等开销。因此比值是完整单元口径，不能跨不同计时范围独占解释为算子效率。失败或未完成epoch未知时保留NA，不拿计划80轮补算。',
        '', '优势：有效mask、真实调用合同和实测单元墙钟可逐项追溯；风险：逻辑节点次数不是profiling trace，空间扫描和padding的真实计算量可能不同，重复epoch也不是新组织。并行单元吞吐不相加充当全GPU的全局速度。']
    (output/'PATCH_THROUGHPUT_AUDIT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare-indices','prepare-outer-indices','report'),required=True)
    parser.add_argument('--assets',required=True,type=Path);parser.add_argument('--ledger-inputs',required=True,type=Path)
    parser.add_argument('--core-private',required=True,type=Path);parser.add_argument('--path-private',required=True,type=Path)
    parser.add_argument('--index-cache',required=True,type=Path)
    parser.add_argument('--outer-private',type=Path)
    parser.add_argument('--asset-audit',type=Path,default=ROOT/'reports/bgb_en/R0_ASSET_AUDIT.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en')
    args=parser.parse_args();registry=read(args.asset_audit)['cohort']
    require(registry['patients']==512 and registry['DSS_events']==65,'Canonical asset registry differs')
    before=protected_snapshot(args.core_private)
    if args.stage=='prepare-indices':
        for part in (0,1,2):index_identity(args.core_private,args.index_cache,1,part,registry,create=True)
        guard=unchanged_guard(before,protected_snapshot(args.core_private))
        (args.index_cache/'PREPARE_INDICES_GUARD.json').write_text(json.dumps(guard,indent=2)+'\n')
        print(json.dumps(dict(status='THREE_PRIVATE_INDEX_CACHES_READY',model_forward_calls=0,new_optimizer_updates=0,GPU_used=False)));return
    if args.stage=='prepare-outer-indices':
        result=qualify_outer_indices(args,read(args.assets),registry)
        result['source_and_selection_guard']=unchanged_guard(before,protected_snapshot(args.core_private))
        args.output.mkdir(parents=True,exist_ok=True)
        payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
        (args.output/'PATCH_THROUGHPUT_R4_INDEX_QUALIFICATION.json').write_text(payload)
        (args.index_cache/'R4_INDEX_QUALIFICATION.json').write_text(payload)
        print(json.dumps({key:result[key] for key in ('status','prepared_bundles_verified','model_forward_calls','new_optimizer_updates','GPU_used')}));return
    result=generate(args,read(args.ledger_inputs),read(args.assets),registry)
    result['source_and_selection_guard']=unchanged_guard(before,protected_snapshot(args.core_private))
    result['mixed_bundle_index_extraction_guard']=read(args.index_cache/'PREPARE_INDICES_GUARD.json')
    if (args.index_cache/'R4_INDEX_QUALIFICATION.json').is_file():
        result['R4_index_extraction_qualification']=read(args.index_cache/'R4_INDEX_QUALIFICATION.json')
    write_reports(result,args.output)
    print(json.dumps({key:result[key] for key in ('status','available_receipt_or_attempt_units','positive_visual_throughput_units','zero_visual_units','NA_or_unsupported_units','GPU_used')}))


if __name__=='__main__':main()

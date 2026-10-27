"""Anonymous receipt-only progress snapshot; no tensors, model forward or fits.

This is a lineage sidebar, not the final execution ledger or scientific
decision. Outer COMPLETE files are accessed only for accounting and hashes;
their ``results`` and other performance fields are never indexed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_recovery_audit import matrix_inventory,require,sha256,verify_recovery_receipt


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_guard(mapping):
    require(isinstance(mapping,dict) and bool(mapping),'EMPTY_SOURCE_CLOSURE')
    for name,digest in mapping.items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'SOURCE_PATH_OUTSIDE_REPOSITORY')
        require(sha256(ROOT/relative)==digest,'SOURCE_SHA_CHANGED')
    return len(mapping)


def source_maps(value):
    """Only explicit repository-relative maps; data hashes are not code maps."""
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('source','sources','auxiliary_source','additional_sources','sensitivity_sources') and isinstance(item,dict):
                if item and all(str(name).startswith(('scripts/','src/','configs/')) for name in item):
                    found.append(item)
                else:found.extend(source_maps(item))
            else:found.extend(source_maps(item))
    return found


def arg(task,name):
    values=task['args'];require(values.count(name)==1,'PLAN_ARGUMENT_MISSING_OR_DUPLICATE')
    return values[values.index(name)+1]


def task_cell(task):
    root=Path(arg(task,'--private'));part=int(arg(task,'--partition'));seed=int(arg(task,'--seed'))
    script=Path(task['script']).name
    if script=='bgb_en_path_sensitivity.py':
        scenario=arg(task,'--encoder')+'_B'+arg(task,'--budget');arm=arg(task,'--arm')
        return root/'survival'/scenario/'cells'/f'R1_outer1_inner{part}_{arm}_seed{seed}','pathology_budget_and_encoder'
    if script=='bgb_en_omics_sensitivity.py':
        return root/'cells'/f"{arg(task,'--variant')}_p{part}_seed{seed}",'omics_factor_and_order'
    if script=='bgb_en_aux_run.py':
        return root/'cells'/f"R3_outer1_inner{part}_{arg(task,'--arm')}_seed{seed}",'external_source_transfer'
    if script=='bgb_en_budget_retrain.py':
        return root/'cells'/f"R2_newfit{arg(task,'--epochs')}_outer1_inner{part}_{arg(task,'--arm')}_seed{seed}",'training_budget'
    if script=='bgb_en_matched_support.py':
        variant=arg(task,'--variant');stage='R2' if variant=='O4_MATCHED_SUPPORT' else 'R3'
        require(variant in ('O4_MATCHED_SUPPORT','F5_MATCHED_SUPPORT'),'MATCHED_VARIANT_OUTSIDE_LOCK')
        return root/'cells'/f'{stage}_outer1_inner{part}_{variant}_seed{seed}','matched_support_annotation_null'
    raise ValueError('UNRECOGNIZED_LOCKED_TASK_SCRIPT')


def inspect_task(task,core_lineage,prepared_hashes):
    cell,group=task_cell(task);done=cell/'COMPLETE.json';failure=cell/'FAILED.json'
    value=dict(unit=task['label'],group=group,script=task['script'])
    if not done.exists():
        value['status']='FAILED' if failure.exists() else 'STARTED_NO_COMPLETE' if (cell/'START.json').exists() else 'NOT_STARTED'
        if failure.exists():
            row=read(failure);value['failure_receipt_sha256']=sha256(failure)
            value['exception_type']=row.get('exception_type',row.get('error_type'))
            value['known_completed_updates']=row.get('actual_optimizer_updates_completed')
            if Path(task['script']).name=='bgb_en_budget_retrain.py':
                lineage=row['lineage']
                require(lineage['core_origin']==core_lineage and lineage['independent_script_sha256']==sha256(ROOT/task['script'])
                        and lineage['prepared_origin_sha256']==prepared_hashes[str(lineage['inner_partition'])],
                        'FAILED_BUDGET_SOURCE_OR_PREPARED_CHANGED')
                require((lineage['arm'],lineage['seed'],lineage['inner_partition'],lineage['epochs'])
                        ==(arg(task,'--arm'),int(arg(task,'--seed')),int(arg(task,'--partition')),int(arg(task,'--epochs'))),
                        'FAILED_BUDGET_IDENTITY_CHANGED')
                value['failed_saved_model_exists']=(cell/'model.pt').exists()
                if value['failed_saved_model_exists']:value['failed_saved_model_sha256']=sha256(cell/'model.pt')
        return value
    require(not failure.exists(),'TASK_HAS_BOTH_COMPLETE_AND_FAILED_MARKERS')
    receipt=read(done);require(receipt['status'] in ('COMPLETE','COMPLETE_INDEPENDENT_BUDGET_FIT'),'TASK_NOT_COMPLETE')
    require(receipt.get('outer_test_evaluated',receipt.get('lineage',{}).get('outer_test_evaluated')) is False,
            'ADDITIONAL_TASK_OUTER_SCOPE_CHANGED')
    lineage=receipt['lineage'];script=Path(task['script']).name
    require(receipt.get('seed',lineage.get('seed'))==int(arg(task,'--seed'))
            and receipt.get('inner_partition',lineage.get('inner_partition'))==int(arg(task,'--partition')),
            'ADDITIONAL_TASK_IDENTITY_CHANGED')
    if script=='bgb_en_path_sensitivity.py':
        added=lineage['path_budget_sensitivity']
        require({k:v for k,v in lineage.items() if k!='path_budget_sensitivity'}==core_lineage,'PATH_CORE_ORIGIN_CHANGED')
        require(added['shared_teacher_origin_lineage']==core_lineage,'PATH_TEACHER_ORIGIN_CHANGED')
        require(added['lock_sha256']==sha256(Path(arg(task,'--private'))/'PATH_SENSITIVITY_LOCK.json'),'PATH_LOCK_CHANGED')
        require(added['scenario']==arg(task,'--encoder')+'_B'+arg(task,'--budget')
                and added['base_arm']==arg(task,'--arm') and receipt['arm']==arg(task,'--arm'),
                'PATH_SCENARIO_OR_ARM_CHANGED')
    elif script=='bgb_en_omics_sensitivity.py':
        require(lineage['core_lineage']==core_lineage and lineage['auxiliary_script_sha256']==sha256(ROOT/task['script']),
                'OMICS_SOURCE_CHANGED')
        require(lineage['prepared_sha256']==prepared_hashes[str(receipt['inner_partition'])],'OMICS_PREPARED_CHANGED')
        require(receipt['variant']==arg(task,'--variant') and lineage['variant']==receipt['variant'],
                'OMICS_VARIANT_CHANGED')
    elif script=='bgb_en_aux_run.py':
        extra={'auxiliary_source','shared_teacher_origin_lineage','UNI2_cache_receipt_sha256','external_projection_sha256'}
        require({k:v for k,v in lineage.items() if k not in extra}==core_lineage
                and lineage['shared_teacher_origin_lineage']==core_lineage,'AUX_CORE_ORIGIN_CHANGED')
        require(receipt['arm']==arg(task,'--arm'),'AUX_ARM_CHANGED')
        if '--project' in task['args']:
            require(lineage['external_projection_sha256']==sha256(Path(arg(task,'--project'))),'AUX_SOURCE_PROJECT_CHANGED')
    elif script=='bgb_en_budget_retrain.py':
        require(lineage['core_origin']==core_lineage and lineage['independent_script_sha256']==sha256(ROOT/task['script']),
                'BUDGET_SOURCE_CHANGED')
        require(lineage['prepared_origin_sha256']==prepared_hashes[str(lineage['inner_partition'])],'BUDGET_PREPARED_CHANGED')
        require(lineage['arm']==arg(task,'--arm') and lineage['epochs']==int(arg(task,'--epochs'))
                and lineage['budget_lock_sha256']==sha256(ROOT/'configs/bgb_en/budget_sensitivity_v1.json'),
                'BUDGET_LOCK_OR_IDENTITY_CHANGED')
    else:
        added=lineage['matched_support_auxiliary'];lock=read(Path(arg(task,'--private'))/'LOCK.json')
        require({k:v for k,v in lineage.items() if k!='matched_support_auxiliary'}==core_lineage
                and added['source']==lock['lineage'] and added['variant']==arg(task,'--variant'),
                'MATCHED_SUPPORT_SOURCE_OR_VARIANT_CHANGED')
        require(added['lock_sha256']==sha256(Path(arg(task,'--private'))/'LOCK.json')
                and added['null']['prepared_sha256']==prepared_hashes[str(receipt['inner_partition'])],
                'MATCHED_SUPPORT_LOCK_OR_PREPARED_CHANGED')
    maps=source_maps(lineage)
    require(bool(maps),'NO_EXPLICIT_CODE_SOURCE_MAP')
    for mapping in maps:source_guard(mapping)
    require(sha256(cell/'model.pt')==receipt['artifact_sha256'],'ADDITIONAL_ARTIFACT_SHA_CHANGED')
    require(receipt['epochs']==(int(arg(task,'--epochs')) if script=='bgb_en_budget_retrain.py' else 80),
            'ADDITIONAL_FIXED_EPOCH_CHANGED')
    value.update(status='COMPLETE_VERIFIED_METADATA_AND_SOURCE',receipt_sha256=sha256(done),
        artifact_sha256=receipt['artifact_sha256'],source_maps_verified=len(maps),epochs=receipt['epochs'],
        source_guard_passed=True,performance_fields_exported=False)
    return value


def plan_snapshot(path,core_lineage,prepared_hashes):
    plan=read(path);tasks=plan['tasks'];rows=[inspect_task(task,core_lineage,prepared_hashes) for task in tasks]
    groups=[]
    for name in sorted({row['group'] for row in rows}):
        selected=[row for row in rows if row['group']==name];counts=Counter(row['status'] for row in selected)
        groups.append(dict(group=name,planned=len(selected),completed_verified=counts['COMPLETE_VERIFIED_METADATA_AND_SOURCE'],
            started_without_complete=counts['STARTED_NO_COMPLETE'],failed=counts['FAILED'],not_started=counts['NOT_STARTED']))
    complete=sum(row['status']=='COMPLETE_VERIFIED_METADATA_AND_SOURCE' for row in rows)
    failed=sum(row['status']=='FAILED' for row in rows);ended=complete+failed
    return dict(status='COMPLETE_VERIFIED_PLANNED_RECEIPTS' if complete==len(rows)
                else 'ALL_ATTEMPTS_ENDED_WITH_ORIGINAL_FAILURES_RETAINED' if ended==len(rows) else 'PARTIAL',
        plan_sha256=sha256(path),planned=len(rows),completed_verified=complete,failed=failed,
        terminated_attempts=ended,all_attempts_ended=ended==len(rows),groups=groups,units=rows)


def batch_snapshot(log_root,plan_path):
    final=log_root/'BATCH_RECEIPT.json';progress=log_root/'BATCH_PROGRESS.json'
    path=final if final.exists() else progress
    if not path.exists():return dict(status='NO_BATCH_RECEIPT_OR_PROGRESS_OBSERVED',complete=False)
    row=read(path);contract=row['contract']
    require(contract['plan_sha256']==sha256(plan_path),'BATCH_PLAN_SHA_CHANGED')
    source_guard(contract['scripts'])
    require(contract['scheduler_sha256']==sha256(ROOT/'scripts/bgb_en_batch.py'),'BATCH_SCHEDULER_SOURCE_CHANGED')
    good=sum(unit.get('returncode')==0 for unit in row['units']);bad=sum(unit.get('returncode')!=0 for unit in row['units'])
    require(len(row['units'])==row['completed_units'] and len({unit['unit'] for unit in row['units']})==len(row['units']),
            'BATCH_RECEIPT_UNIT_COUNT_OR_DUPLICATES')
    complete=final.exists() and row.get('status')=='COMPLETE'
    if complete:
        require(good==row['planned_units']==row['completed_units'] and bad==0,'BATCH_COMPLETE_HAS_INCOMPLETE_OR_FAILED_UNITS')
    return dict(status=row.get('status','RUNNING_PROGRESS_ONLY'),complete=complete,
        planned_units=row['planned_units'],completed_process_returncodes=good,failed_process_returncodes=bad,
        receipt_sha256=sha256(path),contract_source_verified=True)


def external_snapshot(private):
    groups=[]
    definitions=[('RAW_V1','external_v1',7,False),('TRUE','external_v2_unitnorm',7,True),
                 ('WRONG_DONOR','external_v2_WRONG_DONOR',1,True),('SPATIAL_SHUFFLE','external_v2_SPATIAL_SHUFFLE',1,True)]
    for label,name,expected,eligible in definitions:
        rows=[]
        for path in sorted((private/name).glob('fit_*/COMPLETE.json')):
            row=read(path);require(row['status'] in ('COMPLETE_FIXED_SOURCE_FIT','COMPLETE_NEGATIVE_SOURCE_FIT')
                and row['epochs']==20 and row['optimizer_updates']==20 and row['actual_new_supervised_fits']==1,
                'EXTERNAL_FIXED_FIT_CONTRACT_CHANGED')
            for filename,digest in row['artifacts'].items():
                require(Path(filename).name==filename and sha256(path.parent/filename)==digest,'EXTERNAL_ARTIFACT_SHA_CHANGED')
            lineage=row.get('lineage',row.get('contract',{}).get('lineage',{}))
            if eligible:
                maps=source_maps(lineage);require(bool(maps),'EXTERNAL_CODE_SOURCE_MAP_MISSING')
                for mapping in maps:source_guard(mapping)
            rows.append(dict(fit_index=int(path.parent.name.split('_')[-1]),receipt_sha256=sha256(path),
                artifact_count_verified=len(row['artifacts']),optimizer_updates=20,
                current_code_source_comparison='PASS' if eligible else 'HISTORICAL_RAW_SOURCE_NOT_EXPECTED_TO_MATCH_REPAIRED_CODE'))
        require(len(rows)<=expected,'EXTERNAL_UNEXPECTED_EXTRA_FITS')
        groups.append(dict(group=label,planned=expected,completed_verified=len(rows),
            current_unitnorm_transfer_eligible=eligible,receipts=rows))
    count=sum(group['completed_verified'] for group in groups)
    return dict(status='COMPLETE_16_SOURCE_FITS_METADATA' if count==16 else 'PARTIAL',
        actual_fits=count,actual_updates=count*20,groups=groups,
        clinical_survival_fits_included=0,raw7_are_real_historical_cost_not_transferred=True,
        independent_tissue_evidence='author-released coarse pathologist spot labels; not cell source/fraction truth')


def matched_snapshot(private,core_lineage,prepared_hashes):
    directory=private/'matched_support_v1';plan=directory/'JOB_PLAN.json'
    if not plan.exists():
        return dict(status='PLANNED_NO_LOCKED_PLAN_OBSERVED',planned_new_neural_fits=12,
            actual_completed_not_inferred=None,changes_R4_candidate_pool=False)
    launch_path=directory/'LAUNCH_LOCK.json';launch=read(launch_path);lock_path=directory/'LOCK.json';lock=read(lock_path)
    require(launch['status']=='LOCKED_BEFORE_ADDITIONAL_FITS'
            and launch['private_job_plan_sha256']==sha256(plan)
            and launch['source_lock_sha256']==sha256(lock_path)
            and launch['additional_optimizer_fits_at_launch_lock']==0,'MATCHED_SUPPORT_LAUNCH_LOCK_CHANGED')
    require(lock['lineage']['core_lineage']==core_lineage and launch['R4_selection_modified'] is False,
            'MATCHED_SUPPORT_SCOPE_CHANGED')
    for mapping in source_maps(lock['lineage']):source_guard(mapping)
    for name,digest in launch['selection_sha256'].items():
        require(Path(name).name==name and sha256(ROOT/'reports/bgb_en'/name)==digest,'MATCHED_SUPPORT_SELECTION_LOCK_CHANGED')
    snapshot=plan_snapshot(plan,core_lineage,prepared_hashes)
    require(snapshot['planned']==12,'MATCHED_SUPPORT_PLAN_SIZE_CHANGED')
    snapshot.update(status='COMPLETE_12_VERIFIED_MATCHED_SUPPORT_FITS' if snapshot['completed_verified']==12
                    else 'LOCKED_PENDING_OR_PARTIAL_NOT_COMPLETE',
        launch_lock_sha256=sha256(launch_path),source_lock_sha256=sha256(lock_path),
        post_hoc_exploratory=True,changes_R4_candidate_pool=False,teacher_refits=0,
        excluded_from_84_and24_plans=True,
        scope='same active gene union and training measurement-mask counts; only annotation changes')
    io_path=directory/'IO_RUNTIME_LOCK.json'
    if io_path.exists():
        io=read(io_path);runtime_path=directory/'RUNTIME_IO_PLAN.json';runtime=read(runtime_path);scientific=read(plan)
        require(io['scientific_plan_sha256']==sha256(plan) and io['runtime_io_plan_sha256']==sha256(runtime_path)
                and io['scientific_source_lock_sha256']==sha256(lock_path)
                and io['immutable_launch_lock_sha256']==sha256(launch_path),'MATCHED_IO_RUNTIME_SHA_CHANGED')
        require(io['runtime_plan_builder_sha256']==sha256(ROOT/'scripts/bgb_en_matched_support_io_plan.py')
                and io['scientific_driver_sha256']==sha256(ROOT/'scripts/bgb_en_matched_support.py'),
                'MATCHED_IO_RUNTIME_CODE_CHANGED')
        require(len(runtime['tasks'])==len(scientific['tasks'])==12,'MATCHED_IO_RUNTIME_COUNT_CHANGED')
        destinations=set()
        for old,new in zip(scientific['tasks'],runtime['tasks']):
            require(new['script']==old['script'] and new['label']==old['label']
                    and new['args'][:-2]==old['args'] and new['args'][-2]=='--output','MATCHED_NON_IO_ARGUMENT_CHANGED')
            output=Path(new['args'][-1]).resolve()
            require(output.parent==(directory/'unit_public_views').resolve() and output not in destinations,
                    'MATCHED_IO_OUTPUT_NOT_UNIQUE_OR_PRIVATE')
            destinations.add(output)
        snapshot['runtime_IO_derivation']=dict(status='PASS_ONLY_UNIQUE_PRIVATE_OUTPUT_ADDED',
            runtime_plan_sha256=sha256(runtime_path),runtime_lock_sha256=sha256(io_path),
            scientific_plan_retained_sha256=sha256(plan),scientific_source_unchanged=True)
    return snapshot


def outer_snapshot(private):
    baselines=[]
    for fold in range(1,6):
        path=private/'outer'/f'fold{fold}'/'COMPLETE.json'
        if not path.exists():continue
        row=read(path);require(row['status']=='COMPLETE_OUTER_BASELINES' and row['fold']==fold,'R4_BASELINE_METADATA_CHANGED')
        require(sha256(path.parent/'models.pt')==row['artifact_sha256'],'R4_BASELINE_ARTIFACT_SHA_CHANGED')
        for mapping in source_maps(row['contract']):source_guard(mapping)
        require(row['contract']['source_sha256']==sha256(ROOT/'scripts/bgb_en_outer.py'),'R4_BASELINE_DRIVER_SOURCE_CHANGED')
        for name,digest in row['contract']['module_sha256'].items():
            require(Path(name).name==name and sha256(ROOT/'src/bgb_en'/name)==digest,'R4_BASELINE_MODULE_SOURCE_CHANGED')
        baselines.append(dict(fold=fold,actual_Cox_fits=row['actual_Cox_fits'],
            actual_scalar_fits=row['actual_scalar_calibration_fits'],receipt_sha256=sha256(path),
            artifact_sha256=row['artifact_sha256'],performance_fields_accessed=False))
    nested=[]
    for path in sorted((private/'nested_inner'/'cells').glob('*/COMPLETE.json')):
        row=read(path)
        require(row['status']=='COMPLETE' and row['stage']=='R3' and row['outer_fold'] in (2,3,4,5)
                and row['inner_partition'] in (0,1,2) and row['seed'] in (31,47) and row['epochs']==80
                and row['actual_optimizer_fits']==1 and row['outer_test_evaluated'] is False,
                'R4_NESTED_INNER_METADATA_CHANGED')
        require(sha256(path.parent/'model.pt')==row['artifact_sha256'],'R4_NESTED_INNER_ARTIFACT_SHA_CHANGED')
        source_guard(row['lineage']['source'])
        nested.append(dict(fold=row['outer_fold'],partition=row['inner_partition'],seed=row['seed'],arm=row['arm'],
            receipt_sha256=sha256(path),artifact_sha256=row['artifact_sha256'],unique_optimizer_fits=1))
    require(len({(r['fold'],r['partition'],r['seed'],r['arm']) for r in nested})==len(nested),
            'R4_NESTED_INNER_DUPLICATE_IDENTITY')
    preparation=[]
    for path in sorted((private/'nested_inner'/'prepared').glob('outer*_inner*.json')):
        require(path.stem in {f'outer{fold}_inner{part}' for fold in (2,3,4,5) for part in (0,1,2)},
                'R4_NESTED_PREPARATION_UNIT_OUTSIDE_PLAN')
        row=read(path)
        require(row['status']=='COMPLETE' and row['upstream_optimizer_fits']==13
                and row['upstream_preprocessing_fits']==4 and 'scalar_beta' in row,
                'R4_NESTED_PREPARATION_ACCOUNTING_CHANGED')
        require(sha256(path.with_suffix('.pt'))==row['sha256'],'R4_NESTED_PREPARATION_ARTIFACT_SHA_CHANGED')
        source_guard(row['lineage']['source'])
        preparation.append(dict(unit=path.stem,receipt_sha256=sha256(path),artifact_sha256=row['sha256'],
            actual_Cox_fits=13,actual_beta_fits=1,actual_preprocessing_fits=4,
            independent_CPU_precompute_or_later_scheduler_reuse_counted_once=True))
    final=[]
    for path in sorted((private/'neural').rglob('COMPLETE.json')):
        row=read(path)
        require(row['status']=='COMPLETE_OUTER_NEURAL' and row['actual_new_optimizer_fits']==1
                and row['epochs']==80 and row['fold'] in (1,2,3,4,5) and row['seed'] in (31,47,83)
                and row['outer_test_read_for_selection'] is False,'R4_FINAL_NEURAL_METADATA_CHANGED')
        require(sha256(path.parent/'model.pt')==row['artifact_sha256'],'R4_FINAL_NEURAL_ARTIFACT_SHA_CHANGED')
        for mapping in source_maps(row['contract']):source_guard(mapping)
        final.append(dict(fold=row['fold'],seed=row['seed'],receipt_sha256=sha256(path),
            artifact_sha256=row['artifact_sha256'],unique_optimizer_fits=1,parameters_active_requires_grad=row['parameters']))
    require(len({(r['fold'],r['seed']) for r in final})==len(final),'R4_FINAL_DUPLICATE_IDENTITY')
    require(len(preparation)<=12 and len(nested)<=96 and len(final)<=15,'R4_UNEXPECTED_EXTRA_OPTIMIZATION_RECEIPTS')
    baseline_fits=sum(row['actual_Cox_fits']+row['actual_scalar_fits'] for row in baselines)
    teacher_fits=sum(row['actual_Cox_fits']+row['actual_beta_fits'] for row in preparation)
    actual_total=baseline_fits+teacher_fits+len(nested)+len(final)
    return dict(status='ACCOUNTING_SNAPSHOT_OUTER_SCORES_NOT_ACCESSED',
        baseline_bundles_planned=5,baseline_bundles_completed_verified=len(baselines),baseline_bundles=baselines,
        baseline_unique_optimizer_fits_actual=baseline_fits,
        baseline_unique_optimizer_fits_planned=140,baseline_unique_CP_risk_vectors_planned=5,
        baseline_unique_CP_SCALAR_risk_vectors_planned=15,baseline_logical_seed_model_evaluation_cells_planned=30,
        baseline_logical_evaluation_cells_are_not_unique_fits=True,
        nested_new_teacher_preparation_bundles_planned=12,nested_new_teacher_preparation_receipts_verified=len(preparation),
        nested_preparation_receipts=preparation,nested_teacher_optimizer_fits_actual=teacher_fits,
        nested_teacher_optimizer_fits_planned=168,nested_new_inner_neural_fits_planned=96,
        nested_inner_fits_metadata_and_hash_verified=len(nested),nested_inner_receipts=nested,
        outer1_inner24_reused_core_fits_not_new=24,final_neural_fits_planned=15,
        final_neural_fits_metadata_and_hash_verified=len(final),final_neural_receipts=final,
        total_unique_optimizer_fits_planned=419,total_unique_optimizer_fits_actual=actual_total,
        independent_numerical_model_replay_not_performed_by_this_metadata_audit=True,outer_score_fields_accessed=False,
        note='140 baseline + 168 nested teachers + 96 new inner + 15 final; actual completion is receipt based, not 45 logical evaluation cells')


def run(args):
    private=args.private_root;core=private/'formal_v1';configuration=read(ROOT/'configs/bgb_en/prototype_v1.json')
    matrix=matrix_inventory(core,configuration);lineage=next(iter(matrix.values()))[1]['lineage']
    source_guard(lineage['source']);require(all(row['lineage']==lineage for _,row in matrix.values()),'CORE_MIXED_SOURCES')
    prepared={}
    for part in range(3):
        path=core/'prepared'/f'outer1_inner{part}.json';row=read(path)
        require(row['status']=='COMPLETE' and row['lineage']==lineage and row['upstream_optimizer_fits']==13,
                'CORE_PREPARATION_METADATA_CHANGED')
        require(sha256(path.with_suffix('.pt'))==row['sha256'],'CORE_PREPARED_ARTIFACT_SHA_CHANGED')
        prepared[str(part)]=row['sha256']
    recovery=verify_recovery_receipt(private/'F7_recovery_v1'/'RECOVERY_RECEIPT.json',core,
        config_path=ROOT/'configs/bgb_en/prototype_v1.json')
    additional=plan_snapshot(private/'ADDITIONAL_PLAN_LOCK.private.json',lineage,prepared)
    batch=batch_snapshot(private/'additional_v1_logs',private/'ADDITIONAL_PLAN_LOCK.private.json')
    budget=plan_snapshot(private/'BUDGET_PLAN_LOCK.private.json',lineage,prepared)
    external=external_snapshot(private)
    pca=read(ROOT/'reports/bgb_en/R2_PCA_GEOMETRY_REPLAY_AUDIT.json')
    require(pca['status']=='PASS_ALL_NINE_SAVED_COX_REPLAYS' and pca['actual_saved_Cox_coefficients_replayed']==9
            and pca['original_solver_attempts']==9 and pca['precision_retry_attempts']==2,'PCA_REPLAY_ACCOUNTING_CHANGED')
    matched=matched_snapshot(private,lineage,prepared);outer=outer_snapshot(args.outer_private)
    source_replay_path=ROOT/'reports/bgb_en/EXTERNAL_CHECKPOINT_REPLAY_WITH_CONTROLS.json'
    source_replay=read(source_replay_path)
    require(source_replay['status']=='PASS_EXTERNAL_CHECKPOINT_REPLAY'
            and source_replay['actual_recorded_source_fits']==16 and source_replay['audit_optimizer_updates']==0,
            'SIXTEEN_SOURCE_INDEPENDENT_REPLAY_EVIDENCE_CHANGED')
    reports={}
    for name in ('PATH_BUDGET_SENSITIVITY_RESULTS.json','R2_OMICS_SENSITIVITY_AUDIT.json',
                 'AUXILIARY_TRANSFER_RESULTS.json','AUX_PROJECTION_DRIFT.json'):
        path=ROOT/'reports/bgb_en'/name
        if path.exists():reports[name]=dict(status=read(path)['status'],sha256=sha256(path))
    units=additional['units']+budget['units']+matched.get('units',[])
    completed=[row for row in units if row['status']=='COMPLETE_VERIFIED_METADATA_AND_SOURCE']
    matched_count=matched.get('completed_verified',0)
    primary_fits=216+external['actual_fits']+9+len(completed)+outer['total_unique_optimizer_fits_actual']
    known_primary_updates=162*80+external['actual_updates']+sum(row['epochs'] for row in completed)
    known_primary_updates+=80*(outer['nested_inner_fits_metadata_and_hash_verified']+outer['final_neural_fits_metadata_and_hash_verified'])
    decoder_count=93+2*sum(row['group']=='omics_factor_and_order' for row in completed)
    decoder_count+=2*sum(row['group']=='training_budget' or row['unit'].startswith('O4_MATCHED_SUPPORT') for row in completed)
    failed_rows=[row for row in units if row['status']=='FAILED']
    failed_updates=sum(row['known_completed_updates'] for row in failed_rows if row.get('known_completed_updates') is not None)
    failed_decoder_fits=0;saved_budget_diagnostic=None
    diagnostic_path=ROOT/'reports/bgb_en/R2_BUDGET_SAVED_CHAIN_AUDIT.json'
    if diagnostic_path.exists():
        diagnostic=read(diagnostic_path);diagnostic_lock=ROOT/'reports/bgb_en/R2_BUDGET_SAVED_CHAIN_AUDIT_LOCK.json'
        require(diagnostic['status']=='COMPLETE_24_SAVED_CHAIN_DIAGNOSIS_ORIGINAL_23_ACCEPTED_1_FAILED_RETAINED'
                and diagnostic['audit_lock_sha256']==sha256(diagnostic_lock)
                and diagnostic['source_sha256']==sha256(ROOT/'scripts/bgb_en_budget_saved_chain_audit.py'),
                'SAVED_BUDGET_DIAGNOSTIC_SOURCE_CHANGED')
        locked_models={row['unit']:row for row in read(diagnostic_lock)['models']}
        for row in failed_rows:
            # Labels in the schedule use a short prefix; marker identities and
            # source-bound saved artifact SHA identify the physical failed fit.
            matches=[model for model in locked_models.values()
                     if model['artifact_sha256']==row.get('failed_saved_model_sha256')]
            require(len(matches)==1 and matches[0]['original_role']=='FAILED_PARENT80_NUMERICAL_GUARD',
                    'FAILED_ARTIFACT_DIFFERS_FROM_ZERO_FIT_DIAGNOSTIC_LOCK')
        failed_decoder_fits=diagnostic['failed_model_existing_risk_decoders']
        saved_budget_diagnostic=dict(status=diagnostic['status'],sha256=sha256(diagnostic_path),
            lock_sha256=sha256(diagnostic_lock),original_failed_status_unchanged=True,
            physical_updates=diagnostic['actual_saved_fit_optimizer_updates'],
            physical_decoders=diagnostic['actual_existing_risk_decoders'],
            qualification_references_not_replaced=True)
    all_done=(additional['completed_verified']==84 and batch['complete'] and budget['all_attempts_ended']
              and matched_count==12 and outer['total_unique_optimizer_fits_actual']==419)
    return dict(schema='bgb_en_experiment_lineage_snapshot_v1',
        status=('COMPLETE_EXECUTION_ACCOUNTING_WITH_ORIGINAL_BUDGET_FAILURE_RETAINED' if failed_rows
                else 'COMPLETE_ACCOUNTING_SOURCE_SNAPSHOT_NOT_SCIENTIFIC_DECISION') if all_done
                else 'PARTIAL_EXECUTION_SNAPSHOT_NOT_FINAL_DECISION',
        timestamp_UTC=datetime.now(timezone.utc).isoformat(),inventory_script_sha256=sha256(__file__),
        audit_work=dict(GPU_calls=0,model_deserializations=0,forward_calls=0,backward_calls=0,new_fits=0,
            outer_performance_fields_accessed=False),
        core=dict(status='COMPLETE_186_ARTIFACT_HASH_AND_SOURCE_VERIFIED',unique_evaluation_cells=186,
            neural_fits=162,new_Cox_fits=12,reference_cells_no_new_optimizer=12,
            upstream_Cox_fits=39,upstream_beta_fits=3,upstream_total_optimizer_fits=42,
            total_primary_core_optimizer_fits=216,source_sha256=lineage['source'],prepared_artifact_sha256=prepared,
            analytical_decoder_fits=93,logical_seed_reuse_reference_cells=24,
            checkpoint_replay_boundary='original independent checkpoint audit remains184PASS/2FAILED O9; separate30forward diagnosis preserved unchanged ranking, no audit promotion'),
        F7_recovery=dict(status=recovery['status'],certificate_sha256=sha256(private/'F7_recovery_v1'/'RECOVERY_RECEIPT.json'),
            original_attempts=6,original_completed_fits_archived=3,original_failed_attempts_archived=3,
            original_failed_update_counts=None,new_fresh_fits=6,new_updates=480,teacher_refits=0,
            new6_are_already_in_core162_not_added_again=True,old3_success_are_additional_historical_optimizer_cost=True,
            old3_success_updates=240,original_partial_schedule_retained=True),
        external_source=external,PCA_geometry=dict(valid_saved_Cox_fits=9,actual_attempts=11,
            original_attempts=9,original_successes=7,original_failed_attempts=2,precision_retry_successes=2,
            independent_replay_sha256=sha256(ROOT/'reports/bgb_en/R2_PCA_GEOMETRY_REPLAY_AUDIT.json'),
            no_PCA_or_teacher_refits=True),additional84=additional,budget24=budget,R4=outer,
        matched_support12=matched,
        additional_batch=batch,
        reporting_gate=dict(additional_final_reports_allowed=additional['completed_verified']==84 and batch['complete'],
            final_reports_not_invoked_by_this_script=True,aux_projection_drift_not_invoked_by_this_script=True,
            no_R5_training_authorization_from_this_draft=True),
        final_additional_report_status=reports,
        saved_budget_diagnostic=saved_budget_diagnostic,
        independent_source_replay16=dict(status=source_replay['status'],sha256=sha256(source_replay_path),
            actual_source_fits_replayed=16,audit_optimizer_updates=0,
            boundary='the frozen auxiliary reporter separately retains its older14-fit replay reference; this16-fit certificate also verifies both negative-source checkpoints'),
        receipt_derived_accounting_crosscheck=dict(
            successful_supervision_fits_excluding_only_historical_F7_success=primary_fits,
            historical_successful_F7_fits_separate=3,all_successful_supervision_fits_including_F7_history=primary_fits+3,
            raw_source7_included_as_real_cost_not_eligible_target_models=True,
            known_primary_neural_and_source_updates=known_primary_updates,
            known_updates_including_historical_F7_success=known_primary_updates+240,
            known_new_failed_updates_separate=failed_updates,
            known_total_neural_and_source_updates_including_history_and_failed=known_primary_updates+240+failed_updates,
            known_failed_attempts=5+len(failed_rows),historical_failed_updates_still_unknown_not_fabricated=True,
            completed_analytical_decoder_fits=decoder_count,
            known_failed_artifact_decoder_fits_separate=failed_decoder_fits,
            actual_analytical_decoder_fits_including_failed_artifacts=decoder_count+failed_decoder_fits,
            PCA_valid9_bound_to_independent_replay_evidence_not_new_models=True,
            final_execution_ledger_remains_authoritative=True,expected_full_matrix_totals_not_substituted_for_receipts=True),
        interpretation='Counts distinguish evaluation cells, supervised optimization fits, analytical decoders and references. A partial or planned block is never promoted to COMPLETE by expected counts.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-root',type=Path,required=True);p.add_argument('--outer-private',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en/EXPERIMENT_LINEAGE.json')
    args=p.parse_args();value=run(args);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=value['status'],additional_complete=value['additional84']['completed_verified'],
        budget_complete=value['budget24']['completed_verified'],source_fits=value['external_source']['actual_fits'],
        outer_baseline_bundles=value['R4']['baseline_bundles_completed_verified'],new_fits=0)))


if __name__=='__main__':main()

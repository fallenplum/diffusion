"""Metadata-only execution and resource ledger for explicit private roots.

The private input JSON names roots, roles, scheduler receipts and resource CSVs.
No patient arrays, model deserialization, forward pass or optimizer is used.
Public output never contains supplied paths, process IDs or exception messages.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
KINDS={'core','extra','budget','source','pca','outer','archive'}
ROLES={'PRIMARY','HISTORICAL'}
STATES={'RUNNING','CLOSED','NOT_RUN'}
MARKERS={'COMPLETE.json','FAILED.json','START.json','STARTED.json'}
COUNTS=('reference_cells','completed_Cox_fits','completed_beta_fits','completed_neural_fits',
        'completed_external_fits','completed_supervised_optimizer_fits','analytical_decoder_fits','completed_preprocessing_fits',
        'logical_seed_reuse_references','failed_attempts','unfinished_attempts',
        'unknown_failed_or_unfinished_update_counts','known_neural_external_optimizer_updates',
        'known_failed_or_unfinished_optimizer_updates')


def require(value,message):
    if not bool(value):raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_label(value):
    require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_]+',value),'Only anonymous alphanumeric labels are permitted')
    return value


def nonnegative(value,name):
    require(not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>=0,
            'Invalid nonnegative metadata field: '+name)
    return value


def empty_counts():return {key:0 for key in COUNTS}


def decoder_count(row):
    if 'analytical_risk_decoder_fits' in row:return int(nonnegative(row['analytical_risk_decoder_fits'],'decoder count'))
    retention=row.get('retention',{})
    if not retention:return 0
    if 'complete_tokens' in retention and 'pooled32' in retention:return 2
    return int(retention.get('decoder_available') is True)


def classify_complete(row,kind,prepared=False):
    """One receipt may represent multiple Cox/beta fits, never multiple attempts."""
    counts=empty_counts();status=row.get('status');category=None;updates=None
    if prepared:
        require(status=='COMPLETE' and row.get('upstream_optimizer_fits')==13
                and row.get('upstream_preprocessing_fits')==4 and 'scalar_beta' in row,
                'Unsupported prepared-anchor accounting contract')
        counts.update(completed_Cox_fits=13,completed_beta_fits=1,completed_preprocessing_fits=4)
        category='UPSTREAM_ANCHOR_BUNDLE'
    elif status in ('COMPLETE_FIXED_SOURCE_FIT','COMPLETE_NEGATIVE_SOURCE_FIT'):
        require(kind in ('source','archive') and row.get('actual_new_supervised_fits')==1,
                'Source fit outside explicit source/history root')
        counts['completed_external_fits']=1;updates=int(nonnegative(row['optimizer_updates'],'source updates'))
        counts['known_neural_external_optimizer_updates']=updates;category='EXTERNAL_SOURCE'
    elif status=='COMPLETE_OUTER_BASELINES':
        counts['completed_Cox_fits']=int(nonnegative(row['actual_Cox_fits'],'outer Cox fits'))
        counts['completed_beta_fits']=int(nonnegative(row['actual_scalar_calibration_fits'],'outer beta fits'))
        counts['logical_seed_reuse_references']=int(row['CP_seed_references'])-int(row['deterministic_CP_test_vector_fits'])
        require(counts['logical_seed_reuse_references']>=0,'Outer deterministic reference count is invalid')
        category='OUTER_ANCHOR_BUNDLE'
    elif status=='COMPLETE_OUTER_NEURAL':
        require(row.get('actual_new_optimizer_fits')==1,'Outer neural fit accounting differs')
        counts['completed_neural_fits']=1;updates=int(nonnegative(row['epochs'],'outer neural epochs'))
        counts['known_neural_external_optimizer_updates']=updates;category='NEURAL'
    elif status=='COMPLETE_INDEPENDENT_BUDGET_FIT':
        require(row.get('actual_neural_optimizer_fits')==1,'Budget neural fit accounting differs')
        counts['completed_neural_fits']=1;updates=int(nonnegative(row['optimizer_updates'],'budget updates'))
        counts['known_neural_external_optimizer_updates']=updates
        counts['analytical_decoder_fits']=decoder_count(row);category='NEURAL'
    elif status=='COMPLETE' and kind=='pca':
        require(row.get('actual_optimizer_fits',row.get('actual_new_Cox_fits'))==1,'PCA Cox accounting differs')
        counts['completed_Cox_fits']=1;category='COX'
    elif status=='COMPLETE' and row.get('stage') in ('R1','R2','R3','R2_SENSITIVITY'):
        actual=row.get('actual_optimizer_fits',row.get('actual_new_optimizer_fits'))
        require(actual in (0,1),'Cell optimizer-fit count is missing or unsupported')
        epochs=int(nonnegative(row['epochs'],'cell epochs'))
        if actual==0:
            require(epochs==0,'Reference cell claims training epochs')
            counts['reference_cells']=1;category='REFERENCE'
        elif epochs==0:
            counts['completed_Cox_fits']=1;category='COX'
        else:
            counts['completed_neural_fits']=1;updates=int(row.get('optimizer_updates',epochs))
            counts['known_neural_external_optimizer_updates']=updates;category='NEURAL'
        counts['analytical_decoder_fits']=decoder_count(row)
        if epochs==0 and row.get('seed')==31 and (row.get('arm') in ('P0','O0','O1','F0','F1') or str(row.get('arm','')).startswith('O2_')):
            counts['logical_seed_reuse_references']=1
    else:
        raise ValueError('Unsupported COMPLETE metadata schema; no accounting inferred')
    counts['completed_supervised_optimizer_fits']=sum(counts[key] for key in
        ('completed_Cox_fits','completed_beta_fits','completed_neural_fits','completed_external_fits'))
    for key,value in counts.items():nonnegative(value,key)
    return category,counts,updates


def validate_roots(configuration):
    roots=[];names=set();locations={}
    for value in configuration.get('roots',[]):
        name=safe_label(value['name']);require(name not in names,'Duplicate ledger root name');names.add(name)
        require(value['kind'] in KINDS and value.get('role','PRIMARY') in ROLES
                and value.get('state','RUNNING') in STATES,'Unknown ledger root kind, role or state')
        path=Path(value['path']).resolve();role=value.get('role','PRIMARY')
        if path in locations:require(locations[path]==role,'The same physical root cannot be primary and historical')
        locations[path]=role
        roots.append(dict(value,name=name,path=path,role=role,state=value.get('state','RUNNING')))
    require(roots,'Ledger requires explicit private roots')
    return roots


def owning_root(path,roots):
    owners=[root for root in roots if path.is_relative_to(root['path'])]
    require(owners,'Metadata lies outside all explicit roots')
    return max(owners,key=lambda root:(len(root['path'].parts),-roots.index(root)))


def artifact_record(folder,row,prepared=False):
    if prepared:return dict(declared_sha256=row.get('sha256'),payload_present=folder.with_suffix('.pt').is_file())
    digest=row.get('artifact_sha256')
    if digest is not None:
        expected=folder/('models.pt' if row.get('status')=='COMPLETE_OUTER_BASELINES' else 'model.pt')
        return dict(declared_sha256=digest,payload_present=expected.is_file())
    artifacts=row.get('artifacts',{})
    if artifacts:
        require(all(Path(name).name==name for name in artifacts),'Source artifact name is not a basename')
        return dict(declared_checkpoint_sha256=artifacts.get('checkpoint.pt'),
                    payload_present=all((folder/name).is_file() for name in artifacts))
    return dict(declared_sha256=None,payload_present=False)


def attempt_record(folder,root):
    complete=folder/'COMPLETE.json';failed=folder/'FAILED.json'
    require(not (complete.is_file() and failed.is_file()),'One attempt has both COMPLETE and FAILED; resolve archive provenance explicitly')
    markers={name:sha(folder/name) for name in MARKERS if (folder/name).is_file()}
    chosen=complete if complete.is_file() else failed if failed.is_file() else next(folder/name for name in ('START.json','STARTED.json') if (folder/name).is_file())
    row=read(chosen);counts=empty_counts();update=None;category='UNKNOWN_ATTEMPT'
    if complete.is_file():
        category,counts,update=classify_complete(row,root['kind'])
        state='COMPLETE';artifact=artifact_record(folder,row)
    else:
        state='FAILED' if failed.is_file() else 'UNFINISHED_RUNNING_OR_INTERRUPTED'
        counts['failed_attempts' if failed.is_file() else 'unfinished_attempts']=1
        for field in ('actual_optimizer_updates_completed','actual_new_updates','optimizer_updates'):
            if field in row and row[field] is not None:update=int(nonnegative(row[field],field));break
        if update is None:counts['unknown_failed_or_unfinished_update_counts']=1
        else:counts['known_failed_or_unfinished_optimizer_updates']=update
        artifact=None
    epoch=row.get('epochs') if complete.is_file() else None
    elapsed=row.get('elapsed_seconds')
    if elapsed is not None:nonnegative(elapsed,'elapsed seconds')
    peak=row.get('peak_GPU_bytes')
    if peak is not None:nonnegative(peak,'process peak GPU bytes')
    identity={key:row[key] for key in ('stage','arm','variant','seed','inner_partition','outer_fold','fold') if key in row}
    # Only known scalar identities leave; no arbitrary contract or exception is copied.
    for key,value in identity.items():
        require(isinstance(value,(str,int)) and (not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9_]+',value)),
                'Nonanonymous cell identity')
    start=read(folder/'START.json').get('start') if (folder/'START.json').is_file() else None
    if start is not None:nonnegative(start,'recorded start timestamp')
    return dict(root=root['name'],role=root['role'],status=state,category=category,**identity,
        epoch=epoch,known_optimizer_updates=update,counts=counts,
        receipt_sha256=sha(chosen),marker_sha256=markers,
        saved_payload_metadata=artifact,artifact_bytes_rehashed_by_this_ledger=False,
        unit_elapsed_seconds=elapsed,process_PyTorch_peak_GPU_bytes=peak,
        start_timestamp_recorded=start is not None,
        _start_unix=start,_directory=str(folder))


def enumerate_records(roots):
    folders=set();preparations=set()
    for root in roots:
        if not root['path'].is_dir():continue
        for name in MARKERS:
            folders.update(path.parent.resolve() for path in root['path'].rglob(name))
        if root['kind'] in ('core','outer'):
            preparations.update(path.resolve() for path in root['path'].glob('**/prepared/*.json'))
    records=[]
    for folder in sorted(folders):records.append(attempt_record(folder,owning_root(folder,roots)))
    # Exact path identity deduplicates overlapping root declarations. It does
    # not merge separately recorded fresh attempts just because weights match.
    for path in sorted(preparations):
        root=owning_root(path,roots);row=read(path)
        category,counts,updates=classify_complete(row,root['kind'],prepared=True)
        records.append(dict(root=root['name'],role=root['role'],status='COMPLETE',category=category,
            epoch=None,known_optimizer_updates=updates,counts=counts,receipt_sha256=sha(path),
            marker_sha256={},saved_payload_metadata=artifact_record(path,row,True),
            artifact_bytes_rehashed_by_this_ledger=False,unit_elapsed_seconds=row.get('elapsed_seconds'),
            process_PyTorch_peak_GPU_bytes=None,start_timestamp_recorded=False,_start_unix=None,_directory=str(path)))
    return records


def sum_counts(rows):
    return {key:sum(row['counts'][key] for row in rows) for key in COUNTS}


def apply_failure_accounting(records,specifications):
    """Identify original PCA failures by preserved START/COMPLETE hashes only.

    Initial PCA solver exceptions did not write individual FAILED files. The
    independent precision-repair lock preserves nine starts, seven successes
    and the two predeclared failures. It does not recover their iteration count.
    """
    audits=[]
    for spec in specifications:
        name=safe_label(spec['name']);root=safe_label(spec['root']);path=Path(spec['path'])
        if not path.is_file():audits.append(dict(name=name,status='NOT_RUN_ACCOUNTING_CERTIFICATE_ABSENT'));continue
        value=read(path)
        require(value.get('original_solver_attempts')==9 and value.get('original_valid_units')==7
                and value.get('original_failed_units')==2,'Unknown original-PCA failure certificate')
        rows=[row for row in records if row['root']==root]
        starts=value['original_START_sha256'];complete=value['original_complete_sha256']
        require(len(starts)==9 and len(complete)==7
                and {row['marker_sha256'].get('START.json') for row in rows}==set(starts.values())
                and {row['receipt_sha256'] for row in rows if row['status']=='COMPLETE'}==set(complete.values()),
                'PCA failure certificate does not exactly match preserved starts and successes')
        failing=set(starts)-set(complete)
        intended={f"PCA{unit['dimension']}_UNWHITENED_p{unit['partition']}" for unit in value['failed_units']}
        require(failing==intended and len(failing)==2,'PCA failure identities differ from independent precision lock')
        hashes={starts[name] for name in failing};marked=0
        for row in rows:
            if row['marker_sha256'].get('START.json') not in hashes:continue
            require(row['status']=='UNFINISHED_RUNNING_OR_INTERRUPTED','PCA certificate tries to relabel a completed or explicitly failed attempt')
            row['counts']['unfinished_attempts']-=1;row['counts']['failed_attempts']+=1
            row['status']='FAILED_CERTIFIED_NO_LAST_STATE_SAVED'
            row['failure_accounting_certificate_sha256']=sha(path);marked+=1
        require(marked==2,'PCA certificate did not resolve exactly two failed attempts')
        audits.append(dict(name=name,status='PASS_NINE_START_SEVEN_COMPLETE_TWO_FAILED_ACCOUNTING',
                           certificate_sha256=sha(path),certified_failed_attempts=marked,failed_optimizer_updates=None))
    return audits


def scheduler_reports(specifications):
    result=[];seen=set()
    for item in specifications:
        name=safe_label(item['name']);path=Path(item['path']).resolve()
        if path in seen:continue
        seen.add(path)
        if not path.is_file():
            state=item.get('state')
            require(state is None or state in STATES,'Unknown explicit scheduler execution state')
            result.append(dict(name=name,status=state+'_FINAL_RECEIPT_ABSENT' if state else
                               'NO_FINAL_RECEIPT_EXECUTION_STATE_UNKNOWN'));continue
        row=read(path);status=row.get('status','PROGRESS_ONLY')
        require(isinstance(status,str) and re.fullmatch('[A-Z0-9_]+',status),'Scheduler status must be an anonymous code')
        elapsed=row.get('elapsed_seconds')
        if elapsed is not None:nonnegative(elapsed,'scheduler makespan')
        units=row.get('units',[])
        unit_seconds=[unit.get('seconds',unit.get('elapsed_seconds')) for unit in units]
        unit_seconds=[float(value) for value in unit_seconds if value is not None]
        require(all(math.isfinite(value) and value>=0 for value in unit_seconds),'Invalid scheduled unit elapsed')
        entry=dict(name=name,status=status,receipt_sha256=sha(path),
            observed_scheduler_elapsed_seconds=elapsed,
            sum_reported_unit_elapsed_seconds=sum(unit_seconds),
            reported_returned_units=len(units),reported_nonzero_returncodes=sum(unit.get('returncode',0)!=0 for unit in units),
            explicitly_recorded_failure_entries=len(row.get('failures',[])),
            planned_units=row.get('planned_units',row.get('unique_units')),completed_units=row.get('completed_units'),
            elapsed_is_scheduler_scope_not_sum_of_worker_times=True)
        start=item.get('start_unix');end=item.get('end_unix')
        require((start is None)==(end is None),'Global timeline needs both explicit recorded scheduler bounds')
        if start is not None:
            require(end>=start and elapsed is not None and abs((end-start)-elapsed)<max(1.,elapsed*1e-4),
                    'Explicit scheduler timeline does not match recorded elapsed')
            entry['_interval']=(float(start),float(end))
        result.append(entry)
    return result


def interval_union(intervals):
    if not intervals:return None
    chosen=sorted(intervals);left,right=chosen[0];covered=0.
    for a,b in chosen[1:]:
        if a>right:covered+=right-left;left,right=a,b
        else:right=max(right,b)
    covered+=right-left
    return dict(first_to_last_elapsed_seconds=float(max(v[1] for v in chosen)-min(v[0] for v in chosen)),
                union_covered_seconds=float(covered),overlapping_scopes_not_double_counted=True)


def resource_report(specification):
    name=safe_label(specification['name']);path=Path(specification['path'])
    if not path.is_file():return dict(name=name,status='NOT_RUN_RESOURCE_CSV_ABSENT')
    required=('elapsed_seconds','GPU_util_percent','GPU_memory_MiB','GPU_total_MiB',
              'CPU_util_percent','RAM_available_MiB','disk_available_MiB')
    with path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle);require(set(required)<=set(reader.fieldnames or []),'Unknown resource CSV schema')
        rows=[{key:float(row[key]) for key in required} for row in reader]
    if not rows:return dict(name=name,status='EMPTY_RESOURCE_CSV',resource_csv_sha256=sha(path),samples=0)
    require(all(math.isfinite(v) and v>=0 for row in rows for v in row.values()),'Invalid resource values')
    require(all(row['GPU_util_percent']<=100 and row['CPU_util_percent']<=100
                and row['GPU_memory_MiB']<=row['GPU_total_MiB']+1 for row in rows),'Resource range invalid')
    dt=[rows[i+1]['elapsed_seconds']-rows[i]['elapsed_seconds'] for i in range(len(rows)-1)]
    require(all(value>0 for value in dt),'Resource timestamps must strictly increase')
    nominal=statistics.median(dt) if dt else None
    supported=[i for i,value in enumerate(dt) if value<=3*nominal] if dt else []
    duration=sum(dt[i] for i in supported)
    weighted=lambda key:sum(.5*(rows[i][key]+rows[i+1][key])*dt[i] for i in supported)/duration if duration else None
    return dict(name=name,status='OBSERVED_MONITOR_WINDOW_ONLY',resource_csv_sha256=sha(path),samples=len(rows),
        elapsed_first_sample=rows[0]['elapsed_seconds'],elapsed_last_sample=rows[-1]['elapsed_seconds'],
        observed_window_seconds=rows[-1]['elapsed_seconds']-rows[0]['elapsed_seconds'],
        supported_interpolation_seconds=duration,excluded_long_gap_seconds=sum(dt)-duration,
        nominal_sample_interval_seconds=nominal,time_weighting='trapezoidal; omit gaps greater than three median sample intervals',
        GPU_util_time_weighted_mean_percent=weighted('GPU_util_percent'),
        GPU_util_sample_median_percent=statistics.median(row['GPU_util_percent'] for row in rows),
        GPU_util_sample_max_percent=max(row['GPU_util_percent'] for row in rows),
        sampled_device_GPU_memory_peak_MiB=max(row['GPU_memory_MiB'] for row in rows),
        sampled_device_GPU_total_MiB=max(row['GPU_total_MiB'] for row in rows),
        host_CPU_util_time_weighted_mean_percent=weighted('CPU_util_percent'),
        host_RAM_available_min_MiB=min(row['RAM_available_MiB'] for row in rows),
        filesystem_available_min_MiB=min(row['disk_available_MiB'] for row in rows),
        host_procfs_is_not_container_or_process_quota=True,
        GPU_device_samples_are_not_individual_process_peaks=True,
        unsampled_training_periods_not_inferred=True,monitor_may_include_queue_wait_and_idle=True)


def generate(configuration):
    roots=validate_roots(configuration);records=enumerate_records(roots)
    failure_certificates=apply_failure_accounting(records,configuration.get('failure_accounting',[]))
    summaries=[]
    for root in roots:
        rows=[row for row in records if row['root']==root['name']]
        bundles=sum(row['category']=='UPSTREAM_ANCHOR_BUNDLE' for row in rows)
        state=('NOT_RUN' if not rows else 'RUNNING' if root['state']=='RUNNING' else
               'STARTED_SINCE_BINDING' if root['state']=='NOT_RUN' else 'CLOSED_ACCOUNTING')
        summaries.append(dict(name=root['name'],kind=root['kind'],role=root['role'],declared_execution_state=root['state'],
            observed_status=state,root_present=root['path'].is_dir(),recorded_attempt_or_bundle_units=len(rows),
            upstream_preparation_bundle_units=bundles,attempt_units_excluding_preparation_bundles=len(rows)-bundles,
            expected_units=root.get('expected_units'),counts=sum_counts(rows),
            sum_per_unit_elapsed_seconds=sum(float(r['unit_elapsed_seconds']) for r in rows if r['unit_elapsed_seconds'] is not None),
            units_without_recorded_elapsed_seconds=sum(r['unit_elapsed_seconds'] is None for r in rows),
            units_without_recorded_process_GPU_peak=sum(r['process_PyTorch_peak_GPU_bytes'] is None for r in rows),
            process_PyTorch_peak_GPU_bytes=max((r['process_PyTorch_peak_GPU_bytes'] for r in rows if r['process_PyTorch_peak_GPU_bytes'] is not None),default=None)))
    scheduler=scheduler_reports(configuration.get('schedulers',[]))
    timeline=interval_union([row['_interval'] for row in scheduler if '_interval' in row])
    for row in scheduler:row.pop('_interval',None)
    resources=[resource_report(item) for item in configuration.get('resources',[])]
    primary=[row for row in records if row['role']=='PRIMARY'];historical=[row for row in records if row['role']=='HISTORICAL']
    duplicate_artifacts={}
    for row in records:
        metadata=row.get('saved_payload_metadata') or {}
        digest=metadata.get('declared_sha256',metadata.get('declared_checkpoint_sha256'))
        if digest:duplicate_artifacts.setdefault(digest,[]).append(row['receipt_sha256'])
    duplicate_groups=sum(len(value)>1 for value in duplicate_artifacts.values())
    # A true separately started deterministic fit can have identical final bytes.
    # Never silently deduplicate distinct physical attempt directories by weights.
    for row in records:row.pop('_directory',None);row.pop('_start_unix',None)
    result=dict(schema='bgb_en_actual_execution_metadata_ledger_v1',
        status='RUNNING' if any(row['observed_status'] in ('RUNNING','STARTED_SINCE_BINDING') for row in summaries)
            or any(row['counts']['unfinished_attempts'] for row in records) else 'CLOSED_METADATA_ACCOUNTING_NOT_SCIENTIFIC_SUCCESS',
        primary_counts=sum_counts(primary),historical_counts=sum_counts(historical),all_recorded_counts=sum_counts(records),
        roots=summaries,records=records,schedulers=scheduler,resource_windows=resources,
        failure_accounting_certificates=failure_certificates,
        exact_global_makespan=timeline,
        global_makespan_status='AVAILABLE_FOR_EXPLICITLY_TIMESTAMPED_SCHEDULER_SCOPES_ONLY' if timeline else 'UNKNOWN_NO_COMMON_RECORDED_WALLCLOCK_BOUNDS',
        distinct_attempts_with_identical_declared_artifact_hash_groups=duplicate_groups,
        deduplication='canonical metadata directory paths once across overlapping roots; most-specific explicit root assigns role; distinct fresh attempts are not merged by equal weights',
        external_replay_reports_counted_as_fits=False,
        failed_updates_boundary='unknown unless FAILED metadata explicitly saved a completed-update count; planned epochs are never substituted',
        successful_update_boundary='neural/source one-complete-riskset-update-per-epoch under frozen drivers; Cox/beta solver iterations are a different quantity and not inferred',
        unit_time_boundary='sum per-fit elapsed is compute exposure, not parallel makespan; scheduler scope elapsed is reported separately',
        artifact_verification_boundary='receipt and marker bytes are hashed; declared artifact hashes bound, payload existence checked; weights not deserialized or rehashed by this metadata-only ledger',
        expected_units_are_not_counted_as_execution=True,model_forward_calls=0,new_optimizer_updates=0,
        decoder_counts_are_known_completed_receipt_lower_bound=True,
        GPU_used=False,private_paths_or_patient_arrays_exported=False,script_sha256=sha(__file__))
    return result


def write_reports(result,output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# BG-BEN 实际执行与资源账本','',f"**状态：{result['status']}。计划、锁、回放和资格测试不当作已完成拟合。**",'',
           '| 类型 | 主结果范围 | 历史/替换前范围 | 全部已记录 |','|---|---:|---:|---:|']
    for key in COUNTS:lines.append(f"| {key} | {result['primary_counts'][key]} | {result['historical_counts'][key]} | {result['all_recorded_counts'][key]} |")
    lines+=['','## 根目录覆盖','','| 匿名范围 | 角色 | 状态 | 尝试目录（不含prepare） | prepare bundle | 预期单元（不计执行） |','|---|---|---|---:|---:|---:|']
    for row in result['roots']:lines.append(f"| {row['name']} | {row['role']} | {row['observed_status']} | {row['attempt_units_excluding_preparation_bundles']} | {row['upstream_preparation_bundle_units']} | {row['expected_units']} |")
    lines+=['','同一物理元数据目录只计一次；重叠根由更具体的显式根决定主结果或历史角色。原F7成功归档仍是实际历史拟合，但不进入当前科学主结果。另一次全新同seed拟合即使权重相同，也不因artifact相同被静默合并。来源16个拟合只由其真实COMPLETE计数，不叠加旧14/新16回放摘要。',
            '', '## 更新数与时间边界','',
            '失败/中断单元若未保存已完成更新数，更新数标为未知；不能用80/160/320计划补齐。神经和来源完整拟合依其冻结的一轮一次完整风险集更新计算已记录更新；Cox和β的solver迭代不能与这些更新混算。解析风险decoder另外计数。',
            '', '已记录拟合耗时之和是计算暴露，不是并行日历耗时；缺少耗时的失败或中断另报数量，不能补0或假装该和覆盖全部成本。scheduler真实elapsed独立列出；只有配置显式提供并匹配已记录elapsed的共同时间边界，才汇总覆盖区间。当前全局时间状态：**'+result['global_makespan_status']+'**。',
            '', '部分原scheduler只把成功返回写入units；其中非零返回数为0不代表没有失败，阶段failure_entries及真实FAILED另列。',
            '', '| 调度范围 | 状态 | scheduler elapsed(s) | worker耗时和(s) | 非零返回数 | 明列failure_entries |','|---|---|---:|---:|---:|---:|']
    for row in result['schedulers']:
        lines.append(f"| {row['name']} | {row['status']} | {row.get('observed_scheduler_elapsed_seconds')} | {row.get('sum_reported_unit_elapsed_seconds')} | {row.get('reported_nonzero_returncodes')} | {row.get('explicitly_recorded_failure_entries')} |")
    lines+=['','## 实际资源采样窗口','','| 窗口 | 样本数 | 覆盖秒 | GPU利用率时间均值 | 设备显存峰值MiB |','|---|---:|---:|---:|---:|']
    for row in result['resource_windows']:
        lines.append(f"| {row['name']} | {row.get('samples',0)} | {row.get('observed_window_seconds')} | {row.get('GPU_util_time_weighted_mean_percent')} | {row.get('sampled_device_GPU_memory_peak_MiB')} |")
    lines+=['','仅对真实采样窗口报告资源；长于三个采样中位间隔的空档不插值成持续高利用率。nvidia-smi设备显存不等于单进程PyTorch峰值；CPU/RAM来自host procfs，不能称容器或进程quota。未采样训练期不推测，窗口可能包含排队与空闲。',
            '', '优势：引用、真正优化、历史成功、失败及未知更新分别计数，计划和回放不会膨胀执行数。限制：账本只验证收据字节和声明artifact身份及存在性，不加载/重新哈希权重；科学有效性仍依独立检查点回放。元数据缺失时只能保留未知，不能恢复未记录的失败epoch或完整全局makespan。',
            '', '本脚本没有模型前向、优化器或GPU调用，不输出私有根目录、患者标识、权重或风险数组。']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True,type=Path,help='private JSON: roots, schedulers, resources; paths never exported')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en/EXECUTION_LEDGER.json')
    args=parser.parse_args();result=generate(read(args.inputs));result['private_input_binding_sha256']=sha(args.inputs)
    write_reports(result,args.output)
    print(json.dumps(dict(status=result['status'],all_recorded_counts=result['all_recorded_counts'],GPU_used=False,new_optimizer_updates=0)))


if __name__=='__main__':main()

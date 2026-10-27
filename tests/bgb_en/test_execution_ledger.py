import json
from pathlib import Path

import pytest
from scripts import bgb_en_execution_ledger as ledger


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def neural(folder,status='COMPLETE',seed=31):
    write(folder/'START.json',{'start':1000.+seed})
    if status=='COMPLETE':
        (folder/'model.pt').write_bytes(b'no deserialization')
        write(folder/'COMPLETE.json',dict(status='COMPLETE',stage='R3',arm='F7',seed=seed,epochs=80,
            actual_optimizer_fits=1,artifact_sha256='a'*64,elapsed_seconds=10.,peak_GPU_bytes=2048))
    elif status=='FAILED':write(folder/'FAILED.json',{'status':'FAILED','type':'RuntimeError'})


def test_overlapping_roots_do_not_double_count_and_preparation_adds_beta(tmp_path):
    root=tmp_path/'core';neural(root/'cells'/'case')
    write(root/'prepared'/'inner.json',dict(status='COMPLETE',sha256='b'*64,upstream_optimizer_fits=13,
        upstream_preprocessing_fits=4,scalar_beta=.2,elapsed_seconds=5.))
    (root/'prepared'/'inner.pt').write_bytes(b'not loaded')
    config={'roots':[dict(name='core',kind='core',path=str(root),state='CLOSED'),
                     dict(name='overlap',kind='extra',path=str(root/'cells'),state='CLOSED')]}
    value=ledger.generate(config);count=value['all_recorded_counts']
    assert len(value['records'])==2 and count['completed_neural_fits']==1
    assert count['completed_Cox_fits']==13 and count['completed_beta_fits']==1
    assert count['known_neural_external_optimizer_updates']==80
    assert str(tmp_path) not in json.dumps(value)


def test_history_attempts_are_real_cost_but_not_primary_and_future_not_counted(tmp_path):
    core=tmp_path/'core';archive=core/'archived'
    neural(archive/'old_complete',seed=31);neural(archive/'old_failed','FAILED',47)
    config={'roots':[dict(name='core',kind='core',path=str(core),state='RUNNING',expected_units=6),
        dict(name='old',kind='archive',path=str(archive),role='HISTORICAL',state='CLOSED'),
        dict(name='future',kind='extra',path=str(tmp_path/'future'),state='NOT_RUN',expected_units=6)]}
    value=ledger.generate(config)
    assert value['primary_counts']['completed_neural_fits']==0
    assert value['historical_counts']['completed_neural_fits']==1
    assert value['historical_counts']['failed_attempts']==1
    assert value['historical_counts']['unknown_failed_or_unfinished_update_counts']==1
    assert next(row for row in value['roots'] if row['name']=='future')['observed_status']=='NOT_RUN'


def test_source_receipts_count_once_and_replay_summary_adds_no_fits(tmp_path):
    root=tmp_path/'source'
    for index in range(2):
        folder=root/f'fit_{index:02d}';write(folder/'STARTED.json',{'fit_index':index})
        write(folder/'COMPLETE.json',dict(status='COMPLETE_FIXED_SOURCE_FIT',actual_new_supervised_fits=1,
            epochs=20,optimizer_updates=20,artifacts={'checkpoint.pt':str(index)*64}))
        (folder/'checkpoint.pt').write_bytes(b'not loaded')
    write(root/'EXTERNAL_CHECKPOINT_REPLAY.json',{'actual_recorded_source_fits':14})
    write(root/'EXTERNAL_CHECKPOINT_REPLAY_WITH_CONTROLS.json',{'actual_recorded_source_fits':16})
    value=ledger.generate({'roots':[dict(name='source',kind='source',path=str(root),state='CLOSED')]})
    assert value['all_recorded_counts']['completed_external_fits']==2
    assert value['all_recorded_counts']['known_neural_external_optimizer_updates']==40


def test_scheduler_parallel_elapsed_and_actual_monitor_window_are_separate(tmp_path):
    core=tmp_path/'empty';core.mkdir()
    scheduler=write(tmp_path/'scheduler.json',dict(status='COMPLETE',elapsed_seconds=10.,units=[
        {'returncode':0,'seconds':8.},{'returncode':0,'seconds':8.}]))
    resource=tmp_path/'resource.csv'
    resource.write_text('elapsed_seconds,GPU_util_percent,GPU_memory_MiB,GPU_total_MiB,CPU_util_percent,RAM_available_MiB,disk_available_MiB\n'
        '0,0,100,1000,10,3000,5000\n10,100,200,1000,20,2000,4000\n20,100,150,1000,30,1000,3000\n')
    value=ledger.generate({'roots':[dict(name='empty',kind='core',path=str(core),state='CLOSED')],
        'schedulers':[dict(name='parallel',path=str(scheduler))],
        'resources':[dict(name='observed',path=str(resource))]})
    assert value['schedulers'][0]['observed_scheduler_elapsed_seconds']==10
    assert value['schedulers'][0]['sum_reported_unit_elapsed_seconds']==16
    assert value['exact_global_makespan'] is None
    monitor=value['resource_windows'][0]
    assert monitor['observed_window_seconds']==20
    assert monitor['GPU_util_time_weighted_mean_percent']==75
    assert monitor['host_procfs_is_not_container_or_process_quota'] is True


def test_nine_original_pca_attempts_plus_two_precision_retries_are_eleven(tmp_path):
    original=tmp_path/'original';repair=tmp_path/'repair';starts={};successes={}
    failed={(128,2),(256,0)}
    for dimension in (32,128,256):
        for part in (0,1,2):
            label=f'PCA{dimension}_UNWHITENED_p{part}';folder=original/'cells'/label
            start=write(folder/'START.json',{'dimension':dimension,'partition':part});starts[label]=ledger.sha(start)
            if (dimension,part) not in failed:
                done=write(folder/'COMPLETE.json',dict(status='COMPLETE',actual_optimizer_fits=1,artifact_sha256='c'*64))
                (folder/'model.pt').write_bytes(b'not loaded');successes[label]=ledger.sha(done)
    for dimension,part in failed:
        folder=repair/'cells'/f'PCA{dimension}_UNWHITENED_p{part}'
        write(folder/'START.json',{'dimension':dimension,'partition':part,'precision':'strict'})
        write(folder/'COMPLETE.json',dict(status='COMPLETE',actual_optimizer_fits=1,artifact_sha256='d'*64))
        (folder/'model.pt').write_bytes(b'not loaded')
    certificate=write(repair/'INITIAL_FAILURE_ACCOUNTING.json',dict(original_solver_attempts=9,original_valid_units=7,
        original_failed_units=2,original_START_sha256=starts,original_complete_sha256=successes,
        failed_units=[{'dimension':128,'partition':2},{'dimension':256,'partition':0}]))
    value=ledger.generate({'roots':[dict(name='original',kind='pca',path=str(original),state='CLOSED'),
        dict(name='precision',kind='pca',path=str(repair),state='CLOSED')],
        'failure_accounting':[dict(name='pca_original_failures',root='original',path=str(certificate))]})
    assert len(value['records'])==11
    assert value['all_recorded_counts']['completed_Cox_fits']==9
    assert value['all_recorded_counts']['failed_attempts']==2
    assert value['all_recorded_counts']['unfinished_attempts']==0
    assert value['all_recorded_counts']['unknown_failed_or_unfinished_update_counts']==2


def test_distinct_fresh_attempts_not_merged_by_equal_declared_weights(tmp_path):
    core=tmp_path/'core';neural(core/'cells'/'one',seed=31);neural(core/'cells'/'two',seed=47)
    value=ledger.generate({'roots':[dict(name='core',kind='core',path=str(core),state='CLOSED')]})
    assert value['all_recorded_counts']['completed_neural_fits']==2
    assert value['distinct_attempts_with_identical_declared_artifact_hash_groups']==1

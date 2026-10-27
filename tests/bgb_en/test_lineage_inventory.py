import json
from pathlib import Path

import pytest

from scripts import bgb_en_lineage_inventory as audit


def task(tmp_path, script='bgb_en_omics_sensitivity.py'):
    return dict(label='omics_fixed_p0_s31',script='scripts/'+script,
        args=['--private',str(tmp_path),'--partition','0','--seed','31','--variant','O8_F2'])


def test_no_start_and_started_are_not_completed(tmp_path):
    spec=task(tmp_path);core={'source':{'scripts/example.py':'0'*64}}
    assert audit.inspect_task(spec,core,{})['status']=='NOT_STARTED'
    folder,_=audit.task_cell(spec);folder.mkdir(parents=True)
    (folder/'START.json').write_text('{}')
    assert audit.inspect_task(spec,core,{})['status']=='STARTED_NO_COMPLETE'
    (folder/'FAILED.json').write_text('{}')
    assert audit.inspect_task(spec,core,{})['status']=='FAILED'


def test_status_audit_validates_artifact_and_sources_without_accessing_metrics(tmp_path,monkeypatch):
    spec=task(tmp_path);folder,_=audit.task_cell(spec);folder.mkdir(parents=True)
    (folder/'model.pt').write_bytes(b'private-checkpoint-not-unpickled')
    core={'source':{'scripts/example.py':'0'*64}}
    row=dict(status='COMPLETE',outer_test_evaluated=False,inner_partition=0,seed=31,variant='O8_F2',epochs=80,
        artifact_sha256=audit.sha256(folder/'model.pt'),
        lineage=dict(core_lineage=core,auxiliary_script_sha256='script-hash',prepared_sha256='prepared',variant='O8_F2'))
    (folder/'COMPLETE.json').write_text(json.dumps(row))
    original_sha=audit.sha256
    monkeypatch.setattr(audit,'sha256',lambda path: 'script-hash' if Path(path).name==Path(spec['script']).name else original_sha(path))
    calls=[];monkeypatch.setattr(audit,'source_guard',lambda mapping:calls.append(mapping))
    out=audit.inspect_task(spec,core,{'0':'prepared'})
    assert out['status']=='COMPLETE_VERIFIED_METADATA_AND_SOURCE' and calls==[core['source']]
    assert out['performance_fields_exported'] is False
    assert 'metrics' not in out and 'risk' not in out
    (folder/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='ARTIFACT_SHA_CHANGED'):
        audit.inspect_task(spec,core,{'0':'prepared'})


def test_paths_and_source_maps_fail_closed(tmp_path):
    with pytest.raises(ValueError,match='UNRECOGNIZED_LOCKED_TASK'):
        audit.task_cell(task(tmp_path,'not_a_locked_driver.py'))
    with pytest.raises(ValueError,match='SOURCE_PATH_OUTSIDE'):
        audit.source_guard({'../private.py':'0'*64})
    assert audit.source_maps({'source_data_sha256':'0'*64,'lineage':{'sources':{'src/x.py':'1'*64}}})==[{'src/x.py':'1'*64}]


def test_queue_drained_with_failed_attempt_is_not_renamed_complete(tmp_path,monkeypatch):
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(dict(tasks=[{'label':'accepted'},{'label':'failed'}])))
    def inspect(spec,*_args):
        return dict(unit=spec['label'],group='training_budget',
                    status='COMPLETE_VERIFIED_METADATA_AND_SOURCE' if spec['label']=='accepted' else 'FAILED')
    monkeypatch.setattr(audit,'inspect_task',inspect)
    result=audit.plan_snapshot(path,{}, {})
    assert result['all_attempts_ended'] is True and result['terminated_attempts']==2
    assert result['completed_verified']==1 and result['failed']==1
    assert result['status']=='ALL_ATTEMPTS_ENDED_WITH_ORIGINAL_FAILURES_RETAINED'

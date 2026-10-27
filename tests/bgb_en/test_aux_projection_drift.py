import json
import numpy as np
import pytest
import torch
from scripts import bgb_en_aux_projection_drift as drift


def test_tensor_drift_definitions_and_zero_norm_are_explicit():
    a=np.array([1.,2.],np.float32)
    same=drift.drift_statistics(a,a.copy())
    assert same['delta_RMS']==0 and same['relative_L2']==0
    assert same['cosine']==pytest.approx(1.)
    changed=drift.drift_statistics(a,2*a)
    assert changed['relative_L2']==pytest.approx(1.)
    assert changed['delta_RMS']==pytest.approx(np.sqrt(2.5))
    assert changed['cosine']==pytest.approx(1.)
    zero=drift.drift_statistics(np.zeros(2),a)
    assert zero['relative_L2'] is None and zero['cosine'] is None
    assert zero['relative_L2_status']=='NA_ZERO_SOURCE_NORM'
    with pytest.raises(ValueError,match='finite'):
        drift.drift_statistics(a,np.array([1.,np.nan]))


def fixture(tmp_path):
    source={'0.weight':np.ones((2,2),np.float32),'0.bias':np.array([.1,.2],np.float32)}
    origin={'source':{'science.py':'a'*64}}
    audit={'source_transfer_sha256':'b'*64}
    lineage={**origin,'shared_teacher_origin_lineage':origin,'external_projection_sha256':'b'*64,
             'UNI2_cache_receipt_sha256':'c'*64}
    metrics={'Harrell':.5}
    state={f'path.project.{key}':torch.tensor(value*1.1) for key,value in source.items()}
    saved=dict(lineage=lineage,arm='F11_UNI2',stage='R3',seed=31,epochs=80,optimizer_updates=80,
               snapshots={80:{'state_dict':state,'metrics':metrics}})
    torch.save(saved,tmp_path/'model.pt')
    row=dict(status='COMPLETE',stage='R3',arm='F11_UNI2',inner_partition=0,seed=31,
             outer_fold=1,outer_test_evaluated=False,epochs=80,actual_optimizer_fits=1,
             lineage=lineage,metrics=metrics,artifact_sha256=drift.sha256(tmp_path/'model.pt'))
    path=tmp_path/'COMPLETE.json';path.write_text(json.dumps(row))
    return path,source,audit,row


def test_real_final80_tensor_prefix_and_source_hash_are_bound(tmp_path):
    path,source,audit,row=fixture(tmp_path)
    result=drift.audit_final_cell(path,'TRUE',0,31,source,audit)
    assert len(result['per_tensor'])==2
    assert result['all_projection_parameters']['relative_L2']==pytest.approx(.1,rel=1e-5)
    assert result['source_projection_loaded_by_recorded_lineage'] is True
    with pytest.raises(ValueError,match='initialized'):
        drift.audit_final_cell(path,'TRUE',0,31,source,{'source_transfer_sha256':'d'*64})


def test_changed_or_unfinished_checkpoint_never_enters_drift(tmp_path):
    path,source,audit,row=fixture(tmp_path)
    (tmp_path/'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        drift.audit_final_cell(path,'TRUE',0,31,source,audit)
    row['status']='RUNNING';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='completed'):
        drift.audit_final_cell(path,'TRUE',0,31,source,audit)

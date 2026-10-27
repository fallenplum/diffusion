"""Anonymous auxiliary reporting fails closed on provenance and partial cells."""
import copy
import json
from pathlib import Path

import pytest
from scripts import bgb_en_report_aux as report


def metric(num,den):
    return dict(Harrell_num=num,Harrell_den=den,Uno_num=2*num,Uno_den=2*den,
                Harrell=num/den,Uno60=num/den)


def fake_cell(part,seed,num,den):
    return dict(partition=part,seed=seed,train_n=200,valid_n=100,parameters=1234,
                elapsed_seconds=2.,peak_GPU_bytes=0,**metric(num,den))


def test_concordance_pools_counts_per_seed_before_mean():
    cells={}
    for seed in (31,47):
        for part,(num,den) in enumerate(((1,2),(90,100),(90,100))):
            cells[('TRUE','F5_UNI2',part,seed)]=fake_cell(part,seed,num,den)
    summary,seeds,paired=report.summarize_aux(cells)
    actual=next(row for row in summary if row['arm']=='F5_UNI2')
    assert actual['Harrell']==pytest.approx(181/202)
    assert actual['Harrell']!=pytest.approx((.5+.9+.9)/3)
    assert all(row['Harrell_den']==202 for row in seeds if row['arm']=='F5_UNI2')
    assert all(row['delta_Harrell'] is None for row in paired)


def test_partial_seed_remains_na_and_pair_capacity_mismatch_rejected():
    cells={('TRUE','F5_UNI2',0,31):fake_cell(0,31,5,10)}
    summary,seeds,_=report.summarize_aux(cells)
    assert all(row['Harrell'] is None for row in summary)
    assert all(row['Harrell'] is None for row in seeds)
    changed=copy.deepcopy(cells[('TRUE','F5_UNI2',0,31)])
    changed['parameters']+=1
    cells[('TRUE','F11_UNI2',0,31)]=changed
    with pytest.raises(ValueError,match='capacity'):
        report.summarize_aux(cells)


def source_receipt(tmp_path):
    path=tmp_path/'COMPLETE.json'
    (tmp_path/'checkpoint.pt').write_bytes(b'synthetic weights; never deserialized')
    row=dict(status='COMPLETE_FIXED_SOURCE_FIT',epochs=20,optimizer_updates=20,
             actual_new_supervised_fits=1,artifacts={'checkpoint.pt':report.sha256(tmp_path/'checkpoint.pt')},
             contract=dict(fit_index=0,role='leave_one_donor_out',lineage={
                 'source_data_sha256':'a'*64,'source_manifest_sha256':'b'*64}),
             train_metrics={'program_standardized_Huber':.2,'tissue_balanced_accuracy':.7},
             valid_metrics={'program_standardized_Huber':.4,'tissue_balanced_accuracy':.6})
    path.write_text(json.dumps(row))
    return path


def test_source_raw_record_is_preserved_but_not_transfer_eligible_and_hashes_checked(tmp_path):
    path=source_receipt(tmp_path)
    row=report.verify_source_receipt(path,'RAW_V1',0)
    assert row['transfer_eligible_under_current_unitnorm_contract'] is False
    assert row['independent_heldout_evaluation'] is True
    (tmp_path/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        report.verify_source_receipt(path,'RAW_V1',0)


def test_auxiliary_checkpoint_is_verified_without_model_loading(tmp_path):
    folder=tmp_path/'auxiliary_TRUE_v2'/'cells'/'cell'
    folder.mkdir(parents=True)
    (folder/'model.pt').write_bytes(b'not a pickle; report must only hash')
    values=metric(6,10)
    origin={'source':{'science.py':'a'*64}}
    lineage=copy.deepcopy(origin)
    lineage.update(shared_teacher_origin_lineage=origin,UNI2_cache_receipt_sha256='c'*64,
                   auxiliary_source={'wrapper.py':'d'*64},external_projection_sha256=None)
    row=dict(status='COMPLETE',stage='R3',arm='F5_UNI2',seed=31,inner_partition=0,
             outer_fold=1,outer_test_evaluated=False,epochs=80,actual_optimizer_fits=1,
             curves=[{'epoch':80,'valid':values}],metrics=values,lineage=lineage,
             artifact_sha256=report.sha256(folder/'model.pt'),train_n=200,valid_n=100,
             parameters=1234,elapsed_seconds=2.,peak_GPU_bytes=0)
    (folder/'COMPLETE.json').write_text(json.dumps(row))
    cells,binding=report.load_aux_cells(tmp_path,{})
    assert len(cells)==1 and binding['UNI2_cache_READY_sha256']=='c'*64
    row['lineage']['source']['science.py']='e'*64
    (folder/'COMPLETE.json').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='relabelled'):
        report.load_aux_cells(tmp_path,{})

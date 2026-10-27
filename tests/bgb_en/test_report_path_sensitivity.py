import json

import numpy as np
import pytest
from scripts import bgb_en_report_path_sensitivity as report


def cell(part,seed,num,den):
    return dict(partition=part,seed=seed,train_n=200,valid_n=100,parameters=1234,elapsed_seconds=2.,
                Harrell_num=num,Harrell_den=den,Uno_num=2*num,Uno_den=2*den,Harrell=num/den,Uno60=num/den)


def test_risk_distribution_does_not_return_patient_vectors():
    value=report.risk_summary(np.array([1.,2.,3.]),np.array([.1,.1,.1]))
    assert value['full_SD']==pytest.approx(np.std([1.,2.,3.]))
    assert value['anchor_mean']==pytest.approx(1.9)
    assert value['delta_q50']==pytest.approx(.1)
    assert value['delta_SD_over_anchor_SD']==pytest.approx(0.,abs=1e-15)
    assert all(item is None or np.isscalar(item) for item in value.values())
    with pytest.raises(ValueError,match='invalid'):
        report.risk_summary(np.array([np.nan]),np.array([0.]))


def test_full_seed_uses_numden_and_incomplete_sensitivity_is_na():
    cells={}
    for seed in (31,47):
        for part,(num,den) in enumerate(((1,2),(90,100),(90,100))):
            cells[('CONCH_B512','P1',part,seed)]=cell(part,seed,num,den)
    summary,seeds,pairs=report.summarize(cells)
    baseline=next(row for row in summary if row['scenario']=='CONCH_B512' and row['arm']=='P1')
    assert baseline['Harrell']==pytest.approx(181/202)
    assert all(row['delta_Harrell'] is None for row in pairs)
    assert all(row['Harrell'] is None for row in seeds if row['scenario']!='CONCH_B512' or row['arm']!='P1')


def test_sensitivity_requires_exact_prefit_lock_and_original_teacher(tmp_path):
    origin={'source':{'science.py':'a'*64}}
    locked={'shared_teacher_origin_lineage':origin,'sensitivity_sources':{'wrapper.py':'b'*64},
            'cases':{'CONCH_B128':{'READY_sha256':'c'*64}}}
    metrics={key:cell(0,31,6,10)[key] for key in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den')}
    (tmp_path/'model.pt').write_bytes(b'only byte hashing at receipt qualification')
    lineage={**origin,'path_budget_sensitivity':dict(scenario='CONCH_B128',base_arm='P1',lock_sha256='d'*64,
        shared_teacher_origin_lineage=origin,sources=locked['sensitivity_sources'],selected_cache_READY_sha256='c'*64)}
    row=dict(status='COMPLETE',stage='R1',arm='P1',inner_partition=0,seed=31,outer_fold=1,
             outer_test_evaluated=False,epochs=80,actual_optimizer_fits=1,metrics=metrics,
             curves=[{'epoch':80,'valid':metrics}],lineage=lineage,artifact_sha256=report.sha256(tmp_path/'model.pt'))
    path=tmp_path/'COMPLETE.json';path.write_text(json.dumps(row))
    assert report.verify_cell(path,'CONCH_B128','P1',0,31,locked,'d'*64)[1]==tmp_path/'model.pt'
    with pytest.raises(ValueError,match='prefit'):
        report.verify_cell(path,'CONCH_B128','P1',0,31,locked,'e'*64)
    row['lineage']['source']={'science.py':'f'*64};path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='teacher'):
        report.verify_cell(path,'CONCH_B128','P1',0,31,locked,'d'*64)


def test_paired_budget_denominators_must_match():
    cells={('CONCH_B128','P1',0,31):cell(0,31,5,10),('CONCH_B512','P1',0,31):cell(0,31,6,11)}
    with pytest.raises(ValueError,match='IPCW'):
        report.summarize(cells)

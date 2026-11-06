"""无需TCGA或任何第三方病理权重的版本合约检查。"""
from pathlib import Path
import json
import numpy as np
from r4surv.cli import run_demo
from r4surv.model import load_model,predict_model
from r4surv.io import read_npz
from r4surv.reference import efron_weights,evaluate


def test_efron_tied_events():
    t=np.array([1.,1.,2.,3.]);e=np.array([1,1,0,1]);a=efron_weights(t,e)
    assert a.shape==(3,4)
    np.testing.assert_allclose(a[0],[1,1,1,1]);np.testing.assert_allclose(a[1],[.5,.5,1,1])


def test_demo_end_to_end(tmp_path):
    result=run_demo(tmp_path,n=144,seed=20261107)
    assert result['status']=='SYNTHETIC_SMOKE_TEST_ONLY'
    assert result['test_patients']>0
    assert result['dataset']['events']>0
    assert (tmp_path/'r4_model.json').is_file()
    assert (tmp_path/'synthetic_predictions.csv').is_file()
    raw=read_npz(tmp_path/'synthetic_features.npz',require_outcome=True)
    model=load_model(tmp_path/'r4_model.json')
    pred=predict_model(raw,model)
    assert len(pred['risk_score'])==144
    np.testing.assert_allclose(pred['risk_score'],pred['cp_risk']+model['beta']*pred['rna_risk'],rtol=0,atol=1e-12)
    assert np.std(pred['risk_score'])>0
    summary=json.loads((tmp_path/'demo_receipt.json').read_text())
    assert summary['explicit_no_TCGA_result'] is True


def test_input_schema_rejects_invalid(tmp_path):
    from r4surv.io import read_npz
    import pytest
    f=tmp_path/'invalid.npz';np.savez_compressed(f,patient_id=np.array(['fake']))
    with pytest.raises(ValueError,match='缺少字段'):
        read_npz(f)

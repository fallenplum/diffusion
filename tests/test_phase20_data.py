"""Fail-closed inherited data access and exact offsets without new solvers."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.phase20_selective import data
from src.phase10_trimodal.data import digest


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(data.prior,'P',tmp_path)
    monkeypatch.setattr(data,'cfg',lambda:{'upstream_source':'prior-source'})
    f={'X':np.arange(42,dtype=np.float32).reshape(6,7)/100,'CS':np.arange(18,dtype=np.float32).reshape(6,3)/100}
    fit=np.arange(4);valid=np.array([4,5]);audit={'source':'prior-source','wrong_controls':{'seed':31}}
    ds=SimpleNamespace(role='inner',fold=1,f=f,ft={},fit=fit,valid=valid,upstream_audit=audit,
                       visual=torch.empty(1))
    models={}
    for arm,key in [('J0','X'),('J_CS','CS')]:
        w=np.linspace(-.2,.1,f[key].shape[1]);models[arm]={'w':w,'risk':np.asarray(f[key][valid]@w,np.float32)}
    bank={'source':'prior-source','fit':fit,'valid':valid,'preprocess_audit':{'source':'prior-source'},'models':models}
    path=tmp_path/'simple/inner_fold1.pt';path.parent.mkdir()
    torch.save(bank,path)
    path.with_suffix('.json').write_text(json.dumps({'source':'prior-source','checkpoint_sha256':digest(path)}))
    return ds,path


def test_exact_offset_keeps_original_valid_values_and_train_only_population_scale(tmp_path,monkeypatch):
    ds,path=fixture(tmp_path,monkeypatch)
    data.add_anchor(ds)
    np.testing.assert_array_equal(ds.f['ANCHOR'][ds.valid],ds.inherited_simple['models']['J0']['risk'])
    fit_risk=ds.f['X'][ds.fit].astype(float)@ds.anchor_w
    assert ds.anchor_sd_fit==float(np.std(fit_risk,ddof=0))
    assert ds.anchor_reference['new_convex_solver_calls']==0
    assert ds.f['ANCHOR'].dtype==np.float32
    assert ds.anchor_w.dtype==np.float64


def test_tampered_or_mismatched_reference_fails_instead_of_refitting(tmp_path,monkeypatch):
    ds,path=fixture(tmp_path,monkeypatch)
    bank=torch.load(path,weights_only=False);bank['fit']=np.array([1,2,3,4])
    torch.save(bank,path)
    with pytest.raises(ValueError,match='source/hash'): data.reference_audit(ds)
    path.with_suffix('.json').write_text(json.dumps({'source':'prior-source','checkpoint_sha256':digest(path)}))
    with pytest.raises(ValueError,match='split'): data.reference_audit(ds)


def test_missing_parent_preprocessing_never_invokes_parent_builder(tmp_path,monkeypatch):
    monkeypatch.setattr(data.prior,'P',tmp_path)
    monkeypatch.setattr(data.prior,'source',lambda:'prior-source')
    monkeypatch.setattr(data,'cfg',lambda:{'upstream_source':'prior-source'})
    def forbidden(*args,**kwargs): raise AssertionError('parent builder invoked')
    monkeypatch.setattr(data.prior.Data,'__init__',forbidden)
    with pytest.raises(ValueError,match='no automatic refit'): data.Data(1,'inner',31,device='cpu')


def test_source_identity_requires_all_training_tests_and_binds_their_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'ROOT',tmp_path);monkeypatch.setattr(data.prior,'ROOT',tmp_path)
    monkeypatch.setattr(data.prior,'source_files',lambda:{})
    config_path=tmp_path/'configs/phase20_selective_attention.json'
    config_path.parent.mkdir();config_path.write_text('{}')
    monkeypatch.setattr(data,'CONFIG',config_path)
    (tmp_path/'src/phase20_selective').mkdir(parents=True)
    (tmp_path/'src/phase20_selective/__init__.py').write_text('')
    (tmp_path/'scripts').mkdir();(tmp_path/'scripts/run_phase20.py').write_text('')
    (tmp_path/'tests').mkdir()
    for name in data.TRAINING_TESTS[:-1]:
        (tmp_path/f'tests/test_phase20_{name}.py').write_text('')
    with pytest.raises(ValueError,match='required training source missing'): data.source()
    target=tmp_path/f'tests/test_phase20_{data.TRAINING_TESTS[-1]}.py'
    target.write_text('')
    before=data.source();target.write_text('# changed qualification\n')
    assert before!=data.source()

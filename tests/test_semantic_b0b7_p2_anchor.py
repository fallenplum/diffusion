"""Synthetic full-pipeline teacher isolation without a real R4 optimizer fit."""
import numpy as np
import pytest

from src.semantic_b0b7.p2_anchor import (
    LEGACY_VISUAL_SOURCE_SHA256,array_sha,fit_r4_anchor,
)


def raw_fixture():
    rng=np.random.default_rng(17);n=96;g=600
    return {"patient_ids":np.arange(n).astype(str),"folds":np.resize(np.arange(1,6),n),
            "t":np.linspace(1,100,n),"e":(np.arange(n)%3==0).astype(int),"valid":np.ones(n,bool),
            "S32":rng.normal(size=(n,32)),"rna":rng.normal(size=(n,g)),"cna":rng.normal(size=(n,g)),
            "rna_symbols":np.array(["G"+str(i) for i in range(g)]),
            "cna_symbols":np.array(["G"+str(i) for i in range(g)]),
            "clinical_raw":{"AGE":np.full(n,"60")},
            "conch_local":rng.normal(size=(n,768)),"conch_mean":rng.normal(size=(n,768)),
            "_direct_gene_universe":np.ones(g,bool),
            "audit":{"visual_sources_sha256":LEGACY_VISUAL_SOURCE_SHA256}}


def fake_backend(call_log):
    def backend(raw,fit,held,membership,seed):
        assert np.isnan(raw["t"][held]).all() and np.isnan(raw["e"][held]).all()
        assert np.isfinite(raw["t"][fit]).all() and np.isfinite(raw["e"][fit]).all()
        call_log.append({"fit_ids":set(raw["patient_ids"][fit]),"held_ids":set(raw["patient_ids"][held]),"seed":seed})
        # A beta genuinely depending on every permitted fit label makes the
        # isolation counterexample nontrivial; no real Cox fit is credited.
        beta=float(np.mean(raw["e"][fit]+.001*raw["t"][fit]))
        cp=raw["conch_local"][:,0]+.01*np.mean(raw["t"][fit])
        gene=raw["rna"][:,1]
        return {"base":cp+beta*gene,"beta":beta,"features":{"state":{}},
                "teachers":[],"full_models":{},"scalar_audit":{},"risk_distribution":{},
                "pair_teacher_rule":"synthetic_backend_only","full_risks":{"LOCAL":cp,"R":gene},
                "upstream_optimizer_fits":13,"upstream_preprocessing_fits":4}
    return backend


def fit(raw,log,*,crossfit=True):
    return fit_r4_anchor(raw,np.arange(72),np.arange(72,96),np.ones((50,600),bool),
                         expected_union_sha256=array_sha(raw["_direct_gene_universe"]),
                         backend=fake_backend(log),teacher_seed=27142,pipeline_crossfit_seed=27142,
                         downstream_crossfit=crossfit)


def test_own_held_labels_cannot_change_full_pipeline_crossfit_offset():
    raw=raw_fixture();calls=[];base=fit(raw,calls)
    held=base.state["crossfit"][0]["held_rows"]
    changed=dict(raw);changed["t"]=raw["t"].copy();changed["e"]=raw["e"].copy()
    changed["t"][held]+=500;changed["e"][held]=1-changed["e"][held]
    other=fit(changed,[])
    assert np.array_equal(base.training_offset[held],other.training_offset[held])
    assert not np.array_equal(base.full_fit_train_risk_diagnostic[held],other.full_fit_train_risk_diagnostic[held])
    assert len(calls)==4 and base.receipt["Cox_optimizer_fits"]==52
    assert base.receipt["scalar_pair_optimizer_fits"]==4


def test_evaluation_labels_do_not_change_any_fit_or_offset():
    raw=raw_fixture();base=fit(raw,[])
    changed=dict(raw);changed["t"]=raw["t"].copy();changed["e"]=raw["e"].copy()
    changed["t"][72:]+=999;changed["e"][72:]=1-changed["e"][72:]
    other=fit(changed,[])
    assert np.array_equal(base.training_offset,other.training_offset)
    assert np.array_equal(base.evaluation_offset,other.evaluation_offset)


def test_b0_exact_evaluation_is_shared_full_pipeline_and_train_diagnostic_is_not_claimed_oof():
    raw=raw_fixture();full=fit(raw,[]);exact=fit(raw,[],crossfit=False)
    assert np.array_equal(full.b0_exact_evaluation,exact.evaluation_offset)
    assert exact.receipt["Cox_optimizer_fits"]==13 and exact.receipt["scalar_pair_optimizer_fits"]==1
    assert not exact.receipt["beta_supervision_excluded_for_each_downstream_training_patient"]
    assert exact.receipt["training_offset_is_R4_old_recipe_diagnostic_only"]


def test_wrong_visual_or_redefined_union_cannot_be_called_exact_r4():
    raw=raw_fixture();raw["audit"]={"visual_sources_sha256":"not_original_GRID"}
    with pytest.raises(ValueError,match="ORIGINAL_AUDITED_GRID"):
        fit(raw,[])
    raw=raw_fixture()
    with pytest.raises(ValueError,match="LEGACY_EIGHT_PROGRAM_UNION"):
        fit_r4_anchor(raw,np.arange(72),np.arange(72,96),np.ones((50,600),bool),
                      expected_union_sha256="0"*64,backend=fake_backend([]))


def test_nested_teacher_never_even_receives_other_outer_rows():
    raw=raw_fixture();log=[]
    fit_r4_anchor(raw,np.arange(60),np.arange(60,80),np.ones((50,600),bool),
                  expected_union_sha256=array_sha(raw["_direct_gene_universe"]),backend=fake_backend(log))
    assert all((x["fit_ids"]|x["held_ids"])<=set(raw["patient_ids"][:80]) for x in log)

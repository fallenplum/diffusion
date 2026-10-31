"""Synthetic P2 risk qualification; no cohort or outer-test input is loaded."""
from dataclasses import replace
import numpy as np
import pytest

from src.semantic_b0b7.p2_survival import (
    BlockProjection,CompactDesign,EfronProblem,FeatureContract,OffsetCox,
    evaluate_safe,fit_offset_cox,fit_risk_cell,replay_risk_cell,
    risk_repair_damage,select_lambda,frozen_feature_views,
)
from src.semantic_b0b7.survival import efron_numpy


def features(n=42,seed=1):
    r=np.random.default_rng(seed)
    return {k:r.normal(size=(n,18 if k!="mechanism" else 2))
            for k in ("pathology","omics","clinical","mechanism")}


def test_compact_design_has_26_real_columns_and_evaluation_cannot_change_it():
    train=features();evaluation=features(12,2)
    state=CompactDesign.fit(train)
    z=state.transform(train)
    assert z.shape==(42,26)
    assert np.allclose(z.mean(0),0,atol=1e-12) and np.allclose(z.std(0),1)
    before={k:v.components.copy() for k,v in state.blocks.items()}
    state.transform({k:1000*v for k,v in evaluation.items()})
    assert all(np.array_equal(state.blocks[k].components,before[k]) for k in before)
    assert state.receipt["active_nonconstant_columns"]==26


def test_exact_eight_input_columns_are_valid_without_padding():
    x=np.random.default_rng(5).normal(size=(30,8))
    state=BlockProjection.fit(x,8)
    assert state.transform(x).shape==(30,8)
    with pytest.raises(ValueError,match="NO_PADDING"):
        BlockProjection.fit(x[:,:7],8)


def test_mechanism_zero_padding_or_collapsed_blocks_are_rejected():
    x=features();x["mechanism"][:,1]=0
    with pytest.raises(ValueError,match="TWO_REAL"):
        CompactDesign.fit(x)
    with pytest.raises(ValueError,match="INACTIVE|RANK_INSUFFICIENT"):
        BlockProjection.fit(np.ones((30,12)),8)


def test_rank_three_mean_embedding_cannot_be_whitened_into_eight_real_columns():
    rng=np.random.default_rng(5)
    latent=rng.normal(size=(60,3))@rng.normal(size=(3,64))
    # Float32 rounding creates tiny nonzero extra singular values, not signal.
    rounded=latent.astype(np.float32).astype(np.float64)
    with pytest.raises(ValueError,match="RELATIVE_RANK_INSUFFICIENT"):
        BlockProjection.fit(rounded,8)


def test_full_efron_cached_gradient_matches_scalar_oracle_and_finite_differences():
    t=np.array([1,1,1,2,3,3,4.]);e=np.array([1,1,0,1,1,0,0])
    r=np.linspace(-.4,.5,7)
    problem=EfronProblem(t,e)
    value,gradient=problem.loss_gradient(r)
    assert value==pytest.approx(efron_numpy(r,t,e),abs=1e-14)
    step=1e-6;fd=[]
    for k in range(len(r)):
        plus=r.copy();minus=r.copy();plus[k]+=step;minus[k]-=step
        fd.append((efron_numpy(plus,t,e)-efron_numpy(minus,t,e))/(2*step))
    assert np.max(np.abs(np.asarray(fd)-gradient))<5e-10


def test_zero_delta_is_exact_anchor_and_convex_head_converges():
    rng=np.random.default_rng(13)
    x=rng.normal(size=(60,26));t=rng.integers(1,100,60).astype(float)
    e=(np.arange(60)%3==0).astype(int);r0=rng.normal(size=60)
    zero=OffsetCox(np.zeros(26),1.,{})
    assert np.array_equal(zero.predict(x,r0),r0)
    model=fit_offset_cox(x,t,e,r0,ridge_lambda=1.)
    assert model.receipt["gradient_max"]<2e-5
    assert model.receipt["parameters"]==26
    assert model.receipt["oracle_loss_abs_error"]<1e-10
    assert not model.receipt["validation_used_for_solver"]


def test_uno_support_failure_is_na_not_zero_or_extrapolation():
    result=evaluate_safe([1,2,3],[1,1,0],[.5,2.5,4],[1,0,0],[3,2,1])
    assert result["Harrell"]==1
    assert result["Uno60"] is None and result["Uno_num"] is None
    assert not result["Uno60_valid"]


def test_lambda_selection_is_lexicographic_and_each_metric_has_its_own_denominator():
    def rows(h,u):
        return [{"Harrell_num":h,"Harrell_den":10,"Uno_num":u,"Uno_den":20,"Uno60_valid":True} for _ in range(3)]
    selected=select_lambda({.1:rows(8,1),1.:rows(7,20),10.:rows(8,15)})
    assert selected["ridge_lambda"]==10.
    tied=select_lambda({.1:rows(8,15),1.:rows(8,15),10.:rows(8,15)})
    assert tied["ridge_lambda"]==10.
    assert not tied["uses_outer_test"]


def test_lambda_selection_uno_missing_uses_harrell_then_stronger_ridge():
    rows=[{"Harrell_num":8,"Harrell_den":10,"Uno_num":None,"Uno_den":None,"Uno60_valid":False} for _ in range(3)]
    selected=select_lambda({.1:rows,1.:rows,10.:rows})
    assert selected["ridge_lambda"]==10.
    assert all(x["Uno60"] is None for x in selected["candidate_summaries"])


def test_repair_minus_damage_equals_paired_harrell_delta():
    t=[1,2,3,4];e=[1,1,0,1];base=[2,3,1,0];risk=[4,2,3,1]
    train_t=[1,2,20,80];train_e=[1,0,1,1]
    result=risk_repair_damage(t,e,risk,base,train_t,train_e)
    a=evaluate_safe(train_t,train_e,t,e,risk)["Harrell"]
    b=evaluate_safe(train_t,train_e,t,e,base)["Harrell"]
    assert result["net_Harrell_change"]==pytest.approx(a-b)


def test_real_synthetic_cell_persisted_head_replay_and_outer_selection_gate():
    train,evaluation=features(54,17),features(15,18)
    t=np.linspace(1,100,54);e=(np.arange(54)%3==0).astype(int)
    et=np.linspace(2,90,15);ee=(np.arange(15)%4==0).astype(int)
    tr_offset=np.linspace(-1,1,54);ev_offset=np.linspace(-.5,.7,15)
    with pytest.raises(ValueError,match="FROZEN_INNER_SELECTION"):
        fit_risk_cell(train,evaluation,t,e,et,ee,tr_offset,ev_offset,ridge_lambda=1.,role="outer_final",bindings={})
    cell=fit_risk_cell(train,evaluation,t,e,et,ee,tr_offset,ev_offset,ridge_lambda=1.,role="synthetic_qualification",bindings={})
    result=replay_risk_cell(cell,evaluation,ev_offset,t,e,et,ee)
    assert result["max_abs_risk_error"]==0
    assert cell.receipt["risk_optimizer_fits"]==1
    assert not cell.receipt["confidence_intervals_computed"]


def test_frozen_views_keep_slot_program_identity_and_own_missing_mask():
    rng=np.random.default_rng(52);n=12
    p=rng.normal(size=(n,8,64));g=rng.normal(size=(n,50,64))
    measured=np.ones((n,50),bool);measured[0,0]=False;g[0,0]=np.nan
    # Deliberately collapse the old pooled G: new main effects must not use it.
    outputs={"additive_image":p.mean(1),"additive_gene":np.zeros((n,64)),
             "additive_clinical":rng.normal(size=(n,64)),
             "pre_residual_visual_tokens":p,"pre_residual_gene_tokens":g,"gene_mask":measured}
    operator=lambda arrays,mu_p,mu_g,arm:np.column_stack([arrays["visual_tokens"].mean((1,2)),
                                                       arrays["gene_tokens"].mean((1,2))])
    result=frozen_feature_views(outputs,None,np.arange(8),np.arange(8,12),arm="B4",operator=operator)
    assert result["train"]["pathology"].shape==(8,512)
    assert result["train"]["omics"].shape==(8,3200)
    assert np.array_equal(result["train"]["pathology"],p[:8].reshape(8,-1))
    assert np.all(result["train"]["omics"][0,:64]==0)
    assert np.isfinite(result["train"]["omics"]).all()
    assert np.any(result["train"]["omics"]!=0)


def test_frozen_background_uses_one_coordinate_system_and_excludes_eval_targets():
    rng=np.random.default_rng(53);n=24
    outputs={"additive_image":rng.normal(size=(n,64)),"additive_gene":rng.normal(size=(n,64)),
             "additive_clinical":rng.normal(size=(n,64)),"visual_tokens":rng.normal(size=(n,8,64)),
             "gene_tokens":rng.normal(size=(n,50,64)),"gene_mask":np.ones((n,50),bool)}
    clinical={"AGE":np.array([str(40+i) for i in range(n)]),
              "SEX":np.array(["F","M"]*(n//2)),
              "AJCC_PATHOLOGIC_TUMOR_STAGE":np.array(["II","III"]*(n//2)),
              "PATH_T_STAGE":np.full(n,"T2"),"PATH_N_STAGE":np.full(n,"N0"),"PATH_M_STAGE":np.full(n,"M0")}
    def operator(arrays,mp,mg,arm):
        assert mp.shape==arrays["visual_tokens"].shape and mg.shape==arrays["gene_tokens"].shape
        return np.column_stack([(arrays["visual_tokens"]-mp).sum((1,2)),(arrays["gene_tokens"]-mg).sum((1,2))])
    first=frozen_feature_views(outputs,clinical,np.arange(18),np.arange(18,24),arm="B5",operator=operator)
    changed={k:v.copy() for k,v in outputs.items()}
    changed["visual_tokens"][18:]+=100;changed["gene_tokens"][18:]-=100
    other=frozen_feature_views(changed,clinical,np.arange(18),np.arange(18,24),arm="B5",operator=operator)
    assert np.array_equal(first["train"]["mechanism"],other["train"]["mechanism"])
    assert first["background"]["audit"]["DSS_read"] is False
    assert first["one_encoder_coordinate_system"] is True

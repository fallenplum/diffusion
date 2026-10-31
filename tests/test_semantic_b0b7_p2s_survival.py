"""Synthetic frozen-probe/readout contracts; no cohort or outer results."""
import copy
import numpy as np
import pytest
import torch

from src.semantic_b0b7 import p2s_survival as p
from src.semantic_b0b7.models import ModelConfig
from src.semantic_b0b7.p1r_models import RepairRepresentation, RepairSpec, frozen_readout_operator


def tokens(n=12,seed=7):
    rng=np.random.default_rng(seed)
    return (rng.normal(size=(n,8,64)),rng.normal(size=(n,50,64)),rng.random((n,50))>.2,
            (rng.random((8,50))>.8).astype(float),rng.normal(size=(2,64)),rng.normal(size=(2,64)))


def main_features(n,seed):
    rng=np.random.default_rng(seed)
    return {"pathology":rng.normal(size=(n,512)),"omics":rng.normal(size=(n,3200)),
            "clinical":rng.normal(size=(n,64))}


def test_locked_grid_has18_distinct_heads54_cox_per_state_and_real_parameter_counts():
    specs=p.readout_specs()
    assert len(specs)==18 and len(specs)*len(p.RIDGE_LAMBDAS)==54
    assert sum(s.mode=="NO_INTER" for s in specs)==2
    assert {s.coefficients for s in specs}=={24,26,28,40,42,44}
    assert p.PRIMARY_LAMBDA==1.
    for invalid in (p.ReadoutSpec(8,2,"B4","NO_INTER"),p.ReadoutSpec(16,4,"SHARED","INTER")):
        with pytest.raises(ValueError):invalid.validate()


def test_attention_is_a_separate_four_head_grid_only_original_inter_r2():
    specs=p.attention_readout_specs()
    assert len(specs)==4 and len(specs)*36*len(p.RIDGE_LAMBDAS)==432
    assert all(s.base_arm!="ATTN_MATCHED" for s in p.readout_specs())
    assert sum(s.mode=="NO_INTER" for s in specs)==2
    for spec in specs:spec.validate()
    for bad in (p.ReadoutSpec(8,4,"ATTN_MATCHED","INTER"),p.ReadoutSpec(8,2,"ATTN_MATCHED","B7"),
                p.ReadoutSpec(16,2,"ATTN_NATIVE","INTER")):
        with pytest.raises(ValueError):bad.validate()
    visual,gene,mask,prior,u,v=tokens(n=2)
    probe=p.FrozenFactorProbe.construct(u,v)
    with pytest.raises(ValueError,match="FROZEN_MECHANISM_SPEC"):
        p.frozen_factor_mechanism(visual,gene,mask,probe,prior,factor_rank=2,mode="INTER",base_arm="ATTN_MATCHED")


def test_supplied_fp32_attention_features_have_normalized_hashes_without_any_fit(monkeypatch):
    train=main_features(48,201);ev=main_features(6,202)
    main=p.MainProjection.fit(train,width=8)
    mechanism=np.random.default_rng(203).normal(size=(48,2)).astype(np.float32)
    evaluation=np.random.default_rng(204).normal(size=(6,2)).astype(np.float32)
    design=p.ReadoutDesign.fit(p.ReadoutSpec(8,2,"ATTN_MATCHED","INTER"),main,train,mechanism)
    np.testing.assert_array_equal(design.transform(train,mechanism),design.transform(train,mechanism.astype(np.float64)))
    assert design.receipt["mechanism_train_input_sha256"]==p.array_sha(mechanism.astype(np.float64))
    assert design.receipt["mechanism_original_train_input_sha256"]==p.array_sha(mechanism)
    assert "original_frozen_FP32_attention" in design.receipt["ATTN_mechanism_source_contract"]
    def no_fit(*args,**kwargs):raise RuntimeError("REACHED_SOLVER_AFTER_VALID_HASH_NO_FIT_RUN")
    monkeypatch.setattr(p,"fit_diagnostic_cox",no_fit)
    with pytest.raises(RuntimeError,match="REACHED_SOLVER_AFTER_VALID_HASH"):
        p.fit_diagnostic_cell(design,train,ev,None,None,None,None,None,None,ridge_lambda=1.,
            train_mechanism=mechanism.astype(np.float64),evaluation_mechanism=evaluation,scope="synthetic_qualification")


def test_fixed_completion_preserves_original_rows_orthogonality_norms_and_global_rng():
    _,_,_,_,u,v=tokens()
    np.random.seed(123);before=np.random.get_state()
    a=p.FrozenFactorProbe.construct(u,v);after=np.random.get_state()
    assert all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y for x,y in zip(before,after))
    np.testing.assert_array_equal(a.u[:2],u);np.testing.assert_array_equal(a.v[:2],v)
    np.testing.assert_allclose(a.u[2:]@u.T,0.,atol=1e-13)
    np.testing.assert_allclose(a.v[2:]@v.T,0.,atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(a.u[2:],axis=1),np.linalg.norm(u,axis=1).mean(),atol=1e-13)
    assert np.linalg.matrix_rank(a.u)==np.linalg.matrix_rank(a.v)==4
    b=p.FrozenFactorProbe.construct(u,v)
    np.testing.assert_array_equal(a.u,b.u);np.testing.assert_array_equal(a.v,b.v)
    assert not np.array_equal(a.u[2:],a.v[2:]) and not a.receipt["factor4_pretrained"]
    with pytest.raises(ValueError,match="DEPENDENT_NO_RETRY"):
        p.FrozenFactorProbe.construct(np.repeat(u[:1],2,axis=0),v)
    with pytest.raises(ValueError,match="PRELOCKED"):
        p.FrozenFactorProbe.construct(u,v,seed=p.PROBE_SEED+1)


@pytest.mark.parametrize("arm",["B4","B5"])
@pytest.mark.parametrize("mode",["INTER","B7"])
def test_rank2_matches_immutable_original_operator_in_independent_fp64_math(arm,mode):
    visual,gene,mask,prior,_,_=tokens(n=5)
    rng=np.random.default_rng(103)
    with torch.random.fork_rng(devices=[]):
        model=RepairRepresentation(torch.randn(8,768),torch.randn(8,768),torch.tensor(prior),
            spec=RepairSpec(relation_mode="dual"),config=ModelConfig(),seed=31).double().eval().requires_grad_(False)
    probe=p.FrozenFactorProbe.construct(model.u.weight.detach().numpy(),model.v.weight.detach().numpy())
    mp=rng.normal(size=visual.shape) if arm=="B5" else None
    mg=rng.normal(size=gene.shape) if arm=="B5" else None
    actual=p.frozen_factor_mechanism(visual,gene,mask,probe,prior,factor_rank=2,mode=mode,base_arm=arm,
                                   mu_visual=mp,mu_gene=mg)
    outputs={"pre_residual_visual_tokens":visual,"pre_residual_gene_tokens":gene,"gene_mask":mask,
             "additive_image":np.zeros((5,64)),"additive_gene":np.zeros((5,64)),"additive_clinical":np.zeros((5,64))}
    # The original B7 is the residual variant. For raw B4 use original B1's
    # same additive operator, keeping masks/support identical.
    original_arm="B7" if mode=="B7" and arm=="B5" else "B1" if mode=="B7" else arm
    expected=frozen_readout_operator(model,outputs,mp,mg,arm=original_arm)["mechanism"]
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_b7_is_separable_even_when_gene_measurement_masks_differ_but_interaction_is_not():
    visual,gene,mask,prior,u,v=tokens(n=2)
    probe=p.FrozenFactorProbe.construct(u,v)
    pp=visual[[0,1,0,1]];gg=gene[[0,1,1,0]];mm=mask[[0,1,1,0]]
    additive=p.frozen_factor_mechanism(pp,gg,mm,probe,prior,factor_rank=4,mode="B7")
    np.testing.assert_allclose(additive[0]+additive[1]-additive[2]-additive[3],0.,atol=1e-12)
    joint=p.frozen_factor_mechanism(pp,gg,mm,probe,prior,factor_rank=4,mode="INTER")
    assert np.linalg.norm(joint[0]+joint[1]-joint[2]-joint[3])>1e-3
    empty=p.dual_relation_weights(prior,np.zeros((2,50),bool))
    assert np.isfinite(empty).all() and not empty.any()
    assert np.allclose(p.dual_relation_weights(prior,mask).sum((1,2)),1.)


def test_main_pca_shared_across_heads_and_no_interaction_has_no_fake_zero_columns():
    train={k:v.astype(np.float32) for k,v in main_features(48,9).items()}
    ev={k:v.astype(np.float32) for k,v in main_features(6,10).items()}
    main=p.MainProjection.fit(train,width=8)
    zero=p.ReadoutSpec(8,0,"SHARED","NO_INTER")
    nointer=p.ReadoutDesign.fit(zero,main,train)
    assert nointer.transform(train).shape==(48,24) and nointer.transform(ev).shape==(6,24)
    assert nointer.receipt["active_coefficients"]==24 and nointer.receipt["zero_padding_columns"]==0
    rng=np.random.default_rng(99);m=rng.normal(size=(48,4))
    joint=p.ReadoutDesign.fit(p.ReadoutSpec(8,4,"B4","INTER"),main,train,m)
    np.testing.assert_array_equal(joint.transform(train,m)[:,:24],nointer.transform(train))
    assert joint.main is nointer.main and joint.receipt["additional_PCA_fits"]==0
    with pytest.raises(ValueError,match="HIDDEN_MECHANISM"):
        p.ReadoutDesign.fit(zero,main,train,np.zeros((48,0)))
    with pytest.raises(ValueError,match="RANK_INSUFFICIENT"):
        p.ReadoutDesign.fit(p.ReadoutSpec(8,4,"B4","INTER"),main,train,np.column_stack((m[:,:2],m[:,:2])))
    main16=p.MainProjection.fit(train,width=16)
    assert main16.transform(train).shape==(48,40) and main16.blocks["clinical"].components.shape[0]==8
    changed=copy.deepcopy(train);changed["pathology"][0,0]+=.01
    with pytest.raises(ValueError,match="MAIN_MAP_TRAINING_INPUT_CHANGED"):
        p.ReadoutDesign.fit(zero,main,changed)


def test_source_background_uses_one_frozen_encoder_and_heldout_values_cannot_change_training_views():
    visual,gene,mask,prior,u,v=tokens(n=40)
    c=np.random.default_rng(21).normal(size=(40,64))
    outputs={"pre_residual_visual_tokens":visual,"pre_residual_gene_tokens":gene,"gene_mask":mask,"additive_clinical":c}
    clinical={k:np.asarray([str(30+i) if k=="AGE" else str(i%3) for i in range(40)]) for k in
              ("AGE","SEX","AJCC_PATHOLOGIC_TUMOR_STAGE","PATH_T_STAGE","PATH_N_STAGE","PATH_M_STAGE")}
    fit=np.arange(30);ev=np.arange(30,40)
    original=p.frozen_source_views(outputs,clinical,fit,ev,u_weight=u,v_weight=v,relation_prior=prior)
    changed=copy.deepcopy(outputs);changed["pre_residual_visual_tokens"][ev]*=10
    changed["pre_residual_gene_tokens"][ev]+=5;changed["additive_clinical"][ev]-=4
    changed_clinical=copy.deepcopy(clinical);changed_clinical["AGE"][ev]="99"
    counter=p.frozen_source_views(changed,changed_clinical,fit,ev,u_weight=u,v_weight=v,relation_prior=prior)
    for key in original.train:np.testing.assert_array_equal(original.train[key],counter.train[key])
    for key in original.mechanisms:np.testing.assert_array_equal(original.mechanisms[key]["train"],counter.mechanisms[key]["train"])
    assert original.receipt["background_models"]==4 and len(original.mechanisms)==8
    assert original.background["audit"]["encoder_coordinates"]=="one_frozen_current_partition_model"
    with pytest.raises(ValueError,match="OUTCOME_FREE"):
        p.frozen_source_views({"e":np.ones(2)},None,None,None,u_weight=u,v_weight=v,relation_prior=prior)


@pytest.mark.parametrize("width",[24,26,28,40,42,44])
def test_full_risk_efron_ridge_gradient_for_each_true_active_parameter_budget(width):
    rng=np.random.default_rng(66);x=rng.normal(size=(52,width));w=rng.normal(size=width)*.02
    offset=rng.normal(size=52)*.1;t=np.repeat(np.arange(1,27),2).astype(float);e=(np.arange(52)%3==0).astype(int)
    problem=p.EfronProblem(t,e)
    loss,dr=problem.loss_gradient(offset+x@w);gradient=x.T@dr+10*w
    def f(coef):return problem.loss_gradient(offset+x@coef)[0]+5*(coef@coef)
    epsilon=1e-6
    oracle=np.array([(f(w+np.eye(width)[i]*epsilon)-f(w-np.eye(width)[i]*epsilon))/(2*epsilon) for i in range(width)])
    np.testing.assert_allclose(gradient,oracle,rtol=1e-6,atol=3e-9)
    assert np.isfinite(loss)


def test_synthetic_joint_vs_true_refit_and_frozen_dependency_ablation_are_distinct():
    train=main_features(60,101);ev=main_features(8,102);rng=np.random.default_rng(104)
    main=p.MainProjection.fit(train,width=8)
    m=rng.normal(size=(60,2));me=rng.normal(size=(8,2))
    joint=p.ReadoutDesign.fit(p.ReadoutSpec(8,2,"B4","INTER"),main,train,m)
    nointer=p.ReadoutDesign.fit(p.ReadoutSpec(8,0,"SHARED","NO_INTER"),main,train)
    t=np.arange(1,61,dtype=float)*2;e=(np.arange(60)%3==0).astype(int)
    et=np.arange(1,9,dtype=float)*10;ee=(np.arange(8)%3==0).astype(int)
    offset=rng.normal(size=60);eo=rng.normal(size=8)
    a=p.fit_diagnostic_cell(joint,train,ev,t,e,et,ee,offset,eo,ridge_lambda=10.,
        train_mechanism=m,evaluation_mechanism=me,scope="synthetic_qualification")
    b=p.fit_diagnostic_cell(nointer,train,ev,t,e,et,ee,offset,eo,ridge_lambda=10.,scope="synthetic_qualification")
    assert len(a.model.coefficient)==26 and len(b.model.coefficient)==24
    assert a.model.receipt["gradient_max"]<=2e-5 and b.model.receipt["gradient_max"]<=2e-5
    assert not a.model.receipt["lambda_selected_using_validation"]
    z=joint.transform(ev,me)
    frozen=p.frozen_main_only_ablation(a.model,z,eo,main_columns=24)
    np.testing.assert_allclose(a.evaluation_risk,frozen["risk"]+frozen["mechanism_delta"],atol=1e-14)
    drift=p.main_coefficient_drift(a.model,b.model,joint_design=joint,no_interaction_design=nointer)
    assert drift["L2_difference"]>0 and not np.array_equal(frozen["risk"],b.evaluation_risk)
    assert "in_sample" in a.receipt["training_metrics_scope"] and a.receipt["new_neural_parameters"]==0
    decomposition=p.decompose_joint_vs_refit(a,b,ev,me,eo)
    np.testing.assert_allclose(decomposition["joint_risk"]-decomposition["real_refit_no_interaction_risk"],
        decomposition["Z_times_beta_full_minus_beta_refit"]+decomposition["M_times_gamma"],atol=1e-14)
    np.testing.assert_array_equal(decomposition["frozen_extra_features_off_risk"],frozen["risk"])
    assert decomposition["new_fits"]==0 and a.receipt["primary_lambda"]==1.
    assert "post_P2_development" in a.receipt["study_scope"]
    altered=copy.deepcopy(b.model);altered.receipt["fit_offset_sha256"]="0"*64
    with pytest.raises(ValueError,match="SAME_ENDPOINTS_AND_R0"):
        p.main_coefficient_drift(a.model,altered,joint_design=joint,no_interaction_design=nointer)
    changed=copy.deepcopy(train);changed["pathology"][0,0]+=.01
    with pytest.raises(ValueError,match="FEATURES_CHANGED_AFTER_MAP_FIT"):
        p.fit_diagnostic_cell(joint,changed,ev,t,e,et,ee,offset,eo,ridge_lambda=10.,
            train_mechanism=m,evaluation_mechanism=me,scope="synthetic_qualification")


def test_outer_or_unbound_partition_rejected_before_design_or_endpoints_are_touched():
    with pytest.raises(ValueError,match="OUTER_DIAGNOSTIC"):
        p.fit_diagnostic_cell(None,None,None,None,None,None,None,None,None,ridge_lambda=10.,scope="outer_final")
    with pytest.raises(ValueError,match="ONLY_EXPLICIT_INNER"):
        p.fit_diagnostic_cell(None,None,None,None,None,None,None,None,None,ridge_lambda=10.,
            bindings={"split":"outer1_final","outer_test_accessed":False})

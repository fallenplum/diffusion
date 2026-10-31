"""Frozen-representation, inner-only P2 training-sufficiency readout probes.

The four-factor extension has two pretrained factors plus two fixed, outcome-
blind orthogonal-complement probes. It is not a pretrained rank-four encoder.
All main maps, backgrounds and feature scales are fitted on current fit rows.
No module in this file loads data, checkpoints, endpoints or outer predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from .data import background_crossfit
from .p2_survival import BlockProjection, EfronProblem, array_sha, matrix, evaluate_safe, risk_repair_damage
from .survival import efron_numpy

PROBE_SEED=271703
PRIMARY_LAMBDA=1.
RIDGE_LAMBDAS=(.1,1.,10.)
RANK_RELATIVE_TOLERANCE=1e-5
FACTOR_NUMERICAL_TOLERANCE=1e-10
MODES=("INTER","B7","NO_INTER")


def require(condition,message):
    if not condition:raise ValueError(message)


@dataclass(frozen=True)
class ReadoutSpec:
    main_pca:int
    factor_rank:int
    base_arm:str
    mode:str

    def validate(self):
        require(self.main_pca in (8,16) and self.mode in MODES,"P2S_LOCKED_READOUT_GRID")
        if self.mode=="NO_INTER":
            require(self.factor_rank==0 and self.base_arm=="SHARED","P2S_NO_INTER_MUST_BE_SHARED_ONCE")
        elif self.base_arm=="ATTN_MATCHED":
            require(self.mode=="INTER" and self.factor_rank==2,"P2S_ATTN_ONLY_ORIGINAL_INTERACTION_R2")
        else:
            require(self.factor_rank in (2,4) and self.base_arm in ("B4","B5"),"P2S_FACTOR_ARM_GRID")

    @property
    def name(self):
        self.validate()
        return f"D{self.main_pca}_{self.base_arm}_R{self.factor_rank}_{self.mode}"

    @property
    def main_coefficients(self):return 2*self.main_pca+8

    @property
    def coefficients(self):return self.main_coefficients+self.factor_rank


def readout_specs():
    specs=[]
    for width in (8,16):
        specs.append(ReadoutSpec(width,0,"SHARED","NO_INTER"))
        specs.extend(ReadoutSpec(width,rank,arm,mode) for arm in ("B4","B5")
                     for rank in (2,4) for mode in ("INTER","B7"))
    require(len(specs)==len({s.name for s in specs})==18,"P2S_GRID_CARDINALITY")
    return tuple(specs)


def attention_readout_specs():
    """Four heads per ATTN state; supplied original attention features only.

    These are separate from the eighteen MAIN heads. The original frozen FP32
    attention operator, not the semantic-factor probe, must supply INTER's two
    mechanism columns. No ATTN rank-four or B7 operator is authorized.
    """
    return tuple(spec for width in (8,16) for spec in
        (ReadoutSpec(width,0,"SHARED","NO_INTER"),ReadoutSpec(width,2,"ATTN_MATCHED","INTER")))


def orthogonal_completion(weight,rng):
    """Copy original two rows; append one fixed QR complement, never retry."""
    original=np.asarray(weight)
    require(original.shape==(2,64) and original.dtype.kind=="f" and np.isfinite(original).all(),
            "P2S_PRETRAINED_TWO_BY64_FACTORS_REQUIRED")
    w=original.astype(np.float64)
    singular=np.linalg.svd(w,compute_uv=False)
    require(singular[0]>0 and singular[-1]>FACTOR_NUMERICAL_TOLERANCE*singular[0],
            "P2S_ORIGINAL_FACTOR_ROWS_DEPENDENT_NO_RETRY")
    q,_=np.linalg.qr(w.T,mode="reduced")
    candidate=rng.standard_normal((64,2))
    # A second projection removes numerical residue in the original span.
    candidate=candidate-q@(q.T@candidate)
    candidate=candidate-q@(q.T@candidate)
    s=np.linalg.svd(candidate,compute_uv=False)
    require(s[-1]>FACTOR_NUMERICAL_TOLERANCE*s[0],"P2S_FIXED_COMPLEMENT_DEGENERATE_NO_RETRY")
    extra,_=np.linalg.qr(candidate,mode="reduced")
    for column in range(2):
        pivot=int(np.argmax(np.abs(extra[:,column])))
        if extra[pivot,column]<0:extra[:,column]*=-1
    scale=float(np.linalg.norm(w,axis=1).mean())
    extra=extra.T*scale
    combined=np.concatenate((w,extra),axis=0)
    require(np.array_equal(combined[:2],original) and
            np.max(np.abs(extra@w.T))<=FACTOR_NUMERICAL_TOLERANCE*max(1.,scale*np.linalg.norm(w)),
            "P2S_ORIGINAL_FACTOR_OR_COMPLEMENT_CHANGED")
    return combined,{"original_weight_sha256":array_sha(original),"extended_weight_sha256":array_sha(combined),
        "original_rows_retained_exact_numeric_values":True,"original_dtype":str(original.dtype),
        "computation_dtype":"float64","original_row_norms":np.linalg.norm(w,axis=1).tolist(),
        "extra_row_norms":np.linalg.norm(extra,axis=1).tolist(),"extra_target_norm":scale,
        "original_effective_rank":2,"extended_effective_rank":int(np.linalg.matrix_rank(combined)),
        "extra_original_max_abs_inner_product":float(np.max(np.abs(extra@w.T))),
        "outcomes_or_patient_features_used":False,"resampled_projection_candidates":0}


@dataclass
class FrozenFactorProbe:
    u:np.ndarray
    v:np.ndarray
    receipt:dict

    @classmethod
    def construct(cls,u_weight,v_weight,*,seed=PROBE_SEED):
        require(seed==PROBE_SEED,"P2S_PROBE_SEED_IS_PRELOCKED")
        rng=np.random.default_rng(seed)
        u,ua=orthogonal_completion(u_weight,rng)
        v,va=orthogonal_completion(v_weight,rng)
        return cls(u,v,{"seed":seed,"P":ua,"G":va,
            "factor2":"two_frozen_pretrained_projection_rows",
            "factor4":"same_two_pretrained_plus_two_fixed_orthogonal_complement_probes",
            "factor4_pretrained":False,"new_neural_parameters":0,"patient_or_DSS_selection":False,
            "independent_modality_draws":"one_local_generator_P_then_G_no_global_RNG_mutation"})


def dual_relation_weights(prior,gene_mask,*,known_fraction=.5):
    prior=np.asarray(prior,np.float64);mask=np.asarray(gene_mask,bool)
    require(prior.shape==(8,50) and mask.ndim==2 and mask.shape[1]==50 and
            np.isfinite(prior).all() and (prior>=0).all() and known_fraction==.5,"P2S_FIXED_DUAL_RELATION_CONTRACT")
    known=(prior>0)[None]&mask[:,None,:]
    unknown=(prior==0)[None]&mask[:,None,:]
    nk=known.sum((1,2));nu=unknown.sum((1,2))
    fraction=np.full(len(mask),known_fraction)
    fraction=np.where(nk==0,0.,fraction)
    fraction=np.where((nu==0)&(nk>0),1.,fraction)
    weights=(fraction[:,None,None]*known/np.maximum(nk,1)[:,None,None]+
             (1-fraction[:,None,None])*unknown/np.maximum(nu,1)[:,None,None])
    return weights


def frozen_factor_mechanism(visual,gene,gene_mask,probe,prior,*,factor_rank,mode,
                            base_arm="B4",mu_visual=None,mu_gene=None):
    require(factor_rank in (2,4) and mode in ("INTER","B7") and base_arm in ("B4","B5"),
            "P2S_FROZEN_MECHANISM_SPEC")
    p=np.asarray(visual,np.float64);g=np.asarray(gene,np.float64);mask=np.asarray(gene_mask,bool)
    require(p.ndim==g.ndim==3 and p.shape[1:]==(8,64) and g.shape==(len(p),50,64) and
            mask.shape==(len(p),50) and np.isfinite(p).all() and np.isfinite(g).all(),
            "P2S_FROZEN_TOKEN_SHAPES")
    if base_arm=="B5":
        mp=np.asarray(mu_visual,np.float64);mg=np.asarray(mu_gene,np.float64)
        require(mp.shape==p.shape and mg.shape==g.shape and np.isfinite(mp).all() and np.isfinite(mg).all(),
                "P2S_SAME_ENCODER_CLINICAL_BACKGROUNDS_REQUIRED")
        p=p-mp;g=np.where(mask[...,None],g-mg,0.)
    else:
        require(mu_visual is None and mu_gene is None,"P2S_B4_UNEXPECTED_RESIDUALIZATION")
        g=np.where(mask[...,None],g,0.)
    uu=p@probe.u[:factor_rank].T;vv=g@probe.v[:factor_rank].T
    if mode=="INTER":
        weights=dual_relation_weights(prior,mask)
        return np.einsum("nsr,nkr,nsk->nr",uu,vv,weights,optimize=True)
    # Independent P-only and G-only functions. In particular, a patient's G
    # observation mask cannot change the visual term's relation support.
    fixed=dual_relation_weights(prior,np.ones_like(mask))
    return (np.einsum("nsr,nsk->nr",uu*uu,fixed,optimize=True)+
            np.einsum("nkr,nsk->nr",vv*vv*mask[...,None],fixed,optimize=True))


@dataclass
class FrozenSourceViews:
    train:dict
    evaluation:dict
    mechanisms:dict
    background:dict
    probe:FrozenFactorProbe
    receipt:dict

    def mechanism(self,spec):
        spec.validate()
        require(spec.mode!="NO_INTER","P2S_NO_INTER_HAS_ZERO_MECHANISM_FEATURES")
        return self.mechanisms[f"{spec.base_arm}_R{spec.factor_rank}_{spec.mode}"]


def frozen_source_views(outputs,clinical_raw,fit_rows,evaluation_rows,*,u_weight,v_weight,relation_prior,
                        probe_seed=PROBE_SEED,background_seed=103017):
    require(not {"t","e","DSS","PFI","OS","time","event","risk","offset"}.intersection(outputs),
            "P2S_SOURCE_VIEWS_MUST_BE_OUTCOME_FREE")
    p=np.asarray(outputs["pre_residual_visual_tokens"])
    g=np.asarray(outputs["pre_residual_gene_tokens"])
    mask=np.asarray(outputs["gene_mask"],bool)
    c=matrix(outputs["additive_clinical"],"clinical")
    require(c.shape[1]==64 and p.shape==(len(c),8,64) and g.shape==(len(c),50,64) and mask.shape==(len(c),50) and
            np.isfinite(p).all() and np.isfinite(g).all(),"P2S_FROZEN_SOURCE_DIMENSIONS")
    tr=np.asarray(fit_rows);ev=np.asarray(evaluation_rows)
    require(tr.ndim==ev.ndim==1 and tr.dtype.kind in "iu" and ev.dtype.kind in "iu" and len(tr)>16 and
            np.array_equal(np.sort(np.r_[tr,ev]),np.arange(len(c))),"P2S_SOURCE_ROWS_MUST_ONLY_BE_FIT_AND_EVALUATION")
    probe=FrozenFactorProbe.construct(u_weight,v_weight,seed=probe_seed)
    g=np.where(mask[...,None],g,0.)
    flat=np.column_stack((p.reshape(len(c),-1),g.reshape(len(c),-1)))
    background=background_crossfit(c,flat,tr,ev,clinical_raw=clinical_raw,seed=background_seed,alpha=1.)
    mean=background["mean"]
    mp=mean[:,:512].reshape(p.shape);mg=mean[:,512:].reshape(g.shape)
    main={"pathology":p.reshape(len(c),-1),"omics":g.reshape(len(c),-1),"clinical":c}
    mechanisms={}
    for arm in ("B4","B5"):
        for rank in (2,4):
            for mode in ("INTER","B7"):
                value=frozen_factor_mechanism(p,g,mask,probe,relation_prior,factor_rank=rank,mode=mode,base_arm=arm,
                    mu_visual=mp if arm=="B5" else None,mu_gene=mg if arm=="B5" else None)
                mechanisms[f"{arm}_R{rank}_{mode}"]={"train":value[tr],"evaluation":value[ev],
                    "feature_sha256":array_sha(value),"factor_rank":rank,"mode":mode,"base_arm":arm}
    return FrozenSourceViews({k:v[tr] for k,v in main.items()},{k:v[ev] for k,v in main.items()},
        mechanisms,background,probe,{"main_features":"same_original_pre_residual_512P_3200G_64C",
            "background_models":background["audit"]["background_models"],"background_encoder_coordinates":"one_frozen_source_encoder",
            "background_outcome_access":False,"fit_rows_sha256":array_sha(tr),"evaluation_rows_sha256":array_sha(ev),
            "probe":probe.receipt,"mechanism_computation_dtype":"float64","mechanism2_is_old_algebra_not_FP32_bitwise_replay":True,
            "new_neural_parameters":0,"DSS_used_for_source_features":False})


@dataclass
class MainProjection:
    width:int
    blocks:dict
    receipt:dict

    @classmethod
    def fit(cls,features:Mapping[str,np.ndarray],*,width):
        require(width in (8,16) and set(features)=={"pathology","omics","clinical"},"P2S_MAIN_PCA_GRID")
        arrays={k:matrix(v,k) for k,v in features.items()}
        require(len({len(x) for x in arrays.values()})==1,"P2S_MAIN_ROW_ALIGNMENT")
        blocks={k:BlockProjection.fit(arrays[k],8 if k=="clinical" else width,
            minimum_sd=1e-10,relative_singular_tolerance=RANK_RELATIVE_TOLERANCE) for k in arrays}
        return cls(width,blocks,{"main_pca":width,"clinical_pca":8,"actual_PCA_fits":3,
            "main_coefficients":2*width+8,"train_only":True,"outcomes_used":False,
            "input_train_sha256":{k:array_sha(v) for k,v in arrays.items()},
            "blocks":{k:v.receipt for k,v in blocks.items()},"reusable_across_all_modes_factors_lambdas":True})

    def transform(self,features):
        require(set(features)=={"pathology","omics","clinical"},"P2S_EXACT_MAIN_BLOCKS_REQUIRED")
        rows=[self.blocks[k].transform(features[k]) for k in ("pathology","omics","clinical")]
        require(len({len(x) for x in rows})==1,"P2S_MAIN_PREDICTION_ROWS")
        return np.column_stack(rows)


@dataclass
class ReadoutDesign:
    spec:ReadoutSpec
    main:MainProjection
    mechanism_mean:np.ndarray
    mechanism_sd:np.ndarray
    receipt:dict

    @classmethod
    def fit(cls,spec,main,train_main,train_mechanism=None):
        spec.validate();require(main.width==spec.main_pca,"P2S_SHARED_MAIN_MAP_WIDTH_CHANGED")
        require({k:array_sha(matrix(v,k)) for k,v in train_main.items()}==main.receipt["input_train_sha256"],
                "P2S_MAIN_MAP_TRAINING_INPUT_CHANGED")
        zmain=main.transform(train_main)
        if spec.mode=="NO_INTER":
            require(train_mechanism is None,"P2S_NO_INTER_MUST_NOT_RECEIVE_HIDDEN_MECHANISM")
            mean=sd=np.empty(0,np.float64);effective=0
        else:
            x=matrix(train_mechanism,"mechanism")
            require(x.shape==(len(zmain),spec.factor_rank),"P2S_MECHANISM_WIDTH")
            mean=x.mean(0);sd=x.std(0)
            require((sd>1e-10).all(),"P2S_MECHANISM_HAS_INACTIVE_COLUMNS_NO_PADDING")
            singular=np.linalg.svd((x-mean)/sd,compute_uv=False)
            effective=int((singular>singular[0]*RANK_RELATIVE_TOLERANCE).sum()) if singular[0]>0 else 0
            require(effective==spec.factor_rank,"P2S_MECHANISM_EFFECTIVE_RANK_INSUFFICIENT_NO_RETRY")
        whole=zmain if spec.mode=="NO_INTER" else np.column_stack((zmain,(x-mean)/sd))
        receipt={"spec":vars(spec),"main_map_reused":True,"additional_PCA_fits":0,
            "nominal_coefficients":spec.coefficients,"active_coefficients":spec.coefficients,
            "active_main_columns":spec.main_coefficients,"active_mechanism_columns":effective,
            "mechanism_standardization_fit_only":True,"zero_padding_columns":0,
            "mechanism_train_input_sha256":array_sha(x) if spec.mode!="NO_INTER" else None,
            "mechanism_original_train_input_sha256":array_sha(np.asarray(train_mechanism)) if spec.mode!="NO_INTER" else None,
            "mechanism_training_hash_dtype":"float64_normalized_numeric_values",
            "ATTN_mechanism_source_contract":"supplied_original_frozen_FP32_attention_operator_not_semantic_factor_probe"
                if spec.base_arm=="ATTN_MATCHED" else None,
            "main_train_design_sha256":array_sha(zmain),"new_neural_parameters":0,
            "main_train_design_rank":int(np.linalg.matrix_rank(zmain)),
            "joint_train_design_rank":int(np.linalg.matrix_rank(whole)),
            "factor4_contains_two_fixed_nonpretrained_probes":spec.factor_rank==4}
        return cls(spec,main,mean,sd,receipt)

    def transform(self,main_features,mechanism=None):
        z=self.main.transform(main_features)
        if self.spec.mode=="NO_INTER":
            require(mechanism is None,"P2S_NO_INTER_HIDDEN_FEATURES")
            return z
        x=matrix(mechanism,"mechanism_prediction")
        require(x.shape==(len(z),self.spec.factor_rank),"P2S_MECHANISM_PREDICTION_WIDTH")
        return np.column_stack((z,(x-self.mechanism_mean)/self.mechanism_sd))


@dataclass
class DiagnosticCox:
    coefficient:np.ndarray
    ridge_lambda:float
    receipt:dict

    def predict(self,design,offset):
        x=matrix(design,"diagnostic_prediction");o=np.asarray(offset,np.float64)
        require(x.shape[1]==len(self.coefficient) and o.shape==(len(x),) and np.isfinite(o).all(),
                "P2S_DESIGN_OFFSET_DIMENSIONS")
        return o+x@self.coefficient


def fit_diagnostic_cox(design,time,event,offset,*,ridge_lambda,maxiter=4000,gradient_tolerance=2e-5):
    x=matrix(design,"diagnostic_fit");o=np.asarray(offset,np.float64)
    require(x.shape[1] in (24,26,28,40,42,44) and ridge_lambda in RIDGE_LAMBDAS and
            o.shape==(len(x),) and np.isfinite(o).all(),"P2S_DYNAMIC_ACTIVE_COEFFICIENT_CONTRACT")
    active=int((x.std(0)>1e-10).sum());require(active==x.shape[1],"P2S_NO_DUMMY_OPTIMIZER_COEFFICIENTS")
    problem=EfronProblem(time,event)
    require(len(problem.time)==len(x),"P2S_ENDPOINT_ROWS")
    def objective(w):
        loss,dr=problem.loss_gradient(o+x@w)
        return loss+ridge_lambda/2*(w@w),x.T@dr+ridge_lambda*w
    trace=[]
    def callback(w):
        value,gradient=objective(w)
        trace.append({"iteration":len(trace)+1,"objective":float(value),"gradient_max":float(np.max(np.abs(gradient)))})
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",callback=callback,
        options={"maxiter":maxiter,"gtol":1e-8,"ftol":1e-13,"maxls":50})
    value,gradient=objective(result.x);maximum=float(np.max(np.abs(gradient)))
    require(np.isfinite(value) and np.isfinite(result.x).all() and maximum<=gradient_tolerance,
            "P2S_CONVEX_COX_CONVERGENCE_FAILED")
    prediction=o+x@result.x;reference=efron_numpy(prediction,time,event)
    cached,_=problem.loss_gradient(prediction)
    require(abs(reference-cached)<=1e-10,"P2S_EFRON_ORACLE_DISAGREEMENT")
    return DiagnosticCox(result.x.copy(),float(ridge_lambda),{"solver":"L-BFGS-B_full_risk_Efron_ridge",
        "active_coefficients":active,"optimized_coefficients":len(result.x),"ridge_lambda":float(ridge_lambda),
        "primary_prelocked_lambda":PRIMARY_LAMBDA,"lambda_selected_using_validation":False,
        "objective":float(value),"gradient_max":maximum,"gradient_tolerance":gradient_tolerance,
        "iterations":int(result.nit),"optimizer_success_flag":bool(result.success),"optimizer_message":str(result.message),
        "nonzero_estimated_coefficients":int((np.abs(result.x)>1e-10).sum()),
        "oracle_loss_abs_error":abs(reference-cached),"training_loss_curve":trace,
        "fit_design_sha256":array_sha(x),"fit_time_sha256":array_sha(np.asarray(time)),
        "fit_event_sha256":array_sha(np.asarray(event)),"fit_offset_sha256":array_sha(o),
        "fit_patients":len(x),"fit_events":int(problem.event.sum()),
        "validation_used_for_optimizer_or_lambda":False,"risk_optimizer_fits":1})


def frozen_main_only_ablation(model,design_matrix,offset,*,main_columns):
    x=matrix(design_matrix,"frozen_ablation");o=np.asarray(offset,np.float64)
    require(x.shape[1]==len(model.coefficient) and 0<main_columns<=x.shape[1] and o.shape==(len(x),),
            "P2S_FROZEN_ABLATION_DIMENSIONS")
    return {"risk":o+x[:,:main_columns]@model.coefficient[:main_columns],
            "mechanism_delta":x[:,main_columns:]@model.coefficient[main_columns:],
            "scope":"frozen_joint_coefficients_dependency_intervention_not_a_no_interaction_refit",
            "new_fits":0}


def main_coefficient_drift(joint,no_interaction,*,joint_design,no_interaction_design):
    require(no_interaction_design.spec.mode=="NO_INTER" and
            joint_design.receipt["main_train_design_sha256"]==no_interaction_design.receipt["main_train_design_sha256"] and
            joint.ridge_lambda==no_interaction.ridge_lambda,"P2S_COEFFICIENT_DRIFT_REQUIRES_SAME_MAIN_MAP_AND_LAMBDA")
    require(all(joint.receipt[k]==no_interaction.receipt[k] for k in
                ("fit_time_sha256","fit_event_sha256","fit_offset_sha256")),
            "P2S_COEFFICIENT_DRIFT_REQUIRES_SAME_ENDPOINTS_AND_R0")
    width=no_interaction_design.spec.main_coefficients
    a=joint.coefficient[:width];b=no_interaction.coefficient
    require(a.shape==b.shape== (width,),"P2S_MAIN_COEFFICIENT_SHAPES")
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return {"main_coefficients":width,"L2_difference":float(np.linalg.norm(a-b)),
        "max_abs_difference":float(np.max(np.abs(a-b))),"joint_L2":float(na),"refit_no_interaction_L2":float(nb),
        "cosine":float(a@b/(na*nb)) if na>1e-12 and nb>1e-12 else None,
        "scope":"joint_refit_main_coefficients_vs_true_no_interaction_refit_same_features_lambda"}


@dataclass
class DiagnosticCell:
    design:ReadoutDesign
    model:DiagnosticCox
    training_risk:np.ndarray
    evaluation_risk:np.ndarray
    frozen_main_only_training_risk:np.ndarray
    frozen_main_only_evaluation_risk:np.ndarray
    receipt:dict


def fit_diagnostic_cell(design,train_main,evaluation_main,train_time,train_event,evaluation_time,evaluation_event,
                        training_offset,evaluation_offset,*,ridge_lambda,train_mechanism=None,evaluation_mechanism=None,
                        bindings=None,scope="inner_diagnostic"):
    require(scope in ("inner_diagnostic","synthetic_qualification"),"P2S_OUTER_DIAGNOSTIC_NOT_AUTHORIZED")
    bindings={} if bindings is None else dict(bindings)
    if scope=="inner_diagnostic":
        require(re.fullmatch(r"outer[1-5]_inner[0-2]",bindings.get("split","")) is not None and
                bindings.get("outer_test_accessed") is False,"P2S_ONLY_EXPLICIT_INNER_PARTITIONS")
    ztrain=design.transform(train_main,train_mechanism);zeval=design.transform(evaluation_main,evaluation_mechanism)
    require(array_sha(ztrain[:,:design.spec.main_coefficients])==design.receipt["main_train_design_sha256"] and
            (design.spec.mode=="NO_INTER" or array_sha(matrix(train_mechanism,"mechanism"))==design.receipt["mechanism_train_input_sha256"]),
            "P2S_READOUT_TRAIN_FEATURES_CHANGED_AFTER_MAP_FIT")
    model=fit_diagnostic_cox(ztrain,train_time,train_event,training_offset,ridge_lambda=ridge_lambda)
    train=model.predict(ztrain,training_offset);evaluation=model.predict(zeval,evaluation_offset)
    atr=frozen_main_only_ablation(model,ztrain,training_offset,main_columns=design.spec.main_coefficients)
    aev=frozen_main_only_ablation(model,zeval,evaluation_offset,main_columns=design.spec.main_coefficients)
    metrics=evaluate_safe(train_time,train_event,evaluation_time,evaluation_event,evaluation)
    receipt={"schema":"semantic_p2s_inner_frozen_readout_cell_v1","scope":scope,"bindings":bindings,
        "spec":vars(design.spec),"DSS_coefficients":design.spec.coefficients,"design":design.receipt,
        "risk_optimizer":model.receipt,"ridge_lambda":float(ridge_lambda),"primary_lambda":PRIMARY_LAMBDA,
        "lambda_selected":False,"risk_optimizer_fits":1,"new_neural_fits":0,"new_neural_parameters":0,
        "training_metrics":evaluate_safe(train_time,train_event,train_time,train_event,train),
        "training_metrics_scope":"head_in_sample_over_shared_whole_pipeline_crossfit_r0_not_end_to_end_OOF",
        "evaluation_metrics":metrics,
        "anchor_evaluation_metrics":evaluate_safe(train_time,train_event,evaluation_time,evaluation_event,evaluation_offset),
        "frozen_mechanism_zero_evaluation_metrics":evaluate_safe(train_time,train_event,evaluation_time,evaluation_event,aev["risk"]),
        "frozen_ablation_scope":aev["scope"],
        "repair_damage_vs_shared_r0":risk_repair_damage(evaluation_time,evaluation_event,evaluation,evaluation_offset,train_time,train_event),
        "training_risk_sha256":array_sha(train),"evaluation_risk_sha256":array_sha(evaluation),
        "training_offset_sha256":array_sha(np.asarray(training_offset)),"evaluation_offset_sha256":array_sha(np.asarray(evaluation_offset)),
        "train_design_sha256":array_sha(ztrain),"evaluation_design_sha256":array_sha(zeval),
        "factor4_is_fixed_probe_not_four_pretrained_factors":design.spec.factor_rank==4,
        "absolute_survival_probability_calibration_performed":False,"CI_or_p_values_computed":False,
        "study_scope":"post_P2_development_inner_training_sufficiency_diagnostic_not_independent_confirmation"}
    return DiagnosticCell(design,model,train,evaluation,atr["risk"],aev["risk"],receipt)


def decompose_joint_vs_refit(joint,refit_no_interaction,main_features,mechanism,offset):
    """Separate main-coefficient drift from M gamma against a real refit.

    Returned arrays are private diagnostic outputs, not public report fields.
    The frozen-interaction-off prediction is a dependency intervention, whereas
    the comparator has actually refitted all its main-effect coefficients.
    """
    require(joint.design.spec.mode in ("INTER","B7"),"P2S_DECOMPOSITION_REQUIRES_EXTRA_FEATURE_HEAD")
    drift=main_coefficient_drift(joint.model,refit_no_interaction.model,
        joint_design=joint.design,no_interaction_design=refit_no_interaction.design)
    z=joint.design.transform(main_features,mechanism)
    zbase=refit_no_interaction.design.transform(main_features)
    width=refit_no_interaction.design.spec.main_coefficients
    require(np.array_equal(z[:,:width],zbase),"P2S_DECOMPOSITION_MAIN_COORDINATES_DIFFER")
    full=joint.model.predict(z,offset)
    base=refit_no_interaction.model.predict(zbase,offset)
    main=zbase@(joint.model.coefficient[:width]-refit_no_interaction.model.coefficient)
    extra=z[:,width:]@joint.model.coefficient[width:]
    frozen=frozen_main_only_ablation(joint.model,z,offset,main_columns=width)
    error=float(np.max(np.abs((full-base)-(main+extra))))
    require(error<=1e-12,"P2S_REFIT_DECOMPOSITION_IDENTITY_FAILED")
    return {"joint_risk":full,"real_refit_no_interaction_risk":base,
        "Z_times_beta_full_minus_beta_refit":main,"M_times_gamma":extra,
        "frozen_extra_features_off_risk":frozen["risk"],"main_coefficient_drift":drift,
        "max_abs_decomposition_error":error,"new_fits":0,
        "scope":"joint_vs_true_refit_and_frozen_dependency_diagnostics_not_pure_causal_interaction"}

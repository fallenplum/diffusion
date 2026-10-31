"""P2 frozen 26-coefficient survival readout and training-only feature maps.

No raw dataset is opened here. The caller provides one current fit partition;
all labels belong to that partition, and evaluation arrays never fit a map.
Metrics delegate to the locked P0/P1 scorer. No bootstrap/CI is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA

from .data import background_crossfit
from .survival import efron_numpy, evaluate_fold, harrell_components, tau_support


def array_sha(value):
    x = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(x.dtype), "shape": list(x.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header+x.tobytes()).hexdigest()


def matrix(value, name):
    x = np.asarray(value, np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("FINITE_NONEMPTY_MATRIX_REQUIRED:"+name)
    return x


@dataclass(frozen=True)
class FeatureContract:
    pathology: int = 8
    omics: int = 8
    clinical: int = 8
    mechanism: int = 2
    minimum_sd: float = 1e-10
    relative_singular_tolerance: float = 1e-5

    @property
    def coefficients(self):
        return self.pathology+self.omics+self.clinical+self.mechanism

    def validate(self):
        if (self.pathology,self.omics,self.clinical,self.mechanism)!=(8,8,8,2):
            raise ValueError("P2_PRIMARY_REQUIRES_8P8G8C2M")
        if not np.isfinite(self.minimum_sd) or self.minimum_sd <= 0:
            raise ValueError("FEATURE_SD_THRESHOLD_INVALID")
        if not 0 < self.relative_singular_tolerance < 1:
            raise ValueError("FEATURE_RELATIVE_RANK_THRESHOLD_INVALID")


@dataclass
class BlockProjection:
    mean: np.ndarray
    sd: np.ndarray
    components: np.ndarray
    pca_mean: np.ndarray
    output_mean: np.ndarray
    output_sd: np.ndarray
    fit_hash: str
    receipt: dict

    @classmethod
    def fit(cls, x, width, *, minimum_sd=1e-10,relative_singular_tolerance=1e-5):
        x=matrix(x,"frozen_block")
        if len(x)<=width or x.shape[1]<width:
            raise ValueError("INSUFFICIENT_REAL_FEATURE_DIMENSIONS_NO_PADDING")
        mean=x.mean(0); sd=x.std(0)
        z=(x-mean)/np.where(sd>minimum_sd,sd,1.)
        pca=PCA(n_components=width,svd_solver="full").fit(z)
        singular=pca.singular_values_
        effective=int((singular>relative_singular_tolerance*singular[0]).sum()) if singular[0]>0 else 0
        if effective<width:
            raise ValueError("COMPACT_BLOCK_RELATIVE_RANK_INSUFFICIENT_NO_NOISE_WHITENING")
        projected=pca.transform(z)
        after_mean=projected.mean(0); after_sd=projected.std(0)
        active=after_sd>minimum_sd
        if not active.all():
            raise ValueError("COMPACT_BLOCK_HAS_INACTIVE_DIMENSION_NO_FAKE_CAPACITY")
        return cls(mean,np.where(sd>minimum_sd,sd,1.),pca.components_,pca.mean_,
                   after_mean,after_sd,array_sha(x),
                   {"input_columns":x.shape[1],"input_variable_columns":int((sd>minimum_sd).sum()),
                    "output_columns":width,"output_active_columns":int(active.sum()),
                    "effective_retained_rank":effective,"relative_singular_tolerance":relative_singular_tolerance,
                    "retained_singular_values":singular.tolist(),
                    "fit_patients":len(x),"fit_used_outcomes":False,
                    "explained_variance":pca.explained_variance_.tolist()})

    def transform(self, x):
        x=matrix(x,"block_prediction")
        if x.shape[1]!=len(self.mean):
            raise ValueError("FROZEN_FEATURE_DIMENSION_CHANGED")
        pc=((x-self.mean)/self.sd-self.pca_mean)@self.components.T
        return (pc-self.output_mean)/self.output_sd


@dataclass
class CompactDesign:
    blocks: dict
    mechanism_mean: np.ndarray
    mechanism_sd: np.ndarray
    contract: FeatureContract
    receipt: dict

    @classmethod
    def fit(cls, features:Mapping[str,np.ndarray], *, contract=None):
        c=contract or FeatureContract(); c.validate()
        if set(features)!={"pathology","omics","clinical","mechanism"}:
            raise ValueError("EXACT_FOUR_FROZEN_BLOCKS_REQUIRED")
        arrays={k:matrix(v,k) for k,v in features.items()}
        n={len(x) for x in arrays.values()}
        if len(n)!=1 or arrays["mechanism"].shape[1]!=c.mechanism:
            raise ValueError("FROZEN_FEATURE_ROWS_OR_MECHANISM_DIMENSION")
        states={k:BlockProjection.fit(arrays[k],getattr(c,k),minimum_sd=c.minimum_sd,
                                     relative_singular_tolerance=c.relative_singular_tolerance)
                for k in ("pathology","omics","clinical")}
        m=arrays["mechanism"].mean(0); s=arrays["mechanism"].std(0)
        if (s<=c.minimum_sd).any():
            raise ValueError("MECHANISM_CONTROL_MUST_HAVE_TWO_REAL_VARIABLE_COLUMNS")
        receipt={"coefficients":c.coefficients,"feature_group_widths":[8,8,8,2],
                 "fit_patients":next(iter(n)),"fit_used_outcomes":False,
                 "blocks":{k:v.receipt for k,v in states.items()},
                 "train_input_sha256":{k:array_sha(v) for k,v in arrays.items()},
                 "mechanism_variance":(s*s).tolist()}
        result=cls(states,m,s,c,receipt)
        z=result.transform(arrays)
        result.receipt["design_rank"]=int(np.linalg.matrix_rank(z))
        result.receipt["active_nonconstant_columns"]=int((z.std(0)>c.minimum_sd).sum())
        result.receipt["train_design_sha256"]=array_sha(z)
        return result

    def transform(self, features):
        if set(features)!={"pathology","omics","clinical","mechanism"}:
            raise ValueError("EXACT_FOUR_FROZEN_BLOCKS_REQUIRED")
        rows=[self.blocks[k].transform(features[k]) for k in ("pathology","omics","clinical")]
        mechanism=matrix(features["mechanism"],"mechanism")
        if mechanism.shape[1]!=self.contract.mechanism or len({len(x) for x in [*rows,mechanism]})!=1:
            raise ValueError("PREDICT_FEATURE_ALIGNMENT_CHANGED")
        return np.column_stack([*rows,(mechanism-self.mechanism_mean)/self.mechanism_sd])


class EfronProblem:
    """Cached complete risk-set log weights; objective checked by scorer oracle."""
    def __init__(self,time,event):
        t=np.asarray(time,np.float64); e=np.asarray(event)
        if (t.ndim!=1 or e.shape!=t.shape or not np.isfinite(t).all()
                or (t<0).any() or not np.isin(e,(0,1)).all() or e.sum()==0):
            raise ValueError("EFRON_FIT_ENDPOINTS_INVALID")
        rows=[]
        for at in np.unique(t[e==1]):
            deaths=(t==at)&(e==1); d=int(deaths.sum())
            for ell in range(d):
                weight=(t>=at).astype(float)-ell/d*deaths
                with np.errstate(divide="ignore"):
                    rows.append(np.log(weight))
        self.time=t;self.event=e.astype(float);self.log_weight=np.stack(rows)

    def loss_gradient(self,risk):
        r=np.asarray(risk,np.float64)
        if r.shape!=self.time.shape or not np.isfinite(r).all():
            raise ValueError("EFRON_FIT_RISK_INVALID")
        score=self.log_weight+r[None]
        den=logsumexp(score,axis=1)
        n=self.event.sum()
        loss=(den.sum()-self.event@r)/n
        gradient=(np.exp(score-den[:,None]).sum(0)-self.event)/n
        return float(loss),gradient


@dataclass
class OffsetCox:
    coefficient:np.ndarray
    ridge_lambda:float
    receipt:dict

    def delta(self,design):
        z=matrix(design,"risk_design")
        if z.shape[1]!=len(self.coefficient):
            raise ValueError("COX_DESIGN_WIDTH_CHANGED")
        return z@self.coefficient

    def predict(self,design,offset):
        value=self.delta(design)
        base=np.asarray(offset,np.float64)
        if base.shape!=value.shape or not np.isfinite(base).all():
            raise ValueError("COX_FROZEN_OFFSET_INVALID")
        return base+value


def fit_offset_cox(design,time,event,offset,*,ridge_lambda,maxiter=4000,gradient_tolerance=2e-5):
    x=matrix(design,"risk_design"); base=np.asarray(offset,np.float64)
    if (x.shape[1]!=26 or base.shape!=(len(x),) or not np.isfinite(base).all()
            or ridge_lambda not in (.1,1.,10.)):
        raise ValueError("P2_26_COEFFICIENT_OR_LAMBDA_CONTRACT")
    problem=EfronProblem(time,event)
    if len(problem.time)!=len(x):
        raise ValueError("COX_ENDPOINT_DESIGN_ROWS")
    def objective(w):
        loss,dr=problem.loss_gradient(base+x@w)
        return loss+ridge_lambda/2*(w@w), x.T@dr+ridge_lambda*w
    trace=[]
    def callback(w):
        value,gradient=objective(w)
        trace.append({"iteration":len(trace)+1,"objective":float(value),"gradient_max":float(np.abs(gradient).max())})
    result=minimize(objective,np.zeros(26),jac=True,method="L-BFGS-B",callback=callback,
                    options={"maxiter":maxiter,"gtol":1e-8,"ftol":1e-13,"maxls":50})
    value,gradient=objective(result.x); gradient_max=float(np.abs(gradient).max())
    if not np.isfinite(value) or not np.isfinite(result.x).all() or gradient_max>gradient_tolerance:
        raise RuntimeError("P2_CONVEX_COX_CONVERGENCE_FAILED")
    predicted=base+x@result.x
    independent=efron_numpy(predicted,time,event)
    cached,_=problem.loss_gradient(predicted)
    if abs(independent-cached)>1e-10:
        raise RuntimeError("P2_EFRON_ORACLE_DISAGREEMENT")
    receipt={"solver":"L-BFGS-B_full_risk_Efron_ridge","ridge_lambda":ridge_lambda,
             "parameters":26,"active_design_columns":int((x.std(0)>1e-10).sum()),
             "nonzero_estimated_coefficients":int((np.abs(result.x)>1e-10).sum()),
             "objective":float(value),"gradient_max":gradient_max,
             "gradient_tolerance":gradient_tolerance,"iterations":int(result.nit),
             "optimizer_success_flag":bool(result.success),"optimizer_message":str(result.message),
             "oracle_loss_abs_error":abs(independent-cached),"training_loss_curve":trace,
             "fit_design_sha256":array_sha(x),"fit_time_sha256":array_sha(np.asarray(time)),
             "fit_event_sha256":array_sha(np.asarray(event)),"fit_offset_sha256":array_sha(base),
             "fit_patients":len(x),"fit_events":int(problem.event.sum()),
             "validation_used_for_solver":False,"bootstrap_or_CI":False}
    return OffsetCox(result.x.copy(),float(ridge_lambda),receipt)


def evaluate_safe(train_time,train_event,time,event,risk):
    """Fixed scorer, with predeclared Harrell-only fallback for absent Uno support."""
    h=harrell_components(time,event,risk)
    support=tau_support(train_time,train_event,tau=60.)
    if support["supported"]:
        try:
            return {**evaluate_fold(train_time,train_event,time,event,risk),"Uno60_valid":True,"Uno60_NA_reason":None}
        except ValueError as exc:
            if not any(x in str(exc) for x in ("CENSOR","IPCW")):
                raise
            reason=str(exc)
    else:
        reason=support["reason"]
    return {"Harrell":h.score,"Harrell_num":h.numerator,"Harrell_den":h.denominator,
            "Uno60":None,"Uno_num":None,"Uno_den":None,"Uno60_valid":False,
            "Uno60_NA_reason":reason,"tau_support":support,"comparable_pairs":h.comparable_pairs}


def risk_repair_damage(time,event,risk,reference,train_time,train_event):
    from .survival import uno_weights
    t,e,r,b=map(np.asarray,(time,event,risk,reference))
    good=bad=0.; u_good=u_bad=u_den=0.; count=0
    try:
        weight=uno_weights(train_time,train_event,t,e,tau=60.) if tau_support(train_time,train_event)["supported"] else None
    except ValueError:
        weight=None
    def credit(a):
        return np.where(np.abs(a)<=1e-8,.5,(a>0).astype(float))
    for i in np.flatnonzero(e==1):
        j=np.flatnonzero((t>t[i])|((t==t[i])&(e==0)))
        delta=credit(r[i]-r[j])-credit(b[i]-b[j])
        good+=float(np.maximum(delta,0).sum());bad+=float(np.maximum(-delta,0).sum());count+=len(j)
        if weight is not None:
            u_good+=float(weight[i]*np.maximum(delta,0).sum())
            u_bad+=float(weight[i]*np.maximum(-delta,0).sum());u_den+=float(weight[i]*len(j))
    return {"repair_credit":good,"damage_credit":bad,"comparable_pairs":count,
            "net_Harrell_change":(good-bad)/count if count else None,
            "Uno_repair_credit":u_good if weight is not None else None,
            "Uno_damage_credit":u_bad if weight is not None else None,
            "Uno_pair_weight_denominator":u_den if weight is not None else None,
            "risk_delta_sd":float(np.std(r-b)),"anchor_sd":float(np.std(b)),
            "delta_sd_over_anchor_sd":float(np.std(r-b)/np.std(b)) if np.std(b)>1e-12 else None}


def select_lambda(candidates:Mapping[float,list[dict]]):
    """Three-inner-fold micro H, then supported Uno60, then stronger ridge."""
    if set(candidates)!={.1,1.,10.}:
        raise ValueError("ALL_THREE_SAME_BUDGET_LAMBDAS_REQUIRED")
    summaries=[]
    for lam,rows in candidates.items():
        if len(rows)!=3:
            raise ValueError("THREE_INNER_FOLDS_REQUIRED")
        hn=sum(x["Harrell_num"] for x in rows);hd=sum(x["Harrell_den"] for x in rows)
        if hd<=0:
            raise ValueError("INNER_HARRELL_UNSUPPORTED")
        supported=all(x["Uno60_valid"] and x["Uno_den"] is not None and x["Uno_den"]>0 for x in rows)
        un=sum(x["Uno_num"] for x in rows) if supported else None
        ud=sum(x["Uno_den"] for x in rows) if supported else None
        summaries.append({"ridge_lambda":float(lam),"Harrell":hn/hd,"Harrell_num":hn,"Harrell_den":hd,
                          "Uno60":un/ud if supported else None,"Uno_num":un,"Uno_den":ud,
                          "Uno60_complete_support":supported})
    chosen=max(summaries,key=lambda x:(x["Harrell"],x["Uno60"] if x["Uno60"] is not None else -float("inf"),x["ridge_lambda"]))
    return {"ridge_lambda":chosen["ridge_lambda"],"selection_rule":"fold_micro_Harrell_then_Uno60_then_stronger_ridge",
            "candidate_summaries":summaries,"uses_outer_test":False,"confidence_intervals_computed":False}


def frozen_feature_views(outputs,clinical_raw,fit_rows,evaluation_rows,*,arm,operator,background_seed=103017):
    """Clinical backgrounds use ONE frozen encoder; operator owns arm algebra.

    ``operator`` is the locked model module's frozen_readout_operator and must
    return the two-dimensional mechanism array, without changing main effects.
    """
    required=("additive_image","additive_gene","additive_clinical","visual_tokens","gene_tokens","gene_mask")
    aliases={"visual_tokens":"pre_residual_visual_tokens","gene_tokens":"pre_residual_gene_tokens"}
    values=dict(outputs)
    for key,other in aliases.items():
        if key not in values and other in values:
            values[key]=values[other]
    if any(k not in values for k in required):
        raise ValueError("FROZEN_ENCODER_OUTPUT_CONTRACT")
    arrays={k:np.asarray(values[k]) for k in required}
    arrays["gene_tokens"]=np.where(np.asarray(arrays["gene_mask"],bool)[...,None],arrays["gene_tokens"],0.)
    tr=np.asarray(fit_rows,int);ev=np.asarray(evaluation_rows,int)
    joined=np.r_[tr,ev]
    if not np.array_equal(np.sort(joined),np.arange(len(arrays["additive_image"]))):
        raise ValueError("FROZEN_FEATURE_SCOPE_MUST_ONLY_CONTAIN_CURRENT_FIT_AND_EVALUATION_ROWS")
    arrays["pre_residual_visual_tokens"]=arrays["visual_tokens"]
    arrays["pre_residual_gene_tokens"]=arrays["gene_tokens"]
    residual=arm in ("B5","B6a","B7","B5_CLINICAL_CAPACITY_CONTROL","B5_CLINICAL_CAPACITY",
                     "B5_CLINICAL_COMPLEMENTARITY","B6a_WRONG_AUX_RETRAIN","B7_NO_CROSS_PRODUCT")
    backgrounds=None
    if residual:
        visual=arrays["visual_tokens"];gene=arrays["gene_tokens"]
        combined=np.column_stack([visual.reshape(len(visual),-1),gene.reshape(len(gene),-1)])
        background=background_crossfit(arrays["additive_clinical"],combined,tr,ev,clinical_raw=clinical_raw,
                                      seed=background_seed,alpha=1.)
        mu=background["mean"]
        split=int(np.prod(visual.shape[1:]))
        backgrounds={"mu_visual":mu[:,:split].reshape(visual.shape),"mu_gene":mu[:,split:].reshape(gene.shape),
                     "audit":background["audit"],"state":background}
    mechanism=operator(arrays,None if backgrounds is None else backgrounds["mu_visual"],
                       None if backgrounds is None else backgrounds["mu_gene"],arm)
    if isinstance(mechanism,dict):
        mechanism=mechanism["interaction_features"]
    # Preserve slot/program identity. A shared Linear(6,64) followed by gene
    # mean pooling has rank <=6 (often only 3); whitening that mean into eight
    # apparent PCs would amplify float32 noise rather than create information.
    visual=arrays["visual_tokens"];gene=arrays["gene_tokens"]
    if visual.shape[1:]!=(8,64) or gene.shape[1:]!=(50,64):
        raise ValueError("PRIMARY_MAIN_EFFECT_REQUIRES_8_VISUAL_AND_50_GENE_TOKENS")
    valid_gene=np.asarray(arrays["gene_mask"],bool)
    gene=np.where(valid_gene[...,None],gene,0.)
    views={"pathology":visual.reshape(len(visual),-1),"omics":gene.reshape(len(gene),-1),
           "clinical":arrays["additive_clinical"],"mechanism":np.asarray(mechanism,np.float64)}
    return {"train":{k:v[tr] for k,v in views.items()},"evaluation":{k:v[ev] for k,v in views.items()},
            "background":backgrounds,"clinical_background_used_DSS":False,
            "one_encoder_coordinate_system":True,
            "main_effect_contract":"P8x64_and_G50x64_flatten_then_current_train_PCA8_not_mean_pool_G"}


@dataclass
class RiskCell:
    design:CompactDesign
    model:OffsetCox
    training_risk:np.ndarray
    evaluation_risk:np.ndarray
    receipt:dict


def fit_risk_cell(train_features,evaluation_features,train_time,train_event,evaluation_time,evaluation_event,
                  training_offset,evaluation_offset,*,ridge_lambda,role,bindings,selection_record=None):
    """One explicitly counted low-capacity risk fit, after frozen Phase A.

    Outer evaluation is gated by a previously frozen inner-selection receipt;
    no optimizer or preprocessing selection sees evaluation labels.
    """
    if role not in ("inner","outer_final","synthetic_qualification"):
        raise ValueError("UNKNOWN_P2_RISK_FIT_ROLE")
    if role=="outer_final":
        if (not isinstance(selection_record,dict) or selection_record.get("uses_outer_test")
                or selection_record.get("ridge_lambda")!=ridge_lambda):
            raise ValueError("OUTER_RISK_FIT_REQUIRES_FROZEN_INNER_SELECTION")
    projection=CompactDesign.fit(train_features)
    ztrain=projection.transform(train_features);zeval=projection.transform(evaluation_features)
    model=fit_offset_cox(ztrain,train_time,train_event,training_offset,ridge_lambda=ridge_lambda)
    fit_risk=model.predict(ztrain,training_offset);evaluation_risk=model.predict(zeval,evaluation_offset)
    fitted=evaluate_safe(train_time,train_event,train_time,train_event,fit_risk)
    evaluated=evaluate_safe(train_time,train_event,evaluation_time,evaluation_event,evaluation_risk)
    anchor=evaluate_safe(train_time,train_event,evaluation_time,evaluation_event,evaluation_offset)
    repair=risk_repair_damage(evaluation_time,evaluation_event,evaluation_risk,evaluation_offset,train_time,train_event)
    receipt={"schema":"semantic_b0b7_p2_risk_cell_v1","status":"FITTED_AND_EVALUATED",
             "role":role,"ridge_lambda":ridge_lambda,"risk_optimizer_fits":1,
             "DSS_coefficients":26,"source_and_scope_bindings":dict(bindings),
             "projection":projection.receipt,"risk_optimizer":model.receipt,
             "train_metrics":fitted,"evaluation_metrics":evaluated,"anchor_evaluation_metrics":anchor,
             "evaluation_risk_sha256":array_sha(evaluation_risk),"training_risk_sha256":array_sha(fit_risk),
             "evaluation_offset_sha256":array_sha(np.asarray(evaluation_offset)),
             "evaluation_time_sha256":array_sha(np.asarray(evaluation_time)),
             "evaluation_event_sha256":array_sha(np.asarray(evaluation_event)),
             "train_delta_sd":float(np.std(fit_risk-np.asarray(training_offset))),
             "train_anchor_sd":float(np.std(training_offset)),"repair_damage":repair,
             "selection_uses_evaluation_labels":role=="inner","confidence_intervals_computed":False}
    return RiskCell(projection,model,fit_risk,evaluation_risk,receipt)


def replay_risk_cell(cell,evaluation_features,evaluation_offset,train_time,train_event,evaluation_time,evaluation_event):
    """Reconstruct persisted linear-risk predictions; no neural encoder forward."""
    prediction=cell.model.predict(cell.design.transform(evaluation_features),evaluation_offset)
    maximum=float(np.max(np.abs(prediction-cell.evaluation_risk)))
    metric=evaluate_safe(train_time,train_event,evaluation_time,evaluation_event,prediction)
    if maximum>1e-12 or metric!=cell.receipt["evaluation_metrics"]:
        raise ValueError("P2_PERSISTED_RISK_REPLAY_MISMATCH")
    return {"actual_linear_head_replay":True,"neural_forward":False,"max_abs_risk_error":maximum,
            "fold_metric_components_equal":True,"risk_sha256":array_sha(prediction)}

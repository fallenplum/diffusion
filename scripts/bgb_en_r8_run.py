"""Finite R8 frozen-source qualification, convex cells and actual saved-risk replay."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="2"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from pathlib import Path
import json
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_r8_common import (parser,require,read,write,save,sha,canonical,validate_config,
    prepare_features,create_lock,target_context,load_features,metrics,risk_summary,ARMS,PARAMETERS,PARTITIONS,SEEDS)
from src.bgb_en_r8.cox import fit_protected_offset_cox,replay_cox_certificate
from src.bgb_en_r8.features import build_low_capacity_features
from src.phase10_trimodal.reference import pairs


def cell_name(arm,partition,seed):
    return f"{arm}_p{partition}_seed{seed}"


def source_kind(arm):
    return "RANDOM_NO_ST" if arm=="R8-8a_RANDOM_SOURCE" else "MATCHED"


def feature_matrix(payload,arm):
    return payload["transform"].features(payload["data"],arm,
        query_mapping=payload["query_mapping"] if arm=="R8-9_WRONG_PATIENT" else None)


def frozen_predictions(payload,arm,model,x,risk):
    scope=payload["scoped"];idx=scope["valid"];base=scope["base"]
    output=dict(residual_off=base.copy())
    if arm=="R8-5":
        d=payload["data"];require(d.wrong_conditioned_image_tokens is not None,
            "R8_5_FROZEN_WRONG_QUERY_EVIDENCE_MUST_BE_QUALIFIED")
        wrong=payload["transform"].conditional_pca8.transform(
            np.asarray(d.wrong_conditioned_image_tokens).reshape(d.rows,-1))
        output["wrong_query"]=model.predict(wrong,base)
    bilinear=arm in ("R8-6","R8-8a_RANDOM_SOURCE","R8-8b_WRONG_PROGRAM","R8-9_WRONG_PATIENT")
    if bilinear:
        no=x.copy();no[:,16:]=model.mean[16:]
        output["interaction_off"]=model.predict(no,base)
        t,d=payload["transform"],payload["data"]
        h=t.image_blocks2.transform(d.image_tokens);g=t.gene_blocks2.transform(d.gene_tokens,d.gene_program_mask)
        for name,query in (("matched_query",g),("wrong_query",g[payload["query_mapping"]]),
            ("train_mean_query",np.broadcast_to(g[scope["train"]].mean(0),g.shape))):
            z=build_low_capacity_features(h,g,mode="bilinear_rank2",query_gene2=query)
            require(np.array_equal(z[:,:16],x[:,:16]),"R8_FROZEN_INTERVENTION_CHANGED_RECIPIENT_MAIN_EFFECT")
            output[name]=model.predict(z,base)
    records={k:dict(frozen_not_retrained=True,
        metrics_vs_anchor=metrics(scope,idx,r[idx],base[idx]),
        full_relative_to_control=metrics(scope,idx,risk[idx],r[idx]),
        valid_risk_difference=risk_summary(risk[idx]-r[idx],base[idx]),
        recipient_gene_values_masks_RNA512_offset_and_images_unchanged=True)
        for k,r in output.items()}
    return output,records


def run_fit(args,ctx):
    require(args.arm in ARMS and args.partition in PARTITIONS and args.seed in SEEDS,
        "R8_CELL_OUTSIDE_PRELOCKED_MATRIX")
    payload,origin=load_features(args,ctx,args.partition,source_kind(args.arm))
    directory=args.private/"cells"/cell_name(args.arm,args.partition,args.seed)
    contract=dict(contract_sha256=ctx["lock"]["contract_sha256"],config_sha256=sha(args.config),
        feature_receipt_sha256=ctx["lock"]["feature_receipt_sha256"][f"p{args.partition}_{source_kind(args.arm)}"],
        arm=args.arm,partition=args.partition,seed=args.seed,source_state_sha256=origin["source_state_sha256"],
        registered_coefficients=PARAMETERS[args.arm],gamma=.1,ridge_lambda=1.,
        source_neural_updates=0,teacher_refits=0,target_neural_updates=0,outer_test_evaluated=False)
    if (directory/"COMPLETE.json").exists():
        prior=read(directory/"COMPLETE.json")
        require(args.resume and prior["status"]=="COMPLETE_R8_CONVEX_FIT" and prior["contract"]==contract
            and prior["artifact_sha256"]==sha(directory/"model.private.pt"),"R8_COMPLETED_FIT_CHANGED_NO_RETRY")
        return prior
    require(not directory.exists(),"R8_FAILED_OR_PARTIAL_CELL_REQUIRES_REVIEW_NO_AUTORETRY")
    directory.mkdir(parents=True);write(directory/"START.json",dict(status="STARTED_R8_CONVEX_FIT",contract=contract))
    started=time.monotonic()
    try:
        scope=payload["scoped"];tr,va=scope["train"],scope["valid"];x=feature_matrix(payload,args.arm)
        model=fit_protected_offset_cox(x[tr],scope["t"][tr],scope["e"][tr],scope["base"][tr],
            gamma=.1,ridge_lambda=1.,fit_identity_sha256=scope["original_train_identity_sha256"])
        require(model.receipt["registered_coefficients"]==PARAMETERS[args.arm],"R8_ACTUAL_PARAMETER_BUDGET_CHANGED")
        risk=model.predict(x,scope["base"]);delta=risk-scope["base"]
        controls,control_records=frozen_predictions(payload,args.arm,model,x,risk)
        saved=dict(contract=contract,model=model,risk=risk,delta=delta,
            frozen_predictions=controls,feature_matrix_sha256=canonical_feature(x),
            train=tr.copy(),valid=va.copy(),offset=scope["base"].copy())
        save(directory/"model.private.pt",saved)
        result=dict(status="COMPLETE_R8_CONVEX_FIT",contract=contract,arm=args.arm,partition=args.partition,seed=args.seed,
            actual_convex_fits=1,actual_neural_fits=0,neural_updates=0,source_refits=0,teacher_refits=0,
            registered_coefficients=PARAMETERS[args.arm],actual_fit=model.receipt,
            deterministic_solver_no_seed_randomization=True,nominal_seeds_are_not_independent_random_cohorts=True,
            metrics=metrics(scope,va,risk[va],scope["base"][va]),
            metrics_vs_CP=metrics(scope,va,risk[va],scope["CP"][va]),
            train_metrics=metrics(scope,tr,risk[tr],scope["base"][tr]),
            PFI_frozen_DSS_risk=metrics(scope,va,risk[va],scope["base"][va],pfi=True),
            risk_diagnostics=dict(train=risk_summary(delta[tr],scope["base"][tr]),
                                  validation=risk_summary(delta[va],scope["base"][va])),
            frozen_controls=control_records,feature_matrix_sha256=saved["feature_matrix_sha256"],
            source_origin=origin["source_origin"],source_version=origin["source_version"],
            source_program_definition_sha256=origin["source_program_definition_sha256"],
            PCA=payload["transform"].audit(),feature_receipt_sha256=contract["feature_receipt_sha256"],
            artifact_sha256=sha(directory/"model.private.pt"),elapsed_seconds=time.monotonic()-started,
            new_outer_evaluation=False,PFI_model_selected_or_trained=False)
        write(directory/"COMPLETE.json",result);return result
    except BaseException as error:
        write(directory/"ERROR_RECEIPT.json",dict(status="FAILED_R8_CELL_NO_AUTOMATIC_RETRY",contract=contract,
            failure_type=type(error).__name__,elapsed_seconds=time.monotonic()-started))
        (directory/"error.private.log").write_text(traceback.format_exc())
        raise


def canonical_feature(x):
    from src.bgb_en.protocol import array_hash
    return array_hash(np.asarray(x,np.float64))


def references(args,ctx,partition):
    payload,_=load_features(args,ctx,partition,"MATCHED");s=payload["scoped"];va=s["valid"]
    out=[]
    for arm,key in (("R8-0","CP"),("R8-1","base")):
        for seed in SEEDS:
            value=dict(status="COMPLETE_R8_ZERO_FIT_REFERENCE",arm=arm,partition=partition,seed=seed,
                contract_sha256=ctx["lock"]["contract_sha256"],actual_convex_fits=0,actual_neural_fits=0,
                actual_new_teacher_fits=0,metrics=metrics(s,va,s[key][va],s["base"][va]),
                PFI_frozen_DSS_risk=metrics(s,va,s[key][va],s["base"][va],pfi=True),
                exact_saved_CP_RNA_teacher_reference=True)
            p=args.private/"references"/(cell_name(arm,partition,seed)+".json")
            if p.exists():require(read(p)==value,"R8_SAVED_BASELINE_REFERENCE_CHANGED")
            else:write(p,value)
            out.append(value)
    return out


def _credit(time,event,risk):
    i,j=pairs(time,event);d=np.asarray(risk)[i]-np.asarray(risk)[j]
    return np.where(np.abs(d)<=1e-8,.5,(d>0).astype(float))


def _metric_error(a,b):
    require(set(a)==set(b),"R8_REPLAY_METRIC_KEYS_CHANGED")
    values=[]
    for k in a:
        require((a[k] is None)==(b[k] is None),"R8_REPLAY_METRIC_DEFINED_STATUS_CHANGED")
        if a[k] is not None:values.append(abs(float(a[k])-float(b[k])))
    return max(values,default=0.)


def replay_one(args,ctx,arm,part,seed):
    directory=args.private/"cells"/cell_name(arm,part,seed);done=read(directory/"COMPLETE.json")
    require(done["status"]=="COMPLETE_R8_CONVEX_FIT" and done["contract"]["contract_sha256"]==ctx["lock"]["contract_sha256"]
        and done["artifact_sha256"]==sha(directory/"model.private.pt") and not (directory/"ERROR_RECEIPT.json").exists(),
        "R8_REPLAY_INCOMPLETE_CHANGED_OR_FAILED_MODEL")
    require(done["arm"]==arm and done["partition"]==part and done["seed"]==seed
            and done["actual_convex_fits"]==1 and done["actual_neural_fits"]==done["neural_updates"]==0
            and done["registered_coefficients"]==PARAMETERS[arm],"R8_REPLAY_CELL_IDENTITY_OR_BUDGET_CHANGED")
    saved=torch.load(directory/"model.private.pt",map_location="cpu",weights_only=False)
    require(saved["contract"]==done["contract"],"R8_REPLAY_SAVED_CONTRACT_CHANGED")
    payload,meta=load_features(args,ctx,part,source_kind(arm));s=payload["scoped"];tr,va=s["train"],s["valid"]
    x=feature_matrix(payload,arm);model=saved["model"]
    require(canonical_feature(x)==done["feature_matrix_sha256"]==saved["feature_matrix_sha256"],
            "R8_REPLAY_FROZEN_FEATURES_CHANGED")
    require(model.gamma==.1 and model.ridge_lambda==1. and model.fit_identity_sha256==s["original_train_identity_sha256"]
        and np.array_equal(saved["offset"],s["base"]) and np.array_equal(saved["train"],tr) and np.array_equal(saved["valid"],va),
        "R8_REPLAY_OFFSET_TRAIN_SCOPE_OR_RECIPE_CHANGED")
    certificate=replay_cox_certificate(model,x[tr],s["t"][tr],s["e"][tr],s["base"][tr])
    risk=model.predict(x,s["base"]);error=float(np.max(np.abs(risk-saved["risk"])))
    changed=sum(int(np.sum(_credit(s["t"][idx],s["e"][idx],risk[idx])!=
                          _credit(s["t"][idx],s["e"][idx],saved["risk"][idx]))) for idx in (tr,va))
    require(error<=2e-10 and changed==0,"R8_ACTUAL_SAVED_RISK_OR_PAIR_CREDITS_DID_NOT_REPLAY")
    m=metrics(s,va,risk[va],s["base"][va]);tm=metrics(s,tr,risk[tr],s["base"][tr])
    pfi=metrics(s,va,risk[va],s["base"][va],pfi=True)
    errors=[_metric_error(m,done["metrics"]),_metric_error(tm,done["train_metrics"]),_metric_error(pfi,done["PFI_frozen_DSS_risk"])]
    controls,records=frozen_predictions(payload,arm,model,x,risk)
    require(set(controls)==set(saved["frozen_predictions"])==set(done["frozen_controls"]),"R8_REPLAY_FROZEN_CONTROL_INVENTORY_CHANGED")
    for name,value in controls.items():
        errors.append(float(np.max(np.abs(value-saved["frozen_predictions"][name]))))
        errors.append(_metric_error(records[name]["full_relative_to_control"],done["frozen_controls"][name]["full_relative_to_control"]))
    require(max(errors)<=2e-10,"R8_REPLAY_METRICS_OR_FROZEN_CONTROLS_CHANGED")
    return dict(status="PASS_ACTUAL_R8_SAVED_COEFFICIENT_AND_FEATURE_REPLAY",arm=arm,partition=part,seed=seed,
        actual_saved_risk_forward_replayed=True,maximum_risk_error=error,maximum_metric_or_control_error=max(errors),
        changed_pair_credits=changed,convex_KKT=certificate,new_optimizer_fits=0,new_neural_updates=0,
        completion_sha256=sha(directory/"COMPLETE.json"),artifact_sha256=done["artifact_sha256"],
        feature_receipt_sha256=done["feature_receipt_sha256"]),risk


COMPARISONS=(("R8-2","R8-1"),("R8-3","R8-2"),("R8-4","R8-1"),("R8-5","R8-3"),
    ("R8-6","R8-1"),("R8-6","R8-4"),("R8-6","R8-8a_RANDOM_SOURCE"),
    ("R8-6","R8-8b_WRONG_PROGRAM"),("R8-6","R8-9_WRONG_PATIENT"),("R8-6","R8-10_NO_INTERACTION"))


def replay_all(args,ctx):
    planned={cell_name(a,p,s) for a in ARMS for p in PARTITIONS for s in SEEDS}
    found={p.parent.name for p in (args.private/"cells").glob("*/COMPLETE.json")}
    require(found==planned and not list((args.private/"cells").glob("*/ERROR_RECEIPT.json")),
        "R8_ALL_81_FITS_REQUIRED_BEFORE_COMPLETE_REPLAY")
    require(not (args.private/"REPLAY_AUDIT.json").exists(),"R8_EXISTING_REPLAY_REQUIRES_EXPLICIT_REVIEW")
    reference_rows=[]
    for part in PARTITIONS:
        payload,_=load_features(args,ctx,part,"MATCHED");s=payload["scoped"];va=s["valid"]
        for arm,key in (("R8-0","CP"),("R8-1","base")):
            for seed in SEEDS:
                path=args.private/"references"/(cell_name(arm,part,seed)+".json");v=read(path)
                require(v["status"]=="COMPLETE_R8_ZERO_FIT_REFERENCE"
                    and v["contract_sha256"]==ctx["lock"]["contract_sha256"]
                    and v["arm"]==arm and v["partition"]==part and v["seed"]==seed
                    and v["actual_convex_fits"]==v["actual_neural_fits"]==0,
                    "R8_ZERO_FIT_BASELINE_REFERENCE_CHANGED")
                err=max(_metric_error(v["metrics"],metrics(s,va,s[key][va],s["base"][va])),
                    _metric_error(v["PFI_frozen_DSS_risk"],metrics(s,va,s[key][va],s["base"][va],pfi=True)))
                require(err<=2e-10,"R8_FROZEN_CP_SCALAR_REFERENCE_METRICS_DID_NOT_REPLAY")
                reference_rows.append(dict(arm=arm,partition=part,seed=seed,maximum_metric_error=err,
                    actual_baseline_values_replayed=True,reference_sha256=sha(path),new_optimizer_fits=0))
    rows=[];risks={}
    for arm in ARMS:
        for part in PARTITIONS:
            for seed in SEEDS:
                terminal=args.private/"REPLAY_PRIVATE"/cell_name(arm,part,seed)
                require(not terminal.with_suffix(".PASS.json").exists() and not terminal.with_suffix(".FAIL.json").exists(),
                    "R8_EXISTING_REPLAY_TERMINAL_PRESERVED_NO_AUTORETRY")
                try:
                    row,risk=replay_one(args,ctx,arm,part,seed)
                    write(terminal.with_suffix(".PASS.json"),row)
                except BaseException as error:
                    write(terminal.with_suffix(".FAIL.json"),dict(status="FAIL_R8_SAVED_MODEL_REPLAY_NO_AUTORETRY",
                        arm=arm,partition=part,seed=seed,contract_sha256=ctx["lock"]["contract_sha256"],
                        failure_type=type(error).__name__,new_optimizer_fits=0))
                    terminal.with_suffix(".error.private.log").write_text(traceback.format_exc())
                    raise
                rows.append(row);risks[(arm,part,seed)]=risk
                print(json.dumps(dict(replayed_unit=cell_name(arm,part,seed),status=row["status"])),flush=True)
    direct=[]
    for part in PARTITIONS:
        payload,_=load_features(args,ctx,part,"MATCHED");scope=payload["scoped"];va=scope["valid"]
        for seed in SEEDS:
            for candidate,reference in COMPARISONS:
                a=risks[(candidate,part,seed)]
                b=scope["base"] if reference=="R8-1" else risks[(reference,part,seed)]
                direct.append(dict(candidate=candidate,reference=reference,partition=part,seed=seed,
                    candidate_vs_reference=metrics(scope,va,a[va],b[va]),
                    PFI_frozen_DSS_risk=metrics(scope,va,a[va],b[va],pfi=True),
                    direct_actual_saved_risk_pair_credits=True,bootstrap_CI_or_pvalue_computed=False))
    seed_checks=[]
    for arm in ARMS:
        for part in PARTITIONS:
            current=[risks[(arm,part,seed)] for seed in SEEDS]
            seed_checks.append(dict(arm=arm,partition=part,
                bitwise_identical_predictions=all(np.array_equal(current[0],r) for r in current[1:]),
                maximum_absolute_difference=max(float(np.max(np.abs(current[0]-r))) for r in current[1:]),
                identical_deterministic_repetitions_are_not_independent_random_evidence=True))
    value=dict(status="PASS_COMPLETE_R8_81_SAVED_RISK_REPLAYS",contract_sha256=ctx["lock"]["contract_sha256"],
        actual_saved_models=81,passed_units=81,failed_units=0,units=rows,paired_retrained_controls=direct,
        zero_fit_reference_replays=reference_rows,deterministic_seed_checks=seed_checks,
        all_complete_81_executed=True,source_or_teacher_refits=0,new_optimizer_fits=0,
        source_family_selected_by_DSS=False,outer_test_evaluated=False)
    write(args.private/"REPLAY_AUDIT.json",value);return value


def main():
    p=parser(__doc__);p.add_argument("--mode",choices=("features","lock","fit","references","replay","matrix"),required=True)
    p.add_argument("--partition",type=int,choices=PARTITIONS);p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--arm",choices=ARMS);args=p.parse_args()
    if args.mode=="features":
        value=prepare_features(args);print(json.dumps({k:value[k] for k in ("status","feature_families","target_convex_fits")}));return
    if args.mode=="lock":
        value=create_lock(args);print(json.dumps({k:value[k] for k in ("status","contract_sha256","planned_convex_fits")}));return
    ctx=target_context(args)
    if args.mode=="fit":value=run_fit(args,ctx)
    elif args.mode=="references":
        require(args.partition in PARTITIONS,"R8_REFERENCE_PARTITION_REQUIRED");value=dict(status="COMPLETE_R8_REFERENCES",rows=len(references(args,ctx,args.partition)))
    elif args.mode=="replay":value=replay_all(args,ctx)
    else:
        for part in PARTITIONS:references(args,ctx,part)
        for arm in ARMS:
            for part in PARTITIONS:
                for seed in SEEDS:
                    args.arm,args.partition,args.seed=arm,part,seed
                    cell=run_fit(args,ctx);print(json.dumps(dict(unit=cell_name(arm,part,seed),status=cell["status"])),flush=True)
        value=replay_all(args,ctx)
    print(json.dumps({k:value[k] for k in ("status","actual_saved_models","passed_units","failed_units") if k in value}))


if __name__=="__main__":main()

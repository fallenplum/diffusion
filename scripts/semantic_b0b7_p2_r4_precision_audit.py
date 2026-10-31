"""Read-only numeric provenance diagnostic; no PCA/SVD, Cox or neural fitting.

Compare already saved states and reproduce only fixed arithmetic/mean/std/median
reductions from authorized covariates. No endpoint array or new score is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import semantic_b0b7_p2_r4_history_audit as history
from scripts.semantic_b0b7_p2_dss_isolation_audit import fingerprint

MASTER_SHA = history.MASTER_SHA
HISTORY_AUDIT_SHA = "006eb262b1a097e47b64f5a1cc4c0c4e026d44d079858f9e04b239aff1e2f46d"
require, sha, read = history.require, history.sha, history.read


def array_metadata(value):
    a = np.asarray(value)
    return {"dtype":str(a.dtype),"shape":list(a.shape),"C_contiguous":bool(a.flags.c_contiguous),
            "F_contiguous":bool(a.flags.f_contiguous),"value_sha256":fingerprint(a)}


def difference(a, b):
    x, y = np.asarray(a), np.asarray(b)
    result = {"old":array_metadata(x),"current":array_metadata(y),
              "dtype_equal":x.dtype == y.dtype,"shape_equal":x.shape == y.shape,
              "exact_normalized_value_bytes":fingerprint(x) == fingerprint(y)}
    if x.shape == y.shape:
        numeric = x.dtype.kind in "biufc" and y.dtype.kind in "biufc"
        if numeric:
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            finite = np.isfinite(xf) & np.isfinite(yf)
            result["nonfinite_pattern_equal"] = bool(np.array_equal(np.isfinite(xf), np.isfinite(yf)))
            result["changed_finite_values"] = int((xf[finite] != yf[finite]).sum())
            result["max_abs_difference"] = float(np.max(np.abs(xf[finite]-yf[finite]))) if finite.any() else None
        else:
            result["changed_values"] = int(np.count_nonzero(x != y))
    return result


def leaf_comparisons(old, current, path):
    output = []
    def visit(a, b, at):
        if isinstance(a, dict) and isinstance(b, dict):
            require(set(a) == set(b), "PREPROCESSING_DICTIONARY_KEYS_CHANGED")
            # Saved insertion order is explicit; first is not a causal claim.
            for key in a: visit(a[key], b[key], at+"."+str(key))
        elif isinstance(a, (tuple,list)) and isinstance(b, (tuple,list)):
            require(type(a) is type(b) and len(a) == len(b), "PREPROCESSING_SEQUENCE_STRUCTURE_CHANGED")
            for i, (x,y) in enumerate(zip(a,b)): visit(x,y,at+"."+str(i))
        else:
            output.append({"path":at,**difference(a,b)})
    visit(old,current,path)
    different = [row for row in output if not row["exact_normalized_value_bytes"]]
    return {"leaf_count":len(output),"different_leaf_count":len(different),
            "all_leaves_exact":not different,"first_different_saved_order_leaf":different[0]["path"] if different else None,
            "different_leaves":different}


def direct_scale(x, fit):
    a = np.asarray(x,np.float64)[fit]
    mean, sd = a.mean(0), a.std(0)
    return {"mean":mean,"sd":np.where(sd > 1e-8,sd,1.)}


def compare_reductions(original, copied, fit, local_fit, old_state, current_state):
    old_scale, new_scale = direct_scale(original,fit), direct_scale(copied,local_fit)
    by_stat = {}
    for key in ("mean","sd"):
        by_stat[key] = {"original_vs_stored_old":difference(old_scale[key],old_state[key]),
                        "copy_vs_stored_current":difference(new_scale[key],current_state[key]),
                        "original_vs_copy_reduction":difference(old_scale[key],new_scale[key]),
                        "stored_old_vs_current":difference(old_state[key],current_state[key])}
    return {"original_whole":array_metadata(original),"copy_whole":array_metadata(copied),
            "original_training_view":array_metadata(np.asarray(original)[fit]),
            "copy_training_view":array_metadata(np.asarray(copied)[local_fit]),"statistics":by_stat}


def molecular_reductions(original, copied, fit, local_fit, old_state, current_state):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        old_median = np.nanmedian(original[fit],axis=0)
        new_median = np.nanmedian(copied[local_fit],axis=0)
    old_median = np.where(np.isfinite(old_median),old_median,0.)
    new_median = np.where(np.isfinite(new_median),new_median,0.)
    old_filled = np.where(np.isfinite(original),original,old_median)
    new_filled = np.where(np.isfinite(copied),copied,new_median)
    stats = compare_reductions(old_filled,new_filled,fit,local_fit,old_state["scale"],current_state["scale"])
    old_z = (old_filled-old_state["scale"]["mean"])/old_state["scale"]["sd"]
    new_z = (new_filled-current_state["scale"]["mean"])/current_state["scale"]["sd"]
    old_z = np.where(np.isfinite(original)&(old_state["measured"][None]>0),old_z,0.).astype(np.float32)
    new_z = np.where(np.isfinite(copied)&(current_state["measured"][None]>0),new_z,0.).astype(np.float32)
    old_r = old_z[:,old_state["selected"]].astype(np.float64)
    new_r = new_z[:,current_state["selected"]].astype(np.float64)
    old_r = (old_r-old_state["selected_scale"]["mean"])/old_state["selected_scale"]["sd"]
    new_r = (new_r-current_state["selected_scale"]["mean"])/current_state["selected_scale"]["sd"]
    return {"original_raw_layout":array_metadata(original),"copy_raw_layout":array_metadata(copied),
            "old_median_vs_stored":difference(old_median,old_state["median"]),
            "copy_median_vs_stored":difference(new_median,current_state["median"]),
            "filled_scale_reductions":stats,
            "R512_training_input":difference(old_r[fit],new_r[local_fit]),
            "R512_training_layouts":{"old":array_metadata(old_r[fit]),"current":array_metadata(new_r[local_fit])},
            "member_selection_repeated":False}


def run_audit(master_path, expected_master_sha, *, assets_path, legacy_root, historical_root,
              formal_private, original_audit_path, output, formal_complete_authorized=False):
    require(formal_complete_authorized,"EXPLICIT_POSTFORMAL_PRECISION_AUDIT_REQUIRED_BEFORE_INPUT_READ")
    require(expected_master_sha == MASTER_SHA and sha(master_path) == MASTER_SHA,"PRECISION_MASTER_LOCK_CHANGED")
    master = read(master_path); history.qualification.source_set(master)
    count = history.assert_formal_complete(formal_private,master)
    require(sha(original_audit_path) == HISTORY_AUDIT_SHA,"ORIGINAL_HISTORY_AUDIT_CHANGED")
    original_audit = read(original_audit_path)
    require(not Path(output).exists(),"PRECISION_OUTPUT_MUST_BE_NEW")
    require(sha(assets_path) == master["assets"]["assets_json_sha256"],"PRECISION_ASSETS_CHANGED")
    from scripts.semantic_b0b7_prepare import legacy_data_module
    assets = read(assets_path); loader,_ = legacy_data_module(legacy_root)
    raw = loader.load_raw(assets,include_endpoints=False)
    require(raw["audit"]["endpoint_read"] is False,"UNEXPECTED_ENDPOINT_ACCESS")
    # Match the original R4 driver's full-GRID source, not generic GLOBAL_ALL.
    full_mean = Path(assets["patch_grid"])/"full_mean.npy"
    require(sha(full_mean) == master["B0_R4_EXACT"]["full_CONCH_mean_sha256"],"ORIGINAL_FULL_CONCH_SOURCE_CHANGED")
    raw["conch_mean"] = np.asarray(np.load(full_mean,mmap_mode="r"),np.float64)
    folds = []
    for fold in range(1,6):
        old, complete = history.verified_historical_load(Path(historical_root)/"outer"/("fold"+str(fold)))
        current_path = Path(formal_private)/"anchors"/(f"outer{fold}_final.private.pkl")
        receipt_path = current_path.with_suffix(".json"); receipt=read(receipt_path)
        require(receipt["p2_lock_sha256"] == MASTER_SHA and sha(current_path) == receipt["artifact_sha256"],"CURRENT_ANCHOR_BYTES_CHANGED")
        anchor = history.independent.passive_pickle(current_path)
        fit, evaluation = np.flatnonzero(raw["folds"] != fold),np.flatnonzero(raw["folds"] == fold)
        history.check_row_order(old,anchor,fit,evaluation)
        selected = np.sort(np.r_[fit,evaluation]); local_fit=np.searchsorted(selected,fit)
        require(np.array_equal(selected,np.arange(512)),"OUTER_ALL_PATIENT_COVARIATE_SCOPE_CHANGED")
        before, after = old["full_preprocessing"],anchor.state["full"]["preprocessing"]
        groups = {name:leaf_comparisons(before[name],after[name],"preprocessing."+name)
                  for name in ("clinical","S32","molecular")}
        for visual in ("LOCAL","GLOBAL"):
            groups["visual_"+visual] = leaf_comparisons(before["visual"][visual],after["visual"][visual],"preprocessing.visual."+visual)
        layout = {}
        for key, state_path in (("S32",("S32",)),("conch_local",("visual","LOCAL","before")),
                                ("conch_mean",("visual","GLOBAL","before"))):
            x=np.asarray(raw[key],np.float64);y=x[selected].copy();old_scale,new_scale=before,after
            for part in state_path:old_scale,new_scale=old_scale[part],new_scale[part]
            layout[key]=compare_reductions(x,y,fit,local_fit,old_scale,new_scale)
            old_z=(x-old_scale["mean"])/old_scale["sd"]
            new_z=(y-new_scale["mean"])/new_scale["sd"]
            layout[key]["standardized_training_input"]=difference(old_z[fit],new_z[local_fit])
        x=np.asarray(raw["rna"],np.float64);y=x[selected].copy()
        molecular=molecular_reductions(x,y,fit,local_fit,before["molecular"],after["molecular"])
        model_differences = {key:leaf_comparisons(
            {k:old["full_models"][key][k] for k in ("coef","scale")},
            {k:anchor.state["full"]["full_models"][key][k] for k in ("coef","scale")},"full_models."+key)
            for key in ("LOCAL","R")}
        raw_layouts={key:{"original":array_metadata(np.asarray(raw[key])),
                         "partition_copy":array_metadata(np.asarray(raw[key])[selected].copy())} for key in ("rna","cna")}
        raw_reduction_copy_equal = all(
            all(row["original_vs_copy_reduction"]["exact_normalized_value_bytes"] and
                row["original_vs_stored_old"]["exact_normalized_value_bytes"] and
                row["copy_vs_stored_current"]["exact_normalized_value_bytes"] for row in detail["statistics"].values())
            for detail in layout.values())
        standardized_visual_input_equal=all(layout[key]["standardized_training_input"]["exact_normalized_value_bytes"]
                                            for key in ("conch_local","conch_mean"))
        direct_layout_effect=any(not row["original_vs_copy_reduction"]["exact_normalized_value_bytes"]
                                 for detail in layout.values() for row in detail["statistics"].values())
        folds.append({"fold":fold,"historical_artifact_sha256":complete["artifact_sha256"],
            "current_artifact_sha256":receipt["artifact_sha256"],"groups":groups,
            "layout_and_direct_reduction_checks":layout,"molecular_reduction_checks":molecular,
            "raw_omics_layouts":raw_layouts,"full_model_differences":model_differences,
            "copy_layout_effect_detected_in_direct_S32_or_visual_before_reductions":direct_layout_effect,
            "raw_copy_reductions_exact_and_match_both_saved_states":raw_reduction_copy_equal,
            "standardized_visual_PCA_training_inputs_exact":standardized_visual_input_equal})
        del old,anchor
    all_reductions=all(f["raw_copy_reductions_exact_and_match_both_saved_states"] for f in folds)
    all_pca_inputs=all(f["standardized_visual_PCA_training_inputs_exact"] for f in folds)
    only_visual=all(all(f["groups"][group]["all_leaves_exact"] for group in ("clinical","S32","molecular")) for f in folds)
    status=("RAW_COPY_LAYOUT_REDUCTION_HYPOTHESIS_NOT_SUPPORTED_FIRST_DIVERGENCE_SAVED_PCA_OUTPUT"
            if all_reductions and all_pca_inputs and only_visual else "READONLY_NUMERIC_DIVERGENCE_LOCALIZED_REQUIRES_COMPONENT_REVIEW")
    report={"schema":"semantic_b0b7_p2_r4_readonly_precision_diagnostic_v1","status":status,
        "master_lock_sha256":MASTER_SHA,"source_sha256":sha(__file__),"original_history_audit_sha256":HISTORY_AUDIT_SHA,
        "original_history_source_sha256":sha(history.__file__),"original_audit_unchanged":sha(original_audit_path)==HISTORY_AUDIT_SHA,
        "formal_source_fits_unchanged":count,"folds":folds,
        "original_history_risk_max_abs_error":max(f["max_abs_risk_difference"] for f in original_audit["folds"]),
        "original_history_metric_components_still_exact":all(f["historical_seed31_unified_metrics"]==f["current_B0_unified_metrics"] for f in original_audit["folds"]),
        "first_difference_order":"saved_state_dictionary_order_not_a_new_fitting_trace",
        "copy_layout_hypothesis_test_scope":"original_C_or_F_covariates_vs_exact_outer_selected_copy_C_order_and_direct_train_reductions",
        "no_direct_proof_of_specific_historical_BLAS_LAPACK_version_or_thread_cause":True,
        "interpretation":"Same observed standardized PCA training inputs with differences first in stored PCA basis is evidence for a numerical linear-algebra path difference; it does not identify the historical backend or justify rewriting exact flags.",
        "statistical_reductions_and_fixed_transforms_only":True,"new_PCA_or_SVD_fits":0,"new_Cox_fits":0,
        "new_neural_fits":0,"new_neural_forwards":0,"new_risk_predictions_or_scoring_calls":0,
        "endpoint_arrays_read":False,"DSS_PFI_OS_labels_read":False,
        "historical_blob_can_decode_existing_aggregate_metrics_not_used":True,
        "learned_parameters_or_predictions_changed":False,"historical_or_current_artifacts_rewritten":False,
        "confidence_intervals_computed":False,"p_values_computed":False}
    history.qualification.public_write(output,report)
    return {"status":status,"folds":len(folds),"report_sha256":sha(output),"new_fits":0,"endpoint_arrays_read":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("master-lock","expected-master-sha","assets","legacy-root","historical-root","formal-private","original-audit","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--formal-complete-authorized",action="store_true")
    a=parser.parse_args()
    print(json.dumps(run_audit(a.master_lock,a.expected_master_sha,assets_path=a.assets,legacy_root=a.legacy_root,
        historical_root=a.historical_root,formal_private=a.formal_private,original_audit_path=a.original_audit,
        output=a.output,formal_complete_authorized=a.formal_complete_authorized)))


if __name__ == "__main__":main()

"""Post-formal, zero-fit comparison against original R4 seed-31 outer risks.

Historical own-study Torch containers are decoded only after their COMPLETE
checksums match. They can contain already-computed PFI aggregate metrics; those
fields are never used, and no original PFI label array is loaded by this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import semantic_b0b7_p2_leakage_audit as qualification
from scripts import verify_semantic_b0b7_p2 as independent
from scripts.semantic_b0b7_p2_dss_isolation_audit import fingerprint

MASTER_SHA = "b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
HISTORICAL_PRODUCER_SHA = "c016a7434d89243bb1971bb1e81df00d6502a0708077c14318c17b2cb70ad568"
require, sha, read = qualification.require, qualification.sha, qualification.read


def old_array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def lineage_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def assert_formal_complete(private, master):
    private = Path(private)
    require(not (private/"ACTIVE_SCHEDULER_OWNER.private.json").exists(), "FORMAL_SCHEDULER_STILL_ACTIVE")
    count = 0
    for outer in range(1, 6):
        for index in (*range(3), None):
            split = f"outer{outer}_"+("final" if index is None else f"inner{index}")
            for family in master["family_readouts"]:
                for seed in (31, 47, 83):
                    folder = private/"phase_a"/split/(family+"_seed"+str(seed))
                    receipt = read(folder/"cell.private.json")
                    require(receipt["status"] == "COMPLETE" and receipt["DSS_OS_PFI_accessed"] is False and
                            receipt["master_lock_sha256"] == qualification.canonical(master["phase_a_lock"]) and
                            receipt["frozen_sources"] == master["phase_a_lock"]["frozen_sources"] and
                            (folder/"RISK_BUNDLE_COMPLETE.private.json").is_file(), "FORMAL_MATRIX_NOT_FULLY_COMPLETE")
                    count += 1
    require(count == 480, "FORMAL_MATRIX_CARDINALITY_CHANGED")
    return count


def verified_historical_load(folder):
    folder = Path(folder)
    complete = read(folder/"COMPLETE.json")
    path = folder/"models.pt"
    require(complete["status"] == "COMPLETE_OUTER_BASELINES" and complete["selection_used_outer_test"] is False and
            sha(path) == complete["artifact_sha256"], "HISTORICAL_ARTIFACT_NOT_HASH_QUALIFIED")
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(value["contract"] == complete["contract"], "HISTORICAL_EMBEDDED_CONTRACT_DIFFERS_FROM_COMPLETE")
    return value, complete


def validate_contract(contract, master, expected_modules, fit, evaluation):
    expected = {"cohort_sha256": master["assets"]["cohort_sha256"],
                "endpoints_sha256": master["assets"]["endpoints_sha256"],
                "visual_sources_sha256": master["assets"]["visual_sources_sha256"],
                "full_CONCH_mean_sha256": master["B0_R4_EXACT"]["full_CONCH_mean_sha256"],
                "source_sha256": HISTORICAL_PRODUCER_SHA,
                "train_sha256": old_array_hash(fit), "test_sha256": old_array_hash(evaluation)}
    for key, value in expected.items():
        require(contract[key] == value, "HISTORICAL_CONTRACT_BINDING_CHANGED:"+key)
    require(contract["module_sha256"] == expected_modules, "HISTORICAL_DIRECT_MODULE_SOURCES_CHANGED")
    lineage = contract["source_lineage"]
    require(lineage_hash(lineage) == contract["source_lineage_sha256"] and
            lineage["cohort_sha256"] == expected["cohort_sha256"] and
            lineage["endpoints_sha256"] == expected["endpoints_sha256"] and
            lineage["GMT_sha256"] == master["assets"]["GMT_sha256"] and
            lineage["legacy_group_sha256"] == master["assets"]["legacy_groups_sha256"],
            "HISTORICAL_LINEAGE_OR_GENE_MEMBERSHIP_CHANGED")


def check_row_order(historical, anchor, fit, evaluation):
    require(np.array_equal(historical["train"], fit) and np.array_equal(historical["test"], evaluation) and
            np.array_equal(anchor.train_rows, fit) and np.array_equal(anchor.evaluation_rows, evaluation),
            "HISTORICAL_OR_CURRENT_CANONICAL_ROW_ORDER_CHANGED")


def vector_comparison(historical_risk, current_risk, old_beta, new_beta):
    a, b = np.asarray(historical_risk), np.asarray(current_risk)
    require(a.shape == b.shape and a.ndim == 1 and np.isfinite(a).all() and np.isfinite(b).all(),
            "HISTORICAL_CURRENT_RISK_SHAPE_OR_FINITE_CHECK_FAILED")
    return {"risk_values_exact": bool(np.array_equal(a, b)),
            "risk_dtype_and_bytes_exact": fingerprint(a) == fingerprint(b),
            "max_abs_risk_difference": float(np.max(np.abs(a-b))),
            "risk_RMS_difference": float(np.sqrt(np.mean((a-b)**2))),
            "historical_risk_sha256": independent.array_sha(a), "current_risk_sha256": independent.array_sha(b),
            "historical_beta_sha256": fingerprint(float(old_beta)),
            "current_beta_sha256": fingerprint(float(new_beta)),
            "beta_exact": fingerprint(float(old_beta)) == fingerprint(float(new_beta)),
            "beta_abs_difference": abs(float(old_beta)-float(new_beta))}


def micro(rows, name):
    require(len(rows) == 5 and {row["fold"] for row in rows} == set(range(1, 6)), "HISTORY_FIVE_FOLDS_REQUIRED")
    result = {"seed": 31, "independent_refit_seeds": 1, "cross_fold_risk_pairs": False}
    for metric, nk, dk in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
        supported = all(row[name][nk] is not None and row[name][dk] is not None and row[name][dk] > 0 for row in rows)
        n = math.fsum(row[name][nk] for row in rows) if supported else None
        d = math.fsum(row[name][dk] for row in rows) if supported else None
        result.update({metric: n/d if supported else None, nk:n, dk:d})
    return result


def run_audit(master_path, expected_master_sha, *, assets_path, legacy_root, historical_root,
              formal_private, output, formal_complete_authorized=False):
    require(formal_complete_authorized, "EXPLICIT_POSTFORMAL_HISTORY_AUDIT_AUTHORIZATION_REQUIRED_BEFORE_INPUT_READ")
    require(expected_master_sha == MASTER_SHA and sha(master_path) == MASTER_SHA, "P2_MASTER_LOCK_CHANGED")
    master = read(master_path)
    qualification.source_set(master)
    formal_count = assert_formal_complete(formal_private, master)
    require(not Path(output).exists(), "HISTORY_AUDIT_OUTPUT_MUST_BE_NEW")
    require(sha(assets_path) == master["assets"]["assets_json_sha256"], "HISTORY_CURRENT_ASSETS_CHANGED")
    assets = read(assets_path)
    patient, folds, time, event = independent.canonical_labels(assets["cohort"], assets["endpoints"])
    legacy_root = Path(legacy_root)
    require(sha(legacy_root/"scripts/bgb_en_outer.py") == HISTORICAL_PRODUCER_SHA, "HISTORICAL_PRODUCER_SOURCE_CHANGED")
    expected_modules = {name: sha(legacy_root/"src/bgb_en"/name)
                        for name in ("protocol.py", "baselines.py", "survival.py", "data.py", "omics.py")}
    for relative, expected in master["B0_R4_EXACT"]["legacy_source_sha256"].items():
        require(sha(legacy_root/relative) == expected, "CURRENT_PINNED_R4_DEPENDENCY_CHANGED")
    require(expected_modules["data.py"] == master["assets"]["legacy_data_source_sha256"], "CURRENT_R4_READER_CHANGED")
    results = []
    for fold in range(1, 6):
        historical_folder = Path(historical_root)/"outer"/("fold"+str(fold))
        historical, complete = verified_historical_load(historical_folder)
        fit, evaluation = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        require(complete["fold"] == fold, "HISTORICAL_FOLD_ID_CHANGED")
        validate_contract(historical["contract"], master, expected_modules, fit, evaluation)
        for name, expected in historical["contract"]["source_lineage"]["source"].items():
            path = (legacy_root/name).resolve()
            require(path.is_relative_to(legacy_root.resolve()) and path.is_file() and sha(path) == expected,
                    "HISTORICAL_RECORDED_SOURCE_LINEAGE_CHANGED")
        current_folder = Path(formal_private)/"anchors"
        current_receipt = read(current_folder/(f"outer{fold}_final.private.json"))
        current_path = current_folder/(f"outer{fold}_final.private.pkl")
        require(current_receipt["p2_lock_sha256"] == MASTER_SHA and
                current_receipt["artifact_sha256"] == sha(current_path) and
                current_receipt["teacher_seed"] == current_receipt["pipeline_crossfit_seed"] == 27011+100*fold+31,
                "CURRENT_R4_ANCHOR_BINDING_CHANGED")
        anchor = independent.passive_pickle(current_path)
        check_row_order(historical, anchor, fit, evaluation)
        old_teacher = historical["teachers"][31]
        old_risk = np.asarray(old_teacher["test_CP_SCALAR_risk"])
        current_risk = np.asarray(anchor.b0_exact_evaluation)
        require(np.array_equal(current_risk, anchor.evaluation_offset) and
                independent.array_sha(current_risk) == current_receipt["evaluation_offset_sha256"], "CURRENT_B0_OFFSET_IDENTITY_CHANGED")
        comparison = vector_comparison(old_risk, current_risk, old_teacher["beta"], anchor.state["full"]["beta"])
        old_metric = independent.independent_metrics(time[fit], event[fit], time[evaluation], event[evaluation], old_risk)
        current_metric = independent.independent_metrics(time[fit], event[fit], time[evaluation], event[evaluation], current_risk)
        results.append({"fold":fold,"seed":31,"test_patients":len(evaluation),"teacher_seed":27011+100*fold+31,
            "historical_COMPLETE_sha256":sha(historical_folder/"COMPLETE.json"),
            "historical_model_artifact_sha256":complete["artifact_sha256"],
            "historical_contract_sha256":qualification.canonical(historical["contract"]),
            "current_anchor_receipt_sha256":sha(current_folder/(f"outer{fold}_final.private.json")),
            "current_anchor_artifact_sha256":current_receipt["artifact_sha256"],
            "canonical_rows_and_source_contract_qualified":True, **comparison,
            "full_preprocessing_state_exact":fingerprint(historical["full_preprocessing"]) == fingerprint(anchor.state["full"]["preprocessing"]),
            "RNA512_selected_member_array_exact":np.array_equal(historical["full_preprocessing"]["molecular"]["selected"],
                                                               anchor.state["full"]["preprocessing"]["molecular"]["selected"]),
            "RNA512_selected_member_sha256":independent.array_sha(historical["full_preprocessing"]["molecular"]["selected"]),
            "historical_seed31_unified_metrics":old_metric,"current_B0_unified_metrics":current_metric})
        del historical, anchor
    exact = all(row["risk_values_exact"] and row["beta_exact"] for row in results)
    old_micro, current_micro = micro(results,"historical_seed31_unified_metrics"), micro(results,"current_B0_unified_metrics")
    report = {"schema":"semantic_b0b7_p2_historical_R4_seed31_audit_v1",
        "status":"EXACT_SEED31_RISK_AND_BETA_REPRODUCTION" if exact else "SOURCE_QUALIFIED_NUMERIC_DIFFERENCES_REQUIRE_RECIPE_DIAGNOSIS",
        "audit_source_sha256":sha(__file__),"master_lock_sha256":MASTER_SHA,
        "independent_scorer_source_sha256":sha(independent.__file__),"state_hash_helper_source_sha256":sha(ROOT/"scripts/semantic_b0b7_p2_dss_isolation_audit.py"),
        "historical_producer_source_sha256":HISTORICAL_PRODUCER_SHA,"formal_neural_fits_unchanged":formal_count,
        "folds":results,"historical_seed31_five_fold_micro":old_micro,"current_B0_five_fold_micro":current_micro,
        "three_seed_historical_average_was_not_reconstructed_or_optimized_to":True,
        "fold_aggregation":"sum_within_fold_components_single_seed31_no_crossfold_risk_ordering",
        "new_fits":0,"new_model_forwards":0,"optimizer_updates":0,"architectures_or_lambdas_reselected":False,
        "learned_beta_values_published":False,
        "DSS_labels_read":True,"original_PFI_OS_label_arrays_read":False,
        "own_historical_blob_decoded_on_CPU_after_checksum_qualification":True,
        "historical_blob_can_contain_preexisting_PFI_aggregate_metrics_not_used":True,
        "historical_fields_accessed":["contract","train","test","teachers.seed31.test_CP_SCALAR_risk","teachers.seed31.beta","full_preprocessing"],
        "historical_transitive_phase10_math_reference_SHA_not_explicitly_recorded_in_old_contract":True,
        "current_transitive_math_reference_bytes_checked_against_P2_lock":True,
        "training_offset_comparison_not_requested":"P2 training offset crossfits entire beta pipeline; original own-beta train vector is not the same object",
        "scores_from_current_unified_scorer_not_old_result_fields":True,
        "confidence_intervals_computed":False,"p_values_computed":False,
        "internal_historical_recipe_and_saved_risk_check_not_external_or_from_raw_H5_reproduction":True}
    qualification.public_write(output, report)
    return {"status":report["status"],"folds":len(results),"report_sha256":sha(output),"new_fits":0,"new_model_forwards":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("master-lock","expected-master-sha","assets","legacy-root","historical-root","formal-private","output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--formal-complete-authorized",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run_audit(args.master_lock,args.expected_master_sha,assets_path=args.assets,legacy_root=args.legacy_root,
        historical_root=args.historical_root,formal_private=args.formal_private,output=args.output,
        formal_complete_authorized=args.formal_complete_authorized)))


if __name__ == "__main__":
    main()

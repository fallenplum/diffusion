"""Prelocked held-out mutation qualification, separate from formal P2 fits.

``facts`` reads only existing public metadata. ``run`` is disabled until the
formal matrix has finished, the GPU is idle, and explicit completion authority
is supplied. It runs two fixed 50-update MAIN Phase-A units, never a DSS head.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASTER_SHA = "b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
ASSET_SHA = "107269a30053ae98f346992644f650dd89a6fc2a13344b792c1e4b3990eef2cf"
PROFILE_SHA = "becf224dece6ec249dd00966f1b174a5ec05be3aabb65f8da4a84217d4ce8914"
API_SHA = "aad1ed8a1c5a22602e0908e41e0f014bfd51c70eab0b26cfef55460eb2249ee5"
E0_SHA = "71879e676464a45294021e05d04d0c14039d26fbabe7bf293d9e6ecf9967fca7"
AUDIT_SCRIPT = "scripts/semantic_b0b7_p2_leakage_audit.py"
AUDIT_TEST = "tests/test_semantic_b0b7_p2_leakage_audit.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(a.dtype), "shape": list(a.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header+a.tobytes()).hexdigest()


def public_write(path, value):
    # Exactly the old generic field/text rules; actual private-ID scan remains
    # a separate server publication operation, never claimed here.
    sys.path.insert(0, str(ROOT))
    from scripts import bgb_en_publication as old
    old.validate_public_json(value)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    require(len(raw) <= 2*1024**2 and not any(p.search(raw.decode()) for p in old.PATTERNS.values()), "PUBLIC_AUDIT_RULE_FAILED")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)


def public_markdown(path, text):
    sys.path.insert(0,str(ROOT))
    from scripts import bgb_en_publication as old
    raw=text.encode("utf-8")
    require(len(raw)<=2*1024**2 and not any(p.search(text) for p in old.PATTERNS.values()),"PUBLIC_MARKDOWN_RULE_FAILED")
    with Path(path).open("xb") as f:f.write(raw)


def source_set(master):
    for relative, digest in master["frozen_sources"].items():
        path = (ROOT/relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, "FORMAL_SCIENTIFIC_SOURCE_CHANGED")
    return canonical(master["frozen_sources"])


def derangement(n, seed):
    require(n >= 2, "NONTRIVIAL_EVALUATION_SCOPE_REQUIRED")
    order = np.random.default_rng(seed).permutation(n)
    result = np.empty(n, np.int64)
    result[order] = np.roll(order, 1)
    require((result != np.arange(n)).all(), "DERANGEMENT_CONSTRUCTION_FAILED")
    return result


def perturb_raw(raw, evaluation, *, seed):
    """Joint held-out donor permutation; no fit row or image evidence changes."""
    rows = np.asarray(evaluation, np.int64)
    perm = derangement(len(rows), seed)
    changed = dict(raw)
    fields = ("rna", "rna_measured", "cna", "cna_measured", "platform")
    audit = {}
    for key in fields:
        values = np.array(raw[key], copy=True)
        values[rows] = np.asarray(raw[key])[rows[perm]]
        changed[key] = values
        audit[key+"_heldout_changed"] = array_sha(values[rows]) != array_sha(np.asarray(raw[key])[rows])
    changed["clinical_raw"] = {}
    clinical_changed = 0
    for key, value in raw["clinical_raw"].items():
        values = np.array(value, copy=True)
        values[rows] = np.asarray(value)[rows[perm]]
        changed["clinical_raw"][key] = values
        clinical_changed += int(not np.array_equal(values[rows], np.asarray(value)[rows]))
    changed["audit"] = {**raw.get("audit", {}), "counterfactual_evaluation_only_for_software_qualification": True}
    audit.update(clinical_fields_changed=clinical_changed, derangement_sha256=array_sha(perm),
                 selected_using_DSS_or_model_scores=False)
    return changed, perm, audit


def perturb_dss(outcome, evaluation, permutation):
    rows = np.asarray(evaluation, np.int64)
    changed = {key: np.array(value, copy=True) for key, value in outcome.items()}
    for key in ("t", "e"):
        changed[key][rows] = np.asarray(outcome[key])[rows[permutation]]
    return changed


def payload_fingerprints(data, indices):
    # Imported only for real audit execution or explicitly synthetic tests.
    from scripts.semantic_b0b7_p1r import tensor_hash
    rows = np.asarray(indices, np.int64)
    result = {}
    for section in ("image", "omics"):
        for key, value in data[section].items():
            if value is not None:
                result[section+"."+key] = tensor_hash(value[rows])
    for key in ("target", "target_mask"):
        result[key] = tensor_hash(data[key][rows])
    for key, value in data["metadata"].items():
        result["metadata."+key] = tensor_hash(np.asarray(value)[rows])
    for key, value in data["clinical_raw"].items():
        result["raw_clinical."+key] = tensor_hash(np.asarray(value)[rows])
    return result


def exact_tree(a, b):
    """Complete recursive byte equality, including tensor dtype and signed zero."""
    import torch
    stats = {"torch_tensor_leaves": 0, "numpy_array_leaves": 0, "scalar_leaves": 0}
    def compare(left, right):
        if torch.is_tensor(left) or torch.is_tensor(right):
            if not (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape):
                return False
            stats["torch_tensor_leaves"] += 1
            lb = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            rb = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return lb == rb
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            stats["numpy_array_leaves"] += 1
            return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.dtype == right.dtype and
                    left.shape == right.shape and left.tobytes() == right.tobytes())
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return list(left) == list(right) and all(compare(left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(compare(x, y) for x, y in zip(left, right))
        stats["scalar_leaves"] += 1
        if isinstance(left, float):
            return math.isfinite(left) and math.isfinite(right) and struct.pack("!d", left) == struct.pack("!d", right)
        if isinstance(left, np.generic):
            return left.dtype == right.dtype and left.tobytes() == right.tobytes()
        return left == right
    return {"equal": bool(compare(a, b)), **stats}


def compare_checkpoints(first, second):
    import torch
    a = torch.load(first, map_location="cpu", weights_only=False)
    b = torch.load(second, map_location="cpu", weights_only=False)
    components = {key: exact_tree(a[key], b[key]) for key in ("model", "optimizer", "cpu_rng", "cuda_rng", "expected_replay")}
    complete = exact_tree(a, b)
    require(all(x["equal"] for x in components.values()) and complete["equal"], "HELDOUT_MUTATION_CHANGED_DECODED_CHECKPOINT_STATE")
    archive_equal = sha(first) == sha(second)
    return {"entire_decoded_checkpoint_exact": True, "components": components,
        "checkpoint_archives_byte_identical": archive_equal,
        "archive_byte_difference_reason": None if archive_equal else
            "serialized_archive_bytes_differ_but_every_decoded_field_and_tensor_byte_is_identical_no_semantic_difference_detected",
        "archive_byte_identity_was_not_required": True, "new_model_forwards_for_comparison": 0}


def validate_qualification_lock(master_path, qualification_path, expected_qualification_sha):
    require(sha(master_path) == MASTER_SHA and sha(qualification_path) == expected_qualification_sha, "QUALIFICATION_OR_MASTER_LOCK_CHANGED")
    master, lock = read(master_path), read(qualification_path)
    require(lock["schema"] == "semantic_b0b7_p2_leakage_qualification_lock_v1" and lock["status"] == "LOCKED_BEFORE_TWO_AUDIT_FITS" and
            lock["P2_master_lock_sha256"] == MASTER_SHA and lock["formal_scientific_source_set_sha256"] == source_set(master),
            "QUALIFICATION_SOURCE_BINDING_INVALID")
    require(lock["outer"] == 1 and lock["fit_patients"] == 409 and lock["evaluation_patients"] == 103 and
            lock["seed"] == 31 and lock["family"] == "MAIN" and lock["fixed_epochs"] == 50 and
            lock["planned_audit_fits"] == 2 and lock["planned_audit_optimizer_updates"] == 100 and
            lock["phase_a_definition"] == master["phase_a_families"]["MAIN"] and
            lock["heldout_permutation_seed"] == 27131 and lock["DSS_values_never_passed_to_Phase_A_optimizer"] is True and
            lock["evaluation_metrics_used"] is False and lock["architecture_selection_performed"] is False,
            "UNLOCKED_QUALIFICATION_BUDGET_OR_MODEL")
    for relative, digest in lock["audit_sources"].items():
        require(relative in (AUDIT_SCRIPT, AUDIT_TEST) and sha(ROOT/relative) == digest, "AUDIT_SOURCE_CHANGED_AFTER_PRELOCK")
    require(set(lock["audit_sources"]) == {AUDIT_SCRIPT, AUDIT_TEST}, "AUDIT_SOURCE_CLOSURE_INCOMPLETE")
    return master, lock


def assert_formal_finished(formal_private, master):
    root = Path(formal_private)
    require(not (root/"ACTIVE_SCHEDULER_OWNER.private.json").exists(), "FORMAL_SCHEDULER_STILL_ACTIVE")
    count = 0
    for outer in range(1, 6):
        for inner in (*range(3), None):
            split = f"outer{outer}_"+("final" if inner is None else f"inner{inner}")
            for family in master["family_readouts"]:
                for seed in (31,47,83):
                    folder = root/"phase_a"/split/(family+"_seed"+str(seed))
                    receipt = read(folder/"cell.private.json")
                    require(receipt["status"] == "COMPLETE" and receipt["DSS_OS_PFI_accessed"] is False and
                            receipt["master_lock_sha256"] == canonical(master["phase_a_lock"]) and
                            receipt["frozen_sources"] == master["phase_a_lock"]["frozen_sources"] and
                            (folder/"RISK_BUNDLE_COMPLETE.private.json").is_file(), "FORMAL_MATRIX_NOT_FULLY_COMPLETE")
                    count += 1
    require(count == 480, "FORMAL_MATRIX_CARDINALITY_CHANGED")
    # No risk bundle content or outer metrics are read for this safety check.
    process = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    require(process.returncode == 0 and not [r for r in process.stdout.splitlines() if r.strip()], "GPU_NOT_VERIFIABLY_IDLE_FOR_AUDIT")
    return count


def run_qualification(master_path, qualification_path, expected_qualification_sha, *, assets_path, legacy_root,
                      frozen_images_path, formal_private, private_output, public_output, formal_complete_authorized=False):
    require(formal_complete_authorized, "EXPLICIT_POSTFORMAL_GPU_AUDIT_AUTHORIZATION_REQUIRED_BEFORE_INPUT_READ")
    master, lock = validate_qualification_lock(master_path, qualification_path, expected_qualification_sha)
    formal_count = assert_formal_finished(formal_private, master)
    require(not Path(private_output).exists() and not Path(public_output).exists(), "AUDIT_OUTPUTS_MUST_BE_NEW")
    require(sha(assets_path) == master["assets"]["assets_json_sha256"] and
            sha(frozen_images_path) == master["assets"]["frozen_images_sha256"], "AUDIT_ASSET_BYTES_CHANGED")
    import torch
    sys.path.insert(0, str(ROOT))
    from scripts.semantic_b0b7_prepare import legacy_data_module
    from scripts.semantic_b0b7_p1r import run_cell, validate_scope
    from scripts.semantic_b0b7_p2_matrix import training_input_binding
    from src.semantic_b0b7.p1r_data import build_partition_inputs, molecular_memberships, require_no_outcomes
    from src.semantic_b0b7.p2_data import read_dss_only
    require(sha(Path(legacy_root)/"src/bgb_en/data.py") == master["assets"]["legacy_data_source_sha256"], "LEGACY_READER_SOURCE_CHANGED")
    assets = read(assets_path)
    loader,_ = legacy_data_module(legacy_root)
    raw = loader.load_raw(assets, include_endpoints=False)
    memberships = molecular_memberships(raw, assets["gmt"])
    fit, evaluation = np.flatnonzero(raw["folds"] != 1), np.flatnonzero(raw["folds"] == 1)
    require(len(fit) == 409 and len(evaluation) == 103, "AUDIT_OUTER1_SCOPE_CHANGED")
    frozen = torch.load(frozen_images_path, map_location="cpu", weights_only=False)
    frozen["frozen_input_sha256"] = master["assets"]["frozen_images_sha256"]
    altered, permutation, perturbation = perturb_raw(raw, evaluation, seed=lock["heldout_permutation_seed"])
    require(perturbation["rna_heldout_changed"] and perturbation["cna_heldout_changed"] and perturbation["clinical_fields_changed"] > 0,
            "HELDOUT_COVARIATE_INTERVENTION_WAS_TRIVIAL")
    dss, endpoint_audit = read_dss_only(assets["endpoints"], raw["patient_ids"])
    altered_dss = perturb_dss(dss, evaluation, permutation)
    dss_changed = any(not np.array_equal(dss[k][evaluation],altered_dss[k][evaluation]) for k in ("t","e"))
    require(dss_changed and all(np.array_equal(dss[k][fit],altered_dss[k][fit]) for k in dss), "DSS_INTERVENTION_OR_TRAIN_ISOLATION_FAILED")
    try:
        require_no_outcomes({**altered, **altered_dss})
    except ValueError as exc:
        require(str(exc) == "ENDPOINT_BEARING_MAPPING_FORBIDDEN_IN_PHASE_A", "UNEXPECTED_ENDPOINT_REJECTION_RULE")
    else:
        raise ValueError("ENDPOINT_BEARING_RAW_WAS_NOT_REJECTED")
    out = Path(private_output);out.mkdir(parents=True)
    (out/"LOCK_BINDING.private.json").write_text(json.dumps({"master_lock_sha256":MASTER_SHA,
        "qualification_lock_sha256":expected_qualification_sha,"counterfactual_DSS_arrays_never_passed_to_Phase_A_optimizer":True}))
    train_fingerprints, eval_fingerprints, input_shas, receipts, post_rng = [], [], [], [], []
    for name, covariates in (("baseline",raw),("heldout_permuted",altered)):
        data,_ = build_partition_inputs(covariates,frozen,fit,evaluation,memberships,split_name="qualification_outer1_final")
        tr,ev = data["train_indices"].numpy(),data["valid_indices"].numpy()
        validate_scope(data,data["train_indices"],None,evaluate_valid=False)
        try:
            validate_scope({**data,"e":altered_dss["e"]},data["train_indices"],None,evaluate_valid=False)
        except ValueError as exc:
            require(str(exc)=="OUTCOME_BEARING_PHASE_A_PAYLOAD_FORBIDDEN","UNEXPECTED_PAYLOAD_ENDPOINT_REJECTION")
        else:
            raise ValueError("ENDPOINT_BEARING_PAYLOAD_WAS_NOT_REJECTED")
        train_fingerprints.append(payload_fingerprints(data,tr));eval_fingerprints.append(payload_fingerprints(data,ev))
        input_shas.append(training_input_binding(data,master["assets"]["frozen_images_sha256"]))
        if name=="heldout_permuted":
            require(train_fingerprints[0]==train_fingerprints[1] and input_shas[0]==input_shas[1], "HELDOUT_CHANGE_ALTERED_TRAINING_INPUTS")
        random.seed(lock["seed"]);np.random.seed(lock["seed"])
        try:
            receipt = run_cell(data,lock["phase_a_definition"],lock["seed"],out/name,master["phase_a_lock"],
                input_sha256=input_shas[-1],fixed_epoch=50,evaluate_valid=False,device="cuda",threads=2)
        except Exception:
            public_write(public_output,{"schema":"semantic_b0b7_p2_heldout_mutation_audit_v1","status":"FAILED_AUDIT_FIT_NO_INVARIANCE_CLAIM",
                "master_lock_sha256":MASTER_SHA,"qualification_lock_sha256":expected_qualification_sha,
                "attempted_audit_fits":len(receipts)+1,"completed_audit_fits":len(receipts),
                "confirmed_completed_updates":sum(r["optimizer_steps"] for r in receipts),
                "failed_fit_exact_update_count_known":False,"failed_fit_update_bounds":[0,50],
                "new_DSS_head_fits":0,"outer_risk_metrics_evaluated":0,"confidence_intervals_computed":False,"p_values_computed":False})
            raise
        receipts.append(receipt)
        post_rng.append({"cpu":torch.get_rng_state(),"cuda_all":torch.cuda.get_rng_state_all(),
                         "numpy_global":np.random.get_state(),"python_global":random.getstate()})
        del data;gc.collect();torch.cuda.empty_cache()
        source_set(master)
    require(train_fingerprints[0]==train_fingerprints[1] and input_shas[0]==input_shas[1] and
            eval_fingerprints[0]!=eval_fingerprints[1], "TRAIN_SCOPE_INVARIANCE_OR_EVAL_PERTURBATION_FAILED")
    for receipt in receipts:
        require(receipt["optimizer_steps"]==50 and receipt["selected_epoch"]==50 and not receipt["evaluate_valid"] and
                receipt["selected_validation"] is None and receipt["DSS_OS_PFI_accessed"] is False,
                "AUDIT_FIT_BUDGET_OR_OUTCOME_ACCESS_CHANGED")
    files=[out/name/receipt["source_state_file"] for name,receipt in zip(("baseline","heldout_permuted"),receipts)]
    try:
        checkpoint_comparison=compare_checkpoints(*files)
        rng_comparison=exact_tree(*post_rng)
        require(rng_comparison["equal"] and receipts[0]["history"]==receipts[1]["history"], "AUDIT_RNG_OR_TRAINING_HISTORY_DIFFERED")
    except Exception:
        public_write(public_output,{"schema":"semantic_b0b7_p2_heldout_mutation_audit_v1","status":"FAILED_EXACT_STATE_COMPARISON_NO_INVARIANCE_CLAIM",
            "master_lock_sha256":MASTER_SHA,"qualification_lock_sha256":expected_qualification_sha,
            "actual_audit_neural_fits":2,"actual_audit_optimizer_updates":sum(r["optimizer_steps"] for r in receipts),
            "new_DSS_head_fits":0,"outer_risk_metrics_evaluated":0,"confidence_intervals_computed":False,"p_values_computed":False})
        raise
    result={"schema":"semantic_b0b7_p2_heldout_mutation_audit_v1","status":"PASS_REAL_TWO_FIT_PHASE_A_HELDOUT_MUTATION_INVARIANCE",
        "master_lock_sha256":MASTER_SHA,"qualification_lock_sha256":expected_qualification_sha,
        "audit_source_sha256":sha(__file__),"formal_neural_fits_unchanged":formal_count,
        "actual_audit_neural_fits":2,"actual_audit_optimizer_updates":sum(r["optimizer_steps"] for r in receipts),
        "actual_audit_saved_checkpoint_replays":sum(len(r["states"]) for r in receipts),
        "actual_audit_elapsed_seconds_sum":sum(r["elapsed_seconds"] for r in receipts),
        "actual_audit_peak_cuda_allocated_bytes":max(r["peak_cuda_allocated_bytes"] for r in receipts),
        "training_input_sha256":input_shas[0],"all_training_tensor_bytes_identical":True,"heldout_covariates_changed":perturbation,
        "DSS_intervention_changed_real_heldout_values":True,"DSS_inputs_quarantined_not_accepted_by_Phase_A":True,
        "endpoint_bearing_raw_and_payload_rejected_by_original_entrypoints":True,
        "DSS_only_fields_read":endpoint_audit["arrays_read"],"PFI_OS_read":False,
        "checkpoint_sha256":[sha(p) for p in files],"checkpoint_comparison":checkpoint_comparison,
        "postfit_all_visible_Torch_and_global_Python_NumPy_RNG_exact":rng_comparison,
        "training_history_scientific_fields_exact":True,"elapsed_and_allocator_receipt_bytes_not_required_equal":True,
        "new_DSS_head_fits":0,"outer_risk_metrics_evaluated":0,"used_for_architecture_or_epoch_selection":False,
        "qualification_does_not_establish_DSS_gain_or_full_pipeline_live_label_invariance":True,
        "confidence_intervals_computed":False,"p_values_computed":False}
    public_write(public_output,result)
    return {"status":result["status"],"public_audit_sha256":sha(public_output),"actual_extra_neural_fits":2,"actual_extra_updates":100}


def build_fact_reports(master_path, qualification_path, *, asset_binding, profile, api_receipt, e0_receipt,
                       parent_binding, software_junit, output):
    require(sha(master_path)==MASTER_SHA and sha(asset_binding)==ASSET_SHA and sha(profile)==PROFILE_SHA and
            sha(api_receipt)==API_SHA and sha(e0_receipt)==E0_SHA,"FACT_SOURCE_BYTES_CHANGED")
    validate_qualification_lock(master_path,qualification_path,sha(qualification_path))
    master,assets,resource,api,e0,parent=map(read,(master_path,asset_binding,profile,api_receipt,e0_receipt,parent_binding))
    source_set(master)
    require(parent["P2_master_lock_sha256"]==MASTER_SHA and parent["P1R_commit_sha"]=="d2568ad4d57bb9c83366029cacee6baea6c3522d" and
            parent["P1R_frozen_candidates_sha256"]==master["P1R_frozen_candidates_sha256"] and not parent["P2_outer_scores_read_before_this_binding"],"PARENT_CONTENT_OR_TIMELINE_BINDING_CHANGED")
    require(resource["optimizer_updates"]==resource["new_neural_fits"]==0 and resource["endpoints_read"] is False and
            len(resource["profiles"])==resource["actual_saved_state_replays"]==6,"RESOURCE_PROFILE_SCOPE_CHANGED")
    for name,digest in resource["sources"].items():require(sha(ROOT/name)==digest,"PROFILE_SOURCE_CHANGED")
    require(api["synthetic_cases"]==100 and api["API_calls"]==200 and api["models_fitted"]==api["model_forwards"]==0 and
            not api["raw_cohort_read"] and not api["real_DSS_evaluation"],"OFFICIAL_API_QUALIFICATION_SCOPE_CHANGED")
    suites=ET.parse(software_junit).getroot().findall("testsuite")
    counts={k:sum(int(s.attrib[k]) for s in suites) for k in ("tests","failures","errors","skipped")}
    require(counts=={"tests":21,"failures":0,"errors":0,"skipped":0},"EXPECTED_21_SOFTWARE_TEST_RECEIPT_CHANGED")
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    reports={}
    reports["00_PARENT_AND_PROTOCOL_LOCK.json"]={"schema":"semantic_p2_protocol_fact_audit_v1","status":"PRE_OUTER_FACTUAL_PROTOCOL_AUDIT",
        "master_lock_sha256":MASTER_SHA,"scientific_sources_verified":len(master["frozen_sources"]),
        "HEAD_at_scientific_prelock":master["P1R_branch_HEAD_when_execution_prepared"],
        "content_bound_P1R_qualification_sha256":master["P1R_frozen_candidates_sha256"],
        "actual_P1R_publication_commit":parent["P1R_commit_sha"],"actual_P1R_manifest_sha256":parent["P1R_publication_manifest_sha256"],
        "parent_publication_binding_sha256":sha(parent_binding),"publication_after_inner_started_but_before_outer_scoring":parent["Git_publication_after_P2_inner_training_started"],
        "qualification_content_and_science_did_not_change_at_publication":True,"all_five_outer_choices_must_freeze_before_first_outer_score":True,
        "DSS_primary":True,"PFI_OS_evaluation_started":False,"confidence_intervals_computed":False,"p_values_computed":False,
        "current_document_is_not_a_formal_DSS_result_or_completion_claim":True}
    profile_rows=[];comparisons=[]
    for p in resource["profiles"]:
        require(p["actual_saved_reload_forward_replay"] and p["optimizer_steps"]==0 and
                all(v is not None and math.isfinite(v) and v>0 for v in p["gradient_norms"].values()),"REAL_PROFILE_GRADIENT_OR_REPLAY_INVALID")
        keys=("role","arm","train_patients","parameters","forward_profile_seconds","forward_backward_profile_seconds", "peak_cuda_allocated_bytes",
            "forward_supported_matrix_FLOPs","forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs","checkpoint_sha256","architecture")
        profile_rows.append({k:p[k] for k in keys})
    for role in ("inner","outer_final"):
        reference=next(p for p in profile_rows if p["role"]==role and p["arm"]=="B4_DUAL")
        matched=next(p for p in profile_rows if p["role"]==role and p["arm"]=="B2_MATCHED")
        errors={k:matched[k]/reference[k]-1 for k in ("parameters","forward_supported_matrix_FLOPs","forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs")}
        require(all(abs(v)<=.1 for v in errors.values()),"FINAL_MATCHED_PROFILE_EXCEEDS_LOCKED_MATRIX_COMPUTE_ENVELOPE")
        comparisons.append({"role":role,"matched_relative_errors":errors,"within10percent_supported_matrix_compute":True})
    reports["01_SAME_INPUT_CAPACITY_AND_FEATURE_AUDIT.json"]={"schema":"semantic_p2_data_capacity_fact_audit_v1","status":"REAL_ASSET_BINDING_AND_ZERO_UPDATE_RESOURCE_QUALIFIED",
        "asset_binding_sha256":ASSET_SHA,"asset_hashes":{k:v for k,v in assets.items() if k.endswith("sha256")},
        "patients":512,"DSS_events":65,"fold_patients":assets["fold_counts"],"fold_events":assets["fold_events"],"model_seeds":[31,47,83],
        "canonical_RNA_columns":20511,"canonical_CNA_columns":25101,"dimension_source":"pinned_canonical_loader_contract_and_existing_P0_real_asset_audit",
        "canonical_RNA_missing_columns_histogram":{"0":309,"3015":203},"canonical_CNA_missing_entries":260,
        "missingness_platform_groups_are_proxies_not_verified_sequencing_platform":True,
        "legacy_RNA_union_external":assets["legacy_union_external"],"legacy_RNA_union_canonical":assets["legacy_union_canonical"],
        "gene_selection_for_old_R4":"current_train_variance_within_fixed_legacy_union_then_512_no_DSS_selection",
        "Phase_A_programs":50,"RNA_CNA_gain_loss_separate_and_measured_masks_preserved":True,
        "common_risk_head":{"P":"8x64_flat_then_current_train_PCA8","G":"50x64_flat_then_current_train_PCA8","C":"64_to_current_train_PCA8","mechanism":2,"DSS_parameters":26},
        "B2_input":"512_raw_patch_budget_to8_nonsemantic_image_readers_by50_program_tokens",
        "B2_MATCHED":"one_head_width8_no_checkpoint_recomputation","B2_NATIVE":"four_heads_width64_independent_QKVO_with_recomputation_not_official_paper_native_reproduction",
        "real_profile_sha256":PROFILE_SHA,"real_zero_update_profiles":profile_rows,"same_scope_compute_comparisons":comparisons,
        "clinical_input_width_can_vary_with_train_fitted_levels_same_scope_matched_parameters_equal":True,
        "profile_timings_are_single_measurements_not_speed_benchmark":True,"supported_matrix_FLOPs_exclude_other_operators":True,
        "wall_time_and_peak_memory_match_not_claimed":True,"profile_fit_or_DSS_model_count":0,"formal_run_completion_not_inferred":True}
    reports["02_EFRON_UNO60_SCORER_AUDIT.json"]={"schema":"semantic_p2_survival_scorer_fact_audit_v1","status":"REAL_OFFICIAL_API_MATHEMATICAL_QUALIFICATION_NOT_DSS_EFFICACY",
        "source_API_receipt_sha256":API_SHA,"scorer_source_sha256":master["frozen_sources"]["src/semantic_b0b7/survival.py"],
        "actual_official_API_qualification":api,"same_time_event_censor_comparable":True,"same_time_events_not_compared":True,"risk_tie_tolerance":1e-8,
        "Uno60":"training_only_reverse_KM_right_continuous_after_tied_deaths_leave_then_censor_jump_event_anchor_time_strictly_less_than60",
        "Uno_support":"positive_G60_and_eligible_event_support_required_no_epsilon_clipping_no_test_KM",
        "aggregation":"sum_within_fold_numerators_denominators_per_seed_then_equal_three_seed_mean_not_crossfold_risk_pooling",
        "B0":"five_folds_single_deterministic_seed31_original_teacher_not_fifteen_fake_cells",
        "independent_185_real_outer_risk_verification":"NOT_RUN_AWAITS_FULL_FORMAL_COMPLETION_AND_AUTHORIZATION",
        "confidence_intervals_computed":False,"p_values_computed":False}
    reports["03_NO_DSS_PRETRAIN_AND_CLINICAL_CROSSFIT_AUDIT.json"]={"schema":"semantic_p2_leakage_boundary_fact_audit_v1","status":"MIXED_REAL_E0_AND_SYNTHETIC_SOFTWARE_EVIDENCE_LIVE_HELDOUT_AUDIT_PRELOCKED_NOT_RUN",
        "scientific_source_set_sha256":canonical(master["frozen_sources"]),"software_tests":{"receipt_sha256":sha(software_junit),**counts,
            "scope":"synthetic_independent_math_and_teacher_own_label_exclusion_tests_not_21_real_DSS_fits"},
        "clinical_background":"3_OOF_plus_full_Ridge_with_each_raw_clinical_encoder_refit_in_its_subtrain_one_current_frozen_encoder_coordinate_system",
        "background_source_sha256":master["frozen_sources"]["src/semantic_b0b7/data.py"],
        "r0":"whole_CP_SCALAR_pipeline_crossfit_including_supervised_beta_4pipelines_each13Cox1scalar_total52Cox4scalar",
        "real_E0_profile_receipt_sha256":E0_SHA,"real_E0":{"Cox_fits":e0["actual_Cox_fits"],"scalar_fits":e0["actual_scalar_pair_fits"],
            "scope":e0["scope"],"fit_patients":e0["fit_patients"],"evaluation_patients":e0["evaluation_patients"],
            "Cox_gradient_max":e0["Cox_gradient_max"],"scalar_gradient_max":e0["scalar_gradient_max"],"fit_seconds":e0["anchor_fit_seconds"],
            "outer_test_metrics_evaluated":e0["outer_test_metric_evaluations"],"standalone_lock_sha256":e0["standalone_lock_sha256"]},
        "E0_formal_artifact_reuse_not_claimed_different_top_level_lock":True,"E0_52plus4_profile_cost_separate_from_formal_teacher_cost":True,
        "no_DSS_Phase_A_boundary":"covariate_reader_include_endpoints_false_builder_and_runner_reject_endpoint_bearing_raw_or_payload",
        "live_heldout_mutation_qualification":"PRELOCKED_NOT_RUN","qualification_lock_sha256":sha(qualification_path),
        "planned_separate_software_qualification_cost":{"audit_neural_fits":2,"updates_per_fit":50,"total_updates":100,"formal_neural_fits":480},
        "DSS_perturbation_scope":"real_DSS_quarantined_plus_entrypoint_rejection_not_a_DSS_argument_to_Phase_A",
        "all_checkpoints_or_all_formal_cells_are_verified_by_this_document":False,"DSS_predictive_gain_inferred":False,
        "confidence_intervals_computed":False,"p_values_computed":False}
    markdowns={}
    markdowns["00_PARENT_AND_PROTOCOL_LOCK.md"]=("# P2 父提交与协议锁\n\n"
        f"科学主锁 SHA256：`{MASTER_SHA}`，20 份科学源码字节已核对。本报告没有读取任何新 outer 风险。\n\n"
        f"锁定时 HEAD 为 `{master['P1R_branch_HEAD_when_execution_prepared']}`；P1R 资格内容已以 `{master['P1R_frozen_candidates_sha256']}` 绑定。"
        f"实际发布父提交为 `{parent['P1R_commit_sha']}`，对应 manifest `{parent['P1R_publication_manifest_sha256']}`。\n\n"
        "P1R Git 发布发生在 P2 内层启动后、外折选择和评分前；原始主锁保留当时 HEAD，后续父提交绑定另存，没有为追认 Git 时间线改写科学锁或训练配置。"
        "五个 outer 的所有 epoch/λ 必须先冻结，再产生第一个 outer 成绩。\n\n"
        "优势：来源内容与后续发布提交均可逐字节追踪。限制：该队列已被历史研究多次开发，不能称独立外部验证。本文件不宣布正式矩阵完成或 DSS 增益；不计算 CI、bootstrap 或 p 值。\n")
    lines=["# P2 同输入、容量与特征审计","", "来源为既有真实资产绑定和 v3 的六次真实 GPU 零更新 profile；没有读取外折风险。", "",
        "512 名患者、65 个 DSS events；五折人数 103/103/102/102/102，事件数 13/14/12/13/13；seed 为 31、47、83。"
        "当前 canonical RNA/CNA 列数为 20511/25101；RNA 缺失列直方图为 309 人无缺失、203 人缺失 3015 列，不能把缺失当成零表达。", "",
        "共享下游读出为 P8＋G8＋C8＋机制2，共26个系数；前三组均仅当前训练分区拟合 PCA。G 保留50个程序的位置后展平，不采用低秩均值噪声白化。", "",
        "| scope | 模型 | 活跃参数 | forward 矩阵 FLOPs | forward+backward 矩阵 FLOPs | 峰值显存 bytes |", "|---|---|---:|---:|---:|---:|"]
    for p in profile_rows:lines.append(f"| {p['role']} | {p['arm']} | {p['parameters']} | {p['forward_supported_matrix_FLOPs']} | {p['forward_backward_including_checkpoint_recomputation_supported_matrix_FLOPs']} | {p['peak_cuda_allocated_bytes']} |")
    lines += ["", "同一 scope 下 MATCHED 与 B4 参数完全相同，forward 和含重算的 forward+backward 支持矩阵 FLOPs 差异均在预锁10%范围内。"
        "MATCHED 为1头×8维、不做 activation checkpoint；NATIVE 为4头×64维、独立QKVO且做重算。两者都是8个无语义图像读取位置×50程序的普通注意力适配，不是官方论文原生复现。", "",
        "优势：依据实际执行、非零梯度与保存状态回放校验预算。风险：这些 FLOPs 不覆盖所有算子；单次带 profiler 的时间不构成稳定速度基准，显存和墙钟时间也没有宣称匹配。不同训练 scope 临床编码宽度可变，因此参数数目不能跨 scope 直接当作容量不一致。", ""]
    markdowns["01_SAME_INPUT_CAPACITY_AND_FEATURE_AUDIT.md"]="\n".join(lines)
    markdowns["02_EFRON_UNO60_SCORER_AUDIT.md"]=("# P2 Efron、Harrell 与 Uno60 资格\n\n"
        "已有真实服务器隔离环境执行 scikit-survival 0.23.1 的100个合成案例、200次真实 API 调用。Harrell 最大绝对差为0，Uno60为3.3306690738754696e-16；包括风险近似 ties 和同刻 event/censor。"
        f"v2 收据为 `{API_SHA}`，仅修正了日期/依赖表述，没有把同一批调用重复计数。\n\n"
        "风险越高代表更早事件风险；同刻 event/censor 可比较，同刻两个 event 不比较，风险差≤1e-8记半分。Uno 使用 fit-train reverse-KM；同刻观察事件先离开，再发生删失跳变，取该时刻右连续 G(t)，仅事件时间严格小于60月的 anchor 具有权重。"
        "G60/事件支持不足按锁设 NA，不用 epsilon 裁剪，不在 test 拟合 KM。\n\n"
        "完整训练风险集的 Efron＋ridge 使用独立 NumPy 数学参考与有限差分软件检查。汇总先在每 seed 内加总五折 numerator/denominator，再等权平均三个 seed，不能跨折比较风险或把15 cells叫作独立 seed 均值。B0只有五个确定性重拟合。\n\n"
        "优势：已用真实官方 API 检验评分公式。限制：API 输入是合成案例，尚不是185个正式外测风险的独立验证；该验收等待正式训练与报告全部完成后的授权。隔离包只验证所用 metrics API，未声称完整估计器依赖生态已安装齐全；不计算 CI 或 p 值。\n")
    markdowns["03_NO_DSS_PRETRAIN_AND_CLINICAL_CROSSFIT_AUDIT.md"]=("# P2 无DSS预训练、背景交叉拟合与教师风险\n\n"
        "已核对冻结入口：Phase-A 以 include_endpoints=False 读协变量，raw builder 与 run_cell 均拒绝带结局的输入；临床背景为同一冻结编码坐标内3个OOF＋1个完整训练 Ridge，每个子拟合从原始六字段重做临床编码。不能把不同编码器的 latent 直接拼成 OOF。\n\n"
        "r0 对整个 CP_SCALAR 管线再次交叉拟合，包含受监督的标量 β：每个分区4套管线，每套13个 Cox＋1个 scalar，共52＋4，避免患者自身标签参与其训练侧 offset 的 β。"
        f"真实 E0 在 outer1_inner0 的272/137人范围已完成52＋4；Cox最大梯度 {e0['Cox_gradient_max']:.12g}，scalar最大梯度 {e0['scalar_gradient_max']:.12g}，anchor拟合 {e0['anchor_fit_seconds']:.6f} 秒，0次 outer 指标。\n\n"
        "E0 是单独锁的真实 profile，顶层 source/lock 与正式矩阵不完全相同，因此不冒称已免费复用为正式 teacher；其52＋4成本需单列。既有21个软件测试全部通过，但其中的标签扰动与教师隔离是合成反例，不是21套真实 DSS 实验。\n\n"
        "真正 held-out 置换资格已另行预锁：outer1 fit409/eval103、MAIN dual、seed31、50步×2次。从 raw 层仅置换 eval RNA/CNA/clinical，要求训练 tensors 与 inputSHA 不变。真实 DSS 只在隔离对象中置换，并测试带结局输入会被原入口拒绝；不向 Phase-A 添加 DSS 参数。"
        "两次 checkpoint 要求完整解码的参数、Adam、CPU/CUDA RNG、所有其他字段逐字节精确；容器原文件 SHA 另报，不强迫计时或分配器 receipt 字节相同。\n\n"
        "当前状态：该2-fit资格尚未运行，必须等480个正式拟合结束、GPU空闲、收到授权后执行。额外100 updates 独立计入软件资格，不能混进 primary metrics、架构搜索或临时选择。"
        "优势：能直接发现当前 Phase-A 对 held-out 数据的实现依赖。限制：即使通过，也不能代替整个 DSS 训练管线的真实标签扰动测试或证明预后增益；不计算 CI/p 值。\n")
    outputs=[]
    for name,value in reports.items():
        value["fact_builder_source_sha256"]=sha(__file__)
        value["new_training_by_fact_builder"]=0
        value["new_outer_risk_reads_by_fact_builder"]=0
        public_write(out/name,value);outputs.append({"file":name,"sha256":sha(out/name)})
    for name,text in markdowns.items():
        public_markdown(out/name,text);outputs.append({"file":name,"sha256":sha(out/name)})
    return {"status":"FOUR_PRE_OUTER_FACTUAL_AUDITS_JSON_AND_MD_NO_TRAINING_OR_OUTER_RISK_READ","files":outputs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("facts","run"))
    for key in ("master-lock","qualification-lock","expected-qualification-sha256","assets","legacy-root","frozen-images","formal-private",
                "private-output","public-output","asset-binding","profile","api-receipt","e0-receipt","parent-binding","software-junit","output"):
        p.add_argument("--"+key)
    p.add_argument("--formal-complete-authorized",action="store_true");a=p.parse_args()
    if a.command=="facts":
        result=build_fact_reports(a.master_lock,a.qualification_lock,asset_binding=a.asset_binding,profile=a.profile,
            api_receipt=a.api_receipt,e0_receipt=a.e0_receipt,parent_binding=a.parent_binding,software_junit=a.software_junit,output=a.output)
    else:
        result=run_qualification(a.master_lock,a.qualification_lock,a.expected_qualification_sha256,
            assets_path=a.assets,legacy_root=a.legacy_root,frozen_images_path=a.frozen_images,formal_private=a.formal_private,
            private_output=a.private_output,public_output=a.public_output,formal_complete_authorized=a.formal_complete_authorized)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))


if __name__=="__main__":main()

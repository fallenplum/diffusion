"""Prelocked, post-formal held-out DSS isolation audit of the frozen P2 pipeline.

Two new CPU teacher/head pipelines are software qualification only. No neural
weights, epochs, lambda choices, primary scores, or scientific source are changed.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import semantic_b0b7_p2_leakage_audit as qualification

MASTER_SHA = "b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
SCRIPT = "scripts/semantic_b0b7_p2_dss_isolation_audit.py"
TEST = "tests/test_semantic_b0b7_p2_dss_isolation_audit.py"
COST = {"teacher_Cox_fits": 104, "downstream_Cox_fits": 2, "all_Cox_fits": 106,
        "scalar_fits": 8, "teacher_preprocessing_fits": 32, "risk_PCA_fits": 6,
        "background_multioutput_Ridge_fits": 8, "new_neural_fits": 0, "neural_optimizer_updates": 0}
require, sha, read = qualification.require, qualification.sha, qualification.read


def fingerprint(value):
    """Exact numeric/structural state hashing, including dtype and signed zero."""
    h = hashlib.sha256()
    def emit(tag, payload):
        h.update(tag); h.update(struct.pack("!Q", len(payload))); h.update(payload)
    def visit(item):
        if isinstance(item, np.ndarray):
            require(not item.dtype.hasobject, "OBJECT_ARRAY_NOT_A_LEARNED_NUMERIC_STATE")
            emit(b"ndarray", str(item.dtype).encode()+json.dumps(list(item.shape)).encode()+np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            emit(b"numpy_scalar", str(item.dtype).encode()+item.tobytes())
        elif isinstance(item, dict):
            require(all(isinstance(key, str) for key in item), "STATE_DICTIONARY_KEYS_MUST_BE_STRINGS")
            emit(b"dict", str(len(item)).encode())
            for key in sorted(item):
                emit(b"key", key.encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            emit(type(item).__name__.encode(), str(len(item)).encode())
            for sub in item: visit(sub)
        elif is_dataclass(item):
            emit(b"dataclass", (type(item).__module__+"."+type(item).__name__).encode())
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif type(item).__module__.startswith("sklearn.") and hasattr(item, "__dict__"):
            emit(b"sklearn", (type(item).__module__+"."+type(item).__name__).encode()); visit(vars(item))
        elif item is None:
            emit(b"none", b"")
        elif isinstance(item, bool):
            emit(b"bool", bytes((item,)))
        elif isinstance(item, int):
            emit(b"int", str(item).encode())
        elif isinstance(item, float):
            emit(b"float", struct.pack("!d", item))
        elif isinstance(item, str):
            emit(b"str", item.encode())
        else:
            raise ValueError("UNSUPPORTED_LEARNED_STATE_TYPE")
    visit(value)
    return h.hexdigest()


def permute_heldout_dss(raw, fit, evaluation, seed):
    fit, evaluation = np.asarray(fit, np.int64), np.asarray(evaluation, np.int64)
    require(not np.intersect1d(fit, evaluation).size, "QUALIFICATION_TRAIN_EVALUATION_OVERLAP")
    permutation = qualification.derangement(len(evaluation), seed)
    changed = dict(raw)
    for key in ("t", "e"):
        changed[key] = np.array(raw[key], copy=True)
        changed[key][evaluation] = np.asarray(raw[key])[evaluation[permutation]]
        require(fingerprint(changed[key][fit]) == fingerprint(np.asarray(raw[key])[fit]), "TRAINING_DSS_WAS_CHANGED")
    require(any(not np.array_equal(changed[k][evaluation], np.asarray(raw[k])[evaluation]) for k in ("t", "e")),
            "HELDOUT_DSS_INTERVENTION_WAS_TRIVIAL")
    require(fingerprint({k: v for k, v in raw.items() if k not in ("t", "e")}) ==
            fingerprint({k: v for k, v in changed.items() if k not in ("t", "e")}), "NONLABEL_INPUT_CHANGED")
    return changed, {"permutation_sha256": fingerprint(permutation),
                     "training_label_values_changed": 0,
                     "evaluation_time_values_changed": int((changed["t"][evaluation] != np.asarray(raw["t"])[evaluation]).sum()),
                     "evaluation_event_values_changed": int((changed["e"][evaluation] != np.asarray(raw["e"])[evaluation]).sum()),
                     "evaluation_pairs_jointly_permuted": True, "covariates_changed": False}


def compare_components(first, second):
    require(set(first) == set(second), "AUDIT_COMPONENT_SET_CHANGED")
    result = {key: {"baseline_sha256": fingerprint(first[key]), "mutated_sha256": fingerprint(second[key])}
              for key in sorted(first)}
    for key, row in result.items():
        row["exact"] = row["baseline_sha256"] == row["mutated_sha256"]
        require(row["exact"], "HELDOUT_DSS_CHANGED_LEARNED_PIPELINE:"+key)
    return result


def validate_lock(master_path, qualification_path, expected_qualification_sha):
    require(sha(master_path) == MASTER_SHA and sha(qualification_path) == expected_qualification_sha,
            "ISOLATION_OR_MASTER_LOCK_BYTES_CHANGED")
    master, lock = read(master_path), read(qualification_path)
    require(lock["schema"] == "semantic_b0b7_p2_dss_pipeline_isolation_lock_v1" and
            lock["status"] == "LOCKED_BEFORE_ADDITIONAL_CPU_QUALIFICATION_FITS" and
            lock["P2_master_lock_sha256"] == MASTER_SHA and
            lock["formal_scientific_source_set_sha256"] == qualification.source_set(master), "ISOLATION_SCIENCE_BINDING_CHANGED")
    require(lock["outer"] == 1 and lock["family"] == "MAIN" and lock["arm"] == "B5" and lock["seed"] == 31 and
            lock["fit_patients"] == 409 and lock["evaluation_patients"] == 103 and
            lock["heldout_pair_permutation_seed"] == 27132 and lock["planned_cost"] == COST and
            lock["frozen_lambda_source"] == "existing_ALL_OUTER_SELECTIONS_outer1_B5_seed31_no_reselection" and
            lock["performance_evidence_or_new_primary_cell"] is False and
            lock["does_not_replace_original_two_by_50_Phase_A_F1"] is True, "ISOLATION_BUDGET_OR_SELECTION_CHANGED")
    require(set(lock["audit_sources"]) == {SCRIPT, TEST}, "ISOLATION_SOURCE_CLOSURE_INCOMPLETE")
    for name, digest in lock["audit_sources"].items():
        require(sha(ROOT/name) == digest, "ISOLATION_SOURCE_CHANGED_AFTER_PRELOCK")
    require(sha(qualification.__file__) == lock["existing_F1_helper_source_sha256"], "EXISTING_F1_HELPER_CHANGED")
    return master, lock


def write_private(path, value):
    with Path(path).open("xb") as stream:
        pickle.dump(value, stream, protocol=5)


def run_qualification(master_path, qualification_path, expected_qualification_sha, *, assets_path, legacy_root,
                      frozen_images_path, formal_private, private_output, public_output,
                      formal_complete_authorized=False, feature_device="cuda", threads=1):
    # No source/input/endpoint file is read before this authority boundary.
    require(formal_complete_authorized, "EXPLICIT_POSTFORMAL_DSS_ISOLATION_AUTHORIZATION_REQUIRED_BEFORE_INPUT_READ")
    master, lock = validate_lock(master_path, qualification_path, expected_qualification_sha)
    formal_count = qualification.assert_formal_finished(formal_private, master)
    require(not Path(private_output).exists() and not Path(public_output).exists(), "ISOLATION_OUTPUTS_MUST_BE_NEW")
    require(feature_device in ("cpu", "cuda") and threads == 1, "LOCKED_FEATURE_DEVICE_OR_CPU_THREADS_CHANGED")
    import torch
    from threadpoolctl import threadpool_limits
    from scripts import semantic_b0b7_p2_matrix as matrix
    from scripts.semantic_b0b7_p1r import load_checkpoint_model, extract_frozen_features
    from src.semantic_b0b7.p1r_models import frozen_readout_operator
    from src.semantic_b0b7.p2_anchor import fit_r4_anchor
    from src.semantic_b0b7.p2_survival import frozen_feature_views, fit_risk_cell

    torch.set_num_threads(threads)
    args = SimpleNamespace(lock=str(master_path), expected_lock_sha256=MASTER_SHA, assets=str(assets_path),
        legacy_root=str(legacy_root), frozen_images=str(frozen_images_path), private=str(formal_private),
        split="outer1_final", family="MAIN", seed=31, threads=threads)
    matrix.verify_lock(args)
    selection = matrix.frozen_outer_selection(args)
    selection_path = Path(formal_private)/"ALL_OUTER_SELECTIONS_FROZEN.private.json"
    barrier = read(Path(formal_private)/"ALL_OUTER_SELECTION_BARRIER_BINDING.private.json")
    require(selection["outer_results_read"] is False and barrier["p2_lock_sha256"] == MASTER_SHA and
            barrier["selection_file_sha256"] == sha(selection_path) and
            barrier["status"] == "VERIFIED_ALL_FIVE_OUTER_SELECTIONS_BEFORE_FINAL_STAGE", "GLOBAL_SELECTION_BARRIER_CHANGED")
    risk_choice = selection["entries"]["1"]["risk"]["B5"]["31"]
    require(risk_choice["uses_outer_test"] is False and risk_choice["ridge_lambda"] in (.1, 1., 10.), "FROZEN_LAMBDA_SCOPE_CHANGED")
    lam = risk_choice["ridge_lambda"]
    source = Path(formal_private)/"phase_a/outer1_final/MAIN_seed31"
    phase, binding = read(source/"cell.private.json"), read(source/"MATRIX_BINDING.private.json")
    require(binding["p2_lock_sha256"] == MASTER_SHA and binding["cell_receipt_sha256"] == sha(source/"cell.private.json") and
            phase["evaluate_valid"] is False and phase["DSS_OS_PFI_accessed"] is False and
            phase["selected_epoch"] == selection["entries"]["1"]["pretraining"]["MAIN"]["31"]["epoch"], "FINAL_CHECKPOINT_SCOPE_CHANGED")
    checkpoint = source/phase["source_state_file"]
    checkpoint_sha = sha(checkpoint)
    require(checkpoint_sha == phase["selected_checkpoint_sha256"], "FINAL_CHECKPOINT_BYTES_CHANGED")
    started = time.monotonic()
    data, preprocessing, split, input_sha = matrix.build_inputs(args, master)
    require(input_sha == phase["input_sha256"] == binding["training_only_input_sha256"], "FROZEN_INPUT_SCOPE_CHANGED")
    model, _ = load_checkpoint_model(checkpoint, device=feature_device, freeze=True, expected_input_sha256=input_sha)
    # One immutable extraction is deliberately shared; the audit retrains only
    # CPU nuisance teachers/backgrounds/PCA and the locked convex DSS head.
    outputs = extract_frozen_features(model, data, torch.arange(len(data["global_rows"])), batch_size=32)
    model.cpu()
    if feature_device == "cuda": torch.cuda.empty_cache()
    def operator(arrays, mu_visual, mu_gene, arm):
        tensors = {key: torch.as_tensor(value) for key, value in arrays.items()}
        with torch.no_grad():
            result = frozen_readout_operator(model, tensors,
                None if mu_visual is None else torch.as_tensor(mu_visual, dtype=torch.float32),
                None if mu_gene is None else torch.as_tensor(mu_gene, dtype=torch.float32), arm=arm)
        return result["mechanism"].cpu().numpy()
    raw, membership, _ = matrix.raw_inputs(args, endpoints=True)
    fit, evaluation = np.asarray(split["fit"], np.int64), np.asarray(split["evaluation"], np.int64)
    require(len(fit) == 409 and len(evaluation) == 103, "FIXED_OUTER1_ROW_COUNTS_CHANGED")
    rows = data["global_rows"].numpy(); tr, ev = data["train_indices"].numpy(), data["valid_indices"].numpy()
    require(np.array_equal(rows[tr], fit) and np.array_equal(rows[ev], evaluation), "NEURAL_AND_TEACHER_CANONICAL_ROWS_CHANGED")
    changed, intervention = permute_heldout_dss(raw, fit, evaluation, lock["heldout_pair_permutation_seed"])
    out = Path(private_output); out.mkdir(parents=True)
    recipe, runs, summaries = master["B0_R4_EXACT"], [], []
    with threadpool_limits(limits=1):
        for name, payload in (("baseline", raw), ("heldout_DSS_permutation", changed)):
            at = time.monotonic()
            anchor = fit_r4_anchor(payload, fit, evaluation, membership[1],
                expected_union_sha256=recipe["legacy_eight_program_union_sha256"], legacy_root=legacy_root,
                expected_legacy_source_hashes=recipe["legacy_source_sha256"], teacher_seed=split["teacher_seed"],
                pipeline_crossfit_seed=split["teacher_seed"], downstream_crossfit=True)
            views = frozen_feature_views(outputs, data["clinical_raw"], tr, ev, arm="B5", operator=operator)
            cell = fit_risk_cell(views["train"], views["evaluation"], payload["t"][fit], payload["e"][fit],
                payload["t"][evaluation], payload["e"][evaluation], anchor.training_offset, anchor.evaluation_offset,
                ridge_lambda=lam, role="outer_final", bindings={"qualification_only": True, "lock_sha256":MASTER_SHA},
                selection_record=risk_choice)
            require(anchor.receipt["Cox_optimizer_fits"] == 52 and anchor.receipt["scalar_pair_optimizer_fits"] == 4 and
                    anchor.receipt["preprocessing_fits"] == 16 and views["background"]["audit"]["background_models"] == 4 and
                    cell.receipt["DSS_coefficients"] == 26, "ACTUAL_QUALIFICATION_COST_CHANGED")
            components = {"teacher_entire_learned_state": anchor.state, "teacher_receipt": anchor.receipt,
                "crossfit_training_offset": anchor.training_offset, "full_evaluation_offset": anchor.evaluation_offset,
                "background_state": views["background"]["state"], "risk_PCA_all_statistics": cell.design,
                "training_feature_views": views["train"], "evaluation_feature_views": views["evaluation"],
                "risk_coefficients_and_solver_state": cell.model,
                "training_predictions": cell.training_risk, "evaluation_predictions": cell.evaluation_risk,
                "training_metrics": cell.receipt["train_metrics"]}
            write_private(out/(name+".private.pkl"), {"anchor": anchor, "risk": cell, "background": views["background"]})
            runs.append(components)
            summaries.append({"run":name,"elapsed_seconds":time.monotonic()-at,
                "artifact_sha256":sha(out/(name+".private.pkl")),
                "teacher_Cox_gradient_max":max(float(m["optimizer"]["gradient_max"])
                    for state in [anchor.state["full"]]+[s["state"] for s in anchor.state["crossfit"]]
                    for teacher in [state["full_models"]]+[t["models"] for t in state["teachers"]] for m in teacher.values()),
                "risk_gradient_max":cell.model.receipt["gradient_max"]})
            if name == "baseline": original_metrics = cell.receipt["evaluation_metrics"]
            else: altered_metrics = cell.receipt["evaluation_metrics"]
    exact = compare_components(*runs)
    require(sha(checkpoint) == checkpoint_sha, "FROZEN_ENCODER_CHECKPOINT_WAS_CHANGED")
    report = {"schema":"semantic_b0b7_p2_real_DSS_pipeline_isolation_v1",
        "status":"PASS_REAL_HELDOUT_DSS_INVARIANT_LEARNED_PIPELINE",
        "master_lock_sha256":MASTER_SHA,"qualification_lock_sha256":expected_qualification_sha,
        "audit_source_sha256":sha(__file__),"formal_neural_fits_unchanged":formal_count,
        "phase_a_checkpoint_sha256":checkpoint_sha,"phase_a_receipt_sha256":sha(source/"cell.private.json"),
        "all_outer_selection_sha256":sha(selection_path),"fixed_family":"MAIN","fixed_arm":"B5","seed":31,
        "ridge_lambda":float(lam),"lambda_reselected":False,"epoch_reselected":False,
        "fit_patients":409,"evaluation_patients":103,"intervention":intervention,"exact_components":exact,
        "shared_frozen_preprocessing_sha256":fingerprint(preprocessing),
        "actual_additional_compute":COST,"frozen_encoder_feature_extractions":1,
        "frozen_encoder_branch_forward_batches_per_modality":int(math.ceil(len(rows)/32)),
        "frozen_feature_device":feature_device,"frozen_encoder_optimizer_updates":0,
        "heldout_metrics_equal":original_metrics == altered_metrics,
        "heldout_metric_values_published":False,"heldout_scoring_is_software_diagnostic_not_new_performance_evidence":True,
        "original_two_by_50_Phase_A_F1_replaced":False,"primary_predictions_or_selections_changed":False,
        "PFI_OS_arrays_read":False,"confidence_intervals_computed":False,"p_values_computed":False,
        "elapsed_seconds":time.monotonic()-started,"runs":summaries,
        "qualification_limit":"One fixed outer1 checkpoint and lambda; not a second full data-to-prediction replication or external validation"}
    qualification.public_write(public_output, report)
    (out/"COMPLETE.private.json").write_text(json.dumps({"status":report["status"],"public_report_sha256":sha(public_output),
        "qualification_lock_sha256":expected_qualification_sha,"actual_additional_compute":COST},indent=2)+"\n")
    return {"status":report["status"],"public_report_sha256":sha(public_output),"actual_additional_compute":COST}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("master-lock", "qualification-lock", "expected-qualification-sha", "assets", "legacy-root",
                 "frozen-images", "formal-private", "private-output", "public-output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--formal-complete-authorized", action="store_true")
    parser.add_argument("--feature-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_qualification(args.master_lock,args.qualification_lock,args.expected_qualification_sha,
        assets_path=args.assets,legacy_root=args.legacy_root,frozen_images_path=args.frozen_images,
        formal_private=args.formal_private,private_output=args.private_output,public_output=args.public_output,
        formal_complete_authorized=args.formal_complete_authorized,feature_device=args.feature_device)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

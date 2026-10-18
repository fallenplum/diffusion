"""Audit the six completed private feature caches; never run a forward/fit.

Only anonymous counts, numerical invariant maxima and artifact SHA receipts are
published. No prepared label payload, raw feature source, encoder inference,
patient-level feature or risk array is loaded into the public report.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.phase26_multiquery import data as old_data, model as old_model, runtime as old_runtime, train as old_train
from src.phase27_readout import features, math as readout_math, runtime


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def array(value, shape, name, *, float64=False):
    result = np.asarray(value)
    require(result.shape == shape and np.isfinite(result).all(), name+" shape/finite mismatch")
    if float64:
        require(result.dtype == np.float64, name+" is not stored float64")
    return result


def exact(left, right, name):
    require(np.array_equal(np.asarray(left), np.asarray(right)), name+" differs from immutable reference")


@contextlib.contextmanager
def guards():
    calls = {}
    with contextlib.ExitStack() as stack:
        for obj, name, label in (
            (old_data.Data, "__init__", "Phase26.Data"),
            (old_data.old.Data, "__init__", "Phase25.Data"),
            (old_data.Data, "fit_labels", "Phase26.fit_labels"),
            (old_data.old.Data, "fit_labels", "Phase25.fit_labels"),
            (old_data.old.parent, "endpoints", "endpoints"),
            (old_data.old.parent, "splits", "splits"),
            (old_train, "train_arm", "neural_training"),
            (old_train, "_forward", "legacy_forward"),
            (old_model.MultiQuery, "forward", "encoder_forward"),
            (features, "extract_cell", "feature_extraction"),
            (old_data, "prepare", "prepare"),
            (old_data, "read_prepared", "prepared_label_payload_load"),
            (old_data.old, "read_prepared", "legacy_prepared_label_payload_load"),
            (old_data, "fit_ridge", "legacy_ridge_fit"),
            (old_data.old.PCA, "fit", "PCA_fit"),
            (readout_math, "fit_projection", "new_PCA_fit"),
            (readout_math, "fit_offset_ridge", "new_Cox_fit"),
            (torch.optim.Optimizer, "__init__", "optimizer_construction"),
        ):
            calls[label] = 0
            def blocked(*args, _label=label, **kwargs):
                calls[_label] += 1
                raise RuntimeError("saved-feature audit forbids "+_label)
            stack.enter_context(patch.object(obj, name, blocked))
        yield calls


def inherited_evidence():
    report = old_runtime.REPORT/"MAIN_REPLAY_AUDIT.json"
    main = json.loads(report.read_text(encoding="utf-8-sig"))
    main_script = ROOT/"scripts/replay_phase26_main.py"
    require(main["status"] == "PASS_NO_FIT_MAIN_REPLAY" and main["source"] == runtime.LEGACY_SOURCE
            and main["replay_script_sha256"] == runtime.sha(main_script)
            and main["prepared_cells"] == 3 and main["risk_vectors_replayed"] == 48
            and main["maximum_absolute_risk_error"] <= 1e-10
            and main["maximum_feature_replay_error"] <= 1e-10
            and main["endpoint_API_reads"] == main["new_predictor_fits"] == main["new_PCA_fits"] == 0,
            "independent legacy main evidence mismatch")
    neural_path = old_runtime.REPORT/"REPLAY_AUDIT.json"
    neural = json.loads(neural_path.read_text(encoding="utf-8-sig"))
    require(neural["status"] == "PASS" and neural["stage"] == "prototype"
            and neural["source"] == runtime.LEGACY_SOURCE
            and neural["replay_script_sha256"] == runtime.sha(ROOT/"scripts/replay_phase26.py")
            and neural["maximum_absolute_risk_error"] <= 1e-7,
            "independent legacy neural evidence mismatch")
    prepared = {int(key): value for key, value in main["prepared_sha256_by_fold"].items()}
    require(set(prepared) == {1, 2, 3}, "legacy prepared coverage mismatch")
    for fold, digest in prepared.items():
        # Hash bytes only: do not deserialize the prepared fit labels.
        require(runtime.sha(old_data.prepared_path(fold)) == digest, "legacy prepared bytes changed")
    return prepared, {"main_report_sha256": runtime.sha(report),
                      "main_script_sha256": runtime.sha(main_script),
                      "neural_report_sha256": runtime.sha(neural_path),
                      "neural_script_sha256": runtime.sha(ROOT/"scripts/replay_phase26.py"),
                      "raw_main_or_encoder_replay_repeated_here": False}


def preflight(identity, prepared):
    entries = []
    config = runtime.cfg()
    require(config["seeds"] == [31, 47] and config["folds"] == [1, 2, 3], "six-cell lock changed")
    for seed in config["seeds"]:
        for fold in config["folds"]:
            path = runtime.feature_path(seed, fold)
            receipt_path = path.with_suffix(".json")
            require(path.is_file() and receipt_path.is_file(), "all six complete feature caches required")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            for key, value in (("status", "COMPLETE"), ("stage", "prototype"), ("source", identity),
                               ("seed", seed), ("fold", fold), ("new_neural_steps", 0), ("new_predictor_fits", 0)):
                require(receipt.get(key) == value, "feature receipt differs: "+key)
            require(receipt["sha256"] == runtime.sha(path), "feature cache SHA mismatch")
            old_path = old_runtime.cell_dir(seed, fold)
            old_runtime.verify_done(old_path, runtime.LEGACY_SOURCE, seed, fold)
            require(receipt["legacy_checkpoint_sha256"] == runtime.sha(old_path/"checkpoint.pt")
                    and receipt["prepared_sha256"] == prepared[fold], "feature lineage receipt differs")
            entries.append((seed, fold, path, receipt_path, receipt, old_path))
    require(len(entries) == 6, "six feature cells required")
    return entries


def audit_cell(seed, fold, path, receipt_path, receipt, old_path, identity):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    legacy = torch.load(old_path/"checkpoint.pt", map_location="cpu", weights_only=False)
    for key, value in (("source", identity), ("legacy_source", runtime.LEGACY_SOURCE), ("stage", "prototype"),
                       ("seed", seed), ("fold", fold)):
        require(saved.get(key) == value, "saved feature identity differs: "+key)
    for key in ("legacy_checkpoint_sha256", "prepared_sha256"):
        require(saved[key] == receipt[key], "feature payload lineage differs: "+key)
    require(saved["private_fit_label_bytes_deserialized_but_not_used"] is True,
            "extraction fit-label byte boundary disclosure missing")
    require(not set(saved) & {"time", "event", "fit_time", "fit_event", "DSS", "endpoint", "endpoints"},
            "endpoint escaped into feature cache")
    fit, valid = np.asarray(saved["fit"]), np.asarray(saved["valid"])
    for key, index in (("fit", fit), ("valid", valid)):
        require(index.ndim == 1 and np.issubdtype(index.dtype, np.integer) and len(index)
                and (index >= 0).all() and len(np.unique(index)) == len(index), "invalid feature partition")
        exact(index, legacy[key], "legacy "+key)
    require(not np.intersect1d(fit, valid).size, "fit/valid overlap")
    array(saved["base_fit"], (len(fit),), "fit offset", float64=True)
    for key in ("M0_valid", "CP_valid", "RNA_valid"):
        array(saved[key], (len(valid),), key, float64=True)
        exact(saved[key], legacy[key], key)
    exact(saved["M0_valid"], saved["CP_valid"]+saved["RNA_valid"], "unit-sum main risk")
    require(set(saved["fit_features"]) == set(runtime.ARMS)
            and set(saved["valid_features"]) == set(runtime.ARMS)
            and set(saved["old_risks"]) == set(runtime.ARMS)
            and set(saved["old_fit_risks"]) == set(runtime.ARMS), "all three feature arms required")
    report = saved["audit"]
    require(report == receipt["audit"] and report["status"] == "PASS_FROZEN_SLOT_EXTRACTION"
            and report["source"] == runtime.LEGACY_SOURCE and report["stage"] == "prototype"
            and report["seed"] == seed and report["fold"] == fold
            and report["fit_patients"] == len(fit) and report["valid_patients"] == len(valid)
            and report["slot_shape"] == [8, 32] and report["feature_width"] == 256
            and report["patient_batch_size"] == 32
            and report["aggregation"] == "equal_WSI_mean_then_flatten"
            and report["fit_representation_crossfitted"] is False,
            "feature extraction audit/shape/scope differs")
    require(report["old_risk_tolerance"] == 1e-7 and report["distribution_tolerance"] == 1e-6
            and report["old_risk_vectors_replayed"] == 15
            and report["endpoint_API_reads"] == report["new_neural_training_steps"] == report["new_predictor_fits"] == 0,
            "feature extraction numerical or isolation contract changed")
    arm_rows, maximum_risk, maximum_distribution = {}, 0., 0.
    for arm in runtime.ARMS:
        array(saved["fit_features"][arm], (len(fit), 256), arm+" fit feature", float64=True)
        require(set(saved["valid_features"][arm]) == set(runtime.CONTROLS), "five frozen controls required")
        for control in runtime.CONTROLS:
            array(saved["valid_features"][arm][control], (len(valid), 256), arm+" "+control, float64=True)
            if arm == "QUERY":
                exact(saved["valid_features"][arm][control], saved["valid_features"][arm]["MATCHED"],
                      "QUERY disabled "+control)
        for key, shape, legacy_key in (("old_risks", (len(valid),), "risk"),
                                        ("old_fit_risks", (len(fit),), "fit_risk")):
            array(saved[key][arm], shape, arm+" "+key, float64=True)
            exact(saved[key][arm], legacy["arms"][arm][legacy_key], arm+" saved old risk")
        row = report["arms"][arm]
        require(row["model_state_unchanged"] is True and row["old_risk_vectors_replayed"] == 5
                and row["QUERY_controls_exact_matched"] == (arm == "QUERY"), "frozen encoder audit differs")
        require(set(row["old_risk_errors"]) == {"FIT_MATCHED", "VALID_MATCHED", "RNA_MEAN_QUERY", "WRONG_RNA_QUERY", "VALUE_OFF"},
                "five old-risk replays required")
        errors = np.asarray(list(row["old_risk_errors"].values()), float)
        require(np.isfinite(errors).all() and (errors >= 0).all() and errors.max() <= 1e-7,
                "old risk replay exceeded prelocked tolerance")
        maximum_risk = max(maximum_risk, float(errors.max()))
        for scope in ("fit", "valid"):
            section = row[scope]
            require(section["inputs_unchanged"] is True and section["values_unchanged"] is True
                    and section["own_RNA_initial_slots_unchanged"] is True
                    and section["probability_distribution_claim"] == (arm != "QUERY"),
                    "input/Value/query invariance audit missing")
        spatial = row["valid"]
        require(spatial["WSI_count"] == spatial["deranged_WSI_count"]+spatial["single_patch_unpermuted_WSI_count"]
                and spatial["WSI_count"] >= len(valid), "spatial control WSI counts disagree")
        require(spatial["routing_calls"] == (0 if arm == "QUERY" else 2*spatial["patient_batches"]),
                "routing did not use exactly two matched matrices per batch")
        invariants = spatial["maximum_distribution_error"]
        if arm == "QUERY":
            require(invariants == {}, "QUERY disabled weights must have no distribution interpretation")
        else:
            require(set(invariants) == {"row_weight_multiset", "row_mass", "entropy", "effective_patch_count",
                                       "soft_slot_overlap_gram", "assignment_row_fraction", "slot_mass"},
                    "spatial distribution invariants incomplete")
            values = np.asarray(list(invariants.values()), float)
            require(np.isfinite(values).all() and (values >= 0).all() and values.max() <= 1e-6,
                    "frozen routing distribution invariance exceeded tolerance")
            require(invariants["row_weight_multiset"] == 0 and invariants["slot_mass"] == 0,
                    "exact weight multiset/slot-mass preservation failed")
            maximum_distribution = max(maximum_distribution, float(values.max()))
        arm_rows[arm] = {"fit_shape": [len(fit), 256], "valid_shape_each_control": [len(valid), 256],
                         "valid_controls": 5, "old_risk_max_error": float(errors.max()),
                         "valid_spatial_counts": {key: spatial[key] for key in
                            ("WSI_count", "deranged_WSI_count", "single_patch_unpermuted_WSI_count", "routing_calls")},
                         "maximum_distribution_errors": invariants,
                         "extraction_recorded_model_state_unchanged": True,
                         "QUERY_controls_exact_matched": arm == "QUERY"}
    require(report["maximum_old_risk_error"] == maximum_risk, "cell old-risk maximum disagrees")
    return {"seed": seed, "fold": fold, "feature_receipt_sha256": runtime.sha(receipt_path),
            "feature_sha256": receipt["sha256"], "legacy_checkpoint_sha256": receipt["legacy_checkpoint_sha256"],
            "prepared_sha256": receipt["prepared_sha256"], "fit_patients": len(fit), "valid_patients": len(valid),
            "old_risk_vectors_replayed_at_extraction": 15, "old_risk_max_error": maximum_risk,
            "maximum_distribution_error": maximum_distribution, "arms": arm_rows}


def main(expected_source=None):
    began = time.monotonic()
    target = runtime.REPORT/"FEATURE_AUDIT.json"
    require(not target.exists(), "feature audit already exists; refusing silent rerun")
    identity, script_sha = runtime.source(), runtime.sha(Path(__file__))
    if expected_source is not None:
        require(identity == expected_source, "explicitly authorized source differs")
    with guards() as calls:
        prepared, evidence = inherited_evidence()
        entries = preflight(identity, prepared)
        rows = [audit_cell(*entry, identity) for entry in entries]
        for _, _, path, receipt_path, receipt, old_path in entries:
            require(runtime.sha(path) == receipt["sha256"]
                    and runtime.sha(receipt_path) == next(row["feature_receipt_sha256"] for row in rows
                        if row["seed"] == receipt["seed"] and row["fold"] == receipt["fold"])
                    and runtime.sha(old_path/"checkpoint.pt") == receipt["legacy_checkpoint_sha256"],
                    "feature receipt/cache/legacy checkpoint changed during read-only audit")
        require(runtime.source() == identity and runtime.sha(Path(__file__)) == script_sha
                and not any(calls.values()), "source/script/isolation changed during audit")
    report = {"status": "PASS", "stage": "prototype", "source": identity,
              "audit_script_sha256": script_sha, "legacy_source": runtime.LEGACY_SOURCE,
              "cells": 6, "frozen_neural_models_referenced": 18, "private_feature_matrices_checked": 108,
              "fit_feature_matrices": 18, "valid_feature_matrices_by_control": {key: 18 for key in runtime.CONTROLS},
              "old_risk_vectors_replayed_at_extraction": 90,
              "maximum_old_risk_error": max(row["old_risk_max_error"] for row in rows),
              "maximum_distribution_error": max(row["maximum_distribution_error"] for row in rows),
              "old_risk_tolerance": 1e-7, "distribution_tolerance": 1e-6,
              "encoder_forward_calls": 0, "new_neural_training_steps": 0, "new_predictor_fits": 0,
              "endpoint_API_reads": 0, "prepared_label_payloads_deserialized_here": 0,
              "extraction_disclosed_legacy_fit_label_bytes_deserialized_unused": True,
              "forbidden_API_call_counts": calls, "inherited_evidence": evidence,
              "raw_weights_arrays_risks_indices_or_labels_published": False,
              "model_state_unchanged_scope": "SHA_unchanged_original_files_and_source_bound_extraction_runtime_flags;no_new_encoder_forward",
              "spatial_interpretation": "frozen_two_round_routing_correspondence_not_natural_query_or_medical_truth",
              "overlap_interpretation": "soft_Gram_only_no_topk_tie_claim",
              "fit_representation_crossfitted": False, "rows": rows,
              "wall_seconds": time.monotonic()-began}
    runtime.atomic_json(target, report)
    print(json.dumps({"status": report["status"], "cells": 6,
                      "maximum_old_risk_error": report["maximum_old_risk_error"],
                      "maximum_distribution_error": report["maximum_distribution_error"],
                      "bytes": target.stat().st_size}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source")
    main(parser.parse_args().expected_source)

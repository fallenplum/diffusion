"""Full completed Phase26 prototype neural replay without fitting/endpoints.

The independently completed MAIN_REPLAY_AUDIT is hash-bound and reused; no raw
main replay is repeated. A feature-only Data-equivalent view bypasses the new
Data constructor's legacy fit_labels() request, while both generations of that
API are blocked. Existing private fit-label bytes are still deserialized by
prepared loaders, and that boundary is explicitly disclosed in the report.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.phase26_multiquery import data, train
from src.phase26_multiquery.model import MultiQuery
from src.phase26_multiquery.runtime import (PRIVATE, REPORT, NEURAL_ARMS, cfg, source, sha, verify_done, atomic_json)

ATOL = 1e-7


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def compare(actual, expected, label, *, exact=False):
    if isinstance(actual, torch.Tensor): actual = actual.detach().cpu().numpy()
    if isinstance(expected, torch.Tensor): expected = expected.detach().cpu().numpy()
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all(),
            label+" has invalid dimensions/values")
    error = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    require(np.array_equal(a, b) if exact else error <= ATOL, label+" replay mismatch")
    return error


def guards():
    calls = {}
    def forbid(label):
        calls[label] = 0
        def fail(*args, **kwargs):
            calls[label] += 1
            raise RuntimeError("read-only replay forbids " + label)
        return fail
    # Both API generations are guarded. The feature-only view construction
    # never invokes either fit_labels method and retains no supervision array.
    for obj, name, label in (
        (data.Data, "fit_labels", "phase26.Data.fit_labels"),
        (data.old.Data, "fit_labels", "phase25.Data.fit_labels"),
        (train, "train_arm", "phase26.train_arm"), (train, "run_cell", "phase26.run_cell"),
        (data, "build_prepared", "phase26.build_prepared"), (data, "prepare", "phase26.prepare"),
        (data, "fit_ridge", "phase26.fit_ridge"),
        (data.old, "fit_preprocessor", "phase25.fit_preprocessor"),
        (data.old, "build_prepared", "phase25.build_prepared"),
        (data.old, "prepare", "phase25.prepare"), (data.old, "fit_ridge", "phase25.fit_ridge"),
        (data.old.parent, "endpoints", "parent.endpoints"),
        (data.old.parent, "splits", "parent.splits"),
        (data.old.PCA, "fit", "PCA.fit"),
        (torch.optim.Optimizer, "__init__", "Optimizer.__init__")):
        setattr(obj, name, forbid(label))
    from src.phase10_trimodal import math as cox
    cox.fit_ridge = forbid("legacy_cox.fit_ridge")
    return calls


def verify_main(identity):
    report, script = REPORT/"MAIN_REPLAY_AUDIT.json", ROOT/"scripts/replay_phase26_main.py"
    require(report.is_file() and script.is_file(), "completed independent main replay missing")
    saved = json.loads(report.read_text())
    require(saved.get("status") == "PASS_NO_FIT_MAIN_REPLAY" and saved.get("source") == identity
            and saved.get("replay_script_sha256") == sha(script), "main replay status/source/script SHA failed")
    for key, expected in (("prepared_cells", 3), ("new_predictor_fits", 0),
                          ("new_PCA_fits", 0), ("endpoint_API_reads", 0)):
        require(saved.get(key) == expected, "main replay count mismatch: " + key)
    require(saved.get("absolute_tolerance") == 1e-10
            and np.isfinite(saved["maximum_absolute_risk_error"])
            and saved["maximum_absolute_risk_error"] <= 1e-10, "main replay numerical qualification failed")
    require(saved.get("risk_vectors_replayed") == 48 and saved.get("legacy_preprocessing_states_reused") == 12
            and saved.get("valid_labels_loaded_or_used") is False
            and saved.get("fitter_prepare_PCA_and_endpoint_APIs_hard_blocked") is True
            and np.isfinite(saved["maximum_feature_replay_error"])
            and saved["maximum_feature_replay_error"] <= 1e-10, "main replay coverage/isolation failed")
    prepared = {int(key): value for key, value in saved["prepared_sha256_by_fold"].items()}
    require(set(prepared) == {1, 2, 3}, "main replay prepared coverage changed")
    rows = {int(row["fold"]): row for row in saved["fold_rows"]}
    require(set(rows) == set(prepared) and all(rows[k]["prepared_sha256"] == prepared[k] for k in rows),
            "main replay fold rows and prepared map disagree")
    require(all(row["CP_exact_legacy_anchor"] for row in rows.values()), "main replay CP identity failed")
    return {"report_path": report.relative_to(ROOT).as_posix(), "report_sha256": sha(report),
            "script_path": script.relative_to(ROOT).as_posix(), "script_sha256": sha(script),
            "source": identity, "status": saved["status"], "prepared_sha256_by_fold": prepared,
            "legacy_prepared_sha256_by_fold": {k: rows[k]["legacy_prepared_sha256"] for k in rows},
            "raw_state_sha256_by_fold": {k: {"full": rows[k]["fullfit_preprocessing_state_sha256"],
                "teachers": [r["preprocessing_state_sha256"] for r in rows[k]["teacher_rows"]]} for k in rows},
            "legacy_main_replay_binding": saved["legacy_main_replay_binding"],
            "maximum_absolute_risk_error": saved["maximum_absolute_risk_error"],
            "raw_main_replay_calls_repeated_here": 0}


def safe_features(fold, seed, device, prepared_sha256, legacy_sha256):
    inherited = data.old.Data(fold, "prototype", seed, device=device)
    saved, receipt = data.read_prepared(fold)
    legacy, legacy_receipt = data.old.read_prepared(fold, "prototype")
    require(receipt["sha256"] == prepared_sha256 and saved["legacy_prepared_sha256"] == legacy_sha256
            and inherited.prepared_sha256 == legacy_sha256 == legacy_receipt["sha256"],
            "independent main audit and replay prepared bytes differ")
    for key in ("fit", "valid", "active"):
        np.testing.assert_array_equal(getattr(inherited, key), saved[key])
    compare(inherited.R, np.asarray(legacy["R"], np.float32), "original RNA query input", exact=True)
    compare(inherited.C, np.asarray(legacy["C"], np.float32), "original Clinical query input", exact=True)
    # These are exactly the input attributes copied by Phase26 Data.__init__,
    # excluding its old fit_labels request and all private supervision.
    fields = {key: getattr(inherited, key) for key in
              ("fit", "valid", "active", "visual", "mask", "owner", "R", "C", "wrong_rna", "wrong_c")}
    fields.update(fold=fold, seed=seed, stage="prototype", role="inner",
                  base=torch.as_tensor(np.array(saved["base"], copy=True), dtype=torch.float64, device=device),
                  scale=float(saved["scale"]),
                  R0_valid=np.array(saved["M0_valid"], np.float64, copy=True),
                  M0_valid=np.array(saved["M0_valid"], np.float64, copy=True),
                  CP_valid=np.array(saved["CP_valid"], np.float64, copy=True),
                  RNA_valid=np.array(saved["RNA_valid"], np.float64, copy=True))
    raw_state_hashes = {"full": saved["mainstate"]["full_state"]["state_hash"],
                        "teachers": [r["state"]["state_hash"] for r in saved["mainstate"]["teachers"]]}
    for state in [saved["mainstate"]["full_state"]]+[r["state"] for r in saved["mainstate"]["teachers"]]:
        require(state["state_hash"] == data.old.state_hash({k: v for k, v in state.items() if k != "state_hash"}),
                "immutable raw preprocessing state hash changed")
    audit = copy.deepcopy(saved["audit"])
    audit.update(prepared_sha256=receipt["sha256"], local_visual=copy.deepcopy(inherited.audit["local_visual"]),
                 wrong_controls=copy.deepcopy(inherited.audit["wrong_controls"]))
    feature = SimpleNamespace(**fields)
    require(not any(hasattr(feature, key) for key in ("fit_labels", "_fit_time", "_fit_event", "ep", "endpoints")),
            "supervision escaped feature-only Data view")
    del inherited, saved, legacy
    return feature, audit, raw_state_hashes


@torch.no_grad()
def replay_cell(seed, fold, directory, identity, device, main):
    record = torch.load(directory/"checkpoint.pt", map_location="cpu", weights_only=False)
    require(all(record.get(k) == v for k, v in
                (("source", identity), ("stage", "prototype"), ("seed", seed), ("fold", fold))),
            "neural checkpoint identity changed")
    require(set(record["arms"]) == set(NEURAL_ARMS), "neural arm matrix changed")
    ds, audit, raw_hashes = safe_features(fold, seed, device, main["prepared_sha256_by_fold"][fold],
                                        main["legacy_prepared_sha256_by_fold"][fold])
    require(raw_hashes == main["raw_state_sha256_by_fold"][fold], "neural view raw-state hashes differ from main replay")
    np.testing.assert_array_equal(record["fit"], ds.fit)
    np.testing.assert_array_equal(record["valid"], ds.valid)
    maximum = 0.
    for name in ("M0_valid", "CP_valid", "RNA_valid"):
        maximum = max(maximum, compare(record[name], getattr(ds, name), "exact reference "+name, exact=True))
    compare(ds.M0_valid, ds.CP_valid+ds.RNA_valid, "own M0 independent CP+RNA sum", exact=True)
    require(record["data_audit"]["prepared_sha256"] == audit["prepared_sha256"]
            and record["data_audit"]["legacy_prepared_sha256"] == audit["legacy_prepared_sha256"],
            "trained input and replay prepared hashes differ")
    common = record["arms"]["QUERY"]["initial_state"]
    for arm in NEURAL_ARMS:
        initial = record["arms"][arm]["initial_state"]
        require(set(initial) == set(common) and all(torch.equal(initial[k], common[k]) for k in common),
                "arm full initialized module state was not bitwise shared")
        require(not initial["head.2.weight"].any() and not initial["clinical_up.weight"].any(),
                "zero output/Clinical-U initialization changed")
    base_snapshot = ds.base.detach().clone()
    arm_rows, risk_vectors = {}, 0
    for arm in NEURAL_ARMS:
        trained = record["arms"][arm]
        require(trained["training_audit"]["actual_epochs"] == trained["training_audit"]["optimizer_steps"] ==
                len(trained["history"]) == 80 and trained["training_audit"]["heldout_endpoint_reads"] == 0,
                "neural budget/endpoint receipt changed")
        model = MultiQuery(int(trained["clinical_dim"]), arm).to(device).eval()
        model.load_state_dict(trained["model_state"], strict=True)
        weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        risk_row = {}
        for partition, stored_key in (("fit", "fit_risk"), ("valid", "risk")):
            risk, units, _ = train._forward(model, ds, getattr(ds, partition))
            e = compare(risk, trained[stored_key], arm+" "+partition+" risk")
            maximum, risk_vectors = max(maximum, e), risk_vectors+1
            require(float((risk-ds.base[getattr(ds, partition)]).abs().max()) <= .1+1e-12,
                    "fixed log-risk correction bound changed")
            risk_row[partition+"_maximum_absolute_error"] = e
        require(set(trained["sensitivities"]) == set(train.CONTROLS) and len(train.CONTROLS) == 6,
                "six frozen query controls missing or changed")
        controls = {}
        for condition in train.CONTROLS:
            risk, _, _ = train._forward(model, ds, ds.valid, intervention=condition)
            e = compare(risk, trained["sensitivities"][condition], arm+" "+condition)
            maximum, risk_vectors = max(maximum, e), risk_vectors+1
            controls[condition] = {"maximum_absolute_error": e}
            if condition == "OFF":
                compare(risk, ds.M0_valid, arm+" OFF own M0", exact=True)
                controls[condition]["exact_own_M0"] = True
            elif arm == "QUERY":
                compare(risk, trained["risk"], "QUERY inactive score/Value control", exact=True)
                controls[condition]["exact_matched_QUERY_path"] = True
        require(torch.equal(ds.base, base_snapshot), "common main risk mutated during replay")
        require(all(torch.equal(value.detach().cpu(), weights[key]) for key, value in model.state_dict().items()),
                "neural weights/buffers mutated during replay")
        arm_rows[arm] = {**risk_row, "frozen_controls": controls, "weights_and_buffers_unchanged": True,
                         "full_common_initialized_module_state_bitwise_shared": True,
                         "all_frozen_main_offsets_unchanged": True}
        del model
    require(risk_vectors == 24, "cell did not replay all twenty-four neural risk vectors")
    report = {"seed": seed, "fold": fold, "neural_risk_vectors": risk_vectors,
              "maximum_absolute_risk_error": maximum, "reference_vectors": 3,
              "prepared_sha256": audit["prepared_sha256"],
              "legacy_prepared_sha256": audit["legacy_prepared_sha256"],
              "raw_preprocessing_state_hashes": raw_hashes,
              "main_source": audit["source"], "legacy_source": audit["legacy_source"],
              "local_visual_receipt_sha256": audit["local_visual"]["receipt_sha256"],
              "fit_patients": len(ds.fit), "heldout_patients": len(ds.valid), "arms": arm_rows}
    del ds, record
    gc.collect()
    if str(device).startswith("cuda"): torch.cuda.empty_cache()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-root", type=Path, default=PRIVATE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=REPORT/"REPLAY_AUDIT.json")
    a = p.parse_args()
    require(not a.output.exists(), "append-only replay refuses an existing report")
    identity, configuration = source(), cfg()
    matrix = [(int(seed), int(fold)) for fold in configuration["prototype"]["folds"]
              for seed in configuration["prototype"]["seeds"]]
    require(len(matrix) == len(set(matrix)) == 6, "prototype requires six locked cells")
    paths = []
    for seed, fold in matrix:
        path = a.private_root/"prototype"/f"seed{seed}_fold{fold}"
        verify_done(path, identity, seed, fold)
        paths.append((seed, fold, path))
    main_evidence = verify_main(identity)
    blocked_calls = guards()
    train.seed_all(matrix[0][0], matrix[0][1])
    rows = []
    for seed, fold, path in paths:
        rows.append(replay_cell(seed, fold, path, identity, a.device, main_evidence))
        print(f"REPLAY_CELL_PASS seed={seed} fold={fold}", flush=True)
    require(not any(blocked_calls.values()), "endpoint/fitting/optimizer API was called")
    require(source() == identity, "training source changed during replay")
    require(sha(ROOT/main_evidence["report_path"]) == main_evidence["report_sha256"] and
            sha(ROOT/main_evidence["script_path"]) == main_evidence["script_sha256"],
            "bound independent main report/script bytes changed")
    for seed, fold, path in paths: verify_done(path, identity, seed, fold)
    vectors = sum(row["neural_risk_vectors"] for row in rows)
    require(vectors == 144, "incomplete 18x8 neural replay matrix")
    maximum = max(row["maximum_absolute_risk_error"] for row in rows)
    report = {"status": "PASS", "schema": "phase26_full_prototype_endpoint_unused_neural_replay_v1",
        "source": identity, "stage": "prototype", "replay_script_sha256": sha(__file__), "cells": 6, "neural_models": 18,
        "neural_risk_vectors": vectors, "reference_risk_vectors": 18,
        "maximum_absolute_risk_error": maximum, "absolute_tolerance": ATOL,
        "neural_weights_unchanged": True, "all_initialized_module_states_bitwise_shared": True,
        "OFF_exact_own_M0": True, "QUERY_five_inactive_controls_exact_matched": True,
        "new_predictor_fits": 0, "new_optimizer_steps": 0, "endpoint_API_reads": 0,
        "forbidden_API_call_counts": blocked_calls,
        "feature_only_Data_equivalent_view_bypasses_legacy_fit_labels_method": True,
        "private_prepared_fit_label_bytes_deserialized_but_not_used": True,
        "endpoint_bytes_never_loaded_claimed": False,
        "independent_main_replay_reused_no_raw_main_replay_here": True,
        "main_replay_evidence": main_evidence, "patient_arrays_or_weights_published": False,
        "cell_replay": rows}
    atomic_json(a.output, report)
    print(json.dumps({"status": "PASS", "cells": 6, "neural_models": 18,
                      "risk_vectors": vectors, "maximum_absolute_risk_error": maximum,
                      "report_bytes": a.output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()

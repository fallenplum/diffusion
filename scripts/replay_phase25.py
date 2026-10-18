"""Read-only completed Phase25-B prototype checkpoints and anchor replay.

All six DONE/source/byte receipts are checked before model/data loading. No
endpoint method, preprocessing fit, convex fit or neural optimizer may run.
Independent completed main-replay evidence is reused; only trained neural
weights and their common stored offsets are replayed here. The
loader necessarily deserializes private fit-label bytes already in prepared
files; none are requested or used by this executable.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.phase25_local import data, train
from src.phase25_local.model import LocalRouting
from src.phase25_local.runtime import (PRIVATE, REPORT, NEURAL_ARMS, cfg, source, sha, verify_done, atomic_json)

ATOL = 1e-7


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def error(actual, expected, name, *, exact=False):
    if isinstance(actual, torch.Tensor): actual = actual.detach().cpu().numpy()
    if isinstance(expected, torch.Tensor): expected = expected.detach().cpu().numpy()
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all(),
            name+" shape/nonfinite mismatch")
    value = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    require(np.array_equal(a, b) if exact else value <= ATOL, name+" replay tolerance failed")
    return value


def forbidden_guards():
    calls = {}
    def blocker(name):
        calls[name] = 0
        def fail(*args, **kwargs):
            calls[name] += 1
            raise RuntimeError("read-only replay forbids " + name)
        return fail
    for module, name in ((data.Data, "fit_labels"), (train, "train_arm"), (train, "run_cell"),
                         (data.parent, "endpoints"), (data, "fit_preprocessor"),
                         (data, "build_prepared"), (data, "prepare"), (data, "fit_ridge"),
                         (data.PCA, "fit")):
        setattr(module, name, blocker(module.__name__+"."+name))
    from src.phase10_trimodal import math as cox
    cox.fit_ridge = blocker("src.phase10_trimodal.math.fit_ridge")
    return calls


def verify_main_replay(identity):
    path, script = REPORT/"MAIN_REPLAY_AUDIT.json", ROOT/"scripts/replay_phase25_main.py"
    require(path.is_file() and script.is_file(), "independent main replay report/script missing")
    record = json.loads(path.read_text())
    require(record.get("status") == "PASS_NO_FIT_MAIN_REPLAY" and record.get("source") == identity
            and record.get("replay_script_sha256") == sha(script), "main replay status/source/script hash failed")
    for key, expected in (("prepared_cells", 3), ("stored_convex_fits_referenced", 15),
                          ("new_predictor_fits", 0), ("new_PCA_fits", 0), ("endpoint_API_reads", 0),
                          ("teacher_held_risk_vectors_replayed", 9), ("full_R0_valid_vectors_replayed", 3),
                          ("full_CP_valid_vectors_replayed", 3), ("fit_crossfit_offset_vectors_replayed", 3),
                          ("complete_teacher_or_full_preprocessing_states_checked", 12)):
        require(record.get(key) == expected, "main replay completeness mismatch: " + key)
    require(record.get("absolute_tolerance") == 1e-10, "main replay locked tolerance changed")
    require(all(np.isfinite(record[key]) and record[key] <= 1e-10 for key in
                ("maximum_risk_replay_error", "maximum_fullfit_feature_replay_error", "maximum_raw_statistic_check_error")),
            "main replay error exceeds its locked tolerance")
    require(record.get("heldout_labels_read") is False and record.get("endpoint_and_fitter_APIs_hard_blocked") is True,
            "main replay endpoint/fit isolation missing")
    rows = {int(row["fold"]): row["prepared_sha256"] for row in record["fullfit_rows"]}
    require(set(rows) == {1, 2, 3}, "main replay fold coverage changed")
    return {"report_path": path.relative_to(ROOT).as_posix(), "report_sha256": sha(path),
            "script_path": script.relative_to(ROOT).as_posix(), "script_sha256": sha(script),
            "source": identity, "status": record["status"], "new_replay_or_fit_calls_here": 0,
            "maximum_risk_replay_error": record["maximum_risk_replay_error"], "prepared_sha256_by_fold": rows}


@torch.no_grad()
def replay_cell(seed, fold, directory, identity, device, prepared_sha256):
    record = torch.load(directory/"checkpoint.pt", map_location="cpu", weights_only=False)
    require(all(record.get(key) == value for key, value in
                (("source", identity), ("stage", "prototype"), ("seed", seed), ("fold", fold))),
            "checkpoint cell identity changed")
    require(set(record["arms"]) == set(NEURAL_ARMS), "neural arms differ from lock")
    ds = data.Data(fold, "prototype", seed, device=device)
    require(ds.prepared_sha256 == prepared_sha256, "neural replay main receipt differs from independent main audit")
    # Make label requests impossible after the constructor's private byte load.
    del ds._fit_time, ds._fit_event
    np.testing.assert_array_equal(record["fit"], ds.fit)
    np.testing.assert_array_equal(record["valid"], ds.valid)
    maximum = max(error(record["R0_valid"], ds.R0_valid, "shared R0 reference", exact=True),
                  error(record["CP_valid"], ds.CP_valid, "shared CP reference", exact=True))
    common = record["arms"]["R1"]["initial_state"]
    require(set(common) == {"q0", "keys.weight", "values.weight", "head.weight"}, "shared initial modules changed")
    arm_rows = {}
    base_snapshot = ds.base.detach().clone()
    for arm in NEURAL_ARMS:
        stored = record["arms"][arm]
        for key, value in common.items():
            require(torch.equal(stored["initial_state"][key], value), "arm common initial state differs")
        for name in ("rna.weight", "clinical.weight"):
            if name in stored["initial_state"]:
                require(not stored["initial_state"][name].any(), "conditional projection did not initialize zero")
        require(stored["training_audit"]["actual_epochs"] == stored["training_audit"]["optimizer_steps"] ==
                len(stored["history"]) == 80 and stored["training_audit"]["heldout_endpoint_reads"] == 0,
                "completed neural training receipt changed")
        model = LocalRouting(int(stored["clinical_dim"]), arm, 16).to(device).eval()
        model.load_state_dict(stored["model_state"], strict=True)
        weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        current = {}
        for partition, key in (("fit", "fit_risk"), ("valid", "risk")):
            risk, _, _ = train._forward(model, ds, getattr(ds, partition))
            err = error(risk, stored[key], arm+" "+partition+" risk")
            current[partition+"_maximum_absolute_error"] = err
            maximum = max(maximum, err)
        expected = {"OFF"}
        if arm in ("R2", "R3"): expected.update(("RNA_MEAN", "WRONG_RNA"))
        if arm == "R3": expected.update(("C_MEAN", "WRONG_C"))
        require(set(stored["sensitivities"]) == expected, "frozen query controls differ from lock")
        sensitivities = {}
        for condition in sorted(expected):
            risk, _, _ = train._forward(model, ds, ds.valid, intervention=condition)
            err = error(risk, stored["sensitivities"][condition], arm+" frozen "+condition)
            sensitivities[condition] = {"maximum_absolute_error": err}
            maximum = max(maximum, err)
            if condition == "OFF":
                error(risk, ds.R0_valid, arm+" local OFF own R0", exact=True)
                sensitivities[condition]["own_main_risk_bitwise_unchanged"] = True
        require(torch.equal(ds.base, base_snapshot), "replay changed shared own main offsets")
        require(all(torch.equal(value.detach().cpu(), weights[key]) for key, value in model.state_dict().items()),
                "read-only replay changed model weights")
        arm_rows[arm] = {**current, "frozen_query_controls": sensitivities,
                         "common_initial_modules_bitwise_shared": True,
                         "model_weights_unchanged": True,
                         "parameters": int(sum(p.numel() for p in model.parameters()))}
        del model
    result = {"seed": seed, "fold": fold, "stage": "prototype", "maximum_absolute_risk_error": maximum,
              "R0_reference_bitwise_replayed": True, "CP_reference_bitwise_replayed": True,
              "arms": arm_rows, "patient_counts": {"fit": len(ds.fit), "valid": len(ds.valid)}}
    del ds, record
    gc.collect()
    if str(device).startswith("cuda"): torch.cuda.empty_cache()
    return result


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
    require(len(matrix) == len(set(matrix)) == 6, "prototype must contain exactly six locked cells")
    paths = []
    # Refuse partial matrices before touching any prepared/model payload.
    for seed, fold in matrix:
        path = a.private_root/"prototype"/f"seed{seed}_fold{fold}"
        verify_done(path, identity, "prototype", seed, fold)
        paths.append((seed, fold, path))
    main_audit = verify_main_replay(identity)
    calls = forbidden_guards()
    train.seed_all(configuration["prototype"]["seeds"][0], configuration["prototype"]["folds"][0])
    cells = []
    for seed, fold, path in paths:
        cells.append(replay_cell(seed, fold, path, identity, a.device, main_audit["prepared_sha256_by_fold"][fold]))
        print(f"REPLAY_CELL_PASS seed={seed} fold={fold}", flush=True)
    require(not any(calls.values()), "replay attempted an endpoint/fitting call")
    require(source() == identity, "frozen training source changed during replay")
    for seed, fold, path in paths: verify_done(path, identity, "prototype", seed, fold)
    require(sha(ROOT/main_audit["report_path"]) == main_audit["report_sha256"] and
            sha(ROOT/main_audit["script_path"]) == main_audit["script_sha256"], "independent main audit bytes changed")
    maximum = max(c["maximum_absolute_risk_error"] for c in cells)
    report = {"status": "PASS", "schema": "phase25_complete_prototype_endpoint_unused_checkpoint_replay_v1",
        "source": identity, "replay_script_sha256": sha(__file__), "stage": "prototype",
        "cells": len(cells), "neural_models": len(cells)*3, "R0_reference_slots": len(cells),
        "independent_main_replay_report_verified_and_reused": True,
        "raw_main_transform_replay_calls_here": 0,
        "maximum_absolute_risk_error": maximum, "risk_replay_absolute_tolerance": ATOL,
        "new_neural_or_convex_fits": 0, "forbidden_method_call_counts": calls,
        "fit_labels_method_called": False, "parent_endpoints_called": False,
        "prepared_files_deserialize_existing_private_fit_label_bytes": True,
        "fit_labels_never_loaded_from_disk_claimed": False,
        "local_OFF_own_main_risk_bitwise_unchanged": True,
        "model_weights_and_shared_initial_modules_verified": True,
        "patient_arrays_weights_or_identifiers_published": False,
        "main_replay_evidence": main_audit, "cell_replay": cells}
    atomic_json(a.output, report)
    print(json.dumps({"status": report["status"], "neural_models": report["neural_models"],
                      "R0_reference_slots": report["R0_reference_slots"],
                      "maximum_absolute_risk_error": maximum, "report_bytes": a.output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()

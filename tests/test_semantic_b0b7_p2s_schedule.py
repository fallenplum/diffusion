"""Finite scheduler tests with fake processes/resources; zero real workers."""
from argparse import Namespace
from copy import deepcopy
import itertools
import json
from pathlib import Path

import pytest

from scripts import semantic_b0b7_p2s_schedule as schedule


def cells():
    return [{"split": f"outer1_inner{i}", "family": family, "seed": seed, "lambda_pair": weight,
        **{k: "1" * 64 for k in ("input_sha256", "parent_receipt_sha256", "parent_binding_sha256", "parent_state400_sha256")}}
        for i, family, seed, weight in itertools.product(range(3), ("MAIN", "ATTN_MATCHED"), (31, 47, 83), (.1, .03))]


def resources():
    return {"cgroup_usage_bytes": 8 * schedule.GIB, "cgroup_limit_bytes": 96 * schedule.GIB,
        "cgroup_available_bytes": 88 * schedule.GIB, "gpu_total_bytes": 32 * schedule.GIB,
        "gpu_used_bytes": 0, "gpu_utilization_percent": 0, "gpu_compute_apps": [],
        "root_free_bytes": 4 * schedule.GIB, "data_free_bytes": 4 * schedule.GIB,
        "root_and_data_same_filesystem": False, "new_root_written_bytes": 0}


def test_long_trajectories_first_and_interleaved_exact36():
    result = schedule.ordered_cells(cells())
    assert len(result) == len({schedule.key(c) for c in result}) == 36
    assert all(c["lambda_pair"] == .03 for c in result[:18])
    assert all(c["lambda_pair"] == .1 for c in result[18:])
    assert {c["family"] for c in result[:2]} == {"MAIN", "ATTN_MATCHED"}
    assert {c["seed"] for c in result[:18]} == {31, 47, 83}
    assert str(schedule.cell_path("new", result[0])).endswith("MAIN_seed31_pair003")


def active(n):
    return {i: {"process": Namespace(pid=100 + i)} for i in range(n)}


def test_reservations_prevent_startup_overbooking_before_allocations():
    r = resources()
    assert schedule.admission(r, active(5), host_baseline=8 * schedule.GIB, gpu_baseline=0)[0]
    ok, why = schedule.admission(r, active(6), host_baseline=8 * schedule.GIB, gpu_baseline=0)
    assert not ok and "GPU_FUTURE_RESIDENT_RESERVATION" in why


@pytest.mark.parametrize("kind", ["cgroup", "root", "data", "gpu", "nvml_over", "nvml_missing", "root_ceiling"])
def test_resource_limits_stop_new_admission(kind):
    r = resources()
    a = active(1)
    if kind == "cgroup":
        r["cgroup_available_bytes"] = 20 * schedule.GIB
    elif kind == "root":
        r["root_free_bytes"] = 2 * schedule.GIB + schedule.ROOT_OUTPUT_BUDGET
    elif kind == "data":
        r["data_free_bytes"] = 2 * schedule.GIB + schedule.DATA_FEATURE_BUFFER
    elif kind == "gpu":
        r["gpu_used_bytes"] = r["gpu_total_bytes"] - 2 * schedule.GIB
    elif kind == "nvml_over":
        r["gpu_compute_apps"] = [{"pid": 100, "used_MiB": 6000}]
    elif kind == "nvml_missing":
        r["gpu_compute_apps"] = [{"pid": 100, "used_MiB": None}]
    else:
        r["new_root_written_bytes"] = schedule.ROOT_OUTPUT_BUDGET + 1
    assert not schedule.admission(r, a, host_baseline=8 * schedule.GIB, gpu_baseline=0)[0]


def test_disk_budgets_combine_only_on_same_filesystem():
    r = resources()
    r["root_free_bytes"] = r["data_free_bytes"] = 2 * schedule.GIB + 1_000_000_000
    assert schedule.admission(r, {}, host_baseline=8 * schedule.GIB, gpu_baseline=0)[0]
    r["root_and_data_same_filesystem"] = True
    assert not schedule.admission(r, {}, host_baseline=8 * schedule.GIB, gpu_baseline=0)[0]


def completed_fixture(tmp_path, monkeypatch, weight):
    cell = cells()[0]
    cell["lambda_pair"] = weight
    root = tmp_path / "new"
    folder = schedule.cell_path(root, cell)
    folder.mkdir(parents=True)
    lock = tmp_path / "lock.json"
    schedule.write(lock, {"frozen_sources": {"source": "a" * 64}})
    args = Namespace(output_root=str(root), lock=str(lock), expected_lock_sha256="b" * 64)
    old = {"negative_support": {"same": True}}
    parent = tmp_path / "original400.private.pt"
    parent.write_bytes(b"unchanged original checkpoint")
    cell["parent_state400_sha256"] = schedule.sha(parent)
    monkeypatch.setattr(schedule, "parent_metadata", lambda unused_args, unused_cell: (old, parent.resolve()))
    steps = 400 if weight == .1 else 800
    states = []
    for epoch in (400, 800):
        ref = epoch == 400 and weight == .1
        path = parent if ref else folder / f"state_epoch{epoch}.private.pt"
        if not ref:
            path.write_bytes(f"new checkpoint epoch {epoch}".encode())
        states.append({"epoch": epoch, "file": str(path.resolve()) if ref else path.name,
            "sha256": schedule.sha(path), "identity": "VERIFIED_FORMAL400_REFERENCE_NO_NEW_UPDATES" if ref else "NEW_P2S_SAVED_STATE",
            "actual_replay": True, "replay_max_error": 0., "replay_preserved_training_rng": True})
    receipt = {"schema": "semantic_b0b7_p2s_cell_v1", "status": "COMPLETE",
        "master_lock_sha256": args.expected_lock_sha256, "base_p2_master_lock_sha256": schedule.P2_SHA,
        "parent_cell": cell, "input_sha256": cell["input_sha256"], "DSS_OS_PFI_accessed": False,
        "frozen_sources": {"source": "a" * 64}, "split": cell["split"], "family": cell["family"],
        "seed": cell["seed"], "spec": {"lambda_pair": weight},
        "optimizer_steps": steps, "optimizer_lifetime_steps": 800, "resume_epoch": 800 - steps,
        "selected_epoch": None, "no_DSS_or_auxiliary_epoch_selection": True,
        "negative_support": old["negative_support"], "negative_mask_sha256": "c" * 64, "runtime": {},
        "history": [{"epoch": e, "inactive_parameters": [], "gradient_norms": {"weight": 1.},
                     "active_parameter_elements": 1} for e in range(801 - steps, 801)],
        "states": states, "checkpoint_paths": {str(s["epoch"]): str((folder / s["file"]).resolve()) for s in states}}
    schedule.write(folder / "PRE_FIT_INPUT_BINDING.private.json", receipt)
    schedule.write(folder / "cell.private.json", receipt)
    return args, cell, folder, receipt


@pytest.mark.parametrize("weight,snapshots,steps", [(.1, 1, 400), (.03, 2, 800)])
def test_only_complete_hash_qualified_checkpoints_are_reused(tmp_path, monkeypatch, weight, snapshots, steps):
    args, cell, _, _ = completed_fixture(tmp_path, monkeypatch, weight)
    row = schedule.checked_complete(args, cell)
    assert row["new_snapshots"] == snapshots and row["optimizer_steps"] == steps
    assert row["actual_replays"] == 2


@pytest.mark.parametrize("tamper", ["missing_receipt", "weight_bytes", "wrong_epoch", "missing_replay", "gradient_zero",
                                  "history_gap", "scope", "prefit", "DSS", "selected"])
def test_partial_or_tampered_cell_is_never_automatically_resumed(tmp_path, monkeypatch, tamper):
    args, cell, folder, receipt = completed_fixture(tmp_path, monkeypatch, .03)
    path = folder / "cell.private.json"
    if tamper == "missing_receipt":
        path.unlink()
    elif tamper == "weight_bytes":
        (folder / receipt["states"][1]["file"]).write_bytes(b"changed")
    else:
        if tamper == "wrong_epoch":
            receipt["states"][1]["epoch"] = 799
        elif tamper == "missing_replay":
            receipt["states"][0]["actual_replay"] = False
        elif tamper == "gradient_zero":
            receipt["history"][0]["gradient_norms"]["weight"] = 0.
        elif tamper == "history_gap":
            receipt["history"][20]["epoch"] = 17
        elif tamper == "scope":
            receipt["input_sha256"] = "z" * 64
        elif tamper == "prefit":
            receipt["negative_mask_sha256"] = "z" * 64
        elif tamper == "DSS":
            receipt["DSS_OS_PFI_accessed"] = True
        else:
            receipt["selected_epoch"] = 800
        schedule.write(path, receipt, replace=True)
    with pytest.raises(ValueError):
        schedule.checked_complete(args, cell)


def test_launch_command_keeps_scientific_parameters_out_of_scheduler_controls(tmp_path):
    a = Namespace(**{k: k for k in ("lock", "expected_lock_sha256", "p2_lock", "private", "assets", "legacy_root", "frozen_images")},
                  output_root=str(tmp_path))
    c = cells()[0]
    argv = schedule.command(a, c)
    assert "--lr" not in argv and "--epoch" not in argv and "--resume" not in argv
    assert argv[argv.index("--threads") + 1] == "2"
    env = schedule.worker_environment()
    assert all(env[k] == "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_worker_failure_drains_other_worker_without_signalling(tmp_path, monkeypatch):
    root = tmp_path / "new"
    old = tmp_path / "original"
    old.mkdir()
    lock = {"cells": cells()}
    monkeypatch.setattr(schedule, "verify_locks", lambda unused_args: lock)
    monkeypatch.setattr(schedule, "parent_metadata", lambda unused_args, unused_cell: None)
    done, processes = set(), []
    def complete(unused_args, cell):
        if schedule.key(cell) not in done:
            return None
        return {"optimizer_steps": 800, "actual_replays": 2, "new_snapshots": 2}
    monkeypatch.setattr(schedule, "checked_complete", complete)
    monkeypatch.setattr(schedule, "telemetry", lambda unused_args: resources())
    monkeypatch.setattr(schedule.time, "sleep", lambda unused: None)
    class FakeProcess:
        def __init__(self, argv):
            self.pid = 100 + len(processes)
            self.cell = {"split": argv[argv.index("--split") + 1], "family": argv[argv.index("--family") + 1],
                "seed": int(argv[argv.index("--seed") + 1]), "lambda_pair": float(argv[argv.index("--lambda-pair") + 1])}
            processes.append(self)
        def poll(self):
            if self.pid == 100:
                return 1
            done.add(schedule.key(self.cell))
            return 0
        def terminate(self):
            raise AssertionError("must not signal workers")
        kill = terminate
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda argv, **unused_kwargs: FakeProcess(argv))
    argv = ["scheduler", "--lock", "lock", "--expected-lock-sha256", "sha", "--execution-lock", "child",
        "--expected-execution-lock-sha256", "childsha", "--p2-lock", "parent", "--private", str(old),
        "--output-root", str(root), "--assets", "assets", "--legacy-root", "legacy", "--frozen-images", "frozen",
        "--data-path", str(tmp_path), "--max-parallel", "2"]
    monkeypatch.setattr(schedule.sys, "argv", argv)
    with pytest.raises(RuntimeError, match="DRAINED_OR_FAILED"):
        schedule.main()
    assert len(processes) == 2 and not (root / "ACTIVE_SCHEDULER_OWNER.private.json").exists()
    receipts = list(root.glob("SCHEDULER_FAILURE_*.private.json"))
    assert len(receipts) == 1
    result = schedule.read(receipts[0])
    assert result["worker_signals_sent"] == 0 and result["complete_cells"] == 1
    assert result["drain_reason"] == "WORKER_FAILURE_DRAIN_REMAINING_NATURALLY"

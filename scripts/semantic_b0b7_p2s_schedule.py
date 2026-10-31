"""Metadata-only finite36 P2S scheduler; failures drain, never kill workers.

The original formal tree is read-only. Admission controls execution resources,
not learning rates, architectures, data partitions, or training budgets. A
partial cell is never restarted; only COMPLETE, byte-qualified cells are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/semantic_b0b7_p2s_schedule.py"
TEST = "tests/test_semantic_b0b7_p2s_schedule.py"
WORKER = "scripts/semantic_b0b7_p2s_pretrain.py"
P2_SHA = "b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
GIB = 1024 ** 3
GPU_RESERVATION = 4_800_000_000
HOST_RESERVATION = 6 * GIB
ROOT_OUTPUT_BUDGET = 750_000_000
DATA_FEATURE_BUFFER = 800_000_000


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".writing")
        with temp.open("x", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write("\n")
        os.replace(temp, path)
    else:
        with path.open("x", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write("\n")


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def key(cell):
    return cell["split"], cell["family"], int(cell["seed"]), float(cell["lambda_pair"])


def cell_path(output_root, cell):
    return Path(output_root) / "phase_a" / cell["split"] / (
        cell["family"] + "_seed" + str(cell["seed"]) + "_pair" + str(round(cell["lambda_pair"] * 100)).zfill(3))


def ordered_cells(cells):
    lookup = {key(c): c for c in cells}
    result = []
    for pair in (.03, .1):
        for turn, (seed, inner) in enumerate(itertools.product((31, 47, 83), range(3))):
            families = ("MAIN", "ATTN_MATCHED") if turn % 2 == 0 else ("ATTN_MATCHED", "MAIN")
            for family in families:
                result.append(lookup[("outer1_inner" + str(inner), family, seed, pair)])
    require(len(result) == 36 and len({key(c) for c in result}) == 36, "P2S_EXACT36_REQUIRED")
    return result


def verify_locks(args):
    require(sha(args.lock) == args.expected_lock_sha256, "P2S_PRETRAINING_LOCK_CHANGED")
    lock = read(args.lock)
    require(lock.get("schema") == "semantic_b0b7_p2s_lock_v1" and
        lock.get("status") == "LOCKED_BEFORE_P2S_OPTIMIZER" and
        lock.get("parent_master_sha256") == P2_SHA, "P2S_PRETRAINING_SCOPE_NOT_LOCKED")
    require(sha(args.p2_lock) == P2_SHA, "P2S_ORIGINAL_MASTER_CHANGED")
    parent = read(args.p2_lock)
    require(set(parent["frozen_sources"]) <= set(lock["frozen_sources"]), "P2S_PARENT_SOURCE_CLOSURE_INCOMPLETE")
    require(all(lock["frozen_sources"][k] == v for k, v in parent["frozen_sources"].items()),
            "P2S_PARENT_SOURCE_REBOUND")
    require(sha(args.execution_lock) == args.expected_execution_lock_sha256, "P2S_EXECUTION_LOCK_CHANGED")
    execution = read(args.execution_lock)
    require(execution.get("schema") == "semantic_b0b7_p2s_execution_lock_v1" and
        execution.get("status") == "LOCKED_BEFORE_P2S_SCHEDULER_LAUNCH" and
        execution.get("parent_pretraining_lock_sha256") == args.expected_lock_sha256 and
        set(execution.get("execution_sources", {})) == {SCRIPT, TEST}, "P2S_EXECUTION_SCOPE_NOT_LOCKED")
    sources = {**lock["frozen_sources"], **execution["execution_sources"]}
    require(WORKER in sources, "P2S_WORKER_SOURCE_NOT_LOCKED")
    for name, expected in sources.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == expected,
                "P2S_SOURCE_CHANGED:" + name)
    require(sha(args.frozen_images) == parent["assets"]["frozen_images_sha256"] and
        sha(args.assets) == parent["assets"]["assets_json_sha256"], "P2S_FROZEN_INPUT_CONTAINER_CHANGED")
    expected = set(itertools.product(("outer1_inner0", "outer1_inner1", "outer1_inner2"),
                                    ("MAIN", "ATTN_MATCHED"), (31, 47, 83), (.1, .03)))
    cells = lock.get("cells", [])
    require(len(cells) == 36 and {key(c) for c in cells} == expected, "P2S_MATRIX_CHANGED")
    return lock


def parent_metadata(args, cell):
    folder = Path(args.private) / "phase_a" / cell["split"] / (cell["family"] + "_seed" + str(cell["seed"]))
    rp, bp = folder / "cell.private.json", folder / "MATRIX_BINDING.private.json"
    require(sha(rp) == cell["parent_receipt_sha256"] and sha(bp) == cell["parent_binding_sha256"],
            "P2S_PARENT_RECEIPT_CHANGED")
    receipt, binding = read(rp), read(bp)
    require(receipt["input_sha256"] == binding["training_only_input_sha256"] == cell["input_sha256"] and
        binding["p2_lock_sha256"] == P2_SHA and binding["cell_receipt_sha256"] == cell["parent_receipt_sha256"] and
        binding["split"] == cell["split"] and binding["family"] == cell["family"] and binding["seed"] == cell["seed"],
        "P2S_PARENT_INPUT_BINDING_CHANGED")
    states = [r for r in receipt["states"] if r["epoch"] == 400]
    require(len(states) == 1 and states[0]["actual_replay"] and states[0]["sha256"] == cell["parent_state400_sha256"],
            "P2S_PARENT400_INCOMPLETE")
    path = folder / states[0]["file"]
    require(path.resolve().parent == folder.resolve() and not path.is_symlink() and
        sha(path) == cell["parent_state400_sha256"], "P2S_PARENT400_CHECKPOINT_CHANGED")
    return receipt, path.resolve()


def checked_complete(args, cell):
    folder = cell_path(args.output_root, cell)
    if not folder.exists():
        return None
    require(folder.is_dir() and not folder.is_symlink() and folder.resolve().is_relative_to(Path(args.output_root).resolve()),
            "P2S_OUTPUT_PATH_ESCAPES_ROOT")
    path = folder / "cell.private.json"
    require(path.is_file(), "P2S_PARTIAL_CELL_REQUIRES_MANUAL_REVIEW_NO_AUTOMATIC_RESTART")
    receipt = read(path)
    old, parent400 = parent_metadata(args, cell)
    steps = 400 if cell["lambda_pair"] == .1 else 800
    require(receipt.get("schema") == "semantic_b0b7_p2s_cell_v1" and receipt.get("status") == "COMPLETE" and
        receipt.get("master_lock_sha256") == args.expected_lock_sha256 and
        receipt.get("base_p2_master_lock_sha256") == P2_SHA and receipt.get("parent_cell") == cell and
        receipt.get("input_sha256") == cell["input_sha256"] and receipt.get("DSS_OS_PFI_accessed") is False,
        "P2S_COMPLETE_RECEIPT_BINDING_CHANGED")
    require(receipt.get("frozen_sources") == read(args.lock)["frozen_sources"] and
        receipt.get("split") == cell["split"] and receipt.get("family") == cell["family"] and
        receipt.get("seed") == cell["seed"] and receipt.get("spec", {}).get("lambda_pair") == cell["lambda_pair"],
        "P2S_COMPLETE_SOURCE_OR_VARIANT_CHANGED")
    prefit = read(folder / "PRE_FIT_INPUT_BINDING.private.json")
    require(all(prefit.get(k) == receipt.get(k) for k in ("master_lock_sha256", "base_p2_master_lock_sha256",
        "input_sha256", "frozen_sources", "parent_cell", "negative_mask_sha256", "spec", "runtime")),
        "P2S_PREFIT_BINDING_DIFFERS_FROM_COMPLETE")
    require(receipt.get("optimizer_steps") == steps and receipt.get("optimizer_lifetime_steps") == 800 and
        receipt.get("resume_epoch") == 800 - steps and receipt.get("selected_epoch") is None and
        receipt.get("no_DSS_or_auxiliary_epoch_selection") is True and
        receipt.get("negative_support") == old["negative_support"], "P2S_COMPLETE_BUDGET_OR_SUPPORT_CHANGED")
    history = receipt.get("history", [])
    require(len(history) == steps and [r["epoch"] for r in history] == list(range(801 - steps, 801)) and
        all(not r["inactive_parameters"] and finite(r) and r.get("active_parameter_elements", 0) > 0 and
            bool(r.get("gradient_norms")) and all(v is not None and v > 0 for v in r["gradient_norms"].values())
            for r in history), "P2S_HISTORY_OR_GRADIENT_INCOMPLETE")
    states = receipt.get("states", [])
    require([r["epoch"] for r in states] == [400, 800] and
        all(r.get("actual_replay") is True and r.get("replay_max_error") == 0. and
            r.get("replay_preserved_training_rng") is True for r in states), "P2S_REAL_REPLAY_INCOMPLETE")
    new_snapshots = 0
    for row in states:
        checkpoint = folder / row["file"]
        if cell["lambda_pair"] == .1 and row["epoch"] == 400:
            require(checkpoint.resolve() == parent400 and row["sha256"] == cell["parent_state400_sha256"] and
                row["identity"] == "VERIFIED_FORMAL400_REFERENCE_NO_NEW_UPDATES", "P2S_PARENT400_REFERENCE_CHANGED")
        else:
            require(not Path(row["file"]).is_absolute() and checkpoint.resolve().is_relative_to(folder.resolve()) and
                not checkpoint.is_symlink() and row["identity"] == "NEW_P2S_SAVED_STATE", "P2S_NEW_STATE_PATH_UNSAFE")
            new_snapshots += 1
        require(checkpoint.is_file() and sha(checkpoint) == row["sha256"], "P2S_SAVED_CHECKPOINT_BYTES_CHANGED")
        require(receipt["checkpoint_paths"][str(row["epoch"])] == str(checkpoint.resolve()), "P2S_CHECKPOINT_PATH_BINDING_CHANGED")
    return {"split": cell["split"], "family": cell["family"], "seed": cell["seed"],
        "lambda_pair": cell["lambda_pair"], "receipt_sha256": sha(path), "optimizer_steps": steps,
        "actual_replays": 2, "new_snapshots": new_snapshots}


def tree_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        total += sum((Path(root) / f).stat().st_size for f in files if not (Path(root) / f).is_symlink())
    return total


def telemetry(args):
    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1.exists():
        limit = int(v1.read_text()); usage = int(v1.with_name("memory.usage_in_bytes").read_text())
    else:
        maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        require(maximum != "max", "P2S_FINITE_CGROUP_LIMIT_REQUIRED")
        limit = int(maximum); usage = int(Path("/sys/fs/cgroup/memory.current").read_text())
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = [[int(x.strip()) for x in line.split(",")] for line in gpu.stdout.strip().splitlines()]
    require(len(rows) == 1, "P2S_SINGLE_ORIGINAL_GPU_TELEMETRY_REQUIRED")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    pids = []
    for line in processes.stdout.strip().splitlines():
        pid, mem = (x.strip() for x in line.split(","))
        pids.append({"pid": int(pid), "used_MiB": int(mem) if mem.isdigit() else None})
    root, data = Path(args.output_root), Path(args.data_path)
    return {"epoch_seconds": time.time(), "cgroup_usage_bytes": usage, "cgroup_limit_bytes": limit,
        "cgroup_available_bytes": limit - usage, "gpu_total_bytes": rows[0][0] * 1024 ** 2,
        "gpu_used_bytes": rows[0][1] * 1024 ** 2, "gpu_utilization_percent": rows[0][2],
        "gpu_compute_apps": pids, "gpu_compute_apps_raw": processes.stdout,
        "root_free_bytes": shutil.disk_usage(root).free, "data_free_bytes": shutil.disk_usage(data).free,
        "root_and_data_same_filesystem": os.stat(root).st_dev == os.stat(data).st_dev,
        "new_root_written_bytes": tree_bytes(root)}


def admission(resource, active, *, host_baseline, gpu_baseline):
    reasons = []
    if resource["cgroup_available_bytes"] <= 20 * GIB:
        reasons.append("ACTUAL_CGROUP_AVAILABLE_NOT_ABOVE20GiB")
    if max(resource["cgroup_usage_bytes"], host_baseline + len(active) * HOST_RESERVATION) + HOST_RESERVATION >= resource["cgroup_limit_bytes"] - 20 * GIB:
        reasons.append("HOST_STARTUP_RESERVATION")
    remaining = max(0, ROOT_OUTPUT_BUDGET - resource["new_root_written_bytes"])
    if resource["new_root_written_bytes"] > ROOT_OUTPUT_BUDGET:
        reasons.append("ROOT_OUTPUT750MB_CEILING_EXCEEDED")
    root_need = 2 * GIB + remaining + (DATA_FEATURE_BUFFER if resource["root_and_data_same_filesystem"] else 0)
    if resource["root_free_bytes"] <= root_need:
        reasons.append("ROOT_DISK_RESERVE_WITH_REMAINING_OUTPUT_BUDGET")
    if resource["data_free_bytes"] <= 2 * GIB + DATA_FEATURE_BUFFER + (remaining if resource["root_and_data_same_filesystem"] else 0):
        reasons.append("DATA_DISK_RESERVE_WITH_FEATURE_BUFFER")
    reserved = len(active) * GPU_RESERVATION
    outside = max(gpu_baseline, resource["gpu_used_bytes"] - reserved, 0)
    if resource["gpu_total_bytes"] - resource["gpu_used_bytes"] <= 2 * GIB:
        reasons.append("ACTUAL_GPU2GiB_RESERVE")
    if reserved + GPU_RESERVATION + outside > resource["gpu_total_bytes"] - 2 * GIB:
        reasons.append("GPU_FUTURE_RESIDENT_RESERVATION")
    owned = {p["process"].pid for p in active.values()}
    for row in resource["gpu_compute_apps"]:
        if row["pid"] in owned:
            if row["used_MiB"] is None:
                reasons.append("OWNED_NVML_MEMORY_UNAVAILABLE")
            elif row["used_MiB"] * 1024 ** 2 > GPU_RESERVATION:
                reasons.append("OWNED_NVML_MEMORY_EXCEEDS_PROFILE_RESERVATION")
    return not reasons, sorted(set(reasons))


def command(args, cell):
    result = [sys.executable, str(ROOT / WORKER)]
    for name in ("lock", "expected_lock_sha256", "p2_lock", "private", "assets", "legacy_root", "frozen_images"):
        result += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    result += ["--output-newdir", str(cell_path(args.output_root, cell)), "--threads", "2", "--device", "cuda"]
    for name in ("split", "family", "seed", "lambda_pair"):
        result += ["--" + name.replace("_", "-"), str(cell[name])]
    return result


def worker_environment():
    return {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "expected-lock-sha256", "execution-lock", "expected-execution-lock-sha256", "p2-lock",
                 "private", "output-root", "assets", "legacy-root", "frozen-images", "data-path"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--max-parallel", type=int, default=6)
    a = p.parse_args()
    require(1 <= a.max_parallel <= 6, "P2S_EXECUTION_PARALLELISM_MUST_BE1_TO6")
    lock = verify_locks(a)
    root = Path(a.output_root).resolve()
    require(root != Path(a.private).resolve() and not root.is_relative_to(Path(a.private).resolve()) and
        not Path(a.private).resolve().is_relative_to(root), "P2S_OLD_FORMAL_TREE_MUST_REMAIN_SEPARATE_READONLY")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    owner = root / "ACTIVE_SCHEDULER_OWNER.private.json"
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    fd = os.open(owner, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "run_id": run_id, "pretraining_lock_sha256": a.expected_lock_sha256,
                   "execution_lock_sha256": a.expected_execution_lock_sha256}, stream)
    pending, active, complete, launched, reused, failures = [], {}, [], [], [], []
    draining = {"reason": None}
    def signal_drain(signum, unused_frame):
        draining["reason"] = "OWNER_SIGNAL_" + str(signum) + "_NO_WORKER_SIGNAL_SENT"
    signal.signal(signal.SIGTERM, signal_drain)
    signal.signal(signal.SIGINT, signal_drain)
    last_resource, blocked, host_baseline, gpu_baseline = None, [], 0, 0
    try:
        try:
            for cell in ordered_cells(lock["cells"]):
                result = checked_complete(a, cell)
                if result is None:
                    pending.append(cell)
                else:
                    complete.append(result); reused.append(result)
        except BaseException as exc:
            failures.append({"type": type(exc).__name__, "message": str(exc)})
            draining["reason"] = "INITIAL_QUALIFICATION_FAILURE_NO_WORKER_STARTED"
        while pending or active:
            try:
                for jobkey, entry in list(active.items()):
                    exit_code = entry["process"].poll()
                    if exit_code is not None:
                        entry["log"].close()
                        del active[jobkey]
                        if exit_code != 0:
                            failures.append({"cell": entry["cell"], "pid": entry["process"].pid, "exit_code": exit_code})
                            draining["reason"] = "WORKER_FAILURE_DRAIN_REMAINING_NATURALLY"
                        else:
                            result = checked_complete(a, entry["cell"])
                            require(result is not None, "P2S_WORKER_EXIT0_WITHOUT_QUALIFIED_COMPLETE")
                            complete.append(result)
                last_resource = telemetry(a)
                if not active:
                    host_baseline, gpu_baseline = last_resource["cgroup_usage_bytes"], last_resource["gpu_used_bytes"]
                blocked = []
                while pending and len(active) < a.max_parallel and draining["reason"] is None:
                    admitted, blocked = admission(last_resource, active, host_baseline=host_baseline, gpu_baseline=gpu_baseline)
                    if not admitted:
                        break
                    cell = pending[0]
                    verify_locks(a)
                    parent_metadata(a, cell)  # exact input/config context is cryptographically bound, not re-deserialized here.
                    path = cell_path(root, cell)
                    require(not path.exists(), "P2S_NEW_CELL_APPEARED_OR_PARTIAL_NO_RESTART")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    name = cell["split"] + "_" + cell["family"] + "_" + str(cell["seed"]) + "_pair" + str(round(cell["lambda_pair"] * 100))
                    log = (root / "logs" / (run_id + "_" + name + ".private.log")).open("x", encoding="utf-8")
                    argv = command(a, cell)
                    try:
                        process = subprocess.Popen(argv, cwd=ROOT, env=worker_environment(), stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close(); raise
                    active[key(cell)] = {"cell": cell, "process": process, "log": log, "command": argv}
                    launched.append({"cell": cell, "pid": process.pid, "command": argv})
                    pending.pop(0)
                totals = {"complete_cells": len(complete), "actual_completed_optimizer_updates": sum(c["optimizer_steps"] for c in complete),
                    "actual_checkpoint_replays": sum(c["actual_replays"] for c in complete),
                    "new_saved_snapshots": sum(c["new_snapshots"] for c in complete)}
                write(root / "LIVE_STATE.private.json", {"schema": "semantic_b0b7_p2s_live_v1", "run_id": run_id,
                    "owner_pid": os.getpid(), "pretraining_lock_sha256": a.expected_lock_sha256,
                    "execution_lock_sha256": a.expected_execution_lock_sha256, **totals,
                    "pending_cells": len(pending), "active_workers": [{"pid": e["process"].pid, "cell": e["cell"],
                        "command": e["command"], "gpu_reserved_bytes": GPU_RESERVATION} for e in active.values()],
                    "max_parallel": a.max_parallel, "admission_blocked_reasons": blocked,
                    "fixed_execution_reserves": {"GPU_per_candidate_bytes": GPU_RESERVATION,
                        "GPU_free_reserve_bytes": 2 * GIB, "cgroup_actual_available_minimum_bytes": 20 * GIB,
                        "new_root_outputs_ceiling_bytes": ROOT_OUTPUT_BUDGET,
                        "data_feature_buffer_bytes": DATA_FEATURE_BUFFER},
                    "drain_reason": draining["reason"], "resource": last_resource}, replace=True)
            except BaseException as exc:
                failures.append({"type": type(exc).__name__, "message": str(exc)})
                draining["reason"] = "SCHEDULER_EXCEPTION_DRAIN_REMAINING_NATURALLY"
            if draining["reason"] is not None and not active:
                break
            if pending or active:
                time.sleep(20)
        success = not failures and not pending and len(complete) == 36 and draining["reason"] is None
        totals = {"complete_cells": len(complete), "actual_completed_optimizer_updates": sum(c["optimizer_steps"] for c in complete),
            "actual_checkpoint_replays": sum(c["actual_replays"] for c in complete),
            "new_saved_snapshots": sum(c["new_snapshots"] for c in complete)}
        if success:
            require(totals == {"complete_cells": 36, "actual_completed_optimizer_updates": 21600,
                "actual_checkpoint_replays": 72, "new_saved_snapshots": 54}, "P2S_FINITE_TOTAL_BUDGET_MISMATCH")
        write(root / (("SCHEDULER_COMPLETE_" if success else "SCHEDULER_FAILURE_") + run_id + ".private.json"), {
            "schema": "semantic_b0b7_p2s_schedule_receipt_v1", "status": "COMPLETE" if success else "FAILED_OR_DRAINED",
            "pretraining_lock_sha256": a.expected_lock_sha256, "execution_lock_sha256": a.expected_execution_lock_sha256,
            "run_id": run_id, "cells": complete, "launched_this_invocation": launched, "reused_complete": reused,
            "failures": failures, "pending_cells": pending, "worker_signals_sent": 0,
            "drain_reason": draining["reason"], "resource": last_resource, **totals})
        if not success:
            raise RuntimeError("P2S_DRAINED_OR_FAILED_NO_PARTIAL_RESTART")
    finally:
        # Normal signals/exceptions reach here only after owned workers exited.
        # SIGKILL leaves the O_EXCL marker in place for explicit manual review.
        if not active and owner.exists() and read(owner).get("run_id") == run_id:
            owner.unlink()


if __name__ == "__main__":
    main()

"""Finite 15-cell queue for ONE role; failures never trigger automatic refits."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase23_cdei.runtime import (PRIVATE, atomic_json, cell_directory, cell_name,
                                     cfg, require_run_authorization, source, verify_done)


@contextmanager
def scheduler_lock(role):
    import fcntl
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / f"{role}_SCHEDULER.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_queue(role, *, workers=3, device="cuda"):
    config, identity = cfg(), source()
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= workers <= 15:
        raise ValueError("workers must be between 1 and 15")
    workers = int(workers)
    matrix = [(int(seed), int(fold)) for fold in config["folds"] for seed in config["seeds"]]
    if len(matrix) != 15 or len(set(matrix)) != 15:
        raise ValueError("scheduler requires the fixed fifteen unique seed/fold cells")
    require_run_authorization(role, identity)
    with scheduler_lock(role):
        todo = list(matrix)
        running, completed, failed = [], [], []
        logs = PRIVATE / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        progress_path = PRIVATE / f"{role}_SCHEDULE_PRIVATE.json"
        started = time.time()

        def progress(status):
            record = {"status": status, "source": identity, "role": role,
                      "total": 15, "completed": len(completed), "failed": len(failed),
                      "pending": len(todo), "workers": workers, "device": device,
                      "running": [{"cell": row["cell"], "pid": row["child"].pid}
                                  for row in running], "completed_cells": completed,
                      "failed_cells": failed, "unix_start": started, "unix_update": time.time(),
                      "automatic_retries": 0}
            atomic_json(progress_path, record)
            return record

        while todo or running:
            while todo and len(running) < workers:
                seed, fold = todo.pop(0)
                label = cell_name(seed, fold)
                target = cell_directory(role, seed, fold)
                if (target / "DONE_PRIVATE.json").exists():
                    try:
                        verify_done(target, role=role, seed=seed, fold=fold, identity=identity)
                        completed.append(label)
                    except Exception as error:
                        failed.append({"cell": label, "reason": "INVALID_EXISTING_DONE",
                                       "error_type": type(error).__name__})
                    continue
                # A prior incomplete attempt is evidence of work already done.
                # Do not silently restart it from epoch zero after a scheduler
                # interruption; the root must inspect/reconcile that attempt.
                attempts = target / "ATTEMPTS_PRIVATE.json"
                if attempts.exists():
                    failed.append({"cell": label, "reason": "PRIOR_INCOMPLETE_ATTEMPT_REQUIRES_REVIEW"})
                    continue
                try:
                    current_source = source()
                except Exception as error:
                    failed.append({"cell": label, "reason": "SOURCE_UNAVAILABLE_BEFORE_LAUNCH",
                                   "error_type": type(error).__name__})
                    continue
                if current_source != identity:
                    failed.append({"cell": label, "reason": "SOURCE_DRIFT_BEFORE_LAUNCH"})
                    continue
                log = (logs / f"{role}_{label}.log").open("a", encoding="utf-8")
                try:
                    child = subprocess.Popen(
                        [sys.executable, str(ROOT / "scripts/run_phase23_cdei.py"),
                         "--role", role, "--seed", str(seed), "--fold", str(fold),
                         "--device", device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                except Exception as error:
                    log.close()
                    failed.append({"cell": label, "reason": "LAUNCH_FAILED",
                                   "error_type": type(error).__name__})
                    continue
                running.append({"child": child, "log": log, "cell": label,
                                "seed": seed, "fold": fold, "target": target})
                print("START", role, label, child.pid, flush=True)
            for row in list(running):
                code = row["child"].poll()
                if code is None:
                    continue
                row["log"].close()
                running.remove(row)
                if code != 0:
                    failed.append({"cell": row["cell"], "reason": "NONZERO_CHILD_EXIT", "exit_code": code})
                else:
                    try:
                        verify_done(row["target"], role=role, seed=row["seed"],
                                    fold=row["fold"], identity=identity)
                        completed.append(row["cell"])
                        print("DONE", role, row["cell"], len(completed), flush=True)
                    except Exception as error:
                        failed.append({"cell": row["cell"], "reason": "EXIT_ZERO_WITHOUT_VALID_DONE",
                                       "error_type": type(error).__name__})
            progress("RUNNING")
            if running:
                time.sleep(3)
        result = progress("COMPLETE" if not failed and len(completed) == 15 else "PARTIAL_FAILED")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inner", "outer"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_queue(args.role, workers=args.workers, device=args.device)
    print("SCHEDULE_FINISHED", result["status"], result["completed"], result["failed"], flush=True)
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()

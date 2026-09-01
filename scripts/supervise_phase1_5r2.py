"""Persistent supervisor for the Phase1.5R2 resumable runner.

The Windows desktop terminal can reclaim a silent/long-lived child process.
This small wrapper keeps an observable heartbeat and restarts the runner only
when its state is still RUNNING and the child has exited.  The runner itself
owns the lock and all resume semantics; this wrapper never edits result files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise resumable Phase1.5R2 runner")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = root / "outputs" / "phase1_5r2_functional_repair" / "run_state.json"
    runner = root / "scripts" / "run_phase1_5r2.py"
    interval = max(5, int(args.interval))
    print(f"supervisor started root={root}", flush=True)
    while True:
        state = read_state(state_path)
        if state.get("status") in {"COMPLETE", "FAILED"}:
            print(
                f"terminal state={state.get('status')} stage={state.get('stage')} "
                f"completed={len(state.get('completed_jobs', []))} "
                f"failed={len(state.get('failed_jobs', []))}",
                flush=True,
            )
            return 0 if state.get("status") == "COMPLETE" else 1

        cmd = [sys.executable, str(runner), "--root", str(root), "--resume"]
        print(f"starting runner command={' '.join(cmd)}", flush=True)
        child = subprocess.Popen(cmd, cwd=str(root))
        while child.poll() is None:
            time.sleep(interval)
            state = read_state(state_path)
            print(
                f"heartbeat pid={child.pid} status={state.get('status')} "
                f"stage={state.get('stage')} current={state.get('current_job')} "
                f"completed={len(state.get('completed_jobs', []))} "
                f"failed={len(state.get('failed_jobs', []))}",
                flush=True,
            )

        state = read_state(state_path)
        print(
            f"runner exited pid={child.pid} returncode={child.returncode} "
            f"status={state.get('status')} stage={state.get('stage')} "
            f"completed={len(state.get('completed_jobs', []))} "
            f"failed={len(state.get('failed_jobs', []))}",
            flush=True,
        )
        if state.get("status") in {"COMPLETE", "FAILED"}:
            return 0 if state.get("status") == "COMPLETE" else 1
        # A killed process can leave a stale lock; the runner removes it after
        # verifying that the recorded PID is no longer alive.
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())

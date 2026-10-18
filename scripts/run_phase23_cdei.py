"""One source-qualified Phase23 cell with an exclusive lock and durable steps."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase23_cdei.runtime import (atomic_json, cell_directory, cfg,
                                     require_run_authorization, sha, source, verify_done)


@contextmanager
def exclusive_cell_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inner", "outer"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config, identity = cfg(), source()
    if args.seed not in config["seeds"] or args.fold not in config["folds"]:
        raise ValueError("cell is outside the fixed Phase23 matrix")
    require_run_authorization(args.role, identity)
    target = cell_directory(args.role, args.seed, args.fold)
    with exclusive_cell_lock(target / "CELL.lock"):
        if (target / "DONE_PRIVATE.json").exists():
            verify_done(target, role=args.role, seed=args.seed, fold=args.fold, identity=identity)
            print("VERIFIED_COMPLETE", args.role, target.name, flush=True)
            return
        attempt_path = target / "ATTEMPTS_PRIVATE.json"
        ledger = (json.loads(attempt_path.read_text(encoding="utf-8")) if attempt_path.exists()
                  else {"attempts": []})
        attempt = {"source": identity, "role": args.role, "seed": args.seed,
                   "fold": args.fold, "status": "RUNNING", "unix_start": time.time(),
                   "callback_calls": 0, "kernel_optimizer_steps": 0}
        ledger["attempts"].append(attempt)
        atomic_json(attempt_path, ledger)

        def stepped(kind, epoch):
            attempt["callback_calls"] += 1
            attempt["last_stage"] = str(kind)
            attempt["last_epoch"] = int(epoch)
            if kind == "kernel_warmup":
                attempt["kernel_optimizer_steps"] += 1
            atomic_json(attempt_path, ledger)

        try:
            # Lazy import keeps --help and artifact checks free of fit setup.
            from src.phase23_cdei.train import run_cell
            result = run_cell(role=args.role, seed=args.seed, fold=args.fold,
                              device=args.device, output_dir=target, source=identity,
                              step_callback=stepped)
            if not isinstance(result, dict) or source() != identity:
                raise ValueError("invalid returned DONE or training source drift")
            if result.get("status", "COMPLETE") not in ("COMPLETE", "PASS"):
                raise ValueError("training did not return a complete qualifying cell")
            fixed = {"source": identity, "role": args.role, "seed": args.seed, "fold": args.fold}
            if any(key in result and result[key] != value for key, value in fixed.items()):
                raise ValueError("training returned a different cell identity")
            artifacts = {}
            for name in ("kernel.pt", "readouts.pt"):
                path = target / name
                if not path.is_file() or path.stat().st_size < 1:
                    raise ValueError("training did not persist both nonempty private checkpoints")
                artifacts[name] = sha(path)
            done = {**result, **fixed, "status": "COMPLETE", "artifacts": artifacts,
                    "unix_complete": time.time()}
            atomic_json(target / "DONE_PRIVATE.json", done)
            verify_done(target, role=args.role, seed=args.seed, fold=args.fold, identity=identity)
        except BaseException as error:
            attempt.update(status="FAILED", error_type=type(error).__name__, unix_end=time.time())
            atomic_json(attempt_path, ledger)
            raise
        attempt.update(status="COMPLETE", unix_end=time.time())
        atomic_json(attempt_path, ledger)
        print("CELL_COMPLETE", args.role, target.name, flush=True)


if __name__ == "__main__":
    main()

"""Run a validated experiment request and produce an auditable result bundle.

The runner deliberately keeps the existing Phase P1 scripts untouched.  A
``smoke`` request executes a deterministic synthetic metric check.  A ``full``
request must provide an explicit, repository-relative command; no shell is
used and patient data are resolved only through an environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - requirements install this in normal use
    Draft202012Validator = None

from phase1_utils import censoring_km, harrell_c, uno_c


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "automation" / "schemas" / "experiment_request.schema.json"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,95}$")
ABSOLUTE_PATH = re.compile(r"(^|[=\s])(?:[A-Za-z]:[\\/]|/|\\\\)")
SHELL_META = re.compile(r"[;&|<>`\r\n]")


class RequestError(ValueError):
    """A user-facing request validation failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestError(f"YAML file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise RequestError(f"invalid YAML: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RequestError(f"YAML root must be an object: {path}")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config_root = (root / "configs").resolve()
    resolved = config_path.resolve()
    if not _inside(resolved, config_root):
        raise RequestError("config.path must resolve inside configs/")
    payload = load_yaml(resolved)
    parent_name = payload.pop("extends", None)
    if parent_name is None:
        return payload
    if not isinstance(parent_name, str) or Path(parent_name).is_absolute() or ".." in Path(parent_name).parts:
        raise RequestError("config extends must be a relative path without '..'")
    parent = (resolved.parent / parent_name).resolve()
    if not _inside(parent, config_root):
        raise RequestError("config extends must remain inside configs/")
    return deep_merge(load_config(root, parent), payload)


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _validate_without_jsonschema(request: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment",
        "code",
        "config",
        "execution",
        "dataset",
        "reproducibility",
        "outputs",
    }
    missing = sorted(required - set(request))
    if missing:
        raise RequestError(f"request is missing required keys: {missing}")
    if request.get("schema_version") != 1:
        raise RequestError("schema_version must be 1")


def validate_request(root: Path, request: dict[str, Any], request_path: Path) -> tuple[Path, dict[str, Any]]:
    if Draft202012Validator is not None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(request), key=lambda error: list(error.path))
        if errors:
            location = ".".join(str(part) for part in errors[0].path) or "<root>"
            raise RequestError(f"request schema error at {location}: {errors[0].message}")
    else:
        _validate_without_jsonschema(request)

    experiment = request["experiment"]
    execution = request["execution"]
    dataset = request["dataset"]
    config_spec = request["config"]
    if not SAFE_ID.fullmatch(experiment["id"]):
        raise RequestError("experiment.id contains unsafe characters")
    if execution["mode"] == "full" and not dataset["requires_patient_data"]:
        raise RequestError("full requests must set dataset.requires_patient_data: true")
    if execution["device"] == "cuda" and not execution["expected_gpu"]:
        raise RequestError("execution.device=cuda requires expected_gpu: true")
    env_name = dataset["data_root_env"]
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", env_name):
        raise RequestError("dataset.data_root_env must be an uppercase environment variable name")

    for value in _walk_strings(request):
        if ABSOLUTE_PATH.search(value):
            raise RequestError("request must not contain an absolute filesystem path")

    config_value = config_spec["path"]
    config_path = (root / config_value).resolve()
    if not _inside(config_path, root / "configs"):
        raise RequestError("config.path must remain inside configs/")
    config = load_config(root, config_path)

    required_dirs = dataset.get("required_directories", [])
    for relative in required_dirs:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RequestError("dataset.required_directories must be relative")

    command = execution.get("command")
    if execution["mode"] == "full" and not command:
        raise RequestError("full requests must provide execution.command")
    if command:
        for token in command:
            if SHELL_META.search(token) or ABSOLUTE_PATH.search(token):
                raise RequestError("execution.command must be a relative no-shell command")
            if token.endswith((".py", ".ps1", ".sh")):
                script_path = (root / token).resolve()
                if not _inside(script_path, root):
                    raise RequestError("execution.command script must remain inside the repository")

    requested_commit = str(request["code"]["base_commit"])
    current_commit = git_commit_sha(root)
    if re.fullmatch(r"[0-9a-fA-F]{40}", requested_commit) and current_commit and requested_commit != current_commit:
        raise RequestError(
            f"request base_commit {requested_commit} does not match checked-out commit {current_commit}"
        )
    if request_path.resolve() == SCHEMA_PATH.resolve():
        raise RequestError("request path cannot be the schema itself")
    return config_path, config


def git_commit_sha(root: Path) -> str | None:
    if os.environ.get("GIT_COMMIT_SHA"):
        return os.environ["GIT_COMMIT_SHA"].strip()
    commands = [
        ["git", "rev-parse", "HEAD"],
        ["git", "--git-dir=.git_local", "--work-tree=.", "rev-parse", "HEAD"],
    ]
    for command in commands:
        try:
            result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        value = result.stdout.strip()
        if value:
            return value
    return None


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _request_hash(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[tuple[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["metric,value"] + [f"{name},{value}" for name, value in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _smoke_experiment(run_dir: Path, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray([f"SMOKE-P{i:03d}" for i in range(1, 13)])
    durations = np.arange(2.0, 14.0)
    events = np.asarray([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    risks = np.linspace(1.5, -1.5, len(patient_ids)) + rng.normal(0.0, 0.01, len(patient_ids))
    censor = censoring_km(durations, events)
    g = censor(durations, left_limit=True)
    harrell = harrell_c(durations, events, risks)
    uno = uno_c(durations, events, risks, g)
    predictions = run_dir / "smoke_predictions.csv"
    prediction_lines = ["patient_id,dss_months,dss_status,risk"]
    prediction_lines.extend(
        f"{pid},{time},{event},{risk}" for pid, time, event, risk in zip(patient_ids, durations, events, risks)
    )
    predictions.write_text("\n".join(prediction_lines) + "\n", encoding="utf-8")
    _write_csv(
        run_dir / "metrics.csv",
        [
            ("n_patients", len(patient_ids)),
            ("n_events", int(events.sum())),
            ("harrell_c", f"{harrell:.8f}"),
            ("uno_c", f"{uno:.8f}"),
        ],
    )
    return {
        "synthetic": True,
        "n_patients": len(patient_ids),
        "n_events": int(events.sum()),
        "harrell_c": float(harrell),
        "uno_c": float(uno),
    }


def _run_external_command(root: Path, run_dir: Path, request: dict[str, Any]) -> dict[str, Any]:
    command = list(request["execution"]["command"])
    python_aliases = {"python", "python3", "python.exe"}
    if command and Path(command[0]).name.lower() in python_aliases:
        command[0] = sys.executable
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["DIFFUSION_EXPERIMENT_ID"] = request["experiment"]["id"]
    environment["DIFFUSION_RUN_DIR"] = str(run_dir)
    timeout = int(request["execution"]["timeout_minutes"] * 60)
    log_path = run_dir / "logs" / "experiment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(partial, encoding="utf-8")
        raise TimeoutError(f"experiment command exceeded {request['execution']['timeout_minutes']} minutes") from exc
    output = (completed.stdout or "") + (completed.stderr or "")
    log_path.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"experiment command failed with exit code {completed.returncode}")
    metrics_available = (run_dir / "metrics.csv").exists()
    if not metrics_available:
        _write_csv(run_dir / "metrics.csv", [("external_command_completed", 1)])
    return {
        "external_command": command,
        "return_code": completed.returncode,
        "metrics_available": metrics_available,
    }


def _run_preflight(root: Path, run_dir: Path, request: dict[str, Any]) -> None:
    command = [
        sys.executable,
        str(root / "scripts" / "gpu_preflight.py"),
        "--root",
        str(root),
        "--data-root-env",
        request["dataset"]["data_root_env"],
        "--report-path",
        str(run_dir / "preflight_report.json"),
        "--min-free-gb",
        "0" if request["execution"]["mode"] == "smoke" else "5",
    ]
    if request["dataset"]["requires_patient_data"]:
        command.append("--require-data")
    if request["execution"]["expected_gpu"] or request["execution"]["device"] == "cuda":
        command.append("--require-gpu")
    for relative in request["dataset"].get("required_directories", []):
        command.extend(["--required-dir", relative])
    result = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True, timeout=120)
    log_path = run_dir / "logs" / "preflight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("GPU/data preflight failed; see preflight_report.json and logs/preflight.log")


def _initial_state(request: dict[str, Any], commit_sha: str | None, request_path: Path, root: Path) -> dict[str, Any]:
    relative_request = _relative_to_root(root, request_path)
    return {
        "schema_version": 1,
        "experiment_id": request["experiment"]["id"],
        "commit_sha": commit_sha,
        "request_path": relative_request,
        "config_path": request["config"]["path"],
        "mode": request["execution"]["mode"],
        "status": "PENDING",
        "start_time": None,
        "end_time": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "resume_command": f"python scripts/run_research_experiment.py --request {relative_request} --resume",
    }


def run_request(root: Path, request_path: Path, *, resume: bool = False, dry_run: bool = False) -> int:
    request_path = request_path.resolve()
    if not _inside(request_path, root / "automation" / "requests"):
        raise RequestError("request must be inside automation/requests/")
    request = load_yaml(request_path)
    config_path, config = validate_request(root, request, request_path)
    commit_sha = git_commit_sha(root)
    experiment_id = request["experiment"]["id"]
    run_dir = root / "outputs" / "automated_runs" / experiment_id
    if dry_run:
        plan = {
            "dry_run": True,
            "experiment_id": experiment_id,
            "mode": request["execution"]["mode"],
            "request_path": _relative_to_root(root, request_path),
            "config_path": _relative_to_root(root, config_path),
            "commit_sha": commit_sha,
            "data_root_env": request["dataset"]["data_root_env"],
            "requires_patient_data": request["dataset"]["requires_patient_data"],
            "expected_gpu": request["execution"]["expected_gpu"],
            "command": request["execution"].get("command"),
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    state_path = run_dir / "run_state.json"
    if run_dir.exists() and not resume and any(run_dir.iterdir()):
        raise RequestError(f"run directory already exists; use --resume: {run_dir}")
    if resume:
        if not state_path.exists():
            raise RequestError(f"cannot resume without run_state.json: {run_dir}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE":
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
    else:
        state = _initial_state(request, commit_sha, request_path, root)
    run_dir.mkdir(parents=True, exist_ok=True)
    state["status"] = "RUNNING"
    state["start_time"] = state.get("start_time") or utc_now()
    state["commit_sha"] = commit_sha
    atomic_write_json(state_path, state)
    active_job = "preflight"
    result_details: dict[str, Any] = {}
    try:
        _run_preflight(root, run_dir, request)
        if active_job not in state["completed_jobs"]:
            state["completed_jobs"].append(active_job)
        active_job = "smoke.synthetic" if request["execution"]["mode"] == "smoke" else "external.command"
        if active_job in state["completed_jobs"] and (run_dir / "summary.json").exists():
            result_details = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")).get("metrics", {})
        elif request["execution"]["mode"] == "smoke":
            result_details = _smoke_experiment(run_dir, int(request["reproducibility"]["seeds"][0]))
        else:
            result_details = _run_external_command(root, run_dir, request)
        if active_job not in state["completed_jobs"]:
            state["completed_jobs"].append(active_job)
        state["status"] = "COMPLETE"
    except TimeoutError as exc:
        state["status"] = "PARTIAL"
        state["failed_jobs"].append(active_job)
        state["error"] = str(exc)
        result_details = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - state/report must be written for every failure
        state["status"] = "FAILED"
        state["failed_jobs"].append(active_job)
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc(limit=8)
        result_details = {"error": str(exc)}
    finally:
        state["end_time"] = utc_now()
        atomic_write_json(state_path, state)
        result_class = "smoke" if request["execution"]["mode"] == "smoke" else "primary"
        summary = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "result_class": result_class,
            "status": state["status"],
            "commit_sha": commit_sha,
            "request_path": _relative_to_root(root, request_path),
            "config_path": _relative_to_root(root, config_path),
            "metrics": result_details,
            "synthetic": request["execution"]["mode"] == "smoke",
            "note": "Smoke metrics are synthetic and are not formal scientific results."
            if result_class == "smoke"
            else "Metrics originate from the explicitly requested external experiment command.",
        }
        atomic_write_json(run_dir / "summary.json", summary)
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "result_class": result_class,
            "request_sha256": _request_hash(request),
            "commit_sha": commit_sha,
            "request_path": _relative_to_root(root, request_path),
            "config_path": _relative_to_root(root, config_path),
            "mode": request["execution"]["mode"],
            "device": request["execution"]["device"],
            "data_root_env": request["dataset"]["data_root_env"],
            "requires_patient_data": request["dataset"]["requires_patient_data"],
            "seed_values": request["reproducibility"]["seeds"],
            "fold_values": request["reproducibility"]["folds"],
            "started_at": state["start_time"],
            "finished_at": state["end_time"],
        }
        atomic_write_json(run_dir / "experiment_manifest.json", manifest)
        (run_dir / "SUMMARY.md").write_text(
            "# Automated experiment run\n\n"
            f"- experiment_id: `{experiment_id}`\n"
            f"- status: **{state['status']}**\n"
            f"- result_class: `{result_class}`\n"
            f"- commit_sha: `{commit_sha or 'unknown'}`\n\n"
            + ("This is a synthetic smoke run; do not use it as a scientific result.\n"
               if result_class == "smoke" else "This is a primary run; inspect metrics and logs before interpretation.\n"),
            encoding="utf-8",
        )
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["status"] == "COMPLETE" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        return run_request(args.root.resolve(), args.request, resume=args.resume, dry_run=args.dry_run)
    except RequestError as exc:
        print(f"request validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Check the host before a research run and emit a machine-readable report."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _git_sha(root: Path) -> str | None:
    candidates = [
        (["git", "rev-parse", "HEAD"], root),
        (["git", "--git-dir=.git_local", "--work-tree=.", "rev-parse", "HEAD"], root),
    ]
    for command, cwd in candidates:
        try:
            result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        sha = result.stdout.strip()
        if sha:
            return sha
    return None


def _write_probe(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".preflight-", dir=directory, delete=True):
            pass
    except (OSError, PermissionError):
        return False
    return True


def _torch_info() -> tuple[dict, list[str]]:
    info: dict = {"installed": False}
    errors: list[str] = []
    try:
        import torch
    except ImportError as exc:
        info["error"] = str(exc)
        return info, errors

    info.update(
        {
            "installed": True,
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    )
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info, errors


def run_preflight(
    *,
    root: Path,
    data_root_env: str,
    require_data: bool,
    required_directories: list[str],
    require_gpu: bool,
    report_path: Path,
    min_free_gb: float,
) -> tuple[dict, int]:
    started = datetime.now(timezone.utc)
    errors: list[str] = []
    warnings: list[str] = []
    torch_info, torch_errors = _torch_info()
    errors.extend(torch_errors)

    data_root_value = os.environ.get(data_root_env)
    data_info = {
        "environment_variable": data_root_env,
        "configured": bool(data_root_value),
        "exists": False,
        "required_directories": {},
    }
    if data_root_value:
        data_root = Path(data_root_value).expanduser()
        data_info["exists"] = data_root.is_dir()
        if not data_root.is_dir():
            errors.append(f"{data_root_env} is set but its directory does not exist")
        else:
            for relative in required_directories:
                candidate = (data_root / relative).resolve()
                try:
                    candidate.relative_to(data_root.resolve())
                except ValueError:
                    errors.append(f"required directory escapes data root: {relative}")
                    data_info["required_directories"][relative] = False
                    continue
                exists = candidate.is_dir()
                data_info["required_directories"][relative] = exists
                if not exists:
                    errors.append(f"required data directory is missing: {relative}")
    elif require_data:
        errors.append(f"required data environment variable is not set: {data_root_env}")
    elif required_directories:
        warnings.append("required_directories were supplied but patient data is not required")

    if require_gpu:
        if not torch_info.get("installed"):
            errors.append("PyTorch is not installed; GPU preflight cannot continue")
        elif not torch_info.get("cuda_available"):
            errors.append("CUDA is unavailable; a GPU runner is required for this experiment")
    elif not torch_info.get("installed"):
        warnings.append("PyTorch is not installed; CPU-only preflight skipped torch checks")

    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    disk_info = {"free_gb": round(free_gb, 3), "minimum_free_gb": min_free_gb}
    if free_gb < min_free_gb:
        errors.append(f"free disk space {free_gb:.2f} GiB is below {min_free_gb:.2f} GiB")

    output_parent = report_path.parent if report_path.parent != Path("") else root
    writable = _write_probe(output_parent)
    if not writable:
        errors.append(f"report directory is not writable: {output_parent}")

    report = {
        "schema_version": 1,
        "status": "FAIL" if errors else "PASS",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "pytorch": torch_info,
        "cuda_required": require_gpu,
        "data": data_info,
        "disk": disk_info,
        "write_permission": writable,
        "git_commit_sha": _git_sha(root),
        "warnings": warnings,
        "errors": errors,
    }
    if not report["git_commit_sha"]:
        report["errors"].append("unable to determine the checked-out git commit SHA")
        report["status"] = "FAIL"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, 1 if report["status"] == "FAIL" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root-env", default="DIFFUSION_DATA_ROOT")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--required-dir", action="append", default=[])
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--report-path", type=Path, default=ROOT / "preflight_report.json")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = root / report_path
    report, code = run_preflight(
        root=root,
        data_root_env=args.data_root_env,
        require_data=args.require_data,
        required_directories=args.required_dir,
        require_gpu=args.require_gpu,
        report_path=report_path,
        min_free_gb=args.min_free_gb,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail closed if protected data or large binary artifacts are tracked."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reproducibility import protected_path_violations  # noqa: E402


def tracked_paths(root: Path) -> list[str]:
    commands = [
        ["git", "ls-files", "-z"],
        ["git", "--git-dir=.git_local", "--work-tree=.", "ls-files", "-z"],
    ]
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        return [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]
    raise RuntimeError("Unable to inspect tracked files with git")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = tracked_paths(root)
    violations = protected_path_violations(paths)
    report = {"tracked_count": len(paths), "violations": violations}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if violations:
        print("Protected data/model artifacts are tracked; remove them before merging.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

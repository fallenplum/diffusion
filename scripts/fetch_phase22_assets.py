"""Fetch only a pre-enumerated, pinned Phase22 asset plan onto the local E disk.

Authentication is read from the existing HF store or a process environment
variable; credentials never enter plans, receipts, source or subprocess args.
Plans/receipts contain private source identifiers and stay outside publication.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import get_token, hf_hub_download


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--token-env")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = Path("E:/diffusion/dataset/spatial_bridge").resolve()
    for path in (args.destination, args.receipt):
        if not path.resolve().is_relative_to(root):
            raise ValueError("download and receipt must remain in the authorized E data root")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if len(plan["revision"]) != 40 or not 1 <= args.workers <= 4:
        raise ValueError("pinned revision and finite concurrency required")
    token = os.environ.get(args.token_env) if args.token_env else get_token()
    if not token:
        raise ValueError("authorized credential unavailable")
    args.destination.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    files = plan["files"]
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("duplicate planned paths")
    started = time.monotonic()
    done, errors = [], []
    print("PINNED_DOWNLOAD_PLAN", len(files), sum(item["bytes"] for item in files), flush=True)

    def fetch(item):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe planned relative path")
        path = Path(hf_hub_download(
            plan["repo"], item["path"], repo_type=plan.get("repo_type", "dataset"),
            revision=plan["revision"], token=token, local_dir=args.destination,
        ))
        actual = file_sha256(path)
        if path.stat().st_size != item["bytes"] or (item.get("sha256") and actual != item["sha256"]):
            raise ValueError("download size or SHA mismatch")
        return {"path": item["path"], "bytes": path.stat().st_size,
                "sha256": actual, "repository_LFS_SHA_verified": bool(item.get("sha256"))}

    def receipt(status):
        record = {"status": status, "repo": plan["repo"], "revision": plan["revision"],
                  "plan_sha256": file_sha256(args.plan), "expected_files": len(files),
                  "verified_files": len(done), "files": sorted(done, key=lambda x: x["path"]),
                  "errors": errors, "seconds": time.monotonic() - started,
                  "credentials_persisted": False}
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.receipt)

    receipt("RUNNING")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, item): item for item in files}
        for future in as_completed(pending):
            item = pending[future]
            try:
                done.append(future.result())
                print("VERIFIED_ASSET", item["path"], item["bytes"], flush=True)
            except Exception as error:
                # Avoid URLs/headers in exceptions from gated downloads.
                errors.append({"path": item["path"], "error_class": type(error).__name__})
                print("ASSET_FAILED", item["path"], type(error).__name__, flush=True)
            receipt("RUNNING")
    status = "COMPLETE" if len(done) == len(files) and not errors else "INCOMPLETE"
    receipt(status)
    print("ASSET_DOWNLOAD", status, len(done), len(files), flush=True)
    if status != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

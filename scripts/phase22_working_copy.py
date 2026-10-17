"""Send ephemeral official HF download tickets to an authorized GPU working copy.

Primary assets/archives stay on E. This avoids the slow local upload path for
the finite GPU subset. Tickets are capabilities: stdout must be a private SSH
pipe, never a report/file. Neither tokens nor signed URLs enter receipts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def official_url(value):
    parsed = urlparse(value)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and (host == "huggingface.co" or host.endswith(".hf.co") or host.endswith(".huggingface.co"))


def emit(args):
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
    root = Path("E:/diffusion/dataset/spatial_bridge")
    if args.scope == "weights":
        plan = json.loads((root / "metadata/UNI2_AUTHORIZED_MODEL_INDEX.json").read_text())
        repo_type = "model"
        files = [x for x in plan["files"] if x["path"] == "pytorch_model.bin"]
        destinations = ["models/uni2-h/pytorch_model.bin"]
    else:
        plan = json.loads((root / "metadata/CRC16_QUALIFICATION_ASSET_PLAN.json").read_text())
        repo_type = "dataset"
        excluded = set(args.exclude_sections or [])
        files = [x for x in plan["files"] if x["path"].startswith("patches/") and Path(x["path"]).stem not in excluded]
        destinations = ["raw/hest/" + x["path"] for x in files]
    token = os.environ.get(args.token_env) if args.token_env else get_token()
    tickets = []
    for item, destination in zip(files, destinations):
        metadata = get_hf_file_metadata(hf_hub_url(plan["repo"], item["path"], repo_type=repo_type, revision=plan["revision"]), token=token)
        if not official_url(metadata.location) or metadata.size != item["bytes"] or metadata.etag != item["sha256"]:
            raise ValueError("official transfer source size/hash/host mismatch")
        tickets.append({"url": metadata.location, "bytes": item["bytes"], "sha256": item["sha256"], "relative_path": destination})
    # Caller must connect this stdout directly to the private SSH pipe.
    json.dump({"scope": args.scope, "repo": plan["repo"], "revision": plan["revision"], "files": tickets}, sys.stdout)
    sys.stdout.write("\n"); sys.stdout.flush()


def receive(args):
    import requests
    root = args.root.resolve()
    if root != Path("/root/autodl-tmp/phase22_spatial_bridge"):
        raise ValueError("unapproved GPU working-copy root")
    plan = json.load(sys.stdin)
    files = plan["files"]
    if plan["scope"] not in {"weights", "patches"} or len(files) > 16 or sum(x["bytes"] for x in files) > 9_000_000_000:
        raise ValueError("finite working-copy budget exceeded")
    missing = 0
    for item in files:
        if not official_url(item["url"]):
            raise ValueError("nonofficial artifact host")
        target = root / item["relative_path"]
        if not target.resolve().is_relative_to(root):
            raise ValueError("unsafe working-copy relative path")
        if target.exists():
            if target.stat().st_size != item["bytes"] or sha(target) != item["sha256"]:
                raise ValueError("existing working-copy file is ambiguous; refuse clobber")
        else:
            missing += item["bytes"]
    if shutil.disk_usage(root).free < missing + 2 * 1024**3:
        raise ValueError("working-copy budget must retain 2GiB free")
    print("GPU_FINITE_WORKING_COPY_START", plan["scope"], len(files), missing, flush=True)
    started = time.monotonic()
    verified, errors = [], []

    def fetch(item):
        target = root / item["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".working-copy.part")
            # Signed capabilities are never written to disk or HTTP logs.
            with requests.get(item["url"], stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(chunk_size=8 * 1024**2):
                        if block:
                            stream.write(block)
            if temporary.stat().st_size != item["bytes"] or sha(temporary) != item["sha256"]:
                raise ValueError("GPU artifact checksum mismatch")
            temporary.replace(target)
        return {key: item[key] for key in ("relative_path", "bytes", "sha256")}

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(fetch, item): item for item in files}
        for future in as_completed(pending):
            try:
                result = future.result(); verified.append(result)
                print("GPU_WORKING_ASSET_VERIFIED", result["relative_path"], result["bytes"], flush=True)
            except Exception as error:
                errors.append({"relative_path": pending[future]["relative_path"], "error_class": type(error).__name__})
                print("GPU_WORKING_ASSET_FAILED", type(error).__name__, flush=True)
    record = {"status": "COMPLETE" if not errors else "INCOMPLETE", "scope": plan["scope"],
              "repo": plan["repo"], "revision": plan["revision"], "expected_files": len(files),
              "files": verified, "errors": errors, "seconds": time.monotonic() - started,
              "primary_archive": "local_E_disk", "signed_URLs_or_tokens_persisted": False}
    path = root / "metadata" / ("SERVER_WORKING_COPY_" + plan["scope"].upper() + "_RECEIPT.json")
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print("GPU_WORKING_COPY_DONE", record["status"], len(verified), record["seconds"], flush=True)
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["emit", "receive"])
    parser.add_argument("--scope", choices=["weights", "patches"])
    parser.add_argument("--token-env")
    parser.add_argument("--exclude-sections", nargs="*")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    try:
        emit(args) if args.mode == "emit" else receive(args)
    except Exception as error:
        print("WORKING_COPY_FAILED", type(error).__name__, file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Resume a manifest-checked CONCH v1.5 H5 transfer to the Phase10 server.

Run this on the workstation that has E:\diffusion_conch_v15_features mounted.
The password is requested interactively and is never written to disk.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import posixpath
import shlex
from pathlib import Path

import paramiko
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remote_sha(client, path: str) -> str | None:
    _, out, _ = client.exec_command("sha256sum -- " + shlex.quote(path), timeout=120)
    value = out.read().decode(errors="replace").strip().split()
    return value[0].lower() if value else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/phase6_conch_v15_download_audit/cohort_manifest.csv")
    ap.add_argument("--source-root", default=r"E:\diffusion_conch_v15_features")
    ap.add_argument("--remote-root", default="/root/autodl-tmp/conch_v15_features")
    ap.add_argument("--host", default="connect.westc.seetacloud.com")
    ap.add_argument("--port", type=int, default=36973)
    ap.add_argument("--user", default="root")
    ap.add_argument("--key", default=None, help="optional private key path; avoids password prompt")
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    rows = manifest[manifest.status.eq("EXACT_SLIDE_MATCH")].sort_values(["project", "filename"])
    password = None if args.key else getpass.getpass(f"SSH password for {args.user}@{args.host}: ")
    client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=password, key_filename=args.key, timeout=30,
                   banner_timeout=30, auth_timeout=30)
    sftp = client.open_sftp(); transferred = skipped = failed = 0
    for row in rows.itertuples(index=False):
        rel = str(row.delivery_relative_path).replace("/", os.sep)
        local = Path(args.source_root) / rel
        remote = posixpath.join(args.remote_root, str(row.delivery_relative_path).replace("\\", "/"))
        if not local.is_file():
            print(f"MISSING_LOCAL {local}"); failed += 1; continue
        expected_size = int(row.conch_size_bytes); expected_sha = str(row.conch_sha256).lower()
        if local.stat().st_size != expected_size or digest(local) != expected_sha:
            print(f"LOCAL_HASH_FAIL {local}"); failed += 1; continue
        parent = posixpath.dirname(remote)
        client.exec_command("mkdir -p " + shlex.quote(parent))[1].channel.recv_exit_status()
        # A previous worker may already have atomically completed this path.
        # Verify the final object before touching any .part file.
        try:
            final_size = int(sftp.stat(remote).st_size)
        except IOError:
            final_size = -1
        if final_size == expected_size:
            final_digest = remote_sha(client, remote)
            if final_digest == expected_sha:
                skipped += 1
                print(f"OK_EXISTING {row.filename} bytes={expected_size}")
                continue
        remote_size = 0
        try: remote_size = int(sftp.stat(remote + ".part").st_size)
        except IOError: pass
        if remote_size > expected_size: remote_size = 0
        if remote_size < expected_size:
            with local.open("rb") as src, sftp.open(remote + ".part", "ab") as dst:
                src.seek(remote_size); remaining = expected_size - remote_size
                while remaining:
                    block = src.read(min(8 * 1024 * 1024, remaining))
                    if not block: raise IOError(f"unexpected EOF: {local}")
                    dst.write(block); remaining -= len(block)
            if final_size == expected_size:
                sftp.remove(remote)
            sftp.rename(remote + ".part", remote); transferred += 1
        remote_digest = remote_sha(client, remote)
        if remote_digest != expected_sha:
            print(f"REMOTE_HASH_FAIL {remote} expected={expected_sha} got={remote_digest}"); failed += 1
            continue
        skipped += 1
        print(f"OK {row.filename} bytes={expected_size}")
    sftp.close(); client.close()
    print({"status": "PASS" if failed == 0 and skipped + transferred == len(rows) else "FAIL",
           "expected_slides": len(rows), "transferred": transferred, "verified": skipped, "failed": failed})
    raise SystemExit(0 if failed == 0 and skipped + transferred == len(rows) else 2)


if __name__ == "__main__": main()



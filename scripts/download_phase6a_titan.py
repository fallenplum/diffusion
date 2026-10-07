"""Download the pinned, authorized TITAN slide/text snapshot without saving tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download

REPO = "MahmoodLab/TITAN"
REVISION = "dac6773d9961cfc75503440676ff157a2c6e8d2e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path(r"E:\diffusion_phase6a_titan\model"))
    args = parser.parse_args()
    token = os.environ.get("PHASE6A_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set PHASE6A_HF_TOKEN or HF_TOKEN in the process environment; do not put it in code")
    args.destination.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=token).model_info(REPO, revision=REVISION, files_metadata=True)
    names = [f.rfilename for f in info.siblings if f.rfilename.endswith(".py") or f.rfilename in
             {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"}]
    started = time.monotonic()
    files = []
    for name in names:
        path = Path(hf_hub_download(REPO, name, revision=REVISION, token=token, local_dir=str(args.destination)))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(data)
        metadata = next(f for f in info.siblings if f.rfilename == name)
        if path.stat().st_size != metadata.size:
            raise RuntimeError("Downloaded file size mismatch")
        if name == "model.safetensors":
            expected = metadata.lfs["sha256"] if isinstance(metadata.lfs, dict) else metadata.lfs.sha256
            if digest.hexdigest() != expected:
                raise RuntimeError("Model LFS SHA256 mismatch")
        files.append({"filename": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        print("VERIFIED", name, path.stat().st_size, flush=True)
    manifest = {"repo": REPO, "revision": REVISION, "weight_sha256_verified": True,
                "elapsed_seconds": round(time.monotonic()-started, 2), "files": files}
    (args.destination / "DOWNLOAD_COMPLETE.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

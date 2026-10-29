"""Restore frozen extraction artifacts from explicitly labelled public projections.

Only immutable, public HEST directory metadata is requested. No image, expression,
weight, credential, clinical or survival files are downloaded. The original
artifact bytes are reconstructed into a NEW output directory and SHA-checked.
The projections are NOT executable replacements for the frozen artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "reports/spgel/HEST_FEATURE_PUBLIC_PROJECTIONS_20261029.json"
REVISION = "b039b5965ed7092f280d4f336da7972ef9a10e1a"
REPOSITORY = "MahmoodLab/hest-bench"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def relative_name(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in name
            and ":" not in name and str(p) == name, "UNSAFE_ARTIFACT_PATH")
    return p


def resolve_aliases(manifest, directory_records):
    require(manifest["dataset_repo"] == REPOSITORY
            and manifest["dataset_revision"] == REVISION,
            "IMMUTABLE_DATA_SOURCE_CHANGED")
    bindings = manifest["aliases"]
    require(len(bindings) == 8 and len({r["alias"] for r in bindings}) == 8,
            "EIGHT_UNIQUE_PUBLIC_ALIASES_REQUIRED")
    result = {}
    for binding in bindings:
        task = binding["task"]
        require(task in {"COAD", "READ"}, "UNKNOWN_TASK")
        matches = []
        for row in directory_records[task]:
            path = PurePosixPath(row["path"])
            lfs = row.get("lfs", {})
            oid = lfs.get("oid", lfs.get("sha256"))
            if (row.get("type") == "file" and path.parts[:2] == (task, "patches")
                    and len(path.parts) == 3 and path.suffix == ".h5"
                    and row.get("size") == binding["raw_RGB_bytes"]
                    and oid == binding["raw_RGB_sha256"]):
                matches.append(path.stem)
        require(len(matches) == 1 and matches[0].isascii()
                and matches[0].replace("_", "").isalnum(),
                "PUBLIC_RGB_HASH_DID_NOT_RESOLVE_UNIQUELY")
        result[binding["alias"]] = matches[0]
    require(len(set(result.values())) == 8, "DUPLICATE_PROVIDER_NAME")
    return result


def reconstruct(manifest, aliases, root):
    require(manifest["schema"] == "hest_lossless_public_artifact_projections_v1"
            and manifest["original_artifacts_modified"] is False
            and manifest["projection_execution_supported"] is False,
            "PROJECTION_CONTRACT_CHANGED")
    rows = manifest["artifacts"]
    require(len(rows) == 3 and len({r["original_file"] for r in rows}) == 3,
            "EXACT_THREE_FROZEN_ARTIFACTS_REQUIRED")
    outputs = []
    for row in rows:
        relative_name(row["original_file"])
        projection = root / str(relative_name(row["projection_file"]))
        raw = projection.read_bytes()
        require(len(raw) == row["projection_bytes"]
                and digest(raw) == row["projection_sha256"], "PROJECTION_BYTES_CHANGED")
        envelope = row["projection_envelope"]
        prefix, suffix = envelope["prefix"].encode(), envelope["suffix"].encode()
        require(prefix and suffix and raw.startswith(prefix) and raw.endswith(suffix),
                "PROJECTION_ENVELOPE_CHANGED")
        restored = raw[len(prefix):-len(suffix)]
        for alias, provider in sorted(aliases.items(), key=lambda v: -len(v[0])):
            restored = restored.replace(alias.encode("ascii"), provider.encode("ascii"))
        require(len(restored) == row["original_bytes"]
                and digest(restored) == row["original_sha256"],
                "FROZEN_ORIGINAL_NOT_RESTORED_BYTE_EXACTLY")
        outputs.append((row["original_file"], restored))
    return outputs


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    require(not output.exists(), "OUTPUT_MUST_BE_A_NEW_DIRECTORY")
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    records = {}
    for task in ("COAD", "READ"):
        url = (f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/"
               f"{REVISION}/{task}/patches?recursive=false&expand=false")
        with urlopen(Request(url, headers={"User-Agent": "HEST-fixed-public-metadata"}), timeout=60) as stream:
            require(not stream.headers.get("Link"), "UNEXPECTED_PAGINATED_DIRECTORY")
            raw = stream.read(2_000_001)
        require(len(raw) <= 2_000_000, "PUBLIC_DIRECTORY_TOO_LARGE")
        records[task] = json.loads(raw)
        require(isinstance(records[task], list), "INVALID_PUBLIC_DIRECTORY")
    aliases = resolve_aliases(manifest, records)
    restored = reconstruct(manifest, aliases, root)
    # No output is written until all three original hashes have passed.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in restored:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    print(json.dumps(dict(status="PASS_THREE_BYTE_EXACT_FROZEN_ARTIFACTS_RESTORED",
        restored_artifacts=len(restored), public_directory_requests=2,
        image_expression_or_weight_downloads=0, original_files_modified=False)))


if __name__ == "__main__":
    main()

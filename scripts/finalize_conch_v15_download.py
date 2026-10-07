"""Verify an offline HF snapshot and stage exact cohort matches on the same disk.

No credentials, labels, feature extraction, or model fitting are involved.
The source inventory contains hashes and extraction metadata from existing
UNI2 H5 files; missing slides are retained explicitly in the cohort manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


PROJECTS = {"TCGA-COAD", "TCGA-READ"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_coords(a: np.ndarray, sort: bool = False) -> str:
    a = np.asarray(a, dtype="<i8").reshape(-1, 2)
    if sort:
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_slide(path: Path, metadata: dict, root: Path) -> dict:
    file_hash = sha256(path)
    expected_hash = metadata.get("lfs_sha256")
    if not expected_hash or file_hash != expected_hash:
        raise ValueError("HF LFS SHA256 is missing or does not match")
    if path.stat().st_size != metadata["size"]:
        raise ValueError("file size does not match HF tree metadata")
    with h5py.File(path, "r") as h5:
        feats, coords_ds = h5["features"], h5["coords"]
        if feats.ndim != 2 or feats.shape[1] != 768 or len(feats) == 0:
            raise ValueError(f"invalid features shape: {feats.shape}")
        if coords_ds.shape != (len(feats), 2) or coords_ds.dtype.kind not in "iu":
            raise ValueError(f"invalid coords: {coords_ds.shape}, {coords_ds.dtype}")
        if feats.dtype != np.dtype("float32"):
            raise ValueError(f"unexpected features dtype: {feats.dtype}")
        coords = coords_ds[:]
        if np.any(coords < 0) or len(np.unique(coords, axis=0)) != len(coords):
            raise ValueError("negative or duplicate patch coordinates")
        for start in range(0, len(feats), 8192):
            if not np.isfinite(feats[start:start + 8192]).all():
                raise ValueError("non-finite feature value")
        attrs = coords_ds.attrs
        if int(attrs["patch_size"]) != 512 or int(attrs["target_magnification"]) != 20:
            raise ValueError("unexpected CONCH patch size / target magnification")
        if str(feats.attrs["encoder"]) != "conch_v15":
            raise ValueError("encoder attribute is not conch_v15")
        return {
            "filename": path.name,
            "project": path.parent.name,
            "patient_id": path.name[:12],
        "relative_path": (Path("TCGA") / path.parent.name / path.name).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_hash,
            "hf_sha256_match": True,
            "n_patches": len(feats),
            "feature_dim": 768,
            "feature_dtype": str(feats.dtype),
            "feature_values_finite": True,
            "coords_sha256_ordered": digest_coords(coords),
            "coords_sha256_sorted": digest_coords(coords, sort=True),
            "patch_size_at_20x": int(attrs["patch_size"]),
            "patch_size_level0": int(attrs["patch_size_level0"]),
            "target_magnification": int(attrs["target_magnification"]),
            "level0_magnification": float(attrs["level0_magnification"]),
            "patch_overlap": int(attrs["overlap"]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--tree-json", type=Path, required=True)
    parser.add_argument("--uni-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delivery-root", type=Path, required=True)
    parser.add_argument("--download-seconds", type=float, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot_root.resolve(strict=True)
    delivery = args.delivery_root.resolve()
    if delivery == snapshot or snapshot in delivery.parents or delivery in snapshot.parents:
        raise SystemExit("delivery and snapshot directories must be independent")
    if delivery.drive != snapshot.drive:
        raise SystemExit("same-drive staging required; refusing a second feature copy")
    tree = json.loads(args.tree_json.read_text(encoding="utf-8"))["files"]
    inventory = json.loads(args.uni_inventory.read_text(encoding="utf-8-sig"))
    expected = {
        key: value for key, value in tree.items()
        if len(Path(key).parts) == 3 and Path(key).parts[0] == "TCGA"
        and Path(key).parts[1] in PROJECTS and key.endswith(".h5")
    }
    if not expected or not inventory["slides"]:
        raise SystemExit("empty snapshot metadata or UNI inventory")
    rows, failures = [], []
    for index, (relative, metadata) in enumerate(sorted(expected.items()), 1):
        try:
            source = snapshot / relative
            if not source.exists():
                source = delivery / relative
            rows.append(verify_slide(source, metadata, snapshot))
        except Exception as exc:
            failures.append({"relative_path": relative, "error_type": type(exc).__name__, "error": str(exc)})
        if index % 100 == 0 or index == len(expected):
            print(f"verified {index}/{len(expected)}; failures={len(failures)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        write_json(args.output_dir / "validation.json", {"download_integrity": "FAIL", "failures": failures})
        raise SystemExit("snapshot verification failed; no delivery performed")
    by_name = {row["filename"]: row for row in rows}
    if len(by_name) != len(rows):
        raise SystemExit("duplicate slide filenames across projects")
    if len({row['filename'] for row in inventory['slides']}) != len(inventory['slides']):
        raise SystemExit("duplicate slide filenames in UNI inventory")
    cohort, missing = [], []
    delivery.mkdir(parents=True, exist_ok=True)
    for uni in sorted(inventory["slides"], key=lambda row: row["filename"]):
        conch = by_name.get(uni["filename"])
        uc = uni["coords"]["coords_patching"]
        dest_relative = ""
        storage_method = ""
        if conch:
            if uni["project"] != conch["project"]:
                raise SystemExit("same slide found under different TCGA project")
            src = snapshot / conch["relative_path"]
            dest = delivery / conch["relative_path"]
            if not src.resolve().is_relative_to(snapshot) or not dest.resolve().is_relative_to(delivery):
                raise SystemExit("resolved staging paths escape the approved source or destination")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if sha256(dest) != conch["sha256"]:
                    raise SystemExit(f"destination exists with a different checksum: {dest}")
                storage_method = "REUSED_VERIFIED_FILE"
            else:
                try:
                    os.link(src, dest)
                    storage_method = "HARDLINK"
                except OSError as exc:
                    # exFAT does not implement hard links (Windows error 1).
                    if getattr(exc, "winerror", None) not in (1, 50):
                        raise
                    os.replace(src, dest)
                    storage_method = "MOVED_ON_SAME_DRIVE"
                if dest.stat().st_size != conch["size_bytes"]:
                    raise SystemExit("staged file size differs from verified source")
            dest_relative = conch["relative_path"]
        else:
            missing.append(uni["filename"])
        cohort.append({
            "filename": uni["filename"],
            "project": uni["project"],
            "patient_id": uni["filename"][:12],
            "status": "EXACT_SLIDE_MATCH" if conch else "MISSING_CONCH_SLIDE",
            "delivery_relative_path": dest_relative,
            "storage_method": storage_method,
            "conch_size_bytes": conch["size_bytes"] if conch else "",
            "conch_sha256": conch["sha256"] if conch else "",
            "uni_n_patches": uc["n"],
            "conch_n_patches": conch["n_patches"] if conch else "",
            "uni_patch_size_at_20x": uc["attrs"].get("patch_size", ""),
            "conch_patch_size_at_20x": conch["patch_size_at_20x"] if conch else "",
            "uni_patch_size_level0": uc["attrs"].get("patch_size_level0", ""),
            "conch_patch_size_level0": conch["patch_size_level0"] if conch else "",
            "ordered_coords_equal": bool(conch and uc["sha256_ordered"] == conch["coords_sha256_ordered"]),
            "unordered_coords_equal": bool(conch and uc["sha256_sorted"] == conch["coords_sha256_sorted"]),
            "identical_patch_size_level0": bool(conch and int(uc["attrs"].get("patch_size_level0", -1)) == conch["patch_size_level0"]),
        })
    matched = [row for row in cohort if row["status"] == "EXACT_SLIDE_MATCH"]
    source_patients = {row["patient_id"] for row in cohort}
    matched_patients = {row["patient_id"] for row in matched}
    missing_patient_alternatives = {
        name: [row["filename"] for row in matched if row["patient_id"] == name[:12]]
        for name in missing
    }
    summary = {
        "dataset_repo": "sofieneb/conch_v15_features",
        "dataset_revision": snapshot.name,
        "download_seconds": args.download_seconds,
        "downloaded_slides": len(rows),
        "downloaded_bytes": sum(row["size_bytes"] for row in rows),
        "downloaded_project_counts": dict(Counter(row["project"] for row in rows)),
        "downloaded_patch_count": sum(row["n_patches"] for row in rows),
        "uni_slide_count": len(cohort),
        "uni_patient_count": len(source_patients),
        "exact_matched_slides": len(matched),
        "exact_matched_patients": len(matched_patients),
        "slide_coverage_fraction": len(matched) / len(cohort),
        "patient_coverage_fraction": len(matched_patients) / len(source_patients),
        "delivery_matched_bytes": sum(row["conch_size_bytes"] for row in matched),
        "delivery_matched_project_counts": dict(Counter(row["project"] for row in matched)),
        "missing_slides": missing,
        "same_patient_already_matched_slides_for_missing": missing_patient_alternatives,
        "ordered_coordinate_matches": sum(row["ordered_coords_equal"] for row in matched),
        "unordered_coordinate_matches": sum(row["unordered_coords_equal"] for row in matched),
        "same_patch_size_level0_matches": sum(row["identical_patch_size_level0"] for row in matched),
        "uni_patch_size_at_20x_counts": dict(Counter(row["uni_patch_size_at_20x"] for row in cohort)),
        "conch_patch_size_at_20x_counts": dict(Counter(row["patch_size_at_20x"] for row in rows)),
        "conch_patch_size_level0_counts": dict(Counter(row["patch_size_level0"] for row in rows)),
        "survival_labels_read": False,
        "features_uploaded_to_server": False,
        "delivery_uses_hardlinks": all(row["storage_method"] == "HARDLINK" for row in matched),
        "delivery_storage_method_counts": dict(Counter(row["storage_method"] for row in matched)),
    }
    validation = {
        "download_integrity": "PASS",
        "hf_sha256_verified_slides": len(rows),
        "full_feature_values_finite_verified_slides": len(rows),
        "h5_shapes_metadata_and_coordinates": "PASS",
        "all_original_slides_covered": "PASS" if not missing else "INCOMPLETE",
        "all_original_patients_covered": "PASS" if source_patients == matched_patients else "INCOMPLETE",
        "uni_conch_exact_patch_identity": "PASS" if not missing and all(row["ordered_coords_equal"] and row["identical_patch_size_level0"] for row in matched) else "FAIL",
        "ctf2_patch_text_alignment": "NOT_ESTABLISHED",
        "failures": [],
    }
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "tree_metadata_sha256": sha256(args.tree_json),
        "uni_inventory_sha256": sha256(args.uni_inventory),
        "dataset_revision": snapshot.name,
        "python_version": platform.python_version(),
        "h5py_version": h5py.__version__,
        "numpy_version": np.__version__,
        "scope": "offline download integrity and slide/patient/patch correspondence; no outcome labels or training",
    }
    write_csv(args.output_dir / "download_manifest.csv", rows)
    write_csv(args.output_dir / "cohort_manifest.csv", cohort)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "validation.json", validation)
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    write_csv(delivery / "cohort_manifest.csv", cohort)
    write_json(delivery / "summary.json", summary)
    write_json(delivery / "validation.json", validation)
    (delivery / "README.md").write_text(
        "# CONCH v1.5 cohort features\n\n"
        f"{len(matched)} exact slide matches for the original {len(source_patients)}-patient cohort. "
        "Read cohort_manifest.csv: the original DX1 slide for TCGA-QG-A5Z1 is missing; "
        "that patient has an already matched DX2 slide. No slide was renamed or replaced.\n\n"
        "Features are 768-dimensional, 512-pixel patches at 20X. "
        "UNI2 uses different 256-pixel patches; rows must not be paired directly. "
        "Native CONCH v1.5 patch features are not proven compatible with CONCH v1 text embeddings.\n\n"
        "Files were staged on the same E: drive by hard link where supported, "
        "otherwise by move (exFAT); no duplicate feature payload. "
        "The HF snapshot may no longer contain every staged file; use this delivery directory. "
        "Treat feature files as read-only. No feature files were uploaded to the server.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

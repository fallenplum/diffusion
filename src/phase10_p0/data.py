"""Strict manifest and HDF5 ingestion for Phase10-P0 CONCH v1.5 patches."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Slide:
    filename: str
    patient_id: str
    project: str
    path: Path
    features: np.ndarray
    coords: np.ndarray
    patch_size_level0: int
    patch_size: int
    magnification: int


def read_slide(path: Path, expected_sha: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if expected_sha and digest.lower() != str(expected_sha).lower():
        raise ValueError(f"SHA256 mismatch: {path.name}")
    with h5py.File(path, "r") as handle:
        if not {"features", "coords"}.issubset(handle.keys()):
            raise ValueError(f"missing features/coords datasets: {path.name}")
        feature_ds, coord_ds = handle["features"], handle["coords"]
        x = np.asarray(feature_ds, dtype=np.float32).reshape(-1, feature_ds.shape[-1])
        coords = np.asarray(coord_ds).reshape(-1, 2)
        feature_attrs = {str(k): str(v) for k, v in feature_ds.attrs.items()}
        coord_attrs = {str(k): str(v) for k, v in coord_ds.attrs.items()}
        if x.ndim != 2 or x.shape[1] != 768 or len(x) < 1:
            raise ValueError(f"expected CONCH v1.5 features [N,768], got {x.shape}: {path.name}")
        if coords.shape != (len(x), 2) or not np.issubdtype(coords.dtype, np.integer):
            raise ValueError(f"feature/coordinate length or dtype mismatch: {path.name}")
        if not np.isfinite(x).all() or (coords < 0).any() or len(np.unique(coords, axis=0)) != len(coords):
            raise ValueError(f"nonfinite features or invalid coordinates: {path.name}")
        if feature_attrs.get("encoder") != "conch_v15":
            raise ValueError(f"feature encoder metadata is not conch_v15: {path.name}")
        patch_size = int(coord_attrs["patch_size"])
        level0_size = int(coord_attrs["patch_size_level0"])
        magnification = int(coord_attrs["target_magnification"])
        if patch_size != 512 or magnification != 20 or level0_size <= 0:
            raise ValueError(f"unexpected CONCH scale metadata: {path.name}")
        grid = (coords - coords.min(axis=0)) // level0_size
        if len(np.unique(grid, axis=0)) != len(coords):
            raise ValueError(f"duplicate level-0 grid cells: {path.name}")
    return x, coords.astype(np.int64, copy=False), {
        "sha256": digest, "n_patches": int(len(x)), "feature_dim": int(x.shape[1]),
        "feature_attrs": feature_attrs, "coord_attrs": coord_attrs,
        "patch_size_level0": level0_size, "patch_size": patch_size,
        "target_magnification": magnification,
    }


def load_cohort(manifest_path: Path, feature_root: Path, expected_patient_ids: list[str] | np.ndarray):
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    matched = manifest.loc[manifest.status.eq("EXACT_SLIDE_MATCH")].copy()
    expected = set(map(str, expected_patient_ids))
    if set(matched.patient_id) != expected:
        raise ValueError("CONCH manifest patient set does not exactly match frozen survival cohort")
    if matched.filename.duplicated().any() or matched.patient_id.isna().any():
        raise ValueError("duplicate slide filename or empty patient mapping")
    root = Path(feature_root)
    slides: dict[str, Slide] = {}
    missing = []
    rows = []
    for row in matched.sort_values(["patient_id", "filename"]).itertuples(index=False):
        path = root / row.delivery_relative_path
        if not path.is_file():
            missing.append({"filename": row.filename, "relative_path": row.delivery_relative_path,
                            "expected_sha256": row.conch_sha256, "expected_bytes": int(row.conch_size_bytes)})
            continue
        x, coords, meta = read_slide(path, row.conch_sha256)
        slides[row.filename] = Slide(row.filename, row.patient_id, row.project, path, x, coords,
                                     meta["patch_size_level0"], meta["patch_size"], meta["target_magnification"])
        rows.append({"patient_id": row.patient_id, "filename": row.filename, "n_patches": meta["n_patches"],
                     "sha256": meta["sha256"], "patch_size_level0": meta["patch_size_level0"],
                     "patch_size": meta["patch_size"], "magnification": meta["target_magnification"]})
    if missing:
        return None, {"status": "BLOCKED", "reason": "missing CONCH feature files", "expected_slides": int(len(matched)),
                      "available_slides": int(len(slides)), "patients_expected": int(len(expected)),
                      "missing": missing, "validated_slides": rows}
    patients = {pid: [] for pid in sorted(expected)}
    for slide in slides.values():
        patients[slide.patient_id].append(slide)
    if any(not patient_slides for patient_slides in patients.values()):
        raise ValueError("one or more patients has no matched CONCH slide")
    return patients, {"status": "PASS", "patients": len(patients), "slides": len(slides),
                      "patches": int(sum(len(s.features) for s in slides.values())),
                      "validated_slides": rows, "missing": []}

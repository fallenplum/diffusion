"""Small, dependency-light helpers shared by automation and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROTECTED_ROOTS = {"tcga", "data"}
PROTECTED_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".parquet",
    ".feather",
    ".tar",
    ".gz",
    ".zip",
}
PROTECTED_NAMES = {
    "patient_manifest_used.csv",
    "slide_manifest_used.csv",
    "coords_audit.csv",
    "oof_predictions.csv",
}


def protected_path_violations(paths: Iterable[str | Path]) -> list[str]:
    """Return repository-relative paths that must not be tracked.

    This is intentionally filename-only: it never opens patient data or model
    binaries.  The guard is therefore safe to run in CI and on a self-hosted
    runner without copying sensitive content into logs.
    """
    violations: list[str] = []
    for raw_path in paths:
        path = str(raw_path).replace("\\", "/").lstrip("./")
        parts = [part for part in path.lower().split("/") if part]
        if not parts:
            continue
        if parts[0] in PROTECTED_ROOTS:
            violations.append(path)
            continue
        name = parts[-1]
        suffix = Path(name).suffix.lower()
        if suffix in PROTECTED_SUFFIXES or name in PROTECTED_NAMES:
            violations.append(path)
    return sorted(set(violations))


def validate_patient_level_split(slides, split) -> bool:
    """Validate that every slide follows exactly one patient-level fold."""
    for column in ("patient_id", "outer_fold", "split"):
        if column not in split.columns:
            raise ValueError(f"split is missing required column: {column}")
    if "patient_id" not in slides.columns:
        raise ValueError("slides is missing required column: patient_id")
    if split["patient_id"].duplicated().any():
        raise ValueError("a patient appears more than once in the split table")
    if not bool(split["split"].isin(["train", "test"]).all()):
        raise ValueError("split column must contain only train or test")
    mapping = dict(zip(split["patient_id"].astype(str), split["outer_fold"].astype(int)))
    slide_ids = slides["patient_id"].astype(str)
    missing = sorted(set(slide_ids) - set(mapping))
    if missing:
        raise ValueError(f"slides contain patients absent from split: {missing[:3]}")
    fold_counts = slides.assign(_fold=slide_ids.map(mapping)).groupby("patient_id")[
        "_fold"
    ].nunique()
    if (fold_counts > 1).any():
        raise ValueError("a patient has slides assigned to multiple outer folds")
    if "outer_fold" in slides.columns:
        expected_folds = slide_ids.map(mapping).astype(int)
        observed_folds = slides["outer_fold"].astype(int)
        if not bool((expected_folds == observed_folds).all()):
            raise ValueError("a slide outer_fold disagrees with its patient fold")
    train_ids = set(split.loc[split["split"] == "train", "patient_id"].astype(str))
    test_ids = set(split.loc[split["split"] == "test", "patient_id"].astype(str))
    if train_ids & test_ids:
        raise ValueError("train and test patients overlap")
    if train_ids | test_ids != set(mapping):
        raise ValueError("each patient must be assigned to train or test exactly once")
    return True


def validate_oof_predictions(predictions, expected_patient_ids: Iterable[str]) -> bool:
    """Validate one and only one out-of-fold prediction per patient."""
    if "patient_id" not in predictions.columns:
        raise ValueError("predictions is missing patient_id")
    ids = predictions["patient_id"].astype(str)
    expected = {str(pid) for pid in expected_patient_ids}
    if ids.duplicated().any():
        raise ValueError("OOF predictions contain duplicate patients")
    actual = set(ids)
    if actual != expected:
        raise ValueError(f"OOF patient set mismatch: missing={sorted(expected - actual)[:3]}, extra={sorted(actual - expected)[:3]}")
    return True

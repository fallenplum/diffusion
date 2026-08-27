from __future__ import annotations

import argparse
import io
import re
import tarfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def patient_id(value: object) -> str:
    match = PATIENT_RE.match(str(value).strip().upper())
    if not match:
        raise ValueError(f"Cannot parse TCGA patient ID from {value!r}")
    return match.group(1)


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def read_features(blob: bytes) -> np.ndarray:
    with h5py.File(io.BytesIO(blob), "r") as handle:
        if "features" not in handle:
            raise KeyError(f"Missing 'features'; keys={list(handle.keys())}")
        features = np.asarray(handle["features"])
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Invalid feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity")
    return features.astype(np.float16, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream UNI2-h H5 archives and export float16 patch bags."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.root.resolve()

    patient_npz = root / "outputs" / "features" / "uni2_h_patient_mean.npz"
    with np.load(patient_npz, allow_pickle=False) as data:
        eligible = {str(x) for x in data["patient_ids"]}

    archives = sorted((root / "TCGA").glob("TCGA-*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No TCGA-*.tar.gz under {root / 'TCGA'}")

    bag_root = root / "data" / "uni2_h_bags"
    audit_root = root / "outputs" / "audit"
    bag_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    expected_dim: int | None = None
    selected = 0
    scanned = 0
    started = time.time()

    for archive in archives:
        print(f"[archive] {archive.name}", flush=True)
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".h5"):
                    continue
                scanned += 1
                slide_name = Path(member.name).name
                try:
                    pid = patient_id(slide_name)
                except ValueError:
                    continue
                if pid not in eligible:
                    continue

                patient_dir = bag_root / pid
                patient_dir.mkdir(parents=True, exist_ok=True)
                output_path = patient_dir / f"{safe_stem(slide_name)}.npy"
                if output_path.exists():
                    cached = np.load(output_path, mmap_mode="r", allow_pickle=False)
                    if cached.ndim != 2 or cached.shape[0] == 0:
                        raise ValueError(f"Invalid cached bag: {output_path}")
                    current_dim = int(cached.shape[1])
                    if expected_dim is None:
                        expected_dim = current_dim
                    elif current_dim != expected_dim:
                        raise ValueError(
                            f"Cached feature dimension mismatch: {output_path}"
                        )
                    rows.append(
                        {
                            "patient_id": pid,
                            "slide_name": slide_name,
                            "bag_path": output_path.relative_to(root).as_posix(),
                            "n_patches": int(cached.shape[0]),
                            "feature_dim": current_dim,
                            "dtype": str(cached.dtype),
                        }
                    )
                    selected += 1
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    print(f"[warning] Cannot extract {member.name}", flush=True)
                    continue
                try:
                    features = read_features(extracted.read())
                except Exception as exc:
                    print(f"[warning] Skip {member.name}: {exc}", flush=True)
                    continue

                if expected_dim is None:
                    expected_dim = int(features.shape[1])
                elif features.shape[1] != expected_dim:
                    raise ValueError(
                        f"Feature dimension mismatch for {member.name}: "
                        f"{features.shape[1]} != {expected_dim}"
                    )

                np.save(output_path, features, allow_pickle=False)
                rows.append(
                    {
                        "patient_id": pid,
                        "slide_name": slide_name,
                        "bag_path": output_path.relative_to(root).as_posix(),
                        "n_patches": int(features.shape[0]),
                        "feature_dim": int(features.shape[1]),
                        "dtype": str(features.dtype),
                    }
                )
                selected += 1
                if selected % 25 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[progress] selected={selected}, scanned={scanned}, "
                        f"elapsed={elapsed / 60:.1f} min",
                        flush=True,
                    )

    manifest = pd.DataFrame(rows).sort_values(["patient_id", "slide_name"])
    manifest_path = audit_root / "uni2_bag_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    observed_patients = set(manifest["patient_id"])
    missing = sorted(eligible - observed_patients)
    print(
        f"[done] slides={len(manifest)}, patients={len(observed_patients)}, "
        f"missing_patients={len(missing)}, dim={expected_dim}, "
        f"elapsed={(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    if missing:
        raise RuntimeError(f"Missing eligible patients after export: {missing[:10]}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
SEED = 20260723


def patient_id(value: object) -> str:
    match = PATIENT_RE.match(str(value).strip().upper())
    if not match:
        raise ValueError(f"Cannot parse TCGA patient id: {value!r}")
    return match.group(1)


def finite_dataset(dataset: h5py.Dataset, chunk_rows: int = 2048) -> bool:
    if dataset.ndim == 3 and dataset.shape[0] == 1:
        n = dataset.shape[1]
        return all(
            np.isfinite(dataset[:, start : start + chunk_rows]).all()
            for start in range(0, n, chunk_rows)
        )
    if dataset.ndim == 2:
        n = dataset.shape[0]
        return all(
            np.isfinite(dataset[start : start + chunk_rows]).all()
            for start in range(0, n, chunk_rows)
        )
    return bool(np.isfinite(np.asarray(dataset)).all())


def squeeze_features_shape(shape: tuple[int, ...]) -> tuple[int, int] | None:
    if len(shape) == 3 and shape[0] == 1:
        return int(shape[1]), int(shape[2])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None


def squeeze_coords(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Invalid coords shape: {array.shape}")
    return np.asarray(array, dtype=np.int64)


def resolve_manifest(root: Path) -> pd.DataFrame:
    path = root / "outputs" / "audit" / "uni2_patient_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Historical patient manifest missing: {path}")
    frame = pd.read_csv(path)
    needed = {"patient_id", "has_uni2", "dss_status", "dss_months"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"Patient manifest missing columns: {missing}")
    frame["patient_id"] = frame["patient_id"].astype(str).str.upper()
    usable = frame.loc[frame["has_uni2"].astype(bool)].copy()
    usable["dss_status"] = pd.to_numeric(usable["dss_status"], errors="raise").astype(int)
    usable["dss_months"] = pd.to_numeric(usable["dss_months"], errors="raise")
    if not usable["dss_status"].isin([0, 1]).all():
        raise ValueError("DSS status outside {0,1}")
    if not (np.isfinite(usable["dss_months"]) & (usable["dss_months"] > 0)).all():
        raise ValueError("DSS time is missing, non-finite, or non-positive")
    if usable["patient_id"].duplicated().any():
        raise ValueError("Duplicated patient ids in historical manifest")
    return usable.sort_values("patient_id").reset_index(drop=True)


def scan_tcga(root: Path, patients: set[str], output: Path) -> tuple[pd.DataFrame, dict]:
    tcga = root / "TCGA"
    npy_root = tcga / "uni2_h_bags"
    h5_paths = sorted(tcga.glob("TCGA-*/*.h5"))
    npy_paths = sorted(npy_root.glob("*/*.npy"))
    if not h5_paths or not npy_paths:
        raise FileNotFoundError("Expected H5 and NPY bags under TCGA")

    h5_by_name = defaultdict(list)
    for path in h5_paths:
        h5_by_name[path.name].append(path)
    npy_by_name = defaultdict(list)
    for path in npy_paths:
        npy_by_name[path.stem + ".h5"].append(path)

    coords_cache = output / "cache" / "coords"
    coords_cache.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    stats = Counter()
    selected_h5 = {name for name in npy_by_name if patient_id(name) in patients}

    for idx, name in enumerate(sorted(selected_h5), start=1):
        pid = patient_id(name)
        h5_matches = h5_by_name.get(name, [])
        npy_matches = npy_by_name.get(name, [])
        if len(h5_matches) != 1 or len(npy_matches) != 1:
            rows.append({
                "patient_id": pid,
                "slide_id": Path(name).stem,
                "h5_matches": len(h5_matches),
                "npy_matches": len(npy_matches),
                "valid": False,
                "error": "non-unique or missing H5/NPY",
            })
            stats["invalid_match"] += 1
            continue
        h5_path, npy_path = h5_matches[0], npy_matches[0]
        row: dict[str, object] = {
            "patient_id": pid,
            "slide_id": Path(name).stem,
            "cancer_site": h5_path.parent.name,
            "h5_path": h5_path.relative_to(root).as_posix(),
            "bag_path": npy_path.relative_to(root).as_posix(),
            "h5_matches": 1,
            "npy_matches": 1,
            "valid": False,
            "error": "",
        }
        try:
            bag = np.load(npy_path, mmap_mode="r", allow_pickle=False)
            row.update({
                "n_patches": int(bag.shape[0]) if bag.ndim >= 1 else 0,
                "feature_dim": int(bag.shape[1]) if bag.ndim == 2 else -1,
                "bag_dtype": str(bag.dtype),
                "bag_has_nan_inf": not finite_dataset_like_array(bag),
                "empty_bag": bool(bag.ndim != 2 or bag.shape[0] == 0),
            })
            with h5py.File(h5_path, "r") as handle:
                keys = sorted(handle.keys())
                if "features" not in handle:
                    raise KeyError(f"features absent; keys={keys}")
                fshape = squeeze_features_shape(handle["features"].shape)
                if fshape is None:
                    raise ValueError(f"invalid H5 feature shape {handle['features'].shape}")
                coords_key = "coords" if "coords" in handle else (
                    "coords_patching" if "coords_patching" in handle else ""
                )
                coords = squeeze_coords(np.asarray(handle[coords_key])) if coords_key else None
                h5_finite = finite_dataset(handle["features"])
                row.update({
                    "h5_keys": "|".join(keys),
                    "h5_feature_shape": "x".join(map(str, handle["features"].shape)),
                    "h5_feature_dtype": str(handle["features"].dtype),
                    "h5_has_nan_inf": not h5_finite,
                    "coords_available": coords is not None,
                    "coords_key": coords_key,
                    "coords_shape": "" if coords is None else "x".join(map(str, coords.shape)),
                    "coords_dtype": "" if coords is None else str(coords.dtype),
                    "coords_aligned": bool(coords is not None and len(coords) == fshape[0] == len(bag)),
                    "coords_unique": 0 if coords is None else int(len(np.unique(coords, axis=0))),
                })
                if bag.ndim != 2 or bag.shape != fshape:
                    raise ValueError(f"NPY/H5 shape mismatch: {bag.shape} vs {fshape}")
                if not h5_finite or row["bag_has_nan_inf"]:
                    raise ValueError("NaN/Inf in features")
                if coords is None or len(coords) != len(bag):
                    raise ValueError("coords absent or misaligned")
                coord_path = coords_cache / pid / f"{Path(name).stem}.npy"
                coord_path.parent.mkdir(parents=True, exist_ok=True)
                if not coord_path.exists():
                    np.save(coord_path, coords, allow_pickle=False)
                else:
                    cached = np.load(coord_path, mmap_mode="r", allow_pickle=False)
                    if cached.shape != coords.shape or not np.array_equal(cached, coords):
                        raise ValueError("cached coords mismatch")
                row["coords_path"] = coord_path.relative_to(root).as_posix()
                row["valid"] = True
                stats["valid_slides"] += 1
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            stats["invalid_slides"] += 1
        rows.append(row)
        if idx % 25 == 0:
            print(f"[audit] {idx}/{len(selected_h5)} slides", flush=True)

    frame = pd.DataFrame(rows).sort_values(["patient_id", "slide_id"]).reset_index(drop=True)
    raw_patient_counts = Counter(patient_id(p.name) for p in h5_paths)
    stats.update({
        "raw_h5_total": len(h5_paths),
        "raw_h5_patients": len(raw_patient_counts),
        "tcga_npy_total": len(npy_paths),
        "selected_patients": int(frame.loc[frame["valid"], "patient_id"].nunique()),
        "multi_slide_patients": int((frame.loc[frame["valid"]].groupby("patient_id").size() > 1).sum()),
    })
    return frame, dict(stats)


def finite_dataset_like_array(array: np.ndarray, chunk_rows: int = 2048) -> bool:
    if array.ndim != 2:
        return False
    return all(np.isfinite(array[start : start + chunk_rows]).all() for start in range(0, len(array), chunk_rows))


def make_splits(patient_frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    events = patient_frame["dss_status"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.zeros(len(patient_frame), dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(events)), events), start=1):
        folds[test_idx] = fold
    result = patient_frame[["patient_id", "dss_status", "dss_months"]].copy()
    result["outer_fold"] = folds
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, 6):
        frame = result[["patient_id", "dss_status", "dss_months"]].copy()
        frame.insert(1, "split", np.where(result["outer_fold"] == fold, "test", "train"))
        frame.to_csv(split_dir / f"outer_fold_{fold}.csv", index=False, encoding="utf-8-sig")
    return result


def save_run_state(output: Path) -> None:
    path = output / "run_state.json"
    if path.exists():
        return
    jobs = [f"B0/fold_{fold}/seed_{SEED}" for fold in range(1, 6)]
    payload = {
        "stage": "stage_0_audit",
        "fold": 0,
        "model": "",
        "seed": 0,
        "status": "RUNNING",
        "completed_jobs": [],
        "pending_jobs": jobs,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "outputs" / "pathology_context_phase1"
    audit_dir = output / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    save_run_state(output)

    patients = resolve_manifest(root)
    slides, stats = scan_tcga(root, set(patients["patient_id"]), output)
    valid = slides.loc[slides["valid"]].copy()
    observed = set(valid["patient_id"])
    expected = set(patients["patient_id"])
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Patient mismatch missing={missing[:10]} extra={extra[:10]}")

    counts = valid.groupby("patient_id").agg(
        n_slides=("slide_id", "size"), n_patches=("n_patches", "sum")
    )
    used = patients.drop(columns=[c for c in ["has_uni2", "n_slides", "n_patches"] if c in patients]).merge(
        counts, left_on="patient_id", right_index=True, how="inner"
    )
    split_audit = make_splits(used, output)
    used = used.merge(split_audit[["patient_id", "outer_fold"]], on="patient_id", how="left")
    used.to_csv(audit_dir / "patient_manifest_used.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(audit_dir / "slide_manifest_used.csv", index=False, encoding="utf-8-sig")
    coord_cols = [
        "patient_id", "slide_id", "n_patches", "coords_available", "coords_key",
        "coords_shape", "coords_dtype", "coords_aligned", "coords_unique", "coords_path",
    ]
    valid[coord_cols].to_csv(audit_dir / "coords_audit.csv", index=False, encoding="utf-8-sig")
    split_rows = []
    for fold in range(1, 6):
        test = split_audit["outer_fold"] == fold
        split_rows.append({
            "outer_fold": fold,
            "n_train": int((~test).sum()),
            "n_test": int(test.sum()),
            "events_train": int(split_audit.loc[~test, "dss_status"].sum()),
            "events_test": int(split_audit.loc[test, "dss_status"].sum()),
            "patient_overlap": 0,
            "slide_overlap": 0,
            "seed": SEED,
            "split_logic": "StratifiedKFold(n_splits=5,shuffle=True,random_state=20260723)",
        })
    pd.DataFrame(split_rows).to_csv(audit_dir / "split_audit.csv", index=False, encoding="utf-8-sig")

    patch_counts = valid["n_patches"].to_numpy(dtype=int)
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "input_scope": "TCGA only; root/data excluded",
        "label_source": "outputs/audit/uni2_patient_manifest.csv (historical audit; labels originally from cached clinical metadata)",
        "n_patients": int(len(used)),
        "n_events": int(used["dss_status"].sum()),
        "n_slides": int(len(valid)),
        "feature_dim_values": sorted(valid["feature_dim"].astype(int).unique().tolist()),
        "bag_dtype_values": sorted(valid["bag_dtype"].astype(str).unique().tolist()),
        "patch_count": {
            "min": int(patch_counts.min()), "median": float(np.median(patch_counts)),
            "max": int(patch_counts.max()), "total": int(patch_counts.sum()),
        },
        "coords": {
            "available_all": bool(valid["coords_available"].all()),
            "aligned_all": bool(valid["coords_aligned"].all()),
            "source": "raw TCGA H5 coords dataset",
            "fabricated": False,
        },
        "multi_slide_patients": int((valid.groupby("patient_id").size() > 1).sum()),
        "empty_bags": int(valid["empty_bag"].sum()),
        "bags_with_nan_inf": int(valid["bag_has_nan_inf"].sum()),
        "h5_with_nan_inf": int(valid["h5_has_nan_inf"].sum()),
        "scan_stats": stats,
        "split": {"seed": SEED, "outer_folds": 5, "exact_historical_logic_reproduced": True},
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    (audit_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"""# Phase P1 Stage 0 数据与泄漏审计

## 结论

- 实际纳入 **{len(used)}** 位患者、**{int(used['dss_status'].sum())}** 个 DSS events、**{len(valid)}** 张切片。
- UNI2-h 特征维数为 **{sorted(valid['feature_dim'].astype(int).unique().tolist())}**，patch 数范围 **{patch_counts.min()}–{patch_counts.max()}**，中位数 **{np.median(patch_counts):.0f}**。
- 所有纳入 H5 均包含真实 `coords`，与 feature patch 数逐一对齐；坐标来自原始 TCGA H5，未伪造。
- 所有纳入 NPY/H5 feature 均通过 NaN/Inf 检查，无空 bag。
- 患者级 5 折使用历史代码同一 `StratifiedKFold` 逻辑和 seed `{SEED}`，可精确复现旧划分生成过程。
- 本阶段输入仅来自 `TCGA/`；根目录 `data/` 未作为特征输入。

## 风险与限制

- DSS 事件仅 {int(used['dss_status'].sum())}/{len(used)}，统计功效有限，必须同时报告 Uno C 与配对 bootstrap。
- 8GB GPU 下 1024-token Transformer 需 AMP、梯度累积和逐切片 lazy loading。
- 历史患者标签沿用已审计 manifest；其原始来源是缓存 clinical metadata，而不是本阶段无关的 `data/` 数据集。
"""
    (audit_dir / "data_audit.md").write_text(md, encoding="utf-8")
    (audit_dir / "split_audit.md").write_text(
        "# Split audit\n\n5-fold patient-level split exactly follows the historical baseline code: "
        "`StratifiedKFold(n_splits=5, shuffle=True, random_state=20260723)`. "
        "All slides inherit their patient's fold; train/test patient and slide overlap are zero.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

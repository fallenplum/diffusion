from __future__ import annotations

import argparse
import io
import re
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def patient_id(value: object) -> str:
    text = str(value).strip().upper()
    match = PATIENT_RE.match(text)
    if not match:
        raise ValueError(f"无法从以下值解析TCGA患者ID: {value!r}")
    return match.group(1)


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"期望 {directory / pattern} 唯一匹配，实际找到 {len(matches)} 个: {matches}"
        )
    return matches[0]


def load_cohort(root: Path) -> tuple[pd.DataFrame, set[str]]:
    data_dir = root / "data"
    metadata_dir = root / ".cache" / "metadata"
    clinical_path = find_one(metadata_dir, "merged_clinical_data_preprocess*.csv")
    rna_path = find_one(data_dir, "merged_rna_data_2_preprocess_*.csv")
    cna_path = find_one(data_dir, "merged_cna_data_preprocess_*.csv")

    clinical = pd.read_csv(clinical_path, low_memory=False)
    clinical["patient_id"] = clinical["PATIENT_ID"].map(patient_id)

    rna_ids = pd.read_csv(rna_path, usecols=[0]).iloc[:, 0].map(patient_id)
    cna_ids = pd.read_csv(cna_path, usecols=[0]).iloc[:, 0].map(patient_id)

    common = set(clinical["patient_id"]) & set(rna_ids) & set(cna_ids)
    clinical = clinical.loc[clinical["patient_id"].isin(common)].copy()
    clinical["DSS_STATUS"] = pd.to_numeric(clinical["DSS_STATUS"], errors="coerce")
    clinical["DSS_MONTHS"] = pd.to_numeric(clinical["DSS_MONTHS"], errors="coerce")
    clinical["valid_dss"] = (
        clinical["DSS_STATUS"].isin([0, 1])
        & clinical["DSS_MONTHS"].notna()
        & (clinical["DSS_MONTHS"] > 0)
    )
    eligible = set(clinical.loc[clinical["valid_dss"], "patient_id"])
    return clinical, eligible


def read_h5_feature(blob: bytes) -> tuple[np.ndarray, int, int]:
    with h5py.File(io.BytesIO(blob), "r") as handle:
        if "features" not in handle:
            raise KeyError(f"H5缺少features，现有键: {list(handle.keys())}")
        features = np.asarray(handle["features"])
        if features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        if features.ndim != 2:
            raise ValueError(f"异常features形状: {features.shape}")
        if features.shape[0] == 0:
            raise ValueError("features没有patch")
        mean_feature = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        return mean_feature, int(features.shape[0]), int(features.shape[1])


def stream_archives(
    archives: list[Path], eligible: set[str]
) -> tuple[list[str], list[str], list[np.ndarray], list[int], Counter]:
    slide_names: list[str] = []
    patient_ids: list[str] = []
    slide_features: list[np.ndarray] = []
    patch_counts: list[int] = []
    stats: Counter = Counter()
    feature_dim: int | None = None

    started = time.time()
    for archive in archives:
        print(f"\n[读取] {archive.name}", flush=True)
        archive_started = time.time()
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".h5"):
                    continue
                stats["h5_total"] += 1
                try:
                    pid = patient_id(Path(member.name).name)
                except ValueError:
                    stats["unparsed_name"] += 1
                    continue
                if pid not in eligible:
                    stats["not_eligible"] += 1
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    stats["extract_failed"] += 1
                    continue
                try:
                    blob = extracted.read()
                    feature, n_patches, current_dim = read_h5_feature(blob)
                except Exception as exc:
                    stats["invalid_h5"] += 1
                    print(f"[警告] 跳过 {member.name}: {exc}", flush=True)
                    continue
                if feature_dim is None:
                    feature_dim = current_dim
                elif current_dim != feature_dim:
                    raise ValueError(
                        f"特征维数不一致: 期望 {feature_dim}, 得到 {current_dim}, "
                        f"文件 {member.name}"
                    )
                slide_names.append(Path(member.name).name)
                patient_ids.append(pid)
                slide_features.append(feature)
                patch_counts.append(n_patches)
                stats["selected_h5"] += 1
                if stats["selected_h5"] % 50 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[进度] 已处理 {stats['selected_h5']} 张目标切片，"
                        f"扫描 {stats['h5_total']} 张，耗时 {elapsed / 60:.1f} 分钟",
                        flush=True,
                    )
        print(
            f"[完成] {archive.name}: {time.time() - archive_started:.1f} 秒",
            flush=True,
        )
    return slide_names, patient_ids, slide_features, patch_counts, stats


def save_outputs(
    root: Path,
    clinical: pd.DataFrame,
    eligible: set[str],
    slide_names: list[str],
    patient_ids: list[str],
    slide_features: list[np.ndarray],
    patch_counts: list[int],
) -> None:
    output_features = root / "outputs" / "features"
    output_audit = root / "outputs" / "audit"
    output_features.mkdir(parents=True, exist_ok=True)
    output_audit.mkdir(parents=True, exist_ok=True)

    feature_matrix = np.stack(slide_features).astype(np.float32)
    np.savez_compressed(
        output_features / "uni2_h_slide_mean.npz",
        features=feature_matrix,
        slide_names=np.asarray(slide_names),
        patient_ids=np.asarray(patient_ids),
        patch_counts=np.asarray(patch_counts, dtype=np.int32),
    )

    rows_by_patient: dict[str, list[int]] = defaultdict(list)
    for index, pid in enumerate(patient_ids):
        rows_by_patient[pid].append(index)

    clinical_index = clinical.set_index("patient_id")
    manifest_rows = []
    patient_features = []
    patient_order = []
    for pid in sorted(eligible):
        indices = rows_by_patient.get(pid, [])
        row = clinical_index.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        manifest_rows.append(
            {
                "patient_id": pid,
                "has_uni2": bool(indices),
                "n_slides": len(indices),
                "n_patches": int(sum(patch_counts[i] for i in indices)),
                "dss_status": int(row["DSS_STATUS"]),
                "dss_months": float(row["DSS_MONTHS"]),
                "tissue_source_site": pid.split("-")[1],
            }
        )
        if indices:
            patient_order.append(pid)
            patient_features.append(feature_matrix[indices].mean(axis=0))

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(
        output_audit / "uni2_patient_manifest.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        output_features / "uni2_h_patient_mean.npz",
        features=np.stack(patient_features).astype(np.float32),
        patient_ids=np.asarray(patient_order),
    )

    usable = manifest.loc[manifest["has_uni2"]]
    print("\n[患者级汇总]", flush=True)
    print(f"符合三张表且DSS时间>0的患者: {len(manifest)}", flush=True)
    print(f"同时具有UNI2-h特征的患者: {len(usable)}", flush=True)
    print(f"DSS事件数: {int(usable['dss_status'].sum())}", flush=True)
    print(
        f"每患者切片数: 中位数={usable['n_slides'].median():.1f}, "
        f"最大={usable['n_slides'].max()}",
        flush=True,
    )
    print(
        f"每切片patch数: 中位数={np.median(patch_counts):.0f}, "
        f"范围=[{min(patch_counts)}, {max(patch_counts)}]",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    clinical, eligible = load_cohort(root)
    archives = sorted((root / "TCGA").glob("TCGA-*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"{root / 'TCGA'} 中没有TCGA-*.tar.gz")

    print(f"待扫描压缩包: {[x.name for x in archives]}", flush=True)
    print(f"目标患者数: {len(eligible)}", flush=True)
    results = stream_archives(archives, eligible)
    slide_names, patient_ids, features, patch_counts, stats = results
    if not features:
        raise RuntimeError("没有提取到任何UNI2-h特征")
    save_outputs(
        root,
        clinical,
        eligible,
        slide_names,
        patient_ids,
        features,
        patch_counts,
    )
    print(f"\n[扫描统计] {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()

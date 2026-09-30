from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
SAMPLE_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})", re.IGNORECASE)
CLINICAL_CATEGORICAL = (
    "SEX",
    "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "PATH_T_STAGE",
    "PATH_N_STAGE",
    "PATH_M_STAGE",
)


@dataclass
class RawInputs:
    patient_ids: np.ndarray
    outer_folds: np.ndarray
    dss_time: np.ndarray
    dss_event: np.ndarray
    cancer_type: np.ndarray
    rna: np.ndarray
    cna: np.ndarray
    clinical: pd.DataFrame
    h5_files: dict[str, list[Path]]
    audit: dict[str, Any]


@dataclass
class FoldArrays:
    path_tokens: np.ndarray
    rna: np.ndarray
    cna: np.ndarray
    clinical: np.ndarray
    audit: dict[str, Any]


def patient_id(value: object) -> str | None:
    match = PATIENT_RE.match(str(value).strip().upper())
    return match.group(1) if match else None


def _table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False, dtype=str)


def _status(value: object) -> float:
    match = re.match(r"\s*([01])", str(value))
    return float(match.group(1)) if match else float("nan")


def _sample_priority(value: str) -> tuple[int, str]:
    match = SAMPLE_RE.match(value)
    if not match:
        return (99, value)
    order = {"01": 0, "02": 1, "06": 2, "11": 3}
    return (order.get(match.group(2), 10), value)


def _load_patient_table(path: Path, patient_ids: list[str]) -> pd.DataFrame:
    table = _table(path)
    if "PATIENT_ID" not in table.columns:
        raise ValueError(f"{path} has no PATIENT_ID column")
    table["patient_id"] = table["PATIENT_ID"].map(patient_id)
    table = table.dropna(subset=["patient_id"]).drop_duplicates("patient_id", keep="first")
    table = table.set_index("patient_id")
    return table.reindex(patient_ids)


def _load_omics_table(path: Path, patient_ids: list[str]) -> np.ndarray:
    table = _table(path)
    if table.shape[1] < 3:
        raise ValueError(f"{path} does not contain sample columns")
    sample_columns: dict[str, list[str]] = {}
    for column in table.columns[2:]:
        p = patient_id(column)
        if p is not None and p in patient_ids:
            sample_columns.setdefault(p, []).append(column)
    missing = [p for p in patient_ids if p not in sample_columns]
    if missing:
        raise ValueError(f"{path.name}: missing {len(missing)} patient samples; first={missing[:5]}")
    numeric = table.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    column_index = {name: i for i, name in enumerate(table.columns[2:])}
    output = np.empty((len(patient_ids), numeric.shape[0]), dtype=np.float32)
    for row, p in enumerate(patient_ids):
        columns = sorted(sample_columns[p], key=_sample_priority)
        # Primary tumour samples are preferred. Multiple primary samples are averaged
        # deterministically; this is fit-free sample selection and does not use labels.
        primary = [x for x in columns if _sample_priority(x)[0] == 0]
        selected = primary or columns
        output[row] = np.nanmean(numeric[:, [column_index[x] for x in selected]], axis=1).astype(np.float32)
    return output


def _merge_clinical(primary: pd.DataFrame, supplement: pd.DataFrame, patient_ids: list[str]) -> pd.DataFrame:
    columns = ["AGE", *CLINICAL_CATEGORICAL]
    result = pd.DataFrame(index=patient_ids)
    for column in columns:
        result[column] = primary[column] if column in primary.columns else np.nan
    supplement = supplement.reindex(patient_ids)
    # The M2R protocol deliberately excludes staging edition, histology and
    # primary-site fields.  Supplementary clinical data may fill only the
    # six declared protocol fields, never introduce extra covariates.
    for target in ["SEX", *CLINICAL_CATEGORICAL[1:]]:
        if target in supplement.columns:
            result[target] = result[target].where(result[target].notna(), supplement[target])
    return result.reset_index(names="patient_id")


def build_h5_index(data_root: Path, patient_ids: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {p: [] for p in patient_ids}
    for project in ("TCGA-COAD", "TCGA-READ"):
        directory = data_root / "TCGA" / project
        if not directory.exists():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.h5")):
            p = patient_id(path.name)
            if p in result:
                result[p].append(path)
    missing = [p for p, paths in result.items() if not paths]
    if missing:
        raise ValueError(f"Missing pathology H5 for {len(missing)} split patients: {missing[:5]}")
    return result


def load_raw_inputs(data_root: Path, repo_root: Path) -> RawInputs:
    split_path = repo_root / "outputs" / "phase_m1r" / "fixed_patient_folds.csv"
    split = pd.read_csv(split_path, dtype={"patient_id": str, "fold": int})
    required = {"patient_id", "fold"}
    if set(split.columns) != required:
        raise ValueError(f"fixed split columns must be {required}, got {set(split.columns)}")
    split["patient_id"] = split["patient_id"].map(patient_id)
    if split["patient_id"].isna().any() or split["patient_id"].duplicated().any() or len(split) != 512:
        raise ValueError("M2 requires the committed 512-patient fixed split")
    split = split.sort_values("patient_id").reset_index(drop=True)
    ids = split["patient_id"].tolist()

    base = data_root / "data" / "data_8" / "coadread_tcga_pan_can_atlas_2018"
    supplement_path = data_root / "data" / "data_8" / "coadread_tcga" / "data_clinical_patient.txt"
    primary = _load_patient_table(base / "data_clinical_patient.txt", ids)
    supplement = _load_patient_table(supplement_path, ids)
    clinical = _merge_clinical(primary, supplement, ids)
    labels = primary.reindex(ids)
    dss_time = pd.to_numeric(labels["DSS_MONTHS"], errors="coerce").to_numpy(dtype=float)
    dss_event = labels["DSS_STATUS"].map(_status).to_numpy(dtype=float)
    cancer = labels["CANCER_TYPE_ACRONYM"].fillna("UNKNOWN").astype(str).to_numpy()
    if not np.all(np.isfinite(dss_time)) or not np.all(dss_time > 0) or not np.all(np.isin(dss_event, [0, 1])):
        raise ValueError("DSS labels contain missing/non-positive/non-binary values")
    if int(np.sum(dss_event)) != 65:
        raise ValueError(f"M2 fixed split must contain 65 DSS events, got {int(np.sum(dss_event))}")

    rna = _load_omics_table(base / "data_mrna_seq_v2_rsem.txt", ids)
    cna = _load_omics_table(base / "data_log2_cna.txt", ids)
    h5_files = build_h5_index(data_root, ids)
    audit = {
        "n_patients": len(ids),
        "dss_events": int(np.sum(dss_event)),
        "fold_counts": {str(k): int(v) for k, v in split["fold"].value_counts().sort_index().items()},
        "h5_slides": int(sum(len(v) for v in h5_files.values())),
        "h5_unique_patients": int(sum(bool(v) for v in h5_files.values())),
        "rna_raw_dim": int(rna.shape[1]),
        "cna_raw_dim": int(cna.shape[1]),
        "rna_input_scale_audit": _scale_audit(rna, "RNA"),
        "cna_input_scale_audit": _scale_audit(cna, "CNA"),
        "clinical_columns": ["AGE", *CLINICAL_CATEGORICAL],
        "split_path": "outputs/phase_m1r/fixed_patient_folds.csv",
        "data_root_env": "DIFFUSION_DATA_ROOT",
    }
    return RawInputs(
        patient_ids=np.asarray(ids, dtype=str),
        outer_folds=split["fold"].to_numpy(dtype=int),
        dss_time=dss_time,
        dss_event=dss_event.astype(int),
        cancer_type=cancer,
        rna=rna,
        cna=cna,
        clinical=clinical,
        h5_files=h5_files,
        audit=audit,
    )


def _read_selected_rows(path: Path, indices: np.ndarray) -> np.ndarray:
    # h5py requires strictly increasing fancy-index arrays. Random sampling
    # uses replacement when a slide has fewer than K patches, so duplicates
    # are possible. Read unique sorted rows and restore the caller's order.
    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError(f"patch indices must be one-dimensional, got {requested.shape}")
    if requested.size == 0:
        return np.zeros((0, 1536), dtype=np.float32)
    order = np.argsort(requested, kind="stable")
    sorted_requested = requested[order]
    unique_requested, inverse = np.unique(sorted_requested, return_inverse=True)
    with h5py.File(path, "r") as handle:
        dataset = handle["features"]
        if dataset.ndim == 3:
            unique_values = np.asarray(dataset[0, unique_requested, :], dtype=np.float32)
        else:
            unique_values = np.asarray(dataset[unique_requested, :], dtype=np.float32)
    sorted_values = unique_values[inverse]
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def _sample_patient(files: list[Path], k: int, seed: int, deterministic: bool) -> np.ndarray:
    lengths: list[int] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            shape = handle["features"].shape
            lengths.append(int(shape[1] if len(shape) == 3 else shape[0]))
    total = sum(lengths)
    if deterministic:
        indices = np.linspace(0, total - 1, num=k, dtype=np.int64) if total >= k else np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=k, replace=total < k).astype(np.int64)
        indices.sort()
    chunks: list[np.ndarray] = []
    start = 0
    for path, length in zip(files, lengths):
        local = indices[(indices >= start) & (indices < start + length)] - start
        if len(local):
            order = np.argsort(local)
            values = _read_selected_rows(path, local[order])
            restored = np.empty_like(values)
            restored[order] = values
            chunks.append((local + start, restored))
        start += length
    if chunks:
        # Reassemble in the global sampled-index order, preserving slide order.
        all_indices = np.concatenate([x[0] for x in chunks])
        all_values = np.concatenate([x[1] for x in chunks], axis=0)
        order = np.argsort(all_indices, kind="stable")
        values = all_values[order]
    else:
        values = np.zeros((0, 1536), dtype=np.float32)
    if len(values) < k:
        rng = np.random.default_rng(seed + 17)
        extra = rng.choice(len(values), size=k - len(values), replace=True)
        values = np.concatenate([values, values[extra]], axis=0)
    return values[:k].astype(np.float16, copy=False)


def prepare_path_cache(
    raw: RawInputs,
    runtime_root: Path,
    k: int = 1024,
    train_seed: int = 20260904,
    pool_k: int = 2048,
) -> dict[str, Any]:
    """Prepare fixed test tokens and a compact train sampling pool.

    The old M2 cache sampled one training view once and replayed it for every
    epoch.  M2R stores a per-patient pool, then :class:`PathTokenProvider`
    chooses a deterministic subset from that pool as a function of
    ``(seed, epoch, patient)``.  This keeps reruns reproducible while ensuring
    that different epochs see different patch subsets without writing one
    multi-gigabyte array per epoch.
    """
    runtime_root.mkdir(parents=True, exist_ok=True)
    metadata_path = runtime_root / "path_tokens_meta.json"
    paths = {
        "deterministic": runtime_root / "path_tokens_deterministic.npy",
        "train_pool": runtime_root / "path_tokens_train_pool.npy",
    }
    shapes = {"deterministic": (len(raw.patient_ids), k, 1536), "train_pool": (len(raw.patient_ids), pool_k, 1536)}
    for mode, path in paths.items():
        shape = shapes[mode]
        if path.exists():
            mapped = np.load(path, mmap_mode="r")
            if tuple(mapped.shape) == shape:
                continue
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        for i, patient in enumerate(raw.patient_ids):
            seed = train_seed ^ int(hashlib.sha256(patient.encode()).hexdigest()[:8], 16)
            mapped[i] = _sample_patient(raw.h5_files[patient], shape[1], seed, mode == "deterministic")
        mapped.flush()
        del mapped
    payload = {
        "deterministic_shape": list(shapes["deterministic"]),
        "train_pool_shape": list(shapes["train_pool"]),
        "dtype": "float16",
        "k": k,
        "train_pool_k": pool_k,
        "feature_dim": 1536,
        "train_sampling": "deterministic_epochwise_without_replacement_from_patient_pool",
        "train_sampling_seed": train_seed,
        "test_sampling": "deterministic_evenly_spaced_per_patient",
        "patient_order_sha256": hashlib.sha256("\n".join(raw.patient_ids).encode()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _scale_audit(values: np.ndarray, modality: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"modality": modality, "finite": False}
    quantiles = np.quantile(finite, [0.01, 0.5, 0.99]).tolist()
    # RSEM/count-like matrices have a large positive dynamic range.  A small
    # negative tail is treated as an input artifact and clipped before log1p;
    # logged matrices (roughly centred around zero with no large positive
    # tail) are left untouched.
    count_like = modality == "RNA" and (float(np.max(finite)) > 50.0 or (float(np.min(finite)) >= 0.0 and float(np.max(finite)) > 10.0))
    return {
        "modality": modality,
        "finite": True,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "q01": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q99": float(quantiles[2]),
        "zero_fraction": float(np.mean(finite == 0.0)),
        "negative_fraction": float(np.mean(finite < 0.0)),
        "count_like_detected": bool(count_like),
        "decision": "log2(x+1) after clipping negative input to zero" if count_like else "no log transform",
    }


def _fit_numeric(raw: np.ndarray, train_idx: np.ndarray, output_dim: int, modality: str, center: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    train = np.asarray(raw[train_idx], dtype=np.float32)
    med = np.nanmedian(train, axis=0)
    med[~np.isfinite(med)] = 0.0
    clean = np.where(np.isfinite(raw), raw, med).astype(np.float32)
    audit = _scale_audit(clean, modality)
    negative_clipped = 0
    if modality == "RNA" and audit.get("count_like_detected", False):
        negative_clipped = int(np.sum(clean < 0.0))
        clean = np.maximum(clean, 0.0)
        clean = np.log2(clean + 1.0).astype(np.float32)
        transform = "log2(x+1)"
    else:
        transform = "none"
    train_clean = clean[train_idx]
    # Feature ranking is deliberately performed on the imputed, transformed
    # training matrix before any standardisation.  The old implementation
    # ranked the post-scaled matrix, making every retained variance nearly 1.
    raw_variance = np.var(train_clean, axis=0)
    finite_variance = np.isfinite(raw_variance)
    candidate = finite_variance & (raw_variance > 1e-12)
    if modality == "CNA":
        nonzero_fraction = np.mean(np.abs(train_clean) > 1e-12, axis=0)
        candidate &= nonzero_fraction > 0.0
    if not np.any(candidate):
        candidate[:] = finite_variance
    ranked = np.flatnonzero(candidate)
    ranked = ranked[np.argsort(-raw_variance[ranked], kind="stable")]
    keep = ranked[: min(output_dim, len(ranked))]
    if len(keep) < min(output_dim, clean.shape[1]):
        remaining = np.setdiff1d(np.arange(clean.shape[1]), keep, assume_unique=False)
        keep = np.concatenate([keep, remaining[: min(output_dim, clean.shape[1]) - len(keep)]])
    selected_train = train_clean[:, keep]
    mean = np.mean(selected_train, axis=0) if center else np.zeros(selected_train.shape[1], dtype=np.float32)
    std = np.std(selected_train, axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    transformed = ((clean[:, keep] - mean) / std).astype(np.float32)
    audit.update({
        "raw_dim": int(raw.shape[1]),
        "selected_dim": int(len(keep)),
        "transform": transform,
        "negative_values_clipped": negative_clipped,
        "imputation": "outer_train_median",
        "variance_ranking": "outer_train_transformed_unscaled_variance_descending",
        "selection_before_standardization": True,
        "selected_indices_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
        "selected_variance_max": float(np.max(raw_variance[keep])) if len(keep) else 0.0,
        "selected_variance_min": float(np.min(raw_variance[keep])) if len(keep) else 0.0,
        "standardizer": "outer_train_selected_features_mean_std",
    })
    return transformed, audit


def _fit_clinical(frame: pd.DataFrame, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[np.ndarray] = []
    age = pd.to_numeric(frame["AGE"], errors="coerce").to_numpy(dtype=float)
    train_age = age[train_idx]
    median = float(np.nanmedian(train_age)) if np.isfinite(train_age).any() else 0.0
    age = np.where(np.isfinite(age), age, median)
    scale = float(np.std(age[train_idx]))
    scale = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
    values.append(((age - median) / scale).reshape(-1, 1).astype(np.float32))
    category_counts: dict[str, int] = {}
    for column in CLINICAL_CATEGORICAL:
        series = frame[column].fillna("MISSING").astype(str).replace({"": "MISSING", "[Not Available]": "MISSING"})
        levels = sorted(set(series.iloc[train_idx].tolist()))
        if "MISSING" not in levels:
            levels.append("MISSING")
        level_index = {level: i for i, level in enumerate(levels)}
        encoded = np.zeros((len(series), len(levels)), dtype=np.float32)
        for row, item in enumerate(series):
            encoded[row, level_index.get(item, level_index["MISSING"])] = 1.0
        values.append(encoded)
        category_counts[column] = len(levels)
    return np.concatenate(values, axis=1), {"dim": int(sum(x.shape[1] for x in values)), "age_median": median, "category_counts": category_counts}


def preprocess_fold(raw: RawInputs, path_tokens: np.ndarray, fold: int) -> FoldArrays:
    train_idx = np.flatnonzero(raw.outer_folds != fold)
    rna, rna_audit = _fit_numeric(raw.rna, train_idx, output_dim=512, modality="RNA")
    cna, cna_audit = _fit_numeric(raw.cna, train_idx, output_dim=256, modality="CNA")
    clinical, clinical_audit = _fit_clinical(raw.clinical, train_idx)
    return FoldArrays(
        path_tokens=path_tokens,
        rna=rna,
        cna=cna,
        clinical=clinical,
        audit={"outer_fold": fold, "rna": rna_audit, "cna": cna_audit, "clinical": clinical_audit},
    )

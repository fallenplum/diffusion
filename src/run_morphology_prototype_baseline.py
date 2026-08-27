from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from run_multimodal_baselines import load_inputs, transform_clinical, transform_pca
from run_uni2_cox_baseline import bootstrap_ci, concordance_index, fit_cox


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def make_sparse_projection(
    input_dim: int, output_dim: int, seed: int
) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    nonzero_per_output = max(16, round(math.sqrt(input_dim)))
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    scale = 1.0 / math.sqrt(nonzero_per_output)
    for output_index in range(output_dim):
        chosen = rng.choice(input_dim, size=nonzero_per_output, replace=False)
        signs = rng.choice(np.asarray([-scale, scale]), size=nonzero_per_output)
        rows.extend(chosen.tolist())
        cols.extend([output_index] * nonzero_per_output)
        values.extend(signs.tolist())
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(input_dim, output_dim),
        dtype=np.float32,
    )


def sample_patient_bag(
    paths: list[Path], max_patches: int, seed: int
) -> np.ndarray:
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths]
    counts = np.asarray([len(array) for array in arrays], dtype=np.int64)
    total = int(counts.sum())
    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(total, size=min(max_patches, total), replace=False)
    )
    offsets = np.concatenate([[0], np.cumsum(counts)])
    pieces = []
    for index, array in enumerate(arrays):
        mask = (selected >= offsets[index]) & (selected < offsets[index + 1])
        if mask.any():
            local = selected[mask] - offsets[index]
            pieces.append(np.asarray(array[local], dtype=np.float32))
    return np.concatenate(pieces, axis=0)


def prepare_projected_cache(
    root: Path,
    patient_ids: np.ndarray,
    output_dim: int,
    max_patches: int,
    seed: int,
) -> dict[str, np.ndarray]:
    bag_manifest = pd.read_csv(root / "outputs" / "audit" / "uni2_bag_manifest.csv")
    paths_by_patient = {
        pid: [root / path for path in group["bag_path"].tolist()]
        for pid, group in bag_manifest.groupby("patient_id")
    }
    input_dims = bag_manifest["feature_dim"].unique()
    if len(input_dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions: {input_dims}")
    projection = make_sparse_projection(int(input_dims[0]), output_dim, seed)

    cache_dir = (
        root
        / "outputs"
        / "features"
        / f"uni2_h_rp{output_dim}_sample{max_patches}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    projected: dict[str, np.ndarray] = {}
    for position, pid in enumerate(patient_ids, start=1):
        cache_path = cache_dir / f"{pid}.npy"
        if cache_path.exists():
            values = np.load(cache_path, allow_pickle=False)
        else:
            paths = paths_by_patient.get(pid, [])
            if not paths:
                raise FileNotFoundError(f"No patch bag for {pid}")
            patches = sample_patient_bag(
                paths, max_patches, stable_seed(pid, seed)
            )
            values = np.asarray(patches @ projection, dtype=np.float32)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            values /= np.maximum(norms, 1e-8)
            np.save(cache_path, values.astype(np.float16), allow_pickle=False)
        projected[pid] = np.asarray(values, dtype=np.float32)
        if position % 50 == 0:
            print(f"[projection] patients={position}/{len(patient_ids)}", flush=True)
    return projected


def learn_prototypes(
    patient_ids: np.ndarray,
    projected: dict[str, np.ndarray],
    train_index: np.ndarray,
    n_prototypes: int,
    train_patches_per_patient: int,
    seed: int,
) -> np.ndarray:
    sampled = []
    for index in train_index:
        pid = patient_ids[index]
        values = projected[pid]
        rng = np.random.default_rng(stable_seed(pid, seed))
        chosen = rng.choice(
            len(values),
            size=min(train_patches_per_patient, len(values)),
            replace=False,
        )
        sampled.append(values[chosen])
    train_patches = np.concatenate(sampled, axis=0)
    model = MiniBatchKMeans(
        n_clusters=n_prototypes,
        batch_size=4096,
        n_init=5,
        max_iter=150,
        random_state=seed,
        reassignment_ratio=0.01,
    )
    model.fit(train_patches)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    return centers


def summarize_patient(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    similarity = values @ centers.T
    labels = similarity.argmax(axis=1)
    assigned_similarity = similarity[np.arange(len(values)), labels]
    counts = np.bincount(labels, minlength=len(centers)).astype(np.float32)
    proportions = counts / counts.sum()
    mean_similarity = np.zeros(len(centers), dtype=np.float32)
    for cluster in range(len(centers)):
        mask = labels == cluster
        if mask.any():
            mean_similarity[cluster] = assigned_similarity[mask].mean()
    return np.concatenate([proportions, mean_similarity])


def tune_alpha(
    features: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    alpha_grid: list[float],
    seed: int,
) -> tuple[float, float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {alpha: [] for alpha in alpha_grid}
    for train_index, valid_index in inner.split(features, events):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(features[train_index])
        valid_x = scaler.transform(features[valid_index])
        for alpha in alpha_grid:
            try:
                params = fit_cox(
                    durations[train_index], events[train_index], train_x, alpha
                )
                score = concordance_index(
                    durations[valid_index],
                    events[valid_index],
                    valid_x @ params,
                )
            except Exception:
                score = float("nan")
            scores[alpha].append(score)
    ranked = sorted(
        (
            (float(np.nanmean(values)), -alpha, alpha)
            for alpha, values in scores.items()
        ),
        reverse=True,
    )
    return ranked[0][2], ranked[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--train-patches-per-patient", type=int, default=256)
    parser.add_argument("--prototypes", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()

    inputs = load_inputs(root)
    patient_ids = np.asarray(inputs["patient_ids"]).astype(str)
    manifest = inputs["manifest"]
    clinical = inputs["clinical"]
    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(clinical, pd.DataFrame)
    durations = manifest["dss_months"].to_numpy(dtype=float)
    events = manifest["dss_status"].to_numpy(dtype=int)

    projected = prepare_projected_cache(
        root,
        patient_ids,
        args.projection_dim,
        args.max_patches,
        args.seed,
    )
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(outer.split(np.zeros(len(events)), events))
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]
    model_names = [
        "prototype_only",
        "uni2_mean_pca32",
        "uni2_mean_prototype",
        "clinical_prototype",
        "clinical_uni2_mean",
        "clinical_uni2_mean_prototype",
    ]
    oof = {name: np.full(len(events), np.nan) for name in model_names}
    fold_rows: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(folds, start=1):
        centers = learn_prototypes(
            patient_ids,
            projected,
            train_index,
            args.prototypes,
            args.train_patches_per_patient,
            args.seed + fold,
        )
        prototype_features = np.stack(
            [
                summarize_patient(projected[pid], centers)
                for pid in patient_ids
            ]
        )
        clinical_train, clinical_test = transform_clinical(
            clinical.iloc[train_index], clinical.iloc[test_index]
        )
        uni_train, uni_test = transform_pca(
            np.asarray(inputs["uni"])[train_index],
            np.asarray(inputs["uni"])[test_index],
            n_components=32,
            random_state=args.seed + fold,
        )
        clinical_all = np.zeros(
            (len(patient_ids), clinical_train.shape[1]), dtype=np.float32
        )
        clinical_all[train_index] = clinical_train
        clinical_all[test_index] = clinical_test
        uni_all = np.zeros((len(patient_ids), uni_train.shape[1]), dtype=np.float32)
        uni_all[train_index] = uni_train
        uni_all[test_index] = uni_test

        raw_features = {
            "prototype_only": prototype_features,
            "uni2_mean_pca32": uni_all,
            "uni2_mean_prototype": np.concatenate(
                [uni_all, prototype_features], axis=1
            ),
            "clinical_prototype": np.concatenate(
                [clinical_all, prototype_features], axis=1
            ),
            "clinical_uni2_mean": np.concatenate(
                [clinical_all, uni_all], axis=1
            ),
            "clinical_uni2_mean_prototype": np.concatenate(
                [clinical_all, uni_all, prototype_features], axis=1
            ),
        }
        for model_name, all_features in raw_features.items():
            alpha, inner_c = tune_alpha(
                all_features[train_index],
                durations[train_index],
                events[train_index],
                alpha_grid,
                args.seed + fold,
            )
            scaler = StandardScaler()
            train_x = scaler.fit_transform(all_features[train_index])
            test_x = scaler.transform(all_features[test_index])
            params = fit_cox(
                durations[train_index], events[train_index], train_x, alpha
            )
            risks = test_x @ params
            oof[model_name][test_index] = risks
            test_c = concordance_index(
                durations[test_index], events[test_index], risks
            )
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_prototypes": args.prototypes,
                    "projection_dim": args.projection_dim,
                    "max_patches_per_patient": args.max_patches,
                    "alpha": alpha,
                    "inner_c_index": inner_c,
                    "test_c_index": test_c,
                    "events_test": int(events[test_index].sum()),
                }
            )
            print(
                f"{model_name} fold={fold} C={test_c:.4f} "
                f"inner={inner_c:.4f} alpha={alpha}",
                flush=True,
            )

    summaries = []
    for model_name in model_names:
        ci = bootstrap_ci(durations, events, oof[model_name], seed=args.seed)
        model_folds = pd.DataFrame(fold_rows).query("model == @model_name")
        summaries.append(
            {
                "model": model_name,
                "n_patients": int(len(events)),
                "n_events": int(events.sum()),
                "oof_c_index": float(
                    concordance_index(durations, events, oof[model_name])
                ),
                "bootstrap_95_ci": [float(ci[0]), float(ci[1])],
                "fold_mean_c_index": float(
                    model_folds["test_c_index"].mean()
                ),
                "fold_std_c_index": float(
                    model_folds["test_c_index"].std(ddof=1)
                ),
            }
        )

    output_dir = root / "outputs" / "baselines" / "morphology_prototypes"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = manifest[
        ["patient_id", "dss_months", "dss_status", "tissue_source_site"]
    ].copy()
    for model_name in model_names:
        predictions[f"{model_name}_oof_risk"] = oof[model_name]
    predictions.to_csv(
        output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

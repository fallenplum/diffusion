from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from phase1_utils import (
    censoring_km,
    fit_cox,
    harrell_c,
    scaler_pca,
    uno_c,
    update_run_state,
)


SEED = 20260723


def compute_patient_means(root: Path, patients: pd.DataFrame, slides: pd.DataFrame) -> np.ndarray:
    cache = root / "outputs" / "pathology_context_phase1" / "cache" / "tcga_patient_mean_float16_source.npz"
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as bundle:
                cached_ids = bundle["patient_ids"].astype(str)
                if np.array_equal(cached_ids, patients["patient_id"].astype(str).to_numpy()):
                    return np.asarray(bundle["features"], dtype=np.float32)
        except (ValueError, OSError):
            # Older runs could have saved pandas object arrays.  They are not
            # safe to unpickle in a resumable run, so recompute this cache.
            pass
    vectors = []
    for idx, pid in enumerate(patients["patient_id"].astype(str), start=1):
        slide_vectors = []
        for bag_path in slides.loc[slides["patient_id"] == pid, "bag_path"]:
            bag = np.load(root / str(bag_path), mmap_mode="r", allow_pickle=False)
            slide_vectors.append(np.asarray(bag, dtype=np.float32).mean(axis=0, dtype=np.float64))
        vectors.append(np.stack(slide_vectors).mean(axis=0))
        if idx % 50 == 0:
            print(f"[mean] {idx}/{len(patients)} patients", flush=True)
    features = np.asarray(vectors, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Force a NumPy Unicode array; object arrays require pickle and break the
    # ``allow_pickle=False`` integrity boundary used by Phase1R resume.
    patient_ids = np.asarray(patients["patient_id"].astype(str).to_numpy(), dtype="<U64")
    np.savez_compressed(cache, patient_ids=patient_ids, features=features)
    return features


def tune(x, times, events, seed):
    component_grid = [16, 32, 64]
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    for n_components in component_grid:
        for alpha in alpha_grid:
            scores = []
            for train_idx, val_idx in splitter.split(x, events):
                x_train, x_val, _, _ = scaler_pca(x[train_idx], x[val_idx], n_components, seed)
                beta = fit_cox(times[train_idx], events[train_idx], x_train, alpha)
                scores.append(harrell_c(times[val_idx], events[val_idx], x_val @ beta))
            rows.append((float(np.nanmean(scores)), -n_components, -alpha, n_components, alpha))
    rows.sort(reverse=True)
    return rows[0][3], rows[0][4], rows[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    phase = root / "outputs" / "pathology_context_phase1"
    patients = pd.read_csv(phase / "audit" / "patient_manifest_used.csv")
    slides = pd.read_csv(phase / "audit" / "slide_manifest_used.csv")
    features = compute_patient_means(root, patients, slides)
    times = patients["dss_months"].to_numpy(dtype=float)
    events = patients["dss_status"].to_numpy(dtype=int)
    folds = patients["outer_fold"].to_numpy(dtype=int)
    risks = np.full(len(patients), np.nan)
    g_values = np.full(len(patients), np.nan)
    fold_rows = []
    started = time.perf_counter()

    output = phase / "fold_metrics" / "B0"
    output.mkdir(parents=True, exist_ok=True)
    oof_dir = phase / "oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = phase / "embeddings" / "B0_mean"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):
        job_started = time.perf_counter()
        test = folds == fold
        train = ~test
        n_components, alpha, inner_c = tune(features[train], times[train], events[train], SEED + fold)
        x_train, x_test, _, pca = scaler_pca(features[train], features[test], n_components, SEED + fold)
        beta = fit_cox(times[train], events[train], x_train, alpha)
        risks[test] = x_test @ beta
        censor = censoring_km(times[train], events[train])
        g_values[test] = censor(times[test], left_limit=True)
        fold_rows.append({
            "model": "B0", "seed": SEED, "outer_fold": fold,
            "n_train": int(train.sum()), "n_test": int(test.sum()),
            "events_train": int(events[train].sum()), "events_test": int(events[test].sum()),
            "pca_components": n_components, "alpha": alpha, "inner_harrell_c": inner_c,
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "test_harrell_c": harrell_c(times[test], events[test], risks[test]),
            "test_uno_c": uno_c(times[test], events[test], risks[test], g_values[test]),
            "training_time_seconds": time.perf_counter() - job_started,
        })
        pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
        update_run_state(root, f"B0/fold_{fold}/seed_{SEED}", "stage_1_b0")
        print(json.dumps(fold_rows[-1], indent=2), flush=True)

    if not np.isfinite(risks).all() or not np.isfinite(g_values).all():
        raise RuntimeError("Incomplete B0 OOF predictions")
    predictions = patients[["patient_id", "outer_fold", "dss_months", "dss_status"]].copy()
    predictions["seed"] = SEED
    predictions["oof_risk"] = risks
    predictions["censor_survival_outer_train"] = g_values
    predictions.to_csv(oof_dir / f"B0_seed_{SEED}.csv", index=False, encoding="utf-8-sig")
    embedding = predictions.copy()
    embedding["embedding"] = [json.dumps(row.tolist()) for row in features]
    embedding.to_csv(embedding_dir / f"seed_{SEED}.csv", index=False, encoding="utf-8-sig")

    historical_path = root / "outputs" / "baselines" / "uni2_mean_cox" / "summary.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    result = {
        "model": "B0",
        "source": "TCGA/uni2_h_bags float16 bags, slide mean then patient mean",
        "seed": SEED,
        "n_patients": len(patients),
        "n_events": int(events.sum()),
        "oof_harrell_c": harrell_c(times, events, risks),
        "oof_uno_c": uno_c(times, events, risks, g_values),
        "fold_mean_harrell_c": float(pd.DataFrame(fold_rows)["test_harrell_c"].mean()),
        "fold_std_harrell_c": float(pd.DataFrame(fold_rows)["test_harrell_c"].std(ddof=1)),
        "historical_harrell_c": historical["overall_oof_c_index"],
        "delta_vs_historical": harrell_c(times, events, risks) - historical["overall_oof_c_index"],
        "training_time_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_uni2_cox_baseline import bootstrap_ci, concordance_index, fit_cox


PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
SAMPLE_TYPE_RE = re.compile(
    r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})", re.IGNORECASE
)

CLINICAL_CONTINUOUS = ["AGE"]
CLINICAL_CATEGORICAL = [
    "SEX",
    "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "AJCC_STAGING_EDITION",
    "PATH_T_STAGE",
    "PATH_N_STAGE",
    "PATH_M_STAGE",
    "HISTOLOGICAL_DIAGNOSIS",
    "PRIMARY_SITE_PATIENT",
]


def patient_id(value: object) -> str:
    match = PATIENT_RE.match(str(value).strip().upper())
    if not match:
        raise ValueError(f"无法解析患者ID: {value!r}")
    return match.group(1)


def sample_priority(value: object) -> int:
    match = SAMPLE_TYPE_RE.match(str(value).strip().upper())
    if not match:
        return 99
    order = {"01": 0, "02": 1, "06": 2, "11": 3}
    return order.get(match.group(2), 10)


def load_inputs(root: Path) -> dict[str, object]:
    feature_bundle = np.load(
        root / "outputs" / "features" / "uni2_h_patient_mean.npz"
    )
    patient_ids = feature_bundle["patient_ids"].astype(str)
    uni = np.asarray(feature_bundle["features"], dtype=np.float32)

    manifest = pd.read_csv(root / "outputs" / "audit" / "uni2_patient_manifest.csv")
    manifest = manifest.set_index("patient_id").loc[patient_ids].reset_index()

    clinical_path = next(
        (root / ".cache" / "metadata").glob(
            "merged_clinical_data_preprocess*.csv"
        )
    )
    clinical = pd.read_csv(clinical_path, low_memory=False)
    clinical["patient_id"] = clinical["PATIENT_ID"].map(patient_id)
    clinical = clinical.set_index("patient_id").loc[patient_ids]
    clinical_values = clinical[
        CLINICAL_CONTINUOUS + CLINICAL_CATEGORICAL
    ].apply(pd.to_numeric, errors="coerce")

    rna_path = next((root / "data").glob("merged_rna_data_2_preprocess_*.csv"))
    rna = pd.read_csv(rna_path, low_memory=False)
    id_col = rna.columns[0]
    rna["patient_id"] = rna[id_col].map(patient_id)
    rna["_priority"] = rna[id_col].map(sample_priority)
    rna = (
        rna.sort_values(["patient_id", "_priority"])
        .drop_duplicates("patient_id", keep="first")
        .set_index("patient_id")
        .loc[patient_ids]
    )
    rna_values = rna.drop(columns=[id_col, "_priority"]).apply(
        pd.to_numeric, errors="coerce"
    )
    negative_count = int((rna_values.to_numpy() < 0).sum())
    rna_values = np.log2(np.clip(rna_values.to_numpy(dtype=np.float32), 0, None) + 1)

    return {
        "patient_ids": patient_ids,
        "uni": uni,
        "rna": rna_values,
        "clinical": clinical_values,
        "manifest": manifest,
        "rna_negative_count": negative_count,
    }


def transform_pca(
    train: np.ndarray,
    evaluation: np.ndarray,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    eval_scaled = scaler.transform(evaluation)
    n_components = min(n_components, len(train_scaled) - 1, train_scaled.shape[1])
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    return pca.fit_transform(train_scaled), pca.transform(eval_scaled)


def transform_clinical(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    continuous_scaler = StandardScaler()
    train_continuous = continuous_scaler.fit_transform(
        train[CLINICAL_CONTINUOUS].fillna(0)
    )
    eval_continuous = continuous_scaler.transform(
        evaluation[CLINICAL_CONTINUOUS].fillna(0)
    )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_categorical = encoder.fit_transform(
        train[CLINICAL_CATEGORICAL].fillna(-1).astype(str)
    )
    eval_categorical = encoder.transform(
        evaluation[CLINICAL_CATEGORICAL].fillna(-1).astype(str)
    )
    return (
        np.concatenate([train_continuous, train_categorical], axis=1),
        np.concatenate([eval_continuous, eval_categorical], axis=1),
    )


def transform_model(
    model_name: str,
    inputs: dict[str, object],
    train_index: np.ndarray,
    eval_index: np.ndarray,
    n_components: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "clinical":
        clinical = inputs["clinical"]
        assert isinstance(clinical, pd.DataFrame)
        return transform_clinical(
            clinical.iloc[train_index], clinical.iloc[eval_index]
        )
    if model_name == "rna":
        assert n_components is not None
        return transform_pca(
            np.asarray(inputs["rna"])[train_index],
            np.asarray(inputs["rna"])[eval_index],
            n_components,
            random_state,
        )
    if model_name == "uni2":
        assert n_components is not None
        return transform_pca(
            np.asarray(inputs["uni"])[train_index],
            np.asarray(inputs["uni"])[eval_index],
            n_components,
            random_state,
        )
    if model_name == "clinical_rna_uni2":
        assert n_components is not None
        uni_train, uni_eval = transform_pca(
            np.asarray(inputs["uni"])[train_index],
            np.asarray(inputs["uni"])[eval_index],
            n_components,
            random_state,
        )
        rna_train, rna_eval = transform_pca(
            np.asarray(inputs["rna"])[train_index],
            np.asarray(inputs["rna"])[eval_index],
            n_components,
            random_state + 1000,
        )
        clinical = inputs["clinical"]
        assert isinstance(clinical, pd.DataFrame)
        clinical_train, clinical_eval = transform_clinical(
            clinical.iloc[train_index], clinical.iloc[eval_index]
        )
        combined_train = np.concatenate(
            [uni_train, rna_train, clinical_train], axis=1
        )
        combined_eval = np.concatenate(
            [uni_eval, rna_eval, clinical_eval], axis=1
        )
        final_scaler = StandardScaler()
        return (
            final_scaler.fit_transform(combined_train),
            final_scaler.transform(combined_eval),
        )
    raise ValueError(f"未知模型: {model_name}")


def tune_model(
    model_name: str,
    inputs: dict[str, object],
    outer_train: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    component_grid: list[int | None],
    alpha_grid: list[float],
    random_state: int,
) -> tuple[int | None, float, float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    scores = {
        (components, alpha): []
        for components in component_grid
        for alpha in alpha_grid
    }
    for local_train, local_valid in inner.split(outer_train, events[outer_train]):
        train_index = outer_train[local_train]
        valid_index = outer_train[local_valid]
        for components in component_grid:
            train_x, valid_x = transform_model(
                model_name,
                inputs,
                train_index,
                valid_index,
                components,
                random_state,
            )
            for alpha in alpha_grid:
                try:
                    params = fit_cox(
                        durations[train_index],
                        events[train_index],
                        train_x,
                        alpha,
                    )
                    score = concordance_index(
                        durations[valid_index],
                        events[valid_index],
                        valid_x @ params,
                    )
                except Exception:
                    score = float("nan")
                scores[(components, alpha)].append(score)

    ranked = []
    for (components, alpha), values in scores.items():
        mean_score = float(np.nanmean(values))
        complexity = components or 0
        ranked.append((mean_score, -complexity, -alpha, components, alpha))
    ranked.sort(reverse=True)
    best = ranked[0]
    return best[3], best[4], best[0]


def evaluate_model(
    model_name: str,
    inputs: dict[str, object],
    durations: np.ndarray,
    events: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    component_grid: list[int | None],
    alpha_grid: list[float],
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    oof = np.full(len(durations), np.nan)
    fold_rows = []
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        components, alpha, inner_c = tune_model(
            model_name,
            inputs,
            train_index,
            durations,
            events,
            component_grid,
            alpha_grid,
            seed + fold,
        )
        train_x, test_x = transform_model(
            model_name,
            inputs,
            train_index,
            test_index,
            components,
            seed + fold,
        )
        params = fit_cox(
            durations[train_index], events[train_index], train_x, alpha
        )
        oof[test_index] = test_x @ params
        test_c = concordance_index(
            durations[test_index], events[test_index], oof[test_index]
        )
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "components_per_high_dim_modality": components,
                "alpha": alpha,
                "inner_c_index": inner_c,
                "test_c_index": test_c,
                "events_test": int(events[test_index].sum()),
            }
        )
        print(
            f"{model_name} fold={fold} C={test_c:.4f} "
            f"inner={inner_c:.4f} components={components} alpha={alpha}",
            flush=True,
        )
    overall = concordance_index(durations, events, oof)
    ci = bootstrap_ci(durations, events, oof, seed=seed)
    fold_frame = pd.DataFrame(fold_rows)
    summary = {
        "model": model_name,
        "n_patients": int(len(durations)),
        "n_events": int(events.sum()),
        "oof_c_index": float(overall),
        "bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        "fold_mean_c_index": float(fold_frame["test_c_index"].mean()),
        "fold_std_c_index": float(fold_frame["test_c_index"].std(ddof=1)),
    }
    return summary, fold_frame, oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = load_inputs(root)
    manifest = inputs["manifest"]
    assert isinstance(manifest, pd.DataFrame)
    durations = manifest["dss_months"].to_numpy(dtype=float)
    events = manifest["dss_status"].to_numpy(dtype=int)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(outer.split(np.zeros(len(events)), events))
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]

    configs = {
        "clinical": [None],
        "rna": [16, 32, 64],
        "uni2": [16, 32, 64],
        "clinical_rna_uni2": [16, 32],
    }
    summaries = []
    fold_frames = []
    predictions = manifest[
        ["patient_id", "dss_months", "dss_status", "tissue_source_site"]
    ].copy()
    print(
        f"患者={len(events)}, 事件={events.sum()}, "
        f"RNA负值裁剪数={inputs['rna_negative_count']}",
        flush=True,
    )
    for model_name, component_grid in configs.items():
        summary, fold_frame, oof = evaluate_model(
            model_name,
            inputs,
            durations,
            events,
            folds,
            component_grid,
            alpha_grid,
            args.seed,
        )
        summaries.append(summary)
        fold_frames.append(fold_frame)
        predictions[f"{model_name}_oof_risk"] = oof

    output_dir = root / "outputs" / "baselines" / "multimodal_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    predictions.to_csv(
        output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

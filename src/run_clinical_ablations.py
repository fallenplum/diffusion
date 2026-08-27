from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_multimodal_baselines import load_inputs
from run_uni2_cox_baseline import bootstrap_ci, concordance_index, fit_cox


CONFIGS = {
    "age_sex": {
        "continuous": ["AGE"],
        "categorical": ["SEX"],
    },
    "stage_only": {
        "continuous": [],
        "categorical": ["AJCC_PATHOLOGIC_TUMOR_STAGE"],
    },
    "tnm_age_sex": {
        "continuous": ["AGE"],
        "categorical": [
            "SEX",
            "PATH_T_STAGE",
            "PATH_N_STAGE",
            "PATH_M_STAGE",
        ],
    },
    "full_safe_clinical": {
        "continuous": ["AGE"],
        "categorical": [
            "SEX",
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "AJCC_STAGING_EDITION",
            "PATH_T_STAGE",
            "PATH_N_STAGE",
            "PATH_M_STAGE",
            "HISTOLOGICAL_DIAGNOSIS",
            "PRIMARY_SITE_PATIENT",
        ],
    },
}


def transform(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    eval_parts = []
    if continuous:
        scaler = StandardScaler()
        train_parts.append(scaler.fit_transform(train[continuous].fillna(0)))
        eval_parts.append(scaler.transform(evaluation[continuous].fillna(0)))
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_parts.append(
            encoder.fit_transform(train[categorical].fillna(-1).astype(str))
        )
        eval_parts.append(
            encoder.transform(evaluation[categorical].fillna(-1).astype(str))
        )
    return np.concatenate(train_parts, axis=1), np.concatenate(eval_parts, axis=1)


def tune_alpha(
    clinical: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    outer_train: np.ndarray,
    config: dict[str, list[str]],
    alpha_grid: list[float],
    seed: int,
) -> tuple[float, float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {alpha: [] for alpha in alpha_grid}
    for local_train, local_valid in inner.split(outer_train, events[outer_train]):
        train_index = outer_train[local_train]
        valid_index = outer_train[local_valid]
        train_x, valid_x = transform(
            clinical.iloc[train_index],
            clinical.iloc[valid_index],
            config["continuous"],
            config["categorical"],
        )
        for alpha in alpha_grid:
            params = fit_cox(
                durations[train_index], events[train_index], train_x, alpha
            )
            scores[alpha].append(
                concordance_index(
                    durations[valid_index],
                    events[valid_index],
                    valid_x @ params,
                )
            )
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
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = load_inputs(root)
    manifest = inputs["manifest"]
    clinical = inputs["clinical"]
    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(clinical, pd.DataFrame)
    durations = manifest["dss_months"].to_numpy(dtype=float)
    events = manifest["dss_status"].to_numpy(dtype=int)
    folds = list(
        StratifiedKFold(
            n_splits=5, shuffle=True, random_state=args.seed
        ).split(np.zeros(len(events)), events)
    )
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]
    summaries = []
    predictions = manifest[["patient_id", "dss_months", "dss_status"]].copy()
    fold_rows = []

    for name, config in CONFIGS.items():
        oof = np.full(len(events), np.nan)
        for fold, (train_index, test_index) in enumerate(folds, start=1):
            alpha, inner_c = tune_alpha(
                clinical,
                durations,
                events,
                train_index,
                config,
                alpha_grid,
                args.seed + fold,
            )
            train_x, test_x = transform(
                clinical.iloc[train_index],
                clinical.iloc[test_index],
                config["continuous"],
                config["categorical"],
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
                    "model": name,
                    "fold": fold,
                    "alpha": alpha,
                    "inner_c_index": inner_c,
                    "test_c_index": test_c,
                }
            )
        overall = concordance_index(durations, events, oof)
        ci = bootstrap_ci(durations, events, oof, seed=args.seed)
        summaries.append(
            {
                "model": name,
                "oof_c_index": overall,
                "bootstrap_95_ci": [ci[0], ci[1]],
            }
        )
        predictions[f"{name}_oof_risk"] = oof
        print(
            f"{name}: C={overall:.4f}, 95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]",
            flush=True,
        )

    output_dir = root / "outputs" / "baselines" / "clinical_ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(
        output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

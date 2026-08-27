from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def concordance_index(
    durations: np.ndarray, events: np.ndarray, risks: np.ndarray
) -> float:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in range(len(durations)):
        if events[i] != 1:
            continue
        mask = durations > durations[i]
        if not np.any(mask):
            continue
        other_risks = risks[mask]
        comparable += int(mask.sum())
        concordant += float(np.sum(risks[i] > other_risks))
        concordant += 0.5 * float(np.sum(risks[i] == other_risks))
    return concordant / comparable if comparable else float("nan")


def fit_transform(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    n_components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    eval_scaled = scaler.transform(x_eval)
    components = min(n_components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=random_state,
    )
    return (
        pca.fit_transform(train_scaled),
        pca.transform(eval_scaled),
        scaler,
        pca,
    )


def fit_cox(
    durations: np.ndarray,
    events: np.ndarray,
    features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    event_indices = np.flatnonzero(events == 1)
    if len(event_indices) == 0:
        raise ValueError("训练集中没有事件")
    risk_masks = [durations >= durations[i] for i in event_indices]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = features @ beta
        log_likelihood = 0.0
        gradient_log_likelihood = np.zeros_like(beta)
        for event_index, risk_mask in zip(event_indices, risk_masks):
            risk_eta = eta[risk_mask]
            normalizer = logsumexp(risk_eta)
            weights = np.exp(risk_eta - normalizer)
            weighted_feature = weights @ features[risk_mask]
            log_likelihood += eta[event_index] - normalizer
            gradient_log_likelihood += features[event_index] - weighted_feature
        scale = float(len(event_indices))
        loss = -log_likelihood / scale + 0.5 * alpha * float(beta @ beta)
        gradient = -gradient_log_likelihood / scale + alpha * beta
        return loss, gradient

    initial = np.zeros(features.shape[1], dtype=float)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Cox优化失败: {result.message}")
    return np.asarray(result.x, dtype=float)


def tune_parameters(
    x: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    component_grid: list[int],
    alpha_grid: list[float],
    random_state: int,
) -> tuple[int, float, float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    scores: dict[tuple[int, float], list[float]] = {
        (n, alpha): [] for n in component_grid for alpha in alpha_grid
    }
    for train_index, valid_index in inner.split(x, events):
        for n_components in component_grid:
            x_train, x_valid, _, _ = fit_transform(
                x[train_index], x[valid_index], n_components, random_state
            )
            for alpha in alpha_grid:
                try:
                    params = fit_cox(
                        durations[train_index],
                        events[train_index],
                        x_train,
                        alpha,
                    )
                    risk = x_valid @ params
                    score = concordance_index(
                        durations[valid_index], events[valid_index], risk
                    )
                except Exception:
                    score = float("nan")
                scores[(n_components, alpha)].append(score)

    ranked = []
    for (n_components, alpha), values in scores.items():
        finite = np.asarray(values, dtype=float)
        mean_score = (
            float(np.nanmean(finite))
            if np.any(np.isfinite(finite))
            else float("-inf")
        )
        ranked.append((mean_score, -n_components, -alpha, n_components, alpha))
    ranked.sort(reverse=True)
    best = ranked[0]
    if not np.isfinite(best[0]):
        raise RuntimeError("所有内层Cox配置均拟合失败")
    return best[3], best[4], best[0]


def bootstrap_ci(
    durations: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 20260723,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(durations)
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, n)
        score = concordance_index(durations[index], events[index], risks[index])
        if np.isfinite(score):
            scores.append(score)
    return tuple(np.quantile(scores, [0.025, 0.975]).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    root = args.root.resolve()

    feature_file = root / "outputs" / "features" / "uni2_h_patient_mean.npz"
    manifest_file = root / "outputs" / "audit" / "uni2_patient_manifest.csv"
    bundle = np.load(feature_file)
    features = np.asarray(bundle["features"], dtype=np.float32)
    patient_ids = bundle["patient_ids"].astype(str)

    manifest = pd.read_csv(manifest_file)
    manifest = manifest.set_index("patient_id").loc[patient_ids].reset_index()
    durations = manifest["dss_months"].to_numpy(dtype=float)
    events = manifest["dss_status"].to_numpy(dtype=int)

    if not np.isfinite(features).all():
        raise ValueError("UNI2-h患者特征存在NaN或Inf")
    if len(features) != len(manifest):
        raise ValueError("特征数和患者标签数不一致")

    component_grid = [16, 32, 64]
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0]
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_risk = np.full(len(features), np.nan, dtype=float)
    fold_rows = []

    print(
        f"患者={len(features)}, 事件={events.sum()}, 特征维数={features.shape[1]}",
        flush=True,
    )
    for fold, (train_index, test_index) in enumerate(
        outer.split(features, events), start=1
    ):
        n_components, alpha, inner_score = tune_parameters(
            features[train_index],
            durations[train_index],
            events[train_index],
            component_grid,
            alpha_grid,
            args.seed + fold,
        )
        x_train, x_test, _, pca = fit_transform(
            features[train_index],
            features[test_index],
            n_components,
            args.seed + fold,
        )
        params = fit_cox(
            durations[train_index],
            events[train_index],
            x_train,
            alpha,
        )
        risk = x_test @ params
        oof_risk[test_index] = risk
        fold_c = concordance_index(
            durations[test_index], events[test_index], risk
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "events_train": int(events[train_index].sum()),
                "events_test": int(events[test_index].sum()),
                "pca_components": n_components,
                "pca_explained_variance": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "alpha": alpha,
                "inner_c_index": inner_score,
                "test_c_index": fold_c,
            }
        )
        print(
            f"fold={fold} test_C={fold_c:.4f} inner_C={inner_score:.4f} "
            f"PCA={n_components} alpha={alpha}",
            flush=True,
        )

    if not np.isfinite(oof_risk).all():
        raise RuntimeError("存在缺失的OOF预测")
    overall_c = concordance_index(durations, events, oof_risk)
    ci_low, ci_high = bootstrap_ci(durations, events, oof_risk)

    output_dir = root / "outputs" / "baselines" / "uni2_mean_cox"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    prediction_frame = manifest[
        ["patient_id", "dss_months", "dss_status", "tissue_source_site"]
    ].copy()
    prediction_frame["oof_risk"] = oof_risk
    prediction_frame.to_csv(
        output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "model": "UNI2-h patch mean -> patient slide mean -> fold-wise scaler/PCA -> ridge Cox",
        "n_patients": int(len(features)),
        "n_events": int(events.sum()),
        "feature_dim": int(features.shape[1]),
        "outer_folds": 5,
        "overall_oof_c_index": float(overall_c),
        "bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "fold_mean_c_index": float(fold_frame["test_c_index"].mean()),
        "fold_std_c_index": float(fold_frame["test_c_index"].std(ddof=1)),
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

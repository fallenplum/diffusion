"""Numerical HEST recipe pinned to the audited upstream source commit.

This is the 50-HVG task, not the historical four-program CP10K task. The
round-four split-return boundary is part of the upstream aggregation contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HEST_COMMIT = "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3"
UPSTREAM_BLOBS = {
    "benchmark.py": "f7179acb4462dfc84b949eb16f8728f72387af6a",
    "trainer.py": "397649041ed227c82c61ea4ef67fb033d6550465",
    "st_dataset.py": "57655967cc9d540c315806cec31948aa2c62d23f",
}
DATASET_REVISION = "b039b5965ed7092f280d4f336da7972ef9a10e1a"
ALPHA = 100.0 / (256 * 50)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def log1p_only_selected_expression(values):
    """Upstream selects rows/50 genes, casts float64, then log1p only."""
    if sparse.issparse(values):
        matrix = values.astype(np.float64).copy()
        require(np.isfinite(matrix.data).all() and (matrix.data > -1).all(), "INVALID_LOG1P_EXPRESSION")
        matrix.data = np.log1p(matrix.data)
        return matrix.toarray()
    matrix = np.array(values, dtype=np.float64, copy=True)
    require(matrix.ndim == 2 and np.isfinite(matrix).all() and (matrix > -1).all(),
            "INVALID_LOG1P_EXPRESSION")
    return np.log1p(matrix)


@dataclass
class OfficialRidge:
    pipeline: Pipeline
    estimator: Ridge
    feature_dimension: int
    seed: int

    def transform(self, features):
        x = np.asarray(features)
        require(x.ndim == 2 and x.shape[1] == self.feature_dimension and np.isfinite(x).all(),
                "HEST_PREDICTION_FEATURE_SHAPE_OR_FINITE")
        # torch.Tensor(pipe.transform(...)) in upstream is an explicit FP32 cast.
        return np.asarray(self.pipeline.transform(x), dtype=np.float32)

    def predict(self, features):
        return self.estimator.predict(self.transform(features))


def fit_official_ridge(features, targets, *, seed=1):
    x, y = np.asarray(features), np.asarray(targets)
    require(x.ndim == y.ndim == 2 and len(x) == len(y) and y.shape[1] == 50
            and np.isfinite(x).all() and np.isfinite(y).all(), "HEST_RIDGE_TRAIN_INPUTS")
    require(min(x.shape) >= 256, "OFFICIAL_PCA256_INFEASIBLE_NO_SILENT_DIMENSION_REDUCTION")
    require(y.dtype == np.float64, "OFFICIAL_LOG1P_TARGETS_MUST_REMAIN_FLOAT64")
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("PCA", PCA(n_components=256, random_state=int(seed)))])
    train_x = np.asarray(pipe.fit_transform(x), dtype=np.float32)
    estimator = Ridge(solver="lsqr", alpha=ALPHA, random_state=int(seed),
                      fit_intercept=False, max_iter=1000)
    estimator.fit(train_x, y)
    return OfficialRidge(pipe, estimator, x.shape[1], int(seed))


def _mean_without_dropping(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def upstream_raw_metrics(targets, predictions, genes):
    """Preserve upstream NaN/Inf in memory; JSON conversion is explicit below."""
    y, p = np.asarray(targets), np.asarray(predictions)
    require(y.shape == p.shape and y.ndim == 2 and y.shape[1] == len(genes) == 50
            and len(set(genes)) == 50 and np.isfinite(y).all() and np.isfinite(p).all(),
            "HEST_METRIC_50_GENE_SHAPES")
    errors, r2, correlations = [], [], []
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        for column, gene in enumerate(genes):
            target, prediction = y[:, column], p[:, column]
            errors.append(float(np.mean((prediction - target)**2)))
            r2.append(float(1 - np.sum((target - prediction)**2) / np.sum((target - np.mean(target))**2)))
            correlations.append(float(pearsonr(target, prediction)[0]))
    result = dict(l2_errors=errors, r2_scores=r2,
        pearson_corrs=[dict(name=str(gene), pearson_corr=c) for gene, c in zip(genes, correlations)],
        pearson_mean=_mean_without_dropping(correlations), pearson_std=float(np.std(correlations)))
    for prefix, values in (("l2_error", errors), ("r2_score", r2)):
        result.update({prefix + "_q1": float(np.percentile(values, 25)),
                       prefix + "_q2": float(np.median(values)),
                       prefix + "_q3": float(np.percentile(values, 75))})
    return result


def official_round4(value):
    """Match benchmark._sanitize_metrics BEFORE fold aggregation."""
    if isinstance(value, dict):
        return {k: official_round4(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [official_round4(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return round(value, 4) if isinstance(value, float) else value


def json_safe_nonfinite(value):
    """NaN/Inf become null, never zero and never excluded from an official mean."""
    if isinstance(value, dict):
        return {k: json_safe_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_nonfinite(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def from_json_nonfinite(value):
    if isinstance(value, dict):
        return {k: from_json_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_json_nonfinite(v) for v in value]
    return np.nan if value is None else value


def aggregate_upstream_round4(split_metrics):
    """Upstream averages the already rounded split return values, then rounds."""
    require(len(split_metrics) == 2, "EXACT_TWO_OFFICIAL_FOLDS_REQUIRED")
    split_metrics = [from_json_nonfinite(x) for x in split_metrics]
    genes = [x["name"] for x in split_metrics[0]["pearson_corrs"]]
    require(all([x["name"] for x in m["pearson_corrs"]] == genes for m in split_metrics),
            "OFFICIAL_GENE_ORDER_CHANGED_ACROSS_FOLDS")
    rows = []
    for column, gene in enumerate(genes):
        values = [m["pearson_corrs"][column]["pearson_corr"] for m in split_metrics]
        rows.append(dict(name=gene, pearson_corrs=values, mean=float(np.mean(values)), std=float(np.std(values))))
    means = [m["pearson_mean"] for m in split_metrics]
    # Finite genes retain upstream's descending order. Undefined ranks have no
    # scientific meaning, and are retained at the end with their support explicit.
    rows = sorted(rows, key=lambda r: r["mean"] if np.isfinite(r["mean"]) else -np.inf, reverse=True)
    return json_safe_nonfinite(official_round4(dict(pearson_corrs=rows, pearson_mean=float(np.mean(means)),
        pearson_std=float(np.std(means)), mean_per_split=means)))


def supplementary_metrics(targets, predictions):
    y, p = np.asarray(targets), np.asarray(predictions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrs = np.asarray([pearsonr(y[:, k], p[:, k])[0] for k in range(50)])
    return json_safe_nonfinite(dict(MAE=float(np.mean(np.abs(y - p))), MSE=float(np.mean((y - p)**2)),
        median_per_gene_Pearson=float(np.median(corrs)), defined_Pearson_genes=int(np.isfinite(corrs).sum()),
        total_genes=50, mean_drops_undefined_genes=False,
        nonfinite_serialization="IEEE_NaN_or_Inf_to_null_without_imputation_or_gene_dropping"))

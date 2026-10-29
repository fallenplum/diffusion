"""Synthetic mathematical certification, not a fit on HEST or patient data."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from src.spgel_hest.official import (ALPHA, UPSTREAM_BLOBS, aggregate_upstream_round4,
    fit_official_ridge, log1p_only_selected_expression, official_round4,
    json_safe_nonfinite, supplementary_metrics, upstream_raw_metrics)


def test_target_is_log1p_of_selected_counts_without_total_normalization():
    x = np.array([[1, 3], [20, 7]], dtype=np.int32)
    a = log1p_only_selected_expression(x)
    b = log1p_only_selected_expression(sparse.csr_matrix(x))
    assert a.dtype == np.float64 and np.array_equal(a, np.log1p(x.astype(float)))
    assert np.array_equal(a, b) and np.array_equal(x, [[1, 3], [20, 7]])
    assert not np.allclose(a, np.log1p(x / x.sum(1, keepdims=True) * 10000.))


def test_pca256_never_silently_shrinks_to_data_size():
    with pytest.raises(ValueError, match="NO_SILENT_DIMENSION_REDUCTION"):
        fit_official_ridge(np.ones((100, 300), np.float32), np.ones((100, 50), np.float64))


def test_actual_fit_uses_train_only_preprocessing_and_predict_accepts_no_expression():
    rng = np.random.default_rng(19)
    train = rng.normal(size=(270, 300)).astype(np.float32)
    targets = rng.normal(size=(270, 50)).astype(np.float64)
    test = rng.normal(size=(10, 300)).astype(np.float32)
    with threadpool_limits(limits=2):
        model = fit_official_ridge(train, targets)
        mean = model.pipeline["scaler"].mean_.copy()
        prediction = model.predict(test)
        model.predict(test * 1e8)
    assert np.array_equal(mean, model.pipeline["scaler"].mean_)
    assert np.allclose(mean, train.mean(0), atol=1e-7)
    assert model.transform(test).dtype == np.float32 and prediction.shape == (10, 50)
    assert model.estimator.alpha == ALPHA and model.estimator.fit_intercept is False


def test_all_fifty_gene_metric_keeps_undefined_gene_as_null_not_zero_or_dropped():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(30, 50));p = y.copy();y[:, 0] = 1.
    raw = upstream_raw_metrics(y, p, [f"G{k}" for k in range(50)])
    assert np.isnan(raw["pearson_mean"])
    safe = json_safe_nonfinite(raw)
    assert safe["pearson_mean"] is None and safe["pearson_corrs"][0]["pearson_corr"] is None
    diagnostic = supplementary_metrics(y, p)
    assert diagnostic["defined_Pearson_genes"] == 49 and diagnostic["median_per_gene_Pearson"] is None


def test_official_aggregate_uses_rounded_split_return_values():
    rows = []
    for value in [.234449, .345549]:
        rows.append(official_round4(dict(pearson_mean=value,
            pearson_corrs=[dict(name=f"G{k}", pearson_corr=value) for k in range(50)])))
    result = aggregate_upstream_round4(rows)
    assert result["mean_per_split"] == [.2344, .3455]
    # Upstream first casts np.generic to Python float; numpy's scalar round
    # is observably different at this deliberately chosen halfway case.
    assert result["pearson_mean"] == round(float(np.mean([.2344, .3455])), 4)
    assert result["pearson_mean"] != round(float(np.mean([.234449, .345549])), 4)


@pytest.mark.parametrize("ntrain,nfeatures,solver", [(280, 300, "full"), (400, 1536, "randomized")])
def test_numerical_prediction_and_metrics_against_actual_pinned_upstream_trainer(ntrain, nfeatures, solver):
    path = Path(__file__).parents[1] / "tmp/hest_official_blobs/trainer.py"
    if not path.is_file():
        pytest.skip("certification requires the separately fetched, exact primary Git blob")
    payload = path.read_bytes()
    assert hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() == UPSTREAM_BLOBS["trainer.py"]
    specification = importlib.util.spec_from_file_location("pinned_hest_upstream_trainer", path)
    upstream = importlib.util.module_from_spec(specification);specification.loader.exec_module(upstream)
    rng = np.random.default_rng(91)
    train = rng.normal(size=(ntrain, nfeatures)).astype(np.float32);test = rng.normal(size=(24, nfeatures)).astype(np.float32)
    y = rng.normal(size=(ntrain, 50)).astype(np.float64);yt = rng.normal(size=(24, 50)).astype(np.float64)
    genes = [f"SYNTHETIC_G{k}" for k in range(50)]
    with threadpool_limits(limits=2):
        model = fit_official_ridge(train, y)
        pipe = Pipeline([("scaler", StandardScaler()), ("PCA", PCA(n_components=256, random_state=1))])
        x = np.asarray(pipe.fit_transform(train), np.float32);xt = np.asarray(pipe.transform(test), np.float32)
        reference, dump = upstream.train_test_reg(x, xt, y, yt, random_state=1, genes=genes, method="ridge")
        predicted = model.predict(test)
    assert np.array_equal(predicted, dump["preds_all"])
    assert upstream_raw_metrics(yt, predicted, genes) == reference
    assert model.pipeline["PCA"]._fit_svd_solver == solver

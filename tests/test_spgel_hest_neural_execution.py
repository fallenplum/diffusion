"""Synthetic execution: real PCA reconstruction, 12 fits, 36 saved forwards."""
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.spgel_hest.assets import canonical_hash, read_json, sha256
from src.spgel_hest_neural import neural_execution as execution
from src.spgel_hest_neural.models import ARMS

CONFIG = Path(__file__).parents[1] / "configs/spgel_hest_neural/pilot_v1.json"


@pytest.mark.parametrize("shape,solver", [((260, 280), "full"), ((550, 330), "randomized")])
def test_reconstruction_checks_actual_saved_parameters_and_preserves_train_us(shape, solver):
    rng = np.random.default_rng(18)
    x = rng.normal(size=shape).astype(np.float32)
    test = rng.normal(size=(19, shape[1])).astype(np.float32)
    pipeline = Pipeline([("scaler", StandardScaler()), ("PCA", PCA(256, random_state=1))])
    with threadpool_limits(limits=2):
        original_us = pipeline.fit_transform(x).astype(np.float32)
    train_us, test_x, audit = execution.reconstruct_train_us(pipeline, x, test)
    assert np.array_equal(train_us, original_us)
    with threadpool_limits(limits=2):
        expected_test = pipeline.transform(test).astype(np.float32)
    assert np.array_equal(test_x, expected_test)
    assert all(r["exact_array_equal"] for r in audit["parameters"].values())
    assert audit["actual_PCA_solver"] == solver
    assert audit["new_PCA_fits"] == 1 and not audit["old_train_US_bitwise_comparison_claimed"]
    pipeline["PCA"].components_[0, 0] += .1
    with pytest.raises(ValueError, match="NOT_EXACTLY_EQUAL_components"):
        execution.reconstruct_train_us(pipeline, x, test)


def test_real_zero_optimizer_profile_cannot_instantiate_optimizer(monkeypatch):
    import torch
    rng = np.random.default_rng(19)
    cache = dict(train_inputs=rng.normal(size=(20, 256)).astype(np.float32),
        train_targets=rng.normal(size=(20, 50)), train_coords=np.c_[np.arange(20), np.zeros(20)],
        train_specimen_index=np.r_[np.zeros(10, int), np.ones(10, int)])
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("PROFILE_MUST_NOT_CREATE_OPTIMIZER")))
    result = execution.zero_optimizer_profile(cache)
    assert result["optimizer_steps"] == result["new_neural_fits"] == 0
    assert len(result["units"]) == 3
    assert all(len(r["gradient_norms"]) > 0 and r["optimizer_instances"] == 0 for r in result["units"])
    identity = next(r for r in result["units"] if r["arm"] == "IDENTITY_GNN")
    assert identity["gradient_norms"]["neighbour_projection.weight"] > 0


def _synthetic_runtime(tmp_path, monkeypatch):
    """Mock only asset IO/certificates; PCA, models, graph and solver are real."""
    rng = np.random.default_rng(20)
    train_x = rng.normal(size=(258, 280)).astype(np.float32)
    test_x = rng.normal(size=(21, 280)).astype(np.float32)
    pipe = Pipeline([("scaler", StandardScaler()), ("PCA", PCA(256, random_state=1))])
    with threadpool_limits(limits=2):
        pipe.fit_transform(train_x)
    w = rng.normal(size=(280, 50)) * .05
    partition_cache = {}
    for task in ("COAD", "READ"):
        for fold in (0, 1):
            for split, x in (("train", train_x), ("test", test_x)):
                donor = None if task == "COAD" else ("A" if split == "train" else "B")
                partition_cache[(task, fold, split)] = SimpleNamespace(features=x,
                    targets=np.asarray(x @ w, dtype=np.float64), barcodes=np.array([f"B{k}" for k in range(len(x))]),
                    specimen_index=np.zeros(len(x), dtype=np.int64), specimen_ids=[task + split],
                    donor_codes=[donor], genes=[f"G{k}" for k in range(50)],
                    ordering_sha256=canonical_hash([task, fold, split]))
    assets_path = tmp_path / "synthetic_assets.json"
    assets_path.write_text("{}", encoding="utf-8")
    ridge_private = tmp_path / "old_synthetic_ridge"
    ridge_private.mkdir()
    old_lock = dict(internal_contract_sha256="c" * 64, source_files={})
    (ridge_private / "HEST_RUN_LOCK.json").write_text("{}", encoding="utf-8")
    for task in ("COAD", "READ"):
        for fold in (0, 1):
            folder = ridge_private / "cells" / execution.ridge_cell_name(task, fold)
            folder.mkdir(parents=True)
            # No supervised estimator is provided at all. Accessing one or any
            # old prediction path would fail this end-to-end test.
            joblib.dump(SimpleNamespace(pipeline=pipe), folder / "model.private.joblib")
            execution.write_json(folder / "COMPLETE.json", dict(status="COMPLETE_ONE_OFFICIAL_RIDGE_FIT",
                internal_contract_sha256=old_lock["internal_contract_sha256"],
                train_ordering_sha256=partition_cache[(task, fold, "train")].ordering_sha256,
                test_ordering_sha256=partition_cache[(task, fold, "test")].ordering_sha256,
                model_sha256=sha256(folder / "model.private.joblib")))
    monkeypatch.setattr(execution, "validate_ridge_lock", lambda *args: (old_lock, {}, {}))
    monkeypatch.setattr(execution, "validate_assets", lambda *args: {})
    monkeypatch.setattr(execution, "load_partition", lambda assets, task, fold, split: partition_cache[(task, fold, split)])
    monkeypatch.setattr(execution, "partition_coordinates", lambda assets, task, part:
        np.c_[np.arange(len(part.features), dtype=np.int64), np.zeros(len(part.features), dtype=np.int64)])
    private = tmp_path / "new_synthetic_neural"
    private.mkdir()
    monkeypatch.setattr(execution, "private_directory", lambda path: Path(path))
    return assets_path, ridge_private, private


def test_full_actual_synthetic_four_pca_twelve_fits_36_replays_and_tamper_guard(tmp_path, monkeypatch):
    assets, old, private = _synthetic_runtime(tmp_path, monkeypatch)
    lock = execution.make_lock(assets, CONFIG, private, old)
    assert lock["actual_preparation_new_PCA_fits"] == 4
    assert lock["actual_preparation_new_neural_fits"] == lock["actual_preparation_new_Ridge_fits"] == 0
    assert len(lock["partitions"]) == 4
    for task in ("COAD", "READ"):
        for fold in (0, 1):
            done = [execution.fit_cell(assets, CONFIG, private, task, fold, arm) for arm in ARMS]
            assert len({r["input_cache_sha256"] for r in done}) == 1
            assert len({r["initial_state_tensor_sha256"]["input_projection.weight"] for r in done}) == 1
            assert len({r["initial_state_tensor_sha256"]["output_projection.weight"] for r in done}) == 1
            assert done[1]["initial_state_tensor_sha256"] == done[2]["initial_state_tensor_sha256"]
            assert done[1]["parameters"] == done[2]["parameters"] > done[0]["parameters"]
            assert all(r["actual_optimizer_updates"] == 40 and not r["test_RNA_in_predict"] for r in done)
    replay = execution.replay_all(assets, CONFIG, private)
    assert replay["passed_states"] == 36 and len(replay["units"]) == 36
    assert all(r["maximum_prediction_error"] == 0 for r in replay["units"])
    report = execution.report(assets, CONFIG, private, tmp_path / "anonymous.json")
    assert report["ledger"] == dict(new_neural_fits=12, actual_optimizer_updates=480,
        actual_saved_state_replays=36, new_unsupervised_PCA_reconstructions=4,
        new_scaler_fits=0, new_Ridge_fits=0, old_source_weight_loads=0, automatic_retry_fits=0)
    assert not report["scope"]["fixed40_budget_proves_convergence"]
    assert len(report["per_cell"]) == 12
    replay_path = private / "NEURAL_REPLAY_AUDIT.json"
    original_replay = replay_path.read_bytes()
    for field, bad in (("actual_saved_state_loaded", False), ("new_PCA_fits", 1),
                       ("state_sha256", "a" * 64), ("maximum_prediction_error", 1.)):
        forged = read_json(replay_path)
        forged["units"][0][field] = bad
        replay_path.write_text(__import__("json").dumps(forged), encoding="utf-8")
        with pytest.raises(ValueError, match="REPLAY_EXECUTION_FLAGS_OR_ARTIFACT_BINDING_CHANGED"):
            execution.report(assets, CONFIG, private, tmp_path / "forged_must_not_publish.json")
        replay_path.write_bytes(original_replay)
    folder = private / "cells" / execution.cell_name("COAD", 0, "MLP")
    artifact = folder / "predictions_epoch10.private.npz"
    original = artifact.read_bytes();artifact.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="SNAPSHOT_ARTIFACT_CHANGED"):
        execution.report(assets, CONFIG, private, tmp_path / "must_not_publish.json")
    artifact.write_bytes(original)
    with pytest.raises(ValueError, match="NO_AUTOMATIC_RETRY"):
        execution.fit_cell(assets, CONFIG, private, "COAD", 0, "MLP")
    with pytest.raises(ValueError, match="NO_AUTOMATIC_RETRY"):
        execution.make_lock(assets, CONFIG, private, old)
    assert not (tmp_path / "must_not_publish.json").exists()


def test_shared_input_cache_binding_fails_closed(tmp_path):
    folder = tmp_path / "inputs" / "COAD_fold0_seed1"
    folder.mkdir(parents=True)
    (folder / "inputs.private.npz").write_bytes(b"not the bound cache")
    (folder / "UNSUPERVISED_CACHE.private.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHARED_CACHE_CHANGED"):
        execution._load_cache(tmp_path, dict(partition="COAD_fold0_seed1",
            input_cache_sha256="a" * 64, cache_receipt_sha256="b" * 64))


def test_no_old_source_or_ridge_supervised_paths_and_private_namespace_guard():
    source = (Path(__file__).parents[1] / "src/spgel_hest_neural/neural_execution.py").read_text(encoding="utf-8")
    assert ".estimator" not in source and ".coef_" not in source
    assert "fit_official_ridge" not in source and "from ..spgel." not in source
    assert "predictions.private.npz" not in source  # older Ridge file name
    with pytest.raises(ValueError, match="NEW_NEURAL_PRIVATE_NAMESPACE"):
        execution.private_directory(execution.REPO / "tmp" / "forbidden_neural_outputs")

"""Four image-only input caches, twelve fixed fits and 36 actual state replays.

The older Ridge object is deserialized solely to retrieve its unsupervised
pipeline. Its supervised estimator, coefficients and predictions are never
accessed. No older spatial-source model is imported or loaded.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import time
import traceback

import h5py
import joblib
import numpy as np
import psutil
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import torch

from ..spgel_hest.assets import (bound_file, canonical_hash, load_partition,
    read_json, sha256, validate_assets)
from ..spgel_hest.execution import (cell_name as ridge_cell_name, identity_status,
    validate_lock as validate_ridge_lock, write_json)
from ..spgel_hest.official import (aggregate_upstream_round4, json_safe_nonfinite,
    official_round4, require, supplementary_metrics, upstream_raw_metrics)
from .graph import build_graph
from .models import ARMS, build_model, parameter_count
from .recipe import REPO, recipe, runtime, source_hashes

SNAPSHOTS = (10, 20, 40)


def private_directory(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(REPO)
        and not any((p / ".git").exists() for p in (path, *path.parents))
        and not (path / "SOURCE_RUN_LOCK.json").exists()
        and not (path / "HEST_RUN_LOCK.json").exists(),
        "NEW_NEURAL_PRIVATE_NAMESPACE_REQUIRED")
    path.mkdir(parents=True, exist_ok=True, mode=0o777 if os.name == "nt" else 0o700)
    return path


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def partition_name(dataset, fold):
    require(dataset in ("COAD", "READ") and fold in (0, 1), "UNREGISTERED_NEURAL_PARTITION")
    return f"{dataset}_fold{fold}_seed1"


def cell_name(dataset, fold, arm):
    require(arm in ARMS, "UNREGISTERED_NEURAL_ARM")
    return partition_name(dataset, fold) + "_" + arm


def partition_coordinates(assets, dataset, partition):
    """Only coordinates aligned to the already qualified feature barcode order."""
    values = []
    for index, specimen in enumerate(partition.specimen_ids):
        spec = assets["tasks"][dataset]["samples"][specimen]
        with h5py.File(bound_file(spec["features"]), "r") as handle:
            barcodes = handle["barcodes"][:].flatten().astype(str)
            coords = handle["coords"][:]
        selected = partition.specimen_index == index
        require(np.array_equal(barcodes, partition.barcodes[selected])
            and coords.shape == (int(selected.sum()), 2) and np.isfinite(coords).all(),
            "NEURAL_COORDINATE_BARCODE_ALIGNMENT_CHANGED")
        values.append(coords)
    return np.concatenate(values)


def reconstruct_train_us(pipeline, train_features, test_features):
    """One new unsupervised PCA fit, no new scaler/Ridge/NN fit.

    Original fit_transform U*S was not persisted by the Ridge implementation;
    this reconstruction verifies its PCA basis exactly, not an unavailable old
    train tensor. All new arms consume the one newly persisted train U*S.
    """
    require(list(pipeline.named_steps) == ["scaler", "PCA"], "SAVED_UNSUPERVISED_PIPELINE_CHANGED")
    scaler, saved = pipeline["scaler"], pipeline["PCA"]
    require(saved.n_components == 256 and saved.random_state == 1
        and saved.svd_solver == "auto" and not saved.whiten,
        "SAVED_OFFICIAL_PCA256_RECIPE_CHANGED")
    fresh = clone(saved)
    with threadpool_limits(limits=2):
        scaled = scaler.transform(train_features)
        train_x = np.asarray(fresh.fit_transform(scaled), dtype=np.float32)
        test_x = np.asarray(pipeline.transform(test_features), dtype=np.float32)
    parity = {}
    for field in ("mean_", "components_", "singular_values_"):
        value, expected = getattr(fresh, field), getattr(saved, field)
        equal = np.array_equal(value, expected)
        parity[field] = dict(exact_array_equal=bool(equal),
            reconstructed_sha256=array_sha256(value), saved_sha256=array_sha256(expected))
        require(equal, "RECONSTRUCTED_PCA_PARAMETERS_NOT_EXACTLY_EQUAL_" + field)
    require(train_x.shape == (len(train_features), 256)
        and test_x.shape == (len(test_features), 256)
        and np.isfinite(train_x).all() and np.isfinite(test_x).all(), "NEURAL_PCA_CACHE_NONFINITE_OR_SHAPE")
    return train_x, test_x, dict(parameters=parity,
        actual_PCA_solver=getattr(fresh, "_fit_svd_solver", None), new_PCA_fits=1,
        new_scaler_fits=0, new_Ridge_fits=0, new_neural_fits=0,
        old_train_US_persisted=False, old_train_US_bitwise_comparison_claimed=False,
        test_uses_saved_pipeline_transform=True)


def _load_cache(private, expected):
    folder = Path(private) / "inputs" / expected["partition"]
    require(sha256(folder / "inputs.private.npz") == expected["input_cache_sha256"]
        and sha256(folder / "UNSUPERVISED_CACHE.private.json") == expected["cache_receipt_sha256"],
        "NEURAL_SHARED_CACHE_CHANGED")
    with np.load(folder / "inputs.private.npz", allow_pickle=False) as data:
        cache = {name: data[name] for name in data.files}
    receipt = read_json(folder / "UNSUPERVISED_CACHE.private.json")
    require({name: array_sha256(value) for name, value in cache.items()} == receipt["array_sha256"],
        "NEURAL_CACHED_TENSOR_CHANGED")
    return cache


def _tensors_and_graph(cache, split, arm):
    x = torch.from_numpy(np.ascontiguousarray(cache[split + "_inputs"]))
    graph = None if arm == "MLP" else build_graph(cache[split + "_coords"],
        cache[split + "_specimen_index"], identity=(arm == "IDENTITY_GNN"))
    adjacency = None if graph is None else graph.torch_adjacency()
    return x, graph, adjacency


def _finite_gradients(model):
    values = {}
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
            "MISSING_OR_NONFINITE_TRAIN_GRADIENT")
        values[name] = float(parameter.grad.norm().item())
    return values


def zero_optimizer_profile(cache):
    """Actual full-batch forward/backward, no optimizer object or step."""
    torch.set_num_threads(2)
    target = torch.from_numpy(np.asarray(cache["train_targets"], dtype=np.float32))
    rows = []
    available_before = int(psutil.virtual_memory().available)
    initial_RSS = int(psutil.Process().memory_info().rss)
    for arm in ARMS:
        start, rss_before = time.monotonic(), psutil.Process().memory_info().rss
        model = build_model(arm, seed=1)
        initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        x, graph, adjacency = _tensors_and_graph(cache, "train", arm)
        with threadpool_limits(limits=2):
            prediction = model(x, adjacency)
            loss = torch.nn.functional.mse_loss(prediction, target)
            require(torch.isfinite(loss).item(), "NONFINITE_ZERO_OPTIMIZER_TRAIN_PROFILE")
            loss.backward()
        gradients = _finite_gradients(model)
        require(all(torch.equal(value, model.state_dict()[name]) for name, value in initial_state.items()),
            "ZERO_OPTIMIZER_PROFILE_CHANGED_PARAMETERS")
        memory = psutil.Process().memory_info()
        rows.append(dict(arm=arm, train_rows=len(x), train_loss=float(loss.detach()),
            parameters=parameter_count(model), elapsed_seconds=time.monotonic() - start,
            process_RSS_before_bytes=rss_before, process_RSS_after_backward_bytes=memory.rss,
            process_lifetime_peak_RSS_bytes=int(getattr(memory, "peak_wset", memory.rss)),
            gradient_norms=gradients, graph_audit=None if graph is None else graph.audit,
            model_state_exactly_unchanged=True,
            test_labels_used=False, optimizer_instances=0, optimizer_steps=0, new_fits=0))
        model.zero_grad(set_to_none=True)
    # This gate uses measured process RSS conservatively; it is a point-in-time
    # two-worker feasibility check, not a guarantee against unrelated RAM users.
    worker_RSS = max(r["process_lifetime_peak_RSS_bytes"] for r in rows)
    required = 2 * worker_RSS + 1024**3
    require(available_before + initial_RSS >= required, "INSUFFICIENT_RAM_FOR_TWO_FIXED_CPU_WORKERS")
    return dict(status="PASS_REAL_TRAIN_FULL_BATCH_FORWARD_BACKWARD_ZERO_OPTIMIZER",
        units=rows, zero_optimizer_profiles=len(rows), new_neural_fits=0,
        optimizer_steps=0, profile_partition_selected_only_by_train_row_count=True,
        available_RAM_before_bytes=available_before, initial_process_RSS_bytes=initial_RSS,
        maximum_observed_worker_RSS_bytes=worker_RSS, conservative_two_worker_with_1GiB_reserve_bytes=required,
        point_in_time_two_worker_RAM_gate_passed=True,
        concurrent_unrelated_RAM_growth_guaranteed_safe=False)


def make_lock(assets_path, config_path, private, ridge_private):
    config = recipe(config_path)
    private = private_directory(private)
    require(not (private / "PREPARATION_STARTED.private.json").exists()
        and not (private / "HEST_NEURAL_RUN_LOCK.json").exists(), "NEURAL_PREPARATION_NO_AUTOMATIC_RETRY")
    before = dict(source_files=source_hashes(), config_sha256=sha256(config_path),
        assets_sha256=sha256(assets_path), runtime=runtime())
    write_json(private / "PREPARATION_STARTED.private.json", dict(status="STARTED_ZERO_NEURAL_FITS",
        new_PCA_reconstructions_planned=4, new_scaler_fits=0, new_Ridge_fits=0, new_neural_fits=0,
        pre_preprocessing_scientific_contract=before))
    count = 0
    try:
        ridge_private = Path(ridge_private).resolve()
        old_lock, _, assets = validate_ridge_lock(assets_path,
            REPO / "configs/spgel_hest/ridge_v1.json", ridge_private)
        partitions = []
        for dataset in config["tasks"]:
            for fold in config["folds"]:
                train = load_partition(assets, dataset, fold, "train")
                test = load_partition(assets, dataset, fold, "test")
                old_folder = ridge_private / "cells" / ridge_cell_name(dataset, fold)
                old = read_json(old_folder / "COMPLETE.json")
                require(old["status"] == "COMPLETE_ONE_OFFICIAL_RIDGE_FIT"
                    and old["internal_contract_sha256"] == old_lock["internal_contract_sha256"]
                    and old["train_ordering_sha256"] == train.ordering_sha256
                    and old["test_ordering_sha256"] == test.ordering_sha256
                    and sha256(old_folder / "model.private.joblib") == old["model_sha256"],
                    "SAVED_RIDGE_UNSUPERVISED_INPUT_BINDING_CHANGED")
                # The serialized object contains the estimator; no supervised
                # estimator attribute or old prediction artifact is consumed.
                pipeline = joblib.load(old_folder / "model.private.joblib").pipeline
                count += 1
                train_x, test_x, parity = reconstruct_train_us(pipeline, train.features, test.features)
                name = partition_name(dataset, fold)
                folder = private / "inputs" / name
                folder.mkdir(parents=True, exist_ok=False)
                payload = dict(train_inputs=train_x, test_inputs=test_x,
                    train_targets=train.targets, test_targets=test.targets,
                    train_coords=partition_coordinates(assets, dataset, train),
                    test_coords=partition_coordinates(assets, dataset, test),
                    train_specimen_index=train.specimen_index, test_specimen_index=test.specimen_index,
                    train_barcodes=train.barcodes, test_barcodes=test.barcodes,
                    genes=np.asarray(train.genes))
                with (folder / "inputs.private.npz").open("xb") as stream:
                    np.savez_compressed(stream, **payload)
                joblib.dump(pipeline, folder / "unsupervised_pipeline.private.joblib", compress=0)
                receipt = dict(schema="spgel_hest_neural_shared_unsupervised_cache_v1", status="PASS",
                    dataset=dataset, fold=fold, array_sha256={k: array_sha256(v) for k, v in payload.items()},
                    input_cache_sha256=sha256(folder / "inputs.private.npz"), PCA_reconstruction=parity,
                    saved_pipeline_sha256=sha256(folder / "unsupervised_pipeline.private.joblib"),
                    original_Ridge_model_sha256=old["model_sha256"],
                    original_Ridge_COMPLETE_sha256=sha256(old_folder / "COMPLETE.json"),
                    serialized_Ridge_object_physically_deserialized=True,
                    Ridge_estimator_coefficients_or_predictions_accessed=False,
                    train_ordering_sha256=train.ordering_sha256, test_ordering_sha256=test.ordering_sha256)
                write_json(folder / "UNSUPERVISED_CACHE.private.json", receipt)
                partitions.append(dict(partition=name, dataset=dataset, fold=fold,
                    train_rows=len(train_x), test_rows=len(test_x), genes=list(train.genes),
                    input_cache_sha256=receipt["input_cache_sha256"],
                    cache_receipt_sha256=sha256(folder / "UNSUPERVISED_CACHE.private.json"),
                    train_ordering_sha256=train.ordering_sha256, test_ordering_sha256=test.ordering_sha256,
                    donor_separation_status=identity_status(dataset, train, test),
                    train_specimens=len(train.specimen_ids), test_specimens=len(test.specimen_ids)))
        largest = max(partitions, key=lambda row: row["train_rows"])
        profile = zero_optimizer_profile(_load_cache(private, largest))
        profile["profile_partition"] = largest["partition"]
        write_json(private / "ZERO_OPTIMIZER_PROFILE.private.json", profile)
        require(before == dict(source_files=source_hashes(), config_sha256=sha256(config_path),
            assets_sha256=sha256(assets_path), runtime=runtime()),
            "SCIENTIFIC_CONTRACT_CHANGED_DURING_UNSUPERVISED_PREPARATION")
        require(all(sha256(REPO / name) == value for name, value in old_lock["source_files"].items()),
            "INHERITED_RIDGE_SCIENCE_CHANGED_DURING_PREPARATION")
        lock = dict(schema="spgel_hest_neural_run_lock_v1", status="LOCKED_BEFORE_TWELVE_NEURAL_FITS",
            **before,
            inherited_Ridge_source_files=deepcopy(old_lock["source_files"]),
            original_Ridge_lock_sha256=sha256(ridge_private / "HEST_RUN_LOCK.json"),
            original_Ridge_internal_contract_sha256=old_lock["internal_contract_sha256"],
            zero_optimizer_profile_sha256=sha256(private / "ZERO_OPTIMIZER_PROFILE.private.json"),
            partitions=partitions, budget=deepcopy(config["budget"]),
            actual_preparation_new_PCA_fits=count, actual_preparation_new_scaler_fits=0,
            actual_preparation_new_Ridge_fits=0, actual_preparation_new_neural_fits=0,
            old_source_weights_loaded=False, DSS_or_PFI_or_R8_read=False,
            snapshot_epochs_for_diagnostics_only=list(SNAPSHOTS), primary_epoch=40,
            fixed_40_epochs_claimed_converged=False)
        lock["internal_contract_sha256"] = canonical_hash(lock)
        write_json(private / "HEST_NEURAL_RUN_LOCK.json", lock)
        return lock
    except BaseException as error:
        write_json(private / "PREPARATION_ERROR.private.json", dict(status="FAILED_PREPARATION_NO_AUTOMATIC_RETRY",
            failure_type=type(error).__name__, attempted_new_PCA_reconstructions=count,
            new_neural_fits=0, new_Ridge_fits=0))
        (private / "preparation_error.private.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def validate_lock(assets_path, config_path, private):
    config = recipe(config_path)
    lock = read_json(Path(private) / "HEST_NEURAL_RUN_LOCK.json")
    require(lock.get("schema") == "spgel_hest_neural_run_lock_v1"
        and lock.get("status") == "LOCKED_BEFORE_TWELVE_NEURAL_FITS"
        and lock["internal_contract_sha256"] == canonical_hash({k: v for k, v in lock.items()
            if k != "internal_contract_sha256"}) and lock["source_files"] == source_hashes()
        and lock["config_sha256"] == sha256(config_path) and lock["assets_sha256"] == sha256(assets_path)
        and lock["runtime"] == runtime() and lock["budget"] == config["budget"]
        and lock["actual_preparation_new_PCA_fits"] == 4
        and lock["actual_preparation_new_scaler_fits"] == lock["actual_preparation_new_Ridge_fits"]
            == lock["actual_preparation_new_neural_fits"] == 0,
        "NEURAL_LOCK_CODE_RUNTIME_ASSET_OR_BUDGET_CHANGED")
    require({(r["dataset"], r["fold"]) for r in lock["partitions"]}
        == {(d, f) for d in ("COAD", "READ") for f in (0, 1)} and len(lock["partitions"]) == 4,
        "EXACT_FOUR_NEURAL_INPUT_PARTITIONS_REQUIRED")
    require(sha256(Path(private) / "ZERO_OPTIMIZER_PROFILE.private.json") == lock["zero_optimizer_profile_sha256"],
        "ZERO_OPTIMIZER_QUALIFICATION_CHANGED")
    for name, expected in lock["inherited_Ridge_source_files"].items():
        require(sha256(REPO / name) == expected, "INHERITED_RIDGE_SCIENCE_CHANGED")
    return lock, config, validate_assets(assets_path)


def _expected(lock, dataset, fold):
    name = partition_name(dataset, fold)
    return next(row for row in lock["partitions"] if row["partition"] == name)


def _snapshot_paths(folder, epoch):
    return (folder / f"state_epoch{epoch:02d}.private.pt", folder / f"predictions_epoch{epoch:02d}.private.npz")


def fit_cell(assets_path, config_path, private, dataset, fold, arm):
    private = private_directory(private)
    lock, config, _ = validate_lock(assets_path, config_path, private)
    expected = _expected(lock, dataset, fold)
    name = cell_name(dataset, fold, arm)
    folder = private / "cells" / name
    require(not folder.exists(), "EXISTING_NEURAL_CELL_NO_AUTOMATIC_RETRY")
    folder.mkdir(parents=True)
    started = time.monotonic()
    steps = 0
    try:
        torch.set_num_threads(2)
        cache = _load_cache(private, expected)
        model = build_model(arm, seed=1)
        initial = {k: array_sha256(v.detach().numpy()) for k, v in model.state_dict().items()}
        train_x, train_graph, train_adj = _tensors_and_graph(cache, "train", arm)
        test_x, test_graph, test_adj = _tensors_and_graph(cache, "test", arm)
        # Only train targets are converted to an input for autograd.
        train_y = torch.from_numpy(np.asarray(cache["train_targets"], dtype=np.float32))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"])
        curves, snapshots = [], []
        with threadpool_limits(limits=2):
            for epoch in range(1, 41):
                model.train();optimizer.zero_grad(set_to_none=True)
                prediction = model(train_x, train_adj)
                loss = torch.nn.functional.mse_loss(prediction, train_y)
                require(torch.isfinite(loss).item(), "NONFINITE_NEURAL_TRAIN_LOSS")
                loss.backward()
                gradients = _finite_gradients(model)
                optimizer.step();steps += 1
                require(all(torch.isfinite(p).all().item() for p in model.parameters()), "NONFINITE_NEURAL_PARAMETERS")
                curves.append(dict(epoch=epoch, pre_update_train_MSE=float(loss.detach()),
                    gradient_norms=gradients, actual_optimizer_updates=steps))
                if epoch not in SNAPSHOTS:
                    continue
                model.eval()
                with torch.inference_mode():
                    train_prediction = model(train_x, train_adj).numpy()
                    test_prediction = model(test_x, test_adj).numpy()
                require(np.isfinite(train_prediction).all() and np.isfinite(test_prediction).all(),
                    "NONFINITE_NEURAL_SNAPSHOT_PREDICTIONS")
                state_path, prediction_path = _snapshot_paths(folder, epoch)
                torch.save(dict(schema="spgel_hest_neural_state_v1", arm=arm, epoch=epoch, seed=1,
                    state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    input_cache_sha256=expected["input_cache_sha256"],
                    internal_contract_sha256=lock["internal_contract_sha256"]), state_path)
                with prediction_path.open("xb") as stream:
                    np.savez_compressed(stream, train_predictions=train_prediction, test_predictions=test_prediction)
                # Evaluation is after prediction and never enters model/optimizer.
                raw = upstream_raw_metrics(cache["test_targets"], test_prediction, expected["genes"])
                snapshots.append(dict(epoch=epoch, state_sha256=sha256(state_path),
                    predictions_sha256=sha256(prediction_path),
                    post_update_train_MSE=float(np.mean((train_prediction - cache["train_targets"])**2)),
                    full_precision_metrics=json_safe_nonfinite(raw),
                    official_rounded4_split_metrics=json_safe_nonfinite(official_round4(raw)),
                    supplementary=supplementary_metrics(cache["test_targets"], test_prediction)))
        result = dict(schema="spgel_hest_neural_cell_v1", status="COMPLETE_ONE_FIXED40_NEURAL_FIT",
            dataset=dataset, fold=fold, seed=1, arm=arm, new_neural_fits=1,
            actual_optimizer_updates=steps, actual_epochs=40, new_Ridge_fits=0,
            new_scaler_fits=0, new_PCA_fits=0, parameters=parameter_count(model),
            initial_state_tensor_sha256=initial, learning_curve=curves, snapshots=snapshots,
            train_graph=None if train_graph is None else train_graph.audit,
            test_graph=None if test_graph is None else test_graph.audit,
            train_rows=len(train_x), test_rows=len(test_x),
            input_cache_sha256=expected["input_cache_sha256"],
            internal_contract_sha256=lock["internal_contract_sha256"],
            elapsed_seconds=time.monotonic() - started, train_loss_only=True,
            test_RNA_in_predict=False, test_labels_select_epoch=False,
            source_pretrained_weights_loaded=False, fixed40_budget_proves_convergence=False)
        write_json(folder / "COMPLETE.json", result)
        return result
    except BaseException as error:
        write_json(folder / "ERROR_RECEIPT.private.json", dict(status="FAILED_NEURAL_CELL_NO_AUTOMATIC_RETRY",
            failure_type=type(error).__name__, actual_optimizer_updates=steps,
            failed_fit_attempts=1, new_Ridge_fits=0, automatic_retry_permitted=False))
        (folder / "error.private.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _validate_done(folder, done, lock, expected, arm):
    require(done["status"] == "COMPLETE_ONE_FIXED40_NEURAL_FIT"
        and done["arm"] == arm and done["seed"] == 1
        and done["internal_contract_sha256"] == lock["internal_contract_sha256"]
        and done["input_cache_sha256"] == expected["input_cache_sha256"]
        and done["actual_epochs"] == done["actual_optimizer_updates"] == 40
        and len(done["learning_curve"]) == 40
        and [r["epoch"] for r in done["snapshots"]] == list(SNAPSHOTS),
        "NEURAL_COMPLETE_BINDING_OR_FIXED_BUDGET_CHANGED")
    for snapshot in done["snapshots"]:
        state, predictions = _snapshot_paths(folder, snapshot["epoch"])
        require(sha256(state) == snapshot["state_sha256"]
            and sha256(predictions) == snapshot["predictions_sha256"], "NEURAL_SNAPSHOT_ARTIFACT_CHANGED")


def replay_cell(assets_path, config_path, private, dataset, fold, arm):
    lock, _, _ = validate_lock(assets_path, config_path, private)
    expected = _expected(lock, dataset, fold)
    folder = Path(private) / "cells" / cell_name(dataset, fold, arm)
    done = read_json(folder / "COMPLETE.json")
    _validate_done(folder, done, lock, expected, arm)
    cache = _load_cache(private, expected)
    torch.set_num_threads(2)
    train_x, _, train_adj = _tensors_and_graph(cache, "train", arm)
    test_x, _, test_adj = _tensors_and_graph(cache, "test", arm)
    rows = []
    with threadpool_limits(limits=2):
        for snapshot in done["snapshots"]:
            epoch = snapshot["epoch"]
            state_path, prediction_path = _snapshot_paths(folder, epoch)
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            require(state["arm"] == arm and state["epoch"] == epoch and state["seed"] == 1
                and state["input_cache_sha256"] == expected["input_cache_sha256"]
                and state["internal_contract_sha256"] == lock["internal_contract_sha256"],
                "NEURAL_SAVED_STATE_BINDING_CHANGED")
            model = build_model(arm, seed=1)
            model.load_state_dict(state["state_dict"], strict=True);model.eval()
            with torch.inference_mode():
                train_prediction = model(train_x, train_adj).numpy()
                test_prediction = model(test_x, test_adj).numpy()
            with np.load(prediction_path, allow_pickle=False) as stored:
                error = max(float(np.max(np.abs(train_prediction - stored["train_predictions"]), initial=0.)),
                    float(np.max(np.abs(test_prediction - stored["test_predictions"]), initial=0.)))
            require(error <= 1e-6, "NEURAL_ACTUAL_STATE_REPLAY_PREDICTION_ERROR")
            raw = json_safe_nonfinite(upstream_raw_metrics(cache["test_targets"], test_prediction, expected["genes"]))
            require(raw == snapshot["full_precision_metrics"], "NEURAL_REPLAY_METRICS_CHANGED")
            rows.append(dict(dataset=dataset, fold=fold, seed=1, arm=arm, epoch=epoch, status="PASS",
                actual_saved_state_loaded=True, actual_train_and_test_forward_replayed=True,
                maximum_prediction_error=error, new_neural_fits=0, new_Ridge_fits=0, new_PCA_fits=0,
                COMPLETE_sha256=sha256(folder / "COMPLETE.json"), state_sha256=snapshot["state_sha256"],
                predictions_sha256=snapshot["predictions_sha256"], input_cache_sha256=expected["input_cache_sha256"]))
    return rows


def replay_all(assets_path, config_path, private):
    lock, config, _ = validate_lock(assets_path, config_path, private)
    output = Path(private) / "NEURAL_REPLAY_AUDIT.json"
    require(not output.exists(), "EXISTING_NEURAL_REPLAY_PRESERVED")
    rows = [r for dataset in config["tasks"] for fold in config["folds"] for arm in ARMS
        for r in replay_cell(assets_path, config_path, private, dataset, fold, arm)]
    require(len(rows) == 36, "EXACT_THIRTY_SIX_ACTUAL_STATE_REPLAYS_REQUIRED")
    audit = dict(schema="spgel_hest_neural_replay_v1", status="PASS_36_ACTUAL_SAVED_STATE_REPLAYS",
        internal_contract_sha256=lock["internal_contract_sha256"], units=rows,
        passed_states=36, passed_fits=12, new_neural_fits=0, new_Ridge_fits=0, new_PCA_fits=0,
        optimizer_updates=0, model_or_epoch_selection_performed=False)
    write_json(output, audit)
    return audit


def report(assets_path, config_path, private, output):
    lock, config, _ = validate_lock(assets_path, config_path, private)
    replay_path = Path(private) / "NEURAL_REPLAY_AUDIT.json"
    replay = read_json(replay_path)
    require(replay["status"] == "PASS_36_ACTUAL_SAVED_STATE_REPLAYS"
        and replay["internal_contract_sha256"] == lock["internal_contract_sha256"]
        and replay["passed_states"] == len(replay["units"]) == 36, "NEURAL_REPORT_REQUIRES_ACTUAL_REPLAYS")
    rows = []
    for dataset in config["tasks"]:
        for fold in config["folds"]:
            expected = _expected(lock, dataset, fold)
            _load_cache(private, expected)
            for arm in ARMS:
                folder = Path(private) / "cells" / cell_name(dataset, fold, arm)
                done = read_json(folder / "COMPLETE.json")
                _validate_done(folder, done, lock, expected, arm)
                matching = [r for r in replay["units"] if (r["dataset"], r["fold"], r["arm"]) == (dataset, fold, arm)]
                require(len(matching) == 3 and {r["epoch"] for r in matching} == set(SNAPSHOTS)
                    and all(r["COMPLETE_sha256"] == sha256(folder / "COMPLETE.json")
                        and r["status"] == "PASS" for r in matching), "NEURAL_ARTIFACT_CHANGED_AFTER_REPLAY")
                for replayed in matching:
                    snapshot = next(s for s in done["snapshots"] if s["epoch"] == replayed["epoch"])
                    error = replayed["maximum_prediction_error"]
                    require(replayed["actual_saved_state_loaded"] is True
                        and replayed["actual_train_and_test_forward_replayed"] is True
                        and replayed["new_neural_fits"] == replayed["new_Ridge_fits"]
                            == replayed["new_PCA_fits"] == 0
                        and replayed["state_sha256"] == snapshot["state_sha256"]
                        and replayed["predictions_sha256"] == snapshot["predictions_sha256"]
                        and replayed["input_cache_sha256"] == expected["input_cache_sha256"]
                        and isinstance(error, (int, float)) and np.isfinite(error) and 0 <= error <= 1e-6,
                        "NEURAL_REPLAY_EXECUTION_FLAGS_OR_ARTIFACT_BINDING_CHANGED")
                rows.append(done)
    require(sum(r["actual_optimizer_updates"] for r in rows) == 480, "EXACT_480_OPTIMIZER_UPDATES_REQUIRED")
    # No private paths, specimen IDs, coordinates, labels, predictions or weights
    # leave the derived execution namespace.
    tasks = {}
    for dataset in config["tasks"]:
        tasks[dataset] = {}
        for arm in ARMS:
            cells = sorted([r for r in rows if r["dataset"] == dataset and r["arm"] == arm], key=lambda r: r["fold"])
            tasks[dataset][arm] = dict(primary_epoch=40, official_round4=aggregate_upstream_round4([
                r["snapshots"][-1]["official_rounded4_split_metrics"] for r in cells]),
                diagnostics_by_epoch={str(epoch): aggregate_upstream_round4([
                    next(s for s in r["snapshots"] if s["epoch"] == epoch)["official_rounded4_split_metrics"]
                    for r in cells]) for epoch in SNAPSHOTS},
                fold_support=[dict(fold=r["fold"], train_rows=r["train_rows"], test_rows=r["test_rows"],
                    parameters=r["parameters"], actual_epochs=r["actual_epochs"],
                    elapsed_seconds=r["elapsed_seconds"]) for r in cells])
    result = dict(schema="spgel_hest_neural_fixed12_report_v1", status="COMPLETE_12_FIXED40_FITS_AND_36_ACTUAL_REPLAYS",
        internal_contract_sha256=lock["internal_contract_sha256"], config_sha256=lock["config_sha256"],
        assets_sha256=lock["assets_sha256"], code_sha256=lock["source_files"], tasks=tasks,
        ledger=dict(new_neural_fits=12, actual_optimizer_updates=480, actual_saved_state_replays=36,
            new_unsupervised_PCA_reconstructions=4, new_scaler_fits=0, new_Ridge_fits=0,
            old_source_weight_loads=0, automatic_retry_fits=0),
        per_cell=[dict(dataset=r["dataset"], fold=r["fold"], arm=r["arm"], seed=1,
            parameters=r["parameters"], learning_curve=r["learning_curve"], snapshots=r["snapshots"],
            train_graph=r["train_graph"], test_graph=r["test_graph"]) for r in rows],
        scope=dict(primary_epoch=40, snapshot10_20_40_diagnostic_only=True,
            fixed40_budget_proves_convergence=False, test_labels_in_predict_or_epoch_selection=False,
            new_independent_cohort_claimed=False, COAD_preview_donor_identity="UNCONFIRMED",
            READ_donors_previously_used_in_source_development=True,
            graph_versus_identity_parameters_and_initialization_exactly_matched=True,
            MLP_has_fewer_parameters_than_graph=True,
            graph_versus_MLP_difference_exclusively_attributed_to_topology=False,
            native_pixel_k16_is_not_a_physical_or_cellular_neighbourhood_claim=True),
        replay_audit_sha256=sha256(replay_path), maximum_actual_replay_error=max(
            r["maximum_prediction_error"] for r in replay["units"]))
    write_json(output, result)
    return result

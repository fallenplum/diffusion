"""Exactly four separately locked HEST Ridge fits and actual model replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time
import traceback

import anndata
import h5py
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .assets import bound_file, canonical_hash, load_partition, read_json, sha256, validate_assets
from .official import (ALPHA, DATASET_REVISION, HEST_COMMIT, aggregate_upstream_round4,
    fit_official_ridge, json_safe_nonfinite, official_round4, require,
    supplementary_metrics, upstream_raw_metrics)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "src/spgel_hest/__init__.py", "src/spgel_hest/official.py", "src/spgel_hest/assets.py",
    "src/spgel_hest/execution.py", "scripts/spgel_hest_ridge.py",
    "configs/spgel_hest/ridge_v1.json", "tests/test_spgel_hest_official.py",
    "tests/test_spgel_hest_assets.py", "tests/test_spgel_hest_execution.py",
    "src/spgel_hest/NOTICE.md", "src/spgel_hest/HEST_LICENSE.md")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, anndata=anndata.__version__, h5py=h5py.__version__,
        pandas=pd.__version__, joblib=joblib.__version__, Ridge_default_tol=Ridge().get_params()["tol"])


def recipe(path):
    config = read_json(path)
    require(set(config) == {"schema", "status", "upstream_commit", "dataset_revision", "tasks", "folds",
        "seed", "logical_fits", "method", "input", "targets", "preprocessing", "ridge", "aggregation",
        "scope", "runtime"}, "UNREGISTERED_OFFICIAL_RECIPE_KEYS")
    require(config["schema"] == "spgel_hest_official_ridge_recipe_v1"
        and config["upstream_commit"] == HEST_COMMIT and config["dataset_revision"] == DATASET_REVISION
        and config["tasks"] == ["COAD", "READ"] and config["folds"] == [0, 1]
        and config["seed"] == 1 and config["logical_fits"] == 4 and config["method"] == "ridge",
        "EXACT_FOUR_OFFICIAL_RIDGE_CELLS_REQUIRED")
    require(config["input"] == dict(encoder="UNI2-h", dimension=1536, forward_precision="bfloat16", row_L2_normalized=False)
        and config["targets"] == dict(genes=50, selection="official_var_50genes_order", dtype="float64",
            normalization="log1p_only_after_official_gene_and_barcode_selection", library_normalization=False),
        "OFFICIAL_IMAGE_OR_TARGET_CONTRACT_CHANGED")
    require(config["preprocessing"] == dict(fit_scope="official_train_samples_only", steps=["StandardScaler", "PCA256"],
        PCA_solver="auto", PCA_random_state=1, after_PCA_dtype="float32", silent_dimension_reduction=False),
        "OFFICIAL_TRAIN_ONLY_PCA256_CONTRACT_CHANGED")
    require(config["ridge"] == dict(solver="lsqr", alpha=ALPHA, fit_intercept=False, max_iter=1000,
        random_state=1, tol="locked_sklearn_default"), "OFFICIAL_RIDGE_KWARGS_CHANGED")
    require(config["aggregation"] == dict(upstream_split_round_digits=4, fold_mean_uses_rounded_split_return=True,
        final_round_digits=4, full_precision_diagnostic_separate=True, undefined_correlations_imputed_or_dropped=False),
        "OFFICIAL_ROUND4_AGGREGATION_CHANGED")
    require(config["scope"] == dict(uses_old_source_pretrained_weights=False, DSS_or_PFI_or_R8_read=False,
        new_independent_cohort_claimed=False, MLP_or_GNN_in_this_lock=False, automatic_retry_fits=0)
        and config["runtime"] == dict(threads_per_process=2, ridge_device="CPU"),
        "OFFICIAL_SURVIVAL_SOURCE_ISOLATION_OR_BUDGET_CHANGED")
    return config


def private_directory(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(ROOT) and not any((p / ".git").exists() for p in (path, *path.parents))
        and not (path / "SOURCE_RUN_LOCK.json").exists(), "NEW_OFFICIAL_PRIVATE_NAMESPACE_REQUIRED")
    path.mkdir(parents=True, exist_ok=True, mode=0o777 if os.name == "nt" else 0o700)
    return path


def identity_status(dataset, train, test):
    left = {x for x in train.donor_codes if x is not None}
    right = {x for x in test.donor_codes if x is not None}
    require(not left.intersection(right), "KNOWN_DONOR_CROSSES_PINNED_OFFICIAL_SPLIT")
    unknown = any(x is None for x in [*train.donor_codes, *test.donor_codes])
    if dataset == "READ":
        require(not unknown and len(left.union(right)) == 2,
                "READ_TWO_KNOWN_DONORS_REQUIRED_BY_QUALIFIED_VERSION")
        return "READ_TWO_DONORS_VERIFIED_NO_CROSSING"
    require(unknown, "COAD_PREVIEW_DONOR_MUST_REMAIN_UNCONFIRMED_NOT_FABRICATED_UNIQUE_ID")
    return "COAD_IDENTITY_UNCONFIRMED"


def subgroup_metrics(partition, prediction, task):
    """Supplementary specimen/donor tables never replace official fold means."""
    known = sorted({s["donor_code"] for s in task["samples"].values()
                    if s.get("donor_code") is not None}, key=str)
    roster = sorted(task["samples"])
    specimen_rows, donor_rows = [], []
    for index, specimen in enumerate(partition.specimen_ids):
        selected = partition.specimen_index == index
        donor = partition.donor_codes[index]
        raw = upstream_raw_metrics(partition.targets[selected], prediction[selected], partition.genes)
        specimen_rows.append(dict(specimen_index=roster.index(specimen), spots_count=int(selected.sum()),
            donor_index=known.index(donor) if donor is not None else None,
            donor_identity_known=donor is not None,
            full_precision_Pearson_mean=json_safe_nonfinite(raw["pearson_mean"]),
            supplementary=supplementary_metrics(partition.targets[selected], prediction[selected])))
    for donor in known:
        specimens = [i for i, label in enumerate(partition.donor_codes) if label == donor]
        if not specimens:
            continue
        selected = np.isin(partition.specimen_index, specimens)
        raw = upstream_raw_metrics(partition.targets[selected], prediction[selected], partition.genes)
        donor_rows.append(dict(donor_index=known.index(donor), specimens_count=len(specimens),
            spots_count=int(selected.sum()), full_precision_Pearson_mean=json_safe_nonfinite(raw["pearson_mean"]),
            supplementary=supplementary_metrics(partition.targets[selected], prediction[selected])))
    return dict(per_specimen=specimen_rows, known_donors=donor_rows,
        unknown_donor_specimens_count=sum(x is None for x in partition.donor_codes),
        unknown_donors_not_fabricated_as_distinct_patients=True,
        official_mean_replaced_by_subgroup_mean=False)


def make_lock(assets_path, config_path, private):
    """Only shape/identity/hash forwards, no scaler/PCA/Ridge fitting."""
    config = recipe(config_path)
    private = private_directory(private)
    require(not (private / "HEST_RUN_LOCK.json").exists(), "EXISTING_OFFICIAL_LOCK_PRESERVED")
    assets = validate_assets(assets_path)
    rows = []
    for dataset in config["tasks"]:
        for fold in config["folds"]:
            train = load_partition(assets, dataset, fold, "train")
            test = load_partition(assets, dataset, fold, "test")
            require(min(train.features.shape) >= 256 and train.genes == test.genes,
                    "OFFICIAL_PCA256_NOT_FEASIBLE_OR_GENE_ORDER_CHANGED")
            rows.append(dict(dataset=dataset, fold=fold, seed=1, train_rows=len(train.features),
                test_rows=len(test.features), train_ordering_sha256=train.ordering_sha256,
                test_ordering_sha256=test.ordering_sha256,
                donor_separation_status=identity_status(dataset, train, test)))
    lock = dict(schema="spgel_hest_ridge_run_lock_v1", status="LOCKED_BEFORE_FOUR_OFFICIAL_RIDGE_FITS",
        config_sha256=sha256(config_path), assets_sha256=sha256(assets_path), source_files=source_hashes(),
        upstream_commit=HEST_COMMIT, dataset_revision=DATASET_REVISION, cells=rows,
        runtime=runtime(), official_Ridge_fits=4, neural_fits=0, preprocessing_fits_are_not_Ridge_fits=True,
        source_weights_loaded=False, DSS_or_PFI_or_R8_read=False, automatic_retry_fits=0)
    lock["internal_contract_sha256"] = canonical_hash(lock)
    write_json(private / "HEST_RUN_LOCK.json", lock)
    write_json(private / "HEST_QUALIFICATION.private.json", dict(
        status="PASS_FOUR_OFFICIAL_CELLS_FIFTY_GENES_PCA256_INPUT_AND_IDENTITY_ZERO_FITS",
        lock_internal_contract_sha256=lock["internal_contract_sha256"], assets_sha256=sha256(assets_path),
        new_scaler_fits=0, new_PCA_fits=0, new_Ridge_fits=0, rows=rows))
    write_json(private / "HEST_ASSET_POINTER.private.json", dict(path=str(Path(assets_path).resolve()), sha256=sha256(assets_path)))
    return lock


def validate_lock(assets_path, config_path, private):
    config = recipe(config_path)
    lock = read_json(Path(private) / "HEST_RUN_LOCK.json")
    require(lock.get("schema") == "spgel_hest_ridge_run_lock_v1"
        and lock.get("status") == "LOCKED_BEFORE_FOUR_OFFICIAL_RIDGE_FITS"
        and lock.get("internal_contract_sha256") == canonical_hash({k: v for k, v in lock.items() if k != "internal_contract_sha256"})
        and lock["config_sha256"] == sha256(config_path) and lock["assets_sha256"] == sha256(assets_path)
        and lock["source_files"] == source_hashes() and lock["runtime"] == runtime(),
        "OFFICIAL_RUN_LOCK_ASSETS_CODE_OR_RUNTIME_CHANGED")
    require(lock["official_Ridge_fits"] == len(lock["cells"]) == 4 and lock["neural_fits"] == 0
        and {(r["dataset"], r["fold"], r["seed"]) for r in lock["cells"]}
            == {(d, f, 1) for d in ("COAD", "READ") for f in (0, 1)}, "EXACT_FOUR_OFFICIAL_CELLS_CHANGED")
    return lock, config, validate_assets(assets_path)


def cell_name(dataset, fold):
    require(dataset in ("COAD", "READ") and fold in (0, 1), "UNREGISTERED_OFFICIAL_FIT")
    return f"{dataset}_fold{fold}_seed1"


def fit_cell(assets_path, config_path, private, dataset, fold):
    private = private_directory(private)
    lock, config, assets = validate_lock(assets_path, config_path, private)
    folder = private / "cells" / cell_name(dataset, fold)
    require(not folder.exists(), "EXISTING_OFFICIAL_CELL_NO_AUTOMATIC_RETRY")
    folder.mkdir(parents=True)
    started = time.monotonic()
    try:
        train = load_partition(assets, dataset, fold, "train")
        test = load_partition(assets, dataset, fold, "test")
        expected = next(r for r in lock["cells"] if r["dataset"] == dataset and r["fold"] == fold)
        require(train.ordering_sha256 == expected["train_ordering_sha256"]
            and test.ordering_sha256 == expected["test_ordering_sha256"], "OFFICIAL_PARTITION_ORDERING_CHANGED")
        with threadpool_limits(limits=2):
            model = fit_official_ridge(train.features, train.targets, seed=1)
            prediction = model.predict(test.features)
        # Evaluation consumes test targets; model.fit/predict never receives them.
        raw = upstream_raw_metrics(test.targets, prediction, train.genes)
        rounded = official_round4(raw)
        joblib.dump(model, folder / "model.private.joblib", compress=0)
        with (folder / "predictions.private.npz").open("xb") as stream:
            np.savez_compressed(stream, predictions=prediction, targets=test.targets,
                barcodes=test.barcodes, specimen_index=test.specimen_index)
        result = dict(schema="spgel_hest_official_ridge_cell_v1", status="COMPLETE_ONE_OFFICIAL_RIDGE_FIT",
            dataset=dataset, fold=fold, seed=1, train_rows=len(train.features), test_rows=len(test.features),
            genes=train.genes, new_Ridge_fits=1, new_scaler_fits=1, new_PCA_fits=1, neural_fits=0,
            registered_Ridge_coefficients=int(model.estimator.coef_.size),
            Ridge_kwargs=model.estimator.get_params(), actual_lsqr_iterations=model.estimator.n_iter_.tolist(),
            actual_PCA_solver=getattr(model.pipeline["PCA"], "_fit_svd_solver", None),
            internal_contract_sha256=lock["internal_contract_sha256"], config_sha256=sha256(config_path),
            assets_sha256=sha256(assets_path), train_ordering_sha256=train.ordering_sha256,
            test_ordering_sha256=test.ordering_sha256, model_sha256=sha256(folder / "model.private.joblib"),
            predictions_sha256=sha256(folder / "predictions.private.npz"), elapsed_seconds=time.monotonic() - started,
            full_precision_metrics=json_safe_nonfinite(raw), official_rounded4_split_metrics=json_safe_nonfinite(rounded),
            supplementary=supplementary_metrics(test.targets, prediction),
            subgroup_metrics=subgroup_metrics(test, prediction, assets["tasks"][dataset]),
            donor_separation_status=identity_status(dataset, train, test),
            test_RNA_in_predict=False, source_pretrained_weights_loaded=False, DSS_or_R8_read=False)
        write_json(folder / "COMPLETE.json", result)
        return result
    except BaseException as error:
        write_json(folder / "ERROR_RECEIPT.private.json", dict(status="FAILED_OFFICIAL_CELL_NO_AUTOMATIC_RETRY",
            failure_type=type(error).__name__, new_fit_retry_permitted=False))
        (folder / "error.private.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def replay_cell(assets_path, config_path, private, dataset, fold):
    lock, _, assets = validate_lock(assets_path, config_path, private)
    folder = Path(private) / "cells" / cell_name(dataset, fold)
    done = read_json(folder / "COMPLETE.json")
    require(done["internal_contract_sha256"] == lock["internal_contract_sha256"]
        and done["model_sha256"] == sha256(folder / "model.private.joblib")
        and done["predictions_sha256"] == sha256(folder / "predictions.private.npz"), "OFFICIAL_MODEL_ARTIFACT_CHANGED")
    test = load_partition(assets, dataset, fold, "test")
    require(test.ordering_sha256 == done["test_ordering_sha256"], "OFFICIAL_REPLAY_ORDER_CHANGED")
    model = joblib.load(folder / "model.private.joblib")
    with threadpool_limits(limits=2):
        prediction = model.predict(test.features)
    with np.load(folder / "predictions.private.npz", allow_pickle=False) as archive:
        require(np.array_equal(test.targets, archive["targets"])
            and np.array_equal(test.barcodes, archive["barcodes"]), "OFFICIAL_REPLAY_TRUTH_OR_BARCODES_CHANGED")
        maximum_error = float(np.max(np.abs(prediction - archive["predictions"]), initial=0.))
    require(maximum_error <= 1e-6, "OFFICIAL_ACTUAL_MODEL_REPLAY_PREDICTION_ERROR")
    raw = json_safe_nonfinite(upstream_raw_metrics(test.targets, prediction, test.genes))
    require(raw == done["full_precision_metrics"], "OFFICIAL_REPLAY_FULL_PRECISION_METRICS_CHANGED")
    return dict(dataset=dataset, fold=fold, seed=1, status="PASS", actual_saved_model_loaded=True,
        actual_forward_replayed=True, maximum_prediction_error=maximum_error, new_Ridge_fits=0,
        new_scaler_fits=0, new_PCA_fits=0, COMPLETE_sha256=sha256(folder / "COMPLETE.json"),
        model_sha256=done["model_sha256"])


def replay_all(assets_path, config_path, private):
    lock, config, _ = validate_lock(assets_path, config_path, private)
    output = Path(private) / "REPLAY_AUDIT.json"
    require(not output.exists(), "EXISTING_OFFICIAL_REPLAY_PRESERVED")
    rows = [replay_cell(assets_path, config_path, private, d, f) for d in config["tasks"] for f in config["folds"]]
    audit = dict(schema="spgel_hest_official_replay_v1", status="PASS_FOUR_ACTUAL_OFFICIAL_RIDGE_MODEL_REPLAYS",
        internal_contract_sha256=lock["internal_contract_sha256"], passed_models=4, failed_models=0,
        units=rows, new_Ridge_fits=0, new_scaler_fits=0, new_PCA_fits=0)
    write_json(output, audit)
    return audit


def report(assets_path, config_path, private, output):
    lock, config, assets = validate_lock(assets_path, config_path, private)
    audit = read_json(Path(private) / "REPLAY_AUDIT.json")
    require(audit["status"] == "PASS_FOUR_ACTUAL_OFFICIAL_RIDGE_MODEL_REPLAYS"
        and audit["internal_contract_sha256"] == lock["internal_contract_sha256"]
        and len(audit["units"]) == audit["passed_models"] == 4 and audit["failed_models"] == 0,
        "FOUR_REAL_MODEL_REPLAYS_REQUIRED_BEFORE_PUBLIC_REPORT")
    summaries = {}
    for dataset in config["tasks"]:
        cells = []
        for fold in config["folds"]:
            folder = Path(private) / "cells" / cell_name(dataset, fold)
            done = read_json(folder / "COMPLETE.json")
            replay = next(r for r in audit["units"] if r["dataset"] == dataset and r["fold"] == fold)
            require(replay["status"] == "PASS" and replay["actual_forward_replayed"] is True
                and replay["actual_saved_model_loaded"] is True
                and replay["new_Ridge_fits"] == replay["new_scaler_fits"] == replay["new_PCA_fits"] == 0
                and replay["COMPLETE_sha256"] == sha256(folder / "COMPLETE.json")
                and replay["model_sha256"] == done["model_sha256"] == sha256(folder / "model.private.joblib")
                and done["predictions_sha256"] == sha256(folder / "predictions.private.npz"),
                "OFFICIAL_COMPLETION_OR_ARTIFACT_CHANGED_AFTER_REPLAY")
            cells.append(done)
        summaries[dataset] = dict(official_rounded4_fold_aggregate=aggregate_upstream_round4(
            [c["official_rounded4_split_metrics"] for c in cells]),
            full_precision_fold_pearson_mean=[c["full_precision_metrics"]["pearson_mean"] for c in cells],
            supplementary=[c["supplementary"] for c in cells],
            subgroup_metrics=[c["subgroup_metrics"] for c in cells],
            donor_separation_status=cells[0]["donor_separation_status"],
            fixed_seed=1, official_folds=2, source_pretrained_weights_loaded=False)
    result = dict(schema="spgel_hest_official_ridge_report_v1", status="COMPLETE_FOUR_OFFICIAL_RIDGE_FITS_AND_ACTUAL_REPLAYS",
        source_run_lock_file_sha256=sha256(Path(private) / "HEST_RUN_LOCK.json"),
        internal_contract_sha256=lock["internal_contract_sha256"], upstream_commit=HEST_COMMIT,
        dataset_revision=DATASET_REVISION, actual_Ridge_fits=4, actual_replayed_models=4,
        neural_fits=0, source_weights_loaded=False, DSS_or_PFI_or_R8_read=False,
        independent_new_cohort_confirmation=False, tasks=summaries, runtime=lock["runtime"])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result

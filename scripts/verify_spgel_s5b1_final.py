"""Independent, post-completion S5-B1 metadata/hash/arithmetic review.

No producer imports, model/array deserialization, forwards or fits. Input paths
are supplied at runtime; only anonymous logical roles and digests are published.
The administrative COMPLETE gate is checked BEFORE reading experiment results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


ARMS = ("MLP", "GRAPHSAGE", "IDENTITY", "QUERY4", "SPGEL50_ADAPT")
PARTITIONS = tuple((d, f) for d in ("COAD", "READ") for f in (0, 1))
EPOCHS = (80, 160, 320, 640, 1280)
SOURCE_CANONICAL = "20966b4d2d92df28f70d18ee83f111e0b6db0e15154b662b70a9efc119057b24"
SOURCE_PATHS = (
    "src/spgel_s5_hest/__init__.py", "src/spgel_s5_hest/protocol.py",
    "src/spgel_s5_hest/data.py", "src/spgel_s5_hest/models.py",
    "src/spgel_s5_hest/execution.py", "scripts/spgel_s5_hest.py",
    "configs/spgel_s5_hest/shared_feature_v1.json", "tests/test_spgel_s5_hest.py",
    "src/spgel_hest/assets.py", "src/spgel_hest/official.py",
    "src/spgel_hest_neural/graph.py", "src/spgel_hest_neural/models.py",
)
PARAMETERS = {
    "MLP": (55858, 55858, 0), "GRAPHSAGE": (72242, 72242, 0),
    "IDENTITY": (72242, 72242, 0), "QUERY4": (64242, 64242, 0),
    "SPGEL50_ADAPT": (69042, 64242, 4800),
}


class ReviewError(RuntimeError):
    """A public-safe error code; never echo private paths or sample identifiers."""


def need(ok: bool, code: str) -> None:
    if not ok:
        raise ReviewError(code)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def read_json(path: Path):
    def nonfinite(_):
        raise ReviewError("NONSTANDARD_NONFINITE_JSON")
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=_unique_pairs, parse_constant=nonfinite)


def canonical(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def digest_format(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def check_seal(value):
    need(value.get("internal_contract_sha256") == canonical({
        k: v for k, v in value.items() if k != "internal_contract_sha256"
    }), "METADATA_SEAL_MISMATCH")


def require_complete_gate(state_path: Path):
    """The only experiment file read before COMPLETE; safe to call while running."""
    state = read_json(state_path)
    need(state.get("status") == "COMPLETE_20_DEV_20_FINAL_120_ACTUAL_REPLAYS",
         "NOT_COMPLETE_NO_RESULT_OR_PRIVATE_METADATA_READ")
    need(state.get("development_completed") == 20 and state.get("final_completed") == 20
         and not state.get("active") and not state.get("failed"), "ADMIN_COMPLETE_COUNTS_INVALID")
    need(state.get("source_canonical_sha256") == SOURCE_CANONICAL
         and state.get("expression_transferred_to_server") is False
         and state.get("GPU_used") is False, "ADMIN_SOURCE_OR_COMPUTE_SCOPE_CHANGED")
    return state


class FileHashes:
    """Stream bytes, track immutable size/mtime, publish logical role only."""

    def __init__(self):
        self.by_path = {}
        self.inventory = {}

    def get(self, path: Path, role: str, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        if path in self.by_path:
            old_stamp, digest = self.by_path[path]
            need(stamp == old_stamp, "ARTIFACT_CHANGED_DURING_REVIEW")
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            after = path.stat()
            need(stamp == (after.st_size, after.st_mtime_ns), "ARTIFACT_CHANGED_WHILE_HASHING")
            digest = hasher.hexdigest()
            self.by_path[path] = (stamp, digest)
        need(role not in self.inventory or self.inventory[role]["sha256"] == digest,
             "LOGICAL_ARTIFACT_ROLE_COLLISION")
        self.inventory[role] = {"sha256": digest, "bytes": stamp[0]}
        if expected is not None:
            need(digest == expected, "ARTIFACT_SHA256_MISMATCH:" + role)
        return digest

    def unchanged(self):
        for path, (stamp, _) in self.by_path.items():
            stat = path.stat()
            need(stamp == (stat.st_size, stat.st_mtime_ns), "ARTIFACT_CHANGED_BEFORE_REVIEW_COMMIT")


class Arithmetic:
    def __init__(self):
        self.maximum_error = 0.0
        self.scalar_comparisons = 0
        self.nonfinite_R2_quantile_checks_not_reconstructible = 0

    def same(self, actual, expected, code="ARITHMETIC_MISMATCH", tolerance=0.0):
        if isinstance(expected, dict):
            need(isinstance(actual, dict) and set(actual) == set(expected), code + "_KEYS")
            for key in expected:
                self.same(actual[key], expected[key], code, tolerance)
        elif isinstance(expected, list):
            need(isinstance(actual, list) and len(actual) == len(expected), code + "_LENGTH")
            for a, e in zip(actual, expected):
                self.same(a, e, code, tolerance)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            need(isinstance(actual, (int, float)) and not isinstance(actual, bool), code + "_TYPE")
            need(math.isfinite(float(actual)) and math.isfinite(float(expected)), code + "_FINITE")
            error = abs(float(actual) - float(expected))
            self.maximum_error = max(self.maximum_error, error)
            self.scalar_comparisons += 1
            need(error <= tolerance, code)
        else:
            need(actual == expected and type(actual) is type(expected), code)


def number_or_nan(value):
    return np.nan if value is None else float(value)


def finite_or_null(value):
    return float(value) if np.isfinite(value) else None


def rounded4(value):
    if isinstance(value, dict):
        return {k: rounded4(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded4(v) for v in value]
    return round(value, 4) if isinstance(value, float) else value


def independently_validate_split(raw, genes, arithmetic):
    """Recompute summary arithmetic from all 50 recorded per-gene metrics."""
    need([r["name"] for r in raw["pearson_corrs"]] == genes, "SPLIT_GENE_ORDER_CHANGED")
    need(len(raw["l2_errors"]) == len(raw["r2_scores"]) == len(genes) == 50,
         "FIFTY_GENE_METRIC_SUPPORT_REQUIRED")
    pearson = np.array([number_or_nan(r["pearson_corr"]) for r in raw["pearson_corrs"]])
    need(all(x is None or -1.000000000001 <= x <= 1.000000000001 for x in (
        r["pearson_corr"] for r in raw["pearson_corrs"])), "PCC_OUTSIDE_RANGE")
    arithmetic.same(raw["pearson_mean"], finite_or_null(np.mean(pearson)), tolerance=1e-14)
    arithmetic.same(raw["pearson_std"], finite_or_null(np.std(pearson)), tolerance=1e-14)
    for prefix, key in (("l2_error", "l2_errors"), ("r2_score", "r2_scores")):
        if any(value is None for value in raw[key]):
            # JSON null loses the distinction between NaN and signed infinity.
            # A quantile of mixed finite/-Inf values can itself be finite, so
            # substituting NaN here would manufacture a false arithmetic check.
            need(prefix == "r2_score", "NONFINITE_MSE_IN_RECORDED_METRICS")
            arithmetic.nonfinite_R2_quantile_checks_not_reconstructible += 3
            continue
        values = np.array([number_or_nan(v) for v in raw[key]])
        for suffix, quantile in (("q1", 25), ("q2", 50), ("q3", 75)):
            arithmetic.same(raw[prefix + "_" + suffix], finite_or_null(np.percentile(values, quantile)),
                            tolerance=1e-14)
    return rounded4(raw)


def independently_aggregate(split0, split1):
    """Official round-first-fold arithmetic; no producer function is imported."""
    need([g["name"] for g in split0["pearson_corrs"]] ==
         [g["name"] for g in split1["pearson_corrs"]], "AGGREGATE_GENE_ORDER_CHANGED")
    rows = []
    for left, right in zip(split0["pearson_corrs"], split1["pearson_corrs"]):
        values = [left["pearson_corr"], right["pearson_corr"]]
        array = np.array([number_or_nan(v) for v in values])
        rows.append({"name": left["name"], "pearson_corrs": values,
                     "mean": finite_or_null(np.mean(array)), "std": finite_or_null(np.std(array))})
    rows.sort(key=lambda r: -np.inf if r["mean"] is None else r["mean"], reverse=True)
    means = [split0["pearson_mean"], split1["pearson_mean"]]
    a = np.array([number_or_nan(v) for v in means])
    return rounded4({"pearson_corrs": rows, "pearson_mean": finite_or_null(np.mean(a)),
                     "pearson_std": finite_or_null(np.std(a)), "mean_per_split": means})


def unique_exact(rows, key, expected, code):
    actual = [key(row) for row in rows]
    need(len(actual) == len(set(actual)) == len(expected) and set(actual) == set(expected), code)


def select_own_fold(dev_cells):
    scores = {str(epoch): float(np.mean([
        next(s["validation_MSE"] for s in dev_cells[a]["snapshots"] if s["epoch"] == epoch)
        for a in ARMS])) for epoch in EPOCHS}
    need(all(math.isfinite(v) for v in scores.values()), "INNER_SELECTION_NONFINITE")
    chosen = min(EPOCHS, key=lambda e: (scores[str(e)], e))
    return scores, chosen


def csv_text(rows):
    need(bool(rows), "EMPTY_PUBLIC_CSV")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _valid_graph(graph):
    need(graph["cross_specimen_edges"] == 0 and graph["physical_radius_used"] is False,
         "GRAPH_PARTITION_SCOPE_CHANGED")
    msg = graph["graphsage_messages"]
    need(msg["self_edges"] == 0 and msg["maximum_real_neighbors"] <= 15
         and msg["mean_uses_actual_nonself_neighbor_count"] is True,
         "GRAPHSAGE_NONSELF_MEAN_CONTRACT_CHANGED")


def audit_all(args):
    state = require_complete_gate(args.state)
    root = args.root.resolve()
    hashes = FileHashes()
    arithmetic = Arithmetic()
    hashes.get(args.state, "administrative_final_state")
    freeze = read_json(args.freeze)
    hashes.get(args.freeze, "v3_root_code_freeze")
    source_files = freeze["sources"]
    need(set(source_files) == set(SOURCE_PATHS) and canonical(source_files) ==
         freeze["sources_canonical_sha256"] == SOURCE_CANONICAL, "FROZEN_TWELVE_SOURCE_SET_CHANGED")
    for relative, digest in source_files.items():
        path = (root / relative).resolve()
        need(path.is_relative_to(root), "SOURCE_ESCAPES_ROOT")
        hashes.get(path, "source:" + relative, digest)
    config_hash = hashes.get(args.config, "recipe", source_files["configs/spgel_s5_hest/shared_feature_v1.json"])
    cfg = read_json(args.config)
    need(cfg["arms"] == list(ARMS) and cfg["development_checkpoints"] == list(EPOCHS)
         and cfg["development_max_epochs"] == 1280 and cfg["seed"] == 1
         and cfg["development_fits"] == cfg["final_fits"] == 20
         and cfg["paired_MSE_or_gene_reconstruction_objective"] is False,
         "RECIPE_SCOPE_CHANGED")
    need(np.__version__ == freeze["runtime"]["numpy"], "REVIEW_NUMPY_VERSION_DIFFERS_FROM_FROZEN_ARITHMETIC")
    private = args.private
    producer = read_json(args.producer)
    producer_sha = hashes.get(args.producer, "producer_public_report")
    need(producer["status"] == "COMPLETE_TWENTY_DEVELOPMENT_AND_TWENTY_FINAL_FITS_AND_120_REPLAYS",
         "PRODUCER_REPORT_NOT_COMPLETE")
    need(producer["source_files"] == source_files and producer["config_sha256"] == config_hash,
         "REPORT_SCIENCE_BINDING_CHANGED")
    expected_public_counts = (20, 20, 120, 8, 8)
    need(tuple(producer[k] for k in ("actual_development_fits", "actual_final_fits",
         "actual_saved_state_replays", "new_scaler_fits", "new_PCA_fits")) == expected_public_counts,
         "PRODUCER_ACTUAL_COUNTS_CHANGED")

    locks = {}
    for label, filename, public_key in (
        ("development", "S5B_DEV_RUN_LOCK.json", "development_lock_sha256"),
        ("schedule", "S5B_FINAL_SCHEDULE_LOCK.json", "final_schedule_sha256"),
        ("final_inputs", "S5B_FINAL_INPUT_LOCK.json", "final_inputs_sha256"),
    ):
        locks[label] = read_json(private / filename)
        check_seal(locks[label])
        hashes.get(private / filename, "lock:" + label, producer[public_key])
        need(locks[label]["source_files"] == source_files
             and locks[label]["official_test_expression_loaded"] is False
             and locks[label]["new_model_fits"] == 0, "LOCK_SCOPE_CHANGED")
    dev, schedule, final = (locks[k] for k in ("development", "schedule", "final_inputs"))
    need(dev["config_sha256"] == config_hash and dev["runtime"] == freeze["runtime"], "LOCK_RUNTIME_OR_CONFIG_CHANGED")
    need(dev["status"] == "LOCKED_BEFORE_TWENTY_DEVELOPMENT_FITS"
         and schedule["status"] == "LOCKED_FROM_20_DEVELOPMENT_FITS_NO_OFFICIAL_TEST"
         and final["status"] == "LOCKED_FULL_TRAIN_INPUTS_BEFORE_TWENTY_FINAL_FITS",
         "LOCK_STAGE_STATUS_CHANGED")
    need(schedule["development_contract"] == final["development_contract"] == dev["internal_contract_sha256"]
         and final["schedule_sha256"] == producer["final_schedule_sha256"], "LOCK_CHAIN_CHANGED")
    need(dev["new_scaler_fits"] == dev["new_PCA_fits"] == final["new_scaler_fits"] == final["new_PCA_fits"] == 4,
         "EIGHT_INNER_AND_OUTER_PREPROCESSOR_FITS_REQUIRED")
    initial = read_json(private / "S5B_PREPARATION_STARTED.private.json")
    hashes.get(private / "S5B_PREPARATION_STARTED.private.json", "preparation_start")
    need(initial == {k: dev[k] for k in ("source_files", "runtime", "config_sha256", "assets_sha256")},
         "PREPARATION_START_BINDING_CHANGED")
    hashes.get(args.assets, "runtime_assets", dev["assets_sha256"])
    assets = read_json(args.assets)
    genes = {}
    for d in ("COAD", "READ"):
        gene_file = assets["tasks"][d]["gene_file"]
        hashes.get(Path(gene_file["path"]), "official_50_gene_file:" + d, gene_file["sha256"])
        genes[d] = read_json(Path(gene_file["path"]))["genes"]
        need(len(genes[d]) == len(set(genes[d])) == 50, "OFFICIAL_FIFTY_UNIQUE_GENES_REQUIRED")
    profile_path = private / "ZERO_OPTIMIZER_PROFILE.private.json"
    hashes.get(profile_path, "zero_optimizer_qualification", dev["zero_optimizer_profile_sha256"])
    profile = read_json(profile_path)
    need(profile["status"] == "PASS_FIVE_REAL_INNER_TRAIN_FORWARD_BACKWARD_ZERO_OPTIMIZER"
         and profile["optimizer_steps"] == profile["new_fits"] == 0, "ZERO_UPDATE_QUALIFICATION_CHANGED")
    unique_exact(profile["units"], lambda r: r["arm"], ARMS, "ZERO_UPDATE_ARM_SCOPE_CHANGED")
    need(all(r["optimizer_steps"] == 0 and r["model_state_unchanged"] is True
         and r["test_expression_used"] is False for r in profile["units"]), "ZERO_UPDATE_PROFILE_INVALID")

    cache_meta, cache_sha, partition_rows = {}, {}, {}
    for stage, folder, lock, scope in (
        ("DEV", "dev_inputs", dev, "INNER_TRAIN_ONLY"),
        ("FINAL", "final_inputs", final, "FULL_OFFICIAL_OUTER_TRAIN_ONLY"),
        ("TEST", "test_inputs", None, "NO_TEST_FITS_TRANSFORM_ONLY"),
    ):
        if lock is not None:
            unique_exact(lock["partitions"], lambda r: (r["dataset"], r["fold"]), PARTITIONS,
                         "FOUR_PARTITIONS_REQUIRED")
        for d, f in PARTITIONS:
            role = f"{stage}:{d}:fold{f}"
            folder_path = private / folder / f"{d}_fold{f}"
            meta_path = folder_path / "CACHE.private.json"
            meta = read_json(meta_path)
            expected = next((r for r in lock["partitions"] if (r["dataset"], r["fold"]) == (d, f)), None) if lock else None
            cache_sha[stage, d, f] = hashes.get(meta_path, "cache_metadata:" + role,
                                               expected["cache_metadata_sha256"] if expected else None)
            need(meta["dataset"] == d and meta["fold"] == f and meta["genes"] == genes[d]
                 and meta["status"] == "PASS" and meta["preprocessing_fit_scope"] == scope,
                 "CACHE_IDENTITY_GENE_OR_FIT_SCOPE_CHANGED")
            hashes.get(folder_path / "inputs.private.npz", "cache_arrays:" + role, meta["input_file_sha256"])
            hashes.get(folder_path / "pipeline.private.joblib", "cache_pipeline:" + role, meta["pipeline_sha256"])
            need(meta["new_scaler_fits"] == meta["new_PCA_fits"] == (0 if stage == "TEST" else 1),
                 "CACHE_PREPROCESSOR_FIT_COUNT_CHANGED")
            need(all(digest_format(v) for v in meta["arrays_sha256"].values()), "INVALID_DECLARED_ARRAY_HASH")
            if stage != "TEST":
                need(meta["official_test_expression_loaded"] is False and expected["train_rows"] > 0,
                     "TRAIN_CACHE_ROLE_CHANGED")
                partition_rows[stage, d, f] = expected["train_rows"]
            if stage == "DEV":
                split = meta["inner_split"]
                need(split == expected["split"] and split["uses_expression_or_outcomes"] is False
                     and split["graph_cross_inner_partition_edges"] == 0
                     and split["donor_heldout_validation_claimed"] is False
                     and split["spatial_buffer_between_inner_train_and_validation"] is False,
                     "INNER_SPLIT_SCOPE_CHANGED")
            cache_meta[stage, d, f] = meta

    unique_exact(producer["cells"], lambda r: (r["stage"], r["dataset"], r["fold"], r["arm"]),
                 {(s, d, f, a) for s in ("DEV", "FINAL") for d, f in PARTITIONS for a in ARMS},
                 "REPORT_REQUIRES_FORTY_UNIQUE_CELLS")
    unique_exact(schedule["selections"], lambda r: (r["dataset"], r["fold"]), PARTITIONS,
                 "FOUR_OWN_FOLD_SCHEDULES_REQUIRED")
    arithmetic.same(producer["selections"], schedule["selections"], "REPORT_SCHEDULE_CHANGED")
    selected = {(r["dataset"], r["fold"]): r for r in schedule["selections"]}
    cells, ledger_cells, expected_replays = {}, [], set()
    for stage, folder, contract in (("DEV", "dev_cells", dev["internal_contract_sha256"]),
                                   ("FINAL", "final_cells", final["internal_contract_sha256"])):
        for d, f in PARTITIONS:
            for arm in ARMS:
                cell = f"{d}_fold{f}_{arm}"
                role = f"{stage}:{d}:fold{f}:{arm}"
                cell_folder = private / folder / cell
                r = read_json(cell_folder / "COMPLETE.private.json")
                completion_sha = hashes.get(cell_folder / "COMPLETE.private.json", "completion:" + role)
                epochs = 1280 if stage == "DEV" else selected[d, f]["selected_epoch"]
                snaps = list(EPOCHS) if stage == "DEV" else [epochs]
                need((r["dataset"], r["fold"], r["arm"], r["phase"], r["seed"]) == (d, f, arm, stage, 1)
                     and r["status"] == ("COMPLETE_ONE_DEVELOPMENT_FIT" if stage == "DEV" else "COMPLETE_ONE_FINAL_FIT")
                     and r["epochs"] == r["optimizer_updates"] == len(r["learning_curve"]) == epochs
                     and r["new_model_fits"] == 1 and r["internal_contract_sha256"] == contract
                     and r["input_cache_metadata_sha256"] == cache_sha[stage, d, f], "ACTUAL_FIT_BINDING_CHANGED")
                need(r["official_test_expression_loaded"] is False and r["inference_reads_RNA"] is False
                     and r["auxiliary_supervision_only_train_50genes"] is True,
                     "ACTUAL_FIT_INFORMATION_BUDGET_CHANGED")
                need(tuple(r["parameters"][k] for k in ("total", "image_prediction", "training_only_gene_auxiliary"))
                     == PARAMETERS[arm], "EFFECTIVE_PARAMETER_BUDGET_CHANGED")
                need([s["epoch"] for s in r["snapshots"]] == snaps, "SAVED_EPOCH_SET_NOT_EXACT")
                need([c["epoch"] for c in r["learning_curve"]] == list(range(1, epochs + 1)),
                     "LEARNING_CURVE_EPOCHS_INCOMPLETE")
                n_train = partition_rows[stage, d, f]
                for c in r["learning_curve"]:
                    need(c["optimizer_updates"] == c["epoch"] and all(math.isfinite(c[k]) for k in (
                        "pre_update_main_MSE", "pre_update_auxiliary_loss", "pre_update_total_loss")),
                        "TRAINING_UPDATE_OR_LOSS_INVALID")
                    expected_total = float(np.float32(np.float32(c["pre_update_main_MSE"]) +
                                                     np.float32(0.1) * np.float32(c["pre_update_auxiliary_loss"])))
                    arithmetic.same(c["pre_update_total_loss"], expected_total,
                                    "TRAINING_LOSS_DECOMPOSITION_CHANGED")
                    need(bool(c["gradient_norms"]) and all(math.isfinite(v) and v >= 0
                         for v in c["gradient_norms"].values()), "TRAIN_GRADIENT_LOG_INVALID")
                    pairing = c["paired_ITC_audit"]
                    if arm == "SPGEL50_ADAPT":
                        need(pairing is not None and pairing["train_rows"] == n_train
                             and pairing["anchor_count"] == min(512, n_train)
                             and pairing["negative_count"] == min(8, n_train - 1)
                             and pairing["pairings_per_direction"] == min(512, n_train) * (min(8, n_train - 1) + 1)
                             and pairing["temperature"] == 0.1
                             and pairing["candidate_condition_recomputed_both_directions"] is True
                             and pairing["positive_conditioned_embeddings_reused"] is False,
                             "CANDIDATE_SPECIFIC_ITC_AUDIT_CHANGED")
                        expected_aux = float(np.float32(np.float32(pairing["image_anchor_CE"]) +
                                                       np.float32(pairing["gene_anchor_CE"])) / np.float32(2))
                        arithmetic.same(c["pre_update_auxiliary_loss"], expected_aux,
                                        "TWO_ANCHOR_ITC_ARITHMETIC_CHANGED")
                    else:
                        need(pairing is None and c["pre_update_auxiliary_loss"] == 0,
                             "BASELINE_HAS_UNREGISTERED_AUXILIARY_OBJECTIVE")
                for snap in r["snapshots"]:
                    e = snap["epoch"]
                    hashes.get(cell_folder / f"epoch{e:03d}.private.pt", "state:" + role + f":{e}", snap["state_sha256"])
                    hashes.get(cell_folder / f"epoch{e:03d}.private.npz", "saved_predictions:" + role + f":{e}", snap["predictions_sha256"])
                    expected_replays.add((stage, d, f, arm, e, snap["state_sha256"], completion_sha))
                    need(math.isfinite(snap["post_update_train_MSE"]) and
                         (math.isfinite(snap["validation_MSE"]) if stage == "DEV" else snap["validation_MSE"] is None),
                         "SAVED_STATE_MSE_INVALID")
                _valid_graph(r["train_graph"])
                if stage == "DEV":
                    _valid_graph(r["validation_graph"])
                else:
                    need(r["validation_graph"] is None, "FINAL_HAS_UNREGISTERED_SELECTION_CONTEXT")
                public = next(v for v in producer["cells"] if (v["stage"], v["dataset"], v["fold"], v["arm"]) == (stage, d, f, arm))
                for k in ("epochs", "optimizer_updates", "parameters", "snapshots", "elapsed_seconds"):
                    arithmetic.same(public[k], r[k], "REPORT_CELL_DIFFERS_FROM_ACTUAL_COMPLETE")
                need(public["COMPLETE_sha256"] == completion_sha, "REPORT_COMPLETE_HASH_CHANGED")
                curve = [v for v in r["learning_curve"] if v["epoch"] == 1 or v["epoch"] % 20 == 0 or v["epoch"] == epochs]
                arithmetic.same(public["diagnostic_curve_stride20"], curve, "REPORT_CURVE_DIFFERS_FROM_COMPLETE")
                first = r["learning_curve"][0]["pre_update_main_MSE"]
                last = r["learning_curve"][-1]["pre_update_main_MSE"]
                ledger_cells.append({"stage": stage, "dataset": d, "fold": f, "arm": arm,
                    "epochs": epochs, "optimizer_updates": r["optimizer_updates"], "parameters": r["parameters"],
                    "saved_states": len(snaps), "fit_loop_elapsed_seconds": r["elapsed_seconds"],
                    "COMPLETE_sha256": completion_sha, "input_cache_metadata_sha256": cache_sha[stage, d, f],
                    "first_pre_update_train_MSE": first, "last_pre_update_train_MSE": last,
                    "last_over_first_pre_update_train_MSE": last / first if first else None,
                    "budget_assessment": "FIXED_DEVELOPMENT_BUDGET_NOT_CONVERGENCE_PROOF" if stage == "DEV" else selected[d, f]["budget_assessment"]})
                cells[stage, d, f, arm] = r

    for stage in ("DEV", "FINAL"):
        for d, f in PARTITIONS:
            need(cells[stage, d, f, "GRAPHSAGE"]["initial_state_sha256"] ==
                 cells[stage, d, f, "IDENTITY"]["initial_state_sha256"], "GRAPH_IDENTITY_INITIALIZATION_NOT_MATCHED")
    for d, f in PARTITIONS:
        scores, epoch = select_own_fold({a: cells["DEV", d, f, a] for a in ARMS})
        row = selected[d, f]
        arithmetic.same(row["inner_mean_MSE"], scores, "OWN_FOLD_INNER_MEAN_SELECTION_CHANGED")
        need(row["selected_epoch"] == epoch and row["uses_only_this_outer_fold_training_partition"] is True
             and row["budget_assessment"] == ("BUDGET_EDGE_NOT_CONVERGENCE_CONFIRMED" if epoch == 1280
                                               else "INNER_SELECTED_NOT_CONVERGENCE_PROOF"), "OWN_FOLD_COMMON_EPOCH_RULE_CHANGED")
        need(set(row["development_completion_sha256"]) == set(ARMS), "SELECTION_REQUIRES_FIVE_ARMS")
        for a, sha in row["development_completion_sha256"].items():
            need(sha == hashes.inventory[f"completion:DEV:{d}:fold{f}:{a}"]["sha256"], "SELECTION_DEV_SHA_CHANGED")
    need(len(expected_replays) == 120, "EXPECTED_REPLAY_SET_NOT_120")

    gate_path = private / "OFFICIAL_TEST_OPEN_GATE.private.json"
    gate = read_json(gate_path)
    gate_sha = hashes.get(gate_path, "official_test_open_gate")
    need(gate["status"] == "OPENED_AFTER_ALL_20_FINAL_STATES_FROZEN" and gate["new_model_fits"] == 0,
         "OFFICIAL_TEST_GATE_CHANGED")
    expected_final_bindings = {f"{d}_fold{f}_{a}": hashes.inventory[f"completion:FINAL:{d}:fold{f}:{a}"]["sha256"]
                               for d, f in PARTITIONS for a in ARMS}
    need(gate["final_completion_sha256"] == expected_final_bindings, "TEST_GATE_NOT_BOUND_TO_ALL_TWENTY_FINAL_STATES")
    for d, f in PARTITIONS:
        need(cache_meta["TEST", d, f]["official_test_open_gate_sha256"] == gate_sha, "TEST_CACHE_GATE_CHANGED")
    evaluation_path = private / "S5B_OFFICIAL_EVALUATION.private.json"
    evaluation = read_json(evaluation_path)
    hashes.get(evaluation_path, "official_evaluation", producer["evaluation_sha256"])
    need(evaluation["status"] == "COMPLETE_TWENTY_FROZEN_FINAL_TEST_EVALUATIONS"
         and evaluation["new_model_fits"] == 0 and evaluation["test_gate_sha256"] == gate_sha
         and evaluation["final_input_contract"] == final["internal_contract_sha256"], "OFFICIAL_EVALUATION_CHAIN_CHANGED")
    unique_exact(evaluation["units"], lambda r: (r["dataset"], r["fold"], r["arm"]),
                 {(d, f, a) for d, f in PARTITIONS for a in ARMS}, "TWENTY_FINAL_EVALUATIONS_REQUIRED")
    fold_rows, gene_rows, by_fold = [], [], {}
    for ev in evaluation["units"]:
        d, f, a = ev["dataset"], ev["fold"], ev["arm"]
        role = f"FINAL:{d}:fold{f}:{a}"
        folder = private / "final_cells" / f"{d}_fold{f}_{a}"
        own = read_json(folder / "EVALUATION.private.json")
        hashes.get(folder / "EVALUATION.private.json", "cell_evaluation:" + role)
        arithmetic.same(own, ev, "EVALUATION_UNIT_NOT_IDENTICAL_TO_CELL")
        snap = cells["FINAL", d, f, a]["snapshots"][0]
        need(ev["epoch"] == snap["epoch"] and ev["state_sha256"] == snap["state_sha256"]
             and ev["COMPLETE_sha256"] == expected_final_bindings[f"{d}_fold{f}_{a}"]
             and ev["test_cache_metadata_sha256"] == cache_sha["TEST", d, f]
             and ev["first_test_evaluation_after_all_twenty_final_fits"] is True
             and ev["new_model_fits"] == 0 and ev["test_rows"] > 0, "FINAL_EVALUATION_BINDING_CHANGED")
        hashes.get(folder / "official_test_prediction.private.npz", "official_test_prediction:" + role,
                   ev["test_prediction_sha256"])
        _valid_graph(ev["test_graph"])
        rounded = independently_validate_split(ev["full_precision_metrics"], genes[d], arithmetic)
        arithmetic.same(ev["official_round4"], rounded, "OFFICIAL_SPLIT_ROUND4_CHANGED")
        by_fold[d, f, a] = rounded
        raw = ev["full_precision_metrics"]
        fold_rows.append({"dataset": d, "fold": f, "arm": a, "selected_epoch": ev["epoch"],
            "test_spots": ev["test_rows"], "total_genes": 50,
            "defined_Pearson_genes": sum(r["pearson_corr"] is not None for r in raw["pearson_corrs"]),
            "Pearson_mean_full_precision": raw["pearson_mean"], "Pearson_std_full_precision": raw["pearson_std"],
            "Pearson_mean_official_round4": rounded["pearson_mean"],
            "MSE_mean_from_50genes": float(np.mean(raw["l2_errors"])),
            "MSE_q1_official_round4": rounded["l2_error_q1"], "MSE_q2_official_round4": rounded["l2_error_q2"],
            "MSE_q3_official_round4": rounded["l2_error_q3"], "R2_q2_official_round4": rounded["r2_score_q2"],
            "state_sha256": ev["state_sha256"], "test_prediction_sha256": ev["test_prediction_sha256"]})
        for i, gene in enumerate(genes[d]):
            gene_rows.append({"dataset": d, "fold": f, "arm": a, "gene_ordinal": i, "gene": gene,
                "Pearson_full_precision": raw["pearson_corrs"][i]["pearson_corr"],
                "Pearson_official_round4": rounded["pearson_corrs"][i]["pearson_corr"],
                "MSE_full_precision": raw["l2_errors"][i], "MSE_official_round4": rounded["l2_errors"][i],
                "R2_full_precision": raw["r2_scores"][i], "R2_official_round4": rounded["r2_scores"][i]})
    need(len(fold_rows) == 20 and len(gene_rows) == 1000, "PUBLIC_FOLD_OR_GENE_SUPPORT_INCOMPLETE")
    aggregates = {d: {a: independently_aggregate(by_fold[d, 0, a], by_fold[d, 1, a]) for a in ARMS}
                  for d in ("COAD", "READ")}
    arithmetic.same(producer["tasks"], aggregates, "OFFICIAL_ROUND_FIRST_TWO_FOLD_AGGREGATION_CHANGED")

    replay_path = private / "S5B_REPLAY.private.json"
    replay = read_json(replay_path)
    hashes.get(replay_path, "actual_saved_state_replays", producer["replay_sha256"])
    need(replay["status"] == "PASS_120_ACTUAL_SAVED_STATE_REPLAYS" and replay["passed_states"] == 120
         and replay["new_model_fits"] == replay["optimizer_steps"] == 0
         and replay["official_evaluation_sha256"] == producer["evaluation_sha256"], "ACTUAL_REPLAY_SCOPE_CHANGED")
    key = lambda r: (r["stage"], r["dataset"], r["fold"], r["arm"], r["epoch"], r["state_sha256"], r["COMPLETE_sha256"])
    unique_exact(replay["units"], key, expected_replays, "REPLAY_SET_NOT_EXACTLY_120_UNIQUE_ACTUAL_STATES")
    need(all(r["status"] == "PASS" and r["actual_state_loaded"] is True
             and r["actual_forward_replayed"] is True and r["new_model_fits"] == r["optimizer_steps"] == 0
             and 0 <= r["maximum_prediction_error"] <= 1e-6 for r in replay["units"]),
         "SAVED_STATE_ACTUAL_REPLAY_FAILED")
    replay_error = max(r["maximum_prediction_error"] for r in replay["units"])
    arithmetic.same(producer["maximum_replay_error"], replay_error, "MAXIMUM_REPLAY_ERROR_CHANGED")
    total_updates = sum(r["optimizer_updates"] for r in ledger_cells)
    need(total_updates == 25600 + 5 * sum(r["selected_epoch"] for r in selected.values())
         == producer["actual_optimizer_updates"], "ACTUAL_OPTIMIZER_UPDATE_ARITHMETIC_CHANGED")

    admin_path = private / "admin" / "EXECUTION_PRIVATE.json"
    admin = read_json(admin_path)
    hashes.get(admin_path, "administrative_execution_receipt")
    arithmetic.same(admin["state"], state, "ADMIN_FINAL_STATE_CHANGED")
    need(admin["new_fits_during_replay"] == 0 and admin["official_test_selects_epoch_or_model"] is False,
         "ADMIN_EXECUTION_SCOPE_CHANGED")
    expected_steps = {(op, f"{op}_{d}_fold{f}_{a}") for op in ("fit-dev", "fit-final") for d, f in PARTITIONS for a in ARMS}
    expected_steps |= {(op, op) for op in ("prepare-dev", "select", "prepare-final", "evaluate", "replay", "report")}
    unique_exact(admin["steps"], lambda r: (r["operation"], r["unit"]), expected_steps,
                 "ADMIN_OPERATION_SET_NOT_EXACTLY_46")
    need(all(r["exit_code"] == 0 and r["elapsed_seconds"] >= 0 for r in admin["steps"]), "ADMIN_STEP_FAILED")
    telemetry = admin["telemetry"]
    need(bool(telemetry), "NO_ACTUAL_RESOURCE_TELEMETRY")
    superseded = read_json(args.superseded_cost)
    superseded_sha = hashes.get(args.superseded_cost, "superseded_pretest_cost")
    need(superseded["schema"] == "spgel_s5b_superseded_pretest_cost_v1"
         and superseded["completed_development_fits"] == 8
         and superseded["interrupted_development_processes"] == 2
         and superseded["interrupted_initialized_fit_cells"] == 1
         and superseded["interrupted_before_fit_initialization"] == 1
         and superseded["completed_formal_fits"] == superseded["official_test_evaluations"] == 0
         and superseded["known_complete_optimizer_updates"] == 2560
         and superseded["partial_optimizer_updates_exact"] is None
         and superseded["performance_results_used_in_final_tables"] is False,
         "SUPERSEDED_PRETEST_COST_SCOPE_CHANGED")
    need(len(superseded["completed_units"]) == 8 and
         sum(r["observed_optimizer_updates"] for r in superseded["completed_units"]) == 2560,
         "SUPERSEDED_COMPLETED_STEP_ARITHMETIC_CHANGED")
    stopped = root / "tmp/SPGEL_S5B1_PRETEST_FIDELITY_REPAIR_RECEIPT_20261029.json"
    hashes.get(stopped, "superseded_stopped_receipt", superseded["stopped_receipt_sha256"])
    hashes.get(root / "tmp/S5B1_SUPERSEDED_PAIRED_MSE_SOURCE_20261029.zip",
               "superseded_scientific_source_archive", superseded["archived_source_zip_sha256"])
    expected_boundaries = {
        "official_test_not_used_for_selection": True, "tissue_truth_or_teacher_used": False,
        "extra_gene_membership_used": False, "queries_are_anonymous_not_medical_programs": True,
        "SPGEL50_is_not_original_A5": True, "BLEEP_or_TRIPLEX_reproduced": False,
        "optimizer_convergence_proved": False, "new_independent_cohort_claimed": False,
        "DSS_PFI_or_R8_read": False, "own_fold_test_excluded_from_training_and_selection": True,
        "reverse_CV_train_access_to_same_source_is_permitted": True,
        "single_slide_inner_block_has_spatial_exclusion_buffer": False,
    }
    arithmetic.same(producer["boundaries"], expected_boundaries, "PUBLIC_METHOD_BOUNDARY_CHANGED")
    hashes.unchanged()
    ledger = {"schema": "spgel_s5b1_actual_ledger_v1", "status": "PASS_COMPLETE_ACTUAL_EXECUTION_LEDGER",
        "counts": {"development_fits": 20, "final_fits": 20, "saved_state_replays": 120,
                   "scaler_fits": 8, "PCA_fits": 8, "official_final_test_evaluations": 20},
        "optimizer_updates": {"development": 25600, "final": total_updates - 25600, "total": total_updates},
        "administrative_wall_elapsed_seconds": state["elapsed_seconds"],
        "fit_loop_seconds_sum": sum(r["fit_loop_elapsed_seconds"] for r in ledger_cells),
        "fit_administrative_operation_seconds_sum": sum(r["elapsed_seconds"] for r in admin["steps"] if r["operation"] in ("fit-dev", "fit-final")),
        "timing_definitions": {"administrative_wall": "entire_queue_including_preparation_IO_import_training_and_replay",
                               "fit_loop": "sum_of_COMPLETE_elapsed_seconds_after_context_optimizer_creation_no_parallel_speedup_claim",
                               "fit_administrative": "sum_of_subprocess_wall_times_including_startup_IO_and_setup"},
        "resources": {"workers": state["workers"], "threads_per_worker": state["threads_per_worker"],
                      "device": "CPU", "expression_transferred_to_server": False,
                      "peak_sampled_process_RSS_bytes": max(t["process_RSS_bytes"] for t in telemetry),
                      "minimum_sampled_available_RAM_bytes": min(t["available_RAM_bytes"] for t in telemetry),
                      "minimum_sampled_disk_free_bytes": min(t["disk_free_bytes"] for t in telemetry),
                      "telemetry_samples": len(telemetry)},
        "selected_schedules": [{k: r[k] for k in ("dataset", "fold", "selected_epoch", "budget_assessment")}
                               for r in schedule["selections"]],
        "cells": ledger_cells,
        "superseded_pretest_cost": {"public_receipt_sha256": superseded_sha,
            "completed_development_fits": 8, "interrupted_processes": 2,
            "interrupted_initialized_fit_cells": 1, "interrupted_before_fit_initialization": 1,
            "known_complete_optimizer_updates": 2560, "partial_optimizer_updates_exact": None,
            "partial_optimizer_updates_upper_bound_total": superseded["partial_optimizer_updates_upper_bound_total"],
            "formal_fits": 0, "official_test_evaluations": 0, "results_merged_into_v3": False},
        "producer_report_sha256": producer_sha, "source_canonical_sha256": SOURCE_CANONICAL,
        "convergence_not_proven": True}
    return {"hashes": hashes, "arithmetic": arithmetic, "producer_sha": producer_sha,
            "source_files": source_files, "ledger": ledger, "aggregates": aggregates,
            "fold_rows": sorted(fold_rows, key=lambda r: (r["dataset"], r["fold"], ARMS.index(r["arm"]))),
            "gene_rows": sorted(gene_rows, key=lambda r: (r["dataset"], r["fold"], ARMS.index(r["arm"]), r["gene_ordinal"])),
            "by_fold": by_fold, "maximum_replay_error": replay_error}


def fmt(value):
    return "未定义" if value is None else f"{value:.4f}"


def make_markdown(result):
    ledger, aggregate, by_fold = result["ledger"], result["aggregates"], result["by_fold"]
    lines = ["# S5-B1：共同冻结特征正式比较", "",
        "**本轮完成 20 次开发拟合、20 次正式拟合及 120 个已保存状态的实际重载回放。** 独立脚本只读取 JSON 并流式校验文件哈希，复算逐折、逐基因和官方聚合算术；没有再训练或执行模型前向。", "",
        "## 方法与预算", "",
        "五模型共享官方冻结 UNI2-h 输入、训练折内 StandardScaler/PCA256、50 基因 log1p 目标及坐标邻域。每个外折分别以该折五模型内部验证 MSE 均值选择一个共同轮数。全部 20 个正式状态保存后才打开官方测试。两向 CV 中同一来源可以是另一折的合法训练数据。", "",
        "QUERY4 的四个 slot 匿名。SPGEL50_ADAPT 仅用训练分区 50 基因值、ID、掩码作为辅助双向候选 ITC，候选分别重新条件化并重读原始证据；图像预测不读取表达。它没有组织监督，不是原 A5 完整医学程序版本，也不是 BLEEP/TRIPLEX 严格复现。", "",
        "| 数据 | 外折 | 五模型共同轮数 | 预算解释 |", "|---|---:|---:|---|"]
    for row in ledger["selected_schedules"]:
        lines.append(f"| {row['dataset']} | {row['fold']} | {row['selected_epoch']} | {row['budget_assessment']} |")
    lines += ["", "开发拟合均运行 1280 轮，保存 80/160/320/640/1280 状态。边界最优表示预算边界，不能当作收敛。选择更早也不等于证明优化充分。单训练切片的内部坐标块没有排除缓冲，不消除空间自相关；多切片留一验证不保证供体互斥。", "",
              "## 官方测试结果", "", "下表为先逐折四位小数、再按官方规则聚合两个折的 Pearson 均值与折间标准差。标准差不是置信区间。未定义基因保持为空，不填零或剔除。", "",
              "| 模型 | COAD Pearson | READ Pearson | 总参数 / 图像预测参数 |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        left, right = aggregate["COAD"][arm], aggregate["READ"][arm]
        lines.append(f"| {arm} | {fmt(left['pearson_mean'])} ± {fmt(left['pearson_std'])} | {fmt(right['pearson_mean'])} ± {fmt(right['pearson_std'])} | {PARAMETERS[arm][0]:,} / {PARAMETERS[arm][1]:,} |")
    lines += ["", "| 配对描述性差值 | COAD 折0 / 折1 | READ 折0 / 折1 |", "|---|---:|---:|"]
    for label, left, right in (("GraphSAGE − Identity", "GRAPHSAGE", "IDENTITY"),
                                ("SPGEL50_ADAPT − QUERY4", "SPGEL50_ADAPT", "QUERY4")):
        vals = {}
        for d in ("COAD", "READ"):
            vals[d] = [None if by_fold[d, f, left]["pearson_mean"] is None or by_fold[d, f, right]["pearson_mean"] is None
                       else by_fold[d, f, left]["pearson_mean"] - by_fold[d, f, right]["pearson_mean"] for f in (0, 1)]
        lines.append(f"| {label} | {fmt(vals['COAD'][0])} / {fmt(vals['COAD'][1])} | {fmt(vals['READ'][0])} / {fmt(vals['READ'][1])} |")
    lines += ["", "这些差值只描述同折性能变化。两个折、一个 seed 不能提供强泛化结论，也不能证明医学程序语义或双向生物学协同。", "",
        "## 实际成本与复算边界", "",
        f"新 v3 开发优化更新 25,600 次，正式更新 {ledger['optimizer_updates']['final']:,} 次；总计 {ledger['optimizer_updates']['total']:,} 次。管理队列墙钟 {ledger['administrative_wall_elapsed_seconds'] / 3600:.3f} 小时；40 个拟合循环的耗时之和 {ledger['fit_loop_seconds_sum'] / 3600:.3f} 小时，包含并行重叠，不能作为串行墙钟。运行使用两个 CPU worker，每个两线程。", "",
        "首版被静态目标审查否决并单独归档：8 次开发拟合完成，两个进程中断，其中一个已创建拟合目录、另一个在拟合初始化前停止；不能称为 10 次已执行拟合。已完成部分 2560 次更新，部分拟合更新数未知、上界 320。首版正式拟合和官方测试均为零，成绩没有并入新结果。", "",
        f"生产流程实际重载回放 120 个唯一状态，最大预测绝对误差 {result['maximum_replay_error']:.9g}。独立复算核验这些回放条目与当前文件、状态、完成凭据的哈希集合，算术比较最大绝对误差 {result['arithmetic'].maximum_error:.9g}。独立脚本没有解包权重或数组，因而不声称自己从逐 spot 数组重新计算了 Pearson。", "",
        "## 证据与局限", "",
        "优势是共享输入预算、非自身邻居的 GraphSAGE/同参数 Identity 对照、逐外折内部选择、50 基因辅助目标隔离和完整状态回放。风险是供体少、单 seed、单切片内部空间相关性、预算上限及语义监督缺失。任何正负结果均不能直接推广到 DSS 生存互补性、细胞来源正确性或原 A5 完整版本。", "",
        "本轮不根据官方测试继续搜索轮数、架构或辅助损失。BLEEP、TRIPLEX 的严格原始输入和实现资格仍以独立资格报告为准。", "",
        "公开制品：", "",
        "- [20 个逐折指标](S5B1_FOLD_METRICS.csv)",
        "- [1000 个逐基因指标](S5B1_GENE_METRICS.csv)",
        "- [实际成本台账](S5B1_ACTUAL_LEDGER.json)",
        "- [独立算术及哈希核验](S5B1_INDEPENDENT_ARITHMETIC_AUDIT.json)",
        "- [锁定协议](S5B1_PROTOCOL.md)",
        "- [官方方法资格](S5B_OFFICIAL_QUALIFICATION.md)", ""]
    return "\n".join(lines)


def publish_review(args, result):
    output = args.output_dir.resolve()
    root = args.root.resolve()
    need(output.is_relative_to(root) and output.relative_to(root).as_posix() == "reports/spgel_s5",
         "REVIEW_PUBLIC_OUTPUT_SCOPE_MUST_BE_REPORTS_SPGEL_S5")
    payloads = {
        "S5B1_FOLD_METRICS.csv": csv_text(result["fold_rows"]),
        "S5B1_GENE_METRICS.csv": csv_text(result["gene_rows"]),
        "S5B1_ACTUAL_LEDGER.json": json_text(result["ledger"]),
        "S5B1_RESULTS.md": make_markdown(result),
    }
    audit_name = "S5B1_INDEPENDENT_ARITHMETIC_AUDIT.json"
    need(all(not (output / name).exists() for name in (*payloads, audit_name)), "INDEPENDENT_REVIEW_OUTPUT_ALREADY_EXISTS_NO_OVERWRITE")
    script_sha = result["hashes"].get(Path(__file__), "independent_reviewer_script")
    public_outputs = {"reports/spgel_s5/" + name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                      for name, content in payloads.items()}
    for name in ("S5B1_PROTOCOL.md", "S5B1_SUPERSEDED_PRETEST_COST.json"):
        public_outputs["reports/spgel_s5/" + name] = result["hashes"].get(output / name, "public_context:" + name)
    audit = {"schema": "spgel_s5b1_independent_arithmetic_audit_v1", "status": "PASS_INDEPENDENT_METADATA_HASH_AND_ARITHMETIC_REVIEW",
        "counts": {"development_fits": 20, "final_fits": 20, "saved_state_replays": 120,
                   "fold_metric_rows": 20, "gene_metric_rows": 1000},
        "source_files": result["source_files"], "source_canonical_sha256": SOURCE_CANONICAL,
        "producer_report_sha256": result["producer_sha"], "reviewer_script_sha256": script_sha,
        "public_outputs": public_outputs,
        "operations": {"model_fits": 0, "model_forwards": 0, "array_or_model_deserializations": 0,
                       "current_artifact_file_hashing": True},
        "maximum_absolute_arithmetic_error": result["arithmetic"].maximum_error,
        "arithmetic_scalar_comparisons": result["arithmetic"].scalar_comparisons,
        "nonfinite_R2_quantile_checks_not_reconstructible_from_JSON_null":
            result["arithmetic"].nonfinite_R2_quantile_checks_not_reconstructible,
        "production_actual_replay_maximum_prediction_error": result["maximum_replay_error"],
        "convergence_not_proven": True, "SOTA_strict_reproduction_claimed": False,
        "own_fold_selection_recomputed_independently": True,
        "exact_120_unique_replay_tuple_set_verified": True,
        "raw_prediction_Pearson_recomputed_by_reviewer": False,
        "review_scope": "recorded_per_gene_metric_arithmetic_and_current_artifact_SHA256_binding_not_private_array_recomputation",
        "nonfinite_policy": "preserve_null_never_impute_or_drop_genes",
        "artifact_hash_inventory": result["hashes"].inventory}
    payloads[audit_name] = json_text(audit)
    result["hashes"].unchanged()
    staging = Path(tempfile.mkdtemp(prefix="S5B1_INDEPENDENT_REVIEW_", dir=root / "tmp"))
    try:
        for name, content in payloads.items():
            (staging / name).write_text(content, encoding="utf-8", newline="")
        output.mkdir(parents=True, exist_ok=True)
        for name in payloads:
            need(not (output / name).exists(), "PUBLIC_OUTPUT_APPEARED_DURING_REVIEW")
        for name in payloads:
            os.replace(staging / name, output / name)
    finally:
        shutil.rmtree(staging)
    return {"status": audit["status"], "counts": audit["counts"],
            "maximum_absolute_arithmetic_error": audit["maximum_absolute_arithmetic_error"],
            "outputs": ["reports/spgel_s5/" + name for name in payloads]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--superseded-cost", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_all(args)
        receipt = publish_review(args, result)
        print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))
        return 0
    except ReviewError as error:
        print(json.dumps({"status": "REVIEW_NOT_PASSED_NO_PUBLIC_RESULT_WRITTEN", "code": str(error)}))
        return 3


if __name__ == "__main__":
    sys.exit(main())

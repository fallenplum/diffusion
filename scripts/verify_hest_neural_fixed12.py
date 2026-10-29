"""Independent metadata arithmetic; never loads checkpoints or patient arrays.

The producer already performed actual state forwards. This reviewer checks its
receipts and current artifact bytes, and recomputes all possible arithmetic
from saved per-gene metrics without another model call or fitted estimator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("MLP", "SPATIAL_GNN", "IDENTITY_GNN")
EPOCHS = (10, 20, 40)
CHECKS = 0
ARITHMETIC_CHECKS = 0


def check(condition, code):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(code)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            check(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON_LITERAL")))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return round(value, 4) if isinstance(value, float) else value


def number(value):
    return np.nan if value is None else float(value)


def clean(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def close(actual, expected, code):
    global ARITHMETIC_CHECKS
    ARITHMETIC_CHECKS += 1
    check((actual is None and not np.isfinite(expected))
        or (actual is not None and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=2e-12)), code)


def aggregate(split_metrics):
    """Independent reconstruction of split-round4 -> mean -> final-round4."""
    genes = [x["name"] for x in split_metrics[0]["pearson_corrs"]]
    check(all([x["name"] for x in m["pearson_corrs"]] == genes for m in split_metrics), "GENE_ORDER_ACROSS_FOLDS")
    rows = []
    for index, gene in enumerate(genes):
        values = [number(m["pearson_corrs"][index]["pearson_corr"]) for m in split_metrics]
        rows.append(dict(name=gene, pearson_corrs=values, mean=float(np.mean(values)), std=float(np.std(values))))
    rows.sort(key=lambda r: r["mean"] if np.isfinite(r["mean"]) else -np.inf, reverse=True)
    means = [number(m["pearson_mean"]) for m in split_metrics]
    return clean(rounded(dict(pearson_corrs=rows, pearson_mean=float(np.mean(means)),
        pearson_std=float(np.std(means)), mean_per_split=means)))


def graph_checks(graph, arm, rows, specimens):
    if arm == "MLP":
        check(graph is None, "MLP_HAS_NO_GRAPH")
        return
    check(graph["schema"] == "spgel_hest_neural_pixel_knn_v1"
        and graph["cross_specimen_edges"] == 0 and graph["physical_radius_used"] is False
        and graph["rows"] == rows and len(graph["groups"]) == specimens,
        "PIXEL_GRAPH_SCOPE_OR_SUPPORT")
    check(sum(g["rows"] for g in graph["groups"]) == rows
        and len({g["specimen_code"] for g in graph["groups"]}) == specimens, "GRAPH_GROUP_ROWS")
    identity = arm == "IDENTITY_GNN"
    check(graph["maximum_neighbours_including_self"] == (1 if identity else 16), "GRAPH_MAXIMUM_DEGREE")
    check(graph["method"] == ("IDENTITY" if identity else "SELF_PLUS_15_PIXEL_KNN_COMPLETE_DISTANCE_TIES"), "GRAPH_METHOD")
    for group in graph["groups"]:
        check(group["degree"] == (1 if identity else min(16, group["rows"])), "REAL_NEIGHBOUR_COUNT")
    check(graph["edges"] == sum(g["rows"] * g["degree"] for g in graph["groups"]), "GRAPH_EDGE_COUNT")


def verify(args):
    repo = Path(__file__).resolve().parents[1]
    private = args.private.resolve()
    freeze, report = read(args.freeze), read(args.report)
    lock = read(private / "HEST_NEURAL_RUN_LOCK.json")
    replay = read(private / "NEURAL_REPLAY_AUDIT.json")
    check(report["schema"] == "spgel_hest_neural_fixed12_report_v1"
        and report["status"] == "COMPLETE_12_FIXED40_FITS_AND_36_ACTUAL_REPLAYS", "PRODUCER_REPORT_COMPLETE")
    check(canonical(freeze["sources"]) == freeze["sources_canonical_sha256"], "FREEZE_CANONICAL_SHA")
    check(freeze["sources"] == lock["source_files"] == report["code_sha256"], "SCIENTIFIC_SOURCE_MANIFEST_BINDING")
    for path, expected in freeze["sources"].items():
        check(sha(repo / path) == expected, "CURRENT_FROZEN_SCIENTIFIC_BYTES")
    for path, expected in lock["inherited_Ridge_source_files"].items():
        check(sha(repo / path) == expected, "OLD_RIDGE_SCIENCE_UNCHANGED")
    check(canonical({k: v for k, v in lock.items() if k != "internal_contract_sha256"}) == lock["internal_contract_sha256"], "NEURAL_LOCK_CANONICAL_SHA")
    check(lock["internal_contract_sha256"] == report["internal_contract_sha256"] == replay["internal_contract_sha256"], "RUN_CONTRACT_LINEAGE")
    check(report["config_sha256"] == lock["config_sha256"] == sha(repo / "configs/spgel_hest_neural/pilot_v1.json"), "RECIPE_CURRENT_SHA")
    check(report["assets_sha256"] == lock["assets_sha256"] == sha(args.assets), "OFFICIAL_ASSET_POINTER_CURRENT_SHA")
    check(sha(private / "NEURAL_REPLAY_AUDIT.json") == report["replay_audit_sha256"], "REPLAY_FILE_CURRENT_SHA")
    check(lock["budget"] == dict(neural_fits=12, epochs_per_fit=40, optimizer_updates_per_fit=40,
        optimizer_updates_total=480, snapshot_epochs=[10, 20, 40], actual_state_replays=36,
        new_unsupervised_PCA_reconstructions=4, new_Ridge_fits=0, automatic_retry_fits=0), "EXACT_REGISTERED_BUDGET")
    check(lock["actual_preparation_new_PCA_fits"] == 4 and lock["actual_preparation_new_neural_fits"]
        == lock["actual_preparation_new_Ridge_fits"] == lock["actual_preparation_new_scaler_fits"] == 0, "PREPARATION_FIT_COUNTS")
    check(report["ledger"] == dict(new_neural_fits=12, actual_optimizer_updates=480,
        actual_saved_state_replays=36, new_unsupervised_PCA_reconstructions=4, new_scaler_fits=0,
        new_Ridge_fits=0, old_source_weight_loads=0, automatic_retry_fits=0), "PRODUCER_ACTUAL_COUNTS")
    check(replay["status"] == "PASS_36_ACTUAL_SAVED_STATE_REPLAYS" and replay["passed_states"] == 36
        and replay["passed_fits"] == 12 and len(replay["units"]) == 36 and replay["optimizer_updates"]
        == replay["new_neural_fits"] == replay["new_Ridge_fits"] == replay["new_PCA_fits"] == 0,
        "REPLAY_ACTUAL_COUNTS")
    replay_index = {(r["dataset"], r["fold"], r["arm"], r["epoch"]): r for r in replay["units"]}
    check(len(replay_index) == 36, "REPLAY_NO_DUPLICATE_KEYS")
    public = {(r["dataset"], r["fold"], r["arm"]): r for r in report["per_cell"]}
    check(len(public) == len(report["per_cell"]) == 12, "PUBLIC_TWELVE_UNIQUE_CELLS")
    partitions = {(r["dataset"], r["fold"]): r for r in lock["partitions"]}
    check(set(partitions) == {(d, f) for d in ("COAD", "READ") for f in (0, 1)}, "FOUR_INPUT_PARTITIONS")
    profile_path = private / "ZERO_OPTIMIZER_PROFILE.private.json"
    check(sha(profile_path) == lock["zero_optimizer_profile_sha256"], "ZERO_OPTIMIZER_PROFILE_SHA")
    profile = read(profile_path)
    check(profile["optimizer_steps"] == profile["new_neural_fits"] == 0
        and profile["point_in_time_two_worker_RAM_gate_passed"] is True
        and all(r["model_state_exactly_unchanged"] and r["optimizer_instances"] == r["optimizer_steps"]
            == r["new_fits"] == 0 for r in profile["units"]), "ACTUAL_ZERO_OPTIMIZER_QUALIFICATION")
    check(sha(args.ridge_private / "HEST_RUN_LOCK.json") == lock["original_Ridge_lock_sha256"], "ORIGINAL_RIDGE_LOCK_UNCHANGED")
    metrics, fold_rows, gene_rows = {}, [], []
    updates = fits = cache_PCA = 0
    for (dataset, fold), expected in partitions.items():
        cache_folder = private / "inputs" / expected["partition"]
        cache_path = cache_folder / "UNSUPERVISED_CACHE.private.json"
        check(sha(cache_path) == expected["cache_receipt_sha256"], "SHARED_CACHE_RECEIPT_SHA")
        cache = read(cache_path)
        check(sha(cache_folder / "inputs.private.npz") == expected["input_cache_sha256"] == cache["input_cache_sha256"], "SHARED_INPUT_BYTES_CURRENT_SHA")
        check(sha(cache_folder / "unsupervised_pipeline.private.joblib") == cache["saved_pipeline_sha256"], "UNSUPERVISED_PIPELINE_CURRENT_SHA")
        old_folder = args.ridge_private / "cells" / expected["partition"]
        check(sha(old_folder / "model.private.joblib") == cache["original_Ridge_model_sha256"]
            and sha(old_folder / "COMPLETE.json") == cache["original_Ridge_COMPLETE_sha256"], "ORIGINAL_RIDGE_BYTES_UNCHANGED")
        old = read(old_folder / "COMPLETE.json")
        check(old["train_ordering_sha256"] == cache["train_ordering_sha256"] == expected["train_ordering_sha256"]
            and old["test_ordering_sha256"] == cache["test_ordering_sha256"] == expected["test_ordering_sha256"], "UNCHANGED_INPUT_IDENTITY_ORDER")
        parity = cache["PCA_reconstruction"]
        check(parity["new_PCA_fits"] == 1 and parity["new_scaler_fits"] == parity["new_Ridge_fits"]
            == parity["new_neural_fits"] == 0 and parity["old_train_US_persisted"] is False
            and parity["old_train_US_bitwise_comparison_claimed"] is False
            and parity["test_uses_saved_pipeline_transform"] is True, "UNSUPERVISED_US_RECONSTRUCTION_SCOPE")
        for value in parity["parameters"].values():
            check(value["exact_array_equal"] is True and value["reconstructed_sha256"] == value["saved_sha256"], "PCA_PARAMETER_EXACT_EQUAL_RECEIPT")
        cache_PCA += parity["new_PCA_fits"]
        complete = {}
        for arm in ARMS:
            folder = private / "cells" / f"{expected['partition']}_{arm}"
            done_path = folder / "COMPLETE.json"
            done = read(done_path)
            complete[arm] = done
            check(done["dataset"] == dataset and done["fold"] == fold and done["arm"] == arm
                and done["seed"] == 1 and done["status"] == "COMPLETE_ONE_FIXED40_NEURAL_FIT", "CELL_IDENTITY_STATUS")
            check(done["actual_epochs"] == done["actual_optimizer_updates"] == 40
                and done["new_neural_fits"] == 1 and done["new_PCA_fits"] == done["new_scaler_fits"]
                == done["new_Ridge_fits"] == 0, "CELL_ACTUAL_FIT_BUDGET")
            check(done["input_cache_sha256"] == expected["input_cache_sha256"]
                and done["internal_contract_sha256"] == lock["internal_contract_sha256"], "SAME_INPUT_AND_RUN_CONTRACT")
            check(done["train_rows"] == expected["train_rows"] and done["test_rows"] == expected["test_rows"], "EXACT_OFFICIAL_SPOT_SUPPORT")
            check(done["test_RNA_in_predict"] is False and done["test_labels_select_epoch"] is False
                and done["source_pretrained_weights_loaded"] is False and done["train_loss_only"] is True,
                "NO_TEST_LABEL_FORWARD_SELECTION_OR_SOURCE_WEIGHTS")
            support = next(r for r in report["tasks"][dataset][arm]["fold_support"] if r["fold"] == fold)
            check(support == {key: done[key] for key in ("fold", "train_rows", "test_rows", "parameters", "actual_epochs", "elapsed_seconds")}, "PUBLIC_FOLD_SUPPORT_AND_TIMING")
            check(done["parameters"] == (55858 if arm == "MLP" else 72242), "ACTUAL_EFFECTIVE_PARAMETER_COUNT")
            curve = done["learning_curve"]
            check(len(curve) == 40 and [r["epoch"] for r in curve] == list(range(1, 41))
                and [r["actual_optimizer_updates"] for r in curve] == list(range(1, 41)), "EXACT_FORTY_TRAIN_UPDATES")
            for row in curve:
                check(np.isfinite(row["pre_update_train_MSE"]) and row["pre_update_train_MSE"] >= 0
                    and all(np.isfinite(g) and g >= 0 for g in row["gradient_norms"].values()), "FINITE_TRAIN_LOSS_AND_GRADIENT_DIAGNOSTICS")
            graph_checks(done["train_graph"], arm, expected["train_rows"], expected["train_specimens"])
            graph_checks(done["test_graph"], arm, expected["test_rows"], expected["test_specimens"])
            pub = public[(dataset, fold, arm)]
            for key in ("snapshots", "learning_curve", "train_graph", "test_graph", "parameters"):
                check(pub[key] == done[key], "PUBLIC_PRIVATE_CELL_METADATA_EQUAL")
            check([s["epoch"] for s in done["snapshots"]] == list(EPOCHS), "THREE_FIXED_DIAGNOSTIC_SNAPSHOTS")
            for snapshot in done["snapshots"]:
                epoch = snapshot["epoch"]
                state = folder / f"state_epoch{epoch:02d}.private.pt"
                prediction = folder / f"predictions_epoch{epoch:02d}.private.npz"
                check(sha(state) == snapshot["state_sha256"] and sha(prediction) == snapshot["predictions_sha256"], "CURRENT_SNAPSHOT_AND_PREDICTION_BYTES")
                replayed = replay_index[(dataset, fold, arm, epoch)]
                check(replayed["status"] == "PASS" and replayed["actual_saved_state_loaded"] is True
                    and replayed["actual_train_and_test_forward_replayed"] is True
                    and replayed["new_neural_fits"] == replayed["new_Ridge_fits"] == replayed["new_PCA_fits"] == 0
                    and replayed["COMPLETE_sha256"] == sha(done_path)
                    and replayed["state_sha256"] == snapshot["state_sha256"]
                    and replayed["predictions_sha256"] == snapshot["predictions_sha256"]
                    and replayed["input_cache_sha256"] == expected["input_cache_sha256"]
                    and np.isfinite(replayed["maximum_prediction_error"])
                    and 0 <= replayed["maximum_prediction_error"] <= 1e-6, "ACTUAL_STATE_FORWARD_REPLAY_AND_BYTES")
                raw = snapshot["full_precision_metrics"]
                split = snapshot["official_rounded4_split_metrics"]
                check(split == rounded(raw), "SPLIT_FULL_TO_ROUND4_ALL_FIELDS")
                genes = [g["name"] for g in raw["pearson_corrs"]]
                corr = np.asarray([number(g["pearson_corr"]) for g in raw["pearson_corrs"]])
                check(genes == expected["genes"] and len(set(genes)) == 50
                    and len(raw["l2_errors"]) == len(raw["r2_scores"]) == 50, "EXACT_FIFTY_GENE_ORDER_SUPPORT")
                close(raw["pearson_mean"], np.mean(corr), "FIFTY_GENE_PEARSON_MEAN")
                close(raw["pearson_std"], np.std(corr), "FIFTY_GENE_PEARSON_STD")
                for prefix, values in (("l2_error", raw["l2_errors"]), ("r2_score", raw["r2_scores"])):
                    values = np.asarray([number(v) for v in values])
                    for suffix, value in zip(("q1", "q2", "q3"), (np.percentile(values, 25), np.median(values), np.percentile(values, 75))):
                        close(raw[prefix + "_" + suffix], value, "GENE_METRIC_QUANTILE")
                supplementary = snapshot["supplementary"]
                close(supplementary["MSE"], np.mean(raw["l2_errors"]), "ALL_SPOT_GENE_MSE_FROM_GENE_MEANS")
                close(supplementary["median_per_gene_Pearson"], np.median(corr), "STRICT_GENE_MEDIAN_PEARSON")
                check(supplementary["defined_Pearson_genes"] == int(np.isfinite(corr).sum())
                    and supplementary["total_genes"] == 50 and supplementary["mean_drops_undefined_genes"] is False,
                    "UNDEFINED_GENE_SUPPORT_NOT_IMPUTED_OR_DROPPED")
                metrics[(dataset, fold, arm, epoch)] = snapshot
                fold_rows.append(dict(dataset=dataset, fold=fold, seed=1, arm=arm, epoch=epoch,
                    primary_epoch=(epoch == 40), train_rows=expected["train_rows"], test_rows=expected["test_rows"],
                    parameters=done["parameters"], defined_Pearson_genes=supplementary["defined_Pearson_genes"],
                    total_genes=50, Pearson_mean_full=raw["pearson_mean"], Pearson_std_full=raw["pearson_std"],
                    Pearson_mean_split_round4=split["pearson_mean"], MSE=supplementary["MSE"],
                    MAE=supplementary["MAE"], strict_median_gene_Pearson=supplementary["median_per_gene_Pearson"]))
                for index, gene in enumerate(genes):
                    gene_rows.append(dict(dataset=dataset, fold=fold, seed=1, arm=arm, epoch=epoch,
                        primary_epoch=(epoch == 40), gene=gene, Pearson_full=raw["pearson_corrs"][index]["pearson_corr"],
                        Pearson_split_round4=split["pearson_corrs"][index]["pearson_corr"],
                        MSE_full=raw["l2_errors"][index], R2_full=raw["r2_scores"][index]))
            updates += done["actual_optimizer_updates"]
            fits += done["new_neural_fits"]
        check(complete["SPATIAL_GNN"]["initial_state_tensor_sha256"] == complete["IDENTITY_GNN"]["initial_state_tensor_sha256"], "GRAPH_IDENTITY_ALL_INITIAL_PARAMETERS_EQUAL")
        for projection in ("input_projection.weight", "input_projection.bias", "output_projection.weight", "output_projection.bias"):
            check(len({complete[a]["initial_state_tensor_sha256"][projection] for a in ARMS}) == 1, "THREE_ARMS_COMMON_INPUT_OUTPUT_INITIALIZATION")
    check(fits == 12 and updates == 480 and cache_PCA == 4 and len(fold_rows) == 36 and len(gene_rows) == 1800, "FINAL_ACTUAL_COUNTS")
    primary = {}
    for dataset in ("COAD", "READ"):
        primary[dataset] = {}
        for arm in ARMS:
            for epoch in EPOCHS:
                snapshots = [metrics[(dataset, fold, arm, epoch)] for fold in (0, 1)]
                recomputed = aggregate([s["official_rounded4_split_metrics"] for s in snapshots])
                check(recomputed == report["tasks"][dataset][arm]["diagnostics_by_epoch"][str(epoch)], "OFFICIAL_TWO_FOLD_ROUND4_DIAGNOSTIC_AGGREGATION")
                if epoch == 40:
                    check(recomputed == report["tasks"][dataset][arm]["official_round4"], "PRIMARY40_OFFICIAL_AGGREGATION")
                    full = [number(s["full_precision_metrics"]["pearson_mean"]) for s in snapshots]
                    primary[dataset][arm] = dict(official_Pearson_mean_round4=recomputed["pearson_mean"],
                        full_precision_equal_fold_Pearson_mean=clean(float(np.mean(full))),
                        full_precision_fold_Pearson_means=clean(full), genes_each_fold=50, official_folds=2,
                        primary_epoch=40, primary_epoch_selected_on_test=False)
    check(report["scope"]["fixed40_budget_proves_convergence"] is False
        and report["scope"]["new_independent_cohort_claimed"] is False
        and report["scope"]["COAD_preview_donor_identity"] == "UNCONFIRMED"
        and report["scope"]["READ_donors_previously_used_in_source_development"] is True, "HONEST_FIXED_BUDGET_AND_COHORT_BOUNDARIES")
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_refs = {}
    for name, rows in (("HEST_NEURAL_FIXED12_FOLD_METRICS_20261029.csv", fold_rows),
                       ("HEST_NEURAL_FIXED12_GENE_METRICS_20261029.csv", gene_rows)):
        target = args.output / name
        check(not target.exists(), "INDEPENDENT_OUTPUTS_MUST_BE_NEW")
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        artifact_refs[name] = dict(sha256=sha(target), rows=len(rows))
    result = dict(schema="hest_neural_fixed12_metadata_arithmetic_audit_v1",
        status="PASS_12_FITS_480_UPDATES_36_ACTUAL_REPLAYS_AND_METRIC_ARITHMETIC",
        independent_review_script_sha256=sha(Path(__file__)), metadata_checks=CHECKS,
        independent_scalar_arithmetic_assertions=ARITHMETIC_CHECKS,
        producer_report_sha256=sha(args.report), source_freeze_manifest_sha256=sha(args.freeze),
        source_canonical_sha256=freeze["sources_canonical_sha256"],
        neural_lock_file_sha256=sha(private / "HEST_NEURAL_RUN_LOCK.json"),
        internal_contract_sha256=lock["internal_contract_sha256"],
        actual_replay_file_sha256=sha(private / "NEURAL_REPLAY_AUDIT.json"),
        actual_counts=dict(neural_fits=12, optimizer_updates=480, saved_state_replays=36,
            new_unsupervised_PCA_reconstructions=4, new_Ridge_fits=0),
        reviewer_operations=dict(model_fits=0, model_forwards=0, PCA_fits=0,
            model_or_array_artifact_deserializations=0, current_artifact_file_hashing=True,
            actual_replays_run_by_producer_not_reviewer=True),
        outputs=artifact_refs, primary_epoch40=primary,
        boundaries=dict(targets_are_official50genes_log1p_not_four_program_CP10K=True,
            no_test_target_used_for_forward_or_selection=True, graph_scope="same_slide_same_partition_pixel_k16",
            COAD_preview_donor_identity="UNCONFIRMED_OFFICIAL_SETTING_ONLY",
            READ_known_donors="TWO_DONORS_WITH_PRIOR_SOURCE_DEVELOPMENT_EXPOSURE",
            new_independent_cohort=False, fixed40_epochs_prove_convergence=False,
            train_US_reconstruction_checks_saved_PCA_basis_not_unavailable_old_US_tensor=True,
            two_folds_and_seed1_do_not_establish_broad_generalization=True))
    target = args.output / "HEST_NEURAL_FIXED12_INDEPENDENT_AUDIT_20261029.json"
    check(not target.exists(), "INDEPENDENT_AUDIT_MUST_BE_NEW")
    target.write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(status=result["status"], metadata_checks=CHECKS, audit_sha256=sha(target),
        fold_metric_rows=len(fold_rows), gene_metric_rows=len(gene_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--ridge-private", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args()), allow_nan=False))


if __name__ == "__main__":
    main()

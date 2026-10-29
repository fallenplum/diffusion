"""Finite source-only execution, actual saved-state replay and provenance.

The training loop is minibatch Adam: forty epochs are NOT forty updates.
Every image prediction is computed before any optional paired gene scoring.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
import torch

from .contracts import (PROGRAMS, GeneBatch, SourceStatistics, array_hash, canonical_hash,
                        load_registered_r6_cache, require, sha256)
from .geometry import hard_negative_rows
from .losses import source_objective
from .metrics import image_prediction_metrics, localization_metrics, stratified_within_slide
from .models import ARMS, NEURAL_ARMS, SpatialProgramEncoder, match_image_mlp_capacity


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "configs/spgel/pilot_v1.json", "configs/spgel/s4_export_v1.json",
    "src/spgel/__init__.py", "src/spgel/contracts.py", "src/spgel/geometry.py", "src/spgel/metrics.py",
    "src/spgel/models.py", "src/spgel/losses.py", "src/spgel/training.py", "src/spgel/interface.py",
    "scripts/spgel_source_fit.py", "scripts/spgel_source_replay.py", "scripts/spgel_source_qualify.py",
    "scripts/spgel_source_export.py", "scripts/spgel_source_report.py",
    "tests/test_spgel_contracts.py", "tests/test_spgel_models.py", "tests/test_spgel_training.py",
    "tests/test_spgel_interface.py", "tests/test_spgel_report.py")


def read_json(path):
    def pairs(rows):
        result = {}
        for k, v in rows:
            require(k not in result, "DUPLICATE_JSON_KEY")
            result[k] = v
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def write_json(path, value, *, progress=False):
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if progress:
        temporary = path.with_name(path.name + ".writing")
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(data)


def torch_write(path, value):
    path = Path(path)
    require(not path.exists(), "EXISTING_MODEL_OR_CHECKPOINT_PRESERVED")
    with path.open("xb") as stream:
        torch.save(value, stream)


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def configure_runtime():
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (None, ":4096:8"), "CUBLAS_RUNTIME_CONFLICT")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


def validate_recipe(config):
    require(config.get("schema") == "spgel_source_pilot_recipe_v1"
            and tuple(config["trainable_arms"]) == ARMS and config["primary_candidate"] == "S_A5"
            and config["heldout_donor_codes"] == list(range(6)) and config["seeds"] == [31, 47]
            and tuple(config["program_order"]) == PROGRAMS, "SOURCE_FIXED_PILOT_RECIPE_CHANGED")
    t, loss = config["training"], config["loss"]
    require(t["epochs"] == 40 and t["batch_size"] == 512 and t["drop_last"] is False
            and t["save_epochs"] == [10, 20, 40] and t["primary_epoch"] == 40
            and t["model_or_epoch_selection_from_heldout"] is False
            and t["optimizer"] == "Adam" and t["learning_rate"] == .001 and t["weight_decay"] == .0001
            and t["dropout"] == 0 and t["gradient_clip_norm"] == 5., "SOURCE_TRAINING_BUDGET_OR_OPTIMIZER_CHANGED")
    require(loss["program_Huber_weight"] == loss["Huber_delta"] == 1
            and loss["ITC_weight_A3_A4_A5"] == loss["ITC_temperature"] == loss["tissue_CE_weight_A4_A5"] == .1
            and loss["masked_gene_weight"] == loss["matching_weight"] == loss["spatial_smoothing_weight"] == 0,
            "SOURCE_PILOT_LOSS_PACKAGE_CHANGED")
    require(config["budget"]["pilot_logical_cells"] == 72 and config["budget"]["pilot_actual_replay_states"] == 192
            and config["scope"]["DSS_or_PFI_read"] is False and config["scope"]["R8_results_influence_source"] is False,
            "SOURCE_SURVIVAL_ISOLATION_OR_FIXED_MATRIX_CHANGED")


def load_assets(assets_path, config):
    assets = read_json(assets_path)
    require(set(assets) == {"cache_root", "fresh_source_receipt"}, "SOURCE_ONLY_PRIVATE_ASSET_KEYS_REQUIRED")
    binding = config["source_binding"]
    receipt = Path(assets["fresh_source_receipt"])
    require(sha256(receipt) == binding["fresh_requalification_sha256"], "FRESH_REQUALIFICATION_RECEIPT_SHA_CHANGED")
    fresh = read_json(receipt)
    require(fresh.get("schema") == "spgel_source_requalification_v1"
            and fresh.get("status") == "PASS_FRESH_RAW_FULL_LIBRARY_AND_EXISTING_CACHE_REPLAY"
            and fresh.get("source_cache_manifest_sha256") == binding["cache_manifest_sha256"]
            and fresh.get("full_library_counts_replayed_from_raw") is True
            and fresh.get("program_union_count_cache_shape") == [13011, 760]
            and fresh.get("cache_is_full_36601_gene_matrix") is False
            and fresh.get("DSS_or_PFI_read") is False and fresh.get("original_ST_test_read") is False
            and fresh.get("historical_ST_test_is_new_independent_confirmation") is False
            and fresh.get("source_files_mutated") is False
            and fresh.get("new_model_fits") == fresh.get("new_model_forwards") == 0,
            "FRESH_SOURCE_REQUALIFICATION_NOT_ACTUAL_OR_OUTCOME_FREE")
    exact = fresh.get("fresh_cache_fields_exact", {})
    names = {"features_unit", "measured_mask", "log1p_cp10k", "program_targets", "tissue_y", "program_up",
        "program_down", "coords_um", "section_codes", "donor_codes", "library_size", "n_detected_genes",
        "neighbors", "neighbor_counts", "neighbor_mean_of_unit", "neighbor_mean_unit"}
    require(set(exact) == names and all(v is True for v in exact.values()), "FRESH_CACHE_FIELDS_NOT_EXACT")
    cache = Path(assets["cache_root"])
    require(sha256(cache / "source_arrays.private.npz") == binding["source_arrays_sha256"]
            and sha256(cache / "gene_counts.private.npz") == binding["gene_counts_sha256"], "SOURCE_PAYLOAD_BINDING_CHANGED")
    source = load_registered_r6_cache(cache, expected_manifest_sha256=binding["cache_manifest_sha256"])
    require(source.features.shape == (13011, 1536) and source.genes.shape == (13011, 760)
            and len(np.unique(source.donor)) == 6 and len(np.unique(source.section)) == 12
            and int(source.up.sum() + source.down.sum()) == 765, "QUALIFIED_SOURCE_DIMENSIONS_CHANGED")
    return source


def private_directory(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(ROOT) and not any((p / ".git").exists() for p in (path, *path.parents)),
            "SOURCE_PRIVATE_OUTPUT_INSIDE_GIT_WORKSPACE")
    # Python 3.13's Windows 0700 ACL can exclude the desktop sandbox identity.
    # Windows inherits the already private destination ACL; POSIX is owner-only.
    path.mkdir(parents=True, exist_ok=True, mode=0o777 if os.name == "nt" else 0o700)
    return path


def budget_for_source(source):
    rows = []
    for donor in range(6):
        train, test = source.split(donor)
        rows.append(dict(heldout_donor=donor, train_rows=len(train), heldout_rows=len(test),
                         updates_per_neural_fit=40 * math.ceil(len(train) / 512)))
    return dict(pilot_logical_cells=72, neural_fits=60, convex_fits=12,
        neural_optimizer_updates=sum(x["updates_per_neural_fit"] for x in rows) * 2 * 5,
        actual_saved_states=192, folds=rows, s4_neural_fits=2,
        s4_neural_optimizer_updates=2 * 40 * math.ceil(len(source.features) / 512), s4_saved_states=6)


def validate_lock(private, config_path, config):
    lock = read_json(Path(private) / "SOURCE_RUN_LOCK.json")
    require(lock.get("schema") == "spgel_source_run_lock_v1" and lock.get("status") == "LOCKED_BEFORE_SPGEL_FITS"
            and lock.get("config_sha256") == sha256(config_path)
            and lock.get("contract_sha256") == canonical_hash({k: v for k, v in lock.items() if k != "contract_sha256"})
            and lock.get("sources") == source_hashes(), "SOURCE_RUN_LOCK_OR_CODE_CHANGED")
    require(lock["pilot"]["logical_cells"] == 72 and lock["pilot"]["actual_saved_states"] == 192,
            "SOURCE_RUN_LOCK_MATRIX_CHANGED")
    qualification = read_json(Path(private) / "QUALIFICATION.private.json")
    require(qualification.get("status") == "PASS_REAL_SOURCE_IMAGE_ONLY_AND_PAIRED_GRADIENT_PATHS"
            and qualification.get("contract_sha256") == lock["contract_sha256"]
            and qualification.get("config_sha256") == sha256(config_path)
            and qualification.get("new_optimizer_fits") == qualification.get("optimizer_instances")
                == qualification.get("optimizer_updates") == 0, "ACTUAL_SOURCE_QUALIFICATION_REQUIRED")
    validate_recipe(config)
    return lock


def model_for(config, source, stats, arm, seed, device):
    capacity = match_image_mlp_capacity(source.up, source.down, source.gene_symbols,
        width=config["model"]["width"], hidden=config["model"]["hidden"])
    require(capacity["hidden"] == config["model"]["image_mlp_hidden"], "PRELOCK_IMAGE_MLP_CAPACITY_WIDTH_CHANGED")
    return SpatialProgramEncoder(arm, source.up, source.down, source.gene_symbols,
        width=config["model"]["width"], hidden=config["model"]["hidden"],
        image_mlp_hidden=capacity["hidden"], tissue_program_mean=stats.tissue_program_mean, seed=seed).to(device)


def fit_tissue_classifier(source, train, stats, config, seed):
    specification = config["tissue_only"]
    classifier = LogisticRegression(C=specification["C"], solver="lbfgs", fit_intercept=True,
        class_weight="balanced", max_iter=specification["maximum_iterations"],
        tol=specification["tolerance"], random_state=int(seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(source.features[train].astype(np.float64), source.tissue[train])
    require(not any(issubclass(x.category, ConvergenceWarning) for x in caught)
            and np.array_equal(classifier.classes_, np.arange(3)), "TISSUE_LOGISTIC_DID_NOT_CONVERGE")
    coefficients, intercept = classifier.coef_.astype(np.float64), classifier.intercept_.astype(np.float64)
    x, labels = source.features[train].astype(np.float64), source.tissue[train]
    probability = softmax(x @ coefficients.T + intercept, axis=1)
    residual = probability - np.eye(3)[labels]
    residual *= stats.class_weights[labels, None] / len(train)
    gradient = residual.T @ x + coefficients / (specification["C"] * len(train))
    maximum = max(float(np.max(np.abs(gradient))), float(np.max(np.abs(residual.sum(0)))))
    require(maximum <= specification["gradient_infinity_tolerance"], "TISSUE_LOGISTIC_KKT_TOLERANCE_FAILED")
    return dict(coef=coefficients, intercept=intercept, tissue_program_mean=stats.tissue_program_mean.copy(),
                iterations=int(classifier.n_iter_.max()), gradient_infinity_norm=maximum,
                optimized_parameters=coefficients.size + intercept.size, analytic_program_table_coefficients=12)


def collect_images(model, source, rows, *, allowed_rows, device="cpu", batch_size=512, classifier=None):
    rows = np.asarray(rows, np.int64)
    if classifier is not None:
        probability = softmax(source.features[rows].astype(np.float64) @ classifier["coef"].T + classifier["intercept"], axis=1)
        return dict(program_predictions=probability @ classifier["tissue_program_mean"], tissue_probs=probability), 0
    pieces, calls = {}, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            out = model.image_only_predict(source.image_batch(rows[start:start + batch_size], allowed_rows=allowed_rows, device=device))
            calls += 1
            for name, value in out.items():
                pieces.setdefault(name, []).append(value.detach().cpu().numpy())
    return {name: np.concatenate(values) for name, values in pieces.items()}, calls


def tissue_evaluation(probability, truth):
    if probability is None:
        return dict(status="NOT_APPLICABLE_NO_TISSUE_HEAD")
    truth = np.asarray(truth, int)
    prediction = np.asarray(probability).argmax(1)
    recall = [float(np.mean(prediction[truth == k] == k)) if (truth == k).any() else None for k in range(3)]
    return dict(status="COMPLETE", balanced_accuracy=float(np.mean(recall)) if all(x is not None for x in recall) else None,
                per_class_recall=recall, true_class_counts=np.bincount(truth, minlength=3).tolist())


def evaluate_images(outputs, source, rows, stats, *, include_groups):
    rows = np.asarray(rows, int)
    pz = np.asarray(outputs["program_predictions"], float)
    raw = stats.inverse_targets(pz)
    task_a = image_prediction_metrics(pz, source.targets[rows], source.target_mask[rows], stats.target_mean, stats.target_sd)
    task_c = localization_metrics(raw, source.targets[rows], source.target_mask[rows], stats.high_program_threshold)
    probability = outputs.get("predicted_tissue_probabilities", outputs.get("tissue_probs"))
    result = dict(task_A=task_a, task_C=task_c,
        tissue_metrics=dict(image_predicted_tissue=tissue_evaluation(probability, source.tissue[rows])))
    if include_groups:
        per = []
        for section in np.unique(source.section[rows]):
            selected = source.section[rows] == section
            per.append(dict(section_code=int(section), spots_count=int(selected.sum()), task_A=image_prediction_metrics(
                pz[selected], source.targets[rows][selected], source.target_mask[rows][selected], stats.target_mean, stats.target_sd)))
        result["section_metrics"] = dict(task="within_slide_program_prediction", sections=per)
        result["tissue_metrics"]["within_slide_tissue"] = stratified_within_slide(
            raw, source.targets[rows], source.target_mask[rows], source.section[rows], source.tissue[rows])
    return result


def _rank_summary(positive, full_scores, candidate_indices, query_indices):
    # Complete candidate library contains the true same-spot positive once.
    candidate_indices, query_indices = np.asarray(candidate_indices), np.asarray(query_indices)
    score = np.asarray(full_scores, float)
    require(score.shape == (len(query_indices), len(candidate_indices)) and np.isfinite(score).all(), "FULL_LIBRARY_SCORE_MATRIX")
    greater = (score > np.asarray(positive)[:, None]).sum(1)
    ties_including_positive = (score == np.asarray(positive)[:, None]).sum(1)
    require((ties_including_positive >= 1).all(), "RETRIEVAL_TRUE_PAIR_NOT_IN_CANDIDATE_LIBRARY")
    credit = lambda k: np.clip((k - greater) / ties_including_positive, 0., 1.)
    negative_count = len(candidate_indices) - 1
    auc = (negative_count - greater - .5 * (ties_including_positive - 1)) / max(negative_count, 1)
    return dict(Recall_at_1=float(credit(1).mean()),
        Recall_at_5=float(credit(5).mean()) if len(candidate_indices) >= 5 else None,
        median_rank=float(np.median(1 + greater + .5 * (ties_including_positive - 1))),
        pair_AUC=float(auc.mean()) if negative_count else None, queries_count=len(query_indices),
        candidates_count=len(candidate_indices), candidate_negative_count=negative_count,
        tie_rule="expected_uniform_credit_average_rank", query_indices_sha256=array_hash(query_indices.astype(np.int64)),
        candidate_indices_sha256=array_hash(candidate_indices.astype(np.int64)))


def _summary_across_sections(sections, direction):
    fields = ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC")
    result = {name: float(np.mean([s[direction][name] for s in sections]))
              if sections and all(s[direction][name] is not None for s in sections) else None for name in fields}
    result.update(queries_count=sum(s[direction]["queries_count"] for s in sections),
                  aggregation="equal_slide_macro_not_independent_spots", sections_count=len(sections))
    return result


def evaluate_retrieval(model, outputs, source, rows, stats, *, native, device="cpu", anchor_cap=128,
                       candidate_chunk=32):
    """All same-slide candidates; fixed row-equidistant queries, no heldout QC filters."""
    if native and (model is None or not model.has_genes):
        return dict(status="NOT_APPLICABLE_NO_TRAINED_GENE_ENCODER"), {}, 0
    rows = np.asarray(rows, np.int64)
    require(source.target_mask[rows].all(), "COMMON_RETRIEVAL_REQUIRES_QUALIFIED_FOUR_PROGRAM_MEASUREMENTS")
    n = len(rows)
    visual = np.asarray(outputs["program_predictions"], float)
    molecular = stats.targets_z(source, rows)
    visual = visual / np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    molecular = molecular / np.maximum(np.linalg.norm(molecular, axis=1, keepdims=True), 1e-12)
    gene_batch = stats.genes_z(source, rows, device) if native else None
    tensor_out = {k: torch.as_tensor(v, device=device) for k, v in outputs.items()} if native else None
    per, arrays, calls = [], {}, 0
    with torch.no_grad():
        # This cache is single-modality and current-weight: it contains no
        # positive-pair condition and cannot reveal which candidate is correct.
        gene_base = model.gene_encoder(gene_batch) if native else None
        for section in np.unique(source.section[rows]):
            candidates = np.flatnonzero(source.section[rows] == section)
            count = min(anchor_cap, len(candidates))
            anchors = candidates[np.floor((np.arange(count) + .5) * len(candidates) / count).astype(int)]
            require(len(np.unique(anchors)) == count, "ROW_EQUIDISTANT_QUERY_DUPLICATION")
            matrices = {}
            for reverse in (False, True):
                if not native:
                    matrices[reverse] = molecular[anchors] @ visual[candidates].T if reverse else visual[anchors] @ molecular[candidates].T
                    continue
                matrix = np.empty((count, len(candidates)), np.float64)
                for start in range(0, count, 32):
                    aa = anchors[start:start + 32]
                    for begin in range(0, len(candidates), candidate_chunk):
                        cc = candidates[begin:begin + candidate_chunk]
                        ia = np.broadcast_to(aa[:, None], (len(aa), len(cc))).copy()
                        ib = np.broadcast_to(cc[None, :], ia.shape).copy()
                        ir, gr = (ib, ia) if reverse else (ia, ib)
                        pair = model.paired_scores(tensor_out, gene_batch,
                            torch.as_tensor(ir, device=device), torch.as_tensor(gr, device=device), paired_mode=True,
                            _gene_base=gene_base)
                        require(pair["valid"].all(), "NATIVE_RETRIEVAL_MISSING_PAIR_PROGRAM_SUPPORT")
                        matrix[start:start + len(aa), begin:begin + len(cc)] = pair["scores"].cpu().numpy()
                        calls += 1
                matrices[reverse] = matrix
            position = {int(v): k for k, v in enumerate(candidates)}
            positive_column = np.array([position[int(v)] for v in anchors])
            item = dict(section_code=int(section), anchors_count=count, candidates_count=len(candidates),
                anchor_selection="floor((arange(min(128,N))+.5)*N/min(128,N))_in_registered_row_order",
                anchor_indices_sha256=array_hash(rows[anchors]), candidate_indices_sha256=array_hash(rows[candidates]))
            for reverse, name in ((False, "image_to_gene"), (True, "gene_to_image")):
                matrix = matrices[reverse]
                positive = matrix[np.arange(count), positive_column]
                item[name] = _rank_summary(positive, matrix, rows[candidates], rows[anchors])
                arrays[f"section{int(section)}_{name}"] = matrix
            per.append(item)
    return dict(status="COMPLETE", candidate_scope="entire_same_heldout_slide_no_tissue_depth_filter",
        query_anchor_cap_per_slide=anchor_cap, query_selection_uses_outcomes=False,
        image_to_gene=_summary_across_sections(per, "image_to_gene"),
        gene_to_image=_summary_across_sections(per, "gene_to_image"), per_section=per,
        native_paired_mode=native, common_four_program_score_baseline=not native,
        true_local_pairing_is_not_biological_negative_truth=True), arrays, calls


def _snapshot(model, source, train, held, stats, classifier, *, device, epoch, stage, checkpoint_label,
              epoch_losses, include_retrieval):
    train_out, train_calls = collect_images(model, source, train, allowed_rows=train, device=device, classifier=classifier)
    train_metrics = evaluate_images(train_out, source, train, stats, include_groups=False)
    arrays = dict(train_prediction=train_out["program_predictions"])
    held_metrics, native_calls = dict(status="NOT_APPLICABLE_FULL_SOURCE_FIT"), 0
    held_calls = 0
    if len(held):
        held_out, held_calls = collect_images(model, source, held, allowed_rows=held, device=device, classifier=classifier)
        held_metrics = evaluate_images(held_out, source, held, stats, include_groups=True)
        arrays["heldout_prediction"] = held_out["program_predictions"]
        if include_retrieval:
            for native, name in ((False, "task_B_common"), (True, "task_B_native")):
                metrics, private_arrays, calls = evaluate_retrieval(model, held_out, source, held, stats,
                    native=native, device=device)
                held_metrics[name] = metrics
                arrays.update({name + "_" + k: v for k, v in private_arrays.items()})
                native_calls += calls
        else:
            held_metrics.update(task_B_common=dict(status="NOT_EVALUATED_FOR_TASKB"),
                                task_B_native=dict(status="NOT_EVALUATED_FOR_TASKB"))
    return dict(label=checkpoint_label, epoch=epoch, train=train_metrics, heldout=held_metrics,
        losses=epoch_losses, actual_image_forward_calls=train_calls + held_calls,
        actual_paired_score_calls=native_calls, true_RNA_in_image_prediction=False), arrays


def unit_name(arm, donor, seed, stage="pilot"):
    return f"{arm}_full_seed{seed}" if stage == "s4" else f"{arm}_d{donor}_seed{seed}"


def _s4_gate(private, lock):
    audit = read_json(Path(private) / "pilot" / "REPLAY_AUDIT.json")
    require(audit.get("status") == "PASS_COMPLETE_SPGEL_PILOT_72_FITS_192_REPLAYS"
            and audit.get("contract_sha256") == lock["contract_sha256"]
            and audit.get("passed_units") == 72 and audit.get("passed_states") == 192
            and audit.get("failed_units") == 0, "S4_REQUIRES_COMPLETE_ACTUALLY_REPLAYED_PILOT")


def fit_cell(source, config, config_path, private, *, arm, donor, seed, device="cpu", stage="pilot"):
    configure_runtime()
    private = private_directory(private)
    lock = validate_lock(private, config_path, config)
    require(arm in ARMS and seed in config["seeds"], "UNPLANNED_SOURCE_CELL")
    if stage == "s4":
        require(arm in ("S_A1", "S_A5") and donor == -1 and seed == 31, "UNPLANNED_S4_CELL")
        _s4_gate(private, lock)
        train, held = np.arange(len(source.features)), np.empty(0, np.int64)
    else:
        require(stage == "pilot" and donor in range(6), "UNPLANNED_PILOT_DONOR")
        train, held = source.split(donor)
    name = unit_name(arm, donor, seed, stage)
    directory = Path(private) / stage / "cells" / name
    require(not directory.exists(), "SOURCE_EXISTING_CELL_NO_AUTOMATIC_RETRY_OR_OVERWRITE")
    directory.mkdir(parents=True, mode=0o777 if os.name == "nt" else 0o700)
    start = time.monotonic()
    stats = SourceStatistics.fit(source, train, heldout_donors=([donor] if donor >= 0 else ()))
    stats_audit = stats.audit()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    snapshots, saved_arrays, updates, negative_summaries = [], {}, 0, []
    try:
        if arm == "S_A0":
            classifier = fit_tissue_classifier(source, train, stats, config, seed)
            state = dict(classifier=classifier, stats=asdict(stats), arm=arm, heldout_donor=donor, seed=seed,
                stage=stage, config_sha256=sha256(config_path), contract_sha256=lock["contract_sha256"],
                train_rows_sha256=array_hash(train), heldout_rows_sha256=array_hash(held), optimizer_updates=0)
            checkpoint = directory / "checkpoint_solver_final.pt"
            torch_write(checkpoint, state)
            snapshot, arrays = _snapshot(None, source, train, held, stats, classifier, device="cpu", epoch=None,
                stage=stage, checkpoint_label="solver_final", epoch_losses=dict(
                    solver_iterations=classifier["iterations"], solver_gradient_infinity_norm=classifier["gradient_infinity_norm"]),
                include_retrieval=True)
            snapshot["checkpoint_sha256"] = sha256(checkpoint)
            snapshots.append(snapshot); saved_arrays["solver_final"] = arrays
            parameters = dict(optimized=classifier["optimized_parameters"], registered=classifier["optimized_parameters"],
                              analytic_program_table_coefficients=12)
            model = None
        else:
            model = model_for(config, source, stats, arm, seed, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
            parameters = dict(optimized=model.descriptor()["parameters"], registered=model.descriptor()["registered_parameters"],
                              analytic_program_table_coefficients=0)
            weights = torch.as_tensor(stats.class_weights, dtype=torch.float32, device=device)
            for epoch in range(1, 41):
                model.train()
                order = np.random.default_rng(np.random.SeedSequence([seed, epoch, donor + 1, 71001])).permutation(train)
                epoch_means = {k: 0. for k in ("total", "prediction_Huber", "tissue_CE", "ITC")}
                negative_totals = dict(epoch=epoch, training_only=True, valid_negative_count=0,
                    same_tissue_depth_count=0, same_tissue_count=0, cross_slide_count=0,
                    anchors_count=0, no_negative_anchor_count=0, active_ITC_image_anchors=0,
                    active_ITC_gene_anchors=0)
                negative_hash = hashlib.sha256()
                for batch_id, begin in enumerate(range(0, len(order), 512)):
                    rows = order[begin:begin + 512]
                    image = source.image_batch(rows, allowed_rows=train, device=device)
                    target = torch.as_tensor(stats.targets_z(source, rows), dtype=torch.float32, device=device)
                    mask = torch.as_tensor(source.target_mask[rows], device=device)
                    labels = torch.as_tensor(source.tissue[rows], dtype=torch.long, device=device)
                    genes = stats.genes_z(source, rows, device) if model.has_genes else None
                    negative = None
                    if model.has_genes:
                        if len(rows) == 1:
                            neg = np.full((1, 8), -1, np.int64)
                            na = dict(valid_negative_count=0, same_tissue_depth_count=0,
                                      same_tissue_count=0, cross_slide_count=0, anchors_count=1)
                        else:
                            neg, na = hard_negative_rows(section=source.section[rows], tissue=source.tissue[rows],
                                library=source.library[rows], xy_um=source.xy_um[rows], rows=np.arange(len(rows)),
                                train_depth_quartiles=stats.log_depth_quartiles, count=8,
                                seed=int(np.random.SeedSequence([seed, epoch, batch_id, donor + 1]).generate_state(1)[0]))
                        for key in ("valid_negative_count", "same_tissue_depth_count", "same_tissue_count",
                                    "cross_slide_count", "anchors_count"):
                            negative_totals[key] += na[key]
                        negative_totals["no_negative_anchor_count"] += int(np.sum(~(neg >= 0).any(1)))
                        negative_hash.update(np.ascontiguousarray(rows, np.int64).tobytes())
                        negative_hash.update(np.ascontiguousarray(neg, np.int64).tobytes())
                        negative = torch.as_tensor(neg, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model.image_only_predict(image)
                    loss, audit = source_objective(model, outputs, target, mask, labels, weights,
                        genes=genes, negative_index=negative)
                    if model.has_genes:
                        negative_totals["active_ITC_image_anchors"] += audit["contrastive"]["active_image_anchors"]
                        negative_totals["active_ITC_gene_anchors"] += audit["contrastive"]["active_gene_anchors"]
                    require(torch.isfinite(loss), "NONFINITE_SOURCE_TRAINING_LOSS")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    require(torch.isfinite(norm), "NONFINITE_SOURCE_TRAINING_GRADIENT")
                    optimizer.step(); updates += 1
                    scale = len(rows) / len(train)
                    epoch_means["total"] += float(loss.detach()) * scale
                    for key in ("prediction_Huber", "tissue_CE", "ITC"):
                        epoch_means[key] += float(audit[key].detach()) * scale
                require(updates == epoch * math.ceil(len(train) / 512), "LAST_MINIBATCH_WAS_DROPPED_OR_UPDATE_BUDGET_CHANGED")
                negative_totals.update(mapping_sha256=negative_hash.hexdigest(),
                    status="RECORDED_ACTUAL_TRAIN_NEGATIVES" if model.has_genes else "NOT_USED_NO_ITC")
                negative_summaries.append(negative_totals)
                write_json(directory / "PROGRESS.json", dict(status="RUNNING", arm=arm, heldout_donor=donor,
                    seed=seed, stage=stage, epoch=epoch, optimizer_updates=updates,
                    elapsed_seconds=time.monotonic() - start), progress=True)
                if epoch in (10, 20, 40):
                    label = f"epoch{epoch}"
                    checkpoint = directory / f"checkpoint_{label}.pt"
                    state = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        optimizer_state=optimizer.state_dict(), descriptor=model.descriptor(), stats=asdict(stats),
                        arm=arm, heldout_donor=donor, seed=seed, stage=stage, epoch=epoch,
                        optimizer_updates=updates, config_sha256=sha256(config_path), contract_sha256=lock["contract_sha256"],
                        train_rows_sha256=array_hash(train), heldout_rows_sha256=array_hash(held))
                    torch_write(checkpoint, state)
                    snapshot, arrays = _snapshot(model, source, train, held, stats, None, device=device, epoch=epoch,
                        stage=stage, checkpoint_label=label,
                        epoch_losses={"epoch_minibatch_preupdate_mean_" + k: v for k, v in epoch_means.items()},
                        include_retrieval=epoch == 40)
                    snapshot["checkpoint_sha256"] = sha256(checkpoint)
                    snapshots.append(snapshot); saved_arrays[label] = arrays
        artifact = dict(schema="spgel_source_private_model_artifact_v1", snapshots=saved_arrays,
            final_checkpoint=snapshots[-1]["label"], stats=asdict(stats), source_cache_manifest_sha256=config["source_binding"]["cache_manifest_sha256"])
        torch_write(directory / "model.pt", artifact)
        done = dict(schema="spgel_source_cell_v1", status="COMPLETE_SPGEL_SOURCE_FIT", arm=arm,
            heldout_donor=donor, seed=seed, stage=stage, fit_kind="convex" if arm == "S_A0" else "neural",
            epochs=None if arm == "S_A0" else 40, optimizer_updates=updates,
            convex_optimizer_fits=int(arm == "S_A0"), neural_optimizer_fits=int(arm != "S_A0"),
            train_rows=len(train), heldout_rows=len(held), parameters=parameters,
            elapsed_seconds=time.monotonic() - start,
            peak_GPU_bytes=int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0,
            config_sha256=sha256(config_path), contract_sha256=lock["contract_sha256"],
            artifact_sha256=sha256(directory / "model.pt"), snapshots=snapshots,
            preprocessing=stats_audit, image_prediction_input_is_image_and_geometry_only=True,
            original_ST_test_read=False, DSS_or_PFI_read=False, no_model_selection_from_heldout=True,
            source_cache_manifest_sha256=config["source_binding"]["cache_manifest_sha256"])
        done["training_negative_audit"] = dict(scope="current_training_minibatches_only",
            heldout_depth_tissue_not_candidate_filters=True, epochs=negative_summaries)
        write_json(directory / "COMPLETE.json", done)
        return done
    except BaseException as exc:
        write_json(directory / "ERROR_RECEIPT.json", dict(status="FAILED_SOURCE_FIT_NO_AUTOMATIC_RETRY", arm=arm,
            heldout_donor=donor, seed=seed, stage=stage, failure_type=type(exc).__name__,
            optimizer_updates_before_failure=updates, elapsed_seconds=time.monotonic() - start,
            contract_sha256=lock["contract_sha256"]))
        (directory / "ERROR_PRIVATE.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _metric_difference(a, b):
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), "REPLAY_METADATA_KEY_SET_CHANGED")
        return max((_metric_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), "REPLAY_METADATA_LIST_CHANGED")
        return max((_metric_difference(x, y) for x, y in zip(a, b)), default=0.)
    if type(a) in (int, float) and type(b) in (int, float):
        require(np.isfinite(a) and np.isfinite(b), "REPLAY_NONFINITE_METRIC")
        return abs(float(a) - float(b))
    require(a == b, "REPLAY_METADATA_VALUE_CHANGED")
    return 0.


def replay_cell(source, config, config_path, private, *, arm, donor, seed, stage="pilot", device="cpu"):
    configure_runtime(); lock = validate_lock(private, config_path, config)
    directory = Path(private) / stage / "cells" / unit_name(arm, donor, seed, stage)
    complete_path = directory / "COMPLETE.json"
    done = read_json(complete_path)
    require(not (directory / "ERROR_RECEIPT.json").exists() and done.get("status") == "COMPLETE_SPGEL_SOURCE_FIT"
            and done["contract_sha256"] == lock["contract_sha256"] and done["config_sha256"] == sha256(config_path)
            and done["artifact_sha256"] == sha256(directory / "model.pt"), "SOURCE_REPLAY_COMPLETION_OR_ASSET_CHANGED")
    train, held = source.split(donor) if donor >= 0 else (np.arange(len(source.features)), np.empty(0, np.int64))
    stats = SourceStatistics.fit(source, train, heldout_donors=([donor] if donor >= 0 else ()))
    artifact = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    expected_labels = ["solver_final"] if arm == "S_A0" else ["epoch10", "epoch20", "epoch40"]
    require([s["label"] for s in done["snapshots"]] == expected_labels
            and set(artifact["snapshots"]) == set(expected_labels), "SOURCE_REPLAY_SNAPSHOT_INVENTORY")
    model = None if arm == "S_A0" else model_for(config, source, stats, arm, seed, device)
    rows = []
    for stored in done["snapshots"]:
        label = stored["label"]
        path = directory / f"checkpoint_{label}.pt"
        require(sha256(path) == stored["checkpoint_sha256"], "SOURCE_CHECKPOINT_BYTES_CHANGED")
        state = torch.load(path, map_location="cpu", weights_only=False)
        require(state["contract_sha256"] == lock["contract_sha256"]
                and state["config_sha256"] == sha256(config_path)
                and state["train_rows_sha256"] == array_hash(train) and state["heldout_rows_sha256"] == array_hash(held),
                "SOURCE_CHECKPOINT_LINEAGE_CHANGED")
        for key, value in asdict(stats).items():
            actual = state["stats"][key]
            require(np.array_equal(actual, value) if isinstance(value, np.ndarray) else actual == value,
                    "SOURCE_TRAIN_ONLY_SCALER_REPLAY_CHANGED")
        classifier = state["classifier"] if arm == "S_A0" else None
        if model is not None:
            model.load_state_dict(state["state_dict"], strict=True)
            require(state["optimizer_updates"] == state["epoch"] * math.ceil(len(train) / 512),
                    "SOURCE_SAVED_UPDATE_COUNTER_CHANGED")
        replay, arrays = _snapshot(model, source, train, held, stats, classifier, device=device,
            epoch=stored["epoch"], stage=stage, checkpoint_label=label, epoch_losses=stored["losses"],
            include_retrieval=label in ("epoch40", "solver_final"))
        reference = artifact["snapshots"][label]
        require(set(arrays) == set(reference), "SOURCE_REPLAY_PREDICTION_ARRAY_INVENTORY")
        error = max((float(np.max(np.abs(arrays[k] - reference[k]), initial=0.)) for k in arrays), default=0.)
        require(error <= 2e-6, "SOURCE_REPLAY_PREDICTION_TOLERANCE_FAILED")
        metric_error = max(_metric_difference(replay[split], stored[split]) for split in ("train", "heldout"))
        require(metric_error <= 1e-8, "SOURCE_REPLAY_METRIC_TOLERANCE_FAILED")
        rows.append(dict(label=label, checkpoint_sha256=sha256(path), actual_state_forward_replayed=True,
                         maximum_prediction_error=error, maximum_metric_error=metric_error))
    return dict(arm=arm, heldout_donor=donor, seed=seed, stage=stage, status="PASS",
        completion_sha256=sha256(complete_path), artifact_sha256=sha256(directory / "model.pt"),
        snapshots=rows, actual_saved_state_loaded=True, new_optimizer_fits=0, new_optimizer_updates=0)


def qualify_source(source, config, config_path, private, *, device="cpu"):
    """Real source forwards/backwards, zero optimizer objects or saved weights."""
    configure_runtime(); validate_recipe(config); private = private_directory(private)
    require(not (private / "SOURCE_RUN_LOCK.json").exists() and not (private / "QUALIFICATION.private.json").exists(),
            "EXISTING_SOURCE_QUALIFICATION_OR_LOCK_PRESERVED")
    sources = source_hashes()
    budget = budget_for_source(source)
    require(budget["neural_optimizer_updates"] == 52000 and budget["s4_neural_optimizer_updates"] == 2080,
            "QUALIFIED_SOURCE_ROWS_CHANGED_UPDATE_BUDGET")
    train, held = source.split(0)
    stats = SourceStatistics.fit(source, train, heldout_donors=[0])
    # One first row per available slide is metadata-only and does not pick good outcomes.
    probe = np.array([train[np.flatnonzero(source.section[train] == k)[0]]
                      for k in np.unique(source.section[train])], np.int64)
    image = source.image_batch(probe, allowed_rows=train, device=device)
    genes = stats.genes_z(source, probe, device)
    target = torch.as_tensor(stats.targets_z(source, probe), dtype=torch.float32, device=device)
    observed = torch.as_tensor(source.target_mask[probe], device=device)
    labels = torch.as_tensor(source.tissue[probe], dtype=torch.long, device=device)
    weights = torch.as_tensor(stats.class_weights, dtype=torch.float32, device=device)
    neg, negative_audit = hard_negative_rows(section=source.section[probe], tissue=source.tissue[probe],
        library=source.library[probe], xy_um=source.xy_um[probe], rows=np.arange(len(probe)),
        train_depth_quartiles=stats.log_depth_quartiles, count=8, seed=31)
    negative = torch.as_tensor(neg, device=device)
    records = []
    for arm in NEURAL_ARMS:
        model = model_for(config, source, stats, arm, 31, device)
        model.eval()
        state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        first = model.image_only_predict(image)
        repeated = model.image_only_predict(image)
        repeat_error = float((first["program_predictions"] - repeated["program_predictions"]).abs().max().detach())
        require(repeat_error == 0., "REAL_SOURCE_REPEATED_FORWARD_NOT_DETERMINISTIC")
        model.zero_grad(set_to_none=True)
        pure = first["program_predictions"].square().mean()
        pure.backward()
        require(model.backbone.project[0].weight.grad is not None
                and torch.isfinite(model.backbone.project[0].weight.grad).all(), "IMAGE_PREDICTOR_GRADIENT_MISSING")
        if model.has_genes:
            require(all(p.grad is None for p in model.gene_encoder.parameters()), "GENES_ENTERED_IMAGE_ONLY_PREDICTION")
        model.zero_grad(set_to_none=True)
        outputs = model.image_only_predict(image)
        loss, _ = source_objective(model, outputs, target, observed, labels, weights,
                                   genes=genes if model.has_genes else None,
                                   negative_index=negative if model.has_genes else None)
        loss.backward()
        dual = {}
        if model.bidirectional:
            for name in ("gene_to_image", "image_to_gene"):
                gradient = getattr(model, name).weight.grad
                require(gradient is not None and torch.isfinite(gradient).all()
                        and torch.count_nonzero(gradient) > 0, "REAL_SOURCE_DUAL_PATH_GRADIENT_MISSING")
                dual[name] = float(gradient.norm().detach())
        require(all(torch.equal(state_before[k], v.detach().cpu()) for k, v in model.state_dict().items()),
                "ZERO_OPTIMIZER_QUALIFICATION_MUTATED_MODEL_STATE")
        records.append(dict(arm=arm, repeated_image_forward_max_error=repeat_error,
            image_prediction_gene_gradient_present=False, dual_query_gradient_norms=dual,
            parameters=model.descriptor()["parameters"], actual_image_forward_calls=3,
            actual_backward_calls=2, optimizer_instances=0, optimizer_updates=0))
        del model
    require(source_hashes() == sources, "SOURCE_FILES_CHANGED_DURING_QUALIFICATION")
    lock = dict(schema="spgel_source_run_lock_v1", status="LOCKED_BEFORE_SPGEL_FITS",
        config_sha256=sha256(config_path), sources=sources, source_binding=config["source_binding"],
        program_order=list(PROGRAMS), primary_candidate="S_A5", model_or_epoch_selection_from_heldout=False,
        pilot=dict(logical_cells=72, neural_fits=60, convex_fits=12, actual_saved_states=192,
                   optimizer_updates=52000),
        s4=dict(neural_fits=2, actual_saved_states=6, optimizer_updates=2080),
        measured_budget=budget, R8_or_DSS_used=False, official_HVG_or_new_source_in_this_lock=False,
        retrieval_query_sampler="per_slide_min_128_row_equidistant_without_labels_full_slide_candidate_library")
    lock["contract_sha256"] = canonical_hash(lock)
    qualification = dict(schema="spgel_source_qualification_v1",
        status="PASS_REAL_SOURCE_IMAGE_ONLY_AND_PAIRED_GRADIENT_PATHS", contract_sha256=lock["contract_sha256"],
        config_sha256=sha256(config_path), records=records, source_cache_manifest_sha256=config["source_binding"]["cache_manifest_sha256"],
        full_source_data_replay_passed=True, new_optimizer_fits=0, optimizer_instances=0, optimizer_updates=0,
        source_probe_rows_count=len(probe), source_probe_selection="first_current_train_row_of_each_slide_no_outcome_selection",
        probe_rows_sha256=array_hash(probe), negative_audit=negative_audit,
        saved_qualification_weights=False, source_statistics_are_not_optimizer_fits=True,
        actual_backward_calls=10, DSS_or_PFI_read=False, original_ST_test_read=False)
    write_json(private / "SOURCE_RUN_LOCK.json", lock)
    write_json(private / "QUALIFICATION.private.json", qualification)
    return qualification


def replay_all(source, config, config_path, private, *, stage="pilot", device="cpu"):
    lock = validate_lock(private, config_path, config)
    units = [(a, d, s) for a in ARMS for d in range(6) for s in (31, 47)] if stage == "pilot" else [(a, -1, 31) for a in ("S_A5", "S_A1")]
    expected_states = 192 if stage == "pilot" else 6
    base = Path(private) / stage
    require(not (base / "REPLAY_AUDIT.json").exists(), "EXISTING_SOURCE_REPLAY_PRESERVED_NO_OVERWRITE")
    found = {p.parent.name for p in (base / "cells").glob("*/COMPLETE.json")}
    require(found == {unit_name(a, d, s, stage) for a, d, s in units}, "EXACT_SOURCE_MATRIX_REQUIRED_FOR_REPLAY")
    terminal = base / "REPLAY_PRIVATE"
    terminal.mkdir(exist_ok=True)
    results = []
    for arm, donor, seed in units:
        name = unit_name(arm, donor, seed, stage)
        require(not (terminal / (name + ".json")).exists(), "EXISTING_REPLAY_UNIT_PRESERVED")
        try:
            result = replay_cell(source, config, config_path, private, arm=arm, donor=donor, seed=seed, stage=stage, device=device)
        except BaseException as exc:
            result = dict(arm=arm, heldout_donor=donor, seed=seed, stage=stage, status="FAIL", failure_type=type(exc).__name__,
                          new_optimizer_fits=0, new_optimizer_updates=0, actual_saved_state_loaded=False)
            (terminal / (name + ".log")).write_text(traceback.format_exc(), encoding="utf-8")
        write_json(terminal / (name + ".json"), result)
        results.append(result)
        write_json(base / "REPLAY_PROGRESS.json", dict(expected_units=len(units), units=results,
            passed_units=sum(r["status"] == "PASS" for r in results), new_optimizer_fits=0), progress=True)
        print(json.dumps(dict(unit=name, status=result["status"])), flush=True)
    passed = sum(r["status"] == "PASS" for r in results)
    states = sum(len(r.get("snapshots", [])) for r in results if r["status"] == "PASS")
    complete = passed == len(units) and states == expected_states
    success = "PASS_COMPLETE_SPGEL_PILOT_72_FITS_192_REPLAYS" if stage == "pilot" else "PASS_COMPLETE_SPGEL_S4_2_FITS_6_REPLAYS"
    audit = dict(schema="spgel_source_replay_v1", status=success if complete else "FAILED_SOURCE_REPLAY",
        expected_units=len(units), passed_units=passed, failed_units=len(units) - passed,
        expected_states=expected_states, passed_states=states, units=results,
        contract_sha256=lock["contract_sha256"], config_sha256=sha256(config_path),
        new_optimizer_fits=0, new_optimizer_updates=0, stage=stage)
    write_json(base / "REPLAY_AUDIT.json", audit)
    require(complete, "SOURCE_ACTUAL_REPLAY_FAILED_NO_MODEL_RETRY")
    return audit

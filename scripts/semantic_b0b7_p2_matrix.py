"""Finite P2 matrix orchestration; outer scoring requires frozen inner receipts.

Run on the authorized private server. Commands write only new private outputs;
publication is a separate allowlisted operation. No bootstrap is implemented.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_prepare import legacy_data_module, read_json, file_sha
from scripts.semantic_b0b7_p1r import (run_cell, canonical_hash, tensor_hash,
    load_checkpoint_model, extract_frozen_features)
from src.semantic_b0b7.p1r_data import build_partition_inputs, molecular_memberships
from src.semantic_b0b7.p1r_models import frozen_readout_operator
from src.semantic_b0b7.p2_data import read_dss_only, attach_legacy_union, nested_split_plan
from src.semantic_b0b7.p2_anchor import fit_r4_anchor
from src.semantic_b0b7.p2_survival import (frozen_feature_views, fit_risk_cell,
    replay_risk_cell, select_lambda, evaluate_safe, array_sha)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def write_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        pickle.dump(value, f, protocol=5)


def shared_preprocessing(private, split, value):
    """Atomic content-addressed reuse across same-partition seeds/families."""
    encoded = pickle.dumps(value, protocol=5)
    digest = hashlib.sha256(encoded).hexdigest()
    folder = Path(private) / "preprocessing"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (split + "." + digest + ".private.pkl")
    with tempfile.NamedTemporaryFile(dir=folder, delete=False, prefix=".preprocessing-") as f:
        f.write(encoded)
        temporary = Path(f.name)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        temporary.unlink()
    if file_sha(target) != digest:
        raise ValueError("SHARED_PREPROCESSING_BYTES_CHANGED")
    return {"file": target.name, "sha256": digest}


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def verified_artifact(path):
    path = Path(path)
    receipt = read_json(path.with_suffix(".json"))
    if file_sha(path) != receipt["artifact_sha256"]:
        raise ValueError("PRIVATE_ARTIFACT_HASH_CHANGED")
    return load_pickle(path), receipt


def verify_lock(a):
    if file_sha(a.lock) != a.expected_lock_sha256:
        raise ValueError("P2_MASTER_LOCK_HASH_CHANGED")
    lock = read_json(a.lock)
    if lock["status"] != "LOCKED_BEFORE_OUTER_EVALUATION":
        raise ValueError("P2_NOT_LOCKED_BEFORE_OUTER")
    for rel, expected in lock["frozen_sources"].items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or file_sha(path) != expected:
            raise ValueError("P2_SOURCE_CHANGED:" + rel)
    assets = read_json(a.assets)
    for key, binding in (("gmt", "GMT_sha256"), ("legacy_groups", "legacy_groups_sha256")):
        if file_sha(assets[key]) != lock["assets"][binding]:
            raise ValueError("P2_EXTERNAL_ANNOTATION_CHANGED:" + key)
    if file_sha(Path(a.legacy_root) / "src/bgb_en/data.py") != lock["assets"]["legacy_data_source_sha256"]:
        raise ValueError("P2_LEGACY_COVARIATE_READER_CHANGED")
    if file_sha(Path(assets["patch_grid"]) / "full_mean.npy") != lock["B0_R4_EXACT"]["full_CONCH_mean_sha256"]:
        raise ValueError("P2_R4_ORIGINAL_FULL_CONCH_MEAN_CHANGED")
    return lock


def raw_inputs(a, *, endpoints=False):
    assets = read_json(a.assets)
    legacy, _ = legacy_data_module(a.legacy_root)
    raw = legacy.load_raw(assets, include_endpoints=False)
    membership = molecular_memberships(raw, assets["gmt"])
    if endpoints:
        outcome, endpoint_audit = read_dss_only(assets["endpoints"], raw["patient_ids"])
        raw, union_audit = attach_legacy_union(raw, assets["legacy_groups"])
        raw.update(outcome)
        raw["audit"] = {**raw["audit"], **endpoint_audit, **union_audit, "endpoint_read": True}
        # Original R4 driver overrides this GLOBAL-only feature source. LOCAL
        # remains the byte-verified visual_sources LOCAL (old GRID selection).
        mean_path = Path(assets["patch_grid"]) / "full_mean.npy"
        raw["conch_mean"] = np.asarray(np.load(mean_path, mmap_mode="r"), np.float64)
        raw["audit"]["R4_full_CONCH_mean_sha256"] = file_sha(mean_path)
    return raw, membership, assets


def split_entry(a):
    plan = read_json(Path(a.private) / "SPLITS.private.json")
    if plan["p2_lock_sha256"] != a.expected_lock_sha256:
        raise ValueError("NESTED_SPLIT_MASTER_LOCK_CHANGED")
    return plan["partitions"][a.split]


def training_input_binding(data, frozen_sha):
    tr = data["train_indices"]
    hashes = {section + "." + k: tensor_hash(v[tr])
              for section in ("image", "omics") for k, v in data[section].items() if v is not None}
    for k in ("target", "target_mask"):
        hashes[k] = tensor_hash(data[k][tr])
    for k, value in data["metadata"].items():
        hashes["metadata." + k] = tensor_hash(np.asarray(value)[tr.numpy()])
    return canonical_hash({"training_only_tensors": hashes,
        "fit_rows": tensor_hash(data["global_rows"][tr]), "frozen_parent": frozen_sha,
        "preprocessing": data["provenance"]["molecular_fit_hash"]})


def build_inputs(a, lock):
    raw, memberships, _ = raw_inputs(a, endpoints=False)
    if file_sha(a.frozen_images) != lock["assets"]["frozen_images_sha256"]:
        raise ValueError("P2_FROZEN_FM_CACHE_CHANGED")
    frozen = torch.load(a.frozen_images, map_location="cpu", weights_only=False)
    frozen["frozen_input_sha256"] = lock["assets"]["frozen_images_sha256"]
    split = split_entry(a)
    data, preprocessing = build_partition_inputs(raw, frozen, split["fit"], split["evaluation"], memberships,
                                                 split_name=a.split)
    del frozen, raw
    sha = training_input_binding(data, lock["assets"]["frozen_images_sha256"])
    return data, preprocessing, split, sha


def phase_path(a):
    return Path(a.private) / "phase_a" / a.split / (a.family + "_seed" + str(a.seed))


def inner_phase_choices(a, outer, family, seed):
    selected, bindings = [], []
    for inner in range(3):
        path = Path(a.private) / "phase_a" / f'outer{outer}_inner{inner}' / (family + "_seed" + str(seed)) / "cell.private.json"
        row = read_json(path)
        bind = read_json(path.parent / "MATRIX_BINDING.private.json")
        if (bind["p2_lock_sha256"] != a.expected_lock_sha256 or bind["cell_receipt_sha256"] != file_sha(path)
                or bind["split"] != f'outer{outer}_inner{inner}' or bind["family"] != family or bind["seed"] != seed
                or file_sha(path.parent / row["source_state_file"]) != row["selected_checkpoint_sha256"]):
            raise ValueError("INNER_PRETRAIN_SELECTION_RECEIPT_CHANGED")
        selected.append(int(row["selected_epoch"]))
        bindings.append({"inner": inner, "receipt_sha256": file_sha(path),
                         "selected_state_sha256": row["selected_checkpoint_sha256"]})
    return {"epoch": int(sorted(selected)[1]), "inner_selected_epochs": selected,
            "selection_rule": "median_of_three_outcome_free_inner_selected_checkpoints", "bindings": bindings}


def inner_risk_choice(a, outer, arm, seed):
    candidates, bindings = {}, []
    for lam in (.1, 1., 10.):
        candidates[lam] = []
        for inner in range(3):
            path = Path(a.private) / "risk" / f'outer{outer}_inner{inner}' / (arm + "_seed" + str(seed)) / ("lambda" + str(lam) + ".private.json")
            prior = read_json(path)
            bind = prior["source_and_scope_bindings"]
            if (bind["lock_sha256"] != a.expected_lock_sha256 or bind["arm"] != arm or bind["seed"] != seed
                    or bind["split"] != f'outer{outer}_inner{inner}'
                    or not prior["saved_linear_head_replay"]["fold_metric_components_equal"]
                    or file_sha(path.with_suffix(".pkl")) != prior["artifact_sha256"]):
                raise ValueError("INNER_RISK_SELECTION_RECEIPT_CHANGED")
            candidates[lam].append(prior["evaluation_metrics"])
            bindings.append({"inner": inner, "lambda": lam, "receipt_sha256": file_sha(path),
                             "artifact_sha256": prior["artifact_sha256"]})
    return {**select_lambda(candidates), "bindings": bindings}


def freeze_outer_selection(a, lock):
    if (Path(a.private) / "B0").exists() or any((Path(a.private) / "risk" / f"outer{x}_final").exists() for x in range(1, 6)):
        raise ValueError("CANNOT_LOCK_SELECTION_AFTER_ANY_OUTER_SCORE")
    entries = {}
    for outer in range(1, 6):
        pretraining, risk = {}, {}
        for family, arms in lock["family_readouts"].items():
            pretraining[family] = {}
            for seed in (31, 47, 83):
                pretraining[family][str(seed)] = inner_phase_choices(a, outer, family, seed)
                for arm in arms:
                    risk.setdefault(arm, {})[str(seed)] = inner_risk_choice(a, outer, arm, seed)
        entries[str(outer)] = {"pretraining": pretraining, "risk": risk}
    value = {"schema": "semantic_p2_all_outer_selections_v1", "status": "ALL_FIVE_OUTERS_SELECTED_BEFORE_FIRST_OUTER_SCORE",
             "p2_lock_sha256": a.expected_lock_sha256, "outer_results_read": False, "entries": entries}
    write_json(Path(a.private) / "ALL_OUTER_SELECTIONS_FROZEN.private.json", value)
    return {"status": value["status"], "folds": 5}


def frozen_outer_selection(a):
    value = read_json(Path(a.private) / "ALL_OUTER_SELECTIONS_FROZEN.private.json")
    if value["p2_lock_sha256"] != a.expected_lock_sha256 or value["status"] != "ALL_FIVE_OUTERS_SELECTED_BEFORE_FIRST_OUTER_SCORE":
        raise ValueError("ALL_OUTER_SELECTIONS_NOT_FROZEN")
    return value


def final_epoch(a):
    split = split_entry(a)
    if split["role"] != "outer_final":
        return None
    value = frozen_outer_selection(a)
    expected = value["entries"][str(split["outer"])]["pretraining"][a.family][str(a.seed)]
    if expected != inner_phase_choices(a, split["outer"], a.family, a.seed):
        raise ValueError("FROZEN_OUTER_EPOCH_SELECTION_CHANGED")
    return expected["epoch"]


def phase_a(a, lock):
    data, prep, split, sha = build_inputs(a, lock)
    definition = lock["phase_a_families"][a.family]
    epoch = final_epoch(a)
    receipt = run_cell(data, definition, a.seed, phase_path(a), lock["phase_a_lock"],
        input_sha256=sha, fixed_epoch=epoch, evaluate_valid=split["role"] == "inner", threads=a.threads)
    # Only fitted partition statistics; large immutable images are not duplicated.
    preprocessing_binding = shared_preprocessing(a.private, a.split, prep)
    write_json(phase_path(a) / "MATRIX_BINDING.private.json", {
        "p2_lock_sha256": a.expected_lock_sha256, "split": a.split, "family": a.family,
        "seed": a.seed, "training_only_input_sha256": sha, "fixed_epoch": epoch,
        "shared_preprocessing": preprocessing_binding,
        "cell_receipt_sha256": file_sha(phase_path(a) / "cell.private.json")})
    return {"status": receipt["status"], "selected_epoch": receipt["selected_epoch"]}


def anchor(a, lock):
    raw, membership, _ = raw_inputs(a, endpoints=True)
    split = split_entry(a)
    start = time.monotonic()
    recipe = lock["B0_R4_EXACT"]
    bundle = fit_r4_anchor(raw, split["fit"], split["evaluation"], membership[1],
        expected_union_sha256=recipe["legacy_eight_program_union_sha256"],
        legacy_root=a.legacy_root, expected_legacy_source_hashes=recipe["legacy_source_sha256"],
        teacher_seed=split["teacher_seed"], pipeline_crossfit_seed=split["teacher_seed"],
        downstream_crossfit=True)
    out = Path(a.private) / "anchors" / (a.split + ".private.pkl")
    write_pickle(out, bundle)
    receipt = {**bundle.receipt, "elapsed_seconds": time.monotonic() - start,
               "p2_lock_sha256": a.expected_lock_sha256, "artifact_sha256": file_sha(out),
               "split": a.split, "old_scores_reused": False}
    if split["role"] == "inner":
        tr, ev = np.asarray(split["fit"]), np.asarray(split["evaluation"])
        receipt["B0_inner_metrics"] = evaluate_safe(raw["t"][tr], raw["e"][tr], raw["t"][ev], raw["e"][ev], bundle.b0_exact_evaluation)
    # Formal outer B0 score is emitted only by risk_bundle after all necessary
    # inner receipts exist. The teacher itself never receives evaluation labels.
    write_json(out.with_suffix(".json"), receipt)
    return {"status": "COMPLETE_ANCHOR", "split": a.split,
            "Cox_fits": bundle.receipt["Cox_optimizer_fits"]}


def risk_bundle(a, lock):
    data, _, split, sha = build_inputs(a, lock)
    source = phase_path(a)
    receipt = read_json(source / "cell.private.json")
    binding = read_json(source / "MATRIX_BINDING.private.json")
    if binding["p2_lock_sha256"] != a.expected_lock_sha256 or binding["training_only_input_sha256"] != sha:
        raise ValueError("PHASE_A_TRAINING_BINDING_CHANGED")
    checkpoint = source / receipt["source_state_file"]
    if file_sha(checkpoint) != receipt["selected_checkpoint_sha256"]:
        raise ValueError("PHASE_A_SELECTED_CHECKPOINT_CHANGED")
    model, _ = load_checkpoint_model(checkpoint, device="cuda", freeze=True, expected_input_sha256=sha)
    outputs = extract_frozen_features(model, data, torch.arange(len(data["global_rows"])))
    model.cpu()
    torch.cuda.empty_cache()
    # The closure performs no fitting and cannot read endpoints.
    def operator(arrays, mu_visual, mu_gene, arm):
        tensors = {k: torch.as_tensor(v) for k, v in arrays.items()}
        with torch.no_grad():
            result = frozen_readout_operator(model, tensors,
                None if mu_visual is None else torch.as_tensor(mu_visual, dtype=torch.float32),
                None if mu_gene is None else torch.as_tensor(mu_gene, dtype=torch.float32), arm=arm)
        return result["mechanism"].cpu().numpy()
    raw, _, _ = raw_inputs(a, endpoints=True)
    rows = data["global_rows"].numpy()
    tr, ev = data["train_indices"].numpy(), data["valid_indices"].numpy()
    anchor_bundle, anchor_receipt = verified_artifact(Path(a.private) / "anchors" / (a.split + ".private.pkl"))
    if anchor_receipt["p2_lock_sha256"] != a.expected_lock_sha256:
        raise ValueError("ANCHOR_MASTER_LOCK_CHANGED")
    if not np.array_equal(anchor_bundle.train_rows, rows[tr]) or not np.array_equal(anchor_bundle.evaluation_rows, rows[ev]):
        raise ValueError("ANCHOR_AND_NEURAL_PARTITION_ORDER_CHANGED")
    arms = lock["family_readouts"][a.family]
    records = []
    for arm in arms:
        destination = Path(a.private) / "risk" / a.split / (arm + "_seed" + str(a.seed))
        features = frozen_feature_views(outputs, data["clinical_raw"], tr, ev, arm=arm, operator=operator)
        selection = None
        lambdas = [.1, 1., 10.]
        if split["role"] == "outer_final":
            chosen = frozen_outer_selection(a)
            selection = chosen["entries"][str(split["outer"])]["risk"][arm][str(a.seed)]
            if selection != inner_risk_choice(a, split["outer"], arm, a.seed):
                raise ValueError("FROZEN_OUTER_RISK_SELECTION_CHANGED")
            write_json(destination / "SELECTION_BEFORE_OUTER_EVALUATION.private.json", selection)
            lambdas = [selection["ridge_lambda"]]
        for lam in lambdas:
            result = fit_risk_cell(features["train"], features["evaluation"],
                raw["t"][rows[tr]], raw["e"][rows[tr]], raw["t"][rows[ev]], raw["e"][rows[ev]],
                anchor_bundle.training_offset, anchor_bundle.evaluation_offset,
                ridge_lambda=lam, role=split["role"], selection_record=selection,
                bindings={"lock_sha256": a.expected_lock_sha256, "split": a.split, "arm": arm,
                    "seed": a.seed, "phase_a_source_family": a.family,
                    "phase_a_selected_state_sha256": receipt["selected_checkpoint_sha256"],
                    "input_training_sha256": sha, "anchor_training_sha256": array_sha(anchor_bundle.training_offset)})
            path = destination / ("lambda" + str(lam) + ".private.pkl")
            write_pickle(path, result)
            saved = load_pickle(path)
            replay = replay_risk_cell(saved, features["evaluation"], anchor_bundle.evaluation_offset,
                raw["t"][rows[tr]], raw["e"][rows[tr]], raw["t"][rows[ev]], raw["e"][rows[ev]])
            row = {**result.receipt, "artifact_sha256": file_sha(path), "saved_linear_head_replay": replay}
            row["actual_feature_map_refits"] = {"PCA": 3, "current_training_only": True,
                "same_features_refit_per_lambda_not_counted_as_free": True}
            row["background_models_in_current_frozen_views"] = features["background"]["audit"]["background_models"] if features["background"] else 0
            row["shared_r0_contains_TITAN_S32"] = True
            row["textless_control_is_incremental_encoder_only"] = arm == "B3_TEXTLESS"
            if arm == "B3_TEXTLESS":
                row.update(explicit_text_in_adapter=False, common_anchor_contains_TITAN_S32=True,
                           whole_risk_text_free=False, reported_arm="B3_TEXTLESS_INCREMENT")
            write_json(path.with_suffix(".json"), row)
            records.append({"arm": arm, "ridge_lambda": lam, "metrics": row["evaluation_metrics"]})
            # Exact post-map designs are small. Do not duplicate raw 50x64
            # latent views for every arm: retain model/statistics for optional
            # full reconstruction, and identify this replay as head-only.
            write_pickle(destination / ("lambda" + str(lam) + ".compact.private.pkl"), {
                "train_design": result.design.transform(features["train"]),
                "evaluation_design": result.design.transform(features["evaluation"]),
                "training_offset": anchor_bundle.training_offset,
                "evaluation_offset": anchor_bundle.evaluation_offset,
                "evaluation_time": raw["t"][rows[ev]], "evaluation_event": raw["e"][rows[ev]],
                "train_time": raw["t"][rows[tr]], "train_event": raw["e"][rows[tr]],
                "feature_reconstruction": "retained_frozen_encoder_plus_partition_preprocessing_and_refit_no_DSS_background",
                "background_audit": features["background"]["audit"] if features["background"] else None})
    if split["role"] == "outer_final" and a.family == lock["B0_R4_EXACT"]["report_with_family"] and a.seed == 31:
        metrics = evaluate_safe(raw["t"][rows[tr]], raw["e"][rows[tr]], raw["t"][rows[ev]], raw["e"][rows[ev]], anchor_bundle.b0_exact_evaluation)
        write_json(Path(a.private) / "B0" / (a.split + ".private.json"), {"arm": "B0_R4_EXACT", "fold": split["outer"],
            "DSS_metrics": metrics, "seed": 31, "deterministic_single_refit": True,
            "teacher_seed": split["teacher_seed"], "historical_three_seed_average_reproduced": False,
            "source_anchor_sha256": file_sha(Path(a.private) / "anchors" / (a.split + ".private.pkl"))})
    write_json(source / "RISK_BUNDLE_COMPLETE.private.json", {"status": "COMPLETE", "records": records,
        "frozen_source_checkpoint_sha256": receipt["selected_checkpoint_sha256"], "p2_lock_sha256": a.expected_lock_sha256})
    return {"status": "COMPLETE_RISK_BUNDLE", "split": a.split, "seed": a.seed, "readouts": len(arms)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("split-plan", "anchor", "phase-a", "risk-bundle", "cell", "freeze-outer-selection"))
    for name in ("lock", "expected-lock-sha256", "private", "assets", "legacy-root", "frozen-images"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split")
    p.add_argument("--family")
    p.add_argument("--seed", type=int, choices=(31, 47, 83))
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    lock = verify_lock(a)
    if a.command == "split-plan":
        raw, _, _ = raw_inputs(a, endpoints=True)
        plan = nested_split_plan(raw["folds"], raw["e"], seed=lock["inner_split_seed"])
        write_json(Path(a.private) / "SPLITS.private.json", {"p2_lock_sha256": a.expected_lock_sha256,
            "partitions": plan, "outcome_fields_in_plan": False})
        result = {"status": "COMPLETE_SPLIT_PLAN", "partitions": len(plan)}
    elif a.command == "freeze-outer-selection":
        result = freeze_outer_selection(a, lock)
    elif a.command == "anchor":
        result = anchor(a, lock)
    elif a.command == "phase-a":
        result = phase_a(a, lock)
    elif a.command == "cell":
        phase_a(a, lock)
        torch.cuda.empty_cache()
        result = risk_bundle(a, lock)
    else:
        result = risk_bundle(a, lock)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

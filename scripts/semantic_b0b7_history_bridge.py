"""P0-only, zero-training rescoring of saved historical M2R2 risk vectors.

This process is separate from P1. It may read the pre-existing DSS endpoint
and private saved risk vectors, but never imports a legacy model/runner,
loads weights, calls a model, fits a risk model, or changes old artifacts.
Private paths are supplied by CLI and never copied into the public report.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantic_b0b7.survival import aggregate_folds, evaluate_fold


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_array(values):
    array = np.ascontiguousarray(values)
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_canonical_loader(root):
    file = Path(root) / "src/bgb_en/data.py"
    spec = importlib.util.spec_from_file_location("semantic_b0b7_bridge_canonical_data", file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, file


def legacy_labels(clinical_path, patient_ids):
    """Reproduce only the old label parser; no legacy feature/model loading."""
    table = pd.read_csv(clinical_path, sep="\t", comment="#", low_memory=False, dtype=str)
    needed = {"PATIENT_ID", "DSS_MONTHS", "DSS_STATUS"}
    if not needed.issubset(table.columns):
        raise ValueError("LEGACY_CLINICAL_ENDPOINT_COLUMNS_MISSING")
    def normalize(value):
        match = re.match(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(value).strip().upper())
        return match.group(1) if match else None
    def status(value):
        match = re.match(r"\s*([01])", str(value))
        return float(match.group(1)) if match else float("nan")
    table["_patient"] = table["PATIENT_ID"].map(normalize)
    table = table.dropna(subset=["_patient"]).drop_duplicates("_patient", keep="first").set_index("_patient")
    table = table.reindex(patient_ids)
    time = pd.to_numeric(table["DSS_MONTHS"], errors="coerce").to_numpy(dtype=np.float64)
    event = table["DSS_STATUS"].map(status).to_numpy(dtype=np.float64)
    if not np.isfinite(time).all() or (time <= 0).any() or not np.isin(event, (0, 1)).all():
        raise ValueError("LEGACY_DSS_ENDPOINT_INVALID")
    return time, event.astype(np.int64)


def model_labels(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODEL_LABELS" for t in node.targets):
            labels = ast.literal_eval(node.value)
            if labels.get("B5") != "MCAT":
                raise ValueError("LEGACY_MCAT_METHOD_ID_CHANGED")
            return labels
    raise ValueError("LEGACY_METHOD_LABEL_MAP_MISSING")


def legacy_metric_functions(source):
    """Load only four existing NumPy scorer bodies, never a legacy model."""
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    wanted = {"harrell_c", "censoring_survival", "lookup_censor_survival", "uno_c_index"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError("LEGACY_SCORER_FUNCTIONS_MISSING")
    # Preserve deferred annotation evaluation without importing the old torch
    # module. Only the original four NumPy metric functions are executed.
    future = ast.parse("from __future__ import annotations\n").body[0]
    namespace = {"np": np}
    exec(compile(ast.Module(body=[future] + nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def bridge(args):
    scorer = ROOT / "src/semantic_b0b7/survival.py"
    if sha_file(scorer) != args.expected_scorer_sha256:
        raise ValueError("UNIFIED_SCORER_SOURCE_SHA_MISMATCH")
    source_root = Path(args.legacy_source_root)
    legacy_data = source_root / "src/phase_m2/data.py"
    legacy_runner = source_root / "src/phase_m2/runner.py"
    legacy_common = source_root / "src/phase_m2/common.py"
    split_path = source_root / "outputs/phase_m1r/fixed_patient_folds.csv"
    clinical_path = Path(args.legacy_data_root) / "data/data_8/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
    config = json.loads(Path(args.assets).read_text(encoding="utf-8"))
    loader, loader_file = load_canonical_loader(args.canonical_loader_root)
    raw = loader.load_raw(config, include_endpoints=True)
    ids, folds = np.asarray(raw["patient_ids"], str), np.asarray(raw["folds"], np.int64)
    time, event = np.asarray(raw["t"], np.float64), np.asarray(raw["e"], np.int64)
    split = pd.read_csv(split_path, dtype={"patient_id": str, "fold": int})
    split = split.sort_values("patient_id").reset_index(drop=True)
    if (len(split) != 512 or split["patient_id"].duplicated().any()
            or not np.array_equal(split["patient_id"].to_numpy(str), ids)
            or not np.array_equal(split["fold"].to_numpy(np.int64), folds)):
        raise ValueError("LEGACY_CANONICAL_PATIENT_ORDER_OR_FOLD_MISMATCH")
    old_time, old_event = legacy_labels(clinical_path, ids)
    endpoint_identity = np.array_equal(old_time, time) and np.array_equal(old_event, event)
    if not endpoint_identity:
        raise ValueError("LEGACY_CANONICAL_DSS_ENDPOINT_NOT_BITWISE_EQUAL_NO_FORCED_RESCORING")
    labels = model_labels(legacy_runner)
    legacy_scorers = legacy_metric_functions(legacy_common)
    rows, audit_rows, missing = [], [], []
    legacy_metric_replay_errors = []
    inventory = []
    seen = set()
    for ordinal, root_value in enumerate(args.runtime_root):
        runtime = Path(root_value)
        cache_metadata = runtime / "path_tokens_meta.json"
        cells = runtime / "cells"
        if not cells.is_dir():
            missing.append({"runtime_ordinal": ordinal, "reason": "SAVED_RISK_DIRECTORY_MISSING"})
            continue
        meta = json.loads(cache_metadata.read_text(encoding="utf-8"))
        if meta.get("patient_order_sha256") != raw["audit"]["patient_order_sha256"]:
            raise ValueError("LEGACY_RISK_PATIENT_ORDER_PROVENANCE_MISMATCH")
        files = sorted(cells.glob("*.json"))
        inventory.append({"runtime_ordinal": ordinal, "cell_metadata_files": len(files),
                          "cache_metadata_sha256": sha_file(cache_metadata)})
        for path in files:
            match = re.fullmatch(r"(B[0-6])_(\d+)_([1-5])\.json", path.name)
            if match is None:
                continue
            method, seed, fold = match[1], int(match[2]), int(match[3])
            key = method, seed, fold
            if key in seen:
                raise ValueError("DUPLICATE_HISTORICAL_CELL")
            seen.add(key)
            metadata = json.loads(path.read_text(encoding="utf-8"))
            row, profile = metadata["row"], metadata["profile"]
            if (row.get("status") != "completed" or profile.get("status") != "completed"
                    or row.get("method") != method or row.get("seed") != seed
                    or row.get("outer_fold") != fold):
                raise ValueError("HISTORICAL_CELL_IDENTITY_OR_COMPLETION_MISMATCH")
            prediction_path = path.with_suffix(".npy")
            if not prediction_path.is_file():
                missing.append({"method": method, "seed": seed, "fold": fold, "reason": "SAVED_RISK_MISSING_NO_RETRAIN"})
                continue
            risk = np.load(prediction_path, allow_pickle=False)
            test, train = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
            if risk.shape != (len(test),) or not np.isfinite(risk).all():
                raise ValueError("HISTORICAL_RISK_SHAPE_OR_FINITE_MISMATCH")
            if row["n_test"] != len(test) or row["events_test"] != int(event[test].sum()):
                raise ValueError("HISTORICAL_TEST_SUPPORT_MISMATCH")
            old_h = legacy_scorers["harrell_c"](time[test], event[test], risk)
            old_u, old_un, old_ud = legacy_scorers["uno_c_index"](time[test], event[test], risk, time[train], event[train])
            old_recomputed = np.asarray([old_h, old_u, old_un, old_ud])
            old_saved = np.asarray([row["harrell_c"], row["uno_c"], row["uno_numerator"], row["uno_denominator"]])
            if not np.isfinite(old_recomputed).all() or not np.allclose(old_recomputed, old_saved, rtol=0, atol=1e-10):
                raise ValueError("SAVED_RISK_DOES_NOT_REPLAY_ORIGINAL_FOLD_METRICS")
            old_error = float(np.max(np.abs(old_recomputed-old_saved)))
            legacy_metric_replay_errors.append(old_error)
            metric = evaluate_fold(time[train], event[train], time[test], event[test], risk, tau=60.)
            if not metric["tau_support"]["supported"]:
                raise ValueError("HISTORICAL_BRIDGE_UNO60_TAU_SUPPORT_FAILURE")
            entry = {"method": method, "method_label": labels[method], "seed": seed, "fold": fold,
                     "test_patients": len(test), "test_events": int(event[test].sum()), **metric,
                     "old_saved_fold_Harrell": float(row["harrell_c"]),
                     "old_saved_fold_Uno": float(row["uno_c"]),
                     "original_metric_replay_max_abs_error": old_error,
                     "risk_file_sha256": sha_file(prediction_path),
                     "metadata_sha256": sha_file(path), "risk_array_sha256": sha_array(risk)}
            rows.append(entry)
            audit_rows.append({"method": method, "seed": seed, "fold": fold,
                               "runtime": str(runtime), "risk_path": str(prediction_path),
                               "metadata_path": str(path), "test_row_hash": sha_array(test)})
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["seed"]), []).append(row)
    seeds = []
    for (method, seed), cells in sorted(grouped.items()):
        if len(cells) != 5 or {x["fold"] for x in cells} != {1, 2, 3, 4, 5}:
            missing.append({"method": method, "seed": seed, "reason": "INCOMPLETE_FIVE_FOLD_SAVED_RISK"})
            continue
        seeds.append({"method": method, "method_label": labels[method], "seed": seed,
                      **aggregate_folds(cells), "saved_fold_cells": 5})
    methods = []
    for method in sorted({x["method"] for x in seeds}):
        cells = [x for x in seeds if x["method"] == method]
        methods.append({"method": method, "method_label": labels[method],
                        "seeds": [x["seed"] for x in cells], "seed_count": len(cells),
                        "mean_within_fold_Harrell": float(np.mean([x["Harrell"] for x in cells])),
                        "mean_within_fold_Uno60": float(np.mean([x["Uno60"] for x in cells])),
                        "neural_or_risk_models_refitted": 0,
                        "comparison_status": "HISTORICAL_ADAPTED_RECIPE_BRIDGE_NOT_SAME_INPUT_ARCHITECTURE_COMPARISON"})
    report = {"schema": "semantic_b0b7_p0_historical_baseline_bridge_v1",
              "status": "RESCORED_SAVED_RISKS_NO_RETRAIN" if methods and not missing else "PARTIAL_SAVED_RISK_BRIDGE" if rows else "MISSING_NO_RETRAIN",
              "scope": "P0_READONLY_ENDPOINT_BRIDGE_SEPARATE_FROM_OUTCOME_FREE_P1",
              "cohort": {"patients": len(ids), "DSS_events": int(event.sum()), "time_unit": "months",
                         "patient_order_sha256": raw["audit"]["patient_order_sha256"],
                         "fold_array_sha256": sha_array(folds), "DSS_time_array_sha256": sha_array(time),
                         "DSS_event_array_sha256": sha_array(event),
                         "canonical_cohort_sha256": raw["audit"]["cohort_sha256"],
                         "canonical_endpoint_file_sha256": raw["audit"]["endpoints_sha256"],
                         "legacy_split_file_sha256": sha_file(split_path),
                         "legacy_primary_clinical_file_sha256": sha_file(clinical_path),
                         "patient_order_exact_match": True, "patient_fold_identity_exact_match": True,
                         "DSS_time_and_event_bitwise_equal_after_declared_numeric_parsing": True,
                         "fold_patient_counts": {str(f): int((folds == f).sum()) for f in range(1, 6)},
                         "fold_event_counts": {str(f): int(event[folds == f].sum()) for f in range(1, 6)}},
              "source_sha256": {"scripts/semantic_b0b7_history_bridge.py": sha_file(__file__),
                                "src/semantic_b0b7/survival.py": sha_file(scorer),
                                "legacy_data_loader": sha_file(legacy_data),
                                "legacy_runner": sha_file(legacy_runner),
                                "legacy_numpy_metric_source": sha_file(legacy_common),
                                "audited_canonical_loader": sha_file(loader_file)},
              "inventory": inventory, "saved_fold_risks_scored": len(rows),
              "complete_seed_five_fold_sets": len(seeds), "fold_metrics": rows,
              "original_metric_replayed_fold_vectors": len(legacy_metric_replay_errors),
              "original_metric_replay_max_abs_error": max(legacy_metric_replay_errors) if legacy_metric_replay_errors else None,
              "seed_metrics": seeds, "method_metrics": methods, "missing": missing,
              "new_training_fits": 0, "optimizer_updates": 0, "model_forwards": 0,
              "model_checkpoints_loaded": 0, "patient_level_arrays_exported": False,
              "historical_result_files_modified": False,
              "evaluation_contract": "same new scorer; fold-local Harrell and train-fold reverse-KM Uno60; each seed sums within-fold numerators/denominators, then seed mean",
              "historical_evaluation_contract": "legacy Harrell excluded event/censor equal-time pairs; saved Uno had no 60-month truncation and used the original censor-KM implementation",
              "historical_training_contract": "saved UNI2-h K1024 / clinical6 / RNA-CNA or signature adapted models; historical Breslow Cox recipe retained, no Efron retraining",
              "reproduction_status": "HISTORICAL_REIMPLEMENTATION_ADAPTED_TO_PROJECT_INPUTS_NOT_NATIVE_PAPER_REPRODUCTION",
              "limitations": ["Risk vectors inherit their historical training and selection limitations.",
                              "Inputs differ from the new CONCH/TITAN semantic matrix; scores do not establish architecture superiority.",
                              "No cross-fold pooled risk concordance is a primary metric.",
                              "The same TCGA cohort has been repeatedly used for development; this is not independent external validation.",
                              "Evaluation repair cannot repair a historical training objective or preprocessing lineage."]}
    write_new(args.output_public, report)
    write_new(args.output_private, {"schema": "semantic_b0b7_historical_bridge_private_lineage_v1",
                                  "public_report_sha256": sha_file(args.output_public),
                                  "legacy_source_root": str(source_root), "cells": audit_rows})
    print(json.dumps({"status": report["status"], "saved_fold_risks_scored": len(rows),
                      "complete_seed_five_fold_sets": len(seeds), "method_metrics": methods,
                      "public_report_sha256": sha_file(args.output_public)}, sort_keys=True))


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets", required=True)
    cli.add_argument("--canonical-loader-root", required=True)
    cli.add_argument("--legacy-source-root", required=True)
    cli.add_argument("--legacy-data-root", required=True)
    cli.add_argument("--runtime-root", action="append", required=True)
    cli.add_argument("--expected-scorer-sha256", required=True)
    cli.add_argument("--output-public", required=True)
    cli.add_argument("--output-private", required=True)
    return cli


if __name__ == "__main__":
    bridge(parser().parse_args())

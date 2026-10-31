"""Receipt-only P2S development diagnostic reporting: no fit, forward or scoring.

Only JSON receipts are opened. Within-partition metric components already saved
by the independently locked producer are aggregated; patient risks, features,
weights, endpoints and the previous P2 outer predictions are never loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (31, 47, 83)
SPLITS = tuple("outer1_inner" + str(i) for i in range(3))
FAMILIES = ("MAIN", "ATTN_MATCHED")
PAIR_WEIGHTS = (.1, .03)
EPOCHS = (400, 800)
RIDGES = (.1, 1., 10.)
PRIMARY_RIDGE = 1.
MAX_BYTES = 2 * 1024 ** 2
MAX_CSV_LINES = 20_000
MAX_JSON_LIST = 1_000
MAX_NUMERIC_LIST = 128
METRIC_COMPONENTS = {"Harrell": ("Harrell_num", "Harrell_den"), "Uno60": ("Uno_num", "Uno_den")}
FORBIDDEN_TEXT = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:root|home|mnt|lustre|autodl-tmp)/|hf_[A-Za-z0-9]{12}|TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|connect\.west[a-z0-9]*\.seetacloud\.com|-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----)")
RAW_JSON_KEYS = frozenset(("patient_ids", "patient_id", "barcodes", "barcode", "rowids", "patient_indices", "train_indices", "test_indices",
                         "fit_indices", "valid_indices", "coordinates", "coords", "risk_vector", "OOF_predictions", "predictions", "RNA_matrix", "CNA_matrix", "state_dict", "optimizer_state"))
ARRAY_ONLY_PRIVATE_KEYS = frozenset(("risk", "risks", "pred", "prediction", "weights", "coefficients", "features", "embedding", "embeddings"))
STATUS_FIT = "FITTED_AND_REPLAYED"
STATUS_NA = "NA_NO_SUPPORTED_RANK"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, expected_sha=None):
    path = Path(path)
    require(path.suffix == ".json", "REPORTER_ONLY_OPENS_JSON_RECEIPTS")
    if expected_sha is not None:
        require(sha(path) == expected_sha, "RECEIPT_BYTES_CHANGED")
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def safe_receipt(root, relative, expected_sha):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "UNSAFE_RECEIPT_REFERENCE")
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "RECEIPT_ESCAPES_ROOT")
    return read_json(path, expected_sha)


def public_value(value):
    if isinstance(value, dict):
        require(not RAW_JSON_KEYS.intersection(value), "PUBLIC_ROW_LEVEL_FIELD_FORBIDDEN")
        require(not any(k in ARRAY_ONLY_PRIVATE_KEYS and isinstance(v, (tuple, list)) for k, v in value.items()), "PUBLIC_RAW_ARRAY_FIELD_FORBIDDEN")
        return {str(k): public_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        require(len(value) <= MAX_JSON_LIST, "PUBLIC_JSON_LIST_TOO_LONG")
        if value and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            require(len(value) <= MAX_NUMERIC_LIST, "PUBLIC_NUMERIC_LIST_TOO_LONG")
        return [public_value(v) for v in value]
    require(value is None or isinstance(value, (bool, int, float, str)), "PUBLIC_SCALARS_ONLY_NO_ARRAYS")
    if isinstance(value, float):
        require(math.isfinite(value), "NONFINITE_PUBLIC_NUMBER")
    if isinstance(value, str):
        require(FORBIDDEN_TEXT.search(value) is None, "PRIVATE_PATH_IDENTIFIER_OR_CREDENTIAL")
    return value


def json_bytes(value):
    data = (json.dumps(public_value(value), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(len(data) <= MAX_BYTES, "PUBLIC_FILE_TOO_LARGE")
    return data


def write_file(path, payload):
    path = Path(path)
    require(len(payload) <= MAX_BYTES, "PUBLIC_FILE_TOO_LARGE")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return {"file": path.name, "bytes": len(payload), "sha256": sha(path)}


def write_json(path, value):
    return write_file(path, json_bytes(value))


def flatten(value, prefix="", *, excluded=()):
    """Copy scalar diagnostics only; arbitrary vectors never become CSV cells."""
    result = {}
    for key, item in value.items():
        if key in excluded:
            continue
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name + ".", excluded=excluded))
        elif item is None or isinstance(item, (bool, int, float, str)):
            result[name] = public_value(item)
        elif isinstance(item, (list, tuple)):
            # PCA spectra/program lists have their own table; no vector escape.
            continue
        else:
            raise ValueError("NONSCALAR_DIAGNOSTIC:" + name)
    return result


def write_csv_parts(output, stem, rows):
    """Normal UTF-8 CSV, capped by actual bytes and header-inclusive line count."""
    rows = list(rows)
    if not rows:
        return []
    fields = sorted({key for row in rows for key in row})
    header_stream = io.StringIO(newline="")
    csv.writer(header_stream, lineterminator="\n").writerow(fields)
    header = header_stream.getvalue().encode("utf-8")
    pieces, current, size = [], [], len(header)
    for raw in rows:
        row = public_value(raw)
        require(all(not isinstance(v, (dict, list)) for v in row.values()), "CSV_MUST_CONTAIN_SCALARS")
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerow([row.get(key) for key in fields])
        encoded = stream.getvalue().encode("utf-8")
        require(b"\n" not in encoded.rstrip(b"\n") and b"\r" not in encoded,
                "MULTILINE_CSV_FIELDS_FORBIDDEN")
        require(len(header) + len(encoded) <= MAX_BYTES, "SINGLE_CSV_ROW_TOO_LARGE")
        if current and (len(current) + 2 > MAX_CSV_LINES or size + len(encoded) > MAX_BYTES):
            pieces.append(header + b"".join(current)); current, size = [], len(header)
        current.append(encoded); size += len(encoded)
    if current:
        pieces.append(header + b"".join(current))
    return [write_file(Path(output) / (stem + ("" if len(pieces) == 1 else "_%02d" % (i + 1)) + ".csv"), piece)
            for i, piece in enumerate(pieces)]


def write_json_parts(output, stem, rows):
    pieces, current, size = [], [], 5
    for row in rows:
        encoded = json_bytes(row).rstrip(b"\n")
        encoded = b"  " + encoded.replace(b"\n", b"\n  ")
        require(len(encoded) + 5 <= MAX_BYTES, "SINGLE_JSON_ROW_TOO_LARGE")
        if current and (len(current) >= MAX_JSON_LIST or size + 2 + len(encoded) > MAX_BYTES):
            pieces.append(b"[\n" + b",\n".join(current) + b"\n]\n")
            current, size = [], 5
        size += len(encoded) + (2 if current else 0)
        current.append(encoded)
    if current:
        pieces.append(b"[\n" + b",\n".join(current) + b"\n]\n")
    return [write_file(Path(output) / (stem + ("" if len(pieces) == 1 else "_%02d" % (i + 1)) + ".json"), piece)
            for i, piece in enumerate(pieces)]


def expected_states():
    return set(itertools.product(SPLITS, FAMILIES, SEEDS, PAIR_WEIGHTS, EPOCHS))


def specs(family):
    require(family in FAMILIES, "UNKNOWN_FAMILY")
    result = []
    for width in (8, 16):
        result.append((width, "SHARED", 0, "NO_INTER"))
        if family == "MAIN":
            result.extend((width, arm, rank, mode) for arm in ("B4", "B5")
                          for rank in (2, 4) for mode in ("INTER", "B7"))
        else:
            result.append((width, "ATTN_MATCHED", 2, "INTER"))
    return tuple(result)


def spec_key(row):
    s = row["spec"]
    return int(s["main_pca"]), s["base_arm"], int(s["factor_rank"]), s["mode"]


def config_key(row):
    return (row["family"], float(row["lambda_pair"]), int(row["epoch"]), *spec_key(row), float(row["ridge_lambda"]))


def config_dict(key):
    return dict(zip(("family", "lambda_pair", "epoch", "main_pca", "base_arm", "factor_rank", "mode", "ridge_lambda"), key))


def metric_value(metric, name):
    nkey, dkey = METRIC_COMPONENTS[name]
    n, d = metric.get(nkey), metric.get(dkey)
    if n is None or d is None or d <= 0 or (name == "Uno60" and metric.get("Uno60_valid") is not True):
        require(metric.get(name) is None, "UNSUPPORTED_METRIC_MUST_BE_EXPLICIT_NA")
        return None
    require(isinstance(n, (int, float)) and isinstance(d, (int, float)) and math.isfinite(n) and math.isfinite(d),
            "INVALID_METRIC_COMPONENTS")
    value = n / d
    require(-1e-12 <= value <= 1 + 1e-12 and metric.get(name) is not None and math.isclose(value, metric[name], abs_tol=1e-11, rel_tol=1e-11),
            "METRIC_COMPONENTS_DISAGREE")
    return value


def fold_micro(rows, name):
    require(len(rows) == 3 and {r["split"] for r in rows} == set(SPLITS), "THREE_INNER_PARTITIONS_REQUIRED")
    require(len({r["split"] for r in rows}) == len(rows), "DUPLICATE_INNER_PARTITION")
    metrics = [r.get("evaluation_metrics") if r.get("status") == STATUS_FIT else None for r in rows]
    values = [metric_value(x, name) if x is not None else None for x in metrics]
    if any(x is None for x in values):
        return {"value": None, "num": None, "den": None, "supported_partitions": sum(x is not None for x in values),
                "status": "NA_INCOMPLETE_OR_UNSUPPORTED_PARTITION"}
    nk, dk = METRIC_COMPONENTS[name]
    n, d = math.fsum(m[nk] for m in metrics), math.fsum(m[dk] for m in metrics)
    return {"value": n / d, "num": n, "den": d, "supported_partitions": 3, "status": "COMPLETE"}


def aggregate_config(rows):
    require(len(rows) == 9 and len({(r["seed"], r["split"]) for r in rows}) == 9 and
            {r["seed"] for r in rows} == set(SEEDS), "EXACT_NINE_PAIRED_CELLS_REQUIRED")
    result = {"required_cells": 9, "fitted_cells": sum(r["status"] == STATUS_FIT for r in rows), "seed_results": []}
    for seed in SEEDS:
        subset = [r for r in rows if r["seed"] == seed]
        result["seed_results"].append({"seed": seed, **{m: fold_micro(subset, m) for m in METRIC_COMPONENTS}})
    for metric in METRIC_COMPONENTS:
        values = [x[metric]["value"] for x in result["seed_results"]]
        result[metric] = statistics.fmean(values) if all(x is not None for x in values) else None
        result[metric + "_seed_min"] = min(values) if all(x is not None for x in values) else None
        result[metric + "_seed_max"] = max(values) if all(x is not None for x in values) else None
    result["status"] = "COMPLETE" if result["fitted_cells"] == 9 else "COMPLETE_WITH_EXPLICIT_NA"
    result["aggregation"] = "within_inner_fold_component_micro_per_seed_then_equal_three_seed_mean"
    return result


def compare_terms(name, terms, indexed):
    """A linear contrast of metrics; no cross-fold pairs or reused-patient OOF."""
    require(terms and all(sign in (-1, 1) for sign, _ in terms), "INVALID_COMPARISON_TERMS")
    require(len({key for _, key in terms}) == len(terms), "COMPARISON_REPEATS_A_CONFIGURATION")
    per_cell, seed_rows = [], []
    for seed in SEEDS:
        for split in SPLITS:
            rows = [indexed[key][(seed, split)] for _, key in terms]
            out = {"comparison": name, "seed": seed, "split": split}
            for metric, (_, dk) in METRIC_COMPONENTS.items():
                ms = [r.get("evaluation_metrics") if r["status"] == STATUS_FIT else None for r in rows]
                values = [metric_value(m, metric) if m is not None else None for m in ms]
                supported = all(v is not None for v in values)
                if supported:
                    require(all(math.isclose(ms[0][dk], m[dk], rel_tol=1e-12, abs_tol=1e-10) for m in ms[1:]),
                            "PAIRED_COMPARISON_HAS_DIFFERENT_WITHIN_PARTITION_DENOMINATORS")
                out[metric + "_delta"] = math.fsum(sign * val for (sign, _), val in zip(terms, values)) if supported else None
                out[metric + "_status"] = "COMPLETE" if supported else "NA_SAME_CELL_UNSUPPORTED"
            per_cell.append(out)
        seed_out = {"seed": seed}
        for metric in METRIC_COMPONENTS:
            values = [fold_micro(list(indexed[key][(seed, split)] for split in SPLITS), metric)["value"] for _, key in terms]
            seed_out[metric + "_delta"] = math.fsum(sign * val for (sign, _), val in zip(terms, values)) if all(v is not None for v in values) else None
        seed_rows.append(seed_out)
    summary = {"comparison": name, "terms": [{"sign": sign, **config_dict(key)} for sign, key in terms], "seed_results": seed_rows,
               "required_paired_cells": 9, "is_new_selection_rule": False, "CI_or_p_value": False}
    for metric in METRIC_COMPONENTS:
        values = [r[metric + "_delta"] for r in seed_rows]
        deltas = [r[metric + "_delta"] for r in per_cell]
        summary[metric + "_delta"] = statistics.fmean(values) if all(v is not None for v in values) else None
        summary[metric + "_cell_positive"] = sum(v is not None and v > 0 for v in deltas)
        summary[metric + "_cell_negative"] = sum(v is not None and v < 0 for v in deltas)
        summary[metric + "_cell_zero"] = sum(v == 0 for v in deltas if v is not None)
        summary[metric + "_cell_NA"] = sum(v is None for v in deltas)
        valid = [r for r in per_cell if r[metric + "_delta"] is not None]
        summary[metric + "_worst_cell"] = min(valid, key=lambda r: r[metric + "_delta"]) if valid else None
    return summary, per_cell


def comparisons(indexed):
    """Pre-fixed contrasts, including conditional-increment difference-in-differences."""
    definitions = []
    def add(label, terms):
        if all(key in indexed for _, key in terms):
            definitions.append((label, terms))
        else:
            raise ValueError("LOCKED_COMPARISON_CONFIGURATION_MISSING:" + label)
    for pair, epoch, ridge in itertools.product(PAIR_WEIGHTS, EPOCHS, RIDGES):
        def k(family, d, arm, rank, mode):
            return family, pair, epoch, d, arm, rank, mode, ridge
        prefix = f"p{pair}_e{epoch}_ridge{ridge}"
        for d, arm, rank in itertools.product((8, 16), ("B4", "B5"), (2, 4)):
            inter, additive, no = k("MAIN", d, arm, rank, "INTER"), k("MAIN", d, arm, rank, "B7"), k("MAIN", d, "SHARED", 0, "NO_INTER")
            add(prefix + f"_MAIN_D{d}_{arm}_R{rank}_INTER_minus_NOINTER", [(1, inter), (-1, no)])
            add(prefix + f"_MAIN_D{d}_{arm}_R{rank}_INTER_minus_B7", [(1, inter), (-1, additive)])
            add(prefix + f"_MAIN_D{d}_{arm}_R{rank}_B7_minus_NOINTER", [(1, additive), (-1, no)])
        for d, arm, mode in itertools.product((8, 16), ("B4", "B5"), ("INTER", "B7")):
            add(prefix + f"_MAIN_D{d}_{arm}_{mode}_R4_minus_R2", [(1, k("MAIN", d, arm, 4, mode)), (-1, k("MAIN", d, arm, 2, mode))])
        for arm, rank, mode in itertools.product(("B4", "B5"), (2, 4), ("INTER", "B7")):
            add(prefix + f"_MAIN_{arm}_R{rank}_{mode}_D16_minus_D8", [(1, k("MAIN", 16, arm, rank, mode)), (-1, k("MAIN", 8, arm, rank, mode))])
        for d, rank, mode in itertools.product((8, 16), (2, 4), ("INTER", "B7")):
            add(prefix + f"_MAIN_D{d}_R{rank}_{mode}_B5_minus_B4", [(1, k("MAIN", d, "B5", rank, mode)), (-1, k("MAIN", d, "B4", rank, mode))])
        for d in (8, 16):
            add(prefix + f"_ATTN_D{d}_INTER_minus_NOINTER", [(1, k("ATTN_MATCHED", d, "ATTN_MATCHED", 2, "INTER")), (-1, k("ATTN_MATCHED", d, "SHARED", 0, "NO_INTER"))])
            add(prefix + f"_MAIN_B5_vs_ATTN_D{d}_R2", [(1, k("MAIN", d, "B5", 2, "INTER")), (-1, k("ATTN_MATCHED", d, "ATTN_MATCHED", 2, "INTER"))])
    # Effects of budget/pair-weight are reported for every fixed head; conditional
    # increments additionally subtract the actual same-state no-interaction refit.
    for family in FAMILIES:
        for d, arm, rank, mode in specs(family):
            for pair, ridge in itertools.product(PAIR_WEIGHTS, RIDGES):
                keys = [(family, pair, e, d, arm, rank, mode, ridge) for e in EPOCHS]
                add(f"{family}_p{pair}_ridge{ridge}_D{d}_{arm}_R{rank}_{mode}_800_minus400", [(1, keys[1]), (-1, keys[0])])
                if mode == "INTER":
                    for baseline_mode in ("NO_INTER", "B7") if family == "MAIN" else ("NO_INTER",):
                        baseline = [(family, pair, e, d, "SHARED" if baseline_mode == "NO_INTER" else arm,
                                     0 if baseline_mode == "NO_INTER" else rank, baseline_mode, ridge) for e in EPOCHS]
                        add(f"{family}_p{pair}_ridge{ridge}_D{d}_{arm}_R{rank}_COND_{baseline_mode}_800_minus400",
                            [(1, keys[1]), (-1, baseline[1]), (-1, keys[0]), (1, baseline[0])])
            for epoch, ridge in itertools.product(EPOCHS, RIDGES):
                keys = [(family, p, epoch, d, arm, rank, mode, ridge) for p in (.1, .03)]
                add(f"{family}_e{epoch}_ridge{ridge}_D{d}_{arm}_R{rank}_{mode}_pair003_minus010", [(1, keys[1]), (-1, keys[0])])
                if mode == "INTER":
                    for baseline_mode in ("NO_INTER", "B7") if family == "MAIN" else ("NO_INTER",):
                        baseline = [(family, p, epoch, d, "SHARED" if baseline_mode == "NO_INTER" else arm,
                                     0 if baseline_mode == "NO_INTER" else rank, baseline_mode, ridge) for p in (.1, .03)]
                        add(f"{family}_e{epoch}_ridge{ridge}_D{d}_{arm}_R{rank}_COND_{baseline_mode}_pair003_minus010",
                            [(1, keys[1]), (-1, baseline[1]), (-1, keys[0]), (1, baseline[0])])
    return definitions


def auxiliary_metrics(metrics):
    program = metrics["program"]
    gaps = [metrics["retrieval_hard"][direction]["paired_gap_mean"] for direction in ("image_to_gene", "gene_to_image")]
    gaps = [g for g in gaps if g is not None]
    return {"program_Huber": program["patient_equal_masked_huber"], "macro_program_Pearson": program["macro_program_pearson"],
        "valid_Pearson_programs": program["valid_pearson_programs"], "hard_gap": statistics.fmean(gaps) if gaps else None,
        "patients": program["patients"], "patients_with_measurement": program["patients_with_measurement"],
        "measured_target_cells": program["measured_target_cells"], "coverage_fraction": program["coverage_fraction"],
        "median_prediction_to_target_sd_ratio": metrics.get("median_prediction_to_target_sd_ratio")}


def collect_pretraining(private, lock, *, expected_lock_sha):
    cells = lock["cells"]
    expected = set(itertools.product(SPLITS, FAMILIES, SEEDS, PAIR_WEIGHTS))
    require(len(cells) == 36 and {(c["split"], c["family"], c["seed"], c["lambda_pair"]) for c in cells} == expected,
            "PRETRAINING_EXACT36_LOCK_REQUIRED")
    receipts, checkpoints, curves, gradients, programs, strata, provenance = [], [], [], [], [], [], []
    for cell in cells:
        identity = {k: cell[k] for k in ("split", "family", "seed", "lambda_pair")}
        relative = Path("phase_a") / cell["split"] / (cell["family"] + "_seed" + str(cell["seed"]) + "_pair%03d" % round(cell["lambda_pair"] * 100)) / "cell.private.json"
        r = safe_receipt(private, relative, None)
        require(r.get("schema") == "semantic_b0b7_p2s_cell_v1" and r.get("status") == "COMPLETE" and
                r.get("parent_cell") == cell and r.get("DSS_OS_PFI_accessed") is False and r.get("selected_epoch") is None and
                r.get("no_DSS_or_auxiliary_epoch_selection") is True and r.get("master_lock_sha256") == expected_lock_sha and
                r.get("base_p2_master_lock_sha256") == lock["parent_master_sha256"], "UNQUALIFIED_PRETRAINING_RECEIPT")
        require(r["split"] == cell["split"] and r["family"] == cell["family"] and r["seed"] == cell["seed"] and
                r["spec"]["lambda_pair"] == cell["lambda_pair"] and r["input_sha256"] == cell["input_sha256"] and
                r["frozen_sources"] == lock["frozen_sources"], "PRETRAINING_IDENTITY_OR_SOURCE_CHANGED")
        steps = 400 if cell["lambda_pair"] == .1 else 800
        require(r["optimizer_steps"] == steps and r["optimizer_lifetime_steps"] == 800 and
                r["resume_epoch"] == 800 - steps and len(r["history"]) == steps and
                [x["epoch"] for x in r["history"]] == list(range(801 - steps, 801)), "PRETRAINING_UPDATE_BUDGET_CHANGED")
        require([x["epoch"] for x in r["states"]] == list(EPOCHS), "PRETRAINING_FIXED_STATES_MISSING")
        for state in r["states"]:
            require(state["actual_replay"] is True and state["replay_max_error"] == 0 and
                    state["replay_preserved_training_rng"] is True, "PRETRAINING_ACTUAL_REPLAY_MISSING")
            metrics = state["validation"]
            checkpoints.append({**identity, "epoch": state["epoch"], **auxiliary_metrics(metrics), "checkpoint_sha256": state["sha256"],
                                "snapshot_identity": state["identity"], "primary_auxiliary_metrics_not_weighted_total_loss": True})
            for row in metrics["program"]["per_program"]:
                programs.append({**identity, "epoch": state["epoch"], **flatten(row)})
            for variable, levels in metrics.get("strata", {}).items():
                for level, payload in levels.items():
                    strata.append({**identity, "epoch": state["epoch"], "stratum_variable": variable,
                                   "stratum": str(level), **flatten(payload, excluded=("per_program",))})
        for history in r["history"]:
            require(not history["inactive_parameters"] and history["optimizer_step"] == history["epoch"], "PRETRAINING_GRADIENT_OR_STEP_INVALID")
            grad = history["gradient_norms"]
            require(grad and all(x is not None and math.isfinite(x) and x > 0 for x in grad.values()), "INACTIVE_PRETRAINING_PATH")
            curves.append({**identity, **{k: history[k] for k in ("epoch", "optimizer_step", "train_program_loss", "train_pair_loss", "train_auxiliary_loss", "active_parameter_elements")},
                           "gradient_norm_min": min(grad.values()), "gradient_norm_max": max(grad.values()), "new_update": True})
            if history["epoch"] in EPOCHS:
                for name, value in grad.items():
                    gradients.append({**identity, "epoch": history["epoch"], "parameter": name, "gradient_norm": value,
                                      "weight_norm": history["weight_norms"].get(name)})
        receipts.append({**identity, "new_optimizer_updates": steps, "lifetime_updates": 800, "new_initialization": cell["lambda_pair"] == .03,
                         "actual_checkpoint_replays": 2, "new_snapshots": 1 if cell["lambda_pair"] == .1 else 2,
                         "reused_original400_snapshots": int(cell["lambda_pair"] == .1), "elapsed_seconds": r["elapsed_seconds"],
                         "peak_cuda_allocated_bytes": r["peak_cuda_allocated_bytes"], **flatten(r["parameter_counts"], "parameter_counts.")})
        provenance.append({**identity, "receipt_sha256": sha(Path(private) / relative), "input_sha256": cell["input_sha256"],
                           "parent_receipt_sha256": cell["parent_receipt_sha256"], "parent_state400_sha256": cell["parent_state400_sha256"]})
    require(len(curves) == 21600 and len(checkpoints) == 72 and sum(r["new_snapshots"] for r in receipts) == 54,
            "PRETRAINING_TOTAL_BUDGET_NOT21600_72_54")
    return dict(receipts=receipts, checkpoints=checkpoints, curves=curves, gradients=gradients, programs=programs, strata=strata, provenance=provenance)


def auxiliary_comparisons(checkpoints):
    bank = {(r["family"], r["lambda_pair"], r["epoch"], r["seed"], r["split"]): r for r in checkpoints}
    require(len(bank) == 72 and len(checkpoints) == 72, "AUXILIARY_FIXED72_STATE_SET")
    definitions = []
    for family, pair in itertools.product(FAMILIES, PAIR_WEIGHTS):
        definitions.append((f"{family}_pair{pair}_800_minus400", (family, pair, 800), (family, pair, 400)))
    for family, epoch in itertools.product(FAMILIES, EPOCHS):
        definitions.append((f"{family}_epoch{epoch}_pair003_minus010", (family, .03, epoch), (family, .1, epoch)))
    for pair, epoch in itertools.product(PAIR_WEIGHTS, EPOCHS):
        definitions.append((f"MAIN_minus_ATTN_pair{pair}_epoch{epoch}", ("MAIN", pair, epoch), ("ATTN_MATCHED", pair, epoch)))
    details, summaries = [], []
    for name, a, b in definitions:
        local = []
        for seed, split in itertools.product(SEEDS, SPLITS):
            left, right = bank[(*a, seed, split)], bank[(*b, seed, split)]
            row = {"comparison": name, "seed": seed, "split": split}
            for metric in ("program_Huber", "macro_program_Pearson", "hard_gap"):
                row[metric + "_delta"] = left[metric] - right[metric] if left[metric] is not None and right[metric] is not None else None
            local.append(row)
        result = {"comparison": name, "paired_cells": 9, "selection_performed": False,
                  "weighted_total_auxiliary_loss_compared": False, "Huber_lower_is_better": True}
        for metric in ("program_Huber", "macro_program_Pearson", "hard_gap"):
            values = [r[metric + "_delta"] for r in local]
            result[metric + "_mean_paired_delta"] = statistics.fmean(values) if all(x is not None for x in values) else None
            result[metric + "_positive_cells"] = sum(v is not None and v > 0 for v in values)
            result[metric + "_negative_cells"] = sum(v is not None and v < 0 for v in values)
            result[metric + "_NA_cells"] = sum(v is None for v in values)
        details.extend(local); summaries.append(result)
    return summaries, details


def collect_readouts(root, manifest, *, pretrain_sha, readout_sha):
    require(manifest.get("schema") == "semantic_b0b7_p2s_readout_manifest_v1" and manifest.get("status") == "COMPLETE" and
            manifest.get("pretraining_lock_sha256") == pretrain_sha and manifest.get("readout_lock_sha256") == readout_sha,
            "READOUT_MANIFEST_NOT_COMPLETE_OR_LOCKED")
    jobs = manifest.get("jobs", [])
    state_keys = [(x["split"], x["family"], x["seed"], x["lambda_pair"], x["epoch"]) for x in jobs]
    require(len(jobs) == 72 and set(state_keys) == expected_states(), "EXACT72_READOUT_JOBS_REQUIRED")
    rows, flat_rows, traces, provenance, costs, diagnostics = [], [], [], [], [], []
    for job in jobs:
        receipt = safe_receipt(root, job["receipt_file"], job["receipt_sha256"])
        require(receipt.get("schema") == "semantic_b0b7_p2s_readout_state_v1" and
                receipt.get("status") in ("COMPLETE", "COMPLETE_WITH_NONPRIMARY_RANK_NA", "PARTIAL_PRIMARY_UNSUPPORTED_RANK"), "READOUT_JOB_NOT_QUALIFIED")
        identity = {k: job[k] for k in ("split", "family", "seed", "lambda_pair", "epoch")}
        require(all(receipt["meta"][k] == v for k, v in identity.items()), "READOUT_JOB_IDENTITY_CHANGED")
        require(receipt.get("readout_lock_sha256") == readout_sha and receipt.get("study_lock_sha256") == pretrain_sha and
                receipt.get("no_selection") is True and receipt.get("all_lambdas_reported_without_selection") is True and
                receipt.get("primary_lambda") == PRIMARY_RIDGE and receipt.get("outer_test_accessed") is False and
                receipt.get("actual_new_teacher_or_scalar_fits") == 0, "READOUT_RECEIPT_SOURCE_OR_SELECTION_POLICY_CHANGED")
        if "source_checkpoint_sha256" in receipt["meta"]:
            require(receipt["meta"]["source_checkpoint_sha256"] == receipt["meta"]["checkpoint_sha256"], "READOUT_CHECKPOINT_ALIAS_CONFLICT")
        if "source_checkpoint_sha256" in job:
            require(job["source_checkpoint_sha256"] == receipt["meta"]["checkpoint_sha256"], "READOUT_MANIFEST_CHECKPOINT_CHANGED")
        require(job.get("input_sha256", receipt["meta"]["input_sha256"]) == receipt["meta"]["input_sha256"], "READOUT_MANIFEST_INPUT_CHANGED")
        records = receipt["rows"]
        expected = {(s, lam) for s in specs(job["family"]) for lam in RIDGES}
        actual = [(spec_key(r), float(r["ridge_lambda"])) for r in records]
        require(len(actual) == len(expected) and set(actual) == expected, "READOUT_HEAD_GRID_MISSING_DUPLICATE_OR_ADDED")
        fits = 0
        for original in records:
            row = {**original, **identity}
            require(row["status"] in (STATUS_FIT, STATUS_NA) and row.get("primary_lambda") == PRIMARY_RIDGE and
                    row.get("is_primary_lambda") is (row["ridge_lambda"] == PRIMARY_RIDGE), "READOUT_STATUS_OR_PRIMARY_POLICY_CHANGED")
            if row["status"] == STATUS_FIT:
                fits += 1
                require(row["DSS_coefficients"] == 2 * row["spec"]["main_pca"] + 8 + row["spec"]["factor_rank"], "READOUT_TRUE_COEFFICIENT_BUDGET_CHANGED")
                for metrics in (row["train_metrics"], row["evaluation_metrics"], row["anchor_evaluation_metrics"]):
                    for name in METRIC_COMPONENTS:
                        metric_value(metrics, name)
                require(row["risk_optimizer"]["validation_used_for_optimizer_or_lambda"] is False and
                        row["risk_optimizer"]["lambda_selected_using_validation"] is False, "READOUT_VALIDATION_SELECTION_FORBIDDEN")
                replay = row.get("saved_head_replay", {})
                require(all(replay.get(k) is True for k in ("actual_saved_head_replay", "training_risk_byte_hash_equal",
                    "evaluation_risk_byte_hash_equal", "scorer_components_exact")) and replay.get("max_abs_risk_error") == 0 and
                    replay.get("neural_forward") is False and replay.get("new_optimizer_fits") == 0, "READOUT_REAL_SAVED_HEAD_REPLAY_NOT_QUALIFIED")
            else:
                require(isinstance(row.get("NA_reason"), str) and row["NA_reason"] and row.get("evaluation_metrics") is None,
                        "RANK_FAILURE_MUST_REMAIN_EXPLICIT_NA")
            rows.append(row)
            flat_row = {**identity, **flatten(row["spec"], "spec."), "ridge_lambda": row["ridge_lambda"],
                        "is_primary_lambda": row["is_primary_lambda"], "status": row["status"], "NA_reason": row.get("NA_reason")}
            for field in ("train_metrics", "evaluation_metrics", "anchor_evaluation_metrics", "repair_vs_r0", "repair_vs_nointer", "repair_vs_B7",
                          "decomposition_scalar_sd", "main_coefficient_drift", "risk_optimizer", "frozen_extra_off_evaluation_metrics", "saved_head_replay"):
                if row.get(field) is not None:
                    flat_row.update(flatten(row[field], field + ".", excluded=("training_loss_curve",)))
            if row.get("design") is not None:
                flat_row.update(flatten({k: v for k, v in row["design"].items() if k != "blocks"}, "design."))
            flat_row["DSS_coefficients"] = row.get("DSS_coefficients")
            flat_rows.append(flat_row)
            if row.get("risk_optimizer") is not None:
                for point in row["risk_optimizer"].get("training_loss_curve", []):
                    traces.append({**identity, **flatten(row["spec"], "spec."), "ridge_lambda": row["ridge_lambda"], **flatten(point)})
        require(receipt["actual_Cox_fits"] == fits and receipt["NA_count"] == len(records) - fits and
                receipt["expected_fits"] == len(expected) and receipt["new_neural_fits"] == 0 and receipt["new_model_forwards"] == 0,
                "READOUT_ACTUAL_FIT_COUNTS_DISAGREE")
        require(receipt.get("actual_saved_head_replays") == fits and receipt.get("actual_Cox_fit_attempts") == fits,
                "READOUT_FIT_OR_REPLAY_LEDGER_INCOMPLETE")
        costs.append({**identity, **{k: receipt[k] for k in ("actual_Cox_fits", "NA_count", "expected_fits", "actual_PCA_fits", "elapsed_seconds", "new_neural_fits", "new_model_forwards")},
                      "actual_bgRidge": receipt.get("actual_bgRidge", receipt.get("actual_bgRidge_fits")),
                      "actual_Cox_fit_attempts": receipt.get("actual_Cox_fit_attempts"), "actual_saved_head_replays": receipt.get("actual_saved_head_replays")})
        provenance.append({**identity, "receipt_sha256": job["receipt_sha256"], "input_sha256": receipt["meta"]["input_sha256"],
                           "checkpoint_sha256": receipt["meta"]["checkpoint_sha256"]})
        diagnostics.append({**identity, "receipt_status": receipt["status"],
            **{k: receipt[k] for k in ("anchor", "main_map_receipts", "probe_receipt", "mechanism_source", "feature_extraction_compute") if k in receipt}})
    require(len(rows) == 2376 and sum(c["expected_fits"] for c in costs) == 2376, "READOUT2376_GRID_INCOMPLETE")
    return dict(rows=rows, flat_rows=flat_rows, traces=traces, provenance=provenance, costs=costs, diagnostics=diagnostics)


def crosscheck_state_sources(pretraining, readout):
    keys = ("split", "family", "seed", "lambda_pair")
    input_hashes = {tuple(row[k] for k in keys): row["input_sha256"] for row in pretraining["provenance"]}
    state_hashes = {tuple(row[k] for k in (*keys, "epoch")): row["checkpoint_sha256"] for row in pretraining["checkpoints"]}
    actual = {tuple(row[k] for k in (*keys, "epoch")): row for row in readout["provenance"]}
    require(len(input_hashes) == 36 and len(state_hashes) == len(actual) == 72 and set(actual) == set(state_hashes),
            "READOUT_STATES_DO_NOT_MATCH_FIXED_PRETRAINING_SCOPE")
    for identity, row in actual.items():
        require(row["checkpoint_sha256"] == state_hashes[identity] and row["input_sha256"] == input_hashes[identity[:-1]],
                "READOUT_CHECKPOINT_OR_INPUT_DIFFERS_FROM_PRETRAINING")
    return {"states_compared": 72, "checkpoint_bytes_hashes_equal": True, "training_input_hashes_equal": True,
            "model_or_array_deserialization": False}


def crosscheck_shared_anchors(diagnostics, child):
    bank = {}
    for row in diagnostics:
        anchor = row.get("anchor", {})
        locked = child["anchors"][row["split"]]
        require(anchor.get("artifact_sha256") == locked["sha256"] and anchor.get("receipt_sha256") == locked["receipt_sha256"] and
                anchor.get("new_teacher_or_scalar_fits") == 0 and anchor.get("full_pipeline_crossfit_for_training_offset") is True and
                anchor.get("own_labels_excluded_from_teacher_and_beta") is True and anchor.get("reused_original_teacher_Cox_fits") == 52 and
                anchor.get("reused_original_scalar_fits") == 4, "SHARED_R0_SOURCE_OR_LABEL_EXCLUSION_CHANGED")
        if row["split"] in bank:
            require(bank[row["split"]] == anchor, "SAME_PARTITION_R0_NOT_EXACTLY_SHARED")
        else:
            bank[row["split"]] = anchor
    require(set(bank) == set(SPLITS), "THREE_SHARED_R0_SOURCES_REQUIRED")
    return {"unique_original_anchor_partitions_reused": 3, "original_teacher_Cox_fits_referenced_not_new": 156,
            "original_scalar_fits_referenced_not_new": 12, "new_teacher_or_scalar_fits": 0}


def build_index(rows):
    indexed = {}
    for row in rows:
        key = config_key(row); identity = row["seed"], row["split"]
        require(identity not in indexed.setdefault(key, {}), "DUPLICATE_READOUT_CELL")
        indexed[key][identity] = row
    require(len(indexed) == 264, "EXPECTED264_FIXED_CONFIGURATIONS")
    require(all(set(x) == set(itertools.product(SEEDS, SPLITS)) for x in indexed.values()), "MISSING_NINE_CELL_CONFIG")
    return indexed


def validate_report_locks(study, child, design, *, study_sha, design_sha):
    require(study.get("schema") == "semantic_b0b7_p2s_lock_v1" and study.get("status") == "LOCKED_BEFORE_P2S_OPTIMIZER",
            "PRETRAINING_LOCK_NOT_PREOPTIMIZER")
    require(study.get("study_design") == design and
            study.get("frozen_sources", {}).get("configs/semantic_b0b7/p2s_design_precommit.json") == design_sha,
            "REPORT_DESIGN_NOT_BOUND_TO_PREOPTIMIZER_STUDY")
    require(child.get("schema") == "semantic_b0b7_p2s_readout_lock_v1" and child.get("status") == "LOCKED_BEFORE_P2S_DSS_FITS" and
            child.get("parent_study_lock_sha256") == study_sha and child.get("expected_feature_states") == 72 and
            child.get("expected_Cox_fits") == 2376 and child.get("primary_ridge_lambda") == PRIMARY_RIDGE and
            child.get("ridge_candidates") == list(RIDGES) and child.get("outer_evaluation") is False,
            "READOUT_CHILD_LOCK_OR_PRIMARY_PROTOCOL_CHANGED")
    require(set(child.get("anchors", {})) == set(SPLITS), "THREE_SAVED_INNER_ANCHORS_REQUIRED")
    require(all(child.get("readout_sources", {}).get(k, v) == v for k, v in study["frozen_sources"].items()),
            "READOUT_CHILD_MUST_NOT_REBIND_SCIENTIFIC_SOURCE")


def produce(pretrain_root, readout_root, pretrain_lock, pretrain_sha, readout_lock, readout_sha, design_path, design_sha, manifest_path, output):
    lock = read_json(pretrain_lock, pretrain_sha)
    child = read_json(readout_lock, readout_sha)
    design = read_json(design_path, design_sha)
    require(design.get("schema") == "semantic_b0b7_p2s_design_precommit_v1" and design["D"]["primary_ridge_lambda"] == PRIMARY_RIDGE and
            design["D"]["expected_total_Cox_fits"] == 2376 and design["A_C"]["new_optimizer_updates"] == 21600,
            "FULL_DIAGNOSTIC_DESIGN_CHANGED")
    validate_report_locks(lock, child, design, study_sha=pretrain_sha, design_sha=design_sha)
    pre = collect_pretraining(pretrain_root, lock, expected_lock_sha=pretrain_sha)
    readout = collect_readouts(readout_root, read_json(manifest_path), pretrain_sha=pretrain_sha, readout_sha=readout_sha)
    source_match = crosscheck_state_sources(pre, readout)
    anchors = crosscheck_shared_anchors(readout["diagnostics"], child)
    indexed = build_index(readout["rows"])
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    files = []
    aux_summaries, aux_details = auxiliary_comparisons(pre["checkpoints"])
    for stem, rows in (("01_A_C_checkpoints", pre["checkpoints"]), ("01_A_C_programs", pre["programs"]), ("01_A_C_strata", pre["strata"]),
                       ("02_A_C_paired_cells", aux_details), ("03_new_optimizer_epoch_history", pre["curves"]),
                       ("03_checkpoint_parameter_gradients", pre["gradients"]), ("04_all_2376_readout_cells", readout["flat_rows"]),
                       ("04_Cox_optimizer_traces", readout["traces"]), ("08_neural_costs", pre["receipts"]), ("08_readout_costs", readout["costs"])):
        files.extend(write_csv_parts(output, stem, rows))
    files.extend(write_json_parts(output, "02_A_C_paired_summaries", aux_summaries))
    aggregates = [{**config_dict(key), **aggregate_config(list(bank.values())), "primary_ridge_lambda": PRIMARY_RIDGE,
                   "is_primary_lambda": key[-1] == PRIMARY_RIDGE}
                  for key, bank in sorted(indexed.items())]
    files.extend(write_json_parts(output, "05_seed_micro_aggregates", aggregates))
    summaries, paired = [], []
    for name, terms in comparisons(indexed):
        summary, details = compare_terms(name, terms, indexed)
        summaries.append(summary); paired.extend(details)
    files.extend(write_json_parts(output, "06_fixed_comparison_summaries", summaries))
    files.extend(write_csv_parts(output, "06_fixed_comparison_cells", paired))
    files.extend(write_json_parts(output, "07_receipt_provenance", pre["provenance"] + readout["provenance"]))
    files.extend(write_json_parts(output, "07_shared_design_and_probe_diagnostics", readout["diagnostics"]))
    fitted = sum(r["status"] == STATUS_FIT for r in readout["rows"])
    costs = {"new_neural_trajectories": 36, "fresh_neural_initializations": 18, "continued_formal400_trajectories": 18,
        "new_optimizer_updates": len(pre["curves"]), "reused_old_optimizer_updates_not_new": 7200,
        "new_snapshots": 54, "reused_original400_snapshots": 18, "actual_checkpoint_replays": 72,
        "expected_readout_states": 72, "expected_MAIN_Cox_fits": 1944, "expected_ATTN_Cox_fits": 432,
        "actual_Cox_fits": fitted, "explicit_NA_head_cells": len(readout["rows"]) - fitted,
        "actual_PCA_fits": sum(c["actual_PCA_fits"] for c in readout["costs"]),
        "actual_background_Ridge_fits": sum(c["actual_bgRidge"] for c in readout["costs"] if c["actual_bgRidge"] is not None),
        "new_teacher_fits": 0, "reporter_model_forwards": 0, "reporter_optimizer_fits": 0,
        "reporter_patient_risk_feature_endpoint_or_weight_arrays_read": 0,
        "new_neural_elapsed_seconds_sum": math.fsum(c["elapsed_seconds"] for c in pre["receipts"]),
        "readout_elapsed_seconds_sum": math.fsum(c["elapsed_seconds"] for c in readout["costs"])}
    costs.update(anchors)
    extraction = [r["feature_extraction_compute"] for r in readout["diagnostics"] if "feature_extraction_compute" in r]
    require(len(extraction) == 72 and all(r["frozen_encoder_feature_extractions"] == 1 for r in extraction),
            "ACTUAL_FROZEN_EXTRACTION_COST_RECEIPTS_REQUIRED")
    costs["producer_frozen_extraction_receipts_available"] = len(extraction)
    costs["producer_frozen_encoder_extractions"] = sum(r["frozen_encoder_feature_extractions"] for r in extraction)
    costs["producer_frozen_extraction_seconds_sum"] = math.fsum(r["elapsed_seconds"] for r in extraction)
    costs["timing_scope"] = "sum_of_saved_worker_stage_times_not_pipeline_wall_clock; frozen_extraction_excludes_checkpoint_load"
    files.append(write_json(output / "08_COMPUTE_COST.json", costs))
    status = "COMPLETE_FIXED_GRID" if fitted == 2376 else "COMPLETE_FIXED_GRID_WITH_EXPLICIT_NA"
    decision = {"schema": "semantic_p2s_reporting_completion_v1", "status": status, "scientific_mechanism_decision": "NOT_AUTOMATICALLY_ASSIGNED",
        "scope": "post_P2_outer1_inner_development_diagnostic_not_independent_confirmation",
        "primary_ridge_lambda": PRIMARY_RIDGE, "all_ridge_lambdas_reported": list(RIDGES), "epoch_pair_weight_or_ridge_winner_selected": False,
        "CIs_p_values_or_bootstrap_computed": False, "full_trained_rank4_architecture_tested": False,
        "factor4_scope": "original_two_frozen_learned_axes_plus_two_fixed_orthogonal_complement_probe_axes",
        "rank2_scope": "same_old_algebra_recomputed_in_float64_not_old_float32_bitwise_replay",
        "PCA16_scope": "changes_main_effect_information_capacity_not_a_pure_cross_interaction_change",
        "negative_fixed_probe_excludes_all_learned_rank4_models": False, "no_more_model_search_authorized_by_this_report": True,
        "configurations": len(indexed), "fixed_comparisons": len(summaries), "costs": costs}
    files.append(write_json(output / "09_COMPLETION_NOT_SCIENTIFIC_DECISION.json", decision))
    text = """# P2S 固定范围训练充分性与冻结读出诊断

本报告只读取已保存的 JSON 标量收据，不加载预测数组、终点、特征或权重，不执行前向、训练或重新评分。所有固定预算与正则候选均保留，主正则为预先固定的 1；没有从本轮内层结果选择新赢家。

预训练包含18条旧400轮续训和18条新轨迹，共21600次新增优化更新。旧轨迹的前7200次历史更新通过原收据SHA引用，不计为新增训练。72次固定状态回放包括54份新快照和18份原400轮引用。A/C分别比较Huber、宏程序Pearson与匹配检索gap，不跨不同λpair比较加权总损失。

生存诊断的每个seed先累加三个内层分区的折内配对分子和分母，再对三个seed等权平均。逐折配对与条件增量的差分另列；不拼接跨模型风险形成患者对。NA保留其原因，不填零。所有2376个计划头位置、实际拟合成本及优化轨迹均保留。

四轴版本只是原两个已学习轴加固定正交补探针，不是训练过的rank4模型；阴性结果不能排除所有rank4架构。P/G从8到16也改变主效应容量，不能将其差值全部归于交互。B7保留同容量可分主效应；冻结关闭交互属于依赖干预，与真实无交互重拟合分开。主效应系数漂移与机制项的分解用于解释风险变化，不是因果证明。

本轮使用已经参与开发的同一队列及内层分区，属于最后一次限域诊断，不构成独立验证。完成元数据不自动宣告机制成立，也不证明800轮已经收敛。科学决策需要结合全部seed、分区、预算方向与机制诊断另行作出。
"""
    files.append(write_file(output / "RESULTS.md", text.encode("utf-8")))
    index = {"schema": "semantic_p2s_receipt_report_index_v1", "status": status, "files": files,
        "source_sha256": sha(__file__), "pretraining_lock_sha256": pretrain_sha, "readout_lock_sha256": readout_sha,
        "design_precommit_sha256": design_sha, "readout_manifest_sha256": sha(manifest_path),
        "frozen_scientific_sources": lock["frozen_sources"], "readout_source_closure": child.get("readout_sources", child.get("frozen_sources", {})),
        "parent_P2_master_sha256": lock["parent_master_sha256"], "parent_git_commit": lock.get("parent_git_commit"),
        "outer_results_loaded": False, "new_fits_or_forwards": 0, "self_hash_cycle": False}
    index["actual_pretraining_readout_state_crosscheck"] = source_match
    write_json(output / "INDEX.json", index)
    return {"status": status, "output_files_excluding_index": len(files), "actual_Cox_fits": fitted,
            "explicit_NA_head_cells": 2376 - fitted, "new_reporter_fits_or_forwards": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pretrain-root", "readout-root", "pretraining-lock", "expected-pretraining-lock-sha256", "readout-lock", "expected-readout-lock-sha256",
                 "design-precommit", "expected-design-sha256", "readout-manifest", "output-newdir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--completed-diagnostics-authorized", action="store_true")
    args = parser.parse_args()
    require(args.completed_diagnostics_authorized, "COMPLETED_RECEIPT_REPORT_AUTHORIZATION_REQUIRED")
    result = produce(args.pretrain_root, args.readout_root, args.pretraining_lock, args.expected_pretraining_lock_sha256,
                     args.readout_lock, args.expected_readout_lock_sha256, args.design_precommit, args.expected_design_sha256,
                     args.readout_manifest, args.output_newdir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

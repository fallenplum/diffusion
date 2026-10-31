"""Independent post-save P2S Z/coefficient audit, only after all 72 jobs.

No producer predictor/scorer, neural implementation, raw H5/gene matrix or
optimizer is invoked. Passive records plus stored designs reconstruct o+Z beta;
independent NumPy risk-set/pair loops and the real pinned official API check it.
This does not independently repeat the raw-input-to-feature extraction pipeline.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import sys

STARTUP_THREADS = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_semantic_b0b7_p2 as oracle

HELPER = "scripts/verify_semantic_b0b7_p2.py"
HELPER_SHA = "e27bc68ce9c0d4e384c2befe459603cee2003c2456789c3ab43bdbdc55e0d143"
SPLITS = tuple("outer1_inner" + str(i) for i in range(3))
SEEDS = (31, 47, 83)
FAMILIES = ("MAIN", "ATTN_MATCHED")
PAIRS = (.1, .03)
EPOCHS = (400, 800)
LAMBDAS = (.1, 1., 10.)
MAX_BYTES = 2 * 1024 ** 2
MAX_CSV_LINES = 20_000
P2_SHA = "b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
METRICS = ("Harrell", "Uno60")
FIT = "FITTED_AND_REPLAYED"
NA = "NA_NO_SUPPORTED_RANK"


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path, expected=None):
    require(Path(path).suffix == ".json", "JSON_RECEIPT_REQUIRED")
    if expected is not None:
        require(oracle.sha(path) == expected, "JSON_RECEIPT_SHA_CHANGED")
    return oracle.read(path)


def state_key(row):
    return row["split"], row["family"], int(row["seed"]), float(row["lambda_pair"]), int(row["epoch"])


def checkpoint_hash(row):
    """Canonical producer field; legacy alias may coexist only at equal value."""
    value = row.get("checkpoint_sha256", row.get("source_checkpoint_sha256"))
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "CHECKPOINT_HASH_REQUIRED")
    require("checkpoint_sha256" not in row or "source_checkpoint_sha256" not in row or
            row["checkpoint_sha256"] == row["source_checkpoint_sha256"], "CHECKPOINT_HASH_ALIASES_CONFLICT")
    return value


def validate_manifest(manifest, study_sha, child_sha):
    require(manifest.get("schema") == "semantic_b0b7_p2s_readout_manifest_v1" and manifest.get("status") == "COMPLETE" and
            manifest.get("pretraining_lock_sha256") == study_sha and manifest.get("readout_lock_sha256") == child_sha,
            "ALL72_COMPLETED_READOUT_MANIFEST_REQUIRED")
    jobs = manifest.get("jobs", [])
    expected = set(itertools.product(SPLITS, FAMILIES, SEEDS, PAIRS, EPOCHS))
    keys = [state_key(j) for j in jobs]
    require(len(keys) == 72 and len(set(keys)) == 72 and set(keys) == expected, "EXACT72_STATES_REQUIRED_NO_MISSING_DUPLICATE_OR_OUTER")
    return jobs


def validate_study_state_bindings(jobs, study):
    cells = study.get("cells", [])
    expected = set(itertools.product(SPLITS, FAMILIES, SEEDS, PAIRS))
    bank = {(c["split"], c["family"], c["seed"], c["lambda_pair"]): c for c in cells}
    require(len(cells) == len(bank) == 36 and set(bank) == expected, "INDEPENDENT_PRETRAINING36_SOURCE_SCOPE_CHANGED")
    for job in jobs:
        source = bank[state_key(job)[:-1]]
        require(job["input_sha256"] == source["input_sha256"], "MANIFEST_INPUT_NOT_FROM_LOCKED_PRETRAINING_CELL")
        if job["lambda_pair"] == .1 and job["epoch"] == 400:
            require(checkpoint_hash(job) == source["parent_state400_sha256"], "MANIFEST_ORIGINAL400_STATE_SOURCE_CHANGED")
        else:
            checkpoint_hash(job)


class PassiveP2SUnpickler(oracle.PassiveUnpickler):
    def find_class(self, module, name):
        if module == "src.semantic_b0b7.p2s_survival" and name in {
                "ReadoutSpec", "FrozenFactorProbe", "FrozenSourceViews", "MainProjection", "ReadoutDesign", "DiagnosticCox", "DiagnosticCell"}:
            return oracle.PassiveRecord
        if module == "scripts.verify_semantic_b0b7_p2" and name == "PassiveRecord":
            return oracle.PassiveRecord
        return super().find_class(module, name)


def passive_gzip(path, expected_sha):
    require(oracle.sha(path) == expected_sha, "GZIP_ARTIFACT_SHA_CHANGED_BEFORE_DECODE")
    with gzip.open(path, "rb") as stream:
        payload = PassiveP2SUnpickler(stream).load()
    require(oracle.sha(path) == expected_sha, "GZIP_ARTIFACT_CHANGED_DURING_DECODE")
    return payload


def specs(family):
    require(family in FAMILIES, "UNLOCKED_FAMILY")
    out = []
    for d in (8, 16):
        out.append((d, "SHARED", 0, "NO_INTER"))
        if family == "MAIN":
            out.extend((d, arm, r, mode) for arm in ("B4", "B5") for r in (2, 4) for mode in ("INTER", "B7"))
        else:
            out.append((d, "ATTN_MATCHED", 2, "INTER"))
    return out


def row_spec(row):
    s = row["spec"]
    return int(s["main_pca"]), s["base_arm"], int(s["factor_rank"]), s["mode"]


def config_key(row):
    return row["family"], row["lambda_pair"], row["epoch"], *row_spec(row), row["ridge_lambda"]


def finite_array(value, shape, name):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float64") and value.shape == shape and np.isfinite(value).all(),
            "POSTSAVE_FLOAT64_ARRAY_INVALID:" + name)
    return value


def metrics_check(actual, expected):
    require(isinstance(expected, dict), "FITTED_HEAD_MISSING_METRICS")
    error = 0.
    for key in ("Harrell", "Harrell_num", "Harrell_den", "Uno60", "Uno_num", "Uno_den"):
        a, b = actual[key], expected.get(key)
        require((a is None) == (b is None), "INDEPENDENT_METRIC_SUPPORT_MISMATCH:" + key)
        if a is not None:
            error = max(error, abs(float(a) - float(b)))
            require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-10), "INDEPENDENT_METRIC_COMPONENT_MISMATCH:" + key)
    require(actual["Uno60_valid"] is expected.get("Uno60_valid"), "INDEPENDENT_UNO_VALIDITY_MISMATCH")
    return error


def assert_na_row(row, cells_by_spec):
    require(isinstance(row.get("NA_reason"), str) and row["NA_reason"] and row.get("DSS_coefficients") is None and
            all(row.get(k) is None for k in ("train_metrics", "evaluation_metrics", "risk_optimizer")),
            "NA_MUST_NOT_CONTAIN_FABRICATED_ZERO_METRICS_OR_COEFFICIENTS")
    require(str(float(row["ridge_lambda"])) not in cells_by_spec.get(row["spec_name"], {}), "NA_HEAD_HAS_HIDDEN_FITTED_CELL")


def audit_head(payload, row):
    """Direct saved Z @ coefficient, never a model method or producer scorer."""
    name, lam = row["spec_name"], float(row["ridge_lambda"])
    cell = payload["cells_by_spec"][name][str(lam)]
    require(payload["cells"].get(name + "/lambda" + str(lam)) is cell, "POSTSAVE_FLAT_AND_SHARED_HEAD_ALIASES_DISAGREE")
    z = payload["design_matrices"][name]
    t, e = oracle.endpoints(payload["train_time"], payload["train_event"])
    vt, ve = oracle.endpoints(payload["evaluation_time"], payload["evaluation_event"])
    d, arm, rank, mode = row_spec(row); width = 2 * d + 8 + rank
    require(name == f"D{d}_{arm}_R{rank}_{mode}" and
            {k: getattr(cell.design.spec, k) for k in ("main_pca", "base_arm", "factor_rank", "mode")} == row["spec"] and
            payload["designs"].get(name) is cell.design and cell.design.main is payload["main_maps"].get(d),
            "POSTSAVE_DESIGN_IDENTITY_OR_COMMON_PCA_MAP_DIFFERS")
    require(row["DSS_coefficients"] == width and cell.model.ridge_lambda == lam, "POSTSAVE_HEAD_SPEC_OR_LAMBDA_CHANGED")
    x = finite_array(z["train"], (len(t), width), "Z_train")
    xx = finite_array(z["evaluation"], (len(vt), width), "Z_evaluation")
    shared = payload["design_matrices"][f"D{d}_SHARED_R0_NO_INTER"]
    main_width = 2 * d + 8
    require(oracle.array_sha(x[:, :main_width]) == oracle.array_sha(shared["train"]) and
            oracle.array_sha(xx[:, :main_width]) == oracle.array_sha(shared["evaluation"]),
            "POSTSAVE_MAIN_EFFECT_COORDINATES_NOT_EXACTLY_SHARED_WITH_NOINTER")
    w = finite_array(cell.model.coefficient, (width,), "coefficient")
    o = finite_array(payload["offsets"]["train"], (len(t),), "training_offset")
    oo = finite_array(payload["offsets"]["evaluation"], (len(vt),), "evaluation_offset")
    require(np.array_equal(o, payload["training_offset"]) and np.array_equal(oo, payload["evaluation_offset"]), "OFFSET_ALIASES_DISAGREE")
    train, evaluation = o + x @ w, oo + xx @ w
    saved_train = finite_array(cell.training_risk, train.shape, "saved_training_risk")
    saved_eval = finite_array(cell.evaluation_risk, evaluation.shape, "saved_evaluation_risk")
    train_exact = oracle.array_sha(train) == oracle.array_sha(saved_train)
    eval_exact = oracle.array_sha(evaluation) == oracle.array_sha(saved_eval)
    error = max(float(np.max(np.abs(train - saved_train))), float(np.max(np.abs(evaluation - saved_eval))))
    require(train_exact and eval_exact, "INDEPENDENT_OFFSET_PLUS_Z_COEF_RISK_BYTES_DIFFER")
    fitted = row["risk_optimizer"]
    require(fitted["ridge_lambda"] == lam and fitted["gradient_tolerance"] == 2e-5 and
            fitted["optimized_coefficients"] == width and fitted["active_coefficients"] == width,
            "POSTSAVE_CONVEX_OPTIMIZER_CONTRACT_CHANGED")
    for key, value in (("fit_design_sha256", x), ("fit_time_sha256", np.asarray(payload["train_time"])),
                       ("fit_event_sha256", np.asarray(payload["train_event"])), ("fit_offset_sha256", o)):
        require(fitted[key] == oracle.array_sha(value), "POSTSAVE_TRAINING_SOURCE_HASH_DIFFERS:" + key)
    require(cell.receipt["training_risk_sha256"] == oracle.array_sha(train) and cell.receipt["evaluation_risk_sha256"] == oracle.array_sha(evaluation),
            "POSTSAVE_RISK_RECEIPT_HASH_DIFFERS")
    train_metrics = oracle.independent_metrics(t, e, t, e, train)
    eval_metrics = oracle.independent_metrics(t, e, vt, ve, evaluation)
    metric_error = max(metrics_check(train_metrics, row["train_metrics"]), metrics_check(eval_metrics, row["evaluation_metrics"]))
    objective, gradient = oracle.cox_objective(x, w, o, t, e, lam)
    gradient_max = float(np.max(np.abs(gradient)))
    objective_error, gradient_error = abs(objective - fitted["objective"]), abs(gradient_max - fitted["gradient_max"])
    require(objective_error <= 1e-10 and gradient_error <= 1e-9 and gradient_max <= fitted["gradient_tolerance"] + 1e-9,
            "INDEPENDENT_EFRON_RIDGE_OBJECTIVE_OR_GRADIENT_MISMATCH")
    output = {"status": "FITTED_AUDITED", "spec_name": name, "ridge_lambda": lam, "DSS_coefficients": width,
        "training_risk_bytes_equal": train_exact, "evaluation_risk_bytes_equal": eval_exact, "risk_max_abs_error": error,
        "independent_metric_component_max_abs_error": metric_error, "independent_objective_abs_error": objective_error,
        "independent_gradient_max": gradient_max, "gradient_max_abs_error": gradient_error,
        "design_train_sha256": oracle.array_sha(x), "design_evaluation_sha256": oracle.array_sha(xx),
        "coefficient_bytes_sha256": oracle.array_sha(w), "risk_eval_sha256": oracle.array_sha(evaluation),
        "training_patients": len(t), "evaluation_patients": len(vt), "training_events": int(e.sum()), "evaluation_events": int(ve.sum())}
    output.update({"evaluation_" + k: eval_metrics[k] for k in ("Harrell", "Harrell_num", "Harrell_den", "Uno60", "Uno_num", "Uno_den", "Uno60_valid")})
    case = {"train_time": t, "train_event": e, "time": vt, "event": ve, "risk": evaluation, "expected": eval_metrics}
    return output, case, eval_metrics


def audit_payload(payload, receipt, identity):
    require(payload.get("schema") == "semantic_p2s_private_shared_design_cells_v1" and
            all(payload["meta"][k] == identity[k] for k in ("split", "family", "seed", "lambda_pair", "epoch")), "POSTSAVE_ARTIFACT_SCOPE_CHANGED")
    require(payload["meta"]["input_sha256"] == receipt["meta"]["input_sha256"] and
            payload["meta"]["checkpoint_sha256"] == receipt["meta"]["checkpoint_sha256"], "POSTSAVE_ARTIFACT_ENCODER_BINDING_CHANGED")
    rows = receipt["rows"]
    expected = {(s, lam) for s in specs(identity["family"]) for lam in LAMBDAS}
    keys = [(row_spec(r), float(r["ridge_lambda"])) for r in rows]
    require(len(keys) == len(expected) and set(keys) == expected, "POSTSAVE_EXACT_HEAD_GRID_CHANGED")
    checks, cases, metrics, fitted_keys = [], [], [], set()
    for row in rows:
        require(row["status"] in (FIT, NA), "POSTSAVE_UNKNOWN_HEAD_STATUS")
        if row["status"] == NA:
            assert_na_row(row, payload["cells_by_spec"])
            check = {"status": NA, "spec_name": row["spec_name"], "ridge_lambda": row["ridge_lambda"], "NA_reason": row["NA_reason"]}
            actual_metrics = None
        else:
            check, case, actual_metrics = audit_head(payload, row)
            cases.append(case); fitted_keys.add((row["spec_name"], str(float(row["ridge_lambda"]))))
        checks.append({**identity, **{k: row["spec"][k] for k in ("main_pca", "base_arm", "factor_rank", "mode")}, **check})
        metrics.append({**identity, "spec": row["spec"], "ridge_lambda": row["ridge_lambda"], "status": row["status"], "metrics": actual_metrics})
    stored = {(name, lam) for name, values in payload["cells_by_spec"].items() for lam in values}
    require(stored == fitted_keys and receipt["actual_Cox_fits"] == len(cases) and receipt["actual_saved_head_replays"] == len(cases) and
            receipt["NA_count"] + len(cases) == len(expected), "POSTSAVE_FIT_NA_OR_HIDDEN_HEAD_LEDGER_CHANGED")
    return checks, cases, metrics


def audit_anchor(payload, receipt, locked):
    anchor = receipt["anchor"]
    require(anchor["artifact_sha256"] == locked["sha256"] and anchor["receipt_sha256"] == locked["receipt_sha256"] and
            anchor["new_teacher_or_scalar_fits"] == 0 and anchor["full_pipeline_crossfit_for_training_offset"] is True and
            anchor["own_labels_excluded_from_teacher_and_beta"] is True, "INDEPENDENT_SHARED_R0_SOURCE_CHANGED")
    require(oracle.array_sha(payload["offsets"]["train"]) == anchor["training_offset_sha256"] and
            oracle.array_sha(payload["offsets"]["evaluation"]) == anchor["evaluation_offset_sha256"],
            "POSTSAVE_OFFSETS_DIFFER_FROM_LOCKED_SHARED_R0")
    return {k: anchor[k] for k in ("artifact_sha256", "receipt_sha256", "training_offset_sha256", "evaluation_offset_sha256")}


def config_from_summary(row):
    return row["family"], row["lambda_pair"], row["epoch"], row["main_pca"], row["base_arm"], row["factor_rank"], row["mode"], row["ridge_lambda"]


def independent_aggregates(rows):
    grouped = {}
    for row in rows:
        bank = grouped.setdefault(config_key(row), {})
        identity = row["seed"], row["split"]
        require(identity not in bank, "DUPLICATE_INDEPENDENT_CONFIG_CELL")
        bank[identity] = row
    require(len(grouped) == 264, "INDEPENDENT264_CONFIGURATIONS_REQUIRED")
    output = {}
    for key, bank in grouped.items():
        require(set(bank) == set(itertools.product(SEEDS, SPLITS)), "INDEPENDENT_NINE_CELL_SUPPORT_REQUIRED")
        seeds = {}
        for seed in SEEDS:
            seeds[seed] = {}
            for name, nk, dk in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
                vals = [bank[(seed, split)]["metrics"] for split in SPLITS]
                good = all(v is not None and v[name] is not None and v[dk] is not None and v[dk] > 0 for v in vals)
                num = math.fsum(v[nk] for v in vals) if good else None
                den = math.fsum(v[dk] for v in vals) if good else None
                seeds[seed][name] = {"value": num / den if good else None, "num": num, "den": den}
        record = {"seed_results": seeds}
        for name in METRICS:
            vals = [seeds[s][name]["value"] for s in SEEDS]
            record[name] = math.fsum(vals) / 3 if all(v is not None for v in vals) else None
        output[key] = record
    return output


def compare_reports(directory, verified_rows):
    root = Path(directory)
    index = read_json(root / "INDEX.json")
    require(index["outer_results_loaded"] is False and index["new_fits_or_forwards"] == 0, "REPORT_SCOPE_CHANGED")
    declared = {r["file"]: r for r in index["files"]}
    parts = sorted(n for n in declared if n.startswith("05_seed_micro_aggregates") and n.endswith(".json"))
    require(parts, "REPORT_AGGREGATES_MISSING")
    rows = []
    for part in parts:
        rows.extend(read_json(oracle.checked(root, part, declared[part]["sha256"])))
    summaries = independent_aggregates(verified_rows)
    require(len(rows) == 264 and len({config_from_summary(r) for r in rows}) == 264 and
            {config_from_summary(r) for r in rows} == set(summaries), "REPORT264_CONFIG_SET_CHANGED")
    checked = 0
    for row in rows:
        independent = summaries[config_from_summary(row)]
        for name in METRICS:
            require(oracle.close(row[name], independent[name]), "REPORT_SEED_MICRO_AGGREGATE_MISMATCH")
        by_seed = {x["seed"]: x for x in row["seed_results"]}
        require(set(by_seed) == set(SEEDS), "REPORT_SEED_SET_CHANGED")
        for seed in SEEDS:
            for name in METRICS:
                for key in ("value", "num", "den"):
                    require(oracle.close(by_seed[seed][name][key], independent["seed_results"][seed][name][key]), "REPORT_FOLD_MICRO_COMPONENT_MISMATCH")
        checked += 1
    return {"status": "PASS_264_CONFIGURATION_AGGREGATE_ARITHMETIC", "configuration_summaries_checked": checked,
            "all1104_contrast_set_independently_rederived": False, "contrast_tables_independently_recomputed": 0,
            "aggregation": "within_partition_pair_components_then_seed_micro_then_equal_three_seed_mean"}


def write_csv_parts(directory, rows):
    fields = sorted({k for r in rows for k in r})
    header_io = io.StringIO(newline=""); csv.writer(header_io, lineterminator="\n").writerow(fields)
    header = header_io.getvalue().encode()
    chunks, current, size = [], [], len(header)
    for row in rows:
        stream = io.StringIO(newline=""); csv.writer(stream, lineterminator="\n").writerow([row.get(k) for k in fields])
        raw = stream.getvalue().encode()
        require(len(raw) + len(header) <= MAX_BYTES, "INDEPENDENT_CSV_ROW_TOO_LARGE")
        if current and (len(current) + 2 > MAX_CSV_LINES or size + len(raw) > MAX_BYTES):
            chunks.append(header + b"".join(current)); current, size = [], len(header)
        current.append(raw); size += len(raw)
    if current: chunks.append(header + b"".join(current))
    files = []
    for i, raw in enumerate(chunks):
        path = Path(directory) / ("INDEPENDENT_HEAD_CHECKS_%02d.csv" % (i + 1))
        with path.open("xb") as stream: stream.write(raw)
        files.append({"file": path.name, "sha256": oracle.sha(path), "bytes": len(raw)})
    return files


def public_json(path, value):
    from scripts import bgb_en_publication as guard
    guard.validate_public_json(value)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= MAX_BYTES and not any(p.search(raw.decode()) for p in guard.PATTERNS.values()), "INDEPENDENT_PUBLIC_RULE_FAILED")
    with Path(path).open("xb") as stream: stream.write(raw)
    return {"file": Path(path).name, "bytes": len(raw), "sha256": oracle.sha(path)}


def verify_run(root, manifest_path, study_path, study_sha, child_path, child_sha, output, *, sksurv_target, report_dir=None, authorized=False):
    require(authorized, "ALL72_COMPLETED_AUDIT_AUTHORIZATION_REQUIRED")
    require(STARTUP_THREADS == {k: "2" for k in STARTUP_THREADS}, "INDEPENDENT_STARTUP_BLAS2_REQUIRED")
    require(oracle.sha(ROOT / HELPER) == HELPER_SHA, "PINNED_INDEPENDENT_ARITHMETIC_HELPER_CHANGED")
    study, child = read_json(study_path, study_sha), read_json(child_path, child_sha)
    require(study["schema"] == "semantic_b0b7_p2s_lock_v1" and study["parent_master_sha256"] == P2_SHA and
            study["status"] == "LOCKED_BEFORE_P2S_OPTIMIZER" and child["schema"] == "semantic_b0b7_p2s_readout_lock_v1" and
            child["parent_study_lock_sha256"] == study_sha and child["status"] == "LOCKED_BEFORE_P2S_DSS_FITS" and
            child["expected_feature_states"] == 72 and child["expected_Cox_fits"] == 2376 and
            child["primary_ridge_lambda"] == 1. and child["ridge_candidates"] == list(LAMBDAS) and
            set(child["anchors"]) == set(SPLITS) and child["outer_evaluation"] is False, "INDEPENDENT_STUDY_CHILD_SCOPE_CHANGED")
    manifest = read_json(manifest_path)
    jobs = validate_manifest(manifest, study_sha, child_sha)
    validate_study_state_bindings(jobs, study)
    checks, cases, metrics, job_checks = [], [], [], []
    anchors = {}
    for job in jobs:
        receipt_path = oracle.checked(root, job["receipt_file"], job["receipt_sha256"])
        receipt = read_json(receipt_path)
        identity = {k: job[k] for k in ("split", "family", "seed", "lambda_pair", "epoch")}
        require(receipt.get("schema") == "semantic_b0b7_p2s_readout_state_v1" and
                receipt["status"] in ("COMPLETE", "COMPLETE_WITH_NONPRIMARY_RANK_NA", "PARTIAL_PRIMARY_UNSUPPORTED_RANK") and
                receipt["readout_lock_sha256"] == child_sha and receipt["study_lock_sha256"] == study_sha and
                all(receipt["meta"][k] == v for k, v in identity.items()), "INDEPENDENT_JOB_RECEIPT_SCOPE_CHANGED")
        require(checkpoint_hash(receipt["meta"]) == checkpoint_hash(job) and
                receipt["meta"]["input_sha256"] == job["input_sha256"], "INDEPENDENT_MANIFEST_STATE_HASH_CHANGED")
        artifact_relative = str(Path(job["receipt_file"]).with_name("READOUT_MODELS.private.pkl.gz")).replace("\\", "/")
        artifact = oracle.checked(root, artifact_relative, receipt["artifact_sha256"])
        payload = passive_gzip(artifact, receipt["artifact_sha256"])
        training_time, _ = oracle.endpoints(payload["train_time"], payload["train_event"])
        evaluation_time, _ = oracle.endpoints(payload["evaluation_time"], payload["evaluation_event"])
        require(len(training_time) + len(evaluation_time) == 409, "POSTSAVE_DSS_SCOPE_MUST_ONLY_CONTAIN_OUTER1_TRAINING409")
        anchor = audit_anchor(payload, receipt, child["anchors"][identity["split"]])
        if identity["split"] in anchors:
            require(anchor == anchors[identity["split"]], "PARTITION_SHARED_R0_CHANGED_BETWEEN_ENCODER_STATES")
        else:
            anchors[identity["split"]] = anchor
        heads, official, numerical = audit_payload(payload, receipt, identity)
        checks.extend(heads); cases.extend(official); metrics.extend(numerical)
        job_checks.append({**identity, "receipt_sha256": job["receipt_sha256"], "artifact_sha256": receipt["artifact_sha256"],
            "positions": len(heads), "fitted_heads_audited": len(official), "rank_NA_positions": len(heads) - len(official),
            "passive_decode_no_model_import": True})
        del payload
    require(len(job_checks) == 72 and len(checks) == 2376 and len(cases) + sum(r["status"] == NA for r in checks) == 2376,
            "INDEPENDENT72_2376_LEDGER_INCOMPLETE")
    api_cases = [c for c in cases if c["expected"]["Harrell"] is not None]
    official_batches = [oracle.official_oracle(api_cases[i:i + 256], sksurv_target) for i in range(0, len(api_cases), 256)]
    official_summary = {"status": "PASS_REAL_PINNED_OFFICIAL_API" if official_batches else "NO_COMPARABLE_PAIR_SUPPORTED_OFFICIAL_CASES",
        "package_version": "0.23.1" if official_batches else None, "head_cases": len(api_cases), "batches": len(official_batches),
        "Harrell_NA_by_saved_comparable_pair_support": len(cases) - len(api_cases),
        "actual_API_calls": sum(r["actual_API_calls"] for r in official_batches),
        "Harrell_max_abs_error": max((r["Harrell_max_abs_error"] for r in official_batches), default=None),
        "Uno60_max_abs_error": max((r["Uno60_max_abs_error"] for r in official_batches), default=None),
        "Uno_NA_by_prelocked_support": sum(r["Uno_NA_by_prelocked_support"] for r in official_batches)}
    require(official_summary["actual_API_calls"] == sum(1 + int(c["expected"]["Uno60_valid"] and c["expected"]["Uno60"] is not None) for c in api_cases),
            "OFFICIAL_API_CALL_COUNT_CHANGED")
    if report_dir is not None:
        report_index = read_json(Path(report_dir) / "INDEX.json")
        require(report_index["pretraining_lock_sha256"] == study_sha and report_index["readout_lock_sha256"] == child_sha and
                report_index["readout_manifest_sha256"] == oracle.sha(manifest_path), "REPORT_INDEX_SOURCE_BINDING_CHANGED")
        expected_reporter = child.get("readout_sources", {}).get("scripts/semantic_b0b7_p2s_report.py")
        if expected_reporter is not None:
            require(report_index["source_sha256"] == expected_reporter, "REPORTER_SOURCE_DIFFERS_FROM_READOUT_CHILD_LOCK")
        report = compare_reports(report_dir, metrics)
    else:
        report = {"status": "NOT_REQUESTED_NO_REPORT_DIR"}
    target = Path(output); target.mkdir(parents=True, exist_ok=False)
    files = write_csv_parts(target, checks)
    files.append(public_json(target / "INDEPENDENT_72_JOB_CHECKS.json", job_checks))
    fitted = [r for r in checks if r["status"] == "FITTED_AUDITED"]
    status = ("ALL_HEADS_EXPLICIT_RANK_NA_NO_NUMERICAL_MODEL_QUALIFICATION" if not cases else
              "PASS_WITH_EXPLICIT_RANK_NA" if len(cases) < 2376 else "PASS_POSTSAVE_Z_COEFFICIENT_METRIC_AND_OBJECTIVE_AUDIT")
    summary = {"schema": "semantic_p2s_independent_postsave_Z_coefficient_audit_v1", "status": status,
        "readout_jobs": 72, "registered_head_positions": 2376, "fitted_heads_audited": len(cases), "explicit_rank_NA": 2376 - len(cases),
        "unique_exact_shared_r0_partitions": len(anchors),
        "official_API": official_summary, "report_arithmetic": report, "files": files,
        "max_risk_abs_error": max((r["risk_max_abs_error"] for r in fitted), default=None),
        "max_metric_component_abs_error": max((r["independent_metric_component_max_abs_error"] for r in fitted), default=None),
        "max_objective_abs_error": max((r["independent_objective_abs_error"] for r in fitted), default=None),
        "max_gradient_max_abs_error": max((r["gradient_max_abs_error"] for r in fitted), default=None),
        "verifier_source_sha256": oracle.sha(__file__), "independent_helper_source_sha256": HELPER_SHA,
        "study_lock_sha256": study_sha, "readout_lock_sha256": child_sha, "manifest_sha256": oracle.sha(manifest_path),
        "scope": "saved_design_matrices_and_coefficients_only_not_raw_feature_chain_independent_reproduction",
        "model_predict_calls": 0, "producer_scorer_calls": 0, "new_optimizer_fits": 0, "new_neural_forwards": 0,
        "raw_gene_matrix_or_H5_reads": 0, "outer_predictions_read": False, "CI_pvalues_bootstrap": False,
        "DSS_access_boundary": "verifier_reads_saved_training409_scopes_only; original_producer_canonical_DSS_reader_decoded512_before_subsetting409; outer103_not_fitted_or_scored",
        "scientific_mechanism_decision_assigned": False}
    public_json(target / "INDEPENDENT_SUMMARY.json", summary)
    return {"status": summary["status"], "fitted_heads_audited": len(cases), "explicit_rank_NA": 2376 - len(cases), "actual_official_API_calls": official_summary["actual_API_calls"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-root", "readout-manifest", "study-lock", "expected-study-lock-sha256", "readout-lock", "expected-readout-lock-sha256", "sksurv-target", "output-newdir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--report-dir")
    parser.add_argument("--all72-jobs-completed-authorized", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_run(args.manifest_root, args.readout_manifest, args.study_lock, args.expected_study_lock_sha256,
        args.readout_lock, args.expected_readout_lock_sha256, args.output_newdir, sksurv_target=args.sksurv_target,
        report_dir=args.report_dir, authorized=args.all72_jobs_completed_authorized), sort_keys=True))


if __name__ == "__main__":
    main()

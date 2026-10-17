"""Independent anonymous audit of completed Phase22 external spatial fits.

Metadata checks run on CPU. GPU inference requires explicit --replay; neither
mode reads external ST-test expression or TCGA outcomes. Runtime identities and
arrays remain private and never appear in the public audit report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
import torch
from threadpoolctl import threadpool_limits

from scripts.run_phase22_external_spatial import (
    ARMS, SEEDS, SCHEDULE, align_genes, load_dictionary, load_frozen_model,
    private_output, qualification_rows, region_schedule, source_hashes,
)


class AuditError(ValueError):
    """Messages are fixed audit codes, with no runtime identifiers or paths."""


def require(condition, code):
    if not condition:
        raise AuditError(code)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def independent_section_metrics(mu, log_mu, counts, measured, arm):
    """Ranks, centered residual error, and count objectives from first principles.

    This implementation does not call runner.validation, section_metrics, or
    spatial_objective. Rankings use SciPy average-tie ranks; the listwise loss
    uses its independently normalized NumPy/SciPy probability distribution.
    """
    y = np.asarray(counts, dtype=np.float64)
    m = np.asarray(measured, dtype=bool)
    p = np.asarray(mu, dtype=np.float64)
    lm = np.asarray(log_mu, dtype=np.float64)
    require(y.ndim == 2 and y.shape == p.shape == lm.shape and m.shape == (y.shape[1],),
            "REPLAY_DIMENSION_MISMATCH")
    y, p, lm = y[:, m], p[:, m], lm[:, m]
    require(np.isfinite(y).all() and (y >= 0).all() and np.array_equal(y, y.round()),
            "REPLAY_INVALID_MEASURED_COUNTS")
    require(np.isfinite(p).all() and np.isfinite(lm).all() and (p >= 0).all(),
            "REPLAY_NONFINITE_MEASURED_PREDICTION")
    library = y.sum(axis=1)
    active = library > 0
    require(active.sum() >= 3 and (p[active].sum(axis=1) > 0).all(),
            "REPLAY_INSUFFICIENT_VALID_PROFILES")
    truth = y[active] / library[active, None]
    prediction = p[active] / p[active].sum(axis=1, keepdims=True)
    variable = np.ptp(truth, axis=0) > 0
    require(variable.any(), "REPLAY_NO_SPATIALLY_VARIABLE_GENES")
    rt = rankdata(truth[:, variable], axis=0, method="average")
    rp = rankdata(prediction[:, variable], axis=0, method="average")
    rt -= rt.mean(axis=0)
    rp -= rp.mean(axis=0)
    denominator = np.sqrt((rt * rt).sum(axis=0) * (rp * rp).sum(axis=0))
    correlations = np.divide((rt * rp).sum(axis=0), denominator,
                             out=np.zeros_like(denominator), where=denominator > 0)
    centered_residual = prediction - truth
    centered_residual -= centered_residual.mean(axis=0, keepdims=True)
    mass = library.sum()
    yy = y[active]
    logits = lm[active]
    positive = yy > 0
    composition_logp = logits - logsumexp(logits, axis=1, keepdims=True)
    composition = -float(np.sum(yy[positive] * composition_logp[positive])) / mass
    spatial_logits = logits + np.log(library[active, None])
    spatial_logp = spatial_logits - logsumexp(spatial_logits, axis=0, keepdims=True)
    relative = -float(np.sum(yy[positive] * spatial_logp[positive])) / mass
    return dict(gene_spearman=float(correlations.mean()),
                spatially_centered_MSE=float(np.mean(centered_residual**2)),
                composition=composition, relative=relative,
                objective=composition + (relative if arm == "A2" else 0.))


def compare_metrics(actual, expected):
    errors = {key: abs(float(value) - float(expected[key])) for key, value in actual.items()}
    require(all(math.isfinite(v) for v in errors.values()), "NONFINITE_REPLAY_ERROR")
    tolerances = dict(gene_spearman=1e-6, spatially_centered_MSE=1e-9,
                      composition=1e-8, relative=1e-8, objective=2e-8)
    require(all(errors[key] <= tolerances[key] for key in actual), "REPLAY_METRIC_MISMATCH")
    return errors


def check_journals(folder, curves, done):
    attempts = read_lines(folder / "ATTEMPTS_PRIVATE.jsonl")
    updates = read_lines(folder / "STEP_EVENTS_PRIVATE.jsonl")
    starts, completions = {}, []
    prefixes = np.r_[0, np.cumsum([r["optimizer_steps"] for r in curves])]
    for event in attempts:
        kind = event.get("event")
        attempt = event.get("attempt_id")
        require(isinstance(attempt, str) and attempt, "INVALID_ATTEMPT_ID")
        if kind == "PROCESS_START":
            require(attempt not in starts, "DUPLICATE_PROCESS_START")
            epoch = event["resumed_from_epoch"]
            require(isinstance(epoch, int) and 0 <= epoch <= len(curves), "INVALID_RESUME_EPOCH")
            require(event["committed_optimizer_steps"] == int(prefixes[epoch]), "RESUME_STEP_PREFIX_MISMATCH")
            starts[attempt] = event
        elif kind == "PROCESS_COMPLETED":
            require(attempt in starts, "COMPLETION_WITHOUT_PROCESS_START")
            require(event["completed_epoch_count"] == len(curves) and
                    event["committed_optimizer_steps"] == done["optimizer_steps"],
                    "JOURNAL_COMPLETION_TOTAL_MISMATCH")
            completions.append(attempt)
        else:
            raise AuditError("UNSUPPORTED_ATTEMPT_EVENT")
    require(starts and completions, "MISSING_COMPLETED_ATTEMPT_JOURNAL")
    per_attempt = Counter()
    last_epoch = {}
    covered = set()
    for event in updates:
        require(event.get("event") == "OPTIMIZER_STEP_COMPLETED", "UNSUPPORTED_UPDATE_EVENT")
        attempt = event.get("attempt_id")
        require(attempt in starts, "UPDATE_WITHOUT_PROCESS_START")
        per_attempt[attempt] += 1
        start = starts[attempt]
        epoch, sequence = event["epoch"], event["committed_sequence"]
        require(isinstance(epoch, int) and start["resumed_from_epoch"] < epoch <= len(curves),
                "INVALID_UPDATE_EPOCH")
        require(epoch >= last_epoch.get(attempt, epoch), "UPDATE_EPOCH_ORDER_MISMATCH")
        last_epoch[attempt] = epoch
        require(sequence == start["committed_optimizer_steps"] + per_attempt[attempt],
                "UPDATE_SEQUENCE_MISMATCH")
        require(int(prefixes[epoch - 1]) < sequence <= int(prefixes[epoch]),
                "UPDATE_OUTSIDE_COMMITTED_EPOCH_PREFIX")
        covered.add((epoch, sequence))
    required = {(r["epoch"], sequence) for r in curves
                for sequence in range(int(prefixes[r["epoch"] - 1]) + 1, int(prefixes[r["epoch"]]) + 1)}
    require(required <= covered, "COMMITTED_TRAJECTORY_LACKS_RECORDED_UPDATE_EVIDENCE")
    require(len(updates) >= done["optimizer_steps"], "UPDATE_JOURNAL_TOO_SHORT")
    return dict(process_attempts=len(starts), completed_attempts=len(completions),
                recorded_optimizer_update_events=len(updates),
                repeated_recorded_updates_after_interruptions=len(updates) - done["optimizer_steps"])


def check_cell(folder, arm, seed, expected_identity, dictionary, genes, train, replay_sections, device, replay):
    done = read_json(folder / "DONE_PRIVATE.json")
    require(done.get("status") == "COMPLETE" and done.get("formal_completed_fit_cells") == 1,
            "FIT_NOT_FORMALLY_COMPLETE")
    identity = done["identity"]
    for key, value in expected_identity.items():
        require(identity.get(key) == value, "FIT_SOURCE_DATA_DICTIONARY_IDENTITY_MISMATCH")
    require(identity.get("arm") == arm and identity.get("seed") == seed, "FIT_ARM_SEED_MISMATCH")
    require(done.get("schedule") == SCHEDULE, "FIT_SCHEDULE_MISMATCH")
    require(done.get("test_labels_or_counts_read") is False and done.get("dictionary_refit") is False,
            "FIT_DISALLOWED_TEST_ACCESS_OR_DICTIONARY_REFIT")
    require(done.get("dictionary_fit_reference") == expected_identity["dictionary_npz_sha256"],
            "FIT_DICTIONARY_REFERENCE_MISMATCH")
    require(done.get("model_sha256") == digest(folder / "model.pt"), "MODEL_CHECKPOINT_HASH_MISMATCH")
    curve_record = read_json(folder / "TRAINING_CURVES_PRIVATE.json")
    require(curve_record.get("identity") == identity, "TRAINING_CURVE_IDENTITY_MISMATCH")
    curves = curve_record["curves"]
    epochs = len(curves)
    require(SCHEDULE["min_epochs"] <= epochs <= SCHEDULE["max_epochs"], "ACTUAL_EPOCHS_OUTSIDE_LOCKED_BUDGET")
    require(done.get("actual_epochs") == epochs and [r["epoch"] for r in curves] == list(range(1, epochs + 1)),
            "EPOCH_SEQUENCE_OR_COUNT_MISMATCH")
    require(done.get("stopped_at_budget") == (epochs == SCHEDULE["max_epochs"]), "BUDGET_STOP_FLAG_MISMATCH")
    expected_regions = len({s.donor_id for s in train}) * SCHEDULE["regions_per_donor"]
    require(expected_regions == 32, "FINITE_CRC_TRAINING_REGION_COUNT_MISMATCH")
    best, best_epoch, first_stop = math.inf, None, None
    for row in curves:
        epoch = row["epoch"]
        objective = float(row["validation"]["objective"])
        require(all(math.isfinite(float(row[key])) for key in
                    ["train_objective", "train_composition", "train_relative"])
                and all(math.isfinite(float(v)) for v in row["validation"].values() if isinstance(v, (int, float))),
                "NONFINITE_TRAINING_OR_VALIDATION_CURVE")
        steps, skipped = row["optimizer_steps"], row["zero_count_regions_skipped"]
        require(isinstance(steps, int) and isinstance(skipped, int) and steps > 0 and skipped >= 0
                and steps + skipped == expected_regions, "EPOCH_STEP_OR_SKIP_ACCOUNTING_MISMATCH")
        if skipped:
            true_skips = sum(align_genes(section, genes)[0][index].sum() <= 0
                             for section, index in region_schedule(train, seed, epoch))
            require(skipped == true_skips, "RECORDED_ZERO_COUNT_SKIP_NOT_REPRODUCIBLE")
        if objective < best:
            best, best_epoch = objective, epoch
        if first_stop is None and epoch >= SCHEDULE["min_epochs"] and epoch - best_epoch >= SCHEDULE["patience"]:
            first_stop = epoch
    require(done["selected_epoch"] == best_epoch, "SELECTED_EPOCH_NOT_FIRST_MINIMUM_VALIDATION_OBJECTIVE")
    require(first_stop is None and epochs == SCHEDULE["max_epochs"] or first_stop == epochs,
            "EARLY_STOPPING_TRAJECTORY_MISMATCH")
    require(done["optimizer_steps"] == sum(r["optimizer_steps"] for r in curves), "COMMITTED_STEP_TOTAL_MISMATCH")
    selected_metrics = curves[best_epoch - 1]["validation"]
    require(done["validation"] == selected_metrics, "DONE_METRICS_DO_NOT_MATCH_SELECTED_EPOCH")
    run = read_json(folder / "RUN_PRIVATE.json")
    require(run.get("identity") == identity and run.get("schedule") == SCHEDULE and
            run.get("formal_fit_cell") is True and run.get("smoke") is False,
            "RUN_RECEIPT_MISMATCH")
    journals = check_journals(folder, curves, done)
    last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
    require(last.get("identity") == identity and last.get("schedule") == SCHEDULE,
            "LAST_CHECKPOINT_LINEAGE_MISMATCH")
    require(last.get("epoch") == epochs and last.get("steps") == done["optimizer_steps"] and
            last.get("best_epoch") == best_epoch and last.get("best") == best and last.get("curves") == curves,
            "LAST_CHECKPOINT_TRAJECTORY_MISMATCH")
    del last
    model, checkpoint = load_frozen_model(folder / "model.pt", device=device if replay else "cpu")
    require(checkpoint.get("identity") == identity and checkpoint.get("arm") == arm and checkpoint.get("seed") == seed,
            "SELECTED_CHECKPOINT_IDENTITY_MISMATCH")
    require(checkpoint.get("selected_epoch") == best_epoch and checkpoint.get("selection_validation") == selected_metrics,
            "SELECTED_CHECKPOINT_METRIC_OR_EPOCH_MISMATCH")
    require(checkpoint.get("gene_names") == list(genes), "SELECTED_CHECKPOINT_GENE_ORDER_MISMATCH")
    require(torch.equal(model.dictionary.cpu(), torch.as_tensor(dictionary, dtype=torch.float32)),
            "CHECKPOINT_DICTIONARY_VALUES_MISMATCH")
    if replay:
        model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    require(checkpoint.get("trainable_parameters") == parameters, "CHECKPOINT_PARAMETER_COUNT_MISMATCH")
    errors = None
    if replay:
        grouped = defaultdict(list)
        with torch.inference_mode():
            for section in replay_sections:
                count, measured = align_genes(section, genes)
                prediction = model(torch.as_tensor(section.features, dtype=torch.float32, device=device),
                    torch.as_tensor(section.coords_um, dtype=torch.float64, device=device),
                    [section.section_id] * len(count))
                mu = prediction.mu.cpu().numpy()
                lm = prediction.log_mu.cpu().numpy()
                require(np.isfinite(mu).all() and np.max(np.abs(mu.sum(axis=1) - 1.)) <= 1e-5,
                        "REPLAY_GENE_SIMPLEX_VIOLATION")
                grouped[section.donor_id].append(independent_section_metrics(mu, lm, count, measured, arm))
        donor_metrics = [{k: float(np.mean([r[k] for r in rows])) for k in rows[0]} for rows in grouped.values()]
        replay_metrics = {k: float(np.mean([r[k] for r in donor_metrics])) for k in donor_metrics[0]}
        errors = compare_metrics(replay_metrics, selected_metrics)
        require(done["validation"]["donors"] == len(grouped) and
                done["validation"]["sections"] == len(replay_sections), "REPLAY_VALIDATION_GROUP_COUNT_MISMATCH")
    del model, checkpoint
    return dict(arm=arm, seed=seed, actual_epochs=epochs, selected_epoch=best_epoch,
                committed_optimizer_steps=done["optimizer_steps"], parameters=parameters,
                zero_count_regions_skipped=sum(r["zero_count_regions_skipped"] for r in curves),
                replayed=bool(replay), replay_absolute_errors=errors, **journals)


def audit(args):
    out = private_output(args.output_root)
    complete = [(arm, seed) for arm in ARMS for seed in SEEDS
                if (out / "fits" / arm / ("seed%03d" % seed) / "DONE_PRIVATE.json").is_file()]
    report = dict(schema_version="phase22_independent_artifact_audit_v1", status="RUNNING",
        expected_formal_fits=9, completed_receipts_found=len(complete), verified_formal_fits=0,
        missing_formal_fits=9 - len(complete), metadata_verified_fits=0, replayed_fits=0,
        require_complete=bool(args.require_complete), replay_requested=bool(args.replay),
        audit_reads="ST_train_and_validation_only", DSS_or_PFI_read=False, ST_test_expression_read=False,
        public_report_contains_identifiers_or_arrays=False,
        independent_replay_metrics=["gene_spearman", "spatially_centered_MSE", "composition", "relative", "objective"],
        training_update_journal_scope="recorded_events_include_repeats; unjournaled_hardware_updates_cannot_be_inferred",
        fits=[], failures=[])
    try:
        lock_sha, qualification_sha = digest(args.split_lock), digest(args.qualification)
        marker = read_json(args.qualification)
        require(marker.get("qualification_code_sha256") == dict(
            script=digest(REPO / "scripts/qualify_phase22_spatial_data.py"),
            data_contract=digest(REPO / "src/phase22_spatial_bridge/data_contract.py")),
            "DATA_QUALIFICATION_SOURCE_CODE_DRIFT")
        sections, observed_lock, _ = qualification_rows(args.data_root, args.split_lock, args.qualification,
            roles=("train", "validation"), with_features=bool(complete))
        require(observed_lock == lock_sha, "SPLIT_LOCK_FILE_DRIFT")
        dictionary, genes, receipt, dictionary_sha = load_dictionary(out)
        require(receipt["identity"]["split_lock_sha256"] == lock_sha and
                receipt["identity"]["data_qualification_sha256"] == qualification_sha,
                "DICTIONARY_DATA_SPLIT_LINEAGE_MISMATCH")
        expected = dict(split_lock_sha256=lock_sha, data_qualification_sha256=qualification_sha,
                        dictionary_npz_sha256=dictionary_sha, source_hashes=source_hashes())
        expected_sections = {s.section_id: s.evidence for s in sections}
        train, validation = [s for s in sections if s.split == "train"], [s for s in sections if s.split == "validation"]
        report.update(training_donors=len({s.donor_id for s in train}),
                      validation_donors=len({s.donor_id for s in validation}),
                      training_sections=len(train), validation_sections=len(validation),
                      dictionary_states=int(dictionary.shape[0]), dictionary_genes=int(dictionary.shape[1]))
        device = args.device
        if args.replay:
            require(device != "cuda" or torch.cuda.is_available(), "REQUESTED_REPLAY_CUDA_UNAVAILABLE")
            torch.set_num_threads(SCHEDULE["threads"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
        for arm, seed in complete:
            folder = out / "fits" / arm / ("seed%03d" % seed)
            try:
                evidence = read_json(folder / "DONE_PRIVATE.json")["identity"]["source_sections"]
                require(set(evidence) == set(expected_sections), "FIT_SOURCE_SECTION_SET_MISMATCH")
                for key in expected_sections:
                    # Absolute feature cache locations are transferable runtime
                    # paths; all identity/content/receipt hashes must agree.
                    actual = {k: v for k, v in evidence[key].items() if k != "features_path"}
                    wanted = {k: v for k, v in expected_sections[key].items() if k != "features_path"}
                    require(actual == wanted, "FIT_SOURCE_FEATURE_OR_COUNT_EVIDENCE_MISMATCH")
                row = check_cell(folder, arm, seed, expected, dictionary, genes, train, validation, device, args.replay)
                report["fits"].append(row)
                report["metadata_verified_fits"] += 1
                report["replayed_fits"] += int(args.replay)
                report["verified_formal_fits"] += 1
            except Exception as error:
                report["failures"].append(dict(arm=arm, seed=seed,
                    code=str(error) if isinstance(error, AuditError) else "UNEXPECTED_PRIVATE_ARTIFACT_ERROR",
                    error_class=type(error).__name__))
    except Exception as error:
        report["failures"].append(dict(code=str(error) if isinstance(error, AuditError) else "PREREQUISITE_ARTIFACT_ERROR",
                                     error_class=type(error).__name__))
    if report["failures"]:
        report["status"] = "FAIL_ARTIFACT_AUDIT"
    elif len(complete) < 9:
        report["status"] = "FAIL_INCOMPLETE_FORMAL_MATRIX" if args.require_complete else "PARTIAL_INCOMPLETE_FORMAL_MATRIX"
    else:
        report["status"] = "PASS_INDEPENDENT_REPLAY" if args.replay else "PASS_METADATA_REPLAY_NOT_RUN"
    errors = [r["replay_absolute_errors"] for r in report["fits"] if r["replayed"]]
    report["maximum_absolute_replay_errors"] = ({key: max(row[key] for row in errors) for key in errors[0]} if errors else None)
    report["parameters_by_arm"] = {arm: sorted({r["parameters"] for r in report["fits"] if r["arm"] == arm}) for arm in ARMS}
    report["total_committed_optimizer_steps"] = sum(r["committed_optimizer_steps"] for r in report["fits"])
    report["total_recorded_update_events"] = sum(r["recorded_optimizer_update_events"] for r in report["fits"])
    write_report(args.report, report)
    print(json.dumps({key: report[key] for key in ["status", "verified_formal_fits", "missing_formal_fits", "replayed_fits"]}))
    return 2 if report["status"].startswith("FAIL") else 0


def self_test():
    from scipy.stats import spearmanr
    truth = np.asarray([[9, 1, 2], [6, 4, 2], [3, 7, 2], [1, 9, 2], [0, 0, 0]], dtype=float)
    prediction = np.asarray([[8, 2, 2], [5, 5, 2], [2, 8, 2], [1, 9, 2], [1, 1, 1]], dtype=float)
    prediction /= prediction.sum(axis=1, keepdims=True)
    result = independent_section_metrics(prediction, np.log(prediction), truth, np.ones(3, bool), "A0")
    observed = truth[:-1] / truth[:-1].sum(axis=1, keepdims=True)
    expected = np.mean([spearmanr(observed[:, k], prediction[:-1, k]).statistic for k in range(2)])
    require(abs(result["gene_spearman"] - expected) < 1e-14, "SELF_TEST_TIED_RANK_CORRELATION")
    residual = prediction[:-1] - observed
    require(abs(result["spatially_centered_MSE"] - np.mean((residual - residual.mean(0))**2)) < 1e-14,
            "SELF_TEST_CENTERED_ERROR")
    # A missing gene with arbitrary payload must not change observed metrics.
    masked = independent_section_metrics(np.c_[prediction, np.full(5, np.nan)],
        np.c_[np.log(prediction), np.full(5, np.nan)], np.c_[truth, np.full(5, np.nan)],
        np.asarray([True, True, True, False]), "A0")
    require(result == masked, "SELF_TEST_UNMEASURED_PAYLOAD")
    perfect = independent_section_metrics(np.maximum(observed, 1e-10), np.log(np.maximum(observed, 1e-10)),
        truth[:-1], np.ones(3, bool), "A2")
    require(abs(perfect["gene_spearman"] - 1.) < 1e-14 and perfect["spatially_centered_MSE"] < 1e-20,
            "SELF_TEST_IDENTITY_PREDICTION")
    print("SELF_TEST_PASS: independent ranks, centered error, missing mask, and perfect prediction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-lock", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path, default=REPO / "reports/phase22_spatial_bridge/INDEPENDENT_ARTIFACT_AUDIT.json")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if any(getattr(args, name) is None for name in ["data_root", "split_lock", "qualification", "output_root"]):
        parser.error("artifact audit requires explicit private data, split, qualification and output roots")
    with threadpool_limits(limits=SCHEDULE["threads"]):
        return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())

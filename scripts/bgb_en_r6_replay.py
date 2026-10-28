"""Independent R6 saved-state/paired-credit replay: zero optimizer updates.

Only all 90 prelocked cells can pass. PASS receipts may be strictly resumed;
FAIL receipts and their original private logs are preserved and never retried
by this program. Partial runs are explicitly INCOMPLETE, never subset-PASS.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from scipy.special import expit
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_r6_common import (parser, context, qualified_prepared, construct, read, write,
    sha256, require, validate_qualification, public_snapshot_guard, configure_deterministic_runtime,
    validate_deterministic_runtime, canonical_hash, PARTITIONS, SEEDS, TARGET_ARMS)
from scripts.bgb_en_r6_run import cell_name
from src.bgb_en.r6_training import predict, saved_state, state_hash, SCALARS
from src.bgb_en.r6_pathology import source_model_state_sha256
from src.bgb_en.replay import apply_saved_preprocessing, replay_full_risks
from src.phase10_trimodal.reference import components, pairs, uno_weights

FROZEN_CONTROLS = ("matched_query", "wrong_query", "mean_query", "interaction_off", "value_off",
                   "attention_permutation", "whole_omics")
PAIR_REFERENCES = ("B1", "B3", "B4", "B5", "B6", "B8", "B9", "B11", "B13")
METRIC_KEYS = ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den",
               "repair", "damage", "Uno_repair", "Uno_damage", "risk_shift_sd_ratio")


def credit(time, event, risk):
    i, j = pairs(time, event)
    risk = np.asarray(risk, float)
    require(risk.shape == np.asarray(time).shape and np.isfinite(risk).all(), "R6_REPLAY_INVALID_RISK_VECTOR")
    difference = risk[i] - risk[j]
    return np.where(np.abs(difference) <= 1e-8, .5, (difference > 0).astype(float))


def independent_metrics(train_time, train_event, time, event, risk, reference=None):
    """Recompute every numerator and direct pair repair/damage from risk."""
    risk = np.asarray(risk, float)
    w = uno_weights(train_time, train_event, time, event, tau=60.)
    hn, hd = components(time, event, risk)
    un, ud = components(time, event, risk, w)
    result = dict(Harrell=hn/hd if hd else None, Uno60=un/ud if ud else None,
                  Harrell_num=hn, Harrell_den=hd, Uno_num=un, Uno_den=ud)
    if reference is not None:
        i, _ = pairs(time, event)
        difference = credit(time, event, risk) - credit(time, event, reference)
        result.update(repair=float(np.maximum(difference, 0).sum()/hd) if hd else None,
            damage=float(np.maximum(-difference, 0).sum()/hd) if hd else None,
            Uno_repair=float(np.dot(w[i], np.maximum(difference, 0))/ud) if ud else None,
            Uno_damage=float(np.dot(w[i], np.maximum(-difference, 0))/ud) if ud else None,
            risk_shift_sd_ratio=float(np.std(risk-reference)/max(np.std(reference), 1e-12)))
    return result


def assert_metrics(actual, expected, *, code="R6_INDEPENDENT_METRICS_CHANGED"):
    maximum = 0.
    for key, value in actual.items():
        require(key in expected, code)
        target = expected[key]
        if value is None or target is None:
            require(value is None and target is None, code)
        else:
            require(np.isfinite(value) and np.isfinite(target), code)
            error = abs(float(value)-float(target))
            maximum = max(maximum, error)
            require(error <= 1e-10, code)
    return maximum


def metrics_for(raw, prepared, indices, risk, *, reference=None, endpoint="DSS"):
    train = prepared["train"]
    if endpoint == "DSS":
        return independent_metrics(raw["t"][train], raw["e"][train], raw["t"][indices], raw["e"][indices], risk, reference)
    require(endpoint == "PFI", "R6_UNKNOWN_REPLAY_ENDPOINT")
    train = train[raw["pfi_valid"][train]]
    usable = raw["pfi_valid"][indices]
    return independent_metrics(raw["pfi_t"][train], raw["pfi_e"][train],
        raw["pfi_t"][indices][usable], raw["pfi_e"][indices][usable], np.asarray(risk)[usable],
        None if reference is None else np.asarray(reference)[usable])


def upstream_replay(ctx, prepared):
    """13 coefficient models and one fixed beta, without fitting anything."""
    raw, errors = ctx["raw"], []
    features = apply_saved_preprocessing(raw, prepared["features"]["state"])
    for key, value in features.items():
        if key == "state":
            continue
        target = prepared["features"][key]
        error = float(np.max(np.abs(value.astype(float)-target.astype(float))))
        require(error <= (2e-12 if value.dtype == np.float64 else 0.), "R6_SAVED_PREPROCESSING_REPLAY_FAILED")
    full = replay_full_risks(features, prepared["full_models"])
    coefficient_models = len(full)
    errors.extend(float(np.max(np.abs(value-prepared["full_risks"][key]))) for key, value in full.items())
    for teacher in prepared["teachers"]:
        teacher_features = apply_saved_preprocessing(raw, teacher["preprocessing"])
        risks = replay_full_risks(teacher_features, teacher["models"])
        coefficient_models += len(risks)
        for key, target in (("LOCAL", "CP"), ("CS", "CS"), ("R", "RNA")):
            held = teacher["held"]
            errors.append(float(np.max(np.abs(risks[key][held]-prepared[target][held]))))
    require(coefficient_models == 13 and max(errors) <= 2e-12, "R6_UPSTREAM_COX_COEFFICIENT_REPLAY_FAILED")
    train, left, right = prepared["train"], [], []
    for group in np.unique(prepared["teacher_groups"][train]):
        idx = train[prepared["teacher_groups"][train] == group]
        a, b = pairs(raw["t"][idx], raw["e"][idx])
        left.extend(idx[a]); right.extend(idx[b])
    a, b = np.asarray(left, int), np.asarray(right, int)
    difference = prepared["RNA"][a]-prepared["RNA"][b]
    margin = prepared["CP"][a]-prepared["CP"][b]+prepared["beta"]*difference
    gradient = float(np.mean(-expit(-margin)*difference)+prepared["beta"])
    require(abs(gradient) <= 1e-6, "R6_SAME_TEACHER_BETA_REPLAY_FAILED")
    return dict(status="PASS", coefficient_models=13, beta_gradient_abs=abs(gradient),
        maximum_risk_error=max(errors), new_optimizer_fits=0, new_preprocessing_fits=0,
        teacher_heldout_labels_excluded_from_teacher_fits=True, validation_labels_excluded=True)


def upstream_audit(args, ctx, prepared_cache, origins, qualification):
    path = args.private / "UPSTREAM_REPLAY_AUDIT.json"
    expected_sha = qualification.get("upstream_replay_audit_sha256")
    if expected_sha is not None:
        require(sha256(path) == expected_sha, "R6_QUALIFIED_UPSTREAM_REPLAY_AUDIT_SHA_CHANGED")
        result = read(path)
        require(result.get("status") == "PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY"
                and result["contract_sha256"] == ctx["lock"]["contract_sha256"]
                and result["prepared_receipt_sha256"] == {str(p): origins[p]["prepared_receipt_sha256"] for p in PARTITIONS}
                and len(result["partitions"]) == 3 and sum(r["coefficient_models"] for r in result["partitions"]) == 39
                and all(r["status"] == "PASS" and r["beta_gradient_abs"] <= 1e-6 for r in result["partitions"]),
                "R6_QUALIFIED_UPSTREAM_REPLAY_AUDIT_SCOPE_CHANGED")
        return result, sha256(path)
    rows = [dict(partition=p, **upstream_replay(ctx, prepared_cache[p])) for p in PARTITIONS]
    result = dict(status="PASS_R6_39_COEFFICIENTS_3_BETA_PREPROCESSING_REPLAY",
        contract_sha256=ctx["lock"]["contract_sha256"], partitions=rows,
        prepared_receipt_sha256={str(p): origins[p]["prepared_receipt_sha256"] for p in PARTITIONS},
        new_optimizer_fits=0, new_preprocessing_fits=0)
    if path.exists():
        require(read(path) == result, "R6_EXISTING_UPSTREAM_REPLAY_AUDIT_CHANGED")
    else:
        write(path, result)
    return result, sha256(path)


def planned_units():
    return {cell_name(arm, part, seed): (arm, part, seed)
            for arm in TARGET_ARMS for part in PARTITIONS for seed in SEEDS}


def reference_replay(args, ctx, prepared_cache, origins):
    """Independently bind the 18 zero-fit baseline entries to saved anchors."""
    expected = {f"{arm}_p{part}_seed{seed}.json": (arm, part, seed)
                for arm in ("B0", "B1") for part in PARTITIONS for seed in SEEDS}
    paths = sorted((args.private/"references").glob("*.json"))
    require({path.name for path in paths} == set(expected), "R6_EXACT_18_BASELINE_REFERENCES_REQUIRED")
    rows = []
    for path in paths:
        arm, part, seed = expected[path.name]
        metadata, prepared = read(path), prepared_cache[part]
        require(metadata.get("status") == "R6_BASELINE_REFERENCE" and metadata.get("group") == "REFERENCE"
                and metadata.get("arm") == arm and metadata.get("partition") == part and metadata.get("seed") == seed
                and metadata.get("prepared_origin") == origins[part]
                and metadata.get("contract_sha256") == ctx["lock"]["contract_sha256"]
                and metadata.get("actual_new_optimizer_fits") == metadata.get("teacher_refits") == 0
                and metadata.get("deterministic_seed_reference") is (seed != 31),
                "R6_BASELINE_REFERENCE_IDENTITY_OR_PROVENANCE_CHANGED")
        valid = prepared["valid"]
        risk = prepared["CP" if arm == "B0" else "base"][valid]
        primary = metrics_for(ctx["raw"], prepared, valid, risk, reference=prepared["base"][valid])
        assert_metrics(primary, metadata["metrics"], code="R6_BASELINE_REFERENCE_METRICS_CHANGED")
        assert_metrics(primary, metadata["metrics_vs"]["CP_SCALAR"], code="R6_BASELINE_REFERENCE_METRICS_CHANGED")
        assert_metrics(metrics_for(ctx["raw"], prepared, valid, risk, reference=prepared["CP"][valid]),
                       metadata["metrics_vs"]["CP_STRONG"], code="R6_BASELINE_REFERENCE_METRICS_CHANGED")
        assert_metrics(metrics_for(ctx["raw"], prepared, valid, risk, endpoint="PFI"),
                       metadata["metrics_vs"]["PFI_frozen_DSS_risk"], code="R6_BASELINE_REFERENCE_METRICS_CHANGED")
        rows.append(dict(arm=arm, partition=part, seed=seed, status="PASS", receipt_sha256=sha256(path),
                         independent_fixed_anchor_metrics_recomputed=True, new_optimizer_fits=0))
    return rows


def validate_entry(receipt, metadata, artifact, ctx, args, origin, qualification_sha):
    arm, part, seed = metadata["arm"], metadata["partition"], metadata["seed"]
    require((arm, part, seed) == planned_units().get(receipt.parent.name)
            and metadata["status"] == "COMPLETE_R6_TARGET_FIT", "R6_COMPLETE_UNIT_IDENTITY_CHANGED")
    contract = metadata["contract"]
    require(contract == artifact["contract"] and contract["contract_sha256"] == ctx["lock"]["contract_sha256"]
            and contract["prepared_origin"] == origin and contract["qualification_sha256"] == qualification_sha
            and contract["config_sha256"] == sha256(args.config)
            and contract["source_config_sha256"] == sha256(args.source_config)
            and contract["fixed_deterministic_runtime"] == validate_deterministic_runtime()
            and contract["arm"] == arm and contract["partition"] == part and contract["seed"] == seed
            and contract["epochs"] == metadata["epochs"] == metadata["optimizer_updates"]
            == artifact["epochs"] == artifact["optimizer_updates"] == 80
            and metadata["actual_new_optimizer_fits"] == 1 and metadata["teacher_refits"] == 0
            and metadata["source_training_updates"] == 0 and metadata["outer_test_used"] is False,
            "R6_PRIVATE_PUBLIC_BUDGET_OR_INPUT_CONTRACT_CHANGED")
    expected_key = "RANDOM_NO_ST" if arm in ("B2", "B13") else "WRONG_DONOR" if arm == "B8" else "MATCHED"
    require(metadata["source_origin"] == artifact["model_origin"]
            and metadata["source_origin"]["source_key"] == expected_key
            and metadata["source_origin"]["source_origin"] == ctx["source_audits"][expected_key],
            "R6_COMPLETE_SOURCE_FAMILY_OR_TRANSFER_CHANGED")


def replay_one(args, ctx, prepared, origin, receipt, metadata, artifact, qualification_sha):
    """Deserialize the private state and actually forward its fixed patients."""
    validate_entry(receipt, metadata, artifact, ctx, args, origin, qualification_sha)
    require(np.array_equal(artifact["train"], prepared["train"])
            and np.array_equal(artifact["valid"], prepared["valid"]), "R6_SAVED_PATIENT_SCOPE_CHANGED")
    part, seed, arm = metadata["partition"], metadata["seed"], metadata["arm"]
    local_args = argparse.Namespace(**(vars(args) | dict(partition=part)))
    model, current_origin = construct(local_args, ctx, prepared, arm, seed)
    require(current_origin == metadata["source_origin"], "R6_CURRENT_SOURCE_PATCH_OR_PAIRING_ORIGIN_CHANGED")
    initial = saved_state(model)
    require(state_hash(initial) == metadata["initialization"]["full_state_sha256"]
            and state_hash(initial, target_only=True) == metadata["initialization"]["target_state_sha256"],
            "R6_INITIAL_STATE_RECONSTRUCTION_CHANGED")
    require(artifact["initialization"] == metadata["initialization"]
            and set(artifact["snapshots"]) == {20, 40, 80}
            and state_hash(artifact["snapshots"][80]["state_dict"]) == state_hash(artifact["final_state_dict"]),
            "R6_PRELOCKED_FINAL_CHECKPOINT_CHANGED")
    model.load_state_dict(artifact["final_state_dict"], strict=True)
    require(source_model_state_sha256(model.core.foundation) ==
            current_origin["source_origin"]["complete_source_state_sha256"], "R6_FROZEN_FOUNDATION_WEIGHTS_CHANGED_DURING_TARGET_FIT")
    require(np.array_equal(model.query_map, artifact["query_map"]), "R6_SAVED_QUERY_DERANGEMENT_CHANGED")
    model.refresh_training_query_mean()
    mean_error = float((model.mean_context.tokens.cpu()-artifact["mean_query"]).abs().max())
    require(mean_error <= 2e-6 and torch.equal(model.mean_context.program_mask.cpu(), artifact["mean_query_mask"]),
            "R6_TRAIN_ONLY_MEAN_REFERENCE_CHANGED")
    raw, risks, split_errors, metric_errors = ctx["raw"], {}, {}, {}
    for split, indices in (("train", prepared["train"]), ("validation", prepared["valid"])):
        current, target = predict(model, indices), artifact["predictions"][split]
        maximum = 0.
        for key in SCALARS:
            require(np.asarray(target[key]).shape == (len(indices),), "R6_SAVED_SCALAR_PREDICTION_SHAPE_CHANGED")
            maximum = max(maximum, float(np.max(np.abs(current[key]-target[key]))))
        require(maximum <= 2e-6, "R6_SAVED_RISK_NUMERIC_TOLERANCE_FAILED")
        require(np.array_equal(credit(raw["t"][indices], raw["e"][indices], current["risk"]),
                               credit(raw["t"][indices], raw["e"][indices], target["risk"])),
                "R6_SAVED_RISK_PAIR_CREDIT_CHANGED")
        expected = metadata["metrics_vs"] if split == "validation" else metadata["train_metrics"]
        for label, reference in (("CP_STRONG", prepared["CP"][indices]), ("CP_SCALAR", prepared["base"][indices])):
            metric_errors[split+"_"+label] = assert_metrics(metrics_for(raw, prepared, indices, current["risk"], reference=reference), expected[label])
        metric_errors[split+"_PFI"] = assert_metrics(metrics_for(raw, prepared, indices, current["risk"], endpoint="PFI"),
                                                    expected["PFI_frozen_DSS_risk"])
        split_errors[split] = maximum
        risks[split] = current["risk"]
    assert_metrics(metrics_for(raw, prepared, prepared["valid"], risks["validation"],
                              reference=prepared["base"][prepared["valid"]]), metadata["metrics"])
    valid, controls = prepared["valid"], {}
    require(set(artifact["frozen_control_predictions"]) == set(metadata["frozen_controls"]) == set(FROZEN_CONTROLS),
            "R6_FROZEN_CONTROL_INVENTORY_CHANGED")
    for tag in FROZEN_CONTROLS:
        current = predict(model, valid, control=tag)
        target = artifact["frozen_control_predictions"][tag]
        require(np.asarray(target["risk"]).shape == (len(valid),), "R6_FROZEN_CONTROL_RISK_SHAPE_CHANGED")
        error = float(np.max(np.abs(current["risk"]-target["risk"])))
        require(error <= 2e-6, "R6_FROZEN_QUERY_RISK_REPLAY_FAILED")
        require(np.array_equal(credit(raw["t"][valid], raw["e"][valid], current["risk"]),
                               credit(raw["t"][valid], raw["e"][valid], target["risk"])),
                "R6_FROZEN_QUERY_PAIR_CREDIT_CHANGED")
        stored = metadata["frozen_controls"][tag]
        assert_metrics(metrics_for(raw, prepared, valid, current["risk"], reference=prepared["base"][valid]),
                       stored["metrics_vs_base"], code="R6_FROZEN_CONTROL_BASE_METRICS_CHANGED")
        assert_metrics(metrics_for(raw, prepared, valid, risks["validation"], reference=current["risk"]),
                       stored["relative_to_trained"], code="R6_FROZEN_CONTROL_DIRECT_PAIR_CREDIT_CHANGED")
        controls[tag] = error
    return dict(status="PASS", maximum_risk_error=max(split_errors.values()), risk_errors_by_split=split_errors,
        changed_pair_credits=0, maximum_metric_error=max(metric_errors.values()),
        frozen_control_risk_errors=controls, mean_query_error=mean_error,
        actual_state_dict_forward_replayed=True, source_frozen_weight_hash_replayed=True,
        saved_query_mapping_recomputed=True, new_optimizer_updates=0), risks["validation"]


def paired_comparisons(raw, prepared_cache, risks):
    """Paired candidate-vs-reference credit, never differences of summaries."""
    main, raw_gene = [], []
    for part in PARTITIONS:
        prepared = prepared_cache[part]
        valid = prepared["valid"]
        for seed in SEEDS:
            for candidate, reference in [("B7", ref) for ref in PAIR_REFERENCES]+[("B6", "B5")]:
                a = risks[(part, seed, candidate)]
                b = prepared["base"][valid] if reference == "B1" else risks[(part, seed, reference)]
                row = dict(candidate=candidate, reference=reference, partition=part, seed=seed,
                    candidate_vs_reference=metrics_for(raw, prepared, valid, a, reference=b),
                    PFI_frozen_DSS_risk=metrics_for(raw, prepared, valid, a, reference=b, endpoint="PFI"),
                    risk_SD_ratio=float(np.std(a-b)/max(np.std(prepared["base"][valid]),1e-12)),
                    direct_risk_pair_credits=True, bootstrap_or_CI_computed=False)
                (raw_gene if candidate == "B6" else main).append(row)
    return main, raw_gene


def _persist_marker(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, "R6_REPLAY_TERMINAL_MARKER_CANNOT_BE_OVERWRITTEN")
    else:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)


def replay_inventory(private):
    expected = planned_units()
    entries = {}
    for receipt in sorted((Path(private)/"cells").glob("*/COMPLETE.json")):
        require(receipt.parent.name in expected, "R6_REPLAY_UNEXPECTED_COMPLETED_UNIT")
        require(not any((receipt.parent/name).exists() for name in ("ERROR_RECEIPT.json", "FAILED.json")),
                "R6_REPLAY_CONFLICTING_FIT_TERMINAL_MARKERS")
        metadata = read(receipt)
        require((metadata["arm"], metadata["partition"], metadata["seed"]) == expected[receipt.parent.name]
                and metadata["status"] == "COMPLETE_R6_TARGET_FIT", "R6_COMPLETE_UNIT_IDENTITY_CHANGED")
        entries[receipt.parent.name] = (receipt, metadata)
    return expected, entries


def run(args, ctx=None):
    require(args.limit >= 0, "R6_NEGATIVE_REPLAY_LIMIT")
    ctx = context(args) if ctx is None else ctx
    qualification_sha = validate_qualification(args, ctx)
    qualification = read(args.private/"QUALIFICATION.json")
    parent = public_snapshot_guard()
    require(ctx["lock"]["sources"].get("scripts/bgb_en_r6_replay.py") == sha256(__file__),
            "R6_REPLAY_SCRIPT_MUST_BE_IN_TARGET_LOCK")
    output, progress = args.private/"REPLAY_AUDIT.json", args.private/"REPLAY_PROGRESS.json"
    terminal = args.private/"REPLAY_PRIVATE"
    require(args.resume or not (output.exists() or progress.exists() or terminal.exists()),
            "R6_REPLAY_ALREADY_STARTED_EXPLICIT_RESUME_REQUIRED")
    if output.exists():
        require(read(output)["contract_sha256"] == ctx["lock"]["contract_sha256"], "R6_REPLAY_RESUME_CONTRACT_CHANGED")
    prepared_cache, origins = {}, {}
    for part in PARTITIONS:
        prepared_cache[part], origins[part] = qualified_prepared(args, ctx, part)
    upstream, upstream_sha = upstream_audit(args, ctx, prepared_cache, origins, qualification)
    references = reference_replay(args, ctx, prepared_cache, origins)
    expected, entries = replay_inventory(args.private)
    rows, risks, new, reused = [], {}, 0, 0
    for unit in sorted(entries):
        receipt, metadata = entries[unit]
        artifact_path = receipt.parent/"model.pt"
        identity = dict(unit=unit, arm=metadata["arm"], partition=metadata["partition"], seed=metadata["seed"],
            contract_sha256=ctx["lock"]["contract_sha256"], completion_sha256=sha256(receipt),
            artifact_sha256=sha256(artifact_path) if artifact_path.is_file() else None, qualification_sha256=qualification_sha,
            upstream_replay_audit_sha256=upstream_sha, replay_script_sha256=sha256(__file__))
        pass_path, fail_path = terminal/(unit+".PASS.json"), terminal/(unit+".FAIL.json")
        require(not (pass_path.exists() and fail_path.exists()), "R6_CONFLICTING_REPLAY_TERMINAL_MARKERS")
        if args.resume and (pass_path.exists() or fail_path.exists()):
            prior = read(pass_path if pass_path.exists() else fail_path)
            require(all(prior.get(k) == v for k, v in identity.items()), "R6_RESUMED_REPLAY_ORIGIN_CHANGED")
            require(identity["artifact_sha256"] == metadata["artifact_sha256"], "R6_RESUMED_FIT_ARTIFACT_SHA_CHANGED")
            rows.append(prior); reused += 1
            if pass_path.exists():
                require(prior["status"] == "PASS" and prior.get("actual_state_dict_forward_replayed") is True,
                        "R6_RESUMED_PASS_WAS_NOT_AN_ACTUAL_STATE_REPLAY")
                artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
                validate_entry(receipt, metadata, artifact, ctx, args, origins[metadata["partition"]], qualification_sha)
                risks[(metadata["partition"], metadata["seed"], metadata["arm"])] = artifact["predictions"]["validation"]["risk"]
            else:
                require(prior["status"] == "FAIL", "R6_INVALID_PRESERVED_REPLAY_FAILURE")
            continue
        if args.limit and new >= args.limit:
            continue
        result = dict(identity, status="FAIL")
        try:
            require(identity["artifact_sha256"] == metadata["artifact_sha256"], "R6_COMPLETE_ARTIFACT_SHA_CHANGED")
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
            details, risk = replay_one(args, ctx, prepared_cache[metadata["partition"]], origins[metadata["partition"]],
                                       receipt, metadata, artifact, qualification_sha)
            result.update(details)
            risks[(metadata["partition"], metadata["seed"], metadata["arm"])] = risk
            _persist_marker(pass_path, result)
        except Exception as exc:
            message = str(exc)
            result.update(failure_type=type(exc).__name__,
                failure_code=message if re.fullmatch("[A-Z0-9_]+", message) else "R6_RUNTIME_REPLAY_FAILED")
            terminal.mkdir(parents=True, exist_ok=True)
            with (terminal/(unit+".log")).open("x", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
            _persist_marker(fail_path, result)
        rows.append(result); new += 1
        write(progress, dict(contract_sha256=ctx["lock"]["contract_sha256"], units=rows, new_actual_replays=new,
                             expected_units=90, new_optimizer_fits=0))
        print(json.dumps(dict(unit=unit, status=result["status"])), flush=True)
    passed, failed = sum(r["status"] == "PASS" for r in rows), sum(r["status"] == "FAIL" for r in rows)
    complete = len(entries) == len(rows) == passed == 90 and failed == 0
    paired, gene = paired_comparisons(ctx["raw"], prepared_cache, risks) if complete else ([], [])
    require(public_snapshot_guard() == parent, "R6_HISTORICAL_REPORTS_CHANGED_BY_REPLAY")
    result = dict(status="PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY" if complete else
        "FAILED_R6_TARGET_CHECKPOINT_REPLAY" if failed else "INCOMPLETE_R6_TARGET_CHECKPOINT_REPLAY",
        contract_sha256=ctx["lock"]["contract_sha256"], qualification_sha256=qualification_sha,
        expected_units=90, available_units=len(entries), passed_units=passed, failed_units=failed,
        missing_or_unreplayed_units=sorted(set(expected)-{r["unit"] for r in rows}),
        new_actual_replays=new, reused_terminal_receipts=reused, units=rows,
        upstream_reference_replay=upstream, upstream_replay_audit_sha256=upstream_sha,
        zero_fit_baseline_reference_replays=references,
        paired_retrained_controls=paired, paired_raw_gene_controls=gene,
        new_optimizer_fits=0, new_preprocessing_fits=0, patient_arrays_exported=False,
        numerical_risk_tolerance=2e-6, maximum_allowed_changed_pair_credits=0,
        failed_cells_automatically_retried=False, PFI_is_frozen_DSS_risk_secondary_only=True)
    write(output, result)
    return result


def main():
    configure_deterministic_runtime()
    torch.set_num_threads(2)
    p = parser(__doc__)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "passed_units", "failed_units", "available_units")}), flush=True)
    if result["failed_units"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

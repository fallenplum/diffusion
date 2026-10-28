"""Read-only, post-fit R6 subgroup diagnostics from certified saved risks.

No model is constructed, forwarded or optimized. Complete target replay and
fixed-anchor certificates are verified before the endpoint loader is called.
Only aggregate counts and within-partition concordance components are public.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.phase10_trimodal.reference import components, uno_weights

ARMS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B13")
NEURAL = ARMS[2:]
SEEDS, PARTITIONS = (31, 47, 83), (0, 1, 2)
COMPONENTS = ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
ENDPOINTS = (("DSS", "t", "e", "valid"), ("PFI_frozen_DSS_risk", "pfi_t", "pfi_e", "pfi_valid"))
IPCW_REFERENCE = "full_current_training_partition_endpoint_valid_rows_not_subgroup_refitted"


def require(condition, code):
    if not bool(condition):
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def unit_name(arm, partition, seed):
    return f"{arm}_p{partition}_seed{seed}"


def preflight(config_path, private_root, primary_private):
    """Metadata/SHA only; no cohort, endpoints or private tensor decoding."""
    private, primary = Path(private_root), Path(primary_private)
    config, lock = read(config_path), read(private / "targetLOCK.json")
    require(config.get("schema") == "bgb_en_r6_inner_v1" and tuple(config.get("trainable_arms", [])) == NEURAL
            and config.get("primary_endpoint") == "DSS" and config.get("secondary_endpoint") == "PFI_frozen_DSS_risk_only"
            and config["prototype"]["outer_fold"] == 1 and config["prototype"]["inner_partitions"] == list(PARTITIONS)
            and config["prototype"]["neural_seeds"] == list(SEEDS) and config["prototype"]["outer_test_access"] is False,
            "SUBGROUP_R6_PROTOCOL_CHANGED")
    payload = {k: v for k, v in lock.items() if k not in ("contract_sha256", "created_UTC")}
    require(lock.get("schema") == "bgb_en_r6_target_protocol_lock_v1" and lock.get("status") == "LOCKED_BEFORE_TARGET_FITS"
            and canonical(payload) == lock.get("contract_sha256") and lock.get("config_sha256") == sha256(config_path)
            and lock.get("planned_neural_fits") == 90 and lock.get("outer_test_access") is False,
            "SUBGROUP_TARGET_LOCK_CHANGED")
    for relative, expected in lock.get("sources", {}).items():
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(ROOT) and source.is_file() and sha256(source) == expected, "SUBGROUP_LOCKED_SCIENTIFIC_SOURCE_CHANGED")
    require(bool(lock.get("sources")), "SUBGROUP_TARGET_SOURCE_PINS_MISSING")
    qualification = read(private / "QUALIFICATION.json")
    qualification_sha = sha256(private / "QUALIFICATION.json")
    require(qualification.get("status") == "PASS_R6_TARGET_REAL_DATA_AND_SOURCE_TRANSFER"
            and qualification.get("contract_sha256") == lock["contract_sha256"]
            and qualification.get("new_optimizer_updates") == qualification.get("new_neural_fits") == 0
            and qualification.get("source_parameters_frozen") is True
            and qualification.get("original_gene_values_in_dynamic_reread") is True,
            "SUBGROUP_TARGET_QUALIFICATION_REQUIRED")
    replay = read(private / "REPLAY_AUDIT.json")
    require(replay.get("status") == "PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY"
            and replay.get("contract_sha256") == lock["contract_sha256"]
            and replay.get("qualification_sha256") == qualification_sha
            and replay.get("expected_units") == replay.get("available_units") == replay.get("passed_units") == 90
            and replay.get("failed_units") == replay.get("new_optimizer_fits") == replay.get("new_preprocessing_fits") == 0
            and replay.get("numerical_risk_tolerance") == 2e-6 and replay.get("maximum_allowed_changed_pair_credits") == 0,
            "SUBGROUP_REQUIRES_COMPLETE_NINETY_ACTUAL_REPLAYS")
    upstream = private / "UPSTREAM_REPLAY_AUDIT.json"
    require(upstream.is_file() and qualification.get("upstream_replay_audit_sha256") == replay.get("upstream_replay_audit_sha256") == sha256(upstream),
            "SUBGROUP_UPSTREAM_ANCHOR_CERTIFICATE_CHANGED")
    entries = replay.get("units", [])
    expected = {(arm, p, s) for arm in NEURAL for p in PARTITIONS for s in SEEDS}
    identities = [(r.get("arm"), r.get("partition"), r.get("seed")) for r in entries]
    require(len(entries) == len(set(identities)) == 90 and set(identities) == expected, "SUBGROUP_REPLAY_UNIT_MATRIX_CHANGED")
    certified = {}
    for entry in entries:
        arm, part, seed = entry["arm"], entry["partition"], entry["seed"]
        name = unit_name(arm, part, seed)
        directory = private / "cells" / name
        receipt, artifact = directory / "COMPLETE.json", directory / "model.pt"
        require(entry.get("unit") == name and entry.get("status") == "PASS"
                and entry.get("actual_state_dict_forward_replayed") is True and entry.get("changed_pair_credits") == 0
                and finite(entry.get("maximum_risk_error")) and entry["maximum_risk_error"] <= 2e-6
                and entry.get("contract_sha256") == lock["contract_sha256"]
                and entry.get("qualification_sha256") == qualification_sha
                and entry.get("completion_sha256") == sha256(receipt)
                and entry.get("artifact_sha256") == sha256(artifact), "SUBGROUP_ACTUAL_REPLAY_OR_ARTIFACT_CHANGED")
        done = read(receipt)
        contract = done.get("contract", {})
        require(done.get("status") == "COMPLETE_R6_TARGET_FIT" and done.get("artifact_sha256") == entry["artifact_sha256"]
                and done.get("epochs") == done.get("optimizer_updates") == 80
                and (done.get("arm"), done.get("partition"), done.get("seed")) == (arm, part, seed)
                and contract.get("contract_sha256") == lock["contract_sha256"]
                and contract.get("config_sha256") == lock["config_sha256"] and contract.get("qualification_sha256") == qualification_sha
                and contract.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][f"outer1_inner{part}"]
                and (contract.get("arm"), contract.get("partition"), contract.get("seed")) == (arm, part, seed)
                and done.get("outer_test_used") is False, "SUBGROUP_COMPLETE_RISK_LINEAGE_CHANGED")
        certified[(arm, part, seed)] = dict(receipt=receipt, artifact=artifact, metadata=done)
    references = replay.get("zero_fit_baseline_reference_replays", [])
    expected_refs = {(arm, p, s) for arm in ARMS[:2] for p in PARTITIONS for s in SEEDS}
    ref_ids = [(r.get("arm"), r.get("partition"), r.get("seed")) for r in references]
    require(len(references) == len(set(ref_ids)) == 18 and set(ref_ids) == expected_refs, "SUBGROUP_REQUIRES_EIGHTEEN_REFERENCE_REPLAYS")
    for entry in references:
        arm, part, seed = entry["arm"], entry["partition"], entry["seed"]
        path = private / "references" / (unit_name(arm, part, seed) + ".json")
        done = read(path)
        require(entry.get("status") == "PASS" and entry.get("independent_fixed_anchor_metrics_recomputed") is True
                and entry.get("new_optimizer_fits") == 0 and entry.get("receipt_sha256") == sha256(path)
                and done.get("status") == "R6_BASELINE_REFERENCE" and done.get("contract_sha256") == lock["contract_sha256"]
                and (done.get("arm"), done.get("partition"), done.get("seed")) == (arm, part, seed)
                and done.get("actual_new_optimizer_fits") == done.get("teacher_refits") == 0
                and done.get("prepared_origin", {}).get("prepared_receipt_sha256") == lock["prepared_receipt_sha256"][f"outer1_inner{part}"],
                "SUBGROUP_FIXED_ANCHOR_REFERENCE_CHANGED")
        certified[(arm, part, seed)] = dict(receipt=path, artifact=None, metadata=done)
    prepared = {}
    for part in PARTITIONS:
        stem = f"outer1_inner{part}"
        receipt, artifact = primary / "prepared" / (stem + ".json"), primary / "prepared" / (stem + ".pt")
        require(sha256(receipt) == lock["prepared_receipt_sha256"][stem], "SUBGROUP_PREPARED_RECEIPT_CHANGED")
        done = read(receipt)
        require(done.get("status") == "COMPLETE" and done.get("sha256") == sha256(artifact), "SUBGROUP_PREPARED_ARTIFACT_CHANGED")
        prepared[part] = artifact
    return config, lock, certified, prepared, dict(target_lock_sha256=sha256(private / "targetLOCK.json"),
        qualification_sha256=qualification_sha, replay_sha256=sha256(private / "REPLAY_AUDIT.json"),
        certified_neural_risks=90, certified_zero_fit_references=18,
        no_endpoints_or_risk_arrays_read_before_certificate_checks=True)


def define_groups(rna_finite_mask, cancer, platform_proxy):
    """Outcome-blind group definitions; true platform identity is not available."""
    measured = np.asarray(rna_finite_mask)
    cancer, proxy = np.asarray(cancer).astype(str), np.asarray(platform_proxy)
    require(measured.ndim == 2 and measured.dtype == bool and len(measured) > 0
            and cancer.shape == proxy.shape == (len(measured),) and proxy.dtype.kind in "iu",
            "SUBGROUP_COVARIATE_SHAPE_OR_MASK_INVALID")
    packed = np.packbits(measured, axis=1, bitorder="little")
    unique, first, pattern = np.unique(packed, axis=0, return_index=True, return_inverse=True)
    missing = (~measured).sum(1)
    pattern_missing = missing[first]
    block_candidates = np.flatnonzero(pattern_missing == 3024)
    coverage = np.full(len(measured), "RNA_OTHER_MISSING_PATTERN", dtype=object)
    coverage[missing == 0] = "RNA_FULL_FINITE"
    if len(block_candidates) == 1:
        coverage[pattern == block_candidates[0]] = "RNA_BLOCK_MISSING_3024"
    cancer_labels = np.full(len(measured), "OTHER_CANCER", dtype=object)
    for label in ("COAD", "READ"):
        cancer_labels[np.isin(cancer, (label, "TCGA-" + label))] = label
    proxy_levels = np.unique(proxy)
    proxy_labels = np.asarray(["RAW_RNA_MISSING_PROXY_" + str(int(np.searchsorted(proxy_levels, value))) for value in proxy])
    dimensions = {"RNA_COVERAGE": coverage, "CANCER": cancer_labels, "PLATFORM_MISSINGNESS_PROXY": proxy_labels}
    groups = [("ALL", "ALL", np.ones(len(measured), bool))]
    for dimension, labels in dimensions.items():
        for label in sorted(set(labels)):
            groups.append((dimension, label, labels == label))
    pattern_rows = [dict(pattern_code=int(code), patients=int((pattern == code).sum()), missing_genes=int(pattern_missing[code]))
                    for code in range(len(unique))]
    crosstab = [dict(RNA_coverage=str(coverage_value), platform_proxy=str(proxy_value),
        patients=int(((coverage == coverage_value) & (proxy_labels == proxy_value)).sum()))
        for coverage_value in sorted(set(coverage)) for proxy_value in sorted(set(proxy_labels))]
    one_to_one = all(len(np.unique(proxy_labels[coverage == label])) == 1 for label in np.unique(coverage))
    one_to_one &= all(len(np.unique(coverage[proxy_labels == label])) == 1 for label in np.unique(proxy_labels))
    actual = dict(full_finite_patients=int((coverage == "RNA_FULL_FINITE").sum()),
        block_3024_patients=int((coverage == "RNA_BLOCK_MISSING_3024").sum()),
        other_missing_patients=int((coverage == "RNA_OTHER_MISSING_PATTERN").sum()))
    confirmed = measured.shape == (512, 20511) and actual["full_finite_patients"] == 309 and actual["block_3024_patients"] == 203 and actual["other_missing_patients"] == 0
    return groups, dict(status="CANONICAL_309_203_BLOCK_CONFIRMED" if confirmed else "ACTUAL_MASK_DIFFERS_FROM_NAMED_EXPECTATION",
        cohort_covariate_patients=len(measured), RNA_genes=measured.shape[1], **actual,
        exact_finite_patterns=pattern_rows, platform_proxy_levels=len(proxy_levels),
        platform_is_verified_sequencing_identity=False,
        platform_definition="exact_raw_RNA_missing_pattern_proxy_from_audited_loader",
        RNA_coverage_platform_crosstab=crosstab, coverage_proxy_one_to_one=bool(one_to_one),
        proxy_and_coverage_are_not_independent_effects_when_one_to_one=True,
        group_definitions_use_outcomes=False, full_cohort_endpoint_subgroup_counts_computed=False)


def subgroup_components(train_time, train_event, time, event, risk):
    tt, te, t, e, r = (np.asarray(x) for x in (train_time, train_event, time, event, risk))
    require(t.ndim == 1 and t.shape == e.shape == r.shape and tt.shape == te.shape
            and np.isfinite(t).all() and np.isfinite(tt).all() and np.isfinite(r).all()
            and np.isin(e, (0, 1)).all() and np.isin(te, (0, 1)).all(), "SUBGROUP_METRIC_INPUT_INVALID")
    hn, hd = components(t, e, r)
    uno_status = "AVAILABLE"
    try:
        if np.any((e == 1) & (t < 60.)) and not len(tt):
            raise ValueError("EMPTY_TRAIN_IPCW")
        w = uno_weights(tt, te, t, e, tau=60.) if len(t) else np.empty(0)
        un, ud = components(t, e, r, w)
        if ud == 0:
            uno_status = "NO_WEIGHTED_COMPARABLE_PAIRS"
    except ValueError:
        un = ud = None
        uno_status = "IPCW_TRAINING_SUPPORT_FAILURE"
    return dict(patients=len(t), events=int(e.sum()), censored=int((e == 0).sum()),
        events_before_60=int(((e == 1) & (t < 60.)).sum()),
        Harrell_num=hn, Harrell_den=hd, Uno_num=un, Uno_den=ud,
        Harrell=hn / hd if hd else None, Uno60=un / ud if ud else None,
        Harrell_status="AVAILABLE" if hd else "NO_COMPARABLE_PAIRS", Uno60_status=uno_status,
        IPCW_reference=IPCW_REFERENCE)


def _load_tensor_artifact(path):
    # These are owned private artifacts whose expected SHA was certified above.
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_raw(args):
    from src.bgb_en.data import load_raw
    return load_raw(read(args.assets), include_endpoints=True)


def verify_raw(raw, lock):
    require(all(raw["audit"].get(key) == value for key, value in lock["data_fingerprints"].items()), "SUBGROUP_CANONICAL_DATA_FINGERPRINT_CHANGED")
    mask = np.asarray(raw["rna_measured"])
    require(np.array_equal(mask, np.isfinite(raw["rna"])), "SUBGROUP_ACTUAL_RNA_FINITE_MASK_CHANGED")
    packed = np.packbits(mask, axis=1, bitorder="little")
    digest = hashlib.sha256(str(mask.shape).encode() + packed.tobytes()).hexdigest()
    require(digest == lock["RNA_finite_mask_sha256"], "SUBGROUP_LOCKED_RNA_MASK_SHA_CHANGED")


def verify_scope(train, valid, expected, folds):
    train, valid = np.asarray(train), np.asarray(valid)
    require(train.dtype.kind in "iu" and valid.dtype.kind in "iu" and np.array_equal(train, expected[0])
            and np.array_equal(valid, expected[1]) and not np.intersect1d(train, valid).size
            and len(np.unique(train)) == len(train) and len(np.unique(valid)) == len(valid)
            and np.all(np.asarray(folds)[np.r_[train, valid]] != 1), "SUBGROUP_PRIVATE_RISK_ROW_SCOPE_CHANGED")
    return train.astype(np.int64), valid.astype(np.int64)


def assert_all_group(metrics, expected):
    for key in (*COMPONENTS, "Harrell", "Uno60"):
        a, b = metrics.get(key), expected.get(key)
        require(a is None and b is None or finite(a) and finite(b) and abs(a - b) <= 1e-10,
                "SUBGROUP_ALL_GROUP_INDEPENDENT_REPLAY_DIFFERS")


def rows_for_risk(raw, train, valid, risk, groups, *, arm, partition, seed):
    risk = np.asarray(risk, np.float64)
    require(risk.shape == (len(valid),) and np.isfinite(risk).all(), "SUBGROUP_RISK_VECTOR_SHAPE_INVALID")
    result = []
    for endpoint, time_key, event_key, valid_key in ENDPOINTS:
        eligible = np.asarray(raw[valid_key], bool)
        all_train = train[eligible[train]]
        for dimension, label, group in groups:
            selected = eligible[valid] & group[valid]
            indices = valid[selected]
            metric = subgroup_components(raw[time_key][all_train], raw[event_key][all_train],
                raw[time_key][indices], raw[event_key][indices], risk[selected])
            fit_group = all_train[group[all_train]]
            result.append(dict(arm=arm, partition=partition, seed=seed, endpoint=endpoint,
                group_dimension=dimension, group=label, training_patients_full_IPCW=len(all_train),
                training_events_full_IPCW=int(raw[event_key][all_train].sum()),
                training_patients_in_group=len(fit_group), training_events_in_group=int(raw[event_key][fit_group].sum()), **metric))
    return result


def aggregate_rows(rows):
    keys = sorted({(r["arm"], r["endpoint"], r["group_dimension"], r["group"]) for r in rows})
    per_seed, overall = [], []
    for arm, endpoint, dimension, label in keys:
        seed_rows = []
        for seed in SEEDS:
            selected = [r for r in rows if (r["arm"], r["endpoint"], r["group_dimension"], r["group"], r["seed"])
                        == (arm, endpoint, dimension, label, seed)]
            require(len(selected) == 3 and {r["partition"] for r in selected} == set(PARTITIONS), "SUBGROUP_MISSING_OR_DUPLICATE_PARTITION")
            sums = {key: sum(r[key] for r in selected) if all(r[key] is not None for r in selected) else None for key in COMPONENTS}
            record = dict(arm=arm, endpoint=endpoint, group_dimension=dimension, group=label, seed=seed,
                patients=sum(r["patients"] for r in selected), events=sum(r["events"] for r in selected),
                censored=sum(r["censored"] for r in selected), events_before_60=sum(r["events_before_60"] for r in selected),
                Harrell=sums["Harrell_num"] / sums["Harrell_den"] if sums["Harrell_den"] else None,
                Uno60=sums["Uno_num"] / sums["Uno_den"] if sums["Uno_den"] else None, **sums)
            per_seed.append(record); seed_rows.append(record)
        for key in ("patients", "events", "censored", "events_before_60"):
            require(len({r[key] for r in seed_rows}) == 1, "SUBGROUP_SUPPORT_CHANGED_ACROSS_NEURAL_SEEDS")
        overall.append(dict(arm=arm, endpoint=endpoint, group_dimension=dimension, group=label,
            seeds=3, patients_per_seed=seed_rows[0]["patients"], events_per_seed=seed_rows[0]["events"],
            Harrell=statistics.mean(r["Harrell"] for r in seed_rows) if all(r["Harrell"] is not None for r in seed_rows) else None,
            Uno60=statistics.mean(r["Uno60"] for r in seed_rows) if all(r["Uno60"] is not None for r in seed_rows) else None,
            aggregation="within_partition_num_den_sum_per_seed_then_three_seed_mean",
            no_cross_partition_risk_comparisons=True))
    return per_seed, overall


def run_audit(args):
    config, lock, certified, prepared_paths, certificate = preflight(args.config, args.private_root, args.primary_private)
    raw = _load_raw(args)  # Endpoint access is deliberately below all preflight guards.
    verify_raw(raw, lock)
    from src.bgb_en.protocol import fixed_inner_splits
    splits = fixed_inner_splits(raw["folds"], raw["e"], outer=1)
    groups, covariate_audit = define_groups(raw["rna_measured"], raw["cancer"], raw["platform"])
    all_rows, risk_audit = [], []
    for partition in PARTITIONS:
        prepared_receipt = prepared_paths[partition].with_suffix(".json")
        require(sha256(prepared_receipt) == lock["prepared_receipt_sha256"][f"outer1_inner{partition}"]
                and sha256(prepared_paths[partition]) == read(prepared_receipt)["sha256"],
                "SUBGROUP_PREPARED_CHANGED_AFTER_PREFLIGHT")
        prepared = _load_tensor_artifact(prepared_paths[partition])
        train, valid = verify_scope(prepared["train"], prepared["valid"], splits[partition], raw["folds"])
        anchors = {"B0": np.asarray(prepared["CP"], float)[valid].copy(), "B1": np.asarray(prepared["base"], float)[valid].copy()}
        del prepared
        for arm in ARMS:
            for seed in SEEDS:
                entry = certified[(arm, partition, seed)]
                if arm in anchors:
                    risk = anchors[arm]
                else:
                    require(sha256(entry["artifact"]) == entry["metadata"]["artifact_sha256"],
                            "SUBGROUP_RISK_ARTIFACT_CHANGED_AFTER_PREFLIGHT")
                    artifact = _load_tensor_artifact(entry["artifact"])
                    require(artifact.get("contract") == entry["metadata"]["contract"] and artifact.get("epochs") == artifact.get("optimizer_updates") == 80,
                            "SUBGROUP_SAVED_RISK_CONTRACT_CHANGED")
                    verify_scope(artifact["train"], artifact["valid"], (train, valid), raw["folds"])
                    risk = np.asarray(artifact["predictions"]["validation"]["risk"], float).copy()
                    del artifact
                rows = rows_for_risk(raw, train, valid, risk, groups, arm=arm, partition=partition, seed=seed)
                for endpoint, metric_key in (("DSS", "CP_SCALAR"), ("PFI_frozen_DSS_risk", "PFI_frozen_DSS_risk")):
                    actual = next(r for r in rows if r["endpoint"] == endpoint and r["group_dimension"] == "ALL")
                    assert_all_group(actual, entry["metadata"]["metrics_vs"][metric_key])
                all_rows.extend(rows)
                risk_audit.append(dict(arm=arm, partition=partition, seed=seed, private_risk_reused=True,
                    full_validation_metrics_independently_reproduced=True, new_optimizer_fits=0))
    per_seed, summaries = aggregate_rows(all_rows)
    audit = dict(schema="bgb_en_r6_postfit_subgroup_audit_v1", status="COMPLETE_108_SAVED_RISK_SUBGROUP_AUDITS",
        script_sha256=sha256(__file__), config_sha256=sha256(args.config), contract_sha256=lock["contract_sha256"],
        **certificate, covariates=covariate_audit, risk_units=risk_audit, risk_units_expected=108,
        raw_data_fingerprints=dict(lock["data_fingerprints"]), group_dimensions=["ALL", "RNA_COVERAGE", "CANCER", "PLATFORM_MISSINGNESS_PROXY"],
        primary_endpoint="DSS", secondary_endpoint="PFI_frozen_DSS_risk", time_unit="months", tau=60.,
        IPCW_reference=IPCW_REFERENCE, no_subgroup_censor_model_refit=True,
        IPCW_independently_recomputed_using_same_full_training_partition=True,
        event_counts_and_risks_only_from_outer1_training_population=True, outer_test_outcomes_evaluated=False,
        cohort_counts_use_covariates_only=True, new_optimizer_fits=0, new_optimizer_updates=0,
        model_forward_calls=0, subgroup_or_epoch_or_model_selection=False, bootstrap_or_CI_or_pvalues=False,
        patient_identifiers_risk_vectors_features_or_paths_exported=False, three_seeds_are_not_independent_cohorts=True)
    return audit, all_rows, per_seed, summaries


def write_results(audit, rows, per_seed, summaries, output):
    output = Path(output).resolve()
    protected = (ROOT / "reports/bgb_en", ROOT / "reports/bgb_en_r5", ROOT / "reports/sami_surv_stage_b")
    require(not any(output == p.resolve() or output.is_relative_to(p.resolve()) for p in protected), "SUBGROUP_HISTORICAL_REPORTS_PRESERVED")
    output.mkdir(parents=True, exist_ok=True)
    for filename, values, columns in (
        ("R6_SUBGROUP_METRICS.csv", rows, ["arm", "partition", "seed", "endpoint", "group_dimension", "group", "patients", "events", "censored", "events_before_60",
            "training_patients_full_IPCW", "training_events_full_IPCW", "training_patients_in_group", "training_events_in_group", *COMPONENTS, "Harrell", "Uno60", "Harrell_status", "Uno60_status", "IPCW_reference"]),
        ("R6_SUBGROUP_SEED_SUMMARY.csv", per_seed, ["arm", "seed", "endpoint", "group_dimension", "group", "patients", "events", "censored", "events_before_60", *COMPONENTS, "Harrell", "Uno60"]),
        ("R6_SUBGROUP_SUMMARY.csv", summaries, ["arm", "endpoint", "group_dimension", "group", "seeds", "patients_per_seed", "events_per_seed", "Harrell", "Uno60", "aggregation", "no_cross_partition_risk_comparisons"])):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, columns, extrasaction="ignore"); writer.writeheader(); writer.writerows(values)
    (output / "R6_SUBGROUP_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# R6 事后只读分组审计", "", f"状态：**{audit['status']}**。复用108个已认证风险单元，新增拟合/更新/模型前向均为0。", "",
        "RNA组依据真实有限测量mask。完整309/3024基因块缺失203只在实际矩阵匹配时确认；不按proxy编号猜测。", "",
        "平台字段只是原始RNA缺失pattern代理，未核验真实测序平台；若与RNA覆盖一一对应，两个分组不是独立效应。", "",
        "Harrell和Uno60均仅在同一inner-valid分区、同一组内比较患者。Uno60沿用完整当前训练分区的删失KM，不按分组重估。每seed求三个分区分子/分母之和，再平均三个seed；不跨折排序风险，不将三个seed计作三倍患者。", "",
        "删失机制可能随亚组变化；沿用整体训练KM是在统一协议下描述结果，不代表已经完成组内删失校准或因果调整。", "",
        "空组或无可比较事件不生成虚构的C-index。小亚组事件数有限，分组结果仅作描述；没有CI、p值、重训或事后选择，也不能证明平台因果效应或医学泛化。", "",
        "全512人的协变量人数可以审计；事件计数与风险指标仅来自outer1训练人群，未评估封存outer-test结局。", ""]
    (output / "R6_SUBGROUPS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--primary-private", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bgb_en/r6_inner_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bgb_en_r6")
    args = parser.parse_args()
    result = run_audit(args)
    write_results(*result, args.output)
    print(json.dumps(dict(status=result[0]["status"], risk_units=108, new_optimizer_fits=0)))


if __name__ == "__main__":
    main()

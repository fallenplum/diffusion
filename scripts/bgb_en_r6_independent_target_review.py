"""Independent arithmetic review of anonymous R6 target reports only.

Standard-library implementation; no training/report implementation is imported.
No private artifacts, individual risk vectors, labels or models are accessed.
Execution and patient-pair replay claims remain attestations from the producer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B13")
SEEDS, PARTITIONS = (31, 47, 83), (0, 1, 2)
ENDPOINTS = ("DSS", "PFI_frozen_DSS_risk")
CORE_REFS = ("B1", "B3", "B4", "B5", "B6", "B8", "B9")
PAIR_REFS = CORE_REFS + ("B11", "B13")
COMPONENTS = ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
TOL = 1e-10


def require(value, code):
    if not value:
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b):
    return a is None and b is None or finite(a) and finite(b) and abs(a - b) <= TOL


def number(value):
    if value == "":
        return None
    answer = float(value)
    require(math.isfinite(answer), "ANONYMOUS_CSV_NONFINITE")
    return answer


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(rows, fields, expected, code):
    result = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        require(key in expected and key not in result, code)
        result[key] = row
    require(set(result) == expected, code)
    return result


def ratio_metrics(row):
    require(all(finite(row.get(key)) and row[key] >= 0 for key in COMPONENTS), "CONCORDANCE_COMPONENT_INVALID")
    for name, num, den in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
        require(row[num] <= row[den] + TOL, "CONCORDANCE_NUMERATOR_EXCEEDS_DENOMINATOR")
        value = row[num] / row[den] if row[den] else None
        require(close(row.get(name), value), "WITHIN_PARTITION_RATIO_MISMATCH")


def independently_aggregate(fold_rows):
    expected = {(endpoint, arm, part, seed) for endpoint in ENDPOINTS for arm in ARMS for part in PARTITIONS for seed in SEEDS}
    converted = []
    for raw in fold_rows:
        row = dict(endpoint=raw["endpoint"], arm=raw["arm"], partition=int(raw["partition"]), seed=int(raw["seed"]))
        row.update({key: number(raw[key]) for key in (*COMPONENTS, "Harrell", "Uno60")})
        ratio_metrics(row)
        converted.append(row)
    folds = indexed(converted, ("endpoint", "arm", "partition", "seed"), expected, "COMPLETE_216_FOLD_ROWS_REQUIRED")
    summaries = {}
    for endpoint in ENDPOINTS:
        for arm in ARMS:
            seeds = []
            for seed in SEEDS:
                parts = [folds[(endpoint, arm, part, seed)] for part in PARTITIONS]
                total = {key: sum(row[key] for row in parts) for key in COMPONENTS}
                total.update(seed=seed, Harrell=total["Harrell_num"] / total["Harrell_den"] if total["Harrell_den"] else None,
                             Uno60=total["Uno_num"] / total["Uno_den"] if total["Uno_den"] else None)
                seeds.append(total)
            summaries[(endpoint, arm)] = dict(arm=arm, per_seed=seeds,
                Harrell=statistics.mean(row["Harrell"] for row in seeds) if all(row["Harrell"] is not None for row in seeds) else None,
                Uno60=statistics.mean(row["Uno60"] for row in seeds) if all(row["Uno60"] is not None for row in seeds) else None)
    # The two fixed references are one saved risk vector per partition.
    for endpoint in ENDPOINTS:
        for arm in ARMS[:2]:
            for part in PARTITIONS:
                rows = [folds[(endpoint, arm, part, seed)] for seed in SEEDS]
                require(all(close(row[key], rows[0][key]) for row in rows for key in (*COMPONENTS, "Harrell", "Uno60")),
                        "ZERO_FIT_REFERENCE_SEED_METRICS_CHANGED")
    return folds, summaries


def verify_summaries(report, summaries):
    for endpoint, label in (("DSS", "primary_DSS"), ("PFI_frozen_DSS_risk", "secondary_PFI_frozen_DSS_risk")):
        rows = indexed(report[label], ("arm",), {(arm,) for arm in ARMS}, "TWELVE_ARM_SUMMARIES_REQUIRED")
        for arm in ARMS:
            actual, recomputed = rows[(arm,)], summaries[(endpoint, arm)]
            require(actual.get("complete_cells") == actual.get("expected_cells") == 9
                    and actual.get("status") == "COMPLETE_NINE_CELLS"
                    and actual.get("cross_partition_risk_ranking") is False
                    and actual.get("aggregation") == "within_partition_components_sum_per_seed_then_equal_three_seed_mean",
                    "SUMMARY_COMPLETION_OR_AGGREGATION_CHANGED")
            require(all(close(actual.get(key), recomputed[key]) for key in ("Harrell", "Uno60")), "ARM_SUMMARY_ARITHMETIC_MISMATCH")
            seed_rows = indexed(actual["per_seed"], ("seed",), {(seed,) for seed in SEEDS}, "SUMMARY_SEEDS_CHANGED")
            for row in recomputed["per_seed"]:
                require(all(close(row[key], seed_rows[(row["seed"],)].get(key)) for key in (*COMPONENTS, "Harrell", "Uno60")),
                        "PER_SEED_COMPONENT_AGGREGATION_MISMATCH")


def independently_compare(folds, summaries, pair_rows):
    expected = {("B7", ref, part, seed) for ref in PAIR_REFS for part in PARTITIONS for seed in SEEDS}
    expected |= {("B6", "B5", part, seed) for part in PARTITIONS for seed in SEEDS}
    converted = []
    for raw in pair_rows:
        row = dict(candidate=raw["candidate"], reference=raw["reference"], partition=int(raw["partition"]), seed=int(raw["seed"]))
        require(raw.get("pair_credits_from_actual_replay") == "True", "DIRECT_CREDIT_ACTUAL_REPLAY_ATTESTATION_REQUIRED")
        row.update({key: number(raw[key]) for key in ("Harrell_delta", "Uno60_delta", "direct_repair", "direct_damage",
            "direct_Uno_repair", "direct_Uno_damage", "risk_SD_ratio")})
        a = folds[("DSS", row["candidate"], row["partition"], row["seed"])]
        b = folds[("DSS", row["reference"], row["partition"], row["seed"])]
        require(all(close(a[key], b[key]) for key in ("Harrell_den", "Uno_den")), "PAIRED_PATIENT_SCOPE_DENOMINATORS_CHANGED")
        require(finite(row["risk_SD_ratio"]) and row["risk_SD_ratio"] >= 0, "PAIRED_RISK_SCALE_INVALID")
        for metric, delta, repair, damage in (("Harrell", "Harrell_delta", "direct_repair", "direct_damage"),
            ("Uno60", "Uno60_delta", "direct_Uno_repair", "direct_Uno_damage")):
            require(all(finite(row[key]) and 0 <= row[key] <= 1 for key in (repair, damage)), "PAIR_REPAIR_DAMAGE_INVALID")
            expected_delta = a[metric] - b[metric] if a[metric] is not None and b[metric] is not None else None
            require(close(row[delta], expected_delta) and (expected_delta is None or close(row[repair] - row[damage], expected_delta)),
                    "PAIR_REPAIR_DAMAGE_OR_DELTA_IDENTITY_MISMATCH")
        converted.append(row)
    pairs = indexed(converted, ("candidate", "reference", "partition", "seed"), expected, "COMPLETE_NINETY_DIRECT_PAIR_ROWS_REQUIRED")
    comparisons = {}
    for candidate, reference in [("B7", ref) for ref in PAIR_REFS] + [("B6", "B5")]:
        a, b = summaries[("DSS", candidate)], summaries[("DSS", reference)]
        comparison = dict(candidate=candidate, reference=reference, complete=True, per_seed=[])
        for metric in ("Harrell", "Uno60"):
            comparison[metric + "_delta"] = a[metric] - b[metric] if a[metric] is not None and b[metric] is not None else None
        for ma, mb in zip(a["per_seed"], b["per_seed"]):
            seed = ma["seed"]
            record = dict(seed=seed)
            for metric in ("Harrell", "Uno60"):
                record[metric + "_delta"] = ma[metric] - mb[metric] if ma[metric] is not None and mb[metric] is not None else None
            for key, den in (("direct_repair", "Harrell_den"), ("direct_damage", "Harrell_den"),
                             ("direct_Uno_repair", "Uno_den"), ("direct_Uno_damage", "Uno_den")):
                rows = [(pairs[(candidate, reference, part, seed)], folds[("DSS", candidate, part, seed)][den]) for part in PARTITIONS]
                denominator = sum(weight for _, weight in rows)
                record[key] = sum(row[key] * weight for row, weight in rows) / denominator if denominator else None
            comparison["per_seed"].append(record)
        for key in ("direct_repair", "direct_damage", "direct_Uno_repair", "direct_Uno_damage"):
            values = [row[key] for row in comparison["per_seed"]]
            comparison[key] = statistics.mean(values) if all(value is not None for value in values) else None
        comparison["direct_credit_complete"] = all(comparison[key] is not None for key in
            ("direct_repair", "direct_damage", "direct_Uno_repair", "direct_Uno_damage"))
        comparison["small_positive_delta_below_0_003"] = any(finite(comparison[key]) and 0 < comparison[key] < .003 for key in ("Harrell_delta", "Uno60_delta"))
        comparisons[f"{candidate}_minus_{reference}"] = comparison
    return comparisons


def check_comparisons(reported, recomputed):
    require(set(reported) == set(recomputed), "PAIRED_COMPARISON_SET_CHANGED")
    for name, actual in recomputed.items():
        row = reported[name]
        require(row.get("candidate") == actual["candidate"] and row.get("reference") == actual["reference"]
                and row.get("complete") is True and row.get("direct_credit_complete") == actual["direct_credit_complete"]
                and row.get("small_positive_delta_below_0_003") == actual["small_positive_delta_below_0_003"], "PAIR_SUMMARY_METADATA_CHANGED")
        values = ("Harrell_delta", "Uno60_delta", "direct_repair", "direct_damage", "direct_Uno_repair", "direct_Uno_damage")
        require(all(close(row.get(key), actual[key]) for key in values), "PAIR_SUMMARY_ARITHMETIC_MISMATCH")
        seeds = indexed(row["per_seed"], ("seed",), {(seed,) for seed in SEEDS}, "PAIR_SUMMARY_SEEDS_CHANGED")
        for expected in actual["per_seed"]:
            require(all(close(seeds[(expected["seed"],)].get(key), expected[key]) for key in values), "PAIR_SEED_ARITHMETIC_MISMATCH")


def positive(comparison, *, strict=True):
    h, u = comparison["Harrell_delta"], comparison["Uno60_delta"]
    if not finite(h) or not finite(u):
        return None
    return h > 0 and u > 0 if strict else h >= 0 and u >= 0 and (h > 0 or u > 0)


def independently_decide(source, comparisons):
    primary = {ref: positive(comparisons[f"B7_minus_{ref}"]) for ref in CORE_REFS}
    reverse = positive(comparisons["B7_minus_B11"], strict=False)
    raw, prior_raw = (positive(comparisons[name], strict=False) for name in ("B6_minus_B5", "B7_minus_B5"))
    local = {"SUPPORTED": True, "NOT_SUPPORTED": False, "INCONCLUSIVE": None}[source["status"]]
    raw_flag = None if None in (reverse, raw, prior_raw) else (raw or prior_raw) and reverse
    flags = dict(ST_LOCAL_EVIDENCE_SUPPORTED=local, TRUE_GENE_REREAD_WORKS=raw_flag,
        CORRECT_QUERY_BEATS_WRONG=positive(comparisons["B7_minus_B9"]), DSS_BEATS_CP_SCALAR=positive(comparisons["B7_minus_B1"]))
    known = local is not None and reverse is not None and all(value is not None for value in primary.values())
    supported = known and local is True and all(primary.values()) and reverse
    trigger_values = [positive(comparisons[f"B7_minus_{ref}"]) for ref in ("B3", "B4", "B5", "B6")]
    trigger = None if any(value is None for value in trigger_values) else all(trigger_values)
    optional = {arm: "PENDING" if trigger is None or trigger else "NOT_ACTIVATED" for arm in ("B10", "B12", "B14")}
    optional["B15"] = "TEXT_BLOCKED"
    return dict(decision="INCONCLUSIVE" if not known else "INNER_GATE_PASSED_OPTIONAL_PENDING" if supported else "NOT_SUPPORTED",
        core_flags=flags, complete_source_and_target_evidence=True, preliminary_outer_criteria_met=bool(supported),
        outer_start_allowed=False, optional_activation_trigger=trigger, optional_followup=optional,
        actual_optional_followup_required_before_external_stage=bool(supported), formal_target_matrix_denominator=90,
        optional_not_counted_as_failure=True, strict_primary_comparisons=primary, reverse_reread_condition=reverse,
        raw_reread_attribution=dict(strict_same_source_no_prior_B6_vs_B5=raw, B7_vs_B5_also_changes_source_prior=prior_raw, reverse_B7_vs_B11=reverse),
        wrong_ST_is_separate_from_wrong_patient_query=True, PFI_used_for_selection=False, CI_or_pvalues_computed=False,
        biological_or_causal_mechanism_proven=False, all_medical_ideas_rejected=False,
        source_or_numeric_failure_is_not_performance_negative=True, historical_outer_is_new_independent_validation=False)


def execution_complete(report):
    expected = {(arm, part, seed) for arm in ARMS for part in PARTITIONS for seed in SEEDS}
    ledger = indexed(report.get("ledger", []), ("arm", "partition", "seed"), expected, "COMPLETE_108_LEDGER_IDENTITIES_REQUIRED")
    neural = [row for row in ledger.values() if row["arm"] not in ARMS[:2]]
    refs = [row for row in ledger.values() if row["arm"] in ARMS[:2]]
    complete = (all(row.get("status") == "PASS_FIT_AND_ACTUAL_REPLAY" for row in neural)
        and all(row.get("status") == "PASS_ZERO_FIT_REFERENCE" for row in refs)
        and report.get("actual_replay", {}).get("complete") is True
        and report.get("actual_replay", {}).get("status") == "PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY"
        and report.get("qualification", {}).get("passed") is True
        and report.get("source", {}).get("complete_source_and_full_transfer") is True)
    if not complete:
        return False
    require(all(row.get("actual_optimizer_updates") == 80 and row.get("actual_new_optimizer_fits") == 1
                and row.get("expected_updates") == 80 and row.get("group") == "NEURAL" and row.get("actual_replay_passed") is True for row in neural)
            and all(row.get("actual_optimizer_updates") == row.get("actual_new_optimizer_fits") == row.get("expected_updates") == 0
                and row.get("group") == "REFERENCE" and row.get("actual_replay_passed") is True for row in refs), "FORMAL_FIT_UPDATE_OR_REFERENCE_BUDGET_CHANGED")
    require(report.get("planned_new_target_fits") == report.get("confirmed_new_target_fits") == 90
            and report.get("planned_baseline_reference_rows") == 18
            and report.get("planned_target_updates") == report.get("confirmed_target_updates") == 7200
            and report.get("unknown_cost_units") == 0, "TARGET_90_FITS_7200_UPDATES_REQUIRED")
    require(report.get("counts") == {"PASS_FIT_AND_ACTUAL_REPLAY": 90, "PASS_ZERO_FIT_REFERENCE": 18}, "TARGET_STATUS_COUNTS_CHANGED")
    pair_audit = report.get("direct_pair_audit", {})
    require(pair_audit.get("expected_rows") == pair_audit.get("valid_rows") == 90 and pair_audit.get("missing_rows") == 0
            and pair_audit.get("validation_failures") == [] and pair_audit.get("complete") is True, "COMPLETE_DIRECT_REPLAY_PAIR_AUDIT_REQUIRED")
    return True


def review(report_dir, source_decision_path, config_path):
    directory = Path(report_dir)
    inputs = {name: directory / name for name in ("R6_4_INNER_MATRIX.json", "R6_SEED_FOLD.csv", "R6_MECHANISM_ABLATIONS.csv", "decision.json")}
    report, source, config = read(inputs["R6_4_INNER_MATRIX.json"]), read(source_decision_path), read(config_path)
    hashes = {name: sha256(path) for name, path in inputs.items()}
    hashes.update(source_decision_sha256=sha256(source_decision_path), config_sha256=sha256(config_path))
    require(report.get("schema") == "bgb_en_r6_target_inner_report_v1" and report.get("config_sha256") == hashes["config_sha256"], "ANONYMOUS_REPORT_CONFIG_MISMATCH")
    require(config.get("schema") == "bgb_en_r6_inner_v1" and tuple(config.get("trainable_arms", [])) == ARMS[2:]
            and config["outer_gate"]["B7_both_metrics_exceed"] == list(CORE_REFS)
            and config["outer_gate"]["reverse_reread"] == "B7_minus_B11_one_metric_positive_other_nonnegative"
            and config["prototype"]["inner_partitions"] == list(PARTITIONS) and config["prototype"]["neural_seeds"] == list(SEEDS)
            and config["primary_endpoint"] == "DSS" and config["secondary_endpoint"] == "PFI_frozen_DSS_risk_only", "LOCKED_SELECTION_RULE_CHANGED")
    require(read(inputs["decision.json"]) == report.get("decision"), "DECISION_FILE_DISAGREES_WITH_REPORT")
    require(source.get("status") in ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE") and source.get("source_mechanism_only") is True
            and report["source"].get("decision_sha256") == hashes["source_decision_sha256"]
            and report["source"].get("status") == source["status"], "SOURCE_DECISION_SCOPE_OR_HASH_MISMATCH")
    require(report.get("no_cross_partition_risk_ranking") is True and report.get("seed_means_are_not_independent_cohorts") is True
            and report.get("no_epoch_selection") is True and report.get("no_model_or_array_deserialization") is True
            and report.get("no_raw_patient_rows_read") is True and report.get("repair_damage_never_invented_from_Cindex_differences") is True,
            "ANONYMOUS_REPORT_EVALUATION_BOUNDARY_CHANGED")
    require(report.get("outer_results", {}).get("status") == "NOT_RUN" and report["outer_results"].get("Harrell") is None
            and report["outer_results"].get("Uno60") is None and report["decision"].get("outer_start_allowed") is False,
            "OUTER_NOT_RUN_BOUNDARY_CHANGED")
    base = dict(schema="bgb_en_r6_independent_anonymous_target_review_v1", reviewer_sha256=sha256(__file__), input_sha256=hashes,
        private_artifacts_labels_models_or_individual_risks_read=False, new_optimizer_fits=0, model_forward_calls=0,
        CI_pvalues_or_model_epoch_selection=False, cross_partition_risk_ranking=False, outer_results="NOT_RUN",
        underlying_model_replay_not_reexecuted=True, original_label_correctness_or_medical_truth_independently_verified=False)
    if not execution_complete(report):
        return dict(**base, status="INCONCLUSIVE_METADATA_INCOMPLETE", overall_winner=None, comparisons=[],
                    reason="COMPLETE_90_ACTUAL_REPLAYS_18_REFERENCES_AND_FULL_SOURCE_REQUIRED")
    folds, summaries = independently_aggregate(csv_rows(inputs["R6_SEED_FOLD.csv"]))
    verify_summaries(report, summaries)
    comparisons = independently_compare(folds, summaries, csv_rows(inputs["R6_MECHANISM_ABLATIONS.csv"]))
    check_comparisons(report["paired_comparisons"], comparisons)
    decision = independently_decide(source, comparisons)
    require(decision == report["decision"], "INDEPENDENT_DSS_SELECTION_OR_MECHANISM_GATE_MISMATCH")
    require(report["source"].get("local_support") is decision["core_flags"]["ST_LOCAL_EVIDENCE_SUPPORTED"], "SOURCE_LOCAL_SUPPORT_FLAG_CHANGED")
    return dict(**base, status="PASS_ANONYMOUS_ARITHMETIC_AND_CONTRACT_REVIEW", overall_winner=None,
        computation=dict(logical_rows=108, actual_neural_replays=90, zero_fit_reference_replays=18, formal_target_optimizer_updates=7200,
            endpoints=2, arms=12, fold_metric_rows=216, direct_credit_rows=90, per_seed_components_independently_rebuilt=72,
            aggregation="sum_three_within_partition_num_den_per_seed_then_equal_three_seed_mean", completed_no_partial_mean=True),
        arithmetic_summaries=[dict(endpoint=endpoint, **summary) for (endpoint, _), summary in summaries.items()],
        comparisons=comparisons, independently_rebuilt_decision=decision,
        clinical_boundary=dict(primary="DSS", secondary="PFI_frozen_DSS_risk", PFI_used_for_selection=False,
            repeated_seeds_are_not_independent_cohorts=True, no_new_outer_validation=True, no_confidence_or_causal_claim=True),
        mechanism_boundary=dict(core_flags=decision["core_flags"], strict_same_source_no_prior_raw_gene_comparison="B6_minus_B5",
            B7_minus_B5_also_changes_source_prior=True, wrong_ST_and_wrong_patient_query_are_distinct=True,
            direct_credits_reconciled_but_original_patient_pairs_not_reexecuted=True,
            biological_or_cell_source_ground_truth_proven=False, negative_gate_does_not_reject_all_medical_ideas=True))


def write_results(result, output):
    output = Path(output).resolve()
    require(not any(output == path.resolve() or output.is_relative_to(path.resolve()) for path in
                    (ROOT / "reports/bgb_en", ROOT / "reports/bgb_en_r5", ROOT / "reports/sami_surv_stage_b")), "HISTORICAL_REPORT_NAMESPACES_PRESERVED")
    output.mkdir(parents=True, exist_ok=True)
    (output / "R6_TARGET_INDEPENDENT_REVIEW.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# R6 匿名目标结果独立复核", "", f"状态：**{result['status']}**。", "", "## 计算复核", "",
        "本脚本独立实现折内分子/分母求和与三seed均值，只读匿名JSON/CSV，不读取患者风险、标签、模型或私有目录。"]
    if result["status"] == "PASS_ANONYMOUS_ARITHMETIC_AND_CONTRACT_REVIEW":
        lines += ["", "12臂×3seed×3分区的DSS/PFI共216行逐项一致；90次神经拟合实际重放、18行零拟合参考及7200更新的匿名账本一致。90行真实重放配对信用与其分母加权聚合一致。",
            "", "## 临床边界", "", "DSS单独决定预锁门槛；PFI仅复算，不参与选择。三个seed来自同一队列，当前只有内层证据，外层为NOT_RUN。没有置信区间、p值、epoch或新模型选择。",
            "", "## 机制边界", "", f"独立重建决策：**{result['independently_rebuilt_decision']['decision']}**。",
            "B6−B5才是同源、无新增先验的原基因重读对照；B7−B5还改变源先验。错ST来源与错患者Query分别解释。通过算术复核不等于已证明细胞来源、医学真值或因果机制。"]
    else:
        lines += ["", "必要执行/重放资料未完整，未输出总体胜者或性能比较。"]
    lines += ["", "## 优势与局限", "", "优势：独立标准库实现可发现聚合、配对信用、预算与选择规则的报表不一致，且不会干扰训练。",
        "局限：模型重放和患者配对仍依赖上游认证；本脚本无法从匿名汇总重新认证原始标签、患者身份、冻结权重或实际医学对应关系。输入SHA仅绑定本次审阅版本。", ""]
    (output / "R6_TARGET_INDEPENDENT_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--source-decision", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bgb_en/r6_inner_v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = review(args.report_dir, args.source_decision, args.config)
    write_results(result, args.output or args.report_dir)
    print(json.dumps(dict(status=result["status"], outer="NOT_RUN", new_optimizer_fits=0)))


if __name__ == "__main__":
    main()

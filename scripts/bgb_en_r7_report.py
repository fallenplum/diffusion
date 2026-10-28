"""R7 anonymous report from complete receipts and independent actual replay.

No patient arrays, weights, outcomes, or risk vectors are loaded. Concordance
components are summed inside each seed; risk scores never cross partitions.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bgb_en_r6_report_target import (canonical_hash, sha256, validate_metrics,
    combine_components, aggregate_arm, require, read, write_json, write_csv)

ARMS = ("ADD", "RAW", "NORM", "INDEP", "SLOT", "GATE1", "GATE2",
        "SLOT_WRONG_RNA", "GATE1_WRONG_RNA", "GATE1_WRONG_PATH", "GATE2_WRONG_RNA")
PARTITIONS, SEEDS = (0, 1, 2), (31, 47, 83)
FIELDS = ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den")


def difference(a, b):
    return None if a is None or b is None else a - b


def optional_mean(values):
    return None if any(v is None for v in values) else statistics.mean(values)


def formatted(value, signed=False):
    return "NA" if value is None else format(value, "+.9f" if signed else ".9f")


def pair_summary(candidate, reference, summaries, rows):
    result = dict(candidate=candidate, reference=reference, per_seed=[],
                  repair_damage_are_normalized_pair_credits_not_unique_patient_pair_counts=True)
    a, b = summaries[candidate], summaries[reference]
    require(a["complete_cells"] == b["complete_cells"] == 9, "R7_INCOMPLETE_PAIRED_ARM")
    expected = {(p, s) for p in PARTITIONS for s in SEEDS}
    require(len(rows) == 9 and {(r["partition"], r["seed"]) for r in rows} == expected,
            "R7_ALL_NINE_DIRECT_RISK_PAIR_RECORDS_REQUIRED")
    result.update(delta_Harrell=difference(a["Harrell"], b["Harrell"]),
                  delta_Uno60=difference(a["Uno60"], b["Uno60"]))
    for seed in SEEDS:
        source = [r["candidate_vs_reference"] for r in rows if r["seed"] == seed]
        component = combine_components(source)
        direct = {}
        for label, denominator in (("repair", "Harrell_den"), ("damage", "Harrell_den"),
                                    ("Uno_repair", "Uno_den"), ("Uno_damage", "Uno_den")):
            den = sum(x[denominator] for x in source)
            require(all(x.get(label) is None if x[denominator] == 0 else
                        isinstance(x.get(label), (float, int)) and math.isfinite(x[label])
                        and 0 <= x[label] <= 1 for x in source), "R7_INVALID_NORMALIZED_PAIR_CREDIT")
            direct[label] = sum(x[label] * x[denominator] for x in source if x[denominator] > 0) / den if den else None
        sa = next(r for r in a["per_seed"] if r["seed"] == seed)
        sb = next(r for r in b["per_seed"] if r["seed"] == seed)
        dh, du = difference(sa["Harrell"], sb["Harrell"]), difference(sa["Uno60"], sb["Uno60"])
        require(dh is None if direct["repair"] is None else
                dh is not None and abs(direct["repair"] - direct["damage"] - dh) <= 1e-10,
                "R7_DIRECT_HARRELL_REPAIR_DAMAGE_DOES_NOT_MATCH_PAIRED_DELTA")
        require(du is None if direct["Uno_repair"] is None else
                du is not None and abs(direct["Uno_repair"] - direct["Uno_damage"] - du) <= 1e-10,
                "R7_DIRECT_UNO_REPAIR_DAMAGE_DOES_NOT_MATCH_PAIRED_DELTA")
        result["per_seed"].append(dict(seed=seed, delta_Harrell=dh, delta_Uno60=du, **direct))
    for label in ("repair", "damage", "Uno_repair", "Uno_damage"):
        result[label] = optional_mean([r[label] for r in result["per_seed"]])
    h, u = result["delta_Harrell"], result["delta_Uno60"]
    known = h is not None and u is not None
    result["metric_status"] = "DEFINED" if known else "INCONCLUSIVE_UNDEFINED_METRIC"
    result["joint_directional_support"] = known and h > 0 and u > 0
    result["mixed_metric_direction"] = known and (h > 0) != (u > 0)
    result["small_delta_is_weak_evidence"] = known and (abs(h) < .003 or abs(u) < .003)
    result["positive_both_seed_count"] = sum(r["delta_Harrell"] is not None and r["delta_Uno60"] is not None
                                              and r["delta_Harrell"] > 0 and r["delta_Uno60"] > 0
                                              for r in result["per_seed"])
    return result


def validate_replay(replay, lock):
    require(replay.get("status") == "PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY"
            and replay.get("contract_sha256") == lock["contract_sha256"]
            and replay.get("expected_units") == replay.get("available_units") == replay.get("passed_units") == 99
            and replay.get("failed_units") == 0
            and replay.get("expected_checkpoints") == replay.get("passed_checkpoints") == 297
            and not replay.get("missing_or_unreplayed_units"), "R7_COMPLETE_297_CHECKPOINT_REPLAY_REQUIRED")
    expected = {(a, p, s) for a in ARMS for p in PARTITIONS for s in SEEDS}
    rows = replay["units"]
    require(len(rows) == 99 and {(r["arm"], r["partition"], r["seed"]) for r in rows} == expected,
            "R7_REPLAY_MATRIX_CHANGED")
    for row in rows:
        require(row.get("status") == "PASS" and row.get("all_checkpoints_actual_forward_replayed") is True
                and row.get("actual_checkpoint_count") == 3 and row.get("changed_pair_credits") == 0
                and row.get("source_frozen_weight_hash_replayed") is True
                and row.get("saved_query_mapping_recomputed") is True
                and row.get("new_optimizer_updates") == 0,
                "R7_UNIT_NOT_THREE_ACTUAL_REPLAYS")
        cps = row.get("checkpoints", [])
        require([c["epoch"] for c in cps] == [20, 40, 80] and all(
            c.get("status") == "PASS" and c.get("actual_state_dict_forward_replayed") is True
            and c.get("new_optimizer_updates") == 0 and c.get("changed_pair_credits") == 0
            and 0 <= c.get("maximum_risk_error", float("inf")) <= 2e-6 for c in cps),
            "R7_SAVED_CHECKPOINT_NOT_ACTUALLY_REPLAYED")
    return {r["unit"]: r for r in rows}


def build(private, config_path, output):
    config, lock = read(config_path), read(private / "targetLOCK.json")
    require(config.get("schema") == "bgb_en_r7_inner_v1" and tuple(config["trainable_arms"]) == ARMS
            and config["formal_target_neural_fits"] == 99 and config["primary_candidate"] == "GATE1",
            "R7_FIXED_PROTOCOL_CHANGED")
    require(lock.get("status") == "LOCKED_BEFORE_R7_FITS" and lock["config_sha256"] == sha256(config_path)
            and canonical_hash({k: v for k, v in lock.items() if k != "contract_sha256"}) == lock["contract_sha256"],
            "R7_SOURCE_CONFIG_LOCK_CHANGED")
    require("scripts/bgb_en_r7_report.py" in lock.get("sources", {})
            and lock.get("planned_neural_fits") == 99 and lock.get("teacher_refits") == 0
            and lock.get("source_refits") == 0 and lock.get("outer_test_access") is False,
            "R7_SCIENTIFIC_BUDGET_OR_SOURCE_LOCK_MISSING")
    for name, digest in lock["sources"].items():
        require((ROOT / name).resolve().is_relative_to(ROOT) and sha256(ROOT / name) == digest,
                "R7_SCIENTIFIC_SOURCE_BYTES_CHANGED")
    qpath = private / "QUALIFICATION.json"
    qualification = read(qpath)
    require(qualification["status"] == "PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER"
            and qualification["contract_sha256"] == lock["contract_sha256"], "R7_QUALIFICATION_CHANGED")
    rpath = private / "REPLAY_AUDIT.json"
    replay = read(rpath)
    replay_by_name = validate_replay(replay, lock)
    require(replay["qualification_sha256"] == sha256(qpath), "R7_REPLAY_QUALIFICATION_SHA_CHANGED")
    metrics, pfi, fold_rows, curves, costs, diagnostic_parts = {}, {}, [], [], [], []
    expected_names = {f"{a}_p{p}_seed{s}" for a in ARMS for p in PARTITIONS for s in SEEDS}
    found = {p.parent.name for p in (private / "cells").glob("*/COMPLETE.json")}
    require(found == expected_names, "R7_ACTUAL_FIT_INVENTORY_INCOMPLETE_OR_UNKNOWN")
    output.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for part in PARTITIONS:
            diagnostic_rows = []
            for seed in SEEDS:
                unit = f"{arm}_p{part}_seed{seed}"
                directory = private / "cells" / unit
                path, artifact = directory / "COMPLETE.json", directory / "model.pt"
                cell, rp = read(path), replay_by_name[unit]
                c = cell["contract"]
                require(cell["status"] == "COMPLETE_R7_TARGET_FIT" and
                    (cell["arm"], cell["partition"], cell["seed"]) == (arm, part, seed)
                    and c["contract_sha256"] == lock["contract_sha256"] and c["qualification_sha256"] == sha256(qpath)
                    and c["config_sha256"] == sha256(config_path)
                    and cell["epochs"] == cell["optimizer_updates"] == c["epochs"] == 80
                    and cell["actual_new_optimizer_fits"] == 1
                    and cell["teacher_refits"] == cell["source_training_updates"] == 0
                    and cell["outer_test_used"] is False, "R7_COMPLETE_CELL_CONTRACT_CHANGED")
                require(not (directory / "ERROR_RECEIPT.json").exists() and sha256(path) == rp["completion_sha256"]
                    and sha256(artifact) == rp["artifact_sha256"] == cell["artifact_sha256"], "R7_CELL_REPLAY_SHA_CHANGED")
                for epoch in (20, 40, 80):
                    cp = next(x for x in rp["checkpoints"] if x["epoch"] == epoch)
                    require(sha256(directory / f"checkpoint_epoch{epoch}.pt") ==
                            cell["checkpoint_artifact_sha256"][str(epoch)] == cp["checkpoint_sha256"],
                            "R7_CHECKPOINT_SHA_CHANGED")
                require([x["epoch"] for x in cell["curves"]] == [20, 40, 80]
                    and set(cell["checkpoint_diagnostics"]) == {"20", "40", "60", "80"}, "R7_BUDGET_DIAGNOSTICS_INCOMPLETE")
                for epoch in (20, 40, 60, 80):
                    require(read(directory / f"DIAGNOSTIC_epoch{epoch}.json") == cell["checkpoint_diagnostics"][str(epoch)],
                            "R7_DIAGNOSTIC_RECEIPT_CHANGED")
                m = validate_metrics(cell["metrics"])
                require(m == cell["metrics_vs"]["CP_SCALAR"], "R7_PRIMARY_METRIC_FIELDS_DISAGREE")
                metrics[(arm, part, seed)] = m
                pfi[(arm, part, seed)] = validate_metrics(cell["metrics_vs"]["PFI_frozen_DSS_risk"])
                fold_rows.append(dict(arm=arm, partition=part, seed=seed, **{k: m[k] for k in FIELDS}))
                costs.append(dict(arm=arm, partition=part, seed=seed, fits=1, updates=80,
                    trainable_parameters=cell["parameters"]["requires_grad"], registered_parameters=cell["parameters"]["registered"],
                    elapsed_seconds=cell["elapsed_seconds"], peak_GPU_bytes=cell["peak_GPU_bytes"]))
                for curve in cell["curves"]:
                    curves.append(dict(arm=arm, partition=part, seed=seed, epoch=curve["epoch"],
                        cox_loss=curve["cox_loss"], parameter_regularizer=curve["parameter_regularizer"],
                        train_Harrell=curve["train"]["Harrell"], valid_Harrell=curve["valid"]["Harrell"],
                        train_Uno60=curve["train"]["Uno60"], valid_Uno60=curve["valid"]["Uno60"],
                        primary_checkpoint=curve["epoch"] == 80))
                diagnostic_rows.append(dict(arm=arm, partition=part, seed=seed,
                    checkpoint_diagnostics=cell["checkpoint_diagnostics"],
                    final_risk_diagnostics=cell["risk_diagnostics"], frozen_controls=cell["frozen_controls"]))
            name = f"R7_DIAGNOSTICS_{arm}_p{part}.json"
            blob = json.dumps(dict(schema="bgb_en_r7_diagnostic_shard_v1", rows=diagnostic_rows),
                              ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            require(len(blob) <= 2 * 1024**2, "R7_DIAGNOSTIC_SHARD_EXCEEDS_PUBLIC_LIMIT")
            (output / name).write_bytes(blob)
            diagnostic_parts.append(dict(file=name, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), rows=3))
    refs = replay["zero_fit_baseline_reference_replays"]
    require(len(refs) == 18 and {(r["arm"], r["partition"], r["seed"]) for r in refs} ==
        {(a, p, s) for a in ("B0", "B1") for p in PARTITIONS for s in SEEDS}, "R7_18_REFERENCES_REQUIRED")
    for row in refs:
        path = private / "references" / f"{row['arm']}_p{row['partition']}_seed{row['seed']}.json"
        ref = read(path)
        require(row["status"] == "PASS" and sha256(path) == row["receipt_sha256"]
            and ref["actual_new_optimizer_fits"] == 0 and ref["contract_sha256"] == lock["contract_sha256"],
            "R7_FIXED_REFERENCE_CHANGED")
        key = (row["arm"], row["partition"], row["seed"])
        metrics[key] = validate_metrics(ref["metrics"])
        pfi[key] = validate_metrics(ref["metrics_vs"]["PFI_frozen_DSS_risk"])
        fold_rows.append(dict(arm=key[0], partition=key[1], seed=key[2], **{k: metrics[key][k] for k in FIELDS}))
    summaries = {arm: aggregate_arm(arm, metrics) for arm in ("B0", "B1") + ARMS}
    secondary = {arm: aggregate_arm(arm, pfi) for arm in ("B0", "B1") + ARMS}
    raw_pairs = replay["paired_retrained_controls"]
    require(len(raw_pairs) == len(config["paired_comparisons"]) * 9 and all(
        r["direct_risk_pair_credits"] is True and r["primary_checkpoint_epoch"] == 80 for r in raw_pairs),
        "R7_ACTUAL_DIRECT_PAIR_CREDITS_REQUIRED")
    pairs = []
    for item in config["paired_comparisons"]:
        source = [r for r in raw_pairs if r["candidate"] == item["candidate"] and r["reference"] == item["reference"]]
        pairs.append(dict(question=item["question"], **pair_summary(item["candidate"], item["reference"], summaries, source)))
    gate_refs = ("ADD", "B1", "GATE1_WRONG_RNA", "GATE1_WRONG_PATH")
    gate = [next(x for x in pairs if x["candidate"] == "GATE1" and x["reference"] == ref) for ref in gate_refs]
    directional = all(x["joint_directional_support"] for x in gate)
    defined = all(x["metric_status"] == "DEFINED" for x in gate)
    decision = dict(status="INCONCLUSIVE_UNDEFINED_METRIC" if not defined else
        "DIRECTIONAL_PATIENT_CONTEXT_INCREMENT" if directional else "CURRENT_GATE1_NOT_SUPPORTED",
        primary_candidate="GATE1", GATE2_selected_from_validation=False, complete_formal_fits=99,
        complete_actual_checkpoint_replays=297, outer_results="NOT_RUN", CI_or_pvalues_computed=False,
        PFI_used_for_selection=False, all_medical_ideas_rejected=False,
        causal_biology_or_cell_source_truth_proven=False, independent_external_validation=False,
        gate_comparisons=gate, all_99_executed_without_performance_adaptive_budget=True)
    result = dict(schema="bgb_en_r7_complete_report_v1", status="COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS",
        contract_sha256=lock["contract_sha256"], config_sha256=sha256(config_path),
        qualification_sha256=sha256(qpath), replay_sha256=sha256(rpath), primary_DSS=summaries,
        secondary_PFI_frozen_DSS_risk=secondary, paired_comparisons=pairs, decision=decision,
        budget=dict(formal_neural_fits=99, formal_optimizer_updates=7920, source_refits=0,
                    teacher_or_beta_refits=0, unsupervised_train_RMS_calibrations=9,
                    zero_fit_baseline_reference_rows=18, diagnostics_optimizer_updates=0),
        no_cross_partition_risk_ranking=True, seed_means_are_not_independent_cohorts=True,
        diagnostic_index=dict(parts=diagnostic_parts, logical_cells=99, epochs_per_cell=[20,40,60,80]))
    write_json(output / "R7_RESULTS.json", result)
    write_json(output / "decision.json", decision)
    write_csv(output / "R7_FOLD_METRICS.csv", fold_rows, ["arm", "partition", "seed", *FIELDS])
    write_csv(output / "R7_COSTS.csv", costs, list(costs[0]))
    write_csv(output / "R7_LEARNING_CURVES.csv", curves, list(curves[0]))
    flat = [{k:v for k,v in row.items() if k not in ("per_seed",)} for row in pairs]
    write_csv(output / "R7_PAIRED_COMPARISONS.csv", flat, list(flat[0]))
    lines = ["# R7 预锁机制瓶颈审计结果", "", "状态：全部99个正式拟合与297份检查点实际回放完成。",
        "仅为既有外折训练侧的开发比较；三个seed不是独立队列。固定80轮，不选择20/40轮。", "",
        "| 模型 | DSS Harrell | DSS Uno60 |", "|---|---:|---:|"]
    for arm, row in summaries.items():
        lines.append(f"| {arm} | {formatted(row['Harrell'])} | {formatted(row['Uno60'])} |")
    lines += ["", "## 逐阶段配对", "", "| 候选−对照 | ΔHarrell | ΔUno60 | 双正seed/3 |", "|---|---:|---:|---:|"]
    for row in pairs:
        lines.append(f"| {row['candidate']}−{row['reference']} | {formatted(row['delta_Harrell'], True)} | {formatted(row['delta_Uno60'], True)} | {row['positive_both_seed_count']} |")
    lines += ["", "## 判断与边界", "", f"固定主候选GATE1：`{decision['status']}`。",
        "只给方向性判断，不计算CI/p；小于0.003的差异仍为弱线索。正确/错查询的冻结响应与真实重训分开。",
        "独立反向相对零查询仍可能携带剩余主效应，不等于统计纯交互；slot自由度与固定L2几何需同时解释。",
        "源六供体已被开发复用，来源与TCGA患者互斥未证实。注意力和null/gate不是组织学或细胞来源真值。",
        "本轮不启动外层或扩展架构。PFI仅使用固定DSS风险。", "",
        "优势是固定强主效应、同患者折、原基因值与mask保留及完整实际回放；风险是有限事件、开发集复用和源目标域差异。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--private", "--private-root", dest="private", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.private, args.config, args.output)
    print(json.dumps(dict(status=result["status"], decision=result["decision"]["status"])))


if __name__ == "__main__":
    main()

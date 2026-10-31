"""Append factual P2 research conclusions to independently verified public tables.

Consumes anonymous report files only; no training, endpoint or private model read.
The original report index, engineering decision and RESULTS remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/semantic_b0b7_p2"
MASTER = "b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"


def read(name):
    return json.loads((REPORT / name).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = REPORT / name
    assert not path.exists(), "Append-only output already exists: " + name
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
           if isinstance(value, dict) else value)
    path.write_bytes(raw.encode("utf-8"))


def main():
    assert sha(ROOT / "configs/semantic_b0b7/p2_lock.json") == MASTER
    for name, digest in json.loads((ROOT / "configs/semantic_b0b7/p2_lock.json").read_text())["frozen_sources"].items():
        assert sha(ROOT / name) == digest
    engineering = read("11_decision.json")
    assert engineering["status"] == "COMPLETE" and not engineering["confidence_intervals_computed"]
    audits = [read(f"INDEPENDENT_P2_OUTER{i}_AUDIT.json") for i in range(1, 6)]
    assert all(x["status"] == "PASS_INDEPENDENT_SAVED_RISK_SCORER_AND_HASH_AUDIT" for x in audits)
    independent = audits[0]["global_summary"]
    metrics = {x["arm"]: x for x in independent["seed_aggregates"]}
    pairs = read("06_paired_comparisons.json")
    assert len(pairs) == 12
    lookup = {x["comparison"]: x for x in pairs}
    assert lookup["B5-B0"]["Harrell_delta"] < 0 and lookup["B5-B0"]["Uno60_delta"] < 0
    budget, cost = read("09_TRAINING_BUDGET_AUDIT.json"), read("09_cost_ledger.json")
    inner = [x for x in budget["family_scope"] if x["role"] == "inner"]
    main_scope = next(x for x in inner if x["family"] == "MAIN")
    assert sum(x["fits"] for x in inner) == 360
    f1 = read("03_LIVE_PHASE_A_HELDOUT_MUTATION_AUDIT.json")
    isolation = read("03_LIVE_DSS_PIPELINE_LABEL_ISOLATION_AUDIT.json")
    assert f1["status"] == "PASS_REAL_TWO_FIT_PHASE_A_HELDOUT_MUTATION_INVARIANCE"
    assert isolation["status"] == "PASS_REAL_HELDOUT_DSS_INVARIANT_LEARNED_PIPELINE"
    assert f1["checkpoint_comparison"]["entire_decoded_checkpoint_exact"]
    assert all(x["exact"] for x in isolation["exact_components"].values())
    history = read("00_R4_EXACT_HISTORICAL_RISK_AUDIT.json")
    assert all(x["canonical_rows_and_source_contract_qualified"] and x["RNA512_selected_member_array_exact"] for x in history["folds"])
    assert all(x["historical_seed31_unified_metrics"] == x["current_B0_unified_metrics"] for x in history["folds"])
    inputs = ["11_decision.json", "06_paired_comparisons.json", "09_cost_ledger.json",
              "09_TRAINING_BUDGET_AUDIT.json", "03_LIVE_PHASE_A_HELDOUT_MUTATION_AUDIT.json",
              "03_LIVE_DSS_PIPELINE_LABEL_ISOLATION_AUDIT.json", "00_R4_EXACT_HISTORICAL_RISK_AUDIT.json",
              "00_R4_PRECISION_LAYOUT_DIAGNOSTIC.json"] + [f"INDEPENDENT_P2_OUTER{i}_AUDIT.json" for i in range(1, 6)]
    boundary = {
        "inner_fits": 360,
        "raw_Huber_unique_minimum_at400": sum(x["raw_Huber_unique_minimum_at400_count"] for x in inner),
        "selected_epoch400": sum(x["selected_epoch_at400_count"] for x in inner),
        "validation_Huber_improved200to400": sum(x["val_patient_equal_masked_huber_200to400"]["improved_direction_count"] for x in inner),
        "validation_Huber_improvement_exceeds_original_band": sum(x["val_Huber200to400_improvement_exceeds_band_count"] for x in inner),
        "MAIN_inner_scope": main_scope,
        "interpretation": "fixed_budget_complete_not_convergence_or_sufficient_training_certificate",
        "extension_run": False,
    }
    decision = {
        "schema": "semantic_p2_final_research_decision_v1",
        "status": "COMPLETE_FORMAL_MATRIX_AND_ACTUAL_QUALIFICATIONS",
        "scientific_decision": "INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET",
        "specific_unclosed_gate": "Phase_A_training_sufficiency_under_the_prelocked400_budget",
        "known_software_or_leakage_failure_in_formal_results": False,
        "fixed_budget_observation": "NO_CONFIRMED_DSS_INCREMENT_IN_THIS_IMPLEMENTATION",
        "P1R_scope": "query_identity_pretext_signal_under_shared_S32_not_all_semantic_or_DSS_success",
        "master_lock_sha256": MASTER,
        "all_primary_arm_seed_fold_predictions_complete": True,
        "formal_source_neural_fits": 480,
        "formal_source_optimizer_updates": cost["actual_phase_A_optimizer_steps"],
        "formal_source_checkpoint_producer_replays": cost["actual_saved_checkpoint_replays"],
        "formal_Cox_fits": cost["actual_risk_Cox_fits"] + cost["actual_r0_Cox_fits"],
        "separate_software_neural_fits": f1["actual_audit_neural_fits"],
        "separate_software_Cox_fits": isolation["actual_additional_compute"]["all_Cox_fits"],
        "core_point_estimates": engineering["core_point_estimates"],
        "B5_B0_each_seed_negative_in_both_metrics": all(x["delta"] < 0 for key in ("Harrell_seed_deltas", "Uno60_seed_deltas") for x in lookup["B5-B0"][key]),
        "B5_B0_fold_seed_directions": {k: lookup["B5-B0"][k] for k in (
            "Harrell_positive_fold_seed_pairs", "Harrell_negative_fold_seed_pairs", "Harrell_zero_fold_seed_pairs",
            "Uno60_positive_fold_seed_pairs", "Uno60_negative_fold_seed_pairs", "Uno60_zero_fold_seed_pairs")},
        "training_budget_boundary": boundary,
        "B0_risks_bitwise_equal_to_historical_seed31": False,
        "B0_historical_unified_pair_components_exact_all_folds": True,
        "B0_historical_max_abs_risk_difference": max(x["max_abs_risk_difference"] for x in history["folds"]),
        "B0_historical_numerical_cause_fully_proven": False,
        "new_architecture_or_further_training_started": False,
        "formal_results_replaced_or_retuned_after_outer_evaluation": False,
        "confidence_intervals_computed": False, "bootstrap_replicates": 0, "p_values_computed": False,
        "confidence_ratings": {"recorded_fixed_budget_metrics_and_no_observed_B5_gain": "HIGH",
                               "attribution_to_insufficient_training_as_the_cause_of_DSS_failure": "LOW",
                               "universal_claim_that_the_research_idea_cannot_work": "NOT_SUPPORTED"},
        "source_anonymous_reports": [{"file": name, "bytes": (REPORT/name).stat().st_size,
                                       "sha256": sha(REPORT/name)} for name in inputs],
        "new_model_fits_by_this_reporter": 0, "new_model_forwards_by_this_reporter": 0,
    }
    write("11_RESEARCH_DECISION.json", decision)
    text = f"""# P2 最终研究裁决

**正式矩阵与实际软件资格已完成；主科学裁决为 `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET`，未闭合的是 Phase-A 训练充分性，不是已发现正式结果存在泄漏。**

固定合同下的观察结论明确：B5 Harrell={metrics['B5']['Harrell']:.9f}、Uno60={metrics['B5']['Uno60']:.9f}，低于B0的{metrics['B0']['Harrell']:.9f}/{metrics['B0']['Uno60']:.9f}；Δ分别为{lookup['B5-B0']['Harrell_delta']:+.9f}/{lookup['B5-B0']['Uno60_delta']:+.9f}。三个seed的两指标均为负。15个同fold×seed比较中，Harrell为2正/12负/1平，Uno为3正/12负；这些不是15个独立队列。

B5也未超过同输入B1、B2_MATCHED、B2_NATIVE、B4、B7和临床容量控制。B5−B6a只有Harrell极小正向，Uno为负；B4真Query未超过保守随机Query，真关系相对置乱关系两指标方向冲突。现有结果不支持“语义、临床残差、患者配对及跨模态乘积共同产生稳定DSS增量”。所有12个预锁比较完整保留在06表，不挑单一指标。

排序损伤大于修复：B5相对B0每seed五折累计的修复credit均值19、损伤34（含风险ties半分），与负向Harrell一致。最终B5总风险增量SD/anchor SD中位数在训练和外折分别约0.05854/0.05783，并未显示整体修正幅度明显坍塌；训练头ΔHarrell中位数约+0.00446，外折约−0.00383。这里的训练头是交叉拟合teacher offset上的样本内拟合；这些描述支持泛化排序损伤的诊断，不能独立证明过拟合、某个机制失效或“RNA信息不存在”。完整作用范围见08附录。

**预算限制有真实曲线依据，但不能被说成负结果的已证实原因。** 在实际评测的预锁检查点中，360个inner有{boundary['raw_Huber_unique_minimum_at400']}个原始验证Huber最低点在400，{boundary['validation_Huber_improved200to400']}个200→400仍改善，{boundary['validation_Huber_improvement_exceeds_original_band']}个改善超过原先训练侧定义的容忍带。MAIN自己45个inner中35个最低点在400，35个验证Huber改善、31个超原容忍带。因此固定预算执行完整不等于已充分优化；也没有数据证明延长能改善DSS。主锁禁止自动扩到800，未追加训练或事后重选。见09完整曲线和预算审计。

P1R确有共享S32条件下Query身份相关的pretext正向与槽位保留证据，但双通道修复并未全面胜过原结构，真实关系亦未稳定胜过置乱。它与当前DSS负增量是不同任务的事实，不互相覆盖。不能选`MECHANISM_SUPPORTED_WITHIN_DEVELOPMENT_COHORT`或`PREDICTION_GAIN_ONLY`；“仅有pretext信号”是当前观察范围，预算尚未闭合时不把它升级为该idea必然无预后作用。

对照范围必须保留：B3_TEXTLESS在表中标为`B3_TEXTLESS_INCREMENT`，仅增量编码路径无显式文本，整体r0仍保留TITAN S32；B1/B7共享MAIN无DSS跨模态预训练，只分别控制Phase-B的加性读出/跨模态乘积，不是全训练过程无交互。B2_MATCHED/NATIVE均为同token预算下的普通注意力适配，NATIVE不代表论文官方复现；matched只在已profile的参数与受支持矩阵FLOPs范围合格，不声称墙钟、显存或全部算子匹配。B0含原GRID-CONCH32和旧RNA512，B0比较不是同输入架构因果消融。

**验收的证据范围：** 480个正式源神经拟合、189800次更新、1909次生产者保存状态回放；1800个下游Cox及1040个共同教师Cox。独立核验重算185份已存风险、12项核心对比和370次官方API调用；它不是原始H5到神经输出的第二次完整复现。另2个50更新真实训练通过held-out协变量隔离；另106个Cox的标签隔离管线通过，均单列软件成本，没有替换主预测。

R4五折当前/历史seed31风险并非bitwise相同，最大差{decision['B0_historical_max_abs_risk_difference']:.3g}，但每折统一评分分子/分母完全一致。逐叶诊断发现保存PCA数值输出和RNA Cox参数的微小差异，未证明具体历史数值库原因；不把此报告伪写成逐字节复现，也不调参追旧分数。

优势：强r0、同输入与容量控制、整管线交叉拟合、冻结外折选择、独立评分和实际反例均保留。风险：65个事件、反复开发过的同一队列、PCA8/rank2容量限制、FM预训练重叠及缺乏组织学真值，均限制外推。r0已含RNA512风险，新增分支无增益不意味着RNA无独立信息。

置信度：已保存风险的固定预算评分与B5未见增益为高；“训练不足导致DSS失败”的原因归因为低；全idea无效不获支持。未计算CI、bootstrap、p值。停止本次主机制claim，不自动启动P3。
"""
    write("11_RESEARCH_DECISION.md", text)
    write("11_DECISION.md", "# 最终裁决入口\n\n正式工程状态与科学裁决分别记录。完整科学结论见 [11_RESEARCH_DECISION.md](11_RESEARCH_DECISION.md) 和 [对应JSON](11_RESEARCH_DECISION.json)。原始 `11_decision.json` 只表示矩阵/报告生成完成，其字节按原index保留。\n")
    summary = "# P1-R / P2 最终结果与验收\n\n" + text.split("\n", 1)[1]
    summary += "\n## 完整模型结果\n\n| 模型 | Harrell | Uno60 |\n|---|---:|---:|\n"
    for arm, row in metrics.items():
        label = "B3_TEXTLESS_INCREMENT" if arm == "B3_TEXTLESS" else arm
        summary += f"| {label} | {row['Harrell']:.9f} | {row['Uno60']:.9f} |\n"
    summary += "\n原生成器的 [RESULTS.md](RESULTS.md) 与 `00_report_index.json` 原字节保留；当前文件是完成预算和软件资格后的最终研究解释。精确文件映射见 `DELIVERY_FILE_MAP.json`，实际终检与文件哈希见PUBLICATION_MANIFEST。P1R完整结果位于相邻 `semantic_b0b7_p1r` 目录和已发布P1R分支。\n"
    write("FINAL_RESULTS.md", summary)
    replay = {"schema": "semantic_p2_final_independent_evidence_index_v1",
        "status": "COMPLETE_WITH_EXPLICIT_SCOPE_AND_TRAINING_BUDGET_LIMITATION",
        "master_lock_sha256": MASTER, "actual_official_scorer": independent["official_package_API"],
        "actual_report_math": independent["report_comparison"],
        "actual_saved_risks_recomputed": independent["risk_vectors_independently_recomputed"],
        "actual_source_receipts_hash_verified": independent["phase_a_receipts_verified"],
        "actual_saved_checkpoint_byte_hash_verified": independent["actual_checkpoint_files_byte_hash_verified"],
        "new_independent_neural_full_chain_replays": 0,
        "raw_H5_to_prediction_second_replication": False,
        "inner_risk_vectors_second_rescoring": False,
        "Phase_A_heldout_mutation_actual_neural_fits": f1["actual_audit_neural_fits"],
        "Phase_A_raw_checkpoint_bytes_equal": f1["checkpoint_comparison"]["checkpoint_archives_byte_identical"],
        "Phase_A_decoded_model_optimizer_RNG_exact": True,
        "full_DSS_pipeline_heldout_label_isolation": isolation["status"],
        "full_DSS_pipeline_components_exact": all(x["exact"] for x in isolation["exact_components"].values()),
        "additional_software_cost": isolation["actual_additional_compute"],
        "history_replay_scope": "five_saved_old_seed31_versus_new_B0_no_new_fit_or_forward",
        "history_risk_byte_exact": False, "history_fold_pair_components_exact": True,
        "evidence_reports": decision["source_anonymous_reports"],
        "CI_computed": False, "new_training_by_index": 0,
    }
    write("10_INDEPENDENT_REPLAY_AUDIT.json", replay)
    print(json.dumps({"status": "COMPLETE_APPEND_ONLY_FINAL_RESEARCH_REPORTS", "files": [
        "11_RESEARCH_DECISION.json", "11_RESEARCH_DECISION.md", "11_DECISION.md", "FINAL_RESULTS.md", "10_INDEPENDENT_REPLAY_AUDIT.json"]}))


if __name__ == "__main__":
    main()

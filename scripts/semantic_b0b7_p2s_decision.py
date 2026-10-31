"""Derive the bounded P2S scientific report from anonymized aggregate tables only.

No fitting, scoring of patient risks, endpoint access, or model selection occurs here.
The narrative is an explicitly post-results interpretation of the prelocked grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "02_A_C_paired_summaries.json": "1e00d81bf97a02eb89a4d415fc8cd850b81692d4476958e7d09acb4d0159d4ea",
    "05_seed_micro_aggregates.json": "515bafac34607584f79445357dbca1456195a286f40c19feb8c18e121ecc5f6e",
    "PRIMARY_R0_ANCHOR_MICRO.json": "165b79b8cf6121de49ced1715c9a3f7eb10235050a1021fb453e6c01e1717549",
    "06_fixed_comparison_summaries_01.json": "10258456faf7dfe0e50545a151ba04fff05388bd40870d15ba6aa16bf1fde417",
}
KEYS = ("family", "epoch", "lambda_pair", "main_pca", "base_arm", "factor_rank", "mode", "ridge_lambda")


def load_checked(root, name):
    raw = (root / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED[name], name
    return json.loads(raw)


def delta(a, b):
    out = {}
    for metric in ("Harrell", "Uno60"):
        ds = [{"seed": x["seed"], "delta": x[metric]["value"] - y[metric]["value"]}
              for x, y in zip(a["seed_results"], b["seed_results"])]
        assert all(x["seed"] == y["seed"] for x, y in zip(a["seed_results"], b["seed_results"]))
        out[metric] = {"delta": a[metric] - b[metric], "seed_deltas": ds,
                       "positive_seeds": sum(x["delta"] > 0 for x in ds),
                       "negative_seeds": sum(x["delta"] < 0 for x in ds),
                       "zero_seeds": sum(x["delta"] == 0 for x in ds)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output-newdir", type=Path)
    a = p.parse_args()
    root = a.reports
    aux = load_checked(root, "02_A_C_paired_summaries.json")
    rows = load_checked(root, "05_seed_micro_aggregates.json")
    anchor = load_checked(root, "PRIMARY_R0_ANCHOR_MICRO.json")
    paired = load_checked(root, "06_fixed_comparison_summaries_01.json")
    paired_ix = {x["comparison"]: x for x in paired}
    assert len(rows) == 264 and all(x["status"] == "COMPLETE" for x in rows)
    ix = {tuple(x[k] for k in KEYS): x for x in rows}
    assert len(ix) == len(rows)

    def get(f="MAIN", e=400, pair=.1, d=8, arm="B5", rank=2, mode="INTER", ridge=1.):
        return ix[f, e, pair, d, arm, rank, mode, ridge]

    primary = []
    for x in rows:
        if x["ridge_lambda"] != 1. or x["mode"] != "INTER":
            continue
        common = dict(f=x["family"], e=x["epoch"], pair=x["lambda_pair"], d=x["main_pca"])
        no = get(**common, arm="SHARED", rank=0, mode="NO_INTER")
        entry = {k: x[k] for k in KEYS}
        entry.update(Harrell=x["Harrell"], Uno60=x["Uno60"], versus_actual_NO_INTER_refit=delta(x, no))
        if x["family"] == "MAIN":
            b7 = get(**common, arm=x["base_arm"], rank=x["factor_rank"], mode="B7")
            entry["versus_capacity_matched_B7"] = delta(x, b7)
        primary.append(entry)
    assert len(primary) == 40

    fixed_sensitivity = []
    for ridge in (.1, 1., 10.):
        for contrast in ("PCA16_minus8", "factor4_minus2", "epoch800_minus400", "pair003_minus01"):
            ds = []
            for x in rows:
                if x["ridge_lambda"] != ridge:
                    continue
                key = [x[k] for k in KEYS]
                if contrast == "PCA16_minus8" and x["main_pca"] == 16:
                    key[3] = 8
                elif contrast == "factor4_minus2" and x["factor_rank"] == 4:
                    key[5] = 2
                elif contrast == "epoch800_minus400" and x["epoch"] == 800:
                    key[1] = 400
                elif contrast == "pair003_minus01" and x["lambda_pair"] == .03:
                    key[2] = .1
                else:
                    continue
                y = ix[tuple(key)]
                ds.append((x["Harrell"] - y["Harrell"], x["Uno60"] - y["Uno60"]))
            fixed_sensitivity.append({"ridge_lambda": ridge, "contrast": contrast,
                "correlated_configuration_pairs_not_independent_samples": len(ds),
                "positive_Harrell_configurations": sum(h > 0 for h, u in ds),
                "positive_Uno60_configurations": sum(u > 0 for h, u in ds),
                "both_positive_configurations": sum(h > 0 and u > 0 for h, u in ds),
                "mean_Harrell_delta_descriptive_only": sum(h for h, u in ds) / len(ds),
                "mean_Uno60_delta_descriptive_only": sum(u for h, u in ds) / len(ds),
                "configuration_or_lambda_selection": False})

    positive_probe = get(e=800, d=16)
    no_probe = get(e=800, d=16, arm="SHARED", rank=0, mode="NO_INTER")
    b7_probe = get(e=800, d=16, mode="B7")
    budget_probe = get(e=400, d=16)
    direction = paired_ix["p0.1_e800_ridge1.0_MAIN_D16_B5_R2_INTER_minus_NOINTER"]
    assert direction["Harrell_cell_positive"] == 6 and direction["Uno60_cell_positive"] == 7
    probe = {"role": "post_results_description_of_one_prelocked_cell_not_a_selected_winner",
             "family": "MAIN", "base_arm": "B5", "epoch": 800, "lambda_pair": .1,
             "main_pca": 16, "factor_rank": 2, "ridge_lambda": 1.,
             "Harrell": positive_probe["Harrell"], "Uno60": positive_probe["Uno60"],
             "versus_actual_NO_INTER_refit": delta(positive_probe, no_probe),
             "versus_capacity_matched_B7": delta(positive_probe, b7_probe),
             "versus_same_configuration400": delta(positive_probe, budget_probe),
             "positive_inner_fold_cells_versus_NO_INTER": {
                 "Harrell": direction["Harrell_cell_positive"], "Uno60": direction["Uno60_cell_positive"],
                 "required_paired_cells": direction["required_paired_cells"]},
             "new_independent_confirmation": False,
             "new_wrong_patient_retraining_control_for_this_cell": False}
    regularization = []
    for ridge in (.1, 1., 10.):
        rr = [x for x in rows if x["ridge_lambda"] == ridge]
        regularization.append({"ridge_lambda": ridge, "all_prelocked_configuration_count": len(rr),
            "Harrell_min": min(x["Harrell"] for x in rr), "Harrell_max": max(x["Harrell"] for x in rr),
            "Uno60_min": min(x["Uno60"] for x in rr), "Uno60_max": max(x["Uno60"] for x in rr),
            "Harrell_above_exact_r0": sum(x["Harrell"] > anchor["Harrell"] for x in rr),
            "Uno60_above_exact_r0": sum(x["Uno60"] > anchor["Uno60"] for x in rr),
            "both_above_exact_r0": sum(x["Harrell"] > anchor["Harrell"] and x["Uno60"] > anchor["Uno60"] for x in rr),
            "grid_extrema_are_descriptive_not_selected_performance": True})
    assert all(x["Harrell"] < anchor["Harrell"] and x["Uno60"] < anchor["Uno60"]
               for x in rows if x["ridge_lambda"] == 1.)
    assert any(x["mode"] == "NO_INTER" and x["Harrell"] > anchor["Harrell"]
               for x in rows if x["ridge_lambda"] == 10.)
    decision = {
        "schema": "semantic_p2s_bounded_scientific_decision_v1",
        "status": "A_B_C_D_COMPLETE_DIAGNOSTIC_ONLY",
        "derivation": "post_results_interpretation_of_prelocked_aggregate_grid_no_new_fits_or_patient_scoring",
        "source_reports": [{"file": n, "sha256": h} for n, h in EXPECTED.items()],
        "scope": {"outer_training_patients": 409, "existing_inner_partitions": 3, "seeds": [31,47,83],
            "outer_evaluation": False, "independent_confirmation": False,
            "aggregation": "within_inner_fold_pair_component_micro_per_seed_then_equal_three_seed_mean",
            "primary_ridge_lambda": 1., "ridge_lambdas_all_reported": [.1,1.,10.],
            "confidence_intervals": False, "bootstrap": False, "p_values": False},
        "actual_execution": {"neural_trajectories": 36, "new_optimizer_updates": 21600,
            "checkpoint_replays": 72, "frozen_feature_states": 72, "Cox_fits": 2376,
            "explicit_NA_cells": 0, "independent_official_API_calls": 4752,
            "new_teacher_or_scalar_fits": 0},
        "A_auxiliary_budget": {"summary": "0.1下两种架构均仍有辅助泛化指标改善；800轮不等于证明收敛，也未产生一致DSS增长。",
            "all_auxiliary_comparisons": aux},
        "B_readout": {"summary": "PCA8存在部分可解码容量限制的线索，收益依赖权重、预算和正则；无一致四因子增益。",
            "factor4_is_pretrained_rank4": False,
            "PCA16_also_changes_main_effect_information": True,
            "fixed_grid_sensitivity_descriptive_not_inference": fixed_sensitivity},
        "C_pair_weight": {"summary": "0.03并未一致改善辅助质量与条件DSS增量；不据此替换0.1。",
            "weighted_total_loss_not_valid_cross_weight_comparison": True},
        "D_conditional_increment": {"summary": "d8/r2无一致双指标收益；d16/r2在0.1/800有三seed均正线索，未形成跨设置稳定机制证据。",
            "positive_but_unconfirmed_prelocked_cell": probe,
            "shared_strong_r0": {"Harrell": anchor["Harrell"], "Uno60": anchor["Uno60"],
                                 "unique_fold_seed_rows": anchor["unique_fold_seed_metric_rows"]},
            "regularization_grid_against_exact_r0": regularization,
            "all_primary_lambda_configurations_below_r0_both_metrics": True,
            "lambda10_Harrell_positive_cells_include_true_NO_INTER": True,
            "joint_minus_refitted_main_includes_main_coefficient_drift": True,
            "frozen_mechanism_off_is_not_retraining": True},
        "all40_primary_interaction_configurations": primary,
        "next_action": "STOP_AT_PRELOCKED800_NO_NEW_SEARCH_NO_AUTOMATIC_PHASE_ADVANCE",
        "idea_globally_rejected": False,
        "epoch_pair_weight_or_ridge_winner_selected": False,
        "mechanism_established": False,
        "confidence_rating": {"execution_and_saved_design_metric_arithmetic": "高",
                              "some_auxiliary_information_continues_improving": "中",
                              "independent_generalizable_interaction_benefit": "低"},
        "advantages": ["同内层分区、固定训练预算与强r0；实际无交互重拟合和等容量B7；全部正负结果保留。",
                       "源锁、保存重放、独立线性预测和官方评分API核验闭合。"],
        "methodological_risks": ["重复使用已查看的开发数据，多个相关配置不能充当独立重复。",
                  "PCA扩维同时改善主效应；四因子仅为冻结投影扩展而非预训练rank4。",
                  "有限正向配置尚无新的错患者重训/独立外部验证；不能证明医学语义正确或因果交互。"],
    }

    lines = ["# P2S：最后一次范围锁定的训练充分性诊断", "",
        "**A–D 已全部完成。辅助表征继续改善，并不等于 DSS 增量继续增长；本轮保留一项有限正向条件增量线索，不将整个 idea 判为无效，也不据此自动推进模型。**", "",
        "只使用原 outer1 的409名训练患者、三个既有内层划分及 seed 31/47/83。当前 inner 结果已属于重复开发诊断，不是新的外折或独立确认。主 ridge λ=1 事前固定，0.1/1/10 全部报告，不选最优轮数、权重、λ或配置；按要求未计算置信区间、bootstrap 或 p 值。", "",
        "## A：400→800轮是否仍有辅助信息增长", "",
        "下表均为同划分、同seed的9个配对单元均值；Huber下降为改善。它们不是9个独立队列。", "",
        "|模型|配对权重|ΔHuber|Δ程序Pearson|Δhard gap|", "|---|---:|---:|---:|---:|"]
    for x in aux:
        if "800_minus400" in x["comparison"]:
            model, rest = x["comparison"].split("_pair")
            pair = rest.split("_")[0]
            lines.append(f"|{model}|{pair}|{x['program_Huber_mean_paired_delta']:+.6f}|{x['macro_program_Pearson_mean_paired_delta']:+.6f}|{x['hard_gap_mean_paired_delta']:+.6f}|")
    lines += ["", "λpair=0.1 时，MAIN 与 Attention 的程序Pearson均为9/9单元改善，Huber均为8/9改善。这支持400轮以后仍有辅助信息增长，也说明增长不是 MAIN 独占。800轮仅是本次预算终点，不能声称训练已经达到数学收敛。", "",
        "## B：PCA8与rank2是否构成读出瓶颈", "",
        "P/G从8维扩到16维，Clinical保持8维。四因子只增加两个固定、无结局的正交补投影，不是重训的rank4网络。以下固定展示λpair=0.1的全部 B5 维度/预算组合；完整B4、其他权重与所有正则见机器可读表。", "",
        "|轮数|P/G PCA|因子|Harrell|Uno60|ΔH vs无交互重拟合|ΔU vs无交互重拟合|", "|---|---:|---:|---:|---:|---:|---:|"]
    for e in (400,800):
        for d in (8,16):
            for r in (2,4):
                x = get(e=e,d=d,rank=r)
                no = get(e=e,d=d,arm="SHARED",rank=0,mode="NO_INTER")
                lines.append(f"|{e}|{d}|{r}|{x['Harrell']:.6f}|{x['Uno60']:.6f}|{x['Harrell']-no['Harrell']:+.6f}|{x['Uno60']-no['Uno60']:+.6f}|")
    lines += ["", "扩维并非一致增益。在0.1/400/rank2，B5的Uno60由0.770925升到0.776653；同表示无交互头也由0.771849升到0.778184，因此不能把该提升全归给交互。0.03/400时扩维的方向反转。四因子也没有一致改善，负结果仅适用于本次冻结探针。", "",
        "正则敏感性同样存在：在预锁λ=10的44组PCA配对配置中，40组H/U都上升；主λ=1仅20/44组双指标上升。它们是高度相关的配置计数，不能作为样本量或显著性检验，也没有据此改选λ。", "",
        "## C：0.03与0.1", "",
        "降低配对权重没有一致地改善未加权辅助指标及DSS。主λ=1下44组既定配对配置中，Harrell为23组上升、Uno60为19组上升，仅14组双指标均升；这不是替换权重的充分依据。MAIN 0.03在800相对400的Huber反而增加0.002032。两个权重的加权总损失不可直接比较。", "",
        "## D：共享强锚点后的条件DSS增量", "",
        "下列B5均保持原rank2，所有表示和头共享对应训练分区的精确r0；NO_INTER是真实重拟合，B7控制额外单模态读出容量。", "",
        "|配对权重|轮数|P/G PCA|ΔH vs NO_INTER|ΔU vs NO_INTER|ΔH vs B7|ΔU vs B7|", "|---|---:|---:|---:|---:|---:|---:|"]
    for pair in (.1,.03):
        for e in (400,800):
            for d in (8,16):
                x=get(e=e,pair=pair,d=d); no=get(e=e,pair=pair,d=d,arm="SHARED",rank=0,mode="NO_INTER"); b7=get(e=e,pair=pair,d=d,mode="B7")
                lines.append(f"|{pair}|{e}|{d}|{x['Harrell']-no['Harrell']:+.6f}|{x['Uno60']-no['Uno60']:+.6f}|{x['Harrell']-b7['Harrell']:+.6f}|{x['Uno60']-b7['Uno60']:+.6f}|")
    lines += ["", "**有限正向线索：0.1/800/PCA16/rank2。** B5相对无交互重拟合为ΔH=+0.004096、ΔU=+0.005911，相对B7为+0.003490/+0.004327；两种比较的三个seed均为H/U正方向。相对NO_INTER，九个内层单元为Harrell 6/9、Uno60 7/9正方向。它属于已锁定网格中观察到的正向单元，不能事后改称预定赢家或独立确认。", "",
        "这一配置的绝对成绩是0.789258/0.775851，仍低于同配置400轮的0.793279/0.776653。其无交互对照也随预算变化，必须同时区分‘联合模型变好’和‘联合模型相对变化中的主效应对照更好’。同一扩维在0.03下，相对NO_INTER未重复双指标正增量。", "",
        "原d8/rank2中，B5在0.1/400相对NO_INTER是+0.003869/−0.000924，在800是+0.002048/−0.004585；普通Attention在权重0.1、同维度的两个预算下，两指标均低于真实NO_INTER。因此，增加辅助预算没有普遍补上条件风险收益。", "",
        "**必须再看共享r0本身，而不只看新增表示后的NO_INTER。** 同一内层协议下，r0为Harrell 0.825899、Uno60 0.787859。主λ=1的全部88个配置两项均低于r0，包含上述有限正向条件增量单元。新增头的正则强度因此是实质性敏感因素，不能把当前低成绩仅解释成预训练不足。", "",
        "预锁λ=10下，88个配置的Harrell范围为0.817478–0.826809，Uno60为0.787347–0.799462；4个配置的Harrell超过r0，其中一个是真实NO_INTER，另外三个为400轮/PCA16的B5交互配置。Uno60有87/88个超过r0。这里完整描述固定正则网格的范围和计数，不把最大值作为选优后的新成绩，也不把收益全部归因于交互。没有据此将主λ改为10。", "",
        "## 实际执行和复核", "",
        "36条神经轨迹：18条从零到800、18条从原400继续到800；新增21,600次优化更新。54个新快照加18个旧400引用，共72个实际重载回放。72组冻结特征完成2,376个Cox头拟合和保存后重放，NA为0；另外拟合144个临床背景Ridge和432个PCA。精确复用三个原r0锚点，新增教师/标量拟合为0。", "",
        "独立复核重算已保存Z、系数与offset的线性预测、Cox目标和评分；2,376个头及4,752次官方Harrell/Uno API调用均通过，另核对264个配置聚合。它不是从原H5重新抽特征的第二套独立复现，也没有独立重推全部1,104组差分。", "",
        "优势是固定分区、强共同锚点、真实无交互重拟合、等容量控制和完整负结果；风险是重复开发数据、相关配置多、PCA扩维同时改变主效应、四因子不是预训练rank4，以及缺少此正向单元的新错配重训与外部确认。数值执行正确性的置信度高；可推广交互机制的置信度低。", "",
        "## 范围终点", "",
        "本次诊断到此结束，不再增加轮数、通路、模型或搜索。结论不是idea无效，而是：本合同下已观察到辅助信息增长和一项表示容量相关的条件增量线索，但没有得到跨预算/权重一致的DSS改善，尚不能确立可推广的三模态机制。更完整的边界见[限制说明](12_LIMITATIONS.md)、[复现说明](13_REPRODUCE.md)与[全部固定配置](05_seed_micro_aggregates.json)。", ""]

    outputs = {"decision.json": json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
               "RESULTS_FINAL.md": "\n".join(lines)}
    output = a.output_newdir or root
    if a.output_newdir is not None:
        output.mkdir(parents=True, exist_ok=False)
    for name, text in outputs.items():
        with (output/name).open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(json.dumps({"status":"COMPLETE_AGGREGATE_ONLY", "new_fits":0, "patient_scoring":0,
                      "files":[{"file":name,"bytes":(output/name).stat().st_size,"sha256":hashlib.sha256((output/name).read_bytes()).hexdigest()} for name in outputs]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create auditable aggregate outputs for the Phase6B token-prompt audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
    "S0_static_concepts",
    "S1_static_token_prompt",
    "S2_rna_token_prompt",
    "S3_static_concepts_plus_rna",
    "S4_wrong_patient_rna_token_prompt",
]
COMPARISON_ORDER = ["S2-S1", "S2-S3", "S2-S4"]


def fmt(x: object, digits: int = 4) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    metrics = pd.read_csv(out / "metrics.csv")
    paired = pd.read_csv(out / "paired_comparisons_by_seed.csv")
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    done = json.loads((out / "DONE.json").read_text(encoding="utf-8"))
    if done["status"] != "PASS" or validation["status"] != "PASS" or len(metrics) != 60 or len(paired) != 36:
        raise RuntimeError("refusing to summarize an incomplete experiment")
    if metrics.duplicated(["endpoint", "method", "seed"]).any() or not (metrics.groupby(["endpoint", "method"]).seed.nunique() == 3).all():
        raise RuntimeError("invalid seed-level OOF metric rows")
    metric_index = metrics.set_index(["endpoint", "method", "seed"]).balanced_accuracy
    pairs = {"S2-S1": (METHOD_ORDER[2], METHOD_ORDER[1]),
             "S2-S3": (METHOD_ORDER[2], METHOD_ORDER[3]),
             "S2-S4": (METHOD_ORDER[2], METHOD_ORDER[4])}
    for row in paired.itertuples(index=False):
        a, b = pairs[row.comparison]
        expected = metric_index.loc[row.endpoint, a, row.seed] - metric_index.loc[row.endpoint, b, row.seed]
        if abs(expected - row.delta_balanced_accuracy) > 1e-10:
            raise RuntimeError("paired point delta is not the exact OOF metric difference")

    # The primary seed-level unit is the 5-fold OOF aggregate already emitted
    # by the runner.  Do not call the mean of 15 cells a seed mean.
    method_summary = (
        metrics.groupby(["endpoint", "method"], as_index=False)
        .agg(
            seed_mean_balanced_accuracy=("balanced_accuracy", "mean"),
            seed_sd_balanced_accuracy=("balanced_accuracy", "std"),
            seed_mean_macro_f1=("macro_f1", "mean"),
            seed_mean_accuracy=("accuracy", "mean"),
            seed_mean_macro_ovr_auroc=("macro_ovr_auroc", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    method_summary["method"] = pd.Categorical(method_summary["method"], METHOD_ORDER, ordered=True)
    method_summary = method_summary.sort_values(["endpoint", "method"])
    method_summary.to_csv(out / "aggregate_by_endpoint_method.csv", index=False)

    paired_summary = (
        paired.groupby(["endpoint", "comparison"], as_index=False)
        .agg(
            mean_delta_balanced_accuracy=("delta_balanced_accuracy", "mean"),
            seed_sd_delta_balanced_accuracy=("delta_balanced_accuracy", "std"),
            min_delta_balanced_accuracy=("delta_balanced_accuracy", "min"),
            max_delta_balanced_accuracy=("delta_balanced_accuracy", "max"),
            all_seed_ci_above_zero=("ci98_75_low", lambda s: bool((s > 0).all())),
            all_seed_ci_below_zero=("ci98_75_high", lambda s: bool((s < 0).all())),
            seeds=("seed", "nunique"),
        )
    )
    paired_summary["comparison"] = pd.Categorical(paired_summary["comparison"], COMPARISON_ORDER, ordered=True)
    paired_summary = paired_summary.sort_values(["endpoint", "comparison"])
    paired_summary.to_csv(out / "aggregate_paired_comparisons.csv", index=False)

    lines = [
        "# Phase6B token-level TITAN prompt audit — v2 mask repair",
        "",
        "本报告对应真正的 token-level prompt：连续 prompt 插入 TITAN 文本序列的 BOS 之后，随后经过冻结的 TITAN text transformer。TITAN 图像编码器、文本编码器和原始静态概念原型均冻结。",
        "",
        "v1 在计算层面完成，但其 CLS padding mask 未随插入 token 移动，且模板融合顺序与官方不同，机制比较已作废。v2 只重跑 S1/S2，复用经输入与逐 cell 校验的 S0/S3；完整 S4 OOF 也已在服务器私有保存。",
        "",
        "## Protocol",
        "",
        "- 3 seeds × 5 outer folds；每个 outer-train 使用固定 3 epoch schedule。没有使用 inner-val 标签选择 epoch 或 upstream main effect，避免 inner-val 标签进入配置训练。",
        "- S0 static TITAN concepts；S1 static learnable token prompt；S2 RNA-conditioned token prompt；S3 static concepts + RNA additive；S4 为 S2 的 wrong-patient RNA permutation。",
        "- RNA 只经过 fold-local Top512/512→16 adapter；S2 的 prompt pool 为 4×4×768；T/N/M/Stage 是唯一读取的临床目标。DSS/OS 未读取。",
        "- 每个 seed 先拼接 5-fold OOF，再计算该 seed 的指标；本报告的 seed mean 是 3 个 seed-level OOF 结果的均值，不是 15 个 cells 的直接均值。",
        "- 配对 point delta 为精确 OOF BA 差值；2000 次 class-stratified paired patient bootstrap 保持稀少类别计数。98.75% 区间是逐比较的描述性区间，不代表全部 12 个 endpoint/comparison 的联合错误率已控制。",
        "- 计算优化只跳过被 mask 的尾部 padding，并保留原始 CLS 位置；S1 batch 共享相同静态 text prototype。已通过 full/compact 前向和梯度一致性门禁。修复前段已完成的正确 full-padding cells 保留，误差见 encoder audit。",
        "",
        "## Validation",
        "",
        "```json",
        json.dumps(validation, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Seed-level OOF metrics",
        "",
        "| Endpoint | Method | Seed mean BA | Seed SD BA | Seed mean macro-F1 | Seed mean AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in method_summary.itertuples(index=False):
        lines.append(
            f"| {row.endpoint} | {row.method} | {fmt(row.seed_mean_balanced_accuracy)} | {fmt(row.seed_sd_balanced_accuracy)} | {fmt(row.seed_mean_macro_f1)} | {fmt(row.seed_mean_macro_ovr_auroc)} |"
        )
    lines += [
        "",
        "## Mechanism paired comparisons",
        "",
        "正向 delta 定义为左侧方法 − 右侧方法。支持 token-level patient-specific interpretation 需要同时观察 S2>S1、S2>S3、S2>S4；单独优于 S0 不足以证明该机制。",
        "",
        "| Endpoint | Comparison | Mean delta BA | Seed SD | Seed delta range | Every 98.75% CI > 0 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_summary.itertuples(index=False):
        lines.append(
            f"| {row.endpoint} | {row.comparison} | {fmt(row.mean_delta_balanced_accuracy)} | {fmt(row.seed_sd_delta_balanced_accuracy)} | [{fmt(row.min_delta_balanced_accuracy)}, {fmt(row.max_delta_balanced_accuracy)}] | {bool(row.all_seed_ci_above_zero)} |"
        )

    s2s1 = paired_summary[paired_summary["comparison"] == "S2-S1"]
    s2s3 = paired_summary[paired_summary["comparison"] == "S2-S3"]
    s2s4 = paired_summary[paired_summary["comparison"] == "S2-S4"]
    positive_endpoints = []
    robust_endpoints = []
    for endpoint, group in paired_summary.groupby("endpoint"):
        if len(group) == 3 and (group.mean_delta_balanced_accuracy > 0).all():
            positive_endpoints.append(endpoint)
        if len(group) == 3 and group.all_seed_ci_above_zero.all():
            robust_endpoints.append(endpoint)
    verdict = ("三项平均配对 delta 均为正的 endpoint：" + (", ".join(positive_endpoints) or "无") + "。 "
               "三项差值在每个 seed 的 98.75% CI 均高于零的 endpoint：" + (", ".join(robust_endpoints) or "无") + "。")
    if not robust_endpoints:
        verdict += " 当前协议未形成稳健的 patient-specific token-conditioning 机制证据；这不等于该 idea 已被证明无效。"
    else:
        verdict += " 证据限于上述 endpoint；三项控制是必要的经验门槛，不是医学语义正确性的直接证明。"
    lines += ["", "## Mechanism verdict", "", verdict, ""]
    lines += [
        "## Limitations",
        "",
        "- 固定 3 epoch schedule 无 inner-val 标签驱动的选择，但不保证收敛；相同 epoch 数不保证不同容量模型同等充分训练，不能把负结果直接解释为 idea 不成立。",
        "- 目标是 T/N/M/Stage proxy，而非 DSS survival endpoint；结果不能直接替代生存分析。",
        "- S4 使用 outer-test 上的循环错配 RNA 作为机制对照；它不是独立训练的第五模型。",
        "- S1/S2 与 S3 的函数形式和参数量不同；即使控制通过，也尚缺独立形态概念标注来验证语义解释。",
        "- 没有因看到 v1/v2 outer-test 结果而调整学习率、epoch、词表或 prompt 容量。后续协议变更需按探索性或新确认实验解释。",
        "",
    ]
    counts = json.loads((out / "parameter_counts.json").read_text(encoding="utf-8"))
    lines += ["## Trainable parameters", "", "| Endpoint / method | Parameters |", "|---|---:|"]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    history = pd.read_csv(out / "training_history.csv")
    loss_table = history.groupby(["endpoint", "method", "epoch"], as_index=False).mean_batch_train_ce.mean()
    loss_table.to_csv(out / "training_loss_summary.csv", index=False)
    lines += ["", "## Training diagnostics", "",
              "仅保存修复后 S1/S2 的 training CE；S0/S3 为原始有效 OOF 复用，无原轮次逐 epoch loss。",
              "", "| Endpoint | Method | Epoch 1 mean CE | Epoch 3 mean CE |", "|---|---|---:|---:|"]
    for (endpoint, method), group in loss_table.groupby(["endpoint", "method"]):
        by_epoch = group.set_index("epoch").mean_batch_train_ce
        lines.append(f"| {endpoint} | {method} | {fmt(by_epoch.loc[1])} | {fmt(by_epoch.loc[3])} |")
    lines += ["", "**Storage:** TITAN 权重保留在服务器；模型权重、RNA、患者 embeddings、OOF 和逐患者标签均未推送。", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

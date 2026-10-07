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
        "# Phase6B token-level TITAN prompt audit",
        "",
        "本报告对应真正的 token-level prompt：连续 prompt 插入 TITAN 文本序列的 BOS 之后，随后经过冻结的 TITAN text transformer。TITAN 图像编码器、文本编码器和原始静态概念原型均冻结。",
        "",
        "## Protocol",
        "",
        "- 3 seeds × 5 outer folds；每个 outer-train 使用固定 3 epoch schedule。没有使用 inner-val 标签选择 epoch 或 upstream main effect，避免 inner-val 标签进入配置训练。",
        "- S0 static TITAN concepts；S1 static learnable token prompt；S2 RNA-conditioned token prompt；S3 static concepts + RNA additive；S4 为 S2 的 wrong-patient RNA permutation。",
        "- RNA 只经过 fold-local Top512/512→16 adapter；S2 的 prompt pool 为 4×4×768；T/N/M/Stage 是唯一读取的临床目标。DSS/OS 未读取。",
        "- 每个 seed 先拼接 5-fold OOF，再计算该 seed 的指标；本报告的 seed mean 是 3 个 seed-level OOF 结果的均值，不是 15 个 cells 的直接均值。",
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
    strict = bool(
        len(s2s1) == len(s2s3) == len(s2s4) == 4
        and s2s1["all_seed_ci_above_zero"].all()
        and s2s3["all_seed_ci_above_zero"].all()
        and s2s4["all_seed_ci_above_zero"].all()
    )
    if strict:
        verdict = "三项 paired comparison 均满足跨 endpoint 的 98.75% seed-level bootstrap CI > 0，支持该 token-level patient-specific interpretation 机制。"
    else:
        verdict = "未同时满足 S2>S1、S2>S3、S2>S4 的严格判据；若只有部分 endpoint 为正，应将结论限定为 endpoint-specific evidence，不能宣称机制被整体证实。"
    lines += ["", "## Mechanism verdict", "", verdict, ""]
    lines += [
        "## Limitations",
        "",
        "- 固定 3 epoch schedule 解决了 inner-val 标签泄漏，但不等同于对所有 endpoint 的最优训练轮数；这是保守的预注册式公平比较。",
        "- 目标是 T/N/M/Stage proxy，而非 DSS survival endpoint；结果不能直接替代生存分析。",
        "- S4 使用 outer-test 上的循环错配 RNA 作为机制对照；它不是独立训练的第五模型。",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

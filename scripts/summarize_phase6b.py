"""Create the publishable Phase6B aggregate report from server summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    report = args.report
    metrics = pd.read_csv(report / "metrics.csv")
    pairs = pd.read_csv(report / "paired_comparisons_by_seed.csv")
    params = json.loads((report / "parameter_counts.json").read_text(encoding="utf-8"))
    aggregate = (metrics.groupby(["endpoint", "method"], as_index=False)
                 .agg(seed_mean_balanced_accuracy=("balanced_accuracy", "mean"),
                      seed_sd_balanced_accuracy=("balanced_accuracy", "std"),
                      seed_min_balanced_accuracy=("balanced_accuracy", "min"),
                      seed_max_balanced_accuracy=("balanced_accuracy", "max")))
    aggregate.to_csv(report / "aggregate_metrics.csv", index=False)
    pair_summary = (pairs.groupby(["endpoint", "comparison"], as_index=False)
                    .agg(seed_mean_delta=("delta_balanced_accuracy", "mean"),
                         seed_min_delta=("delta_balanced_accuracy", "min"),
                         seed_max_delta=("delta_balanced_accuracy", "max"),
                         min_ci_low=("ci98_75_low", "min"),
                         max_ci_high=("ci98_75_high", "max")))
    pair_summary.to_csv(report / "aggregate_paired_comparisons.csv", index=False)
    lines = [
        "# Phase6B RNA-conditioned TITAN concept audit",
        "",
        "## 结论",
        "",
        "**本轮没有证明 RNA-conditioned prompt 改善了 pathology concept interpretation。** RNA-conditioned concepts 在 T、N、M、Stage 的 seed-level OOF balanced accuracy 均低于 static concepts；与 `static concepts + RNA additive` 的 paired 差异接近 0，98.75% patient bootstrap 区间均跨过 0。因此不能把增量解释为 RNA 改变了 concept 含义。",
        "",
        "matched 与 wrong-patient RNA 的差异在四个 endpoint 都是正点估计；N endpoint 的 3 个 seed 区间均为正，Stage 只有一个 seed 区间为正，T/M 区间主要跨 0。这说明 RNA 的患者匹配关系可能携带疾病状态信息，但尚不足以支持完整的 patient-specific concept co-adaptation 机制。",
        "",
        "## Protocol audit",
        "",
        "- 3 seeds × 5 outer folds；每个 outer-train 内 3-fold inner selection。",
        "- RNA 的缺失填补、方差筛选和标准化只在 inner-train 或最终 outer-train 拟合。",
        "- TITAN 图像/文本编码器、Phase6A 图像表示和 32 个形态概念原型全部冻结。",
        "- 只使用 T/N/M/Stage；未读取 DSS、OS 或任何生存列。",
        "- wrong-patient control 在 outer-test 内循环置换 RNA，匹配模型、图像和分类头保持不变。",
        "",
        "## Seed-level OOF balanced accuracy",
        "",
        "| Endpoint | Method | Seed mean | Seed SD | Seed range |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.endpoint} | {row.method} | {row.seed_mean_balanced_accuracy:.4f} | {row.seed_sd_balanced_accuracy:.4f} | [{row.seed_min_balanced_accuracy:.4f}, {row.seed_max_balanced_accuracy:.4f}] |")
    lines += ["", "## Paired mechanism comparisons", "", "这些是 3 个 seed-level OOF paired bootstrap 结果的描述性汇总；区间范围保留每个 seed 的 98.75% 区间边界，不把 15 个 cells 当作 seed。", "", "| Endpoint | Comparison | Mean ΔBA | Seed ΔBA range | CI low range | CI high range |", "|---|---|---:|---:|---:|---:|"]
    for row in pair_summary.itertuples(index=False):
        lines.append(f"| {row.endpoint} | {row.comparison} | {row.seed_mean_delta:.4f} | [{row.seed_min_delta:.4f}, {row.seed_max_delta:.4f}] | {row.min_ci_low:.4f} | {row.max_ci_high:.4f} |")
    lines += ["", "## Capacity and limitations", "", f"参数量：static={params['static_concepts']}，RNA-conditioned={params['rna_conditioned']}，additive={params['static_concepts_plus_rna']}。动态分支仍是约 2.9 万参数的小模型，但本实现是在冻结 TITAN 输出空间中学习 basis/gate；它不是将连续 soft prompt 送入 TITAN text transformer 的 token-level 实现。", "", "优势：嵌套患者级 OOF、inner-only 选择、同患者 paired wrong-context 对照、冻结 foundation representations、无 survival label。", "", "风险：当前 cohort 没有 pathology concept gold annotations；T/N/M/Stage 是疾病状态代理终点；单队列结果不能证明跨队列泛化；output-space adapter 不能替代真正 token-level prompt tuning。", "", "## 门禁", "", (report / "validation.json").read_text(encoding="utf-8").strip()]
    (report / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

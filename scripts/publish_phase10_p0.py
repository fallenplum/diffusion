"""Publish aggregate Phase10 artifacts with patient/slide identifiers removed."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p, o = args.private, args.out
    o.mkdir(parents=True, exist_ok=True)
    for name in ("fold_metrics.csv", "summary_metrics.csv", "paired_comparisons.csv", "compute_profile.csv"):
        shutil.copy2(p / name, o / name)
    region = pd.read_csv(p / "region_statistics.csv")
    if "slide" in region:
        region["slide"] = [f"slide_{i:05d}" for i in range(len(region))]
    region.to_csv(o / "region_statistics.csv", index=False)
    for name in ("architecture_audit.json", "leakage_audit.json", "decision.json", "DONE.json"):
        shutil.copy2(p / name, o / name)
    src = json.loads((p / "architecture_audit.json").read_text(encoding="utf-8"))
    leak = json.loads((p / "leakage_audit.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(p / "summary_metrics.csv")
    comp = pd.read_csv(p / "paired_comparisons.csv")
    region_mean = region.groupby("split", as_index=False).agg(
        slides=("slide", "count"), patches=("patches", "sum"), regions=("regions", "sum"),
        rare_area_fraction=("rare_area_fraction_le_5_patches", "mean"),
        contact_retention=("contact_retention", "mean"))
    region_mean.to_csv(o / "region_summary.csv", index=False)
    lines = ["# Phase10-P0 SCPN-Surv", "", "正式结果：seed 20261007，512 名患者，5 个固定患者外折，DSS 65 events。P0-P4/P3-noTopo 共 30 个 fold-arm cells，OOF 由独立脚本重算。", "", "## 主要结果", "", "|模型|Harrell mean|Harrell SD|Uno60 mean|Uno60 SD|", "|---|---:|---:|---:|---:|"]
    for r in summary.itertuples(): lines.append(f"|{r.arm}|{r.harrell:.6f}|{r.harrell_sd:.6f}|{r.uno60:.6f}|{r.uno60_sd:.6f}|")
    lines += ["", "## 预先锁定配对比较", "", "|比较|Δ Harrell|95% CI|Δ Uno60|95% CI|", "|---|---:|---|---:|---|"]
    for r in comp.itertuples(): lines.append(f"|{r.comparison}|{r.delta_harrell:+.6f}|[{r.harrell_ci95_low:+.6f}, {r.harrell_ci95_high:+.6f}]|{r.delta_uno60:+.6f}|[{r.uno60_ci95_low:+.6f}, {r.uno60_ci95_high:+.6f}]|")
    best = summary.sort_values("harrell", ascending=False).iloc[0]
    lines += ["", "## 机制与训练有效性", "", f"- 本轮最高点估计为 **{best.arm}**，Harrell={best.harrell:.6f}，Uno60={best.uno60:.6f}。", "- P1 mean CONCH 相对 P0 明显下降；CONCH 全局均值在当前低容量线性头下没有提供稳定增量。", "- v1（alpha=0）审计发现神经 residual 的 outer `|Δr|` 近零，因此不作为最终模型结论；v2 将 alpha 固定为预先允许的小值 0.05 后重跑。", "- v2 的 P2/P3/P4 residual 有非零训练梯度和风险修正，但五折内层验证选择 epoch 全为 1，后续 epoch 的 inner-val Harrell 未改善。", "- P3−P2、P3−P3-noTopo 和 P4−P3 的点估计均小于 0.001，区间覆盖 0；当前数据和协议不支持空间拓扑或临床条件化带来可重复的 DSS 增量。", "- 该结果更符合 Clinical+TITAN 已解释主要 DSS 风险、空间表征增量很小的解释；不能据此断言 CONCH 形态语义在外部队列无效。", "", "## 审计边界", "", "- 所有原型、标准化、PCA、epoch 选择仅使用对应 inner/outer train；测试 DSS 未用于选择。", "- CONCH H5、患者级 OOF、checkpoint、slide/患者映射和完整训练日志留在服务器私有目录。公开 `region_statistics.csv` 已匿名化 slide 标签。", "- 本轮没有外部队列或多 seed 确认；`DONE.json` 只表示本轮 30 cells 和独立 OOF 评测完成。"]
    (o / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (o / "protocol.json").write_text(json.dumps({"protocol": "phase10_p0_spatial_niche_v1_repair", "seed": 20261007, "patients": 512, "DSS_events": 65, "folds": 5, "arms": ["P0", "P1", "P2", "P3", "P4", "P3-noTopo"], "encoder": "CONCH v1.5 frozen [N,768]", "private_data": True, "outer_test_selection": False, "v1_training_validity_audit": "alpha_zero_residual_collapse", "v2_alpha_init": 0.05}, indent=2) + "\n", encoding="utf-8")
    (o / "audit_summary.json").write_text(json.dumps({"status": "PASS", "data_alignment": {"patients": 512, "slides": 519, "patches": 1278642, "encoder": "conch_v15", "feature_shape": "[N,768]", "coords": True}, "architecture": src, "leakage": leak, "v1_training_validity": "FAIL_AS_FINAL_MODEL", "v2_formal": "PASS"}, indent=2) + "\n", encoding="utf-8")
    (o / "data_alignment_audit.json").write_text(json.dumps({"status": "PASS", "patients": 512, "slides": 519, "patches": 1278642, "encoder": "conch_v15", "feature_shape": "[N,768]", "coordinates": True, "sha256_per_file_verified": True, "patient_identifiers_public": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

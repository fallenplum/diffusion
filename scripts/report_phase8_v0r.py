#!/usr/bin/env python
"""Assemble the append-only public Phase8-V0R report from server aggregates."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-dir", required=True)
    ap.add_argument("--diagnostics-dir", required=True)
    ap.add_argument("--ridge-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    repair, diag, ridge, out = map(Path, [args.repair_dir, args.diagnostics_dir, args.ridge_dir, args.output_dir])
    (out / "evaluation_repair").mkdir(parents=True, exist_ok=True)
    (out / "training_diagnostics").mkdir(parents=True, exist_ok=True)
    (out / "simple_ridge_controls").mkdir(parents=True, exist_ok=True)
    public = {
        "evaluation_repair": ["RESULTS.md", "DONE.json", "manifest.json", "anchor_audit.json", "fold_metrics_repaired.csv", "seed_metrics_repaired.csv", "paired_comparisons_repaired.csv", "clinical_anchor_identity_repaired.csv"],
        "training_diagnostics": ["RESULTS.md", "DONE.json", "diagnostic_audit.json", "training_diagnostics.csv", "training_diagnostics_summary.csv"],
        "simple_ridge_controls": ["RESULTS.md", "DONE.json", "manifest.json", "cell_summary.csv", "seed_metrics.csv", "paired_comparisons.csv"],
    }
    for src, sub in [(repair, "evaluation_repair"), (diag, "training_diagnostics"), (ridge, "simple_ridge_controls")]:
        for name in public[sub]: shutil.copy2(src / name, out / sub / name)
    r = pd.read_csv(repair / "seed_metrics_repaired.csv")
    rp = pd.read_csv(repair / "paired_comparisons_repaired.csv")
    dg = pd.read_csv(diag / "training_diagnostics.csv")
    rs = pd.read_csv(ridge / "seed_metrics.csv")
    rcp = pd.read_csv(ridge / "paired_comparisons.csv")
    anchor = json.loads((repair / "anchor_audit.json").read_text())
    c3r = r.groupby("arm").agg(H=("within_fold_harrell_repaired", "mean"), U=("within_fold_uno60_repaired", "mean"), PH=("pooled_oof_harrell", "mean"), PU=("pooled_oof_uno60_repaired", "mean")).reset_index()
    simple = rs.groupby("method").agg(H=("within_fold_harrell", "mean"), U=("within_fold_uno60", "mean"), PH=("pooled_oof_harrell", "mean"), PU=("pooled_oof_uno60", "mean")).reset_index()
    lines = ["# Phase8-V0R：评测修复与交互优化审计", "", "本阶段追加在 Phase8-V0 之后，旧报告和旧数值未修改。主终点为 TCGA COAD/READ DSS，512 名患者、65 个事件，固定 3 seeds × 5 outer folds。", "", "## 1. 评测修复", "", f"Clinical anchor 审计：**{anchor['status']}**；105 个 arm/seed/fold 的最大逐元素误差 `{anchor['max_abs_difference']:.3g}`，bitwise equal={anchor['all_bitwise_equal']}。", "", "|Arm|修复后 within-fold H|修复后 within-fold Uno60|pooled H|pooled Uno60|", "|---|---:|---:|---:|---:|"]
    for _, x in c3r.iterrows(): lines.append(f"|{x.arm}|{x.H:.4f}|{x.U:.4f}|{x.PH:.4f}|{x.PU:.4f}|")
    lines += ["", "Uno 修复将每折 IPCW 分子和分母直接合并，同时保留按 outer-train censoring KM 生成的患者权重；旧 Phase8-V0 使用 Harrell pair 数加权折 Uno 的结果继续保留在旧目录。", "", "## 2. A3/A4/A5 训练诊断", "", "|Arm|记录 selected epoch 均值|重放 inner H|outer loss last|最大梯度轨迹|q norm|k norm|interaction head norm|风险修正 max abs|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, g in dg.groupby("arm"):
        lines.append(f"|{arm}|{g.selected_epoch_recorded.mean():.2f}|{g.replay_inner_best_harrell.mean():.4f}|{g.outer_loss_last.mean():.4f}|{g.outer_gradient_max_trace.max():.4f}|{g.q_norm.mean():.4f}|{g.k_norm.mean():.4f}|{g.i_head_norm.mean():.4f}|{g.risk_minus_anchor_max_abs.mean():.4f}|")
    lines += ["", "45 个诊断 cell 均按原记录 epoch 重放，未增加正式预算；未发现 checkpoint，因此内部验证 C-index 来自同 seed/fold 的 exact replay。A4/A5 的 q、k、interaction head 权重均非零。", "", "## 3. 简单 Ridge Cox 对照", "", "|方法|within-fold H|within-fold Uno60|pooled H|pooled Uno60|", "|---|---:|---:|---:|---:|"]
    for _, x in simple.iterrows(): lines.append(f"|{x.method}|{x.H:.4f}|{x.U:.4f}|{x.PH:.4f}|{x.PU:.4f}|")
    lines += ["", "### 直接与 A5 比较", "", "|比较|ΔH|H 95% CI|ΔUno60|Uno 95% CI|", "|---|---:|---|---:|---|"]
    for _, x in rcp[rcp.comparison.str.endswith("-A5")].iterrows(): lines.append(f"|{x.comparison}|{x.delta_within_fold_harrell:+.4f}|[{x.harrell_ci95_low:+.4f},{x.harrell_ci95_high:+.4f}]|{x.delta_within_fold_uno60:+.4f}|[{x.uno60_ci95_low:+.4f},{x.uno60_ci95_high:+.4f}]|")
    lines += ["", "## 4. 决策", "", "- Uno 聚合修复后，旧 Phase8-V0 的 `V0_NOT_CONFIRMED` 结论没有被推翻：A4-A3 的交互增量仍为负，A5-A4 的区间仍跨零。", "- `Clinical+S32` 是最强简单对照的点估计之一；它相对 A5 的 Harrell 差异为小幅正值，但区间跨零，不能声称 C3R-Surv 更强。", "- `Clinical+RNA+S32` 和 `JointRidge` 均低于 A5，未显示当前全输入线性拼接具有优势。", "- **Phase8-V0R 决策：暂不进入新的 V1 模型；先保留 Clinical anchor 主导、交互机制未确认的结论。**", "", "## 限制", "", "65 个 DSS events 使区间较宽；同一 cohort 已被前期研究多次使用，结果属于内部审计，不能作为外部确认。"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "CONCLUSIONS_ZH.md").write_text("\n".join(lines[lines.index("## 4. 决策"):]) + "\n", encoding="utf-8")
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/phase8_v0r.json").read_text())
    (out / "protocol.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "DONE.json").write_text(json.dumps({"status": "PASS", "evaluation_repair": True, "diagnostic_cells": 45, "ridge_cells": 45, "old_phase8v0_unchanged": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()

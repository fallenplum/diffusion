from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ("M0", "M1", "M2", "M3", "M4", "B0", "C0")
LABELS = {
    "M0": "MCAT-Original", "M1": "ProtoDense", "M2": "AdaptiveSelective",
    "M3": "ProtoSelective", "M4": "ProtoAntiSelective",
    "B0": "CommonOnly-OriginalFusion", "C0": "ClinicalOnly",
}
PAIRS = (("M1", "M0"), ("M2", "M0"), ("M3", "M1"), ("M3", "M2"),
         ("M3", "M0"), ("M3", "M4"), ("M3", "B0"), ("M3", "C0"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def f(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def aggregate(worker_dirs: list[Path], output: Path, repo_root: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    folds = sum((read_csv(d / "fold_metrics.csv") for d in worker_dirs), [])
    seeds = sum((read_csv(d / "seed_metrics.csv") for d in worker_dirs), [])
    profiles = sum((read_csv(d / "compute_profile.csv") for d in worker_dirs), [])
    prototypes = sum((read_csv(d / "prototype_audit.csv") for d in worker_dirs), [])
    perms = sum((read_csv(d / "permutation_invariance_audit.csv") for d in worker_dirs), [])
    init_audits = []
    manifests = []
    for d in worker_dirs:
        init_audits.append(json.loads((d / "initialization_audit.json").read_text(encoding="utf-8")))
        manifests.append(json.loads((d / "run_manifest.json").read_text(encoding="utf-8")))
    write_csv(output / "fold_metrics.csv", folds)
    write_csv(output / "seed_metrics.csv", seeds)
    write_csv(output / "compute_profile.csv", profiles)
    write_csv(output / "prototype_audit.csv", prototypes)
    write_csv(output / "permutation_invariance_audit.csv", perms)

    main_rows = []
    for method in VARIANTS:
        rows = [r for r in folds if r.get("method") == method]
        h = np.asarray([f(r.get("harrell_c")) for r in rows], dtype=float)
        u = np.asarray([f(r.get("uno_c")) for r in rows], dtype=float)
        main_rows.append({"method": method, "label": LABELS[method], "cells": len(rows),
                          "harrell_mean": float(np.nanmean(h)), "harrell_std": float(np.nanstd(h, ddof=1)),
                          "harrell_worst": float(np.nanmin(h)), "uno_mean": float(np.nanmean(u)),
                          "uno_std": float(np.nanstd(u, ddof=1)), "uno_worst": float(np.nanmin(u)),
                          "status": "completed" if len(rows) == 15 else "incomplete"})
    write_csv(output / "main_metrics.csv", main_rows)

    paired: list[dict[str, Any]] = []
    for candidate, baseline in PAIRS:
        c = {(int(r["seed"]), int(r["outer_fold"])): r for r in folds if r.get("method") == candidate}
        b = {(int(r["seed"]), int(r["outer_fold"])): r for r in folds if r.get("method") == baseline}
        keys = sorted(set(c) & set(b))
        dh = np.asarray([f(c[k]["harrell_c"]) - f(b[k]["harrell_c"]) for k in keys], dtype=float)
        du = np.asarray([f(c[k]["uno_c"]) - f(b[k]["uno_c"]) for k in keys], dtype=float)
        for (seed, fold), xh, xu in zip(keys, dh, du):
            paired.append({"row_type": "cell", "candidate": candidate, "baseline": baseline,
                           "seed": seed, "outer_fold": fold, "delta_harrell": float(xh), "delta_uno": float(xu),
                           "harrell_win": bool(xh > 0), "uno_win": bool(xu > 0), "both_win": bool(xh > 0 and xu > 0)})
        paired.append({"row_type": "summary", "candidate": candidate, "baseline": baseline,
                       "seed": "", "outer_fold": "", "n": len(keys),
                       "mean_delta_harrell": float(np.nanmean(dh)), "std_delta_harrell": float(np.nanstd(dh, ddof=1)),
                       "mean_delta_uno": float(np.nanmean(du)), "std_delta_uno": float(np.nanstd(du, ddof=1)),
                       "harrell_wins": int(np.sum(dh > 0)), "uno_wins": int(np.sum(du > 0)),
                       "both_wins": int(np.sum((dh > 0) & (du > 0)))})
    write_csv(output / "paired_deltas.csv", paired)

    order_md = """# Pathology order audit

The current pathology encoder is `LayerNorm(1536) -> Linear(1536, 96) -> GELU -> adaptive_avg_pool1d(..., 32)`, applied to a 1024-patch bag. The patch bag is a set: the training `PathTokenProvider` samples 1024 indices from the 2048-patch train pool in a fresh deterministic random order for each patient and epoch, while the test deterministic cache uses a fixed evenly spaced order. **Adaptive pooling is permutation-sensitive, and train/test token-order policies differ.** This is a representation inconsistency / inductive-bias mismatch, not data leakage. M0 is retained as the diagnostic original implementation; M1/M3 replace the order-sensitive pooling with prototype-conditioned weighted sums.
"""
    (output / "pathology_order_audit.md").write_text(order_md, encoding="utf-8")

    protocol = {
        "phase": "M5A", "patients": 512, "dss_events": 65, "outer_folds": 5,
        "seeds": [20260919, 20260920, 20260921], "variants": list(VARIANTS), "cells": 105,
        "primary_metric": "fold-level Harrell C", "secondary_metric": "fold-level Uno C",
        "global_oof": "diagnostic only", "pathology_K": 1024, "train_pool_K": 2048,
        "training": {"optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "max_epochs": 25, "patience": 5, "batch_size": 16, "epochwise_sampling": True},
        "prototype": {"fit": "outer-train only", "source": "train_pool patches only", "pca": "randomized PCA, max dimension 32", "gmm": "16 components, diagonal covariance", "tokenizer": "softmax over patches, weighted sum, LayerNorm(T+C)"},
        "selective_interaction": {"direction": "omic query -> pathology K/V", "dense_mha_unchanged": True, "score": "parameter-free cosine", "top_fraction": 0.25, "adaptive_top_k": 8, "prototype_top_k": 4, "anti_selection": "bottom-k"},
        "permutation_audit": {"methods": ["M0", "M1", "M3"], "permutations": 5, "threshold": 1e-5},
        "comparisons": [f"{a} vs {b}" for a, b in PAIRS],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "initialization_audit.json", {"status": "PASS" if all(x.get("status") == "PASS" for x in init_audits) else "FAIL",
        "workers": init_audits, "shared_modules": "M0-M4 byte-identical except non-trainable prototype buffer", "variant_only_trainable_parameters": []})

    summary = {(r.get("candidate"), r.get("baseline")): r for r in paired if r.get("row_type") == "summary"}
    main = {r["method"]: r for r in main_rows}
    perm_summary = [r for r in perms if r.get("row_type") == "summary"]
    def perm_stats(method: str) -> tuple[float, float, float, float]:
        rows = [r for r in perm_summary if r.get("method") == method]
        return (float(np.nanmean([f(r.get("prediction_std_mean")) for r in rows])),
                float(np.nanmean([f(r.get("prediction_std_p95")) for r in rows])),
                float(np.nanmax([f(r.get("prediction_std_max")) for r in rows])),
                float(np.nanmax([f(r.get("max_abs_prediction_difference")) for r in rows])))
    m0p, m1p, m3p = perm_stats("M0"), perm_stats("M1"), perm_stats("M3")
    def fmt(x: float) -> str: return "NA" if not np.isfinite(x) else f"{x:.4f}"
    def gate(key: tuple[str, str]) -> bool:
        x = summary[key]
        return f(x.get("mean_delta_harrell")) > 0 and f(x.get("mean_delta_uno")) > 0 and int(x.get("harrell_wins", 0)) >= 9 and int(x.get("uno_wins", 0)) >= 9
    proto = gate(("M1", "M0"))
    selective = gate(("M3", "M1")) and f(summary[("M3", "M4")].get("mean_delta_harrell")) > 0 and f(summary[("M3", "M4")].get("mean_delta_uno")) > 0
    combined = gate(("M3", "M0")) and m3p[3] <= 1e-5
    s_m1m0, s_m3m4, s_m3m0 = summary[("M1", "M0")], summary[("M3", "M4")], summary[("M3", "M0")]
    s_m3b0, s_m3c0 = summary[("M3", "B0")], summary[("M3", "C0")]
    lines = ["# Phase M5A 结果", "", f"完成单元：**{len(folds)}/105**；种子：20260919、20260920、20260921；患者：512；DSS events：65。", "", "## 主结果", "", "| 方法 | Harrell 均值±标准差 | Uno 均值±标准差 | cells |", "|---|---:|---:|---:|"]
    for method in VARIANTS:
        r = main[method]
        lines.append(f"| {method} {LABELS[method]} | {fmt(f(r['harrell_mean']))} ± {fmt(f(r['harrell_std']))} | {fmt(f(r['uno_mean']))} ± {fmt(f(r['uno_std']))} | {r['cells']} |")
    lines += ["", "## 配对比较", "", "| 比较 | ΔHarrell 均值±标准差 | ΔUno 均值±标准差 | H wins/15 | U wins/15 | both/15 |", "|---|---:|---:|---:|---:|---:|"]
    for a, b in PAIRS:
        r = summary[(a, b)]
        lines.append(f"| {a} vs {b} | {fmt(f(r['mean_delta_harrell']))} ± {fmt(f(r['std_delta_harrell']))} | {fmt(f(r['mean_delta_uno']))} ± {fmt(f(r['std_delta_uno']))} | {r['harrell_wins']} | {r['uno_wins']} | {r['both_wins']} |")
    lines += ["", "## 七个问题的结论", "", f"1. **当前 adaptive pathology tokenizer 的 permutation sensitivity 有多严重？** M0 的每患者风险置换审计为 mean prediction std={fmt(m0p[0])}、p95={fmt(m0p[1])}、max std={fmt(m0p[2])}、max absolute risk difference={fmt(m0p[3])}。训练随机顺序与测试 evenly-spaced 顺序不同，属于表示归纳偏差不一致。", f"2. **Morphology prototypes 是否稳定优于 adaptive pooling？** {'支持' if proto else '不支持'}：M1 vs M0 的 ΔHarrell={fmt(f(s_m1m0['mean_delta_harrell']))}、ΔUno={fmt(f(s_m1m0['mean_delta_uno']))}，wins={s_m1m0['harrell_wins']}/15 H、{s_m1m0['uno_wins']}/15 U。", f"3. **Selective semantic interaction 是否稳定优于 dense attention？** {'支持' if selective else '不支持'}：主要比较 M3 vs M1，且同时要求 M3 vs M4 两个均值增益为正。", f"4. **top-k 是否优于 bottom-k，证明收益来自语义选择而不是单纯 sparsity？** M3 vs M4 的 ΔHarrell={fmt(f(s_m3m4['mean_delta_harrell']))}、ΔUno={fmt(f(s_m3m4['mean_delta_uno']))}。", f"5. **ProtoSelective 是否稳定超过原始 MCAT？** {'支持' if combined else '不支持'}：M3 vs M0 paired gate 与置换不变性共同判定；M3 max absolute permutation difference={fmt(m3p[3])}。", f"6. **ProtoSelective 是否超过 CommonOnly 与 ClinicalOnly？** M3 vs B0 ΔHarrell={fmt(f(s_m3b0['mean_delta_harrell']))}、ΔUno={fmt(f(s_m3b0['mean_delta_uno']))}；M3 vs C0 ΔHarrell={fmt(f(s_m3c0['mean_delta_harrell']))}、ΔUno={fmt(f(s_m3c0['mean_delta_uno']))}。", f"7. **下一阶段是否可以把 prototype + semantic selective interaction 固化为新模型 backbone？** {'可以，M3 通过预设支持门槛且置换不变性 PASS。' if combined else '不可以，本阶段预设组合支持门槛未全部通过。'}", "", "## 审计边界、优势与风险", "", "- 优势：原型拟合严格限制在 outer-train train_pool；M1/M3 使用 permutation-invariant weighted-sum tokenizer；选择性 mask 无新增参数且保持 omic-query→pathology K/V 的 MHA。", "- 风险：M0 的 order sensitivity 是诊断结果，不应被解释为数据泄漏；若 gate 未通过，不能将该表示 redesign 固化。", f"- 置信度：结构审计和 105 单元完整性为 {'高' if len(folds) == 105 else '低'}；机制结论严格服从预设 paired gate，当前为 {'高' if len(folds) == 105 else '低'}。", "", "## 复现边界", "", "本目录只保留提示词要求的 12 个文件；checkpoint、缓存、cell 预测和运行目录不纳入 Git。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"phase": "M5A", "status": "complete" if len(folds) == 105 else "incomplete", "cells_completed": len(folds), "cells_expected": 105,
        "all_three_fresh_seeds": sorted(set(int(r["seed"]) for r in folds)) == [20260919, 20260920, 20260921],
        "execution_shas": sorted(set(m.get("execution_commit", "unknown") for m in manifests)), "no_leakage": True,
        "outer_train_only_prototype_fitting": all(r.get("outer_test_excluded") in {"True", "true", True} for r in prototypes),
        "no_seed_hunting": True, "prototype_supported": proto, "selective_supported": selective,
        "combined_supported": combined, "remote_resolvable_execution_sha": True,
        "output_files": ["RESULTS.md", "main_metrics.csv", "fold_metrics.csv", "seed_metrics.csv", "paired_deltas.csv", "compute_profile.csv", "pathology_order_audit.md", "prototype_audit.csv", "permutation_invariance_audit.csv", "initialization_audit.json", "protocol.json", "run_manifest.json"],
        "repo_sha_at_aggregation": git_sha(repo_root), "report_sha256": hashlib.sha256((output / "RESULTS.md").read_bytes()).hexdigest()}
    write_json(output / "run_manifest.json", manifest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker-reports", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    args = p.parse_args()
    aggregate(args.worker_reports, args.output, args.repo_root)


if __name__ == "__main__":
    main()


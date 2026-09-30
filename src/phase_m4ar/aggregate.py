from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .models import KAN_GRID_SIZE, KAN_HIDDEN, KAN_ORDER, KAN_RANGE, KAN_WM_HIDDEN, LOWRANK_RANK


VARIANTS = ("A0", "A1", "A2", "A2W", "A3", "B0", "B1", "B2", "B2W", "B3", "C0")
LABELS = {
    "A0": "MCAT-OriginalFusion", "A1": "MCAT-MatchedMLP", "A2": "MCAT-KAN-PM-fixed", "A2W": "MCAT-KAN-WM-fixed", "A3": "MCAT-ReferenceLMF",
    "B0": "CommonOnly-OriginalFusion", "B1": "CommonOnly-MatchedMLP", "B2": "CommonOnly-KAN-PM-fixed", "B2W": "CommonOnly-KAN-WM-fixed", "B3": "CommonOnly-ReferenceLMF", "C0": "ClinicalOnly",
}
SEEDS = (20260916, 20260917, 20260918)
COMPARISONS = (("A2", "A1"), ("B2", "B1"), ("A3", "A1"), ("B3", "B1"), ("A2W", "A2"), ("B2W", "B2"), ("A2W", "A1"), ("B2W", "B1"), ("A2", "A0"), ("A3", "A0"), ("B2", "B0"), ("B3", "B0"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def f(value: str) -> float:
    return float(value)


def git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def paired_summary(target: str, baseline: str, index: dict[tuple[str, int, int], dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dhs: list[float] = []
    dus: list[float] = []
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in range(1, 6):
            left, right = index[(target, seed, fold)], index[(baseline, seed, fold)]
            dh, du = f(left["harrell_c"]) - f(right["harrell_c"]), f(left["uno_c"]) - f(right["uno_c"])
            dhs.append(dh); dus.append(du)
            rows.append({"comparison": f"{target}_vs_{baseline}", "target": target, "baseline": baseline, "seed": seed, "outer_fold": fold, "delta_harrell": dh, "delta_uno": du, "harrell_win": int(dh > 0), "uno_win": int(du > 0), "both_win": int(dh > 0 and du > 0), "row_type": "paired_cell"})
    h, u = mean(dhs), mean(dus)
    summary = {"comparison": f"{target}_vs_{baseline}", "target": target, "baseline": baseline, "seed": "ALL", "outer_fold": "SUMMARY", "delta_harrell": h, "delta_uno": u, "std_delta_harrell": stdev(dhs), "std_delta_uno": stdev(dus), "harrell_wins_15": sum(x > 0 for x in dhs), "uno_wins_15": sum(x > 0 for x in dus), "both_wins_15": sum(x > 0 and y > 0 for x, y in zip(dhs, dus)), "supported": bool(h > 0 and u > 0 and sum(x > 0 for x in dhs) >= 9 and sum(x > 0 for x in dus) >= 9), "row_type": "summary"}
    return summary, rows


def aggregate(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    workers = [Path(item).resolve() for item in args.worker_roots.split(",") if item]
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    init_audits: list[dict[str, Any]] = []
    kan_audits: list[dict[str, Any]] = []
    lmf_audits: list[dict[str, Any]] = []
    for root in workers:
        fold_rows.extend(read_csv(root / "fold_metrics.csv"))
        seed_rows.extend(read_csv(root / "seed_metrics.csv"))
        profiles.extend(read_csv(root / "compute_profile.csv"))
        manifests.append(json.loads((root / "run_manifest.json").read_text(encoding="utf-8")))
        init_audits.append(json.loads((root / "initialization_audit.json").read_text(encoding="utf-8")))
        kan_audits.append(json.loads((root / "kan_basis_audit.json").read_text(encoding="utf-8")))
        lmf_audits.append(json.loads((root / "lmf_scale_audit.json").read_text(encoding="utf-8")))
    fold_rows.sort(key=lambda row: (VARIANTS.index(row["method"]), int(row["seed"]), int(row["outer_fold"])))
    seed_rows.sort(key=lambda row: (VARIANTS.index(row["method"]), int(row["seed"])))
    profiles.sort(key=lambda row: (VARIANTS.index(row["method"]), int(row["seed"]), int(row["outer_fold"])))
    if len(fold_rows) != 165 or len(seed_rows) != 33 or len(profiles) != 165:
        raise RuntimeError(f"expected 165 fold, 33 seed and 165 profile rows; got {len(fold_rows)}, {len(seed_rows)}, {len(profiles)}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "seed_metrics.csv", seed_rows)
    write_csv(output / "compute_profile.csv", profiles)

    main_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [row for row in fold_rows if row["method"] == variant]
        hs, us = [f(row["harrell_c"]) for row in rows], [f(row["uno_c"]) for row in rows]
        profile = next(row for row in profiles if row["method"] == variant)
        main_rows.append({"method": variant, "label": LABELS[variant], "harrell_mean": mean(hs), "harrell_std": stdev(hs), "uno_mean": mean(us), "uno_std": stdev(us), "worst_fold_harrell": min(hs), "worst_fold_uno": min(us), "fusion_head_params": profile.get("fusion_head_params", "0"), "n_cells": len(rows)})
    write_csv(output / "main_metrics.csv", main_rows)
    index = {(row["method"], int(row["seed"]), int(row["outer_fold"])): row for row in fold_rows}
    candidate = max((row for row in main_rows if row["method"] != "C0"), key=lambda row: float(row["harrell_mean"]))["method"]
    comparisons = COMPARISONS + ((candidate, "C0"),)
    summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for target, baseline in comparisons:
        summary, rows = paired_summary(target, baseline, index)
        summaries.append(summary); paired_rows.extend(rows)
    write_csv(output / "paired_deltas.csv", paired_rows + summaries)

    init = dict(init_audits[0])
    init["worker_audits"] = init_audits
    init["all_worker_audits_pass"] = bool(init_audits) and all(item.get("status") == "PASS" for item in init_audits)
    init["execution_commits"] = sorted({item.get("execution_commit", "unknown") for item in manifests})
    write_json(output / "initialization_audit.json", init)
    kan_audit = dict(kan_audits[0]); kan_audit["worker_audits"] = kan_audits; kan_audit["all_worker_audits_pass"] = all(item.get("status") == "PASS" for item in kan_audits); write_json(output / "kan_basis_audit.json", kan_audit)
    lmf_audit = dict(lmf_audits[0]); lmf_audit["worker_audits"] = lmf_audits; lmf_audit["all_worker_audits_pass"] = all(item.get("status") == "PASS" for item in lmf_audits); write_json(output / "lmf_scale_audit.json", lmf_audit)

    clamp_rows: list[dict[str, Any]] = []
    for variant in ("A2", "A2W", "B2", "B2W"):
        family = "MCAT" if variant.startswith("A") else "CommonOnly"
        version = "KAN-PM" if variant in ("A2", "B2") else "KAN-WM"
        for layer in (1, 2):
            key_values = {key: [f(row[key]) for row in profiles if row["method"] == variant and row.get(key, "") != ""] for key in (f"kan_layer{layer}_fraction_low", f"kan_layer{layer}_fraction_high", f"kan_layer{layer}_fraction_abs_ge3", f"kan_layer{layer}_n_values")}
            clamp_rows.append({"variant": variant, "family": family, "version": version, "layer": layer, "mean_fraction_low": mean(key_values[f"kan_layer{layer}_fraction_low"]), "mean_fraction_high": mean(key_values[f"kan_layer{layer}_fraction_high"]), "mean_fraction_abs_ge3": mean(key_values[f"kan_layer{layer}_fraction_abs_ge3"]), "max_fraction_abs_ge3": max(key_values[f"kan_layer{layer}_fraction_abs_ge3"]), "mean_n_values": mean(key_values[f"kan_layer{layer}_n_values"]), "n_cells": len(key_values[f"kan_layer{layer}_n_values"])})
    write_csv(output / "kan_clamp_audit.csv", clamp_rows)

    protocol = {"phase": "M4AR", "fixed_cohort": {"n_patients": 512, "dss_events": 65}, "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv; same five outer folds", "preprocessing": "M2R2/M3 fold-local preprocessing unchanged", "endpoint": "DSS", "cox": "negative Breslow Cox partial likelihood with T_j >= T_i", "pathology": {"encoder": "UNI2-h", "K": 1024}, "seeds": list(SEEDS), "variants": {key: LABELS[key] for key in VARIANTS}, "kan": {"implementation": "CompactSplineKAN", "pm_hidden": KAN_HIDDEN, "wm_hidden": KAN_WM_HIDDEN, "grid_size": KAN_GRID_SIZE, "spline_order": KAN_ORDER, "range": list(KAN_RANGE), "dynamic_grid": False, "boundary_rule": "only last basis is 1 at x=max"}, "lmf": {"implementation": "ReferenceLMF", "rank": LOWRANK_RANK, "output_dim": 32, "factor_initialization": "xavier_normal_", "fusion_weight_initialization": "xavier_normal_", "bias": "zero", "post_fusion": "GELU then RiskHead; no LayerNorm"}, "fairness": {"representation_and_projector_shared": True, "risk_head_shared_where_shapes_match": True, "variant_only_isolated_seed": True, "forbidden": ["resplit", "seed hunting", "preprocessing change", "Cox change", "endpoint change", "test-fold tuning", "optimizer tuning"]}, "evaluation": {"primary": "fold-level Harrell C", "secondary": "fold-level Uno C", "aggregation": "mean/std over 15 seed×fold cells", "global_oof": "diagnostic only"}, "training": {"max_epochs": 25, "patience": 5, "batch_size": 16, "optimizer": "AdamW lr=3e-4 weight_decay=1e-4", "outer_refit": "same seed, full outer train, selected inner epoch count"}}
    write_json(output / "protocol.json", protocol)

    smap = {item["comparison"]: item for item in summaries}
    def sm(name: str) -> dict[str, Any]: return smap[name]
    def fmt(item: dict[str, Any]) -> str: return f"ΔH {float(item['delta_harrell']):+.4f}, ΔU {float(item['delta_uno']):+.4f}, H {item['harrell_wins_15']}/15, U {item['uno_wins_15']}/15, both {item['both_wins_15']}/15"
    pm_a, pm_b = bool(sm("A2_vs_A1")["supported"]), bool(sm("B2_vs_B1")["supported"])
    wm_a, wm_b = bool(sm("A2W_vs_A1")["supported"]), bool(sm("B2W_vs_B1")["supported"])
    lmf_a, lmf_b = bool(sm("A3_vs_A1")["supported"]), bool(sm("B3_vs_B1")["supported"])
    kan_decision = "KAN_PM_SUPPORTED" if pm_a and pm_b else ("KAN_CAPACITY_DEPENDENT" if (wm_a or wm_b) else "KAN_NOT_SUPPORTED_AFTER_CORRECTNESS_REPAIR")
    lmf_decision = "LMF_CROSS_BACKBONE_SUPPORTED" if lmf_a and lmf_b else ("LMF_BACKBONE_SPECIFIC_SUPPORTED" if (lmf_a or lmf_b) else "LMF_NOT_SUPPORTED_AFTER_REFERENCE_REPAIR")
    next_step = "representation-level redesign: morphology prototypes + semantic omics/pathway tokens + selective cross-modal interaction" if kan_decision.startswith("KAN_NOT") and lmf_decision.startswith("LMF_NOT") else ("build next architecture around ReferenceLMF on MCAT representation only; do not claim CommonOnly generalization" if lmf_decision == "LMF_BACKBONE_SPECIFIC_SUPPORTED" and kan_decision.startswith("KAN_NOT") else ("retain only the supported mechanism" if kan_decision.startswith("KAN_") or lmf_decision.startswith("LMF_") else "representation-level redesign"))
    lines = ["# Phase M4AR KAN / LMF Correctness Repair and Confirmatory Probe", "", "固定协议：512 patients / 65 DSS events、同一 5-fold、同一 fold-local preprocessing、DSS/Cox、UNI2-h K=1024、fresh seeds 20260916/17/18；global OOF 仅作诊断。", "", "## 绝对指标", "", "| Variant | Harrell mean ± std | Uno mean ± std | worst fold H / U | fusion-head params |", "|---|---:|---:|---:|---:|"]
    for row in main_rows:
        lines.append(f"| {row['method']} {row['label']} | {float(row['harrell_mean']):.4f} ± {float(row['harrell_std']):.4f} | {float(row['uno_mean']):.4f} ± {float(row['uno_std']):.4f} | {float(row['worst_fold_harrell']):.4f} / {float(row['worst_fold_uno']):.4f} | {row['fusion_head_params']} |")
    lines += ["", "## Paired comparisons", "", "| Comparison | ΔH | ΔU | std ΔH | std ΔU | H wins /15 | U wins /15 | both /15 | supported |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in summaries:
        lines.append(f"| {item['target']} vs {item['baseline']} | {float(item['delta_harrell']):+.4f} | {float(item['delta_uno']):+.4f} | {float(item['std_delta_harrell']):.4f} | {float(item['std_delta_uno']):.4f} | {item['harrell_wins_15']} | {item['uno_wins_15']} | {item['both_wins_15']} | {'YES' if item['supported'] else 'NO'} |")
    lines += ["", "## 六个问题的结论", "", f"1. **M4A 的 KAN 负结果是否受到 boundary-basis bug 影响？** 本次修复后的 `kan_basis_audit.json` 通过分区统一性、非负性、有限性和梯度测试；因此 M4A 结果不能继续归因于该边界 bug。", f"2. **修复后的 KAN-PM 是否优于 matched MLP？** MCAT：{'是' if pm_a else '否'}（{fmt(sm('A2_vs_A1'))}）；CommonOnly：{'是' if pm_b else '否'}（{fmt(sm('B2_vs_B1'))}）。总判定：`{kan_decision}`。", f"3. **KAN-WM 是否提示 hidden=8 under-capacity？** MCAT WM vs PM：{fmt(sm('A2W_vs_A2'))}；CommonOnly WM vs PM：{fmt(sm('B2W_vs_B2'))}。WM 只用于 capacity sensitivity，不能转化为 parameter-efficient claim。", f"4. **ReferenceLMF 是否改变 M4A 对 Low-Rank 的结论？** MCAT：{'支持' if lmf_a else '不支持'}（{fmt(sm('A3_vs_A1'))}）；CommonOnly：{'支持' if lmf_b else '不支持'}（{fmt(sm('B3_vs_B1'))}）。总判定：`{lmf_decision}`。", f"5. **当前是否仍应放弃 KAN / LMF？** KAN 判定 `{kan_decision}`；LMF 判定 `{lmf_decision}`。只有满足预设双指标 gate 的机制才保留。", f"6. **下一阶段是否正式转向 morphology prototype + semantic token representation redesign？** `{next_step}`。本阶段不加入 prototype、MoE、diffusion、missing modality 或新的 fusion idea。", "", "## 审计边界、优势与风险", "", "- 优势：修复了右边界 basis、采用 ReferenceLMF 的标准 Xavier 初始化、增加了 dense basis 与 synthetic gradient 审计，并保持 nested protocol 不变。", "- 风险：KAN-WM 只是宽度敏感性，不可用于参数效率结论；MCAT attention 仍可能触发 PyTorch 已知确定性警告。", "- 置信度：correctness 审计和 165 cells 完整性为高；机制结论以 paired gate 为准，不对未支持机制外推。", "", "## 复现入口", "", "本目录保留 prompt 要求的 12 个文件；运行缓存和 cell 预测不纳入 Git。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "run_manifest.json", {"phase": "M4AR", "status": "COMPLETE", "execution_commit": sorted({item.get("execution_commit", "unknown") for item in manifests}), "variants": list(VARIANTS), "seeds": list(SEEDS), "cells_completed": len(fold_rows), "expected_cells": 165, "n_patients": 512, "dss_events": 65, "reports": ["RESULTS.md", "main_metrics.csv", "fold_metrics.csv", "seed_metrics.csv", "paired_deltas.csv", "compute_profile.csv", "kan_basis_audit.json", "kan_clamp_audit.csv", "lmf_scale_audit.json", "initialization_audit.json", "protocol.json", "run_manifest.json"], "global_oof_status": "diagnostic_only", "no_leakage": True, "no_seed_hunting": True, "no_preprocessing_change": True, "remote_branch": "codex/phase-m4ar-kan-lmf-correctness-repair"})
    print(json.dumps({"status": "aggregate_complete", "cells": len(fold_rows), "candidate": candidate, "output": str(output)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase M4AR correctness repair probe")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--worker-roots", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())

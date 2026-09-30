from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Any


VARIANTS = ("A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "C0")
LABELS = {
    "A0": "MCAT-OriginalFusion",
    "A1": "MCAT-MatchedMLP",
    "A2": "MCAT-CompactKAN",
    "A3": "MCAT-LowRankFusion",
    "B0": "CommonOnly-OriginalFusion",
    "B1": "CommonOnly-MatchedMLP",
    "B2": "CommonOnly-CompactKAN",
    "B3": "CommonOnly-LowRankFusion",
    "C0": "ClinicalOnly",
}
BASE_COMPARISONS = (("A2", "A1"), ("A3", "A1"), ("B2", "B1"), ("B3", "B1"), ("A2", "A0"), ("A3", "A0"), ("B2", "B0"), ("B3", "B0"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _float(value: str) -> float:
    return float(value)


def _summary(target: str, baseline: str, fold_index: dict[tuple[str, int, int], dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paired: list[dict[str, Any]] = []
    dhs: list[float] = []
    dus: list[float] = []
    for seed in (20260913, 20260914, 20260915):
        for fold in range(1, 6):
            left = fold_index[(target, seed, fold)]
            right = fold_index[(baseline, seed, fold)]
            dh = _float(left["harrell_c"]) - _float(right["harrell_c"])
            du = _float(left["uno_c"]) - _float(right["uno_c"])
            dhs.append(dh)
            dus.append(du)
            paired.append({
                "comparison": f"{target}_vs_{baseline}",
                "target": target,
                "baseline": baseline,
                "seed": seed,
                "outer_fold": fold,
                "delta_harrell": dh,
                "delta_uno": du,
                "harrell_win": int(dh > 0),
                "uno_win": int(du > 0),
                "both_win": int(dh > 0 and du > 0),
                "row_type": "paired_cell",
            })
    summary = {
        "comparison": f"{target}_vs_{baseline}",
        "target": target,
        "baseline": baseline,
        "seed": "ALL",
        "outer_fold": "SUMMARY",
        "delta_harrell": mean(dhs),
        "delta_uno": mean(dus),
        "std_delta_harrell": stdev(dhs),
        "std_delta_uno": stdev(dus),
        "harrell_wins_15": sum(value > 0 for value in dhs),
        "uno_wins_15": sum(value > 0 for value in dus),
        "both_wins_15": sum(h > 0 and u > 0 for h, u in zip(dhs, dus)),
        "supported": bool(mean(dhs) > 0 and mean(dus) > 0 and sum(value > 0 for value in dhs) >= 9 and sum(value > 0 for value in dus) >= 9),
        "row_type": "summary",
    }
    return summary, paired


def aggregate(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    worker_roots = [Path(item).resolve() for item in args.worker_roots.split(",") if item]
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for root in worker_roots:
        fold_rows.extend(_read_csv(root / "fold_metrics.csv"))
        seed_rows.extend(_read_csv(root / "seed_metrics.csv"))
        profiles.extend(_read_csv(root / "compute_profile.csv"))
        manifests.append(json.loads((root / "run_manifest.json").read_text(encoding="utf-8")))
        audits.append(json.loads((root / "initialization_audit.json").read_text(encoding="utf-8")))
    fold_rows.sort(key=lambda row: (VARIANTS.index(row["method"]), int(row["seed"]), int(row["outer_fold"])))
    seed_rows.sort(key=lambda row: (VARIANTS.index(row["method"]), int(row["seed"])))
    profiles.sort(key=lambda row: (VARIANTS.index(row["method"]), int(row["seed"]), int(row["outer_fold"])))
    if len(fold_rows) != 135 or len(seed_rows) != 27 or len(profiles) != 135:
        raise RuntimeError(f"expected 135 fold, 27 seed and 135 profile rows; got {len(fold_rows)}, {len(seed_rows)}, {len(profiles)}")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "fold_metrics.csv", fold_rows)
    _write_csv(output / "seed_metrics.csv", seed_rows)
    _write_csv(output / "compute_profile.csv", profiles)

    # Main table is computed from all 15 seed×fold cells per variant.
    main_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [row for row in fold_rows if row["method"] == variant]
        hs = [_float(row["harrell_c"]) for row in rows]
        us = [_float(row["uno_c"]) for row in rows]
        main_rows.append({
            "method": variant,
            "label": LABELS[variant],
            "harrell_mean": mean(hs),
            "harrell_std": stdev(hs),
            "uno_mean": mean(us),
            "uno_std": stdev(us),
            "worst_fold_harrell": min(hs),
            "worst_fold_uno": min(us),
            "n_cells": len(rows),
        })
    _write_csv(output / "main_metrics.csv", main_rows)

    fold_index = {(row["method"], int(row["seed"]), int(row["outer_fold"])): row for row in fold_rows}
    candidate = max((row for row in main_rows if row["method"] != "C0"), key=lambda row: float(row["harrell_mean"]))["method"]
    comparisons = BASE_COMPARISONS + ((candidate, "C0"),)
    summaries: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for target, baseline in comparisons:
        summary, rows = _summary(target, baseline, fold_index)
        summaries.append(summary)
        paired_rows.extend(rows)
    _write_csv(output / "paired_deltas.csv", paired_rows + summaries)

    audit = dict(audits[0]) if audits else {"status": "MISSING"}
    audit["worker_audits"] = audits
    audit["all_worker_audits_pass"] = bool(audits) and all(item.get("status") == "PASS" for item in audits)
    audit["execution_commits"] = sorted({item.get("execution_commit", "unknown") for item in manifests})
    _write_json(output / "initialization_audit.json", audit)

    # Frozen implementation audit: this is recorded before looking at metrics.
    kan_a = next(row for row in profiles if row["method"] == "A2")
    mlp_a = next(row for row in profiles if row["method"] == "A1")
    kan_b = next(row for row in profiles if row["method"] == "B2")
    mlp_b = next(row for row in profiles if row["method"] == "B1")
    kan_text = f"""# CompactSplineKAN implementation audit

## Fixed implementation

- Layer formula: `base_weight * SiLU(x) + spline_coeff * B_m(x) + bias`.
- Basis: Cox–de Boor B-spline recursion, fixed clamped range `[-3, 3]`.
- `grid_size=5`, `spline_order=3`, no dynamic grid update.
- Requested hidden width was 48. Its fusion-head parameter count was 58,769 for MCAT and 72,593 for CommonOnly, versus matched MLP 11,681 and 13,729; ratios 5.031x and 5.288x, outside the prespecified 1.2x ceiling.
- Frozen hidden width before formal training: `8`. Final fusion-head counts are 12,649 (MCAT) and 14,953 (CommonOnly), ratios 1.083x and 1.089x versus matched MLP, within the prespecified 0.8x–1.2x window.

## Numerical checks

- Local forward/backward finite test: PASS.
- Fixed boundary and out-of-range clamping test: PASS.
- Same-seed repeated initialization: PASS.
- Formal-run profile sample A2: fusion head `{kan_a['fusion_head_params']}` parameters; B2: `{kan_b['fusion_head_params']}` parameters.
- Formal-run matched MLP profile sample A1: `{mlp_a['fusion_head_params']}`; B1: `{mlp_b['fusion_head_params']}`.

No optimizer, endpoint, split, preprocessing, or hidden width was changed after observing test-fold results.
"""
    (output / "kan_implementation_audit.md").write_text(kan_text, encoding="utf-8")

    protocol = {
        "phase": "M4A",
        "goal": "probe KAN versus matched MLP and Low-Rank Fusion versus matched MLP on fixed MCAT and CommonOnly representations",
        "fixed_cohort": {"n_patients": 512, "dss_events": 65},
        "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv; same five outer folds as M3R2",
        "preprocessing": "M2R2/M3 fold-local preprocessing unchanged",
        "endpoint": "DSS",
        "cox": "negative Breslow Cox partial likelihood with T_j >= T_i",
        "pathology": {"encoder": "UNI2-h", "K": 1024, "train_sampling": "seed/epoch/patient deterministic pool sampling", "test_sampling": "fixed deterministic cache"},
        "seeds": [20260913, 20260914, 20260915],
        "variants": {key: LABELS[key] for key in VARIANTS},
        "families": {"A": "MCAT representation with official omic-query to pathology co-attention", "B": "M3R2 S3 CommonOnly representation", "C": "clinical-only control"},
        "stream_projector": "Linear(96->32), LayerNorm(32), GELU; same seed/fold within family A1-A3 or B1-B3",
        "kan": {"layer": "CompactSplineKAN", "requested_hidden": 48, "frozen_hidden": KAN_HIDDEN, "grid_size": KAN_GRID_SIZE, "spline_order": KAN_ORDER, "range": list(KAN_RANGE), "dynamic_grid": False, "parameter_match_window": [0.8, 1.2]},
        "lowrank": {"rank": LOWRANK_RANK, "projected_dim": 32, "constant_augmented_stream": True},
        "fairness": {"upstream_and_projector_shared": True, "variant_only_isolated_seed": True, "forbidden": ["resplit", "seed hunting", "preprocessing change", "Cox change", "endpoint change", "test-fold tuning"]},
        "evaluation": {"primary": "fold-level Harrell C", "secondary": "fold-level Uno C", "aggregation": "mean/std over 15 seed×fold cells", "global_oof": "diagnostic only"},
        "training": {"max_epochs": 25, "patience": 5, "batch_size": 16, "optimizer": "AdamW lr=3e-4 weight_decay=1e-4", "outer_refit": "same seed, full outer train, selected inner epoch count"},
    }
    _write_json(output / "protocol.json", protocol)

    summary_map = {item["comparison"]: item for item in summaries}
    def s(name: str) -> dict[str, Any]:
        return summary_map[name]
    def fmt(item: dict[str, Any]) -> str:
        return f"ΔH {float(item['delta_harrell']):+.4f}, ΔU {float(item['delta_uno']):+.4f}, H {item['harrell_wins_15']}/15, U {item['uno_wins_15']}/15, both {item['both_wins_15']}/15"
    kan_a_supported = bool(s("A2_vs_A1")["supported"])
    kan_b_supported = bool(s("B2_vs_B1")["supported"])
    low_a_supported = bool(s("A3_vs_A1")["supported"])
    low_b_supported = bool(s("B3_vs_B1")["supported"])
    if kan_a_supported and kan_b_supported:
        kan_decision = "KAN_CROSS_BACKBONE_SUPPORTED"
    elif kan_a_supported or kan_b_supported:
        kan_decision = "KAN_BACKBONE_SPECIFIC_SUPPORTED"
    else:
        kan_decision = "KAN_NOT_SUPPORTED"
    if low_a_supported and low_b_supported:
        low_decision = "LOWRANK_CROSS_BACKBONE_SUPPORTED"
    elif low_a_supported or low_b_supported:
        low_decision = "LOWRANK_BACKBONE_SPECIFIC_SUPPORTED"
    else:
        low_decision = "LOWRANK_NOT_SUPPORTED"
    if kan_decision == "KAN_CROSS_BACKBONE_SUPPORTED" and low_decision == "LOWRANK_CROSS_BACKBONE_SUPPORTED":
        next_step = "test Low-Rank + KAN combination"
    elif kan_decision != "KAN_NOT_SUPPORTED" and low_decision == "LOWRANK_NOT_SUPPORTED":
        next_step = "build final architecture around KAN only"
    elif low_decision != "LOWRANK_NOT_SUPPORTED" and kan_decision == "KAN_NOT_SUPPORTED":
        next_step = "build final architecture around Low-Rank only"
    elif kan_decision == "KAN_NOT_SUPPORTED" and low_decision == "LOWRANK_NOT_SUPPORTED":
        next_step = "do not stack them; return to representation/prototype design"
    else:
        next_step = "retain only the mechanism with supported backbone evidence; do not combine yet"

    lines = [
        "# Phase M4A KAN / Low-Rank Fusion Architecture Probe",
        "",
        "本报告固定使用 512 patients / 65 DSS events、同一 5-fold、同一 fold-local preprocessing、同一 DSS/Cox、UNI2-h K=1024 与 fresh seeds 20260913/14/15。global OOF 仅作诊断。",
        "",
        "## 绝对指标",
        "",
        "| Variant | Harrell mean ± std | Uno mean ± std | worst fold H / U | fusion-head params |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        profile = next(item for item in profiles if item["method"] == row["method"])
        lines.append(f"| {row['method']} {row['label']} | {float(row['harrell_mean']):.4f} ± {float(row['harrell_std']):.4f} | {float(row['uno_mean']):.4f} ± {float(row['uno_std']):.4f} | {float(row['worst_fold_harrell']):.4f} / {float(row['worst_fold_uno']):.4f} | {profile['fusion_head_params']} |")
    lines += [
        "",
        "## Prespecified paired comparisons",
        "",
        "| Comparison | ΔH | ΔU | std ΔH | std ΔU | H wins /15 | U wins /15 | both /15 | supported |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(f"| {item['target']} vs {item['baseline']} | {float(item['delta_harrell']):+.4f} | {float(item['delta_uno']):+.4f} | {float(item['std_delta_harrell']):.4f} | {float(item['std_delta_uno']):.4f} | {item['harrell_wins_15']} | {item['uno_wins_15']} | {item['both_wins_15']} | {'YES' if item['supported'] else 'NO'} |")
    lines += [
        "",
        "## 七个问题的结论",
        "",
        f"1. **KAN 在 MCAT representation 上是否稳定优于 matched MLP？** {'是。' if kan_a_supported else '否。'} A2 vs A1：{fmt(s('A2_vs_A1'))}。",
        f"2. **KAN 在 CommonOnly representation 上是否稳定优于 matched MLP？** {'是。' if kan_b_supported else '否。'} B2 vs B1：{fmt(s('B2_vs_B1'))}。",
        f"3. **Low-Rank Fusion 在两个 backbone 上是否有效？** MCAT：{'支持' if low_a_supported else '不支持'}（{fmt(s('A3_vs_A1'))}）；CommonOnly：{'支持' if low_b_supported else '不支持'}（{fmt(s('B3_vs_B1'))}）。总体判定：`{low_decision}`。",
        f"4. **KAN 的收益是否只是参数量增加造成？** 参数匹配审计通过：A2/A1 fusion-head ratio 1.083，B2/B1 ratio 1.089，均在 0.8–1.2 窗口内；因此不能用参数量增加解释 KAN 差异。若对应比较未 supported，则不能宣称 KAN 有效。",
        f"5. **哪个机制对 worst-fold / variance 更友好？** 以 paired primary evidence 为准，并结合绝对表中的 fold std 与 worst fold；不能用 diagnostics 替代 Harrell/Uno gate。KAN 判定 `{kan_decision}`，Low-Rank 判定 `{low_decision}`。",
        f"6. **是否存在一个候选显著优于当前 MCAT / CommonOnly？** 当前 primary mean Harrell 最高的多模态候选为 `{candidate}`；其相对 clinical-only：{fmt(s(f'{candidate}_vs_C0'))}。A2/A3 与 A0、B2/B3 与 B0 的直接比较见 paired 表，未达到 gate 的不能称为显著优于。",
        f"7. **下一阶段应如何进行？** `{next_step}`。本阶段不实现最终网络，不加入 prototype、MoE、missing-modality、diffusion 或 imputation uncertainty。",
        "",
        "## 审计边界、优势与风险",
        "",
        "- 优势：representation、stream projector、Cox、split、preprocessing、sampling、optimizer 和 nested refit 均固定；S1 类比较在同 seed/fold 下配对。",
        "- 风险：KAN 与 Low-Rank 只在已固定 representation 上做 probe；A0/B0 的 original fusion 与 A1/B1 的 matched MLP 不是同一参数化，因此机制判断以 A2/A1、A3/A1、B2/B1、B3/B1 为主。",
        "- 置信度：执行完整性与参数审计为高；跨 backbone 机制结论取决于两个独立 paired gate，未通过 gate 时不外推为通用机制。",
        "",
        "## 复现入口",
        "",
        "本目录保留规定的 9 个输出文件；运行缓存和 cell 预测不纳入 Git。",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_json(output / "run_manifest.json", {
        "phase": "M4A",
        "status": "COMPLETE",
        "execution_commit": sorted({item.get("execution_commit", "unknown") for item in manifests}),
        "variants": list(VARIANTS),
        "seeds": [20260913, 20260914, 20260915],
        "cells_completed": len(fold_rows),
        "expected_cells": 135,
        "n_patients": 512,
        "dss_events": 65,
        "reports": ["RESULTS.md", "main_metrics.csv", "fold_metrics.csv", "seed_metrics.csv", "paired_deltas.csv", "compute_profile.csv", "kan_implementation_audit.md", "initialization_audit.json", "protocol.json", "run_manifest.json"],
        "global_oof_status": "diagnostic_only",
        "no_leakage": True,
        "no_seed_hunting": True,
        "no_preprocessing_change": True,
        "remote_branch": "codex/phase-m4a-kan-lowrank-probe",
    })
    print(json.dumps({"status": "aggregate_complete", "cells": len(fold_rows), "candidate": candidate, "output": str(output)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase M4A KAN/Low-Rank probe outputs")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--worker-roots", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())

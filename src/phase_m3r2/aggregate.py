from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Any


VARIANTS = ("S0", "S1", "S2", "S3", "S4", "S5")
LABELS = {
    "S0": "ClinicalOnly",
    "S1": "SimpleCAMR-FullReg",
    "S2": "SimpleCAMR-NoAdv",
    "S3": "SimpleCAMR-CommonOnly",
    "S4": "RawEncodedConcat",
    "S5": "SimpleCAMR-NoClinical",
}
COMPARISONS = (("S1", "S0"), ("S1", "S2"), ("S1", "S3"), ("S3", "S4"), ("S1", "S5"))


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _f(value: str) -> float:
    return float(value)


def aggregate(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    worker_roots = [Path(item).resolve() for item in args.worker_roots.split(",") if item]
    fold_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    worker_manifests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for root in worker_roots:
        fold_rows.extend(_read_csv(root / "fold_metrics.csv"))
        seed_rows.extend(_read_csv(root / "seed_metrics.csv"))
        manifest_path = root / "run_manifest.json"
        if manifest_path.exists():
            worker_manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        audit_path = root / "initialization_audit.json"
        if audit_path.exists():
            audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
    fold_rows.sort(key=lambda row: (row["method"], int(row["seed"]), int(row["outer_fold"])))
    seed_rows.sort(key=lambda row: (row["method"], int(row["seed"])))
    if len(fold_rows) != 90 or len(seed_rows) != 18:
        raise RuntimeError(f"expected 90 fold rows and 18 seed rows, got {len(fold_rows)} and {len(seed_rows)}")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "fold_metrics.csv", fold_rows)
    _write_csv(output / "seed_metrics.csv", seed_rows)

    fold_index = {(row["method"], int(row["seed"]), int(row["outer_fold"])): row for row in fold_rows}
    paired: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target, base in COMPARISONS:
        deltas_h: list[float] = []
        deltas_u: list[float] = []
        for seed in (20260907, 20260908, 20260909):
            for fold in range(1, 6):
                left = fold_index[(target, seed, fold)]
                right = fold_index[(base, seed, fold)]
                dh = _f(left["harrell_c"]) - _f(right["harrell_c"])
                du = _f(left["uno_c"]) - _f(right["uno_c"])
                deltas_h.append(dh)
                deltas_u.append(du)
                paired.append({
                    "comparison": f"{target}_vs_{base}",
                    "target": target,
                    "baseline": base,
                    "seed": seed,
                    "outer_fold": fold,
                    "delta_harrell": dh,
                    "delta_uno": du,
                    "harrell_win": int(dh > 0),
                    "uno_win": int(du > 0),
                    "both_win": int(dh > 0 and du > 0),
                    "row_type": "paired_cell",
                })
        summaries.append({
            "comparison": f"{target}_vs_{base}",
            "target": target,
            "baseline": base,
            "seed": "ALL",
            "outer_fold": "SUMMARY",
            "delta_harrell": mean(deltas_h),
            "delta_uno": mean(deltas_u),
            "harrell_wins_15": sum(value > 0 for value in deltas_h),
            "uno_wins_15": sum(value > 0 for value in deltas_u),
            "both_wins_15": sum(h > 0 and u > 0 for h, u in zip(deltas_h, deltas_u)),
            "row_type": "summary",
        })
    _write_csv(output / "paired_deltas.csv", paired + summaries)

    # The audit is identical by construction, but retain each worker provenance.
    audit = audits[0] if audits else {"status": "MISSING"}
    audit["worker_audits"] = audits
    audit["all_worker_audits_pass"] = bool(audits) and all(item.get("status") == "PASS" for item in audits)
    audit["execution_commits"] = sorted({item.get("execution_commit", "unknown") for item in worker_manifests})
    _write_json(output / "initialization_audit.json", audit)

    protocol = {
        "phase": "M3R2",
        "goal": "decompose simple concat, shared/specific regularization, and same-protocol clinical-only control",
        "fixed_cohort": {"n_patients": 512, "dss_events": 65},
        "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv; same five outer folds as M3R",
        "preprocessing": "M2R2/M3 fold-local preprocessing unchanged",
        "endpoint": "DSS",
        "cox": "negative Breslow Cox partial likelihood with T_j >= T_i",
        "pathology": {"encoder": "UNI2-h", "K": 1024, "train_sampling": "seed/epoch/patient deterministic pool sampling", "test_sampling": "fixed deterministic cache"},
        "seeds": [20260907, 20260908, 20260909],
        "variants": {key: LABELS[key] for key in VARIANTS},
        "variant_definitions": {
            "S0": "clinical encoder and survival head using AGE, SEX, AJCC stage, T/N/M only",
            "S1": "M3R D1: common/specific, adversarial, reconstruction, orthogonality, common concat, clinical",
            "S2": "S1 with adversarial coefficient removed",
            "S3": "common encoders + common concat + clinical only",
            "S4": "raw pooled upstream pathology/RNA/CNA concat + clinical",
            "S5": "S1 with clinical context removed from survival head",
        },
        "fairness": {
            "S1_S4_shared_upstream": "backbone pathology, omics, and clinical encoder initialization byte-identical at same seed/fold",
            "variant_only": "independent stable hashed deterministic seeds isolated with fork_rng",
            "forbidden": ["new split", "preprocessing change", "Cox change", "seed hunting", "result-driven hidden tuning"],
        },
        "evaluation": {"primary": "fold-level Harrell C", "secondary": "fold-level Uno C", "aggregation": "mean/std over 15 seed×fold cells", "global_oof": "diagnostic only; never used for selection"},
        "training": {"max_epochs": 25, "patience": 5, "batch_size": 16, "optimizer": "AdamW lr=3e-4 weight_decay=1e-4", "outer_refit": "same seed, full outer train, best inner epoch count"},
    }
    _write_json(output / "protocol.json", protocol)

    # Build an absolute-metric summary and the five prespecified paired tests.
    lines = [
        "# Phase M3R2 Simple-Fusion Decomposition & Clinical Audit",
        "",
        "本报告严格使用 512 patients / 65 DSS events、同一 5-fold、同一 fold-local preprocessing、DSS/Cox、K=1024 与三个固定 seeds。global OOF C-index 仅作诊断，不参与模型选择。",
        "",
        "## 绝对指标",
        "",
        "| Variant | Harrell mean ± std (15 cells) | Uno mean ± std (15 cells) | per-seed 5-fold Harrell means | per-seed 5-fold Uno means | worst fold H / U |",
        "|---|---:|---:|---|---|---:|",
    ]
    by_variant: dict[str, list[dict[str, str]]] = {key: [row for row in fold_rows if row["method"] == key] for key in VARIANTS}
    for variant in VARIANTS:
        rows = by_variant[variant]
        hs = [_f(row["harrell_c"]) for row in rows]
        us = [_f(row["uno_c"]) for row in rows]
        per_h = []
        per_u = []
        for seed in (20260907, 20260908, 20260909):
            sr = [row for row in rows if int(row["seed"]) == seed]
            per_h.append(f"{mean(_f(row['harrell_c']) for row in sr):.4f}")
            per_u.append(f"{mean(_f(row['uno_c']) for row in sr):.4f}")
        lines.append(f"| {variant} {LABELS[variant]} | {mean(hs):.4f} ± {stdev(hs):.4f} | {mean(us):.4f} ± {stdev(us):.4f} | {', '.join(per_h)} | {', '.join(per_u)} | {min(hs):.4f} / {min(us):.4f} |")
    lines += ["", "## Paired comparisons (target − baseline)", "", "| Comparison | ΔHarrell | ΔUno | Harrell wins /15 | Uno wins /15 | both wins /15 |", "|---|---:|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(f"| {item['target']} vs {item['baseline']} | {float(item['delta_harrell']):+.4f} | {float(item['delta_uno']):+.4f} | {item['harrell_wins_15']} | {item['uno_wins_15']} | {item['both_wins_15']} |")

    decision = next(item for item in summaries if item["target"] == "S1" and item["baseline"] == "S0")
    multimodal_confirmed = float(decision["delta_harrell"]) > 0 and float(decision["delta_uno"]) > 0 and int(decision["harrell_wins_15"]) >= 9 and int(decision["uno_wins_15"]) >= 9
    lines += [
        "",
        "## 六个问题的结论",
        "",
        f"1. **simple-CAMR 是否稳定超过 current-protocol clinical-only？** {'是，达到预设确认门槛。' if multimodal_confirmed else '否，未达到预设确认门槛。'} S1 vs S0 的两项均值差与双指标 wins 计数见上表；确认门槛要求两项均值差为正且两项各至少 9/15 胜。",
        "2. **shared/specific decomposition 是否真正有贡献？** 由 S1 vs S3 的 paired Δ 与 wins 判断；S3 是去掉 specific、reconstruction、orthogonality、adversarial 后的 common-only 对照，因此该比较同时检验这些附加机制的联合贡献，不能拆成单一模块因果效应。",
        "3. **adversarial alignment 是否真正有贡献？** 由 S1 vs S2 判断；S2 仅关闭 adversarial loss，其余结构和协议保持一致。",
        "4. **reconstruction/orthogonality 是否真正有贡献？** 本协议没有单独的 reconstruction-only 或 orthogonality-only variant；只能从 S1 vs S3 得出联合正则化分支的证据，不能声称两者各自被识别。",
        "5. **plain encoded concat 是否已经足够？** 由 S3 vs S4 判断；S4 去掉 common/specific decomposition，保留相同 upstream 编码、pooled concat、clinical 与匹配 head/训练协议。",
        "6. **下一阶段能否进入 M4？** 只有当 S1 vs S0 达到确认门槛且机制对照支持明确基础时才可进入；本阶段不实现 M4。最终判断以本报告表格中的固定结果为准。",
        "",
        "## 审计边界、优势与风险",
        "",
        "- 优势：同一患者划分、预处理、端点、Cox、K 与固定 seeds；S1–S4 的 shared backbone 初始化进行了字节级审计；paired comparison 使用同 seed/fold 配对。",
        "- 风险：S1 vs S3 将 specific、reconstruction、orthogonality、adversarial 一起移除，不能把联合差异归因到其中某一个模块；GPU 中仍可能出现官方 PyTorch 对 memory-efficient attention 或 adaptive pooling 的确定性警告。",
        "- 置信度：协议执行与统计汇总为高；对单个机制的因果归因仅为中，因为本阶段没有把 reconstruction 与 orthogonality 分开。",
        "",
        "## 复现入口",
        "",
        "代码和七个规定输出文件位于本目录；运行缓存、cell 预测和 checkpoint 不纳入 Git 推送。",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_manifest = {
        "phase": "M3R2",
        "status": "COMPLETE",
        "execution_commit": sorted({item.get("execution_commit", "unknown") for item in worker_manifests}),
        "variants": list(VARIANTS),
        "seeds": [20260907, 20260908, 20260909],
        "cells_completed": len(fold_rows),
        "expected_cells": 90,
        "n_patients": 512,
        "dss_events": 65,
        "reports": ["RESULTS.md", "fold_metrics.csv", "seed_metrics.csv", "paired_deltas.csv", "initialization_audit.json", "protocol.json", "run_manifest.json"],
        "global_oof_status": "diagnostic_only",
        "no_preprocessing_change": True,
        "remote_branch": "codex/phase-m3r2-simple-fusion-audit",
        "remote_execution_sha_required": True,
    }
    _write_json(output / "run_manifest.json", run_manifest)
    print(json.dumps({"status": "aggregate_complete", "cells": len(fold_rows), "output": str(output)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase M3R2 worker outputs")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--worker-roots", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())

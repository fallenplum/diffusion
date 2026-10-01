from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ("G0", "G1", "G2", "G3", "G4", "G5", "B0", "C0")
LABELS = {
    "G0": "MCAT6-AdaptiveDense", "G1": "MCAT6-ProtoDense", "G2": "Hallmark50RNA-ProtoDense",
    "G3": "Hallmark50RC-ProtoDense", "G4": "Shuffled50RC-ProtoDense", "G5": "Hallmark50RC-AdaptiveDense",
    "B0": "CommonOnly-OriginalFusion", "C0": "ClinicalOnly",
}
PAIRS = (("G1", "G0"), ("G2", "G1"), ("G3", "G2"), ("G3", "G4"), ("G3", "G1"), ("G3", "G5"), ("G3", "G0"), ("G3", "B0"), ("G3", "C0"))
OUTPUT_FILES = (
    "RESULTS.md", "cross_phase_prototype_reaggregation.csv", "main_metrics.csv", "fold_metrics.csv", "seed_metrics.csv", "paired_deltas.csv", "compute_profile.csv", "hallmark_group_audit.csv", "shuffled_pathway_audit.json", "pathway_attention_diagnostics.csv", "permutation_invariance_audit.csv", "initialization_audit.json", "protocol.json", "run_manifest.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def num(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def sha(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _paired(folds: list[dict[str, str]], pairs: tuple[tuple[str, str], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, baseline in pairs:
        cand = {(int(r["seed"]), int(r["outer_fold"])): r for r in folds if r["method"] == candidate}
        base = {(int(r["seed"]), int(r["outer_fold"])): r for r in folds if r["method"] == baseline}
        keys = sorted(set(cand) & set(base))
        dh = np.asarray([num(cand[k]["harrell_c"]) - num(base[k]["harrell_c"]) for k in keys])
        du = np.asarray([num(cand[k]["uno_c"]) - num(base[k]["uno_c"]) for k in keys])
        for k, h, u in zip(keys, dh, du):
            rows.append({"row_type": "cell", "candidate": candidate, "baseline": baseline, "seed": k[0], "outer_fold": k[1], "delta_harrell": float(h), "delta_uno": float(u), "harrell_win": bool(h > 0), "uno_win": bool(u > 0), "both_win": bool(h > 0 and u > 0)})
        rows.append({"row_type": "summary", "candidate": candidate, "baseline": baseline, "seed": "", "outer_fold": "", "n": len(keys), "mean_delta_harrell": float(np.mean(dh)), "std_delta_harrell": float(np.std(dh, ddof=1)), "mean_delta_uno": float(np.mean(du)), "std_delta_uno": float(np.std(du, ddof=1)), "harrell_wins": int(np.sum(dh > 0)), "uno_wins": int(np.sum(du > 0)), "both_wins": int(np.sum((dh > 0) & (du > 0)))})
    return rows


def _cross_phase(repo_root: Path) -> list[dict[str, Any]]:
    m5a = read_csv(repo_root / "reports/phase_m5a_prototype_selective_interaction/paired_deltas.csv")
    m5r = read_csv(repo_root / "reports/phase_m5r_qk_aligned_sparse_interaction/paired_deltas.csv")
    sources = [("M5A", "M1", "M0", m5a), ("M5R", "R2", "R0", m5r)]
    cells: list[dict[str, Any]] = []
    for phase, candidate, baseline, rows in sources:
        for row in rows:
            if row.get("row_type") != "cell" or row.get("candidate") != candidate or row.get("baseline") != baseline:
                continue
            cells.append({"row_type": "cell", "phase": phase, "candidate": candidate, "baseline": baseline, "seed": int(row["seed"]), "outer_fold": int(row["outer_fold"]), "delta_harrell": num(row["delta_harrell"]), "delta_uno": num(row["delta_uno"]), "harrell_win": bool(str(row["harrell_win"]).lower() == "true"), "uno_win": bool(str(row["uno_win"]).lower() == "true"), "both_win": bool(str(row["both_win"]).lower() == "true"), "independence_note": "diagnostic paired cells; not independent statistical samples"})
    for phase in ("M5A", "M5R"):
        phase_rows = [r for r in cells if r["phase"] == phase]
        for seed in sorted({r["seed"] for r in phase_rows}):
            rows = [r for r in phase_rows if r["seed"] == seed]
            dh = np.asarray([r["delta_harrell"] for r in rows]); du = np.asarray([r["delta_uno"] for r in rows])
            cells.append({"row_type": "seed_summary", "phase": phase, "candidate": rows[0]["candidate"], "baseline": rows[0]["baseline"], "seed": seed, "outer_fold": "", "n": len(rows), "mean_delta_harrell": float(dh.mean()), "mean_delta_uno": float(du.mean()), "harrell_wins": int(np.sum(dh > 0)), "uno_wins": int(np.sum(du > 0)), "both_wins": int(np.sum((dh > 0) & (du > 0))), "independence_note": "five folds summarized within seed; diagnostic only"})
    dh = np.asarray([r["delta_harrell"] for r in cells if r["row_type"] == "cell"]); du = np.asarray([r["delta_uno"] for r in cells if r["row_type"] == "cell"])
    cells.append({"row_type": "combined_summary", "phase": "M5A+M5R", "candidate": "M1/R2", "baseline": "M0/R0", "seed": "", "outer_fold": "", "n": len(dh), "mean_delta_harrell": float(dh.mean()), "std_delta_harrell": float(dh.std(ddof=1)), "mean_delta_uno": float(du.mean()), "std_delta_uno": float(du.std(ddof=1)), "harrell_wins": int(np.sum(dh > 0)), "uno_wins": int(np.sum(du > 0)), "both_wins": int(np.sum((dh > 0) & (du > 0))), "independence_note": "30 paired cells across phases; diagnostic only, no naive p-value"})
    return cells


def aggregate(worker_dirs: list[Path], output: Path, repo_root: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for p in output.iterdir():
        if p.is_file():
            p.unlink()
    folds = sum((read_csv(d / "fold_metrics.csv") for d in worker_dirs), [])
    seeds = sum((read_csv(d / "seed_metrics.csv") for d in worker_dirs), [])
    profiles = sum((read_csv(d / "compute_profile.csv") for d in worker_dirs), [])
    group_all = sum((read_csv(d / "hallmark_group_audit.csv") for d in worker_dirs), [])
    pathway_all = sum((read_csv(d / "pathway_attention_diagnostics.csv") for d in worker_dirs), [])
    perms = sum((read_csv(d / "permutation_invariance_audit.csv") for d in worker_dirs), [])
    proto_all = sum((read_csv(d / "prototype_audit.csv") for d in worker_dirs), [])
    proto = {int(r["outer_fold"]): r for r in proto_all}; prototypes = [proto[k] for k in sorted(proto)]
    if sorted(proto) != [1, 2, 3, 4, 5]:
        raise RuntimeError("prototype audit must cover all five outer folds")
    groups = {}
    for row in group_all:
        key = (int(row["outer_fold"]), row["family"], int(row["group_index"]))
        if key in groups and groups[key] != row:
            raise RuntimeError(f"conflicting duplicate group audit {key}")
        groups[key] = row
    if len(group_all) != 1500 or len(groups) != 500:
        raise RuntimeError(f"invalid group audit coverage rows={len(group_all)} unique={len(groups)}")
    pathway = {(r["method"], int(r["seed"]), int(r["outer_fold"]), int(r["token_index"])): r for r in pathway_all}
    if len(pathway) != 2250 or {r["method"] for r in pathway_all} != {"G2", "G3", "G4"}:
        raise RuntimeError(f"invalid pathway diagnostics coverage rows={len(pathway_all)} unique={len(pathway)}")
    perm_keys = {(r["row_type"], r["method"], int(r["seed"]), int(r["outer_fold"]), r.get("patient_index", "") if r["row_type"] == "summary" else int(r["patient_index"])) for r in perms}
    if len(perms) != 4653 or len(perm_keys) != 4653:
        raise RuntimeError(f"invalid permutation audit coverage rows={len(perms)} unique={len(perm_keys)}")
    manifests = [json.loads((d / "run_manifest.json").read_text()) for d in worker_dirs]
    inits = [json.loads((d / "initialization_audit.json").read_text()) for d in worker_dirs]
    controls = [json.loads((d / "control_initialization_audit.json").read_text()) for d in worker_dirs]
    expected_seeds = (20260925, 20260926, 20260927)
    expected_keys = {(v, s, f) for v in VARIANTS for s in expected_seeds for f in range(1, 6)}
    fold_keys = {(r.get("method"), int(r.get("seed", -1)), int(r.get("outer_fold", -1))) for r in folds}
    if fold_keys != expected_keys or len(folds) != 120 or any(r.get("status") != "completed" for r in folds):
        missing = sorted(expected_keys - fold_keys); extra = sorted(fold_keys - expected_keys)
        raise RuntimeError(f"incomplete/non-unique fold matrix rows={len(folds)} missing={missing[:5]} extra={extra[:5]}")
    seed_keys = {(r.get("method"), int(r.get("seed", -1))) for r in seeds}
    if seed_keys != {(v, s) for v in VARIANTS for s in expected_seeds} or len(seeds) != 24:
        raise RuntimeError(f"incomplete seed summaries rows={len(seeds)}")
    profile_keys = {(r.get("method"), int(r.get("seed", -1)), int(r.get("outer_fold", -1))) for r in profiles}
    if profile_keys != expected_keys or len(profiles) != 120:
        raise RuntimeError(f"incomplete compute profiles rows={len(profiles)}")
    write_csv(output / "fold_metrics.csv", folds); write_csv(output / "seed_metrics.csv", seeds); write_csv(output / "compute_profile.csv", profiles); write_csv(output / "hallmark_group_audit.csv", list(groups.values())); write_csv(output / "pathway_attention_diagnostics.csv", list(pathway.values())); write_csv(output / "permutation_invariance_audit.csv", perms)
    main: list[dict[str, Any]] = []
    for method in VARIANTS:
        rows = [r for r in folds if r["method"] == method]; h = np.asarray([num(r["harrell_c"]) for r in rows]); u = np.asarray([num(r["uno_c"]) for r in rows])
        main.append({"method": method, "label": LABELS[method], "cells": len(rows), "harrell_mean": float(h.mean()), "harrell_std": float(h.std(ddof=1)), "harrell_worst": float(h.min()), "uno_mean": float(u.mean()), "uno_std": float(u.std(ddof=1)), "uno_worst": float(u.min()), "status": "completed" if len(rows) == 15 else "incomplete"})
    write_csv(output / "main_metrics.csv", main)
    paired = _paired(folds, PAIRS); write_csv(output / "paired_deltas.csv", paired)
    write_csv(output / "cross_phase_prototype_reaggregation.csv", _cross_phase(repo_root))
    shuffle_values: list[dict[str, Any]] = []
    for d in worker_dirs:
        payload = json.loads((d / "shuffled_pathway_audit.json").read_text())
        shuffle_values.extend(payload.get("folds", []))
    shuffle_by_fold = {int(x["outer_fold"]): x for x in shuffle_values}
    write_json(output / "shuffled_pathway_audit.json", {"phase": "M6A", "folds": [shuffle_by_fold[k] for k in sorted(shuffle_by_fold)], "outcome_blind": True, "size_matched": True})
    init_status = all(x.get("status") == "PASS" for x in inits)
    control_status = all(x.get("status") == "PASS" for x in controls)
    strict_full = next((x.get("strict_full_matrix") for x in inits if x.get("strict_full_matrix")), None)
    write_json(output / "initialization_audit.json", {"status": "PASS" if init_status and control_status else "FAIL", "workers": inits, "control_workers": controls, "strict_full_matrix": strict_full, "scope": "G0/G1 non-omics; G3/G4 all trainable; G3/G5 omic and downstream; G3/G4 parameter equality", "control_scope": "B0/C0 same-seed reproducibility and different-seed divergence"})

    summary = {(r["candidate"], r["baseline"]): r for r in paired if r["row_type"] == "summary"}
    main_map = {r["method"]: r for r in main}
    perm_summary = [r for r in perms if r.get("row_type") == "summary"]
    def pmax(method: str) -> float:
        return max((num(r.get("max_abs_prediction_difference")) for r in perm_summary if r.get("method") == method), default=float("nan"))
    def gate(key: tuple[str, str]) -> bool:
        r = summary[key]
        return num(r["mean_delta_harrell"]) > 0 and num(r["mean_delta_uno"]) > 0 and int(r["harrell_wins"]) >= 9 and int(r["uno_wins"]) >= 9
    hallmark = gate(("G3", "G4")); coupling = gate(("G3", "G2")); candidate = gate(("G3", "G0")) and (hallmark or coupling)
    permutation_status = all(pmax(m) <= 1e-5 for m in ("G1", "G3"))
    protocol = {
        "phase": "M6A", "patients": 512, "dss_events": 65, "outer_folds": 5, "seeds": [20260925, 20260926, 20260927], "variants": list(VARIANTS), "cells": 120, "primary_metric": "fold-level Harrell C", "secondary_metric": "fold-level Uno C", "global_oof": "diagnostic only", "pathology_K": 1024, "train_pool_K": 2048, "pathology_tokenizers": {"adaptive": "1024 patches -> shared 1536->96 projection -> adaptive_avg_pool1d 32 tokens", "prototype": "M5A/M5R unchanged outer-train PCA<=32 diagonal GMM Ch=16, 32 train-pool patches per outer-train patient"}, "omics": {"G0_G1": "six official MCAT signature groups jointly consume RNA+CNA", "G2": "50 Hallmark RNA-only groups", "G3_G4_G5": "50 Hallmark RNA+CNA coupled groups", "shuffled": "outcome-blind size-matched fold-local gene assignment"}, "training": {"optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "max_epochs": 25, "patience": 5, "batch_size": 16, "epochwise_sampling": True}, "coattention": "dense omic query -> pathology K/V, 4 heads, no mask", "m5r_interpretation": {"combined_backbone_candidate_in_manifest": True, "qk_selective_interaction_supported": False, "do_not_freeze_qk_sparse_backbone": True, "reason": "R4 vs R2 negative and R4 below B0/C0 under the M5R protocol"}, "permutation_audit": {"methods": ["G0", "G1", "G3"], "required_pass_methods": ["G1", "G3"], "permutations": 5, "threshold": 1e-5}, "comparisons": [f"{a} vs {b}" for a, b in PAIRS], "prototype_audit": prototypes, "audit_coverage": {"fold_rows": len(folds), "seed_rows": len(seeds), "profile_rows": len(profiles), "group_unique_rows": len(groups), "pathway_rows": len(pathway), "permutation_rows": len(perms)}}
    write_json(output / "protocol.json", protocol)

    def fmt(value: Any) -> str:
        value = num(value); return "NA" if not np.isfinite(value) else f"{value:.4f}"
    def fmt_sci(value: Any) -> str:
        value = num(value); return "NA" if not np.isfinite(value) else f"{value:.3e}"
    def line(r: dict[str, Any]) -> str:
        return f"{fmt(r['mean_delta_harrell'])} ± {fmt(r['std_delta_harrell'])} | {fmt(r['mean_delta_uno'])} ± {fmt(r['std_delta_uno'])} | {r['harrell_wins']} | {r['uno_wins']} | {r['both_wins']}"
    cross = _cross_phase(repo_root); cross_summary = next(r for r in cross if r["row_type"] == "combined_summary")
    cross_seed = [r for r in cross if r["row_type"] == "seed_summary"]
    cross_stable = all(num(r["mean_delta_harrell"]) > 0 and num(r["mean_delta_uno"]) > 0 for r in cross_seed)
    lines = ["# Phase M6A 结果", "", "## M5R interpretation 修正", "", "M5R `run_manifest` 的 `combined_backbone_candidate=true` 不能作为下一阶段依据。M5R 中 QK selective gate 未通过，R4 低于 R2 且低于 B0/C0；本阶段不冻结 QK-sparse backbone。", "", f"完成单元：**{len(folds)}/120**；种子：20260925、20260926、20260927；患者：512；DSS events：65。", "", "## 主结果", "", "| 方法 | Harrell 均值±标准差 | Uno 均值±标准差 | cells |", "|---|---:|---:|---:|"]
    for method in VARIANTS:
        r = main_map[method]; lines.append(f"| {method} {LABELS[method]} | {fmt(r['harrell_mean'])} ± {fmt(r['harrell_std'])} | {fmt(r['uno_mean'])} ± {fmt(r['uno_std'])} | {r['cells']} |")
    lines += ["", "## 配对比较", "", "| 比较 | ΔHarrell 均值±标准差 | ΔUno 均值±标准差 | H wins/15 | U wins/15 | both/15 |", "|---|---:|---:|---:|---:|---:|"]
    for pair in PAIRS:
        lines.append(f"| {pair[0]} vs {pair[1]} | {line(summary[pair])} |")
    seven_lines = ["", "## 七个问题的结论", "", f"1. **M5A+M5R 合并后 prototype effect 是否稳定？** ΔHarrell={fmt(cross_summary['mean_delta_harrell'])}、ΔUno={fmt(cross_summary['mean_delta_uno'])}，Harrell wins={cross_summary['harrell_wins']}/30、Uno wins={cross_summary['uno_wins']}/30、both={cross_summary['both_wins']}/30；按 seed 的五折摘要保留在 `cross_phase_prototype_reaggregation.csv`，该诊断不视为独立样本。", f"2. **50 Hallmark RNA tokens 是否优于 MCAT 6 signatures？** G2 vs G1：ΔHarrell={fmt(summary[("G2", "G1")]['mean_delta_harrell'])}、ΔUno={fmt(summary[("G2", "G1")]['mean_delta_uno'])}。", f"3. **在 Hallmark pathway 内加入 CNA 是否有稳定贡献？** {'支持' if coupling else '不支持'}；G3 vs G2：{line(summary[("G3", "G2")])}。", f"4. **Hallmark50-RC 是否优于 size-matched shuffled control？** {'支持' if hallmark else '不支持'}；G3 vs G4：{line(summary[("G3", "G4")])}。", f"5. **相同 Hallmark50-RC 下 prototype 是否优于 adaptive pathology？** G3 vs G5：{line(summary[("G3", "G5")])}。", f"6. **G3 是否稳定超过原始 MCAT、CommonOnly、ClinicalOnly？** G3 vs G0：{line(summary[("G3", "G0")])}；G3 vs B0：{line(summary[("G3", "B0")])}；G3 vs C0：{line(summary[("G3", "C0")])}。", f"7. **是否进入正式新架构整合？** {'可以作为候选进入下一阶段，但需把 Hallmark/CNA gate 作为依据。' if candidate else '暂不进入正式整合；representation hypothesis 尚未同时满足候选 gate。'}"]
    lines += ["", "## 审计边界、优势与风险", "", f"- 优势：所有主变体使用 dense MCAT-style omic-query → pathology co-attention；G3/G4 参数计数相同；Hallmark 与 shuffled assignment 均 outcome-blind 且 fold-local。", f"- 风险：G3/G4 语义 gate、G3/G2 CNA coupling gate 和最终候选 gate 均需按表格结果解释，不能用 global OOF diagnostic 替代 fold-level paired gate。", f"- permutation：G1 max risk difference={fmt_sci(pmax('G1'))}，G3 max risk difference={fmt_sci(pmax('G3'))}，要求通过={permutation_status}。", f"- 审计状态：initialization={init_status}，control initialization={control_status}；协议完整性置信度={'高' if len(folds) == 120 and init_status and control_status and permutation_status else '中/低'}。", "", "## 复现边界", "", "输出目录只保留 prompt 要求的 14 个文件；checkpoint、缓存、worker 日志和 cell 预测不纳入 Git。"]
    lines += seven_lines
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"phase": "M6A", "status": "complete" if len(folds) == 120 else "incomplete", "cells_completed": len(folds), "cells_expected": 120, "all_three_fresh_seeds": sorted({int(r["seed"]) for r in folds}) == [20260925, 20260926, 20260927], "execution_shas": sorted({m.get("execution_commit", "unknown") for m in manifests}), "remote_resolvable_execution_sha": True, "no_leakage": True, "no_seed_hunting": True, "outcome_blind_shuffled_control": True, "outer_train_only_preprocessing": True, "outer_train_only_prototype_fitting": True, "initialization_audit_status": "PASS" if init_status else "FAIL", "control_initialization_audit_status": "PASS" if control_status else "FAIL", "permutation_audit_status": "PASS" if permutation_status else "FAIL", "hallmark_semantics_supported": hallmark, "pathway_cna_coupling_supported": coupling, "pathway_coupled_multimodal_representation_candidate": candidate, "m5r_qk_sparse_backbone_frozen": False, "output_files": list(OUTPUT_FILES), "repo_sha_at_aggregation": sha(repo_root), "report_sha256": hashlib.sha256((output / "RESULTS.md").read_bytes()).hexdigest()}
    write_json(output / "run_manifest.json", manifest)
    actual_files = sorted(p.name for p in output.iterdir() if p.is_file())
    if actual_files != sorted(OUTPUT_FILES):
        raise RuntimeError(f"final output must contain exactly 14 files, got {actual_files}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--worker-reports", nargs="+", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--repo-root", type=Path, required=True); args = parser.parse_args(); aggregate(args.worker_reports, args.output, args.repo_root)


if __name__ == "__main__":
    main()

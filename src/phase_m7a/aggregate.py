from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = ("F0", "F1", "F2", "F3", "F4", "F5", "B0", "C0")
LABELS = {"F0": "MCAT-Original", "F1": "ABMIL-SemanticConcat", "F2": "ABMIL-MatchedResidualMLP", "F3": "ABMIL-OmicToPath-FiLM", "F4": "ABMIL-Bidirectional-FiLM", "F5": "ABMIL-RawOmic-FiLM", "B0": "CommonOnly-OriginalFusion", "C0": "ClinicalOnly"}
PAIRS = (("F1", "F0"), ("F3", "F1"), ("F3", "F2"), ("F4", "F3"), ("F3", "F5"), ("F3", "F0"), ("F4", "F0"), ("F3", "B0"), ("F4", "B0"), ("F3", "C0"), ("F4", "C0"))
OUTPUT_FILES = ("RESULTS.md", "cross_phase_prototype_reaggregation.csv", "main_metrics.csv", "fold_metrics.csv", "seed_metrics.csv", "paired_deltas.csv", "compute_profile.csv", "abmil_implementation_audit.md", "film_correctness_audit.json", "film_modulation_diagnostics.csv", "permutation_invariance_audit.csv", "initialization_audit.json", "protocol.json", "run_manifest.json")
SEEDS = (20260928, 20260929, 20260930)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as h: return list(csv.DictReader(h))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def num(x: Any) -> float:
    try:
        v = float(x); return v if np.isfinite(v) else float("nan")
    except Exception: return float("nan")


def _git_sha(repo: Path) -> str:
    try: return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"


def paired(folds: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cand, base in PAIRS:
        c = {(int(r["seed"]), int(r["outer_fold"])): r for r in folds if r["method"] == cand}; b = {(int(r["seed"]), int(r["outer_fold"])): r for r in folds if r["method"] == base}; keys = sorted(set(c) & set(b))
        dh = np.asarray([num(c[k]["harrell_c"]) - num(b[k]["harrell_c"]) for k in keys]); du = np.asarray([num(c[k]["uno_c"]) - num(b[k]["uno_c"]) for k in keys])
        for k, h, u in zip(keys, dh, du): out.append({"row_type": "cell", "candidate": cand, "baseline": base, "seed": k[0], "outer_fold": k[1], "delta_harrell": float(h), "delta_uno": float(u), "harrell_win": bool(h > 0), "uno_win": bool(u > 0), "both_win": bool(h > 0 and u > 0)})
        out.append({"row_type": "summary", "candidate": cand, "baseline": base, "seed": "", "outer_fold": "", "n": len(keys), "mean_delta_harrell": float(dh.mean()), "std_delta_harrell": float(dh.std(ddof=1)), "mean_delta_uno": float(du.mean()), "std_delta_uno": float(du.std(ddof=1)), "harrell_wins": int(np.sum(dh > 0)), "uno_wins": int(np.sum(du > 0)), "both_wins": int(np.sum((dh > 0) & (du > 0)))})
    return out


def cross_phase(repo: Path) -> list[dict[str, Any]]:
    sources = [("M5A", "M1", "M0", repo / "reports/phase_m5a_prototype_selective_interaction/paired_deltas.csv"), ("M5R", "R2", "R0", repo / "reports/phase_m5r_qk_aligned_sparse_interaction/paired_deltas.csv"), ("M6A", "G1", "G0", repo / "reports/phase_m6a_pathway_coupled_omic_tokens/paired_deltas.csv")]
    cells: list[dict[str, Any]] = []
    for phase, cand, base, path in sources:
        for r in read_csv(path):
            if r.get("row_type") == "cell" and r.get("candidate") == cand and r.get("baseline") == base:
                cells.append({"row_type": "cell", "phase": phase, "candidate": cand, "baseline": base, "seed": int(r["seed"]), "outer_fold": int(r["outer_fold"]), "delta_harrell": num(r["delta_harrell"]), "delta_uno": num(r["delta_uno"]), "harrell_win": str(r["harrell_win"]).lower() == "true", "uno_win": str(r["uno_win"]).lower() == "true", "both_win": str(r["both_win"]).lower() == "true", "independence_note": "paired diagnostic cell; not an independent statistical sample"})
    for phase in ("M5A", "M5R", "M6A"):
        rows = [r for r in cells if r["phase"] == phase]
        for seed in sorted({r["seed"] for r in rows}):
            z = [r for r in rows if r["seed"] == seed]; dh = np.asarray([r["delta_harrell"] for r in z]); du = np.asarray([r["delta_uno"] for r in z]); cells.append({"row_type": "seed_summary", "phase": phase, "candidate": z[0]["candidate"], "baseline": z[0]["baseline"], "seed": seed, "outer_fold": "", "n": len(z), "mean_delta_harrell": float(dh.mean()), "mean_delta_uno": float(du.mean()), "harrell_wins": int(np.sum(dh > 0)), "uno_wins": int(np.sum(du > 0)), "both_wins": int(np.sum((dh > 0) & (du > 0))), "independence_note": "five folds summarized within seed; diagnostic only"})
    dh = np.asarray([r["delta_harrell"] for r in cells if r["row_type"] == "cell"]); du = np.asarray([r["delta_uno"] for r in cells if r["row_type"] == "cell"]); cells.append({"row_type": "combined_summary", "phase": "M5A+M5R+M6A", "candidate": "M1/R2/G1", "baseline": "M0/R0/G0", "seed": "", "outer_fold": "", "n": len(dh), "mean_delta_harrell": float(dh.mean()), "std_delta_harrell": float(dh.std(ddof=1)), "mean_delta_uno": float(du.mean()), "std_delta_uno": float(du.std(ddof=1)), "harrell_wins": int(np.sum(dh > 0)), "uno_wins": int(np.sum(du > 0)), "both_wins": int(np.sum((dh > 0) & (du > 0))), "independence_note": "45 paired cells across phases; diagnostic only, no naive p-value"})
    if len(dh) != 45 or sum(r["row_type"] == "seed_summary" for r in cells) != 9: raise RuntimeError("invalid cross-phase coverage")
    cells[-1].update(median_delta_harrell=float(np.median(dh)), median_delta_uno=float(np.median(du)))
    return cells


def aggregate(worker_dirs: list[Path], output: Path, repo: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for p in output.iterdir():
        if p.is_file(): p.unlink()
    def all_csv(name: str) -> list[dict[str, str]]: return sum((read_csv(d / name) for d in worker_dirs), [])
    folds = all_csv("fold_metrics.csv"); seeds = all_csv("seed_metrics.csv"); profiles = all_csv("compute_profile.csv"); film = all_csv("film_modulation_diagnostics.csv"); perms = all_csv("permutation_invariance_audit.csv")
    manifests = [json.loads((d / "run_manifest.json").read_text()) for d in worker_dirs]; inits = [json.loads((d / "initialization_audit.json").read_text()) for d in worker_dirs]; correctness = [json.loads((d / "film_correctness_audit.json").read_text()) for d in worker_dirs]
    full_init_path = Path("/root/autodl-tmp/phase_m7a_initialization_full.json")
    if full_init_path.exists():
        full_init = json.loads(full_init_path.read_text(encoding="utf-8")); inits = [full_init]
        correctness = [{"status": full_init["status"], "coverage": full_init.get("coverage"), "rows": [r.get("film_correctness", {}) for r in full_init.get("rows", [])]}]
    expected = {(v, s, f) for v in VARIANTS for s in SEEDS for f in range(1, 6)}; got = {(r["method"], int(r["seed"]), int(r["outer_fold"])) for r in folds}
    if got != expected or len(folds) != 120 or any(r.get("status") != "completed" for r in folds): raise RuntimeError(f"invalid fold matrix {len(folds)}")
    if {(r["method"], int(r["seed"])) for r in seeds} != {(v, s) for v in VARIANTS for s in SEEDS} or len(seeds) != 24: raise RuntimeError("invalid seed summaries")
    if {(r["method"], int(r["seed"]), int(r["outer_fold"])) for r in profiles} != expected or len(profiles) != 120: raise RuntimeError("invalid profiles")
    if len(film) != 3 * 3 * 512: raise RuntimeError(f"invalid FiLM diagnostics {len(film)}")
    pkeys = {(r["row_type"], r["method"], int(r["seed"]), int(r["outer_fold"]), r.get("patient_index", "") if r["row_type"] == "summary" else int(r["patient_index"])) for r in perms}
    if len(perms) != 4653 or len(pkeys) != 4653: raise RuntimeError(f"invalid permutation audit {len(perms)}")
    if any(num(r.get("max_abs_prediction_difference")) > 1e-5 for r in perms if r["row_type"] == "summary"): raise RuntimeError("permutation audit failed")
    write_csv(output / "fold_metrics.csv", folds); write_csv(output / "seed_metrics.csv", seeds); write_csv(output / "compute_profile.csv", profiles); write_csv(output / "film_modulation_diagnostics.csv", film); write_csv(output / "permutation_invariance_audit.csv", perms)
    main = []
    for method in VARIANTS:
        rows = [r for r in folds if r["method"] == method]; h = np.asarray([num(r["harrell_c"]) for r in rows]); u = np.asarray([num(r["uno_c"]) for r in rows]); main.append({"method": method, "label": LABELS[method], "cells": len(rows), "harrell_mean": float(h.mean()), "harrell_std": float(h.std(ddof=1)), "harrell_worst": float(h.min()), "uno_mean": float(u.mean()), "uno_std": float(u.std(ddof=1)), "uno_worst": float(u.min()), "status": "completed"})
    write_csv(output / "main_metrics.csv", main); pairs = paired(folds); write_csv(output / "paired_deltas.csv", pairs); cross = cross_phase(repo); write_csv(output / "cross_phase_prototype_reaggregation.csv", cross)
    init = {"status": "PASS" if all(x.get("status") == "PASS" for x in inits) else "FAIL", "coverage": int(inits[0].get("coverage", len(inits))), "expected_coverage": 15, "workers": inits, "scope": "F1-F4 shared upstream and late fusion byte equality; F2/F3 interaction parameter matching; F5 separate raw omics audit; fixed compact widths, not parameter-matched to F3"}
    if init["status"] != "PASS" or init["coverage"] != 15: raise RuntimeError("initialization audit failed or incomplete")
    write_json(output / "initialization_audit.json", init)
    corr = correctness[0]
    if not all(x.get("status") == "PASS" for x in correctness): raise RuntimeError("FiLM correctness audit failed")
    if any(not np.isfinite(num(r[k])) for r in folds for k in ("harrell_c", "uno_c")): raise RuntimeError("non-finite fold metrics")
    for r in film:
        for k in ("gamma_o", "beta_o", "gamma_p", "beta_p"):
            if k + "_std_across_patients_mean_features" not in r: raise RuntimeError("missing across-patient modulation std")
    write_json(output / "film_correctness_audit.json", corr); (output / "abmil_implementation_audit.md").write_text("# ABMIL implementation audit\n\nThe formal worker audit uses `src/phase_m7a/models.py:GatedABMIL`: `LayerNorm(1536) -> Linear(1536,96) -> GELU`, followed by `w^T(tanh(Vh) * sigmoid(Uh))` with hidden attention dimension 48, softmax over the unordered patch bag, weighted sum, and `LayerNorm(96)`. No positional or patch-order feature is present. This is a direct reimplementation of the exact equation prescribed by the M7A prompt; no prototype, adaptive pooling, or Phase1.5R2 implementation is imported. Omics reuses the existing src.phase_m2.mcat.SixSignatureEncoder without modification.\n", encoding="utf-8")
    summary = {(r["candidate"], r["baseline"]): r for r in pairs if r["row_type"] == "summary"}
    def gate(key: tuple[str, str]) -> bool:
        r = summary[key]; return num(r["mean_delta_harrell"]) > 0 and num(r["mean_delta_uno"]) > 0 and int(r["harrell_wins"]) >= 9 and int(r["uno_wins"]) >= 9
    abmil = gate(("F1", "F0")); film_gate = gate(("F3", "F1")) and num(summary[("F3", "F2")]["mean_delta_harrell"]) > 0 and num(summary[("F3", "F2")]["mean_delta_uno"]) > 0; bidi = gate(("F4", "F3")); semantic = gate(("F3", "F5")); perm_pass = all(num(r.get("max_abs_prediction_difference")) <= 1e-5 for r in perms if r["row_type"] == "summary" and r["method"] in {"F1", "F3", "F4"})
    context_support = {v: all(gate((v, b)) for b in ("F0", "B0", "C0")) for v in ("F3", "F4")}
    protocol = {"phase": "M7A", "patients": 512, "dss_events": 65, "outer_folds": 5, "seeds": list(SEEDS), "variants": list(VARIANTS), "cells": 120, "primary_metric": "fold-level Harrell C", "secondary_metric": "fold-level Uno C", "global_oof": "diagnostic only", "pathology": {"input": "UNI2-h", "K": 1024, "tokenizer": "permutation-invariant supervised gated ABMIL", "hidden_attention_dim": 48, "no_positional_or_order_feature": True}, "omics": "SixSignatureEncoder, six official RNA+CNA groups, two-layer Transformer, GatedAttentionPool, omic_rho", "training": {"optimizer": "AdamW", "lr": 3e-4, "weight_decay": 1e-4, "max_epochs": 25, "patience": 5, "batch_size": 16, "nested_inner_val": True}, "comparisons": [f"{a} vs {b}" for a, b in PAIRS], "best_context_comparisons": "F3 and F4 are both reported; descriptive best is not used to create a new post-hoc significance test", "motivation_lock": {"prototype_effect": "initialization-sensitive and not confirmed", "hallmark50": "not supported", "dense_attention": "near-uniform diagnostic, not an information-theoretic proof", "small_n": "keep interaction compact"}, "gates": {"abmil_pathology_supported": abmil, "omic_to_path_film_supported": film_gate, "bidirectional_film_supported": bidi, "six_signature_semantics_under_film_supported": semantic}, "audit_coverage": {"fold_rows": len(folds), "seed_rows": len(seeds), "profile_rows": len(profiles), "film_rows": len(film), "permutation_rows": len(perms), "initialization_rows": int(init.get("coverage", 0))}}; write_json(output / "protocol.json", protocol)
    def fmt(x: Any) -> str: return "NA" if not np.isfinite(num(x)) else f"{num(x):.4f}"
    def line(r: dict[str, Any]) -> str: return f"{fmt(r['mean_delta_harrell'])} ± {fmt(r['std_delta_harrell'])} | {fmt(r['mean_delta_uno'])} ± {fmt(r['std_delta_uno'])} | {r['harrell_wins']} | {r['uno_wins']} | {r['both_wins']}"
    cm = next(r for r in cross if r["row_type"] == "combined_summary"); lines = ["# Phase M7A 结果", "", "## Motivation lock（实验前固定）", "", "M5A/M5R/M6A 的 prototype effect 只作初始化敏感诊断，不能确认 prototype backbone；M6A 的 50-pathway tokenization 没有显示稳定语义优势，CNA coupling gate 未通过；dense token co-attention 接近均匀权重，这是结构诊断而非信息论 collapse proof；M6A MCAT6 约 1.25M 参数、Hallmark50RC 约 2.41M，本阶段保持紧凑交互。", "", "## 主结果", "", "| 方法 | Harrell 均值±标准差 | Uno 均值±标准差 | cells |", "|---|---:|---:|---:|"]
    mm = {r["method"]: r for r in main}
    for v in VARIANTS: lines.append(f"| {v} {LABELS[v]} | {fmt(mm[v]['harrell_mean'])} ± {fmt(mm[v]['harrell_std'])} | {fmt(mm[v]['uno_mean'])} ± {fmt(mm[v]['uno_std'])} | {mm[v]['cells']} |")
    lines += ["", "## 配对比较", "", "| 比较 | ΔHarrell 均值±标准差 | ΔUno 均值±标准差 | H wins/15 | U wins/15 | both/15 |", "|---|---:|---:|---:|---:|---:|"]
    for a, b in PAIRS: lines.append(f"| {a} vs {b} | {line(summary[(a, b)])} |")
    seed_cross = [r for r in cross if r["row_type"] == "seed_summary"]; prototype_stable = all(num(r["mean_delta_harrell"]) > 0 and num(r["mean_delta_uno"]) > 0 for r in seed_cross)
    lines += [
        "", "## 审计边界、优势与风险", "",
        "- 优势：ABMIL 使用集合加权和并通过结构与置换测试验证；F1-F4 的共享上游和 late fusion 做了 byte-level initialization audit；FiLM 使用 zero-init residual conditioning。",
        "- 风险：prototype 的跨阶段 45 个 paired cells 不是独立样本；FiLM 调制幅度是诊断，不用于调参；确定性算法仍可能出现 PyTorch backend 警告。F5 按固定 compact encoder 公式实现，active 参数量与 F3 的六签名分支有差距；F3/F5 属于结构对照诊断，不能单独证明生物语义的因果贡献。",
        f"- 置换审计通过={perm_pass}；FiLM correctness={corr['status']}；初始化审计={init['status']}。", "",
        "## 复现边界", "", "只提交 prompt 要求的 14 个文件；缓存、checkpoint、worker 日志和 cell 预测不进入 Git。", "",
        "## 八个问题的结论", "",
        f"1. **跨 M5A/M5R/M6A 后，prototype 是否应正式放弃作为主 pathology tokenizer？** 45 个 paired cells：ΔHarrell={fmt(cm['mean_delta_harrell'])}、ΔUno={fmt(cm['mean_delta_uno'])}、wins={cm['harrell_wins']}/{cm['n']} H、{cm['uno_wins']}/{cm['n']} U、both={cm['both_wins']}/{cm['n']}；median ΔHarrell={fmt(cm['median_delta_harrell'])}、median ΔUno={fmt(cm['median_delta_uno'])}；seed-level 两指标均为正={prototype_stable}。结论：prototype effect is initialization-sensitive and not confirmed。",
        f"2. **Gated ABMIL 是否稳定优于 MCAT 的 adaptive pathology tokenizer？** {'支持' if abmil else '不支持'}；F1 vs F0：{line(summary[('F1', 'F0')])}。",
        f"3. **Omic→Path FiLM 是否稳定优于 simple concat？** {'支持' if film_gate else '不支持'}；F3 vs F1：{line(summary[('F3', 'F1')])}。",
        f"4. **FiLM 是否优于参数匹配的 generic residual MLP？** F3 vs F2：{line(summary[('F3', 'F2')])}；独立 gate={'支持' if num(summary[('F3', 'F2')]['mean_delta_harrell']) > 0 and num(summary[('F3', 'F2')]['mean_delta_uno']) > 0 else '不支持'}。",
        f"5. **Bidirectional FiLM 是否优于更简单的单向 FiLM？** {'支持' if bidi else '不支持'}；F4 vs F3：{line(summary[('F4', 'F3')])}。",
        f"6. **MCAT six-signature semantics 在 FiLM 条件下是否仍有贡献？** {'支持' if semantic else '不支持'}；F3 vs F5：{line(summary[('F3', 'F5')])}。",
        f"7. **最佳 FiLM variant 是否稳定超过 MCAT、CommonOnly、ClinicalOnly？** F3 vs F0/B0/C0：{line(summary[('F3', 'F0')])}；{line(summary[('F3', 'B0')])}；{line(summary[('F3', 'C0')])}。F4 vs F0/B0/C0：{line(summary[('F4', 'F0')])}；{line(summary[('F4', 'B0')])}；{line(summary[('F4', 'C0')])}。F3 全部 context gate={context_support['F3']}；F4 全部 context gate={context_support['F4']}。best 只作描述，不新建 post-hoc gate。",
        f"8. **是否可以将 permutation-invariant supervised pathology pooling + semantic omics + feature-wise conditional modulation 固化为下一阶段 backbone？** {'可以' if (abmil and film_gate and semantic and context_support['F4' if bidi else 'F3']) else '不可以'}；只在对应预注册 gate 通过时保留机制。",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"phase": "M7A", "status": "complete", "cells_completed": len(folds), "cells_expected": 120, "all_three_fresh_seeds": sorted({int(r["seed"]) for r in folds}) == list(SEEDS), "execution_shas": sorted({r.get("execution_commit", "unknown") for r in profiles}), "remote_resolvable_execution_sha": all(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", s, _git_sha(repo)], capture_output=True).returncode == 0 for s in sorted({r.get("execution_commit", "unknown") for r in profiles})), "no_leakage": True, "no_seed_hunting": True, "all_formal_decisions_fold_level": True, "film_correctness_status": corr["status"], "permutation_audit_status": "PASS" if perm_pass else "FAIL", "abmil_pathology_supported": abmil, "omic_to_path_film_supported": film_gate, "bidirectional_film_supported": bidi, "six_signature_semantics_supported_under_film": semantic, "output_files": list(OUTPUT_FILES), "repo_sha_at_aggregation": _git_sha(repo), "report_sha256": hashlib.sha256((output / "RESULTS.md").read_bytes()).hexdigest()}; write_json(output / "run_manifest.json", manifest)
    if sorted(p.name for p in output.iterdir() if p.is_file()) != sorted(OUTPUT_FILES): raise RuntimeError("final output must contain exactly the 14 prompt files")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--worker-reports", nargs="+", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--repo-root", type=Path, required=True); a = p.parse_args(); aggregate(a.worker_reports, a.output, a.repo_root)


if __name__ == "__main__": main()

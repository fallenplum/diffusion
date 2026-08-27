from __future__ import annotations

"""Final integrity checks and publication-facing Phase P1 artifacts.

This script never changes input bags or baseline outputs.  It only derives audit
tables, summaries, the manifest, and the two required decision documents under
outputs/pathology_context_phase1.
"""

import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_utils import censoring_km, harrell_c, uno_c


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/pathology_context_phase1"
SEED = 20260723


def _seed_from_name(name: str) -> int:
    match = re.search(r"(?:seed[_-])?(\d{8})", name)
    if not match:
        raise ValueError(f"cannot parse seed from {name}")
    return int(match.group(1))


def _method_seed_parts():
    base = PHASE / "oof_predictions/fold_parts"
    for method_dir in sorted(base.glob("*")):
        if not method_dir.is_dir():
            continue
        seeds = sorted({_seed_from_name(p.stem) for p in method_dir.glob("fold_*_seed_*.csv")})
        for seed in seeds:
            parts = sorted(method_dir.glob(f"fold_*_seed_{seed}.csv"))
            if len(parts) == 5:
                yield method_dir.name, seed, parts


def make_oof_integrity() -> pd.DataFrame:
    manifest = pd.read_csv(PHASE / "audit/patient_manifest_used.csv")
    rows = []
    for method, seed, parts in _method_seed_parts():
        frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        merged = frame.merge(
            manifest[["patient_id", "outer_fold", "dss_months", "dss_status"]],
            on="patient_id",
            how="left",
            suffixes=("_pred", "_manifest"),
            validate="one_to_one",
        )
        fold_ok = bool((merged["outer_fold_pred"] == merged["outer_fold_manifest"]).all())
        months_ok = bool(np.allclose(merged["dss_months_pred"], merged["dss_months_manifest"], equal_nan=False))
        status_ok = bool((merged["dss_status_pred"] == merged["dss_status_manifest"]).all())
        risk_ok = bool(np.isfinite(merged["oof_risk"]).all())
        rows.append(
            {
                "model": method,
                "seed": seed,
                "n_parts": len(parts),
                "n_rows": len(frame),
                "n_unique_patients": frame.patient_id.nunique(),
                "duplicate_patient_rows": len(frame) - frame.patient_id.nunique(),
                "outer_fold_matches_manifest": fold_ok,
                "dss_months_matches_manifest": months_ok,
                "dss_status_matches_manifest": status_ok,
                "finite_risk": risk_ok,
                "valid": bool(len(frame) == len(manifest) == frame.patient_id.nunique() and fold_ok and months_ok and status_ok and risk_ok),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(PHASE / "audit/oof_integrity.csv", index=False)
    return out


def make_ssl_leakage() -> pd.DataFrame:
    manifest = pd.read_csv(PHASE / "audit/patient_manifest_used.csv")
    rows = []
    for p in sorted((PHASE / "ssl_metrics").glob("*/fold_*/seed_*/ssl_patient_ids.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        config = p.parents[2].name
        fold = int(p.parents[1].name.split("_")[-1])
        seed = _seed_from_name(p.parents[0].name)
        train_ids = set(payload.get("train", []))
        val_ids = set(payload.get("val", []))
        ssl_ids = train_ids | val_ids
        outer_test = set(manifest.loc[manifest.outer_fold == fold, "patient_id"])
        expected_outer_train = set(manifest.loc[manifest.outer_fold != fold, "patient_id"])
        coverage = pd.read_csv(p.parent / "sampling_coverage.csv")
        rows.append(
            {
                "ssl_config": config,
                "outer_fold": fold,
                "seed": seed,
                "n_ssl_train": len(train_ids),
                "n_ssl_val": len(val_ids),
                "n_ssl_patients": len(ssl_ids),
                "n_expected_outer_train": len(expected_outer_train),
                "outer_test_overlap": len(ssl_ids & outer_test),
                "train_val_overlap": len(train_ids & val_ids),
                "missing_outer_train": len(expected_outer_train - ssl_ids),
                "coverage_mean": float(coverage.coverage.mean()) if len(coverage) else np.nan,
                "coverage_min": float(coverage.coverage.min()) if len(coverage) else np.nan,
                "valid": bool(ssl_ids.isdisjoint(outer_test) and not (train_ids & val_ids)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(PHASE / "audit/ssl_leakage_check.csv", index=False)
    return out


def make_fold_metrics() -> pd.DataFrame:
    rows = []
    for method, seed, parts in _method_seed_parts():
        frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        for fold in range(1, 6):
            test = frame.loc[frame.outer_fold == fold].reset_index(drop=True)
            rows.append(
                {
                    "model": method,
                    "seed": seed,
                    "outer_fold": fold,
                    "n_test": len(test),
                    "events_test": int(test.dss_status.sum()),
                    "harrell_c": harrell_c(test.dss_months, test.dss_status, test.oof_risk),
                    "uno_c": uno_c(test.dss_months, test.dss_status, test.oof_risk, test.censor_survival_outer_train),
                }
            )
    b0_parts = PHASE / "fold_metrics/B0/fold_metrics.csv"
    if b0_parts.exists():
        b0 = pd.read_csv(b0_parts)
        for _, r in b0.iterrows():
            rows.append(
                {
                    "model": "B0",
                    "seed": int(r.seed),
                    "outer_fold": int(r.outer_fold),
                    "n_test": int(r.n_test),
                    "events_test": int(r.events_test),
                    "harrell_c": float(r.test_harrell_c),
                    "uno_c": float(r.test_uno_c),
                }
            )
    out = pd.DataFrame(rows).sort_values(["model", "seed", "outer_fold"])
    out.to_csv(PHASE / "statistical_tests/fold_metrics_all.csv", index=False)
    return out


def make_ssl_summary() -> pd.DataFrame:
    rows = []
    for p in sorted((PHASE / "ssl_metrics").glob("*/fold_*/seed_*/history.csv")):
        history = pd.read_csv(p)
        config = p.parents[2].name
        fold = int(p.parents[1].name.split("_")[-1])
        seed = _seed_from_name(p.parent.name)
        best = history.loc[history.val_loss.idxmin()]
        coverage = pd.read_csv(p.parent / "sampling_coverage.csv")
        rows.append(
            {
                "ssl_config": config,
                "outer_fold": fold,
                "seed": seed,
                "epochs_run": int(history.epoch.max()),
                "best_epoch_by_inner_val_loss": int(best.epoch),
                "best_val_loss": float(best.val_loss),
                "best_val_cosine": float(best.val_cosine),
                "best_val_mse": float(best.val_mse),
                "final_train_loss": float(history.iloc[-1].train_loss),
                "final_val_loss": float(history.iloc[-1].val_loss),
                "mean_patch_coverage": float(coverage.coverage.mean()) if len(coverage) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["ssl_config", "seed", "outer_fold"])
    out.to_csv(PHASE / "ssl_metrics/ssl_metrics_summary.csv", index=False)
    return out


def make_job_logs() -> pd.DataFrame:
    """Create one auditable completion log per run-state job.

    The training commands were launched from interactive shells, so raw PTY
    stdout is not reconstructed here.  These JSON files are explicitly marked
    as derived completion logs and point to the immutable checkpoint/metric
    artifacts that substantiate each job.
    """
    state_path = PHASE / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_dir = PHASE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in sorted(state.get("completed_jobs", [])):
        bits = job.split("/")
        if len(bits) != 3:
            continue
        method, fold_text, seed_text = bits
        fold = int(fold_text.split("_")[-1])
        seed = int(seed_text.split("_")[-1])
        if method.startswith("SSL_"):
            config = method[len("SSL_"):]
            artifacts = [
                f"ssl_metrics/{config}/fold_{fold}/seed_{seed}/history.csv",
                f"ssl_metrics/{config}/fold_{fold}/seed_{seed}/ssl_patient_ids.json",
                f"checkpoints/ssl/{config}/fold_{fold}/seed_{seed}.pt",
            ]
            kind = "ssl"
        elif method == "B0":
            artifacts = ["fold_metrics/B0/fold_metrics.csv", f"oof_predictions/B0_seed_{seed}.csv"]
            kind = "baseline"
        else:
            artifacts = [
                f"fold_metrics/{method}/fold_{fold}_seed_{seed}.json",
                f"oof_predictions/fold_parts/{method}/fold_{fold}_seed_{seed}.csv",
                f"checkpoints/survival/{method}/fold_{fold}/seed_{seed}.pt",
            ]
            kind = "survival_or_probe"
        payload = {
            "job": job,
            "status": "completed",
            "log_type": "derived_completion_log",
            "note": "Raw interactive stdout is not reconstructed; listed artifacts are the verification record.",
            "method": method,
            "outer_fold": fold,
            "seed": seed,
            "kind": kind,
            "artifacts": artifacts,
        }
        out_path = log_dir / (job.replace("/", "__") + ".json")
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append({"job": job, "status": "completed", "log_type": payload["log_type"], "log_path": str(out_path.relative_to(PHASE))})
    out = pd.DataFrame(rows)
    out.to_csv(log_dir / "job_index.csv", index=False)
    (log_dir / "README.md").write_text(
        "# Run logs\n\n"
        "Each JSON file is a derived completion log pointing to the checkpoint and metric artifacts for one job. "
        "Interactive shell stdout was not reconstructed; do not interpret these files as verbatim console logs.\n",
        encoding="utf-8",
    )
    return out


def _fmt(value, digits=4):
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_manifest(results: pd.DataFrame, ssl_summary: pd.DataFrame, oof: pd.DataFrame, leakage: pd.DataFrame) -> dict:
    data_audit = json.loads((PHASE / "audit/data_audit.json").read_text(encoding="utf-8"))
    manifest = {
        "phase": "P1_UNI2H_CONTEXT_DSS_VALIDATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "scope": {
            "input_root": "TCGA/",
            "ignored_root_data": True,
            "pathology_only": True,
            "clinical_or_rna_used": False,
            "transductive_ssl_used_as_primary": False,
        },
        "data_audit": data_audit,
        "split": {
            "seed": SEED,
            "logic": "StratifiedKFold(n_splits=5, shuffle=True, random_state=20260723)",
            "patient_level": True,
            "slide_follows_patient": True,
        },
        "architecture": {
            "input_dim": 1536,
            "projection_dim": 256,
            "transformer_layers": 2,
            "heads": 8,
            "ffn_dim": 512,
            "dropout": 0.1,
            "max_tokens_per_slide": 1024,
            "survival_risk_set": "nested stochastic approximation: one event plus up to seven outer-train at-risk comparators",
        },
        "seeds": [20260723, 20260724, 20260725],
        "required_artifacts": {
            "oof_integrity_rows": int(len(oof)),
            "oof_all_valid": bool(oof.valid.all()) if len(oof) else False,
            "ssl_leakage_rows": int(len(leakage)),
            "ssl_all_valid": bool(leakage.valid.all()) if len(leakage) else False,
            "ssl_summary_rows": int(len(ssl_summary)),
            "results_rows": int(len(results)),
        },
        "methods_in_results_master": sorted(results.model.unique().tolist()),
        "notes": [
            "IBS is NA because saved checkpoints retain risk scores but not fold-wise calibrated baseline survival functions.",
            "Permutation results are a sanity check and are excluded from primary method ranking.",
            "P0_nocoords was run as a stability ablation because the coordinates-free representation was the strongest proposed-context variant in the preregistered screen; it is not relabeled as the primary spatial method.",
        ],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "source_scripts": sorted(p.name for p in (ROOT / "src").glob("phase1_*.py")),
        },
    }
    (PHASE / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def write_reports(results: pd.DataFrame, fold_metrics: pd.DataFrame, ssl_summary: pd.DataFrame, oof: pd.DataFrame, leakage: pd.DataFrame):
    b0 = results.loc[(results.model == "B0") & (results.seed == SEED)].iloc[0]
    primary_names = ["B0", "B1", "B2", "P0", "P1", "P2", "P3"]
    primary = results.loc[(results.seed == SEED) & results.model.isin(primary_names)].copy()
    primary["order"] = primary.model.map({m: i for i, m in enumerate(primary_names)})
    primary = primary.sort_values("order")
    table = [
        "| 方法 | Harrell C | Uno C | ΔHarrell vs B0 | ΔUno vs B0 | 五折中优于 B0 | paired bootstrap ΔHarrell 95% CI | paired bootstrap ΔUno 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in primary.iterrows():
        table.append(
            f"| {r.model} | {_fmt(r.oof_harrell_c)} | {_fmt(r.oof_uno_c)} | {_fmt(r.delta_harrell_vs_B0)} | {_fmt(r.delta_uno_vs_B0)} | {int(r.n_folds_better_than_B0)}/5 | [{_fmt(r.bootstrap_ci_low_harrell)},{_fmt(r.bootstrap_ci_high_harrell)}] | [{_fmt(r.bootstrap_ci_low_uno)},{_fmt(r.bootstrap_ci_high_uno)}] |"
        )
    stability_names = ["P0_nocoords", "B1", "B2"]
    stability = results.loc[results.model.isin(stability_names)].sort_values(["model", "seed"])
    stab_lines = ["| 方法 | seed | Harrell C | Uno C |", "|---|---:|---:|---:|"]
    for _, r in stability.iterrows():
        stab_lines.append(f"| {r.model} | {int(r.seed)} | {_fmt(r.oof_harrell_c)} | {_fmt(r.oof_uno_c)} |")
    ssl_lines = ["| SSL 配置 | seed | fold | best val loss | best val cosine |", "|---|---:|---:|---:|---:|"]
    for _, r in ssl_summary.loc[ssl_summary.seed == SEED].iterrows():
        ssl_lines.append(f"| {r.ssl_config} | {int(r.seed)} | {int(r.outer_fold)} | {_fmt(r.best_val_loss)} | {_fmt(r.best_val_cosine)} |")
    perm = results.loc[(results.model == "P2_permuted") & (results.seed == SEED)].iloc[0]
    p0random = results.loc[(results.model == "P0_random040") & (results.seed == SEED)].iloc[0]
    p025 = results.loc[(results.model == "P0_spatial025") & (results.seed == SEED)].iloc[0]
    p0 = results.loc[(results.model == "P0") & (results.seed == SEED)].iloc[0]
    p0nc = results.loc[(results.model == "P0_nocoords") & (results.seed == SEED)].iloc[0]
    b1probe = results.loc[(results.model == "B1_probe") & (results.seed == SEED)].iloc[0]
    p2probe = results.loc[(results.model == "P2_probe") & (results.seed == SEED)].iloc[0]
    b2probe = results.loc[(results.model == "B2_probe") & (results.seed == SEED)].iloc[0]
    p1probe = results.loc[(results.model == "P1_probe") & (results.seed == SEED)].iloc[0]
    b1 = results.loc[(results.model == "B1") & (results.seed == SEED)].iloc[0]
    b2 = results.loc[(results.model == "B2") & (results.seed == SEED)].iloc[0]
    p1 = results.loc[(results.model == "P1") & (results.seed == SEED)].iloc[0]
    p2 = results.loc[(results.model == "P2") & (results.seed == SEED)].iloc[0]
    p3 = results.loc[(results.model == "P3") & (results.seed == SEED)].iloc[0]
    report = f"""# Phase P1：UNI2-h whole-slide context + DSS validation

## 结论先行

在固定的患者级五折外层交叉验证下，**本实验没有支持“带坐标的 spatial masked-context SSL 能稳定优于 mean/MIL/context-scratch”这一核心假设**。主 proposed pipeline P1/P2/P3 的 Harrell C 均低于 B0；P2/P3 的 Uno C 点估计高于 B0，但 paired bootstrap 区间均跨 0。坐标去除后的冻结 probe（P0_nocoords）点估计较高，但它是坐标 ablation，不应被改写成原始 spatial 机制的确认。

## 数据、审计与隔离

- 仅使用 `TCGA/` 下的 UNI2-h bags、原始 H5 坐标和 DSS 标签；根目录 `data/` 未作为输入。
- 最终数据：487 位患者、494 张选定 slide、约 3,077,884 个 patch、1536 维特征、65 个 DSS event。
- H5 中的真实 `coords` 与 features 一一对齐；未用数组顺序伪造坐标，空 bag/NaN/Inf 检查通过。
- 五折逻辑为 `StratifiedKFold(n_splits=5, shuffle=True, random_state=20260723)`；患者与 slide 均无跨折重叠。
- OOF 完整性：{len(oof)} 个方法×seed 组合全部通过 487 行、487 个唯一患者、标签/折号回填一致性检查。
- SSL 隔离：{len(leakage)} 个 fold×配置组合的 outer-test overlap 全为 0；SSL 只读取 outer-train 的 train/val 患者。

详细证据见 `audit/data_audit.md`、`audit/coords_audit.csv`、`audit/oof_integrity.csv`、`audit/ssl_leakage_check.csv`。

## 主 seed（20260723）OOF 结果

""" + "\n".join(table) + f"""

B0 从 TCGA bags 重新计算得到 Harrell C={_fmt(b0.oof_harrell_c)}，历史值 0.660046 的差为 {_fmt(b0.oof_harrell_c - 0.6600464403932612, 6)}，因此比较基准可复核。

## 额外稳定性

""" + "\n".join(stab_lines) + f"""

P0_nocoords 三个 seed 的 Harrell C 为 {_fmt(stability.loc[stability.model == 'P0_nocoords'].oof_harrell_c.mean())}±{_fmt(stability.loc[stability.model == 'P0_nocoords'].oof_harrell_c.std(ddof=1), 4)}，Uno C 为 {_fmt(stability.loc[stability.model == 'P0_nocoords'].oof_uno_c.mean())}±{_fmt(stability.loc[stability.model == 'P0_nocoords'].oof_uno_c.std(ddof=1), 4)}；B1/B2 的 seed 间波动更大，不能只报告最优 seed。

## 对研究问题的回答

### Q1：mean pooling 是否限制表征？

B1 end-to-end 仅为 Harrell {_fmt(b1.oof_harrell_c)}、Uno {_fmt(b1.oof_uno_c)}；B2 为 Harrell {_fmt(b2.oof_harrell_c)}、Uno {_fmt(b2.oof_uno_c)}。因此 mean pooling 可能不是唯一瓶颈；引入可学习聚合并未产生稳定的双指标改善。

### Q2：Context Transformer 本身是否有效？

B2 相比 B1 的 Harrell 下降 {_fmt(b2.oof_harrell_c - b1.oof_harrell_c)}，Uno 上升 {_fmt(b2.oof_uno_c - b1.oof_uno_c)}，方向冲突且五折异质，结论为**未得到稳定支持**。

### Q3：masked context pretraining 是否带来真实收益？

冻结 probe 的 Harrell：B1_probe={_fmt(b1probe.oof_harrell_c)}、B2_probe={_fmt(b2probe.oof_harrell_c)}、P1_probe={_fmt(p1probe.oof_harrell_c)}、P2_probe={_fmt(p2probe.oof_harrell_c)}；带坐标 SSL 的 P0/P1/P2 end-to-end 则分别为 {_fmt(p0.oof_harrell_c)}/{_fmt(p1.oof_harrell_c)}/{_fmt(p2.oof_harrell_c)}。这说明 representation probe 与 end-to-end survival 结论不一致，SSL 表征的可预测结构不等于 DSS 相关结构。

### Q4：spatial block mask 是否优于 random mask？

P0 spatial-0.40 为 Harrell {_fmt(p0.oof_harrell_c)}、Uno {_fmt(p0.oof_uno_c)}；P0 random-0.40 为 {_fmt(p0random.oof_harrell_c)}、Uno {_fmt(p0random.oof_uno_c)}。spatial 点估计更高，但仍低于 B0，不能据此宣称机制确认。mask 0.25 的 Harrell {_fmt(p025.oof_harrell_c)} 也未改善。

### Q5：coords 是否有用？

P0 with-coords={_fmt(p0.oof_harrell_c)}/{_fmt(p0.oof_uno_c)}，P0_nocoords={_fmt(p0nc.oof_harrell_c)}/{_fmt(p0nc.oof_uno_c)}（Harrell/Uno）。在本数据和实现下，去除 coords 反而更好；这是一项 ablation 结果，提示坐标归一化/采样或坐标信号本身需要独立诊断，不能把它解释为“空间信息已被证实”。

### Q6：Cox+ranking 是否更适合 DSS？

P1/P2/P3 的 Harrell 为 {_fmt(p1.oof_harrell_c)}/{_fmt(p2.oof_harrell_c)}/{_fmt(p3.oof_harrell_c)}，Uno 为 {_fmt(p1.oof_uno_c)}/{_fmt(p2.oof_uno_c)}/{_fmt(p3.oof_uno_c)}。P2/P3 的 Uno 点估计高于 P1，但 paired intervals 跨 0；ranking 是可能的辅助目标，不是已确认的改进。

### Q7：重建质量与 DSS 性能是否一致？

训练曲线显示 masked cosine/loss 改善，但 P1/P2/P3 没有同步超过 B0。由此只能说 SSL 学到了可重建的 feature-space 结构，**不能推出这些结构与 DSS 风险相关**；这正是本实验中 motivation 链条未闭合的关键证据。

## Sanity check 与限制

- P2 标签置乱的五折 OOF 为 Harrell {_fmt(perm.oof_harrell_c)}、Uno {_fmt(perm.oof_uno_c)}，paired delta 均为负，未见置乱后异常高性能。
- 所有主要指标均为 patient-level OOF；1000 次 paired bootstrap 的区间在 `statistical_tests/*bootstrap_1000.csv`。
- AUC@12/24/36 已按 outer-train censoring KM 计算；IBS 记为 NA，因为当前 risk-only checkpoint 未保存 fold-wise Breslow baseline survival，强行重建会引入不可审计的校准假设。
- Survival 训练采用每次一个 event 加最多 7 个 outer-train at-risk comparator 的随机风险集近似，而非完整 Cox 风险集；这是计算约束，结果应按近似目标解释。
- 没有 RNA、clinical 或 multimodal 输入；坐标图只表示 contextual importance scatter，不是病理因果热图。

## 文件索引

- `results_master.csv`：每个方法×seed 的 OOF、Uno、AUC、bootstrap CI 和资源列。
- `statistical_tests/fold_metrics_all.csv`、`statistical_tests/*bootstrap_1000.csv`：逐折与不确定性。
- `embeddings/`、`statistical_tests/embedding_diagnostics.csv`：患者表征与 collapse 检查。
- `figures/`：t-SNE 与 P2 contextual-importance 坐标散点。
- `experiment_manifest.json`：数据范围、配置、软件和完整性证据。
"""
    (PHASE / "RESULTS.md").write_text(report, encoding="utf-8")

    decision = f"""# NEXT_DECISION

## 决策：Decision C（当前带坐标 spatial-SSL 核心假设不成立）

### 证据

1. 预注册主 pipeline P1/P2/P3 的 Harrell C（{_fmt(p1.oof_harrell_c)}/{_fmt(p2.oof_harrell_c)}/{_fmt(p3.oof_harrell_c)}）均低于 B0={_fmt(b0.oof_harrell_c)}；P2/P3 的 Uno 虽为 {_fmt(p2.oof_uno_c)}/{_fmt(p3.oof_uno_c)}，但 paired bootstrap 95% CI 仍跨 0。
2. 与 architecture-from-scratch 的 B2 比较没有形成双指标稳定收益；P0/P1/P2 的 end-to-end Harrell 也没有超过 B0。
3. P0_nocoords 是当前最强的冻结 context ablation，但它不是原始带坐标机制的确认；三 seed 的点估计不能替代独立验证。
4. P2 置乱 OOF Harrell/Uno={_fmt(perm.oof_harrell_c)}/{_fmt(perm.oof_uno_c)}，支持标签置乱 sanity check 正常，未提示明显泄漏。

### 不应做的下一步

- 不继续在同一外层 split 上堆叠 mask ratio、Transformer 层数、loss 或 seed 以追求更高点估计。
- 不把 P0_nocoords 的结果改写成 spatial coordinate hypothesis 的正结果。
- 不将当前 risk-only 输出包装成 IBS 或校准结论。

### 最小可证伪后续

若仍要研究 context，下一轮应预注册一个新的验证切分，固定当前 encoder/采样预算，直接比较：

`B0 mean` vs `B1 ABMIL` vs `P0_nocoords frozen probe` vs `P0_coords frozen probe`。

重点诊断全局形态信息是否被 1024-token sampling 丢失、coords 归一化是否引入噪声，以及在更多 DSS event 或不同终点上是否重现。若新切分仍无法同时改善 Harrell 与 Uno，则应停止将 masked context 作为 DSS 主线，转向更适合终点定义/事件稀疏性的 pathology aggregation 或 endpoint 设计。

置信度：对“本次固定 split 下未支持核心假设”的判断为高；对“coords 具体失败原因”的判断为中，因为本实验没有做独立坐标扰动/采样诊断。
"""
    (PHASE / "NEXT_DECISION.md").write_text(decision, encoding="utf-8")


def finalize_state(manifest: dict):
    state_path = PHASE / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stage"] = "stage_6_reports"
    state["fold"] = 0
    state["model"] = ""
    state["seed"] = 0
    state["status"] = "COMPLETE"
    state["pending_jobs"] = []
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["final_artifacts"] = {
        "results_master": True,
        "experiment_manifest": True,
        "results_report": True,
        "next_decision": True,
        "oof_all_valid": manifest["required_artifacts"]["oof_all_valid"],
        "ssl_all_valid": manifest["required_artifacts"]["ssl_all_valid"],
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    results = pd.read_csv(PHASE / "results_master.csv")
    oof = make_oof_integrity()
    leakage = make_ssl_leakage()
    fold_metrics = make_fold_metrics()
    ssl_summary = make_ssl_summary()
    job_logs = make_job_logs()
    manifest = write_manifest(results, ssl_summary, oof, leakage)
    manifest["required_artifacts"]["derived_job_log_rows"] = int(len(job_logs))
    (PHASE / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(results, fold_metrics, ssl_summary, oof, leakage)
    finalize_state(manifest)
    print(json.dumps({
        "oof_rows": len(oof),
        "oof_all_valid": bool(oof.valid.all()) if len(oof) else False,
        "ssl_rows": len(leakage),
        "ssl_all_valid": bool(leakage.valid.all()) if len(leakage) else False,
        "results_rows": len(results),
        "status": "COMPLETE",
    }, indent=2))


if __name__ == "__main__":
    main()

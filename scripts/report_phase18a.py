"""Report the locked Phase18A 2x2 only after independent evidence gates pass.

This script never trains, changes selection, or reads patient identities/risks.
All scores come from the independent evaluator's aggregate tables.  The
--protocol-only mode writes design/reproduction text without inventing results.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase18a_readout.data import P, R, cfg, source, write

ARMS = ["F0", "F1", "L0", "L1"]
REFERENCES = ["ANCHOR_CS", "R1_A3_noDirect", "HIST_CS_S32_CONCH32"]
METRICS = ["Harrell", "Uno60"]
DID = "DID=(L1-L0)-(F1-F0)"
CONTRASTS = {
    "L0-F0": {"L0": 1, "F0": -1},
    "L1-F1": {"L1": 1, "F1": -1},
    "L1-L0": {"L1": 1, "L0": -1},
    DID: {"L1": 1, "L0": -1, "F1": -1, "F0": 1},
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text(path, value):
    Path(path).write_text(value.rstrip() + "\n", encoding="utf-8")


def table(frame):
    frame = frame.copy()
    if frame.empty:
        return "无符合该表定义的记录。\n"
    for column in frame:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda x: "NA" if pd.isna(x) else f"{x:.6g}")
    return ("| " + " | ".join(map(str, frame.columns)) + " |\n|" +
            "|".join(["---"] * len(frame.columns)) + "|\n" +
            "\n".join("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |"
                      for row in frame.itertuples(index=False, name=None)) + "\n")


def require(condition, explanation):
    if not bool(condition):
        raise ValueError(explanation)


def protocol_documents(config, locked_source):
    tc = config["parent_training"]
    protocol = f"""# Phase18A：局部状态读出 × 患者条件交互协议

## 研究范围与来源

独立、有限的探索性 2×2。父提交 `{config['parent_commit']}`；父训练来源 `{config['parent_training_source']}`；本轮锁定训练来源 `{locked_source}`。Phase17/17R 原始结果封存，旧 R3 不启动。本轮不扩展扫描方向、gamma 幅度、delta/B 幅度或新的损失。

Mamba 原始逐位置输出与 MambaMIL attention pooling 的实际对照先于训练交付，见 [论文代码映射](READOUT_PAPER_CODE_MAPPING.md)。常规 SSM 局部输出不作为原创算法。此前 `{config['earlier_independent_attention_commit']}` 的四路径注意力是独立 inner-only 探索，不属于本轮 S6，也不并入本轮主比较。

## 唯一操作差异

F0/F1：`z[d]=sum_s h_T[d,s] * sum_t(m_t*C_t[s])/sum_t(m_t)`。

L0/L1：`z[d]=sum_t(m_t*sum_s(C_t[s]*h_t[d,s]))/sum_t(m_t)`。

两条路径保持相同 u、dt、B、C、A、有效 mask、对应模型原写入门及读出参数。局部读出采用 chunk={config['scan']['chunk_size']} 的精确分段 scan/checkpointing；段间不 detach，不使用截断 BPTT。无新增参数、归一化、attention pooling、D skip、残差或正则任务。

|臂|读出|状态条件|训练来源|
|---|---|---|---|
|F0|终态 × 平均 C|无跨模态状态条件|父 R1_S6_noDirect，仅通过资格/重放审计后引用|
|F1|终态 × 平均 C|delta/B 与原写入门|父 R1_A8_noDirect，仅通过资格/重放审计后引用|
|L0|逐位置 C_t h_t 加权池化|无跨模态状态条件|本轮 fresh inner/outer 拟合|
|L1|逐位置 C_t h_t 加权池化|delta/B 与原写入门|本轮 fresh inner/outer 拟合|

无条件模型仍由病理驱动选择性参数；自身 Clinical、RNA/CNA token late 主效应与同源 Clinical+S32 锚点保留。无高维 OmicsDirect 不等于移除组学。F1/L1 的条件因子是整套机制，不能从本轮拆分 delta、B、门的独立因果效应。

## 数据与拟合隔离

原队列 512 名患者、65 个 DSS events；原 5-fold patient CV × seeds 31/47/83。固定 UNI2-h 在 CONCH 足迹上的面积聚合、坐标顺序、每患者至多 512 patch 与多 WSI 等权；仅 geometry-only，未完成原图组织学 QA。所有预处理、基因选择、缺失处理及 anchor 拟合仅用对应训练分区；inner-val 只用于选 epoch；fresh outer 按所选轮数训练，outer-test 不用于任何选择。

共同训练：AdamW，lr={tc['learning_rate']}，weight_decay={tc['weight_decay']}，dropout={tc['dropout']}，clip={tc['gradient_clip']}，最大 {tc['max_epochs']} epochs，EMA alpha={tc['validation_ema_alpha']}，patience={tc['early_stopping_patience']}，microbatch={tc['microbatch_patients']}。微批量仅管理前向显存；损失及其梯度使用完整训练患者风险集的 Efron Cox。保持 anchor lambda={tc['anchor_lambda']}；不以训练内风险再拟合自由 stacking 标量，不拼接独立 folds 的 latent 坐标。

## 统计对象与八项家族

主要比较为 L0−F0、L1−F1、L1−L0、(L1−L0)−(F1−F0)，各自 Harrell/Uno60，共八项。每 seed 先汇总五折同折可比较对的分子/分母，再平均三个 seed 指标；不平均风险、不称 15 cells 为 seed mean。

20,000 次固定 outer-fold 内患者 multinomial 配对 bootstrap；所有方法和 seeds 共用患者权重，pair 权重 count_i×count_j；Uno60 的 censoring reverse-KM 来自原 outer-train。DID 每次重算四个完整 C 指标后作差，不拼接单方法区间。报告 95% 与 Bonferroni8 的 99.375% 区间。区间条件于已训练/已选择模型，不含完整重训练与研究者反复选择方差。

PFI 仅冻结 DSS 风险迁移，无 PFI 训练、epoch 选择或配置筛选。ANCHOR_CS、R1_A3_noDirect 和历史 Clinical+S32+CONCH32 旁列；最后一项视觉和训练协议不同，不称同输入消融。

## 诊断定义与工程停止规则

32 名训练患者按原顺序无标签选取，记录风险/状态差、患者等权 WSI 的 z/SSM 风险方差、独立条件路径梯度；只更换条件输入的分子状态，自身 late 证据不变。位置梯度分别对终态平方范数与最终患者风险求导，每 checkpoint 为第一名训练患者的所有 WSI 等权；探针重复患者数不是独立病例总数。位置为几何顺序，无医学时间含义。所选 checkpoint 隔离梯度和同 epoch 真实 Efron 梯度分开。

A：只有 L0−F0 与 L1−F1 在两个指标均正，才登记读出方向一致改善；否则停止自定义 S6 修补。B：读出改善而条件增量门不成立，记录 READOUT_HELPFUL_CONDITION_NOT_ESTABLISHED。C：L1−L0 与 DID 两指标均正，分别至少 2/3 seeds 同向，且 L1 对 ANCHOR_CS 与 R1_A3_noDirect 的两指标均值不下降，才登记 DIRECTIONAL_CANDIDATE。D：校正区间支持仍仅为当前反复研究队列内证据。工程筛选不是统计确认；无显著性不是等效或无信息。

任何状态/梯度增强都不能替代预测门。无论结果如何，完成本轮后停止，不自动追加 Phase18B、第四种读出、额外幅度或大规模 SOTA 矩阵。

## 计数、隐私与交付

正式新神经网络拟合 30 inner + 30 outer。历史 F0/F1 的 30 inner+30 outer、A3 的15 outer、历史 CONCH32 的15 outer 独立计为 90 次引用，不记为新神经网络拟合。标签突变检查的全部实际短拟合（包括审计 helper 报错前完成者）另行计数。冻结重放/探针不更新神经网络参数；但父 Data 会确定性重解同一 Clinical+S32 Ridge anchor：完整合格链的75次历史回放、60次历史探针、60次新checkpoint回放共195次验证重解，均与原checkpoint逐元素/bitwise一致，不产生新模型选择或新的OOF。195不是所有阶段底层solver调用总数，正式/短拟合内部及旧superseded探针的数据构造另外存在，不混称“审计没有任何优化”。

患者级 OOF、检查点、真实标识、原始矩阵和凭据留私有目录；公开仅源码、协议、匿名聚合及修复日志。新增文件隐私检查与全仓库继承旧 ZIP guard 问题分开；不得改历史以使检查表面通过。
"""
    text(R / "PROTOCOL.md", protocol)
    reproduce = """# Phase18A 复现入口

## 环境与私有前提

服务器 repo 工作目录执行，Python `/root/miniconda3/bin/python`；环境包版本与父 Phase17R `ENVIRONMENT.json` 经 historical audit 核对。完整父数据、固定视觉 cache、fold manifests 与 checkpoints 为私有资产，不随公开仓库发布。只取得源码无法凭空重建这些资产。

```bash
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest tests/test_phase18a_readout.py tests/test_phase18a_model.py -q
```

## 按证据依赖运行

1. 核对 `PROTOCOL_LOCK.json` 与实际 source hash，保留 parent 历史文件；算子/完整模型 CPU/CUDA 输出与梯度通过后登记 `OPERATOR_PARITY.json`。benchmark 是合成实现核验，不是正式训练。
2. `python scripts/validate_phase18a.py --historical-only --replay`，随后 `--aggregate-historical --shards 1`。并行时每个 shard 显式提供 `--shard k --shards n`，聚合必须覆盖全部且无重复。F0/F1/A3 来源、SHA、患者分区、epoch、预处理、软件及四条件重放全部通过，才允许历史引用。
3. `python scripts/audit_phase18a_leakage.py` 运行两个短拟合，内存中只改 outer-test 标签；没有 smoke 权重/风险文件。生产执行曾因比较 bool buffer 的 audit helper 报错而重做该检查，累计四个真实短拟合、八次 optimizer steps；不能把复现一次的两次写成生产总数。
4. `python scripts/start_phase18a.py` 启动可恢复调度，先全部30 inner，锁定 epoch，再30 fresh outer。`WORKERS.json` 只调整并发，不更改数学定义或预算；正式生产未盲目增加 epoch。
5. `python scripts/validate_phase18a.py --historical-probes-only`，再 `--aggregate-historical-probes --shards 1`；冻结父权重，补齐患者级/WSI级及两种位置梯度目标。旧未完成探针 shard 已标为 superseded，不当完成，也不计优化。
6. `python scripts/validate_phase18a.py --new-replay` 覆盖60新 checkpoint（含 inner/outer）；`python scripts/validate_phase18a.py` 独立重建 OOF/anchor、核对 sksurv、重算20,000次患者配对区间和冻结 PFI。
7. `python scripts/report_phase18a.py` 只在各证据门 PASS 后生成结果和决策。可先用 `--protocol-only` 生成设计/复现说明，不输出未产生的成绩。

禁止在已经完成的正式 run 上为了“复现命令”无意义重复短拟合；如确实重做，应增加真实执行计数。已有正式 checkpoint 只有来源与文件摘要一致才复用，不把日志或文件存在当完成。

## 公开与私有边界

`CELL_MANIFEST.csv`、逐折/逐seed指标、配对区间与诊断汇总可审阅；患者级 OOF、原始 feature/gene identities、checkpoint 和对应私有 manifest 不上传。正式源码锁与评估/报告源码分别记录；修复评估 helper 不静默改变训练模型或旧报告。完成 Phase18A 后停止自动扩展。
"""
    text(R / "REPRODUCE.md", reproduce)


def check_gates(locked_source):
    names = ["INDEPENDENT_VALIDATION.json", "CHECKPOINT_REPLAY.json", "HISTORICAL_REUSE_AUDIT.json",
             "OPERATOR_PARITY.json", "DATA_LEAKAGE_AUDIT.json", "HISTORICAL_READOUT_PROBE_AUDIT.json",
             "BOOTSTRAP_AUDIT.json", "FIT_AND_REFERENCE_COUNTS.json", "EVALUATION_MATH_AUDIT.json",
             "CONTRAST_POINT_AUDIT.json", "PARENT_PUBLISHED_METRIC_PARITY.json", "PROBE_COVERAGE_AUDIT.json"]
    gates = {name: read(R / name) for name in names}
    for name, gate in gates.items():
        require(gate["status"] == "PASS", name + " is not PASS")
        if name in ["INDEPENDENT_VALIDATION.json", "OPERATOR_PARITY.json", "DATA_LEAKAGE_AUDIT.json"]:
            require(gate["source"] == locked_source, name + " uses a different training source")
    done = read(P / "TRAINING_DONE.json")
    require(done["status"] == "PASS" and done["source"] == locked_source, "formal training incomplete/source changed")
    require(done["new_inner_fits"] == 30 and done["new_outer_fits"] == 30, "formal fit count mismatch")
    require(not done["phase18b_started"], "Phase18B outside protocol")
    return gates, done


def seed_contrasts(seeds):
    records = []
    for comparison, coefficients in CONTRASTS.items():
        for seed in sorted(seeds.seed.unique()):
            values = seeds[seeds.seed == seed].set_index("arm")
            row = {"comparison": comparison, "seed": int(seed)}
            for metric in METRICS:
                row[metric] = float(sum(weight * values.loc[arm, metric] for arm, weight in coefficients.items()))
            records.append(row)
    return pd.DataFrame(records)


def decide(summary, paired, seed_delta):
    primary = paired[paired.family == "primary"]
    require(len(primary) == 8 and set(primary.comparison) == set(CONTRASTS), "primary family coverage differs")
    require((primary.family_tests == 8).all(), "primary multiplicity not Bonferroni8")
    require(not primary.duplicated(["comparison", "metric"]).any(), "duplicate primary statistic")
    rows = []
    for comparison in CONTRASTS:
        for metric in METRICS:
            row = primary[(primary.comparison == comparison) & (primary.metric == metric)].iloc[0]
            sd = seed_delta[seed_delta.comparison == comparison][metric]
            require(abs(row.delta - sd.mean()) < 1e-10, "reported contrast does not match complete seed C statistics")
            rows.append({"comparison": comparison, "metric": metric, "mean_delta": float(row.delta),
                         "positive_seeds": int((sd > 0).sum()), "direction_positive": bool(row.delta > 0),
                         "Bonferroni8_positive": bool(row.Bonferroni_low > 0)})
    checks = pd.DataFrame(rows)
    readout = bool(checks[checks.comparison.isin(["L0-F0", "L1-F1"])].direction_positive.all())
    condition = checks[checks.comparison.isin(["L1-L0", DID])]
    condition_direction = bool(condition.direction_positive.all() and (condition.positive_seeds >= 2).all())
    strong_checks = []
    for reference in ["ANCHOR_CS", "R1_A3_noDirect"]:
        for metric in METRICS:
            delta = float(summary.loc["L1", metric] - summary.loc[reference, metric])
            strong_checks.append({"reference": reference, "metric": metric, "L1_minus_reference": delta,
                                  "no_mean_harm": bool(delta >= 0)})
    no_harm = all(row["no_mean_harm"] for row in strong_checks)
    if not readout:
        status = "STOP_CUSTOM_S6_READOUT_NOT_SUPPORTED"
    elif not (condition_direction and no_harm):
        status = "READOUT_HELPFUL_CONDITION_NOT_ESTABLISHED"
    else:
        status = "DIRECTIONAL_CANDIDATE"
    reasons = []
    if not readout:
        reasons.append("L0-F0 and L1-F1 are not both positive in both Harrell and Uno60")
    if not condition_direction:
        reasons.append("L1-L0 and DID fail positive two-metric means and/or at least two positive seeds per metric")
    if not no_harm:
        reasons.append("L1 does not preserve both-metric mean performance against all locked strong references")
    return {"status": status, "readout_helpful_direction_gate": readout,
            "condition_direction_and_seed_gate": condition_direction, "strong_reference_no_mean_harm": no_harm,
            "reasons": reasons, "primary_checks": rows, "strong_reference_checks": strong_checks,
            "corrected_positive_primary_statistics": checks[checks.Bonferroni8_positive][["comparison", "metric"]].to_dict("records"),
            "condition_both_metrics_and_DID_corrected_positive": bool(condition.Bonferroni8_positive.all()),
            "external_confirmation": False, "confirmed_new_mechanism": False,
            "non_significance_is_not_equivalence": True, "stop_after_phase18a": True, "phase18b_started": False}


def matched_epoch_table(curves):
    rows = []
    for left, right in [("L0", "F0"), ("L1", "F1"), ("L1", "L0")]:
        for epoch in [1, 5, 10, 20, 40, 80, 160]:
            a = curves[(curves.role == "inner") & (curves.arm == left) & (curves.epoch == epoch)]
            b = curves[(curves.role == "inner") & (curves.arm == right) & (curves.epoch == epoch)]
            common = a.merge(b, on=["seed", "fold"], suffixes=("_left", "_right"))
            if common.empty:
                continue
            for key in ["g_attention_grad", "delta_grad", "B_condition_grad", "write_grad", "readout_grad", "gamma_grad"]:
                x, y = common[key + "_left"].to_numpy(), common[key + "_right"].to_numpy()
                valid = y > 1e-12
                rows.append({"comparison": left + "-" + right, "epoch": epoch, "group": key,
                             "paired_cells": len(common), "left_median": float(np.median(x)),
                             "right_median": float(np.median(y)), "nonzero_reference_cells": int(valid.sum()),
                             "median_paired_ratio": float(np.median(x[valid] / y[valid])) if valid.any() else np.nan})
    return pd.DataFrame(rows)


def telemetry():
    rows = []
    for role in ["inner", "outer"]:
        values = pd.DataFrame(read(P / (role + "_GPU.json"))).sort_values("unix_time")
        require(len(values) >= 2, "insufficient GPU telemetry")
        duration = np.diff(values.unix_time.to_numpy())
        require((duration > 0).all(), "telemetry time is not increasing")
        row = {"role": role, "samples": len(values), "sample_span_seconds": float(duration.sum()),
               "workers_limit_max": int(values.workers_limit.max()), "running_max": int(values.running.max()),
               "GPU_percent_interval_weighted": float(np.average(values.GPU_percent.to_numpy()[:-1], weights=duration)),
               "GPU_percent_median": float(values.GPU_percent.median()), "GPU_percent_max": float(values.GPU_percent.max()),
               "memory_MiB_interval_weighted": float(np.average(values.memory_MiB.to_numpy()[:-1], weights=duration)),
               "memory_MiB_max": float(values.memory_MiB.max())}
        rows.append(row)
    return pd.DataFrame(rows)


def report():
    config = cfg(); locked_source = source(); lock = read(R / "PROTOCOL_LOCK.json")
    require(lock["source"] == locked_source, "training source changed after lock")
    gates, done = check_gates(locked_source)
    protocol_documents(config, locked_source)
    cells = pd.read_csv(R / "CELL_MANIFEST.csv")
    seeds = pd.read_csv(R / "SEED_METRICS.csv")
    folds = pd.read_csv(R / "FOLD_METRICS.csv")
    paired = pd.read_csv(R / "PAIRED_COMPARISONS.csv")
    mechanism = pd.read_csv(R / "READOUT_MECHANISM.csv")
    curves = pd.read_csv(R / "TRAINING_CURVES.csv")
    compute = pd.read_csv(R / "COMPUTE_PROFILE.csv")
    require(len(cells[cells.fit_classification == "new_fit"]) == 60, "manifest new fit count differs")
    require(len(cells[cells.fit_classification == "historical_reference"]) == 60, "manifest primary reference count differs")
    require(len(cells[cells.fit_classification == "historical_secondary_reference"]) == 15, "manifest A3 reference count differs")
    require(not cells.duplicated(["arm", "role", "seed", "fold"]).any(), "duplicate checkpoint cell")
    for arm in ARMS + REFERENCES:
        require(len(seeds[seeds.arm == arm]) == 3, "missing complete seed result: " + arm)
    require(not seeds.duplicated(["arm", "seed"]).any(), "duplicate seed result")
    summary = seeds.groupby("arm")[METRICS].mean()
    seed_delta = seed_contrasts(seeds)
    decision = decide(summary, paired, seed_delta)
    smoke = read(P / "SMOKE_DIAGNOSTIC_COUNTS.json")
    require(smoke["diagnostic_short_fits"] == 4 and smoke["optimizer_steps"] == 8,
            "actual completed short-fit accounting changed; update explicitly before report")
    require(done["diagnostic_short_fits"] == smoke["diagnostic_short_fits"], "scheduler short-fit receipt differs")
    require(smoke["no_checkpoints_saved"], "unexpected persisted smoke checkpoints")
    counts = {"new_formal_inner_fits": 30, "new_formal_outer_fits": 30, "new_formal_model_fits": 60,
              "diagnostic_short_fits": smoke["diagnostic_short_fits"],
              "diagnostic_short_optimizer_steps": smoke["optimizer_steps"],
              "total_new_neural_fits_including_diagnostics": 60 + smoke["diagnostic_short_fits"],
              "historical_primary_inner_references": 30, "historical_primary_outer_references": 30,
              "historical_A3_outer_references": 15, "historical_CONCH32_outer_references": 15,
              "evaluation_reference_count": 90, "historical_frozen_readout_probes_completed": 60,
              "superseded_probe_records_in_incomplete_batch": 4, "phase18b_fits": 0,
              "completed_anchor_audit_re_solves": 195,
              "anchor_audit_re_solve_scope": "75 historical replays + 60 completed historical probes + 60 new replays; excludes training/short-fit/superseded underlying solver calls",
              "anchor_re_solves_produce_new_neural_weights_or_new_OOF": False,
              "historical_references_and_frozen_probes_are_not_new_neural_training": True}
    decision.update(source=locked_source, parent_commit=config["parent_commit"],
                    generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), counts=counts,
                    all_primary_95_intervals_cover_zero=bool(((paired.loc[paired.family == "primary", "CI95_low"] <= 0) &
                                                             (paired.loc[paired.family == "primary", "CI95_high"] >= 0)).all()),
                    evidence_gates={key: value["status"] for key, value in gates.items()})
    seed_delta.to_csv(R / "SEED_PRIMARY_CONTRASTS.csv", index=False)
    matched = matched_epoch_table(curves)
    matched.to_csv(R / "MATCHED_EPOCH_GRADIENTS.csv", index=False)
    gpu = telemetry(); gpu.to_csv(R / "GPU_PROFILE.csv", index=False)
    primary = paired[paired.family == "primary"]
    same_epoch_chosen = curves[(curves.role == "inner") & (curves.epoch == curves.selected_epoch) & curves.arm.isin(ARMS)]
    selected_columns = ["arm", "role", "seed", "fold", "selected_epoch", "epoch", "train_loss",
                        "validation_loss", "validation_loss_EMA", "validation_Harrell", "g_attention_grad",
                        "delta_grad", "B_condition_grad", "write_grad", "readout_grad", "gamma_grad", "gamma"]
    require(len(same_epoch_chosen) == 60, "selected epoch training rows incomplete")
    same_epoch_chosen[selected_columns].to_csv(R / "SELECTED_EPOCH_DIAGNOSTICS.csv", index=False)
    inner_compute = compute[(compute.role == "inner") & compute.arm.isin(ARMS)]
    budget = inner_compute.groupby("arm").agg(cells=("fold", "count"),
                selected_epoch_median=("selected_epoch", "median"), selected_epoch_min=("selected_epoch", "min"),
                selected_epoch_max=("selected_epoch", "max"), selected_epoch1_cells=("selected_epoch", lambda x: int((x == 1).sum())),
                epochs_ran_median=("epochs_ran", "median"),
                hit160=("epochs_ran", lambda x: int((x == config["parent_training"]["max_epochs"]).sum()))).reset_index()
    # Two targets share scalar checkpoint diagnostics: de-duplicate before aggregation.
    scalars = mechanism.drop_duplicates(["arm", "role", "seed", "fold"])
    main = scalars[scalars.arm.isin(ARMS)]
    scalar_columns = ["z_matched_wrong_relative_L2", "h_relative_L2", "matched_wrong_risk_MAE",
                      "wrong_context_relative_risk_drift", "patient_z_variance", "patient_z_mean_channel_variance",
                      "patient_SSM_risk_SD", "patient_anchor_SD", "patient_SSM_over_anchor_SD", "gamma",
                      "isolated_SSM_g_gradient_L2", "isolated_SSM_RNA_gradient_L2"]
    require(all(column in main for column in scalar_columns), "missing patient-level selected-checkpoint diagnostics")
    require(main[scalar_columns].notna().all().all(), "incomplete primary-arm mechanism diagnostics")
    probes = mechanism[mechanism.arm.isin(ARMS) & mechanism.position_probe_target.isin(
                        ["terminal_state_squared_norm", "final_patient_risk"])].copy()
    require(len(probes) == 240, "position probes must include two targets for all 120 primary checkpoints")
    require((probes.position_probe_patients == 1).all(), "unexpected gradient probe population")
    probe_summary = probes.groupby(["role", "arm", "position_probe_target"]).agg(
        checkpoints=("fold", "count"), WSI_probe_instances=("position_probe_WSIs", "sum"),
        tail10_mass_median=("position_probe_tail10_gradient_L2_mass", "median"),
        first10_mass_median=("position_probe_first10_gradient_L2_mass", "median"),
        position_entropy_median=("position_probe_normalized_position_entropy_mean", "median")).reset_index()
    coverage = pd.DataFrame(gates["PROBE_COVERAGE_AUDIT.json"]["groups"])
    require(len(coverage) == 8 and not coverage.duplicated(["arm", "role"]).any(), "probe coverage group mismatch")
    require((coverage.checkpoint_count == 15).all() and
            (coverage.gradient_probe_patients_per_checkpoint == 1).all() and
            (coverage.variance_probe_patients_per_checkpoint == 32).all(), "probe selection definition changed")
    coverage_columns = ["arm", "role", "checkpoint_count", "gradient_distinct_patient_count",
                        "gradient_distinct_WSI_count", "gradient_WSIs_per_checkpoint_min",
                        "gradient_WSIs_per_checkpoint_max", "variance_distinct_patient_count",
                        "variance_distinct_WSI_count", "variance_WSIs_per_checkpoint_min",
                        "variance_WSIs_per_checkpoint_max"]
    resource = compute.groupby(["fit_classification", "role", "arm"]).agg(
        cells=("fold", "count"), selected_epoch_median=("selected_epoch", "median"),
        cell_seconds_median=("seconds", "median"), cell_seconds_sum=("seconds", "sum"),
        peak_GPU_MiB_max=("peak_GPU_MB", "max"), parameters=("registered_parameters", "max"),
        last_step_gradient_parameters_median=("last_step_gradient_parameters", "median")).reset_index()
    pfi = pd.read_csv(R / "PFI_SEED_METRICS.csv").groupby("arm")[METRICS].mean()
    pfi_folds = pd.read_csv(R / "PFI_FOLD_METRICS.csv")
    pfi_one = pfi_folds[(pfi_folds.arm == "L0") & (pfi_folds.seed == config["seeds"][0])]
    pfi_n, pfi_e = int(pfi_one.patients.sum()), int(pfi_one.events.sum())
    checkpoints = gates["CHECKPOINT_REPLAY.json"]
    strong = pd.DataFrame(decision["strong_reference_checks"])
    body = "# Phase18A — Local State Readout × Patient Conditioning\n\n## 判定与适用范围\n\n"
    body += f"**{decision['status']}**。本轮严格限定为读出 × 条件机制的探索性 2×2；常规逐位置 SSM 读出不是原创算法。\n\n"
    body += f"训练来源 `{locked_source}`，父提交 `{config['parent_commit']}`。正式新增 **30 inner + 30 outer = 60 次神经网络拟合**；标签突变诊断另有 **4 次真实短拟合、8 次神经网络 optimizer steps**。历史60个主要 inner/outer、15个A3 outer及15个CONCH32 outer，共90次引用，历史神经网络重训0次。\n\n"
    body += "锚点审计另计：父Data在完整合格的75次历史回放、60次历史冻结probe与60次新checkpoint回放中，确定性重解同一Clinical+S32 Ridge anchor共195次，并强制与原checkpoint逐元素/bitwise一致。它们不更新历史神经网络、不生成新OOF或模型选择；不能写成“审计完全没有优化”。195仅覆盖该完整合格验证链，不是包含正式/短拟合内部及superseded流程的全部底层solver调用总数。\n\n"
    body += "## DSS 主结果\n\n" + table(summary.loc[ARMS + REFERENCES].reset_index())
    body += "\n原512人、65个DSS events；每seed先汇总五折同折可比较对分子/分母，再平均3seed C值。不是跨折平均风险，也不是将15个cell称为seed mean。历史CONCH32为不同视觉/训练协议的强参照，不能据其数值作同输入因果消融。\n\n"
    body += "## 四个预锁定主要比较\n\n" + table(primary[["comparison", "metric", "delta", "CI95_low", "CI95_high", "Bonferroni_low", "Bonferroni_high"]])
    body += "\n20,000次固定折内患者配对bootstrap，所有模型和seed共享患者计数，pair权重为count_i×count_j，Uno60使用对应outer-train的反向KM。DID每次重算四个完整C值后求差。主要家族8项，校正区间覆盖率99.375%；普通95%区间同时披露。这些区间条件于已训练模型，不包含完整重训练或反复研究选择方差。未检出差异不等于统计等效、无信息或否定所有读出机制。\n\n"
    ba = gates["BOOTSTRAP_AUDIT.json"]
    body += f"实际有效draws={ba['valid']}/{ba['draws']}，因零分母排除={ba['zero_denominator_draws']}；抽样及分母审计见BOOTSTRAP_AUDIT.json。\n\n"
    body += "### 逐 seed 方向与工程门\n\n" + table(seed_delta)
    body += "\n" + table(pd.DataFrame(decision["primary_checks"]))
    body += "\nL1 相对预锁定强参照的均值门（Clinical+S32锚点为输入子集，A3为同输入神经模型；历史CONCH32单列，不当同输入参照）：\n\n" + table(strong)
    body += "\n" + ("\n".join("- " + x for x in decision["reasons"]) if decision["reasons"] else "所有预锁定方向、seed和强参照工程门通过；仍非独立外部确认。") + "\n\n"
    secondary = paired[(paired.family == "secondary") & paired.comparison.isin(
        ["L0-ANCHOR_CS", "L1-ANCHOR_CS", "L0-R1_A3_noDirect", "L1-R1_A3_noDirect",
         "L1-HIST_CS_S32_CONCH32", "L1-L1__WRONG", "F1-F1__WRONG"])]
    body += "### 预列强参照与正确/错误条件比较\n\n" + table(secondary[["comparison", "metric", "delta", "CI95_low", "CI95_high"]])
    body += "\n该表是次要比较，完整次要家族及其校正区间见PAIRED_COMPARISONS.csv，不挤占或混淆主要八项家族。WRONG只换状态条件中的RNA/CNA，自身late证据不变；冻结推理不是重训消融。\n\n"
    body += "## 状态与患者风险的读出诊断\n\n所选checkpoint第一批32名训练患者、无标签/错误样本筛选。以下分别按inner/outer列出15个checkpoint的中位数；标量诊断已按checkpoint去重，未把两个位置目标重复当成两个模型。\n\n"
    for role in ["inner", "outer"]:
        body += "### " + role + "：所选 checkpoint\n\n" + table(main[main.role == role].groupby("arm")[scalar_columns].median().reset_index()) + "\n"
    body += "`patient_z_*`先在同一患者内等权聚合各WSI，再统计患者间方差；旧`ssm_over_anchor_SD`等字段沿用WSI粒度，只作历史兼容，不冒充患者级值。z尺度/方差变化也会影响优化，不能将读出改善自动归因于纯信息损失。gamma与readout可缩放耦合，不凭gamma大小推断分支是否被忽略。\n\n"
    body += "`isolated_SSM_*`以SSM风险为目标，隔离late与anchor且不使用生存标签；是真实路径敏感性探针，不是Efron训练损失的梯度。正确/错配状态差或梯度更大不构成推进门。\n\n"
    body += "### 两种明确区分的位置梯度目标\n\n" + table(probe_summary)
    body += "\n每checkpoint一个按原顺序选取的训练患者，纳入其全部WSI，最终风险为WSI等权患者风险；表中WSI_probe_instances是重复探针实例总数，不能称独立患者数量。terminal_state_squared_norm针对h_T的平方范数，final_patient_risk针对实际患者风险。位置仅为坐标几何顺序；末10%梯度质量不是末10%组织的因果生存贡献。各checkpoint的探针分布保存在READOUT_MECHANISM.csv，不能只报告一个代表病例。\n\n"
    body += "真实探针去重覆盖（由每个checkpoint的fit[0]/fit[:32]及原owner映射独立重建，不公开患者/WSI标识）：\n\n" + table(coverage[coverage_columns])
    body += "\n15个checkpoint不等于15名独立梯度探针患者：seeds/folds可能重复同一病例。上表distinct数是在各arm×role内取并集，不应跨组再相加当独立人数；32人方差探针同样存在重复。位置梯度表描述这些固定训练样本上的模型探针，不是队列总体梯度分布或生存风险依赖比例。\n\n"
    body += "## 优化预算与真实梯度\n\n" + table(budget)
    body += "\n继承160上限、min_epoch1、EMA alpha0.2、patience40；未按外折成绩扩大预算。所选epoch可以为1，这可能使观察受初始尺度/正则及短训练影响，不能称纯信息恢复因果证据。hit160单元仍有预算截断风险；零触顶也不证明全局优化收敛。全部逐epoch训练/内层验证损失、Harrell和梯度保存在TRAINING_CURVES.csv，选中行另存SELECTED_EPOCH_DIAGNOSTICS.csv。fresh outer严格训练内层所选轮数。\n\n"
    body += f"本轮实际汇总：触顶={int(budget.hit160.sum())}/{int(budget.cells.sum())}，选epoch1={int(budget.selected_epoch1_cells.sum())}/{int(budget.cells.sum())}。实际是否发生截断或极早选择以此计数为准，不能将协议允许的情况写成本轮已发生。\n\n"
    chosen_cols = ["g_attention_grad", "delta_grad", "B_condition_grad", "write_grad", "readout_grad", "gamma_grad", "gamma"]
    body += "所选内层epoch的真实Efron梯度中位数（选中训练长度可能不同）：\n\n" + table(same_epoch_chosen.groupby("arm")[chosen_cols].median().reset_index())
    body += "\n同epoch、同seed/fold的真实Efron梯度比较：\n\n" + table(matched[(matched.group == "g_attention_grad") & matched.epoch.isin([1, 5, 10, 20, 40])])
    body += "\n完整多梯度组见MATCHED_EPOCH_GRADIENTS.csv。参照梯度为0时比值NA；不能把零初始化或无条件臂的预期零路径当错误。所选checkpoint隔离梯度、所选epoch训练梯度、同epoch比较是不同对象，不能用不同训练长度证明唯一的梯度竞争原因。\n\n"
    body += "## 算力与并行\n\n" + table(resource)
    body += "\npeak_GPU_MB沿用旧字段名，实际单位MiB（bytes/2^20）。父历史训练时间只作历史描述，不计本轮训练成本。并行下cell_seconds_sum不是墙钟时间，也不是隔离速度排名。局部/终态读出参数相同，但计算路径不同，不宣称FLOPs完全一致。\n\n" + table(gpu)
    body += "\nGPU_PROFILE以调度器10秒级采样相邻时间区间作左值保持加权，只覆盖正式inner/outer队列的采样跨度；整卡显存/利用率可能包含并发进程，不等于单模型显存或完整端到端时长。未为占显存制造无效计算。\n\n"
    body += f"## PFI：冻结 DSS 风险迁移\n\n可用{pfi_n}人、{pfi_e}个事件；未以PFI训练或选择本轮配置。\n\n" + table(pfi.loc[ARMS + REFERENCES].reset_index())
    body += "\nPFI不是独立外部队列；其终点与DSS不同，不能用有利PFI结果改写预锁定DSS决策。\n\n"
    body += "## 实现、资格及审计证据\n\n" + table(pd.DataFrame([{"gate": key, "status": value["status"]} for key, value in gates.items()]))
    body += f"\n27项算子/模型测试加5项独立评估/公开隐私回归测试，共32项通过；CPU/CUDA逐位置输出、输入/参数梯度oracle通过，N=1、padding、tiny dt、非零h0、gate=0等已覆盖。全部{checkpoints['new_replayed']}新checkpoint与{checkpoints['historical_replayed']}历史checkpoint四条件重放最大误差={checkpoints['maximum_error']:.6g}。独立anchor重建、stationarity及跨arm/seed逐元素一致性通过；指标独立核对sksurv，历史逐seed和三seed均值与已发表父结果核对。主比较点估计及DID独立审计PASS，共享bootstrap的每折患者计数SHA已保存；公开位置探针已剔除单患者raw target_value。每项检查只证明各自边界，不构成生物机制确认。\n\n"
    body += "外折标签突变检查本次合格对的参数、inner风险、所选epoch和完整训练历史一致；审计过程中先前两次已完成的短拟合因bool比较helper报错而追加重做，累计四次如实记录。未保存smoke检查点或患者风险。首个历史probe批次完成4条记录后中止，以补充患者/WSI定义，旧不完整批次被标记superseded；后续完整60个冻结probe补齐。它们不更新神经网络参数，不算神经网络训练失败或新拟合；Data中的anchor验证重解已单列。详见FAILURE_LOG.md。\n\n"
    body += "## 优势、限制与结论解释\n\n优势：只改读出公式的2×2、无新增参数；历史来源与软件资格核验、完整checkpoint重放、患者级完整风险集Efron及独立指标/配对DID；患者风险与状态目标、患者/WSI粒度及新拟合/引用分别披露。\n\n"
    body += "风险：65个DSS事件及反复开发的单队列；inner epoch选择噪声、潜在预算截断；读出尺度与优化条件同时变化；空间顺序无天然医学时间；patch只做geometry-only对齐，无原图形态核验。外部资源目前仍是资格清单，不能提升本轮证据等级。\n\n"
    body += f"**最终执行判断：{decision['status']}。** 置信度：算子及本轮数值/计数正确性高；将观测归结为唯一生物学原因的置信度低。本轮未证明RNA无条件剩余信号，也未证明所有Mamba或读出方法无效。完成Phase18A后停止；不自动追加Phase18B、归一化、调幅或第四种读出。\n\n"
    def delta_pair(name):
        values = primary[primary.comparison == name].set_index("metric")
        return f"Harrell {values.loc['Harrell','delta']:+.6f} / Uno60 {values.loc['Uno60','delta']:+.6f}"
    inner_medians = main[main.role == "inner"].groupby("arm")[scalar_columns].median()
    outer_medians = main[main.role == "outer"].groupby("arm")[scalar_columns].median()
    selected_budget = budget.set_index("arm")
    body += "### 读出瓶颈问题的直接回答\n\n"
    body += f"1. **读出作用**：无条件 L0−F0 为{delta_pair('L0-F0')}，有条件 L1−F1 为{delta_pair('L1-F1')}。按预锁定规则，读出方向一致门={'通过' if decision['readout_helpful_direction_gate'] else '未通过'}；这只界定本次公式与训练族的工程推进，不能推广为所有局部读出无效/有效。\n\n"
    body += f"2. **条件机制作用**：局部读出下 L1−L0 为{delta_pair('L1-L0')}，DID 为{delta_pair(DID)}。方向/seed门={'通过' if decision['condition_direction_and_seed_gate'] else '未通过'}，强参照均值门={'通过' if decision['strong_reference_no_mean_harm'] else '未通过'}；应同时看八项区间，不能以单个有利差值代替机制证据。\n\n"
    body += f"3. **条件是否到达实际读出**：inner所选checkpoint的正确/错分子条件 z 相对差中位数，F1={inner_medians.loc['F1','z_matched_wrong_relative_L2']:.6g}、L1={inner_medians.loc['L1','z_matched_wrong_relative_L2']:.6g}；终态 h_T 相对差则为{inner_medians.loc['F1','h_relative_L2']:.6g}与{inner_medians.loc['L1','h_relative_L2']:.6g}。这是不同重训模型的冻结敏感性，不是把同一checkpoint替换读出后的纯因果信息损失实验。\n\n"
    body += f"4. **优化/尺度替代解释**：F1/L1所选inner epoch中位数分别{selected_budget.loc['F1','selected_epoch_median']:.6g}/{selected_budget.loc['L1','selected_epoch_median']:.6g}，选epoch1的cell分别{int(selected_budget.loc['F1','selected_epoch1_cells'])}/{int(selected_budget.loc['L1','selected_epoch1_cells'])}；outer训练探针患者级SSM/anchor风险SD比中位数为{outer_medians.loc['F1','patient_SSM_over_anchor_SD']:.6g}/{outer_medians.loc['L1','patient_SSM_over_anchor_SD']:.6g}。训练长度与尺度变化必须同读出结果一起解释；本轮没有据外折结果追加归一化或调幅。\n\n"
    outer_risk_probes = probes[(probes.role == "outer") & (probes.position_probe_target == "final_patient_risk")]
    tail = outer_risk_probes.groupby("arm").position_probe_tail10_gradient_L2_mass.median()
    outer_coverage = coverage[(coverage.arm == "L1") & (coverage.role == "outer")].iloc[0]
    body += f"5. **空间梯度扩散不等于预测改善**：冻结outer模型的实际患者风险目标，末10%位置梯度质量中位数从F1的{tail.loc['F1']:.6g}变为L1的{tail.loc['L1']:.6g}。这些探针仅覆盖{int(outer_coverage.gradient_distinct_patient_count)}名不同训练患者、{int(outer_coverage.gradient_distinct_WSI_count)}张不同WSI，不能泛化为队列比例或医学组织重要性。此实现确实改变了探针上的梯度位置分配，但未满足方向一致的预测收益门；因此不能据此把终态遗忘认定为已解决的唯一瓶颈。\n\n"
    body += "**拓展思考／你可能想知道：** 若读出改善而条件增量未建立，允许保留局部读出作为普通对照；这与提出新的患者特异性交互机制是两个独立的证据问题。\n"
    text(R / "RESULTS.md", body)
    text(R / "DECISION.md", "# Phase18A 决策\n\n**" + decision["status"] + "**\n\n" +
         table(pd.DataFrame(decision["primary_checks"])) + "\n" + table(strong) + "\n" +
         ("\n".join("- " + x for x in decision["reasons"]) if decision["reasons"] else "方向性工程门通过，不等于统计确认。") +
         "\n\n校正区间支持的主要统计量：" + str(decision["corrected_positive_primary_statistics"]) +
         "\n\n局部读出是已有SSM输出的聚合改动，不作为原创算法。区间条件于已拟合模型；无显著差异不是无信息或等效。完成本轮后停止，Phase18B不自动启动。完整结果、尺度/优化限制和外部验证边界见RESULTS.md。")
    text(R / "FAILURE_LOG.md", """# Phase18A 审计修复与失败记录

## 1. 外折标签突变比较 helper

第一次审计已真实完成两个2-epoch短拟合，比较state_dict时对bool buffer作减法引发异常。修复仅对比较helper：浮点tensor比较绝对差，bool/整型buffer比较torch.equal。未改变正式训练source、模型、协议或数据。完整重做两次短拟合后PASS。生产累计4个真实短拟合、8次optimizer steps；均未保存smoke checkpoints/患者风险，正式60拟合不受影响。

## 2. 历史诊断的患者/WSI粒度补全

最早历史冻结probe在完成4条记录后中止，以补充患者等权WSI的z/风险方差和真正final_patient_risk目标的定义。旧不完整产物更名为superseded，不用作完成证据。后续按固定新diagnostics SHA对60个F0/F1 inner/outer checkpoints完成冻结无标签probe。未更新神经网络参数或选择epoch；不计新神经网络拟合或训练失败。完整诊断与DSS评估彼此分开。父Data仍会确定性重解同一Ridge anchor并校验一致，不能写成没有任何优化调用；完整合格回放/探针链195次anchor验证重解另列于结果和计数。

## 正式训练及范围

最终报告要求TRAINING_DONE、全部60新模型和历史来源/重放、独立指标、算子、标签隔离审计通过。上述两项是审计流程修复，不是模型训练失败，也不能隐去已完成短拟合以缩减真实运行计数。没有自动追加旧R3或Phase18B，没有按外折结果更改模型/预算。未来如有失败必须追加，不能改写历史。

## 3. 发布格式修复

发布前 `git diff --check` 检出报告复核JSON的CRLF与非训练评估脚本的额外EOF空白，分别规范化为LF及单一文件结尾。已锁定训练source中的 `src/phase18a_readout/data.py` 额外EOF空行原样保留，以保护既有训练SHA；发布器只对该文件这一明确EOF提示作显式豁免，其余空白检查照常。报告重生成并更新复核摘要；所有指标、训练文件及模型权重不变，没有重训。
""")
    write(R / "decision.json", decision)
    write(R / "EXECUTION_COUNTS.json", counts)
    print("PHASE18A_REPORT_COMPLETE", decision["status"], counts, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    if args.protocol_only:
        locked_source = source()
        require(read(R / "PROTOCOL_LOCK.json")["source"] == locked_source, "source lock mismatch")
        protocol_documents(cfg(), locked_source)
        print("PHASE18A_PROTOCOL_ONLY_NO_RESULTS", flush=True)
    else:
        report()


if __name__ == "__main__":
    main()

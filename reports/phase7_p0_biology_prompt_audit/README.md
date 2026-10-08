# Phase7-P0 正式结果与复现入口

先读 [CONCLUSIONS_ZH.md](CONCLUSIONS_ZH.md)，覆盖与机制补充见 [MECHANISM_SUMMARY_ZH.md](MECHANISM_SUMMARY_ZH.md)。

## 结果范围

512 位患者；T/N/M/Stage；3 个 inner-split seed × 5 个固定 outer folds；10 个方法。540 个 outer 模型、1620 个 inner 模型均收敛，另有 60 个复用 B2 权重的错配控制，共 600 份正式预测、120 行 seed-level OOF 指标。没有读取 DSS，没有训练 TITAN/token prompts。

代码中的 Bridge 是保留静态病理和分子加性路径的有界概念交互可行性审计，不能代表完整可学习 molecular prompt 架构。Clinical Text 为可选补充，本轮暂缓。

## 复现步骤

在原服务器及 hash 匹配的数据/运行环境中：

```bash
cd /root/diffusion_phase5_20261007
export PHASE7_OUT=/root/phase7_p0_biology_audit_v4
bash scripts/run_phase7_p0_biology_prompt_audit.sh
bash scripts/finish_phase7_p0_biology_prompt_audit.sh
/root/phase6a_titan_runtime/env/bin/python scripts/supplement_phase7_p0_audit.py --out "$PHASE7_OUT"
```

补充脚本只报告已有结果，不拟合或调参。运行脚本对已完成目录拒绝重跑；独立验证可重复执行。迁移服务器时须先映射代码常量中的输入根目录，再使用新的输出目录和新的 source lock。

输入 SHA、源码 SHA、固定基因集、概念顺序和匹配规则见 `manifest.json`；完整定义见 `protocol.json` 和 `configs/phase7_p0_molecular_morphology_prior.json`。固定 folds 与 RNA/CNA 及 TITAN cache 必须匹配这些 SHA。

## 文件含义

- `metrics.csv`：每个 seed 各自拼五折的真正 OOF 指标。
- `aggregate_metrics.csv`：三个 OOF seed 的均值/SD，不是 15 cells 的 seed mean。
- `paired_comparisons.csv`：同 seed 差值；按患者共用重采样、平均三个 seed 差值的 98.75% 条件 bootstrap 区间。
- `encoder_audit.json`、`coverage_audit.json`、`effective_gene_set_coverage.csv`：编码、患者对齐和有效成员覆盖。
- `budget_audit.json`：2160 次拟合预算、C 选择次数及搜索边界。
- `independent_validation.json`：源/输入 SHA、120 个预处理分区、600 份 OOF 和 60 个 counterfactual 的独立校验。

## 数据边界

仓库只保留固定先验、代码、汇总指标和有信息量的报告。原始 RNA/CNA、患者 ID、fold 内预处理参数、模型系数及 OOF probabilities 仅保存在服务器 `/root/phase7_p0_biology_audit_v4`，未推送。未保留或发布 smoke/exploratory 结果。

旧 v3 在不完整运行期间因 Top512 与错配控制错误作废，没有使用其性能选择本轮协议。v4 在有效 OOF 评估之前锁定，独立验证完成后才汇总解释。

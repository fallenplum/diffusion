## 机制判断

研究门槛状态：**V0_NOT_CONFIRMED**。
- `A3-A0` 加性主效应：未通过。
- `A4-A3` 低秩 S×O 交互：未通过。
- `A5-A4` Clinical residualization：未通过。

这组结果只支持当前冻结表征、低容量网络和统一 Cox 协议下的内部机制判断，不是外部确认或最终 SOTA 排名。Phase M2R2、M3R2、M8-P1R、Phase7A–7D 的历史数字未混入本表。

## 审计状态

- 完成 cell：105/105。
- `A0` outer-test Clinical anchor 在同 seed/fold 被全部 residual arms 数值复用。
- 训练/测试 offset、PCA/scaler、残差投影和 epoch 选择均记录在 cell JSON。
- 失败 cell、NaN、无事件 inner-val、外层测试参与选择都会使 validate 失败。

## 文件

- `source_audit.json`、`data_registry.json`、`baseline_registry.csv`
- `architecture_audit.json`、`leakage_audit.json`、`clinical_anchor_identity.json`、`gradient_audit.json`、`independent_validation.json`
- `fold_metrics.csv`、`seed_metrics.csv`、`paired_comparisons.csv`、`ablation_summary.json`、`compute_profile.csv`

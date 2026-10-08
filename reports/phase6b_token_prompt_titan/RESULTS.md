# Phase6B token-level TITAN prompt audit

**状态：v2 mask 修复正在运行，正式结果尚未完成。**

v1 已在计算层面完成，但 S1/S2 的 CLS padding mask 和模板归一化顺序
存在实现问题。原始汇总保留于 `invalidated_v1/`，不能用于正式机制结论。

v2 保留相同的固定 3 epoch schedule 和 S0–S4 对照，复用有效的 S0/S3，
只重跑 S1/S2。已通过无 prompt 官方输出一致性、prompt mask 移动、
prompt 非零梯度和冻结 TITAN 零参数梯度门禁。

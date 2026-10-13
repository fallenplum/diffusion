# 计算图审计

ScalarConceptTokenizer、RNA、CP/RP QKV、Patch projection、head都有预期非零有限梯度，至少2seed×2batch合成验证，最终真实checkpoint扰动验收。CP两方向及B0/B1/B2/B3/B4真实query shape hook通过。B6/A5/A6/D3无RNA预测；D3 auxiliary仅RNA模块接收其监督梯度。

注册、grad非None、非零梯度元素分别报告，不能称所有arm有效容量严格相同。C2增加活跃gate；B2额外C→RNA模块活跃；无RNA臂有inactive RNA参数。D3 main loss不能更新RNA，其额外监督仅诊断，且dropout RNG消费与A6不同，不能视作纯容量因果比较。head*.1是尺度，不是严格risk bound。

默认CP真正S32查询；A5 unguided learned queries；B5/D1同模型重编号复用且不能当两项独立机制证据。梯度范数不等于模态因果贡献。

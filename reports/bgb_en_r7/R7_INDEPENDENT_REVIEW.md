# R7 独立匿名算术复核

状态：**PASS_ANONYMOUS_ARITHMETIC_AND_HASH_BINDINGS**。固定 GATE1 判定：`CURRENT_GATE1_NOT_SUPPORTED`。

核对13模型×9单元=117行；每seed先合并3分区的Harrell/Uno分子分母，再平均3seeds，未拼接患者风险。
11项比较×9直接配对记录分别按Harrell/Uno分母加权；逐分区和逐seed核验repair−damage=指标差。None保留为无定义并给出INCONCLUSIVE，未替换成0。
33份诊断分片的字节数、SHA、模型身份和4个诊断epoch齐全。次要终点：`PASS_117_FROZEN_DSS_RISK_PFI_ROWS`。

预算99拟合/7920更新、18条零拟合引用及297份actual-checkpoint记录来自上游PASS收据，本脚本验证其匿名metadata绑定；**本脚本没有重新执行297次模型回放**。合成测试fixture也不构成正式拟合或实际回放证据。

优势是独立算术、逐分区直接credit和原字节绑定；局限是不能仅凭匿名统计重新确认权重的数值前向、患者无重叠或生物学真值。没有新训练、数组/权重/标签读取、风险重建、CI/p值或模型选择。

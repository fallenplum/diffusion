# Phase25-A：C4 冻结 RNA 查询与局部检索归因

## 范围与数值资格

**15个原始 C4 检查点全部通过 heldout 原风险重放，最大绝对误差0.0。**容差在运行前固定为 `1e-5`，没有根据结果调阈值。模型权重、Clinical/S32 anchor、CP分支、Patch K/V、RNA本人 Query 残差及风险 head 均未更新。

这是对尚未进行过的 **RNA→Patch 注意力 q-only / Value-only** 干预的单次审计，不重复原来的 RNA 全路径均值/错配试验。新增训练、bootstrap和患者级公开数组均为0。纯合成资格先验证同公式输出/attention逐元素相同、mask不变及 Value-off 残差公式相同，再执行15模型GPU审计。

## 干预公式

原 RNA→Patch 分支为 `h = RT_i + A(Q(RT_i),K(P_i))V(P_i)`，随后 `h + FF(h)`。

|条件|唯一变化|
|---|---|
|MATCHED|保留原计算，用于重放旧OOF|
|VALUE_OFF|令 `h=RT_i`，再用原 `h+FF(h)`；保留患者RNA残差，删除检索Value消息|
|QUERY_MEAN|只将attention q输入换为outer-train患者编码后RT逐槽均值；残差仍为 `RT_i`|
|QUERY_WRONG|只将attention q输入换为另一个heldout患者RT；残差仍为 `RT_i`|

RT均值为当前冻结RNA编码器输出的 `8×64` 向量，先按患者等权、逐槽求均值，不使用heldout统计。错配按癌种×完整raw RNA缺测模式在当前heldout分区内零自配对，采用预先固定seed公式；没有按DSS、Stage、Age或Sex匹配。缺测模式只是平台代理，不是真实平台标签。

## 结果

先逐seed汇总五折 within-fold 指标分子/分母，再对三个seed均值；从未平均风险后计算 C-index。

|固定模型条件|Harrell|Uno60|相对MATCHED ΔHarrell|相对MATCHED ΔUno60|
|---|---:|---:|---:|---:|
|MATCHED|0.813691|0.802965|—|—|
|VALUE_OFF|0.819572|0.795359|+0.005881|−0.007606|
|QUERY_MEAN|0.810162|0.784256|−0.003529|−0.018709|
|QUERY_WRONG|0.806477|0.777534|−0.007214|−0.025431|

VALUE_OFF的Harrell在3/3 seed提高、Uno60在3/3下降。QUERY_MEAN/QUERY_WRONG的Uno60均3/3下降；Harrell各2/3下降。没有新增配对置信区间，这些是固定风险的描述性方向，不能表述为统计确认。

### 配对修复与损伤信用质量

方向为“干预−MATCHED”；删除后修复表示该原风险发生了有利改排，不代表训练时应删除该分支。

|干预|指标|修复质量|损伤质量|净变化|
|---|---|---:|---:|---:|
|VALUE_OFF|Harrell|0.022348|0.016467|+0.005881|
|VALUE_OFF|Uno60|0.020383|0.027989|−0.007606|
|QUERY_MEAN|Harrell|0.024386|0.027915|−0.003529|
|QUERY_MEAN|Uno60|0.021429|0.040138|−0.018709|
|QUERY_WRONG|Harrell|0.026190|0.033404|−0.007214|
|QUERY_WRONG|Uno60|0.026466|0.051897|−0.025431|

质量是在同折可比患者对上计算的tie半信用/Uno加权信用，不是患者人数，也不包含重新训练方差。

## 解释与边界

- **患者特异性RNA attention q确实改变有用的排序关系。**在本人RNA Query残差保持不变时，仅替换q便降低两项总体点估计，特别是Uno60。这比此前RNA全路径干预更直接定位到检索的查询部分。
- **局部Value分支没有展现双指标一致必要性。**删除Value后Harrell提高、Uno60下降，说明它同时修复和损伤不同排序关系；不能概括为“检索完全无用”或“检索已确认有效”。
- q-only错配能产生下降，不证明RNA信息纯主效应已经被完全排除：Query改变的是与Patch K的兼容关系，但没有独立形态学真值或医学因果解释。
- 当前没有同一RNA表示下的纯加性模型真实重训对照；冻结干预仍不能估计替代模型重新训练后的表现，也不能独立确认跨模态交互超过低容量主效应。
- 未依据这些outer风险删除PC、选择新监督患者对或调参；后续B的设计必须在读取其自身heldout结果前锁定。

完整匿名单元、seed及配对统计见 `C4_ROUTING_ATTRIBUTION.json`；脚本为 `scripts/audit_phase25_a_c4_routing.py`，运行时替换方法，不修改任何历史model源码或检查点。

# Phase21 独立统计复核

## 范围和证据

本复核只读已完成的匿名表及服务器审计记录，没有训练、重新选模、修改风险、重算 bootstrap 或修改冻结源码。

- 训练/评估来源：`d30dcdc413d57d25aeb041b6ea5ac83835580cad117530ea699d33f35940aaef`。
- 评估脚本 SHA256：`8b74d0696f33b7c3c3825eee7fe6f5e53deb5c72d4721709500ea19171648c90`，本地和服务器一致。
- 逐数复核：`SUMMARY.csv`、`PAIRED_COMPARISONS.csv`、`REPAIR_DAMAGE.csv`、`SEED_REPAIR_DAMAGE.csv`。
- 服务器实际只读核验：`BOOTSTRAP_AUDIT.json`、`PAIR_CREDIT_IDENTITY_AUDIT.json`、`CHECKPOINT_REPLAY.json`、`EPOCH_LOCK_AUDIT.json`，均为上述来源的 PASS。
- `INDEPENDENT_VALIDATION.json`：90 个合格 checkpoint，540 份预测回放逐元素一致；30 个新 DIRECT 和60个合格原控制；48份完整模型—seed OOF。15份新 outer 的 epoch、inner PT SHA、inner receipt 与锁记录一致。

## 四个主检验

主检验对象恰为两个预设模型差乘两个指标，共4项。点估计是在每个 seed 内累计五折分子/分母后计算 C，再平均三个 seed 的差；没有平均风险后计算 C。

| 比较 | 指标 | 差值 | 95% CI | 家族4项校正区间 |
| --- | --- | ---: | --- | --- |
| DIRECT−P16 | Harrell | −0.004155885 | [−0.008872926, −0.000477668] | [−0.010494881, 0.000398540] |
| DIRECT−P16 | Uno60 | −0.003308209 | [−0.007253969, −0.000703175] | [−0.008834212, −0.000161070] |
| DIRECT−S16 | Harrell | −0.001097781 | [−0.005387015, 0.003245146] | [−0.006883565, 0.004598692] |
| DIRECT−S16 | Uno60 | −0.004607907 | [−0.015037757, 0.001703019] | [−0.019730624, 0.002931657] |

因此，DIRECT−P16 的 Uno60 在预设家族校正后仍为负；Harrell 普通区间为负、校正区间跨零。不能将两者写成“双指标均在校正后显著劣于”。DIRECT−S16 两指标的区间均跨零。其余16行比较属于诊断，未冒充新增主检验。

## 修复、损伤与 tie

复核采用已经锁定的风险差 tie 容差 `1e−8` 和0/0.5/1 credit。20份模型比较—指标点表与60份 seed 表均满足：

`repair − damage = net = C1 − C0`。

从落盘 CSV 独立检查，分子/分母、三乘三转移和点估计闭合的最大绝对误差为 `1.11e−15`；summary 的模型差与 paired point 差的最大误差为 `1.64e−16`。服务器评估直接以内存完整 seed C 校验的最大误差为 `5.90e−17`。这是 CSV 读写精度范围，未修改数值。

所有实际 point/seed 表的 risk-tie 转移质量均为零；本轮严格错→对/对→错计数可以直接解释为 Harrell repair/damage 的质量。实现仍支持 tie 半 credit，不能将本轮无 tie 推广为数据普遍无 tie。

### DIRECT matched−同 checkpoint OFF

每个 seed 的 within-fold 可比对数均为4251。

| seed | Harrell 严格修复对 | Harrell 严格损伤对 | Harrell 净 credit | Uno60 修复质量 | Uno60 损伤质量 | Uno60 净 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 0 | 0 | 0 | 0 | 0 | 0 |
| 47 | 2 | 2 | 0 | 3.198226200 | 2.428797422 | +0.000068873 |
| 83 | 4 | 3 | 1 | 9.144231781 | 9.420880443 | −0.000024763 |

平均 Harrell 净增量为 `1/(4251×3)=0.0000784129`。Uno60 采用冻结 outer-train IPCW，质量不是患者对个数，因此不能把4比3的 Harrell 净修复直接当作 Uno60 必须提高的证据。seed83恰是 Harrell 增加而 Uno60 减少。

matched−OFF 的 Harrell 95% CI 为 `[−0.000770634, 0.001007826]`，Uno60 为 `[−0.000925925, 0.000778832]`。非零修复/损伤没有形成稳定净收益。

### 净零不等于所有排序都没变

- S16 matched−OFF、matched−wrongC、matched−wrongO 的逐 seed repair、damage、net 均为0。这里支持“已审计可比对的 credit 没有变化”；不支持“原始风险/注意力/梯度全为零”。
- DIRECT matched−wrongC 的 Harrell 净变化每个 seed 都为0，但修复/损伤分别为 `0/0`、`1/1`、`2/2`。这是局部修复与损伤抵消，不能写成“错误 Clinical 不改变任何排序”。
- DIRECT matched−wrongO 的平均 Harrell/Uno60 点差约 `+0.000156826/+0.000156695`，诊断区间均跨零；没有确立患者特异性增益。

## Bootstrap 统计对象

实际审计为20,000次请求、20,000次有效 draw、0次零分母。固定五折患者数为103、103、102、102、102；每折生成一个患者 multinomial count 矩阵，10个比较乘3个算法 seed 的30组向量共用同一矩阵及其 SHA。pair权重为 `count_i×count_j`，只使用折内可比对；不独立重采样 pair、不重采样算法 seed、不进行风险 ensemble。

Uno60 权重由对应 outer-train 反向KM冻结，仅 source event owner 使用 IPCW，不在 bootstrap 中重拟合。四项主检验使用0.05/4的预设尾概率；诊断区间未宣称同时校正。区间只覆盖固定已拟合模型的患者抽样不确定性，不包含重新训练、选择或该队列历次研究决策的不确定性。

## 机制解释的边界

`DIRECT_DIAGNOSTICS.csv` 的15个 outer checkpoint 中，heldout 患者层面诊断先在各 checkpoint 内聚合，再取 checkpoint 均值：P→O attention TV约 `0.000962641`，O→P约 `0.002161030`；患者修正风险 SD约 `0.003091557`。这些量非零，不支持“更新完全消失”。它们是 checkpoint 均值，不能写成15份独立患者数据或全体患者混池 SD。

same-checkpoint OFF 比较保留训练时已经共同优化的 backbone，测试的是冻结模型内关闭修正的效果。DIRECT−P16/S16 比较包含独立重新训练后的 backbone 差异，不能当作同一冻结 backbone 上算子的纯局部因果效应。J0已包含强多模态主效应；本轮没有重做独立 RNA 信号检验。

**结论：未发现配对数学或 bootstrap 对象错误。本轮 direct 归一化改动产生了非零、很小的排序修复与损伤，但没有实现稳定净增量；相对 P16 的 Uno60 在预设校正后降低。这支持停止扩大当前算子分支，不能外推为所有跨模态机制或分子信息无效。**

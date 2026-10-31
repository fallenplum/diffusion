# P2 同输入、容量与特征审计

来源为既有真实资产绑定和 v3 的六次真实 GPU 零更新 profile；没有读取外折风险。

512 名患者、65 个 DSS events；五折人数 103/103/102/102/102，事件数 13/14/12/13/13；seed 为 31、47、83。当前 canonical RNA/CNA 列数为 20511/25101；RNA 缺失列直方图为 309 人无缺失、203 人缺失 3015 列，不能把缺失当成零表达。

共享下游读出为 P8＋G8＋C8＋机制2，共26个系数；前三组均仅当前训练分区拟合 PCA。G 保留50个程序的位置后展平，不采用低秩均值噪声白化。

| scope | 模型 | 活跃参数 | forward 矩阵 FLOPs | forward+backward 矩阵 FLOPs | 峰值显存 bytes |
|---|---|---:|---:|---:|---:|
| inner | B4_DUAL | 310770 | 30041177424 | 58680317341 | 2149122560 |
| inner | B2_MATCHED | 310770 | 30911790128 | 61467304669 | 2630226944 |
| inner | B2_NATIVE | 343538 | 39297404464 | 94320543261 | 2211610624 |
| outer_final | B4_DUAL | 310834 | 45439840996 | 88951590102 | 3379105792 |
| outer_final | B2_MATCHED | 310834 | 47405339167 | 95248605483 | 4450857984 |
| outer_final | B2_NATIVE | 343602 | 65640286831 | 168211922589 | 3293761536 |

同一 scope 下 MATCHED 与 B4 参数完全相同，forward 和含重算的 forward+backward 支持矩阵 FLOPs 差异均在预锁10%范围内。MATCHED 为1头×8维、不做 activation checkpoint；NATIVE 为4头×64维、独立QKVO且做重算。两者都是8个无语义图像读取位置×50程序的普通注意力适配，不是官方论文原生复现。

优势：依据实际执行、非零梯度与保存状态回放校验预算。风险：这些 FLOPs 不覆盖所有算子；单次带 profiler 的时间不构成稳定速度基准，显存和墙钟时间也没有宣称匹配。不同训练 scope 临床编码宽度可变，因此参数数目不能跨 scope 直接当作容量不一致。


## 2026-10-31 正式完成后的实际状态补充

本文件上文保留编写当时的预运行事实，不把当时NOT_RUN追溯改写为PASS。当前480个正式源神经拟合、185个外折风险与实际独立核验均已完成；无DSS两次50更新的held-out协变量扰动、另106个Cox的完整DSS管线标签扰动均真实通过。详见 `00_FINAL_PROTOCOL_AND_QUALIFICATION_STATUS.json`、两份`03_LIVE_*`及`10_INDEPENDENT_REPLAY_AUDIT.json`。

完整曲线仍显示Phase-A预算上界活跃，故最终研究裁决为 `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET`，具体是训练充分性未闭合；固定预算下B5未见DSS增量这一负向事实不变。最终解释见 `11_RESEARCH_DECISION.md` / `FINAL_RESULTS.md`，未计算CI或新增架构训练。原生成器42报告及其index按原字节保留。

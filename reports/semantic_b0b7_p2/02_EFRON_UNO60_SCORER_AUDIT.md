# P2 Efron、Harrell 与 Uno60 资格

已有真实服务器隔离环境执行 scikit-survival 0.23.1 的100个合成案例、200次真实 API 调用。Harrell 最大绝对差为0，Uno60为3.3306690738754696e-16；包括风险近似 ties 和同刻 event/censor。v2 收据为 `aad1ed8a1c5a22602e0908e41e0f014bfd51c70eab0b26cfef55460eb2249ee5`，仅修正了日期/依赖表述，没有把同一批调用重复计数。

风险越高代表更早事件风险；同刻 event/censor 可比较，同刻两个 event 不比较，风险差≤1e-8记半分。Uno 使用 fit-train reverse-KM；同刻观察事件先离开，再发生删失跳变，取该时刻右连续 G(t)，仅事件时间严格小于60月的 anchor 具有权重。G60/事件支持不足按锁设 NA，不用 epsilon 裁剪，不在 test 拟合 KM。

完整训练风险集的 Efron＋ridge 使用独立 NumPy 数学参考与有限差分软件检查。汇总先在每 seed 内加总五折 numerator/denominator，再等权平均三个 seed，不能跨折比较风险或把15 cells叫作独立 seed 均值。B0只有五个确定性重拟合。

优势：已用真实官方 API 检验评分公式。限制：API 输入是合成案例，尚不是185个正式外测风险的独立验证；该验收等待正式训练与报告全部完成后的授权。隔离包只验证所用 metrics API，未声称完整估计器依赖生态已安装齐全；不计算 CI 或 p 值。


## 2026-10-31 正式完成后的实际状态补充

本文件上文保留编写当时的预运行事实，不把当时NOT_RUN追溯改写为PASS。当前480个正式源神经拟合、185个外折风险与实际独立核验均已完成；无DSS两次50更新的held-out协变量扰动、另106个Cox的完整DSS管线标签扰动均真实通过。详见 `00_FINAL_PROTOCOL_AND_QUALIFICATION_STATUS.json`、两份`03_LIVE_*`及`10_INDEPENDENT_REPLAY_AUDIT.json`。

完整曲线仍显示Phase-A预算上界活跃，故最终研究裁决为 `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET`，具体是训练充分性未闭合；固定预算下B5未见DSS增量这一负向事实不变。最终解释见 `11_RESEARCH_DECISION.md` / `FINAL_RESULTS.md`，未计算CI或新增架构训练。原生成器42报告及其index按原字节保留。

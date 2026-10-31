# P2 无DSS预训练、背景交叉拟合与教师风险

已核对冻结入口：Phase-A 以 include_endpoints=False 读协变量，raw builder 与 run_cell 均拒绝带结局的输入；临床背景为同一冻结编码坐标内3个OOF＋1个完整训练 Ridge，每个子拟合从原始六字段重做临床编码。不能把不同编码器的 latent 直接拼成 OOF。

r0 对整个 CP_SCALAR 管线再次交叉拟合，包含受监督的标量 β：每个分区4套管线，每套13个 Cox＋1个 scalar，共52＋4，避免患者自身标签参与其训练侧 offset 的 β。真实 E0 在 outer1_inner0 的272/137人范围已完成52＋4；Cox最大梯度 1.73414577273e-06，scalar最大梯度 2.38795481652e-10，anchor拟合 11.714625 秒，0次 outer 指标。

E0 是单独锁的真实 profile，顶层 source/lock 与正式矩阵不完全相同，因此不冒称已免费复用为正式 teacher；其52＋4成本需单列。既有21个软件测试全部通过，但其中的标签扰动与教师隔离是合成反例，不是21套真实 DSS 实验。

真正 held-out 置换资格已另行预锁：outer1 fit409/eval103、MAIN dual、seed31、50步×2次。从 raw 层仅置换 eval RNA/CNA/clinical，要求训练 tensors 与 inputSHA 不变。真实 DSS 只在隔离对象中置换，并测试带结局输入会被原入口拒绝；不向 Phase-A 添加 DSS 参数。两次 checkpoint 要求完整解码的参数、Adam、CPU/CUDA RNG、所有其他字段逐字节精确；容器原文件 SHA 另报，不强迫计时或分配器 receipt 字节相同。

当前状态：该2-fit资格尚未运行，必须等480个正式拟合结束、GPU空闲、收到授权后执行。额外100 updates 独立计入软件资格，不能混进 primary metrics、架构搜索或临时选择。优势：能直接发现当前 Phase-A 对 held-out 数据的实现依赖。限制：即使通过，也不能代替整个 DSS 训练管线的真实标签扰动测试或证明预后增益；不计算 CI/p 值。


## 2026-10-31 正式完成后的实际状态补充

本文件上文保留编写当时的预运行事实，不把当时NOT_RUN追溯改写为PASS。当前480个正式源神经拟合、185个外折风险与实际独立核验均已完成；无DSS两次50更新的held-out协变量扰动、另106个Cox的完整DSS管线标签扰动均真实通过。详见 `00_FINAL_PROTOCOL_AND_QUALIFICATION_STATUS.json`、两份`03_LIVE_*`及`10_INDEPENDENT_REPLAY_AUDIT.json`。

完整曲线仍显示Phase-A预算上界活跃，故最终研究裁决为 `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET`，具体是训练充分性未闭合；固定预算下B5未见DSS增量这一负向事实不变。最终解释见 `11_RESEARCH_DECISION.md` / `FINAL_RESULTS.md`，未计算CI或新增架构训练。原生成器42报告及其index按原字节保留。

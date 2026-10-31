# P2 父提交与协议锁

科学主锁 SHA256：`b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c`，20 份科学源码字节已核对。本报告没有读取任何新 outer 风险。

锁定时 HEAD 为 `98f42e298975c39e9e3bde70f2250b2334caca8b`；P1R 资格内容已以 `d74060162761be2dacb251e4dbcfea05df7640a0ab43e854bd5ec2f70823d60c` 绑定。实际发布父提交为 `d2568ad4d57bb9c83366029cacee6baea6c3522d`，对应 manifest `2e914a8326614843520e9fbae4cd6b5b6a13608ebcc9eb0dd6518d5efd16d598`。

P1R Git 发布发生在 P2 内层启动后、外折选择和评分前；原始主锁保留当时 HEAD，后续父提交绑定另存，没有为追认 Git 时间线改写科学锁或训练配置。五个 outer 的所有 epoch/λ 必须先冻结，再产生第一个 outer 成绩。

优势：来源内容与后续发布提交均可逐字节追踪。限制：该队列已被历史研究多次开发，不能称独立外部验证。本文件不宣布正式矩阵完成或 DSS 增益；不计算 CI、bootstrap 或 p 值。


## 2026-10-31 正式完成后的实际状态补充

本文件上文保留编写当时的预运行事实，不把当时NOT_RUN追溯改写为PASS。当前480个正式源神经拟合、185个外折风险与实际独立核验均已完成；无DSS两次50更新的held-out协变量扰动、另106个Cox的完整DSS管线标签扰动均真实通过。详见 `00_FINAL_PROTOCOL_AND_QUALIFICATION_STATUS.json`、两份`03_LIVE_*`及`10_INDEPENDENT_REPLAY_AUDIT.json`。

完整曲线仍显示Phase-A预算上界活跃，故最终研究裁决为 `INCONCLUSIVE_DUE_TO_IMPLEMENTATION_OR_BUDGET`，具体是训练充分性未闭合；固定预算下B5未见DSS增量这一负向事实不变。最终解释见 `11_RESEARCH_DECISION.md` / `FINAL_RESULTS.md`，未计算CI或新增架构训练。原生成器42报告及其index按原字节保留。

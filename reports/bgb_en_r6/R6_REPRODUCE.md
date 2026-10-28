# R6 执行与复核说明

## 固定范围

本阶段从 `ecd273d2036d98e40dd9e997dd45e8de9eba7947` 独立开展，不继承 R5 的表现最优读出配方。R5 门槛为 `R5_READOUT_STOP_R6_ELIGIBLE`，实际 108 个检查点回放通过后才创建 R6 源训练锁。原始数据、患者级预测、系数、坐标和权重均在仓库外。

源任务包含 147 个逻辑单元：144 个留一供体单元及 3 个完整来源迁移模型。其中 12 个 S0 是零优化引用，正式优化为 135 个模型、每个 80 次完整目标更新。真实 GPU 计算资格另有一个优化器实例、2 次非正式更新，未保存其权重，不计入正式矩阵。

目标原型为既有 outer1 的训练侧 3 个划分 × 3 seeds × 10 个神经模型，共 90 次拟合；另有 18 个 CP/CP+RNA 固定风险引用，均不新增教师拟合。主检查点固定第 80 轮，20/40 轮仅作预算诊断，不据已见验证分数选 epoch。

## 数据与源训练

先运行 `scripts/bgb_en_r6_qualify.py`，由授权的原始 ST 开发文件核对组合键、真实坐标、基因 counts 和程序定义，再生成独立私有缓存。缓存加载器要求 manifest 及加载源码 SHA 一致。4 张 ST 测试切片不进入本轮。

运行 `scripts/bgb_en_r6_source_eligibility.py`，只依据真实目标的样本量与方差预先确定组织内相关的可计算组。不得按模型预测情况更换支持组。

`scripts/bgb_en_r6_lock_source.py --help` 列出完整参数。该工具检查 R5 的 decision、协议和实际回放清单，并绑定数据资格证书与源码，产生 `configs/bgb_en/r6_source_v1.json` 和私有启动门槛。已存在的锁不可覆盖。

```powershell
python scripts/bgb_en_r6_compute_qualify.py --help
python scripts/bgb_en_r6_schedule_source.py --help
python scripts/bgb_en_r6_report_source.py --help
```

依次执行真实计算资格、固定源矩阵、逐单元实际检查点回放和匿名源报告。源并发上限为 8。调度器遇错停止排队，不自动重试；保留原始失败收据。

完整迁移模型预先固定为 S7、seed31 的 MATCHED、WRONG_DONOR、WITHIN_SECTION_TISSUE_DEPTH。三者只在全部开发来源上训练，不进入留一供体性能均值；不得从留一供体成绩事后挑一个供体模型迁移。

## 目标资格与正式矩阵

在三份完整源模型及实际回放都存在后，执行 `scripts/bgb_en_r6_qualify_target.py`。它创建独立 `targetLOCK.json`、资格专用的冻结病理缓存，复核 39 组既有 Cox 系数及 3 个标量系数，并对所有目标臂进行真实前向一致性检查；该阶段不做优化器更新。

所有目标命令共用六个必需路径参数：`--assets`、`--private`、`--primary-private`、`--source-private`、`--config`、`--source-config`。私有根必须位于仓库外，首次资格还需已绑定 SHA 的 `--launch-gate` 与 `--launch-gate-sha256`。

```powershell
python scripts/bgb_en_r6_qualify_target.py --help
python scripts/bgb_en_r6_schedule_target.py --help
python scripts/bgb_en_r6_replay.py --help
python scripts/bgb_en_r6_report_target.py --help
```

调度器最多 4 路，先串行建立 18 个引用，再执行 54 个主要拟合及 36 个必需控制，最后实际回放全部训练/验证风险与冻结干预，再生成目标报告。source 训练性能门槛影响医学声称和外阶段资格，不用于删减已经数据合格的必需比较。

每个神经模型每轮都使用完整患者风险集的 Efron Cox。分批前向的风险梯度缓存不是小批 Cox；输入与反向重放须一致。CUBLAS workspace 固定 `:4096:8`，严格确定性算法开启，TF32 关闭，dropout 为零。正式运行禁止修改已锁科学源码、放宽回放阈值或按表现追加 seed。

## RNA 信息保真

`scripts/bgb_en_r6_retention.py` 使用独立配置及私有根，按 `--mode lock`、`fit`、`replay`、`report` 顺序执行。逻辑上限 18 个表示×划分单元；sum/mean 只有全部标准化数值及 mask 逐元素完全相同时才复用一次凸拟合。系数、风险、独立 Efron 目标及梯度均需回放。

旧 RNA512、fullRNA、RNA512-PCA256 依精确 SHA 引用，不重新拟合。旧 fullRNA 使用 λ10，其余新表示固定 λ1；这些坐标上的 L2 并不自动等价于相同 gene-L2。该审计检验可读出性，不能单独证明生存条件增量。

## 统计与解释边界

Harrell 和 Uno60 分别按每个 seed 累加划分内分子/分母，再平均 seeds；不拼不同划分的风险排序。PFI 只评价固定 DSS 模型，不训练或选择 PFI 模型。不计算 bootstrap、置信区间或 p 值。固定风险配对同时报告修复及损伤，不将冻结干预当作真实重训消融。

六个 ST 开发供体已被历史研究复用，本轮留一供体不构成新的外部确认。原研究与 TCGA 的患者登记缺共同交叉表，患者互斥仍未证实。组织概率不是细胞比例，FAP/MARCO 相关程序不是细胞来源真值。原始基因重读说明计算路径回到原始测量，不等于证明真实基因调控或病理—基因位置对应。

## 优势与风险

优势：源对应任务与 DSS 分开、冻结强主效应、原基因测量 mask 保留、真实重训控制及逐检查点回放。

风险：来源少、同队列反复开发、ST 与 bulk 的测量域不同、目标空间坐标仅核验原生单位而非微米、随机 gate 仍保留置信度信息。固定预算结果不能称为数学收敛。

## 拓展思考

只有源任务和患者级条件增量分别获得支持，才讨论可迁移的医学交互机制；二者不能相互替代。

## 你可能想知道

所有本地私有文件路径由运行者提供；仓库只交付显式白名单中的源码、配置和匿名汇总。`scripts/bgb_en_r6_publication.py` 在提交前校验完整回放矩阵，并扫描真实标识符拒绝名单。

单文件发布上限仍为 2 MiB。若完整匿名 JSON 因缩进超过上限，先用 `scripts/bgb_en_r6_compact_reports.py --root . --private-backup PRIVATE_BACKUP_DIRECTORY` 只删除 JSON 空白，逐项验证解析值完全不变，并在仓库外保存原始版本及新旧字节哈希；不删除字段、不放宽扫描器上限。该步骤改变文本序列化，不改变模型、统计量或科学结论。


## 完成后的只读复核

独立重算脚本分别为 `bgb_en_r6_independent_source_review.py`、`bgb_en_r6_independent_retention_review.py`、`bgb_en_r6_independent_target_review.py`。它们检验已保存匿名分子/分母或回放导出的数值，不创建新的模型或统计选择。

深度相关使用 `bgb_en_r6_source_depth_audit.py`；患者亚组使用 `bgb_en_r6_subgroups.py`。RNA 缺失 pattern 只是平台代理，不是已验证测序平台。亚组只对预定内层患者风险评价，不能把全 512 人协变量计数误称为完整外折结果。

风险诊断先用 `bgb_en_r6_risk_diagnostics.py --mode export` 从完整私有收据导出匿名标量，再以 `--mode analyze` 汇总。私有 whitelist 不发布。第 20/40 次更新未保存的分支梯度和 SD 标记 NOT_SAVED，不根据总损失反推。所列纯 Cox 路径梯度是固定首八名训练患者对完整风险集目标的贡献，不是全队列梯度；参数 L2 梯度使用完整参数，二者不能直接相减分解。

`bgb_en_r6_attention_diagnostics.py` 对 90 个最终检查点各做 5 次无优化前向，检验原查询、均值查询、错查询、assignment shuffle 和关闭交互。三分区各固定首八名验证患者，共 24 名不同患者，不能把重复模型前向计为独立样本。其完整匿名标量表以索引和十个固定 arm 分片交付，载入函数逐一验证字节 SHA 与逻辑 SHA。

```powershell
python scripts/bgb_en_r6_execution_ledger.py --source-private SOURCE_PRIVATE --target-private TARGET_PRIVATE --retention-private RETENTION_PRIVATE --reports reports/bgb_en_r6 --output reports/bgb_en_r6
python scripts/bgb_en_r6_plot_diagnostics.py --report-dir reports/bgb_en_r6 --output reports/bgb_en_r6/figures
python scripts/bgb_en_r6_publication.py --root . --denylist PRIVATE_DENYLIST
```

总账保持 source、target、retention 三种任务各自的配方 SHA，并核对共同源模型、cohort/endpoints、prepared receipt 和启动门槛。首版报告器错误地要求 retention 与 target 的不同配方 SHA 相同；其失败脚本和收据已在私有目录保留。只修报告器后重算总账，不改科学锁、不新增训练。

图表为七张固定的匿名统计诊断，每图同时输出 PNG/SVG，绑定生成器及输入报告字节 SHA，不含患者级曲线或组织学热图。数据表变化后应重新生成全部绑定，不能手改图上的数值。图表聚合不计算 CI 或 p 值。

## 最终实际预算与停止范围

源正式 135 个模型 ×80 更新；目标正式 90 个模型 ×80 更新；另有源资格计算 2 次非正式更新。源 12 个、目标 18 个及 retention 9 个旧模型引用不计为新拟合。retention 18 个凸模型共 609 次 LBFGS 迭代。所有正式神经检查点与凸模型均完成实际回放。执行完整并不意味着有限 80 次更新达到数学收敛。

B10、B12、B14 在负门槛后不启动；B15 因本线 UNI2 图文对齐资格未建立而 TEXT_BLOCKED。外层保持 NOT_RUN，不能把内层开发结果作为独立外层性能。当前预测几乎完全来自主效应增量，交互数值塌缩；这是本配方的失败诊断，不是形态—分子关系不存在或 DSS 事件数是唯一原因的证据。

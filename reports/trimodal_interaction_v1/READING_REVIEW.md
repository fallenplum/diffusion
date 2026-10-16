# 机制报告精读与本轮实施边界

日期：2026-10-16。用户原报告保留原样；本文追加阅读审查及与最新修复证据的协调。本文**没有本轮预填成绩**。

## 一、已核验与修正

1. 保留原报告首选 A：Clinical 条件下的 RNA-to-patch 四参考图差分；不合并 copula、slots、专家、Mamba 或新的辅助 loss。代数上可检验，不等于医学机制已成立。
2. SlotSPE 官方代码现在可取得，revision `02051a2083add7b427727e0200e4263516903561`。本次读到动态 slots、可微 top-k、模态内及 WSI→omics 重建；`eval()` 仍含采样，需要记录推理 RNG。原报告“可运行代码未确认”更新为“官方核心与入口已取得，原生全流程未运行”。[作者代码](https://github.com/zylvemvet/SlotSPE/tree/02051a2083add7b427727e0200e4263516903561)
3. MDCS 完整公开 `main→engine→loss/optimizer` 未找到独立 router 更新；argmax().item() 截断任务及特征匹配损失梯度。隔离原 AST 探针验证 8 个 router 参数梯度均为 None；专家使用 CPU stub，因此不冒充完整原生 benchmark。正式 CVF 页本次 403，全文未成功取得。[版本与证据](SOURCE_REVISIONS.json)、[隔离探针](MDCS_ROUTER_ISOLATED_PROBE.json)
4. 官方Diff普通4头对应2个差分头，headwise RMSNorm带可学习affine。实施审查发现原A3使用affine=False，属于未完整遵循官方norm的缺陷适配；原40/160结果保留且标记，不作最终合法norm对照。A3F仅修复affine=True，以独立实现source按160上限重训9单元，最终A5−A3F替代原A5−A3，不在二者间择优。两者仍是cross-attention/Cox适配，**算量未匹配**，不能称严格官方原生复现。[固定官方代码](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/Diff-Transformer/multihead_diffattn.py)

完整来源、Q/K/V、softmax 轴、loss 与适配差异见 [文献矩阵](../../docs/literature_mechanism_matrix.md)；引用组件/工程适配/待证实贡献见 [创新边界](../../docs/novelty_ledger.md)。

## 二、与 Phase17R 最新证据协调

以下数字全部是**历史已完成 Phase17R**，来源 [原报告](../phase17r_conditional_repair/RESULTS.md)，不属于本轮结果：

- 删除高维 Direct 后，Harrell 点估计差为 +0.009958、Uno60 为 +0.023991，配对区间均跨零。不能认定删除旁路已经确认提高性能。
- 所选 checkpoint 梯度差混有 epoch 1→17 的训练长度差异；同 epoch 的 g-attention 梯度增强约 1.1–1.9 倍。它不足以识别唯一的梯度竞争原因。
- 条件 S6 的调制幅度提高后，正确/错患者状态及预测差仍弱，未通过 R3 门，历史结论是停止该结构支线，而非推翻 RNA 独立信号或所有多模态机制。

因此本轮“保留 raw RNA”意为 **所有臂共同拥有、共同正则的信息保留主效应对照**，不是把高维自由 OmicsDirect 重新加入后默认有效。本轮采用同一 C+S32+LOCAL32+RNA512 L2=1 凸初始化及共同训练，既检验主效应表现，也检验额外交互是否被学习；不能将旁路本身的提升归因于新核。SSM/扫描分支不在本轮续堆。

## 三、首轮实际范围

| 项目 | 锁定实施范围 |
|---|---|
| 任务与分区 | DSS；既有外折 1/3/5 的 inner-train/inner-val × seeds 31/47/83；仅内层探索，不产生完整外折 OOF |
| 病理 | 既有冻结单视觉主干；从原 512 个几何匹配 patch 中均匀取 128；不是新的 RNA 选区或新 WSI 真值 |
| RNA | 512 gene 及 identity/missing 显式信息；不先压为 8 个程序标量；自身 RNA 主效应保持 |
| 共同主效应 | C+S32+LOCAL32+RNA512；训练分区内预处理及 L2=1 凸初始化后共同训练 |
| 容量/预算 | hidden64；普通4heads；原40epoch cap、至少8epoch、patience12、dropout0；观察内层触顶后独立追加160敏感性，另有A3F修复 |
| 模型 | A0–A7，另 F0 pre-attention FiLM、PR-attention+C-late；未训练原生 SlotSPE/MDCS 大矩阵 |
| A7 | 普通 A11−uniform 消息按 detached norm 缩放到同参 mixed 消息范数，再用同 gate；匹配不取标签 |
| 泄露控制 | inner-train 拟合参考、选择、scaler/PCA/初始化；无 teacher 或风险 stacking；outer-test 不用于选择 |
| 数量记录 | real_fit_count 与 evaluation_reference_count 分开；相同凸风险、重放和历史对照不冒充独立训练 |

该范围低于原报告建议的完整5-fold正式机制矩阵；其作用是先判断算子和条件化路径是否值得投入正式训练。历史完整外折结果不可与当前inner-val指标直接排名；本轮不计算CI或p值。

160阶段仅从相同初始化重训原40触顶的M个神经单元，其他完整early-stop单元引用原checkpoint，逐项核验前40轮历史。A3F另增加9个真实神经拟合及9次ridge初始化，使用原source+修复脚本SHA+配置SHA形成独立来源。两个追加均不是原先独立确认，原40/160及原A3不被覆盖。

预算门允许每臂至多20%单元触顶，**门通过不等于全部优化收敛**。即使仅A1仍有1/9触顶，也须明确披露，不写成全部单元已充分收敛。Harrell和Uno60若呈相反方向，不选择有利指标宣称整体增量；判断仍依据锁定主方向门及全部描述结果。

## 四、证据与停止边界

先检查参考零点、行和、固定 V 平移、padding、患者置换、hash/分区拒绝、预期梯度与初始化。通过后仍须检查 held-out 内层主效应、普通条件注意力、一般差分、gate-off、错配及收缩控制。单独 attention 熵、梯度非零或消息幅度增加不能替代预测收益。

优势：共用主效应与参考算子令归因可审计。风险：少事件、内层分区重叠、128 patch 覆盖不足、A3 计算预算未匹配，以及没有原图形态核验。当前对具体代码/代数性质置信度高；对新机制的预后有效性及生物学解释不作提前评级。

本轮若不能稳定优于强加性及直接机制控制，应保留负结果和主效应收益，停止扩大该实现；不能由此宣称整个三模态 idea 不成立，也不能通过追加基因/seed/backbone 搜索改写成功标准。

# Phase6B：完整训练预算与动态概念机制审计

**范围：512 位患者、3 seeds × 5 folds × 4 endpoints；240 个拟合 cell、300 份完整 OOF 文件。目标仅为 T/N/M/Stage，没有使用 DSS。**

## 核心结论

- 内部验证达到预置饱和条件：**240/240**；到 960 epoch 仍受上限约束：**0/240**。
- 内部训练损失达到预置数值平台：**50/240**。验证收益饱和与训练损失数值收敛是不同的判断。
- 每个 seed 先拼接五折 OOF，再计算指标；以下均是三个完整 seed OOF 的均值，没有把 15 个 fold cell 叫 seed_mean。

## 1. 泛化表现

| Endpoint | S0 | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| T | 0.3481 | 0.4197 | 0.3013 | 0.2935 | 0.3027 |
| N | 0.4325 | 0.4508 | 0.3889 | 0.4279 | 0.3537 |
| M | 0.5722 | 0.6214 | 0.5333 | 0.6222 | 0.5162 |
| Stage | 0.3219 | 0.3775 | 0.3198 | 0.3333 | 0.2757 |

指标为 balanced accuracy。S1 是普通静态 token tuning，S2 才包含患者 RNA 配对；S3 保留直接 RNA 加性通路，S4 使用相同 S2 参数，仅替换为错误患者 RNA。

## 2. 训练预算是否充分

每个 outer-train 内固定一次分层 80/20 split，RNA 插补/Top512/标准化与所有可训练参数仅在 inner-train 拟合。最多五个 epoch 的 trailing mean 验证 CE 选 epoch；epoch 1–4 同样可选。至少观察 30 epoch，内部验证 CE 与 BA 都连续 20 epoch 无实质改善才停止；未饱和依次扩展 120 → 240 → 960。

配置锁定后按所选 epoch，从同一 final seed 在完整 outer-train 重新训练，学习率、batch、优化器和模型容量不变。最早的 warmup selection 限制在任何本轮 OOF 指标汇总前被修正；未使用该版本的测试结果。

| Endpoint | Model | Selected epoch median (range) | Observed median | Val saturated /15 | Train numerical plateau /15 |
|---|---|---:|---:|---:|---:|
| M | S0 静态概念 | 54 (1–357) | 71 | 15 | 2 |
| M | S1 静态 token prompt | 19 (1–76) | 46 | 15 | 0 |
| M | S2 RNA 动态 token prompt | 1 (1–28) | 34 | 15 | 0 |
| M | S3 概念 + RNA 加性 | 1 (1–3) | 30 | 15 | 5 |
| N | S0 静态概念 | 87 (7–149) | 101 | 15 | 13 |
| N | S1 静态 token prompt | 21 (4–65) | 49 | 15 | 1 |
| N | S2 RNA 动态 token prompt | 3 (1–13) | 31 | 15 | 0 |
| N | S3 概念 + RNA 加性 | 1 (1–2) | 30 | 15 | 3 |
| Stage | S0 静态概念 | 86 (6–158) | 105 | 15 | 11 |
| Stage | S1 静态 token prompt | 35 (12–55) | 54 | 15 | 2 |
| Stage | S2 RNA 动态 token prompt | 3 (1–22) | 38 | 15 | 0 |
| Stage | S3 概念 + RNA 加性 | 1 (1–1) | 30 | 15 | 3 |
| T | S0 静态概念 | 95 (13–232) | 111 | 15 | 7 |
| T | S1 静态 token prompt | 31 (8–84) | 61 | 15 | 0 |
| T | S2 RNA 动态 token prompt | 5 (1–50) | 39 | 15 | 0 |
| T | S3 概念 + RNA 加性 | 1 (1–2) | 30 | 15 | 3 |

### 相对已修复的固定 3 epoch：同 seed OOF 配对

| Endpoint | Model | Delta BA | Positive seeds /3 | Every seed CI >0 |
|---|---|---:|---:|---:|
| T | S0 静态概念 | +0.0928 | 3 | True |
| T | S1 静态 token prompt | +0.0971 | 3 | False |
| T | S2 RNA 动态 token prompt | +0.0109 | 2 | False |
| T | S3 概念 + RNA 加性 | -0.0004 | 1 | False |
| T | S4 错误患者 RNA | +0.0128 | 2 | False |
| N | S0 静态概念 | +0.0715 | 3 | False |
| N | S1 静态 token prompt | +0.0328 | 3 | False |
| N | S2 RNA 动态 token prompt | -0.0103 | 1 | False |
| N | S3 概念 + RNA 加性 | +0.0314 | 3 | False |
| N | S4 错误患者 RNA | +0.0064 | 2 | False |
| M | S0 静态概念 | +0.0693 | 3 | False |
| M | S1 静态 token prompt | +0.0384 | 3 | False |
| M | S2 RNA 动态 token prompt | -0.0173 | 1 | False |
| M | S3 概念 + RNA 加性 | -0.0299 | 1 | False |
| M | S4 错误患者 RNA | -0.0022 | 2 | False |
| Stage | S0 静态概念 | +0.0449 | 3 | False |
| Stage | S1 静态 token prompt | +0.0360 | 3 | False |
| Stage | S2 RNA 动态 token prompt | +0.0235 | 3 | False |
| Stage | S3 概念 + RNA 加性 | +0.0046 | 1 | False |
| Stage | S4 错误患者 RNA | +0.0068 | 2 | False |

**判读规则：**所选 epoch >3 只能说明内部选择偏好更长训练；还需要 OOF 配对改善才能说旧预算限制了泛化。所选 epoch <3 则指向更早停止，而非继续加训练。这里同时改变了可用训练预算和 epoch selection，不能把全部差异归因于纯粹增加 epoch。

## 3. patient-specific 机制的必要控制

| Endpoint | S2−S1 | S2−S3 | S2−S4 | All three means >0 | All three: every seed CI >0 |
|---|---:|---:|---:|---|---|
| T | -0.1184 | +0.0078 | -0.0014 | False | False |
| N | -0.0620 | -0.0390 | +0.0351 | False | False |
| M | -0.0881 | -0.0889 | +0.0171 | False | False |
| Stage | -0.0577 | -0.0135 | +0.0441 | False | False |

- 三项均值控制同时通过的 endpoint：**无**。
- 三项控制且每个 seed 的区间下界均 >0：**无**。
- S2>S0 单独只能说明整个可训练 prompt pipeline 有收益；S2>S1 才排除普通 prompt tuning 的解释；S2>S3 才超出直接增加 RNA 的解释；S2>S4 才表明患者配对有用。配对有用仍不自动等价于医学概念语义正确。

**S4 的解释范围：**它复用已在匹配 RNA 上训练的 S2，只在测试时替换 RNA。一个主要利用 RNA 独立分类信息的模型也可能因错误患者 RNA 而变差，因此 S2>S4 是配对敏感性的证据，不能单独证明 pathology–RNA 的语义互补。

## 4. 结构层面的诊断

| Endpoint | Gate normalized entropy | Patient gate mean SD | Dynamic-to-static concept cosine | Score effective rank |
|---|---:|---:|---:|---:|
| T | 0.2339 | 0.3394 | 0.3734 | 1.6314 |
| N | 0.2700 | 0.3313 | 0.3593 | 1.5691 |
| M | 0.2974 | 0.3073 | 0.4669 | 1.8247 |
| Stage | 0.2640 | 0.3208 | 0.2798 | 1.3823 |

这些诊断不使用标签，限每个测试折的前 64 人。低患者 gate 变化提示弱条件化；低概念余弦提示明显语义漂移；它们均不是机制成立或医学语义正确的独立证明。

**结构推断的边界：**S2 的 RNA 只能经四个 prompt basis 的混合与 32 个 pathology cosine 传给分类器，S3 有直接 RNA 通路。如果 S2 持续弱于 S3，这与该中介通路限制信息传递相容；本轮没有专门隔离该瓶颈的消融，不能把它当作已经证明的因果原因。

源码中 RNA adapter 与 gate 都是线性层，中间没有激活：`alpha = softmax(W_gate W_adapter x + b)`。四个 mixture weights 只有三个独立自由度，因此 16D adapter 不等于向 text encoder 提供 16 个独立 context 维度。不过四分类线性模型也只需要三个独立 log-odds；这条数学观察本身不能证明当前信息瓶颈导致失败。

S3 的 RNA adapter 与最终 head 也都是线性层，合成后仍是 RNA 的仿射分类函数。S2 则包含 softmax gate 和冻结 text transformer 的非线性。因此，即使 S2 优于 S3，也仍需区分非线性 RNA 预测能力、跨模态交互与医学语义适配；本轮没有容量匹配的纯 RNA 非线性控制，不能单凭性能差异把三者等同。

### 与 CTF_2 原文的范围差异

原文 GCSP 联合使用 concept-specific global prompt、互补模态的 context prompt 和双模态 shared prompt，并在两个 VLM 分支对称运行。本轮按照已指定的低容量方案，只检验 RNA 驱动的 context prompt → 冻结 TITAN text encoder → pathology concept scores；没有原文全部 GCSP 组件、对称双 VLM 或 Clinical conditioner。这个范围差异由 [原论文 §3.2–3.3](https://openreview.net/pdf/4b3fab8d88280ab054b8f5acb353e783df7c6d2d.pdf) 与本轮源码直接核对。

因此，负结果表示当前受检配置没有展示出预期的性能优势，不能直接否定完整 CTF、医学先验或全部 tri-modal co-adaptation。反过来，这个范围差异也不能用来忽略当前配置的负结果。

原文 Table 5 的 context pool 为 16、gating MLP 含 ReLU，AdamW 初始学习率 0.001；当前方案是 pool 4、线性 adapter/gate、学习率 0.005。本轮固定这些设置，仅审计训练预算与内部 epoch selection，没有证明优化器或所有架构超参数已达到最优。

## 优势与风险

| 优势 | 劣势 / 风险 |
|---|---|
| 官方 TITAN 冻结编码及 prompt 梯度、mask、template aggregation 已校验 | cosine/gate 的数值有效不保证医学语义保持 |
| inner-only preprocessing / training / epoch selection 后重新训练 | 单个 inner split 的小类别验证噪声仍存在 |
| 完整五折 OOF、同 seed 配对与错误患者控制 | 三个 seeds 使用同一 cohort，不能当作三次独立人群验证 |
| 延长预算并区分 train plateau 与 validation saturation | 到上限未饱和的 cell 必须标记预算受限；有限训练不能证明全局最优 |

## 置信度与后续判断

- **高：**完整矩阵、OOF 指标重算、数据边界和 epoch selection 的计算审计结果；对应 `independent_validation.json`。
- **中：**当前 cohort、当前五个控制及固定超参数下的性能排序；跨 cohort 和其他架构尚未确认。
- **低：**将失败归因于某一个结构瓶颈、将成功解释为医学语义成立、或外推到 DSS。当前实验未直接检验这些命题。

## 拓展思考

优先根据 S2−S1/S3/S4 的完整证据判断是否值得保留动态概念中介，而不是只看 S2−S0。若仅静态 tuning 受益，应将它与 patient-specific co-adaptation 分开叙述。

## 你可能想知道

后续若要证明医学解释，仍需独立形态标注或外部队列验证；本轮没有看 DSS，也不能回答 DSS 监督事件是否足够。

## 审计与文件

- 独立重算最大指标误差：8.33e-17。
- TITAN 权重、私有 OOF 和可恢复 checkpoints 留在服务器；仓库只包含源码、protocol、无患者身份的 aggregate / learning curves / 审计报告。
- 98.75% paired bootstrap 区间为逐比较描述性区间，条件于已拟合的 pipelines，没有全家族错误率保证，也未完整包含模型重训的不确定性。
- 当前 cohort 已被历史研究查看，本轮是内部机制与预算审计，不是未触碰的外部确认。
- model selection 的偏差与方差风险参见 [Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)。

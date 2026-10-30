# P1-R｜语义机制修复与无 DSS 资格实验：Codex 执行 Prompt

## 你的角色与任务

你是本项目的科研工程负责人。直接在 GitHub 仓库 `fallenplum/diffusion` 最新 `research/semantic-complementarity-b0b7`（基准 commit `98f42e298975c3`）上实施 **P1-R：零生存标签的结构辨识与受控修复**。先核对远端 HEAD；若 HEAD 已前进，基于实际 HEAD 分支开展，记录父 SHA。新建工作分支 `research/semantic-b0b7-p1r`，只推源码、配置、匿名聚合结果及哈希审计；患者数据、权重、预测、日志中的 ID、凭据留私有服务器。执行真实训练与保存状态回放，不接受只有计划或 synthetic PASS 的交付。**本轮不得读 DSS/OS/PFI、生存时间或事件来做特征构建、预训练、选轮、模型选择。**

任务是定位上一轮 B4 真文本未超过 B3 随机查询的原因，明确区分：(i) Query 本身没有语义收益，(ii) 概念得分旁路掩盖 Query 差异，(iii) 已知关系被未知关系稀释，(iv) 8-slot→mean64 的信息损失，(v) `L_prog` 与 `L_pair` 竞争或模型泛化不足。**不能仅仅提高 P1 Pearson 就宣称 DSS 模型成功；也不能只调一次超参就否定或支持 Idea。**

## 0. 先读取并锁定的事实

逐一检查这些当前文件，不能凭提示覆盖仓库事实：

- `reports/semantic_b0b7/00_PROTOCOL_LOCK.md`, `LOCK.json`, `04_PRETEXT_PAIRED_RETENTION_RESULTS.md`, `10_DECISION.md`, `INDEPENDENT_P1_AUDIT.json`, `P1_SAVED_SCALE_ROUTE_AUDIT.json`, `03_FAIRNESS_CAPACITY_AND_GRADIENT_AUDIT.md`, `P0_HISTORICAL_BASELINE_BRIDGE.md`, `HISTORY_OPERATOR_AUDIT.md`。
- `src/semantic_b0b7/{models.py,pretext.py,data.py,survival.py}`；`scripts/semantic_b0b7_{p1,prepare,gate,report,checkpoint_diagnostics,verify_semantic_b0b7_p1}.py`（按实际存在的名称查找），`configs/semantic_b0b7/p0_p1_v1.json`, `allowed_relations_v1.json`。
- 原主方案 `B0_B7_语义条件互补三模态_严格实验方案_Codex.md`，历史 Phase6A/6B、Phase11/16/19、R0–R8 仅用于重叠审计，不重跑旧实验。

既有 P1：原 outer-fold 1 的409名训练患者，其中 inner-train272、inner-valid137；`B3/B4/B6a/B4_MAP_SHUFFLE × 31/47/83` 共12次神经拟合、4800 optimizer updates、48真实状态回放。数据为512名患者、65 DSS事件，但这些结局不得进入 P1-R。既有宏 Pearson：B3 `0.063007`，B4 `0.057733`，B6a `0.059187`，B4_MAP_SHUFFLE `0.057998`；冻结 PATHOLOGY Ridge `0.240092` / Huber `0.092206`，中位数 Huber `0.044910`。这些值仅用作**只读历史校验**，不得作为新目标硬编码。

特别审计源码事实：当前 B3 虽旋转 CONCH/TITAN 文本 Query，却仍收到 `titan_scores` 真实文本得分；B4 的未知关系总权重约87.2%–87.3%；`image_only()`将8槽平均后进入64维 `program_head`；模型先验/辅助目标的真正可用梯度可能很弱。训练200→400 epoch共12/12个单元验证辅助损失变差，不能再无条件延长至800。以上均是**待受控证伪的候选原因**，不是已确立因果结论。

冻结原始32概念、两模板、8组、Hallmark50、既有外折/患者身份和主预处理定义；未经预锁文献与输入资格审计，不许换词、基因集、癌种、患者或结局。

## 1. 执行门禁 R0：无训练源码与状态诊断

编写只读审计脚本，针对原 48 状态及实际私有 checkpoint/hash 完成如下分析，不能冒充新拟合：

1. **文本路径精确分解。** 记录 B3/B4 两个原型的 Gram、范数、查询投影、固定 cosine、`titan_scores` 与相应病理 aligned 特征的实际路由；证明 B3/B4 除查询基底外参数、有效操作、源数据、初始化（除旋转生成）一致。对 B3 实施 `titan_scores` 置空/同训练分区统计代替的只读前向，仅作为通路归因，不当成重训模型性能。
2. **关系稀释。** 对每个支持有效分子程序统计原始 known/unknown 边数、归一化前后质量、每个 slot 的注意力/实际贡献、`unknown_logit` 的时间轨迹以及映射置乱后分数变化；统计当前8×50中的真实已知关系覆盖。未知 ≠ 错误；不能声称15条历史候选关系是人工真值。
3. **表示保留与梯度。** 记录8槽各自的患者间方差、有效秩、slot相互相关性、mean64前后患者级分子预测线性可解码性。对 `L_prog` 和 `L_pair` 分别 backward，统计共享 `query_projection`,`query_residual`,`aligned_key`,`raw_value`,`native_projection`,`gene_encoder`,`u`,`v` 的梯度范数和夹角；绝不能以两个损失的绝对值比率推导梯度主导。报告 train/valid 同步学习曲线、风险之外的预测幅度收缩、各程序测量覆盖。
4. **无目标 RNA 泄漏。** 图像→程序预测函数接受 path+clinical，不接受同患者 gene target/omics；随机替换验证 RNA 时 image-only预测字节保持不变；辅助检索允许用合法验证 RNA 计算评价，但不参与 image-only infer。测试数据预处理/负例缓存绝不拟合到验证患者。
5. **留痕。** 创建 `reports/semantic_b0b7_p1r/00_PARENT_AND_GATES.md/json`、`01_SAVED_STATE_DIAGNOSTICS.md/json`，包括 source/checkpoint/input SHA、单测、缺失资产、未能核验内容。不能因为历史 JSON 写了 PASS 就跳过真实资产检查。

## 2. 预锁新的修复矩阵：每次只动一个机制

在训练任何新模型之前生成 `configs/semantic_b0b7/p1r_lock.json`，写明模型算子、输入、无 DSS 目标、训练种子、负例支持、学习率、有效容量、预算、评估标准、允许的阶段性裁决与新源 SHA。不得根据当前137人验证集既有分数事后优化概念或关系映射；所有以下结果为**开发实验**。

### T1｜彻底区分文本语义与视觉基础模型优势

- `T1a_B3_CONSERVATIVE`：保留原 B3 定义（随机旋转原型，但原 `titan_scores` 作为所有臂的共同输入）；`T1b_B4_TRUE`：仅固定真实原型，与T1a完全同层同参数/同监督。
- `T1c_TEXTLESS`: 接收**相同 CONCH raw、TITAN native、RNA/CNA、Clinical**，不使用固定病理文本 Query、概念 cosine 或 TITAN 的32个 concept scores。用参数数/参数参与率/读出维度相匹配的**无语义可学习或固定随机读取器**替代这些算子；可保留视觉基础模型本身的 frozen pretrained weights，但不得继续偷偷调用 text prototypes。对 `titan_scores` 分支设可训练同容量视觉 projection/随机基底输入，不能简单删除导致容量减少；准确命名“无显式文本锚点”，不能声称 CONCH/TITAN 预训练从未使用文本。
- `T1d_SCORE_ROUTE_CONTROL`: 与B3/B4相同其它部分，但对 TITAN32 scores 使用**当前训练分区固定、与测试真值无关**的匹配统计 surrogate（若组内分布无法严谨匹配，则只做被动消融并标容量差异）。必须分别报告“Query身份增量”“全显式文本先验增量”，不可混成一个 Claim。
- 如果触及模型输入接口，逐项重新验证CONCH官方 image/text projection 与 TITAN官方 slide projection，禁跨编码器未对齐直接 cosine。

### T2｜8槽信息丢失是否是瓶颈

固定相同已训练编码器，比较 **相同病理图像与临床输入、相同50程序输出** 的三个程序读出：

- `T2_mean64`：现有8槽mean后64维，忠实重放；
- `T2_keep_slots`：保留8×64=512维，不跨slot平均；使用训练侧正则化多目标Ridge，输出50个程序，并仅使用当前训练分区调正则；
- `T2_capacity_control`：将8槽经无语义的固定正交/训练侧主成分降为与 `keep_slots` 同信息预算的控制，或使用相同 trainable 数量的单向MLP但仍仅接收mean64。必须固定主要控制选择以拆分“保留槽位”与“自由度更多”两个因素。

对 B3/B4 **相同编码器checkpoint**各做以上只读特征抽取及训练侧头拟合；不要把 8×768+TITAN 的历史 PATHOLOGY Ridge直接当成8×64模型对照。匹配训练／验证目标测量mask与Huber计算。若训练新神经读出，B3/B4同 seed 完全等成本重训，且单独计训练拟合数。

### T3｜已知／未知关系的归一化与语义边的必要性

固定8×50关系**来源与边集合**，不改变既有15条来源候选的生物学含义。实现一个受控两通道算子：

`S = a * mean_valid_known(pair_score) + (1-a) * mean_valid_unknown(pair_score)`，按每患者/程序 measured mask 分别归一化；不支持已知连接的程序必须安全回退未知通道并标记，不能除零或把无 known 支持强行置0。预锁 `a ∈ {0.5, 0.75}` 作为诊断两档，原始实现为强对照；a 不得用 DSS 选取。报告各通道实际平均贡献、标准差、覆盖率和有效梯度，不把 a 当成生物学可信度。

- `T3_true_map` 与 `T3_shuffled_map` 保持相同已知边数量、slot有效支持与程序测量支持。尽量使用交换操作保持行列度；若严格度保持不可达则显式失败/降级标注，不可宣称完全匹配。
- 与B3同结构随机查询对照共享两个通道与超参，**不能只给B4有效的关系机制而不给B3**。两通道若使模型有效参数数改变，提供匹配的无语义分支。
- 把已知/未知的数量、归一化后的权重分布与语义映射的正/错差值同时输出；如置乱效果几乎不变，要明确是语义关系不被利用还是正则/覆盖限制。

### T4｜对比监督是否挤占程序预测

先做 R0 已保存 checkpoint 的双损失分支梯度诊断，再预锁 `lambda_pair∈{0,0.03,0.1}` 的**同预算、同seed、同负例集合**受控训练。所有臂的 `L_prog`、paired retrieval、random retrieval、combined `L_aux` 原始值分别报告；**不同λ的 combined loss 不可直接互比作为唯一选模依据**。

主评估采用无 DSS 的 `(masked patient-equal program Huber, across-patient macro Pearson, hard matched retrieval gap)` 三项分开排序。checkpoint 选择使用各 arm 同一预锁规则：优先较低验证 program Huber，若差小于训练侧定义的固定 tolerance，则较高 hard-gap，再取较早 epoch；**不能按已看到的 DSS 调整权重**。对比0配对损失时，检索仍只能作为诊断，不能用未训练目标与训练过检索目标的差异直接当因果胜负。

## 3. 真正训练的最小充分矩阵与执行顺序

先 `R0`（0新神经拟合）→ `T2`（复用checkpoint，外加小规模固定头）→ `T1/T3/T4`。在同一已用开发 inner partition 0，种子固定31/47/83；所有成对臂共用患者、初始化、负例支持、程序mask、checkpoint候选。开发矩阵如下（允许缓存共享，但不得混淆训练次数）：

| 组别 | 必须训练的对照 | 重训要求 |
|---|---|---|
| T1 | B3保守 / B4真文本 / TEXTLESS / score-route 控制 | 历史B3/B4只作冻结参照；若改变输入/评分路径并需重新优化，必须共同重训比较组 |
| T2 | B3、B4各有mean/keep-slots/容量匹配读出 | 同checkpoint做冻结读出；若训练表示层，两个对照一视同仁 |
| T3 | B3与B4各自原始关系 / 双通道关系，外加B4映射shuffle | 所有真实结构变更都做独立训练，按同seed、同预算比较 |
| T4 | B3与B4各lambda=0/0.03/0.1 | 记录真实从头训练；原lambda0.1符合原输入合同可安全复用才复用 |

**优先复用未改输入合同且 hash 合格的历史 B3/B4 checkpoint，不要为了刷“拟合数”重跑；但任何前向输入路径被改变后需要学习适应的对照必须真实重训。** 记录 `NEW_FIT`、`REPLAY_ONLY`、`REFIT_HEAD_ONLY` 三种身份。任何测试修复不能只对B4生效而不对B3。

预算先用既有 `50/100/200/400` epoch 单元评估，不盲目延长；允许统一增加 `25` epoch checkpoint以测试早期最优（必须对该组所有臂共同运行，且事先锁定）。若验证Laux和程序Huber在200→400均变差，不续800；若有共同未收敛签名，可预锁一轮训练侧等预算扩展。除旧学习率3e-4，补充1e-4**仅作为全对照组共同且事前固定的敏感性**，不能从众多组合中以后验最好分数作“确认”。每轮验证训练/推理所有张量非NaN、不出现全零信息流、图像Query梯度非零。

完成开发矩阵后，**冻结最多两个候选结构修复**（例如slot-preserving和关系双通道）与原始B4为并列对照，在当前outer-fold1的另外两个**预先锁定、互不取代**的inner partitions做每组31/47/83重复。不能根据重复结果继续无限换模块；每次选择与消融均报告已复用验证数据的事实。任何新可训练的program head / ROI读取 / adapter必须只利用对应 inner-train 输入拟合；不能共用含inner-valid患者训练的旧checkpoint。

## 4. 验收指标与决策（不计算CI）

每seed及合并描述性结果均报告：50个程序逐一有效患者数/测量覆盖、患者等权Huber、跨患者宏Pearson/Spearman、预测幅度SD比、双向hard/random Top1和与**逐anchor精确chance**的差值、任务loss分量、梯度份额、有效容量、关系已知/未知质量、inner分区、最佳epoch。NA不可填0。

主要判别：

- 文本查询必要性：`B4 - B3_conservative` 与 `B4 - TEXTLESS` 同向；前者回答**Query身份**，后者回答**是否需要显式文本**。二者不一致时不能宣称完全否定/证明医学语义。
- 关系先验必要性：同结构 `true_map` 是否优于 `shuffled_map`；同时观察真实 known 通路是否有激活与覆盖。
- Slot瓶颈：相同checkpoint下 `keep_slots` 与 `mean64` 性能差，且容量匹配控制不能解释全部差值；只看与历史维数不匹配Ridge的差不合格。
- 配对有效性：B4 与 B6a 真/错正对，`L_pair=0` 与 `>0`，应分开报告 matched retrieval 与 program prediction；真实配对的检索优势≠预后优势。
- 预算：所有比较臂早停曲线与前/后阶段数据保持一致，允许记录负结果。**不因缺少置信区间判停或判胜；不得用同队列多seed冒充独立样本。**

预先决策：
- `P1R_SEMANTIC_EVIDENCE`：至少文本查询或显式文本、正确映射、真实配对三条证据中的具体路径在多个seed及另两个inner分区方向一致，且有真实验证效果；说明通过的是哪条，不虚构全部通过。
- `P1R_REPRESENTATION_ONLY`：只证明保留槽位/输入容量改善，无文本特异收益；P2仍可测试实际DSS增量，但主Claim降级。
- `P1R_NEGATIVE_WITH_ADEQUATE_TRAINING`：全部等容量对照及预算审计合格仍无效果；冻结当前合同，不再无限调词/权重。
- `P1R_INCONCLUSIVE`：缺checkpoint、训练失败、同容量比较不成立、损失或梯度无法诊断；明确修复资格，而不是给科学负结果。

## 5. 测试与安全边界

编写自动测试：input hash/模型数据来源；`DSS/OS/PFI`不可读取；随机改变validation RNA不改变image_only prediction；test RNA/DSS不参与fit统计；三seed同负例集合；保谱原型旋转确保持Gram；失活文本控制不可通过`TITAN_scores`旁路；关系无已知边时fallback正确；换错患者仍不换自身真实程序目标；已知/未知归一化保留measured mask；B3/B4相同预测激活参数（容差须记录）；每个真实新checkpoint保存和重新加载前向数值一致。软件测试通过≠科研成功。

## 6. 输出与停止条件

创建：

```
configs/semantic_b0b7/p1r_lock.json
src/semantic_b0b7/  # 只新增必要独立算子，不破坏P1历史重放
scripts/semantic_b0b7_p1r_*.py
reports/semantic_b0b7_p1r/
  00_PARENT_AND_GATES.md / json
  01_SAVED_STATE_DIAGNOSTICS.md / json
  02_TEXT_ROUTE_AND_INPUT_FAIRNESS.md / json
  03_SLOT_READOUT_COMPARISON.csv
  04_RELATION_NORMALIZATION_AND_SHUFFLE.csv
  05_AUXILIARY_LOSS_GRADIENT_AND_WEIGHT.csv
  06_INNER_PARTITION_SEED_METRICS.csv
  07_TRAINING_AND_COMPUTE_LEDGER.csv
  08_INDEPENDENT_REPLAY_AUDIT.json
  09_DECISION.md / json
  10_P2_FROZEN_CANDIDATES.json
  11_REPRODUCE.md
```

禁止输出文件空壳、想象结果、私有患者数据、脱敏失败资产。报告 `n_new_real_fits / n_refit_heads / n_saved_replays / GPU实际开销`；重复检查独立算术的范围（如仅验证哈希/算术不能称独立从原始图像重推）。任何资格门失败先停止相关训练并修复，只对受影响**整个匹配比较组**重跑。最后给出**不超过两种预锁修复候选＋原始结构**进入P2的确切源码 SHA、参数、输入输出维度、预训练目标、固定选轮和成本。不要等待进一步确认，不要把新DSS评分加入P1-R。

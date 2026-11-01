# RSCIL v0：来源、理论与接口审计

研究日期：2026-11-01。本轮不重新解释既往外折成绩，不新增真实数据或终点。创新状态为 `NOT_ESTABLISHED`。

## 来源与版本

用户提供两份研究方案及原始交付压缩包；包 SHA256 为 `7554b6b5a7d715ab685c8a65922f25c5938749683fe60f6438433c3922bf8c92`。压缩包中的 riskset_core.py、synthetic_gate.py、test_riskset_core.py、synthetic_results.json 均已读取，原字节副本分别保存在本轮 reference 子目录。原脚本的 36 个线性模拟已实际重算，Harrell 结果一致；优化器等风险数值误差存在约 1e-8 量级差异，不能写成所有浮点数逐字节复现。

新服务器的训练目录是源码副本，不是有效 Git 仓库；另一个可读取的历史仓库 HEAD 为 `9040f5cb32d311832b3107bbba1569a8f6f98761`，分支为 `codex/phase10-trimodal-mechanism-audit`，已跟踪文件状态为空。未重置或覆盖它。通过账号绑定的 GitHub 接口核实目标研究分支 HEAD 为 `974e2bfa0664cfba6aa4240a3eab9a1e9c4642e1`，tree 为 `80a7f7ccaab2a3f827d3755d613727eafaccbe8c`，完整树含 4200 个文件。新分支 `research/rscil-riskset-interaction-v0` 从此提交创建；发布时只添加本轮命名空间文件。

本轮训练源锁为 `configs/rscil/run_lock.json`，包含新源码、配置、测试、原始参考脚本和实际调用的旧模型文件哈希；冻结发生在首个正式神经拟合之前。运行位置和进程号只记录在私有运行目录，不写入公开附件。

旧研究的 52 个锁定源文件哈希重新检查，变化为 0；上述七个实际接口文件的 Git blob SHA 也与参考提交逐一一致。未将训练目录副本的状态冒充最新研究分支的 Git 状态。

初版只回放最终选择权重，不满足全部快照验收，因此停止继续派发，活动单元自然完成，共保留 81 个旧版单元，排除在主裁决之外。`replay_amendment.json` 只补齐所有快照的磁盘回放，不修改数据分布、模型、初始化、损失、预算、选轮或门槛；新版源锁 SHA256 为 `e9992023adeb10aae73156b26a690d7a9661ab88e008e7aa2b42f0690c62813c`。初版源锁和执行器原字节单列保留，实际额外训练成本记入 fit_ledger。

## 实际接口与调用关系

`bgb_en.baselines.prepare_anchors` 在当前训练分区构建预处理与交叉拟合锚点；`bgb_en.protocol` 在训练患者中拟合临床编码、RNA512 和 RNA/CNA 缺失通道。旧生存终点不由本轮合成程序加载。

真实数据原接口为 `semantic_m123r_common.build → semantic_m123.data.prepare → semantic_m123r.data.repair_inputs → RepairModel`。P 包括 S32、固定 CONCH GRID32、TITAN native/aligned；G 保留 RNA512 与原程序值、mask、coverage；C 的类别编码维数随当前训练分区确定，不能写死。旧源文件和本轮真实特征资格不是同一件事：本轮尚未运行该 TCGA 数据入口，不能宣称已经重新核验 512 人数据。

原 `Branch.forward` 输出标量，隐藏表示应显式重建为 `linear(x)+residual(linear(x))`。原 R2 还存在 `mean(raw normalized patch)→768×64 projection→local_out→pathology head`，本轮完整保留，并在病理 hidden/risk 中计入该路径。`S0` 的初始化函数与当前训练教师对齐；没有加载旧外折风险作为模型输入。数值测试还覆盖 local_out 非零时与旧 R2 的一致性。

旧 R2/R3 的选轮规则曾优先 Harrell，且旧总预算为 140；本轮改为预锁的 90 轮与内层验证 Cox 选轮。因此历史 R2 分数不能替代同本轮协议的新 S0，不能把跨研究的差值全部归因于投影。

## 经典数学与真正待检验的部分

Efron ties 的观测信息等于各虚拟事件风险集加权协方差之和；事件平均损失的 Hessian 需再除以事件数。新实现采用逐风险集 log-sum-exp/Softmax，避免全局 exp 缩放在极端风险差下下溢。

固定 B、Z 时，`[B,Z]` 与 `[B,Z−BA]` 的线性风险函数类相同。系数映射为 `beta_B_new=beta_B+A beta_Z`；独立 L2 不保持不变。投影不能称为增加表达能力，也不能把条件信息矩阵非零解释为具有事件预测信号。

本轮 rho=1e-4，故 `I_BZ_perp=rho*A`，并非严格零。无 ridge 的测试另验证精确消去；正式训练不偷偷改为无 ridge 或伪逆。只去除三个主效应风险标量张成的方向，未去除全部单模态非线性信息。

可检验的候选改变仅是：学习 Z 时每轮采用当前完整训练风险集的几何、冻结求得的 A，再训练同一个双线性头。`r_main`、B 和 A 在几何路径停止梯度，Z 保留梯度。这是指定的停止梯度训练规则；它不是对包含 A(theta) 的全部复合函数求完整导数。不能把这套规则本身等同于已证明的有效评分或最优估计理论。

## 前作核验与创新边界

- 参数正交与 nuisance 信息几何属于经典统计工具：[Cox & Reid, 1987](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1987.tb01422.x)。该来源不是本算法原创证明。
- [CAMR](https://pmc.ncbi.nlm.nih.gov/articles/PMC9857974/) 已使用共享和模态特异表示及 Cox 预测，不能将“共享/特异解耦”重新命名为原创。
- [MOTCat](https://arxiv.org/abs/2306.08330) 的 OT co-attention 和 [SurvPath](https://arxiv.org/abs/2304.06819) 的通路—病理 token 交互已存在，普通多模态乘积或关联学习不能独占为新机制。
- [BMLSurv](https://www.sciencedirect.com/science/article/abs/pii/S0031320325010593) 处理多模态生存学习失衡与风险排序知识转移；与本轮扣除三维主效应得分方向的几何并非同一操作。该区别不足以构成全面排重结论。

已核对这些直接相关的原始来源，未完成穷尽式相似算法检索或原创性证明。理论问题清楚与论文创新成立是两项不同判断。

## 优势、风险与置信度

优势：新增算子范围小，完整保留强主效应；S1/S2/S3 共用模型结构与新增参数；投影训练/预测身份明确，有可证伪的合成准入门槛。

风险：只有三个 nuisance 得分；几何依赖有限事件；投影可能只改变正则化/优化坐标；合成数据不代表全部 TCGA 情形；负结果只能约束本轮锁定实现，不能证明所有互补信息不存在。

数学恒等式及现有接口核验的置信度为高。可泛化收益和独立算法创新尚未建立，不能给予高置信度结论。最终科学裁决见本轮 decision.json，不以软件测试通过替代预测门槛。

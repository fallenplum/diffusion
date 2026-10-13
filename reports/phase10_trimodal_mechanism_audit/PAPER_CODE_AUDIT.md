# 论文与官方代码审计

实际 commit、文件 SHA 与历史来源见 `SOURCE_REGISTRY.json`。以下是本队列内部的机制适配实验，**没有任何 arm 被标为论文 EXACT 复现**。论文 OS 与当前 512 人、65 DSS events 不同；论文数字不能和本轮 C-index 排名。

|论文|可取得资料|本轮级别|明确阻塞|
|---|---|---|---|
|SurvPGC|原文全文、作者代码|B 组 ADAPTED|原 scFoundation/临床文本编码、病理骨干与离散 NLL 未搬用|
|CiMP|作者代码；IEEE 全文页需验证，未取得全文|C 组 ADAPTED_CODE_MECHANISM|paper-exact、Stage Prompt 原始步骤及原论文数值 BLOCKED|
|MRePath|9 页原文 PDF、作者代码；公式页已渲染查看|E0 ADAPTED；E1 INSPIRED|不含 Sheaf，置信度头未实现原文 TCP 校准|
|BMLSurv|出版信息及作者摘要；全文/官方源码未取得|D 组 SEL/PAL INSPIRED|不得称 faithful，不报告无法核实的原公式或消融数字|

## SurvPGC

[论文](https://www.nature.com/articles/s41746-025-02257-y)，[可读全文镜像](https://api.repository.cam.ac.uk/server/api/core/bitstreams/73682287-4948-40ed-93e2-098a4780cea0/content)，[固定作者代码](https://github.com/Houjiaxin123/SurvPGC/tree/dc467984c0a59216c7266246612523c176dd9730)。论文发表于 2026，文章号 9:76。其 COAD/READ 样本为 298，终点 OS；另有 LIHC/BRCA。论文以 scFoundation 的 4 个组学 token、6 个临床文本 token 和病理 patch 构造两个交互路径。上下文 token 接受上下文自身及病理，病理只从上下文接收信息，不做独立 P→P attention。输出四个分支表征。

原文 p9 Eq1–12 与 `models/model_SurvPGC_foundation.py`、`models/layers/cross_attention.py` 是本轮 Q/K/V 与拼接顺序锚点。原损失 p9 Eq13 为离散 4-bin 生存 NLL；Eq14 相似性项不在其主表的默认模型中，不作为必要复制项。

原始数值：p4 Table1 COADREAD SurvPGC **0.676±0.087**，SurvPath **0.644±0.138**；p7 Table4，late C+P **0.525±0.083**、mixed **0.520±0.104**、high-dimensional **0.581±0.085**、SurvPC **0.670±0.129**。p7 Table3 的 genomics/pathology **0.339±0.045** 与 clinical/pathology **0.661±0.045** 是 IG 归因比例，**不是消融性能或因果贡献**。

本轮不同点：CONCH768→64 共用 patch Transformer、8 个 RNA Top512 学习 token、结构化 Clinical→6 token、连续 Cox-Efron、train-local Clinical Ridge 初值后联合训练、40 epoch 内部 EMA 选择。两分支 FF 独立，并非原作者共享 FF 的完全复制。B3 交换自注意力所属模态，仍为双向交互，不能称单向 attention。B4 共用 QKV，同时让原第二投影进入额外分支局部映射以维持有效容量，故只检验这项已声明的共享配置，不能概括所有共享注意力。原 mask 的零值处理替换为 masked softmax。

## CiMP

[论文入口](https://ieeexplore.ieee.org/document/11080170/)，[固定代码](https://github.com/DeepMed-Lab-ECNU/CiMP/tree/f1a64b8e01ad887835a786cf9644b6b46c24a58a)。TMI DOI `10.1109/TMI.2025.3588836`。全文未取得，因此本报告不编造原队列、终点明细、公式或消融数值；只陈述能够从源码核对的步骤。

`models/cimp.py` 包含六组 SNN、64D GroupMSA、2 bottleneck、CLS、两层 Nyström TransMIL，临床 512D 文本向量在末端平均后参与预测。`train.py` 引用不存在的 `dataset` 和 `models.snn_model`，并以缺少病理/临床参数的方式调用 forward。原 GroupMSA 的 timm Attention 输入布局 `[6,1,64]`，将 6 视为 batch、1 视为 token，该子层没有六组之间的 attention；后续 torch Transformer 又采用另一布局。固定 `.cuda()` 也妨碍 CPU/设备一致性。不能未经修复称作者入口直接可运行。

本轮显式 batch-first，8 个固定 Hallmark 分组，每组训练内最多 64 genes；共用普通 patch Transformer 替代 Nyström，C 组各 arm 均使用第二次相同 backbone 调用。C1 使用常量 token 而保留真正 RNA late fusion，C2 清除所有患者 RNA，而非仅清除 bottleneck。C4 利用相同权重张量做逐 token QKV 线性变换/FF，不做内容依赖的 inter-token MSA，属于容量匹配的自定义对照。C5 以均值/方差分箱内全局基因置换保留组大小和重叠；只有一次锁定随机 grouping，不能当作完整随机零分布。Stage 文本 Prompt 未做；合法数值 Clinical 保留。

## MRePath

[原论文 PDF](https://www.ijcai.org/proceedings/2025/0201.pdf)，[固定源码](https://github.com/MCPathology/MRePath/tree/e3c133b0beb5ded80426afbe49ab41f1003a4ed5)。原文以空间/语义超边、Sheaf 表征及病理/组学 IFA 预测 OS，涉及 BLCA/BRCA/COREAD/HNSC/STAD。p7 Table3，COREAD equal IFA **70.8±8.4%**、dynamic IFA **80.8±5.8%**；五癌均值 **67.8→71.5%**。p7 Table4 的 CONCH COREAD **83.2±7.7%** 来源/协议与本轮不同。

`models/model_HGNN.py` 前向计算了 `w_path/w_gene`，但未将其乘入融合张量；confidence 使用 detach，风险目标没有通过这些权重学习置信度。论文 Eq8 的 mono-confidence 分子属于同一模态，公开实现交换了两模态分子。`models/layers/fusion.py` 返回 `[P,R]`，调用方却按 `[R,P]` 切片。另有 eval 下强制 dropout、非 HGNN 情形未定义变量等问题。

本轮 E0/E1 均返回明确 `(R_mean,P_mean)`，无 Sheaf。E1 用论文同模态分子的公式，实际乘入 token，取消无效 detach，以可训练 sigmoid 头产生数值。**这些头没有原文 true-class-probability 校准训练，不能称已复现 mono/holo confidence**。E0/E1 参数注册相同，但 E1 增加 130 个有效 gate 参数，属于 gate 增益与容量同时变化；记录有效参数量。保存两种可审计行为：`source_behavior=True` 交换分子、权重不进输出；默认 E1 公式版权重进输出。合成梯度测试及完整风险 perturbation 验证前者死 gate、后者活 gate。收益需结合真实配对结果，不以 gate 激活本身判定医学有效。

## BMLSurv

[出版入口](https://www.sciencedirect.com/science/article/pii/S0031320325010593)，Pattern Recognition 172 (2026), 112398。作者摘要可核实 SEL、PAL 与训练模态不均衡的方向；全文/官方源码不可获得，不能复制原作者公式或报告未知实验值。搜索出现的 BPDSurv 代码是另一篇工作，未混用。

D0 固定承载 B0；D1 使用训练内三模态 auxiliary Cox loss 与 detached softmax 权重；D2 用严格 cross-fitted C+S32 teacher，inner heldout 的 survival labels 不进入训练或 lambda 选择，软风险排序目标仅作用于可比较患者对；D3 两项均为 0.1 固定系数。全部是 INSPIRED 探针。projection 梯度范数及输出轴平均梯度夹角仅为诊断代理，**不是原文 ICG，也不是因果模态贡献**。冻结 CONCH/TITAN 输入特征，监测可训练 projection/patch Transformer，不能把 foundation model 没有梯度称为梯度饥饿。

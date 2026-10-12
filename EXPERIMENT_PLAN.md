# Phase10-P0：SCPN-Surv 实验锁定方案

锁定时间：2026-10-12。数据：原 TCGA-COAD/READ 512 患者、65 DSS events；原 5 个患者外折；seed=20261007。

## 对照与选择

正式运行 P0、P1、P2、P3、P4、P3-noTopo，共 30 个 cells。P0 在本轮重新拟合 Clinical 六字段 + frozen TITAN S32 Ridge Cox，lambda 网格 [0.01,0.1,1,10]。P1 使用相同锚点，加 CONCH 多切片等权 mean 的 train-local PCA16 残差 Ridge Cox。P2–P4 在同一锚点上训练神经残差；内层 train/val 与 Phase8 cache 保持相同，所有数据驱动变换按 inner/outer 各自训练集合拟合。

epoch 上限 35，patience=7，lr=3e-4，AdamW weight_decay=1e-4，residual squared penalty=0.01，投影 128D。初次 smoke/formal 发现 alpha=0 与非零但很小的 risk head 结合时，inner-val 常在第 1 epoch 平台，outer residual 几乎没有训练；这被记录为训练有效性失败，而不是模型负结果。修复重跑将 alpha 固定为预先允许的小值 0.05，使 residual head 在第一个 epoch 即有可审计梯度；该值不由 outer-test 选择。采用全训练患者 Efron 风险集；使用两遍 bag 重计算保持精确 Cox 梯度，第一遍取风险及其 Cox 梯度，第二遍逐患者 bag 反传；不以小 minibatch Cox 替换全风险集。初始默认无超参数搜索。P3/noTopo 初始化及训练预算相同；P4 仅更换 rare query 条件。

P0 offset 在神经训练患者上来自 3-fold crossfit：每个 heldout 患者的标签不参与该子模型训练、lambda 选择及标准化。内层 train 的 crossfit 从 inner-train 内再切分，inner-val 标签完全不进入 upstream。每个子模型风险以子训练集合的预测均值居中，不改变尺度；outer-test 用完整 outer-train 拟合风险及其训练均值。Clinical 完整向量与 S32 提供给每个神经 residual head；不只提供压缩 Clinical 风险。

## Region Builder

仅 CONCH v1.5 768D 与其自带 coords。16 个 normalized MiniBatchKMeans 原型；每个训练 slide 最多均匀随机抽样 256 patches，seed 固定，不读取 DSS。inner/outer 原型独立拟合。区域生成使用全部 patches。

读取每张 slide 的 patch_size_level0。规则网格使用真实四邻接；不规则坐标使用半径 <=1.05 × level0 patch 边长的邻接，不连接远距离组织。相同原型邻接连通分量为初始区域。每 slide 上限 256 regions：按接触关系确定性合并较小相邻区域，保留全部 patches 与混合原型直方图；孤立分量不删除。若超过上限且没有可合并接触边，将最小剩余分量汇总到明确标记、无空间边的 other token，记录数量、面积及被移除接触比例。不会将该 token 宣称为空间连续区域。

region 保存均值、log1p面积、离散度、质心、16原型比例、原 patch->region 引用和接触边。多 slide 各建图、禁止跨 slide 边；slide 表示等权 patient 聚合。global gated attention + rare soft top-k（k=8），P3/noTopo 的固定 query 经与 P4 临床 query 相同的映射网络生成，尽量匹配参数量。P4 使用完整 Clinical 向量查询，区域图和双分支保持一致。

## 顺序与报告

先完成全部 519 H5/hash/coords 审计，再做真实单患者结构验证与 1-fold smoke（临时目录，完成后只保留聚合 PASS 标记）；随后正式五折。主统计 sum fold concordant numerator / sum comparable denominator；Uno60 同理，删失 KM 只来自相应 outer-train。固定比较 P2-P1、P3-P2、P3-P3-noTopo、P4-P3、P4-P0。报告正式逐折指标、epoch/loss/gradient/alpha/风险修正、区域面积/拓扑保留、模型参数量。

Region maps 仅用坐标绘制，无伪造 H&E 底图。公开图使用匿名 slide 标签。患者 ID、H5、原型/checkpoint、训练细日志、OOF 保留私有。仅有五折完整实际训练及独立评测通过后写 DONE=PASS；否则报告 BLOCKED/FAIL 并提供恢复命令。

本机原 E 盘当前未挂载；服务器 36973 在首轮检查连通、随后两次连接拒绝。数据可用性须重新核实，禁止以历史数值填写本轮指标。

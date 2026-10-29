# R8 源证据接口

固定源版本为 `R6_S7_locked`。源模型、完整状态和既有 patch 缓存均通过 SHA256 绑定，不按 DSS 选择源模型。本轮不使用 R7 训练 adapter、新 SPGEL 状态、CNA、文本或新增 WSI 特征。

## 图像证据

既有 UNI2-h 特征经冻结 S7 得到 `h:[B,S,N,64]`、四程序分数及 patch mask。旧缓存未保存的程序 query，由同一冻结 tissue head 和 query 模块在 `h` 上重算；重算分数必须与原缓存最大误差不超过 `2e-6`。此过程不重新投影 1536 维原特征，不训练目标投影器。

四个程序依次为 iCMS2、iCMS3、FAP-fibroblast-associated top25 和 MARCO-macrophage-associated top25。它们描述源监督的程序代理，不构成测得的单细胞来源或患者组织学标签。S7 在本配置不使用空间 GNN；空间配对监督不能被描述为图像前向已经包含空间图传播。

图像独立读取仅用冻结图像程序分数，绝不输入目标 RNA。WSI 内部 softmax，再对非空 WSI 等权平均；padding 不参与。所有图像独立特征跨 RNA 内层分区必须逐元素一致。

R8-5 另外以冻结基因程序向量作为 query，加上与 `normalize(h+source_query)` 的余弦兼容度改变读取权重；value 仍为本人冻结图像状态。缺失 query 精确回退到无条件读取。固定源分数系数为 0.1，余弦温度为 1。

## 基因证据与单位

实际源 GeneEncoder 训练输入是源训练侧 observed-only 逐基因 `Z(log1p CP10K)`；CP10K 的分母来自完整可测基因库，而非程序基因子集。

目标输入沿用 TCGA 当前内层训练侧中位数填补后拟合的逐基因均值/标准差，再附加真实缺失 mask。不能把源 spot scaler 直接套用到 bulk。共同为 Z 坐标仅是接口形式一致，不证明组织组成、平台或生物状态已经对齐。

主 G 表示固定为冻结源 GeneEncoder 的四个 64 维程序向量，再于当前目标训练行内各做 PCA2。真实缺失通过 program mask 处理，观测零不当作缺失。记录实际奇异值、有效秩和常数列，不加 jitter 伪造第二维。

signed mean 和 member dispersion 仅供无标签接口/域审计，不进入任何 DSS 头，也不扩展正式模型矩阵。错患者 query 可用性 mask 与本人基因 measurement mask 分别保留。

## 随机源与拒绝条件

随机源复用既有固定 seed31 全 S7 随机状态及均匀三组织先验，未进行 ST 优化；图像和基因编码器都随机，不只替换一个风险头。

源版本/字节变化、缓存分数无法重放、无有效图像、无效维度或新特征超出409人范围均使资格失败关闭。患者级数组、mask、风险、索引、权重及原始数据仅存私有目录，不进入公开报告或 GitHub。

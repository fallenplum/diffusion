# HEST 官方 50-HVG：四次 Ridge 基准合同

当前状态：实现和合成认证已完成；正式数据拟合尚未启动。八组新 BF16 特征、逐文件资格与新的运行锁均通过后，才运行四个预定单元。原四程序开发 pilot 的 74 拟合、198 次回放及其权重不进入本任务。

## 固定来源和范围

官方源码固定为 [HEST commit 3ddb5eaf](https://github.com/mahmoodlab/HEST/tree/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3)。数据版本为 `MahmoodLab/hest-bench` 的 `b039b5965ed7092f280d4f336da7972ef9a10e1a`。原始训练代码、读取代码与 benchmark 的文件 Git blob 均逐项核验；不是把 README 身份当成完整源码版本。

本轮只有 COAD、READ 各两个原始官方 fold，seed 均为 1，共四个多输出 Ridge 估计器。每次另计一次训练折 StandardScaler 和一次 PCA 拟合；它们不冒充额外的神经模型训练。MLP、GNN 和源模型迁移不在本合同中。

新图像特征由冻结 UNI2-h 在官方 RGB patch 上进行 BF16 autocast 推理，FP32 容器存储，保留原始 embedding 尺度。无行 L2 归一化。旧 FP32 提取缓存保留历史身份，不混入严格新特征集。

## 数值路径

1. 按官方 CSV 中 sample 顺序拼接；每个样本按 feature H5 的原始 barcode 顺序索引 H5AD，再按官方 `var_50genes.json` 顺序选择 50 基因。
2. 表达转 float64 后只做 `log1p`。不做总文库归一化、CP10K、目标 Z 标准化或测试折基因重选。图像 QC 子集小于完整表达矩阵时，只取对应 barcode；不补齐图像、不截断或重排真实对应。
3. 仅在官方训练样本拟合 `StandardScaler → PCA(256, random_state=1)`；测试只 transform。PCA 后显式转 float32，匹配官方 `torch.Tensor(...)` 边界。训练数据不足 256 维/行时拒绝，不静默降维。
4. `Ridge(solver='lsqr', alpha=100/(256×50)=0.0078125, fit_intercept=False, max_iter=1000, random_state=1)`。记录实际 scikit-learn 版本与默认 tol，不另做参数选择。
5. 官方每基因 Pearson、50 基因平均先按 split round4，再对两个 fold 的已返回数值平均，最后 round4。完整精度结果、median PCC、MAE/MSE、供体/切片差异另列，不替换官方主表。

常数目标/预测导致的 NaN 保持未定义。为生成有效 JSON，NaN/Inf 编码成 null，并保留定义基因数；不填 0，不通过丢掉未定义基因改变官方平均。

## 资格和回放

数据资格证书逐项绑定 HVG、八个 CSV、表达 SHA、原 RGB patch SHA 与供体身份；特征资格证书逐样本绑定实际 feature SHA、原 patch SHA、模型/配置/提取器身份、精度、维度和行数。通用 PASS 字符串不足以替换这些映射。

真实 H5 另核样本、patch SHA、数据版本、精度、归一化属性、唯一 barcode、原坐标及可测基因。表达文件 basename 必须对应官方 CSV，防止不同 Visium 样本复用 barcode 后出现错误配对。

每个拟合保存私有 pipeline/Ridge joblib 与预测。随后实际载入 joblib 重算预测和全部完整精度指标，回放过程不创建拟合。出公开报告前再次核当前模型、预测和 COMPLETE 的 SHA，防止回放后文件变化仍被标记完整。

本机 CPU 执行，最多两进程、每进程两线程，按实际可用 RAM 调度；不占用本机 GPU。原始表达、逐 spot 输出与模型均留在 E 盘的独立 derived/private 版本目录，不传到 GPU 服务器，不进入 GitHub。

## 身份与解释边界

READ 为两个已核实供体的官方折、已知供体不跨训练/测试；但这两个供体有历史研究曝光，本轮从零拟合头，不借旧 source 权重，也不宣称新独立确认。

COAD preview 的供体身份未确认，保留 `COAD_IDENTITY_UNCONFIRMED`。其结果只表示官方固定 split 的算法复现，不能伪造一个唯一患者编号来声称完整 donor-disjoint。

优势是直接复现公开主任务、数值边界和已发布划分，并用实际保存模型回放验证。限制是独立患者很少、部分身份未知、基础模型预训练重叠未完全确定；四个 Ridge 不能构成 SOTA 或跨模态机制结论。

许可与归属见 [NOTICE](../../src/spgel_hest/NOTICE.md) 和未改动的 [HEST 原许可证](../../src/spgel_hest/HEST_LICENSE.md)。本实现把官方 recipe 作为复现对象，不将其称为新算子。

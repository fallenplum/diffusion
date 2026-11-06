# 多模态肿瘤生存风险预测与分析软件 V1.0

> **定位：**预编码的病理语义S32、CONCH GRID/LOCAL768、患者临床变量和RNA表达联合的**研究用DSS相对风险预测软件**。本发行版是 diffusion BG-BEN R4 中 `Clinical + S32 + LOCAL-CONCH PCA32 + RNA512 scalar` 计算路径的独立、可演示实现，不提供原SVS切片、CONCH/TITAN权重或受限制患者资料。

## 一分钟演示

```bash
python -m pip install -r requirements.txt
python -m r4surv demo --out demo_output
python -m pytest -q tests
```

输出 `demo_output/synthetic_features.npz`、`r4_model.json`、`synthetic_predictions.csv` 和 `demo_receipt.json`。演示数据完全合成，不能用于发表TCGA效果、临床评估或算法优越性声明。

## 输入和命令

- 数据接口：患者级`npz`，包括 `patient_id`、`S32`（N×32）、`conch_local`（N×768）、`rna`（N×G）、`gene_universe`（G维布尔）及6项 `clinical_` 开头的临床字段。训练时还需要DSS月数`t`和状态`e`。
- `python -m r4surv train --input your_authorized_feature_file.npz --output model.json`
- `python -m r4surv predict --input feature_file.npz --model model.json --output risk.csv`
- 输出为**相对风险分数**，较高代表模型认为较高的事件风险，不是五年生存概率，更不是临床诊断。
- 同一模型的训练与预测必须保持RNA列顺序一致；建议实际发行时另外提供不可逆的基因列哈希与模型登记台账。

## 算法与来源

- 强CP分支：训练折局部标准化的临床变量与TITAN来源S32，加上CONCH局部768维经训练折PCA32形成的特征，采用Ridge-Efron Cox。
- RNA分支：训练折内按可观测性、方差和固定注释宇宙选择512维，再用Ridge-Efron Cox拟合分子风险。
- 融合：训练数据内部3折交叉拟合CP和RNA风险，再根据同一teacher分区内可比较患者对估计标量融合系数，最终重新拟合全训练分支并输出 `r_CP + beta*r_RNA`。
- **历史上真实R4成绩不属于此演示的验收结论。** 在没有受控TCGA资产、原冻结权重和完整原pipeline的条件下，此V1.0不声称精确复现历史每位患者的R4预测，也不包含历史源的CNA旁路（历史R4 CP_SCALAR不依赖其生存风险）。
- 来源仓库：`fallenplum/diffusion`，原研究分支 `research/qpr-fusion-v1`，R4主文件 `src/bgb_en/protocol.py`、`src/bgb_en/survival.py`、`src/bgb_en/baselines.py`。**原仓库无显式 LICENSE，软件著作权归属及对外发布权限须由项目成员和学校核实。**

## 安全、软件与权属边界

此软件代码并不包含第三方CONCH/TITAN参数或第三方代码。`patient_id`仅用于结果对齐，不收集真实患者个人信息；演示文件全部使用`SYNTH`前缀合成编号。模型文件JSON是机器可读的固定推理状态，拒绝任意Pickle对象。软件对未见类别使用训练集词表，未见类别按全零one-hot处理。不得从本演示指标推断医学有效性。签署或提交软著前必须核对申请主体、合作开发及原学校大创任务书中的成果权属。

见 `docs/使用说明书.md`、`docs/技术设计说明.md`、`submission/登记材料准备清单.md`。

# S5-B 官方实现资格：BLEEP 与 TRIPLEX

核验日期：2026-10-29。**已完成源码与输入合同审计，尚未复现这两种方法的正式训练。** 本次没有下载原图、表达矩阵或权重，没有新增真实拟合，没有修改旧 SPGEL/Ridge/神经控制结果。JSON保留29个实际读取的官方文件之commit、blob、大小和URL。

## 版本与论文边界

| 对象 | 实际审计版本 | 许可与当前状态 |
|---|---|---|
| BLEEP | `23959670c7407510c1716a173184bdcc18833a35` | 根Apache-2.0文本已读取；NOT_REPRODUCED |
| TRIPLEX | `4750051b9a8ba1122c9d4a30a40943edc418a1c5` | 未找到覆盖整个仓库的根许可；仅有嵌套CLAM许可，不能据此假定整个仓库的再分发权限；NOT_REPRODUCED |

BLEEP论文为[NeurIPS 2023官方页面](https://proceedings.neurips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html)，TRIPLEX为[CVPR 2024官方页面](https://openaccess.thecvf.com/content/CVPR2024/html/Chung_Accurate_Spatial_Gene_Expression_Prediction_by_Integrating_Multi-Resolution_Features_CVPR_2024_paper.html)。后者页面直读403，但出版方搜索索引实际核对了题名、作者、会议和11591–11600页，未伪称成功下载全文。

TRIPLEX当前作者[README](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/README.md)明确列出2025年的输出截断、APEG和配置修改。API还能定位到2024年底的`a712c174b2261821403a4a6dfd45c7e6cf317224`；BLEEP论文年份内的近邻commit是`948ac07bbb7b180d55618060773d756863d42381`。本次没有审计这两个旧快照，不能把当前代码与论文原始实验版本混称。

## BLEEP：已有RGB可以接入，但当前发布入口不能直接称训练资格通过

BLEEP使用原始图像编码与表达投影的对比学习；测试图像查询训练参考表达。官方例子的reference明确排除了被查询的留出切片，**不能把测试RNA加进reference**。原实现使用ResNet50与256维双投影，默认表达维度3467；这是其原生任务，不能直接等同本项目固定50HVG。来源：[模型](https://github.com/bowang-lab/BLEEP/blob/23959670c7407510c1716a173184bdcc18833a35/models.py)、[配置](https://github.com/bowang-lab/BLEEP/blob/23959670c7407510c1716a173184bdcc18833a35/config.py)、[推理notebook](https://github.com/bowang-lab/BLEEP/blob/23959670c7407510c1716a173184bdcc18833a35/BLEEP_inference.ipynb)。

实际核查到三个需要显式处理的接口/合同问题：

1. 原[Dataset](https://github.com/bowang-lab/BLEEP/blob/23959670c7407510c1716a173184bdcc18833a35/dataset.py)加载全切片再裁224块。`is_train`没有在构造中定义；ToTensor已给出CHW后，getitem又进行了一次维度置换。服务器以原类AST和一个合成patch做CPU资格：缺少该属性时实际报AttributeError；明确给定非训练属性后，transform为3×224×224，getitem成为224×3×224。没有构造图像编码器、没有优化器，也没有真实数据读取。机器收据为`S5B_BLEEP_SERVER_DATASET_CONTRACT_PROBE_20261029.json`。本机另一次尝试因缺torchvision停止，不把本机依赖缺失误认为作者模型本身失败。
2. 当前[训练入口](https://github.com/bowang-lab/BLEEP/blob/23959670c7407510c1716a173184bdcc18833a35/BLEEP_main.py)把训练切片合并后按spot做80/20随机内部划分，以内部验证对比损失保存checkpoint；它不是供体独立的内部验证。当前入口默认4轮也不能当作充分训练证明。
3. [预处理notebook](https://github.com/bowang-lab/BLEEP/blob/23959670c7407510c1716a173184bdcc18833a35/preprocess.ipynb)以四份切片共同构造HVG并联合Harmony。直接用于本项目会让外测试表达进入预处理，不符合现有训练折内合同。这里评价的是代码接入本项目的条件，不据此裁决原论文实验。

因此，现有八份RGB可以支持一个**保留原生图像编码器/对比算子、明确修复H5读取与形状的HEST适配**；需要逐barcode绑定、原图色彩/几何核验、仅训练参考和开发侧预算。把UNI2 embedding直接替换其图像编码器，只能另称固定特征适配对照。

## TRIPLEX：真实连续邻域是当前最直接的数据缺口

当前[模型](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/model/TRIPLEX/TRIPLEX.py)分别读取target图像、neighbor特征及全切片spot-context。target使用CIGA ResNet18；公开release的checkpoint大小92,148,628字节，仅核对release元数据，没有下载。

作者[预处理](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/preprocess/prepare_data.py)在0.5μm/px下裁224×224 target和1120×1120连续neighbor，再用[提取器](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/preprocess/extract_img_features.py)形成5×5邻域特征与真实mask。当前E盘相关raw/features目录的只读文件清点发现8份target H5，未发现WSI类型文件、neighbor H5或global命名特征；这是限定目录的文件级核查，不声称整块E盘没有其他文件。

**global分支并非必须新增一份dense全WSI图像**：当前作者代码可以把所有target patches按官方编码器重算为全切片spot-context。因此“没有现成global H5”不是永久阻断。真正尚缺的是target范围之外的连续1120邻域像素/官方对应特征；不能用邻近spot的稀疏224块、零填充或UNI2替代后声称原版多尺度输入已满足。

还需处理三项源码合同问题：

- [DataInterface](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/dataset/data_interface.py)在fit阶段把phase=test当validation；[底层Dataset](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/dataset/base_dataset.py)据此读取同fold测试清单，而[callbacks](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/utils/train_utils.py)用val_target选checkpoint、early stopping与调学习率。若直接对接本项目官方外折，会利用外测试表达选轮数；必须另设开发内层或冻结开发预算。
- 默认预处理为cigar，但[TriDataset](https://github.com/NEXGEM/TRIPLEX/blob/4750051b9a8ba1122c9d4a30a40943edc418a1c5/src/dataset/tri_dataset.py)只在部分训练global调用传播配置模型名，其余neighbor及测试调用仍默认uni_v1。需要实际检查文件路径、编码器和维度绑定，不能静默混用。
- 作者有HEST训练脚本，但本次固定tree未包含脚本引用的config/hest配置文件。项目默认表达处理也可能含CPM/平滑，与当前官方仅log1p合同不同；不能默默沿用。仓库许可和完整依赖仍待运行资格确认。

## 对本阶段的执行决定

优势是两个作者实现的关键路径、版本和输入缺口已有可追溯证据，可以将“共享UNI2特征公平比较”与“原生模型实现接入”分开。风险是少量供体、历史暴露、几何与邻域缺口、源码接口和版本差异仍存在。上述源代码/文件清点事实置信度高；正式训练可运行性尚未通过，性能没有新增证据。

下一步先用开发数据锁定共享特征控制的充分预算；BLEEP优先做显式输入修复和train-only参考资格；TRIPLEX先补连续邻域及模型名绑定、解决版本/许可，才进入真实训练。当前两项都保持NOT_REPRODUCED，不给没有运行的方法填写成绩。

# 论文基线核查（执行中）

核查日期：2026-10-16。以下原始研究/官方代码是来源；不采用文献跨队列分数充当本队列结果。

|方法|原生输入与来源|代码与许可核查|当前处理|
|---|---|---|---|
|MGCT|病理+组学；[论文](https://arxiv.org/abs/2311.11659)|[作者仓库](https://github.com/lmxmercy/MGCT)，固定`daa60d7ae602e329299b86563cf29f571259e34d`；根目录未见独立LICENSE|私有检出供运行；公开仅提供适配器与SHA。Clinical/CNA适配与原方法区分。|
|TransSurv|病理多尺度MS-Trans、RNA、CNA和Clinical；[原论文记录](https://pubmed.ncbi.nlm.nih.gov/35976825/)，DOI10.1109/TCBB.2022.3199244|当前未找到可核验作者代码/许可，继续核查；现有单尺度预提取特征不能原样复现多尺度MS-Trans|不能用挂名通用Transformer代替。若仍缺代码及多尺度资产，将记录阻塞，不填写伪造分数。|
|ME-Mamba|病理+组学；[原论文](https://arxiv.org/abs/2509.16900)、[正式发表](https://pubmed.ncbi.nlm.nih.gov/41734694/)|作者摘要表示将公开代码；尚无可核验仓库|不把自行简化模块称为官方复现。|
|MambaMIL|原生WSI，含生存任务；[原论文](https://arxiv.org/abs/2403.06800)|[作者仓库](https://github.com/isyangshu/MambaMIL)，固定`9aea39efc13fdcd3703b5b54c28a64887b4acbd7`；README标注MIT，根目录未见独立LICENSE|作为可复现Mamba论文候选；Clinical/omics晚期适配单列，不能称原生三模态。|
|CAMR|原生病理+组学；[原论文](https://academic.oup.com/bioinformatics/article/39/1/btad025/6986972)|[作者仓库](https://github.com/wxq-ustc/CAMR)，固定`8fbae0a0bc4e9da31547b5eab5d773e6b2049e93`，根目录未见独立LICENSE|完成清单核查，原生并无Clinical。|
|Multimodal Prototyping|病理+组学；[论文](https://arxiv.org/abs/2407.00224)|[作者仓库](https://github.com/mahmoodlab/MMP)|核查清单；不是原生Clinical三模态。|

## 可比性约束

统一患者折、DSS、可测RNA基因、同坐标ROI预算、Efron Cox。更换生存头和特征维数是明确适配，并不等于原论文训练协议的逐项复现。记录原生输入、统一适配输入、注册与实际有梯度参数、GPU预算、依赖失败，保留公平简单Ridge对照。

条件S6实现使用exact ZOH；官方Mamba selective scan使用的输入离散化与返回最后状态的反传接口有差异。因此当前项目的SSD-Fusion命名不表示使用Mamba-2 SSD内核。

## 实际适配与依赖核验

MGCT、MambaMIL已各完成15内层+15外层正式cell，另各1个预锁定内层诊断；实际分数见RESULTS.md。Clinical/S32是统一适配，并非两篇论文原生三模态支持。MGCT批量端口与原层输出/梯度误差0；MambaMIL兼容kernel输出/梯度通过N=1/16/128/512字面递推检查。

MMP固定提交`1fd75e37592f7ab47b787f737733ba3c5a7e714c`，LICENSE.md首行为`## creative commons`；本轮只核查清单，没有声称训练MMP/CAMR/ME-Mamba。TransSurv于2026-10-16再次按题名、DOI、作者搜索，仍未取得可核验官方仓库；[作者论文条目](https://smt.bit.edu.cn/szdw/jsml/bssds/6de6308941ca4ed0bfb252e52087598a.htm)可确认研究，但没有替代的可运行实现。

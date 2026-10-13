# 限制

- 512/65同队列反复研究，所有显著性探索性；三seed只随机初始化。
- 一个inner split，80epoch允许最小10；部分边界未饱和，完整训练不等于全局收敛。
- B0/C0、B5/D1、P0/A6、P3/B0精确复用，不能当独立证据。
- 不同模块有效参数与gate/aux不同，匹配注册数不等于功能容量。
- Hallmark仅覆盖部分Top512，fallback不是医学先验；一个随机对照，不是完整随机基因集零分布。
- 模型CP/RP为原论文适配，不是官方EXACT；未微调foundation。
- patient S32与局部CONCH条件关联不证明概念定位或因果基因—病理关系；UNI2网格未局部配对。
- 30,000患者bootstrap不含重新训练/inner选择方差，也未重估训练IPCW；MC极端分位区间记录。
- subgroup事件稀少，只描述，不追逐显著性。CNA本轮无正式模型；未更换终点。
- v1缺反向CP的114记录INVALID并保存；v2全矩阵重跑，不混排名。
- 私有患者OOF/checkpoint不公开；公开包不能独立重算无授权患者数据。Git历史已有的旧公开记录未重写。
- D3仅诊断，监督更新不进入主预测，额外RNA计算消耗dropout RNG，不保证A6与D3逐位相同。
- S32强anchor联合可训练，不是固定风险上界；可能被训练损害，修正幅度实际记录。

- 每cell elapsed包含patch加载和计算，未逐次拆分原训练的数据加载耗时；GPU遥测包含CPU probe及加载间歇，不能当作纯模型吞吐。
- A6相对A5同时增加Clinical及S32内容作为Patch query，query数量也改变；A6−A5检验组合引导，不能单独归因S32。B0−B5维持Clinical query并仅以常数query替代S32内容，是更直接的语义内容对照。
- D0错配S32同时影响query、最终head和联合训练线性通路；B0−D0检查患者级S32总体贡献，不能单独定位query机制。
- RNA probe采用8个token均值的64D读出（及train PCA16），不是全部8×64展开；均值池化的损失不能归因于整个token体系的信息完全消失。

- 归档发现缺tests包入口，查看外层数值后进行了P2打包修复（POST_HOC_PACKAGING_ONLY）；正式source hash、checkpoint与OOF均未改变，失败日志保留。

- A6与B6这两个完全无有效RNA的配置在15个seed×fold上OOF逐元素一致。两组均按原计划实际拟合，fit计数不改成复用；B0−A6与B0−B6不是独立机制证据，原保守16项校正保持不变。

# P1-R 严格复现实验记录

已完成36个开发拟合、另两个inner partition的48个从头复验拟合；合计84×400次优化器更新和336次真实保存状态前向回放。旧B3/B4六参照只从既有四state按本轮统一规则重选，历史Laux结果和T2旧六状态均未改。

所有程序相关、Huber、检索和分层值均来自各自合法训练/验证范围；同患者多seed不视为独立队列，不计算置信区间。

P1R复验fraction为0.5，来自先锁规则六cell开发比较；P2另以理论事前固定fraction=.5，不能倒写成因P1R胜出才选。

## 核验范围与限制

- 训练只读无结局输入；没有DSS/OS/PFI风险拟合。P2必须重新建立对应患者折的预处理与encoder，不能复用P1R已覆盖患者的权重。
- 新模型336次实际回放由训练producer执行。另一个程序从保存state读取pair-valid数组并核360份mask，追加逐anchor exact chance；后者没有独立从原图或基础特征重推。
- 不同lambda的combined loss不作为唯一胜负标准；三类指标和同seed方向分开保留。lambda=0时未获辅助训练的gene/u/v等路径显式标记，不作为P2主预训练候选。
- TEXTLESS指无显式文本锚点；CONCH/TITAN预训练本身可能使用文本。score surrogate只匹配训练均值/协方差，不声称匹配完整分布或条件医学真值。
- known关系为历史候选，unknown不是负关系；归一化权重与实际贡献均单列，不把混合系数视为生物学可信度。
- 新训练relation_trace的known/unknown贡献为除temperature之前的兼容度，换算InfoNCE logit须除以固定temperature=.2；R0的分解直接用logit单位。双通道的unknown_logit是未使用buffer，不能把它的常数轨迹视为训练失效。
- 公开文件保留所需聚合统计，完整private receipt由SHA绑定而非声称全文无损公开；患者级预测、数组和权重均留服务器。

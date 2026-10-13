# 历史证据与本轮对照

历史来源的实际 commit/blob hash 见 SOURCE_REGISTRY.json。历史报告的数字不可跨不同终点、特征与聚合口径排序。

|历史来源|已有证据/风险|本轮对照|仍值得测试及消除的混杂|
|---|---|---|---|
|M2R2/M3/M3R2|官方编码器/方向与简单融合历史；不同 C 指标协议|B0–B5/A3–A6|当前同 CONCH patch 和同连续 Efron，消除骨干和任务损失差异；ADAPTED 而非官方 exact|
|M5R/M6|QK 稀疏/Hallmark 语义未有稳定确认|C0/C4/C5|同组数/维度/参数；随机全局基因置换保留组大小和重叠，消除低维本身优势|
|M7/M8P1R|FiLM 小增益未超 CommonOnly；旧 residual 曾用错弱 anchor、OOF 选择泄漏已修|A0/A3/D2/D3|Clinical skip 联合优化且初值 train-only；教师每个 heldout 独立于训练和内层选择|
|M9/PO0|RNA-linear 有独立信号；encoder 降低不等于 RNA 无信息|A1/R_PCA16/B0/C0/冻结探针|输入严格 train Top512；比较表征训练前后，不按不同历史协议排名|
|PO0 pathway repair|真实 Pathway-RNA 必须字段正确，CNA 对照同 seed|C5/A8|真实 vs随机组相同体积；CNA gain/loss 分开，不把 signed 平均接近零解释成无异常|
|PO1 repair/PO2/PO3|inner-train upstream 必须不看 inner-val；oracle headroom 未可实现|所有 arms / CP_selected|完整内层 pipeline；外层重新拟合；不拼接独立 ABMIL latent；合法 inner winner 不等于外层赢家|
|CPO Phase4|Clinical-condition 方向性并非稳定胜出|B0/B2/B4|保留合法 Clinical skip，去掉 CP 交互时仍有 Clinical 输入，隔离分支交互|
|Phase6/6A|CONCH coords 与 TITAN native/text projection 分工已审计|A2/A3/A4|独立聚合；不把 native 当 cosine；519 slides 全量来源查 SHA，无原始 SVS 重算|
|Phase8-V0R|C+S32≈0.80585/0.77364；修复 fold IPCW numerator/denominator|A3/A4/A5 / CP_selected|当前全部统一新 inner split；S32 不等于 PCA32；旧私有 OOF 缺失，无法宣称历史逐点复算|
|Phase9 repairs|内外结构一致、随机图统计对象一致；低连通图的负结果有限|匹配容量/三 seed配对/硬门控|本轮不扩图；风险同患者配对，平均 seed 指标而不平均不同模型风险后冒充原统计量|

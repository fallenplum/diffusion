# R2：RNA/CNA、成员来源与顺序解释

## 两条不同的成员谱系

RNA512 直接风险支路沿用旧八程序 union（977 外部成员、955 个 canonical RNA symbol）；在当前训练折按覆盖率至少 0.9 和表达方差选择 512 个基因，不足时显式使用非成员后备。每个交叉拟合教师独立重复该选择。

通路 token 使用独立冻结的 Hallmark50 GMT：union 4,384 成员，canonical RNA 覆盖 4,191，CNA 覆盖 4,252。GMT 标准化字节的 SHA256 为 `ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772`。同名旧八程序与新 GMT 的成员不同，不能偷偷互换。逐程序覆盖及来源见 R0_GENE_SOURCES.json；[MSigDB Hallmark 官方说明](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H)。

## 观测掩码与表示

- RNA 原始有限值 mask 在填补前保存；203 人共同缺失的原表 3,024 列对应 collapsed-symbol 后 3,015 个基因。该缺失模式不等于已确认的平台身份。
- 每基因训练均值/尺度、覆盖、方差、RNA512 选择、PCA 及风险教师都仅由当前训练分区拟合。
- CNA 为连续 signed log2 dosage，先分成 gain/loss 再按训练折分别标准化，不通过正负简单均值抵消异常。未测通道保留 mask，不冒充正常拷贝数。
- 稀疏 gene–program 成员边允许多重归属。每通路保留四种可学习读取模式和 32D token，不使用全基因二次复杂度 dense attention。
- O3–O9 都保留 RNA512 风险旁路；冻结关闭旁路与真实重训去旁路回答不同问题，不能互相替代。

## 顺序和知识负对照

实际冻结 Mamba 输入为 GMT 枚举顺序、反向和两个 seed270101 的固定随机顺序。配置文字曾误写为成员重叠拓扑顺序，已在 PROTOCOL_ACCOUNTING_ADDENDUM.json 更正说明；不修改已训练源码或旧锁。多顺序集成不证明生物序列存在。独立四随机顺序的真实重训敏感性使用另行锁定的 seed270901。

O9 通过固定基因列置换保持各程序大小及重叠计数，打破真实基因身份与成员关系；RNA direct 分支仍保持真实成员。它检验新通路分支的知识增量，不删除整个 RNA 模态。

## 信息保真与结论边界

最终神经检查点分别从完整 tokens 和池化 32D 表示用训练行拟合 ridge 风险解码器，评估 held-out RNA-only 教师风险的 Pearson/R²。解码器不使用验证患者结局拟合；可解码不等于新增条件 DSS 信息。O8 2/4/8 因子及顺序重训结果单独报告，未执行项明确 NOT_RUN。

优势是来源、缺失和容量可审计；限制是 Hallmark 成员并非患者级程序活性真值，弱风险分支或负对照无差异都不能单独证明全部生物先验无效。

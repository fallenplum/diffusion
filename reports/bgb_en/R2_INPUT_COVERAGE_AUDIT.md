# R2 输入集合覆盖审计（0 拟合）

本审计只核查输入支持集，不读取保存风险、不评估生存终点，也不修改训练结果或选择锁。

## 数据来源与覆盖

状态：**COMPLETE_INPUT_METADATA_AUDIT**。核验 3 份主准备状态及 9 份已有教师准备状态；新增优化器/PCA/教师拟合/GPU调用均为 0。

基因符号按既有 `load_inputs` 的规范顺序及 GMT exact-symbol 映射核查；下表只给匿名计数。

| 量 | 数值 |
|---|---:|
| canonical_measured_gene_universe_count | 20511 |
| legacy_union_measured_gene_count | 955 |
| Hallmark50_union_measured_gene_count | 4191 |
| legacy_Hallmark50_intersection_measured_gene_count | 953 |
| legacy_union_not_in_Hallmark50_measured_gene_count | 2 |
| Hallmark50_union_not_in_legacy_measured_gene_count | 3238 |
| legacy_union_covered_by_Hallmark50_fraction | 0.9979057591623036 |
| Hallmark50_union_covered_by_legacy_fraction | 0.22739203054163684 |

## RNA512 与通路 Token 的实际输入重合

真实 Hallmark50 用于 O3–O8；O9 使用预锁 seed 270099 的列置换，其实际支持集另列。

| 准备状态 | 真实通路覆盖 / 512 | 比例 | O9 置换覆盖 / 512 | 原旧并集外回退基因 |
|---|---:|---:|---:|---:|
| partition_0 | 510 / 512 | 0.996094 | 98 / 512 | 0 |
| partition_1 | 510 / 512 | 0.996094 | 95 / 512 | 0 |
| partition_2 | 510 / 512 | 0.996094 | 97 / 512 | 0 |
| partition_0_teacher_0 | 510 / 512 | 0.996094 | 96 / 512 | 0 |
| partition_0_teacher_1 | 511 / 512 | 0.998047 | 97 / 512 | 0 |
| partition_0_teacher_2 | 510 / 512 | 0.996094 | 98 / 512 | 0 |
| partition_1_teacher_0 | 510 / 512 | 0.996094 | 97 / 512 | 0 |
| partition_1_teacher_1 | 510 / 512 | 0.996094 | 95 / 512 | 0 |
| partition_1_teacher_2 | 510 / 512 | 0.996094 | 95 / 512 | 0 |
| partition_2_teacher_0 | 510 / 512 | 0.996094 | 95 / 512 | 0 |
| partition_2_teacher_1 | 510 / 512 | 0.996094 | 96 / 512 | 0 |
| partition_2_teacher_2 | 510 / 512 | 0.996094 | 98 / 512 | 0 |

各所选基因的通路隶属次数分布、准备文件和来源 SHA256 见同名 JSON；不公开基因名称、选择索引或患者数据。

## 解释边界

- 未覆盖基因确实不进入该通路分支；这与压缩/池化造成的表示损失是不同环节。
- 即使全部 512 基因被覆盖，也只证明输入集合包含关系，不能证明聚合保留排序信息。
- 线性 decoder 的低可解码性不能统一归因于池化；需同时考虑输入支持、有限解码器、正则几何与训练。
- O3–O9 的直接 RNA512 风险路径仍被保留；此审计不改变既有整体模型性能结论。
- 本统计基于规范 RNA 基因列表的静态成员关系，不是逐患者测量覆盖率或细胞来源真值。

## 优势与局限

**优势：**复用既有准备状态并校验哈希，零拟合，直接排查两条分支输入集合是否一致。

**局限：**不测表示可逆性、不提供预后增量证据；分支中的实际数值缺失与 CNA 来源还需要各自审计。

**置信度：高（限上述代码与文件哈希匹配后的计数结论）。**

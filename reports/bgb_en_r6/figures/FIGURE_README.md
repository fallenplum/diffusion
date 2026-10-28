# R6 普通诊断图

本目录只读取匿名报告生成普通统计诊断图，不执行训练、模型前向、患者级风险重算或模型选择。每个 SVG/PNG 配对来自同一个 matplotlib figure，均为单坐标轴；没有置信区间或 p 值。

| 文件 basename | 统计口径 |
|---|---|
| source_huber | 四个程序的全局 Huber 先在程序内等权；两 seed 先在供体内平均，再展示六供体点及供体等权总体值。 |
| source_within_tissue_rho | 沿用模型无关的组织/程序 eligibility；先组内 rho，再供体内两 seed 平均。12 次拟合不是12名独立供体。 |
| retention_Harrell | 九种表示的三个分区点；总体值为各分区 Harrell 分子和除以分母和。旧 fullRNA20511 固定 L2=10，其他为1。 |
| target_Harrell_delta_B1 | 每 seed 先合并三个分区的分子分母，再与同 seed B1 作差；展示三 seed 点及其等权均值。 |
| target_Uno60_delta_B1 | 与上图相同，使用既有 Uno60 加权分子/分母；不拼接风险后跨分区排序。 |
| risk_branch_SD | 第80轮 validation 中 main/cross log-risk 修正的已存标准差，展示九 cell 点与各自均值；线性坐标，B2/B3 交互结构零保持为零。 |
| query_attention_TV | 每 cell 使用固定前8验证患者的 patch+null TV 平均值；同一24名患者反复用于90模型，720次评价不等于720名独立患者。 |

图内小点只描述供体、分区、seed 或模型单元的离散情况，不构成独立外部验证。SD不是解释方差比例；注意力敏感性不是组织学真值或因果证据。缺失或无定义值显示NA，不用零替换。未完成矩阵拒绝生成整套图，已有同名产物拒绝覆盖。

`R6_FIGURE_BINDINGS.json` 绑定所有实际读取匿名文件的字节数/SHA、绘图器SHA、各图来源与轴单位，以及14个图文件的字节数/SHA。SVG只有向量图元和字体定义，不包含栅格 image 或外部 href。

# P2 风险读出与强锚点：锁前设计及资格审计

本报告记录锁前的软件与协议裁决，**不是完整 B0–B7 DSS 结果**。不计算置信区间或 bootstrap。

## 固定读出合同

所有适用臂统一保留 26 个可训练 Cox 系数：病理 8、组学 8、临床 8、机制/加性容量控制 2。病理输入使用同一冻结 encoder 的 8×64 原始 slot 展平为 512 维；组学使用 50×64 原始 program tokens 展平为 3200 维；临床为其独立 64 维表示。每一当前训练分区独立标准化、PCA8、再标准化，机制两列只按当前训练分布标准化。

锁前发现旧 pooled-gene 是共享 Linear(6,64) 后均值，其可变秩至多 6，完整 mask 时常只有 RNA/gain/loss 三个均值。硬将这种表示白化为八维会放大 float32 舍入噪声。因此所有臂统一保留 program 身份，不使用 mean-G 作为主风险输入。PCA 保留维度必须满足奇异值大于最大奇异值的 `1e-5`；不足则报资格失败，不用零列或噪声冒充活跃容量。

主风险为固定 `r0 + Xw`，完整当前训练风险集的 per-event Efron Cox 加 `lambda/2 * ||w||²`。三个 lambda 固定为 0.1/1/10；按三内折各自分子分母汇总后的 Harrell、Uno60、较强 ridge 的字典序选择。Uno60 缺乏合法删失支持时保留 NA，按预锁 Harrell 规则降级。没有截断测试风险、测试均值校准或额外截距。

## B0 与下游 r0 必须区分

原 R4 精确评估配方调用旧 `prepare_anchors`：旧 GRID-local CONCH、固定 S32、六项临床、旧八程序 union 范围的 RNA512；三个 teacher 块及各自训练侧风险标准化，使用同 teacher-heldout 患者对拟合标量 beta。每次原配方实际为 13 个 Cox optimizer fits 加 1 个标量优化。

原 `CP_SCALAR` 的训练向量虽然使用 crossfit CP/RNA，但 beta 本身仍用这些训练患者 DSS 拟合。因此下游训练 r0 对**整个原配方，包括 beta**再做三块 crossfit；验证/测试用当前 fit-train 的完整原 R4 配方。每个此类 anchor 分区实际成本是 **52 Cox＋4 scalar、16 次旧预处理**，不能继续记为 13。同一分区/所有 seed/所有臂共享该强锚点；B0 的完整评估来源可以共享其中一次 full pipeline，不重复虚增拟合计数。

历史 R4 outer 脚本 seed31 的 teacher seed 为 `27011 + outer*100 + 31`；inner 为 `27011 + outer*100 + inner`。历史 seed47/83 另外进行了标量校准，故本轮确定性 seed31 B0 不是历史三 seed 均值的保证值。旧 GLOBAL source 还由 GRID cache 的 `full_mean` 覆盖，必须记录其真实哈希；不能拿 REGIONS512 或新 Hallmark union 替换旧 B0 而声称精确复现。

所有 teacher 调用仅接受当前 fit＋evaluation 的局部行；evaluation 的 DSS time/event 在传入旧 backend 前遮蔽，其他 outer-test 行不进入该调用。嵌套背景回归保留一个冻结 encoder 坐标，临床填补/类别/尺度在每个背景子训练内重拟合。

## 已实际完成的资格

- 风险、锚点和 DSS-only 数据边界共 **21 个合成单测通过**，包含 Efron 标量 oracle/有限差分、零增量精确退化、26 列、相对秩反例、原始 token 信息保持、背景训练边界、完整 pipeline crossfit 的 beta 标签隔离、outer-test 标签扰动不改变本 outer 内折，以及原患者 dtype-byte 哈希。
- 服务器真实 E0 anchor-only profile 已完成 outer1/inner0：teacher seed27111、CPU2线程，52 Cox＋4 scalar；Cox 最大梯度 **1.73415e-6**，anchor fitting **11.7146秒**，进程峰 RSS **1.675 GiB**。没有 neural fit、没有 outer-test 指标。该 profile 不等于完整 E0 pipeline，更不等于 E1/E2 完成；其训练成本须另记，只有正式锁兼容才可复用。
- 官方 scikit-survival **0.23.1 真包**在隔离 target 导入成功，兼容现有 sklearn1.5.2。100 组合成例、200 次真实 Harrell/Uno60 API 调用：最大误差分别 **0**、**3.33e-16**。共享环境未改；ecos/osqp/numexpr 等声明依赖未安装，此次 metrics API 不需要它们，不能据此声称该包全部估计器生态已验收。

## 锁前科学解释裁决

1. P2 主修复使用理论等权常数 `known_fraction=0.5`、`lambda_pair=0.1`，**不读取 P1-R dev winner 来选择这两个值**。P1-R 原预锁实验保持不变；其若选出0.75，不能谎称本轮0.5已经完成同一复验。原 B4 lambda0.1 单列诊断；lambda0.03 仅保留在 P1-R，不新增 P2 结构搜索。软件合格但前任务阴性仍可检验 DSS，语义机制预期应降低。
2. B2 使用 raw512 经八个无语义视觉 readers 后进行 8×50 普通 cross-attention。NATIVE 仅指其独立 Q/K/V/O 更大参数版本；这是 shared-token 普通注意力适配，**不是直接 512×50 的 patch 注意力，更不是原论文原生复现**。
3. 同一强 r0 含 TITAN S32，与“整个风险完全 textless”不能同时满足。本轮 `B3_TEXTLESS_INCREMENT` 只移除增量 encoder 的显式文本 query/概念分数；共同 r0 保留。必须标记 adapter 无显式文本、common anchor 含 S32、whole risk 并非 text-free。B4 对该控制的差值只检验固定强语义锚点之外的显式文本增量，不能证明全部文本先验必要性。
4. 全部 outer 的模型 epoch 与全部 arm/seed 的风险 lambda 必须先统一写入冻结选择记录，之后才能执行第一项 outer 评分；不能边输出某臂外测分数、边冻结下一臂选择。

## 优势与风险

优势是强主效应、低容量风险、真实嵌套标签边界与机制对照分离，且计入完整 teacher 成本。风险是仅65事件、反复开发的同一队列、仍可能显著的 crossfit/full-fit 风险尺度差异、预训练对应不等于 DSS 增量，以及同参数量并不等于相同函数先验。以上资格不证明语义或预后增益；最终判断仍须来自完整同协议逐折、逐 seed、逐对照的真实结果。

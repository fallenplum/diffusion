# Phase7A：Foundation Representation Survival Headroom Audit

**正式完成：17 个预先指定表示组合 × 2 个独立终点 × 3 inner-split seeds × 5 固定患者外折；510 outer fits、2040 inner fits、102 个完整 seed OOF 模型。P0R 完成且独立验证通过后才开始本轮拟合。**

## 决策二：Clinical 以外的稳定 DSS 增量尚未确立

**六项预注册 Clinical 条件比较均未通过稳健正增量判据。静态概念是值得保留的小容量候选，但当前结果不足以扩大 molecular prompt 或宣称已解决 DSS。**
有一个预先指定的敏感性比较通过：Clinical+RNA+concepts 相对 Clinical+RNA 的 Harrell 增量。但 Clinical+RNA 本身略弱于 Clinical，不能据此宣布概念在最强 Clinical anchor 上已得到稳定增量。

## 1. 数据和终点

- DSS 主终点：原始 DSS_MONTHS/DSS_STATUS，512 人、65 事件。
- PFS 次要终点：原始 PFS_MONTHS/PFS_STATUS，510 人、129 事件；排除两名时间无效病例，保留已有患者外折，不插补结局。
- 外部 TCGA-CDR 核验：DSS 对 DSS，源文件 PFS 对 CDR PFI，512 人的状态一致，月数与天数/30.417 在 5.5e-8 月内一致。**此处的 PFS 实际源于 TCGA-CDR PFI，不等同于临床试验判定的 PFS。**
- DSS/PFS 独立拟合、独立汇总；不根据 PFS 替换主终点。[GDC 临床来源](https://gdc.cancer.gov/about-data/publications/pancanatlas)。

## 2. 模型和泄露防护

所有固定表示使用相同单阶段 ridge Cox（Efron ties，float64），候选 λ=0.01/0.1/1/10；inner-train 拟合预处理与 Cox，inner-val 只选 λ；整个 outer-train 同算法重拟合，最后评一次 outer-test。
Clinical 仅 Age/Sex/Stage/T/N/M；Age 中位数、类别词表、标准化、RNA Top512、有效基因筛选与状态评分标准化全部 fit-local。仅使用 train variance 排 Top512，再做 z-score。
联合模型直接拼接冻结表征后拟合一个 Cox，没有 in-sample prediction stacking，也没有拼接不同模型的 latent coordinates。TITAN/UNI2-h 不训练，基因集/概念词表不改。
UNI2-h 全量读 520 张 slide、3295129 个 patches：每 slide 全 patch 均值，再等 slide 权重聚合到 512 个患者。**这是 mean-pool 的线性 headroom，不是 ABMIL，也不是 UNI2-h 所有可能下游模型的上限。**
RNA 8 状态和 RNA+CNA 16 状态保持原定义；signed CNA dosage 不被解释为 pathway activation。

## 3. 六类基础表示：真正三 seed OOF

| 表示 | DSS Harrell | DSS Uno60 | PFS/PFI Harrell | PFS/PFI Uno60 |
|---|---:|---:|---:|---:|
| Clinical | 0.7767 | 0.7534 | 0.6982 | 0.6624 |
| UNI2-h full mean | 0.6928 | 0.6849 | 0.6342 | 0.6588 |
| RNA-linear Top512 | 0.6310 | 0.6236 | 0.5899 | 0.5877 |
| TITAN native | 0.7274 | 0.6964 | 0.6482 | 0.6582 |
| Static TITAN concepts | 0.6838 | 0.6594 | 0.6249 | 0.6058 |
| True RNA states | 0.5052 | 0.5853 | 0.5677 | 0.6107 |
| True RNA+CNA states | 0.5446 | 0.5827 | 0.5605 | 0.5686 |

数值为每个 seed 先拼五外折 OOF、算指标，再平均三个 seed；不是 15 个 fold-cell 指标的平均。所有 seed 使用同一患者队列，不能当成三个独立队列。

OOF risk 仅按所属模型 outer-train 的 risk 均值/SD 标定；没有用测试分布。Harrell 保留完整时间；Uno 预锁 60 月，outer-train reverse censor KM，事件时点左极限权重，无 G clipping。
相同时间的 event/censor 可比较，双 event tie 不比较；实现与验证独立逐事件核算。OOF Uno 是各患者所属训练折的删失权重拼接版本；另提供不比较跨 fold 患者的 pair-weighted 指标。

## 4. Clinical 条件增量（主要比较族）

| 终点 | 联合−Clinical | ΔHarrell | 条件 95% CI | Bonferroni 族区间 | ΔUno60 | 判据 |
|---|---|---:|---:|---:|---:|---|
| DSS | CR-C | -0.0068 | [-0.0542, +0.0462] | [-0.0696, +0.0654] | +0.0293 | 证据不足 |
| DSS | CU-C | -0.0170 | [-0.0701, +0.0372] | [-0.0865, +0.0578] | +0.0038 | 证据不足 |
| DSS | CT-C | +0.0027 | [-0.0484, +0.0536] | [-0.0618, +0.0666] | +0.0163 | 证据不足 |
| DSS | CS-C | +0.0201 | [-0.0124, +0.0556] | [-0.0232, +0.0705] | +0.0059 | 证据不足 |
| DSS | CM-C | -0.0158 | [-0.0348, +0.0005] | [-0.0415, +0.0055] | -0.0093 | 证据不足 |
| DSS | CMC-C | -0.0223 | [-0.0455, -0.0000] | [-0.0535, +0.0064] | -0.0006 | 证据不足 |
| PFS | CR-C | -0.0269 | [-0.0685, +0.0155] | [-0.0841, +0.0299] | +0.0016 | 证据不足 |
| PFS | CU-C | -0.0282 | [-0.0777, +0.0208] | [-0.0982, +0.0340] | +0.0190 | 证据不足 |
| PFS | CT-C | -0.0080 | [-0.0506, +0.0345] | [-0.0643, +0.0480] | +0.0225 | 证据不足 |
| PFS | CS-C | +0.0040 | [-0.0138, +0.0231] | [-0.0200, +0.0313] | -0.0031 | 证据不足 |
| PFS | CM-C | -0.0047 | [-0.0234, +0.0127] | [-0.0282, +0.0180] | +0.0139 | 证据不足 |
| PFS | CMC-C | -0.0204 | [-0.0519, +0.0099] | [-0.0623, +0.0182] | -0.0034 | 证据不足 |

代码：R=RNA-linear；U=UNI2-h；T=TITAN native；S=static concepts；M=true RNA states；MC=true RNA+CNA states。每个终点六个主要对比，使用 99.1667% bootstrap 分位区间作为 Bonferroni 族区间。
通过条件为：Harrell 族区间下界>0、三个 seed Harrell 差值均>0、平均 Uno 差值>0。PFS 使用同规则单独描述，不升级为确认性主终点。

## 5. Clinical+RNA anchor 的预注册敏感性比较

| 终点 | 对比 | ΔHarrell | 95% CI | 四项族区间 | ΔUno60 | 判据 |
|---|---|---:|---:|---:|---:|---|
| DSS | CRU-CR | -0.0073 | [-0.0595, +0.0507] | [-0.0735, +0.0677] | -0.0192 | 证据不足 |
| DSS | CRT-CR | +0.0203 | [-0.0339, +0.0730] | [-0.0439, +0.0875] | +0.0133 | 证据不足 |
| DSS | CRS-CR | +0.0281 | [+0.0079, +0.0477] | [+0.0031, +0.0544] | +0.0239 | PASS |
| DSS | CRMC-CR | -0.0060 | [-0.0175, +0.0059] | [-0.0218, +0.0093] | -0.0062 | 证据不足 |
| PFS | CRU-CR | +0.0065 | [-0.0371, +0.0507] | [-0.0517, +0.0589] | +0.0152 | 证据不足 |
| PFS | CRT-CR | -0.0008 | [-0.0365, +0.0359] | [-0.0458, +0.0443] | +0.0026 | 证据不足 |
| PFS | CRS-CR | +0.0019 | [-0.0076, +0.0127] | [-0.0100, +0.0155] | +0.0017 | 证据不足 |
| PFS | CRMC-CR | -0.0105 | [-0.0215, +0.0006] | [-0.0244, +0.0037] | -0.0153 | 证据不足 |

**解释限制：**DSS CR=0.7699、C=0.7767；CRS=0.7979。CRS−CR=+0.0281 的 Harrell 四项族区间 [+0.0031,+0.0544]，三个 seed 同向；Uno60 +0.0239 的 95% 区间 [+0.0025,+0.0476]，但族区间 [−0.0031,+0.0535] 跨零。
CS−C=+0.0201 的 95% 区间 [−0.0124,+0.0556] 跨零。因此“静态概念为当前 RNA 联合模型补充信息”有条件内部证据，“稳定改善 Clinical 最强 anchor”尚未确立；这更不能支持 RNA→prompt、真实 mapping 或 patient-specific conditioning 机制。

## 6. 训练预算和审计

- 全部 2550 个 inner/outer Cox 均达到梯度判据；最大 outer iterations **785/5000**，inner **811/5000**，最大绝对梯度 **2.2e-06**，上限 1e-5。
- 内外层相同收敛训练，不使用 early stopping。本轮不存在需增加 epoch 才收敛的证据；收敛充分不证明正则网格或表示已达到最优。
- 独立验证 510 份 OOF/coefficients、60 个 train-only Top512/分割检查、102 份完整 seed OOF；指标误差 **6.66e-16**，预测重构误差 **5.33e-15**。
- 缓存和 OOF 持久化在服务器，可复算，不需要重新训练已有 raw cells。

## 7. 风险与结论置信度

**优势：**统一凸线性 head；严格 inner/outer 预处理；真实 OOF；训练删失权重；同患者配对和两个终点分开；P0R 独立完成后才进生存拟合。
**劣势：**单队列且仅 65 DSS 事件；一次 inner hold-out；三个 seed 共用患者/外折；bootstrap 仅条件于固定 OOF，没有重训 pipeline。Bonferroni 处理固定比较族的区间，不消除历次研究探索偏差；2000 次 bootstrap 的尾部分位还有 Monte Carlo 误差。
Cox 只估计线性风险；UNI2-h 均值池化可能丢局部信息；分子状态均值压缩可能丢预后信息。联合全局 ridge 也会改变 Clinical 系数，因此负点差不能证明新增模态没有条件信息。正则网格有边界选择，不能据收敛证明搜索范围已完备。
风险标定会改变跨折配对排序；raw-risk 与 within-fold 指标全部公开。部分模型 raw pooled 与标定后 pooled 点值相差约 0.02–0.03，不能把本轮数字直接与旧未标定指标混作同一协议排名。
- **高：**数据来源、泄露防护、收敛和 OOF 数值验证。
- **中：**当前固定表示/共同线性头下的点差与条件区间；CS 是容量小、值得保留的候选。
- **低：**外部稳定收益、分子—形态因果机制、复杂 prompt 的可行性，以及“事件少是唯一原因”。

## 拓展思考

Stage 的真实基因集优势与分子状态 DSS 条件增量可以同时前者成立、后者证据不足。当前结果将下一步问题收窄到：静态病理语义能否稳定补充最强 Clinical anchor，而非先扩大分子 prompt 容量。

## 你可能想知道

这轮不能区分“65 事件导致方差”与“表示丢失相关信号”的全部贡献，也不能因 PFS/PFI 有 129 事件仍无增量就排除监督预算限制：两个终点机制不同。仅凭这轮结果不应宣布医学知识 idea 成立或无效。

## 复现

顺序：cache → prepare（校验 P0R 两个完成门）→ worker-id 0–5（六进程）→ aggregate → 独立 validate → report。CLI 见 `phase7a_foundation_survival_headroom.py --help`。源哈希、协议、收敛、每 seed OOF 指标与配对对比均随报告公开；原始数据、患者预测、weights 不公开。

- [Efron Cox ties 官方说明](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)。
- [IPCW/训练删失支持说明](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)。
- [reverse KM ties 原始实现](https://github.com/sebp/scikit-survival/blob/master/sksurv/nonparametric.py)。

# R7 执行与复核说明

说明日期：2026-10-29。本文是复现操作说明，不修改既有协议的创建日期、科学配置或运行锁，也不是完成结果报告。

本轮真实数据资格已经通过，运行合同 SHA256 为 `66f35d52dfbec2002e89b1e3ac7b3114eea0447e82e6555406fc73380560811d`，绑定40份科学文件。正式拟合已启动；在完整收据、297个检查点回放和报告验收之前，不填成绩、不标正式实验完成。

## 1. 固定范围与预算

基础是已发布 R6 提交 `38fda0fb999099af494c16f0153405ab94d9db9e`。只使用既有 outer1 的409名训练患者，保留原三个 inner 分区与 seeds31/47/83；outer1 的103名测试患者继续封存。它是同一开发人群的机制实验，不是新的外部验证。

| 正式训练臂 | 作用 |
|---|---|
| ADD | 同预算可训练主效应，无交互真实重训 |
| RAW | 新引擎下原始尺度、串联双差分真实重训 |
| NORM | 六个固定 train-RMS 交互接口 |
| INDEP | 两个可以独立产生修正的交互方向 |
| SLOT | 四程序独立线性读出，不提前合并为共享均值头 |
| GATE1 | 同步单轮状态门控；**事前固定主候选** |
| GATE2 | 同参数同步双轮，仅作单/双轮对照 |
| SLOT_WRONG_RNA | SLOT 的错初始分子上下文真实重训 |
| GATE1_WRONG_RNA | GATE1 的错初始分子上下文真实重训 |
| GATE1_WRONG_PATH | GATE1 反向错初始病理上下文真实重训 |
| GATE2_WRONG_RNA | 双轮持续保留错初始分子条件的真实重训 |

全部11臂×3分区×3seeds＝**99次新神经拟合、每次80次完整风险集更新，共7,920次正式更新**。B0/B1另有18行零拟合固定参考，不计作新增拟合。源教师、原RNA512教师、CP、same-teacher scalar β和相关预处理精确复用，新 source/teacher/β 拟合均为0；旧R6 B7/B3仅可作历史只读背景，不作为新R7结果。

源完整S7模型冻结，包含projection、四Query、tissue/conditioning和gene stem。固定offset仍为原 `CP + β×RNA512 risk`；共同病理、基因和Hallmark主效应按同预算训练，原始值/mask不换。全部99次执行，不因早期成绩增减臂、seed、轮数或epoch。GATE2不用于验证集选优，不能事后替代GATE1。

## 2. 顺序：只读轨迹 → 校准/资格 → LOCK → 正式矩阵

1. 只读既有R6初始/20/40/80状态，完成R7.0尺度轨迹。
2. 验证既有R6资格、全部实际回放、完整源模型、已资格化source/Patch缓存与上游prepared来源；这些资产先齐备并核SHA，不重建教师或β。
3. 用每个partition/seed的未训练NORM模型建立训练侧RMS私有缓存，共9份、每份6值＝54个应用标量；完成真实数据前向/梯度资格，优化器更新为0。
4. 校准缓存和最终源码均存在后才创建不可覆盖运行LOCK，再写资格收据。正式运行只能消费该LOCK与SHA匹配的缓存。
5. 并发资格从4路实测建立；具备合格证书后，正式调度允许最多6路，已有资格时可直接六路。并发增加只改变排队速度，不增加模型或正式更新预算。
6. 执行99次拟合，随后实际回放全部20/40/80检查点，再生成匿名报告。

已经生成的资格/校准/LOCK不得为了重新运行命令而覆盖。当前实例已通过这些步骤，不应再次运行资格命令去“刷新”已锁数据。复现新私有工作区时，从相同资产与版本重新执行；已有工作区只按其收据继续。

## 3. 路径与CLI

以下是服务器 Bash 示例；所有私有路径都使用占位符 `PRIVATE_PATH`，实际替换为授权、位于Git工作区之外的路径。公开源码/配置保持仓库相对路径。示例不包含服务器地址、账号、令牌或真实患者路径。

### 3.1 R7.0：只读R6的36个状态前向

```bash
python scripts/bgb_en_r7_interface_audit.py \
  --assets PRIVATE_PATH/assets.json \
  --private PRIVATE_PATH/r6 \
  --primary-private PRIVATE_PATH/primary \
  --source-private PRIVATE_PATH/source_formal \
  --config configs/bgb_en/r6_inner_v1.json \
  --source-config PRIVATE_PATH/r6_source_v1.json \
  --device cuda \
  --output PRIVATE_PATH/interface_audit/R7_0_R6_INTERFACE_AUDIT.json
```

该命令使用R6资产/配置和已保存R6 B7状态，不使用R7目标私有目录。3分区×3seeds×初始/20/40/80＝36次真实前向；每分区固定首8名训练患者，不选验证病例，不新训练、无反向，不保存新的资格权重。输出路径必须尚不存在，已归档审计不可覆盖。重复患者与状态不当作36个独立队列，轨迹不能用来挑验证集最优缩放。

### 3.2 R7共同参数与零更新资格

```bash
r7_common_args=(
  --assets PRIVATE_PATH/assets.json
  --private PRIVATE_PATH/r7
  --primary-private PRIVATE_PATH/primary
  --source-private PRIVATE_PATH/source_formal
  --config configs/bgb_en/r7_inner_v1.json
  --source-config PRIVATE_PATH/r6_source_v1.json
  --r6-private PRIVATE_PATH/r6
  --r6-config configs/bgb_en/r6_inner_v1.json
  --device cuda
)

python scripts/bgb_en_r7_qualify.py "${r7_common_args[@]}"
```

`--private`是新R7命名空间，`--r6-private`仅为上游只读资产，二者不能相同。资格创建9份 `calibration/p{partition}_seed{seed}.json`，核验共同初始化、RAW与R6初始前向、冻结源、本人证据不变和梯度路径，然后创建 `targetLOCK.json` 与 `QUALIFICATION.json`。校准不是神经拟合；资格有明确前向/反向调用计数，但优化器实例/更新为0，不将其计入99次正式拟合。

### 3.3 有资源资格后的固定调度

```bash
python scripts/bgb_en_r7_schedule.py "${r7_common_args[@]}" --max-workers 6
```

默认或未获资源资格时最多4路；即便请求6路，没有合格 `RESOURCE_QUALIFICATION.json` 也不会直接启动6路。资源资格绑定本次合同和真实资格SHA，需要至少4个并发模型的真实前向/反向、包含最大双轮GATE2，并保留GPU/RAM实测峰值和每增加worker的上界。调度器检查增加worker后的估计占用不超过实测容量的90%，最大只到6。

本次私有资源观测工具及其原始收据单独保留，不是新增科学模型：4路GATE2各3次前向/反向、优化更新0、不保存观测权重。缺少工具、收据、SHA或足够余量时保持4路，不手填PASS、不用OOM重试寻找并发上限。容器内存应采用实际cgroup容量/当前占用；宿主 `psutil` RAM和宿主CPU百分比不能代替cgroup可用资源。监测窗口可能包含队列空闲，不叫作train-only利用率。

调度器先生成18行固定参考，再执行99次拟合，结束后自动调用检查点回放与报告器。单臂调试CLI仅在本协议正式范围内使用，不授权重复拟合；例如新工作区中计划单元的调用形式为：

```bash
python scripts/bgb_en_r7_run.py "${r7_common_args[@]}" \
  --mode fit --partition 0 --seed 31 --arm GATE1
```

### 3.4 独立回放与匿名报告

```bash
python scripts/bgb_en_r7_replay.py "${r7_common_args[@]}"

python scripts/bgb_en_r7_report.py \
  --private-root PRIVATE_PATH/r7 \
  --config configs/bgb_en/r7_inner_v1.json \
  --output reports/bgb_en_r7
```

若调度器已完成相同步骤，不重复启动新回放；显式 `--resume` 只允许复用身份、合同、资产SHA和终态收据均一致的已完成内容，不是修复失败或继续部分训练的开关。报告/发布失败不能自动触发模型重训。

## 4. 六个RMS的精确边界

每接口一个全局正标量：有效float32标量提升到float64计算平方和/计数，求平方根后最小为1e−6。有效支持为0则阻断。接口是 context、patch、query、evidence、gene_key、reverse_query；gene Key按union760本人真实可测gene mask统计，不按765条程序隶属边重复。

先由未训练模型的raw G0、adaptedPatch、baseQuery和geneKey确定四值；再用已缩放Query/Patch重算P0确定evidence；最后由 `query_projection(P0/evidence)` 确定reverse_query。所有归一化臂共享同part/seed的缓存，不按arm或训练成绩重拟合。只用当前inner-train，不读取标签/validation，不作每患者L2、LayerNorm或Δ归一化；主效应始终raw。交互gene Value除context RMS，保持与G0状态同坐标；gene Value独立RMS仅作诊断。

NORM−RAW同时改变尺度、参数化与固定L2的有效几何，不能唯一归因某个接口。四slot头的有效可训练参数双向增加384；各slot初值等原共享权重，slot头L2为λ/4乘四权重平方和，保持tied初始函数和L2数值，不声称训练轨迹或总容量完全相同。

## 5. 训练、诊断与297个实际检查点回放

80轮为唯一主预算，主性能检查点固定80。每轮完整当前训练风险集的Efron Cox通过确定性两遍output-gradient cache求导，不在小微批分别计算Cox。Adam lr1e−3、clip5和共同惩罚按已锁配置执行；严格确定性、TF32关闭，CUBLAS workspace为 `:4096:8`。

每模型保存20/40/80的state、optimizer及train/validation预测。**99×3＝297个检查点**均须重新加载state，实际前向train与validation，最大风险误差≤2e−6且pair-credit变化为0；不是只检查权重文件存在或SHA相同。源冻结状态、身份/映射、六尺度和固定train mean也在回放合同内。20/40用于诊断，不能选epoch。

20/40/60/80均记录完整train风险SD、程序风险抵消与完整风险集梯度：同一完整Cox权重下的total/cross/两个方向参数梯度，total/cross residual-penalty梯度，以及独立parameter-L2梯度（含slot的1/4几何）。不同对象的范数不能相减作因果解释。

Query TV、接口范数、null/gate与轮间轨迹使用每分区预固定首8名**训练**患者，另标探针范围，不能冒充全部409人；这些探针与full-train梯度口径分开。60轮只做train诊断，不读validation；诊断无优化更新，调用/耗时另记。未保存量保留 `NOT_SAVED`，不以最终量回填早期轮。

## 6. 错配、mask与状态更新

train/validation各自沿用无自配对Hungarian双射，优先实际RNA缺失pattern、COAD/READ、Stage和平台缺失代理；不读DSS/PFI，映射跨seed共享。代理平台不当作核验过的测序平台。

- wrongRNA每轮为 `foreignG0 + (ownG_l−ownG0)`，对应foreign初始source mask；保留错误初始分子条件和本人的反馈定义，不在第二轮恢复本人G0，也不借用对方更新后的状态。它不是整模态替换。
- wrongPATH只在GATE1真实重训，反向上下文为 `foreignP0 + (ownP_l−ownP0)`；foreignP0按当前参数实时核算。本人Patch/gene Value、本人测量mask、raw主效应输入、RNA scalar和offset不换。
- 两方向同步从当前状态计算，更新锚定P0/G0；gate为零权重/bias初始化的 `2 sigmoid(Linear128→1)`，初始1、范围0–2，非医学置信度。两轮共享参数，再次读取本人原始证据。
- 全mask/null零修正时退回原状态。自然零表达仍可测；未测值不当作零表达，up/down独立mask与归一。非空gene读取不自动具有独立learned null。

“主效应逐元素不变”只针对同一参数状态的冻结反事实；真实重训臂的可训练主效应可以学成不同参数。冻结关闭说明模型依赖，真实重训才是学习必要性对照，二者不混用。

## 7. 评测、失败处理与公开交付

Harrell和Uno60分别按每seed累加三个分区内部的concordance分子/分母，再等权三个seed；不拼跨分区风险。Uno的删失参考仅来自当前train。配对修复/损伤由真实同分区风险信用直接计算，不由C-index相减补造。PFI仅评价冻结DSS风险，不训练/选择PFI模型；不计算CI、bootstrap或p值，不按已看val挑缩放或单/双轮。

不完整矩阵、失败单元或未完成回放不得以部分均值代替完整结论。partial/ERROR/FAIL保留原始收据，不autoRetry、不新增seed绕开失败、不修改已锁文件或放宽容差。报告器只在99拟合和297回放完整后生成完成报告；任何当前进度都不是最终成绩。

匿名交付以PROTOCOL/config/LOCK、资格、逐折分子分母、固定比较、学习曲线、诊断与真实回放收据为证据。患者标识、原始表达/坐标、风险向量、权重、凭据和真实私有路径不公开。旧R6报告与资产保持只读；本说明是新非科学文档，不延伸或重写已锁40份文件。

## 优势、风险与置信度

优势是有限真重训、强主效应保留、分阶段比较、初始身份控制和逐检查点实际回放。风险是同队列反复开发、有限事件、ST与bulk域差、六供体复用以及source/TCGA互斥仍为 `UNVERIFIED`；gate、null、程序名称和注意力均不是组织学或细胞来源真值。独立反向还可能含剩余主效应。

对已锁执行合同与固定预算的置信度高；尚未完成的性能和医学机制结论未知。新外层仍为 `NOT_RUN`，本轮不启动外层、不按结果扩展架构，负结果不否定全部医学idea。

## 拓展思考

路径幅度变大、程序选区不同、风险修正非零与本人条件带来可泛化排序增量是不同问题，需分别对照，不用单个诊断替代整条机制证据链。

## 你可能想知道

99次拟合、9次无监督RMS校准、36次R6只读前向和297次检查点回放属于不同预算单位；任何无更新诊断都不能冒充新增独立训练或临床验证。

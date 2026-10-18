# Phase25-B：强主效应下的条件局部残差路由

## 根据Phase25-A锁定的假设

归因审计完成后才锁定本协议。原高点估计不构成已确认机制：全局CONCH有主效应，C4对患者查询有响应但检索必要性双指标分裂，一维RNA方向可保留但条件泛化未确认。因此只检验：相同强主效应之外，RNA条件局部读取及额外Clinical条件能否产生可重复的内层增量。

固定原患者划分的inner分区1/2/3、seed31/47、R0/R1/R2/R3，共24个模型槽位。三个分区的患者可重叠，不称唯一患者OOF；固定80轮末轮，不选epoch。不同模型的共同输入、预处理、主效应、预算以及局部K/V/head宽度一致。

## 共同主效应

R0是原始Clinical、S32、全部坐标匹配候选的GLOBAL_MATCHED CONCH PCA32、Hallmark富集512 RNA、CNA gain8/loss8、RNA/CNA coverage各8的单一凸Efron Ridge Cox，所有列训练标准化，λ固定10。RNA是强岭约束的线性读出，**不是Phase15的一维SCALAR复现，也不称512维为低维**。CNA扩增/缺失先分离，按已选RNA符号匹配，再使用测得成员均值及单独覆盖率。

Clinical/S32/CONCH的系数在R0共同凸拟合时允许重优化；随后共同主效应在R0/R1/R2/R3间逐元素相同且冻结。另拟合同预处理CP-only λ10作为护栏，防止弱化主效应后制造局部增益。

每个fit另做固定、outcome-blind的3-fold R0交叉拟合。每个teacher必须从原始协变量重做Clinical编码、RNA选基因/填补/缩放、CONCH PCA、CNA及覆盖率处理，仅使用teacher-train；无λ选择。局部残差的fit offset只用该OOF风险；valid使用完整fit的R0风险。不拼接不同teacher的latent，不用教师或outer错排定义监督。逐患者OOF不被称为双方同教师排除的成对监督。

交叉拟合教师与完整fit的风险存在样本量、尺度及误差分布变化，记录训练OOF/fullfit风险SD与相关性，不声称完全解决这项分布差异。原始结局文件可由legacy接口暂时加载，训练接口仅接收fit标签，不宣称磁盘结局字节从未被读过。

## 局部网络

统一复用冻结CONCH v1.5坐标匹配缓存，每患者≤512patch，未按RNA选择候选。宽度16，输入乘固定sqrt(768)，K=tanh(Wk P)，V=Wv P。

- R1：q=tanh(q0)，与患者组学无关的静态学习查询。
- R2：q=tanh(q0+Wr RNA512)。
- R3：q=tanh(q0+Wr RNA512+Wc Clinical)。

各WSI使用同一softmax(qK/√16)，读取 `sum A·V − masked_uniform_mean(V)`，再同宽线性头。患者内等WSI平均，风险修正乘fit交叉拟合R0的SD，再加本人主效应一次。RNA/CNA主效应始终保留；CNA本轮不进入查询，ΔOmics具体检验RNA条件查询。

没有RNA/Clinical到local head的直接旁路，没有最终tanh；采用标准λ/2约定、λ=1，目标中的输出约束实际为 `0.5·mean(unit_residual²)`，不是硬幅度上限。AdamW lr0.001/wd0.01/clip5。完整fit患者Efron风险集，microbatch32仅用于显存计算。q0/K/V/head共同初始化，条件投影零初始化，使R1/R2/R3初始前向相同；不称最终容量完全相同。

均值中心化消除了uniform V消息，但不能证明所有单模态信息都被排除；该模型也没有显式三线性项，不称纯三元作用已被识别。

## 冻结干预与诊断

R2/R3保存RNA均值和错RNA查询；R3另保存Clinical均值和错Clinical查询。所有干预只改local query，K/V/head、同一检查点本人主效应保持不变。配对限制于当前fit或heldout分区的癌种×完整原RNA缺测pattern，零自配，不按DSS、Stage、Age或Sex匹配。

记录实际epoch、fit loss与末段变化、分组梯度、风险修正、attention归一熵/有效patch数、真实/错配注意力差异及top32交集。空间坐标只用于确认原候选一致，没有原图组织学真值。

## 预锁筛选门

全部24槽完成、来源/风险/标签隔离资格通过后才评测。每seed先加总三个inner分区的within-fold分子/分母，再取两seed均值；Harrell和修复Uno60分开。

- ΔOmics=R2−R1、ΔClinical=R3−R2：两项指标均值分别>0，且各2/2seed为正。
- R3−R0、R3−同协议CP：两项均值≥0；这是点估计护栏，不是统计非劣效。
- R3本人−错RNA、错Clinical、Clinical均值：每项两指标均值≥0且至少一项严格正。
- 全部满足才GO，锁定原配置进入完整5外折×3seed；否则当前候选不进入outer，不加深、不换主干、不自动搜索新正则或预算。

预先定义R1有效：R1−R0两项均值>0且2/2seed正，同时R1−CP两项均值≥0。R2有效：R2−R1满足相同正向门、R2−R0/CP护栏通过，且本人−错RNA双指标非负、至少一项严格正。若仅R1或R2有效，只报告并检查查询/选区统计，不把其事后改为新的outer主要候选。

配对修复与损伤对全部可比较患者对计信用质量，核验repair−damage=ΔC。没有内层确认性bootstrap；不跨fold排序raw风险，不将两个指标混成单一排名。

## 计数与交付

24是评测槽位，不等于24次独立神经拟合：R0与教师分块在神经seed间共享，prototype共有3个fullfit R0、9个R0教师和3个CP护栏凸拟合，以及18个80轮神经拟合。资格短测单列计数，权重和数组不保存；正式原型的检查点与风险留私有供回放。有效参数、共同冻结参数、真实耗时和显存均按实际报告。

优势是主效应、局部形态、RNA条件和Clinical条件有递进控制；风险是样本/事件有限、同队列反复研究、教师与完整fit分布变化，以及查询容量随输入增加。创新性未确立。拓展思考／你可能想知道：本轮只在门通过后才有理由扩展完整论文方法比较，不根据原型最高点选择额外架构。

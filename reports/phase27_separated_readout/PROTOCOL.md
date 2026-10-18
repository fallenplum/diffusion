# Phase27：分离局部多查询表示与风险读出

## 首轮问题与范围

冻结Phase26全部18个局部编码器，不再同时改竞争算子、slot数量、分组或训练预算。只替换风险读出，检验已有多slot表示能否在非饱和、低容量读出下提供条件排序信息。强Clinical+S32+CONCH32、独立RNA512及其固定相加主风险逐元素不变，八RNA Token和本人Query residual全部保留。

这是旧模型在已反复研究内层分区上的探索性读出实验，不是独立确认。Phase26的fit表示曾使用这些fit患者DSS训练，不能称交叉拟合表示；仅主效应训练offset严格CF。新head对同fit表示拟合是固定两阶段监督pipeline，但不能因此认定旧encoder已经学到可泛化的交互。本轮不自动进入新的局部网络训练或完整outer。

事件组/删失组方向相关本身并不证明忽略时间：Cox用事件时刻的完整风险集处理删失。故同时保留Efron训练损失，并将可比较对分为event–event与event–censor，计算logistic排序损失与concordance，仅诊断、不改监督目标。

## 固定读出

每WSI取原两轮更新后的8×32 slot，先等WSI平均（逐slot对齐），再展平256维；不先mean八slot。每个arm/seed/fold仅用matched-fit患者拟合full-SVD PCA16，再以fit score总体SD标准化。所有controls使用同一PCA、尺度与head；无按control重拟合或验证后重选维度。

唯一新监督参数为16个线性系数，最小化 `Efron_NLL(M0_CF + Xw) + 10/2·||w||²`。无intercept、tanh、clip、MLP或可学习输出倍数。训练PCA列经总体SD标准化后协方差为单位阵（数值容差内），岭惩罚也直接约束训练修正方差；这不是heldout风险的硬界。固定λ10、rank16，不进行网格或选epoch。三个臂×三分区×两seed，共18个新凸head、0神经训练步骤、0主效应重拟合。

原Phase26风险作为同checkpoint旧非线性head参考，直接复用既有风险向量，不重训旧模型。新head结果不同不能全归于tanh：读出同时改为低秩线性与强岭约束，是锁定的整体读出替换；PCA16也可能丢失有用信息，不能把其失败扩展为256D表示完全没有信号。

## 对照与配对关系

QUERY、ATTN、COMP各自冻结原encoder。matched/fit均值RNA/原错配RNA/Value-off四条件沿用Phase26定义；RNA只变score-query，本人RNA residual、主效应及Patch K/V不变。Value-off关闭消息，冻结干预不能替代独立encoder重训。

另加入PATCH_PERMUTE：先保存本人matched两轮Attention，再对每WSI有效Patch列作固定、无自配对置换；全部slot、两轮共享该置换，Value行不随之移动。每行权重多重集合、熵、有效支持及slot间软重合保持不变，改变权重与具体Patch evidence的对应；不承诺有tie时重新排序的top-k集合恒等。第二轮使用原matched权重，不按已改变第一轮slot重新生成，因此这是显式冻结routing的诊断，不是自然查询替换。单Patch无法零自配对时固定保留并报告计数，无标签驱动回退。

比较COMP本人相对均值RNA、错RNA、Value-off和PATCH_PERMUTE。前两者检验查询身份依赖，最后者检验局部对应是否超越相同分散度；任何正向结果都不是独立组织学事件真值。QUERY的全零attention是禁用占位，不作概率分布解释。

## 训练、回放与评测门

先验证所有旧DONE/检查点SHA、prepared/原主效应、两份独立回放和原STOP。全部特征在无结局接口视图中抽取；仅拟合head时读取已存fit标签。全部18head完成、独立风险重放后才一次读取原DSS valid作统一评测。原encoder权重及所有历史报告前后字节保持不变。私有缓存保留正式slot特征、PCA/head及风险用于审计，不保存冒烟权重。

沿用within-fold Harrell与Uno60、同fit-derived IPCW、1e−8风险tie；每seed先汇总三个分区分子/分母，再mean两seed，不跨fold比较风险、不平均风险后求C。修复/损伤以全部可比较对信用质量归一，闭合到ΔC；训练对损失与验证指标分开，二者不混为单一排名。无bootstrap或新确认区间。

ATTN−QUERY与COMP−ATTN在H/Uno60均值均正且各2/2seed正；COMP相对原M0及CP双指标均值非负；COMP本人相对四项干预各双指标非负、至少一项正。失败则停止当前固定读出候选，保留每个单独比较的正负发现。通过也只支持另锁下一阶段表示学习，不自动启动outer。绝不按结果换rank、λ、checkpoint或对照方式。

## 优势与风险

优势是单独剥离旧非线性风险头、保留多slot对应及强主效应，并通过等分散度控制审计局部权重对应。风险包括反复使用同一512/65队列、旧表示监督记忆、PCA压缩、CF/fullfit风险变化、两个seed与inner患者重叠；没有原图/独立组织学真值。计算资格与具体诊断可高置信报告，医学语义与独立泛化需另证。

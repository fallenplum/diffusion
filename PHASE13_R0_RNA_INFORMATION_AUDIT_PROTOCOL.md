# Phase13-R0：RNA预后信息保留与条件增量诊断

## 第一性原理与范围

RNA-only 排序强不意味着在 Clinical+TITAN-S32 之外有条件增量。新增增量不足也不能自动证明冗余、编码失败或事件不足。先用三个独立证据链定位问题，不增加新的融合模块。

固定 Phase12 的 Top512、Hallmark富集512、匹配非通路512训练折基因索引及分组。DSS为主终点；PFI为锁定次要终点，必须与 NCI 发布的原 TCGA-CDR PFI逐患者核对，使用同一512队列的正时间有效交集，不扩大队列。

## 分层诊断

1. **条件信息**：RNA-only线性/PCA16；Clinical、S32、Clinical+S32；联合分块正则Cox加入同一RNA。基线正则在inner选择后固定；RNA正则仅5个候选，包含关闭RNA的精确基线。联合拟合系数，不用训练内风险做stacking。RNA对C/S32的无监督残差与heldout风险可预测性只作为冗余诊断。
2. **信息保留**：复用45个既有融合outer-checkpoint；仅重建未保存的45个inner-checkpoint并逐元素核对旧表征。提取初始化/训练后完整8×64 RNA token、均值64D、RNA→Patch完整输出、均值及联合slots。每个producer内训练/验证/测试保持同一坐标系，禁止拼不同模型的latent做meta-training。
3. **RNA-only监督**：同RNA tokenizer与分组，完整token小线性head，分别DSS/PFI训练共90个3-seed×5-fold×3-gene-set管线。与融合模型匹配RNA初始状态；80内层epoch、inner EMA选择、fresh outer refit。旧融合表征在PFI下是DSS producer transfer，单列。
4. **固定probe与排序变化**：所有特征train-local标准化，固定Ridge λ=1。同时交付完整token与mean读出，比较哪些合法生存排序被保留/丢失，分析RNA能否挽救强基线错误排序。完整token低分仅表明当前线性读出不足，不等于信息论上的信息消失。
5. **事件不确定性**：共享患者bootstrap、逐事件影响与有效事件质量；PFI生物终点不同，不能将PFI提升直接归因于事件更多。不人为把事件改成删失，不用事后剔除患者改变主结论。

## 锁定主要比较

DSS三组各自 `Clinical+S32+RNA − Clinical+S32`，以及三组各自 `融合后完整RNA token固定probe − 原RNA512固定probe`；共6个比较×Harrell/Uno60=12项校正。其它编码、pooling、RNA-only训练、条件probe与PFI均次要，保留全部负结果，不事后升级比较。

## 防泄漏与审计

- 所有有监督上游模型仅inner-train训练，inner-val仅用于完整管线预锁选择；outer-test不参与任何选择。固定probe不再利用producer验证标签选择正则。
- 对PFI复用固定基因输入，RNA/Clinical/S32统计重新在PFI有效fit患者计算。DSS-trained冻结producer用于PFI的路径明确标记transfer。
- 父数据、患者折、基因索引、membership、模型/OOF hash锁定；旧产物只读。独立评价器不导入训练损失或指标函数，逐fold与sksurv核对。
- 真实运行外折标签扰动、防跨fold映射、输入RNA响应和padding检查；所有checkpoint回放、完整事件对及bootstrap对象核对后才发布。
- 不保存冒烟预测。正式OOF/checkpoint只在服务器保留；公开代码、匿名表格、完整结论与故障修复记录，新分支 `codex/phase13-rna-information-retention-audit`。

## 优势、风险与判定边界

优势是同输入、同患者折、低容量加性上限与多级表征读出相互补充。风险包括65个DSS事件、反复研究同队列、监督encoder可能训练内过拟合、PFI不是纯样本量实验。不能仅凭相关性高或增量CI含零宣布条件独立；不能仅凭token probe下降判定所有非线性读出均无信息；事件不足只能作为与数据一致的限制解释。

输入与计算核验通过可给高执行置信度；原因归属须依赖完整配对结果，外部泛化和医学机制置信度不能由本队列自行提升。

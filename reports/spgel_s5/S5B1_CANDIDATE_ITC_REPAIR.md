# S5-B1 配对目标忠实性修复与新预算

本记录是正式评价前的静态实现修复。旧配方使用归一化MSE对齐与gene reconstruction，不能作为原A5候选级双向ITC的50基因机制证据。旧源码12文件已先归档至仅含源码的ZIP，SHA256为 `aee67a092f6b4bf5a532694a4a8a5f87d862255eca8f8fa71a6f78a5a1dac377`；旧预正式运行保留成本与superseded谱系，不混入新性能表，也不自动重试原锁。

## 修复后实际算子

影像预测路径、MLP、GraphSAGE、Identity以及QUERY4不变。SPGEL50_ADAPT移除gene decoder、gene reconstruction与配对MSE，保留50个官方训练基因的测量值、ID与mask。两个32×32条件查询矩阵以小幅非零值初始化。

每一个候选(image_i, gene_j)分别以gene_j的单模态上下文更新image query，重新读取image_i的原局部key/value；再以image_i的单模态上下文更新gene query，重新读取gene_j的原基因值与ID证据。四个匿名slot的条件读出以平均余弦形成候选score。反向anchor明确计算(image_candidate_j, gene_anchor_i)，同样逐候选重条件化；不复用已在各自正配对上条件化的embedding。

训练辅助目标是温度0.1的两个anchor方向cross entropy的均值，总训练损失为主50基因MSE加0.1倍该ITC。没有新辅助任务，没有组织teacher/标签，没有额外760基因，不输入DSS/PFI。主要prediction在任何配对计算之前独立得到，数值与梯度均不依赖传入RNA。

基因读取使用线性key/value的精确因式分解，避免保存N×50×32中间证据；与显式逐基因读出已做数值等价测试。仅单模态基础bank可复用，条件pair读出必须重算。

## 有限采样和预算

每epoch从当前训练分区按等间距取至多512个anchor，并按epoch平移一个训练行。每个训练行的至多8个负候选由固定(seed, row)产生，无self、无重复，跨epoch保持该行负候选不变。采样不访问标签值、预测分数、外测试或组织真值。每方向score最大512×9，不构建N×N匹配矩阵。

在新的真实拟合和官方测试之前，统一锁定20个开发拟合、每拟合最多1280轮，在80/160/320/640/1280保存五态；每个outer fold仅用自己的inner-validation五臂平均MSE选择共同epoch。再进行20个全outer-train正式拟合，只保存各自被选epoch一态。100个开发态加20个正式态，共120次实际回放。若1280仍是预算边界最优，继续标记budget edge，不能声称已证明收敛。该预算不由已看过的官方测试分数确定。

反向CV中某来源可以作为另一fold的合法训练数据；保护含义是每个模型自己的outer-test不用于训练或选参。单切片inner空间块没有排除buffer，不能声称消除了空间自相关或实现供体独立验证。

新模型69,042个可训练参数，其中64,242属于image prediction，4,800仅属于训练配对分支。梯度隔离测试确认prediction-only损失不会给任何gene或两向condition参数产生梯度，combined ITC则给两个condition矩阵非零梯度。匿名四slot、50基因范围和无组织监督意味着它仍是明确的HEST适配，不是完整原四程序A5，也不构成BLEEP/TRIPLEX严格复现。

## 冻结前验证

新增定向测试覆盖：因式分解与显式原证据读取；正负候选逐对独立重算；错误positive-conditioned cache反例；两向ITC与手算CE；两向condition梯度；训练采样容量、复现性和排除self；RNA不影响预测；四个非配对算子不变；五个开发epoch集合和120个唯一回放态；参数实际梯度可达性。验证没有使用真实开发曲线或官方测试成绩。

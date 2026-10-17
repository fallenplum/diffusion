# 执行与导出记录

## 正式训练

本轮30次 DIRECT 正式拟合均成功，没有失败或中断；1672个正式优化步骤。另2次资格短拟合共4步，未保存短拟合模型或患者预测。实际新拟合共32次、1676步，不能把60个原P16/S16引用模型算新训练。

全30次新正式模型的训练源均为 `d30dcdc413d57d25aeb041b6ea5ac83835580cad117530ea699d33f35940aaef`。训练期间未修改已锁src/tests/config及训练入口。原S16/P16保留实际D1/D2来源、原checkpoint和receipt，未重新标成新来源。

## 首次正式评估前的输出格式兼容

原Phase18A/20评测函数导出两份summary时使用多行重复列名，而本轮匿名包装要求CSV列名唯一。在首次正式评估尚未开始、等待训练结束期间，仅修改本轮 `validate_phase21.py` 的两份summary导出：读取原四列多级表，改为 `arm,Harrell_mean,Harrell_std,Uno60_mean,Uno60_std`。

- 初稿评估脚本SHA：`4facd765ea63b061eeabcc3693f6be96972be29a2d0c7d88e6ced0640fb415ab`，未用于本轮正式评估。
- 实际正式评估脚本SHA：`8b74d0696f33b7c3c3825eee7fe6f5e53deb5c72d4721709500ea19171648c90`。
- 未修改原Phase18A/20文件、指标公式、风险值、配对检验或训练资格；没有为该格式调整重新训练任何模型。

实际90模型/540向量回放、30个DIRECT的fit/heldout诊断、20k共享患者bootstrap均绑定最终评估脚本SHA。历史2362个tracked文件由独立包装器逐SHA及Git blob校验，结论见PARENT_IMMUTABILITY_AUDIT。

## 时间与GPU口径

EXECUTION_COUNTS中的8497.9653秒来自已完成拟合的内部计时之和，不是并发墙钟耗时或GPU活跃秒。GPU_PROFILE只汇总实际采样，不补造未记录区间；尾部只剩少数单元时不启动额外拟合填充显存。全部inner完成并锁定后才运行outer；全部正式训练闭合后才运行独立全矩阵评估。

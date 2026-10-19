# Phase28 运行边界

## 当前可复核产物

- `INPUT_AUDIT.json`：实际输入、预处理和原H5指纹。
- `DEPENDENCY_QUALIFICATION.json`：正常官方CUDA导入及选择性扫描参考校验。
- `NUMERICAL_QUALIFICATION.json`：五臂完整模型梯度缓存资格。
- `SYNTHETIC_TESTS.json`：162项合成测试统一运行回执。
- `SOURCE_LOCK.json`：训练源码闭包及本轮资格报告SHA。
- `IMPLEMENTATION_REVIEW.json`：独立静态审查与回执复核，明确未执行正式训练。

正式运行依赖服务器已有的私有数据、冻结风险及pinned历史方法源码。单独复制新目录不能绕过继承来源校验；本机缺少历史pinned baseline文件时会拒绝给出正式来源资格，不使用替代实现。

## 未来内部矩阵

仅在进入30拟合矩阵后使用。下例是其中一个固定单元，不代表该单元已运行：

```bash
/root/autodl-tmp/phase28_reread_private/env/bin/python \
  /root/phase16_routing_repo/scripts/run_phase28.py \
  --seed 31 --fold 1 --arm C11 \
  --source 0ccd4897d30e654ebcb3f55d4891a3b76312f5e6a28dcc30070520000fc5372b \
  --execute-formal
```

矩阵为seed `31,47`，内部划分`1,2,3`，结构`C00,C10,C01,C11,DIRECT`。每次更新使用完整fit风险集；不增删单元、不以内部验证选择80轮中最优epoch。

中断仅可用相同源码和预处理指纹显式`--resume`从最近十轮私有optimizer状态恢复。不得把已存在的部分目录当作新随机重启，也不得把PROGRESS当作DONE。

并发数需要按真实最大切片尺寸的峰值显存与吞吐测量确定。合成单样本峰值不能直接推导并发上限；本轮没有执行多训练进程吞吐试验。

## 回放与评测顺序

1. 全部30份`DONE_PRIVATE.json`、80轮记录及checkpoint SHA完整。
2. 显式执行`replay_phase28.py --stage prototype --source ... --execute-replay`，先核当前资格与完整输入指纹，再独立加载模型和原始数据，复算300个新风险向量及90个原主效应参考向量。
3. 独立回放通过才可执行`evaluate_phase28.py --stage prototype --source ... --execute-evaluation`。评测入口用唯一私有attempt标记管理一次验证结局访问；失败不能自动重试读取结局。具体必需参数以各脚本`--help`为准。
4. 汇总规则、比较、修复/损伤及停止条件均按`PROTOCOL.md`。本轮没有自动外折入口。

患者风险、fit/valid索引、供体映射、模型和optimizer状态、H5路径及原坐标只留私有目录。Git只发布源码、配置、匿名报告和SHA清单。

## 语义与辅助项

当前仅实现第一阶段核心。后续辅助Cox、遮蔽恢复与文本桥需另立配置、来源与对照，不能把当前配置中的零值改为非零就声称已实现这些损失。无原WSI/独立形态标签时，TITAN或CONCH分数仍是教师代理。

# 重算本轮合成机制实验

只需 Python、PyTorch、NumPy、SciPy、pytest；使用包含参考父提交的完整仓库。基础特征提取器和真实患者数据不是运行本轮合成实验的依赖。GPU 执行环境为 PyTorch 2.5.1+cu124，全部模型 FP64，无 AMP/TF32。

1. `python -m pytest tests/rscil -q`：数值、参考脚本对齐、真实更新、测试标签隔离、选轮门槛及隐私范围测试。
2. `python scripts/rscil/reference/synthetic_gate.py --reps 12 --out reports/rscil/reference/recomputed_linear_results.json`：原固定线性参考实验；它不是神经机制验证。
3. `python scripts/rscil/launch_matrix.py --output <RUN_DIR> --workers 16 --device cuda`：预锁 96 个单元；每单元 5 臂，每臂内部选择与完整开发集重新拟合各一次。执行前验证 configs/rscil/run_lock.json 的科学源码闭包，不要改锁再追求已有分数。
4. `python scripts/rscil/audit.py --input <RUN_DIR> --output reports/rscil`：核验完整覆盖、训练有效性和回放后，调用预锁 protocol.synthetic_gate。若提供 `--retired <RETIRED_V1_DIR>`，另列初版实际成本与新旧相同性；初版结果不进入主裁决。

本次本机/服务器测试的实际凭据为 SOFTWARE_TEST_LEDGER.json；audit.py 验证该凭据及参考重算结果后才输出理论资格。该凭据是本次执行记录，不代表别的机器自动通过；复现实验应生成自己的实际测试凭据。

输出目录必须独立；程序拒绝覆盖已有完成单元。调度器复用已有相同源锁的完整单元，不把复用计作新增拟合。若 source lock 不一致或快照回放失败，应停止修复软件问题，保留旧执行记录，不借机改变科学门槛。

所有合成教师仅拟合当前 fit，选轮仅用内部验证 Cox；真正测试结局没有进入训练函数。A 每轮由当前 fit 风险集求得，快照保存前重算，测试只使用保存的 A。每一快照对训练和对应评估分区进行临时磁盘保存、重新载入和预测校验；验证后删除临时权重。

预留的 TCGA 入口不是完成的真实队列训练器。`run_inner.py` 在合成门槛前拒绝执行；`run_outer.py` 在内层门槛前拒绝执行。若本轮为 SYNTHETIC_NO_GO，不执行这些入口，不创建虚构的内外层结果文件。

原始输入、源锁、完整重复结果和负例均保留。本轮报告没有患者 OOF、权重、H5、账号或机器路径；复用真实项目历史成绩时应以其原研究合同解释，不能代替 fresh S0。

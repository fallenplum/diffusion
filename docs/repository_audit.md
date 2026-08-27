# Repository audit (2026-08-27)

本审计只记录仓库工程事实，不修改 Phase P1 的科研数值、endpoint、outer-fold 或 seed。

## 代码与依赖

扫描 `src/` 后确认：

- 数值与表格：`numpy`、`pandas`、`scipy`、`scikit-learn`。
- 原始 H5/坐标读取：`h5py`。
- 图表：`matplotlib`。
- 神经模型、CUDA 和 checkpoint：`torch`。
- 新增自动化入口：`PyYAML`、`jsonschema`。
- `argparse` CLI 已存在于审计、B0、neural、permutation、posthoc 和多个 baseline 脚本。

正式运行依赖收录在 `requirements.txt`，CPU CI 依赖收录在 `requirements-ci.txt`。当前普通系统 Python 3.13 缺少 SciPy/PyTorch；本机已核验的研究环境为 Python 3.11.14 + PyTorch 2.4.1+cu121，CUDA 12.1，GPU 为 RTX 4060 Laptop GPU。这个本机 GPU 环境不等于 GitHub self-hosted runner 已注册。

## 数据与输出

- Phase P1 源码读取 `TCGA/` 下的 bags/H5 和既有审计输出；根目录 `data/` 不作为该阶段输入。
- 患者数据、H5/NPY/NPZ、checkpoint、embedding、缓存和大输出必须留在本地，并由 `.gitignore` 与 CI guard 双重拦截。
- 可审阅的 Phase P1 结论和轻量审计汇总继续保留在 `outputs/pathology_context_phase1/`。
- 自动化运行写入被忽略的 `outputs/automated_runs/<experiment_id>/`，只通过 Actions artifact 返回。

## 当前缺口与修复

原 CI 只做 compile/import，不能发现模型 forward、mask 边界、患者级划分和 artifact 协议错误。本分支新增 CPU 合成测试、request schema、统一 runner、GPU preflight、self-hosted workflow、失败 artifact 上传和文档。

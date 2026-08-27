# Runtime environment

## 正式环境

安装正式运行依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 覆盖当前 `src/` 实际 import 的 NumPy、Pandas、SciPy、scikit-learn、h5py、Matplotlib、PyTorch、PyYAML 和 jsonschema。PyTorch 只声明兼容版本范围；CUDA wheel 必须由目标机器/环境自行选择，不能从仓库文件推断或下载错误的 CUDA 版本。

## 已核验本机研究环境

| 项目 | 已观测值 |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.4.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| CUDA available | true |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

审计时该环境尚未安装 `h5py`、`pytest`、`ruff` 和 `jsonschema`；安装上面的依赖后才可视为完整运行环境。系统默认 Python 3.13 当时也没有 SciPy/PyTorch，不能用它声称实验可运行。

## CI 与 GPU 实验的边界

- GitHub-hosted CPU CI 使用 Python 3.11、官方 CPU PyTorch wheel、合成数据和轻量测试；不会下载 TCGA/UNI2-h 数据或模型。
- 正式 `mode: full` 只能在标记为 `self-hosted,diffusion,gpu` 的 runner 上执行，并由 `DIFFUSION_DATA_ROOT` 提供本机数据根目录。
- GPU preflight 会检查 Python、PyTorch、CUDA、GPU 名称、数据环境变量、所需目录、磁盘空间、写权限和 Git SHA；条件不满足时返回非零状态。

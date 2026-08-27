# Local data layout

实验 request 只写环境变量名，不写真实绝对路径。默认变量为：

```text
DIFFUSION_DATA_ROOT
```

运行机器应提供类似下面的**逻辑布局**（下方不是需要复制的真实路径）：

```text
<DIFFUSION_DATA_ROOT>/
└── TCGA/
    ├── <patient-or-archive>/
    │   └── *.h5
    └── uni2_h_bags/
        └── <patient>/<slide>.npy
```

Phase P1 的坐标必须来自原始 H5 或可信导出索引，不能使用数组顺序伪造规则网格。`data/` 目录不是该阶段的输入源，且仓库不跟踪 `TCGA/`、`data/`、H5/NPY/NPZ 或 checkpoint。

正式 request 可以声明 `required_directories: [TCGA]`；preflight 只检查目录存在性，不把数据根目录值写进报告或日志。数据许可、访问控制和备份由运行机器管理员负责。

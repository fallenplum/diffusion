# 自动科研 request 协议

`automation/requests/*.yaml` 是一次实验的可审计入口。它们描述实验身份、代码基线、配置文件、运行模式、数据环境变量、seed/fold 和 artifact 名称；**不允许包含真实患者数据路径、数据内容、secret 或 runner token**。

## 最小生命周期

1. 在分支上创建或修改 request。
2. CPU CI 检查 schema、源码、测试和受保护文件。
3. 同仓库 PR（禁止 fork PR）触发 self-hosted GPU workflow。
4. runner 读取本机 `DIFFUSION_DATA_ROOT`，先执行 preflight，再调用 `scripts/run_research_experiment.py`。
5. 运行目录写入 `outputs/automated_runs/<experiment_id>/`；该目录被 `.gitignore` 忽略，只通过 Actions artifact 返回。
6. ChatGPT 根据 `run_state.json`、`summary.json`、`metrics.csv` 和日志判断下一步。

## 模式边界

- `mode: smoke` 只允许合成数据或极小链路验证，结果必须标记 `result_class: smoke`，不能进入论文正式结果。
- `mode: full` 必须声明 `requires_patient_data: true`，由运行机器提供数据环境变量，并且必须使用明确的无 shell command；结果标记 `result_class: primary`。

## 文件协议

每个运行目录至少包含：

```text
run_state.json
experiment_manifest.json
metrics.csv
summary.json
SUMMARY.md
logs/
preflight_report.json
```

失败也应尽量留下上述轻量状态。大型 checkpoint、embedding、H5/NPY bags 不得进入 artifact。

schema 位于 `automation/schemas/experiment_request.schema.json`，统一执行器位于 `scripts/run_research_experiment.py`。

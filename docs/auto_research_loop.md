# Automated research loop

本项目的控制平面是代码、配置、request、运行状态和 Actions artifact；原始患者数据与大型模型文件留在本地 GPU 机器。

## 标准流程

1. ChatGPT 读取当前代码、配置和上一轮轻量结果。
2. 在 `codex/auto-research-infra` 或后续实验分支上修改代码/config。
3. 新建 `automation/requests/<experiment>.yaml`，记录代码基线、配置、模式、seed/fold 和 artifact 名称。
4. 创建 Pull Request；CPU CI 自动执行 ruff、compileall、pytest、dry-run 和 protected-data guard。
5. 对同仓库 PR，`research-experiment.yml` 选择唯一 request 并排队到 `self-hosted,diffusion,gpu` runner。
6. runner 先执行 GPU/data preflight，再调用统一 runner。
7. runner 在 `outputs/automated_runs/<experiment_id>/` 写入 `run_state.json`、`experiment_manifest.json`、`metrics.csv`、`summary.json`、`SUMMARY.md` 和日志。
8. workflow 使用 `if: always()` 上传状态、preflight、日志和轻量 metrics；失败也保留 traceback。
9. ChatGPT 读取 workflow/job/log/artifact，基于不确定性判断 hypothesis 是否得到支持。
10. 必要时建立下一轮分支/request；研究方向、昂贵实验监督和最终 merge 保留 human-in-the-loop。

## 模式边界

`smoke` 只使用合成数据或极小链路，`summary.json` 的 `result_class` 必须为 `smoke`，不得写入 Phase P1 正式结果。`full` 必须明确数据环境变量和无 shell 的 repository-relative command，`result_class` 才为 `primary`。

## 失败与恢复

状态协议为 `PENDING`、`RUNNING`、`FAILED`、`PARTIAL`、`COMPLETE`。失败时仍写状态和 artifact；下一轮使用：

```bash
python scripts/run_research_experiment.py --request automation/requests/<experiment>.yaml --resume
```

如果 request、代码 SHA 或配置已改变，应新建 experiment id，而不是覆盖旧运行目录。

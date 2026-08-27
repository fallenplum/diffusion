# Self-hosted GPU runner

仓库已准备 workflow，但**没有也不会保存 runner registration token**。注册必须由有仓库管理权限的用户在 GitHub 临时完成。

## 注册步骤

1. 打开仓库 `Settings` → `Actions` → `Runners`。
2. 点击 `New self-hosted runner`，选择目标操作系统和架构。
3. 按 GitHub 页面显示的官方命令在 GPU 机器上下载、解压并配置 runner。
4. 配置 labels 时保留默认 self-hosted，并额外加入：`diffusion`、`gpu`。
5. 将 `DIFFUSION_DATA_ROOT` 配置为 runner 服务可见的机器环境变量；不要把真实路径写入 request、YAML、日志或仓库。
6. 以服务方式启动 runner，并在仓库的 Runners 页面确认状态为 Online。

注册 token 是一次性/短期凭据，不要提交、截图、粘贴到聊天或写入 GitHub Actions Secrets。若 token 泄露，应立即在 GitHub 重新生成并废弃旧 token。

## 运行前检查

在 runner 工作目录执行：

```bash
python scripts/gpu_preflight.py --require-gpu --require-data --required-dir TCGA
```

缺少 CUDA、数据环境变量、`TCGA` 目录、磁盘空间或写权限时，preflight 必须失败。workflow 只允许同仓库 PR 使用 GPU runner，fork PR 会被安全条件拦截。

## 安全边界

- workflow 使用 `contents: read` 和 `persist-credentials: false`。
- workflow 不自动安装/下载患者数据、不提交实验结果、不保存 PAT/HF/SSH/runner token。
- 第一个版本只上传 Actions artifact，避免“Action commit → 再触发 Action”的循环。
- self-hosted runner 应使用专用账户、最小目录权限和不含个人凭据的工作区；合并 PR 前由人工审查昂贵实验。

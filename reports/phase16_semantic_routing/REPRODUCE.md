# 复现

起点04150dc976057f825702d5fe3502696060790dd7；分支codex/phase16-semantic-roi-routing。需已有Phase11–15私有只读资产与原始H5；公开仓库不包含患者数据/权重。

```bash
export PYTHONPATH=/root/phase10_trimodal_eval_deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/root/miniconda3/bin/python
$PY scripts/audit_phase16.py
$PY scripts/prepare_phase16.py
$PY scripts/coordinate_audit_phase16.py
$PY scripts/pin_phase16_inputs.py
$PY scripts/launch_phase16.py --pilot
$PY scripts/execute_phase16.py
$PY scripts/pin_phase16_inputs.py
$PY scripts/reconcile_phase16_scope.py
$PY scripts/report_phase16.py
$PY scripts/publish_phase16.py
```

原始H5只读，prepare缓存易失；已有同源cells可恢复，但输入哈希/协议不符时报错。所有新的正式outer fit有inner/outer checkpoint和实际ROI索引；仅正确完成且独立验收后发布。Pilot只存资源/梯度摘要，不存探索性预测或权重。

访问权生效后应作为新的预注册执行批次，先做官方projection/text功能核验、实际Prompt sanity和1seed×2fold数值pilot，再启动语义/G3；本分支不能伪装已完成这些实验。G4必须补原始WSI与30组原图QA。

一键重建当前受限范围：`/root/miniconda3/bin/python scripts/pipeline_phase16.py`；加`--publish`仅在完整验收后提交公开清单。该命令不会解锁被模型授权或原图QA阻塞的分支。

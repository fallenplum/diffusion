# 复现

从74b7885分支继续，使用相同私有数据、Phase12预处理、Phase16候选H5/坐标映射、已授权CONCH-Z权重。仓库不含患者数据及受限权重。

```bash
export PYTHONPATH=/root/phase10_trimodal_eval_deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
/root/miniconda3/bin/python scripts/pipeline_phase16_diagnostic.py
```

CPU RNA与双视觉拟合可并行，GPU只做冻结文本/图像投影推理；已有source相同的Cox cell恢复复用。查询审计会精确重建重启后丢失的/dev/shm视觉缓存。独立审计后才生成报告。原图不在前置条件中，明确不声称已完成原图QA。逐折OOF及患者表位于/root/phase16_prompt_rna_diagnostic_private，不发布。

主比较和日程在两个config中锁定；直接DUAL64与条件CS的补充配对是读取主分析点估计后定义，见SUPPLEMENTARY_CONTRAST_SCOPE.json，属于探索性分析。

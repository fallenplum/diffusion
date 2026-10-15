# 复现

从dd559068c61fe42f5764a8e080d5889ad135ba51继续；需要相同私有512患者数据、Phase12/15/16检查点与缓存。仓库不含患者数据或权重。

```bash
export PYTHONPATH=/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/root/miniconda3/bin/python scripts/pipeline_phase16r_light_rna.py
```

私有输出/root/phase16r_light_rna_private；训练程序仅同源缓存恢复，source不匹配即拒绝。官方CONCH-Z权重本轮无需加载。若重启导致/dev/shm/phase16_routing_full.npy丢失，先按上一轮query审计脚本重建并核验原始cache哈希，再执行。

15个单元每个19输入×两日程，另含机制控制、自适应compact和四类历史引用；固定λ1的非错患者模型跨seed明确复用。C4引用真实历史OOF，不声称重新训练C4或将其变成凸模型。patient文件、基因索引、逐患者风险、模型参数全部私有。

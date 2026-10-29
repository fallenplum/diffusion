# SPGEL 源任务 pilot：复现与交付边界

本页给出固定配方的可执行入口；真实完成量以 `S3_PILOT_SUMMARY.json`、逐单元 COMPLETE 和实际 REPLAY 为准。官方 HEST 50-HVG、新独立来源和论文强方法尚未运行，不能用本 pilot 替代。[数据资格](S0_DATA_REGISTRY.md)、[完整边界](S0_DELIVERY_BOUNDARY.md)。

## 1. 环境与只读资产

参考执行环境为 Python、PyTorch 2.5.1+cu124、NumPy、SciPy、scikit-learn、h5py。冻结 UNI2-h 1536 维输入逐行 L2；所有训练折矩、程序阈值和基因尺度仅从当前训练供体计算。source 的资产 JSON 仅有 `cache_root` 与 `fresh_source_receipt` 两个路径字段；它不得包含 TCGA、DSS、PFI 或 R8 预测的路径。

私有缓存需要 `manifest.private.json`、`source_arrays.private.npz`、`gene_counts.private.npz`。其 SHA 和原始 H5AD 全测量库的独立重核收据由 [S0_TASK_CONTRACT.json](S0_TASK_CONTRACT.json) 固定。760 列是程序基因并集缓存，CP10K 分母必须来自原始 36,601 列测量库，不能将缓存行和误当成全库总数。

下面的环境变量由有权限的使用者在本机设置，示例不包含服务器地址、账号、凭据、真实患者/供体路径。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
# SOURCE_ASSETS: 两字段的私有资产 JSON
# RUN_PRIVATE: 全新、非 Git 管理的私有输出目录
python -B scripts/spgel_source_qualify.py \
  --assets "$SOURCE_ASSETS" --private "$RUN_PRIVATE" \
  --config configs/spgel/pilot_v1.json --device cuda
```

资格阶段运行真实前向/反向且不创建 optimizer，不改变参数；随后固定 `SOURCE_RUN_LOCK.json`。任何真实失败都应保留，不能覆盖原输出或自动重试。报告器预算 schema 的一次修复只影响交付验证，旧零拟合资格/锁保留为 superseded，正式训练使用新的独立输出目录和新锁，模型配方没有改变。

## 2. 72 个固定 pilot 单元

以下串行循环是可复现的最小入口，实际实验使用有资源限额的行政并行调度；两者调用同一冻结 `fit_cell`，不改变 minibatch、种子、epoch 或候选集合。

```bash
for donor in 0 1 2 3 4 5; do
  for seed in 31 47; do
    for arm in S_A0 S_A1 S_A2 S_A3 S_A4 S_A5; do
      python -B scripts/spgel_source_fit.py \
        --assets "$SOURCE_ASSETS" --private "$RUN_PRIVATE" \
        --config configs/spgel/pilot_v1.json --stage pilot \
        --arm "$arm" --heldout-donor "$donor" --seed "$seed" --device cuda
    done
  done
done
python -B scripts/spgel_source_replay.py \
  --assets "$SOURCE_ASSETS" --private "$RUN_PRIVATE" \
  --config configs/spgel/pilot_v1.json --stage pilot --device cuda
```

12 个 S_A0 是真实 L2 logistic 凸拟合与训练组织程序均值表，只有一个 solver-final state；60 个神经拟合各 40 epochs，保存并实际回放 10/20/40 三个状态。总计 52,000 次神经 optimizer updates、192 个保存状态。主评价固定为 epoch40；epoch10/20 不用于 TASK-B，不能复制 epoch40 的检索分数填入。

TASK-A/C 使用全部留出 spots，预测器只接收图像、相对几何和有效性。TASK-B 允许配对 ST：同片全部 spots 为候选，每片最多 128 个按固定行索引选出的 query；须把所有方法的共同四分数检索与 A3–A5 的 native paired 检索分表。常数目标、无高活性阳性或负例的指标保留 null 与支持数。

## 3. 全源训练与冻结接口

只有 72/72 拟合及 192/192 实际状态回放全部通过后，才执行两个预先固定的全源导出拟合。不能根据 R8 的 DSS 数值选 A1 或 A5。

```bash
for arm in S_A5 S_A1; do
  python -B scripts/spgel_source_fit.py \
    --assets "$SOURCE_ASSETS" --private "$RUN_PRIVATE" \
    --config configs/spgel/pilot_v1.json --stage s4 \
    --arm "$arm" --heldout-donor -1 --seed 31 --device cuda
done
python -B scripts/spgel_source_replay.py \
  --assets "$SOURCE_ASSETS" --private "$RUN_PRIVATE" \
  --config configs/spgel/pilot_v1.json --stage s4 --device cuda
python -B scripts/spgel_source_export.py \
  --private "$RUN_PRIVATE" --config configs/spgel/pilot_v1.json
python -B scripts/spgel_source_report.py \
  --private "$RUN_PRIVATE" --config configs/spgel/pilot_v1.json \
  --output "$EMPTY_PUBLIC_REPORT_DIR" --include-s4
```

S4 增加 2,080 次更新、6 次状态回放。全部预算为 74 次 optimizer 拟合、54,080 次神经更新和 198 次保存状态回放。导出 `encoder_state.pt`、训练侧 scalers 和基因顺序留在私有目录；公开的是检查点/接口 SHA、支持数和匿名聚合指标。

`FrozenSpatialProgramInterface` 的图像接口输出 `[B,N,4,128]`。A1 不具备学习到的程序 slots，其复制四份的基线 embedding 不能称为四种医学语义。A1 的分子/条件路由接口明确拒绝。A5 的 `route_bulk_to_wsi` 接收 `[B,N,128]` 的预编码 patch values，不能把图像接口的四程序 tensor 不经资格适配就直接串接。

`xy_um` 必须由调用者确认是微米；API 不能自行识别被误标成微米的像素。源 112μm 图像上下文、Visium 捕获区域和 TCGA patch 取材不同，见 [几何清单](source_to_tcga_geometry_manifest.json)。Bulk 基因输入要求当前 R8 训练侧的独立 gene-Z、基因顺序及测量 mask，不会自动应用 ST scaler。

当前 R8 保持 `R6_S7_locked`。未来使用这些新权重必须另立 `R8_SOURCE_SPGEL_V1`，重新资格并单独报告；本次 S4 导出不等于完成该新生存实验。

公开报告采用无损分片：冻结报告器产生的 4,491,342 字节原件完整保留在私有档案；公开主 JSON 保留全部72个供体单元、指标、配对和计数，将192条检查点诊断放在单独的 `S3_CHECKPOINT_DIAGNOSTICS.json`。新 schema 和显式引用说明这一存储格式，避免超过既有单文件2MiB发布上限。下面的行政入口能逐字节还原原报告，并核对原件 SHA；没有删除指标或重训模型。

```bash
python -B scripts/shard_spgel_public_report.py \
  --original "$ORIGINAL_PRODUCER_REPORT" --output "$EMPTY_SHARDED_REPORT_DIR"
python -B scripts/shard_spgel_public_report.py \
  --output "$EMPTY_SHARDED_REPORT_DIR" --verify-only
```

公开版恢复完整原字典时，调用 `load_original_public_report(report_directory)`。`S3_LOSSLESS_SERIALIZATION_AUDIT.json` 记录原件、分片和语义哈希；独立聚合核查针对实际公开主表再次执行，1,316项算术比较完全一致。

## 4. 核查、资源与公开范围

冻结源模型/合同/损失/训练/接口与报告器共 46 项测试通过，其中报告器 17 项测试包含真实固定 config、更新总量、逐供体更新和额外 budget 键篡改拒绝。测试通过与真实拟合完成是两种证据。

行政并发可按实测吞吐和 cgroup 限额调整，不能改变 74 个预锁单元，也不能重启运行中的拟合伪造加速。应保留每 PID、完成件 SHA、时间、实际更新、CPU/RAM/GPU 和故障/交接记录；显存占用率不能代替吞吐比较。

公开采用逐文件明确允许清单与真实 TCGA/ST 身份拒绝清单。禁止公开风险向量、患者或 spot 级数组/坐标、表达、权重和凭据。原始/派生数据与模型的原许可继续适用，HEST 数据须遵守其研究使用约束。六个历史开发供体不是六份新的外部确认，旧四份 ST-test 也不能改称从未见过的外部测试。新数据按用户要求写入 E 盘；E 盘未就绪时不改到其他盘下载。

# Phase M3R2 Simple-Fusion Decomposition & Clinical Audit

本报告严格使用 512 patients / 65 DSS events、同一 5-fold、同一 fold-local preprocessing、DSS/Cox、K=1024 与三个固定 seeds。global OOF C-index 仅作诊断，不参与模型选择。

## 绝对指标

| Variant | Harrell mean ± std (15 cells) | Uno mean ± std (15 cells) | per-seed 5-fold Harrell means | per-seed 5-fold Uno means | worst fold H / U |
|---|---:|---:|---|---|---:|
| S0 ClinicalOnly | 0.7783 ± 0.0644 | 0.7792 ± 0.1126 | 0.7688, 0.7770, 0.7890 | 0.7643, 0.7700, 0.8034 | 0.6014 / 0.5568 |
| S1 SimpleCAMR-FullReg | 0.7740 ± 0.0612 | 0.7978 ± 0.0507 | 0.7538, 0.7755, 0.7929 | 0.7757, 0.7955, 0.8221 | 0.6735 / 0.7090 |
| S2 SimpleCAMR-NoAdv | 0.7904 ± 0.0574 | 0.7786 ± 0.0723 | 0.7840, 0.8017, 0.7855 | 0.7694, 0.8131, 0.7532 | 0.7054 / 0.6471 |
| S3 SimpleCAMR-CommonOnly | 0.7954 ± 0.0561 | 0.8027 ± 0.0569 | 0.7890, 0.7754, 0.8218 | 0.8095, 0.7710, 0.8277 | 0.6939 / 0.7076 |
| S4 RawEncodedConcat | 0.7800 ± 0.0763 | 0.7621 ± 0.0866 | 0.7779, 0.7801, 0.7819 | 0.7539, 0.7785, 0.7539 | 0.6881 / 0.6129 |
| S5 SimpleCAMR-NoClinical | 0.6277 ± 0.0867 | 0.6522 ± 0.0941 | 0.6078, 0.6348, 0.6404 | 0.6594, 0.6377, 0.6594 | 0.4935 / 0.4806 |

## Paired comparisons (target − baseline)

| Comparison | ΔHarrell | ΔUno | Harrell wins /15 | Uno wins /15 | both wins /15 |
|---|---:|---:|---:|---:|---:|
| S1 vs S0 | -0.0042 | +0.0185 | 7 | 8 | 6 |
| S1 vs S2 | -0.0164 | +0.0192 | 4 | 8 | 4 |
| S1 vs S3 | -0.0214 | -0.0050 | 6 | 9 | 5 |
| S3 vs S4 | +0.0154 | +0.0406 | 10 | 10 | 8 |
| S1 vs S5 | +0.1464 | +0.1456 | 15 | 14 | 14 |

## 六个问题的结论

1. **simple-CAMR 是否稳定超过 current-protocol clinical-only？** 否，未达到预设确认门槛。 S1 vs S0 的两项均值差与双指标 wins 计数见上表；确认门槛要求两项均值差为正且两项各至少 9/15 胜。
2. **shared/specific decomposition 是否真正有贡献？** 由 S1 vs S3 的 paired Δ 与 wins 判断；S3 是去掉 specific、reconstruction、orthogonality、adversarial 后的 common-only 对照，因此该比较同时检验这些附加机制的联合贡献，不能拆成单一模块因果效应。
3. **adversarial alignment 是否真正有贡献？** 由 S1 vs S2 判断；S2 仅关闭 adversarial loss，其余结构和协议保持一致。
4. **reconstruction/orthogonality 是否真正有贡献？** 本协议没有单独的 reconstruction-only 或 orthogonality-only variant；只能从 S1 vs S3 得出联合正则化分支的证据，不能声称两者各自被识别。
5. **plain encoded concat 是否已经足够？** 由 S3 vs S4 判断；S4 去掉 common/specific decomposition，保留相同 upstream 编码、pooled concat、clinical 与匹配 head/训练协议。
6. **下一阶段能否进入 M4？** 只有当 S1 vs S0 达到确认门槛且机制对照支持明确基础时才可进入；本阶段不实现 M4。最终判断以本报告表格中的固定结果为准。

## 审计边界、优势与风险

- 优势：同一患者划分、预处理、端点、Cox、K 与固定 seeds；S1–S4 的 shared backbone 初始化进行了字节级审计；paired comparison 使用同 seed/fold 配对。
- 风险：S1 vs S3 将 specific、reconstruction、orthogonality、adversarial 一起移除，不能把联合差异归因到其中某一个模块；GPU 中仍可能出现官方 PyTorch 对 memory-efficient attention 或 adaptive pooling 的确定性警告。
- 置信度：协议执行与统计汇总为高；对单个机制的因果归因仅为中，因为本阶段没有把 reconstruction 与 orthogonality 分开。

## 复现入口

代码和七个规定输出文件位于本目录；运行缓存、cell 预测和 checkpoint 不纳入 Git 推送。

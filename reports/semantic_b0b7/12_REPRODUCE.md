# 复现入口

源码运行需Python、PyTorch2.5.1/cu124、numpy2.1.3、scipy1.14.1、scikit-learn1.5.2与h5py；官方CONCH加载使用现有Transformers5.18/timm1.0.30环境，TITAN worker使用原Transformers4.46/timm1.0.3环境。模型、canonical数据、私有测量mask及原切片均需合法持有，不随Git交付。

以下变量指向执行者自己的合法私有目录。原数据保持只读，输出必须是新目录。两模型权重及源码清单首先按SHA核验；不在新环境直接忽略missing/unexpected keys。

```bash
python scripts/semantic_b0b7_audit.py --assets "$ASSETS" --legacy-root "$LEGACY_ROOT" --titan-cache "$TITAN_CACHE" --conch-model "$CONCH_MODEL" --titan-model "$TITAN_MODEL" --output "$PRIVATE/P0_DATA.json"
python scripts/semantic_b0b7_prepare.py --assets "$ASSETS" --legacy-root "$LEGACY_ROOT" --titan-cache "$TITAN_CACHE" --conch-model "$CONCH_MODEL" --titan-model "$TITAN_MODEL" --titan-python "$TITAN_PYTHON" --concepts configs/phase6a_titan_crc_morphology_concepts.json --prior configs/semantic_b0b7/allowed_relations_v1.json --output-newdir "$PRIVATE/prepared" --device cuda
python scripts/semantic_b0b7_gate.py --inputs "$INPUTS" --expected-input-sha256 "$INPUT_SHA" --data-audit "$PRIVATE/P0_DATA.json" --alignment-audit "$PRIVATE/prepared/COMPLETE.private.json" --output "$GATE" --device cuda
python scripts/semantic_b0b7_p1.py profile --inputs "$INPUTS" --expected-input-sha256 "$INPUT_SHA" --gate "$GATE" --output "$PROFILE" --device cuda --threads 2
python scripts/semantic_b0b7_p1.py baselines --inputs "$INPUTS" --expected-input-sha256 "$INPUT_SHA" --gate "$GATE" --private-root "$FITS" --threads 2
```

对seed=31/47/83和arm=B3/B4/B6a/B4_MAP_SHUFFLE各执行一次下面单元，共12拟合。并行前应对训练及验证同时驻留的实际峰值进行计时；本次使用6进程、每进程2线程。

```bash
python scripts/semantic_b0b7_p1.py cell --inputs "$INPUTS" --expected-input-sha256 "$INPUT_SHA" --gate "$GATE" --profile "$PROFILE" --private-root "$FITS" --arm "$ARM" --seed "$SEED" --max-epochs 400 --device cuda --threads 2
python scripts/semantic_b0b7_p1.py report --inputs "$INPUTS" --expected-input-sha256 "$INPUT_SHA" --gate "$GATE" --private-root "$FITS" --output "$AGGREGATE" --extension-output "$EXTENSION_GATE"
python scripts/semantic_b0b7_chance.py --inputs "$INPUTS" --expected-input-sha256 "$INPUT_SHA" --output "$CHANCE"
python scripts/semantic_b0b7_checkpoint_diagnostics.py --inputs "$INPUTS" --expected-input-sha256 "$INPUT_SHA" --aggregate "$AGGREGATE" --private-root "$FITS" --output "$DIAGNOSTIC"
```

只有共同extension gate触发时，全部12单元才以`--resume --max-epochs 800 --extension-gate "$EXTENSION_GATE"`续训；本次0/12触发条件，未续训。各保存状态均真实重新载入做验证前向；RNG保护保证回放不改变后续训练流。

公开普通JSON分片和index可用semantic_b0b7_report.py中的restore函数重建原14,079,334字节聚合报告，并核验原字节SHA及canonical语义SHA。训练集原始数组、逐患者预测、优化器和权重保持私有；独立verify_semantic_b0b7_p1.py只做收据/字节/算术核对，不声称额外前向。

## 下一轮矩阵与预算边界

下一轮仍为B0–B7主矩阵：B0精确重建强锚点，其余7臂原5外折×3seed最多105个最终主单元、3内层最多315个主选择单元。B2容量控制、B4乱关系、B5临床等容量和B6b真实错交互重训等额外单元另计，不算免费。共享无DSS表示只计一次真实预训练。

本次相同规模400epoch的单worker训练循环实测84.33–93.50秒，六并发完整12单元队列约199.58秒。在完全相同272/137规模和并行条件下，420独立同类单元的计算参考约1.64–1.82小时；这不是P2总体耗时承诺。409人outer-final、额外候选、强锚点、数据读取及新控制须另profile。400与一次800扩展的预算上界分别为168000/336000主表示更新，尚未实际执行。

新DSS风险头、完整强方法输入与背景残差必须通过额外资格后再运行。不能把P1单分区checkpoint跨用到包含其训练患者的其他外折测试。未来评估遵循2026-10-31用户补充：不计算CI，以理论、同协议点增量、方向和机制控制判断。

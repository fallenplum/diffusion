# NEXT_DECISION

## 决策：Decision C（当前带坐标 spatial-SSL 核心假设不成立）

### 证据

1. 预注册主 pipeline P1/P2/P3 的 Harrell C（0.5995/0.6118/0.5985）均低于 B0=0.6600；P2/P3 的 Uno 虽为 0.6758/0.6706，但 paired bootstrap 95% CI 仍跨 0。
2. 与 architecture-from-scratch 的 B2 比较没有形成双指标稳定收益；P0/P1/P2 的 end-to-end Harrell 也没有超过 B0。
3. P0_nocoords 是当前最强的冻结 context ablation，但它不是原始带坐标机制的确认；三 seed 的点估计不能替代独立验证。
4. P2 置乱 OOF Harrell/Uno=0.5140/0.4427，支持标签置乱 sanity check 正常，未提示明显泄漏。

### 不应做的下一步

- 不继续在同一外层 split 上堆叠 mask ratio、Transformer 层数、loss 或 seed 以追求更高点估计。
- 不把 P0_nocoords 的结果改写成 spatial coordinate hypothesis 的正结果。
- 不将当前 risk-only 输出包装成 IBS 或校准结论。

### 最小可证伪后续

若仍要研究 context，下一轮应预注册一个新的验证切分，固定当前 encoder/采样预算，直接比较：

`B0 mean` vs `B1 ABMIL` vs `P0_nocoords frozen probe` vs `P0_coords frozen probe`。

重点诊断全局形态信息是否被 1024-token sampling 丢失、coords 归一化是否引入噪声，以及在更多 DSS event 或不同终点上是否重现。若新切分仍无法同时改善 Harrell 与 Uno，则应停止将 masked context 作为 DSS 主线，转向更适合终点定义/事件稀疏性的 pathology aggregation 或 endpoint 设计。

置信度：对“本次固定 split 下未支持核心假设”的判断为高；对“coords 具体失败原因”的判断为中，因为本实验没有做独立坐标扰动/采样诊断。

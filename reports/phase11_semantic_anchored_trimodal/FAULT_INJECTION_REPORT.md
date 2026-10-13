# 故障注入

FAULT_INJECTION.json保留19项合成错误payload、实际拒绝日志、修复后clean通过。涵盖DSS隐名输入、patient跨slide折、全队列scaler/TopK/PCA、网格错配、语义/RNA detach、mini-risk-set Cox、padding、跨split错配、event反向、risk反向及checkpoint/OOF/metadata错误。合成测试不包含患者记录；实际checkpoint反向risk亦由逐元素回放门控拒绝。实际test扰动两次完整80epoch内层和fresh refit全state bitwise一致。

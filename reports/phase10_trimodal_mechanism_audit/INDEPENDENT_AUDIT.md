# 独立审计边界

**PASS：服务器内420份原始OOF及manifest重算，不读取RESULTS作为数字来源。** reference.py不import训练math、Phase8评测或模型代码；事件逐对与reverse-KM逻辑另写，并逐cell与scikit-survival核对。Clinical anchor另用pandas/numpy重建编码，再以独立事件循环目标梯度验证保存系数。300checkpoint重新forward；D0 15份按B0风险逐元素核对。

PASS：inner/outer/teacher训练与selection边界、epoch refit规则、有限预测、原矩阵与基因顺序、模型与数据哈希。真实单cell对测试标签/输入进行干预，重复选择与refit的权重bitwise一致。主配对bootstrap共享同患者多项式counts，三个seed分别评测再取delta均值。

WARN：源码适配/不同有效容量/少数最大epoch边界/平台身份未独立映射，见KNOWN_LIMITATIONS。BLOCKED：旧Phase8私有OOF逐点复算、CiMP/BML全文级复现、没有授权私有数据的公开读者重新评测。

独立指标模块与训练核心分开；诊断checkpoint前向必然使用交付模型实现，因此前向复放不是第二个完整神经网络实现。QKV、loss、编码、gate各有独立性质/手算测试。不要称公开包fully independently reproducible：私有输入只在用户服务器；公开包提供审计入口、代码版本、哈希和已实跑汇总。

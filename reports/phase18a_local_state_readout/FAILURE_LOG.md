# Phase18A 审计修复与失败记录

## 1. 外折标签突变比较 helper

第一次审计已真实完成两个2-epoch短拟合，比较state_dict时对bool buffer作减法引发异常。修复仅对比较helper：浮点tensor比较绝对差，bool/整型buffer比较torch.equal。未改变正式训练source、模型、协议或数据。完整重做两次短拟合后PASS。生产累计4个真实短拟合、8次optimizer steps；均未保存smoke checkpoints/患者风险，正式60拟合不受影响。

## 2. 历史诊断的患者/WSI粒度补全

最早历史冻结probe在完成4条记录后中止，以补充患者等权WSI的z/风险方差和真正final_patient_risk目标的定义。旧不完整产物更名为superseded，不用作完成证据。后续按固定新diagnostics SHA对60个F0/F1 inner/outer checkpoints完成冻结无标签probe。未更新神经网络参数或选择epoch；不计新神经网络拟合或训练失败。完整诊断与DSS评估彼此分开。父Data仍会确定性重解同一Ridge anchor并校验一致，不能写成没有任何优化调用；完整合格回放/探针链195次anchor验证重解另列于结果和计数。

## 正式训练及范围

最终报告要求TRAINING_DONE、全部60新模型和历史来源/重放、独立指标、算子、标签隔离审计通过。上述两项是审计流程修复，不是模型训练失败，也不能隐去已完成短拟合以缩减真实运行计数。没有自动追加旧R3或Phase18B，没有按外折结果更改模型/预算。未来如有失败必须追加，不能改写历史。

## 3. 发布格式修复

发布前 `git diff --check` 检出报告复核JSON的CRLF与非训练评估脚本的额外EOF空白，分别规范化为LF及单一文件结尾。已锁定训练source中的 `src/phase18a_readout/data.py` 额外EOF空行原样保留，以保护既有训练SHA；发布器只对该文件这一明确EOF提示作显式豁免，其余空白检查照常。报告重生成并更新复核摘要；所有指标、训练文件及模型权重不变，没有重训。

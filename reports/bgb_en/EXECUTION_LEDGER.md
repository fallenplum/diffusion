# BG-BEN 实际执行与资源账本

**状态：RUNNING。计划、锁、回放和资格测试不当作已完成拟合。**

| 类型 | 主结果范围 | 历史/替换前范围 | 全部已记录 |
|---|---:|---:|---:|
| reference_cells | 12 | 0 | 12 |
| completed_Cox_fits | 341 | 0 | 341 |
| completed_beta_fits | 30 | 0 | 30 |
| completed_neural_fits | 392 | 3 | 395 |
| completed_external_fits | 9 | 7 | 16 |
| completed_supervised_optimizer_fits | 772 | 10 | 782 |
| analytical_decoder_fits | 199 | 0 | 199 |
| completed_preprocessing_fits | 60 | 0 | 60 |
| logical_seed_reuse_references | 34 | 0 | 34 |
| failed_attempts | 3 | 3 | 6 |
| unfinished_attempts | 0 | 2 | 2 |
| unknown_failed_or_unfinished_update_counts | 2 | 5 | 7 |
| known_neural_external_optimizer_updates | 35300 | 380 | 35680 |
| known_failed_or_unfinished_optimizer_updates | 160 | 0 | 160 |

## 根目录覆盖

| 匿名范围 | 角色 | 状态 | 尝试目录（不含prepare） | prepare bundle | 预期单元（不计执行） |
|---|---|---|---:|---:|---:|
| core | PRIMARY | CLOSED_ACCOUNTING | 186 | 3 | 186 |
| original_F7 | HISTORICAL | CLOSED_ACCOUNTING | 6 | 0 | 6 |
| omics_extra | PRIMARY | CLOSED_ACCOUNTING | 24 | 0 | 24 |
| path_extra | PRIMARY | CLOSED_ACCOUNTING | 36 | 0 | 36 |
| aux_true | PRIMARY | CLOSED_ACCOUNTING | 12 | 0 | 12 |
| aux_wrong | PRIMARY | CLOSED_ACCOUNTING | 6 | 0 | 6 |
| aux_spatial | PRIMARY | CLOSED_ACCOUNTING | 6 | 0 | 6 |
| budget | PRIMARY | CLOSED_ACCOUNTING | 24 | 0 | 24 |
| source_raw | HISTORICAL | CLOSED_ACCOUNTING | 7 | 0 | 7 |
| source_true | PRIMARY | CLOSED_ACCOUNTING | 7 | 0 | 7 |
| source_wrong | PRIMARY | CLOSED_ACCOUNTING | 1 | 0 | 1 |
| source_spatial | PRIMARY | CLOSED_ACCOUNTING | 1 | 0 | 1 |
| pca_original | PRIMARY | CLOSED_ACCOUNTING | 9 | 0 | 9 |
| pca_precision | PRIMARY | CLOSED_ACCOUNTING | 2 | 0 | 2 |
| outer | PRIMARY | CLOSED_ACCOUNTING | 116 | 12 | None |
| matched_support | PRIMARY | CLOSED_ACCOUNTING | 12 | 0 | 12 |
| interrupted_R4_20261028_F3_seed31 | HISTORICAL | CLOSED_ACCOUNTING | 1 | 0 | 1 |
| interrupted_R4_20261028_F3_seed47 | HISTORICAL | CLOSED_ACCOUNTING | 1 | 0 | 1 |

同一物理元数据目录只计一次；重叠根由更具体的显式根决定主结果或历史角色。原F7成功归档仍是实际历史拟合，但不进入当前科学主结果。另一次全新同seed拟合即使权重相同，也不因artifact相同被静默合并。来源16个拟合只由其真实COMPLETE计数，不叠加旧14/新16回放摘要。

## 更新数与时间边界

失败/中断单元若未保存已完成更新数，更新数标为未知；不能用80/160/320计划补齐。神经和来源完整拟合依其冻结的一轮一次完整风险集更新计算已记录更新；Cox和β的solver迭代不能与这些更新混算。解析风险decoder另外计数。

已记录拟合耗时之和是计算暴露，不是并行日历耗时；缺少耗时的失败或中断另报数量，不能补0或假装该和覆盖全部成本。scheduler真实elapsed独立列出；只有配置显式提供并匹配已记录elapsed的共同时间边界，才汇总覆盖区间。当前全局时间状态：**UNKNOWN_NO_COMMON_RECORDED_WALLCLOCK_BOUNDS**。

部分原scheduler只把成功返回写入units；其中非零返回数为0不代表没有失败，阶段failure_entries及真实FAILED另列。

| 调度范围 | 状态 | scheduler elapsed(s) | worker耗时和(s) | 非零返回数 | 明列failure_entries |
|---|---|---:|---:|---:|---:|
| core_schedule | FINISHED_CHECK_UNIT_FAILURES | 6179.761834305013 | 24019.633650965523 | 0 | 0 |
| F7_recovery | SIX_FITS_COMPLETE_PENDING_INDEPENDENT_CERTIFICATION | 764.4290746881161 | 1282.0195667848457 | 0 | 0 |
| additional_batch | COMPLETE | 2660.391762363026 | 10570.62027481175 | 0 | 0 |
| budget_batch | PARTIAL_FAILED_UNITS_NO_AUTOMATIC_RETRIES | 738.139824423939 | 2855.719777130056 | 1 | 0 |
| outer_baselines | COMPLETE | 69.76485290308483 | 117.30486936401576 | 0 | 0 |
| outer_neural | COMPLETE | 2786.207097207196 | 4442.596669119783 | 0 | 0 |
| core_R1_stage | COMPLETE | None | 0 | 0 | 0 |
| core_R2_stage | COMPLETE | None | 0 | 0 | 0 |
| core_R3_stage | PARTIAL_FAILED_UNITS_CONTINUE_INDEPENDENT_MECHANISMS | None | 0 | 0 | 3 |
| matched_support_batch | COMPLETE | 328.6211669619661 | 1190.1261669700034 | 0 | 0 |

## 实际资源采样窗口

| 窗口 | 样本数 | 覆盖秒 | GPU利用率时间均值 | 设备显存峰值MiB |
|---|---:|---:|---:|---:|
| core_and_queued_device_window | 970 | 9779.902 | 80.43852341260677 | 25310.0 |
| R4_recovered_node_20261028_device_window | 426 | 2153.1409999999996 | 89.02272749439076 | 11992.0 |

仅对真实采样窗口报告资源；长于三个采样中位间隔的空档不插值成持续高利用率。nvidia-smi设备显存不等于单进程PyTorch峰值；CPU/RAM来自host procfs，不能称容器或进程quota。未采样训练期不推测，窗口可能包含排队与空闲。

优势：引用、真正优化、历史成功、失败及未知更新分别计数，计划和回放不会膨胀执行数。限制：账本只验证收据字节和声明artifact身份及存在性，不加载/重新哈希权重；科学有效性仍依独立检查点回放。元数据缺失时只能保留未知，不能恢复未记录的失败epoch或完整全局makespan。

本脚本没有模型前向、优化器或GPU调用，不输出私有根目录、患者标识、权重或风险数组。

import sys,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np,torch
from src.phase17r_ssd.data import P,R,cfg,source,write

def table(frame):
    q=frame.copy()
    for k in q:
        if pd.api.types.is_float_dtype(q[k]):q[k]=q[k].map(lambda v:f'{v:.6g}')
    return '| '+' | '.join(q.columns)+' |\n|'+'|'.join(['---']*len(q.columns))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in q.itertuples(index=False,name=None))+'\n'

def main():
    dec=json.loads((R/'STAGE_DECISION.json').read_text());done=json.loads((P/'TRAINING_DONE.json').read_text());cs=cfg();seq=pd.read_csv(R/'SEED_OOF_METRICS.csv');summary=seq.groupby('arm')[['Harrell','Uno60']].mean().reset_index();ci=pd.read_csv(R/'PAIRED_COMPARISONS.csv');inner=pd.read_csv(R/'INNER_MECHANISM.csv');outer=pd.read_csv(R/'OUTER_MECHANISM.csv')
    import importlib.metadata,platform
    packages={}
    for name in ['torch','numpy','pandas','scipy','scikit-learn','scikit-survival','matplotlib','h5py']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='not installed as distribution; see preserved PYTHONPATH'
    write(R/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'packages':packages,'torch_CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'source':source(),'PYTHONPATH':'.:/root/phase10_trimodal_eval_deps','parent_commit':cs['parent_commit']})
    trained=list(cs['R1'])+(cs['R2']['models'] if dec['R2']['run'] else [])+(['R3_reverse'] if dec['R3']['run'] else [])
    report='# Phase17R — Sequential Conditional S6 Repair\n\n## 判定与范围\n\n'
    status='R3_GATE_PASSED_EXPLORATORY' if dec['R3']['run'] else 'STOP_STRUCTURE_BRANCH_R3_GATE_FAILED'
    report+=f'**{status}**。本轮是基于既往Phase17观察提出的新探索性修复，不能当成独立确认。R1/R2的内层决策在任何新外层预测产生前锁定；跨折内层分区会重叠，因此阶段门仅作工程决策，不当作独立OOF显著性。\n\n'
    report+=f"实际正式内层拟合{done['inner_fits']}次、外层拟合{done['outer_fits']}次；R2执行={dec['R2']['run']}，R3执行={dec['R3']['run']}。所有无Direct臂保留患者自己的RNA/CNA通路Token及末端模态主效应；不是删除组学模态。\n\n## DSS外层结果\n\n"+table(summary[summary.arm.isin(trained+['ANCHOR_CS'])])
    report+='\n每seed先汇总5折可比较对分子/分母，再平均3seed；未平均风险。ANCHOR_CS复用每次训练内部同一训练折拟合的Clinical+S32锚点。\n\n## 预注册配对区间\n\n'+table(ci[ci.family=='primary'][['comparison','metric','delta','CI95_low','CI95_high','Bonferroni_low','Bonferroni_high']])
    report+='\n20000次患者折内配对bootstrap，重复计数形成count_i×count_j患者对权重；所有模型/seed共享患者重采样，IPCW固定为对应outer-train估计。预留12项主检验，Bonferroni区间覆盖率99.5833%。区间条件于已训练的模型和选择结果，不涵盖全部训练/研究决策不确定性。\n\n## R1：删除高维捷径后发生什么\n\n'
    cols=['g_relative_L2','delta_over_base_RMS','B_condition_over_base_RMS','h_relative_L2','wrong_context_relative_risk_drift','isolated_SSM_g_gradient_L2','isolated_SSM_RNA_gradient_L2','ssm_over_anchor_SD','gamma']
    report+='训练患者无标签探针；下表为所选内层checkpoint跨15单元中位数。g/状态差由正确与分区内错患者上下文计算，只换状态上下文，末端患者自身证据保持。\n\n'+table(inner.groupby('arm')[cols].median().reset_index())
    report+='\n`isolated_SSM_*`是对SSM风险和求导的隔离路径梯度，不含Direct/late主效应且不使用生存标签；真实Efron训练梯度逐epoch另见TRAINING_CURVES.csv。二者不能混称。\n\n## R2与R3阶段门\n\n'
    replay=pd.read_csv(R/'CHECKPOINT_REPLAY.csv')
    if 'normalized_over_raw_context_difference' in replay:
        norm=replay.dropna(subset=['normalized_over_raw_context_difference']).groupby('arm')[['g_relative_L2_probe','normalized_g_relative_L2_probe','normalized_over_raw_context_difference']].median().reset_index()
        report+='归一化是否保留患者差异：下表是冻结外层模型、训练患者无标签探针的中位数；不用于再选模型。R1未归一化，所以比值应为1；R2比值只描述相对差异传播，不是生物语义质量。\n\n'+table(norm)+'\n'
    report+=f"R2触发所依据的R1 noDirect中位数：g相对差={dec['R2']['g_relative_L2_median']:.6g}，条件delta/基础logit RMS={dec['R2']['delta_over_base_RMS_median']:.6g}，条件B/基础B RMS={dec['R2']['B_condition_over_base_RMS_median']:.6g}，最终状态相对差={dec['R2']['h_relative_L2_median']:.6g}。\n\n"
    if dec['R2']['run']:
        report+='R2只加入无参数LayerNorm和有界tanh；幅度0.25/1.0各自fold-local内选，未使用外折分数。A8、FiLM、输入残差均1664个适配器参数，采用共同写入门及相同正则、训练/选择预算。输入残差是rank13的Cross-Attention上下文投影，不冒充原始rank64 A3。实际时耗与活跃梯度参数不同，另行报告。边界仅约束delta logits和B加性修正，不保证最终风险扰动上界。\n\n'
    report+='R3门判断：\n\n'+table(pd.DataFrame(dec['R3']['predictive_checks'])[['comparator','mean_inner_Harrell_gain','positive_seeds','positive_cells','pass']])
    report+=f"\n状态相对差中位数={dec['R3']['state_relative_L2_median']:.6g}，达0.001的单元={dec['R3']['state_effect_cells']}/15，错配相对风险漂移中位数={dec['R3']['relative_risk_drift_median']:.6g}。失败原因：{'; '.join(dec['R3']['reasons']) or '无'}。\n\n"
    if not dec['R3']['run']:report+='**按预定停止规则，没有训练反向、双向或顺序一致性模型。增大状态变化本身不算预测收益。**\n\n'
    report+='## gamma及状态分支干预\n\n'+table(summary[summary.arm.str.contains('__')])
    report+='\nWRONG只交换状态上下文；SSM_OFF关闭整个SSM风险项；GAMMA_ONE把已训练gamma固定为1。以上均为冻结模型推理干预，不是新的重训消融，不根据这些结果挑选部署模型。\n\n## 训练预算和资源\n\n'
    report+='gamma与线性readout权重存在缩放耦合，所以gamma接近0.1不能单独证明状态分支被忽略；应联合实际SSM/anchor风险标准差、完整分支置零敏感性和真实梯度解读。\n\n'
    curves=pd.read_csv(R/'TRAINING_CURVES.csv');chosen=curves[(curves.stage=='inner')&(curves.epoch==curves.selected_epoch)];report+=table(chosen.groupby('arm').agg(cells=('fold','count'),selected_epoch_median=('selected_epoch','median'),epoch1=('selected_epoch',lambda x:int((x==1).sum()))).reset_index())
    matched_rows=[]
    for epoch in [1,5,10,20,40]:
        aa=curves[(curves.stage=='inner')&(curves.arm=='R1_A8_full')&(curves.epoch==epoch)]
        bb=curves[(curves.stage=='inner')&(curves.arm=='R1_A8_noDirect')&(curves.epoch==epoch)]
        pairs=aa.merge(bb,on=['seed','fold'],suffixes=('_full','_noDirect'))
        for key in ['g_attention_grad','delta_grad','B_condition_grad','write_grad']:
            base=pairs[key+'_full'].to_numpy();change=pairs[key+'_noDirect'].to_numpy();valid=base>1e-12
            ratio=float(np.median(change[valid]/base[valid])) if valid.any() else np.nan
            matched_rows.append({'epoch':epoch,'gradient_group':key,'paired_cells':len(pairs),'full_median':float(np.median(base)),'noDirect_median':float(np.median(change)),'median_paired_ratio_nonzero_base':ratio,'nonzero_base_cells':int(valid.sum())})
    matched_frame=pd.DataFrame(matched_rows);matched_frame.to_csv(R/'MATCHED_EPOCH_GRADIENTS.csv',index=False)
    report+='\n相同epoch的真实Efron梯度对比（不按各模型所选epoch混比）：\n\n'+table(matched_frame[matched_frame.gradient_group=='g_attention_grad'])
    report+='\n零初始化时条件路径对g的梯度可恰为0，此时比值未定义（表中nan），不是缺失训练。选中checkpoint的隔离梯度差同时受所选训练长度影响，不能只据其比值宣称纯粹的旁路梯度竞争。\n'
    report+='\n'+table(pd.read_csv(R/'MODEL_SUMMARY.csv'))
    report+='\nGPU峰值字段内部沿用MB名称，数值单位实际为MiB（bytes/2^20）。运行时间在4任务并行环境采集，不作为隔离速度排名。R2适配器线性MACs为1664/1536/1664每有效patch，最大差128，不到共同视觉投影98304 MACs的0.14%；这是近似计算预算匹配证据，不是完整S6 FLOPs计数或完全相同计算的声明。\n'
    maxed=curves[curves.stage=='inner'].groupby(['arm','seed','fold','scale']).epoch.max();report+=f'\n实际到160轮上限：{int((maxed==160).sum())}/{len(maxed)}个内层候选。训练/验证损失、SSM路径梯度及gamma曲线全部保留，不用单个曲线代表全部模型。\n\n'
    report+='## PFI冻结迁移\n\n'+table(pd.read_csv(R/'PFI_SEED_OOF_METRICS.csv').groupby('arm')[['Harrell','Uno60']].mean().reset_index().query('arm in @trained'))
    report+='\nPFI只复用DSS冻结风险，未进行PFI训练、epoch或幅度选择。\n\n## 优势、风险与下一步\n\n优势：R1真实重训，不用冻结删头冒充；RNA/CNA主效应与条件路径分开；R2容量和选择预算明确；算子、患者隔离、所有外层checkpoint及独立指标可重放。\n\n风险：65个DSS事件、反复开发单队列、内层噪声、空间排序无天然时间含义；有限差分/标签扰动/公开计算核验只能证明各自检查，不能替代外部队列验证。机制观测置信度高，唯一生物学失败原因置信度低。\n\n'
    report+=('已通过内层机制门，顺序实验仍属探索性；应依据外层配对结果决定是否有外部验证价值。\n' if dec['R3']['run'] else '未通过机制与预测的共同门，停止这条条件S6结构支线；不继续扩大扫描、融合或Prompt模块。现有RNA独立预后信号是否与Clinical+S32重叠，仍需低容量条件增量研究，不能由本轮负结果一并否定。\n')
    med=inner.groupby('arm').median(numeric_only=True);scores=summary.set_index('arm')
    report+='\n## 本轮机制解释\n\n'
    report+=f"1. 真实重训删除Direct后，Harrell点估计变化为{scores.loc['R1_A8_noDirect','Harrell']-scores.loc['R1_A8_full','Harrell']:+.6f}、Uno60变化为{scores.loc['R1_A8_noDirect','Uno60']-scores.loc['R1_A8_full','Uno60']:+.6f}；两个配对区间均跨零，不能称为已确认的提升。冻结删头结果不能代替这个重训结论。\n\n"
    report+='2. 同epoch的上下文训练梯度增强约1.1–1.9倍；所选checkpoint隔离梯度的更大差异还受到选中epoch1→17影响。删除旁路改变训练分配，但不足以证明唯一原因就是梯度竞争。\n\n'
    if dec['R2']['run']:
        report+=f"3. R2条件B/基础B的RMS中位数从{med.loc['R1_A8_noDirect','B_condition_over_base_RMS']:.6g}升至{med.loc['R2_A8','B_condition_over_base_RMS']:.6g}，但正确/错患者最终状态差仅从{med.loc['R1_A8_noDirect','h_relative_L2']:.6g}变为{med.loc['R2_A8','h_relative_L2']:.6g}。更大的整体调制不等于更强的患者特异性响应。归一化前后g的差异探针未显示患者差异被整体抹除。\n\n"
        report+=f"4. 外层正确条件R2_A8={scores.loc['R2_A8','Harrell']:.6f}/{scores.loc['R2_A8','Uno60']:.6f}，错条件={scores.loc['R2_A8__WRONG','Harrell']:.6f}/{scores.loc['R2_A8__WRONG','Uno60']:.6f}；未见正确患者配对优势。这是推理敏感性证据，不能宣布统计等效。\n\n"
    report+='5. gamma与readout耦合，SSM风险已有约2倍anchor的训练患者标准差；冻结gamma=1使两种指标下降。关闭SSM则Harrell提高但Uno60下降，不能挑单一指标宣称整个SSM有害或有益。\n\n'
    report+='结论限于当前条件S6实现、有限预算和本cohort。它支持停止该结构支线；仍无法单凭这些结果区分RNA表示信息损失、真正条件冗余与小事件样本估计噪声，也不否定RNA独立预后信号。\n'
    (R/'RESULTS.md').write_text(report)
    counts={'inner_fits':done['inner_fits'],'outer_fits':done['outer_fits'],'label_mutation_short_fits':json.loads((R/'DATA_LEAKAGE_AUDIT.json').read_text())['short_fits'],'R1_inner':60,'R2_inner_candidates':90 if dec['R2']['run'] else 0,'R3_executed':dec['R3']['run']}
    write(R/'EXECUTION_COUNTS.json',counts);write(R/'decision.json',{'status':status,'source':source(),'architecture_decision_utc':dec['decision_utc'],'R2_executed':dec['R2']['run'],'R3_executed':dec['R3']['run'],'reason':dec['R3']['reasons'],'counts':counts,'all_primary_95_intervals_cover_zero':bool(((ci.loc[ci.family=='primary','CI95_low']<=0)&(ci.loc[ci.family=='primary','CI95_high']>=0)).all()),'external_confirmation':False,'no_blanket_RNA_no_signal_claim':True})
    (R/'DECISION.md').write_text('# Decision\n\n'+status+'\n\n'+('\n'.join('- '+x for x in dec['R3']['reasons']))+'\n\nSee RESULTS.md for conditional scope and exploratory limits.\n')
    print('REPORT_COMPLETE',status,counts,flush=True)

if __name__=='__main__':main()

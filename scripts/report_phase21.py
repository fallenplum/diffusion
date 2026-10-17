"""Artifact-backed report; saved private risks reduced to anonymous magnitudes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.phase21_relative.data import ROOT,P,R,source,write
from src.phase10_trimodal.data import digest

def read(name): return json.loads((R/name).read_text())
def table(frame,columns):
    rows=['|'+'|'.join(columns)+'|','|'+'|'.join('---' for _ in columns)+'|']
    for _,q in frame.iterrows():
        rows.append('|'+'|'.join(f'{q[c]:.8g}' if isinstance(q[c],(float,np.floating)) else str(q[c]) for c in columns)+'|')
    return '\n'.join(rows)

def mechanism_table(path,label):
    raw=pd.read_csv(path);raw=raw[raw.role=='outer'];rows=[]
    for population in ['fit','heldout']:
        for direction in ['po','op']:
            prefix=f'diagnostics.{population}.metrics.'
            specs={'raw_row_SD':direction+'_raw_delta_row_SD_patient.mean',
                'weighted_centered_variance':direction+'_weighted_centered_variance_patient.mean',
                'weighted_centered_RMS':direction+'_weighted_centered_RMS_patient.mean',
                'common_fraction_mean_of_ratios':direction+'_common_energy_fraction_patient.mean',
                'weighted_raw_second_moment':direction+'_weighted_raw_second_moment_patient.mean',
                'attention_TV':direction+'_attention_TV_patient.mean',
                'selected_probability_mass':direction+'_selected_mass_patient.mean',
                'message_triangle_efficiency':direction+'_message_over_triangle_bound_patient.mean',
                'query_average_survival':direction+'_query_average_survival_ratio_patient.mean',
                'value_covariance_efficiency':direction+'_taylor_covariance_efficiency_patient.mean',
                'head_projection_efficiency':'chain_head_projection_efficiency_patient.mean',
                'tanh_survival':'chain_tanh_survival_ratio_patient.mean',
                'patient_risk_correction_SD':'chain_actual_patient_correction_risk.SD',
                'patient_risk_correction_abs_mean':'chain_actual_patient_correction_risk_absolute.mean'}
            out={'arm':label,'role':'outer','population':population,'direction':direction,'checkpoints':len(raw)}
            for name,key in specs.items():
                if prefix+key not in raw: raise ValueError('missing measured chain '+prefix+key)
                values=raw[prefix+key];out[name]=float(values.mean());out[name+'_ck_median']=float(values.median());out[name+'_ck_min']=float(values.min());out[name+'_ck_max']=float(values.max())
            # A ratio of averaged energies is deliberately distinct from the
            # mean of each row's common-energy fraction above.
            out['relative_energy_ratio_of_mean_energies']=out['weighted_centered_variance']/out['weighted_raw_second_moment']
            rows.append(out)
    return pd.DataFrame(rows)

def main():
    identity=source();done=json.loads((P/'EVALUATION_DONE.json').read_text())
    if done['status']!='PASS' or done['source']!=identity: raise ValueError('evaluation incomplete')
    for name in ['QUALIFICATION.json','CHECKPOINT_REPLAY.json','INDEPENDENT_VALIDATION.json','BOOTSTRAP_AUDIT.json','PAIR_CREDIT_IDENTITY_AUDIT.json','SAVED_S16_AUDIT.json','DIRECT_DIAGNOSTIC_AUDIT.json']:
        if read(name)['status']!='PASS': raise ValueError('failed gate '+name)
    mechan=pd.concat([mechanism_table(R/'SAVED_S16_DIAGNOSTICS.csv','S16'),mechanism_table(R/'DIRECT_DIAGNOSTICS.csv','DIRECT')],ignore_index=True)
    mechan.to_csv(R/'MECHANISM_CHAIN_SUMMARY.csv',index=False)
    from scripts.package_phase21 import execution_summary
    write(R/'EXECUTION_COUNTS.json',execution_summary(P,identity))
    import sys
    import torch
    import scipy
    import sklearn
    import sksurv
    write(R/'ENVIRONMENT.json',{'status':'PASS','source':identity,'python':sys.version,
        'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
        'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'scikit_survival':sksurv.__version__,
        'torch_CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),
        'GPU_total_MiB':float(torch.cuda.get_device_properties(0).total_memory/1024**2),
        'parallel_training_worker_limit':8,'full_patient_risk_set':True})
    samples=[]
    for role in ['inner','outer']:
        samples.extend({'stage':role,**q} for q in json.loads((P/(role+'_GPU.json')).read_text()))
    gpu=pd.DataFrame(samples);gpu.to_csv(R/'GPU_PROFILE.csv',index=False)
    write(R/'GPU_PROFILE.json',{'status':'PASS','source':identity,
        'sampling_scope':'arithmetic mean of actually recorded queue samples; not total-walltime utilization',
        'stages':[{'stage':role,'samples':len(q),'mean_GPU_percent':float(q.GPU_percent.mean()),
                   'mean_memory_MiB':float(q.GPU_MiB.mean()),'max_memory_MiB':float(q.GPU_MiB.max())}
                  for role,q in gpu.groupby('stage')]})
    from scripts.validate_phase21 import checkpoint
    magnitudes=[]
    for arm in ['P16','S16','DIRECT']:
        for seed in [31,47,83]:
            for fold in [1,2,3,4,5]:
                ck,receipt,_=checkpoint(arm,seed,fold,'outer')
                base=np.asarray(ck['interaction_off_risk'],np.float64)
                anchor=np.asarray(ck['anchor_risk'],np.float64)
                for mode,field in [('matched','matched_risk'),('wrongC','wrongC_risk'),('wrongO','wrongO_risk')]:
                    delta=np.asarray(ck[field],np.float64)-base
                    magnitudes.append({'arm':arm,'seed':seed,'fold':fold,'mode':mode,'heldout_patients':len(delta),
                        'checkpoint_sha256':receipt['checkpoint_sha256'],'actual_fit_source':ck['source'],
                        'full_prediction_difference_mean':float(delta.mean()),'full_prediction_difference_SD':float(delta.std(ddof=0)),
                        'full_prediction_difference_abs_mean':float(abs(delta).mean()),'full_prediction_difference_abs_max':float(abs(delta).max()),
                        'heldout_anchor_SD':float(anchor.std(ddof=0)),
                        'difference_SD_over_heldout_anchor_SD':float(delta.std(ddof=0)/anchor.std(ddof=0)) if anchor.std(ddof=0)>0 else None})
                del ck
    magnitude=pd.DataFrame(magnitudes);magnitude.to_csv(R/'RISK_CORRECTION_MAGNITUDES.csv',index=False)
    magnitude_summary=magnitude[magnitude['mode']=='matched'].groupby('arm')[['full_prediction_difference_SD','full_prediction_difference_abs_mean','full_prediction_difference_abs_max','difference_SD_over_heldout_anchor_SD']].mean().reset_index()
    summary=pd.read_csv(R/'SUMMARY.csv')
    pairs=pd.read_csv(R/'PAIRED_COMPARISONS.csv');repair=pd.read_csv(R/'REPAIR_DAMAGE.csv')
    source_audit=read('MODEL_SOURCE_AUDIT.json');budget=read('BUDGET_VERIFICATION.json')
    primary=pairs[pairs.family=='primary']
    directional=bool((primary.delta>0).all() and (primary.positive_seeds>=2).all())
    statistical=bool((primary.Bonferroni_low>0).all())
    status=('BUDGET_UNRESOLVED' if not budget['budget_gate_pass'] else
            'STATISTICAL_SUPPORT_WITHIN_REUSED_COHORT' if statistical else
            'DIRECTIONAL_ONLY' if directional else 'NOT_SUPPORTED_IN_THIS_OPERATOR_CONTROL')
    local=repair[repair.comparison.isin(['S16-OWN_OFF','DIRECT-OWN_OFF'])]
    specificity=repair[repair.comparison.str.contains('__WRONG_')]
    decision={'status':'COMPLETE','source':identity,'operator_comparison':status,
        'primary_directional_gate':directional,'primary_corrected_interval_gate':statistical,
        'primary_negative_corrected_intervals':primary[primary.Bonferroni_high<0][['comparison','metric','delta','Bonferroni_low','Bonferroni_high']].to_dict('records'),
        'optimization_budget_gate':budget['budget_gate_pass'],'new_formal_neural_fits':30,'qualified_reused_control_models':60,
        'mechanism_chain':'common components dominate many rows and are removed by the operator; centered variation and message survival are nonzero; no proof that removed common variation contained extra DSS information',
        'direct_scope':'same first-order centered response; tests higher-order normalization, does not restore a common row offset',
        'local_correction_pair_credit_status':('NO_OBSERVED_HELDOUT_PAIR_CREDIT_CHANGE' if (local.repair==0).all() and (local.damage==0).all() else 'OBSERVED_PAIR_CREDIT_CHANGES_REPAIR_AND_DAMAGE_REPORTED_SEPARATELY'),
        'local_correction_points':local[['comparison','metric','repair','damage','net']].to_dict('records'),
        'patient_conditioning_points':specificity[['comparison','metric','repair','damage','net']].to_dict('records'),
        'removed_common_component_incremental_DSS_information':'NOT_MEASURED',
        'no_new_selector_encoder_head_or_scale_search':True,'no_external_confirmation':True,
        'next_step':'stop this finite operator matrix; do not expand selector, capacity or scaling from outer outcomes',
        'not_concluded':['RNA lacks complementary information','65 events are the unique bottleneck','attention equals morphology truth','all conditional information vanished']}
    write(R/'decision.json',decision)
    small=summary[summary.arm.isin(['J0','P16','S16','DIRECT'])]
    mainrepair=repair[repair.comparison.isin(['S16-OWN_OFF','DIRECT-OWN_OFF','DIRECT-S16','DIRECT-P16','DIRECT-J0'])]
    seedrepair=pd.read_csv(R/'SEED_REPAIR_DAMAGE.csv')
    seedrepair=seedrepair[(seedrepair.metric=='Harrell')&seedrepair.comparison.isin(['S16-OWN_OFF','DIRECT-OWN_OFF','DIRECT-S16','DIRECT-P16'])].copy()
    seedrepair['net_credit_mass']=seedrepair.repair_credit_mass-seedrepair.damage_credit_mass
    columns=['comparison','metric','C0','C1','repair','damage','net','repair_credit_mass','damage_credit_mass']
    mechsmall=mechan[mechan.population=='heldout']
    mc=['arm','direction','raw_row_SD','weighted_centered_RMS','common_fraction_mean_of_ratios','attention_TV','query_average_survival','patient_risk_correction_SD']
    u=primary[(primary.comparison=='DIRECT-P16')&(primary.metric=='Uno60')].iloc[0]
    interval_sign='完全低于零' if u.Bonferroni_high<0 else '完全高于零' if u.Bonferroni_low>0 else '跨零'
    hlocal=pairs[(pairs.comparison=='DIRECT-OWN_OFF')&(pairs.metric=='Harrell')].iloc[0]
    rel=mechan[(mechan.population=='heldout')&(mechan.direction=='po')].set_index('arm')
    scale_ratio=rel.loc['DIRECT','patient_risk_correction_SD']/rel.loc['S16','patient_risk_correction_SD']
    quick=(f'**本轮效能判定：DIRECT没有稳定超过无修正P16或原S16。** DIRECT−P16的Uno60差为{u.delta:.8f}，'
        f'Bonferroni4区间[{u.Bonferroni_low:.8f}, {u.Bonferroni_high:.8f}]，{interval_sign}；'
        '该负结果限于当前算子与已反复开发的队列。DIRECT−S16的两项区间跨零。\n\n'
        f'DIRECT的15个outer checkpoint修正风险SD均值为S16的{scale_ratio:.3f}倍，但中位数分别仅'
        f'{rel.loc["DIRECT","patient_risk_correction_SD_ck_median"]:.8f}和{rel.loc["S16","patient_risk_correction_SD_ck_median"]:.8f}，不能称所有模型普遍放大。'
        f'同checkpoint开启修正的Harrell净差仅{hlocal.delta:.8f}，'
        f'95%区间[{hlocal.CI95_low:.8f}, {hlocal.CI95_high:.8f}]。S16没有改变任何DSS可比较对的credit；'
        'DIRECT三个seed修复/损伤分别为0/0、2/2、4/3。增大数值变化没有形成可靠排序收益。'
        '独立训练模型的总差异还包含主干优化变化，不能全归给局部修正。\n\n')
    text='''# Phase21 — S16 行内修正、消息传播与配对排序审计

## 结论边界

**共同偏移确实被归一化删除，但“概率重分配互相抵消，导致风险信号完全消失”没有被完整证实。** 原模型的相对差异、注意力变化和最终修正均非零。query 平均与风险投影需要分别看，不能把概率行和为零当成值消息为零。被删除的 δ 能量是否包含超越 J0 的 DSS 信息，本轮没有证明。

所有30份原S16做 fit/heldout 逐级诊断。120条原模型风险向量bitwise回放通过；三臂90模型共540向量独立回放通过。P16/S16共60模型是已存在控制，DIRECT仅新增30正式fit，无旧神经网络初始化；实际原D1/D2来源见 MODEL_SOURCE_AUDIT。2次资格短fit共4步，模型和患者预测不持久化。

## 完整的未见患者数值链

以下仅15 outer checkpoint，各 checkpoint 内 query→等WSI→等患者，再对15checkpoint取均值；范围及fit分区在 MECHANISM_CHAIN_SUMMARY.csv。公共分量是逐行比例的均值，不是先合并全部能量再求比值。两种统计对象另列，禁止互换。

'''+table(mechsmall,mc)+'''

加权中心化方差使用 w=a0*g；常数δ单独代入原算子返回零，减掉加权行均值保持原更新至浮点误差。Taylor项为 rho*w*(δ−μw)。消息三角界效率、value协方差、query/WSI平均、head投影与tanh分别报告；它们单位和分母不同，不能跨指标宣称某层“损失最多”。tanh接近恒等也不能被说成发生输出饱和。

上表risk SD为逐WSI有界修正分支按患者聚合后的量。下表另以实际保存的完整预测计算 matched−ownoff（先把原float32值无损转为float64再相减），包含浮点加法/聚合效应；各值为15个outer checkpoint统计量的平均，绝不把最大值均值叫全体最大值。逐checkpoint/干预明细见RISK_CORRECTION_MAGNITUDES：

'''+table(magnitude_summary,list(magnitude_summary.columns))+'''

## 三臂结果与配对区间

DIRECT保留原门控、kernel、encoder、values、head和最终riskcap，只将selected-mass保守更新改为全局归一化的g*(δ−μw) logit更新。注册参数/初始化布局相同；P16不激活修正分支，实际梯度参数更少。DIRECT与保守规则在零附近一阶相同，均对行公共shift不变；不将它包装为已解决公共偏移的新架构。

'''+table(small,list(small.columns))+ '\n\n'+table(primary,['comparison','metric','delta','CI95_low','CI95_high','Bonferroni_low','Bonferroni_high','positive_seeds'])+'''

每seed先汇总5外折同折可比较对分子/分母，再3seed平均；不平均风险、不把15cells称seedmean。20k共享患者bootstrap、4主检验校正；固定模型区间不包含完整重训练或反复研究选择的不确定性。PFI_SUMMARY是冻结DSS风险迁移，不训练、不切换主终点。

## 修复与损伤同时计量

'''+table(mainrepair,columns)+'''

repair/damage均以全部可比较质量为分母；credit为0/.5/1，tie半分完整计入。mass列为三seed平均的修复/损伤credit质量；Uno60为IPCW质量，不能称独立患者数。所有seed与20k共享患者draw均核验 repair−damage=Cnew−Cbase。9种wrong/tied/correct转移及诊断区间见REPAIR_DAMAGE/PAIR_TRANSITION_INTERVALS。配对不是独立样本，重采样权重为count_i*count_j。

### 每个seed的严格患者对计数

以下Harrell表中 wrong_to_correct_mass 是严格错排→正确的患者对数，correct_to_wrong_mass 是严格正确→错排；tied行另外计量，不与严格错排混称。各seed病例相同，不能把三行相加当作独立患者对。

'''+table(seedrepair,['comparison','seed','wrong_to_correct_mass','correct_to_wrong_mass','wrong_to_tied_mass','tied_to_correct_mass','correct_to_tied_mass','tied_to_wrong_mass','net_credit_mass'])+'''

OWN_OFF指同一checkpoint的own J0+own pair主干，隔离局部修正；DIRECT−P16/S16是独立训练完整pipeline的比较，还包含主干优化差异；J0是三臂共同冻结主效应锚点。wrongC/wrongO仅干预修正上下文，own values/base/main不改，结果单列而不据外折选择模型。wrongO沿用own与donor的覆盖交集，支持变化须与患者身份效应共同解释。外折错对从未作为新监督或教师生成输入。

## 执行、优势与限制

预算状态：'''+status+f'''；DIRECT触顶比例{budget['DIRECT_hit_maximum_fraction']:.6g}，选中epoch中位{budget['selected_epoch_median']:.6g}，实际inner epoch中位{budget['actual_epoch_median']:.6g}。触顶检查不是全局收敛证明。训练曲线、梯度、逐折、原source和失败账本均保留；无自动预算/selector/encoder/head扩展。

优势：原模型逐元素回放；同参数布局、单算子变化；归一化、消息、风险和患者对闭合；保留所有阴性/不确定结果。

局限：重复开发同一512人/65事件cohort，未做外部确认；DIRECT只比较高阶更新几何；同ck冻结干预是敏感性评价，不能替代重训结果。共同偏移去除本身不能说明有用DSS信息被丢失。注意力和RNA程序均没有独立病理形态真值。

## 可复核文件

PROTOCOL/PROTOCOL_LOCK、QUALIFICATION、SAVED_S16_AUDIT、DIRECT_DIAGNOSTIC_AUDIT、CHECKPOINT_REPLAY、MODEL_SOURCE_AUDIT、PAIR_CREDIT_IDENTITY_AUDIT、BOOTSTRAP_AUDIT、BUDGET_VERIFICATION、MECHANISM_CHAIN_SUMMARY及REPRODUCE。患者级OOF、权重、地图、原始特征、标识和凭据均留在服务器私有目录；公开范围见PUBLICATION_SCOPE_AUDIT。
'''
    text=text.replace('## 完整的未见患者数值链',quick+'## 完整的未见患者数值链',1)
    (R/'RESULTS.md').write_text(text,encoding='utf-8')
    inputs=[p for p in R.rglob('*') if p.is_file() and p.name not in ['RESULTS.md','decision.json','REPORT_INPUT_AUDIT.json','ARTIFACT_MANIFEST.json','PUBLICATION_SCOPE_AUDIT.json','PARENT_IMMUTABILITY_AUDIT.json']]
    write(R/'REPORT_INPUT_AUDIT.json',{'status':'PASS','source':identity,
        'files':{p.relative_to(ROOT).as_posix():digest(p) for p in inputs+[ROOT/'configs/phase21_relative_attention.json',Path(__file__).resolve()]},
        'results_sha256':digest(R/'RESULTS.md'),'decision_sha256':digest(R/'decision.json')})
    print('PHASE21_REPORT_READY',status,flush=True)

if __name__=='__main__': main()

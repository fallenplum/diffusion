"""Report actual qualified Phase20 outcomes, with separate scientific axes."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_phase20 import ARMS,PRIMARY,PRIMARY_FAMILY_TESTS,DID_NAME,write

METRICS=['Harrell','Uno60']

def require(condition,message):
    if not condition: raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table(value):
    frame=value if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
    if frame.empty: return '无对应记录。\n'
    def cell(v):
        return (f'{v:.7g}' if np.isfinite(v) else 'NA') if isinstance(v,(float,np.floating)) else str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join('---' for _ in frame.columns)+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])+'\n'

def scientific_decision(seed,paired,budget,behavior):
    """Mechanism evidence must never be negated merely by lower absolute rank."""
    require(not seed.duplicated(['arm','seed']).any(),'duplicate seed C rows')
    means=seed.groupby('arm')[METRICS].mean();primary=paired[paired.family=='primary']
    require(len(primary)==14 and set(zip(primary.comparison,primary.metric))=={(c,m) for c in PRIMARY for m in METRICS},'primary factorial family incomplete')
    require((primary.family_tests==14).all(),'primary correction family differs')
    checks=[]
    for q in primary.to_dict('records'):
        expected=sum(weight*means.loc[arm,q['metric']] for arm,weight in PRIMARY[q['comparison']].items())
        require(abs(expected-q['delta'])<1e-11,'reported contrast point differs from complete seed Cs')
        checks.append({'comparison':q['comparison'],'metric':q['metric'],'delta':q['delta'],
            'positive_seeds':int(q['positive_seeds']),'direction_and_seed_pass':q['delta']>0 and q['positive_seeds']>=2,
            'corrected_positive':q['Bonferroni_low']>0})
    caps={q['arm']:q['budget_gate_pass'] for q in budget['rows']}
    require(set(caps)==set(ARMS),'budget arm matrix differs')
    groups={
        'representation':(['P16-P1'],['P1','P16']),
        'selective_clinical_correction':(['S16-'+a for a in ['P16','D16','F16','WC16','WO16']],['P16','S16','D16','F16','WC16','WO16']),
        'representation_by_correction_interaction':([DID_NAME],['P1','P16','S1','S16']),
    }
    axes={}
    for name,(comparisons,arms) in groups.items():
        rows=[q for q in checks if q['comparison'] in comparisons]
        sufficient=all(caps[arm] for arm in arms)
        direction=all(q['direction_and_seed_pass'] for q in rows)
        corrected=all(q['corrected_positive'] for q in rows)
        status=('BUDGET_UNRESOLVED' if not sufficient else 'WITHIN_COHORT_CORRECTED_SUPPORT' if direction and corrected else
                'DIRECTIONAL_CANDIDATE' if direction else 'NOT_SUPPORTED_IN_THIS_ARCHITECTURE')
        axes[name]={'status':status,'budget_sufficient':sufficient,'direction_and_seed_gate':direction,
                    'all_corrected_intervals_positive':corrected,'checks':rows}
    require(behavior['status']=='PASS','selection behavior audit itself failed')
    selective=axes['selective_clinical_correction']
    selective['predictive_contrast_status']=selective['status']
    selective['selectivity_behavior_gate_pass']=bool(behavior['behavior_gate_pass'])
    if selective['budget_sufficient'] and not behavior['behavior_gate_pass']:
        selective['status']=('CONDITIONED_MODULATION_CANDIDATE_WITHOUT_SPARSE_SELECTION' if selective['direction_and_seed_gate']
                             else 'SELECTIVE_MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE')
    practical=[]
    for arm in ['J0','J_CS']:
        for metric in METRICS:
            delta=float(means.loc['S16',metric]-means.loc[arm,metric])
            practical.append({'reference':arm,'metric':metric,'delta_S16_minus_reference':delta,'no_mean_harm':delta>=0})
    utility=all(q['no_mean_harm'] for q in practical)
    status=axes['selective_clinical_correction']['status']
    return {'status':status,'axes':axes,'practical_prediction_value':{
        'status':'MEAN_NO_HARM_CANDIDATE' if utility else 'NO_FINAL_PREDICTIVE_INCREMENT_DEMONSTRATED',
        'all_J0_J_CS_mean_no_harm':utility,'checks':practical,
        'scope':'practical comparator gate separate from within-architecture mechanism evidence; a negative mean is not an equivalence test'},
        'primary_family_tests':14,'corrected_interval_coverage':1-.05/14,
        'independent_external_confirmation':False,'cohort_scope':'exploratory, repeatedly developed TCGA cohort',
        'not_established':['biological causality','independent morphology truth','absence of all RNA information','global optimization convergence'],
        'next_action':'report both axes and actual limitations; no automatic architecture or endpoint expansion'}

def gate_inputs():
    from src.phase20_selective import data
    from src.phase20_selective import provenance
    identity=data.source();config=data.cfg();P,R=data.P,data.R
    done=provenance.require_record(P/'TRAINING_DONE.json');provenance.require_record(R/'PROTOCOL_LOCK.json')
    evaluation=provenance.require_record(P/'EVALUATION_DONE.json')
    require(done['status']==evaluation['status']=='PASS' and done['source']==evaluation['source']==identity,'training/evaluation source/completion differs')
    require(config['arms']==ARMS and len(config['seeds'])==3 and len(config['folds'])==5,'locked matrix differs')
    require(done['new_inner_fits']==135 and done['new_outer_fits']==135 and done['new_formal_fits']==270,'formal training counts differ')
    require(evaluation['actual_new_checkpoint_replays']==270 and evaluation['bootstrap_draws']==20000,'actual evaluation counts differ')
    gates={}
    for name in ['DATA_AUDIT','DATA_LEAKAGE_AUDIT','OPERATOR_AND_PIPELINE_TESTS','MODEL_QUALIFICATION','INDEPENDENT_VALIDATION',
                 'CHECKPOINT_REPLAY','BOOTSTRAP_AUDIT','CONTRAST_POINT_AUDIT','BUDGET_VERIFICATION',
                 'FIT_AND_REFERENCE_COUNTS','PROBE_COVERAGE_AUDIT','SELECTIVITY_BEHAVIOR_AUDIT','EVALUATION_MATH_AUDIT',
                 'AUDIT_REPAIR_TESTS','AUDIT_REPAIR_COMPATIBILITY','HISTORICAL_REFERENCE_REPAIR_AUDIT']:
        q=provenance.require_record(R/(name+'.json'));require(q['status']=='PASS','gate not PASS: '+name)
        if name=='OPERATOR_AND_PIPELINE_TESTS':provenance.verify_qualified_tests(q)
        gates[name]=q
    repair=provenance.require_record(R/'AUDIT_REPAIR_LOCK.json')
    require(repair['status']=='LOCKED_BEFORE_RESTART_AFTER_AUDIT_ONLY_REPAIR' and repair['source']==identity,'audit-only repair lock differs')
    gates['AUDIT_REPAIR_LOCK']=repair
    math=gates['EVALUATION_MATH_AUDIT']
    history_repair=gates['HISTORICAL_REFERENCE_REPAIR_AUDIT']
    require(history_repair['actual_math_tests']==7 and history_repair['actual_history_regression_tests']==5
            and history_repair['failures_errors_skips']==0 and history_repair['historical_cell_count']==15
            and history_repair['math_audit_sha256']==sha(R/'EVALUATION_MATH_AUDIT.json')
            and history_repair['evaluator_after_sha256']==sha(ROOT/'scripts/validate_phase20.py'),'historical-only evaluation repair differs')
    require(evaluation['evaluator_sha256']==math['evaluator_sha256']==sha(ROOT/'scripts/validate_phase20.py'),'evaluator identity differs')
    require(math['test_source_sha256']==sha(ROOT/'tests/test_phase20_evaluation.py') and math['server_actual_tests']==7
            and math['failures_errors_skips']==0,'math test artifact differs')
    replay=gates['CHECKPOINT_REPLAY'];boot=gates['BOOTSTRAP_AUDIT']
    require(replay['new_neural_checkpoints']==270 and replay['modes_per_checkpoint']==5 and replay['additional_anchor_replays']==270
            and replay['additional_representation_replays']==120,'replay coverage differs')
    require(replay['maximum_error']<2e-6 and replay['anchor_elementwise_identical'] and replay['frozen_main_buffer'],'anchor/model replay failed')
    require(boot['draws']==20000 and boot['valid']>=19800 and boot['primary_family_tests']==14,'bootstrap object differs')
    prior=provenance.require_record(R/'BUDGET_AUDIT.json');require(prior['no_outer_results_used'],'preouter budget evidence differs')
    return data,identity,config,done,gates

def execution_counts(P,identity,done,manifest,compute,fit):
    from src.phase20_selective import provenance
    neural=manifest[manifest.fit_classification=='new_neural_fit']
    inherited=manifest[manifest.fit_classification=='qualified_inherited_convex_reference']
    require(len(neural)==270 and len(inherited)==20 and not neural.duplicated(['arm','seed','fold','role']).any(),'manifest incomplete/duplicated')
    require(all((neural.role==role).sum()==135 for role in ['inner','outer']) and len(compute)==270,'role/compute counts differ')
    require(fit['new_unique_convex_models']==0 and fit['qualified_inherited_unique_convex_models']==20,'convex reuse miscounted')
    ledger=read(P/'SMOKE_LEDGER.json');attempts=ledger['attempts'];smoke=read(P/'SMOKE_COUNTS.json')
    generator=sum(int(q.get('generator_steps',0)) for q in attempts)
    discriminator=sum(int(q.get('discriminator_steps',0)) for q in attempts)
    real=sum(q.get('generator_steps',0)>0 for q in attempts)
    require(smoke['real_short_fits']==real and smoke['generator_optimizer_steps']==generator and discriminator==0,'diagnostic optimizer ledger differs')
    formal_attempts=[];attempt_receipts=[]
    for q in neural.to_dict('records'):
        path=P/'attempts'/q['role']/f"{q['arm']}_seed{int(q['seed'])}_fold{int(q['fold'])}.json"
        current=read(path)['attempts'];require(current,'missing actual formal fit attempts')
        expected_key=(q['arm'],int(q['seed']),int(q['fold']),q['role'])
        for index,item in enumerate(current):
            provenance.require_attempt(item,path,index)
            require((item['arm'],item['seed'],item['fold'],item['role'])==expected_key,'formal attempt identity differs')
        successful=[item for item in current if item['status']=='RETURNED_SUCCESSFULLY']
        require(len(successful)==1 and successful[0]['generator_steps']==int(q['actual_epochs']),
                'successful checkpoint epochs differ from actual optimizer-step ledger')
        require(successful[0]['source']==q['actual_fit_source'],'completed fit source differs from manifest')
        formal_attempts.extend(current);attempt_receipts.append({'arm':q['arm'],'role':q['role'],'seed':int(q['seed']),
            'fold':int(q['fold']),'ledger_sha256':sha(path),'attempts':len(current)})
    formal_real=sum(q.get('generator_steps',0)>0 for q in formal_attempts)
    formal_all_steps=sum(int(q.get('generator_steps',0)) for q in formal_attempts)
    interrupted_attempts=sum(q['status'] not in ['RETURNED_SUCCESSFULLY','FAILED'] for q in formal_attempts)
    zero_step_interrupted=sum(q['status'] not in ['RETURNED_SUCCESSFULLY','FAILED'] and q.get('generator_steps',0)==0 for q in formal_attempts)
    completed_steps=int(compute.actual_epochs.sum())
    require(formal_real>=270 and formal_all_steps>=completed_steps,'formal execution evidence undercounts completed matrix')
    source_counts={str(key):int(value) for key,value in neural.actual_fit_source.value_counts().items()}
    return {'status':'PASS','source':identity,'new_formal_inner_neural_fits':135,'new_formal_outer_neural_fits':135,
        'new_formal_neural_fits':270,'formal_generator_optimizer_steps':completed_steps,
        'completed_formal_actual_fit_source_counts':source_counts,
        'retained_pre_audit_repair_formal_fits':int((neural.actual_fit_source!=identity).sum()),
        'completed_post_audit_repair_formal_fits':int((neural.actual_fit_source==identity).sum()),
        'formal_attempts_recorded':len(formal_attempts),'formal_real_fit_attempts_with_generator_steps':formal_real,
        'formal_zero_committed_step_interrupted_attempts':zero_step_interrupted,
        'formal_real_fit_attempts_lower_bound':formal_real,
        'formal_real_fit_attempts_upper_bound':formal_real+zero_step_interrupted,
        'formal_all_attempt_generator_optimizer_steps':formal_all_steps,
        'formal_interrupted_attempts_with_step_commit_uncertainty':interrupted_attempts,
        'formal_all_attempt_generator_optimizer_steps_lower_bound':formal_all_steps,
        'formal_all_attempt_generator_optimizer_steps_upper_bound':formal_all_steps+interrupted_attempts,
        'formal_failed_or_unfinished_attempts_with_generator_steps':sum(q['status']!='RETURNED_SUCCESSFULLY' and q.get('generator_steps',0)>0 for q in formal_attempts),
        'formal_partial_attempt_generator_optimizer_steps':formal_all_steps-completed_steps,
        'formal_attempt_ledger_audit':attempt_receipts,
        'formal_discriminator_optimizer_steps':0,'new_convex_fits':0,'qualified_inherited_unique_convex_models':20,
        'new_native_fits':0,'new_PCA_or_KMeans_fits':0,'previous_neural_fits_counted_as_current':0,
        'smoke_attempts_recorded':len(attempts),'smoke_real_short_fits_with_generator_steps':real,
        'smoke_generator_optimizer_steps':generator,'smoke_discriminator_optimizer_steps':0,
        'all_recorded_real_neural_fits':formal_real+real,
        'all_real_neural_fits_lower_bound':formal_real+real,
        'all_real_neural_fits_upper_bound':formal_real+real+zero_step_interrupted,
        'all_recorded_generator_optimizer_steps':formal_all_steps+generator,
        'all_recorded_generator_optimizer_steps_lower_bound':formal_all_steps+generator,
        'all_recorded_generator_optimizer_steps_upper_bound':formal_all_steps+generator+interrupted_attempts,
        'scope':'completed/failed callbacks are exact; committed interrupted-attempt steps are a lower bound, with at most one uncommitted optimizer step per interrupted attempt; positive-step real fits counted separately'}

def gpu_profile(P,R):
    rows=[];maximum_interval_seconds=60.
    for role in ['inner','outer']:
        frame=pd.DataFrame(read(P/(role+'_GPU.json'))).sort_values('unix_time');times=frame.unix_time.to_numpy(float);dt=np.diff(times)
        require(len(frame)>1 and (dt>=0).all() and dt.sum()>0,'invalid GPU telemetry')
        observed=(dt>0)&(dt<=maximum_interval_seconds)
        gaps=dt>maximum_interval_seconds
        require(observed.any(),'no observed GPU telemetry intervals within 60 seconds')
        q={'role':role,'samples':len(frame),'sampled_interval_seconds':float(dt[observed].sum()),
            'wall_clock_span_seconds':float(dt.sum()),'observed_interval_count':int(observed.sum()),
            'maximum_counted_interval_seconds':maximum_interval_seconds,
            'excluded_unknown_gap_count':int(gaps.sum()),'excluded_unknown_gap_seconds':float(dt[gaps].sum()),
            'zero_length_interval_count':int((dt==0).sum()),'maximum_running_workers':int(frame.running.max())}
        for key in ['GPU_percent','memory_MiB']:
            values=frame[key].to_numpy(float);require(np.isfinite(values).all(),'nonfinite GPU telemetry')
            q[key+'_time_weighted_mean']=float(np.average(values[:-1][observed],weights=dt[observed]));q[key+'_maximum']=float(values.max())
        rows.append(q)
    pd.DataFrame(rows).to_csv(R/'GPU_PROFILE.csv',index=False)
    write(R/'GPU_PROFILE.json',{'status':'PASS','rows':rows,
        'scope':'whole-card observed intervals only; nominal 10-second sampling; left-held weights only when 0<gap<=60 seconds; longer gaps unknown and excluded, never imputed; maxima use actual sampled values'})
    return rows

def report():
    data,identity,config,done,gates=gate_inputs();P,R=data.P,data.R
    seed=pd.read_csv(R/'SEED_METRICS.csv');paired=pd.read_csv(R/'PAIRED_COMPARISONS.csv')
    compute=pd.read_csv(R/'COMPUTE_PROFILE.csv');manifest=pd.read_csv(R/'CELL_MANIFEST.csv')
    mechanism=pd.read_csv(R/'ATTENTION_MECHANISM.csv')
    decision=scientific_decision(seed,paired,gates['BUDGET_VERIFICATION'],gates['SELECTIVITY_BEHAVIOR_AUDIT']);decision['source']=identity
    counts=execution_counts(P,identity,done,manifest,compute,gates['FIT_AND_REFERENCE_COUNTS'])
    gpu=gpu_profile(P,R)
    summary=seed.groupby('arm')[METRICS].agg(['mean','std','min','max'])
    display=pd.DataFrame({'arm':summary.index,**{metric+'_'+stat:summary[metric][stat].to_numpy() for metric in METRICS for stat in ['mean','std','min','max']}})
    primary=paired[paired.family=='primary'];secondary=paired[paired.family!='primary']
    pfi=pd.read_csv(R/'PFI_SEED_METRICS.csv').groupby('arm')[METRICS].agg(['mean','std']).reset_index()
    pfi.columns=['arm','Harrell','Harrell_seed_SD','Uno60','Uno60_seed_SD']
    coverage=gates['PROBE_COVERAGE_AUDIT']
    body='# Phase20 — Rich Omics Tokens × Selective Clinical Correction\n\n'
    body+='## 判定\n\n'+f"**{decision['status']}**。表示、选择性机制、二者交互及实用预测增量分别判定，不能用未胜Clinical+S32替代机制检验。\n\n"
    body+=f'当前审计source `{identity}`；135 inner + 135 fresh outer = 270个正式神经模型；20个继承凸参考仅校验、没有重新求解。\n\n'
    body+='审计路径修复前后实际拟合source分别保留，原checkpoint及receipt未改写：\n\n'+table([
        {'actual_fit_source':key,'completed_models':value} for key,value in counts['completed_formal_actual_fit_source_counts'].items()])
    body+='\nD1在多WSI患者的anchor-only审计中出现浮点聚合identity失败；D2仅验证各WSI offset等于本人缓存后取首值，不改任何正式预测、训练梯度或epoch选择路径。135 inner + 54 outer = 189份D1模型经原SHA及全部正式模式逐元素兼容核验保留，剩余81份outer在D2继续训练；不是270份全新D2拟合。失败与中断的实际步骤另计，原记录保留，见审计修复文件及EXECUTION_COUNTS。\n\n'
    body+='独立评估另修复一处历史LOCAL引用资格错误：原Phase16R的OOF来自独立重建特征后的float64风险重算，Phase19既有门为绝对误差≤1e−9；Phase20误用bitwise相等而中止。15cell原SHA、患者折、shape及float64均通过，最大差仅'+f"{gates['HISTORICAL_REFERENCE_REPAIR_AUDIT']['maximum_historical_reference_error']:.16g}"+'，全pair严格顺序和1e−8 tie类别也相同。仅恢复原历史引用规则并新增回归；不cast、不改OOF原值，当前270预测/anchor门及主检验代码不变，保留已通过的1740回放，只恢复统计。原失败日志、新旧代码与实际7+5测试见HISTORICAL_REFERENCE_REPAIR_AUDIT.json。\n\n'
    body+='## DSS完整外折指标\n\n'+table(display)
    body+='\n每seed先拼完整五折并加总同折可比较对分子/分母，再平均3个seed的C。3seed不是独立患者队列，不平均患者风险。历史LOCAL输入/协议不同，只是强参照。\n\n'
    body+='## 七项预锁定配对比较\n\n'+table(primary[['comparison','metric','delta','CI95_low','CI95_high','Bonferroni_low','Bonferroni_high','positive_seeds','positive_fold_cells']])
    body+='\n20,000次外折内患者bootstrap，共享所有模型/seed的患者计数；pair权重count_i×count_j，Uno60采用对应outer-train反向KM。DID每draw由四个完整C相减。14项家族校正区间覆盖率'+f'{100*(1-.05/14):.6f}%。区间条件于已拟合模型，不包括完整重训和反复研究者选择。\n\n'
    for name,axis in decision['axes'].items(): body+='### '+name+'\n\n'+f"**{axis['status']}**；预算门={axis['budget_sufficient']}；方向/seed门={axis['direction_and_seed_gate']}；全部校正区间为正={axis['all_corrected_intervals_positive']}。\n\n"
    body+='### 选择性行为门\n\n'+table([{'direction':key,**value} for key,value in gates['SELECTIVITY_BEHAVIOR_AUDIT']['directions'].items()])
    body+='\n仅取15个selected inner S16的all-fit label-blind探针；先WSI内query均值、患者内等WSI、患者等权，再取checkpoint中位数。两方向support(g>1e−6, eligible≥2)中位数均须严格<0.75；退化row与全部row支持另披露，不根据outer成绩修改。该门只证明操作上选择了部分边，不证明临床有效。\n\n'
    body+='## 实用预测收益及冻结干预\n\n'+table(decision['practical_prediction_value']['checks'])+'\n'+table(secondary)
    body+='\nS16的wrongC/wrongO只替换新增修正的context；本人main、两两主干及value保持匹配。WC16/WO16为真正重训对照，冻结S16干预不是重训。constantC使delta为零，与correction_off应逐元素一致；这两个相同干预不构成两份独立机制证据。off保留J0+pairwise，不是main-only；独立anchor模式另逐元素核验。\n\n'
    body+='P1/P16另有冻结fullOmicsWrong（替换整个P–O分支的Omics，main/Clinical/visual仍匹配）及uniformPair（双向map变为valid均匀，values不变）。它们用于区分注意力交互与非线性组学主效应，不是重训消融。DID_uniform由四个完整C计算，作为单列探索；不追加到14项主要家族，也不将探索结果写成确认机制。\n\n'
    body+='## 表示和机制边界\n\nP1/P16使用相同16个learned signed gene-linear filters及encoder，P1仅经预固定rank1投影和幅度匹配广播；P16保留全16坐标。它们是监督训练的丰富表示，不是PCA；L0才是原scalar机制的新鲜参考。共同main冻结，Clinical修正最终风险有训练折anchor尺度下的显式上限；小logits本身不能保证小预测影响。\n\n'
    body+='程序基因覆盖、有效表达秩、missingness/平台关联及真实query覆盖必须按审计披露。attention、gate或TITAN分数不是真实形态标签；错患者是机制干预控制，不是疾病因果实验。当前65 DSS事件、反复开发队列下，CI跨零不证明等效、无RNA信息或所有交互无效。\n\n'
    body+='共同J0已经包含Clinical、RNA512及其他全部输入主效应；因此B检验的是冻结全输入强anchor之后额外Token容量的条件增量，没有重新检验独立RNA预后能力。Clinical没有新增修正head的直接通路且Cref严格归零，但仍可能通过slot及可区分value的重加权间接表达Clinical信息；即使配对控制通过，也不足以单独证明真实分子—形态学机制。\n\n'
    body+='P16−P1同时改变RNA、CNA扩增与缺失三个通道，不能单独归因于RNA。rank1与完整摘要的功能、梯度及有效正则不同；各臂独立联合训练的两两主干也不同，S16−P16不是同一冻结checkpoint上的局部修正效应。F16仅gate固定Clinical参考，delta仍由本人Clinical条件化。J0在fit患者上的offset来自同fit拟合，不是交叉拟合teacher；本轮冻结offset且没有可学stacking权重，但不能将其称为cross-fitted anchor。\n\n'
    body+='选择器的25%预算限制sum(g)，不保证仅25%的边非零。若实测支持接近全体边，不能称已定位少数关键Patch–Pathway关系；rank1的能量匹配也只对各向同性等方差filter输出具有名义解释，须同时报告实际响应SD/rank。\n\n'
    representation_columns=['RNA_summary16_before_effective_rank','RNA_summary16_after_effective_rank',
        'RNA_tokens_effective_rank','RNA_measured_response_map_rank_after_mean',
        'summary16_empirical_after_before_energy_ratio','RNA_program_union_genes','RNA_genes_outside_program_union',
        'RNA_anchor_beta_program_union_energy_fraction','RNA_anchor_beta_nominal_filter_span_energy_fraction']
    require(set(representation_columns)<=set(mechanism.columns),'registered representation diagnostics missing')
    representation=mechanism[mechanism.arm.isin(['P1','P16','S1','S16'])].groupby(['arm','role'])[representation_columns].median().reset_index()
    representation.to_csv(R/'REPRESENTATION_MECHANISM_SUMMARY.csv',index=False)
    body+='### 实测表示诊断（checkpoint中位数）\n\n'+table(representation)
    body+='\n这里的covariance effective rank不是输入Jacobian秩；rank1仅指每程序输入摘要瓶颈，非线性token输出仍可具有更高协方差秩。J0系数投影仅是名义all-genes-measured欧氏几何，不是已解释的DSS信息或患者级信息保真下界。filters先前受DSS训练，诊断不读取新生存标签不等于表征无监督。\n\n'
    correction_columns=['anchor_risk_SD','pair_risk_SD','correction_risk_SD','correction_SD_over_anchor_SD',
        'po_selectivity_support_patient_mean','op_selectivity_support_patient_mean',
        'po_selected_base_probability_mass_mean','op_selected_base_probability_mass_mean',
        'po_TV_over_rho_M_mean','op_TV_over_rho_M_mean','matched_wrongC_risk_MAE','matched_wrongO_risk_MAE',
        'wrongO_eligible_pair_loss_fraction_mean','isolated_correction_clinical_per_patient_gradient_L2_mean',
        'isolated_correction_RNA_per_patient_gradient_L2_mean']
    require(set(correction_columns)<=set(mechanism.columns),'registered correction diagnostics missing')
    correction=mechanism[mechanism.arm.isin(['S1','S16','D16','F16','WC16','WO16'])].groupby(['arm','role'])[correction_columns].median().reset_index()
    correction.to_csv(R/'CORRECTION_MECHANISM_SUMMARY.csv',index=False)
    body+='### 实测修正诊断（checkpoint中位数）\n\n'+table(correction)
    body+='\nS/D共享未加权sum(g)及最终risk cap，不代表实际M=sum(a0*g)、TV或风险修正幅度相等。错Omics的eligible交集损失必须同时披露。这里是selected模型的all-fit匹配context探针；8查询的输入梯度只隔离新增修正。训练日志/summary保存的最后优化步参数梯度来自实际最后epoch，不可冒称selected epoch参数梯度。\n\n'
    outer_s16=mechanism[(mechanism.arm=='S16')&(mechanism.role=='outer')]
    frozen=paired[paired.comparison.isin(['S16-S16__'+tag for tag in ['WRONG_C','WRONG_O','CONSTANT_C','CORRECTION_OFF']])]
    if len(frozen)==8 and (frozen[['delta','CI95_low','CI95_high','Bonferroni_low','Bonferroni_high']].to_numpy()==0).all() and (outer_s16.correction_risk_SD>0).any():
        body+='S16的四种冻结wrong/off干预在本次Harrell与Uno60点值及20k区间中均为零差，**这不等于修正风险为零或梯度断路**。outer角色模型的all-fit探针中，correction risk SD均值为'+f'{outer_s16.correction_risk_SD.mean():.7g}'+'，相对anchor SD均值为'+f'{outer_s16.correction_SD_over_anchor_SD.mean():.7g}'+'；Clinical/RNA隔离输入梯度和attention修正也非零。应解释为“观察到非零但小的修正，未改变当前排名指标”；幅度小是实测表象，尚不能认定为唯一因果瓶颈。这些幅度/梯度是训练患者探针，不能冒充独立形态学真值。\n\n'
    body+='## 训练预算\n\n'+table(gates['BUDGET_VERIFICATION']['rows'])+'\n预算门只检验预注册patience/cap，不能证明全局最优。\n\n'
    capped=[{'arm':q['arm'],'capped_inner_cells':int(round(q['cap_fraction']*q['inner_cells'])),
             'inner_cells':q['inner_cells']} for q in gates['BUDGET_VERIFICATION']['rows'] if q['cap_fraction']>0]
    body+=('存在训练上限触顶的inner单元：\n\n'+table(capped)+'\n低于预注册容许比例只代表预算资格通过，不能把这些触顶单元称为已收敛。\n\n'
           if capped else '本轮没有inner单元触及最大epoch；这仍不证明全局收敛。\n\n')
    body+='## 机制查询覆盖\n\n'+table(coverage['rows'])+'\n只公开独立患者/WSI数量，不公开indices；checkpoint数不等于独立患者数。\n\n'
    body+='## PFI次要终点\n\n'+table(pfi)+'\n仅迁移冻结DSS风险，未使用PFI训练、选择epoch或挑选架构。\n\n'
    body+='## 执行与GPU（仅已观察区间）\n\n'+table(gpu)+'\nGPU原始采样不改写。名义采样间隔10秒，仅对相邻间隔0<dt≤60秒做前值保持时间加权；超过60秒的故障、断电或审计间隙均记为unknown并整段剔除，报告其数量和秒数，不估填离线利用率。最大值只取实际采样点，不将已观察区间均值推广到完整墙钟时间。\n\n正式拟合、实际发生更新的诊断短拟合和继承参考分别计数，见EXECUTION_COUNTS.json。无新增native/PCA/KMeans训练，历史官方方法仅作为背景，不混入当前统一协议榜单。患者级风险、gene信息、权重和所有原始数据仅保留私有目录。\n'
    body+=f"\n账本可确认实际发生更新的神经拟合至少{counts['all_real_neural_fits_lower_bound']}次、至多{counts['all_real_neural_fits_upper_bound']}次；差额来自{counts['formal_zero_committed_step_interrupted_attempts']}个已提交0步的中断尝试，每个可能已有1步发生但尚未提交。已完成和正常失败尝试的优化步回调完整。{counts['formal_interrupted_attempts_with_step_commit_uncertainty']}个被终止尝试的committed步数为记录下界，每个最多另有1步已发生但未提交；全部拟合优化步总量区间为[{counts['all_recorded_generator_optimizer_steps_lower_bound']}, {counts['all_recorded_generator_optimizer_steps_upper_bound']}]，不把记录下界称为精确实际步数。\n"
    parameter_columns=[name for name in ['registered_parameters','trainable_parameters','last_step_gradient_parameters','actual_epochs','seconds','peak_GPU_MiB'] if name in compute.columns]
    body+='\n### 实际参数与计算\n\n'+table(compute.groupby('arm')[parameter_columns].agg(['mean','max']).reset_index().pipe(
        lambda frame:frame.set_axis(['_'.join(str(v) for v in col if v) if isinstance(col,tuple) else col for col in frame.columns],axis=1)))
    (R/'RESULTS.md').write_text(body,encoding='utf-8')
    write(R/'decision.json',decision);write(R/'EXECUTION_COUNTS.json',counts)
    (R/'DECISION.md').write_text('# Phase20 decision\n\n'+json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    query=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True)
    write(R/'ENVIRONMENT.json',{'status':'PASS','source':identity,'python':platform.python_version(),
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','torch','scikit-learn','scikit-survival']},
        'GPU_query':query.strip().splitlines(),
        'scope':'actual report-time package versions; no environment/credential dump'})
    attempts=read(P/'SMOKE_LEDGER.json')['attempts']
    failures=[{key:q.get(key,'') for key in ['id','arm','source','status','generator_steps','discriminator_steps','error_type']} for q in attempts]
    failure_body='# Phase20 execution records\n\n'+table(failures)
    failure_body+='\n只按实际optimizer更新账本计数；诊断短拟合不保存checkpoint或患者预测，不冒充正式270模型。预备资格中的失败在此保留，不能从完成checkpoint数量反推总尝试次数。\n\n'
    if (P/'FAILED.json').exists():
        failed=read(P/'FAILED.json');failure_body+='存在正式scheduler失败收据：\n\n'+table([{k:failed.get(k,'') for k in ['source','role','cell','exit']}])
    else:
        failure_body+='本次读取时未发现正式scheduler FAILED.json；这不证明没有未记录的中断或失败尝试。\n'
    formal_failures=[]
    for path in sorted((P/'attempts').glob('*/*.json')):
        for item in read(path)['attempts']:
            if item['status']!='RETURNED_SUCCESSFULLY':
                formal_failures.append({key:item.get(key,'') for key in ['arm','seed','fold','role','source','status','generator_steps','error_type']})
    failure_body+='\n## 正式模型增量尝试账本\n\n'+table(formal_failures)
    failure_body+='\n完成270个正式模型与实际发生更新的失败/中断尝试分计，全部optimizer步数见EXECUTION_COUNTS.json；不将重试的partial步骤隐去。\n'
    failure_body+='\n## 历史引用评估修复\n\n独立评估曾因historical LOCAL identity mismatch中止；这是历史float64独立重算与错误bitwise资格的差异，未触发训练失败或重新回放。原失败日志和旧evaluator/math凭据已按SHA保存；仅恢复既有Phase19的1e−9历史门，并新增全pair顺序/tie一致性、5项回归及原7项数学测试。具体15cell身份、AST不变和1740回放保留证据见HISTORICAL_REFERENCE_REPAIR_AUDIT.json，原失败不得静默省略。\n'
    (R/'FAILURE_LOG.md').write_text(failure_body,encoding='utf-8')
    reproduce=f'''# Phase20 reproduction

当前审计source: `{identity}`。原PROTOCOL_LOCK.json保留D1，AUDIT_REPAIR_LOCK.json锁定D2；旧记录必须经精确SHA兼容清单核验，不能仅允许旧source字符串。独立评估另绑定当前哈希。运行环境见ENVIRONMENT.json。

## 必需私有输入

沿用已核验的Phase19原始输入、冻结视觉特征、10个role/fold预处理bank和20个J0/J_CS凸参考；还必须持有本次135 inner+54 outer的D1 checkpoint/原receipt、81 outer的D2 checkpoint、原D1归档/资格/epoch锁、完整optimizer账本及私有ANCHOR_AUDIT_REPAIR_COMPATIBILITY清单。路径由configs/phase20_selective_attention.json及data.py确定。原始数据、患者级OOF、模型与gene列表不随公开仓库发布。祖先cache缺失或SHA不符必须停止，不能在旧目录静默重建或将旧神经模型冒充新OOF。

## 本发行包的执行边界

本次证据来自D1正式训练、anchor-only审计修复资格和D2续跑，当前validator严格校验189份原D1及81份D2的逐文件SHA兼容谱系。公开代码与本次私有历史共同支持结果审核/回放；不声称仅凭公开发行包即可从空目录完整重现同一混来源运行。

**本次原目录禁止重新运行prepare、start/scheduler、repair或audit_phase20_history_reference。** prepare会在末尾源检查之前重写资格记录并追加短拟合，破坏冻结D1清单；历史引用审计会写数学凭据，也不是可重复的只读检查。首次D1准备、D2恢复与历史引用技术修复均为已完成的执行，不提供再次启动它们的可执行命令。

fresh full-D2训练必须使用新目录和独立运行清单，重新定义/核验全D2来源及完整资格；当前严格189/81的validator不能直接用于宣称该新运行合格。这属于新的复现运行，不在本次已完成证据范围内。

## 已有私有结果的审核与回放

仅在正式TRAINING_DONE、EVALUATION_DONE真实完成、没有训练/回放任务运行时，由持有上述全部私有输入者在服务器仓库根目录执行。下列命令会重新计算并覆盖本次匿名评估表及私有OOF cache；审核已有结果可仅读取发布的审计文件，不必再次计算。需要保留原评估字节时，应先对输出作只读备份；不得迁移或重写兼容清单绑定的原绝对路径。

```bash
export PYTHONPATH=.:/root/phase10_trimodal_eval_deps
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

本次先完成135 inner并锁epoch，再fresh refit135 outer；D1已完成54 outer，D2只续跑剩余81。以原件SHA、兼容证明、TRAINING_DONE和checkpoint receipts为证据，不能以日志或checkpoint存在代替。审核时不再启动任何训练。

符合上述条件后，回放可四分片并行，必须全部成功退出：

```bash
pids=()
for shard in 0 1 2 3; do
  python scripts/validate_phase20.py --replay --shard "$shard" --shards 4 &
  pids+=("$!")
done
failed=0
for pid in "${{pids[@]}}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then exit 1; fi
python scripts/validate_phase20.py
python scripts/report_phase20.py
```

正式validator收齐270份5context、270anchor和120额外表征干预，重建60个完整OOF向量/300折指标，独立sksurv核验、14主检验/16次检验/10探索统计及20k共享患者bootstrap，通过后自行写EVALUATION_DONE。失败/缺失/错源会拒绝完成，不靠手填marker。

## 计数与边界

270为完成的正式模型数；实际发生更新的失败/中断正式尝试与诊断短拟合由增量账本另计，20个凸模型只是继承参考、0新增求解。PFI只迁移冻结DSS风险。A、B、DID、选择性行为及实用收益分别判断；科学阴性/预算未决可以发布，技术审计不完整不能冒充完成。
'''
    (R/'REPRODUCE.md').write_text(reproduce,encoding='utf-8')
    names=['INDEPENDENT_VALIDATION.json','CHECKPOINT_REPLAY.json','BOOTSTRAP_AUDIT.json','CONTRAST_POINT_AUDIT.json',
           'AUDIT_REPAIR_LOCK.json','AUDIT_REPAIR_TESTS.json','AUDIT_REPAIR_COMPATIBILITY.json',
           'HISTORICAL_REFERENCE_REPAIR_AUDIT.json',
           'PROBE_COVERAGE_AUDIT.json','SELECTIVITY_BEHAVIOR_AUDIT.json','BUDGET_VERIFICATION.json','SEED_METRICS.csv','PAIRED_COMPARISONS.csv','CELL_MANIFEST.csv']
    write(R/'REPORT_INPUT_AUDIT.json',{'status':'PASS','source':identity,'files':{name:sha(R/name) for name in names},
        'reporter_sha256':sha(Path(__file__)),'results_sha256':sha(R/'RESULTS.md'),'decision_sha256':sha(R/'decision.json')})
    print('PHASE20_REPORT_PASS',decision['status'],flush=True)

if __name__=='__main__': report()

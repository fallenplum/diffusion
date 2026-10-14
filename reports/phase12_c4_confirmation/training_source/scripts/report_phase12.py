"""Six explicit mechanism answers from audited locked experiments only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from validate_phase11_independent import components,ipcw
R=ROOT/'reports/phase12_c4_confirmation';P=Path('/root/phase12_c4_private');F=P/'formal'


def table(df):
    out=['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']
    for row in df.itertuples(index=False,name=None):out.append('|'+'|'.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+'|')
    return '\n'.join(out)
def write(name,text):(R/name).write_text(text.rstrip()+'\n',encoding='utf8')


def main():
    cfg=json.loads((ROOT/'configs/phase12_c4_confirmation.json').read_text());validation=json.loads((R/'ALL_INDEPENDENT_VALIDATION.json').read_text());replay=json.loads((R/'REPLAY_AUDIT.json').read_text());literal=json.loads((R/'LITERAL_BOOTSTRAP_AUDIT.json').read_text())
    if validation['status']!='PASS' or validation['verified_records']!=465 or replay['status']!='PASS' or replay['checkpoint_records_replayed']!=465 or literal['status']!='PASS':raise RuntimeError('audits/matrix incomplete')
    seed=pd.read_csv(R/'SEED_OOF_METRICS.csv');fold=pd.read_csv(R/'FOLD_METRICS.csv');paired=pd.read_csv(R/'ALL_PAIRED.csv')
    summary=seed.groupby('arm').agg(Harrell=('Harrell','mean'),Harrell_seed_SD=('Harrell','std'),Uno60=('Uno60','mean'),Uno60_seed_SD=('Uno60','std')).reset_index()
    summary=summary.merge(fold.groupby('arm').agg(worst_Harrell=('Harrell','min'),worst_Uno60=('Uno60','min')).reset_index(),on='arm');summary.to_csv(R/'MODEL_SUMMARY.csv',index=False)
    matrix=[];training=[];gradients=[];actual=0;neural=0;budget_sensitive=[];active_arms=[]
    for arm,spec in cfg['arms'].items():
        cells=[json.loads((F/'cells'/f'{arm}_seed{s}_fold{f}.json').read_text()) for s in cfg['seeds'] for f in cfg['folds']]
        actual+=sum(c['actual_fit'] for c in cells)
        for c in cells:
            if 'inner' not in c:continue
            neural+=1;il=c['inner']['logs'];ol=c['outer']['logs'];ep=c['selected_epoch'];slope=float(np.polyfit(np.arange(10),[v['val_EMA'] for v in il[-10:]],1)[0]);sensitive=ep==80 and slope<0
            if sensitive:budget_sensitive.append({'arm':arm,'seed':c['seed'],'fold':c['fold']})
            training.append({'arm':arm,'seed':c['seed'],'fold':c['fold'],'inner_epochs':c['inner']['actual_epochs'],'selected_epoch':ep,'fresh_refit_epochs':c['outer']['actual_epochs'],
                'inner_train_loss_first':il[0]['train_loss_before'],'inner_train_loss_last':il[-1]['train_loss_after'],'inner_val_EMA_selected':il[ep-1]['val_EMA'],'inner_val_EMA_last':il[-1]['val_EMA'],'inner_Harrell_selected':il[ep-1]['val_Harrell'],
                'final10_EMA_slope':slope,'TRAINING_BUDGET_SENSITIVE':sensitive,'at_minimum':ep==10,'at_maximum':ep==80,'refit_loss_first':ol[0]['train_loss_before'],'refit_loss_last':ol[-1]['train_loss_after'],
                'connected_parameters':max(v['connected_parameters'] for v in ol),'nonzero_gradient_elements':max(v['nonzero_gradient_elements'] for v in ol),'correction_sd':ol[-1]['correction_sd'],'correction_max':ol[-1]['correction_max'],'attention_entropy':ol[-1]['entropy'],
                'seconds':c['seconds'],'peak_MB':max(c['inner']['peak_cuda_MB'],c['outer']['peak_cuda_MB']),'loader_host_submission_seconds':c['patch_loader_seconds_host_submission'],'loader_calls':c['patch_loader_calls'],'preload_seconds':c['gpu_cache_preload_seconds']})
            for role,logs in [('inner',il),('outer',ol)]:
                for module in logs[0]['gradients']:gradients.append({'arm':arm,'seed':c['seed'],'fold':c['fold'],'partition':role,'module':module,'first_norm':logs[0]['gradients'][module],'last_norm':logs[-1]['gradients'][module],'mean_norm':np.mean([v['gradients'][module] for v in logs])})
        mine=[v for v in training if v['arm']==arm]
        capacity=mine
        if not capacity and spec.get('reuse') and spec['budget']:
            parent=Path('/root/phase11_semantic_anchor_private/formal/cells')
            capacity=[]
            for c in cells:
                previous=json.loads((parent/f'{spec["reuse"]}_seed{c["seed"]}_fold{c["fold"]}.json').read_text())
                logs=previous['outer']['logs']
                capacity.append({'connected_parameters':max(v['connected_parameters'] for v in logs),
                                 'nonzero_gradient_elements':max(v['nonzero_gradient_elements'] for v in logs)})
        matrix.append({'arm':arm,**spec,'records':15,'new_fits':sum(c['actual_fit'] for c in cells),'registered_parameters_min':min(c['parameters'] for c in cells),'registered_parameters_max':max(c['parameters'] for c in cells),
            'clinical_inputs':'NONE' if spec['kind']=='ridgeRNA' else 'AGE,SEX,Stage,T,N,M','S32_used':spec['kind']!='ridgeRNA','real_RNA_used':spec['kind'] not in ['null','legacy_noRNA','ridge'],
            'connected_parameters_min':min(v['connected_parameters'] for v in capacity) if capacity else None,'nonzero_gradient_elements_min':min(v['nonzero_gradient_elements'] for v in capacity) if capacity else None,
            'gradient_parameter_evidence':'parent training logs,exact checkpoint reuse' if spec.get('reuse') and capacity else ('new training logs' if capacity else 'linear coefficients;see independent Ridge gradient/refit'),
            'query_backbone':'C/S32<->Patch and RNA->Patch,8RNA+32S32+6Clinical tokens' if spec['budget'] else 'Ridge Cox','head':('Ridge Efron' if not spec['budget'] else ('single_joint_scaled0.1' if spec.get('no_anchor') else 'joint_C+S32_linear_plus0.1_deep')),
            'constant_token_input':'512 learned const params,real gene projection inactive' if spec['kind']=='null' else '', 'replication':'ADAPTED/INSPIRED'} )
        if (F/'probes'/f'{arm}_seed31_fold1.npz').exists():active_arms.append(arm)
    if actual!=390 or neural!=360:raise RuntimeError('new actual fit count')
    pd.DataFrame(matrix).to_csv(R/'MODEL_MATRIX.csv',index=False);pd.DataFrame(training).to_csv(R/'TRAINING_DIAGNOSTICS.csv',index=False);pd.DataFrame(gradients).to_csv(R/'MODULE_GRADIENTS.csv',index=False)
    coverage=[]
    for f in cfg['folds']:
        for role in ['inner','outer']:
            a=json.loads((P/'preprocessing'/f'fold{f}_{role}.json').read_text());g=a['phase12'];match=g['matching'];coverage.append({'fold':f,'partition':role,'original_covered':a['groups']['biologic_covered'],'original_fallback':a['groups']['fallback_genes'],'enriched_covered':g['enriched_covered'],'enriched_fallback':g['enriched_fallback'],'eligible_pathway_genes':g['eligible_pathway_genes'],'enriched_original_intersection':g['original_top_overlap'],'matched_original_intersection':g['original_top_matched_overlap'],'strict_match_genes':sum(v['strict_bin_match'] for v in match),'nearest_rank_fallback_genes':sum(not v['strict_bin_match'] for v in match),'max_mean_rank_gap':max(v['mean_rank_gap'] for v in match),'max_SD_rank_gap':max(v['SD_rank_gap'] for v in match),'max_enriched_missing_fraction':max(g['selected_missing_fractions']['enriched']),'max_matched_missing_fraction':max(g['selected_missing_fractions']['matched'])})
    cv=pd.DataFrame(coverage);cv.to_csv(R/'PATHWAY_COVERAGE.csv',index=False)
    # Never average RNA-risk values across separate group coordinate systems.
    group_arms=['R0_128']+['G'+str(g)+'_128' for g in range(1,8)];groups=seed[seed.arm.isin(group_arms)].copy();groups['group']=groups.arm.map({a:'g'+str(i) for i,a in enumerate(group_arms)});groups.to_csv(R/'GROUPING_8_REPLICATES.csv',index=False)
    group_mean=groups.groupby('group')[['Harrell','Uno60']].mean().reset_index();group_mean.to_csv(R/'GROUPING_8_SUMMARY.csv',index=False)
    factorial=[]
    for b in [128,512,1024]:
        for metric in ['Harrell','Uno60']:
            r='R0_'+str(b);n='N_'+str(b);w='W_'+str(b)
            for init in cfg['seeds']:
                get=lambda arm:float(seed[(seed.arm==arm)&(seed.seed==init)][metric].iloc[0])
                factorial.append({'patches_per_slide':b,'seed':init,'metric':metric,'RealRNA':get(r),'NullRNA':get(n),'WrongRNA':get(w),'Real_minus_Null':get(r)-get(n),'Real_minus_Wrong':get(r)-get(w),'scope':'same patients,within-fold pairs; patch SIZE512px@20X unchanged'})
    pd.DataFrame(factorial).to_csv(R/'FACTORIAL_PATCH_RNA.csv',index=False)
    d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));t=d['DSS_time'];e=d['DSS_event'];ff=d['folds'];weights=np.zeros(512)
    for f in cfg['folds']:weights[ff==f]=ipcw(t[ff!=f],e[ff!=f],t[ff==f],e[ff==f])
    prows=[]
    for arm in active_arms:
        for init in cfg['seeds']:
            first=np.load(F/'probes'/f'{arm}_seed{init}_fold1.npz')
            for name in [k for k in first if k!='test_indices']:
                sums=np.zeros(4)
                for f in cfg['folds']:
                    z=np.load(F/'probes'/f'{arm}_seed{init}_fold{f}.npz');ix=z['test_indices'];hn,hd=components(t[ix],e[ix],z[name]);un,ud=components(t[ix],e[ix],z[name],weights[ix]);sums+=np.array([hn,hd,un,ud])
                prows.append({'arm':arm,'seed':init,'probe':name,'Harrell':sums[0]/sums[1],'Uno60':sums[2]/sums[3],'kind':cfg['arms'][arm]['kind'],'scope':'WrongRNA probes describe wrong-paired inputs,not true recipient RNA;64D mean token readout'})
    pd.DataFrame(prows).to_csv(R/'RNA_SIGNAL_PROBES.csv',index=False)
    subgroup_rows=[];subsets={'COAD':d['cancer_type']=='COAD','READ':d['cancer_type']=='READ','RNA_complete':np.isnan(d['raw_rna']).sum(1)==0,'RNA_missing3024':np.isnan(d['raw_rna']).sum(1)==3024}
    for arm in cfg['arms']:
        for init in cfg['seeds']:
            for name,selected in subsets.items():
                sums=np.zeros(4)
                for f in cfg['folds']:
                    z=np.load(F/'oof'/f'{arm}_seed{init}_fold{f}.npz');keep=selected[z['test_indices']];ix=z['test_indices'][keep];risk=z['risk'][keep]
                    hn,hd=components(t[ix],e[ix],risk);un,ud=components(t[ix],e[ix],risk,weights[ix]);sums+=np.array([hn,hd,un,ud])
                events=int(e[selected].sum());subgroup_rows.append({'arm':arm,'seed':init,'subgroup':name,'n':int(selected.sum()),'events':events,'Harrell':sums[0]/sums[1] if sums[1] else None,'Uno60':sums[2]/sums[3] if sums[3] else None,'Harrell_comparable':sums[1],'Uno_weighted_comparable':sums[3],'scope':'LOW_EVENT_DESCRIPTIVE_ONLY' if events<20 else 'DESCRIPTIVE_ONLY'})
    pd.DataFrame(subgroup_rows).to_csv(R/'SUBGROUP_METRICS.csv',index=False)
    decisions=[]
    for name,frame in paired.groupby('comparison',sort=False):
        primary=name in cfg['primary'];positive=bool((frame.mean_delta>0).all());consistent=bool((frame.positive_seeds==3).all());passed=bool((frame.simultaneous_low>0).all());affected=set(a for a,c in ({**cfg['primary'],**cfg['secondary']}[name]));limited=any(v['arm'] in affected for v in budget_sensitive)
        if primary and positive and consistent and passed and not limited:label='SUPPORTED_EXPLORATORY_INTERNAL'
        elif positive:label='DIRECTIONAL_ONLY'
        elif (frame.simultaneous_high<0).all() and not limited:label='NOT_SUPPORTED_IN_THIS_CONFIGURATION'
        else:label='INCONCLUSIVE'
        decisions.append({'comparison':name,'scope':'primary' if primary else 'secondary','decision':label,'both_metrics_positive':positive,'all3_seeds_both_positive':consistent,'corrected_both_intervals_positive':passed,'budget_sensitive':limited})
    decision={'overall':'SUPPORTED_EXPLORATORY_INTERNAL' if any(v['decision']=='SUPPORTED_EXPLORATORY_INTERNAL' for v in decisions) else 'INCONCLUSIVE','comparisons':decisions,'training_budget_sensitive_cells':budget_sensitive,'research_scope':'first execution/audit,external reviewer pending; repeatedly studied512/65 cohort','records':465,'new_fits':390,'new_neural_fits':360,'new_Ridge_fits':30,'exact_parent_reuse_records':75}
    write('decision.json',json.dumps(decision,indent=2));write('DECISION.md','# 预锁规则下的初步审计判断\n\n'+table(pd.DataFrame(decisions))+'\n\n这是第一轮可复算证据，不构成外部审计或临床泛化裁判。')
    def contrast(name):
        frame=paired[paired.comparison==name]
        return '；'.join(f'{v.metric} Δ={v.mean_delta:+.5f},95%CI[{v.CI95_low:+.5f},{v.CI95_high:+.5f}],校正/次要区间[{v.simultaneous_low:+.5f},{v.simultaneous_high:+.5f}],正向seed{v.positive_seeds}/3' for v in frame.itertuples())
    answers=['# Phase12 — C4机制确认','',f'**465条记录，390次新拟合（360神经+30RNA-only Ridge），75条历史风险精确复用。总体规则标签：{decision["overall"]}。**','',
        '## 机制关系与信息边界','',
        'RNA分组改变基因信息如何被压缩为8个Token；Patch128/512/1024改变额外局部检索分支的采样覆盖；Real/Null/Wrong改变该RNA上下文是否包含正确患者的信息。冻结的S32始终来自原完整TITAN表征，预算只约束新增CONCH局部分支。原特征粒度始终512px@20X，本轮不检验切块物理尺寸。',
        '', '## 直接回答：目前支持什么？','',
        '**总体为 INCONCLUSIVE：六个主要比较的原始95%和12项校正区间均包含零。128 Patch存在一致的患者RNA正向趋势，但尚未达到预锁的内部确认门槛。**',
        '', '|检验|完整结果的解释|', '|---|---|',
        '|正确患者RNA|128下Real同时高于Null/Wrong，两指标均3/3 seed正向；患者抽样区间仍含零。512下Real−Wrong的Uno60为负，不能推广为各Patch预算普遍成立。|',
        '|随机分组|g0在锁定8组中两项均值最高；其余7组平均较低，但配对区间含零，不能断言g0是统计异常。8组平均并未确认超过A1。|',
        '|Patch数量|512−128的RNA增量差中差为负且区间含零，1024也未给出增强证据。不能把局部采样数量称为物理Patch粒度改变。|',
        '|真实通路分组|覆盖修复到512/512后，真实−随机的Harrell略正、Uno60为负，未验证生物共同分组收益。|',
        '|线性风险锚点|移除锚点后总性能下降，RNA相对Null的两指标方向分裂。当前结果明显依赖直接C+S32读出；不等于已隔离其唯一原因。|',
        '|审计|全部465记录、150训练内输入矩阵、独立指标及患者级差中差复核通过；外部审计未完成。|',
        '', 'RNA分组决定512个基因如何形成8×64D的RNA tokens；RNA token作为Query检索局部CONCH Patch，同时保留RNA Query残差；Clinical/S32还提供其它查询与线性风险路径。因而Real/Null/Wrong比较的是这条完整RNA条件路径的作用，不能独自分离RNA主效应与RNA–Patch交互。',
        '', '**矩阵范围：Patch×RNA因子只在g0下测试；八随机分组的稳健性只在128 Patch下测试。没有完整的分组×Patch×RNA三因素矩阵，不能声称关系适用于所有分组。**',
        '', '## 同协议全表','',table(summary),'','## 六个必须回答的问题','',
        '### 1. 正确患者RNA是否同时超过常量与错配？','',contrast('Real128-Null128')+'\n\n'+contrast('Real128-Wrong128'),
        '', '**回答：128下点估计同时优于两对照且跨seed正向，患者级抽样证据不足；目前为DIRECTIONAL_ONLY。** 只有两种对照都满足双指标、三seed与校正区间门槛，才有较强内部患者RNA贡献证据。Null保留8个可学习常量Token、inter-token网络、Query和所有融合槽；其基因投影参数不参与预测，有效容量差异单列。Wrong保留全部真实RNA编码参数，并分别在四split内部错配。',
        '', '### 2. g0是特例，还是8种随机分组普遍有效？','',table(group_mean),
        '',f'固定8组的Harrell均值跨组SD={group_mean.Harrell.std():.5f}，Uno跨组SD={group_mean.Uno60.std():.5f}；8组平均Harrell={group_mean.Harrell.mean():.5f}，Uno60={group_mean.Uno60.mean():.5f}。这些组共享512患者，不是8批独立队列。其余7组平均−g0：'+contrast('MeanOtherGroups-Group0')+'\n\n8组平均相对Null：'+contrast('Mean8Groups-Null')+'\n\n8组平均相对A1：'+contrast('Mean8Groups-A1'),
        '', '### 3. 512 Patch是否提高RNA真实增量？','',contrast('Real512-Null512')+'\n\n'+contrast('Real512-Wrong512')+'\n\n**配对差中差**：'+contrast('DID512vs128'),
        '', '**回答：未支持512提高真实RNA增量；差中差的两指标点估计均为负，且Uno60三个seed均为负，但区间仍含零，不能反过来确认更大覆盖必然有害。** 应看Real−Null的增量及差中差，不以模型总分提高代替RNA作用增强。1024为预锁次要敏感性；不同覆盖也可能改变优化难度，训练预算敏感单元单列。',
        '', '### 4. 充分覆盖的真实Hallmark是否优于匹配随机？','',f'真实基因覆盖：原Top512={int(cv.original_covered.min())}–{int(cv.original_covered.max())}/512；富集={int(cv.enriched_covered.min())}–{int(cv.enriched_covered.max())}/512。可测通路候选={int(cv.eligible_pathway_genes.min())}–{int(cv.eligible_pathway_genes.max())}。同基因集真实对随机：'+contrast('EnrichedTrue-Random128')+'\n\n'+contrast('EnrichedTrue-Random512'),
        '', '**回答：原来的约4%覆盖不足已修复，但未得到真实分组优于随机的双指标证据。这限制当前分组和融合配置的归因，不能否定RNA通路信息本身。**',
        '', '基因集合改变本身的控制：'+contrast('EnrichedRandom-MatchedSet128')+'\n\n'+contrast('EnrichedRandom-MatchedSet512'),
        '',f'512个匹配控制均来自固定注释之外，与富集集合不重叠；其中严格训练均值/SD分箱匹配{int(cv.strict_match_genes.min())}–{int(cv.strict_match_genes.max())}/512，其余按预锁最近rank距离回退，不伪称完全匹配。单个富集随机分组不能代表完整生物随机零分布；语义标签未作为自然语言输入，检验的是基因共同分组结构。',
        '', '### 5. 去掉C+S32线性风险路径后是否仍有贡献？','',contrast('NoAnchorReal-Null')+'\n\n'+contrast('NoAnchor-Anchored'),
        '', '**回答：去掉锚点后未复现双指标一致的RNA收益；Harrell总分下降的次要95%区间完全低于零，Uno60区间略跨零。** No-anchor仅由一个联合非线性head输出风险，原0.1尺度和优化器保持；无有效线性加法路径。直接原始C/S32读出被移除，容量和优化条件也随之改变，因此这是诊断而不是纯交互因果检验。继承的Ridge初始化计算结果存于零输出buffer且不贡献风险，未恢复锚点。',
        '', '### 6. 独立审计、问题与边界？','',f'75历史checkpoint及原H5/缓存、全部465正式checkpoint、{replay["preprocessing_arrays_independently_rebuilt"]}独立训练内输入矩阵、隔离评价器/sksurv、25次显式重复患者bootstrap（包含差中差）均通过；{len(budget_sensitive)}个训练预算敏感单元。见独立程序和源hash，允许授权外部审计员在服务器重跑。',
        '', '## RNA独立信号与训练有效性','',
        table(summary[summary.arm.isin(['RNA512_ref','RNA_H1','RNA_MR'])][['arm','Harrell','Uno60']]),
        '', 'RNA-only富集集合的线性Cox点估计高于原Top512及匹配非通路集合。这是预锁次要表示诊断；没有额外注册该差值的配对区间，不将点估计提高表述为显著性或融合机制已成立。基因集合信号与同一集合的分组语义是两个不同问题。',
        '',f'{neural}个新增神经cell均完成80内层epochs，全部inner/refit训练loss下降；选中epoch范围{int(pd.DataFrame(training).selected_epoch.min())}–{int(pd.DataFrame(training).selected_epoch.max())}，最低epoch单元{int(pd.DataFrame(training).at_minimum.sum())}，80边界单元{int(pd.DataFrame(training).at_maximum.sum())}，预锁预算敏感单元{len(budget_sensitive)}。没有触发同步延长预算的条件；这仅排除当前选择规则的预算边界迹象，不证明全局最优或所有架构优化难度相同。',
        '',table(pd.read_csv(R/'FIXED_MODEL_SENSITIVITY_SUMMARY.csv')),
        '', '固定原C4的15个checkpoint下，训练均值输入及20次锁定split内错配只改测试RNA，权重逐元素未变。匹配RNA在三个seed、两项指标均高于20错配上下文均值；它说明模型对RNA身份有响应，不把20上下文当20独立样本，也不代替训练时Real/Wrong配对检验。',
        '', '## Bug与执行修复','',
        '- 1024 Patch原五路并发实际OOM，保存失败日志/原调度源码后，仅将该预算的并发降到四路；完成记录不改，仅恢复缺失cell。数学训练source hash与数值协议不变，见EXECUTION_AMENDMENT。',
        '- 最后测试敏感性审计曾将GPU状态与CPU快照直接torch.equal而报错；原源码/日志已保留，改为同在CPU上精确比较，全部15个checkpoint重新完成扰动。未影响任何训练或正式OOF。',
        '- 发布前修复复跑说明：初始parent HEAD校验改为parent祖先与原文件hash校验，允许从发布后的Phase12提交复跑；命令补齐最终统计重建。只修复交付入口，旧训练source hash不变。',
        '- 两项故障均已修复，没有已知未解决的执行Bug。源码审计与独立统计PASS仍不等于外部裁判验收。',
        '', '## 重要归因限制','',
        '- Real相对Null/Wrong的正向趋势涉及模型中患者RNA信息的可用贡献；RNA→Patch块有RNA Query残差，仍包含RNA主效应。没有同RNA表示的纯加性对照，不能单凭本轮宣称语义交互优于加性信息。',
        '- 训练规则完全继承Phase11 C4：risk=联合可训练C+S32线性通路+0.1deep head；不能称全新无残差主架构。',
        '- 8随机组和3初始化都共享患者。Bootstrap只含固定已训练风险的患者抽样方差，不包含重新训练、历史模型选择、内层选择或重估训练IPCW。',
        '- Ridge各seed重复拟合同一训练数据与正则配置，不能把这些记录解释为独立初始化样本。',
        '- RNA固定3024缺失区块仅作平台代理，未证实测序平台身份；所有重要子组只描述，少事件无法判断。',
        '- RNA probe是8个token均值64D与train-PCA16，不能把低分解释为完整Token信息论损失。',
        '- 输入/数学执行核验置信度高；机制普适性与外部泛化证据有限。所有新结果仍是反复研究队列的探索性内部证据。',
        '', '## 优势与风险','', '优势：三RNA条件×三嵌套预算、独立分组、严格训练内生物/随机集合控制、测试时固定模型敏感性、全checkpoint与独立差中差复核。风险：仅65事件，常量输入有效自由度不同，No-anchor优化尺度与信息读出改变，单个富集随机组及内部重复研究。',
        '', '## 拓展思考 / 你可能想知道','', '先确认患者RNA贡献与差中差是否成立，再讨论需要哪种表示或交互改动。随机组赢家不能作为医学知识增益证据；更大Patch总分提高也不等于RNA增量提高。']
    write('RESULTS.md','\n'.join(answers));write('README.md','# Phase12\n\n先读RESULTS/DECISION，复算见REPRODUCE。匿名聚合证据可公开；患者矩阵、OOF、checkpoint与映射仅服务器保留。')
    write('RUN_MANIFEST.json',json.dumps({'status':'COMPLETE_AUDITED',**{k:decision[k] for k in ['records','new_fits','new_neural_fits','new_Ridge_fits','exact_parent_reuse_records']},'source_hash':json.loads((R/'HARD_GATES.json').read_text())['source_hash'],'private_directory':str(P),'branch':'codex/phase12-c4-mechanism-confirmation'},indent=2))
    write('INDEPENDENT_VALIDATION.json',json.dumps({'status':'PASS','OOF_metrics':validation,'checkpoint_and_provenance':replay,'literal_bootstrap':literal,'parent':json.loads((R/'PARENT_HARD_GATE.json').read_text()),'external_auditor_completed':False},indent=2))
    bugs=P/'BUG_JOURNAL.json';entries=json.loads(bugs.read_text()) if bugs.exists() else []
    pd.DataFrame(entries,columns=['severity','stage','bug','affected','repair','post_hoc','source_hash']).to_csv(R/'BUG_LOG.csv',index=False)
    telemetry=json.loads((F/'GPU_TELEMETRY.json').read_text());use=[v for v in telemetry if v.get('active',0)>0 and 'GPU' in v]
    write('GPU_UTILIZATION_SUMMARY.json',json.dumps({'samples':len(use),'sampling_seconds':10,'mean_busy_percent':float(np.mean([float(v['GPU'].split(',')[0]) for v in use])),'peak_memory_MB':max(float(v['GPU'].split(',')[1]) for v in use),'scope':'resumed scheduler only,including complete-cell skips andCPU Ridge/probe work;original scheduler telemetry retained in resource_failure_1024_workers5;not a full-run utilization estimate','interpretation':'includesCPU probes andloading;GPUcached tensors audited exactly equivalent;per-cell loader time ishost submission,not GPU kernel duration'},indent=2))
    write('REPRODUCE.md','# 复算与运行\n\n## 依赖与私有输入\n\n真实版本见SOURCE_REGISTRY；Python3.12.3/torch2.5.1+cu124、sksurv0.23.1隔离目录/root/phase10_trimodal_eval_deps。设置PYTHONPATH该目录、CUBLAS_WORKSPACE_CONFIG=:4096:8、OPENBLAS/OMP/MKL_NUM_THREADS=2。大矩阵与519CONCH H5及Phase11嵌套缓存沿用原只读路径，完整hash见DATA_AUDIT。\n\n## 新输出隔离\n\n/root/phase12_c4_private/；不覆盖Phase11。复跑请使用新目录并修改run_phase12中PRIVATE及对应独立审计路径；同源COMPLETE会skip，禁止覆盖历史患者预测。\n\n```bash\npython -m unittest tests.test_phase12_c4 -v\npython scripts/prepare_phase12.py\npython scripts/audit_phase12_sources.py\npython scripts/gate_phase12.py\npython scripts/schedule_phase12.py\npython scripts/validate_phase12_independent.py --statistics\npython scripts/replay_phase12.py\npython scripts/audit_phase12_statistics.py\npython scripts/audit_phase12_test_sensitivity.py\npython scripts/report_phase12.py\npython scripts/package_phase12.py\npython scripts/check_phase12_delivery.py\n```\n\n单cell：`python scripts/run_phase12.py --arms R0_512 --seed 31 --fold 1`。仅重建原OOF指标：`python scripts/validate_phase12_independent.py --statistics`。新矩阵/比较以phase12配置的arms、primary、secondary字段为准；合并协议中primary_comparisons/exact_aliases/patch_arms等旧Phase11字段只为历史继承记录，本轮不执行。发布分支允许Phase11 parent SHA的后继提交，仍逐文件核对原历史源码不变。矩阵配置中6个主要线性对比包括配对差中差；不能从不同risk尺度跨fold比较患者。原始临床DSS标签重建由独立raw_source脚本完成。\n\n训练入口和原Phase11网络forward真实交付，不只PASS文件；授权外部审计员可读取私有OOF/checkpoint重新运行。\n')
    print(json.dumps(decision))


if __name__=='__main__':main()

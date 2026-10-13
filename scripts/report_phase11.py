"""Aggregate audited predictions only; do not alter model configurations."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from validate_phase11_independent import components,ipcw
R=ROOT/'reports/phase11_semantic_anchored_trimodal';P=Path('/root/phase11_semantic_anchor_private');OUT=P/'formal'


def write(name,text): (R/name).write_text(text+'\n',encoding='utf8')

def table(frame):
    cols=list(frame.columns);lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame.itertuples(index=False,name=None):lines.append('|'+'|'.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+'|')
    return '\n'.join(lines)


NAMES={'A0':'Clinical Ridge','A1':'Clinical+TITAN S32 Ridge','A2':'Clinical+TITAN native PCA32 Ridge','A3':'RNA Top512 Ridge','A4':'Clinical+S32+RNA Top512 Ridge',
       'A5':'C+S32+Patch, unguided Patch queries, no RNA','A6':'C/S32<->Patch, no RNA','A7':'A6+RNA tokenwise late concat',
       'B0':'C/S32<->Patch and RNA->Patch, independent QKV','B1':'C/S32<->Patch and RNA->S32','B2':'C->RNA->Patch plus C/S32<->Patch',
       'B3':'B0 reverse RNA path: Patch->RNA','B4':'B0 bidirectional RNA<->Patch','B5':'C/constant queries<->Patch, S32 late head, RNA->Patch','B6':'B0 without any effective RNA input',
       'B7':'B0 split-local wrong-patient RNA','C0':'REUSED_IDENTICAL B0, sequential8x64 supports + GroupMSA','C1':'sequential8x64 supports, same-weight tokenwise QKV/FF',
       'C2':'Top512 patient gene-wise soft weighted token projection','C3':'Top512 Hallmark membership+explicit uncovered fallback','C4':'C3 train-bin-matched global gene permutation','C5':'C1+RNA-only auxiliary Efron(.1)',
       'D0':'B0 split-local wrong-patient S32','D1':'REUSED_IDENTICAL B5: S32 only final head/linear anchor','D2':'B0 Age/Sex/Stage clinical subset','D3':'A6+RNA aux(.1), RNA prohibited from prediction',
       'P0':'REUSED_IDENTICAL A6 128 patch','P1':'A6 512 patch','P2':'A6 1024 patch','P3':'REUSED_IDENTICAL B0 128 patch','P4':'B0 512 patch','P5':'B0 1024 patch'}


def main():
    cfg=json.loads((ROOT/'configs/phase11_semantic_anchored.json').read_text(encoding='utf-8-sig'))
    audit=json.loads((R/'ALL_INDEPENDENT_VALIDATION.json').read_text());replay=json.loads((R/'REPLAY_AND_PROVENANCE_AUDIT.json').read_text());literal=json.loads((R/'LITERAL_BOOTSTRAP_AUDIT.json').read_text())
    if any(z['status']!='PASS' for z in [audit,replay,literal]):raise RuntimeError('final audit incomplete')
    s=pd.read_csv(R/'SEED_OOF_METRICS.csv');fold=pd.read_csv(R/'FOLD_METRICS.csv');paired=pd.read_csv(R/'PAIRED_COMPARISONS.csv')
    summary=s.groupby('arm').agg(Harrell=('Harrell','mean'),Harrell_init_SD=('Harrell','std'),Uno60=('Uno60','mean'),Uno_init_SD=('Uno60','std')).reset_index()
    summary=summary.merge(fold.groupby('arm').agg(worst_Harrell=('Harrell','min'),worst_Uno60=('Uno60','min')).reset_index(),on='arm',how='left')
    summary['model']=summary.arm.map(NAMES);summary.to_csv(R/'MODEL_SUMMARY.csv',index=False)
    diagnostics=[];matrix=[];grad=[];recorded=0;fits=0;neurals=0;tail_flags=[]
    for arm in cfg['arms']+list(cfg['patch_arms'])+['P0','P3']:
        cells=[json.loads((OUT/'cells'/f'{arm}_seed{seed}_fold{f}.json').read_text()) for seed in cfg['seeds'] for f in cfg['folds']]
        recorded+=len(cells);fits+=sum(c['actual_fit'] for c in cells)
        params=[c['parameters'] for c in cells];active=[];nonzero=[]
        for c in cells:
            if not c['actual_fit']:continue
            if 'inner' not in c:continue
            neurals+=1;logs=c['inner']['logs'];fl=c['outer']['logs'];selected=c['selected_epoch'];tail=np.array([r['val_EMA'] for r in logs[-10:]])
            slope=float(np.polyfit(np.arange(len(tail)),tail,1)[0]);boundary=selected==len(logs);sensitive=boundary and slope<0
            if sensitive:tail_flags.append({'arm':arm,'seed':c['seed'],'fold':c['fold']})
            active.append(max(r['connected_parameters'] for r in fl));nonzero.append(max(r['nonzero_gradient_elements'] for r in fl))
            diagnostics.append({'arm':arm,'seed':c['seed'],'fold':c['fold'],'inner_epochs':c['inner']['actual_epochs'],'selected_epoch':selected,
                 'refit_epochs':c['outer']['actual_epochs'],'inner_train_loss_first':logs[0]['train_loss_before'],'inner_train_loss_last':logs[-1]['train_loss_after'],
                 'inner_val_EMA_at_selected':logs[selected-1]['val_EMA'],'inner_val_EMA_last':logs[-1]['val_EMA'],'inner_val_Harrell_at_selected':logs[selected-1]['val_Harrell'],
                 'final10_EMA_slope':slope,'at_minimum':selected==cfg['minimum_epoch'],'at_maximum':boundary,'TRAINING_BUDGET_SENSITIVE':sensitive,
                 'refit_train_loss_first':fl[0]['train_loss_before'],'refit_train_loss_last':fl[-1]['train_loss_after'],'correction_sd':fl[-1]['correction_sd'],'correction_max':fl[-1]['correction_max'],
                 'attention_entropy_last':fl[-1]['entropy'],'connected_parameters':active[-1],'nonzero_gradient_elements':nonzero[-1],
                 'seconds':c['seconds'],'peak_MB':max(c['inner']['peak_cuda_MB'],c['outer']['peak_cuda_MB'])})
            for role,rows in [('inner',logs),('outer',fl)]:
                for module in rows[-1]['gradients']:
                    grad.append({'arm':arm,'seed':c['seed'],'fold':c['fold'],'partition':role,'module':module,
                                 'first_norm':rows[0]['gradients'][module],'last_norm':rows[-1]['gradients'][module],'mean_norm':np.mean([r['gradients'][module] for r in rows])})
        matrix.append({'arm':arm,'model':NAMES[arm],'records':15,'actual_fits':sum(c['actual_fit'] for c in cells),
                      'parameters_min':min(params),'parameters_max':max(params),'connected_parameters_min':min(active) if active else None,'connected_parameters_max':max(active) if active else None,
                      'nonzero_gradient_elements_min':min(nonzero) if nonzero else None,'nonzero_gradient_elements_max':max(nonzero) if nonzero else None,
                      'S32_used':arm not in ['A0','A2','A3'],'RNA_prediction':arm not in ['A0','A1','A2','A5','A6','B6','D3','P0','P1','P2'],
                      'patch_budget':cells[0].get('patch_budget',0),'replication':'BASELINE' if arm.startswith('A') else 'ADAPTED/INSPIRED'})
    pd.DataFrame(diagnostics).to_csv(R/'TRAINING_DIAGNOSTICS.csv',index=False);pd.DataFrame(matrix).to_csv(R/'MODEL_MATRIX.csv',index=False);pd.DataFrame(grad).to_csv(R/'MODULE_GRADIENTS.csv',index=False)
    module_comps=['A6-A5','B0-B5','B0-B6','B0-B7','C3-C4','C5-C1','D3-A6'];paired[paired.comparison.isin(module_comps)].to_csv(R/'MODULE_ABLATIONS.csv',index=False)
    paired[paired.comparison.str.contains('P')].to_csv(R/'PATCH_BUDGET_COMPARISONS.csv',index=False)
    paired[paired.comparison.isin(['B0-B7','B0-D0','D3-A6','B0-B6'])].to_csv(R/'NEGATIVE_CONTROLS.csv',index=False)
    decisions=[]
    for name,frame in paired.groupby('comparison',sort=False):
        primary=frame.iloc[0]['family']=='primary';both_positive=bool((frame.mean_delta>0).all());all_seeds=bool((frame.positive_seeds==3).all());passed=bool((frame.simultaneous_low>0).all())
        if primary and passed and all_seeds:status='SUPPORTED_INTERNAL'
        elif both_positive:status='DIRECTIONAL_ONLY'
        elif bool((frame.simultaneous_high<0).all()) and bool((frame.positive_seeds==0).all()):status='NOT_SUPPORTED_IN_CONFIGURATION'
        else:status='INCONCLUSIVE'
        a,b=name.split('-');aliases=cfg['exact_aliases'];affected={v['arm'] for v in tail_flags}
        budget_limited=bool({aliases.get(a,a),aliases.get(b,b)}&affected)
        if primary and budget_limited:status='INCONCLUSIVE' if status=='SUPPORTED_INTERNAL' else status
        decisions.append({'comparison':name,'family':frame.iloc[0]['family'],'decision':status,'both_point_directions_positive':both_positive,'all_3_seeds_both_positive':all_seeds,'both_simultaneous_intervals_positive':passed,'training_budget_sensitive_comparison':budget_limited})
    decision={'overall':'SUPPORTED_INTERNAL' if any(r['decision']=='SUPPORTED_INTERNAL' for r in decisions if r['family']=='primary') else 'INCONCLUSIVE',
              'comparisons':decisions,'training_budget_sensitive_cells':tail_flags,'scope':'exploratory internal, repeatedly studied cohort, no external validation',
              'records_main_matrix':390,'records_additional_patch':60,'bookkeeping_patch_aliases':30,'record_files_total':recorded,
              'actual_fits':fits,'actual_neural_fits':neurals,'actual_Ridge_fits':fits-neurals,'old_v1_invalid_excluded':114,'formal_status':'COMPLETE_AUDITED'}
    write('decision.json',json.dumps(decision,indent=2,ensure_ascii=False))
    # RNA-only frozen probes: predictions are private; publication is aggregate.
    d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));t=d['DSS_time'];e=d['DSS_event'];ff=d['folds'];weights=np.zeros(512)
    for f in range(1,6):weights[ff==f]=ipcw(t[ff!=f],e[ff!=f],t[ff==f],e[ff==f])
    probes=[]
    for arm in ['C0','C1','C2','C3','C4','C5']:
        for seed in cfg['seeds']:
            first=dict(np.load(OUT/'probes'/f'{arm}_seed{seed}_fold1.npz'))
            for name in [k for k in first if k!='test_indices']:
                sums=np.zeros(4)
                for f in range(1,6):
                    z=dict(np.load(OUT/'probes'/f'{arm}_seed{seed}_fold{f}.npz'));ix=z['test_indices'];hn,hd=components(t[ix],e[ix],z[name]);un,ud=components(t[ix],e[ix],z[name],weights[ix]);sums+=np.array([hn,hd,un,ud])
                probes.append({'arm':arm,'seed':seed,'probe':name,'Harrell':sums[0]/sums[1],'Uno60':sums[2]/sums[3]})
    pd.DataFrame(probes).to_csv(R/'RNA_SIGNAL_PROBES.csv',index=False)
    subgroup=[]
    subsets={'COAD':d['cancer_type']=='COAD','READ':d['cancer_type']=='READ','RNA_complete':np.isnan(d['raw_rna']).sum(1)==0,'RNA_missing3024':np.isnan(d['raw_rna']).sum(1)==3024}
    for stage in np.unique(d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str)):subsets['Stage_'+stage]=d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str)==stage
    for arm in ['A1','A3','A6','B0','B1','B2','B7','C0','C1','C3','C4','P1','P2','P4','P5']:
        for seed in cfg['seeds']:
            for name,selected in subsets.items():
                sums=np.zeros(4)
                for f in range(1,6):
                    z=dict(np.load(OUT/'oof'/f'{arm}_seed{seed}_fold{f}.npz'));keep=selected[z['test_indices']];ix=z['test_indices'][keep];r=z['risk'][keep]
                    hn,hd=components(t[ix],e[ix],r);un,ud=components(t[ix],e[ix],r,weights[ix]);sums+=np.array([hn,hd,un,ud])
                subgroup.append({'arm':arm,'seed':seed,'subgroup':name,'n':int(selected.sum()),'events':int(e[selected].sum()),'Harrell':sums[0]/sums[1] if sums[1] else None,'Uno60':sums[2]/sums[3] if sums[3] else None,'Harrell_comparable':sums[1],'Uno_weighted_comparable':sums[3],'interpretation':'LOW_EVENT_DESCRIPTIVE_ONLY' if e[selected].sum()<20 else 'DESCRIPTIVE_ONLY'})
    pd.DataFrame(subgroup).to_csv(R/'SUBGROUP_METRICS.csv',index=False)
    bio=[]
    for fold in range(1,6):
        for role in ['inner','outer']:
            a=json.loads((P/'preprocessing'/f'fold{fold}_{role}.json').read_text());g=a['groups']
            bio.append({'fold':fold,'partition':role,'biologic_covered':g['biologic_covered'],'uncovered_fallback':g['fallback_genes'],'sizes':str([len(v) for v in g['true_members']]),'singleton_bins':g['singleton_bins']})
    pd.DataFrame(bio).to_csv(R/'RNA_GROUP_COVERAGE.csv',index=False)
    write('DECISION.md','# Phase11 固定规则裁决\n\n**'+decision['overall']+'**。以下统计全部是同一既有队列的探索性内部证据。\n\n'+table(pd.DataFrame(decisions)))
    primary=paired[paired.family=='primary'][['comparison','metric','mean_delta','CI95_low','CI95_high','simultaneous_low','simultaneous_high','positive_seeds']]
    main_summary=summary[summary.arm.isin(cfg['arms']+list(cfg['patch_arms']))][['arm','Harrell','Harrell_init_SD','Uno60','Uno_init_SD','worst_Harrell','worst_Uno60']]
    nums={r.arm:(r.Harrell,r.Uno60) for r in summary.itertuples()}
    fmt=lambda arm:f'{arm}: {nums[arm][0]:.5f} / {nums[arm][1]:.5f}'
    results=['# Phase11 — S32语义锚定三模态机制审计','',f'**正式完成390主矩阵记录+60新增patch记录；另30条P0/P3记账复用。有效正式拟合{fits}次，其中{neurals}神经拟合、{fits-neurals} Ridge拟合。** B0→C0、B5→D1共30主矩阵记录逐元素复用。初版114记录因双向路径缺失INVALID，另12个B0保存失败已重训；这些及门控训练开销不混入有效fit计数。旧版证据保留且不参与此表。',
             '', '**总体裁决：'+decision['overall']+'。** 512患者、65 DSS事件；五固定患者折×3初始化。每seed拼五折后按fold-local numerator/denominator micro，再计算3seed均值；不是15cell平均。Ridge三seed相同不构成独立重复。',
             '', '## 同协议主表','',table(main_summary),'','## 八项主要机制比较','',table(primary),'','## 十项研究问题','',
             '1. **S32实际进入所有深度网络**：Scalar MLP+固定词项ID形成32token；C/S32→Patch和Patch→C/S32共用CP分支QKV；B5/D1保留S32晚融合但移除S32内容参与attention。另有C+S32线性初值通路联合训练，非固定风险残差或in-sample stacking。模块梯度、输入扰动及全部checkpoint回放独立验收。Phase10深度主干未输入的TITAN患者级语义在本轮显式输入；仍不等价病理patch医学标注。',
             '2. **强锚点与深度CP**：'+fmt('A1')+'；'+fmt('A5')+'；'+fmt('A6')+'。A1不是PCA32，'+fmt('A2')+'单列。A6−A1、A6−A5区间在paired表；不能只按最高点估计称胜出。A1系数与当前输入逐元素回放，独立Efron梯度验收。',
             '3. **RNA条件增量**：'+fmt('A3')+'是RNA线性独立信号；B0−A6、B0−B6、B0−B7、B0−A7均属主要family，必须同时看双指标、校正区间及三seed方向。RNA-only可预测不能推出给定Clinical+S32+Patch后的增量。详见下表裁决。',
             '4. **Query对象/临床引导**：'+fmt('B0')+'；'+fmt('B1')+'；'+fmt('B2')+'。三者RNA/patch/scalers/预算/80epoch selector统一；B3反向、B4双向均实际方向hook通过。没有使用外折挑选某一路成为新确认性假设。',
             '5. **RNA表征保真**：同Top512的C0–C5保存前/后raw/processed64D及trainPCA16 probe；probe lambda由严格inner模型表征选定。见RNA_SIGNAL_PROBES.csv。低维probe不证明信息论丢失；投影/缩放/正则与联合训练改变可解码性。'+fmt('C0')+'；'+fmt('C1')+'；'+fmt('C2')+'；'+fmt('C5')+'。',
             '6. **真实分组对照**：'+fmt('C3')+'；'+fmt('C4')+'；C3−C4为预锁描述性比較。真实Hallmark只覆盖Top512的一部分，剩余有非生物补充槽，覆盖逐折公开；随机分组保持大小/所有重叠且在train均值/SD分箱内置换。只有一个随机分组，不能证明所有医学通路先验有效或无效。',
             '7. **Patch预算**：A6基准128与P1/P2的512/1024、B0与P4/P5同理；128⊂512⊂1024在全部519slide核对。少patch不重复补足。原总1,278,642；三档实际66,432/261,064/474,959。线性query×patch交互，无全局patch² attention；同80epoch/optimizer/selector，覆盖、runtime和训练曲线同报。S32始终来自原TITAN全slide编码；这些百分比只限制新增CONCH局部分支，不能称整个模型只接触5.2%的病理信息。',
             '8. **点估计赢家与双指标**：最高Harrell为'+summary.loc[summary.Harrell.idxmax(),'arm']+'，最高Uno60为'+summary.loc[summary.Uno60.idxmax(),'arm']+'。这两个排名是探索性描述；只有预锁paired门槛能支持机制，不能把排名包装成SOTA。对强A1的系统比较单列探索性CI。',
             f'9. **Bug/重训**：P0初版漏CP反向，保存114旧记录、原源码tar及INVALID声明；在查看外折性能前修复v2并从头重训全部正式矩阵。随后B0保存inner/outer probe数组重复键名失败，P2仅改键命名；已独立回放A组120 checkpoint和50预处理矩阵，A原OOF/cell未改且保留旧真实源码hash，使用明确source lineage，不无意义重训。未完成B0诊断日志已归档并重训。合成门控还修复任意loss差阈值和字典属性读取错误；均未产生有效正式结果。当前独立OOF/sksurv、逐checkpoint、train-only原矩阵重建、25次字面展开bootstrap全通过。当前有{len(tail_flags)}个最大epoch且末10 EMA下降的单元，保留TRAINING_BUDGET_SENSITIVE，不宣称所有优化收敛。',
             '10. **下一阶段边界**：先按下表辨别哪项机制有内部支持；若无联合family证据，保留强C+S32并停止把新增attention/更大patch/通路组自动视作有效。RNA表征probe和合法inner模型选择可提供局部诊断，但65事件和重复研究使泛化创新仍需独立队列。禁止从单次阴性推断所有深度融合均无效。',
             '',table(pd.DataFrame(decisions)), '', '## 优势、劣势与置信度','',
             '- 优势：S32在网络中可验、同输入负对照、full-risk-set Efron、nested preprocessing/selector、共享患者paired重采样及全checkpoint复放。',
             '- 劣势/风险：一个反复研究的队列，仅65 events；注册参数不等于有效容量；Hallmark未覆盖全部Top512；bootstrap不含重新拟合/模型选择或训练IPCW再估计；某些预算边界仍未饱和。',
             '- 置信度：当前指标与执行正确性高；机制普适归因中；外部临床泛化低。此评级不是未经数据支持的百分比。',
             '', '## 拓展思考 / 你可能想知道','', 'S32内容查询通过计算路径验收，只证明网络能利用该输入；是否带来生存增量仍由配对负对照决定。扩大patch覆盖同时改变优化难度，不能把负结果全部归因于RNA无信号。']
    write('RESULTS.md','\n'.join(results))
    write('RNA_PREPROCESS_AND_TOKEN_AUDIT.md','# RNA输入与Token\n\n所有inner/outer各自train-only中位数/标准化/90%观察率Top512方差筛选，gene索引/符号/mask/具体membership在私有manifest。C0–C5基因完全相同；C3未知基因明确fallback。\n\n'+table(pd.DataFrame(bio))+'\n\nRNA原表309完整+203缺3024；未证实真实平台身份。原log-cache与当前float32核存在舍入差；float64 log正确舍入参考≤1 float32 ULP，历史缓存未改。')
    write('ARCHITECTURE_GRADIENT_AUDIT.md','# 计算图审计\n\nScalarConceptTokenizer、RNA、CP/RP QKV、Patch projection、head都有预期非零有限梯度，至少2seed×2batch合成验证，最终真实checkpoint扰动验收。CP两方向及B0/B1/B2/B3/B4真实query shape hook通过。B6/A5/A6/D3无RNA预测；D3 auxiliary仅RNA模块接收其监督梯度。\n\n注册、grad非None、非零梯度元素分别报告，不能称所有arm有效容量严格相同。C2增加活跃gate；B2额外C→RNA模块活跃；无RNA臂有inactive RNA参数。D3 main loss不能更新RNA，其额外监督仅诊断，且dropout RNG消费与A6不同，不能视作纯容量因果比较。head*.1是尺度，不是严格risk bound。\n\n默认CP真正S32查询；A5 unguided learned queries；B5/D1同模型重编号复用且不能当两项独立机制证据。梯度范数不等于模态因果贡献。')
    data=json.loads((R/'PATIENT_SLIDE_ALIGNMENT_AUDIT.json').read_text())
    write('DATA_AUDIT.md','# 数据复查\n\n当前512/65、519匹配CONCH H5、原RNA20531→20511、CNA25131→25101、TITAN native768/S32固定32均实查。各H5 SHA/整数非负唯一coords/finite/encoder512@20X审计通过。UNI2为256@20X，本轮没有任何按patch行号局部配对。缺失原DX1未替换，同患者原匹配DX2保留。患者全slide同折，等权slide聚合且budgets不变病例集合。\n\n原临床/DSS逐元素重建、源RNA/CNA及原fold清单审计通过，详见raw_*json。训练预处理由独立第二程序重建共50输入矩阵，checkpoint全部逐患者回放。\n\n'+json.dumps(data,indent=2))
    write('S32_SEMANTIC_SOURCE_AUDIT.md','# S32语义来源\n\n来自原固定outcome-blind32个CRC形态/临床病理描述词及两模板，官方TITAN视觉proj+L2和text embedding cosine，非native PCA32或softmax概率。缓存patient_representations与当前队列逐元素一致；确切revision/protocol/vocab hashes见SOURCE_REGISTRY。Foundation权重冻结且本轮不重新编码/微调。\n\ntrainable concept ID仅固定词项身份，不自动带医学语义；S32→CONCH局部查询属于患者条件的学习相关性，不是已标注patch概念真值，也没有使用CONCHv1 text混合v1.5特征。')
    write('PAPER_CODE_DIFFERENCES.md','# Paper/code与本轮\n\n|来源|核查|本轮差异|\n|---|---|---|\n|SurvPGC官方指定commit|双路径不对称context/P交互，源码版本及sha已核验|structured Clinical、S32 scalar concepts、Top512 gene tokens、Cox替换foundation omics/text及离散hazards；ADAPTED|\n|CiMP官方指定commit|六组SNN、GroupMSA、bottlenecks与临床晚融合；原timm布局groups成batch|batch-first8tokens、临床→RNA与线性query/patch；INSPIRED，未取得全文不称faithful|\n\n[SurvPGC官方](https://github.com/Houjiaxin123/SurvPGC)，[CiMP官方](https://github.com/DeepMed-Lab-ECNU/CiMP)。固定版本文件SHA见PAPER_SOURCE_RECHECK.json；本机working copy CRLF与git blob LF等价，不是假定不同模型。')
    write('FAULT_INJECTION_REPORT.md','# 故障注入\n\nFAULT_INJECTION.json保留19项合成错误payload、实际拒绝日志、修复后clean通过。涵盖DSS隐名输入、patient跨slide折、全队列scaler/TopK/PCA、网格错配、语义/RNA detach、mini-risk-set Cox、padding、跨split错配、event反向、risk反向及checkpoint/OOF/metadata错误。合成测试不包含患者记录；实际checkpoint反向risk亦由逐元素回放门控拒绝。实际test扰动两次完整80epoch内层和fresh refit全state bitwise一致。')
    write('KNOWN_LIMITATIONS.md','# 限制\n\n- 512/65同队列反复研究，所有显著性探索性；三seed只随机初始化。\n- 一个inner split，80epoch允许最小10；部分边界未饱和，完整训练不等于全局收敛。\n- B0/C0、B5/D1、P0/A6、P3/B0精确复用，不能当独立证据。\n- 不同模块有效参数与gate/aux不同，匹配注册数不等于功能容量。\n- Hallmark仅覆盖部分Top512，fallback不是医学先验；一个随机对照，不是完整随机基因集零分布。\n- 模型CP/RP为原论文适配，不是官方EXACT；未微调foundation。\n- patient S32与局部CONCH条件关联不证明概念定位或因果基因—病理关系；UNI2网格未局部配对。\n- 30,000患者bootstrap不含重新训练/inner选择方差，也未重估训练IPCW；MC极端分位区间记录。\n- subgroup事件稀少，只描述，不追逐显著性。CNA本轮无正式模型；未更换终点。\n- v1缺反向CP的114记录INVALID并保存；v2全矩阵重跑，不混排名。\n- 私有患者OOF/checkpoint不公开；公开包不能独立重算无授权患者数据。Git历史已有的旧公开记录未重写。\n- D3仅诊断，监督更新不进入主预测，额外RNA计算消耗dropout RNG，不保证A6与D3逐位相同。\n- S32强anchor联合可训练，不是固定风险上界；可能被训练损害，修正幅度实际记录。')
    bugs=[{'severity':'P0','stage':'pre_outer_inspection','bug':'v1 missing reverse Patch->C/S32','affected':'114 preserved records; all v1 declared INVALID','repair':'v2 CP bidirectional, unit direction hooks, all fresh fits','post_hoc_outer_based':False},
          {'severity':'P2','stage':'B0_durable_save','bug':'inner/outer representation duplicate kwargs','affected':'12 B0 save failures; zero COMPLETE B records','repair':'namespace inner keys; A120/50 transforms replay unchanged, original source hashes retained, failed B0 fits rerun','post_hoc_outer_based':False},
          {'severity':'P2','stage':'synthetic_preflight','bug':'arbitrary loss-difference>1 assertion','affected':'no valid formal cells','repair':'require unequal full/subset loss with tolerance and independent Efron oracle','post_hoc_outer_based':False},
          {'severity':'P2','stage':'data_preflight','bug':'dict read as attribute in cache count','affected':'no cache/formal cells completed','repair':'count actual H5 shapes before allocation','post_hoc_outer_based':False}]
    amendment=json.loads((R/'P2_SOURCE_AMENDMENT.json').read_text())
    for bug in bugs:
        bug['repair_source_hash']=amendment['new_source_hash'] if bug['stage']=='B0_durable_save' else amendment['prior_source_hash']
        bug['source_hash_scope']='first repaired formal training lineage; earlier synthetic-only edit not claimed as a separate Git commit'
        bug['retrain_evidence']='CHECKPOINT_REPLAY.csv and RUN_MANIFEST.json' if bug['stage'] in ['pre_outer_inspection','B0_durable_save'] else 'HARD_GATES.json; no valid formal cells affected'
    pd.DataFrame(bugs).to_csv(R/'BUG_LOG.csv',index=False)
    write('INDEPENDENT_AUDIT.md','# 独立验收\n\n450正式记录由隔离评价器直接读私有OOF/time/event/manifest重算，并逐cell核对scikit-survival。指标脚本不import训练数学或特征模块。另有train-only独立原矩阵重建50输入，420实际fit checkpoint回放，其中75Ridge保存系数与独立事件循环梯度通过；345神经checkpoint复放及模态/padding属性通过。25次bootstrap显式重复展开核对450项差，非先平均风险。\n\n模型forward复放使用交付网络，不是第二个完整神经实现；独立小矩阵oracle、query shape hook和streamed全loss梯度校验负责其数学路径。公开读者仍需授权私有输入，不能从PASS文字独立复算。')
    combined={'status':'PASS','OOF':audit,'checkpoint_provenance':replay,'literal_bootstrap':literal,'hard_gates':json.loads((R/'HARD_GATES.json').read_text())}
    write('INDEPENDENT_VALIDATION.json',json.dumps(combined,indent=2))
    telemetry=json.loads((OUT/'GPU_TELEMETRY.json').read_text());good=[r for r in telemetry if 'GPU' in r and r.get('active_jobs',0)>0]
    def values(row):return list(map(float,row['GPU'].split(',')))
    write('GPU_UTILIZATION_SUMMARY.json',json.dumps({'samples':len(good),'sampling_seconds':10,'mean_busy_percent':np.mean([values(r)[0] for r in good]),'peak_memory_MB':max(values(r)[1] for r in good),'stage_means':{stage:np.mean([values(r)[0] for r in good if r['family']==stage]) for stage in cfg['family_order']},'interpretation':'sampled telemetry after P2 resume; A in this stream is cache reuse, not original full A training. Full-fit A peak/time remain per cell. Includes I/O/CPU probes; not complete history or instantaneous benchmark'},indent=2))
    registry=json.loads((R/'SOURCE_REGISTRY.json').read_text());registry['paper_sources']=json.loads((R/'PAPER_SOURCE_RECHECK.json').read_text())['sources'];registry['training_source_hash']=json.loads((R/'HARD_GATES.json').read_text())['source_hash'];registry['training_protocol_sha256']=digest_file(ROOT/'configs/phase11_semantic_anchored.json');write('SOURCE_REGISTRY.json',json.dumps(registry,indent=2))
    write('RUN_MANIFEST.json',json.dumps({'status':'COMPLETE_AUDITED',**{k:decision[k] for k in ['records_main_matrix','records_additional_patch','bookkeeping_patch_aliases','record_files_total','actual_fits','actual_neural_fits','actual_Ridge_fits']},'training_source_hash':registry['training_source_hash'],'private_directory':str(P),'public_branch':'codex/phase11-semantic-anchored-trimodal'},indent=2))
    write('RUN_INSTRUCTIONS.md','# 运行与审计\n\n## 环境\n\nPython3.12.3 / torch2.5.1+cu124 / numpy2.1.3 / pandas2.2.3 / scipy1.14.1 / sklearn1.5.2 / h5py3.16.0 / scikit-survival0.23.1；完整依赖见SOURCE_REGISTRY。设置OPENBLAS/OMP/MKL_NUM_THREADS=2、CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONPATH包含隔离sksurv依赖目录。\n\n## 私有输入约定\n\n/root/phase7a_foundation_headroom_v1/private_dataset.npz；/root/autodl-tmp/phase10_conch；/root/phase10_trimodal_private/cache/private_cohort_manifest.csv与private_fixed_patient_folds.csv；/root/phase6a_titan_runtime/cache_v1/patient_representations.npz。sha见来源表。原H5、矩阵和标签不在公开包，先取得用户授权。改变根路径时同时审查entrypoints中的PRIVATE和来源路径；不得拿不同队列悄悄替代。\n\n## 顺序\n\n```bash\npython -m unittest tests.test_phase11_semantic -v\npython scripts/prepare_phase11.py\npython scripts/audit_phase11_sources.py\npython scripts/gate_phase11.py\npython scripts/schedule_phase11.py --workers 12\npython scripts/replay_phase11.py\npython scripts/audit_phase11_statistics.py\npython scripts/report_phase11.py\npython scripts/package_phase11.py\n```\n\n单cell：`python scripts/run_phase11.py --arms B0 --seed 31 --fold 1`。同源码已有COMPLETE会skip；重跑请使用全新独立输出目录并更新PRIVATE常量，不能覆盖旧OOF。只重建评测：`python scripts/validate_phase11_independent.py --statistics`。每stage evaluator --family A/B/C/D/P。`recheck_phase11_papers.py`在本地已存在固定官方git clone时运行，server归档其结果；不是训练依赖。\n\n源码改变即拒绝旧cell/source hash；不能改HARD_GATES为PASS。P0必须归档旧版并按影响重训。父分支当前SHA9040f5c已fetch实际核对；不以旧报告分数作程序验证。')
    write('README.md','# Phase11 交付\n\n先读RESULTS/DECISION/KNOWN_LIMITATIONS；运行见RUN_INSTRUCTIONS。PUBLIC包只含源码、配置、聚合统计、真实验收证据。PRIVATE包仅用户服务器，不入Git。全部主矩阵+预锁patch预算完成，CNA探索未做。公开历史ref仍保留旧实验；本轮无患者标识输出。')
    print(json.dumps(decision,ensure_ascii=False))


def digest_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if __name__=='__main__':main()

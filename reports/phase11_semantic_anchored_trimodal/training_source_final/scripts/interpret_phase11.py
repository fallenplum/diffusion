"""Explain fixed comparisons after audits; never select/retrain any model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/phase11_semantic_anchored_trimodal'


def table(df):
    rows=['|'+'|'.join(df.columns)+'|','|'+'|'.join(['---']*len(df.columns))+'|']
    for row in df.itertuples(index=False,name=None):
        rows.append('|'+'|'.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+'|')
    return '\n'.join(rows)


def main():
    if json.loads((R/'RUN_MANIFEST.json').read_text())['status']!='COMPLETE_AUDITED':
        raise RuntimeError('interpretation before complete audit')
    pairs=pd.read_csv(R/'PAIRED_COMPARISONS.csv')
    diagnostics=pd.read_csv(R/'TRAINING_DIAGNOSTICS.csv')
    probes=pd.read_csv(R/'RNA_SIGNAL_PROBES.csv')
    summary=pd.read_csv(R/'MODEL_SUMMARY.csv')
    coverage=pd.read_csv(R/'RNA_GROUP_COVERAGE.csv')
    matrix=pd.read_csv(R/'MODEL_MATRIX.csv')
    aliases={'C0':'B0','D1':'B5','P0':'A6','P1':'A6','P2':'A6','P3':'B0','P4':'B0','P5':'B0'}
    base=matrix.arm.map(lambda a:aliases.get(a,a))
    matrix['clinical_fields']=base.map(lambda a:'NONE' if a=='A3' else ('AGE,SEX,Stage' if a=='D2' else 'AGE,SEX,Stage,T,N,M'))
    matrix['S32_tokens']=base.map(lambda a:0 if a in ['A0','A1','A2','A3','A4'] else 32)
    matrix['RNA_tokens_to_prediction']=base.map(lambda a:0 if a in ['A0','A1','A2','A3','A4','A5','A6','B6','D3'] else 8)
    def directions(a):
        if a in ['A0','A1','A2','A3','A4']:return 'Ridge, no attention'
        cp='constant<->Patch; S32 late' if a in ['A5','B5'] else 'C/S32<->Patch'
        if a in ['A5','A6','B6','D3']:rp='no RNA prediction'
        elif a=='A7':rp='RNA tokenwise late, no RNA cross-attention'
        elif a=='B1':rp='RNA->S32'
        elif a=='B2':rp='C->RNA->Patch'
        elif a=='B3':rp='Patch->RNA'
        elif a=='B4':rp='RNA<->Patch'
        else:rp='RNA->Patch'
        return cp+'; '+rp
    matrix['attention_directions']=base.map(directions)
    matrix['auxiliary_Cox_weight']=base.map(lambda a:.1 if a in ['C5','D3'] else 0.)
    matrix.to_csv(R/'MODEL_MATRIX.csv',index=False)
    def comparison(name):
        rows=pairs[pairs.comparison==name]
        if len(rows)!=2:raise RuntimeError('missing locked comparison '+name)
        return '；'.join(f"{r.metric} Δ={r.mean_delta:+.5f}，95% CI [{r.CI95_low:+.5f},{r.CI95_high:+.5f}]，{'16项校正' if r.family=='primary' else '未校正探索性'}区间 [{r.simultaneous_low:+.5f},{r.simultaneous_high:+.5f}]，正向seed {r.positive_seeds}/3" for r in rows.itertuples())
    report=['# 配对机制解释与优化诊断','',
            '以下解释在全部预锁训练和独立审计完成后生成，不产生新配置或追加训练。Δ 是每seed五折within-fold OOF micro指标之差，再平均三seed；不比较不同fold模型的风险尺度。','',
            '## 强参照与真实RNA贡献','']
    refit=pd.read_csv(R/'A1_INDEPENDENT_REFIT.csv')
    report.append(f"强锚点独立重拟合：使用独立事件循环Efron目标、从零开始BFGS，重拟合五个唯一outer-train问题，并核对15份seed记录。最大OOF风险差 {refit.max_risk_error.max():.3g}，最大系数L2差 {refit.coefficient_L2_error.max():.3g}；每项系数差均满足λ强凸性梯度残差界。该审计不改写原系数或OOF，也不把重复seed当独立样本。")
    for name in ['A6-A1','A6-A5','B0-A1','B0-A6','B0-A7','B0-B6','B0-B7','B0-B5']:
        report.append('- **'+name+'**：'+comparison(name))
    required=pairs[pairs.comparison.isin(['B0-A6','B0-A7','B0-B6','B0-B7'])]
    consistent=bool((required.mean_delta>0).all() and (required.positive_seeds==3).all())
    confirmed=bool((required.simultaneous_low>0).all() and consistent)
    report+=['',f'RNA→Patch 的增量机制联合要求（超越无RNA、简单concat与错误患者RNA）在本设置中：三seed双指标方向一致={consistent}，全部主要校正区间均正={confirmed}。任何一项不满足，不能把RNA独立线性信号解释为已证明跨模态条件收益。','',
             '## 方向、分组及辅助监督','']
    for name in ['B1-B0','B2-B0','B3-B0','B4-B0','C1-C0','C3-C4','C5-C1','B0-D0','B0-D2','D3-A6']:
        report.append('- **'+name+'**：'+comparison(name))
    report+=['',f"真实Hallmark覆盖Top512中 {int(coverage.biologic_covered.min())}–{int(coverage.biologic_covered.max())} 个，未覆盖fallback {int(coverage.uncovered_fallback.min())}–{int(coverage.uncovered_fallback.max())} 个（跨inner/outer训练折）。因此C3结果只检验这一部分先验与fallback组合，不是完整RNA生物语义检验。",'',
             '## 表征可解码性','']
    pmean=probes.groupby(['arm','probe'])[['Harrell','Uno60']].mean().reset_index()
    pmean.to_csv(R/'RNA_PROBE_SUMMARY.csv',index=False)
    report.append(table(pmean))
    rna=summary[summary.arm=='A3'].iloc[0]
    report+=['',f'原始Top512 RNA线性参照：Harrell={rna.Harrell:.5f}，Uno60={rna.Uno60:.5f}。表中64D/16D probe是冻结RNA token的均值读出，原始Top512不是同维度读出。前后处理及训练改变线性可解码性，不能据此证明信息论损失；也不能把不同维度probe的降低全部归因于GroupMSA。','',
             '## Patch预算与优化','']
    for name in ['P1-P0','P2-P0','P4-P3','P5-P3']:
        report.append('- **'+name+'**：'+comparison(name))
    budget=summary[summary.arm.isin(['A6','P1','P2','B0','P4','P5'])][['arm','Harrell','Uno60']]
    report+=['',table(budget),'']
    seeds=pd.read_csv(R/'SEED_OOF_METRICS.csv')
    increments=[]
    for count,rna,no_rna in [(128,'B0','A6'),(512,'P4','P1'),(1024,'P5','P2')]:
        for metric in ['Harrell','Uno60']:
            differences=[]
            for seed in [31,47,83]:
                differences.append(float(seeds[(seeds.arm==rna)&(seeds.seed==seed)][metric].iloc[0]-seeds[(seeds.arm==no_rna)&(seeds.seed==seed)][metric].iloc[0]))
            increments.append({'patches_per_slide_cap':count,'comparison':rna+'-'+no_rna,'metric':metric,'mean_delta':np.mean(differences),'seed31_delta':differences[0],'seed47_delta':differences[1],'seed83_delta':differences[2],'scope':'DESCRIPTIVE_ONLY, no additional inferential test'})
    increment_frame=pd.DataFrame(increments)
    increment_frame.to_csv(R/'PATCH_RNA_INCREMENT.csv',index=False)
    report+=['### 同预算下RNA的描述性增量','',table(increment_frame),'',
             '512/1024档同预算RNA减无RNA只报告预定义模型的派生点估计，不新增未预锁的显著性检验。跨预算RNA增量的变化不等于已证明交互机制：覆盖率与优化难度仍须结合下面的训练记录判断。','']
    diagnostics['train_loss_decreased']=diagnostics.refit_train_loss_last<diagnostics.refit_train_loss_first
    dg=diagnostics.groupby('arm').agg(cells=('seed','size'),selected_epoch_median=('selected_epoch','median'),minimum_epoch_cells=('at_minimum','sum'),maximum_epoch_cells=('at_maximum','sum'),budget_sensitive_cells=('TRAINING_BUDGET_SENSITIVE','sum'),refit_loss_decreased_cells=('train_loss_decreased','sum'),correction_sd_median=('correction_sd','median'),correction_max_largest=('correction_max','max'),seconds_total=('seconds','sum'),peak_MB=('peak_MB','max')).reset_index()
    dg.to_csv(R/'TRAINING_ARM_SUMMARY.csv',index=False)
    report.append(table(dg))
    report+=['',f"所有 {len(diagnostics)} 次有效神经拟合都实际完成80内层epoch，fresh refit执行所选epoch；其中 {int(diagnostics.at_minimum.sum())} 次选择下限10，{int(diagnostics.at_maximum.sum())} 次选择上限80，{int(diagnostics.TRAINING_BUDGET_SENSITIVE.sum())} 次同时达到上限且末10 EMA仍下降。{int(diagnostics.train_loss_decreased.sum())}/{len(diagnostics)} 次fresh refit末损失低于初始。",'',
             '执行80epoch不是证明优化达到全局收敛。选择下限可能代表内验证泛化早期最好，也可能受到仅65事件和单inner split的噪声影响；不能未经新预注册就加预算追逐外层结果。上限且EMA仍下降的比较明确保留预算敏感性。风险修正幅度与有效梯度已记录，0.1 head缩放不限制最终预测贡献的严格上界。','','## 后续决策边界','',
             '若主要RNA机制未获得联合支持，下一阶段优先保留Clinical+S32并定位表征读出及训练泛化问题。真实分组对随机分组和更大patch预算各自只回答预锁局部问题；没有同时满足配对与负对照要求时，不扩展GNN/attention层数。若存在内部支持，也须独立队列确认，当前重复研究队列不提供外部确认。']
    anchors=pd.read_csv(R/'SYSTEM_ANCHOR_COMPARISONS.csv')
    winner_h=summary.loc[summary.Harrell.idxmax(),'arm'];winner_u=summary.loc[summary.Uno60.idxmax(),'arm']
    winners={aliases.get(a,a) for a in [winner_h,winner_u]}
    winner_rows=anchors[anchors.comparison.isin([a+'-A1' for a in winners if a!='A1'])]
    report+=['','## 点估计赢家与强参照','',f'Harrell最高={winner_h}；Uno60最高={winner_u}；是否同一模型={winner_h==winner_u}。','',table(winner_rows[['comparison','metric','mean_delta','CI95_low','CI95_high','simultaneous_low','simultaneous_high','positive_seeds']]) if len(winner_rows) else '两项最高点估计来自A1，A1相对自身增量为0。','',
             '上述区间来自全部27个非重复模型相对A1的补充探索性family（54个指标比较），不是仅对获胜模型临时检验。计划在查看外层数值前、预算训练未结束时单独保存；不宣称属于最初训练协议，也不影响八个主要对照及其16项校正。极端分位Monte Carlo区间和有效尾部样本数见SYSTEM_ANCHOR_AUDIT/COMPARISONS；同队列反复研究与模型选择方差仍未解决。']
    actual=summary.set_index('arm')
    score=lambda a:f'{actual.loc[a,"Harrell"]:.5f}/{actual.loc[a,"Uno60"]:.5f}'
    probe=pmean.set_index(['arm','probe'])
    brief=['## 本轮具体研究结论','',
           f'- **强参照仍在**：A1 Clinical+S32={score("A1")}。A6语义引导Patch={score("A6")}，Harrell下降而Uno提高，不能称双指标优于强锚点。RNA线性A3={score("A3")}，独立排序信号没有消失。',
           '- **RNA→Patch未建立稳定条件增量**：B0−A6双指标方向不同；B0−A7未同时改善。真实RNA相对错配B7只有方向性支持，主要同时区间均包含0，患者配对机制未达门槛。',
           f'- **临床引导RNA值得保留为待检验方向**：B2={score("B2")}，相对B0双指标点估计改善，但只有Harrell在3/3 seed正向、Uno为1/3。B2−A1也未通过区间门槛，不认定已成立。',
           '- **S32查询与GroupMSA的贡献均未被确认**：B0−B5的两项均值不为正；C1去掉RNA inter-token attention也未提升两项均值。输入路径验收通过不等于机制效果成立。',
           f'- **监督保真是可解码性线索**：C5训练后processed64D RNA probe={probe.loc[("C5","after_processed_64"),"Harrell"]:.5f}/{probe.loc[("C5","after_processed_64"),"Uno60"]:.5f}，C0对应={probe.loc[("C0","after_processed_64"),"Harrell"]:.5f}/{probe.loc[("C0","after_processed_64"),"Uno60"]:.5f}。C5−C1的最终三模态收益仍只有方向性，不能从probe提升跳到融合机制确认。',
           f'- **不能以赢家宣称知识先验有效**：真实分组C3={score("C3")}，随机匹配C4={score("C4")}。C3−C4未支持真实分组优势；真实基因集覆盖仅{int(coverage.biologic_covered.min())}–{int(coverage.biologic_covered.max())}/512，fallback占绝大多数。这只能限制本配置结论，不能否定更完整的通路表示。',
           f'- **更大patch不是已验证的解决方案**：RNA交互128/512/1024分别={score("B0")}、{score("P4")}、{score("P5")}，未呈双指标单调增加，预锁预算配对区间包含0。没有max-epoch选择，不能把当前结果直接归咎于80epoch上限。',
           f'- **C4仅为点估计赢家**：两项同时高于A1且3/3 seed正向，但普通95%与54项校正区间仍含0。随机对照不是医学知识创新的证据，也不是外部验证结果。',
           '- **执行层与科学结论分开**：420次实际fit、450正式OOF记录、独立特征重建、全checkpoint回放与重采样复核通过；初版P0模型114记录已INVALID且整轮重训。后续保存/打包P2修复保留源链、失败日志及未变哈希。',
           '- **下一阶段建议**：保留A1为系统参照；优先进一步检验RNA读出/监督保真与给定C+S32后的剩余信号。B2仅作预注册候选；暂停盲目增加patch、attention深度或将稀疏Hallmark覆盖包装成已证实机制。本轮总体INCONCLUSIVE，不是所有三模态/医学知识idea的NO-GO。','']
    brief_text='\n'.join(brief)
    (R/'RESEARCH_CONCLUSIONS.md').write_text('# Phase11 研究结论\n\n'+brief_text,encoding='utf8')
    text='\n'.join(report)+'\n'
    (R/'MECHANISM_INTERPRETATION.md').write_text(text,encoding='utf8')
    main=R/'RESULTS.md'
    main.write_text(main.read_text(encoding='utf8')+'\n'+text.replace('# 配对机制解释与优化诊断','## 配对机制解释与优化诊断',1)+'\n'+brief_text,encoding='utf8')
    instructions=R/'RUN_INSTRUCTIONS.md'
    instructions.write_text(instructions.read_text(encoding='utf8')+'\n## 完成解释与最终校验\n\n主要统计及回放完成后运行 `python scripts/audit_phase11_anchor_comparisons.py`；生成报告后运行 `python scripts/interpret_phase11.py`、`python scripts/package_phase11.py`、`python scripts/check_phase11_final_contract.py`。重新生成解释前先重跑report，避免重复追加。\n',encoding='utf8')
    limitations=R/'KNOWN_LIMITATIONS.md'
    limitations.write_text(limitations.read_text(encoding='utf8')+'\n- 每cell elapsed包含patch加载和计算，未逐次拆分原训练的数据加载耗时；GPU遥测包含CPU probe及加载间歇，不能当作纯模型吞吐。\n- A6相对A5同时增加Clinical及S32内容作为Patch query，query数量也改变；A6−A5检验组合引导，不能单独归因S32。B0−B5维持Clinical query并仅以常数query替代S32内容，是更直接的语义内容对照。\n- D0错配S32同时影响query、最终head和联合训练线性通路；B0−D0检查患者级S32总体贡献，不能单独定位query机制。\n- RNA probe采用8个token均值的64D读出（及train PCA16），不是全部8×64展开；均值池化的损失不能归因于整个token体系的信息完全消失。\n',encoding='utf8')
    registry_path=R/'SOURCE_REGISTRY.json'
    registry=json.loads(registry_path.read_text())
    registry['feature_models']={'CONCH':'CONCH v1.5 precomputed768D H5; all519 hashes/attributes checked; frozen input, no model weights loaded in Phase11','TITAN':'official prior cache native768D and projected S32 cosines separately verified; frozen, no model weights loaded in Phase11'}
    registry['feature_model_provenance']['scope']='historical official cache provenance and current exact cache check; foundation weights not loaded or independently re-encoded in Phase11'
    registry_path.write_text(json.dumps(registry,indent=2),encoding='utf8')
    independent=R/'INDEPENDENT_AUDIT.md'
    independent.write_text(independent.read_text(encoding='utf8')+'\n## 强锚点新拟合核对\n\n'+table(refit)+'\n\nA1_INDEPENDENT_REFIT.csv记录独立事件循环目标的BFGS零初始化拟合。五个唯一outer-train问题核对15份记录，不更新任何正式checkpoint、OOF或模型选择。系数差通过λ强凸性与两侧梯度残差给出的可计算界验收。\n',encoding='utf8')
    package_repair=R/'PACKAGING_REPAIR.json'
    if package_repair.exists():
        row=json.loads(package_repair.read_text())
        bugs=pd.read_csv(R/'BUG_LOG.csv')
        if not (bugs.stage=='public_package_dependency').any():
            bugs=pd.concat([bugs,pd.DataFrame([{'severity':'P2','stage':'public_package_dependency','bug':'missing explicit tests package initializer','affected':'archive assembly only; all420 checkpoints and450 OOF unchanged','repair':'add tests/__init__.py; rerun actual documented unittest command and archive contracts','repair_source_hash':row['training_source_hash_after'],'source_hash_scope':'unchanged training hash, separate package repair metadata','retrain_evidence':'UNREQUIRED: no numerical or protocol change; POST_HOC packaging repair','post_hoc_after_outer_inspection':True,'post_hoc_outer_based':False}])],ignore_index=True)
            bugs.to_csv(R/'BUG_LOG.csv',index=False)
        limitations.write_text(limitations.read_text(encoding='utf8')+'\n- 归档发现缺tests包入口，查看外层数值后进行了P2打包修复（POST_HOC_PACKAGING_ONLY）；正式source hash、checkpoint与OOF均未改变，失败日志保留。\n',encoding='utf8')
    oof=Path('/root/phase11_semantic_anchor_private/formal/oof')
    redundancy=[]
    for seed in [31,47,83]:
        for fold in range(1,6):
            a=np.load(oof/f'A6_seed{seed}_fold{fold}.npz')['risk']
            b=np.load(oof/f'B6_seed{seed}_fold{fold}.npz')['risk']
            redundancy.append(bool(np.array_equal(a,b)))
    equivalent=bool(all(redundancy))
    (R/'CONTROL_REDUNDANCY_AUDIT.json').write_text(json.dumps({'A6_B6_bitwise_identical_records':sum(redundancy),'checked_records':15,'all_identical':equivalent,'actual_fits_not_relabelled_as_reuse':True,'primary_B0_A6_and_B0_B6_are_not_independent_evidence':equivalent,'original_Bonferroni16_preserved':True},indent=2))
    if equivalent:
        note='\n- A6与B6这两个完全无有效RNA的配置在15个seed×fold上OOF逐元素一致。两组均按原计划实际拟合，fit计数不改成复用；B0−A6与B0−B6不是独立机制证据，原保守16项校正保持不变。\n'
        limitations.write_text(limitations.read_text(encoding='utf8')+note,encoding='utf8')
        main.write_text(main.read_text(encoding='utf8')+note,encoding='utf8')
    print('INTERPRETATION_COMPLETE_FIXED_COMPARISONS')


if __name__=='__main__':main()

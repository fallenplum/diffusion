"""Anonymous aggregate report, explicit public inventory and reproducible delivery."""
import hashlib,json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from run_phase14 import PRIVATE,REPORT,CONFIG,cfg,source,sha,write


def summary():
    s=pd.read_csv(REPORT/'SEED_OOF_METRICS.csv');c=pd.read_csv(REPORT/'PAIRED_COMPARISONS.csv');d=pd.read_csv(REPORT/'TEACHER_DECODABILITY.csv')
    print('HALL_MEANS\n'+s[s.gene=='HALL'].groupby(['endpoint','model'])[['Harrell','Uno60']].mean().to_string())
    print('DSS_HALL_PRIMARY\n'+c[(c.endpoint=='DSS')&(c.gene=='HALL')&c.primary].to_string(index=False))
    print('DSS_TEACHER\n'+d[(d.endpoint=='DSS')&(d.gene=='HALL')].groupby('model')[['teacher_R2','teacher_pair_agreement']].mean().to_string())
    if (REPORT/'BUDGET_AUDIT.csv').exists():
        b=pd.read_csv(REPORT/'BUDGET_AUDIT.csv');g=pd.read_csv(REPORT/'INTERACTION_GRADIENTS.csv')
        print('BUDGET\n'+b.groupby(['endpoint','family','arm']).agg(selected_median=('selected_epoch','median'),selected_min=('selected_epoch','min'),selected_max=('selected_epoch','max'),ceiling_improving=('ceiling_and_improving','sum')).to_string())
        print('INTERACTION\n'+g[g.arm=='INTERACT'].groupby(['endpoint','gene'])[['left_gradient','right_gradient','mix_gradient','train_correction_RMS','train_correction_max']].mean().to_string())


def finish():
    independent=json.loads((REPORT/'INDEPENDENT_AUDIT.json').read_text());extra=json.loads((REPORT/'ADDITIONAL_AUDIT.json').read_text())
    if independent['status']!='PASS' or extra['status']!='PASS':raise RuntimeError('audit incomplete')
    s=pd.read_csv(REPORT/'SEED_OOF_METRICS.csv');c=pd.read_csv(REPORT/'PAIRED_COMPARISONS.csv');d=pd.read_csv(REPORT/'TEACHER_DECODABILITY.csv');b=pd.read_csv(REPORT/'BUDGET_AUDIT.csv');g=pd.read_csv(REPORT/'INTERACTION_GRADIENTS.csv')
    grouped=s.groupby(['endpoint','gene','model'])[['Harrell','Uno60']].agg(['mean','std']).reset_index();grouped.columns=['_'.join(v).strip('_') for v in grouped.columns];grouped.to_csv(REPORT/'MODEL_SUMMARY.csv',index=False)
    primary=c[c.primary];mechanism=primary[primary.comparison.isin(['INTERACT-ADDITIVE','INTERACT-COX','INTERACT-WRONG_INTERACT'])]
    supported=bool((mechanism.corrected_low>0).all());rank=primary[primary.comparison=='RANK64-TOKEN64'];rank_supported=bool((rank.corrected_low>0).all())
    decision={'status':'POSITIVE_MECHANISM_EVIDENCE' if supported else 'INCONCLUSIVE_MECHANISM','ranking_constraint_confirmed_both_metrics':rank_supported,'interaction_exceeds_additive_convex_and_wrong_both_corrected_metrics':supported,
              'budget_sensitive_inner_cells':int(b.ceiling_and_improving.sum()),'interpretation':'No blanket no-go. Ranking retention,conditional increment andinteraction are separate gates. Single-cohort exploratory findings;no conditional-independence or causal claim.',
              'primary_family_tests':10,'endpoint_counts':{'DSS':{'n':512,'events':65},'PFI':{'n':510,'events':129}},'source_hash':source()}
    write(REPORT/'decision.json',decision)
    means=s.groupby(['endpoint','gene','model'])[['Harrell','Uno60']].mean();text=['# Phase14-R0 RNA 排序保留与条件交互审计','',
        '**完成：90个完整两阶段流程；1710个模型逐折记录。DSS主终点512/65；PFI次要终点510/129。**','',
        '固定三组基因、同一患者折、3 seeds；没有根据本轮测试结果改变通路、维度、损失权重、交互深度或正则。所有数值是每seed先聚合5-fold同折比较对，再平均3seeds；不跨fold排序风险。','',
        '## 1. 主要表征与融合结果','', '|输入/表示（DSS，HALL）|Harrell|Uno60|','|---|---:|---:|']
    for name in ['RAW512','TOKEN512','RANK512','RAW64','TOKEN64','POOL64','RANK64','RANKPOOL64','CS','COXRAW','COX','COXPOOL','ADDITIVE','INTERACT','WRONG_COX','WRONG_INTERACT']:
        a=means.loc['DSS','HALL',name];text.append(f'|{name}|{a.Harrell:.6f}|{a.Uno60:.6f}|')
    text+=['','512D与64D对照分别保持相同读出容量；TOKEN64/RANK64为训练局部PCA64，POOL64为同一encoder的固定均值。原始RNA无上游神经producer，但所有诊断读出的归一化、λ=1 Cox及评测完全一致。PCA与均值不是相同压缩算子，维度相同不等于信息容量相同。','',
           '## 2. 预先锁定的主要配对比较','', '|DSS/HALL比较|指标|Δ|95% CI|Bonferroni10 CI|正向seeds|','|---|---|---:|---|---|---:|']
    for row in primary.itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.CI_low:+.6f}, {row.CI_high:+.6f}]|[{row.corrected_low:+.6f}, {row.corrected_high:+.6f}]|{row.positive_seeds}/3|')
    text+=['','三项交互比较全部通过两指标校正区间，才支持交互机制。只优于静态CS不够；只优于错误配对也不足以说明交互胜过RNA加性主效应。','',
           '**本轮结果解释：** 在64D匹配后，均值池化−完整Token的区间跨零，不能沿用上一轮512D对64D差距宣称已确认纯池化损失。排序约束改善很小，双指标校正区间跨零。匹配RNA交互明显高于错配的Harrell，却未超过加性/条件Cox；错配下降也可能是加入不相关RNA破坏主效应拟合，而非交互本身有效。','',
           '## 3. 排序可解码性','', '|DSS/HALL表示|Held-out教师风险R²|教师患者对排序一致率|','|---|---:|---:|']
    recovery=d[(d.endpoint=='DSS')&(d.gene=='HALL')].groupby('model')[['teacher_R2','teacher_pair_agreement']].mean()
    for name in ['RAW512','RAW64','TOKEN512','TOKEN64','POOL64','RANK512','RANK64','RANKPOOL64']:
        a=recovery.loc[name];text.append(f'|{name}|{a.teacher_R2:.4f}|{a.teacher_pair_agreement:.4f}|')
    text+=['','教师目标在producer训练分区内3折生成，标签不跨入held-out；教师只是训练局部RNA线性风险，不是真实疾病风险。诊断R²或排序一致性改善不能替代DSS增量。教师属于label-OOF：子折共享producer训练分区的固定基因和RNA无监督缩放，非全预处理嵌套OOF；未接触producer的inner-val/outer-test，但此共享预处理是限制。','',
           '从RNA512到完整Token512仍保留较多教师排序，而PCA64和均值64的解码恢复程度下降；这定位了读出压缩限制，但跨模型维度/监督不同，不能解释为生物信息消失的因果证明。原始512D RNA旁路仍应保留，不能用这次弱效排序约束替代。','',
           '## 4. 训练有效性与预算','',f'- 内层producer/readout共{len(b)}次80-epoch选择，选到上限且最后10epoch仍改善：**{int(b.ceiling_and_improving.sum())}**。最低epoch由10修复到1，未依据本轮outer-test进行调整。',
           '- Outer实际训练epoch逐项等于对应inner所选epoch。Clinical+S32系数/风险与父阶段逐元素校验；没有in-sample风险stacking。',
           '- ADDITIVE与INTERACT从相同条件Cox最优主效应开始，冻结同一RANK64；交互只增加CS×RNA的双线性项，预测风险幅度严格受限±0.25。',
           '- INTERACTION_GRADIENTS.csv提供交互左右投影、mix实际梯度与训练风险修正；BUDGET_AUDIT.csv提供选中epoch和末期趋势。总梯度非零不保证交互已学到有用结构。','',
           '## 5. 优势、风险与结论边界','',
           '**优势：** 同一producer内比较完整/池化表示；64D读出匹配；教师交叉拟合；完整inner pipeline选择；错患者保留癌种×缺失代理边际；凸Cox提供优化充分参照。','',
           '**劣势与风险：** 重复使用同一cohort，65个DSS事件；PCA不是监督最优压缩；OOF教师带有噪声；producer的训练表示仍可能in-sample适应；不同producer的监督项是有意实验差异；bootstrap固定已训练模型，未包含全流程重训练不确定性。','',
           '**不能从条件增量不显著推出条件独立，也不能从压缩比较推出信息论意义的不可逆损失。** PFI有不同结局定义，不是纯粹增加事件数的因果检验。RNA平台字段未被确认，本轮strata使用缺失代理，不能称为平台控制。','',
           f'程序判定：**{decision["status"]}**；排序约束双指标确认：{rank_supported}；交互双指标三机制比较确认：{supported}。','',
           '置信度：执行/协议一致性高（独立审计通过）；单cohort生物机制解释有限，按配对区间逐项判断，不以点估计下no-go结论。','',
           '## 6. 审计和故障记录','',
           f'- 1710逐折指标均与scikit-survival交叉校验；checkpoint最大回放误差{independent["max_checkpoint_replay_error"]:.3g}。',
           f'- {extra["literal_bootstrap_checks"]}项逐患者展开bootstrap与矩阵统计吻合，最大误差{extra["literal_bootstrap_max_error"]:.3g}；{extra["rejected_faults"]}项故障拒绝。',
           '- 正式运行首批12任务曾因未排序inner分区使用searchsorted失败；没有写入正式结果。改用显式索引映射并增加硬门后90单元完整重跑。','',
           '## 7. 复现和下一步','',
           '环境与原数据沿用Phase13；服务器Python3.12/torch2.5.1 CUDA/scikit-survival0.23.1。先执行gate_phase14.py，再launch_phase14.py，随后audit_phase14.py和report_phase14.py --finish。',
           '源码hash与父版本、artifact hashes见配置和审计。公共仓库不含患者ID、OOF风险、教师目标、权重、原始数据；这些产物保留服务器私有目录。','',
           '拓展思考：若排序恢复改善但条件增量未改善，下一步应区分RNA主效应冗余与teacher监督方向，不直接扩大交互。若交互幅度/梯度不足，先解释优化路径；它不构成所有融合机制的否定。','',
           '你可能想知道：PFI全部结果单独列于MODEL_SUMMARY.csv和PAIRED_COMPARISONS.csv；次要区间未校正，不能替代DSS主比较。','',
           '参考：[survival knowledge distillation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11387617/)、[官方IPCW定义](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.concordance_index_ipcw.html)。']
    appendix=json.loads((REPORT/'BUDGET_APPENDIX_AUDIT.json').read_text())
    ab=pd.read_csv(REPORT/'BUDGET_APPENDIX_TRAINING.csv')
    primary_budget=ab[(ab.endpoint=='DSS')&(ab.gene=='HALL')&(ab.arm=='INTERACT')]
    primary_ceiling=int(primary_budget.ceiling_and_improving.sum())
    if appendix['status']!='PASS':raise RuntimeError('budget appendix incomplete')
    am=pd.read_csv(REPORT/'BUDGET_APPENDIX_SEED_METRICS.csv').groupby(['endpoint','gene','model'])[['Harrell','Uno60']].mean();ac=pd.read_csv(REPORT/'BUDGET_APPENDIX_PAIRED.csv')
    text+=['','## 8. 统一1280-epoch读出预算附录（看过80-epoch结果后追加）','',
           '该附录只修改共同读出上限80→1280，冻结同一RNA producer/基因/PCA/初始化/损失/正则/错配。原90流程和1710记录不覆盖。四个读出均使用同样上限，并按各自inner EMA选择outer训练epoch。','',
           f'前80 epoch EMA轨迹最大差异：{appendix["first80_EMA_max_error"]:.3g}；另训练360个读出逐折记录，全部独立重算和回放；仍在1280上限改善的内层单元：**{appendix["remaining_ceiling_improving"]}**。','',
           f'其中7项为错患者交互，1项为DSS/MATCH真实交互；DSS/HALL真实交互15个内层选择中，仍受上限截断：**{primary_ceiling}**。此处只能说本HALL对照的epoch选择未被上限截断，不代表所有训练参数达到全局最优。','',
           '|DSS/HALL预算附录|Harrell|Uno60|','|---|---:|---:|']
    for name in ['CS','COX','ADDITIVE','INTERACT','WRONG_INTERACT']:
        aa=am.loc['DSS','HALL',name];text.append(f'|{name}|{aa.Harrell:.6f}|{aa.Uno60:.6f}|')
    text+=['','|DSS/HALL附录比较|指标|Δ|95% CI|Bonferroni24 CI|','|---|---|---:|---|---|']
    for row in ac[(ac.endpoint=='DSS')&(ac.gene=='HALL')].itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.CI_low:+.6f}, {row.CI_high:+.6f}]|[{row.corrected_low:+.6f}, {row.corrected_high:+.6f}]|')
    text+=['','附录用于解除已发现的优化预算疑问；不改写主实验。更长预算仍无优势时，结论仅限这个rank8、固定惩罚和冻住RNA的模型，不否定所有跨模态交互。修复浮点回放差异后重新运行，输入/PCA参数逐元素一致。','',
           '复现附录：budget_phase14.py --launch → evaluate_budget_phase14.py → report_phase14.py --finish。']
    decision['budget_appendix_remaining_ceiling_improving']=appendix['remaining_ceiling_improving']
    decision['DSS_HALL_real_interaction_ceiling_improving']=primary_ceiling
    ad=ac[(ac.endpoint=='DSS')&(ac.gene=='HALL')&ac.comparison.isin(['INTERACT-ADDITIVE','INTERACT-COX','INTERACT-WRONG_INTERACT'])]
    decision['post_inspection_budget_interaction_confirmed']=bool((ad.corrected_low>0).all());write(REPORT/'decision.json',decision)
    (REPORT/'RESULTS.md').write_text('\n'.join(text),encoding='utf8')
    sources=[CONFIG,ROOT/'PHASE14_R0_PROTOCOL.md',ROOT/'src/phase14_rna/__init__.py',ROOT/'src/phase14_rna/models.py']+[ROOT/'scripts'/f'{name}_phase14.py' for name in ['run','gate','launch','evaluate','audit','report']]
    sources += [ROOT/'configs/phase14_budget_appendix.json',ROOT/'scripts/budget_phase14.py',ROOT/'scripts/evaluate_budget_phase14.py',ROOT/'scripts/publish_phase14.py']
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    public=sources+sorted(p for p in REPORT.glob('*') if p.is_file() and p.suffix in ['.csv','.json','.md'] and p.name not in ['PUBLIC_FILES.json','DELIVERY.json'])
    inventory=[]
    for p in public:
        data=p.read_bytes()
        if pattern.search(data):raise RuntimeError('protected identifier '+str(p))
        if p.suffix=='.csv':
            columns=pd.read_csv(p,nrows=0).columns.tolist()
            if set(columns)&{'patient_id','patient_ids','test_indices','risk','time','event','RNA','Clinical','S32'}:raise RuntimeError('patient data column')
        else:columns=[]
        inventory.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':len(data),'columns':columns})
    write(REPORT/'PUBLIC_FILES.json',{'status':'PASS','scope':'source and anonymous model-level aggregate only','files':inventory})
    archive=REPORT/'PHASE14_PUBLIC_AUDIT.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in public:z.write(p,str(p.relative_to(ROOT)))
        z.write(REPORT/'PUBLIC_FILES.json',str((REPORT/'PUBLIC_FILES.json').relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP CRC')
        for item in inventory:
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:raise RuntimeError('ZIP hash')
    delivery={'status':'PASS','pipelines':90,'records':1710,'source_hash':source(),'public_file_count':len(inventory)+1,'public_zip_sha256':sha(archive),'public_zip_bytes':archive.stat().st_size,'private_data_published':False}
    write(REPORT/'DELIVERY.json',delivery);write(PRIVATE/'DONE.json',delivery);summary();print(json.dumps(delivery))


if __name__=='__main__':finish() if '--finish' in sys.argv else summary()

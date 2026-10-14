"""Three cautious cause diagnoses from the complete independently audited matrix."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from validate_phase13_independent import expected_models
R=ROOT/'reports/phase13_rna_information_audit';P=Path('/root/phase13_rna_private');F=P/'formal'


def write(name,text):(R/name).write_text(text.rstrip()+'\n',encoding='utf8')
def table(df):
    header='|'+'|'.join(df.columns)+'|\n|'+'|'.join(['---']*len(df.columns))+'|\n'
    return header+'\n'.join('|'+'|'.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+'|' for row in df.itertuples(index=False,name=None))


def main():
    cfg=json.loads((ROOT/'configs/phase13_rna_information_audit.json').read_text());metrics=json.loads((R/'INDEPENDENT_METRICS.json').read_text());replay=json.loads((R/'REPLAY_AUDIT.json').read_text());literal=json.loads((R/'LITERAL_BOOTSTRAP_AUDIT.json').read_text())
    if metrics['status']!='PASS' or metrics['verified_records']!=3330 or replay['status']!='PASS' or replay['checkpoint_records']!=3330 or literal['status']!='PASS':raise RuntimeError('independent audits incomplete')
    seed=pd.read_csv(R/'SEED_OOF_METRICS.csv');fold=pd.read_csv(R/'FOLD_METRICS.csv');paired=pd.read_csv(R/'ALL_PAIRED.csv')
    summary=seed.groupby(['endpoint','model']).agg(Harrell=('Harrell','mean'),Harrell_seed_SD=('Harrell','std'),Uno60=('Uno60','mean'),Uno60_seed_SD=('Uno60','std'),test_Efron_loss=('test_Efron_loss','mean')).reset_index();summary.to_csv(R/'MODEL_SUMMARY.csv',index=False)
    matrix=[];inventory=[]
    for path in sorted((F/'cells').glob('*.json')):
        c=json.loads(path.read_text());m=json.loads(Path(c['manifest_path']).read_text());design=c['design'];inventory.append({'endpoint':c['endpoint'],'model':c['model'],'seed':c['seed'],'fold':c['fold'],'OOF_sha256':c['prediction_sha256'],'checkpoint_sha256':c['checkpoint_sha256'],'manifest_sha256':c['manifest_sha256'],'actual_fit':c.get('actual_fit'),'source_hash':c['source_hash']})
    inv=pd.DataFrame(inventory);inv.to_csv(R/'ARTIFACT_HASH_INDEX.csv',index=False)
    for ep in cfg['endpoints']:
        for name in expected_models(cfg):
            cells=[json.loads((F/'cells'/f'{ep}_{name}_seed{s}_fold{f}.json').read_text()) for s in cfg['seeds'] for f in cfg['folds']];sample=cells[0];design=sample['design'];scope='RNA-only'
            if design.get('base') or name in ['C','S','CS']:scope='Clinical/pathology conditioned or baseline'
            if design.get('tap','').endswith('_joint') or name.startswith('F_') and name.endswith('_ORIGINAL'):scope='joint Clinical/S32/Pathology/RNA,not RNA-only'
            matrix.append({'endpoint':ep,'model':name,'records':15,'family':design['family'],'producer_kind':design.get('producer_kind','NONE'),'information_scope':scope,'gene_set':design.get('gene','NONE'),'tap':design.get('tap','NONE'),'parameters_min':min(c['parameters'] for c in cells),'parameters_max':max(c['parameters'] for c in cells),'actual_outer_fits':sum(bool(c.get('actual_fit')) for c in cells),'three_seeds_independent_initializations':design['family']!='BASE','RNA_penalty_OFF_outer_folds':sum(c.get('RNA_lambda')=='OFF' for c in cells)/3})
    pd.DataFrame(matrix).to_csv(R/'MODEL_MATRIX.csv',index=False)
    predictability=[];recovery=[]
    for path in sorted((P/'diagnostics').glob('*.json')):
        value=json.loads(path.read_text())
        if isinstance(value,list):recovery.extend(value)
        else:predictability.append(value)
    pd.DataFrame(predictability).to_csv(R/'CS_RNA_RISK_PREDICTABILITY.csv',index=False);pd.DataFrame(recovery).to_csv(R/'REFERENCE_RISK_RECOVERY.csv',index=False)
    train=[];reconstructed=[]
    for path in sorted((P/'producers').glob('*_RNA_ONLY_*.json')):
        c=json.loads(path.read_text())
        for role in ['inner','outer']:
            info=c[role];logs=info['logs'];row={k:c[k] for k in ['endpoint','gene','seed','fold']};row.update(partition=role,epochs=info['actual_epochs'],selected_epoch=info['selected_epoch'],train_loss_first=logs[0]['train_loss_before'],train_loss_last=logs[-1]['train_loss_after'],registered_parameters=info['parameters'],max_connected_parameters=max(v['connected_parameters'] for v in logs),max_nonzero_gradient_elements=max(v['nonzero_gradient_elements'] for v in logs),min_encoder_gradient=min(v['encoder_gradient'] for v in logs),peak_cuda_MB=info['peak_cuda_MB'])
            if role=='inner':
                minimum=min(logs,key=lambda v:v['val_EMA']);selected=logs[info['selected_epoch']-1];slope=float(np.polyfit(np.arange(10),[v['val_EMA'] for v in logs[-10:]],1)[0]);row.update(unrestricted_best_EMA_epoch=minimum['epoch'],selected_EMA=selected['val_EMA'],unrestricted_min_EMA=minimum['val_EMA'],minimum_epoch_rule_cost=selected['val_EMA']-minimum['val_EMA'],final10_EMA_slope=slope,TRAINING_BUDGET_SENSITIVE=info['selected_epoch']==80 and slope<0)
            train.append(row)
    for path in sorted((P/'producers').glob('*_reconstruction.json')):reconstructed.append(json.loads(path.read_text()))
    training=pd.DataFrame(train);training.to_csv(R/'TRAINING_DIAGNOSTICS.csv',index=False);pd.DataFrame(reconstructed).to_csv(R/'FUSION_INNER_RECONSTRUCTION.csv',index=False)
    if len(reconstructed)!=45 or len(training)!=180:raise RuntimeError('producer pipeline counts')
    diagnosis=[]
    for gene in cfg['genes']:
        conditional=paired[(paired.endpoint=='DSS')&(paired.comparison==gene+'_CS_RAW-CS')];retention=paired[(paired.endpoint=='DSS')&(paired.comparison==gene+'_FUSED_processed_full-RAW_fixed1')]
        independent=bool((conditional.corrected_low>0).all()&(conditional.positive_seeds==3).all());loss=bool((retention.corrected_high<0).all()&(retention.negative_seeds==3).all())
        diagnosis.append({'gene':gene,'conditional_increment':'SUPPORTED_EXPLORATORY_ADDITIVE' if independent else 'INCONCLUSIVE_OR_NOT_DETECTED','matched_full_token_readout_loss':'SUPPORTED_IN_THIS_READOUT_CONFIGURATION' if loss else 'INCONCLUSIVE','overlap_causal_claim':'NOT_IDENTIFIED','event_scarcity_causal_claim':'NOT_IDENTIFIED'})
    inner=training[training.partition=='inner'];budget=int(inner.TRAINING_BUDGET_SENSITIVE.sum());too_early=int((inner.unrestricted_best_EMA_epoch<cfg['training']['minimum_epoch']).sum())
    decision={'overall':'EXPLORATORY_DIAGNOSTIC_EVIDENCE','hypotheses':diagnosis,'records':3330,'RNA_only_complete_pipelines':90,'new_inner_reconstructions':45,'parent_final_fusion_checkpoints_reused':45,'RNA_only_budget_sensitive_cells':budget,'RNA_only_min10_selection_restricts_earlier_best_cells':too_early,'external_auditor_completed':False,'repeated_cohort':True,'PFI_is_pure_power_experiment':False}
    write('decision.json',json.dumps(decision,indent=2));write('DECISION.md','# 三种解释的诊断边界\n\n'+table(pd.DataFrame(diagnosis))+'\n\n置信区间含零不能证明条件独立。线性读出损失不等于所有非线性读出都无信息。PFI不是单纯增加DSS事件数的实验。')
    def contrast(name,ep='DSS'):
        frame=paired[(paired.endpoint==ep)&(paired.comparison==name)]
        return '；'.join(f'{row.metric} Δ={row.mean_delta:+.5f},95%CI[{row.CI95_low:+.5f},{row.CI95_high:+.5f}],校正/次要CI[{row.corrected_low:+.5f},{row.corrected_high:+.5f}],正向seed{row.positive_seeds}/3' for row in frame.itertuples())
    wanted=['C','S','CS']+[g+'_'+n for g in cfg['genes'] for n in ['R_tuned','R_fixed1','PCA16_tuned','RESID_tuned','CS_RAW']]
    concise=summary[(summary.endpoint=='DSS')&summary.model.isin(wanted)]
    endpoint_audit=json.loads((R/'ENDPOINT_AUDIT.json').read_text());pfi=endpoint_audit['PFI']
    appendix_summary=pd.read_csv(R/'APPENDIX_FIXED_RAW_SUMMARY.csv');appendix_pairs=pd.read_csv(R/'APPENDIX_PAIRED.csv')
    focused=['HALL_R_fixed1','F_HALL_before_processed_full','F_HALL_after_processed_full','F_HALL_after_processed_mean','F_HALL_after_RNApatch_full','F_HALL_after_RNApatch_mean','RO_HALL_HEAD','RO_HALL_after_processed_full','RO_HALL_after_processed_mean']
    focused_summary=summary[(summary.endpoint=='DSS')&summary.model.isin(focused)][['model','Harrell','Uno60','test_Efron_loss']]
    rows=['# Phase13-R0 — RNA预后信息保留与强条件增量诊断','',
          '**3330条正式模型/读出记录全部完成并独立复算；90个RNA-only训练管线、45个缺失inner状态精确重建、45个既有outer融合checkpoint复用。**','',
          f'三组输入固定Phase12训练折基因索引及分组：Top512、Hallmark富集512、匹配非通路512。固定原5外折×3初始化；DSS512/65为主，原CDR验证后的PFI{pfi["n"]}/{pfi["events"]}为次要。原月值=原CDR天数/30.417，未改历史时间；两名PFI零时间事件患者按历史有效性规则排除。缓存总计131个事件，正式正时间交集129个事件。',
          '', '## 当前最具体的解释','',
          '|可能原因|证据与限制|\n|---|---|\n|与强基线重叠|Hallmark RNA残差排序大幅下降、直接RNA加性增量未确认；CS对RNA风险的R²只约−0.16到0.23，不能说大多数分数方差已被解释，更不能证明条件独立。|\n|编码/融合未保留|完整融合RNA token仍有0.70313/0.65237排序；相对原RNA的主要区间跨零。mean只有0.65044/0.61079，完整−mean的次要差异提示压缩/读出问题；512vs64容量不同，且次要比较未多重校正。|\n|事件稀缺|DSS仅65事件，Uno信用Neff约37.25；PFI有效129事件，Neff约70.75，但RNA条件增量仍不稳定。这个终点对比不能隔离纯功效因素。|',
          '', '## 主要事实表','',table(concise),
          '', 'RNA-only富集集合的点估计较高，但它相对Top512/匹配集合的Harrell与Uno60配对95%区间仍含零；不能把点估计优势当成已确认的基因集合优越性。MODEL_ABSOLUTE_CI提供所有模型绝对区间，属于描述性结果。',
          '', '## 1. 是否主要与Clinical/Pathology重叠？','',
          '联合分块正则Cox分别拟合Clinical/S32与RNA系数；基线正则inner-only确定，RNA五候选包括精确关闭RNA。不用训练内风险stacking。RNA主效应以原512基因直接进入，因此不经过旧神经encoder或RNA→Patch融合。',
          '', *[f'**{gene}：** '+contrast(gene+'_CS_RAW-CS') for gene in cfg['genes']],
          '',table(pd.DataFrame(predictability)),
          '', '预测性重叠是连续诊断：训练内无监督CS→RNA线性map，用heldout RNA-only Cox参考分数检验可预测性；参考分数不是患者真值。相关性高不能证明条件独立；条件增量未检出也可能来自模型限制或区间过宽。Clinical-only/S32-only条件增量及RNA残差模型见ALL_PAIRED。',
          '', '## 2. 编码/融合是否保留RNA排序？','',
          table(focused_summary),
          '本轮同时读出完整8×64 token与64D均值，不再只测mean。匹配主要比较是同512维、同train-local标准化、同Ridge λ=1的原始RNA与融合后完整token。',
          '', *[f'**{gene}：** '+contrast(gene+'_FUSED_processed_full-RAW_fixed1') for gene in cfg['genes']],
          '', *[f'**{gene}均值−完整：** '+contrast(gene+'_Fused_mean-full') for gene in cfg['genes']],
          '', *[f'**{gene}RNA-only学习−原始固定读出：** '+contrast(gene+'_RNAonly_full-RAW_fixed1') for gene in cfg['genes']],
          '', 'PAIR_RETENTION_TRANSITIONS逐事件对报告原正确排序被丢失/原错误排序被挽救，以及Clinical+S32错误对上的RNA线索。REFERENCE_RISK_RECOVERY以train-only RNA Cox风险为参考，检查各级表征的heldout可解码性。此处只识别当前线性读出的信息保留，不能声称信息论上的互信息消失。',
          '', '**均值pooling对照属于预锁次要检验，未做多重比较校正；完整512维与均值64维读出的有效容量/正则也不同。因此现有结果支持进一步检查聚合/读出瓶颈，尚非纯粹信息损失的因果证明。**',
          '', 'RNA-only与融合使用相同RNA tokenizer/分组与匹配初始化，但RNA-only用flatten小head，融合用RNA→Patch/mean/联合head及锚点。差别同时涉及训练目标与producer读出路径，不能单独归因为多任务目标抑制。旧融合latent对PFI属于DSS-producer transfer，PFI新RNA-only则按PFI训练，已单列。',
          '', '## 3. 65个事件是否足以归因？','',table(pd.read_csv(R/'EVENT_INFORMATION_MASS.csv')),
          '', 'DSS/PFI的观测事件数、可比较事件数、IPCW信用集中程度不同；effective_event_credit_Neff是权重集中诊断，不是有效独立患者或正式功效计算。逐事件删除仅作固定模型影响分析，不用于改变主队列/主要比较。Bootstrap不包含重训、历史探索与内层模型选择方差。',
          '', *[f'**PFI {gene}条件增量：** '+contrast(gene+'_CS_RAW-CS','PFI') for gene in cfg['genes']],
          '', 'PFI事件数更多，但生物终点、有效患者、删失与事件时间同时变化。PFI改善最多是与监督稀缺解释一致的敏感性线索，不能证明DSS失败仅因65个事件；若PFI也不改善，也不能否定所有组学互补信息。',
          '', '## 训练与选择规则诊断','',
          f'90个RNA-onlyinner均完成80epochs，选中epoch范围{int(inner.selected_epoch.min())}–{int(inner.selected_epoch.max())}；预算边界敏感单元{budget}。{too_early}个单元在未限制epoch范围时最佳EMA早于10。保留继承的min10规则，未依据新外折成绩改训练预算或选择规则；这个限制可能影响新RNA-only目标，不能把低分只归于encoder结构。完整损失/梯度和epoch表交付。',
          '', '## 研究判断与置信度','',table(pd.DataFrame(diagnosis)),
          '', '## 匹配正则附录：结果查看后的探索性控制','',table(appendix_summary),
          '', '原RAW条件模型选择RNA正则，token条件probe固定λ=1。新增30个凸Cox拟合，对三输入和两终点全部使用既有probe的λ=1匹配控制，基线λ仍来自原inner-only选择。APPENDIX_PROTOCOL_LOCK明确记为posthoc，36项区间作家族校正但仍不等于新的独立确认；不改变3330条主矩阵及12项主要结果。',
          '', 'Hallmark固定RAW在DSS条件下为0.80028/0.77431，较CS为−0.00565/+0.00059；普通与附录校正区间均含零。这个控制没有揭示被正则选择隐藏的稳定DSS正增量。完整token条件读出与固定RAW的差值同样不能通过附录校正，见APPENDIX_PAIRED全部负结果。',
          '', 'Top/MATCH的CS+RAW模型在五个outer fit都选择了OFF，因此预测与CS精确相同、增量区间为[0,0]。这只描述预锁选择程序的输出，不能证明真实RNA条件效应为零。凸Ridge各seed记录复制同一个唯一拟合，不是三个独立初始化或额外患者证据。',
          '', 'Harrell-based正则选择与Efron目标存在差别：Hallmark RNA-only tuned的heldout Efron loss明显高于固定λ=1，条件模型也可能产生更极端风险而没有排序增量。该loss是同协议的heldout partial-risk-set诊断，并非独立绝对风险校准验证。早于min10的最佳验证时点与正则选择不稳定，都属于当前训练/选择配置限制；未事后优化并宣称原预注册。',
          '', '本轮结果可以区分哪些配置保留排序、哪些加性增量可检出；三种解释可能并存，尚不能由单一相关性、CI含零或PFI对比排他地确定原因。执行/输入与数学核验置信度高；医学机制及外部泛化置信度有限。独立程序复算不等于独立外部审稿人验收。',
          '', '## 优势与风险','',
          '优势：三组固定输入、强条件基线、低容量联合Cox、完整token与mean、匹配RNA初始化、固定probe及两种终点共享患者折。风险：65个DSS事件、同队列多轮探索、监督producer训练内适配可能过拟合、producer目标/head不同、固定λprobe不能覆盖所有可解码函数。RNA3024缺失区块仅做代理，不伪称已验证平台身份；子组少事件诊断见SUBGROUP_METRICS。',
          '', '## 拓展思考 / 你可能想知道','',
          '若直接RNA加性模型已有稳定条件增量而旧token读出不足，优先保留低容量RNA风险/表示再比较新的交互；若连加性条件增量也不稳定，应先报告区间与重叠诊断，再决定外部验证或更丰富终点。不得仅因使用医学通路就认定知识语义创新成立。',
          '', '## 审计和来源','',f'指标3330记录与sksurv交叉验证；checkpoint3330回放；{replay["independent_preprocessing_matrices"]}独立特征矩阵、{replay["independent_PCA_residual_transforms"]}PCA/残差transform、{literal["scalar_checks"]}显式重复患者bootstrap标量检查通过。六主要比较×两指标作12项校正；所有PFI与其它诊断为次要未校正，不能事后升级。原Phase12产物保持只读。']
    write('RESULTS.md','\n\n'.join(rows))
    # Anonymous subgroup within-fold metrics, no pooled cross-model coordinates.
    d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));subrows=[]
    from validate_phase11_independent import components,ipcw
    for ep in cfg['endpoints']:
        z=np.load(P/'endpoints.npz');valid=z[ep+'_valid'];t=z[ep+'_time'];e=z[ep+'_event'];weight=np.zeros(512)
        for f in cfg['folds']:tr=valid&(d['folds']!=f);te=valid&(d['folds']==f);weight[te]=ipcw(t[tr],e[tr],t[te],e[te])
        subsets={'COAD':d['cancer_type']=='COAD','READ':d['cancer_type']=='READ','RNA_complete':np.isnan(d['raw_rna']).sum(1)==0,'RNA_missing3024':np.isnan(d['raw_rna']).sum(1)==3024}
        names=['CS']+[g+'_CS_RAW' for g in cfg['genes']]+[g+'_R_tuned' for g in cfg['genes']]
        for model in names:
            for s in cfg['seeds']:
                for name,subset in subsets.items():
                    sums=np.zeros(4)
                    for f in cfg['folds']:
                        zz=np.load(F/'oof'/f'{ep}_{model}_seed{s}_fold{f}.npz');keep=subset[zz['test_indices']];ii=zz['test_indices'][keep];rr=zz['risk'][keep];hn,hd=components(t[ii],e[ii],rr);un,ud=components(t[ii],e[ii],rr,weight[ii]);sums+=np.array([hn,hd,un,ud])
                    selected=subset&valid;subrows.append({'endpoint':ep,'model':model,'seed':s,'subgroup':name,'n':int(selected.sum()),'events':int(e[selected].sum()),'Harrell':sums[0]/sums[1] if sums[1] else None,'Uno60':sums[2]/sums[3] if sums[3] else None,'scope':'DESCRIPTIVE_ONLY,proxy identity unverified'})
    pd.DataFrame(subrows).to_csv(R/'SUBGROUP_METRICS.csv',index=False)
    write('INDEPENDENT_VALIDATION.json',json.dumps({'status':'PASS','metrics':metrics,'replay':replay,'literal_bootstrap':literal,'external_auditor_completed':False},indent=2))
    write('RUN_MANIFEST.json',json.dumps({'status':'COMPLETE_AUDITED','records':3330,'posthoc_appendix_unique_newfits':30,'source_hash':json.loads((R/'HARD_GATES.json').read_text())['source_hash'],'parent_commit':cfg['parent_commit'],'private_directory':str(P),'branch':'codex/phase13-rna-information-retention-audit'},indent=2))
    write('REPRODUCE.md','# 运行与复算\n\nPython3.12.3/torch2.5.1+cu124/numpy2.1.3/scipy1.14.1/pandas2.2.3/sklearn1.5.2/sksurv0.23.1。环境PYTHONPATH=/root/phase10_trimodal_eval_deps，CUBLAS_WORKSPACE_CONFIG=:4096:8，OPENBLAS/OMP/MKL_NUM_THREADS=2。父提交96a25fd93da8ff39c7b081f078edd3880b0803af及Phase12私有目录只读依赖；原CDR来自SOURCE_REGISTRY公开链接并仅服务器保存。\n\n正式文件与源hash绑定；复跑使用隔离输出目录，不覆盖原记录。原已有COMPLETE不重写。\n\n```bash\npython -m unittest tests.test_phase13_rna -v\npython scripts/prepare_phase13.py\npython scripts/gate_phase13.py\npython scripts/schedule_phase13.py\npython scripts/validate_phase13_independent.py\npython scripts/replay_phase13.py\npython scripts/audit_phase13_statistics.py\npython scripts/report_phase13.py\npython scripts/package_phase13.py\n```\n\n表征与head按producer/fold各自训练，不能拼接独立latent坐标做训练。ARTIFACT_HASH_INDEX提供匿名逐cell索引；私有formal/cells包含checkpoint/OOF/manifests路径与全部hash。模型forward和统计实现均实交付；授权外部审计员可在服务器重新复算。\n')
    journal=P/'BUG_JOURNAL.json';bugs=json.loads(journal.read_text()) if journal.exists() else [];pd.DataFrame(bugs,columns=['severity','stage','bug','affected','repair','post_hoc','source_hash']).to_csv(R/'BUG_LOG.csv',index=False)
    write('README.md','# Phase13 RNA诊断\n\n先读RESULTS，主要对比见PRIMARY_PAIRED，完整负结果见ALL_PAIRED。源码与匿名结果公开；个体输入、OOF及checkpoint仅服务器。')
    telemetry=json.loads((P/'GPU_TELEMETRY.json').read_text());samples=[v for v in telemetry if v.get('active') and ',' in v.get('gpu','')]
    write('GPU_UTILIZATION.json',json.dumps({'samples':len(samples),'busy_mean_percent':float(np.mean([float(v['gpu'].split(',')[0]) for v in samples])),'peak_memory_MB':max(float(v['gpu'].split(',')[1]) for v in samples),'scope':'includesCPU convex readout stages;RNA-only lacksheavy patchcomputation,no artificialmemory filling'},indent=2))
    print(json.dumps(decision))


if __name__=='__main__':main()

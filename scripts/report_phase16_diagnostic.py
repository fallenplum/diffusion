"""Deliver only anonymous aggregates; preserve the published Phase16 batches."""
import sys,json,re,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from src.phase16_diagnostic.core import *
from run_phase16_visual_diagnostic import vsource

def main():
    for name in ['EVALUATION_AUDIT','QUERY_AUDIT','SPATIAL_AUDIT','SOURCES_INDEPENDENT_AUDIT','MORPHOLOGY_SOURCE_AUDIT','SUPPLEMENTARY_AUDIT']:
        if json.loads((R/(name+'.json')).read_text())['status']!='PASS':raise ValueError('missing delivery audit '+name)
    if source()!=json.loads((P/'LOCK.json').read_text())['training_source'] or vsource()!=json.loads((P/'VISUAL_LOCK.json').read_text())['source']:raise ValueError('locked training source changed')
    query_sha=sha(ROOT/'scripts/audit_phase16_query_sensitivity.py')
    if query_sha!=json.loads((P/'QUERY_SOURCE_LOCK.json').read_text())['script_sha256'] or query_sha!=json.loads((R/'QUERY_AUDIT.json').read_text())['query_source_sha256']:raise ValueError('executed query source mismatch')
    # Both previously published report batches must stay byte-identical.
    old_files=[]
    for root in [ROOT/'reports/phase16_semantic_routing',ROOT/'reports/phase16_semantic_routing/semantic_completion']:
        manifest=json.loads((root/'ARTIFACT_MANIFEST.json').read_text())
        for row in manifest['public_files']:
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('previous public artifact modified '+row['path'])
            old_files.append(row['path'])
    summary=pd.read_csv(R/'SUMMARY.csv');paired=pd.read_csv(R/'PAIRED_COMPARISONS.csv');supp=pd.read_csv(R/'SUPPLEMENTARY_PAIRED_COMPARISONS.csv');sp=pd.read_csv(R/'HELDOUT_SPATIAL_SEED.csv');q=pd.read_csv(R/'QUERY_SENSITIVITY_FOLD.csv');cf=pd.read_csv(R/'COUNTERFACTUAL_FOLD.csv');noise=pd.read_csv(R/'NUMERICAL_CONTROL.csv');vis=json.loads((R/'PAIRED_VISUAL_AUDIT.json').read_text());muc=pd.read_csv(R/'MUCINOUS_PAIRED_DIAGNOSTIC.csv')
    optimization=pd.read_csv(R/'OPTIMIZATION_AUDIT.csv');actual=int((optimization['lambda'].astype(str)!='OFF').sum());constant=len(optimization)-actual
    def value(domain,model,schedule='FIXED1',endpoint='DSS'):
        row=summary[(summary.domain==domain)&(summary.model==model)&(summary.schedule==schedule)&(summary.endpoint==endpoint)].iloc[0];return f'{row.Harrell:.6f} / {row.Uno60:.6f}'
    def interval(row):return f'[{row.corrected_low:+.6f}, {row.corrected_high:+.6f}]'
    decision={'status':'DIAGNOSTICS_COMPLETED_NO_NEW_FUSION_MODEL','RNA_finding':'large Harrell loss already after same512-to8 pooling and PCA8;quantization adds no confirmed further loss','RNA_claim_boundary':'Uno60 compression contrasts include zero;not a proof that every8D supervised representation fails','query_finding':'RNA presence changes ROI more than swapping patient RNA states;regional medical meaning unverified','visual_finding':'dual64 positive visual-only point differences but intervals include zero;dual32 matched capacity not superior','next_gate':'preserve RNA512 reference and verify conditional utility before further prompt/fusion growth','raw_WSI_QA':'DEFERRED_BY_USER','unique_outer_model_units':180,'actual_outer_optimizer_fits':actual,'constant_OFF_references':constant,'seed_fold_evaluation_slots':540,'metric_records':1080,'deterministic_references_not_independent_seeds':True,'no_patient_data_public':True};write(R/'decision.json',decision)
    text=['# Phase16 后续诊断：查询敏感性、RNA信息保留与双视觉互补','',
          '**核心发现：当前RNA问题不能主要归因于状态量化。显著的Harrell损失在512维到8维的无监督压缩时已经出现；患者特异性RNA查询只小幅改变选区；双视觉尚无稳定配对优势。**','',
          '本轮按用户补充增加RNA-free CONCH/UNI2/坐标匹配双视觉简单基线；原图按用户安排延后。没有增加融合Transformer，未修改前两批Phase16文件。全部结论限于反复探索的同一队列。','',
          '## 1. 协议与执行范围','',
          '- 同一512人/65 DSS事件、原5折患者划分。PFI510人/129事件仅迁移已训练DSS风险，不使用PFI选择模型。',
          '- RNA9种表示×固定λ1/统一inner选λ两种日程，共90个外层模型单元；其中13个选择OFF常数风险，77个实际优化。视觉8种输入加CS基线、两种日程，共90个实际优化。合计180个模型单元、167个实际外层优化和13个OFF常数引用。',
          '- 同一折划分、确定性的凸Cox和full-SVD PCA不受seed影响；31/47/83输出270+270个评测槽，其中360个是确定性引用，绝不能当成三次独立验证。共1080条DSS/PFI逐折记录。',
          '- λ1是主要同协议表示比较。敏感性统一grid0.01/0.1/1/10/OFF，inner-train训练、inner-val选λ，随后fresh outer-train fit。无神经epoch或教师，无外折标签参与预处理/选择。',
          '- 全部基因填补、可测性、基因选择、RNA均值/SD、量化阈值、PCA及下游标准化都由本角色训练患者决定。所有外折指标独立对照scikit-survival；最大独立优化梯度约9.25e-8。','',
          '## 2. RNA：信息在哪一步没有保留','', '|RNA-only表示|固定λ1：Harrell / Uno60|inner选λ：Harrell / Uno60|','|---|---:|---:|']
    for name in cfg()['representations']:text.append(f'|{name}|{value("RNA",name)}|{value("RNA",name,"SELECTED")}|')
    text+=['','- TOP512：原始方差Top512。HALL512：历史同源Hallmark富集512基因，并非本轮按DSS重选。',
           '- HALL_POOL8：只用同一HALL512基因，按真实8组成员平均；HALL_PCA8：同一512维训练折PCA8。两者均无生存监督，分别测试生物均值聚合和一般低维压缩。',
           '- PATH8_CONT：query实际采用的完整可测基因8程序连续均值。PATH8_QUANT_ORD：对应low/typical/high=-1/0/+1，unmeasured=0；另用4状态one-hot32维检查量化编码形式。',
           '- COV变体保留各程序缺失比例。当前可测性筛选后这些比例无新增区分，COV与无COV结果相同；不是证明原始3024列缺失块没有影响。','',
           '### 预定义主要配对：6项×2指标，Bonferroni12','', '|比较|指标|Δ|校正CI|','|---|---|---:|---|']
    for row in paired[(paired.domain=='RNA')&(paired.schedule=='FIXED1')&(paired.endpoint=='DSS')].itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|{interval(row)}|')
    text+=['','**解释：** HALL512→HALL_POOL8的Harrell下降约0.194，HALL512→PCA8也下降约0.193，校正区间都排除零。这支持当前两种无监督8维压缩没有保留可线性解码的DSS排序。连续8维→量化8维的点差很小、区间跨零，因此不能声称主要问题发生在量化这一步。','',
           '**边界：** Uno60的上述压缩配对区间仍含零；65个事件下不确定性明显。PCA8结果不证明所有8维监督表示都失败；平均程序可能丢失组内基因差异及正负权重，但本实验没有逐基因验证这一解释。HALL512胜过TOP512仅是点估计，配对区间包含零。RNA-only有效不自动代表Clinical+S32条件增量。','',
           'PAIR_TRANSITIONS.csv记录同折可比较患者对的改善/丢失质量，RISK_CORRELATIONS.csv只报告同折风险秩关联；没有用外折错排患者对生成新监督任务。','',
           '## 3. 查询是否真正改变选区','',
           '旧统计约95.6%是共享比例，不是Jaccard，且旧表包含每个角色的全部患者。以下只计外折测试患者、每seed512人，采用患者均值；多切片按各切片选区数量加权。该口径下真实/错RNA512共享约95.19%，Jaccard约91.03%。','',
           '|比较|预算|共享比例|Jaccard|8×8分布TV|归一化质心位移|','|---|---:|---:|---:|---:|---:|']
    for (comparison,budget),g in sp.groupby(['comparison','budget']):text.append(f'|{comparison}|{budget}|{g.intersection_fraction_mean.mean():.4f}|{g.Jaccard_mean.mean():.4f}|{g.grid_total_variation_mean.mean():.4f}|{g.centroid_shift_mean.mean():.4f}|')
    text+=['','|比较|查询余弦|Patch全排序Spearman|','|---|---:|---:|']
    for comparison,g in q[q.budget==512].groupby('comparison'):text.append(f'|{comparison}|{np.average(g.query_cosine_mean,weights=g.units):.6f}|{np.average(g.patch_rank_spearman_mean,weights=g.units):.6f}|')
    text+=['','**为什么高重合：** 相比删除整句RNA，替换患者的八程序状态对文本向量的改变量小得多。真实/错RNA查询余弦约0.9937、候选Patch排序相关约0.9961；而加入整句RNA时余弦约0.9235，并使512选区替换约15.8%。这支持“共同的RNA语境变化大于当前患者状态差异”的描述。并不能据此断言文本塔完全忽略RNA。','',
           '512预算比128更容易纳入相同高分候选：正确/错RNA128共享约89.4%，512约95.2%。这里还包含32概念轮询并集的稳定性及候选数量限制，不能把全部重合归因于一种机制。QUERY_DELTA_DECOMPOSITION.csv的公共/概念特异向量能量分解是事后描述，不是语义真实性的量度。','',
           '### 独立程序状态与数值负对照','',
           '每折固定抽6名外折患者，共30人；逐个程序强制low/high、保持另外7程序和Clinical相同。该压力测试可能构造不常见的状态组合，不是实际患者干预。','', '|程序编号|128共享|512共享|查询余弦|','|---|---:|---:|---:|']
    for group,g in cf.groupby('group'):
        a=g[g.budget==128];b=g[g.budget==512];text.append(f'|{group}|{a.intersection_fraction.mean():.4f}|{b.intersection_fraction.mean():.4f}|{b.query_cosine_mean.mean():.6f}|')
    text+=['','程序编号顺序：cell cycle、EMT、IFNγ、hypoxia、TGFβ、DNA repair、WNTβ、angiogenesis。512单程序low/high通常仍共享约97.4%–98.8%区域。相同文本独立重编码共享约99.85%，说明程序变化高于半精度批次舍入，但幅度依然小。新的投影缓存与旧缓存逐字节一致，抽样的旧ROI重放共享为1.0。','',
           '## 4. 有无真实形态学信息','',
           '补充使用独立supplementary HISTOLOGICAL_DIAGNOSIS：505名有效患者，66名黏液性、7名未知。用固定C10文本对选区Patch打分并平均，不训练分类头，也不使用DSS。CLIN_RNA512的患者黏液性AUC约0.7552，错RNA约0.7518；直接差约+0.00338，95%CI包含零。','',
           '这是患者级病理诊断的相关性检查：能够说明冻结CONCH语义具有部分亚型信息，但不能证明被换入/换出的区域真的更符合黏液形态。患者标签不是patch标签，文本分数也不是医学真值。用户安排先寻找机制线索，再补少量原图，因此原图QA仍明确待完成。30例几何检查/审阅队列已留在私有目录。','',
           '## 5. RNA-free双视觉：先验证表示互补','',
           f'519张切片覆盖512名患者，共选{vis["selected_patches"]:,}个正面积匹配CONCH足迹；各组使用完全相同的区域。候选池仍是519/520张CONCH切片。原GRID1024排序经匹配过滤，配额不足时按坐标顺序补齐，共{vis["fallback_rows"]:,}行，因此不是不受覆盖影响的纯GRID。',
           '- CONCH原始768维逐patch L2归一化；UNI2-h1536维逐patch归一化，按矩形交集面积聚合至同一CONCH足迹，再归一化。每张切片先均值，患者多切片等权。',
           '- CONCH32/UNI32各用训练折PCA32；DUAL32为各取16维、参数维度匹配；DUAL64保留各32维，有更高容量。PCA前后标准化都训练折拟合。',
           '- 各模型仅Ridge Cox，不使用RNA、文本检索或attention。另加同协议Clinical+S32条件版本，CS是本轮日程的基线，不能直接等同Phase15不同λ日程的历史风险。','',
           '|输入|固定λ1：Harrell / Uno60|inner选λ：Harrell / Uno60|','|---|---:|---:|']
    for context in ['VISUAL_ONLY','CS_PLUS_VISUAL']:
        for rep in ['CONCH32','UNI32','DUAL32','DUAL64']:
            name=context+'__'+rep;text.append(f'|{name}|{value("VISUAL",name)}|{value("VISUAL",name,"SELECTED")}|')
    text.append(f'|本轮CS|{value("VISUAL","CS")}|{value("VISUAL","CS","SELECTED")}|')
    text+=['','视觉核心比较每个日程单独Bonferroni12。为直接回答DUAL64相对各单视觉，以及条件版本相对CS，读取点估计后追加既有OOF直接配对，单列supplementary family16；没有新训练，没有追认为主检验。','',
           '|直接视觉比较|指标|Δ|补充校正CI|','|---|---|---:|---|']
    select=supp[(supp.endpoint=='DSS')&(supp.schedule=='FIXED1')&supp.comparison.str.startswith('VISUAL_ONLY__DUAL64')]
    for row in select.itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|{interval(row)}|')
    text+=['','**结论：** DUAL64视觉单独预测有小幅正向点差，但尚未排除零；维度匹配DUAL32没有优于两种单视觉。Clinical+S32条件下，DUAL32的正向点差也未形成稳定配对优势，且inner选择敏感性没有保持相同排序。因此当前不足以启动更复杂的语义双视觉融合。','',
           '30张切片、240个足迹通过独立坐标矩形重算及稀疏聚合核验；这是计算正确性，不是组织学正确性。平均池化、低维PCA可能丢失局部信息；负结果不排除其他低容量表示或局部互补机制。','',
           '## 6. 决策、优势与风险','',
           '**优先决策：** 保留HALL512/Top512作为RNA信息保留基线。若后续做Prompt，应将八程序作为辅助上下文，并检验是否真能改变有意义区域；不能只让已明显丢失DSS排序的八程序承担全部分子预后信息。必须另行验证Clinical+S32条件增量，不能仅靠RNA-only点估计扩大模型。','',
           '**优势：** 相同任务/患者折/模型类、凸目标收敛核验、同基因池化及维度匹配控制、只计heldout检索、单程序与数值控制、独立患者形态字段、几何源文件核验、保留旧报告。','',
           '**风险：** 同队列反复探索；65事件、IPCW尾部方差；固定模型bootstrap未包含全流程重训；确定性seed引用不能增加样本量；HALL512/8维输入维度不同；视觉PCA及均值池化有限；匹配池受覆盖/坐标补齐影响；反事实状态可能不自然；没有原图与区域标注。','',
           '**置信度：** 已执行的数值和源文件审计为高；当前无监督压缩降低Harrell可解码性为中；RNA驱动区域的医学意义及双视觉稳定增量仍未确认。','',
           '## 拓展思考','', '需要区分“向量变化”“选区变化”“区域形态变化”“条件生存排序变化”四个层次。当前仅前两层可定量验证，患者级黏液性检查不能代替后两层。','',
           '## 你可能想知道','', '所有患者OOF、状态、ROI坐标、空间匹配及权重都留在私有位置；CONCH-Z仍保留于E盘和服务器。原图按用户安排延后获取，没有额外下载WSI。']
    (R/'RESULTS.md').write_text('\n'.join(text),encoding='utf8')
    (R/'REPRODUCE.md').write_text('''# 复现\n\n从74b7885分支继续，使用相同私有数据、Phase12预处理、Phase16候选H5/坐标映射、已授权CONCH-Z权重。仓库不含患者数据及受限权重。\n\n```bash\nexport PYTHONPATH=/root/phase10_trimodal_eval_deps\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false\n/root/miniconda3/bin/python scripts/pipeline_phase16_diagnostic.py\n```\n\nCPU RNA与双视觉拟合可并行，GPU只做冻结文本/图像投影推理；已有source相同的Cox cell恢复复用。查询审计会精确重建重启后丢失的/dev/shm视觉缓存。独立审计后才生成报告。原图不在前置条件中，明确不声称已完成原图QA。逐折OOF及患者表位于/root/phase16_prompt_rna_diagnostic_private，不发布。\n\n主比较和日程在两个config中锁定；直接DUAL64与条件CS的补充配对是读取主分析点估计后定义，见SUPPLEMENTARY_CONTRAST_SCOPE.json，属于探索性分析。\n''',encoding='utf8')
    write(R/'ENVIRONMENT.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'GPU_reported':torch.cuda.get_device_name(0),'CPU_fold_workers':5,'training_objective':'convex full-risk-set Efron','new_fusion_Transformer':False})
    files=[CONFIG,ROOT/'configs/phase16_visual_diagnostic.json']+sorted((ROOT/'src/phase16_diagnostic').glob('*.py'))+[ROOT/'scripts'/n for n in ['run_phase16_rna_diagnostic.py','run_phase16_visual_diagnostic.py','audit_phase16_query_sensitivity.py','evaluate_phase16_diagnostic.py','audit_phase16_diagnostic_sources.py','compare_phase16_diagnostic_supplement.py','report_phase16_diagnostic.py','pipeline_phase16_diagnostic.py','publish_phase16_diagnostic.py']]+[p for p in sorted(R.rglob('*')) if p.suffix in ['.json','.md','.csv'] and p.name not in ['ARTIFACT_MANIFEST.json','DELIVERY.json']]
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----');rows=[]
    for file in files:
        content=file.read_bytes()
        if pattern.search(content):raise ValueError('protected identifier in '+str(file))
        if file.suffix=='.csv' and set(pd.read_csv(file,nrows=0).columns)&{'patient_index','patient_id','risk','time','event','coords'}:raise ValueError('private row table in report')
        rows.append({'path':file.relative_to(ROOT).as_posix(),'bytes':len(content),'sha256':sha(file)})
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','public_files':rows,'previous_files_verified':len(set(old_files)),'source':source(),'visual_source':vsource(),'query_source':query_sha,'no_patient_level_upload':True})
    write(R/'DELIVERY.json',{'status':'PASS','unique_outer_model_units':180,'actual_outer_optimizer_fits':actual,'constant_OFF_references':constant,'deterministic_seed_fold_evaluation_slots':540,'metric_records':1080,'manifest_sha256':sha(R/'ARTIFACT_MANIFEST.json'),'public_files':len(rows),'raw_image_QA':'DEFERRED_BY_USER'});print('DIAGNOSTIC_DELIVERY_PASS',len(rows),flush=True)

if __name__=='__main__':main()

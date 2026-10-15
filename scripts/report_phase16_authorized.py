"""Audited Phase16 completion report;old blocked-batch outputs remain immutable."""
import sys,json,re,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from src.phase16_semantic.omics import *
from run_phase16_mil_two_stage import LP,lsource

def main():
    for p in [R/'FINAL_INDEPENDENT_AUDIT.json',R/'QUERY_REPLAY_AUDIT.json',R/'DIRECT_COMPARISON_AUDIT.json',R/'MECHANISM_AUDIT.json',R/'G5/INDEPENDENT_AUDIT.json',R/'MIL/INDEPENDENT_AUDIT.json']:
        if json.loads(p.read_text())['status']!='PASS':raise ValueError('delivery gate absent: '+str(p))
    means=pd.read_csv(R/'AGGREGATE_METRICS.csv');primary=pd.read_csv(R/'PAIRED_COMPARISONS.csv');gp=pd.read_csv(R/'G5/PAIRED_COMPARISONS.csv');direct=pd.read_csv(R/'DIRECT_PAIRED_COMPARISONS.csv');mil=pd.read_csv(R/'MIL/PAIRED_COMPARISONS.csv');models=pd.read_csv(R/'UNIFIED_MODEL_SUMMARY.csv');g1=pd.read_csv(R/'G1_COVERAGE.csv');fallback=pd.read_csv(R/'WRONG_PROMPT_FALLBACKS.csv');ablation=pd.read_csv(R/'PROMPT_INFERENCE_ABLATION_SEED.csv');comp=pd.read_csv(R/'RISK_COMPONENTS.csv');selection=pd.read_csv(R/'OMICS_ONLY_SELECTION.csv');aud=json.loads((R/'FINAL_INDEPENDENT_AUDIT.json').read_text());prep=json.loads((P/'PREPARE_BATCH_DONE.json').read_text());query=json.loads((R/'QUERY_REPLAY_AUDIT.json').read_text());ma=json.loads((R/'MECHANISM_AUDIT.json').read_text())
    mainp=primary[(primary.endpoint=='DSS')&(primary.family=='primary')];confirmed=[]
    for comparison,g in mainp.groupby('comparison'):
        if len(g)==2 and (g.corrected_low>0).all() and (g.positive_seeds==3).all():confirmed.append(comparison)
    status='EXPLORATORY_SIGNAL_REQUIRES_CONFIRMATION' if confirmed else 'NO_CONFIRMED_INCREMENT';decision={'status':status,'semantic_space':'PASS_OFFICIAL_PROJECTION_AND_TEXT','primary_comparisons_with_both_metrics_confirmed':confirmed,'primary_tests':16,'all_50arms_completed':True,'outer_fold_evaluations':750,'new_outer_fits_this_authorization':540,'cumulative_new_outer_fits':690,'historical_refs':60,'G4':'BLOCKED_RAW_WSI_QA','candidate_pool_limited':True,'slides':'519/520','patients':512,'DSS_events':65,'PFI_valid_patients':510,'PFI_valid_events':129,'execution_confidence':'high','patient_specific_RNA_semantic_mechanism':'not confirmed','training_budget_sufficiency':'not established;original80epoch schedule retained','RNA_information_scope':'only8fixedHallmark programs;not originalTop512/Hallmark-enriched512','source':source(),'omics_source':msource(),'MIL_source':lsource()};write(R/'decision.json',decision)
    def point(arm,ep='DSS'):
        row=means[(means.endpoint==ep)&(means.arm==arm)].iloc[0];return f'{row.Harrell_mean:.6f} / {row.Uno60_mean:.6f}'
    text=['# Phase16 — 授权后的语义路由完整审计','',f'**研究决策：{status}。官方Patch↔Text空间已验证；患者特异性RNA路由的稳定增量尚未确认。**','',
          '本报告接续a19eb457冻结的受阻批次，保留旧报告与数值不变。模型授权现已生效，不再是语义分支的阻塞项。G4双视觉分支仍受原始WSI/30组原图QA阻塞。','',
          '## 1. 实际完成范围','', '|模块|arms|外层单元|本次新增训练|','|---|---:|---:|---:|',
          '|G2：8 selectors×128/512及6个基准|22|330|120|','|G3：5类错Prompt×128/512|10|150|150|','|G5：12核心及2随机mask、2omics-only|16|240|240|','|MIL两阶段补充消融|2|30|30|','|合计|50|750|540|','',
          '累计690个新outer-fold拟合，60个历史引用；本次新增540个，旧批150个训练不重跑。PFI只迁移同一DSS模型，不另算训练。1500条DSS/PFI逐折记录均交叉核验scikit-survival。Inner拟合、超参trial和两阶段selector训练另外记录，不谎报为新的独立外层cell。G4腾出的预算用于必要的组学/mask及MIL对照；未执行原图不支持的双视觉。','',
          '## 2. 数据与官方接口','',
          '- 512名患者、65个DSS事件；PFI有效510名/129事件；沿用同一患者折和seed31/47/83。',
          '- **candidate-pool-limited**：519/520张原始切片有CONCH，512/512患者覆盖；使用1,278,642个原始可用patch，不使用128缓存冒充全库。缺失切片不替换/重命名。',
          '- 官方CONCH-Z固定revision60211fced9c25a014494e248eedb6a6be80109f7，完整权重约1.67GB；E盘与服务器各保留一份，不公开权重或派生患者级向量。',
          '- 504条权重精确加载，无missing/unexpected keys。原始CONCH视觉向量经官方768×768 proj_contrast，再L2归一化，才能和CONCH-Z文本余弦；没有用TITAN text直接匹配raw CONCH。',
          f'- 1024个投影样本独立重放误差{query["projection_error"]:.3g}；720组预算/ROI排序抽样检查完全一致；重新编码文本的最大半精度批次舍入差{query["fresh_text_max_roundoff"]:.3g}。',
          '- 官方编码器128-token上下文保留一个CLS位置；所有实际新文本编码都检查≤127，不允许静默截断。本批最长新文本为'+str(prep['maximum_new_text_tokens'])+' tokens。',
          '- 仍无原图：519张切片仅做计算矩形重叠，平均覆盖约74.2%、约20.1%未匹配；原图QA完成0组，G4保持BLOCKED。不得将这些数字称为组织学对齐正确率。','',
          '## 3. 实际Prompt与检索','',
          '32个CRC形态概念沿用外部知识词表，不按DSS筛选。实施采用新配置中的简短固定模板，在读取本次语义结果前锁定；与先前受阻批次的预备长模板不同。队列历经多阶段探索，所有结果均属内部探索，不是全新未接触队列的外部确认。','',
          '```text',cfg()['concept_template'],cfg()['clinical_template'],cfg()['RNA_template'],'```','',
          'RNA分子状态为8个固定Hallmark程序：cell cycle、EMT、IFNγ、hypoxia、TGFβ、DNA repair、WNTβ-catenin、angiogenesis。使用本角色训练患者的基因缺失填补、标准化、可测性筛选与33/67分位阈值；量化为low/typical/high，覆盖不足标unmeasured，不把缺失当零表达，也不描述为突变。Clinical只使用原先6个合法字段。','',
          'Clinical/RNA上下文与每个形态句子**在编码前拼接**，经过完全冻结的官方文本塔，形成患者×概念query。不是编码后给静态向量加RNA偏移。每概念按Patch余弦降序，32概念round-robin取并集、去重；使用与旧对照相同的每患者配额和多切片合并规则。没有额外用DSS选择词表、ROI、表达阈值或query权重。','',
          '同癌种/Stage或缺失模式优先进行分区内derangement；单例时按癌种、再按本分区退化，所有次数见WRONG_PROMPT_FALLBACKS.csv。没有跨inner-train/inner-val/outer-train/outer-test借用患者。并非所有错配都能保持最严格分层，不能隐去退化。','',
          '## 4. DSS主要数值','', '|模型|Harrell / Uno60（3seed OOF均值）|','|---|---:|']
    showcase=['CS_P15','C4_LEGACY','RAND_512','GRID_512','MIL_512','MIL_TWO_STAGE_512','FIXED_TEXT_512','CLIN_TEXT_512','RNA_TEXT_512','CLIN_RNA_TEXT_512','JOINT_WRONG_RNA_512','SHUFFLED_FIELDS_128','G5_RNA_ONLY','G5_CNA_ONLY','G5_FIXED_TEXT_512_RNA','G5_CLIN_RNA_TEXT_512_RNA','G5_CLIN_RNA_TEXT_512_RC_UNGATED','G5_CLIN_RNA_TEXT_512_RC_GATED']
    for arm in showcase:text.append(f'|{arm}|{point(arm)}|')
    text+=['','每seed先按同折可比较对分子/分母拼5-fold OOF，再均值3个seed；没有平均15个cell的C，也没有风险跨fold排序或把seeds合成新患者。历史C4还含RNA交互且预算为每张WSI128，是历史强参考，不能当作面积/输入完全匹配的ROI消融。固定Ridge omics-only在本协议下跨seed完全相同，seed_sd=0不代表3次独立验证。','',
          '## 5. 预定义主要比较：Bonferroni16','', '|比较|指标|Δ|95% CI|校正CI|正向seed|','|---|---|---:|---|---|---:|']
    for row in mainp.itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.CI_low:+.6f},{row.CI_high:+.6f}]|[{row.corrected_low:+.6f},{row.corrected_high:+.6f}]|{row.positive_seeds}/3|')
    text+=['','CLIN_RNA_TEXT512相对强CS点估计约+0.00439/+0.00894，但两项校正区间均包含零；相对GRID512，Uno60点估计也没有提高。不能据较高Harrell单项宣称胜过历史C4，因为其Uno60仍低于C4。','',
          '## 6. RNA患者身份与语义负对照','', '|比较|指标|Δ|校正CI（次要family32）|','|---|---|---:|---|']
    negative=primary[(primary.endpoint=='DSS')&primary.comparison.str.contains('JOINT_WRONG_RNA|SHUFFLED_FIELDS')]
    for row in negative.itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.corrected_low:+.6f},{row.corrected_high:+.6f}]|')
    text+=['','真实RNA512对错RNA512的点差只有约+0.000627/+0.000981，校正bootstrap下界为0，未排除零增量。3seed都为正仍不足以通过本轮门槛。错误字段128对照的点估计更高，也削弱“正确患者语义必然推动DSS增益”的解释；这不赋予错误文本任何医学含义。','',
          'G1_COVERAGE.csv给出ROI overlap、8×8空间占用/集中度与重复行；它们是无标签检索诊断，没有ROI真值，不能称为检索准确率。wrong-prompt改变文本或ROI，也不等于改变了有效预后证据。','',
          '## 7. Query贡献与只在预测头加入组学','',
          'G5的RNA预测头保留与query同源的连续8程序分数及缺失比例；加入CNA时RNA部分不变。CNA采用同gene ID对齐的gain/loss分开聚合；same-gene gate为预先固定sigmoid(RNA基因zscore)×CNA gain/loss，不是学习得到的调控关系。True/random mask保持基因数、组间重叠和RNA/CNA同gene身份，按训练RNA均值/SD及可测类别匹配；未匹配染色体空间结构。','',
          'G5使用相同ROI/CS主干，加极小分子线性修正；没有新增fusion Transformer。RNA16维、RNA+CNA40维包含显式缺失比例，因此组学输入不同会改变头部参数数目；同类型true/random的参数预算一致。RNA-only与CNA-only为无Clinical/S32/ROI的低容量Ridge对照。','',
          '|同组学头下CLIN_RNA query−FIXED query|指标|Δ|校正CI（补充family20）|','|---|---|---:|---|']
    for row in direct[(direct.endpoint=='DSS')&direct.comparison.str.startswith('G5_CLIN_RNA')].itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.corrected_low:+.6f},{row.corrected_high:+.6f}]|')
    text+=['','这些直接配对用于区分RNA只进入头部与RNA改变ROI选择；没有被追认为primary。G5核心/随机mask的完整配对另见G5/PAIRED_COMPARISONS.csv（family28）。补充direct family20覆盖6个同头比较、2个两阶段MIL基准及2个omics-only对常数风险的比较。各family分别列明，没有从胜出结果挑选主检验。','',
          '本轮RNA-only八程序表示的Harrell约0.499、Uno60约0.581；CNA-only约0.525/0.515。不能将它们当成此前Top512或富集512的强RNA预后表征。它们只是固定低维输入，本次未证明其稳定的DSS双指标独立排序能力，也不能据此否定完整RNA或CNA信息。','',
          f'低容量Cox由独立逐事件Efron梯度核验，最大绝对梯度{ma["low_capacity_independent_gradient_max"]:.3g}。它排除了这些固定λ拟合的明显优化未收敛，不能排除内层λ选择方差、表示压缩或小事件效应。OMICS_ONLY_SELECTION.csv保留OFF选择，不把OFF当作生物信息为零的证据。','',
          '## 8. MIL两阶段及训练预算','',
          '新增两阶段MIL使用相同部署容量：StageA只在fit患者训练，固定80epoch，不用inner-val挑selector epoch；StageB冻结同一attention向量并重新拟合CS/ROI头，只由inner-val选择StageB epoch。没有第二套部署评分网络。30个外层流程额外包含60个StageA角色拟合，预算明确多于原端到端MIL；不是计算量匹配的纯架构因果对照。','',
          '|两阶段−端到端MIL|指标|Δ|校正CI（family4）|','|---|---|---:|---|']
    for row in mil[mil.endpoint=='DSS'].itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.corrected_low:+.6f},{row.corrected_high:+.6f}]|')
    text+=['','全部新核心、G3、G5神经流程保留原80epoch/Adam0.001/EMA0.3 validation-Cox选择规则，从同seed初始化fresh outer fit，不根据外折分数扩预算。先前16个仅内层320epoch诊断中14个loss继续下降，但Harrell提高6个、下降6个、相同4个。**训练预算充分性仍未确立。**','',
          '|模块|选中80epoch的单元|','|---|---:|']
    for label_group,table in [('语义/G3',pd.read_csv(R/'MODEL_SUMMARY.csv').query('new_in_this_batch == True')),('G5神经头',pd.read_csv(R/'G5/MODEL_SUMMARY.csv').query('selected_epoch > 0')),('两阶段StageB',pd.read_csv(R/'MIL/MODEL_SUMMARY.csv'))]:text.append(f'|{label_group}|{int((table.selected_epoch==80).sum())}/{len(table)}|')
    text+=['','训练诊断保留各epoch损失、inner Harrell、真实梯度和mask/冻结检查。RISK_COMPONENTS.csv只作校正幅度/CS重优化的预测分解，不当作重训的因果消融。','',
          '## 9. Stage/RNA与缺失敏感性','',
          'Prompt消融固定预测头Clinical/S32输入，只改变选区文本：去TNM/AJCC、仅Stage、去RNA。结果见PROMPT_INFERENCE_ABLATION_SEED.csv；它们不新增训练，但存在分布改变，不等于重新训练的公平因果对照。PFI只迁移DSS已训练风险，未用PFI选择参数；不能把PFI当成仅增加事件数的干预。','',
          '原始RNA203人×3024列缺失块按缺失模式单独分层，见MISSING_BLOCK_SENSITIVITY.csv。基因去重后的可用集合由训练折决定；没有证实测序平台身份。亚组数值是描述性敏感性，不引入新的主假设或外折反馈。','',
          '## 10. 五个研究问题与边界','',
          '1. **合法Patch图文空间：已成立。** 官方CONCH-Z投影/文本接口、权重和数值均验证；不是病理标注真值认证。',
          '2. **患者条件路由胜过FIXED/GRID/MIL：尚未确认。** 预定义校正检验没有形成稳定的双指标支持。',
          '3. **真RNA胜过错RNA：仅见极小点差，区间仍包含零。** 不能宣称患者特异性语义机制成立。',
          '4. **UNI2正确匹配后的净增量：未检验。** 缺原图QA，G4 BLOCKED。',
          '5. **同时超过强CS与C4且跨seed稳定：未满足。** 不发布SOTA或医学机制成立结论。','',
          '**优势：** 官方空间、全可用H5候选、统一患者预算、真实S32 forward、完整负对照/条件头比较、同gene RNA/CNA与分离gain/loss、独立逐checkpoint/统计重算。',
          '**风险：** 同队列反复探索、65事件、80epoch上限、固定8程序和量化信息瓶颈、随机mask单次实现/未匹配CNA染色体结构、固定gate缺少患者级调控真值、无原图/ROI金标准、固定模型bootstrap不含全流程重训方差。部分极端校正CI的Monte Carlo尾部样本有限。',
          '**置信度：** 已执行范围的代码/指标审计为高；患者特异性语义和医学因果解释仍缺支持。负结果限于当前表示、选区和训练协议，不能简化成idea NO-GO。','',
          '## 拓展思考','', '优先核查用于Prompt的分子表示是否保留已知RNA预后信息，并区分ROI变化与有效排序变化；当前不支持继续盲目扩展融合深度。','',
          '## 你可能想知道','', 'E盘与服务器的CONCH-Z权重都保留。患者OOF、临床文本、基因测量、ROI坐标及权重不公开；持久化了小型私有文本重放胶囊，完整大缓存位于/dev/shm可重建。','',
          '官方接口依据：[CONCH-Z/LIMIT](https://github.com/mahmoodlab/LIMIT)、[模型页面](https://huggingface.co/MahmoodLab/CONCH-Z)。']
    (R/'RESULTS.md').write_text('\n'.join(text),encoding='utf8');(R/'DECISION.md').write_text('# 决策\n\n'+json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf8')
    (base.R/'README.md').write_text('# Phase16 报告索引\n\n**最新：模型授权后语义路由及组学完整审计，共50 arms/750外层单元。**\n\n- [最新RESULTS](semantic_completion/RESULTS.md)\n- [统一1500条逐折指标](semantic_completion/UNIFIED_FOLD_METRICS.csv)\n- [最新决策](semantic_completion/decision.json)\n\n本目录原RESULTS.md及CSV是a19eb457时的受阻批次历史快照，未覆盖；最新事实请以上述报告为准。\n',encoding='utf8')
    reproduce='''# 授权后复现\n\n父提交a19eb457bc8f7fcbf11ffaf752f00c5aa0d8cf61，分支codex/phase16-semantic-roi-routing。需要已授权个人账号下载的CONCH-Z固定revision、Phase10–15私有数据/原始H5；仓库不含这些数据或权重。\n\n```bash\nexport PYTHONPATH=/root/phase10_trimodal_eval_deps\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n/root/miniconda3/bin/python scripts/pipeline_phase16_authorized.py\n```\n\n该脚本按官方空间、模块pilot、formal、独立审计与汇总顺序执行。模型目录/root/phase16_conch_z/model，E盘原件E:/diffusion_phase16_conch_z/model。原始候选H5只读。已有同源formal cells复用；source/数据/缓存/权重SHA不匹配时拒绝。脚本不尝试绕过许可或自动解锁缺原图的G4。\n\n私有目录：/root/phase16_semantic_private、/root/phase16_omics_private、/root/phase16_mil_private。语义模型与G5不同hash独立保存，旧Phase16文件不可变。重要checkpoint、ROI和小型重放胶囊持久化；/dev/shm中的标准化视觉/投影/完整文本缓存可重建，重放胶囊允许审计在完整文本缓存丢失后继续。\n\n各模块实际命令见pipeline脚本；最终publish_phase16_authorized.py仅发布ARTIFACT_MANIFEST中的匿名列表，并校验远端commit/blob。\n'''
    (R/'REPRODUCE.md').write_text(reproduce,encoding='utf8');write(R/'ENVIRONMENT.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu_reported_name':torch.cuda.get_device_name(0),'workers':8,'additional_MIL_workers':2,'text_batch':512,'sksurv':'0.23.1','official_model_revision':cfg()['official_model_revision'],'runtime_offline_after_download':True})
    # Before packaging, prove the previous public batch was not overwritten.
    old=json.loads((base.R/'ARTIFACT_MANIFEST.json').read_text())
    for row in old['public_files']:
        if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('old Phase16 public artifact modified '+row['path'])
    sources=[ROOT/'configs'/n for n in ['phase16_semantic_continuation.json','phase16_omics_continuation.json','phase16_mil_two_stage.json','phase16_direct_comparisons.json']]+sorted((ROOT/'src/phase16_semantic').glob('*.py'))+[p for p in sorted((ROOT/'scripts').glob('*phase16*.py')) if str(p.relative_to(ROOT)) not in {r['path'] for r in old['public_files']}]
    public=sources+[base.R/'README.md']+[p for p in sorted(R.rglob('*')) if p.suffix in ['.md','.csv','.json'] and p.name not in ['ARTIFACT_MANIFEST.json','DELIVERY.json']]
    regex=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----');files=[]
    for p in public:
        content=p.read_bytes()
        if regex.search(content):raise ValueError('protected identifier in public artifact '+str(p))
        if p.suffix=='.csv' and set(pd.read_csv(p,nrows=0).columns)&{'patient_id','patient_ids','risk','test_indices','time','event'}:raise ValueError('patient-level CSV prohibited')
        files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':len(content)})
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','public_files':files,'previous_unchanged_public_files':len(old['public_files']),'parent_commit':'a19eb457bc8f7fcbf11ffaf752f00c5aa0d8cf61','private_data_public':False,'private_model_weights_sha256':json.loads((R/'MODEL_ACCESS.json').read_text())['weights_sha256'],'private_text_replay_capsule_sha256':query['persistent_private_replay_capsule_sha256'],'private_data_locations':[str(P),str(MP),str(LP)],'training_sources':{'semantic':source(),'omics':msource(),'MIL':lsource()}});delivery={'status':'PASS','arms':50,'outer_fold_evaluations':750,'actual_new_fits_cumulative':690,'historical_refs':60,'new_fits_this_authorized_batch':540,'metric_records':1500,'decision':status,'G4':'BLOCKED_RAW_WSI_QA','public_files':len(files),'manifest_sha256':sha(R/'ARTIFACT_MANIFEST.json'),'no_patient_level_upload':True};write(R/'DELIVERY.json',delivery);write(P/'DONE_AUTHORIZED.json',delivery);print(json.dumps(delivery),flush=True)
if __name__=='__main__':main()

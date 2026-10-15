"""Confirmed execution, bounded clinical claims, and compact-RNA retention gates."""
import sys,json,re,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from src.phase16r_light.core import *

def diagnostics():
    modelrows=[];fallback=[];basis=[];references=[]
    for seed in cfg()['seeds']:
        for fold in cfg()['folds']:
            z=torch.load(P/'artifacts'/f'{label(seed,fold)}.pt',map_location='cpu',weights_only=False);outer=torch.load(z['outer_feature_path'],map_location='cpu',weights_only=False);fit,test=z['part']['outer_train'],z['part']['outer_test'];h=outer['x']['HALL512'];pca=outer['state']['RNA']['components'];w=z['models']['HALL512__CV3']['w'];r=h[test]@w;total=np.var(h[fit],axis=0).sum()
            for dim in [8,16,32,64,128]:
                pcs=pca[:dim];projection=pcs.T@(pcs@w);rp=h[test]@projection;basis.append({'seed':seed,'fold':fold,'dimension':dim,'training_variance_explained':float(np.var(h[fit]@pcs.T,axis=0).sum()/total),'Cox_coefficient_energy_in_PC_span':float(projection@projection/max(w@w,1e-20)),'heldout_fullRNA_risk_projection_spearman':float(spearmanr(r,rp).statistic),'heldout_fullRNA_risk_projection_relative_MSE':float(np.mean((rp-r)**2)/max(np.var(r),1e-20)),'scope':'posthoc projection diagnostic;no new fit,distillation or outer-derived supervision'})
            for model,m in z['models'].items():
                modelrows.append({'seed':seed,'fold':fold,'model':model,'parameters':m.get('parameters'),'lambda':m.get('lambda'),'base_lambda':m.get('base_lambda'),'path_lambda':m.get('path_lambda'),'actual_optimizer_fit':m.get('actual_optimizer_fit',False),'reference_or_reuse':bool(m.get('reference_path') or m.get('reuse_model'))})
                if m.get('reference_path'):references.append({'seed':seed,'fold':fold,'model':model,'sha256':m['reference_sha256'],'reference_scope':m['reference_kind']})
            for role,record in [('outer',z['outer_wrong'])]+[('inner'+str(k),v['wrong']) for k,v in enumerate(z['inner'])]:
                for partition,counts in record['fallback'].items():fallback.append({'seed':seed,'fold':fold,'role':role,'partition':partition,**counts})
    pd.DataFrame(modelrows).to_csv(R/'MODEL_DETAILS.csv',index=False);pd.DataFrame(fallback).to_csv(R/'WRONG_PAIRING_FALLBACKS.csv',index=False);pd.DataFrame(basis).to_csv(R/'RNA_BASIS_DIAGNOSTICS.csv',index=False);pd.DataFrame(references).to_csv(R/'REFERENCE_REGISTRY.csv',index=False)
    f=pd.read_csv(R/'FOLD_METRICS.csv');records=[]
    for a,b in cfg()['clinical_primary']+[('LOCAL__CV3','GLOBAL_MATCHED__CV3')]:
        for ep in ['DSS','PFI']:
            aa=f[(f.model==a)&(f.endpoint==ep)];bb=f[(f.model==b)&(f.endpoint==ep)];g=aa.merge(bb,on=['seed','fold','endpoint'],suffixes=('_a','_b'))
            for row in g.itertuples():records.append({'endpoint':ep,'a':a,'b':b,'seed':row.seed,'fold':row.fold,'Harrell_delta':row.Harrell_a-row.Harrell_b,'Uno60_delta':row.Uno60_a-row.Uno60_b})
    pd.DataFrame(records).to_csv(R/'FOLD_STABILITY.csv',index=False)

def main():
    audit=json.loads((R/'INDEPENDENT_AUDIT.json').read_text())
    if audit['status']!='PASS' or audit['source']!=source() or json.loads((P/'LOCK.json').read_text())['source']!=source():raise ValueError('independent delivery gate missing')
    diagnostics();summary=pd.read_csv(R/'SUMMARY.csv');pairs=pd.read_csv(R/'PAIRED_COMPARISONS.csv');lambdas=pd.read_csv(R/'LAMBDA_SELECTION.csv');basis=pd.read_csv(R/'RNA_BASIS_DIAGNOSTICS.csv');stable=pd.read_csv(R/'FOLD_STABILITY.csv');primary=pairs[(pairs.endpoint=='DSS')&(pairs.family=='clinical_primary')];clinical_pass=[]
    for key,g in primary.groupby('b'):
        if len(g)==2 and (g.corrected_low>0).all() and (g.positive_seeds==3).all():clinical_pass.append(key)
    retained=[];retention=[]
    for schedule in ['FIXED1','CV3']:
        for family in ['PCA','BPCA']:
            for dim in [16,32,64,128]:
                model=family+str(dim)+'__'+schedule;g=pairs[(pairs.endpoint=='DSS')&(pairs.a==model)&(pairs.b=='HALL512__'+schedule)];passed=len(g)==2 and all(float(g[g.metric==metric].iloc[0].corrected_low)>-margin for metric,margin in [('Harrell',.01),('Uno60',.02)])
                retention.append({'schedule':schedule,'representation':family+str(dim),'dimension':dim,'engineering_retention_passed':passed,**{metric+'_corrected_lower':float(g[g.metric==metric].iloc[0].corrected_low) for metric in ['Harrell','Uno60']}})
                if schedule=='FIXED1' and passed:retained.append(family+str(dim))
    pd.DataFrame(retention).to_csv(R/'RNA_RETENTION_GATES.csv',index=False)
    decision={'status':'COMPLETED','clinical_candidate':'Clinical+S32+nonsemanticCONCH32','clinical_reproduced_point':True,'clinical_confirmed_comparators':clinical_pass,'stable_superiority_confirmed':len(clinical_pass)==3,'local_selection_specificity_confirmed':False,'regularization_finding':'CS andLOCAL both choose10 inall15 CV3cells;pointdifference isnot explained by different CS lambda;extra pathology penalty strongly affects joint fit','RNA_retention_fixed1_passed':retained,'RNA_adaptive_scope':'locked inner selection only;no outer winner','primary_scope':'internal repeatedly studied cohort,not external confirmation','audit':audit,'raw_WSI_QA':'deferred','patient_data_public':False};write(R/'decision.json',decision)
    def point(model,ep='DSS'):
        row=summary[(summary.model==model)&(summary.endpoint==ep)].iloc[0];return f'{row.Harrell_mean:.6f} / {row.Uno60_mean:.6f}'
    def interval(row):return f'[{row.corrected_low:+.6f}, {row.corrected_high:+.6f}]'
    text=['# Phase16R：轻量临床—病理基线与紧凑RNA表示','',
          '**Clinical+S32+CONCH32的0.825923 / 0.806707得到三折内层选择的精确复现，但仍未统计确认超过Clinical+S32及C4。RNA压缩到128维接近完整512维，尚无方案通过预先锁定的信息保留门槛。**','',
          '## 1. 实验协议与审计','',
          '- 原512名患者/65 DSS事件、固定5个外折；PFI仅迁移DSS风险，510名有效患者/129事件。沿用修复后的within-fold Harrell及Uno60，不跨折排序风险。',
          '- 新轻量模型统一full-risk Efron Ridge Cox、同一λ网格0.01/0.1/1/10/OFF；保留固定λ1日程，另用每个outer-train内3折交叉验证选择λ。',
          '- seed31/47/83只改变内层分区和错患者置换。每个inner-train单独拟合Clinical编码、S32标准化、基因筛选、PCA及PC标准化；没有把全outer-train的上游变换提前用于inner-val。',
          '- C4保留原神经架构及15份已验证OOF，统一患者折/终点/评测，但不能称为同一凸模型训练。C4还含RNA，属于历史强模型参照；没有把其带监督latent用作新模型选择输入。',
          f'- 共{audit["outer_model_units"]}个模型评测单元：{audit["actual_outer_optimizer_fits"]}次真实外层优化、{audit["outer_OFF_constants"]}个OFF常数、{audit["outer_references"]}个历史或确定性引用。1410条DSS/PFI逐折指标均独立核验scikit-survival。',
          f'- 独立核验{audit["inner_optimizer_fits_checked"]}次内层优化、{audit["preprocess_banks_independently_checked"]}套预处理；其中8套PCA重新拟合验证。最大梯度{audit["gradient_max"]:.3g}、预处理误差{audit["preprocess_max_error"]:.3g}、风险重放误差{audit["risk_replay_error"]:.3g}。','',
          '## 2. 最强轻量候选能否确认','', '|模型|DSS Harrell / Uno60|','|---|---:|']
    for model in ['CS_P15_REF','C4_REF','CS__CV3','LOCAL_SINGLE_INNER_REF','LOCAL__CV3','GLOBAL_MATCHED__CV3','GLOBAL_ALL__CV3','WRONG_LOCAL__CV3']:text.append(f'|{model}|{point(model)}|')
    text+=['','LOCAL为上轮坐标匹配GRID筛出的每患者≤512 Patch，取冻结CONCH均值后训练折PCA32；不是新训练的局部attention模型。GLOBAL_MATCHED使用全部正面积匹配候选，GLOBAL_ALL使用全部可用CONCH候选。所有池化都先在WSI内均值、再患者等权多切片平均。','',
           '### 锁定主要检验：3比较×2指标，Bonferroni6','', '|比较|指标|Δ|95% CI|校正CI|正向seed|','|---|---|---:|---|---|---:|']
    for row in primary.itertuples():text.append(f'|{row.a}−{row.b}|{row.metric}|{row.delta:+.6f}|[{row.CI_low:+.6f},{row.CI_high:+.6f}]|{interval(row)}|{row.positive_seeds}/3|')
    text+=['','**结论：** 在本次真实配对轻量候选中，LOCAL的点估计最高，可以继续保留为强基线候选；当前主要检验未确认其稳定超过三个参照。相对C4的Uno60点差仅约+0.00374，尤其不能仅凭Harrell较高宣称全面优胜。','',
           '三种内层划分在15个CS和15个LOCAL单元均选择λ10，所以它们的outer预测跨seed完全相同：这是选择结果对本次分区一致，不能当成三次独立数据验证。LOCAL相对本轮CS的Harrell为3折提高、1折下降、1折相同，Uno60为4折提高、1折下降；不能称为每折一致改善。C4原本有神经初始化差异，P15原本使用不同的嵌套选择网格。FOLD_STABILITY.csv保留所有方向。','',
           '## 3. 收益来自局部特征，还是Clinical正则变化','', '|控制|Harrell / Uno60|','|---|---:|']
    for model in ['CS__FIXED1','LOCAL__FIXED1','CS__CV3','CS_AT_LOCAL_LAMBDA','LOCAL_AT_CS_LAMBDA','LOCAL_BLOCK','WRONG_LOCAL_BLOCK']:text.append(f'|{model}|{point(model)}|')
    text+=['','- CS_AT_LOCAL_LAMBDA：不增加病理，只按LOCAL选中的λ重训CS。它与CS_CV3预测相同，因此不同Clinical λ选择在本轮不能解释增益。',
           '- LOCAL_AT_CS_LAMBDA：用CS选中的同一λ拟合含病理的联合模型；它也复现0.825923/0.806707。',
           '- LOCAL_BLOCK：CS惩罚维持10，额外32维病理惩罚固定1，联合重拟合；点估计下降到约0.7791/0.7719。',
           '- λ1全联合模型也未优于CS的Harrell。强正则对新增病理分支及联合系数重优化很重要；这与“Clinical自己换了λ就变好”是不同解释。',
           '- LOCAL对全部匹配GLOBAL的点差仅约+0.000706/+0.002803，配对区间包含零。不能把现有收益确定归于局部选区，现有全局CONCH也几乎达到同样点估计。','',
           '### 机制控制：8比较×2指标，Bonferroni16','', '|比较|指标|Δ|校正CI|','|---|---|---:|---|']
    for row in pairs[(pairs.endpoint=='DSS')&(pairs.family=='clinical_mechanism')].itertuples():text.append(f'|{row.a}−{row.b}|{row.metric}|{row.delta:+.6f}|{interval(row)}|')
    text+=['','WRONG_LOCAL只在各自训练/验证/测试分区内置换，优先癌种+Stage，单例退化次数见WRONG_PAIRING_FALLBACKS.csv；没有跨分区配对。其与真实配对的差异仍未通过校正检验。风险分解见CLINICAL_PATH_RISK_COMPONENTS.csv，只用于计算贡献，不赋予因果含义。','',
           '## 4. 16/32/64/128维RNA表示','',
           '所有方法使用同一训练折内重新选择的Hallmark富集512基因。PCA为全局PCA，BPCA在8个真实通路成员子集分别PCA，均衡分配精确总维数；重叠基因保留，不虚构成员。PCA/BPCA输出逐维训练标准化，沿用上一轮PC处理。BPCA缺少匹配随机分组对照，不能单凭点差证明通路先验有效。','',
           '|固定λ1表示|全局PCA Harrell / Uno60|通路内PCA Harrell / Uno60|不重标PC的ORTHO Harrell / Uno60|','|---|---:|---:|---:|']
    for dim in [16,32,64,128]:text.append(f'|{dim}维|{point("PCA"+str(dim)+"__FIXED1")}|{point("BPCA"+str(dim)+"__FIXED1")}|{point("ORTHO"+str(dim)+"__FIXED1")}|')
    text+=['',f'完整HALL512固定λ1：**{point("HALL512__FIXED1")}**。三折选择后的完整512：{point("HALL512__CV3")}。','',
           '|三折内层选λ表示|全局PCA Harrell / Uno60|通路内PCA Harrell / Uno60|ORTHO Harrell / Uno60|','|---|---:|---:|---:|']
    for dim in [16,32,64,128]:text.append(f'|{dim}维|{point("PCA"+str(dim)+"__CV3")}|{point("BPCA"+str(dim)+"__CV3")}|{point("ORTHO"+str(dim)+"__CV3")}|')
    text+=['','ORTHO使用相同全局PCA基，但不再除以各PC标准差；正交投影下保持原基因空间的系数L2范数。PC标准化会改变该惩罚，因此将ORTHO单列为归一化控制。通路内拼接因重叠而非全局正交，也不宣称正则完全等价。','',
           '### 是否保留足够排序信息','',
           '预先锁定工程容许差：Harrell≤0.01、Uno60≤0.02；不是临床非劣界值。必须两项校正CI下界都高于负容许差才通过，不能用P>0.05或点估计接近来宣布保留。主要8个PCA/BPCA比较共Bonferroni16，CV3另作同样family的敏感性。','',
           '|表示|日程|Harrell校正下界|Uno60校正下界|保留门槛|','|---|---|---:|---:|---|']
    for row in retention:text.append(f'|{row["representation"]}|{row["schedule"]}|{row["Harrell_corrected_lower"]:+.6f}|{row["Uno60_corrected_lower"]:+.6f}|{"PASS" if row["engineering_retention_passed"] else "NOT_ESTABLISHED"}|')
    text+=['','目前没有主要紧凑表示通过上述门槛。128维比16维更接近512的Harrell；PCA128的Uno60点估计甚至稍高，但区间仍不足以证明整体排序信息已保留。也不能把所有未通过门槛的方法直接判为显著更差。','',
           '## 5. 只能在inner选择维度，不能选outer最高点','',
           f'锁定的COMPACT_ADAPTIVE从PCA/BPCA×16/32/64/128×λ中只按inner-CV选择，outer直接复用对应已完成模型。其DSS结果为**{point("COMPACT_ADAPTIVE")}**；与完整512-CV3的Harrell差约−0.04987，校正区间不含零。',
           'ADAPTIVE_COMPACT_SELECTION.csv展示各折所选16至128维及不同家族/λ，选择并不一致。该负结果限制了根据外折点估计追选“最佳维度”的解释；不证明所有监督紧凑表示均不可行。','',
           '## 6. 归一化与丢失方向的诊断','',
           'ORTHO8约0.5253/0.5413，未明显好于历史标准化PCA8约0.5283/0.5438。16–128维的严格相同basis归一化比较另列RNA_normalization family8；PC缩放不能单独解释此前8维的排序损失。8维历史参考的输入重标准化有浮点级差异，严格同basis控制以16–128维为准。','',
           '|维度|训练方差解释比例均值|完整Cox系数在PC子空间的能量比例|heldout原始RNA风险投影秩相关|','|---|---:|---:|---:|']
    for dim,g in basis.groupby('dimension'):text.append(f'|{dim}|{g.training_variance_explained.mean():.4f}|{g.Cox_coefficient_energy_in_PC_span.mean():.4f}|{g.heldout_fullRNA_risk_projection_spearman.mean():.4f}|')
    text+=['','这些是将outer-train已拟合512维Cox系数投影到训练PCA子空间的事后诊断，没有训练新教师、没有蒸馏，也未用外折错排定义监督。解释表达方差较多，不等于保留了与预后线性排序相关的全部方向；该表衡量的是已拟合RNA风险的保持，不是患者级医学机制真值。','',
           '## 7. 决策、优势与风险','',
           '**临床—病理：** 保留Clinical+S32+CONCH32作为高点估计的轻量候选，统一协议下其数值可复现；尚未确认统计优于CS/C4，也未确认局部选区独有价值。记录λ10的联合岭正则作为当前候选配置，不追逐更多融合层。','',
           '**RNA：** 继续保留完整Hallmark512作为信息保留基线。当前PCA/通路内PCA的紧凑方案没有通过预设门槛，inner自动选择还出现明显排序损失；不以低维表示直接替代完整RNA。RNA-only表现不等于Clinical+S32条件增量，后续必须另行验证条件互补。','',
           '**优势：** 相同新模型目标/λ网格、完整内层预处理隔离、固定λ与同λ机制控制、全局/局部及错患者对照、原始512与维度匹配表示、凸优化梯度与独立指标核验。','',
           '**风险：** 同队列反复探索、65事件、固定模型bootstrap不包含全流程重训方差；三seed内层λ一致不是独立样本证据；C4原架构/模态/训练日程不同；局部池化受计算坐标匹配覆盖限制，仍无原图QA；BPCA无随机分组控制；工程保留容许差不是临床界值。','',
           '**置信度：** 数值执行与隔离审计为高；当前点估计受正则影响的计算证据为高；泛化优胜、局部区域机制和紧凑RNA医学充分性仍未确认。','',
           '## 拓展思考','', '应将“可复现的高点估计”“配对增量已确认”“局部病理机制成立”分开。当前第一项成立，后两项还没有足够证据。','',
           '## 你可能想知道','', '本轮只使用已冻结特征及CPU凸优化，没有新增GPU网络、下载WSI或公开患者数据。常用权重与正确历史检查点继续保留。']
    (R/'RESULTS.md').write_text('\n'.join(text),encoding='utf8')
    (R/'REPRODUCE.md').write_text('''# 复现\n\n从dd559068c61fe42f5764a8e080d5889ad135ba51继续；需要相同私有512患者数据、Phase12/15/16检查点与缓存。仓库不含患者数据或权重。\n\n```bash\nexport PYTHONPATH=/root/phase10_trimodal_eval_deps\nexport OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n/root/miniconda3/bin/python scripts/pipeline_phase16r_light_rna.py\n```\n\n私有输出/root/phase16r_light_rna_private；训练程序仅同源缓存恢复，source不匹配即拒绝。官方CONCH-Z权重本轮无需加载。若重启导致/dev/shm/phase16_routing_full.npy丢失，先按上一轮query审计脚本重建并核验原始cache哈希，再执行。\n\n15个单元每个19输入×两日程，另含机制控制、自适应compact和四类历史引用；固定λ1的非错患者模型跨seed明确复用。C4引用真实历史OOF，不声称重新训练C4或将其变成凸模型。patient文件、基因索引、逐患者风险、模型参数全部私有。\n''',encoding='utf8')
    write(R/'ENVIRONMENT.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'workers':5,'CPU_threads_per_library':2,'GPU_training':False,'inner_CV':3,'source':source()})
    # Preserve all three published Phase16 deliveries, including source files.
    previous=[]
    for folder in ['phase16_semantic_routing','phase16_semantic_routing/semantic_completion','phase16_prompt_rna_diagnostic']:
        m=json.loads((ROOT/'reports'/folder/'ARTIFACT_MANIFEST.json').read_text())
        for row in m['public_files']:
            if sha(ROOT/row['path'])!=row['sha256']:raise ValueError('previous artifact modified '+row['path'])
            previous.append(row['path'])
    files=[CONFIG]+sorted((ROOT/'src/phase16r_light').glob('*.py'))+[ROOT/'scripts'/name for name in ['run_phase16r_light_rna.py','validate_phase16r_light_rna.py','report_phase16r_light_rna.py','pipeline_phase16r_light_rna.py','publish_phase16r_light_rna.py']]+[p for p in sorted(R.rglob('*')) if p.suffix in ['.csv','.json','.md'] and p.name not in ['ARTIFACT_MANIFEST.json','DELIVERY.json']]
    pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----');manifest=[]
    for file in files:
        data=file.read_bytes()
        if pattern.search(data):raise ValueError('public artifact contains protected data '+str(file))
        if file.suffix=='.csv' and set(pd.read_csv(file,nrows=0).columns)&{'patient_index','patient_id','risk','time','event','coords'}:raise ValueError('patient-level table not public')
        manifest.append({'path':file.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':sha(file)})
    write(R/'ARTIFACT_MANIFEST.json',{'status':'PASS','source':source(),'public_files':manifest,'previous_public_files_unchanged':len(set(previous)),'no_patient_data_public':True});write(R/'DELIVERY.json',{'status':'PASS','source':source(),'public_files':len(manifest),'manifest_sha256':sha(R/'ARTIFACT_MANIFEST.json'),'audit':audit,'clinical_superiority_confirmed':decision['stable_superiority_confirmed'],'RNA_retention_passed':retained});print('PHASE16R_DELIVERY_PASS',len(manifest),flush=True)

if __name__=='__main__':main()

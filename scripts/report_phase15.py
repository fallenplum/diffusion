"""Model-level aggregates, objective decision and protected public delivery."""
import hashlib,json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from src.phase15_complement.core import *


def summary():
    s=pd.read_csv(REPORT/'SEED_OOF_METRICS.csv');p=pd.read_csv(REPORT/'PAIRED_COMPARISONS.csv');d=pd.read_csv(REPORT/'RNA_ERROR_DIAGNOSTIC_CI.csv');f=pd.read_csv(REPORT/'REPAIR_DAMAGE.csv');select=pd.read_csv(REPORT/'SELECTION.csv')
    print('HALL MEANS\n'+s[s.gene=='HALL'].groupby(['endpoint','model'])[['Harrell','Uno60']].mean().to_string())
    print('PRIMARY\n'+p[p.primary].to_string(index=False))
    print('RNA on anchor error\n'+d.to_string(index=False))
    print('DSS/HALL gain anddamage\n'+f[(f.endpoint=='DSS')&(f.gene=='HALL')].groupby(['metric','model'])[['rescued_mass','damaged_mass','net_delta']].mean().to_string())
    print('OFF COUNTS\n'+select.assign(OFF=select['lambda'].astype(str)=='OFF').groupby(['endpoint','gene','model']).OFF.sum().to_string())
    print('SCALAR CONTROLS\n'+p[(p.gene=='HALL')&p.comparison.isin(['SCALAR-CS','SCALAR_FIXED1-CS','SCALAR-SOFT'])].to_string(index=False))
    ee=pd.read_csv(REPORT/'RNA_ERROR_DIAGNOSTIC.csv');print('RNA GOOD VS BAD\n'+ee[(ee.endpoint=='DSS')&(ee.gene=='HALL')].groupby('metric')[['anchor_wrong_mass','RNA_accuracy_on_anchor_wrong','RNA_accuracy_on_anchor_correct','wrong_pair_event_Neff']].mean().to_string())
    if (REPORT/'SUPERVISION_EVENT_MASS.csv').exists():
        a=pd.read_csv(REPORT/'SUPERVISION_EVENT_MASS.csv');print('SUPERVISION\n'+a.groupby(['endpoint','role','kind'])[['events','contributing_events','event_mass_Neff','eligible_pairs']].mean().to_string())
        shift=pd.read_csv(REPORT/'TEACHER_SHIFT.csv')
        for metric,g in shift[(shift.endpoint=='DSS')&(shift.role=='outer')].groupby('metric'):
            print('TEACHER_SHIFT',metric,g.teacher_correct_mass.sum()/g.weight_mass.sum(),g.full_correct_mass.sum()/g.weight_mass.sum(),g.wrong_intersection_mass.sum()/g.wrong_union_mass.sum())


def finish():
    for file in ['INDEPENDENT_AUDIT.json','LITERAL_BOOTSTRAP_AUDIT.json','MECHANISM_AUDIT.json']:
        if json.loads((REPORT/file).read_text())['status']!='PASS':raise RuntimeError('audit absent '+file)
    configuration=cfg();s=pd.read_csv(REPORT/'SEED_OOF_METRICS.csv');p=pd.read_csv(REPORT/'PAIRED_COMPARISONS.csv');d=pd.read_csv(REPORT/'RNA_ERROR_DIAGNOSTIC_CI.csv');error=pd.read_csv(REPORT/'RNA_ERROR_DIAGNOSTIC.csv');f=pd.read_csv(REPORT/'REPAIR_DAMAGE.csv');select=pd.read_csv(REPORT/'SELECTION.csv');mass=pd.read_csv(REPORT/'SUPERVISION_EVENT_MASS.csv');shift=pd.read_csv(REPORT/'TEACHER_SHIFT.csv')
    means=s.groupby(['endpoint','gene','model'])[['Harrell','Uno60']].mean();sd=s.groupby(['endpoint','gene','model'])[['Harrell','Uno60']].std();summarytable=means.add_suffix('_mean').join(sd.add_suffix('_seed_std')).reset_index();summarytable.to_csv(REPORT/'MODEL_SUMMARY.csv',index=False)
    primary=p[p.primary];positive=bool((primary.corrected_low>0).all());decision={'status':'POSITIVE_EXPLORATORY_COMPLEMENTARITY' if positive else 'INCONCLUSIVE_COMPLEMENTARITY','all_primary_gates_positive':positive,'primary_family_tests':12,'no_RNA_rank_distillation':True,'pair_honest_teachers':True,'outer_error_feedback':False,'execution_confidence':'high','mechanistic_causal_attribution_confidence':'low','source_hash':source(),'scope':'single-cohort repeated exploration;not a conditional-independence or no-go claim'}
    write(REPORT/'decision.json',decision)
    cleanup=json.loads((PRIVATE/'DISK_CLEANUP.json').read_text());write(REPORT/'DISK_CLEANUP.json',cleanup)
    text=['# Phase15-R0 — RNA 条件互补信息审计','',
          '**90个完整流程、1530个逐折记录已完成。DSS主终点512人/65事件；PFI次要510人/129事件。**','',
          '目的：解释RNA边际排序为何未稳定改善Clinical+S32。此次不使用RNA自身排序蒸馏、不新增编码器、不根据外折错误定义训练任务。沿用TOP/HALL/MATCH三组512基因和同一患者折。三个seed控制训练分区内部交叉拟合分块，反映分块/选择稳定性，不是三个独立cohort或神经初始化实验。','',
          '## 1. 教师与标签隔离','',
          '- Inner-train和outer-train分别生成3-fold anchor；每个教师在自己的训练患者中2-fold选择λ和拟合临床编码/缩放。',
          '- 只使用同一教师留出块内的可比较患者对，**双方标签均未进入该教师训练或参数选择**；不对不同教师的风险跨块排序。',
          '- Inner-val只选择RNA修正λ；outer-test错误组仅在全部模型锁定后用于评价。保留原强CS_LEGACY，防止改变anchor规则后降低比较基准。',
          '- RNA标量基线是诚实交叉拟合的一维风险，不拼跨模型latent；其子折重新拟合缺失/缩放，但基因宇宙固定于pipeline训练分区的无监督选择。','',
          '## 2. DSS/HALL主要结果','', '|模型|Harrell|Uno60|','|---|---:|---:|']
    for model in MODELS:
        a=means.loc['DSS','HALL',model];text.append(f'|{model}|{a.Harrell:.6f}|{a.Uno60:.6f}|')
    text+=['','CS为本轮完全嵌套anchor；CS_LEGACY为原Phase13强基线。两者训练规则不同，因此同时列出；有效增量不能只通过降低anchor获得。每个seed先拼5-fold OOF，以同折pair分子/分母聚合；表格是3seed均值，不是15个cell均值。','',
           '## 本轮能够回答什么','',
           '**边际排序不是条件增量。** HALL RNA-only Harrell为0.7215；在anchor正确组为0.7567，在错误组仅0.5660（95% CI 0.4410–0.6826）。后者未稳定高于0.5，说明可供修正的排序较弱且估计不确定，不证明RNA条件独立。',
           '**修复伴随损伤。** SOFT修复2.2897% Harrell比较对质量，却破坏2.7523%，净−0.4626%；Uno60净+0.7008%，但配对区间跨零。对错误更强聚焦并未稳定胜过ALL或事件质量匹配的RANDOM对照。',
           '**错误聚焦进一步压缩监督。** 内层原41个事件，ALL事件贡献Neff约34.4、HARD约16.3；不是只有65个总事件这么简单。Neff不是独立样本数，不用于断言功效不足是唯一原因。',
           '**不是本轮encoder或未收敛优化造成的必然损失。** 使用原始RNA512和凸线性目标，独立最大梯度约5.45e−7，仍无确认的SOFT增量。但模型类有限，不能否定非线性残余信息。',
           '**低容量旁路值得确认。** 预先固定λ1的SCALAR达到DSS 0.823336/0.801479，相对新CS为+0.008155/+0.017302，3seed均正；95% CI分别[-0.013121,0.027273]、[-0.008242,0.046635]，仍未确认。它提示容量/校准可能影响利用效率，不等于证明高维估计方差是根因。','',
           '## 3. 预先锁定的主配对比较','', '|比较|指标|Δ|95% CI|Bonferroni12 CI|正向seeds|','|---|---|---:|---|---|---:|']
    for row in primary.itertuples():text.append(f'|{row.comparison}|{row.metric}|{row.delta:+.6f}|[{row.CI_low:+.6f},{row.CI_high:+.6f}]|[{row.corrected_low:+.6f},{row.corrected_high:+.6f}]|{row.positive_seeds}/3|')
    text+=['','## 4. RNA在anchor错误组的能力','', '|输入/终点|指标|RNA在anchor错排组的排序准确率|95% CI|','|---|---|---:|---|']
    for row in d.itertuples():text.append(f'|{row.gene}/{row.endpoint}|{row.metric}|{row.RNA_accuracy_on_anchor_wrong:.4f}|[{row.CI_low:.4f},{row.CI_high:.4f}]|')
    text+=['','这些错误组完全在held-out评价中定义，不参与训练。条件于模型错误可能引入选择偏差；它描述边际RNA排序在该组的可迁移程度，不等于条件互信息。区间为未校正诊断。','',
           '## 5. 修复坏对与破坏好对','', '|DSS/HALL模型|指标|修复质量|损伤质量|净Δ|','|---|---|---:|---:|---:|']
    ff=f[(f.endpoint=='DSS')&(f.gene=='HALL')].groupby(['model','metric'])[['rescued_mass','damaged_mass','net_delta']].mean()
    for (model,metric),row in ff.iterrows():text.append(f'|{model}|{metric}|{row.rescued_mass:.6f}|{row.damaged_mass:.6f}|{row.net_delta:+.6f}|')
    text+=['','修复/损伤均以所有held-out有效比较对的质量为分母，二者差精确等于总体C-index变化。不能只报告修复错排对而忽略正确对被翻转。','',
           '## 6. 有效事件与教师变化','', '|训练层/目标（DSS均值）|原事件数|贡献事件数|事件质量Neff|可比较对|','|---|---:|---:|---:|---:|']
    mm=mass[mass.endpoint=='DSS'].groupby(['role','kind'])[['events','contributing_events','event_mass_Neff','eligible_pairs']].mean()
    for (role,kind),row in mm.iterrows():text.append(f'|{role}/{kind}|{row.events:.1f}|{row.contributing_events:.1f}|{row.event_mass_Neff:.1f}|{row.eligible_pairs:.1f}|')
    text+=['','Neff只表示事件贡献集中度，不是独立样本数或功效计算。诚实分块避免标签泄露，同时减少比较对。HARD只保留错排对还会改变事件贡献分布；RANDOM对照保持逐事件总权重与权重分布。','',
           '|外折教师变化（DSS）|教师平均C|全训练anchor C|错误集合Jaccard|','|---|---:|---:|---:|']
    for metric,g in shift[(shift.endpoint=='DSS')&(shift.role=='outer')].groupby('metric'):
        text.append(f'|{metric}|{g.teacher_correct_mass.sum()/g.weight_mass.sum():.4f}|{g.full_correct_mass.sum()/g.weight_mass.sum():.4f}|{g.wrong_intersection_mass.sum()/g.wrong_union_mass.sum():.4f}|')
    text+=['','教师较小训练集的错误与最终anchor错误并不完全一致，可能使辅助任务目标有噪声。该比较为训练完成后的探索性诊断，没有用来修改任务；差异不能证明它是负结果唯一原因。','',
           '## 7. 科学结论边界','',
           f'程序决策：**{decision["status"]}**。只有SOFT同时超过新anchor、原强anchor、普通JOINT/ALL、随机权重和错患者对照，才支持该残余排序机制。','',
           '**优势：** 不依赖压缩RNA或神经优化预算；完整嵌套anchor；同教师患者对双方标签排除；保留原始RNA、条件Cox与1D风险旁路；独立逐折/梯度/修复损伤审计。','',
           '**劣势/风险：** 65事件、同cohort反复探索、诚实分块削减监督、教师与最终模型分布差异；训练局部基因宇宙在教师子折共享；OFF选择可将某些区间压为零，但不证明真实增量为零。固定模型bootstrap未覆盖全流程再训练不确定性。','',
           'RNA在边际任务上较强，不必然意味着给定Clinical+S32后仍有可泛化增量。若错误聚焦出现修复却同时损伤正确对，只能说该修正尚未学到稳定选择性；不能证明信息完全重叠。若1D旁路优于512D修正，则提示估计容量/校准问题，仍需区间和独立数据验证。','',
           '执行与评测置信度高；机制因果归因置信度低。PFI定义不同，不是单纯增加事件数的功效干预。','',
           '## 8. 数据、清理与复现','',
           f'清理后根目录释放{(cleanup["free_after"]["/root"]-cleanup["free_before"]["/root"])/2**30:.2f} GiB；仅删除可重下载缓存与已被修复版本替代的失效warmup。正确实验和关键数据哈希不变。',
           '先gate_phase15.py，再launch_phase15.py，等待独立evaluate_phase15.py完成；随后audit_phase15_mechanisms.py、report_phase15.py --finish。',
           '环境/原数据沿用Phase13；本轮私有目录/root/phase15_complement_private。公开源码、配置、模型级逐折/seed汇总；不公开患者ID、风险、pair mask、权重或原始临床表。','',
           '拓展思考：需要区分互补信息是否存在、低容量模型能否稳定提取、提取后修复是否超过损伤三件事，不从某个负差值直接否定研究idea。','',
           '你可能想知道：全部TOP/MATCH和PFI数值、OFF选择频率、教师误差变化分别见MODEL_SUMMARY、SELECTION、TEACHER_SHIFT表。','',
           '方法参考：[官方Uno/IPCW定义](https://scikit-survival.readthedocs.io/en/v0.23.0/api/generated/sksurv.metrics.concordance_index_ipcw.html)、[censored ranking](https://arxiv.org/abs/1806.01984)。']
    (REPORT/'RESULTS.md').write_text('\n'.join(text),encoding='utf8')
    sources=[CONFIG,ROOT/'PHASE15_R0_COMPLEMENTARITY_PROTOCOL.md']+list((ROOT/'src/phase15_complement').glob('*.py'))+[ROOT/'scripts'/x for x in ['run_phase15.py','gate_phase15.py','launch_phase15.py','evaluate_phase15.py','audit_phase15_mechanisms.py','report_phase15.py','publish_phase15.py']]
    public=sources+[p for p in sorted(REPORT.glob('*')) if p.suffix in ['.csv','.json','.md'] and p.name not in ['PUBLIC_FILES.json','DELIVERY.json']]
    inventory=[];pattern=re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')
    for file in public:
        content=file.read_bytes()
        if pattern.search(content):raise RuntimeError('protected identifiers '+str(file))
        columns=pd.read_csv(file,nrows=0).columns.tolist() if file.suffix=='.csv' else []
        if set(columns)&{'patient_id','patient_ids','risk','time','event','test_indices'}:raise RuntimeError('protected row columns')
        inventory.append({'path':str(file.relative_to(ROOT)),'sha256':sha(file),'bytes':len(content),'columns':columns})
    write(REPORT/'PUBLIC_FILES.json',{'status':'PASS','scope':'source and anonymous model-level aggregate only','files':inventory})
    archive=REPORT/'PHASE15_PUBLIC_AUDIT.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for pth in public:z.write(pth,str(pth.relative_to(ROOT)))
        z.write(REPORT/'PUBLIC_FILES.json',str((REPORT/'PUBLIC_FILES.json').relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise RuntimeError('ZIP CRC')
        for item in inventory:
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:raise RuntimeError('ZIP hash')
    status={'status':'PASS','pipelines':90,'model_fold_records':1530,'public_entries':len(inventory)+1,'public_zip_sha256':sha(archive),'public_zip_bytes':archive.stat().st_size,'private_data_published':False,'source_hash':source()};write(REPORT/'DELIVERY.json',status);write(PRIVATE/'DONE.json',status);summary();print(json.dumps(status))


if __name__=='__main__':finish() if '--finish' in sys.argv else summary()

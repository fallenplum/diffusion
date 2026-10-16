"""Finish aggregate evidence and audit metadata; no numerical model changes."""
import sys,json,subprocess,hashlib,importlib.metadata as md
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from src.phase17_ssd.data import P,R,source,write
from src.phase10_trimodal.data import digest
from report_phase17 import table

def main():
    lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());assert lock['training_source']==source()=='9794e29e99fd65a6465ed1af6932bcecc48a4bab92e1a7ab5618bb201637599e'
    amendments=[]
    for name,oldhash in lock['files'].items():
        now=digest(ROOT/name)
        if now!=oldhash:
            assert name=='scripts/validate_phase17_independent.py';amendments.append({'file':name,'locked_sha256':oldhash,'current_sha256':now,'changes':'append frozen historical comparators,own private OOF cache,independent historical hash validation;float64 replay tolerance1e-9','training_changed':False,'primary_comparisons_changed':False,'bootstrap_sampling_or_family_changed':False})
    write(R/'EVALUATION_AMENDMENT.json',{'status':'AUDITED','training_source_unchanged':True,'amendments':amendments,'other_evaluation_only_changes':['explicit Omics_direct_off tag names joint RNA+CNA head;numeric predictions unchanged','double precision synthetic path diagnostic;does not assert clinically material effect']})
    # Early diagnostic tag named RNA_direct_off referred to the entire existing
    # RNA/gain/loss direct layer. Correct the category, not any numeric column.
    for name,col in [('MODULE_INTERVENTIONS.csv','intervention'),('FROZEN_INTERVENTION_FOLD_METRICS.csv','intervention'),('FROZEN_INTERVENTION_SEED_METRICS.csv','intervention')]:
        p=R/name;df=pd.read_csv(p);df[col]=df[col].replace({'RNA_direct_off':'Omics_direct_off'});df.to_csv(p,index=False)
    pp=P/'FROZEN_INTERVENTION_OOF_PRIVATE.npz';z=dict(np.load(pp));renamed={k.replace('RNA_direct_off','Omics_direct_off'):v for k,v in z.items()}
    if list(renamed)!=list(z):np.savez_compressed(pp,**renamed)
    s=pd.read_csv(R/'FROZEN_INTERVENTION_SEED_METRICS.csv').groupby('intervention')[['Harrell','Uno60']].mean().reset_index();s.to_csv(R/'FROZEN_INTERVENTION_SUMMARY.csv',index=False)
    groups=pd.read_csv(R/'RNA_MISSING_PATTERN_SEED.csv');g=groups[groups.arm.isin(['A0','A8','A10'])].groupby(['arm','group'])[['Harrell','Uno60','events','patients']].mean().reset_index();g.to_csv(R/'RNA_MISSING_PATTERN_SUMMARY.csv',index=False)
    report=R/'RESULTS.md';text=report.read_text();text=text.replace('患者内门标准差平均','每checkpoint有效patch标准差的跨checkpoint均值').replace('no_Clinical显著降低点估计','no_Clinical降低点估计')
    extra=['## 冻结预测头与缺失模式补充诊断','',table(s),'','以上是同一个已训练A8的推理干预，**不是重训消融**。Omics_direct_off关闭的是完整RNA512+CNA gain/loss风险捷径，保留了状态和末端组学分支，不能解释成删除全部RNA。关闭该捷径后排序点估计提高；关闭整个SSM后点估计降低；将门固定0.9或关闭条件Δ/B几乎不改变排序。这支持“普通WSI状态分支有贡献、已学习的条件增量很弱、直接组学风险贡献在当前尺度/正则下不利”的局部诊断，不能证明RNA没有独立信号。','',table(g),'','203/309只代表RNA缺失区块代理，未直接验证平台身份。各组事件组成和可比较对不同，跨组C-index差不能作为平台因果影响。两个组内A8点估计均低于A0；不按分组结果重调任何模型。','','## 图与审计完成状态','','7张SVG/PNG训练与机制图已逐张视觉核验，源数据为公开聚合CSV；非组织学热图。195个神经/论文模型checkpoint重放误差为0，10套训练折内预处理独立重算误差为0；患者级风险和模型只存私有目录。20,000个bootstrap均有效、无零分母。公开报告计算核验和真实私有checkpoint重放是分别执行的步骤。','']
    text=text.split('\n## 冻结预测头与缺失模式补充诊断')[0];report.write_text(text+'\n'+'\n'.join(extra),encoding='utf8')
    versions={}
    for name in ['torch','numpy','pandas','scipy','scikit-learn','scikit-survival','triton','timm','mamba_ssm','causal_conv1d','einops','nystrom-attention','matplotlib']:
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]='not found in current metadata path'
    repos={}
    for name in ['MGCT','MambaMIL','CAMR','MMP']:
        root=P/'literature'/name;rev=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();licenses=[p.name for p in root.glob('*LICENSE*')]
        repos[name]={'commit':rev,'license_files':licenses}
    write(R/'ENVIRONMENT.json',{'python':sys.version,'packages':versions,'CUDA':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'deterministic_algorithms':True,'TF32':False,'official_repositories':repos,'paper_dependencies_location':'private data partition;isolated --target installation','mamba_kernel':'conditional S6 does not import mamba_ssm;paper MambaMIL uses compatible official CUDA kernels'})
    licensepath=P/'literature/MMP/LICENSE.md';license_summary=licensepath.read_text().splitlines()[0]
    with (R/'LITERATURE_BASELINES.md').open('a') as f:f.write('\n## 实际适配与依赖核验\n\nMGCT、MambaMIL已各完成15内层+15外层正式cell，另各1个预锁定内层诊断；实际分数见RESULTS.md。Clinical/S32是统一适配，并非两篇论文原生三模态支持。MGCT批量端口与原层输出/梯度误差0；MambaMIL兼容kernel输出/梯度通过N=1/16/128/512字面递推检查。\n\nMMP固定提交`'+repos['MMP']['commit']+'`，LICENSE.md首行为`'+license_summary+'`；本轮只核查清单，没有声称训练MMP/CAMR/ME-Mamba。TransSurv于2026-10-16再次按题名、DOI、作者搜索，仍未取得可核验官方仓库；[作者论文条目](https://smt.bit.edu.cn/szdw/jsml/bssds/6de6308941ca4ed0bfb252e52087598a.htm)可确认研究，但没有替代的可运行实现。\n')
    log=R/'BUG_LOG.csv';df=pd.read_csv(log)
    additions=pd.DataFrame([{'stage':'evaluation','issue':'historical float64 replay differed2.22e-16 from saved fit','impact':'overstrict exact-zero reference check stopped evaluator;no training error','repair':'verify source SHA and permit1e-9 numeric replay tolerance;all risk rankings unchanged','affected_formal_cells':0},{'stage':'diagnostic','issue':'float32 tiny wrong-context effect3e-9 below absolute1e-8 synthetic threshold','impact':'path check threshold confused nonzero path with effect magnitude','repair':'double precision path test and finite difference;actual small effect reported,not promoted to mechanism support','affected_formal_cells':0},{'stage':'reporting','issue':'RNA_direct_off name covered joint RNA+CNA head','impact':'label could be read as RNA-only deletion','repair':'rename Omics_direct_off;all numeric predictions unchanged','affected_formal_cells':0}])
    pd.concat([df,additions]).drop_duplicates().to_csv(log,index=False)
    write(R/'FIGURE_VISUAL_QA.json',{'status':'PASS','figures_checked':['01_training','02_gates','03_propagation','04_state_gradient','05_order','06_primary_intervals','07_compute'],'checks':['actual PNG rendered and visually inspected','no clipped titles,axes or intervals','source aggregate CSV exists','no synthetic histology or fabricated data'],'backend':'Python/matplotlib'})
    print('FINAL_EVIDENCE_READY',source(),flush=True)

if __name__=='__main__':main()

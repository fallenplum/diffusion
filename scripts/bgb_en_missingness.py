"""Read-only missingness strata from existing completed final-risk predictions.

No model forward, training, optimizer, subgroup refit or GPU access is used.
Every subgroup keeps the original complete fitting partition for censoring
weights. Patient risks and indices are consumed privately and never exported.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.bgb_en.data import load_raw,sha256
from src.bgb_en.protocol import fixed_inner_splits
from src.phase10_trimodal.reference import components,uno_weights

GROUPS=('ALL','RNA_COMPLETE','RNA_MISSING_BLOCK','CNA_COMPLETE','CNA_MINOR_MISSING')


def require(value,message):
    if not bool(value):raise ValueError(message)


def deterministic(arm):
    return arm in ('P0','O0','O1','F0','F1') or arm.startswith('O2_')


def missingness_groups(raw,canonical=True):
    rna=np.isfinite(raw['rna']);cna=np.isfinite(raw['cna'])
    require(rna.ndim==cna.ndim==2 and len(rna)==len(cna),'Invalid gene measurement masks')
    complete=rna.all(1);cc=cna.all(1)
    groups={'ALL':np.ones(len(rna),bool),'RNA_COMPLETE':complete,'RNA_MISSING_BLOCK':~complete,
            'CNA_COMPLETE':cc,'CNA_MINOR_MISSING':~cc}
    if canonical:
        require(len(rna)==512 and int(complete.sum())==309 and int((~complete).sum())==203,
                'Canonical RNA missingness group sizes changed')
        missing=~rna[~complete]
        require(np.all(missing==missing[0]) and int(missing[0].sum())==3015,
                'RNA incomplete rows are not the audited shared collapsed-gene block')
        require(int(cc.sum())==382 and np.all((~cna[~cc]).sum(1)==2),
                'Canonical CNA missingness group sizes/pattern changed')
    return groups


def stratified_metrics(train_time,train_event,time,event,risk,groups):
    """Estimate IPCW once from original fit, then subset the same evaluated risk."""
    t,e,r=np.asarray(time),np.asarray(event),np.asarray(risk,float)
    require(t.ndim==1 and e.shape==r.shape==t.shape and np.isfinite(r).all(),'Invalid saved final risks')
    weights=uno_weights(train_time,train_event,t,e,tau=60.)
    rows={}
    for group,mask in groups.items():
        mask=np.asarray(mask,bool);require(mask.shape==t.shape,'Subgroup mask mismatch')
        hn,hd=components(t[mask],e[mask],r[mask])
        un,ud=components(t[mask],e[mask],r[mask],weights[mask])
        rows[group]=dict(n=int(mask.sum()),events=int(e[mask].sum()),events_before60=int(((e==1)&(t<60)&mask).sum()),
            Harrell=hn/hd if hd else None,Uno60=un/ud if ud else None,
            Harrell_num=hn,Harrell_den=hd,Uno_num=un,Uno_den=ud,
            pair_status='DEFINED' if hd>0 and ud>0 else 'NA_NO_ELIGIBLE_PAIRS_FOR_ONE_OR_BOTH_METRICS')
    return rows


def planned_units(config,stages):
    plan=[]
    for stage in stages:
        arms=config[stage]['arms'][:]
        if stage=='R3':arms+=config[stage].get('additional_real_retraining',[])
        for arm in arms:
            for part in range(config['prototype']['inner_splits']):
                for seed in ([31] if deterministic(arm) else config['prototype']['neural_seeds']):
                    plan.append((stage,arm,part,seed))
    return plan


def load_final_cell(receipt_path,raw,config):
    receipt=json.loads(receipt_path.read_text());artifact=receipt_path.parent/'model.pt'
    require(receipt.get('status')=='COMPLETE','A completed-cell receipt has invalid status')
    require(artifact.is_file() and sha256(artifact)==receipt.get('artifact_sha256'),'Saved model artifact hash mismatch')
    require(receipt.get('outer_test_evaluated') is False,'Expected inner-only completed cell')
    require(receipt['lineage']['cohort_sha256']==raw['audit']['cohort_sha256']
            and receipt['lineage']['endpoints_sha256']==raw['audit']['endpoints_sha256'],
            'Completed risk uses a different canonical cohort/endpoint source')
    model=torch.load(artifact,map_location='cpu',weights_only=False)
    require(model.get('lineage')==receipt['lineage'] and model.get('arm')==receipt['arm']
            and model.get('stage')==receipt['stage'] and model.get('seed')==receipt['seed'],
            'Private risk payload identity differs from completed receipt')
    train,valid=np.asarray(model['train'],int),np.asarray(model['valid'],int)
    expected=fixed_inner_splits(raw['folds'],raw['e'],outer=int(receipt['outer_fold']))[int(receipt['inner_partition'])]
    require(np.array_equal(train,expected[0]) and np.array_equal(valid,expected[1]),'Saved risk partition differs from original fixed inner split')
    if deterministic(receipt['arm']):
        require(receipt['epochs']==0 and 'risk' in model,'Convex final-risk contract mismatch')
        risk=np.asarray(model['risk'],float)
    else:
        require(receipt['epochs']==config['training']['epochs']==80 and model.get('epochs')==80,
                'Only fully completed fixed-80 neural risk can enter sensitivity analysis')
        require(80 in model.get('snapshots',{}),'Final epoch80 risk missing')
        risk=np.asarray(model['snapshots'][80]['risk'],float)
    require(risk.shape==valid.shape and np.isfinite(risk).all(),'Saved validation risk dimension/finite mismatch')
    return receipt,train,valid,risk


def summarize(rows,plan,config):
    output=[]
    for stage,arm in sorted(set((key[0],key[1]) for key in plan)):
        keys=[key for key in plan if key[:2]==(stage,arm)]
        available=[row for row in rows if (row['stage'],row['arm'],row['inner_partition'],row['seed']) in keys]
        complete=len(available)==len(keys)*len(GROUPS)
        for group in GROUPS:
            chosen=[row for row in available if row['group']==group]
            item=dict(stage=stage,arm=arm,group=group,status='COMPLETE' if complete else 'RUNNING',
                      completed_unique_cells=len(chosen),planned_unique_cells=len(keys),Harrell=None,Uno60=None)
            if complete:
                seeds=[31] if deterministic(arm) else config['prototype']['neural_seeds']
                values=[]
                for seed in seeds:
                    selected=[row for row in chosen if row['seed']==seed]
                    hn=sum(row['Harrell_num'] for row in selected);hd=sum(row['Harrell_den'] for row in selected)
                    un=sum(row['Uno_num'] for row in selected);ud=sum(row['Uno_den'] for row in selected)
                    values.append((hn/hd if hd else None,un/ud if ud else None))
                item['Harrell']=float(np.mean([v[0] for v in values])) if all(v[0] is not None for v in values) else None
                item['Uno60']=float(np.mean([v[1] for v in values])) if all(v[1] is not None for v in values) else None
            output.append(item)
    return output


def write_csv(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def generate(assets,private,output,config,stages):
    raw=load_raw(assets);groups=missingness_groups(raw)
    plan=planned_units(config,stages);planned=set(plan);rows=[];completed=set();hashes=[];lineages=set()
    for path in sorted((private/'cells').glob('*/COMPLETE.json')):
        check=json.loads(path.read_text())
        key=(check.get('stage'),check.get('arm'),check.get('inner_partition'),check.get('seed'))
        if key[0] not in stages:continue
        require(key in planned and key not in completed,'Unknown/duplicate completed unit in missingness analysis')
        receipt,train,valid,risk=load_final_cell(path,raw,config)
        lineages.add(hashlib.sha256(json.dumps(receipt['lineage'],sort_keys=True).encode()).hexdigest())
        values=stratified_metrics(raw['t'][train],raw['e'][train],raw['t'][valid],raw['e'][valid],risk,
                                  {name:mask[valid] for name,mask in groups.items()})
        for name in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den'):
            a,b=values['ALL'][name],receipt['metrics'][name]
            require((a is None and b is None) or (a is not None and b is not None and np.isclose(a,b,rtol=1e-12,atol=1e-9)),
                    'Independent full-validation replay disagrees with completed receipt')
        for group,value in values.items():
            rows.append(dict(stage=key[0],arm=key[1],inner_partition=key[2],seed=key[3],group=group,
                original_fit_n=len(train),original_fit_events=int(raw['e'][train].sum()),**value,
                IPCW_fit_scope='original_complete_fit_partition',prediction='same_saved_final_risk',
                subgroup_refit=False,epoch=receipt['epochs']))
        hashes.append(sha256(path));completed.add(key)
    require(len(lineages)<=1,'Mixed formal source lineages in missingness analysis')
    output.mkdir(parents=True,exist_ok=True)
    fields=['stage','arm','inner_partition','seed','group','original_fit_n','original_fit_events','n','events',
            'events_before60','Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den','pair_status',
            'IPCW_fit_scope','prediction','subgroup_refit','epoch']
    write_csv(output/'MISSINGNESS_SENSITIVITY.csv',rows,fields)
    summary=summarize(rows,plan,config)
    status='COMPLETE_REQUESTED_STAGES' if completed==planned else 'RUNNING'
    counts={stage:{'completed':sum(key[0]==stage for key in completed),'planned':sum(key[0]==stage for key in planned)} for stage in stages}
    audit=dict(schema='bgb_en_missingness_saved_risk_v1',status=status,stages=counts,
        completed_unique_cells=len(completed),planned_unique_cells=len(planned),
        canonical_group_counts={name:int(mask.sum()) for name,mask in groups.items()},
        source_lineage_sha256=next(iter(lineages),None),receipt_sha256=hashes,
        optimizer_fits=0,model_forward_calls=0,GPU_used=False,new_patient_predictions_generated=False,
        subgroup_model_refit=False,original_fit_censor_distribution_retained=True,
        independent_validation=False,platform_identity='unverified;RNA measurement-missingness proxy only')
    (output/'MISSINGNESS_SENSITIVITY_AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    content=['# BG-BEN：已有最终风险的缺失数据敏感性','',f'**状态：{status}；已完成 {len(completed)}/{len(planned)} 个请求单元。**','',
        '仅使用已有 COMPLETE checkpoint 的固定 epoch80 风险；凸模型使用原保存风险。没有模型前向、重新训练、组内选参或 GPU 运算。',
        '所有组沿用原完整拟合分区的删失分布估计 IPCW；按组只筛选评测患者，不重新拟合 censor model。',
        'RNA完整309、共享缺失区块203，CNA完整382、少量缺失130是同一开发队列的测量分组。RNA缺失模式不能称为已验证的真实测序平台。',
        '表中完整 arm 先在各 seed 汇总三个分区的折内一致性分子/分母，再对 seed 求均值。确定性模型不重复计算相同 seed。未完成 arm 不汇总性能；无有效可比对为 NA。',
        '不同组的事件数、病例构成与可比对数量不同；组间 C-index 差不能独占解释为缺失值造成的损伤，也不是新的独立验证。','',
        '|阶段|模型|分组|状态|Harrell|Uno60|','|---|---|---|---|---:|---:|']
    number=lambda value:'NA' if value is None else f'{value:.6f}'
    for row in summary:
        content.append(f"|{row['stage']}|{row['arm']}|{row['group']}|{row['status']}|{number(row['Harrell'])}|{number(row['Uno60'])}|")
    content+=['','逐已完成 cell 的匿名人数、事件数、分子分母和 NA 原因见同名 CSV。没有输出患者标识、分组索引或患者风险向量。']
    (output/'MISSINGNESS_SENSITIVITY.md').write_text('\n'.join(content)+'\n',encoding='utf-8')
    return audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',required=True,type=Path);parser.add_argument('--private',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json')
    parser.add_argument('--stage',choices=('R1','R2','R3','all'),default='all')
    args=parser.parse_args();torch.set_num_threads(1)
    stages=['R1','R2','R3'] if args.stage=='all' else [args.stage]
    result=generate(json.loads(args.assets.read_text()),args.private,args.output,json.loads(args.config.read_text()),stages)
    print(json.dumps({key:result[key] for key in ('status','completed_unique_cells','planned_unique_cells','stages','optimizer_fits','GPU_used')}),flush=True)


if __name__=='__main__':main()

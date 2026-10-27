"""Anonymous locked outer reports; aggregate pair counts before seed means."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.bgb_en.data import sha256
from src.bgb_en.survival import combine_metrics


def dump_csv(path,rows,columns):
    with path.open('w',newline='',encoding='utf-8') as handle:
        out=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore');out.writeheader();out.writerows(rows)


def aggregate(rows,model,endpoint):
    selected=[row for row in rows if row['model']==model and row['endpoint']==endpoint]
    identities=[(row['fold'],row['seed']) for row in selected]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate model endpoint fold seed')
    seeds={}
    for seed in (31,47,83):
        subset=[row for row in selected if row['seed']==seed]
        if {row['fold'] for row in subset}!={1,2,3,4,5}:continue
        seeds[seed]=combine_metrics(subset)
    if len(seeds)!=3:return {'status':'NOT_ALL_FIVE_FOLDS_THREE_SEEDS','seeds':seeds}
    return dict(status='COMPLETE_FIVE_FOLD_THREE_SEED',seeds=seeds,
        Harrell=float(np.mean([x['Harrell'] for x in seeds.values()])),
        Uno60=float(np.mean([x['Uno60'] for x in seeds.values()])),
        worst_fold_Harrell=min(row['Harrell'] for row in selected),
        worst_fold_Uno60=min(row['Uno60'] for row in selected))


def comparisons(rows,candidate,reference,endpoint):
    columns=('fold','seed')
    for model in (candidate,reference):
        identities=[(row['fold'],row['seed']) for row in rows if row['model']==model and row['endpoint']==endpoint]
        if len(set(identities))!=len(identities):raise ValueError('Duplicate model endpoint fold seed')
    cand={tuple(row[k] for k in columns):row for row in rows if row['model']==candidate and row['endpoint']==endpoint}
    ref={tuple(row[k] for k in columns):row for row in rows if row['model']==reference and row['endpoint']==endpoint}
    pairs=[]
    for key in sorted(cand.keys()&ref.keys()):
        a,b=cand[key],ref[key]
        if abs(a['Harrell_den']-b['Harrell_den'])>1e-9 or abs(a['Uno_den']-b['Uno_den'])>1e-8:
            raise ValueError('Different populations or IPCW definitions in outer comparison')
        extra=a.get('_reference_metrics',{}).get(reference,a)
        row=dict(candidate=candidate,reference=reference,endpoint=endpoint,fold=key[0],seed=key[1],
            paired_delta_H=a['Harrell']-b['Harrell'],paired_delta_U60=a['Uno60']-b['Uno60'],
            status='FOLD_SEED_POINT_ESTIMATE',risk_shift_sd_ratio=extra.get('risk_shift_sd_ratio'),
            repair=extra.get('repair'),damage=extra.get('damage'),Uno_repair=extra.get('Uno_repair'),Uno_damage=extra.get('Uno_damage'))
        if reference!='CP_STRONG' and reference not in a.get('_reference_metrics',{}):
            # These are relative to CP in baseline receipts. Do not label
            # them as comparisons against some other risk vector.
            for name in ('risk_shift_sd_ratio','repair','damage','Uno_repair','Uno_damage'):row[name]=None
        pairs.append(row)
    for seed in (31,47,83):
        keys=[(fold,seed) for fold in range(1,6)]
        if not all(key in cand and key in ref for key in keys):continue
        a=combine_metrics([cand[key] for key in keys]);b=combine_metrics([ref[key] for key in keys])
        denh=a['Harrell_den'];denu=a['Uno_den']
        pair=dict(candidate=candidate,reference=reference,endpoint=endpoint,fold='WITHIN_FOLD_COUNTS',seed=seed,
            paired_delta_H=a['Harrell']-b['Harrell'],paired_delta_U60=a['Uno60']-b['Uno60'],status='SEED_AGGREGATE')
        details=[cand[key].get('_reference_metrics',{}).get(reference,cand[key]) for key in keys]
        if (reference=='CP_STRONG' or all(reference in cand[key].get('_reference_metrics',{}) for key in keys)) and all(row.get('repair') is not None for row in details):
            for name in ('repair','damage'):
                pair[name]=sum(row[name]*cand[key]['Harrell_den'] for row,key in zip(details,keys))/denh
            for name in ('Uno_repair','Uno_damage'):
                pair[name]=sum(row[name]*cand[key]['Uno_den'] for row,key in zip(details,keys))/denu
        pairs.append(pair)
    return pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path)
    p.add_argument('--selection-lock',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    lock_sha=sha256(args.selection_lock);lock=json.loads(args.selection_lock.read_text());rows=[];cost=[];artifacts=[]
    for path in sorted((args.private/'outer').glob('fold*/COMPLETE.json')):
        value=json.loads(path.read_text());model_path=path.parent/'models.pt'
        if value['status']!='COMPLETE_OUTER_BASELINES' or value['contract']['selection_lock_sha256']!=lock_sha:
            raise ValueError('Outer receipt is not bound to this finalized selection lock')
        if sha256(model_path)!=value['artifact_sha256']:raise ValueError('Outer artifact changed')
        fold=value['fold']
        if fold not in range(1,6) or set(value['results'])!={'31','47','83'}:raise ValueError('Invalid baseline fold or seed scope')
        for seed,result in value['results'].items():
            for model,metric in result['models'].items():
                for endpoint,m in [('DSS',metric),('PFI',metric['PFI_frozen_DSS_risk'])]:
                    row=dict(fold=fold,seed=int(seed),model=model,endpoint=endpoint,beta=result['beta'],
                        seed_is_deterministic_reference=(model=='CP_STRONG' and int(seed)!=31),**{k:v for k,v in m.items() if k!='PFI_frozen_DSS_risk'})
                    rows.append(row)
        cost.append(dict(kind='BASELINES',fold=fold,actual_Cox_fits=value['actual_Cox_fits'],actual_scalar_fits=value['actual_scalar_calibration_fits'],actual_neural_fits=0,
            seconds=value['elapsed_seconds'],unique_full_CP_test_vectors=value['deterministic_CP_test_vector_fits'],CP_seed_references=value['CP_seed_references'],
            checkpoint_sha256=value['artifact_sha256']))
        artifacts.append(dict(fold=fold,receipt_sha256=sha256(path),checkpoint_sha256=value['artifact_sha256']))
    baseline_folds=len(cost)
    if len({row['fold'] for row in cost})!=len(cost):raise ValueError('Duplicate outer-fold receipts')
    for path in sorted((args.private/'neural').glob('*/COMPLETE.json')):
        value=json.loads(path.read_text());artifact=path.parent/'model.pt'
        if value['status']!='COMPLETE_OUTER_NEURAL' or value['contract']['selection_lock_sha256']!=lock_sha or value['artifact_sha256']!=sha256(artifact):
            raise ValueError('Outer neural contract changed')
        if value['outer_test_read_for_selection'] is not False:raise ValueError('Outer test was used for selection')
        if (value['fold'] not in range(1,6) or value['seed'] not in (31,47,83) or value['epochs']!=80
            or value['selected_arm'] not in ('F2','F3','F5','F10')
            or value['contract']['seed']!=value['seed'] or value['contract']['arm']!=value['selected_arm']):
            raise ValueError('Outer neural fold, seed, arm or epoch identity mismatch')
        fold_lock=args.private/'selection'/f"outer{value['fold']}.json"
        if sha256(fold_lock)!=value['contract']['fold_lock_sha256']:raise ValueError('Per-outer selection changed after final fitting')
        selected=json.loads(fold_lock.read_text())
        if selected['outer_fold']!=value['fold'] or selected['selected_arm']!=value['selected_arm']:raise ValueError('Wrong per-outer selection identity')
        for endpoint,m in [('DSS',value['metrics']['CP_STRONG']),('PFI',value['PFI_frozen_DSS_risk'])]:
            rows.append(dict(fold=value['fold'],seed=value['seed'],model='NESTED_INTERACTION',endpoint=endpoint,
                selected_arm=value['selected_arm'],**m,_reference_metrics=value['metrics'] if endpoint=='DSS' else {}))
        cost.append(dict(kind='NESTED_NEURAL',fold=value['fold'],seed=value['seed'],actual_Cox_fits=0,actual_scalar_fits=0,
            actual_neural_fits=1,parameters=value['parameters'],peak_GPU_bytes=value['peak_GPU_bytes'],
            seconds=value['elapsed_seconds'],checkpoint_sha256=value['artifact_sha256']))
        artifacts.append(dict(kind='NESTED_NEURAL',fold=value['fold'],seed=value['seed'],
            receipt_sha256=sha256(path),checkpoint_sha256=value['artifact_sha256'],fold_lock_sha256=sha256(fold_lock)))
    models=lock['finalists'];summary={model:{endpoint:aggregate(rows,model,endpoint) for endpoint in ('DSS','PFI')} for model in models}
    pairs=[('CP_SCALAR','CP_STRONG')]
    if 'NESTED_INTERACTION' in models:pairs+=[('NESTED_INTERACTION','CP_STRONG'),('NESTED_INTERACTION','CP_SCALAR')]
    paired=[row for endpoint in ('DSS','PFI') for a,b in pairs for row in comparisons(rows,a,b,endpoint)]
    metrics_columns=['model','endpoint','fold','seed','Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den',
        'beta','seed_is_deterministic_reference','selected_arm','repair','damage','Uno_repair','Uno_damage','risk_shift_sd_ratio']
    dump_csv(output/'R4_FOLD_SEED_METRICS.csv',rows,metrics_columns)
    dump_csv(output/'R4_PAIRED_COMPARISONS.csv',paired,['candidate','reference','endpoint','fold','seed','paired_delta_H','paired_delta_U60',
        'repair','damage','Uno_repair','Uno_damage','risk_shift_sd_ratio','status'])
    dump_csv(output/'R4_COMPUTE_COST.csv',cost,['kind','fold','seed','actual_Cox_fits','actual_scalar_fits','actual_neural_fits','parameters','peak_GPU_bytes','seconds','unique_full_CP_test_vectors','CP_seed_references','checkpoint_sha256'])
    complete=all(summary[model]['DSS']['status']=='COMPLETE_FIVE_FOLD_THREE_SEED' for model in models)
    lines=['# R4 锁定外层评测','',f"状态：{'COMPLETE_LOCKED_FINALISTS' if complete else 'RUNNING'}；简单基线已完成 {baseline_folds}/5 外折。",'',
        '每 seed 分别汇总外折内 concordant / eligible，再平均三个 seed；不跨外折比较风险。CP 强锚点的确定性向量跨 seed 引用，不当成三次独立训练。PFI 仅评估冻结的 DSS 风险。',
        '', '| 模型 | 终点 | Harrell | Uno60 | 最差折 H | 最差折 Uno60 |','|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'' if x is None else f'{x:.6f}'
    for model in models:
        for endpoint in ('DSS','PFI'):
            m=summary[model][endpoint]
            lines.append(f"| {model} | {endpoint} | {fmt(m.get('Harrell'))} | {fmt(m.get('Uno60'))} | {fmt(m.get('worst_fold_Harrell'))} | {fmt(m.get('worst_fold_Uno60'))} |")
    lines+=['','## 同患者配对差值','']
    for endpoint in ('DSS','PFI'):
        for candidate,reference in pairs:
            comparison_rows=[row for row in paired if row['endpoint']==endpoint and row['candidate']==candidate and row['reference']==reference]
            seed_rows=[row for row in comparison_rows if row['status']=='SEED_AGGREGATE']
            if len(seed_rows)!=3:continue
            dh=np.mean([row['paired_delta_H'] for row in seed_rows]);du=np.mean([row['paired_delta_U60'] for row in seed_rows])
            per_fold=[row for row in comparison_rows if row['status']=='FOLD_SEED_POINT_ESTIMATE']
            lines.append(f"- {endpoint}：{candidate}−{reference} ΔH={dh:+.6f}，ΔUno60={du:+.6f}；正向 seed：H {sum(row['paired_delta_H']>0 for row in seed_rows)}/3、Uno60 {sum(row['paired_delta_U60']>0 for row in seed_rows)}/3；正向 fold×seed：H {sum(row['paired_delta_H']>0 for row in per_fold)}/15、Uno60 {sum(row['paired_delta_U60']>0 for row in per_fold)}/15。")
    lines+=['','## 解释边界','',
        '- 此报告仅包含选择锁允许的简单最终候选；未完成的神经候选不能以此文件表示完成。',
        '- 多次历史开发使用了同一 TCGA 队列；本轮外层拟合隔离不等于新的外部确认队列。',
        '- 不生成置信区间、bootstrap、p 值或显著性结论。正向点估计不自动证明医学机制。',
        '- 风险 SD 比例按折保留，不把不同折任意尺度的风险直接拼接后算总体 SD。','']
    (output/'R4_FINAL_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    audit=dict(schema='bgb_en_outer_report_v1',status='COMPLETE_LOCKED_FINALISTS' if complete else 'RUNNING',
        selection_lock_sha256=lock_sha,completed_outer_folds=baseline_folds,expected_outer_folds=5,summary=summary,
        actual_Cox_fits=sum(row['actual_Cox_fits'] for row in cost),actual_scalar_fits=sum(row['actual_scalar_fits'] for row in cost),
        actual_neural_fits=sum(row['actual_neural_fits'] for row in cost),
        pairing_is_same_fold_same_seed=True,cross_fold_risk_comparisons=False,outer_sources=artifacts,
        report_script_sha256=sha256(__file__),patient_data_exported=False)
    (output/'R4_REPORT_AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'],completed_outer_folds=baseline_folds)))


if __name__=='__main__':main()

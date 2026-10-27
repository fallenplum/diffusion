"""Read-only R1 patch-budget diagnostics from locked final80 artifacts.

No fit, model forward, optimizer or GPU is used. Original patient risks stay
private: only within-cell distribution summaries and fold/seed metrics leave.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.bgb_en_report_aux import canonical_hash,formatted,mean,require,sha256,write_csv
from scripts.bgb_en_report_inner import aggregate,validate_metrics

SCENARIOS=('CONCH_B512','CONCH_B128','CONCH_B1024','UNI2_B128')
ARMS=('P1','P3')
PAIR_SPECS=tuple((f'{case}_{arm}_minus_CONCH_B512_{arm}',(case,arm),('CONCH_B512',arm),
                 'CONCH same native grid, changed prelocked selected-patch budget')
                for case in ('CONCH_B128','CONCH_B1024') for arm in ARMS)+tuple(
                (f'UNI2_B128_{arm}_minus_CONCH_B128_{arm}',('UNI2_B128',arm),('CONCH_B128',arm),
                 'joint encoder, native grid and physical patch footprint change; not pure encoder') for arm in ARMS)+tuple(
                (f'{case}_P3_minus_P1',(case,'P3'),(case,'P1'),
                 'same selected patches and backbone inputs; spatial encoder architecture differs') for case in SCENARIOS)


def risk_summary(risk,delta):
    risk,delta=np.asarray(risk,float),np.asarray(delta,float)
    require(risk.ndim==1 and len(risk)>0 and delta.shape==risk.shape
            and np.isfinite(risk).all() and np.isfinite(delta).all(),'Saved final risk/delta is invalid')
    output={}
    for name,array in (('full',risk),('anchor',risk-delta),('delta',delta)):
        output.update({f'{name}_mean':float(array.mean()),f'{name}_SD':float(array.std()),
                       f'{name}_max_abs':float(np.max(np.abs(array)))})
        output.update({f'{name}_q{int(q*100):02d}':float(np.quantile(array,q)) for q in (.05,.25,.5,.75,.95)})
    output['delta_SD_over_anchor_SD']=output['delta_SD']/output['anchor_SD'] if output['anchor_SD']>0 else None
    return output


def verify_cell(path,scenario,arm,part,seed,lock,lock_sha):
    receipt=json.loads(path.read_text())
    require(receipt.get('status')=='COMPLETE' and receipt.get('stage')=='R1'
            and receipt.get('arm')==arm and receipt.get('inner_partition')==part
            and receipt.get('seed')==seed and receipt.get('outer_fold')==1
            and receipt.get('outer_test_evaluated') is False,'Unexpected pathology-sensitivity cell identity')
    require(receipt.get('epochs')==80 and receipt.get('actual_optimizer_fits')==1,'Wrong final fixed-budget receipt')
    origin=lock['shared_teacher_origin_lineage'];lineage=receipt['lineage']
    if scenario=='CONCH_B512':
        require(lineage==origin,'Primary CONCH512 reference source differs')
    else:
        added=lineage.get('path_budget_sensitivity',{})
        require({k:v for k,v in lineage.items() if k!='path_budget_sensitivity'}==origin,
                'Sensitivity relabelled the original teacher source')
        require(added.get('scenario')==scenario and added.get('base_arm')==arm
                and added.get('lock_sha256')==lock_sha and added.get('shared_teacher_origin_lineage')==origin
                and added.get('sources')==lock['sensitivity_sources']
                and added.get('selected_cache_READY_sha256')==lock['cases'][scenario]['READY_sha256'],
                'Sensitivity artifact does not match prefit budget/source lock')
    validate_metrics(receipt['metrics'])
    final=[row for row in receipt.get('curves',[]) if row.get('epoch')==80]
    require(len(final)==1 and final[0].get('valid')==receipt['metrics'],'Receipt selected a nonfinal training snapshot')
    model_path=path.parent/'model.pt'
    require(model_path.is_file() and sha256(model_path)==receipt['artifact_sha256'],'Pathology checkpoint byte SHA mismatch')
    return receipt,model_path


def checkpoint_summaries(path,receipt):
    import torch
    saved=torch.load(path,map_location='cpu',weights_only=False)
    require(saved.get('lineage')==receipt['lineage'] and saved.get('stage')=='R1'
            and saved.get('arm')==receipt['arm'] and saved.get('seed')==receipt['seed']
            and saved.get('epochs')==80 and saved.get('optimizer_updates')==80
            and saved.get('full_riskset') is True and saved.get('gradient_cache') is True,
            'Saved checkpoint identity or exact risk-set schedule differs')
    require(saved.get('curves')==receipt.get('curves'),'Private learning curves differ from completed receipt')
    snapshots=saved.get('snapshots',{});require(80 in snapshots,'Final80 risk is absent')
    final=snapshots[80];require(final.get('metrics')==receipt['metrics'],'Saved final80 metrics differ')
    require(len(np.asarray(final['risk']))==receipt['valid_n'],'Saved final80 evaluation population differs')
    return risk_summary(final['risk'],final['delta'])


def load_cells(primary,sensitivity,lock,lock_sha):
    cells={};curves=[];risks=[]
    for scenario in SCENARIOS:
        parent=Path(primary) if scenario=='CONCH_B512' else Path(sensitivity)/'survival'/scenario
        for arm in ARMS:
            for part in (0,1,2):
                for seed in (31,47):
                    path=parent/'cells'/f'R1_outer1_inner{part}_{arm}_seed{seed}'/'COMPLETE.json'
                    if not path.is_file():continue
                    receipt,model_path=verify_cell(path,scenario,arm,part,seed,lock,lock_sha)
                    row=dict(scenario=scenario,arm=arm,partition=part,seed=seed,
                        train_n=receipt['train_n'],valid_n=receipt['valid_n'],parameters=receipt['parameters'],
                        elapsed_seconds=receipt['elapsed_seconds'],peak_GPU_bytes=receipt['peak_GPU_bytes'],
                        receipt_sha256=sha256(path),artifact_sha256=receipt['artifact_sha256'],**receipt['metrics'])
                    cells[(scenario,arm,part,seed)]=row
                    summary=checkpoint_summaries(model_path,receipt)
                    risks.append(dict(scenario=scenario,arm=arm,partition=part,seed=seed,
                        evaluated_rows=receipt['valid_n'],**summary,
                        training_delta_SD_final=next(r['delta_train_SD'] for r in receipt['curves'] if r['epoch']==80)))
                    for point in receipt['curves']:
                        require(point['epoch'] in (20,40,80),'Unexpected learning-curve evaluation time')
                        curve=dict(scenario=scenario,arm=arm,partition=part,seed=seed,epoch=point['epoch'])
                        for name in ('cox_loss','regularizer','gradient_norm','replay_max',
                                     'delta_train_SD','delta_valid_SD','gradient_tensor_nonzero_ratio'):
                            if name in point:curve[name]=point[name]
                        for split in ('train','valid'):
                            validate_metrics(point[split])
                            for key in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den'):
                                curve[split+'_'+key]=point[split][key]
                        curves.append(curve)
    return cells,curves,risks


def summarize(cells):
    seed_rows=[];summaries=[];pairs=[]
    for scenario in SCENARIOS:
        for arm in ARMS:
            complete=all((scenario,arm,part,seed) in cells for part in (0,1,2) for seed in (31,47))
            available=[r for (s,a,p,seed),r in cells.items() if s==scenario and a==arm]
            for seed in (31,47):
                rows=[cells[(scenario,arm,part,seed)] for part in (0,1,2) if (scenario,arm,part,seed) in cells]
                value=aggregate(rows) if len(rows)==3 else {key:None for key in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den')}
                seed_rows.append(dict(scenario=scenario,arm=arm,seed=seed,status='COMPLETE' if len(rows)==3 else 'RUNNING',
                    completed_partitions=len(rows),planned_partitions=3,**value))
            seeds=[r for r in seed_rows if r['scenario']==scenario and r['arm']==arm]
            summaries.append(dict(scenario=scenario,arm=arm,status='COMPLETE' if complete else 'RUNNING',
                completed_cells=len(available),planned_cells=6,
                Harrell=mean([r['Harrell'] for r in seeds]) if complete else None,
                Uno60=mean([r['Uno60'] for r in seeds]) if complete else None,
                parameters_mean=mean([r['parameters'] for r in available]),
                elapsed_seconds_sum=sum(r['elapsed_seconds'] for r in available)))
    for name,left,right,boundary in PAIR_SPECS:
        counts=[]
        for seed in (31,47):
            ll=[];rr=[]
            for part in (0,1,2):
                l=cells.get((*left,part,seed));r=cells.get((*right,part,seed))
                if l is not None and r is not None:
                    require(l['train_n']==r['train_n'] and l['valid_n']==r['valid_n']
                        and all(np.isclose(l[k],r[k],rtol=1e-12,atol=1e-9) for k in ('Harrell_den','Uno_den')),
                        'Paired budgets use different evaluation populations or IPCW references')
                    ll.append(l);rr.append(r)
            counts.append(len(ll));lval=aggregate(ll) if len(ll)==3 else {};rval=aggregate(rr) if len(rr)==3 else {}
            pairs.append(dict(comparison=name,seed=seed,status='COMPLETE' if len(ll)==3 else 'RUNNING',
                completed_pairs=len(ll),planned_pairs=3,
                delta_Harrell=lval['Harrell']-rval['Harrell'] if len(ll)==3 and lval['Harrell'] is not None and rval['Harrell'] is not None else None,
                delta_Uno60=lval['Uno60']-rval['Uno60'] if len(ll)==3 and lval['Uno60'] is not None and rval['Uno60'] is not None else None,
                interpretation_boundary=boundary))
        chosen=[r for r in pairs if r['comparison']==name]
        pairs.append(dict(comparison=name,seed='MEAN',status='COMPLETE' if counts==[3,3] else 'RUNNING',
            completed_pairs=sum(counts),planned_pairs=6,
            delta_Harrell=mean([r['delta_Harrell'] for r in chosen]) if counts==[3,3] else None,
            delta_Uno60=mean([r['delta_Uno60'] for r in chosen]) if counts==[3,3] else None,
            interpretation_boundary=boundary))
    return summaries,seed_rows,pairs


def write_reports(output,cells,curves,risks,lock_sha):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary,seeds,pairs=summarize(cells)
    completed_new=sum(key[0]!='CONCH_B512' for key in cells)
    primary=sum(key[0]=='CONCH_B512' for key in cells)
    result=dict(schema='bgb_en_path_budget_saved_risk_report_v1',
        status='COMPLETE_LOCKED_BUDGET_SENSITIVITY' if completed_new==36 and primary==12 else 'RUNNING',
        completed_new_survival_fits=completed_new,planned_new_survival_fits=36,
        verified_primary_reference_cells=primary,planned_primary_reference_cells=12,
        summary=summary,seed_metrics=seeds,paired_comparisons=pairs,
        prefit_sensitivity_lock_sha256=lock_sha,report_script_sha256=sha256(__file__),
        risk_scope='within-cell final validation risk, delta and risk-minus-delta anchor; only training delta SD is saved',
        cross_fold_risk_values_pooled=False,unseen_outer_evaluation=False,
        model_forward_calls=0,optimizer_updates=0,GPU_used=False,
        new_independent_validation=False,patient_level_data_exported=False)
    (output/'PATH_BUDGET_SENSITIVITY_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(output/'PATH_BUDGET_SENSITIVITY_FOLD_METRICS.csv',list(cells.values()),
              ('scenario','arm','partition','seed','Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den'))
    write_csv(output/'PATH_BUDGET_SENSITIVITY_SEED_METRICS.csv',seeds)
    write_csv(output/'PATH_BUDGET_SENSITIVITY_PAIRED.csv',pairs)
    write_csv(output/'PATH_BUDGET_SENSITIVITY_LEARNING_CURVES.csv',curves,
              ('scenario','arm','partition','seed','epoch','train_Harrell','valid_Harrell'))
    write_csv(output/'PATH_BUDGET_SENSITIVITY_RISK_DISTRIBUTION.csv',risks,
              ('scenario','arm','partition','seed','evaluated_rows','full_SD','anchor_SD','delta_SD'))
    lines=['# R1 Patch 预算与基础表征敏感性','',
           f"**状态：{result['status']}；新增预算拟合 {completed_new}/36，原512预算参照 {primary}/12。**",'',
           '| 场景 | 模型 | 完成 | Harrell | Uno60 |','|---|---|---:|---:|---:|']
    for row in summary:lines.append(f"| {row['scenario']} | {row['arm']} | {row['completed_cells']}/6 | {formatted(row['Harrell'])} | {formatted(row['Uno60'])} |")
    lines+=['','## 同 seed 直接配对','','| 比较 | 配对完成 | ΔHarrell | ΔUno60 |','|---|---:|---:|---:|']
    for row in pairs:
        if row['seed']=='MEAN':lines.append(f"| {row['comparison']} | {row['completed_pairs']}/6 | {formatted(row['delta_Harrell'])} | {formatted(row['delta_Uno60'])} |")
    lines+=['','三个内层分区先按 seed 汇总折内 concordance 分子与分母，再求两 seed 均值；没有跨折风险比较。未完整完成的 arm 和比较保留 NA。最终性能固定为 epoch80；20/40轮只作学习曲线诊断，不用于选预算或选检查点。',
            '', '## 风险分布与范围','',
            '同名 RISK_DISTRIBUTION.csv 保留每个匿名拟合单元的最终验证 full risk、delta、full−delta 锚点的均值、SD、分位数及最大绝对值。训练端仅报告已有曲线保存的 delta SD，未生成或推断未保存的训练 full-risk 分布。不同折的风险尺度不直接拼接。',
            '', 'R1 的 offset 是原共同 Clinical+S32 风险；本阶段比较病理分支和预算，不将其改称独立 RNA 增量或完整三模态机制。',
            '', '## 优势与风险','',
            '优势：128/512/1024 同一CONCH来源的预定预算比较，P1/P3保持原训练、教师、患者分区和真实几何规则，所有产物有锁与SHA绑定。P0全量均值与预算无关，仅引用既有P0，不重复拟合。',
            '', '风险：UNI2-128 与 CONCH-128 同时改变基础编码器、原生网格与patch物理尺寸；不能将其差值解释为纯编码器作用或纯空间监督作用。候选不足会使实际1024预算未满。结果属于同一历史开发队列内的敏感性分析，不能称作独立确认。',
            '', '读取所有模型时强制CPU映射，只消费已保存的风险与标量曲线，没有模型前向、优化器更新或GPU调用。患者标识、索引、风险向量及权重不公开。']
    (output/'PATH_BUDGET_SENSITIVITY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-private',required=True,type=Path)
    parser.add_argument('--sensitivity-private',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en')
    args=parser.parse_args();lock_path=args.sensitivity_private/'PATH_SENSITIVITY_LOCK.json'
    lock=json.loads(lock_path.read_text());lock_sha=sha256(lock_path)
    require(lock.get('status')=='LOCKED_BEFORE_SENSITIVITY_SURVIVAL_FITS'
            and lock.get('planned_new_neural_fits')==36,'Expected fixed prefit sensitivity lock')
    cells,curves,risks=load_cells(args.primary_private,args.sensitivity_private,lock,lock_sha)
    result=write_reports(args.output,cells,curves,risks,lock_sha)
    print(json.dumps({k:result[k] for k in ('status','completed_new_survival_fits','verified_primary_reference_cells','optimizer_updates','GPU_used')}))


if __name__=='__main__':main()

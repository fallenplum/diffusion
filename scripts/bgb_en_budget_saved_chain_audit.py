"""Locked, post hoc, zero-fit audit of all24 physically saved budget models.

The original23 COMPLETE and one FAILED remain unchanged. Pairing each new fit's
own saved epoch80 with its own final epoch is diagnostic, not a replacement for
the original comparable-parent80 qualification or an independent performance
confirmation. Only the failed model receives one CPU actual-weight replay.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bgb_en_recovery_audit import require,sha256
from scripts.bgb_en_budget_retrain import compare_parent80
from scripts.bgb_en_summarize_budget import independent_metrics,aggregate_metrics
from src.bgb_en.omics import load_gmt_membership
from src.bgb_en.training import MechanismModel,predict

AUDIT_SOURCES=('scripts/bgb_en_budget_saved_chain_audit.py','scripts/bgb_en_summarize_budget.py',
    'scripts/bgb_en_budget_retrain.py','scripts/bgb_en_anchor_cast_audit.py','configs/bgb_en/budget_sensitivity_v1.json',
    'src/phase10_trimodal/math.py')
FAILED_UNIT='R2_newfit160_outer1_inner0_O4_seed47'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def specs():
    return [(arm,part,seed,epochs) for arm in ('O3','O4') for part in (0,1,2) for seed in (31,47) for epochs in (160,320)]


def label(arm,part,seed,epochs):return f'R2_newfit{epochs}_outer1_inner{part}_{arm}_seed{seed}'


def source_guard(lineage):
    require(lineage['protocol_sha256']==sha256(ROOT/'configs/bgb_en/prototype_v1.json'),'CORE_PROTOCOL_CHANGED')
    for name,digest in lineage['source'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'UNSAFE_SOURCE_PATH')
        require(sha256(ROOT/relative)==digest,'CORE_SOURCE_CHANGED')


def make_lock(args):
    assets=read(args.assets);rows=[];preps=[];parents=[];origin=None
    for part in (0,1,2):
        receipt_path=args.primary_private/'prepared'/f'outer1_inner{part}.json';receipt=read(receipt_path)
        require(receipt['status']=='COMPLETE' and receipt['sha256']==sha256(receipt_path.with_suffix('.pt')),'PREPARED_HASH_CHANGED')
        if origin is None:origin=receipt['lineage'];source_guard(origin)
        require(receipt['lineage']==origin,'PREPARED_ORIGINS_DIFFER')
        preps.append(dict(partition=part,artifact_sha256=receipt['sha256'],receipt_sha256=sha256(receipt_path)))
        for arm in ('O3','O4'):
            for seed in (31,47):
                folder=args.primary_private/'cells'/f'R2_outer1_inner{part}_{arm}_seed{seed}';parent=read(folder/'COMPLETE.json')
                require(parent['status']=='COMPLETE' and parent['lineage']==origin
                        and parent['artifact_sha256']==sha256(folder/'model.pt'),'PARENT_ARTIFACT_OR_ORIGIN_CHANGED')
                parents.append(dict(arm=arm,partition=part,seed=seed,artifact_sha256=parent['artifact_sha256'],
                    receipt_sha256=sha256(folder/'COMPLETE.json')))
    require(sha256(assets['endpoints'])==origin['endpoints_sha256'] and sha256(assets['cohort'])==origin['cohort_sha256']
            and sha256(assets['gmt'])==origin['GMT_sha256'],'AUDIT_INPUT_SHA_CHANGED')
    for arm,part,seed,epochs in specs():
        name=label(arm,part,seed,epochs);folder=args.private/'cells'/name
        start_path=folder/'START.json';start=read(start_path)
        marker_path=folder/('FAILED.json' if name==FAILED_UNIT else 'COMPLETE.json');marker=read(marker_path)
        require(start['lineage']==marker['lineage'] and marker['lineage']['core_origin']==origin,'MARKER_START_ORIGIN_CHANGED')
        require(marker['lineage']['independent_script_sha256']==sha256(ROOT/'scripts/bgb_en_budget_retrain.py')
                and marker['lineage']['budget_lock_sha256']==sha256(ROOT/'configs/bgb_en/budget_sensitivity_v1.json'),
                'BUDGET_ORIGINAL_SOURCE_CHANGED')
        require((marker['lineage']['arm'],marker['lineage']['inner_partition'],marker['lineage']['seed'],marker['lineage']['epochs'])
                ==(arm,part,seed,epochs),'BUDGET_MARKER_IDENTITY_CHANGED')
        artifact=folder/'model.pt';digest=sha256(artifact)
        if name==FAILED_UNIT:
            require(marker['exception_type']=='ValueError' and marker['actual_optimizer_updates_completed']==160
                    and not (folder/'COMPLETE.json').exists(),'ORIGINAL_FAILED_STATUS_CHANGED')
            role='FAILED_PARENT80_NUMERICAL_GUARD'
        else:
            require(marker['status']=='COMPLETE_INDEPENDENT_BUDGET_FIT' and marker['artifact_sha256']==digest
                    and marker['optimizer_updates']==epochs and not (folder/'FAILED.json').exists(),
                    'ORIGINAL_ACCEPTED_STATUS_OR_ARTIFACT_CHANGED')
            role='ORIGINAL_COMPLETE'
        rows.append(dict(unit=name,arm=arm,partition=part,seed=seed,epochs=epochs,original_role=role,
            artifact_sha256=digest,START_sha256=sha256(start_path),marker_sha256=sha256(marker_path)))
    return dict(schema='bgb_en_posthoc_saved_budget_chain_audit_lock_v1',status='LOCKED_BEFORE_ZERO_FIT_DIAGNOSTIC_EXECUTION',
        audit_sources={name:sha256(ROOT/name) for name in AUDIT_SOURCES},core_source=origin['source'],
        assets_configuration_sha256=sha256(args.assets),endpoint_sha256=origin['endpoints_sha256'],
        cohort_sha256=origin['cohort_sha256'],GMT_sha256=origin['GMT_sha256'],prepared=preps,parent80_artifacts=parents,models=rows,
        intended_source_models=24,original_COMPLETE_models=23,original_FAILED_models=1,
        comparison='within each new fit own saved epoch80 to its own final160/320; all original-parent80 to new-own80 also retained',
        actual_weight_forward_scope='one failed O4/p0/seed47 final160 CPU replay, validation batch8, original2e-6 guard retained',
        no_best_budget_selection=True,not_pre_registered_performance_confirmation=True,
        original_receipts_trainers_and_thresholds_modified=False,new_fits=0,new_decoders=0,GPU_calls=0,
        R4_selection_or_budget_changed=False)


def with_delta(metric):
    return {**metric,'delta_H_vs_own_offset':metric['repair']-metric['damage'],
        'delta_Uno60_vs_own_offset':metric['Uno_repair']-metric['Uno_damage']}


def aggregate_cells(rows):
    seeds=[];means=[]
    for arm in ('O3','O4'):
        for epochs in (160,320):
            chosen=[row for row in rows if row['arm']==arm and row['epochs']==epochs]
            require(len(chosen)==6,'FULL_SAVED_DIAGNOSTIC_HAS_MISSING_CELL')
            for seed in (31,47):
                group=[row for row in chosen if row['seed']==seed];require(len(group)==3,'SEED_MISSING_PARTITION')
                item=dict(arm=arm,epochs=epochs,seed=seed,partitions=3,
                    original_COMPLETE_cells=sum(row['original_role']=='ORIGINAL_COMPLETE' for row in group),
                    original_FAILED_cells=sum(row['original_role']!='ORIGINAL_COMPLETE' for row in group))
                for field in ('own80','final_vs_own80','final_vs_offset','train80','train_final'):
                    item[field]=aggregate_metrics([row[field] for row in group])
                for field in ('complete_tokens','pooled32'):
                    values=[row['final_decoder'][field]['R2'] for row in group]
                    item[field+'_decoder_R2']=float(np.mean(values)) if all(value is not None for value in values) else None
                item['mean_training_Cox_change']=float(np.mean([row['training_Cox_final_minus_own80'] for row in group]))
                item['mean_final_delta_SD_over_offset_SD']=float(np.mean([row['final_delta_SD_over_offset_SD'] for row in group]))
                seeds.append(item)
            selected=[row for row in seeds if row['arm']==arm and row['epochs']==epochs]
            result=dict(arm=arm,epochs=epochs,cells=6,seeds=2,
                original_COMPLETE_cells=sum(row['original_COMPLETE_cells'] for row in selected),
                original_FAILED_cells=sum(row['original_FAILED_cells'] for row in selected))
            for field in ('own80','final_vs_own80','final_vs_offset','train80','train_final'):
                result[field]={key:float(np.mean([row[field][key] for row in selected])) for key in
                    ('Harrell','Uno60','repair','damage','Uno_repair','Uno_damage','delta_H_vs_own_offset','delta_Uno60_vs_own_offset')}
            for field in ('complete_tokens_decoder_R2','pooled32_decoder_R2','mean_training_Cox_change','mean_final_delta_SD_over_offset_SD'):
                values=[row[field] for row in selected]
                result[field]=float(np.mean(values)) if all(value is not None for value in values) else None
            means.append(result)
    return seeds,means


def run_audit(args,lock):
    require(make_lock(args)==lock,'LOCKED_AUDIT_SOURCE_OR_ARTIFACT_CHANGED')
    torch.set_num_threads(2);require(not torch.cuda.is_initialized(),'CUDA_CONTEXT_MUST_NOT_EXIST')
    assets=read(args.assets)
    with np.load(assets['endpoints'],allow_pickle=False) as file:time,event=np.array(file['DSS_time']),np.array(file['DSS_event'])
    rows=[];initial={};replays=[];failed_forward=None
    for part in (0,1,2):
        prepared=torch.load(args.primary_private/'prepared'/f'outer1_inner{part}.pt',map_location='cpu',weights_only=False)
        train,valid=prepared['train'],prepared['valid'];offset=np.asarray(prepared['RNA'][valid],np.float32)
        for arm in ('O3','O4'):
            for seed in (31,47):
                parent=torch.load(args.primary_private/'cells'/f'R2_outer1_inner{part}_{arm}_seed{seed}'/'model.pt',map_location='cpu',weights_only=False)
                for epochs in (160,320):
                    name=label(arm,part,seed,epochs);binding=next(row for row in lock['models'] if row['unit']==name)
                    model=torch.load(args.private/'cells'/name/'model.pt',map_location='cpu',weights_only=False)
                    require(model['lineage']==read(args.private/'cells'/name/'START.json')['lineage']
                            and model['epochs']==model['optimizer_updates']==epochs
                            and model['actual_teacher_refits']==model['actual_preprocessing_refits']==0
                            and model['final_retention_decoder_fits']==2 and model['select_best_epoch'] is False,
                            'SAVED_FIT_PROTOCOL_CHANGED')
                    require(np.array_equal(model['train'],train) and np.array_equal(model['valid'],valid),'SAVED_ROWS_CHANGED')
                    key=(arm,part,seed);initial.setdefault(key,{})[epochs]=model['initial_state_sha256']
                    own80,final=model['snapshots'][80],model['snapshots'][epochs]
                    for snapshot in (own80,final):
                        require(np.array_equal((offset+np.asarray(snapshot['delta'],np.float32)).astype(float),snapshot['risk']),
                                'SAVED_FLOAT32_OFFSET_PLUS_DELTA_MISMATCH')
                    original_comparison=compare_parent80(time[valid],event[valid],time[train],event[train],
                        parent['snapshots'][80]['risk'],own80['risk'],2e-6)
                    require(original_comparison==model['parent80_replay'],'ORIGINAL_PARENT80_COMPARISON_CERTIFICATE_CHANGED')
                    replays.append(dict(unit=name,original_role=binding['original_role'],**original_comparison))
                    curve80=next(row for row in model['curves'] if row['epoch']==80)
                    curve_final=next(row for row in model['curves'] if row['epoch']==epochs)
                    item={key:binding[key] for key in ('unit','arm','partition','seed','epochs','original_role')}
                    metric_args=(time[valid],event[valid],time[train],event[train])
                    item['own80']=with_delta(independent_metrics(*metric_args,own80['risk'],offset.astype(float)))
                    item['final_vs_own80']=with_delta(independent_metrics(*metric_args,final['risk'],own80['risk']))
                    item['final_vs_offset']=with_delta(independent_metrics(*metric_args,final['risk'],offset.astype(float)))
                    for snapshot,curve,metric in ((own80,curve80,item['own80']),(final,curve_final,item['final_vs_offset'])):
                        require(snapshot['metrics']==curve['valid'],'SAVED_CURVE_SNAPSHOT_METRICS_DIFFER')
                        require(all(abs(metric[key]-snapshot['metrics'][key])<1e-9 for key in
                            ('Harrell_num','Harrell_den','Uno_num','Uno_den','Harrell','Uno60','repair','damage','Uno_repair','Uno_damage')),
                            'INDEPENDENT_SAVED_VALID_METRIC_DIFFERS')
                    item.update(train80=curve80['train'],train_final=curve_final['train'],
                        training_Cox_own80=curve80['cox_loss'],training_Cox_final=curve_final['cox_loss'],
                        training_Cox_final_minus_own80=curve_final['cox_loss']-curve80['cox_loss'],
                        final_delta_SD_over_offset_SD=float(np.asarray(final['delta']).std()/offset.astype(float).std()),
                        final_decoder=model['retention'],final_decoder_fits_already_performed=2,
                        own80_decoder_not_persisted_and_not_refitted=True,
                        train_metrics_boundary='saved scalar curves relative to own CF training offset; full train-risk arrays were not persisted',
                        parameters_registered=model['parameters'],actual_optimizer_updates=model['optimizer_updates'])
                    rows.append(item)
                    if name==FAILED_UNIT:
                        with np.load(assets['cohort'],allow_pickle=True) as file:symbols=np.array(file['rna_gene_symbols']).astype(str)
                        member,_=load_gmt_membership(assets['gmt'],symbols,expected_sha256=lock['GMT_sha256'])
                        torch.manual_seed(seed);live=MechanismModel('R2','O4',prepared,member,None,'cpu',gamma=.3)
                        live.load_state_dict(final['state_dict'],strict=True)
                        actual,delta=predict(live,valid,batch_size=8)
                        comparison=compare_parent80(*metric_args,final['risk'],actual,2e-6)
                        failed_forward=dict(unit=name,mode='CPU_actual_final160_weights_batch8_one_fixed_replay',
                            status='PASS_ORIGINAL_CPU_RISK_GATE' if comparison['within_locked_tolerance'] else 'CPU_FLOAT32_RISK_GATE_EXCEPTION_RETAINED',
                            comparison=comparison,actual_vs_saved_delta_max_abs_error=float(np.max(np.abs(delta-final['delta']))),
                            parameters_unchanged=all(torch.equal(value,live.state_dict()[key]) for key,value in final['state_dict'].items()),
                            threshold_unchanged=True,original_failed_budget_receipt_promoted=False)
                        del live
                    del model
                del parent
        del prepared
    initial_results=[]
    for (arm,part,seed),budgets in sorted(initial.items()):
        require(set(budgets)=={160,320},'SAME_UNIT_BUDGET_PAIR_MISSING')
        require(budgets[160]==budgets[320],'NEW160_NEW320_INITIAL_STATE_HASH_DIFFER')
        initial_results.append(dict(arm=arm,partition=part,seed=seed,named_states=len(budgets[160]),
            exact_new160_new320_initial_state_hashes=True,original_parent80_initial_hashes_not_inferred=True))
    seed_rows,means=aggregate_cells(rows)
    require(len(rows)==24 and failed_forward is not None and not torch.cuda.is_initialized(),'AUDIT_SCOPE_OR_CUDA_CHANGED')
    return dict(schema='bgb_en_zero_fit_budget_saved_chain_audit_v1',
        status='COMPLETE_24_SAVED_CHAIN_DIAGNOSIS_ORIGINAL_23_ACCEPTED_1_FAILED_RETAINED',
        audit_lock_sha256=sha256(args.lock),source_sha256=sha256(__file__),models=24,original_COMPLETE=23,original_FAILED=1,
        actual_saved_fit_optimizer_updates=sum(row['actual_optimizer_updates'] for row in rows),
        successful_qualified_optimizer_updates=sum(row['actual_optimizer_updates'] for row in rows if row['original_role']=='ORIGINAL_COMPLETE'),
        failed_but_physically_finished_optimizer_updates=160,actual_existing_risk_decoders=48,
        qualified_risk_decoders=46,failed_model_existing_risk_decoders=2,
        initial_state_pairs=initial_results,parent80_to_own80_all24=replays,
        parent80_guard_failures=sum(not row['within_locked_tolerance'] for row in replays),
        parent80_comparable_pair_credit_changes=sum(row['changed_pair_credits'] for row in replays),
        failed_actual_weight_CPU_replay=failed_forward,per_cell=rows,per_seed=seed_rows,seed_mean_full_saved_diagnostic=means,
        aggregation='same seed first sum three within-partition numerators and denominators, then mean seeds; no cross-partition risk ranking',
        original_qualification_changed=False,full_saved_diagnostic_is_not_24_qualified_performance=True,
        no_best_epoch_or_budget_selection=True,R4_or_R5_lock_changed=False,
        audit_work=dict(new_fits=0,new_decoders=0,optimizers=0,backward_calls=0,GPU_calls=0,
            CPU_model_constructions=1,CPU_patient_risk_replays=1,CPU_replay_batch_size=8),
        boundary='post hoc numerical/same-chain diagnostic; CPU gate exceptions remain explicit, and failed records are never manufactured as COMPLETE')


def write_reports(report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'R2_BUDGET_SAVED_CHAIN_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# 全24保存实物的同链预算诊断','',
        '**原状态保留：23 COMPLETE、1 FAILED_PARENT80_NUMERICAL_GUARD。此处不是24个合格预算模型的新性能确认。**','',
        '每个新拟合使用自身保存80轮→最终160/320轮配对；先在seed内合并三个分区分子分母，再对seed求均值。未替换原parent80对照、未选择最佳预算、未修改R4。','',
        '|arm|final epoch|原接受/失败|own80 H/U|final H/U|ΔH/own80|ΔU/own80|修复/损伤|train Cox变化|',
        '|---|---:|---:|---|---|---:|---:|---|---:|']
    for row in report['seed_mean_full_saved_diagnostic']:
        b,f=row['own80'],row['final_vs_own80']
        lines.append(f"|{row['arm']}|{row['epochs']}|{row['original_COMPLETE_cells']}/{row['original_FAILED_cells']}|{b['Harrell']:.6f}/{b['Uno60']:.6f}|{f['Harrell']:.6f}/{f['Uno60']:.6f}|{f['delta_H_vs_own_offset']:+.6f}|{f['delta_Uno60_vs_own_offset']:+.6f}|{f['repair']:.6f}/{f['damage']:.6f}|{row['mean_training_Cox_change']:+.6f}|")
    forward=report['failed_actual_weight_CPU_replay'];comparison=forward['comparison']
    lines+=['','## 原门槛与真实前向','',
        f"全部24个旧parent80→各新拟合own80比较：{report['parent80_guard_failures']}个超过原2e-6风险误差门；全部可比对credit变化合计{report['parent80_comparable_pair_credit_changes']}。每个160/320配对的实际初始state SHA逐项一致，但旧parent80未保存该初始化证据，不能仅凭seed声称其逐位同初值。不同arm的实际state数见JSON，不硬写均为19。",'',
        f"失败160只执行一次固定CPU权重回放：{forward['status']}；max风险误差{comparison['max_abs_risk_error']:.9g}，credit变化{comparison['changed_pair_credits']}，ΔH={comparison['Harrell_change']:.9g}、ΔU={comparison['Uno60_change']:.9g}。原FAILED、2e-6门槛与源文件均未改变。",'',
        '实际已发生5760次神经更新与48个解析decoder，其中合格5600次更新/46decoder，失败实物已发生160次更新/2decoder；这不是新增工作量。CPU审计新优化、解码器拟合、反传和GPU调用均0。', '',
        '训练列来自保存曲线，相对本人crossfit offset；没有完整训练风险数组的独立重算。最终RNA decoder直接引用已保存评估，own80 decoder未持久化，不重新拟合。优势是完整保留失败实物和同链变化；风险是事后诊断和CPU/GPU浮点差异，不能提升为独立确认或据此修改外层锁。','']
    (output/'R2_BUDGET_SAVED_CHAIN_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('lock','audit'),required=True)
    parser.add_argument('--private',type=Path,required=True);parser.add_argument('--primary-private',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/bgb_en')
    args=parser.parse_args()
    if args.stage=='lock':
        lock=make_lock(args);args.lock.parent.mkdir(parents=True,exist_ok=True)
        if args.lock.exists():require(read(args.lock)==lock,'EXISTING_AUDIT_LOCK_CHANGED')
        else:args.lock.write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(dict(status=lock['status'],models=24,new_fits=0,model_deserializations=0,GPU_calls=0)))
    else:
        report=run_audit(args,read(args.lock));write_reports(report,args.output_dir)
        print(json.dumps({key:report[key] for key in ('status','parent80_guard_failures','parent80_comparable_pair_credit_changes','failed_actual_weight_CPU_replay')}))


if __name__=='__main__':main()

"""Read-only anonymous source-initialization report; no model deserialization.

All artifact bytes are checked against COMPLETE receipts. Concordance is first
pooled by eligible within-partition numerator/denominator for each seed. Source
controls and target survival fits remain distinct scientific data contracts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bgb_en_report_inner import aggregate,validate_metrics

GROUPS=('TRUE','WRONG_DONOR','SPATIAL_SHUFFLE')
ARMS={'TRUE':('F5_UNI2','F11_UNI2'),'WRONG_DONOR':('F11_UNI2',),'SPATIAL_SHUFFLE':('F11_UNI2',)}
SOURCE_NAMES={'RAW_V1':'external_v1','TRUE':'external_v2_unitnorm',
              'WRONG_DONOR':'external_v2_WRONG_DONOR','SPATIAL_SHUFFLE':'external_v2_SPATIAL_SHUFFLE'}
SOURCE_ARTIFACTS={'checkpoint.pt','PRIVATE_PARTITION.npz','UNI2_PROJECT_TRANSFER.pt','PRIVATE_TARGET_MAP.npz'}
COMPARISONS=(('TRUE_F11_minus_TRUE_F5',('TRUE','F11_UNI2'),('TRUE','F5_UNI2')),
             ('WRONG_DONOR_minus_TRUE',('WRONG_DONOR','F11_UNI2'),('TRUE','F11_UNI2')),
             ('SPATIAL_SHUFFLE_minus_TRUE',('SPATIAL_SHUFFLE','F11_UNI2'),('TRUE','F11_UNI2')))


def require(value,message):
    if not bool(value):raise ValueError(message)


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def mean(values):
    return statistics.mean(values) if values and all(value is not None for value in values) else None


def metric_subset(value):
    names=('rows','program_standardized_Huber','program_standardized_MSE','tissue_balanced_accuracy',
           'per_program_Spearman','per_program_within_section_tissue_Spearman','tissue_recall')
    return {name:value[name] for name in names if name in value} if value is not None else None


def verify_source_receipt(path,version,index):
    """Read scalar receipts and hash artifact bytes; never open private arrays."""
    row=json.loads(path.read_text(encoding='utf-8'))
    negative=version in ('WRONG_DONOR','SPATIAL_SHUFFLE')
    require(row.get('status')==('COMPLETE_NEGATIVE_SOURCE_FIT' if negative else 'COMPLETE_FIXED_SOURCE_FIT'),
            'Source COMPLETE status differs')
    require(row.get('epochs')==20 and row.get('optimizer_updates')==20
            and row.get('actual_new_supervised_fits')==1,'Source fit did not complete exactly twenty updates')
    require(isinstance(row.get('artifacts'),dict) and set(row['artifacts'])<=SOURCE_ARTIFACTS
            and 'checkpoint.pt' in row['artifacts'],'Unrecognized source artifact receipt')
    for name,digest in row['artifacts'].items():
        require((path.parent/name).is_file() and sha256(path.parent/name)==digest,'Source artifact byte SHA mismatch')
    if negative:
        lineage=row['lineage'];require(lineage.get('source_control')==version
            and lineage.get('TCGA_DSS_read') is False and lineage.get('original_ST_test_read') is False
            and lineage.get('independent_heldout_evaluation') is False
            and lineage.get('tissue_CE_targets_rematched') is False,'Negative source control contract differs')
        require(lineage['target_scaler_source']=='original_unrematched_full_source_train_targets',
                'Negative source fit did not retain original true source scaling')
        role='full_source_transfer';train=row.get('rematched_training_target_metrics');valid=None
        original=metric_subset(row.get('original_true_target_training_metrics'))
        source_data=lineage['source_data_sha256'];source_manifest=lineage['source_manifest_sha256']
    else:
        contract=row['contract'];require(contract.get('fit_index')==index,'Source fit index differs')
        role=contract['role'];require(role==('full_source_transfer' if index==6 else 'leave_one_donor_out'),
                                     'Source heldout role differs')
        lineage=contract['lineage'];source_data=lineage['source_data_sha256'];source_manifest=lineage['source_manifest_sha256']
        train=row.get('train_metrics');valid=row.get('valid_metrics');original=None
        require((valid is None)==(index==6),'Source heldout metrics present for full fit or absent for heldout fit')
    return dict(version=version,fit_index=index,role=role,epochs=20,optimizer_updates=20,
        transfer_eligible_under_current_unitnorm_contract=version!='RAW_V1',
        independent_heldout_evaluation=role=='leave_one_donor_out',
        train_metrics=metric_subset(train),heldout_metrics=metric_subset(valid),
        original_true_target_training_metrics=original,
        elapsed_seconds=row.get('elapsed_seconds'),receipt_sha256=sha256(path),
        artifacts_sha256=row['artifacts'],source_data_sha256=source_data,source_manifest_sha256=source_manifest)


def source_report(private_root,checkpoint_audit=None):
    rows=[];projects={};expected=16
    for version,name in SOURCE_NAMES.items():
        for index in (range(7) if version in ('RAW_V1','TRUE') else (6,)):
            path=Path(private_root)/name/f'fit_{index:02d}'/'COMPLETE.json'
            if path.is_file():
                row=verify_source_receipt(path,version,index);rows.append(row)
                if index==6 and version!='RAW_V1':
                    require('UNI2_PROJECT_TRANSFER.pt' in row['artifacts_sha256'],'Complete transfer source lacks projection')
                    projects[version]=row['artifacts_sha256']['UNI2_PROJECT_TRANSFER.pt']
    identities={(row['source_data_sha256'],row['source_manifest_sha256']) for row in rows}
    require(len(identities)<=1,'Source fits mix different data or annotation manifests')
    heldout={}
    for version in ('RAW_V1','TRUE'):
        choose=[row for row in rows if row['version']==version and row['role']=='leave_one_donor_out']
        heldout[version]=dict(completed_donor_heldout_fits=len(choose),planned_donor_heldout_fits=6,
            averaging='unweighted donor mean, reported only after all six fixed source folds',
            mean_program_standardized_Huber=mean([r['heldout_metrics']['program_standardized_Huber'] for r in choose]) if len(choose)==6 else None,
            mean_tissue_balanced_accuracy=mean([r['heldout_metrics']['tissue_balanced_accuracy'] for r in choose]) if len(choose)==6 else None)
    replay=None
    if checkpoint_audit is not None and Path(checkpoint_audit).is_file():
        independent=json.loads(Path(checkpoint_audit).read_text())
        require(independent.get('status')=='PASS_FOURTEEN_EXTERNAL_CHECKPOINT_REPLAY'
                and independent.get('actual_recorded_source_fits')==14
                and independent.get('raw_v1_transfer_eligible_under_v2_contract') is False,
                'Independent external replay does not meet its declared qualification')
        replay=dict(audit_sha256=sha256(checkpoint_audit),status=independent['status'],
                    annotation=independent['annotation'],audit_optimizer_updates=independent['audit_optimizer_updates'])
    result=dict(schema='bgb_en_external_pretraining_receipt_report_v1',
        status='COMPLETE_SIXTEEN_SOURCE_FITS' if len(rows)==expected else 'RUNNING',
        actual_completed_source_fits=len(rows),planned_source_fits=expected,
        actual_recorded_optimizer_updates=sum(row['optimizer_updates'] for row in rows),
        source_counts={version:sum(row['version']==version for row in rows) for version in SOURCE_NAMES},
        raw_v1_transferred=False,raw_v1_transfer_eligible_under_current_contract=False,
        source_truth='independent tissue spot annotations and measured local program proxies; not cell-source truth',
        source_annotation_scope='tumor, stroma, tumor_stroma_mixed; not purified cells or calibrated cell fractions',
        source_input_scope='raw-v1 excluded from migration; L2-v2 matches target per-patch normalization',
        negative_control_scope='complete six-donor source training; remapped program targets retain original tissue CE and true-target scaling; no heldout claim',
        heldout_summary=heldout,source_projection_sha256=projects,receipts=rows,
        independent_checkpoint_replay=replay,TCGA_DSS_read=False,original_ST_test_read=False,
        report_optimizer_updates=0,GPU_used=False,patient_identifiers_exported=False)
    return result,projects


def load_aux_cells(private_root,projects):
    cells={};origins=set();caches=set();sources=set()
    for group in GROUPS:
        folder=Path(private_root)/f'auxiliary_{group}_v2'/'cells'
        for path in sorted(folder.glob('*/COMPLETE.json')):
            row=json.loads(path.read_text())
            require(row.get('status')=='COMPLETE' and row.get('stage')=='R3' and row.get('arm') in ARMS[group],
                    'Auxiliary receipt contains an unknown group or treatment')
            seed,part=int(row['seed']),int(row['inner_partition'])
            require(seed in (31,47) and part in (0,1,2) and row.get('outer_fold')==1
                    and row.get('outer_test_evaluated') is False,'Auxiliary receipt violates inner-only evaluation')
            require(row.get('epochs')==80 and row.get('actual_optimizer_fits')==1,'Auxiliary fit has wrong fixed budget')
            final=[point for point in row.get('curves',[]) if point.get('epoch')==80]
            require(len(final)==1 and final[0].get('valid')==row['metrics'],'Auxiliary endpoint is not final epoch80')
            validate_metrics(row['metrics'])
            require((path.parent/'model.pt').is_file()
                    and sha256(path.parent/'model.pt')==row['artifact_sha256'],'Auxiliary checkpoint byte SHA mismatch')
            lineage=row['lineage'];origin=lineage['shared_teacher_origin_lineage']
            require(all(lineage.get(key)==value for key,value in origin.items()),'Auxiliary teacher origin was relabelled')
            origins.add(canonical_hash(origin));caches.add(lineage['UNI2_cache_receipt_sha256'])
            sources.add(canonical_hash(lineage['auxiliary_source']))
            projection=lineage.get('external_projection_sha256')
            require(projection==(projects.get(group) if row['arm']=='F11_UNI2' else None)
                    and (row['arm']!='F11_UNI2' or group in projects),
                    'Target initialization does not match its verified source projection')
            key=group,row['arm'],part,seed;require(key not in cells,'Duplicate auxiliary logical unit')
            cells[key]=dict(group=group,arm=row['arm'],partition=part,seed=seed,
                train_n=row['train_n'],valid_n=row['valid_n'],parameters=row['parameters'],
                elapsed_seconds=row['elapsed_seconds'],peak_GPU_bytes=row['peak_GPU_bytes'],
                receipt_sha256=sha256(path),artifact_sha256=row['artifact_sha256'],
                source_projection_sha256=projection,**row['metrics'])
    require(len(origins)<=1 and len(caches)<=1 and len(sources)<=1,
            'Auxiliary comparisons mix source origins, UNI2 patch caches or execution sources')
    return cells,dict(shared_teacher_origin_sha256=next(iter(origins),None),
        UNI2_cache_READY_sha256=next(iter(caches),None),auxiliary_execution_sources_sha256=next(iter(sources),None))


def summarize_aux(cells):
    seed_rows=[];summaries=[];paired=[]
    for group in GROUPS:
        for arm in ARMS[group]:
            selected=[v for (g,a,p,s),v in cells.items() if g==group and a==arm]
            for seed in (31,47):
                rows=[cells[(group,arm,part,seed)] for part in (0,1,2) if (group,arm,part,seed) in cells]
                values=aggregate(rows) if len(rows)==3 else {name:None for name in ('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den')}
                seed_rows.append(dict(group=group,arm=arm,seed=seed,status='COMPLETE' if len(rows)==3 else 'RUNNING',
                    completed_partitions=len(rows),planned_partitions=3,**values))
            complete=len(selected)==6
            ss=[r for r in seed_rows if r['group']==group and r['arm']==arm]
            summaries.append(dict(group=group,arm=arm,status='COMPLETE' if complete else 'RUNNING',
                completed_cells=len(selected),planned_cells=6,
                Harrell=mean([r['Harrell'] for r in ss]) if complete else None,
                Uno60=mean([r['Uno60'] for r in ss]) if complete else None,
                parameters_mean=mean([r['parameters'] for r in selected]),
                elapsed_seconds_sum=sum(r['elapsed_seconds'] for r in selected)))
    for name,left,right in COMPARISONS:
        pairs=[]
        for seed in (31,47):
            ll=[];rr=[]
            for part in (0,1,2):
                l=cells.get((*left,part,seed));r=cells.get((*right,part,seed))
                if l is not None and r is not None:
                    require(l['train_n']==r['train_n'] and l['valid_n']==r['valid_n']
                            and l['parameters']==r['parameters'],'Source-initialization comparison changed population or model capacity')
                    require(all(math.isclose(l[key],r[key],rel_tol=1e-12,abs_tol=1e-10) for key in ('Harrell_den','Uno_den')),
                            'Paired auxiliary cells have different eligible comparison denominators')
                    ll.append(l);rr.append(r)
            pairs.append(len(ll));lval=aggregate(ll) if len(ll)==3 else {};rval=aggregate(rr) if len(rr)==3 else {}
            paired.append(dict(comparison=name,seed=seed,status='COMPLETE' if len(ll)==3 else 'RUNNING',
                completed_pairs=len(ll),planned_pairs=3,
                delta_Harrell=lval['Harrell']-rval['Harrell'] if len(ll)==3 and lval['Harrell'] is not None and rval['Harrell'] is not None else None,
                delta_Uno60=lval['Uno60']-rval['Uno60'] if len(ll)==3 and lval['Uno60'] is not None and rval['Uno60'] is not None else None))
        ss=[row for row in paired if row['comparison']==name and row['seed']!='MEAN']
        paired.append(dict(comparison=name,seed='MEAN',status='COMPLETE' if pairs==[3,3] else 'RUNNING',
            completed_pairs=sum(pairs),planned_pairs=6,
            delta_Harrell=mean([row['delta_Harrell'] for row in ss]) if pairs==[3,3] else None,
            delta_Uno60=mean([row['delta_Uno60'] for row in ss]) if pairs==[3,3] else None))
    return summaries,seed_rows,paired


def write_csv(path,rows,empty_fields=()):
    fields=list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    require(fields,'Anonymous CSV requires a declared header even when no model has completed')
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def formatted(value):return 'NA' if value is None else f'{value:.6f}'


def write_reports(output,source,cells,lineage):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summaries,seeds,paired=summarize_aux(cells)
    result=dict(schema='bgb_en_auxiliary_initialization_report_v1',status='COMPLETE_PRELOCKED_AUXILIARY_FITS' if len(cells)==24 else 'RUNNING',
        completed_unique_survival_fits=len(cells),planned_unique_survival_fits=24,
        summary=summaries,seed_metrics=seeds,paired_comparisons=paired,**lineage,
        source_projection_sha256=source['source_projection_sha256'],
        aggregation='sum within-partition concordance numerators/denominators for each seed, then mean of two seeds; no cross-partition risk comparisons',
        endpoint='fixed final epoch80; no checkpoint selection',
        comparison_scope='same UNI2-h REGIONS128 input, same teachers, folds, model capacity and optimization; source initialization differs',
        CONCH512_is_not_a_matched_source_initialization_control=True,
        confidence_intervals_computed=False,new_independent_validation=False,outer_test_evaluated=False,
        report_optimizer_updates=0,GPU_used=False,model_deserialization=False,patient_identifiers_exported=False,
        report_script_sha256=sha256(__file__))
    (output/'AUXILIARY_TRANSFER_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(output/'AUXILIARY_TRANSFER_FOLD_METRICS.csv',list(cells.values()),
              ('group','arm','partition','seed','train_n','valid_n','parameters','elapsed_seconds',
               'peak_GPU_bytes','receipt_sha256','artifact_sha256','source_projection_sha256',
               'Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den'))
    write_csv(output/'AUXILIARY_TRANSFER_SEED_METRICS.csv',seeds)
    write_csv(output/'AUXILIARY_TRANSFER_PAIRED.csv',paired)
    lines=['# 外部来源初始化：匹配 UNI2-128 只读报告','',
           f"**状态：{result['status']}；已完成 {len(cells)}/24 个预定生存拟合。**",'',
           '| 来源/模型 | 完成 | Harrell | Uno60 |','|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['group']} / {r['arm']} | {r['completed_cells']}/6 | {formatted(r['Harrell'])} | {formatted(r['Uno60'])} |")
    lines+=['','## 直接配对比较','','| 比较 | 完成配对 | ΔHarrell | ΔUno60 |','|---|---:|---:|---:|']
    for r in paired:
        if r['seed']=='MEAN':lines.append(f"| {r['comparison']} | {r['completed_pairs']}/6 | {formatted(r['delta_Harrell'])} | {formatted(r['delta_Uno60'])} |")
    lines+=['','每个 seed 先相加三个内层分区的可比患者对分子和分母，再计算 C-index；最终报告两个 seed 的均值。未完成全部预定分区时不生成该 seed 的聚合性能。',
            '', '同一模型的 COMPLETE、model.pt 字节 SHA、最终80轮快照、教师源谱系、UNI2缓存与初始化投影均被核验。这里没有反序列化模型、前向推理、优化器或GPU调用。',
            '', '## 优势与限制','',
            '优势：真正同输入、同结构、同容量的源初始化比较；TRUE 与同一 UNI2-128 的 F5基线比较，负来源与 TRUE 逐 seed、逐分区配对。',
            '', '风险：本报告是已开发队列的内层诊断，不是外部生存验证，也不提供独立置信区间。CONCH512 与 UNI2-128 的差值不能归因于空间监督。冻结干预、重训源初始化与真实细胞来源验证是不同问题。',
            '', '组织来源标签来自独立 spot 组织学标注（tumor/stroma/mixed）；它们不是真实单细胞来源、细胞比例或医学因果证据。raw-v1 的7个拟合仅保留审计记录，因范数合同不匹配不迁移。']
    (output/'AUXILIARY_TRANSFER_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (output/'EXTERNAL_PRETRAINING_RESULTS.json').write_text(json.dumps(source,indent=2,allow_nan=False)+'\n')
    lines=['# 外部组织/分子程序预训练：来源拟合核验','',
           f"**状态：{source['status']}；实际完成 {source['actual_completed_source_fits']}/16 个来源拟合，共 {source['actual_recorded_optimizer_updates']} 次已记录优化更新。**",'',
           '| 来源 | 实际拟合 | 可用于当前目标初始化 | 独立留供体评价 |', '|---|---:|---|---|']
    for name,count in source['source_counts'].items():lines.append(f"| {name} | {count} | {'否' if name=='RAW_V1' else '仅完整来源投影'} | {'六个预定留供体分区' if name in ('RAW_V1','TRUE') else '否，仅完整来源训练'} |")
    lines+=['','每个来源拟合固定20轮、20次更新。逐项 COMPLETE 及其所有声明产物均通过字节SHA验证；本报告没有重新训练或加载权重推理。raw-v1不迁移；L2-v2来源与目标使用相同逐行L2归一化定义。',
            '', '组织标注为独立病理专家的 spot 组织类别（肿瘤、间质、混合）；分子程序是局部测量代理，不等于细胞来源真值。负来源只改变程序目标对应，保留组织CE标签及真实原目标的标准化，训练指标不能称为留出验证。',
            '', '## 留供体汇总','','| 来源 | 完成 | Huber（低为好） | 组织BA |', '|---|---:|---:|---:|']
    for name,r in source['heldout_summary'].items():lines.append(f"| {name} | {r['completed_donor_heldout_fits']}/6 | {formatted(r['mean_program_standardized_Huber'])} | {formatted(r['mean_tissue_balanced_accuracy'])} |")
    lines+=['','六折完整后采用等供体均值；完整来源训练、负来源训练不混入留供体汇总。优势是源合同和字节谱系可核查；风险是组织学代理任务的改善不自动等于DSS增量。']
    (output/'EXTERNAL_PRETRAINING_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/bgb_en')
    parser.add_argument('--checkpoint-audit',type=Path,default=ROOT/'reports/bgb_en/EXTERNAL_CHECKPOINT_REPLAY.json')
    args=parser.parse_args()
    source,projects=source_report(args.private_root,args.checkpoint_audit)
    cells,lineage=load_aux_cells(args.private_root,projects)
    result=write_reports(args.output,source,cells,lineage)
    print(json.dumps(dict(status=result['status'],source_fits=source['actual_completed_source_fits'],
                          target_survival_fits=len(cells),report_optimizer_updates=0,GPU_used=False)),flush=True)


if __name__=='__main__':main()

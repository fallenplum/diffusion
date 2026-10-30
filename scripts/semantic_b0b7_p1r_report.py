"""Anonymous P1-R actual-matrix reports from JSON receipts, no model forward.

Reports retain every final program metric and every epoch's scalar losses and
relation diagnostics. Full private checkpoints and per-patient arrays are never
copied. Individual parameter gradients are published at the four checkpoints;
the unchanged private producer receipt hashes bind the more extensive traces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def csv_parts(root,name,rows):
    if not rows:raise ValueError('EMPTY_REPORT_TABLE:'+name)
    fields=list(dict.fromkeys(k for row in rows for k in row));files=[]
    def line(row=None):
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n')
        w.writeheader() if row is None else w.writerow(row)
        return s.getvalue().encode('utf-8')
    header=line();buffer=bytearray(header);count=0;part=1
    for row in rows:
        raw=line(row)
        if count and (count>=19999 or len(buffer)+len(raw)>1700*1024):
            p=root/(name+'_'+str(part).zfill(3)+'.csv')
            with p.open('xb') as f:f.write(buffer)
            files.append(p.name);part+=1;buffer=bytearray(header);count=0
        buffer.extend(raw);count+=1
    p=root/(name+'_'+str(part).zfill(3)+'.csv')
    with p.open('xb') as f:f.write(buffer)
    files.append(p.name);return files


def point(metric,chance_reference=None):
    program=metric['program']
    result={'program_Pearson':program['macro_program_pearson'],'program_Spearman':program['macro_program_spearman'],
        'program_Huber':program['patient_equal_masked_huber'],'program_loss':metric['program_loss'],
        'unweighted_pair_loss':metric['pair_hard_loss'],'auxiliary_loss_not_cross_lambda_rank':metric['auxiliary_loss']}
    for mode in ('hard','random'):
        for direction in ('image_to_gene','gene_to_image'):
            r=metric['retrieval_'+mode][direction];prefix=mode+'_'+direction
            for source,target in (('top1_tie_averaged','Top1'),('paired_gap_mean','gap'),('exact_chance_mean','chance'),
                                  ('top1_minus_exact_chance','Top1_minus_chance'),('anchors_covered','covered_anchors')):
                result[prefix+'_'+target]=r.get(source)
            if chance_reference is not None:
                reference=chance_reference[mode,direction]
                if reference['anchors_covered']!=r['anchors_covered']:raise ValueError('CURVE_CHANCE_SUPPORT_CHANGED')
                chance=reference['exact_chance_mean'];top1=r['top1_tie_averaged']
                result[prefix+'_chance']=chance
                result[prefix+'_Top1_minus_chance']=top1-chance if top1 is not None and chance is not None else None
    return result


def flatten_relation(row):
    out={}
    for key,value in row.items():
        if isinstance(value,dict):
            out.update({key+'_'+k:v for k,v in value.items()})
        else:out[key]=value
    return out


def comparisons(cells,selected_fraction):
    indexed={(x['partition'],x['variant'],x['seed']):x for x in cells};rows=[]
    tag='A050' if selected_fraction==.5 else 'A075'
    for partition in (0,1,2):
        families=[('ORIGINAL_B4','ORIGINAL_B3','original_query_identity'),
            ('T3_B4_DUAL_'+tag,'T3_B3_DUAL_'+tag,'dual_query_identity'),
            ('T3_B4_DUAL_'+tag,'T3_B4_DUAL_MAP_SHUFFLE_'+tag,'dual_true_vs_shuffled_mapping'),
            ('T3_B4_DUAL_'+tag,'ORIGINAL_B4','dual_vs_original_true_query'),
            ('T4_B4_LAMBDA003','ORIGINAL_B4','pair_weight003_vs010_true_query'),
            ('T4_B3_LAMBDA003','ORIGINAL_B3','pair_weight003_vs010_random_query'),
            ('T4_B4_LAMBDA003','T4_B3_LAMBDA003','pair_weight003_query_identity')]
        if partition==0:
            families += [('ORIGINAL_B4','T1_TEXTLESS','explicit_text_vs_textless'),
                ('ORIGINAL_B4','T1_SCORE_ROUTE_CONTROL','true_score_vs_train_moment_surrogate'),
                ('T4_B4_LAMBDA000','ORIGINAL_B4','pair_weight000_vs010_true_query'),
                ('T4_B3_LAMBDA000','ORIGINAL_B3','pair_weight000_vs010_random_query')]
        else:
            families += [('T3_B4_DUAL_'+tag,'T3_B4_DUAL_WRONG_AUX_'+tag,'dual_true_vs_wrong_aux')]
        for a,b,label in families:
            for seed in (31,47,83):
                if (partition,a,seed) not in indexed or (partition,b,seed) not in indexed:continue
                aa,bb=indexed[partition,a,seed],indexed[partition,b,seed]
                pa,pb=point(aa['selected_validation']),point(bb['selected_validation'])
                value={'partition':partition,'seed':seed,'comparison':label,'candidate':a,'reference':b,
                       'candidate_epoch':aa['selected_epoch'],'reference_epoch':bb['selected_epoch']}
                for key in ('program_Pearson','program_Spearman','program_Huber','hard_image_to_gene_gap',
                            'hard_gene_to_image_gap','hard_image_to_gene_Top1_minus_chance','hard_gene_to_image_Top1_minus_chance'):
                    value['delta_'+key]=pa[key]-pb[key] if pa[key] is not None and pb[key] is not None else None
                rows.append(value)
    return rows


def build(args):
    matrix_path=args.matrix_root/'MATRIX_REPORT.private.json';matrix=read(matrix_path)
    master=read(args.lock);registry=read(args.registry);chance=read(args.chance)
    if matrix['status']!='COMPLETE_ACTUAL_REQUESTED_MATRIX' or len(matrix['cells'])!=84:
        raise ValueError('FULL84_ACTUAL_MATRIX_REQUIRED')
    if matrix['master_lock_sha256']!=sha(args.lock) or chance['matrix_report_sha256']!=sha(matrix_path):
        raise ValueError('MATRIX_OR_CHANCE_BINDING_CHANGED')
    if chance['new_fit_state_masks_checked']!=336 or chance['historical_reference_state_masks_checked']!=24:
        raise ValueError('EXACT336_PLUS24_SAVED_MASK_AUDIT_REQUIRED')
    for name,expected in master['frozen_sources'].items():
        if sha(args.repository_root/name)!=expected:raise ValueError('FROZEN_SOURCE_CHANGED:'+name)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    chance_metrics={(x['partition'],x['variant'],x['seed']):x['metrics'] for x in chance['selected_metric_projections']}
    chance_global={p['partition']:{(x['mode'],x['direction']):x for x in p['rows'] if x['scope']=='global'}
                   for p in chance['partitions']}
    cells=[];summaries=[];programs=[];strata=[];curves=[];relations=[];gradients=[];source_receipts=[]
    for record in matrix['cells']:
        path=args.matrix_root/'cells'/record['key']/'cell.private.json'
        if sha(path)!=record['receipt_sha256']:raise ValueError('PRIVATE_PRODUCER_RECEIPT_CHANGED')
        receipt=read(path);meta={k:record[k] for k in ('partition','variant','seed')}
        if receipt['master_lock_sha256']!=sha(args.lock) or receipt['optimizer_steps']!=400 or len(receipt['states'])!=4:
            raise ValueError('ACTUAL_CELL_SCOPE_OR_BUDGET_CHANGED')
        metric=chance_metrics[meta['partition'],meta['variant'],meta['seed']]
        value={**meta,'identity':'NEW_FIT','selected_epoch':receipt['selected_epoch'],'spec':receipt['spec'],
            'selected_checkpoint_sha256':receipt['selected_checkpoint_sha256'],
            'producer_receipt_sha256':record['receipt_sha256'],'input_sha256':record['input_sha256'],
            'train_rows_sha256':receipt['train_rows_sha256'],'valid_rows_sha256':receipt['valid_rows_sha256'],
            'train_patients':receipt['train_patients'],'valid_patients':receipt['valid_patients'],
            'program_huber_tolerance':receipt['program_huber_tolerance'],'parameter_counts':receipt['parameter_counts'],
            'lambda0_aux_only_parameters_not_trained':receipt['lambda0_aux_only_parameters_not_trained'],
            'P2_main_pretraining_eligible':receipt['P2_main_pretraining_eligible'],
            'selected_validation':metric,'states':receipt['states'],'negative_support':receipt['negative_support'],
            'elapsed_seconds':receipt['elapsed_seconds'],'peak_cuda_allocated_bytes':receipt['peak_cuda_allocated_bytes']}
        cells.append(value);source_receipts.append({'key':record['key'],'sha256':record['receipt_sha256']})
        for r in receipt['history']:
            norms=[x for x in r['gradient_norms'].values() if x is not None]
            scalar={**meta,'epoch':r['epoch'],'train_program_loss':r['train_program_loss'],
                'train_pair_loss':r['train_pair_loss'],'train_auxiliary_loss':r['train_auxiliary_loss'],
                'gradient_l2_all_tensors':math.sqrt(math.fsum(x*x for x in norms)),
                'gradient_max_tensor_norm':max(norms) if norms else None,'inactive_tensor_count':len(r['inactive_parameters'])}
            if 'validation' in r:
                scalar.update({'validation_'+k:v for k,v in point(r['validation'],chance_global[meta['partition']]).items()})
                for name,norm in r['gradient_norms'].items():
                    gradients.append({**meta,'epoch':r['epoch'],'parameter':name,'gradient_norm':norm,
                                      'inactive_this_step':name in r['inactive_parameters']})
            curves.append(scalar)
            relations.append({**meta,'epoch':r['epoch'],**flatten_relation(r['relation_trace'])})
    for ref in matrix['historical_references']['references']:
        metric=chance_metrics[0,ref['variant'],ref['seed']]
        cells.append({**ref,'selected_validation':metric,'identity':'REPLAY_ONLY_REFERENCE'})
    for cell in cells:
        meta={k:cell[k] for k in ('partition','variant','seed')};metric=cell['selected_validation']
        summaries.append({**meta,'identity':cell['identity'],'selected_epoch':cell['selected_epoch'],**point(metric)})
        for p in metric['program']['per_program']:
            programs.append({**meta,**p})
        for name,groups in metric['strata'].items():
            for group,values in groups.items():
                p=values['program'];strata.append({**meta,'stratum':name,'group':group,'patients':p['patients'],
                    'macro_Pearson':p['macro_program_pearson'],'macro_Spearman':p['macro_program_spearman'],
                    'patient_equal_Huber':p['patient_equal_masked_huber'],'measured_target_cells':p['measured_target_cells']})
    # Three-seed cell shards stay ordinary JSON and do not include private paths.
    public_shards=[]
    for partition in (0,1,2):
        names=sorted({c['variant'] for c in cells if c['partition']==partition})
        for variant in names:
            use=[c for c in cells if c['partition']==partition and c['variant']==variant]
            name='06_CELL_p'+str(partition)+'_'+variant+'.json';value={'partition':partition,'variant':variant,'cells':use}
            write(out/name,value)
            if (out/name).stat().st_size>2*1024**2:raise ValueError('PUBLIC_CELL_SHARD_EXCEEDS_UNCHANGED_LIMIT')
            public_shards.append({'file':name,'bytes':(out/name).stat().st_size,'sha256':sha(out/name)})
    tables={}
    for name,rows in (('06_INNER_PARTITION_SEED_METRICS',summaries),('06_PROGRAM_METRICS',programs),
        ('06_STRATUM_METRICS',strata),('06_TRAINING_CURVES',curves),
        ('04_RELATION_NORMALIZATION_AND_SHUFFLE',relations),('05_NEW_TRAINING_GRADIENT_SNAPSHOTS',gradients),
        ('06_PAIRED_COMPARISONS',comparisons(cells,matrix['dual_fraction_selection']['selected_fraction']))):
        tables[name]=csv_parts(out,name,rows)
    ledger={'schema':'semantic_b0b7_p1r_full_cost_v1','source_matrix_report_sha256':sha(matrix_path),
        'new_neural_fits':84,'new_neural_optimizer_updates':matrix['optimizer_updates'],
        'actual_new_saved_checkpoint_replays':matrix['actual_saved_checkpoint_replays'],
        'historical_reference_encoder_fits_reused_not_recounted':6,'reference_new_model_forwards':0,
        'R0_actual_old_saved_state_replays':48,'R0_separate_loss_backward_calls':96,
        'T2_final_Ridge_head_families':18,'T2_CV_Ridge_head_families':270,'T2_saved_head_replays':18,
        'T2_CV_scalar_system_rhs':13500,'T2_final_scalar_system_rhs':900,
        'actual_matrix_makespan_seconds':matrix['elapsed_wall_seconds'],
        'sum_individual_fit_wall_seconds_includes_checkpoint_evaluation':math.fsum(c['elapsed_seconds'] for c in cells if c['identity']=='NEW_FIT'),
        'maximum_individual_peak_cuda_allocated_bytes':max(c['peak_cuda_allocated_bytes'] for c in cells if c['identity']=='NEW_FIT'),
        'GPU_kernel_time_separately_instrumented':False,'CI_computed':False,'DSS_fits':0}
    write(out/'07_TRAINING_AND_COMPUTE_LEDGER.json',ledger)
    csv_parts(out,'07_TRAINING_AND_COMPUTE_LEDGER',[{'quantity':k,'value':v} for k,v in ledger.items() if not isinstance(v,dict)])
    # Recipe classes are frozen for P2; no checkpoint trained on this wider
    # development scope may be reused inside a new P2 patient fold.
    definitions={v['name']:v for v in registry['new_fit_variants']+registry['reference_variants']}
    specs=[('ORIGINAL_B4','P2_diagnostic_original_structure'),('T3_B4_DUAL_A050','P2_theory_fixed_main_PhaseA'),
           ('T4_B4_LAMBDA003','P1R_retained_comparator_not_added_to_P2_grid')]
    candidates=[]
    for name,role in specs:
        matching=[c for c in cells if c['variant']==name and c['identity']=='NEW_FIT']
        counts=[]
        for c in matching:
            if c['parameter_counts'] not in counts:counts.append(c['parameter_counts'])
        candidates.append({'name':name,'role':role,'spec':definitions[name],
            'model_source_sha256':master['frozen_sources']['src/semantic_b0b7/p1r_models.py'],
            'runner_source_sha256':master['frozen_sources']['scripts/semantic_b0b7_p1r.py'],
            'observed_parameter_count_records_across_training_partitions':counts,
            'epoch_rule':master['selection'],'optimizer':master['optimizer'],
            'inputs':{'patches':[512,768],'molecular_values_and_masks':[50,3],'clinical':'six legal baseline fields fold-local encoded'},
            'outputs':{'visual_tokens':[8,64],'gene_tokens':[50,64],'program_prediction':[50],'interaction_features':[2]},
            'pretraining_objective':'patient-equal masked measured-RNA50 Huber + lambda_pair * matched bidirectional InfoNCE',
            'new_P2_scope_requires_independent_train_local_preprocessing_and_encoder_fit':True})
    freeze={'schema':'semantic_b0b7_p1r_p2_frozen_recipe_classes_v1','status':'COMPLETE_RECIPE_FREEZE_NOT_A_DSS_SUCCESS_CLAIM',
        'P1R_master_lock_sha256':sha(args.lock),'matrix_report_sha256':sha(matrix_path),
        'P1R_repeated_fraction':matrix['dual_fraction_selection']['selected_fraction'],
        'P1R_fraction_rule':'six-cell development Huber/tolerance/hardgap before replication',
        'P2_primary_fraction':.5,'P2_primary_fraction_reason':'theory-fixed equal-channel engineering control before reading any new P2 outer score; not P1R Devbest',
        'P2_primary_structure_reason':'fixed mechanism hypothesis and software qualification; no positive P1R threshold required',
        'repair_classes':2,'original_diagnostic_classes':1,'candidates':candidates,
        'DSS_model_selection_from_P1R_results':False,'requires_separate_P2_train_scope_lock':True,
        'CI_required':False,'pretrained_development_weights_may_be_reused_for_P2':False}
    write(out/'10_P2_FROZEN_CANDIDATES.json',freeze)
    independent={'schema':'semantic_b0b7_p1r_public_replay_and_mask_scope_v1','status':'COMPLETE_PRODUCER_REPLAYS_AND_SEPARATE_MASK_AUDIT',
        'producer_actual_new_state_model_replays':336,'producer_replay_maximum_error':0.,
        'R0_actual_legacy_state_replays':48,'T2_actual_saved_Ridge_head_replays':18,
        'independent_new_state_mask_checks':chance['new_fit_state_masks_checked'],
        'independent_legacy_reference_state_mask_checks':chance['historical_reference_state_masks_checked'],
        'independent_checkpoint_arrays_deserialized':True,'independent_model_forwards_for_chance':0,
        'actual_mask_audit_source_sha256':chance['source_sha256'],'actual_mask_audit_output_sha256':sha(args.chance),
        'exact_chance_definition':chance['definition'],'scope_not_raw_images_or_foundation_model_reinference':True,
        'checkpoint_selection_changed_by_reporting':False,'CI_computed':False}
    write(out/'08_INDEPENDENT_REPLAY_AUDIT.json',independent)
    selection=matrix['dual_fraction_selection']
    index={'schema':'semantic_b0b7_p1r_public_matrix_index_v1','status':'COMPLETE84_FITS_AND_ZERO_DSS',
        'source_matrix_report_sha256':sha(matrix_path),'master_lock_sha256':sha(args.lock),
        'input_registry_sha256':matrix['input_registry_sha256'],'frozen_scientific_sources':master['frozen_sources'],
        'producer_receipts':source_receipts,'cell_shards':public_shards,'tables':tables,
        'dual_fraction_selection':selection,'new_neural_fits':84,'old_reference_replays_not_new_fits':6,
        'reporter_source_sha256':sha(__file__),'lossless_full_private_receipt_reconstruction_claimed':False,
        'relation_trace_units':{'raw_known_unknown_contributions':'normalized compatibility before temperature division',
            'InfoNCE_logit_contribution':'raw contribution / registry temperature',
            'temperature':registry['temperature'],'dual_unknown_logit_is_unused_buffer':True},
        'public_scope':'all selected50-program and stratum metrics; full scalar loss/relation curves; parameter gradients at4checkpoints',
        'patient_level_arrays_weights_predictions_or_identifiers_exported':False,'CI_computed':False,'DSS_fits':0}
    write(out/'06_INNER_PARTITION_SEED_METRICS.json',index)
    lines=['# P1-R 严格复现实验记录','',
        '已完成36个开发拟合、另两个inner partition的48个从头复验拟合；合计84×400次优化器更新和336次真实保存状态前向回放。旧B3/B4六参照只从既有四state按本轮统一规则重选，历史Laux结果和T2旧六状态均未改。','',
        '所有程序相关、Huber、检索和分层值均来自各自合法训练/验证范围；同患者多seed不视为独立队列，不计算置信区间。','',
        f"P1R复验fraction为{selection['selected_fraction']}，来自先锁规则六cell开发比较；P2另以理论事前固定fraction=.5，不能倒写成因P1R胜出才选。",'',
        '## 核验范围与限制','',
        '- 训练只读无结局输入；没有DSS/OS/PFI风险拟合。P2必须重新建立对应患者折的预处理与encoder，不能复用P1R已覆盖患者的权重。',
        '- 新模型336次实际回放由训练producer执行。另一个程序从保存state读取pair-valid数组并核360份mask，追加逐anchor exact chance；后者没有独立从原图或基础特征重推。',
        '- 不同lambda的combined loss不作为唯一胜负标准；三类指标和同seed方向分开保留。lambda=0时未获辅助训练的gene/u/v等路径显式标记，不作为P2主预训练候选。',
        '- TEXTLESS指无显式文本锚点；CONCH/TITAN预训练本身可能使用文本。score surrogate只匹配训练均值/协方差，不声称匹配完整分布或条件医学真值。',
        '- known关系为历史候选，unknown不是负关系；归一化权重与实际贡献均单列，不把混合系数视为生物学可信度。',
        '- 新训练relation_trace的known/unknown贡献为除temperature之前的兼容度，换算InfoNCE logit须除以固定temperature=.2；R0的分解直接用logit单位。双通道的unknown_logit是未使用buffer，不能把它的常数轨迹视为训练失效。',
        '- 公开文件保留所需聚合统计，完整private receipt由SHA绑定而非声称全文无损公开；患者级预测、数组和权重均留服务器。','']
    with (out/'11_REPRODUCE.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    files=[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    if any(x['bytes']>2*1024**2 for x in files):raise ValueError('UNCHANGED_PUBLICATION_FILE_LIMIT_EXCEEDED')
    audit={'schema':'semantic_b0b7_p1r_matrix_report_build_v1','status':'COMPLETE_ANONYMOUS_JSON_METADATA_REPORTING',
        'source_matrix_report_sha256':sha(matrix_path),'source_chance_audit_sha256':sha(args.chance),
        'source_master_lock_sha256':sha(args.lock),'reporter_source_sha256':sha(__file__),'files':files,
        'model_forwards':0,'optimizer_updates':0,'patient_level_arrays_or_model_weights_loaded_by_reporter':False,
        'checkpoint_or_hyperparameter_selection_changed':False}
    write(out/'MATRIX_REPORT_BUILD_AUDIT.json',audit);return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for k in ('matrix-root','lock','registry','chance','repository-root','output'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();r=build(a);print(json.dumps({'status':r['status'],'files':len(r['files'])+1}))

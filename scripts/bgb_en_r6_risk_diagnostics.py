"""Metadata-only R6 risk/gradient/budget diagnosis after complete replay.

export reads only explicit COMPLETE.json paths after the 90-model/18-reference
report gate. analyze reads the anonymous whitelist and report. Neither mode
imports torch, loads checkpoints/arrays/labels, selects epochs/scales, changes
the source decision, or chooses a new model. Missing historical fields stay
NOT_SAVED rather than being reconstructed by a new forward or backward.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ARMS=('B2','B3','B4','B5','B6','B7','B8','B9','B11','B13')
SEEDS,PARTITIONS,EPOCHS=(31,47,83),(0,1,2),(20,40,80)
CONTROLS=('matched_query','wrong_query','mean_query','interaction_off','value_off','attention_permutation','whole_omics')
SCALARS=('delta','main_delta','cross_delta','path_main_delta','gene_main_delta','hallmark_main_delta',
         'path_cross_delta','gene_cross_delta','delta_E_L2','delta_G_L2')
STATS=('n','mean','sd','min','q10','median','q90','max')
METRICS=('Harrell','Uno60','Harrell_num','Harrell_den','Uno_num','Uno_den',
         'repair','damage','Uno_repair','Uno_damage','risk_shift_sd_ratio')
CURVE_SCALARS=('cox_loss','parameter_regularizer','total_risk_penalty','cross_risk_penalty','gradient_norm',
    'risk_cache_replay_max','cross_cache_replay_max','train_SD_delta_over_offset','valid_SD_delta_over_offset')
GROUPS=('core.spatial_adapter','core.base_queries','core.query_condition','core.gene_rereader',
        'core.path_main','core.gene_main','core.hallmark_main','core.path_cross','core.gene_cross','hallmark.rna')
CROSS_ONLY_GROUPS=('core.base_queries','core.query_condition','core.gene_rereader','core.path_cross','core.gene_cross')
PROBE_SCOPE='first_train_microbatch_contribution_to_full_riskset_Cox_no_regularizers'
ROOT=Path(__file__).resolve().parents[1]
PUBLIC_FILE_LIMIT=2*1024**2


def require(value,code):
    if not bool(value):raise ValueError(code)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def digest(value):return isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef')


def planned():return {(arm,p,s) for arm in ARMS for p in PARTITIONS for s in SEEDS}
def key(row):return row['arm'],row['partition'],row['seed']


def complete_report(path):
    """Gate before any COMPLETE directory access; performance signs ignored."""
    report=read(path)
    require(report.get('schema')=='bgb_en_r6_target_inner_report_v1' and digest(report.get('contract_sha256')),
            'R6_RISK_REPORT_SCHEMA_OR_CONTRACT_INVALID')
    require(report.get('actual_replay',{}).get('complete') is True
            and report['actual_replay'].get('status')=='PASS_COMPLETE_R6_TARGET_CHECKPOINT_REPLAY'
            and digest(report['actual_replay'].get('sha256'))
            and report.get('qualification',{}).get('passed') is True
            and report.get('source',{}).get('complete_source_and_full_transfer') is True,
            'R6_COMPLETE_SOURCE_QUALIFICATION_AND_REPLAY_REQUIRED_BEFORE_DIAGNOSTIC_IO')
    ledger=report.get('ledger',[])
    require(len(ledger)==108 and len({key(r) for r in ledger})==108,'R6_RISK_EXACT_108_LEDGER_ROWS_REQUIRED')
    cells={key(r):r for r in ledger if r['arm'] in ARMS}
    refs={key(r):r for r in ledger if r['arm'] in ('B0','B1')}
    expected_refs={(a,p,s) for a in ('B0','B1') for p in PARTITIONS for s in SEEDS}
    require(set(cells)==planned() and set(refs)==expected_refs,'R6_RISK_LOCKED_MATRIX_REQUIRED')
    require(all(r.get('status')=='PASS_FIT_AND_ACTUAL_REPLAY' and r.get('actual_replay_passed') is True
                and r.get('actual_optimizer_updates')==80 and r.get('actual_new_optimizer_fits')==1
                and digest(r.get('receipt_sha256')) for r in cells.values())
            and all(r.get('status')=='PASS_ZERO_FIT_REFERENCE' and r.get('actual_replay_passed') is True
                and r.get('actual_optimizer_updates')==r.get('actual_new_optimizer_fits')==0 for r in refs.values()),
            'R6_RISK_90_MODELS_18_REFERENCE_ACTUAL_PASS_REQUIRED')
    require(report.get('confirmed_new_target_fits')==90 and report.get('confirmed_target_updates')==7200
            and report.get('unknown_cost_units')==0 and report.get('direct_pair_audit',{}).get('complete') is True,
            'R6_RISK_COMPLETE_ACCOUNTING_AND_PAIRED_REPLAY_REQUIRED')
    require(report.get('outer_results',{}).get('status')=='NOT_RUN'
            and report.get('decision',{}).get('outer_start_allowed') is False,
            'R6_RISK_DIAGNOSTICS_ARE_INNER_ONLY')
    return report,cells


def metric(value):
    if value is None:return None
    require(isinstance(value,dict),'R6_RISK_METRIC_NOT_MAPPING')
    out={k:value[k] for k in METRICS if k in value}
    require(all(v is None or finite(v) for v in out.values()),'R6_RISK_METRIC_NONNUMERIC')
    for label,num,den in (('Harrell','Harrell_num','Harrell_den'),('Uno60','Uno_num','Uno_den')):
        require(finite(out.get(num)) and finite(out.get(den)) and out[den]>=0 and 0<=out[num]<=out[den]+1e-10,
                'R6_RISK_METRIC_COMPONENT_INVALID')
        target=out[num]/out[den] if out[den] else None
        require(out.get(label) is None if target is None else finite(out.get(label)) and abs(out[label]-target)<=1e-10,
                'R6_RISK_METRIC_RATIO_INVALID')
    return out


def quantities(value):
    result={}
    for name in SCALARS:
        raw=value.get(name)
        if raw is None:continue
        require(isinstance(raw,dict),'R6_RISK_QUANTITY_NOT_MAPPING')
        selected={k:raw[k] for k in STATS if k in raw}
        require(type(selected.get('n')) is int and selected['n']>0
                and all(finite(selected.get(k)) for k in STATS if k!='n') and selected['sd']>=0,
                'R6_RISK_QUANTITY_NONFINITE')
        result[name]=selected
    return result


def gradient_probe(raw):
    if raw is None:return None
    require(raw.get('scope')==PROBE_SCOPE and type(raw.get('patients')) is int and 0<raw['patients']<=8
            and raw.get('new_optimizer_updates')==0,'R6_RISK_PURE_COX_PROBE_SCOPE_CHANGED')
    paths=raw.get('groups_by_path',{})
    require(set(paths)=={'cross_delta','risk'},'R6_RISK_TWO_GRADIENT_PATHS_REQUIRED')
    require(set(paths['cross_delta'])==set(paths['risk']) and set(paths['risk'])<=set(GROUPS),
            'R6_RISK_UNEXPECTED_GRADIENT_GROUP')
    selected={}
    for path in ('cross_delta','risk'):
        selected[path]={}
        for group in GROUPS:
            if group not in paths[path]:continue
            row=paths[path][group]
            require(type(row.get('tensors')) is int and row['tensors']>0
                    and type(row.get('nonzero')) is int and 0<=row['nonzero']<=row['tensors']
                    and finite(row.get('L2_squared')) and row['L2_squared']>=0,'R6_RISK_GRADIENT_SUMMARY_INVALID')
            selected[path][group]={k:row[k] for k in ('tensors','nonzero','L2_squared')}
    return dict(scope=PROBE_SCOPE,patients=raw['patients'],groups_by_path=selected,new_optimizer_updates=0)


def allowlisted_receipt(raw,arm,part,seed):
    require(raw.get('status')=='COMPLETE_R6_TARGET_FIT' and key(raw)==(arm,part,seed)
            and raw.get('epochs')==raw.get('optimizer_updates')==80,'R6_RISK_COMPLETE_IDENTITY_CHANGED')
    curves=[]
    for row in raw.get('curves',[]):
        require(row.get('epoch') in EPOCHS and row.get('optimizer_updates_completed')==row['epoch'],
                'R6_RISK_FIXED_BUDGET_CHANGED')
        small={k:row[k] for k in CURVE_SCALARS if k in row}
        require(all(finite(v) for v in small.values()),'R6_RISK_CURVE_NONFINITE')
        curves.append(dict(epoch=row['epoch'],optimizer_updates_completed=row['epoch'],**small,
                           train=metric(row.get('train')),valid=metric(row.get('valid'))))
    require([r['epoch'] for r in curves]==list(EPOCHS),'R6_RISK_THREE_PRELOCKED_SNAPSHOTS_REQUIRED')
    magnitudes={}
    for split in ('train','validation'):
        value=raw.get('risk_diagnostics',{}).get(split,{})
        ratios={k:value[k] for k in ('SD_total_delta_over_offset','SD_cross_delta_over_offset') if k in value}
        require(all(finite(v) and v>=0 for v in ratios.values()),'R6_RISK_FINAL_SD_RATIO_INVALID')
        magnitudes[split]=dict(quantities=quantities(value.get('quantities',{})),**ratios)
    controls={}
    for control in CONTROLS:
        row=raw.get('frozen_controls',{}).get(control)
        if row is None:continue
        require(row.get('frozen_not_retrained') is True and finite(row.get('risk_SD_ratio'))
                and row['risk_SD_ratio']>=0,'R6_RISK_FROZEN_CONTROL_INVALID')
        controls[control]=dict(risk_SD_ratio=row['risk_SD_ratio'],frozen_not_retrained=True,
            whole_modality_replaced=row.get('whole_modality_replaced') is True,
            metrics_vs_base=metric(row.get('metrics_vs_base')),relative_to_trained=metric(row.get('relative_to_trained')))
    return dict(arm=arm,partition=part,seed=seed,curves=curves,risk_diagnostics=magnitudes,
        gradient_probe=gradient_probe(raw.get('gradient_probe')),frozen_controls=controls,
        metrics_vs={name:metric(raw.get('metrics_vs',{}).get(name)) for name in ('CP_STRONG','CP_SCALAR')})


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'R6_RISK_DIAGNOSTIC_EXISTING_OUTPUT_PRESERVED')
    with path.open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')


def export_receipts(report_path,complete_root,output_path):
    report,cells=complete_report(report_path)  # MUST precede COMPLETE directory IO.
    output_path=Path(output_path).resolve()
    require(not any(output_path.is_relative_to(p.resolve()) for p in (ROOT/'reports/bgb_en',ROOT/'reports/bgb_en_r5'))
            and output_path!=Path(report_path).resolve(),'R6_RISK_PROTECTED_HISTORICAL_NAMESPACE')
    require(not output_path.exists(),'R6_RISK_DIAGNOSTIC_EXISTING_OUTPUT_PRESERVED')
    rows=[]
    for arm in ARMS:
        for part in PARTITIONS:
            for seed in SEEDS:
                path=Path(complete_root)/f'{arm}_p{part}_seed{seed}'/'COMPLETE.json'
                original_sha=sha(path)
                require(original_sha==cells[(arm,part,seed)]['receipt_sha256'],'R6_RISK_ORIGINAL_COMPLETE_SHA_CHANGED')
                raw=read(path)
                require(raw.get('contract',{}).get('contract_sha256')==report['contract_sha256'],
                        'R6_RISK_COMPLETE_CONTRACT_CHANGED')
                row=allowlisted_receipt(raw,arm,part,seed)
                row['original_COMPLETE_sha256']=original_sha
                rows.append(row)
    result=dict(schema='bgb_en_r6_anonymous_risk_receipts_v1',status='PASS_90_ALLOWLISTED_ANONYMOUS_RECEIPTS',
        contract_sha256=report['contract_sha256'],report_sha256=sha(report_path),
        qualification_sha256=report['qualification']['sha256'],actual_replay_sha256=report['actual_replay']['sha256'],
        exporter_code_sha256=sha(__file__),units=rows,unit_count=90,
        new_optimizer_fits=0,new_optimizer_updates=0,new_model_forwards=0,new_backward_passes=0,
        model_or_array_or_label_files_read=False,paths_source_origin_and_participant_arrays_exported=False,
        unsaved_fields=['20/40 main/cross/directional risk SD','20/40 pure Cox path gradients',
                        'cross/main/shared-parameter gradient inner products','patient-level risk covariance'])
    write_json(output_path,result)
    return result


def summary(values,expected=9):
    observed=[v for v in values if finite(v)]
    good=len(observed)==expected
    return dict(available=len(observed),expected=expected,mean=mean(observed) if good else None,
        median=median(observed) if good else None,min=min(observed) if good else None,
        max=max(observed) if good else None,exact_zero_count=sum(v==0 for v in observed))


def component_totals(records,split):
    """Same within-partition concordance aggregation, not an average of C."""
    result=[]
    for seed in SEEDS:
        values=[r.get(split) for r in records if r['seed']==seed]
        if len(values)!=3 or any(v is None for v in values):return None
        parts={name:sum(v[name] for v in values) for name in ('Harrell_num','Harrell_den','Uno_num','Uno_den')}
        result.append(dict(seed=seed,**parts,Harrell=parts['Harrell_num']/parts['Harrell_den'] if parts['Harrell_den'] else None,
            Uno60=parts['Uno_num']/parts['Uno_den'] if parts['Uno_den'] else None))
    return dict(per_seed=result,Harrell=mean(r['Harrell'] for r in result) if all(finite(r['Harrell']) for r in result) else None,
                Uno60=mean(r['Uno60'] for r in result) if all(finite(r['Uno60']) for r in result) else None)


def final_magnitude(raw):
    q=raw.get('quantities',{})
    sd=lambda name:q.get(name,{}).get('sd')
    scale=None;scale_from=None
    for name,ratio_name in (('delta','SD_total_delta_over_offset'),('cross_delta','SD_cross_delta_over_offset')):
        if finite(sd(name)) and sd(name)>0 and finite(raw.get(ratio_name)) and raw[ratio_name]>0:
            scale=sd(name)/raw[ratio_name];scale_from=ratio_name;break
    result=dict(effective_anchor_scale_inferred=scale,inference_source=scale_from,
        SD_total_over_anchor=raw.get('SD_total_delta_over_offset'),SD_cross_over_anchor=raw.get('SD_cross_delta_over_offset'))
    for name in ('main_delta','path_main_delta','gene_main_delta','hallmark_main_delta','path_cross_delta','gene_cross_delta'):
        result['SD_'+name]=sd(name)
        result['SD_'+name+'_over_anchor']=sd(name)/scale if finite(sd(name)) and finite(scale) and scale>0 else None
    result['SD_cross_over_main']=sd('cross_delta')/sd('main_delta') if finite(sd('cross_delta')) and finite(sd('main_delta')) and sd('main_delta')>0 else None
    result['directional_risk_means']={name:q.get(name,{}).get('mean') for name in ('path_cross_delta','gene_cross_delta')}
    result['original_update_L2_means']={name:q.get(name,{}).get('mean') for name in ('delta_E_L2','delta_G_L2')}
    result['SD_ratios_are_not_additive_variance_or_causal_contribution_fractions']=True
    return result


def gradient_values(probe):
    if probe is None:return None
    result={}
    for path,groups in probe['groups_by_path'].items():
        total=sum(v['L2_squared'] for v in groups.values())
        result[path]={name:dict(L2=math.sqrt(v['L2_squared']),L2_squared=v['L2_squared'],
            tensors=v['tensors'],nonzero_tensors=v['nonzero'],
            nonzero_tensor_fraction=v['nonzero']/v['tensors'],
            squared_norm_share_within_path=v['L2_squared']/total if total>0 else None)
            for name,v in groups.items()}
    result['exclusive_cross_group_norm_differences']={g:result['risk'][g]['L2']-result['cross_delta'][g]['L2']
        for g in CROSS_ONLY_GROUPS if g in result['risk']}
    result['main_gradient_not_inferred_by_subtraction']=True
    result['scope']=probe['scope'];result['patients']=probe['patients'];result['new_optimizer_updates']=0
    return result


def frozen_values(raw,primary):
    result={}
    for name,row in raw.items():
        control,tr=row.get('metrics_vs_base'),row.get('relative_to_trained')
        value=dict(risk_SD_difference_over_anchor=row['risk_SD_ratio'],
                   frozen_not_retrained=True,whole_modality_replaced=row['whole_modality_replaced'])
        if primary is not None and control is not None and tr is not None:
            require(all(abs(primary[k]-control[k])<=1e-10 for k in ('Harrell_den','Uno_den')),
                    'R6_RISK_FROZEN_CONTROL_SCOPE_CHANGED')
            require(all(abs(primary[k]-tr[k])<=1e-10 for k in ('Harrell_num','Harrell_den','Uno_num','Uno_den')),
                    'R6_RISK_FROZEN_TRAINED_SCORE_CHANGED')
            for label in ('Harrell','Uno60'):
                value['trained_minus_control_'+label]=primary[label]-control[label] if finite(primary[label]) and finite(control[label]) else None
            for repair,damage,label in (('repair','damage','Harrell'),('Uno_repair','Uno_damage','Uno60')):
                value['trained_minus_control_'+repair]=tr.get(repair)
                value['trained_minus_control_'+damage]=tr.get(damage)
                if finite(tr.get(repair)) and finite(tr.get(damage)) and finite(value['trained_minus_control_'+label]):
                    require(abs(tr[repair]-tr[damage]-value['trained_minus_control_'+label])<=1e-10,
                            'R6_RISK_DIRECT_FROZEN_REPAIR_DAMAGE_IDENTITY_FAILED')
        result[name]=value
    return result


def analyze(report_path,whitelist_path,output):
    report,cells=complete_report(report_path)
    white=read(whitelist_path)
    require(white.get('schema')=='bgb_en_r6_anonymous_risk_receipts_v1'
            and white.get('status')=='PASS_90_ALLOWLISTED_ANONYMOUS_RECEIPTS'
            and white.get('contract_sha256')==report['contract_sha256']
            and white.get('report_sha256')==sha(report_path)
            and white.get('qualification_sha256')==report['qualification']['sha256']
            and white.get('actual_replay_sha256')==report['actual_replay']['sha256']
            and white.get('exporter_code_sha256')==sha(__file__),
            'R6_RISK_ANONYMOUS_WHITELIST_ORIGIN_CHANGED')
    raw_rows=white.get('units',[])
    require(len(raw_rows)==90 and {key(r) for r in raw_rows}==planned(),'R6_RISK_EXACT_90_ANONYMOUS_UNITS_REQUIRED')
    rows=[];curves=defaultdict(list)
    for raw in raw_rows:
        arm,part,seed=key(raw)
        require(raw.get('original_COMPLETE_sha256')==cells[(arm,part,seed)]['receipt_sha256'],
                'R6_RISK_WHITELIST_COMPLETE_SHA_NOT_IN_LEDGER')
        # Reapply the allowlist on anonymous input too; never copy unknown fields.
        row=allowlisted_receipt(dict(raw,status='COMPLETE_R6_TARGET_FIT',epochs=80,optimizer_updates=80),arm,part,seed)
        final={split:final_magnitude(row['risk_diagnostics'][split]) for split in ('train','validation')}
        grad=gradient_values(row['gradient_probe'])
        for curve in row['curves']:
            curves[(arm,curve['epoch'])].append(dict(curve,seed=seed,partition=part))
        rows.append(dict(arm=arm,partition=part,seed=seed,
            original_COMPLETE_sha256=raw['original_COMPLETE_sha256'],final_epoch=80,
            reverse_mode='token' if arm=='B5' else 'none' if arm in ('B2','B3','B4','B11') else 'original_gene',
            trained_query_mode='wrong_query' if arm=='B9' else 'matched_query',
            cross_structurally_disabled=arm in ('B2','B3'),final_magnitude=final,gradient_probe=grad,
            saved_scalar_distributions=row['risk_diagnostics'],budget_snapshots=row['curves'],
            frozen_interventions=frozen_values(row['frozen_controls'],row['metrics_vs'].get('CP_SCALAR'))))
    learning=[]
    for arm in ARMS:
        for epoch in EPOCHS:
            values=curves[(arm,epoch)]
            require(len(values)==9,'R6_RISK_BUDGET_GRID_MISSING')
            learning.append(dict(arm=arm,epoch=epoch,
                train_ranking=component_totals(values,'train'),inner_validation_ranking=component_totals(values,'valid'),
                scalar_summaries={name:summary([v.get(name) for v in values]) for name in CURVE_SCALARS},
                main_cross_directional_SD_at_this_epoch='AVAILABLE_IN_FINAL_RECEIPTS_ONLY' if epoch==80 else 'NOT_SAVED',
                best_epoch_not_selected=True))
    arm_summaries=[]
    for arm in ARMS:
        selected=[r for r in rows if r['arm']==arm]
        final={split:{name:summary([r['final_magnitude'][split].get(name) for r in selected])
                      for name in ('SD_total_over_anchor','SD_cross_over_anchor','SD_main_delta_over_anchor',
                          'SD_path_cross_delta_over_anchor','SD_gene_cross_delta_over_anchor','SD_cross_over_main')}
               for split in ('train','validation')}
        paths={}
        for path in ('cross_delta','risk'):
            paths[path]={}
            for group in GROUPS:
                available=[r['gradient_probe'][path][group] for r in selected
                           if r['gradient_probe'] is not None and group in r['gradient_probe'][path]]
                probe_count=sum(r['gradient_probe'] is not None for r in selected)
                status='STRUCTURAL_NOT_TRAINABLE' if probe_count==9 and not available else 'COMPLETE_9_PROBES' if len(available)==9 else 'MISSING_PROBE_COVERAGE'
                paths[path][group]=dict(status=status,
                    L2=summary([v['L2'] for v in available]),nonzero_tensor_fraction=summary([v['nonzero_tensor_fraction'] for v in available]),
                    squared_norm_share_within_path=summary([v['squared_norm_share_within_path'] for v in available]))
        arm_summaries.append(dict(arm=arm,final_magnitude=final,gradient_paths=paths,
            probe_available_cells=sum(r['gradient_probe'] is not None for r in selected),
            correction_risk_SD_ratio_is_not_predictive_variance_explained=True))
    missing=sum(r['gradient_probe'] is None for r in rows)
    result=dict(schema='bgb_en_r6_risk_diagnostics_v1',
        status='COMPLETE_SAVED_METADATA_DIAGNOSTICS' if missing==0 else 'PARTIAL_SAVED_METADATA_DIAGNOSTIC_COVERAGE',
        contract_sha256=report['contract_sha256'],report_sha256=sha(report_path),
        anonymous_whitelist_sha256=sha(whitelist_path),diagnostic_code_sha256=sha(__file__),
        units=rows,arm_summaries=arm_summaries,learning_curves=learning,
        coverage=dict(formal_cells=90,gradient_probe_cells=90-missing,
            curves_per_cell=[20,40,80],directional_risk_and_gradient_epoch=80,
            main_cross_directional_SD_epochs20_40='NOT_SAVED',gradient_epochs20_40='NOT_SAVED',
            shared_parameter_gradient_inner_products='NOT_SAVED',patient_level_covariance='NOT_SAVED'),
        existing_direct_pair_comparisons_canonical_sha256=canonical(report.get('paired_comparisons',{})),
        existing_rank_credit_analysis_recomputed_or_replaced=False,
        new_optimizer_fits=0,new_optimizer_updates=0,new_model_forwards=0,new_backward_passes=0,
        model_or_array_or_label_files_read=False,epoch_or_scale_or_model_selection=False,
        interpretations=dict(pure_Cox_scope=PROBE_SCOPE,probe_patients_per_cell_max=8,
            gradient_probe_is_not_full_patient_cohort_gradient=True,
            full_minus_cross_gradient_norm_is_not_main_gradient=True,
            normalized_risk_SD_not_additive_variance_contribution=True,
            group_gradient_norms_parameterization_dependent=True,
            frozen_controls_dependency_not_retrained_necessity=True,
            train_anchor_crossfit_validation_anchor_fullfit_not_same_distribution=True,
            step_loss_and_penalties_pre_update_snapshot_metrics_post_update=True,
            attention_diagnosis_reuses_existing_report_summaries_only=True,
            no_threshold_for_declaring_main_effect_suppression=True,
            low_cross_SD_alone_does_not_establish_no_ranking_information=True))
    output=Path(output).resolve()
    require(not any(output.is_relative_to(p.resolve()) for p in (ROOT/'reports/bgb_en',ROOT/'reports/bgb_en_r5')),
            'R6_RISK_PROTECTED_HISTORICAL_NAMESPACE')
    require(not any((output/name).exists() for name in ('R6_RISK_DIAGNOSTICS.json','R6_RISK_DIAGNOSTICS.md')),
            'R6_RISK_DIAGNOSTIC_EXISTING_OUTPUT_PRESERVED')
    write_diagnostic_bundle(output,result)
    write_markdown(output/'R6_RISK_DIAGNOSTICS.md',result)
    return result


def encoded_size(value):
    return len((json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))


def write_diagnostic_bundle(output,result):
    """Single file when it fits; otherwise lossless arm parts, never field pruning."""
    output=Path(output);index=output/'R6_RISK_DIAGNOSTICS.json'
    if encoded_size(result)<=PUBLIC_FILE_LIMIT:
        write_json(index,result);return
    parts=[];objects=[]
    for arm in ARMS:
        rows=[row for row in result['units'] if row['arm']==arm]
        require(len(rows)==9,'R6_RISK_ARM_CHUNK_MISSING_ROWS')
        obj=dict(schema='bgb_en_r6_risk_diagnostic_arm_v1',contract_sha256=result['contract_sha256'],arm=arm,units=rows)
        require(encoded_size(obj)<=PUBLIC_FILE_LIMIT,'R6_RISK_ARM_CHUNK_OVER_2MIB')
        path=output/f'R6_RISK_DIAGNOSTICS_{arm}.json'
        require(not path.exists(),'R6_RISK_DIAGNOSTIC_EXISTING_OUTPUT_PRESERVED')
        objects.append((path,obj))
        parts.append(dict(file=path.name,arm=arm,count=len(rows),sha256=hashlib.sha256((json.dumps(obj,sort_keys=True,
            separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')).hexdigest()))
    small=dict(schema='bgb_en_r6_risk_diagnostic_index_v1',logical_sha256=canonical(result),unit_count=90,parts=parts,
        logical_metadata={key:value for key,value in result.items() if key!='units'})
    require(encoded_size(small)<=PUBLIC_FILE_LIMIT,'R6_RISK_INDEX_OVER_2MIB')
    for path,obj in objects:write_json(path,obj)
    write_json(index,small)


def read_diagnostic_bundle(path):
    """Root may roundtrip either representation without losing a single field."""
    path=Path(path);value=read(path)
    if value.get('schema')=='bgb_en_r6_risk_diagnostics_v1':return value
    require(value.get('schema')=='bgb_en_r6_risk_diagnostic_index_v1' and value.get('unit_count')==90,
            'R6_RISK_DIAGNOSTIC_INDEX_INVALID')
    parts=value.get('parts',[])
    require(len(parts)==10 and {row['arm'] for row in parts}==set(ARMS),'R6_RISK_DIAGNOSTIC_PART_MATRIX_CHANGED')
    units=[]
    for row in parts:
        require(row['file']==f"R6_RISK_DIAGNOSTICS_{row['arm']}.json" and row['count']==9,
                'R6_RISK_DIAGNOSTIC_PART_PATH_OR_COUNT_INVALID')
        part_path=path.parent/row['file']
        require(sha(part_path)==row['sha256'],'R6_RISK_DIAGNOSTIC_PART_SHA_CHANGED')
        part=read(part_path)
        require(part.get('schema')=='bgb_en_r6_risk_diagnostic_arm_v1' and part.get('arm')==row['arm']
                and part.get('contract_sha256')==value['logical_metadata']['contract_sha256']
                and len(part.get('units',[]))==9 and all(x['arm']==row['arm'] for x in part['units']),
                'R6_RISK_DIAGNOSTIC_PART_SCHEMA_CHANGED')
        units.extend(part['units'])
    require({key(row) for row in units}==planned(),'R6_RISK_DIAGNOSTIC_PART_UNIT_MATRIX_CHANGED')
    # The producer's units use fixed arm/partition/seed order; do not reorder facts.
    units=sorted(units,key=lambda row:(ARMS.index(row['arm']),row['partition'],SEEDS.index(row['seed'])))
    result=dict(value['logical_metadata'],units=units)
    require(canonical(result)==value['logical_sha256'],'R6_RISK_DIAGNOSTIC_LOGICAL_OBJECT_CHANGED')
    return result


def write_markdown(path,result):
    def fmt(value):return 'NA' if not finite(value) else f'{value:.6g}'
    lines=['# R6 既有风险、梯度与训练预算诊断','',f"状态：**{result['status']}**。90模型与18参考完整实际回放后，仅解析匿名报告/白名单收据。新增拟合、前向、反向均为0。",'',
        '## 最终风险幅度','',
        '下表是9个cell的描述性均值，不是C-index聚合。风险SD比不是方差贡献分解；两方向可能相关或抵消。没有用幅度比挑模型或缩放。','',
        '| 臂 | valid 总修正/anchor SD | valid 主效应/anchor SD | valid 交互/anchor SD | P方向/anchor SD | G方向/anchor SD | 交互SD/主效应SD |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['arm_summaries']:
        v=row['final_magnitude']['validation']
        fields=('SD_total_over_anchor','SD_main_delta_over_anchor','SD_cross_over_anchor','SD_path_cross_delta_over_anchor','SD_gene_cross_delta_over_anchor','SD_cross_over_main')
        lines.append('| '+row['arm']+' | '+' | '.join(fmt(v[k]['mean']) for k in fields)+' |')
    lines+=['','## 纯Cox路径梯度','',
        '梯度仅为固定首8训练患者对完整训练风险集Cox目标的微批贡献，未混入L2。下表按组给出9个cell的L2范数中位数；不同参数化/参数数量不可当成因果重要性排序。full/cross范数不能相减得到main梯度。','',
        '| 臂 | query条件 cross | reverse Q cross | spatial cross | spatial full | Hallmark main full |','|---|---:|---:|---:|---:|---:|']
    for row in result['arm_summaries']:
        g=row['gradient_paths'];values=[g['cross_delta']['core.query_condition']['L2']['median'],
            g['cross_delta']['core.gene_rereader']['L2']['median'],g['cross_delta']['core.spatial_adapter']['L2']['median'],
            g['risk']['core.spatial_adapter']['L2']['median'],g['risk']['hallmark.rna']['L2']['median']]
        lines.append('| '+row['arm']+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','## 预算与数据覆盖','',
        '- JSON保留20/40/80的train/inner-valid总修正SD比、Cox/L2/风险惩罚、验证排序。每seed先汇总三分区分子/分母，再平均三个seed；没有挑最优epoch。',
        '- 每轮step记录的Cox/loss/惩罚在该次optimizer update前计算，快照train/valid排序在update后计算；二者不是严格同一次参数状态。',
        '- 20/40的main/cross/方向风险SD与分组纯Cox梯度未被原收据保存，明确标NOT_SAVED；不加载checkpoint补造，也不把最终值回填早期轮。',
        '- 最终epoch80的两方向风险分量、全部10个允许参数组的cross/full梯度、每cell非零张量数和冻结控制直接修复/损伤均在JSON中。',
        '- frozen relative_to_trained的修复/损伤方向是“已训练风险减控制风险”。B9默认以错误query训练；matched_query干预不能被误叫其原训练路径。whole_omics还换RNA主效应/offset，不是纯query检验。',
        '- B2/B3交互关闭、B4/B11逆向关闭属于结构性不激活；不存在的参数组是NA而非梯度失败。',
        '- 既有B7等真实重训配对归因由原完整报告引用且记录canonical SHA，本脚本不重复或替换其排名、repair/damage与门槛。注意力权重另由固定样本审查处理，这里不复制原权重。','',
        '## 能回答与不能回答','',
        '可观察交互风险是否远小于主效应、query/重读组是否获得纯Cox梯度，以及关闭query/局部Value后的实际风险与排序变化是否一致。只报告数值，不用事后幅度阈值宣布“淹没”。',
        '单一最终快照、首8患者梯度和相关风险分量不能确定80轮中的因果训练瓶颈；缺少共享参数梯度内积时不能恢复main梯度，缺少患者级协方差时不能做方差归因。train交叉拟合anchor与valid完整fit anchor分布不同，不凭两者C或loss差异宣布欠拟合。','']
    Path(path).write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('export','analyze'),required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--whitelist',type=Path,required=True)
    parser.add_argument('--complete-root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='export':
        require(args.complete_root is not None,'R6_RISK_EXPORT_REQUIRES_EXPLICIT_COMPLETE_ROOT')
        result=export_receipts(args.report,args.complete_root,args.whitelist)
    else:
        result=analyze(args.report,args.whitelist,args.output or args.report.parent)
    print(json.dumps(dict(status=result['status'],new_optimizer_fits=0,new_model_forwards=0,new_backward_passes=0)))


if __name__=='__main__':main()

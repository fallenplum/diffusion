"""Recompute anonymous R6 source reports, without models, private data or torch.

Outputs are new R6_SOURCE_INDEPENDENT_REVIEW.json/.md artifacts. This audit
does not overwrite the source decision, select a new source family, reproduce
model training, or authorize target/outer experiments.
"""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ('iCMS2','iCMS3','FAP_fibroblast_associated_top25','MARCO_macrophage_associated_top25')
SEEDS = (31,47)
FAMILIES = (('S0','MATCHED'),('S1','MATCHED'),('S2','MATCHED'),('S3','MATCHED'),
    ('S4','MATCHED'),('S5','WRONG_DONOR'),('S4','WITHIN_SECTION_TISSUE_DEPTH'),
    ('S4','RANDOM_GATE'),('S6','MATCHED'),('S6','WRONG_NEIGHBOR'),
    ('S7','MATCHED'),('S7','WRONG_SPOT_ALIGNMENT'))
SPECS = {
    'S4_vs_S3_ADDITIVE': (('S4','MATCHED'),('S3','MATCHED')),
    'S4_vs_S4_RANDOM_GATE': (('S4','MATCHED'),('S4','RANDOM_GATE')),
    'S4_vs_S4_WITHIN_SECTION_TISSUE_DEPTH': (('S4','MATCHED'),('S4','WITHIN_SECTION_TISSUE_DEPTH')),
    'S4_vs_S5_WRONG_DONOR': (('S4','MATCHED'),('S5','WRONG_DONOR')),
    'S6_vs_S4_NO_NEIGHBOR': (('S6','MATCHED'),('S4','MATCHED')),
    'S6_vs_S6_WRONG_NEIGHBOR': (('S6','MATCHED'),('S6','WRONG_NEIGHBOR')),
    'S7_vs_S4_NO_GENE_ALIGNMENT': (('S7','MATCHED'),('S4','MATCHED')),
    'S7_vs_S7_WRONG_SPOT_ALIGNMENT': (('S7','MATCHED'),('S7','WRONG_SPOT_ALIGNMENT')),
}


def check(condition, code):
    if not condition:
        raise AssertionError(code)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def close(a,b):
    return a is b if a is None or b is None else abs(a-b)<=1e-12


def ident(row):
    return row['arm'],row['control'],int(row['heldout_donor']),int(row['seed'])


def review(reports, source_config):
    reports=Path(reports)
    report_path=reports/'R6_1_SOURCE_SUPERVISION.json'
    config_path=Path(source_config)
    eligibility_path=reports/'R6_SOURCE_TARGET_ELIGIBILITY.json'
    qualification_path=reports/'R6_REAL_SOURCE_COMPUTE_QUALIFICATION.json'
    report,config,eligibility,qualification=map(load,(report_path,config_path,eligibility_path,qualification_path))
    evidence_files=(report_path,config_path,eligibility_path,qualification_path,
        reports/'R6_SOURCE_CELL_METRICS.csv',reports/'R6_SOURCE_COSTS.csv',
        reports/'R6_SOURCE_DECISION.json',reports/'R6_SOURCE_REPLAY_AUDIT.json',reports/'R6_SOURCE_GATE_AUDIT.csv')
    hashes={p.name:sha(p) for p in evidence_files}
    check(report.get('schema')=='bgb_en_r6_source_supervision_report_v1'
          and config.get('schema')=='bgb_en_r6_source_pretrain_lock_v1'
          and config.get('status')=='LOCKED_BEFORE_SOURCE_FITS','REPORT_AND_LOCK_SCHEMA')
    check(report['source_config_sha256']==sha(config_path),'CONFIG_HASH')
    check(report['target_eligibility_sha256']==sha(eligibility_path),'ELIGIBILITY_HASH')
    check(report['source_code_sha256']==config['source_code_sha256'],'CODE_PIN_MAP')
    check(report['decision']==load(reports/'R6_SOURCE_DECISION.json'),'DECISION_OBJECT')
    report_code=ROOT/'scripts/bgb_en_r6_report_source.py'
    local_report_code_matches=report['report_code_sha256']==sha(report_code) if report_code.is_file() else None
    check(local_report_code_matches is not False,'LOCAL_REPORT_CODE_PIN')
    expected={ident(row) for row in config['cells']}
    locked_matrix={(a,c,d,s) for a,c in FAMILIES for d in range(6) for s in SEEDS}
    locked_matrix|={('S7',c,-1,31) for c in ('MATCHED','WRONG_DONOR','WITHIN_SECTION_TISSUE_DEPTH')}
    check(expected==locked_matrix and len(config['cells'])==147,'FIXED_147_SOURCE_IDENTITIES')
    ledger={ident(row):row for row in report['ledger']}
    check(len(expected)==len(ledger)==len(report['ledger'])==147 and set(ledger)==expected,'MATRIX_147')
    check(all(r['status']=='PASS_COMPLETE_AND_REPLAYED' and r['independent_replay_passed']
              and r['metadata_validated'] for r in ledger.values()),'ALL_REPLAY_FLAGS')
    check(all(r['actual_optimizer_fits']==(0 if r['arm']=='S0' else 1)
              and r['actual_optimizer_updates']==(0 if r['arm']=='S0' else 80)
              for r in ledger.values()),'PER_UNIT_COST')
    fits=sum(r['actual_optimizer_fits'] for r in ledger.values())
    updates=sum(r['actual_optimizer_updates'] for r in ledger.values())
    full=[r for r in ledger.values() if r['heldout_donor']<0]
    lodo=[r for r in ledger.values() if r['heldout_donor']>=0]
    check(fits==report['confirmed_formal_optimizer_fits']==135,'FIT_SUM_135')
    check(updates==report['confirmed_formal_optimizer_updates']==10800,'UPDATE_SUM_10800')
    check(len(full)==3 and len(lodo)==144 and sum(r['actual_optimizer_fits'] for r in full)==3
          and sum(r['arm']=='S0' for r in lodo)==12,'FULL_AND_LODO_SPLIT')
    check(report['unknown_fit_cost_units']==report['unknown_update_cost_units']==0,'UNKNOWN_COST')
    with (reports/'R6_SOURCE_COSTS.csv').open(encoding='utf-8-sig',newline='') as handle:
        cost_rows=list(csv.DictReader(handle))
    check(len(cost_rows)==147 and {ident(r) for r in cost_rows}==expected,'COST_CSV_IDENTITIES')
    check(sum(int(r['actual_optimizer_fits']) for r in cost_rows)==fits
          and sum(int(r['actual_optimizer_updates']) for r in cost_rows)==updates,'COST_CSV_SUMS')
    qual=report['qualification_cost_separate_from_formal_matrix']
    check(qual['receipt_sha256']==sha(qualification_path),'QUALIFICATION_HASH')
    check(qual['optimizer_updates']==qualification['qualification_updates']==2
          and qual['optimizer_instances']==qualification['separate_qualification_optimizer_instances']==1
          and qual['formal_source_fits']==qualification['formal_source_fits']==0,'QUALIFICATION_COST')
    check(qualification['TCGA_DSS_or_PFI_read'] is False and qualification['saved_qualification_weights'] is False
          and qualification['source_test_sections_read'] is False
          and qualification['performance_based_hyperparameter_choice'] is False
          and qualification['config_sha256']==sha(config_path)
          and qualification['source_cache_manifest_sha256']==report['source_cache_manifest_sha256'],
          'QUALIFICATION_BOUNDARY')
    target_groups={(r['heldout_donor'],r['program'],r['tissue_class']):r for r in eligibility['groups']}
    check(len(target_groups)==72 and sum(r['eligible'] for r in target_groups.values())==report['target_eligible_groups'],'GROUP_72')
    with (reports/'R6_SOURCE_CELL_METRICS.csv').open(encoding='utf-8-sig',newline='') as handle:
        metric_rows=list(csv.DictReader(handle))
    grouped=defaultdict(list)
    for row in metric_rows:
        grouped[ident(row)].append(row)
    check(len(metric_rows)==1728 and set(grouped)=={ident(r) for r in lodo},'METRIC_1728_ROWS')
    scores={}
    for key,rows in grouped.items():
        check(len(rows)==12 and {(r['program'],int(r['tissue_class'])) for r in rows}
              =={(p,t) for p in PROGRAMS for t in range(3)},'UNIT_12_GROUPS')
        huber=[];rhos=[];ba={float(r['tissue_BA']) for r in rows}
        check(len(ba)==1,'REPEATED_BA_CONSISTENCY')
        for p in PROGRAMS:
            values={float(r['program_global_Huber']) for r in rows if r['program']==p}
            check(len(values)==1,'REPEATED_PROGRAM_HUBER_CONSISTENCY')
            huber.append(values.pop())
        for row in rows:
            truth=target_groups[(key[2],row['program'],int(row['tissue_class']))]
            check(int(row['support'])==truth['support']
                  and (row['eligible_for_within_tissue_rho']=='True')==truth['eligible'],'FIXED_ELIGIBILITY')
            if truth['eligible']:
                check(row['Spearman']!='' and math.isfinite(float(row['Spearman'])),'NO_DROPPED_UNDEFINED_RHO')
                rhos.append(float(row['Spearman']))
        scores[key]=dict(program_Huber=mean(huber),within_tissue_Spearman=mean(rhos),tissue_BA=ba.pop())
    summaries=[];max_family_error=0.
    check(len(report['complete_family_summaries'])==12 and
          {(r['arm'],r['control']) for r in report['complete_family_summaries']}==set(FAMILIES),'FAMILY_12_IDENTITIES')
    for saved in report['complete_family_summaries']:
        arm,control=saved['arm'],saved['control']
        by_seed=[]
        for seed in SEEDS:
            cells=[scores[(arm,control,donor,seed)] for donor in range(6)]
            value={name:mean(r[name] for r in cells) for name in ('program_Huber','within_tissue_Spearman','tissue_BA')}
            expected_seed=next(r for r in saved['per_seed'] if r['seed']==seed)
            for name in value:
                check(close(value[name],expected_seed[name]),'FAMILY_SEED_RECOMPUTE')
            by_seed.append(dict(seed=seed,**value))
        result={name:mean(r[name] for r in by_seed) for name in ('program_Huber','within_tissue_Spearman','tissue_BA')}
        for name in result:
            check(close(result[name],saved[name]),'FAMILY_TOTAL_RECOMPUTE')
            max_family_error=max(max_family_error,abs(result[name]-saved[name]))
        summaries.append(dict(arm=arm,control=control,per_seed=by_seed,**result))
    comparisons={}
    for name,(true,reference) in SPECS.items():
        saved=report['comparisons'][name];pairs=[]
        for seed in SEEDS:
            for donor in range(6):
                a,b=scores[(*true,donor,seed)],scores[(*reference,donor,seed)]
                pair=dict(seed=seed,heldout_donor=donor,H=a['program_Huber']*-1+b['program_Huber'],
                          rho=a['within_tissue_Spearman']-b['within_tissue_Spearman'])
                original=next(r for r in saved['pairs'] if r['seed']==seed and r['heldout_donor']==donor)
                check(close(pair['H'],original['Huber_negative_minus_true'])
                      and close(pair['rho'],original['within_tissue_rho_true_minus_negative']),'PAIR_RECOMPUTE')
                pairs.append(pair)
        per_seed=[dict(seed=s,H=mean(r['H'] for r in pairs if r['seed']==s),
            rho=mean(r['rho'] for r in pairs if r['seed']==s)) for s in SEEDS]
        donor_mean=[dict(heldout_donor=d,H=mean(r['H'] for r in pairs if r['heldout_donor']==d),
            rho=mean(r['rho'] for r in pairs if r['heldout_donor']==d)) for d in range(6)]
        h,rho=mean(r['H'] for r in per_seed),mean(r['rho'] for r in per_seed)
        check(close(h,saved['Huber_negative_minus_true']) and close(rho,saved['within_tissue_rho_true_minus_negative']),'COMPARISON_TOTAL')
        status='SUPPORTED' if h>0 and rho>0 else 'NOT_SUPPORTED'
        check(status==saved['status'],'DIRECTION_GATE_RECOMPUTE')
        omitted=[dict(omitted_donor=d,H=mean(r['H'] for r in donor_mean if r['heldout_donor']!=d),
            rho=mean(r['rho'] for r in donor_mean if r['heldout_donor']!=d)) for d in range(6)]
        comparisons[name]=dict(H=h,rho=rho,status=status,per_seed=per_seed,pairs=pairs,donor_means=donor_mean,
            positive_H_pairs=sum(r['H']>0 for r in pairs),positive_rho_pairs=sum(r['rho']>0 for r in pairs),
            positive_both_pairs=sum(r['H']>0 and r['rho']>0 for r in pairs),
            positive_both_donor_means=sum(r['H']>0 and r['rho']>0 for r in donor_mean),
            rho_seed_sign_reversal=per_seed[0]['rho']*per_seed[1]['rho']<0,
            omitted_donor_diagnostic=omitted,
            omitted_donor_still_both_positive=sum(r['H']>0 and r['rho']>0 for r in omitted),
            no_omission_used_for_model_selection=True)
    required=('S4_vs_S5_WRONG_DONOR','S4_vs_S4_WITHIN_SECTION_TISSUE_DEPTH')
    gate='SUPPORTED' if all(comparisons[n]['status']=='SUPPORTED' for n in required) else 'NOT_SUPPORTED'
    check(gate==report['decision']['status'],'CORE_SOURCE_GATE')
    random_rows=[]
    for row in lodo:
        if (row['arm'],row['control'])!=('S4','RANDOM_GATE'):
            continue
        g=row['gate_metadata']['heldout']['total']
        check(g['row_probability_multiset_exact'] and g['used_probabilities_follow_reported_permutation']
              and g['between_model_distributions_exactly_matched_claimed'] is False
              and g['confidence_information_removed_claimed'] is False,'RANDOM_NULL_BOUNDARY')
        random_rows.append(dict(seed=row['seed'],heldout_donor=row['heldout_donor'],rows=g['rows'],
            identity_rate=g['identity_rate'],permutation_counts=g['permutation_counts'],
            unique_permutations=g['unique_permutations'],row_entropy_max_abs_difference=g['row_entropy_max_abs_difference'],
            row_multiset_max_abs_difference=g['row_probability_multiset_max_abs_difference'],
            original_perclass_probability_mean=g['original_perclass_probability_mean'],
            used_perclass_probability_mean=g['used_perclass_probability_mean']))
    check(len(random_rows)==12,'RANDOM_GATE_12_HELDOUT')
    random_summary=[]
    for seed in SEEDS:
        rows=[r for r in random_rows if r['seed']==seed]
        total=sum(r['rows'] for r in rows)
        counts={p:sum(r['permutation_counts'][p] for r in rows) for p in rows[0]['permutation_counts']}
        random_summary.append(dict(seed=seed,heldout_spots_summed_once=total,permutation_counts=counts,
            weighted_identity_rate=counts['012']/total,
            max_row_entropy_change=max(r['row_entropy_max_abs_difference'] for r in rows),
            max_row_multiset_change=max(r['row_multiset_max_abs_difference'] for r in rows),
            minimum_donor_unique_permutations=min(r['unique_permutations'] for r in rows)))
    result=dict(schema='r6_anonymous_source_independent_review_v1',status='PASS_ANONYMOUS_ACCOUNTING_AND_AGGREGATION_REVIEW',
        input_file_sha256=hashes,compact_report_canonical_object_sha256=canonical(report),
        auditor_code_sha256=sha(Path(__file__)),local_frozen_report_code_pin_checked=local_report_code_matches,
        no_weights_or_raw_measurements_read=True,no_new_model_forward=True,new_optimizer_fits=0,
        new_optimizer_updates=0,source_replay_is_from_signed_anonymous_receipts_not_new_forward=True,
        costs=dict(logical=147,LODO=144,full_source=3,formal_source_fits=fits,formal_source_updates=updates,
            zero_optimizer_references=12,qualification_optimizer_instances=1,qualification_optimizer_updates=2,
            qualification_formal_source_fits=0,formal_plus_qualification_updates=updates+2),
        eligibility_groups=72,eligible_groups=report['target_eligible_groups'],independent_metric_rows=1728,
        recomputed_source_units=144,maximum_family_aggregation_abs_error=max_family_error,
        family_summaries=summaries,comparisons=comparisons,source_gate_recomputed=gate,
        source_gate_does_not_establish_DSS_or_external_confirmation=True,
        random_gate_per_seed=random_summary,random_gate_per_donor_seed=random_rows,
        qualification_config_sha_matches_formal=qualification.get('config_sha256')==sha(config_path),
        limits=dict(source_donors_are_previous_development_donors=True,source_TCGA_patient_nonoverlap='UNVERIFIED',
            heldout_source_donors_not_new_external_confirmatory_cohort=True,target_outer='NOT_REVIEWED_NOT_RUN_BY_THIS_AUDIT',
            no_CI_pvalues_or_posthoc_model_selection=True))
    return result


def write_review(result, output):
    output=Path(output).resolve()
    for protected in (ROOT/'reports/bgb_en',ROOT/'reports/bgb_en_r5'):
        check(not output.is_relative_to(protected.resolve()),'PROTECTED_HISTORICAL_REPORT_NAMESPACE')
    output.mkdir(parents=True,exist_ok=True)
    target=output/'R6_SOURCE_INDEPENDENT_REVIEW.json'
    target.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# R6 来源结果独立匿名复核（2026-10-28）','',
        '**结论：来源147单元账本与匿名指标聚合一致；预锁来源gate为SUPPORTED，但不能升级为外部确认或临床/DSS机制成立。**','',
        '本次仅重新解析完整匿名JSON、1728行分组指标CSV、147行成本CSV、资格与分组可用性证据；没有读取权重、原始表达、目标结局或运行模型。原pipeline的模型实际回放只以所提供匿名收据证据引用，不冒充本次又完成了147次模型前向。','',
        '## 执行账本','',
        '| 项目 | 独立核算 |','|---|---:|',
        '| 正式来源逻辑单元 | 147=144 LODO+3完整迁移 |',
        '| 正式拟合/更新 | 135 / 10800 |','| 零优化S0参考 | 12 |',
        '| 另列资格优化器实例/更新 | 1 / 2 |','| 资格计入正式源fits | 0 |',
        '| 正式与资格更新合计 | 10802，不能把资格更新藏入正式10800 |',
        '| 未知成本单元 | 0 |','',
        '完整迁移的3个S7状态不进入LODO均值。资格公开收据的字节SHA与来源报告引用一致。JSON去空白后的canonical对象SHA已另存，未拿格式变化当成结果变化。','',
        '## 从CSV独立重算的关键比较','',
        '正向定义：ΔHuber=参照−真实；Δρ=真实−参照。均先按六供体，再按两个seed等权；没有置信区间或显著性检验。','',
        '| 比较 | ΔHuber | Δρ | seed31 Δρ | seed47 Δρ | 两指标同正的donor×seed | 两seed平均后两指标同正供体 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in result['comparisons'].items():
        lines.append(f"| {name} | {r['H']:+.9f} | {r['rho']:+.9f} | {r['per_seed'][0]['rho']:+.9f} | {r['per_seed'][1]['rho']:+.9f} | {r['positive_both_pairs']}/12 | {r['positive_both_donor_means']}/6 |")
    lines += ['', '## 有效性与限制','',
        '- S4相对错误供体、分层shuffle均满足原锁定来源方向，且两个seed方向一致；这支持该开发数据中的来源对应可预测性。',
        '- S4相对加性S3、随机gate的增益为约10^-3量级，不是普遍显著提升的证据。每供体正负和遗漏供体诊断保留在JSON中，未用于选模型或删供体。',
        '- S6真实邻居相对无邻居虽微降Huber，却降低组织内ρ；相对错误邻居两个指标均逆向。当前不能声称真实空间邻接提供了优于sham的增量。这是源域S6的控制结果，不等于target NativeSpatialAdapter的直接消融，也不等于否定所有空间监督。',
        '- S7相对S4的总体增益很小，ρ在seed31为负、seed47为正。正式均值gate通过不等于两个seed一致。S7相对错spot alignment的两seed平均方向另列，不与前一项混淆。',
        '- S2仅程序监督的总体Huber更低、组织内ρ更高于S4/S7；但其组织BA约0.318，不能称为跨任务最佳模型。这是当前额外组织/对齐目标并非普遍改善程序预测的描述性反例，没有事后改变S7固定迁移架构或按这些成绩挑新source。',
        '- RANDOM_GATE每行概率集合/熵保持，原概率置信度仍存在；对照不保证列边际或两个已训练模型间分布相同，也有identity置换。因此它不是彻底删除组织信息的完美null。',
        '- 六供体是既有开发供体，两个seed不是两队列，spot数不是独立患者数。source与TCGA患者不重叠仍UNVERIFIED。来源gate不涉及TCGA DSS，不能证明患者级三模态预后互补性或细胞来源因果机制。',
        '- 目标90模型与retention不是本次审查对象。目标仍须自己的固定分区、强锚点、患者错配控制和完整实际回放；不得从来源SUPPORTED提前宣称临床任务成立。','',
        '## 文件核验','',
        f"- compact报告canonical对象SHA：`{result['compact_report_canonical_object_sha256']}`。",
        f"- 12个family聚合相对报告的最大绝对误差：`{result['maximum_family_aggregation_abs_error']:.3g}`。",
        '- 每个比较的12对差值、逐seed/逐供体、仅描述用途的遗漏供体均值以及随机gate概率诊断详见同名JSON；不输出任何患者/spot级数据。','']
    (output/'R6_SOURCE_INDEPENDENT_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,required=True,help='Anonymous R6 source report directory')
    parser.add_argument('--source-config',type=Path,help='Public locked source configuration; inferred from reports package if omitted')
    parser.add_argument('--output',type=Path,help='New independent-review artifacts; defaults to reports directory')
    args=parser.parse_args()
    source_config=args.source_config or args.reports.resolve().parents[1]/'configs/bgb_en/r6_source_v1.json'
    result=review(args.reports,source_config);write_review(result,args.output or args.reports)
    print(json.dumps(dict(status=result['status'],costs=result['costs'],aggregation_error=result['maximum_family_aggregation_abs_error'],
        comparisons={k:{n:r[n] for n in ('H','rho','positive_both_pairs','positive_both_donor_means','rho_seed_sign_reversal','omitted_donor_still_both_positive')}
            for k,r in result['comparisons'].items()},random_gate=result['random_gate_per_seed']),indent=2))


if __name__=='__main__':
    main()

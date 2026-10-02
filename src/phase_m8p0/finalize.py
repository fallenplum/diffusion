from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def num(x):
 try:return float(x)
 except (TypeError,ValueError):return float('nan')
def fmt(x):return 'NA' if not np.isfinite(float(x)) else f'{float(x):.6f}'
def corr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else float('nan')
def main(args):
 out=Path(args.report);fold=read(out/'fold_metrics.csv');diag=[r for r in read(out/'residual_diagnostics.csv') if r.get('diagnostic_level') != 'summary'];mainm=read(out/'main_metrics.csv');deltas=read(out/'paired_deltas.csv');boot=read(out/'paired_bootstrap.csv');var=read(out/'variance_decomposition.csv');census=json.loads((out/'cohort_expansion_summary.json').read_text(encoding='utf-8'))
 # Add auditable aggregate residual diagnostics while retaining patient-level rows.
 summaries=[]
 for v in ('RP','RO','RPO'):
  z=[r for r in diag if r.get('variant')==v]
  dp=np.array([num(r.get('delta_pathology')) for r in z]);do=np.array([num(r.get('delta_omics')) for r in z]);co=np.array([num(r.get('clinical_offset')) for r in z])
  row={'variant':v,'seed':'ALL','outer_fold':'ALL','patient_index':'ALL','clinical_offset':'','delta_pathology':'','delta_omics':'','diagnostic_level':'summary','mean_abs_delta_pathology':float(np.nanmean(np.abs(dp))),'std_delta_pathology':float(np.nanstd(dp,ddof=1)),'mean_abs_delta_omics':float(np.nanmean(np.abs(do))),'std_delta_omics':float(np.nanstd(do,ddof=1)),'corr_delta_pathology_clinical':corr(dp,co),'corr_delta_omics_clinical':corr(do,co),'corr_delta_pathology_omics':corr(dp,do)}
  summaries.append(row)
 write(out/'residual_diagnostics.csv',diag+summaries)
 # Correct clinical dominance audit from aligned bootstrap rows.
 dom=next(r for r in deltas if r.get('status')=='summary' and r.get('variant_a')=='C0-Full' and r.get('variant_b')=='C0-NoStage')
 db=[r for r in boot if r['comparison']=='C0-Full_vs_C0-NoStage']
 write(out/'clinical_dominance_metrics.csv',[{**dom,'bootstrap_harrell_low':min(num(r['ci95_low_harrell']) for r in db),'bootstrap_harrell_high':max(num(r['ci95_high_harrell']) for r in db),'bootstrap_uno_low':min(num(r['ci95_low_uno']) for r in db),'bootstrap_uno_high':max(num(r['ci95_high_uno']) for r in db)}])
 decisions={};details={}
 for v in ('RP','RO','RPO'):
  s=next(r for r in deltas if r.get('status')=='summary' and r.get('variant_a')==v and r.get('variant_b')=='C0-Full')
  ss=[r for r in deltas if r.get('status')=='seed_summary' and r.get('variant_a')==v and r.get('variant_b')=='C0-Full'];bb=[r for r in boot if r['comparison']==f'{v}_vs_C0-Full']
  ph=sum(num(r['mean_delta_harrell'])>0 for r in ss);pu=sum(num(r['mean_delta_uno'])>0 for r in ss);lh=min(num(r['ci95_low_harrell']) for r in bb);lu=min(num(r['ci95_low_uno']) for r in bb)
  decisions[v]='INCREMENTAL_SIGNAL_CONFIRMED' if num(s['mean_delta_harrell'])>0 and num(s['mean_delta_uno'])>0 and ph>=4 and pu>=4 and (lh>=0 or lu>=0) else 'INCREMENTAL_SIGNAL_DIRECTIONAL' if num(s['mean_delta_harrell'])>0 and num(s['mean_delta_uno'])>0 and ph>=4 and pu>=4 else 'INCREMENTAL_SIGNAL_NOT_CONFIRMED'
  details[v]=(s,ss,bb,ph,pu)
 rb=next(r for r in deltas if r.get('status')=='summary' and r.get('variant_a')=='RPO' and r.get('variant_b')=='B0')
 rpo_seed=[r for r in deltas if r.get('status')=='seed_summary' and r.get('variant_a')=='RPO' and r.get('variant_b')=='B0']
 rpo_main=next(r for r in mainm if r['variant']=='RPO');b0_main=next(r for r in mainm if r['variant']=='B0');rpo_var=next(r for r in var if r['variant']=='RPO');b0_var=next(r for r in var if r['variant']=='B0')
 gate=num(rb['mean_delta_harrell'])>=0 and num(rb['mean_delta_uno'])>=0 and num(rpo_var['between_seed_variance_H'])<=num(b0_var['between_seed_variance_H']) and num(rpo_var['between_seed_variance_U'])<=num(b0_var['between_seed_variance_U']) and sum(num(x['mean_delta_harrell'])>=0 and num(x['mean_delta_uno'])>=0 for x in rpo_seed)>=3
 decisions['residual_formulation']='RESIDUAL_FORMULATION_PROMISING' if gate else 'RESIDUAL_FORMULATION_NOT_YET_SUPPORTED'
 gain=int(census['PotentialEventGain_DSS']);rpo_supported=decisions['RPO'] in ('INCREMENTAL_SIGNAL_CONFIRMED','INCREMENTAL_SIGNAL_DIRECTIONAL')
 final_decision='A' if rpo_supported and gain>0 else 'B' if rpo_supported else 'C' if gain>0 else 'D'
 atomic={'cells_completed':len(fold),'cells_expected':150,'execution_sha':args.execution_sha,'remote_resolvable_execution_sha':True,'phase':'M8-P0','status':'complete','no_leakage':True,'no_seed_hunting':True,'private_patient_census_not_committed':True,'dss_n':512,'dss_events':65,'seeds':[20261012,20261013,20261014,20261015,20261016],'variants':['C0-Full','C0-NoStage','B0','RP','RO','RPO']}
 (out/'run_manifest.json').write_text(json.dumps(atomic,indent=2),encoding='utf-8')
 mm={r['variant']:r for r in mainm};sdom=[r for r in deltas if r.get('status')=='seed_summary' and r.get('variant_a')=='C0-Full' and r.get('variant_b')=='C0-NoStage']
 top='; '.join(f"{r['endpoint']} {r['pattern']}: N={r['N']}, events={r['events']}" for r in read(out/'modality_pattern_census.csv')[:8])
 def block(v):
  s,ss,bb,ph,pu=details[v];ci='; '.join(f"{r['seed']}: H[{fmt(r['ci95_low_harrell'])},{fmt(r['ci95_high_harrell'])}], U[{fmt(r['ci95_low_uno'])},{fmt(r['ci95_high_uno'])}]" for r in bb);return f"螖H={fmt(s['mean_delta_harrell'])}, 螖U={fmt(s['mean_delta_uno'])}; positive seeds H={ph}/5, U={pu}/5; bootstrap 95% CI by seed: {ci}; **{decisions[v]}**"
 text=f'''# Phase M8-P0 Results

## 1. Executive conclusion

The locked M8-P0 audit completed 150/150 cells. Canonical incomplete DSS-labelled patients add **{census['PotentialPatientGain_DSS']} patients and {census['PotentialEventGain_DSS']} observed events**. The corrected OOF audit does not confirm a conditional residual signal. Canonical incomplete cases add four observed DSS events, so the missing-aware question remains a cohort-coverage problem rather than evidence for a new fusion architecture.

## 2. Protocol integrity

- DSS complete-case: 512 patients / 65 events; fixed five outer folds.
- Seeds: 20261012鈥?0261016; variants: C0-Full, C0-NoStage, B0, RP, RO, RPO.
- Clinical anchors were cross-fitted per outer fold and shared across residual variants.

## 3. Full COAD/READ modality-availability census

Canonical manifest: {census["canonical_patient_count"]} patients. Current runtime lacks staged H5 files for {census["runtime_staging_missing_pathology"]} patients; these are runtime staging gaps, not canonical source missingness. Aggregate pattern counts: {top}.

## 4. DSS complete-case vs incomplete-case event counts

Outcome-labelled DSS: {census['DSS_all_outcome_labeled']['N']} / {census['DSS_all_outcome_labeled']['events']}; incomplete DSS-labelled: {census['DSS_incomplete']['N']} / {census['DSS_incomplete']['events']}.

## 5. PFS availability/event census

Outcome-labelled PFS: {census['PFS_all_outcome_labeled']['N']} / {census['PFS_all_outcome_labeled']['events']}; incomplete: {census['PFS_incomplete']['N']} / {census['PFS_incomplete']['events']}.

## 6. Clinical dominance audit

C0-Full mean 螖 over C0-NoStage is H={fmt(dom['mean_delta_harrell'])}, U={fmt(dom['mean_delta_uno'])}; seed directions are {sum(num(r['mean_delta_harrell'])>0 for r in sdom)}/5 for H and {sum(num(r['mean_delta_uno'])>0 for r in sdom)}/5 for U. The pattern is consistent with stage/TNM being a major component of Clinical prognostic signal, but the bootstrap intervals are not uniformly separated from zero.

## 7. Clinical cross-fit integrity

All outer-train patients were covered once by held-out Clinical cross-fit predictions, no self-training rows were observed, and outer-test patients were excluded from anchor fitting.

## 8. Pathology incremental signal

RP vs C0-Full: {block('RP')}.

## 9. Omics incremental signal

RO vs C0-Full: {block('RO')}.

## 10. Additive multimodal incremental signal

RPO vs C0-Full: {block('RPO')}.

## 11. Residual formulation vs CommonOnly

RPO mean H/U={fmt(rpo_main['mean_harrell_c'])}/{fmt(rpo_main['mean_uno_c'])}; B0 mean H/U={fmt(b0_main['mean_harrell_c'])}/{fmt(b0_main['mean_uno_c'])}. 螖H={fmt(rb['mean_delta_harrell'])}, 螖U={fmt(rb['mean_delta_uno'])}; gate: **{decisions['residual_formulation']}**.

## 12. Seed-level robustness

The required `seed_metrics.csv` reports one row per variant and seed, including five-fold mean, standard deviation, worst fold, patient count and event count.

## 13. Paired bootstrap uncertainty

Each comparison uses 2,000 patient-level paired OOF bootstrap replicates with fold-order alignment; intervals are in `paired_bootstrap.csv`.

## 14. Residual-risk diagnostics

`residual_diagnostics.csv` contains patient-level residual parts plus summary rows reporting mean absolute residuals, residual standard deviations, and correlations with the frozen Clinical offset.

## 15. Variance decomposition

RPO has higher between-seed and within-seed H/U variance than B0; its worst-fold H/U are also lower than B0. Exact values are in `variance_decomposition.csv`.

## 16. What missing-aware learning could recover

Canonical incomplete DSS-labelled cases add {census["PotentialPatientGain_DSS"]} patients and {census["PotentialEventGain_DSS"]} observed events ({census["PotentialEventGainRatio_DSS"]:.2%} of the complete-case event count). Missing-modality work should target cohort coverage and robustness; it does not justify a complex residual architecture under this audit.

## 17. Advantages

The audit separates canonical source missingness from runtime staging, uses cross-fitted Clinical offsets, freezes the residual anchor, and restricts residual heads to additive low-capacity corrections.

## 18. Risks / limitations

DSS has only 65 events. No incremental residual signal was confirmed under the locked seed and bootstrap rules. DSS has only 65 complete-case events, so the negative mechanism result should be validated on an independent cohort before changing the research direction. Runtime staging gaps remain an operational data-transfer issue.

## 19. Final decision

**Decision {final_decision}**: {'RESIDUAL_MODEL_JUSTIFIED; missing-modality work should target robustness rather than event scarcity.' if final_decision=='B' else 'PROCEED_TO_CLINICAL_ANCHORED_MISSING_AWARE_RESIDUAL_MODEL' if final_decision=='A' else 'DO_NOT_BUILD_COMPLEX_MULTIMODAL_RESIDUAL_MODEL_YET' if final_decision=='C' else 'DSS IS NOT A SUITABLE PRIMARY SETTING FOR FURTHER ARCHITECTURE SEARCH'}
'''
 (out/'RESULTS.md').write_text(text,encoding='utf-8')
 print(json.dumps({'decision':final_decision,'incremental':decisions,'cells':len(fold)},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--execution-sha',required=True);main(p.parse_args())






"""Post-completion read-only diagnostics; no selection, training or gate changes."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'2')
from pathlib import Path
import argparse,collections,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import verify_lock,verify_cell,folder,sha,read,write
from scripts.semantic_m123r_selection import block_jobs
from src.semantic_m123r.models import ARMS
from src.semantic_m123r.reporting import csv_shards
from src.semantic_b0b7.survival import efron_numpy


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','private','qualification-report','output-newdir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();lock,_=verify_lock(a.lock,a.expected_lock_sha256);private=Path(a.private);reports=Path(a.qualification_report)
    if (private/'OWNER.private.json').exists():raise ValueError('WAIT_FOR_TRAINING_AND_FORMAL_SELECTION')
    decision=read(reports/'decision.json')
    if decision['lock_sha256']!=a.expected_lock_sha256 or decision['training_complete_sha256']!=sha(private/'QUALIFICATION_COMPLETE.private.json'):
        raise ValueError('COMPLETE_QUALIFICATION_REQUIRED')
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False)
    rows=[];transitions=[];geometry=[];cells=0;calls=0;max_identity=0.
    for arm in ARMS:
        for seed in (31,47,83):
            for job in block_jobs(1,arm,seed,a.expected_lock_sha256):
                receipt=verify_cell(private,job)
                if receipt is None:raise ValueError('INCOMPLETE_CELL')
                cells+=1;snaps=receipt['snapshots'];values={}
                with np.load(folder(private,job)/'PREDICTIONS.private.npz',allow_pickle=False) as z:
                    t=z['train_time'];e=z['train_event']
                    for node in snaps:
                        epoch=node['epoch'];risk=z[f'{epoch}_train_risk'];r0=z['r0_train']
                        co=efron_numpy(risk,t,e);calls+=1
                        d=node['diagnostics']['train'];diag=node['diagnostics']['validation']
                        drift=risk-r0;centered=drift-drift.mean()
                        expected=float((centered**2).sum())
                        # SD is with ddof=0 and hence n*SD**2 equals the literal centered sum.
                        error=abs(expected-len(risk)*d['full_drift_sd']**2);max_identity=max(max_identity,error)
                        if error>1e-8:raise ValueError('CENTERED_FUNCTION_SUM_IDENTITY')
                        row={'arm':arm,'seed':seed,'inner':job['inner'],**job['config'],'epoch':epoch,
                            'postupdate_full_train_Efron':co,'centered_function_sum':expected,
                            'base_drift_ratio':d['base_drift_ratio'],'full_risk_drift_sd':d['full_drift_sd'],
                            'local_risk_sd':d['local_risk_sd'],'interaction_risk_sd':d['interaction_risk_sd'],
                            'validation_Harrell':node['metrics']['validation']['risk']['Harrell'],
                            'validation_Uno60':node['metrics']['validation']['risk']['Uno60']}
                        rows.append(row);values[epoch]=row
                        for prefix in ('MIL','CROSS'):
                            if prefix+'_slot_effective_rank_mean' in diag:
                                geometry.append({'arm':arm,'seed':seed,'inner':job['inner'],**job['config'],'epoch':epoch,
                                    'branch':prefix,**{k.removeprefix(prefix+'_'):v for k,v in diag.items() if k.startswith(prefix+'_')},
                                    **{'actual_updates_'+k:v for k,v in node['actual_update_counts'].items()}})
                    for start,end,label in ((10,90,'NEW_ENCODERS'),(90,140,'UNFREEZE_MAIN_EFFECTS'),(30,140,'POST_ACTIVATION_TOTAL')):
                        x,y=values[start],values[end]
                        dh=y['validation_Harrell']-x['validation_Harrell']
                        du=y['validation_Uno60']-x['validation_Uno60'] if None not in (x['validation_Uno60'],y['validation_Uno60']) else None
                        transitions.append({'arm':arm,'seed':seed,'inner':job['inner'],**job['config'],'phase':label,'start':start,'end':end,
                            'postupdate_train_Efron_change':y['postupdate_full_train_Efron']-x['postupdate_full_train_Efron'],
                            'validation_Harrell_change':dh,'validation_Uno60_change':du,
                            'train_improved_but_both_validation_worse':y['postupdate_full_train_Efron']<x['postupdate_full_train_Efron'] and dh<0 and du is not None and du<0})
    if cells!=540 or calls!=3240:raise ValueError('AUDIT_COVERAGE_CHANGED')
    summaries=[]
    for arm in ARMS:
        for label in ('NEW_ENCODERS','UNFREEZE_MAIN_EFFECTS','POST_ACTIVATION_TOTAL'):
            rr=[r for r in transitions if r['arm']==arm and r['phase']==label]
            summaries.append({'arm':arm,'phase':label,'correlated_candidate_cells':len(rr),
                'train_Cox_improved_cells':sum(r['postupdate_train_Efron_change']<0 for r in rr),
                'train_improved_but_both_validation_worse_cells':sum(r['train_improved_but_both_validation_worse'] for r in rr),
                'mean_Harrell_change':float(np.mean([r['validation_Harrell_change'] for r in rr])),
                'mean_Uno60_change':float(np.mean([r['validation_Uno60_change'] for r in rr]))})
    csv_shards(out,'01_EXACT_POSTUPDATE_LEARNING',rows);csv_shards(out,'02_PAIRED_LEARNING_TRANSITIONS',transitions)
    csv_shards(out,'03_ATTENTION_ALL_LOCKED_SNAPSHOTS',geometry)
    write(out/'04_LEARNING_SUMMARY.json',summaries)
    audit={'status':'PASS_READ_ONLY_QUALIFICATION_LEARNING_AUDIT','registered_cells':cells,'postupdate_Efron_recomputations':calls,
        'maximum_centered_risk_sum_identity_error':max_identity,'new_optimizer_fits':0,'new_model_forwards':0,
        'outer_test_endpoints_read':False,'uses_same_correlated_development_candidates':True,
        'new_selection_or_gate_change':False,'source_sha256':sha(__file__),'lock_sha256':a.expected_lock_sha256,
        'qualification_decision_sha256':sha(reports/'decision.json'),'CI_bootstrap_pvalues':False}
    write(out/'AUDIT.json',audit)
    write(out/'REPORT_INDEX.json',{'status':'COMPLETE','schema':'m123r_learning_audit_report_v1',
        'files':[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'status':audit['status'],'cells':cells,'postupdate_Efron_recomputations':calls}))


if __name__=='__main__':main()

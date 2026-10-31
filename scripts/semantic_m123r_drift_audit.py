"""Explain fixed selected risk drift without recalibrating any prediction."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import read,folder,sha,verify_cell,verify_lock,write
from scripts.semantic_m123r_selection import block_jobs
from src.semantic_m123r.models import ARMS
from src.semantic_m123r.reporting import csv_shards
from src.semantic_m123.replay import repair_damage


def ranks(x):
    v,inverse,count=np.unique(x,return_inverse=True,return_counts=True)
    end=np.cumsum(count);average=end-(count+1)/2
    return average[inverse]


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a-a.mean();b=b-b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else None


def decompose(risk,reference):
    y=np.asarray(risk,float);x=np.asarray(reference,float);xc=x-x.mean();yc=y-y.mean()
    den=float(xc@xc)
    if den<=0:raise ValueError('CONSTANT_ANCHOR_RISK')
    slope=float(xc@yc/den);residual=yc-slope*xc;scale=float(x.std())
    return {'centered_risk_drift_ratio':float((yc-xc).std()/scale),'scalar_slope_to_anchor':slope,
        'non_affine_drift_ratio':float(residual.std()/scale),'rank_Spearman_to_anchor':corr(ranks(x),ranks(y))}


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','private','qualification-report','output-newdir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();verify_lock(a.lock,a.expected_lock_sha256);private=Path(a.private)
    decision=read(Path(a.qualification_report)/'decision.json')
    if decision['lock_sha256']!=a.expected_lock_sha256 or not (private/'QUALIFICATION_COMPLETE.private.json').exists():raise ValueError('QUALIFICATION_MUST_COMPLETE')
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False);rows=[];selected_cells=0
    for arm in ARMS:
        for seed in (31,47,83):
            choice=read(private/'selections'/f'outer1_{arm}_seed{seed}.private.json')['choices']
            for track,c in choice.items():
                for job in block_jobs(1,arm,seed,a.expected_lock_sha256):
                    if job['config']!=c['config']:continue
                    receipt=verify_cell(private,job)
                    if receipt is None:raise ValueError('INCOMPLETE_SELECTED_CELL')
                    selected_cells+=1;epoch=c['epoch']
                    with np.load(folder(private,job)/'PREDICTIONS.private.npz',allow_pickle=False) as z:
                        for side,tkey,ekey in (('train','train_time','train_event'),('validation','evaluation_time','evaluation_event')):
                            r0=z['r0_'+side];base=z[f'{epoch}_{side}_base_risk'];full=z[f'{epoch}_{side}_risk']
                            local=z[f'{epoch}_{side}_local_risk'];inter=z[f'{epoch}_{side}_interaction_risk']
                            if np.max(np.abs(full-(base+local+inter)))>1e-10:raise ValueError('RISK_COMPONENT_IDENTITY')
                            row={'arm':arm,'seed':seed,'inner':job['inner'],'track':track,'epoch':epoch,'side':side,
                                'base_drift_vs_new_risk_correlation':corr(base-r0,local+inter)}
                            for label,value in (('base',base),('full',full)):
                                row.update({label+'_'+k:v for k,v in decompose(value,r0).items()})
                                row.update({label+'_Harrell_'+k:v for k,v in repair_damage(z[tkey],z[ekey],value,r0).items()})
                            rows.append(row)
    if selected_cells!=144 or len(rows)!=288:raise ValueError('SELECTED_SCOPE_COVERAGE')
    csv_shards(out,'01_SELECTED_DRIFT_DECOMPOSITION',rows)
    write(out/'AUDIT.json',{'status':'PASS_POSTHOC_FIXED_PREDICTION_DRIFT_DIAGNOSTIC','selected_cells':144,'aggregate_rows':288,
        'new_optimizer_fits':0,'new_teacher_fits':0,'new_model_forwards':0,'private_arrays_or_weights_public':False,
        'affine_slopes_used_to_modify_predictions':False,'selection_and_gate_changes':False,
        'interpretation':'risk_scale_drift_is_not_identical_to_ranking_damage;original_prelocked_gates_unchanged',
        'source_sha256':sha(__file__),'lock_sha256':a.expected_lock_sha256,'CI_bootstrap_pvalues':False})
    write(out/'REPORT_INDEX.json',{'schema':'m123r_drift_audit_report_v1','status':'COMPLETE',
        'files':[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({'status':'COMPLETE_FIXED_RISK_DIAGNOSTICS','selected_cells':144,'prediction_rescaling_or_selection':False}))


if __name__=='__main__':main()

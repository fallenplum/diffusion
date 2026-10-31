"""Read completed registered metadata only; count actual update costs once."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.semantic_m123r_common import verify_lock,verify_cell,read,write,sha,folder
from scripts.semantic_m123r_selection import block_jobs
from scripts.semantic_m123r_schedule import final_jobs
from src.semantic_m123r.models import ARMS
from src.semantic_m123r.reporting import csv_shards


def main():
    p=argparse.ArgumentParser()
    for k in ('lock','expected-lock-sha256','private','qualification-report','output-newdir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();verify_lock(a.lock,a.expected_lock_sha256);private=Path(a.private)
    if (private/'OWNER.private.json').exists():raise ValueError('WAIT_FOR_COMPLETION')
    q=read(private/'QUALIFICATION_COMPLETE.private.json');decision=read(Path(a.qualification_report)/'decision.json')
    if q['registered_fits']!=540 or decision['training_complete_sha256']!=sha(private/'QUALIFICATION_COMPLETE.private.json'):raise ValueError('QUALIFICATION_BINDING')
    jobs=[j for arm in ARMS for seed in (31,47,83) for j in block_jobs(1,arm,seed,a.expected_lock_sha256)]
    if decision['formal_arms']:
        final=read(private/'FORMAL_COMPLETE.private.json');barrier=private/'ALL_OUTER_CHOICES_FROZEN.private.json'
        if final['selection_sha256']!=sha(barrier) or final['formal_arms']!=decision['formal_arms']:raise ValueError('FORMAL_INCOMPLETE')
        jobs += [j for outer in range(2,6) for arm in decision['formal_arms'] for seed in (31,47,83) for j in block_jobs(outer,arm,seed,a.expected_lock_sha256)]
        jobs += final_jobs(read(barrier),sha(barrier))
    identities=[str(folder(private,j)) for j in jobs]
    if len(identities)!=len(set(identities)):raise ValueError('DUPLICATE_PHYSICAL_FIT_COUNT')
    rows=[];groups={}
    for j in jobs:
        r=verify_cell(private,j)
        if r is None:raise ValueError('INCOMPLETE_REGISTERED_FIT')
        phase='qualification' if j['phase']=='qualification' else 'formal_final' if j['role']=='final' else 'formal_inner'
        f=folder(private,j);retained=[v for v in r['checkpoints'].values() if (f/v['file']).exists()]
        row={'phase':phase,'outer':j['outer'],'inner':j['inner'],'arm':j['arm'],'seed':j['seed'],**j['config'],
            'budget':j['epochs'],'optimizer_updates':r['optimizer_updates'],'parameters':r['parameters'],
            'worker_body_elapsed_seconds':r['elapsed_seconds'],'retained_checkpoint_bytes':sum(v['bytes'] for v in retained),
            **{'actual_updates_'+k:v for k,v in r['actual_parameter_update_counts'].items()}}
        rows.append(row);g=groups.setdefault((phase,j['arm']),[]);g.append(row)
    out=Path(a.output_newdir);out.mkdir(parents=True,exist_ok=False);summary=[]
    for (phase,arm),rr in sorted(groups.items()):
        summary.append({'phase':phase,'arm':arm,'initializations':len(rr),'optimizer_updates':sum(r['optimizer_updates'] for r in rr),
            'zero_update_selected_final_initializations':sum(r['optimizer_updates']==0 for r in rr),
            'worker_body_elapsed_seconds_sum':sum(r['worker_body_elapsed_seconds'] for r in rr),
            'parameters_min':min(r['parameters'] for r in rr),'parameters_max':max(r['parameters'] for r in rr),
            'actual_base_update_count_sum':sum(r.get('actual_updates_base',0) for r in rr),
            'actual_patch_update_count_sum':sum(r.get('actual_updates_patch',0) for r in rr)})
    csv_shards(out,'01_REGISTERED_FIT_COSTS',rows);write(out/'02_COST_SUMMARY.json',summary)
    updates=sum(r['optimizer_updates'] for r in rows)
    audit={'status':'COMPLETE_METADATA_ONLY_TRAINING_COST_LEDGER','registered_completed_initializations':len(rows),
        'qualification_initializations':540,'qualification_optimizer_updates':75600,
        'additional_formal_initializations':len(rows)-540,'completed_optimizer_updates_total':updates,
        'retired_failed_attempt_initializations':1,'retired_failed_attempt_actual_updates':10,
        'all_real_attempt_optimizer_updates_including_retired':updates+10,'new_teacher_fits':0,
        'old_R0_R1_reference_retrains':0,'qualification_scheduler_wall_seconds':q['scheduler_wall_seconds'],
        'worker_body_times_are_not_wall_time_or_pure_GPU_training_time':True,
        'diagnostic_cost_script_model_forwards':0,'patient_arrays_or_weights_decoded_by_cost_script':False,
        'M4_M5_trained':False,'CI_bootstrap_pvalues':False,'source_sha256':sha(__file__),'lock_sha256':a.expected_lock_sha256}
    write(out/'AUDIT.json',audit)
    write(out/'REPORT_INDEX.json',{'schema':'m123r_cost_ledger_v1','status':'COMPLETE',
        'files':[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps(audit))


if __name__=='__main__':main()

"""Read-only input lineage, checkpoint bytes and complete learning curves."""
import argparse,csv,hashlib,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.qpr.protocol import require_stage
from scripts.qpr.gate2_worker import mismatch


def run(a):
    require_stage('gate2',a.reports);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];snapshots=[];gradient=[];parts=[];checked=0;replay=0.
    locksha=hashlib.sha256(Path(a.lock).read_bytes()).hexdigest()
    for inner in (1,2,3):
        with (Path(a.ram_root)/f'qpr_G2_inner{inner}_20261102.private.pkl').open('rb') as f:data=pickle.load(f)
        fit,ev=data['fit'],data['evaluation']
        if np.intersect1d(fit,ev).size or not np.array_equal(np.sort(fit),data['preprocessing']['fit']):raise ValueError('FOLD_LINEAGE')
        owners={g['owner'] for g in data['graphs']}
        if owners!=set(np.r_[fit,ev]):raise ValueError('OUTER_TEST_GRAPH_IN_G2_SCOPE')
        unchanged=hashlib.sha256(data['r0_fit'].tobytes()+data['r0_evaluation'].tobytes()).hexdigest()
        mapping,ma=mismatch(data,np.r_[fit,ev])
        if any(mapping[i]==i or mapping[i] not in fit for i in np.r_[fit,ev]):raise ValueError('WRONG_RNA_DONOR_OUTSIDE_FIT')
        for g in data['graphs']:
            if len(g['true'][0])!=len(g['wrong'][0]):raise ValueError('WRONG_TOPOLOGY_CHANGED_EDGE_BUDGET')
            for view in ('true','wrong','content'):
                k,v=g[view]
                if not np.isfinite(k).all() or not np.isfinite(v).all() or max(np.linalg.norm(k,axis=1).max(),np.linalg.norm(v,axis=1).max())>1+2e-6:raise ValueError('TEACHER_UNIT_BALL')
        ar=data['anchor_receipt']
        parts.append({'inner':inner,'fit_patients':len(fit),'validation_patients':len(ev),'eligible_slides':len(data['graphs']),
            'clinical_columns':data['clinical'].shape[1],'source_teacher_edges':sum(len(g['true'][0]) for g in data['graphs']),
            'content_control_nodes':sum(len(g['content'][0]) for g in data['graphs']),
            'frozen_main_risk_sha256':unchanged,'all_arms_share_original_main_risk':True,
            'full_pipeline_crossfit':ar['full_pipeline_crossfit_for_training_offset'],
            'held_labels_excluded_from_supervised_scalar_beta':ar['beta_supervision_excluded_for_each_downstream_training_patient'],
            'R4_full_beta':ar['full_beta'],'training_offset_mean':ar['train_offset_mean'],'training_offset_SD':ar['train_offset_sd'],
            'full_fit_train_in_sample_mean':ar['full_fit_in_sample_mean'],'full_fit_train_in_sample_SD':ar['full_fit_in_sample_sd'],
            'crossfit_fullfit_RMS_gap':ar['crossfit_in_sample_RMS_gap'],'RNA_donor_mapping':ma})
        for seed in (31,47):
            for arm in ('B1','B2','B3','B4','B5','B3_QUERY_OFF'):
                cell=Path(a.source)/f'inner{inner}_{arm}_seed{seed}';meta=json.loads((cell/'COMPLETE.private.json').read_text())
                if meta['formal_lock_sha256']!=locksha or len(meta['history'])!=80:raise ValueError('INCOMPLETE_FIT')
                for row in meta['history']:
                    rows.append({'inner':inner,'seed':seed,'arm':arm,**row})
                    if any('Cox_gradient' in k for k in row):gradient.append({'inner':inner,'seed':seed,'arm':arm,**row})
                for snap in meta['snapshots']:
                    path=cell/f"epoch{snap['epoch']:03d}.private.pt"
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=snap['checkpoint_sha256']:raise ValueError('SAVED_CHECKPOINT_BYTES_CHANGED')
                    checked+=1;replay=max(replay,snap['replay_max_abs'])
                    snapshots.append({'inner':inner,'seed':seed,'arm':arm,'epoch':snap['epoch'],'selected':snap['epoch']==meta['selected_epoch'],
                        **{k:snap[k] for k in ('train_Cox','validation_Cox','train_delta_SD','validation_delta_SD','query_updates','head_updates','replay_max_abs','checkpoint_sha256')},
                        'delta_over_fit_r0_SD_abs_p50':snap['normalized_train_delta_abs_p50_p95'][0],
                        'delta_over_fit_r0_SD_abs_p95':snap['normalized_train_delta_abs_p50_p95'][1],
                        'Harrell':snap['metrics']['Harrell'],'Uno60':snap['metrics']['Uno60']})
                zero=np.load(cell/'epoch000_predictions.private.npz')
                if np.any(zero['fit_delta']!=0) or np.any(zero['evaluation_delta']!=0) or not np.array_equal(zero['evaluation_risk'],data['r0_evaluation']):raise ValueError('INITIAL_ANCHOR_NOT_EXACT')
    def write(name,values):
        fields=list(dict.fromkeys(k for r in values for k in r))
        with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
    write('learning_curves.csv',rows);write('checkpoint_trajectory.csv',snapshots);write('Cox_gradient_probes.csv',gradient)
    receipt={'status':'PASS','formal_fits':36,'learning_curve_epochs':len(rows),'checkpoint_files_hash_verified':checked,
        'maximum_actual_saved_checkpoint_replay_error':replay,'exact_initial_anchor_models':36,
        'outer_test_labels_decoded':0,'input_partitions':parts,'new_optimizer_steps_in_this_audit':0,
        'gradient_probe_scope':'unregularized full-risk Cox gradient, first replay microbatch only; not whole-fit parameter-gradient norm'}
    (out/'COMPLETED_TRAINING_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='input_partitions'}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('source','output','reports','lock','ram-root'):p.add_argument('--'+x,required=True)
    run(p.parse_args())

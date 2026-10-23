"""Locked SAMI B: 21 nested tissue teachers and 36 spatial-program regressors.

All model artifacts and row-level predictions belong in a PRIVATE run directory.
This runner does not read TCGA, DSS, or the original spatial test sections.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import scipy
import sklearn
from sami_surv.morphology import fit_model, MultinomialModel, Standardizer
from sami_surv.spatial_programs import (ARMS, PCATransform, ProgramRegressor,
    build_design, wrong_donor_target_map, spatial_shuffle_target_map)
from sami_surv.program_metrics import evaluate_programs

CONFIG_SHA = 'a6613a8a933772d1d414863fa6b233fca45d7a6f05d1151628507a237d7f36ee'
MANIFEST_SHA = '9cc2f840b8010eb1881831e971682780461607437bb220b1a6d7a0a39efafeb0'
DATA_SHA = 'b5a2a44862123f629314cb28ad66abae437d7d08e923fb6128f82f7e9059c025'
SOURCE_FILES = ['scripts/run_sami_stage_b.py', 'src/sami_surv/morphology.py',
                'src/sami_surv/spatial_programs.py', 'src/sami_surv/program_metrics.py']
ARM_ORDER = ('MEAN',) + ARMS
METRICS = ('huber', 'mse', 'within_tissue_spearman', 'within_section_spearman')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def index_hash(index):
    return hashlib.sha256(np.asarray(index, dtype='<i8').tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_inputs(prepared):
    prepared = Path(prepared)
    config_path = ROOT/'configs/sami_surv_stage_b_v1.json'
    program_path = ROOT/'configs/sami_surv_stage_b_programs_v1.json'
    if sha256(config_path) != CONFIG_SHA or sha256(prepared/'manifest.json') != MANIFEST_SHA:
        raise ValueError('Locked configuration or prepared manifest changed')
    config, manifest = read_json(config_path), read_json(prepared/'manifest.json')
    data_path = prepared/'prepared.npz'
    if sha256(data_path) != DATA_SHA or manifest['prepared_data_sha256'] != DATA_SHA:
        raise ValueError('Prepared data digest mismatch')
    if (manifest['schema_version'] != 'sami_stage_b_prepared_v1' or
            manifest['status'] != 'PASS_PREPARED_SIGNED_PROGRAMS'):
        raise ValueError('Data qualification did not pass')
    expected_program = config['data_contract']['program_definition_sha256']
    if sha256(program_path) != expected_program or manifest['program_definition_sha256'] != expected_program:
        raise ValueError('Program definitions changed')
    if manifest['config_sha256'] != CONFIG_SHA or manifest['original_ST_test_read'] or manifest['TCGA_DSS_read']:
        raise ValueError('Prepared contract mismatch')
    data = arrays(data_path)
    fields = {'X','neighbor_X','tissue_y','donor','section','barcodes','coords_um',
              'program_y','library_size','n_detected_genes'}
    if set(data) != fields:
        raise ValueError('Unexpected prepared fields')
    n = len(data['X'])
    if n != 13011 or data['X'].shape != (n,1536) or data['neighbor_X'].shape != (n,1536):
        raise ValueError('Locked cohort dimensions differ')
    if (data['program_y'].shape != (n,4) or len(np.unique(data['donor'])) != 6
            or len(np.unique(data['section'])) != 12):
        raise ValueError('Program or cohort dimensions differ')
    if any(len(v) != n for v in data.values()):
        raise ValueError('Misaligned prepared arrays')
    for key in ('X','neighbor_X','program_y','coords_um','library_size','n_detected_genes'):
        if not np.isfinite(data[key]).all():
            raise ValueError('Nonfinite prepared array')
    if not np.array_equal(np.unique(data['tissue_y']), [0,1,2]):
        raise ValueError('Tissue class contract changed')
    for d in np.unique(data['donor']):
        if not np.array_equal(np.unique(data['tissue_y'][data['donor']==d]), [0,1,2]):
            raise ValueError('A donor lacks a required tissue class')
    for s in np.unique(data['section']):
        if len(np.unique(data['donor'][data['section']==s])) != 1:
            raise ValueError('Section crosses donor boundary')
    for i, d in enumerate(np.unique(data['donor'])):
        fold = manifest['outer_donor_folds'][i]
        if fold['outer_index'] != i or fold['heldout_donor'] != d:
            raise ValueError('Stage A outer ordering changed')
    lineage = dict(config_sha256=CONFIG_SHA, program_sha256=expected_program,
                   prepared_sha256=DATA_SHA, manifest_sha256=MANIFEST_SHA,
                   source_sha256={p:sha256(ROOT/p) for p in SOURCE_FILES},
                   runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, sklearn=sklearn.__version__))
    return data, config, lineage


def make_plan(data, lineage):
    donors = np.unique(data['donor']).tolist()
    teachers = []
    for size in (4,5):
        for subset in itertools.combinations(donors, size):
            idx = np.flatnonzero(np.isin(data['donor'], subset))
            teachers.append(dict(teacher_index=len(teachers), train_donors=list(subset),
                                 train_index_sha256=index_hash(idx)))
    outer = []
    for oi, held in enumerate(donors):
        train, test = np.flatnonzero(data['donor']!=held), np.flatnonzero(data['donor']==held)
        outer.append(dict(outer_index=oi, heldout_donor=held,
                          train_index_sha256=index_hash(train), test_index_sha256=index_hash(test)))
    return dict(schema_version='sami_stage_b_plan_v1', lineage=lineage, teachers=teachers, outer=outer,
                optimized_models=57, hyperparameter_selection=False)


def check_done(directory, lineage, artifacts):
    done = read_json(directory/'COMPLETE.json')
    if done['status'] != 'COMPLETE' or done['lineage'] != lineage:
        raise ValueError('Completed artifact lineage mismatch')
    for name in artifacts:
        if sha256(directory/(name+'.npz')) != done[name+'_sha256']:
            raise ValueError('Completed artifact digest mismatch: '+name)
    return done


def claim(directory, expected):
    directory.mkdir(parents=True, exist_ok=True)
    # Any interrupted fit requires investigation, not an automatic retrain.
    with (directory/'START.json').open('x', encoding='utf-8') as f:
        json.dump(dict(**expected, pid=os.getpid(), started_unix=time.time()), f, indent=2)


def fail(directory, expected, exc):
    write_json(directory/'FAILED.json', dict(**expected, status='FAILED',
                exception_type=type(exc).__name__, message=str(exc)))


def run_teacher(run, data, lineage, spec):
    directory = run/'teachers'/f"t_{spec['teacher_index']:02d}"
    if (directory/'COMPLETE.json').exists():
        return check_done(directory, lineage, ['model','indices'])
    expected = dict(lineage=lineage, teacher_index=spec['teacher_index'],
                    train_index_sha256=spec['train_index_sha256'])
    claim(directory, expected)
    start = time.monotonic()
    try:
        train = np.flatnonzero(np.isin(data['donor'],spec['train_donors']))
        if index_hash(train) != spec['train_index_sha256']:
            raise ValueError('Teacher row lineage mismatch')
        model = fit_model(data['X'][train],data['tissue_y'][train],data['donor'][train],
                          data['section'][train],n_classes=3,C=1.)
        np.savez_compressed(directory/'model.npz', **model.arrays())
        np.savez_compressed(directory/'indices.npz', train_index=train)
        result = dict(**expected,status='COMPLETE',fit_audit=model.fit_audit,
                       model_sha256=sha256(directory/'model.npz'),
                       indices_sha256=sha256(directory/'indices.npz'),elapsed_seconds=time.monotonic()-start)
        write_json(directory/'COMPLETE.json',result)
        print(json.dumps(dict(teacher=spec['teacher_index'],status='COMPLETE',seconds=result['elapsed_seconds'])),flush=True)
        return result
    except Exception as exc:
        fail(directory,expected,exc)
        raise


def teacher_features(run, data, plan, outer_index):
    n = len(data['X'])
    held = plan['outer'][outer_index]['heldout_donor']
    outer_donors = set(np.unique(data['donor'])) - {held}
    p_local,p_neighbor = np.full((n,3),np.nan),np.full((n,3),np.nan)
    teacher_for_row = np.full(n,-1,np.int64)
    lookup = {frozenset(t['train_donors']):t for t in plan['teachers']}
    for recipient in np.unique(data['donor']):
        train_donors = outer_donors - {recipient}
        spec = lookup[frozenset(train_donors)]
        if held in train_donors or recipient in train_donors:
            raise AssertionError('Teacher recipient or outer donor leaked')
        directory = run/'teachers'/f"t_{spec['teacher_index']:02d}"
        receipt = check_done(directory,plan['lineage'],['model','indices'])
        if receipt['train_index_sha256'] != spec['train_index_sha256']:
            raise ValueError('Teacher cache belongs to another training subset')
        model = MultinomialModel.from_arrays(arrays(directory/'model.npz'))
        idx = np.flatnonzero(data['donor']==recipient)
        p_local[idx] = model.predict_proba(data['X'][idx])
        p_neighbor[idx] = model.predict_proba(data['neighbor_X'][idx])
        teacher_for_row[idx] = spec['teacher_index']
    if not np.isfinite(p_local).all() or not np.isfinite(p_neighbor).all() or (teacher_for_row<0).any():
        raise AssertionError('Incomplete nested teacher probabilities')
    return p_local,p_neighbor,teacher_for_row


def metric(data, test, pred, mean, scale):
    return evaluate_programs(data['program_y'][test],pred,data['donor'][test],
        data['section'][test],data['tissue_y'][test],mean,scale,
        data['library_size'][test],data['n_detected_genes'][test])


def run_outer(run, data, plan, oi):
    lineage = plan['lineage']
    directory = run/'outer'/f'o_{oi:02d}'
    if (directory/'COMPLETE.json').exists():
        completed = check_done(directory,lineage,[])
        for arm in ARM_ORDER:
            check_done(directory/'arms'/arm,lineage,
                       ['predictions'] if arm=='MEAN' else ['model','predictions','target_map'])
        context = read_json(directory/'context_receipt.json')
        for key in ('pca','context'):
            if sha256(directory/(key+'.npz')) != context[key+'_sha256']:
                raise ValueError('Completed context modified')
        return completed
    expected = dict(lineage=lineage,outer_index=oi)
    claim(directory,expected)
    start=time.monotonic()
    try:
        held = plan['outer'][oi]['heldout_donor']
        train,test = np.flatnonzero(data['donor']!=held),np.flatnonzero(data['donor']==held)
        p_local,p_neighbor,teacher_for_row = teacher_features(run,data,plan,oi)
        pca = PCATransform.fit(data['X'][train],data['donor'][train],data['section'][train])
        local_pc,neighbor_pc = pca.transform(data['X']),pca.transform(data['neighbor_X'])
        np.savez_compressed(directory/'pca.npz', **pca.arrays())
        np.savez_compressed(directory/'context.npz',train_index=train,test_index=test,
                            local_pc=local_pc,neighbor_pc=neighbor_pc,p_local=p_local,
                            p_neighbor=p_neighbor,teacher_for_row=teacher_for_row)
        write_json(directory/'context_receipt.json',dict(**expected,
            pca_sha256=sha256(directory/'pca.npz'),context_sha256=sha256(directory/'context.npz'),
            pca_fit_audit=pca.fit_audit))
        target_scaler=Standardizer.fit(data['program_y'][train],data['donor'][train],data['section'][train])
        arm_results={}
        for arm in ARM_ORDER:
            cell=directory/'arms'/arm
            exp=dict(**expected,arm=arm)
            claim(cell,exp)
            cell_start=time.monotonic()
            if arm=='MEAN':
                pred=np.repeat(target_scaler.mean[None,:],len(test),axis=0)
                details=dict(no_fit=True,ymean=target_scaler.mean.tolist(),yscale=target_scaler.scale.tolist())
            else:
                design=build_design(local_pc,neighbor_pc,p_local,p_neighbor,arm)
                remap=np.arange(len(train),dtype=np.int64)
                remap_audit=dict(kind='identity',all_four_targets_share_same_map=True)
                meta=(data['donor'][train],data['section'][train],data['tissue_y'][train])
                if arm=='WRONG_DONOR':
                    remap,remap_audit=wrong_donor_target_map(*meta)
                elif arm=='SPATIAL_SHUFFLE':
                    remap,remap_audit=spatial_shuffle_target_map(*meta)
                fit=ProgramRegressor.fit(design[train],data['program_y'][train],
                        data['donor'][train],data['section'][train],target_index_map=remap)
                if not np.array_equal(fit.y_standardizer.mean,target_scaler.mean) or not np.array_equal(fit.y_standardizer.scale,target_scaler.scale):
                    raise AssertionError('Arm changed shared true training target scale')
                pred=fit.predict(design[test])
                np.savez_compressed(cell/'model.npz',**fit.arrays())
                np.savez_compressed(cell/'target_map.npz',index=remap)
                standardized=fit.x_standardizer.transform(design[train])
                # Report numerical rank, not allocated feature count as degrees of freedom.
                design_rank=int(np.linalg.matrix_rank(standardized))
                reuse=np.bincount(remap,minlength=len(train))
                values,counts=np.unique(reuse,return_counts=True)
                details=dict(fit_audit=fit.fit_audit,remap_audit=remap_audit,
                    standardized_design_rank=design_rank,
                    target_source_reuse_frequency_histogram={str(v):int(c) for v,c in zip(values,counts)},
                    model_sha256=sha256(cell/'model.npz'),target_map_sha256=sha256(cell/'target_map.npz'))
            np.savez_compressed(cell/'predictions.npz',row_index=test,prediction=pred)
            metrics=metric(data,test,pred,target_scaler.mean,target_scaler.scale)
            result=dict(**exp,status='COMPLETE',**details,metrics=metrics,
                predictions_sha256=sha256(cell/'predictions.npz'),elapsed_seconds=time.monotonic()-cell_start)
            write_json(cell/'COMPLETE.json',result)
            arm_results[arm]=dict(metrics=metrics,elapsed_seconds=result['elapsed_seconds'],
                                  **{k:details[k] for k in ('fit_audit','standardized_design_rank') if k in details})
            print(json.dumps(dict(outer=oi,arm=arm,status='COMPLETE')),flush=True)
        result=dict(**expected,status='COMPLETE',arms=arm_results,pca_fit_audit=pca.fit_audit,
                    elapsed_seconds=time.monotonic()-start)
        write_json(directory/'COMPLETE.json',result)
        return result
    except Exception as exc:
        fail(directory,expected,exc)
        raise


def paired_bootstrap(candidate,control,metric_name):
    if any(v is None for v in candidate+control):
        return dict(status='UNDEFINED_MISSING_PROGRAM_OR_DONOR',n_donors=len(candidate))
    candidate,control=np.asarray(candidate),np.asarray(control)
    if candidate.shape!=(6,) or control.shape!=(6,):
        raise ValueError('Exactly six fixed OOF donor estimates required')
    delta=control-candidate if metric_name in ('huber','mse') else candidate-control
    indices=np.random.default_rng(7032005).integers(0,6,size=(20000,6))
    means=delta[indices].mean(1)
    return dict(status='COMPLETE',mean_improvement=float(delta.mean()),
                ci95=np.quantile(means,[.025,.975]).tolist(),positive_donors=int((delta>0).sum()),
                per_fold_improvement=delta.tolist(),bootstrap_repeats=20000,
                bootstrap_unit='donor',scope='descriptive_fixed_OOF_not_full_pipeline_refit')


def finalize(run,data,plan):
    lineage=plan['lineage']
    if any(not (run/'outer'/f'o_{i:02d}'/'COMPLETE.json').exists() for i in range(6)):
        raise ValueError('Cannot finalize incomplete outer folds')
    teachers=[check_done(run/'teachers'/f't_{i:02d}',lineage,['model','indices']) for i in range(21)]
    folds=[]
    for oi in range(6):
        directory=run/'outer'/f'o_{oi:02d}'
        folds.append(check_done(directory,lineage,[]))
        context=read_json(directory/'context_receipt.json')
        if context['lineage'] != lineage:
            raise ValueError('Context lineage differs during finalization')
        for name in ('pca','context'):
            if sha256(directory/(name+'.npz')) != context[name+'_sha256']:
                raise ValueError('Context changed during finalization')
        for arm in ARM_ORDER:
            check_done(directory/'arms'/arm,lineage,
                       ['predictions'] if arm=='MEAN' else ['model','predictions','target_map'])
    predictions=np.full((len(ARM_ORDER),len(data['X']),4),np.nan)
    outer_fold=np.full(len(data['X']),-1,np.int64)
    for oi in range(6):
        directory=run/'outer'/f'o_{oi:02d}'
        for ai,arm in enumerate(ARM_ORDER):
            result=arrays(directory/'arms'/arm/'predictions.npz')
            rows=result['row_index']
            if not np.array_equal(rows,np.flatnonzero(data['donor']==plan['outer'][oi]['heldout_donor'])):
                raise ValueError('OOF rows differ from held out donor')
            if np.isfinite(predictions[ai,rows]).any():
                raise ValueError('Duplicate OOF predictions')
            predictions[ai,rows]=result['prediction']
            outer_fold[rows]=oi
    if not np.isfinite(predictions).all() or (outer_fold<0).any():
        raise ValueError('Incomplete OOF coverage')
    np.savez_compressed(run/'OOF_PRIVATE.npz',predictions=predictions,row_index=np.arange(len(outer_fold)),outer_fold=outer_fold)
    aggregate={}
    for arm in ARM_ORDER:
        metrics=[f['arms'][arm]['metrics'] for f in folds]
        aggregate[arm]={m:float(np.mean([v[m] for v in metrics])) if all(v[m] is not None for v in metrics) else None for m in METRICS}
        aggregate[arm]['per_program']=[dict(program_index=k,**{
            m:float(np.mean([v['per_program'][k][m] for v in metrics]))
            if all(v['per_program'][k][m] is not None for v in metrics) else None for m in METRICS}) for k in range(4)]
    contrasts={}
    for control in ('COMPOSITION','ADDITIVE','RANDOM_GATE','WRONG_DONOR','SPATIAL_SHUFFLE'):
        contrasts[control]={m:paired_bootstrap([f['arms']['TISSUE_COND']['metrics'][m] for f in folds],
            [f['arms'][control]['metrics'][m] for f in folds],m) for m in METRICS}
    result=dict(schema_version='sami_stage_b_results_v1',status='COMPLETE_PENDING_INDEPENDENT_AUDIT',
        lineage=lineage,n_spots=13011,n_donors=6,n_sections=12,program_order=read_json(ROOT/'configs/sami_surv_stage_b_v1.json')['program_order'],
        optimized_teacher_models=21,optimized_program_models=36,outer_PCA_transforms=6,
        arm_order=list(ARM_ORDER),aggregate=aggregate,paired_contrasts=contrasts,
        per_fold=[dict(outer_index=i,arms=f['arms'],pca_fit_audit=f['pca_fit_audit'],elapsed_seconds=f['elapsed_seconds']) for i,f in enumerate(folds)],
        teacher_fit_audits=[dict(teacher_index=i,**t['fit_audit']) for i,t in enumerate(teachers)],
        original_ST_test_read=False,TCGA_DSS_read=False,source_truth_claim=False,
        OOF_private_sha256=sha256(run/'OOF_PRIVATE.npz'))
    write_json(run/'RESULTS.json',result)
    print(json.dumps(dict(status=result['status'],optimized_models=57)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--mode',required=True,choices=['plan','teachers','outer','finalize'])
    parser.add_argument('--teacher-index',type=int)
    parser.add_argument('--outer-fold',type=int)
    args=parser.parse_args()
    data,config,lineage=load_inputs(args.prepared)
    args.run.mkdir(parents=True,exist_ok=True)
    expected=make_plan(data,lineage)
    path=args.run/'PLAN.json'
    if args.mode=='plan':
        if path.exists() and read_json(path)!=expected:
            raise ValueError('Existing plan differs; no silent resume')
        if not path.exists():
            write_json(path,expected)
        print(json.dumps(dict(status='PLAN_LOCKED',teachers=21,program_regressors=36)),flush=True)
        return
    plan=read_json(path)
    if plan!=expected:
        raise ValueError('Plan or runtime/source lineage mismatch')
    if args.mode=='teachers':
        chosen=plan['teachers'] if args.teacher_index is None else [plan['teachers'][args.teacher_index]]
        for spec in chosen:
            run_teacher(args.run,data,lineage,spec)
    elif args.mode=='outer':
        if args.outer_fold is None or not 0<=args.outer_fold<6:
            raise ValueError('Explicit outer fold 0..5 required')
        run_outer(args.run,data,plan,args.outer_fold)
    else:
        # Finalization is not permitted to launch a missing training unit.
        if any(not (args.run/'outer'/f'o_{i:02d}'/'COMPLETE.json').exists() for i in range(6)):
            raise ValueError('Cannot finalize incomplete outer folds')
        finalize(args.run,data,plan)


if __name__=='__main__':
    main()
